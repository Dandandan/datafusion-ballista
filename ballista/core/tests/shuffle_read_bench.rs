// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Local shuffle-read micro benchmark (run with `--ignored --nocapture`).
//!
//! Writes N node-local shuffle files in the hash-shuffle on-disk format and
//! measures how long `ShuffleReaderExec::execute(0)` takes to drain all of
//! them, which is the reduce-side path used whenever the producing task ran
//! on the same executor.

use std::sync::Arc;
use std::time::Instant;

use ballista_core::execution_plans::{ShuffleReaderExec, create_shuffle_path};
use ballista_core::serde::scheduler::{
    ExecutorMetadata, ExecutorOperatingSystemSpecification, ExecutorSpecification,
    PartitionId, PartitionLocation, PartitionStats,
};
use ballista_core::utils::create_write_options;
use datafusion::arrow::array::{Float64Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::ipc::CompressionType;
use datafusion::arrow::ipc::writer::StreamWriter;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
use datafusion::prelude::SessionContext;
use futures::StreamExt;

const NUM_INT_COLS: usize = 8;
const NUM_FLOAT_COLS: usize = 6;
const NUM_STR_COLS: usize = 2;
const ROWS_PER_BATCH: usize = 8192;

fn build_schema() -> SchemaRef {
    let mut fields = Vec::new();
    for i in 0..NUM_INT_COLS {
        fields.push(Field::new(format!("i{i}"), DataType::Int64, false));
    }
    for i in 0..NUM_FLOAT_COLS {
        fields.push(Field::new(format!("f{i}"), DataType::Float64, false));
    }
    for i in 0..NUM_STR_COLS {
        fields.push(Field::new(format!("s{i}"), DataType::Utf8, false));
    }
    Arc::new(Schema::new(fields))
}

fn build_batch(schema: &SchemaRef, seed: usize) -> RecordBatch {
    let mut columns: Vec<Arc<dyn datafusion::arrow::array::Array>> = Vec::new();
    for c in 0..NUM_INT_COLS {
        let v: Vec<i64> = (0..ROWS_PER_BATCH)
            .map(|r| ((r * 31 + c * 7 + seed) % 100_000) as i64)
            .collect();
        columns.push(Arc::new(Int64Array::from(v)));
    }
    for c in 0..NUM_FLOAT_COLS {
        let v: Vec<f64> = (0..ROWS_PER_BATCH)
            .map(|r| (r as f64) * 1.5 + (c + seed) as f64)
            .collect();
        columns.push(Arc::new(Float64Array::from(v)));
    }
    for c in 0..NUM_STR_COLS {
        let v: Vec<String> = (0..ROWS_PER_BATCH)
            .map(|r| format!("value-{}-{}-{}", seed, c, r % 977))
            .collect();
        columns.push(Arc::new(StringArray::from(v)));
    }
    RecordBatch::try_new(schema.clone(), columns).unwrap()
}

fn executor_meta() -> ExecutorMetadata {
    ExecutorMetadata {
        id: "executor_1".to_string(),
        host: "executor_1".to_string(),
        port: 7070,
        grpc_port: 8080,
        specification: ExecutorSpecification::default().with_vcores(1),
        os_info: ExecutorOperatingSystemSpecification::default(),
    }
}

/// Write `num_files` shuffle files, each with `batches_per_file` batches, and
/// return the matching `PartitionLocation`s plus total uncompressed bytes.
fn write_shuffle_files(
    work_dir: &std::path::Path,
    schema: &SchemaRef,
    num_files: usize,
    batches_per_file: usize,
    compression: Option<CompressionType>,
) -> (Vec<PartitionLocation>, usize, usize) {
    let mut locations = Vec::new();
    let mut in_memory_bytes = 0usize;
    let mut on_disk_bytes = 0usize;
    for f in 0..num_files {
        let path =
            create_shuffle_path(work_dir, &"job".into(), 1, 0, Some(f as u64), false)
                .unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let file = std::fs::File::create(&path).unwrap();
        let file = std::io::BufWriter::new(file);
        let options = create_write_options(compression).unwrap();
        let mut writer =
            StreamWriter::try_new_with_options(file, schema.as_ref(), options).unwrap();
        let mut rows = 0u64;
        for b in 0..batches_per_file {
            let batch = build_batch(schema, f * 1000 + b);
            in_memory_bytes += batch.get_array_memory_size();
            rows += batch.num_rows() as u64;
            writer.write(&batch).unwrap();
        }
        writer.finish().unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        on_disk_bytes += len as usize;
        locations.push(PartitionLocation {
            map_partition_id: f,
            partition_id: PartitionId {
                job_id: "job".into(),
                stage_id: 1,
                partition_id: 0,
            },
            executor_meta: executor_meta(),
            partition_stats: PartitionStats::new(
                Some(rows),
                Some(batches_per_file as u64),
                Some(len),
            ),
            file_id: Some(f as u64),
            is_sort_shuffle: false,
        });
    }
    (locations, in_memory_bytes, on_disk_bytes)
}

#[test]
#[ignore]
fn bench_local_shuffle_read() {
    let num_files: usize = std::env::var("BENCH_FILES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let batches_per_file: usize = std::env::var("BENCH_BATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let iterations: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let worker_threads: usize = std::env::var("BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let schema = build_schema();
    let tmp = tempfile::tempdir().unwrap();
    let work_dir = tmp.path().to_path_buf();

    let (locations, in_memory_bytes, on_disk_bytes) = write_shuffle_files(
        &work_dir,
        &schema,
        num_files,
        batches_per_file,
        Some(CompressionType::LZ4_FRAME),
    );

    println!(
        "files={num_files} batches/file={batches_per_file} \
         in_memory={:.1} MiB on_disk={:.1} MiB threads={worker_threads}",
        in_memory_bytes as f64 / 1024.0 / 1024.0,
        on_disk_bytes as f64 / 1024.0 / 1024.0,
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()
        .unwrap();

    let readers = std::env::var("BENCH_LOCAL_READERS").unwrap_or_else(|_| "4".into());
    println!("max_local_readers={readers}");
    let cfg = <datafusion::prelude::SessionConfig as ballista_core::extension::SessionConfigExt>::new_with_ballista()
        .set_str("ballista.shuffle.reader.max_local_readers", &readers);
    let ctx = SessionContext::new_with_config(cfg);
    let task_ctx: Arc<TaskContext> = ctx.task_ctx();

    let mut times = Vec::new();
    for iter in 0..iterations {
        let reader = ShuffleReaderExec::try_new(
            1,
            vec![locations.clone()],
            schema.clone(),
            Partitioning::UnknownPartitioning(1),
        )
        .unwrap()
        .with_work_dir(work_dir.to_string_lossy().to_string());

        let task_ctx = task_ctx.clone();
        let start = Instant::now();
        let (rows, batches) = rt.block_on(async move {
            let mut stream = reader.execute(0, task_ctx).unwrap();
            let mut rows = 0usize;
            let mut batches = 0usize;
            while let Some(b) = stream.next().await {
                let b = b.unwrap();
                rows += b.num_rows();
                batches += 1;
            }
            (rows, batches)
        });
        let elapsed = start.elapsed();
        println!(
            "iter {iter}: {:?} rows={rows} batches={batches} \
             throughput={:.1} MiB/s (decoded)",
            elapsed,
            in_memory_bytes as f64 / 1024.0 / 1024.0 / elapsed.as_secs_f64()
        );
        times.push(elapsed);
    }

    times.sort();
    println!("median: {:?}  min: {:?}", times[times.len() / 2], times[0]);
}

/// Same shape as `bench_local_shuffle_read`, but over sort-shuffle output —
/// the format hash-partitioned stages actually produce. One reduce partition
/// reads its byte range out of every map file.
#[test]
#[ignore]
fn bench_local_sort_shuffle_read() {
    use ballista_core::execution_plans::SortShuffleWriterExec;
    use ballista_core::execution_plans::sort_shuffle::SortShuffleConfig;
    use datafusion::arrow::array::{StringArray as StrArr, UInt32Array, UInt64Array};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::physical_plan::expressions::Column;

    let map_tasks: usize = std::env::var("BENCH_MAPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let batches_per_map: usize = std::env::var("BENCH_BATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let num_partitions: usize = std::env::var("BENCH_PARTITIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let iterations: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let readers = std::env::var("BENCH_LOCAL_READERS").unwrap_or_else(|_| "4".into());

    let schema = build_schema();
    let tmp = tempfile::tempdir().unwrap();
    let work_dir = tmp.path().to_string_lossy().to_string();

    let partitions: Vec<Vec<RecordBatch>> = (0..map_tasks)
        .map(|m| {
            (0..batches_per_map)
                .map(|b| build_batch(&schema, m * 100 + b))
                .collect()
        })
        .collect();
    let source =
        Arc::new(MemorySourceConfig::try_new(&partitions, schema.clone(), None).unwrap());
    let input: Arc<dyn ExecutionPlan> = Arc::new(DataSourceExec::new(source));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let cfg = <datafusion::prelude::SessionConfig as ballista_core::extension::SessionConfigExt>::new_with_ballista()
        .set_str("ballista.shuffle.reader.max_local_readers", &readers);
    let ctx = SessionContext::new_with_config(cfg);

    // Write the shuffle, then recover each map file's id from the writer's
    // metadata output so the reader can address it.
    let writer = Arc::new(
        SortShuffleWriterExec::try_new(
            "job".into(),
            1,
            input,
            work_dir.clone(),
            Partitioning::Hash(
                vec![Arc::new(Column::new(schema.field(0).name(), 0))],
                num_partitions,
            ),
            SortShuffleConfig::new(true, 8192),
        )
        .unwrap(),
    );
    let meta = {
        let w = writer.clone();
        let task_ctx = ctx.task_ctx();
        rt.block_on(async move {
            let mut handles = Vec::new();
            for p in 0..num_partitions {
                let w = w.clone();
                let ctx = task_ctx.clone();
                handles.push(tokio::spawn(async move {
                    let mut s = w.execute(p, ctx).unwrap();
                    let mut out = Vec::new();
                    while let Some(b) = s.next().await {
                        out.push(b.unwrap());
                    }
                    out
                }));
            }
            let mut all = Vec::new();
            for h in handles {
                all.extend(h.await.unwrap());
            }
            all
        })
    };

    // partition -> [(file_id, num_bytes)]
    let mut per_partition: Vec<Vec<(u64, u64)>> =
        (0..num_partitions).map(|_| Vec::new()).collect();
    for b in &meta {
        let part = b.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
        let file_id = b.column(2).as_any().downcast_ref::<UInt64Array>().unwrap();
        let stats = b
            .column(3)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StructArray>()
            .unwrap();
        let num_bytes = stats
            .column_by_name("num_bytes")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let _ = std::any::type_name::<StrArr>();
        for r in 0..b.num_rows() {
            per_partition[part.value(r) as usize]
                .push((file_id.value(r), num_bytes.value(r)));
        }
    }

    let target = per_partition
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| v.iter().map(|(_, b)| *b).sum::<u64>())
        .map(|(i, _)| i)
        .unwrap();
    let locations: Vec<PartitionLocation> = per_partition[target]
        .iter()
        .map(|&(file_id, num_bytes)| PartitionLocation {
            map_partition_id: file_id as usize,
            partition_id: PartitionId {
                job_id: "job".into(),
                stage_id: 1,
                partition_id: target,
            },
            executor_meta: executor_meta(),
            partition_stats: PartitionStats::new(None, None, Some(num_bytes)),
            file_id: Some(file_id),
            is_sort_shuffle: true,
        })
        .collect();

    println!(
        "map_tasks={map_tasks} partitions={num_partitions} \
         reduce_partition={target} blocks={} max_local_readers={readers}",
        locations.len()
    );

    let task_ctx: Arc<TaskContext> = ctx.task_ctx();
    let mut times = Vec::new();
    for iter in 0..iterations {
        let reader = ShuffleReaderExec::try_new(
            1,
            vec![locations.clone()],
            schema.clone(),
            Partitioning::UnknownPartitioning(1),
        )
        .unwrap()
        .with_work_dir(work_dir.clone());
        let task_ctx = task_ctx.clone();
        let start = Instant::now();
        let (rows, decoded) = rt.block_on(async move {
            let mut stream = reader.execute(0, task_ctx).unwrap();
            let mut rows = 0usize;
            let mut decoded = 0usize;
            while let Some(b) = stream.next().await {
                let b = b.unwrap();
                rows += b.num_rows();
                decoded += b.get_array_memory_size();
            }
            (rows, decoded)
        });
        let elapsed = start.elapsed();
        println!(
            "iter {iter}: {elapsed:?} rows={rows} decoded={:.1} MiB",
            decoded as f64 / 1024.0 / 1024.0
        );
        times.push(elapsed);
    }
    times.sort();
    println!("median: {:?}  min: {:?}", times[times.len() / 2], times[0]);
}
