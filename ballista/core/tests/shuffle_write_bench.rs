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

//! Measurements for the shuffle write path and the block-transport decoder.
//! Run with `--ignored --nocapture`.

use std::sync::Arc;
use std::time::Instant;

use ballista_core::client::BlockDataStream;
use datafusion::arrow::array::{Float64Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::ipc::CompressionType;
use datafusion::arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use datafusion::arrow::record_batch::RecordBatch;
use futures::StreamExt;

fn wide_schema(num_cols: usize) -> SchemaRef {
    let fields: Vec<Field> = (0..num_cols)
        .map(|i| match i % 3 {
            0 => Field::new(format!("column_name_{i}"), DataType::Int64, true),
            1 => Field::new(format!("column_name_{i}"), DataType::Float64, true),
            _ => Field::new(format!("column_name_{i}"), DataType::Utf8, true),
        })
        .collect();
    Arc::new(Schema::new(fields))
}

fn schema_message_bytes(
    schema: &SchemaRef,
    compression: Option<CompressionType>,
) -> usize {
    let mut buf: Vec<u8> = Vec::new();
    let options = IpcWriteOptions::default()
        .try_with_compression(compression)
        .unwrap();
    let mut w =
        StreamWriter::try_new_with_options(&mut buf, schema.as_ref(), options).unwrap();
    w.finish().unwrap();
    buf.len()
}

/// How many bytes of a sort-shuffle data file are pure schema metadata:
/// one leading header stream plus one schema message per non-empty output
/// partition.
#[test]
#[ignore]
fn measure_schema_overhead() {
    for num_cols in [10usize, 50, 100, 300] {
        let schema = wide_schema(num_cols);
        let bytes = schema_message_bytes(&schema, Some(CompressionType::LZ4_FRAME));
        for k in [64usize, 200, 1000] {
            println!(
                "cols={num_cols:>3} partitions={k:>4}  schema stream = {bytes:>6} B  \
                 per-file schema overhead = {:>8.2} MiB ({} copies)",
                (bytes * (k + 1)) as f64 / 1024.0 / 1024.0,
                k + 1
            );
        }
    }
}

fn build_batch(schema: &SchemaRef, rows: usize, seed: usize) -> RecordBatch {
    let columns: Vec<Arc<dyn datafusion::arrow::array::Array>> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| -> Arc<dyn datafusion::arrow::array::Array> {
            match f.data_type() {
                DataType::Int64 => Arc::new(Int64Array::from(
                    (0..rows)
                        .map(|r| (r * 7 + i + seed) as i64)
                        .collect::<Vec<_>>(),
                )),
                DataType::Float64 => Arc::new(Float64Array::from(
                    (0..rows)
                        .map(|r| r as f64 * 0.5 + i as f64)
                        .collect::<Vec<_>>(),
                )),
                _ => Arc::new(StringArray::from(
                    (0..rows)
                        .map(|r| format!("v-{seed}-{i}-{}", r % 1013))
                        .collect::<Vec<_>>(),
                )),
            }
        })
        .collect();
    RecordBatch::try_new(schema.clone(), columns).unwrap()
}

/// End-to-end sort-shuffle write, reporting the writer's own metric
/// breakdown (repart / spill / write time).
#[test]
#[ignore]
fn bench_sort_shuffle_write() {
    use ballista_core::execution_plans::SortShuffleWriterExec;
    use ballista_core::execution_plans::sort_shuffle::SortShuffleConfig;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::physical_plan::Partitioning;
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion::physical_plan::expressions::Column;
    use datafusion::physical_plan::metrics::MetricValue;
    use datafusion::prelude::SessionContext;

    let num_cols: usize = env_usize("BENCH_COLS", 100);
    let num_batches: usize = env_usize("BENCH_BATCHES", 40);
    let num_partitions: usize = env_usize("BENCH_PARTITIONS", 200);
    let iterations: usize = env_usize("BENCH_ITERS", 5);

    let schema = wide_schema(num_cols);
    let batches: Vec<RecordBatch> = (0..num_batches)
        .map(|b| build_batch(&schema, 8192, b))
        .collect();
    let decoded: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
    let source =
        Arc::new(MemorySourceConfig::try_new(&[batches], schema.clone(), None).unwrap());
    let input: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(Arc::new(
        DataSourceExec::new(source),
    )));

    println!(
        "cols={num_cols} batches={num_batches} partitions={num_partitions} \
         in_memory={:.1} MiB",
        decoded as f64 / 1024.0 / 1024.0
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let tmp = tempfile::tempdir().unwrap();

    let mut times = Vec::new();
    for iter in 0..iterations {
        let ctx = SessionContext::new();
        let task_ctx = ctx.task_ctx();
        let writer = Arc::new(
            SortShuffleWriterExec::try_new(
                "bench_job".into(),
                1,
                input.clone(),
                tmp.path().to_string_lossy().to_string(),
                Partitioning::Hash(
                    vec![Arc::new(Column::new(schema.field(0).name(), 0))],
                    num_partitions,
                ),
                SortShuffleConfig::new(true, 8192),
            )
            .unwrap(),
        );
        let start = Instant::now();
        let w = writer.clone();
        rt.block_on(async move {
            let mut handles = Vec::new();
            for p in 0..num_partitions {
                let w = w.clone();
                let ctx = task_ctx.clone();
                handles.push(tokio::spawn(async move {
                    let mut s = w.execute(p, ctx).unwrap();
                    while let Some(b) = s.next().await {
                        b.unwrap();
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
        let elapsed = start.elapsed();

        let mut repart_ns = 0usize;
        let mut write_ns = 0usize;
        if let Some(m) = writer.metrics() {
            for m in m.iter() {
                match m.value() {
                    MetricValue::Time { name, time } if name == "repart_time" => {
                        repart_ns += time.value()
                    }
                    MetricValue::Time { name, time } if name == "write_time" => {
                        write_ns += time.value()
                    }
                    _ => {}
                }
            }
        }
        println!(
            "iter {iter}: total={elapsed:?} repart={:?} write={:?} \
             throughput={:.1} MiB/s",
            std::time::Duration::from_nanos(repart_ns as u64),
            std::time::Duration::from_nanos(write_ns as u64),
            decoded as f64 / 1024.0 / 1024.0 / elapsed.as_secs_f64(),
        );
        times.push(elapsed);
    }
    times.sort();
    println!("median: {:?} min: {:?}", times[times.len() / 2], times[0]);
}

/// How much shuffle data a map-side partial aggregate actually saves, and what
/// DataFusion's adaptive `skip_partial_aggregation` heuristic costs when the
/// consumer is a network shuffle rather than a local operator.
#[test]
#[ignore]
fn bench_partial_aggregate_shuffle_size() {
    use ballista_core::execution_plans::SortShuffleWriterExec;
    use ballista_core::execution_plans::sort_shuffle::SortShuffleConfig;
    use datafusion::datasource::MemTable;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::physical_plan::Partitioning;
    use datafusion::physical_plan::expressions::Column;
    use datafusion::prelude::{SessionConfig, SessionContext};

    let num_rows: usize = env_usize("BENCH_ROWS", 4_000_000);
    let num_partitions: usize = env_usize("BENCH_PARTITIONS", 16);

    // group cardinality as a fraction of row count
    let ratios: Vec<f64> = vec![0.01, 0.05, 0.1, 0.25, 0.5];
    // `hashed` scatters keys uniformly; `cyclic` walks the key space in order
    // (`row % cardinality`), which is what clustered / time-ordered data looks
    // like and is the case DataFusion's probe window mis-reads.
    let layouts = ["hashed", "cyclic"];

    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
        Field::new("pad", DataType::Utf8, false),
    ]));

    println!(
        "rows={num_rows} shuffle_partitions={num_partitions}\n\
         {:<8} {:<8} {:<14} {:>14} {:>14} {:>9}",
        "layout", "ratio", "skip_partial", "shuffle bytes", "rows shuffled", "time"
    );

    for layout in layouts {
        for &ratio in &ratios {
            let cardinality = ((num_rows as f64) * ratio).max(1.0) as i64;
            let rows_per_batch = 8192;
            let mut batches = Vec::new();
            let mut produced = 0usize;
            while produced < num_rows {
                let n = rows_per_batch.min(num_rows - produced);
                let keys: Vec<i64> = (0..n)
                    .map(|r| {
                        let i = (produced + r) as i64;
                        if layout == "hashed" {
                            (i * 2_654_435_761) % cardinality
                        } else {
                            i % cardinality
                        }
                    })
                    .collect();
                let vals: Vec<i64> = (0..n).map(|r| (produced + r) as i64).collect();
                let pads: Vec<String> = (0..n)
                    .map(|r| format!("padding-value-{}", (produced + r) % 97))
                    .collect();
                batches.push(
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int64Array::from(keys)),
                            Arc::new(Int64Array::from(vals)),
                            Arc::new(StringArray::from(pads)),
                        ],
                    )
                    .unwrap(),
                );
                produced += n;
            }

            for skip_enabled in [true, false] {
                let mut cfg = SessionConfig::new().with_target_partitions(4);
                if !skip_enabled {
                    // Never give up on map-side reduction: downstream is a shuffle.
                    cfg.options_mut()
                        .execution
                        .skip_partial_aggregation_probe_ratio_threshold = 1.0;
                }
                let ctx = SessionContext::new_with_config(cfg);
                let table =
                    MemTable::try_new(schema.clone(), vec![batches.clone()]).unwrap();
                ctx.register_table("t", Arc::new(table)).unwrap();

                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(4)
                    .enable_all()
                    .build()
                    .unwrap();

                let plan = rt.block_on(async {
                    ctx.sql("SELECT k, sum(v) FROM t GROUP BY k")
                        .await
                        .unwrap()
                        .create_physical_plan()
                        .await
                        .unwrap()
                });

                let partial =
                    find_partial_aggregate(&plan).expect("no partial aggregate");

                let tmp = tempfile::tempdir().unwrap();
                let writer = Arc::new(
                    SortShuffleWriterExec::try_new(
                        "agg_job".into(),
                        1,
                        partial.clone(),
                        tmp.path().to_string_lossy().to_string(),
                        Partitioning::Hash(
                            vec![Arc::new(Column::new("k", 0))],
                            num_partitions,
                        ),
                        SortShuffleConfig::new(true, 8192),
                    )
                    .unwrap(),
                );

                let task_ctx = ctx.task_ctx();
                let start = Instant::now();
                let w = writer.clone();
                rt.block_on(async move {
                    let mut handles = Vec::new();
                    for p in 0..num_partitions {
                        let w = w.clone();
                        let ctx = task_ctx.clone();
                        handles.push(tokio::spawn(async move {
                            let mut s = w.execute(p, ctx).unwrap();
                            while let Some(b) = s.next().await {
                                b.unwrap();
                            }
                        }));
                    }
                    for h in handles {
                        h.await.unwrap();
                    }
                });
                let elapsed = start.elapsed();

                let shuffle_bytes = dir_size(tmp.path());
                let mut rows_out = 0u64;
                if let Some(ms) = partial.metrics() {
                    rows_out = ms.output_rows().unwrap_or(0) as u64;
                }
                println!(
                    "{layout:<8} {ratio:<8} {:<14} {:>10.1} MiB {:>14} {:>8.2}s",
                    skip_enabled,
                    shuffle_bytes as f64 / 1024.0 / 1024.0,
                    rows_out,
                    elapsed.as_secs_f64()
                );
            }
        }
    }
}

fn find_partial_aggregate(
    plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>,
) -> Option<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
    use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
    if let Some(agg) = plan.downcast_ref::<AggregateExec>()
        && *agg.mode() == AggregateMode::Partial
    {
        return Some(plan.clone());
    }
    for child in plan.children() {
        if let Some(found) = find_partial_aggregate(child) {
            return Some(found);
        }
    }
    None
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for e in entries.flatten() {
            let meta = match e.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.is_dir() {
                total += dir_size(&e.path());
            } else if e.file_name() != "data.arrow.index" {
                total += meta.len();
            }
        }
    }
    total
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Throughput of `BlockDataStream` (the default remote shuffle transport
/// decoder) over an in-memory IPC stream delivered in fixed-size chunks.
#[test]
#[ignore]
fn bench_block_data_stream() {
    let chunk_size: usize = std::env::var("BENCH_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64 * 1024);
    let num_batches: usize = std::env::var("BENCH_BATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let iterations: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let schema = wide_schema(16);
    let mut buf: Vec<u8> = Vec::new();
    let options = IpcWriteOptions::default()
        .try_with_compression(Some(CompressionType::LZ4_FRAME))
        .unwrap();
    let mut w =
        StreamWriter::try_new_with_options(&mut buf, schema.as_ref(), options).unwrap();
    let mut decoded_bytes = 0usize;
    for b in 0..num_batches {
        let batch = build_batch(&schema, 8192, b);
        decoded_bytes += batch.get_array_memory_size();
        w.write(&batch).unwrap();
    }
    w.finish().unwrap();
    let wire_bytes = buf.len();
    let bytes = prost::bytes::Bytes::from(buf);

    println!(
        "wire={:.1} MiB decoded={:.1} MiB chunk={} KiB batches={num_batches}",
        wire_bytes as f64 / 1024.0 / 1024.0,
        decoded_bytes as f64 / 1024.0 / 1024.0,
        chunk_size / 1024,
    );

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut times = Vec::new();
    for iter in 0..iterations {
        let chunks: Vec<_> = (0..bytes.len())
            .step_by(chunk_size)
            .map(|off| {
                let end = (off + chunk_size).min(bytes.len());
                Ok(bytes.slice(off..end))
            })
            .collect();
        let start = Instant::now();
        let rows = rt.block_on(async move {
            let stream = futures::stream::iter(chunks);
            let mut decoder = BlockDataStream::try_new(Box::pin(stream)).await.unwrap();
            let mut rows = 0usize;
            while let Some(b) = decoder.next().await {
                rows += b.unwrap().num_rows();
            }
            rows
        });
        let elapsed = start.elapsed();
        println!(
            "iter {iter}: {elapsed:?} rows={rows} decode={:.1} MiB/s (wire)",
            wire_bytes as f64 / 1024.0 / 1024.0 / elapsed.as_secs_f64()
        );
        times.push(elapsed);
    }
    times.sort();
    println!("median: {:?} min: {:?}", times[times.len() / 2], times[0]);
}
