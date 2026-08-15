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

//! Micro benchmarks for the scheduler hot paths. These are `#[ignore]`d tests rather
//! than criterion benchmarks because they need `crate::test_utils`, which is only
//! compiled for tests.
//!
//! Run with:
//!
//! ```text
//! cargo test -p ballista-scheduler --release --lib -- --ignored --nocapture bench_
//! ```

use std::time::Instant;

use crate::state::execution_graph::ExecutionGraph;
use crate::test_utils::{
    mock_completed_task, mock_executor, revive_graph_and_complete_next_stage,
    test_aggregation_plan,
};

/// Number of plan input partitions to benchmark a stage with
const PARTITIONS: usize = 4_000;

fn report(name: &str, units: usize, elapsed: std::time::Duration) {
    println!(
        "{name:<40} {units:>7} x in {elapsed:>10.3?} ({:>8.1?} each)",
        elapsed / units as u32
    );
}

/// Build an aggregation graph and drive it to the point where the wide final stage
/// (`PARTITIONS` partitions) is running, which is the stage the scheduler works on.
async fn wide_stage_graph() -> impl ExecutionGraph {
    let mut graph = test_aggregation_plan(PARTITIONS).await;
    // The leaf stage only has as many tasks as the scan has partitions; complete it so
    // the wide final stage resolves.
    revive_graph_and_complete_next_stage(&mut graph).unwrap();
    graph.revive();
    assert_eq!(graph.available_tasks(), PARTITIONS);
    graph
}

/// Task status updates, the path every finished task goes through.
#[tokio::test]
#[ignore]
async fn bench_update_task_status() {
    for batch_size in [1usize, 64] {
        let executor = mock_executor("executor-1".to_string());
        let mut graph = wide_stage_graph().await;

        let mut statuses = vec![];
        while let Some(task) = graph.pop_next_task(&executor.id).unwrap() {
            statuses.push(mock_completed_task(task, &executor.id));
        }
        let total = statuses.len();

        let start = Instant::now();
        for batch in statuses.chunks(batch_size) {
            graph
                .update_task_status(&executor, batch.to_vec(), 4, 4)
                .unwrap();
        }
        let elapsed = start.elapsed();

        report(
            &format!("update_task_status/batch={batch_size}"),
            total,
            elapsed,
        );
    }
}

/// Pull based dispatch: hand out every task of the stage.
#[tokio::test]
#[ignore]
async fn bench_pop_next_task() {
    let executor = mock_executor("executor-1".to_string());
    let mut graph = wide_stage_graph().await;

    let start = Instant::now();
    let mut popped = 0;
    while graph.pop_next_task(&executor.id).unwrap().is_some() {
        popped += 1;
    }
    let elapsed = start.elapsed();

    assert!(popped > 0);
    report("pop_next_task", popped, elapsed);
}

/// Serialization done for every task that is launched: the session configuration is
/// turned into key/value pairs and shipped with each `MultiTaskDefinition`.
#[tokio::test]
#[ignore]
async fn bench_session_config_to_key_value_pairs() {
    use ballista_core::extension::{SessionConfigExt, SessionConfigHelperExt};
    use datafusion::prelude::SessionConfig;

    let config = SessionConfig::new_with_ballista();
    println!(
        "session config has {} key value pairs",
        config.to_key_value_pairs().len()
    );

    let iterations = 10_000;

    let start = Instant::now();
    let mut total = 0;
    for _ in 0..iterations {
        total += config.to_key_value_pairs().len();
    }
    report("to_key_value_pairs", iterations, start.elapsed());

    // What it costs once the result is cached and only has to be cloned per task
    let cached = config.to_key_value_pairs();
    let start = Instant::now();
    for _ in 0..iterations {
        total += cached.clone().len();
    }
    report("clone cached key value pairs", iterations, start.elapsed());

    assert!(total > 0);
}
