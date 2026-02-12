use anyhow::Result;

use crate::problem::Problem;
use crate::solution::Solution;

// TODO: implement real scheduler later.
pub fn scheduler_solution(_problem: &Problem) -> Result<Solution> {

  let n_ops = _problem.op_types.len();

  let mut subgraphs = Vec::with_capacity(n_ops);
  let mut granularities = Vec::with_capacity(n_ops);
  let mut tensors_to_retain = Vec::with_capacity(n_ops);
  let mut traversal_orders = Vec::with_capacity(n_ops);
  let mut subgraph_latencies = Vec::with_capacity(n_ops);

  for idx in 0..n_ops {
    subgraphs.push(vec![idx]);

    let out_tensor_id = _problem.outputs[idx][0];
    let w = _problem.widths[out_tensor_id];
    let h = _problem.heights[out_tensor_id];
    let k = 1_i64;

    granularities.push([w, h, k]);
    tensors_to_retain.push(vec![]);
    traversal_orders.push(None);
    subgraph_latencies.push(0.0);
  }

  Ok(Solution {
    subgraphs: subgraphs,
    granularities: granularities,
    tensors_to_retain: tensors_to_retain,
    traversal_orders: traversal_orders,
    subgraph_latencies: subgraph_latencies,
  })
}

