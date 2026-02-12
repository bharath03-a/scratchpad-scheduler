use anyhow::Result;

use crate::problem::Problem;
use crate::solution::Solution;

// TODO: implement real scheduler later.
pub fn scheduler_solution(_problem: &Problem) -> Result<Solution> {
  Ok(Solution {
    subgraphs: Vec::new(),
    granularities: Vec::new(),
    tensors_to_retain: Vec::new(),
    traversal_orders: Vec::new(),
    subgraph_latencies: Vec::new(),
  })
}

