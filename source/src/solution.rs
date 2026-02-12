use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Solution {
  pub subgraphs: Vec<Vec<usize>>,
  pub granularities: Vec<[i64; 3]>,
  pub tensors_to_retain: Vec<Vec<usize>>,
  pub traversal_orders: Vec<Option<Vec<i64>>>,
  pub subgraph_latencies: Vec<f64>,
}

