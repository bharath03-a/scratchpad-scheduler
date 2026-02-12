use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Problem {
  pub widths: Vec<i64>,
  pub heights: Vec<i64>,
  pub op_types: Vec<String>,
  pub inputs: Vec<Vec<usize>>,
  pub outputs: Vec<Vec<usize>>,
  pub base_costs: Vec<i64>,
  pub fast_memory_capacity: i64,
  pub slow_memory_bandwidth: i64,
  pub native_granularity: [i64; 2],
}

