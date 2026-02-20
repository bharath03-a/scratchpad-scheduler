use anyhow::Result;

use crate::problem::Problem;
use crate::solution::Solution;

/// Helper: Compute tensor size in elements
fn tensor_size(problem: &Problem, tensor_id: usize) -> i64 {
    problem.widths[tensor_id] * problem.heights[tensor_id]
}

/// Helper: Compute latency for a single op subgraph
/// This implements the roofline model: latency = max(compute_time, memory_time)
fn compute_op_latency(problem: &Problem, op_idx: usize) -> f64 {
    let bandwidth = problem.slow_memory_bandwidth as f64;
    
    // Compute time = base cost of the operation
    let compute_time = problem.base_costs[op_idx] as f64;
    
    // Memory time = time to load inputs + time to store outputs
    let mut memory_time_in = 0.0;
    for &input_tensor_id in &problem.inputs[op_idx] {
        let size = tensor_size(problem, input_tensor_id);
        memory_time_in += size as f64 / bandwidth;
    }
    
    let mut memory_time_out = 0.0;
    for &output_tensor_id in &problem.outputs[op_idx] {
        let size = tensor_size(problem, output_tensor_id);
        memory_time_out += size as f64 / bandwidth;
    }
    
    let total_memory_time = memory_time_in + memory_time_out;
    
    // Roofline: bottleneck is the slower resource
    compute_time.max(total_memory_time)
}

/// Helper: Estimate working set size for a group of ops
/// Returns (working_set_size, fits_in_memory)
/// For now, we use a simple heuristic: sum of all input and output tensor sizes
fn estimate_working_set(problem: &Problem, ops: &[usize]) -> (i64, bool) {
    use std::collections::HashSet;
    
    let mut tensor_ids = HashSet::new();
    
    // Collect all tensors used by this group
    for &op_idx in ops {
        // Add input tensors
        for &tensor_id in &problem.inputs[op_idx] {
            tensor_ids.insert(tensor_id);
        }
        // Add output tensors
        for &tensor_id in &problem.outputs[op_idx] {
            tensor_ids.insert(tensor_id);
        }
    }
    
    // Sum their sizes
    let total_size: i64 = tensor_ids.iter()
        .map(|&tid| tensor_size(problem, tid))
        .sum();
    
    let fits = total_size <= problem.fast_memory_capacity;
    (total_size, fits)
}

/// Helper: Compute latency for a grouped subgraph
/// When ops are grouped, intermediate tensors become ephemeral (zero cost)
/// Only boundary tensors (graph inputs/outputs) need memory transfers
fn compute_group_latency(problem: &Problem, ops: &[usize]) -> f64 {
    use std::collections::HashSet;
    
    let bandwidth = problem.slow_memory_bandwidth as f64;
    
    // Total compute time = sum of all op costs
    let compute_time: f64 = ops.iter()
        .map(|&op_idx| problem.base_costs[op_idx] as f64)
        .sum();
    
    // Find boundary tensors: graph inputs and graph outputs
    // Graph inputs: tensors that are NOT produced by any op in this group
    // Graph outputs: tensors that are NOT consumed by any op in this group
    
    let mut produced_tensors = HashSet::new();
    let mut consumed_tensors = HashSet::new();
    
    for &op_idx in ops {
        for &out_tid in &problem.outputs[op_idx] {
            produced_tensors.insert(out_tid);
        }
        for &in_tid in &problem.inputs[op_idx] {
            consumed_tensors.insert(in_tid);
        }
    }
    
    // Graph inputs: consumed but not produced by this group
    let mut memory_time_in = 0.0;
    for &tensor_id in &consumed_tensors {
        if !produced_tensors.contains(&tensor_id) {
            let size = tensor_size(problem, tensor_id);
            memory_time_in += size as f64 / bandwidth;
        }
    }
    
    // Graph outputs: produced but not consumed by this group
    let mut memory_time_out = 0.0;
    for &tensor_id in &produced_tensors {
        if !consumed_tensors.contains(&tensor_id) {
            let size = tensor_size(problem, tensor_id);
            memory_time_out += size as f64 / bandwidth;
        }
    }
    
    let total_memory_time = memory_time_in + memory_time_out;
    
    compute_time.max(total_memory_time)
}

/// Main scheduler: Greedy grouping strategy
/// 
/// Strategy:
/// 1. Try to group consecutive ops in topological order
/// 2. For each op, check if adding it to current group would fit in memory
/// 3. If yes, extend the group; if no, finalize current group and start new one
/// 4. Fallback: if grouping fails, use per-op subgraphs
pub fn scheduler_solution(problem: &Problem) -> Result<Solution> {
    let n_ops = problem.op_types.len();
    
    // Strategy: Try greedy grouping first
    let mut subgraphs = Vec::new();
    let mut granularities = Vec::new();
    let mut tensors_to_retain = Vec::new();
    let mut traversal_orders = Vec::new();
    let mut subgraph_latencies = Vec::new();
    
    let mut current_group = Vec::new();
    
    // Process ops in order (assuming topological order)
    for op_idx in 0..n_ops {
        // Try to add this op to current group
        let mut candidate_group = current_group.clone();
        candidate_group.push(op_idx);
        
        let (_working_set_size, fits) = estimate_working_set(problem, &candidate_group);
        
        if fits && !current_group.is_empty() {
            // Can extend current group
            current_group = candidate_group;
        } else {
            // Cannot extend (or first op): finalize current group if any
            if !current_group.is_empty() {
                // Finalize current group
                let group_ops = current_group.clone();
                let latency = compute_group_latency(problem, &group_ops);
                
                // Choose granularity based on last op's output
                let last_op = group_ops[group_ops.len() - 1];
                let out_tensor_id = problem.outputs[last_op][0];
                let w = problem.widths[out_tensor_id];
                let h = problem.heights[out_tensor_id];
                let k = 1_i64; // For now, use k=1 (will improve for MatMul later)
                
                subgraphs.push(group_ops);
                granularities.push([w, h, k]);
                tensors_to_retain.push(vec![]); // No retention for now
                traversal_orders.push(None);
                subgraph_latencies.push(latency);
            }
            
            // Start new group with this op
            current_group = vec![op_idx];
        }
    }
    
    // Finalize last group
    if !current_group.is_empty() {
        let latency = compute_group_latency(problem, &current_group);
        
        let last_op = current_group[current_group.len() - 1];
        let out_tensor_id = problem.outputs[last_op][0];
        let w = problem.widths[out_tensor_id];
        let h = problem.heights[out_tensor_id];
        let k = 1_i64;
        
        subgraphs.push(current_group);
        granularities.push([w, h, k]);
        tensors_to_retain.push(vec![]);
        traversal_orders.push(None);
        subgraph_latencies.push(latency);
    }
    
    // Fallback: If no groups were created (shouldn't happen), use per-op
    if subgraphs.is_empty() {
        return fallback_per_op_schedule(problem);
    }
    
    Ok(Solution {
        subgraphs,
        granularities,
        tensors_to_retain,
        traversal_orders,
        subgraph_latencies,
    })
}

/// Fallback: One subgraph per op (always valid, but slower)
fn fallback_per_op_schedule(problem: &Problem) -> Result<Solution> {
    let n_ops = problem.op_types.len();

  let mut subgraphs = Vec::with_capacity(n_ops);
  let mut granularities = Vec::with_capacity(n_ops);
  let mut tensors_to_retain = Vec::with_capacity(n_ops);
  let mut traversal_orders = Vec::with_capacity(n_ops);
  let mut subgraph_latencies = Vec::with_capacity(n_ops);

    for op_idx in 0..n_ops {
        subgraphs.push(vec![op_idx]);
        
        let out_tensor_id = problem.outputs[op_idx][0];
        let w = problem.widths[out_tensor_id];
        let h = problem.heights[out_tensor_id];
    let k = 1_i64;

    granularities.push([w, h, k]);
    tensors_to_retain.push(vec![]);
    traversal_orders.push(None);
        
        // Compute correct latency
        let latency = compute_op_latency(problem, op_idx);
        subgraph_latencies.push(latency);
  }

  Ok(Solution {
        subgraphs,
        granularities,
        tensors_to_retain,
        traversal_orders,
        subgraph_latencies,
  })
}

