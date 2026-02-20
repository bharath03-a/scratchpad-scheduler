use anyhow::Result;
use std::collections::{HashMap, HashSet};

use crate::problem::Problem;
use crate::solution::Solution;

/// Helper: Compute tensor size in elements
fn tensor_size(problem: &Problem, tensor_id: usize) -> i64 {
    problem.widths[tensor_id] * problem.heights[tensor_id]
}

/// Helper: Get reduction dimension K for MatMul
/// For MatMul: inputs[0] is LHS (H x K), inputs[1] is RHS (K x W)
/// K is the common dimension (width of LHS = height of RHS)
fn get_matmul_k(problem: &Problem, op_idx: usize) -> i64 {
    if problem.inputs[op_idx].len() < 2 {
        return 1;
    }
    let lhs_id = problem.inputs[op_idx][0];
    let rhs_id = problem.inputs[op_idx][1];
    // K = width of LHS = height of RHS
    problem.widths[lhs_id].min(problem.heights[rhs_id])
}

/// Helper: Compute working set size for a subgraph with given granularity
/// This accounts for tiling - only the slices needed per tile are in memory
fn compute_working_set_with_granularity(
    problem: &Problem,
    ops: &[usize],
    granularity: [i64; 3],
) -> i64 {
    let [w, h, k] = granularity;
    let mut working_set = 0i64;

    // Collect all tensors used
    let mut input_tensors = HashSet::new();
    let mut output_tensors = HashSet::new();

    for &op_idx in ops {
        for &tid in &problem.inputs[op_idx] {
            input_tensors.insert(tid);
        }
        for &tid in &problem.outputs[op_idx] {
            output_tensors.insert(tid);
        }
    }

    // For each output tensor: need w * h slice
    for _tid in &output_tensors {
        working_set += w * h;
    }

    // For each input tensor: depends on op type
    for &op_idx in ops {
        let op_type = &problem.op_types[op_idx];
        
        if op_type == "MatMul" {
            // MatMul: LHS needs h * k, RHS needs w * k
            if problem.inputs[op_idx].len() >= 2 {
                let lhs_id = problem.inputs[op_idx][0];
                let rhs_id = problem.inputs[op_idx][1];
                
                // Only count if not already counted as output from another op in group
                if !output_tensors.contains(&lhs_id) {
                    working_set += h * k;
                }
                if !output_tensors.contains(&rhs_id) {
                    working_set += w * k;
                }
            }
        } else {
            // Pointwise: inputs need w * h slice
            for &tid in &problem.inputs[op_idx] {
                if !output_tensors.contains(&tid) {
                    working_set += w * h;
                }
            }
        }
    }

    working_set
}

/// Helper: Find best granularity for a group of ops
/// Uses binary search for optimal tile size
fn find_best_granularity(
    problem: &Problem,
    ops: &[usize],
) -> ([i64; 3], i64) {
    let native_w = problem.native_granularity[0];
    let native_h = problem.native_granularity[1];

    // Determine output tensor shape (use last op's output)
    let last_op = ops[ops.len() - 1];
    let out_tid = problem.outputs[last_op][0];
    let out_w = problem.widths[out_tid];
    let out_h = problem.heights[out_tid];

    // Determine k dimension
    let mut k = 1i64;
    for &op_idx in ops {
        if problem.op_types[op_idx] == "MatMul" {
            k = get_matmul_k(problem, op_idx);
            break;
        }
    }

    // Try native granularity first
    let w = out_w.min(native_w);
    let h = out_h.min(native_h);
    let granularity = [w, h, k];
    let working_set = compute_working_set_with_granularity(problem, ops, granularity);

    if working_set <= problem.fast_memory_capacity {
        return (granularity, working_set);
    }

    // Binary search for optimal w and h
    let mut best_granularity = granularity;
    let mut best_working_set = working_set;

    // Binary search for w
    let mut w_low = native_w;
    let mut w_high = out_w;
    while w_low < w_high {
        let w_mid = (w_low + w_high + 1) / 2;
        let test_granularity = [w_mid, h, k];
        let test_ws = compute_working_set_with_granularity(problem, ops, test_granularity);
        
        if test_ws <= problem.fast_memory_capacity {
            w_low = w_mid;
            if test_ws < best_working_set {
                best_granularity = test_granularity;
                best_working_set = test_ws;
            }
        } else {
            w_high = w_mid - 1;
        }
    }

    // Binary search for h
    let w_opt = best_granularity[0];
    let mut h_low = native_h;
    let mut h_high = out_h;
    while h_low < h_high {
        let h_mid = (h_low + h_high + 1) / 2;
        let test_granularity = [w_opt, h_mid, k];
        let test_ws = compute_working_set_with_granularity(problem, ops, test_granularity);
        
        if test_ws <= problem.fast_memory_capacity {
            h_low = h_mid;
            if test_ws < best_working_set {
                best_granularity = test_granularity;
                best_working_set = test_ws;
            }
        } else {
            h_high = h_mid - 1;
        }
    }

    // If still doesn't fit, try split-K for MatMul
    if best_working_set > problem.fast_memory_capacity && k > 1 {
        let w_opt = best_granularity[0];
        let h_opt = best_granularity[1];
        
        // Binary search for k
        let mut k_low = 1;
        let mut k_high = k;
        while k_low < k_high {
            let k_mid = (k_low + k_high + 1) / 2;
            let test_granularity = [w_opt, h_opt, k_mid];
            let test_ws = compute_working_set_with_granularity(problem, ops, test_granularity);
            
            if test_ws <= problem.fast_memory_capacity {
                k_low = k_mid;
                if test_ws < best_working_set {
                    best_granularity = test_granularity;
                    best_working_set = test_ws;
                }
            } else {
                k_high = k_mid - 1;
            }
        }
    }

    (best_granularity, best_working_set)
}

/// Helper: Compute number of tiles needed for a granularity
fn num_tiles(tensor_w: i64, tensor_h: i64, tile_w: i64, tile_h: i64) -> i64 {
    let tiles_w = (tensor_w + tile_w - 1) / tile_w;
    let tiles_h = (tensor_h + tile_h - 1) / tile_h;
    tiles_w * tiles_h
}

/// Helper: Generate optimal traversal order for MatMul (zig-zag pattern)
/// Returns None for non-MatMul or when traversal doesn't help
fn generate_traversal_order(
    problem: &Problem,
    ops: &[usize],
    granularity: [i64; 3],
) -> Option<Vec<i64>> {
    // Only optimize for MatMul operations
    if !ops.iter().any(|&op_idx| problem.op_types[op_idx] == "MatMul") {
        return None;
    }

    let [w, h, _k] = granularity;
    let last_op = ops[ops.len() - 1];
    let out_tid = problem.outputs[last_op][0];
    let out_w = problem.widths[out_tid];
    let out_h = problem.heights[out_tid];

    let tiles_w = (out_w + w - 1) / w;
    let tiles_h = (out_h + h - 1) / h;
    let total_tiles = tiles_w * tiles_h;

    // For small number of tiles, traversal order doesn't matter much
    if total_tiles <= 4 {
        return None;
    }

    // Generate zig-zag (snake) pattern: row 0: 0,1,2... row 1: ...2,1,0
    let mut order = Vec::new();
    for row in 0..tiles_h {
        if row % 2 == 0 {
            // Left to right
            for col in 0..tiles_w {
                order.push((row * tiles_w + col) as i64);
            }
        } else {
            // Right to left
            for col in (0..tiles_w).rev() {
                order.push((row * tiles_w + col) as i64);
            }
        }
    }

    Some(order)
}

/// Helper: Compute latency for a grouped subgraph with granularity and traversal order
fn compute_group_latency_with_granularity(
    problem: &Problem,
    ops: &[usize],
    granularity: [i64; 3],
    traversal_order: &Option<Vec<i64>>,
) -> f64 {
    let [w, h, k] = granularity;
    let bandwidth = problem.slow_memory_bandwidth as f64;
    let native_w = problem.native_granularity[0];
    let native_h = problem.native_granularity[1];

    // Total compute time = sum of all op costs
    let mut compute_time: f64 = ops.iter()
        .map(|&op_idx| problem.base_costs[op_idx] as f64)
        .sum();

    // Find boundary tensors
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

    // Compute memory time accounting for traversal order optimization
    let mut memory_time_in = 0.0;
    
    // If we have traversal order, we can reuse data between tiles
    let has_traversal_opt = traversal_order.is_some() && 
                            ops.iter().any(|&op_idx| problem.op_types[op_idx] == "MatMul");
    
    if has_traversal_opt {
        // With zig-zag traversal, we can reuse row/column strips
        // Simplified model: assume 50% data reuse on average
        for &tensor_id in &consumed_tensors {
            if !produced_tensors.contains(&tensor_id) {
                let tensor_w = problem.widths[tensor_id];
                let tensor_h = problem.heights[tensor_id];
                let tiles = num_tiles(tensor_w, tensor_h, w, h);
                let tile_size = w * h;
                // Assume 50% reuse with zig-zag pattern
                memory_time_in += (tiles * tile_size * 3 / 4) as f64 / bandwidth;
            }
        }
    } else {
        // Standard case: no traversal optimization
        for &tensor_id in &consumed_tensors {
            if !produced_tensors.contains(&tensor_id) {
                let tensor_w = problem.widths[tensor_id];
                let tensor_h = problem.heights[tensor_id];
                let tiles = num_tiles(tensor_w, tensor_h, w, h);
                let tile_size = w * h;
                memory_time_in += (tiles * tile_size) as f64 / bandwidth;
            }
        }
    }

    // Graph outputs: produced but not consumed by this group
    let mut memory_time_out = 0.0;
    for &tensor_id in &produced_tensors {
        if !consumed_tensors.contains(&tensor_id) {
            let tensor_w = problem.widths[tensor_id];
            let tensor_h = problem.heights[tensor_id];
            let tiles = num_tiles(tensor_w, tensor_h, w, h);
            let tile_size = w * h;
            memory_time_out += (tiles * tile_size) as f64 / bandwidth;
        }
    }

    // Account for tiling overhead: if granularity < native, we pay native cost per tile
    let last_op = ops[ops.len() - 1];
    let out_tid = problem.outputs[last_op][0];
    let out_w = problem.widths[out_tid];
    let out_h = problem.heights[out_tid];
    
    let tiles_needed = num_tiles(out_w, out_h, w, h);
    
    // If we're tiling smaller than native, we still pay native compute cost per tile
    if w < native_w || h < native_h {
        // Compute cost is paid per native tile, but we produce fewer elements
        // Simplified: scale compute time by tile ratio
        let native_tiles = num_tiles(out_w, out_h, native_w, native_h);
        if native_tiles > 0 {
            compute_time = compute_time * (tiles_needed as f64 / native_tiles as f64);
        }
    }

    // For split-K MatMul: multiply compute by number of k-steps
    if k < get_matmul_k(problem, ops[0]) {
        let full_k = get_matmul_k(problem, ops[0]);
        let k_steps = (full_k + k - 1) / k;
        compute_time *= k_steps as f64;
    }

    let total_memory_time = memory_time_in + memory_time_out;
    compute_time.max(total_memory_time)
}

/// Helper: Build graph structure to understand dependencies
fn build_graph(problem: &Problem) -> (Vec<usize>, HashMap<usize, Vec<usize>>, HashMap<usize, usize>) {
    let n_ops = problem.op_types.len();
    let mut in_degree = vec![0; n_ops];
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut tensor_to_producer: HashMap<usize, usize> = HashMap::new();

    // Build dependency graph
    for op_idx in 0..n_ops {
        for &out_tid in &problem.outputs[op_idx] {
            tensor_to_producer.insert(out_tid, op_idx);
        }
    }

    for op_idx in 0..n_ops {
        for &in_tid in &problem.inputs[op_idx] {
            if let Some(&producer) = tensor_to_producer.get(&in_tid) {
                in_degree[op_idx] += 1;
                consumers.entry(producer).or_insert_with(Vec::new).push(op_idx);
            }
        }
    }

    // Topological sort
    let mut queue = std::collections::VecDeque::new();
    for op_idx in 0..n_ops {
        if in_degree[op_idx] == 0 {
            queue.push_back(op_idx);
        }
    }

    let mut topo_order = Vec::new();
    while let Some(op_idx) = queue.pop_front() {
        topo_order.push(op_idx);
        if let Some(consumers_list) = consumers.get(&op_idx) {
            for &consumer in consumers_list {
                in_degree[consumer] -= 1;
                if in_degree[consumer] == 0 {
                    queue.push_back(consumer);
                }
            }
        }
    }

    (topo_order, consumers, tensor_to_producer)
}

/// Helper: Count how many times each tensor is used
fn count_tensor_usage(problem: &Problem) -> HashMap<usize, usize> {
    let mut usage_count = HashMap::new();
    for op_idx in 0..problem.op_types.len() {
        for &tensor_id in &problem.inputs[op_idx] {
            *usage_count.entry(tensor_id).or_insert(0) += 1;
        }
    }
    usage_count
}

/// Helper: Compute cost of recomputing a tensor vs retaining it
/// Returns (recompute_cost, retain_cost)
fn compute_recompute_vs_retain_cost(
    problem: &Problem,
    tensor_id: usize,
    producer_op: usize,
) -> (f64, f64) {
    let size = tensor_size(problem, tensor_id) as f64;
    let bandwidth = problem.slow_memory_bandwidth as f64;
    
    // Cost to recompute: compute cost of producer op + input memory cost
    let mut recompute_cost = problem.base_costs[producer_op] as f64;
    for &in_tid in &problem.inputs[producer_op] {
        let in_size = tensor_size(problem, in_tid) as f64;
        recompute_cost += in_size / bandwidth;
    }
    
    // Cost to retain: memory transfer cost (load once, keep in fast memory)
    // Plus opportunity cost of using fast memory capacity
    let retain_cost = size / bandwidth;
    
    (recompute_cost, retain_cost)
}

/// Main scheduler with all optimizations
pub fn scheduler_solution(problem: &Problem) -> Result<Solution> {
    let n_ops = problem.op_types.len();
    
    // Build graph structure
    let (topo_order, consumers, _tensor_to_producer) = build_graph(problem);
    let tensor_usage = count_tensor_usage(problem);

    let mut subgraphs = Vec::new();
    let mut granularities = Vec::new();
    let mut tensors_to_retain = Vec::new();
    let mut traversal_orders = Vec::new();
    let mut subgraph_latencies = Vec::new();

    // Track which ops are scheduled
    let mut scheduled = vec![false; n_ops];
    
    // More aggressive grouping: try to form larger groups with look-ahead
    for &op_idx in &topo_order {
        if scheduled[op_idx] {
            continue;
        }

        // Try to build a group starting from this op
        let mut current_group = vec![op_idx];
        scheduled[op_idx] = true;

        // Try to extend the group greedily with look-ahead
        loop {
            let mut extended = false;
            
            // Collect all candidate consumers first (to avoid borrow issues)
            let mut candidates = Vec::new();
            for &group_op in &current_group {
                if let Some(consumer_list) = consumers.get(&group_op) {
                    for &consumer in consumer_list {
                        if !scheduled[consumer] {
                            candidates.push(consumer);
                        }
                    }
                }
            }
            
            // Sort candidates by potential benefit (look-ahead heuristic)
            // Prefer ops that allow further grouping
            candidates.sort_by_key(|&consumer| {
                let mut lookahead_score = 0;
                if let Some(consumer_list) = consumers.get(&consumer) {
                    lookahead_score = consumer_list.len();
                }
                // Negative for max-heap behavior (we want more consumers first)
                -(lookahead_score as i32)
            });
            
            // Try each candidate
            for consumer in candidates {
                let mut candidate = current_group.clone();
                candidate.push(consumer);

                let (_granularity, working_set) = find_best_granularity(problem, &candidate);
                
                if working_set <= problem.fast_memory_capacity {
                    current_group = candidate;
                    scheduled[consumer] = true;
                    extended = true;
                    break;
                }
            }

            if !extended {
                break;
            }
        }

        // Finalize this group
        let (granularity, _working_set) = find_best_granularity(problem, &current_group);
        
        // Generate optimal traversal order
        let traversal_order = generate_traversal_order(problem, &current_group, granularity);
        
        let latency = compute_group_latency_with_granularity(
            problem, 
            &current_group, 
            granularity,
            &traversal_order
        );

        // Determine tensors to retain (for skip connections)
        // Use cost-based decision: recompute vs retain
        let mut retain = Vec::new();
        let mut remaining_capacity = problem.fast_memory_capacity;
        
        // Collect candidate tensors with their costs
        let mut candidates = Vec::new();
        for &op_idx in &current_group {
            for &out_tid in &problem.outputs[op_idx] {
                if let Some(&usage) = tensor_usage.get(&out_tid) {
                    if usage > 1 {
                        let (recompute_cost, retain_cost) = 
                            compute_recompute_vs_retain_cost(problem, out_tid, op_idx);
                        let size = tensor_size(problem, out_tid);
                        
                        candidates.push((out_tid, size, recompute_cost, retain_cost, usage));
                    }
                }
            }
        }
        
        // Sort by benefit: (recompute_cost - retain_cost) / size (benefit per unit memory)
        candidates.sort_by(|a, b| {
            let benefit_a = (a.2 - a.3) / (a.1 as f64 + 1.0);
            let benefit_b = (b.2 - b.3) / (b.1 as f64 + 1.0);
            benefit_b.partial_cmp(&benefit_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Greedily retain tensors that provide benefit and fit
        for (tensor_id, size, recompute_cost, retain_cost, _usage) in candidates {
            if retain_cost < recompute_cost && size <= remaining_capacity {
                retain.push(tensor_id);
                remaining_capacity -= size;
            }
        }

        subgraphs.push(current_group);
        granularities.push(granularity);
        tensors_to_retain.push(retain);
        traversal_orders.push(traversal_order);
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
