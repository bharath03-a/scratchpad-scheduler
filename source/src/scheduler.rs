use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::problem::Problem;
use crate::solution::Solution;

// ===== HELPERS =====

fn ceil_div(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// K_full for a MatMul op = width of LHS tensor (reduction dimension = k parameter in spec).
fn matmul_k_full(problem: &Problem, op_idx: usize) -> i64 {
    let lhs_id = problem.inputs[op_idx][0];
    problem.widths[lhs_id]
}

/// Classify tensors in a subgraph:
/// - boundary_inputs: consumed by ops in group but NOT produced by any op in group
/// - final_outputs: produced by ops in group AND either a graph output or consumed outside group
/// - ephemeral: produced by one op in group and consumed by another op in group
fn classify_boundary(
    problem: &Problem,
    ops: &[usize],
    tensor_to_producer: &HashMap<usize, usize>,
) -> (Vec<usize>, Vec<usize>) {
    let op_set: HashSet<usize> = ops.iter().cloned().collect();

    // tensors produced within the group
    let mut produced: HashSet<usize> = HashSet::new();
    for &op in ops {
        for &t in &problem.outputs[op] {
            produced.insert(t);
        }
    }

    // tensors consumed within the group
    let mut consumed_within: HashSet<usize> = HashSet::new();
    for &op in ops {
        for &t in &problem.inputs[op] {
            consumed_within.insert(t);
        }
    }

    // boundary_inputs: consumed but not produced within group
    let mut boundary_inputs: Vec<usize> = consumed_within
        .iter()
        .filter(|&&t| !produced.contains(&t))
        .cloned()
        .collect();
    boundary_inputs.sort();
    boundary_inputs.dedup();

    // final_outputs: produced by group but not consumed by any op in the group
    // (i.e., must eventually be written to slow memory)
    let mut final_outputs: Vec<usize> = produced
        .iter()
        .filter(|&&t| {
            // check if consumed by any op OUTSIDE the group
            // if a tensor is only consumed within the group → ephemeral, skip
            let consumed_outside = (0..problem.op_types.len()).any(|other_op| {
                !op_set.contains(&other_op)
                    && problem.inputs[other_op].contains(&t)
            });
            // also include if it's a graph output (not consumed by any op)
            let is_graph_output = !problem.inputs.iter().any(|inputs| inputs.contains(&t));
            consumed_outside || is_graph_output
        })
        .cloned()
        .collect();
    final_outputs.sort();
    final_outputs.dedup();

    (boundary_inputs, final_outputs)
}

/// The "primary" output tensor for spatial tiling: the output of the last op in the group
fn primary_output_tensor(problem: &Problem, ops: &[usize]) -> usize {
    // Use last op's first output
    let last_op = *ops.last().unwrap();
    problem.outputs[last_op][0]
}

// ===== TENSOR ROLE ANALYSIS =====

/// Pre-computed boundary tensor roles for a subgraph.
/// Shared between compute_working_set and compute_latency to avoid duplicating
/// the produced/LHS/RHS/pointwise classification loops.
struct SubgraphTensors {
    /// Non-ephemeral MatMul LHS tensors: (tensor_id, K_full)
    /// Each is kept fully resident in fast memory across all k-steps.
    lhs: Vec<(usize, i64)>,
    /// Non-ephemeral MatMul RHS tensor IDs (streamed k-strip per k-step)
    rhs: Vec<usize>,
    /// Non-MatMul boundary input tensor IDs (pointwise / extra inputs)
    pointwise: Vec<usize>,
    /// K_full of the first MatMul op (governs k_steps for the group)
    k_full: i64,
    has_matmul: bool,
}

impl SubgraphTensors {
    fn analyze(problem: &Problem, ops: &[usize]) -> Self {
        let produced: HashSet<usize> = ops.iter()
            .flat_map(|&op| problem.outputs[op].iter().cloned())
            .collect();

        let has_matmul = ops.iter().any(|&op| problem.op_types[op] == "MatMul");
        let mut lhs = Vec::new();
        let mut rhs = Vec::new();
        let mut counted: HashSet<usize> = HashSet::new();

        if has_matmul {
            for &op in ops {
                if problem.op_types[op] == "MatMul" && problem.inputs[op].len() >= 2 {
                    let lhs_id = problem.inputs[op][0];
                    if !produced.contains(&lhs_id) && counted.insert(lhs_id) {
                        lhs.push((lhs_id, problem.widths[lhs_id]));
                    }
                    let rhs_id = problem.inputs[op][1];
                    if !produced.contains(&rhs_id) && counted.insert(rhs_id) {
                        rhs.push(rhs_id);
                    }
                }
            }
        }

        let mut pointwise = Vec::new();
        for &op in ops {
            for &t in &problem.inputs[op] {
                if !produced.contains(&t) && counted.insert(t) {
                    pointwise.push(t);
                }
            }
        }

        let k_full = if has_matmul {
            let matmul_op = *ops.iter().find(|&&op| problem.op_types[op] == "MatMul").unwrap();
            matmul_k_full(problem, matmul_op)
        } else {
            1
        };

        SubgraphTensors { lhs, rhs, pointwise, k_full, has_matmul }
    }
}

// ===== WORKING SET COMPUTATION =====

/// Compute the peak working set size in fast memory for one execution step.
pub fn compute_working_set(problem: &Problem, ops: &[usize], gran: [i64; 3]) -> i64 {
    let [w, h, k] = gran;
    let t = SubgraphTensors::analyze(problem, ops);

    let mut ws = w * h; // output accumulator

    for (_, k_full_lhs) in &t.lhs {
        ws += h * k_full_lhs; // LHS fully resident: h × K_full
    }
    for _ in &t.rhs {
        ws += k * w; // RHS k-strip per step
    }
    for _ in &t.pointwise {
        ws += w * h; // pointwise tile
    }

    ws
}

// ===== LATENCY COMPUTATION =====

/// Compute total subgraph latency with the given granularity and traversal order.
///
/// Key model (validated against reference examples):
/// - Spatial tiles: tiles_w × tiles_h over the output tensor
/// - Split-K steps: k_steps = ceil(K_full / k) per spatial tile
/// - Compute per k-step = sum(base_costs) / k_steps  (constant regardless of tile size)
/// - Memory per non-last k-step: (lhs_slice + rhs_slice) / bw
/// - Memory per last k-step: (lhs_slice + rhs_slice + out_slice) / bw
/// - Tile latency = max(compute_per_step, memory_per_step)
/// - Total = sum over all (spatial_tile, k_step)
///
/// For zig-zag traversal (MatMul, k=K_full, no split-K):
/// - First tile: load LHS + RHS + write Out
/// - Within-row col change: load RHS + write Out
/// - Row transition: load LHS + write Out
pub fn compute_latency(
    problem: &Problem,
    ops: &[usize],
    gran: [i64; 3],
    use_zigzag: bool,
    resident_tensors: &HashSet<usize>,
) -> f64 {
    let [w, h, k] = gran;
    let bw = problem.slow_memory_bandwidth as f64;
    let native_w = problem.native_granularity[0];
    let native_h = problem.native_granularity[1];

    let dummy_producer: HashMap<usize, usize> = HashMap::new();
    let (_, final_outputs) = classify_boundary(problem, ops, &dummy_producer);

    let out_tensor = primary_output_tensor(problem, ops);
    let out_w = problem.widths[out_tensor];
    let out_h = problem.heights[out_tensor];

    let tiles_w = ceil_div(out_w, w);
    let tiles_h = ceil_div(out_h, h);
    let total_spatial = tiles_w * tiles_h;

    // Total base cost: paid per native tile × scaling for sub-native padding
    // When w ≤ native_w and h ≤ native_h: each spatial tile pays full native cost
    // When w > native_w or h > native_h: multiple native tiles per spatial tile
    let native_tiles_per_spatial =
        ceil_div(w, native_w).max(1) * ceil_div(h, native_h).max(1);
    let base_cost: f64 = ops.iter().map(|&op| problem.base_costs[op] as f64).sum::<f64>()
        * native_tiles_per_spatial as f64;

    let t = SubgraphTensors::analyze(problem, ops);

    if !t.has_matmul {
        // === POINTWISE-ONLY ===
        let mem_in_per_tile: f64 = t.pointwise.iter()
            .filter(|&&tid| !resident_tensors.contains(&tid))
            .count() as f64
            * (w * h) as f64 / bw;
        let mem_out_per_tile: f64 = final_outputs.len() as f64 * (w * h) as f64 / bw;
        let lat_per_tile = base_cost.max(mem_in_per_tile + mem_out_per_tile);
        return total_spatial as f64 * lat_per_tile;
    }

    // === MATMUL (or mixed MatMul+Pointwise) ===
    let k_steps = ceil_div(t.k_full, k);
    let compute_per_kstep = base_cost / k_steps as f64;
    let out_slice = h * w;

    // LHS: sum h × K_full for each non-resident LHS tensor
    let total_lhs_full: i64 = t.lhs.iter()
        .filter(|(tid, _)| !resident_tensors.contains(tid))
        .map(|(_, k_full_lhs)| h * k_full_lhs)
        .sum();
    let has_any_lhs = !t.lhs.is_empty();

    // RHS: k × w per strip; track first for zig-zag RHS-reuse formula
    let rhs_strips_total: i64 = t.rhs.iter()
        .map(|&tid| if resident_tensors.contains(&tid) { 0 } else { k * w })
        .sum();
    let first_rhs_strip: i64 = t.rhs.first()
        .map(|&tid| if resident_tensors.contains(&tid) { 0 } else { k * w })
        .unwrap_or(0);

    // Pointwise boundary: any non-LHS/RHS boundary inputs, loaded in the last k-step
    let pointwise_boundary: i64 = t.pointwise.iter()
        .filter(|&&tid| !resident_tensors.contains(&tid))
        .count() as i64
        * h * w;

    // Zig-zag eligibility: need multiple column tiles for LHS reuse
    // For k_steps==1 (single-step MatMul): zig-zag enables BOTH LHS reuse (within row)
    //   AND RHS reuse (across row transitions).
    // For k_steps>1 (split-K): zig-zag enables only LHS reuse within rows.
    let use_zigzag_actual = use_zigzag && tiles_w > 1 && has_any_lhs;

    if !use_zigzag_actual {
        // === RASTER ORDER: every spatial tile independently loads LHS+RHS ===
        // LHS is NOT kept resident across tiles in raster order.
        let lat_per_tile = if k_steps == 1 {
            compute_per_kstep.max(
                (total_lhs_full + rhs_strips_total + out_slice + pointwise_boundary) as f64 / bw,
            )
        } else {
            let lat_step1 = compute_per_kstep.max((total_lhs_full + rhs_strips_total) as f64 / bw);
            let lat_middle = compute_per_kstep.max(rhs_strips_total as f64 / bw);
            let lat_last = compute_per_kstep
                .max((rhs_strips_total + out_slice + pointwise_boundary) as f64 / bw);
            lat_step1 + (k_steps - 2) as f64 * lat_middle + lat_last
        };
        return total_spatial as f64 * lat_per_tile;
    }

    // === ZIG-ZAG ORDER: LHS reuse within rows ===
    // Traversal: row-major zig-zag (snake pattern).
    // Within a row: first tile of each row loads ALL LHS tensors, subsequent tiles reuse them.
    // Row transition: reload all LHS tensors (new h-strip).

    if k_steps == 1 {
        // k_steps==1: also enables RHS reuse across row transitions.
        // lat_row_change: reload all LHS, keep last RHS from previous row's last col.
        // lat_col_change: keep all LHS, load new RHS.
        // Pointwise boundary is loaded per tile (every tile writes output + runs Pointwise).
        let lat_full = compute_per_kstep.max(
            (total_lhs_full + first_rhs_strip + out_slice + pointwise_boundary) as f64 / bw,
        );
        let lat_row_change = compute_per_kstep
            .max((total_lhs_full + out_slice + pointwise_boundary) as f64 / bw);
        let lat_col_change = compute_per_kstep
            .max((first_rhs_strip + out_slice + pointwise_boundary) as f64 / bw);
        return 1.0 * lat_full
            + (tiles_h - 1) as f64 * lat_row_change
            + (tiles_h * (tiles_w - 1)) as f64 * lat_col_change;
    }

    // k_steps>1: LHS reuse within rows only; no RHS reuse across row transitions.
    // Pointwise boundary is loaded in the last k-step (when output is written).
    let lat_step1_with_lhs = compute_per_kstep.max((total_lhs_full + rhs_strips_total) as f64 / bw);
    let lat_step1_no_lhs = compute_per_kstep.max(rhs_strips_total as f64 / bw);
    let lat_middle = compute_per_kstep.max(rhs_strips_total as f64 / bw);
    let lat_last =
        compute_per_kstep.max((rhs_strips_total + out_slice + pointwise_boundary) as f64 / bw);
    // First tile in each row: load LHS_full + RHS_strips in step1
    let lat_first_tile = lat_step1_with_lhs + (k_steps - 2) as f64 * lat_middle + lat_last;
    // Subsequent tiles in same row: LHS already resident → step1 is lat_step1_no_lhs
    let lat_rest_tile = lat_step1_no_lhs + (k_steps - 2) as f64 * lat_middle + lat_last;
    let lat_per_row = lat_first_tile + (tiles_w - 1) as f64 * lat_rest_tile;
    tiles_h as f64 * lat_per_row
}

// ===== TRAVERSAL ORDER GENERATION =====

/// Generate zig-zag (snake) traversal order for a grid of tiles_h × tiles_w.
/// Returns None if single tile (no optimization needed).
pub fn gen_zigzag_order(tiles_h: i64, tiles_w: i64) -> Option<Vec<i64>> {
    if tiles_h <= 1 && tiles_w <= 1 {
        return None;
    }
    let mut order = Vec::with_capacity((tiles_h * tiles_w) as usize);
    for row in 0..tiles_h {
        if row % 2 == 0 {
            for col in 0..tiles_w {
                order.push(row * tiles_w + col);
            }
        } else {
            for col in (0..tiles_w).rev() {
                order.push(row * tiles_w + col);
            }
        }
    }
    Some(order)
}

// ===== GRANULARITY SEARCH =====

/// Find the best granularity [w, h, k] for a subgraph that:
/// 1. Keeps working set ≤ fast_memory_capacity
/// 2. Minimizes total latency
///
/// Returns (granularity, traversal_order, latency)
pub fn find_best_granularity(
    problem: &Problem,
    ops: &[usize],
    resident_tensors: &HashSet<usize>,
) -> ([i64; 3], Option<Vec<i64>>, f64) {
    let native_w = problem.native_granularity[0];
    let native_h = problem.native_granularity[1];
    let cap = problem.fast_memory_capacity;

    // Determine output dimensions
    let out_tensor = primary_output_tensor(problem, ops);
    let out_w = problem.widths[out_tensor];
    let out_h = problem.heights[out_tensor];

    let has_matmul = ops.iter().any(|&op| problem.op_types[op] == "MatMul");

    // Find K_full for MatMul ops
    let k_full = if has_matmul {
        let matmul_op = *ops.iter().find(|&&op| problem.op_types[op] == "MatMul").unwrap();
        matmul_k_full(problem, matmul_op)
    } else {
        1
    };

    // Generate candidate w values: multiples of native_w that divide the output range
    // Also include sub-native candidates for tight memory situations
    let w_candidates = gen_dimension_candidates(out_w, native_w);
    let h_candidates = gen_dimension_candidates(out_h, native_h);

    // Generate k candidates: from k_full down to native_w (or 1 if smaller)
    let k_candidates: Vec<i64> = if has_matmul {
        gen_k_candidates(k_full, native_w)
    } else {
        vec![1]
    };

    let mut best_gran = [native_w.min(out_w), native_h.min(out_h), k_full.min(1i64.max(native_w))];
    let mut best_lat = f64::INFINITY;
    let mut best_traversal: Option<Vec<i64>> = None;

    // For each combination, find feasible ones and pick best latency
    // Strategy: try k from largest to smallest (prefer no split-K)
    // For each k, try largest (w, h) that fits
    for &k in k_candidates.iter().rev() {
        for &w in w_candidates.iter().rev() {
            for &h in h_candidates.iter().rev() {
                let gran = [w, h, k];
                let ws = compute_working_set(problem, ops, gran);
                if ws > cap {
                    continue; // doesn't fit
                }

                // Try with and without zigzag
                let tiles_w = ceil_div(out_w, w);
                let tiles_h = ceil_div(out_h, h);

                // Raster order
                let lat_raster = compute_latency(problem, ops, gran, false, resident_tensors);
                if lat_raster < best_lat {
                    best_lat = lat_raster;
                    best_gran = gran;
                    best_traversal = None;
                }

                // Zig-zag order: enables LHS reuse within rows (always), and
                // also RHS reuse across row transitions when k_steps==1.
                // Beneficial whenever there are multiple column tiles (tiles_w > 1).
                if has_matmul && tiles_w > 1 {
                    let lat_zz = compute_latency(problem, ops, gran, true, resident_tensors);
                    if lat_zz < best_lat {
                        best_lat = lat_zz;
                        best_gran = gran;
                        best_traversal = gen_zigzag_order(tiles_h, tiles_w);
                    }
                }
            }
        }
    }

    // Emergency fallback: try smallest possible tile
    if best_lat.is_infinite() {
        // Try minimum tile sizes
        for k in (1..=k_full).rev().step_by(1) {
            let w = native_w.min(out_w);
            let h = native_h.min(out_h);
            let gran = [w, h, k];
            let ws = compute_working_set(problem, ops, gran);
            if ws <= cap {
                let lat = compute_latency(problem, ops, gran, false, resident_tensors);
                best_gran = gran;
                best_lat = lat;
                best_traversal = None;
                break;
            }
        }
        // If still doesn't fit, use sub-native tiles
        if best_lat.is_infinite() {
            let mut w = native_w.min(out_w);
            let mut h = native_h.min(out_h);
            loop {
                let gran = [w, h, 1i64.max(k_full / 64)];
                let ws = compute_working_set(problem, ops, gran);
                if ws <= cap {
                    let lat = compute_latency(problem, ops, gran, false, resident_tensors);
                    best_gran = gran;
                    best_lat = lat;
                    break;
                }
                w = (w / 2).max(1);
                h = (h / 2).max(1);
                if w == 1 && h == 1 {
                    break;
                }
            }
        }
    }

    (best_gran, best_traversal, best_lat)
}

/// Generate candidate tile widths/heights.
/// Includes: multiples of native up to tensor dim, and sub-native powers of 2.
fn gen_dimension_candidates(tensor_dim: i64, native: i64) -> Vec<i64> {
    let mut candidates: HashSet<i64> = HashSet::new();

    // Sub-native candidates (needed when memory is tight)
    let mut sub = native;
    while sub >= 1 {
        candidates.insert(sub.min(tensor_dim));
        sub /= 2;
    }
    candidates.insert(1);

    // Native multiples up to tensor_dim
    let mut n = native;
    while n <= tensor_dim {
        candidates.insert(n);
        n += native;
    }
    candidates.insert(tensor_dim); // always include full dim

    let mut v: Vec<i64> = candidates.into_iter().filter(|&x| x > 0).collect();
    v.sort();
    v
}

/// Generate candidate k values: from k_full down in powers-of-2 and native-size steps.
fn gen_k_candidates(k_full: i64, native: i64) -> Vec<i64> {
    let mut candidates: HashSet<i64> = HashSet::new();
    candidates.insert(k_full);

    // Powers of 2 divisions of k_full
    let mut k = k_full;
    while k >= 1 {
        candidates.insert(k);
        k /= 2;
    }

    // Native-size multiples down
    let mut kn = native;
    while kn <= k_full {
        candidates.insert(kn);
        kn += native;
    }
    candidates.insert(1);

    let mut v: Vec<i64> = candidates.into_iter().filter(|&x| x > 0).collect();
    v.sort();
    v
}

// ===== GRAPH STRUCTURE =====

pub struct GraphInfo {
    pub topo_order: Vec<usize>,
    pub consumers: HashMap<usize, Vec<usize>>,
    pub tensor_to_producer: HashMap<usize, usize>,
    pub tensor_usage_count: HashMap<usize, usize>,
}

pub fn build_graph(problem: &Problem) -> GraphInfo {
    let n_ops = problem.op_types.len();
    let mut in_degree = vec![0usize; n_ops];
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut tensor_to_producer: HashMap<usize, usize> = HashMap::new();
    let mut tensor_usage_count: HashMap<usize, usize> = HashMap::new();

    for op in 0..n_ops {
        for &t in &problem.outputs[op] {
            tensor_to_producer.insert(t, op);
        }
    }

    for op in 0..n_ops {
        for &t in &problem.inputs[op] {
            *tensor_usage_count.entry(t).or_insert(0) += 1;
            if let Some(&producer) = tensor_to_producer.get(&t) {
                in_degree[op] += 1;
                consumers.entry(producer).or_default().push(op);
            }
        }
    }

    // Topological sort (Kahn's algorithm)
    let mut queue: VecDeque<usize> = (0..n_ops).filter(|&op| in_degree[op] == 0).collect();
    let mut topo_order = Vec::new();
    let mut deg = in_degree.clone();

    while let Some(op) = queue.pop_front() {
        topo_order.push(op);
        if let Some(cons) = consumers.get(&op) {
            for &c in cons {
                deg[c] -= 1;
                if deg[c] == 0 {
                    queue.push_back(c);
                }
            }
        }
    }

    GraphInfo {
        topo_order,
        consumers,
        tensor_to_producer,
        tensor_usage_count,
    }
}

// ===== TENSORS TO RETAIN =====

/// Decide which output tensors of a subgraph to retain in fast memory.
/// A tensor is worth retaining if the next scheduled subgraph uses it
/// (avoids write + future re-load costs).
///
/// Constraints:
/// - Retained tensors must fit in fast memory alongside the NEXT subgraph's working set
/// - Only retain if it provides latency benefit
fn decide_retain(
    problem: &Problem,
    current_ops: &[usize],
    current_gran: [i64; 3],
    next_subgraph_ops: Option<&[usize]>,
    next_gran: Option<[i64; 3]>,
    graph: &GraphInfo,
) -> Vec<usize> {
    let [w, h, _k] = current_gran;
    let bw = problem.slow_memory_bandwidth as f64;
    let cap = problem.fast_memory_capacity;

    // Collect output tensors of current subgraph that are used by the next
    let dummy_producer: HashMap<usize, usize> = HashMap::new();
    let (_, final_outputs) = classify_boundary(problem, current_ops, &dummy_producer);

    let next_ops = match next_subgraph_ops {
        Some(ops) => ops,
        None => return vec![], // last subgraph, no need to retain
    };

    let (next_boundary_inputs, _) = classify_boundary(problem, next_ops, &dummy_producer);
    let next_input_set: HashSet<usize> = next_boundary_inputs.iter().cloned().collect();

    // Only consider retaining tensors needed by the next subgraph
    let candidates: Vec<usize> = final_outputs
        .iter()
        .filter(|&&t| next_input_set.contains(&t))
        .cloned()
        .collect();

    if candidates.is_empty() {
        return vec![];
    }

    // Check if retaining fits in fast memory for the next subgraph
    let next_gran_val = match next_gran {
        Some(g) => g,
        None => return vec![],
    };

    let next_ws = compute_working_set(problem, next_ops, next_gran_val);
    let mut available = cap - next_ws;
    if available <= 0 {
        return vec![];
    }

    let mut retained = Vec::new();
    for &t in &candidates {
        let tensor_size = problem.widths[t] * problem.heights[t];

        // Benefit: save write cost (eviction) + save future load cost
        let save_write = (w * h) as f64 / bw; // approximate: size of one tile write... actually full tensor
        let save_load = tensor_size as f64 / bw;
        let benefit = save_write + save_load;

        // Cost: occupy fast memory (may cause higher granularity constraints for next)
        // For simplicity: retain if it fits
        if tensor_size <= available && benefit > 0.0 {
            retained.push(t);
            available -= tensor_size;
        }
    }

    retained
}

// ===== MAIN SCHEDULER =====

pub fn scheduler_solution(problem: &Problem) -> Result<Solution> {
    let graph = build_graph(problem);
    let n_ops = problem.op_types.len();

    let mut subgraphs: Vec<Vec<usize>> = Vec::new();
    let mut granularities: Vec<[i64; 3]> = Vec::new();
    let mut tensors_to_retain: Vec<Vec<usize>> = Vec::new();
    let mut traversal_orders: Vec<Option<Vec<i64>>> = Vec::new();
    let mut subgraph_latencies: Vec<f64> = Vec::new();

    let mut scheduled = vec![false; n_ops];
    let mut resident_tensors: HashSet<usize> = HashSet::new();

    // Build all subgraphs first (grouping phase), then compute latencies
    let mut all_groups: Vec<Vec<usize>> = Vec::new();

    for &op in &graph.topo_order {
        if scheduled[op] {
            continue;
        }

        let mut group = vec![op];
        scheduled[op] = true;

        // Try to extend the group greedily
        // Use beam search: keep top-K candidates
        'extend: loop {
            let mut best_candidate: Option<usize> = None;
            let mut best_lat = f64::INFINITY;

            // Collect all schedulable consumers of ops in current group
            let mut candidates: Vec<usize> = Vec::new();
            for &gop in &group {
                if let Some(cons) = graph.consumers.get(&gop) {
                    for &c in cons {
                        if !scheduled[c] && !candidates.contains(&c) {
                            // Check if ALL predecessors of c are in the group or already scheduled
                            let all_preds_ready = problem.inputs[c].iter().all(|&t| {
                                if let Some(&producer) = graph.tensor_to_producer.get(&t) {
                                    scheduled[producer] || group.contains(&producer)
                                } else {
                                    true // graph input
                                }
                            });
                            if all_preds_ready {
                                candidates.push(c);
                            }
                        }
                    }
                }
            }

            if candidates.is_empty() {
                break 'extend;
            }

            // Try each candidate: pick the one that gives best latency improvement
            for &candidate in &candidates {
                let mut trial = group.clone();
                trial.push(candidate);

                // Quick feasibility: minimum working set is with tile [1,1,1]
                // n_boundary_inputs × 1 + 1 (output) → always tiny. Just allow it.
                // The full search in find_best_granularity handles OOM cases.

                let (trial_gran, trial_traversal, trial_lat) =
                    find_best_granularity(problem, &trial, &resident_tensors);

                if trial_lat < best_lat {
                    best_lat = trial_lat;
                    best_candidate = Some(candidate);
                }
            }

            // Compare: is adding the best candidate better than not adding it?
            if let Some(c) = best_candidate {
                let (solo_gran, _, solo_lat) =
                    find_best_granularity(problem, &group, &resident_tensors);

                // Estimate cost of running c separately after the group
                let (c_gran, _, c_lat) =
                    find_best_granularity(problem, &[c], &resident_tensors);

                // Adding c to group is beneficial if group_lat_with_c < group_lat + c_lat
                // i.e., it saves latency
                if best_lat < solo_lat + c_lat {
                    group.push(c);
                    scheduled[c] = true;
                } else {
                    break 'extend;
                }
            } else {
                break 'extend;
            }
        }

        all_groups.push(group);
    }

    // Post-process: merge consecutive groups if it reduces total latency.
    // The greedy forward pass can't look ahead past a "worse" intermediate state,
    // so this catch-up pass handles cases where merging across group boundaries helps.
    let mut merge_improved = true;
    while merge_improved {
        merge_improved = false;
        let mut i = 0;
        while i + 1 < all_groups.len() {
            let mut merged = all_groups[i].clone();
            merged.extend_from_slice(&all_groups[i + 1]);
            let empty_resident: HashSet<usize> = HashSet::new();
            let (_, _, lat_merged) = find_best_granularity(problem, &merged, &empty_resident);
            let (_, _, lat_i) = find_best_granularity(problem, &all_groups[i], &empty_resident);
            let (_, _, lat_i1) = find_best_granularity(problem, &all_groups[i + 1], &empty_resident);
            if lat_merged < lat_i + lat_i1 - 1e-6 {
                all_groups[i] = merged;
                all_groups.remove(i + 1);
                merge_improved = true;
            } else {
                i += 1;
            }
        }
    }

    // Now compute final granularities, latencies, and retain decisions
    // Pass through all groups with resident tensor tracking
    for (idx, group) in all_groups.iter().enumerate() {
        let (gran, traversal, _lat) =
            find_best_granularity(problem, group, &resident_tensors);

        // Compute actual latency with resident tensors
        let lat = compute_latency(
            problem,
            group,
            gran,
            traversal.is_some(),
            &resident_tensors,
        );

        // Decide what to retain for next subgraph
        let next_ops = all_groups.get(idx + 1).map(|g| g.as_slice());
        let next_gran = if let Some(nops) = next_ops {
            let (ng, _, _) = find_best_granularity(problem, nops, &HashSet::new());
            Some(ng)
        } else {
            None
        };

        let retain = decide_retain(problem, group, gran, next_ops, next_gran, &graph);

        // Update resident tensors for next subgraph
        // Remove all tensors that are no longer resident (evicted)
        // Add newly retained tensors
        let dummy_producer: HashMap<usize, usize> = HashMap::new();
        let (_, final_outputs) = classify_boundary(problem, group, &dummy_producer);
        for &t in &final_outputs {
            resident_tensors.remove(&t);
        }
        // Also remove tensors used as inputs that weren't retained
        for &t in &problem.inputs.iter().flatten().cloned().collect::<HashSet<usize>>() {
            if !retain.contains(&t) {
                resident_tensors.remove(&t);
            }
        }
        for &t in &retain {
            resident_tensors.insert(t);
        }

        subgraphs.push(group.clone());
        granularities.push(gran);
        tensors_to_retain.push(retain);
        traversal_orders.push(traversal);
        subgraph_latencies.push(lat);
    }

    Ok(Solution {
        subgraphs,
        granularities,
        tensors_to_retain,
        traversal_orders,
        subgraph_latencies,
    })
}
