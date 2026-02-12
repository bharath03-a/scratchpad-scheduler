mod problem;
mod solution;
mod scheduler;

use anyhow::{Context, Result};
use std::fs;

use crate::problem::Problem;
use crate::scheduler::scheduler_solution;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: mlsys <problem_file.json> <solution_file.json>");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    // Read the problem file
    let input_text = fs::read_to_string(input_path)
        .with_context(|| format!("Failed to read input file: {}", input_path))?;
    let problem: Problem = serde_json::from_str(&input_text)
        .with_context(|| format!("Failed to parse input file: {}", input_path))?;

    // Build the (temporary empty) solution
    let solution = scheduler_solution(&problem)?;

    // Write the solution to the output file
    let output_text = serde_json::to_string_pretty(&solution)?;
    fs::write(output_path, output_text)
        .with_context(|| format!("Failed to write output file: {}", output_path))?;

    Ok(())
}