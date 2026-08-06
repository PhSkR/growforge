//! `growforge bench`: times the linear solve of a real assembled problem on
//! every backend this build can reach.
//!
//! No synthetic matrix. The operator is the one the first SIMP iteration of the
//! given configuration would build - the starting design pushed through
//! [`crate::engine::simp_moduli`] - and the right hand side is the first load
//! case assembled through [`Problem::case_forces`], self weight included. What
//! is timed is therefore the same work a run spends almost all of its time on.
//!
//! Every backend is taken to the same [`constants::BENCH_CG_TOLERANCE`] from the
//! same cold start, so the ratio of the times is a like for like speedup rather
//! than a comparison of two different amounts of work.

use std::time::Instant;

use anyhow::{Context, Result};

use crate::config::SolverBackend;
use crate::constants;
use crate::engine;
use crate::fea::{LinearSolver, StiffnessOperator, hex8_stiffness};
use crate::grid::CellKind;
use crate::problem::Problem;

/// What one backend achieved on the benchmark problem.
#[derive(Debug, Clone)]
pub struct BackendTiming {
    /// Backend that produced these numbers.
    pub backend: SolverBackend,
    /// Human readable description of the hardware it ran on.
    pub description: String,
    /// Wall clock seconds of the fastest of the timed solves.
    pub best_s: f64,
    /// Mean wall clock seconds over the timed solves.
    pub mean_s: f64,
    /// Iterations the last solve spent.
    pub iterations: usize,
    /// Relative residual the last solve reached.
    pub relative_residual: f64,
    /// True when the solve stopped on the iteration cap rather than the
    /// tolerance, which makes its time incomparable with a converged one.
    pub capped: bool,
}

/// A backend that could not be timed, and why.
#[derive(Debug, Clone)]
pub struct SkippedBackend {
    /// Backend that was skipped.
    pub backend: SolverBackend,
    /// Formatted reason.
    pub reason: String,
}

/// The result of one benchmark.
#[derive(Debug, Clone)]
pub struct BenchReport {
    /// Project name of the benchmarked configuration.
    pub name: String,
    /// Cells in the grid.
    pub n_cells: usize,
    /// Degrees of freedom of the linear system.
    pub n_dof: usize,
    /// Load case that was solved.
    pub load_case: String,
    /// Relative residual every solve was taken to.
    pub tolerance: f64,
    /// Timed solves per backend.
    pub solves: usize,
    /// One entry per backend that ran.
    pub timings: Vec<BackendTiming>,
    /// One entry per backend that could not run here.
    pub skipped: Vec<SkippedBackend>,
}

impl BenchReport {
    /// Speedup of `timing` against the CPU baseline, when both are present.
    pub fn speedup(&self, timing: &BackendTiming) -> Option<f64> {
        let baseline = self
            .timings
            .iter()
            .find(|t| t.backend == SolverBackend::Cpu)?;
        if timing.backend == SolverBackend::Cpu || timing.best_s <= 0.0 {
            return None;
        }
        Some(baseline.best_s / timing.best_s)
    }
}

/// The density field the first optimization iteration would analyse.
fn starting_densities(problem: &Problem) -> Vec<f64> {
    problem
        .grid
        .cells
        .iter()
        .map(|kind| match kind {
            CellKind::Void => constants::DENSITY_MIN,
            CellKind::Solid => constants::DENSITY_MAX,
            CellKind::Design => problem.optimization.mass_fraction,
        })
        .collect()
}

/// Time the configured problem on every backend that is available here.
pub fn run(problem: &Problem) -> Result<BenchReport> {
    let grid = &problem.grid;
    let case = problem
        .load_cases
        .first()
        .context("the problem has no load case to benchmark")?;

    let densities = starting_densities(problem);
    let mut moduli = vec![0.0; grid.n_cells()];
    engine::simp_moduli(
        grid,
        &densities,
        problem.material.youngs_modulus_mpa,
        problem.optimization.penalty,
        &mut moduli,
    );
    let ke0 = hex8_stiffness(problem.material.poisson_ratio, grid.h);
    let operator = StiffnessOperator::new(grid, &ke0, &moduli);
    let diagonal = operator.diagonal();
    let mut forces = vec![0.0; grid.n_dof()];
    problem.case_forces(0, &densities, &mut forces);

    let mut timings = Vec::new();
    let mut skipped = Vec::new();
    for backend in [SolverBackend::Cpu, SolverBackend::Gpu] {
        let mut solver = match LinearSolver::for_backend(backend, grid, &problem.fixed) {
            Ok(solver) => solver,
            Err(error) => {
                skipped.push(SkippedBackend {
                    backend,
                    reason: format!("{error:#}"),
                });
                continue;
            }
        };
        let description = solver.description();
        match time_backend(&mut solver, &operator, &diagonal, &problem.fixed, &forces) {
            Ok(mut timing) => {
                timing.description = description;
                timings.push(timing);
            }
            Err(error) => skipped.push(SkippedBackend {
                backend,
                reason: format!("{error:#}"),
            }),
        }
    }

    Ok(BenchReport {
        name: problem.name.clone(),
        n_cells: grid.n_cells(),
        n_dof: grid.n_dof(),
        load_case: case.name.clone(),
        tolerance: constants::BENCH_CG_TOLERANCE,
        solves: constants::BENCH_SOLVES,
        timings,
        skipped,
    })
}

/// Run [`constants::BENCH_SOLVES`] cold solves and keep the best and the mean.
fn time_backend(
    solver: &mut LinearSolver,
    operator: &StiffnessOperator,
    diagonal: &[f64],
    fixed: &[bool],
    forces: &[f64],
) -> Result<BackendTiming> {
    let backend = solver.kind();
    let mut bound = solver.bind(operator, diagonal, fixed)?;
    let mut x = vec![0.0; forces.len()];
    let mut best = f64::INFINITY;
    let mut total = 0.0;
    let mut iterations = 0;
    let mut relative_residual = 0.0;
    for _ in 0..constants::BENCH_SOLVES {
        // Cold every time: a warm start would time the second solve of a
        // sequence rather than the one the table claims to be about.
        x.fill(0.0);
        let start = Instant::now();
        // Nothing may stop a benchmark part way through: a timing of half a
        // solve is not a timing, so the limits carry no cancellation probe.
        let solution = bound
            .solve(
                forces,
                &mut x,
                crate::fea::SolveLimits::new(
                    constants::BENCH_CG_TOLERANCE,
                    constants::BENCH_CG_MAX_ITERATIONS,
                ),
            )?
            .converged()
            .expect("an unwatched solve is never cancelled");
        let elapsed = start.elapsed().as_secs_f64();
        best = best.min(elapsed);
        total += elapsed;
        iterations = solution.iterations;
        relative_residual = solution.relative_residual;
    }
    Ok(BackendTiming {
        backend,
        description: String::new(),
        best_s: best,
        mean_s: total / constants::BENCH_SOLVES as f64,
        iterations,
        relative_residual,
        capped: iterations >= constants::BENCH_CG_MAX_ITERATIONS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    const TINY: &str = r#"
[project]
name = "bench"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 16.0

[output]
stl_path = "bench.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [32.0, 16.0, 16.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 16.5, 16.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [32.0, 8.0, 8.0], radius = 4.0 }
vector = [0.0, 0.0, -20.0]
"#;

    fn problem() -> Problem {
        let config = Config::parse(TINY).expect("parse");
        Problem::build(&config, &PathBuf::from(".")).expect("build")
    }

    #[test]
    fn the_starting_design_is_the_one_the_first_iteration_would_analyse() {
        let problem = problem();
        let densities = starting_densities(&problem);
        assert_eq!(densities.len(), problem.grid.n_cells());
        for (cell, kind) in problem.grid.cells.iter().enumerate() {
            let expected = match kind {
                CellKind::Void => constants::DENSITY_MIN,
                CellKind::Solid => constants::DENSITY_MAX,
                CellKind::Design => problem.optimization.mass_fraction,
            };
            assert_eq!(densities[cell], expected);
        }
    }

    #[test]
    fn the_benchmark_always_reports_the_cpu_baseline() {
        let problem = problem();
        let report = run(&problem).expect("bench");
        assert_eq!(report.n_dof, problem.grid.n_dof());
        assert_eq!(report.load_case, "tip");
        assert_eq!(report.solves, constants::BENCH_SOLVES);

        let cpu = report
            .timings
            .iter()
            .find(|t| t.backend == SolverBackend::Cpu)
            .expect("a cpu timing");
        assert!(cpu.best_s > 0.0 && cpu.best_s <= cpu.mean_s + 1e-9);
        assert!(cpu.iterations > 0 && !cpu.capped);
        assert!(cpu.relative_residual < constants::BENCH_CG_TOLERANCE);
        assert!(report.speedup(cpu).is_none(), "the baseline has no speedup");

        // Whatever this machine can do, every backend is accounted for.
        assert_eq!(report.timings.len() + report.skipped.len(), 2);
        for skipped in &report.skipped {
            assert!(!skipped.reason.is_empty());
            println!("{:?} skipped: {}", skipped.backend, skipped.reason);
        }
        for timing in &report.timings {
            if timing.backend == SolverBackend::Gpu {
                let speedup = report.speedup(timing).expect("a speedup");
                assert!(speedup.is_finite() && speedup > 0.0);
                assert!(timing.relative_residual < constants::BENCH_CG_TOLERANCE);
            }
        }
    }
}
