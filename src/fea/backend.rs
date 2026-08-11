//! The seam every linear solve of a run goes through.
//!
//! [`LinearSolver`] owns whatever state the configured backend needs for the
//! whole of one problem - nothing for the CPU, a device and its buffers for the
//! GPU - and hands out a [`BoundSolver`] for one design. Binding is separate
//! from solving because the two have different rates: the density derived half
//! of the operator changes once per optimization iteration, while the right hand
//! side changes once per load case, and a GPU backend that re-uploaded the
//! design per load case would spend more on the bus than it saves in the solve.
//!
//! The CPU path is the reference. It calls [`solver::solve`] with exactly the
//! arguments it was called with before this seam existed, so the recorded
//! optimization trajectories stay bit-for-bit reproducible.

use anyhow::Result;

use crate::config::SolverBackend;
use crate::fea::{Solve, SolveLimits, StiffnessOperator, solver};
use crate::problem::Problem;

/// Where one problem's linear solves run.
#[derive(Debug)]
pub struct LinearSolver {
    backend: Backend,
    fallbacks: usize,
}

#[derive(Debug)]
enum Backend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(Box<crate::fea::gpu::GpuSolver>),
}

impl LinearSolver {
    /// The CPU conjugate gradient, whatever the configuration says.
    ///
    /// Used where the backend is not a choice: the finite difference gradient
    /// checks, and anything that has to reproduce a recorded trajectory.
    pub fn cpu() -> LinearSolver {
        LinearSolver {
            backend: Backend::Cpu,
            fallbacks: 0,
        }
    }

    /// Build the backend `problem` selected.
    ///
    /// Errors when the configuration **asked for** a device this machine or
    /// this build cannot provide; the message says how to fall back to the CPU.
    /// A backend nobody asked for by name - the default - falls back to the CPU
    /// instead of failing, and says so once: a machine with no adapter is a
    /// machine to run on, not a configuration error.
    pub fn new(problem: &Problem) -> Result<LinearSolver> {
        let wanted = problem.solver.backend;
        match LinearSolver::for_backend(wanted, &problem.grid, &problem.fixed) {
            Ok(solver) => Ok(solver),
            Err(error) if !problem.solver.explicit && wanted != SolverBackend::Cpu => {
                report_fallback(&error);
                LinearSolver::for_backend(SolverBackend::Cpu, &problem.grid, &problem.fixed)
            }
            Err(error) => Err(error),
        }
    }

    /// Build a named backend for a grid and its constraint mask.
    pub fn for_backend(
        backend: SolverBackend,
        grid: &crate::grid::Grid,
        fixed: &[bool],
    ) -> Result<LinearSolver> {
        let backend = match backend {
            SolverBackend::Cpu => Backend::Cpu,
            #[cfg(feature = "gpu")]
            SolverBackend::Gpu => {
                Backend::Gpu(Box::new(crate::fea::gpu::GpuSolver::new(grid, fixed)?))
            }
            // Unreachable through a parsed configuration, which rejects the
            // backend while it is resolved; reachable from a library caller.
            #[cfg(not(feature = "gpu"))]
            SolverBackend::Gpu => {
                let _ = (grid, fixed);
                anyhow::bail!(
                    "this build of growforge 3D was compiled without the `gpu` feature, so it has \
                     no compute backend; rebuild with the default features (cargo build \
                     --release), or set [solver] backend = \"{}\"",
                    SolverBackend::Cpu.label()
                )
            }
        };
        Ok(LinearSolver {
            backend,
            fallbacks: 0,
        })
    }

    /// Which backend this is.
    pub fn kind(&self) -> SolverBackend {
        match &self.backend {
            Backend::Cpu => SolverBackend::Cpu,
            #[cfg(feature = "gpu")]
            Backend::Gpu(_) => SolverBackend::Gpu,
        }
    }

    /// Human readable description of what the solves run on.
    pub fn description(&self) -> String {
        match &self.backend {
            Backend::Cpu => format!("cpu, {} rayon threads", rayon::current_num_threads()),
            #[cfg(feature = "gpu")]
            Backend::Gpu(gpu) => format!("gpu, {}", gpu.description()),
        }
    }

    /// How many solves this solver has had to finish on the CPU because single
    /// precision could not resolve the system it was handed.
    ///
    /// Zero for a CPU backend, and zero for a GPU backend on every design the
    /// device can actually solve. A non-zero count is the machine-readable half
    /// of the notice [`BoundSolver::solve`] prints: the run's answers are still
    /// the ones its tolerance asked for, and that many of them cost CPU time.
    pub fn cpu_fallbacks(&self) -> usize {
        self.fallbacks
    }

    /// Bind the operator of one design, ready for its load cases.
    pub fn bind<'a>(
        &'a mut self,
        operator: &'a StiffnessOperator<'a>,
        diagonal: &'a [f64],
        fixed: &'a [bool],
    ) -> Result<BoundSolver<'a>> {
        match &mut self.backend {
            Backend::Cpu => {}
            #[cfg(feature = "gpu")]
            Backend::Gpu(gpu) => gpu.bind(operator, diagonal, fixed)?,
        }
        Ok(BoundSolver {
            solver: self,
            operator,
            diagonal,
            fixed,
        })
    }
}

/// One design's operator, ready to be solved against any number of right hand
/// sides.
pub struct BoundSolver<'a> {
    solver: &'a mut LinearSolver,
    operator: &'a StiffnessOperator<'a>,
    diagonal: &'a [f64],
    fixed: &'a [bool],
}

impl BoundSolver<'_> {
    /// Solve `K u = f` on the free subspace.
    ///
    /// `x` carries the initial guess in and the solution out, whichever backend
    /// runs, so warm starting works the same either way.
    ///
    /// `limits` is where the cancellation probe enters the solver: both
    /// backends honour it, at their own natural checkpoint, and both answer a
    /// stop with [`Solve::Cancelled`] rather than an error. A caller that
    /// passes [`SolveLimits::new`] - which is every command line path - is
    /// running the arithmetic it always ran.
    pub fn solve(&mut self, b: &[f64], x: &mut [f64], limits: SolveLimits<'_>) -> Result<Solve> {
        match &mut self.solver.backend {
            Backend::Cpu => {}
            // A device that could not resolve this system in single precision
            // has not failed the caller: it has said which of the two backends
            // has to do this one. Falling through to the CPU here rather than
            // returning the error is what keeps a run alive across the handful
            // of optimization iterations whose design has run past what f32 can
            // describe. Every other error - a lost device, a buffer that would
            // not map - is still an error.
            #[cfg(feature = "gpu")]
            Backend::Gpu(gpu) => match gpu.solve(self.operator, self.fixed, b, x, limits) {
                Err(error)
                    if error
                        .downcast_ref::<crate::fea::gpu::SinglePrecisionLimit>()
                        .is_some() =>
                {
                    report_single_precision_fallback(&error);
                    self.solver.fallbacks += 1;
                }
                other => return other,
            },
        }
        self.on_the_cpu(b, x, limits)
    }

    /// The reference solve, on the operator this binding holds.
    fn on_the_cpu(&self, b: &[f64], x: &mut [f64], limits: SolveLimits<'_>) -> Result<Solve> {
        solver::solve(
            |v, out| self.operator.apply(v, out),
            self.diagonal,
            self.fixed,
            b,
            x,
            limits,
        )
    }
}

/// Say that a solve the compute device could not finish was finished on the
/// CPU, and why.
///
/// Never silent, and never once per process: a design whose stiffness contrast
/// has run past what single precision can resolve is a property of *that
/// iteration*, the run is paying CPU prices for it, and which iterations those
/// were is exactly what the user needs to see to decide whether to set
/// `[solver] backend = "cpu"` and stop the device trying. The advice is printed
/// with the first one only, because it does not change.
#[cfg(feature = "gpu")]
fn report_single_precision_fallback(error: &anyhow::Error) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        println!("solver         again on the CPU ({error})");
        return;
    }
    println!(
        "solver         this solve was finished on the CPU ({error}). The answer meets the same \
         tolerance the GPU one would have, at CPU speed; set [solver] backend = \"cpu\" to stop \
         the device trying."
    );
}

/// Say once per process that the default backend could not be opened and what
/// is running instead.
///
/// Printed here because here is the only place that knows: whether an adapter
/// exists is not knowable while a configuration is being validated, and a run
/// that silently used a different backend from the one its summary named would
/// be the one thing a report may not do. Once per process, because a run opens a
/// solver for its optimization and another for its stress report, and the notice
/// is about the machine rather than about either of them.
fn report_fallback(error: &anyhow::Error) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    println!(
        "solver         {} ({error:#})",
        crate::config::SolverFallback::NoAdapter.note()
    );
}

/// Build a GPU solver for a test, or explain why the test is being skipped.
///
/// Machines and continuous integration runners without a compute adapter must
/// still pass the suite, and a silently skipped test is worse than no test, so
/// the reason is printed.
#[cfg(test)]
pub(crate) fn gpu_or_skip(
    what: &str,
    grid: &crate::grid::Grid,
    fixed: &[bool],
) -> Option<LinearSolver> {
    if !cfg!(feature = "gpu") {
        println!("skipping {what}: this build has no `gpu` feature");
        return None;
    }
    match LinearSolver::for_backend(SolverBackend::Gpu, grid, fixed) {
        Ok(solver) => {
            println!("{what}: {}", solver.description());
            Some(solver)
        }
        Err(error) => {
            println!("skipping {what}: {error:#}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{
        CG_MAX_ITERATIONS, CG_RELATIVE_TOLERANCE, DOF_PER_NODE, SIMP_EMIN_FRACTION, SIMP_PENALTY,
    };
    use crate::fea::CgSolution;
    use crate::fea::hex8_stiffness;
    use crate::geometry::Aabb;
    use crate::grid::{CellKind, Grid};
    use std::cell::Cell;

    /// The limits the comparisons below solve to.
    fn limits() -> SolveLimits<'static> {
        SolveLimits::new(CG_RELATIVE_TOLERANCE, CG_MAX_ITERATIONS)
    }

    fn solid_grid() -> Grid {
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [2.0, 1.0, 1.0],
            },
            1.0,
        );
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Solid;
        }
        grid
    }

    #[test]
    fn the_cpu_backend_reproduces_the_bare_solver() {
        let grid = solid_grid();
        let ke0 = hex8_stiffness(0.3, grid.h);
        let moduli = vec![2300.0; grid.n_cells()];
        let operator = StiffnessOperator::new(&grid, &ke0, &moduli);
        let diagonal = operator.diagonal();
        let mut fixed = vec![false; grid.n_dof()];
        for j in 0..grid.nny() {
            for k in 0..grid.nnz() {
                let node = grid.node_index(0, j, k);
                for d in 0..crate::constants::DOF_PER_NODE {
                    fixed[node * crate::constants::DOF_PER_NODE + d] = true;
                }
            }
        }
        let mut forces = vec![0.0; grid.n_dof()];
        forces[grid.node_index(2, 1, 1) * crate::constants::DOF_PER_NODE + 2] = -10.0;

        let mut reference = vec![0.0; grid.n_dof()];
        let direct = solver::solve(
            |v, out| operator.apply(v, out),
            &diagonal,
            &fixed,
            &forces,
            &mut reference,
            limits(),
        )
        .expect("solve")
        .converged()
        .expect("converged");

        let mut solver = LinearSolver::cpu();
        assert_eq!(solver.kind(), SolverBackend::Cpu);
        assert!(solver.description().starts_with("cpu"));
        let mut through_seam = vec![0.0; grid.n_dof()];
        let bound = solver
            .bind(&operator, &diagonal, &fixed)
            .expect("bind")
            .solve(&forces, &mut through_seam, limits())
            .expect("solve")
            .converged()
            .expect("converged");

        // The seam may not perturb a single bit of the CPU path.
        assert_eq!(bound.iterations, direct.iterations);
        assert_eq!(bound.relative_residual, direct.relative_residual);
        assert_eq!(through_seam, reference);
    }

    /// The default backend is a preference and a named one is an instruction.
    ///
    /// Opening the default must never fail for want of a device: a machine with
    /// no adapter is a machine to run on. Opening one the configuration named
    /// still fails, which is the contract `growforge check` reports against.
    #[test]
    fn the_default_backend_falls_back_and_a_named_one_does_not() {
        let grid = solid_grid();
        let fixed = vec![false; grid.n_dof()];
        let problem = |backend, explicit| crate::problem::Problem {
            name: "fallback".to_string(),
            engine: crate::constants::DEFAULT_ENGINE.to_string(),
            grid: grid.clone(),
            material: crate::config::Material {
                youngs_modulus_mpa: 2300.0,
                poisson_ratio: 0.36,
                density_g_cm3: 1.24,
                yield_strength_mpa: None,
            },
            optimization: crate::config::OptimizationParams {
                mass_fraction: 0.4,
                min_feature_mm: 4.0,
                filter_radius_mm: 2.0,
                penalty: crate::constants::SIMP_PENALTY,
                stiffness_floor: crate::constants::SIMP_EMIN_FRACTION,
                max_iterations: 1,
                convergence_tol: 1e-3,
                update: crate::config::UpdateScheme::Oc,
                overhang: None,
                wireframe: None,
                local_volume: None,
            },
            growth: None,
            solver: crate::config::SolverParams {
                backend,
                explicit,
                fell_back: None,
                tolerance: crate::constants::CG_RELATIVE_TOLERANCE,
            },
            output: crate::config::OutputParams {
                stl_path: std::path::PathBuf::from("fallback.stl"),
                iso_level: 0.5,
                smoothing_iterations: 0,
                supersample: 1,
                voids: crate::config::VoidPolicy::Warn,
                islands: crate::config::IslandPolicy::Cull,
                boundaries: crate::config::BoundaryFidelity::Exact,
                trim: crate::config::TrimPolicy::Off,
                trim_stress_fraction: crate::constants::TRIM_STRESS_FRACTION,
                reinforce: crate::config::ReinforcePolicy::Off,
                reinforce_thickness_mm: 4.0,
                flush: crate::config::FlushPolicy::Off,
                flush_depth_mm: None,
                stress_json: None,
            },
            // This fixture never exports, so it needs no analytic boundaries.
            boundaries: crate::geometry::Boundaries::default(),
            fixed: fixed.clone(),
            load_cases: Vec::new(),
            supports: Vec::new(),
            keepins: Vec::new(),
            counts: crate::grid::CellCounts::default(),
            warnings: Vec::new(),
        };

        // Whatever this machine has, the default opens something.
        let soft = LinearSolver::new(&problem(SolverBackend::Gpu, false))
            .expect("the default backend must never fail for want of a device");
        println!("the default resolved to {}", soft.description());
        assert!(
            cfg!(feature = "gpu") || soft.kind() == SolverBackend::Cpu,
            "a build with no compute backend must have fallen back"
        );

        // Named, it is the one that runs - or the one that fails.
        let named = LinearSolver::new(&problem(SolverBackend::Gpu, true));
        if cfg!(feature = "gpu") {
            if let Ok(solver) = named {
                assert_eq!(solver.kind(), SolverBackend::Gpu);
            }
        } else {
            let error = named
                .expect_err("a named backend this build cannot provide must fail")
                .to_string();
            assert!(error.contains("gpu"), "unexpected error: {error}");
        }

        // And a named CPU backend is the CPU backend, fallback or not.
        let cpu = LinearSolver::new(&problem(SolverBackend::Cpu, true)).expect("cpu");
        assert_eq!(cpu.kind(), SolverBackend::Cpu);
    }

    /// A cantilever bar clamped on its `x = 0` face and pulled down at the far
    /// end, with the moduli a caller chooses.
    struct Fixture {
        grid: Grid,
        ke0: crate::fea::Ke,
        fixed: Vec<bool>,
        forces: Vec<f64>,
    }

    fn fixture() -> Fixture {
        cantilever(2.0, [16.0, 8.0, 8.0])
    }

    /// The fixture above at a chosen voxel size and extent.
    fn cantilever(h: f64, extent: [f64; 3]) -> Fixture {
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: extent,
            },
            h,
        );
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Design;
        }
        let mut fixed = vec![false; grid.n_dof()];
        for j in 0..grid.nny() {
            for k in 0..grid.nnz() {
                let node = grid.node_index(0, j, k);
                for d in 0..DOF_PER_NODE {
                    fixed[node * DOF_PER_NODE + d] = true;
                }
            }
        }
        let mut forces = vec![0.0; grid.n_dof()];
        for j in 0..grid.nny() {
            for k in 0..grid.nnz() {
                let node = grid.node_index(grid.nx, j, k);
                forces[node * DOF_PER_NODE + 2] = -50.0 / (grid.nny() * grid.nnz()) as f64;
            }
        }
        let ke0 = hex8_stiffness(0.36, h);
        Fixture {
            grid,
            ke0,
            fixed,
            forces,
        }
    }

    /// Solve `fixture` on both backends and return the two displacement fields.
    ///
    /// Both are taken to [`CG_RELATIVE_TOLERANCE`], so a GPU field that comes
    /// back at all satisfies the criterion the CPU one does and the comparison
    /// is of two answers to the same question.
    fn both_backends(
        what: &str,
        fixture: &Fixture,
        moduli: &[f64],
    ) -> Option<(Vec<f64>, Vec<f64>, CgSolution, usize)> {
        let mut gpu = gpu_or_skip(what, &fixture.grid, &fixture.fixed)?;
        let operator = StiffnessOperator::new(&fixture.grid, &fixture.ke0, moduli);
        let diagonal = operator.diagonal();

        let mut reference = vec![0.0; fixture.grid.n_dof()];
        LinearSolver::cpu()
            .bind(&operator, &diagonal, &fixture.fixed)
            .expect("bind")
            .solve(&fixture.forces, &mut reference, limits())
            .unwrap_or_else(|e| panic!("{what}: cpu solve: {e:#}"))
            .converged()
            .expect("converged");

        let mut candidate = vec![0.0; fixture.grid.n_dof()];
        let solution = gpu
            .bind(&operator, &diagonal, &fixture.fixed)
            .expect("bind")
            .solve(&fixture.forces, &mut candidate, limits())
            .unwrap_or_else(|e| panic!("{what}: gpu solve: {e:#}"))
            .converged()
            .expect("converged");
        Some((reference, candidate, solution, gpu.cpu_fallbacks()))
    }

    /// SIMP moduli of a density field, so a fixture carries the same 1e9
    /// stiffness contrast a real late optimization iteration does.
    ///
    /// Pinned to the default floor rather than taking `[optimization]
    /// stiffness_floor`: what these fixtures are for is the *worst* contrast a
    /// configuration can hand the backends, and that is the default one. A
    /// higher floor is a strictly easier system, and measuring the backends on
    /// one would be measuring nothing.
    fn simp_moduli(densities: &[f64], e0: f64) -> Vec<f64> {
        let emin = SIMP_EMIN_FRACTION * e0;
        densities
            .iter()
            .map(|d| emin + d.powf(SIMP_PENALTY) * (e0 - emin))
            .collect()
    }

    /// Relative L2 difference of two displacement fields.
    fn relative_l2(a: &[f64], b: &[f64]) -> f64 {
        let difference: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        let scale = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        difference / scale.max(f64::MIN_POSITIVE)
    }

    #[test]
    fn gpu_displacements_match_the_cpu_reference() {
        let what = "gpu_displacements_match_the_cpu_reference";
        let fixture = fixture();
        let Some(mut gpu) = gpu_or_skip(what, &fixture.grid, &fixture.fixed) else {
            return;
        };
        let mut cpu = LinearSolver::cpu();
        let e0 = 2300.0;
        let cells = fixture.grid.n_cells();

        // Three designs: a solid block, a smoothly graded field, and a nearly
        // binary one whose void cells sit at the SIMP stiffness floor. The last
        // is what a late optimization iteration hands the solver, and the only
        // one where single precision has anything to lose.
        let designs: [(&str, Vec<f64>); 3] = [
            ("solid", vec![1.0; cells]),
            (
                "graded",
                (0..cells)
                    .map(|e| 0.15 + 0.85 * ((e % 11) as f64 / 10.0))
                    .collect(),
            ),
            (
                "late simp",
                (0..cells)
                    .map(|e| {
                        let (i, j, k) = fixture.grid.cell_ijk(e);
                        // A diagonal truss through an otherwise empty block.
                        if k == j || i.is_multiple_of(3) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            ),
        ];

        for (name, densities) in designs {
            let moduli = simp_moduli(&densities, e0);
            let contrast = moduli.iter().fold(0.0f64, |m, v| m.max(*v))
                / moduli.iter().fold(f64::MAX, |m, v| m.min(*v));
            let operator = StiffnessOperator::new(&fixture.grid, &fixture.ke0, &moduli);
            let diagonal = operator.diagonal();

            let mut reference = vec![0.0; fixture.grid.n_dof()];
            cpu.bind(&operator, &diagonal, &fixture.fixed)
                .expect("bind")
                .solve(&fixture.forces, &mut reference, limits())
                .unwrap_or_else(|e| panic!("{name}: cpu solve: {e:#}"));

            let mut candidate = vec![0.0; fixture.grid.n_dof()];
            let solution = gpu
                .bind(&operator, &diagonal, &fixture.fixed)
                .expect("bind")
                .solve(&fixture.forces, &mut candidate, limits())
                .unwrap_or_else(|e| panic!("{name}: gpu solve: {e:#}"))
                .converged()
                .expect("converged");
            assert!(
                solution.relative_residual < CG_RELATIVE_TOLERANCE,
                "{name}: the GPU solve returned without meeting its own tolerance"
            );

            // Both solutions satisfy the same f64 residual criterion, so they
            // may only differ by what that residual leaves undetermined:
            // `2 * tolerance * cond(K)` in the worst case, which for a 1e9
            // contrast is no bound at all. The gate therefore stays at the
            // portable 1e-4 rather than at what one machine measures - on the
            // development machine these three come out at 3.0e-11, 4.5e-11 and
            // 1.4e-14 - and the measurement is printed so a regression that
            // stays inside the gate is still visible.
            let difference = relative_l2(&reference, &candidate);
            println!(
                "{name}: stiffness contrast {contrast:.2e}, {} gpu iterations, relative L2 \
                 difference {difference:.3e}",
                solution.iterations
            );
            assert!(
                difference < 1e-4,
                "{name}: relative L2 difference {difference:.3e} between the CPU and GPU \
                 displacement fields"
            );
        }
    }

    #[test]
    fn a_zero_right_hand_side_solves_to_zero_on_the_gpu() {
        let what = "a_zero_right_hand_side_solves_to_zero_on_the_gpu";
        let fixture = fixture();
        let Some(mut gpu) = gpu_or_skip(what, &fixture.grid, &fixture.fixed) else {
            return;
        };
        let moduli = simp_moduli(&vec![1.0; fixture.grid.n_cells()], 2300.0);
        let operator = StiffnessOperator::new(&fixture.grid, &fixture.ke0, &moduli);
        let diagonal = operator.diagonal();
        let mut x = vec![1.0; fixture.grid.n_dof()];
        let solution = gpu
            .bind(&operator, &diagonal, &fixture.fixed)
            .expect("bind")
            .solve(&vec![0.0; fixture.grid.n_dof()], &mut x, limits())
            .expect("solve")
            .converged()
            .expect("converged");
        assert_eq!(solution.iterations, 0);
        assert!(x.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn a_gpu_warm_start_at_the_solution_costs_no_iterations() {
        let what = "a_gpu_warm_start_at_the_solution_costs_no_iterations";
        let fixture = fixture();
        let Some(mut gpu) = gpu_or_skip(what, &fixture.grid, &fixture.fixed) else {
            return;
        };
        let moduli = simp_moduli(&vec![0.6; fixture.grid.n_cells()], 2300.0);
        let operator = StiffnessOperator::new(&fixture.grid, &fixture.ke0, &moduli);
        let diagonal = operator.diagonal();

        let mut x = vec![0.0; fixture.grid.n_dof()];
        LinearSolver::cpu()
            .bind(&operator, &diagonal, &fixture.fixed)
            .expect("bind")
            .solve(&fixture.forces, &mut x, limits())
            .expect("solve");

        let solution = gpu
            .bind(&operator, &diagonal, &fixture.fixed)
            .expect("bind")
            .solve(&fixture.forces, &mut x, limits())
            .expect("solve")
            .converged()
            .expect("converged");
        assert_eq!(
            solution.iterations, 0,
            "a warm start that already meets the tolerance must not run the device"
        );
    }

    /// The cancellation seam reaches both backends through the one entry point
    /// every solve of a run goes through, and a stop there is an outcome rather
    /// than an error.
    #[test]
    fn a_stop_reaches_whichever_backend_is_running() {
        let fixture = fixture();
        let moduli = simp_moduli(&vec![0.4; fixture.grid.n_cells()], 2300.0);
        let operator = StiffnessOperator::new(&fixture.grid, &fixture.ke0, &moduli);
        let diagonal = operator.diagonal();

        let mut backends = vec![LinearSolver::cpu()];
        if let Some(gpu) = gpu_or_skip(
            "a_stop_reaches_whichever_backend_is_running",
            &fixture.grid,
            &fixture.fixed,
        ) {
            backends.push(gpu);
        }
        // A probe that says stop from the outset: whichever backend this is, the
        // first checkpoint it reaches is the last thing it does.
        let stop = || true;
        for mut solver in backends {
            let kind = solver.kind();
            let mut x = vec![0.0; fixture.grid.n_dof()];
            let outcome = solver
                .bind(&operator, &diagonal, &fixture.fixed)
                .expect("bind")
                .solve(&fixture.forces, &mut x, limits().watching(&stop))
                .unwrap_or_else(|e| panic!("{kind:?}: a stop must not be an error: {e:#}"));
            assert_eq!(outcome, Solve::Cancelled, "{kind:?}");
            assert!(
                outcome.converged().is_none(),
                "{kind:?}: a cancelled solve has no statistics to report"
            );
        }
    }

    /// A stop that lands *inside* a device solve, after an earlier refinement
    /// pass has already folded its correction into `x`.
    ///
    /// The guarantee is that the half-finished correction is dropped rather than
    /// folded: `x` has to come back bit-identical to what the last completed
    /// pass left. A probe that says stop from the outset never reaches this - the
    /// pass loop catches it before any pass runs - so the stop has to be placed,
    /// and the placing is what most of this test is.
    #[test]
    fn a_stop_inside_a_refinement_pass_drops_the_correction_it_was_building() {
        let what = "a_stop_inside_a_refinement_pass_drops_the_correction_it_was_building";
        let fixture = fixture();
        let Some(mut gpu) = gpu_or_skip(what, &fixture.grid, &fixture.fixed) else {
            return;
        };
        // A nearly binary design, so the 1e9 stiffness contrast forces the
        // refinement loop to take more than one pass to reach 1e-8.
        let densities: Vec<f64> = (0..fixture.grid.n_cells())
            .map(|e| {
                let (i, j, k) = fixture.grid.cell_ijk(e);
                if k == j || i.is_multiple_of(3) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let moduli = simp_moduli(&densities, 2300.0);
        let operator = StiffnessOperator::new(&fixture.grid, &fixture.ke0, &moduli);
        let diagonal = operator.diagonal();

        // Stop at the `n`th question the solve asks, whichever asks it: the pass
        // loop asks once per pass, the inner solve once per readback batch.
        let solve_stopping_at = |gpu: &mut LinearSolver, n: usize| {
            let asked = Cell::new(0usize);
            let stop = || {
                asked.set(asked.get() + 1);
                asked.get() >= n
            };
            let mut x = vec![0.0; fixture.grid.n_dof()];
            let outcome = gpu
                .bind(&operator, &diagonal, &fixture.fixed)
                .expect("bind")
                .solve(&fixture.forces, &mut x, limits().watching(&stop))
                .expect("a stop is not an error");
            (outcome, x)
        };

        // The first question whose answer leaves a correction behind is the pass
        // loop's own, at the top of the second pass: everything before it is
        // inside the first pass, which folds nothing until it is done.
        let zero = vec![0.0; fixture.grid.n_dof()];
        let mut boundary = None;
        for n in 1..=200 {
            let (outcome, x) = solve_stopping_at(&mut gpu, n);
            if outcome == Solve::Cancelled && x != zero {
                boundary = Some((n, x));
                break;
            }
        }
        let Some((second_pass, after_first)) = boundary else {
            println!("skipping {what}: this device solves the fixture in one pass");
            return;
        };
        println!("{what}: the second pass begins at question {second_pass}");

        // One question later is the first readback batch of that second pass:
        // the stop now lands inside the device solve, part way through a Krylov
        // recurrence, and the correction it was building is dropped.
        let (outcome, x) = solve_stopping_at(&mut gpu, second_pass + 1);
        assert_eq!(outcome, Solve::Cancelled);
        assert_eq!(
            x, after_first,
            "a stop inside a refinement pass folded a half-built correction into the solution"
        );
        assert_ne!(x, zero, "the first pass must have folded something");
    }

    /// The design fraction the skeletal regime the tests below stand for runs
    /// at, and the one the configurations they are minimized from use.
    const SKELETAL_MASS_FRACTION: f64 = 0.10;

    /// A cantilever at that mass fraction, with the forced solid load pad a
    /// keepin region puts under the load.
    ///
    /// Eighty cells long and ten square at one voxel per cell, which is the
    /// smallest thing that still reproduces what the shipped configuration does.
    /// Slenderness rather than size is what does it: a cantilever's deflection
    /// grows with the cube of its length, so a long thin bar at a tenth of the
    /// solid modulus has a solution enormous against its own right hand side,
    /// and it is the rounding of that solution - accumulated over the thousands
    /// of device steps such a body needs - that used to swamp the correction.
    /// Measured here before the fix: 1700 device iterations leaving a relative
    /// residual of 1.2e1 against the 1.0 the pass started from.
    fn skeletal_fixture() -> (Fixture, Vec<f64>) {
        let mut fixture = cantilever(1.0, [80.0, 10.0, 10.0]);
        let grid = &fixture.grid;
        let pad_i = grid.nx - 4;
        let pad_j = (grid.ny / 2 - 2)..(grid.ny / 2 + 3);
        let pad_k = (grid.nz / 2 - 2)..(grid.nz / 2 + 3);
        let densities: Vec<f64> = (0..grid.n_cells())
            .map(|e| {
                let (i, j, k) = grid.cell_ijk(e);
                if i >= pad_i && pad_j.contains(&j) && pad_k.contains(&k) {
                    1.0
                } else {
                    SKELETAL_MASS_FRACTION
                }
            })
            .collect();

        // The load goes on the pad rather than over the whole end face, which
        // is what puts the 1e3 stiffness step in the middle of the load path.
        let nodes: Vec<usize> = pad_j
            .clone()
            .flat_map(|j| pad_k.clone().map(move |k| (j, k)))
            .map(|(j, k)| grid.node_index(grid.nx, j, k))
            .collect();
        let share = -50.0 / nodes.len() as f64;
        fixture.forces.fill(0.0);
        for node in nodes {
            fixture.forces[node * DOF_PER_NODE + 2] = share;
        }
        let moduli = simp_moduli(&densities, 2300.0);
        (fixture, moduli)
    }

    /// The same shape with its whole design domain driven to the SIMP density
    /// floor, leaving only the forced solid load pad with any stiffness.
    ///
    /// That is what an aggressive first MMA step produces at a skeletal mass
    /// fraction: measured on the 58k cell configuration this is minimized from,
    /// a third of the cells sit at exactly [`SIMP_EMIN_FRACTION`] and 0.6 % of
    /// them - the pad - carry more than a twentieth of the solid modulus. Whole
    /// regions are then held together by nothing but the stiffness floor, and
    /// their near rigid body modes put the condition number of the system nine
    /// decades above the structure's own.
    fn floor_fixture() -> (Fixture, Vec<f64>) {
        let fixture = cantilever(1.0, [24.0, 8.0, 8.0]);
        let grid = &fixture.grid;
        let pad_i = grid.nx - 4;
        let pad_j = (grid.ny / 2 - 2)..(grid.ny / 2 + 2);
        let pad_k = (grid.nz / 2 - 2)..(grid.nz / 2 + 2);
        let densities: Vec<f64> = (0..grid.n_cells())
            .map(|e| {
                let (i, j, k) = grid.cell_ijk(e);
                if i >= pad_i && pad_j.contains(&j) && pad_k.contains(&k) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let moduli = simp_moduli(&densities, 2300.0);
        (fixture, moduli)
    }

    /// The GPU solve of a skeletal design, which is where the device's
    /// correction used to come back worse than no correction at all.
    ///
    /// Before the solution accumulator carried a compensation limb this solve
    /// failed outright, with the refinement loop reporting a relative residual
    /// of 1.388 - above the 1.0 the zero start gave it - while the device's own
    /// recursive residual read 6.7e-6. The device was not stalling at a
    /// precision floor; it was handing back a correction whose f32 rounding,
    /// accumulated over the thousands of steps such a compliant body needs, was
    /// larger than the answer it was correcting. So the fallback count is
    /// asserted as well as the answer: this regime has to be solved *on the
    /// device*, not rescued.
    #[test]
    fn a_skeletal_design_solves_on_the_gpu_to_the_same_answer_as_the_cpu() {
        let what = "a_skeletal_design_solves_on_the_gpu_to_the_same_answer_as_the_cpu";
        let (fixture, moduli) = skeletal_fixture();
        let Some((reference, candidate, solution, fallbacks)) =
            both_backends(what, &fixture, &moduli)
        else {
            return;
        };
        let difference = relative_l2(&reference, &candidate);
        println!(
            "{what}: {} cells, {} gpu iterations, relative residual {:.3e}, relative L2 \
             difference {difference:.3e}",
            fixture.grid.n_cells(),
            solution.iterations,
            solution.relative_residual,
        );
        assert_eq!(
            fallbacks, 0,
            "the device gave this solve up and the CPU finished it; the skeletal regime is \
             inside what single precision can do and has to be solved on the device"
        );
        assert!(
            solution.relative_residual < CG_RELATIVE_TOLERANCE,
            "the GPU solve returned without meeting its own tolerance"
        );
        assert!(
            difference < 1e-4,
            "relative L2 difference {difference:.3e} between the CPU and GPU displacement fields"
        );
    }

    /// A design at the density floor is past what single precision can do - the
    /// exact solution *rounded to f32* leaves a residual larger than the right
    /// hand side it was solving against - so the device is expected to give up
    /// on it. What is gated here is that giving up is not failing: the solve
    /// still comes back converged to the caller's tolerance, with the same
    /// answer the CPU backend would have produced.
    ///
    /// A device that does solve it passes too. The gate is on the answer, which
    /// is the thing that has to hold everywhere; which side of the seam produced
    /// it is a property of this machine's single precision, and is printed.
    #[test]
    fn a_design_at_the_density_floor_still_returns_a_converged_answer() {
        let what = "a_design_at_the_density_floor_still_returns_a_converged_answer";
        let (fixture, moduli) = floor_fixture();
        let Some((reference, candidate, solution, fallbacks)) =
            both_backends(what, &fixture, &moduli)
        else {
            return;
        };
        let difference = relative_l2(&reference, &candidate);
        println!(
            "{what}: {} cells, {} iterations, {fallbacks} cpu fallback(s), relative residual \
             {:.3e}, relative L2 difference {difference:.3e}",
            fixture.grid.n_cells(),
            solution.iterations,
            solution.relative_residual,
        );
        assert!(
            solution.relative_residual < CG_RELATIVE_TOLERANCE,
            "the solve returned without meeting the caller's tolerance"
        );
        assert!(
            difference < 1e-4,
            "relative L2 difference {difference:.3e} between the CPU and GPU displacement fields"
        );
    }
}
