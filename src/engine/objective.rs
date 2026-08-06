//! One evaluation of the weighted compliance objective and its sensitivities.
//!
//! Pulled out of the SIMP loop so that the same code path serves the optimizer,
//! the post-run stress analysis and the finite difference gradient check.
//!
//! With a design dependent load the compliance sensitivity is
//!
//! ```text
//!   dC/drho_e = 2 u^T (df/drho_e) - u^T (dK/drho_e) u
//! ```
//!
//! The second term is the classical SIMP one. The first is the self-weight
//! term: it vanishes for force and torque loads, and it is what can push a
//! sensitivity positive. Both are then chained back through the density chain's
//! transposes to reach the design variables.

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::constants;
use crate::engine::chain::DensityChain;
use crate::fea::{
    CancelProbe, Ke, LinearSolver, SolveLimits, StiffnessOperator, gravity, hex8_stiffness,
};
use crate::grid::CellKind;
use crate::problem::Problem;

/// Everything one objective evaluation needs to keep between calls.
#[derive(Debug, Clone)]
pub struct Workspace {
    /// Density-filtered intermediate densities.
    pub filtered: Vec<f64>,
    /// Printed physical densities, the ones the analysis sees.
    pub printed: Vec<f64>,
    /// Effective Young's modulus per cell, in MPa.
    pub moduli: Vec<f64>,
    /// Warm started displacements, one vector per load case.
    pub displacements: Vec<Vec<f64>>,
    energies: Vec<f64>,
    energy_sum: Vec<f64>,
    load_term: Vec<f64>,
    d_printed: Vec<f64>,
    d_filtered: Vec<f64>,
    forces: Vec<f64>,
}

impl Workspace {
    /// Allocate the buffers for `problem`, with cold displacements.
    pub fn new(problem: &Problem) -> Workspace {
        let cells = problem.grid.n_cells();
        let dof = problem.grid.n_dof();
        Workspace {
            filtered: vec![0.0; cells],
            printed: vec![0.0; cells],
            moduli: vec![0.0; cells],
            displacements: problem.load_cases.iter().map(|_| vec![0.0; dof]).collect(),
            energies: vec![0.0; cells],
            energy_sum: vec![0.0; cells],
            load_term: vec![0.0; cells],
            d_printed: vec![0.0; cells],
            d_filtered: vec![0.0; cells],
            forces: vec![0.0; dof],
        }
    }
}

/// Result of one objective evaluation.
#[derive(Debug, Clone)]
pub struct Value {
    /// Weighted compliance in N*mm.
    pub compliance: f64,
    /// Conjugate gradient iterations spent per load case.
    pub cg_iterations: Vec<usize>,
}

/// The weighted compliance objective over a problem and a density chain.
pub struct Objective<'a> {
    problem: &'a Problem,
    chain: &'a DensityChain,
    ke0: Ke,
    e0: f64,
    /// `[optimization] stiffness_floor` times [`Objective::e0`], formed once
    /// here so the sensitivity below differentiates the same interpolation
    /// [`crate::engine::simp_moduli`] evaluated: a forward sweep and an adjoint
    /// taken at two different floors is a gradient nothing reports as wrong.
    emin: f64,
    penalty: f64,
    /// Relative residual the load case solves are taken to: the problem's
    /// `[solver] tolerance`, unless [`Objective::with_cg_tolerance`] replaced
    /// it.
    cg_tolerance: f64,
    has_gravity: bool,
}

impl<'a> Objective<'a> {
    /// Bind the objective to a problem and the chain that maps its design
    /// variables to physical densities.
    pub fn new(problem: &'a Problem, chain: &'a DensityChain) -> Objective<'a> {
        let e0 = problem.material.youngs_modulus_mpa;
        Objective {
            problem,
            chain,
            ke0: hex8_stiffness(problem.material.poisson_ratio, problem.grid.h),
            e0,
            emin: problem.optimization.stiffness_floor * e0,
            penalty: problem.optimization.penalty,
            cg_tolerance: problem.solver.tolerance,
            has_gravity: problem.has_gravity(),
        }
    }

    /// Override the relative residual the linear solves are taken to, past
    /// whatever the problem's `[solver] tolerance` resolved to. Used by the
    /// finite difference gradient check, which needs the objective itself to be
    /// far more accurate than the difference step it takes - tighter than any
    /// configuration is allowed to ask for, which is why it is code and not a
    /// key.
    pub fn with_cg_tolerance(mut self, tolerance: f64) -> Objective<'a> {
        self.cg_tolerance = tolerance;
        self
    }

    /// Push design variables through the chain into `work.filtered` and
    /// `work.printed`.
    pub fn densities(&self, x: &[f64], work: &mut Workspace) {
        self.chain.apply(x, &mut work.filtered, &mut work.printed);
    }

    /// Evaluate the objective at `x` and write `dC/dx` into `dc`.
    ///
    /// `work.displacements` carries the warm start in and the solution out.
    /// `solver` is where the linear solves run; it is passed in rather than
    /// owned so that one device is opened per run instead of per iteration.
    ///
    /// Returns `Ok(None)` when `cancel` stopped one of the load case solves.
    /// Nothing was computed from it, `dc` is untouched, and the caller's loop is
    /// expected to unwind: the physical densities in `work` are still the ones
    /// this evaluation started from, so the design the last completed iteration
    /// left behind is intact.
    pub fn evaluate(
        &self,
        x: &[f64],
        work: &mut Workspace,
        dc: &mut [f64],
        solver: &mut LinearSolver,
        cancel: CancelProbe<'_>,
    ) -> Result<Option<Value>> {
        self.densities(x, work);
        crate::engine::simp_moduli(
            &self.problem.grid,
            &work.printed,
            self.e0,
            self.penalty,
            self.problem.optimization.stiffness_floor,
            &mut work.moduli,
        );

        let grid = &self.problem.grid;
        let operator = StiffnessOperator::new(grid, &self.ke0, &work.moduli);
        let diagonal = operator.diagonal();
        let mut bound = solver
            .bind(&operator, &diagonal, &self.problem.fixed)
            .context("binding the design to the linear solver")?;

        let mut compliance = 0.0;
        let mut cg_iterations = Vec::with_capacity(self.problem.load_cases.len());
        work.energy_sum.iter_mut().for_each(|v| *v = 0.0);
        work.load_term.iter_mut().for_each(|v| *v = 0.0);

        for (index, case) in self.problem.load_cases.iter().enumerate() {
            self.problem
                .case_forces(index, &work.printed, &mut work.forces);
            let solution = bound
                .solve(
                    &work.forces,
                    &mut work.displacements[index],
                    SolveLimits {
                        tolerance: self.cg_tolerance,
                        max_iterations: constants::CG_MAX_ITERATIONS,
                        cancel,
                    },
                )
                .with_context(|| format!("solving load case \"{}\"", case.name))?;
            let Some(solution) = solution.converged() else {
                return Ok(None);
            };
            cg_iterations.push(solution.iterations);

            let u = &work.displacements[index];
            let case_compliance: f64 = work
                .forces
                .par_iter()
                .zip(u.par_iter())
                .map(|(f, d)| f * d)
                .sum();
            compliance += case.weight * case_compliance;

            operator.unit_strain_energies(u, &mut work.energies);
            let weight = case.weight;
            work.energy_sum
                .par_iter_mut()
                .zip(work.energies.par_iter())
                .for_each(|(acc, energy)| *acc += weight * energy);

            if let Some(acceleration) = case.gravity {
                gravity::accumulate_sensitivity(
                    grid,
                    u,
                    acceleration,
                    self.problem.material.density_g_cm3,
                    weight,
                    &mut work.load_term,
                );
            }
        }

        // dC/drho = -p rho^(p-1) (E0 - Emin) sum_i w_i u_i^T ke0 u_i, plus the
        // self-weight load term where there is one.
        let kinds = &grid.cells;
        let (e0, emin, penalty) = (self.e0, self.emin, self.penalty);
        let printed = &work.printed;
        let energy_sum = &work.energy_sum;
        work.d_printed
            .par_iter_mut()
            .enumerate()
            .for_each(|(e, value)| {
                *value = if kinds[e] == CellKind::Design {
                    -penalty * printed[e].powf(penalty - 1.0) * (e0 - emin) * energy_sum[e]
                } else {
                    0.0
                };
            });
        if self.has_gravity {
            let load_term = &work.load_term;
            work.d_printed
                .par_iter_mut()
                .enumerate()
                .for_each(|(e, value)| {
                    if kinds[e] == CellKind::Design {
                        *value += load_term[e];
                    }
                });
        }

        self.chain.apply_transpose(
            &work.filtered,
            &work.printed,
            &work.d_printed,
            &mut work.d_filtered,
            dc,
        );
        Ok(Some(Value {
            compliance,
            cg_iterations,
        }))
    }

    /// Sensitivity of the mean printed density over the design cells with
    /// respect to the design variables, at the point currently in `work`.
    ///
    /// Constant for a linear chain; the additive manufacturing stage makes it
    /// design dependent, which is why it is recomputed per iteration when that
    /// stage is present.
    pub fn volume_sensitivity(&self, work: &mut Workspace, design_cells: &[usize], dv: &mut [f64]) {
        let share = 1.0 / design_cells.len() as f64;
        work.d_printed.iter_mut().for_each(|v| *v = 0.0);
        for &e in design_cells {
            work.d_printed[e] = share;
        }
        self.chain.apply_transpose(
            &work.filtered,
            &work.printed,
            &work.d_printed,
            &mut work.d_filtered,
            dv,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    /// A grid small enough to differentiate exhaustively, loaded so that both
    /// sensitivity terms are alive.
    fn config(extra_optimization: &str, gravity: &str) -> String {
        format!(
            r#"
[project]
name = "fd"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 6.0
max_iterations = 4
{extra_optimization}

[output]
stl_path = "fd.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [12.0, 12.0, 12.0]

[[supports]]
region = {{ shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 12.5, 12.5] }}

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = {{ shape = "sphere", center = [12.0, 6.0, 6.0], radius = 3.0 }}
vector = [0.0, 0.0, -30.0]
{gravity}
"#
        )
    }

    fn problem(text: &str) -> Problem {
        let config = Config::parse(text).expect("parse");
        Problem::build(&config, &PathBuf::from(".")).expect("build")
    }

    /// Largest relative gap between the adjoint gradient and a central finite
    /// difference of the objective, over a stride through the design cells.
    fn gradient_error(problem: &Problem, chain: &DensityChain) -> f64 {
        let cells = problem.grid.n_cells();
        let design: Vec<usize> = (0..cells)
            .filter(|&e| problem.grid.cells[e] == CellKind::Design)
            .collect();
        // A non-uniform starting design, so no term of the chain is at a
        // symmetric point that could hide a sign error.
        let mut x = vec![0.0; cells];
        for (n, &e) in design.iter().enumerate() {
            x[e] = 0.25 + 0.5 * ((n % 7) as f64 / 7.0);
        }

        let objective = Objective::new(problem, chain).with_cg_tolerance(1e-13);
        let mut solver = LinearSolver::cpu();
        let mut work = Workspace::new(problem);
        let mut adjoint = vec![0.0; cells];
        objective
            .evaluate(&x, &mut work, &mut adjoint, &mut solver, CancelProbe::NONE)
            .expect("evaluate")
            .expect("nothing asked this evaluation to stop");

        let step = 1e-5;
        let scale = adjoint.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut worst: f64 = 0.0;
        for &e in design.iter().step_by(13) {
            let mut shifted = x.clone();
            shifted[e] = x[e] + step;
            let mut work = Workspace::new(problem);
            let mut scratch = vec![0.0; cells];
            let plus = objective
                .evaluate(
                    &shifted,
                    &mut work,
                    &mut scratch,
                    &mut solver,
                    CancelProbe::NONE,
                )
                .expect("evaluate")
                .expect("not cancelled")
                .compliance;
            shifted[e] = x[e] - step;
            let mut work = Workspace::new(problem);
            let minus = objective
                .evaluate(
                    &shifted,
                    &mut work,
                    &mut scratch,
                    &mut solver,
                    CancelProbe::NONE,
                )
                .expect("evaluate")
                .expect("not cancelled")
                .compliance;

            let numeric = (plus - minus) / (2.0 * step);
            let error = (numeric - adjoint[e]).abs() / scale;
            worst = worst.max(error);
        }
        worst
    }

    fn chain_for(problem: &Problem) -> DensityChain {
        DensityChain::new(
            &problem.grid,
            problem.optimization.filter_radius_mm,
            problem.optimization.overhang,
        )
    }

    #[test]
    fn the_plain_gradient_matches_finite_differences() {
        let problem = problem(&config("", ""));
        let error = gradient_error(&problem, &chain_for(&problem));
        assert!(error < 1e-7, "relative gradient error {error}");
    }

    /// The penalization exponent is a configured number, and it enters the
    /// adjoint through `p rho^(p-1)` - a term that stays *self consistent* under
    /// an off-by-one as long as everything reads the same constant. Only a
    /// gradient taken at an exponent the constant does not hold can see a stage
    /// that kept reading the constant while the forward sweep moved, so the
    /// check is run at two of them on either side of the default.
    #[test]
    fn the_gradient_matches_finite_differences_at_a_non_default_penalty() {
        for penalty in [2.0, 4.0] {
            let problem = problem(&config(&format!("penalty = {penalty}"), ""));
            assert_eq!(problem.optimization.penalty, penalty);
            let error = gradient_error(&problem, &chain_for(&problem));
            println!("penalty {penalty}: relative gradient error {error:.3e}");
            assert!(
                error < 1e-7,
                "relative gradient error {error} at p = {penalty}"
            );
        }
    }

    /// And the exponent really reaches the analysis: the same design is stiffer
    /// under a smaller penalty, because intermediate density is priced less
    /// harshly, so its compliance is lower.
    #[test]
    fn the_penalty_reaches_the_stiffness_interpolation() {
        let cells = problem(&config("", "")).grid.n_cells();
        let x = vec![0.5; cells];
        let compliance = |penalty: &str| {
            let problem = problem(&config(penalty, ""));
            let chain = chain_for(&problem);
            let objective = Objective::new(&problem, &chain);
            let mut solver = LinearSolver::cpu();
            let mut work = Workspace::new(&problem);
            let mut dc = vec![0.0; cells];
            objective
                .evaluate(&x, &mut work, &mut dc, &mut solver, CancelProbe::NONE)
                .expect("evaluate")
                .expect("not cancelled")
                .compliance
        };
        let soft = compliance("penalty = 2.0");
        let default = compliance("");
        let hard = compliance("penalty = 4.0");
        assert!(
            soft < default && default < hard,
            "compliance has to grow with the penalty at a grey design: {soft}, {default}, {hard}"
        );
    }

    /// The stiffness floor is a configured number for the same reason and with
    /// the same failure mode: `Emin` enters the forward interpolation and the
    /// adjoint's `(E0 - Emin)` factor separately, so a stage left reading the
    /// constant while the rest moved is a gradient that is quietly wrong rather
    /// than a run that stops. Only a floor the constant does not hold can see
    /// it, and 1e-6 is three decades from the default.
    ///
    /// The whole range is walked rather than one value: the floor scales the
    /// term it appears in, and at the upper bound it is a thousandth of the
    /// solid modulus rather than a billionth.
    #[test]
    fn the_gradient_matches_finite_differences_at_a_non_default_stiffness_floor() {
        for floor in [
            constants::STIFFNESS_FLOOR_MIN,
            1e-6,
            constants::STIFFNESS_FLOOR_MAX,
        ] {
            let problem = problem(&config(&format!("stiffness_floor = {floor:e}"), ""));
            assert_eq!(problem.optimization.stiffness_floor, floor);
            let error = gradient_error(&problem, &chain_for(&problem));
            println!("stiffness_floor {floor:e}: relative gradient error {error:.3e}");
            assert!(
                error < 1e-7,
                "relative gradient error {error} at Emin/E0 = {floor:e}"
            );
        }
    }

    /// And the floor really reaches the interpolation the solve is assembled
    /// from, at both ends of a design cell's range: an emptied cell carries
    /// exactly the fraction the configuration named, and a full one carries the
    /// solid modulus whatever the floor is.
    #[test]
    fn the_stiffness_floor_reaches_the_moduli_the_solve_is_assembled_from() {
        let moduli_at = |extra: &str, density: f64| {
            let problem = problem(&config(extra, ""));
            let e0 = problem.material.youngs_modulus_mpa;
            let cells = problem.grid.n_cells();
            let design = (0..cells)
                .find(|&e| problem.grid.cells[e] == CellKind::Design)
                .expect("the fixture has design cells");
            let mut moduli = vec![0.0; cells];
            crate::engine::simp_moduli(
                &problem.grid,
                &vec![density; cells],
                e0,
                problem.optimization.penalty,
                problem.optimization.stiffness_floor,
                &mut moduli,
            );
            (moduli[design], e0)
        };

        // The default is what it always was, to the bit.
        let (empty, e0) = moduli_at("", 0.0);
        assert_eq!(empty, constants::SIMP_EMIN_FRACTION * e0);
        // And a floor the file names is the one the assembly gets.
        let (empty, e0) = moduli_at("stiffness_floor = 1e-6", 0.0);
        assert_eq!(empty, 1e-6 * e0);
        let (full, e0) = moduli_at("stiffness_floor = 1e-6", 1.0);
        assert_eq!(full, e0, "a solid design cell carries E0 at any floor");
    }

    #[test]
    fn the_overhang_gradient_matches_finite_differences() {
        let problem = problem(&config(
            "[optimization.overhang]\nbuild_direction = \"z+\"",
            "",
        ));
        assert!(problem.optimization.overhang.is_some());
        let error = gradient_error(&problem, &chain_for(&problem));
        assert!(error < 1e-7, "relative gradient error {error}");
    }

    #[test]
    fn the_self_weight_gradient_matches_finite_differences() {
        let problem = problem(&config(
            "",
            "[[loadcases.loads]]\ntype = \"gravity\"\ng_mm_s2 = 9810.0",
        ));
        assert!(problem.load_cases[0].gravity.is_some());
        let error = gradient_error(&problem, &chain_for(&problem));
        assert!(error < 1e-7, "relative gradient error {error}");
    }

    #[test]
    fn self_weight_adds_a_load_term_the_plain_objective_does_not_have() {
        let plain = problem(&config("", ""));
        let heavy = problem(&config(
            "",
            // An absurd gravity, so the self-weight term dominates and its sign
            // cannot be mistaken for noise.
            "[[loadcases.loads]]\ntype = \"gravity\"\ng_mm_s2 = 9.81e6",
        ));
        let cells = plain.grid.n_cells();
        let x = vec![0.5; cells];

        let mut compliances = Vec::new();
        let mut positives = Vec::new();
        for problem in [&plain, &heavy] {
            let chain = chain_for(problem);
            let objective = Objective::new(problem, &chain);
            let mut solver = LinearSolver::cpu();
            let mut work = Workspace::new(problem);
            let mut dc = vec![0.0; cells];
            let value = objective
                .evaluate(&x, &mut work, &mut dc, &mut solver, CancelProbe::NONE)
                .expect("evaluate")
                .expect("not cancelled");
            compliances.push(value.compliance);
            positives.push(dc.iter().filter(|v| **v > 0.0).count());
        }
        assert!(
            compliances[1] > compliances[0],
            "self weight must add compliance: {compliances:?}"
        );
        assert_eq!(positives[0], 0, "a plain load can only have dC/dx <= 0");
        assert!(
            positives[1] > 0,
            "a dominant self weight must push some sensitivities positive"
        );
    }

    /// The solves are taken to the tolerance the *configuration* asked for, and
    /// the gradient check's override still wins over it.
    ///
    /// The two are different things on purpose. `[solver] tolerance` is what a
    /// run's physics is worth solving to; `with_cg_tolerance` is what a test
    /// needs the objective itself to be accurate to, which is tighter than any
    /// configuration is allowed to name.
    #[test]
    fn the_solves_are_taken_to_the_configured_tolerance_unless_a_caller_overrides_it() {
        let named = problem(&config("[solver]\ntolerance = 3e-8", ""));
        assert_eq!(named.solver.tolerance, 3e-8);
        let chain = chain_for(&named);
        assert_eq!(Objective::new(&named, &chain).cg_tolerance, 3e-8);
        assert_eq!(
            Objective::new(&named, &chain)
                .with_cg_tolerance(1e-13)
                .cg_tolerance,
            1e-13,
            "the finite difference harness has to keep its own tighter target"
        );

        // A configuration that says nothing is the constant, which is what
        // every recorded trajectory was recorded at.
        let plain = problem(&config("", ""));
        assert_eq!(
            Objective::new(&plain, &chain_for(&plain)).cg_tolerance,
            constants::CG_RELATIVE_TOLERANCE
        );
    }

    /// And it reaches the solve: a looser target costs fewer iterations and
    /// buys the same physics.
    ///
    /// Which is the whole case for the key. Compliance is a quadratic
    /// functional of the displacement, so what a decade of residual costs it is
    /// smaller than the residual itself by an order that no printed part - and
    /// no single trilinear hexahedron per voxel - could resolve.
    #[test]
    fn a_looser_tolerance_is_fewer_iterations_and_the_same_compliance() {
        let evaluate = |solver: &str| {
            let problem = problem(&config(solver, ""));
            let chain = chain_for(&problem);
            let cells = problem.grid.n_cells();
            let objective = Objective::new(&problem, &chain);
            let mut linear = LinearSolver::cpu();
            let mut work = Workspace::new(&problem);
            let mut dc = vec![0.0; cells];
            let value = objective
                .evaluate(
                    &vec![0.5; cells],
                    &mut work,
                    &mut dc,
                    &mut linear,
                    CancelProbe::NONE,
                )
                .expect("evaluate")
                .expect("not cancelled");
            (value.compliance, value.cg_iterations[0])
        };

        let (tight, tight_iterations) = evaluate("");
        let (loose, loose_iterations) = evaluate("[solver]\ntolerance = 1e-4");
        let error = (loose - tight).abs() / tight.abs();
        println!(
            "1e-8: {tight_iterations} iterations, 1e-4: {loose_iterations}, relative compliance \
             difference {error:.3e}"
        );
        assert!(
            loose_iterations < tight_iterations,
            "a looser target has to cost fewer iterations: {loose_iterations} against \
             {tight_iterations}"
        );
        assert!(
            error < 1e-4,
            "four decades of residual moved the compliance by {error:.3e}"
        );
    }

    /// A stop inside a load case solve unwinds the whole evaluation, and leaves
    /// the physical densities the caller handed in exactly where they were: the
    /// SIMP loop keeps the design of the last completed iteration on that.
    #[test]
    fn a_cancelled_load_case_solve_unwinds_the_whole_evaluation() {
        let problem = problem(&config("", ""));
        let chain = chain_for(&problem);
        let objective = Objective::new(&problem, &chain);
        let mut solver = LinearSolver::cpu();
        let cells = problem.grid.n_cells();
        let x = vec![0.5; cells];
        let mut work = Workspace::new(&problem);
        let mut dc = vec![f64::NAN; cells];

        let stop = || true;
        let outcome = objective
            .evaluate(
                &x,
                &mut work,
                &mut dc,
                &mut solver,
                CancelProbe::watching(&stop),
            )
            .expect("a stop is not an error");
        assert!(outcome.is_none(), "a stopped evaluation has no value");
        // The chain ran before the solve, so the densities are the ones this
        // design implies; the sensitivities were never written.
        assert!(work.printed.iter().all(|d| d.is_finite()));
        assert!(dc.iter().all(|v| v.is_nan()));
    }
}
