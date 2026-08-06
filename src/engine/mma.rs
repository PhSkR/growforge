//! Svanberg's method of moving asymptotes, specialized to this problem shape:
//! minimize the compliance approximation subject to one volume constraint and
//! the box `[DENSITY_MIN, DENSITY_MAX]` with a move limit.
//!
//! # The approximation
//!
//! Every design variable carries a lower and an upper asymptote,
//! `L_e < x_e < U_e`. A function `f` with sensitivity `df/dx_e` at the current
//! point `x^k` is replaced by the separable convex approximation
//!
//! ```text
//!   f(x) ~ const + sum_e [ p_e / (U_e - x_e) + q_e / (x_e - L_e) ]
//!   p_e = (U_e - x_e^k)^2 max( df/dx_e, 0)
//!   q_e = (x_e^k - L_e)^2 max(-df/dx_e, 0)
//! ```
//!
//! which matches `f` and its gradient at `x^k` and is convex between the
//! asymptotes. This is the plain 1987 form: no perturbation is added for strict
//! convexity, because the one degenerate case it would paper over - a variable
//! with `p_e = q_e = 0` in *both* functions - is handled explicitly below.
//!
//! # The subproblem and its dual
//!
//! With one constraint the Lagrangian is
//! `L(x, lambda) = f0~(x) + lambda f1~(x)`, separable in `x`, so its minimizer
//! is available per variable in closed form. Writing
//! `p = p0_e + lambda p1_e` and `q = q0_e + lambda q1_e`, stationarity
//!
//! ```text
//!   p / (U_e - x)^2 - q / (x - L_e)^2 = 0
//! ```
//!
//! gives, since `L_e < x < U_e` makes both brackets positive,
//!
//! ```text
//!   x_e(lambda) = (L_e sqrt(p) + U_e sqrt(q)) / (sqrt(p) + sqrt(q))
//! ```
//!
//! clipped to `[alpha_e, beta_e]`, the intersection of the box, the move limit
//! and a margin that keeps the trial point clear of the poles at the asymptotes
//! ([`constants::MMA_BOUND_ASYMPTOTE_FRACTION`]). The dual function
//! `W(lambda) = min_x L(x, lambda)` is therefore scalar and concave, and
//! `dW/dlambda` is the constraint value at `x(lambda)`: maximizing the dual is
//! finding the multiplier at which the volume constraint is met.
//!
//! `x_e(lambda)` is non-increasing in `lambda` (a larger multiplier prices
//! material higher and shifts the weight in the formula towards `L_e`) and the
//! density chain is monotone, so the printed volume is non-increasing in
//! `lambda` too. That is what makes the multiplier bisectable, exactly as in
//! [`super::oc`], and the residual is measured the same way: on the mean
//! *printed* density at the far end of the chain rather than on the separable
//! approximation of it. The two agree to first order at `x^k`, and measuring the
//! real one is what leaves the accepted step feasible in the constraint the run
//! actually reports.
//!
//! # Two constraints
//!
//! With `[optimization.local_volume]` set there is a second constraint - see
//! [`super::local_volume`] - and the subproblem becomes a dual in
//! `(lambda1, lambda2) >= 0`. The primal is the same closed form with
//! `p = p0 + lambda1 p1 + lambda2 p2` and `q` likewise, and the dual
//! `W(lambda) = min_x L(x, lambda)` is still concave: it is a pointwise infimum
//! of functions affine in `lambda`. Its partial derivatives are the two
//! constraint values at `x(lambda)`, each non-increasing in its own multiplier
//! for the same reason the scalar one is, so it is maximized by **exact
//! coordinate ascent**: bisect `lambda1` with `lambda2` held, then `lambda2`
//! with `lambda1` held, and sweep until neither moves. Each coordinate bisection
//! is the scalar one, under the same [`constants::MMA_BISECTION_MAX_STEPS`] and
//! [`constants::MMA_BISECTION_TOL`] budget; the sweeps are capped by
//! [`constants::LOCAL_VOLUME_DUAL_MAX_SWEEPS`] and the joint residual is
//! measured against [`constants::LOCAL_VOLUME_DUAL_TOLERANCE`], so the solve is
//! deterministic and bounded either way. A coordinate whose constraint is
//! already slack at `lambda_i = 0` stays there, which is the KKT condition for a
//! multiplier on the boundary of the non-negative orthant.
//!
//! **The two-constraint dual trusts the separable approximations of both
//! constraints, which is a deliberate exception to the rule above.** The scalar
//! dual re-measures the real printed volume at every trial multiplier because it
//! is cheap to: one pass of the density filter. The local constraint's kernel is
//! one to two orders of magnitude more taps than that filter, and the dual takes
//! hundreds of trial points per iteration; re-measuring it at each of them would
//! cost more than the finite element solve the iteration exists for. So the
//! dual is solved on the approximations - which match value *and* gradient at
//! `x^k`, and are anchored on the constraint values really measured there - and
//! both constraints are measured for real once per outer iteration, on the step
//! that was accepted, which is what the run reports and what the next
//! iteration's approximations are anchored on. This is standard Svanberg
//! practice; what it costs is that the volume fraction of an accepted
//! two-constraint step can sit a little off the target while the design is still
//! moving, rather than on it to [`constants::MMA_VOLUME_TOLERANCE`].
//!
//! # Positive objective sensitivities
//!
//! MMA needs no equivalent of the optimality criteria step's self-weight
//! sensitivity shift. A positive `dC/dx_e`, which a design dependent load
//! produces, lands in `p0_e` instead of `q0_e` and pushes that variable towards
//! its lower asymptote; the approximation stays convex and the dual stays
//! monotone either way. [`Step::shift`] is therefore always zero here.
//!
//! K. Svanberg, "The method of moving asymptotes - a new method for structural
//! optimization", Int J Numer Meth Engng 24 (1987) 359-373.

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::constants;
use crate::engine::update::{Buffers, Constraint, LocalConstraint, Step, design_mean};

/// What MMA carries from one iteration to the next: the asymptotes and the two
/// previous design points the adaptive rule needs to see a trend.
#[derive(Debug, Clone)]
pub struct State {
    /// Lower asymptote per cell; only the design cells are ever read.
    lower: Vec<f64>,
    /// Upper asymptote per cell.
    upper: Vec<f64>,
    /// Design variables of the previous iteration.
    previous: Vec<f64>,
    /// Design variables of the iteration before that.
    older: Vec<f64>,
    /// Steps taken so far, which is what selects the initial asymptote rule.
    steps: usize,
}

/// One design variable's share of the subproblem: its asymptotes, the bounds
/// the trial point is clipped to and the four approximation coefficients.
#[derive(Debug, Clone, Copy)]
struct Variable {
    /// Cell the variable belongs to.
    cell: usize,
    /// Lower asymptote.
    lower: f64,
    /// Upper asymptote.
    upper: f64,
    /// Lower bound of the trial point.
    alpha: f64,
    /// Upper bound of the trial point.
    beta: f64,
    /// Objective coefficient on the upper asymptote term.
    p0: f64,
    /// Objective coefficient on the lower asymptote term.
    q0: f64,
    /// Volume constraint coefficient on the upper asymptote term.
    p1: f64,
    /// Volume constraint coefficient on the lower asymptote term.
    q1: f64,
    /// Local volume constraint coefficient on the upper asymptote term; zero
    /// when there is no second constraint.
    p2: f64,
    /// Local volume constraint coefficient on the lower asymptote term.
    q2: f64,
    /// What the two constraint approximations' own terms sum to at `x^k`, in
    /// constraint order.
    ///
    /// Only the two-constraint dual reads them: an approximation that is to be
    /// *evaluated* rather than only differentiated has to be anchored on the
    /// constraint value really measured at `x^k`, and this is the part of it
    /// that is subtracted off. The scalar dual needs no anchor because it never
    /// evaluates the approximation - it measures the real printed volume at
    /// every trial point instead.
    anchor: [f64; 2],
}

/// Minimizer of the subproblem's Lagrangian for one variable at `lambda`.
///
/// A variable whose four coefficients all vanish has a zero row in the whole
/// density chain: it changes no physical density, so it changes neither the
/// compliance nor the volume, and both approximations are flat in it. The
/// self-supporting filter creates exactly that situation for blueprint material
/// it erases, and such a variable decays to its lower bound rather than
/// floating - for the reason [`super::oc`] gives at its own decay branch: a
/// value nothing depends on today becomes material the moment support grows
/// underneath it. A linear chain never produces a zero row for a design cell.
fn primal(v: &Variable, lambda: f64) -> f64 {
    let p = v.p0 + lambda * v.p1;
    let q = v.q0 + lambda * v.q1;
    stationary(v, p, q)
}

/// The same for the two-constraint subproblem.
///
/// Deliberately not `primal` with a zero second multiplier: the single
/// constraint path is the one every existing trajectory was recorded against,
/// and it does the arithmetic it always did, term for term.
fn primal_two(v: &Variable, lambda: [f64; 2]) -> f64 {
    let p = v.p0 + lambda[0] * v.p1 + lambda[1] * v.p2;
    let q = v.q0 + lambda[0] * v.q1 + lambda[1] * v.q2;
    stationary(v, p, q)
}

/// Stationary point of one variable's share of the Lagrangian, given the
/// combined coefficients, clipped to the bounds.
fn stationary(v: &Variable, p: f64, q: f64) -> f64 {
    if p <= 0.0 && q <= 0.0 {
        return v.alpha;
    }
    let (root_p, root_q) = (p.sqrt(), q.sqrt());
    ((v.lower * root_p + v.upper * root_q) / (root_p + root_q)).clamp(v.alpha, v.beta)
}

/// Evaluate the whole trial design for one multiplier.
fn trial(variables: &[Variable], lambda: f64, out: &mut [f64]) {
    let updates: Vec<(usize, f64)> = variables
        .par_iter()
        .map(|v| (v.cell, primal(v, lambda)))
        .collect();
    for (cell, value) in updates {
        out[cell] = value;
    }
}

/// The same for the two-constraint subproblem.
fn trial_two(variables: &[Variable], lambda: [f64; 2], out: &mut [f64]) {
    let updates: Vec<(usize, f64)> = variables
        .par_iter()
        .map(|v| (v.cell, primal_two(v, lambda)))
        .collect();
    for (cell, value) in updates {
        out[cell] = value;
    }
}

/// Both separable constraint approximations at the trial point `x(lambda)`.
///
/// These are the two partial derivatives of the dual, so a zero is what each
/// coordinate of the ascent bisects for. `offsets` carries the anchoring
/// constants: the constraint value really measured at `x^k`, less what the
/// approximation's own terms contribute there.
fn approximations(variables: &[Variable], lambda: [f64; 2], offsets: [f64; 2]) -> [f64; 2] {
    let sums = variables
        .par_iter()
        .map(|v| {
            let x = primal_two(v, lambda);
            let (up, down) = (v.upper - x, x - v.lower);
            [v.p1 / up + v.q1 / down, v.p2 / up + v.q2 / down]
        })
        .reduce(|| [0.0; 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
    [offsets[0] + sums[0], offsets[1] + sums[1]]
}

impl State {
    /// Cold state for a grid of `n_cells` cells.
    pub fn new(n_cells: usize) -> State {
        State {
            lower: vec![0.0; n_cells],
            upper: vec![0.0; n_cells],
            previous: vec![0.0; n_cells],
            older: vec![0.0; n_cells],
            steps: 0,
        }
    }

    /// Place the asymptotes for the step about to be taken from `x`.
    ///
    /// The first [`constants::MMA_INITIAL_ASYMPTOTE_STEPS`] steps take a
    /// symmetric range around the current point, because the adaptive rule needs
    /// two previous points to see whether a variable is oscillating. After that
    /// the distance to each asymptote is carried over from the previous step and
    /// scaled by Svanberg's gamma: shrunk when the variable reversed direction,
    /// expanded when it kept going, unchanged when it stood still. The result is
    /// clamped to a fraction of the box, so damping can never freeze a variable
    /// and expansion can never flatten the approximation into a linear one.
    fn move_asymptotes(&mut self, x: &[f64], design_cells: &[usize]) {
        let width = constants::DENSITY_MAX - constants::DENSITY_MIN;
        let initial = constants::MMA_ASYMPTOTE_INIT * width;
        let (min_gap, max_gap) = (
            constants::MMA_ASYMPTOTE_MIN * width,
            constants::MMA_ASYMPTOTE_MAX * width,
        );
        if self.steps < constants::MMA_INITIAL_ASYMPTOTE_STEPS {
            for &e in design_cells {
                self.lower[e] = x[e] - initial;
                self.upper[e] = x[e] + initial;
            }
            return;
        }
        for &e in design_cells {
            let trend = (x[e] - self.previous[e]) * (self.previous[e] - self.older[e]);
            let gamma = if trend < 0.0 {
                constants::MMA_ASYMPTOTE_SHRINK
            } else if trend > 0.0 {
                constants::MMA_ASYMPTOTE_EXPAND
            } else {
                1.0
            };
            let lower = x[e] - gamma * (self.previous[e] - self.lower[e]);
            let upper = x[e] + gamma * (self.upper[e] - self.previous[e]);
            self.lower[e] = lower.clamp(x[e] - max_gap, x[e] - min_gap);
            self.upper[e] = upper.clamp(x[e] + min_gap, x[e] + max_gap);
        }
    }

    /// Build the subproblem the current asymptotes and sensitivities define.
    ///
    /// `dg` is the second constraint's sensitivity when there is one; without it
    /// the second pair of coefficients is zero and only [`primal`] ever runs.
    fn variables(
        &self,
        x: &[f64],
        dc: &[f64],
        dv: &[f64],
        dg: Option<&[f64]>,
        design_cells: &[usize],
    ) -> Vec<Variable> {
        let move_limit = constants::MMA_MOVE_LIMIT;
        let albefa = constants::MMA_BOUND_ASYMPTOTE_FRACTION;
        design_cells
            .par_iter()
            .map(|&e| {
                let (xe, lower, upper) = (x[e], self.lower[e], self.upper[e]);
                let (to_lower, to_upper) = (xe - lower, upper - xe);
                let local = dg.map_or(0.0, |dg| dg[e]);
                let (p1, q1) = (
                    to_upper * to_upper * dv[e].max(0.0),
                    to_lower * to_lower * (-dv[e]).max(0.0),
                );
                let (p2, q2) = (
                    to_upper * to_upper * local.max(0.0),
                    to_lower * to_lower * (-local).max(0.0),
                );
                Variable {
                    cell: e,
                    lower,
                    upper,
                    alpha: (lower + albefa * to_lower)
                        .max(xe - move_limit)
                        .max(constants::DENSITY_MIN),
                    beta: (upper - albefa * to_upper)
                        .min(xe + move_limit)
                        .min(constants::DENSITY_MAX),
                    p0: to_upper * to_upper * dc[e].max(0.0),
                    q0: to_lower * to_lower * (-dc[e]).max(0.0),
                    p1,
                    q1,
                    p2,
                    q2,
                    // p/(U - x^k) + q/(x^k - L) with the squares cancelled.
                    anchor: [
                        to_upper * dv[e].max(0.0) + to_lower * (-dv[e]).max(0.0),
                        to_upper * local.max(0.0) + to_lower * (-local).max(0.0),
                    ],
                }
            })
            .collect()
    }

    /// Perform one MMA update.
    ///
    /// `x` holds the current design variables, `dc` and `dv` the objective and
    /// volume sensitivities with respect to them (already pushed through the
    /// chain transposes). Only the design cells move; every other cell comes
    /// back exactly as it went in.
    pub fn update(
        &mut self,
        x: &[f64],
        dc: &[f64],
        dv: &[f64],
        constraint: &Constraint<'_>,
        buffers: Buffers<'_>,
    ) -> Result<Step> {
        let Buffers {
            x_new,
            filtered,
            printed,
        } = buffers;
        x_new.copy_from_slice(x);
        let design_cells = constraint.design_cells;

        self.move_asymptotes(x, design_cells);
        let variables = self.variables(
            x,
            dc,
            dv,
            constraint.local.as_ref().map(|local| local.gradient),
            design_cells,
        );

        let (lambda, volume_fraction) = match &constraint.local {
            Some(local) => {
                two_constraint_step(&variables, constraint, local, x_new, filtered, printed)?
            }
            None => volume_step(&variables, constraint, x_new, filtered, printed)?,
        };

        let max_change = design_cells
            .par_iter()
            .map(|&e| (x_new[e] - x[e]).abs())
            .reduce(|| 0.0, f64::max);

        self.older.copy_from_slice(&self.previous);
        self.previous.copy_from_slice(x);
        self.steps += 1;

        Ok(Step {
            max_change,
            volume_fraction,
            lambda,
            shift: 0.0,
        })
    }
}

/// The single-constraint step: bisect the volume multiplier against the **real**
/// mean printed density of every trial point.
///
/// Returns the multiplier and the volume fraction the accepted step reached.
fn volume_step(
    variables: &[Variable],
    constraint: &Constraint<'_>,
    x_new: &mut [f64],
    filtered: &mut [f64],
    printed: &mut [f64],
) -> Result<(f64, f64)> {
    let design_cells = constraint.design_cells;
    let mut lower = constants::MMA_LAMBDA_LOWER;
    let mut upper = constants::MMA_LAMBDA_UPPER;
    let mut lambda = 0.5 * (lower + upper);
    let mut volume_fraction = 0.0;

    for step in 0..constants::MMA_BISECTION_MAX_STEPS {
        lambda = 0.5 * (lower + upper);
        trial(variables, lambda, x_new);
        constraint.chain.apply(x_new, filtered, printed);
        volume_fraction = design_mean(printed, design_cells);

        if !volume_fraction.is_finite() {
            bail!(
                "moving asymptotes update produced a non-finite volume fraction at bisection step {step}"
            );
        }
        if (volume_fraction - constraint.target_volume_fraction).abs()
            < constants::MMA_VOLUME_TOLERANCE
        {
            break;
        }
        if volume_fraction > constraint.target_volume_fraction {
            lower = lambda;
        } else {
            upper = lambda;
        }
        let width = upper - lower;
        let scale = upper + lower;
        if scale <= 0.0 || width / scale < constants::MMA_BISECTION_TOL {
            break;
        }
    }
    Ok((lambda, volume_fraction))
}

/// The two-constraint step: maximize the dual over both multipliers by exact
/// coordinate ascent on the separable approximations, then measure the real
/// volume of the step that came out of it once.
///
/// Returns the volume multiplier - the one the step reports, as the single
/// constraint path does - and that measured volume fraction.
fn two_constraint_step(
    variables: &[Variable],
    constraint: &Constraint<'_>,
    local: &LocalConstraint<'_>,
    x_new: &mut [f64],
    filtered: &mut [f64],
    printed: &mut [f64],
) -> Result<(f64, f64)> {
    let offsets = dual_offsets(
        variables,
        local.current_volume_fraction - constraint.target_volume_fraction,
        local.value,
    );
    let lambda = maximize_dual(variables, offsets)?;

    trial_two(variables, lambda, x_new);
    constraint.chain.apply(x_new, filtered, printed);
    let volume_fraction = design_mean(printed, constraint.design_cells);
    if !volume_fraction.is_finite() {
        bail!(
            "moving asymptotes update produced a non-finite volume fraction at multipliers              [{}, {}]",
            lambda[0],
            lambda[1]
        );
    }
    Ok((lambda[0], volume_fraction))
}

/// What each approximation has to be shifted by to take its constraint's really
/// measured value at `x^k`: `f_i(x^k)` less the approximation's own terms there.
///
/// `volume_gap` is the volume constraint's value at `x^k` - the mean printed
/// density less the target - and `local_value` the local cap's.
fn dual_offsets(variables: &[Variable], volume_gap: f64, local_value: f64) -> [f64; 2] {
    let anchors = variables
        .par_iter()
        .map(|v| v.anchor)
        .reduce(|| [0.0; 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
    [volume_gap - anchors[0], local_value - anchors[1]]
}

/// Maximize the two-constraint dual by exact coordinate ascent.
///
/// One multiplier is bisected with the other held, then the other, and the sweep
/// repeats until neither moves or both constraints are met at multipliers only
/// tight constraints carry - the KKT conditions of a concave maximization over
/// the non-negative orthant. Bounded by
/// [`constants::LOCAL_VOLUME_DUAL_MAX_SWEEPS`] either way.
fn maximize_dual(variables: &[Variable], offsets: [f64; 2]) -> Result<[f64; 2]> {
    let tolerance = constants::LOCAL_VOLUME_DUAL_TOLERANCE;
    let mut lambda = [constants::MMA_LAMBDA_LOWER; 2];
    for _ in 0..constants::LOCAL_VOLUME_DUAL_MAX_SWEEPS {
        let mut moved: f64 = 0.0;
        for coordinate in 0..2 {
            let before = lambda[coordinate];
            let after = bisect_coordinate(variables, lambda, offsets, coordinate, tolerance)?;
            lambda[coordinate] = after;
            moved = moved.max((after - before).abs() / before.abs().max(after).max(1.0));
        }
        // The joint condition, which one coordinate's bisection cannot establish
        // on its own: every constraint is met, and a multiplier is priced only
        // where its own constraint is tight.
        let residual = approximations(variables, lambda, offsets);
        let satisfied = (0..2).all(|i| {
            residual[i] < tolerance
                && (lambda[i] <= constants::MMA_LAMBDA_LOWER || residual[i].abs() < tolerance)
        });
        if satisfied || moved < constants::MMA_BISECTION_TOL {
            break;
        }
    }
    Ok(lambda)
}

/// Maximize the dual along one coordinate with the other held: bisect the
/// multiplier until its own constraint approximation reads zero, or leave it at
/// zero when the constraint is already slack there.
fn bisect_coordinate(
    variables: &[Variable],
    lambda: [f64; 2],
    offsets: [f64; 2],
    coordinate: usize,
    tolerance: f64,
) -> Result<f64> {
    let at = |value: f64| {
        let mut trial = lambda;
        trial[coordinate] = value;
        approximations(variables, trial, offsets)[coordinate]
    };
    let lower_bound = constants::MMA_LAMBDA_LOWER;
    let slack = at(lower_bound);
    if !slack.is_finite() {
        bail!(
            "moving asymptotes update produced a non-finite constraint {} at a zero multiplier",
            coordinate + 1
        );
    }
    // Slack at a zero price: the constraint is not binding, and its multiplier
    // belongs on the boundary.
    if slack <= tolerance {
        return Ok(lower_bound);
    }
    let (mut lower, mut upper) = (lower_bound, constants::MMA_LAMBDA_UPPER);
    let mut value = 0.5 * (lower + upper);
    for _ in 0..constants::MMA_BISECTION_MAX_STEPS {
        value = 0.5 * (lower + upper);
        let residual = at(value);
        if !residual.is_finite() {
            bail!(
                "moving asymptotes update produced a non-finite constraint {} at multiplier \
                 {value}",
                coordinate + 1
            );
        }
        if residual.abs() < tolerance {
            break;
        }
        // The approximation is non-increasing in its own multiplier, because a
        // larger one moves every variable towards its lower asymptote and both
        // constraints are non-decreasing in the design.
        if residual > 0.0 {
            lower = value;
        } else {
            upper = value;
        }
        let (width, scale) = (upper - lower, upper + lower);
        if scale <= 0.0 || width / scale < constants::MMA_BISECTION_TOL {
            break;
        }
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::chain::DensityChain;
    use crate::geometry::Aabb;
    use crate::grid::{CellKind, Grid};

    /// A cube of design cells with an identity density chain, so a test sees the
    /// MMA step alone.
    fn setup(n: usize) -> (Grid, DensityChain, Vec<usize>) {
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [n as f64, n as f64, n as f64],
            },
            1.0,
        );
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Design;
        }
        // A filter radius below one voxel makes the filter the identity.
        let chain = DensityChain::new(&grid, 0.5, None);
        let design: Vec<usize> = (0..grid.n_cells()).collect();
        (grid, chain, design)
    }

    fn take(
        state: &mut State,
        x: &[f64],
        dc: &[f64],
        dv: &[f64],
        constraint: &Constraint<'_>,
        x_new: &mut [f64],
    ) -> Step {
        let n = x.len();
        let (mut filtered, mut printed) = (vec![0.0; n], vec![0.0; n]);
        state
            .update(
                x,
                dc,
                dv,
                constraint,
                Buffers {
                    x_new,
                    filtered: &mut filtered,
                    printed: &mut printed,
                },
            )
            .expect("mma step")
    }

    /// The closed form against a case worked out by hand.
    ///
    /// `L = 0.2`, `U = 0.8`, `x = 0.5`, `dC/dx = -2`, `dV/dx = 0.5` give
    /// `p0 = 0`, `q0 = 0.09 * 2 = 0.18`, `p1 = 0.09 * 0.5 = 0.045`, `q1 = 0`. At
    /// `lambda = 2` that is `p = 0.09` and `q = 0.18`, so
    /// `x = (0.2 sqrt(0.09) + 0.8 sqrt(0.18)) / (sqrt(0.09) + sqrt(0.18))`
    /// `= 0.399411254969543 / 0.724264068711928 = 0.5514718...`, inside the
    /// `[0.3, 0.7]` the move limit leaves.
    #[test]
    fn the_primal_matches_a_hand_computed_case() {
        let v = Variable {
            cell: 0,
            lower: 0.2,
            upper: 0.8,
            alpha: 0.3,
            beta: 0.7,
            p0: 0.0,
            q0: 0.18,
            p1: 0.045,
            q1: 0.0,
            p2: 0.0,
            q2: 0.0,
            anchor: [0.0; 2],
        };
        let x = primal(&v, 2.0);
        assert!(
            (x - 0.551_471_862_576_143).abs() < 1e-12,
            "closed form gave {x}"
        );
        // And it really is the stationary point of the Lagrangian: the two terms
        // of dL/dx cancel there.
        let (p, q) = (v.p0 + 2.0 * v.p1, v.q0 + 2.0 * v.q1);
        let gradient = p / (v.upper - x).powi(2) - q / (x - v.lower).powi(2);
        assert!(gradient.abs() < 1e-12, "dL/dx was {gradient}");

        // A larger multiplier prices material higher and moves the variable
        // down; a smaller one lets it grow. That monotonicity is what makes the
        // dual bisectable.
        assert!(primal(&v, 20.0) < x && x < primal(&v, 0.2));
        // The bounds are hard: at the extremes the trial point sits on them.
        assert!((primal(&v, 1e12) - v.alpha).abs() < 1e-12);
        assert!((primal(&v, 0.0) - v.beta).abs() < 1e-12);
    }

    /// A variable that changes neither the compliance nor the volume decays
    /// instead of floating, exactly as the optimality criteria step's zero
    /// volume sensitivity branch does.
    #[test]
    fn a_variable_nothing_depends_on_decays() {
        let v = Variable {
            cell: 0,
            lower: 0.2,
            upper: 0.8,
            alpha: 0.3,
            beta: 0.7,
            p0: 0.0,
            q0: 0.0,
            p1: 0.0,
            q1: 0.0,
            p2: 0.0,
            q2: 0.0,
            anchor: [0.0; 2],
        };
        assert_eq!(primal(&v, 1.0), v.alpha);
        // And under a second constraint it cannot depend on either: the decay
        // branch is the same one.
        assert_eq!(primal_two(&v, [1.0, 1.0]), v.alpha);
    }

    #[test]
    fn the_dual_bisection_hits_the_target_volume() {
        let (grid, chain, design) = setup(6);
        let n = grid.n_cells();
        let x = vec![0.5; n];
        let dc: Vec<f64> = (0..n).map(|e| -(1.0 + (e % 7) as f64)).collect();
        let dv = vec![1.0 / n as f64; n];
        let mut state = State::new(n);
        let mut x_new = vec![0.0; n];
        for target in [0.3, 0.5, 0.7] {
            let taken = take(
                &mut state,
                &x,
                &dc,
                &dv,
                &Constraint {
                    design_cells: &design,
                    target_volume_fraction: target,
                    chain: &chain,
                    local: None,
                },
                &mut x_new,
            );
            assert!(
                (taken.volume_fraction - target).abs() < constants::MMA_VOLUME_TOLERANCE,
                "volume fraction {} missed the {target} target",
                taken.volume_fraction
            );
            assert_eq!(taken.shift, 0.0, "MMA never shifts sensitivities");
            assert!(taken.lambda > 0.0, "the volume constraint has to be priced");
            assert!(taken.max_change <= constants::MMA_MOVE_LIMIT + 1e-12);
            for value in &x_new {
                assert!((constants::DENSITY_MIN..=constants::DENSITY_MAX).contains(value));
            }
        }
    }

    /// A two-constraint subproblem on the fixture above: the same design and
    /// objective, a uniform volume sensitivity, and a local cap whose
    /// sensitivity is concentrated on part of the domain.
    ///
    /// `local_value` is the cap's residual at the current design - negative is
    /// slack - and the returned tuple is the multipliers the dual settled on and
    /// the two constraint approximations there.
    fn dual(n: usize, target: f64, local_value: f64, weight: f64) -> ([f64; 2], [f64; 2]) {
        let (grid, _, design) = setup(n);
        let cells = grid.n_cells();
        let x = vec![0.5; cells];
        let dc: Vec<f64> = (0..cells).map(|e| -(1.0 + (e % 7) as f64)).collect();
        let dv = vec![1.0 / cells as f64; cells];
        // Half the domain carries the local cap's sensitivity, which is what
        // makes the two constraints pull on different variables and the dual a
        // genuinely two dimensional problem.
        let dg: Vec<f64> = (0..cells)
            .map(|e| {
                if e % 2 == 0 {
                    weight / cells as f64
                } else {
                    0.0
                }
            })
            .collect();
        let mut state = State::new(cells);
        state.move_asymptotes(&x, &design);
        let variables = state.variables(&x, &dc, &dv, Some(&dg), &design);
        let offsets = dual_offsets(&variables, 0.5 - target, local_value);
        let lambda = maximize_dual(&variables, offsets).expect("dual");
        (lambda, approximations(&variables, lambda, offsets))
    }

    /// A slack local cap costs nothing: its multiplier stays on the boundary and
    /// the volume constraint is priced alone, exactly as the scalar dual prices
    /// it.
    #[test]
    fn a_slack_local_cap_is_not_priced() {
        let (lambda, residual) = dual(6, 0.4, -0.25, 1.0);
        assert_eq!(
            lambda[1],
            constants::MMA_LAMBDA_LOWER,
            "a constraint that is not binding may not be priced"
        );
        assert!(lambda[0] > 0.0, "the volume constraint has to be priced");
        assert!(
            residual[0].abs() < constants::LOCAL_VOLUME_DUAL_TOLERANCE,
            "the volume constraint was left unmet: {residual:?}"
        );
        assert!(
            residual[1] < 0.0,
            "the local cap has to stay slack: {residual:?}"
        );
    }

    /// Both binding: both multipliers are positive and both constraints are met
    /// to the dual's tolerance, which one coordinate's bisection cannot do on its
    /// own - moving either multiplier moves the other's residual.
    #[test]
    fn a_binding_local_cap_is_priced_beside_the_volume() {
        let (lambda, residual) = dual(6, 0.48, 0.08, 4.0);
        println!("two-constraint dual: lambda {lambda:?}, residuals {residual:?}");
        assert!(lambda[0] > 0.0 && lambda[1] > 0.0, "{lambda:?}");
        for value in residual {
            assert!(
                value.abs() < constants::LOCAL_VOLUME_DUAL_TOLERANCE,
                "a constraint was left unmet: {residual:?}"
            );
        }
        // And it is the same answer every time: the dual is a deterministic
        // search over a fixed subproblem, so a repeat run reproduces it exactly.
        for _ in 0..3 {
            assert_eq!(dual(6, 0.48, 0.08, 4.0), (lambda, residual));
        }
    }

    /// The dual through the update step itself: the local cap changes the design
    /// the step produces, and every bound the single-constraint step respects is
    /// still respected.
    #[test]
    fn a_two_constraint_step_stays_inside_every_bound_and_moves_the_design() {
        let (grid, chain, design) = setup(6);
        let n = grid.n_cells();
        let x = vec![0.5; n];
        let dc: Vec<f64> = (0..n).map(|e| -(1.0 + (e % 7) as f64)).collect();
        let dv = vec![1.0 / n as f64; n];
        let dg: Vec<f64> = (0..n)
            .map(|e| if e % 2 == 0 { 4.0 / n as f64 } else { 0.0 })
            .collect();
        let volume = |local: Option<LocalConstraint<'_>>| {
            let mut state = State::new(n);
            let mut x_new = vec![0.0; n];
            let step = take(
                &mut state,
                &x,
                &dc,
                &dv,
                &Constraint {
                    design_cells: &design,
                    target_volume_fraction: 0.48,
                    chain: &chain,
                    local,
                },
                &mut x_new,
            );
            (step, x_new)
        };
        let (plain, plain_design) = volume(None);
        let (capped, capped_design) = volume(Some(LocalConstraint {
            value: 0.08,
            gradient: &dg,
            current_volume_fraction: 0.5,
        }));

        assert_eq!(capped.shift, 0.0, "MMA never shifts sensitivities");
        assert!(capped.max_change <= constants::MMA_MOVE_LIMIT + 1e-12);
        for value in &capped_design {
            assert!((constants::DENSITY_MIN..=constants::DENSITY_MAX).contains(value));
        }
        // The cap really priced something: measured against the step that did
        // not carry it, the half it weighs is held back and the half it does not
        // takes the material up. The two halves are compared through that
        // difference rather than against each other, because the objective
        // sensitivity already tells them apart on its own.
        let mean = |field: &[f64], parity: usize| {
            let taken: Vec<f64> = (0..n)
                .filter(|e| e % 2 == parity)
                .map(|e| field[e])
                .collect();
            taken.iter().sum::<f64>() / taken.len() as f64
        };
        let held = mean(&capped_design, 0) - mean(&plain_design, 0);
        let free = mean(&capped_design, 1) - mean(&plain_design, 1);
        assert!(
            held < 0.0 && held < free,
            "the capped half has to be held back: {held} against {free}"
        );
        // The volume is measured for real on the accepted step either way, but
        // the capped one *lands* on the target through the separable
        // approximation rather than by bisecting on that measurement, so it is
        // close rather than exact - and it is close from **below**. The
        // approximation understates a decrease and overstates an increase, both
        // by about the square of a variable's own step over its asymptote gap,
        // so a step that redistributes material as hard as this one does (half
        // the cells against the other half, at the move limit) reads 0.4680
        // against the 0.48 asked for. It is a transient: the next iteration
        // anchors on the 0.4680 that was measured, and the error follows the
        // step size to zero as the design settles.
        assert!(
            (plain.volume_fraction - 0.48).abs() < constants::MMA_VOLUME_TOLERANCE,
            "{}",
            plain.volume_fraction
        );
        println!(
            "two-constraint step: volume {:.6} against the 0.48 target",
            capped.volume_fraction
        );
        assert!(
            capped.volume_fraction < 0.48 && (capped.volume_fraction - 0.48).abs() < 0.02,
            "the capped step left the volume target: {}",
            capped.volume_fraction
        );
    }

    /// Mixed sign sensitivities, as a design dependent load produces, need no
    /// shift: a positive one lands in the upper asymptote term and pushes its
    /// variable down.
    ///
    /// A variable whose compliance sensitivity is non-negative while its volume
    /// sensitivity is positive costs material and buys nothing, so the
    /// subproblem sends it to its lower bound - not because the update cannot
    /// represent it, which is what the optimality criteria step's shift exists
    /// to prevent, but because that is the minimizer. The step stays continuous
    /// across `dC/dx = 0` either way, and the ranking of the variables that are
    /// worth keeping survives.
    #[test]
    fn positive_sensitivities_need_no_shift_and_keep_the_ranking() {
        let (grid, chain, design) = setup(5);
        let n = grid.n_cells();
        let x = vec![0.5; n];
        let dc: Vec<f64> = (0..n).map(|e| (e % 5) as f64 - 2.0).collect();
        let dv = vec![1.0 / n as f64; n];
        let mut state = State::new(n);
        let mut x_new = vec![0.0; n];
        let taken = take(
            &mut state,
            &x,
            &dc,
            &dv,
            &Constraint {
                design_cells: &design,
                target_volume_fraction: 0.4,
                chain: &chain,
                local: None,
            },
            &mut x_new,
        );
        assert_eq!(taken.shift, 0.0);
        assert!((taken.volume_fraction - 0.4).abs() < constants::MMA_VOLUME_TOLERANCE);
        let at = |sensitivity: f64| {
            x_new[(0..n)
                .find(|e| dc[*e] == sensitivity)
                .expect("a sensitivity")]
        };
        assert!(at(-2.0) > at(-1.0), "the ranking was inverted");
        assert!(at(-1.0) > at(2.0), "a positive sensitivity was favoured");
        assert!(at(2.0) >= constants::DENSITY_MIN);
        // Continuity across zero: the barely negative and the barely positive
        // sensitivity end up in the same place, rather than one of them falling
        // off a cliff the other does not see.
        assert!((at(0.0) - at(1.0)).abs() < 1e-12);
    }

    /// Cells that are not design variables are not the optimizer's to move.
    #[test]
    fn only_design_cells_move() {
        let (grid, chain, _) = setup(4);
        let n = grid.n_cells();
        // Every fourth cell is a design variable; the rest stand for the forced
        // solid and void cells and have to come back untouched.
        let design: Vec<usize> = (0..n).step_by(4).collect();
        let x: Vec<f64> = (0..n).map(|e| if e % 4 == 0 { 0.5 } else { 1.0 }).collect();
        let dc = vec![-1.0; n];
        let dv = vec![1.0 / design.len() as f64; n];
        let mut state = State::new(n);
        let mut x_new = vec![0.0; n];
        take(
            &mut state,
            &x,
            &dc,
            &dv,
            &Constraint {
                design_cells: &design,
                target_volume_fraction: 0.4,
                chain: &chain,
                local: None,
            },
            &mut x_new,
        );
        for e in 0..n {
            if e % 4 != 0 {
                assert_eq!(x_new[e], x[e], "cell {e} is not a design variable");
            }
        }
    }

    /// The asymptote rules: symmetric to start with, then shrinking on an
    /// oscillation, expanding on monotone progress, and clamped either way.
    #[test]
    fn the_asymptotes_shrink_on_oscillation_and_expand_on_progress() {
        let design = vec![0usize, 1, 2, 3];
        let mut state = State::new(design.len());
        let width = constants::DENSITY_MAX - constants::DENSITY_MIN;

        // The opening steps take the symmetric initial range.
        let x = vec![0.5; design.len()];
        state.move_asymptotes(&x, &design);
        for &e in &design {
            assert!((state.lower[e] - (0.5 - constants::MMA_ASYMPTOTE_INIT * width)).abs() < 1e-12);
            assert!((state.upper[e] - (0.5 + constants::MMA_ASYMPTOTE_INIT * width)).abs() < 1e-12);
        }
        state.steps = constants::MMA_INITIAL_ASYMPTOTE_STEPS;

        // Cell 0 reversed direction, cell 1 kept going, cell 2 stood still, and
        // cell 3 has asymptotes far enough out to hit the outer clamp.
        state.previous = vec![0.5, 0.4, 0.5, 0.5];
        state.older = vec![0.3, 0.3, 0.5, 0.3];
        state.lower = vec![0.0, 0.0, 0.0, -100.0];
        state.upper = vec![1.0, 1.0, 1.0, 100.0];
        let x = vec![0.4, 0.5, 0.5, 0.7];
        let before: Vec<(f64, f64)> = design
            .iter()
            .map(|&e| {
                (
                    state.previous[e] - state.lower[e],
                    state.upper[e] - state.previous[e],
                )
            })
            .collect();
        state.move_asymptotes(&x, &design);
        let gap = |state: &State, e: usize| (x[e] - state.lower[e], state.upper[e] - x[e]);

        for (e, factor) in [
            (0, constants::MMA_ASYMPTOTE_SHRINK),
            (1, constants::MMA_ASYMPTOTE_EXPAND),
            (2, 1.0),
        ] {
            let (down, up) = gap(&state, e);
            assert!(
                (down - factor * before[e].0).abs() < 1e-12,
                "cell {e} lower gap {down} is not {factor} x {}",
                before[e].0
            );
            assert!(
                (up - factor * before[e].1).abs() < 1e-12,
                "cell {e} upper gap {up} is not {factor} x {}",
                before[e].1
            );
        }
        let (down, up) = gap(&state, 3);
        assert!((down - constants::MMA_ASYMPTOTE_MAX * width).abs() < 1e-12);
        assert!((up - constants::MMA_ASYMPTOTE_MAX * width).abs() < 1e-12);

        // A variable that keeps reversing is damped geometrically, and stops at
        // the inner clamp rather than freezing in place.
        let oscillating = [0usize];
        let (mut older, mut previous) = (0.4, 0.6);
        let mut current = 0.0;
        for k in 0..40 {
            current = if k % 2 == 0 { 0.4 } else { 0.6 };
            state.older[0] = older;
            state.previous[0] = previous;
            let mut point = x.clone();
            point[0] = current;
            state.move_asymptotes(&point, &oscillating);
            assert!(
                state.lower[0] < current && current < state.upper[0],
                "the asymptotes closed over the design variable"
            );
            (older, previous) = (previous, current);
        }
        assert!((current - state.lower[0] - constants::MMA_ASYMPTOTE_MIN * width).abs() < 1e-12);
        assert!((state.upper[0] - current - constants::MMA_ASYMPTOTE_MIN * width).abs() < 1e-12);
    }

    /// The bounds of the subproblem: the box, the move limit, and a margin that
    /// keeps the trial point off the poles at the asymptotes.
    #[test]
    fn the_trial_bounds_respect_the_box_the_move_limit_and_the_asymptotes() {
        let design = vec![0usize, 1];
        let mut state = State::new(design.len());
        state.steps = constants::MMA_INITIAL_ASYMPTOTE_STEPS;
        let (dc, dv) = (vec![-1.0; 2], vec![0.5; 2]);

        // Wide asymptotes: the move limit binds on the cell in mid-box, and the
        // box itself binds on the one sitting at the floor.
        state.lower = vec![-1.0, -1.0];
        state.upper = vec![2.0, 2.0];
        let x = vec![0.5, 0.0];
        let variables = state.variables(&x, &dc, &dv, None, &design);
        assert!((variables[0].alpha - (0.5 - constants::MMA_MOVE_LIMIT)).abs() < 1e-12);
        assert!((variables[0].beta - (0.5 + constants::MMA_MOVE_LIMIT)).abs() < 1e-12);
        assert_eq!(variables[1].alpha, constants::DENSITY_MIN);

        // Narrow asymptotes bind instead once they are inside the move limit,
        // and the trial point stays strictly between them.
        state.lower = vec![0.45, 0.45];
        state.upper = vec![0.55, 0.55];
        let x = vec![0.5, 0.5];
        let variables = state.variables(&x, &dc, &dv, None, &design);
        let albefa = constants::MMA_BOUND_ASYMPTOTE_FRACTION;
        assert!((variables[0].alpha - (0.45 + albefa * 0.05)).abs() < 1e-12);
        assert!((variables[0].beta - (0.55 - albefa * 0.05)).abs() < 1e-12);
        assert!(variables[0].alpha > state.lower[0] && variables[0].beta < state.upper[0]);
    }

    /// A uniform design already at its target is a fixed point: nothing about
    /// the step may move it.
    #[test]
    fn a_uniform_design_at_the_target_stays_put() {
        let (grid, chain, design) = setup(5);
        let n = grid.n_cells();
        let x = vec![0.4; n];
        let dc = vec![-1.0; n];
        let dv = vec![1.0 / n as f64; n];
        let mut state = State::new(n);
        let mut x_new = vec![0.0; n];
        let taken = take(
            &mut state,
            &x,
            &dc,
            &dv,
            &Constraint {
                design_cells: &design,
                target_volume_fraction: 0.4,
                chain: &chain,
                local: None,
            },
            &mut x_new,
        );
        assert!(
            taken.max_change < 1e-6,
            "max change was {}",
            taken.max_change
        );
    }
}
