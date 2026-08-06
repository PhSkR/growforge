//! Optimality criteria update for the volume constrained compliance problem.
//!
//! The multiplicative update `x <- x * (-dc / (lambda dv))^eta` clipped to a
//! move limit is the classical OC step; `lambda` is found by bisection so that
//! the mean physical density over the design cells lands on the requested mass
//! fraction. The volume is measured on the *printed* densities, which is why the
//! whole density chain has to be applied inside the bisection.
//!
//! This is the default scheme and the reference one: the recorded regression
//! trajectories and the shipped examples were written against it. The
//! alternative, selected with `[optimization] update = "mma"`, is
//! [`super::mma`]; the seam they share is [`super::update`].

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::constants;
use crate::engine::chain::DensityChain;
use crate::engine::update::{Buffers, Step, design_mean};

/// Evaluate the trial design for one multiplier.
///
/// A design variable whose volume sensitivity is zero has a zero row in the
/// whole density chain: it changes no physical density, so it changes neither
/// the volume nor the compliance, and its objective sensitivity has vanished
/// with it. The self-supporting filter creates exactly that situation for
/// blueprint material it erases. Such a variable decays instead of floating,
/// because a value nothing depends on today becomes material the moment support
/// grows underneath it - a design that ambushes itself several iterations later.
/// A linear chain never produces a zero row for a design cell, so this branch
/// only ever fires with the additive manufacturing filter in play.
fn trial(x: &[f64], dc: &[f64], dv: &[f64], design_cells: &[usize], lambda: f64, out: &mut [f64]) {
    let move_limit = constants::OC_MOVE_LIMIT;
    let eta = constants::OC_DAMPING;
    let updates: Vec<(usize, f64)> = design_cells
        .par_iter()
        .map(|&e| {
            let xe = x[e];
            let base = if dv[e] > 0.0 {
                if lambda > 0.0 {
                    (-dc[e] / (lambda * dv[e])).max(0.0)
                } else {
                    f64::INFINITY
                }
            } else {
                0.0
            };
            let candidate = if base.is_finite() {
                xe * base.powf(eta)
            } else {
                constants::DENSITY_MAX
            };
            let lower = (xe - move_limit).max(constants::DENSITY_MIN);
            let upper = (xe + move_limit).min(constants::DENSITY_MAX);
            (e, candidate.clamp(lower, upper))
        })
        .collect();
    for (e, v) in updates {
        out[e] = v;
    }
}

/// Shift the objective sensitivities along the volume gradient until every one
/// of them is strictly negative, and report the shift.
///
/// A design dependent load such as self weight can make `dC/dx_e` positive,
/// which the multiplicative OC ratio cannot represent: it would collapse the
/// element to its lower move limit no matter how the rest of the design looks.
/// Replacing `dC/dx` by `dC/dx - mu dV/dx` fixes that, and because the shift is
/// a multiple of the *volume* gradient it only renames the Lagrange multiplier
/// (`lambda -> lambda + mu`): the stationarity conditions of the volume
/// constrained problem, and therefore the optimum, are untouched. What changes
/// is the size of the step, which the shift damps - the behaviour the
/// self-weight literature asks for (Bruyneel and Duysinx, "Note on topology
/// optimization of continuum structures including self-weight", Struct
/// Multidisc Optim 30 (2005) 245-256).
///
/// Returns `None` when no sensitivity is positive, so a problem without a design
/// dependent load takes exactly the update it always did.
fn shift_sensitivities(dc: &[f64], dv: &[f64], design_cells: &[usize]) -> Option<(Vec<f64>, f64)> {
    let ratio = |e: usize| dc[e] / dv[e];
    let usable = || design_cells.par_iter().copied().filter(|&e| dv[e] > 0.0);
    let worst = usable().map(ratio).reduce(|| f64::NEG_INFINITY, f64::max);
    if worst <= 0.0 || worst.is_nan() {
        return None;
    }
    let scale = usable().map(|e| ratio(e).abs()).reduce(|| 0.0, f64::max);
    let mu = worst + constants::OC_SELF_WEIGHT_SHIFT_MARGIN * scale;
    let mut shifted = dc.to_vec();
    for &e in design_cells {
        if dv[e] > 0.0 {
            shifted[e] = dc[e] - mu * dv[e];
        }
    }
    Some((shifted, mu))
}

/// Perform one optimality criteria update.
///
/// `x` holds the current design variables, `dc` and `dv` the objective and
/// volume sensitivities with respect to the design variables (already pushed
/// through the chain transposes).
pub fn update(
    x: &[f64],
    dc: &[f64],
    dv: &[f64],
    design_cells: &[usize],
    target_volume_fraction: f64,
    chain: &DensityChain,
    buffers: Buffers<'_>,
) -> Result<Step> {
    let Buffers {
        x_new,
        filtered,
        printed,
    } = buffers;
    x_new.copy_from_slice(x);

    let shifted = shift_sensitivities(dc, dv, design_cells);
    let (dc, shift) = match &shifted {
        Some((values, mu)) => (values.as_slice(), *mu),
        None => (dc, 0.0),
    };

    let mut lower = constants::OC_LAMBDA_LOWER;
    let mut upper = constants::OC_LAMBDA_UPPER;
    let mut lambda = 0.5 * (lower + upper);
    let mut volume_fraction = 0.0;

    for step in 0..constants::OC_BISECTION_MAX_STEPS {
        lambda = 0.5 * (lower + upper);
        trial(x, dc, dv, design_cells, lambda, x_new);
        chain.apply(x_new, filtered, printed);
        volume_fraction = design_mean(printed, design_cells);

        if !volume_fraction.is_finite() {
            bail!(
                "optimality criteria update produced a non-finite volume fraction at bisection step {step}"
            );
        }
        if (volume_fraction - target_volume_fraction).abs() < constants::OC_VOLUME_TOLERANCE {
            break;
        }
        if volume_fraction > target_volume_fraction {
            lower = lambda;
        } else {
            upper = lambda;
        }
        let width = upper - lower;
        let scale = upper + lower;
        if scale <= 0.0 || width / scale < constants::OC_BISECTION_TOL {
            break;
        }
    }

    let max_change = design_cells
        .par_iter()
        .map(|&e| (x_new[e] - x[e]).abs())
        .reduce(|| 0.0, f64::max);

    Ok(Step {
        max_change,
        volume_fraction,
        lambda,
        shift,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;
    use crate::grid::{CellKind, Grid};

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
        // A filter radius below one voxel makes the filter the identity, which
        // isolates the OC step in these tests.
        let chain = DensityChain::new(&grid, 0.5, None);
        let design: Vec<usize> = (0..grid.n_cells()).collect();
        (grid, chain, design)
    }

    fn buffers<'a>(
        x_new: &'a mut [f64],
        filtered: &'a mut [f64],
        printed: &'a mut [f64],
    ) -> Buffers<'a> {
        Buffers {
            x_new,
            filtered,
            printed,
        }
    }

    #[test]
    fn bisection_hits_the_target_volume() {
        let (grid, chain, design) = setup(6);
        let n = grid.n_cells();
        let x = vec![0.5; n];
        // A sensitivity field that varies across the domain.
        let dc: Vec<f64> = (0..n).map(|e| -(1.0 + (e % 7) as f64)).collect();
        let dv = vec![1.0 / n as f64; n];
        let (mut x_new, mut filtered, mut printed) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let step = update(
            &x,
            &dc,
            &dv,
            &design,
            0.3,
            &chain,
            buffers(&mut x_new, &mut filtered, &mut printed),
        )
        .expect("oc step");
        assert!(
            (step.volume_fraction - 0.3).abs() < 1e-6,
            "volume fraction {} missed the target",
            step.volume_fraction
        );
        assert_eq!(step.shift, 0.0);
    }

    #[test]
    fn move_limit_and_box_constraints_are_respected() {
        let (grid, chain, design) = setup(4);
        let n = grid.n_cells();
        let x = vec![0.5; n];
        let dc: Vec<f64> = (0..n).map(|e| -(1.0 + e as f64)).collect();
        let dv = vec![1.0 / n as f64; n];
        let (mut x_new, mut filtered, mut printed) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let step = update(
            &x,
            &dc,
            &dv,
            &design,
            0.6,
            &chain,
            buffers(&mut x_new, &mut filtered, &mut printed),
        )
        .expect("oc step");
        assert!(step.max_change <= constants::OC_MOVE_LIMIT + 1e-12);
        for v in &x_new {
            assert!(*v >= constants::DENSITY_MIN - 1e-12);
            assert!(*v <= constants::DENSITY_MAX + 1e-12);
        }
    }

    #[test]
    fn uniform_sensitivities_keep_a_uniform_design() {
        let (grid, chain, design) = setup(5);
        let n = grid.n_cells();
        let x = vec![0.4; n];
        let dc = vec![-1.0; n];
        let dv = vec![1.0 / n as f64; n];
        let (mut x_new, mut filtered, mut printed) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let step = update(
            &x,
            &dc,
            &dv,
            &design,
            0.4,
            &chain,
            buffers(&mut x_new, &mut filtered, &mut printed),
        )
        .expect("oc step");
        assert!(step.max_change < 1e-6, "max change was {}", step.max_change);
    }

    #[test]
    fn a_positive_sensitivity_engages_the_shift_and_keeps_the_ranking() {
        let (grid, chain, design) = setup(5);
        let n = grid.n_cells();
        let x = vec![0.5; n];
        // Mixed signs, as a self-weight problem produces.
        let dc: Vec<f64> = (0..n).map(|e| (e % 5) as f64 - 2.0).collect();
        let dv = vec![1.0 / n as f64; n];
        let (mut x_new, mut filtered, mut printed) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let step = update(
            &x,
            &dc,
            &dv,
            &design,
            0.5,
            &chain,
            buffers(&mut x_new, &mut filtered, &mut printed),
        )
        .expect("oc step");
        assert!(step.shift > 0.0, "the shift did not engage");
        assert!(
            (step.volume_fraction - 0.5).abs() < 1e-6,
            "volume fraction {} missed the target",
            step.volume_fraction
        );
        // Without the shift every element with dc >= 0 would have collapsed to
        // its lower move limit; with it they are merely the least favoured.
        let worst = x_new[(0..n).find(|e| dc[*e] == 2.0).expect("a positive one")];
        let best = x_new[(0..n).find(|e| dc[*e] == -2.0).expect("a negative one")];
        assert!(worst > constants::DENSITY_MIN, "the shift left a hard zero");
        assert!(best > worst, "the shift inverted the ranking");
    }

    #[test]
    fn the_shift_leaves_a_wholly_negative_sensitivity_field_alone() {
        let n = 8;
        let dc: Vec<f64> = (0..n).map(|e| -(1.0 + e as f64)).collect();
        let dv = vec![0.125; n];
        let design: Vec<usize> = (0..n).collect();
        assert!(shift_sensitivities(&dc, &dv, &design).is_none());

        // One non-negative entry is enough to engage it.
        let mut dc = dc;
        dc[3] = 0.5;
        let (shifted, mu) = shift_sensitivities(&dc, &dv, &design).expect("a shift");
        assert!(mu > 0.0);
        assert!(
            shifted.iter().all(|v| *v < 0.0),
            "the shift left a non-negative sensitivity: {shifted:?}"
        );
        // The shift is exactly a multiple of the volume gradient, so gaps
        // between sensitivities survive it untouched.
        for e in 1..n {
            assert!(
                ((shifted[e] - shifted[e - 1]) - (dc[e] - dc[e - 1])).abs() < 1e-12,
                "the shift is not uniform"
            );
        }
    }
}
