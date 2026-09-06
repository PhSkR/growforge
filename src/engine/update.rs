//! The seam between the SIMP loop and the design variable update it takes.
//!
//! Both schemes solve the same subproblem - move every design variable under
//! one volume constraint and the box `[DENSITY_MIN, DENSITY_MAX]` - and both
//! measure the volume on the *printed* densities at the far end of the density
//! chain, so the same [`Buffers`] and the same [`Step`] describe either one.
//! They differ only in the approximation they minimize: see [`super::oc`] and
//! [`super::mma`].
//!
//! [`super::beso`] plugs in here too, and it is not one of those two: it is not
//! a scheme a configuration can select but the update `[optimization.reduce]
//! method = "beso"` brings with it, it moves cells rather than densities, and it
//! answers the loop's settling question itself. The seam is the same one all the
//! same - a step in, a [`Step`] out - which is what lets the stage schedule run
//! either method without knowing which one it has.

use anyhow::Result;
use rayon::prelude::*;

use crate::config::UpdateScheme;
use crate::engine::chain::DensityChain;
use crate::engine::{beso, mma, oc};

/// Result of one design variable update.
#[derive(Debug, Clone, Copy)]
pub struct Step {
    /// Largest absolute change of any design variable.
    ///
    /// Under [`super::beso`] every change is the whole box - a cell is kept or
    /// it is cut - so what one step reports there is the share of the design
    /// cells that flipped.
    pub max_change: f64,
    /// Mean printed density over the design cells after the step.
    pub volume_fraction: f64,
    /// Lagrange multiplier of the volume constraint the bisection settled on.
    ///
    /// Under [`super::beso`] the volume constraint is met by a cut through the
    /// ranking rather than by a multiplier, and this is where that cut fell: the
    /// sensitivity of the last cell the iteration kept.
    pub lambda: f64,
    /// Sensitivity shift the self-weight guard applied, zero when it did not
    /// engage. Always zero under MMA, which represents a positive objective
    /// sensitivity natively; see [`super::mma`], and under [`super::beso`],
    /// which ranks sensitivities rather than dividing by them and needs no
    /// positive ones.
    pub shift: f64,
}

/// Buffers the volume bisection works in. `x_new` holds the updated design
/// variables when it returns, `printed` the physical densities that belong to
/// them and `filtered` the intermediate stage of the chain.
pub struct Buffers<'a> {
    /// Updated design variables.
    pub x_new: &'a mut [f64],
    /// Density-filtered intermediate densities.
    pub filtered: &'a mut [f64],
    /// Printed physical densities.
    pub printed: &'a mut [f64],
}

/// What one update step is taken against: which cells are design variables,
/// the volume fraction they have to average out to, and the chain that turns
/// them into the printed densities the volume is measured on.
pub struct Constraint<'a> {
    /// Indices of the design cells, in ascending order.
    pub design_cells: &'a [usize],
    /// Requested mean printed density over the design cells.
    pub target_volume_fraction: f64,
    /// The density chain the volume is measured through.
    pub chain: &'a DensityChain,
    /// The local volume cap, when `[optimization.local_volume]` asked for one.
    ///
    /// Only [`super::mma`] can take it - a second constraint needs a second
    /// multiplier, and the optimality criteria step has room for exactly one -
    /// and the configuration rejects every other combination before a run
    /// starts, so [`super::oc`] never sees this set.
    pub local: Option<LocalConstraint<'a>>,
}

/// The second constraint of a two-constraint step, evaluated at the design the
/// step is being taken *from*.
///
/// Everything here is measured at `x^k` rather than at a trial point, which is
/// what the moving asymptotes subproblem needs to anchor its separable
/// approximations: see [`super::mma`] for why the second constraint is trusted
/// to those approximations while the volume is not.
pub struct LocalConstraint<'a> {
    /// `g(x^k) = aggregate - max_fraction`, feasible at or below zero.
    pub value: f64,
    /// `dg/dx` over the design cells.
    pub gradient: &'a [f64],
    /// Mean printed density over the design cells at `x^k`.
    ///
    /// The volume constraint's own value at the point the approximations are
    /// built at, which the two-constraint dual needs for the same anchoring and
    /// the single-constraint bisection does not: that one measures the real
    /// volume at every trial point instead.
    pub current_volume_fraction: f64,
}

/// Mean physical density over the design cells: the quantity the volume
/// constraint is stated in, and what both schemes bisect their multiplier
/// against.
pub(crate) fn design_mean(x_phys: &[f64], design_cells: &[usize]) -> f64 {
    if design_cells.is_empty() {
        return 0.0;
    }
    let sum: f64 = design_cells.par_iter().map(|&e| x_phys[e]).sum();
    sum / design_cells.len() as f64
}

/// The selected update scheme, together with whatever state it carries between
/// iterations.
///
/// The optimality criteria step is stateless; MMA carries its asymptotes and
/// the two previous design points, which is what lets it see whether a variable
/// is oscillating; the evolutionary update carries its volume target, its
/// sensitivity history and the cells it may not cut.
#[derive(Debug)]
pub enum Updater {
    /// Optimality criteria, [`super::oc`].
    Oc,
    /// Method of moving asymptotes, [`super::mma`].
    Mma(mma::State),
    /// The bi-directional evolutionary update, [`super::beso`].
    Evolutionary(beso::State),
}

impl Updater {
    /// Build the updater a configuration selected, sized for `n_cells`.
    pub fn new(scheme: UpdateScheme, n_cells: usize) -> Updater {
        match scheme {
            UpdateScheme::Oc => Updater::Oc,
            UpdateScheme::Mma => Updater::Mma(mma::State::new(n_cells)),
        }
    }

    /// Which scheme this is, and `None` for the evolutionary update, which is
    /// not one of the schemes `[optimization] update` selects between.
    pub fn scheme(&self) -> Option<UpdateScheme> {
        match self {
            Updater::Oc => Some(UpdateScheme::Oc),
            Updater::Mma(_) => Some(UpdateScheme::Mma),
            Updater::Evolutionary(_) => None,
        }
    }

    /// Put the updater back onto a design it did not itself produce, which the
    /// stage schedule does when a refinement restarts from the design that held
    /// its target.
    ///
    /// Only the evolutionary update carries state that such a jump invalidates:
    /// its volume target is where the last cut left the design, and a design it
    /// did not cut is somewhere else. The optimality criteria step is stateless
    /// and MMA's asymptotes are a scaling it re-derives from the points it is
    /// given, so both take the new design as they would any other.
    pub fn restart(&mut self, x: &[f64], design_cells: &[usize]) {
        if let Updater::Evolutionary(state) = self {
            state.restart(x, design_cells);
        }
    }

    /// The evolutionary update's own settling verdict for this iteration, and
    /// `None` from a scheme that does not have one - whose caller then asks its
    /// own questions of the step instead.
    pub fn evolutionary_settling(
        &mut self,
        compliance: f64,
        tolerance: f64,
    ) -> Option<beso::Settling> {
        match self {
            Updater::Evolutionary(state) => Some(state.observe(compliance, tolerance)),
            _ => None,
        }
    }

    /// Take one step from `x`, given the objective sensitivity `dc` and the
    /// volume sensitivity `dv` with respect to the design variables.
    pub fn update(
        &mut self,
        x: &[f64],
        dc: &[f64],
        dv: &[f64],
        constraint: &Constraint<'_>,
        buffers: Buffers<'_>,
    ) -> Result<Step> {
        match self {
            Updater::Oc => oc::update(
                x,
                dc,
                dv,
                constraint.design_cells,
                constraint.target_volume_fraction,
                constraint.chain,
                buffers,
            ),
            Updater::Mma(state) => state.update(x, dc, dv, constraint, buffers),
            Updater::Evolutionary(state) => state.update(x, dc, constraint, buffers),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;
    use crate::grid::{CellKind, Grid};

    /// A cubic grid of design cells with an identity density chain, so a test
    /// sees the update step alone.
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
        let chain = DensityChain::new(&grid, 0.5, None);
        let design: Vec<usize> = (0..grid.n_cells()).collect();
        (grid, chain, design)
    }

    #[test]
    fn the_updater_reports_the_scheme_it_was_built_for() {
        assert_eq!(
            Updater::new(UpdateScheme::Oc, 8).scheme(),
            Some(UpdateScheme::Oc)
        );
        assert_eq!(
            Updater::new(UpdateScheme::Mma, 8).scheme(),
            Some(UpdateScheme::Mma)
        );
        assert_eq!(UpdateScheme::default(), UpdateScheme::Oc);
    }

    /// Both schemes have to hit the same volume target on the same problem, or
    /// the volume constraint would mean two different things.
    #[test]
    fn both_schemes_meet_the_volume_target() {
        let (grid, chain, design) = setup(6);
        let n = grid.n_cells();
        let x = vec![0.5; n];
        let dc: Vec<f64> = (0..n).map(|e| -(1.0 + (e % 7) as f64)).collect();
        let dv = vec![1.0 / n as f64; n];
        for scheme in [UpdateScheme::Oc, UpdateScheme::Mma] {
            let mut updater = Updater::new(scheme, n);
            let (mut x_new, mut filtered, mut printed) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            let step = updater
                .update(
                    &x,
                    &dc,
                    &dv,
                    &Constraint {
                        design_cells: &design,
                        target_volume_fraction: 0.3,
                        chain: &chain,
                        local: None,
                    },
                    Buffers {
                        x_new: &mut x_new,
                        filtered: &mut filtered,
                        printed: &mut printed,
                    },
                )
                .expect("step");
            assert!(
                (step.volume_fraction - 0.3).abs() < 1e-6,
                "{}: volume fraction {} missed the target",
                scheme.label(),
                step.volume_fraction
            );
        }
    }
}
