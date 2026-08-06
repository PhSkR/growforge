//! The local volume constraint: an upper bound on the material inside every
//! neighbourhood of the design, on top of the global volume target.
//!
//! A global target alone is met most cheaply by concentrating material - a few
//! thick members, flush surfaces, a solid interior. Capping the *local* share
//! forbids that concentration, and what the optimizer builds instead is a
//! network of many thinner members with porous interiors: the bone-like
//! structures of the infill optimization literature.
//!
//! # The field
//!
//! The local share of a design cell is a second [`DensityFilter`] - the same
//! cone kernel the density filter is, at [`LocalVolumeParams::radius_mm`] -
//! applied to the **printed** densities. Forced solid neighbours count as one
//! and void neighbours are excluded from the average, exactly as they are in the
//! density filter, so a design cell packed against a keepin pad reads the pad's
//! material as material and a cell at the domain boundary is not diluted by the
//! space outside it.
//!
//! # The aggregate
//!
//! Constraining every cell would be one constraint per design variable. They are
//! aggregated into one scalar by the p-mean
//!
//! ```text
//!   agg = ( (1/N) sum_e f_e^P )^(1/P) ,  P = LOCAL_VOLUME_AGGREGATION_EXPONENT
//! ```
//!
//! evaluated as `peak * ((1/N) sum_e (f_e/peak)^P)^(1/P)` so that no power can
//! overflow or flush to zero, on the pattern of [`super::am_filter`]'s smooth
//! maximum. The constraint is `g = agg - max_fraction <= 0`.
//!
//! **The p-mean approaches the worst cell from below**, so a design that meets
//! `agg <= max_fraction` may still hold neighbourhoods above it. That gap is a
//! property of the exponent and is documented at
//! [`constants::LOCAL_VOLUME_AGGREGATION_EXPONENT`]; the run reports the *true*
//! worst local fraction every iteration rather than the aggregate alone, so what
//! the design really holds is never hidden behind the smoothing.
//!
//! # The gradient
//!
//! With `S = (1/N) sum_e f_e^P` and `t = S / peak^P`,
//!
//! ```text
//!   d agg / d f_e = t^(1/P - 1) (1/N) (f_e / peak)^(P-1)
//! ```
//!
//! which is the same factorization by the peak, and every factor of it bounded.
//! That is then pushed through the transpose of the local kernel to reach the
//! printed densities and through the density chain's transposes to reach the
//! design variables, which is the seam
//! [`super::objective::Objective::volume_sensitivity`] uses for the global
//! constraint.
//!
//! **It is recomputed every iteration, unconditionally.** The global volume
//! sensitivity is a constant for a linear chain and the SIMP loop computes it
//! once; this one is not, whatever the chain is, because the p-mean is nonlinear
//! in the design itself: its weights follow the field.
//!
//! # What a guide wireframe does to it
//!
//! `[optimization.wireframe]` holds a density floor along its guide *after* the
//! update step has taken it, so a neighbourhood along the wire can sit above
//! `max_fraction` while the hold is on. That is the volume overshoot
//! [`super::wireframe`] already documents, in a second currency: it is accepted
//! rather than corrected, it self-corrects in the first iteration after the
//! release, and neither verdict about a settling design is asked before then.
//!
//! J. Wu, N. Aage, R. Westermann, O. Sigmund, "Infill optimization for additive
//! manufacturing - approaching bone-like porous structures", IEEE Trans Vis
//! Comput Graph 24 (2018) 1127-1140.

use rayon::prelude::*;

use crate::config::LocalVolumeParams;
use crate::constants;
use crate::engine::chain::DensityChain;
use crate::engine::filter::DensityFilter;
use crate::grid::Grid;

/// What one evaluation of the local field says about the design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measurement {
    /// The p-mean of the per-cell local fractions: the quantity the constraint
    /// is stated on, and an under-estimate of `worst`.
    pub aggregate: f64,
    /// The largest local fraction over the design cells: what the design really
    /// holds, and what the run reports.
    pub worst: f64,
}

/// The local volume constraint over the design cells of a grid.
pub struct LocalVolume {
    kernel: DensityFilter,
    max_fraction: f64,
    /// Local fraction per cell, in grid element order. Only the design cells
    /// enter the aggregate.
    field: Vec<f64>,
    /// `d agg / d f_e` scattered over the cells, the input of the kernel
    /// transpose.
    seed: Vec<f64>,
    /// The same derivative after the kernel transpose: `d agg / d printed`.
    d_printed: Vec<f64>,
    /// Intermediate the density chain's transposes pass through.
    scratch: Vec<f64>,
    /// Peak of the last measured field.
    peak: f64,
    /// `(1/N) sum_e (f_e / peak)^P` of the last measured field, which is what
    /// keeps the gradient's leading factor bounded.
    normalized: f64,
    /// The last measurement, so the constraint value and the gradient describe
    /// the same field.
    measurement: Measurement,
}

impl LocalVolume {
    /// Build the constraint for `grid`.
    pub fn new(grid: &Grid, params: LocalVolumeParams) -> LocalVolume {
        let cells = grid.n_cells();
        LocalVolume {
            kernel: DensityFilter::new(grid, params.radius_mm),
            max_fraction: params.max_fraction,
            field: vec![0.0; cells],
            seed: vec![0.0; cells],
            d_printed: vec![0.0; cells],
            scratch: vec![0.0; cells],
            peak: 0.0,
            normalized: 0.0,
            measurement: Measurement {
                aggregate: 0.0,
                worst: 0.0,
            },
        }
    }

    /// Cap the neighbourhoods are held to.
    pub fn max_fraction(&self) -> f64 {
        self.max_fraction
    }

    /// Taps of the local kernel's stencil, for reporting.
    pub fn stencil_size(&self) -> usize {
        self.kernel.stencil_size()
    }

    /// The last measured field, in grid element order.
    pub fn field(&self) -> &[f64] {
        &self.field
    }

    /// The last measurement.
    pub fn measurement(&self) -> Measurement {
        self.measurement
    }

    /// Constraint residual `g = aggregate - max_fraction` of the last
    /// measurement; feasible at or below zero.
    pub fn residual(&self) -> f64 {
        self.measurement.aggregate - self.max_fraction
    }

    /// Evaluate the local field of a printed design and aggregate it.
    pub fn measure(&mut self, printed: &[f64], design_cells: &[usize]) -> Measurement {
        self.kernel.apply(printed, &mut self.field);
        let field = &self.field;
        let count = design_cells.len();
        let worst = design_cells
            .par_iter()
            .map(|&e| field[e])
            .reduce(|| 0.0, f64::max);
        // An empty design has no neighbourhood to cap; the problem builder
        // rejects one long before this, and the arithmetic below would divide by
        // its cell count.
        let (aggregate, normalized) = if count == 0 || worst <= 0.0 {
            (0.0, 0.0)
        } else {
            let exponent = constants::LOCAL_VOLUME_AGGREGATION_EXPONENT;
            let normalized = design_cells
                .par_iter()
                .map(|&e| (field[e] / worst).powf(exponent))
                .sum::<f64>()
                / count as f64;
            (worst * normalized.powf(1.0 / exponent), normalized)
        };
        self.peak = worst;
        self.normalized = normalized;
        self.measurement = Measurement { aggregate, worst };
        self.measurement
    }

    /// Sensitivity of the aggregate with respect to the design variables, at the
    /// point the last [`LocalVolume::measure`] was taken at.
    ///
    /// `filtered` and `printed` are the pair [`DensityChain::apply`] produced for
    /// that point, and `out` receives `d agg / dx` over the design cells.
    pub fn sensitivity(
        &mut self,
        chain: &DensityChain,
        filtered: &[f64],
        printed: &[f64],
        design_cells: &[usize],
        out: &mut [f64],
    ) {
        self.seed.iter_mut().for_each(|v| *v = 0.0);
        if self.peak > 0.0 && !design_cells.is_empty() {
            let exponent = constants::LOCAL_VOLUME_AGGREGATION_EXPONENT;
            let scale = self.normalized.powf(1.0 / exponent - 1.0) / design_cells.len() as f64;
            for &e in design_cells {
                self.seed[e] = scale * (self.field[e] / self.peak).powf(exponent - 1.0);
            }
        }
        self.kernel.apply_transpose(&self.seed, &mut self.d_printed);
        chain.apply_transpose(filtered, printed, &self.d_printed, &mut self.scratch, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BuildDirection;
    use crate::geometry::Aabb;
    use crate::grid::CellKind;

    fn design_grid(n: usize) -> Grid {
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
        grid
    }

    fn design_cells(grid: &Grid) -> Vec<usize> {
        (0..grid.n_cells())
            .filter(|&e| grid.cells[e] == CellKind::Design)
            .collect()
    }

    /// A non-uniform starting design, so no cell of the field sits at a
    /// symmetric point that could hide a sign error.
    fn varied(grid: &Grid, cells: &[usize]) -> Vec<f64> {
        let mut x = vec![0.0; grid.n_cells()];
        for (n, &e) in cells.iter().enumerate() {
            x[e] = 0.2 + 0.6 * ((n % 11) as f64 / 11.0);
        }
        x
    }

    fn params(radius_mm: f64) -> LocalVolumeParams {
        LocalVolumeParams {
            max_fraction: constants::LOCAL_VOLUME_MAX_FRACTION,
            radius_mm,
        }
    }

    /// The aggregate of a design, through the chain, as a function of the design
    /// variables alone: what the finite difference below differentiates.
    fn aggregate_at(
        local: &mut LocalVolume,
        chain: &DensityChain,
        x: &[f64],
        cells: &[usize],
    ) -> f64 {
        let n = x.len();
        let (mut filtered, mut printed) = (vec![0.0; n], vec![0.0; n]);
        chain.apply(x, &mut filtered, &mut printed);
        local.measure(&printed, cells).aggregate
    }

    /// The gradient against central differences of the aggregate itself, on the
    /// conventions of the objective's own check: a strided walk through the
    /// design cells, and a relative error against the largest entry.
    fn gradient_error(chain: &DensityChain, radius_mm: f64, grid: &Grid) -> f64 {
        let cells = design_cells(grid);
        let n = grid.n_cells();
        let x = varied(grid, &cells);
        let mut local = LocalVolume::new(grid, params(radius_mm));

        let (mut filtered, mut printed) = (vec![0.0; n], vec![0.0; n]);
        chain.apply(&x, &mut filtered, &mut printed);
        local.measure(&printed, &cells);
        let mut adjoint = vec![0.0; n];
        local.sensitivity(chain, &filtered, &printed, &cells, &mut adjoint);

        let step = 1e-6;
        let scale = adjoint.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "the constraint has to depend on the design");
        let mut worst: f64 = 0.0;
        for &e in cells.iter().step_by(7) {
            let mut shifted = x.clone();
            shifted[e] = x[e] + step;
            let plus = aggregate_at(&mut local, chain, &shifted, &cells);
            shifted[e] = x[e] - step;
            let minus = aggregate_at(&mut local, chain, &shifted, &cells);
            let numeric = (plus - minus) / (2.0 * step);
            worst = worst.max((numeric - adjoint[e]).abs() / scale);
        }
        worst
    }

    #[test]
    fn the_gradient_matches_finite_differences_through_the_plain_chain() {
        let grid = design_grid(8);
        let chain = DensityChain::new(&grid, 1.5, None);
        let error = gradient_error(&chain, 3.5, &grid);
        println!("plain chain: relative gradient error {error:.3e}");
        assert!(error < 1e-7, "relative gradient error {error}");
    }

    /// The same through the self-supporting filter, which is where the chain
    /// transpose stops being a constant operator.
    #[test]
    fn the_gradient_matches_finite_differences_through_the_overhang_chain() {
        let grid = design_grid(8);
        let chain = DensityChain::new(&grid, 1.5, Some(BuildDirection::ZPlus));
        let error = gradient_error(&chain, 3.5, &grid);
        println!("overhang chain: relative gradient error {error:.3e}");
        assert!(error < 1e-7, "relative gradient error {error}");
    }

    /// And with keepin and keepout cells in the neighbourhood, which is what
    /// decides whether the local fraction reads a pad as material.
    #[test]
    fn the_gradient_matches_finite_differences_around_forced_cells() {
        let mut grid = design_grid(7);
        for k in 0..7 {
            let (solid, void) = (grid.cell_index(1, 1, k), grid.cell_index(5, 5, k));
            grid.cells[solid] = CellKind::Solid;
            grid.cells[void] = CellKind::Void;
        }
        let chain = DensityChain::new(&grid, 1.5, None);
        let error = gradient_error(&chain, 3.0, &grid);
        println!("forced cells: relative gradient error {error:.3e}");
        assert!(error < 1e-7, "relative gradient error {error}");
    }

    /// The aggregate itself: bracketed by the mean and the worst cell, exact on
    /// a uniform field, and equal to the density it reads.
    #[test]
    fn the_aggregate_sits_between_the_mean_and_the_worst_cell() {
        let grid = design_grid(8);
        let cells = design_cells(&grid);
        let mut local = LocalVolume::new(&grid, params(3.5));

        // A uniform field: every neighbourhood holds exactly that share, so all
        // three numbers agree and the aggregate is exact.
        let uniform = vec![0.37; grid.n_cells()];
        let measured = local.measure(&uniform, &cells);
        assert!((measured.aggregate - 0.37).abs() < 1e-12, "{measured:?}");
        assert!((measured.worst - 0.37).abs() < 1e-12, "{measured:?}");

        // A field with a solid block in one corner: the aggregate has to be
        // above the mean of the local fractions and below their worst.
        let mut lumpy = vec![0.0; grid.n_cells()];
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    lumpy[grid.cell_index(i, j, k)] = 1.0;
                }
            }
        }
        let measured = local.measure(&lumpy, &cells);
        let mean: f64 = cells.iter().map(|&e| local.field()[e]).sum::<f64>() / cells.len() as f64;
        assert!(
            mean < measured.aggregate && measured.aggregate < measured.worst,
            "mean {mean}, {measured:?}"
        );
        // The under-estimate is the documented one and not a rounding artefact:
        // the p-mean cannot fall below the worst cell by more than the count's
        // own P-th root.
        let floor = measured.worst
            * (cells.len() as f64).powf(-1.0 / constants::LOCAL_VOLUME_AGGREGATION_EXPONENT);
        assert!(measured.aggregate >= floor, "{measured:?} below {floor}");
    }

    /// An empty field is flat: no aggregate, no gradient, and nothing that
    /// divides by a peak of zero.
    #[test]
    fn an_empty_design_has_no_aggregate_and_no_gradient() {
        let grid = design_grid(5);
        let cells = design_cells(&grid);
        let chain = DensityChain::new(&grid, 1.5, None);
        let mut local = LocalVolume::new(&grid, params(3.0));
        let empty = vec![0.0; grid.n_cells()];
        let measured = local.measure(&empty, &cells);
        assert_eq!(measured.aggregate, 0.0);
        assert_eq!(measured.worst, 0.0);
        assert_eq!(local.residual(), -local.max_fraction());

        let mut out = vec![f64::NAN; grid.n_cells()];
        local.sensitivity(&chain, &empty, &empty, &cells, &mut out);
        assert!(out.iter().all(|v| *v == 0.0), "{out:?}");
    }

    /// Every entry of the gradient is non-negative: material can only raise a
    /// local fraction. The two-constraint dual bisects on that monotonicity.
    #[test]
    fn the_gradient_is_non_negative_everywhere() {
        let grid = design_grid(7);
        let cells = design_cells(&grid);
        let chain = DensityChain::new(&grid, 1.5, None);
        let mut local = LocalVolume::new(&grid, params(3.0));
        let x = varied(&grid, &cells);
        let n = grid.n_cells();
        let (mut filtered, mut printed) = (vec![0.0; n], vec![0.0; n]);
        chain.apply(&x, &mut filtered, &mut printed);
        local.measure(&printed, &cells);
        let mut out = vec![0.0; n];
        local.sensitivity(&chain, &filtered, &printed, &cells, &mut out);
        assert!(out.iter().all(|v| *v >= 0.0), "a negative sensitivity");
        assert!(out.iter().any(|v| *v > 0.0), "the constraint is flat");
    }
}
