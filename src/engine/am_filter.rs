//! Langelaar's additive manufacturing filter: the self-supporting (overhang)
//! constraint.
//!
//! The filter sweeps the grid layer by layer along the build direction. The
//! first layer sits on the build plate and is printed exactly as designed;
//! every later layer's printed density is
//! `smin(blueprint, smax(printed densities of the supporting region))`, so an
//! element can only carry material where the layer below already does. On a
//! cubic grid the supporting region is the element directly below plus its four
//! lateral face neighbours in that layer, which is what makes the constraint a
//! 45 degree self-supporting angle. The angle is a property of that stencil, not
//! a tunable: a different angle needs a different stencil.
//!
//! Forced solid (keepin) cells count as printable support at density one and
//! void cells as zero; neither is touched by the sweep, because neither is a
//! design variable.
//!
//! Sensitivities travel back through the exact adjoint of the sweep, which is
//! the same sweep run from the top layer down.
//!
//! M. Langelaar, "An additive manufacturing filter for topology optimization of
//! print-ready designs", Struct Multidisc Optim 55 (2017) 871-883.
//!
//! Two documented approximations come with the smoothing:
//! * the smooth minimum overestimates the true minimum by up to
//!   `sqrt(AM_SMOOTH_MIN_EPS) / 2`, so a printed density can sit that far above
//!   one where the blueprint is fully solid, and
//! * the P-norm smooth maximum is made exact at `AM_SMOOTH_MAX_ANCHOR` and
//!   overestimates elsewhere, most visibly near density one.

use rayon::prelude::*;

use crate::config::BuildDirection;
use crate::constants;
use crate::grid::{CellKind, Grid};

/// P-norm exponent of the smooth maximum. An integer exponent turns the power
/// into a handful of multiplications instead of a `powf` call, which matters
/// because the filter runs once per bisection step of the volume constraint.
const P: i32 = constants::AM_SMOOTH_MAX_EXPONENT;

/// Smooth maximum of a supporting region and the sum of powers behind it.
///
/// `(sum_i v_i^P)^(1/q)`, where `q` is the corrected exponent that makes the
/// estimate exact for an all-equal region at [`constants::AM_SMOOTH_MAX_ANCHOR`]
/// instead of overestimating it by `n^(1/P)`. The sum comes back with the value
/// because the adjoint needs it to weight the supports.
fn smooth_max(supports: &[f64; constants::AM_SUPPORT_COUNT], q: f64) -> (f64, f64) {
    let sum: f64 = supports.iter().map(|v| v.max(0.0).powi(P)).sum();
    if sum > 0.0 {
        (sum.powf(1.0 / q), sum)
    } else {
        // Every support is zero: the maximum is zero and so is its gradient,
        // which is also the limit of the smooth form from any direction.
        (0.0, 0.0)
    }
}

/// The two factors the adjoint of [`smooth_max`] is built from, scaled so that
/// neither can overflow.
///
/// The derivative is
/// `d smax / d v_i = (P/q) S^(1/q - 1) v_i^(P-1)` with `S = sum_j v_j^P`, and
/// written that way it cannot be evaluated: a region of densities around 1e-8
/// drives `S` into the subnormals while the derivative itself is still of order
/// one, so `S^(1/q - 1)` overflows and the `inf` it produces meets the `0` of
/// `v_i^(P-1)` further down the chain. Factoring out the largest member `m`,
///
/// ```text
///   d smax / d v_i = (P/q) m^(P/q - 1) t^(1/q - 1) (v_i / m)^(P-1)
///   t = sum_j (v_j / m)^P  in [1, n]
/// ```
///
/// leaves every factor bounded for densities in the unit box: this returns `m`
/// and the leading `(P/q) m^(P/q - 1) t^(1/q - 1)`, and the caller supplies
/// `(v_i / m)^(P-1)`, which is at most one because `m` is the largest member.
/// A region that is entirely zero has a zero gradient and reports `m = 0`.
fn smooth_max_gradient_scale(supports: &[f64; constants::AM_SUPPORT_COUNT], q: f64) -> (f64, f64) {
    let peak = supports.iter().fold(0.0f64, |peak, v| peak.max(*v));
    if peak <= 0.0 {
        return (0.0, 0.0);
    }
    let normalized: f64 = supports.iter().map(|v| (v.max(0.0) / peak).powi(P)).sum();
    let power = P as f64 / q;
    (
        peak,
        power * peak.powf(power - 1.0) * normalized.powf(1.0 / q - 1.0),
    )
}

/// Smooth minimum of two values.
fn smooth_min(a: f64, b: f64) -> f64 {
    let eps = constants::AM_SMOOTH_MIN_EPS;
    let d = a - b;
    0.5 * ((a + b) - (d * d + eps).sqrt() + eps.sqrt())
}

/// Partial derivatives of [`smooth_min`] with respect to both arguments. They
/// sum to one, as a minimum's subgradients must.
fn smooth_min_gradient(a: f64, b: f64) -> (f64, f64) {
    let eps = constants::AM_SMOOTH_MIN_EPS;
    let d = a - b;
    let r = (d * d + eps).sqrt();
    (0.5 * (1.0 - d / r), 0.5 * (1.0 + d / r))
}

/// How the adjoint splits one element's sensitivity while the reverse sweep
/// passes through it: the share its own blueprint keeps, and the share the
/// smooth maximum hands to its supporting region, scaled by the peak of that
/// region (see [`smooth_max_gradient_scale`]).
#[derive(Debug, Clone, Copy, Default)]
struct Split {
    /// Sensitivity with respect to this element's blueprint density.
    blueprint: f64,
    /// Leading factor of the sensitivity handed to every supporting element.
    weight: f64,
    /// Largest printed density in the supporting region; zero when the region
    /// carries no material at all, which is also when `weight` is zero.
    peak: f64,
}

/// The additive manufacturing filter over the cells of a grid.
pub struct AmFilter {
    /// Element counts along x, y, z.
    dims: [usize; 3],
    /// Axis the layers stack along.
    axis: usize,
    /// The two remaining axes, in increasing order.
    lateral: [usize; 2],
    /// True when layer 0 is at the low end of `axis`.
    positive: bool,
    /// Corrected P-norm exponent `q = P + ln(n) / ln(xi_0)`.
    q: f64,
    kinds: Vec<CellKind>,
}

impl AmFilter {
    /// Build the filter for `grid` and a build direction.
    pub fn new(grid: &Grid, direction: BuildDirection) -> AmFilter {
        let axis = direction.axis();
        let lateral = match axis {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let n = constants::AM_SUPPORT_COUNT as f64;
        AmFilter {
            dims: [grid.nx, grid.ny, grid.nz],
            axis,
            lateral,
            positive: direction.is_positive(),
            q: P as f64 + n.ln() / constants::AM_SMOOTH_MAX_ANCHOR.ln(),
            kinds: grid.cells.clone(),
        }
    }

    /// Number of layers the sweep walks through.
    pub fn layer_count(&self) -> usize {
        self.dims[self.axis]
    }

    /// Element counts across a layer, along the two lateral axes.
    pub fn lateral_dims(&self) -> (usize, usize) {
        (self.dims[self.lateral[0]], self.dims[self.lateral[1]])
    }

    /// Coordinate along the build axis of layer `layer`, counting from the
    /// build plate.
    fn along(&self, layer: usize) -> usize {
        if self.positive {
            layer
        } else {
            self.dims[self.axis] - 1 - layer
        }
    }

    /// Linear cell index from a build axis coordinate and a lateral position.
    fn cell(&self, along: usize, u: usize, v: usize) -> usize {
        let mut c = [0usize; 3];
        c[self.axis] = along;
        c[self.lateral[0]] = u;
        c[self.lateral[1]] = v;
        c[0] + self.dims[0] * (c[1] + self.dims[1] * c[2])
    }

    /// The printed density a non-design cell is pinned at, or `value` for a
    /// design cell.
    fn pinned(&self, cell: usize, value: f64) -> f64 {
        match self.kinds[cell] {
            CellKind::Void => constants::DENSITY_MIN,
            CellKind::Solid => constants::DENSITY_MAX,
            CellKind::Design => value,
        }
    }

    /// The five stencil positions of a lateral coordinate, padded outside the
    /// grid with zero so the supporting region always has the same size and the
    /// corrected exponent stays constant across the whole layer.
    ///
    /// The stencil is its own mirror: the elements supported by `(u, v)` in the
    /// layer above sit at exactly these positions too, which is what lets the
    /// adjoint gather instead of scatter.
    fn stencil(
        u: usize,
        v: usize,
        nu: usize,
        nv: usize,
    ) -> [Option<usize>; constants::AM_SUPPORT_COUNT] {
        let at = |du: isize, dv: isize| -> Option<usize> {
            let su = u as isize + du;
            let sv = v as isize + dv;
            if su < 0 || sv < 0 || su >= nu as isize || sv >= nv as isize {
                None
            } else {
                Some(su as usize + nu * sv as usize)
            }
        };
        [at(0, 0), at(-1, 0), at(1, 0), at(0, -1), at(0, 1)]
    }

    /// Values of the supporting region of `(u, v)`, taken from a lateral slab.
    fn supports(
        slab: &[f64],
        u: usize,
        v: usize,
        nu: usize,
        nv: usize,
    ) -> [f64; constants::AM_SUPPORT_COUNT] {
        let mut out = [0.0; constants::AM_SUPPORT_COUNT];
        for (slot, position) in out.iter_mut().zip(Self::stencil(u, v, nu, nv)) {
            if let Some(index) = position {
                *slot = slab[index];
            }
        }
        out
    }

    /// Sweep the blueprint densities into printed densities.
    ///
    /// `blueprint` and `printed` are full length cell arrays; `printed` receives
    /// zero for void cells, one for forced solid cells and the filtered value
    /// for design cells.
    pub fn apply(&self, blueprint: &[f64], printed: &mut [f64]) {
        debug_assert_eq!(blueprint.len(), self.kinds.len());
        debug_assert_eq!(printed.len(), self.kinds.len());
        let (nu, nv) = self.lateral_dims();
        let mut below = vec![0.0; nu * nv];
        let mut layer_values = vec![0.0; nu * nv];

        for layer in 0..self.layer_count() {
            let along = self.along(layer);
            layer_values
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, slot)| {
                    let (u, v) = (index % nu, index / nu);
                    let cell = self.cell(along, u, v);
                    *slot = if layer == 0 {
                        // The build plate supports everything on it.
                        self.pinned(cell, blueprint[cell])
                    } else {
                        match self.kinds[cell] {
                            CellKind::Void => constants::DENSITY_MIN,
                            CellKind::Solid => constants::DENSITY_MAX,
                            CellKind::Design => {
                                let (support, _) =
                                    smooth_max(&Self::supports(&below, u, v, nu, nv), self.q);
                                smooth_min(blueprint[cell], support)
                            }
                        }
                    };
                });
            for (index, value) in layer_values.iter().enumerate() {
                printed[self.cell(along, index % nu, index / nu)] = *value;
            }
            std::mem::swap(&mut below, &mut layer_values);
        }
    }

    /// Chain a sensitivity with respect to the printed densities back to the
    /// blueprint densities, by running the sweep in reverse.
    ///
    /// `blueprint` and `printed` are the pair [`AmFilter::apply`] produced and
    /// fix the point the Jacobian is taken at. Only design cells receive a
    /// derivative; the pinned cells terminate the chain, because their printed
    /// density does not depend on any design variable.
    pub fn apply_transpose(
        &self,
        blueprint: &[f64],
        printed: &[f64],
        d_printed: &[f64],
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), self.kinds.len());
        out.fill(0.0);
        let (nu, nv) = self.lateral_dims();
        let layers = self.layer_count();

        // Sensitivity accumulated on the printed densities of the layer being
        // processed, seeded from the direct dependence of the objective.
        let seed = |along: usize| -> Vec<f64> {
            (0..nu * nv)
                .into_par_iter()
                .map(|index| {
                    let cell = self.cell(along, index % nu, index / nu);
                    if self.kinds[cell] == CellKind::Design {
                        d_printed[cell]
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let mut lambda = seed(self.along(layers - 1));

        for layer in (1..layers).rev() {
            let along = self.along(layer);
            let below = self.along(layer - 1);
            // Per element of this layer: how much of its sensitivity the smooth
            // minimum hands to its own blueprint, and the weight the smooth
            // maximum spreads over its supporting region.
            let split: Vec<Split> = (0..nu * nv)
                .into_par_iter()
                .map(|index| {
                    let (u, v) = (index % nu, index / nu);
                    let cell = self.cell(along, u, v);
                    if self.kinds[cell] != CellKind::Design {
                        return Split::default();
                    }
                    let supports = self.supports_below(printed, below, u, v, nu, nv);
                    let (support, _) = smooth_max(&supports, self.q);
                    let (d_blueprint, d_support) = smooth_min_gradient(blueprint[cell], support);
                    let (peak, scale) = smooth_max_gradient_scale(&supports, self.q);
                    Split {
                        blueprint: lambda[index] * d_blueprint,
                        weight: lambda[index] * d_support * scale,
                        peak,
                    }
                })
                .collect();
            for (index, share) in split.iter().enumerate() {
                let cell = self.cell(along, index % nu, index / nu);
                if self.kinds[cell] == CellKind::Design {
                    out[cell] = share.blueprint;
                }
            }

            lambda = (0..nu * nv)
                .into_par_iter()
                .map(|index| {
                    let (u, v) = (index % nu, index / nu);
                    let cell = self.cell(below, u, v);
                    if self.kinds[cell] != CellKind::Design {
                        return 0.0;
                    }
                    let value = printed[cell].max(0.0);
                    let supported: f64 = Self::stencil(u, v, nu, nv)
                        .into_iter()
                        .flatten()
                        .map(|above| {
                            let share = &split[above];
                            if share.peak > 0.0 {
                                share.weight * (value / share.peak).powi(P - 1)
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    d_printed[cell] + supported
                })
                .collect();
        }

        // The build plate layer prints its blueprint unchanged.
        let along = self.along(0);
        for (index, value) in lambda.iter().enumerate() {
            let cell = self.cell(along, index % nu, index / nu);
            if self.kinds[cell] == CellKind::Design {
                out[cell] = *value;
            }
        }
    }

    /// Printed densities of the supporting region of `(u, v)`, read straight out
    /// of the full cell array of the layer below.
    fn supports_below(
        &self,
        printed: &[f64],
        below: usize,
        u: usize,
        v: usize,
        nu: usize,
        nv: usize,
    ) -> [f64; constants::AM_SUPPORT_COUNT] {
        let mut out = [0.0; constants::AM_SUPPORT_COUNT];
        for (slot, position) in out.iter_mut().zip(Self::stencil(u, v, nu, nv)) {
            if let Some(index) = position {
                *slot = printed[self.cell(below, index % nu, index / nu)];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;

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

    fn filtered(grid: &Grid, blueprint: &[f64], direction: BuildDirection) -> Vec<f64> {
        let filter = AmFilter::new(grid, direction);
        let mut printed = vec![0.0; grid.n_cells()];
        filter.apply(blueprint, &mut printed);
        printed
    }

    /// Every build direction, so an axis or a sign that stops being honoured
    /// fails a test rather than quietly printing the wrong way up.
    const ALL_DIRECTIONS: [BuildDirection; 6] = [
        BuildDirection::XPlus,
        BuildDirection::XMinus,
        BuildDirection::YPlus,
        BuildDirection::YMinus,
        BuildDirection::ZPlus,
        BuildDirection::ZMinus,
    ];

    /// Cell index of a position expressed relative to the build direction:
    /// `layer` counts from the build plate, `u` and `v` index the two lateral
    /// axes in increasing order. Mirrors what the filter itself does, so a test
    /// written once reads the same in all six frames.
    fn at(grid: &Grid, direction: BuildDirection, layer: usize, u: usize, v: usize) -> usize {
        let axis = direction.axis();
        let lateral = match axis {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let dims = [grid.nx, grid.ny, grid.nz];
        let mut c = [0usize; 3];
        c[axis] = if direction.is_positive() {
            layer
        } else {
            dims[axis] - 1 - layer
        };
        c[lateral[0]] = u;
        c[lateral[1]] = v;
        grid.cell_index(c[0], c[1], c[2])
    }

    #[test]
    fn a_floating_blob_is_erased() {
        let n = 8;
        let grid = design_grid(n);
        for direction in ALL_DIRECTIONS {
            let mut blueprint = vec![0.0; grid.n_cells()];
            // A solid 2 x 2 x 2 blob floating four layers above the build plate.
            for layer in 4..6 {
                for v in 3..5 {
                    for u in 3..5 {
                        blueprint[at(&grid, direction, layer, u, v)] = 1.0;
                    }
                }
            }
            let printed = filtered(&grid, &blueprint, direction);
            let worst = printed.iter().copied().fold(0.0f64, f64::max);
            assert!(
                worst < 0.02,
                "{}: the unsupported blob survived with density {worst}",
                direction.label()
            );
        }
    }

    #[test]
    fn a_column_standing_on_the_plate_survives() {
        let n = 8;
        let grid = design_grid(n);
        for direction in ALL_DIRECTIONS {
            let mut blueprint = vec![0.0; grid.n_cells()];
            for layer in 0..n {
                blueprint[at(&grid, direction, layer, 4, 4)] = 1.0;
            }
            let printed = filtered(&grid, &blueprint, direction);
            for layer in 0..n {
                let value = printed[at(&grid, direction, layer, 4, 4)];
                assert!(
                    (value - 1.0).abs() < 0.02,
                    "{}: layer {layer} of the column printed at {value}",
                    direction.label()
                );
            }
        }
    }

    #[test]
    fn a_45_degree_ramp_survives_unchanged() {
        let n = 8;
        let grid = design_grid(n);
        let mut blueprint = vec![0.0; grid.n_cells()];
        // Each layer steps one cell along x, which is exactly the steepest
        // overhang the supporting stencil allows.
        for k in 0..n {
            for j in 0..n {
                blueprint[grid.cell_index(k, j, k)] = 1.0;
            }
        }
        let printed = filtered(&grid, &blueprint, BuildDirection::ZPlus);
        for k in 0..n {
            for j in 0..n {
                let value = printed[grid.cell_index(k, j, k)];
                assert!(
                    (value - 1.0).abs() < 0.02,
                    "ramp cell ({k}, {j}, {k}) printed at {value}"
                );
            }
        }
    }

    #[test]
    fn the_build_plate_layer_prints_its_blueprint() {
        let n = 5;
        let grid = design_grid(n);
        for direction in ALL_DIRECTIONS {
            let mut blueprint = vec![0.0; grid.n_cells()];
            for v in 0..n {
                for u in 0..n {
                    blueprint[at(&grid, direction, 0, u, v)] = 0.3 + 0.1 * u as f64;
                }
            }
            let printed = filtered(&grid, &blueprint, direction);
            for v in 0..n {
                for u in 0..n {
                    let e = at(&grid, direction, 0, u, v);
                    assert!(
                        (printed[e] - blueprint[e]).abs() < 1e-12,
                        "{}: plate cell ({u}, {v}) changed from {} to {}",
                        direction.label(),
                        blueprint[e],
                        printed[e]
                    );
                }
            }
            // And the layer above it is the only thing that plate supports, so
            // nothing else in the block can have come out of the sweep solid.
            for layer in 2..n {
                for v in 0..n {
                    for u in 0..n {
                        let value = printed[at(&grid, direction, layer, u, v)];
                        assert!(
                            value < 0.02,
                            "{}: layer {layer} printed at {value} off a bare plate",
                            direction.label()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_build_direction_picks_the_plate() {
        let n = 6;
        let grid = design_grid(n);
        let mut blueprint = vec![0.0; grid.n_cells()];
        // A slab hugging the z maximum face: supported when building along z-,
        // floating when building along z+.
        for j in 0..n {
            for i in 0..n {
                blueprint[grid.cell_index(i, j, n - 1)] = 1.0;
            }
        }
        let down = filtered(&grid, &blueprint, BuildDirection::ZMinus);
        let up = filtered(&grid, &blueprint, BuildDirection::ZPlus);
        let probe = grid.cell_index(3, 3, n - 1);
        assert!((down[probe] - 1.0).abs() < 1e-12);
        assert!(
            up[probe] < 0.02,
            "the floating slab survived at {}",
            up[probe]
        );
    }

    #[test]
    fn solid_cells_support_and_void_cells_do_not() {
        let n = 6;
        let mut grid = design_grid(n);
        // A forced solid pad in the middle of layer 2, with nothing under it.
        for j in 2..4 {
            for i in 2..4 {
                let cell = grid.cell_index(i, j, 2);
                grid.cells[cell] = CellKind::Solid;
            }
        }
        let mut blueprint = vec![0.0; grid.n_cells()];
        for j in 2..4 {
            for i in 2..4 {
                for k in 2..5 {
                    blueprint[grid.cell_index(i, j, k)] = 1.0;
                }
            }
        }
        let printed = filtered(&grid, &blueprint, BuildDirection::ZPlus);
        // The keepin pad is printable support at density one, so the tower on
        // top of it survives even though the pad itself floats.
        assert_eq!(printed[grid.cell_index(2, 2, 2)], 1.0);
        for k in 3..5 {
            let value = printed[grid.cell_index(2, 2, k)];
            assert!(value > 0.9, "layer {k} above the pad printed at {value}");
        }
        // Layer 1 is void of material, so nothing there prints.
        assert_eq!(printed[grid.cell_index(2, 2, 1)], 0.0);
    }

    #[test]
    fn the_filter_never_adds_material() {
        let n = 6;
        let grid = design_grid(n);
        let mut state = 0x0BAD_C0DE_1234_5678u64;
        let blueprint: Vec<f64> = (0..grid.n_cells())
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect();
        let printed = filtered(&grid, &blueprint, BuildDirection::ZPlus);
        for e in 0..grid.n_cells() {
            // The smooth minimum sits at most sqrt(eps)/2 above the exact one.
            let slack = 0.5 * constants::AM_SMOOTH_MIN_EPS.sqrt();
            assert!(
                printed[e] <= blueprint[e] + slack + 1e-12,
                "cell {e} printed at {} from a blueprint of {}",
                printed[e],
                blueprint[e]
            );
            assert!(printed[e] >= -1e-12);
        }
    }

    #[test]
    fn the_adjoint_matches_the_forward_jacobian() {
        // Dot product test: <J a, b> must equal <a, J^T b> for the Jacobian at
        // an interior point, where J a is measured by a central difference.
        let n = 5;
        let mut grid = design_grid(n);
        let solid = grid.cell_index(1, 1, 1);
        let void = grid.cell_index(3, 3, 2);
        grid.cells[solid] = CellKind::Solid;
        grid.cells[void] = CellKind::Void;
        let cells = grid.n_cells();

        let mut state = 0xFEED_FACE_C0DE_0001u64;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        // Away from zero, where the smooth maximum's own gradient vanishes.
        let blueprint: Vec<f64> = (0..cells).map(|_| 0.2 + 0.6 * rand()).collect();
        let a: Vec<f64> = (0..cells).map(|_| rand() - 0.5).collect();
        let b: Vec<f64> = (0..cells).map(|_| rand() - 0.5).collect();

        for direction in ALL_DIRECTIONS {
            let filter = AmFilter::new(&grid, direction);
            let mut printed = vec![0.0; cells];
            filter.apply(&blueprint, &mut printed);
            let mut adjoint = vec![0.0; cells];
            filter.apply_transpose(&blueprint, &printed, &b, &mut adjoint);

            let step = 1e-6;
            let perturbed = |sign: f64| -> Vec<f64> {
                let shifted: Vec<f64> = blueprint
                    .iter()
                    .zip(a.iter())
                    .enumerate()
                    .map(|(e, (x, d))| {
                        if grid.cells[e] == CellKind::Design {
                            x + sign * step * d
                        } else {
                            *x
                        }
                    })
                    .collect();
                let mut out = vec![0.0; cells];
                filter.apply(&shifted, &mut out);
                out
            };
            let plus = perturbed(1.0);
            let minus = perturbed(-1.0);

            let lhs: f64 = (0..cells)
                .filter(|&e| grid.cells[e] == CellKind::Design)
                .map(|e| (plus[e] - minus[e]) / (2.0 * step) * b[e])
                .sum();
            let rhs: f64 = (0..cells)
                .filter(|&e| grid.cells[e] == CellKind::Design)
                .map(|e| a[e] * adjoint[e])
                .sum();
            assert!(
                (lhs - rhs).abs() < 1e-6 * lhs.abs().max(1.0),
                "{}: directional derivative {lhs} does not match the adjoint {rhs}",
                direction.label()
            );
            // A frame that never moved would pass vacuously.
            assert!(
                lhs.abs() > 1e-9,
                "{}: the probe direction produced no response at all",
                direction.label()
            );
        }
    }

    #[test]
    fn the_smooth_maximum_is_exact_at_its_anchor() {
        let n = constants::AM_SUPPORT_COUNT as f64;
        let q = P as f64 + n.ln() / constants::AM_SMOOTH_MAX_ANCHOR.ln();
        let anchor = constants::AM_SMOOTH_MAX_ANCHOR;
        let (value, _) = smooth_max(&[anchor; constants::AM_SUPPORT_COUNT], q);
        assert!(
            (value - anchor).abs() < 1e-12,
            "an all-equal region at the anchor gave {value}"
        );
        // A single large member dominates the rest.
        let (value, _) = smooth_max(&[0.9, 0.1, 0.1, 0.0, 0.0], q);
        assert!((value - 0.9).abs() < 0.02, "one-sided maximum gave {value}");
        let (value, sum) = smooth_max(&[0.0; constants::AM_SUPPORT_COUNT], q);
        assert_eq!((value, sum), (0.0, 0.0));
    }

    /// The adjoint has to survive a supporting region so faint that the raw
    /// power sum underflows.
    ///
    /// `v^40` leaves the normal range below about 2e-8, and the derivative
    /// written as `smax / sum` then divides an order-one number by a subnormal
    /// one: it used to overflow to `inf`, meet the `0` of `v^(P-1)` in the next
    /// gather and turn the whole sensitivity field into `NaN` - which is what a
    /// design variable sitting exactly on the lower box bound produces within a
    /// few iterations. The scaled form has no such intermediate.
    #[test]
    fn the_adjoint_survives_a_supporting_region_that_underflows() {
        let n = 5;
        let grid = design_grid(n);
        let cells = grid.n_cells();
        let q = P as f64
            + (constants::AM_SUPPORT_COUNT as f64).ln() / constants::AM_SMOOTH_MAX_ANCHOR.ln();

        // The scale factors stay bounded across thirty decades of density.
        for faint in [1e-3, 1e-8, 1e-12, 1e-30, f64::MIN_POSITIVE] {
            let (peak, scale) = smooth_max_gradient_scale(&[faint; constants::AM_SUPPORT_COUNT], q);
            assert_eq!(peak, faint);
            assert!(
                scale.is_finite() && scale >= 0.0,
                "a region at {faint} gave a scale of {scale}"
            );
            // The derivative the caller ends up with is scale * (v/peak)^(P-1),
            // which for an all-equal region is scale itself: at most the P/q of
            // an exact maximum, never an overflow.
            assert!(scale <= P as f64 / q + 1e-12, "scale {scale} at {faint}");
        }

        for direction in ALL_DIRECTIONS {
            let filter = AmFilter::new(&grid, direction);
            // A blueprint of near-zero material, with one ordinary cell so the
            // sweep still has something to propagate.
            let mut blueprint = vec![1e-9; cells];
            blueprint[grid.cell_index(2, 2, 0)] = 0.5;
            let mut printed = vec![0.0; cells];
            filter.apply(&blueprint, &mut printed);
            assert!(printed.iter().all(|v| v.is_finite()));

            let d_printed = vec![1.0; cells];
            let mut adjoint = vec![0.0; cells];
            filter.apply_transpose(&blueprint, &printed, &d_printed, &mut adjoint);
            assert!(
                adjoint.iter().all(|v| v.is_finite()),
                "{}: the adjoint of a faint design is not finite",
                direction.label()
            );
        }
    }

    #[test]
    fn the_smooth_minimum_brackets_the_exact_one() {
        let slack = 0.5 * constants::AM_SMOOTH_MIN_EPS.sqrt();
        for (a, b) in [(0.2, 0.8), (0.8, 0.2), (0.5, 0.5), (0.0, 1.0)] {
            let value = smooth_min(a, b);
            let exact = a.min(b);
            assert!(value >= exact - 1e-12 && value <= exact + slack + 1e-12);
            let (da, db) = smooth_min_gradient(a, b);
            assert!((da + db - 1.0).abs() < 1e-12);
            assert!(da >= 0.0 && db >= 0.0);
        }
    }
}
