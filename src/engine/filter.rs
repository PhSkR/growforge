//! Linear cone density filter.
//!
//! The design variables `x` are smoothed into physical densities
//! `x_tilde = filter(x)` before they reach the finite element model, which is
//! what imposes a minimum length scale and removes checkerboarding. Sensitivity
//! information travels back through the transpose of the same operator.
//!
//! Only design cells are filtered. Forced solid neighbours contribute a density
//! of one, void neighbours are excluded from both the sum and its
//! normalisation, so the filter never pulls material out of a keepout.

use rayon::prelude::*;

use crate::grid::{CellKind, Grid};

/// One entry of the precomputed neighbour stencil.
#[derive(Debug, Clone, Copy)]
struct StencilEntry {
    di: i32,
    dj: i32,
    dk: i32,
    weight: f64,
}

/// Cone filter over the cells of a grid.
pub struct DensityFilter {
    nx: i32,
    ny: i32,
    nz: i32,
    stencil: Vec<StencilEntry>,
    /// Per cell sum of the stencil weights over non-void neighbours; zero for
    /// cells that are not design variables.
    normalization: Vec<f64>,
    kinds: Vec<CellKind>,
}

impl DensityFilter {
    /// Build the filter for `grid` with a cone radius of `radius_mm`.
    pub fn new(grid: &Grid, radius_mm: f64) -> DensityFilter {
        let reach = (radius_mm / grid.h).floor() as i32;
        let mut stencil = Vec::new();
        for dk in -reach..=reach {
            for dj in -reach..=reach {
                for di in -reach..=reach {
                    let dist = grid.h * ((di * di + dj * dj + dk * dk) as f64).sqrt();
                    let weight = radius_mm - dist;
                    if weight > 0.0 {
                        stencil.push(StencilEntry { di, dj, dk, weight });
                    }
                }
            }
        }

        let mut filter = DensityFilter {
            nx: grid.nx as i32,
            ny: grid.ny as i32,
            nz: grid.nz as i32,
            stencil,
            normalization: Vec::new(),
            kinds: grid.cells.clone(),
        };
        let normalization: Vec<f64> = (0..grid.n_cells())
            .into_par_iter()
            .map(|e| {
                if filter.kinds[e] != CellKind::Design {
                    return 0.0;
                }
                let (i, j, k) = filter.ijk(e);
                filter
                    .neighbours(i, j, k)
                    .filter(|(n, _)| filter.kinds[*n] != CellKind::Void)
                    .map(|(_, w)| w)
                    .sum()
            })
            .collect();
        filter.normalization = normalization;
        filter
    }

    fn ijk(&self, e: usize) -> (i32, i32, i32) {
        let nx = self.nx as usize;
        let ny = self.ny as usize;
        (
            (e % nx) as i32,
            ((e / nx) % ny) as i32,
            (e / (nx * ny)) as i32,
        )
    }

    fn index(&self, i: i32, j: i32, k: i32) -> usize {
        (i + self.nx * (j + self.ny * k)) as usize
    }

    /// Iterate the in-bounds stencil neighbours of a cell as
    /// `(cell index, weight)` pairs.
    fn neighbours(&self, i: i32, j: i32, k: i32) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.stencil.iter().filter_map(move |s| {
            let ni = i + s.di;
            let nj = j + s.dj;
            let nk = k + s.dk;
            if ni < 0 || nj < 0 || nk < 0 || ni >= self.nx || nj >= self.ny || nk >= self.nz {
                None
            } else {
                Some((self.index(ni, nj, nk), s.weight))
            }
        })
    }

    /// Number of stencil taps, for reporting.
    pub fn stencil_size(&self) -> usize {
        self.stencil.len()
    }

    /// Map design variables to physical densities.
    ///
    /// `x` and `out` are full length cell arrays. Void cells map to zero,
    /// forced solid cells to one, and design cells to the normalised cone
    /// average of their non-void neighbourhood.
    pub fn apply(&self, x: &[f64], out: &mut [f64]) {
        let kinds = &self.kinds;
        out.par_iter_mut().enumerate().for_each(|(e, slot)| {
            *slot = match kinds[e] {
                CellKind::Void => 0.0,
                CellKind::Solid => 1.0,
                CellKind::Design => {
                    let (i, j, k) = self.ijk(e);
                    let sum: f64 = self
                        .neighbours(i, j, k)
                        .map(|(n, w)| match kinds[n] {
                            CellKind::Void => 0.0,
                            CellKind::Solid => w,
                            CellKind::Design => w * x[n],
                        })
                        .sum();
                    let norm = self.normalization[e];
                    if norm > 0.0 { sum / norm } else { 0.0 }
                }
            };
        });
    }

    /// Apply the transpose of the filter, which is the chain rule for a
    /// quantity differentiated with respect to the physical densities.
    ///
    /// `dfdxt` holds the derivatives with respect to the physical densities of
    /// the design cells; entries of non-design cells are ignored. `out`
    /// receives the derivatives with respect to the design variables.
    pub fn apply_transpose(&self, dfdxt: &[f64], out: &mut [f64]) {
        let kinds = &self.kinds;
        let scaled: Vec<f64> = (0..dfdxt.len())
            .into_par_iter()
            .map(|e| {
                if kinds[e] == CellKind::Design && self.normalization[e] > 0.0 {
                    dfdxt[e] / self.normalization[e]
                } else {
                    0.0
                }
            })
            .collect();
        out.par_iter_mut().enumerate().for_each(|(e, slot)| {
            *slot = if kinds[e] == CellKind::Design {
                let (i, j, k) = self.ijk(e);
                self.neighbours(i, j, k)
                    .filter(|(n, _)| kinds[*n] == CellKind::Design)
                    .map(|(n, w)| w * scaled[n])
                    .sum()
            } else {
                0.0
            };
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;

    fn design_grid(n: usize, h: f64) -> Grid {
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [n as f64 * h, n as f64 * h, n as f64 * h],
            },
            h,
        );
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Design;
        }
        grid
    }

    #[test]
    fn stencil_contains_the_centre_and_is_symmetric() {
        let grid = design_grid(8, 1.0);
        let f = DensityFilter::new(&grid, 2.0);
        assert!(f.stencil_size() > 1);
        for s in &f.stencil {
            assert!(
                f.stencil.iter().any(|t| t.di == -s.di
                    && t.dj == -s.dj
                    && t.dk == -s.dk
                    && t.weight == s.weight),
                "stencil is not symmetric"
            );
        }
    }

    #[test]
    fn uniform_field_is_unchanged_including_at_the_boundary() {
        let grid = design_grid(6, 1.5);
        let f = DensityFilter::new(&grid, 4.0);
        let x = vec![0.37; grid.n_cells()];
        let mut out = vec![0.0; grid.n_cells()];
        f.apply(&x, &mut out);
        for (e, v) in out.iter().enumerate() {
            assert!((v - 0.37).abs() < 1e-12, "cell {e} became {v}");
        }
    }

    #[test]
    fn filtered_field_respects_the_density_bounds() {
        let grid = design_grid(6, 1.0);
        let f = DensityFilter::new(&grid, 2.5);
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let x: Vec<f64> = (0..grid.n_cells())
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect();
        let mut out = vec![0.0; grid.n_cells()];
        f.apply(&x, &mut out);
        let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for v in &out {
            assert!(
                *v >= lo - 1e-12 && *v <= hi + 1e-12,
                "value {v} left the bounds"
            );
        }
    }

    #[test]
    fn solid_neighbours_count_as_full_density_and_void_is_excluded() {
        let mut grid = design_grid(5, 1.0);
        let solid = grid.cell_index(2, 2, 2);
        grid.cells[solid] = CellKind::Solid;
        let void = grid.cell_index(1, 2, 2);
        grid.cells[void] = CellKind::Void;
        let f = DensityFilter::new(&grid, 1.5);
        let x = vec![0.0; grid.n_cells()];
        let mut out = vec![0.0; grid.n_cells()];
        f.apply(&x, &mut out);
        // A design cell next to the solid one picks up density from it.
        let probe = grid.cell_index(3, 2, 2);
        assert!(out[probe] > 0.0);
        assert_eq!(out[solid], 1.0);
        assert_eq!(out[void], 0.0);

        // With every design variable at one, the void neighbour must not drag
        // the filtered value below one.
        let x = vec![1.0; grid.n_cells()];
        f.apply(&x, &mut out);
        let next_to_void = grid.cell_index(0, 2, 2);
        assert!((out[next_to_void] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn transpose_matches_the_forward_operator() {
        let mut grid = design_grid(5, 1.0);
        let solid = grid.cell_index(1, 1, 1);
        let void = grid.cell_index(3, 3, 3);
        grid.cells[solid] = CellKind::Solid;
        grid.cells[void] = CellKind::Void;
        let f = DensityFilter::new(&grid, 2.2);
        let n = grid.n_cells();

        let mut state = 0xDEAD_BEEF_CAFE_1234u64;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let a: Vec<f64> = (0..n).map(|_| rand()).collect();
        let b: Vec<f64> = (0..n).map(|_| rand()).collect();

        // <F a, b> and <a, F^T b> must agree on the design subspace, so compare
        // the linear parts by differencing against the zero design vector.
        let zero = vec![0.0; n];
        let mut fa = vec![0.0; n];
        let mut f0 = vec![0.0; n];
        f.apply(&a, &mut fa);
        f.apply(&zero, &mut f0);
        let mut ftb = vec![0.0; n];
        f.apply_transpose(&b, &mut ftb);

        let lhs: f64 = (0..n)
            .filter(|&e| grid.cells[e] == CellKind::Design)
            .map(|e| (fa[e] - f0[e]) * b[e])
            .sum();
        let rhs: f64 = (0..n)
            .filter(|&e| grid.cells[e] == CellKind::Design)
            .map(|e| a[e] * ftb[e])
            .sum();
        assert!(
            (lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0),
            "{lhs} vs {rhs}"
        );
    }

    #[test]
    fn radius_below_one_voxel_is_the_identity() {
        let grid = design_grid(4, 2.0);
        let f = DensityFilter::new(&grid, 1.0);
        assert_eq!(f.stencil_size(), 1);
        let mut state = 7u64;
        let x: Vec<f64> = (0..grid.n_cells())
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) as f64 / (1u64 << 31) as f64
            })
            .collect();
        let mut out = vec![0.0; grid.n_cells()];
        f.apply(&x, &mut out);
        for e in 0..grid.n_cells() {
            assert!((out[e] - x[e]).abs() < 1e-12);
        }
    }
}
