//! The density chain: design variables to the physical densities the analysis,
//! the volume constraint and the exported surface all see.
//!
//! ```text
//!   x  --density filter-->  x_tilde  --AM filter-->  x_phys
//! ```
//!
//! The additive manufacturing stage is optional; without it the chain is the
//! bare density filter and produces exactly the values it always did.
//! Sensitivities travel back through both transposes in the opposite order.

use crate::config::BuildDirection;
use crate::engine::am_filter::AmFilter;
use crate::engine::filter::DensityFilter;
use crate::grid::Grid;

/// Design variables in, printed physical densities out.
pub struct DensityChain {
    filter: DensityFilter,
    overhang: Option<AmFilter>,
}

impl DensityChain {
    /// Build the chain for `grid`. `overhang` switches the self-supporting
    /// constraint on and fixes the build direction.
    pub fn new(grid: &Grid, radius_mm: f64, overhang: Option<BuildDirection>) -> DensityChain {
        DensityChain {
            filter: DensityFilter::new(grid, radius_mm),
            overhang: overhang.map(|direction| AmFilter::new(grid, direction)),
        }
    }

    /// The density filter stage.
    pub fn filter(&self) -> &DensityFilter {
        &self.filter
    }

    /// The additive manufacturing stage, when the chain has one.
    pub fn overhang(&self) -> Option<&AmFilter> {
        self.overhang.as_ref()
    }

    /// True when the chain is a linear operator, so its transpose does not
    /// depend on the point it is taken at and quantities derived from it can be
    /// computed once.
    pub fn is_linear(&self) -> bool {
        self.overhang.is_none()
    }

    /// Map design variables to printed physical densities.
    ///
    /// `filtered` receives the intermediate density-filtered field, which the
    /// transpose needs again as the point its Jacobian is taken at.
    pub fn apply(&self, x: &[f64], filtered: &mut [f64], printed: &mut [f64]) {
        self.filter.apply(x, filtered);
        match &self.overhang {
            Some(am) => am.apply(filtered, printed),
            None => printed.copy_from_slice(filtered),
        }
    }

    /// Chain a sensitivity with respect to the printed densities back to the
    /// design variables.
    ///
    /// `filtered` and `printed` are the pair [`DensityChain::apply`] produced.
    /// `scratch` is a full length cell array the intermediate derivative passes
    /// through.
    pub fn apply_transpose(
        &self,
        filtered: &[f64],
        printed: &[f64],
        d_printed: &[f64],
        scratch: &mut [f64],
        out: &mut [f64],
    ) {
        match &self.overhang {
            Some(am) => {
                am.apply_transpose(filtered, printed, d_printed, scratch);
                self.filter.apply_transpose(scratch, out);
            }
            None => self.filter.apply_transpose(d_printed, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn without_an_overhang_stage_the_chain_is_the_bare_filter() {
        let grid = design_grid(6);
        let chain = DensityChain::new(&grid, 2.5, None);
        assert!(chain.is_linear());
        let cells = grid.n_cells();
        let mut state = 12345u64;
        let x: Vec<f64> = (0..cells)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) as f64 / (1u64 << 31) as f64
            })
            .collect();

        let mut filtered = vec![0.0; cells];
        let mut printed = vec![0.0; cells];
        chain.apply(&x, &mut filtered, &mut printed);
        let mut reference = vec![0.0; cells];
        chain.filter().apply(&x, &mut reference);
        assert_eq!(printed, reference);
        assert_eq!(filtered, reference);

        let mut scratch = vec![0.0; cells];
        let mut out = vec![0.0; cells];
        chain.apply_transpose(&filtered, &printed, &x, &mut scratch, &mut out);
        let mut reference = vec![0.0; cells];
        chain.filter().apply_transpose(&x, &mut reference);
        assert_eq!(out, reference);
    }

    #[test]
    fn an_overhang_stage_erases_what_the_filter_leaves_floating() {
        let grid = design_grid(8);
        let chain = DensityChain::new(&grid, 0.5, Some(BuildDirection::ZPlus));
        assert!(!chain.is_linear());
        assert!(chain.overhang().is_some());
        let cells = grid.n_cells();
        let mut x = vec![0.0; cells];
        for k in 5..7 {
            for j in 3..5 {
                for i in 3..5 {
                    x[grid.cell_index(i, j, k)] = 1.0;
                }
            }
        }
        let mut filtered = vec![0.0; cells];
        let mut printed = vec![0.0; cells];
        chain.apply(&x, &mut filtered, &mut printed);
        assert!((filtered[grid.cell_index(3, 3, 5)] - 1.0).abs() < 1e-12);
        assert!(printed.iter().cloned().fold(0.0, f64::max) < 0.02);
    }
}
