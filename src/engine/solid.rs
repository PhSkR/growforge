//! The engine that exports the domain itself: every design cell filled, and
//! nothing optimized.
//!
//! There is no loop here, no linear solve and no design variable. The field it
//! returns is the cell classification read as a density - void cells empty,
//! forced solid cells and design cells full - which is the discrete form of "the
//! part is the design space". Everything downstream of the engine is untouched
//! and runs exactly as it does after a SIMP or a growth run: the cavity pass,
//! the stress solve on the field that ships, the trim and reinforcement passes
//! (which `Config::check_solid` refuses for this engine, since they would alter
//! the very thing it exports), the isosurface, the boundary clamp and the STL.
//!
//! What it is for is the part that was never a topology problem. Before it, such
//! a part had to be faked through SIMP at a `mass_fraction` just under one, and
//! that is not the same object: the optimizer still redistributes material to
//! buy compliance it was not asked to buy, and the density filter still erodes
//! the cells at the boundary, so what comes out has walls that ripple by a
//! fraction of a voxel. Here the surface is exact by construction - the only
//! thing between the configuration's own CSG and the file is the isosurface and
//! the clamp that holds it to those analytic shapes.
//!
//! It reports **no iterations at all**: `Reporter::iteration` is never called,
//! so a console prints no progress lines and a viewer's panel shows no
//! per-iteration block. The run summary and the problem summary say what ran
//! instead, keyed on the engine rather than on this field - see
//! [`crate::problem::Problem::is_solid`].

use anyhow::Result;
use rayon::prelude::*;

use crate::constants;
use crate::engine::{DensityField, Engine, StopReason, growth};
use crate::grid::CellKind;
use crate::problem::Problem;
use crate::report::Reporter;

/// The solid engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SolidEngine;

impl Engine for SolidEngine {
    fn name(&self) -> &'static str {
        constants::SOLID_ENGINE
    }

    fn optimize(&self, problem: &Problem, _reporter: &dyn Reporter) -> Result<DensityField> {
        let grid = &problem.grid;
        // The same three-way read of the classification every other engine ends
        // on (see `crate::engine::growth::rasterize::densities`), with the
        // design cells full rather than rasterized.
        let densities: Vec<f64> = grid
            .cells
            .par_iter()
            .map(|kind| match kind {
                CellKind::Void => constants::DENSITY_MIN,
                CellKind::Solid | CellKind::Design => constants::DENSITY_MAX,
            })
            .collect();
        // Measured rather than assumed, so it is the mean of the field that was
        // actually built: one for a problem with design cells, and zero for one
        // whose domain is entirely forced solid, where there is no design cell
        // to take a mean over. Through the growth engine's helper because that
        // is the crate's one statement of the quantity `mass_fraction` measures,
        // keyed on the classification rather than on an index list.
        let volume_fraction = growth::rasterize::design_volume_fraction(grid, &densities);
        Ok(DensityField {
            densities,
            // Nothing was iterated, so there is nothing to count and no
            // compliance to quote. Zero here is not a measurement standing in
            // for a missing one: the run summary keys on the engine and prints
            // neither number.
            iterations: 0,
            compliance: 0.0,
            initial_compliance: 0.0,
            volume_fraction,
            max_change: 0.0,
            // The design settled before it began: there is nothing left to do,
            // which is what `Converged` means for an engine with no design
            // variables - the reading the growth engine already gives it.
            stop: StopReason::Converged,
            overhang_residual: None,
            growth: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::report::{IterationStats, Reporter};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A block with a hole through it: design cells, forced solid cells (the
    /// keepin) and void cells (the keepout) all present, so the field has all
    /// three answers to give.
    const BLOCK: &str = r#"
engine = "solid"

[project]
name = "solid_block"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[solver]
backend = "cpu"

[optimization]
min_feature_mm = 8.0

[output]
stl_path = "growforge_solid_block.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [32.0, 16.0, 16.0]

[[keepout]]
shape = "cylinder"
p1 = [16.0, 8.0, -1.0]
p2 = [16.0, 8.0, 17.0]
radius = 3.0

[[keepin]]
shape = "box"
min = [28.0, 4.0, 4.0]
max = [32.0, 12.0, 12.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 16.5, 16.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [31.5, -0.5, -0.5], max = [32.5, 16.5, 16.5] }
vector = [0.0, 0.0, -20.0]
"#;

    fn block() -> Problem {
        let config = Config::parse(BLOCK).expect("parse");
        Problem::build(&config, &PathBuf::from(".")).expect("build")
    }

    /// Counts the progress lines an engine emits.
    #[derive(Default)]
    struct Counting(AtomicUsize);

    impl Reporter for Counting {
        fn iteration(&self, _stats: &IterationStats) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
        fn note(&self, _message: &str) {}
    }

    #[test]
    fn every_design_cell_comes_out_full_and_the_classification_is_kept() {
        let problem = block();
        let reporter = Counting::default();
        let field = SolidEngine.optimize(&problem, &reporter).expect("solid");

        assert_eq!(field.densities.len(), problem.grid.n_cells());
        let (mut design, mut solid, mut void) = (0, 0, 0);
        for (e, kind) in problem.grid.cells.iter().enumerate() {
            match kind {
                CellKind::Design => {
                    design += 1;
                    assert_eq!(field.densities[e], constants::DENSITY_MAX);
                }
                CellKind::Solid => {
                    solid += 1;
                    assert_eq!(field.densities[e], constants::DENSITY_MAX);
                }
                CellKind::Void => {
                    void += 1;
                    assert_eq!(field.densities[e], constants::DENSITY_MIN);
                }
            }
        }
        assert!(
            design > 0 && solid > 0 && void > 0,
            "the fixture has {design} design, {solid} solid and {void} void cells, so it cannot \
             tell the three apart"
        );

        // The shape of the field, as every consumer downstream reads it.
        assert_eq!(field.iterations, 0);
        assert_eq!(field.compliance, 0.0);
        assert_eq!(field.initial_compliance, 0.0);
        assert_eq!(field.volume_fraction, constants::DENSITY_MAX);
        assert_eq!(field.max_change, 0.0);
        assert_eq!(field.stop, StopReason::Converged);
        assert!(field.overhang_residual.is_none());
        assert!(field.growth.is_none());

        // A run with no iterations reports none: the console prints no progress
        // line and the viewer's panel gets no per-iteration block.
        assert_eq!(reporter.0.load(Ordering::Relaxed), 0);
    }

    /// The registry is the seam: `optimize` has to reach this engine by name,
    /// without any caller knowing which one ran.
    #[test]
    fn the_field_is_the_same_through_the_registry() {
        let problem = block();
        let direct = SolidEngine
            .optimize(&problem, &crate::report::SilentReporter)
            .expect("solid");
        let dispatched =
            crate::optimize(&problem, &crate::report::SilentReporter).expect("dispatched");
        assert_eq!(direct.densities, dispatched.densities);
        assert_eq!(dispatched.iterations, 0);
    }
}
