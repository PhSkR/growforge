//! Optimization engines.
//!
//! An engine takes a discrete [`Problem`] and returns a converged density
//! field. Two are registered below: `simp`, the finite element driven topology
//! optimizer, and `growth`, a fast deterministic growth heuristic. The registry
//! is the seam they plug into; neither the CLI, the post-processing nor the mesh
//! pipeline knows which one ran.
//!
//! [`wireframe`] is a setup stage of the SIMP engine rather than an engine of its
//! own: it seeds a guide the optimizer is free to discard, and
//! [`local_volume`] is a second constraint of that engine's subproblem rather
//! than an engine either.

pub mod am_filter;
pub mod chain;
pub mod filter;
pub mod growth;
pub mod local_volume;
pub mod mma;
pub mod objective;
pub mod oc;
pub mod simp;
pub mod stall;
pub mod update;
pub mod wireframe;

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::config::SymmetryParams;
use crate::constants;
use crate::grid::{CellKind, Grid};
use crate::problem::Problem;
use crate::report::Reporter;

/// SIMP material interpolation `E(rho) = Emin + rho^p (E0 - Emin)` in MPa.
///
/// Void cells carry no stiffness at all and forced solid cells the full modulus;
/// only design cells are interpolated. The optimizer and the post-run stress
/// analysis share this so a reported stress can never belong to a different
/// structure than the one that was optimized - which is also why both numbers
/// the formula holds are passed in rather than read from [`constants`]: they are
/// `[optimization] penalty` and `[optimization] stiffness_floor`, and a stress
/// report taken at either of them different would describe a different
/// structure. `stiffness_floor` is the fraction of `E0` the floor sits at, so
/// `Emin` is formed here and nowhere else.
pub fn simp_moduli(
    grid: &Grid,
    densities: &[f64],
    youngs_modulus_mpa: f64,
    penalty: f64,
    stiffness_floor: f64,
    out: &mut [f64],
) {
    let e0 = youngs_modulus_mpa;
    let emin = stiffness_floor * e0;
    let kinds = &grid.cells;
    out.par_iter_mut().enumerate().for_each(|(e, slot)| {
        *slot = match kinds[e] {
            CellKind::Void => 0.0,
            CellKind::Solid => e0,
            CellKind::Design => emin + densities[e].powf(penalty) * (e0 - emin),
        };
    });
}

/// Physical density of a cell as every stage of the pipeline reads it: pinned
/// for void and forced solid cells, taken from the array for design cells.
pub fn cell_density(grid: &Grid, densities: &[f64], cell: usize) -> f64 {
    match grid.cells[cell] {
        CellKind::Void => constants::DENSITY_MIN,
        CellKind::Solid => constants::DENSITY_MAX,
        CellKind::Design => densities[cell],
    }
}

/// The result of an optimization run.
#[derive(Debug, Clone)]
pub struct DensityField {
    /// Physical density of every cell, in grid element order.
    pub densities: Vec<f64>,
    /// Number of optimization iterations performed.
    pub iterations: usize,
    /// Weighted compliance of the last analysed design, in N*mm. Zero when
    /// `growth` is set: the growth engine never evaluates a compliance, and the
    /// post-run stress report is what says how good its result is.
    pub compliance: f64,
    /// Weighted compliance of the first iteration, in N*mm. Zero for a growth
    /// run, see [`DensityField::compliance`].
    pub initial_compliance: f64,
    /// Mean physical density over the design cells.
    pub volume_fraction: f64,
    /// Largest absolute design variable change in the last iteration. Zero for a
    /// growth run, which has no design variables.
    pub max_change: f64,
    /// Why the run stopped.
    pub stop: StopReason,
    /// How far the printed design still sits from the blueprint the optimizer
    /// asked for, or `None` when no overhang constraint was active.
    pub overhang_residual: Option<OverhangResidual>,
    /// What the growth engine grew, or `None` when another engine ran.
    pub growth: Option<GrowthSummary>,
}

/// Why an engine stopped iterating.
///
/// Three of the four are outcomes a finished run reports, and all three are
/// finished runs: a stalled run and a capped run export, are stress analysed and
/// exit zero exactly as a converged one does. What separates them is what the
/// design *is* - an answer, an iterate the problem will not improve on, or the
/// iterate the budget happened to end on - and saying which is the whole point
/// of reporting them apart.
///
/// The fourth is not an outcome at all: it is the caller taking the run back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The design settled: the largest design variable change fell below
    /// `[optimization] convergence_tol`. A growth run reports it when the
    /// colonization ended of its own accord rather than on `[growth] max_steps`,
    /// which for that engine is the same statement - there was nothing left to
    /// do.
    Converged,
    /// The run stopped making progress and said so: over the last
    /// [`constants::STALL_WINDOW_ITERATIONS`] iterations every step was clipped
    /// by the update scheme's move limit and the compliance established no
    /// better design in any of them. SIMP only; see [`stall`].
    Stalled,
    /// `[optimization] max_iterations`, or `[growth] max_steps`, ran out. The
    /// design is the iterate the budget ended on and may still have been moving.
    IterationCap,
    /// The reporter asked for the run to stop and the engine did, between
    /// iterations. Every console reporter answers that question `false`, so only
    /// the editor's stop button and its auto-regrow can produce it; nothing is
    /// exported from a run that ends this way.
    Cancelled,
}

impl StopReason {
    /// True when the run reached a design it had stopped moving, rather than one
    /// of the three ways of stopping short of that.
    pub fn converged(self) -> bool {
        self == StopReason::Converged
    }

    /// Short label used in the run summary and the viewer panel.
    pub fn label(self) -> &'static str {
        match self {
            StopReason::Converged => "converged",
            StopReason::Stalled => "stalled",
            StopReason::IterationCap => "iteration cap",
            StopReason::Cancelled => "stopped",
        }
    }
}

/// What a finished growth run produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrowthSummary {
    /// Guaranteed load paths routed from a load region to a support region.
    pub backbones: usize,
    /// Strut segments in the final skeleton.
    pub segments: usize,
    /// Attraction points scattered through the design domain.
    pub attractors: usize,
    /// Attraction points a branch reached and consumed.
    pub consumed: usize,
    /// Smallest and largest strut radius in the exported skeleton, in mm.
    pub radius_range_mm: (f64, f64),
    /// Volume fraction the radius clamps allow, when they stopped the
    /// normalization from reaching `mass_fraction`. `None` when the target was
    /// met.
    pub clamped_volume_fraction: Option<f64>,
    /// Attraction points seeded on the structural surfaces, one per patch of
    /// keepin, support region or load region.
    pub surface_targets: usize,
    /// How many of those no branch ever fused to.
    pub unreached_surfaces: usize,
    /// Branch nodes removed because they ended on nothing. Zero when
    /// `[growth] prune` is off.
    pub pruned_nodes: usize,
    /// Branch tips, other than the backbone tips, that fused to a load region
    /// and therefore carry part of its load.
    pub fused_tips: usize,
    /// What the run replicated, when `[growth.symmetry]` asked it to.
    pub symmetry: Option<GrowthSymmetry>,
}

/// What a symmetric growth run grew and replicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrowthSymmetry {
    /// The symmetry the structure was grown in and replicated by.
    pub params: SymmetryParams,
    /// Design cells inside the fundamental domain: the volume the growth
    /// actually had to fill.
    ///
    /// **Approximately** the problem's design cells divided by the sector
    /// count, and exactly that only when no cell centre lies on the boundary.
    /// An axis with an odd number of cells puts a whole layer of centres on a
    /// mirror plane, and that layer belongs to the fundamental domain - a layer
    /// belonging to neither half would be a one-voxel gap through the part - so
    /// the count runs high by half a layer per such plane. Five cells across a
    /// mirror plane are owned three to two.
    pub fundamental_design_cells: usize,
    /// True when every copy takes a cell centre to a cell centre, so the
    /// exported field is symmetric to the bit rather than to within a voxel.
    ///
    /// The skeleton is exact either way; see
    /// [`crate::engine::growth::symmetry::Symmetry::maps_cell_centres`] for
    /// which transforms qualify and what the others cost.
    pub exact_on_the_voxel_lattice: bool,
}

/// `|printed - blueprint|` over the design cells of a finished design.
///
/// A small residual says the additive manufacturing filter is no longer fighting
/// the design: what the optimizer asked for is what a printer can lay down. The
/// maximum is a single worst cell and is easily dominated by one variable the
/// filter is still erasing, so the mean is reported next to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverhangResidual {
    /// Largest absolute difference over the design cells.
    pub max: f64,
    /// Mean absolute difference over the design cells.
    pub mean: f64,
}

/// An optimization strategy: problem in, converged density field out.
pub trait Engine: Send + Sync + std::fmt::Debug {
    /// Registry key of this engine.
    fn name(&self) -> &'static str;

    /// Run the optimization.
    fn optimize(&self, problem: &Problem, reporter: &dyn Reporter) -> Result<DensityField>;
}

/// Constructor of a registered engine.
type EngineFactory = fn() -> Box<dyn Engine>;

fn make_simp() -> Box<dyn Engine> {
    Box::new(simp::SimpEngine)
}

fn make_growth() -> Box<dyn Engine> {
    Box::new(growth::GrowthEngine)
}

/// All engines growforge knows about, keyed by their config value.
const REGISTRY: &[(&str, EngineFactory)] = &[
    (constants::DEFAULT_ENGINE, make_simp),
    (constants::GROWTH_ENGINE, make_growth),
];

/// Registered engine keys, in registration order.
pub fn available() -> Vec<&'static str> {
    REGISTRY.iter().map(|(key, _)| *key).collect()
}

/// Fail unless `name` refers to a registered engine.
///
/// Called while the problem is built so that `growforge check` rejects an
/// unknown engine before any work is done.
pub fn ensure_registered(name: &str) -> Result<()> {
    if REGISTRY.iter().any(|(key, _)| *key == name) {
        return Ok(());
    }
    bail!(
        "unknown engine \"{name}\"; available engines are {}",
        available().join(", ")
    )
}

/// Instantiate an engine by key.
pub fn create(name: &str) -> Result<Box<dyn Engine>> {
    ensure_registered(name)?;
    let (_, factory) = REGISTRY
        .iter()
        .find(|(key, _)| *key == name)
        .expect("the registry lookup already succeeded");
    Ok(factory())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_engine_is_registered() {
        let engine = create(constants::DEFAULT_ENGINE).expect("default engine");
        assert_eq!(engine.name(), constants::DEFAULT_ENGINE);
    }

    #[test]
    fn the_growth_engine_is_registered() {
        let engine = create(constants::GROWTH_ENGINE).expect("growth engine");
        assert_eq!(engine.name(), constants::GROWTH_ENGINE);
        assert_eq!(
            available(),
            vec![constants::DEFAULT_ENGINE, constants::GROWTH_ENGINE]
        );
    }

    #[test]
    fn unknown_engines_list_the_alternatives() {
        let err = create("nope").unwrap_err().to_string();
        assert!(
            err.contains("nope")
                && err.contains(constants::DEFAULT_ENGINE)
                && err.contains(constants::GROWTH_ENGINE)
        );
    }
}
