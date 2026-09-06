//! Optimization engines.
//!
//! An engine takes a discrete [`Problem`] and returns a converged density
//! field. Three are registered below: `simp`, the finite element driven topology
//! optimizer, `growth`, a fast deterministic growth heuristic, and `solid`,
//! which optimizes nothing and hands back the domain itself. The registry is the
//! seam they plug into; neither the CLI, the post-processing nor the mesh
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
pub mod solid;
pub mod stall;
pub mod update;
pub mod wireframe;

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::config::{ReduceMethod, SymmetryParams};
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
    /// What an `[optimization.reduce]` schedule removed, or `None` when the run
    /// had no such table.
    pub reduce: Option<ReduceSummary>,
}

/// Why an engine stopped iterating.
///
/// Four of the five are outcomes a finished run reports, and all four are
/// finished runs: a stalled run, a capped run and a run that reached the end of
/// its reduction schedule export, are stress analysed and exit zero exactly as a
/// converged one does. What separates them is what the design *is* - an answer,
/// an iterate the problem will not improve on, the iterate the budget happened
/// to end on, or the stage a reduction schedule kept - and saying which is the
/// whole point of reporting them apart.
///
/// The fifth is not an outcome at all: it is the caller taking the run back.
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
    /// An `[optimization.reduce]` schedule ran out of stages: the design is the
    /// one the schedule kept - the lightest stage that held the target safety
    /// factor, or the fraction the run started from when no stage did. Which of
    /// the two it was, and at what safety factor, is [`ReduceSummary`]'s to say.
    ReduceComplete,
}

impl StopReason {
    /// True when the run reached a design it had stopped moving, rather than one
    /// of the three ways of stopping short of that.
    ///
    /// A finished reduction schedule counts: every stage of it ends on the same
    /// convergence test a plain run ends on, so the design that comes back is one
    /// that settled. Whether it *held* the safety factor it was aiming at is a
    /// different question, and [`ReduceSummary`] is where that one is answered.
    pub fn converged(self) -> bool {
        match self {
            StopReason::Converged | StopReason::ReduceComplete => true,
            StopReason::Stalled | StopReason::IterationCap | StopReason::Cancelled => false,
        }
    }

    /// Short label used in the run summary and the viewer panel.
    pub fn label(self) -> &'static str {
        match self {
            StopReason::Converged => "converged",
            StopReason::Stalled => "stalled",
            StopReason::IterationCap => "iteration cap",
            StopReason::Cancelled => "stopped",
            StopReason::ReduceComplete => "reduce complete",
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

/// What a finished `[optimization.reduce]` run took away, stage by stage.
///
/// The reduction schedule's counterpart of [`GrowthSummary`]: every volume target
/// the run asked for, what the design came back at, whether it still held the
/// target safety factor, and which of those stages is the design in the file.
/// `None` on every run without an `[optimization.reduce]` table.
#[derive(Debug, Clone, PartialEq)]
pub struct ReduceSummary {
    /// How material was taken away, as `[optimization.reduce] method` named it.
    pub method: ReduceMethod,
    /// Safety factor the schedule was asked to hold, `target_safety_factor`.
    pub target_safety_factor: f64,
    /// The stage whose design was exported: the lightest one that held the
    /// target, or the fraction the run started from when none did.
    ///
    /// A copy of its entry in [`ReduceSummary::stages`] rather than a second
    /// reading of the design - the exported fraction and safety factor are the
    /// stage's own - so a reader that only wants to know what shipped needs the
    /// record and not the schedule.
    pub exported: ReduceStage,
    /// One record per stage the schedule ran, in the order it ran them.
    pub stages: Vec<ReduceStage>,
}

impl ReduceSummary {
    /// True when the exported design does not hold the target safety factor.
    ///
    /// The one way that happens: no stage held it, not even the fraction the run
    /// started from, so what shipped is that starting design and the run has to
    /// say so rather than let a lighter-is-better summary read as a pass.
    pub fn missed_the_target(&self) -> bool {
        !self.exported.passed
    }
}

/// One stage of a reduction schedule: the volume target it was given, what it
/// converged to, and whether that design still held the target safety factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReduceStage {
    /// One-based stage number, the number the console and the report print.
    /// Stages are recorded in the order they ran, so it is also this record's
    /// position in [`ReduceSummary::stages`] plus one.
    pub index: usize,
    /// Volume fraction the stage was asked for.
    pub target_fraction: f64,
    /// Mean physical density over the design cells the stage converged to.
    pub achieved_fraction: f64,
    /// Iterations the stage spent reaching it.
    pub iterations: usize,
    /// Yield strength over the peak von Mises stress of the stage's design.
    /// `None` when no stress was solved for it - a stage whose load paths are
    /// broken is failed on the geometry alone, which costs no solve.
    pub safety_factor: Option<f64>,
    /// True when the stage held the target safety factor.
    pub passed: bool,
    /// True when the stage is one of the `refine_stages` bisections between the
    /// last target that held and the first that did not, rather than a step of
    /// the `ratio` schedule.
    pub refine: bool,
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

fn make_solid() -> Box<dyn Engine> {
    Box::new(solid::SolidEngine)
}

/// All engines growforge knows about, keyed by their config value.
const REGISTRY: &[(&str, EngineFactory)] = &[
    (constants::DEFAULT_ENGINE, make_simp),
    (constants::GROWTH_ENGINE, make_growth),
    (constants::SOLID_ENGINE, make_solid),
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
    }

    #[test]
    fn the_solid_engine_is_registered() {
        let engine = create(constants::SOLID_ENGINE).expect("solid engine");
        assert_eq!(engine.name(), constants::SOLID_ENGINE);
        // The whole registry, in registration order: the combo box of the
        // editor and every "available engines are ..." message read it.
        assert_eq!(
            available(),
            vec![
                constants::DEFAULT_ENGINE,
                constants::GROWTH_ENGINE,
                constants::SOLID_ENGINE
            ]
        );
    }

    #[test]
    fn every_stop_reason_is_labelled_and_says_whether_it_settled() {
        let labelled = [
            (StopReason::Converged, "converged", true),
            (StopReason::Stalled, "stalled", false),
            (StopReason::IterationCap, "iteration cap", false),
            (StopReason::Cancelled, "stopped", false),
            (StopReason::ReduceComplete, "reduce complete", true),
        ];
        for (reason, label, converged) in labelled {
            assert_eq!(reason.label(), label);
            assert_eq!(reason.converged(), converged, "{label}");
        }
    }

    #[test]
    fn a_reduce_summary_reads_its_verdict_off_the_exported_stage() {
        let stage = |index: usize, fraction: f64, factor: f64, passed: bool| ReduceStage {
            index,
            target_fraction: fraction,
            achieved_fraction: fraction,
            iterations: 40,
            safety_factor: Some(factor),
            passed,
            refine: false,
        };
        let passing = stage(2, 0.8, 5.3, true);
        let held = ReduceSummary {
            method: ReduceMethod::Continuation,
            target_safety_factor: 5.0,
            exported: passing,
            stages: vec![stage(1, 1.0, 7.1, true), passing],
        };
        assert!(!held.missed_the_target());
        // Nothing held, so the design in the file is the one the run started
        // from and the summary is the warning, whatever the stage list says.
        let start = stage(1, 1.0, 3.2, false);
        let missed = ReduceSummary {
            exported: start,
            stages: vec![start],
            ..held
        };
        assert!(missed.missed_the_target());
    }

    #[test]
    fn unknown_engines_list_the_alternatives() {
        let err = create("nope").unwrap_err().to_string();
        assert!(
            err.contains("nope")
                && err.contains(constants::DEFAULT_ENGINE)
                && err.contains(constants::GROWTH_ENGINE)
                && err.contains(constants::SOLID_ENGINE)
        );
    }
}
