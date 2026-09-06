//! Console reporting: per-iteration progress, problem summaries and mesh
//! statistics.
//!
//! Engines talk to a [`Reporter`] rather than to stdout so that later phases
//! (a viewer, a batch runner) can consume the same stream.

use crate::bench::BenchReport;
use crate::config::{
    IslandPolicy, ReduceMethodParams, SolverBackend, SolverParams, UpdateScheme, VoidPolicy,
};
use crate::constants;
use crate::engine::{ReduceStage, ReduceSummary};
use crate::flush::FlushReport;
use crate::mesh::clamp::ClampReport;
use crate::mesh::islands::{AnchorSet, IslandReport};
use crate::mesh::validate::MeshStats;
use crate::problem::Problem;
use crate::reinforce::ReinforceReport;
use crate::stress::StressOutcome;
use crate::trim::TrimReport;
use crate::voids::{SolidBody, VoidReport};

/// Stage a growth run is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthPhase {
    /// Routing the guaranteed load paths from the loads to the supports.
    Backbone,
    /// Growing the organic canopy by space colonization.
    Branching,
    /// Removing the branches that ended on nothing.
    Pruning,
    /// Sizing the struts and normalizing them against the volume target.
    Thickening,
}

impl GrowthPhase {
    /// Short label used in the console and in the viewer panel.
    pub fn label(self) -> &'static str {
        match self {
            GrowthPhase::Backbone => "backbone",
            GrowthPhase::Branching => "branching",
            GrowthPhase::Pruning => "pruning",
            GrowthPhase::Thickening => "thickening",
        }
    }
}

/// The part of a progress line that only the growth engine has.
///
/// The growth engine never evaluates a compliance, so the compliance oriented
/// fields of [`IterationStats`] say nothing about it; these are what it reports
/// instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrowthProgress {
    /// Stage the run is in.
    pub phase: GrowthPhase,
    /// Strut segments in the skeleton so far.
    pub segments: usize,
    /// Attraction points not yet consumed.
    pub attractors_remaining: usize,
}

/// The part of a progress line that only a material reduction run has.
///
/// Everything else on the line - the compliance, the fraction, the change - is
/// the inner loop of one stage of the schedule, which is the ordinary SIMP loop;
/// this is what says which stage they belong to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReduceProgress {
    /// One-based number of the stage now running, the number
    /// [`crate::engine::ReduceStage::index`] carries once it is recorded.
    pub stage: usize,
    /// Volume fraction this stage is driving the design to.
    pub target_fraction: f64,
    /// Stages that have finished and been recorded before this one.
    pub completed: usize,
}

/// One line of optimization progress.
#[derive(Debug, Clone)]
pub struct IterationStats {
    /// One-based iteration number.
    pub iteration: usize,
    /// Weighted compliance objective in N*mm. Zero for the growth engine, which
    /// never evaluates one.
    pub compliance: f64,
    /// Mean physical density over the design cells.
    pub volume_fraction: f64,
    /// Largest local volume fraction over the design cells of the design this
    /// iteration **analysed** - the one its compliance belongs to, and the one
    /// the step was priced against - or `None` when no
    /// `[optimization.local_volume]` cap is active.
    ///
    /// The true worst neighbourhood rather than the p-mean the constraint is
    /// stated on, which under-estimates it; see
    /// [`crate::engine::local_volume`].
    pub worst_local_fraction: Option<f64>,
    /// Largest absolute design variable change in this iteration. Zero for the
    /// growth engine, which has no design variables.
    pub max_change: f64,
    /// Conjugate gradient iterations spent per load case. Empty for the growth
    /// engine, which solves nothing.
    pub cg_iterations: Vec<usize>,
    /// Seconds elapsed since the optimization started.
    pub elapsed_s: f64,
    /// Growth engine progress, when the growth engine produced this line. The
    /// SIMP engine leaves it `None`, and every consumer keeps its existing
    /// format for such a line.
    pub growth: Option<GrowthProgress>,
    /// Which stage of an `[optimization.reduce]` schedule this line belongs to,
    /// or `None` on a run without one - where the line is the one it always was.
    pub reduce: Option<ReduceProgress>,
}

/// Sink for engine progress.
pub trait Reporter: Sync {
    /// Called once per optimization iteration.
    fn iteration(&self, stats: &IterationStats);
    /// Called for one-off messages such as convergence notices.
    fn note(&self, message: &str);
    /// Called once per optimization iteration with the physical density of
    /// every cell, in grid element order, right after [`Reporter::iteration`].
    ///
    /// This is the observer seam the viewer hangs off. The densities are
    /// borrowed, never copied, so a reporter that does not want them (the
    /// default implementation, the console and the silent reporter) costs the
    /// engine one virtual call per iteration and nothing else.
    fn densities(&self, _stats: &IterationStats, _densities: &[f64]) {}

    /// True once the caller no longer wants the result of this run.
    ///
    /// The cooperative cancellation seam, and the counterpart of
    /// [`Reporter::densities`]: an engine that iterates checks it between
    /// iterations and stops early, returning the design it had reached with
    /// `converged` false and `iterations` counting what it really ran. It is
    /// never an error - the caller asked - and nothing is written either way,
    /// because writing happens after an engine returns.
    ///
    /// The default is false, so every existing reporter (the console, the
    /// silent one, the viewer's mirror) leaves every run exactly as it was. The
    /// editor's auto-regrow is the one caller that returns true, when an edit
    /// has made the run it started obsolete.
    fn cancelled(&self) -> bool {
        false
    }
}

/// Writes progress to stdout.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConsoleReporter {
    quiet: bool,
}

impl ConsoleReporter {
    /// Build a reporter; `quiet` suppresses the per-iteration lines but keeps
    /// notes.
    pub fn new(quiet: bool) -> Self {
        ConsoleReporter { quiet }
    }
}

/// The per-iteration line of every engine that has design variables, built as a
/// string so its format can be read back by a test rather than off a terminal.
fn iteration_line(stats: &IterationStats) -> String {
    let cg: Vec<String> = stats.cg_iterations.iter().map(|c| c.to_string()).collect();
    // The local cap's own column exists only while the cap does, and the stage
    // prefix only while a reduction schedule does, so a run with neither prints
    // exactly the line it always did.
    let local = match stats.worst_local_fraction {
        Some(worst) => format!("  local {worst:>6.4}"),
        None => String::new(),
    };
    let stage = match stats.reduce {
        Some(reduce) => format!("stage {:>2}  ", reduce.stage),
        None => String::new(),
    };
    format!(
        "{stage}iter {:>4}  compliance {:>12.6e}  vol {:>6.4}{local}  change {:>8.5}  cg [{}]  \
         {:>7.2} s",
        stats.iteration,
        stats.compliance,
        stats.volume_fraction,
        stats.max_change,
        cg.join(", "),
        stats.elapsed_s
    )
}

/// What one finished stage of a reduction schedule reports: the target it was
/// given, the design it came back with, and the verdict on it.
///
/// A note rather than a progress line - it is said once, when the stage ends -
/// so it reaches the console and the editor's panel through the one channel
/// every other run-time announcement uses.
pub fn reduce_stage_note(stage: &ReduceStage) -> String {
    format!(
        "reduce stage {}{}: target {:.4}, fraction {:.4}, {} iterations, safety factor {}, {}",
        stage.index,
        if stage.refine { " (refinement)" } else { "" },
        stage.target_fraction,
        stage.achieved_fraction,
        stage.iterations,
        safety_factor(stage.safety_factor),
        if stage.passed { "pass" } else { "fail" }
    )
}

/// What the whole schedule reports when it is done: which stage is the design in
/// the file, and how its safety factor stands against the target.
///
/// A schedule that never found a design holding the target exports the fraction
/// it started from, which is a warning: the part is the one the run was given,
/// not a lighter one that passed.
pub fn reduce_summary_note(summary: &ReduceSummary) -> String {
    let stage = summary.exported;
    if summary.missed_the_target() {
        // What is left to try depends on what shipped. A design with voids in it
        // can be given more material; one at full density is already every cell
        // the domain has, and telling its user to add material is telling them
        // to do the one thing they cannot.
        let remedy = if stage.achieved_fraction < constants::DENSITY_MAX {
            "lower target_safety_factor, or give the design more material or a stiffer material \
             to hold it with"
        } else {
            "the design is already solid, so lower target_safety_factor, enlarge the domain, \
             relieve the load, or choose a stiffer or stronger material"
        };
        return format!(
            "warning: [optimization.reduce]: no stage held the target safety factor of {:.2}; \
             exporting stage {} at fraction {:.4}, safety factor {} - {remedy}",
            summary.target_safety_factor,
            stage.index,
            stage.achieved_fraction,
            safety_factor(stage.safety_factor)
        );
    }
    format!(
        "reduce: exported stage {} at fraction {:.4}, safety factor {} >= {:.2}",
        stage.index,
        stage.achieved_fraction,
        safety_factor(stage.safety_factor),
        summary.target_safety_factor
    )
}

/// What the part in the file measures against the target its schedule was held
/// to, when it no longer holds it.
///
/// The schedule chooses a design and stops; the `[output]` passes then trim,
/// flush and reinforce the field it chose, and the stress table beside the STL
/// describes what they left. That part is the deliverable, so a target those
/// passes cost it is a warning carrying both numbers and every pass that ran
/// between them - `passes`, in the order the pipeline runs them.
///
/// `None` when the exported part still holds the target, which is the ordinary
/// outcome and one [`reduce_summary_note`] has already reported.
pub fn reduce_finished_note(summary: &ReduceSummary, passes: &[&str]) -> Option<String> {
    if summary.finished_meets_target() {
        return None;
    }
    // What stands between the stage's verdict and the part is what there is to
    // do about it: those passes when any of them ran, and the cavity pass every
    // export resolves the field with when none did.
    let tail = match passes.split_last() {
        Some((last, [])) => format!(
            "the [output] {last} pass ran on that design afterwards - turn it off, or raise \
             target_safety_factor to leave it the margin it spends"
        ),
        Some((last, rest)) => format!(
            "the [output] {} and {last} passes ran on that design afterwards - turn them off, or \
             raise target_safety_factor to leave them the margin they spend",
            rest.join(", ")
        ),
        None => "nothing but the export's cavity pass ran on that design afterwards - raise \
                 target_safety_factor, or check the [output] voids policy"
            .to_string(),
    };
    Some(format!(
        "warning: [optimization.reduce]: the part in the file measures a safety factor of {}, \
         below the target of {:.2}; the schedule chose stage {}, which measured {}, and {tail}",
        safety_factor(summary.finished_safety_factor),
        summary.target_safety_factor,
        summary.exported.index,
        safety_factor(summary.exported.safety_factor)
    ))
}

/// A stage's safety factor as the console says it, with the same `n/a` the
/// stress table uses for a design no yield strength can be measured against.
fn safety_factor(factor: Option<f64>) -> String {
    match factor {
        Some(factor) => format!("{factor:.2}"),
        None => "n/a".to_string(),
    }
}

impl Reporter for ConsoleReporter {
    fn iteration(&self, stats: &IterationStats) {
        if self.quiet {
            return;
        }
        if let Some(growth) = stats.growth {
            println!(
                "step {:>4}  {:<11} segments {:>7}  attractors {:>7}  vol {:>6.4}  {:>7.2} s",
                stats.iteration,
                growth.phase.label(),
                growth.segments,
                growth.attractors_remaining,
                stats.volume_fraction,
                stats.elapsed_s
            );
            return;
        }
        println!("{}", iteration_line(stats));
    }

    fn note(&self, message: &str) {
        println!("{message}");
    }
}

/// Discards everything; used by tests and library callers.
#[derive(Debug, Clone, Copy, Default)]
pub struct SilentReporter;

impl Reporter for SilentReporter {
    fn iteration(&self, _stats: &IterationStats) {}
    fn note(&self, _message: &str) {}
}

/// Print any non-fatal problems found while building the model.
pub fn print_warnings(warnings: &[String]) {
    for w in warnings {
        eprintln!("warning: {w}");
    }
}

/// Print the problem summary shown by `growforge check` and at the start of a
/// run.
pub fn print_problem_summary(problem: &Problem) {
    let g = &problem.grid;
    let bounds = g.bounds();
    println!("project        {}", problem.name);
    println!("engine         {}", problem.engine);
    println!(
        "material       E = {} MPa, nu = {}, rho = {} g/cm3{}",
        problem.material.youngs_modulus_mpa,
        problem.material.poisson_ratio,
        problem.material.density_g_cm3,
        match problem.material.yield_strength_mpa {
            Some(y) => format!(", yield = {y} MPa"),
            None => String::new(),
        }
    );
    println!(
        "grid           {} x {} x {} cells at {:.4} mm",
        g.nx, g.ny, g.nz, g.h
    );
    println!(
        "grid bounds    [{:.3}, {:.3}, {:.3}] .. [{:.3}, {:.3}, {:.3}] mm",
        bounds.min[0], bounds.min[1], bounds.min[2], bounds.max[0], bounds.max[1], bounds.max[2]
    );
    println!(
        "cells          {} total, {} design, {} solid, {} void",
        g.n_cells(),
        problem.counts.design,
        problem.counts.solid,
        problem.counts.void
    );
    println!(
        "nodes          {} ({} degrees of freedom)",
        g.n_nodes(),
        g.n_dof()
    );
    let fixed_dof = problem.fixed.iter().filter(|f| **f).count();
    println!("constrained    {fixed_dof} degrees of freedom");
    for s in &problem.supports {
        let dirs: Vec<String> = s
            .directions
            .iter()
            .map(|d| format!("{d:?}").to_lowercase())
            .collect();
        println!(
            "  support {}    {} nodes, directions [{}]",
            s.index,
            s.node_count,
            dirs.join(", ")
        );
    }
    for case in &problem.load_cases {
        println!("  loadcase \"{}\"  weight {}", case.name, case.weight);
        for (i, load) in case.loads.iter().enumerate() {
            let detail = if load.detail.is_empty() {
                String::new()
            } else {
                format!(", {}", load.detail)
            };
            println!(
                "    load {}     {} on {} nodes{detail}",
                i + 1,
                load.kind,
                load.node_count
            );
        }
    }
    let solid_volume = problem.counts.active() as f64 * problem.cell_volume_mm3();
    if problem.is_solid() {
        // No mass target was set and none is invented: the domain is the target,
        // which is the whole of what this engine is.
        println!("domain volume  {solid_volume:.1} mm3 fully solid, which is what is exported");
    } else {
        println!(
            "domain volume  {:.1} mm3 fully solid, target {:.1} mm3 at mass_fraction {}",
            solid_volume,
            (problem.counts.solid as f64
                + problem.counts.design as f64 * problem.optimization.mass_fraction)
                * problem.cell_volume_mm3(),
            problem.optimization.mass_fraction
        );
    }
    // The density filter, the self-supporting filter and the update scheme are
    // stages of the SIMP density chain. A growth run has none of them and reports
    // the knobs it does have in their place; a solid run has nothing to report at
    // all, and says so in one line rather than echoing defaults nothing will read.
    if problem.is_solid() {
        println!("solid          every design cell fully dense; nothing is optimized");
    } else {
        match &problem.growth {
            Some(growth) => {
                println!(
                    "growth seed    {} (same seed and config, same STL)",
                    growth.seed
                );
                println!(
                    "growth strut   {:.3} .. {:.3} mm radius from min_feature_mm {:.3}, Murray n = {}",
                    0.5 * problem.optimization.min_feature_mm,
                    growth.max_radius_mm,
                    problem.optimization.min_feature_mm,
                    growth.murray_exponent
                );
                println!(
                    "growth field   {:.3} mm step, {:.3} mm kill, {:.3} mm attraction, {:.2} attractors/cm3, {} steps max",
                    growth.step_mm,
                    growth.kill_radius_mm,
                    growth.attraction_radius_mm,
                    growth.attractor_per_cm3,
                    growth.max_steps
                );
                println!(
                    "growth pruning {}",
                    if growth.prune {
                        "on, branches that end on nothing are removed"
                    } else {
                        "off, branches that end on nothing are kept"
                    }
                );
                println!(
                    "growth symmetry {}",
                    match growth.symmetry {
                        Some(symmetry) => format!(
                            "{}, {} sectors about the domain centre [{:.3}, {:.3}, {:.3}]; one is \
                         grown and copied",
                            symmetry.describe(),
                            symmetry.sectors(),
                            0.5 * (bounds.min[0] + bounds.max[0]),
                            0.5 * (bounds.min[1] + bounds.max[1]),
                            0.5 * (bounds.min[2] + bounds.max[2])
                        ),
                        None => "off, the whole domain is grown".to_string(),
                    }
                );
            }
            None => {
                // First in the block because it changes what the mass target
                // line above it means: under a reduction that fraction is where
                // the run starts rather than where it has to end up.
                println!(
                    "reduce         {}",
                    match problem.optimization.reduce {
                        Some(reduce) => format!(
                            "{}, material removed until the safety factor reaches {:.3}; the \
                             volume target starts at {} and takes {:.3} of itself a stage down to \
                             a floor of {:.3}, {}; every stage of it runs up to the run's own \
                             max_iterations of {} iterations{}",
                            reduce.method.kind().label(),
                            reduce.target_safety_factor,
                            problem.optimization.mass_fraction,
                            reduce.ratio,
                            reduce.min_mass_fraction,
                            match reduce.refine_stages {
                                0 => "and the lightest target that holds is exported unrefined"
                                    .to_string(),
                                1 => "then 1 bisection between the last target that holds and the \
                                      first that does not"
                                    .to_string(),
                                n => format!(
                                    "then {n} bisections between the last target that holds and \
                                     the first that does not"
                                ),
                            },
                            problem.optimization.max_iterations,
                            match reduce.method {
                                ReduceMethodParams::Beso {
                                    evolution_rate,
                                    add_ratio,
                                } => format!(
                                    "; each iteration removes {evolution_rate:.3} of the volume \
                                     and lets up to {add_ratio:.3} of it back"
                                ),
                                ReduceMethodParams::Continuation => String::new(),
                            }
                        ),
                        None => "off, mass_fraction is the target the run meets".to_string(),
                    }
                );
                println!(
                    "filter radius  {:.3} mm ({:.2} voxels) from min_feature_mm {:.3}",
                    problem.optimization.filter_radius_mm,
                    problem.optimization.filter_radius_mm / g.h,
                    problem.optimization.min_feature_mm
                );
                println!(
                    "overhang       {}",
                    match problem.optimization.overhang {
                        Some(direction) => format!(
                            "self-supporting filter, build direction {}, 45 degree limit",
                            direction.label()
                        ),
                        None => "off".to_string(),
                    }
                );
                println!(
                    "update         {}",
                    // The evolutionary method is its own update, so the scheme
                    // this line otherwise names is not the one that will run -
                    // and a configuration cannot name one beside it either.
                    match (
                        problem.optimization.reduce.map(|reduce| reduce.method),
                        problem.optimization.update,
                    ) {
                        (Some(ReduceMethodParams::Beso { .. }), _) =>
                            "beso, the evolutionary update of [optimization.reduce]",
                        (_, UpdateScheme::Oc) =>
                            "oc, optimality criteria (the reproducible default)",
                        (_, UpdateScheme::Mma) => "mma, method of moving asymptotes",
                    }
                );
                println!(
                    "local volume   {}",
                    match problem.optimization.local_volume {
                        Some(cap) => format!(
                            "no neighbourhood of radius {:.3} mm may hold more than {:.3} of its \
                         material, aggregated by a p-mean of exponent {}",
                            cap.radius_mm,
                            cap.max_fraction,
                            constants::LOCAL_VOLUME_AGGREGATION_EXPONENT
                        ),
                        None => "off".to_string(),
                    }
                );
                println!(
                    "wireframe      {}",
                    match problem.optimization.wireframe {
                        Some(wire) => format!(
                            "guide of radius {:.3} mm seeded at density {:.3}, held as a floor for {} \
                         iterations{}",
                            wire.radius_mm,
                            wire.seed_density,
                            wire.hold_iterations,
                            // The claim the summary must not make is the one this
                            // configuration cannot keep: a floor that outlasts the
                            // budget is never released, and the design carries the
                            // wire as forced material.
                            if wire.hold_iterations < problem.optimization.max_iterations {
                                " then released"
                            } else {
                                ", which is the whole iteration budget, so it is never released"
                            }
                        ),
                        None => "off".to_string(),
                    }
                );
            }
        }
    }
    println!("solver         {}", solver_line(&problem.solver));
    println!(
        "enclosed voids {}",
        match problem.output.voids {
            VoidPolicy::Warn => "warn",
            VoidPolicy::Fill => "fill before meshing",
        }
    );
    println!(
        "islands        {}",
        match problem.output.islands {
            IslandPolicy::Cull => "cull the floating fragments of the exported surface",
            IslandPolicy::Keep => "keep every fragment the surface came out in",
        }
    );
    println!(
        "memory est.    {:.1} MiB",
        problem.estimated_memory_bytes() as f64 / constants::BYTES_PER_MIB
    );
    println!("output         {}", problem.output.stl_path.display());
}

/// The `solver` line of the problem summary: what the run's linear solves will
/// really do, rather than what the configuration happens to say.
///
/// Both qualifications exist for that reason. The **fallback** note is there
/// because the default backend is soft: it may already have been resolved to
/// something else, and if it has not been yet it still may be when the device is
/// opened, so the line says so rather than naming a backend the run might not
/// use. The **tolerance** is named for the same reason and only when it is not
/// [`constants::CG_RELATIVE_TOLERANCE`]: a run whose solves stop somewhere other
/// than where every other run of this crate stops them is a fact about that run,
/// and a loosened target that appeared nowhere would be a number nobody could
/// account for afterwards. A configuration that says nothing - or says the
/// default - prints exactly what it always printed.
///
/// Built rather than printed so the wording can be asserted on: the summary is
/// what a user reads a run's terms off, and it is checked the way
/// [`crate::stress::disconnected_bodies_note`] is.
fn solver_line(solver: &SolverParams) -> String {
    format!(
        "{} backend{}{}{}",
        solver.backend.label(),
        match solver.backend {
            // Both promises carry what they are conditional on, because the
            // line is the run's own account of itself: the CPU's reductions are
            // parallel, so its last bits are a function of the pool size the
            // way the GPU's are of the device (see the README's
            // reproducibility section, and `SolverBackend`'s own docs).
            SolverBackend::Cpu => ", bit-for-bit reproducible for this build and thread count",
            SolverBackend::Gpu =>
                ", single precision with f64 refinement; reproducible on this machine only",
        },
        match (solver.fell_back, solver.explicit) {
            (Some(reason), _) => format!(" ({})", reason.note()),
            (None, false) if solver.backend != SolverBackend::Cpu =>
                " (the default; falls back to the cpu if this machine has no adapter)".to_string(),
            (None, _) => String::new(),
        },
        if solver.tolerance == constants::CG_RELATIVE_TOLERANCE {
            String::new()
        } else {
            // The spelling the editor writes and the README uses, so the number
            // in the summary is the number in the file.
            format!(", tolerance {:e}", solver.tolerance)
        }
    )
}

/// Print what the enclosed cavity pass found and did.
pub fn print_void_report(report: &VoidReport) {
    if report.voids.is_empty() {
        println!("enclosed voids none");
        return;
    }
    let filled = report.voids.iter().filter(|v| report.was_filled(v)).count();
    println!(
        "enclosed voids {} found, {} filled, {} left in the exported surface",
        report.voids.len(),
        filled,
        report.remaining()
    );
    for (index, cavity) in report.voids.iter().enumerate() {
        println!(
            "  void {:<8} {:.1} mm3 in {} cells, centroid [{:.2}, {:.2}, {:.2}] mm, {}",
            index + 1,
            cavity.volume_mm3,
            cavity.cells,
            cavity.centroid[0],
            cavity.centroid[1],
            cavity.centroid[2],
            if report.was_filled(cavity) {
                "filled"
            } else if cavity.fillable {
                "exported as a sealed cavity"
            } else {
                "exported as a sealed cavity (overlaps a keepout, never filled)"
            }
        );
    }
    if report.filled_cells > 0 {
        println!(
            "  filled       {:.1} mm3, adding {:.3} g",
            report.filled_volume_mm3, report.added_mass_g
        );
    }
}

/// Print what the `[output] trim` pass removed, or why it removed nothing.
///
/// `None` is the default `trim = "off"`, where the pass never ran and there is
/// nothing to say; a run that trimmed nothing under `trim = "stress"` is a
/// different statement and is printed. A pass the connectivity guard refused
/// prints as a warning: the part in the file is the untrimmed one, which is the
/// safe outcome but not the one that was asked for.
///
/// The stresses quoted are the *pre-trim* ones the pass judged the material on.
/// They are the only place those numbers survive - the field they came from is
/// re-analysed afterwards, and the table further down describes the part that
/// shipped rather than the one the criterion was read off.
pub fn print_trim_report(report: Option<&TrimReport>) {
    let Some(report) = report else {
        return;
    };
    for (index, note) in report.notes().iter().enumerate() {
        let label = if index == 0 { "trim" } else { "" };
        println!("{label:<14} {note}");
    }
}

/// Print what the `[output] flush` pass filled out to the surfaces the part's
/// walls rest on.
///
/// `None` is the default `flush = "off"`, where the pass never ran and there is
/// nothing to say; a run that found no wall standing short of anything under
/// `flush = "walls"` is a different statement and is printed. So is the line
/// that says what the fill joined to a surface, which is the pass's own caveat
/// and not a defect in the result.
///
/// Printed between [`print_trim_report`] and [`print_reinforce_report`], which
/// is the order the pipeline ran them in: what was freed, what was put back
/// where the drawing had it, and what was spent on the arms that cannot be
/// printed.
pub fn print_flush_report(report: Option<&FlushReport>) {
    let Some(report) = report else {
        return;
    };
    for (index, note) in report.notes().iter().enumerate() {
        let label = if index == 0 { "flush" } else { "" };
        println!("{label:<14} {note}");
    }
}

/// Print what the `[output] reinforce` pass thickened, and what it could not.
///
/// `None` is the default `reinforce = "off"`, where the pass never ran and there
/// is nothing to say; a run that found nothing under the floor is a different
/// statement and is printed. A place the fill could not reach - a member pressed
/// against a keepout or the edge of the domain - prints as a warning: the part in
/// the file is still thin there.
///
/// Printed beside [`print_trim_report`], because the two are the two halves of
/// one exchange: the trim frees the mass of what carries nothing, and this spends
/// mass on what cannot be printed.
pub fn print_reinforce_report(report: Option<&ReinforceReport>) {
    let Some(report) = report else {
        return;
    };
    for (index, note) in report.notes().iter().enumerate() {
        let label = if index == 0 { "reinforce" } else { "" };
        println!("{label:<14} {note}");
    }
}

/// Print what the part in the file measures against the target an
/// `[optimization.reduce]` schedule was held to, when the finishing passes cost
/// it that target.
///
/// Nothing at all on a run without the table, and nothing on one whose exported
/// part still holds it - that is what the schedule's own summary line said while
/// the run was going. Printed beside the stress table because the factor it
/// quotes is that table's, and it names the passes that ran between the
/// schedule's verdict and the file.
pub fn print_reduce_finish(
    summary: Option<&ReduceSummary>,
    trim: Option<&TrimReport>,
    flush: Option<&FlushReport>,
    reinforce: Option<&ReinforceReport>,
) {
    let Some(summary) = summary else {
        return;
    };
    // A pass that ran is a pass with a report: `None` is the `off` of all three.
    let passes: Vec<&str> = [
        trim.map(|_| "trim"),
        flush.map(|_| "flush"),
        reinforce.map(|_| "reinforce"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if let Some(note) = reduce_finished_note(summary, &passes) {
        let label = "reduce";
        println!("{label:<14} {note}");
    }
}

/// Print how many separate bodies of material the exported *field* holds.
///
/// One is a part. Anything more is a floating island: material joined to nothing
/// that a slicer lays down as a loose object, carrying no load and holding
/// nothing up. Like the cavity report it warns rather than failing: the part is
/// still the part, and what to do about the island is the user's call.
///
/// This is the cell level reading, and it is deliberately not the one
/// [`print_island_report`] prints. It flood fills the density field at the iso
/// level, which is the object the stress solve and the mass figures describe;
/// the surface that ships is extracted from the node averages of that field and
/// can hold shells this fill has no way to see. Both lines are printed, and each
/// says which object it is talking about, because reading one as the other is
/// exactly how a run came to report one connected body while its STL carried
/// several.
pub fn print_solid_report(bodies: &[SolidBody]) {
    match bodies.len() {
        0 => println!("field bodies   none: the density field holds no material at all"),
        1 => println!("field bodies   1 connected body in the density field"),
        count => {
            println!(
                "field bodies   {count} connected bodies in the density field, so {} of them \
                 float free of the part",
                count - 1
            );
            for (index, body) in bodies.iter().enumerate().skip(1) {
                println!(
                    "  island {:<6} {:.1} mm3 in {} cells, centroid [{:.2}, {:.2}, {:.2}] mm",
                    index,
                    body.volume_mm3,
                    body.cells,
                    body.centroid[0],
                    body.centroid[1],
                    body.centroid[2]
                );
            }
            println!(
                "  warning      these bodies are not joined to the largest one; each prints as a \
                 separate loose object. Whether one of them is debris or a keepin boss the \
                 configuration asked for is decided on the mesh, not here: see the mesh bodies \
                 line below"
            );
        }
    }
}

/// The one line that says what the exported mesh came out in.
///
/// Built rather than printed so the format itself can be checked: it is the
/// line a user reads to tell a part with a cavity from a part with a fragment
/// in the box next to it.
fn island_headline(report: &IslandReport) -> String {
    let cavities = match report.cavity_shells {
        0 => String::new(),
        1 => " (+1 cavity shell)".to_string(),
        n => format!(" (+{n} cavity shells)"),
    };
    let fragments = if report.fragments.is_empty() {
        String::new()
    } else {
        format!(
            "; {} {} floating fragment{} ({:.3} mm3 in total, largest {:.3} mm3)",
            match report.policy {
                IslandPolicy::Cull => "culled",
                IslandPolicy::Keep => "kept",
            },
            report.fragments.len(),
            if report.fragments.len() == 1 { "" } else { "s" },
            report.fragment_volume_mm3(),
            report.largest_fragment_mm3().unwrap_or_default()
        )
    };
    format!(
        "mesh bodies    {} in the exported surface{cavities}{fragments}",
        report.bodies.len()
    )
}

/// What anchors a body, as the report says it: `support, load` or `nothing
/// declared`.
fn anchor_labels(anchors: AnchorSet) -> String {
    if anchors.is_empty() {
        return "nothing declared".to_string();
    }
    anchors
        .kinds()
        .map(|kind| kind.label())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Volume below which a separately exported body counts as tiny: the sphere of
/// diameter `min_feature_mm`, scaled by
/// [`constants::ISLAND_TINY_BODY_SPHERES`].
///
/// The smallest lump the density filter can resolve at all, so a body under it
/// is a stub rather than a feature. It decides nothing about what ships; it
/// decides what is said about it.
fn tiny_body_mm3(min_feature_mm: f64) -> f64 {
    constants::ISLAND_TINY_BODY_SPHERES * std::f64::consts::PI / 6.0 * min_feature_mm.powi(3)
}

/// Indices of the exported bodies that are anchored but tiny.
///
/// Only when more than one body ships: a single tiny body is a tiny part, which
/// is what was asked for, while a tiny body *beside* another one is a loose
/// piece in the same file - kept, because a declared region touches it and
/// geometry a declared region asked for is never removed, and said out loud,
/// because nobody should find it in a slicer instead.
fn tiny_bodies(report: &IslandReport, min_feature_mm: f64) -> Vec<usize> {
    if report.bodies.len() < 2 {
        return Vec::new();
    }
    let limit = tiny_body_mm3(min_feature_mm);
    report
        .bodies
        .iter()
        .enumerate()
        .filter(|(_, body)| body.volume_mm3 < limit)
        .map(|(index, _)| index)
        .collect()
}

/// What the report says a fragment is, which is not one sentence: what a
/// fragment *is* depends on whether anything declared turned out to be inside
/// it.
///
/// "Nothing declared asked for it" is the honest reading only when nothing
/// declared is inside what left. A region spanning two disconnected lobes has
/// its material in both of them and can anchor only one, and the sentence would
/// be false about the other; that case says what really happened and points at
/// the warnings that name it.
fn fragment_note(report: &IslandReport) -> &'static str {
    match (
        report.policy,
        report.culled_fragments,
        report.culled_inside.is_empty(),
    ) {
        (IslandPolicy::Cull, 0, _) => {
            "no body reaches a support, a load or a keepin region, so nothing here can be told \
             from the part and nothing was removed"
        }
        (IslandPolicy::Cull, _, true) => {
            "a fragment touches no support, load or keepin region, so nothing declared asked for \
             it; it was removed before the surface was validated and written. islands = \"keep\" \
             exports the extracted surface, fragments and all"
        }
        (IslandPolicy::Cull, _, false) => {
            "a fragment reaches no support, load or keepin region through its surface or the \
             material it holds, so it was removed before the surface was validated and written - \
             but some of what left lies inside a declared region, named below, and is not debris. \
             islands = \"keep\" exports the extracted surface, fragments and all"
        }
        (IslandPolicy::Keep, _, _) => {
            "islands = \"keep\": the fragments are in the file, each of them a separate loose \
             object a slicer will lay down"
        }
    }
}

/// Print what the exported *mesh* came out in, and what the island policy did
/// about it.
///
/// The counterpart of [`print_solid_report`] one stage later: connected
/// components of the surface that was written, rather than of the field it came
/// from. A cavity shell is counted next to the body that encloses it rather than
/// as a body of its own, so `voids = "warn"` and this line never disagree about
/// how many parts there are. Every body says what anchors it, because that is
/// what decided it was kept, and a support or load region that no body reaches -
/// or a body that ships loose and is too small to be a feature - is named as
/// loudly as anything here gets.
///
/// `min_feature_mm` is the `[optimization]` value the size of a body is read
/// against; nothing is culled by it.
pub fn print_island_report(report: &IslandReport, min_feature_mm: f64) {
    println!("{}", island_headline(report));
    if report.bodies.len() > 1 {
        for (index, body) in report.bodies.iter().enumerate() {
            println!(
                "  body {:<8} {:.3} mm3 in {} triangles, holds {}",
                index + 1,
                body.volume_mm3,
                body.triangles,
                anchor_labels(body.anchors)
            );
        }
    }
    for (index, fragment) in report
        .fragments
        .iter()
        .take(constants::ISLAND_REPORT_MAX_LINES)
        .enumerate()
    {
        println!(
            "  fragment {:<4} {:.3} mm3 in {} triangles, centroid [{:.2}, {:.2}, {:.2}] mm",
            index + 1,
            fragment.volume_mm3,
            fragment.triangles,
            fragment.centroid[0],
            fragment.centroid[1],
            fragment.centroid[2]
        );
    }
    if let Some(rest) = report
        .fragments
        .len()
        .checked_sub(constants::ISLAND_REPORT_MAX_LINES)
        .filter(|rest| *rest > 0)
    {
        println!("  and {rest} more");
    }
    if !report.fragments.is_empty() {
        println!("  note         {}", fragment_note(report));
    }
    for (fragment, region) in &report.culled_inside {
        println!(
            "  warning      culled fragment {} ({:.3} mm3) lies inside {}: material a declared \
             region asked for was disconnected and removed, so the load path may be incomplete",
            fragment + 1,
            report.fragments[*fragment].volume_mm3,
            region
        );
    }
    for region in &report.unserved {
        println!(
            "  warning      {region} is reached by nothing in the exported surface: the part that \
             ships does not connect it"
        );
    }
    let tiny = tiny_bodies(report, min_feature_mm);
    for &index in &tiny {
        let body = &report.bodies[index];
        println!(
            "  warning      body {} holds {} and is kept for it, but encloses {:.3} mm3 against \
             the {:.3} mm3 of a sphere of min_feature_mm ({:.3} mm): it ships as a separate loose \
             piece",
            index + 1,
            anchor_labels(body.anchors),
            body.volume_mm3,
            tiny_body_mm3(min_feature_mm),
            min_feature_mm
        );
    }
    if !tiny.is_empty() {
        println!(
            "  remedy       a body that small is a design that did not resolve rather than a \
             feature; more iterations, a coarser min_feature_mm or a higher mass_fraction join it \
             to the part. It is not removed: a declared region asked for the material in it"
        );
    }
}

/// Print what the `[output] boundaries` clamp moved onto the analytic surfaces.
///
/// `None` is `boundaries = "voxel"`, where the pass never ran and there is
/// nothing to say. Under the default `"exact"` there is always a line, including
/// the one that says nothing needed correcting: the pass changes the exported
/// geometry, and a run that silently did or did not do so would be a run whose
/// STL cannot be accounted for.
///
/// A vertex the pass could not correct prints as a warning. The surface still
/// crosses a boundary there, which is the one outcome of this pass that a user
/// has to act on rather than read.
///
/// `solid` is [`Problem::is_solid`] and `flushing` is [`Problem::is_flushing`];
/// between them they decide whether the pass's other finding - vertices resting
/// near a boundary without sitting on it - prints as a warning, as a line about
/// the flush pass falling short, or not at all. See [`ClampReport::notes`]. They
/// are passed rather than read off a problem here because this printer is handed
/// a report alone, as every printer in this module is.
pub fn print_clamp_report(report: Option<&ClampReport>, solid: bool, flushing: bool) {
    let Some(report) = report else {
        return;
    };
    for (index, note) in report.notes(solid, flushing).iter().enumerate() {
        let label = if index == 0 { "boundaries" } else { "" };
        println!("{label:<14} {note}");
    }
}

/// The line the stress block carries when the exported surface came out in more
/// than one piece, or `None` when it came out as one connected part.
///
/// Built rather than printed for the reason [`island_headline`] is, and worded by
/// [`crate::stress::disconnected_bodies_note`] rather than here: the panel, the
/// editor's console echo and this block say one sentence between them, laid out
/// three ways.
fn stress_body_warning(bodies: usize) -> Option<String> {
    crate::stress::disconnected_bodies_note(bodies).map(|note| format!("  warning      {note}"))
}

/// Print the von Mises stress table, or the reason there is none.
///
/// `bodies` is how many separate bodies the exported surface holds - the length
/// of [`IslandReport::bodies`] of the surface that was written - which is what
/// decides whether the table is a reading of one part or of several. The count
/// comes from the export rather than from the density field: a run reports on the
/// file it left behind.
pub fn print_stress_report(outcome: &StressOutcome, bodies: usize) {
    let report = match outcome {
        StressOutcome::Available(report) => report,
        StressOutcome::Unavailable(reason) => {
            println!("stress         report unavailable: {reason}");
            println!(
                "               the density field, the STL and the cavity report above are \
                 unaffected"
            );
            return;
        }
    };
    println!(
        "stress         {} elements at density >= {:.2}, recovered with the full solid modulus",
        report.evaluated_cells, report.density_threshold
    );
    // Above the table rather than under it: it is the caveat on every number in
    // it, and the safety factor is what a reader stops at.
    if let Some(warning) = stress_body_warning(bodies) {
        println!("{warning}");
    }
    println!(
        "  {:<20} {:>12} {:>12} {:>12} {:>10}",
        "loadcase",
        "max MPa",
        format!("p{:.0} MPa", constants::STRESS_PERCENTILE * 100.0),
        format!("top {:.0}% MPa", constants::STRESS_TOP_FRACTION * 100.0),
        "safety"
    );
    for case in &report.cases {
        println!(
            "  {:<20} {:>12.4} {:>12.4} {:>12.4} {:>10}",
            case.name,
            case.max_mpa,
            case.percentile_mpa,
            case.top_fraction_mean_mpa,
            match case.safety_factor {
                Some(factor) => format!("{factor:.2}"),
                None => "n/a".to_string(),
            }
        );
    }
    if report.yield_strength_mpa.is_none() {
        println!("  no yield_strength_mpa in [material], so no safety factor");
    }
}

/// Print the solver benchmark table.
pub fn print_bench_report(report: &BenchReport) {
    println!("project        {}", report.name);
    println!(
        "problem        {} cells, {} degrees of freedom, load case \"{}\"",
        report.n_cells, report.n_dof, report.load_case
    );
    println!(
        "solves         {} cold solves per backend to a relative residual of {:.1e}",
        report.solves, report.tolerance
    );
    println!(
        "  {:<8} {:>12} {:>12} {:>10} {:>10}  device",
        "backend", "best ms", "mean ms", "cg iters", "speedup"
    );
    for timing in &report.timings {
        println!(
            "  {:<8} {:>12.1} {:>12.1} {:>10} {:>10}  {}",
            timing.backend.label(),
            timing.best_s * constants::MS_PER_S,
            timing.mean_s * constants::MS_PER_S,
            format!(
                "{}{}",
                timing.iterations,
                if timing.capped { "+" } else { "" }
            ),
            match report.speedup(timing) {
                Some(factor) => format!("{factor:.2}x"),
                None => "-".to_string(),
            },
            timing.description
        );
    }
    if report.timings.iter().any(|t| t.capped) {
        println!(
            "  a \"+\" on the iteration count means the solve stopped on the cap of {} rather \
             than the tolerance; its time is not comparable",
            constants::BENCH_CG_MAX_ITERATIONS
        );
    }
    for skipped in &report.skipped {
        println!(
            "  {:<8} not timed: {}",
            skipped.backend.label(),
            skipped.reason
        );
    }
}

/// Self weight of a density field in newtons and the mass behind it in grams.
///
/// The acceleration is never the zero vector: a gravity load with a zero
/// direction or a non-positive `g_mm_s2` is rejected while the config is
/// parsed, and a load case whose gravity loads cancel each other out is
/// rejected while the problem is built. The division below is what depends on
/// that, so it says so.
fn self_weight(problem: &Problem, densities: &[f64], acceleration: [f64; 3]) -> (f64, f64) {
    debug_assert!(
        crate::geometry::length(acceleration) > 0.0,
        "a load case reached the report with a degenerate gravity acceleration"
    );
    let newtons = crate::fea::gravity::total_weight_n(
        &problem.grid,
        densities,
        acceleration,
        problem.material.density_g_cm3,
    );
    let grams = newtons / crate::geometry::length(acceleration) * constants::GRAMS_PER_TONNE;
    (newtons, grams)
}

/// Print the self weight the solver carried, once per load case that has one.
///
/// Nothing is printed for a run without gravity. The densities are the exported
/// ones, so the figure belongs to the same structure as the mass estimate next
/// to it; the two are independent readings of it, one from the voxel field and
/// one from the enclosed volume of the surface.
pub fn print_self_weight(problem: &Problem, densities: &[f64]) {
    for case in &problem.load_cases {
        let Some(acceleration) = case.gravity else {
            continue;
        };
        let (newtons, grams) = self_weight(problem, densities, acceleration);
        println!(
            "self weight    {newtons:.4} N ({grams:.2} g) in load case \"{}\"",
            case.name
        );
    }
}

/// Print the statistics of the exported mesh.
///
/// `supersample` is the `[output]` factor the surface was extracted with; it is
/// named on the triangle line only when it did something, so an unrefined
/// export reads exactly as it always has.
pub fn print_mesh_stats(stats: &MeshStats, density_g_cm3: f64, supersample: usize) {
    let refinement = if supersample > 1 {
        format!(" (supersample {supersample}x)")
    } else {
        String::new()
    };
    println!(
        "mesh           {} triangles, {} vertices{refinement}",
        stats.triangles, stats.vertices
    );
    println!(
        "mesh bounds    [{:.3}, {:.3}, {:.3}] .. [{:.3}, {:.3}, {:.3}] mm",
        stats.bounds.min[0],
        stats.bounds.min[1],
        stats.bounds.min[2],
        stats.bounds.max[0],
        stats.bounds.max[1],
        stats.bounds.max[2]
    );
    println!("enclosed vol   {:.1} mm3", stats.volume_mm3);
    println!(
        "estimated mass {:.2} g at {} g/cm3",
        stats.volume_mm3 * density_g_cm3 / constants::MM3_PER_CM3,
        density_g_cm3
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A reporter that implements only the two required methods and therefore
    /// inherits the default density hook.
    #[derive(Default)]
    struct MinimalReporter {
        calls: AtomicUsize,
    }

    impl Reporter for MinimalReporter {
        fn iteration(&self, _stats: &IterationStats) {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
        fn note(&self, _message: &str) {}
    }

    fn sample_stats() -> IterationStats {
        IterationStats {
            iteration: 1,
            compliance: 1.0,
            volume_fraction: 0.5,
            worst_local_fraction: None,
            max_change: 0.1,
            cg_iterations: vec![7],
            elapsed_s: 0.0,
            growth: None,
            reduce: None,
        }
    }

    #[test]
    fn a_growth_line_never_reaches_the_simp_format() {
        // Both reporters have to accept a growth line; the console one picks a
        // different format for it, which is the whole point of the extension.
        let stats = IterationStats {
            growth: Some(GrowthProgress {
                phase: GrowthPhase::Branching,
                segments: 12,
                attractors_remaining: 34,
            }),
            ..sample_stats()
        };
        ConsoleReporter::new(false).iteration(&stats);
        ConsoleReporter::new(true).iteration(&stats);
        SilentReporter.iteration(&stats);
        assert_eq!(GrowthPhase::Backbone.label(), "backbone");
        assert_eq!(GrowthPhase::Branching.label(), "branching");
        assert_eq!(GrowthPhase::Thickening.label(), "thickening");
    }

    fn sample_stage() -> ReduceStage {
        ReduceStage {
            index: 3,
            target_fraction: 0.512,
            achieved_fraction: 0.5121,
            iterations: 84,
            safety_factor: Some(6.41),
            passed: true,
            refine: false,
        }
    }

    #[test]
    fn a_run_without_a_schedule_prints_the_line_it_always_did() {
        // The pin on the format: a line with neither a local cap nor a stage
        // keeps every column, and every space between them, byte for byte.
        let plain = iteration_line(&sample_stats());
        assert_eq!(
            plain,
            "iter    1  compliance   1.000000e0  vol 0.5000  change  0.10000  cg [7]     0.00 s"
        );
        let staged = IterationStats {
            reduce: Some(ReduceProgress {
                stage: 3,
                target_fraction: 0.512,
                completed: 2,
            }),
            ..sample_stats()
        };
        // The schedule adds a prefix and changes nothing behind it.
        assert_eq!(iteration_line(&staged), format!("stage  3  {plain}"));
        ConsoleReporter::new(false).iteration(&staged);
        ConsoleReporter::new(true).iteration(&staged);
        SilentReporter.iteration(&staged);
    }

    #[test]
    fn a_stage_reports_its_target_its_design_and_its_verdict() {
        assert_eq!(
            reduce_stage_note(&sample_stage()),
            "reduce stage 3: target 0.5120, fraction 0.5121, 84 iterations, safety factor 6.41, \
             pass"
        );
        // A stage the load path gate failed never reached a stress solve, so it
        // has no factor to quote and says so rather than quoting a zero.
        let broken = ReduceStage {
            index: 4,
            safety_factor: None,
            passed: false,
            refine: true,
            ..sample_stage()
        };
        assert_eq!(
            reduce_stage_note(&broken),
            "reduce stage 4 (refinement): target 0.5120, fraction 0.5121, 84 iterations, safety \
             factor n/a, fail"
        );
    }

    #[test]
    fn the_schedule_summary_names_the_exported_stage_and_warns_when_none_held() {
        let held = ReduceSummary {
            method: crate::config::ReduceMethod::Continuation,
            target_safety_factor: 5.0,
            exported: sample_stage(),
            stages: vec![sample_stage()],
            finished_safety_factor: Some(6.41),
        };
        assert_eq!(
            reduce_summary_note(&held),
            "reduce: exported stage 3 at fraction 0.5121, safety factor 6.41 >= 5.00"
        );
        let start = ReduceStage {
            index: 1,
            target_fraction: 1.0,
            achieved_fraction: 1.0,
            safety_factor: Some(3.2),
            passed: false,
            ..sample_stage()
        };
        let missed = ReduceSummary {
            exported: start,
            stages: vec![start],
            ..held
        };
        let note = reduce_summary_note(&missed);
        assert!(
            note.starts_with("warning: [optimization.reduce]: "),
            "{note}"
        );
        assert!(
            note.contains(
                "no stage held the target safety factor of 5.00; exporting stage 1 at fraction \
                 1.0000, safety factor 3.20"
            ),
            "{note}"
        );
        // The exported design is the solid start, so there is no material left
        // to give it and the remedy says what there is instead.
        assert!(
            note.ends_with(
                "the design is already solid, so lower target_safety_factor, enlarge the domain, \
                 relieve the load, or choose a stiffer or stronger material"
            ),
            "{note}"
        );
        // A design with voids in it is told the other thing.
        let porous = ReduceStage {
            achieved_fraction: 0.4,
            ..start
        };
        let note = reduce_summary_note(&ReduceSummary {
            exported: porous,
            stages: vec![porous],
            ..held
        });
        assert!(
            note.ends_with(
                "lower target_safety_factor, or give the design more material or a stiffer \
                 material to hold it with"
            ),
            "{note}"
        );
    }

    /// The part in the file is measured after the finishing passes, and a target
    /// they cost it is the run's own warning - naming both numbers and the
    /// passes that stood between the stage and the file.
    #[test]
    fn the_finished_part_warns_when_the_passes_cost_it_the_target() {
        let held = ReduceSummary {
            method: crate::config::ReduceMethod::Continuation,
            target_safety_factor: 5.0,
            exported: sample_stage(),
            stages: vec![sample_stage()],
            finished_safety_factor: Some(6.0),
        };
        // The exported part holds the target, so there is nothing here the
        // schedule's own summary has not already said.
        assert_eq!(reduce_finished_note(&held, &["trim"]), None);

        let lost = ReduceSummary {
            finished_safety_factor: Some(4.2),
            ..held.clone()
        };
        let note = reduce_finished_note(&lost, &["trim", "reinforce"]).expect("a warning");
        assert!(
            note.starts_with("warning: [optimization.reduce]: "),
            "{note}"
        );
        // Both numbers: what the file carries, and what the schedule chose on.
        assert!(
            note.contains("safety factor of 4.20, below the target of 5.00"),
            "{note}"
        );
        assert!(note.contains("stage 3, which measured 6.41"), "{note}");
        assert!(
            note.contains("[output] trim and reinforce passes ran"),
            "{note}"
        );
        // One pass reads as one, and a run with none of them says what did stand
        // between the stage and the file instead.
        let single = reduce_finished_note(&lost, &["trim"]).expect("a warning");
        assert!(single.contains("[output] trim pass ran"), "{single}");
        let bare = reduce_finished_note(&lost, &[]).expect("a warning");
        assert!(
            bare.contains("nothing but the export's cavity pass ran"),
            "{bare}"
        );
        // An unmeasured part holds nothing, and says so the way the table does.
        let unmeasured = ReduceSummary {
            finished_safety_factor: None,
            ..held
        };
        let note = reduce_finished_note(&unmeasured, &["trim"]).expect("a warning");
        assert!(note.contains("safety factor of n/a"), "{note}");
    }

    #[test]
    fn the_density_hook_defaults_to_a_no_op() {
        let reporter = MinimalReporter::default();
        let densities = vec![0.25; 8];
        reporter.densities(&sample_stats(), &densities);
        reporter.densities(&sample_stats(), &densities);
        // Nothing may observe the snapshot: the reporter's own state is
        // untouched and the borrowed densities come back unchanged.
        assert_eq!(reporter.calls.load(Ordering::Relaxed), 0);
        assert!(densities.iter().all(|d| *d == 0.25));
    }

    #[test]
    fn the_shipped_reporters_ignore_densities() {
        ConsoleReporter::new(true).densities(&sample_stats(), &[1.0]);
        SilentReporter.densities(&sample_stats(), &[1.0]);
    }

    /// The solver line says what the run's solves will really do: the backend
    /// it will end up on, and the tolerance they will stop at when that is not
    /// the one every other run of this crate stops at.
    ///
    /// Both halves are pinned byte for byte on purpose: this line is what a
    /// user reads a run's terms off, so every word of it is a decision rather
    /// than an accident - including the conditions the reproducibility promise
    /// is qualified by, which the README and `SolverBackend`'s own docs carry in
    /// the same words. What may *not* appear is a tolerance equal to the
    /// constant, written out in the file or left out of it: that one is not a
    /// fact about the run, it is what the run would have done anyway.
    #[test]
    fn the_solver_line_names_a_tolerance_only_when_it_is_not_the_default() {
        let default = SolverParams::default();
        assert_eq!(
            solver_line(&default),
            "cpu backend, bit-for-bit reproducible for this build and thread count"
        );

        let loosened = SolverParams {
            tolerance: 3e-8,
            ..default
        };
        assert_eq!(
            solver_line(&loosened),
            "cpu backend, bit-for-bit reproducible for this build and thread count, tolerance 3e-8",
            "a loosened target has to appear somewhere a user can read it"
        );

        // Beside the other qualification the line carries, which is a statement
        // about the backend and stays attached to it.
        let fell_back = SolverParams {
            backend: SolverBackend::Cpu,
            explicit: false,
            fell_back: Some(crate::config::SolverFallback::NoFeature),
            tolerance: 1e-10,
        };
        let line = solver_line(&fell_back);
        assert!(line.ends_with(", tolerance 1e-10"), "{line}");
        assert!(
            line.contains(&format!(
                "({})",
                crate::config::SolverFallback::NoFeature.note()
            )),
            "{line}"
        );

        // And the gpu backend's own wording is untouched by either.
        let gpu = SolverParams {
            backend: SolverBackend::Gpu,
            explicit: true,
            fell_back: None,
            tolerance: constants::CG_RELATIVE_TOLERANCE,
        };
        assert_eq!(
            solver_line(&gpu),
            "gpu backend, single precision with f64 refinement; reproducible on this machine only"
        );
    }

    /// A component of `volume_mm3` anchored by `anchors`, the shape of it being
    /// nothing these lines depend on.
    fn component(volume_mm3: f64, anchors: AnchorSet) -> crate::mesh::islands::SurfaceComponent {
        crate::mesh::islands::SurfaceComponent {
            triangles: 8,
            volume_mm3,
            centroid: [1.0, 2.0, 3.0],
            anchors,
        }
    }

    /// The set a body anchored by a support carries.
    fn held() -> AnchorSet {
        let mut set = AnchorSet::none();
        set.insert(crate::mesh::islands::AnchorKind::Support);
        set
    }

    fn island_report(
        policy: IslandPolicy,
        bodies: usize,
        cavity_shells: usize,
        fragments: Vec<f64>,
    ) -> IslandReport {
        let fragments: Vec<_> = fragments
            .into_iter()
            .map(|volume| component(volume, AnchorSet::none()))
            .collect();
        let kept = match policy {
            IslandPolicy::Cull => 0,
            IslandPolicy::Keep => fragments.len(),
        };
        IslandReport {
            policy,
            components: bodies + cavity_shells + fragments.len(),
            bodies: (0..bodies)
                .map(|index| component(100.0 - index as f64, held()))
                .collect(),
            cavity_shells,
            culled_fragments: fragments.len() - kept,
            culled_triangles: 8 * (fragments.len() - kept),
            culled_volume_mm3: fragments
                .iter()
                .take(fragments.len() - kept)
                .map(|f| f.volume_mm3)
                .sum(),
            fragments,
            unserved: Vec::new(),
            culled_inside: Vec::new(),
        }
    }

    #[test]
    fn the_mesh_body_line_says_what_shipped_and_what_was_taken_out_of_it() {
        // A part on its own says one thing and nothing else: the line a clean
        // run has always been entitled to.
        assert_eq!(
            island_headline(&island_report(IslandPolicy::Cull, 1, 0, vec![])),
            "mesh bodies    1 in the exported surface"
        );
        // A cavity is counted next to the body that encloses it, never as a
        // body of its own, which is what keeps this line and the void report
        // from contradicting each other.
        assert_eq!(
            island_headline(&island_report(IslandPolicy::Cull, 1, 1, vec![])),
            "mesh bodies    1 in the exported surface (+1 cavity shell)"
        );
        assert_eq!(
            island_headline(&island_report(IslandPolicy::Cull, 1, 3, vec![])),
            "mesh bodies    1 in the exported surface (+3 cavity shells)"
        );
        // What was culled is named with its volume, largest first.
        assert_eq!(
            island_headline(&island_report(IslandPolicy::Cull, 1, 2, vec![0.5, 0.25])),
            "mesh bodies    1 in the exported surface (+2 cavity shells); culled 2 floating \
             fragments (0.750 mm3 in total, largest 0.500 mm3)"
        );
        assert_eq!(
            island_headline(&island_report(IslandPolicy::Cull, 1, 0, vec![0.5])),
            "mesh bodies    1 in the exported surface; culled 1 floating fragment (0.500 mm3 in \
             total, largest 0.500 mm3)"
        );
        // Under "keep" the same fragments are still reported, and the count of
        // bodies is what the file really holds.
        assert_eq!(
            island_headline(&island_report(IslandPolicy::Keep, 3, 0, vec![0.5, 0.25])),
            "mesh bodies    3 in the exported surface; kept 2 floating fragments (0.750 mm3 in \
             total, largest 0.500 mm3)"
        );
        // Several bodies is a normal answer, not a defect: a keepin boss and
        // the structure are two of them.
        assert_eq!(
            island_headline(&island_report(IslandPolicy::Cull, 2, 0, vec![])),
            "mesh bodies    2 in the exported surface"
        );
    }

    #[test]
    fn the_fragment_note_stops_claiming_nothing_asked_for_what_left() {
        let plain = island_report(IslandPolicy::Cull, 1, 0, vec![0.5]);
        assert!(
            fragment_note(&plain).contains("nothing declared asked for it"),
            "{}",
            fragment_note(&plain)
        );

        // The same report, with one of the culled fragments found to have been
        // inside a declared region after all.
        let mut inside = plain.clone();
        inside.culled_inside = vec![(0, "load 1 of case \"tip\"".to_string())];
        let note = fragment_note(&inside);
        assert!(!note.contains("nothing declared asked for it"), "{note}");
        assert!(note.contains("is not debris"), "{note}");

        // Nothing was culled, and nothing is claimed either way.
        let none = island_report(IslandPolicy::Cull, 1, 0, vec![]);
        assert!(fragment_note(&none).contains("nothing was removed"));
        let kept = island_report(IslandPolicy::Keep, 2, 0, vec![0.5]);
        assert!(fragment_note(&kept).contains("in the file"));
    }

    /// The adjudicated case: a stub that a load region touches is kept - size
    /// culls nothing - and is named, because a loose piece in the file may never
    /// be a surprise.
    #[test]
    fn a_tiny_body_beside_another_is_warned_about_and_a_normal_one_is_not() {
        // The sphere of a 4 mm feature is 33.510 mm3.
        let feature = 4.0;
        assert!((tiny_body_mm3(feature) - 33.510_321_638_291_12).abs() < 1e-9);

        // One body, however small, is the part rather than a loose piece.
        let mut alone = island_report(IslandPolicy::Cull, 1, 0, vec![]);
        alone.bodies[0].volume_mm3 = 1.0;
        assert!(tiny_bodies(&alone, feature).is_empty());

        // Two, one of them a stub: the stub is named and the part is not.
        let mut stubby = island_report(IslandPolicy::Cull, 2, 0, vec![]);
        stubby.bodies[0].volume_mm3 = 1200.0;
        stubby.bodies[1].volume_mm3 = 16.0;
        assert_eq!(tiny_bodies(&stubby, feature), vec![1]);
        // And nothing is removed by it: the report still says two bodies.
        assert_eq!(stubby.culled_fragments, 0);
        assert_eq!(
            island_headline(&stubby),
            "mesh bodies    2 in the exported surface"
        );

        // A finer feature size makes the same body a feature again.
        assert!(tiny_bodies(&stubby, 2.0).is_empty());
        // Exactly on the threshold is not below it.
        stubby.bodies[1].volume_mm3 = tiny_body_mm3(feature);
        assert!(tiny_bodies(&stubby, feature).is_empty());
    }

    #[test]
    fn a_body_says_what_anchors_it_and_debris_says_it_has_nothing() {
        let mut both = AnchorSet::none();
        both.insert(crate::mesh::islands::AnchorKind::Support);
        both.insert(crate::mesh::islands::AnchorKind::Load);
        assert_eq!(anchor_labels(both), "support, load");
        assert_eq!(anchor_labels(held()), "support");
        assert_eq!(anchor_labels(AnchorSet::none()), "nothing declared");
    }

    #[test]
    fn the_island_report_prints_every_policy_and_caps_its_fragment_list() {
        let feature = 4.0;
        for policy in [IslandPolicy::Cull, IslandPolicy::Keep] {
            print_island_report(&island_report(policy, 1, 0, vec![]), feature);
            print_island_report(&island_report(policy, 1, 1, vec![0.5]), feature);
            // Several bodies, which lists what holds each of them.
            print_island_report(&island_report(policy, 3, 1, vec![0.5]), feature);
            // More fragments than the report lists individually.
            let many: Vec<f64> = (0..constants::ISLAND_REPORT_MAX_LINES + 3)
                .map(|i| 1.0 / (i + 1) as f64)
                .collect();
            print_island_report(&island_report(policy, 1, 0, many), feature);
            // A region nothing that ships reaches, which is the loudest line
            // this report has, and a body that ships loose and is too small to
            // be a feature, which is the next loudest.
            let mut stranded = island_report(policy, 1, 0, vec![0.5]);
            stranded.unserved = vec!["load 1 of case \"tip\"".to_string()];
            print_island_report(&stranded, feature);
            let mut stubby = island_report(policy, 2, 0, vec![]);
            stubby.bodies[1].volume_mm3 = 0.5;
            print_island_report(&stubby, feature);
            // And a fragment that was inside a declared region after all.
            let mut inside = island_report(policy, 1, 0, vec![0.5]);
            inside.culled_inside = vec![(0, "load 1 of case \"tip\"".to_string())];
            print_island_report(&inside, feature);
        }
        // And the field level line, which is the one it must not be read as.
        print_solid_report(&[]);
        print_solid_report(&[SolidBody {
            cells: 4,
            volume_mm3: 4.0,
            centroid: [0.0; 3],
        }]);
    }

    /// The boundary clamp's line, in each of the three things it can say.
    ///
    /// `None` is `boundaries = "voxel"` and prints nothing at all; the other two
    /// always print, because the pass changed the geometry that shipped - or
    /// deliberately did not, at a place the user has to know about.
    #[test]
    fn the_clamp_report_prints_nothing_when_the_pass_never_ran() {
        print_clamp_report(None, false, false);

        let quiet = ClampReport {
            vertices_moved: 0,
            max_displacement_mm: 0.0,
            gave_up: 0,
            adrift: 0,
            max_adrift_mm: 0.0,
        };
        assert_eq!(quiet.notes(false, false).len(), 1);
        print_clamp_report(Some(&quiet), false, false);

        let moved = ClampReport {
            vertices_moved: 412,
            max_displacement_mm: 0.4213,
            ..quiet
        };
        let notes = moved.notes(false, false);
        assert!(notes[0].contains("412"), "{notes:?}");
        assert!(notes[0].contains("0.4213"), "{notes:?}");
        print_clamp_report(Some(&moved), false, false);

        // The one line that has to be loud, in the wording every report in
        // growforge marks a warning with.
        let refused = ClampReport {
            vertices_moved: 3,
            max_displacement_mm: 0.1,
            gave_up: 7,
            ..quiet
        };
        let notes = refused.notes(false, false);
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(
            notes[1].starts_with("warning: 7 vertices were"),
            "{notes:?}"
        );
        print_clamp_report(Some(&refused), false, false);

        // One of them reads as one of them, rather than as "1 vertices".
        let single = ClampReport {
            gave_up: 1,
            ..refused
        };
        assert!(
            single.notes(false, false)[1].starts_with("warning: 1 vertex was"),
            "{:?}",
            single.notes(false, false)
        );
        print_clamp_report(Some(&single), false, false);
    }

    /// The vertices the pass left near a boundary without seating: always
    /// counted, and said - or not said - according to what the run was.
    ///
    /// A solid part *is* the shapes it was drawn from, so a vertex resting off
    /// one is a face about to ship in the wrong place and the line opens with the
    /// word the panel colours, flush or no flush. An optimized or grown part has
    /// free surfaces everywhere, one of which may legitimately pass near a
    /// boundary: with `[output] flush` asked for, the same count is the shortfall
    /// of that pass and is stated plainly with the key that reaches further;
    /// without it, **nothing is said at all** - about a third of a stock
    /// optimized part's vertices are within the window of its own domain box,
    /// and a line on every run is a line nobody reads on the run that is wrong.
    ///
    /// Two counter-checks the whole thing rests on, asserted rather than assumed:
    /// the unflushed design is silent, and a report with nothing adrift says
    /// nothing new in any of the four runs it can belong to.
    #[test]
    fn an_adrift_vertex_is_a_warning_when_drawn_a_note_when_flushed_and_silence_otherwise() {
        let clean = ClampReport {
            vertices_moved: 412,
            max_displacement_mm: 0.0874,
            gave_up: 0,
            adrift: 0,
            max_adrift_mm: 0.0,
        };
        for (solid, flushing) in [(false, false), (false, true), (true, false), (true, true)] {
            let notes = clean.notes(solid, flushing);
            assert_eq!(notes.len(), 1, "{notes:?}");
            assert!(
                !notes.iter().any(|note| note.contains("off the surface")),
                "a clean clamp said something about adrift vertices: {notes:?}"
            );
            print_clamp_report(Some(&clean), solid, flushing);
        }

        let adrift = ClampReport {
            adrift: 37,
            max_adrift_mm: 0.4400,
            ..clean
        };

        // Drawn: a warning either way, because the surface belongs to a shape
        // whether or not a fill pass was asked to reach it.
        for flushing in [false, true] {
            let loud = adrift.notes(true, flushing);
            assert_eq!(loud.len(), 2, "{loud:?}");
            assert!(
                loud[1].starts_with("warning: 37 vertices rest up to 0.4400 mm off the surface"),
                "{loud:?}"
            );
            print_clamp_report(Some(&adrift), true, flushing);
        }

        // Designed and flushed: the pass that was asked for fell short, said
        // plainly and naming what reaches further.
        let flushed = adrift.notes(false, true);
        assert_eq!(flushed.len(), 2, "{flushed:?}");
        assert!(!flushed[1].starts_with("warning"), "{flushed:?}");
        assert!(
            flushed[1]
                .starts_with("[output] flush ran and 37 vertices rest up to 0.4400 mm off the"),
            "{flushed:?}"
        );
        assert!(
            flushed[1].contains("flush_depth_mm"),
            "the remedy is not named: {flushed:?}"
        );
        print_clamp_report(Some(&adrift), false, true);

        // Designed and not flushed: nothing at all, which is the whole point of
        // the rule. Asserted as an exact absence rather than as a length alone.
        let designed = adrift.notes(false, false);
        assert_eq!(designed.len(), 1, "{designed:?}");
        assert!(
            !designed
                .iter()
                .any(|note| note.contains("off the surface") || note.contains("flush")),
            "a design that asked for no flush was told about its free surfaces: {designed:?}"
        );
        print_clamp_report(Some(&adrift), false, false);

        // One of them reads as one of them here too.
        let single = ClampReport {
            adrift: 1,
            ..adrift
        };
        assert!(
            single.notes(true, false)[1].starts_with("warning: 1 vertex rests"),
            "{:?}",
            single.notes(true, false)
        );
        assert!(
            single.notes(false, true)[1].contains("1 vertex rests"),
            "{:?}",
            single.notes(false, true)
        );
    }

    /// A stress table of one load case with a safety factor, for the block that
    /// prints it.
    fn stress_report() -> crate::stress::StressReport {
        crate::stress::StressReport {
            cases: vec![crate::stress::CaseStress {
                name: "tip".to_string(),
                max_mpa: 9.27,
                percentile_mpa: 7.4,
                top_fraction_mean_mpa: 8.1,
                safety_factor: Some(47.0 / 9.27),
                von_mises: Vec::new(),
            }],
            yield_strength_mpa: Some(47.0),
            density_threshold: constants::STRESS_DENSITY_THRESHOLD,
            evaluated_cells: 512,
        }
    }

    /// The console's stress block says the export is in pieces, in this
    /// report's own layout and in the wording the panel and the editor's console
    /// use.
    ///
    /// A safety factor of a surface that came out in several bodies is a factor
    /// of each of them against its own supports, and a run that printed it
    /// without saying so is the incident this line exists for.
    #[test]
    fn the_stress_block_says_when_the_export_came_out_in_pieces() {
        // One body is the case the table was always written for: no line.
        assert_eq!(stress_body_warning(0), None);
        assert_eq!(stress_body_warning(1), None);

        let warning = stress_body_warning(2).expect("an export in pieces must say so");
        assert_eq!(
            warning,
            "  warning      the export is 2 separate bodies - this safety factor describes each \
             piece against its own supports, not one connected part"
        );
        // The sentence is the summary's own, laid out in this report's columns.
        let note = crate::stress::disconnected_bodies_note(2).expect("the shared sentence");
        assert!(warning.ends_with(&note), "{warning}");
        assert!(
            stress_body_warning(5)
                .expect("five is in pieces too")
                .contains("is 5 separate bodies"),
        );

        // And the block prints, in each of the shapes it has: one body, several,
        // no yield strength to divide by, and no report at all.
        let report = stress_report();
        print_stress_report(&StressOutcome::Available(report.clone()), 1);
        print_stress_report(&StressOutcome::Available(report.clone()), 2);
        let mut without = report;
        without.yield_strength_mpa = None;
        without.cases[0].safety_factor = None;
        print_stress_report(&StressOutcome::Available(without), 3);
        print_stress_report(&StressOutcome::Unavailable("it did not converge".into()), 2);
    }

    const HEAVY: &str = r#"
[project]
name = "heavy"

[resolution]
voxel_size_mm = 5.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.5
min_feature_mm = 15.0

[output]
stl_path = "heavy.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [20.0, 10.0, 10.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 10.5, 10.5] }

[[loadcases]]
name = "hanging"
[[loadcases.loads]]
type = "gravity"
"#;

    #[test]
    fn the_reported_self_weight_and_mass_agree_with_the_density_field() {
        use crate::config::Config;
        use std::path::PathBuf;

        let config = Config::parse(HEAVY).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let acceleration = problem.load_cases[0].gravity.expect("self weight");

        // Half the design cells solid, so the answer is not the trivial one.
        let mut densities = vec![0.0; problem.grid.n_cells()];
        let solid = problem.grid.n_cells() / 2;
        for slot in densities.iter_mut().take(solid) {
            *slot = 1.0;
        }
        let (newtons, grams) = self_weight(&problem, &densities, acceleration);

        // The grams must be the mass the same field carries, worked out the
        // long way round from volume and density rather than from the force.
        let volume_mm3 = solid as f64 * problem.cell_volume_mm3();
        let expected_g = volume_mm3 * problem.material.density_g_cm3 / constants::MM3_PER_CM3;
        assert!(
            (grams - expected_g).abs() < 1e-9 * expected_g,
            "{grams} g is not the {expected_g} g the field holds"
        );
        // And the newtons must be that mass times the acceleration.
        let expected_n =
            expected_g / constants::GRAMS_PER_TONNE * crate::geometry::length(acceleration);
        assert!(
            (newtons - expected_n).abs() < 1e-9 * expected_n,
            "{newtons} N is not {expected_n} N"
        );
        assert!(newtons > 0.0, "a self weight is a magnitude");

        // A run without gravity prints nothing at all.
        print_self_weight(&problem, &densities);
        let plain = Config::parse(&HEAVY.replace("type = \"gravity\"", "type = \"force\"\nregion = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 4.0 }\nvector = [0.0, 0.0, -10.0]")).expect("parse");
        let plain = Problem::build(&plain, &PathBuf::from(".")).expect("build");
        assert!(plain.load_cases[0].gravity.is_none());
        print_self_weight(&plain, &densities);
    }
}
