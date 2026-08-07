//! Console reporting: per-iteration progress, problem summaries and mesh
//! statistics.
//!
//! Engines talk to a [`Reporter`] rather than to stdout so that later phases
//! (a viewer, a batch runner) can consume the same stream.

use crate::bench::BenchReport;
use crate::config::{IslandPolicy, SolverBackend, SolverParams, UpdateScheme, VoidPolicy};
use crate::constants;
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
        let cg: Vec<String> = stats.cg_iterations.iter().map(|c| c.to_string()).collect();
        // The local cap's own column exists only while the cap does, so a run
        // without one prints exactly the line it always did.
        let local = match stats.worst_local_fraction {
            Some(worst) => format!("  local {worst:>6.4}"),
            None => String::new(),
        };
        println!(
            "iter {:>4}  compliance {:>12.6e}  vol {:>6.4}{local}  change {:>8.5}  cg [{}]  \
             {:>7.2} s",
            stats.iteration,
            stats.compliance,
            stats.volume_fraction,
            stats.max_change,
            cg.join(", "),
            stats.elapsed_s
        );
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
                    match problem.optimization.update {
                        UpdateScheme::Oc => "oc, optimality criteria (the reproducible default)",
                        UpdateScheme::Mma => "mma, method of moving asymptotes",
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
pub fn print_clamp_report(report: Option<&ClampReport>) {
    let Some(report) = report else {
        return;
    };
    for (index, note) in report.notes().iter().enumerate() {
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
        print_clamp_report(None);

        let quiet = ClampReport {
            vertices_moved: 0,
            max_displacement_mm: 0.0,
            gave_up: 0,
        };
        assert_eq!(quiet.notes().len(), 1);
        print_clamp_report(Some(&quiet));

        let moved = ClampReport {
            vertices_moved: 412,
            max_displacement_mm: 0.4213,
            gave_up: 0,
        };
        assert!(moved.notes()[0].contains("412"), "{:?}", moved.notes());
        assert!(moved.notes()[0].contains("0.4213"), "{:?}", moved.notes());
        print_clamp_report(Some(&moved));

        // The one line that has to be loud, in the wording every report in
        // growforge marks a warning with.
        let refused = ClampReport {
            vertices_moved: 3,
            max_displacement_mm: 0.1,
            gave_up: 7,
        };
        let notes = refused.notes();
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(
            notes[1].starts_with("warning: 7 vertices were"),
            "{notes:?}"
        );
        print_clamp_report(Some(&refused));

        // One of them reads as one of them, rather than as "1 vertices".
        let single = ClampReport {
            gave_up: 1,
            ..refused
        };
        assert!(
            single.notes()[1].starts_with("warning: 1 vertex was"),
            "{:?}",
            single.notes()
        );
        print_clamp_report(Some(&single));
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
