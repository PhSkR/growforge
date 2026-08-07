//! growforge: topology optimization and growth heuristics for 3D printable
//! parts.
//!
//! The pipeline is
//! `config -> voxel grid -> finite element model -> engine -> density field ->
//! cavities -> stress report -> [trim -> flush -> reinforce -> both again] ->
//! isosurface -> smoothing -> island culling -> boundary clamp -> validation ->
//! binary STL`. Everything
//! after the engine is engine agnostic, which is what lets the `growth` engine
//! hand over a field it never solved for - and the `solid` engine one it never
//! optimized - and still get an honest stress report on it.
//!
//! Units are millimetres, newtons, megapascals (N/mm^2), newton-millimetres for
//! torque and g/cm^3 for mass density.

#![deny(missing_docs)]

pub mod bench;
pub mod config;
pub mod constants;
pub mod engine;
pub mod fea;
pub mod flush;
pub mod geometry;
pub mod grid;
pub mod json;
pub mod mesh;
pub mod problem;
pub mod reinforce;
pub mod report;
pub mod stress;
pub mod trim;
#[cfg(feature = "viewer")]
pub mod viewer;
pub mod voids;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::config::{BoundaryFidelity, Config};
use crate::engine::DensityField;
use crate::fea::CancelProbe;
use crate::flush::FlushReport;
use crate::mesh::clamp::ClampReport;
use crate::mesh::islands::IslandReport;
use crate::mesh::validate::MeshStats;
use crate::mesh::{Mesh, SurfaceExport};
use crate::problem::Problem;
use crate::reinforce::ReinforceReport;
use crate::report::Reporter;
use crate::stress::{StressLimits, StressOutcome};
use crate::trim::TrimReport;
use crate::voids::{SolidBody, VoidReport};

/// Everything the post-run analysis of a density field found.
#[derive(Debug, Clone)]
pub struct Analysis {
    /// What the enclosed cavity pass found and did.
    pub voids: VoidReport,
    /// The connected bodies of material in the exported field, largest first.
    pub solids: Vec<SolidBody>,
    /// Von Mises stresses of the exported structure, or why they are missing.
    pub stress: StressOutcome,
}

/// Everything a completed run produced.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    /// Converged density field, with the enclosed cavity policy already applied
    /// to it, so it is exactly the field the surface came from.
    pub field: DensityField,
    /// What the enclosed cavity pass found and did.
    pub voids: VoidReport,
    /// The connected bodies of material in the exported field, largest first.
    /// One body is a part; anything past the first is a floating island.
    pub solids: Vec<SolidBody>,
    /// Von Mises stresses of the exported structure, or why they are missing.
    /// A run whose stress solve failed still writes its STL and still exits
    /// zero; see [`analyse`].
    pub stress: StressOutcome,
    /// What the `[output] trim` pass removed, or why it removed nothing;
    /// `None` under the default `trim = "off"`, where the pass never ran.
    pub trim: Option<TrimReport>,
    /// What the `[output] flush` pass filled out to the surfaces the part's
    /// walls rest on; `None` under the default `flush = "off"`, where the pass
    /// never ran.
    pub flush: Option<FlushReport>,
    /// What the `[output] reinforce` pass thickened, and what it could not;
    /// `None` under the default `reinforce = "off"`, where the pass never ran.
    pub reinforce: Option<ReinforceReport>,
    /// The exported surface, after smoothing, culling and validation.
    pub mesh: Mesh,
    /// Statistics of the exported mesh.
    pub stats: MeshStats,
    /// The connected components of that mesh, and the floating fragments the
    /// `[output] islands` policy removed from it. This describes the shipped
    /// surface; [`RunOutcome::solids`] describes the density field it came
    /// from, and the two are not the same reading.
    pub islands: IslandReport,
    /// What the boundary clamp moved onto the analytic domain and keepout
    /// surfaces; `None` under `[output] boundaries = "voxel"`, where the pass
    /// never ran.
    pub clamp: Option<ClampReport>,
    /// Where the STL was written.
    pub stl_path: PathBuf,
    /// Wall clock seconds spent optimizing.
    pub optimize_s: f64,
    /// Wall clock seconds spent on the cavity pass, the stress solve, the trim,
    /// flush and reinforcement passes and - when any of them changed the field -
    /// the second run of the first two over the field that ships.
    pub analysis_s: f64,
    /// Wall clock seconds spent extracting, smoothing and writing the mesh.
    pub export_s: f64,
}

/// Read a config file and build the discrete problem it describes.
pub fn load_problem(config_path: &Path) -> Result<Problem> {
    Ok(load_config_and_problem(config_path)?.1)
}

/// Directory a configuration's relative output paths resolve against: the one
/// holding the file itself.
///
/// The editor rebuilds a problem from an edited configuration without going
/// back to disk, so this is factored out of [`load_config_and_problem`] rather
/// than duplicated there.
pub fn config_dir(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Read a config file, returning both the parsed configuration and the discrete
/// problem it describes.
///
/// The discrete problem keeps what the solver needs, plus the analytic
/// boundaries the export holds its vertices to
/// ([`crate::problem::Problem::boundaries`]); the viewer wants the rest of the
/// geometry the configuration described - the keepin and load regions, and the
/// keepouts as separate entries rather than as one union - which is why both
/// come back together.
pub fn load_config_and_problem(config_path: &Path) -> Result<(Config, Problem)> {
    let config = Config::load(config_path)?;
    let problem = Problem::build(&config, &config_dir(config_path))
        .with_context(|| format!("building the problem from {}", config_path.display()))?;
    Ok((config, problem))
}

/// Run the configured engine over `problem`.
pub fn optimize(problem: &Problem, reporter: &dyn Reporter) -> Result<DensityField> {
    let engine = engine::create(&problem.engine)?;
    engine.optimize(problem, reporter)
}

/// Extract, smooth, cull and validate the surface of `densities` and write the
/// STL.
///
/// This is the whole export stage; both the command line path and the viewer's
/// worker thread go through it so the exported file can never depend on which
/// one ran. The mesh that comes back is the one that was written, floating
/// fragments already removed under the `[output] islands` policy and every
/// vertex held to the analytic boundaries under `[output] boundaries`, which is
/// what lets the viewer show the shipped surface rather than a superset of it.
///
/// The boundaries come off the *problem*, never off a configuration read again
/// here: the editor's generate-stl button exports a retained problem, and a
/// second reading could describe geometry that problem was never built from.
pub fn export(problem: &Problem, densities: &[f64]) -> Result<SurfaceExport> {
    let boundaries = match problem.output.boundaries {
        BoundaryFidelity::Exact => Some(&problem.boundaries),
        BoundaryFidelity::Voxel => None,
    };
    let export = mesh::surface_from_densities(
        &problem.grid,
        densities,
        problem.output.iso_level,
        problem.output.smoothing_iterations,
        problem.output.supersample,
        problem.output.islands,
        &problem.anchors(densities),
        boundaries,
    )
    .context("extracting the isosurface")?;
    mesh::stl::write(&problem.output.stl_path, &export.mesh)?;
    Ok(export)
}

/// Everything between the last optimization iteration and the export.
///
/// Enclosed cavities are resolved first, because the policy may add material and
/// the stress report and the exported surface both have to describe the same
/// structure. The stress solve then runs once on that final field; it cannot
/// influence the optimization, which is already over.
///
/// Exactly one thing here is allowed to fail softly: a *solve*, by an already
/// opened and already bound solver, that does not converge in its budget or
/// breaks down. The density field is final either way and the part is what the
/// run was asked for, so the reason comes back as
/// [`StressOutcome::Unavailable`], no JSON is written, and the caller is
/// expected to say so.
///
/// Everything else is still an error, including the two that look adjacent to
/// it: a solver backend that cannot be *opened* (no adapter, a device that
/// refuses, a problem past the adapter's buffer limits) and a design that cannot
/// be *bound* to one. Neither says anything about the structure, both mean the
/// run was asked for something this machine cannot do, and
/// [`stress::analyse_with`] is where the two are told apart.
pub fn analyse(problem: &Problem, densities: &mut [f64]) -> Result<Analysis> {
    Ok(analyse_with(
        problem,
        densities,
        StressLimits::default(),
        CancelProbe::NONE,
    )?
    .expect("an unwatched analysis is never cancelled"))
}

/// [`analyse`] with an explicit budget for the stress solve, and a probe that
/// may call the stress solve off.
///
/// `Ok(None)` means the probe did: nothing is written, and the caller is
/// expected to unwind rather than export.
pub fn analyse_with(
    problem: &Problem,
    densities: &mut [f64],
    limits: StressLimits,
    cancel: CancelProbe<'_>,
) -> Result<Option<Analysis>> {
    let Some(analysis) = analysed(problem, densities, limits, cancel)? else {
        return Ok(None);
    };
    write_stress_json(problem, &analysis)?;
    Ok(Some(analysis))
}

/// [`analyse_with`] without the JSON side effect.
///
/// Split out because the trim pass makes the analysis run twice, and the file on
/// disk has to describe the field that ships rather than the one the criterion
/// was read off. The whole-pipeline helper therefore analyses through this and
/// writes once, at the end; every other caller writes as it always did.
fn analysed(
    problem: &Problem,
    densities: &mut [f64],
    limits: StressLimits,
    cancel: CancelProbe<'_>,
) -> Result<Option<Analysis>> {
    let voids = voids::resolve(
        &problem.grid,
        densities,
        problem.output.iso_level,
        problem.output.voids,
        problem.material.density_g_cm3,
    );
    // After the cavity policy, so the bodies are the ones the surface came
    // from rather than the ones before filling.
    let solids = voids::solid_bodies(&problem.grid, densities, problem.output.iso_level);
    // Setup failures propagate out of here; only the solve degrades.
    let Some(stress) = stress::analyse_with(problem, densities, limits, cancel)? else {
        return Ok(None);
    };
    Ok(Some(Analysis {
        voids,
        solids,
        stress,
    }))
}

/// Write the machine readable stress report, when the configuration asked for
/// one and the solve produced one.
///
/// A file the run leaves behind, so a caller that can be stopped calls it past
/// the last boundary a stop is honoured at and never before one.
fn write_stress_json(problem: &Problem, analysis: &Analysis) -> Result<()> {
    if let Some(report) = analysis.stress.report()
        && let Some(path) = &problem.output.stress_json
    {
        stress::write_json(path, problem, report)?;
    }
    Ok(())
}

/// What the post-run pipeline tells its caller, and asks it, between stages.
///
/// The command line has nothing to say between them and never calls a run off;
/// the viewer puts each stage on the window and may be asked to stop at any of
/// them. Both go through this, so the sequence itself - analyse, trim, flush,
/// reinforce, analyse again, export - exists once rather than once per caller.
/// Every method has a default, so [`Unwatched`] is the whole of the headless
/// implementation.
pub trait Stages {
    /// The cavity pass and the stress solve are starting. Called a second time
    /// for the re-analysis the `[output] trim`, `[output] flush` and
    /// `[output] reinforce` passes force between them, because that is the same
    /// work again - once, whichever of them changed the field.
    fn analysing(&self) {}

    /// The surface is being extracted, validated and written.
    fn exporting(&self) {}

    /// True when the caller no longer wants the result. Asked at every stage
    /// boundary, and carried into the stress solve itself; nothing that was
    /// asked to stop ever exports.
    fn stopped(&self) -> bool {
        false
    }
}

/// The stages nobody is watching: the command line's.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unwatched;

impl Stages for Unwatched {}

/// Everything between a finished density field and the file on disk.
#[derive(Debug, Clone)]
pub struct Completed {
    /// What the enclosed cavity pass found and did, on the field that ships.
    pub voids: VoidReport,
    /// The connected bodies of material in the exported field, largest first.
    pub solids: Vec<SolidBody>,
    /// Von Mises stresses of the exported structure, or why they are missing.
    pub stress: StressOutcome,
    /// What the trim pass removed, or why it removed nothing; `None` under
    /// `trim = "off"`.
    pub trim: Option<TrimReport>,
    /// What the flush pass filled out to the surfaces the walls rest on; `None`
    /// under `flush = "off"`.
    pub flush: Option<FlushReport>,
    /// What the reinforcement pass thickened, and what it could not; `None`
    /// under `reinforce = "off"`.
    pub reinforce: Option<ReinforceReport>,
    /// The exported surface, after smoothing, culling and validation.
    pub mesh: Mesh,
    /// Statistics of that mesh.
    pub stats: MeshStats,
    /// Its connected components, and the fragments the island policy removed.
    pub islands: IslandReport,
    /// What the boundary clamp moved; `None` under
    /// `[output] boundaries = "voxel"`.
    pub clamp: Option<ClampReport>,
    /// Wall clock seconds spent analysing, trimming, flushing, reinforcing and
    /// analysing again.
    pub analysis_s: f64,
    /// Wall clock seconds spent extracting, smoothing and writing the mesh.
    pub export_s: f64,
}

/// The whole post-run pipeline: analyse, trim, flush, reinforce, analyse the
/// field those three left, and export it.
///
/// The one sequence both the command line and the viewer run, so the file, the
/// tables printed beside it and the surface a window shows can never depend on
/// which of them asked. `densities` is resolved in place - by the cavity policy,
/// under `[output] trim` by the trim, under `[output] flush` by the flush and
/// under `[output] reinforce` by the reinforcement - so what comes back
/// describes the field the caller now holds.
///
/// The three field passes are what make the analysis run twice, and they share
/// that second run: it happens once if any of them changed anything, because
/// what it exists for is to describe the field that ships. The first analysis is
/// the *only* stress field the trim's criterion could be read from and is thrown
/// away with the cells it condemned; everything a run reports - the stress table,
/// the safety factor, the cavity report, the JSON, the STL - comes from the
/// second, so no deliverable ever describes material that is not in the file.
/// With every pass off, or with none of them finding anything to do, the
/// analysis runs exactly once and the sequence is what it was before any of them
/// existed.
///
/// The order is trim, then flush, then reinforce, and nothing is trimmed
/// afterwards. The trim reads the stresses of the field the engine produced, so
/// it goes first. The flush puts the walls back to the geometry they were drawn
/// with before the reinforcement measures how thin anything is, because a
/// rippled wall reads as a row of thin places and the fill spent on them is
/// wasted - see [`crate::flush`]. And reinforcement material is deliberately
/// unloaded - it is there to be printable, not to carry anything - so trimming
/// after it would remove exactly what it just added. See [`crate::reinforce`].
///
/// `Ok(None)` is [`Stages::stopped`] answering yes at one of the boundaries:
/// nothing was written, and the caller is expected to unwind rather than export.
pub fn complete(
    problem: &Problem,
    densities: &mut [f64],
    limits: StressLimits,
    stages: &dyn Stages,
) -> Result<Option<Completed>> {
    if stages.stopped() {
        return Ok(None);
    }
    let stop = || stages.stopped();

    let start = Instant::now();
    stages.analysing();
    let Some(mut analysis) = analysed(problem, densities, limits, CancelProbe::watching(&stop))?
    else {
        return Ok(None);
    };
    // The criterion is the stresses of the field the engine produced, which is
    // this analysis; where a surface is and how thin a member is are geometry
    // and need no solve at all.
    let trim = trim::resolve(problem, densities, &analysis.stress);
    let flush = flush::resolve(problem, densities);
    let reinforce = reinforce::resolve(problem, densities);
    let changed = trim.as_ref().is_some_and(TrimReport::changed_the_field)
        || flush.as_ref().is_some_and(FlushReport::changed_the_field)
        || reinforce
            .as_ref()
            .is_some_and(ReinforceReport::changed_the_field);
    if changed {
        if stages.stopped() {
            return Ok(None);
        }
        // The same stage over again, and the window says so again: what the
        // cavity pass and the stress solve describe has to be the structure the
        // passes left, because that is the one being exported. Once for the
        // three of them - there is one field, and it is analysed once.
        stages.analysing();
        let Some(reanalysed) = analysed(problem, densities, limits, CancelProbe::watching(&stop))?
        else {
            return Ok(None);
        };
        analysis = reanalysed;
    }
    // The last moment a stop can be honoured, and it has to come before the
    // JSON: that report is a file the run leaves behind exactly as the STL is,
    // and `Ok(None)` promises there is none. Written here rather than inside the
    // analysis because the analysis may run twice and only the second one
    // describes the part.
    if stages.stopped() {
        return Ok(None);
    }
    write_stress_json(problem, &analysis)?;
    let analysis_s = start.elapsed().as_secs_f64();

    stages.exporting();
    let start = Instant::now();
    let SurfaceExport {
        mesh,
        stats,
        islands,
        clamp,
    } = export(problem, densities)?;
    let export_s = start.elapsed().as_secs_f64();

    let Analysis {
        voids,
        solids,
        stress,
    } = analysis;
    Ok(Some(Completed {
        voids,
        solids,
        stress,
        trim,
        flush,
        reinforce,
        mesh,
        stats,
        islands,
        clamp,
        analysis_s,
        export_s,
    }))
}

/// Run the configured engine, post-process and export the resulting surface.
pub fn optimize_and_export(problem: &Problem, reporter: &dyn Reporter) -> Result<RunOutcome> {
    let start = Instant::now();
    let mut field = optimize(problem, reporter)?;
    let optimize_s = start.elapsed().as_secs_f64();

    let Completed {
        voids,
        solids,
        stress,
        trim,
        flush,
        reinforce,
        mesh,
        stats,
        islands,
        clamp,
        analysis_s,
        export_s,
    } = complete(
        problem,
        &mut field.densities,
        StressLimits::default(),
        &Unwatched,
    )?
    .expect("an unwatched pipeline is never stopped");

    Ok(RunOutcome {
        field,
        voids,
        solids,
        stress,
        trim,
        flush,
        reinforce,
        mesh,
        stats,
        islands,
        clamp,
        stl_path: problem.output.stl_path.clone(),
        optimize_s,
        analysis_s,
        export_s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// A problem small enough to run twice in a test and resolved enough that
    /// its density field really crosses the iso level, so the export the
    /// degraded path has to keep is a real one.
    const TINY: &str = r#"
[project]
name = "degraded"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

# The reproducible backend: this fixture's numbers are compared, and the default
# backend is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0
max_iterations = 15

[output]
stl_path = "growforge_degraded.stl"
stress_json = "growforge_degraded_stress.json"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [32.0, 16.0, 16.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 16.5, 16.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [32.0, 8.0, 8.0], radius = 4.0 }
vector = [0.0, 0.0, -20.0]
"#;

    /// A problem that never settles: a miniature of the *deep* shelf bracket -
    /// the shipped one reaching two and a half times as far out from the wall -
    /// at 15 x 2 x 7 cells, and the same configuration the SIMP engine's own
    /// stall test uses. Under the optimality criteria step every iteration takes
    /// a step of the full 0.2 move limit and the compliance goes nowhere, which
    /// is the documented signature this run is stopped on; the loop calls it at
    /// iteration 123.
    ///
    /// Nothing here is contrived to defeat the convergence tolerance: it is the
    /// default 0.01, and the design simply never reaches it. The budget is 250
    /// so that a test which is supposed to stop early cannot cost minutes if it
    /// ever stops doing so.
    const STUBBORN: &str = r#"
[project]
name = "stubborn"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "petg"

[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.16
min_feature_mm = 12.0
max_iterations = 250
convergence_tol = 0.01

  [optimization.overhang]
  build_direction = "z+"

[output]
stl_path = "growforge_stalled.stl"
stress_json = "growforge_stalled_stress.json"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [60.0, 8.0, 28.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 0.0]
max = [4.0, 8.0, 28.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 24.0]
max = [60.0, 8.0, 28.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 8.5, 28.5] }
directions = ["x", "y", "z"]

[[loadcases]]
name = "shelf-load"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [-0.5, -0.5, 24.5], max = [60.5, 8.5, 28.5] }
vector = [0.0, 0.0, -400.0]
[[loadcases.loads]]
type = "gravity"
"#;

    fn tiny_problem() -> Problem {
        let config = Config::parse(TINY).expect("parse");
        Problem::build(&config, &std::env::temp_dir()).expect("build")
    }

    /// Run a configuration all the way through the export and hand back what it
    /// produced and what it said about itself, leaving nothing behind on disk.
    ///
    /// The output paths are rewritten to `name` rather than taken from the
    /// configuration: the suite runs its tests in parallel, and two of them
    /// writing - or asserting the absence of - one path in the temp directory is
    /// a race rather than a test.
    fn finished(text: &str, name: &str) -> (RunOutcome, Vec<String>) {
        use std::sync::Mutex;

        #[derive(Default)]
        struct Notes(Mutex<Vec<String>>);
        impl report::Reporter for Notes {
            fn iteration(&self, _stats: &report::IterationStats) {}
            fn note(&self, message: &str) {
                self.0.lock().expect("notes").push(message.to_string());
            }
        }

        let config = Config::parse(text).expect("parse");
        let directory = std::env::temp_dir();
        let mut problem = Problem::build(&config, &directory).expect("build");
        problem.output.stl_path = directory.join(format!("growforge_{name}.stl"));
        problem.output.stress_json = Some(directory.join(format!("growforge_{name}_stress.json")));
        let notes = Notes::default();
        let outcome = optimize_and_export(&problem, &notes).expect("run");
        std::fs::remove_file(&outcome.stl_path).ok();
        if let Some(json) = &problem.output.stress_json {
            std::fs::remove_file(json).ok();
        }
        (outcome, notes.0.into_inner().expect("notes"))
    }

    /// The three ways a finished run can end, each of them travelling out of the
    /// engine, through the analysis and the export, and into the outcome the
    /// command line reads its summary line from.
    ///
    /// The point of the third one is that it is not special: a run the loop
    /// stopped because the problem would not settle is analysed, meshed,
    /// validated and written exactly like a run that converged or one that spent
    /// its budget. Only the sentence printed about it differs.
    #[test]
    fn every_stop_reason_travels_through_the_run_outcome() {
        let (capped, capped_notes) = finished(TINY, "capped");
        assert_eq!(capped.field.stop, engine::StopReason::IterationCap);
        assert_eq!(capped.field.iterations, 15);
        assert!(!capped.field.stop.converged());
        assert!(
            capped_notes.iter().any(|n| n.contains("iteration cap")),
            "{capped_notes:?}"
        );

        let (converged, converged_notes) = finished(
            &TINY.replace(
                "max_iterations = 15",
                "max_iterations = 250\nconvergence_tol = 0.05",
            ),
            "converged",
        );
        assert_eq!(converged.field.stop, engine::StopReason::Converged);
        assert!(converged.field.stop.converged());
        assert!(converged.field.iterations < 250);
        assert!(
            converged_notes
                .iter()
                .any(|n| n.contains("converged after")),
            "{converged_notes:?}"
        );

        let (stalled, stalled_notes) = finished(STUBBORN, "stalled");
        assert_eq!(stalled.field.stop, engine::StopReason::Stalled);
        assert!(!stalled.field.stop.converged());
        assert!(
            stalled.field.iterations >= constants::STALL_WINDOW_ITERATIONS
                && stalled.field.iterations < 250,
            "the stall has to be what stopped it: {} iterations",
            stalled.field.iterations
        );
        assert!(
            stalled_notes
                .iter()
                .any(|n| n.contains("still traversing") && n.contains("update = \"mma\"")),
            "{stalled_notes:?}"
        );
        // The design that came out of it is a design: on target, printable, and
        // stress analysed like any other. Nothing downstream treats it as a
        // failure, and the run exits zero.
        assert!(
            (stalled.field.volume_fraction - 0.16).abs() < 1e-3,
            "volume fraction {} missed the target",
            stalled.field.volume_fraction
        );
        assert!(stalled.field.overhang_residual.is_some());
        assert!(
            stalled.stress.is_available(),
            "a stalled run is stress analysed like any other: {:?}",
            stalled.stress.reason()
        );

        // Three labels, three sentences, and three real parts.
        assert_eq!(
            [
                capped.field.stop.label(),
                converged.field.stop.label(),
                stalled.field.stop.label()
            ],
            ["iteration cap", "converged", "stalled"]
        );
        for outcome in [&capped, &converged, &stalled] {
            assert!(outcome.stats.triangles > 0 && outcome.stats.volume_mm3 > 0.0);
            assert_eq!(outcome.mesh.triangles.len(), outcome.stats.triangles);
        }
    }

    #[test]
    fn a_stress_solve_that_cannot_converge_degrades_instead_of_failing_the_run() {
        let problem = tiny_problem();
        let json = problem.output.stress_json.clone().expect("a json path");
        std::fs::remove_file(&json).ok();

        let mut field = optimize(&problem, &report::SilentReporter).expect("optimize");
        // One iteration cannot reach any useful residual, which is exactly the
        // conjugate gradient cliff this path exists for.
        let Analysis { voids, stress, .. } = analyse_with(
            &problem,
            &mut field.densities,
            stress::StressLimits {
                tolerance: 1e-12,
                max_iterations: 1,
            },
            CancelProbe::NONE,
        )
        .expect("the analysis itself must still succeed")
        .expect("nothing asked this analysis to stop");

        let reason = stress.reason().expect("a reason");
        assert!(!stress.is_available() && stress.report().is_none());
        assert!(
            reason.contains("did not converge"),
            "the reason must say what went wrong: {reason}"
        );
        assert!(
            reason.contains("tip"),
            "the reason must name the load case: {reason}"
        );
        // The cavity pass ran, no JSON was written, and the surface still
        // exports: the run has lost the check, not the part.
        assert_eq!(voids.policy, problem.output.voids);
        assert!(!json.exists(), "no stress JSON may be written");
        let exported = export(&problem, &field.densities).expect("export");
        assert!(exported.stats.triangles > 0 && !exported.mesh.triangles.is_empty());
        mesh::validate::validate(&exported.mesh).expect("watertight surface");

        // And the console line says so rather than printing a table of zeros.
        report::print_stress_report(&stress, exported.islands.bodies.len());
        std::fs::remove_file(&problem.output.stl_path).ok();
    }

    #[test]
    fn a_converging_stress_solve_writes_its_json_and_reports_a_table() {
        let problem = tiny_problem();
        let json = problem.output.stress_json.clone().expect("a json path");
        std::fs::remove_file(&json).ok();

        let mut field = optimize(&problem, &report::SilentReporter).expect("optimize");
        let analysis = analyse(&problem, &mut field.densities).expect("analyse");
        // A resolved little cantilever is one part, not a scattering of them.
        assert_eq!(
            analysis.solids.len(),
            1,
            "the field came out in {} pieces",
            analysis.solids.len()
        );
        let stress = &analysis.stress;
        assert!(stress.is_available() && stress.reason().is_none());
        assert_eq!(stress.report().expect("a report").cases.len(), 1);
        assert!(json.exists(), "the JSON report must be written");
        std::fs::remove_file(&json).ok();
    }

    /// A cantilever with a long spur hanging off the middle of it, for the
    /// post-run pipeline itself rather than for an engine.
    ///
    /// The field is written by hand below rather than optimized, because what is
    /// under test is what the pipeline does with a shape - a load path and an
    /// appendage that carries nothing - and not whether some engine converges on
    /// one. 16 x 16 x 8 cells of 2 mm.
    const SPURRED: &str = r#"
[project]
name = "spurred"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

# The reproducible backend: this fixture's numbers are compared.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0
max_iterations = 1

[output]
stl_path = "growforge_spurred.stl"
trim = "stress"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [32.0, 32.0, 16.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 32.5, 16.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [32.0, 16.0, 8.0], radius = 2.5 }
vector = [0.0, 0.0, -40.0]
"#;

    /// The fixture's problem, its hand-written field, and the cells of the spur.
    ///
    /// The bar runs the length of the domain at a two-by-two cross-section, from
    /// the support region to the loaded nodes; the spur stands seven cells off
    /// the middle of it into free space and is attached to nothing else. The
    /// background is a tenth rather than the SIMP floor: it is well below the iso
    /// level, so it is not part of the part, and it keeps the stiffness contrast
    /// of a hand-written field near the one an optimized field really has.
    fn spurred(name: &str) -> (Problem, Vec<f64>, Vec<usize>) {
        let config = Config::parse(SPURRED).expect("parse");
        let directory = std::env::temp_dir();
        let mut problem = Problem::build(&config, &directory).expect("build");
        problem.output.stl_path = directory.join(format!("growforge_{name}.stl"));
        problem.output.stress_json = None;

        let grid = &problem.grid;
        assert_eq!((grid.nx, grid.ny, grid.nz), (16, 16, 8));
        let mut densities = vec![0.1; grid.n_cells()];
        for i in 0..grid.nx {
            for j in 7..9 {
                for k in 3..5 {
                    densities[grid.cell_index(i, j, k)] = 1.0;
                }
            }
        }
        let mut spur = Vec::new();
        for i in 7..9 {
            for j in 9..16 {
                for k in 3..5 {
                    spur.push(grid.cell_index(i, j, k));
                }
            }
        }
        spur.sort_unstable();
        for &cell in &spur {
            densities[cell] = 1.0;
        }
        (problem, densities, spur)
    }

    /// Cells of a field that the stress pass would evaluate, which is the
    /// reading every table in a run's report is taken over.
    fn evaluated(problem: &Problem, densities: &[f64]) -> usize {
        (0..problem.grid.n_cells())
            .filter(|&e| {
                engine::cell_density(&problem.grid, densities, e)
                    >= constants::STRESS_DENSITY_THRESHOLD
            })
            .count()
    }

    /// The whole point of the pass, end to end: the spur goes, and every number
    /// the run reports afterwards describes the part without it.
    ///
    /// The comparison is against the same field through the same pipeline with
    /// `trim = "off"`, so what is asserted is the difference the pass makes and
    /// nothing about the fixture's absolute numbers.
    #[test]
    fn a_trimmed_run_reports_the_part_it_exported_rather_than_the_one_it_judged() {
        let (mut untrimmed_problem, densities, spur) = spurred("trim_untrimmed");
        untrimmed_problem.output.trim = crate::config::TrimPolicy::Off;
        let mut untrimmed_field = densities.clone();
        let untrimmed = complete(
            &untrimmed_problem,
            &mut untrimmed_field,
            StressLimits::default(),
            &Unwatched,
        )
        .expect("the untrimmed pipeline")
        .expect("nothing asked it to stop");
        assert!(untrimmed.trim.is_none(), "the pass ran with trim = \"off\"");

        let (mut problem, densities, _) = spurred("trim_trimmed");
        assert_eq!(problem.output.trim, crate::config::TrimPolicy::Stress);
        // Asked for here so the single JSON write can be pinned to the *second*
        // analysis: the file on disk has to describe the trimmed part, never the
        // one the criterion was read off.
        let json = std::env::temp_dir().join("growforge_trim_trimmed_stress.json");
        std::fs::remove_file(&json).ok();
        problem.output.stress_json = Some(json.clone());
        let mut field = densities.clone();
        let trimmed = complete(&problem, &mut field, StressLimits::default(), &Unwatched)
            .expect("the trimmed pipeline")
            .expect("nothing asked it to stop");

        let report = trimmed.trim.as_ref().expect("a trim report");
        assert_eq!(report.ending, trim::TrimEnding::Removed, "{report:?}");
        assert!(report.cells > 0);
        assert!(report.volume_mm3 > 0.0);
        assert!(report.peak_mpa > 0.0);

        // Exactly the spur, and only ever the spur: the load path is what the
        // stresses are in, so nothing on it can fall below a hundredth of the
        // peak.
        let removed: Vec<usize> = (0..problem.grid.n_cells())
            .filter(|&e| densities[e] != field[e])
            .collect();
        assert_eq!(removed.len(), report.cells);
        for cell in &removed {
            assert!(spur.contains(cell), "cell {cell} is not part of the spur");
            assert_eq!(field[*cell], constants::DENSITY_MIN);
        }

        // The re-analysis is the whole contract: the stress table, the cavity
        // report and the mesh all describe the field that was written, not the
        // one the criterion was read off.
        let table = trimmed.stress.report().expect("a stress table");
        assert_eq!(table.evaluated_cells, evaluated(&problem, &field));
        assert!(
            table.evaluated_cells < evaluated(&untrimmed_problem, &untrimmed_field),
            "the second analysis described the same part as the first"
        );
        assert_eq!(trimmed.voids.policy, problem.output.voids);
        assert!(trimmed.stats.volume_mm3 < untrimmed.stats.volume_mm3);
        assert_eq!(trimmed.solids.len(), 1, "{:?}", trimmed.solids);

        // Including the machine readable one, which is written once, after the
        // analysis that describes the part - not once per analysis, and not off
        // the pre-trim field.
        let written = std::fs::read_to_string(&json).expect("the stress JSON");
        assert!(
            written.contains(&format!("\"evaluated_cells\": {}", table.evaluated_cells)),
            "the JSON describes a different field from the one that shipped: {written}"
        );
        std::fs::remove_file(&json).ok();

        // And what the run says about it, on the console the same way the other
        // reports reach it.
        let notes = report.notes();
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].starts_with("removed"), "{notes:?}");
        assert!(notes[0].contains("cm3"), "{notes:?}");
        report::print_trim_report(trimmed.trim.as_ref());
        report::print_trim_report(untrimmed.trim.as_ref());

        // The file is the trimmed surface, and it is smaller than the one the
        // same field exported untrimmed.
        let (_, written) = mesh::stl::read(&problem.output.stl_path).expect("the STL");
        assert_eq!(written.len(), trimmed.stats.triangles);
        std::fs::remove_file(&problem.output.stl_path).ok();
        std::fs::remove_file(&untrimmed_problem.output.stl_path).ok();
    }

    /// The abort contract: a removal that would take the structure apart is
    /// refused whole, and the run exports the untrimmed part it was going to
    /// export anyway.
    ///
    /// Provoked by asking for a threshold of ninety-nine percent of the peak,
    /// which condemns everything but the hottest cells of the root - including
    /// the middle of the load path. Nothing physical about the fixture changes,
    /// so the file it writes must be the untrimmed one, byte for byte.
    #[test]
    fn a_trim_that_would_separate_two_purposes_is_refused_whole() {
        let (mut problem, densities, _) = spurred("trim_refused");
        problem.output.trim_stress_fraction = 0.99;
        let mut field = densities.clone();
        let refused = complete(&problem, &mut field, StressLimits::default(), &Unwatched)
            .expect("the pipeline")
            .expect("nothing asked it to stop");

        let report = refused.trim.as_ref().expect("a trim report");
        let trim::TrimEnding::Refused { first, second } = &report.ending else {
            panic!("the guard let it through: {report:?}");
        };
        assert_eq!(first, "support 1");
        assert_eq!(second, "load 1 of case \"tip\"");
        assert!(report.cells > 0, "the count is what it would have removed");
        assert_eq!(field, densities, "a refused pass may not touch the field");

        // Loudly: the part in the file is not the one that was asked for.
        let notes = report.notes();
        assert!(notes[0].starts_with("warning: refused"), "{notes:?}");
        assert!(notes[0].contains("support 1"), "{notes:?}");
        report::print_trim_report(refused.trim.as_ref());

        // And the same field with the pass off writes the same file.
        let refused_bytes = std::fs::read(&problem.output.stl_path).expect("the refused STL");
        let (mut off, densities, _) = spurred("trim_refused_off");
        off.output.trim = crate::config::TrimPolicy::Off;
        let mut field = densities;
        let plain = complete(&off, &mut field, StressLimits::default(), &Unwatched)
            .expect("the pipeline")
            .expect("nothing asked it to stop");
        assert!(plain.trim.is_none());
        assert_eq!(
            refused_bytes,
            std::fs::read(&off.output.stl_path).expect("the untrimmed STL"),
            "a refused trim wrote something other than the untrimmed part"
        );
        std::fs::remove_file(&problem.output.stl_path).ok();
        std::fs::remove_file(&off.output.stl_path).ok();
    }

    /// The three field passes share one re-analysis, and none of them is what
    /// the count of analyses depends on: it is whether the field changed.
    ///
    /// Five runs of the same fixture - all three passes, each alone, and none -
    /// counted through a scripted [`Stages`]. The set is the case worth
    /// pinning: a re-analysis per pass would solve the same field three times
    /// for nothing, and no re-analysis at all would leave every number the run
    /// reports describing a part that is not in the file.
    #[test]
    fn the_trim_the_flush_and_the_reinforcement_share_one_re_analysis() {
        use crate::config::{FlushPolicy, ReinforcePolicy, TrimPolicy};

        let run = |name: &str, trim: TrimPolicy, flush: FlushPolicy, reinforce: ReinforcePolicy| {
            let (mut problem, densities, _) = spurred(name);
            problem.output.trim = trim;
            problem.output.flush = flush;
            problem.output.reinforce = reinforce;
            let mut field = densities.clone();
            let stages = Scripted::default();
            let completed = complete(&problem, &mut field, StressLimits::default(), &stages)
                .expect("the pipeline")
                .expect("nothing asked it to stop");
            std::fs::remove_file(&problem.output.stl_path).ok();
            (problem, densities, field, completed, stages.counts())
        };

        // All three: one analysis for the trim's criterion, one for the part
        // that ships, and a single export.
        let (problem, before, field, all, (_, analysing, exporting)) = run(
            "reanalysis_all",
            TrimPolicy::Stress,
            FlushPolicy::Walls,
            ReinforcePolicy::MinThickness,
        );
        assert_eq!((analysing, exporting), (2, 1));
        let trimmed = all.trim.as_ref().expect("a trim report");
        let flushed = all.flush.as_ref().expect("a flush report");
        let reinforced = all.reinforce.as_ref().expect("a reinforce report");
        assert!(trimmed.changed_the_field(), "{trimmed:?}");
        assert!(flushed.changed_the_field(), "{flushed:?}");
        assert!(reinforced.changed_the_field(), "{reinforced:?}");
        assert_eq!(
            reinforced.thickness_mm, problem.optimization.min_feature_mm,
            "the floor defaulted to something other than min_feature_mm"
        );
        assert_eq!(
            flushed.depth_mm,
            constants::FLUSH_DEPTH_VOXELS * problem.grid.h,
            "the depth defaulted to something other than the voxel multiple"
        );
        // And what the run reports describes the field it wrote, which is the
        // one all three passes left: the stress table is taken over the filled
        // and reinforced cells, not the ones the criterion was read off.
        let table = all.stress.report().expect("a stress table");
        assert_eq!(table.evaluated_cells, evaluated(&problem, &field));
        assert!(
            evaluated(&problem, &field) > evaluated(&problem, &before),
            "the two filling passes added no material at all"
        );

        // Any one of them alone still forces the second analysis, and none of
        // them leaves the sequence as it was before the passes existed.
        let (_, _, _, trim_only, (_, analysing, _)) = run(
            "reanalysis_trim",
            TrimPolicy::Stress,
            FlushPolicy::Off,
            ReinforcePolicy::Off,
        );
        assert_eq!(analysing, 2);
        assert!(trim_only.trim.expect("a report").changed_the_field());
        assert!(trim_only.flush.is_none());
        assert!(trim_only.reinforce.is_none());

        // The flush alone, which is also where the honesty of the second
        // analysis is asserted on this pass by itself: the stress table counts
        // the cells the fill left, and there are more of them than the field
        // the pipeline was handed had.
        let (flush_problem, before, flush_field, flush_only, (_, analysing, _)) = run(
            "reanalysis_flush",
            TrimPolicy::Off,
            FlushPolicy::Walls,
            ReinforcePolicy::Off,
        );
        assert_eq!(analysing, 2);
        assert!(flush_only.trim.is_none());
        assert!(flush_only.reinforce.is_none());
        let flushed = flush_only.flush.expect("a report");
        assert!(flushed.changed_the_field(), "{flushed:?}");
        let table = flush_only.stress.report().expect("a stress table");
        assert_eq!(
            table.evaluated_cells,
            evaluated(&flush_problem, &flush_field),
            "the safety factor describes the field the fill was read off, not the one that shipped"
        );
        assert!(evaluated(&flush_problem, &flush_field) > evaluated(&flush_problem, &before));
        report::print_flush_report(Some(&flushed));

        let (_, _, _, reinforce_only, (_, analysing, _)) = run(
            "reanalysis_reinforce",
            TrimPolicy::Off,
            FlushPolicy::Off,
            ReinforcePolicy::MinThickness,
        );
        assert_eq!(analysing, 2);
        assert!(reinforce_only.trim.is_none());
        assert!(reinforce_only.flush.is_none());
        assert!(
            reinforce_only
                .reinforce
                .expect("a report")
                .changed_the_field()
        );

        let (_, _, _, neither, (_, analysing, exporting)) = run(
            "reanalysis_neither",
            TrimPolicy::Off,
            FlushPolicy::Off,
            ReinforcePolicy::Off,
        );
        assert_eq!((analysing, exporting), (1, 1));
        assert!(neither.trim.is_none() && neither.reinforce.is_none());
        assert!(neither.flush.is_none());

        // The part that came out of the three is thicker than the one that came
        // out of no pass at all, and both are still parts.
        assert!(all.stats.volume_mm3 > neither.stats.volume_mm3);
        for completed in [&all, &neither] {
            mesh::validate::validate(&completed.mesh).expect("watertight surface");
        }
        report::print_reinforce_report(all.reinforce.as_ref());
        report::print_reinforce_report(neither.reinforce.as_ref());
        report::print_flush_report(neither.flush.as_ref());
    }

    /// A solid block held on one face and pushed on the other, small enough and
    /// solved loosely enough that the conjugate gradient converges inside a
    /// single cancellation checkpoint interval.
    ///
    /// That is what makes it a *boundary* fixture: the solve never puts the stop
    /// question at all, so every question [`complete`] asks is one of its own
    /// stage boundaries and a script can land on exactly one of them. 4 x 2 x 2
    /// cells.
    const BOUNDARY: &str = r#"
[project]
name = "boundary"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "pla"

# The reproducible backend: the count of stop questions is asserted, and a
# device backend asks its own at every readback.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.5
min_feature_mm = 8.0
max_iterations = 1

[output]
stl_path = "growforge_boundary.stl"
stress_json = "growforge_boundary_stress.json"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [16.0, 8.0, 8.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 8.5, 8.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [15.5, -0.5, -0.5], max = [16.5, 8.5, 8.5] }
vector = [0.0, 0.0, -10.0]
"#;

    /// The fixture's problem, with both output paths unique to `name`, and the
    /// solid field it is analysed on.
    fn boundary(name: &str) -> (Problem, Vec<f64>) {
        let config = Config::parse(BOUNDARY).expect("parse");
        let directory = std::env::temp_dir();
        let mut problem = Problem::build(&config, &directory).expect("build");
        problem.output.stl_path = directory.join(format!("growforge_{name}.stl"));
        problem.output.stress_json = Some(directory.join(format!("growforge_{name}_stress.json")));
        let densities = vec![1.0; problem.grid.n_cells()];
        (problem, densities)
    }

    /// A [`Stages`] that answers the stop question from a script and counts
    /// every hook it is put through.
    ///
    /// `stop_on` is the one-based index of the question the answer turns to
    /// "stopped" on, and it stays stopped from there: a stop is not something a
    /// pipeline can talk its way out of by asking again. Zero never stops.
    #[derive(Default)]
    struct Scripted {
        stop_on: usize,
        asked: std::sync::atomic::AtomicUsize,
        analysing: std::sync::atomic::AtomicUsize,
        exporting: std::sync::atomic::AtomicUsize,
    }

    impl Scripted {
        fn counts(&self) -> (usize, usize, usize) {
            use std::sync::atomic::Ordering;
            (
                self.asked.load(Ordering::Relaxed),
                self.analysing.load(Ordering::Relaxed),
                self.exporting.load(Ordering::Relaxed),
            )
        }
    }

    impl Stages for Scripted {
        fn analysing(&self) {
            self.analysing
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        fn exporting(&self) {
            self.exporting
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        fn stopped(&self) -> bool {
            let asked = self
                .asked
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            self.stop_on != 0 && asked >= self.stop_on
        }
    }

    /// The boundary between the analysis and the export writes *nothing*, the
    /// stress JSON included.
    ///
    /// `Ok(None)` says nothing was written, the editor prints "stopped; no file
    /// was written" on the strength of it, and the machine readable report is a
    /// file the run leaves behind exactly as the STL is. It was written before
    /// this check rather than after it, which made both statements false for any
    /// configuration carrying `stress_json`; the JSON was also invisible to the
    /// editor's own record of what it wrote.
    #[test]
    fn a_stop_at_the_export_boundary_leaves_no_file_behind_at_all() {
        // Loose enough that the solve converges in far fewer iterations than
        // the cancellation checkpoint interval, so it never asks the question.
        let limits = StressLimits {
            tolerance: 1e-3,
            max_iterations: 500,
        };
        let (problem, densities) = boundary("stop_boundary");
        let json = problem.output.stress_json.clone().expect("a json path");
        std::fs::remove_file(&json).ok();
        std::fs::remove_file(&problem.output.stl_path).ok();

        // The mirror: nothing stops it, and it writes both files.
        let watched = Scripted::default();
        let mut field = densities.clone();
        let completed = complete(&problem, &mut field, limits, &watched)
            .expect("the pipeline")
            .expect("nothing asked it to stop");
        assert!(completed.stress.is_available());
        assert!(json.exists(), "an unstopped run must write its JSON");
        assert!(problem.output.stl_path.exists());
        let (asked, analysing, exporting) = watched.counts();
        assert_eq!((analysing, exporting), (1, 1));
        assert_eq!(
            asked, 2,
            "the fixture has to put the question at the two stage boundaries and nowhere else, \
             or the script below cannot land on the second of them"
        );
        std::fs::remove_file(&json).ok();
        std::fs::remove_file(&problem.output.stl_path).ok();

        // And the same run, stopped at the second of those questions.
        let stopped = Scripted {
            stop_on: asked,
            ..Scripted::default()
        };
        let mut field = densities;
        let outcome =
            complete(&problem, &mut field, limits, &stopped).expect("a stop is not an error");
        assert!(outcome.is_none(), "a stopped pipeline has no result");
        let (asked, analysing, exporting) = stopped.counts();
        assert_eq!(
            (asked, analysing, exporting),
            (2, 1, 0),
            "the stop has to land at the boundary after the analysis, not inside it"
        );
        assert!(
            !json.exists(),
            "a stopped run left the stress JSON at {}",
            json.display()
        );
        assert!(
            !problem.output.stl_path.exists(),
            "a stopped run left an STL at {}",
            problem.output.stl_path.display()
        );
    }

    /// A stop that lands inside the stress solve unwinds the whole analysis:
    /// no report, no JSON, and nothing for the caller to export.
    #[test]
    fn a_cancelled_stress_solve_unwinds_the_analysis_and_writes_nothing() {
        let problem = tiny_problem();
        let json = problem.output.stress_json.clone().expect("a json path");
        std::fs::remove_file(&json).ok();

        let mut field = optimize(&problem, &report::SilentReporter).expect("optimize");
        let stop = || true;
        let outcome = analyse_with(
            &problem,
            &mut field.densities,
            stress::StressLimits::default(),
            CancelProbe::watching(&stop),
        )
        .expect("a stop is not an error");
        assert!(outcome.is_none(), "a stopped analysis has no result");
        assert!(!json.exists(), "a stopped analysis may not write its JSON");
    }
}
