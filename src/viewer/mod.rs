//! Native 3D viewer.
//!
//! Three entry points share one window: [`view_setup`] shows the problem as the
//! optimizer will see it and runs nothing, [`run_with_view`] shows the density
//! isosurface evolving over a run, and [`editor::edit`] turns the same window
//! into a visual editor of the problem definition itself.
//!
//! Threading, because Windows requires the event loop to own the main thread:
//!
//! ```text
//!   main thread      winit event loop, egui, wgpu
//!   worker thread    engine -> export -> STL
//!   mesher thread    marching cubes on the newest density snapshot
//! ```
//!
//! The three are joined by [`snapshot::LatestSlot`] channels of capacity one,
//! so a slow consumer costs dropped intermediate frames instead of memory.
//! Closing the window detaches the viewer: the worker stops paying for
//! snapshots and finishes the run headless, STL included.

pub mod camera;
pub mod editor;
pub mod scene;
pub mod snapshot;
pub mod tessellate;

mod app;
mod gpu;
mod ui;

pub use crate::viewer::editor::edit;

use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};

use crate::RunOutcome;
use crate::config::Config;
use crate::constants;
use crate::problem::Problem;
use crate::report::Reporter;
use crate::viewer::app::ViewerApp;
use crate::viewer::scene::{Layer, LayerMesh, Shading};
use crate::viewer::snapshot::{Frame, FrameKind, RunStatus, ViewLink, ViewReporter};

fn window_title(problem: &Problem) -> String {
    format!("{} - {}", constants::VIEW_WINDOW_TITLE, problem.name)
}

/// Closes a link's channels when it goes out of scope, on the panicking path
/// as well as the normal one.
///
/// A scope joins its threads after its closure ends. The mesher blocks on the
/// snapshot channel, so an unwind that skipped the close would hang that join
/// forever instead of propagating the panic.
struct CloseOnDrop<'a> {
    link: &'a ViewLink,
    detach: bool,
}

impl Drop for CloseOnDrop<'_> {
    fn drop(&mut self) {
        if self.detach {
            self.link.detach();
        } else {
            self.link.densities.close();
        }
    }
}

/// Marks the run failed if the worker unwinds rather than returning an error.
///
/// A panic never reaches the `inspect_err` that reports a failed `Result`, so
/// without this the window would sit on stale "optimizing" stats until someone
/// closed it. The panic itself is untouched and still propagates out of the
/// worker thread, where the join turns it into the process's error exit.
///
/// Visible to the editor because its on-demand export runs on a thread of its
/// own, with the same window behind it and the same thing to lose.
pub(crate) struct FailOnPanic<'a>(pub(crate) &'a ViewLink);

impl Drop for FailOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.set_status(RunStatus::Failed(PANICKED.to_string()));
        }
    }
}

/// What both the window and the console call a worker that panicked.
const PANICKED: &str = "the optimization thread panicked";

/// Open a window showing the problem setup: the voxelized domain, the keepout
/// and keepin regions, the constrained nodes and the loads. Nothing is
/// optimized and nothing is written.
pub fn view_setup(config: &Config, problem: &Problem) -> Result<()> {
    gpu::probe_adapter()?;
    let scene = scene::build(config, problem)?;
    ViewerApp::new(window_title(problem), scene, None).run()
}

/// Run the optimization on a worker thread behind a live window.
///
/// The window never cancels or alters the run: closing it detaches the viewer
/// and the optimization finishes headless, still writing its STL. Progress
/// keeps going to `reporter`, which is also what the console prints.
pub fn run_with_view(
    config: &Config,
    problem: &Problem,
    reporter: &dyn Reporter,
) -> Result<RunOutcome> {
    gpu::probe_adapter()?;
    let scene = scene::build(config, problem)?;
    let title = window_title(problem);
    let link = ViewLink::new();

    std::thread::scope(|threads| {
        // However the window ends, including a panic, the run is on its own
        // from here and must stop paying for snapshots nobody will draw.
        let _detach = CloseOnDrop {
            link: &link,
            detach: true,
        };
        let worker = threads.spawn(|| run_worker(problem, reporter, &link, &keeps_nothing));
        let window = {
            // The window keeps servicing its message queue until the worker
            // is done, so this borrow has to end before the join below.
            let still_running = || !worker.is_finished();
            ViewerApp::new(title, scene, Some(&link))
                .watching(&still_running)
                .running(&problem.engine)
                .run()
        };
        link.detach();
        if let Err(error) = window {
            eprintln!("warning: the viewer stopped ({error:#}); the run continues headless");
        }
        // Nothing here ever asks a run to stop - only the editor's stop button
        // does - so the outcome is always there.
        worker
            .join()
            .unwrap_or_else(|_| Err(anyhow!(PANICKED)))?
            .ok_or_else(|| anyhow!("the run was stopped before it could export"))
    })
}

/// Optimize, export and publish both to the window.
///
/// Shared with the editor's "run full", so a run started from the editor is the
/// same run, writing the same file, as `growforge run --view`.
///
/// Returns `Ok(None)` when the reporter asked for the run to stop. The engine
/// answers that between its own iterations, the linear solves answer it inside
/// themselves - see [`crate::fea::CancelProbe`] - and [`finish`] asks again at
/// each stage boundary, so a stop reaches the run before anything is analysed
/// and, above all, before anything is written. A stopped run leaves no file
/// behind of its own accord - not a partial one, not a stale one. The console
/// reporters never ask for a stop, so `growforge run --view` cannot see it.
///
/// "Of its own accord" is the whole of it: the editor keeps the field of the run
/// behind its window, and the user may then ask for that field's deliverables
/// explicitly - see [`editor::worker::Worker::generate`], which goes through
/// [`finish`] exactly as this does. Nothing about a stop is what writes then; a
/// button is.
///
/// A run that *fails* is a different thing from one that is stopped: the error
/// travels to the window as [`RunStatus::Failed`] and out to the caller, which
/// on the command line ends the process and in the editor ends only the run.
///
/// `keep` is handed the field the engine produced, once, at the moment it
/// produces it - which is what leaves a design behind for a run that reported no
/// iteration of one. See [`keeps_nothing`] for the callers that have nowhere to
/// keep it.
pub(crate) fn run_worker(
    problem: &Problem,
    reporter: &dyn Reporter,
    link: &ViewLink,
    keep: &dyn Fn(&[f64]),
) -> Result<Option<RunOutcome>> {
    // Declared first so it drops last: whatever unwinds below, the window is
    // told before this function's frame is gone.
    let _fail_on_panic = FailOnPanic(link);
    let view_reporter = ViewReporter::new(reporter, link);

    let start = Instant::now();
    let field = std::thread::scope(|threads| {
        // Ending the mesher here, before the scope joins it, is also what
        // guarantees no preview can land after the final surface below.
        let _stop_mesher = CloseOnDrop {
            link,
            detach: false,
        };
        threads.spawn(|| preview_loop(problem, link));
        crate::optimize(problem, &view_reporter)
    })
    .inspect_err(|error| link.set_status(RunStatus::Failed(format!("{error:#}"))))?;
    let optimize_s = start.elapsed().as_secs_f64();

    let mut field = field;
    // The engine's field, handed over before [`finish`] resolves it in place.
    // The order is the whole of it: what is kept here is what may later be asked
    // for the deliverables of, through this same `finish`, and a field that had
    // already been through the cavity policy and the `[output]` passes would have
    // them applied to it a second time - the flush's fringe reaches further out
    // on every pass, so that is corruption rather than a no-op.
    //
    // Engine-agnostic on purpose. An engine that reports no iteration at all -
    // the solid one reports none, by design - leaves its design behind here
    // exactly as one that reports five hundred does; for the others this is the
    // last reported iteration's field again, which is the same field or the one
    // it settled on.
    keep(&field.densities);
    // The stress solve is the longest single call of the analysis stage, so the
    // stop reaches inside it rather than waiting at the far end of it.
    let stop = || reporter.cancelled();
    let Some(Finished {
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
    }) = finish(problem, &mut field.densities, link, &stop)?
    else {
        return Ok(None);
    };

    Ok(Some(RunOutcome {
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
    }))
}

/// The `keep` of a caller with nowhere to keep a design: `growforge run --view`
/// writes its file and leaves, so nothing outlives the run to export the field
/// again from.
pub(crate) fn keeps_nothing(_densities: &[f64]) {}

/// Everything [`finish`] produced from a density field.
///
/// The part of a [`RunOutcome`] that does not describe the optimization itself,
/// which is what the two callers have in common: one has a field it just
/// optimized, the other a field a run left behind.
pub(crate) struct Finished {
    /// What the enclosed cavity pass found and did.
    pub voids: crate::voids::VoidReport,
    /// The connected bodies of material in the exported field, largest first.
    pub solids: Vec<crate::voids::SolidBody>,
    /// Von Mises stresses of the exported structure, or why they are missing.
    pub stress: crate::stress::StressOutcome,
    /// What the `[output] trim` pass removed, or why it removed nothing;
    /// `None` under the default `trim = "off"`. Its note lines are also on the
    /// link, which is what puts them in the editor's panel.
    pub trim: Option<crate::trim::TrimReport>,
    /// What the `[output] flush` pass filled out to the surfaces the walls rest
    /// on; `None` under the default `flush = "off"`. Its note lines are also on
    /// the link, which is what puts them in the editor's panel.
    pub flush: Option<crate::flush::FlushReport>,
    /// What the `[output] reinforce` pass thickened, and what it could not;
    /// `None` under the default `reinforce = "off"`. Its note lines are also on
    /// the link, which is what puts them in the editor's panel.
    pub reinforce: Option<crate::reinforce::ReinforceReport>,
    /// The exported surface, after smoothing, culling and validation.
    pub mesh: crate::mesh::Mesh,
    /// Statistics of that mesh.
    pub stats: crate::mesh::validate::MeshStats,
    /// Its connected components, and the fragments the island policy removed.
    pub islands: crate::mesh::islands::IslandReport,
    /// What the boundary clamp moved onto the analytic surfaces; `None` under
    /// `boundaries = "voxel"`. Its note lines are also on the link, which is
    /// what puts them in the editor's panel.
    pub clamp: Option<crate::mesh::clamp::ClampReport>,
    /// Wall clock seconds spent analysing, trimming, flushing, reinforcing and
    /// analysing again.
    pub analysis_s: f64,
    /// Wall clock seconds spent extracting, smoothing and writing the mesh.
    pub export_s: f64,
}

/// Everything between a finished density field and the file on disk: the cavity
/// pass, the stress solve, the surface, the STL, and the final frame and status
/// the window shows.
///
/// The tail of [`run_worker`], factored out because the editor exports a field
/// that was kept as well as one a run just produced - see
/// [`editor::worker::Worker::generate`] - and the two must be the same
/// deliverables: the same cavity policy applied to the same field, the same
/// culled mesh in the window and in the file, the same stress layer beside it.
///
/// `densities` is resolved in place by the cavity policy and by the trim, flush
/// and reinforcement passes, so what comes back describes the field the caller
/// now holds. `stop` is asked at every boundary a run asks it at - on the way in,
/// between the analysis and the re-analysis those passes force, and immediately
/// before the write - which is what keeps "nothing that was asked to stop ever
/// exports" true of both callers; `Ok(None)` is that answer.
///
/// The sequence itself is [`crate::complete`]'s, shared with the command line.
/// What is this function's own is the window: the status line at each stage, the
/// final frame with its stress layer, the note lines of the field passes, and
/// the stress summary - all of which reach the editor's panel through the link
/// and the console through the outcome.
pub(crate) fn finish(
    problem: &Problem,
    densities: &mut [f64],
    link: &ViewLink,
    stop: &dyn Fn() -> bool,
) -> Result<Option<Finished>> {
    let stages = ViewStages { link, stop };
    let completed = crate::complete(
        problem,
        densities,
        crate::stress::StressLimits::default(),
        &stages,
    )
    .inspect_err(|error| link.set_status(RunStatus::Failed(format!("{error:#}"))))?;
    let Some(crate::Completed {
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
    }) = completed
    else {
        return Ok(None);
    };

    // The mesh handed back is the culled one, so this frame and the file on
    // disk are the same surface.
    link.frames.push(Frame {
        kind: FrameKind::Final,
        mesh: LayerMesh::from_mesh(&mesh, Layer::Density.info().color, Shading::Smooth),
        stress: stress_layer(problem, densities, &stress, &mesh),
    });
    link.set_trim_notes(
        trim.as_ref()
            .map(crate::trim::TrimReport::notes)
            .unwrap_or_default(),
    );
    link.set_flush_notes(
        flush
            .as_ref()
            .map(crate::flush::FlushReport::notes)
            .unwrap_or_default(),
    );
    link.set_reinforce_notes(
        reinforce
            .as_ref()
            .map(crate::reinforce::ReinforceReport::notes)
            .unwrap_or_default(),
    );
    // What the run was goes into the clamp's lines for the reason the console's
    // copy of them does: whether a vertex resting off a boundary is a defect, a
    // flush that fell short or nothing to say is the run's answer, and the panel
    // has to be told the same one the terminal is.
    link.set_clamp_notes(
        clamp
            .as_ref()
            .map(|report| report.notes(problem.is_solid(), problem.is_flushing()))
            .unwrap_or_default(),
    );
    // The analysis `complete` hands back, which after a trim is the second one:
    // the panel's safety factor describes the part in the file. The body count is
    // the island report's, of that same file, which is what lets the summary say
    // whether the factor holds a part together or describes several pieces.
    link.set_stress_summary(
        stress
            .report()
            .map(|report| report.summary(islands.bodies.len())),
    );
    link.set_status(RunStatus::Finished {
        triangles: stats.triangles,
        volume_mm3: stats.volume_mm3,
        stl_path: problem.output.stl_path.display().to_string(),
    });

    Ok(Some(Finished {
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

/// The window's side of the post-run pipeline: every stage on the status line,
/// and the run's own stop question at every boundary.
struct ViewStages<'a> {
    link: &'a ViewLink,
    stop: &'a dyn Fn() -> bool,
}

impl crate::Stages for ViewStages<'_> {
    fn analysing(&self) {
        // Shown again for the re-analysis the trim, flush and reinforcement
        // passes force: it is the same work, and a window that named it
        // something else would be inventing a stage the pipeline does not have.
        self.link.set_status(RunStatus::Analysing);
    }

    fn exporting(&self) {
        self.link.set_status(RunStatus::Exporting);
    }

    fn stopped(&self) -> bool {
        (self.stop)()
    }
}

/// Colour the exported surface by von Mises stress.
///
/// The load case with the highest peak is the one shown, because that is the
/// one a safety factor is quoted from. The ramp is normalized against the
/// material's yield strength when there is one, so the colours read the same
/// across runs of the same material; without a yield strength it falls back to
/// the peak of the run itself.
///
/// A run whose stress solve did not converge produces no layer at all, which is
/// what leaves the panel's stress switch disabled.
fn stress_layer(
    problem: &Problem,
    densities: &[f64],
    outcome: &crate::stress::StressOutcome,
    mesh: &crate::mesh::Mesh,
) -> Option<LayerMesh> {
    let report = outcome.report()?;
    let worst = report
        .cases
        .iter()
        .max_by(|a, b| a.max_mpa.total_cmp(&b.max_mpa))?;
    let reference = report.yield_strength_mpa.unwrap_or(worst.max_mpa);
    let field = crate::stress::node_stress_field(problem, densities, &worst.von_mises);
    scene::stress_shaded(mesh, &field, reference)
}

/// Extract preview isosurfaces until the snapshot channel closes.
///
/// Only the newest snapshot is ever meshed: anything that arrives while an
/// extraction is running replaces the pending one, so the preview rate adapts
/// to whichever of the solver and the mesher is slower.
///
/// Shared with the editor's auto-regrow, whose previews are meshed by exactly
/// this loop.
pub(crate) fn preview_loop(problem: &Problem, link: &ViewLink) {
    let interval = Duration::from_secs_f64(constants::VIEW_PREVIEW_MIN_INTERVAL_S);
    let mut previous: Option<Instant> = None;
    while let Some(mut snapshot) = link.densities.take_blocking() {
        if let Some(finished) = previous {
            let since = finished.elapsed();
            if since < interval {
                std::thread::sleep(interval - since);
                // Something newer may have landed while we waited.
                if let Some(newer) = link.densities.try_take() {
                    snapshot = newer;
                }
            }
        }
        let Ok(surface) =
            scene::preview_surface(&problem.grid, &snapshot.densities, problem.output.iso_level)
        else {
            // A preview that cannot be extracted is skipped; the run itself
            // reports the same failure when it exports for real.
            previous = Some(Instant::now());
            continue;
        };
        previous = Some(Instant::now());
        let pushed = link.frames.push(Frame {
            kind: FrameKind::Preview {
                iteration: snapshot.iteration,
            },
            mesh: LayerMesh::from_mesh(&surface, Layer::Density.info().color, Shading::Smooth),
            stress: None,
        });
        if !pushed {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::snapshot::DensitySnapshot;
    use std::path::PathBuf;

    const TINY: &str = r#"
[project]
name = "preview"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 16.0

[output]
stl_path = "preview.stl"

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

    fn tiny_problem() -> Problem {
        let config = Config::parse(TINY).expect("parse");
        Problem::build(&config, &PathBuf::from(".")).expect("build")
    }

    #[test]
    fn the_preview_loop_meshes_the_newest_snapshot_and_drops_the_rest() {
        let problem = tiny_problem();
        let cells = problem.grid.n_cells();
        let link = ViewLink::new();
        let flooded = 50;

        std::thread::scope(|threads| {
            let mesher = threads.spawn(|| preview_loop(&problem, &link));
            for iteration in 1..=flooded {
                link.densities.push(DensitySnapshot {
                    iteration,
                    densities: vec![1.0; cells],
                });
            }
            link.densities.close();
            mesher.join().expect("mesher");
        });

        // The mesher can never keep up with a flood, so most snapshots are
        // overwritten in place: memory stays at one pending snapshot.
        assert!(
            link.densities.dropped() > 0,
            "nothing was dropped, so the slot did not replace"
        );
        assert!(!link.densities.is_pending());

        // Whatever it managed to extract, the surface the window ends up with
        // is the newest design that was ever pushed.
        let frame = link.frames.try_take().expect("a preview frame");
        assert_eq!(frame.kind, FrameKind::Preview { iteration: flooded });
        assert!(frame.mesh.triangles() > 0);
        assert!(!link.frames.is_pending());

        // The window is handed a normal per vertex, averaged here on the mesher
        // thread; a preview that arrived facetted would be one the render
        // thread had to smooth itself.
        let facetted = frame.mesh.flat_shaded();
        assert!(
            frame
                .mesh
                .vertices
                .iter()
                .zip(facetted.vertices.iter())
                .any(|(a, b)| a.normal != b.normal),
            "the preview frame carries per-triangle normals"
        );
        for vertex in &frame.mesh.vertices {
            let n = vertex.normal;
            let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-5, "normal {n:?} is not a unit");
        }
    }

    /// The run and setup window names the build as well as the project.
    ///
    /// The version is spelled out here rather than read from the constant the
    /// title is built from, so what is asserted is the string a title bar has to
    /// show and not a second reading of the same value.
    #[test]
    fn the_window_title_names_the_build_and_the_project() {
        let problem = tiny_problem();
        assert_eq!(
            window_title(&problem),
            format!("growforge {} - preview", env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn a_panicking_worker_tells_the_window_before_the_panic_propagates() {
        let link = ViewLink::new();
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = FailOnPanic(&link);
            panic!("the solver exploded");
        }));
        // The panic is reported, not swallowed: the process still fails.
        assert!(unwound.is_err(), "the panic must keep propagating");
        assert_eq!(
            link.progress().status,
            RunStatus::Failed(PANICKED.to_string()),
            "an unwinding worker must not leave the window on stale stats"
        );
    }

    #[test]
    fn a_worker_that_returns_normally_is_never_marked_failed() {
        let link = ViewLink::new();
        {
            let _guard = FailOnPanic(&link);
        }
        assert_eq!(link.progress().status, RunStatus::Optimizing);
        link.set_status(RunStatus::Exporting);
        {
            let _guard = FailOnPanic(&link);
        }
        assert_eq!(link.progress().status, RunStatus::Exporting);
    }

    #[test]
    fn the_preview_loop_stops_when_the_window_detaches() {
        let problem = tiny_problem();
        let cells = problem.grid.n_cells();
        let link = ViewLink::new();
        std::thread::scope(|threads| {
            let mesher = threads.spawn(|| preview_loop(&problem, &link));
            link.densities.push(DensitySnapshot {
                iteration: 1,
                densities: vec![1.0; cells],
            });
            link.detach();
            mesher.join().expect("mesher");
        });
        assert!(link.is_detached());
        assert!(link.densities.is_closed() && link.frames.is_closed());
    }

    /// The same tiny cantilever with a solid block sitting in a domain lobe of
    /// its own, so the exported surface really does come out in two pieces and
    /// the window has something to be wrong about.
    pub const FRAGMENTED: &str = r#"
[project]
name = "fragmented"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "pla"

[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.4
min_feature_mm = 16.0
max_iterations = 5

[output]
stl_path = "growforge_fragmented.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [32.0, 16.0, 16.0]

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 32.0]
max = [16.0, 16.0, 48.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 32.0]
max = [16.0, 16.0, 48.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 16.5, 16.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [32.0, 8.0, 8.0], radius = 4.0 }
vector = [0.0, 0.0, -20.0]
"#;

    /// Two lumps in domain lobes of their own, each holding a support and a load
    /// of its own and joined to the other by nothing.
    ///
    /// The shape of the incident the multi-body warning exists for: every load
    /// reaches a support through material, so the analysis is well posed and the
    /// solve converges, and the safety factor that comes out of it describes two
    /// pieces held apart by air rather than the one part that was wanted. Small
    /// enough - 16 x 4 x 4 cells - to be exported from a saturated field without
    /// optimizing anything.
    pub const TWO_ANCHORED_LUMPS: &str = r#"
[project]
name = "two_lumps"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "pla"

[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.5
min_feature_mm = 16.0
max_iterations = 5

[output]
stl_path = "growforge_two_lumps.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [24.0, 16.0, 16.0]

[[domain]]
op = "add"
shape = "box"
min = [40.0, 0.0, 0.0]
max = [64.0, 16.0, 16.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 16.5, 16.5] }

[[supports]]
region = { shape = "box", min = [63.5, -0.5, -0.5], max = [64.5, 16.5, 16.5] }

[[loadcases]]
name = "pull"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [19.5, -0.5, -0.5], max = [24.5, 16.5, 16.5] }
vector = [0.0, 0.0, -30.0]
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [39.5, -0.5, -0.5], max = [44.5, 16.5, 16.5] }
vector = [0.0, 0.0, 30.0]
"#;

    /// The window's final frame is the file: a run that culls a fragment must
    /// publish the culled surface, not the one marching cubes produced. This is
    /// the seam the editor's "run full" hangs off too, since it is this same
    /// worker.
    #[test]
    fn the_final_frame_is_the_surface_that_was_written() {
        let directory = std::env::temp_dir();
        let config = Config::parse(FRAGMENTED).expect("parse");
        let mut problem = Problem::build(&config, &directory).expect("build");
        problem.output.stl_path = directory.join("growforge_viewer_fragmented.stl");

        let link = ViewLink::new();
        let outcome = run_worker(
            &problem,
            &crate::report::SilentReporter,
            &link,
            &keeps_nothing,
        )
        .expect("the run must finish")
        .expect("nothing asked it to stop");

        // The block in the second lobe is welded to nothing, and it left.
        assert!(
            outcome.islands.components > 1,
            "the fixture exported one piece, so it tests nothing"
        );
        assert_eq!(outcome.islands.bodies.len(), 1);
        assert!(outcome.islands.culled_fragments > 0);
        assert!(outcome.islands.culled_triangles > 0);
        assert_eq!(
            crate::mesh::islands::label_components(&outcome.mesh).components,
            1 + outcome.islands.cavity_shells
        );

        // The frame the window ends up on carries exactly that surface.
        let mut final_frame = None;
        while let Some(frame) = link.frames.try_take() {
            if frame.kind == FrameKind::Final {
                final_frame = Some(frame);
            }
        }
        let frame = final_frame.expect("a final frame");
        assert_eq!(frame.mesh.triangles(), outcome.stats.triangles);
        assert_eq!(frame.mesh.vertices.len(), 3 * outcome.stats.triangles);
        // And so does the stress layer beside it, coloured per vertex of the
        // same mesh: culling removes whole components, so a colour can never
        // end up on the wrong vertex.
        if let Some(stress) = frame.stress {
            assert_eq!(stress.triangles(), outcome.stats.triangles);
            assert_eq!(stress.vertices.len(), frame.mesh.vertices.len());
        }
        // What was written is what was shown.
        let (_, written) = crate::mesh::stl::read(&problem.output.stl_path).expect("the STL");
        assert_eq!(written.len(), outcome.stats.triangles);
        std::fs::remove_file(&problem.output.stl_path).ok();
    }

    /// A trim is reported in two places at once, and they are the same lines:
    /// the outcome the console prints from, and the link the editor's panel
    /// reads. Neither may be the only one that knows.
    ///
    /// What the pass decided about this fixture is deliberately not asserted -
    /// that is [`crate::trim`]'s and the pipeline's business. What is asserted
    /// is that whatever it decided reached both.
    #[test]
    fn a_trim_reports_itself_to_the_window_and_to_the_outcome_alike() {
        let directory = std::env::temp_dir();
        let text = FRAGMENTED.replace(
            "stl_path = \"growforge_fragmented.stl\"",
            "stl_path = \"growforge_fragmented.stl\"\ntrim = \"stress\"",
        );
        let config = Config::parse(&text).expect("parse");
        let mut problem = Problem::build(&config, &directory).expect("build");
        problem.output.stl_path = directory.join("growforge_viewer_trimmed.stl");

        let link = ViewLink::new();
        let outcome = run_worker(
            &problem,
            &crate::report::SilentReporter,
            &link,
            &keeps_nothing,
        )
        .expect("the run must finish")
        .expect("nothing asked it to stop");

        let notes = outcome.trim.as_ref().expect("a trim report").notes();
        assert!(!notes.is_empty(), "the pass ran and said nothing");
        assert_eq!(
            link.progress().trim,
            notes,
            "the panel and the console were told different things"
        );
        // The engine's own note is untouched: why a run ended and what the trim
        // did are two lines, not one displacing the other.
        assert!(link.progress().note.is_none() || !link.progress().trim.is_empty());
        std::fs::remove_file(&problem.output.stl_path).ok();
    }

    /// The default is off, and off is silent: nothing on the outcome, nothing
    /// on the link, and a pipeline that analysed exactly once.
    #[test]
    fn a_run_that_was_not_asked_to_trim_says_nothing_about_it() {
        let directory = std::env::temp_dir();
        let config = Config::parse(FRAGMENTED).expect("parse");
        let mut problem = Problem::build(&config, &directory).expect("build");
        problem.output.stl_path = directory.join("growforge_viewer_untrimmed.stl");
        assert_eq!(problem.output.trim, crate::config::TrimPolicy::Off);

        let link = ViewLink::new();
        let outcome = run_worker(
            &problem,
            &crate::report::SilentReporter,
            &link,
            &keeps_nothing,
        )
        .expect("the run must finish")
        .expect("nothing asked it to stop");
        assert!(outcome.trim.is_none());
        assert!(link.progress().trim.is_empty());
        // The clamp is a separate line and is on by default, so silence about
        // the trim may not be silence about it.
        assert!(outcome.clamp.is_some());
        assert!(!link.progress().clamp.is_empty());
        std::fs::remove_file(&problem.output.stl_path).ok();
    }

    /// The boundary clamp is reported in the same two places the trim is, and
    /// they are the same lines. Its default is the other way round - it runs
    /// unless a configuration says otherwise - so both directions are checked.
    #[test]
    fn the_boundary_clamp_reports_itself_to_the_window_and_to_the_outcome_alike() {
        let directory = std::env::temp_dir();
        let config = Config::parse(FRAGMENTED).expect("parse");
        let mut problem = Problem::build(&config, &directory).expect("build");
        problem.output.stl_path = directory.join("growforge_viewer_clamped.stl");
        assert_eq!(
            problem.output.boundaries,
            crate::config::BoundaryFidelity::Exact
        );

        let link = ViewLink::new();
        let outcome = run_worker(
            &problem,
            &crate::report::SilentReporter,
            &link,
            &keeps_nothing,
        )
        .expect("the run must finish")
        .expect("nothing asked it to stop");
        let notes = outcome
            .clamp
            .expect("a clamp report")
            .notes(problem.is_solid(), problem.is_flushing());
        assert!(!notes.is_empty(), "the pass ran and said nothing");
        assert_eq!(
            link.progress().clamp,
            notes,
            "the panel and the console were told different things"
        );
        std::fs::remove_file(&problem.output.stl_path).ok();

        // And under "voxel" the pass never runs, so neither is told anything.
        let mut off = Problem::build(&config, &directory).expect("build");
        off.output.boundaries = crate::config::BoundaryFidelity::Voxel;
        off.output.stl_path = directory.join("growforge_viewer_unclamped.stl");
        let link = ViewLink::new();
        let outcome = run_worker(&off, &crate::report::SilentReporter, &link, &keeps_nothing)
            .expect("the run must finish")
            .expect("nothing asked it to stop");
        assert!(outcome.clamp.is_none());
        assert!(link.progress().clamp.is_empty());
        std::fs::remove_file(&off.output.stl_path).ok();
    }
}
