//! Running an engine behind the editor window.
//!
//! Three kinds of run go through here. A **preview** is what auto-regrow starts
//! after an edit: it optimizes and nothing else - no cavity pass, no stress
//! solve, no STL - and a newer edit cancels it in favour of a run of the newer
//! configuration. A **full run** is the pipeline `growforge run` itself
//! executes, exports included. An **stl generation** is the tail of that
//! pipeline alone, run on the field a previous run left behind, because the user
//! asked for it: see [`Worker::generate`].
//!
//! Any of them can be stopped: by the stop button, and by the window closing,
//! which in edit mode ends the session rather than detaching it. `run --view`
//! still detaches - a run asked for on the command line has a file to finish
//! writing - but a run asked for *inside* a window has nowhere to report to once
//! that window is gone, and leaving one growing invisibly is what makes two
//! editors fight over one output path.
//!
//! Each run owns its own [`ViewLink`], so stopping one is closing its channels
//! and setting its flag; nothing has to be reset for the next. Cancellation is
//! cooperative: it reaches the SIMP loop through [`Reporter::cancelled`]
//! between iterations, and the viewer's `finish` asks again at every stage
//! boundary, so a stopped run never reaches its export *of its own accord*. What
//! it leaves behind is its newest design, in [`Worker::retained`], which is what
//! the user can then ask to have exported explicitly.
//!
//! Because it is cooperative, a stopped run *ends later than it is stopped* -
//! a stage away, which for an analysis is seconds. [`RunProbe`] is what the
//! window's event loop watches across that interval: it counts every thread
//! this worker has started, from before the thread exists until its body ends,
//! panic included. See `ViewerApp::begin_teardown` for why the loop must keep
//! servicing the message queue until that count is zero.
//!
//! The panel asks a different question - what is running *now*, and what owns
//! the worker - and gets it from the current run rather than from that count:
//! a stopped run is gone from the session at once, whatever its thread is still
//! doing, because it can no longer write anything.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime};

use anyhow::{Result, anyhow};

use crate::config::Config;
use crate::constants;
use crate::engine::ReduceSummary;
use crate::problem::Problem;
use crate::report::{ConsoleReporter, IterationStats, Reporter, SilentReporter};
use crate::stress::StressOutcome;
use crate::viewer::snapshot::{Frame, Progress, RunStatus, ViewLink, ViewReporter};

/// What a run is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunKind {
    /// A fast, capped, coarsened re-run started by an edit. Writes nothing and
    /// is cancelled by the next edit.
    Preview,
    /// The real pipeline, STL included, started by the user.
    Full,
    /// The pipeline's tail alone - cavities, stress, mesh, STL - on the design a
    /// previous run left behind, started by the user. No engine runs.
    Export,
}

impl RunKind {
    /// Label the panel shows.
    pub fn label(self) -> &'static str {
        match self {
            RunKind::Preview => "preview",
            RunKind::Full => "full run",
            RunKind::Export => "stl generation",
        }
    }

    /// Whether a run of this kind writes the output file.
    ///
    /// This is what owns the worker: two writers of one path race each other, so
    /// a run that writes holds it until the file is there - or until the stop
    /// button takes it away - and nothing else may start meanwhile. A preview
    /// an edit asked for is deferred rather than dropped; see
    /// `Editor::pump`.
    pub fn writes(self) -> bool {
        match self {
            RunKind::Preview => false,
            RunKind::Full | RunKind::Export => true,
        }
    }
}

/// Whether any run thread the worker started is still alive.
///
/// One atomic load, cloneable, and answerable from anywhere. Taken fresh each
/// time the window asks, and of the session it is holding then: a session can be
/// replaced without the window going anywhere, and one of these kept from before
/// that would answer for a worker that is over.
///
/// This is deliberately *not* [`Worker::is_running`]. That one is the session's
/// question: what the panel shows, what the stop button ends, and what a
/// stopped run stops being immediately. This one is the thread's: a stopped run
/// only reaches its next cancellation checkpoint some time later, and the
/// window's event loop has to keep servicing its message queue for exactly as
/// long as that takes - a preview, a full run and a run winding down after a
/// stop all count.
#[derive(Debug, Clone, Default)]
pub struct RunProbe {
    live: Arc<AtomicUsize>,
}

impl RunProbe {
    /// True while at least one run thread has not yet ended.
    pub fn is_running(&self) -> bool {
        self.live.load(Ordering::Relaxed) > 0
    }

    /// Count one run's thread in, and hand back the guard that ends the run:
    /// the count, and `done` with it, come back when the guard is dropped.
    fn enter(&self, done: Arc<AtomicBool>) -> RunGuard {
        self.live.fetch_add(1, Ordering::Relaxed);
        RunGuard {
            live: Arc::clone(&self.live),
            done,
        }
    }
}

/// Ends one run when its thread's body ends.
///
/// A guard rather than stores at the foot of the closure, because a run that
/// panics has ended just as much as one that returned: the window's loop must
/// be free to stop, and the panel free to start another. Storing after the run
/// instead left an unwinding thread's run "in flight" for the rest of the
/// session - the panel showing a run that was over and silently refusing to
/// start the next. The unwind itself is untouched and still propagates out of
/// the thread.
struct RunGuard {
    live: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        // The panel's answer first and the window's last, in program order.
        // Both stores are `Relaxed`, so this order is *not* what another thread
        // is guaranteed to observe them in: what makes the pair safe is that
        // nothing reads them across a race. The window only stops pumping when
        // the count reaches zero, and everything that then reads `done` -
        // `Worker::join`, and the panel's queries after it - happens after a
        // thread join, which is a full barrier and publishes both stores.
        self.done.store(true, Ordering::Relaxed);
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The output file a full run wrote, and when it turned out to have been
/// written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputWrite {
    /// Absolute path the run resolved `[output] stl_path` to.
    pub path: PathBuf,
    /// The modification time the file had once the write was done, read back
    /// from the filesystem rather than taken from the clock: it is later
    /// compared against an mtime, and only the filesystem's own timeline makes
    /// that comparison mean anything.
    pub modified: SystemTime,
}

/// A design a run produced, with the problem it belongs to.
///
/// The pair is what makes an on-demand export safe. The grid, the material, the
/// island and cavity policies and the output path all come from the problem, so
/// exporting a kept field against a problem rebuilt from the configuration *as
/// it is now* would export it against whatever the user has edited since - a
/// different grid, and a corrupt file. The two travel together and are replaced
/// together; nothing can hand out one of them beside the other.
///
/// Handed out behind an [`Arc`], so taking it costs an atomic bump and the run
/// that produced it may go on replacing it.
#[derive(Debug)]
pub struct Retained {
    problem: Arc<Problem>,
    densities: Vec<f64>,
    reduce: Option<ReduceSummary>,
}

impl Retained {
    /// The problem the field was produced against, and the one an export of it
    /// must use.
    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    /// Physical density of every cell, in grid element order.
    pub fn densities(&self) -> &[f64] {
        &self.densities
    }

    /// The `[optimization.reduce]` schedule that chose this design, when a
    /// schedule did.
    ///
    /// Set only where the engine hands its finished field over: a design kept
    /// mid-run is one stage of a schedule that has not finished, and there is no
    /// record to be had of it yet. `None` on every run without the table, and on
    /// a run that was stopped - which is what the engine records for one.
    pub fn reduce(&self) -> Option<&ReduceSummary> {
        self.reduce.as_ref()
    }
}

/// Where one run puts its newest design, and the problem that design belongs to.
///
/// One of these per run, holding that run's own problem and writing into the
/// worker's single slot. The slot therefore holds the newest field of the newest
/// run to report one, always beside the problem it came from.
struct Retention {
    problem: Arc<Problem>,
    slot: Arc<Mutex<Option<Arc<Retained>>>>,
}

impl Retention {
    /// The problem this run is of.
    fn problem(&self) -> &Problem {
        &self.problem
    }

    /// Keep `densities` as the newest design of this run, unless this run has
    /// been cancelled.
    ///
    /// Called from two places, and they are the same statement: every iteration
    /// the engine reports one, and the moment the engine hands its finished field
    /// over. A run whose engine reports iterations therefore ends by keeping the
    /// field it settled on over the last snapshot of it - the same field, or the
    /// one it converged to - and a run whose engine reports none, the solid one
    /// above all, keeps its only one there.
    ///
    /// `reduce` is the schedule that chose the field, which only the second of
    /// those two callers has: a design kept between iterations belongs to a
    /// stage the schedule has not finished, and the record of a schedule is
    /// written when it ends.
    ///
    /// The cancellation is checked *under the slot's own lock*, which is what
    /// makes it mean anything. A run is superseded by cancelling it and then
    /// clearing the slot for its replacement, both from the window's thread; the
    /// cancelled run may already be part way through an iteration, and a check
    /// made before taking the lock could pass and then write behind the clear -
    /// leaving a design of the run that was replaced, and of the configuration it
    /// was of, in the slot the new run is repopulating. Inside the lock the
    /// window's store cannot be missed: it precedes an acquisition of this very
    /// mutex, so it is published to whoever acquires it next.
    ///
    /// The copy is made outside the lock. The panel reads the slot every frame
    /// and a field is megabytes; what has to be atomic is the decision and the
    /// write, not the memcpy. Same shape as [`crate::viewer::snapshot::LatestSlot::push`].
    fn keep(&self, densities: &[f64], reduce: Option<&ReduceSummary>, cancel: &AtomicBool) {
        let kept = Arc::new(Retained {
            problem: Arc::clone(&self.problem),
            densities: densities.to_vec(),
            reduce: reduce.cloned(),
        });
        let mut slot = lock(&self.slot);
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        *slot = Some(kept);
    }
}

/// Voxel size a fast preview uses for a grid of `cells` cells at `voxel_mm`.
///
/// A grid inside the budget is left exactly as configured, so a small problem
/// previews at its real resolution. A larger one is coarsened by the cube root
/// of the excess, which is the factor that brings the cell count back to the
/// budget: cells scale with the cube of the voxel size.
pub fn preview_voxel_size(cells: usize, voxel_mm: f64, budget: usize) -> f64 {
    if cells <= budget || budget == 0 || !voxel_mm.is_finite() || voxel_mm <= 0.0 {
        return voxel_mm;
    }
    voxel_mm * (cells as f64 / budget as f64).cbrt()
}

/// The configuration a fast preview of `config` runs.
///
/// Growth grows a whole design in milliseconds and the solid engine fills a
/// field in one pass, so both preview at the real configuration and the only
/// difference is that nothing is exported. Coarsening the solid engine's grid
/// would be worse than pointless besides: the surface it produces *is* the
/// domain, so a preview at a different voxel size would be a preview of a
/// different part. SIMP is capped at
/// [`constants::VIEW_EDIT_PREVIEW_MAX_ITERATIONS`] iterations on a grid
/// coarsened to [`constants::VIEW_EDIT_PREVIEW_CELL_BUDGET`] cells. The solver
/// backend is left alone: the whole `[solver]` table is carried over untouched,
/// so a configuration that asked for the GPU still gets it and one that asked
/// for nothing gets the same default a full run would - including the fallback,
/// which is the machine's business rather than the preview's.
pub fn preview_config(config: &Config, problem: &Problem) -> Config {
    let mut preview = config.clone();
    if config.is_growth() || config.is_solid() {
        return preview;
    }
    let cap = constants::VIEW_EDIT_PREVIEW_MAX_ITERATIONS;
    preview.optimization.max_iterations =
        Some(config.optimization.max_iterations.unwrap_or(cap).min(cap));
    let voxel = preview_voxel_size(
        problem.grid.n_cells(),
        problem.grid.h,
        constants::VIEW_EDIT_PREVIEW_CELL_BUDGET,
    );
    preview.resolution.voxel_size_mm = Some(voxel);
    preview.resolution.target_cells = None;
    preview
}

/// Forwards progress to the window, keeps the design behind it, and answers the
/// engine's cancellation question.
struct EditReporter<'a> {
    inner: ViewReporter<'a>,
    cancel: &'a AtomicBool,
    retain: &'a Retention,
}

impl Reporter for EditReporter<'_> {
    fn iteration(&self, stats: &IterationStats) {
        self.inner.iteration(stats);
    }

    fn note(&self, message: &str) {
        self.inner.note(message);
    }

    fn densities(&self, stats: &IterationStats, densities: &[f64]) {
        self.inner.densities(stats, densities);
        // Not once the run has been asked to stop. The design the *window* is
        // left showing is the last one it was sent before the stop - the
        // snapshot channel is closed with the same click - and what an export
        // uses has to be that same design, not an iteration that finished
        // afterwards and was never seen.
        //
        // An economy rather than the rule: this saves the copy below for a run
        // that is already over, and [`Retention::keep`] is where the answer is
        // authoritative, because only there is it inside the slot's own lock.
        if self.cancelled() {
            return;
        }
        self.retain.keep(densities, None, self.cancel);
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// One run in flight.
struct Run {
    kind: RunKind,
    link: Arc<ViewLink>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    /// Set when the thread's body has ended, by the guard that ends it - so a
    /// run that unwound has ended by this measure too. Each run has its own, so
    /// a thread that outlives its run can only ever end that run.
    done: Arc<AtomicBool>,
}

/// The editor's background runs.
#[derive(Default)]
pub struct Worker {
    current: Option<Run>,
    /// How many run threads are alive, whatever kind they are and whether or
    /// not they have been stopped. Watched by the window's event loop.
    threads: RunProbe,
    /// What the last run to write wrote, set by the thread that wrote it.
    output_write: Arc<Mutex<Option<OutputWrite>>>,
    /// The newest design any run has reported, with the problem it belongs to.
    /// Written by the run's own thread through its [`Retention`], and cleared
    /// when the next run starts.
    retained: Arc<Mutex<Option<Arc<Retained>>>>,
    /// Set when the last run failed to start at all.
    startup_error: Option<String>,
    /// Threads of runs that were stopped and are winding down. They are joined
    /// with everything else at the end, so nothing is still touching a problem
    /// when the process leaves.
    retired: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Worker")
            .field("running", &self.kind())
            .field("full_active", &self.is_running_full())
            .field("writing", &self.is_writing())
            .field("kept", &lock(&self.retained).is_some())
            .field("threads_alive", &self.threads.is_running())
            .finish()
    }
}

impl Worker {
    /// A worker with nothing running.
    pub fn new() -> Worker {
        Worker::default()
    }

    /// What is running, if anything.
    pub fn kind(&self) -> Option<RunKind> {
        self.current.as_ref().map(|run| run.kind)
    }

    /// True while a run is going.
    ///
    /// The session's answer, not the threads': a stopped run is no longer
    /// running by this measure the instant it is stopped, while its thread is
    /// still on its way to the next checkpoint. [`Worker::probe`] is the one
    /// that covers that interval.
    pub fn is_running(&self) -> bool {
        self.current
            .as_ref()
            .is_some_and(|run| !run.done.load(Ordering::Relaxed))
    }

    /// A handle on "is any run thread still alive", for the window's event
    /// loop. Answers for every run of this worker, the ones started after it was
    /// taken included.
    pub fn probe(&self) -> RunProbe {
        self.threads.clone()
    }

    /// True while a full run owns the worker: the panel offers "stop" and
    /// refuses "run full", and nothing else may take the worker from it.
    ///
    /// Derived from the current run rather than latched in a flag of its own,
    /// which makes it the session's answer exactly as [`Worker::is_running`] is
    /// and keeps the two from ever disagreeing. In particular a *stopped* full
    /// run hands the worker back at once: it can no longer reach its export, so
    /// it owns neither the output file nor the button. Its thread may still be
    /// winding down to the next checkpoint - that is [`Worker::probe`]'s
    /// question, and the window's, not the panel's.
    pub fn is_running_full(&self) -> bool {
        self.kind() == Some(RunKind::Full) && self.is_running()
    }

    /// True while a run that writes the output file owns the worker: the full
    /// pipeline, or an on-demand stl generation.
    ///
    /// What gates every button that would start another writer, and what defers
    /// the preview an edit asks for. The same session-level answer
    /// [`Worker::is_running_full`] gives, over both kinds that write; see
    /// [`RunKind::writes`].
    pub fn is_writing(&self) -> bool {
        self.kind().is_some_and(RunKind::writes) && self.is_running()
    }

    /// The newest design any run of this worker has reported, with the problem
    /// it was produced against.
    ///
    /// This is what the window is showing: it is fed by the very per-iteration
    /// callback the preview mesher is fed by, and again when the engine hands its
    /// finished field over, and cleared when a run starts - so there is a design
    /// here exactly when one has been computed for the configuration as it was
    /// then. Whatever ended the run that produced it - convergence, the iteration
    /// cap, the stop button, an engine with no iterations to end - is not
    /// recorded and does not matter: a field is a design either way, and
    /// exporting it is the user's call.
    ///
    /// Always the field *as the engine produced it*, never one the deliverable
    /// passes have been over: [`Worker::generate`] applies those passes to
    /// whatever is here, so a field that had already been through them would come
    /// out of a generation having had them applied twice.
    ///
    /// What it costs is one copy of the field per reported iteration, and one
    /// more where the run ends - the snapshot the preview mesher is sent is a
    /// copy of its own, and these are deliberately not shared: a few megabytes of
    /// memcpy beside a finite element solve is not worth threading shared
    /// ownership through the reporter for - and one field of memory held between
    /// runs.
    pub fn retained(&self) -> Option<Arc<Retained>> {
        lock(&self.retained).clone()
    }

    /// Whether an on-demand export may be started: there is a design to export,
    /// and nothing that writes the output file is running.
    ///
    /// The panel's enabled-state predicate. Deliberately says nothing about the
    /// *configuration* being runnable: the pair being exported carries its own
    /// problem, so a design stays exportable through any edit, however broken.
    pub fn can_generate(&self) -> bool {
        !self.is_writing() && lock(&self.retained).is_some()
    }

    /// Progress of the current run, for the stats panel.
    pub fn progress(&self) -> Option<Progress> {
        self.current.as_ref().map(|run| run.link.progress())
    }

    /// Why the last run could not be started, if it could not.
    pub fn startup_error(&self) -> Option<&str> {
        self.startup_error.as_deref()
    }

    /// Whether a file at `path` whose mtime is `modified` is the one a full run
    /// of this worker left behind.
    ///
    /// The recorded time is the file's own, read back after the write, so an
    /// untouched file compares equal. The tolerance is for the filesystems that
    /// do not resolve a timestamp as finely as the write records it, or that
    /// settle it a moment after the handle is closed; it is the width of the
    /// window in which somebody else's write is mistaken for ours, which is why
    /// [`constants::VIEW_EDIT_OUTPUT_MTIME_TOLERANCE_S`] is small.
    pub fn wrote_output(&self, path: &Path, modified: SystemTime) -> bool {
        let tolerance = Duration::from_secs_f64(constants::VIEW_EDIT_OUTPUT_MTIME_TOLERANCE_S);
        lock(&self.output_write).as_ref().is_some_and(|write| {
            write.path == path
                && modified
                    <= write
                        .modified
                        .checked_add(tolerance)
                        .unwrap_or(write.modified)
        })
    }

    /// Take the newest rendered surface the current run has produced.
    pub fn take_frame(&self) -> Option<Frame> {
        self.current
            .as_ref()
            .and_then(|run| run.link.frames.try_take())
    }

    /// Ask an auto-regrow preview to stop, because a newer edit has made it
    /// obsolete. A full run is left alone: it is producing a file the user
    /// asked for, and only the stop button takes that away.
    pub fn cancel_preview(&mut self) {
        if self
            .current
            .as_ref()
            .is_some_and(|run| run.kind == RunKind::Preview)
        {
            self.stop();
        }
    }

    /// Stop whatever is running, whatever kind it is.
    ///
    /// The flag reaches the engine through [`Reporter::cancelled`] between its
    /// iterations, and the viewer's `finish` asks again at each stage
    /// boundary, so a stopped run never reaches its export: it writes no file,
    /// not a partial one and not by accident. Idempotent, and a no-op when
    /// nothing is running.
    ///
    /// What it does *not* do is throw the design away. The field the run had
    /// reached stays in [`Worker::retained`], because that is what is on screen
    /// and the user may still want its deliverables; [`Worker::generate`] is how
    /// they ask, and only that asking writes anything.
    pub fn stop(&mut self) {
        let Some(mut run) = self.current.take() else {
            return;
        };
        run.cancel.store(true, Ordering::Relaxed);
        run.link.detach();
        // The thread stops at its next boundary. Its handle is kept so that
        // `join` can still wait for it: forgetting it would leave the process
        // free to exit part way through a stage.
        if let Some(handle) = run.handle.take() {
            self.retired.push(handle);
        }
    }

    /// Collect every run this worker started, the stopped ones included.
    ///
    /// Called once the event loop has stopped - which, because the loop watches
    /// [`Worker::probe`], is after those threads have ended. This is what makes
    /// that certain rather than likely, and it is the last thing the process
    /// does.
    pub fn join(&mut self) {
        if let Some(run) = &mut self.current
            && let Some(handle) = run.handle.take()
        {
            let _ = handle.join();
        }
        self.current = None;
        for handle in self.retired.drain(..) {
            let _ = handle.join();
        }
    }

    /// Start a run of `config`, replacing whatever was running.
    ///
    /// Returns the error the problem build produced, if the configuration
    /// cannot be turned into one; the caller shows it and nothing starts.
    ///
    /// [`RunKind::Export`] is not one of these. It runs on a design a run
    /// already produced rather than on a configuration, and [`Worker::generate`]
    /// is where it starts; asked for here, it is refused with that reason rather
    /// than reinterpreted as something that would run an engine.
    pub fn start(&mut self, config: &Config, directory: &Path, kind: RunKind) -> Result<()> {
        // Refused before anything is touched, so a call that asked for the wrong
        // thing does not cancel a preview on its way to being told so.
        if kind == RunKind::Export {
            let error = anyhow!(
                "an stl generation runs on the design already on screen rather than on a \
                 configuration; it is started from the panel's \"generate stl\" button"
            );
            self.startup_error = Some(format!("{error:#}"));
            return Err(error);
        }
        self.cancel_preview();
        if self.is_writing() {
            // A run that writes owns the worker until its file is there - or
            // until the stop button takes it away.
            return Ok(());
        }
        self.startup_error = None;
        let problem = match Problem::build(config, directory) {
            Ok(problem) => problem,
            Err(error) => {
                let message = format!("{error:#}");
                self.startup_error = Some(message.clone());
                return Err(error);
            }
        };
        let retention = Retention {
            problem: Arc::new(problem),
            slot: Arc::clone(&self.retained),
        };
        // Whatever design was kept belonged to the run that produced it and to
        // the configuration that run was of. Until this one has reported one of
        // its own there is nothing on screen, and so nothing to export.
        //
        // This clear comes *after* whatever it supersedes has been cancelled -
        // by `cancel_preview` above, or by the stop that ended a run before this
        // call - and that order is what [`Retention::keep`] relies on to refuse
        // an iteration the superseded run still had in flight. Nothing may be
        // reordered between the two.
        *lock(&self.retained) = None;
        self.launch(kind, move |link, cancel, output_write| {
            run(
                &retention,
                link,
                cancel,
                kind,
                output_write,
                &mut std::io::stdout(),
            );
        });
        Ok(())
    }

    /// Export the design that was kept: the cavity pass, the stress solve, the
    /// mesh and the STL, written to the kept problem's own output path.
    ///
    /// This is the one path on which a *stopped* run leaves a file behind, and
    /// what leaves it is this call rather than the stop: the automatic behaviour
    /// of every run is unchanged, and one asked to stop still writes nothing on
    /// its own. What the user gets back is the deliverable set of the design they
    /// were looking at when they stopped it - which, for a run they have watched
    /// for hours, is the whole point.
    ///
    /// Returns false when there is nothing to export or something that writes is
    /// already running. Both are states the panel renders the button disabled
    /// in, so this is the guard behind that and never the path taken.
    pub fn generate(&mut self) -> bool {
        if !self.can_generate() {
            return false;
        }
        let Some(retained) = self.retained() else {
            return false;
        };
        // An explicit request beats a preview an edit started: the preview is
        // one of a stream and its field is already kept, this is what the user
        // asked for by name.
        self.cancel_preview();
        self.startup_error = None;
        self.launch(RunKind::Export, move |link, cancel, output_write| {
            export_retained(
                &retained,
                link,
                cancel,
                output_write,
                &mut std::io::stdout(),
            );
        });
        true
    }

    /// Make `body` the current run of `kind`, on a thread this worker counts.
    ///
    /// Everything that makes a run stoppable, watchable and joinable is
    /// assembled here and nowhere else, so a new kind of run cannot be given one
    /// of them and forget another. The probe above all: a thread the window does
    /// not know about is one it can close the window on.
    fn launch<F>(&mut self, kind: RunKind, body: F)
    where
        F: FnOnce(&ViewLink, &AtomicBool, &Mutex<Option<OutputWrite>>) + Send + 'static,
    {
        let link = Arc::new(ViewLink::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));

        // Entered here, before the thread exists, so the window can never
        // observe a gap between starting a run and its thread being alive; and
        // everything that says the run is over is this one guard's to do.
        let alive = self.threads.enter(Arc::clone(&done));
        let handle = {
            let link = Arc::clone(&link);
            let cancel = Arc::clone(&cancel);
            let output_write = Arc::clone(&self.output_write);
            std::thread::spawn(move || {
                // Bound, not dropped: the guard is what ends the run, and it
                // has to do so after the body below and on the panicking path
                // as well.
                let _alive = alive;
                body(&link, &cancel, &output_write);
            })
        };
        self.current = Some(Run {
            kind,
            link,
            cancel,
            handle: Some(handle),
            done,
        });
    }
}

/// A poisoned lock is not a reason to take the editor down: the record behind
/// it is one value, and its invariants do not depend on where a panic happened.
fn lock<T>(slot: &Mutex<T>) -> MutexGuard<'_, T> {
    slot.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The body of one run of a configuration, on its own thread.
///
/// Which of the two it is follows from whether the run writes: a preview shows a
/// design and stops there, and everything that writes runs the whole pipeline.
/// The third kind is not started from a configuration at all - see
/// [`export_retained`] - so `kind` here is only ever one of these two.
fn run(
    retention: &Retention,
    link: &ViewLink,
    cancel: &AtomicBool,
    kind: RunKind,
    output_write: &Mutex<Option<OutputWrite>>,
    out: &mut dyn Write,
) {
    if kind.writes() {
        full(retention, link, cancel, output_write, out);
    } else {
        preview(retention, link, cancel);
    }
}

/// The pipeline `growforge run` executes, stoppable at its stage boundaries.
///
/// `out` is the session's console: the stdout of the terminal `growforge edit`
/// was launched from, and a buffer under test. Failures keep going to stderr
/// directly, because that is what they are.
fn full(
    retention: &Retention,
    link: &ViewLink,
    cancel: &AtomicBool,
    output_write: &Mutex<Option<OutputWrite>>,
    out: &mut dyn Write,
) {
    let problem = retention.problem();
    let console = ConsoleReporter::new(false);
    let reporter = EditReporter {
        inner: ViewReporter::new(&console, link),
        cancel,
        retain: retention,
    };
    // The engine's own field, kept the moment the engine hands it over - see
    // [`crate::viewer::run_worker`], which is where the ordering that makes it
    // exportable lives. Not the reporter's business: this run may report no
    // iteration at all, and a run that ends is a design either way.
    let completed = |densities: &[f64], reduce: Option<&ReduceSummary>| {
        retention.keep(densities, reduce, cancel)
    };
    match crate::viewer::run_worker(problem, &reporter, link, &completed) {
        Ok(Some(outcome)) => {
            // Recorded here and nowhere else: this is the one moment at which
            // the file is known to have been written, which is what lets the
            // editor tell its own output from one something else overwrote.
            record_write(output_write, &outcome.stl_path);
            let _ = writeln!(out, "editor wrote  {}", outcome.stl_path.display());
            echo_stress(out, &outcome.stress, outcome.islands.bodies.len());
        }
        // Asked to stop: nothing was analysed past the point it was asked, and
        // nothing at all was written.
        Ok(None) => {
            let _ = writeln!(out, "editor run    stopped; no file was written");
        }
        // The window already carries the reason; the console gets it too,
        // because a run started from the editor is still a run.
        //
        // Deliberately *not* the "error: " prefix `main` uses. That prefix means
        // the process is failing, and this one is not: an editor session
        // survives a run that fails, the panel says so, and the window is still
        // there to fix it in. Printing the two the same way is what made a
        // failed run read as a dead program.
        Err(error) => eprintln!(
            "editor run    failed and wrote nothing: {error:#}\n\
             editor run    the session is unaffected; change the configuration and run again"
        ),
    }
}

/// The pipeline's tail alone, on a design that was kept: the cavity pass, the
/// stress solve, the mesh and the STL.
///
/// Everything here is [`crate::viewer::finish`]'s, which is the same call a full
/// run's tail is, so the file, the final frame and the stress layer are the ones
/// a run that reached its own export would have produced from this field. The
/// problem is the kept one, never a rebuilt one: the configuration may have been
/// edited into a different grid since, and a field is only ever a design of the
/// problem it was computed on.
///
/// `out` is the session's console, as it is for [`full`], and what is said there
/// is the same: where the file went, and what the part came out at.
fn export_retained(
    retained: &Retained,
    link: &ViewLink,
    cancel: &AtomicBool,
    output_write: &Mutex<Option<OutputWrite>>,
    out: &mut dyn Write,
) {
    // Declared first so it drops last: whatever unwinds below, the window is
    // told before this function's frame is gone.
    let _fail_on_panic = crate::viewer::FailOnPanic(link);
    let problem = retained.problem();
    let path = problem.output.stl_path.clone();
    // A copy, because the cavity policy resolves the field in place: the kept
    // design stays the one the run produced, so generating twice generates the
    // same thing twice rather than compounding.
    let mut densities = retained.densities().to_vec();
    let stop = || cancel.load(Ordering::Relaxed);
    // A copy of the schedule's record for the same reason the field is copied:
    // the finishing passes measure what the part in the file came out at and
    // write it onto the record, and the design that was kept has to stay the one
    // the run produced however many times it is generated from.
    let mut reduce = retained.reduce().cloned();
    match crate::viewer::finish(problem, &mut densities, reduce.as_mut(), link, &stop) {
        Ok(Some(finished)) => {
            // Recorded exactly where a full run records it, so the editor's own
            // output stays its own however it was written.
            record_write(output_write, &path);
            let _ = writeln!(out, "editor wrote  {}", path.display());
            echo_stress(out, &finished.stress, finished.islands.bodies.len());
        }
        // Asked to stop before the write, which is where every stop lands.
        Ok(None) => {
            let _ = writeln!(out, "editor stl    stopped; no file was written");
        }
        Err(error) => eprintln!(
            "editor stl    failed and wrote nothing: {error:#}\n\
             editor stl    the session is unaffected; the design is still there to try again from"
        ),
    }
}

/// Say what the part just written came out at, on the session's console.
///
/// The lines the panel draws, from the same formatter, beside the line that says
/// where the file went: a user who has the terminal in front of them reads the
/// safety factor there, and one who has the window in front of them reads the
/// same number in it. A run whose stress solve produced no report says nothing
/// here, exactly as the panel shows no block.
///
/// `bodies` is the island report's count for the surface that was just written,
/// so the warning a multi-body export earns is echoed with the number it
/// qualifies rather than left to the panel alone.
fn echo_stress(out: &mut dyn Write, stress: &StressOutcome, bodies: usize) {
    let Some(report) = stress.report() else {
        return;
    };
    for line in report.summary(bodies).lines() {
        let _ = writeln!(out, "editor stress {line}");
    }
}

/// Note the file a run has just written, with the mtime it turned out to have.
///
/// A file that cannot be stat'ed records nothing, which leaves the editor
/// warning that it was written elsewhere: the harmless way round, since the
/// warning is advice and the alternative is silence about a real overwrite.
fn record_write(slot: &Mutex<Option<OutputWrite>>, path: &Path) {
    let Ok(modified) = std::fs::metadata(path).and_then(|data| data.modified()) else {
        return;
    };
    *lock(slot) = Some(OutputWrite {
        path: path.to_path_buf(),
        modified,
    });
}

/// Optimize and show the result. Nothing is analysed, nothing is written.
///
/// The design is kept like any other run's, which is what makes "generate stl"
/// export what is on screen: a preview is what the window shows for most of an
/// editing session.
fn preview(retention: &Retention, link: &ViewLink, cancel: &AtomicBool) {
    let problem = retention.problem();
    let console = SilentReporter;
    let reporter = EditReporter {
        inner: ViewReporter::new(&console, link),
        cancel,
        retain: retention,
    };
    let field = std::thread::scope(|threads| {
        // The mesher stops when this scope ends, so no preview surface can land
        // after the one below.
        let mesher = threads.spawn(|| crate::viewer::preview_loop(problem, link));
        let field = crate::optimize(problem, &reporter);
        link.densities.close();
        let _ = mesher.join();
        field
    });
    match field {
        Ok(field) => {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            // The field the engine ended on, kept as any reported iteration is.
            // A preview never goes through [`crate::viewer::finish`], so this is
            // the same field a generation would be handed either way; what it
            // adds is the run whose engine reported no iteration to keep one of.
            retention.keep(&field.densities, field.reduce.as_ref(), cancel);
            let surface = crate::viewer::scene::preview_surface(
                &problem.grid,
                &field.densities,
                problem.output.iso_level,
            );
            if let Ok(surface) = surface {
                link.frames.push(Frame {
                    kind: crate::viewer::snapshot::FrameKind::Preview {
                        iteration: field.iterations,
                    },
                    mesh: crate::viewer::scene::LayerMesh::from_mesh(
                        &surface,
                        crate::viewer::scene::Layer::Density.info().color,
                        crate::viewer::scene::Shading::Smooth,
                    ),
                    stress: None,
                });
            }
            // The status is left on `Optimizing`: a preview never analyses,
            // never exports and never writes a file, so none of the stages
            // `RunStatus` describes has happened. That a preview has finished
            // is what the worker's own `is_running` says.
        }
        Err(error) => link.set_status(RunStatus::Failed(format!("{error:#}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::editor::tests::{fixture, growth_fixture, solid_fixture, write_temp};
    use std::time::{Duration, Instant};

    fn problem_of(config: &Config) -> Problem {
        Problem::build(config, &std::env::temp_dir()).expect("build")
    }

    /// A cantilever small enough to optimize, analyse and mesh inside a test, and
    /// resolved enough to have a real surface and a stress report to go with it:
    /// the problem the library's own analysis tests use. On the cpu, so what it
    /// produces is this structure's rather than this machine's.
    const CANTILEVER: &str = r#"
[project]
name = "kept"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0
max_iterations = 15

[output]
stl_path = "kept.stl"

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

    /// Volume of the cantilever's design space, which is what an exported volume
    /// has to come in under.
    const CANTILEVER_DOMAIN_MM3: f64 = 32.0 * 16.0 * 16.0;

    /// Poll until `ready`, taking every frame that arrives, and hand back the
    /// newest one. The channel is drained afterwards as well, because the frame
    /// that matters is often the one pushed as the run ended.
    fn drain_until(
        worker: &Worker,
        seconds: u64,
        ready: impl Fn(&Worker, Option<&Frame>) -> bool,
    ) -> Option<Frame> {
        let deadline = Instant::now() + Duration::from_secs(seconds);
        let mut frame = None;
        while Instant::now() < deadline {
            if let Some(next) = worker.take_frame() {
                frame = Some(next);
            }
            if ready(worker, frame.as_ref()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        while let Some(next) = worker.take_frame() {
            frame = Some(next);
        }
        frame
    }

    #[test]
    fn a_preview_grid_is_coarsened_only_when_it_is_over_budget() {
        // Inside the budget nothing moves, whatever the numbers are.
        assert_eq!(preview_voxel_size(1_000, 2.0, 40_000), 2.0);
        assert_eq!(preview_voxel_size(40_000, 2.0, 40_000), 2.0);
        // Eight times the budget is twice the voxel size, because cells scale
        // with the cube of it.
        let coarse = preview_voxel_size(320_000, 2.0, 40_000);
        assert!((coarse - 4.0).abs() < 1e-9, "coarsened to {coarse}");
        // And the coarsened grid really is about the budget.
        let cells = 320_000.0 * (2.0f64 / coarse).powi(3);
        assert!(
            (cells - 40_000.0).abs() < 1.0,
            "{cells} cells after coarsening"
        );
        // Degenerate requests are left alone rather than producing a zero.
        assert_eq!(preview_voxel_size(100, 2.0, 0), 2.0);
        assert_eq!(preview_voxel_size(100, 0.0, 10), 0.0);
    }

    #[test]
    fn a_simp_preview_is_capped_and_coarsened_and_a_growth_preview_is_not() {
        let config = Config::parse(fixture()).expect("parse");
        let problem = problem_of(&config);
        let preview = preview_config(&config, &problem);
        assert_eq!(
            preview.optimization.max_iterations,
            Some(
                constants::VIEW_EDIT_PREVIEW_MAX_ITERATIONS.min(
                    config
                        .optimization
                        .max_iterations
                        .unwrap_or(constants::VIEW_EDIT_PREVIEW_MAX_ITERATIONS)
                )
            )
        );
        assert!(preview.resolution.voxel_size_mm.is_some());
        assert!(preview.resolution.target_cells.is_none());
        // The preview must still be a problem growforge can build.
        let previewed = problem_of(&preview);
        assert!(previewed.grid.n_cells() <= problem.grid.n_cells());
        // Nothing else about the configuration is touched: same engine, same
        // solver backend, same output path.
        assert_eq!(
            preview.engine_name().unwrap(),
            config.engine_name().unwrap()
        );
        assert_eq!(
            preview.solver_params().unwrap().backend,
            config.solver_params().unwrap().backend
        );
        assert_eq!(preview.output.stl_path, config.output.stl_path);

        // A configuration asking for fewer iterations than the cap keeps its
        // own number rather than being raised to it.
        let mut brief = config.clone();
        brief.optimization.max_iterations = Some(3);
        assert_eq!(
            preview_config(&brief, &problem).optimization.max_iterations,
            Some(3)
        );

        let growth = Config::parse(growth_fixture()).expect("parse");
        let growth_problem = problem_of(&growth);
        let previewed = preview_config(&growth, &growth_problem);
        assert_eq!(
            previewed.resolution.voxel_size_mm, growth.resolution.voxel_size_mm,
            "the growth engine previews at the real resolution"
        );
        assert_eq!(
            previewed.optimization.max_iterations,
            growth.optimization.max_iterations
        );

        // And neither is the solid engine's, which has an extra reason: its
        // surface is the domain itself, so a coarsened preview would be a
        // preview of a different part.
        let solid = Config::parse(solid_fixture()).expect("parse");
        let solid_problem = problem_of(&solid);
        let previewed = preview_config(&solid, &solid_problem);
        assert_eq!(
            previewed.resolution.voxel_size_mm, solid.resolution.voxel_size_mm,
            "the solid engine previews at the real resolution"
        );
        assert_eq!(&previewed, &solid, "a solid preview is the configuration");
    }

    #[test]
    fn a_growth_preview_runs_in_the_background_and_hands_back_a_surface() {
        let (_dir, path) = write_temp("worker_growth", growth_fixture());
        let config = Config::parse(growth_fixture()).expect("parse");
        let mut worker = Worker::new();
        worker
            .start(
                &config,
                path.parent().expect("a directory"),
                RunKind::Preview,
            )
            .expect("start");
        assert_eq!(worker.kind(), Some(RunKind::Preview));
        assert!(!worker.is_running_full(), "a preview is not a full run");

        let deadline = Instant::now() + Duration::from_secs(60);
        let mut frame = None;
        while Instant::now() < deadline {
            if let Some(next) = worker.take_frame() {
                frame = Some(next);
                if !worker.is_running() {
                    break;
                }
            }
            if !worker.is_running() && frame.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let frame = frame.expect("the preview produced no surface");
        assert!(frame.mesh.triangles() > 0);
        assert!(frame.stress.is_none(), "a preview solves no stresses");
        worker.join();

        // And nothing was written: a preview is not an export.
        let stl = path
            .parent()
            .expect("a directory")
            .join(&config.output.stl_path);
        assert!(!stl.exists(), "a preview wrote {}", stl.display());
    }

    /// A full run started from the editor exports through the island policy and
    /// shows what it exported: the panel's triangle count, the surface in the
    /// viewport and the file on disk are one mesh, the culled one.
    #[test]
    fn a_full_editor_run_shows_and_writes_the_culled_surface() {
        let text = crate::viewer::tests::FRAGMENTED;
        let (_dir, path) = write_temp("worker_islands", text);
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(text).expect("parse");
        let mut worker = Worker::new();
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("start");

        let deadline = Instant::now() + Duration::from_secs(120);
        let mut frame = None;
        while Instant::now() < deadline {
            if let Some(next) = worker.take_frame() {
                frame = Some(next);
            }
            if !worker.is_running() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        while let Some(next) = worker.take_frame() {
            frame = Some(next);
        }
        let frame = frame.expect("the run produced no surface");
        assert_eq!(frame.kind, crate::viewer::snapshot::FrameKind::Final);
        let progress = worker.progress().expect("a run to report on");
        let RunStatus::Finished { triangles, .. } = progress.status else {
            panic!("the run did not finish: {:?}", progress.status);
        };
        assert_eq!(frame.mesh.triangles(), triangles);
        // The engine's last word travels with it, which is what the editor's
        // panel shows under the run line. A finished SIMP run always has one:
        // it converged, it stopped because the problem would not settle, or it
        // spent its budget.
        let note = progress
            .note
            .expect("the engine said nothing about the run");
        assert!(
            ["converged", "still traversing", "iteration cap"]
                .iter()
                .any(|reason| note.contains(reason)),
            "the note does not say why the run ended: {note}"
        );

        // The file holds the same surface, and it is one piece: the fragment
        // the second domain lobe minted is in neither.
        let stl = directory.join(&config.output.stl_path);
        let (_, written) = crate::mesh::stl::read(&stl).expect("the STL");
        assert_eq!(written.len(), triangles);
        worker.join();
        std::fs::remove_file(&stl).ok();
    }

    /// The retention slot's own guard: a run that has been cancelled writes no
    /// field into it, and the answer is taken inside the lock the slot is written
    /// under.
    ///
    /// This is what keeps a superseded run out of its replacement's slot. A run is
    /// superseded by cancelling it and then clearing the slot, so a cancelled run
    /// that still had an iteration in flight would otherwise be free to land a
    /// design of the old configuration behind that clear - and the panel would
    /// offer it. The race itself is not deterministically reproducible; the guard
    /// that closes it is exactly this.
    #[test]
    fn a_cancelled_run_keeps_no_field_of_its_own() {
        let config = Config::parse(fixture()).expect("parse");
        let problem = Arc::new(problem_of(&config));
        let cells = problem.grid.n_cells();
        let slot: Arc<Mutex<Option<Arc<Retained>>>> = Arc::new(Mutex::new(None));
        let retention = Retention {
            problem: Arc::clone(&problem),
            slot: Arc::clone(&slot),
        };
        let cancel = AtomicBool::new(false);

        // A live run keeps every field it reports, beside its own problem. An
        // iteration is a stage of a schedule that has not finished, so it
        // carries no record of one.
        retention.keep(&vec![0.25; cells], None, &cancel);
        let kept = lock(&slot).clone().expect("a live run kept nothing");
        assert_eq!(kept.densities().len(), cells);
        assert_eq!(kept.densities()[0], 0.25);
        assert!(kept.reduce().is_none(), "an iteration carried a schedule");

        // The field the engine hands over carries the schedule that chose it,
        // which is what an export of that design reports.
        let summary = ReduceSummary {
            method: crate::config::ReduceMethod::Continuation,
            target_safety_factor: 3.0,
            exported: crate::engine::ReduceStage {
                index: 2,
                target_fraction: 0.8,
                achieved_fraction: 0.8,
                iterations: 21,
                safety_factor: Some(4.2),
                passed: true,
                refine: false,
            },
            stages: Vec::new(),
            finished_safety_factor: None,
        };
        retention.keep(&vec![0.5; cells], Some(&summary), &cancel);
        assert_eq!(
            lock(&slot)
                .clone()
                .expect("the slot was emptied")
                .reduce()
                .cloned(),
            Some(summary),
            "the schedule did not travel with the design it chose"
        );

        // Cancelled, it keeps nothing: what is already there stays.
        cancel.store(true, Ordering::Relaxed);
        retention.keep(&vec![0.75; cells], None, &cancel);
        assert_eq!(
            lock(&slot)
                .clone()
                .expect("the slot was emptied")
                .densities()[0],
            0.5,
            "a cancelled run wrote its field over a newer one"
        );

        // And a slot its replacement has cleared stays cleared, which is the case
        // the guard exists for.
        *lock(&slot) = None;
        retention.keep(&vec![0.75; cells], None, &cancel);
        assert!(
            lock(&slot).is_none(),
            "a cancelled run repopulated a cleared slot"
        );
    }

    /// What is on screen is what is generated. Most of an editing session shows a
    /// preview rather than a finished run, and a preview keeps its design like any
    /// other run: the file "generate stl" writes is the surface in the viewport.
    #[test]
    fn a_preview_keeps_its_design_and_it_is_what_gets_generated() {
        let (_dir, path) = write_temp("worker_preview_generate", growth_fixture());
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(growth_fixture()).expect("parse");
        let stl = directory.join(&config.output.stl_path);
        let mut worker = Worker::new();
        worker
            .start(&config, &directory, RunKind::Preview)
            .expect("start");
        let frame = drain_until(&worker, 300, |worker, frame| {
            !worker.is_running() && frame.is_some()
        })
        .expect("the preview produced no surface");
        assert!(frame.mesh.triangles() > 0);
        assert!(frame.stress.is_none(), "a preview solves no stresses");
        assert!(!stl.exists(), "a preview wrote {}", stl.display());

        // It wrote nothing and kept everything, so the button is offered on it.
        let kept = worker.retained().expect("the preview kept no design");
        assert_eq!(kept.problem().output.stl_path, stl);
        assert_eq!(kept.densities().len(), kept.problem().grid.n_cells());
        assert!(worker.can_generate());

        assert!(worker.generate(), "the generation never started");
        let generated = drain_until(&worker, 300, |worker, _| !worker.is_running())
            .expect("the generation produced no surface");
        assert_eq!(
            generated.kind,
            crate::viewer::snapshot::FrameKind::Final,
            "a generation ends on the surface it wrote"
        );
        assert!(stl.exists(), "the generation wrote nothing");
        worker.join();
        std::fs::remove_file(&stl).ok();
    }

    /// "generate stl", on the design a finished run left behind: the same
    /// deliverables the run itself produced, because it is the same call. The
    /// run's own file is taken away first, so what is asserted afterwards can
    /// only be the generation's.
    #[test]
    fn a_generation_writes_the_deliverables_of_the_design_that_was_kept() {
        let (_dir, path) = write_temp("worker_generate", CANTILEVER);
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(CANTILEVER).expect("parse");
        let stl = directory.join(&config.output.stl_path);
        let mut worker = Worker::new();
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("start");
        drain_until(&worker, 300, |worker, _| !worker.is_running());
        assert!(!worker.is_running(), "the run never finished");
        assert!(stl.exists(), "the full run wrote no file");
        std::fs::remove_file(&stl).expect("take the run's own file away");

        // The design the run ended on is still there, beside the problem it was
        // computed on - which is where an export of it writes.
        let kept = worker.retained().expect("the run kept no design");
        assert_eq!(kept.densities().len(), kept.problem().grid.n_cells());
        assert!(
            kept.densities().iter().any(|&density| density > 0.5),
            "the kept field has nothing solid in it"
        );
        assert_eq!(kept.problem().output.stl_path, stl);
        assert!(worker.can_generate());

        assert!(worker.generate(), "the generation never started");
        assert_eq!(worker.kind(), Some(RunKind::Export));
        assert!(
            worker.is_writing() && !worker.is_running_full(),
            "a generation owns the output file without being a full run"
        );
        let frame = drain_until(&worker, 300, |worker, _| !worker.is_running())
            .expect("the generation produced no surface");
        assert_eq!(frame.kind, crate::viewer::snapshot::FrameKind::Final);
        assert!(
            frame.stress.is_some(),
            "the panel's stress block needs the layer a run's own export produces"
        );
        let progress = worker.progress().expect("a run to report on");
        let RunStatus::Finished {
            triangles,
            volume_mm3,
            stl_path,
        } = progress.status
        else {
            panic!("the generation did not finish: {:?}", progress.status);
        };
        assert_eq!(frame.mesh.triangles(), triangles);
        assert!(
            volume_mm3 > 0.0 && volume_mm3 < CANTILEVER_DOMAIN_MM3,
            "implausible exported volume {volume_mm3} mm3"
        );
        assert_eq!(stl_path, stl.display().to_string());

        // And the file is that very surface.
        let (_, written) = crate::mesh::stl::read(&stl).expect("the STL");
        assert_eq!(written.len(), triangles);
        // Recorded as ours, so the editor never accuses the session of its own
        // generation.
        let modified = std::fs::metadata(&stl)
            .and_then(|data| data.modified())
            .expect("the mtime");
        assert!(worker.wrote_output(&stl, modified));
        worker.join();
        std::fs::remove_file(&stl).ok();
    }

    /// An engine that reports no iteration at all still leaves its design behind,
    /// and generating from it writes the very file the run wrote.
    ///
    /// The defect: retention was fed by the per-iteration callback alone, so a
    /// completed solid run - which reports none, by design - kept nothing, and
    /// "generate stl" stayed disabled for the rest of the session.
    ///
    /// The file the generation writes is compared with the run's own byte for
    /// byte, because a design that generates something *else* is no better than
    /// one that cannot be generated at all. This engine refuses the `[output]`
    /// passes, so what that comparison pins here is the pipeline being the same
    /// one twice over the same field; where in the run the field is kept is
    /// pinned by
    /// `a_completed_run_keeps_the_field_the_deliverable_passes_have_not_touched`,
    /// on an engine that allows them. The engine's silence is asserted beside it:
    /// the fix is where the field is kept, never a report the engine does not
    /// make.
    #[test]
    fn a_solid_run_keeps_the_field_it_produced_and_generates_the_same_file() {
        let (_dir, path) = write_temp("worker_solid_generate", solid_fixture());
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(solid_fixture()).expect("parse");
        let stl = directory.join(&config.output.stl_path);
        let mut worker = Worker::new();
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("start");
        drain_until(&worker, 300, |worker, _| !worker.is_running());
        assert!(!worker.is_running(), "the run never finished");
        assert!(stl.exists(), "the full run wrote no file");
        let written = std::fs::read(&stl).expect("the run's own file");
        std::fs::remove_file(&stl).expect("take the run's own file away");

        // The engine reported nothing, and still reports nothing.
        let progress = worker.progress().expect("a run to report on");
        assert!(
            progress.latest.is_none(),
            "the solid engine was made to report an iteration: {:?}",
            progress.latest
        );
        assert!(matches!(progress.status, RunStatus::Finished { .. }));

        // And the design it produced is there to export.
        let kept = worker.retained().expect("the solid run kept no design");
        assert_eq!(kept.densities().len(), kept.problem().grid.n_cells());
        assert!(
            kept.densities().iter().any(|&density| density > 0.5),
            "the kept field has nothing solid in it"
        );
        assert_eq!(kept.problem().output.stl_path, stl);
        assert!(worker.can_generate(), "the button stayed disabled");

        assert!(worker.generate(), "the generation never started");
        drain_until(&worker, 300, |worker, _| !worker.is_running());
        assert!(!worker.is_running(), "the generation never ended");
        assert!(stl.exists(), "the generation wrote nothing");
        let generated = std::fs::read(&stl).expect("the generation's file");
        assert_eq!(
            generated.len(),
            written.len(),
            "the generation wrote a different surface from the run's own"
        );
        assert!(
            generated == written,
            "the generation's file differs from the one the run wrote"
        );
        worker.join();
        std::fs::remove_file(&stl).ok();
    }

    /// *Which* field a completed run keeps, pinned by the file it produces: the
    /// one the engine handed over, never the one the deliverable pipeline has
    /// already been over.
    ///
    /// `flush = "walls"` is the lever, because it is the pass that is not
    /// idempotent: it reaches `flush_depth_mm` past the material it finds near a
    /// surface, so applying it to a field it has already been applied to extends
    /// that fringe again. A generation puts the kept field through that pipeline,
    /// so a field kept *after* `finish` rather than before it comes out of the
    /// generation as a different part from the one the run wrote - which is what
    /// the byte comparison here catches.
    #[test]
    fn a_completed_run_keeps_the_field_the_deliverable_passes_have_not_touched() {
        let text = growth_fixture().replace(
            "stl_path = \"fixture.stl\"",
            "stl_path = \"fixture.stl\"\nflush = \"walls\"",
        );
        let (_dir, path) = write_temp("worker_flush_generate", &text);
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(&text).expect("parse");
        let stl = directory.join(&config.output.stl_path);
        let mut worker = Worker::new();
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("start");
        drain_until(&worker, 300, |worker, _| !worker.is_running());
        assert!(!worker.is_running(), "the run never finished");
        let progress = worker.progress().expect("a run to report on");
        assert!(
            matches!(progress.status, RunStatus::Finished { .. }),
            "{:?}",
            progress.status
        );
        // The fixture has to make the pass do something, or it pins nothing: a
        // fill that filled nothing fills nothing the second time either.
        assert!(
            progress
                .flush
                .iter()
                .any(|note| note.starts_with("filled ")),
            "the flush pass filled nothing, so a second application of it would be invisible: {:?}",
            progress.flush
        );
        let written = std::fs::read(&stl).expect("the run's own file");
        std::fs::remove_file(&stl).expect("take the run's own file away");

        assert!(worker.can_generate());
        assert!(worker.generate(), "the generation never started");
        drain_until(&worker, 300, |worker, _| !worker.is_running());
        assert!(!worker.is_running(), "the generation never ended");
        let generated = std::fs::read(&stl).expect("the generation's file");
        assert!(
            generated == written,
            "the generation exported a field the passes had already been over: {} bytes against \
             the run's own {}",
            generated.len(),
            written.len()
        );
        worker.join();
        std::fs::remove_file(&stl).ok();
    }

    /// The user's own story. A long run is stopped part way through, which of
    /// itself writes nothing at all; the design it had reached is still on screen,
    /// and asking for its stl writes exactly that design.
    #[test]
    fn a_run_that_was_stopped_can_still_have_its_design_generated() {
        let (_dir, path) = write_temp("worker_stopped_generate", CANTILEVER);
        let directory = path.parent().expect("a directory").to_path_buf();
        let mut config = Config::parse(CANTILEVER).expect("parse");
        // A budget it cannot spend and a tolerance it cannot reach, so the run is
        // certainly still going when it is stopped. Nothing but the stop ends it.
        config.optimization.max_iterations = Some(400);
        config.optimization.convergence_tol = Some(1e-9);
        let stl = directory.join(&config.output.stl_path);
        let mut worker = Worker::new();
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("start");

        // Watched until there is a design on screen: a surface has been extracted
        // from it, which is what "the design I was looking at" means, and the
        // engine has had enough iterations to have resolved one.
        let shown = drain_until(&worker, 300, |worker, frame| {
            let resolved = worker
                .progress()
                .and_then(|progress| progress.latest)
                .is_some_and(|stats| stats.iteration >= 8);
            resolved
                && worker.retained().is_some()
                && frame.is_some_and(|frame| frame.mesh.triangles() > 0)
        })
        .expect("the run put no surface on screen");
        assert!(shown.mesh.triangles() > 0);
        assert!(worker.is_running_full(), "the run ended on its own");
        let kept = worker.retained().expect("the run kept no design");
        let solid = kept
            .densities()
            .iter()
            .filter(|&&density| density > 0.5)
            .count();
        assert!(solid > 0, "the kept field has nothing solid in it");

        worker.stop();
        assert!(!worker.is_running(), "the session forgets a stopped run");
        assert!(
            !stl.exists(),
            "stopping a run wrote {} on its own",
            stl.display()
        );
        // The stop took the run away and left the design: that is the whole
        // point, and the button is offered on it.
        let after = worker.retained().expect("the stop threw the design away");
        assert_eq!(after.densities().len(), kept.densities().len());
        assert!(worker.can_generate());

        assert!(worker.generate(), "the generation never started");
        let frame = drain_until(&worker, 300, |worker, _| !worker.is_running())
            .expect("the generation produced no surface");
        assert_eq!(frame.kind, crate::viewer::snapshot::FrameKind::Final);
        assert!(frame.stress.is_some(), "the stress solve produced nothing");
        let progress = worker.progress().expect("a run to report on");
        let RunStatus::Finished {
            triangles,
            volume_mm3,
            ..
        } = progress.status
        else {
            panic!("the generation did not finish: {:?}", progress.status);
        };
        assert!(
            volume_mm3 > 0.0 && volume_mm3 < CANTILEVER_DOMAIN_MM3,
            "implausible exported volume {volume_mm3} mm3"
        );
        let (_, written) = crate::mesh::stl::read(&stl).expect("the STL");
        assert_eq!(written.len(), triangles);
        assert_eq!(frame.mesh.triangles(), triangles);
        worker.join();
        std::fs::remove_file(&stl).ok();
    }

    /// The session's console says what the part came out at, beside the line
    /// that says where it went.
    ///
    /// Called directly rather than through a run thread, because what is asserted
    /// is what the run printed: `full` writes to the session's console, which is
    /// the process's stdout in a running editor and this buffer here. The lines
    /// are the panel's own, from the same formatter, so a user reading the
    /// terminal and a user reading the window cannot be told two different
    /// numbers.
    #[test]
    fn a_full_run_echoes_the_safety_factor_to_the_session_console() {
        let (_dir, path) = write_temp("worker_console_run", CANTILEVER);
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(CANTILEVER).expect("parse");
        let problem = Arc::new(Problem::build(&config, &directory).expect("build"));
        let stl = problem.output.stl_path.clone();
        let retention = Retention {
            problem,
            slot: Arc::new(Mutex::new(None)),
        };
        let link = ViewLink::new();
        let cancel = AtomicBool::new(false);
        let writes = Mutex::new(None);
        let mut console: Vec<u8> = Vec::new();

        full(&retention, &link, &cancel, &writes, &mut console);
        let printed = String::from_utf8(console).expect("console lines");
        assert!(
            printed.contains(&format!("editor wrote  {}", stl.display())),
            "the run never said what it wrote: {printed}"
        );
        let summary = link
            .progress()
            .stress
            .expect("the run analysed no stresses");
        for line in summary.lines() {
            assert!(
                printed.contains(&format!("editor stress {line}")),
                "the console never said {line:?}: {printed}"
            );
        }
        assert!(
            summary.headline.starts_with("safety factor "),
            "{}",
            summary.headline
        );
        assert_eq!(summary.cases.len(), 1, "{:?}", summary.cases);
        std::fs::remove_file(&stl).ok();
    }

    /// And so does an stl generation: it is the same tail on a design that was
    /// kept, so the deliverables are the same and what is said about them is.
    ///
    /// The stopped half is the other rule: a generation that analysed nothing
    /// says nothing about stresses, rather than echoing a summary of a part it
    /// never wrote.
    #[test]
    fn a_generation_echoes_the_safety_factor_to_the_session_console() {
        let (_dir, path) = write_temp("worker_console_stl", CANTILEVER);
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(CANTILEVER).expect("parse");
        let problem = Arc::new(Problem::build(&config, &directory).expect("build"));
        let stl = problem.output.stl_path.clone();
        let retained = Retained {
            densities: vec![1.0; problem.grid.n_cells()],
            problem,
            reduce: None,
        };
        let link = ViewLink::new();
        let cancel = AtomicBool::new(false);
        let writes = Mutex::new(None);
        let mut console: Vec<u8> = Vec::new();

        export_retained(&retained, &link, &cancel, &writes, &mut console);
        let printed = String::from_utf8(console).expect("console lines");
        assert!(
            printed.contains(&format!("editor wrote  {}", stl.display())),
            "the generation never said what it wrote: {printed}"
        );
        let summary = link
            .progress()
            .stress
            .expect("the generation analysed no stresses");
        for line in summary.lines() {
            assert!(
                printed.contains(&format!("editor stress {line}")),
                "the console never said {line:?}: {printed}"
            );
        }
        std::fs::remove_file(&stl).ok();

        // Stopped before it analysed anything: it wrote no file and quotes no
        // numbers.
        let stopped = AtomicBool::new(true);
        let mut console: Vec<u8> = Vec::new();
        export_retained(&retained, &ViewLink::new(), &stopped, &writes, &mut console);
        let printed = String::from_utf8(console).expect("console lines");
        assert!(
            printed.contains("editor stl    stopped"),
            "unexpected console output: {printed}"
        );
        assert!(
            !printed.contains("editor stress"),
            "a stopped generation quoted a safety factor: {printed}"
        );
        assert!(
            !stl.exists(),
            "a stopped generation wrote {}",
            stl.display()
        );
    }

    /// A generation from a design a reduction chose carries that schedule into
    /// its own report and into the panel.
    ///
    /// The gap this closes: the kept design travelled without its schedule, so
    /// asking a stopped run for its stl wrote a stress report with no `reduce`
    /// object in it - the same design, described as if nothing had chosen it.
    /// The record travels with the field now, and `finish` measures what the
    /// part in the file came out at onto the copy of it this export writes, so
    /// the JSON's `finished_safety_factor` describes this generation and the
    /// panel is shown the same record.
    #[test]
    fn a_generation_carries_the_schedule_that_chose_the_design() {
        let text = CANTILEVER.replace(
            "stl_path = \"kept.stl\"",
            "stl_path = \"kept_reduced.stl\"\nstress_json = \"kept_reduced.stress.json\"",
        );
        let (_dir, path) = write_temp("worker_reduce_stl", &text);
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(&text).expect("parse");
        let problem = Arc::new(Problem::build(&config, &directory).expect("build"));
        let stl = problem.output.stl_path.clone();
        let report = problem
            .output
            .stress_json
            .clone()
            .expect("the fixture asks for a report");
        let stage = crate::engine::ReduceStage {
            index: 2,
            target_fraction: 0.8,
            achieved_fraction: 0.7931,
            iterations: 33,
            safety_factor: Some(4.2),
            passed: true,
            refine: false,
        };
        let retained = Retained {
            densities: vec![1.0; problem.grid.n_cells()],
            problem,
            reduce: Some(ReduceSummary {
                method: crate::config::ReduceMethod::Continuation,
                target_safety_factor: 3.0,
                exported: stage,
                stages: vec![stage],
                // Unmeasured until an export measures it, which is what this
                // test is about.
                finished_safety_factor: None,
            }),
        };
        let link = ViewLink::new();
        let cancel = AtomicBool::new(false);
        let writes = Mutex::new(None);
        let mut console: Vec<u8> = Vec::new();

        export_retained(&retained, &link, &cancel, &writes, &mut console);
        let written = std::fs::read_to_string(&report).expect("the stress report");
        assert!(written.contains("\"reduce\": {"), "{written}");
        assert!(written.contains("\"exported_stage\": 2,"), "{written}");
        assert!(written.contains("\"iterations\": 33,"), "{written}");
        assert!(
            !written.contains("\"finished_safety_factor\": null"),
            "the export never measured the part it wrote: {written}"
        );

        // The panel is shown that same record, finished factor and all, and the
        // kept design still holds the one the run gave it: generating twice
        // generates the same thing twice.
        let shown = link.progress().reduce.expect("the panel was told nothing");
        assert_eq!(shown.exported, stage);
        assert!(shown.finished_safety_factor.is_some());
        assert_eq!(
            retained
                .reduce()
                .and_then(|reduce| reduce.finished_safety_factor),
            None,
            "the export wrote its own measurement back onto the kept design"
        );
        std::fs::remove_file(&stl).ok();
        std::fs::remove_file(&report).ok();
    }

    /// An export that came out in two pieces says so on the session's console,
    /// above the safety factor it qualifies, and the panel is told the same.
    ///
    /// The incident: a part meant to link two rods was exported as two separate
    /// bodies, each load group shunting into its own local support anchor, and
    /// every surface reported "safety factor 5.07" of a thing held together by
    /// air. The solve is truthful about the model it was handed and cannot know
    /// the supports are fictitious - but the mesh came out in pieces, and that is
    /// knowable and now said.
    ///
    /// Exported from a saturated field rather than optimized, which is the same
    /// tail [`crate::viewer::finish`] runs for a full run: what is under test is
    /// what the pipeline says about a surface, not how the surface was arrived
    /// at.
    #[test]
    fn an_export_in_two_bodies_warns_on_the_console_and_in_the_panel() {
        let text = crate::viewer::tests::TWO_ANCHORED_LUMPS;
        let (_dir, path) = write_temp("worker_console_bodies", text);
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(text).expect("parse");
        let problem = Arc::new(Problem::build(&config, &directory).expect("build"));
        let stl = problem.output.stl_path.clone();
        let retained = Retained {
            densities: vec![1.0; problem.grid.n_cells()],
            problem,
            reduce: None,
        };
        let link = ViewLink::new();
        let cancel = AtomicBool::new(false);
        let writes = Mutex::new(None);
        let mut console: Vec<u8> = Vec::new();

        export_retained(&retained, &link, &cancel, &writes, &mut console);
        let printed = String::from_utf8(console).expect("console lines");
        let summary = link
            .progress()
            .stress
            .expect("the generation analysed no stresses");

        // The fixture has to be the case it was written for: two bodies, both
        // kept, and a safety factor that the warning is about.
        let warning = summary
            .warning
            .clone()
            .unwrap_or_else(|| panic!("the export was not reported in pieces: {printed}"));
        assert_eq!(
            warning,
            "warning: the export is 2 separate bodies - this safety factor describes each piece \
             against its own supports, not one connected part"
        );
        assert!(
            summary.headline.starts_with("safety factor ") && !summary.headline.contains("n/a"),
            "the fixture stopped producing a factor to warn about: {}",
            summary.headline
        );
        // Every line the panel is shown is a line the console was sent, the
        // warning first among them.
        assert_eq!(summary.lines()[0], warning);
        for line in summary.lines() {
            assert!(
                printed.contains(&format!("editor stress {line}")),
                "the console never said {line:?}: {printed}"
            );
        }
        std::fs::remove_file(&stl).ok();
    }

    /// With nothing run yet there is nothing to export, and the entry point
    /// behind the disabled button is a guard rather than a path.
    #[test]
    fn with_nothing_kept_there_is_nothing_to_generate() {
        let mut worker = Worker::new();
        assert!(worker.retained().is_none());
        assert!(!worker.can_generate());
        assert!(!worker.generate(), "a generation started from nothing");
        assert_eq!(worker.kind(), None);
        assert!(!worker.is_running() && !worker.is_writing());
        assert!(!worker.probe().is_running(), "a thread was started");
    }

    /// One writer at a time. A full run and a generation both own the output
    /// file, so neither may take the worker from the other, and the preview an
    /// edit asks for takes it from neither.
    #[test]
    fn only_one_run_that_writes_may_own_the_worker() {
        assert!(RunKind::Full.writes() && RunKind::Export.writes());
        assert!(!RunKind::Preview.writes(), "a preview writes nothing");

        let (_dir, path) = write_temp("worker_one_writer", growth_fixture());
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(growth_fixture()).expect("parse");
        let stl = directory.join(&config.output.stl_path);
        let mut worker = Worker::new();

        // A generation in flight, assembled exactly as `launch` assembles one:
        // this is about what the worker refuses while one is going, and a real
        // one would be over before the question could be asked.
        let done = Arc::new(AtomicBool::new(false));
        let alive = worker.threads.enter(Arc::clone(&done));
        worker.current = Some(Run {
            kind: RunKind::Export,
            link: Arc::new(ViewLink::new()),
            cancel: Arc::new(AtomicBool::new(false)),
            handle: None,
            done: Arc::clone(&done),
        });
        assert!(worker.is_writing() && !worker.is_running_full());
        assert!(
            !worker.can_generate(),
            "one generation at a time, whatever is kept"
        );

        // Neither a preview nor a full run may take the file away from it.
        worker
            .start(&config, &directory, RunKind::Preview)
            .expect("start");
        assert_eq!(
            worker.kind(),
            Some(RunKind::Export),
            "a preview took the worker from a generation"
        );
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("start");
        assert_eq!(
            worker.kind(),
            Some(RunKind::Export),
            "a full run took the worker from a generation"
        );
        drop(alive);
        assert!(!worker.is_running(), "the generation is over");
        assert!(!stl.exists(), "something wrote {}", stl.display());

        // And a generation is refused while a full run has the worker.
        let mut worker = Worker::new();
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("start");
        assert!(worker.is_writing());
        assert!(!worker.can_generate(), "a full run owns the output file");
        assert!(!worker.generate());
        assert_eq!(worker.kind(), Some(RunKind::Full));
        worker.stop();
        worker.join();
        std::fs::remove_file(&stl).ok();
    }

    /// A generation is not a kind of run a configuration starts, and asking for
    /// one that way is told so rather than quietly handed something else.
    #[test]
    fn a_generation_is_not_started_from_a_configuration() {
        let config = Config::parse(growth_fixture()).expect("parse");
        let mut worker = Worker::new();
        let error = worker
            .start(&config, &std::env::temp_dir(), RunKind::Export)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("design already on screen"),
            "unexpected error: {error}"
        );
        assert_eq!(worker.startup_error(), Some(error.as_str()));
        assert_eq!(worker.kind(), None);
        assert!(!worker.is_running());
        assert!(!worker.probe().is_running(), "a thread was started");
    }

    #[test]
    fn a_run_that_cannot_be_built_reports_why_and_starts_nothing() {
        let mut config = Config::parse(fixture()).expect("parse");
        config.optimization.mass_fraction = Some(5.0);
        let mut worker = Worker::new();
        let error = worker
            .start(&config, &std::env::temp_dir(), RunKind::Preview)
            .unwrap_err()
            .to_string();
        assert!(error.contains("mass_fraction"), "unexpected error: {error}");
        assert!(worker.startup_error().is_some());
        assert!(!worker.is_running());
        assert_eq!(worker.kind(), None);
    }

    #[test]
    fn cancelling_a_preview_stops_it_and_leaves_the_worker_free() {
        let config = Config::parse(growth_fixture()).expect("parse");
        let mut worker = Worker::new();
        worker
            .start(&config, &std::env::temp_dir(), RunKind::Preview)
            .expect("start");
        worker.cancel_preview();
        assert_eq!(worker.kind(), None, "a cancelled run is forgotten");
        assert!(!worker.is_running());
        // Starting again after a cancel is what an edit arriving mid-run does.
        worker
            .start(&config, &std::env::temp_dir(), RunKind::Preview)
            .expect("restart");
        assert_eq!(worker.kind(), Some(RunKind::Preview));
        // The window's way out is the same stop: nothing an editor started is
        // ever left running behind it.
        worker.stop();
        assert_eq!(worker.kind(), None);
        assert!(!worker.is_running());
        worker.join();
    }

    #[test]
    fn the_run_probe_counts_a_thread_from_before_it_starts_until_it_ends() {
        let worker = Worker::new();
        let probe = worker.probe();
        assert!(!probe.is_running(), "an idle worker has no threads");
        let first_done = Arc::new(AtomicBool::new(false));
        let second_done = Arc::new(AtomicBool::new(false));

        let first = worker.threads.enter(Arc::clone(&first_done));
        assert!(probe.is_running());
        // A stopped run winding down while a new one is going: the count covers
        // both, and the window has to outlive the last of them.
        let second = worker.threads.enter(Arc::clone(&second_done));
        assert!(probe.is_running());
        drop(second);
        assert!(probe.is_running(), "one thread is still alive");
        // And a guard ends its own run and only its own.
        assert!(second_done.load(Ordering::Relaxed));
        assert!(!first_done.load(Ordering::Relaxed));

        drop(first);
        assert!(!probe.is_running());
        assert!(first_done.load(Ordering::Relaxed));
    }

    /// A run thread that panics has ended as much as one that returned. The
    /// stores used to sit at the foot of the closure, which an unwind skips:
    /// the run stayed "in flight" for the rest of the session, the panel
    /// showing a run that was over and `start` silently doing nothing.
    #[test]
    fn a_panicking_run_thread_leaves_the_session_runnable() {
        let (_dir, path) = write_temp("worker_panic", growth_fixture());
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(growth_fixture()).expect("parse");
        let mut worker = Worker::new();
        let probe = worker.probe();

        // A full run in flight, assembled exactly as `start` assembles one:
        // nothing in the engine can be asked to panic on demand, and it is the
        // guard rather than the engine that this is about.
        let done = Arc::new(AtomicBool::new(false));
        let alive = worker.threads.enter(Arc::clone(&done));
        worker.current = Some(Run {
            kind: RunKind::Full,
            link: Arc::new(ViewLink::new()),
            cancel: Arc::new(AtomicBool::new(false)),
            handle: None,
            done: Arc::clone(&done),
        });
        assert!(worker.is_running() && worker.is_running_full());
        assert!(probe.is_running());

        let thread = std::thread::spawn(move || {
            let _alive = alive;
            panic!("the solver exploded");
        });
        assert!(thread.join().is_err(), "the panic must keep propagating");

        // Everything the panel and the window read is true again.
        assert!(!worker.is_running(), "a run that unwound is over");
        assert!(!worker.is_running_full(), "and owns the worker no longer");
        assert!(!probe.is_running(), "and the loop is free to stop");
        // And the session really is usable: with the flag stuck this was a
        // silent no-op that started nothing at all.
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("start");
        assert!(worker.is_running_full(), "the next full run never started");
        worker.stop();
        worker.join();
        assert!(!directory.join(&config.output.stl_path).exists());
    }

    /// What the panel sees the instant a full run is stopped: nothing running
    /// and nothing owning the worker, so "run full" is offered again. The
    /// thread behind it is still winding down - the probe's question, not the
    /// panel's - and can no longer write anything, so it owns no output either.
    ///
    /// The file is what this pins: a stop writes nothing *on its own*, before and
    /// after the button that exports what a stop leaves behind. Nothing here asks
    /// for that export, and so nothing here produces a file - see
    /// `a_run_that_was_stopped_can_still_have_its_design_generated` for the
    /// asking.
    #[test]
    fn stopping_a_full_run_hands_the_worker_back_at_once() {
        let (_dir, path) = write_temp("worker_stop_full", growth_fixture());
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(growth_fixture()).expect("parse");
        let mut worker = Worker::new();
        let probe = worker.probe();
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("start");
        assert!(worker.is_running_full());

        worker.stop();
        assert!(!worker.is_running());
        assert!(
            !worker.is_running_full(),
            "a stopped full run must release the worker at once"
        );
        assert!(probe.is_running(), "its thread is still winding down");

        // Which is to say the editor is usable again: another full run starts
        // rather than being swallowed by the flag of the one that was stopped.
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("restart");
        assert!(worker.is_running_full());
        worker.stop();
        worker.join();
        assert!(!probe.is_running());
        assert!(!directory.join(&config.output.stl_path).exists());
    }

    #[test]
    fn the_run_probe_covers_a_preview_a_full_run_and_a_stopped_one() {
        let (_dir, path) = write_temp("worker_probe", growth_fixture());
        let directory = path.parent().expect("a directory").to_path_buf();
        let config = Config::parse(growth_fixture()).expect("parse");
        let mut worker = Worker::new();
        let probe = worker.probe();
        assert!(!probe.is_running());

        // A preview an edit started.
        worker
            .start(&config, &directory, RunKind::Preview)
            .expect("start");
        assert!(probe.is_running(), "a preview is a thread like any other");
        worker.join();
        assert!(!probe.is_running());

        // A full run, and the gap the panel's own answer leaves: a stopped run
        // is gone from the session at once, while its thread is still on its
        // way to the next checkpoint. That interval is exactly what the window
        // must keep pumping through, so the probe still reports it.
        worker
            .start(&config, &directory, RunKind::Full)
            .expect("start");
        assert!(probe.is_running());
        worker.stop();
        assert!(!worker.is_running(), "the session forgets a stopped run");
        assert!(
            probe.is_running(),
            "a stopped run's thread is still alive and the loop must wait for it"
        );
        worker.join();
        assert!(!probe.is_running());
        // A stopped full run writes nothing, probe or no probe.
        assert!(!directory.join(&config.output.stl_path).exists());
    }
}
