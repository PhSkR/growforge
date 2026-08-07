//! The hand-off between the optimization worker, the preview mesher and the
//! render thread.
//!
//! Every hand-off uses [`LatestSlot`]: a channel of capacity one whose `push`
//! overwrites whatever the consumer has not taken yet. A viewer only ever wants
//! the newest design, so dropping the intermediate ones is the correct
//! behaviour rather than a compromise, and it bounds the memory the link can
//! hold to a single snapshot no matter how far behind the consumer falls.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

use crate::report::{IterationStats, Reporter};
use crate::stress::StressSummary;
use crate::viewer::scene::LayerMesh;

/// A single slot channel that keeps only the most recently pushed value.
#[derive(Debug)]
pub struct LatestSlot<T> {
    state: Mutex<SlotState<T>>,
    arrived: Condvar,
}

#[derive(Debug)]
struct SlotState<T> {
    value: Option<T>,
    closed: bool,
    /// Values that were overwritten before anyone took them; reported by
    /// [`LatestSlot::dropped`] so tests can prove the throttle is working.
    dropped: usize,
}

impl<T> Default for LatestSlot<T> {
    fn default() -> Self {
        LatestSlot {
            state: Mutex::new(SlotState {
                value: None,
                closed: false,
                dropped: 0,
            }),
            arrived: Condvar::new(),
        }
    }
}

impl<T> LatestSlot<T> {
    /// An empty, open slot.
    pub fn new() -> LatestSlot<T> {
        LatestSlot::default()
    }

    fn lock(&self) -> MutexGuard<'_, SlotState<T>> {
        // A panicking producer must not take the viewer down with it: the
        // slot's invariants do not depend on where the panic happened.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Store `value`, discarding any value the consumer has not taken.
    ///
    /// Returns false once the slot is closed, which is how a producer learns
    /// that the consumer has gone away.
    pub fn push(&self, value: T) -> bool {
        let mut state = self.lock();
        if state.closed {
            return false;
        }
        if state.value.replace(value).is_some() {
            state.dropped += 1;
        }
        drop(state);
        self.arrived.notify_all();
        true
    }

    /// Take the pending value without blocking.
    pub fn try_take(&self) -> Option<T> {
        self.lock().value.take()
    }

    /// Block until a value is pending or the slot closes.
    pub fn take_blocking(&self) -> Option<T> {
        let mut state = self.lock();
        loop {
            if let Some(value) = state.value.take() {
                return Some(value);
            }
            if state.closed {
                return None;
            }
            state = self
                .arrived
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Refuse further pushes and wake every waiter. The pending value, if any,
    /// stays takeable so a consumer can drain the slot before it stops.
    pub fn close(&self) {
        self.lock().closed = true;
        self.arrived.notify_all();
    }

    /// True once [`LatestSlot::close`] has been called.
    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }

    /// True while a value is waiting to be taken.
    pub fn is_pending(&self) -> bool {
        self.lock().value.is_some()
    }

    /// How many values were overwritten before anyone took them.
    pub fn dropped(&self) -> usize {
        self.lock().dropped
    }
}

/// Physical densities of one optimization iteration.
#[derive(Debug, Clone)]
pub struct DensitySnapshot {
    /// One-based iteration the densities belong to.
    pub iteration: usize,
    /// Physical density of every cell, in grid element order.
    pub densities: Vec<f64>,
}

/// What a rendered frame of the density layer represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// An unsmoothed, unvalidated preview of an intermediate design.
    Preview {
        /// Iteration the preview was extracted from.
        iteration: usize,
    },
    /// The smoothed, validated surface that was written to the STL.
    Final,
}

/// A render-ready density surface.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Whether this is a preview or the exported surface.
    pub kind: FrameKind,
    /// Flat shaded triangles of the surface.
    pub mesh: LayerMesh,
    /// The same surface coloured by von Mises stress, when the run produced a
    /// stress report. Only a final frame ever carries one.
    pub stress: Option<LayerMesh>,
}

/// Where a run has got to, for the stats overlay.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum RunStatus {
    /// The engine is iterating.
    #[default]
    Optimizing,
    /// The engine finished; enclosed cavities and stresses are being worked out.
    Analysing,
    /// The engine finished and the mesh is being extracted and written.
    Exporting,
    /// The STL was written.
    Finished {
        /// Triangle count of the exported mesh.
        triangles: usize,
        /// Enclosed volume of the exported mesh in cubic millimetres.
        volume_mm3: f64,
        /// Where the STL went.
        stl_path: String,
    },
    /// The run failed; the string is the formatted error chain.
    Failed(String),
}

/// The stats block the side panel shows.
#[derive(Debug, Clone, Default)]
pub struct Progress {
    /// Most recent per-iteration line.
    pub latest: Option<IterationStats>,
    /// Most recent one-off engine message.
    pub note: Option<String>,
    /// What the `[output] trim` pass said, once the run has finished; empty
    /// while it is running and under the default `trim = "off"`. Separate from
    /// `note` because it is not the engine's line and must not displace it:
    /// why a run ended and what the trim removed are two different things a
    /// panel says at once.
    pub trim: Vec<String>,
    /// What the `[output] flush` pass said, once the run has finished; empty
    /// while it is running and under the default `flush = "off"`. Separate from
    /// `trim` for the reason `trim` is separate from `note`, and drawn between
    /// the two: the passes over the field are as many statements as there are
    /// passes, and the panel says all of them.
    pub flush: Vec<String>,
    /// What the `[output] reinforce` pass said, once the run has finished; empty
    /// while it is running and under the default `reinforce = "off"`. Separate
    /// from `trim` for the same reason `trim` is separate from `note`, and drawn
    /// beside it: what a run freed and what it spent are two statements, and the
    /// panel says both.
    pub reinforce: Vec<String>,
    /// What the `[output] boundaries` clamp said, once the run has finished;
    /// empty while it is running and under `boundaries = "voxel"`. Separate from
    /// `trim` for the same reason `trim` is separate from `note`: the two passes
    /// are two different statements about the part, and the panel says both.
    pub clamp: Vec<String>,
    /// What the part came out at, once the run has analysed the field it
    /// exported: the safety factor and the peak of every load case. `None` while
    /// a run is going, and for a run whose stress solve produced no report -
    /// which leaves the block absent rather than showing the last run's numbers.
    pub stress: Option<StressSummary>,
    /// Stage the run is in.
    pub status: RunStatus,
}

/// Everything shared between the optimization worker, the preview mesher and
/// the window.
#[derive(Debug, Default)]
pub struct ViewLink {
    /// Newest density field waiting for the preview mesher.
    pub densities: LatestSlot<DensitySnapshot>,
    /// Newest render-ready surface waiting for the window.
    pub frames: LatestSlot<Frame>,
    /// Latest progress line and run status.
    progress: Mutex<Progress>,
    /// Set when the window has gone away; the run continues headless and stops
    /// paying for snapshots.
    detached: AtomicBool,
}

impl ViewLink {
    /// A link with nothing reported yet.
    pub fn new() -> ViewLink {
        ViewLink::default()
    }

    /// Snapshot of the current progress.
    pub fn progress(&self) -> Progress {
        self.progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Record the stage a run is in.
    pub fn set_status(&self, status: RunStatus) {
        self.progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .status = status;
    }

    /// Record what the trim pass said, for the panel.
    ///
    /// Written once, where the pass ends, rather than through the reporter: the
    /// trim happens after the engine has handed its field over, so there is no
    /// engine left to report it, and the lines have to outlive the moment the
    /// way the finished status does.
    pub fn set_trim_notes(&self, notes: Vec<String>) {
        self.progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .trim = notes;
    }

    /// Record what the flush pass said, for the panel.
    ///
    /// Written once, where the pass ends, for the reason
    /// [`ViewLink::set_trim_notes`] gives: it runs after the engine has handed
    /// its field over, and the lines have to outlive the moment.
    pub fn set_flush_notes(&self, notes: Vec<String>) {
        self.progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .flush = notes;
    }

    /// Record what the reinforcement pass said, for the panel.
    ///
    /// Written once, where the pass ends, for the reason
    /// [`ViewLink::set_trim_notes`] gives: it runs after the engine has handed
    /// its field over, and the lines have to outlive the moment.
    pub fn set_reinforce_notes(&self, notes: Vec<String>) {
        self.progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .reinforce = notes;
    }

    /// Record what the boundary clamp said, for the panel.
    ///
    /// Written where the export ends, for the reason [`ViewLink::set_trim_notes`]
    /// gives: the pass runs after the engine has handed its field over, and the
    /// lines have to outlive the moment.
    pub fn set_clamp_notes(&self, notes: Vec<String>) {
        self.progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clamp = notes;
    }

    /// Record what the part came out at, for the panel.
    ///
    /// Written where the analysis that describes the exported part ends, which
    /// after a trim is the second one: what the panel shows has to be the
    /// structure in the file, exactly as the console table and the JSON are.
    /// `None` is a run that produced no report, and it is written all the same -
    /// a summary is replaced rather than left over.
    pub fn set_stress_summary(&self, summary: Option<StressSummary>) {
        self.progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stress = summary;
    }

    /// True once the window has detached.
    pub fn is_detached(&self) -> bool {
        self.detached.load(Ordering::Relaxed)
    }

    /// Detach: stop collecting snapshots and let every waiter go.
    pub fn detach(&self) {
        self.detached.store(true, Ordering::Relaxed);
        self.densities.close();
        self.frames.close();
    }
}

/// Forwards engine progress to the console reporter it wraps and, while the
/// window is attached, to the viewer.
pub struct ViewReporter<'a> {
    inner: &'a dyn Reporter,
    link: &'a ViewLink,
}

impl<'a> ViewReporter<'a> {
    /// Wrap `inner`, mirroring everything it receives into `link`.
    pub fn new(inner: &'a dyn Reporter, link: &'a ViewLink) -> ViewReporter<'a> {
        ViewReporter { inner, link }
    }
}

impl Reporter for ViewReporter<'_> {
    fn iteration(&self, stats: &IterationStats) {
        self.inner.iteration(stats);
        if self.link.is_detached() {
            return;
        }
        self.link
            .progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .latest = Some(stats.clone());
    }

    fn note(&self, message: &str) {
        self.inner.note(message);
        if self.link.is_detached() {
            return;
        }
        self.link
            .progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .note = Some(message.to_string());
    }

    fn densities(&self, stats: &IterationStats, densities: &[f64]) {
        self.inner.densities(stats, densities);
        // Once the window is gone the copy would be pure overhead, and the run
        // has to finish at full speed.
        if self.link.is_detached() {
            return;
        }
        self.link.densities.push(DensitySnapshot {
            iteration: stats.iteration,
            densities: densities.to_vec(),
        });
    }

    /// Forwarded, and it has to be: this reporter is what the engine is handed,
    /// so a wrapper that answered the default `false` here would swallow the
    /// question on its way in. It did, and the editor's stop button reached a
    /// full run only at the stage boundaries `run_worker` checks itself - which
    /// on a SIMP run is after the whole optimization.
    ///
    /// Detaching is deliberately *not* a stop. Closing the window of a
    /// `run --view` detaches the link and the run finishes headless; only a
    /// reporter that says so stops anything.
    fn cancelled(&self) -> bool {
        self.inner.cancelled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::SilentReporter;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    #[test]
    fn a_slow_consumer_sees_the_newest_value_and_nothing_older() {
        let slot: LatestSlot<usize> = LatestSlot::new();
        for value in 0..100 {
            assert!(slot.push(value));
        }
        assert_eq!(slot.dropped(), 99);
        assert_eq!(slot.try_take(), Some(99));
        assert_eq!(slot.try_take(), None);
        assert!(!slot.is_pending());
    }

    #[test]
    fn the_slot_holds_at_most_one_value_however_much_is_pushed() {
        let slot: LatestSlot<Vec<u8>> = LatestSlot::new();
        let payload = 1024;
        for _ in 0..1000 {
            slot.push(vec![0u8; payload]);
        }
        // Memory is bounded by construction: exactly one payload survives.
        let held = slot.try_take().expect("one value");
        assert_eq!(held.len(), payload);
        assert!(!slot.is_pending());
        assert_eq!(slot.dropped(), 999);
    }

    #[test]
    fn a_blocked_consumer_wakes_on_a_push_and_on_a_close() {
        let slot: Arc<LatestSlot<usize>> = Arc::new(LatestSlot::new());
        let producer = Arc::clone(&slot);
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            producer.push(7);
        });
        assert_eq!(slot.take_blocking(), Some(7));
        handle.join().expect("producer");

        let closer = Arc::clone(&slot);
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            closer.close();
        });
        assert_eq!(slot.take_blocking(), None);
        handle.join().expect("closer");
    }

    #[test]
    fn a_closed_slot_refuses_pushes_but_still_yields_its_pending_value() {
        let slot: LatestSlot<usize> = LatestSlot::new();
        slot.push(1);
        slot.close();
        assert!(!slot.push(2));
        assert!(slot.is_closed());
        assert_eq!(slot.try_take(), Some(1));
        assert_eq!(slot.take_blocking(), None);
    }

    fn stats(iteration: usize) -> IterationStats {
        IterationStats {
            iteration,
            compliance: 1.0 / iteration as f64,
            volume_fraction: 0.3,
            worst_local_fraction: None,
            max_change: 0.05,
            cg_iterations: vec![10],
            elapsed_s: iteration as f64,
            growth: None,
        }
    }

    #[test]
    fn the_reporter_mirrors_progress_until_it_detaches() {
        let link = ViewLink::new();
        let console = SilentReporter;
        let reporter = ViewReporter::new(&console, &link);

        for iteration in 1..=5 {
            let s = stats(iteration);
            reporter.iteration(&s);
            reporter.densities(&s, &[0.5; 16]);
        }
        reporter.note("converged");
        let progress = link.progress();
        assert_eq!(progress.latest.expect("stats").iteration, 5);
        assert_eq!(progress.note.as_deref(), Some("converged"));
        let snapshot = link.densities.try_take().expect("snapshot");
        assert_eq!(snapshot.iteration, 5);
        assert_eq!(snapshot.densities.len(), 16);

        link.detach();
        assert!(link.is_detached());
        let s = stats(6);
        reporter.iteration(&s);
        reporter.densities(&s, &[1.0; 16]);
        reporter.note("cap reached");
        // Detaching freezes the overlay state and stops the snapshot copies.
        let progress = link.progress();
        assert_eq!(progress.latest.expect("stats").iteration, 5);
        assert_eq!(progress.note.as_deref(), Some("converged"));
        assert!(link.densities.try_take().is_none());
    }

    #[test]
    fn a_growth_progress_line_reaches_the_panel_unchanged() {
        use crate::report::{GrowthPhase, GrowthProgress};

        let link = ViewLink::new();
        let console = SilentReporter;
        let reporter = ViewReporter::new(&console, &link);
        let growth = GrowthProgress {
            phase: GrowthPhase::Branching,
            segments: 42,
            attractors_remaining: 7,
        };
        let stats = IterationStats {
            growth: Some(growth),
            ..stats(3)
        };
        reporter.iteration(&stats);
        reporter.densities(&stats, &[0.25; 8]);

        // The panel needs the growth block, and the mesher the field behind it,
        // through exactly the same channels the SIMP engine uses.
        assert_eq!(link.progress().latest.expect("stats").growth, Some(growth));
        assert_eq!(link.densities.try_take().expect("snapshot").iteration, 3);
    }

    /// The wrapper is what the engine is handed, so every question the engine
    /// asks has to go through it - the cancellation one above all, because
    /// swallowing that is a stop button that does nothing.
    #[test]
    fn the_reporter_forwards_the_cancellation_question_and_detaching_is_not_one() {
        struct Stoppable(AtomicBool);
        impl Reporter for Stoppable {
            fn iteration(&self, _stats: &IterationStats) {}
            fn note(&self, _message: &str) {}
            fn cancelled(&self) -> bool {
                self.0.load(Ordering::Relaxed)
            }
        }

        let link = ViewLink::new();
        let inner = Stoppable(AtomicBool::new(false));
        let reporter = ViewReporter::new(&inner, &link);
        assert!(!reporter.cancelled());

        // A closed window is not a stop: `run --view` finishes headless.
        link.detach();
        assert!(!reporter.cancelled(), "detaching must not stop a run");

        inner.0.store(true, Ordering::Relaxed);
        assert!(reporter.cancelled(), "the question was swallowed");

        // And a reporter that never answers it still never does.
        assert!(!ViewReporter::new(&SilentReporter, &ViewLink::new()).cancelled());
    }

    /// Why a run ended is an engine note, and the note line of the panel is
    /// where a window shows it. The new reason is the one worth pinning: a user
    /// watching a run that will not settle has to be told so on screen, and told
    /// what to do about it, rather than finding it in the console the run was
    /// started from.
    #[test]
    fn the_stall_note_reaches_the_panel_intact() {
        use crate::config::UpdateScheme;
        use crate::constants;
        use crate::engine::stall::stall_note;

        let link = ViewLink::new();
        let console = SilentReporter;
        let reporter = ViewReporter::new(&console, &link);
        let note = stall_note(
            437,
            constants::OC_MOVE_LIMIT,
            constants::DEFAULT_CONVERGENCE_TOL,
            UpdateScheme::Oc,
            false,
        );
        reporter.note(&note);
        let shown = link.progress().note.expect("the panel shows no note");
        assert_eq!(shown, note);
        assert!(shown.contains("stopped after 437 iterations"), "{shown}");
        assert!(shown.contains("update = \"mma\""), "{shown}");
    }

    #[test]
    fn the_status_moves_through_the_run_stages() {
        let link = ViewLink::new();
        assert_eq!(link.progress().status, RunStatus::Optimizing);
        link.set_status(RunStatus::Exporting);
        assert_eq!(link.progress().status, RunStatus::Exporting);
        link.set_status(RunStatus::Finished {
            triangles: 12,
            volume_mm3: 1.0,
            stl_path: "out.stl".to_string(),
        });
        assert!(matches!(
            link.progress().status,
            RunStatus::Finished { triangles: 12, .. }
        ));
    }
}
