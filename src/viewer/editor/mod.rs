//! The visual editor: `growforge edit <config.toml>`.
//!
//! The same window the viewer already opens, in a mode where the problem
//! definition is the document. Objects are picked and dragged in the viewport
//! and edited numerically in the side panel; every committed edit re-validates
//! the configuration, re-voxelizes the setup through the very pipeline
//! `growforge view` uses, and - with auto-regrow on - starts a background run
//! whose density surface is drawn exactly like `run --view`'s preview.
//!
//! ```text
//!   main thread    winit event loop, egui, wgpu, the edit and undo state
//!   worker thread  one engine run: a cancellable preview, or the full pipeline
//!   mesher thread  marching cubes on the newest density snapshot of that run
//! ```
//!
//! Three rules the rest of the module is built around:
//!
//! * **The file on disk is the source of truth.** `edit` reads it once and
//!   writes it only when the user saves. A preview never writes anything at all,
//!   and no run ever writes an STL of its own accord unless it is the full
//!   pipeline reaching its own export; "run full" exports what `growforge run`
//!   would, and "generate stl" exports the design already on screen, which is
//!   the one thing that can put a file on disk after a run was stopped.
//! * **An edit is one undo step.** A drag, a number typed into a field, an
//!   object added or deleted: one step each, restoring the configuration and
//!   the selection together.
//! * **The teardown path is the viewer's own.** The unsaved-changes modal
//!   intercepts the close *request*; once teardown has begun it is the existing
//!   single exit, unchanged.
//!
//! A session can also be replaced without the window going anywhere: "open" and
//! "new" pick another file, and the editor switches to it. That is the same
//! thing as a close as far as the document is concerned - it is left behind -
//! so it is the same guard, one [`Intent`] wider: see [`Editor::request_open`],
//! [`Editor::request_new`] and [`Switch`]. What performs the swap is the window,
//! because only it owns the [`Editor`] being replaced; the session that leaves is
//! stopped and joined before the one that arrives is built, so there is never a
//! second worker behind one window.

pub mod dialog;
pub mod gizmo;
pub mod grid;
pub mod measure;
pub mod pick;
pub mod place;
pub mod snap;
pub mod state;
pub mod toml_io;
pub mod ui;
pub mod worker;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::config::ShapeSpec;
use crate::constants;
use crate::geometry::{Aabb, Vec3, difference, sum};
use crate::report::{print_problem_summary, print_warnings};
use crate::stress::StressSummary;
use crate::viewer::app::ViewerApp;
use crate::viewer::camera;
use crate::viewer::editor::measure::{Callout, Measure};
use crate::viewer::editor::pick::Ray;
use crate::viewer::editor::place::Placing;
use crate::viewer::editor::snap::{Flush, Snap};
use crate::viewer::editor::state::{
    EditorState, NewObject, Selection, set_shape_contained, shape_of,
};
use crate::viewer::editor::worker::{RunKind, Worker};
use crate::viewer::scene::{self, Layer, LayerMesh, Scene, Shading};
use crate::viewer::snapshot::FrameKind;
use crate::viewer::tessellate;

/// Owner of the interaction a viewport drag opens.
///
/// The panel's widgets own theirs by egui id; a drag has no widget, so it takes
/// a reserved one that no egui id can be, keeping "one interaction is one undo
/// step" a single rule rather than two.
const DRAG_INTERACTION: u64 = u64::MAX;

/// What the user chose in the unsaved-changes modal.
///
/// The three answers are the only ones there are, whichever [`Intent`] is being
/// asked about: nothing that leaves the document behind ever silently writes the
/// file or silently drops an edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseDecision {
    /// Write the file, then go on with it.
    Save,
    /// Go on with it without writing.
    Discard,
    /// Stay where we are.
    Cancel,
}

/// What the unsaved-changes guard is guarding: the thing the session does once
/// the document it is leaving behind has been dealt with.
///
/// One modal and one set of answers over all three, because the question is the
/// same one every time - what about the edits that are not in the file? - and
/// answering it in three places is how they come to differ.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Intent {
    /// Tear the window down and end the session.
    CloseWindow,
    /// Edit a file that is already there instead of this one.
    OpenFile(PathBuf),
    /// Scaffold a file that is not there yet and edit that instead.
    NewFile(PathBuf),
}

impl Intent {
    /// What the modal's save button says will happen next.
    pub fn verb(&self) -> &'static str {
        match self {
            Intent::CloseWindow => "close",
            Intent::OpenFile(_) => "open",
            Intent::NewFile(_) => "start the new file",
        }
    }

    /// The file the session would move to, for the modal to name; `None` for a
    /// close, which moves to nothing.
    pub fn destination(&self) -> Option<&Path> {
        match self {
            Intent::CloseWindow => None,
            Intent::OpenFile(path) | Intent::NewFile(path) => Some(path),
        }
    }
}

/// Which file dialog the user asked for.
///
/// Recorded rather than raised where it is asked for: the picker is a blocking
/// native modal and the window is what raises it - see
/// [`dialog`] - so the toolbar button and the keyboard shortcut both reach it
/// through this one record, and neither has to open a dialog inside the frame it
/// was clicked in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pick {
    /// The open dialog: an existing configuration.
    Open,
    /// The new dialog: a name that is not there yet.
    New,
}

/// A file switch the guard has cleared and the window has not performed yet.
///
/// Carries which of the two kinds it is rather than deciding by whether the path
/// exists, because that decision is exactly the one that must not be made: a
/// "new" pointed at a file that is there is a mistake to be told about, never an
/// open, and never an overwrite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Switch {
    path: PathBuf,
    create: bool,
}

impl Switch {
    /// A switch to a file that is already there.
    fn to_open(path: PathBuf) -> Switch {
        Switch {
            path,
            create: false,
        }
    }

    /// A switch to a file that has to be scaffolded first.
    fn to_create(path: PathBuf) -> Switch {
        Switch { path, create: true }
    }

    /// The file the session is moving to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Build the session that replaces the current one.
    ///
    /// The same two constructors `growforge edit` itself uses, chosen by what
    /// was asked for: [`Editor::open`] for an open, [`Editor::create`] - and so
    /// [`EditorState::create`], which refuses to touch a file that is there -
    /// for a new one. The refusal is what gates a path that came back from a
    /// save dialog's own replace prompt.
    pub fn open(&self) -> Result<Editor> {
        if self.create {
            Editor::create(&self.path)
        } else {
            Editor::open(&self.path)
        }
    }
}

/// Which parts of the scene one pump of the editor changed.
///
/// The layers are grouped by [`crate::viewer::scene::LayerRole`], because that
/// is what says who produces them: the setup layers come from the
/// configuration, the density layer from a run, the overlays from the
/// selection.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Refresh {
    /// The setup overlays were rebuilt from the configuration.
    pub setup: bool,
    /// The density surface was replaced or dropped.
    pub density: bool,
    /// The selection shell and the gizmo moved.
    pub overlays: bool,
}

impl Refresh {
    /// True when nothing changed and nothing has to be uploaded.
    pub fn is_empty(self) -> bool {
        !self.setup && !self.density && !self.overlays
    }
}

/// A delay that collapses a burst of edits into one refresh.
///
/// Every committed edit pushes the deadline out, so the expensive work - a
/// re-voxelization and a re-run - happens once the user has stopped rather than
/// once per value a drag passes through.
#[derive(Debug, Default, Clone, Copy)]
pub struct Debounce {
    due: Option<Instant>,
}

impl Debounce {
    /// Restart the delay from `now`.
    pub fn touch(&mut self, now: Instant, delay: Duration) {
        self.due = Some(now + delay);
    }

    /// True once, when the delay has run out.
    pub fn take(&mut self, now: Instant) -> bool {
        match self.due {
            Some(due) if now >= due => {
                self.due = None;
                true
            }
            _ => false,
        }
    }

    /// True while a refresh is owed.
    pub fn is_pending(&self) -> bool {
        self.due.is_some()
    }

    /// Forget the owed refresh.
    pub fn cancel(&mut self) {
        self.due = None;
    }
}

/// The editor session behind the window.
pub struct Editor {
    /// The document being edited.
    pub state: EditorState,
    worker: Worker,
    refresh: Debounce,
    auto_regrow: bool,
    /// Set when an edit landed while a full run had the worker, so the preview
    /// it asked for is started once that run is done.
    regrow_owed: bool,
    drag: Option<gizmo::Drag>,
    /// The two-click placement in progress, which owns the viewport's clicks
    /// while it lives: see [`place`]. `None` is the ordinary state, where a
    /// click selects.
    placing: Option<Placing>,
    /// Where the handle the drag is holding is, when the shape it committed no
    /// longer says - which is a tube's bend and nothing else; see
    /// [`gizmo::Placed::handle_at`]. Lives exactly as long as the drag: the
    /// handle of an object nobody is holding is the shape's own business.
    drag_handle_at: Option<Vec3>,
    /// The grab volumes of the selection's handles, rebuilt with the overlays
    /// they are drawn as. Read by the window's input path.
    pub(crate) handles: Vec<gizmo::Handle>,
    /// What a click would select right now, drawn as a thin outline.
    hover: Option<Selection>,
    /// The handle a click would grab instead, drawn brighter.
    hovered_handle: Option<gizmo::HandleKind>,
    /// The ray the hover above was picked with, so a frame in which neither the
    /// pointer nor the camera moved costs no pick at all.
    hover_from: Option<Ray>,
    /// The hover the overlays currently draw.
    shown_hover: Option<Selection>,
    /// The floor grid, and the domain footprint and increment it was derived
    /// from, so it is re-derived when either moves and not otherwise.
    floor_grid: Option<grid::FloorGrid>,
    shown_grid: Option<(Aabb, f64)>,
    /// The run failure the panel has already been told about, so one failure
    /// produces one status line rather than one per frame.
    reported_failure: Option<String>,
    /// Set once the run that is current has been reported as finished, for the
    /// reason [`Editor::reported_failure`] exists: a finished run stays current
    /// until the next one starts, so without this every frame would rewrite the
    /// line - over anything the session has said since.
    reported_success: bool,
    /// Direction the camera is looking, for the drags that move in its plane.
    view_direction: Vec3,
    /// Where the camera projects world points to, for the callouts: the frame's
    /// view-projection matrix, the viewport it was drawn into in physical
    /// pixels, and the points-per-pixel that turns one into the other.
    projection: Option<(camera::Mat4, (f64, f64), f32)>,
    /// The increments drags land on, as the panel has them.
    snap: Snap,
    /// Set while the bypass key is held, which frees the drag in progress from
    /// them.
    snap_bypass: bool,
    /// Whether non-domain objects are kept inside the domain.
    containment: bool,
    /// When a commit was last moved to keep it inside the domain, which is what
    /// the panel's transient note is timed against.
    clamped_at: Option<Instant>,
    /// The number box of the drag in progress, or of the one that just ended.
    callout: Option<Callout>,
    /// Distance from the domain's floor, while a drag is setting it.
    floor: Option<Measure>,
    /// The surface the drag in progress landed flush against, if it landed on
    /// one. Shown in the callout, because a region that is *on* a face is a
    /// different thing from one that is near it.
    flush: Option<Flush>,
    /// Radius of the scene, which is what floors the gizmo's size.
    scene_radius: f64,
    /// Set when the selection overlays no longer describe the selection.
    overlays_stale: bool,
    shown_selection: Option<Selection>,
    /// The shape the overlays currently describe. Compared against the live one
    /// so they can never be left around a size the object no longer has.
    shown_shape: Option<ShapeSpec>,
    /// What the density layer currently shows, for the panel.
    frame_kind: Option<FrameKind>,
    /// Kind of the run that produced it.
    frame_run: Option<RunKind>,
    /// What the modal is up about, while it is up and the window is waiting on
    /// an answer.
    asking: Option<Intent>,
    /// The user has answered, and the window may go.
    may_close: bool,
    /// The file dialog the window owes the user.
    pick: Option<Pick>,
    /// The file switch the guard has cleared and the window has not performed.
    switch: Option<Switch>,
    status: Option<String>,
}

impl Editor {
    /// Open a configuration file for editing.
    pub fn open(path: &Path) -> Result<Editor> {
        Editor::over(EditorState::open(path)?)
    }

    /// Write a starter configuration at `path` and open it.
    ///
    /// Refuses a path that is already there, because [`EditorState::create`]
    /// does: scaffolding is for a name that means nothing yet.
    pub fn create(path: &Path) -> Result<Editor> {
        Editor::over(EditorState::create(path)?)
    }

    /// Open a configuration file, writing a starter one first when the path is
    /// not there yet.
    ///
    /// What the command line does with the path it was given, and the one place
    /// that decides which of the two happened by looking at the filesystem. The
    /// editor's own "open" and "new" do not come through here: each of them
    /// knows which it is, and a "new" that quietly became an open would be the
    /// bug - see [`Switch::open`].
    pub fn open_or_create(path: &Path) -> Result<Editor> {
        if path.exists() {
            Editor::open(path)
        } else {
            Editor::create(path)
        }
    }

    /// Build the session over a state that is already open.
    fn over(state: EditorState) -> Result<Editor> {
        let auto_regrow = if state.config().is_growth() {
            constants::VIEW_EDIT_AUTO_REGROW_GROWTH
        } else {
            constants::VIEW_EDIT_AUTO_REGROW_SIMP
        };
        Ok(Editor {
            state,
            worker: Worker::new(),
            refresh: Debounce::default(),
            auto_regrow,
            regrow_owed: false,
            drag: None,
            placing: None,
            drag_handle_at: None,
            handles: Vec::new(),
            hover: None,
            hovered_handle: None,
            hover_from: None,
            shown_hover: None,
            floor_grid: None,
            shown_grid: None,
            reported_failure: None,
            reported_success: false,
            view_direction: [0.0, 0.0, -1.0],
            projection: None,
            snap: Snap::default(),
            snap_bypass: false,
            containment: constants::VIEW_EDIT_CONTAINMENT_DEFAULT,
            clamped_at: None,
            callout: None,
            floor: None,
            flush: None,
            scene_radius: constants::VIEW_MIN_SCENE_RADIUS_MM,
            overlays_stale: true,
            shown_selection: None,
            shown_shape: None,
            frame_kind: None,
            frame_run: None,
            asking: None,
            may_close: false,
            pick: None,
            switch: None,
            status: None,
        })
    }

    /// The setup scene the window opens on, empty when the configuration does
    /// not build.
    pub fn initial_scene(&self) -> Scene {
        let mut scene = match self.state.problem() {
            Some(problem) => scene::build(self.state.config(), problem).unwrap_or_default(),
            None => Scene::default(),
        };
        // The floor grid is an editing aid, so the switch that governs it is the
        // editor's. `view` and `run --view` never give the layer geometry and
        // never list it, so nothing there is changed by this.
        *scene.visibility_mut(Layer::Grid) = constants::VIEW_EDIT_GRID_DEFAULT_ON;
        scene
    }

    /// Title of the window, with the modified marker.
    pub fn window_title(&self) -> String {
        let name = self
            .state
            .path()
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| self.state.path().display().to_string());
        format!(
            "{} edit - {}{}",
            constants::VIEW_WINDOW_TITLE,
            name,
            if self.state.is_dirty() { "*" } else { "" }
        )
    }

    /// Whether every committed edit starts a background run.
    pub fn auto_regrow(&self) -> bool {
        self.auto_regrow
    }

    /// Switch auto-regrow on or off. Switching it off cancels the preview in
    /// flight, if any.
    pub fn set_auto_regrow(&mut self, on: bool) {
        self.auto_regrow = on;
        if !on {
            self.worker.cancel_preview();
            self.regrow_owed = false;
        } else {
            self.regrow_owed = true;
        }
    }

    /// True while the full pipeline is running.
    pub fn is_running_full(&self) -> bool {
        self.worker.is_running_full()
    }

    /// True while something that writes the output file is running: the full
    /// pipeline, or an stl generation asked for by the button.
    ///
    /// What gates both of those buttons, because the two would be writing one
    /// path at once, and what defers an auto-regrow preview rather than letting
    /// it take the worker away from a run that has a file to finish.
    pub fn is_writing(&self) -> bool {
        self.worker.is_writing()
    }

    /// True while anything is running behind the window.
    pub fn is_running(&self) -> bool {
        self.worker.is_running()
    }

    /// How the window finds out whether any thread this editor started is
    /// still alive.
    ///
    /// Answered by the run threads themselves, so it covers a preview, a full
    /// run, and a run that has been stopped and is winding down to its next
    /// checkpoint. That last case is the one that matters: [`Editor::is_running`]
    /// answers the panel's question and goes false the moment a run is stopped,
    /// while the thread behind it may still have a stage to leave - and the event
    /// loop must keep servicing the window's message queue until it has.
    ///
    /// Asked of the editor the window holds *now*, every time it is asked. A
    /// session can be replaced mid-flight - see [`Switch`] - and a probe taken
    /// once, when the window was built, would go on answering for a worker that
    /// no longer exists while the one that does runs unwatched.
    pub fn run_probe(&self) -> worker::RunProbe {
        self.worker.probe()
    }

    /// Stop whatever is running, at the user's request.
    ///
    /// Idempotent, and a no-op when nothing is running. A stopped run writes
    /// nothing of its own accord - not even a partial STL - and what is already
    /// on screen stays there: stopping is not undoing.
    ///
    /// The design it had reached stays with it, and "generate stl" is then
    /// offered on that design: see [`Editor::generate_stl`]. Stopping is still
    /// what writes nothing; asking for the file is what writes it.
    pub fn stop_run(&mut self) {
        if !self.worker.is_running() {
            return;
        }
        let what = self.worker.kind().map(RunKind::label).unwrap_or("run");
        self.worker.stop();
        self.regrow_owed = false;
        self.status = Some(format!("{what} stopped; no file was written"));
    }

    /// The last thing that happened worth saying, for the panel.
    pub fn status_line(&self) -> Option<&str> {
        self.worker.startup_error().or(self.status.as_deref())
    }

    /// One line describing the run behind the window, if there is one.
    pub fn run_line(&self) -> Option<String> {
        let kind = self.worker.kind()?;
        let progress = self.worker.progress()?;
        let running = self.worker.is_running();
        let stage = match (&progress.status, running) {
            (crate::viewer::snapshot::RunStatus::Failed(error), _) => format!("failed: {error}"),
            (_, true) => "running".to_string(),
            (_, false) => "done".to_string(),
        };
        let iteration = progress
            .latest
            .as_ref()
            .map(|stats| {
                format!(
                    "  step {} vol {:.4}",
                    stats.iteration, stats.volume_fraction
                )
            })
            .unwrap_or_default();
        Some(format!("{} {stage}{iteration}", kind.label()))
    }

    /// Save the file, reporting either way.
    pub fn save(&mut self) {
        self.status = Some(match self.state.save() {
            Ok(()) => format!("saved {}", self.state.path().display()),
            Err(error) => format!("save failed: {error:#}"),
        });
    }

    /// Undo one step, refreshing everything an edit refreshes.
    pub fn undo(&mut self) {
        // A structural change of the document is one a placement in progress
        // cannot survive; see [`Editor::cancel_placing`].
        self.cancel_placing();
        if self.state.undo() {
            self.after_edit();
        }
    }

    /// Redo one step.
    pub fn redo(&mut self) {
        self.cancel_placing();
        if self.state.redo() {
            self.after_edit();
        }
    }

    /// Delete the selected object.
    ///
    /// The one delete path: the panel's button and the `Delete` key both come
    /// here, so what a delete does to a placement in progress is answered once.
    pub fn delete_selection(&mut self) {
        self.cancel_placing();
        if let Some(selection) = self.state.selection() {
            self.state.delete(selection);
            self.after_edit();
        }
    }

    /// Called by the panel once a frame in which a widget changed something.
    pub fn on_edited(&mut self) {
        self.after_edit();
    }

    /// Everything a committed edit sets in motion.
    fn after_edit(&mut self) {
        self.overlays_stale = true;
        self.refresh.touch(
            Instant::now(),
            Duration::from_secs_f64(constants::VIEW_EDIT_REFRESH_DEBOUNCE_S),
        );
    }

    /// Start the real pipeline, which writes the STL.
    pub fn start_full_run(&mut self) {
        let config = self.state.config().clone();
        let directory = self.state.directory().to_path_buf();
        let warning = self.output_warning(&self.state.output_path());
        // Whatever the last run failed with belongs to that run: a new one that
        // fails the same way is a new thing to be told about, and so is a new one
        // that finishes where the last one finished.
        self.reported_failure = None;
        self.reported_success = false;
        match self.worker.start(&config, &directory, RunKind::Full) {
            Ok(()) => {
                let mut status = format!(
                    "running the full pipeline; it will write {}",
                    config.output.stl_path
                );
                if let Some(warning) = warning {
                    status.push_str(&format!(" - {warning}"));
                }
                self.status = Some(status);
                self.frame_run = Some(RunKind::Full);
            }
            Err(error) => self.status = Some(format!("{error:#}")),
        }
    }

    /// Whether something outside this editor has written the file that is about
    /// to be overwritten at `path`.
    ///
    /// Two editors on two configurations that name the same `stl_path` - or an
    /// editor and a `growforge run` - write to the same place, and nothing in
    /// the program can arbitrate that: the path is the user's own instruction.
    /// What the editor can do is notice, and say so before it overwrites.
    ///
    /// "Ours" is answered by the worker, which records the path and the mtime
    /// of every file one of its runs wrote, at the moment the write is known to
    /// have happened. A second run of this session therefore says nothing about
    /// the file the first one left, and a file that changed underneath us
    /// between the two still does.
    ///
    /// The path is passed in rather than read from the configuration, because
    /// the two writers do not always agree on it: a full run writes where the
    /// configuration says now, and an stl generation where the problem its design
    /// belongs to said then.
    fn output_warning(&self, path: &Path) -> Option<String> {
        let modified = std::fs::metadata(path).ok()?.modified().ok()?;
        if self.worker.wrote_output(path, modified) {
            return None;
        }
        Some(format!(
            "note: {} already exists and was last written outside this editor; writing it now \
             overwrites it",
            path.display()
        ))
    }

    /// Whether the panel offers "generate stl": there is a design on screen and
    /// nothing that writes the output file is running.
    ///
    /// Notably not gated on the configuration being valid. What would be
    /// exported is the design that was computed, together with the problem it was
    /// computed on, so an edit made since - even one that breaks the
    /// configuration outright - cannot make it unexportable.
    pub fn can_generate_stl(&self) -> bool {
        self.worker.can_generate()
    }

    /// Write the deliverables of the design on screen: the enclosed cavity pass,
    /// the stress report, the mesh and the STL.
    ///
    /// The one path on which a stopped run leaves a file behind - and what leaves
    /// it is this, not the stop. A run that is stopped still writes nothing of
    /// its own accord; what it leaves is its newest design, and this is how the
    /// user asks for the file that design would have produced. Which is the
    /// answer to having watched five hundred iterations go by and then discovered
    /// that stopping them threw the part away.
    ///
    /// A guarded no-op when there is nothing to export or something is already
    /// writing, both of which the panel renders as a disabled button.
    pub fn generate_stl(&mut self) {
        let Some(kept) = self.worker.retained() else {
            return;
        };
        // Where that design's own problem says to write, which is not necessarily
        // where the configuration says to now.
        let path = kept.problem().output.stl_path.clone();
        let warning = self.output_warning(&path);
        // Whatever the last run failed with, or wrote, belongs to that run.
        self.reported_failure = None;
        self.reported_success = false;
        if !self.worker.generate() {
            return;
        }
        let mut status = format!("generating {} from the design on screen", path.display());
        if let Some(warning) = warning {
            status.push_str(&format!(" - {warning}"));
        }
        self.status = Some(status);
        self.frame_run = Some(RunKind::Export);
    }

    /// Start a fast preview of the current configuration.
    fn start_preview(&mut self) {
        let Some(problem) = self.state.problem() else {
            return;
        };
        let config = worker::preview_config(self.state.config(), problem);
        let directory = self.state.directory().to_path_buf();
        self.reported_failure = None;
        self.reported_success = false;
        if self
            .worker
            .start(&config, &directory, RunKind::Preview)
            .is_ok()
        {
            self.frame_run = Some(RunKind::Preview);
        }
    }

    /// Tell the editor which way the camera is looking, once per frame. The
    /// handles that move in the camera's plane need it, and only it: how large
    /// the gizmo is comes from the scene, not from the camera.
    pub fn set_view(&mut self, view_direction: Vec3) {
        self.view_direction = view_direction;
    }

    /// Tell the editor where the camera puts world points on the screen, so the
    /// callouts can be anchored to the geometry they measure.
    ///
    /// `viewport` is the 3D view's own size in physical pixels - the surface
    /// less the panel - and `pixels_per_point` is what turns that into the
    /// points egui lays out in. Without this the callouts simply do not draw:
    /// there is nowhere on the screen they belong.
    pub fn set_projection(
        &mut self,
        view_projection: camera::Mat4,
        viewport: (f64, f64),
        pixels_per_point: f32,
    ) {
        self.projection = Some((view_projection, viewport, pixels_per_point));
    }

    /// Where a world point lands on the screen, in egui points from the top
    /// left of the window, or `None` when it is behind the camera or off the
    /// 3D view.
    pub fn project(&self, point: Vec3) -> Option<egui::Pos2> {
        let (matrix, (width, height), pixels_per_point) = self.projection?;
        if width <= 0.0 || height <= 0.0 || pixels_per_point <= 0.0 {
            return None;
        }
        let clip = camera::transform_point(&matrix, point);
        if clip[3] <= 0.0 {
            return None;
        }
        let x = (clip[0] / clip[3] + 1.0) * 0.5 * width;
        let y = (1.0 - clip[1] / clip[3]) * 0.5 * height;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let scale = f64::from(pixels_per_point);
        Some(egui::pos2((x / scale) as f32, (y / scale) as f32))
    }

    /// Width of the 3D view in egui points, which is where a callout has to
    /// stay so it never lands under the side panel.
    pub fn viewport_points(&self) -> Option<(f32, f32)> {
        let (_, (width, height), pixels_per_point) = self.projection?;
        (pixels_per_point > 0.0).then(|| {
            (
                (width / f64::from(pixels_per_point)) as f32,
                (height / f64::from(pixels_per_point)) as f32,
            )
        })
    }

    /// The increments drags land on.
    pub fn snap(&self) -> Snap {
        self.snap
    }

    /// The increments drags land on, for the panel to change.
    pub fn snap_mut(&mut self) -> &mut Snap {
        &mut self.snap
    }

    /// Whether the bypass key is held, which frees a drag from snapping.
    ///
    /// Alt: it is the one modifier neither the camera, the selection, the
    /// shortcuts nor egui's own bindings use, and winit reports it like any
    /// other. Control is the shortcut modifier, and shift is what a future
    /// multi-select would want.
    pub fn set_snap_bypass(&mut self, held: bool) {
        self.snap_bypass = held;
    }

    /// The snapping the drag in progress is actually getting.
    pub fn active_snap(&self) -> Snap {
        if self.snap_bypass {
            Snap::OFF
        } else {
            self.snap
        }
    }

    /// Whether non-domain objects are kept inside the domain.
    pub fn containment(&self) -> bool {
        self.containment
    }

    /// Switch containment on or off. Switching it on does not move anything
    /// that is already outside: it applies to what is committed from here.
    pub fn set_containment(&mut self, on: bool) {
        self.containment = on;
        if !on {
            self.clamped_at = None;
        }
    }

    /// The note the panel shows while a commit that was moved to stay inside
    /// the domain is still recent.
    pub fn containment_note(&self) -> Option<&'static str> {
        let at = self.clamped_at?;
        let linger = Duration::from_secs_f64(constants::VIEW_EDIT_CONTAINMENT_NOTE_S);
        (Instant::now().saturating_duration_since(at) < linger)
            .then_some(constants::VIEW_EDIT_CONTAINMENT_NOTE)
    }

    /// Record that a commit had to be moved to keep it inside the domain.
    pub fn note_clamped(&mut self) {
        self.clamped_at = Some(Instant::now());
    }

    /// Commit a shape onto the selected object, keeping it inside the domain
    /// when that is switched on, and noting it when that moved the commit.
    fn commit_shape(&mut self, selection: Selection, spec: ShapeSpec) {
        let containment = self.containment;
        if set_shape_contained(self.state.config_mut(), selection, spec, containment) {
            self.note_clamped();
        }
    }

    /// The number box of the drag in progress, or of the one that just ended.
    pub fn callout(&self) -> Option<&Callout> {
        self.callout.as_ref()
    }

    /// The same, for the panel that draws and edits it.
    pub fn callout_mut(&mut self) -> Option<&mut Callout> {
        self.callout.as_mut()
    }

    /// Put the callout away, whatever state it was in.
    pub fn dismiss_callout(&mut self) {
        if self.callout.take().is_some() {
            self.floor = None;
            self.flush = None;
            self.overlays_stale = true;
        }
    }

    /// The surface the drag in progress has landed flush against, if any.
    pub fn flush(&self) -> Option<Flush> {
        self.flush
    }

    /// The distance to the domain floor a vertical drag is setting, if any.
    pub fn floor_measure(&self) -> Option<&Measure> {
        self.floor.as_ref()
    }

    /// Apply the value typed into the callout, as one undo step.
    ///
    /// Returns false when what was typed is not a number a configuration may
    /// hold, in which case nothing at all happens and the field keeps it.
    pub fn commit_callout(&mut self, text: &str) -> bool {
        let Some(callout) = &self.callout else {
            return false;
        };
        let selection = callout.selection;
        let Some(spec) = callout.committed(text) else {
            return false;
        };
        if !state::exists(self.state.config(), selection) {
            self.dismiss_callout();
            return false;
        }
        // One whole edit, exactly like a drag: the typed number replaces the
        // drag's own result rather than stacking a second step on top of it.
        self.state.end_edit_any();
        let snapshot_selection = selection;
        let containment = self.containment;
        let clamped = self
            .state
            .edit(|config| set_shape_contained(config, snapshot_selection, spec, containment));
        if clamped {
            self.note_clamped();
        }
        if let Some(callout) = &mut self.callout {
            callout.cancel_typing();
            if let Some(current) = shape_of(self.state.config(), selection)
                && let Some(at) = measure::measure(&callout.handle, &callout.original, &current)
            {
                callout.update(at);
                callout.release(Instant::now());
            }
        }
        self.after_edit();
        true
    }

    /// Refresh the measurement of the drag in progress, and the floor distance
    /// beside it.
    fn refresh_measure(&mut self, current: &ShapeSpec) {
        let Some(drag) = &self.drag else { return };
        // The handle as it is *now*, which for the one handle that can be
        // somewhere the shape does not put it is what the callout has to
        // measure to; every other handle is at the position it was grabbed by,
        // which is what the arithmetic below already reads it as.
        let mut handle = drag.handle();
        if let Some(position) = self.drag_handle_at {
            handle.position = position;
        }
        let original = drag.original().clone();
        let Some(at) = measure::measure(&handle, &original, current) else {
            return;
        };
        match &mut self.callout {
            Some(callout) if callout.handle.kind == handle.kind => callout.update(at),
            _ => {
                let Some(selection) = self.state.selection() else {
                    return;
                };
                self.callout = Some(Callout::new(selection, handle, original, at));
            }
        }
        self.floor = self.floor_for(&handle, current);
        self.overlays_stale = true;
    }

    /// The floor-distance measurement of a translation, when the drag is one
    /// that changes it and the domain has a floor to measure from.
    fn floor_for(&self, handle: &gizmo::Handle, current: &ShapeSpec) -> Option<Measure> {
        let free = match handle.kind {
            gizmo::HandleKind::TranslateFree => true,
            gizmo::HandleKind::Translate(_) => false,
            _ => return None,
        };
        let bounds: Aabb = current.to_shape("measure").ok()?.bounds();
        let floor = state::containment_bounds(self.state.config())?.min[2];
        measure::floor_measure(&bounds, floor, handle.axis, free)
    }

    /// Do the work an edit deferred and take whatever the worker produced.
    ///
    /// Returns which parts of the scene changed, so a drag - which moves its
    /// handles every frame - does not re-upload a density surface nothing
    /// touched.
    pub fn pump(&mut self, scene: &mut Scene) -> Refresh {
        let mut refresh = Refresh::default();
        // First, and before anything below can start a run. A run's failure is
        // readable only through the run itself, and starting the next one
        // replaces it: with this after the auto-regrow blocks, a full run that
        // failed while an edit had queued a preview was overwritten by that
        // preview in the same pump, and the panel never said a word.
        self.note_run_failure();
        // Beside it, and for the same reason it is here rather than lower down:
        // the run that finished is readable only through itself, and the preview
        // an edit queued would replace it in this very pump.
        self.note_run_success();
        if self.refresh.take(Instant::now()) {
            self.rebuild_setup(scene);
            refresh.setup = true;
            refresh.density = true;
            if self.auto_regrow && self.state.is_valid() {
                if self.worker.is_writing() {
                    self.regrow_owed = true;
                } else {
                    self.start_preview();
                }
            }
        }
        if self.regrow_owed && self.auto_regrow && !self.worker.is_writing() {
            self.regrow_owed = false;
            if self.state.is_valid() {
                self.start_preview();
            }
        }
        if let Some(frame) = self.worker.take_frame() {
            self.frame_kind = Some(frame.kind);
            self.frame_run = self.worker.kind();
            scene.set(Layer::Density, Some(frame.mesh));
            scene.set_stress(frame.stress);
            refresh.density = true;
        }
        // The grid follows the domain and the snap control, both of which the
        // panel can move between two frames.
        if self.refresh_grid(scene) {
            refresh.overlays = true;
        }
        // A number that has been on screen long enough goes away, taking the
        // dimension lines with it.
        if self.callout.as_ref().is_some_and(|callout| {
            callout.is_expired(
                Instant::now(),
                Duration::from_secs_f64(constants::VIEW_EDIT_CALLOUT_LINGER_S),
            )
        }) {
            self.dismiss_callout();
        }
        // A callout that no longer addresses anything - the object was deleted,
        // an undo took it away - has nothing to measure.
        if let Some(callout) = &self.callout
            && !state::exists(self.state.config(), callout.selection)
        {
            self.dismiss_callout();
        }
        // The shell and the handles are rebuilt whenever they no longer
        // describe what is selected - compared against the shape itself, not
        // against a flag somebody has to remember to set. A resize that ends
        // smaller than it started must not leave the shell around the size it
        // passed through.
        let selection = self.state.selection();
        let shape = selection.and_then(|s| shape_of(self.state.config(), s));
        if self.overlays_stale
            || self.shown_selection != selection
            || self.shown_shape != shape
            || self.shown_hover != self.hover
        {
            self.rebuild_overlays(scene);
            refresh.overlays = true;
        }
        refresh
    }

    /// Re-validate and rebuild the setup overlays from the edited
    /// configuration, through the pipeline `growforge view` uses.
    fn rebuild_setup(&mut self, scene: &mut Scene) {
        self.state.revalidate();
        if let Some(problem) = self.state.problem()
            && let Ok(built) = scene::build(self.state.config(), problem)
        {
            scene.adopt_setup(&built);
        }
        // Whatever design was on screen belonged to the configuration as it
        // was; showing it against the new setup would claim a result the editor
        // does not have.
        scene.clear_density();
        self.frame_kind = None;
        self.overlays_stale = true;
    }

    /// Rebuild the selection shell and the gizmo from the selected object's
    /// current geometry.
    ///
    /// Everything here is derived from the shape as it is at this instant - the
    /// handle positions, the gizmo's length, the shell's margin - so a drag
    /// that is still moving, a number just typed, an undo and an add all get
    /// overlays of the right size on the next frame.
    fn rebuild_overlays(&mut self, scene: &mut Scene) {
        self.overlays_stale = false;
        self.shown_selection = self.state.selection();
        self.shown_shape = self
            .shown_selection
            .and_then(|s| shape_of(self.state.config(), s));
        self.handles.clear();
        scene.set(Layer::Selection, None);
        scene.set(Layer::Hover, None);
        scene.set(Layer::Gizmo, None);
        scene.set(Layer::Measure, None);
        scene.set(Layer::Placement, None);
        let (_, radius) = camera::bounding_sphere(&scene.bounds());
        self.scene_radius = radius;
        // A placement in progress draws itself and nothing else. The selection
        // it suspended keeps its place in the state - cancelling brings its
        // overlays straight back - but nothing of it is drawn while the clicks
        // belong to the placement, because a gizmo that cannot be grabbed is a
        // handle that lies about what a press would do.
        if let Some(placing) = &self.placing {
            scene.set(
                Layer::Placement,
                placing.overlay(
                    state::default_tube_radius(self.state.config()),
                    self.scene_radius * constants::VIEW_EDIT_PLACE_MARKER_SCENE_FRACTION,
                ),
            );
            // What the hover would have drawn, so the next pump does not find
            // the overlays out of date and rebuild this one on every frame.
            self.shown_hover = self.hover;
            return;
        }
        self.rebuild_hover(scene);

        let Some(selection) = self.state.selection() else {
            return;
        };
        let Some(spec) = shape_of(self.state.config(), selection) else {
            return;
        };
        let Ok(shape) = spec.to_shape("selection") else {
            // A half-typed shape has no geometry to draw handles on; the
            // properties panel is where it gets fixed.
            return;
        };
        let length = gizmo::gizmo_length(gizmo::shape_radius(&shape), self.scene_radius);
        self.handles = gizmo::handles(&spec, length);
        // One handle can be somewhere the committed shape does not put it, and
        // only while it is held: see [`gizmo::Placed::handle_at`].
        if let (Some(drag), Some(position)) = (&self.drag, self.drag_handle_at) {
            gizmo::reposition(&mut self.handles, drag.handle().kind, position);
        }
        let shell = tessellate::shape(&gizmo::inflated(
            &shape,
            gizmo::selection_margin(&shape, length),
        ));
        scene.set(
            Layer::Selection,
            Some(LayerMesh::from_mesh(
                &shell,
                Layer::Selection.info().color,
                Shading::Rounded,
            )),
        );
        scene.set(
            Layer::Gizmo,
            Some(gizmo::mesh(
                &self.handles,
                length,
                gizmo::anchor(&spec),
                self.hovered_handle,
            )),
        );
        // The dimension lines of the drag in progress, or of the one still
        // lingering: drawn only while there is a number on screen to go with
        // them.
        if let Some(callout) = &self.callout
            && callout.selection == selection
        {
            scene.set(
                Layer::Measure,
                measure::overlay(&callout.at, self.floor.as_ref(), length),
            );
        }
    }

    /// Draw the thin outline around whatever a click would select.
    ///
    /// Nothing is drawn around the *selected* object: it already has a shell of
    /// its own, and a second one inside it would only be two colours in the same
    /// place. The shell itself is the selection machinery's, at the hover
    /// margin, so the two can never disagree about what a shape's outline is.
    fn rebuild_hover(&mut self, scene: &mut Scene) {
        self.shown_hover = self.hover;
        let Some(hover) = self
            .hover
            .filter(|hover| Some(*hover) != self.state.selection())
        else {
            return;
        };
        let Some(spec) = shape_of(self.state.config(), hover) else {
            return;
        };
        let Ok(shape) = spec.to_shape("hover") else {
            return;
        };
        let length = gizmo::gizmo_length(gizmo::shape_radius(&shape), self.scene_radius);
        let outline = tessellate::shape(&gizmo::inflated(
            &shape,
            gizmo::hover_margin(&shape, length),
        ));
        scene.set(
            Layer::Hover,
            Some(LayerMesh::from_mesh(
                &outline,
                Layer::Hover.info().color,
                Shading::Rounded,
            )),
        );
    }

    /// The placement in progress, for the panel and the tests.
    pub fn placing(&self) -> Option<&Placing> {
        self.placing.as_ref()
    }

    /// True while a placement owns the viewport's clicks.
    pub fn is_placing(&self) -> bool {
        self.placing.is_some()
    }

    /// What the panel says while it does: which list the object will land in,
    /// and which click is next.
    pub fn placement_hint(&self) -> Option<String> {
        let placing = self.placing.as_ref()?;
        Some(format!(
            "placing a {} in {}: {}",
            placing.what.shape_label(),
            placing.what.list_label(),
            placing.hint()
        ))
    }

    /// Start placing `what` by clicking its points in the viewport, or - when
    /// that is already what is being placed - stop.
    ///
    /// The button that opens the mode is the button that closes it, because it
    /// is the same button: an add row that has become a mode has to be able to
    /// give the row back. Asking for a *different* target while one is in
    /// progress starts that one instead, points and all, which is what clicking
    /// another list's row means.
    pub fn toggle_placing(&mut self, what: NewObject) {
        if self
            .placing
            .as_ref()
            .is_some_and(|placing| placing.what == what)
        {
            self.cancel_placing();
            return;
        }
        // The mode owns the viewport from here: whatever the last interaction
        // left on screen belongs to a click that no longer selects.
        self.dismiss_callout();
        // A drag cannot be in flight while the panel is being clicked - the
        // press that started one went to the viewport, and egui takes the panel's
        // clicks before the editor sees them - but a drag dropped without being
        // closed out would leave its undo step half open, so it is closed the way
        // a release closes it.
        if let Some(drag) = self.drag.take()
            && drag.has_changed()
        {
            self.state.end_edit(DRAG_INTERACTION);
        }
        self.drag_handle_at = None;
        self.clear_hover();
        self.placing = Some(Placing::new(what));
        self.overlays_stale = true;
    }

    /// Leave the placement mode, discarding a point it had already taken.
    /// Returns whether there was one to leave.
    ///
    /// The selection it suspended is still the selection: cancelling puts the
    /// overlays of whatever was selected before back exactly as they were - and
    /// that promise is why **a structural change of the document cancels the
    /// mode first**. Deleting the selected object, undoing or redoing while
    /// points are being clicked would leave the mode holding a restoration that
    /// no longer exists, so [`Editor::delete_selection`], [`Editor::undo`] and
    /// [`Editor::redo`] come through here on their way. The mode costs one
    /// button click to re-enter, which is the cheaper of the two prices.
    ///
    /// A *numeric* edit is deliberately not one of them: the properties panel's
    /// fields change values rather than structure, they cannot invalidate a
    /// clicked point - which is a position in the world, not a reference to an
    /// object - and a placement aimed at geometry the panel is still adjusting
    /// is a reasonable thing to be doing.
    pub fn cancel_placing(&mut self) -> bool {
        if self.placing.take().is_none() {
            return false;
        }
        self.clear_hover();
        self.overlays_stale = true;
        true
    }

    /// Where a placement click along `ray` would land, or `None` when it would
    /// land nowhere at all.
    pub fn landing_at(&self, ray: &Ray) -> Option<Vec3> {
        place::landing(
            ray,
            self.state.config(),
            self.floor_grid.as_ref(),
            self.active_snap(),
        )
    }

    /// One click of a placement in progress.
    ///
    /// A click that lands nowhere - at the sky, or past the ruled floor - and a
    /// second click that lands on the first one both leave the placement
    /// exactly as it was: there is nothing to record, and dropping out of the
    /// mode would be answering a mis-aim with a cancelled gesture.
    fn place_click(&mut self, ray: &Ray) {
        let Some(at) = self.landing_at(ray) else {
            return;
        };
        let snap = self.active_snap();
        let Some(placing) = self.placing.as_mut() else {
            return;
        };
        let Some(first) = placing.first else {
            placing.first = Some(at);
            placing.at = Some(at);
            self.overlays_stale = true;
            return;
        };
        if place::coincident(first, at, snap) {
            return;
        }
        let what = placing.what;
        self.finish_placing(what, first, at);
    }

    /// The second click: add the object the two points make, select it, and
    /// leave the mode.
    fn finish_placing(&mut self, what: NewObject, p1: Vec3, p2: Vec3) {
        let radius = state::default_tube_radius(self.state.config());
        let spec = place::tube_between(p1, p2, radius);
        if self.state.add_placed(what, spec, self.containment) {
            self.note_clamped();
        }
        self.placing = None;
        // The pointer is over an object again rather than over a landing point,
        // and the next frame picks it afresh.
        self.clear_hover();
        self.after_edit();
    }

    /// A left press in the viewport. Returns true when it grabbed a handle, in
    /// which case the camera must not treat it as the start of an orbit.
    ///
    /// A placement in progress owns the click: no handle is grabbed and no drag
    /// starts, so the press is the camera's to orbit with and the release is the
    /// placement's to take a point from.
    pub fn press(&mut self, ray: &Ray) -> bool {
        if self.placing.is_some() {
            return false;
        }
        let Some(handle) = gizmo::grab(ray, &self.handles) else {
            return false;
        };
        let Some(selection) = self.state.selection() else {
            return false;
        };
        let Some(spec) = shape_of(self.state.config(), selection) else {
            return false;
        };
        // A new grab replaces whatever the last drag left on screen.
        self.dismiss_callout();
        self.drag = gizmo::Drag::start(handle, spec.clone(), ray, self.view_direction);
        self.flush = None;
        // Whatever the last drag was holding, this one starts from the shape.
        self.drag_handle_at = None;
        if self.drag.is_some() {
            // A drag owns the pointer from here: nothing under it is a click
            // target any more, so the outline goes at the grab rather than at
            // the first frame after it.
            self.clear_hover();
            // The number appears with the grab, showing what the handle is
            // about to change rather than waiting for it to change.
            self.refresh_measure(&spec);
        }
        self.drag.is_some()
    }

    /// The pointer moved while a handle is held.
    pub fn drag_to(&mut self, ray: &Ray) {
        let snap = self.active_snap();
        let Some(selection) = self.state.selection() else {
            return;
        };
        // The faces this object may land flush on, which for anything but a
        // support or a load region is none at all.
        let surfaces = state::surfaces(self.state.config(), selection);
        let Some(drag) = &mut self.drag else {
            return;
        };
        let Some(placed) = drag.placed_at(ray, snap, &surfaces) else {
            return;
        };
        let (shape, flush, handle_at) = (placed.shape, placed.flush, placed.handle_at);
        // Nothing has happened until a frame asks for a shape the drag did not
        // start on, and until then this does nothing at all.
        //
        // A resize still inside its dead zone asks for exactly the shape it
        // started on ([`gizmo::Drag::placed_at`]), and so does any drag the
        // increment rounds back to where it began. Committing that would be a
        // write of what is already there, and *opening an interaction* for it
        // would record an undo step that undoes nothing and **throw the redo
        // stack away** - so a click on a resize handle with an unsteady hand
        // would cost the user the redo they were part way through. Returning
        // instead of committing also leaves the containment clamp to the first
        // real frame, where an undo step covers it: an object loaded outside the
        // domain is not quietly pulled in by a click.
        //
        // Only while the drag has changed nothing. Once it has, every frame is
        // committed, including one that lands back on the original - a drag that
        // returns to where it started has to put the shape back.
        if !drag.has_changed() {
            if shape == *drag.original() {
                return;
            }
            drag.mark_changed();
            self.state.begin_edit(DRAG_INTERACTION);
        }
        self.flush = flush;
        let asked = gizmo::anchor(&shape);
        self.commit_shape(selection, shape);
        // Measured from what was really committed, so a drag stopped at the
        // domain wall reads the distance it actually covered.
        self.drag_handle_at = None;
        if let Some(committed) = shape_of(self.state.config(), selection) {
            // A live handle position is measured against that same committed
            // shape, so it is carried by whatever containment moved the object
            // by - which is a translation and nothing else, so this is exact.
            self.drag_handle_at =
                handle_at.map(|point| sum(point, difference(gizmo::anchor(&committed), asked)));
            self.refresh_measure(&committed);
        }
        self.after_edit();
    }

    /// The left button came up. `click` is set when the press and the release
    /// were the same gesture rather than a drag, which is what tells a
    /// selection click from an orbit.
    ///
    /// While a placement is in progress the click is its point rather than a
    /// selection - and a click with no ray is a click outside the 3D view,
    /// which places nothing for the reason it selects nothing.
    pub fn release(&mut self, ray: Option<&Ray>, click: bool) {
        if self.placing.is_some() {
            if let (Some(ray), true) = (ray, click) {
                self.place_click(ray);
            }
            return;
        }
        if let Some(drag) = self.drag.take() {
            // Nobody is holding the handle any more, so it goes back to where
            // the shape says it is.
            self.drag_handle_at = None;
            if drag.has_changed() {
                // The whole drag is one step, recorded now that it is over.
                self.state.end_edit(DRAG_INTERACTION);
            }
            // The number stays up to be read, and to be typed over.
            if let Some(callout) = &mut self.callout {
                callout.release(Instant::now());
            }
            self.overlays_stale = true;
            return;
        }
        if !click {
            return;
        }
        if let Some(ray) = ray {
            let picked = pick::nearest(ray, &state::targets(self.state.config()));
            if picked != self.state.selection() {
                self.dismiss_callout();
            }
            self.state.select(picked);
        }
    }

    /// True while a handle is held, so the camera stays out of the way.
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Point the hover highlight at whatever is under the pointer.
    ///
    /// `ray` is the very ray a click would be cast along, and `None` for every
    /// reason there is nothing to hover: the pointer is over the side panel, off
    /// the window, or busy with a drag the window is tracking. A drag of the
    /// editor's own suppresses it here rather than at the caller, so no input
    /// path can forget to.
    ///
    /// What is picked is picked by [`pick::nearest`] against
    /// [`state::targets`] - the same ray, the same rank rule, the same target
    /// list a click uses - so the outline is a preview of the click and not an
    /// approximation of one. The domain is not in that list and so is never
    /// hovered, for the reason it is never clicked.
    ///
    /// A pointer over a gizmo handle hovers no object at all: the click there
    /// would grab the handle, and that is what the brightened handle says.
    pub fn hover_to(&mut self, ray: Option<Ray>) {
        let ray = if self.drag.is_some() { None } else { ray };
        if ray == self.hover_from {
            return;
        }
        self.hover_from = ray;
        // A placement owns the pointer as well as the clicks: what is under it
        // is where the next click lands, not what a click would select, so the
        // ray goes to the landing point and the outline machinery is left as it
        // was.
        if self.placing.is_some() {
            let at = ray.and_then(|ray| self.landing_at(&ray));
            if let Some(placing) = self.placing.as_mut()
                && placing.at != at
            {
                placing.at = at;
                self.overlays_stale = true;
            }
            return;
        }
        let (hover, handle) = match ray {
            Some(ray) => match gizmo::grab(&ray, &self.handles) {
                Some(handle) => (None, Some(handle.kind)),
                None => (
                    pick::nearest(&ray, &state::targets(self.state.config())),
                    None,
                ),
            },
            None => (None, None),
        };
        if hover != self.hover || handle != self.hovered_handle {
            self.hover = hover;
            self.hovered_handle = handle;
            self.overlays_stale = true;
        }
    }

    /// Forget whatever is hovered, and the ray it was picked with, so the next
    /// pointer position is picked afresh.
    fn clear_hover(&mut self) {
        self.hover_from = None;
        if self.hover.is_some() || self.hovered_handle.is_some() {
            self.hover = None;
            self.hovered_handle = None;
            self.overlays_stale = true;
        }
    }

    /// What a click would select right now, if anything.
    pub fn hover(&self) -> Option<Selection> {
        self.hover
    }

    /// The gizmo handle a click would grab right now, if any.
    pub fn hovered_handle(&self) -> Option<gizmo::HandleKind> {
        self.hovered_handle
    }

    /// The floor grid as it currently stands, for the panel and the tests.
    pub fn floor_grid(&self) -> Option<&grid::FloorGrid> {
        self.floor_grid.as_ref()
    }

    /// Re-derive the floor grid when the domain footprint or the snap increment
    /// has moved, and return whether it did.
    ///
    /// Cheap to ask and cheap to skip: the comparison is one box and one number,
    /// and the derivation only runs when one of them changed - which is an edit
    /// of the domain or a turn of the panel's snap control, not a frame.
    fn refresh_grid(&mut self, scene: &mut Scene) -> bool {
        let domain = state::containment_bounds(self.state.config());
        let increment = self.snap.millimetres;
        let wanted = domain.map(|bounds| (bounds, increment));
        if wanted == self.shown_grid {
            return false;
        }
        self.shown_grid = wanted;
        self.floor_grid =
            wanted.and_then(|(bounds, increment)| grid::FloorGrid::derive(&bounds, increment));
        scene.set(
            Layer::Grid,
            self.floor_grid.as_ref().and_then(grid::FloorGrid::mesh),
        );
        true
    }

    /// Notice a run that failed, once, and say so in the panel.
    ///
    /// A failed run is the run's failure and not the session's: the window stays
    /// open, the document is untouched, and the next run starts from the panel
    /// exactly as the last one did. What the session owes the user is the
    /// solver's own message, which names the fix.
    fn note_run_failure(&mut self) {
        let Some(progress) = self.worker.progress() else {
            return;
        };
        let crate::viewer::snapshot::RunStatus::Failed(reason) = &progress.status else {
            return;
        };
        if self.reported_failure.as_deref() == Some(reason.as_str()) {
            return;
        }
        self.reported_failure = Some(reason.clone());
        let what = self.worker.kind().map(RunKind::label).unwrap_or("run");
        self.status = Some(format!(
            "{what} failed and wrote nothing: {reason} - the session is unaffected; change the \
             configuration and run again"
        ));
    }

    /// Notice a run that finished on its own, once, and say so in the panel.
    ///
    /// The counterpart of [`Editor::note_run_failure`], and the same shape: a run
    /// that ends says nothing of itself, so the line the *start* of it wrote -
    /// "running the full pipeline; it will write ..." - stayed on the panel for
    /// the rest of the session, describing a run that was over and a file that
    /// was already there. The run line beside it moves to "done" on its own and
    /// is not what this repeats: what this says is that the file was written, and
    /// where.
    ///
    /// Only the two kinds that write ever put a run on the status line, so only
    /// they have a finish to report; a preview writes nothing, says nothing when
    /// it starts, and never reaches
    /// [`crate::viewer::snapshot::RunStatus::Finished`] at all.
    fn note_run_success(&mut self) {
        if self.reported_success {
            return;
        }
        let Some(progress) = self.worker.progress() else {
            return;
        };
        let crate::viewer::snapshot::RunStatus::Finished { stl_path, .. } = &progress.status else {
            return;
        };
        // The path the run resolved and wrote, which for a generation is the one
        // its own design belongs to rather than the one the configuration names
        // now.
        let status = match self.worker.kind() {
            Some(RunKind::Full) => format!("the full pipeline finished; it wrote {stl_path}"),
            Some(RunKind::Export) => format!("generated {stl_path} from the design on screen"),
            _ => return,
        };
        self.reported_success = true;
        self.status = Some(status);
    }

    /// Close whatever interaction the panel left open once the pointer and the
    /// keyboard have moved on, so a value is one undo step rather than one per
    /// frame it was touched in.
    pub fn settle(&mut self, context: &egui::Context) {
        if !self.state.is_editing() || self.drag.is_some() {
            return;
        }
        if !context.egui_is_using_pointer() && !context.egui_wants_keyboard_input() {
            self.state.end_edit_any();
        }
    }

    /// Keyboard shortcuts. Skipped entirely while the guard's question is up,
    /// and while a text field has the keyboard.
    pub fn shortcuts(&mut self, context: &egui::Context) {
        // Every binding, not a list of the dangerous ones. The modal is a layer
        // over the panel and takes every *click* that is not one of its own
        // buttons, but egui routes keys by focus rather than by layer, and with
        // nothing focused behind it none of these would be suppressed by that.
        // Undo, redo, delete and save all change the very document the question
        // is about; Ctrl+O and Ctrl+N would put a second question over the
        // first. One rule over all of them beats an exemption list somebody has
        // to maintain as bindings are added - and the modal's own buttons, which
        // are mouse-driven, remain the way it is answered.
        if self.is_asking() {
            return;
        }
        if context.egui_wants_keyboard_input() {
            return;
        }
        let (save, open, new, undo, redo, delete, escape) = context.input(|input| {
            let command = input.modifiers.command;
            (
                command && input.key_pressed(egui::Key::S),
                command && input.key_pressed(egui::Key::O),
                command && input.key_pressed(egui::Key::N),
                command && !input.modifiers.shift && input.key_pressed(egui::Key::Z),
                command
                    && (input.key_pressed(egui::Key::Y)
                        || (input.modifiers.shift && input.key_pressed(egui::Key::Z))),
                input.key_pressed(egui::Key::Delete),
                input.key_pressed(egui::Key::Escape),
            )
        });
        // Before anything else it could mean: while a placement is in progress
        // Escape is what leaves it, at either of its two stages.
        if escape {
            self.cancel_placing();
        }
        if save {
            self.save();
        }
        // The two that leave this file behind ask for a dialog and nothing more;
        // what happens to whatever comes back is the guard's business, exactly
        // as it is for the buttons these share.
        if open {
            self.ask_to_open();
        }
        if new {
            self.ask_for_new();
        }
        if undo {
            self.undo();
        }
        if redo {
            self.redo();
        }
        if delete {
            self.delete_selection();
        }
    }

    /// A close request arrived. Returns true when the window may go now.
    ///
    /// Unsaved changes put the modal up instead, and the answer comes back
    /// through [`Editor::decide`]; the window is not torn down until then.
    pub fn request_close(&mut self) -> bool {
        self.request(Intent::CloseWindow)
    }

    /// Ask to edit `path` instead of the file this session is on.
    ///
    /// The same guard the close goes through, for the same reason: a switch
    /// leaves the document behind exactly as closing the window does. Returns
    /// true when it may happen - which is not the same as done, because only the
    /// window can replace the session: see [`Editor::take_switch`].
    pub fn request_open(&mut self, path: PathBuf) -> bool {
        self.request(Intent::OpenFile(path))
    }

    /// Ask to scaffold `path` and edit that instead of the file this session is
    /// on.
    ///
    /// Refused outright, with the reason on the status line, when the path is
    /// already there: a "new" is not a way to open a file and never a way to
    /// overwrite one. Refused here so that a mistake costs the session nothing -
    /// nothing is asked, nothing is stopped - and refused again by the scaffold
    /// itself when the new session is built, which is what covers a file that
    /// appears in between and a save dialog that offered to replace one.
    pub fn request_new(&mut self, path: PathBuf) -> bool {
        // Before the refusal too, so a question on screen is not even written
        // over on the status line: see [`Editor::request`].
        if self.asking.is_some() {
            return false;
        }
        if path.exists() {
            self.status = Some(format!(
                "{} {}",
                path.display(),
                constants::VIEW_EDIT_NEW_EXISTS_NOTE
            ));
            return false;
        }
        self.request(Intent::NewFile(path))
    }

    /// Put `intent` through the unsaved-changes guard. True when it may happen
    /// now, false when the modal is up - asking about this intent, or already
    /// asking about another one.
    fn request(&mut self, intent: Intent) -> bool {
        // A question already on screen is the one being answered, and it is
        // immutable until it is. Nothing may replace it - not a repeat of the
        // same intent, and above all not a different one: what the user is about
        // to answer is what they can read, and a modal that had quietly become a
        // question about something else would apply "discard" to the wrong
        // thing. The one way past it is [`Editor::decide`].
        if self.asking.is_some() {
            return false;
        }
        if self.state.is_dirty() {
            self.asking = Some(intent);
            return false;
        }
        // A clean document is left behind without a question. The close is the
        // one intent the editor does not carry out itself - the window does, on
        // this `true` - and latching `may_close` for it here would leave a flag
        // set for a close that has already been taken, which a later cancel
        // could not clear.
        if !matches!(intent, Intent::CloseWindow) {
            self.grant(intent);
        }
        true
    }

    /// The guard is clear: start the thing it was guarding.
    fn grant(&mut self, intent: Intent) {
        match intent {
            Intent::CloseWindow => self.may_close = true,
            Intent::OpenFile(path) => self.switch = Some(Switch::to_open(path)),
            Intent::NewFile(path) => self.switch = Some(Switch::to_create(path)),
        }
    }

    /// True while the modal is up.
    pub fn is_asking(&self) -> bool {
        self.asking.is_some()
    }

    /// What the modal is asking about, while it is up.
    pub fn asking(&self) -> Option<&Intent> {
        self.asking.as_ref()
    }

    /// True once the user has agreed to close.
    pub fn may_close(&self) -> bool {
        self.may_close
    }

    /// Record the answer to the modal.
    pub fn decide(&mut self, decision: CloseDecision) {
        let Some(intent) = self.asking.take() else {
            return;
        };
        match decision {
            CloseDecision::Save => {
                self.save();
                if self.state.is_dirty() {
                    // A save that failed is not an answer: the edits are still
                    // there, the question still stands, and the panel says why.
                    self.asking = Some(intent);
                } else {
                    self.grant(intent);
                }
            }
            CloseDecision::Discard => self.grant(intent),
            CloseDecision::Cancel => {}
        }
    }

    /// The user asked for a file to open; the window raises the picker.
    ///
    /// What the toolbar's "open" button and `Ctrl+O` both do, and all they do:
    /// see [`Pick`].
    pub fn ask_to_open(&mut self) {
        self.ask_for(Pick::Open);
    }

    /// The same for a new file: the toolbar's "new" button and `Ctrl+N`.
    pub fn ask_for_new(&mut self) {
        self.ask_for(Pick::New);
    }

    /// Record the dialog the window owes, unless the guard already has a
    /// question up: there is one question on screen at a time, and it is the one
    /// that has to be answered before anything else happens to this document.
    fn ask_for(&mut self, pick: Pick) {
        if self.is_asking() {
            return;
        }
        self.pick = Some(pick);
    }

    /// The dialog the window owes the user, if it owes one. Nothing is owed
    /// while the guard's question is up.
    pub fn pending_pick(&self) -> Option<Pick> {
        if self.is_asking() { None } else { self.pick }
    }

    /// Take the dialog the window owes the user - once, so a click raises one
    /// picker rather than one per frame it takes to answer it.
    ///
    /// Nothing is owed while the guard's question is up, and a request the
    /// question outlived is dropped rather than kept: the modal is newer than
    /// it, and answering "discard and close" must not then be met with a file
    /// dialog on the way out.
    pub fn take_pick(&mut self) -> Option<Pick> {
        if self.is_asking() {
            self.pick = None;
            return None;
        }
        self.pick.take()
    }

    /// Take the file switch the guard has cleared, if it has cleared one.
    ///
    /// Collected by the window, which is the only thing that can replace the
    /// session: this editor is what is being replaced.
    pub fn take_switch(&mut self) -> Option<Switch> {
        self.switch.take()
    }

    /// True while a switch is waiting for the window to perform it.
    pub fn is_switching(&self) -> bool {
        self.switch.is_some()
    }

    /// Say something in the panel's status line.
    ///
    /// What a switch that could not happen reports through: the session it would
    /// have replaced is the one still there to report it.
    pub fn set_status(&mut self, message: String) {
        self.status = Some(message);
    }

    /// Draw the editor's panel, the callouts floating over the viewport, and -
    /// when it is up - its modal.
    pub fn draw(&mut self, root: &mut egui::Ui, scene: &mut Scene, adapter: &str) {
        // The width overflow it measures is the guard test's business: a window
        // cannot widen itself out of a row that does not fit, and the panel is
        // drawn the same either way.
        let _ = ui::panel(root, self, scene, adapter);
        // Not while the modal is up: a floating number box over a window that
        // is asking whether to save would be one more thing to click on in a
        // moment where there is exactly one question.
        if self.asking.is_none() {
            ui::callouts(root.ctx(), self);
        }
        let decision = self.asking.as_ref().and_then(|intent| {
            let path = self.state.path().display().to_string();
            ui::guard_modal(root.ctx(), intent, &path, self.worker.is_running())
        });
        if let Some(decision) = decision {
            self.decide(decision);
        }
    }

    /// The window is going away, and in edit mode the window *is* the session:
    /// every run behind it stops.
    ///
    /// This is deliberately not what `run --view` does. There, closing the
    /// window detaches a run that was asked for on the command line and that
    /// finishes and writes its file headless. Here the run was asked for inside
    /// the window that is being closed, the process has nothing else to do, and
    /// a growth or optimization carrying on invisibly - writing an STL another
    /// editor may be about to write too - is exactly what a closed window must
    /// not leave behind. A stopped run writes nothing, and with the window gone
    /// there is nobody left to ask for the design it stopped on either.
    pub fn detach(&mut self) {
        self.refresh.cancel();
        self.worker.stop();
    }

    /// Collect the threads of whatever was running. Called after the event loop
    /// stops, which - because the loop watches [`Editor::run_probe`] - is after
    /// they have ended: this makes it certain rather than likely, and it never
    /// waits with a window still to service.
    ///
    /// Also called, after [`Editor::detach`], when a file switch replaces this
    /// session: there the wait is real and the window is frozen for it, because
    /// the session that arrives may not start a run while the one that is
    /// leaving still has a thread in one.
    pub fn finish(&mut self) {
        self.worker.join();
    }

    /// The engine's last one-off message about the run behind the window, for
    /// the panel.
    ///
    /// The same line `growforge run --view` has always shown under its stats
    /// block, and the one place the editor learns *why* a run ended: converged
    /// at an iteration, stopped because the problem would not settle and what to
    /// do about that, or stopped on the budget.
    pub fn run_note(&self) -> Option<String> {
        self.worker.progress()?.note
    }

    /// What the `[output] trim` pass said about the run behind the window, for
    /// the panel.
    ///
    /// Empty until a run that writes has finished one, and empty for good under
    /// the default `trim = "off"`. The console gets the same lines from the
    /// outcome; this is the other half of "a trim is reported in both places".
    pub fn trim_notes(&self) -> Vec<String> {
        self.worker
            .progress()
            .map(|progress| progress.trim)
            .unwrap_or_default()
    }

    /// What the `[output] flush` pass said about the run behind the window, for
    /// the panel.
    ///
    /// Empty until a run that writes has finished one, and empty for good under
    /// the default `flush = "off"`. The console gets the same lines from the
    /// outcome, and the panel draws them between the trim's and the
    /// reinforcement's: what a run freed, what it put back out to the surfaces
    /// the walls rest on, and what it spent.
    pub fn flush_notes(&self) -> Vec<String> {
        self.worker
            .progress()
            .map(|progress| progress.flush)
            .unwrap_or_default()
    }

    /// What the `[output] reinforce` pass said about the run behind the window,
    /// for the panel.
    ///
    /// Empty until a run that writes has finished one, and empty for good under
    /// the default `reinforce = "off"`. The console gets the same lines from the
    /// outcome, and the panel draws them beside the trim's: what a run freed and
    /// what it spent, in one block.
    pub fn reinforce_notes(&self) -> Vec<String> {
        self.worker
            .progress()
            .map(|progress| progress.reinforce)
            .unwrap_or_default()
    }

    /// What the `[output] boundaries` clamp said about the run behind the
    /// window, for the panel.
    ///
    /// Empty until a run that writes has finished one, and empty for good under
    /// `boundaries = "voxel"`. The console gets the same lines from the outcome.
    pub fn clamp_notes(&self) -> Vec<String> {
        self.worker
            .progress()
            .map(|progress| progress.clamp)
            .unwrap_or_default()
    }

    /// What the part behind the window came out at, for the panel: the safety
    /// factor and the peak of every load case.
    ///
    /// `None` until a run or a generation has analysed the field it wrote, and
    /// `None` again the moment the next run starts - each run reads its own
    /// link, so the block describes the design on screen or says nothing at all.
    /// The session's console is told the same thing by the run itself.
    pub fn stress_summary(&self) -> Option<StressSummary> {
        self.worker.progress().and_then(|progress| progress.stress)
    }

    /// What the density layer is showing, for the panel.
    pub fn frame_label(&self) -> Option<String> {
        let kind = self.frame_kind?;
        let run = self.frame_run.map(RunKind::label).unwrap_or("run");
        Some(match kind {
            FrameKind::Preview { iteration } => format!("{run} surface at step {iteration}"),
            FrameKind::Final => format!("{run} exported mesh"),
        })
    }
}

/// Open a configuration file in the editor.
///
/// Prints the same problem summary `growforge view` does when the file
/// describes a runnable problem, and says why it does not when it does not -
/// the window opens either way, because fixing it is what the editor is for.
pub fn edit(config_path: &Path) -> Result<()> {
    crate::viewer::gpu::probe_adapter()?;
    let editor = Editor::open_or_create(config_path)?;
    match editor.state.problem() {
        Some(problem) => {
            print_warnings(&problem.warnings);
            print_problem_summary(problem);
        }
        None => {
            if let Some(error) = editor.state.error() {
                eprintln!("warning: this configuration is not runnable yet: {error}");
            }
        }
    }
    println!();

    let scene = editor.initial_scene();
    let title = editor.window_title();
    // Nothing outlives an editor window - closing it stops the runs behind it -
    // but stopping is cooperative, and a run only ends at its next checkpoint,
    // which for an analysis is a stage away. The loop keeps servicing the
    // window's message queue until the threads have really ended: leaving
    // earlier and waiting outside it is what Windows declares a hung process
    // and kills. The window asks the session it is holding, whichever one that
    // is by then - see `ViewerApp::should_keep_pumping` - so no closure taken
    // here could be left watching a session a file switch has replaced.
    let mut app = ViewerApp::new(title, scene, None).editing(editor);
    let result = app.run();
    // The threads are over; this collects them, so the process never leaves
    // while one is still part way through a stage.
    app.finish_editing();
    result
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::CsgOpSpec;
    use crate::viewer::editor::state::ShapeKind;
    use std::path::PathBuf;

    /// A comment rich configuration, small enough to build and run in a test.
    pub const FIXTURE: &str = r#"# growforge editor fixture: comments on purpose.

[project]
name = "fixture"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "pla"

[optimization]
# what fraction of the design cells to keep; the comment block above a key
# belongs to the key, not to the value, and has to survive an edit of it
mass_fraction = 0.3  # keep this comment
min_feature_mm = 16.0
max_iterations = 4

[output]
stl_path = "fixture.stl"
iso_level = 0.5

# the design space
[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [64.0, 32.0, 32.0]

# the first keepout
[[keepout]]
shape = "cylinder"
p1 = [16.0, 16.0, -1.0]
p2 = [16.0, 16.0, 33.0]
radius = 5.0

# the second keepout
[[keepout]]
shape = "sphere"
center = [50.0, 16.0, 16.0]
radius = 4.0

[[keepin]]
shape = "box"
min = [56.0, 8.0, 8.0]
max = [64.0, 24.0, 24.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 32.5, 32.5] }
directions = ["x", "y", "z"]

[[loadcases]]
name = "tip"
weight = 1.0

  [[loadcases.loads]]
  type = "force"
  region = { shape = "sphere", center = [60.0, 16.0, 16.0], radius = 6.0 }
  vector = [0.0, 0.0, -100.0]
"#;

    /// The same problem run by the growth engine.
    pub fn growth_fixture() -> &'static str {
        // Owned once so tests can hand out a `&'static str` like `fixture`.
        static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        TEXT.get_or_init(|| format!("engine = \"growth\"\n{FIXTURE}"))
    }

    /// The same problem exported by the solid engine, which is the fixture
    /// *without* its mass fraction: that engine refuses the key.
    pub fn solid_fixture() -> &'static str {
        static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        TEXT.get_or_init(|| {
            format!(
                "engine = \"solid\"\n{}",
                FIXTURE.replace("mass_fraction = 0.3  # keep this comment\n", "")
            )
        })
    }

    /// The fixture configuration.
    pub fn fixture() -> &'static str {
        FIXTURE
    }

    /// The same configuration with its second keepout written as a turned
    /// ellipsoid, for the paths that have to work on one.
    pub fn ellipsoid_fixture() -> &'static str {
        static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        TEXT.get_or_init(|| {
            FIXTURE.replace(
                "shape = \"sphere\"\ncenter = [50.0, 16.0, 16.0]\nradius = 4.0",
                "shape = \"ellipsoid\"\ncenter = [50.0, 16.0, 16.0]\nradii = [8.0, 4.0, 4.0]\n\
                 rotation_deg = [0.0, 0.0, 30.0]",
            )
        })
    }

    /// The fixture's sphere keepout as a straight tube, for the bend drag.
    pub fn tube_fixture() -> &'static str {
        static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        TEXT.get_or_init(|| {
            FIXTURE.replace(
                "shape = \"sphere\"\ncenter = [50.0, 16.0, 16.0]\nradius = 4.0",
                "shape = \"tube\"\np1 = [40.0, 16.0, 16.0]\np2 = [60.0, 16.0, 16.0]\n\
                 radius = 3.0",
            )
        })
    }

    /// A configuration whose load end is walled off from its supports.
    ///
    /// The shape of the incident this round is about: the first linear solve of
    /// it is near-singular, so the run fails - fast and identically on both
    /// backends - and what the editor does about that is what is under test.
    pub const STRANDED: &str = r#"
[project]
name = "stranded"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "pla"

# The cpu, so the failure is this structure's rather than this machine's.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.5
min_feature_mm = 16.0
max_iterations = 2

[output]
stl_path = "stranded.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [64.0, 16.0, 16.0]

[[keepout]]
shape = "box"
min = [28.0, -1.0, -1.0]
max = [36.0, 17.0, 17.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 16.5, 16.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [64.0, 8.0, 8.0], radius = 6.0 }
vector = [0.0, 0.0, -50.0]
"#;

    /// A SIMP problem whose first linear solve is long enough that a stop has to
    /// reach inside it to be prompt.
    ///
    /// Cold, at the optimization tolerance of 1e-8, on 27 648 elements: that is
    /// thousands of conjugate gradient iterations, tens of seconds even in a
    /// release build and minutes in a debug one. Cancellation between iterations
    /// cannot touch it; cancellation inside the solve ends it within
    /// [`constants::CG_CANCEL_CHECK_INTERVAL`] iterations.
    pub const SLOW: &str = r#"
[project]
name = "slow"

[resolution]
voxel_size_mm = 1.5

[material]
preset = "pla"

[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.4
min_feature_mm = 6.0
max_iterations = 40

[output]
stl_path = "slow.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [72.0, 36.0, 36.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 36.5, 36.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [72.0, 18.0, 18.0], radius = 6.0 }
vector = [0.0, 0.0, -80.0]
"#;

    /// A directory that removes itself, so a test that writes a file leaves
    /// nothing behind.
    pub struct TempDir(PathBuf);

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A ray straight down through a point on the floor plane, which is what
    /// the placement tests aim with.
    fn down(x: f64, y: f64) -> Ray {
        Ray {
            origin: [x, y, 300.0],
            direction: [0.0, 0.0, -1.0],
        }
    }

    /// A left click along `ray`, as the window delivers one: a press that
    /// grabbed nothing and a release that went nowhere.
    fn click(editor: &mut Editor, ray: &Ray) {
        editor.press(ray);
        editor.release(Some(ray), true);
    }

    /// One frame with `key` pressed under `modifiers`, through the very path
    /// the window uses.
    fn press_key_with(editor: &mut Editor, key: egui::Key, modifiers: egui::Modifiers) {
        let context = egui::Context::default();
        let mut input = egui::RawInput {
            modifiers,
            ..egui::RawInput::default()
        };
        input.events.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        });
        let _ = context.run_ui(input, |_| {});
        editor.shortcuts(&context);
    }

    /// One frame with `key` pressed on its own.
    fn press_key(editor: &mut Editor, key: egui::Key) {
        press_key_with(editor, key, egui::Modifiers::NONE);
    }

    /// One frame with `key` pressed under the shortcut modifier.
    fn press_command_key(editor: &mut Editor, key: egui::Key) {
        press_key_with(editor, key, egui::Modifiers::COMMAND);
    }

    /// How many objects each shape list holds, for the tests that care which
    /// list an add landed in.
    fn counts(editor: &Editor) -> [usize; 4] {
        let config = editor.state.config();
        [
            config.domain.len(),
            config.keepout.len(),
            config.keepin.len(),
            config.supports.len(),
        ]
    }

    /// The one list that grew by one between two counts, named as the tree
    /// heads it, or `None` when that is not what happened.
    fn grown(before: [usize; 4], after: [usize; 4]) -> Option<&'static str> {
        let names = ["domain", "keepout", "keepin", "supports"];
        let mut grew = None;
        for (index, name) in names.iter().enumerate() {
            match after[index].checked_sub(before[index]) {
                Some(0) => continue,
                Some(1) if grew.is_none() => grew = Some(*name),
                _ => return None,
            }
        }
        grew
    }

    /// Write `text` to a fresh directory and return the file's path.
    pub fn write_temp(name: &str, text: &str) -> (TempDir, PathBuf) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "growforge_editor_{name}_{}_{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("temp directory");
        let path = directory.join("config.toml");
        std::fs::write(&path, text).expect("write fixture");
        (TempDir(directory), path)
    }

    #[test]
    fn the_debounce_collapses_a_burst_of_edits_into_one_refresh() {
        let delay = Duration::from_millis(150);
        let start = Instant::now();
        let mut debounce = Debounce::default();
        assert!(!debounce.is_pending());
        assert!(!debounce.take(start), "nothing owed, nothing to do");

        debounce.touch(start, delay);
        assert!(debounce.is_pending());
        assert!(
            !debounce.take(start + Duration::from_millis(100)),
            "too early"
        );
        // Every further edit pushes the deadline out again.
        debounce.touch(start + Duration::from_millis(100), delay);
        assert!(!debounce.take(start + Duration::from_millis(200)));
        assert!(debounce.take(start + Duration::from_millis(260)));
        // It fires once, not once per frame after the deadline.
        assert!(!debounce.take(start + Duration::from_millis(500)));
        assert!(!debounce.is_pending());

        debounce.touch(start, delay);
        debounce.cancel();
        assert!(!debounce.take(start + Duration::from_secs(10)));
    }

    #[test]
    fn opening_the_editor_shows_the_setup_and_marks_nothing_dirty() {
        let (_dir, path) = write_temp("open_editor", fixture());
        let editor = Editor::open(&path).expect("open");
        assert!(!editor.state.is_dirty());
        assert!(editor.window_title().ends_with("config.toml"));
        let scene = editor.initial_scene();
        for layer in [
            Layer::Domain,
            Layer::Keepout,
            Layer::Keepin,
            Layer::Supports,
        ] {
            assert!(scene.get(layer).is_some(), "{layer:?} is missing");
        }
        assert!(scene.get(Layer::Density).is_none());
        assert!(scene.get(Layer::Gizmo).is_none());
    }

    /// The editor's title names the build, the mode and the document, and the
    /// unsaved marker still goes on the end of it.
    ///
    /// The version is spelled out here rather than read from the constant the
    /// title is built from, so what is asserted is the string a title bar has to
    /// show and not a second reading of the same value.
    #[test]
    fn the_window_title_names_the_build_the_mode_and_the_document() {
        let (_dir, path) = write_temp("title", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let clean = format!("growforge {} edit - config.toml", env!("CARGO_PKG_VERSION"));
        assert_eq!(editor.window_title(), clean);

        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.44));
        assert!(editor.state.is_dirty());
        assert_eq!(editor.window_title(), format!("{clean}*"));

        editor.save();
        assert_eq!(editor.window_title(), clean);
    }

    #[test]
    fn a_committed_edit_marks_the_file_dirty_and_refreshes_the_scene() {
        let (_dir, path) = write_temp("refresh", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        let before = scene
            .get(Layer::Keepout)
            .expect("a keepout layer")
            .triangles();

        // The edit a properties field makes: open an interaction, change the
        // configuration, close it.
        editor.state.begin_edit(1);
        let selection = Selection::Keepout(1);
        let spec = shape_of(editor.state.config(), selection).expect("a shape");
        let ShapeSpecSphere { center, radius } = sphere_of(&spec);
        state::set_shape(
            editor.state.config_mut(),
            selection,
            crate::config::ShapeSpec::Sphere {
                center,
                radius: radius * 1.5,
            },
        );
        editor.state.end_edit(1);
        editor.on_edited();
        assert!(editor.state.is_dirty());
        assert!(
            editor.window_title().contains('*'),
            "the title marks unsaved edits"
        );

        // The setup is owed a rebuild and does not get one until the debounce
        // has run out; then it does, and the overlays follow the edit.
        assert!(editor.state.is_stale());
        editor.pump(&mut scene);
        assert!(
            editor.state.is_stale(),
            "the refresh must wait out the burst"
        );
        std::thread::sleep(Duration::from_secs_f64(
            constants::VIEW_EDIT_REFRESH_DEBOUNCE_S * 2.0,
        ));
        assert!(
            editor.pump(&mut scene).setup,
            "the refresh must reach the scene"
        );
        assert!(!editor.state.is_stale());
        assert!(editor.state.is_valid(), "{:?}", editor.state.error());
        let after = scene
            .get(Layer::Keepout)
            .expect("a keepout layer")
            .triangles();
        assert_eq!(before, after, "a sphere keeps its tessellation count");
        // The sphere grew from a radius of 4 to one of 6, so the overlay it is
        // drawn from now reaches further along x than it could have before.
        let bounds = scene.get(Layer::Keepout).expect("a layer").bounds;
        assert!(bounds.max[0] > 55.0, "the layer still shows the old sphere");

        // And the file on disk has not been touched.
        assert_eq!(std::fs::read_to_string(&path).expect("read"), fixture());
    }

    #[test]
    fn saving_writes_the_file_and_clears_the_marker() {
        let (_dir, path) = write_temp("save", fixture());
        let mut editor = Editor::open(&path).expect("open");
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.44));
        assert!(editor.state.is_dirty());
        editor.save();
        assert!(!editor.state.is_dirty());
        assert!(!editor.window_title().contains('*'));
        let saved = std::fs::read_to_string(&path).expect("read");
        assert!(saved.contains("mass_fraction = 0.44"), "{saved}");
        assert!(saved.contains("# keep this comment"), "{saved}");
        assert!(editor.status_line().expect("a status").contains("saved"));
    }

    #[test]
    fn a_close_request_with_unsaved_changes_asks_before_the_window_goes() {
        let (_dir, path) = write_temp("close", fixture());
        let mut editor = Editor::open(&path).expect("open");
        // Nothing to lose: the window may go straight away.
        assert!(editor.request_close());
        assert!(!editor.is_asking());

        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.44));
        assert!(!editor.request_close(), "unsaved edits must be asked about");
        assert!(editor.is_asking() && !editor.may_close());
        editor.decide(CloseDecision::Cancel);

        // An edit undone back to what the file holds is nothing to ask about:
        // the modal would be claiming changes that are not there.
        editor.undo();
        assert!(!editor.state.is_dirty());
        assert!(editor.request_close(), "there is nothing left unsaved");
        assert!(!editor.is_asking());
        editor.redo();

        // Cancelling puts the modal away and keeps the window.
        editor.decide(CloseDecision::Cancel);
        assert!(!editor.is_asking() && !editor.may_close());

        // Saving closes, and really writes.
        assert!(!editor.request_close());
        editor.decide(CloseDecision::Save);
        assert!(editor.may_close() && !editor.state.is_dirty());
        assert!(
            std::fs::read_to_string(&path)
                .expect("read")
                .contains("0.44")
        );

        // And discarding closes without writing.
        let (_dir, path) = write_temp("close_discard", fixture());
        let mut editor = Editor::open(&path).expect("open");
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.9));
        editor.request_close();
        editor.decide(CloseDecision::Discard);
        assert!(editor.may_close());
        assert_eq!(std::fs::read_to_string(&path).expect("read"), fixture());
    }

    /// Switching files is the same guard the close is, one intent wider: the
    /// document is left behind either way, so the same modal asks the same
    /// question - about the switch that was actually asked for.
    #[test]
    fn switching_files_with_unsaved_changes_asks_about_that_switch() {
        let (_dir, path) = write_temp("switch_guard", fixture());
        let other = path.with_file_name("other.toml");
        std::fs::write(&other, fixture()).expect("write the second file");
        let mut editor = Editor::open(&path).expect("open");

        // Nothing to lose: the switch is set up at once, and it is the window's
        // to perform.
        assert!(editor.request_open(other.clone()));
        assert!(!editor.is_asking());
        assert!(editor.is_switching());
        assert_eq!(editor.take_switch().expect("a switch").path(), other);
        assert!(editor.take_switch().is_none(), "one ask is one switch");

        // With edits, the modal asks - and it asks about this switch, not about
        // a close.
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.44));
        assert!(!editor.request_open(other.clone()));
        assert_eq!(editor.asking(), Some(&Intent::OpenFile(other.clone())));
        assert!(
            !editor.is_switching(),
            "nothing switches before the question is answered"
        );

        // Cancel stays on this file, with the edit still unsaved and unwritten.
        editor.decide(CloseDecision::Cancel);
        assert!(!editor.is_asking() && !editor.is_switching());
        assert!(!editor.may_close(), "a switch is not a close");
        assert!(editor.state.is_dirty());
        assert_eq!(editor.state.path(), path);
        assert_eq!(std::fs::read_to_string(&path).expect("read"), fixture());

        // Discard switches without writing.
        assert!(!editor.request_open(other.clone()));
        editor.decide(CloseDecision::Discard);
        assert_eq!(editor.take_switch().expect("a switch").path(), other);
        assert_eq!(std::fs::read_to_string(&path).expect("read"), fixture());

        // Save writes this file, then switches.
        assert!(editor.state.is_dirty(), "a discard undoes nothing");
        assert!(!editor.request_open(other.clone()));
        editor.decide(CloseDecision::Save);
        assert!(!editor.state.is_dirty());
        assert!(
            std::fs::read_to_string(&path)
                .expect("read")
                .contains("0.44")
        );
        assert_eq!(editor.take_switch().expect("a switch").path(), other);
        assert!(
            !editor.may_close(),
            "saving to switch is not agreeing to close"
        );
    }

    /// "New" scaffolds exactly what the command line scaffolds, and refuses a
    /// file that is there exactly as the command line refuses it. It is never a
    /// second way to open one.
    #[test]
    fn a_new_file_is_scaffolded_and_an_existing_one_is_never_taken_for_one() {
        let (_dir, path) = write_temp("switch_new", fixture());
        let fresh = path.with_file_name("brand_new_part.toml");
        let mut editor = Editor::open(&path).expect("open");

        assert!(editor.request_new(fresh.clone()));
        let switch = editor.take_switch().expect("a switch");
        assert_eq!(switch.path(), fresh);
        assert!(!fresh.exists(), "asking for a file must not write one");

        // Performing it writes the starter configuration and opens it - the
        // same file `growforge edit` on a name that is not there writes.
        let opened = switch.open().expect("scaffold and open");
        assert!(fresh.exists(), "the starter configuration was not written");
        assert!(
            opened.state.is_valid(),
            "the starter configuration must validate as written: {:?}",
            opened.state.error()
        );
        assert!(!opened.state.is_dirty(), "a fresh file has nothing to save");
        let config = opened.state.config();
        assert_eq!(config.engine_name().unwrap(), constants::STARTER_ENGINE);
        assert_eq!(config.project.name, "brand_new_part");
        assert_eq!(config.output.stl_path, "brand_new_part.stl");
        assert_eq!(opened.state.directory(), fresh.parent().expect("a parent"));

        // A "new" pointed at a file that is there is refused where it is asked
        // for: nothing is asked about, nothing is switched, nothing is written,
        // and the panel says why.
        let before = std::fs::read_to_string(&path).expect("read");
        assert!(!editor.request_new(path.clone()));
        assert!(!editor.is_switching() && !editor.is_asking());
        assert!(
            editor
                .status_line()
                .expect("a status")
                .contains("use open instead"),
            "{:?}",
            editor.status_line()
        );
        assert_eq!(std::fs::read_to_string(&path).expect("read"), before);
        assert_eq!(editor.state.path(), path, "the session moved");

        // And the scaffold refuses again underneath, for a file that appeared
        // after the asking - or a save dialog that offered to replace one.
        assert!(
            Switch::to_create(path.clone()).open().is_err(),
            "the scaffold overwrote a file"
        );
        assert_eq!(std::fs::read_to_string(&path).expect("read"), before);
        // The other direction is not guarded and must not be: opening a file
        // that is there is the whole point of "open".
        assert!(Switch::to_open(path.clone()).open().is_ok());
    }

    /// A question already on screen is the one being answered: nothing may
    /// replace it and nothing may reach past it. The sequence pinned here is the
    /// one that made it matter - a close asked about, `Ctrl+O` answered with a
    /// file, and "discard" then applied to a question the user never read.
    #[test]
    fn a_question_on_screen_cannot_be_replaced_or_reached_past() {
        let (_dir, path) = write_temp("guard_immutable", fixture());
        let other = path.with_file_name("other.toml");
        std::fs::write(&other, fixture()).expect("write the second file");
        let mut editor = Editor::open(&path).expect("open");
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.44));

        // The close is asked about, and that is the question from here on.
        assert!(!editor.request_close());
        assert_eq!(editor.asking(), Some(&Intent::CloseWindow));

        // Nothing else may become the question - not a switch, not a new file,
        // and not a repeat of the close itself.
        assert!(!editor.request_open(other.clone()));
        assert_eq!(editor.asking(), Some(&Intent::CloseWindow));
        assert!(!editor.is_switching());
        assert!(!editor.request_new(path.with_file_name("newer.toml")));
        assert_eq!(editor.asking(), Some(&Intent::CloseWindow));
        assert!(!editor.request_close());
        assert_eq!(editor.asking(), Some(&Intent::CloseWindow));

        // And nothing reaches past it from the keyboard, whatever the binding
        // is: the modal's own buttons are the only way to answer it.
        editor.state.select(Some(Selection::Keepout(1)));
        let undo_depth = editor.state.undo_depth();
        let keepouts = editor.state.config().keepout.len();
        for key in [
            egui::Key::O,
            egui::Key::N,
            egui::Key::S,
            egui::Key::Z,
            egui::Key::Y,
        ] {
            press_command_key(&mut editor, key);
        }
        press_key(&mut editor, egui::Key::Delete);
        press_key(&mut editor, egui::Key::Escape);
        assert_eq!(
            editor.pending_pick(),
            None,
            "a file dialog was asked for over the modal"
        );
        assert_eq!(editor.take_pick(), None);
        assert_eq!(editor.asking(), Some(&Intent::CloseWindow));
        assert!(!editor.is_switching());
        assert_eq!(
            editor.state.undo_depth(),
            undo_depth,
            "an undo or redo reached past the modal"
        );
        assert_eq!(
            editor.state.config().keepout.len(),
            keepouts,
            "a delete reached past the modal"
        );
        assert!(editor.state.is_dirty(), "a save reached past the modal");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), fixture());

        // So the answer answers what was asked: discarding a close closes, and
        // does not switch files.
        editor.decide(CloseDecision::Discard);
        assert!(
            editor.may_close(),
            "the answer was applied to something the user was not asked"
        );
        assert!(!editor.is_switching(), "discarding a close switched files");
        assert_eq!(editor.state.path(), path);
        assert_eq!(std::fs::read_to_string(&path).expect("read"), fixture());

        // With the question answered, the keyboard is live again.
        press_command_key(&mut editor, egui::Key::O);
        assert_eq!(editor.take_pick(), Some(Pick::Open));
        press_command_key(&mut editor, egui::Key::Z);
        assert!(
            !editor.state.is_dirty(),
            "the shortcuts did not come back after the answer"
        );
    }

    /// The file shortcuts ask for exactly what the toolbar's buttons ask for,
    /// and asking is all either of them does: the dialog itself is the window's,
    /// which is what keeps this path testable at all.
    #[test]
    fn the_file_shortcuts_ask_for_the_same_dialogs_the_buttons_do() {
        let (_dir, path) = write_temp("file_shortcuts", fixture());
        let mut editor = Editor::open(&path).expect("open");
        assert_eq!(editor.pending_pick(), None);

        press_command_key(&mut editor, egui::Key::O);
        assert_eq!(editor.pending_pick(), Some(Pick::Open));
        assert_eq!(editor.take_pick(), Some(Pick::Open));
        assert_eq!(editor.take_pick(), None, "one press is one dialog");

        press_command_key(&mut editor, egui::Key::N);
        assert_eq!(editor.take_pick(), Some(Pick::New));

        // Which is what the buttons do, through the same two methods.
        editor.ask_to_open();
        assert_eq!(editor.take_pick(), Some(Pick::Open));
        editor.ask_for_new();
        assert_eq!(editor.take_pick(), Some(Pick::New));

        // Unmodified, both keys are nothing: they are letters someone is typing.
        press_key(&mut editor, egui::Key::O);
        press_key(&mut editor, egui::Key::N);
        assert_eq!(editor.pending_pick(), None);
        // And neither has touched the document or the session.
        assert!(!editor.state.is_dirty());
        assert!(!editor.is_switching() && !editor.is_asking());
    }

    #[test]
    fn a_click_selects_the_object_under_it_and_empty_space_clears_it() {
        let (_dir, path) = write_temp("clicking", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.pump(&mut scene);

        // A ray straight at the second keepout, a sphere at [50, 16, 16].
        let ray = Ray {
            origin: [50.0, 16.0, 200.0],
            direction: [0.0, 0.0, -1.0],
        };
        editor.release(Some(&ray), true);
        assert_eq!(editor.state.selection(), Some(Selection::Keepout(1)));
        assert!(
            editor.pump(&mut scene).overlays,
            "a selection draws its overlays"
        );
        assert!(scene.get(Layer::Selection).is_some());
        assert!(scene.get(Layer::Gizmo).is_some());

        // Empty space deselects and takes the overlays with it.
        let miss = Ray {
            origin: [1000.0, 1000.0, 200.0],
            direction: [0.0, 0.0, -1.0],
        };
        editor.release(Some(&miss), true);
        assert_eq!(editor.state.selection(), None);
        editor.pump(&mut scene);
        assert!(scene.get(Layer::Selection).is_none());
        assert!(scene.get(Layer::Gizmo).is_none());

        // A release that ended a camera drag never changes the selection.
        editor.release(Some(&ray), false);
        assert_eq!(editor.state.selection(), None);
    }

    #[test]
    fn a_gizmo_drag_moves_the_object_and_commits_exactly_one_undo_step() {
        let (_dir, path) = write_temp("dragging", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.state.select(Some(Selection::Keepout(1)));
        editor.pump(&mut scene);
        assert!(!editor.handles.is_empty(), "a selection has handles");

        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Translate(0))
            .copied()
            .expect("an x arrow");
        let steps = editor.state.undo_depth();
        // Press on the handle, from a direction that is not along x.
        let press = Ray {
            origin: [handle.position[0], handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert!(editor.press(&press), "the handle must take the press");
        assert!(editor.is_dragging());
        assert_eq!(
            editor.state.undo_depth(),
            steps,
            "a grab is not yet an edit"
        );

        for offset in [2.0, 4.0, 6.0] {
            let moved = Ray {
                origin: [handle.position[0] + offset, handle.position[1], 300.0],
                direction: [0.0, 0.0, -1.0],
            };
            editor.drag_to(&moved);
        }
        editor.release(None, false);
        assert!(!editor.is_dragging());
        assert_eq!(
            editor.state.undo_depth(),
            steps + 1,
            "a whole drag is one undo step"
        );
        let spec = shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape");
        let ShapeSpecSphere { center, .. } = sphere_of(&spec);
        assert!(
            (center[0] - 56.0).abs() < 1e-6,
            "the sphere moved to {center:?}"
        );
        // And undoing puts it back exactly.
        editor.undo();
        let spec = shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape");
        let ShapeSpecSphere { center, .. } = sphere_of(&spec);
        assert!((center[0] - 50.0).abs() < 1e-12);
    }

    /// A press on a resize handle that goes nowhere is **not** an edit.
    ///
    /// A latched handle changes nothing until the pointer has left its dead zone
    /// ([`constants::VIEW_EDIT_RESIZE_LATCH_MM`]), which is exactly the travel a
    /// click with an unsteady hand covers - and an interaction opened for it
    /// would record an undo step that undoes nothing *and* clear the redo stack,
    /// so merely touching a handle would cost the user the redo they were part
    /// way through. The interaction therefore opens on the first frame that
    /// really changes the shape, and the drag that does leave the dead zone is
    /// still one step measured from where it started.
    #[test]
    fn a_drag_that_never_leaves_its_dead_zone_is_not_an_undo_step() {
        let (_dir, path) = write_temp("dead_zone", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        // Off, so the drag below is exactly what the pointer asked for rather
        // than what the domain wall left of it.
        editor.set_containment(false);
        let cylinder = Selection::Keepout(0);
        editor.state.select(Some(cylinder));
        editor.pump(&mut scene);

        // Something to redo: an edit, undone.
        editor
            .state
            .edit(|config| config.optimization.max_iterations = Some(7));
        assert!(editor.state.undo(), "there is an edit to undo");
        assert_eq!(editor.state.redo_depth(), 1);
        let steps = editor.state.undo_depth();
        let before = shape_of(editor.state.config(), cylinder).expect("a shape");

        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Endpoint(1))
            .copied()
            .expect("an end handle");
        // Straight down the view direction, so the pointer's own x and y are
        // where the handle goes in the plane it is dragged in.
        let at = |x: f64, y: f64| Ray {
            origin: [x, y, 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert!(
            editor.press(&at(handle.position[0], handle.position[1])),
            "the handle must take the press"
        );
        assert_eq!(
            editor.callout().expect("a callout").handle.kind,
            gizmo::HandleKind::Endpoint(1),
            "the press landed on another handle"
        );

        // A tremor of a third of a millimetre each way, well inside the latch
        // distance however that constant is retuned.
        let tremor = 0.5 * constants::VIEW_EDIT_RESIZE_LATCH_MM;
        editor.drag_to(&at(
            handle.position[0] + tremor,
            handle.position[1] + tremor,
        ));
        assert!(
            !editor.state.is_editing(),
            "a frame that changed nothing opened an interaction"
        );
        editor.release(None, false);
        assert_eq!(
            editor.state.undo_depth(),
            steps,
            "a drag that changed nothing recorded an undo step"
        );
        assert_eq!(
            editor.state.redo_depth(),
            1,
            "a drag that changed nothing threw the redo stack away"
        );
        assert_eq!(
            shape_of(editor.state.config(), cylinder).expect("a shape"),
            before,
            "the dead zone moved the shape"
        );

        // The same handle, past the dead zone: one step, measured from the shape
        // the drag started on rather than from the frames that changed nothing.
        assert!(editor.press(&at(handle.position[0], handle.position[1])));
        for offset in [tremor, 4.0, 8.0] {
            editor.drag_to(&at(handle.position[0] + offset, handle.position[1]));
        }
        editor.release(None, false);
        assert_eq!(
            editor.state.undo_depth(),
            steps + 1,
            "a whole drag is one undo step"
        );
        let moved = shape_of(editor.state.config(), cylinder).expect("a shape");
        assert_ne!(moved, before, "the drag past the dead zone changed nothing");
        editor.undo();
        assert_eq!(
            shape_of(editor.state.config(), cylinder).expect("a shape"),
            before,
            "the undo step was measured from the wrong shape"
        );
    }

    /// The other half of that rule: a frame that changes nothing does not commit
    /// either, so the **containment clamp** waits for one that does.
    ///
    /// A file may hold an object outside the domain - nothing but a commit
    /// through the editor is clamped - and pressing its handle and shaking is
    /// not an edit, so it must not be the moment that object is pulled inside:
    /// there would be no undo step covering the move.
    #[test]
    fn a_dead_zone_frame_does_not_commit_the_containment_clamp() {
        // The fixture's sphere keepout, as a cylinder hovering above the lid:
        // small enough to fit, so the clamp really would move it.
        let text = fixture().replace(
            "shape = \"sphere\"\ncenter = [50.0, 16.0, 16.0]\nradius = 4.0",
            "shape = \"cylinder\"\np1 = [40.0, 16.0, 40.0]\np2 = [60.0, 16.0, 40.0]\n\
             radius = 3.0",
        );
        let (_dir, path) = write_temp("dead_zone_clamp", &text);
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        assert!(editor.containment(), "containment starts on");
        let cylinder = Selection::Keepout(1);
        editor.state.select(Some(cylinder));
        editor.pump(&mut scene);

        let before = shape_of(editor.state.config(), cylinder).expect("a shape");
        let domain = state::containment_bounds(editor.state.config()).expect("a domain");
        assert!(
            state::clamped_into(&before, &domain).1,
            "this fixture is no longer one a commit would move"
        );
        let steps = editor.state.undo_depth();

        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Endpoint(1))
            .copied()
            .expect("an end handle");
        let at = |x: f64, y: f64| Ray {
            origin: [x, y, 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert!(editor.press(&at(handle.position[0], handle.position[1])));
        let tremor = 0.5 * constants::VIEW_EDIT_RESIZE_LATCH_MM;
        editor.drag_to(&at(
            handle.position[0] + tremor,
            handle.position[1] + tremor,
        ));
        editor.release(None, false);
        assert_eq!(
            shape_of(editor.state.config(), cylinder).expect("a shape"),
            before,
            "a frame that changed nothing pulled the object into the domain"
        );
        assert_eq!(editor.state.undo_depth(), steps);
    }

    /// The gesture the tube exists for, through the editor's own paths: press
    /// the middle of a straight tube, pull it into a curve, and bring it back
    /// down onto the line between the ends **off-centre**.
    ///
    /// The shape is straight again at that moment - stored with no bend, which
    /// is what the file has to say - but the pointer is still holding a point
    /// five millimetres along the tube from its middle, and the handle and the
    /// number have to be there rather than where the straightened shape alone
    /// would put them. Letting go is what hands the handle back to the shape.
    #[test]
    fn a_bend_dragged_straight_off_centre_keeps_its_handle_under_the_pointer() {
        let (_dir, path) = write_temp("tube_bend", tube_fixture());
        let mut editor = Editor::open(&path).expect("open");
        assert!(editor.state.is_valid(), "{:?}", editor.state.error());
        let mut scene = editor.initial_scene();
        editor.pump(&mut scene);
        editor.state.select(Some(Selection::Keepout(1)));
        editor.pump(&mut scene);

        let middle = [50.0, 16.0, 16.0];
        let bend_at = |editor: &Editor| -> Vec3 {
            editor
                .handles
                .iter()
                .find(|h| h.kind == gizmo::HandleKind::Bend)
                .expect("a bend handle")
                .position
        };
        assert_eq!(
            bend_at(&editor),
            middle,
            "a straight tube bends from its middle"
        );

        // Looking down z, so the drag plane is the one the tube lies in.
        let down = |x: f64, y: f64| Ray {
            origin: [x, y, 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert!(
            editor.press(&down(middle[0], middle[1])),
            "the middle of a straight tube has to take the press"
        );

        // Out into a curve, five millimetres short of the middle, and then down
        // onto the line a millimetre at a time.
        let mut readings: Vec<(Vec3, f64)> = Vec::new();
        for (x, y) in [(45.0, 22.0), (45.0, 19.0), (45.0, 17.0), (45.0, 16.0)] {
            editor.drag_to(&down(x, y));
            editor.pump(&mut scene);
            let at = editor.callout().expect("a callout").at;
            assert_eq!(at.kind, measure::MeasureKind::Length);
            assert_eq!(
                bend_at(&editor),
                [x, y, 16.0],
                "the handle left the pointer at {x}, {y}"
            );
            readings.push(([x, y, 16.0], at.value));
        }
        // The tube really is straight now - the shape, and so the file, says
        // so - while the handle and the number are still at the pointer.
        let spec = shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape");
        let ShapeSpec::Tube { p1, p2, bend, .. } = spec else {
            panic!("a tube");
        };
        assert_eq!(bend, None, "a bend on the line is stored as no bend");
        assert_eq!((p1, p2), ([40.0, 16.0, 16.0], [60.0, 16.0, 16.0]));
        assert_eq!(bend_at(&editor), [45.0, 16.0, 16.0]);
        // The number is a distance to a fixed point, so it can never change by
        // more than the pointer moved - which is what "no jump" means, and what
        // the last step of all, the one that crosses into straightness, would
        // fail by five millimetres if the callout read the stored bend.
        for (a, b) in readings.iter().zip(readings.iter().skip(1)) {
            let travelled = crate::geometry::length(difference(b.0, a.0));
            assert!(
                (a.1 - b.1).abs() <= travelled + 1e-9,
                "the number moved {} while the pointer moved {travelled}: {readings:?}",
                (a.1 - b.1).abs()
            );
        }
        let last = readings.last().expect("a reading").1;
        assert!(
            (last - 5.0).abs() < 1e-9,
            "straightened off-centre the callout read {last} rather than 5"
        );

        // Letting go hands the handle back to the shape, which is the only
        // thing that puts it in the middle of a straight tube.
        editor.release(None, false);
        editor.pump(&mut scene);
        assert_eq!(bend_at(&editor), middle);
        // And what was left behind is a straight tube nobody has to fix.
        assert!(editor.state.is_valid(), "{:?}", editor.state.error());
    }

    /// A bend dragged past the lid of the domain, with containment on: the
    /// commit is moved back inside, and the handle and the number that follow it
    /// read the tube that was really committed rather than the one that was
    /// asked for.
    ///
    /// The composition under test is `drag_handle_at` through
    /// [`state::clamped_into`]: the clamp is a translation of the whole shape,
    /// so the live handle position has to be carried by it, and the callout -
    /// which measures from the *committed* ends to that handle - then reads the
    /// pull the tube actually has.
    #[test]
    fn a_bend_stopped_at_the_domain_wall_still_measures_what_it_committed() {
        let (_dir, path) = write_temp("bend_containment", tube_fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        assert!(editor.containment(), "containment starts on");
        editor.state.select(Some(Selection::Keepout(1)));
        // Looking along -y, so the bend is dragged in the plane the tube's own
        // axis and the domain's lid are both in.
        editor.set_view([0.0, -1.0, 0.0]);
        editor.pump(&mut scene);

        // Straight along x at z = 16, radius 3, in a domain whose lid is at
        // z = 32: a bend pulled to z = 30 puts the top of the arc at 33, so the
        // commit is moved a millimetre down.
        let at = |z: f64| Ray {
            origin: [50.0, 300.0, z],
            direction: [0.0, -1.0, 0.0],
        };
        assert!(editor.press(&at(16.0)), "the middle takes the press");
        editor.drag_to(&at(30.0));
        editor.pump(&mut scene);

        let spec = shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape");
        let ShapeSpec::Tube { p1, p2, bend, .. } = spec.clone() else {
            panic!("a tube");
        };
        assert_eq!(p1, [40.0, 16.0, 15.0], "the clamp moves the whole shape");
        assert_eq!(p2, [60.0, 16.0, 15.0]);
        assert_eq!(bend, Some([50.0, 16.0, 29.0]));
        let bounds = spec.to_shape("test").expect("a shape").bounds();
        assert!(bounds.max[2] <= 32.0 + 1e-9, "{bounds:?}");
        assert_eq!(
            editor.containment_note(),
            Some(constants::VIEW_EDIT_CONTAINMENT_NOTE),
            "a clamped commit must say so"
        );

        // The handle went with it, so it is still on the bend the tube has -
        // and the number is the 14 mm of pull that survived the clamp, not the
        // 15 the un-carried handle would have read off the moved ends.
        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Bend)
            .expect("a bend handle");
        assert_eq!(handle.position, [50.0, 16.0, 29.0]);
        let callout = editor.callout().expect("a callout");
        assert_eq!(callout.at.kind, measure::MeasureKind::Length);
        assert!(
            (callout.at.value - 14.0).abs() < 1e-9,
            "the callout read {} rather than the 14 mm committed",
            callout.at.value
        );

        editor.release(None, false);

        // The same gesture on the same tube with containment switched off, as
        // the control: nothing is moved, the bend is where it was dragged, and
        // the number is the same 14 - which is what makes the reading above the
        // distance covered rather than the distance asked for.
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.set_containment(false);
        editor.state.select(Some(Selection::Keepout(1)));
        editor.set_view([0.0, -1.0, 0.0]);
        editor.pump(&mut scene);
        assert!(editor.press(&at(16.0)));
        editor.drag_to(&at(30.0));
        editor.pump(&mut scene);
        let spec = shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape");
        let ShapeSpec::Tube { p1, p2, bend, .. } = spec else {
            panic!("a tube");
        };
        assert_eq!((p1, p2), ([40.0, 16.0, 16.0], [60.0, 16.0, 16.0]));
        assert_eq!(bend, Some([50.0, 16.0, 30.0]));
        assert!(editor.containment_note().is_none());
        let callout = editor.callout().expect("a callout");
        assert!((callout.at.value - 14.0).abs() < 1e-9);
        editor.release(None, false);
    }

    /// The tube's own creation flow, from each of the four add rows: the button
    /// opens the placement mode, the first click lands on a surface, the second
    /// on the ruled floor, and what is left behind is a straight tube in *that*
    /// row's list, selected and ready for the bend drag.
    #[test]
    fn a_tube_is_placed_by_two_clicks_from_every_add_row() {
        let (_dir, path) = write_temp("place_rows", fixture());
        for what in [
            NewObject::Domain(ShapeKind::Tube, CsgOpSpec::Add),
            NewObject::Keepout(ShapeKind::Tube),
            NewObject::Keepin(ShapeKind::Tube),
            NewObject::Support(ShapeKind::Tube),
        ] {
            let mut editor = Editor::open(&path).expect("open");
            let mut scene = editor.initial_scene();
            // The floor grid the second click lands on is derived by a pump.
            editor.pump(&mut scene);
            // Containment has a test of its own; here the points are the
            // clicks' own.
            editor.set_containment(false);
            let radius = state::default_tube_radius(editor.state.config());
            let before = counts(&editor);
            let steps = editor.state.undo_depth();

            editor.toggle_placing(what);
            assert!(editor.is_placing());
            let hint = editor.placement_hint().expect("a hint");
            assert!(hint.contains(what.list_label()), "{hint}");
            assert!(
                hint.contains(constants::VIEW_EDIT_PLACE_HINT_FIRST),
                "{hint}"
            );

            // The lid of the domain, which is a landing surface even though it
            // is never selectable. Chosen so the tube the two clicks make runs
            // through the node lattice rather than between it: a support region
            // has to hold something, whichever list this round is placing into.
            click(&mut editor, &down(28.0, 20.0));
            assert_eq!(
                editor.placing().and_then(|placing| placing.first),
                Some([28.0, 20.0, 32.0])
            );
            assert!(editor.is_placing(), "one click is half a placement");
            let hint = editor.placement_hint().expect("a hint");
            assert!(
                hint.contains(constants::VIEW_EDIT_PLACE_HINT_SECOND),
                "{hint}"
            );
            editor.pump(&mut scene);
            assert!(
                scene.get(Layer::Placement).is_some(),
                "the point that has been clicked has to be on screen"
            );

            // And a point beside the domain, on the ruled floor.
            click(&mut editor, &down(-4.0, 20.0));
            assert!(!editor.is_placing(), "the second click ends the placement");
            assert!(editor.placement_hint().is_none());

            let after = counts(&editor);
            assert_eq!(grown(before, after), Some(what.list_label()));
            let selection = editor.state.selection().expect("the placement selects");
            let spec = shape_of(editor.state.config(), selection).expect("a shape");
            let ShapeSpec::Tube {
                p1,
                p2,
                bend,
                radius: placed,
            } = spec
            else {
                panic!("a placed tube is a tube: {spec:?}");
            };
            assert_eq!(p1, [28.0, 20.0, 32.0]);
            assert_eq!(p2, [-4.0, 20.0, 0.0]);
            assert_eq!(bend, None, "a placed tube is straight");
            assert!((placed - radius).abs() < 1e-12, "{placed} against {radius}");
            // A shape growforge reads, with the bounds two points and a radius
            // make. Not the whole problem: a tube placed in the *domain* list
            // enlarges the design space, which moves the node lattice under the
            // fixture's own half-millimetre support region, and that is the
            // fixture's affair rather than the placement's.
            let bounds = spec.to_shape("placed").expect("a shape").bounds();
            assert!(bounds.min[0] <= -4.0 - radius + 1e-9, "{bounds:?}");
            assert!(bounds.max[2] >= 32.0 + radius - 1e-9, "{bounds:?}");

            // The overlays of the object it selected, and nothing of the mode.
            editor.pump(&mut scene);
            assert!(scene.get(Layer::Placement).is_none());
            assert!(
                editor
                    .handles
                    .iter()
                    .any(|handle| handle.kind == gizmo::HandleKind::Bend),
                "the placed tube is left ready for the bend drag"
            );

            // And the whole placement is one undo step, like every other edit.
            assert_eq!(editor.state.undo_depth(), steps + 1);
            editor.undo();
            assert_eq!(counts(&editor), before, "undoing takes the tube away");
        }
    }

    /// A placement click lands on the increment, and the bypass key frees it
    /// exactly as it frees a drag.
    #[test]
    fn a_placement_click_lands_on_the_increment_unless_the_bypass_is_held() {
        let (_dir, path) = write_temp("place_snap", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.pump(&mut scene);
        editor.set_containment(false);
        editor.snap_mut().millimetres = 5.0;

        editor.toggle_placing(NewObject::Keepout(ShapeKind::Tube));
        click(&mut editor, &down(-4.4, 16.6));
        assert_eq!(
            editor.placing().and_then(|placing| placing.first),
            Some([-5.0, 15.0, 0.0]),
            "the floor landing has to come back on the increment"
        );
        // Alt: the same click, exactly where it was aimed.
        editor.set_snap_bypass(true);
        click(&mut editor, &down(-4.4, 21.6));
        editor.set_snap_bypass(false);
        assert!(!editor.is_placing());

        let selection = editor.state.selection().expect("the placement selects");
        let ShapeSpec::Tube { p1, p2, .. } =
            shape_of(editor.state.config(), selection).expect("a shape")
        else {
            panic!("a tube");
        };
        assert_eq!(p1, [-5.0, 15.0, 0.0]);
        assert!(
            (p2[0] + 4.4).abs() < 1e-9 && (p2[1] - 21.6).abs() < 1e-9,
            "{p2:?}"
        );
    }

    /// Escape leaves the mode at either stage, and the button that opened it
    /// closes it. Nothing is added by either, and the selection the mode
    /// suspended comes back with its overlays.
    #[test]
    fn escape_and_the_button_both_leave_a_placement() {
        let (_dir, path) = write_temp("place_cancel", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.state.select(Some(Selection::Keepout(1)));
        editor.pump(&mut scene);
        let before = counts(&editor);
        let steps = editor.state.undo_depth();
        let what = NewObject::Keepout(ShapeKind::Tube);

        // Escape with nothing clicked yet.
        editor.toggle_placing(what);
        editor.pump(&mut scene);
        assert!(scene.get(Layer::Gizmo).is_none(), "the mode owns the view");
        assert!(editor.handles.is_empty(), "and nothing is grabbable");
        press_key(&mut editor, egui::Key::Escape);
        assert!(!editor.is_placing());

        // Escape with the first point down, which discards it.
        editor.toggle_placing(what);
        click(&mut editor, &down(30.0, 20.0));
        assert!(editor.placing().and_then(|placing| placing.first).is_some());
        press_key(&mut editor, egui::Key::Escape);
        assert!(!editor.is_placing());

        // And the button, which is a toggle.
        editor.toggle_placing(what);
        assert!(editor.is_placing());
        editor.toggle_placing(what);
        assert!(!editor.is_placing());
        // Another row's button starts that row's placement instead.
        editor.toggle_placing(what);
        click(&mut editor, &down(30.0, 20.0));
        editor.toggle_placing(NewObject::Keepin(ShapeKind::Tube));
        assert_eq!(
            editor.placing().map(|placing| placing.what),
            Some(NewObject::Keepin(ShapeKind::Tube))
        );
        assert!(
            editor.placing().and_then(|placing| placing.first).is_none(),
            "a restart starts from no points"
        );
        editor.cancel_placing();

        // Nothing was added, nothing was recorded, and what was selected before
        // is selected still - with its overlays back.
        assert_eq!(counts(&editor), before);
        assert_eq!(editor.state.undo_depth(), steps);
        assert_eq!(editor.state.selection(), Some(Selection::Keepout(1)));
        editor.pump(&mut scene);
        assert!(scene.get(Layer::Gizmo).is_some());
        assert!(scene.get(Layer::Placement).is_none());
    }

    /// While a placement owns the clicks, a click on an object places a point
    /// rather than selecting it - and the very same click outside the mode does
    /// select it, which is what makes the suspension the mode's doing.
    #[test]
    fn a_click_during_a_placement_places_instead_of_selecting() {
        let (_dir, path) = write_temp("place_suspends", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.pump(&mut scene);

        // Straight down at the fixture's sphere keepout, which is what this ray
        // selects when nothing is being placed.
        let ray = down(50.0, 16.0);
        editor.release(Some(&ray), true);
        assert_eq!(editor.state.selection(), Some(Selection::Keepout(1)));
        editor.state.select(None);

        editor.toggle_placing(NewObject::Keepout(ShapeKind::Tube));
        // The hover goes to the landing point rather than to the object under
        // it: nothing is outlined, and the marker is drawn instead.
        editor.hover_to(Some(ray));
        assert_eq!(editor.hover(), None);
        assert_eq!(
            editor.placing().and_then(|placing| placing.at),
            Some([50.0, 16.0, 32.0])
        );
        editor.pump(&mut scene);
        assert!(scene.get(Layer::Placement).is_some());
        assert!(scene.get(Layer::Hover).is_none());

        assert!(!editor.press(&ray), "no handle is grabbed while placing");
        assert!(!editor.is_dragging());
        editor.release(Some(&ray), true);
        assert_eq!(
            editor.state.selection(),
            None,
            "the click was a point, not a selection"
        );
        assert_eq!(
            editor.placing().and_then(|placing| placing.first),
            Some([50.0, 16.0, 32.0])
        );
    }

    /// The three clicks a placement ignores: one that lands nowhere, one that
    /// lands on the point already taken, and one outside the 3D view. None of
    /// them places anything, and none of them leaves the mode.
    #[test]
    fn a_placement_ignores_the_clicks_that_say_nothing() {
        let (_dir, path) = write_temp("place_ignored", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.pump(&mut scene);
        let before = counts(&editor);
        editor.toggle_placing(NewObject::Keepout(ShapeKind::Tube));

        // At the sky: no surface, and the floor is behind the ray.
        let sky = Ray {
            origin: [30.0, 20.0, 40.0],
            direction: [0.0, 0.0, 1.0],
        };
        click(&mut editor, &sky);
        assert!(editor.is_placing());
        assert!(editor.placing().and_then(|placing| placing.first).is_none());

        // Outside the 3D view, which is a click with no ray at all: the panel's
        // own business, and never a point.
        editor.release(None, true);
        assert!(editor.is_placing());
        assert!(editor.placing().and_then(|placing| placing.first).is_none());

        // Past the ruled floor, where there is nothing to land on either.
        click(&mut editor, &down(-400.0, 20.0));
        assert!(editor.placing().and_then(|placing| placing.first).is_none());

        // A real first point, and then a second one inside the increment of it:
        // the same point clicked twice makes no tube.
        click(&mut editor, &down(30.0, 20.0));
        click(&mut editor, &down(30.2, 20.1));
        assert!(editor.is_placing(), "a coincident click is ignored");
        assert_eq!(
            editor.placing().and_then(|placing| placing.first),
            Some([30.0, 20.0, 32.0]),
            "and leaves the first point where it was"
        );
        assert_eq!(counts(&editor), before, "nothing was added by any of them");

        // A second point that is really a second point still ends it.
        click(&mut editor, &down(40.0, 20.0));
        assert!(!editor.is_placing());
    }

    /// What a change to the document under a placement in progress does to it.
    ///
    /// A **structural** one - a delete, an undo, a redo - cancels the mode
    /// first, because the selection the mode promised to hand back may not be
    /// there any more and a mode holding a stale promise is worse than a mode
    /// the user re-enters with one click. A **numeric** one does not: a clicked
    /// point is a position in the world rather than a reference to an object,
    /// and nothing the properties panel types can invalidate it.
    #[test]
    fn a_structural_change_cancels_a_placement_and_a_numeric_one_does_not() {
        let (_dir, path) = write_temp("place_structural", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.state.select(Some(Selection::Keepout(1)));
        editor.pump(&mut scene);
        let what = NewObject::Keepout(ShapeKind::Tube);

        // The delete both the panel's button and the Delete key come through.
        editor.toggle_placing(what);
        click(&mut editor, &down(28.0, 20.0));
        assert!(editor.placing().and_then(|placing| placing.first).is_some());
        editor.delete_selection();
        assert!(!editor.is_placing(), "the delete took the mode with it");
        assert_eq!(editor.state.config().keepout.len(), 1);
        assert_eq!(editor.state.selection(), None);
        editor.pump(&mut scene);
        assert!(scene.get(Layer::Placement).is_none(), "and its preview");
        assert!(scene.get(Layer::Gizmo).is_none(), "nothing is selected");

        // The key reaches the same path.
        editor.state.select(Some(Selection::Keepout(0)));
        editor.toggle_placing(what);
        press_key(&mut editor, egui::Key::Delete);
        assert!(!editor.is_placing());
        assert!(editor.state.config().keepout.is_empty());

        // An undo, with a point already clicked, and a redo after it.
        editor.toggle_placing(what);
        click(&mut editor, &down(28.0, 20.0));
        editor.undo();
        assert!(!editor.is_placing(), "the undo took the mode with it");
        assert_eq!(editor.state.config().keepout.len(), 1);
        editor.toggle_placing(what);
        editor.redo();
        assert!(!editor.is_placing(), "and so does the redo");
        assert!(editor.state.config().keepout.is_empty());

        // A number typed into the properties panel is not one of them: the mode
        // and the point it is holding both survive, and the preview goes on
        // being drawn from it.
        editor.undo();
        editor.undo();
        assert_eq!(editor.state.config().keepout.len(), 2, "both are back");
        editor.state.select(Some(Selection::Keepout(1)));
        editor.pump(&mut scene);
        editor.toggle_placing(what);
        click(&mut editor, &down(28.0, 20.0));
        let first = editor.placing().and_then(|placing| placing.first);
        assert!(first.is_some());
        // What one frame of the panel does with a field that changed.
        editor.state.edit(|config| {
            config.keepout[1] = ShapeSpec::Sphere {
                center: [50.0, 16.0, 16.0],
                radius: 6.0,
            };
        });
        editor.on_edited();
        assert!(editor.is_placing(), "a value is not a structure");
        assert_eq!(editor.placing().and_then(|placing| placing.first), first);
        editor.hover_to(Some(down(40.0, 20.0)));
        editor.pump(&mut scene);
        assert!(
            scene
                .get(Layer::Placement)
                .is_some_and(|layer| layer.triangles() > 0),
            "the preview still draws from the point it is holding"
        );
        // And the placement still finishes, on the object list it was opened on.
        click(&mut editor, &down(40.0, 20.0));
        assert!(!editor.is_placing());
        assert_eq!(editor.state.config().keepout.len(), 3);
    }

    /// A placed tube goes through the containment rule exactly as a dragged one
    /// does, and the panel says when it was moved.
    #[test]
    fn a_placed_tube_is_kept_inside_the_domain_when_that_is_on() {
        let (_dir, path) = write_temp("place_containment", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.pump(&mut scene);
        assert!(editor.containment(), "containment starts on");
        let radius = state::default_tube_radius(editor.state.config());

        // Both points on the lid of the domain, which puts the tube half its
        // radius through it: the commit comes back inside.
        editor.toggle_placing(NewObject::Keepout(ShapeKind::Tube));
        click(&mut editor, &down(30.0, 20.0));
        click(&mut editor, &down(40.0, 20.0));
        let selection = editor.state.selection().expect("the placement selects");
        let ShapeSpec::Tube { p1, p2, .. } =
            shape_of(editor.state.config(), selection).expect("a shape")
        else {
            panic!("a tube");
        };
        assert_eq!(p1, [30.0, 20.0, 32.0 - radius]);
        assert_eq!(p2, [40.0, 20.0, 32.0 - radius]);
        assert_eq!(
            editor.containment_note(),
            Some(constants::VIEW_EDIT_CONTAINMENT_NOTE),
            "a clamped placement must say so"
        );

        // Switched off, the same two clicks land exactly where they were aimed.
        editor.set_containment(false);
        editor.toggle_placing(NewObject::Keepout(ShapeKind::Tube));
        click(&mut editor, &down(30.0, 20.0));
        click(&mut editor, &down(40.0, 20.0));
        let selection = editor.state.selection().expect("the placement selects");
        let ShapeSpec::Tube { p1, p2, .. } =
            shape_of(editor.state.config(), selection).expect("a shape")
        else {
            panic!("a tube");
        };
        assert_eq!(p1, [30.0, 20.0, 32.0]);
        assert_eq!(p2, [40.0, 20.0, 32.0]);
        assert!(editor.containment_note().is_none());
    }

    /// A whole editing session on an ellipsoid, through the editor's own paths:
    /// the file opens and builds, a click picks it, its overlays draw, a radius
    /// handle drags one semi-axis, the callout says which, and the save comes
    /// back as a configuration growforge reads.
    #[test]
    fn an_ellipsoid_is_picked_dragged_and_saved_like_any_other_object() {
        let (_dir, path) = write_temp("ellipsoid", ellipsoid_fixture());
        let mut editor = Editor::open(&path).expect("open");
        assert!(editor.state.is_valid(), "{:?}", editor.state.error());
        let mut scene = editor.initial_scene();
        editor.pump(&mut scene);
        assert!(
            scene.get(Layer::Keepout).is_some_and(|l| l.triangles() > 0),
            "the ellipsoid keepout has to be drawn"
        );

        // A click straight down onto it picks it, and only it: the shape is
        // 8 mm along its own x, turned 30 degrees about z, centred at
        // [50, 16, 16].
        let ray = Ray {
            origin: [50.0, 16.0, 200.0],
            direction: [0.0, 0.0, -1.0],
        };
        editor.release(Some(&ray), true);
        assert_eq!(editor.state.selection(), Some(Selection::Keepout(1)));
        assert!(editor.pump(&mut scene).overlays);
        assert!(scene.get(Layer::Selection).is_some());
        assert!(scene.get(Layer::Gizmo).is_some());
        // A click inside the box around it but outside the ellipsoid itself
        // misses: what was picked is the shape, not its bounds. The turned
        // bounds reach x = 42.79 and y = 10.71, so [43, 11] is inside them and
        // 1.19 of the way out of the ellipsoid.
        let past = Ray {
            origin: [43.0, 11.0, 200.0],
            direction: [0.0, 0.0, -1.0],
        };
        let bounds = shape_of(editor.state.config(), Selection::Keepout(1))
            .expect("a shape")
            .to_shape("test")
            .expect("well formed")
            .bounds();
        assert!(
            past.origin[0] > bounds.min[0] && past.origin[1] > bounds.min[1],
            "this test no longer describes the geometry it was written for: {bounds:?}"
        );
        editor.release(Some(&past), true);
        assert_eq!(editor.state.selection(), None);
        editor.release(Some(&ray), true);

        // Drag the handle of the first semi-axis outwards.
        editor.pump(&mut scene);
        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Radius(0))
            .copied()
            .expect("a radius handle for axis 0");
        let steps = editor.state.undo_depth();
        let press = Ray {
            origin: [handle.position[0], handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert!(editor.press(&press), "the handle must take the press");
        let along = |distance: f64| {
            let at = crate::geometry::sum(
                handle.position,
                crate::geometry::scale(handle.axis, distance),
            );
            Ray {
                origin: [at[0], at[1], 300.0],
                direction: [0.0, 0.0, -1.0],
            }
        };
        for distance in [1.0, 2.0, 3.0] {
            editor.drag_to(&along(distance));
        }
        let at = editor.callout().expect("a callout").at;
        assert_eq!(at.kind, crate::viewer::editor::measure::MeasureKind::Radius);
        assert_eq!(at.component, 0, "the callout names the dragged radius");
        assert!((at.value - 11.0).abs() < 1e-6, "showed {}", at.value);
        editor.release(None, false);
        assert_eq!(editor.state.undo_depth(), steps + 1);

        let ShapeSpec::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } = shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape")
        else {
            panic!("an ellipsoid");
        };
        assert!((radii[0] - 11.0).abs() < 1e-6, "{radii:?}");
        assert_eq!([radii[1], radii[2]], [4.0, 4.0], "one axis only");
        assert_eq!(center, [50.0, 16.0, 16.0], "a resize never moves a shape");
        assert_eq!(rotation_deg, Some([0.0, 0.0, 30.0]), "nor turns it");

        // Saved, the file holds the number and still opens.
        editor.save();
        assert!(!editor.state.is_dirty());
        let saved = std::fs::read_to_string(&path).expect("read");
        assert!(saved.contains("radii = [11.0, 4.0, 4.0]"), "{saved}");
        assert!(saved.contains("# the second keepout"), "{saved}");
        let reopened = Editor::open(&path).expect("reopen");
        assert!(reopened.state.is_valid(), "{:?}", reopened.state.error());
    }

    /// The whole callout path through a real drag: a grab raises it, the drag
    /// keeps it live, the release starts its linger, and a number typed into it
    /// lands the object exactly there - as one undo step, not two.
    #[test]
    fn a_drag_raises_a_callout_that_can_be_typed_over_as_one_undo_step() {
        let (_dir, path) = write_temp("callout", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.state.select(Some(Selection::Keepout(1)));
        editor.pump(&mut scene);

        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Translate(0))
            .copied()
            .expect("an x arrow");
        let press = Ray {
            origin: [handle.position[0], handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert!(editor.press(&press));
        let raised = editor.callout().expect("the grab raises the number box");
        assert_eq!(raised.selection, Selection::Keepout(1));
        assert_eq!(raised.at.kind, measure::MeasureKind::Offset);
        assert!(raised.at.value.abs() < 1e-12, "nothing has moved yet");

        editor.drag_to(&Ray {
            origin: [handle.position[0] + 3.0, handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        });
        assert!(
            (editor.callout().expect("a callout").at.value - 3.0).abs() < 1e-6,
            "the number must follow the drag"
        );
        // And the dimension line is drawn beside it.
        editor.pump(&mut scene);
        assert!(
            scene
                .get(Layer::Measure)
                .is_some_and(|mesh| mesh.triangles() > 0),
            "a drag draws no dimension line"
        );

        editor.release(None, false);
        let steps = editor.state.undo_depth();
        assert_eq!(
            editor.callout().expect("a callout").phase(),
            measure::Phase::Lingering
        );

        // Clicked, typed, committed.
        editor.callout_mut().expect("a callout").begin_typing();
        assert_eq!(
            editor.callout().expect("a callout").phase(),
            measure::Phase::Typing
        );
        assert!(editor.commit_callout("8"), "a number must be accepted");
        let spec = shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape");
        let ShapeSpecSphere { center, .. } = sphere_of(&spec);
        assert!(
            (center[0] - 58.0).abs() < 1e-9,
            "the typed offset is measured from where the drag started: {center:?}"
        );
        assert_eq!(
            editor.state.undo_depth(),
            steps + 1,
            "typing a value is one more step, not two"
        );
        // Undoing it puts the drag's own result back, and undoing again the
        // shape the drag started on.
        editor.undo();
        let ShapeSpecSphere { center, .. } =
            sphere_of(&shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape"));
        assert!((center[0] - 53.0).abs() < 1e-9, "{center:?}");
        editor.undo();
        let ShapeSpecSphere { center, .. } =
            sphere_of(&shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape"));
        assert!((center[0] - 50.0).abs() < 1e-9, "{center:?}");

        // What is not a number changes nothing at all.
        let before = editor.state.undo_depth();
        assert!(!editor.commit_callout("banana"));
        assert_eq!(editor.state.undo_depth(), before);
    }

    /// A number typed into a callout and the same number typed into the
    /// properties panel have to land the object in the same place: they are two
    /// spellings of one edit.
    #[test]
    fn a_typed_callout_agrees_with_the_properties_panel() {
        let (_dir, path) = write_temp("callout_parity", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        let target = Selection::Keepout(1);
        editor.state.select(Some(target));
        editor.pump(&mut scene);

        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Translate(1))
            .copied()
            .expect("a y arrow");
        let press = Ray {
            origin: [handle.position[0], handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert!(editor.press(&press));
        editor.drag_to(&Ray {
            origin: [handle.position[0], handle.position[1] + 1.0, 300.0],
            direction: [0.0, 0.0, -1.0],
        });
        editor.release(None, false);
        assert!(editor.commit_callout("6"));
        let typed = shape_of(editor.state.config(), target).expect("a shape");

        // The same number through the panel's own path: the sphere sits at
        // y = 16, so a 6 mm offset along y is y = 22.
        let mut panel = Editor::open(&path).expect("open");
        let ShapeSpecSphere { center, radius } =
            sphere_of(&shape_of(panel.state.config(), target).expect("a shape"));
        panel.state.edit(|config| {
            state::set_shape_contained(
                config,
                target,
                ShapeSpec::Sphere {
                    center: [center[0], center[1] + 6.0, center[2]],
                    radius,
                },
                true,
            );
        });
        assert_eq!(
            typed,
            shape_of(panel.state.config(), target).expect("a shape"),
            "the callout and the panel disagree about the same number"
        );
        let ShapeSpecSphere { center, .. } = sphere_of(&typed);
        assert!((center[1] - 22.0).abs() < 1e-9, "{center:?}");
    }

    /// Containment is on by default, so a drag that would take an object out of
    /// the domain stops at the wall and the panel says why.
    #[test]
    fn a_drag_is_stopped_at_the_domain_wall_and_the_note_says_so() {
        let (_dir, path) = write_temp("containment_drag", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        assert!(editor.containment(), "containment starts on");
        assert!(editor.containment_note().is_none());
        editor.state.select(Some(Selection::Keepout(1)));
        editor.pump(&mut scene);

        // The fixture's domain runs to x = 64 and the sphere is at x = 50 with
        // a radius of 4, so it may move 10 mm and no further.
        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Translate(0))
            .copied()
            .expect("an x arrow");
        let press = Ray {
            origin: [handle.position[0], handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert!(editor.press(&press));
        editor.drag_to(&Ray {
            origin: [handle.position[0] + 40.0, handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        });
        editor.release(None, false);
        let ShapeSpecSphere { center, .. } =
            sphere_of(&shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape"));
        assert!(
            (center[0] - 60.0).abs() < 1e-9,
            "the drag went through the wall to {center:?}"
        );
        assert_eq!(
            editor.containment_note(),
            Some(constants::VIEW_EDIT_CONTAINMENT_NOTE),
            "a clamped commit must say so"
        );
        // The callout measures what really happened, not what was asked for.
        assert!((editor.callout().expect("a callout").at.value - 10.0).abs() < 1e-6);

        // Switched off, the same drag goes where it was aimed.
        editor.set_containment(false);
        assert!(editor.containment_note().is_none(), "the note is cleared");
        assert!(editor.press(&press));
        editor.drag_to(&Ray {
            origin: [handle.position[0] + 40.0, handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        });
        editor.release(None, false);
        let ShapeSpecSphere { center, .. } =
            sphere_of(&shape_of(editor.state.config(), Selection::Keepout(1)).expect("a shape"));
        assert!(center[0] > 90.0, "containment was still on: {center:?}");
        assert!(editor.containment_note().is_none());
    }

    /// A load region dragged towards the pad it loads lands *on* it, and the
    /// callout names what it landed on. A keepin dragged the same way does not:
    /// it is a piece of the model rather than something placed against one.
    #[test]
    fn a_load_region_lands_flush_on_the_pad_it_loads() {
        let (_dir, path) = write_temp("surface_snap", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        let target = Selection::Load { case: 0, load: 0 };

        // The fixture's keepin pad spans x = 56 .. 64. Put the load region's
        // sphere where its own +x face is a little short of the pad's -x face,
        // and drag it the rest of the way.
        editor.state.edit(|config| {
            state::set_shape(
                config,
                target,
                ShapeSpec::Sphere {
                    center: [49.0, 16.0, 16.0],
                    radius: 6.0,
                },
            );
        });
        editor.state.select(Some(target));
        editor.pump(&mut scene);
        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Translate(0))
            .copied()
            .expect("an x arrow");
        let press = Ray {
            origin: [handle.position[0], handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert!(editor.press(&press));
        // 1.4 mm short of flush, which is inside the surface snap distance and
        // is *not* a whole millimetre: the grid alone would land on 50.
        editor.drag_to(&Ray {
            origin: [handle.position[0] + 0.6, handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        });
        let ShapeSpecSphere { center, .. } =
            sphere_of(&shape_of(editor.state.config(), target).expect("a shape"));
        assert!(
            (center[0] - 50.0).abs() < 1e-6,
            "the region landed at {center:?} rather than flush against the pad at x = 56"
        );
        let flush = editor.flush().expect("the landing must be reported");
        assert!((flush.plane.coordinate - 56.0).abs() < 1e-9);
        assert_eq!(flush.plane.what.label(), "flush on keepin 1");
        editor.release(None, false);

        // The same drag with the bypass held goes exactly where it was aimed.
        editor.set_snap_bypass(true);
        assert!(editor.press(&press));
        editor.drag_to(&Ray {
            origin: [handle.position[0] + 0.6, handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        });
        let ShapeSpecSphere { center, .. } =
            sphere_of(&shape_of(editor.state.config(), target).expect("a shape"));
        assert!((center[0] - 50.6).abs() < 1e-6, "{center:?}");
        assert!(editor.flush().is_none());
        editor.release(None, false);
        editor.set_snap_bypass(false);

        // A keepin is offered no surfaces at all, so the same near-miss lands
        // on the millimetre grid.
        assert!(state::surfaces(editor.state.config(), Selection::Keepin(0)).is_empty());
        assert!(!state::surfaces(editor.state.config(), target).is_empty());
        assert!(
            !state::surfaces(editor.state.config(), Selection::Support(0)).is_empty(),
            "a support region is placed against faces too"
        );
    }

    /// A rotation arc drag on a box, through the editor: the shape is turned by
    /// a snapped step, the callout shows the angle, and the file gets the key.
    #[test]
    fn a_rotation_arc_drag_turns_a_box_and_shows_the_angle() {
        let (_dir, path) = write_temp("rotate_drag", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        let target = Selection::Keepin(0);
        editor.state.select(Some(target));
        editor.pump(&mut scene);

        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Rotate(2))
            .copied()
            .expect("a z arc");
        let centre = gizmo::anchor(&shape_of(editor.state.config(), target).expect("a shape"));
        let (u, v) = tessellate::basis(handle.axis);
        let radius = crate::geometry::length(crate::geometry::difference(handle.position, centre));
        // A ray straight down onto a point of the arc's own plane crosses it
        // there, so the angle the drag reads is that point's own. The grab is
        // aimed at the middle of the drawn arc, which is where the handle is.
        let at = |degrees: f64| {
            let (sin, cos) = degrees.to_radians().sin_cos();
            let point = crate::geometry::sum(
                centre,
                crate::geometry::scale(
                    crate::geometry::sum(
                        crate::geometry::scale(u, cos),
                        crate::geometry::scale(v, sin),
                    ),
                    radius,
                ),
            );
            Ray {
                origin: [point[0], point[1], point[2] + 300.0],
                direction: [0.0, 0.0, -1.0],
            }
        };
        let grabbed = 0.5 * constants::VIEW_EDIT_ROTATE_ARC_SWEEP_DEGREES;
        assert!(editor.press(&at(grabbed)));
        editor.drag_to(&at(grabbed + 40.0));
        // 40 degrees swept lands on the 45 degree step.
        let spec = shape_of(editor.state.config(), target).expect("a shape");
        assert_eq!(spec.rotation_deg(), Some([0.0, 0.0, 45.0]), "{spec:?}");
        let callout = editor.callout().expect("a callout");
        assert_eq!(callout.at.kind, measure::MeasureKind::Angle);
        assert!((callout.at.value - 45.0).abs() < 1e-9);
        assert_eq!(callout.at.label(), "45.00 deg");
        editor.release(None, false);

        // Typed over with an exact angle, and saved: the key appears in the
        // file and the file still reads back as what the editor holds.
        assert!(editor.commit_callout("30"));
        assert_eq!(
            shape_of(editor.state.config(), target)
                .expect("a shape")
                .rotation_deg(),
            Some([0.0, 0.0, 30.0])
        );
        editor.save();
        let saved = std::fs::read_to_string(&path).expect("read");
        assert!(saved.contains("rotation_deg = [0.0, 0.0, 30.0]"), "{saved}");
        let reopened = Editor::open(&path).expect("reopen");
        assert!(reopened.state.is_valid(), "{:?}", reopened.state.error());
        assert_eq!(
            shape_of(reopened.state.config(), target)
                .expect("a shape")
                .rotation_deg(),
            Some([0.0, 0.0, 30.0])
        );
    }

    /// The callout is a session thing, not a document thing: it goes away when
    /// the object it measures does, when the selection changes, and when its
    /// linger runs out.
    #[test]
    fn a_callout_goes_away_with_its_object_its_selection_and_its_time() {
        let (_dir, path) = write_temp("callout_life", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        let target = Selection::Keepout(1);

        let raise = |editor: &mut Editor, scene: &mut Scene| {
            editor.state.select(Some(target));
            editor.pump(scene);
            let handle = editor
                .handles
                .iter()
                .find(|h| h.kind == gizmo::HandleKind::Translate(0))
                .copied()
                .expect("an x arrow");
            let press = Ray {
                origin: [handle.position[0], handle.position[1], 300.0],
                direction: [0.0, 0.0, -1.0],
            };
            assert!(editor.press(&press));
            editor.release(None, false);
        };

        // Selecting something else puts it away.
        raise(&mut editor, &mut scene);
        assert!(editor.callout().is_some());
        editor.release(
            Some(&Ray {
                origin: [1000.0, 1000.0, 300.0],
                direction: [0.0, 0.0, -1.0],
            }),
            true,
        );
        assert!(editor.callout().is_none(), "a new selection keeps the box");

        // So does deleting the object it measures, on the next pump.
        raise(&mut editor, &mut scene);
        assert!(editor.callout().is_some());
        editor.state.edit(|config| config.keepout.clear());
        editor.pump(&mut scene);
        assert!(editor.callout().is_none());
        assert!(scene.get(Layer::Measure).is_none());
    }

    #[test]
    fn auto_regrow_starts_a_run_after_an_edit_and_a_frame_comes_back() {
        let (_dir, path) = write_temp("auto_regrow", growth_fixture());
        let mut editor = Editor::open(&path).expect("open");
        assert!(
            editor.auto_regrow(),
            "the growth engine regrows on every edit by default"
        );
        let mut scene = editor.initial_scene();
        editor.pump(&mut scene);
        assert!(scene.get(Layer::Density).is_none(), "nothing has run yet");
        assert!(
            editor.run_note().is_none(),
            "no run has said anything yet either"
        );

        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.2));
        editor.on_edited();
        std::thread::sleep(Duration::from_secs_f64(
            constants::VIEW_EDIT_REFRESH_DEBOUNCE_S * 2.0,
        ));
        editor.pump(&mut scene);
        assert_eq!(editor.worker.kind(), Some(RunKind::Preview));

        let deadline = Instant::now() + Duration::from_secs(120);
        while Instant::now() < deadline {
            editor.pump(&mut scene);
            if scene.get(Layer::Density).is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let density = scene
            .get(Layer::Density)
            .expect("the re-run produced no surface");
        assert!(density.triangles() > 0);
        assert!(editor.frame_label().is_some_and(|l| l.contains("preview")));
        editor.detach();
        editor.finish();

        // A preview writes nothing: the STL the configuration names is absent.
        let stl = path.with_file_name("fixture.stl");
        assert!(!stl.exists(), "a preview wrote {}", stl.display());
        // And the file itself is untouched.
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            growth_fixture()
        );
    }

    /// The same, on the shipped growth example itself - the one whose seed is
    /// larger than the format preserving parser can hold, whose canopy is what
    /// the editor exists to let someone tune, and whose file must come back
    /// untouched.
    #[test]
    fn the_shipped_growth_example_regrows_under_an_edit_and_is_left_alone() {
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("growth_canopy.toml");
        let text = std::fs::read_to_string(&source).expect("read the example");
        let (_dir, path) = write_temp("growth_canopy", &text);
        let mut editor = Editor::open(&path).expect("open");
        assert!(editor.auto_regrow());
        assert!(editor.state.is_valid(), "{:?}", editor.state.error());
        let mut scene = editor.initial_scene();

        // A committed edit of a growth control, the way the panel makes one.
        editor.state.edit(|config| {
            config.optimization.mass_fraction = Some(0.1);
        });
        editor.on_edited();
        std::thread::sleep(Duration::from_secs_f64(
            constants::VIEW_EDIT_REFRESH_DEBOUNCE_S * 2.0,
        ));
        editor.pump(&mut scene);
        assert_eq!(editor.worker.kind(), Some(RunKind::Preview));

        let deadline = Instant::now() + Duration::from_secs(300);
        while Instant::now() < deadline {
            editor.pump(&mut scene);
            if scene.get(Layer::Density).is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            scene
                .get(Layer::Density)
                .is_some_and(|mesh| mesh.triangles() > 0),
            "the re-grown canopy never reached the scene"
        );
        editor.detach();
        editor.finish();

        // Nothing was written, and a save gives the file back byte for byte
        // apart from the one value that changed - the huge seed included.
        assert!(!path.with_file_name("growth_canopy.stl").exists());
        assert_eq!(std::fs::read_to_string(&path).expect("read"), text);
        editor.save();
        let saved = std::fs::read_to_string(&path).expect("read");
        assert_eq!(
            saved,
            text.replace("mass_fraction = 0.12", "mass_fraction = 0.1")
        );
        assert!(saved.contains("seed = 11396317718348371989"), "{saved}");
    }

    /// The stop button, on the run an edit started. Nothing is adopted, the
    /// panel says what happened, and the editor is immediately usable again.
    #[test]
    fn stopping_a_preview_ends_it_and_leaves_the_editor_usable() {
        let (_dir, path) = write_temp("stop_preview", growth_fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.2));
        editor.on_edited();
        std::thread::sleep(Duration::from_secs_f64(
            constants::VIEW_EDIT_REFRESH_DEBOUNCE_S * 2.0,
        ));
        editor.pump(&mut scene);
        assert!(
            editor.is_running(),
            "the edit should have started a preview"
        );

        editor.stop_run();
        assert!(!editor.is_running());
        assert!(editor.worker.kind().is_none());
        assert!(
            editor.status_line().expect("a status").contains("stopped"),
            "{:?}",
            editor.status_line()
        );
        // Nothing the stopped run may have produced is adopted afterwards.
        editor.pump(&mut scene);
        assert!(scene.get(Layer::Density).is_none());

        // Stopping again is a no-op, and so is stopping when nothing runs.
        let before = editor.status_line().map(str::to_string);
        editor.stop_run();
        assert_eq!(editor.status_line().map(str::to_string), before);

        // And the editor still works: another edit starts another run.
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.25));
        editor.on_edited();
        std::thread::sleep(Duration::from_secs_f64(
            constants::VIEW_EDIT_REFRESH_DEBOUNCE_S * 2.0,
        ));
        editor.pump(&mut scene);
        assert_eq!(editor.worker.kind(), Some(RunKind::Preview));
        editor.detach();
        editor.finish();
    }

    /// The same button on the full pipeline. The one thing that must not happen
    /// is a file: a stopped run has not decided what the part is.
    #[test]
    fn stopping_a_full_run_writes_no_file_and_returns_to_editing() {
        let (_dir, path) = write_temp("stop_full", growth_fixture());
        let stl = path.with_file_name("fixture.stl");
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        editor.start_full_run();
        assert!(editor.is_running_full());
        editor.stop_run();
        assert!(!editor.is_running(), "the run must be gone from the panel");
        assert!(
            !editor.is_running_full(),
            "a stopped full run must hand \"run full\" back at once, whatever \
             its thread is still winding down through"
        );
        assert!(
            editor.status_line().expect("a status").contains("stopped"),
            "{:?}",
            editor.status_line()
        );
        editor.finish();
        assert!(!stl.exists(), "a stopped full run wrote {}", stl.display());

        // Straight back to work: a full run started again behaves normally.
        editor.start_full_run();
        editor.finish();
        editor.pump(&mut scene);
        assert!(stl.exists(), "the re-run must write its file");
        assert!(!editor.is_running());
        std::fs::remove_file(&stl).ok();
        std::fs::remove_file(path.with_file_name("fixture_stress.json")).ok();
    }

    /// "generate stl", through the editor, on the design a run that wrote nothing
    /// left on screen - which is what the button exists for.
    ///
    /// The incident: a run of five hundred and twenty-seven iterations was
    /// stopped, and stopping it turned out to write nothing at all, so the design
    /// on screen could not be had as a file. It can now, and the automatic
    /// behaviour is unchanged: the asking is what writes.
    ///
    /// The run here is the preview an edit starts, which writes nothing by
    /// definition and is over in milliseconds on this engine. What a *stopped*
    /// full run leaves behind is pinned where it can be caught mid-flight
    /// deterministically - see
    /// `worker::tests::a_run_that_was_stopped_can_still_have_its_design_generated`.
    #[test]
    fn generating_writes_the_design_a_run_left_on_screen_without_writing() {
        let (_dir, path) = write_temp("generate_stl", growth_fixture());
        let stl = path.with_file_name("fixture.stl");
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        // Nothing has run: there is nothing on screen and nothing to export.
        assert!(!editor.can_generate_stl());
        editor.generate_stl();
        assert!(editor.status_line().is_none(), "a guard, not a path");

        // The preview an edit starts. It exports nothing of its own.
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.2));
        editor.on_edited();
        std::thread::sleep(Duration::from_secs_f64(
            constants::VIEW_EDIT_REFRESH_DEBOUNCE_S * 2.0,
        ));
        editor.pump(&mut scene);
        let deadline = Instant::now() + Duration::from_secs(300);
        while Instant::now() < deadline && !editor.can_generate_stl() {
            editor.pump(&mut scene);
            std::thread::sleep(Duration::from_millis(20));
        }
        // Stopped, if it is still going - which changes nothing about either of
        // the two things that follow.
        editor.stop_run();
        editor.finish();
        assert!(
            !stl.exists(),
            "a preview wrote {} on its own",
            stl.display()
        );
        assert!(
            editor.can_generate_stl(),
            "the design the run reached was thrown away with it"
        );

        // Asked for by name, the design on screen becomes the deliverable set.
        editor.generate_stl();
        let status = editor.status_line().expect("a status").to_string();
        assert!(status.contains("generating"), "{status}");
        assert!(status.contains("fixture.stl"), "{status}");
        assert_eq!(editor.worker.kind(), Some(RunKind::Export));
        assert!(editor.is_writing() && !editor.is_running_full());
        let deadline = Instant::now() + Duration::from_secs(300);
        while Instant::now() < deadline && editor.is_running() {
            editor.pump(&mut scene);
            std::thread::sleep(Duration::from_millis(20));
        }
        editor.pump(&mut scene);
        assert!(!editor.is_running(), "the generation never ended");
        assert!(stl.exists(), "the generation wrote no file");
        // And the panel moves off the line that said it was starting: a
        // generation that is over is a file that is there.
        let status = editor.status_line().expect("a status").to_string();
        assert!(
            status.starts_with("generated ") && status.contains("fixture.stl"),
            "{status}"
        );

        // And the window shows what it wrote: the exported surface, with the
        // stress colouring the panel's own switch hangs off.
        assert_eq!(editor.frame_kind, Some(FrameKind::Final));
        assert_eq!(
            editor.frame_label().as_deref(),
            Some("stl generation exported mesh")
        );
        assert!(
            scene
                .get(Layer::Density)
                .is_some_and(|mesh| mesh.triangles() > 0)
        );
        assert!(scene.has_stress(), "no stress layer reached the panel");
        assert!(
            editor.run_line().expect("a run line").contains("done"),
            "{:?}",
            editor.run_line()
        );
        // Its own file is recognized as its own, so a run started afterwards does
        // not accuse the session of it.
        assert!(editor.output_warning(&stl).is_none());
        std::fs::remove_file(&stl).ok();
    }

    /// A run that ends on its own says so, in place of the line that said it was
    /// starting.
    ///
    /// The defect: the panel had a transition for a run that was stopped and one
    /// for a run that failed, and none for a run that finished - so "running the
    /// full pipeline; it will write ..." stayed on it for the rest of the
    /// session, describing a run that was over and a file that was already there.
    /// The solid engine, whose whole run is one frame, is what made it plain, and
    /// is what this runs for the speed of it; nothing here is that engine's.
    #[test]
    fn a_full_run_that_finishes_says_so_and_says_it_once() {
        let (_dir, path) = write_temp("full_run_finished", solid_fixture());
        let stl = path.with_file_name("fixture.stl");
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        assert!(editor.state.is_valid(), "{:?}", editor.state.error());

        editor.start_full_run();
        assert!(
            editor
                .status_line()
                .expect("a status")
                .contains("running the full pipeline"),
            "{:?}",
            editor.status_line()
        );

        let deadline = Instant::now() + Duration::from_secs(300);
        while Instant::now() < deadline && editor.is_running() {
            editor.pump(&mut scene);
            std::thread::sleep(Duration::from_millis(20));
        }
        editor.pump(&mut scene);
        assert!(!editor.is_running(), "the run never ended");
        assert!(stl.exists(), "the full run wrote no file");

        // What the panel says once it is over: that it is, and where the file
        // went.
        let status = editor.status_line().expect("a status").to_string();
        assert!(
            !status.contains("it will write"),
            "the panel is still describing a run that is over: {status}"
        );
        assert!(
            status.contains("finished") && status.contains("fixture.stl"),
            "{status}"
        );

        // Once. A finished run stays the current one until the next starts, so a
        // pump a frame later must not write its line over whatever has been said
        // since.
        editor.status = Some("a later message".to_string());
        editor.pump(&mut scene);
        assert_eq!(
            editor.status_line(),
            Some("a later message"),
            "the finished run reported itself a second time"
        );

        // And the run after it puts the running line back.
        editor.start_full_run();
        assert!(
            editor
                .status_line()
                .expect("a status")
                .contains("running the full pipeline"),
            "{:?}",
            editor.status_line()
        );
        editor.stop_run();
        editor.finish();
        std::fs::remove_file(&stl).ok();
    }

    /// A run that fails is the run's failure, not the session's.
    ///
    /// The incident: a full run on a marginally connected structure failed its
    /// linear solve, and the failure read as the end of the program. It is not.
    /// The window stays, the panel says what happened in the solver's own words,
    /// nothing is written, and the next run starts from the same panel.
    #[test]
    fn a_failed_full_run_leaves_the_session_open_and_usable() {
        let (_dir, path) = write_temp("failed_run", STRANDED);
        let stl = path.with_file_name("stranded.stl");
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        assert!(editor.state.is_valid(), "{:?}", editor.state.error());

        editor.start_full_run();
        assert!(editor.is_running_full(), "the run under test never started");
        let deadline = Instant::now() + Duration::from_secs(120);
        while Instant::now() < deadline && editor.is_running() {
            editor.pump(&mut scene);
            std::thread::sleep(Duration::from_millis(10));
        }
        editor.pump(&mut scene);
        assert!(!editor.is_running(), "the failed run never ended");
        assert!(
            !editor.is_running_full(),
            "a failed run must hand the worker back"
        );

        // What the panel says: the solver's own message, and that the session
        // is still there.
        let status = editor.status_line().expect("a status").to_string();
        assert!(status.contains("failed"), "{status}");
        assert!(
            status.contains("not positive definite"),
            "the solver's own words are the actionable ones: {status}"
        );
        assert!(status.contains("run again"), "{status}");
        let run_line = editor.run_line().expect("a run line");
        assert!(run_line.contains("failed"), "{run_line}");

        // Nothing was written, exactly as for a stopped run.
        assert!(!stl.exists(), "a failed run wrote {}", stl.display());

        // And the session is fully usable: the document edits, and another full
        // run starts rather than being swallowed by the failed one.
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.4));
        editor.on_edited();
        std::thread::sleep(Duration::from_secs_f64(
            constants::VIEW_EDIT_REFRESH_DEBOUNCE_S * 2.0,
        ));
        editor.pump(&mut scene);
        assert!(editor.state.is_dirty());
        editor.start_full_run();
        assert!(
            editor.is_running_full(),
            "the session refused a second run after a failure"
        );
        assert!(
            !editor.status_line().expect("a status").contains("failed"),
            "the new run inherited the old one's failure: {:?}",
            editor.status_line()
        );
        editor.stop_run();
        editor.finish();
        assert!(!stl.exists());
    }

    /// A full run that fails while an edit has a preview queued behind it.
    ///
    /// Both things have to happen: the failure has to be said, and the queued
    /// preview has to run. The order of `pump` is what decides it - starting the
    /// queued preview replaces the run whose failure has not been read yet - so
    /// this pins the order rather than the statements.
    #[test]
    fn a_full_run_that_fails_with_a_preview_queued_reports_it_and_still_regrows() {
        let (_dir, path) = write_temp("failed_with_queue", STRANDED);
        let stl = path.with_file_name("stranded.stl");
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.set_auto_regrow(true);
        editor.regrow_owed = false;

        editor.start_full_run();
        assert!(editor.is_running_full());
        // What the debounce block above leaves behind when a committed edit
        // lands while a full run owns the worker: the preview is owed, not
        // started. Set here rather than raced for, because this fixture fails in
        // a fraction of the debounce interval.
        editor.regrow_owed = true;

        // Let the run fail on its own, without pumping: the failure and the
        // queued preview are then both waiting for the same pump.
        let deadline = Instant::now() + Duration::from_secs(120);
        while Instant::now() < deadline && editor.is_running() {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!editor.is_running(), "the failed run never ended");

        editor.pump(&mut scene);
        let status = editor.status_line().expect("a status").to_string();
        assert!(
            status.contains("failed") && status.contains("not positive definite"),
            "the failure was swallowed by the preview that replaced it: {status}"
        );
        assert_eq!(
            editor.worker.kind(),
            Some(RunKind::Preview),
            "the queued preview must still run"
        );
        assert!(!editor.regrow_owed, "and must not stay owed");

        editor.stop_run();
        editor.finish();
        assert!(!stl.exists());
    }

    /// The stop button, pressed while the run is deep inside one long linear
    /// solve. Before the solver learned to answer the question, this waited for
    /// that whole solve; now the run ends within a checkpoint of being asked.
    #[test]
    fn stopping_a_run_inside_a_long_solve_ends_it_promptly() {
        let (_dir, path) = write_temp("stop_in_solve", SLOW);
        let stl = path.with_file_name("slow.stl");
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        assert!(editor.state.is_valid(), "{:?}", editor.state.error());

        editor.start_full_run();
        assert!(editor.is_running_full());
        // Long enough to be inside the first solve rather than in front of it,
        // and far short of the tens of seconds that solve takes.
        std::thread::sleep(Duration::from_millis(400));
        assert!(
            editor.is_running_full(),
            "the fixture solved too quickly to be a test of stopping inside one"
        );

        let asked = Instant::now();
        editor.stop_run();
        assert!(!editor.is_running(), "the panel forgets a stopped run");
        assert!(
            editor.status_line().expect("a status").contains("stopped"),
            "{:?}",
            editor.status_line()
        );
        editor.finish();
        let took = asked.elapsed();
        println!(
            "the stop took {:.3} s to reach the solver",
            took.as_secs_f64()
        );
        // The bound is the checkpoint's, with room for a debug build: 32
        // conjugate gradient iterations on 27 648 elements, plus joining the
        // mesher. On this machine it measures 0.06 s in release; the same solve
        // run to its own end takes 36 s, and the whole capped run 36 s more.
        assert!(
            took < Duration::from_secs(8),
            "the stop took {:.3} s, which is a solve rather than a checkpoint",
            took.as_secs_f64()
        );

        // Nothing was written, and the session is usable.
        assert!(!stl.exists(), "a stopped run wrote {}", stl.display());
        editor.pump(&mut scene);
        editor.start_full_run();
        assert!(editor.is_running_full());
        editor.stop_run();
        editor.finish();
        assert!(!stl.exists());
    }

    /// The hover state machine, driven by rays rather than by pixels: the
    /// re-pick guard, the rank rule, the gizmo handles and the drag.
    #[test]
    fn the_hover_picks_like_a_click_guards_re_picks_and_defers_to_drags() {
        let (_dir, path) = write_temp("hover_state", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.pump(&mut scene);

        // Straight down at the load region, which sits inside the keepin pad:
        // two shapes on one ray, and rank rather than depth decides.
        let load = Selection::Load { case: 0, load: 0 };
        let at_load = Ray {
            origin: [60.0, 16.0, 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert_eq!(
            pick::nearest(&at_load, &state::targets(editor.state.config())),
            Some(load),
            "the fixture no longer overlaps, so this proves nothing"
        );
        editor.hover_to(Some(at_load));
        assert_eq!(editor.hover(), Some(load));
        assert!(editor.pump(&mut scene).overlays);
        assert!(
            scene
                .get(Layer::Hover)
                .is_some_and(|mesh| mesh.triangles() > 0)
        );

        // The same ray again changes nothing, which is the guard: no re-pick,
        // no rebuild, no upload.
        editor.hover_to(Some(at_load));
        assert!(
            editor.pump(&mut scene).is_empty(),
            "an unmoved ray must cost nothing"
        );

        // And what it previewed is what the click takes.
        editor.release(Some(&at_load), true);
        assert_eq!(editor.state.selection(), Some(load));
        editor.pump(&mut scene);
        assert!(
            scene.get(Layer::Hover).is_none(),
            "the selected object needs no second outline"
        );

        // Over a gizmo handle: a click there would grab the handle, so nothing
        // is outlined and the handle is what reports the hover.
        editor.state.select(Some(Selection::Keepout(1)));
        editor.pump(&mut scene);
        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == gizmo::HandleKind::Translate(0))
            .copied()
            .expect("an x arrow");
        let at_handle = Ray {
            origin: [handle.position[0], handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        editor.hover_to(Some(at_handle));
        assert_eq!(editor.hover(), None);
        assert_eq!(
            editor.hovered_handle(),
            Some(gizmo::HandleKind::Translate(0))
        );
        editor.pump(&mut scene);
        assert!(scene.get(Layer::Hover).is_none());

        // A drag owns the pointer: nothing is hovered, and the grabbed handle
        // is not left lit either.
        assert!(editor.press(&at_handle));
        assert_eq!(editor.hover(), None);
        assert_eq!(editor.hovered_handle(), None);
        editor.hover_to(Some(at_load));
        assert_eq!(
            editor.hover(),
            None,
            "a ray straight at an object must not hover while a drag is on"
        );
        editor.pump(&mut scene);
        assert!(scene.get(Layer::Hover).is_none());

        // The release hands the pointer back, and the next ray picks again.
        editor.release(None, false);
        editor.hover_to(Some(at_load));
        assert_eq!(editor.hover(), Some(load));

        // Nothing under the pointer at all clears it.
        editor.hover_to(None);
        assert_eq!(editor.hover(), None);
        editor.pump(&mut scene);
        assert!(scene.get(Layer::Hover).is_none());
    }

    /// The floor grid: ruled at the increment the panel is set to, on the floor
    /// of the domain, and a layer like any other.
    #[test]
    fn the_floor_grid_follows_the_snap_increment_and_switches_off() {
        let (_dir, path) = write_temp("floor_grid", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        assert!(
            scene.is_visible(Layer::Grid),
            "the grid starts on in edit mode"
        );
        assert!(scene.get(Layer::Grid).is_none(), "and starts with no lines");

        assert!(editor.pump(&mut scene).overlays);
        let derived = editor.floor_grid().expect("a grid");
        assert_eq!(derived.spacing_mm, constants::VIEW_EDIT_SNAP_MM);
        let lines = derived.line_count();
        assert!(lines > 0 && derived.note().is_none());
        let layer = scene.get(Layer::Grid).expect("grid geometry");
        assert!(layer.triangles() > 0);
        // The fixture's domain sits on z = 0, and the grid is on its floor.
        assert!(layer.bounds.min[2].abs() < 1.0 && layer.bounds.max[2].abs() < 1.0);
        // It reaches past the domain's footprint on both axes.
        assert!(layer.bounds.min[0] < 0.0 && layer.bounds.max[0] > 64.0);
        assert!(layer.bounds.min[1] < 0.0 && layer.bounds.max[1] > 32.0);

        // The panel's snap control re-rules it, and nothing else does.
        assert!(
            editor.pump(&mut scene).is_empty(),
            "a still frame re-derives"
        );
        editor.snap_mut().millimetres = 5.0;
        assert!(editor.pump(&mut scene).overlays, "the grid did not follow");
        let coarser = editor.floor_grid().expect("a grid");
        assert_eq!(coarser.spacing_mm, 5.0);
        assert!(
            coarser.line_count() < lines,
            "{} against {lines}",
            coarser.line_count()
        );

        // And it is a layer: switching it off leaves the geometry where it is.
        *scene.visibility_mut(Layer::Grid) = false;
        assert!(!scene.is_visible(Layer::Grid));
        assert!(scene.get(Layer::Grid).is_some());

        // A setup view builds no grid at all, so `view` and `run --view` are
        // exactly as they were.
        let problem = editor.state.problem().expect("a problem");
        let built = scene::build(editor.state.config(), problem).expect("build");
        assert!(built.get(Layer::Grid).is_none());
    }

    /// The overwrite warning has to tell the file this editor wrote from one
    /// something else did, or the second click of "run full" accuses the
    /// session of its own work.
    #[test]
    fn the_overwrite_warning_tells_our_own_output_from_somebody_elses() {
        let (_dir, path) = write_temp("output_warning", growth_fixture());
        let stl = path.with_file_name("fixture.stl");
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        // A first run, with no file there at all: nothing to warn about.
        assert!(!stl.exists());
        assert!(editor.output_warning(&stl).is_none());
        editor.start_full_run();
        assert!(
            !editor
                .status_line()
                .expect("a status")
                .contains("outside this editor"),
            "{:?}",
            editor.status_line()
        );
        editor.finish();
        editor.pump(&mut scene);
        assert!(stl.exists(), "the full run wrote no file");

        // The file is now ours, and a second run of the same session says so.
        assert!(
            editor.output_warning(&stl).is_none(),
            "{:?}",
            editor.output_warning(&stl)
        );
        editor.start_full_run();
        assert!(
            !editor
                .status_line()
                .expect("a status")
                .contains("outside this editor"),
            "the second run accused this session of its own file: {:?}",
            editor.status_line()
        );
        editor.finish();

        // Something else writes the same path between two runs, which is the
        // case the warning exists for.
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&stl)
            .expect("open the output");
        let later = std::time::SystemTime::now()
            + Duration::from_secs_f64(constants::VIEW_EDIT_OUTPUT_MTIME_TOLERANCE_S)
            + Duration::from_secs(5);
        file.set_modified(later).expect("set the mtime");
        drop(file);
        assert!(
            editor
                .output_warning(&stl)
                .is_some_and(|warning| warning.contains("outside this editor")),
            "a file written elsewhere must still be reported: {:?}",
            editor.output_warning(&stl)
        );

        std::fs::remove_file(&stl).ok();
    }

    /// An editor window is its session: closing it stops the runs behind it,
    /// and nothing carries on headless the way `run --view` deliberately does.
    #[test]
    fn closing_the_editor_stops_everything_behind_it() {
        let (_dir, path) = write_temp("close_stops", growth_fixture());
        let stl = path.with_file_name("fixture.stl");
        let mut editor = Editor::open(&path).expect("open");

        editor.start_full_run();
        assert!(editor.is_running_full());
        // What the window does on its way down.
        editor.detach();
        editor.finish();
        assert!(!editor.is_running());
        assert!(
            !stl.exists(),
            "a run in flight at close wrote {}",
            stl.display()
        );

        // And the same for a preview.
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.2));
        editor.on_edited();
        std::thread::sleep(Duration::from_secs_f64(
            constants::VIEW_EDIT_REFRESH_DEBOUNCE_S * 2.0,
        ));
        editor.pump(&mut scene);
        assert!(editor.is_running());
        editor.detach();
        editor.finish();
        assert!(!editor.is_running());
        assert!(!stl.exists());
    }

    /// `growforge edit` on a name that is not there yet writes a starter
    /// configuration and opens it - a real problem, not a fragment.
    #[test]
    fn editing_a_path_that_does_not_exist_scaffolds_a_configuration() {
        let (_dir, existing) = write_temp("scaffold", fixture());
        let path = existing.with_file_name("brand_new_part.toml");
        assert!(!path.exists());

        let editor = Editor::open_or_create(&path).expect("scaffold and open");
        assert!(path.exists(), "the starter configuration was not written");
        assert!(
            editor.state.is_valid(),
            "the starter configuration must validate as written: {:?}",
            editor.state.error()
        );
        assert!(!editor.state.is_dirty(), "a fresh file has nothing to save");
        assert!(
            editor.state.warnings().is_empty(),
            "the starter configuration warns: {:?}",
            editor.state.warnings()
        );
        // It is a problem: a domain, something to stand on, something pushing.
        let config = editor.state.config();
        assert_eq!(config.engine_name().unwrap(), constants::STARTER_ENGINE);
        assert!(!config.domain.is_empty());
        assert!(!config.keepin.is_empty());
        assert!(!config.supports.is_empty());
        assert_eq!(config.loadcases.len(), 1);
        assert_eq!(config.loadcases[0].loads.len(), 1);
        assert_eq!(config.project.name, "brand_new_part");
        assert_eq!(config.output.stl_path, "brand_new_part.stl");
        // And what was written is exactly what a re-open reads.
        let reopened = Editor::open(&path).expect("reopen");
        assert!(reopened.state.is_valid());

        // An existing file is never scaffolded over.
        let before = std::fs::read_to_string(&existing).expect("read");
        let opened = Editor::open_or_create(&existing).expect("open the existing file");
        assert!(opened.state.is_valid());
        assert_eq!(std::fs::read_to_string(&existing).expect("read"), before);
        assert!(EditorState::create(&existing).is_err(), "never overwrite");
    }

    #[test]
    fn switching_auto_regrow_off_cancels_the_run_it_started() {
        let (_dir, path) = write_temp("auto_off", growth_fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.2));
        editor.on_edited();
        std::thread::sleep(Duration::from_secs_f64(
            constants::VIEW_EDIT_REFRESH_DEBOUNCE_S * 2.0,
        ));
        editor.pump(&mut scene);
        assert!(editor.worker.kind().is_some());
        editor.set_auto_regrow(false);
        assert_eq!(editor.worker.kind(), None, "the preview was cancelled");
        editor.finish();
    }

    /// The fields of a sphere, so the assertions above read as prose.
    struct ShapeSpecSphere {
        center: Vec3,
        radius: f64,
    }

    fn sphere_of(spec: &crate::config::ShapeSpec) -> ShapeSpecSphere {
        match *spec {
            crate::config::ShapeSpec::Sphere { center, radius } => {
                ShapeSpecSphere { center, radius }
            }
            _ => panic!("expected a sphere"),
        }
    }
}
