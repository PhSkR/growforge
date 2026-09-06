//! The winit application: window lifecycle, input, and the per-frame loop that
//! drains the preview channel and repaints.
//!
//! The event loop must own the main thread on Windows, so everything expensive
//! happens elsewhere: the optimization on a worker thread and the preview
//! isosurface extraction on a mesher thread. This module only uploads finished
//! vertex buffers and draws.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::constants;
use crate::viewer::camera::OrbitCamera;
use crate::viewer::editor::pick::{self, Ray};
use crate::viewer::editor::{Editor, Pick, Refresh, dialog};
use crate::viewer::gpu::{EguiFrame, Gpu};
use crate::viewer::scene::{LAYERS, Layer, LayerMesh, LayerRole, Scene};
use crate::viewer::snapshot::{FrameKind, ViewLink};
use crate::viewer::ui;

/// Pointer state, tracked across events.
#[derive(Debug, Clone, Copy, Default)]
struct Pointer {
    /// Where the pointer is now, in physical pixels from the top left of the
    /// window's client area.
    ///
    /// Updated on every `CursorMoved` before anything else looks at the event,
    /// including the ones the camera never sees - the ones egui consumed, and
    /// the ones a gizmo drag took. A press carries no position of its own, so
    /// this is what its ray is cast through: reading a position that only the
    /// camera path maintained left a press after any drag casting from wherever
    /// that drag began.
    position: Option<(f64, f64)>,
    /// Where the camera's last drag delta was measured from. The camera path's
    /// own bookkeeping, deliberately separate from `position`.
    ///
    /// A gizmo drag claims every pointer move it sees, so the camera path does
    /// not run for any of them and this would stand still for the whole drag: the
    /// next camera gesture would then measure its first delta from wherever that
    /// drag began and snap by everything it covered.
    ///
    /// The invariant is therefore that this is always the last position a camera
    /// delta could be asked to measure from, and it is *the claimed move* in
    /// [`ViewerApp::handle_editor_event`] that keeps it: every position a drag
    /// takes is written here as it passes. The grab and the release write it too,
    /// which is belt and braces rather than a second mechanism - by then the
    /// claimed moves have already left it current - and they say so where they
    /// stand.
    last: Option<(f64, f64)>,
    orbiting: bool,
    panning: bool,
    /// Physical pixels the pointer has travelled since the left button went
    /// down. Below [`constants::VIEW_EDIT_CLICK_SLOP_PIXELS`] the gesture was a
    /// click rather than an orbit, which is how the editor gets a select out of
    /// the same button that orbits.
    travel: f64,
}

impl Pointer {
    fn dragging(&self) -> bool {
        self.orbiting || self.panning
    }
}

/// Seconds after the first frame at which the window closes itself, read from
/// [`constants::VIEW_AUTOCLOSE_ENV`]. Unset, empty or unparsable means never,
/// so a stray value can never cut an interactive session short.
pub fn autoclose_after() -> Option<Duration> {
    let raw = std::env::var(constants::VIEW_AUTOCLOSE_ENV).ok()?;
    let seconds: f64 = raw.trim().parse().ok()?;
    (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds))
}

/// Aspect ratio of a viewport, guarding the degenerate zero-height case a
/// minimized window reports on Windows.
fn aspect_of(width: u32, height: u32) -> f64 {
    if height == 0 {
        1.0
    } else {
        width as f64 / height as f64
    }
}

/// True while the shortcut modifier is held: the control key everywhere but
/// macOS, where the same shortcuts are spelled with the command key. Which is
/// what egui itself calls `command`, and these are its bindings.
fn command_held(modifiers: &ModifiersState) -> bool {
    if cfg!(target_os = "macos") {
        modifiers.super_key()
    } else {
        modifiers.control_key()
    }
}

/// True when this press is the `V` of a paste: the letter the layout produced,
/// or - when it produced no Latin letter at all - the position that letter is
/// in on a Latin layout.
///
/// Both, in that order, which is how the egui backend decides the same question
/// for the bindings it answers itself: a Dvorak layout reaches the paste by the
/// key that types a `v` rather than by the one a US layout puts it on, and a
/// layout with no Latin letters at all - Cyrillic, Greek - reaches it by
/// position, because there is no letter there to match.
fn is_paste_chord(logical: &winit::keyboard::Key, physical: PhysicalKey) -> bool {
    match logical {
        winit::keyboard::Key::Character(text) if text.is_ascii() => text.eq_ignore_ascii_case("v"),
        _ => physical == PhysicalKey::Code(KeyCode::KeyV),
    }
}

/// Physical pixels left for the 3D view once an egui panel `panel_points` wide
/// has taken its share of a surface `surface_width` pixels wide.
fn scene_width_for(surface_width: u32, pixels_per_point: f32, panel_points: f32) -> u32 {
    let panel = (panel_points * pixels_per_point).round() as u32;
    surface_width.saturating_sub(panel).max(1)
}

/// The viewer window.
pub struct ViewerApp<'a> {
    title: String,
    scene: Scene,
    camera: OrbitCamera,
    link: Option<&'a ViewLink>,
    autoclose: Option<Duration>,
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    pointer: Pointer,
    /// Keyboard modifiers as winit last reported them. The editor's snap bypass
    /// reads Alt from here rather than from egui, because the gizmo input path
    /// runs before egui has a frame and, in the input-simulation harness, with
    /// no egui state at all.
    modifiers: ModifiersState,
    fitted: bool,
    /// Physical pixels of the surface the 3D view owns, the rest being the
    /// egui panel. Known only once a frame has reported its scale factor.
    scene_width: Option<u32>,
    /// Physical size of the drawing surface, mirrored from the GPU so that the
    /// input path - which has to answer a pointer position without drawing
    /// anything - does not depend on a device being open.
    surface_size: Option<(u32, u32)>,
    first_frame: Option<Instant>,
    frames: u64,
    frame_kind: Option<FrameKind>,
    /// Whether the density buffer on the GPU currently holds the stress
    /// coloured geometry, so a flip of the panel toggle can be noticed.
    stress_shading: bool,
    /// Whether that buffer currently holds per-triangle normals, for the same
    /// reason.
    flat_shading: bool,
    error: Option<anyhow::Error>,
    /// Reports whether the run behind the window is still going. `None` for a
    /// setup view, which has no run to outlive it.
    still_running: Option<&'a dyn Fn() -> bool>,
    /// The editing session, when this window is `growforge edit`. `None` leaves
    /// every path below exactly as `view` and `run --view` have them.
    editor: Option<Editor>,
    /// Engine key of the run behind the window, for the stats block: an engine
    /// that reports no iterations has to be told apart from one whose first
    /// iteration has not landed yet. `None` for a window with no run.
    engine: Option<String>,
    /// Set once the window and its GPU resources have been released.
    torn_down: bool,
    summary_printed: bool,
}

impl<'a> ViewerApp<'a> {
    /// Build a window over `scene`. `link` is `None` for a setup view.
    pub fn new(title: String, scene: Scene, link: Option<&'a ViewLink>) -> ViewerApp<'a> {
        ViewerApp {
            title,
            scene,
            camera: OrbitCamera::default(),
            link,
            autoclose: autoclose_after(),
            window: None,
            gpu: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            pointer: Pointer::default(),
            modifiers: ModifiersState::empty(),
            fitted: false,
            scene_width: None,
            surface_size: None,
            first_frame: None,
            frames: 0,
            frame_kind: None,
            stress_shading: false,
            flat_shading: constants::VIEW_DEFAULT_FLAT_SHADING,
            error: None,
            still_running: None,
            editor: None,
            engine: None,
            torn_down: false,
            summary_printed: false,
        }
    }

    /// Name the engine the run behind this window is using, so the stats block
    /// can say what a run that reports no iterations at all is doing.
    pub fn running(mut self, engine: &str) -> ViewerApp<'a> {
        self.engine = Some(engine.to_string());
        self
    }

    /// Tell the window how to find out whether the run behind it is still
    /// going, so it knows how long it has to keep servicing its message queue
    /// after being closed.
    ///
    /// For the windows that have one run behind them for their whole life:
    /// `run --view`'s. An editing session is not watched this way and needs no
    /// closure, because the session itself can be replaced - see
    /// [`ViewerApp::should_keep_pumping`].
    pub fn watching(mut self, still_running: &'a dyn Fn() -> bool) -> ViewerApp<'a> {
        self.still_running = Some(still_running);
        self
    }

    /// Put the window into editor mode over `editor`.
    pub fn editing(mut self, editor: Editor) -> ViewerApp<'a> {
        self.editor = Some(editor);
        self
    }

    /// Collect the threads the editor left behind. Called once the event loop
    /// has stopped.
    ///
    /// Closing an editor window stops its runs, and the loop stays alive until
    /// they have ended - see [`ViewerApp::should_keep_pumping`] - so this joins
    /// threads that are already over rather than blocking with a message queue
    /// still to service.
    pub fn finish_editing(&mut self) {
        if let Some(editor) = self.editor.as_mut() {
            editor.finish();
        }
    }

    /// Points the side panel takes, which is the editor's panel when there is
    /// one. The 3D viewport is sized against it.
    fn panel_width_points(&self) -> f32 {
        if self.editor.is_some() {
            constants::VIEW_EDIT_PANEL_WIDTH_POINTS
        } else {
            constants::VIEW_PANEL_WIDTH_POINTS
        }
    }

    /// Open the window and run the event loop.
    ///
    /// Returns once the window is gone *and* the run behind it has finished:
    /// see [`ViewerApp::begin_teardown`] for why the loop may not stop at the
    /// moment the window closes.
    pub fn run(&mut self) -> Result<()> {
        let event_loop = EventLoop::new().context("creating the viewer event loop")?;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop
            .run_app(self)
            .context("running the viewer event loop")?;
        self.print_frame_summary();
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        Ok(())
    }

    fn print_frame_summary(&mut self) {
        if self.summary_printed {
            return;
        }
        self.summary_printed = true;
        if let Some(start) = self.first_frame {
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "viewer         {} frames in {:.2} s ({:.1} fps)",
                self.frames,
                elapsed,
                if elapsed > 0.0 {
                    self.frames as f64 / elapsed
                } else {
                    0.0
                }
            );
        }
    }

    /// End the session on an error a frame could not survive, having first put
    /// anything unsaved somewhere it can be got back from.
    ///
    /// Drawing stops, but the loop is not abandoned: the run behind the window
    /// still needs its message queue serviced, which is what
    /// [`ViewerApp::begin_teardown`] leaves it doing. The error is kept and
    /// returned by [`ViewerApp::run`], so a viewer that could not go on is still
    /// a failure at the command line - the recovery below is what is saved from
    /// it, not a reason to call it a success.
    fn fail(&mut self, error: anyhow::Error) {
        self.rescue_unsaved_edits();
        self.error = Some(error);
        self.begin_teardown();
        // After the teardown rather than before it: the teardown stops the
        // session's runs, and a rescue started ahead of it would be stopped with
        // them. The loop then goes on servicing its message queue until the
        // export's thread ends, exactly as it does for a run winding down - see
        // [`ViewerApp::should_keep_pumping`].
        self.rescue_retained_design();
    }

    /// Export the design a dying editing session still holds, and say where it
    /// is going and why.
    ///
    /// The other half of the rescue above: the document is what was typed and
    /// this is what was computed - a run's hours, which the window dying would
    /// otherwise take with it. It goes beside the configured files rather than
    /// onto them, exactly as the document does, so nothing a session that died
    /// writes can land on a good part. A `view` or `run --view` window has no
    /// session and nothing to do here.
    ///
    /// Reported on the console, because the panel that would have carried it
    /// belongs to the window that has just died; the export says where the file
    /// went when it is there.
    fn rescue_retained_design(&mut self) {
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        match editor.rescue_design() {
            Some(path) => println!(
                "editor rescue the viewer stopped with the run's design in it; it is being \
                 written to {}\n\
                 editor rescue the configured output is left as it is",
                path.display()
            ),
            None => println!(
                "editor rescue the viewer stopped with no design behind it; there is nothing to \
                 write"
            ),
        }
    }

    /// Write an editing session's unsaved document beside the file it came
    /// from, and say where it went. `None` when there was nothing to write, or
    /// when writing it failed.
    ///
    /// This is the last moment at which the document exists at all: the window
    /// is going, and everything typed since the last save would go with it. It
    /// is written *beside* the file rather than into it, because a viewer that
    /// died is not the user asking for a save - the file on disk stays the
    /// source of truth it is everywhere else in the editor, and what is recovered
    /// is looked at before it is kept. A `view` or `run --view` window has no
    /// document and nothing to do here.
    ///
    /// Reported on the console: the status line that would have carried it
    /// belongs to the window that has just died. A write that fails says so and
    /// changes nothing else - the fatal error is what the process ends on
    /// either way.
    fn rescue_unsaved_edits(&self) -> Option<PathBuf> {
        let editor = self.editor.as_ref()?;
        if !editor.state.is_dirty() {
            return None;
        }
        match editor.state.write_recovery() {
            Ok(path) => {
                println!(
                    "editor rescue the viewer stopped with unsaved changes; they are in {}\n\
                     editor rescue {} itself is unchanged",
                    path.display(),
                    editor.state.path().display()
                );
                Some(path)
            }
            Err(error) => {
                eprintln!(
                    "editor rescue the viewer stopped with unsaved changes and they could not be \
                     written: {error:#}"
                );
                None
            }
        }
    }

    /// Release the window and every GPU resource, and detach the run.
    ///
    /// This runs *inside* the event loop rather than on the way out of it.
    /// winit destroys a window by posting to its own thread's message queue,
    /// so a window dropped while the loop is already terminating is never
    /// actually destroyed: the HWND survives, visible and unserviced, and
    /// Windows eventually declares the process hung and terminates it, killing
    /// a detached run part way through. Tearing down here leaves the loop
    /// pumping long enough to finish the job.
    fn begin_teardown(&mut self) {
        if self.torn_down {
            return;
        }
        self.torn_down = true;
        self.print_frame_summary();
        self.gpu = None;
        self.egui_state = None;
        self.window = None;
        // The run is on its own from here and must not keep paying for
        // snapshots that nothing will draw.
        if let Some(link) = self.link {
            link.detach();
        }
        // An editor window *is* its session, so its own runs are not detached
        // but stopped, whatever kind they are. They stop cooperatively, at
        // their next checkpoint, which is why the loop goes on pumping until
        // the editor's probe says the threads have ended.
        if let Some(editor) = self.editor.as_mut() {
            editor.detach();
        }
    }

    /// True while the loop must stay alive purely to service the message
    /// queue, because the run behind the closed window has not finished.
    ///
    /// An editing session answers for itself, and it is asked *fresh every
    /// time*: the session the window holds now is the one whose threads it has
    /// to outlive, and a file switch can have replaced the one it opened with -
    /// see [`ViewerApp::switch_editor`]. A closure taken when the window was
    /// built would go on reporting the worker of a session that is over, which
    /// is how the loop would come to leave with a live thread behind it.
    /// [`ViewerApp::watching`] is for the windows that have one run behind them
    /// for their whole life: `run --view`'s.
    fn should_keep_pumping(&self) -> bool {
        if !self.torn_down {
            return false;
        }
        match self.editor.as_ref() {
            Some(editor) => editor.run_probe().is_running(),
            None => self.still_running.is_some_and(|running| running()),
        }
    }

    /// Physical pixels the 3D view owns.
    ///
    /// Until the first frame reports egui's points-per-pixel, the panel's share
    /// is derived from the window's scale factor: fitting against the whole
    /// surface instead would frame geometry the panel goes on to hide.
    fn scene_viewport_width(&self, surface_width: u32) -> u32 {
        match self.scene_width {
            Some(width) => width.min(surface_width),
            None => {
                let scale = self
                    .window
                    .as_ref()
                    .map_or(1.0, |window| window.scale_factor() as f32);
                scene_width_for(surface_width, scale, self.panel_width_points())
            }
        }
    }

    /// The ray through a pointer position, wherever that position is.
    ///
    /// Every quantity here is in physical pixels: winit reports the cursor in
    /// them, the surface is configured in them, and the panel's share is
    /// converted into them with the same points-per-pixel the frame was drawn
    /// with. The ray is then built against the very aspect the renderer set its
    /// viewport to, so a pixel means the same thing to the pointer and to the
    /// picture whatever the display's scale factor is.
    ///
    /// A position outside the 3D view is a point on that same image plane,
    /// further out: [`pick::ray_through`] is linear in the pointer position and
    /// keeps a forward facing unit direction however far past an edge it is
    /// asked about. Only a drag in progress uses it there - see
    /// [`ViewerApp::editor_ray`], which is what a press and a hover cast
    /// through.
    fn pointer_ray(&self, x: f64, y: f64) -> Option<Ray> {
        let (surface_width, height) = self.surface_size?;
        let width = self.scene_viewport_width(surface_width);
        Some(pick::ray_through(
            &self.camera,
            x,
            y,
            f64::from(width),
            f64::from(height),
        ))
    }

    /// The same ray, or `None` when the pointer is over the panel rather than
    /// the 3D view.
    ///
    /// What a press, a release and a hover cast through: outside the 3D view
    /// there is nothing to pick and nothing to grab, and the panel's own widgets
    /// are what a click there means.
    fn editor_ray(&self, x: f64, y: f64) -> Option<Ray> {
        let (surface_width, height) = self.surface_size?;
        let width = self.scene_viewport_width(surface_width);
        if x < 0.0 || y < 0.0 || x >= f64::from(width) || y >= f64::from(height) {
            return None;
        }
        self.pointer_ray(x, y)
    }

    /// Tell the editor what the camera is doing, once per frame.
    ///
    /// Two things depend on it and nothing else does: the drags that move in
    /// the camera's plane need the direction it is looking, and the callouts
    /// need the projection - and the viewport it projects into - to sit on the
    /// geometry they measure. Both are the frame's own, so they are handed over
    /// before the frame is built.
    fn hand_camera_to_editor(&mut self) {
        if self.editor.is_none() {
            return;
        }
        let (forward, _, _) = self.camera.view_basis();
        let viewport = self.surface_size.map(|(surface_width, height)| {
            let width = self.scene_viewport_width(surface_width);
            (
                self.camera.view_projection(aspect_of(width, height)),
                (f64::from(width), f64::from(height)),
            )
        });
        let pixels_per_point = self.egui_ctx.pixels_per_point();
        if let Some(editor) = self.editor.as_mut() {
            editor.set_view(forward);
            if let Some((matrix, size)) = viewport {
                editor.set_projection(matrix, size, pixels_per_point);
            }
        }
    }

    /// Upload the layers the editor rebuilt, and only those.
    fn upload_layers(&mut self, refresh: Refresh) {
        if let Some(gpu) = self.gpu.as_mut() {
            for info in LAYERS {
                let wanted = match info.role {
                    LayerRole::Setup => refresh.setup,
                    LayerRole::Editor => refresh.overlays,
                    // The density layer goes through its own path below, which
                    // is what applies the stress colouring and the shading
                    // switch to it.
                    LayerRole::Density => false,
                };
                if wanted {
                    gpu.upload(info.layer, self.scene.get(info.layer));
                }
            }
        }
        if refresh.density {
            self.upload_density();
        }
    }

    /// True while the viewport takes no editing input at all, which is what the
    /// scene's "show nothing but the part" switch does as well as hiding the
    /// layers.
    ///
    /// Asked at the two roots of the editor's pointer input - the window events
    /// and the per-frame hover - rather than inside the gestures themselves, so
    /// there is one place that decides it and no path can be added that forgets
    /// to ask. What is suspended is picking, grabbing, dragging, hovering and
    /// placement clicks, and nothing else: the camera keeps every event, because
    /// looking at the part from another side is what the mode is for.
    ///
    /// Whatever a gesture was holding when the switch went on is put down here,
    /// once, by the editor itself: see [`Editor::suspend_interaction`].
    fn editing_suspended(&mut self) -> bool {
        if !self.scene.isolate_part() {
            return false;
        }
        if let Some(editor) = self.editor.as_mut() {
            editor.suspend_interaction();
        }
        true
    }

    /// Editor input: gizmo drags and click-to-select. Returns true when a
    /// handle took the event, which is what keeps the camera out of it.
    fn handle_editor_event(&mut self, event: &WindowEvent) -> bool {
        if self.editor.is_none() || self.editing_suspended() {
            return false;
        }
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                // A drag owns the pointer, so it follows it wherever it goes -
                // across the side panel and past the window's own edges. That is
                // not an edge case but the ordinary way out of a framed domain:
                // the view is fitted to the model, so the pointer position that
                // asks for an object *outside* it is usually outside the 3D view
                // too. Dropping those positions stopped every such drag dead at
                // about the domain's wall, which read as containment and
                // survived switching containment off.
                let ray = self
                    .editor
                    .as_ref()
                    .is_some_and(Editor::is_dragging)
                    .then(|| self.pointer_ray(position.x, position.y))
                    .flatten();
                match (self.editor.as_mut(), ray) {
                    (Some(editor), Some(ray)) => {
                        editor.drag_to(&ray);
                        // The camera will not see this move, so this is what
                        // keeps its delta reference following the pointer through
                        // the drag - without it a camera gesture pressed part way
                        // through one, or straight after it, measures from where
                        // the drag began. Skipped while a camera gesture *is*
                        // going, because the camera path then runs after this one
                        // and needs the reference it has not consumed yet: writing
                        // it here would leave that gesture with no delta at all.
                        if !self.pointer.dragging() {
                            self.pointer.last = self.pointer.position;
                        }
                        true
                    }
                    _ => false,
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                self.pointer.travel = 0.0;
                let ray = self
                    .pointer
                    .position
                    .and_then(|(x, y)| self.editor_ray(x, y));
                let grabbed = match (self.editor.as_mut(), ray) {
                    (Some(editor), Some(ray)) => editor.press(&ray),
                    _ => false,
                };
                if grabbed {
                    // The camera is about to stop seeing pointer moves. Belt and
                    // braces on `Pointer::last`'s invariant at the two ends of a
                    // drag: the move that put the pointer on this handle was one
                    // the camera did see, and every move the drag goes on to claim
                    // writes the reference itself, so neither this nor the release
                    // below has anything left to correct.
                    self.pointer.last = self.pointer.position;
                }
                grabbed
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                let click = self.pointer.travel <= constants::VIEW_EDIT_CLICK_SLOP_PIXELS;
                let ray = self
                    .pointer
                    .position
                    .and_then(|(x, y)| self.editor_ray(x, y));
                let was_dragging = self.editor.as_ref().is_some_and(Editor::is_dragging);
                if let Some(editor) = self.editor.as_mut() {
                    editor.release(ray.as_ref(), click);
                }
                if was_dragging {
                    // The other end of the same belt and braces; see the press
                    // above and `Pointer::last`.
                    self.pointer.last = self.pointer.position;
                }
                false
            }
            _ => false,
        }
    }

    /// Keep the window's title on the editor's, so the unsaved marker shows.
    fn refresh_title(&mut self) {
        let Some(editor) = self.editor.as_ref() else {
            return;
        };
        let title = editor.window_title();
        if title == self.title {
            return;
        }
        self.title = title;
        if let Some(window) = self.window.as_ref() {
            window.set_title(&self.title);
        }
    }

    fn aspect(&self) -> f64 {
        match &self.gpu {
            Some(gpu) => {
                let (width, height) = gpu.size();
                aspect_of(self.scene_viewport_width(width), height)
            }
            None => 1.0,
        }
    }

    /// Frame the whole scene.
    ///
    /// `fitted` latches only once a frame has reported the real viewport, so an
    /// F pressed before the first frame still leaves the opening auto-fit to
    /// run against the authoritative width.
    fn fit(&mut self) {
        let bounds = self.scene.bounds();
        if !bounds.is_empty() {
            let aspect = self.aspect();
            self.camera.fit(&bounds, aspect);
            self.fitted = self.scene_width.is_some();
        }
    }

    /// Take whatever the preview mesher finished and hand it to the GPU.
    ///
    /// The camera is left alone: a preview lands inside the domain the opening
    /// fit already framed, and re-fitting every iteration would make the view
    /// crawl while the user is trying to look at it.
    fn drain_frames(&mut self) {
        let Some(link) = self.link else { return };
        let Some(frame) = link.frames.try_take() else {
            return;
        };
        self.frame_kind = Some(frame.kind);
        self.scene.set(Layer::Density, Some(frame.mesh));
        self.scene.set_stress(frame.stress);
        self.upload_density();
    }

    /// Push whichever density geometry the panel currently asks for.
    ///
    /// Flat shading is derived here rather than kept alongside the smooth
    /// geometry: a supersampled surface is large, the scene already holds a
    /// second copy of it for the stress colours, and the switch is usually
    /// never flipped. The cost is one pass over the vertices when it is.
    fn upload_density(&mut self) {
        self.stress_shading = self.scene.stress_shading();
        self.flat_shading = self.scene.flat_shading();
        let flat = self.flat_shading;
        let geometry = self.scene.density_geometry();
        if let Some(gpu) = self.gpu.as_mut() {
            let flattened = flat.then(|| geometry.map(LayerMesh::flat_shaded)).flatten();
            gpu.upload(Layer::Density, flattened.as_ref().or(geometry));
        }
    }

    fn handle_camera_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                if let Some(previous) = self.pointer.last {
                    let (dx, dy) = (current.0 - previous.0, current.1 - previous.1);
                    if self.pointer.orbiting {
                        self.camera.orbit(dx, dy);
                    } else if self.pointer.panning {
                        let height = self.gpu.as_ref().map_or(1.0, |g| g.size().1 as f64);
                        self.camera.pan(dx, dy, height);
                    }
                }
                self.pointer.last = Some(current);
            }
            WindowEvent::CursorLeft { .. } => self.pointer.last = None,
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.pointer.orbiting = pressed,
                    MouseButton::Right | MouseButton::Middle => self.pointer.panning = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y as f64,
                    MouseScrollDelta::PixelDelta(position) => {
                        position.y / constants::VIEW_SCROLL_PIXELS_PER_LINE
                    }
                };
                self.camera.zoom(lines);
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::KeyF) =>
            {
                self.fit();
            }
            _ => {}
        }
    }

    /// Window events never stop the loop themselves: every path out of the
    /// viewer goes through [`ViewerApp::begin_teardown`] and then the drain
    /// check in `about_to_wait`, which is the only place the loop exits.
    ///
    /// The event loop itself is not needed here, which is what lets the whole
    /// input path - pointer positions, rays, gizmo grabs, drags, picking - be
    /// driven end to end by a test with no window: see the harness below.
    pub(crate) fn on_window_event(&mut self, event: WindowEvent) {
        if self.torn_down {
            return;
        }
        // Before anything can decide not to look at this event. Where the
        // pointer is is not one path's business: the camera, the gizmos and the
        // picking all cast from it, and a press carries no position of its own.
        match &event {
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((px, py)) = self.pointer.position {
                    self.pointer.travel +=
                        ((position.x - px).powi(2) + (position.y - py).powi(2)).sqrt();
                }
                self.pointer.position = Some((position.x, position.y));
            }
            WindowEvent::CursorLeft { .. } => self.pointer.position = None,
            // Held modifiers are nobody's business in particular either: the
            // editor's snap bypass reads Alt from here, and a drag in flight
            // has to notice it being pressed and released mid-gesture.
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
                if let Some(editor) = self.editor.as_mut() {
                    editor.set_snap_bypass(self.modifiers.alt_key());
                }
            }
            // The paste press, which is nobody's gesture either - and the one
            // binding that cannot be left to egui: the backend answers it out
            // of the *platform's* clipboard and passes nothing on at all when
            // there is no text there to paste. The editor's clipboard holds
            // objects rather than text, so the press is noted here and answered
            // by [`Editor::shortcuts`], under the guards every binding is under
            // and folded into the one decision - see [`Editor::note_paste_key`].
            // Noted before `consumed` and before the editing suspension, for the
            // reason the modifiers are: neither is this binding's business, and
            // both are answered where every other binding answers them.
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && !event.repeat
                    && command_held(&self.modifiers)
                    && is_paste_chord(&event.logical_key, event.physical_key) =>
            {
                if let Some(editor) = self.editor.as_mut() {
                    editor.note_paste_key();
                }
            }
            _ => {}
        }
        let consumed = match (self.egui_state.as_mut(), self.window.as_ref()) {
            (Some(state), Some(window)) => state.on_window_event(window, &event).consumed,
            _ => false,
        };
        // A press that grabbed a gizmo handle is not the start of an orbit.
        let grabbed = if consumed {
            false
        } else {
            self.handle_editor_event(&event)
        };
        // A drag that started over the scene keeps control even when the
        // pointer wanders across the panel.
        if (!consumed && !grabbed) || self.pointer.dragging() {
            self.handle_camera_event(&event);
        }

        match event {
            // The editor may have unsaved changes to ask about first. Whatever
            // it answers, the way out of the window is still the one path.
            WindowEvent::CloseRequested => {
                let ready = match self.editor.as_mut() {
                    Some(editor) => editor.request_close(),
                    None => true,
                };
                if ready {
                    self.begin_teardown();
                }
            }
            WindowEvent::Resized(size) => {
                // A device that will not give the new size a surface or a depth
                // texture has rejected a frame, which is the renderer's own
                // bounded retry; only the end of that retry gets here.
                let refused = self
                    .gpu
                    .as_mut()
                    .and_then(|gpu| gpu.resize(size.width, size.height).err());
                if let Some(error) = refused {
                    self.fail(error);
                }
                self.surface_size = Some((size.width, size.height));
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.redraw() {
                    self.fail(error);
                }
            }
            _ => {}
        }
    }

    /// Tell the editor what a click would hit right now.
    ///
    /// Cast from the pointer position every frame rather than only when the
    /// pointer moves, because the camera moves the geometry under a pointer that
    /// is standing still; the editor compares the ray against the one it last
    /// picked with, so a frame in which neither moved costs nothing.
    ///
    /// `None` - no hover at all - covers every reason there is nothing under the
    /// pointer to click: it is off the window, over the side panel (which is
    /// what [`ViewerApp::editor_ray`] answers), or in the middle of a camera
    /// drag, where the geometry is moving rather than being pointed at. The one
    /// reason that is not a ray at all - the viewport taking no editing input -
    /// is answered above, where the outline is dropped rather than re-picked.
    fn hand_hover_to_editor(&mut self) {
        if self.editor.is_none() || self.editing_suspended() {
            return;
        }
        let ray = if self.pointer.dragging() {
            None
        } else {
            self.pointer
                .position
                .and_then(|(x, y)| self.editor_ray(x, y))
        };
        if let Some(editor) = self.editor.as_mut() {
            editor.hover_to(ray);
        }
    }

    /// Raise the native file picker the editor asked for, if it asked for one.
    ///
    /// **This blocks**, and deliberately: an OS file dialog owns the input queue
    /// while it is up, and the window behind it stops drawing until it is
    /// answered - which is what a modal dialog is, here and in every other
    /// program on the machine. Nothing behind the window is affected. The run is
    /// on its own thread and keeps going; the channels that feed the window hold
    /// one item and drop what nobody collects, so what a dialog left open costs
    /// is dropped preview frames.
    ///
    /// Raised from the frame loop rather than from under the button, so that the
    /// toolbar and the keyboard shortcut reach it through the one recorded
    /// [`Pick`] - and so that neither opens a nested message loop inside the
    /// egui frame it was clicked in.
    fn raise_file_dialog(&mut self) {
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        let Some(pick) = editor.take_pick() else {
            return;
        };
        // Where the file being edited lives, which is where looking for the next
        // one starts.
        let directory = editor.state.directory().to_path_buf();
        let chosen = match pick {
            Pick::Open => dialog::choose_open(&directory),
            Pick::New => dialog::choose_new(&directory),
        };
        // A cancelled dialog is an answer: nothing was asked for after all.
        let Some(path) = chosen else {
            return;
        };
        // What comes back goes through the unsaved-changes guard, exactly as it
        // does from the buttons. Whether it cleared is not this function's
        // business: what it clears is collected below, in this same pump.
        match pick {
            Pick::Open => editor.request_open(path),
            Pick::New => editor.request_new(path),
        };
    }

    /// Replace the editing session with one over the file the guard cleared.
    ///
    /// The session that is leaving is stopped and its threads collected *before*
    /// the one that arrives is built, so there is never a second worker behind
    /// one window - two of them would be two engines fighting over one output
    /// path. Both steps can take a moment, because a stop is cooperative and is
    /// answered at the run's next checkpoint; the window is frozen for it, as it
    /// was for the dialog that got here.
    ///
    /// What is then opened is a whole new session through the same constructors
    /// `growforge edit` uses, so everything a stale session could carry -
    /// retained design, output mtimes, run probe, undo history - is gone by
    /// construction rather than by being reset.
    ///
    /// A file that will not open leaves the session exactly where it was, with
    /// its runs stopped and the reason on its status line: the document is never
    /// lost to a switch that could not happen.
    fn switch_editor(&mut self) {
        let Some(switch) = self.editor.as_mut().and_then(Editor::take_switch) else {
            return;
        };
        if let Some(editor) = self.editor.as_mut() {
            editor.detach();
            editor.finish();
        }
        match switch.open() {
            Ok(editor) => self.adopt_editor(editor),
            Err(error) => {
                if let Some(editor) = self.editor.as_mut() {
                    editor.set_status(format!(
                        "could not open {}: {error:#}",
                        switch.path().display()
                    ));
                }
            }
        }
    }

    /// Put the window on `editor`: its scene on the GPU, the camera framed on
    /// it, and the title its own.
    ///
    /// The opening path, taken again. A new document is a new model, and a
    /// camera still orbiting where the last one stood would be pointing at
    /// nothing; the fit is the same one the window performs when it opens, and
    /// waits for the first frame in the same way when there is no viewport yet.
    fn adopt_editor(&mut self, editor: Editor) {
        self.scene = editor.initial_scene();
        self.editor = Some(editor);
        // Nothing has been drawn for this document yet, so the panel's frame
        // label describes nothing.
        self.frame_kind = None;
        self.camera = OrbitCamera::default();
        self.fitted = false;
        self.fit();
        self.upload_all_layers();
        self.refresh_title();
    }

    /// Push every layer of the scene as it stands: what a window that has just
    /// been handed a scene owes the GPU, at the opening and at a file switch
    /// alike.
    fn upload_all_layers(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            for info in LAYERS {
                gpu.upload(info.layer, self.scene.get(info.layer));
            }
        }
        // Which also leaves the two shading mirrors on what was just pushed.
        self.upload_density();
    }

    /// The editor's deferred work: the picker it asked for, the file switch that
    /// cleared, the hover under the pointer, the setup rebuild an edit asked
    /// for, the surface its background run produced, and the selection overlays.
    /// Returns true when the editor has been told it may close.
    fn pump_editor(&mut self) -> bool {
        // Both before the frame's work, so a session that has just arrived is
        // the one the rest of this pump is about.
        self.raise_file_dialog();
        self.switch_editor();
        self.hand_hover_to_editor();
        let (refresh, closing) = match self.editor.as_mut() {
            Some(editor) => {
                let scene = &mut self.scene;
                (editor.pump(scene), editor.may_close())
            }
            None => (Refresh::default(), false),
        };
        if !refresh.is_empty() {
            self.upload_layers(refresh);
        }
        closing
    }

    fn redraw(&mut self) -> Result<()> {
        let panel_points = self.panel_width_points();
        // The camera of the frame about to be drawn, which is what the plane
        // drags and the callouts are measured against.
        self.hand_camera_to_editor();
        let (raw_input, adapter) = {
            let (Some(window), Some(gpu), Some(egui_state)) = (
                self.window.as_ref(),
                self.gpu.as_ref(),
                self.egui_state.as_mut(),
            ) else {
                return Ok(());
            };
            (
                egui_state.take_egui_input(window),
                gpu.adapter_description.clone(),
            )
        };
        let progress = self.link.map(ViewLink::progress);
        let context = self.egui_ctx.clone();
        let title = self.title.clone();
        let engine = self.engine.clone();
        let frame_kind = self.frame_kind;
        let output = {
            let scene = &mut self.scene;
            let mut editor = self.editor.as_mut();
            context.run_ui(raw_input, |root| match &mut editor {
                Some(editor) => {
                    editor.draw(root, scene, &adapter);
                }
                None => ui::panel(
                    root,
                    &title,
                    &adapter,
                    scene,
                    progress.as_ref(),
                    engine.as_deref(),
                    frame_kind,
                ),
            })
        };
        if let Some(editor) = self.editor.as_mut() {
            editor.shortcuts(&context);
            editor.settle(&context);
        }
        // The unsaved marker lives in the title bar, so it is refreshed with
        // the frame that could have changed it.
        self.refresh_title();

        let (Some(window), Some(gpu), Some(egui_state)) = (
            self.window.as_ref(),
            self.gpu.as_mut(),
            self.egui_state.as_mut(),
        ) else {
            return Ok(());
        };
        egui_state.handle_platform_output(window, output.platform_output);
        let primitives = context.tessellate(output.shapes, output.pixels_per_point);

        let (width, height) = gpu.size();
        let scene_width = scene_width_for(width, output.pixels_per_point, panel_points);
        self.scene_width = Some(scene_width);
        // The opening fit waits for the first frame, because only then is the
        // panel's width in physical pixels, and so the 3D viewport, known.
        if !self.fitted {
            let bounds = self.scene.bounds();
            if !bounds.is_empty() {
                self.camera.fit(&bounds, aspect_of(scene_width, height));
                self.fitted = true;
            }
        }
        let view_projection = self.camera.view_projection(aspect_of(scene_width, height));
        gpu.render(
            &view_projection,
            &self.scene,
            scene_width,
            EguiFrame {
                primitives,
                textures_delta: output.textures_delta,
                pixels_per_point: output.pixels_per_point,
            },
        )?;

        self.frames += 1;
        if self.first_frame.is_none() {
            self.first_frame = Some(Instant::now());
        }
        Ok(())
    }
}

impl ApplicationHandler for ViewerApp<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                constants::VIEW_WINDOW_WIDTH,
                constants::VIEW_WINDOW_HEIGHT,
            ));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.error = Some(anyhow::Error::new(error).context("opening the viewer window"));
                self.begin_teardown();
                return;
            }
        };
        let gpu = match Gpu::new(Arc::clone(&window)) {
            Ok(gpu) => gpu,
            Err(error) => {
                self.error = Some(error);
                // The window did come up, so it has to go back down through the
                // teardown path like any other, not by abandoning the loop.
                self.window = Some(window);
                self.begin_teardown();
                return;
            }
        };
        println!("viewer adapter {}", gpu.adapter_description);

        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        ));
        self.surface_size = Some(gpu.size());
        self.gpu = Some(gpu);
        self.window = Some(window);
        self.upload_all_layers();
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        self.on_window_event(event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.torn_down {
            // The window is gone. Keep the loop, and therefore the message
            // queue, alive until the run finishes; leaving early is what got
            // the process killed as a hung application.
            if self.should_keep_pumping() {
                event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now()
                        + Duration::from_secs_f64(constants::VIEW_DETACHED_POLL_INTERVAL_S),
                ));
            } else {
                event_loop.exit();
            }
            return;
        }

        self.drain_frames();
        if self.pump_editor() {
            self.begin_teardown();
            return;
        }
        // The side panel toggles the stress colouring and the shading mode
        // while a frame is being drawn, so the swap of the vertex buffer lands
        // on the next one.
        if self.scene.stress_shading() != self.stress_shading
            || self.scene.flat_shading() != self.flat_shading
        {
            self.upload_density();
        }
        if let (Some(after), Some(start)) = (self.autoclose, self.first_frame)
            && start.elapsed() >= after
        {
            self.begin_teardown();
            return;
        }
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Safety net for the paths that leave the loop without a teardown, such
        // as a window or adapter that failed to come up at all.
        self.gpu = None;
        self.egui_state = None;
        self.window = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ShapeSpec;
    use crate::geometry::Vec3;
    use crate::viewer::camera::transform_point;
    use crate::viewer::editor::gizmo::HandleKind;
    use crate::viewer::editor::state::{Selection, shape_of};
    use crate::viewer::editor::tests::{TempDir, fixture, write_temp};
    use crate::viewer::scene::LayerMesh;
    use crate::viewer::tessellate;
    use std::sync::atomic::{AtomicBool, Ordering};
    use winit::dpi::PhysicalPosition;
    use winit::event::DeviceId;

    /// An editor window with no window: the whole input path, driven by
    /// synthesized winit events against a known camera and surface.
    ///
    /// The two numbers a pointer position has to be interpreted against - the
    /// surface's physical size and the share of it the panel takes - are set
    /// here exactly as a drawn frame sets them, which is what lets the same
    /// gestures be replayed at any display scale factor.
    struct Harness<'a> {
        app: ViewerApp<'a>,
        _directory: TempDir,
    }

    impl Harness<'_> {
        /// A window `1280 x 800` logical pixels at `scale`, opened on the
        /// editor fixture and framed on the whole model.
        fn new(scale: f32) -> Harness<'static> {
            Harness::on(fixture(), scale)
        }

        /// The same, on a configuration of the caller's own.
        fn on(text: &str, scale: f32) -> Harness<'static> {
            let (directory, path) = write_temp("harness", text);
            let editor = crate::viewer::editor::Editor::open(&path).expect("open");
            let scene = editor.initial_scene();
            let mut app = ViewerApp::new("harness".to_string(), scene, None).editing(editor);
            let width = (f32::from(constants::VIEW_WINDOW_WIDTH as u16) * scale) as u32;
            let height = (f32::from(constants::VIEW_WINDOW_HEIGHT as u16) * scale) as u32;
            app.surface_size = Some((width, height));
            app.scene_width = Some(scene_width_for(
                width,
                scale,
                constants::VIEW_EDIT_PANEL_WIDTH_POINTS,
            ));
            let bounds = app.scene.bounds();
            app.camera.fit(&bounds, app.aspect());
            app.fitted = true;
            let mut harness = Harness {
                app,
                _directory: directory,
            };
            harness.pump();
            harness
        }

        fn editor(&mut self) -> &mut crate::viewer::editor::Editor {
            self.app.editor.as_mut().expect("an editor")
        }

        /// One turn of the frame loop's editor work.
        fn pump(&mut self) {
            // The camera is what the plane drags and the callouts are measured
            // against; a drawn frame hands it over before the panel is built,
            // and so does this.
            self.app.hand_camera_to_editor();
            self.app.pump_editor();
        }

        /// Press or release the snap bypass key, as a real modifier event.
        fn alt(&mut self, held: bool) {
            let state = if held {
                ModifiersState::ALT
            } else {
                ModifiersState::empty()
            };
            self.app
                .on_window_event(WindowEvent::ModifiersChanged(state.into()));
        }

        /// Physical pixel a world point is drawn at, by the same projection the
        /// renderer uses and the same viewport it sets.
        fn pixel_of(&self, point: Vec3) -> (f64, f64) {
            let (surface_width, height) = self.app.surface_size.expect("a surface");
            let width = self.app.scene_viewport_width(surface_width);
            let matrix = self.app.camera.view_projection(aspect_of(width, height));
            let clip = transform_point(&matrix, point);
            let (x, y) = (clip[0] / clip[3], clip[1] / clip[3]);
            (
                (x + 1.0) * 0.5 * f64::from(width) - 0.5,
                (1.0 - y) * 0.5 * f64::from(height) - 0.5,
            )
        }

        fn move_to(&mut self, point: Vec3) {
            let (x, y) = self.pixel_of(point);
            self.move_to_pixel(x, y);
        }

        fn move_to_pixel(&mut self, x: f64, y: f64) {
            self.app.on_window_event(WindowEvent::CursorMoved {
                device_id: DeviceId::dummy(),
                position: PhysicalPosition::new(x, y),
            });
        }

        fn button(&mut self, button: MouseButton, state: ElementState) {
            self.app.on_window_event(WindowEvent::MouseInput {
                device_id: DeviceId::dummy(),
                state,
                button,
            });
        }

        fn press(&mut self) {
            self.button(MouseButton::Left, ElementState::Pressed);
        }

        fn release(&mut self) {
            self.button(MouseButton::Left, ElementState::Released);
        }

        /// The pan gesture's button, which is the camera's other drag.
        fn right_press(&mut self) {
            self.button(MouseButton::Right, ElementState::Pressed);
        }

        fn right_release(&mut self) {
            self.button(MouseButton::Right, ElementState::Released);
        }

        /// Move onto a point and click it, the way a user does.
        fn click(&mut self, point: Vec3) {
            self.move_to(point);
            self.press();
            self.release();
            self.pump();
        }

        fn selection(&mut self) -> Option<Selection> {
            self.editor().state.selection()
        }

        fn shape(&mut self, selection: Selection) -> ShapeSpec {
            shape_of(self.editor().state.config(), selection).expect("a shape")
        }

        /// The handle of a given kind on the current selection.
        fn handle(&mut self, kind: HandleKind) -> crate::viewer::editor::gizmo::Handle {
            self.editor()
                .handles
                .iter()
                .find(|handle| handle.kind == kind)
                .copied()
                .unwrap_or_else(|| panic!("no {kind:?} handle"))
        }

        /// Grab the handle of a given kind, drag it to `target`, and let go -
        /// the whole gesture through the window's own event path.
        fn drag_handle(&mut self, kind: HandleKind, target: Vec3) {
            let from = self.handle(kind).position;
            self.drag_handle_from(kind, from, target);
        }

        /// The same, pressing at a point of the caller's choosing: the whole
        /// length of an arrow is grabbable, and where along it the press lands
        /// decides which other handles are in the way.
        ///
        /// Asserts that the press really engaged *that* handle, so a grab that
        /// went to an overlapping one reads as what it is rather than as a drag
        /// that did something unexpected.
        fn drag_handle_from(&mut self, kind: HandleKind, press_at: Vec3, target: Vec3) {
            self.move_to(press_at);
            self.press();
            assert!(self.editor().is_dragging(), "no drag started on {kind:?}");
            assert_eq!(
                self.editor().callout().expect("a callout").handle.kind,
                kind,
                "the press landed on another handle"
            );
            self.move_to(target);
            self.pump();
            self.release();
            self.pump();
        }

        /// The world direction the pointer travels in when it moves right across
        /// the screen, which is the plane the free and endpoint drags move in.
        fn screen_right(&self) -> Vec3 {
            self.app.camera.view_basis().1
        }

        /// True when a world point is drawn outside the 3D view.
        ///
        /// Which is where the pointer has to go to push an object out of a
        /// domain the view is fitted to, and so what a drag has to keep
        /// following.
        fn is_off_the_view(&self, point: Vec3) -> bool {
            let (surface_width, height) = self.app.surface_size.expect("a surface");
            let width = f64::from(self.app.scene_viewport_width(surface_width));
            let (x, y) = self.pixel_of(point);
            x < 0.0 || y < 0.0 || x >= width || y >= f64::from(height)
        }
    }

    /// The box the containment rule keeps objects inside.
    fn domain_bounds(harness: &mut Harness<'_>) -> crate::geometry::Aabb {
        crate::viewer::editor::state::containment_bounds(harness.editor().state.config())
            .expect("the fixture's domain builds")
    }

    /// Bounds of a shape, which is what containment clamps.
    fn bounds_of(spec: &ShapeSpec) -> crate::geometry::Aabb {
        spec.to_shape("test").expect("a well formed shape").bounds()
    }

    /// Centre of a shape, for the assertions below.
    fn centre_of(spec: &ShapeSpec) -> Vec3 {
        crate::viewer::editor::gizmo::anchor(spec)
    }

    /// The keepout sphere of the fixture, which is what these tests grab.
    const TARGET: Selection = Selection::Keepout(1);

    fn scene_with_a_box() -> Scene {
        let mut scene = Scene::default();
        let mesh = tessellate::box_mesh([0.0, 0.0, 0.0], [10.0, 20.0, 30.0]);
        scene.set(
            Layer::Domain,
            Some(LayerMesh::from_mesh(
                &mesh,
                Layer::Domain.info().color,
                crate::viewer::scene::Shading::Flat,
            )),
        );
        scene
    }

    /// Every scale factor a Windows desktop is likely to be set to. The
    /// pointer arrives in physical pixels and the surface is configured in
    /// them, so the conversions have to come out the same at all of them - this
    /// is what says so, rather than an argument that they do.
    const SCALES: [f32; 3] = [1.0, 1.25, 1.5];

    /// Which press the window takes for a paste, which is the one binding it
    /// has to recognise itself: see [`ViewerApp::on_window_event`]. The event
    /// carrying it cannot be built outside winit - one of its fields is that
    /// crate's own - so what is pinned here is the decision the branch is
    /// guarded by, on the two things it reads.
    #[test]
    fn the_window_takes_the_paste_press_by_letter_and_falls_back_to_position() {
        use winit::keyboard::{Key, NamedKey};

        let v = PhysicalKey::Code(KeyCode::KeyV);
        let letter = |text: &str| Key::Character(text.into());

        // The letter the layout produced decides, upper case or lower.
        assert!(is_paste_chord(&letter("v"), v));
        assert!(is_paste_chord(&letter("V"), v));
        // A layout that puts another letter on that position is answered by its
        // letter, not by the position: on Dvorak the paste is elsewhere.
        assert!(!is_paste_chord(&letter("."), v));
        assert!(is_paste_chord(
            &letter("v"),
            PhysicalKey::Code(KeyCode::Period)
        ));
        // A layout with no Latin letters at all has no letter to match, so the
        // position is what is left - which is where the binding is printed.
        assert!(is_paste_chord(&letter("м"), v));
        assert!(!is_paste_chord(
            &letter("м"),
            PhysicalKey::Code(KeyCode::KeyC)
        ));
        // And a named key is never it.
        assert!(!is_paste_chord(
            &Key::Named(NamedKey::Enter),
            PhysicalKey::Code(KeyCode::Enter)
        ));

        // The modifier it is held under is the one egui calls `command`.
        assert!(command_held(&ModifiersState::CONTROL) != cfg!(target_os = "macos"));
        assert!(command_held(&ModifiersState::SUPER) == cfg!(target_os = "macos"));
        assert!(!command_held(&ModifiersState::empty()));
        assert!(!command_held(&ModifiersState::SHIFT));
    }

    #[test]
    fn clicking_an_arrow_grabs_that_handle_rather_than_changing_the_selection() {
        for scale in SCALES {
            let mut harness = Harness::new(scale);
            harness.editor().state.select(Some(TARGET));
            harness.pump();

            let handle = harness.handle(HandleKind::Translate(0));
            let centre = centre_of(&harness.shape(TARGET));
            // Halfway along the shaft: what the user aims at is the arrow, not
            // the point at the end of it.
            let middle = [
                0.5 * (centre[0] + handle.position[0]),
                0.5 * (centre[1] + handle.position[1]),
                0.5 * (centre[2] + handle.position[2]),
            ];
            harness.move_to(middle);
            harness.press();
            assert!(
                harness.editor().is_dragging(),
                "scale {scale}: the arrow did not take the press"
            );
            assert_eq!(
                harness.selection(),
                Some(TARGET),
                "scale {scale}: grabbing a handle must not change the selection"
            );
            harness.release();
            assert!(!harness.editor().is_dragging());
            assert_eq!(harness.selection(), Some(TARGET));
        }
    }

    #[test]
    fn dragging_an_arrow_moves_the_object_by_what_the_pointer_covered() {
        for scale in SCALES {
            let mut harness = Harness::new(scale);
            harness.editor().state.select(Some(TARGET));
            harness.pump();
            let before = centre_of(&harness.shape(TARGET));
            let steps = harness.editor().state.undo_depth();

            let handle = harness.handle(HandleKind::Translate(0));
            harness.move_to(handle.position);
            harness.press();
            assert!(harness.editor().is_dragging(), "scale {scale}");
            // The pointer travels to where the tip would be if the object had
            // already moved 8 mm along x, which is therefore how far it moves.
            let travel = 8.0;
            let target = [
                handle.position[0] + travel,
                handle.position[1],
                handle.position[2],
            ];
            harness.move_to(target);
            harness.pump();
            harness.release();
            harness.pump();

            let after = centre_of(&harness.shape(TARGET));
            let moved = [
                after[0] - before[0],
                after[1] - before[1],
                after[2] - before[2],
            ];
            assert!(
                (moved[0] - travel).abs() < 0.3,
                "scale {scale}: moved {moved:?}, expected about {travel} on x"
            );
            assert!(
                moved[1].abs() < 0.05 && moved[2].abs() < 0.05,
                "scale {scale}: an axis drag moved off its axis: {moved:?}"
            );
            assert_eq!(
                harness.editor().state.undo_depth(),
                steps + 1,
                "scale {scale}: a drag is one undo step"
            );
        }
    }

    /// The whole input path on an ellipsoid: a click on it selects it, and a
    /// drag of one of its three radius handles changes that semi-axis by what
    /// the pointer covered along it.
    #[test]
    fn clicking_and_dragging_an_ellipsoid_selects_and_resizes_the_axis_grabbed() {
        use crate::viewer::editor::tests::ellipsoid_fixture;

        for scale in SCALES {
            let mut harness = Harness::on(ellipsoid_fixture(), scale);
            // Looked at from above, where nothing else of the fixture stands
            // between the pointer and the ellipsoid: the keepin pad and the
            // load region are beside it, not over it.
            harness.app.camera.pitch = constants::VIEW_PITCH_LIMIT_DEGREES.to_radians();
            harness.pump();
            // The fixture's second keepout is the ellipsoid; a click on its
            // centre picks it out of the scene.
            harness.click([50.0, 16.0, 16.0]);
            assert_eq!(harness.selection(), Some(TARGET), "scale {scale}");
            let ShapeSpec::Ellipsoid { radii, .. } = harness.shape(TARGET) else {
                panic!("scale {scale}: the fixture's keepout is an ellipsoid");
            };
            assert_eq!(radii, [8.0, 4.0, 4.0]);

            // Grab the handle of the long semi-axis and pull it 3 mm along its
            // own direction, which is the ellipsoid's turned local x. The long
            // one because a radius handle standing shorter than the gizmo's
            // arrows sits inside the shaft of the arrow along the same axis,
            // and the press then goes to the arrow - which is what a sphere's
            // single radius handle has always done.
            let handle = harness.handle(HandleKind::Radius(0));
            harness.editor().snap_mut().millimetres = 0.0;
            harness.move_to(handle.position);
            harness.press();
            assert!(harness.editor().is_dragging(), "scale {scale}");
            let travel = 3.0;
            harness.move_to(crate::geometry::sum(
                handle.position,
                crate::geometry::scale(handle.axis, travel),
            ));
            harness.pump();
            harness.release();
            harness.pump();

            let ShapeSpec::Ellipsoid { center, radii, .. } = harness.shape(TARGET) else {
                panic!("scale {scale}: the shape changed kind");
            };
            assert!(
                (radii[0] - 11.0).abs() < 0.3,
                "scale {scale}: radii came out {radii:?}"
            );
            assert_eq!([radii[1], radii[2]], [4.0, 4.0], "scale {scale}: one axis");
            assert_eq!(center, [50.0, 16.0, 16.0], "scale {scale}: it moved");
        }
    }

    /// A dragged object lands on the snap increment, whatever the display's
    /// scale factor: the pointer arrives in physical pixels, the increment is
    /// in millimetres, and nothing in between may leak the one into the other.
    #[test]
    fn a_snapped_drag_lands_on_the_increment_at_every_scale_factor() {
        for scale in SCALES {
            let mut harness = Harness::new(scale);
            harness.editor().state.select(Some(TARGET));
            harness.pump();
            // The fixture's sphere sits at x = 50, so an increment of 4 mm puts
            // the whole grid off its starting point: landing on a multiple of
            // it is something only snapping can do.
            harness.editor().snap_mut().millimetres = 4.0;
            let before = centre_of(&harness.shape(TARGET));
            assert!((before[0] - 50.0).abs() < 1e-9);

            let handle = harness.handle(HandleKind::Translate(0));
            harness.move_to(handle.position);
            harness.press();
            // 6.3 mm of pointer travel: 56.3 free, 56 snapped.
            harness.move_to([
                handle.position[0] + 6.3,
                handle.position[1],
                handle.position[2],
            ]);
            harness.pump();
            harness.release();
            harness.pump();

            let after = centre_of(&harness.shape(TARGET));
            assert!(
                (after[0] - 56.0).abs() < 1e-6,
                "scale {scale}: landed on {}, not on the 4 mm increment",
                after[0]
            );
            assert!(
                (after[1] - before[1]).abs() < 1e-9 && (after[2] - before[2]).abs() < 1e-9,
                "scale {scale}: snapping moved an axis nobody dragged: {after:?}"
            );
            // The number box says how far it went, and it is the snapped
            // distance rather than the pointer's own.
            let callout = harness.editor().callout().expect("a callout");
            assert!(
                (callout.at.value - 6.0).abs() < 1e-6,
                "{}",
                callout.at.value
            );
        }
    }

    /// The same drag with the bypass key held goes exactly where the pointer
    /// went.
    #[test]
    fn an_alt_drag_ignores_the_increment_at_every_scale_factor() {
        for scale in SCALES {
            let mut harness = Harness::new(scale);
            harness.editor().state.select(Some(TARGET));
            harness.pump();
            harness.editor().snap_mut().millimetres = 4.0;
            let handle = harness.handle(HandleKind::Translate(0));
            harness.move_to(handle.position);
            harness.alt(true);
            harness.press();
            harness.move_to([
                handle.position[0] + 6.3,
                handle.position[1],
                handle.position[2],
            ]);
            harness.pump();
            harness.release();
            harness.pump();
            let after = centre_of(&harness.shape(TARGET));
            assert!(
                (after[0] - 56.3).abs() < 0.05,
                "scale {scale}: a free drag landed on {}, not where the pointer went",
                after[0]
            );
            // Letting go of the key snaps the next drag again.
            harness.alt(false);
            let handle = harness.handle(HandleKind::Translate(0));
            harness.move_to(handle.position);
            harness.press();
            harness.move_to([
                handle.position[0] + 1.0,
                handle.position[1],
                handle.position[2],
            ]);
            harness.pump();
            harness.release();
            harness.pump();
            // 56.3 plus a millimetre is 57.3, which lands back on 56.
            let after = centre_of(&harness.shape(TARGET));
            assert!(
                (after[0] - 56.0).abs() < 1e-6,
                "scale {scale}: the increment did not come back: {}",
                after[0]
            );
        }
    }

    /// A load region dragged through the window towards the pad it loads lands
    /// on the pad's face rather than on the millimetre grid.
    #[test]
    fn a_dragged_load_region_lands_flush_on_a_face_at_every_scale_factor() {
        let load = Selection::Load { case: 0, load: 0 };
        for scale in SCALES {
            let mut harness = Harness::new(scale);
            // The fixture's keepin pad spans x = 56 .. 64; a sphere of radius 6
            // at x = 49 has its own face 1 mm short of it.
            harness.editor().state.edit(|config| {
                crate::viewer::editor::state::set_shape(
                    config,
                    load,
                    ShapeSpec::Sphere {
                        center: [49.0, 16.0, 16.0],
                        radius: 6.0,
                    },
                );
            });
            harness.editor().state.select(Some(load));
            harness.pump();

            let handle = harness.handle(HandleKind::Translate(0));
            harness.move_to(handle.position);
            harness.press();
            // Half a millimetre of pointer travel, which is inside the surface
            // snap distance and short of the increment.
            harness.move_to([
                handle.position[0] + 0.6,
                handle.position[1],
                handle.position[2],
            ]);
            harness.pump();
            harness.release();
            harness.pump();

            let after = centre_of(&harness.shape(load));
            assert!(
                (after[0] - 50.0).abs() < 1e-3,
                "scale {scale}: landed at {after:?} rather than flush against the pad"
            );
        }
    }

    /// A rotation arc dragged through the window: the box turns by a snapped
    /// step about the axis that arc drives.
    #[test]
    fn a_rotation_arc_drag_turns_the_object_by_a_snapped_step() {
        for scale in SCALES {
            let mut harness = Harness::new(scale);
            // The fixture's keepin box, which is a shape that records a
            // rotation of its own.
            let target = Selection::Keepin(0);
            harness.editor().state.select(Some(target));
            harness.pump();
            assert_eq!(harness.shape(target).rotation_deg(), None);

            let handle = harness.handle(HandleKind::Rotate(2));
            let centre = centre_of(&harness.shape(target));
            let radius =
                crate::geometry::length(crate::geometry::difference(handle.position, centre));
            let (u, v) = crate::viewer::tessellate::basis(handle.axis);
            let at = |degrees: f64| {
                let (sin, cos) = degrees.to_radians().sin_cos();
                crate::geometry::sum(
                    centre,
                    crate::geometry::scale(
                        crate::geometry::sum(
                            crate::geometry::scale(u, cos),
                            crate::geometry::scale(v, sin),
                        ),
                        radius,
                    ),
                )
            };
            let grabbed = 0.5 * constants::VIEW_EDIT_ROTATE_ARC_SWEEP_DEGREES;
            harness.move_to(at(grabbed));
            harness.press();
            assert!(
                harness.editor().is_dragging(),
                "scale {scale}: the arc did not take the press"
            );
            harness.move_to(at(grabbed + 40.0));
            harness.pump();
            harness.release();
            harness.pump();

            // 40 degrees swept lands on the 45 degree step, and only that
            // component moved.
            assert_eq!(
                harness.shape(target).rotation_deg(),
                Some([0.0, 0.0, constants::VIEW_EDIT_SNAP_DEGREES * 2.0]),
                "scale {scale}"
            );
            let callout = harness.editor().callout().expect("a callout");
            assert_eq!(callout.at.label(), "45.00 deg", "scale {scale}");
        }
    }

    /// With containment on - which is how a session starts - a drag aimed out
    /// of the domain stops at the wall.
    #[test]
    fn a_drag_stops_at_the_domain_wall_while_containment_is_on() {
        for scale in SCALES {
            let mut harness = Harness::new(scale);
            assert!(harness.editor().containment());
            harness.editor().state.select(Some(TARGET));
            harness.pump();

            // The fixture's domain ends at x = 64 and the sphere has a radius
            // of 4, so 60 is as far as its centre can go.
            let handle = harness.handle(HandleKind::Translate(0));
            harness.move_to(handle.position);
            harness.press();
            harness.move_to([
                handle.position[0] + 30.0,
                handle.position[1],
                handle.position[2],
            ]);
            harness.pump();
            harness.release();
            harness.pump();
            let after = centre_of(&harness.shape(TARGET));
            assert!(
                (after[0] - 60.0).abs() < 1e-6,
                "scale {scale}: the drag went through the wall to {after:?}"
            );
            assert!(
                harness.editor().containment_note().is_some(),
                "scale {scale}: a clamped drag must say so"
            );

            // Switched off, the same drag runs straight out of the domain. It
            // is a shorter one: the pointer has to stay over the 3D view, and
            // past about 90 mm the projected point is off it.
            harness.editor().set_containment(false);
            let handle = harness.handle(HandleKind::Translate(0));
            harness.move_to(handle.position);
            harness.press();
            harness.move_to([
                handle.position[0] + 16.0,
                handle.position[1],
                handle.position[2],
            ]);
            harness.pump();
            harness.release();
            harness.pump();
            let after = centre_of(&harness.shape(TARGET));
            assert!(
                (after[0] - 76.0).abs() < 1e-6,
                "scale {scale}: containment was still on: {after:?}"
            );
        }
    }

    /// The user's second report: *"im trying to move it past the edge via the
    /// editor but it stops when it hits the edge"* - with **keep inside domain**
    /// already unticked.
    ///
    /// Containment was not what stopped it. The view is fitted to the model, so
    /// the pointer position that asks for an object *outside* the domain is
    /// usually outside the 3D view as well - over the side panel, or off the
    /// window entirely - and every such position was dropped, freezing the drag
    /// at about the wall whatever the switch said. Every shape kind here leaves
    /// the domain through a pointer that ends up off the view, by translation and
    /// by resize, on an axis handle and on a plane handle.
    #[test]
    fn with_containment_off_every_kind_of_drag_leaves_the_domain() {
        use crate::geometry::{scale, sum};
        use crate::viewer::editor::tests::ellipsoid_fixture;

        for scale_factor in SCALES {
            // A box moved bodily out through the wall by its own axis arrow.
            let mut harness = Harness::new(scale_factor);
            harness.editor().set_containment(false);
            let domain = domain_bounds(&mut harness);
            let target = Selection::Keepin(0);
            harness.editor().state.select(Some(target));
            harness.pump();
            let handle = harness.handle(HandleKind::Translate(0));
            let to = [
                handle.position[0] + 40.0,
                handle.position[1],
                handle.position[2],
            ];
            assert!(
                harness.is_off_the_view(to),
                "scale {scale_factor}: this drag no longer leaves the 3D view, so it \
                 proves nothing"
            );
            harness.drag_handle(HandleKind::Translate(0), to);
            let moved = bounds_of(&harness.shape(target));
            assert!(
                moved.min[0] > domain.max[0],
                "scale {scale_factor}: the box stopped at {moved:?} inside {domain:?}"
            );

            // The same box grown out through the wall by its +x face handle,
            // which is the handle defect 1 is about: it stands inside the x
            // arrow's shaft.
            let mut harness = Harness::new(scale_factor);
            harness.editor().set_containment(false);
            harness.editor().state.select(Some(target));
            harness.pump();
            let before = bounds_of(&harness.shape(target));
            let handle = harness.handle(HandleKind::Face(0, true));
            let to = [
                handle.position[0] + 30.0,
                handle.position[1],
                handle.position[2],
            ];
            assert!(
                harness.is_off_the_view(to),
                "scale {scale_factor}: this drag no longer leaves the 3D view"
            );
            harness.drag_handle(HandleKind::Face(0, true), to);
            let grown = bounds_of(&harness.shape(target));
            assert!(
                grown.max[0] > domain.max[0],
                "scale {scale_factor}: the face stopped at {grown:?} inside {domain:?}"
            );
            assert!(
                (grown.min[0] - before.min[0]).abs() < 1e-6,
                "scale {scale_factor}: a resize moved the opposite face: {grown:?}"
            );

            // A turned ellipsoid moved bodily out through the lid, which is the
            // shape kind whose bounds - and so whose clamping - come from its
            // support function. Out through z rather than x: this ellipsoid's own
            // x runs nearly along the view direction, where an arrow drag covers
            // a lot of world for very little screen.
            let mut harness = Harness::on(ellipsoid_fixture(), scale_factor);
            harness.editor().set_containment(false);
            harness.editor().state.select(Some(TARGET));
            harness.pump();
            let handle = harness.handle(HandleKind::Translate(2));
            let to = [
                handle.position[0],
                handle.position[1],
                handle.position[2] + 60.0,
            ];
            assert!(
                harness.is_off_the_view(to),
                "scale {scale_factor}: this drag no longer leaves the 3D view"
            );
            harness.drag_handle(HandleKind::Translate(2), to);
            let moved = bounds_of(&harness.shape(TARGET));
            assert!(
                moved.min[2] > domain.max[2],
                "scale {scale_factor}: the ellipsoid stopped at {moved:?} inside {domain:?}"
            );

            // And a cylinder by one cap centre, which is a drag across the
            // camera's plane rather than along an axis.
            let mut harness = Harness::new(scale_factor);
            harness.editor().set_containment(false);
            let cylinder = Selection::Keepout(0);
            harness.editor().state.select(Some(cylinder));
            harness.pump();
            let handle = harness.handle(HandleKind::Endpoint(0));
            let to = sum(handle.position, scale(harness.screen_right(), -80.0));
            assert!(
                harness.is_off_the_view(to),
                "scale {scale_factor}: this drag no longer leaves the 3D view"
            );
            harness.drag_handle(HandleKind::Endpoint(0), to);
            let moved = bounds_of(&harness.shape(cylinder));
            assert!(
                moved.max[0] > domain.max[0],
                "scale {scale_factor}: the cap stopped at {moved:?} inside {domain:?}"
            );
        }
    }

    /// The other half of the same rule: with containment on, those very
    /// gestures are still clamped, and the note still says which switch did it.
    #[test]
    fn with_containment_on_the_same_drags_are_still_clamped() {
        use crate::geometry::{scale, sum};

        let mut harness = Harness::new(1.0);
        let domain = domain_bounds(&mut harness);
        assert!(harness.editor().containment(), "containment starts on");

        // The box, by its arrow and then by its face handle.
        let target = Selection::Keepin(0);
        harness.editor().state.select(Some(target));
        harness.pump();
        let handle = harness.handle(HandleKind::Translate(0));
        harness.drag_handle(
            HandleKind::Translate(0),
            [
                handle.position[0] + 40.0,
                handle.position[1],
                handle.position[2],
            ],
        );
        let kept = bounds_of(&harness.shape(target));
        assert!(
            kept.max[0] <= domain.max[0] + 1e-6,
            "the box left the domain at {kept:?}"
        );
        let note = harness
            .editor()
            .containment_note()
            .expect("a clamped drag must say so");
        assert!(
            note.contains("keep inside domain"),
            "the note has to name the switch that did it: {note}"
        );

        let handle = harness.handle(HandleKind::Face(0, true));
        harness.drag_handle(
            HandleKind::Face(0, true),
            [
                handle.position[0] + 30.0,
                handle.position[1],
                handle.position[2],
            ],
        );
        let kept = bounds_of(&harness.shape(target));
        assert!(
            kept.max[0] <= domain.max[0] + 1e-6,
            "the resized box left the domain at {kept:?}"
        );

        // And the cylinder, by the cap centre that took it out on x above.
        let cylinder = Selection::Keepout(0);
        harness.editor().state.select(Some(cylinder));
        harness.pump();
        let handle = harness.handle(HandleKind::Endpoint(0));
        let to = sum(handle.position, scale(harness.screen_right(), -80.0));
        harness.drag_handle(HandleKind::Endpoint(0), to);
        let kept = bounds_of(&harness.shape(cylinder));
        assert!(
            kept.max[0] <= domain.max[0] + 1e-6,
            "the cap left the domain at {kept:?}"
        );
    }

    /// The user's first report: *"the resize boxes and the move arrows are
    /// overlapping causing the boxes to not be able to be dragged or selected"*.
    ///
    /// The fixture's keepin is 8 mm wide against a gizmo 9 mm long, so its `+x`
    /// face handle stands inside the shaft of the `x` arrow: exactly the part in
    /// hand. A press on it has to resize the box, not move it.
    #[test]
    fn a_press_on_a_face_handle_inside_an_arrow_resizes_rather_than_moves() {
        for scale in SCALES {
            let mut harness = Harness::new(scale);
            let target = Selection::Keepin(0);
            harness.editor().state.select(Some(target));
            harness.pump();
            let face = harness.handle(HandleKind::Face(0, true));
            let arrow = harness.handle(HandleKind::Translate(0));
            assert!(
                arrow.volume.contains(face.position),
                "scale {scale}: the face handle is no longer inside the arrow, so this \
                 test proves nothing"
            );

            // The pointer over it says so before the press: the hover preview is
            // the same ranking a click uses.
            harness.move_to(face.position);
            harness.pump();
            assert_eq!(
                harness.editor().hovered_handle(),
                Some(HandleKind::Face(0, true)),
                "scale {scale}: the arrow took the hover from the face handle"
            );

            let before = bounds_of(&harness.shape(target));
            harness.drag_handle(
                HandleKind::Face(0, true),
                [face.position[0] - 4.0, face.position[1], face.position[2]],
            );
            let after = bounds_of(&harness.shape(target));
            assert!(
                (after.max[0] - (before.max[0] - 4.0)).abs() < 0.3,
                "scale {scale}: the +x face went from {} to {}",
                before.max[0],
                after.max[0]
            );
            assert!(
                (after.min[0] - before.min[0]).abs() < 1e-6,
                "scale {scale}: the box moved instead of being resized: {after:?}"
            );
        }
    }

    /// The centre cube is drawn as one of the same markers, and all three shafts
    /// begin on it, so it needs the rule the resize cubes got. The arrows are
    /// still the arrows anywhere along their length.
    #[test]
    fn a_press_on_the_centre_cube_grabs_it_rather_than_an_arrow() {
        use crate::geometry::{difference, length, scale, sum};

        for scale_factor in SCALES {
            let mut harness = Harness::new(scale_factor);
            harness.editor().state.select(Some(TARGET));
            harness.editor().snap_mut().millimetres = 0.0;
            harness.pump();
            let centre = harness.handle(HandleKind::TranslateFree);
            for d in 0..3 {
                let arrow = harness.handle(HandleKind::Translate(d));
                assert!(
                    arrow.volume.contains(centre.position),
                    "scale {scale_factor}: axis {d}'s arrow no longer starts on the \
                     centre handle, so this test proves nothing"
                );
            }
            harness.move_to(centre.position);
            harness.pump();
            assert_eq!(
                harness.editor().hovered_handle(),
                Some(HandleKind::TranslateFree),
                "scale {scale_factor}: an arrow took the hover from the centre cube"
            );

            // And it drags what it is: the whole shape, across the camera's
            // plane, by what the pointer covered in it.
            let before = centre_of(&harness.shape(TARGET));
            let travel = scale(harness.screen_right(), 6.0);
            harness.drag_handle(HandleKind::TranslateFree, sum(centre.position, travel));
            let after = centre_of(&harness.shape(TARGET));
            let moved = difference(after, before);
            assert!(
                length(difference(moved, travel)) < 0.3,
                "scale {scale_factor}: moved {moved:?} rather than {travel:?}"
            );

            // Half way along the x arrow - away from the centre cube - is the
            // arrow, which is what makes the whole shaft grabbable.
            let arrow = harness.handle(HandleKind::Translate(0));
            let anchor = centre_of(&harness.shape(TARGET));
            let along = sum(anchor, scale(difference(arrow.position, anchor), 0.5));
            harness.move_to(along);
            harness.pump();
            assert_eq!(
                harness.editor().hovered_handle(),
                Some(HandleKind::Translate(0)),
                "scale {scale_factor}: the centre cube reached down the whole shaft"
            );
        }
    }

    /// The same for the radius handles of an ellipsoid, whose short semi-axes
    /// stand well inside the arrows along them.
    ///
    /// The ellipsoid round tested the *long* axis and said in as many words that
    /// the short ones were unreachable; this is that case.
    #[test]
    fn a_radius_handle_inside_an_arrow_can_be_dragged() {
        use crate::viewer::editor::tests::ellipsoid_fixture;

        for scale in SCALES {
            let mut harness = Harness::on(ellipsoid_fixture(), scale);
            harness.editor().state.select(Some(TARGET));
            harness.editor().snap_mut().millimetres = 0.0;
            harness.pump();
            let ShapeSpec::Ellipsoid { radii, .. } = harness.shape(TARGET) else {
                panic!("scale {scale}: the fixture's second keepout is an ellipsoid");
            };
            assert_eq!(radii, [8.0, 4.0, 4.0]);

            let handle = harness.handle(HandleKind::Radius(1));
            let arrow = harness.handle(HandleKind::Translate(1));
            assert!(
                arrow.volume.contains(handle.position),
                "scale {scale}: the radius handle is no longer inside the arrow"
            );
            let travel = 3.0;
            harness.drag_handle(
                HandleKind::Radius(1),
                crate::geometry::sum(handle.position, crate::geometry::scale(handle.axis, travel)),
            );
            let ShapeSpec::Ellipsoid { center, radii, .. } = harness.shape(TARGET) else {
                panic!("scale {scale}: the shape changed kind");
            };
            assert!(
                (radii[1] - 7.0).abs() < 0.3,
                "scale {scale}: radii came out {radii:?}"
            );
            assert_eq!([radii[0], radii[2]], [8.0, 4.0], "scale {scale}: one axis");
            assert_eq!(center, [50.0, 16.0, 16.0], "scale {scale}: it moved");
        }
    }

    /// How far a 4 pixel pan moves the camera from the fitted view, with the
    /// delta reference known to be up to date: the yardstick the two tests below
    /// compare their own 4 pixel pans against.
    ///
    /// The absolute figure is the harness's rather than a window's - with no GPU
    /// the viewport height a pan divides by falls back to one - but it is the same
    /// camera and the same four pixels every time, and what a pixel of pan is
    /// worth depends only on the camera's distance, which neither a gizmo drag nor
    /// a pan changes.
    fn four_pixel_pan_distance() -> f64 {
        use crate::geometry::{difference, length};

        let mut harness = Harness::new(1.0);
        harness.editor().state.select(Some(TARGET));
        harness.pump();
        let anchor = centre_of(&harness.shape(TARGET));
        let (x, y) = harness.pixel_of(anchor);
        harness.move_to_pixel(x, y);
        let before = harness.app.camera.target;
        harness.right_press();
        harness.move_to_pixel(x + 4.0, y);
        harness.right_release();
        let moved = length(difference(harness.app.camera.target, before));
        assert!(moved > 0.0, "the pan did nothing to measure against");
        moved
    }

    /// A gizmo drag claims every pointer move it sees, so the camera never gets
    /// to update the point its next delta is measured from. Unless that point is
    /// caught up as the drag goes, the first camera gesture around one snaps by
    /// everything the drag covered.
    ///
    /// The discriminating gesture has *no* pointer move between the drag's
    /// release and the camera press: any move would refresh the very reference
    /// the bug leaves stale. What holds it up is the drag's own claimed moves
    /// writing that reference as they pass - the catch-up at the release is belt
    /// and braces - and the drag below is deliberately long enough that measuring
    /// from its start instead is off by an order of magnitude.
    #[test]
    fn a_camera_gesture_straight_after_a_gizmo_drag_measures_from_the_pointer() {
        use crate::geometry::{difference, length};

        let clean = four_pixel_pan_distance();

        // The same pan, pressed the instant a gizmo drag many times longer than
        // 4 pixels ends, with the pointer where that drag left it.
        let mut harness = Harness::new(1.0);
        harness.editor().state.select(Some(TARGET));
        harness.pump();
        let handle = harness.handle(HandleKind::Translate(0));
        let to = [
            handle.position[0] + 8.0,
            handle.position[1],
            handle.position[2],
        ];
        let (px, py) = harness.pixel_of(handle.position);
        let (dx, dy) = harness.pixel_of(to);
        let travelled = ((dx - px).powi(2) + (dy - py).powi(2)).sqrt();
        assert!(
            travelled > 20.0,
            "the drag covers {travelled} pixels, too few to tell the two \
             references apart"
        );
        harness.move_to(handle.position);
        harness.press();
        assert!(harness.editor().is_dragging(), "the drag never started");
        harness.move_to(to);
        harness.pump();
        harness.release();
        harness.pump();
        assert!(!harness.editor().is_dragging());

        let before = harness.app.camera.target;
        harness.right_press();
        harness.move_to_pixel(dx + 4.0, dy);
        harness.right_release();
        let after = length(difference(harness.app.camera.target, before));
        assert!(
            (after - clean).abs() < 1e-9,
            "the pan moved the camera {after} mm rather than the {clean} mm four \
             pixels are worth: its delta was measured from where the gizmo drag \
             began, not from where the pointer is"
        );
    }

    /// The same invariant part way *through* a gizmo drag, which nothing at the
    /// ends of it could cover: the drag has already moved, so the reference has to
    /// have followed it position by position.
    ///
    /// A pan pressed here - the left button still down on the handle - measures
    /// its first delta from where the pointer was when it was pressed.
    #[test]
    fn a_camera_gesture_during_a_gizmo_drag_measures_from_where_it_was_pressed() {
        use crate::geometry::{difference, length};

        let clean = four_pixel_pan_distance();

        let mut harness = Harness::new(1.0);
        harness.editor().state.select(Some(TARGET));
        harness.pump();
        let handle = harness.handle(HandleKind::Translate(0));
        let to = [
            handle.position[0] + 8.0,
            handle.position[1],
            handle.position[2],
        ];
        let (px, py) = harness.pixel_of(handle.position);
        let (dx, dy) = harness.pixel_of(to);
        let travelled = ((dx - px).powi(2) + (dy - py).powi(2)).sqrt();
        assert!(
            travelled > 20.0,
            "the drag covers {travelled} pixels, too few to tell the two \
             references apart"
        );
        harness.move_to(handle.position);
        harness.press();
        harness.move_to(to);
        harness.pump();
        assert!(
            harness.editor().is_dragging(),
            "the drag has to still be held for this to be the mid-drag case"
        );

        // The pan starts without the gizmo drag ending: both gestures are live
        // from here, and the camera's first delta is the one under test.
        let before = harness.app.camera.target;
        harness.right_press();
        harness.move_to_pixel(dx + 4.0, dy);
        let during = length(difference(harness.app.camera.target, before));
        assert!(
            (during - clean).abs() < 1e-9,
            "the pan moved the camera {during} mm rather than the {clean} mm four \
             pixels are worth: its delta was measured from where the gizmo drag \
             began, not from where the pointer was when the pan was pressed"
        );
        harness.right_release();
        harness.release();
        harness.pump();
    }

    #[test]
    fn a_press_after_a_drag_casts_from_where_the_pointer_is_now() {
        // The regression: the press ray used a position only the camera path
        // maintained, so after a gizmo drag - which the camera never sees - it
        // was cast from wherever that drag began. A click somewhere else then
        // grabbed or picked whatever was under the *old* point, and the first
        // move of the next drag jumped by the distance between them.
        //
        // The discriminating gesture is a press with *no* cursor move between
        // it and the drag's release: any move would refresh the very position
        // the bug left stale, so a sequence containing one cannot tell the two
        // apart. The drag below is long enough that where it began is empty
        // space - the whole gizmo has moved off it - so a press cast from there
        // grabs nothing at all.
        let mut harness = Harness::new(1.0);
        // This drag deliberately runs the object up and out through the lid of
        // the domain, which is a wall the containment rule would stop it at;
        // what is under test here is the pointer position a press casts from,
        // so the rule is switched off rather than worked around.
        harness.editor().set_containment(false);
        harness.editor().state.select(Some(TARGET));
        harness.pump();
        let handle = harness.handle(HandleKind::Translate(2));
        let travel = 12.0;
        harness.move_to(handle.position);
        harness.press();
        harness.move_to([
            handle.position[0],
            handle.position[1],
            handle.position[2] + travel,
        ]);
        harness.release();
        harness.pump();
        let after_drag = centre_of(&harness.shape(TARGET));
        assert!(
            !harness.editor().is_dragging(),
            "the release ended the drag"
        );

        // The pointer has not moved since the release, and the z arrow is now
        // under it. Pressing must grab that arrow.
        let moved = harness.handle(HandleKind::Translate(2));
        harness.press();
        assert!(
            harness.editor().is_dragging(),
            "a press with no move since the drag must cast from where the \
             pointer is, not from where that drag began"
        );
        assert_eq!(
            harness.selection(),
            Some(TARGET),
            "grabbing a handle must not change the selection"
        );
        // And it is that arrow it grabbed: the object follows it along z again.
        let second = 4.0;
        harness.move_to([
            moved.position[0],
            moved.position[1],
            moved.position[2] + second,
        ]);
        harness.pump();
        harness.release();
        harness.pump();
        let after_second = centre_of(&harness.shape(TARGET));
        assert!(
            (after_second[2] - after_drag[2] - second).abs() < 0.3,
            "the second drag moved the object from {after_drag:?} to {after_second:?}, \
             expected {second} mm along z"
        );
        assert!(
            (after_second[0] - after_drag[0]).abs() < 0.05
                && (after_second[1] - after_drag[1]).abs() < 0.05,
            "an axis drag moved off its axis: {after_drag:?} -> {after_second:?}"
        );
        let after_drag = after_second;

        // Now click on empty space, far from where that drag ended.
        harness.move_to_pixel(4.0, 4.0);
        harness.press();
        harness.release();
        harness.pump();
        assert_eq!(
            harness.selection(),
            None,
            "a click on empty space must deselect, not act on the last drag's position"
        );
        assert!(
            !harness.editor().is_dragging(),
            "a click on empty space must not start a drag"
        );
        let unchanged = centre_of(&harness.shape(TARGET));
        for d in 0..3 {
            assert!(
                (unchanged[d] - after_drag[d]).abs() < 1e-9,
                "the click moved the object: {unchanged:?} against {after_drag:?}"
            );
        }
    }

    #[test]
    fn clicking_an_object_selects_it_and_empty_space_clears_it() {
        // The fixture's load region sits inside its keepin pad, so this also
        // pins the ranking: the ray enters the pad first, and the load - the
        // thing inside it - is what the click means.
        let load = Selection::Load { case: 0, load: 0 };
        for scale in SCALES {
            let mut harness = Harness::new(scale);
            let region = harness.shape(load);
            harness.click(centre_of(&region));
            assert_eq!(
                harness.selection(),
                Some(load),
                "scale {scale}: clicking an object must select it"
            );
            harness.move_to_pixel(2.0, 2.0);
            harness.press();
            harness.release();
            harness.pump();
            assert_eq!(harness.selection(), None, "scale {scale}");
        }
    }

    /// While nothing but the part is on screen, the viewport takes no editing
    /// at all: nothing is hovered, no handle is grabbed, no drag moves anything
    /// and no click selects. The camera is not what was suspended, and
    /// unticking the box - or the part going away under a ticked one - gives
    /// every gesture straight back.
    ///
    /// Driven through the window's own event path, because that is where the
    /// suspension is decided: what the gestures below are refused by is the
    /// mode and not a missing handle, and the handles are asserted to be there
    /// for exactly that reason.
    #[test]
    fn showing_nothing_but_the_part_suspends_editing_in_the_viewport() {
        let load = Selection::Load { case: 0, load: 0 };
        let mut harness = Harness::new(1.0);
        // What a finished run leaves on screen, and what the switch shows on
        // its own. Given to the scene after the opening fit, so the camera
        // these gestures are aimed through is the one every other test drives.
        harness.app.scene.set(
            Layer::Density,
            Some(LayerMesh::from_mesh(
                &tessellate::box_mesh([24.0, 12.0, 12.0], [40.0, 20.0, 20.0]),
                Layer::Density.info().color,
                crate::viewer::scene::Shading::Smooth,
            )),
        );
        let region = centre_of(&harness.shape(load));
        harness.editor().state.select(Some(load));
        harness.pump();
        let handle = harness.handle(HandleKind::Translate(0));
        let before = centre_of(&harness.shape(load));

        *harness.app.scene.isolate_part_mut() = true;
        assert!(
            !harness.app.scene.is_visible(Layer::Gizmo),
            "the handles are still drawn"
        );
        assert!(
            !harness.editor().handles.is_empty(),
            "the handles are hidden, not gone: a gesture at one has to be \
             refused by the mode rather than by there being nothing there"
        );

        // Nothing under the pointer is hovered - neither the object nor the
        // handle that stands at its centre.
        harness.move_to(region);
        harness.pump();
        assert_eq!(
            harness.editor().hover(),
            None,
            "an object nobody can see was hovered"
        );
        assert_eq!(
            harness.editor().hovered_handle(),
            None,
            "a handle nobody can see was hovered"
        );

        // No handle is grabbed, and the drag that would have followed moves
        // nothing.
        harness.move_to(handle.position);
        harness.press();
        assert!(
            !harness.editor().is_dragging(),
            "a handle nobody can see was grabbed"
        );
        harness.move_to([
            handle.position[0] + 8.0,
            handle.position[1],
            handle.position[2],
        ]);
        harness.pump();
        harness.release();
        harness.pump();
        assert_eq!(
            centre_of(&harness.shape(load)),
            before,
            "a suspended drag moved the object"
        );

        // And a click selects nothing.
        harness.editor().state.select(None);
        harness.pump();
        harness.click(region);
        assert_eq!(
            harness.selection(),
            None,
            "an object nobody can see was picked"
        );

        // The camera is not what was suspended: the very gesture that grabs
        // nothing orbits, which is what the mode is for.
        let yaw = harness.app.camera.yaw;
        harness.move_to_pixel(100.0, 100.0);
        harness.press();
        harness.move_to_pixel(160.0, 120.0);
        harness.release();
        assert!(
            (harness.app.camera.yaw - yaw).abs() > 1e-9,
            "the orbit stopped with the editing"
        );

        // Unticking gives the hover and the clicks back - the same pointer
        // position, which is what says the ones above were refused by the mode.
        *harness.app.scene.isolate_part_mut() = false;
        harness.move_to(region);
        harness.pump();
        assert_eq!(
            harness.editor().hover(),
            Some(load),
            "unticking the box did not give the hover back"
        );
        harness.click(region);
        assert_eq!(
            harness.selection(),
            Some(load),
            "unticking the box did not give the clicks back"
        );

        // And so does the part going away under a ticked box, which is what an
        // edit that invalidates the run does to it: the viewport is never left
        // showing nothing and taking nothing.
        *harness.app.scene.isolate_part_mut() = true;
        harness.editor().state.select(None);
        harness.pump();
        harness.app.scene.clear_density();
        harness.click(region);
        assert_eq!(
            harness.selection(),
            Some(load),
            "a part that went away left the viewport inert"
        );
    }

    #[test]
    fn the_domain_is_selected_from_the_tree_and_never_by_a_click() {
        let mut harness = Harness::new(1.0);
        // A point inside the domain that no other object's shape covers, and
        // that nothing else lies in front of along the view direction: the
        // fixture's block spans 64 x 32 x 32 mm, its keepout cylinder stands at
        // x = 16, and its keepin pad and load sit past x = 56.
        harness.click([35.0, 28.0, 5.0]);
        assert_eq!(
            harness.selection(),
            None,
            "a click on bare domain must not select it"
        );

        // From the tree it selects, and once selected its gizmo works like any
        // other object's.
        harness.editor().state.select(Some(Selection::Domain(0)));
        harness.pump();
        assert!(harness.app.scene.get(Layer::Gizmo).is_some());
        let handle = harness.handle(HandleKind::Translate(1));
        let before = centre_of(&harness.shape(Selection::Domain(0)));
        harness.move_to(handle.position);
        harness.press();
        assert!(
            harness.editor().is_dragging(),
            "a domain selected from the tree must still be draggable"
        );
        harness.move_to([
            handle.position[0],
            handle.position[1] + 6.0,
            handle.position[2],
        ]);
        harness.release();
        harness.pump();
        let after = centre_of(&harness.shape(Selection::Domain(0)));
        assert!(
            (after[1] - before[1] - 6.0).abs() < 0.3,
            "the domain did not follow its handle: {before:?} -> {after:?}"
        );
    }

    /// The outline previews the click: what the pointer is over is what a click
    /// there selects, rank rule and all.
    ///
    /// The fixture's load region sits inside its keepin pad, so this is the
    /// overlapping case: depth alone would say the pad, and both the click and
    /// the hover have to say the load.
    #[test]
    fn the_hover_previews_exactly_what_a_click_would_select() {
        let load = Selection::Load { case: 0, load: 0 };
        for scale in SCALES {
            let mut harness = Harness::new(scale);
            let region = harness.shape(load);
            harness.move_to(centre_of(&region));
            harness.pump();
            assert_eq!(
                harness.editor().hover(),
                Some(load),
                "scale {scale}: the hover did not rank the overlap the way a click does"
            );
            // Nothing is selected yet, so the outline really is drawn.
            assert!(
                harness
                    .app
                    .scene
                    .get(Layer::Hover)
                    .is_some_and(|mesh| mesh.triangles() > 0),
                "scale {scale}: the hovered object got no outline"
            );

            // And the click there agrees with it.
            harness.press();
            harness.release();
            harness.pump();
            assert_eq!(harness.selection(), Some(load), "scale {scale}");
            // Once it *is* the selection, the selection shell says so and the
            // hover outline steps aside rather than doubling it up.
            assert_eq!(harness.editor().hover(), Some(load));
            assert!(harness.app.scene.get(Layer::Selection).is_some());
            assert!(harness.app.scene.get(Layer::Hover).is_none());
        }
    }

    /// Empty space, the panel and the bare domain are all "nothing to click",
    /// and the outline says so by not being there.
    #[test]
    fn the_hover_clears_over_empty_space_the_panel_and_the_domain() {
        let mut harness = Harness::new(1.0);
        let region = harness.shape(Selection::Load { case: 0, load: 0 });
        harness.move_to(centre_of(&region));
        harness.pump();
        assert!(harness.editor().hover().is_some());

        // Empty space.
        harness.move_to_pixel(2.0, 2.0);
        harness.pump();
        assert_eq!(harness.editor().hover(), None);
        assert!(harness.app.scene.get(Layer::Hover).is_none());

        // Bare domain: not viewport-pickable, and so not hoverable either.
        harness.move_to([35.0, 28.0, 5.0]);
        harness.pump();
        assert_eq!(
            harness.editor().hover(),
            None,
            "the domain must never be hovered, for the reason it is never clicked"
        );

        // Over the side panel: past the 3D view's width there is no ray at all.
        harness.move_to(centre_of(&region));
        harness.pump();
        assert!(harness.editor().hover().is_some());
        let (width, height) = harness.app.surface_size.expect("a surface");
        let panel_x = f64::from(harness.app.scene_viewport_width(width)) + 4.0;
        harness.move_to_pixel(panel_x, f64::from(height) / 2.0);
        harness.pump();
        assert_eq!(harness.editor().hover(), None, "hover through the panel");
        assert!(harness.app.scene.get(Layer::Hover).is_none());

        // And leaving the window entirely.
        harness.move_to(centre_of(&region));
        harness.pump();
        assert!(harness.editor().hover().is_some());
        harness.app.on_window_event(WindowEvent::CursorLeft {
            device_id: DeviceId::dummy(),
        });
        harness.pump();
        assert_eq!(harness.editor().hover(), None);
    }

    /// A drag owns the pointer: neither a gizmo drag nor a camera orbit leaves
    /// an outline trailing under the cursor.
    #[test]
    fn a_drag_suppresses_the_hover_until_it_ends() {
        let mut harness = Harness::new(1.0);
        harness.editor().state.select(Some(TARGET));
        harness.pump();

        // Over a handle: the object under it is not what a click would take, so
        // nothing is outlined and the handle brightens instead.
        let handle = harness.handle(HandleKind::Translate(0));
        harness.move_to(handle.position);
        harness.pump();
        assert_eq!(
            harness.editor().hover(),
            None,
            "a pointer over a handle would grab it, not pick an object"
        );
        assert_eq!(
            harness.editor().hovered_handle(),
            Some(HandleKind::Translate(0))
        );
        assert!(harness.app.scene.get(Layer::Hover).is_none());

        // Dragging it: still nothing, wherever the pointer goes.
        harness.press();
        assert!(harness.editor().is_dragging());
        harness.move_to([
            handle.position[0] + 6.0,
            handle.position[1],
            handle.position[2],
        ]);
        harness.pump();
        assert_eq!(harness.editor().hover(), None);
        assert_eq!(harness.editor().hovered_handle(), None);
        assert!(harness.app.scene.get(Layer::Hover).is_none());
        harness.release();
        harness.pump();

        // A camera orbit is a drag too: the geometry is moving under the
        // pointer rather than being pointed at. Nothing is selected for this
        // half, so no gizmo is in the way of the pointer.
        harness.editor().state.select(None);
        harness.pump();
        let load = Selection::Load { case: 0, load: 0 };
        let region = harness.shape(load);
        let (x, y) = harness.pixel_of(centre_of(&region));
        harness.move_to_pixel(x, y);
        harness.pump();
        assert_eq!(harness.editor().hover(), Some(load));
        harness.press();
        harness.move_to_pixel(x + 60.0, y);
        harness.pump();
        assert_eq!(
            harness.editor().hover(),
            None,
            "an orbit must not leave an outline under the cursor"
        );
        harness.release();
        harness.pump();
    }

    #[test]
    fn a_press_that_travelled_orbits_instead_of_selecting() {
        let mut harness = Harness::new(1.0);
        let before = harness.app.camera.yaw;
        let sphere = harness.shape(TARGET);
        let (x, y) = harness.pixel_of(centre_of(&sphere));
        harness.move_to_pixel(x, y);
        harness.press();
        // Well past the click slop, so this is a camera drag by any measure.
        harness.move_to_pixel(x + 60.0, y);
        harness.release();
        harness.pump();
        assert_eq!(
            harness.selection(),
            None,
            "a drag across the viewport is an orbit, not a selection"
        );
        assert!(
            (harness.app.camera.yaw - before).abs() > 1e-6,
            "the camera did not orbit"
        );
    }

    #[test]
    fn the_selection_shell_and_handles_follow_the_object_through_a_resize() {
        let mut harness = Harness::new(1.0);
        // The fixture's keepin box, which has faces to drag.
        let target = Selection::Keepin(0);
        harness.editor().state.select(Some(target));
        harness.pump();

        // A face dragged inwards: the shell must come with it rather than
        // staying around the size the object used to be.
        let handle = harness.handle(HandleKind::Face(0, true));
        harness.move_to(handle.position);
        harness.press();
        harness.move_to([
            handle.position[0] - 4.0,
            handle.position[1],
            handle.position[2],
        ]);
        harness.pump();
        harness.release();
        harness.pump();
        assert_shell_fits(&mut harness, target, "after a drag");

        // The handles moved with it: the face handle now sits on the new face.
        let shape = harness
            .shape(target)
            .to_shape("test")
            .expect("a shape")
            .bounds();
        let moved = harness.handle(HandleKind::Face(0, true));
        assert!(
            (moved.position[0] - shape.max[0]).abs() < 1e-9,
            "the face handle stayed at {} while the face is at {}",
            moved.position[0],
            shape.max[0]
        );

        // The same after a numeric edit, and after undoing it.
        harness.editor().state.edit(|config| {
            config.keepin[0] = ShapeSpec::Box {
                min: [10.0, 10.0, 10.0],
                max: [14.0, 14.0, 14.0],
                rotation_deg: None,
            };
        });
        harness.pump();
        assert_shell_fits(&mut harness, target, "after a numeric edit");
        let shell = harness
            .app
            .scene
            .get(Layer::Selection)
            .expect("a shell")
            .bounds;
        assert!(
            shell.max[0] - shell.min[0] < 8.0,
            "the shell is still the size the object used to be: {shell:?}"
        );

        harness.editor().undo();
        harness.pump();
        assert_shell_fits(&mut harness, target, "after an undo");
    }

    /// The selection shell has to enclose the object it is drawn around, and
    /// not by much: it is a highlight, not a second object.
    fn assert_shell_fits(harness: &mut Harness<'_>, selection: Selection, what: &str) {
        let shape = harness
            .shape(selection)
            .to_shape("test")
            .expect("a shape")
            .bounds();
        let shell = harness
            .app
            .scene
            .get(Layer::Selection)
            .expect("a selection shell")
            .bounds;
        for d in 0..3 {
            assert!(
                shell.min[d] <= shape.min[d] && shell.max[d] >= shape.max[d],
                "{what}, axis {d}: the shell {shell:?} does not enclose the shape {shape:?}"
            );
            let extent = shape.max[d] - shape.min[d];
            let slack = (shell.max[d] - shell.min[d]) - extent;
            assert!(
                slack < 0.5 * extent.max(1.0),
                "{what}, axis {d}: the shell is {slack} mm larger than the {extent} mm it draws"
            );
        }
    }

    #[test]
    fn the_scene_viewport_gives_up_exactly_the_panel_width() {
        let points = constants::VIEW_PANEL_WIDTH_POINTS;
        let panel = points as u32;
        assert_eq!(scene_width_for(1280, 1.0, points), 1280 - panel);
        // A high dpi surface spends proportionally more pixels on the panel.
        assert_eq!(scene_width_for(2560, 2.0, points), 2560 - 2 * panel);
        // A window narrower than the panel still leaves a legal viewport.
        assert_eq!(scene_width_for(100, 1.0, points), 1);
        assert_eq!(scene_width_for(0, 1.0, points), 1);
        // The editor's wider panel takes proportionally more, and the viewport
        // is sized against whichever one is showing.
        let editor = constants::VIEW_EDIT_PANEL_WIDTH_POINTS;
        assert_eq!(
            scene_width_for(1280, 1.0, editor),
            1280 - editor as u32,
            "the editor panel is not the one the viewport gave up"
        );
        assert!(scene_width_for(1280, 1.0, editor) < scene_width_for(1280, 1.0, points));
    }

    #[test]
    fn fitting_before_the_first_frame_leaves_the_opening_fit_to_run() {
        let mut app = ViewerApp::new("test".to_string(), scene_with_a_box(), None);
        assert!(!app.fitted);

        // F pressed before any frame: the camera does move, but the flag that
        // gates the opening fit must not latch, because the panel's width in
        // physical pixels is still a guess.
        app.fit();
        // A default camera sits on the origin one millimetre out, so both of
        // these separate a real fit from a silent no-op. The box spans
        // 10 x 20 x 30 mm about the origin: its centre is [5, 10, 15] and its
        // bounding sphere has a radius of 18.71 mm, which a 45 degree field of
        // view frames from about 56 mm away.
        assert_eq!(app.camera.target, [5.0, 10.0, 15.0]);
        assert!(
            (50.0..60.0).contains(&app.camera.distance),
            "expected the fit to back off about 56 mm, got {}",
            app.camera.distance
        );
        assert!(
            !app.fitted,
            "an early fit must not suppress the opening fit"
        );

        // Once a frame has reported the real viewport, a fit is authoritative.
        app.scene_width = Some(1020);
        app.fit();
        assert!(app.fitted);
    }

    #[test]
    fn a_closed_window_keeps_pumping_until_the_run_behind_it_finishes() {
        // The regression this pins: a process that owns window state and stops
        // servicing its message queue is terminated by Windows as hung, which
        // used to kill a detached run part way through. After teardown the loop
        // may only stop once the worker has finished.
        let finished = AtomicBool::new(false);
        let still_running = || !finished.load(Ordering::Relaxed);
        let link = ViewLink::new();
        let mut app = ViewerApp::new("test".to_string(), scene_with_a_box(), Some(&link))
            .watching(&still_running);

        // While the window is up the drain rule does not apply at all.
        assert!(!app.should_keep_pumping());

        app.begin_teardown();
        assert!(app.torn_down);
        assert!(
            app.should_keep_pumping(),
            "the loop must outlive the window while the run is going"
        );
        // Teardown releases the window and detaches the run in one step.
        assert!(app.window.is_none() && app.gpu.is_none() && app.egui_state.is_none());
        assert!(link.is_detached(), "teardown must detach the run");

        finished.store(true, Ordering::Relaxed);
        assert!(
            !app.should_keep_pumping(),
            "the loop must stop once the run is done"
        );
    }

    /// The same rule for an editor window, whose runs are stopped rather than
    /// detached by the close - and stopped only *cooperatively*, at the next
    /// checkpoint of a stage that can be seconds long. Leaving the loop at the
    /// close and waiting for those threads outside it is the hang this whole
    /// contract exists to prevent, and edit mode is not exempt from it.
    #[test]
    fn an_editor_window_keeps_pumping_until_its_worker_ends() {
        use crate::viewer::editor::Editor;
        use crate::viewer::editor::tests::growth_fixture;

        let (_dir, path) = write_temp("app_editor_pump", growth_fixture());
        let stl = path.with_file_name("fixture.stl");
        let editor = Editor::open(&path).expect("open");
        let scene = editor.initial_scene();
        // No closure is taken: an editing session is asked about itself, and
        // asked of whichever session the window is holding - exactly as `edit`
        // hands it over.
        let mut app = ViewerApp::new("test".to_string(), scene, None).editing(editor);

        app.editor.as_mut().expect("an editor").start_full_run();
        assert!(
            app.editor.as_ref().expect("an editor").is_running_full(),
            "the run under test never started"
        );
        // While the window is up the drain rule does not apply at all.
        assert!(!app.should_keep_pumping());

        // The real close path: no window, no GPU, one synthesized event.
        app.on_window_event(WindowEvent::CloseRequested);
        assert!(app.torn_down && app.window.is_none());
        assert!(
            app.should_keep_pumping(),
            "the loop left while an editor thread was still winding down"
        );

        // The loop stops only once the threads have ended - which is what makes
        // this join, the one `edit` performs after `run`, a formality.
        app.finish_editing();
        assert!(
            !app.should_keep_pumping(),
            "the loop must stop once the editor's threads are over"
        );
        // And the semantics of the close are untouched: a stopped run writes
        // nothing, however long the window went on pumping for it.
        assert!(!stl.exists(), "a stopped run wrote {}", stl.display());
    }

    /// A file switch mid-session: the run behind the window is stopped and
    /// collected, a whole new session takes its place, and - the half that has
    /// to be pinned - what the loop watches is the *new* session's worker. A
    /// window still watching the session it opened with would leave with a live
    /// thread behind it, which is the hang the pumping rule exists to prevent.
    #[test]
    fn switching_files_replaces_the_session_and_the_loop_watches_the_new_one() {
        use crate::viewer::editor::Editor;
        use crate::viewer::editor::tests::growth_fixture;

        let (_dir, path) = write_temp("app_switch", growth_fixture());
        let other = path.with_file_name("other.toml");
        std::fs::write(&other, growth_fixture()).expect("write the second file");
        let stl = path.with_file_name("fixture.stl");

        let editor = Editor::open(&path).expect("open");
        let scene = editor.initial_scene();
        let mut app = ViewerApp::new("test".to_string(), scene, None).editing(editor);
        app.scene_width = Some(900);
        app.surface_size = Some((1280, 800));

        // A run behind the window, and then the switch: the document is clean,
        // so nothing is asked and the switch is set up at once.
        app.editor.as_mut().expect("an editor").start_full_run();
        assert!(app.editor.as_ref().expect("an editor").is_running_full());
        assert!(
            app.editor
                .as_mut()
                .expect("an editor")
                .request_open(other.clone())
        );
        app.pump_editor();

        {
            let editor = app.editor.as_ref().expect("an editor");
            assert_eq!(editor.state.path(), other, "the session is on the new file");
            assert!(
                !editor.is_running(),
                "the run of the file that was left behind survived the switch"
            );
            assert!(
                !editor.can_generate_stl(),
                "the design of the old session came with it"
            );
            assert!(!editor.is_switching() && !editor.is_asking());
        }
        assert!(
            app.title.contains("other.toml"),
            "the window is still titled after the old file: {}",
            app.title
        );
        assert!(app.fitted, "the camera was not framed on the new document");
        assert!(!app.scene.bounds().is_empty());
        assert!(
            !stl.exists(),
            "the run that was stopped by the switch wrote {}",
            stl.display()
        );

        // The structural half. A run started after the switch has to keep the
        // closed window pumping, which only the current session's probe knows
        // about.
        app.editor.as_mut().expect("an editor").start_full_run();
        assert!(app.editor.as_ref().expect("an editor").is_running_full());
        assert!(!app.should_keep_pumping(), "the window is still up");

        app.on_window_event(WindowEvent::CloseRequested);
        assert!(app.torn_down && app.window.is_none());
        assert!(
            app.should_keep_pumping(),
            "the loop stopped watching the session the window now holds"
        );
        app.finish_editing();
        assert!(
            !app.should_keep_pumping(),
            "the loop must stop once the current session's threads are over"
        );
        assert!(!stl.exists(), "a stopped run wrote {}", stl.display());
    }

    /// A switch that cannot happen costs the session nothing but its runs: the
    /// document is still there, on the file it was always on, and the panel says
    /// what went wrong.
    #[test]
    fn a_switch_to_a_file_that_will_not_open_keeps_the_session_and_says_why() {
        use crate::viewer::editor::Editor;

        let (_dir, path) = write_temp("app_switch_broken", fixture());
        let broken = path.with_file_name("broken.toml");
        std::fs::write(&broken, "[project]\nname = = \"no\"\n").expect("write the broken file");

        let editor = Editor::open(&path).expect("open");
        let scene = editor.initial_scene();
        let mut app = ViewerApp::new("test".to_string(), scene, None).editing(editor);
        assert!(
            app.editor
                .as_mut()
                .expect("an editor")
                .request_open(broken.clone())
        );
        app.pump_editor();

        let editor = app.editor.as_ref().expect("an editor");
        assert_eq!(
            editor.state.path(),
            path,
            "the document was lost to a switch that could not happen"
        );
        assert!(
            editor
                .status_line()
                .expect("a status")
                .contains("broken.toml"),
            "{:?}",
            editor.status_line()
        );
        assert!(!editor.is_switching(), "the switch must not be retried");
    }

    #[test]
    fn the_density_layer_starts_smooth_and_the_flat_switch_is_noticed() {
        let mut app = ViewerApp::new("test".to_string(), scene_with_a_box(), None);
        assert_eq!(app.flat_shading, constants::VIEW_DEFAULT_FLAT_SHADING);
        assert!(!app.flat_shading, "the surface must start smooth");

        // With the panel and the uploaded buffer agreeing, nothing is re-sent.
        assert_eq!(app.scene.flat_shading(), app.flat_shading);
        *app.scene.flat_shading_mut() = true;
        assert_ne!(app.scene.flat_shading(), app.flat_shading);

        // An upload catches the buffer up with the panel; there is no GPU in a
        // test, so what this pins is the state the frame loop compares against.
        app.upload_density();
        assert!(app.flat_shading);
        assert_eq!(app.scene.flat_shading(), app.flat_shading);
    }

    #[test]
    fn a_setup_view_stops_as_soon_as_its_window_closes() {
        // Nothing runs behind a setup view, so there is nothing to outlive.
        let mut app = ViewerApp::new("test".to_string(), scene_with_a_box(), None);
        app.begin_teardown();
        assert!(app.torn_down);
        assert!(!app.should_keep_pumping());
    }

    #[test]
    fn tearing_down_twice_is_harmless_and_reports_once() {
        let link = ViewLink::new();
        let mut app = ViewerApp::new("test".to_string(), scene_with_a_box(), Some(&link));
        app.begin_teardown();
        assert!(app.summary_printed);
        // A close request followed by autoclose, or the reverse, must not
        // double-report or resurrect any state.
        app.begin_teardown();
        assert!(app.torn_down && app.window.is_none());
    }

    /// The editor's modal intercepts the close *request*; the way out of the
    /// window is still the one teardown path, taken only once the answer is in.
    #[test]
    fn an_editor_with_unsaved_changes_is_asked_before_the_window_is_torn_down() {
        use crate::viewer::editor::tests::{fixture, write_temp};
        use crate::viewer::editor::{CloseDecision, Editor};

        let (_dir, path) = write_temp("app_close", fixture());
        let mut editor = Editor::open(&path).expect("open");
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.42));
        let mut app = ViewerApp::new("test".to_string(), scene_with_a_box(), None).editing(editor);

        // What `WindowEvent::CloseRequested` asks, and what it does with the
        // answer.
        let ready = app.editor.as_mut().expect("an editor").request_close();
        assert!(!ready, "unsaved changes must not close the window");
        assert!(!app.torn_down);

        app.editor
            .as_mut()
            .expect("an editor")
            .decide(CloseDecision::Discard);
        assert!(app.editor.as_ref().expect("an editor").may_close());
        app.begin_teardown();
        assert!(app.torn_down && app.window.is_none());
        // And the file was never written.
        assert_eq!(std::fs::read_to_string(&path).expect("read"), fixture());
    }

    /// The fatal path an unusable frame takes: an editing session's unsaved
    /// document is written beside its file before the window goes, and the error
    /// the process ends on is kept whatever the rescue did.
    #[test]
    fn a_fatal_frame_rescues_the_unsaved_document_before_the_window_goes() {
        use crate::viewer::editor::Editor;

        let (_dir, path) = write_temp("app_rescue", fixture());
        let before = std::fs::read(&path).expect("read");
        let mut editor = Editor::open(&path).expect("open");
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = Some(0.42));
        let recovered = editor.state.recovery_path();
        let mut app = ViewerApp::new("test".to_string(), scene_with_a_box(), None).editing(editor);

        app.fail(anyhow::anyhow!("the GPU rejected 30 consecutive frames"));
        assert!(app.torn_down && app.window.is_none());
        assert!(
            recovered.is_file(),
            "the unsaved document went nowhere: {}",
            recovered.display()
        );
        assert_eq!(
            std::fs::read(&path).expect("read"),
            before,
            "the file being edited is not what a dying window writes to"
        );
        assert!(
            app.editor.as_ref().expect("an editor").state.is_dirty(),
            "a rescue is not a save"
        );

        // The window is still a failure: what was rescued is not a reason to
        // report success.
        let error = app.error.take().expect("the error is kept");
        assert!(error.to_string().contains("rejected"), "{error:#}");
    }

    /// Nothing to lose, nothing written. The clean session and the two windows
    /// that have no document at all take the same fatal path and leave no file
    /// behind them.
    #[test]
    fn a_fatal_frame_with_nothing_unsaved_writes_nothing() {
        use crate::viewer::editor::Editor;

        let (_dir, path) = write_temp("app_rescue_clean", fixture());
        let editor = Editor::open(&path).expect("open");
        let recovered = editor.state.recovery_path();
        let mut app = ViewerApp::new("test".to_string(), scene_with_a_box(), None).editing(editor);
        assert!(app.rescue_unsaved_edits().is_none());
        app.fail(anyhow::anyhow!("a frame that could not be drawn"));
        assert!(
            !recovered.exists(),
            "a session with nothing unsaved has nothing to recover"
        );
        assert!(app.torn_down && app.error.is_some());

        // `view` and `run --view` have no document: the rescue is a no-op rather
        // than a path that has to guess where one would go.
        let mut app = ViewerApp::new("test".to_string(), scene_with_a_box(), None);
        assert!(app.rescue_unsaved_edits().is_none());
        app.fail(anyhow::anyhow!("a frame that could not be drawn"));
        assert!(app.torn_down && app.error.is_some());
    }

    #[test]
    fn the_viewport_width_falls_back_to_the_window_scale_before_the_first_frame() {
        let mut app = ViewerApp::new("test".to_string(), scene_with_a_box(), None);
        // With no window yet the fallback is a scale factor of one.
        assert_eq!(
            app.scene_viewport_width(1280),
            scene_width_for(1280, 1.0, constants::VIEW_PANEL_WIDTH_POINTS)
        );
        // A reported viewport wins, and is never wider than the surface.
        app.scene_width = Some(900);
        assert_eq!(app.scene_viewport_width(1280), 900);
        assert_eq!(app.scene_viewport_width(800), 800);
    }
}
