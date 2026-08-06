//! The editor's side panel: the object tree, the properties of what is
//! selected, the scalar sections, the validation and summary blocks, and the
//! unsaved-changes modal that guards a close and a file switch alike.
//!
//! Every widget that changes something goes through one `Field`, which opens an
//! interaction on the first change and closes it when the widget is let go, so
//! a value dragged through fifty numbers is one undo step and a number typed
//! into a box is another.
//!
//! Every widget that changes something also says what it does on hover, in the
//! documentation's own words. The shared row helpers take that text as a
//! required parameter rather than an optional one, so a row that explains
//! nothing does not compile; the handful of widgets built inline carry theirs at
//! the call site, and the layer switches carry theirs in the layer table.

use crate::config::{
    Axis, BoundaryFidelity, BuildDirection, CsgOpSpec, GrowthConfig, IslandPolicy, LoadSpec,
    LocalVolumeConfig, MaterialConfig, OptimizationConfig, OutputConfig, OverhangConfig,
    ReinforcePolicy, ShapeSpec, SolverBackend, SolverConfig, SymmetryConfig, SymmetryKind,
    TrimPolicy, UpdateScheme, VoidPolicy, WireframeConfig,
};
use crate::constants;
use crate::geometry::Vec3;
use crate::stress::StressSummary;
use crate::viewer::editor::grid;
use crate::viewer::editor::measure::Phase;
use crate::viewer::editor::state::{
    EditorState, LoadKind, NewObject, Selection, ShapeKind, domain_entry, domain_op, label_of,
    set_shape_contained, shape_of,
};
use crate::viewer::editor::{CloseDecision, Editor, Intent};
use crate::viewer::scene::Scene;
use crate::viewer::ui::layer_switches;

/// Wraps the interaction bookkeeping around one widget.
///
/// `changed` opens the interaction, `drag_stopped` or `lost_focus` closes it.
/// The widget's own egui id is what owns it, so two fields on the same row
/// cannot be mistaken for one another.
struct Field<'a> {
    state: &'a mut EditorState,
    touched: bool,
    /// Whether a shape committed through this panel is kept inside the domain.
    containment: bool,
    /// Set when one was, so the panel can say so.
    clamped: bool,
}

impl<'a> Field<'a> {
    fn new(state: &'a mut EditorState, containment: bool) -> Field<'a> {
        Field {
            state,
            touched: false,
            containment,
            clamped: false,
        }
    }

    /// Record what a widget did. Returns whether it changed anything.
    fn apply(&mut self, response: &egui::Response) -> bool {
        let id = response.id.value();
        if response.changed() {
            self.state.begin_edit(id);
            self.touched = true;
        }
        if response.drag_stopped() || response.lost_focus() {
            self.state.end_edit(id);
        }
        response.changed()
    }

    /// True when any widget of this panel changed a value this frame.
    fn touched(&self) -> bool {
        self.touched
    }
}

/// Parse a number the way egui does, and refuse the ones that are not numbers.
///
/// `"nan"`, `"inf"` and `"-inf"` all parse as an `f64`, and a configuration
/// holding one is one growforge refuses to read back - so the field refuses it
/// at the keystroke instead, keeping what was there. The rest of the behaviour
/// is egui's own: whitespace anywhere is ignored, and the typographic minus is
/// accepted as a hyphen.
fn finite_number(text: &str) -> Option<f64> {
    let cleaned: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| if c == '\u{2212}' { '-' } else { c })
        .collect();
    let value: f64 = cleaned.parse().ok()?;
    value.is_finite().then_some(value)
}

/// A drag value for a length in millimetres.
fn length_widget(ui: &mut egui::Ui, value: &mut f64) -> egui::Response {
    ui.add(
        egui::DragValue::new(value)
            .speed(constants::VIEW_EDIT_DRAG_SPEED_MM)
            .max_decimals(constants::VIEW_EDIT_DECIMALS)
            .custom_parser(finite_number),
    )
}

/// A drag value for a dimensionless number, moving by a fraction of itself.
fn number_widget(ui: &mut egui::Ui, value: &mut f64, speed: f64) -> egui::Response {
    ui.add(
        egui::DragValue::new(value)
            .speed(speed)
            .max_decimals(constants::VIEW_EDIT_DECIMALS)
            .custom_parser(finite_number),
    )
}

/// Three length fields on one row, for a point or a vector.
///
/// `tooltip` explains the row, on the name and on all three fields alike.
fn vec3_row(
    ui: &mut egui::Ui,
    field: &mut Field<'_>,
    label: &str,
    value: &mut Vec3,
    tooltip: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(tooltip);
        for component in value.iter_mut() {
            let response = length_widget(ui, component).on_hover_text(tooltip);
            changed |= field.apply(&response);
        }
    });
    changed
}

/// One length field on its own row.
fn length_row(
    ui: &mut egui::Ui,
    field: &mut Field<'_>,
    label: &str,
    value: &mut f64,
    tooltip: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(tooltip);
        let response = length_widget(ui, value).on_hover_text(tooltip);
        changed |= field.apply(&response);
    });
    changed
}

/// One dimensionless field on its own row.
fn number_row(
    ui: &mut egui::Ui,
    field: &mut Field<'_>,
    label: &str,
    value: &mut f64,
    speed: f64,
    tooltip: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(tooltip);
        let response = number_widget(ui, value, speed).on_hover_text(tooltip);
        changed |= field.apply(&response);
    });
    changed
}

/// One whole-number field on its own row.
fn integer_row(
    ui: &mut egui::Ui,
    field: &mut Field<'_>,
    label: &str,
    value: &mut usize,
    tooltip: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(tooltip);
        let response = ui
            .add(egui::DragValue::new(value).speed(1.0))
            .on_hover_text(tooltip);
        changed |= field.apply(&response);
    });
    changed
}

/// A drag value for a relative residual, in scientific notation.
///
/// Not [`number_widget`]: that one rounds to [`constants::VIEW_EDIT_DECIMALS`]
/// places, a floor of 5e-4 that every tolerance a configuration may hold sits
/// below - it would draw 1e-8 as `0` and set it to `0` on the first pixel of a
/// drag. This one moves by a fraction of the value itself, so the pixels buy
/// decades, and carries the range the configuration accepts so a drag or a
/// typed number cannot leave it.
///
/// A value the *file* carries is shown as it is written, out of range or not
/// (`clamp_existing_to_range(false)`): the validation block above the sections
/// is what says a key is wrong, and a panel that quietly rewrote one on the
/// frame it was drawn would be editing a document nobody touched.
fn tolerance_widget(ui: &mut egui::Ui, value: &mut f64) -> egui::Response {
    let speed =
        value.abs().max(constants::CG_TOLERANCE_MIN) * constants::VIEW_EDIT_TOLERANCE_DRAG_FRACTION;
    ui.add(
        egui::DragValue::new(value)
            .speed(speed)
            .range(constants::CG_TOLERANCE_MIN..=constants::CG_TOLERANCE_MAX)
            .clamp_existing_to_range(false)
            .custom_formatter(|value, _| format!("{value:e}"))
            .custom_parser(finite_number),
    )
}

/// An optional value with a checkbox that turns it into the default.
///
/// The default shown in place of the field carries the same explanation as the
/// field would: what the key does is the question either way. `widget` draws the
/// value while there is one and `default_label` spells the default while there
/// is not, so a field shown in scientific notation says its default the same
/// way.
#[allow(clippy::too_many_arguments)]
fn optional_value_row(
    ui: &mut egui::Ui,
    field: &mut Field<'_>,
    label: &str,
    value: &mut Option<f64>,
    default: f64,
    default_label: &str,
    tooltip: &str,
    widget: impl FnOnce(&mut egui::Ui, &mut f64) -> egui::Response,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let mut present = value.is_some();
        let toggle = ui.checkbox(&mut present, label).on_hover_text(tooltip);
        if field.apply(&toggle) {
            *value = present.then_some(default);
            changed = true;
        }
        match value {
            Some(inner) => {
                let response = widget(ui, inner).on_hover_text(tooltip);
                changed |= field.apply(&response);
            }
            None => {
                ui.label(egui::RichText::new(format!("{default_label} (default)")).weak())
                    .on_hover_text(tooltip);
            }
        }
    });
    changed
}

/// The same for the dimensionless fields, which are all of them but one.
fn optional_row(
    ui: &mut egui::Ui,
    field: &mut Field<'_>,
    label: &str,
    value: &mut Option<f64>,
    default: f64,
    speed: f64,
    tooltip: &str,
) -> bool {
    optional_value_row(
        ui,
        field,
        label,
        value,
        default,
        &format!("{default}"),
        tooltip,
        |ui, value| number_widget(ui, value, speed),
    )
}

/// The same for a whole number.
fn optional_integer_row(
    ui: &mut egui::Ui,
    field: &mut Field<'_>,
    label: &str,
    value: &mut Option<usize>,
    default: usize,
    tooltip: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let mut present = value.is_some();
        let toggle = ui.checkbox(&mut present, label).on_hover_text(tooltip);
        if field.apply(&toggle) {
            *value = present.then_some(default);
            changed = true;
        }
        match value {
            Some(inner) => {
                let response = ui
                    .add(egui::DragValue::new(inner).speed(1.0))
                    .on_hover_text(tooltip);
                changed |= field.apply(&response);
            }
            None => {
                ui.label(egui::RichText::new(format!("{default} (default)")).weak())
                    .on_hover_text(tooltip);
            }
        }
    });
    changed
}

/// A dropdown over a fixed set of values.
///
/// The explanation hangs on the closed dropdown rather than on its items, which
/// is where the pointer is when the question is asked: `show_ui` hands back the
/// button's own response beside whatever the popup did.
fn combo<T: PartialEq + Copy>(
    ui: &mut egui::Ui,
    field: &mut Field<'_>,
    label: &str,
    current: &mut T,
    options: &[(T, &str)],
    tooltip: &str,
) -> bool {
    let mut changed = false;
    let selected = options
        .iter()
        .find(|(value, _)| value == current)
        .map(|(_, name)| *name)
        .unwrap_or("");
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(tooltip);
        let inner = egui::ComboBox::from_id_salt(label)
            .selected_text(selected)
            .show_ui(ui, |ui| {
                for (value, name) in options {
                    let response = ui.selectable_value(current, *value, *name);
                    if response.changed() {
                        // A dropdown is chosen in one gesture, so it opens and
                        // closes its own interaction.
                        let id = response.id.value();
                        field.state.begin_edit(id);
                        field.state.end_edit(id);
                        field.touched = true;
                        changed = true;
                    }
                }
            });
        inner.response.on_hover_text(tooltip);
    });
    changed
}

/// The same over a list of names, for the keys that are strings.
///
/// Returns the chosen name when it changed, so the caller can put it wherever
/// the schema keeps it.
fn combo_str(
    ui: &mut egui::Ui,
    field: &mut Field<'_>,
    label: &str,
    current: &str,
    options: &[&str],
    tooltip: &str,
) -> Option<String> {
    let mut chosen: Option<String> = None;
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(tooltip);
        let inner = egui::ComboBox::from_id_salt(label)
            .selected_text(current)
            .show_ui(ui, |ui| {
                for name in options {
                    let response = ui.selectable_label(*name == current, *name);
                    if response.clicked() && *name != current {
                        let id = response.id.value();
                        field.state.begin_edit(id);
                        field.state.end_edit(id);
                        field.touched = true;
                        chosen = Some((*name).to_string());
                    }
                }
            });
        inner.response.on_hover_text(tooltip);
    });
    chosen
}

/// Draw the whole editor panel.
///
/// Returns how far its content laid out past the width the panel declares, in
/// points, which is zero whenever every row fitted. A row that does not fit is
/// not merely clipped: egui measures the frame the content really took, finds
/// it wider than [`constants::VIEW_EDIT_PANEL_WIDTH_POINTS`], pulls the panel
/// rect back to that width by sliding its *left* edge inwards, and paints the
/// panel's separator at the edge it just moved - a full height line over the
/// middle of the panel, on top of everything. This is that overflow, measured
/// inside the frame where it is decided rather than after egui has clamped it
/// away, and it is what the guard test holds at zero for every state the panel
/// can be driven into *without a finished run behind it*.
///
/// The rows that a completed run gates - [`pass_notes`] and [`stress_block`] -
/// are outside that reach, because a headless test has no worker progress to
/// show them. They are single wrapping labels, and a label that wraps cannot
/// overflow horizontally whatever it holds; only a `ui.horizontal` of several
/// widgets can, which is what the two say at their own definitions.
pub fn panel(root: &mut egui::Ui, editor: &mut Editor, scene: &mut Scene, adapter: &str) -> f32 {
    egui::Panel::right("growforge_editor")
        .exact_size(constants::VIEW_EDIT_PANEL_WIDTH_POINTS)
        .resizable(false)
        .show(root, |ui| {
            // Read before anything is added: a widget that overflows expands the
            // region it was added to - `max_rect` as well as `min_rect` - so the
            // budget has to be taken while it still is one.
            let budget = ui.max_rect().right();
            ui.heading(editor.window_title());
            ui.label(
                egui::RichText::new(editor.state.path().display().to_string())
                    .small()
                    .weak(),
            );
            ui.label(egui::RichText::new(adapter).small().weak());
            ui.separator();
            toolbar(ui, editor);
            ui.separator();
            precision(ui, editor);
            ui.separator();
            validation(ui, editor);
            egui::ScrollArea::vertical().show(ui, |ui| {
                let containment = editor.containment();
                let (touched, clamped, placing, delete) = {
                    let mut field = Field::new(&mut editor.state, containment);
                    let placing = tree(ui, &mut field);
                    ui.separator();
                    let delete = properties(ui, &mut field);
                    ui.separator();
                    sections(ui, &mut field);
                    (field.touched(), field.clamped, placing, delete)
                };
                ui.separator();
                summary(ui, &editor.state);
                ui.separator();
                show_block(ui, scene);
                pass_notes(ui, editor);
                stress_summary(ui, editor);
                ui.separator();
                controls(ui);
                if clamped {
                    editor.note_clamped();
                }
                if touched {
                    editor.on_edited();
                }
                // The two the panel hands back rather than applying: both reach
                // past the document into the session, so they are applied once
                // its borrow is over. The delete is the editor's own, so that
                // the button and the `Delete` key do the same thing to a
                // placement in progress.
                if delete {
                    editor.delete_selection();
                }
                if let Some(what) = placing {
                    editor.toggle_placing(what);
                }
            });
            // Measured against what the frame gave the content rather than
            // against the constant, so the margins are egui's own arithmetic
            // and not a second statement of them here.
            (ui.min_rect().right() - budget).max(0.0)
        })
        .inner
}

/// The file, undo, redo and run controls.
fn toolbar(ui: &mut egui::Ui, editor: &mut Editor) {
    let valid = editor.state.is_valid();
    ui.horizontal(|ui| {
        if ui
            .add_enabled(editor.state.is_dirty(), egui::Button::new("save"))
            .clicked()
        {
            editor.save();
        }
        // Beside it, because they are the other two things one does with the
        // file. Both only ask for a dialog: what happens to whatever comes back
        // goes through the same unsaved-changes guard a close does.
        if ui
            .small_button("open")
            .on_hover_text("edit another configuration - unsaved changes are asked about first")
            .clicked()
        {
            editor.ask_to_open();
        }
        if ui
            .small_button("new")
            .on_hover_text(
                "scaffold a starter configuration at a name that is not there yet and edit that; \
                 an existing file is never overwritten",
            )
            .clicked()
        {
            editor.ask_for_new();
        }
        if ui
            .add_enabled(editor.state.undo_depth() > 0, egui::Button::new("undo"))
            .clicked()
        {
            editor.undo();
        }
        if ui
            .add_enabled(editor.state.redo_depth() > 0, egui::Button::new("redo"))
            .clicked()
        {
            editor.redo();
        }
    });
    ui.horizontal(|ui| {
        // The switch itself stays live while the configuration is broken -
        // switching it off is exactly what someone in the middle of a large
        // edit wants. What is gated is the run: nothing starts until the
        // configuration builds.
        let mut auto = editor.auto_regrow();
        if ui.checkbox(&mut auto, "auto-regrow").changed() {
            editor.set_auto_regrow(auto);
        }
        let running = editor.is_running();
        if ui
            .add_enabled(valid && !editor.is_writing(), egui::Button::new("run full"))
            .clicked()
        {
            editor.start_full_run();
        }
        // Whatever is running - the preview an edit started, the full pipeline,
        // an stl generation - one button ends it, and ending it writes nothing.
        if ui.add_enabled(running, egui::Button::new("stop")).clicked() {
            editor.stop_run();
        }
    });
    // On a row of its own: the label is a long one, and the toolbar sits above
    // the panel's scroll area, where a row that overflows is a row that is cut
    // off rather than one that wraps.
    ui.horizontal(|ui| {
        // Enabled by there being a design on screen and nothing already writing,
        // and by nothing else: the pair that would be exported carries its own
        // problem, so a configuration edited - or broken - since the run does not
        // take the design away.
        if ui
            .add_enabled(editor.can_generate_stl(), egui::Button::new("generate stl"))
            .on_hover_text(
                "voids, stress and the STL from the design on screen, whichever run produced it \
                 and however that run ended",
            )
            .on_disabled_hover_text("nothing has been run yet, or a run is writing already")
            .clicked()
        {
            editor.generate_stl();
        }
    });
    ui.label(
        egui::RichText::new(
            "auto-regrow previews are capped, coarsened and never written; \"run full\" writes \
             the STL, and \"generate stl\" writes one from the design on screen - a stopped run \
             writes nothing on its own",
        )
        .small()
        .weak(),
    );
    // What the viewport is waiting for, while it is waiting for it: a placement
    // takes the clicks that would otherwise select, so the panel has to say so
    // for as long as that is true.
    if let Some(hint) = editor.placement_hint() {
        ui.label(
            egui::RichText::new(hint)
                .small()
                .color(ui.visuals().warn_fg_color),
        );
    }
    if let Some(status) = editor.status_line() {
        ui.label(egui::RichText::new(status).small());
    }
}

/// How precisely the viewport drags land: the snap increment, and whether
/// objects are kept inside the domain.
///
/// Both belong to the session rather than to the file - nothing here is ever
/// written to the configuration - which is why they sit above the document's
/// own sections rather than among them.
fn precision(ui: &mut egui::Ui, editor: &mut Editor) {
    // One explanation over the whole row: the dropdown and the box beside it set
    // the same increment, one from the usual list and one from anything at all.
    let snap = "the increment drags land on: a position along an axis, a dimension being resized, \
                a radius. Pick one of the usual ones, off, or type any increment at all beside \
                it. Holding Alt frees a drag from it entirely";
    ui.horizontal(|ui| {
        ui.label("snap mm").on_hover_text(snap);
        let mut increment = editor.snap().millimetres;
        let selected = if increment >= constants::VIEW_EDIT_SNAP_MIN_MM {
            format!("{increment}")
        } else {
            "off".to_string()
        };
        let inner = egui::ComboBox::from_id_salt("growforge_editor_snap")
            .selected_text(selected)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut increment, 0.0, "off");
                for choice in constants::VIEW_EDIT_SNAP_CHOICES_MM {
                    ui.selectable_value(&mut increment, choice, format!("{choice}"));
                }
            });
        inner.response.on_hover_text(snap);
        // And anything else, typed.
        ui.add(
            egui::DragValue::new(&mut increment)
                .speed(constants::VIEW_EDIT_DRAG_SPEED_MM)
                .range(0.0..=f64::MAX)
                .max_decimals(constants::VIEW_EDIT_DECIMALS)
                .custom_parser(finite_number),
        )
        .on_hover_text(snap);
        if increment != editor.snap().millimetres {
            editor.snap_mut().millimetres = increment.max(0.0);
        }
    });
    ui.horizontal(|ui| {
        let mut on = editor.containment();
        if ui
            .checkbox(&mut on, "keep inside domain")
            .on_hover_text(
                "clamp every commit of a keepout, keepin, support or load region so that its \
                 bounding box stays inside the domain's. Only the position is ever changed - \
                 nothing is resized to fit. Switch it off to place something deliberately \
                 outside, such as a keepin pad that overhangs the design space",
            )
            .changed()
        {
            editor.set_containment(on);
        }
    });
    ui.label(
        egui::RichText::new(format!(
            "drags land on the increment and rotations on {} deg; hold Alt for a free drag",
            constants::VIEW_EDIT_SNAP_DEGREES
        ))
        .small()
        .weak(),
    );
    // Said only when the grid is not ruled at the increment above, so the plane
    // on screen is never quietly a different one from the one drags land on.
    if let Some(note) = editor.floor_grid().and_then(grid::FloorGrid::note) {
        ui.label(
            egui::RichText::new(note)
                .small()
                .color(ui.visuals().warn_fg_color),
        );
    }
    if let Some(note) = editor.containment_note() {
        ui.label(
            egui::RichText::new(note)
                .small()
                .color(ui.visuals().warn_fg_color),
        );
    }
}

/// The validation block: what is wrong, or what the run is doing.
fn validation(ui: &mut egui::Ui, editor: &Editor) {
    match editor.state.error() {
        Some(error) => {
            ui.label(
                egui::RichText::new("configuration is not runnable")
                    .strong()
                    .color(ui.visuals().error_fg_color),
            );
            ui.label(egui::RichText::new(error).small());
        }
        None => {
            ui.label(egui::RichText::new("configuration is valid").strong());
            for warning in editor.state.warnings() {
                ui.label(
                    egui::RichText::new(format!("warning: {warning}"))
                        .small()
                        .color(ui.visuals().warn_fg_color),
                );
            }
        }
    }
    if let Some(run) = editor.run_line() {
        ui.monospace(run);
    }
    // What the density surface on screen is, so a coarsened, capped preview is
    // never mistaken for the design a full run would produce.
    ui.monospace(
        editor
            .frame_label()
            .unwrap_or_else(|| "no surface  yet".to_string()),
    );
    // Why the run ended, in the engine's own words: the same note the viewer's
    // run panel shows. A run that stopped because the problem will not settle
    // says so here rather than only in the console the editor was started from.
    if let Some(note) = editor.run_note() {
        ui.label(egui::RichText::new(note).small());
    }
}

/// One list of shape objects in the tree: its heading, how many it holds, and
/// how to address the entry at an index.
type ShapeList = (&'static str, usize, fn(usize) -> Selection);

/// The object tree, with an add menu per list, under a header of its own.
///
/// Returns the object an add row asked to *place* rather than to add, which is
/// the tube button and nothing else: a tube is two clicked points, so its
/// button opens the placement mode instead of dropping a default one at the
/// centre of the domain. The mode is the session's, not the document's, so it
/// is handed back to the caller rather than applied here.
fn tree(ui: &mut egui::Ui, field: &mut Field<'_>) -> Option<NewObject> {
    let mut placing: Option<NewObject> = None;
    egui::CollapsingHeader::new(constants::VIEW_EDIT_BLOCK_OBJECTS)
        .default_open(true)
        .show(ui, |ui| placing = object_lists(ui, field));
    placing
}

/// The five lists themselves, drawn into the body the tree's header opens.
///
/// Its own function because a collapsing header is identified by the ui it is
/// drawn in: the lists laid out here can therefore be laid out anywhere else -
/// a test, in particular - with their ids known, which is what the salts below
/// exist to keep still.
fn object_lists(ui: &mut egui::Ui, field: &mut Field<'_>) -> Option<NewObject> {
    let selection = field.state.selection();

    let mut clicked: Option<Selection> = None;
    let mut added: Option<NewObject> = None;
    let mut placing: Option<NewObject> = None;

    let lists: [ShapeList; 4] = [
        (
            "domain",
            field.state.config().domain.len(),
            Selection::Domain,
        ),
        (
            "keepout",
            field.state.config().keepout.len(),
            Selection::Keepout,
        ),
        (
            "keepin",
            field.state.config().keepin.len(),
            Selection::Keepin,
        ),
        (
            "supports",
            field.state.config().supports.len(),
            Selection::Support,
        ),
    ];
    for (name, count, make) in lists {
        // Salted with the name alone, because the label carries the count and
        // egui identifies a header by its label text unless it is told
        // otherwise: without this, adding or deleting an object would rename
        // the header, and a list the user had opened would shut itself the
        // moment its contents changed.
        egui::CollapsingHeader::new(format!("{name} ({count})"))
            .id_salt(name)
            .default_open(false)
            .show(ui, |ui| {
                for index in 0..count {
                    let entry = make(index);
                    let label = label_of(field.state.config(), entry);
                    if ui
                        .selectable_label(selection == Some(entry), label)
                        .clicked()
                    {
                        clicked = Some(entry);
                    }
                }
                // Wrapped rather than straight: one button per shape kind is a
                // row that grows with the schema, and a row that outgrows the
                // panel's width does not merely spill - it widens the whole
                // panel frame, which egui then clamps back by sliding the
                // panel's left edge inwards and painting the separator there,
                // over the middle of the contents. Wrapping keeps every kind
                // reachable and the row inside the width the panel declares.
                ui.horizontal_wrapped(|ui| {
                    ui.label("add");
                    for kind in ShapeKind::ALL {
                        if !ui.small_button(kind.label()).clicked() {
                            continue;
                        }
                        let what = match name {
                            "domain" => NewObject::Domain(kind, CsgOpSpec::Add),
                            "keepout" => NewObject::Keepout(kind),
                            "keepin" => NewObject::Keepin(kind),
                            _ => NewObject::Support(kind),
                        };
                        // A tube is drawn rather than dropped: its two ends are
                        // clicked in the viewport, and this row's button is what
                        // opens the mode that takes them.
                        match kind {
                            ShapeKind::Tube => placing = Some(what),
                            _ => added = Some(what),
                        }
                    }
                });
            });
    }

    let cases = field.state.config().loadcases.len();
    egui::CollapsingHeader::new(format!("{} ({cases})", constants::VIEW_EDIT_LOAD_CASE_LIST))
        .id_salt(constants::VIEW_EDIT_LOAD_CASE_LIST)
        .default_open(false)
        .show(ui, |ui| {
            for case in 0..cases {
                let entry = Selection::LoadCase(case);
                let label = label_of(field.state.config(), entry);
                if ui
                    .selectable_label(selection == Some(entry), label)
                    .clicked()
                {
                    clicked = Some(entry);
                }
                ui.indent(("loads", case), |ui| {
                    let loads = field.state.config().loadcases[case].loads.len();
                    for load in 0..loads {
                        let entry = Selection::Load { case, load };
                        let label = label_of(field.state.config(), entry);
                        if ui
                            .selectable_label(selection == Some(entry), label)
                            .clicked()
                        {
                            clicked = Some(entry);
                        }
                    }
                    ui.horizontal(|ui| {
                        ui.label("add");
                        for kind in LoadKind::ALL {
                            if ui.small_button(kind.label()).clicked() {
                                added = Some(NewObject::Load(case, kind));
                            }
                        }
                    });
                });
            }
            if ui.small_button("add load case").clicked() {
                added = Some(NewObject::LoadCase);
            }
        });

    if let Some(entry) = clicked {
        field.state.select(Some(entry));
    }
    if let Some(what) = added {
        field.state.add(what);
        field.touched = true;
    }
    placing
}

/// The properties of whatever is selected, under a header of its own.
///
/// Returns whether the delete button was pressed. The delete itself is the
/// caller's to apply, because it is the session's business as well as the
/// document's - it cancels a placement in progress, which is why the button and
/// the `Delete` key both end up in [`Editor::delete_selection`] rather than in
/// two places that have to agree.
///
/// A closed block has no button to press, which is what the missing body says:
/// nothing was clicked, so nothing is deleted.
fn properties(ui: &mut egui::Ui, field: &mut Field<'_>) -> bool {
    egui::CollapsingHeader::new(constants::VIEW_EDIT_BLOCK_PROPERTIES)
        .default_open(true)
        .show(ui, |ui| properties_body(ui, field))
        .body_returned
        .unwrap_or(false)
}

/// The rows of that block, drawn into the body its header opens.
fn properties_body(ui: &mut egui::Ui, field: &mut Field<'_>) -> bool {
    let Some(selection) = field.state.selection() else {
        ui.label("nothing selected; click an object in the viewport or the tree");
        return false;
    };
    // The first row of the body rather than the heading that used to carry it:
    // the delete belongs to the object whose fields are under it, and a control
    // that removes something has to sit with those fields rather than on a
    // header that is there whether or not they are.
    let delete = ui.small_button("delete").clicked();
    // The fields of an object that is going are not drawn, exactly as they were
    // not drawn when the delete happened here.
    if delete {
        return true;
    }
    ui.label(label_of(field.state.config(), selection));

    if let Selection::Domain(index) = selection {
        let mut op = domain_op(&field.state.config().domain[index]);
        if combo(
            ui,
            field,
            "op",
            &mut op,
            &[
                (CsgOpSpec::Add, CsgOpSpec::Add.label()),
                (CsgOpSpec::Subtract, CsgOpSpec::Subtract.label()),
            ],
            "how this entry joins the design space: the domain is the union of the adds minus the \
             subtracts, taken in the order the list holds them",
        ) {
            let spec = field.state.config().domain[index].shape_spec();
            field.state.config_mut().domain[index] = domain_entry(spec, op);
        }
    }

    if let Selection::Support(index) = selection {
        support_directions(ui, field, index);
    }

    if let Selection::LoadCase(index) = selection {
        load_case_fields(ui, field, index);
    }

    if let Selection::Load { case, load } = selection {
        load_fields(ui, field, case, load);
    }

    if let Some(mut shape) = shape_of(field.state.config(), selection)
        && shape_fields(ui, field, &mut shape)
    {
        // The same commit path a gizmo drag takes, containment and all: a
        // number typed into the panel is as much a placement as a drag is.
        let containment = field.containment;
        if set_shape_contained(field.state.config_mut(), selection, shape, containment) {
            field.clamped = true;
        }
    }
    false
}

/// The numeric fields of one shape.
fn shape_fields(ui: &mut egui::Ui, field: &mut Field<'_>, shape: &mut ShapeSpec) -> bool {
    match shape {
        ShapeSpec::Box {
            min,
            max,
            rotation_deg,
        } => {
            let mut changed = vec3_row(
                ui,
                field,
                "min",
                min,
                "the corner of the box at the low end of x, y and z, in mm",
            );
            changed |= vec3_row(
                ui,
                field,
                "max",
                max,
                "the corner of the box at the high end of x, y and z, in mm",
            );
            changed |= rotation_row(ui, field, rotation_deg);
            changed
        }
        ShapeSpec::Cylinder { p1, p2, radius } => {
            let mut changed = vec3_row(
                ui,
                field,
                "p1 ",
                p1,
                "the centre of one end cap, in mm. The cylinder is everything within the radius \
                 of the line between the two",
            );
            changed |= vec3_row(
                ui,
                field,
                "p2 ",
                p2,
                "the centre of the other end cap, in mm. The cylinder is capped flat at both",
            );
            changed |= length_row(
                ui,
                field,
                "radius",
                radius,
                "how far the cylinder reaches from its axis, in mm",
            );
            changed
        }
        ShapeSpec::Sphere { center, radius } => {
            let mut changed = vec3_row(ui, field, "centre", center, "the centre point, in mm");
            changed |= length_row(
                ui,
                field,
                "radius",
                radius,
                "how far the sphere reaches from its centre, in mm",
            );
            changed
        }
        ShapeSpec::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } => {
            let mut changed = vec3_row(ui, field, "centre", center, "the centre point, in mm");
            changed |= vec3_row(
                ui,
                field,
                "radii ",
                radii,
                "the three semi-axes in mm, along the shape's own x, y and z. All three must be \
                 positive",
            );
            changed |= rotation_row(ui, field, rotation_deg);
            changed
        }
        ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        } => {
            let mut changed = vec3_row(
                ui,
                field,
                "p1 ",
                p1,
                "one end of the tube's centre line, in mm. The ends are rounded, not capped flat",
            );
            changed |= vec3_row(
                ui,
                field,
                "p2 ",
                p2,
                "the other end of the tube's centre line, in mm",
            );
            // The bend rows exist only while there is a bend: a straight tube
            // carries no key, and the middle handle in the viewport is what
            // gives it one. Straightening takes it away again.
            let mut straighten = false;
            if let Some(point) = bend.as_mut() {
                changed |= vec3_row(
                    ui,
                    field,
                    "bend ",
                    point,
                    "the point the tube curves through, in mm: with it the centre line is the \
                     circular arc through the three points instead of the straight line between \
                     two",
                );
                ui.horizontal(|ui| {
                    straighten = small_button(
                        ui,
                        field,
                        "straighten",
                        "drop the bend and make the tube straight again, which takes the key out \
                         of the file",
                    )
                });
            }
            if straighten {
                *bend = None;
                changed = true;
            }
            changed |= length_row(
                ui,
                field,
                "radius",
                radius,
                "how far the tube reaches from its centre line, in mm",
            );
            changed
        }
        ShapeSpec::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => {
            let mut changed = vec3_row(
                ui,
                field,
                "p1 ",
                p1,
                "the centre of the wide cap, in mm. The cone is capped flat at both ends, like a \
                 cylinder, with a radius of its own at each",
            );
            changed |= vec3_row(ui, field, "p2 ", p2, "the centre of the other cap, in mm");
            changed |= length_row(
                ui,
                field,
                "radius1",
                radius1,
                "how far the cone reaches from its axis at p1, in mm. Must be positive: it is the \
                 end the degeneracy guard measures",
            );
            changed |= length_row(
                ui,
                field,
                "radius2",
                radius2,
                "how far it reaches from its axis at p2, in mm. Zero is legal and is the apex of a \
                 true cone; equal radii are the cylinder",
            );
            changed
        }
        ShapeSpec::Triangle { a, b, c, thickness } => {
            let mut changed = vec3_row(
                ui,
                field,
                "a ",
                a,
                "the first corner of the triangle, in mm. The three of them fix the plane the \
                 prism is extruded about, so they point it as well as shape it - there is no \
                 rotation key",
            );
            changed |= vec3_row(ui, field, "b ", b, "the second corner, in mm");
            changed |= vec3_row(
                ui,
                field,
                "c ",
                c,
                "the third corner, in mm. Three points on one line have no area, and are refused \
                 rather than taken as something else",
            );
            changed |= length_row(
                ui,
                field,
                "thickness",
                thickness,
                "how deep the prism is, in mm: half of it either side of the plane through the \
                 three corners",
            );
            changed
        }
    }
}

/// A small button that counts as one edit of the document.
///
/// A button has no value to change, so it reports a click rather than a
/// change: the interaction is opened and closed in the same gesture, which is
/// what a dropdown does for the same reason.
fn small_button(ui: &mut egui::Ui, field: &mut Field<'_>, label: &str, tooltip: &str) -> bool {
    let response = ui.small_button(label).on_hover_text(tooltip);
    if !response.clicked() {
        return false;
    }
    let id = response.id.value();
    field.state.begin_edit(id);
    field.state.end_edit(id);
    field.touched = true;
    true
}

/// The rotation of a box or an ellipsoid: an optional key, so a checkbox
/// decides whether the file carries one at all, and three degree fields when it
/// does.
///
/// The angles turn the shape about its own centre, about the fixed world axes
/// in x, y, z order. What is shown is normalized into a half turn either way,
/// which is how an angle reads; what is stored is whatever the user set, so a
/// file that says 720 goes on saying it until someone changes it.
fn rotation_row(ui: &mut egui::Ui, field: &mut Field<'_>, value: &mut Option<Vec3>) -> bool {
    let rotation = "turn the shape about its own centre, in degrees about the fixed world x, then \
                    y, then z axes. Unticked means the shape is axis aligned and the file carries \
                    no rotation at all";
    let mut changed = false;
    ui.horizontal(|ui| {
        let mut present = value.is_some();
        let toggle = ui
            .checkbox(&mut present, "rotation deg")
            .on_hover_text(rotation);
        if field.apply(&toggle) {
            *value = present.then_some([0.0; 3]);
            changed = true;
        }
        match value {
            Some(angles) => {
                for component in angles.iter_mut() {
                    let response = ui
                        .add(
                            egui::DragValue::new(component)
                                .speed(constants::VIEW_EDIT_ROTATION_DRAG_SPEED_DEG)
                                .max_decimals(constants::VIEW_EDIT_DECIMALS)
                                .custom_parser(finite_number)
                                .custom_formatter(|value, _| {
                                    format!(
                                        "{:.*}",
                                        constants::VIEW_EDIT_CALLOUT_DECIMALS,
                                        crate::geometry::normalize_degrees(value)
                                    )
                                }),
                        )
                        .on_hover_text(rotation);
                    changed |= field.apply(&response);
                }
            }
            None => {
                ui.label(egui::RichText::new("axis aligned").weak())
                    .on_hover_text(rotation);
            }
        }
    });
    changed
}

/// The constrained axes of a support.
fn support_directions(ui: &mut egui::Ui, field: &mut Field<'_>, index: usize) {
    let fixed = "the directions the nodes inside this region are held in. All three ticked is a \
                 fully fixed support and is what an absent key means; untick one to let the \
                 region slide along that axis";
    let current = field.state.config().supports[index].directions.clone();
    let mut axes: Vec<Axis> = current.clone().unwrap_or_else(|| Axis::ALL.to_vec());
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("fixed").on_hover_text(fixed);
        for axis in Axis::ALL {
            let mut on = axes.contains(&axis);
            let response = ui.checkbox(&mut on, axis.label()).on_hover_text(fixed);
            if field.apply(&response) {
                if on {
                    axes.push(axis);
                    axes.sort_by_key(|a| a.dof());
                } else {
                    axes.retain(|a| *a != axis);
                }
                changed = true;
            }
        }
    });
    if changed {
        // All three axes is what an absent key means, so that is how it is
        // written back: an editor should not add noise to a file.
        field.state.config_mut().supports[index].directions =
            (axes.len() < Axis::ALL.len()).then_some(axes);
    }
}

/// The name and weight of a load case.
fn load_case_fields(ui: &mut egui::Ui, field: &mut Field<'_>, index: usize) {
    let name_help = "what this case is called in the reports, the stress table and the panel's \
                     own stress block";
    ui.horizontal(|ui| {
        ui.label("name").on_hover_text(name_help);
        let mut name = field.state.config().loadcases[index].name.clone();
        let response = ui.text_edit_singleline(&mut name).on_hover_text(name_help);
        if field.apply(&response) {
            field.state.config_mut().loadcases[index].name = name;
        }
    });
    let mut weight = field.state.config().loadcases[index].weight;
    if optional_row(
        ui,
        field,
        "weight",
        &mut weight,
        constants::DEFAULT_LOADCASE_WEIGHT,
        constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
        "how much this case counts against the others: the objective is the sum of every case's \
         compliance times its weight",
    ) {
        field.state.config_mut().loadcases[index].weight = weight;
    }
}

/// The fields of one load, beyond its region.
fn load_fields(ui: &mut egui::Ui, field: &mut Field<'_>, case: usize, load: usize) {
    let mut spec = field.state.config().loadcases[case].loads[load].clone();
    let mut changed = false;
    match &mut spec {
        LoadSpec::Force { vector, .. } => {
            changed |= vec3_row(
                ui,
                field,
                "force N",
                vector,
                "the total force in newtons, spread evenly over the nodes this load's region \
                 selects",
            );
        }
        LoadSpec::Torque {
            axis_point,
            axis_dir,
            magnitude_nmm,
            ..
        } => {
            changed |= vec3_row(
                ui,
                field,
                "axis point",
                axis_point,
                "a point the torque's axis passes through, in mm",
            );
            changed |= vec3_row(
                ui,
                field,
                "axis dir",
                axis_dir,
                "the direction of the torque's axis. The nodal forces are tangential to it, scale \
                 with the distance from it, and cancel as a net force",
            );
            changed |= length_row(
                ui,
                field,
                "N*mm",
                magnitude_nmm,
                "the total torque about the axis, in newton-millimetres. It follows the right \
                 hand rule; a negative value turns the other way",
            );
        }
        LoadSpec::Gravity { direction, g_mm_s2 } => {
            let pull = "which way gravity pulls; the vector is normalized on use, so only its \
                        direction matters. Unticked leaves the key out and the pull is straight \
                        down the z axis";
            let mut present = direction.is_some();
            let toggle = ui.checkbox(&mut present, "direction").on_hover_text(pull);
            if field.apply(&toggle) {
                *direction = present.then_some(constants::DEFAULT_GRAVITY_DIRECTION);
                changed = true;
            }
            if let Some(direction) = direction {
                changed |= vec3_row(ui, field, "", direction, pull);
            }
            let mut value = *g_mm_s2;
            if optional_row(
                ui,
                field,
                "g mm/s2",
                &mut value,
                constants::DEFAULT_GRAVITY_MM_S2,
                constants::VIEW_EDIT_DRAG_SPEED_MM,
                "gravitational acceleration in mm/s^2. The self weight of an element is its \
                 density times its volume times this, so it is what scales the whole load",
            ) {
                *g_mm_s2 = value;
                changed = true;
            }
        }
    }
    if changed {
        field.state.config_mut().loadcases[case].loads[load] = spec;
    }
}

/// The scalar sections: everything that is not an object.
fn sections(ui: &mut egui::Ui, field: &mut Field<'_>) {
    engine_section(ui, field);
    resolution_section(ui, field);
    material_section(ui, field);
    optimization_section(ui, field);
    growth_section(ui, field);
    output_section(ui, field);
}

/// A scalar section of the panel, for the button that puts one back to its
/// defaults.
///
/// The six sections that have defaults to go back to. The objects - the shapes,
/// the supports, the loads - deliberately have no such button: geometry is what
/// the user drew, and there is no default position for a load to return to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    /// `engine`, `[solver]` and the `[growth]` table the engine decides the
    /// legality of.
    Engine,
    /// `[resolution]`.
    Resolution,
    /// `[material]`.
    Material,
    /// `[optimization]` and its three sub-tables.
    Optimization,
    /// `[growth]`, shown under that engine alone.
    Growth,
    /// `[output]`.
    Output,
}

impl Section {
    /// Put this section's keys back to what a configuration that never
    /// mentioned them would run at.
    ///
    /// `voxel_mm` is the voxel size the configuration is solved on, which one
    /// key needs: `[optimization] min_feature_mm` is a length rather than a
    /// factor, so it has no fixed default and comes back to the smallest
    /// feature the density filter can resolve on *this* grid. Without a voxel
    /// size to measure that against it is left where it is, which is the only
    /// honest answer.
    ///
    /// Every coupling the interactive controls honour is honoured here, because
    /// a reset has to land a configuration that still builds: leaving the growth
    /// engine takes the `[growth]` table with it exactly as the engine combo
    /// does, and the resolution goes on carrying exactly one of its two keys.
    ///
    /// A table whose keys are all optional is emptied rather than removed, which
    /// is the rule [`write_solver_tolerance`] follows and for its reason: an
    /// empty table resolves to precisely what no table at all does, and removing
    /// it would take a comment block the file may have had with no key left to
    /// hang on.
    fn reset(self, config: &mut crate::config::Config, voxel_mm: Option<f64>) {
        match self {
            Section::Engine => {
                config.engine = None;
                config.project.engine = None;
                // A `[growth]` table is only legal under that engine, and the
                // default engine is not it.
                config.growth = None;
                if config.solver.is_some() {
                    config.solver = Some(SolverConfig {
                        backend: None,
                        tolerance: None,
                    });
                }
            }
            Section::Resolution => {
                config.resolution.voxel_size_mm = Some(constants::VIEW_EDIT_DEFAULT_VOXEL_MM);
                config.resolution.target_cells = None;
            }
            Section::Material => {
                config.material = MaterialConfig {
                    preset: Some(constants::MATERIAL_PRESETS[0].key.to_string()),
                    youngs_modulus_mpa: None,
                    poisson_ratio: None,
                    density_g_cm3: None,
                    yield_strength_mpa: None,
                };
            }
            Section::Optimization => {
                // The mass target is design intent and there is no default to
                // invent for it: how heavy the part may be is what the user
                // came to decide. It keeps its value, and the button says so.
                let mass_fraction = config.optimization.mass_fraction;
                let min_feature_mm = voxel_mm.map_or(config.optimization.min_feature_mm, |h| {
                    constants::VIEW_EDIT_RESET_MIN_FEATURE_VOXELS * h
                });
                config.optimization = OptimizationConfig {
                    mass_fraction,
                    min_feature_mm,
                    penalty: None,
                    max_iterations: None,
                    convergence_tol: None,
                    update: None,
                    overhang: None,
                    wireframe: None,
                    local_volume: None,
                };
            }
            Section::Growth => {
                if config.growth.is_some() {
                    config.growth = Some(GrowthConfig {
                        seed: None,
                        attractor_per_cm3: None,
                        attraction_radius_mm: None,
                        kill_radius_mm: None,
                        step_mm: None,
                        murray_exponent: None,
                        max_radius_mm: None,
                        max_steps: None,
                        prune: None,
                        symmetry: None,
                    });
                }
            }
            Section::Output => {
                // The two destinations, carried across rather than reset:
                // where a run writes is a decision about the project and not
                // about the geometry, which is the rule this section draws both
                // paths under and it does not distinguish them. A reset puts
                // properties back; it does not move deliverables. `stl_path` is
                // the required key of the whole schema besides.
                let stl_path = config.output.stl_path.clone();
                let stress_json = config.output.stress_json.clone();
                config.output = OutputConfig {
                    stl_path,
                    iso_level: None,
                    smoothing_iterations: None,
                    supersample: None,
                    voids: None,
                    islands: None,
                    boundaries: None,
                    trim: None,
                    trim_stress_fraction: None,
                    reinforce: None,
                    reinforce_thickness_mm: None,
                    stress_json,
                };
            }
        }
    }

    /// What the button says on hover: what it puts back, what goes off with it,
    /// and that undoing it is one step.
    ///
    /// Required rather than optional, like every other tooltip in this panel -
    /// and here it carries the two things a user cannot see from the row: which
    /// keys a reset reaches past its own section, and which it deliberately
    /// leaves alone.
    fn tooltip(self) -> String {
        match self {
            Section::Engine => format!(
                "put the engine section back to its defaults: the {} engine, and the solver \
                 backend and tolerance a run picks for itself. Leaving the growth engine takes \
                 the [growth] table with it, exactly as switching engine by hand does, so what a \
                 reset lands on still builds. One click is one undo step",
                constants::DEFAULT_ENGINE
            ),
            Section::Resolution => format!(
                "put the resolution back to its default: a voxel size of {} mm, and no cell \
                 target. A configuration carries exactly one of the two, so the reset writes one \
                 and takes the other out. One click is one undo step",
                constants::VIEW_EDIT_DEFAULT_VOXEL_MM
            ),
            Section::Material => format!(
                "put the material back to the {} preset and drop any custom properties with it: a \
                 configuration carries a preset or four values of your own, never both. One click \
                 is one undo step",
                constants::MATERIAL_PRESETS[0].key
            ),
            Section::Optimization => "put the optimization controls back to their defaults, and \
                                      switch the overhang constraint, the guide wireframe and the \
                                      local volume cap off with them - a sub-table that is not \
                                      there is a feature that is not running. Two keys are not \
                                      touched that way: mass fraction is yours and keeps its \
                                      value, and min feature returns to the smallest the filter \
                                      can resolve at this voxel size. One click is one undo step"
                .to_string(),
            Section::Growth => "put every [growth] key back to its default, symmetry included, so \
                                the table resolves to exactly what the engine would pick on its \
                                own and the four lengths follow min feature again. One click is \
                                one undo step"
                .to_string(),
            Section::Output => {
                "put every [output] control back to its default: the iso level, the \
                                smoothing and the supersampling, and the void, island, boundary, \
                                trim and reinforcement policies, which go back to warn, cull, \
                                exact and off. Both paths stay as they are - the stl and the \
                                stress report are where this project writes, which is not a \
                                property of the part. One click is one undo step"
                    .to_string()
            }
        }
    }
}

/// The voxel size the configuration on screen is really solved on, when there
/// is one to be had.
///
/// The key itself when it names one, which is never stale whatever the last
/// build did, and otherwise the grid the last successful build derived from
/// `target_cells`, which is the very figure the run's resolution warning is
/// measured against. A cell target that has never built has no voxel size at
/// all yet, and says so rather than inventing one.
fn effective_voxel_mm(state: &EditorState) -> Option<f64> {
    state
        .config()
        .resolution
        .voxel_size_mm
        .or_else(|| state.problem().map(|problem| problem.grid.h))
}

/// Whether `section` already holds exactly what a reset would leave it holding.
///
/// The predicate is the reset itself, applied to a copy: a section is at its
/// defaults exactly when resetting it would change nothing. There is no second
/// statement of what those defaults are for the first to drift from, and a
/// section reached through the button is the same section this greys out.
fn at_defaults(state: &EditorState, section: Section) -> bool {
    let mut probe = state.config().clone();
    section.reset(&mut probe, effective_voxel_mm(state));
    &probe == state.config()
}

/// Everything one click of a section's reset button does.
///
/// The interaction is opened and closed in the same gesture, as [`small_button`]
/// does it and for the same reason - a button has no value to drag - so the
/// whole section going back to its defaults is one undo step, and one undo puts
/// every key of it back exactly as it was.
fn reset_section(state: &mut EditorState, section: Section, widget: u64) {
    let voxel_mm = effective_voxel_mm(state);
    state.begin_edit(widget);
    section.reset(state.config_mut(), voxel_mm);
    state.end_edit(widget);
}

/// The reset button of one section, on a row of its own inside its header.
///
/// Disabled rather than hidden while there is nothing to put back: the row stays
/// where it was, and says why it is not offering anything.
fn reset_row(ui: &mut egui::Ui, field: &mut Field<'_>, section: Section) {
    ui.horizontal(|ui| {
        let enabled = !at_defaults(field.state, section);
        let response = ui
            .add_enabled(
                enabled,
                egui::Button::new(constants::VIEW_EDIT_RESET_LABEL).small(),
            )
            .on_hover_text(section.tooltip())
            .on_disabled_hover_text(constants::VIEW_EDIT_RESET_AT_DEFAULTS_HINT);
        if response.clicked() {
            reset_section(field.state, section, response.id.value());
            field.touched = true;
        }
    });
}

/// Engine choice and, with it, whether a `[growth]` table is legal.
fn engine_section(ui: &mut egui::Ui, field: &mut Field<'_>) {
    egui::CollapsingHeader::new("engine")
        .default_open(true)
        .show(ui, |ui| {
            reset_row(ui, field, Section::Engine);
            let current = field
                .state
                .config()
                .engine_name()
                .unwrap_or(constants::DEFAULT_ENGINE)
                .to_string();
            let engines = crate::engine::available();
            if let Some(selected) = combo_str(
                ui,
                field,
                "engine",
                &current,
                &engines,
                "which engine builds the part. simp optimizes a density field and is the default; \
                 growth routes every load to a support and grows a branching skeleton in a \
                 fraction of the time, and is a heuristic - read the stress report before you \
                 print one",
            ) {
                let config = field.state.config_mut();
                // The key is written where the file already keeps it.
                if config.project.engine.is_some() {
                    config.project.engine = Some(selected.clone());
                } else {
                    config.engine = Some(selected.clone());
                }
                if selected != constants::GROWTH_ENGINE {
                    // A `[growth]` table is only legal with that engine.
                    config.growth = None;
                }
            }
            let mut chosen = shown_backend(field.state.config());
            if combo(
                ui,
                field,
                "solver",
                &mut chosen,
                &[
                    (SolverBackend::Cpu, SolverBackend::Cpu.label()),
                    (SolverBackend::Gpu, SolverBackend::Gpu.label()),
                ],
                "which device the finite element solve runs on. cpu is double precision \
                 throughout and reproducible bit for bit everywhere, for a given build and thread \
                 count; gpu is several times faster and reproducible on one machine and driver \
                 only. A backend named here is an instruction: gpu fails on a build or a machine \
                 that has none",
            ) {
                write_solver_backend(field.state.config_mut(), chosen);
            }
            let mut tolerance = field
                .state
                .config()
                .solver
                .as_ref()
                .and_then(|s| s.tolerance);
            if optional_value_row(
                ui,
                field,
                "tolerance",
                &mut tolerance,
                constants::CG_RELATIVE_TOLERANCE,
                &format!("{:e}", constants::CG_RELATIVE_TOLERANCE),
                "the relative residual ||r|| / ||b|| the optimization's linear solves are taken \
                 to, on either backend. It is a hard target: a solve that reaches the iteration \
                 cap short of it fails the run, and a model can sit a factor of 1.5 from a number \
                 no printed part could tell apart. Looser is faster and less precise, and well \
                 below 1e-6 that precision is finer than one voxel of mesh resolves anyway, so 3e-8 \
                 is a run rescued rather than a part changed. Unticked leaves the key out and the \
                 solves run to the default",
                tolerance_widget,
            ) {
                write_solver_tolerance(field.state.config_mut(), tolerance);
            }
        });
}

/// Write the backend into the `[solver]` table, keeping the other key.
///
/// The table is one struct and the panel replaces it whole rather than editing
/// it in place, so each of the two widgets over it has to carry the other's key
/// across: a tolerance a user pinned may not be dropped by a click on the
/// backend combo.
fn write_solver_backend(config: &mut crate::config::Config, backend: SolverBackend) {
    let tolerance = config.solver.as_ref().and_then(|s| s.tolerance);
    config.solver = Some(SolverConfig {
        backend: Some(backend),
        tolerance,
    });
}

/// The same the other way round. `None` is the checkbox unticked, which takes
/// the key out and leaves the table: an empty `[solver]` resolves to exactly
/// what no table at all does, and removing it would take a comment block the
/// file may have had in it with no key left to hang on.
fn write_solver_tolerance(config: &mut crate::config::Config, tolerance: Option<f64>) {
    let backend = config.solver.as_ref().and_then(|s| s.backend);
    config.solver = Some(SolverConfig { backend, tolerance });
}

/// The backend the solver combo shows: the one a run of this configuration
/// would use, not the enum's derived default.
///
/// The two are deliberately different things and were being confused here. An
/// absent `[solver]` table resolves to [`constants::DEFAULT_SOLVER_BACKEND`],
/// which is the compute device with a soft fall back to the CPU;
/// `SolverBackend::default()` is the CPU, which is the *reference* backend the
/// determinism promises are written against. A keyless configuration was
/// therefore shown as `cpu` while every run under it asked for the device - and
/// a user clicking `gpu` to correct the display pinned an explicit instruction,
/// which is exactly what forfeits the soft fall back.
///
/// Nothing here opens a device, for the reason
/// [`crate::config::Config::solver_params`] does not: this runs every frame. A
/// machine with no adapter is still shown the backend it asked for, and the run
/// says what it fell back to, as the console summary does. The error path is
/// reachable only for a `gpu` asked for by name in a build without the feature,
/// where the default is what the file itself says.
fn shown_backend(config: &crate::config::Config) -> SolverBackend {
    config
        .solver_params()
        .map(|params| params.backend)
        .unwrap_or(constants::DEFAULT_SOLVER_BACKEND)
}

/// Voxel size or cell target.
fn resolution_section(ui: &mut egui::Ui, field: &mut Field<'_>) {
    egui::CollapsingHeader::new("resolution")
        .default_open(true)
        .show(ui, |ui| {
            reset_row(ui, field, Section::Resolution);
            let by_size = field.state.config().resolution.voxel_size_mm.is_some();
            let mut mode = by_size;
            if combo(
                ui,
                field,
                "given as",
                &mut mode,
                &[(true, "voxel size"), (false, "target cells")],
                "how the grid is described: an edge length for the voxels, or a number of cells \
                 the edge length is derived from the domain's bounding box to meet. A \
                 configuration carries exactly one of the two",
            ) {
                let config = field.state.config_mut();
                if mode {
                    config.resolution.voxel_size_mm = Some(constants::VIEW_EDIT_DEFAULT_VOXEL_MM);
                    config.resolution.target_cells = None;
                } else {
                    config.resolution.target_cells =
                        Some(constants::VIEW_EDIT_DEFAULT_TARGET_CELLS);
                    config.resolution.voxel_size_mm = None;
                }
            }
            if let Some(mut value) = field.state.config().resolution.voxel_size_mm
                && length_row(
                    ui,
                    field,
                    "voxel mm",
                    &mut value,
                    "the edge length of one cubic voxel, in mm. Everything is solved and exported \
                     on this grid, so it sets both the finest detail a run can hold and what the \
                     run costs",
                )
            {
                field.state.config_mut().resolution.voxel_size_mm = Some(value);
            }
            if let Some(mut value) = field.state.config().resolution.target_cells
                && integer_row(
                    ui,
                    field,
                    "cells",
                    &mut value,
                    "how many cells to aim for; the voxel size is derived from the domain's \
                     bounding box to land near it. A grid past the cell budget is refused",
                )
            {
                field.state.config_mut().resolution.target_cells = Some(value);
            }
        });
}

/// Material preset or explicit properties.
fn material_section(ui: &mut egui::Ui, field: &mut Field<'_>) {
    egui::CollapsingHeader::new("material")
        .default_open(false)
        .show(ui, |ui| {
            reset_row(ui, field, Section::Material);
            let preset = field.state.config().material.preset.clone();
            let mut by_preset = preset.is_some();
            if combo(
                ui,
                field,
                "given as",
                &mut by_preset,
                &[(true, "preset"), (false, "custom")],
                "whether the material is a named preset or four values of your own. A \
                 configuration carries one or the other, never both; switching to custom starts \
                 from the preset that was selected",
            ) {
                let config = field.state.config_mut();
                if by_preset {
                    let first = constants::MATERIAL_PRESETS[0];
                    config.material.preset = Some(first.key.to_string());
                    config.material.youngs_modulus_mpa = None;
                    config.material.poisson_ratio = None;
                    config.material.density_g_cm3 = None;
                    config.material.yield_strength_mpa = None;
                } else {
                    // Custom values start from whatever preset was selected, so
                    // switching is a starting point rather than a blank form.
                    let source = constants::MATERIAL_PRESETS
                        .iter()
                        .find(|p| Some(p.key.to_string()) == config.material.preset)
                        .unwrap_or(&constants::MATERIAL_PRESETS[0]);
                    config.material.youngs_modulus_mpa = Some(source.youngs_modulus_mpa);
                    config.material.poisson_ratio = Some(source.poisson_ratio);
                    config.material.density_g_cm3 = Some(source.density_g_cm3);
                    config.material.yield_strength_mpa = Some(source.yield_strength_mpa);
                    config.material.preset = None;
                }
            }
            if let Some(current) = preset {
                let options: Vec<&str> =
                    constants::MATERIAL_PRESETS.iter().map(|p| p.key).collect();
                if let Some(chosen) = combo_str(
                    ui,
                    field,
                    "preset",
                    &current,
                    &options,
                    "which built-in filament to use. The figures are typical printed-part \
                     datasheet values rather than certified minima: switch to custom and enter \
                     your supplier's when you have them",
                ) {
                    field.state.config_mut().material.preset = Some(chosen);
                }
            } else {
                let mut material = field.state.config().material.clone();
                let mut changed = false;
                for (label, slot, speed, tooltip) in [
                    (
                        "E MPa",
                        &mut material.youngs_modulus_mpa,
                        constants::VIEW_EDIT_DRAG_SPEED_MM,
                        "Young's modulus in MPa: how stiff the solid material is. It scales the \
                         whole finite element solve",
                    ),
                    (
                        "nu",
                        &mut material.poisson_ratio,
                        constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                        "Poisson's ratio: how much the material spreads sideways when it is \
                         compressed. Dimensionless",
                    ),
                    (
                        "rho g/cm3",
                        &mut material.density_g_cm3,
                        constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                        "mass density in g/cm3. It drives the self-weight loads and the mass the \
                         run estimates for the part",
                    ),
                    (
                        "yield MPa",
                        &mut material.yield_strength_mpa,
                        constants::VIEW_EDIT_DRAG_SPEED_MM,
                        "the stress the material yields at, in MPa. It is what the safety factor \
                         is measured against and what the stress colouring's red end is, so a \
                         part judged against the wrong number is judged wrongly",
                    ),
                ] {
                    if let Some(value) = slot {
                        changed |= number_row(ui, field, label, value, speed, tooltip);
                    }
                }
                if changed {
                    field.state.config_mut().material = material;
                }
            }
        });
}

/// The optimization controls.
fn optimization_section(ui: &mut egui::Ui, field: &mut Field<'_>) {
    egui::CollapsingHeader::new("optimization")
        .default_open(true)
        .show(ui, |ui| {
            // Before the clone the rest of the section edits, so a reset is what
            // the rows below it then show.
            reset_row(ui, field, Section::Optimization);
            let mut optimization = field.state.config().optimization.clone();
            let mut changed = number_row(
                ui,
                field,
                "mass fraction",
                &mut optimization.mass_fraction,
                constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                "how much of the design space ends up as material, between 0 and 1, measured on \
                 the printed densities. It is the weight target the whole run is held to: less \
                 material is a lighter part and a lower safety factor. The growth engine \
                 normalizes its strut radii to it",
            );
            changed |= length_row(
                ui,
                field,
                "min feature mm",
                &mut optimization.min_feature_mm,
                "the smallest feature the design may hold, in mm: the density filter's radius is \
                 half of it. For the growth engine it is the smallest strut diameter, and the \
                 scale every growth default is derived from. Smaller than about three voxels is \
                 warned about - the filter cannot resolve it",
            );
            changed |= optional_row(
                ui,
                field,
                "penalty",
                &mut optimization.penalty,
                constants::SIMP_PENALTY,
                constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                "the exponent p of E(x) = Emin + x^p (E0 - Emin), which is what prices \
                 intermediate density out of the design. At least 1, and 2 to 5 is the useful \
                 range: higher drives the field black and white at the cost of more local \
                 minima. Every shipped example is at the default 3. simp only",
            );
            changed |= optional_integer_row(
                ui,
                field,
                "max iterations",
                &mut optimization.max_iterations,
                constants::DEFAULT_MAX_ITERATIONS,
                "a budget rather than a target: a run stops on convergence, or because it \
                 stopped making progress, long before it. simp only",
            );
            changed |= optional_row(
                ui,
                field,
                "convergence tol",
                &mut optimization.convergence_tol,
                constants::DEFAULT_CONVERGENCE_TOL,
                constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                "stop once the largest design variable change of an iteration falls below this. \
                 simp only",
            );
            let mut update = optimization.update.unwrap_or_default();
            if combo(
                ui,
                field,
                "update",
                &mut update,
                &[
                    (UpdateScheme::Oc, UpdateScheme::Oc.label()),
                    (UpdateScheme::Mma, UpdateScheme::Mma.label()),
                ],
                "which scheme moves the design variables under the volume constraint. oc, the \
                 optimality criteria step, is the default and the reference; mma damps exactly \
                 the variables that keep crossing the box, which is what a run that will not \
                 settle needs. simp only",
            ) {
                optimization.update = Some(update);
                changed = true;
            }

            let mut overhang = optimization.overhang.is_some();
            let toggle = ui
                .checkbox(&mut overhang, "overhang constraint")
                .on_hover_text(
                    "make the design self-supporting as it is optimized, so what the analysis, \
                     the volume target and the STL all see is already printable. The 45 degree \
                     angle is fixed by the filter's stencil, so there is no angle to set. simp \
                     only: the growth engine rejects it",
                );
            if field.apply(&toggle) {
                optimization.overhang = overhang.then_some(OverhangConfig {
                    build_direction: constants::VIEW_EDIT_DEFAULT_BUILD_DIRECTION,
                });
                changed = true;
            }
            if let Some(config) = &mut optimization.overhang {
                let mut direction = config.build_direction;
                if combo(
                    ui,
                    field,
                    "build direction",
                    &mut direction,
                    &[
                        (BuildDirection::XPlus, BuildDirection::XPlus.label()),
                        (BuildDirection::XMinus, BuildDirection::XMinus.label()),
                        (BuildDirection::YPlus, BuildDirection::YPlus.label()),
                        (BuildDirection::YMinus, BuildDirection::YMinus.label()),
                        (BuildDirection::ZPlus, BuildDirection::ZPlus.label()),
                        (BuildDirection::ZMinus, BuildDirection::ZMinus.label()),
                    ],
                    "the direction the printer's layers stack in, so z+ means the build plate is \
                     the z minimum face of the grid and overhangs open upwards",
                ) {
                    config.build_direction = direction;
                    changed = true;
                }
            }

            // The guide wireframe, on the same pattern: the table being there is
            // what switches it on, and every key inside it is optional, so the
            // rows appear with their defaults showing.
            let mut wireframe = optimization.wireframe.is_some();
            let toggle = ui
                .checkbox(&mut wireframe, "guide wireframe")
                .on_hover_text(
                    "start the run from a thin wire routed through every load, support and keepin \
                 region instead of from a uniform field, held as a density floor for the first \
                 iterations and then let go. It is a guide and never a constraint: afterwards the \
                 optimizer may keep it, thicken it or dissolve it. simp only",
                );
            if field.apply(&toggle) {
                optimization.wireframe = wireframe.then_some(WireframeConfig {
                    radius_mm: None,
                    hold_iterations: None,
                    seed_density: None,
                });
                changed = true;
            }
            if let Some(config) = &mut optimization.wireframe {
                changed |= optional_row(
                    ui,
                    field,
                    "wire radius mm",
                    &mut config.radius_mm,
                    constants::WIREFRAME_RADIUS_MM,
                    constants::VIEW_EDIT_DRAG_SPEED_MM,
                    "how thick the guide wire is rasterized, in mm",
                );
                changed |= optional_integer_row(
                    ui,
                    field,
                    "hold iterations",
                    &mut config.hold_iterations,
                    constants::WIREFRAME_HOLD_ITERATIONS,
                    "how many iterations the wire is held as a density floor before it is \
                     released. A run cannot decide it has finished while the floor is on, and a \
                     hold that outlasts max iterations never releases at all",
                );
                changed |= optional_row(
                    ui,
                    field,
                    "seed density",
                    &mut config.seed_density,
                    constants::WIREFRAME_SEED_DENSITY,
                    constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                    "the density the wire's cells are seeded and held at",
                );
            }

            // The local volume cap, on the same pattern again. Its radius
            // default is derived from min_feature_mm, so the row shows what this
            // configuration would really use rather than a fixed number.
            let mut local_volume = optimization.local_volume.is_some();
            let toggle = ui
                .checkbox(&mut local_volume, "local volume")
                .on_hover_text(
                    "cap how much material any neighbourhood may hold, on top of the global mass \
                     fraction. Without it the cheapest way to meet a mass target is a few thick \
                     members and flush surfaces; with it the optimizer has to spread the same \
                     material over many thinner ones, which is what makes a part bone-like. \
                     REQUIRES update = \"mma\": the cap is a second constraint and the optimality \
                     criteria step prices one. simp only",
                );
            if field.apply(&toggle) {
                optimization.local_volume = local_volume.then_some(LocalVolumeConfig {
                    max_fraction: None,
                    radius_mm: None,
                });
                changed = true;
            }
            let default_radius_mm = constants::LOCAL_VOLUME_RADIUS_FACTOR
                * optimization.min_feature_mm
                / constants::FILTER_RADIUS_DIVISOR;
            if let Some(config) = &mut optimization.local_volume {
                changed |= optional_row(
                    ui,
                    field,
                    "max fraction",
                    &mut config.max_fraction,
                    constants::LOCAL_VOLUME_MAX_FRACTION,
                    constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                    "the largest share of material a neighbourhood may hold, between 0 and 1 and \
                     above mass fraction - the average of the local shares is the global one, so \
                     a cap at or below it cannot be met by any design. The cap is on a smooth \
                     aggregate, so the worst single neighbourhood runs a little above it, and \
                     the run reports that true worst figure every iteration",
                );
                changed |= optional_row(
                    ui,
                    field,
                    "local radius mm",
                    &mut config.radius_mm,
                    default_radius_mm,
                    constants::VIEW_EDIT_DRAG_SPEED_MM,
                    "the radius of the neighbourhood the share is measured over, in mm. Must be \
                     above the density filter radius (half of min feature mm), or it would price \
                     one member's own cross section rather than how the members are spread; the \
                     default is three of them. A much larger radius is warned about - it is a \
                     second filter, and every iteration pays for its stencil twice",
                );
            }
            if changed {
                field.state.config_mut().optimization = optimization;
            }
        });
}

/// The growth controls, shown whenever that engine is selected.
///
/// Every key of the table is optional and so is the table itself, so the
/// section is driven by the engine rather than by the table being there: a
/// growth configuration that pins nothing still gets its controls, and the
/// table appears in the file the moment one of them is pinned.
fn growth_section(ui: &mut egui::Ui, field: &mut Field<'_>) {
    if !field.state.config().is_growth() {
        return;
    }
    egui::CollapsingHeader::new("growth")
        .default_open(true)
        .show(ui, |ui| {
            reset_row(ui, field, Section::Growth);
            let mut growth = field.state.config().growth.clone().unwrap_or(GrowthConfig {
                seed: None,
                attractor_per_cm3: None,
                attraction_radius_mm: None,
                kill_radius_mm: None,
                step_mm: None,
                murray_exponent: None,
                max_radius_mm: None,
                max_steps: None,
                prune: None,
                symmetry: None,
            });
            let mut changed = false;
            let seed_help = "the only randomness a growth run has: the same seed gives a byte \
                             identical part. Changing it moves the canopy and nothing else - the \
                             volume target, the load paths and the clamps are met either way";
            ui.horizontal(|ui| {
                let mut present = growth.seed.is_some();
                let toggle = ui.checkbox(&mut present, "seed").on_hover_text(seed_help);
                if field.apply(&toggle) {
                    growth.seed = present.then_some(constants::DEFAULT_GROWTH_SEED);
                    changed = true;
                }
                match &mut growth.seed {
                    Some(seed) => {
                        let response = ui
                            .add(egui::DragValue::new(seed).speed(1.0))
                            .on_hover_text(seed_help);
                        changed |= field.apply(&response);
                    }
                    None => {
                        ui.label(egui::RichText::new("default").weak())
                            .on_hover_text(seed_help);
                    }
                }
            });
            for (label, slot, default, tooltip) in [
                (
                    "attractor /cm3",
                    &mut growth.attractor_per_cm3,
                    constants::DEFAULT_ATTRACTOR_PER_CM3,
                    "how many attraction points are scattered per cm3 of design domain. They are \
                     what the branches grow towards, and a branch consumes the ones it reaches",
                ),
                (
                    "murray exponent",
                    &mut growth.murray_exponent,
                    constants::DEFAULT_MURRAY_EXPONENT,
                    "n in Murray's law, r_parent^n = sum of r_child^n, which is how a junction's \
                     strut radii are related. Between 1 and 8",
                ),
            ] {
                changed |= optional_row(
                    ui,
                    field,
                    label,
                    slot,
                    default,
                    constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                    tooltip,
                );
            }
            // The four lengths default to multiples of min_feature_mm, so their
            // "off" state is the resolved default rather than a fixed number.
            let resolved = field.state.config().growth_params().ok().flatten();
            for (label, slot, default, tooltip) in [
                (
                    "attraction mm",
                    &mut growth.attraction_radius_mm,
                    resolved.map(|g| g.attraction_radius_mm),
                    "how far a branch tip sees an attraction point, in mm. It has to stay above \
                     the kill radius; unticked it is three times whatever that resolves to",
                ),
                (
                    "kill mm",
                    &mut growth.kill_radius_mm,
                    resolved.map(|g| g.kill_radius_mm),
                    "how close a branch has to come to an attraction point to consume it, in mm. \
                     This is the knob for looks: smaller gives a finer, denser canopy of thinner \
                     branches, larger gives fewer, fatter ones",
                ),
                (
                    "step mm",
                    &mut growth.step_mm,
                    resolved.map(|g| g.step_mm),
                    "how far a branch node moves in one growth step, in mm. It may not be shorter \
                     than a quarter of a voxel",
                ),
                (
                    "max radius mm",
                    &mut growth.max_radius_mm,
                    resolved.map(|g| g.max_radius_mm),
                    "the thickest a single strut may be scaled to, in mm. It may not be below \
                     half the minimum feature, which is the thinnest one",
                ),
            ] {
                changed |= optional_row(
                    ui,
                    field,
                    label,
                    slot,
                    default.unwrap_or_default(),
                    constants::VIEW_EDIT_DRAG_SPEED_MM,
                    tooltip,
                );
            }
            changed |= optional_integer_row(
                ui,
                field,
                "max steps",
                &mut growth.max_steps,
                constants::DEFAULT_GROWTH_MAX_STEPS,
                "the cap on growth iterations. Branching normally ends when the attraction points \
                 run out, well inside it",
            );
            ui.horizontal(|ui| {
                let mut prune = growth.prune.unwrap_or(constants::DEFAULT_GROWTH_PRUNE);
                let response = ui
                    .checkbox(&mut prune, "prune free branch tips")
                    .on_hover_text(
                        "remove every branch that ends on nothing, back to the last junction that \
                         leads somewhere. A free tip carries no load, cannot be printed without \
                         support, and is paid for out of the same mass fraction as the branches \
                         that work. Unticking keeps them, for the decorative growth alone",
                    );
                if field.apply(&response) {
                    growth.prune = Some(prune);
                    changed = true;
                }
            });
            changed |= symmetry_rows(ui, field, &mut growth.symmetry);
            if changed {
                field.state.config_mut().growth = Some(growth);
            }
        });
}

/// The `[growth.symmetry]` rows: whether there is a symmetry at all, which kind,
/// and the keys that kind needs.
///
/// Only the keys of the selected kind are offered, and switching kind replaces
/// the table rather than leaving the other kind's keys behind, because a
/// configuration carrying both is rejected. The second mirror plane cannot be
/// set to the first, for the same reason.
fn symmetry_rows(
    ui: &mut egui::Ui,
    field: &mut Field<'_>,
    slot: &mut Option<SymmetryConfig>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let mut present = slot.is_some();
        let toggle = ui.checkbox(&mut present, "symmetry").on_hover_text(
            "grow one sector of the domain and replicate exact copies of it, so a symmetric \
             problem comes out as a symmetric part instead of as sectors that are all sound and \
             none the same",
        );
        if field.apply(&toggle) {
            *slot = present.then(|| mirror_symmetry(constants::VIEW_EDIT_DEFAULT_SYMMETRY_PLANE));
            changed = true;
        }
    });
    let Some(symmetry) = slot else {
        return changed;
    };

    let mut kind = symmetry.kind;
    if combo(
        ui,
        field,
        "symmetry kind",
        &mut kind,
        &[
            (SymmetryKind::Mirror, SymmetryKind::Mirror.label()),
            (SymmetryKind::Rotational, SymmetryKind::Rotational.label()),
        ],
        "how the grown sector is replicated: mirror reflects it about one or two planes through \
         the centre of the domain, rotational repeats it about an axis. Each kind carries only \
         its own keys",
    ) {
        *symmetry = match kind {
            SymmetryKind::Mirror => mirror_symmetry(constants::VIEW_EDIT_DEFAULT_SYMMETRY_PLANE),
            SymmetryKind::Rotational => SymmetryConfig {
                kind,
                planes: None,
                order: Some(constants::VIEW_EDIT_DEFAULT_SYMMETRY_ORDER),
                axis: Some(constants::DEFAULT_SYMMETRY_AXIS),
            },
        };
        return true;
    }

    match symmetry.kind {
        SymmetryKind::Mirror => {
            let planes = symmetry.planes.clone().unwrap_or_default();
            let mut first = planes
                .first()
                .copied()
                .unwrap_or(constants::VIEW_EDIT_DEFAULT_SYMMETRY_PLANE);
            let mut second = planes.get(1).copied();
            changed |= combo(
                ui,
                field,
                "mirror plane",
                &mut first,
                &axis_options(),
                "the normal of the plane the grown half is mirrored about, through the centre of \
                 the domain's bounding box",
            );
            // The two planes have to be different axes, so the one already
            // chosen is not offered again.
            let mut options: Vec<(Option<Axis>, &str)> = vec![(None, "none")];
            options.extend(
                Axis::ALL
                    .iter()
                    .filter(|axis| **axis != first)
                    .map(|axis| (Some(*axis), axis.label())),
            );
            changed |= combo(
                ui,
                field,
                "mirror plane 2",
                &mut second,
                &options,
                "an optional second mirror plane, which quarters the domain instead of halving \
                 it. It cannot be the axis the first one already uses",
            );
            if changed {
                symmetry.planes = Some(match second.filter(|axis| *axis != first) {
                    Some(second) => vec![first, second],
                    None => vec![first],
                });
            }
        }
        SymmetryKind::Rotational => {
            let mut order = symmetry
                .order
                .unwrap_or(constants::VIEW_EDIT_DEFAULT_SYMMETRY_ORDER);
            if integer_row(
                ui,
                field,
                "rotation order",
                &mut order,
                "how many sectors of the full turn one grown sector is copied into, 2 to 12. 2 \
                 and 4 land exactly on the voxel lattice; the others are exact in the skeleton \
                 and approximate once it is rasterized",
            ) {
                symmetry.order = Some(order.clamp(
                    constants::GROWTH_SYMMETRY_MIN_ORDER,
                    constants::GROWTH_SYMMETRY_MAX_ORDER,
                ));
                changed = true;
            }
            let mut axis = symmetry.axis.unwrap_or(constants::DEFAULT_SYMMETRY_AXIS);
            if combo(
                ui,
                field,
                "rotation axis",
                &mut axis,
                &axis_options(),
                "the axis the sectors are turned about, through the centre of the domain's \
                 bounding box",
            ) {
                symmetry.axis = Some(axis);
                changed = true;
            }
        }
    }
    changed
}

/// A mirror symmetry about one plane, which is what a newly enabled table and a
/// switch back to `mirror` start from.
fn mirror_symmetry(plane: Axis) -> SymmetryConfig {
    SymmetryConfig {
        kind: SymmetryKind::Mirror,
        planes: Some(vec![plane]),
        order: None,
        axis: None,
    }
}

/// The three axes as dropdown options.
fn axis_options() -> [(Axis, &'static str); 3] {
    [
        (Axis::X, Axis::X.label()),
        (Axis::Y, Axis::Y.label()),
        (Axis::Z, Axis::Z.label()),
    ]
}

/// The output controls. The STL path is shown but not editable here: where a
/// run writes is a decision about the project, not about the geometry.
fn output_section(ui: &mut egui::Ui, field: &mut Field<'_>) {
    egui::CollapsingHeader::new("output")
        .default_open(false)
        .show(ui, |ui| {
            reset_row(ui, field, Section::Output);
            // Both output paths are shown and neither is editable: where a run
            // writes is a decision about the project, not about the geometry.
            ui.label(
                egui::RichText::new(format!("stl    {}", field.state.config().output.stl_path))
                    .small()
                    .weak(),
            );
            ui.label(
                egui::RichText::new(match &field.state.config().output.stress_json {
                    Some(path) => format!("stress {path}"),
                    None => "stress no json report".to_string(),
                })
                .small()
                .weak(),
            );
            let mut output = field.state.config().output.clone();
            let mut changed = optional_row(
                ui,
                field,
                "iso level",
                &mut output.iso_level,
                constants::DEFAULT_ISO_LEVEL,
                constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                "the density the exported surface is drawn at: what is denser than this is part, \
                 what is thinner is air. It is also the threshold the cavity report reads",
            );
            changed |= optional_integer_row(
                ui,
                field,
                "smoothing",
                &mut output.smoothing_iterations,
                constants::DEFAULT_SMOOTHING_ITERATIONS,
                "Taubin passes over the extracted surface. They move vertices and never add any, \
                 so past about ten they only round corners the design meant to keep",
            );
            changed |= optional_integer_row(
                ui,
                field,
                "supersample",
                &mut output.supersample,
                constants::DEFAULT_SUPERSAMPLE,
                "mesh the same field on a lattice this many times finer in every axis, 1 to 4. It \
                 takes the staircase off a strut; triangles and file size grow with about its \
                 square and memory with its cube. It cannot invent structure the voxel grid never \
                 had",
            );
            let mut voids = output.voids.unwrap_or_default();
            if combo(
                ui,
                field,
                "voids",
                &mut voids,
                &[
                    (VoidPolicy::Warn, VoidPolicy::Warn.label()),
                    (VoidPolicy::Fill, VoidPolicy::Fill.label()),
                ],
                "what to do about cavities the exported surface would seal, which cannot be \
                 printed without trapped support. warn is the default and reports each one; fill \
                 turns them into material before meshing. One overlapping a keepout or the space \
                 outside the domain is never filled",
            ) {
                output.voids = Some(voids);
                changed = true;
            }
            let mut islands = output.islands.unwrap_or_default();
            if combo(
                ui,
                field,
                "islands",
                &mut islands,
                &[
                    (IslandPolicy::Cull, IslandPolicy::Cull.label()),
                    (IslandPolicy::Keep, IslandPolicy::Keep.label()),
                ],
                "what to do about the pieces of the exported surface that touch no support, load \
                 or keepin region - loose objects to a slicer. cull is the default and removes \
                 them before the mesh is validated; keep exports the surface fragments and all. \
                 Either way they are reported",
            ) {
                output.islands = Some(islands);
                changed = true;
            }
            let mut boundaries = output.boundaries.unwrap_or_default();
            if combo(
                ui,
                field,
                "boundaries",
                &mut boundaries,
                &[
                    (BoundaryFidelity::Exact, BoundaryFidelity::Exact.label()),
                    (BoundaryFidelity::Voxel, BoundaryFidelity::Voxel.label()),
                ],
                "how closely the exported surface has to follow the shapes the configuration \
                 wrote. exact is the default: every vertex that sits inside a keepout or outside \
                 the domain is projected back onto the analytic surface, so a bore comes out as \
                 the cylinder that was asked for rather than a fraction of a voxel under it. \
                 voxel exports the isosurface as the field produced it, which is what growforge \
                 did before 0.22.0",
            ) {
                output.boundaries = Some(boundaries);
                changed = true;
            }
            let mut trim = output.trim.unwrap_or_default();
            if combo(
                ui,
                field,
                "trim",
                &mut trim,
                &[
                    (TrimPolicy::Off, TrimPolicy::Off.label()),
                    (TrimPolicy::Stress, TrimPolicy::Stress.label()),
                ],
                "what to do about material that carries no load: the prongs reaching to nothing \
                 and the wisps between members. off is the default and exports the field as it \
                 came out; stress removes the cells whose von Mises stress over every load case \
                 is a negligible fraction of the peak. Everything the configuration declared is \
                 kept, and the whole pass is refused rather than left half applied if it would \
                 take the structure apart",
            ) {
                output.trim = Some(trim);
                changed = true;
            }
            // Only where it means something: the fraction is the trim's own
            // threshold, and a row offering it under `off` would be a control
            // over a pass that is not running.
            if trim != TrimPolicy::Off {
                changed |= optional_row(
                    ui,
                    field,
                    "trim fraction",
                    &mut output.trim_stress_fraction,
                    constants::TRIM_STRESS_FRACTION,
                    constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                    "the share of the part's peak stress that counts as negligible, between 0 and \
                     1. One percent is deliberately far out on the tail: a member that is really \
                     working carries far more than that, so raising this starts removing \
                     structure",
                );
            }
            let mut reinforce = output.reinforce.unwrap_or_default();
            if combo(
                ui,
                field,
                "reinforce",
                &mut reinforce,
                &[
                    (ReinforcePolicy::Off, ReinforcePolicy::Off.label()),
                    (
                        ReinforcePolicy::MinThickness,
                        ReinforcePolicy::MinThickness.label(),
                    ),
                ],
                "what to do about material too thin to print: the arms a slicer lays down as a \
                 single bead, or misses. off is the default and exports the field as it came out; \
                 min_thickness measures every member on its spine, by the ball that fits inside \
                 it, and thickens the ones under the floor into the design space around them - \
                 never into a keepout or out of the domain. A member pressed against one is \
                 counted and warned about rather than stopping the pass. Nothing to do with the \
                 growth engine's own thickening, which sizes a skeleton's struts as it builds one",
            ) {
                output.reinforce = Some(reinforce);
                changed = true;
            }
            // The same rule the trim fraction follows, and the same reason: a
            // floor is a control over a pass, not over a file. Its default is
            // this configuration's own min_feature_mm rather than a constant -
            // the smallest feature the run was asked to resolve is the smallest
            // one it can be held to.
            let min_feature_mm = field.state.config().optimization.min_feature_mm;
            if reinforce != ReinforcePolicy::Off {
                changed |= optional_row(
                    ui,
                    field,
                    "reinforce mm",
                    &mut output.reinforce_thickness_mm,
                    min_feature_mm,
                    constants::VIEW_EDIT_DRAG_SPEED_FRACTION,
                    "the thickness in millimetres every member of the part is held to, absent the \
                     run's own min_feature_mm. Set it to what the printer can really lay down - \
                     two or three extrusion widths - rather than to what the optimization was \
                     asked to resolve. Where the fill fits, the pass meets it or exceeds it \
                     slightly and never rounds down to it; where it does not, the run says which \
                     places are still thin",
                );
            }
            if changed {
                field.state.config_mut().output = output;
            }
        });
}

/// What the post-run passes did to the part, beside the switch that colours the
/// surface by the stresses one of them judged the material on.
///
/// Nothing at all until a run has finished: the `[output] trim` and
/// `[output] reinforce` passes say nothing under their default `off`, and the
/// `[output] boundaries` clamp says nothing under `voxel`. Each line carries the
/// name of the pass that wrote it, and an outcome that needs saying loudly - a
/// trim the connectivity guard refused, a member the fill could not thicken, a
/// vertex the clamp could not correct - is drawn in the panel's warning colour.
/// None of it is something to find out from the console afterwards.
///
/// The trim's line and the reinforcement's are adjacent on purpose: what the run
/// freed and what it spent on the arms are the two halves of one exchange, and
/// they read as one only side by side.
///
/// One wrapping label per line, and it has to stay that way: these rows only
/// exist once a run has finished, so
/// `no_row_of_the_panel_is_wider_than_the_panel` - which drives no run - cannot
/// reach them. A label wraps and so cannot widen the panel; a `ui.horizontal` of
/// several widgets does not wrap and can. Anything of that shape added here has
/// to be `horizontal_wrapped`, and the guard test has to be given a way to lay
/// this block out.
fn pass_notes(ui: &mut egui::Ui, editor: &Editor) {
    for (pass, notes) in [
        ("trim", editor.trim_notes()),
        ("reinforce", editor.reinforce_notes()),
        ("boundaries", editor.clamp_notes()),
    ] {
        for note in notes {
            let mut text = egui::RichText::new(format!("{pass} {note}")).small();
            if note.starts_with("warning") {
                text = text.color(ui.visuals().warn_fg_color);
            }
            ui.label(text);
        }
    }
}

/// What the part came out at: the safety factor, and the peak of every load
/// case.
///
/// Its own block rather than another pass note, because it is the number a part
/// is judged by and not a report of something a pass did - the headline is drawn
/// strong, the per-case peaks it was taken from sit under it, and what the
/// number is worth is drawn above it.
///
/// Absent until a run or a generation has analysed the field it wrote, and
/// absent again while the next one runs: what it says describes the design on
/// screen, or it says nothing. The console of the session is sent the same lines
/// by the run, from the same formatter.
fn stress_summary(ui: &mut egui::Ui, editor: &Editor) {
    let Some(summary) = editor.stress_summary() else {
        return;
    };
    ui.separator();
    stress_block(ui, &summary);
}

/// The summary itself, header and all, drawn from a summary rather than from
/// the editor, so every state of it - including the export that came out in
/// pieces - can be laid out in a test.
///
/// The header is here rather than in [`stress_summary`] for that same reason:
/// the block exists only behind a finished run, and a header drawn where the
/// summary is would be one no test without a run could ask about. What gates it
/// is unchanged - no summary, no block and no header.
///
/// The warning goes above the headline and in the warning colour, which is the
/// rule [`pass_notes`] draws its own warnings by: what a safety factor is worth
/// has to be read before the factor, not after it. It is the same line the
/// session's console is sent, from the same formatter.
///
/// Behind a finished run like [`pass_notes`], and under the same rule for the
/// same reason: `no_row_of_the_panel_is_wider_than_the_panel` cannot reach a
/// block that needs a run behind it, so every row here is one wrapping label,
/// which cannot widen the panel. A multi-widget `ui.horizontal` added here can,
/// and would have to be `horizontal_wrapped` and brought inside that test.
fn stress_block(ui: &mut egui::Ui, summary: &StressSummary) {
    egui::CollapsingHeader::new(constants::VIEW_EDIT_BLOCK_STRESS)
        .default_open(true)
        .show(ui, |ui| {
            if let Some(warning) = &summary.warning {
                ui.label(
                    egui::RichText::new(warning.as_str())
                        .strong()
                        .color(ui.visuals().warn_fg_color),
                );
            }
            ui.label(egui::RichText::new(summary.headline.as_str()).strong());
            for case in &summary.cases {
                ui.monospace(case.as_str());
            }
        });
}

/// The layer switches, under a header of their own.
///
/// The switches are the viewer's own, so the same scene is controlled the same
/// way in both windows; only the header is the editor's, which is why the wrap
/// is here at the call site rather than in the shared function - and why that
/// function is told not to write the heading a second time.
fn show_block(ui: &mut egui::Ui, scene: &mut Scene) {
    egui::CollapsingHeader::new(constants::VIEW_LAYER_SWITCHES_LABEL)
        .default_open(true)
        .show(ui, |ui| layer_switches(ui, scene, true));
}

/// The live problem summary.
///
/// Closed to begin with: it is a read-out of what the configuration on screen
/// amounts to - grid, counts, memory - and the panel opens on the objects and
/// their properties, which is what a session is there to change.
fn summary(ui: &mut egui::Ui, state: &EditorState) {
    egui::CollapsingHeader::new(constants::VIEW_EDIT_BLOCK_PROBLEM)
        .default_open(false)
        .show(ui, |ui| {
            if state.is_stale() {
                ui.label(egui::RichText::new("refreshing...").weak());
            }
            for line in state.summary() {
                ui.monospace(line);
            }
        });
}

/// The control legend, as the panel lists it.
///
/// Hand written and hand maintained: nothing derives it from the code that
/// implements the bindings, so a binding is listed here only because somebody
/// listed it. Kept as data rather than as a run of calls so that a test can at
/// least ask whether a line is here at all.
const CONTROLS: [&str; 15] = [
    "left click    select",
    "left drag     orbit",
    "drag handle   move or resize (snapped)",
    "drag arc      rotate about that axis",
    "Alt + drag    free drag, no snapping",
    "click number  type an exact value",
    "Esc           cancel the number being typed",
    "right drag    pan",
    "scroll        zoom",
    "F             fit view",
    "Delete        delete selection",
    "Ctrl+Z / +Y   undo / redo",
    "Ctrl+S        save",
    "Ctrl+O        open another configuration",
    "Ctrl+N        new configuration",
];

/// The control legend.
///
/// Closed to begin with, like the problem summary: it is a reference to look up
/// once rather than something a session works in, and it is the longest block
/// the panel has.
fn controls(ui: &mut egui::Ui) {
    egui::CollapsingHeader::new(constants::VIEW_EDIT_BLOCK_CONTROLS)
        .default_open(false)
        .show(ui, |ui| {
            for line in CONTROLS {
                ui.monospace(line);
            }
        });
}

/// The floating number boxes over the viewport: what the drag in progress is
/// measuring, and how far the object is off the floor of the domain.
///
/// Both are anchored to the projected 3D point they measure and kept inside the
/// 3D view, so neither can land under the side panel however the object is
/// dragged. Clicking the measurement turns it into a text field; what is typed
/// there is applied as one undo step, and Escape or a click elsewhere leaves
/// the drag's own value alone.
pub fn callouts(ctx: &egui::Context, editor: &mut Editor) {
    if let Some(floor) = editor.floor_measure()
        && let Some(position) = editor.project(floor.anchor())
        && let Some(viewport) = editor.viewport_points()
    {
        let text = format!("floor {}", floor.label());
        egui::Area::new(egui::Id::new("growforge_editor_floor_callout"))
            .order(egui::Order::Foreground)
            .fixed_pos(placed(position, viewport))
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(egui::RichText::new(text).small());
                });
            });
    }

    let Some(callout) = editor.callout() else {
        return;
    };
    let Some(viewport) = editor.viewport_points() else {
        return;
    };
    let Some(position) = editor.project(callout.at.anchor()) else {
        return;
    };
    let label = callout.at.label();
    let typing = callout.phase() == Phase::Typing;
    // What the drag landed on, when it landed on something: a region that is
    // *on* the face of a pad is a different thing from one that is near it, and
    // the number alone cannot say which happened.
    let flush = editor.flush().map(|flush| flush.plane.what.label());
    let mut text = callout.text().unwrap_or_default().to_string();
    let field = egui::Id::new("growforge_editor_callout_field");
    let (mut clicked, mut commit, mut cancel) = (false, false, false);
    egui::Area::new(egui::Id::new("growforge_editor_callout"))
        .order(egui::Order::Foreground)
        .fixed_pos(placed(position, viewport))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                if typing {
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut text)
                            .id(field)
                            .desired_width(constants::VIEW_EDIT_CALLOUT_FIELD_WIDTH_POINTS),
                    );
                    // Enter commits; anything else that takes the focus away -
                    // Escape, a click elsewhere - cancels.
                    if response.lost_focus() {
                        if ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                            commit = true;
                        } else {
                            cancel = true;
                        }
                    }
                } else {
                    clicked = ui
                        .button(label)
                        .on_hover_text("click to type an exact value")
                        .clicked();
                }
                if let Some(flush) = &flush {
                    ui.label(egui::RichText::new(flush).small().weak());
                }
            });
        });

    if clicked {
        if let Some(callout) = editor.callout_mut() {
            callout.begin_typing();
        }
        ctx.memory_mut(|memory| memory.request_focus(field));
        return;
    }
    if commit {
        // A commit that is not a number changes nothing and puts the field
        // away; the value the drag left is still there to try again from.
        editor.commit_callout(&text);
        if let Some(callout) = editor.callout_mut() {
            callout.cancel_typing();
        }
        return;
    }
    if cancel {
        if let Some(callout) = editor.callout_mut() {
            callout.cancel_typing();
        }
        return;
    }
    if let Some(slot) = editor.callout_mut().and_then(|callout| callout.text_mut()) {
        *slot = text;
    }
}

/// Where a callout sits: beside the point it measures, and never so far right
/// or so far down that it lands under the side panel or off the window.
fn placed(position: egui::Pos2, viewport: (f32, f32)) -> egui::Pos2 {
    let offset = constants::VIEW_EDIT_CALLOUT_OFFSET_POINTS;
    let reserve = constants::VIEW_EDIT_CALLOUT_RESERVE_POINTS;
    egui::pos2(
        (position.x + offset).clamp(0.0, (viewport.0 - reserve[0]).max(0.0)),
        (position.y + offset).clamp(0.0, (viewport.1 - reserve[1]).max(0.0)),
    )
}

/// The unsaved-changes modal, shown when something that would leave a dirty
/// document behind arrives: a close request, or a switch to another file.
/// Returns what the user decided, if they decided anything.
///
/// One modal over all three, because the question is the same one every time.
/// The three answers are the only ones there are: nothing that leaves the
/// document behind ever silently writes the file or silently drops an edit.
pub fn guard_modal(
    ctx: &egui::Context,
    intent: &Intent,
    path: &str,
    running: bool,
) -> Option<CloseDecision> {
    let mut decision = None;
    egui::Modal::new(egui::Id::new("growforge_editor_guard")).show(ctx, |ui| {
        ui.heading("unsaved changes");
        ui.label(format!("{path} has edits that have not been written."));
        // Which file is about to take its place, when one is: "discard" means a
        // different thing to someone who cannot see where they are going.
        if let Some(next) = intent.destination() {
            ui.label(
                egui::RichText::new(format!("{} would be edited instead", next.display())).small(),
            );
        }
        if running {
            ui.label(
                egui::RichText::new(
                    "the run in progress will be stopped, and will not write a file",
                )
                .small(),
            );
        }
        ui.horizontal(|ui| {
            if ui.button(format!("save and {}", intent.verb())).clicked() {
                decision = Some(CloseDecision::Save);
            }
            if ui.button("discard").clicked() {
                decision = Some(CloseDecision::Discard);
            }
            if ui.button("cancel").clicked() {
                decision = Some(CloseDecision::Cancel);
            }
        });
    });
    decision
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::editor::tests::{fixture, growth_fixture, write_temp};

    /// Draw one frame of the panel into a context with no window behind it.
    ///
    /// egui needs neither a surface nor an adapter to lay a panel out, so the
    /// whole panel - every branch of it - can be exercised in a unit test. What
    /// this catches is what a smoke test of the window cannot: a panic in a
    /// properties branch nobody clicked on during the six seconds it was open.
    fn draw(editor: &mut Editor, scene: &mut Scene) {
        let context = egui::Context::default();
        let _ = context.run_ui(egui::RawInput::default(), |root| {
            editor.draw(root, scene, "test adapter");
        });
    }

    /// Draw every labelled block of the panel once, into the root ui of a fresh
    /// context, and hand back that context with the scope their headers hang
    /// under.
    ///
    /// egui identifies a collapsing header by the ui it is drawn in combined
    /// with the header's salt - its label, unless it names one - so the id of a
    /// header is only knowable to whoever knows that ui. Every block here opens
    /// its own header as the first thing it does in the ui it is handed, so
    /// drawing them into a root the test holds gives exactly the ids the panel
    /// draws with, without reproducing the panel's own nesting of scroll area
    /// and headers. The object lists are drawn a second time straight into the
    /// root for that reason: inside the tree they sit in the body its header
    /// opens, whose id is egui's own to invent.
    fn draw_blocks(editor: &mut Editor, scene: &mut Scene) -> (egui::Context, egui::Id) {
        let context = egui::Context::default();
        let mut scope = egui::Id::NULL;
        // The stress block is the one that needs a run behind it in the panel;
        // it is drawn here from a summary of its own, which is what it takes.
        let stress = StressSummary {
            warning: None,
            headline: "safety factor 5.07 (peak 9.2700 MPa vs yield 47 MPa)".to_string(),
            cases: vec!["tip peak 9.2700 MPa".to_string()],
        };
        let _ = context.run_ui(egui::RawInput::default(), |root| {
            scope = header_scope(root);
            let containment = editor.containment();
            {
                let mut field = Field::new(&mut editor.state, containment);
                tree(root, &mut field);
                object_lists(root, &mut field);
                properties(root, &mut field);
                sections(root, &mut field);
            }
            summary(root, &editor.state);
            show_block(root, scene);
            stress_block(root, &stress);
            controls(root);
        });
        (context, scope)
    }

    /// The id every collapsing header drawn straight into `ui` hangs under.
    ///
    /// `CollapsingHeader::show` lays its header and body out inside a vertical
    /// scope of the ui it was handed, and it is that scope - not the ui itself -
    /// that the header's id is salted from. egui gives every sibling scope of
    /// one parent the same stable id, so opening an empty one and asking it is
    /// egui's own arithmetic rather than a second spelling of it here.
    fn header_scope(ui: &mut egui::Ui) -> egui::Id {
        ui.vertical(|ui| ui.id()).inner
    }

    /// The id of the header salted `salt`, under that scope.
    ///
    /// `CollapsingHeader` salts with an `IdSalt` rather than with the text
    /// itself - `IdSalt::new` of its label, or of whatever `id_salt` it was
    /// given - and the two hash differently, so the salt is built the same way
    /// here rather than handed over as a string.
    fn header_id(scope: egui::Id, salt: &str) -> egui::Id {
        scope.with(egui::IdSalt::new(salt))
    }

    /// Whether the header salted `salt`, under the scope `scope` names, is open.
    ///
    /// Panics when nothing was drawn at that id, which is the only way to tell
    /// a closed block from an id that names no block at all: both would
    /// otherwise read as "not open".
    fn header_is_open(context: &egui::Context, scope: egui::Id, salt: &str) -> bool {
        egui::collapsing_header::CollapsingState::load(context, header_id(scope, salt))
            .unwrap_or_else(|| panic!("no collapsing header was drawn at the id salted {salt:?}"))
            .is_open()
    }

    /// Lay the whole panel out in a window the size the editor opens at, with
    /// every section of it expanded, and return how far its content spilled
    /// past the width the panel declares.
    ///
    /// `set_everything_is_visible` is egui's own switch for exactly this: it
    /// opens every collapsing header - the five object lists, `problem`,
    /// `controls`, `material` and `output` are all closed by default, and their
    /// rows would otherwise never be laid out at all - along with every dropdown
    /// and every hover text. Those last two float in areas of their own and so
    /// cannot widen the panel either way; the blocks are the point.
    ///
    /// Every block of the panel is inside a header of its own, so its rows are
    /// laid out one indent deeper than they were drawn before there was one, and
    /// a list's rows two: what this measures is each row at the width its own
    /// depth leaves it.
    ///
    /// Twice, because the first pass is what leaves the scroll area and the
    /// headers the state the second lays out against.
    fn panel_overflow(editor: &mut Editor, scene: &mut Scene) -> f32 {
        let context = egui::Context::default();
        context.memory_mut(|memory| memory.set_everything_is_visible(true));
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(
                    f32::from(constants::VIEW_WINDOW_WIDTH as u16),
                    f32::from(constants::VIEW_WINDOW_HEIGHT as u16),
                ),
            )),
            ..Default::default()
        };
        let mut overflow = 0.0;
        for _ in 0..2 {
            let _ = context.run_ui(input.clone(), |root| {
                overflow = panel(root, editor, scene, "test adapter");
            });
        }
        overflow
    }

    /// Every row of the panel has to fit the width the panel declares.
    ///
    /// A row that does not fit is not clipped and forgotten: the frame reports
    /// the width the content really took, egui finds it wider than
    /// `VIEW_EDIT_PANEL_WIDTH_POINTS`, pulls the panel rect back to that width
    /// by sliding its *left* edge inwards, and paints the panel's separator at
    /// the edge it has just moved - a full height line straight over the
    /// contents, on the parent's unclipped painter. That is what a seventh
    /// shape button did to the `add` row, and the first row to overflow widens
    /// the region every row after it is laid out in, so one row that does not
    /// fit takes the whole panel with it.
    ///
    /// The state driven here is the widest one there is: every kind of object
    /// in the tree, every shape as the selection - the properties of a triangle
    /// or a cone are the longest rows a shape has - every load kind, every
    /// optional table pinned so its rows exist, and both engines, since the
    /// `growth` section only belongs to one of them.
    #[test]
    fn no_row_of_the_panel_is_wider_than_the_panel() {
        for (name, text) in [
            ("panel_width", fixture()),
            ("panel_width_growth", growth_fixture()),
        ] {
            let (_dir, path) = write_temp(name, text);
            let mut editor = Editor::open(&path).expect("open");
            let mut scene = editor.initial_scene();
            // One of every object kind, so every add row and every list is
            // drawn with something in it.
            editor.state.add(NewObject::Load(0, LoadKind::Torque));
            editor.state.add(NewObject::Load(0, LoadKind::Gravity));
            editor.state.add(NewObject::LoadCase);
            editor.state.edit(|config| {
                // Every optional table pinned, and every key inside it, so that
                // the rows that only exist while the table does are laid out.
                config.material.preset = None;
                config.material.youngs_modulus_mpa = Some(2000.0);
                config.material.poisson_ratio = Some(0.35);
                config.material.density_g_cm3 = Some(1.1);
                config.material.yield_strength_mpa = Some(30.0);
                config.optimization.overhang = Some(OverhangConfig {
                    build_direction: constants::VIEW_EDIT_DEFAULT_BUILD_DIRECTION,
                });
                config.optimization.wireframe = Some(WireframeConfig {
                    radius_mm: Some(3.0),
                    hold_iterations: Some(12),
                    seed_density: Some(0.8),
                });
                config.optimization.local_volume = Some(LocalVolumeConfig {
                    max_fraction: Some(0.55),
                    radius_mm: Some(9.0),
                });
                config.output.trim = Some(TrimPolicy::Stress);
                config.output.trim_stress_fraction = Some(0.02);
                config.output.reinforce = Some(ReinforcePolicy::MinThickness);
                config.output.reinforce_thickness_mm = Some(2.0);
                config.output.stress_json = Some("fixture.stress.json".to_string());
                config.growth = Some(GrowthConfig {
                    seed: Some(7),
                    attractor_per_cm3: Some(30.0),
                    attraction_radius_mm: Some(9.0),
                    kill_radius_mm: Some(3.0),
                    step_mm: Some(1.0),
                    murray_exponent: Some(3.0),
                    max_radius_mm: Some(6.0),
                    max_steps: Some(500),
                    prune: Some(true),
                    symmetry: Some(SymmetryConfig {
                        kind: SymmetryKind::Mirror,
                        planes: Some(vec![Axis::X, Axis::Y]),
                        order: None,
                        axis: None,
                    }),
                });
            });

            // Every selection the panel has properties for, and every shape a
            // selected object can be - the rotation key present and absent, and
            // a tube bent, because those rows exist only in one of the two
            // states.
            let mut states: Vec<(String, Selection, Option<ShapeSpec>)> = vec![
                ("domain".to_string(), Selection::Domain(0), None),
                ("keepin".to_string(), Selection::Keepin(0), None),
                ("support".to_string(), Selection::Support(0), None),
                ("load case".to_string(), Selection::LoadCase(0), None),
                (
                    "force".to_string(),
                    Selection::Load { case: 0, load: 0 },
                    None,
                ),
                (
                    "torque".to_string(),
                    Selection::Load { case: 0, load: 1 },
                    None,
                ),
                (
                    "gravity".to_string(),
                    Selection::Load { case: 0, load: 2 },
                    None,
                ),
            ];
            for (label, shape) in shapes_in_every_state() {
                states.push((label, Selection::Keepout(1), Some(shape)));
            }
            for (label, selection, shape) in states {
                if let Some(shape) = shape {
                    editor
                        .state
                        .edit(|config| config.keepout[1] = shape.clone());
                }
                editor.state.select(Some(selection));
                let overflow = panel_overflow(&mut editor, &mut scene);
                assert!(
                    overflow <= 0.0,
                    "{name} with {label} selected laid the panel out {overflow} points past \
                     the {} it declares",
                    constants::VIEW_EDIT_PANEL_WIDTH_POINTS
                );
            }

            // And the rotational symmetry, whose rows replace the mirror's.
            editor.state.edit(|config| {
                if let Some(growth) = config.growth.as_mut() {
                    growth.symmetry = Some(SymmetryConfig {
                        kind: SymmetryKind::Rotational,
                        planes: None,
                        order: Some(6),
                        axis: Some(Axis::Z),
                    });
                }
                // The resolution given the other way, which is the other row.
                config.resolution.voxel_size_mm = None;
                config.resolution.target_cells = Some(200_000);
            });
            let overflow = panel_overflow(&mut editor, &mut scene);
            assert!(
                overflow <= 0.0,
                "{name} with a rotational symmetry laid the panel out {overflow} points past \
                 the {} it declares",
                constants::VIEW_EDIT_PANEL_WIDTH_POINTS
            );

            // A configuration that does not build: the validation block above
            // the sections is a row of the panel too.
            editor
                .state
                .edit(|config| config.optimization.mass_fraction = 4.0);
            editor.state.revalidate();
            assert!(!editor.state.is_valid());
            let overflow = panel_overflow(&mut editor, &mut scene);
            assert!(
                overflow <= 0.0,
                "{name} while invalid laid the panel out {overflow} points past the {} it \
                 declares",
                constants::VIEW_EDIT_PANEL_WIDTH_POINTS
            );
        }
    }

    /// Every shape a selected object can be, in every state its rows have.
    fn shapes_in_every_state() -> Vec<(String, ShapeSpec)> {
        let mut shapes = vec![
            (
                "a cylinder".to_string(),
                ShapeSpec::Cylinder {
                    p1: [46.0, 16.0, 16.0],
                    p2: [54.0, 16.0, 16.0],
                    radius: 4.0,
                },
            ),
            (
                "a sphere".to_string(),
                ShapeSpec::Sphere {
                    center: [50.0, 16.0, 16.0],
                    radius: 4.0,
                },
            ),
            (
                "a cone".to_string(),
                ShapeSpec::Cone {
                    p1: [46.0, 16.0, 16.0],
                    p2: [54.0, 16.0, 16.0],
                    radius1: 4.0,
                    radius2: 2.0,
                },
            ),
            (
                "a triangle".to_string(),
                ShapeSpec::Triangle {
                    a: [46.0, 12.0, 16.0],
                    b: [54.0, 12.0, 16.0],
                    c: [50.0, 20.0, 16.0],
                    thickness: 3.0,
                },
            ),
        ];
        // The rotation rows exist only while the key does, and the bend rows -
        // and the button that clears them - only while a tube has one.
        for rotation_deg in [None, Some([12.0, 34.0, 56.0])] {
            shapes.push((
                format!("a box turned {rotation_deg:?}"),
                ShapeSpec::Box {
                    min: [46.0, 12.0, 12.0],
                    max: [54.0, 20.0, 20.0],
                    rotation_deg,
                },
            ));
            shapes.push((
                format!("an ellipsoid turned {rotation_deg:?}"),
                ShapeSpec::Ellipsoid {
                    center: [50.0, 16.0, 16.0],
                    radii: [8.0, 4.0, 4.0],
                    rotation_deg,
                },
            ));
        }
        for bend in [None, Some([50.0, 16.0, 22.0])] {
            shapes.push((
                format!("a tube bent {bend:?}"),
                ShapeSpec::Tube {
                    p1: [46.0, 16.0, 16.0],
                    p2: [54.0, 16.0, 16.0],
                    bend,
                    radius: 3.0,
                },
            ));
        }
        shapes
    }

    /// What a launched session finds open, block by block.
    ///
    /// The panel opens on what a session works in - the objects, their
    /// properties, the layer switches and the safety factor of what was last
    /// run - and with everything else folded away: the object lists themselves,
    /// which are the longest thing in the panel, the problem read-out and the
    /// control legend. The scalar sections keep the defaults they already had,
    /// which is what pins them here.
    ///
    /// Every one of these is one line of code, and every one of them is a
    /// decision about what the panel looks like when it opens; a test is the
    /// only thing that notices when one of them is changed by accident.
    #[test]
    fn the_panel_opens_on_the_blocks_a_session_works_in() {
        let (_dir, path) = write_temp("panel_defaults", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        // The whole panel first, as a launched session draws it: the blocks
        // asked about below are the ones this frame is made of.
        draw(&mut editor, &mut scene);

        let (context, scope) = draw_blocks(&mut editor, &mut scene);
        for (salt, open) in [
            (constants::VIEW_EDIT_BLOCK_OBJECTS, true),
            (constants::VIEW_EDIT_BLOCK_PROPERTIES, true),
            (constants::VIEW_LAYER_SWITCHES_LABEL, true),
            (constants::VIEW_EDIT_BLOCK_STRESS, true),
            (constants::VIEW_EDIT_BLOCK_PROBLEM, false),
            (constants::VIEW_EDIT_BLOCK_CONTROLS, false),
            // The five lists of the tree, all folded away.
            ("domain", false),
            ("keepout", false),
            ("keepin", false),
            ("supports", false),
            (constants::VIEW_EDIT_LOAD_CASE_LIST, false),
            // And the scalar sections, whose own defaults nothing here changed.
            ("engine", true),
            ("resolution", true),
            ("material", false),
            ("optimization", true),
            ("output", false),
        ] {
            assert_eq!(
                header_is_open(&context, scope, salt),
                open,
                "the {salt:?} block opens {}",
                if open { "closed" } else { "open" }
            );
        }

        // The growth section is engine gated, so it is pinned on a
        // configuration that has one.
        let (_dir, path) = write_temp("panel_defaults_growth", growth_fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        let (context, scope) = draw_blocks(&mut editor, &mut scene);
        assert!(header_is_open(&context, scope, "growth"));
    }

    /// A list the user opened stays open when something is added to it or taken
    /// out of it.
    ///
    /// Its label carries a live count, and egui identifies a collapsing header
    /// by its label text unless the header names a salt of its own - so without
    /// the salt, `keepout (2)` and `keepout (3)` are two different headers, and
    /// the second one has never been opened. A list that shuts itself every time
    /// an object is added is worse than one that never opens: the state the user
    /// put it in is thrown away by the very action they opened it to take.
    #[test]
    fn an_object_list_stays_open_when_its_count_changes() {
        let (_dir, path) = write_temp("panel_list_ids", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let context = egui::Context::default();
        let mut scope = egui::Id::NULL;
        let lists = ["keepout", constants::VIEW_EDIT_LOAD_CASE_LIST];

        // A first frame, which is what leaves the headers' state behind.
        let _ = context.run_ui(egui::RawInput::default(), |ui| {
            scope = header_scope(ui);
            let mut field = Field::new(&mut editor.state, true);
            object_lists(ui, &mut field);
        });

        // Opened the way a click on the header opens them.
        for salt in lists {
            assert!(
                !header_is_open(&context, scope, salt),
                "{salt} did not start closed"
            );
            let mut state =
                egui::collapsing_header::CollapsingState::load(&context, header_id(scope, salt))
                    .expect("a header");
            state.set_open(true);
            state.store(&context);
        }

        // What renames both headers: one object more in each list.
        editor.state.add(NewObject::Keepout(ShapeKind::Sphere));
        editor.state.add(NewObject::LoadCase);
        let _ = context.run_ui(egui::RawInput::default(), |ui| {
            let mut field = Field::new(&mut editor.state, true);
            object_lists(ui, &mut field);
        });

        for salt in lists {
            assert!(
                header_is_open(&context, scope, salt),
                "the {salt} list shut itself when its count changed: its header is identified by \
                 a label that carries that count"
            );
        }
    }

    #[test]
    fn the_panel_draws_every_selection_and_every_section() {
        let (_dir, path) = write_temp("panel", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        // Nothing selected, then every kind of object in turn, including the
        // load kinds whose properties are all different.
        let mut selections = vec![
            None,
            Some(Selection::Domain(0)),
            Some(Selection::Keepout(0)),
            Some(Selection::Keepout(1)),
            Some(Selection::Keepin(0)),
            Some(Selection::Support(0)),
            Some(Selection::LoadCase(0)),
            Some(Selection::Load { case: 0, load: 0 }),
        ];
        editor.state.add(NewObject::Load(0, LoadKind::Torque));
        editor.state.add(NewObject::Load(0, LoadKind::Gravity));
        selections.push(Some(Selection::Load { case: 0, load: 1 }));
        selections.push(Some(Selection::Load { case: 0, load: 2 }));
        // And a selection that no longer addresses anything, which is what an
        // undo under a stale panel looks like.
        selections.push(Some(Selection::Keepout(99)));

        for selection in selections {
            editor.state.select(selection);
            draw(&mut editor, &mut scene);
        }

        // The ellipsoid's own properties, with the rotation key absent and
        // present: three centre fields, three radii and the checkbox.
        editor.state.select(Some(Selection::Keepout(1)));
        for rotation_deg in [None, Some([0.0, 0.0, 30.0])] {
            editor.state.edit(|config| {
                config.keepout[1] = ShapeSpec::Ellipsoid {
                    center: [50.0, 16.0, 16.0],
                    radii: [8.0, 4.0, 4.0],
                    rotation_deg,
                };
            });
            draw(&mut editor, &mut scene);
        }

        // The tube's own properties, straight and bent: the bend rows and the
        // straighten button only exist while there is a bend to clear.
        for bend in [None, Some([50.0, 16.0, 22.0])] {
            editor.state.edit(|config| {
                config.keepout[1] = ShapeSpec::Tube {
                    p1: [46.0, 16.0, 16.0],
                    p2: [54.0, 16.0, 16.0],
                    bend,
                    radius: 3.0,
                };
            });
            draw(&mut editor, &mut scene);
        }

        // The cone's own properties, tapered and at its apex: the second radius
        // is a row like any other, and zero is a value it may hold.
        for radius2 in [2.0, 0.0] {
            editor.state.edit(|config| {
                config.keepout[1] = ShapeSpec::Cone {
                    p1: [46.0, 16.0, 16.0],
                    p2: [54.0, 16.0, 16.0],
                    radius1: 4.0,
                    radius2,
                };
            });
            draw(&mut editor, &mut scene);
        }

        // And the triangle's three corners and its thickness.
        editor.state.edit(|config| {
            config.keepout[1] = ShapeSpec::Triangle {
                a: [46.0, 12.0, 16.0],
                b: [54.0, 12.0, 16.0],
                c: [50.0, 20.0, 16.0],
                thickness: 3.0,
            };
        });
        draw(&mut editor, &mut scene);

        // The custom material branch, and the section that only exists for the
        // growth engine.
        editor.state.edit(|config| {
            config.material.preset = None;
            config.material.youngs_modulus_mpa = Some(2000.0);
            config.material.poisson_ratio = Some(0.35);
            config.material.density_g_cm3 = Some(1.1);
        });
        draw(&mut editor, &mut scene);

        // The guide wireframe rows, which only exist once the table does: with
        // every key on its default and with every one of them pinned.
        for wireframe in [
            WireframeConfig {
                radius_mm: None,
                hold_iterations: None,
                seed_density: None,
            },
            WireframeConfig {
                radius_mm: Some(3.0),
                hold_iterations: Some(12),
                seed_density: Some(0.8),
            },
        ] {
            editor
                .state
                .edit(|config| config.optimization.wireframe = Some(wireframe.clone()));
            draw(&mut editor, &mut scene);
        }
        editor
            .state
            .edit(|config| config.optimization.wireframe = None);
        draw(&mut editor, &mut scene);

        // The local volume rows, the same way: the bare table with every key on
        // its default - the radius one derived from min_feature_mm - and every
        // key pinned.
        for local_volume in [
            LocalVolumeConfig {
                max_fraction: None,
                radius_mm: None,
            },
            LocalVolumeConfig {
                max_fraction: Some(0.55),
                radius_mm: Some(9.0),
            },
        ] {
            editor
                .state
                .edit(|config| config.optimization.local_volume = Some(local_volume.clone()));
            draw(&mut editor, &mut scene);
        }
        editor.state.edit(|config| {
            config.optimization.local_volume = None;
            config.optimization.penalty = Some(4.0);
        });
        draw(&mut editor, &mut scene);

        // A growth configuration that pins no `[growth]` key at all still has
        // to draw the section those keys live in.
        let (_dir, path) = write_temp("panel_growth", growth_fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        assert!(editor.state.config().is_growth() && editor.state.config().growth.is_none());
        editor.state.select(Some(Selection::Support(0)));
        draw(&mut editor, &mut scene);

        // Both kinds of symmetry, each with its own rows: a mirror with one
        // plane and with two, and a rotation.
        for symmetry in [
            SymmetryConfig {
                kind: SymmetryKind::Mirror,
                planes: Some(vec![Axis::X]),
                order: None,
                axis: None,
            },
            SymmetryConfig {
                kind: SymmetryKind::Mirror,
                planes: Some(vec![Axis::X, Axis::Y]),
                order: None,
                axis: None,
            },
            SymmetryConfig {
                kind: SymmetryKind::Rotational,
                planes: None,
                order: Some(6),
                axis: Some(Axis::Z),
            },
        ] {
            editor.state.edit(|config| {
                config.growth = Some(GrowthConfig {
                    seed: None,
                    attractor_per_cm3: None,
                    attraction_radius_mm: None,
                    kill_radius_mm: None,
                    step_mm: None,
                    murray_exponent: None,
                    max_radius_mm: None,
                    max_steps: None,
                    prune: None,
                    symmetry: Some(symmetry.clone()),
                });
            });
            draw(&mut editor, &mut scene);
        }
    }

    /// The output section's own rows, drawn directly.
    ///
    /// The `output` header is closed by default, so a smoke draw of the whole
    /// panel never reaches inside it. Every state the trim rows have is drawn
    /// here instead: the policy absent and on both of its values, and the
    /// fraction absent and pinned - a row that exists only while the pass does.
    #[test]
    fn the_output_section_draws_the_trim_rows_in_every_state() {
        let (_dir, path) = write_temp("panel_output", fixture());
        let mut editor = Editor::open(&path).expect("open");
        for (trim, fraction) in [
            (None, None),
            (Some(TrimPolicy::Off), None),
            (Some(TrimPolicy::Stress), None),
            (Some(TrimPolicy::Stress), Some(0.02)),
        ] {
            editor.state.edit(|config| {
                config.output.trim = trim;
                config.output.trim_stress_fraction = fraction;
            });
            let context = egui::Context::default();
            let _ = context.run_ui(egui::RawInput::default(), |root| {
                let mut field = Field::new(&mut editor.state, true);
                output_section(root, &mut field);
            });
        }
    }

    /// The reinforcement rows, in every state they have: the policy absent and
    /// on both of its values, and the floor absent and pinned - a row that
    /// exists only while the pass does, and whose shown default is this
    /// configuration's own `min_feature_mm` rather than a constant.
    #[test]
    fn the_output_section_draws_the_reinforce_rows_in_every_state() {
        let (_dir, path) = write_temp("panel_reinforce", fixture());
        let mut editor = Editor::open(&path).expect("open");
        assert_eq!(editor.state.config().output.reinforce, None);
        for (reinforce, thickness) in [
            (None, None),
            (Some(ReinforcePolicy::Off), None),
            (Some(ReinforcePolicy::MinThickness), None),
            (Some(ReinforcePolicy::MinThickness), Some(2.5)),
        ] {
            editor.state.edit(|config| {
                config.output.reinforce = reinforce;
                config.output.reinforce_thickness_mm = thickness;
            });
            let context = egui::Context::default();
            let _ = context.run_ui(egui::RawInput::default(), |root| {
                let mut field = Field::new(&mut editor.state, true);
                output_section(root, &mut field);
            });
            assert_eq!(editor.state.config().output.reinforce, reinforce);
            assert_eq!(
                editor.state.config().output.reinforce_thickness_mm,
                thickness
            );
        }
        assert_eq!(ReinforcePolicy::default(), ReinforcePolicy::Off);
        assert_eq!(ReinforcePolicy::Off.label(), "off");
        assert_eq!(ReinforcePolicy::MinThickness.label(), "min_thickness");
    }

    /// The boundary fidelity combo, in each of its states, and the value it
    /// leaves in the document.
    ///
    /// Absent is the state a configuration that never mentions the key is in,
    /// and it is not the same as `exact` written out: the combo has to show the
    /// default without inventing a key nobody asked for.
    #[test]
    fn the_output_section_draws_the_boundaries_combo_in_every_state() {
        let (_dir, path) = write_temp("panel_boundaries", fixture());
        let mut editor = Editor::open(&path).expect("open");
        assert_eq!(editor.state.config().output.boundaries, None);
        for boundaries in [
            None,
            Some(BoundaryFidelity::Exact),
            Some(BoundaryFidelity::Voxel),
        ] {
            editor
                .state
                .edit(|config| config.output.boundaries = boundaries);
            let context = egui::Context::default();
            let _ = context.run_ui(egui::RawInput::default(), |root| {
                let mut field = Field::new(&mut editor.state, true);
                output_section(root, &mut field);
            });
            assert_eq!(editor.state.config().output.boundaries, boundaries);
        }
        assert_eq!(BoundaryFidelity::default(), BoundaryFidelity::Exact);
        assert_eq!(BoundaryFidelity::Exact.label(), "exact");
        assert_eq!(BoundaryFidelity::Voxel.label(), "voxel");
    }

    /// Every section that carries a reset button, in the order the panel draws
    /// them.
    const SECTIONS: [Section; 6] = [
        Section::Engine,
        Section::Resolution,
        Section::Material,
        Section::Optimization,
        Section::Growth,
        Section::Output,
    ];

    /// The widget id the reset clicks below are attributed to. Any one will do:
    /// the interaction is opened and closed inside the call.
    const RESET_WIDGET: u64 = 0x9e5e7;

    /// One click of a section's reset button.
    ///
    /// The commit path a click reaches rather than the click itself: an egui
    /// button cannot be pressed in a headless frame any more than a dropdown can
    /// be opened in one - the reason the solver combo's test drives its writer
    /// directly - so what is driven here is everything the row does once egui
    /// reports the click. The row itself is laid out by the draws below.
    fn click_reset(editor: &mut Editor, section: Section) {
        reset_section(&mut editor.state, section, RESET_WIDGET);
    }

    /// One headless frame of one section's rows.
    fn draw_section(editor: &mut Editor, section: Section) {
        let context = egui::Context::default();
        let _ = context.run_ui(egui::RawInput::default(), |root| {
            let mut field = Field::new(&mut editor.state, true);
            match section {
                Section::Engine => engine_section(root, &mut field),
                Section::Resolution => resolution_section(root, &mut field),
                Section::Material => material_section(root, &mut field),
                Section::Optimization => optimization_section(root, &mut field),
                Section::Growth => growth_section(root, &mut field),
                Section::Output => output_section(root, &mut field),
            }
        });
    }

    /// Push every scalar section off its defaults in one undoable edit, so that
    /// each reset has something to put back and the other five have something
    /// visible to be left alone with.
    fn unsettle(editor: &mut Editor) {
        editor.state.edit(|config| {
            config.engine = Some(constants::GROWTH_ENGINE.to_string());
            config.solver = Some(SolverConfig {
                backend: Some(SolverBackend::Cpu),
                tolerance: Some(3e-8),
            });
            config.resolution.voxel_size_mm = None;
            config.resolution.target_cells = Some(9_999);
            config.material = MaterialConfig {
                preset: None,
                youngs_modulus_mpa: Some(2000.0),
                poisson_ratio: Some(0.35),
                density_g_cm3: Some(1.1),
                yield_strength_mpa: Some(40.0),
            };
            config.optimization = OptimizationConfig {
                mass_fraction: 0.42,
                min_feature_mm: 30.0,
                penalty: Some(4.0),
                max_iterations: Some(9),
                convergence_tol: Some(1e-3),
                update: Some(UpdateScheme::Mma),
                overhang: Some(OverhangConfig {
                    build_direction: BuildDirection::XPlus,
                }),
                wireframe: Some(WireframeConfig {
                    radius_mm: Some(3.0),
                    hold_iterations: Some(12),
                    seed_density: Some(0.8),
                }),
                local_volume: Some(LocalVolumeConfig {
                    max_fraction: Some(0.55),
                    radius_mm: Some(9.0),
                }),
            };
            config.growth = Some(GrowthConfig {
                seed: Some(3),
                attractor_per_cm3: Some(5.0),
                attraction_radius_mm: Some(20.0),
                kill_radius_mm: Some(6.0),
                step_mm: Some(2.0),
                murray_exponent: Some(2.5),
                max_radius_mm: Some(12.0),
                max_steps: Some(500),
                prune: Some(false),
                symmetry: Some(mirror_symmetry(Axis::Z)),
            });
            let stl_path = config.output.stl_path.clone();
            config.output = OutputConfig {
                stl_path,
                iso_level: Some(0.7),
                smoothing_iterations: Some(7),
                supersample: Some(2),
                voids: Some(VoidPolicy::Fill),
                islands: Some(IslandPolicy::Keep),
                boundaries: Some(BoundaryFidelity::Voxel),
                trim: Some(TrimPolicy::Stress),
                trim_stress_fraction: Some(0.02),
                reinforce: Some(ReinforcePolicy::MinThickness),
                reinforce_thickness_mm: Some(2.5),
                stress_json: Some("stress.json".to_string()),
            };
        });
    }

    /// What each of the six buttons puts back - and what it leaves alone.
    ///
    /// "The rest alone" is half the contract: a reset that reached into a
    /// neighbouring table would be a button nobody could trust with a
    /// configuration they had spent an hour on. The one exception is written
    /// into the engine's own reset, because a `[growth]` table under any other
    /// engine is a configuration that does not build.
    #[test]
    fn every_section_reset_writes_its_own_defaults_and_leaves_the_rest_alone() {
        let (_dir, path) = write_temp("reset_sections", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let voxel = editor
            .state
            .config()
            .resolution
            .voxel_size_mm
            .expect("the fixture names a voxel size");
        unsettle(&mut editor);
        let moved = editor.state.config().clone();

        // The engine: the key in either of its spellings, both keys of the
        // `[solver]` table, and the `[growth]` table the default engine may not
        // carry.
        click_reset(&mut editor, Section::Engine);
        {
            let config = editor.state.config();
            assert_eq!(config.engine, None);
            assert_eq!(config.project.engine, None);
            assert_eq!(
                config.engine_name().expect("an engine"),
                constants::DEFAULT_ENGINE
            );
            assert_eq!(
                config.solver,
                Some(SolverConfig {
                    backend: None,
                    tolerance: None
                }),
                "the [solver] keys are the engine section's own"
            );
            assert_eq!(
                config.growth, None,
                "leaving the growth engine has to take its table with it"
            );
            assert_eq!(config.resolution, moved.resolution);
            assert_eq!(config.material, moved.material);
            assert_eq!(config.optimization, moved.optimization);
            assert_eq!(config.output, moved.output);
        }
        assert!(editor.state.undo());

        // The resolution: exactly one of the two keys, whichever one was there.
        click_reset(&mut editor, Section::Resolution);
        {
            let config = editor.state.config();
            assert_eq!(
                config.resolution.voxel_size_mm,
                Some(constants::VIEW_EDIT_DEFAULT_VOXEL_MM)
            );
            assert_eq!(config.resolution.target_cells, None);
            assert_eq!(config.engine, moved.engine);
            assert_eq!(config.optimization, moved.optimization);
        }
        assert!(editor.state.undo());

        // The material: the default preset, and no custom value left beside it.
        click_reset(&mut editor, Section::Material);
        {
            let material = &editor.state.config().material;
            assert_eq!(
                material.preset.as_deref(),
                Some(constants::MATERIAL_PRESETS[0].key)
            );
            assert_eq!(material.youngs_modulus_mpa, None);
            assert_eq!(material.poisson_ratio, None);
            assert_eq!(material.density_g_cm3, None);
            assert_eq!(material.yield_strength_mpa, None);
        }
        assert!(editor.state.undo());

        // The optimization controls, with their two special cases: the mass
        // target is the user's and is left where it is, and the minimum feature
        // comes back to the smallest the density filter resolves - measured
        // here on the grid the last build derived, because this configuration
        // asks for a cell count rather than a voxel size.
        click_reset(&mut editor, Section::Optimization);
        {
            let optimization = &editor.state.config().optimization;
            assert_eq!(
                optimization.mass_fraction, moved.optimization.mass_fraction,
                "the mass target is design intent and has no default to return to"
            );
            assert_eq!(
                optimization.min_feature_mm,
                constants::VIEW_EDIT_RESET_MIN_FEATURE_VOXELS * voxel
            );
            assert_eq!(optimization.penalty, None);
            assert_eq!(optimization.max_iterations, None);
            assert_eq!(optimization.convergence_tol, None);
            assert_eq!(optimization.update, None);
            assert_eq!(optimization.overhang, None);
            assert_eq!(optimization.wireframe, None);
            assert_eq!(optimization.local_volume, None);
            assert_eq!(editor.state.config().growth, moved.growth);
        }
        assert!(editor.state.undo());

        // The growth table, symmetry included: every key of it is optional, so
        // "reset" is the table with nothing in it.
        click_reset(&mut editor, Section::Growth);
        assert_eq!(
            editor.state.config().growth,
            Some(GrowthConfig {
                seed: None,
                attractor_per_cm3: None,
                attraction_radius_mm: None,
                kill_radius_mm: None,
                step_mm: None,
                murray_exponent: None,
                max_radius_mm: None,
                max_steps: None,
                prune: None,
                symmetry: None,
            })
        );
        assert_eq!(editor.state.config().engine, moved.engine);
        assert!(editor.state.undo());

        // The output controls, and the two destinations that are not the
        // button's to decide: where a run writes is a decision about the
        // project rather than about the part.
        click_reset(&mut editor, Section::Output);
        {
            let output = &editor.state.config().output;
            assert_eq!(output.stl_path, moved.output.stl_path);
            assert_eq!(
                output.stress_json, moved.output.stress_json,
                "the stress report is written where the project says, not where a reset says"
            );
            assert!(output.stress_json.is_some(), "nothing was carried across");
            assert_eq!(output.iso_level, None);
            assert_eq!(output.smoothing_iterations, None);
            assert_eq!(output.supersample, None);
            assert_eq!(output.voids, None);
            assert_eq!(output.islands, None);
            assert_eq!(output.boundaries, None);
            assert_eq!(output.trim, None);
            assert_eq!(output.trim_stress_fraction, None);
            assert_eq!(output.reinforce, None);
            assert_eq!(output.reinforce_thickness_mm, None);
            // The absent policies are the documented defaults, which is what
            // makes taking the keys out the same thing as resetting them.
            assert_eq!(BoundaryFidelity::default(), BoundaryFidelity::Exact);
            assert_eq!(TrimPolicy::default(), TrimPolicy::Off);
            assert_eq!(ReinforcePolicy::default(), ReinforcePolicy::Off);
        }
        assert!(editor.state.undo());

        // And the minimum feature measured on a voxel size the configuration
        // names itself, which is the other of its two sources.
        click_reset(&mut editor, Section::Resolution);
        click_reset(&mut editor, Section::Optimization);
        assert_eq!(
            editor.state.config().optimization.min_feature_mm,
            constants::VIEW_EDIT_RESET_MIN_FEATURE_VOXELS * constants::VIEW_EDIT_DEFAULT_VOXEL_MM
        );
    }

    /// One click is one undo step, and that step puts the section back exactly
    /// as it was - every key of it, not the ones the button happened to change.
    #[test]
    fn one_undo_puts_a_reset_section_back_exactly_as_it_was() {
        for section in SECTIONS {
            let (_dir, path) = write_temp("reset_undo", fixture());
            let mut editor = Editor::open(&path).expect("open");
            unsettle(&mut editor);
            let before = editor.state.config().clone();
            let depth = editor.state.undo_depth();

            click_reset(&mut editor, section);
            assert_ne!(
                editor.state.config(),
                &before,
                "{section:?}: the reset changed nothing to undo"
            );
            assert!(editor.state.is_dirty());
            assert_eq!(
                editor.state.undo_depth(),
                depth + 1,
                "{section:?}: the reset is not one undo step"
            );

            assert!(editor.state.undo());
            assert_eq!(
                editor.state.config(),
                &before,
                "{section:?}: one undo did not put the configuration back"
            );
            assert_eq!(editor.state.undo_depth(), depth);
        }
    }

    /// The button is offered exactly while there is something to put back, and
    /// says so when there is not.
    #[test]
    fn a_section_already_at_its_defaults_offers_nothing_to_reset() {
        // The scaffold a new file opens on: two of its sections carry nothing
        // but defaults already, and two carry a choice the starter made.
        let (_dir, path) = write_temp("reset_scaffold", &crate::config::starter_config("scaffold"));
        let editor = Editor::open(&path).expect("open");
        assert!(
            at_defaults(&editor.state, Section::Material),
            "the starter names the default preset and nothing else"
        );
        assert!(
            at_defaults(&editor.state, Section::Output),
            "the starter writes no [output] key but the path"
        );
        assert!(
            !at_defaults(&editor.state, Section::Engine),
            "the starter pins the engine by name, in [project]"
        );
        assert!(
            !at_defaults(&editor.state, Section::Resolution),
            "the starter picks a voxel size of its own"
        );

        // Every section in turn: off its defaults, drawn, reset, drawn again.
        // A fresh session each time, because the engine's reset takes the growth
        // section off the panel altogether.
        for section in SECTIONS {
            let (_dir, path) = write_temp("reset_states", fixture());
            let mut editor = Editor::open(&path).expect("open");
            let mut scene = editor.initial_scene();
            unsettle(&mut editor);
            assert!(
                !at_defaults(&editor.state, section),
                "{section:?}: at its defaults after being moved off them"
            );
            draw_section(&mut editor, section);
            draw(&mut editor, &mut scene);

            click_reset(&mut editor, section);
            assert!(
                at_defaults(&editor.state, section),
                "{section:?}: not at its defaults after a reset of it"
            );
            draw_section(&mut editor, section);
            draw(&mut editor, &mut scene);
        }
    }

    /// A reset lands a configuration that still builds, from every state a
    /// section can be reset out of.
    ///
    /// Which is why the engine's reset clears the `[growth]` table: half of a
    /// coupled pair put back to its default is a file `growforge run` refuses,
    /// and a button that produces one is worse than no button. Both adversarial
    /// states are here - the growth engine with a table under it, and the local
    /// volume cap with the update scheme it requires - and every section is
    /// reset out of each of them.
    #[test]
    fn every_section_reset_lands_a_configuration_that_still_builds() {
        // The growth engine, with a full table and a symmetry under it.
        let (_dir, path) = write_temp("reset_valid_growth", growth_fixture());
        let mut editor = Editor::open(&path).expect("open");
        editor.state.edit(|config| {
            config.growth = Some(GrowthConfig {
                seed: Some(3),
                attractor_per_cm3: Some(5.0),
                attraction_radius_mm: Some(20.0),
                kill_radius_mm: Some(6.0),
                step_mm: Some(2.0),
                murray_exponent: Some(2.5),
                max_radius_mm: Some(12.0),
                max_steps: Some(500),
                prune: Some(false),
                symmetry: Some(mirror_symmetry(Axis::Z)),
            });
        });
        reset_every_section_and_check(&mut editor);

        // The local volume cap, which is legal only under the update scheme
        // that prices a second constraint, beside the other two sub-tables.
        let (_dir, path) = write_temp("reset_valid_simp", fixture());
        let mut editor = Editor::open(&path).expect("open");
        editor.state.edit(|config| {
            config.optimization.update = Some(UpdateScheme::Mma);
            config.optimization.local_volume = Some(LocalVolumeConfig {
                max_fraction: None,
                radius_mm: None,
            });
            config.optimization.overhang = Some(OverhangConfig {
                build_direction: constants::VIEW_EDIT_DEFAULT_BUILD_DIRECTION,
            });
            config.optimization.wireframe = Some(WireframeConfig {
                radius_mm: None,
                hold_iterations: None,
                seed_density: None,
            });
        });
        reset_every_section_and_check(&mut editor);
    }

    /// Reset each section out of the state `editor` is in, checking that what
    /// each one lands still builds, and putting the state back between them.
    fn reset_every_section_and_check(editor: &mut Editor) {
        editor.state.revalidate();
        assert!(
            editor.state.is_valid(),
            "the state being reset out of does not build: {:?}",
            editor.state.error()
        );
        for section in SECTIONS {
            click_reset(editor, section);
            editor.state.revalidate();
            assert!(
                editor.state.is_valid(),
                "{section:?}: the reset landed a configuration that does not build: {:?}",
                editor.state.error()
            );
            assert!(editor.state.undo());
            editor.state.revalidate();
        }
    }

    /// What a reset leaves in the file: the keys it took out are gone, the
    /// required ones are rewritten, and the text still parses to exactly the
    /// configuration on screen.
    ///
    /// A removed key takes its comment with it, which is the file layer's
    /// documented behaviour and is what a reset does six keys at a time.
    /// Everything the reset did not touch keeps its own comments, which is the
    /// half that matters most: the fixture's mass fraction is not the button's
    /// to change, so neither is the block above it.
    #[test]
    fn a_saved_reset_takes_the_keys_it_removed_out_of_the_file() {
        let commented = fixture().replace(
            "iso_level = 0.5",
            "# the density the surface is drawn at\niso_level = 0.5",
        );
        let (_dir, path) = write_temp("reset_save", &commented);
        let mut editor = Editor::open(&path).expect("open");
        assert_eq!(editor.state.config().output.iso_level, Some(0.5));

        click_reset(&mut editor, Section::Output);
        click_reset(&mut editor, Section::Optimization);
        editor.save();
        assert!(!editor.state.is_dirty());

        let saved = std::fs::read_to_string(&path).expect("read");
        assert!(!saved.contains("iso_level"), "{saved}");
        assert!(
            !saved.contains("the density the surface is drawn at"),
            "a removed key takes its comment with it, and that is all it takes: {saved}"
        );
        assert!(!saved.contains("max_iterations"), "{saved}");
        assert!(
            saved.contains("stl_path = \"fixture.stl\""),
            "the required key was not rewritten: {saved}"
        );
        assert!(
            saved.contains("min_feature_mm = 12"),
            "the derived minimum feature was not written: {saved}"
        );
        assert!(
            saved.contains("mass_fraction = 0.3") && saved.contains("# keep this comment"),
            "the reset reached a key it does not own: {saved}"
        );

        let reparsed = crate::config::Config::parse(&saved).expect("the saved file parses");
        assert_eq!(
            &reparsed,
            editor.state.config(),
            "the file and the configuration on screen came apart"
        );
    }

    /// The two ways a reset leaves a table behind, in the file itself: the
    /// engine's reset empties `[solver]` and removes `[growth]` outright.
    ///
    /// The pair is easy to invert, and inverting it costs either a comment block
    /// or a configuration that does not build - an empty `[solver]` resolves to
    /// exactly what no table does and is kept so that whatever was written above
    /// it still has a key to hang on, while a `[growth]` table under any engine
    /// but that one is refused outright. Only the file says which of the two
    /// really happened.
    #[test]
    fn an_engine_reset_empties_the_solver_table_and_removes_the_growth_one() {
        let text = format!(
            "{}\n# the solver notes\n[solver]\nbackend = \"cpu\"\ntolerance = 3e-8\n\n[growth]\n\
             seed = 5\nkill_radius_mm = 6.0\n\n[growth.symmetry]\nkind = \"mirror\"\n\
             planes = [\"x\"]\n",
            growth_fixture()
        );
        let (_dir, path) = write_temp("reset_save_tables", &text);
        let mut editor = Editor::open(&path).expect("open");
        assert!(editor.state.config().is_growth());
        assert!(editor.state.config().growth.is_some());
        assert!(editor.state.config().solver.is_some());

        click_reset(&mut editor, Section::Engine);
        editor.save();
        assert!(!editor.state.is_dirty());

        let saved = std::fs::read_to_string(&path).expect("read");
        assert!(
            saved.contains("[solver]"),
            "the emptied table was removed, and its comment block with it: {saved}"
        );
        assert!(saved.contains("# the solver notes"), "{saved}");
        let body = saved
            .split("[solver]")
            .nth(1)
            .expect("a body")
            .split("\n[")
            .next()
            .expect("a body");
        assert!(
            !body.contains('='),
            "the [solver] table kept a key: {body:?}"
        );
        assert!(
            !saved.contains("[growth"),
            "the [growth] table outlived the engine that made it legal: {saved}"
        );
        assert!(
            !saved.contains("engine ="),
            "the engine key was not taken out: {saved}"
        );

        let reparsed = crate::config::Config::parse(&saved).expect("the saved file parses");
        assert_eq!(
            reparsed.solver,
            Some(SolverConfig {
                backend: None,
                tolerance: None
            }),
            "an empty table has to parse back as an empty table"
        );
        assert_eq!(reparsed.growth, None);
        assert_eq!(
            reparsed.engine_name().expect("an engine"),
            constants::DEFAULT_ENGINE
        );
        assert_eq!(
            &reparsed,
            editor.state.config(),
            "the file and the configuration on screen came apart"
        );
    }

    /// A cell target that has never built has no voxel size to measure a
    /// minimum feature against, so the reset leaves that one key alone and does
    /// the rest of its work.
    ///
    /// The only honest answer: `min_feature_mm` has no default of its own, only
    /// a rule, and the rule needs a grid. Inventing one - the editor's own
    /// switching default, say - would put a length into the file that nothing
    /// about this configuration justifies.
    #[test]
    fn a_cell_target_that_never_built_leaves_the_minimum_feature_alone() {
        // A budget no machine could hold, so the problem never builds and the
        // grid is never laid out; the window opens on it all the same.
        let text = fixture().replace("voxel_size_mm = 4.0", "target_cells = 100_000_000_000");
        let (_dir, path) = write_temp("reset_never_built", &text);
        let mut editor = Editor::open(&path).expect("open");
        assert!(
            editor.state.problem().is_none(),
            "the configuration built after all, and the branch is not the one under test"
        );
        assert_eq!(
            effective_voxel_mm(&editor.state),
            None,
            "a voxel size was found where there is none"
        );

        editor.state.edit(|config| {
            config.optimization.penalty = Some(4.0);
            config.optimization.max_iterations = Some(9);
            config.optimization.wireframe = Some(WireframeConfig {
                radius_mm: Some(3.0),
                hold_iterations: None,
                seed_density: None,
            });
        });
        let before = editor.state.config().optimization.min_feature_mm;

        click_reset(&mut editor, Section::Optimization);
        let optimization = &editor.state.config().optimization;
        assert_eq!(
            optimization.min_feature_mm, before,
            "a minimum feature was derived from a grid that does not exist"
        );
        // The rest of the section reset all the same: the click did something.
        assert_eq!(optimization.penalty, None);
        assert_eq!(optimization.max_iterations, None);
        assert_eq!(optimization.wireframe, None);
    }

    /// The solver combo shows the backend a run of this configuration would
    /// use, which for a configuration that names none is not the enum's own
    /// default.
    ///
    /// It used to read the key through `Option::unwrap_or_default`, so a file
    /// with no `[solver]` table showed `cpu` while every run under it asked for
    /// the compute device. Nothing was written by that - but clicking `gpu` to
    /// correct the display wrote the key, and a backend named in the file is an
    /// instruction that forfeits the soft fall back the default has.
    #[test]
    fn the_solver_combo_shows_the_backend_a_run_would_use() {
        let (_dir, path) = write_temp("panel_solver", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        // The fixture names no backend at all, which is the case that was wrong.
        assert!(editor.state.config().solver.is_none());
        let resolved = editor
            .state
            .config()
            .solver_params()
            .expect("a valid configuration resolves its backend")
            .backend;
        assert_eq!(shown_backend(editor.state.config()), resolved);
        draw(&mut editor, &mut scene);

        // On a build that has the device that is the device - and it is not the
        // enum's derived default, which is what the panel used to show.
        #[cfg(feature = "gpu")]
        {
            assert_eq!(resolved, constants::DEFAULT_SOLVER_BACKEND);
            assert_ne!(
                resolved,
                SolverBackend::default(),
                "the resolved default and the reference backend are the same again"
            );
        }

        // A table without the key inside it is the same absence.
        editor.state.edit(|config| {
            config.solver = Some(SolverConfig {
                backend: None,
                tolerance: None,
            })
        });
        assert_eq!(shown_backend(editor.state.config()), resolved);
        draw(&mut editor, &mut scene);

        // And a backend the file names is shown as it is written, both ways
        // round: what a click writes is what the next frame reads back.
        for backend in [SolverBackend::Cpu, SolverBackend::Gpu] {
            editor.state.edit(|config| {
                config.solver = Some(SolverConfig {
                    backend: Some(backend),
                    tolerance: None,
                })
            });
            assert_eq!(shown_backend(editor.state.config()), backend);
            draw(&mut editor, &mut scene);
        }
    }

    /// The tolerance row draws in both states, and the two keys of the table
    /// are independent: neither widget may take the other's key with it.
    ///
    /// The panel replaces the whole `[solver]` table whenever either changes -
    /// the two are one struct - so a tolerance a user pinned has to survive a
    /// click on the backend combo, which is the shape of bug that costs a key
    /// silently.
    #[test]
    fn the_solver_tolerance_row_draws_and_survives_the_backend_combo() {
        let (_dir, path) = write_temp("panel_tolerance", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        // Absent, which is the state the fixture is in: the row shows the
        // default and nothing is written.
        assert!(editor.state.config().solver.is_none());
        draw(&mut editor, &mut scene);
        assert!(editor.state.config().solver.is_none());
        assert_eq!(
            editor
                .state
                .config()
                .solver_params()
                .expect("resolve")
                .tolerance,
            constants::CG_RELATIVE_TOLERANCE
        );

        // Pinned, with no backend beside it: it draws, and it resolves.
        editor.state.edit(|config| {
            config.solver = Some(SolverConfig {
                backend: None,
                tolerance: Some(3e-8),
            })
        });
        draw(&mut editor, &mut scene);
        assert_eq!(
            editor
                .state
                .config()
                .solver_params()
                .expect("resolve")
                .tolerance,
            3e-8
        );

        // What the backend combo writes when it is clicked, with a tolerance
        // already pinned: the writer itself, because an egui dropdown cannot be
        // opened in a headless frame.
        editor
            .state
            .edit(|config| write_solver_backend(config, SolverBackend::Cpu));
        let table = editor.state.config().solver.clone().expect("a table");
        assert_eq!(
            table.tolerance,
            Some(3e-8),
            "the backend combo dropped a tolerance the user pinned"
        );
        assert_eq!(table.backend, Some(SolverBackend::Cpu));
        assert_eq!(shown_backend(editor.state.config()), SolverBackend::Cpu);
        draw(&mut editor, &mut scene);

        // And the other way: the tolerance row keeps the named backend, and
        // unticking it leaves the backend alone.
        editor
            .state
            .edit(|config| write_solver_tolerance(config, Some(2e-8)));
        let table = editor.state.config().solver.clone().expect("a table");
        assert_eq!(table.backend, Some(SolverBackend::Cpu));
        assert_eq!(table.tolerance, Some(2e-8));
        editor
            .state
            .edit(|config| write_solver_tolerance(config, None));
        let table = editor.state.config().solver.clone().expect("a table");
        assert_eq!(table.backend, Some(SolverBackend::Cpu));
        assert_eq!(table.tolerance, None);
        assert_eq!(
            editor
                .state
                .config()
                .solver_params()
                .expect("resolve")
                .tolerance,
            constants::CG_RELATIVE_TOLERANCE,
            "an unticked key has to resolve to the default again"
        );
        draw(&mut editor, &mut scene);
    }

    /// What the post-run field passes did, in the panel, beside the switch that
    /// colours the surface by the very stresses one of them judged the material
    /// on.
    ///
    /// The console has always had somewhere to print a report; the editor's
    /// panel has not, and a pass that takes material off the part on screen - or
    /// puts material onto it - may not be something the user finds out about in
    /// a terminal behind the window. A full run is what produces one, so this
    /// runs one, with both passes on: what was freed and what was spent are one
    /// exchange and the panel draws them together.
    #[test]
    fn the_panel_shows_what_the_post_run_passes_did() {
        let trimming = growth_fixture().replace(
            "stl_path = \"fixture.stl\"",
            "stl_path = \"fixture.stl\"\ntrim = \"stress\"\nreinforce = \"min_thickness\"",
        );
        let (_dir, path) = write_temp("panel_trim", &trimming);
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        // Nothing has run, so there is nothing to say and nothing is said.
        assert!(editor.trim_notes().is_empty());
        assert!(editor.reinforce_notes().is_empty());
        draw(&mut editor, &mut scene);

        // Waited out rather than joined: joining collects the run and takes its
        // progress with it, and what the panel reads is the progress of the run
        // it is still showing.
        editor.start_full_run();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        while std::time::Instant::now() < deadline && editor.is_running() {
            editor.pump(&mut scene);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        editor.pump(&mut scene);
        assert!(!editor.is_running(), "the run never ended");
        assert!(
            !editor.trim_notes().is_empty(),
            "the trim ran and the panel was never told"
        );
        let reinforce = editor.reinforce_notes();
        assert!(
            !reinforce.is_empty(),
            "the reinforcement ran and the panel was never told"
        );
        assert!(
            reinforce.iter().all(|note| !note.is_empty()),
            "the panel was handed a blank line: {reinforce:?}"
        );
        // The boundary clamp is on by default, so the same block carries its
        // line too - the notes block renders every pass, not one of them.
        let clamp = editor.clamp_notes();
        assert!(
            !clamp.is_empty(),
            "the clamp ran and the panel was never told"
        );
        assert!(
            clamp.iter().all(|note| !note.is_empty()),
            "the panel was handed a blank line: {clamp:?}"
        );
        draw(&mut editor, &mut scene);
        editor.finish();
        std::fs::remove_file(path.with_file_name("fixture.stl")).ok();
    }

    /// "What is the safety factor?" - the number the editor computed after every
    /// run, painted into the stress layer, and never once said in text.
    ///
    /// The block describes the run the window is showing: absent before anything
    /// has run, present with the yield strength named once one has, and absent
    /// again the moment the next run starts, so the number on screen is never the
    /// one before it.
    #[test]
    fn the_panel_shows_the_safety_factor_of_the_run_it_is_showing() {
        let (_dir, path) = write_temp("panel_stress", growth_fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        // Nothing has run, so there is no block at all.
        assert!(editor.stress_summary().is_none());
        draw(&mut editor, &mut scene);

        // Waited out rather than joined, for the reason the trim test gives:
        // what the panel reads is the progress of the run it is showing.
        editor.start_full_run();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        while std::time::Instant::now() < deadline && editor.is_running() {
            editor.pump(&mut scene);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        editor.pump(&mut scene);
        assert!(!editor.is_running(), "the run never ended");

        let summary = editor
            .stress_summary()
            .expect("the run analysed the part and the panel was never told");
        assert!(
            summary.headline.starts_with("safety factor "),
            "the headline is not the safety factor: {}",
            summary.headline
        );
        assert!(
            summary.headline.contains("yield") && summary.headline.contains("MPa"),
            "the units and the yield strength are not named: {}",
            summary.headline
        );
        assert_eq!(summary.cases.len(), 1, "{:?}", summary.cases);
        assert!(
            summary.cases[0].starts_with("tip peak "),
            "the load case is not named: {:?}",
            summary.cases
        );
        draw(&mut editor, &mut scene);

        // The next run's block is its own, and it begins as nothing: a stale
        // safety factor over a design that is being replaced is worse than none.
        editor.start_full_run();
        assert!(
            editor.stress_summary().is_none(),
            "the panel kept the last run's safety factor"
        );
        draw(&mut editor, &mut scene);
        editor.stop_run();
        editor.finish();
        std::fs::remove_file(path.with_file_name("fixture.stl")).ok();
    }

    /// The stress block in the state a whole-panel draw cannot reach without a
    /// run that comes out in pieces: the warning above the safety factor, drawn
    /// in the warning colour the pass notes use.
    ///
    /// Both states are laid out, because the one that must not change is the
    /// single body one: an export in one piece draws exactly what it drew before
    /// there was a warning to draw.
    #[test]
    fn the_stress_block_draws_the_warning_an_export_in_pieces_carries() {
        let cases = vec!["tip peak 9.2700 MPa".to_string()];
        let whole = StressSummary {
            warning: None,
            headline: "safety factor 5.07 (peak 9.2700 MPa vs yield 47 MPa)".to_string(),
            cases: cases.clone(),
        };
        let pieces = StressSummary {
            warning: crate::stress::disconnected_bodies_note(2).map(|n| format!("warning: {n}")),
            ..whole.clone()
        };
        assert!(
            pieces
                .warning
                .as_ref()
                .is_some_and(|line| line.starts_with("warning")),
            "the colouring rule reads the first word: {:?}",
            pieces.warning
        );
        for summary in [whole, pieces] {
            let context = egui::Context::default();
            let _ = context.run_ui(egui::RawInput::default(), |root| {
                stress_block(root, &summary);
            });
        }
    }

    /// A run that was stopped analysed nothing, so there is nothing to say - and
    /// the panel says nothing rather than showing numbers of a design that was
    /// never analysed. Asking for that design's stl is what analyses it, and the
    /// block then describes exactly what was written.
    #[test]
    fn a_stopped_run_shows_no_safety_factor_until_its_stl_is_generated() {
        let (_dir, path) = write_temp("panel_stress_stopped", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        // A budget it cannot spend and a tolerance it cannot reach, so nothing
        // but the stop ends this run.
        editor.state.edit(|config| {
            config.optimization.max_iterations = Some(400);
            config.optimization.convergence_tol = Some(1e-9);
        });
        editor.start_full_run();

        // Stopped once the engine has reported an iteration, which is when there
        // is a design behind the window to export later.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        while std::time::Instant::now() < deadline
            && !editor.run_line().is_some_and(|line| line.contains("step"))
        {
            editor.pump(&mut scene);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(editor.is_running(), "the run ended on its own");
        editor.stop_run();
        assert!(
            editor.stress_summary().is_none(),
            "a run that never analysed anything showed a safety factor"
        );
        draw(&mut editor, &mut scene);
        assert!(editor.can_generate_stl(), "the stop threw the design away");

        editor.generate_stl();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        while std::time::Instant::now() < deadline && editor.is_running() {
            editor.pump(&mut scene);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        editor.pump(&mut scene);
        assert!(!editor.is_running(), "the generation never ended");
        let summary = editor
            .stress_summary()
            .expect("the generation analysed the part and the panel was never told");
        assert!(
            summary.headline.starts_with("safety factor "),
            "the headline is not the safety factor: {}",
            summary.headline
        );
        assert_eq!(summary.cases.len(), 1, "{:?}", summary.cases);
        draw(&mut editor, &mut scene);
        editor.finish();
        std::fs::remove_file(path.with_file_name("fixture.stl")).ok();
    }

    /// The panel while a tube is being placed: the toolbar's hint at both
    /// stages of it, the tree it was started from, and the preview the viewport
    /// draws alongside - the marker under the pointer and the band back to the
    /// point already clicked.
    #[test]
    fn the_panel_draws_a_placement_at_both_of_its_stages() {
        let (_dir, path) = write_temp("panel_placing", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.pump(&mut scene);
        assert!(editor.placement_hint().is_none());

        // Straight down at the lid of the domain, which is where both clicks
        // land here.
        let ray = crate::viewer::editor::pick::Ray {
            origin: [30.0, 20.0, 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        editor.toggle_placing(NewObject::Keepout(ShapeKind::Tube));
        let first = editor.placement_hint().expect("a hint");
        assert!(
            first.contains(constants::VIEW_EDIT_PLACE_HINT_FIRST),
            "{first}"
        );
        draw(&mut editor, &mut scene);

        // The marker under the pointer, before anything is clicked.
        editor.hover_to(Some(ray));
        editor.pump(&mut scene);
        assert!(
            scene
                .get(crate::viewer::scene::Layer::Placement)
                .is_some_and(|layer| layer.triangles() > 0)
        );
        draw(&mut editor, &mut scene);

        // And the band, once there is a point to draw it from.
        editor.release(Some(&ray), true);
        editor.hover_to(Some(crate::viewer::editor::pick::Ray {
            origin: [45.0, 20.0, 300.0],
            direction: [0.0, 0.0, -1.0],
        }));
        editor.pump(&mut scene);
        let second = editor.placement_hint().expect("a hint");
        assert!(
            second.contains(constants::VIEW_EDIT_PLACE_HINT_SECOND),
            "{second}"
        );
        assert!(
            scene
                .get(crate::viewer::scene::Layer::Placement)
                .is_some_and(|layer| layer.triangles() > 0)
        );
        draw(&mut editor, &mut scene);

        // The second click ends it, and the panel goes back to what it was.
        editor.release(
            Some(&crate::viewer::editor::pick::Ray {
                origin: [45.0, 20.0, 300.0],
                direction: [0.0, 0.0, -1.0],
            }),
            true,
        );
        editor.pump(&mut scene);
        assert!(editor.placement_hint().is_none());
        assert!(scene.get(crate::viewer::scene::Layer::Placement).is_none());
        draw(&mut editor, &mut scene);
    }

    /// "generate stl" is offered exactly when there is a design to export, so
    /// the toolbar has to draw both of its states - and drawing either of them
    /// must not write anything.
    #[test]
    fn the_toolbar_draws_generate_stl_enabled_and_disabled() {
        let (_dir, path) = write_temp("panel_generate", growth_fixture());
        let stl = path.with_file_name("fixture.stl");
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        // Nothing has been run, so there is nothing on screen to export.
        assert!(!editor.can_generate_stl());
        draw(&mut editor, &mut scene);

        // The preview an edit starts leaves its design behind, and the button
        // lights up on it.
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = 0.2);
        editor.on_edited();
        std::thread::sleep(std::time::Duration::from_secs_f64(
            constants::VIEW_EDIT_REFRESH_DEBOUNCE_S * 2.0,
        ));
        editor.pump(&mut scene);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        while std::time::Instant::now() < deadline && !editor.can_generate_stl() {
            editor.pump(&mut scene);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(
            editor.can_generate_stl(),
            "the preview left no design to export"
        );
        draw(&mut editor, &mut scene);

        editor.detach();
        editor.finish();
        assert!(!stl.exists(), "drawing the button wrote {}", stl.display());
    }

    /// The panel is where the increment and the containment rule are set, so
    /// what it puts in front of the user has to be what the drags then use.
    #[test]
    fn the_panel_carries_the_snap_increment_and_the_containment_switch() {
        let (_dir, path) = write_temp("precision_panel", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        assert_eq!(editor.snap().millimetres, constants::VIEW_EDIT_SNAP_MM);
        assert_eq!(editor.snap().degrees, constants::VIEW_EDIT_SNAP_DEGREES);
        assert_eq!(
            editor.containment(),
            constants::VIEW_EDIT_CONTAINMENT_DEFAULT
        );
        draw(&mut editor, &mut scene);

        // What the panel writes is what a drag reads, and the bypass key takes
        // it away for as long as it is held.
        for choice in constants::VIEW_EDIT_SNAP_CHOICES_MM {
            editor.snap_mut().millimetres = choice;
            assert_eq!(editor.active_snap().millimetres, choice);
            assert!(editor.snap().snaps_lengths());
            draw(&mut editor, &mut scene);
        }
        // Free entry: anything at all, including a number no dropdown offers.
        editor.snap_mut().millimetres = 0.375;
        assert!((editor.active_snap().length(0.4) - 0.375).abs() < 1e-12);
        editor.set_snap_bypass(true);
        assert!(!editor.active_snap().snaps_lengths());
        assert!((editor.active_snap().length(1.0) - 1.0).abs() < 1e-12);
        editor.set_snap_bypass(false);
        assert!(editor.active_snap().snaps_lengths());
        // Off, through the increment itself.
        editor.snap_mut().millimetres = 0.0;
        assert!(!editor.snap().snaps_lengths());
        draw(&mut editor, &mut scene);

        // The containment switch, and the note a clamped commit leaves.
        editor.set_containment(false);
        assert!(!editor.containment());
        assert!(editor.containment_note().is_none());
        editor.set_containment(true);
        editor.note_clamped();
        assert_eq!(
            editor.containment_note(),
            Some(constants::VIEW_EDIT_CONTAINMENT_NOTE)
        );
        draw(&mut editor, &mut scene);
        // Switching it off takes the note with it: it is describing a rule that
        // no longer applies.
        editor.set_containment(false);
        assert!(editor.containment_note().is_none());
    }

    /// A panel edit of a shape goes through the containment rule, exactly as a
    /// drag does, and says so.
    #[test]
    fn a_number_typed_into_the_properties_panel_is_kept_inside_the_domain() {
        let (_dir, path) = write_temp("panel_containment", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.state.select(Some(Selection::Keepin(0)));

        // What the panel's own commit path does with a box typed outside.
        let outside = ShapeSpec::Box {
            min: [500.0, 8.0, 8.0],
            max: [508.0, 24.0, 24.0],
            rotation_deg: None,
        };
        let containment = editor.containment();
        let clamped = {
            let mut field = Field::new(&mut editor.state, containment);
            let commit = set_shape_contained(
                field.state.config_mut(),
                Selection::Keepin(0),
                outside.clone(),
                field.containment,
            );
            field.clamped = commit;
            field.clamped
        };
        assert!(clamped);
        editor.note_clamped();
        assert!(editor.containment_note().is_some());
        let kept = shape_of(editor.state.config(), Selection::Keepin(0)).expect("a shape");
        let bounds = kept.to_shape("test").expect("a shape").bounds();
        assert!(bounds.max[0] <= 64.0 + 1e-9, "{bounds:?}");
        draw(&mut editor, &mut scene);
    }

    /// Every branch of the callout - lingering, being typed into - has to draw,
    /// and it has to stay inside the 3D view rather than sliding under the
    /// panel.
    #[test]
    fn the_callouts_draw_and_stay_off_the_side_panel() {
        let viewport = (900.0_f32, 600.0_f32);
        let reserve = constants::VIEW_EDIT_CALLOUT_RESERVE_POINTS;
        for point in [
            egui::pos2(0.0, 0.0),
            egui::pos2(450.0, 300.0),
            egui::pos2(2000.0, 2000.0),
            egui::pos2(-500.0, -500.0),
        ] {
            let placed = placed(point, viewport);
            assert!(placed.x >= 0.0 && placed.y >= 0.0, "{placed:?}");
            assert!(placed.x <= viewport.0 - reserve[0], "{placed:?}");
            assert!(placed.y <= viewport.1 - reserve[1], "{placed:?}");
        }
        // A viewport narrower than the space a callout needs still gives a
        // legal position rather than a negative one.
        assert_eq!(
            placed(egui::pos2(10.0, 10.0), (10.0, 10.0)),
            egui::pos2(0.0, 0.0)
        );

        let (_dir, path) = write_temp("callout_draw", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor.state.select(Some(Selection::Keepout(1)));
        editor.pump(&mut scene);
        // The projection a drawn frame hands over.
        let camera = crate::viewer::camera::OrbitCamera::default();
        editor.set_projection(camera.view_projection(1.5), (900.0, 600.0), 1.0);
        assert!(editor.viewport_points().is_some());

        let handle = editor
            .handles
            .iter()
            .find(|h| h.kind == crate::viewer::editor::gizmo::HandleKind::Translate(0))
            .copied()
            .expect("an x arrow");
        let press = crate::viewer::editor::pick::Ray {
            origin: [handle.position[0], handle.position[1], 300.0],
            direction: [0.0, 0.0, -1.0],
        };
        assert!(editor.press(&press));
        draw(&mut editor, &mut scene);
        editor.release(None, false);
        draw(&mut editor, &mut scene);
        editor.callout_mut().expect("a callout").begin_typing();
        draw(&mut editor, &mut scene);
        // A frame with nobody typing must not commit or cancel anything.
        assert_eq!(
            editor.callout().expect("a callout").phase(),
            crate::viewer::editor::measure::Phase::Typing
        );
    }

    #[test]
    fn a_numeric_field_refuses_a_number_that_is_not_one() {
        // What egui's own parser accepts, it still accepts.
        assert_eq!(finite_number("12.5"), Some(12.5));
        assert_eq!(finite_number(" 1 000.5 "), Some(1000.5));
        assert_eq!(
            finite_number("\u{2212}3"),
            Some(-3.0),
            "a typographic minus"
        );
        // What it accepts and the configuration cannot hold, it refuses, so the
        // field keeps the value it had.
        for text in ["nan", "NaN", "inf", "-inf", "infinity", "hello", ""] {
            assert_eq!(finite_number(text), None, "{text} was accepted");
        }
    }

    #[test]
    fn the_panel_draws_an_invalid_configuration_and_its_modal() {
        let (_dir, path) = write_temp("panel_invalid", fixture());
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();

        // A configuration that does not build must still draw: the panel is
        // where the user finds out why.
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = 4.0);
        editor.state.revalidate();
        assert!(!editor.state.is_valid());
        draw(&mut editor, &mut scene);

        // And the unsaved-changes modal draws on top of it.
        assert!(!editor.request_close());
        assert!(editor.is_asking());
        draw(&mut editor, &mut scene);
        assert!(
            editor.is_asking(),
            "a frame with nobody clicking must not decide anything"
        );
        assert!(!editor.may_close());
    }

    /// One modal asks about all three intents, so all three of its shapes have
    /// to draw - including the line that names where the session is going, which
    /// only two of them have.
    #[test]
    fn the_guard_modal_draws_for_every_intent_it_asks_about() {
        let (_dir, path) = write_temp("guard_modal", fixture());
        let elsewhere = path.with_file_name("elsewhere.toml");
        std::fs::write(&elsewhere, fixture()).expect("write the second file");
        let fresh = path.with_file_name("not_there_yet.toml");
        let mut editor = Editor::open(&path).expect("open");
        let mut scene = editor.initial_scene();
        editor
            .state
            .edit(|config| config.optimization.mass_fraction = 0.44);

        let intents = [
            Intent::CloseWindow,
            Intent::OpenFile(elsewhere.clone()),
            Intent::NewFile(fresh.clone()),
        ];
        for intent in intents {
            let asked = match &intent {
                Intent::CloseWindow => editor.request_close(),
                Intent::OpenFile(path) => editor.request_open(path.clone()),
                Intent::NewFile(path) => editor.request_new(path.clone()),
            };
            assert!(!asked, "unsaved edits must be asked about first");
            assert_eq!(editor.asking(), Some(&intent));
            draw(&mut editor, &mut scene);
            assert_eq!(
                editor.asking(),
                Some(&intent),
                "a frame with nobody clicking must not decide anything"
            );
            editor.decide(CloseDecision::Cancel);
            assert!(!editor.is_asking() && !editor.is_switching());
        }
        // Nothing was written, and the session is where it started.
        assert!(!fresh.exists());
        assert_eq!(editor.state.path(), path);
    }

    /// The legend is hand written, and this is the only thing that notices when
    /// a binding was added and never listed.
    #[test]
    fn the_control_legend_lists_the_file_shortcuts() {
        for binding in ["Ctrl+S", "Ctrl+O", "Ctrl+N"] {
            assert!(
                CONTROLS.iter().any(|line| line.starts_with(binding)),
                "the legend does not list {binding}: {CONTROLS:?}"
            );
        }
    }
}
