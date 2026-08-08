//! The egui side panel: layer visibility switches, the live stats block and
//! the control legend.

use crate::constants;
use crate::viewer::scene::{LAYERS, Layer, LayerRole, Scene, SwitchHome};
use crate::viewer::snapshot::{FrameKind, Progress, RunStatus};

/// What a switch says on hover while the layer behind it is empty. A layer with
/// nothing in it is switched off rather than hidden, so the panel keeps its
/// shape from frame to frame - and the checkbox has to say why it cannot be
/// ticked.
const ABSENT_LAYER_HELP: &str = "nothing has produced this layer yet";

/// Draw one layer's visibility switch: the checkbox, the layer's own hover text,
/// and the reason it cannot be ticked while the layer is empty.
///
/// One switch on its own, because not every one of them is drawn in this block:
/// a layer whose switch is homed elsewhere is drawn by whatever block owns it,
/// and it has to be the same checkbox, explained the same way, wherever it is.
pub fn layer_switch(ui: &mut egui::Ui, scene: &mut Scene, layer: Layer) -> egui::Response {
    let info = layer.info();
    let present = scene.get(layer).is_some();
    let visible = scene.visibility_mut(layer);
    ui.add_enabled(present, egui::Checkbox::new(visible, info.label))
        .on_hover_text(info.help)
        .on_disabled_hover_text(ABSENT_LAYER_HELP)
}

/// Draw the layer visibility switches and the two shading toggles.
///
/// Shared with the editor's panel, so the same scene is controlled the same way
/// whichever mode the window is in. Layers the editor alone produces are listed
/// only when `editor` is set; everywhere else they do not exist.
///
/// Layers whose switch is homed elsewhere - [`SwitchHome::Elsewhere`] - are left
/// out here whichever panel is drawing: the block that owns such a switch draws
/// it itself, and the table is what says which those are.
///
/// `editor` decides the block's own heading too: this panel writes it here,
/// while the editor's panel draws the block under a collapsing header that
/// carries the same word, and a heading inside that body would say it twice.
/// The one flag decides both because it is the one question - which of the two
/// panels is drawing.
///
/// Every switch carries the hover text of the layer it draws, which the layer
/// table holds beside the label. Both windows therefore explain the same
/// switches the same way.
pub fn layer_switches(ui: &mut egui::Ui, scene: &mut Scene, editor: bool) {
    if !editor {
        ui.label(egui::RichText::new(constants::VIEW_LAYER_SWITCHES_LABEL).strong());
    }
    for info in LAYERS.iter().filter(|info| {
        (editor || info.role != LayerRole::Editor) && info.switch == SwitchHome::Show
    }) {
        layer_switch(ui, scene, info.layer);
    }
    let has_stress = scene.has_stress();
    ui.add_enabled(
        has_stress,
        egui::Checkbox::new(scene.stress_shading_mut(), "colour by von Mises stress"),
    )
    .on_hover_text(
        "shade the exported mesh by the stress it was analysed at instead of plainly. The load \
         case with the highest peak is the one shown",
    )
    .on_disabled_hover_text("no run has produced a stress report yet");
    if has_stress {
        ui.label(
            egui::RichText::new("blue is unstressed, red is the yield strength")
                .small()
                .weak(),
        );
    }
    // Only the density surface is shaded smooth; the overlays are
    // diagrammatic and stay flat either way.
    let has_density = scene.get(Layer::Density).is_some();
    ui.add_enabled(
        has_density,
        egui::Checkbox::new(scene.flat_shading_mut(), "flat shading"),
    )
    .on_hover_text(
        "draw the density surface with one normal per triangle, which is the honest view of what \
         the mesh really is. It changes nothing about the geometry or the STL, and the overlays \
         keep their own shading either way",
    )
    .on_disabled_hover_text("there is no density surface to shade yet");
}

/// Short label for the stage a run is in.
fn status_label(status: &RunStatus) -> String {
    match status {
        RunStatus::Optimizing => "optimizing".to_string(),
        RunStatus::Analysing => "voids and stress".to_string(),
        RunStatus::Exporting => "exporting mesh".to_string(),
        RunStatus::Finished { stl_path, .. } => format!("finished, wrote {stl_path}"),
        RunStatus::Failed(error) => format!("failed: {error}"),
    }
}

/// Draw the side panel. `progress` is `None` in a setup view, where nothing is
/// running.
///
/// `engine` is the key of the engine behind the run, and it is here for one
/// line: an engine that never reports an iteration - the solid one - has to be
/// told apart from one whose first iteration has not landed yet, and the field
/// cannot say which it is because there is no field yet either.
pub fn panel(
    root: &mut egui::Ui,
    title: &str,
    adapter: &str,
    scene: &mut Scene,
    progress: Option<&Progress>,
    engine: Option<&str>,
    frame: Option<FrameKind>,
) {
    // Fixed width: the 3D viewport is sized against it, so it may not be
    // dragged out from under the camera.
    egui::Panel::right("growforge_panel")
        .exact_size(constants::VIEW_PANEL_WIDTH_POINTS)
        .resizable(false)
        .show(root, |ui| {
            // The heading is the window title, which names the build: see
            // `VIEW_WINDOW_TITLE`. Nothing below it repeats that.
            ui.heading(title);
            ui.label(egui::RichText::new(adapter).small().weak());
            ui.separator();

            layer_switches(ui, scene, false);
            ui.separator();

            match progress {
                None => {
                    ui.label(egui::RichText::new("problem setup").strong());
                    ui.label("nothing is running; this is the model the optimizer will see");
                }
                Some(progress) => {
                    ui.label(egui::RichText::new("run").strong());
                    ui.monospace(status_label(&progress.status));
                    if let Some(stats) = &progress.latest {
                        // The growth engine never evaluates a compliance, so
                        // the block it fills is a different one.
                        match stats.growth {
                            Some(growth) => {
                                ui.monospace(format!("step        {}", stats.iteration));
                                ui.monospace(format!("phase       {}", growth.phase.label()));
                                ui.monospace(format!("segments    {}", growth.segments));
                                ui.monospace(format!(
                                    "attractors  {}",
                                    growth.attractors_remaining
                                ));
                                ui.monospace(format!("volume frac {:.4}", stats.volume_fraction));
                                ui.monospace(format!("elapsed     {:.2} s", stats.elapsed_s));
                            }
                            None => {
                                ui.monospace(format!("iteration   {}", stats.iteration));
                                ui.monospace(format!("compliance  {:.6e}", stats.compliance));
                                ui.monospace(format!("volume frac {:.4}", stats.volume_fraction));
                                // Only while a local volume cap is active, and
                                // then it is the true worst neighbourhood rather
                                // than the aggregate the cap is stated on.
                                if let Some(worst) = stats.worst_local_fraction {
                                    ui.monospace(format!("worst local {worst:.4}"))
                                        .on_hover_text(
                                            "the fullest neighbourhood of the design this \
                                             iteration analysed, against the local volume cap",
                                        );
                                }
                                ui.monospace(format!("max change  {:.5}", stats.max_change));
                                ui.monospace(format!("elapsed     {:.2} s", stats.elapsed_s));
                            }
                        }
                    } else if engine == Some(constants::SOLID_ENGINE) {
                        ui.monospace("no iterations");
                        ui.label(
                            egui::RichText::new(
                                "the solid engine fills the domain and optimizes nothing, so there \
                                 is no progress to show",
                            )
                            .small()
                            .weak(),
                        );
                    } else {
                        ui.monospace("waiting for the first iteration");
                    }
                    ui.monospace(match frame {
                        Some(FrameKind::Preview { iteration }) => {
                            format!("preview of  iteration {iteration}")
                        }
                        Some(FrameKind::Final) => "showing     exported mesh".to_string(),
                        None => "no surface  yet".to_string(),
                    });
                    if let RunStatus::Finished {
                        triangles,
                        volume_mm3,
                        ..
                    } = &progress.status
                    {
                        ui.monospace(format!("triangles   {triangles}"));
                        ui.monospace(format!("volume      {volume_mm3:.1} mm3"));
                    }
                    if let Some(note) = &progress.note {
                        ui.separator();
                        ui.label(egui::RichText::new(note).small());
                    }
                }
            }

            ui.separator();
            ui.label(egui::RichText::new("controls").strong());
            ui.monospace("left drag    orbit");
            ui.monospace("right drag   pan");
            ui.monospace("scroll       zoom");
            ui.monospace("F            fit view");
            ui.monospace("close window detaches");
        });
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The text of every row a finished frame painted, one per line.
    ///
    /// egui has no readable widget tree once a frame is over: what it leaves is
    /// the shapes it painted, so a row is read back through its galley - the
    /// laid out text of the label itself. `Shape::Vec` is walked because a
    /// painter may nest shapes; nothing here paints text any other way.
    ///
    /// Shared with the editor's panel tests, which read their own header the
    /// same way.
    pub fn painted_text(output: &egui::FullOutput) -> String {
        fn collect(shape: &egui::Shape, into: &mut String) {
            match shape {
                egui::Shape::Text(text) => {
                    into.push_str(text.galley.text());
                    into.push('\n');
                }
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        collect(shape, into);
                    }
                }
                _ => {}
            }
        }

        let mut text = String::new();
        for clipped in &output.shapes {
            collect(&clipped.shape, &mut text);
        }
        text
    }

    /// Where the row whose whole text is `needle` was painted: the middle of
    /// that text, which is a point inside the widget that drew it.
    ///
    /// What it is for is driving a click at a row without knowing egui's own
    /// widget ids, which a frame does not hand back: a checkbox responds over
    /// its label as well as its box, so the middle of the label is a point the
    /// switch itself takes the click at.
    pub fn painted_text_center(output: &egui::FullOutput, needle: &str) -> Option<egui::Pos2> {
        fn find(shape: &egui::Shape, needle: &str, found: &mut Option<egui::Pos2>) {
            match shape {
                egui::Shape::Text(text) if found.is_none() && text.galley.text() == needle => {
                    *found = Some(text.pos + text.galley.size() / 2.0);
                }
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        find(shape, needle, found);
                    }
                }
                _ => {}
            }
        }

        let mut found = None;
        for clipped in &output.shapes {
            find(&clipped.shape, needle, &mut found);
        }
        found
    }

    /// A raw input laying the window out at the size the viewer opens at.
    ///
    /// A row is painted only while it is inside the screen rect, so a test that
    /// reads back what was painted has to give the frame one.
    pub fn window_input() -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(
                    f32::from(constants::VIEW_WINDOW_WIDTH as u16),
                    f32::from(constants::VIEW_WINDOW_HEIGHT as u16),
                ),
            )),
            ..Default::default()
        }
    }

    /// The panel names the build that drew it, in its heading.
    ///
    /// The heading is the window title, and the title is what carries the build
    /// since 0.38.0, so the panel is given a title built the way a run window
    /// builds one - the prefix constant and the project - and the painted rows
    /// are read back for it. Compared against a version spelled out here rather
    /// than against the constant, so the assertion is what a screenshot has to
    /// show and not a second reading of the same value.
    #[test]
    fn the_panel_says_which_build_it_is() {
        let context = egui::Context::default();
        let mut scene = Scene::default();
        let title = format!("{} - test project", constants::VIEW_WINDOW_TITLE);
        let output = context.run_ui(window_input(), |root| {
            panel(root, &title, "test adapter", &mut scene, None, None, None);
        });
        let text = painted_text(&output);
        let expected = format!("growforge {} - test project", env!("CARGO_PKG_VERSION"));
        assert!(
            text.lines().any(|line| line == expected),
            "the run panel's heading is not {expected:?}: {text}"
        );
    }

    /// This panel lists the layers a run and a setup view produce, and only
    /// those.
    ///
    /// It has no precision block to home a switch in, so nothing may be taken
    /// out of its list: the floor grid the editor moved beside its snap
    /// increment is an editor layer this panel never drew in the first place.
    /// Read off the table both ways round, so a layer that vanished from the
    /// block and one that appeared in it both fail.
    #[test]
    fn the_run_panel_lists_every_layer_outside_the_editor_and_no_other() {
        let context = egui::Context::default();
        let mut scene = Scene::default();
        let output = context.run_ui(window_input(), |root| {
            panel(root, "test", "test adapter", &mut scene, None, None, None);
        });
        let text = painted_text(&output);
        for info in LAYERS {
            let listed = text.lines().any(|line| line == info.label);
            assert_eq!(
                listed,
                info.role != LayerRole::Editor,
                "the run panel {} the {} switch: {text}",
                if listed { "draws" } else { "does not draw" },
                info.label
            );
        }
    }
}
