//! The egui side panel: layer visibility switches, the live stats block and
//! the control legend.

use crate::constants;
use crate::viewer::scene::{LAYERS, Layer, LayerRole, Scene};
use crate::viewer::snapshot::{FrameKind, Progress, RunStatus};

/// What a switch says on hover while the layer behind it is empty. A layer with
/// nothing in it is switched off rather than hidden, so the panel keeps its
/// shape from frame to frame - and the checkbox has to say why it cannot be
/// ticked.
const ABSENT_LAYER_HELP: &str = "nothing has produced this layer yet";

/// Draw the layer visibility switches and the two shading toggles.
///
/// Shared with the editor's panel, so the same scene is controlled the same way
/// whichever mode the window is in. Layers the editor alone produces are listed
/// only when `editor` is set; everywhere else they do not exist.
///
/// Every switch carries the hover text of the layer it draws, which the layer
/// table holds beside the label. Both windows therefore explain the same
/// switches the same way.
pub fn layer_switches(ui: &mut egui::Ui, scene: &mut Scene, editor: bool) {
    ui.label(egui::RichText::new("show").strong());
    for info in LAYERS
        .iter()
        .filter(|info| editor || info.role != LayerRole::Editor)
    {
        let present = scene.get(info.layer).is_some();
        let visible = scene.visibility_mut(info.layer);
        ui.add_enabled(present, egui::Checkbox::new(visible, info.label))
            .on_hover_text(info.help)
            .on_disabled_hover_text(ABSENT_LAYER_HELP);
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
pub fn panel(
    root: &mut egui::Ui,
    title: &str,
    adapter: &str,
    scene: &mut Scene,
    progress: Option<&Progress>,
    frame: Option<FrameKind>,
) {
    // Fixed width: the 3D viewport is sized against it, so it may not be
    // dragged out from under the camera.
    egui::Panel::right("growforge_panel")
        .exact_size(constants::VIEW_PANEL_WIDTH_POINTS)
        .resizable(false)
        .show(root, |ui| {
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
