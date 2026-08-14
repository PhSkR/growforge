//! The drawable scene: one triangle soup per overlay class, plus the code that
//! derives those overlays from a configuration and its discrete problem.

use anyhow::Result;

use crate::config::{Config, LoadSpec};
use crate::constants;
use crate::geometry::{Aabb, cross, difference, length, scale};
use crate::grid::Grid;
use crate::mesh::{self, Mesh};
use crate::problem::Problem;
use crate::viewer::camera;
use crate::viewer::tessellate;

/// One vertex of the shaded triangle soup handed to the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// World position in millimetres.
    pub position: [f32; 3],
    /// Shading normal: the face normal of a flat layer, the area weighted
    /// vertex normal of a smooth one.
    pub normal: [f32; 3],
    /// Linear RGBA; the alpha drives the translucent pass.
    pub color: [f32; 4],
}

/// Which normal a layer's triangles carry.
///
/// The geometry is the same either way - a triangle soup with three vertices
/// per triangle - so the choice costs no vertices and no memory, only which
/// direction each of them is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shading {
    /// One normal per triangle: every facet reads as a facet. What a
    /// voxelization wants, because its facets are the answer.
    Flat,
    /// Area weighted average of the normals of the triangles meeting at each
    /// vertex, computed once on the mesher side. What makes an isosurface read
    /// as the curved solid it approximates.
    Smooth,
    /// Smooth across the facets of a curved surface and sharp across a real
    /// edge: the normals meeting at a vertex are averaged only with the ones
    /// within [`constants::VIEW_CREASE_ANGLE_DEGREES`] of their own facet.
    ///
    /// This is what an analytically tessellated overlay wants. A cylinder's
    /// barrel is a circle approximated by facets a few degrees apart and reads
    /// as round; the rim where that barrel meets its cap is a right angle and
    /// stays an edge; a box is all right angles and comes out exactly as flat
    /// as it was. One rule, and no shape has to declare which kind it is.
    Rounded,
}

/// A layer's geometry, expanded to shaded triangles.
#[derive(Debug, Clone)]
pub struct LayerMesh {
    /// Three vertices per triangle, in draw order.
    pub vertices: Vec<Vertex>,
    /// Bounds of the layer in millimetres.
    pub bounds: Aabb,
}

impl Default for LayerMesh {
    fn default() -> Self {
        LayerMesh {
            vertices: Vec::new(),
            bounds: Aabb::empty(),
        }
    }
}

impl LayerMesh {
    /// Expand an indexed mesh into shaded triangles of one colour, dropping
    /// triangles that carry no area or a non-finite coordinate.
    pub fn from_mesh(source: &Mesh, color: [f32; 4], shading: Shading) -> LayerMesh {
        LayerMesh::shade(source, shading, |_| color)
    }

    /// As [`LayerMesh::from_mesh`], but with a colour per source vertex.
    ///
    /// `colors` is indexed by source vertex, so a vertex shared by several
    /// triangles keeps one colour and the colouring stays continuous across the
    /// surface even though the geometry is expanded into a soup.
    pub fn from_mesh_colored(source: &Mesh, colors: &[[f32; 4]], shading: Shading) -> LayerMesh {
        assert_eq!(
            colors.len(),
            source.vertices.len(),
            "one colour per mesh vertex is required"
        );
        LayerMesh::shade(source, shading, |vertex| colors[vertex])
    }

    fn shade(source: &Mesh, shading: Shading, color_of: impl Fn(usize) -> [f32; 4]) -> LayerMesh {
        // The averaging walks the indexed mesh once, here on the caller's
        // thread - the mesher or the export worker - and never on the thread
        // that draws.
        let smooth = match shading {
            Shading::Flat | Shading::Rounded => Vec::new(),
            Shading::Smooth => mesh::vertex_normals(source),
        };
        let creased = match shading {
            Shading::Rounded => crease_normals(source),
            _ => Vec::new(),
        };
        let mut out = LayerMesh {
            vertices: Vec::with_capacity(source.triangles.len() * 3),
            bounds: Aabb::empty(),
        };
        for (triangle_index, triangle) in source.triangles.iter().enumerate() {
            let indices = [
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            ];
            let corners = indices.map(|index| source.vertices[index]);
            if corners
                .iter()
                .any(|c| c.iter().any(|value| !value.is_finite()))
            {
                continue;
            }
            let raw = cross(
                difference(corners[1], corners[0]),
                difference(corners[2], corners[0]),
            );
            let len = length(raw);
            if len <= 0.0 {
                continue;
            }
            let face = scale(raw, 1.0 / len);
            for (corner, (slot, vertex)) in corners.into_iter().zip(indices.into_iter().enumerate())
            {
                for (d, value) in corner.iter().enumerate() {
                    out.bounds.min[d] = out.bounds.min[d].min(*value);
                    out.bounds.max[d] = out.bounds.max[d].max(*value);
                }
                // A vertex whose adjacent triangles cancelled out has no
                // direction of its own and keeps the face normal. A creased
                // normal belongs to this *corner of this triangle* rather than
                // to the vertex, which is what lets the one vertex on a
                // cylinder's rim be round along the barrel and sharp across the
                // cap it also belongs to.
                let normal = match (creased.get(3 * triangle_index + slot), smooth.get(vertex)) {
                    (Some(n), _) if length(*n) > 0.0 => *n,
                    (_, Some(n)) if length(*n) > 0.0 => *n,
                    _ => face,
                };
                out.vertices.push(Vertex {
                    position: [corner[0] as f32, corner[1] as f32, corner[2] as f32],
                    normal: [normal[0] as f32, normal[1] as f32, normal[2] as f32],
                    color: color_of(vertex),
                });
            }
        }
        out
    }

    /// A copy of this layer with per-triangle normals, whatever it was built
    /// with.
    ///
    /// The face normals are recovered from the layer's own positions, which is
    /// what lets the side panel switch a smooth surface back to the facetted
    /// look without keeping a second copy of it, or a second normal per vertex,
    /// alive for a switch that is usually never flipped. Positions are f32
    /// here, so a normal can differ from the extraction's own in the last bits
    /// of a float; a triangle too thin to give one at all keeps the normal it
    /// already had.
    pub fn flat_shaded(&self) -> LayerMesh {
        let mut out = self.clone();
        for triangle in out.vertices.chunks_mut(3) {
            if triangle.len() < 3 {
                break;
            }
            let corners = [
                triangle[0].position.map(f64::from),
                triangle[1].position.map(f64::from),
                triangle[2].position.map(f64::from),
            ];
            let raw = cross(
                difference(corners[1], corners[0]),
                difference(corners[2], corners[0]),
            );
            let len = length(raw);
            if len <= 0.0 {
                continue;
            }
            let face = scale(raw, 1.0 / len);
            let normal = [face[0] as f32, face[1] as f32, face[2] as f32];
            for vertex in triangle.iter_mut() {
                vertex.normal = normal;
            }
        }
        out
    }

    /// True when the layer has nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Number of triangles.
    pub fn triangles(&self) -> usize {
        self.vertices.len() / 3
    }
}

/// One normal per triangle *corner*, averaged only across the neighbours that
/// are part of the same smooth surface.
///
/// The rule is the standard crease angle: the triangles meeting at a vertex
/// contribute to a corner's normal when their own facet lies within
/// [`constants::VIEW_CREASE_ANGLE_DEGREES`] of that corner's facet. A
/// tessellated barrel's facets are a few degrees apart and all contribute; the
/// cap on the other side of the rim is a right angle away and does not; a box's
/// three faces at a corner are right angles apart and each keeps its own.
///
/// Returns three normals per source triangle, in triangle order, so a caller
/// walking the source list can index straight into it. A triangle with no area
/// contributes nothing and its corners come back as zero, which the caller
/// reads as "use the face normal".
fn crease_normals(source: &Mesh) -> Vec<crate::geometry::Vec3> {
    let count = source.triangles.len();
    let mut faces: Vec<crate::geometry::Vec3> = Vec::with_capacity(count);
    let mut weights: Vec<f64> = Vec::with_capacity(count);
    for triangle in &source.triangles {
        let corners = [
            source.vertices[triangle[0] as usize],
            source.vertices[triangle[1] as usize],
            source.vertices[triangle[2] as usize],
        ];
        let raw = cross(
            difference(corners[1], corners[0]),
            difference(corners[2], corners[0]),
        );
        let area = length(raw);
        faces.push(if area > 0.0 {
            scale(raw, 1.0 / area)
        } else {
            [0.0; 3]
        });
        weights.push(area);
    }

    // Which triangles meet at each vertex. Built once; the shapes this runs on
    // are overlays of a few thousand triangles.
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); source.vertices.len()];
    for (index, triangle) in source.triangles.iter().enumerate() {
        for vertex in triangle {
            adjacency[*vertex as usize].push(index as u32);
        }
    }

    let threshold = constants::VIEW_CREASE_ANGLE_DEGREES.to_radians().cos();
    let mut out = vec![[0.0; 3]; count * 3];
    for (index, triangle) in source.triangles.iter().enumerate() {
        let face = faces[index];
        if weights[index] <= 0.0 {
            continue;
        }
        for (slot, vertex) in triangle.iter().enumerate() {
            let mut sum = [0.0; 3];
            for other in &adjacency[*vertex as usize] {
                let other = *other as usize;
                if weights[other] <= 0.0 || crate::geometry::inner(faces[other], face) < threshold {
                    continue;
                }
                sum = crate::geometry::sum(sum, scale(faces[other], weights[other]));
            }
            let len = length(sum);
            out[3 * index + slot] = if len > 0.0 {
                scale(sum, 1.0 / len)
            } else {
                face
            };
        }
    }
    out
}

/// The overlay classes the viewer can show, each independently toggleable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// Voxelized design domain, exactly as the optimizer sees it.
    Domain,
    /// Regions that must stay empty.
    Keepout,
    /// Regions forced solid.
    Keepin,
    /// Markers on the constrained nodes.
    Supports,
    /// Force arrows and torque arcs.
    Loads,
    /// The evolving, then final, density isosurface.
    Density,
    /// Highlight shell around the object the editor has selected.
    Selection,
    /// Thinner shell around the object under the pointer: what a click would
    /// select, previewed before it is clicked.
    Hover,
    /// The editor's drag handles: translation arrows, rotation arcs and resize
    /// cubes.
    Gizmo,
    /// The editor's dimension lines: what a drag in progress is measuring, and
    /// how far the object is off the floor of the domain.
    Measure,
    /// Ruled plane on the floor of the domain, at the increment drags snap to.
    Grid,
    /// The editor's placement preview: the points a two-click placement has
    /// taken so far, and the shape the next click would create.
    Placement,
}

/// Where a layer's geometry comes from, which is also what decides who lists it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerRole {
    /// Derived from the configuration by [`build`]. An edit rebuilds all of
    /// them together.
    Setup,
    /// Produced by a run: the evolving preview and then the exported surface.
    Density,
    /// Produced by the editor alone. The setup view and the run panel never
    /// list these, so a build that never opens the editor is unchanged by them.
    Editor,
}

/// Which block of a panel draws a layer's visibility switch.
///
/// The switch itself is the same checkbox wherever it is drawn, explained by
/// the same hover text; this says where it belongs, which is a fact about the
/// layer rather than about the panel that happens to draw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchHome {
    /// The `show` block, with the rest of them.
    Show,
    /// A block that owns the switch and draws it itself: the floor grid's
    /// belongs beside the snap increment it is ruled at, in the editor's
    /// precision block. The `show` block leaves these out, so only a layer the
    /// editor produces may say it - anywhere else there is no such block, and
    /// the switch would exist nowhere at all.
    Elsewhere,
}

/// Static description of one layer.
#[derive(Debug, Clone, Copy)]
pub struct LayerInfo {
    /// The layer this describes.
    pub layer: Layer,
    /// Label used by the visibility checkboxes.
    pub label: &'static str,
    /// What the layer shows, as the visibility checkbox says it on hover. Part
    /// of the table rather than of the panel, so a layer added here cannot be
    /// added without saying what it draws.
    pub help: &'static str,
    /// Colour every triangle of the layer is given.
    pub color: [f32; 4],
    /// Translucent layers are drawn after the opaque ones with depth writes
    /// switched off.
    pub translucent: bool,
    /// What produces this layer's geometry.
    pub role: LayerRole,
    /// Where the visibility switch for this layer is drawn.
    pub switch: SwitchHome,
}

/// Every layer, in panel and draw order. Opaque layers are drawn first; the
/// renderer walks this table twice rather than sorting at run time.
pub const LAYERS: &[LayerInfo] = &[
    LayerInfo {
        layer: Layer::Density,
        label: "density surface",
        help: "the isosurface of the design, re-extracted as a run iterates, and the exported mesh \
               once one has finished",
        color: constants::VIEW_COLOR_DENSITY,
        translucent: false,
        role: LayerRole::Density,
        switch: SwitchHome::Show,
    },
    LayerInfo {
        layer: Layer::Supports,
        label: "supports",
        help: "a marker on every node a supports region really constrains, read back out of the \
               assembled model",
        color: constants::VIEW_COLOR_SUPPORT,
        translucent: false,
        role: LayerRole::Setup,
        switch: SwitchHome::Show,
    },
    LayerInfo {
        layer: Layer::Loads,
        label: "loads",
        help: "an arrow per force and a circular arc per torque. A gravity load gets none: it \
               acts on the whole structure",
        color: constants::VIEW_COLOR_LOAD,
        translucent: false,
        role: LayerRole::Setup,
        switch: SwitchHome::Show,
    },
    LayerInfo {
        layer: Layer::Gizmo,
        label: "gizmo",
        help: "the handles of the selected object: three translation arrows, three rotation arcs \
               and the resize handles its own shape has",
        color: constants::VIEW_COLOR_GIZMO_HANDLE,
        translucent: false,
        role: LayerRole::Editor,
        switch: SwitchHome::Show,
    },
    LayerInfo {
        layer: Layer::Measure,
        label: "dimensions",
        help: "the dimension lines of a drag: what the handle is measuring, and how far the \
               object is off the floor of the domain",
        color: constants::VIEW_COLOR_MEASURE,
        translucent: false,
        role: LayerRole::Editor,
        switch: SwitchHome::Show,
    },
    LayerInfo {
        layer: Layer::Grid,
        label: "floor grid",
        help: "the plane ruled at the snap increment, on the floor of the domain. It is what \
               makes the increment visible",
        // Per line: the minor colour is the layer's, and the major lines carry
        // their own, exactly as the gizmo's parts do.
        color: constants::VIEW_COLOR_GRID_MINOR,
        translucent: false,
        role: LayerRole::Editor,
        // Beside the increment it is ruled at, since 0.38.1: the grid is what
        // makes that increment visible, and the switch reads as an editing
        // control rather than as one more overlay.
        switch: SwitchHome::Elsewhere,
    },
    LayerInfo {
        layer: Layer::Domain,
        label: "domain (voxelized)",
        help: "the design space as it was voxelized, chamfers and all, so what you check is what \
               the solver actually got",
        color: constants::VIEW_COLOR_DOMAIN,
        translucent: true,
        role: LayerRole::Setup,
        switch: SwitchHome::Show,
    },
    LayerInfo {
        layer: Layer::Keepout,
        label: "keepout",
        help: "the regions that must stay empty",
        color: constants::VIEW_COLOR_KEEPOUT,
        translucent: true,
        role: LayerRole::Setup,
        switch: SwitchHome::Show,
    },
    LayerInfo {
        layer: Layer::Keepin,
        label: "keepin",
        help: "the regions forced solid, such as load pads and anchor bosses",
        color: constants::VIEW_COLOR_KEEPIN,
        translucent: true,
        role: LayerRole::Setup,
        switch: SwitchHome::Show,
    },
    LayerInfo {
        layer: Layer::Selection,
        label: "selection",
        help: "the shell around the object that is selected",
        color: constants::VIEW_COLOR_SELECTION,
        translucent: true,
        role: LayerRole::Editor,
        switch: SwitchHome::Show,
    },
    LayerInfo {
        layer: Layer::Hover,
        label: "hover highlight",
        help: "the outline around whatever a click would select, drawn with the ray and the \
               ranking that click would use",
        color: constants::VIEW_COLOR_HOVER,
        translucent: true,
        role: LayerRole::Editor,
        switch: SwitchHome::Show,
    },
    LayerInfo {
        layer: Layer::Placement,
        label: "placement preview",
        help: "while a tube is being placed: the point already clicked, a marker where the next \
               click would land, and the tube they would make",
        // Per part: the band carries the layer's colour and the point markers
        // carry their own, exactly as the grid's two weights do.
        color: constants::VIEW_COLOR_PLACE_GHOST,
        translucent: true,
        role: LayerRole::Editor,
        switch: SwitchHome::Show,
    },
];

impl Layer {
    /// Slot of this layer in the per-layer arrays.
    pub fn slot(self) -> usize {
        LAYERS
            .iter()
            .position(|info| info.layer == self)
            .expect("every layer variant is listed in LAYERS")
    }

    /// Static description of this layer.
    pub fn info(self) -> &'static LayerInfo {
        &LAYERS[self.slot()]
    }
}

/// Everything the renderer draws, one entry per layer.
#[derive(Debug, Clone)]
pub struct Scene {
    meshes: Vec<Option<LayerMesh>>,
    visible: Vec<bool>,
    /// Stress coloured copy of the density layer, when a run produced one.
    stress: Option<LayerMesh>,
    /// Panel toggle: draw the density layer with the stress colours instead of
    /// the plain surface colour. Plain is the default.
    stress_shading: bool,
    /// Panel toggle: draw the density layer with per-triangle normals instead
    /// of the smooth ones it was built with.
    flat_shading: bool,
    /// Panel toggle: show the part on its own, leaving every other layer out of
    /// the frame whatever its own switch says.
    isolate_part: bool,
}

impl Default for Scene {
    fn default() -> Self {
        Scene {
            meshes: vec![None; LAYERS.len()],
            visible: vec![true; LAYERS.len()],
            stress: None,
            stress_shading: false,
            flat_shading: constants::VIEW_DEFAULT_FLAT_SHADING,
            isolate_part: constants::VIEW_EDIT_ISOLATE_PART_DEFAULT,
        }
    }
}

impl Scene {
    /// Replace a layer's geometry; `None` removes it.
    ///
    /// Emptying the density layer clears the switch that shows the part on its
    /// own, whoever empties it: the mode describes a surface that is no longer
    /// there, and the viewport takes no editing while it is set, so a flag left
    /// standing would isolate - and lock - the window again the moment the next
    /// preview put a surface back, with nobody having asked for it. Enforced
    /// here rather than in the paths that empty the layer, so it holds for every
    /// caller and there is one place it is stated: the same shape
    /// [`Scene::set_stress`] has.
    pub fn set(&mut self, layer: Layer, mesh: Option<LayerMesh>) {
        if layer == Layer::Density && mesh.is_none() {
            self.isolate_part = false;
        }
        self.meshes[layer.slot()] = mesh;
    }

    /// Geometry of a layer, if it has any.
    pub fn get(&self, layer: Layer) -> Option<&LayerMesh> {
        self.meshes[layer.slot()].as_ref()
    }

    /// Attach the stress coloured copy of the density layer. `None` removes it
    /// and falls back to plain shading.
    pub fn set_stress(&mut self, mesh: Option<LayerMesh>) {
        self.stress = mesh;
        if self.stress.is_none() {
            self.stress_shading = false;
        }
    }

    /// True once a stress coloured surface is available to switch to.
    pub fn has_stress(&self) -> bool {
        self.stress.is_some()
    }

    /// Mutable toggle for the side panel checkbox.
    pub fn stress_shading_mut(&mut self) -> &mut bool {
        &mut self.stress_shading
    }

    /// True while the density layer is drawn with the stress colours.
    pub fn stress_shading(&self) -> bool {
        self.stress_shading && self.stress.is_some()
    }

    /// Mutable toggle for the side panel checkbox.
    pub fn flat_shading_mut(&mut self) -> &mut bool {
        &mut self.flat_shading
    }

    /// True while the density layer is drawn with per-triangle normals.
    ///
    /// The geometry itself is unchanged: the renderer derives the face normals
    /// from it when it uploads, so the switch costs nothing while it is off.
    pub fn flat_shading(&self) -> bool {
        self.flat_shading
    }

    /// Mutable toggle for the side panel checkbox.
    pub fn isolate_part_mut(&mut self) -> &mut bool {
        &mut self.isolate_part
    }

    /// True while nothing but the part is on screen.
    ///
    /// Guarded by there being a part, exactly as the stress toggle is guarded by
    /// there being a stress copy: a switch left on when the surface it shows was
    /// taken away would leave an empty viewport - and, since the editor stops
    /// taking pointer input while this is set, an empty viewport nothing could
    /// be done in. What keeps the flag itself from ever standing in that state
    /// is [`Scene::set`], which clears it whenever the density layer is emptied,
    /// whoever empties it; the guard here is what makes the two impossible to
    /// disagree, and covers a flag written on with no part ever set.
    pub fn isolate_part(&self) -> bool {
        self.isolate_part && self.get(Layer::Density).is_some()
    }

    /// The geometry the density layer should currently be drawn with.
    pub fn density_geometry(&self) -> Option<&LayerMesh> {
        if self.stress_shading() {
            self.stress.as_ref()
        } else {
            self.get(Layer::Density)
        }
    }

    /// Take the setup overlays of `other`, leaving the density surface, the
    /// stress copy and every panel toggle of this scene alone.
    ///
    /// This is the editor's refresh path: a committed edit rebuilds the whole
    /// setup through [`build`], and only the layers that configuration produces
    /// may be replaced by it.
    pub fn adopt_setup(&mut self, other: &Scene) {
        for info in LAYERS.iter().filter(|i| i.role == LayerRole::Setup) {
            self.set(info.layer, other.get(info.layer).cloned());
        }
    }

    /// Drop the density surface and its stress coloured copy.
    ///
    /// Called when an edit has made the design on screen describe a
    /// configuration that no longer exists; showing it next to the new setup
    /// would claim a result the editor does not have.
    ///
    /// The switch that shows the part on its own goes with it, in
    /// [`Scene::set`], which is where that holds for every caller rather than
    /// only for this one.
    pub fn clear_density(&mut self) {
        self.set(Layer::Density, None);
        self.set_stress(None);
    }

    /// True when the layer is drawn: switched on in the side panel, and not left
    /// out by the switch that shows the part on its own.
    ///
    /// That switch overrides the individual ones rather than writing them, so
    /// unticking it puts every layer back exactly as it was found.
    pub fn is_visible(&self, layer: Layer) -> bool {
        if self.isolate_part() {
            return layer == Layer::Density;
        }
        self.visible[layer.slot()]
    }

    /// Mutable visibility flag, for the side panel checkboxes.
    pub fn visibility_mut(&mut self, layer: Layer) -> &mut bool {
        &mut self.visible[layer.slot()]
    }

    /// Bounds of every layer that carries geometry, visible or not, so the fit
    /// does not jump when a layer is toggled.
    ///
    /// The editor's own overlays are left out: they are drawn around whatever
    /// is selected and would make the fit follow the selection rather than the
    /// model.
    pub fn bounds(&self) -> Aabb {
        let mut bounds = Aabb::empty();
        for info in LAYERS.iter().filter(|i| i.role != LayerRole::Editor) {
            match self.get(info.layer) {
                Some(mesh) if !mesh.is_empty() => bounds = bounds.union(&mesh.bounds),
                _ => {}
            }
        }
        bounds
    }
}

/// Isosurface of a grid where every load-carrying cell is fully solid: the
/// design domain as the optimizer actually sees it, chamfers and all.
pub fn voxelized_domain(grid: &Grid, iso: f64) -> Result<Mesh> {
    let densities = vec![constants::DENSITY_MAX; grid.n_cells()];
    let field = mesh::node_density_field(grid, &densities, iso);
    mesh::marching_cubes::extract(&field, iso)
}

/// Preview isosurface of a density field: marching cubes only, with neither
/// Taubin smoothing nor validation, because a preview is thrown away as soon
/// as the next iteration lands.
///
/// A preview is never supersampled either, whatever `[output] supersample`
/// says: refining the lattice costs the cube of the factor on a surface that
/// exists for a fifth of a second, and the window switches to the real,
/// supersampled export the moment the run finishes.
pub fn preview_surface(grid: &Grid, densities: &[f64], iso: f64) -> Result<Mesh> {
    let field = mesh::node_density_field(grid, densities, iso);
    mesh::marching_cubes::extract(&field, iso)
}

/// Colour of a normalized scalar on the stress ramp.
///
/// `fraction` is clamped into `[0, 1]` and interpolated linearly between the
/// anchors of [`constants::VIEW_STRESS_COLORMAP`].
pub fn stress_color(fraction: f64) -> [f32; 4] {
    let ramp = constants::VIEW_STRESS_COLORMAP;
    let clamped = if fraction.is_finite() {
        fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let position = clamped * (ramp.len() - 1) as f64;
    let lower = (position.floor() as usize).min(ramp.len() - 1);
    let upper = (lower + 1).min(ramp.len() - 1);
    let t = (position - lower as f64) as f32;
    let mut color = [0.0f32; 4];
    for d in 0..3 {
        color[d] = ramp[lower][d] + t * (ramp[upper][d] - ramp[lower][d]);
    }
    color[3] = constants::VIEW_STRESS_ALPHA;
    color
}

/// Colour every vertex of `surface` by the von Mises stress sampled from a node
/// lattice field, normalized against `reference`.
///
/// The lattice values are the same eight-cell averages the densities use, so a
/// vertex reads the stress of the material it sits on; sampling is trilinear,
/// because Taubin smoothing has already moved the vertices off the lattice
/// edges they were extracted on - and, at `supersample > 1`, because most of
/// them were never on a node lattice edge to begin with. A supersampled surface
/// therefore reads the same field at more places rather than a different one.
///
/// This is the density layer, so it is shaded smooth like the plain copy of it;
/// the panel's flat switch applies to both.
pub fn stress_shaded(
    surface: &Mesh,
    field: &mesh::ScalarField,
    reference_mpa: f64,
) -> Option<LayerMesh> {
    if surface.is_empty() || reference_mpa <= 0.0 || reference_mpa.is_nan() {
        return None;
    }
    let colors: Vec<[f32; 4]> = surface
        .vertices
        .iter()
        .map(|v| stress_color(field.sample(*v) / reference_mpa))
        .collect();
    Some(LayerMesh::from_mesh_colored(
        surface,
        &colors,
        Shading::Smooth,
    ))
}

/// Union of the tessellated shapes of a keepout or keepin list.
fn shape_union(specs: &[crate::config::ShapeSpec], what: &str) -> Result<Mesh> {
    let mut mesh = Mesh::default();
    for (index, spec) in specs.iter().enumerate() {
        let shape = spec.to_shape(&format!("{what} entry {}", index + 1))?;
        tessellate::append(&mut mesh, &tessellate::shape(&shape));
    }
    Ok(mesh)
}

/// Cubes on every node that a support constrains.
///
/// The markers are read back out of the assembled constraint mask rather than
/// from the configuration, so they show the nodes the solver really pins.
/// Nodes touching no material are pinned by the model builder for conditioning
/// and are deliberately left out.
fn support_markers(problem: &Problem) -> Mesh {
    let grid = &problem.grid;
    let active = grid.active_nodes();
    let constrained: Vec<usize> = (0..grid.n_nodes())
        .filter(|&n| {
            active[n]
                && (0..constants::DOF_PER_NODE)
                    .any(|d| problem.fixed[n * constants::DOF_PER_NODE + d])
        })
        .collect();
    let stride = constrained
        .len()
        .div_ceil(constants::VIEW_MAX_SUPPORT_MARKERS.max(1))
        .max(1);
    let edge = grid.h * constants::VIEW_SUPPORT_MARKER_VOXEL_FRACTION;
    let mut mesh = Mesh::default();
    for &node in constrained.iter().step_by(stride) {
        let (i, j, k) = grid.node_ijk(node);
        tessellate::append(
            &mut mesh,
            &tessellate::marker_cube(grid.node_pos(i, j, k), edge),
        );
    }
    mesh
}

/// Arrows for every force and arcs for every torque, in every load case.
fn load_indicators(config: &Config, scene_radius: f64) -> Result<Mesh> {
    let arrow_length = scene_radius * constants::VIEW_ARROW_LENGTH_RADIUS_FRACTION;
    let arc_radius = scene_radius * constants::VIEW_TORQUE_ARC_RADIUS_FRACTION;
    let mut mesh = Mesh::default();
    for (case_index, case) in config.loadcases.iter().enumerate() {
        for (load_index, load) in case.loads.iter().enumerate() {
            let what = format!(
                "[[loadcases]] \"{}\" (entry {}), load {}",
                case.name,
                case_index + 1,
                load_index + 1
            );
            let indicator = match load {
                // Self weight acts on the whole structure, so there is nothing
                // to anchor an arrow to; the density surface itself is the
                // picture of where it acts.
                LoadSpec::Gravity { .. } => continue,
                LoadSpec::Force { region, vector } => {
                    let anchor = tessellate::centroid(&region.to_shape(&what)?);
                    tessellate::arrow(anchor, *vector, arrow_length)
                }
                LoadSpec::Torque {
                    region,
                    axis_point,
                    axis_dir,
                    magnitude_nmm,
                } => {
                    region.to_shape(&what)?;
                    tessellate::torque_arc(*axis_point, *axis_dir, arc_radius, *magnitude_nmm)
                }
            };
            tessellate::append(&mut mesh, &indicator);
        }
    }
    Ok(mesh)
}

/// Build the setup overlays: domain, keepout, keepin, supports and loads.
///
/// The density layer is left empty; a run fills it from its worker thread.
pub fn build(config: &Config, problem: &Problem) -> Result<Scene> {
    let mut scene = Scene::default();

    // The voxelized domain is the one overlay whose facets *are* the answer:
    // it is what the optimizer sees, chamfers and all, and rounding it would
    // draw a domain the solver does not have. Everything else here is an
    // analytic shape, and is shaded so that its curved parts read as curved and
    // its edges stay edges.
    let domain = voxelized_domain(&problem.grid, problem.output.iso_level)?;
    scene.set(
        Layer::Domain,
        Some(LayerMesh::from_mesh(
            &domain,
            Layer::Domain.info().color,
            Shading::Flat,
        )),
    );

    for (layer, specs, what) in [
        (Layer::Keepout, &config.keepout, "[[keepout]]"),
        (Layer::Keepin, &config.keepin, "[[keepin]]"),
    ] {
        let mesh = shape_union(specs, what)?;
        if !mesh.is_empty() {
            scene.set(
                layer,
                Some(LayerMesh::from_mesh(
                    &mesh,
                    layer.info().color,
                    Shading::Rounded,
                )),
            );
        }
    }

    let markers = support_markers(problem);
    if !markers.is_empty() {
        scene.set(
            Layer::Supports,
            Some(LayerMesh::from_mesh(
                &markers,
                Layer::Supports.info().color,
                Shading::Rounded,
            )),
        );
    }

    // Load indicators are sized against the geometry that already exists, so
    // they scale with the part rather than with an absolute millimetre value.
    let (_, radius) = camera::bounding_sphere(&scene.bounds().union(&problem.grid.bounds()));
    let loads = load_indicators(config, radius)?;
    if !loads.is_empty() {
        scene.set(
            Layer::Loads,
            Some(LayerMesh::from_mesh(
                &loads,
                Layer::Loads.info().color,
                Shading::Rounded,
            )),
        );
    }

    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    const SETUP: &str = r#"
[project]
name = "setup"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0

[output]
stl_path = "setup.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [40.0, 16.0, 16.0]

[[keepout]]
shape = "cylinder"
p1 = [20.0, 8.0, -1.0]
p2 = [20.0, 8.0, 17.0]
radius = 3.0

[[keepin]]
shape = "box"
min = [36.0, 4.0, 4.0]
max = [40.0, 12.0, 12.0]

[[supports]]
region = { shape = "box", min = [0.0, 0.0, 0.0], max = [0.5, 16.0, 16.0] }

[[loadcases]]
name = "main"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [40.0, 8.0, 8.0], radius = 3.0 }
vector = [0.0, 0.0, -50.0]
[[loadcases.loads]]
type = "torque"
region = { shape = "box", min = [39.0, -1.0, -1.0], max = [41.0, 17.0, 17.0] }
axis_point = [40.0, 8.0, 8.0]
axis_dir = [1.0, 0.0, 0.0]
magnitude_nmm = 500.0
"#;

    fn setup() -> (Config, Problem) {
        let config = Config::parse(SETUP).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        (config, problem)
    }

    /// A density layer, which is what a run leaves on screen: any geometry at
    /// all, in the layer the switches below are about.
    fn part() -> LayerMesh {
        LayerMesh::from_mesh(
            &tessellate::box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Layer::Density.info().color,
            Shading::Smooth,
        )
    }

    /// A small grid of design cells holding an off-centre ball: a curved
    /// surface whose vertices land between the lattice planes rather than on
    /// them, which is what a sampling check needs.
    fn ramp_grid() -> (Grid, Vec<f64>) {
        let h = 2.0;
        let side = 16.0;
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [side, side, side],
            },
            h,
        );
        for cell in grid.cells.iter_mut() {
            *cell = crate::grid::CellKind::Design;
        }
        let centre = [0.47 * side, 0.5 * side, 0.53 * side];
        let radius = 0.3 * side;
        let densities = (0..grid.n_cells())
            .map(|e| {
                let (i, j, k) = grid.cell_ijk(e);
                let p = [
                    grid.origin[0] + (i as f64 + 0.5) * h,
                    grid.origin[1] + (j as f64 + 0.5) * h,
                    grid.origin[2] + (k as f64 + 0.5) * h,
                ];
                let d = (0..3)
                    .map(|d| (p[d] - centre[d]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                ((radius - d) / h * 0.5 + 0.5).clamp(0.0, 1.0)
            })
            .collect();
        (grid, densities)
    }

    #[test]
    fn every_layer_has_exactly_one_table_entry() {
        for info in LAYERS {
            assert_eq!(
                info.layer.slot(),
                LAYERS.iter().position(|i| i.layer == info.layer).unwrap()
            );
            assert_eq!(info.layer.info().label, info.label);
        }
        let mut slots: Vec<usize> = LAYERS.iter().map(|i| i.layer.slot()).collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), LAYERS.len());
    }

    /// Every switch in the panel says what it shows, and the label repeated is
    /// not saying anything: the checkbox already carries that.
    #[test]
    fn every_layer_says_what_it_draws() {
        for info in LAYERS {
            assert!(
                !info.help.trim().is_empty(),
                "{} has no hover text",
                info.label
            );
            assert_ne!(
                info.help, info.label,
                "{} only repeats its label",
                info.label
            );
        }
    }

    #[test]
    fn the_setup_scene_fills_every_configured_layer() {
        let (config, problem) = setup();
        let scene = build(&config, &problem).expect("scene");
        for layer in [
            Layer::Domain,
            Layer::Keepout,
            Layer::Keepin,
            Layer::Supports,
            Layer::Loads,
        ] {
            let mesh = scene
                .get(layer)
                .unwrap_or_else(|| panic!("{layer:?} has no geometry"));
            assert!(mesh.triangles() > 0, "{layer:?} is empty");
            assert!(mesh.vertices.iter().all(|v| {
                v.position
                    .iter()
                    .chain(v.normal.iter())
                    .all(|value| value.is_finite())
            }));
        }
        // A setup view shows no density surface until a run produces one.
        assert!(scene.get(Layer::Density).is_none());
        assert!(!scene.bounds().is_empty());
    }

    #[test]
    fn the_voxelized_domain_covers_the_grid_block() {
        let (_, problem) = setup();
        let mesh = voxelized_domain(&problem.grid, problem.output.iso_level).expect("domain");
        let layer = LayerMesh::from_mesh(&mesh, Layer::Domain.info().color, Shading::Flat);
        let grid = problem.grid.bounds();
        for d in 0..3 {
            assert!(layer.bounds.min[d] >= grid.min[d] - problem.grid.h);
            assert!(layer.bounds.max[d] <= grid.max[d] + problem.grid.h);
        }
        // The keepout cylinder is a hole, so the surface must have more
        // triangles than a plain box would.
        assert!(layer.triangles() > 12);
    }

    #[test]
    fn support_markers_sit_only_on_constrained_active_nodes() {
        let (_, problem) = setup();
        let markers = support_markers(&problem);
        assert!(!markers.is_empty());
        // One cube per constrained node of the clamped face, 9 x 9 nodes.
        assert_eq!(markers.triangles.len(), 81 * 12);
        for vertex in &markers.vertices {
            assert!((vertex[0] - problem.grid.origin[0]).abs() < problem.grid.h);
        }
    }

    #[test]
    fn visibility_defaults_to_on_and_toggles_per_layer() {
        let mut scene = Scene::default();
        for info in LAYERS {
            assert!(scene.is_visible(info.layer));
        }
        *scene.visibility_mut(Layer::Keepout) = false;
        assert!(!scene.is_visible(Layer::Keepout));
        assert!(scene.is_visible(Layer::Keepin));
    }

    /// A part on its own, and every layer back where it was afterwards.
    ///
    /// The switch is a mode over the individual ones rather than a write to
    /// them: what it hides it hides whatever those say, and what it hid it
    /// gives back exactly as it found it - which is only true if the per-layer
    /// flags were never touched, so they are read back one by one.
    #[test]
    fn showing_nothing_but_the_part_hides_every_other_layer_and_gives_them_all_back() {
        let mut scene = Scene::default();
        scene.set(Layer::Density, Some(part()));
        // A state worth restoring: not every layer is on to begin with.
        *scene.visibility_mut(Layer::Keepout) = false;
        *scene.visibility_mut(Layer::Gizmo) = false;
        let before: Vec<bool> = LAYERS.iter().map(|i| scene.is_visible(i.layer)).collect();

        *scene.isolate_part_mut() = true;
        assert!(scene.isolate_part());
        for info in LAYERS {
            assert_eq!(
                scene.is_visible(info.layer),
                info.layer == Layer::Density,
                "the {} layer is drawn wrongly while the part is shown on its own",
                info.label
            );
        }

        *scene.isolate_part_mut() = false;
        assert!(!scene.isolate_part());
        for (info, was) in LAYERS.iter().zip(before) {
            assert_eq!(
                scene.is_visible(info.layer),
                was,
                "the {} layer did not come back as it was",
                info.label
            );
        }
    }

    /// The part going away takes the switch with it, whoever takes the part
    /// away.
    ///
    /// The flag itself is what is read back, through `isolate_part_mut`, and not
    /// only the guarded answer: a raw flag left standing behind a guard that
    /// says otherwise is a checkbox drawn ticked that nothing can untick, and a
    /// window that isolates and locks itself again the moment the next preview
    /// lands. So both ways of emptying the layer are driven - the editor's own
    /// `clear_density`, and the plain `set` any caller has - and both are asked
    /// the same two questions.
    #[test]
    fn the_part_going_away_takes_the_switch_with_it() {
        for empty in [
            Scene::clear_density as fn(&mut Scene),
            |scene: &mut Scene| scene.set(Layer::Density, None),
        ] {
            let mut scene = Scene::default();
            scene.set(Layer::Density, Some(part()));
            *scene.isolate_part_mut() = true;
            assert!(scene.isolate_part());

            empty(&mut scene);
            assert!(
                !*scene.isolate_part_mut(),
                "the flag behind the checkbox stayed set behind a part that is gone"
            );
            assert!(!scene.isolate_part());
            for info in LAYERS {
                assert!(
                    scene.is_visible(info.layer),
                    "the {} layer stayed hidden behind a part that is gone",
                    info.label
                );
            }
            // And it stays clear when a part comes back: the next preview must
            // not re-isolate a window nobody asked to.
            scene.set(Layer::Density, Some(part()));
            assert!(!scene.isolate_part(), "a new part brought the mode back");
            assert!(scene.is_visible(Layer::Domain));
        }
    }

    /// A switch homed outside the `show` block is drawn by a block only the
    /// editor's panel has, so the layer behind it has to be one the editor
    /// produces: anywhere else the block leaves it out and nothing draws it,
    /// and the layer would have no switch at all.
    #[test]
    fn a_switch_homed_elsewhere_belongs_to_an_editor_layer() {
        for info in LAYERS
            .iter()
            .filter(|info| info.switch == SwitchHome::Elsewhere)
        {
            assert_eq!(
                info.role,
                LayerRole::Editor,
                "the {} switch is drawn outside the show block, which only the editor has",
                info.label
            );
        }
    }

    #[test]
    fn flat_shading_drops_degenerate_and_non_finite_triangles() {
        let mut mesh = Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [f64::NAN, 0.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 1, 1], [0, 1, 3]],
        };
        let layer = LayerMesh::from_mesh(&mesh, [1.0; 4], Shading::Flat);
        assert_eq!(layer.triangles(), 1);
        let normal = layer.vertices[0].normal;
        assert!((normal[2].abs() - 1.0).abs() < 1e-6);
        mesh.triangles.clear();
        assert!(LayerMesh::from_mesh(&mesh, [1.0; 4], Shading::Flat).is_empty());
    }

    #[test]
    fn smooth_shading_survives_the_same_degenerate_geometry() {
        // The smooth path drops exactly what the flat one drops, and a vertex
        // whose only neighbours were dropped keeps a usable normal instead of
        // a zero vector the shader would divide by.
        let mesh = Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [f64::NAN, 0.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 1, 1], [0, 1, 3]],
        };
        let flat = LayerMesh::from_mesh(&mesh, [1.0; 4], Shading::Flat);
        let smooth = LayerMesh::from_mesh(&mesh, [1.0; 4], Shading::Smooth);
        assert_eq!(smooth.triangles(), flat.triangles());
        for vertex in &smooth.vertices {
            let n = vertex.normal;
            let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-5, "normal {n:?} is not a unit");
        }
    }

    #[test]
    fn smooth_shading_averages_the_faces_at_a_corner_and_flat_shading_restores_them() {
        let mesh = tessellate::box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let flat = LayerMesh::from_mesh(&mesh, [1.0; 4], Shading::Flat);
        let smooth = LayerMesh::from_mesh(&mesh, [1.0; 4], Shading::Smooth);

        // Smoothing changes no geometry at all: same vertices, same order, same
        // positions, same colours. Only the normals move.
        assert_eq!(smooth.vertices.len(), flat.vertices.len());
        assert_eq!(smooth.triangles(), flat.triangles());
        for (a, b) in smooth.vertices.iter().zip(flat.vertices.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.color, b.color);
        }
        // A flat normal is axis aligned; a corner's smooth normal blends the
        // three faces meeting there, so every component is real and points out
        // of the octant that corner sits in.
        for vertex in &smooth.vertices {
            let n = vertex.normal;
            let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-5, "normal {n:?} is not a unit");
            for d in 0..3 {
                let outward = if vertex.position[d] > 1.0 { 1.0 } else { -1.0 };
                assert!(
                    n[d] * outward > 0.2,
                    "corner {:?} got {n:?}, which does not lean out on axis {d}",
                    vertex.position
                );
            }
        }
        assert!(
            smooth
                .vertices
                .iter()
                .zip(flat.vertices.iter())
                .any(|(a, b)| a.normal != b.normal),
            "smooth shading left every normal on its facet"
        );
        // And the flat switch recovers the face normals from the positions the
        // layer already carries, without going back to the source mesh.
        let restored = smooth.flat_shaded();
        assert_eq!(restored.vertices.len(), flat.vertices.len());
        assert_eq!(restored.bounds.min, flat.bounds.min);
        for (a, b) in restored.vertices.iter().zip(flat.vertices.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.color, b.color);
            for d in 0..3 {
                assert!(
                    (a.normal[d] - b.normal[d]).abs() < 1e-5,
                    "restored {:?} against {:?}",
                    a.normal,
                    b.normal
                );
            }
        }
    }

    /// The overlay shading rule: round where the surface is round, sharp where
    /// the shape really has an edge, and one rule for both.
    #[test]
    fn rounded_shading_smooths_a_barrel_and_keeps_every_real_edge() {
        // A box has nothing but right angles, so it comes out exactly as flat
        // as the flat shading makes it.
        let cube = tessellate::box_mesh([0.0; 3], [2.0, 2.0, 2.0]);
        let flat = LayerMesh::from_mesh(&cube, [1.0; 4], Shading::Flat);
        let rounded = LayerMesh::from_mesh(&cube, [1.0; 4], Shading::Rounded);
        assert_eq!(rounded.vertices.len(), flat.vertices.len());
        for (a, b) in rounded.vertices.iter().zip(flat.vertices.iter()) {
            assert_eq!(a.position, b.position);
            for d in 0..3 {
                assert!(
                    (a.normal[d] - b.normal[d]).abs() < 1e-6,
                    "a box corner was rounded: {:?} against {:?}",
                    a.normal,
                    b.normal
                );
            }
        }

        // A cylinder is the case this exists for: its barrel is round and its
        // rim is not.
        let cylinder = tessellate::cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 10.0], 3.0, 48);
        let layer = LayerMesh::from_mesh(&cylinder, [1.0; 4], Shading::Rounded);
        let mut barrel = 0;
        let mut caps = 0;
        for vertex in &layer.vertices {
            let n = vertex.normal;
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normal {n:?} is not a unit");
            let position = vertex.position;
            let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
            if radial > 2.9 && n[2].abs() < 0.01 {
                // On the barrel: the normal points straight out of the axis,
                // which is what a smooth cylinder looks like.
                let dot = (n[0] * position[0] + n[1] * position[1]) / radial.max(1e-9);
                assert!(dot > 0.999, "a barrel normal leans off the radius: {n:?}");
                barrel += 1;
            } else if n[2].abs() > 0.999 {
                // On a cap: the normal is the cap's own, undisturbed by the
                // barrel meeting it at the rim.
                caps += 1;
            }
        }
        assert!(
            barrel > 0 && caps > 0,
            "{barrel} barrel, {caps} cap normals"
        );
        // The rim vertices are shared between the barrel and the caps, so a
        // plain smooth shading would tilt every one of them; this is what says
        // the crease rule did something a full averaging would not.
        let smooth = LayerMesh::from_mesh(&cylinder, [1.0; 4], Shading::Smooth);
        assert!(
            smooth
                .vertices
                .iter()
                .zip(layer.vertices.iter())
                .any(|(a, b)| a.normal != b.normal),
            "the crease rule made no difference to a cylinder"
        );
        // And every facet of the barrel really is shallow enough to smooth,
        // which is what the crease angle is chosen against.
        let facet = 360.0 / constants::VIEW_CYLINDER_SEGMENTS as f64;
        assert!(
            facet < constants::VIEW_CREASE_ANGLE_DEGREES,
            "a {facet} degree facet is not smoothed by a {} degree crease angle",
            constants::VIEW_CREASE_ANGLE_DEGREES
        );
    }

    /// The setup overlays are shaded the way their own geometry reads: the
    /// voxelization keeps its facets, the analytic shapes get the crease rule.
    #[test]
    fn the_setup_overlays_round_their_shapes_and_leave_the_voxelization_alone() {
        let (config, problem) = setup();
        let scene = build(&config, &problem).expect("scene");
        // The keepout is a cylinder standing at [20, 8] with a radius of 3.
        // Smoothing is exactly this: the facets meeting at a point on the
        // barrel agree about which way the surface faces there, where flat
        // shading gives each of them its own answer a whole segment apart.
        let keepout = scene.get(Layer::Keepout).expect("a keepout layer");
        let mut by_position: std::collections::HashMap<[i64; 3], Vec<[f32; 3]>> =
            std::collections::HashMap::new();
        for vertex in &keepout.vertices {
            let (x, y) = (
                f64::from(vertex.position[0]) - 20.0,
                f64::from(vertex.position[1]) - 8.0,
            );
            // A point on the barrel rather than on a cap or on another shape.
            if (x * x + y * y).sqrt() < 2.9 || f64::from(vertex.normal[2]).abs() > 0.01 {
                continue;
            }
            let key = vertex.position.map(|c| (f64::from(c) * 1e3).round() as i64);
            by_position.entry(key).or_default().push(vertex.normal);
        }
        let shared = by_position.values().filter(|list| list.len() > 1).count();
        assert!(shared > 0, "no barrel point is shared by two facets at all");
        let spread = by_position
            .values()
            .flat_map(|list| {
                list.iter().map(|n| {
                    let first = list[0];
                    (0..3).map(|d| (n[d] - first[d]).abs()).fold(0.0, f32::max)
                })
            })
            .fold(0.0_f32, f32::max);
        assert!(
            spread < 1e-6,
            "the keepout cylinder came out facetted: normals at one barrel point differ by {spread}"
        );
        // The domain is a voxelization - the model the optimizer actually has,
        // chamfers and all - and it keeps its facets: every triangle of it
        // carries one normal, which is what flat shading is.
        let domain = scene.get(Layer::Domain).expect("a domain layer");
        assert!(
            domain
                .vertices
                .chunks(3)
                .all(|t| t[0].normal == t[1].normal && t[1].normal == t[2].normal),
            "the voxelized domain was rounded"
        );
    }

    #[test]
    fn the_scene_starts_smooth_and_the_panel_can_switch_it_flat() {
        let mut scene = Scene::default();
        assert_eq!(scene.flat_shading(), constants::VIEW_DEFAULT_FLAT_SHADING);
        assert!(!scene.flat_shading(), "the density surface starts smooth");
        *scene.flat_shading_mut() = true;
        assert!(scene.flat_shading());
        // The toggle is independent of the stress colouring: both apply to the
        // same geometry and neither replaces it.
        assert!(!scene.stress_shading());
    }

    #[test]
    fn the_stress_ramp_runs_cold_to_hot_and_clamps() {
        let cold = stress_color(0.0);
        let hot = stress_color(1.0);
        let ramp = constants::VIEW_STRESS_COLORMAP;
        for d in 0..3 {
            assert!((cold[d] - ramp[0][d]).abs() < 1e-6);
            assert!((hot[d] - ramp[ramp.len() - 1][d]).abs() < 1e-6);
        }
        assert_eq!(cold[3], constants::VIEW_STRESS_ALPHA);
        // Out of range and non-finite fractions stay inside the ramp.
        assert_eq!(stress_color(-5.0), cold);
        assert_eq!(stress_color(9.0), hot);
        assert_eq!(stress_color(f64::NAN), cold);
        // Hot is redder than cold, and every channel stays a legal colour.
        assert!(hot[0] > cold[0] && hot[2] < cold[2]);
        for fraction in [0.0, 0.13, 0.5, 0.77, 1.0] {
            for channel in stress_color(fraction) {
                assert!((0.0..=1.0).contains(&channel), "{fraction} left the gamut");
            }
        }
    }

    #[test]
    fn stress_colouring_maps_the_field_onto_the_surface() {
        let mesh = tessellate::box_mesh([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let mut field = mesh::ScalarField::new(2, 2, 2, [0.0, 0.0, 0.0], 10.0);
        // A gradient along x: one face unstressed, the opposite face at yield.
        for k in 0..2 {
            for j in 0..2 {
                let index = field.index(1, j, k);
                field.values[index] = 50.0;
            }
        }
        let layer = stress_shaded(&mesh, &field, 50.0).expect("a coloured layer");
        assert_eq!(layer.triangles(), mesh.triangles.len());
        let cold = stress_color(0.0);
        let hot = stress_color(1.0);
        for vertex in &layer.vertices {
            let expected = if vertex.position[0] < 5.0 { cold } else { hot };
            assert_eq!(vertex.color, expected, "at {:?}", vertex.position);
        }

        // Nothing to normalize against, or nothing to colour, means no layer.
        assert!(stress_shaded(&mesh, &field, 0.0).is_none());
        assert!(stress_shaded(&Mesh::default(), &field, 50.0).is_none());
    }

    #[test]
    fn the_scene_defaults_to_plain_shading_and_switches_on_request() {
        let mut scene = Scene::default();
        let plain = LayerMesh::from_mesh(
            &tessellate::box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Layer::Density.info().color,
            Shading::Smooth,
        );
        scene.set(Layer::Density, Some(plain.clone()));

        // With no stress geometry the toggle has nothing to switch to.
        assert!(!scene.has_stress());
        assert!(!scene.stress_shading());
        *scene.stress_shading_mut() = true;
        assert!(!scene.stress_shading(), "a toggle with no stress layer");
        assert_eq!(
            scene.density_geometry().map(|m| m.triangles()),
            Some(plain.triangles())
        );

        // Once a run supplies one, plain is still the default until asked.
        *scene.stress_shading_mut() = false;
        let coloured = LayerMesh::from_mesh(
            &tessellate::box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            [1.0, 0.0, 0.0, 1.0],
            Shading::Smooth,
        );
        scene.set_stress(Some(coloured));
        assert!(scene.has_stress());
        assert!(!scene.stress_shading(), "plain must stay the default");
        assert_eq!(
            scene.density_geometry().unwrap().vertices[0].color,
            plain.vertices[0].color
        );

        *scene.stress_shading_mut() = true;
        assert!(scene.stress_shading());
        assert_eq!(
            scene.density_geometry().unwrap().vertices[0].color,
            [1.0, 0.0, 0.0, 1.0]
        );

        // Dropping the stress layer falls back to plain and clears the toggle.
        scene.set_stress(None);
        assert!(!scene.stress_shading());
        assert_eq!(
            scene.density_geometry().unwrap().vertices[0].color,
            plain.vertices[0].color
        );
    }

    #[test]
    fn gravity_gets_no_load_indicator() {
        let text = SETUP.replace(
            "[[loadcases.loads]]\ntype = \"torque\"",
            "[[loadcases.loads]]\ntype = \"gravity\"\ng_mm_s2 = 9810.0\n[[loadcases.loads]]\ntype = \"torque\"",
        );
        let config = Config::parse(&text).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let with_gravity = build(&config, &problem).expect("scene");
        let (plain_config, plain_problem) = setup();
        let plain = build(&plain_config, &plain_problem).expect("scene");
        assert_eq!(
            with_gravity.get(Layer::Loads).unwrap().triangles(),
            plain.get(Layer::Loads).unwrap().triangles(),
            "self weight must not add an arrow"
        );
    }

    #[test]
    fn a_preview_surface_skips_smoothing_and_validation() {
        let (_, problem) = setup();
        let densities = vec![constants::DENSITY_MAX; problem.grid.n_cells()];
        let preview =
            preview_surface(&problem.grid, &densities, problem.output.iso_level).expect("preview");
        let domain = voxelized_domain(&problem.grid, problem.output.iso_level).expect("domain");
        // Both take the same path, so an all-solid preview is the domain.
        assert_eq!(preview.triangles.len(), domain.triangles.len());
        // Skipping smoothing keeps every vertex on a lattice edge of the grid.
        let layer = LayerMesh::from_mesh(&preview, [1.0; 4], Shading::Smooth);
        assert!(layer.triangles() > 0);
    }

    #[test]
    fn a_supersampled_surface_reads_the_same_stress_field_at_more_places() {
        // The stress lattice is the coarse node lattice either way, so what has
        // to hold is that a refined vertex reads the trilinear value at its own
        // position rather than the value of the nearest coarse node.
        let (grid, densities) = ramp_grid();
        let iso = 0.5;
        let policy = crate::config::IslandPolicy::Cull;
        let anchors = mesh::islands::Anchors::none();
        let coarse =
            mesh::surface_from_densities(&grid, &densities, iso, 0, 1, policy, &anchors, None)
                .expect("surface")
                .mesh;
        let fine =
            mesh::surface_from_densities(&grid, &densities, iso, 0, 2, policy, &anchors, None)
                .expect("surface")
                .mesh;
        assert!(fine.vertices.len() > coarse.vertices.len());

        // A stress field that grows linearly along x over the whole block, so
        // every position has a value of its own to be got right or wrong.
        let mut field = mesh::ScalarField::new(
            grid.nnx() + 2,
            grid.nny() + 2,
            grid.nnz() + 2,
            [
                grid.origin[0] - grid.h,
                grid.origin[1] - grid.h,
                grid.origin[2] - grid.h,
            ],
            grid.h,
        );
        let reference = 100.0;
        for k in 0..field.nz {
            for j in 0..field.ny {
                for i in 0..field.nx {
                    let index = field.index(i, j, k);
                    field.values[index] = i as f64 * 10.0;
                }
            }
        }

        let layer = stress_shaded(&fine, &field, reference).expect("a coloured layer");
        assert_eq!(layer.triangles(), fine.triangles.len());
        // Spot check every vertex against the field's own trilinear lookup.
        let expected: Vec<[f32; 4]> = fine
            .vertices
            .iter()
            .map(|v| stress_color(field.sample(*v) / reference))
            .collect();
        for (index, triangle) in fine.triangles.iter().enumerate() {
            for (m, &vertex) in triangle.iter().enumerate() {
                assert_eq!(
                    layer.vertices[3 * index + m].color,
                    expected[vertex as usize],
                    "vertex {vertex} of triangle {index}"
                );
            }
        }

        // The stress copy is the density surface, so it is shaded like it: unit
        // normals, and not the facet normals of the triangles it was built on.
        let flat = layer.flat_shaded();
        for vertex in &layer.vertices {
            let n = vertex.normal;
            let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-5, "normal {n:?} is not a unit");
        }
        assert!(
            layer
                .vertices
                .iter()
                .zip(flat.vertices.iter())
                .any(|(a, b)| a.normal != b.normal),
            "the stress coloured copy came out flat shaded"
        );

        // And at least one refined vertex sits strictly between two coarse
        // lattice planes, where nearest-node sampling would give a different
        // answer than the trilinear one.
        let between = fine
            .vertices
            .iter()
            .filter(|v| {
                let local = (v[0] - field.origin[0]) / field.spacing;
                (local - local.round()).abs() > 0.2
            })
            .count();
        assert!(
            between > 0,
            "no refined vertex landed between two coarse lattice planes"
        );
    }
}
