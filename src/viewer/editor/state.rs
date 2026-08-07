//! What the editor is editing: the configuration, what is selected in it, the
//! undo stack, and the validation and problem summary derived from it.
//!
//! The state owns two views of the same document. [`Config`] is the typed model
//! everything downstream reads - validation, voxelization, the scene, the
//! engines - and [`toml_io::Document`] is the file's own text, kept so that a
//! save preserves the comments and the formatting of everything the user did
//! not touch. Structural edits (adding and deleting objects) are applied to
//! both together; scalar edits reach the document when it is written.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::config::{
    Axis, Config, CsgOpSpec, DomainEntry, LoadCaseSpec, LoadSpec, ShapeSpec, SupportSpec,
};
use crate::constants;
use crate::geometry::{Aabb, Shape, Vec3};
use crate::problem::Problem;
use crate::viewer::editor::{snap, toml_io};

/// One object of the configuration, as the tree and the viewport address it.
///
/// The index is the object's position in its own list, which is also the
/// position of the table that describes it in the document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// `[[domain]]` entry.
    Domain(usize),
    /// `[[keepout]]` entry.
    Keepout(usize),
    /// `[[keepin]]` entry.
    Keepin(usize),
    /// `[[supports]]` entry.
    Support(usize),
    /// A `[[loadcases]]` entry itself, whose properties are its name and its
    /// weight rather than a shape.
    LoadCase(usize),
    /// One `[[loadcases.loads]]` entry inside a load case.
    Load {
        /// Index of the load case.
        case: usize,
        /// Index of the load inside that case.
        load: usize,
    },
}

/// What kind of object an add button creates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewObject {
    /// A `[[domain]]` entry of the given shape and boolean operation.
    Domain(ShapeKind, CsgOpSpec),
    /// A `[[keepout]]` entry.
    Keepout(ShapeKind),
    /// A `[[keepin]]` entry.
    Keepin(ShapeKind),
    /// A `[[supports]]` entry.
    Support(ShapeKind),
    /// A `[[loadcases]]` entry, with no loads in it yet.
    LoadCase,
    /// A `[[loadcases.loads]]` entry inside the given load case.
    Load(usize, LoadKind),
}

impl NewObject {
    /// Spelling of what this object is, as the add row's own button labels it.
    pub fn shape_label(self) -> &'static str {
        match self {
            NewObject::Domain(kind, _)
            | NewObject::Keepout(kind)
            | NewObject::Keepin(kind)
            | NewObject::Support(kind) => kind.label(),
            NewObject::LoadCase => "load case",
            NewObject::Load(_, kind) => kind.label(),
        }
    }

    /// Name of the list this object is added to, as the tree heads it.
    ///
    /// What the placement mode says while it waits for the clicks that finish
    /// it: four add rows offer the same button, and which list the object lands
    /// in is the one thing the row itself no longer says once the pointer has
    /// left it.
    pub fn list_label(self) -> &'static str {
        match self {
            NewObject::Domain(_, _) => "domain",
            NewObject::Keepout(_) => "keepout",
            NewObject::Keepin(_) => "keepin",
            NewObject::Support(_) => "supports",
            NewObject::LoadCase => "load cases",
            NewObject::Load(_, _) => "loads",
        }
    }
}

/// The shape primitives a region can be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeKind {
    /// Axis aligned box.
    Box,
    /// Capped cylinder.
    Cylinder,
    /// Sphere.
    Sphere,
    /// Ellipsoid.
    Ellipsoid,
    /// Tube: a capsule, bendable into an arc.
    Tube,
    /// Cone: a capped frustum, with a radius at each end.
    Cone,
    /// Triangle: a triangular prism, extruded about its own plane.
    Triangle,
}

impl ShapeKind {
    /// Config spelling.
    pub fn label(self) -> &'static str {
        match self {
            ShapeKind::Box => "box",
            ShapeKind::Cylinder => "cylinder",
            ShapeKind::Sphere => "sphere",
            ShapeKind::Ellipsoid => "ellipsoid",
            ShapeKind::Tube => "tube",
            ShapeKind::Cone => "cone",
            ShapeKind::Triangle => "triangle",
        }
    }

    /// Every kind, in menu order.
    pub const ALL: [ShapeKind; 7] = [
        ShapeKind::Box,
        ShapeKind::Cylinder,
        ShapeKind::Sphere,
        ShapeKind::Ellipsoid,
        ShapeKind::Tube,
        ShapeKind::Cone,
        ShapeKind::Triangle,
    ];
}

/// The three load types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadKind {
    /// A total force in newtons.
    Force,
    /// A total torque in newton-millimetres.
    Torque,
    /// Self weight.
    Gravity,
}

impl LoadKind {
    /// Config spelling.
    pub fn label(self) -> &'static str {
        match self {
            LoadKind::Force => "force",
            LoadKind::Torque => "torque",
            LoadKind::Gravity => "gravity",
        }
    }

    /// Every kind, in menu order.
    pub const ALL: [LoadKind; 3] = [LoadKind::Force, LoadKind::Torque, LoadKind::Gravity];
}

impl Selection {
    /// Order the viewport picks overlapping objects in, lowest first.
    ///
    /// What sits *inside* something else is picked before what contains it: a
    /// load or support region usually sits inside a keepin, so depth alone
    /// would make the outer one the only thing a click could ever reach.
    pub fn pick_rank(self) -> u8 {
        match self {
            Selection::Load { .. } | Selection::Support(_) | Selection::LoadCase(_) => 0,
            Selection::Keepout(_) | Selection::Keepin(_) => 1,
            // Never reached by a ray; see `pickable`.
            Selection::Domain(_) => 2,
        }
    }
}

/// True when a selection still addresses an object that is there.
///
/// A selection outlives the edit that made it - an undo, a delete, a file
/// reloaded under it - and every panel that shows one indexes with it, so this
/// is what keeps a stale index from reaching an index expression.
pub fn exists(config: &Config, selection: Selection) -> bool {
    match selection {
        Selection::Domain(i) => i < config.domain.len(),
        Selection::Keepout(i) => i < config.keepout.len(),
        Selection::Keepin(i) => i < config.keepin.len(),
        Selection::Support(i) => i < config.supports.len(),
        Selection::LoadCase(i) => i < config.loadcases.len(),
        Selection::Load { case, load } => config
            .loadcases
            .get(case)
            .is_some_and(|spec| load < spec.loads.len()),
    }
}

/// Shape of the region an object carries, if it has one.
pub fn shape_of(config: &Config, selection: Selection) -> Option<ShapeSpec> {
    match selection {
        Selection::Domain(i) => config.domain.get(i).map(DomainEntry::shape_spec),
        Selection::Keepout(i) => config.keepout.get(i).cloned(),
        Selection::Keepin(i) => config.keepin.get(i).cloned(),
        Selection::Support(i) => config.supports.get(i).map(|s| s.region.clone()),
        Selection::LoadCase(_) => None,
        Selection::Load { case, load } => config
            .loadcases
            .get(case)
            .and_then(|c| c.loads.get(load))
            .and_then(|l| l.region())
            .cloned(),
    }
}

/// Replace the region of an object that has one; a no-op for the others.
pub fn set_shape(config: &mut Config, selection: Selection, spec: ShapeSpec) {
    match selection {
        Selection::Domain(i) => {
            if let Some(entry) = config.domain.get_mut(i) {
                *entry = domain_entry(spec, domain_op(entry));
            }
        }
        Selection::Keepout(i) => {
            if let Some(entry) = config.keepout.get_mut(i) {
                *entry = spec;
            }
        }
        Selection::Keepin(i) => {
            if let Some(entry) = config.keepin.get_mut(i) {
                *entry = spec;
            }
        }
        Selection::Support(i) => {
            if let Some(entry) = config.supports.get_mut(i) {
                entry.region = spec;
            }
        }
        Selection::LoadCase(_) => {}
        Selection::Load { case, load } => {
            if let Some(entry) = config
                .loadcases
                .get_mut(case)
                .and_then(|c| c.loads.get_mut(load))
            {
                match entry {
                    LoadSpec::Force { region, .. } => *region = spec,
                    LoadSpec::Torque { region, .. } => *region = spec,
                    LoadSpec::Gravity { .. } => {}
                }
            }
        }
    }
}

/// Rebuild a domain entry from a shape and an operation.
pub fn domain_entry(spec: ShapeSpec, op: CsgOpSpec) -> DomainEntry {
    match spec {
        ShapeSpec::Box {
            min,
            max,
            rotation_deg,
        } => DomainEntry::Box {
            op,
            min,
            max,
            rotation_deg,
        },
        ShapeSpec::Cylinder { p1, p2, radius } => DomainEntry::Cylinder { op, p1, p2, radius },
        ShapeSpec::Sphere { center, radius } => DomainEntry::Sphere { op, center, radius },
        ShapeSpec::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } => DomainEntry::Ellipsoid {
            op,
            center,
            radii,
            rotation_deg,
        },
        ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        } => DomainEntry::Tube {
            op,
            p1,
            p2,
            bend,
            radius,
        },
        ShapeSpec::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => DomainEntry::Cone {
            op,
            p1,
            p2,
            radius1,
            radius2,
        },
        ShapeSpec::Triangle { a, b, c, thickness } => DomainEntry::Triangle {
            op,
            a,
            b,
            c,
            thickness,
        },
    }
}

/// Boolean operation of a domain entry, in its config spelling.
///
/// [`DomainEntry::op`] resolves to the geometry crate's operation; the editor
/// needs the one the file writes and the dropdown offers.
pub fn domain_op(entry: &DomainEntry) -> CsgOpSpec {
    match *entry {
        DomainEntry::Box { op, .. }
        | DomainEntry::Cylinder { op, .. }
        | DomainEntry::Sphere { op, .. }
        | DomainEntry::Ellipsoid { op, .. }
        | DomainEntry::Tube { op, .. }
        | DomainEntry::Cone { op, .. }
        | DomainEntry::Triangle { op, .. } => op,
    }
}

/// Short label of an object, as the tree shows it.
pub fn label_of(config: &Config, selection: Selection) -> String {
    let shape = |spec: Option<ShapeSpec>| match spec {
        Some(spec) => spec.kind(),
        None => "-",
    };
    match selection {
        Selection::Domain(i) => {
            let op = match config.domain.get(i).map(domain_op) {
                Some(CsgOpSpec::Subtract) => "subtract",
                _ => "add",
            };
            format!("{} {} {}", i + 1, op, shape(shape_of(config, selection)))
        }
        Selection::Keepout(i) | Selection::Keepin(i) => {
            format!("{} {}", i + 1, shape(shape_of(config, selection)))
        }
        Selection::Support(i) => {
            let directions = config.supports.get(i).and_then(|s| s.directions.as_ref());
            let axes = match directions {
                Some(list) => list.iter().map(|a| a.label()).collect::<Vec<_>>().join(""),
                None => Axis::ALL
                    .iter()
                    .map(|a| a.label())
                    .collect::<Vec<_>>()
                    .join(""),
            };
            format!("{} {} [{axes}]", i + 1, shape(shape_of(config, selection)))
        }
        Selection::LoadCase(i) => match config.loadcases.get(i) {
            Some(case) => format!("{} \"{}\"", i + 1, case.name),
            None => format!("{}", i + 1),
        },
        Selection::Load { case, load } => {
            let kind = config
                .loadcases
                .get(case)
                .and_then(|c| c.loads.get(load))
                .map(LoadSpec::kind)
                .unwrap_or("-");
            format!("{} {kind}", load + 1)
        }
    }
}

/// Every object of a configuration a click in the viewport can pick, in tree
/// order.
///
/// A load case is not pickable itself - it has no geometry - and neither is a
/// gravity load, which acts on the whole structure.
///
/// **The domain is not pickable either.** Every other object lives inside it,
/// so a domain shape under the pointer is nearly always *also* under whatever
/// the user was aiming at; selecting it from the tree instead keeps it from
/// taking clicks meant for the objects inside it. Once selected there it is
/// dragged and resized in the viewport like anything else - it is the
/// selecting that is restricted, not the editing.
pub fn pickable(config: &Config) -> Vec<Selection> {
    let mut out = Vec::new();
    out.extend((0..config.keepout.len()).map(Selection::Keepout));
    out.extend((0..config.keepin.len()).map(Selection::Keepin));
    out.extend((0..config.supports.len()).map(Selection::Support));
    for (case, spec) in config.loadcases.iter().enumerate() {
        for load in 0..spec.loads.len() {
            let selection = Selection::Load { case, load };
            if shape_of(config, selection).is_some() {
                out.push(selection);
            }
        }
    }
    out
}

/// A pickable object and the geometry the ray is tested against.
#[derive(Debug, Clone, Copy)]
pub struct Target {
    /// The object this geometry belongs to.
    pub selection: Selection,
    /// Its shape.
    pub shape: Shape,
}

/// Pick targets for every object whose shape is well formed enough to
/// intersect. A degenerate shape is simply not pickable until it is fixed.
pub fn targets(config: &Config) -> Vec<Target> {
    pickable(config)
        .into_iter()
        .filter_map(|selection| {
            let spec = shape_of(config, selection)?;
            let shape = spec.to_shape("pick").ok()?;
            Some(Target { selection, shape })
        })
        .collect()
}

/// Pick targets a placement click may land on: every one of [`targets`], and
/// the domain's own additive entries as well.
///
/// A different question from the one [`targets`] answers, and so a different
/// list. Selecting the domain by clicking it would take clicks meant for the
/// objects inside it; *landing a new point on* it is the opposite - the design
/// space is the surface most of a model is drawn against, and a click on it is
/// aimed at nothing else.
///
/// Subtractive domain entries are left out: such an entry is a hole cut in the
/// design space rather than a surface, and nothing draws it, so landing a point
/// on one would be landing on geometry that is not on screen.
pub fn placement_targets(config: &Config) -> Vec<Target> {
    let mut out: Vec<Target> = Vec::new();
    for (index, entry) in config.domain.iter().enumerate() {
        if domain_op(entry) != CsgOpSpec::Add {
            continue;
        }
        let spec = entry.shape_spec();
        if let Ok(shape) = spec.to_shape("place") {
            out.push(Target {
                selection: Selection::Domain(index),
                shape,
            });
        }
    }
    out.extend(targets(config));
    out
}

/// Bounds objects are kept inside of: the domain's own, when it has usable
/// ones.
///
/// Deliberately not [`reference_bounds`], which falls back to a cube about the
/// origin so that a new object has a size to be given. Nothing may be clamped
/// against a fallback: a configuration whose domain does not build yet is one
/// where "inside the domain" has no meaning, and the answer is to leave the
/// user's numbers alone.
pub fn containment_bounds(config: &Config) -> Option<Aabb> {
    let bounds = config.domain_csg().ok()?.bounds();
    let extent = bounds.extent();
    let usable = !bounds.is_empty()
        && extent
            .iter()
            .all(|e| e.is_finite() && *e > constants::MIN_SHAPE_EXTENT_MM);
    usable.then_some(bounds)
}

/// The surfaces a dragged object may land flush against.
///
/// Only the regions that describe *where a load enters the structure or where
/// it is held* get them: a support or a load region is placed against a face -
/// the top of a load pad, the wall of the design space - and half a millimetre
/// off that face is a region that selects a different set of nodes. A keepout
/// or a keepin is a piece of the model in its own right and is placed by its
/// own numbers, so it gets none and is snapped to the grid alone.
///
/// The candidates are the faces of the keepins and the outside of the domain.
pub fn surfaces(config: &Config, selection: Selection) -> snap::Surfaces {
    let mut out = snap::Surfaces::default();
    if !matches!(selection, Selection::Support(_) | Selection::Load { .. }) {
        return out;
    }
    if let Some(bounds) = containment_bounds(config) {
        out.push_bounds(&bounds, snap::SurfaceKind::Domain);
    }
    for (index, spec) in config.keepin.iter().enumerate() {
        if let Ok(shape) = spec.to_shape("surface") {
            out.push_bounds(&shape.bounds(), snap::SurfaceKind::Keepin(index));
        }
    }
    out
}

/// True when an object of this kind is one containment applies to.
///
/// The domain is what everything else is kept inside of, so it is not kept
/// inside anything; a load case has no geometry at all.
pub fn is_contained_kind(selection: Selection) -> bool {
    !matches!(selection, Selection::Domain(_) | Selection::LoadCase(_))
}

/// `spec` moved, if it has to be, so that its bounding box lies inside
/// `bounds`. Returns the shape and whether it had to be moved.
///
/// Translation only: an object is never resized to fit, because a size is
/// something the user set on purpose. An object larger than the domain along an
/// axis cannot be contained on that axis at all, and is centred on it instead -
/// which at least puts what does not fit equally either side.
///
/// A rotated box is clamped by the bounds of the box *as it is turned*, which
/// is the space it actually occupies.
pub fn clamped_into(spec: &ShapeSpec, bounds: &Aabb) -> (ShapeSpec, bool) {
    let Ok(shape) = spec.to_shape("containment") else {
        // A half-typed shape has no bounds to clamp; the properties panel is
        // where it gets finished.
        return (spec.clone(), false);
    };
    let current = shape.bounds();
    if current.is_empty() {
        return (spec.clone(), false);
    }
    let mut delta = [0.0; 3];
    for (d, slot) in delta.iter_mut().enumerate() {
        let extent = current.max[d] - current.min[d];
        let room = bounds.max[d] - bounds.min[d];
        *slot = if extent > room {
            // Too large to fit: centred, so it overhangs evenly.
            0.5 * (bounds.min[d] + bounds.max[d]) - 0.5 * (current.min[d] + current.max[d])
        } else if current.min[d] < bounds.min[d] {
            bounds.min[d] - current.min[d]
        } else if current.max[d] > bounds.max[d] {
            bounds.max[d] - current.max[d]
        } else {
            0.0
        };
    }
    if delta.iter().all(|d| *d == 0.0) {
        return (spec.clone(), false);
    }
    (crate::viewer::editor::gizmo::translate(spec, delta), true)
}

/// Replace an object's region, keeping it inside the domain when containment is
/// on. Returns whether the commit had to be moved to keep it there.
///
/// Every path that commits a shape - a gizmo drag, a number typed into the
/// properties panel, a number typed into a callout - goes through here, so
/// there is one answer to "may this object be outside the domain" rather than
/// three.
pub fn set_shape_contained(
    config: &mut Config,
    selection: Selection,
    spec: ShapeSpec,
    containment: bool,
) -> bool {
    let bounds = (containment && is_contained_kind(selection))
        .then(|| containment_bounds(config))
        .flatten();
    let (spec, clamped) = match bounds {
        Some(bounds) => clamped_into(&spec, &bounds),
        None => (spec, false),
    };
    set_shape(config, selection, spec);
    clamped
}

/// Bounding box the editor sizes new objects and gizmos against: the domain's
/// own bounds, or a cube of [`constants::VIEW_EDIT_FALLBACK_EXTENT_MM`] about
/// the origin when there is no usable domain yet.
pub fn reference_bounds(config: &Config) -> Aabb {
    let bounds = config
        .domain_csg()
        .map(|csg| csg.bounds())
        .unwrap_or_else(|_| Aabb::empty());
    if bounds.is_empty() || !bounds.volume().is_finite() || bounds.volume() <= 0.0 {
        let half = 0.5 * constants::VIEW_EDIT_FALLBACK_EXTENT_MM;
        return Aabb {
            min: [-half; 3],
            max: [half; 3],
        };
    }
    bounds
}

/// Centre of `bounds`.
fn centre(bounds: &Aabb) -> Vec3 {
    [
        0.5 * (bounds.min[0] + bounds.max[0]),
        0.5 * (bounds.min[1] + bounds.max[1]),
        0.5 * (bounds.min[2] + bounds.max[2]),
    ]
}

/// Edge length a new object is given: a fraction of the smallest side of the
/// bounds it is sized against.
///
/// Its own function because two paths need it - the add rows, which drop a
/// default shape at the centre of the domain, and the viewport placement, which
/// is handed the geometry but still has to size the rest of the shape - and one
/// answer is what keeps an added tube and a placed one the same tube.
pub fn default_size(config: &Config) -> f64 {
    let extent = reference_bounds(config).extent();
    let smallest = extent
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .max(constants::VIEW_EDIT_MIN_EXTENT_MM);
    smallest * constants::VIEW_EDIT_NEW_OBJECT_SIZE_FRACTION
}

/// Radius a new tube is given, however it is created.
pub fn default_tube_radius(config: &Config) -> f64 {
    constants::VIEW_EDIT_NEW_TUBE_RADIUS_FRACTION * default_size(config)
}

/// A default shape of the requested kind, sized against the domain and placed
/// at its centre.
pub fn default_shape(config: &Config, kind: ShapeKind) -> ShapeSpec {
    let bounds = reference_bounds(config);
    let size = default_size(config);
    let c = centre(&bounds);
    match kind {
        ShapeKind::Box => ShapeSpec::Box {
            min: [c[0] - 0.5 * size, c[1] - 0.5 * size, c[2] - 0.5 * size],
            max: [c[0] + 0.5 * size, c[1] + 0.5 * size, c[2] + 0.5 * size],
            // Absent rather than [0, 0, 0]: a new object writes the keys it
            // needs and no others.
            rotation_deg: None,
        },
        ShapeKind::Cylinder => ShapeSpec::Cylinder {
            p1: [c[0], c[1], c[2] - 0.5 * size],
            p2: [c[0], c[1], c[2] + 0.5 * size],
            radius: 0.5 * size,
        },
        ShapeKind::Sphere => ShapeSpec::Sphere {
            center: c,
            radius: 0.5 * size,
        },
        // A round one, sized exactly as the sphere is: the three radius handles
        // are what shape it, and a default that already fits the domain on
        // every axis is one nothing has to be dragged back into.
        ShapeKind::Ellipsoid => ShapeSpec::Ellipsoid {
            center: c,
            radii: [0.5 * size; 3],
            // Absent rather than [0, 0, 0], for the reason the box gives above.
            rotation_deg: None,
        },
        // Straight, and occupying the same `size` as every other default: the
        // rounded ends are part of its length, so the segment between them is
        // that much shorter. The radius comes from [`default_tube_radius`] -
        // the one answer to how thick a new tube is, wherever it was created -
        // and the segment is one of those either side of the centre.
        ShapeKind::Tube => {
            let radius = default_tube_radius(config);
            ShapeSpec::Tube {
                p1: [c[0], c[1], c[2] - radius],
                p2: [c[0], c[1], c[2] + radius],
                // Absent rather than the midpoint: a new tube is straight, and
                // the file says so by carrying no key at all.
                bend: None,
                radius,
            }
        }
        // The cylinder's sizing, tapered: the same axis and the same wide end,
        // with the narrow one a fraction of it
        // ([`constants::VIEW_EDIT_NEW_CONE_TAPER_FRACTION`]). A default that
        // came out looking like a cylinder would read as an add that failed.
        ShapeKind::Cone => {
            let radius1 = 0.5 * size;
            ShapeSpec::Cone {
                p1: [c[0], c[1], c[2] - 0.5 * size],
                p2: [c[0], c[1], c[2] + 0.5 * size],
                radius1,
                radius2: constants::VIEW_EDIT_NEW_CONE_TAPER_FRACTION * radius1,
            }
        }
        // Equilateral, laid flat, with a **side** of `size`: the longest span
        // of an equilateral triangle is its side, so this is the one dimension
        // every other default occupies. Its circumradius is what that side
        // makes it - `size / sqrt(3)` - because the triangle is sized by the
        // edge that can be seen rather than by a circle nothing draws.
        ShapeKind::Triangle => {
            let circumradius = size / 3.0_f64.sqrt();
            ShapeSpec::Triangle {
                a: [c[0], c[1] + circumradius, c[2]],
                b: [c[0] - 0.5 * size, c[1] - 0.5 * circumradius, c[2]],
                c: [c[0] + 0.5 * size, c[1] - 0.5 * circumradius, c[2]],
                thickness: constants::VIEW_EDIT_NEW_TRIANGLE_THICKNESS_FRACTION * size,
            }
        }
    }
}

/// A default load of the requested kind, anchored on the domain's centre.
pub fn default_load(config: &Config, kind: LoadKind) -> LoadSpec {
    let region = default_shape(config, ShapeKind::Sphere);
    let axis_point = centre(&reference_bounds(config));
    match kind {
        LoadKind::Force => LoadSpec::Force {
            region,
            vector: [0.0, 0.0, -constants::VIEW_EDIT_NEW_FORCE_N],
        },
        LoadKind::Torque => LoadSpec::Torque {
            region,
            axis_point,
            axis_dir: [0.0, 0.0, 1.0],
            magnitude_nmm: constants::VIEW_EDIT_NEW_TORQUE_NMM,
        },
        LoadKind::Gravity => LoadSpec::Gravity {
            direction: None,
            g_mm_s2: None,
        },
    }
}

/// One entry of the undo stack: everything an edit can change.
#[derive(Debug, Clone)]
struct Snapshot {
    config: Config,
    document: toml_io::Document,
    selection: Option<Selection>,
}

/// A bounded undo stack with the usual redo semantics.
#[derive(Debug, Default)]
struct History {
    undo: std::collections::VecDeque<Snapshot>,
    redo: Vec<Snapshot>,
}

impl History {
    /// Record a state to come back to, dropping the oldest step once the stack
    /// is full and discarding any redo branch.
    fn push(&mut self, snapshot: Snapshot) {
        self.redo.clear();
        if constants::VIEW_EDIT_UNDO_DEPTH == 0 {
            return;
        }
        while self.undo.len() >= constants::VIEW_EDIT_UNDO_DEPTH {
            self.undo.pop_front();
        }
        self.undo.push_back(snapshot);
    }
}

/// What the editor is editing.
#[derive(Debug)]
pub struct EditorState {
    path: PathBuf,
    directory: PathBuf,
    config: Config,
    document: toml_io::Document,
    selection: Option<Selection>,
    /// The configuration as the file on disk holds it: what was read at open,
    /// and what was written by the last save. Everything that asks whether
    /// there is anything to save compares against this rather than against a
    /// flag, so an edit undone all the way back is not "unsaved changes".
    saved: Config,
    history: History,
    /// Snapshot taken when the current interaction started, and the widget that
    /// owns it. One interaction - a drag, a number typed into a field - is one
    /// undo step however many values it passes through.
    pending: Option<(Snapshot, u64)>,
    /// Why the configuration is invalid, when it is.
    problem_error: Option<String>,
    /// The discrete problem of the last configuration that produced one.
    problem: Option<Problem>,
    /// Set when the configuration has changed and the setup has not caught up.
    stale: bool,
}

impl EditorState {
    /// Read a configuration file and prepare it for editing.
    ///
    /// The file must parse: an editor over a document it cannot represent would
    /// have nothing to write back. A file that parses but does not validate is
    /// opened, and says why in the validation panel.
    pub fn open(path: &Path) -> Result<EditorState> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        EditorState::from_text(path, &text)
    }

    /// Create `path` from the starter configuration and open it.
    ///
    /// Refuses to touch a file that is already there: scaffolding is for a name
    /// that means nothing yet, and an existing configuration - even an empty or
    /// broken one - is the user's.
    pub fn create(path: &Path) -> Result<EditorState> {
        if path.exists() {
            bail!(
                "{} already exists; `growforge edit` only writes a starter configuration for a \
                 path that is not there yet",
                path.display()
            );
        }
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .filter(|stem| !stem.trim().is_empty())
            .unwrap_or_else(|| constants::PROGRAM_NAME.to_string());
        let text = crate::config::starter_config(&name);
        if let Some(directory) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(directory)
                .with_context(|| format!("creating {}", directory.display()))?;
        }
        std::fs::write(path, &text)
            .with_context(|| format!("writing a starter configuration to {}", path.display()))?;
        println!("wrote a starter configuration to {}", path.display());
        EditorState::from_text(path, &text)
    }

    /// Open a configuration whose text is already in hand.
    fn from_text(path: &Path, text: &str) -> Result<EditorState> {
        let config = Config::parse(text)
            .with_context(|| format!("parsing config file {}", path.display()))?;
        let document = toml_io::Document::parse(text)
            .with_context(|| format!("parsing config file {}", path.display()))?;
        let mut state = EditorState {
            path: path.to_path_buf(),
            directory: crate::config_dir(path),
            saved: config.clone(),
            config,
            document,
            selection: None,
            history: History::default(),
            pending: None,
            problem_error: None,
            problem: None,
            stale: true,
        };
        state.revalidate();
        Ok(state)
    }

    /// The configuration being edited.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The configuration being edited, for a widget that is about to change it.
    /// The caller is responsible for having opened an interaction first.
    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    /// Path of the file this is editing.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Directory relative output paths resolve against.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Where a full run would write its STL, resolved against that directory.
    pub fn output_path(&self) -> PathBuf {
        let raw = Path::new(&self.config.output.stl_path);
        if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            self.directory.join(raw)
        }
    }

    /// True while the edited configuration differs from the file on disk.
    ///
    /// Computed rather than latched: an edit that has been undone back to the
    /// state the file holds is not an unsaved change, and saying it is would
    /// put a modal in front of someone with nothing to lose.
    pub fn is_dirty(&self) -> bool {
        self.config != self.saved
    }

    /// The selected object, and only ever one that is still there.
    pub fn selection(&self) -> Option<Selection> {
        self.selection.filter(|s| exists(&self.config, *s))
    }

    /// Select an object, or nothing.
    pub fn select(&mut self, selection: Option<Selection>) {
        self.selection = selection;
    }

    /// Why the configuration cannot be turned into a problem, when it cannot.
    pub fn error(&self) -> Option<&str> {
        self.problem_error.as_deref()
    }

    /// True when the configuration as it stands is a complete, buildable
    /// problem: something the editor may re-run, and something `growforge run`
    /// would accept.
    pub fn is_valid(&self) -> bool {
        self.problem_error.is_none() && self.problem.is_some()
    }

    /// The discrete problem of the last valid configuration.
    pub fn problem(&self) -> Option<&Problem> {
        self.problem.as_ref()
    }

    /// True while an edit is waiting for the setup to be rebuilt.
    pub fn is_stale(&self) -> bool {
        self.stale
    }

    /// Undo steps available.
    pub fn undo_depth(&self) -> usize {
        self.history.undo.len()
    }

    /// Redo steps available.
    pub fn redo_depth(&self) -> usize {
        self.history.redo.len()
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            config: self.config.clone(),
            document: self.document.clone(),
            selection: self.selection,
        }
    }

    fn restore(&mut self, snapshot: Snapshot) {
        self.config = snapshot.config;
        self.document = snapshot.document;
        self.selection = snapshot.selection;
        self.stale = true;
    }

    /// Open an interaction owned by `widget`, closing any other one first.
    ///
    /// Nothing is recorded until the interaction ends, so a value dragged
    /// through fifty intermediate numbers is still one undo step.
    pub fn begin_edit(&mut self, widget: u64) {
        match &self.pending {
            Some((_, owner)) if *owner == widget => {}
            Some(_) => {
                self.end_edit_any();
                self.pending = Some((self.snapshot(), widget));
            }
            None => self.pending = Some((self.snapshot(), widget)),
        }
    }

    /// Close the interaction owned by `widget`, recording it as one undo step.
    pub fn end_edit(&mut self, widget: u64) {
        if matches!(&self.pending, Some((_, owner)) if *owner == widget) {
            self.end_edit_any();
        }
    }

    /// Close whatever interaction is open, if any.
    pub fn end_edit_any(&mut self) {
        if let Some((snapshot, _)) = self.pending.take() {
            self.history.push(snapshot);
            self.stale = true;
        }
    }

    /// True while an interaction is open.
    pub fn is_editing(&self) -> bool {
        self.pending.is_some()
    }

    /// Mark the configuration as changed without opening an interaction, so the
    /// setup is rebuilt on the next refresh.
    pub fn touch(&mut self) {
        self.stale = true;
    }

    /// Apply one whole edit as a single undo step.
    pub fn edit<T>(&mut self, change: impl FnOnce(&mut Config) -> T) -> T {
        self.end_edit_any();
        let snapshot = self.snapshot();
        let out = change(&mut self.config);
        self.history.push(snapshot);
        self.stale = true;
        out
    }

    /// Undo the last edit, restoring the configuration and the selection it was
    /// made with. Returns false when there is nothing to undo.
    pub fn undo(&mut self) -> bool {
        self.end_edit_any();
        let Some(snapshot) = self.history.undo.pop_back() else {
            return false;
        };
        let current = self.snapshot();
        self.history.redo.push(current);
        self.restore(snapshot);
        true
    }

    /// Redo the last undone edit. Returns false when there is nothing to redo.
    pub fn redo(&mut self) -> bool {
        self.end_edit_any();
        let Some(snapshot) = self.history.redo.pop() else {
            return false;
        };
        let current = self.snapshot();
        self.history.undo.push_back(current);
        self.restore(snapshot);
        true
    }

    /// Add an object of the requested kind and select it.
    ///
    /// The configuration and the document are changed together, so the table
    /// that describes an object stays the table that describes it however many
    /// objects are added and deleted before the file is written.
    pub fn add(&mut self, what: NewObject) {
        self.add_with(what, None, false);
    }

    /// Add an object whose region the caller has placed itself, rather than the
    /// default shape at the centre of the domain, and select it.
    ///
    /// What the viewport's two-click placement commits. The shape goes through
    /// [`set_shape_contained`] like every other commit, so a placement outside
    /// the domain is kept inside it exactly as a drag out there would be;
    /// returns whether it had to be moved, which is what the panel's note is
    /// timed against.
    ///
    /// `what` still decides the list and the document table; only the geometry
    /// is the caller's. An object with no region of its own - a load case -
    /// ignores the shape, because there is nowhere to put it.
    pub fn add_placed(&mut self, what: NewObject, shape: ShapeSpec, containment: bool) -> bool {
        self.add_with(what, Some(shape), containment)
    }

    /// The one add path: push the object, then replace its region when the
    /// caller brought one.
    fn add_with(&mut self, what: NewObject, placed: Option<ShapeSpec>, containment: bool) -> bool {
        self.end_edit_any();
        let snapshot = self.snapshot();
        let selection = match what {
            NewObject::Domain(kind, op) => {
                let shape = default_shape(&self.config, kind);
                self.config.domain.push(domain_entry(shape, op));
                self.document.push_object(toml_io::List::Domain);
                Selection::Domain(self.config.domain.len() - 1)
            }
            NewObject::Keepout(kind) => {
                let shape = default_shape(&self.config, kind);
                self.config.keepout.push(shape);
                self.document.push_object(toml_io::List::Keepout);
                Selection::Keepout(self.config.keepout.len() - 1)
            }
            NewObject::Keepin(kind) => {
                let shape = default_shape(&self.config, kind);
                self.config.keepin.push(shape);
                self.document.push_object(toml_io::List::Keepin);
                Selection::Keepin(self.config.keepin.len() - 1)
            }
            NewObject::Support(kind) => {
                let region = default_shape(&self.config, kind);
                self.config.supports.push(SupportSpec {
                    region,
                    directions: None,
                });
                self.document.push_object(toml_io::List::Supports);
                Selection::Support(self.config.supports.len() - 1)
            }
            NewObject::LoadCase => {
                let index = self.config.loadcases.len();
                self.config.loadcases.push(LoadCaseSpec {
                    name: format!("{}{}", constants::VIEW_EDIT_NEW_CASE_PREFIX, index + 1),
                    weight: None,
                    loads: Vec::new(),
                });
                self.document.push_object(toml_io::List::LoadCases);
                Selection::LoadCase(index)
            }
            NewObject::Load(case, kind) => {
                let load = default_load(&self.config, kind);
                let Some(spec) = self.config.loadcases.get_mut(case) else {
                    return false;
                };
                spec.loads.push(load);
                let index = spec.loads.len() - 1;
                self.document.push_object(toml_io::List::Loads(case));
                Selection::Load { case, load: index }
            }
        };
        // Committed through the same gate every other shape commit uses, so
        // there is one answer to "may this object be outside the domain"
        // whether it was placed, dragged or typed.
        let clamped = match placed {
            Some(spec) => set_shape_contained(&mut self.config, selection, spec, containment),
            None => false,
        };
        self.history.push(snapshot);
        self.selection = Some(selection);
        self.stale = true;
        clamped
    }

    /// Delete the selected object, clearing the selection.
    pub fn delete(&mut self, selection: Selection) {
        self.end_edit_any();
        let snapshot = self.snapshot();
        let removed = match selection {
            Selection::Domain(i) if i < self.config.domain.len() => {
                self.config.domain.remove(i);
                self.document.remove_object(toml_io::List::Domain, i);
                true
            }
            Selection::Keepout(i) if i < self.config.keepout.len() => {
                self.config.keepout.remove(i);
                self.document.remove_object(toml_io::List::Keepout, i);
                true
            }
            Selection::Keepin(i) if i < self.config.keepin.len() => {
                self.config.keepin.remove(i);
                self.document.remove_object(toml_io::List::Keepin, i);
                true
            }
            Selection::Support(i) if i < self.config.supports.len() => {
                self.config.supports.remove(i);
                self.document.remove_object(toml_io::List::Supports, i);
                true
            }
            Selection::LoadCase(i) if i < self.config.loadcases.len() => {
                self.config.loadcases.remove(i);
                self.document.remove_object(toml_io::List::LoadCases, i);
                true
            }
            Selection::Load { case, load } => match self.config.loadcases.get_mut(case) {
                Some(spec) if load < spec.loads.len() => {
                    spec.loads.remove(load);
                    self.document
                        .remove_object(toml_io::List::Loads(case), load);
                    true
                }
                _ => false,
            },
            _ => false,
        };
        if !removed {
            return;
        }
        self.history.push(snapshot);
        self.selection = None;
        self.stale = true;
    }

    /// Re-derive the validation state and the discrete problem.
    ///
    /// The last problem that built is kept when the new configuration does not,
    /// so the viewport still shows the model while a half-typed number is
    /// rejected; the panel says what is wrong either way.
    pub fn revalidate(&mut self) {
        self.stale = false;
        match Problem::build(&self.config, &self.directory) {
            Ok(problem) => {
                self.problem = Some(problem);
                self.problem_error = None;
            }
            Err(error) => self.problem_error = Some(format!("{error:#}")),
        }
    }

    /// Write the configuration back to the file it came from.
    ///
    /// Only the values that changed are written; the comments, the key order
    /// and the formatting of everything else survive. Nothing else is written -
    /// a save is never an export.
    pub fn save(&mut self) -> Result<()> {
        self.end_edit_any();
        // A projection that could not write a value fails here, before the file
        // is touched: the edits stay unsaved, and the panel says why.
        self.document.sync(&self.config)?;
        let text = self.document.render();
        std::fs::write(&self.path, text)
            .with_context(|| format!("writing config file {}", self.path.display()))?;
        // What the file now holds, and therefore what "unsaved" is measured
        // against from here. A write that failed leaves it alone, so the edits
        // are still unsaved and still asked about.
        self.saved = self.config.clone();
        Ok(())
    }

    /// Where an emergency save puts the document: beside the file being edited,
    /// under its own name and [`constants::VIEW_EDIT_RECOVERY_EXTENSION`].
    pub fn recovery_path(&self) -> PathBuf {
        self.path
            .with_extension(constants::VIEW_EDIT_RECOVERY_EXTENSION)
    }

    /// Write the configuration as it stands to [`EditorState::recovery_path`],
    /// replacing whatever was there, and answer where it went.
    ///
    /// The emergency exit, taken when the window is dying with unsaved changes
    /// in it - see the viewer's fatal path. It produces exactly what a save
    /// would have written, through the same format preserving projection, and
    /// it is the one thing about it that matters: what comes back is the
    /// session's own file, comments and all, ready to be renamed over the
    /// original by someone who has read it.
    ///
    /// **Nothing here touches the session.** The projection runs on a copy of
    /// the document, the file being edited is not written to, and what "unsaved"
    /// is measured against is left where it was: a rescue is not a save, and a
    /// session that survived one must still know it has changes to lose. A
    /// projection that cannot write a value fails here, before any file is
    /// touched, exactly as a save does.
    pub fn write_recovery(&self) -> Result<PathBuf> {
        let mut document = self.document.clone();
        document.sync(&self.config)?;
        let path = self.recovery_path();
        std::fs::write(&path, document.render())
            .with_context(|| format!("writing recovered config file {}", path.display()))?;
        Ok(path)
    }

    /// The lines of the live problem summary, or the reason there is none.
    pub fn summary(&self) -> Vec<String> {
        let Some(problem) = &self.problem else {
            return Vec::new();
        };
        let grid = &problem.grid;
        let mut lines = vec![
            format!(
                "grid        {} x {} x {} at {:.3} mm",
                grid.nx, grid.ny, grid.nz, grid.h
            ),
            format!(
                "cells       {} ({} design, {} solid, {} void)",
                grid.n_cells(),
                problem.counts.design,
                problem.counts.solid,
                problem.counts.void
            ),
            format!("nodes       {} ({} dof)", grid.n_nodes(), grid.n_dof()),
            format!(
                "memory      {:.1} MiB estimated",
                problem.estimated_memory_bytes() as f64 / constants::BYTES_PER_MIB
            ),
        ];
        for support in &problem.supports {
            lines.push(format!(
                "support {}   {} nodes",
                support.index, support.node_count
            ));
        }
        for (index, case) in problem.load_cases.iter().enumerate() {
            for load in &case.loads {
                lines.push(format!(
                    "load {}.{}   {} nodes",
                    index + 1,
                    load.kind,
                    load.node_count
                ));
            }
        }
        lines
    }

    /// Non-fatal problems the last successful build reported.
    pub fn warnings(&self) -> &[String] {
        match &self.problem {
            Some(problem) => &problem.warnings,
            None => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::editor::tests::{fixture, write_temp};

    #[test]
    fn opening_a_configuration_validates_it_and_selects_nothing() {
        let (_dir, path) = write_temp("open", fixture());
        let state = EditorState::open(&path).expect("open");
        assert!(state.is_valid() && state.error().is_none());
        assert!(state.selection().is_none());
        assert!(!state.is_dirty());
        assert_eq!(state.undo_depth(), 0);
        let summary = state.summary();
        assert!(summary.iter().any(|l| l.starts_with("grid")));
        assert!(summary.iter().any(|l| l.contains("dof")));
        assert!(summary.iter().any(|l| l.starts_with("support 1")));
        assert!(summary.iter().any(|l| l.contains("load 1.force")));
    }

    #[test]
    fn a_configuration_that_does_not_parse_is_refused_rather_than_half_opened() {
        let (_dir, path) = write_temp("broken", "[project]\nname = ");
        assert!(EditorState::open(&path).is_err());
    }

    #[test]
    fn every_object_type_can_be_added_edited_and_deleted() {
        let (_dir, path) = write_temp("objects", fixture());
        let mut state = EditorState::open(&path).expect("open");

        let before = (
            state.config().domain.len(),
            state.config().keepout.len(),
            state.config().keepin.len(),
            state.config().supports.len(),
            state.config().loadcases.len(),
        );
        for (what, kind) in [
            (
                NewObject::Domain(ShapeKind::Box, CsgOpSpec::Subtract),
                "domain",
            ),
            (NewObject::Keepout(ShapeKind::Cylinder), "keepout"),
            (NewObject::Keepin(ShapeKind::Sphere), "keepin"),
            (NewObject::Support(ShapeKind::Ellipsoid), "supports"),
            (NewObject::LoadCase, "loadcase"),
        ] {
            state.add(what);
            assert!(
                state.selection().is_some(),
                "{kind}: adding must select what it added"
            );
        }
        let case = state.config().loadcases.len() - 1;
        for kind in LoadKind::ALL {
            state.add(NewObject::Load(case, kind));
        }
        assert_eq!(state.config().domain.len(), before.0 + 1);
        assert_eq!(state.config().keepout.len(), before.1 + 1);
        assert_eq!(state.config().keepin.len(), before.2 + 1);
        assert_eq!(state.config().supports.len(), before.3 + 1);
        assert_eq!(state.config().loadcases.len(), before.4 + 1);
        assert_eq!(state.config().loadcases[case].loads.len(), 3);
        assert!(state.is_dirty());

        // Every added object is well formed as far as anything that does not
        // depend on the voxelization can tell. Whether *this* pile of them,
        // all dropped on the domain's centre on top of each other, still
        // selects nodes is a different question, and the next test is where
        // each addition is put through the whole build on its own.
        state
            .config()
            .validate_static()
            .expect("every added object is a shape the validation path accepts");

        // A shape edit through the same path a gizmo drag takes.
        let selection = Selection::Keepout(state.config().keepout.len() - 1);
        let ShapeSpec::Cylinder { p1, p2, radius } = shape_of(state.config(), selection).unwrap()
        else {
            panic!("the keepout was added as a cylinder");
        };
        state.edit(|config| {
            set_shape(
                config,
                selection,
                ShapeSpec::Cylinder {
                    p1,
                    p2,
                    radius: radius * 2.0,
                },
            );
        });
        let ShapeSpec::Cylinder { radius: edited, .. } =
            shape_of(state.config(), selection).unwrap()
        else {
            panic!("the shape kind must not change under an edit");
        };
        assert!((edited - 2.0 * radius).abs() < 1e-12);
        state
            .config()
            .validate_static()
            .expect("an edited shape is still a shape");

        // And deleting every one of them takes the configuration back to what
        // it started as.
        for load in (0..3).rev() {
            state.delete(Selection::Load { case, load });
        }
        state.delete(Selection::LoadCase(case));
        state.delete(Selection::Support(state.config().supports.len() - 1));
        state.delete(Selection::Keepin(state.config().keepin.len() - 1));
        state.delete(Selection::Keepout(state.config().keepout.len() - 1));
        state.delete(Selection::Domain(state.config().domain.len() - 1));
        assert_eq!(
            (
                state.config().domain.len(),
                state.config().keepout.len(),
                state.config().keepin.len(),
                state.config().supports.len(),
                state.config().loadcases.len(),
            ),
            before
        );
        state.revalidate();
        assert!(state.is_valid(), "{:?}", state.error());
        assert!(state.selection().is_none(), "a delete clears the selection");
    }

    /// Every kind of object, added on its own to a configuration that builds,
    /// has to leave one that still builds: a default that lands the user in an
    /// error is a broken default.
    #[test]
    fn one_added_object_leaves_a_configuration_that_still_builds() {
        let (_dir, path) = write_temp("addable", fixture());
        for what in [
            NewObject::Domain(ShapeKind::Box, CsgOpSpec::Add),
            NewObject::Domain(ShapeKind::Cylinder, CsgOpSpec::Subtract),
            NewObject::Domain(ShapeKind::Ellipsoid, CsgOpSpec::Subtract),
            NewObject::Domain(ShapeKind::Tube, CsgOpSpec::Subtract),
            NewObject::Domain(ShapeKind::Cone, CsgOpSpec::Subtract),
            NewObject::Domain(ShapeKind::Triangle, CsgOpSpec::Subtract),
            NewObject::Keepout(ShapeKind::Sphere),
            NewObject::Keepout(ShapeKind::Ellipsoid),
            NewObject::Keepout(ShapeKind::Tube),
            NewObject::Keepout(ShapeKind::Cone),
            NewObject::Keepout(ShapeKind::Triangle),
            NewObject::Keepin(ShapeKind::Box),
            NewObject::Keepin(ShapeKind::Cone),
            NewObject::Support(ShapeKind::Box),
            NewObject::Support(ShapeKind::Triangle),
        ] {
            let mut state = EditorState::open(&path).expect("open");
            state.add(what);
            state.revalidate();
            assert!(state.is_valid(), "{what:?}: {:?}", state.error());
        }
        // A load case is only a problem once it has a load in it, so the two
        // are added together - which is also the only order the tree offers.
        for kind in LoadKind::ALL {
            let mut state = EditorState::open(&path).expect("open");
            state.add(NewObject::LoadCase);
            let case = state.config().loadcases.len() - 1;
            state.add(NewObject::Load(case, kind));
            state.revalidate();
            assert!(state.is_valid(), "{kind:?}: {:?}", state.error());
        }
        // And an empty load case is rejected, which is the editor showing the
        // user what growforge itself would say.
        let mut state = EditorState::open(&path).expect("open");
        state.add(NewObject::LoadCase);
        state.revalidate();
        assert!(!state.is_valid());
        assert!(state.error().expect("a reason").contains("loads"));
    }

    #[test]
    fn an_added_object_lands_where_the_domain_is() {
        let (_dir, path) = write_temp("defaults", fixture());
        let mut state = EditorState::open(&path).expect("open");
        let bounds = reference_bounds(state.config());
        state.add(NewObject::Keepin(ShapeKind::Box));
        let selection = state.selection().expect("selected");
        let shape = shape_of(state.config(), selection)
            .expect("a shape")
            .to_shape("test")
            .expect("well formed");
        let box_bounds = shape.bounds();
        for d in 0..3 {
            assert!(
                box_bounds.min[d] >= bounds.min[d] && box_bounds.max[d] <= bounds.max[d],
                "axis {d}: {box_bounds:?} is not inside the domain {bounds:?}"
            );
        }
        // And it is a usable size rather than a point.
        assert!(shape.min_extent() > constants::VIEW_EDIT_MIN_EXTENT_MM);
    }

    /// Every shape kind the add rows offer produces a default that fits inside
    /// the domain and is a usable size - the tube included, whose rounded ends
    /// are part of its length.
    #[test]
    fn every_default_shape_is_a_usable_size_inside_the_domain() {
        let (_dir, path) = write_temp("every_default", fixture());
        let state = EditorState::open(&path).expect("open");
        let bounds = reference_bounds(state.config());
        let size = bounds
            .extent()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            * constants::VIEW_EDIT_NEW_OBJECT_SIZE_FRACTION;
        for kind in ShapeKind::ALL {
            let spec = default_shape(state.config(), kind);
            assert_eq!(spec.kind(), kind.label());
            let shape = spec.to_shape("test").expect("well formed");
            let own = shape.bounds();
            for d in 0..3 {
                assert!(
                    own.min[d] >= bounds.min[d] && own.max[d] <= bounds.max[d],
                    "{kind:?} axis {d}: {own:?} is not inside {bounds:?}"
                );
            }
            assert!(shape.min_extent() > constants::VIEW_EDIT_MIN_EXTENT_MM);
            // Each of them occupies that one size on its longest axis, so no
            // kind lands noticeably bigger or smaller than another.
            let longest = own.extent().iter().copied().fold(0.0_f64, f64::max);
            assert!(
                (longest - size).abs() < 1e-9,
                "{kind:?} spans {longest} rather than {size}"
            );
        }
        // A new tube is straight, and its ends are far enough apart to leave a
        // middle handle to drag: the rounded caps are the rest of its length.
        let ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        } = default_shape(state.config(), ShapeKind::Tube)
        else {
            panic!("a tube");
        };
        assert_eq!(bend, None, "a new tube carries no bend key");
        assert!(
            (crate::geometry::length(crate::geometry::difference(p2, p1)) - 0.5 * size).abs()
                < 1e-9
        );
        assert!((radius - 0.25 * size).abs() < 1e-9);

        // A new cone tapers visibly: one that came out looking like a cylinder
        // would read as an add that failed.
        let ShapeSpec::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } = default_shape(state.config(), ShapeKind::Cone)
        else {
            panic!("a cone");
        };
        assert!((crate::geometry::length(crate::geometry::difference(p2, p1)) - size).abs() < 1e-9);
        assert!((radius1 - 0.5 * size).abs() < 1e-9);
        assert!(
            (radius2 - constants::VIEW_EDIT_NEW_CONE_TAPER_FRACTION * radius1).abs() < 1e-9,
            "a new cone has to be visibly a cone: {radius1} to {radius2}"
        );
        assert!(radius2 > 0.0, "and not already at its apex");

        // A new triangle is equilateral, with a side of `size`, and thick
        // enough to be a solid rather than a sheet.
        let ShapeSpec::Triangle { a, b, c, thickness } =
            default_shape(state.config(), ShapeKind::Triangle)
        else {
            panic!("a triangle");
        };
        for (from, to) in [(a, b), (b, c), (c, a)] {
            let side = crate::geometry::length(crate::geometry::difference(to, from));
            assert!(
                (side - size).abs() < 1e-9,
                "a side of {side} rather than {size}"
            );
        }
        assert!(
            (thickness - constants::VIEW_EDIT_NEW_TRIANGLE_THICKNESS_FRACTION * size).abs() < 1e-9
        );
        // Laid flat, so it reads as a plate on the floor grid rather than as a
        // shape edge-on to the camera.
        assert!((a[2] - b[2]).abs() < 1e-12 && (b[2] - c[2]).abs() < 1e-12);
    }

    /// "Unsaved changes" has to mean what it says. An edit undone all the way
    /// back leaves the file's own configuration in front of the user, and a
    /// modal claiming otherwise would be false.
    #[test]
    fn dirty_means_the_configuration_differs_from_the_file() {
        let (_dir, path) = write_temp("dirty", fixture());
        let mut state = EditorState::open(&path).expect("open");
        assert!(!state.is_dirty());

        state.edit(|config| config.optimization.mass_fraction = Some(0.44));
        assert!(state.is_dirty());
        assert!(state.undo());
        assert!(
            !state.is_dirty(),
            "an edit undone back to the saved state is not an unsaved change"
        );
        // Redoing it makes it one again.
        assert!(state.redo());
        assert!(state.is_dirty());

        // Saving moves the mark: the file now holds this, and undoing past it
        // is a divergence from the file even though it is where we started.
        state.save().expect("save");
        assert!(!state.is_dirty());
        assert!(state.undo());
        assert!(
            state.is_dirty(),
            "undoing past a save differs from what the file now holds"
        );
        assert!(state.redo());
        assert!(!state.is_dirty());

        // And a value typed back to what it was by hand is not a change either.
        state.edit(|config| config.optimization.min_feature_mm = 20.0);
        assert!(state.is_dirty());
        state.edit(|config| config.optimization.min_feature_mm = 16.0);
        assert!(!state.is_dirty());
    }

    /// The panel's own path for a number too large for a TOML integer: an
    /// interaction, a save, and a file that still reads back as what was typed.
    #[test]
    fn a_count_typed_past_the_signed_range_saves_and_reads_back() {
        let huge = 12345678901234567890_u64 as usize;
        let (_dir, path) = write_temp("huge_count", fixture());
        let mut state = EditorState::open(&path).expect("open");
        let widget = 3;
        state.begin_edit(widget);
        state.config_mut().optimization.max_iterations = Some(huge);
        state.end_edit(widget);
        assert!(state.is_dirty());
        state.save().expect("save");

        let written = std::fs::read_to_string(&path).expect("read");
        let reparsed = Config::parse(&written).expect("the save must still be readable");
        assert_eq!(reparsed.optimization.max_iterations, Some(huge));
        assert!(!state.is_dirty());
        // The file the editor wrote is one the editor can open again, and one
        // growforge still validates.
        let reopened = EditorState::open(&path).expect("reopen");
        assert!(reopened.is_valid(), "{:?}", reopened.error());
    }

    /// A number that would ask for a grid no machine can hold is a thing
    /// someone can type into the resolution field. It has to come back as a
    /// line in the validation panel, not as the window disappearing: the
    /// allocation it used to attempt aborted the process, unsaved edits and
    /// all.
    #[test]
    fn a_resolution_that_would_run_the_grid_away_is_reported_not_attempted() {
        let (_dir, path) = write_temp("runaway_grid", fixture());
        let mut state = EditorState::open(&path).expect("open");
        let good = state.problem().expect("a problem").grid.n_cells();

        state.edit(|config| {
            config.resolution.voxel_size_mm = None;
            config.resolution.target_cells = Some(12345678901234567890_u64 as usize);
        });
        state.revalidate();
        assert!(!state.is_valid());
        let error = state.error().expect("a reason");
        assert!(error.contains("target_cells"), "unexpected error: {error}");
        assert!(
            error.contains(&constants::MAX_GRID_CELLS.to_string()),
            "the panel must quote the budget: {error}"
        );
        // The last model that built is still there to look at, and undoing puts
        // the configuration back.
        assert_eq!(state.problem().expect("a problem").grid.n_cells(), good);
        assert!(state.undo());
        state.revalidate();
        assert!(state.is_valid(), "{:?}", state.error());

        // The same through the other key.
        state.edit(|config| config.resolution.voxel_size_mm = Some(1e-4));
        state.revalidate();
        assert!(!state.is_valid());
        assert!(
            state.error().expect("a reason").contains("voxel_size_mm"),
            "{:?}",
            state.error()
        );
    }

    /// A configuration holding a value that is not a number cannot be compared
    /// with itself, which is what the modified marker is made of. It never gets
    /// that far: the file is refused at the door, naming the key.
    #[test]
    fn a_file_holding_a_number_that_is_not_one_is_refused_at_the_door() {
        let (_dir, path) = write_temp(
            "not_a_number",
            &fixture().replace("mass_fraction = 0.3", "mass_fraction = nan"),
        );
        // Formatted as the command line formats it, chain and all.
        let error = format!("{:#}", EditorState::open(&path).unwrap_err());
        assert!(error.contains("mass_fraction"), "unexpected error: {error}");
        assert!(error.contains("finite"), "unexpected error: {error}");

        // The same for a load the solver would otherwise have been handed.
        let (_dir, path) = write_temp(
            "not_a_number_load",
            &fixture().replace("vector = [0.0, 0.0, -100.0]", "vector = [0.0, 0.0, nan]"),
        );
        let error = format!("{:#}", EditorState::open(&path).unwrap_err());
        assert!(error.contains("vector[2]"), "unexpected error: {error}");
    }

    /// A save that could not write a value must say so and stay unsaved. The
    /// alternative - reporting success while the file keeps the old number - is
    /// the one outcome an editor may never produce.
    #[test]
    fn a_save_that_cannot_be_written_fails_and_stays_unsaved() {
        // A file that already spells every stand-in the editor has for a number
        // past the signed range, so the next one cannot be given one.
        let reserved: Vec<String> = (0..constants::VIEW_EDIT_MAX_BIG_INTEGERS as i64 * 2)
            .map(|offset| (i64::MAX - offset).to_string())
            .collect();
        let text = format!(
            "# reserved: {}\n{}\n[growth]\nseed = 1\n",
            reserved.join(" "),
            fixture()
        );
        let (_dir, path) = write_temp("no_sentinel", &text);
        let mut state = EditorState::open(&path).expect("open");
        state.edit(|config| config.growth.as_mut().expect("growth").seed = Some(u64::MAX));

        let error = state.save().unwrap_err().to_string();
        assert!(error.contains("seed"), "unexpected error: {error}");
        assert!(state.is_dirty(), "a failed save may not clear the marker");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            text,
            "a failed save may not touch the file"
        );
    }

    /// The emergency save the viewer's fatal path takes: the same document a
    /// save would have written, beside the file rather than into it, and a
    /// session left believing exactly what it believed before.
    #[test]
    fn an_emergency_save_writes_what_a_save_would_have_written_and_touches_nothing_else() {
        let (_dir, path) = write_temp("rescue", fixture());
        let before = std::fs::read(&path).expect("read");
        let mut state = EditorState::open(&path).expect("open");
        state.edit(|config| config.optimization.mass_fraction = Some(0.44));
        state.edit(|config| config.optimization.min_feature_mm = 20.0);

        let recovered = state.write_recovery().expect("the rescue writes");
        assert_eq!(
            recovered,
            path.with_extension(constants::VIEW_EDIT_RECOVERY_EXTENSION)
        );
        assert!(
            recovered.ends_with("config.recovered.toml"),
            "an unmistakable name of its own: {}",
            recovered.display()
        );
        assert_eq!(
            recovered.parent(),
            path.parent(),
            "the copy belongs beside the file it came from"
        );
        assert_eq!(
            std::fs::read(&path).expect("read"),
            before,
            "a rescue may never write to the file being edited"
        );
        assert!(
            state.is_dirty(),
            "a rescue is not a save: the changes are still unsaved"
        );

        // And what it holds is what the save it stood in for produces, byte for
        // byte - the format preserving projection, not a re-serialization.
        let rescued = std::fs::read(&recovered).expect("read");
        state.save().expect("save");
        assert_eq!(std::fs::read(&path).expect("read"), rescued);
        assert!(
            Config::parse(&String::from_utf8(rescued).expect("utf8")).is_ok(),
            "what is recovered has to be a configuration that opens"
        );
    }

    /// A second rescue replaces the first: what is beside the file is the last
    /// thing that was in the window, never an older one kept by accident.
    #[test]
    fn an_emergency_save_replaces_the_one_before_it() {
        let (_dir, path) = write_temp("rescue_again", fixture());
        let mut state = EditorState::open(&path).expect("open");
        let recovered = state.recovery_path();
        std::fs::write(&recovered, "# whatever was here before\n").expect("write");

        state.edit(|config| config.optimization.mass_fraction = Some(0.44));
        assert_eq!(
            state.write_recovery().expect("the rescue writes"),
            recovered
        );
        let text = std::fs::read_to_string(&recovered).expect("read");
        assert!(
            !text.contains("whatever was here before"),
            "the previous recovery survived: {text}"
        );
        assert_eq!(
            Config::parse(&text)
                .expect("parse")
                .optimization
                .mass_fraction,
            Some(0.44)
        );
    }

    /// A rescue that cannot be written says so and leaves everything alone.
    /// Its caller is a window that is already dying, so the one thing it may
    /// not do is panic on the way.
    #[test]
    fn an_emergency_save_that_cannot_be_written_reports_it() {
        let (_dir, path) = write_temp("rescue_blocked", fixture());
        let mut state = EditorState::open(&path).expect("open");
        state.edit(|config| config.optimization.mass_fraction = Some(0.44));

        // A directory in the way of the file, which is a write that fails on
        // every platform without needing permissions arranged.
        let blocked = state.recovery_path();
        std::fs::create_dir(&blocked).expect("directory");
        let error = format!("{:#}", state.write_recovery().unwrap_err());
        assert!(
            error.contains(&blocked.display().to_string()),
            "the report has to name the file: {error}"
        );
        assert!(state.is_dirty());

        // The other way it can fail is the projection itself, which fails
        // before any file is touched - see `save`. Nothing is left half
        // written in its name either.
        std::fs::remove_dir(&blocked).expect("remove");
        let reserved: Vec<String> = (0..constants::VIEW_EDIT_MAX_BIG_INTEGERS as i64 * 2)
            .map(|offset| (i64::MAX - offset).to_string())
            .collect();
        let text = format!(
            "# reserved: {}\n{}\n[growth]\nseed = 1\n",
            reserved.join(" "),
            fixture()
        );
        let (_dir, path) = write_temp("rescue_no_sentinel", &text);
        let mut state = EditorState::open(&path).expect("open");
        state.edit(|config| config.growth.as_mut().expect("growth").seed = Some(u64::MAX));
        let error = state.write_recovery().unwrap_err().to_string();
        assert!(error.contains("seed"), "unexpected error: {error}");
        assert!(
            !state.recovery_path().exists(),
            "a projection that failed may not leave a file behind"
        );
    }

    /// Containment clamps an object's bounding box into the domain's, on every
    /// face, by translation alone.
    #[test]
    fn containment_clamps_an_object_at_each_face_of_the_domain() {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [100.0, 60.0, 40.0],
        };
        let inside = ShapeSpec::Box {
            min: [10.0, 10.0, 10.0],
            max: [20.0, 20.0, 20.0],
            rotation_deg: None,
        };
        let (kept, clamped) = clamped_into(&inside, &bounds);
        assert!(!clamped, "an object already inside is not moved");
        assert_eq!(kept, inside);

        // Out through each of the six faces in turn, and back to touching it.
        for (delta, expected) in [
            ([-50.0, 0.0, 0.0], [0.0, 10.0, 10.0]),
            ([200.0, 0.0, 0.0], [90.0, 10.0, 10.0]),
            ([0.0, -50.0, 0.0], [10.0, 0.0, 10.0]),
            ([0.0, 200.0, 0.0], [10.0, 50.0, 10.0]),
            ([0.0, 0.0, -50.0], [10.0, 10.0, 0.0]),
            ([0.0, 0.0, 200.0], [10.0, 10.0, 30.0]),
        ] {
            let moved = crate::viewer::editor::gizmo::translate(&inside, delta);
            let (kept, clamped) = clamped_into(&moved, &bounds);
            assert!(clamped, "{delta:?} was not clamped");
            let ShapeSpec::Box { min, max, .. } = kept else {
                panic!("a box");
            };
            assert_eq!(min, expected, "{delta:?}");
            assert_eq!(
                [max[0] - min[0], max[1] - min[1], max[2] - min[2]],
                [10.0; 3],
                "containment may never resize an object"
            );
        }
    }

    /// An object too large for the domain on some axis cannot be contained on
    /// it; it is centred instead, so what does not fit hangs out evenly.
    #[test]
    fn an_object_larger_than_the_domain_is_centred_on_that_axis() {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [100.0, 60.0, 40.0],
        };
        let huge = ShapeSpec::Box {
            min: [500.0, 5.0, 5.0],
            max: [700.0, 15.0, 15.0],
            rotation_deg: None,
        };
        let (kept, clamped) = clamped_into(&huge, &bounds);
        assert!(clamped);
        let ShapeSpec::Box { min, max, .. } = kept else {
            panic!("a box");
        };
        // 200 mm wide in a 100 mm domain: centred on x, left alone on y and z.
        assert!((0.5 * (min[0] + max[0]) - 50.0).abs() < 1e-9, "{min:?}");
        assert_eq!([min[1], min[2]], [5.0, 5.0]);
    }

    /// A rotated box takes up the room its *turned* bounding box does, and that
    /// is what has to fit.
    #[test]
    fn a_rotated_box_is_clamped_by_the_room_it_really_takes_up() {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [100.0, 100.0, 100.0],
        };
        // A 20 x 4 x 4 bar turned 45 degrees about z reaches
        // 0.5 * (20 + 4) / sqrt(2) = 8.49 mm either side of its centre on x and
        // y, so a centre at x = 4 is 4.49 mm outside the wall.
        let turned = ShapeSpec::Box {
            min: [-6.0, 48.0, 48.0],
            max: [14.0, 52.0, 52.0],
            rotation_deg: Some([0.0, 0.0, 45.0]),
        };
        let reach = 0.5 * (20.0 + 4.0) / 2.0_f64.sqrt();
        let (kept, clamped) = clamped_into(&turned, &bounds);
        assert!(clamped, "the turned bar was not clamped");
        let inner = kept.to_shape("test").expect("a shape").bounds();
        assert!(inner.min[0] >= -1e-9, "{inner:?}");
        assert!((inner.min[0]).abs() < 1e-9, "clamped onto the wall");
        assert!((inner.max[0] - 2.0 * reach).abs() < 1e-9);
        // Left alone, the very same bar without the rotation fits: the turn is
        // what put it outside.
        let ShapeSpec::Box { min, max, .. } = turned else {
            panic!("a box");
        };
        let square = ShapeSpec::Box {
            min,
            max,
            rotation_deg: None,
        };
        let (_, clamped) = clamped_into(&square, &bounds);
        assert!(clamped, "the square bar starts outside on x too");
        // ... but by less, because it reaches less far.
        let (kept, _) = clamped_into(&square, &bounds);
        let inner = kept.to_shape("test").expect("a shape").bounds();
        assert!((inner.max[0] - 20.0).abs() < 1e-9, "{inner:?}");
    }

    /// A rotated ellipsoid takes up the room its support function says it does,
    /// and that is the box containment has to fit into the domain.
    #[test]
    fn a_rotated_ellipsoid_is_clamped_by_the_room_it_really_takes_up() {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [100.0, 100.0, 100.0],
        };
        // Semi-axes of 10 and 2 turned 45 degrees about z reach
        // sqrt(10^2 cos^2 45 + 2^2 sin^2 45) = 7.211 mm either side on x, so a
        // centre at x = 4 is 3.211 mm outside the wall.
        let turned = ShapeSpec::Ellipsoid {
            center: [4.0, 50.0, 50.0],
            radii: [10.0, 2.0, 2.0],
            rotation_deg: Some([0.0, 0.0, 45.0]),
        };
        let reach = (100.0_f64 * 0.5 + 4.0 * 0.5).sqrt();
        let (kept, clamped) = clamped_into(&turned, &bounds);
        assert!(clamped, "the turned ellipsoid was not clamped");
        let inner = kept.to_shape("test").expect("a shape").bounds();
        assert!(
            inner.min[0].abs() < 1e-9,
            "clamped onto the wall: {inner:?}"
        );
        assert!((inner.max[0] - 2.0 * reach).abs() < 1e-9, "{inner:?}");
        let ShapeSpec::Ellipsoid { radii, .. } = kept else {
            panic!("an ellipsoid");
        };
        assert_eq!(radii, [10.0, 2.0, 2.0], "containment never resizes");

        // The same ellipsoid unturned reaches 10 mm on x, so it is clamped
        // further out: the turn is what changed the room it needs.
        let ShapeSpec::Ellipsoid { center, radii, .. } = turned else {
            panic!("an ellipsoid");
        };
        let square = ShapeSpec::Ellipsoid {
            center,
            radii,
            rotation_deg: None,
        };
        let (kept, clamped) = clamped_into(&square, &bounds);
        assert!(clamped);
        let inner = kept.to_shape("test").expect("a shape").bounds();
        assert!((inner.max[0] - 20.0).abs() < 1e-9, "{inner:?}");
    }

    /// The domain is what everything else is kept inside of, so it is exempt;
    /// and with containment off nothing is clamped at all.
    #[test]
    fn containment_applies_to_everything_but_the_domain_and_only_when_it_is_on() {
        let (_dir, path) = write_temp("containment", fixture());
        let mut state = EditorState::open(&path).expect("open");
        let outside = ShapeSpec::Box {
            min: [500.0, 500.0, 500.0],
            max: [510.0, 510.0, 510.0],
            rotation_deg: None,
        };

        // A keepin dragged far outside is brought back to the domain's corner.
        let clamped = set_shape_contained(
            state.config_mut(),
            Selection::Keepin(0),
            outside.clone(),
            true,
        );
        assert!(clamped, "the commit was not clamped");
        let kept = shape_of(state.config(), Selection::Keepin(0)).expect("a shape");
        let inner = kept.to_shape("test").expect("a shape").bounds();
        let domain = containment_bounds(state.config()).expect("domain bounds");
        for d in 0..3 {
            assert!(
                inner.min[d] >= domain.min[d] - 1e-9 && inner.max[d] <= domain.max[d] + 1e-9,
                "axis {d}: {inner:?} is not inside {domain:?}"
            );
        }

        // With containment off it goes exactly where it was put: a keepin that
        // sticks out of the domain is a legitimate thing to model.
        let clamped = set_shape_contained(
            state.config_mut(),
            Selection::Keepin(0),
            outside.clone(),
            false,
        );
        assert!(!clamped);
        assert_eq!(
            shape_of(state.config(), Selection::Keepin(0)),
            Some(outside.clone())
        );

        // The domain itself is never clamped - into itself, it would be pinned
        // where it stands - and neither is a load case, which has no shape.
        let clamped = set_shape_contained(
            state.config_mut(),
            Selection::Domain(0),
            outside.clone(),
            true,
        );
        assert!(!clamped);
        assert_eq!(
            shape_of(state.config(), Selection::Domain(0)),
            Some(outside.clone())
        );
        assert!(!is_contained_kind(Selection::Domain(0)));
        assert!(!is_contained_kind(Selection::LoadCase(0)));
        for selection in [
            Selection::Keepout(0),
            Selection::Keepin(0),
            Selection::Support(0),
            Selection::Load { case: 0, load: 0 },
        ] {
            assert!(is_contained_kind(selection));
        }
    }

    /// Nothing is clamped against a domain that does not build: "inside the
    /// domain" has no meaning there, and the user's numbers are left alone.
    #[test]
    fn a_domain_that_does_not_build_clamps_nothing() {
        let (_dir, path) = write_temp("no_domain", fixture());
        let mut state = EditorState::open(&path).expect("open");
        assert!(containment_bounds(state.config()).is_some());
        state.edit(|config| config.domain.clear());
        assert!(containment_bounds(state.config()).is_none());

        let outside = ShapeSpec::Box {
            min: [500.0, 500.0, 500.0],
            max: [510.0, 510.0, 510.0],
            rotation_deg: None,
        };
        let clamped = set_shape_contained(
            state.config_mut(),
            Selection::Keepin(0),
            outside.clone(),
            true,
        );
        assert!(!clamped);
        assert_eq!(
            shape_of(state.config(), Selection::Keepin(0)),
            Some(outside)
        );
    }

    #[test]
    fn a_selection_that_no_longer_exists_is_no_selection() {
        let (_dir, path) = write_temp("stale", fixture());
        let mut state = EditorState::open(&path).expect("open");
        let last = state.config().keepout.len() - 1;
        state.select(Some(Selection::Keepout(last)));
        assert!(state.selection().is_some());
        // Whatever removes the object - a delete, an undo, a file that came
        // back shorter - the panels must never index with the old selection.
        state.edit(|config| {
            config.keepout.clear();
        });
        assert_eq!(state.selection(), None);
        assert!(!exists(
            state.config(),
            Selection::Load { case: 0, load: 9 }
        ));
        assert!(exists(state.config(), Selection::Load { case: 0, load: 0 }));
    }

    #[test]
    fn undo_and_redo_restore_the_configuration_and_the_selection() {
        let (_dir, path) = write_temp("undo", fixture());
        let mut state = EditorState::open(&path).expect("open");
        let keepouts = state.config().keepout.len();

        state.select(Some(Selection::Domain(0)));
        state.add(NewObject::Keepout(ShapeKind::Sphere));
        assert_eq!(state.config().keepout.len(), keepouts + 1);
        assert_eq!(state.selection(), Some(Selection::Keepout(keepouts)));

        assert!(state.undo());
        assert_eq!(state.config().keepout.len(), keepouts);
        assert_eq!(
            state.selection(),
            Some(Selection::Domain(0)),
            "undo restores the selection the edit was made with"
        );
        assert!(state.redo());
        assert_eq!(state.config().keepout.len(), keepouts + 1);
        assert_eq!(state.selection(), Some(Selection::Keepout(keepouts)));

        // A new edit after an undo discards the redo branch.
        assert!(state.undo());
        assert_eq!(state.redo_depth(), 1);
        state.add(NewObject::Keepin(ShapeKind::Box));
        assert_eq!(state.redo_depth(), 0);
        assert!(!state.redo());
    }

    #[test]
    fn the_undo_stack_is_bounded_and_keeps_the_newest_steps() {
        let (_dir, path) = write_temp("bounded", fixture());
        let mut state = EditorState::open(&path).expect("open");
        let depth = constants::VIEW_EDIT_UNDO_DEPTH;
        for _ in 0..depth + 10 {
            state.add(NewObject::Keepout(ShapeKind::Sphere));
        }
        assert_eq!(state.undo_depth(), depth);
        // Undoing everything the stack still holds walks back exactly `depth`
        // additions and no further.
        let keepouts = state.config().keepout.len();
        let mut undone = 0;
        while state.undo() {
            undone += 1;
        }
        assert_eq!(undone, depth);
        assert_eq!(state.config().keepout.len(), keepouts - depth);
    }

    #[test]
    fn one_interaction_is_one_undo_step() {
        let (_dir, path) = write_temp("interaction", fixture());
        let mut state = EditorState::open(&path).expect("open");
        let widget = 7;
        for value in [0.31, 0.32, 0.33] {
            state.begin_edit(widget);
            state.config_mut().optimization.mass_fraction = Some(value);
        }
        state.end_edit(widget);
        assert_eq!(state.undo_depth(), 1, "a drag is one step, not three");
        assert!(state.undo());
        assert!(
            (state
                .config()
                .optimization
                .mass_fraction
                .expect("a mass target")
                - 0.3)
                .abs()
                < 1e-12
        );

        // A second widget taking over closes the first interaction.
        state.begin_edit(1);
        state.config_mut().optimization.mass_fraction = Some(0.4);
        state.begin_edit(2);
        state.config_mut().optimization.min_feature_mm = 9.0;
        state.end_edit(2);
        assert_eq!(state.undo_depth(), 2);
    }

    #[test]
    fn an_invalid_edit_keeps_the_last_problem_and_says_what_is_wrong() {
        let (_dir, path) = write_temp("invalid", fixture());
        let mut state = EditorState::open(&path).expect("open");
        let cells = state.problem().expect("a problem").grid.n_cells();

        state.edit(|config| config.optimization.mass_fraction = Some(1.5));
        state.revalidate();
        assert!(!state.is_valid() || state.error().is_some());
        let error = state.error().expect("a reason");
        assert!(error.contains("mass_fraction"), "unexpected error: {error}");
        assert_eq!(
            state
                .problem()
                .expect("the last good problem")
                .grid
                .n_cells(),
            cells,
            "the viewport must keep showing the last model that built"
        );

        // And putting it back clears the error.
        assert!(state.undo());
        state.revalidate();
        assert!(state.is_valid() && state.error().is_none());
    }
}
