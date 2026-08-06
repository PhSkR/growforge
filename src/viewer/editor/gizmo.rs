//! The drag handles on the selected object: where they sit, what they look
//! like, and the arithmetic that turns a pointer ray into a new shape.
//!
//! Every handle is one of two kinds. An *axis* handle slides along a line - a
//! translation arrow, a box face, a radius - and its value is the parameter of
//! the point on that line closest to the pointer ray. A *plane* handle moves in
//! the plane through the grab point facing the camera - the free translation
//! handle, a box corner, a cylinder endpoint - and its value is where the ray
//! crosses that plane.
//!
//! A plane handle that *resizes* is then held to one dimension of that plane.
//! A box corner, the end handles of a cylinder, a tube or a cone, and a
//! triangle's vertices latch on to whichever dimension the gesture set off in -
//! once, as soon as it has covered
//! [`constants::VIEW_EDIT_RESIZE_LATCH_MM`] - and keep it until the button comes
//! up, so a hand that wanders while it pulls does not turn a resize into a
//! reshape; before that the drag changes nothing at all. Each of those handles
//! also has a gesture that *places* rather than sizes - an end taken across its
//! axis, a corner taken out of its triangle's plane - and latching on to that
//! one is what keeps a shape aimable. The free translation handle and a tube's
//! bend are placements through and through, and stay free in every direction.
//!
//! A drag is therefore a pure function of the ray, the shape the drag started
//! on, and the one classification it has already made, so it is reproducible and
//! testable without a window, and releasing simply stops asking.

use crate::config::ShapeSpec;
use crate::constants;
use crate::geometry::{
    Shape, Vec3, difference, euler_axis, inner, is_unrotated, length, normalize_degrees, rotate,
    rotate_inverse, rotation_matrix, scale, sum,
};
use crate::mesh::Mesh;
use crate::viewer::editor::pick::Ray;
use crate::viewer::editor::snap::{Flush, Snap, Surfaces};
use crate::viewer::scene::{LayerMesh, Shading};
use crate::viewer::tessellate;

/// What a handle does when it is dragged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleKind {
    /// Move the whole shape along one axis of its own frame, which for
    /// everything but a rotated box is a world axis.
    Translate(usize),
    /// Move the whole shape in the plane facing the camera.
    TranslateFree,
    /// Move one face of a box along its own axis.
    Face(usize, bool),
    /// Move one corner of a box, in the plane facing the camera, growing the
    /// one axis of the box's own frame the drag set off along - read from the
    /// start of the gesture and held for the rest of it, as every resize in a
    /// plane is ([`constants::VIEW_EDIT_RESIZE_LATCH_MM`]).
    Corner([bool; 3]),
    /// Move one end centre of a cylinder, a tube or a cone, in the plane facing
    /// the camera.
    ///
    /// A drag that sets off along the line through the two ends is the shape's
    /// **length** and slides the end along that line; one that sets off across
    /// it places the end anywhere in the plane, which is how these shapes are
    /// pointed somewhere else. Which of the two it is is read from the start of
    /// the gesture and held for the rest of it
    /// ([`constants::VIEW_EDIT_RESIZE_LATCH_MM`]).
    Endpoint(usize),
    /// Move a tube's bend point, in the plane facing the camera: the handle in
    /// the middle of the tube that curves it.
    Bend,
    /// Move one vertex of a triangle, in the plane facing the camera. The index
    /// is which of the three.
    ///
    /// A drag that sets off in the triangle's own plane sizes it: along the edge
    /// the corner faces, which shears it at a constant height, or along the
    /// corner's own height off that edge. One that sets off *out* of that plane
    /// places the corner anywhere, which is the only pose control a prism has.
    /// Which of the three it is is read from the start of the gesture and held
    /// for the rest of it ([`constants::VIEW_EDIT_RESIZE_LATCH_MM`]).
    ///
    /// Deliberately not an [`HandleKind::Endpoint`] with a third value: an
    /// endpoint is one of a *pair* - the far one is what its drag is measured
    /// against, what it is held apart from and what stays put - and that pair
    /// is written into every function a cylinder and a tube share. A triangle's
    /// three vertices are held off a *line* instead, so they get their own
    /// handle and their own arithmetic rather than a wider version of that.
    Vertex(usize),
    /// Grow or shrink a radius. The index is which one: a sphere and a cylinder
    /// have a single radius and use 0, an ellipsoid has one per axis of its own
    /// frame, and a cone has the one at each of its two ends.
    Radius(usize),
    /// Grow or shrink a triangular prism's thickness, along the normal of its
    /// own face.
    ///
    /// The extrusion is symmetric about that face, so the handle moves half of
    /// it: a drag of one millimetre is two millimetres of thickness, and the
    /// face the handle sits on stays under the pointer.
    Thickness,
    /// Turn the shape about one axis through its centre. The index is the
    /// component of the `rotation_deg` triple the arc drives; see
    /// [`crate::geometry::euler_axis`].
    Rotate(usize),
}

impl HandleKind {
    /// Tier the pointer prefers overlapping handles in, lowest first.
    ///
    /// Ordered by how *precise* a target each kind is, because the gizmo's grab
    /// volumes genuinely overlap and depth cannot arbitrate between a small
    /// marker and a long tube drawn through it:
    ///
    /// * The **cubes** - every resize handle (a face, a corner, a cap, a radius)
    ///   and the centre handle - are small markers sitting exactly on what they
    ///   change. Each of them stands *inside* an arrow's shaft: the centre one is
    ///   where all three shafts begin, and on anything near cubic the resize ones
    ///   are on the shaft along their own axis, because a box's `+x` face handle
    ///   is at half its width while the arrow runs to three quarters of its half
    ///   diagonal. The ray therefore enters the arrow first and no cube can ever
    ///   win on depth.
    /// * The **arrows** are what those cubes are buried in. An arrow's volume is
    ///   the whole shaft, deliberately forgiving, and there is plenty of it that
    ///   no cube stands on.
    /// * The **rotation arcs** ring the gizmo from outside everything else, so
    ///   they are grabbed last of all; see [`HandleKind::rings_the_gizmo`].
    ///
    /// The tier decides only between handles that are in the *same place*, which
    /// is what [`grab`] defines and applies.
    ///
    /// A tube's bend handle is one of the cubes. On a straight tube it sits
    /// exactly on the centre handle, where a tier cannot separate them at all -
    /// so it is drawn and grabbed a little larger instead
    /// ([`constants::VIEW_EDIT_BEND_HANDLE_FACTOR`]), which makes it the volume
    /// the ray enters first and the cube the user can see.
    pub fn grab_rank(self) -> u8 {
        match self {
            HandleKind::TranslateFree
            | HandleKind::Face(_, _)
            | HandleKind::Corner(_)
            | HandleKind::Endpoint(_)
            | HandleKind::Bend
            | HandleKind::Vertex(_)
            | HandleKind::Radius(_)
            | HandleKind::Thickness => 0,
            HandleKind::Translate(_) => 1,
            HandleKind::Rotate(_) => 2,
        }
    }

    /// True when the handle's grab volume rings the gizmo from *outside* it.
    ///
    /// The rotation arcs do, so from many camera angles one of them is between
    /// the pointer and an arrow or a resize cube: their depth is no evidence at
    /// all about what a press was aimed at, and they are grabbed only where
    /// nothing else is.
    pub fn rings_the_gizmo(self) -> bool {
        matches!(self, HandleKind::Rotate(_))
    }
}

/// One grab point of the gizmo.
#[derive(Debug, Clone, Copy)]
pub struct Handle {
    /// What dragging it does.
    pub kind: HandleKind,
    /// Where it sits in world space: the point a drag is measured from.
    pub position: Vec3,
    /// Line an axis handle slides along; unused by the plane handles.
    pub axis: Vec3,
    /// The volume the pointer has to hit to grab it.
    ///
    /// A shape rather than a radius, because what the user aims at is what
    /// they see: a translation arrow is a *shaft* running from the gizmo's
    /// centre out to its tip, and a click halfway along it is a click on the
    /// arrow. Testing a small sphere at the tip instead made the whole length
    /// of every arrow miss, fall through to object picking, and select the
    /// domain shell behind it.
    pub volume: Shape,
    /// Colour it is drawn in.
    pub color: [f32; 4],
}

/// Length of the gizmo's arrows for a shape of this bounding radius inside a
/// scene of that one.
///
/// Object relative, so the handles keep their proportion to what they move,
/// with a floor against the scene so a very small object is still grabbable.
pub fn gizmo_length(shape_radius: f64, scene_radius: f64) -> f64 {
    let from_shape = shape_radius * constants::VIEW_EDIT_GIZMO_LENGTH_RADIUS_FRACTION;
    let floor = scene_radius * constants::VIEW_EDIT_GIZMO_MIN_SCENE_FRACTION;
    from_shape
        .max(floor)
        .max(constants::VIEW_EDIT_MIN_EXTENT_MM)
}

/// Bounding sphere radius of a shape, for [`gizmo_length`].
pub fn shape_radius(shape: &Shape) -> f64 {
    let bounds = shape.bounds();
    if bounds.is_empty() {
        return 0.0;
    }
    0.5 * length(bounds.extent())
}

/// Centre of a shape: where the translation gizmo is anchored, and the point a
/// rotation turns about.
pub fn anchor(spec: &ShapeSpec) -> Vec3 {
    match *spec {
        ShapeSpec::Box { min, max, .. } => Shape::box_center(min, max),
        // A tube is anchored between its two ends like a cylinder, bent or
        // not: that is the point its own arithmetic is written about, and the
        // point a turn of it has to keep still. So is a cone, taper and all -
        // the middle of its axis rather than its centre of mass, which no
        // number in the file names.
        ShapeSpec::Cylinder { p1, p2, .. }
        | ShapeSpec::Tube { p1, p2, .. }
        | ShapeSpec::Cone { p1, p2, .. } => midpoint(p1, p2),
        ShapeSpec::Sphere { center, .. } | ShapeSpec::Ellipsoid { center, .. } => center,
        // The centroid of the three vertices, which the symmetric extrusion
        // makes the centre of the prism as well.
        ShapeSpec::Triangle { a, b, c, .. } => scale(sum(sum(a, b), c), 1.0 / 3.0),
    }
}

/// Halfway between two points.
fn midpoint(a: Vec3, b: Vec3) -> Vec3 {
    scale(sum(a, b), 0.5)
}

/// The two end centres of a shape that runs between a pair of points, which is
/// a cylinder, a tube or a cone; `None` for the kinds built about a centre and
/// for the triangle, whose three vertices are [`vertices`] instead.
pub fn endpoints(spec: &ShapeSpec) -> Option<(Vec3, Vec3)> {
    match *spec {
        ShapeSpec::Cylinder { p1, p2, .. }
        | ShapeSpec::Tube { p1, p2, .. }
        | ShapeSpec::Cone { p1, p2, .. } => Some((p1, p2)),
        ShapeSpec::Box { .. }
        | ShapeSpec::Sphere { .. }
        | ShapeSpec::Ellipsoid { .. }
        | ShapeSpec::Triangle { .. } => None,
    }
}

/// The three vertices of a triangle; `None` for every other kind.
pub fn vertices(spec: &ShapeSpec) -> Option<[Vec3; 3]> {
    match *spec {
        ShapeSpec::Triangle { a, b, c, .. } => Some([a, b, c]),
        _ => None,
    }
}

/// Thickness of a triangular prism; `None` for every other kind.
pub fn thickness_of(spec: &ShapeSpec) -> Option<f64> {
    match *spec {
        ShapeSpec::Triangle { thickness, .. } => Some(thickness),
        _ => None,
    }
}

/// The point the radius of `component` is measured from: where that radius
/// handle sits, and where the dimension line of its drag starts.
///
/// The centre for every kind built about one, the middle of the axis for a
/// cylinder - and for a tube the middle of its **curve**, which on a bent one
/// is not the middle of the line between its ends. A cone has a radius at each
/// end and measures each from the cap it belongs to, which is what `component`
/// picks. Matched exhaustively rather than through a catch-all, so a kind added
/// later has to say which it is.
pub fn radius_origin(spec: &ShapeSpec, component: usize) -> Vec3 {
    match *spec {
        ShapeSpec::Box { .. }
        | ShapeSpec::Sphere { .. }
        | ShapeSpec::Ellipsoid { .. }
        | ShapeSpec::Cylinder { .. }
        | ShapeSpec::Triangle { .. } => anchor(spec),
        ShapeSpec::Cone { p1, p2, .. } => {
            if component == 0 {
                p1
            } else {
                p2
            }
        }
        ShapeSpec::Tube { p1, p2, bend, .. } => match crate::geometry::tube_arc(p1, p2, bend) {
            Some(arc) => arc.point(0.5 * arc.span),
            None => midpoint(p1, p2),
        },
    }
}

/// Where a tube's bend handle sits: on the bend point it has, or in the middle
/// of the segment between its ends when it has none - which is the point that
/// bends it. `None` for every other kind.
pub fn bend_of(spec: &ShapeSpec) -> Option<Vec3> {
    let ShapeSpec::Tube { p1, p2, bend, .. } = *spec else {
        return None;
    };
    Some(bend.unwrap_or_else(|| midpoint(p1, p2)))
}

/// A unit vector along world axis `d`.
fn unit(d: usize) -> Vec3 {
    let mut v = [0.0; 3];
    v[d] = 1.0;
    v
}

/// Rotation a shape carries, as the three degrees the gizmo drives. A box and
/// an ellipsoid store one; everything else turns from where it is.
///
/// Matched exhaustively rather than through a catch-all, so a shape kind added
/// later has to say which of the two it is instead of silently reading as
/// unrotated.
pub fn rotation_of(spec: &ShapeSpec) -> Vec3 {
    match *spec {
        ShapeSpec::Box { rotation_deg, .. } | ShapeSpec::Ellipsoid { rotation_deg, .. } => {
            rotation_deg.unwrap_or_default()
        }
        ShapeSpec::Cylinder { .. }
        | ShapeSpec::Sphere { .. }
        | ShapeSpec::Tube { .. }
        | ShapeSpec::Cone { .. }
        | ShapeSpec::Triangle { .. } => [0.0; 3],
    }
}

/// The three axes of a shape's own frame.
///
/// For an unrotated shape these are the world axes, and every drag below is
/// the drag it always was. For a rotated box they are its turned edges, and for
/// a rotated ellipsoid its turned semi-axes, which is what puts the face and
/// radius handles where the shape's own faces and radii are and lets the arrows
/// point the way it is pointing.
pub fn local_axes(spec: &ShapeSpec) -> [Vec3; 3] {
    let rotation = rotation_of(spec);
    if is_unrotated(rotation) {
        return [unit(0), unit(1), unit(2)];
    }
    let matrix = rotation_matrix(rotation);
    [
        rotate(&matrix, unit(0)),
        rotate(&matrix, unit(1)),
        rotate(&matrix, unit(2)),
    ]
}

/// A point of a box given in the box's own coordinates, in world space.
fn from_local(spec: &ShapeSpec, local: Vec3) -> Vec3 {
    let centre = anchor(spec);
    let rotation = rotation_of(spec);
    if is_unrotated(rotation) {
        return local;
    }
    sum(
        centre,
        rotate(&rotation_matrix(rotation), difference(local, centre)),
    )
}

/// True when a shape's rotation gizmo has anything to turn: a box and an
/// ellipsoid store their own rotation, a cylinder turns its axis, a tube its
/// whole curve and a cone its axis with both radii, and a sphere looks the same
/// either way.
///
/// A **triangle** is deliberately absent, for the sphere's reason turned inside
/// out rather than for a missing arm: three free vertices are already complete
/// control of where the shape is and which way it faces, so a rotation gizmo
/// would be a second way to set what they set, with a centre to turn about that
/// the file does not name.
pub fn is_rotatable(spec: &ShapeSpec) -> bool {
    matches!(
        spec,
        ShapeSpec::Box { .. }
            | ShapeSpec::Cylinder { .. }
            | ShapeSpec::Ellipsoid { .. }
            | ShapeSpec::Tube { .. }
            | ShapeSpec::Cone { .. }
    )
}

/// Every handle of a shape: three translation arrows, a free handle, the
/// rotation arcs, and the resize handles its own kind has.
pub fn handles(spec: &ShapeSpec, gizmo_length: f64) -> Vec<Handle> {
    let size = gizmo_length * constants::VIEW_EDIT_HANDLE_SIZE_FRACTION;
    let pick_radius = 0.5 * size * constants::VIEW_EDIT_HANDLE_PICK_FACTOR;
    let centre = anchor(spec);
    let axes = local_axes(spec);
    let mut out = vec![Handle {
        kind: HandleKind::TranslateFree,
        position: centre,
        axis: [0.0, 0.0, 1.0],
        volume: Shape::Sphere {
            center: centre,
            radius: pick_radius,
        },
        color: constants::VIEW_COLOR_GIZMO_PLANE,
    }];
    for (d, axis) in axes.iter().enumerate() {
        let tip = sum(centre, scale(*axis, gizmo_length));
        out.push(Handle {
            kind: HandleKind::Translate(d),
            position: tip,
            axis: *axis,
            // The whole arrow, from the gizmo's centre to its tip.
            volume: Shape::Cylinder {
                p1: centre,
                p2: tip,
                radius: pick_radius,
            },
            color: constants::VIEW_COLOR_GIZMO_AXES[d],
        });
    }
    if is_rotatable(spec) {
        let ring = gizmo_length * constants::VIEW_EDIT_ROTATE_RING_RADIUS_FRACTION;
        let rotation = rotation_of(spec);
        for c in 0..3 {
            let axis = euler_axis(rotation, c);
            let (start, middle, end) = arc_points(centre, axis, ring);
            out.push(Handle {
                kind: HandleKind::Rotate(c),
                position: middle,
                axis,
                // The chord of the drawn arc: the sweep is shallow enough that
                // a tube around it covers the whole arrow, so the pointer grabs
                // what it is aiming at rather than one knob on it.
                volume: Shape::Cylinder {
                    p1: start,
                    p2: end,
                    radius: ring * constants::VIEW_EDIT_ROTATE_HANDLE_PICK_FRACTION,
                },
                color: constants::VIEW_COLOR_GIZMO_ARCS[c],
            });
        }
    }
    let handle = |kind: HandleKind, position: Vec3, axis: Vec3| Handle {
        kind,
        position,
        axis,
        volume: Shape::Sphere {
            center: position,
            radius: pick_radius,
        },
        color: constants::VIEW_COLOR_GIZMO_HANDLE,
    };
    match *spec {
        ShapeSpec::Box { min, max, .. } => {
            let corner = |mask: [bool; 3]| -> Vec3 {
                let mut p = [0.0; 3];
                for d in 0..3 {
                    p[d] = if mask[d] { max[d] } else { min[d] };
                }
                from_local(spec, p)
            };
            for k in 0..8 {
                let mask = [k & 1 != 0, k & 2 != 0, k & 4 != 0];
                out.push(handle(
                    HandleKind::Corner(mask),
                    corner(mask),
                    [0.0, 0.0, 1.0],
                ));
            }
            for d in 0..3 {
                for positive in [false, true] {
                    let mut position = Shape::box_center(min, max);
                    position[d] = if positive { max[d] } else { min[d] };
                    out.push(handle(
                        HandleKind::Face(d, positive),
                        from_local(spec, position),
                        axes[d],
                    ));
                }
            }
        }
        ShapeSpec::Cylinder { p1, p2, radius } => {
            out.push(handle(HandleKind::Endpoint(0), p1, [0.0, 0.0, 1.0]));
            out.push(handle(HandleKind::Endpoint(1), p2, [0.0, 0.0, 1.0]));
            let (radial, _) = tessellate::basis(difference(p2, p1));
            out.push(handle(
                HandleKind::Radius(0),
                sum(centre, scale(radial, radius)),
                radial,
            ));
        }
        ShapeSpec::Sphere { center, radius } => {
            let radial = unit(0);
            out.push(handle(
                HandleKind::Radius(0),
                sum(center, scale(radial, radius)),
                radial,
            ));
        }
        // One radius handle per semi-axis, each on the end of its own axis of
        // the ellipsoid's frame: the sphere's handle, three times over, turned
        // with the shape.
        ShapeSpec::Ellipsoid { center, radii, .. } => {
            for (d, axis) in axes.iter().enumerate() {
                out.push(handle(
                    HandleKind::Radius(d),
                    sum(center, scale(*axis, radii[d])),
                    *axis,
                ));
            }
        }
        ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        } => {
            out.push(handle(HandleKind::Endpoint(0), p1, [0.0, 0.0, 1.0]));
            out.push(handle(HandleKind::Endpoint(1), p2, [0.0, 0.0, 1.0]));
            // The radius handle sits on the surface at the middle of the
            // curve, sliding along the way the tube is thickest there: out of
            // the plane it bends in, so it is never buried in the arc it
            // measures, and the cylinder's own choice when there is no plane.
            let radial = match crate::geometry::tube_arc(p1, p2, bend) {
                Some(arc) => arc.normal,
                None => tessellate::basis(difference(p2, p1)).0,
            };
            out.push(handle(
                HandleKind::Radius(0),
                sum(radius_origin(spec, 0), scale(radial, radius)),
                radial,
            ));
            // The one that makes the curve. Larger than the other cubes,
            // because on a straight tube it lands on the centre handle; see
            // [`HandleKind::grab_rank`].
            let position = bend.unwrap_or_else(|| midpoint(p1, p2));
            out.push(Handle {
                kind: HandleKind::Bend,
                position,
                axis: [0.0, 0.0, 1.0],
                volume: Shape::Sphere {
                    center: position,
                    radius: pick_radius * constants::VIEW_EDIT_BEND_HANDLE_FACTOR,
                },
                color: constants::VIEW_COLOR_GIZMO_BEND,
            });
        }
        ShapeSpec::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => {
            let (radial, _) = tessellate::basis(difference(p2, p1));
            // The radius handles come **before** the end handles, and that
            // order is load bearing at exactly one shape: a true cone, where
            // `radius2` is zero and the second radius handle stands on the
            // apex, which is the second endpoint. Two markers in one place are
            // one press with one winner, and [`grab`] gives it to the first of
            // the equal hits - so it goes to the handle that has to be
            // draggable to bring the cone back off its point, and which
            // separates the two the moment it moves. It is the rule a straight
            // tube's bend handle gets, by ordering rather than by size.
            for (component, origin, radius) in [(0, p1, radius1), (1, p2, radius2)] {
                out.push(handle(
                    HandleKind::Radius(component),
                    sum(origin, scale(radial, radius)),
                    radial,
                ));
            }
            out.push(handle(HandleKind::Endpoint(0), p1, [0.0, 0.0, 1.0]));
            out.push(handle(HandleKind::Endpoint(1), p2, [0.0, 0.0, 1.0]));
        }
        ShapeSpec::Triangle { a, b, c, thickness } => {
            for (component, vertex) in [a, b, c].into_iter().enumerate() {
                out.push(handle(
                    HandleKind::Vertex(component),
                    vertex,
                    [0.0, 0.0, 1.0],
                ));
            }
            // The thickness handle sits on the middle of the face it moves,
            // sliding along the normal: the one direction of a prism its three
            // vertices do not already set.
            if let Some(normal) = crate::geometry::triangle_normal(a, b, c) {
                out.push(handle(
                    HandleKind::Thickness,
                    sum(centre, scale(normal, 0.5 * thickness)),
                    normal,
                ));
            }
        }
    }
    out
}

/// Start, middle and end of a rotation arc of radius `ring` about `axis`
/// through `centre`.
fn arc_points(centre: Vec3, axis: Vec3, ring: f64) -> (Vec3, Vec3, Vec3) {
    let (u, v) = tessellate::basis(axis);
    let sweep = constants::VIEW_EDIT_ROTATE_ARC_SWEEP_DEGREES.to_radians();
    let at = |angle: f64| -> Vec3 {
        let (sin, cos) = angle.sin_cos();
        sum(centre, scale(sum(scale(u, cos), scale(v, sin)), ring))
    };
    (at(0.0), at(0.5 * sweep), at(sweep))
}

/// Angle of the point where `ray` crosses the plane through `centre` with
/// normal `axis`, measured in that plane's own basis, in degrees.
///
/// `None` when the ray runs along the plane, where there is no crossing and a
/// turn would run away.
pub fn arc_angle(ray: &Ray, centre: Vec3, axis: Vec3) -> Option<f64> {
    let point = plane_point(ray, centre, axis)?;
    let (u, v) = tessellate::basis(axis);
    let offset = difference(point, centre);
    let (x, y) = (inner(offset, u), inner(offset, v));
    if x == 0.0 && y == 0.0 {
        return None;
    }
    Some(y.atan2(x).to_degrees())
}

/// The handle the ray grabs, or `None` when it hits none of them.
///
/// Handles are tested before objects, so a handle drawn over the shape it moves
/// wins the click; a press that hits none of them falls through to picking.
///
/// Two steps, because the gizmo's volumes overlap and depth on its own is not
/// evidence of what the user aimed at:
///
/// 1. **What the press landed on** is the nearest volume the ray entered, with
///    the arcs kept out of that contest - they ring the gizmo from outside, so
///    one of them is in front of everything from many angles
///    ([`HandleKind::rings_the_gizmo`]).
/// 2. **What it was aimed at** is a handle of a lower [`HandleKind::grab_rank`]
///    *buried inside* what it landed on - its grab point within that volume -
///    which is one of the cubes standing in the shaft of an arrow: a resize
///    handle on its own axis, or the centre handle where all three shafts begin.
///    It is drawn there, it is the smaller and more precise target, and it can
///    never be the nearer hit, so the press is its.
///
/// Burial is a property of the two handles rather than of the ray, which is what
/// keeps the tier from reaching across the gizmo: a press aimed squarely at an
/// arrow's tip, with a corner handle of the object tens of millimetres further
/// along the same ray, keeps the arrow.
pub fn grab(ray: &Ray, handles: &[Handle]) -> Option<Handle> {
    let hits: Vec<(Handle, f64)> = handles
        .iter()
        .filter_map(|handle| {
            crate::viewer::editor::pick::hit_shape(ray, &handle.volume).map(|t| (*handle, t))
        })
        .collect();
    let landed = *hits.iter().min_by(|(a, at), (b, bt)| {
        a.kind
            .rings_the_gizmo()
            .cmp(&b.kind.rings_the_gizmo())
            .then(at.total_cmp(bt))
    })?;
    let buried = hits
        .iter()
        .filter(|(handle, _)| {
            handle.kind.grab_rank() < landed.0.kind.grab_rank()
                && landed.0.volume.contains(handle.position)
        })
        .min_by(|(a, at), (b, bt)| {
            a.kind
                .grab_rank()
                .cmp(&b.kind.grab_rank())
                .then(at.total_cmp(bt))
        });
    Some(buried.unwrap_or(&landed).0)
}

/// Parameter along the line `origin + t * axis` of the point closest to `ray`.
///
/// `None` when the line and the ray are parallel, where there is no such point
/// and a drag would run away.
pub fn axis_parameter(ray: &Ray, origin: Vec3, axis: Vec3) -> Option<f64> {
    let axis_length = length(axis);
    if axis_length <= 0.0 {
        return None;
    }
    let unit = scale(axis, 1.0 / axis_length);
    let alignment = inner(unit, ray.direction);
    let denominator = 1.0 - alignment * alignment;
    if denominator <= constants::VIEW_EDIT_MIN_DRAG_ANGLE_SINE_SQUARED {
        return None;
    }
    let offset = difference(ray.origin, origin);
    Some((inner(offset, unit) - inner(offset, ray.direction) * alignment) / denominator)
}

/// The direction `axis` runs in *on screen*: its projection into the plane with
/// normal `normal`, as a unit vector paired with how much of the axis is left in
/// it - one for an axis square to the view, falling away to nothing as it turns
/// towards the camera.
///
/// `None` when it has turned so far that what is left is arithmetic noise rather
/// than a direction, which is the near-parallel test every axis drag already
/// applies read as the sine squared it is
/// ([`constants::VIEW_EDIT_MIN_DRAG_ANGLE_SINE_SQUARED`]): an axis pointed
/// straight at the camera has no direction on screen for a gesture to be
/// compared with.
fn in_plane(axis: Vec3, normal: Vec3) -> Option<(Vec3, f64)> {
    let (axis_length, normal_length) = (length(axis), length(normal));
    if axis_length <= 0.0 || normal_length <= 0.0 {
        return None;
    }
    let unit_axis = scale(axis, 1.0 / axis_length);
    let unit_normal = scale(normal, 1.0 / normal_length);
    let projected = difference(unit_axis, scale(unit_normal, inner(unit_axis, unit_normal)));
    let seen = length(projected);
    if seen * seen <= constants::VIEW_EDIT_MIN_DRAG_ANGLE_SINE_SQUARED {
        return None;
    }
    Some((scale(projected, 1.0 / seen), seen))
}

/// Whether a displacement of `moved` across the drag plane is a pull along
/// `axis` or a move across it.
///
/// Both are measured as they are *seen*, in the plane the pointer is really
/// moving in: the part of `moved` along the axis' own projection, against the
/// part left over. The axis takes the tie, because the handles this classifies
/// are resize handles and the dimension is what they are for. An axis with no
/// projection to compare against - the camera looking straight down it - can
/// only be the free drag.
fn latch_along(axis: Vec3, moved: Vec3, normal: Vec3) -> Latch {
    let Some((seen, _)) = in_plane(axis, normal) else {
        return Latch::Lateral;
    };
    let along = inner(moved, seen);
    let across = inner(moved, moved) - along * along;
    if along * along >= across {
        Latch::Axial
    } else {
        Latch::Lateral
    }
}

/// Which of `axes` a displacement of `moved` across the drag plane runs most
/// nearly along, measured as each of them is seen on screen.
///
/// An axis turned towards the camera is no candidate at all: its projection is
/// noise, and a normalized noise direction could win a comparison it has no
/// direction to enter. Of a frame of three that is at most one, so there is
/// always one left to pick; the lowest axis takes a tie, and the impossible case
/// of none at all leaves the drag undecided rather than picking arbitrarily.
fn latch_axis(axes: &[Vec3; 3], moved: Vec3, normal: Vec3) -> Latch {
    let mut best: Option<(usize, f64)> = None;
    for (d, axis) in axes.iter().enumerate() {
        let Some((seen, _)) = in_plane(*axis, normal) else {
            continue;
        };
        let along = inner(moved, seen).abs();
        if best.is_none_or(|(_, most)| along > most) {
            best = Some((d, along));
        }
    }
    match best {
        Some((d, _)) => Latch::Axis(d),
        None => Latch::Undecided,
    }
}

/// The three directions a triangle's `corner` can be dragged in, given the two
/// vertices it faces: along the edge between them, along the corner's own
/// altitude off that edge, and out of the triangle's plane.
///
/// The first two are what the shape is *sized* by - a base and a height - and
/// the third is how it is aimed, because three free vertices are the only pose
/// control a prism has.
///
/// `None` for a triangle with no area to speak of, by the very measurement
/// [`crate::config::ShapeSpec::to_shape`] refuses one with: it has no altitude
/// to slide along and no plane to be taken out of, so there is nothing to hold a
/// drag to and the drag stays free.
fn vertex_axes(corner: Vec3, first: Vec3, second: Vec3) -> Option<(Vec3, Vec3, Vec3)> {
    let flatness = crate::geometry::triangle_shortest_altitude(first, second, corner);
    if flatness <= constants::MIN_SHAPE_EXTENT_MM || flatness.is_nan() {
        return None;
    }
    let edge = difference(second, first);
    let edge = scale(edge, 1.0 / length(edge));
    let offset = difference(corner, first);
    let out = difference(offset, scale(edge, inner(offset, edge)));
    let height = length(out);
    let normal = crate::geometry::triangle_normal(first, second, corner)?;
    Some((edge, scale(out, 1.0 / height), normal))
}

/// Which of a vertex's three directions a displacement of `moved` set off along,
/// measured as each of them is seen on screen.
///
/// The two in the triangle's own plane are dimensions and the one out of it is a
/// pose, so the out-of-plane drag has to *beat* both of them to win: a tie goes
/// to the dimension, which is the rule an end handle's tie already follows. The
/// altitude takes a tie with the edge, being the number a corner drag is usually
/// after. A direction the camera cannot see is no candidate at all, and a corner
/// with none left to compare is placed freely.
fn latch_vertex(axes: (Vec3, Vec3, Vec3), moved: Vec3, normal: Vec3) -> Latch {
    let (edge, altitude, face) = axes;
    let seen = |axis: Vec3| in_plane(axis, normal).map(|(seen, _)| inner(moved, seen).abs());
    let sized = match (seen(edge), seen(altitude)) {
        (Some(along), Some(off)) if along > off => Some((Latch::Edge, along)),
        (Some(along), None) => Some((Latch::Edge, along)),
        (_, Some(off)) => Some((Latch::Altitude, off)),
        (None, None) => None,
    };
    let Some((latched, best)) = sized else {
        return Latch::Lateral;
    };
    match seen(face) {
        Some(out) if out > best => Latch::Lateral,
        _ => latched,
    }
}

/// Where `ray` crosses the plane through `origin` with normal `normal`.
pub fn plane_point(ray: &Ray, origin: Vec3, normal: Vec3) -> Option<Vec3> {
    let denominator = inner(normal, ray.direction);
    if denominator.abs() <= constants::VIEW_EDIT_MIN_DRAG_ANGLE_SINE_SQUARED {
        return None;
    }
    let t = inner(difference(origin, ray.origin), normal) / denominator;
    (t >= 0.0).then(|| ray.at(t))
}

/// What one position of a drag produced: the shape, and the surface it landed
/// flush against if it landed on one.
#[derive(Debug, Clone, PartialEq)]
pub struct Placed {
    /// The shape as the drag would leave it.
    pub shape: ShapeSpec,
    /// The surface a face of it came to rest on, for the callout to name.
    pub flush: Option<Flush>,
    /// Where the handle being dragged now is, when the placed shape no longer
    /// says.
    ///
    /// One handle needs this, and it is the tube's bend. A bend dragged back
    /// onto the line between the two ends is *stored* as no bend at all - the
    /// shape really is straight, and its file really has no key - so the shape
    /// on its own puts the handle back in the middle of the tube while the
    /// pointer is still holding it somewhere else along it. The drag knows
    /// where the pointer is, so it says.
    pub handle_at: Option<Vec3>,
}

impl Placed {
    /// A shape that landed on the increment rather than on a surface, and whose
    /// handle is where the shape itself says it is.
    fn plain(shape: ShapeSpec) -> Placed {
        Placed {
            shape,
            flush: None,
            handle_at: None,
        }
    }
}

/// Move the handle of `kind` to `position`, taking its grab volume with it.
///
/// What is drawn and what the pointer hits are both derived from the shape as
/// it is committed, which is right for every handle but one: see
/// [`Placed::handle_at`]. A drag that reports a live position hands it to this,
/// so the cube stays under the pointer and stays grabbable there.
pub fn reposition(handles: &mut [Handle], kind: HandleKind, position: Vec3) {
    for handle in handles.iter_mut().filter(|handle| handle.kind == kind) {
        handle.volume = moved(&handle.volume, difference(position, handle.position));
        handle.position = position;
    }
}

/// A copy of `shape` moved by `delta`. Matched exhaustively, so a primitive
/// added later has to say how it moves rather than silently staying put.
fn moved(shape: &Shape, delta: Vec3) -> Shape {
    match *shape {
        Shape::Box {
            min,
            max,
            rotation_deg,
        } => Shape::Box {
            min: sum(min, delta),
            max: sum(max, delta),
            rotation_deg,
        },
        Shape::Cylinder { p1, p2, radius } => Shape::Cylinder {
            p1: sum(p1, delta),
            p2: sum(p2, delta),
            radius,
        },
        Shape::Sphere { center, radius } => Shape::Sphere {
            center: sum(center, delta),
            radius,
        },
        Shape::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } => Shape::Ellipsoid {
            center: sum(center, delta),
            radii,
            rotation_deg,
        },
        Shape::Tube {
            p1,
            p2,
            bend,
            radius,
        } => Shape::Tube {
            p1: sum(p1, delta),
            p2: sum(p2, delta),
            bend: bend.map(|bend| sum(bend, delta)),
            radius,
        },
        Shape::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => Shape::Cone {
            p1: sum(p1, delta),
            p2: sum(p2, delta),
            radius1,
            radius2,
        },
        Shape::Triangle { a, b, c, thickness } => Shape::Triangle {
            a: sum(a, delta),
            b: sum(b, delta),
            c: sum(c, delta),
            thickness,
        },
    }
}

/// Axis aligned bounds of a shape specification, or `None` when it is not one
/// yet: half-typed numbers have no bounds to place against.
fn bounds_of(spec: &ShapeSpec) -> Option<crate::geometry::Aabb> {
    spec.to_shape("drag").ok().map(|shape| shape.bounds())
}

/// Where a drag started, in the terms its handle is dragged in.
#[derive(Debug, Clone, Copy)]
enum Reference {
    /// Parameter along the handle's axis at the moment of the grab.
    Axis(f64),
    /// Point on the camera facing plane at the moment of the grab.
    Plane(Vec3),
    /// Angle in the arc's own plane at the moment of the grab, in degrees.
    Angle(f64),
}

/// What a resize drag in a plane has decided it is changing.
///
/// A resize handle sets **one** dimension, and which one it is is read off the
/// start of the gesture rather than off every frame of it: the drag is
/// classified once, as soon as it has covered
/// [`constants::VIEW_EDIT_RESIZE_LATCH_MM`] of the plane it is dragged in, and
/// holds that answer until the button comes up. So a hand that wanders while it
/// pulls a cap out still only pulls it out - the wander is not a second
/// dimension to change as well - and a press that shakes by a few microns
/// changes nothing at all, because inside that dead zone there is no direction
/// to read and the drag says so by leaving the shape alone.
///
/// Only the handles that resize *in a plane* are latched: an axis handle is one
/// dimension by construction, and the free translation handle and a tube's bend
/// are placements, where freedom in every direction is the point of them. Two of
/// the latched handles can be placements as well - an end taken across its axis,
/// a vertex taken out of its triangle's plane - and that is one of the answers
/// they latch on to rather than an exception to latching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Latch {
    /// Still inside the dead zone: the drag has nothing to say yet.
    Undecided,
    /// An end handle setting how far apart the two ends are: the end slides
    /// along the line through the pair and the shape keeps its direction.
    Axial,
    /// A handle *placing* its point across the plane facing the camera rather
    /// than sizing anything with it: an end handle aiming a cylinder, a tube or
    /// a cone somewhere else, and a vertex handle taken out of its triangle's
    /// own plane, which is how a prism is aimed.
    Lateral,
    /// A corner handle growing one axis of the shape's own frame.
    Axis(usize),
    /// A vertex handle sliding along the edge it faces: the triangle is sheared,
    /// its base stays where it is and the corner keeps its height off it.
    Edge,
    /// A vertex handle sliding along its own altitude: the corner's height off
    /// the edge it faces, at a base that does not move.
    Altitude,
}

/// A drag in progress.
///
/// It holds the shape as it was when the handle was grabbed, so every pointer
/// position is applied to the original rather than accumulated: a drag that
/// comes back to where it started leaves the shape exactly as it was.
#[derive(Debug, Clone)]
pub struct Drag {
    handle: Handle,
    original: ShapeSpec,
    reference: Reference,
    /// Normal of the plane a free handle moves in: the camera's view direction
    /// when the handle was grabbed.
    normal: Vec3,
    /// Set once the drag has actually changed the shape, which is what decides
    /// whether it is worth an undo step.
    changed: bool,
    /// What a resize drag in a plane has latched on to, and [`Latch::Undecided`]
    /// until it has travelled far enough to say. The handles that are not
    /// latched never ask.
    latch: Latch,
}

impl Drag {
    /// Begin a drag, or `None` when this pointer position gives the handle
    /// nothing to hold on to.
    pub fn start(
        handle: Handle,
        shape: ShapeSpec,
        ray: &Ray,
        view_direction: Vec3,
    ) -> Option<Drag> {
        let reference = match handle.kind {
            HandleKind::Translate(_)
            | HandleKind::Face(_, _)
            | HandleKind::Radius(_)
            | HandleKind::Thickness => {
                Reference::Axis(axis_parameter(ray, handle.position, handle.axis)?)
            }
            HandleKind::TranslateFree
            | HandleKind::Corner(_)
            | HandleKind::Endpoint(_)
            | HandleKind::Bend
            | HandleKind::Vertex(_) => {
                Reference::Plane(plane_point(ray, handle.position, view_direction)?)
            }
            HandleKind::Rotate(_) => Reference::Angle(arc_angle(ray, anchor(&shape), handle.axis)?),
        };
        Some(Drag {
            handle,
            original: shape,
            reference,
            normal: view_direction,
            changed: false,
            latch: Latch::Undecided,
        })
    }

    /// The handle this drag is holding.
    pub fn handle(&self) -> Handle {
        self.handle
    }

    /// The shape the drag started on, which every position is applied to.
    pub fn original(&self) -> &ShapeSpec {
        &self.original
    }

    /// What the drag is latched on to, classifying it with `decide` the first
    /// time the pointer is far enough from the grab for an answer to mean
    /// anything; see [`Latch`].
    ///
    /// `moved` is the whole displacement since the grab rather than the last
    /// frame's, because what is being read is the direction the *gesture* set
    /// off in, and it is read once: after that the answer stands until the drag
    /// is dropped and a new one starts.
    fn latched(&mut self, moved: Vec3, decide: impl FnOnce() -> Latch) -> Latch {
        if self.latch == Latch::Undecided && length(moved) > constants::VIEW_EDIT_RESIZE_LATCH_MM {
            self.latch = decide();
        }
        self.latch
    }

    /// What the shape should be with the pointer here, or `None` when this
    /// position says nothing (a ray parallel to the handle's axis).
    ///
    /// `snap` is applied to the value the handle is changing and to nothing
    /// else: the position along the drag axis, the dimension being resized, the
    /// angle being turned. It is applied to the *result* rather than to the
    /// movement, so the object lands on the increment however far off it
    /// started - which is what makes two objects dragged to the same grid line
    /// actually meet.
    pub fn shape_at(&mut self, ray: &Ray, snap: Snap) -> Option<ShapeSpec> {
        self.placed_at(ray, snap, &Surfaces::default())
            .map(|placed| placed.shape)
    }

    /// What the shape should be with the pointer here, and whether it landed
    /// flush against one of `surfaces`.
    ///
    /// A translation offered surfaces lands on one of them when a face of it
    /// comes within [`constants::VIEW_EDIT_SURFACE_SNAP_MM`] of a candidate
    /// plane, and on the increment otherwise. The surface wins where both
    /// apply: within that distance the face is what the user is aiming at, and
    /// the grid would park the region an increment short of it.
    ///
    /// Takes the drag by `&mut` for the one thing a drag remembers besides the
    /// shape it started on: the dimension a resize has latched on to, which is
    /// decided here the first time the pointer is far enough away to say and
    /// held from then on ([`constants::VIEW_EDIT_RESIZE_LATCH_MM`]).
    pub fn placed_at(&mut self, ray: &Ray, snap: Snap, surfaces: &Surfaces) -> Option<Placed> {
        match (self.handle.kind, self.reference) {
            (HandleKind::Translate(_), Reference::Axis(start)) => {
                let now = axis_parameter(ray, self.handle.position, self.handle.axis)?;
                let axis = self.handle.axis;
                // The position being changed is where the centre sits along the
                // drag axis; for a world axis that is simply its coordinate.
                let along = inner(anchor(&self.original), axis);
                let raw = now - start;
                if let Some((flush, moved)) = self.flush_along(axis, raw, surfaces, snap) {
                    return Some(Placed {
                        shape: translate(&self.original, scale(axis, moved)),
                        flush: Some(flush),
                        handle_at: None,
                    });
                }
                let moved = snap.length(along + raw) - along;
                Some(Placed::plain(translate(&self.original, scale(axis, moved))))
            }
            (HandleKind::TranslateFree, Reference::Plane(start)) => {
                let now = plane_point(ray, self.handle.position, self.normal)?;
                // Every component of the centre is being changed, so every one
                // of them snaps - and every one of them can land on a surface.
                let centre = anchor(&self.original);
                let moved = difference(now, start);
                let raw = bounds_of(&translate(&self.original, moved));
                let mut delta = [0.0; 3];
                let mut landed: Option<Flush> = None;
                for (d, slot) in delta.iter_mut().enumerate() {
                    let flush = raw
                        .filter(|_| snap.snaps_lengths())
                        .and_then(|bounds| surfaces.flush(d, &bounds, self.surface_distance(snap)));
                    match flush {
                        Some(flush) => {
                            *slot = moved[d] + flush.correction;
                            if landed
                                .is_none_or(|best| flush.correction.abs() < best.correction.abs())
                            {
                                landed = Some(flush);
                            }
                        }
                        None => *slot = snap.length(centre[d] + moved[d]) - centre[d],
                    }
                }
                Some(Placed {
                    shape: translate(&self.original, delta),
                    flush: landed,
                    handle_at: None,
                })
            }
            (HandleKind::Face(axis, positive), Reference::Axis(start)) => {
                let now = axis_parameter(ray, self.handle.position, self.handle.axis)?;
                let sign = if positive { 1.0 } else { -1.0 };
                let extent = box_extent(&self.original).map(|e| e[axis]).unwrap_or(0.0);
                let wanted = snap.length(extent + sign * (now - start));
                Some(Placed::plain(move_face(
                    &self.original,
                    axis,
                    positive,
                    sign * (wanted - extent),
                )))
            }
            (HandleKind::Corner(mask), Reference::Plane(start)) => {
                let now = plane_point(ray, self.handle.position, self.normal)?;
                // A corner moves in the camera's plane, and the box grows along
                // its own edges - so the axis of the box's own frame the gesture
                // set off along is the one it grows, and only that one. It used
                // to grow all three at once, which is what made a corner drag
                // impossible to aim: every pixel off the axis the user meant was
                // a second and a third dimension changing with it.
                let moved = difference(now, start);
                let (axes, normal) = (local_axes(&self.original), self.normal);
                match self.latched(moved, || latch_axis(&axes, moved, normal)) {
                    Latch::Axis(axis) => {
                        let local = to_local_direction(&self.original, moved);
                        let extent = box_extent(&self.original).unwrap_or([0.0; 3]);
                        let sign = if mask[axis] { 1.0 } else { -1.0 };
                        let wanted = snap.length(extent[axis] + sign * local[axis]);
                        let mut delta = [0.0; 3];
                        // One axis moves, so it is snapped on its own exactly as
                        // a face handle's is; the other two are left untouched
                        // rather than re-derived from a movement they no longer
                        // take any part in.
                        delta[axis] = sign * (wanted - extent[axis]);
                        Some(Placed::plain(move_corner(&self.original, mask, delta)))
                    }
                    // Not yet far enough from the grab to say which edge is
                    // being pulled - and an end handle's or a vertex's answer,
                    // which no corner can be given, names no edge of a box
                    // either.
                    Latch::Undecided
                    | Latch::Axial
                    | Latch::Lateral
                    | Latch::Edge
                    | Latch::Altitude => Some(Placed::plain(self.original.clone())),
                }
            }
            (HandleKind::Endpoint(end), Reference::Plane(start)) => {
                let now = plane_point(ray, self.handle.position, self.normal)?;
                let (p1, p2) = endpoints(&self.original)?;
                let (from, other) = if end == 0 { (p1, p2) } else { (p2, p1) };
                let moved = difference(now, start);
                // The line the drag is measured against: from the far end to the
                // one being held, which for a cylinder and a cone is the axis
                // itself and for a tube is the chord its curve spans.
                let (chord, normal) = (difference(from, other), self.normal);
                match self.latched(moved, || latch_along(chord, moved, normal)) {
                    // The length, and nothing but the length: the end stays on
                    // the line through the pair, so a pull that wanders lengthens
                    // the shape without tilting it.
                    Latch::Axial => {
                        // A chord with a direction on screen, which is what the
                        // latch was read from and so what it still has - and a
                        // chord of some length, which is what makes the span
                        // below safe to divide by.
                        let (seen, foreshortening) = in_plane(chord, normal)?;
                        let span = length(chord);
                        // Divided by how much of the axis the camera can see, so
                        // the end keeps up with the pointer along the line it is
                        // drawn on however far the axis leans away from the view.
                        let along = inner(moved, seen) / foreshortening;
                        // The *span* lands on the increment - the number the
                        // callout shows - rather than each coordinate of the end
                        // separately, which would take it off the line and put
                        // back the tilt this arm exists to keep out. Floored
                        // where every other resize is floored, so an end dragged
                        // through the far one stops rather than turning the shape
                        // around; [`held_apart`] would push it out backwards.
                        let wanted = snap
                            .length(span + along)
                            .max(constants::VIEW_EDIT_MIN_EXTENT_MM);
                        let to = sum(other, scale(chord, wanted / span));
                        Some(Placed::plain(place_endpoint(&self.original, end, to)))
                    }
                    // Where the end is *put* rather than how far out it is, which
                    // is the only way to point one of these shapes somewhere
                    // else: free across the plane, and snapped per axis. A
                    // corner's or a vertex's answer, which no end handle can be
                    // given, reads as that same free drag rather than as cases
                    // of its own.
                    Latch::Lateral | Latch::Axis(_) | Latch::Edge | Latch::Altitude => {
                        let mut delta = [0.0; 3];
                        for d in 0..3 {
                            delta[d] = snap.length(from[d] + moved[d]) - from[d];
                        }
                        Some(Placed::plain(move_endpoint(&self.original, end, delta)))
                    }
                    // Not yet far enough from the grab to tell a resize from a
                    // reaim, so the shape is left exactly as it was.
                    Latch::Undecided => Some(Placed::plain(self.original.clone())),
                }
            }
            // The drag that makes the curve, and a placement rather than a
            // resize: free in all three axes like an end handle that has latched
            // on to placing its end, and snapped the same way - what lands on
            // the increment is where the bend point ends up.
            (HandleKind::Bend, Reference::Plane(start)) => {
                let now = plane_point(ray, self.handle.position, self.normal)?;
                let from = bend_of(&self.original)?;
                let moved = difference(now, start);
                let mut delta = [0.0; 3];
                for d in 0..3 {
                    delta[d] = snap.length(from[d] + moved[d]) - from[d];
                }
                // The point the pointer is holding, before the shape decides
                // whether it is a bend at all: a bend that lands back on the
                // line between the ends is stored as none, and the handle would
                // otherwise leave the pointer for the middle of the tube. See
                // [`Placed::handle_at`].
                Some(Placed {
                    shape: move_bend(&self.original, delta),
                    flush: None,
                    handle_at: Some(sum(from, delta)),
                })
            }
            // A triangle is sized *and* aimed by its three corners, so this one
            // handle is both a resize and a placement - and which of the two it
            // is, this drag says at the start and holds to. In the triangle's
            // own plane it is a resize and latches: along the edge the corner
            // faces, or along its own height off that edge. Out of that plane it
            // is the pose gesture, and stays as free as it has always been.
            (HandleKind::Vertex(vertex), Reference::Plane(start)) => {
                let now = plane_point(ray, self.handle.position, self.normal)?;
                let points = vertices(&self.original)?;
                let from = *points.get(vertex)?;
                let (first, second) = opposite_edge(&points, vertex)?;
                let moved = difference(now, start);
                let normal = self.normal;
                let axes = vertex_axes(from, first, second);
                let latched = self.latched(moved, || match axes {
                    Some(axes) => latch_vertex(axes, moved, normal),
                    None => Latch::Lateral,
                });
                // Slid along one line of the triangle's own plane. The increment
                // applies to how far it has slid rather than to a dimension it
                // lands on, because a corner has none of its own: the file holds
                // three points, and the number beside the drag is the median,
                // which neither of these two lines sets on its own.
                let slide = |axis: Vec3| -> Option<Placed> {
                    let (seen, foreshortening) = in_plane(axis, normal)?;
                    let along = snap.length(inner(moved, seen) / foreshortening);
                    Some(Placed::plain(place_vertex(
                        &self.original,
                        vertex,
                        sum(from, scale(axis, along)),
                    )))
                };
                match latched {
                    Latch::Edge => slide(axes?.0),
                    Latch::Altitude => slide(axes?.1),
                    // The pose drag: free across the plane facing the camera and
                    // snapped per axis, which is what every vertex drag was. An
                    // end handle's and a corner's answers cannot reach a vertex,
                    // and read as that same free drag.
                    Latch::Lateral | Latch::Axial | Latch::Axis(_) => {
                        let mut delta = [0.0; 3];
                        for d in 0..3 {
                            delta[d] = snap.length(from[d] + moved[d]) - from[d];
                        }
                        Some(Placed::plain(move_vertex(&self.original, vertex, delta)))
                    }
                    // Not yet far enough from the grab to tell a resize from a
                    // reaim, so the shape is left exactly as it was.
                    Latch::Undecided => Some(Placed::plain(self.original.clone())),
                }
            }
            // The handle sits half a thickness off the face's own plane, so it
            // moves half of what the thickness does - and the number that
            // lands on the increment is the thickness, which is what the
            // callout shows and what the file holds.
            (HandleKind::Thickness, Reference::Axis(start)) => {
                let now = axis_parameter(ray, self.handle.position, self.handle.axis)?;
                let thickness = thickness_of(&self.original)?;
                let wanted = snap.length(thickness + 2.0 * (now - start));
                Some(Placed::plain(resize_thickness(
                    &self.original,
                    wanted - thickness,
                )))
            }
            (HandleKind::Radius(component), Reference::Axis(start)) => {
                let now = axis_parameter(ray, self.handle.position, self.handle.axis)?;
                let radius = radius_of(&self.original, component).unwrap_or(0.0);
                let wanted = snap.length(radius + (now - start));
                Some(Placed::plain(resize_radius(
                    &self.original,
                    component,
                    wanted - radius,
                )))
            }
            (HandleKind::Rotate(component), Reference::Angle(start)) => {
                let now = arc_angle(ray, anchor(&self.original), self.handle.axis)?;
                // Folded into a half turn either way, so a pointer that crosses
                // the arc's own seam does not spin the object round.
                let swept = normalize_degrees(now - start);
                Some(Placed::plain(turn(
                    &self.original,
                    component,
                    self.handle.axis,
                    swept,
                    snap,
                )))
            }
            _ => None,
        }
    }

    /// How near a surface counts as landing on it, which is nowhere at all
    /// while snapping is switched off or bypassed.
    fn surface_distance(&self, snap: Snap) -> f64 {
        if snap.snaps_lengths() {
            constants::VIEW_EDIT_SURFACE_SNAP_MM
        } else {
            0.0
        }
    }

    /// The flush landing of a drag of `raw` millimetres along `axis`, and how
    /// far along that axis the shape has to move to make it.
    ///
    /// The candidate planes are axis aligned, so the drag is matched against
    /// the world axis it mostly runs along - which for everything but a turned
    /// box is the axis it runs along exactly. A drag too oblique for that to
    /// mean anything lands on nothing and takes the increment instead.
    fn flush_along(
        &self,
        axis: Vec3,
        raw: f64,
        surfaces: &Surfaces,
        snap: Snap,
    ) -> Option<(Flush, f64)> {
        let distance = self.surface_distance(snap);
        if distance <= 0.0 || surfaces.is_empty() {
            return None;
        }
        let world = (0..3).max_by(|a, b| axis[*a].abs().total_cmp(&axis[*b].abs()))?;
        let alignment = axis[world];
        if alignment.abs() < constants::VIEW_EDIT_SURFACE_SNAP_MIN_ALIGNMENT {
            return None;
        }
        let bounds = bounds_of(&translate(&self.original, scale(axis, raw)))?;
        let flush = surfaces.flush(world, &bounds, distance)?;
        Some((flush, raw + flush.correction / alignment))
    }

    /// Record that the drag has changed the shape.
    pub fn mark_changed(&mut self) {
        self.changed = true;
    }

    /// True once the drag has changed the shape, and is therefore worth one
    /// undo step when it is released.
    pub fn has_changed(&self) -> bool {
        self.changed
    }
}

/// Edge lengths of a box, in its own frame; `None` for the other kinds.
pub fn box_extent(spec: &ShapeSpec) -> Option<Vec3> {
    match *spec {
        ShapeSpec::Box { min, max, .. } => {
            Some([max[0] - min[0], max[1] - min[1], max[2] - min[2]])
        }
        _ => None,
    }
}

/// Radius of a shape that has one, by component.
///
/// A sphere and a cylinder have a single radius, which is their answer for
/// every component; an ellipsoid has one per axis of its own frame and a cone
/// one at each of its two ends, and the component picks it. A component outside
/// the range that kind has, has no radius.
pub fn radius_of(spec: &ShapeSpec, component: usize) -> Option<f64> {
    match *spec {
        ShapeSpec::Cylinder { radius, .. }
        | ShapeSpec::Sphere { radius, .. }
        | ShapeSpec::Tube { radius, .. } => Some(radius),
        ShapeSpec::Ellipsoid { radii, .. } => radii.get(component).copied(),
        ShapeSpec::Cone {
            radius1, radius2, ..
        } => match component {
            0 => Some(radius1),
            1 => Some(radius2),
            _ => None,
        },
        ShapeSpec::Box { .. } | ShapeSpec::Triangle { .. } => None,
    }
}

/// A world direction in the shape's own frame, which for anything but a
/// rotated box is the direction itself.
pub fn to_local_direction(spec: &ShapeSpec, direction: Vec3) -> Vec3 {
    let rotation = rotation_of(spec);
    if is_unrotated(rotation) {
        return direction;
    }
    rotate_inverse(&rotation_matrix(rotation), direction)
}

/// Turn a shape by `swept` degrees about `axis` through its centre.
///
/// A box and an ellipsoid record the turn in the `rotation_deg` component the
/// arc drives, which is exactly what turning about that component's own axis
/// does - see [`crate::geometry::euler_axis`] - so the snapping is of the
/// *resulting* angle and the file holds the number the callout showed. A
/// cylinder has nowhere to record one, so its caps are moved instead and the
/// snapping is of the sweep; a tube is the same, with its bend point carried
/// round with the two ends so that the curve is turned rather than reshaped.
pub fn turn(spec: &ShapeSpec, component: usize, axis: Vec3, swept: f64, snap: Snap) -> ShapeSpec {
    match *spec {
        ShapeSpec::Box {
            min,
            max,
            rotation_deg,
        } => {
            let mut rotation = rotation_deg.unwrap_or_default();
            rotation[component] = snap.angle(rotation[component] + swept);
            ShapeSpec::Box {
                min,
                max,
                rotation_deg: Some(rotation),
            }
        }
        ShapeSpec::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } => {
            let mut rotation = rotation_deg.unwrap_or_default();
            rotation[component] = snap.angle(rotation[component] + swept);
            ShapeSpec::Ellipsoid {
                center,
                radii,
                rotation_deg: Some(rotation),
            }
        }
        ShapeSpec::Cylinder { p1, p2, radius } => {
            let centre = anchor(spec);
            let matrix = axis_rotation_matrix(axis, snap.angle(swept));
            ShapeSpec::Cylinder {
                p1: sum(centre, rotate(&matrix, difference(p1, centre))),
                p2: sum(centre, rotate(&matrix, difference(p2, centre))),
                radius,
            }
        }
        ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        } => {
            let centre = anchor(spec);
            let matrix = axis_rotation_matrix(axis, snap.angle(swept));
            let turned = |p: Vec3| sum(centre, rotate(&matrix, difference(p, centre)));
            ShapeSpec::Tube {
                p1: turned(p1),
                p2: turned(p2),
                bend: bend.map(turned),
                radius,
            }
        }
        // The cylinder's turn, with a radius at each end that comes along
        // unchanged: turning a cone points it somewhere else rather than
        // reshaping its taper.
        ShapeSpec::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => {
            let centre = anchor(spec);
            let matrix = axis_rotation_matrix(axis, snap.angle(swept));
            ShapeSpec::Cone {
                p1: sum(centre, rotate(&matrix, difference(p1, centre))),
                p2: sum(centre, rotate(&matrix, difference(p2, centre))),
                radius1,
                radius2,
            }
        }
        // A sphere looks the same whichever way it is turned, and a triangle is
        // pointed by its own three vertices: neither carries a rotation gizmo,
        // and neither is changed by one - see [`is_rotatable`].
        ShapeSpec::Sphere { .. } | ShapeSpec::Triangle { .. } => spec.clone(),
    }
}

/// Rotation of `degrees` about an arbitrary unit `axis`, by Rodrigues' formula.
fn axis_rotation_matrix(axis: Vec3, degrees: f64) -> crate::geometry::Mat3 {
    let len = length(axis);
    if len <= 0.0 {
        return rotation_matrix([0.0; 3]);
    }
    let a = scale(axis, 1.0 / len);
    let (sin, cos) = degrees.to_radians().sin_cos();
    let t = 1.0 - cos;
    [
        [
            cos + a[0] * a[0] * t,
            a[0] * a[1] * t - a[2] * sin,
            a[0] * a[2] * t + a[1] * sin,
        ],
        [
            a[1] * a[0] * t + a[2] * sin,
            cos + a[1] * a[1] * t,
            a[1] * a[2] * t - a[0] * sin,
        ],
        [
            a[2] * a[0] * t - a[1] * sin,
            a[2] * a[1] * t + a[0] * sin,
            cos + a[2] * a[2] * t,
        ],
    ]
}

/// Move a whole shape by `delta`.
pub fn translate(spec: &ShapeSpec, delta: Vec3) -> ShapeSpec {
    match *spec {
        ShapeSpec::Box {
            min,
            max,
            rotation_deg,
        } => ShapeSpec::Box {
            min: sum(min, delta),
            max: sum(max, delta),
            rotation_deg,
        },
        ShapeSpec::Cylinder { p1, p2, radius } => ShapeSpec::Cylinder {
            p1: sum(p1, delta),
            p2: sum(p2, delta),
            radius,
        },
        ShapeSpec::Sphere { center, radius } => ShapeSpec::Sphere {
            center: sum(center, delta),
            radius,
        },
        ShapeSpec::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } => ShapeSpec::Ellipsoid {
            center: sum(center, delta),
            radii,
            rotation_deg,
        },
        // The bend point moves with the ends, so the tube keeps its curve.
        ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        } => ShapeSpec::Tube {
            p1: sum(p1, delta),
            p2: sum(p2, delta),
            bend: bend.map(|bend| sum(bend, delta)),
            radius,
        },
        ShapeSpec::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => ShapeSpec::Cone {
            p1: sum(p1, delta),
            p2: sum(p2, delta),
            radius1,
            radius2,
        },
        // All three vertices together, so the prism moves rather than reshapes.
        ShapeSpec::Triangle { a, b, c, thickness } => ShapeSpec::Triangle {
            a: sum(a, delta),
            b: sum(b, delta),
            c: sum(c, delta),
            thickness,
        },
    }
}

/// Slide one face of a box, keeping the opposite one where it is and never
/// letting the box collapse through it.
pub fn move_face(spec: &ShapeSpec, axis: usize, positive: bool, delta: f64) -> ShapeSpec {
    let ShapeSpec::Box {
        mut min,
        mut max,
        rotation_deg,
    } = *spec
    else {
        return spec.clone();
    };
    grow_box(
        &mut min,
        &mut max,
        rotation_deg.unwrap_or_default(),
        axis,
        positive,
        delta,
    );
    ShapeSpec::Box {
        min,
        max,
        rotation_deg,
    }
}

/// Move one corner of a box in all three axes at once, keeping the opposite
/// corner fixed.
pub fn move_corner(spec: &ShapeSpec, mask: [bool; 3], delta: Vec3) -> ShapeSpec {
    let ShapeSpec::Box {
        mut min,
        mut max,
        rotation_deg,
    } = *spec
    else {
        return spec.clone();
    };
    let rotation = rotation_deg.unwrap_or_default();
    for d in 0..3 {
        grow_box(&mut min, &mut max, rotation, d, mask[d], delta[d]);
    }
    ShapeSpec::Box {
        min,
        max,
        rotation_deg,
    }
}

/// Move one face of a box along axis `axis` of its **own** frame by `delta`,
/// keeping the opposite face where it is and never letting the box collapse
/// through it.
///
/// For an axis aligned box that is one corner coordinate moving, which is the
/// arithmetic it has always been. For a rotated one it is not: `min` and `max`
/// describe the box before it is turned, and the turn is about their midpoint,
/// so moving one of them alone would swing the whole box about a centre that
/// slid sideways. The centre is therefore moved along the box's own axis by
/// half the growth, which is exactly what leaves the opposite face where the
/// user can see it is.
fn grow_box(
    min: &mut Vec3,
    max: &mut Vec3,
    rotation: Vec3,
    axis: usize,
    positive: bool,
    delta: f64,
) {
    if is_unrotated(rotation) {
        if positive {
            max[axis] = (max[axis] + delta).max(min[axis] + constants::VIEW_EDIT_MIN_EXTENT_MM);
        } else {
            min[axis] = (min[axis] + delta).min(max[axis] - constants::VIEW_EDIT_MIN_EXTENT_MM);
        }
        return;
    }
    let extent = max[axis] - min[axis];
    let grown = (extent + delta).max(constants::VIEW_EDIT_MIN_EXTENT_MM);
    let half_change = 0.5 * (grown - extent);
    let along = rotate(&rotation_matrix(rotation), unit(axis));
    let step = scale(along, if positive { half_change } else { -half_change });
    *min = sum(*min, step);
    *max = sum(*max, step);
    min[axis] -= half_change;
    max[axis] += half_change;
}

/// Move one end centre of a cylinder, a tube or a cone, keeping the other one,
/// the radii, and - for a tube - the bend point, so that dragging an end
/// reshapes the curve rather than dragging it along.
pub fn move_endpoint(spec: &ShapeSpec, end: usize, delta: Vec3) -> ShapeSpec {
    let Some((p1, p2)) = endpoints(spec) else {
        return spec.clone();
    };
    place_endpoint(spec, end, sum(if end == 0 { p1 } else { p2 }, delta))
}

/// Put one end centre of a cylinder, a tube or a cone **at** `point`, keeping
/// everything [`move_endpoint`] keeps and holding it the same distance off the
/// other end.
///
/// What `move_endpoint` is written in terms of, and what a drag that has latched
/// on to the length uses directly: the end such a drag computes is a point on a
/// line, and taking it there as an offset from where the end was and back would
/// cost it the last bits of that line for nothing.
pub fn place_endpoint(spec: &ShapeSpec, end: usize, point: Vec3) -> ShapeSpec {
    let Some((p1, p2)) = endpoints(spec) else {
        return spec.clone();
    };
    let other = if end == 0 { p2 } else { p1 };
    let moved = held_apart(point, other);
    let (p1, p2) = if end == 0 {
        (moved, other)
    } else {
        (other, moved)
    };
    match *spec {
        ShapeSpec::Cylinder { radius, .. } => ShapeSpec::Cylinder { p1, p2, radius },
        ShapeSpec::Tube { bend, radius, .. } => ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        },
        ShapeSpec::Cone {
            radius1, radius2, ..
        } => ShapeSpec::Cone {
            p1,
            p2,
            radius1,
            radius2,
        },
        _ => spec.clone(),
    }
}

/// Move one vertex of a triangle by `delta`, keeping the other two and the
/// thickness, so that dragging a corner reshapes the prism rather than dragging
/// it along.
///
/// The vertex is held at least [`constants::VIEW_EDIT_MIN_EXTENT_MM`] off the
/// **line through the other two**, which is the rule an endpoint drag applies
/// to its own pair, on the one degeneracy a triangle has: a vertex on that line
/// is a triangle of no area,
/// and a triangle of no area encloses nothing at any thickness - it is refused
/// by [`crate::config::ShapeSpec::to_shape`] rather than treated as something
/// else. A drag that lands there is pushed back out along the perpendicular it
/// came in on, so the shape stays one the configuration can hold and the
/// pointer stays in charge of every direction but the one that would collapse
/// it.
pub fn move_vertex(spec: &ShapeSpec, vertex: usize, delta: Vec3) -> ShapeSpec {
    let Some(points) = vertices(spec) else {
        return spec.clone();
    };
    let Some(from) = points.get(vertex) else {
        return spec.clone();
    };
    place_vertex(spec, vertex, sum(*from, delta))
}

/// Put one vertex of a triangle **at** `point`, keeping everything
/// [`move_vertex`] keeps and holding it off the line through the other two the
/// same way.
///
/// What `move_vertex` is written in terms of, and what a drag that has latched
/// on to the edge or to the altitude uses directly: the corner such a drag
/// computes is a point on a line, and taking it there as an offset and back
/// would cost it the last bits of that line for nothing.
pub fn place_vertex(spec: &ShapeSpec, vertex: usize, point: Vec3) -> ShapeSpec {
    let Some(mut points) = vertices(spec) else {
        return spec.clone();
    };
    let ShapeSpec::Triangle { thickness, .. } = *spec else {
        return spec.clone();
    };
    let Some((first, second)) = opposite_edge(&points, vertex) else {
        return spec.clone();
    };
    points[vertex] = held_off_line(point, first, second);
    ShapeSpec::Triangle {
        a: points[0],
        b: points[1],
        c: points[2],
        thickness,
    }
}

/// The two vertices a corner faces, in order, and `None` for an index that is
/// not one of the three: the pairing every part of a vertex drag is written
/// against, in one place.
fn opposite_edge(points: &[Vec3; 3], vertex: usize) -> Option<(Vec3, Vec3)> {
    match vertex {
        0 => Some((points[1], points[2])),
        1 => Some((points[2], points[0])),
        2 => Some((points[0], points[1])),
        _ => None,
    }
}

/// `moved`, pushed back off the line through `first` and `second` if it landed
/// within the smallest usable extent of it.
///
/// The triangle's [`held_apart`]: what a point may not do is make the shape
/// degenerate, and for a vertex that is landing on the line the other two
/// define. The push is along the perpendicular the point came in on, so the
/// drag keeps every direction that does not collapse the triangle; a point
/// exactly on the line has no such direction and takes any perpendicular to the
/// line. Two other vertices in one place define no line at all - a shape only
/// hand-typed numbers can produce - and are held apart as a pair instead.
fn held_off_line(moved: Vec3, first: Vec3, second: Vec3) -> Vec3 {
    let edge = difference(second, first);
    let span = length(edge);
    if span <= 0.0 {
        return held_apart(moved, first);
    }
    let unit = scale(edge, 1.0 / span);
    let offset = difference(moved, first);
    let foot = sum(first, scale(unit, inner(offset, unit)));
    let out = difference(moved, foot);
    let distance = length(out);
    if distance >= constants::VIEW_EDIT_MIN_EXTENT_MM {
        return moved;
    }
    let direction = if distance > 0.0 {
        scale(out, 1.0 / distance)
    } else {
        tessellate::basis(edge).0
    };
    sum(foot, scale(direction, constants::VIEW_EDIT_MIN_EXTENT_MM))
}

/// `moved`, pushed off `other` along whatever direction it came from if it
/// landed on top of it: a cylinder of zero length is degenerate, and a tube of
/// zero length has no direction left to bend in.
fn held_apart(moved: Vec3, other: Vec3) -> Vec3 {
    let axis = difference(moved, other);
    let len = length(axis);
    if len >= constants::VIEW_EDIT_MIN_EXTENT_MM {
        return moved;
    }
    let direction = if len > 0.0 {
        scale(axis, 1.0 / len)
    } else {
        [0.0, 0.0, 1.0]
    };
    sum(other, scale(direction, constants::VIEW_EDIT_MIN_EXTENT_MM))
}

/// Move a tube's bend point by `delta`, from wherever its handle is: the bend
/// it carries, or the middle of the segment between its ends when it carries
/// none, which is what turns a straight tube into a curved one.
///
/// A bend that lands back on the line through the two ends - within
/// [`crate::constants::TUBE_COLLINEAR_EPS_MM`] of it - **clears the key**
/// rather than being kept as a bend nothing can see. Dragging the middle back
/// straight really does straighten the tube, and the file it is saved to has no
/// `bend` in it. It is the one test [`crate::geometry::tube_arc`] applies, so
/// what the editor calls straight and what the geometry draws straight cannot
/// drift apart; the properties panel's **straighten** button is the way to do
/// it without aiming.
pub fn move_bend(spec: &ShapeSpec, delta: Vec3) -> ShapeSpec {
    let ShapeSpec::Tube {
        p1,
        p2,
        bend,
        radius,
    } = *spec
    else {
        return spec.clone();
    };
    let moved = sum(bend.unwrap_or_else(|| midpoint(p1, p2)), delta);
    ShapeSpec::Tube {
        p1,
        p2,
        bend: crate::geometry::tube_arc(p1, p2, Some(moved))
            .is_some()
            .then_some(moved),
        radius,
    }
}

/// Grow or shrink one radius, never below the smallest usable extent - with one
/// deliberate exception, the narrow end of a cone.
///
/// `component` picks which radius an ellipsoid or a cone changes and is ignored
/// by the kinds that have only one; see [`radius_of`].
///
/// **A cone's second radius floors at zero rather than at
/// [`constants::VIEW_EDIT_MIN_EXTENT_MM`]**, because zero is not a degenerate
/// value there: it is the apex of a true cone, the shape a user dragging that
/// handle inward is aiming at, and a floor a tenth of a millimetre above it
/// would leave a blunt tip nobody asked for and no drag could remove. Its first
/// radius keeps the usual floor - that end has to stay a real disc, and it is
/// what [`crate::geometry::Shape::min_extent`] measures a cone by.
pub fn resize_radius(spec: &ShapeSpec, component: usize, delta: f64) -> ShapeSpec {
    match *spec {
        ShapeSpec::Cylinder { p1, p2, radius } => ShapeSpec::Cylinder {
            p1,
            p2,
            radius: (radius + delta).max(constants::VIEW_EDIT_MIN_EXTENT_MM),
        },
        ShapeSpec::Sphere { center, radius } => ShapeSpec::Sphere {
            center,
            radius: (radius + delta).max(constants::VIEW_EDIT_MIN_EXTENT_MM),
        },
        ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        } => ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius: (radius + delta).max(constants::VIEW_EDIT_MIN_EXTENT_MM),
        },
        ShapeSpec::Ellipsoid {
            center,
            mut radii,
            rotation_deg,
        } => {
            let Some(slot) = radii.get_mut(component) else {
                return spec.clone();
            };
            *slot = (*slot + delta).max(constants::VIEW_EDIT_MIN_EXTENT_MM);
            ShapeSpec::Ellipsoid {
                center,
                radii,
                rotation_deg,
            }
        }
        ShapeSpec::Cone {
            p1,
            p2,
            mut radius1,
            mut radius2,
        } => {
            match component {
                0 => radius1 = (radius1 + delta).max(constants::VIEW_EDIT_MIN_EXTENT_MM),
                // The apex is a legal shape and has to be reachable by drag.
                1 => radius2 = (radius2 + delta).max(0.0),
                _ => return spec.clone(),
            }
            ShapeSpec::Cone {
                p1,
                p2,
                radius1,
                radius2,
            }
        }
        ShapeSpec::Box { .. } | ShapeSpec::Triangle { .. } => spec.clone(),
    }
}

/// Grow or shrink a triangular prism's thickness, never below the smallest
/// usable extent.
///
/// The extrusion is symmetric about the triangle's own plane, so the prism
/// grows equally either side and the face the handle sits on moves half of what
/// the thickness does; see [`HandleKind::Thickness`].
pub fn resize_thickness(spec: &ShapeSpec, delta: f64) -> ShapeSpec {
    let ShapeSpec::Triangle { a, b, c, thickness } = *spec else {
        return spec.clone();
    };
    ShapeSpec::Triangle {
        a,
        b,
        c,
        thickness: (thickness + delta).max(constants::VIEW_EDIT_MIN_EXTENT_MM),
    }
}

/// A copy of `shape` grown by `margin` on every side, which is what the
/// selection shell is drawn from so it never z-fights with the shape itself.
pub fn inflated(shape: &Shape, margin: f64) -> Shape {
    match *shape {
        // Grown along its own edges, so the shell of a turned box is that box
        // turned rather than an axis aligned crate around it.
        Shape::Box {
            min,
            max,
            rotation_deg,
        } => Shape::Box {
            min: [min[0] - margin, min[1] - margin, min[2] - margin],
            max: [max[0] + margin, max[1] + margin, max[2] + margin],
            rotation_deg,
        },
        Shape::Sphere { center, radius } => Shape::Sphere {
            center,
            radius: radius + margin,
        },
        // Every semi-axis grown by the margin, so the shell is the same
        // ellipsoid turned the same way rather than a sphere around it.
        Shape::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } => Shape::Ellipsoid {
            center,
            radii: [radii[0] + margin, radii[1] + margin, radii[2] + margin],
            rotation_deg,
        },
        Shape::Cylinder { p1, p2, radius } => {
            let axis = difference(p2, p1);
            let len = length(axis);
            let along = if len > 0.0 {
                scale(axis, margin / len)
            } else {
                [0.0; 3]
            };
            Shape::Cylinder {
                p1: difference(p1, along),
                p2: sum(p2, along),
                radius: radius + margin,
            }
        }
        // A tube is everything within its radius of its own curve, so growing
        // that radius grows it evenly - the rounded ends included, which is why
        // its points are left alone where a cylinder's flat caps are pushed
        // out.
        Shape::Tube {
            p1,
            p2,
            bend,
            radius,
        } => Shape::Tube {
            p1,
            p2,
            bend,
            radius: radius + margin,
        },
        // A cone's caps are pushed out along its axis like a cylinder's, and
        // its radii grow by more than the margin, because the wall is slanted:
        // a shell a margin clear of it stands `margin / cos(slant)` further out
        // from the axis, which is the slant length over the axis length.
        //
        // That is exact where the two agree - at the **apex**, and whenever the
        // two radii are equal, which is the cylinder this generalizes - and not
        // exact along a sloped wall, deliberately. Translating the caps by the
        // margin while growing both radii by a constant tilts the shell's wall:
        // its slope is `(r2 - r1) / (len + 2 * margin)` rather than
        // `(r2 - r1) / len`, so the clearance runs linearly from **less** than
        // a margin at the wide rim, through exactly a margin halfway along, to
        // more than one at the narrow rim. On a 4 mm to 1 mm taper over 8 mm it
        // is two thirds of a margin at the wide rim.
        //
        // What is guaranteed is that it is never zero and never negative: the
        // clearance is proportional to `slant * (len + 2 * margin) / len - (r1 -
        // r2)`, which stays above `slant - (r1 - r2)` and that is positive for
        // any cone with a length. Strictly positive clearance everywhere is the
        // shell's whole contract - it exists so the highlight does not z-fight
        // with the shape it is drawn around - and the test pins a bounded
        // fraction of the margin rather than the equality this comment used to
        // claim. Meeting the margin exactly at a sloped corner needs the
        // intersection of three offset surfaces, which is machinery a selection
        // overlay does not warrant.
        Shape::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => {
            let axis = difference(p2, p1);
            let len = length(axis);
            if len <= 0.0 {
                return Shape::Cone {
                    p1,
                    p2,
                    radius1: radius1 + margin,
                    radius2: radius2 + margin,
                };
            }
            let along = scale(axis, margin / len);
            let grow = margin * (len * len + (radius1 - radius2).powi(2)).sqrt() / len;
            Shape::Cone {
                p1: difference(p1, along),
                p2: sum(p2, along),
                radius1: radius1 + grow,
                radius2: radius2 + grow,
            }
        }
        // Every one of the prism's five faces pushed out by the margin, which
        // is the box's rule on a shape whose sides are not axis aligned: the
        // thickness grows by twice the margin, half either side, and the
        // triangle is scaled about its **incentre**, where scaling by
        // `(r + margin) / r` moves each of the three edge lines out by exactly
        // the margin.
        Shape::Triangle { a, b, c, thickness } => {
            let inradius = crate::geometry::triangle_inradius(a, b, c);
            let thickness = thickness + 2.0 * margin;
            if inradius <= 0.0 {
                return Shape::Triangle { a, b, c, thickness };
            }
            let (opposite_a, opposite_b, opposite_c) = (
                length(difference(c, b)),
                length(difference(a, c)),
                length(difference(b, a)),
            );
            let perimeter = opposite_a + opposite_b + opposite_c;
            let incentre = scale(
                sum(
                    sum(scale(a, opposite_a), scale(b, opposite_b)),
                    scale(c, opposite_c),
                ),
                1.0 / perimeter,
            );
            let factor = (inradius + margin) / inradius;
            let grown = |p: Vec3| -> Vec3 { sum(incentre, scale(difference(p, incentre), factor)) };
            Shape::Triangle {
                a: grown(a),
                b: grown(b),
                c: grown(c),
                thickness,
            }
        }
    }
}

/// Margin the selection shell is grown by.
pub fn selection_margin(shape: &Shape, gizmo_length: f64) -> f64 {
    (shape.min_extent() * constants::VIEW_EDIT_SELECTION_MARGIN_FRACTION)
        .max(gizmo_length * constants::VIEW_EDIT_SELECTION_MARGIN_GIZMO_FRACTION)
}

/// Margin the hover outline is grown by: the same rule as
/// [`selection_margin`] on its own, thinner, constants, so the two shells are
/// told apart by width as well as by colour.
pub fn hover_margin(shape: &Shape, gizmo_length: f64) -> f64 {
    (shape.min_extent() * constants::VIEW_EDIT_HOVER_MARGIN_FRACTION)
        .max(gizmo_length * constants::VIEW_EDIT_HOVER_MARGIN_GIZMO_FRACTION)
}

/// A colour moved towards white, which is how the handle under the pointer says
/// so without changing size.
pub fn brightened(color: [f32; 4]) -> [f32; 4] {
    let t = constants::VIEW_EDIT_HANDLE_HOVER_BRIGHTEN;
    let mut out = color;
    for channel in out.iter_mut().take(3) {
        *channel += t * (1.0 - *channel);
    }
    out
}

/// Triangles for the whole gizmo: an arrow per axis, a curved arrow per
/// rotation axis, and a cube per resize handle.
///
/// `hovered` is the handle the pointer is over, drawn brighter than the rest.
pub fn mesh(
    handles: &[Handle],
    gizmo_length: f64,
    centre: Vec3,
    hovered: Option<HandleKind>,
) -> LayerMesh {
    let size = gizmo_length * constants::VIEW_EDIT_HANDLE_SIZE_FRACTION;
    let ring = gizmo_length * constants::VIEW_EDIT_ROTATE_RING_RADIUS_FRACTION;
    let mut out = LayerMesh::default();
    for handle in handles {
        // An arrow is drawn tip first, so it runs from the gizmo's centre out
        // to the handle it is grabbed by.
        let part = match handle.kind {
            HandleKind::Translate(_) => tessellate::arrow(
                handle.position,
                difference(handle.position, centre),
                gizmo_length,
            ),
            HandleKind::Rotate(_) => tessellate::arc_arrow(
                centre,
                handle.axis,
                ring,
                constants::VIEW_EDIT_ROTATE_ARC_SWEEP_DEGREES,
                ring * constants::VIEW_EDIT_ROTATE_ARC_TUBE_FRACTION,
            ),
            // The bend marker is the larger cube it is grabbed as, so that on a
            // straight tube - where it stands exactly on the centre handle - it
            // is the one that can be seen as well as the one that answers.
            HandleKind::Bend => tessellate::marker_cube(
                handle.position,
                size * constants::VIEW_EDIT_BEND_HANDLE_FACTOR,
            ),
            _ => tessellate::marker_cube(handle.position, size),
        };
        let color = if hovered == Some(handle.kind) {
            brightened(handle.color)
        } else {
            handle.color
        };
        append(&mut out, &part, color);
    }
    out
}

/// Append a mesh to a layer in one colour.
fn append(out: &mut LayerMesh, mesh: &Mesh, color: [f32; 4]) {
    let part = LayerMesh::from_mesh(mesh, color, Shading::Rounded);
    if part.is_empty() {
        return;
    }
    out.vertices.extend(part.vertices.iter().copied());
    out.bounds = out.bounds.union(&part.bounds);
}

/// The distinct colours a drawn gizmo carries, for the tests below.
#[cfg(test)]
fn colors(layer: &LayerMesh) -> Vec<[f32; 4]> {
    let mut seen: Vec<[f32; 4]> = Vec::new();
    for vertex in &layer.vertices {
        if !seen.contains(&vertex.color) {
            seen.push(vertex.color);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::editor::pick::Ray;

    fn ray_from(origin: Vec3, towards: Vec3) -> Ray {
        let direction = difference(towards, origin);
        Ray {
            origin,
            direction: scale(direction, 1.0 / length(direction)),
        }
    }

    fn unit_box() -> ShapeSpec {
        ShapeSpec::Box {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 10.0, 10.0],
            rotation_deg: None,
        }
    }

    /// The same box, turned by `degrees` about the z axis.
    fn turned_box(degrees: f64) -> ShapeSpec {
        ShapeSpec::Box {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 10.0, 10.0],
            rotation_deg: Some([0.0, 0.0, degrees]),
        }
    }

    /// An ellipsoid with three different radii, so a per-axis mistake shows.
    fn ellipsoid_spec(rotation_deg: Option<Vec3>) -> ShapeSpec {
        ShapeSpec::Ellipsoid {
            center: [1.0, 2.0, 3.0],
            radii: [6.0, 3.0, 2.0],
            rotation_deg,
        }
    }

    /// A tube 20 mm along x about the origin, straight or bent.
    fn tube_spec(bend: Option<Vec3>) -> ShapeSpec {
        ShapeSpec::Tube {
            p1: [0.0, 0.0, 0.0],
            p2: [20.0, 0.0, 0.0],
            bend,
            radius: 2.0,
        }
    }

    /// The bend a tube carries, and a panic when it is not a tube at all.
    fn bend_of_spec(spec: &ShapeSpec) -> Option<Vec3> {
        let ShapeSpec::Tube { bend, .. } = *spec else {
            panic!("expected a tube, got {spec:?}");
        };
        bend
    }

    /// A cone 10 mm along z from the origin, 4 mm wide at the base and
    /// `radius2` at the top: a frustum, or the true cone at zero.
    fn cone_spec(radius2: f64) -> ShapeSpec {
        ShapeSpec::Cone {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            radius1: 4.0,
            radius2,
        }
    }

    /// A right triangle in the z = 0 plane, 3 mm thick, with three different
    /// vertices so a per-corner mistake shows.
    fn triangle_spec() -> ShapeSpec {
        ShapeSpec::Triangle {
            a: [0.0, 0.0, 0.0],
            b: [12.0, 0.0, 0.0],
            c: [0.0, 9.0, 0.0],
            thickness: 3.0,
        }
    }

    /// The three vertices of a triangle, and a panic when it is not one.
    fn corners_of(spec: &ShapeSpec) -> [Vec3; 3] {
        vertices(spec).unwrap_or_else(|| panic!("expected a triangle, got {spec:?}"))
    }

    /// A cylinder 20 mm along x about the origin. Its axis lies in the `z = 0`
    /// plane, which is the plane the latch tests below drag in.
    fn cylinder_spec() -> ShapeSpec {
        ShapeSpec::Cylinder {
            p1: [0.0, 0.0, 0.0],
            p2: [20.0, 0.0, 0.0],
            radius: 2.0,
        }
    }

    /// The handle of `kind` on `spec`, and a panic when it has none.
    fn handle_of(spec: &ShapeSpec, kind: HandleKind) -> Handle {
        handles(spec, 8.0)
            .into_iter()
            .find(|h| h.kind == kind)
            .unwrap_or_else(|| panic!("no {kind:?} handle"))
    }

    /// A pointer over `(x, y)` of the `z = height` plane, seen from straight
    /// above it - the camera of every latch test below, whose view direction is
    /// therefore `-z` and whose drag plane is the one the shape lies in. So a
    /// target here is simply where the handle is being pulled to.
    fn over(x: f64, y: f64, height: f64) -> Ray {
        ray_from([x, y, 100.0], [x, y, height])
    }

    /// The two ends of a shape built between a pair of points, and a panic when
    /// it is not one.
    fn ends_of(spec: &ShapeSpec) -> (Vec3, Vec3) {
        endpoints(spec).unwrap_or_else(|| panic!("expected a pair of ends, got {spec:?}"))
    }

    /// A drag of the handle of `kind`, grabbed from straight above it by the
    /// camera [`over`] describes.
    fn drag_from_above(spec: &ShapeSpec, kind: HandleKind) -> Drag {
        let handle = handle_of(spec, kind);
        let grab = over(handle.position[0], handle.position[1], handle.position[2]);
        Drag::start(handle, spec.clone(), &grab, [0.0, 0.0, -1.0]).expect("a drag")
    }

    /// The two ends the drag leaves the shape with, for the pointer here.
    fn ends_at(drag: &mut Drag, ray: &Ray, snap: Snap) -> (Vec3, Vec3) {
        ends_of(&drag.shape_at(ray, snap).expect("a shape"))
    }

    /// The two corners the drag leaves a box with, for the pointer here.
    fn box_at(drag: &mut Drag, ray: &Ray, snap: Snap) -> (Vec3, Vec3) {
        box_of(&drag.shape_at(ray, snap).expect("a shape"))
    }

    #[test]
    fn an_axis_drag_moves_the_shape_along_that_axis_and_no_other() {
        let shape = unit_box();
        let handles = handles(&shape, 8.0);
        let handle = handles
            .iter()
            .find(|h| h.kind == HandleKind::Translate(0))
            .copied()
            .expect("an x arrow");

        // Looking down z, so the ray is never parallel to the x axis.
        let start = ray_from(
            [handle.position[0], handle.position[1], 100.0],
            handle.position,
        );
        let mut drag =
            Drag::start(handle, shape.clone(), &start, [0.0, 0.0, -1.0]).expect("a drag");
        // The pointer has not moved: neither has the shape.
        let same = drag.shape_at(&start, Snap::OFF).expect("a shape");
        assert_eq!(box_of(&same), box_of(&shape));

        let target = [handle.position[0] + 7.0, handle.position[1] + 3.0, 100.0];
        let moved = ray_from(target, [target[0], target[1], handle.position[2]]);
        let after = drag.shape_at(&moved, Snap::OFF).expect("a shape");
        let (min, max) = box_of(&after);
        assert!((min[0] - 7.0).abs() < 1e-9, "x moved to {}", min[0]);
        assert!(
            min[1].abs() < 1e-9 && min[2].abs() < 1e-9,
            "off axis: {min:?}"
        );
        assert!((max[0] - 17.0).abs() < 1e-9);
        // The shape keeps its size: a translation is not a resize.
        assert_eq!(
            [max[0] - min[0], max[1] - min[1], max[2] - min[2]],
            [10.0, 10.0, 10.0]
        );
        drag.mark_changed();
        assert!(drag.has_changed());
    }

    #[test]
    fn a_free_drag_follows_the_pointer_across_the_camera_plane() {
        let shape = ShapeSpec::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 2.0,
        };
        let handles = handles(&shape, 4.0);
        let handle = handles
            .iter()
            .find(|h| h.kind == HandleKind::TranslateFree)
            .copied()
            .expect("a free handle");
        let normal = [0.0, 0.0, -1.0];
        let start = ray_from([0.0, 0.0, 50.0], [0.0, 0.0, 0.0]);
        let mut drag = Drag::start(handle, shape, &start, normal).expect("a drag");

        // A pointer that moved 5 mm across the plane moves the sphere 5 mm.
        let moved = ray_from([5.0, -3.0, 50.0], [5.0, -3.0, 0.0]);
        let ShapeSpec::Sphere { center, radius } =
            drag.shape_at(&moved, Snap::OFF).expect("a shape")
        else {
            panic!("a sphere stays a sphere");
        };
        assert!((center[0] - 5.0).abs() < 1e-9 && (center[1] + 3.0).abs() < 1e-9);
        assert!(center[2].abs() < 1e-9, "the plane has no depth to move in");
        assert!((radius - 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_face_drag_moves_one_face_and_cannot_turn_the_box_inside_out() {
        let shape = unit_box();
        let out = move_face(&shape, 0, true, 5.0);
        let (min, max) = box_of(&out);
        assert_eq!(min, [0.0, 0.0, 0.0]);
        assert!((max[0] - 15.0).abs() < 1e-12);
        assert_eq!([max[1], max[2]], [10.0, 10.0]);

        // Dragged past the opposite face, the box stops at the minimum extent.
        let collapsed = move_face(&shape, 1, false, 50.0);
        let (min, max) = box_of(&collapsed);
        assert!((max[1] - min[1] - constants::VIEW_EDIT_MIN_EXTENT_MM).abs() < 1e-12);
        assert!(max[1] > min[1], "a box may never invert");

        // The negative face moves the other way.
        let (min, _) = box_of(&move_face(&shape, 2, false, -4.0));
        assert!((min[2] + 4.0).abs() < 1e-12);
    }

    /// The primitive a corner drag is built on grows whichever of the three
    /// axes it is handed, and keeps the opposite corner fixed. What a *drag* of
    /// it hands over is one axis at a time; see the latch tests below.
    #[test]
    fn moving_a_corner_grows_the_axes_it_is_given_and_keeps_the_far_corner() {
        let shape = unit_box();
        let out = move_corner(&shape, [true, true, true], [2.0, -3.0, 4.0]);
        let (min, max) = box_of(&out);
        assert_eq!(min, [0.0, 0.0, 0.0], "the opposite corner is fixed");
        assert_eq!(max, [12.0, 7.0, 14.0]);

        let out = move_corner(&shape, [false, true, false], [1.0, 1.0, 1.0]);
        let (min, max) = box_of(&out);
        assert_eq!(min, [1.0, 0.0, 1.0]);
        assert_eq!(max, [10.0, 11.0, 10.0]);

        // And no drag can invert an axis.
        let squashed = move_corner(&shape, [true, true, true], [-100.0; 3]);
        let (min, max) = box_of(&squashed);
        for d in 0..3 {
            assert!((max[d] - min[d] - constants::VIEW_EDIT_MIN_EXTENT_MM).abs() < 1e-12);
        }
    }

    /// A corner grows the one edge the gesture set off along, and keeps it for
    /// the rest of the drag: aiming at a width is not aiming at a depth and a
    /// height as well, however far the hand wanders on the way out.
    #[test]
    fn a_corner_drag_grows_the_one_axis_the_gesture_set_off_along() {
        let shape = unit_box();
        assert_eq!(
            handle_of(&shape, HandleKind::Corner([true; 3])).position,
            [10.0; 3]
        );
        let mut drag = drag_from_above(&shape, HandleKind::Corner([true; 3]));

        // Four millimetres out along x, one across it along y.
        let (min, max) = box_at(&mut drag, &over(14.0, 11.0, 10.0), Snap::OFF);
        assert_eq!(min, [0.0; 3], "the opposite corner is fixed");
        assert!((max[0] - 14.0).abs() < 1e-9, "{max:?}");
        assert_eq!(
            [max[1], max[2]],
            [10.0, 10.0],
            "the drag grew an axis it was not aimed along"
        );

        // Five times as far along y as along x, and it is still x that grows.
        let (min, max) = box_at(&mut drag, &over(16.0, 30.0, 10.0), Snap::OFF);
        assert_eq!(min, [0.0; 3]);
        assert!((max[0] - 16.0).abs() < 1e-9, "{max:?}");
        assert_eq!([max[1], max[2]], [10.0, 10.0], "the latch let go mid drag");

        // The dimension that is growing lands on the increment, as a face
        // handle's does.
        let (_, max) = box_at(&mut drag, &over(14.4, 11.0, 10.0), Snap::default());
        assert!((max[0] - 14.0).abs() < 1e-9, "{max:?}");

        // A gesture that sets off the other way latches the other way: same
        // corner, same camera, a drag along y.
        let mut drag = drag_from_above(&shape, HandleKind::Corner([true; 3]));
        let (min, max) = box_at(&mut drag, &over(11.0, 14.0, 10.0), Snap::OFF);
        assert_eq!(min, [0.0; 3]);
        assert!((max[1] - 14.0).abs() < 1e-9, "{max:?}");
        assert_eq!([max[0], max[2]], [10.0, 10.0]);
    }

    /// The dead zone again, on the handle whose old three-axis growth made the
    /// smallest press a resize in every direction at once.
    #[test]
    fn a_corner_drag_changes_nothing_until_it_has_left_the_dead_zone() {
        let shape = unit_box();
        let mut drag = drag_from_above(&shape, HandleKind::Corner([true; 3]));
        let inside = 0.5 * constants::VIEW_EDIT_RESIZE_LATCH_MM;
        let same = drag
            .shape_at(&over(10.0 + inside, 10.0 + inside, 10.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(same, shape, "a drag inside the dead zone changed the shape");

        let outside = 4.0 * constants::VIEW_EDIT_RESIZE_LATCH_MM;
        let (_, max) = box_at(
            &mut drag,
            &over(10.0 + outside, 10.0 + inside, 10.0),
            Snap::OFF,
        );
        assert!((max[0] - (10.0 + outside)).abs() < 1e-9, "{max:?}");
        assert_eq!([max[1], max[2]], [10.0, 10.0]);
    }

    /// An axis pointed straight at the camera is no candidate for the latch:
    /// there is no gesture on screen that means it. The drag takes the dominant
    /// one of the two that are left, rather than the noise the third projects
    /// to.
    #[test]
    fn a_corner_drag_never_latches_the_axis_pointed_at_the_camera() {
        let shape = unit_box();
        let handle = handle_of(&shape, HandleKind::Corner([true; 3]));
        // Looking along +x, so the drag plane is the box's own +x face and x is
        // the axis with nothing to show for itself.
        let at = |y: f64, z: f64| ray_from([-100.0, y, z], [10.0, y, z]);
        let mut drag =
            Drag::start(handle, shape.clone(), &at(10.0, 10.0), [-1.0, 0.0, 0.0]).expect("a drag");
        let (min, max) = box_at(&mut drag, &at(12.0, 14.0), Snap::OFF);
        assert_eq!(min, [0.0; 3]);
        assert!((max[2] - 14.0).abs() < 1e-9, "{max:?}");
        assert_eq!([max[0], max[1]], [10.0, 10.0], "{max:?}");
    }

    /// A turned box grows along its **own** edges, so the latch is read in its
    /// own frame: a drag along world +y on a box turned a quarter turn about z
    /// is a drag along its local x, and that is the edge that grows.
    #[test]
    fn a_rotated_corner_drag_latches_an_axis_of_the_boxs_own_frame() {
        let shape = turned_box(90.0);
        let handle = handle_of(&shape, HandleKind::Corner([true; 3]));
        assert!(length(difference(handle.position, [0.0, 10.0, 10.0])) < 1e-9);
        let mut drag = drag_from_above(&shape, HandleKind::Corner([true; 3]));
        let grown = drag
            .shape_at(&over(1.0, 14.0, 10.0), Snap::OFF)
            .expect("a shape");
        let extent = box_extent(&grown).expect("a box");
        assert!((extent[0] - 14.0).abs() < 1e-9, "{extent:?}");
        assert!(
            (extent[1] - 10.0).abs() < 1e-12 && (extent[2] - 10.0).abs() < 1e-12,
            "{extent:?}"
        );
        assert_eq!(
            rotation_of(&grown),
            [0.0, 0.0, 90.0],
            "a resize is not a turn"
        );
    }

    #[test]
    fn a_radius_drag_grows_and_is_floored() {
        let sphere = ShapeSpec::Sphere {
            center: [1.0, 2.0, 3.0],
            radius: 5.0,
        };
        let ShapeSpec::Sphere { center, radius } = resize_radius(&sphere, 0, 2.5) else {
            panic!("a sphere");
        };
        assert_eq!(center, [1.0, 2.0, 3.0], "a radius drag never moves a shape");
        assert!((radius - 7.5).abs() < 1e-12);
        let ShapeSpec::Sphere { radius, .. } = resize_radius(&sphere, 0, -50.0) else {
            panic!("a sphere");
        };
        assert!((radius - constants::VIEW_EDIT_MIN_EXTENT_MM).abs() < 1e-12);

        let cylinder = ShapeSpec::Cylinder {
            p1: [0.0; 3],
            p2: [0.0, 0.0, 10.0],
            radius: 1.0,
        };
        let ShapeSpec::Cylinder { p1, p2, radius } = resize_radius(&cylinder, 0, 1.0) else {
            panic!("a cylinder");
        };
        assert_eq!((p1, p2), ([0.0; 3], [0.0, 0.0, 10.0]));
        assert!((radius - 2.0).abs() < 1e-12);
    }

    /// An ellipsoid has a radius per axis, and each drag changes exactly the
    /// one it belongs to - floored on its own, and never moving the shape.
    #[test]
    fn an_ellipsoid_radius_drag_changes_one_semi_axis_and_no_other() {
        let spec = ellipsoid_spec(None);
        for component in 0..3 {
            let ShapeSpec::Ellipsoid {
                center,
                radii,
                rotation_deg,
            } = resize_radius(&spec, component, 2.5)
            else {
                panic!("an ellipsoid");
            };
            assert_eq!(center, [1.0, 2.0, 3.0], "a radius drag never moves a shape");
            assert_eq!(rotation_deg, None, "nor turns it");
            for d in 0..3 {
                let was = [6.0, 3.0, 2.0][d];
                let expected = if d == component { was + 2.5 } else { was };
                assert!((radii[d] - expected).abs() < 1e-12, "{radii:?}");
            }
            // Dragged through zero, that one radius stops at the floor and the
            // others are untouched.
            let ShapeSpec::Ellipsoid { radii, .. } = resize_radius(&spec, component, -50.0) else {
                panic!("an ellipsoid");
            };
            assert!((radii[component] - constants::VIEW_EDIT_MIN_EXTENT_MM).abs() < 1e-12);
            for d in (0..3).filter(|d| *d != component) {
                assert!((radii[d] - [6.0, 3.0, 2.0][d]).abs() < 1e-12);
            }
            assert_eq!(
                radius_of(&spec, component),
                Some([6.0, 3.0, 2.0][component])
            );
        }
        // A component that is not one of the three is not a radius at all, and
        // asking to resize it changes nothing.
        assert_eq!(radius_of(&spec, 3), None);
        assert_eq!(resize_radius(&spec, 3, 1.0), spec);
    }

    #[test]
    fn an_endpoint_drag_moves_one_cap_and_keeps_the_cylinder_from_collapsing() {
        let cylinder = ShapeSpec::Cylinder {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            radius: 2.0,
        };
        let ShapeSpec::Cylinder { p1, p2, radius } = move_endpoint(&cylinder, 1, [3.0, 0.0, 4.0])
        else {
            panic!("a cylinder");
        };
        assert_eq!(p1, [0.0, 0.0, 0.0], "the other cap stays put");
        assert_eq!(p2, [3.0, 0.0, 14.0]);
        assert!((radius - 2.0).abs() < 1e-12);

        let ShapeSpec::Cylinder { p1, p2, .. } = move_endpoint(&cylinder, 0, [0.0, 0.0, 10.0])
        else {
            panic!("a cylinder");
        };
        let len = length(difference(p2, p1));
        assert!(
            (len - constants::VIEW_EDIT_MIN_EXTENT_MM).abs() < 1e-9,
            "a cap dragged onto the other one stopped at {len}"
        );
    }

    /// The dead zone: until the pointer has really set off in some direction an
    /// end handle changes nothing at all, so the shake of a press cannot tilt
    /// what it grabbed. Past it the very same gesture is the resize it was
    /// aimed at.
    #[test]
    fn an_endpoint_drag_changes_nothing_until_it_has_left_the_dead_zone() {
        let spec = cylinder_spec();
        assert_eq!(
            handle_of(&spec, HandleKind::Endpoint(1)).position,
            [20.0, 0.0, 0.0]
        );
        let mut drag = drag_from_above(&spec, HandleKind::Endpoint(1));

        // Off the axis as well as along it, which before the latch would have
        // both lengthened the cylinder and swung it.
        let inside = 0.5 * constants::VIEW_EDIT_RESIZE_LATCH_MM;
        let same = drag
            .shape_at(&over(20.0 + inside, inside, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(same, spec, "a drag inside the dead zone changed the shape");

        let outside = 4.0 * constants::VIEW_EDIT_RESIZE_LATCH_MM;
        let after = drag
            .shape_at(&over(20.0 + outside, inside, 0.0), Snap::OFF)
            .expect("a shape");
        let (p1, p2) = ends_of(&after);
        assert_eq!(p1, [0.0; 3], "the far end stays put");
        assert!((p2[0] - (20.0 + outside)).abs() < 1e-9, "{p2:?}");
        assert_eq!([p2[1], p2[2]], [0.0, 0.0], "the axis was not held");
    }

    /// The bug this latch was written for: *"dragging the orange end handle
    /// veers into another axis"*. A drag that sets off along the shape's own
    /// axis is its **length**, and stays its length however far the hand wanders
    /// afterwards - the end slides on the line through the two of them and the
    /// cylinder keeps the direction it was pointing.
    #[test]
    fn an_endpoint_drag_along_the_axis_only_lengthens_however_the_hand_wanders() {
        let spec = cylinder_spec();
        let mut drag = drag_from_above(&spec, HandleKind::Endpoint(1));

        // Mostly along the axis, and a little across it.
        let placed = drag
            .placed_at(&over(23.0, 0.5, 0.0), Snap::OFF, &Surfaces::default())
            .expect("a placement, not the wildcard");
        let (p1, p2) = ends_of(&placed.shape);
        assert_eq!(p1, [0.0; 3], "the far end stays put");
        assert!((p2[0] - 23.0).abs() < 1e-9, "{p2:?}");
        assert_eq!(
            [p2[1], p2[2]],
            [0.0, 0.0],
            "the half millimetre across the axis moved the end off it"
        );
        // The handle the user is holding follows the end, and the shape itself
        // is what says where that is.
        assert_eq!(placed.handle_at, None);
        assert_eq!(
            handle_of(&placed.shape, HandleKind::Endpoint(1)).position,
            p2
        );

        // Ten times as far across the axis as along it, and it is still the
        // length being dragged: the classification is made once.
        let wandered = drag
            .shape_at(&over(24.0, 40.0, 0.0), Snap::OFF)
            .expect("a shape");
        let (p1, p2) = ends_of(&wandered);
        assert_eq!(p1, [0.0; 3]);
        assert!((p2[0] - 24.0).abs() < 1e-9, "{p2:?}");
        assert_eq!([p2[1], p2[2]], [0.0, 0.0], "the latch let go mid drag");

        // Back inside the dead zone is back to the shape it started as, because
        // every position is applied to the original rather than accumulated.
        let back = drag
            .shape_at(&over(20.0, 0.0, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(back, spec);
    }

    /// The other half of the latch, and the reason it is not a lock: a drag
    /// that sets off *across* the axis places the end anywhere in the camera's
    /// plane, which is the only way to point one of these shapes somewhere else
    /// - and it keeps that freedom for the rest of the gesture.
    #[test]
    fn an_endpoint_drag_across_the_axis_places_the_end_freely() {
        let spec = cylinder_spec();
        let mut drag = drag_from_above(&spec, HandleKind::Endpoint(1));

        let across = drag
            .shape_at(&over(20.5, 4.0, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(
            across,
            move_endpoint(&spec, 1, [0.5, 4.0, 0.0]),
            "a drag across the axis is the free placement it always was"
        );

        // And now mostly along the axis, which does not take the freedom back.
        let along = drag
            .shape_at(&over(40.0, 4.5, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(along, move_endpoint(&spec, 1, [20.0, 4.5, 0.0]));
    }

    /// A gesture that is exactly as much along the axis as across it is read as
    /// a resize: these are the resize handles, and the length is what they are
    /// for.
    #[test]
    fn an_endpoint_drag_that_splits_the_difference_resizes() {
        let spec = cylinder_spec();
        let mut drag = drag_from_above(&spec, HandleKind::Endpoint(1));
        let (p1, p2) = ends_at(&mut drag, &over(23.0, 3.0, 0.0), Snap::OFF);
        assert_eq!(p1, [0.0; 3]);
        assert!((p2[0] - 23.0).abs() < 1e-9, "{p2:?}");
        assert_eq!([p2[1], p2[2]], [0.0, 0.0]);
    }

    /// A camera looking straight down the shape's own axis can see no length to
    /// drag: the axis has no direction on screen to compare a gesture with, so
    /// the drag is the free placement rather than a division by a projection
    /// that is not there.
    #[test]
    fn an_endpoint_seen_down_its_own_axis_places_rather_than_resizes() {
        let spec = cylinder_spec();
        let handle = handle_of(&spec, HandleKind::Endpoint(1));
        // Looking along +x, so the drag plane is the one the cap sits in.
        let at = |y: f64, z: f64| ray_from([-100.0, y, z], [20.0, y, z]);
        let mut drag =
            Drag::start(handle, spec.clone(), &at(0.0, 0.0), [-1.0, 0.0, 0.0]).expect("a drag");
        let (p1, p2) = ends_of(&drag.shape_at(&at(5.0, 3.0), Snap::OFF).expect("a shape"));
        assert_eq!(p1, [0.0; 3]);
        assert!(
            p2.iter().all(|c| c.is_finite()),
            "the degenerate projection escaped as {p2:?}"
        );
        assert!(length(difference(p2, [20.0, 5.0, 3.0])) < 1e-9, "{p2:?}");
    }

    /// Snapping a latched drag is snapping the **length**, not each coordinate
    /// of the end: on a cylinder that does not lie along a world axis, rounding
    /// the three coordinates separately is exactly what would take the end off
    /// its own line.
    #[test]
    fn a_snapped_axial_endpoint_drag_lands_the_length_on_the_increment() {
        // 10 mm long, along (0.6, 0.8, 0).
        let spec = ShapeSpec::Cylinder {
            p1: [0.0; 3],
            p2: [6.0, 8.0, 0.0],
            radius: 1.0,
        };
        let mut drag = drag_from_above(&spec, HandleKind::Endpoint(1));
        // 3.4 mm further along the axis, which is where the pointer has to be
        // for the end to follow it there.
        let target = over(6.0 + 3.4 * 0.6, 8.0 + 3.4 * 0.8, 0.0);

        let free = drag.shape_at(&target, Snap::OFF).expect("a shape");
        let (p1, p2) = ends_of(&free);
        assert!((length(difference(p2, p1)) - 13.4).abs() < 1e-9, "{p2:?}");
        assert!(length(difference(p2, [8.04, 10.72, 0.0])) < 1e-9, "{p2:?}");

        let snapped = drag.shape_at(&target, Snap::default()).expect("a shape");
        let (p1, p2) = ends_of(&snapped);
        assert!(
            (length(difference(p2, p1)) - 13.0).abs() < 1e-9,
            "the length landed on {}",
            length(difference(p2, p1))
        );
        // Still exactly on the original line: per axis that would have been
        // (8, 11, 0), which is 13.6 mm away in another direction.
        assert!(length(difference(p2, [7.8, 10.4, 0.0])) < 1e-9, "{p2:?}");
    }

    /// A latched drag is floored where every other resize is floored: dragged
    /// at and through the far end, the shape stops at the smallest usable length
    /// still pointing the way it was, rather than turning around.
    #[test]
    fn an_axial_endpoint_drag_stops_at_the_minimum_rather_than_flipping() {
        let spec = cylinder_spec();
        let mut drag = drag_from_above(&spec, HandleKind::Endpoint(1));
        for snap in [Snap::OFF, Snap::default()] {
            // Forty millimetres past the far end.
            let (p1, p2) = ends_at(&mut drag, &over(-40.0, 0.0, 0.0), snap);
            assert_eq!(p1, [0.0; 3], "the far end stays put");
            assert!(p2[0] > 0.0, "the cylinder turned around to {p2:?}");
            assert!(
                (length(difference(p2, p1)) - constants::VIEW_EDIT_MIN_EXTENT_MM).abs() < 1e-9,
                "{p2:?}"
            );
        }
    }

    /// The same arm serves all three shapes built between a pair of points, so
    /// a tube's end keeps its curve while it slides and a cone's keeps both its
    /// radii.
    #[test]
    fn a_tube_and_a_cone_end_latch_the_way_a_cylinder_does() {
        let tube = tube_spec(Some([10.0, 6.0, 0.0]));
        let mut drag = drag_from_above(&tube, HandleKind::Endpoint(1));
        // Along the chord, with a wander across it that must not be applied.
        let pulled = drag
            .shape_at(&over(25.0, 3.0, 0.0), Snap::OFF)
            .expect("a shape");
        let (p1, p2) = ends_of(&pulled);
        assert_eq!(p1, [0.0; 3]);
        assert!((p2[0] - 25.0).abs() < 1e-9, "{p2:?}");
        assert_eq!([p2[1], p2[2]], [0.0, 0.0]);
        assert_eq!(
            bend_of_spec(&pulled),
            Some([10.0, 6.0, 0.0]),
            "the bend is not dragged along"
        );

        // A cone, whose axis runs along z: seen from -y, so its length is on
        // screen and the pointer's own y is the depth it cannot drag in.
        let cone = cone_spec(2.0);
        let handle = handle_of(&cone, HandleKind::Endpoint(1));
        let at = |x: f64, z: f64| ray_from([x, 100.0, z], [x, 0.0, z]);
        let mut drag =
            Drag::start(handle, cone.clone(), &at(0.0, 10.0), [0.0, -1.0, 0.0]).expect("a drag");
        let pulled = drag.shape_at(&at(2.0, 15.0), Snap::OFF).expect("a shape");
        let (p1, p2) = ends_of(&pulled);
        assert_eq!(p1, [0.0; 3]);
        assert!((p2[2] - 15.0).abs() < 1e-9, "{p2:?}");
        assert_eq!([p2[0], p2[1]], [0.0, 0.0], "the cone leaned over to {p2:?}");
        assert_eq!(
            (radius_of(&pulled, 0), radius_of(&pulled, 1)),
            (Some(4.0), Some(2.0)),
            "a length drag is not a taper drag"
        );
    }

    /// The drag the whole shape exists for: dragging the handle in the middle
    /// of a straight tube gives it a bend, and dragging it back onto the line
    /// between the two ends takes that bend away again.
    ///
    /// `Drag::placed_at` ends in a wildcard that produces no shape at all, so a
    /// handle whose arm is missing silently does nothing. This asserts the
    /// mutation rather than the absence of a panic.
    #[test]
    fn a_bend_drag_curves_a_straight_tube_and_straightens_it_again() {
        let spec = tube_spec(None);
        let handle = handles(&spec, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Bend)
            .expect("a bend handle");
        assert_eq!(
            handle.position,
            [10.0, 0.0, 0.0],
            "a straight tube's bend handle sits in the middle of it"
        );
        // Looking down z, so the drag plane is the one the tube lies in.
        let start = ray_from([10.0, 0.0, 100.0], handle.position);
        let mut drag = Drag::start(handle, spec.clone(), &start, [0.0, 0.0, -1.0]).expect("a drag");
        // The pointer has not moved: the tube is still straight.
        let same = drag.shape_at(&start, Snap::OFF).expect("a shape");
        assert_eq!(same, spec, "a drag that went nowhere changed the shape");

        let pulled = ray_from([10.0, 5.0, 100.0], [10.0, 5.0, 0.0]);
        let placed = drag
            .placed_at(&pulled, Snap::OFF, &Surfaces::default())
            .expect("a placement, not the wildcard");
        let bend = bend_of_spec(&placed.shape).expect("the drag must give it a bend");
        assert!(
            length(difference(bend, [10.0, 5.0, 0.0])) < 1e-9,
            "the bend followed the pointer to {bend:?}"
        );
        // It really is a curve now, and the two ends are exactly where they
        // were: bending a tube is not moving it.
        let shape = placed.shape.to_shape("test").expect("a shape");
        assert!(crate::geometry::tube_arc([0.0; 3], [20.0, 0.0, 0.0], Some(bend)).is_some());
        assert_eq!(endpoints(&placed.shape), Some(([0.0; 3], [20.0, 0.0, 0.0])));
        assert!(shape.contains(bend), "the tube passes through its own bend");
        assert!(placed.flush.is_none(), "a bend lands on no surface");
        drag.mark_changed();
        assert!(drag.has_changed());

        // Snapping applies to where the bend lands, as it does to an endpoint.
        let off_grid = ray_from([10.0, 5.4, 100.0], [10.0, 5.4, 0.0]);
        let snapped = drag.shape_at(&off_grid, Snap::default()).expect("a shape");
        assert_eq!(bend_of_spec(&snapped), Some([10.0, 5.0, 0.0]));

        // Dragged back onto the line between the ends, the bend is cleared
        // rather than kept as a curve nothing can see.
        let back = ray_from([10.0, 0.0, 100.0], [10.0, 0.0, 0.0]);
        let straight = drag.shape_at(&back, Snap::OFF).expect("a shape");
        assert_eq!(bend_of_spec(&straight), None, "the tube did not straighten");
        assert_eq!(straight, spec, "and it is the tube it started as");
    }

    /// A bend dragged back onto the line between the ends **anywhere but the
    /// middle** is stored as no bend at all - and the handle has to stay where
    /// the pointer is holding it rather than snapping to the middle of the
    /// tube, which is where the straightened shape on its own would put it.
    #[test]
    fn a_bend_straightened_off_centre_leaves_the_handle_where_the_pointer_is() {
        let spec = tube_spec(None);
        let handle = handles(&spec, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Bend)
            .expect("a bend handle");
        let start = ray_from([10.0, 0.0, 100.0], handle.position);
        let mut drag = Drag::start(handle, spec.clone(), &start, [0.0, 0.0, -1.0]).expect("a drag");
        let to = |x: f64, y: f64| ray_from([x, y, 100.0], [x, y, 0.0]);

        // Out to a curve whose bend is 5 mm short of the middle.
        let placed = drag
            .placed_at(&to(5.0, 6.0), Snap::OFF, &Surfaces::default())
            .expect("a placement");
        assert_eq!(bend_of_spec(&placed.shape), Some([5.0, 6.0, 0.0]));
        assert_eq!(placed.handle_at, Some([5.0, 6.0, 0.0]));

        // And back onto the line, still 5 mm short of the middle. The shape is
        // straight - that is the geometry, and the file says so - but the
        // handle is at the pointer, 5 mm from where the shape would put it.
        let placed = drag
            .placed_at(&to(5.0, 0.0), Snap::OFF, &Surfaces::default())
            .expect("a placement");
        assert_eq!(bend_of_spec(&placed.shape), None, "the shape is straight");
        assert_eq!(
            placed.handle_at,
            Some([5.0, 0.0, 0.0]),
            "the handle left the pointer for the middle of the tube"
        );
        assert_eq!(
            bend_of(&placed.shape),
            Some([10.0, 0.0, 0.0]),
            "which is not what the straightened shape says on its own"
        );

        // Handed to the handles, that position is where the cube is drawn and
        // where it is grabbed - volume and all.
        let mut set = handles(&placed.shape, 8.0);
        reposition(
            &mut set,
            HandleKind::Bend,
            placed.handle_at.expect("a point"),
        );
        let moved = set
            .iter()
            .find(|h| h.kind == HandleKind::Bend)
            .expect("a bend handle");
        assert_eq!(moved.position, [5.0, 0.0, 0.0]);
        assert!(moved.volume.contains([5.0, 0.0, 0.0]));
        assert!(!moved.volume.contains([10.0, 0.0, 0.0]), "it really moved");
        let at_pointer = ray_from([5.0, 0.0, 100.0], [5.0, 0.0, 0.0]);
        assert_eq!(
            grab(&at_pointer, &set).map(|h| h.kind),
            Some(HandleKind::Bend),
            "and the press that follows the pointer still grabs it"
        );
        // Every other handle is where the shape puts it: repositioning one
        // moves one.
        let untouched = handles(&placed.shape, 8.0);
        for (a, b) in set.iter().zip(untouched.iter()) {
            if a.kind != HandleKind::Bend {
                assert_eq!(a.position, b.position, "{:?} moved", a.kind);
            }
        }
    }

    /// A tube that already has a bend is dragged by that bend, from where it
    /// is - and everything else about the tube stays put.
    #[test]
    fn a_bend_drag_moves_the_bend_a_tube_already_has() {
        let spec = tube_spec(Some([10.0, 4.0, 0.0]));
        let handle = handles(&spec, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Bend)
            .expect("a bend handle");
        assert_eq!(handle.position, [10.0, 4.0, 0.0], "on the bend it carries");
        let start = ray_from([10.0, 4.0, 100.0], handle.position);
        let mut drag = Drag::start(handle, spec.clone(), &start, [0.0, 0.0, -1.0]).expect("a drag");
        let moved = ray_from([14.0, 9.0, 100.0], [14.0, 9.0, 0.0]);
        let after = drag.shape_at(&moved, Snap::OFF).expect("a shape");
        assert_eq!(bend_of_spec(&after), Some([14.0, 9.0, 0.0]));
        assert_eq!(endpoints(&after), endpoints(&spec), "the ends stayed put");
        assert_eq!(radius_of(&after, 0), radius_of(&spec, 0), "and the radius");
        // Straight through the middle of the two ends: straight again.
        let onto = ray_from([10.0, 0.0, 100.0], [10.0, 0.0, 0.0]);
        assert_eq!(
            bend_of_spec(&drag.shape_at(&onto, Snap::OFF).expect("a shape")),
            None
        );
    }

    /// On a straight tube the bend handle stands exactly where every other
    /// shape's camera-plane handle stands, so the press has one winner: the
    /// bend, which is what "drag the middle to make a curve" means. Bend the
    /// tube and the centre is the translation handle's again.
    #[test]
    fn the_middle_of_a_straight_tube_is_grabbed_as_its_bend() {
        let straight = handles(&tube_spec(None), 8.0);
        let free = straight
            .iter()
            .find(|h| h.kind == HandleKind::TranslateFree)
            .expect("a free handle");
        let bend = straight
            .iter()
            .find(|h| h.kind == HandleKind::Bend)
            .expect("a bend handle");
        assert_eq!(
            free.position, bend.position,
            "this test is about the two of them being in one place"
        );
        let at_centre = ray_from([10.0, 0.0, 100.0], [10.0, 0.0, 0.0]);
        assert_eq!(
            grab(&at_centre, &straight).map(|h| h.kind),
            Some(HandleKind::Bend)
        );

        let bent = handles(&tube_spec(Some([10.0, 6.0, 0.0])), 8.0);
        assert_eq!(
            grab(&at_centre, &bent).map(|h| h.kind),
            Some(HandleKind::TranslateFree),
            "with the bend elsewhere, the centre moves the tube again"
        );
        let at_bend = ray_from([10.0, 6.0, 100.0], [10.0, 6.0, 0.0]);
        assert_eq!(
            grab(&at_bend, &bent).map(|h| h.kind),
            Some(HandleKind::Bend)
        );
    }

    /// A tube's end handles move its ends and leave its bend where it is, so
    /// dragging an end reshapes the curve rather than dragging it along.
    #[test]
    fn a_tube_endpoint_drag_keeps_the_bend_and_the_far_end() {
        let spec = tube_spec(Some([10.0, 6.0, 0.0]));
        let moved = move_endpoint(&spec, 1, [4.0, 0.0, 3.0]);
        let ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        } = moved
        else {
            panic!("a tube stays a tube");
        };
        assert_eq!(p1, [0.0; 3], "the other end stays put");
        assert_eq!(p2, [24.0, 0.0, 3.0]);
        assert_eq!(
            bend,
            Some([10.0, 6.0, 0.0]),
            "the bend is not dragged along"
        );
        assert!((radius - 2.0).abs() < 1e-12);

        // Dragged onto the other end, it stops at the smallest usable extent,
        // exactly as a cylinder's does.
        let collapsed = move_endpoint(&spec, 0, [20.0, 0.0, 0.0]);
        let (p1, p2) = endpoints(&collapsed).expect("a tube");
        assert!((length(difference(p2, p1)) - constants::VIEW_EDIT_MIN_EXTENT_MM).abs() < 1e-9);
    }

    /// Everything a tube's other handles do, it does with its bend carried
    /// along: a turn turns the curve rather than reshaping it, and a
    /// translation moves it whole.
    #[test]
    fn a_tube_carries_its_bend_through_a_turn_and_a_translation() {
        let spec = tube_spec(Some([10.0, 6.0, 0.0]));
        let moved = translate(&spec, [1.0, -2.0, 3.0]);
        assert_eq!(bend_of_spec(&moved), Some([11.0, 4.0, 3.0]));
        assert_eq!(
            endpoints(&moved),
            Some(([1.0, -2.0, 3.0], [21.0, -2.0, 3.0]))
        );

        // A quarter turn about z through the middle of the two ends.
        let turned = turn(&spec, 2, [0.0, 0.0, 1.0], 90.0, Snap::OFF);
        let (p1, p2) = endpoints(&turned).expect("a tube");
        assert!(length(difference(p1, [10.0, -10.0, 0.0])) < 1e-9, "{p1:?}");
        assert!(length(difference(p2, [10.0, 10.0, 0.0])) < 1e-9, "{p2:?}");
        let bend = bend_of_spec(&turned).expect("a bend");
        assert!(length(difference(bend, [4.0, 0.0, 0.0])) < 1e-9, "{bend:?}");
        // A turn is rigid: the bend is as far off the chord as it ever was.
        let before = tube_spec(Some([10.0, 6.0, 0.0]));
        let (a, b) = (
            crate::geometry::tube_arc([0.0; 3], [20.0, 0.0, 0.0], bend_of_spec(&before)),
            crate::geometry::tube_arc(p1, p2, Some(bend)),
        );
        assert!((a.expect("an arc").radius - b.expect("an arc").radius).abs() < 1e-9);
        assert!((a.expect("an arc").span - b.expect("an arc").span).abs() < 1e-9);
    }

    #[test]
    fn a_drag_is_applied_to_the_shape_it_started_on_rather_than_accumulated() {
        let shape = unit_box();
        let handle = handles(&shape, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Translate(2))
            .expect("a z arrow");
        let start = ray_from([100.0, 0.0, handle.position[2]], handle.position);
        let mut drag =
            Drag::start(handle, shape.clone(), &start, [-1.0, 0.0, 0.0]).expect("a drag");

        let out_and_back = |z: f64| {
            let to = [handle.position[0], handle.position[1], z];
            ray_from([100.0, 0.0, z], to)
        };
        // Fifty intermediate positions, then back to the start.
        for step in 1..50 {
            drag.shape_at(&out_and_back(handle.position[2] + step as f64), Snap::OFF)
                .expect("a shape");
        }
        let back = drag
            .shape_at(&out_and_back(handle.position[2]), Snap::OFF)
            .expect("a shape");
        assert_eq!(
            box_of(&back),
            box_of(&shape),
            "a drag that returned to its start must leave the shape alone"
        );
    }

    #[test]
    fn a_ray_along_a_handle_axis_gives_no_parameter_rather_than_a_runaway() {
        let along = ray_from([0.0, 0.0, -10.0], [0.0, 0.0, 0.0]);
        assert!(axis_parameter(&along, [0.0; 3], [0.0, 0.0, 1.0]).is_none());
        // And a plane seen edge on is not crossed either.
        let edge_on = ray_from([0.0, 0.0, -10.0], [1.0, 0.0, -10.0]);
        assert!(plane_point(&edge_on, [0.0; 3], [0.0, 0.0, 1.0]).is_none());
        // A drag cannot even start on such a ray.
        let handle = handles(&unit_box(), 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Translate(2))
            .expect("a z arrow");
        let parallel = ray_from(
            [handle.position[0], handle.position[1], -50.0],
            handle.position,
        );
        assert!(Drag::start(handle, unit_box(), &parallel, [0.0, 0.0, 1.0]).is_none());
    }

    #[test]
    fn a_gizmo_hit_is_the_nearest_handle_and_a_miss_is_a_miss() {
        let shape = unit_box();
        let handles = handles(&shape, 8.0);
        let x_arrow = handles
            .iter()
            .find(|h| h.kind == HandleKind::Translate(0))
            .expect("an x arrow");
        let ray = ray_from(
            [x_arrow.position[0], x_arrow.position[1], 100.0],
            x_arrow.position,
        );
        assert_eq!(
            grab(&ray, &handles).map(|h| h.kind),
            Some(HandleKind::Translate(0))
        );
        let miss = ray_from([1000.0, 1000.0, 100.0], [1000.0, 1000.0, 0.0]);
        assert!(grab(&miss, &handles).is_none());
        assert!(grab(&ray, &[]).is_none());
    }

    #[test]
    fn every_shape_kind_gets_the_handles_its_own_geometry_needs() {
        let box_handles = handles(&unit_box(), 8.0);
        assert_eq!(
            box_handles
                .iter()
                .filter(|h| matches!(h.kind, HandleKind::Corner(_)))
                .count(),
            8
        );
        assert_eq!(
            box_handles
                .iter()
                .filter(|h| matches!(h.kind, HandleKind::Face(_, _)))
                .count(),
            6
        );
        assert!(
            !box_handles
                .iter()
                .any(|h| matches!(h.kind, HandleKind::Radius(_)))
        );

        let cylinder = ShapeSpec::Cylinder {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            radius: 2.0,
        };
        let handles_c = handles(&cylinder, 8.0);
        assert_eq!(
            handles_c
                .iter()
                .filter(|h| matches!(h.kind, HandleKind::Endpoint(_)))
                .count(),
            2
        );
        assert_eq!(
            handles_c
                .iter()
                .filter(|h| h.kind == HandleKind::Radius(0))
                .count(),
            1
        );

        let sphere = ShapeSpec::Sphere {
            center: [0.0; 3],
            radius: 3.0,
        };
        let handles_s = handles(&sphere, 8.0);
        assert_eq!(handles_s.len(), 5, "three arrows, a free handle, a radius");

        let handles_t = handles(&tube_spec(None), 8.0);
        assert_eq!(
            handles_t
                .iter()
                .filter(|h| matches!(h.kind, HandleKind::Endpoint(_)))
                .count(),
            2
        );
        assert_eq!(
            handles_t
                .iter()
                .filter(|h| h.kind == HandleKind::Radius(0))
                .count(),
            1
        );
        assert_eq!(
            handles_t
                .iter()
                .filter(|h| h.kind == HandleKind::Bend)
                .count(),
            1,
            "a tube has exactly one middle handle, bent or not"
        );
        // And it is the only kind that has one.
        for set in [&box_handles, &handles_c] {
            assert!(!set.iter().any(|h| h.kind == HandleKind::Bend));
        }

        let handles_e = handles(&ellipsoid_spec(None), 8.0);
        assert_eq!(
            handles_e
                .iter()
                .filter(|h| matches!(h.kind, HandleKind::Radius(_)))
                .count(),
            3,
            "an ellipsoid has a radius per axis"
        );
        for d in 0..3 {
            assert!(handles_e.iter().any(|h| h.kind == HandleKind::Radius(d)));
        }

        // A cone: an end handle and a radius handle at each of its two ends.
        let handles_n = handles(&cone_spec(2.0), 8.0);
        for component in 0..2 {
            assert!(
                handles_n
                    .iter()
                    .any(|h| h.kind == HandleKind::Endpoint(component))
            );
            assert!(
                handles_n
                    .iter()
                    .any(|h| h.kind == HandleKind::Radius(component))
            );
        }
        assert!(!handles_n.iter().any(|h| h.kind == HandleKind::Radius(2)));
        assert!(!handles_n.iter().any(|h| h.kind == HandleKind::Thickness));

        // A triangle: a handle on each corner and one for the thickness, and
        // no radius anywhere - it has none to drag.
        let handles_r = handles(&triangle_spec(), 8.0);
        assert_eq!(
            handles_r
                .iter()
                .filter(|h| matches!(h.kind, HandleKind::Vertex(_)))
                .count(),
            3
        );
        for v in 0..3 {
            assert!(handles_r.iter().any(|h| h.kind == HandleKind::Vertex(v)));
        }
        assert_eq!(
            handles_r
                .iter()
                .filter(|h| h.kind == HandleKind::Thickness)
                .count(),
            1
        );
        assert!(
            !handles_r
                .iter()
                .any(|h| matches!(h.kind, HandleKind::Radius(_)))
        );
        // And a vertex handle is the triangle's alone, as the bend is the
        // tube's.
        for set in [&box_handles, &handles_c, &handles_s, &handles_e, &handles_n] {
            assert!(!set.iter().any(|h| matches!(h.kind, HandleKind::Vertex(_))));
            assert!(!set.iter().any(|h| h.kind == HandleKind::Thickness));
        }

        // Every shape gets the four translation handles.
        for set in [
            &box_handles,
            &handles_c,
            &handles_s,
            &handles_e,
            &handles_n,
            &handles_r,
        ] {
            assert!(set.iter().any(|h| h.kind == HandleKind::TranslateFree));
            for d in 0..3 {
                assert!(set.iter().any(|h| h.kind == HandleKind::Translate(d)));
            }
        }
        // A box, a cylinder, an ellipsoid and a cone can be turned and get an
        // arc per axis; a sphere looks the same whichever way it is turned, and
        // a triangle is already pointed by its own three corners, so neither
        // gets any.
        for set in [&box_handles, &handles_c, &handles_e, &handles_n] {
            for c in 0..3 {
                assert!(set.iter().any(|h| h.kind == HandleKind::Rotate(c)));
            }
        }
        for set in [&handles_s, &handles_r] {
            assert!(!set.iter().any(|h| matches!(h.kind, HandleKind::Rotate(_))));
        }
        assert!(!is_rotatable(&triangle_spec()) && !is_rotatable(&handles_s_spec()));
    }

    /// The sphere fixture the inventory above turns down a rotation gizmo for.
    fn handles_s_spec() -> ShapeSpec {
        ShapeSpec::Sphere {
            center: [0.0; 3],
            radius: 3.0,
        }
    }

    /// A cone wears a radius handle at each of its two ends, measured from the
    /// cap it belongs to - which is what makes one drag change one radius.
    #[test]
    fn a_cone_carries_a_radius_handle_at_each_of_its_two_ends() {
        let spec = cone_spec(2.0);
        let set = handles(&spec, 8.0);
        for (component, origin, radius) in [(0, [0.0, 0.0, 0.0], 4.0), (1, [0.0, 0.0, 10.0], 2.0)] {
            let handle = set
                .iter()
                .find(|h| h.kind == HandleKind::Radius(component))
                .unwrap_or_else(|| panic!("a radius handle for end {component}"));
            assert_eq!(radius_origin(&spec, component), origin);
            assert!(
                (length(difference(handle.position, origin)) - radius).abs() < 1e-9,
                "end {component}: {:?} is not {radius} mm from {origin:?}",
                handle.position
            );
            // On the cap's own plane: a radius is measured across the axis.
            assert!((handle.position[2] - origin[2]).abs() < 1e-9);
            assert!((inner(handle.axis, [0.0, 0.0, 1.0])).abs() < 1e-9);
            assert_eq!(radius_of(&spec, component), Some(radius));
        }
        assert_eq!(radius_of(&spec, 2), None, "a cone has two radii, not three");
        // Both ends move, and both radii come along unchanged.
        let moved = move_endpoint(&spec, 1, [3.0, 0.0, 2.0]);
        assert_eq!(endpoints(&moved), Some(([0.0; 3], [3.0, 0.0, 12.0])));
        assert_eq!(radius_of(&moved, 0), Some(4.0));
        assert_eq!(radius_of(&moved, 1), Some(2.0));
        // And a cone turns like the cylinder it generalizes: its caps come
        // round its centre and its radii are not reshaped by the turn.
        let turned = turn(&spec, 0, [1.0, 0.0, 0.0], 90.0, Snap::OFF);
        let (p1, p2) = endpoints(&turned).expect("two ends");
        assert!(length(difference(p1, [0.0, 5.0, 5.0])) < 1e-9, "{p1:?}");
        assert!(length(difference(p2, [0.0, -5.0, 5.0])) < 1e-9, "{p2:?}");
        assert_eq!(radius_of(&turned, 1), Some(2.0));
        assert!(is_rotatable(&spec), "a cone is pointed by its two ends");
    }

    /// The drag the apex exists for: the narrow end goes all the way to zero,
    /// which is the true cone, while the wide end keeps the floor every other
    /// radius has - and both go through `Drag::placed_at`, whose wildcard would
    /// otherwise let a missing arm do nothing at all.
    #[test]
    fn a_cone_radius_drag_reaches_the_apex_and_the_wide_end_never_collapses() {
        let spec = cone_spec(2.0);
        assert_eq!(radius_of(&resize_radius(&spec, 1, -50.0), 1), Some(0.0));
        assert_eq!(
            radius_of(&resize_radius(&spec, 0, -50.0), 0),
            Some(constants::VIEW_EDIT_MIN_EXTENT_MM)
        );
        // The component outside the two it has changes nothing.
        assert_eq!(resize_radius(&spec, 2, 5.0), spec);

        // Through the drag itself, on the handle the pointer would grab.
        let handle = handles(&spec, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Radius(1))
            .expect("the second radius handle");
        let start = ray_from(
            [handle.position[0], handle.position[1], 100.0],
            handle.position,
        );
        let mut drag = Drag::start(handle, spec.clone(), &start, [0.0, 0.0, -1.0]).expect("a drag");
        let to = |along: f64| {
            let target = sum(handle.position, scale(handle.axis, along));
            ray_from([target[0], target[1], 100.0], target)
        };
        // Pulled out, and then dragged well past the axis: the shape comes to
        // its point and stops there.
        let placed = drag
            .placed_at(&to(1.0), Snap::OFF, &Surfaces::default())
            .expect("a placement, not the wildcard");
        assert_eq!(radius_of(&placed.shape, 1), Some(3.0));
        assert_eq!(radius_of(&placed.shape, 0), Some(4.0), "one radius moved");
        assert_eq!(endpoints(&placed.shape), endpoints(&spec), "and no end");
        let apex = drag.shape_at(&to(-9.0), Snap::OFF).expect("a shape");
        assert_eq!(
            radius_of(&apex, 1),
            Some(0.0),
            "the apex has to be reachable by drag"
        );
        assert!(
            apex.to_shape("test").is_ok(),
            "and the cone it leaves is a shape the configuration accepts"
        );
        // Snapping applies to the radius, as it does to every other dimension.
        let snapped = drag.shape_at(&to(1.4), Snap::default()).expect("a shape");
        assert_eq!(radius_of(&snapped, 1), Some(3.0));
        drag.mark_changed();
        assert!(drag.has_changed());
    }

    /// On a true cone the second radius handle stands exactly on the second
    /// end handle - the apex is both points - and the press has one winner: the
    /// radius, which is the handle that can bring the cone back off its point
    /// and which separates the two the moment it moves.
    #[test]
    fn the_apex_of_a_cone_is_grabbed_as_the_radius_that_can_undo_it() {
        let pointed = handles(&cone_spec(0.0), 8.0);
        let radius = pointed
            .iter()
            .find(|h| h.kind == HandleKind::Radius(1))
            .expect("a second radius handle");
        let end = pointed
            .iter()
            .find(|h| h.kind == HandleKind::Endpoint(1))
            .expect("a second end handle");
        assert_eq!(
            radius.position, end.position,
            "this test is about the two of them being in one place"
        );
        let at_apex = ray_from([0.0, 0.0, 100.0], [0.0, 0.0, 10.0]);
        assert_eq!(
            grab(&at_apex, &pointed).map(|h| h.kind),
            Some(HandleKind::Radius(1))
        );

        // With the end blunt again the two are apart, and each is grabbed
        // where it is drawn.
        let blunt = handles(&cone_spec(2.0), 8.0);
        assert_eq!(
            grab(&at_apex, &blunt).map(|h| h.kind),
            Some(HandleKind::Endpoint(1)),
            "with a radius to stand on, the end handle is reachable again"
        );
        let handle = blunt
            .iter()
            .find(|h| h.kind == HandleKind::Radius(1))
            .expect("a second radius handle");
        let at_radius = ray_from(sum(handle.position, [0.0, 0.0, 100.0]), handle.position);
        assert_eq!(
            grab(&at_radius, &blunt).map(|h| h.kind),
            Some(HandleKind::Radius(1))
        );
    }

    /// A triangle is dragged by its corners: one moves, the other two and the
    /// thickness stay, and no drag may flatten it - which is the one thing the
    /// configuration will not hold.
    #[test]
    fn a_triangle_vertex_drag_moves_that_corner_and_is_held_off_flat() {
        let spec = triangle_spec();
        let moved = move_vertex(&spec, 2, [1.0, -2.0, 4.0]);
        let corners = corners_of(&moved);
        assert_eq!(corners[0], [0.0, 0.0, 0.0], "the other corners stay put");
        assert_eq!(corners[1], [12.0, 0.0, 0.0]);
        assert_eq!(corners[2], [1.0, 7.0, 4.0]);
        assert_eq!(thickness_of(&moved), Some(3.0), "and the thickness");

        // Dragged almost onto the line through the other two - the x axis
        // here - it is held off it by the smallest usable extent, along the
        // perpendicular it came in on, so it stays on the side it came from.
        let flattened = move_vertex(&spec, 2, [0.0, -8.95, 0.0]);
        let corners = corners_of(&flattened);
        assert!(
            (corners[2][1] - constants::VIEW_EDIT_MIN_EXTENT_MM).abs() < 1e-9,
            "the corner landed at {:?}",
            corners[2]
        );
        assert!(
            flattened.to_shape("test").is_ok(),
            "a drag may not leave a shape the configuration refuses"
        );
        // Landed exactly on it, there is no perpendicular it came in on and
        // any one will do - what matters is that it is off the line, and that
        // the shape is one the configuration accepts.
        let squashed = move_vertex(&spec, 2, [0.0, -9.0, 0.0]);
        let corner = corners_of(&squashed)[2];
        let off_line = (corner[1] * corner[1] + corner[2] * corner[2]).sqrt();
        assert!(
            (off_line - constants::VIEW_EDIT_MIN_EXTENT_MM).abs() < 1e-9,
            "the corner landed at {corner:?}, {off_line} off the line"
        );
        assert!(squashed.to_shape("test").is_ok());
        // Past it, the drag is free again: the guard is a floor around the
        // line rather than a wall, exactly as `held_apart` is around a point,
        // so a corner really can be dragged through to the other side.
        let through = move_vertex(&spec, 2, [0.0, -20.0, 0.0]);
        assert_eq!(corners_of(&through)[2], [0.0, -11.0, 0.0]);
        // A component outside the three it has changes nothing.
        assert_eq!(move_vertex(&spec, 3, [1.0; 3]), spec);

        // Through the drag itself, whose wildcard would otherwise let a
        // missing arm silently do nothing. Seen face on, so it is one of the
        // two gestures that *size* the triangle rather than aim it; which of
        // them is what the latch tests below are about.
        let mut drag = drag_from_above(&spec, HandleKind::Vertex(1));
        assert_eq!(
            handle_of(&spec, HandleKind::Vertex(1)).position,
            [12.0, 0.0, 0.0]
        );
        let same = drag
            .shape_at(&over(12.0, 0.0, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(same, spec, "a drag that went nowhere changed the shape");
        let placed = drag
            .placed_at(&over(16.0, 5.0, 0.0), Snap::OFF, &Surfaces::default())
            .expect("a placement, not the wildcard");
        // Five millimetres along the edge this corner faces - the line through
        // the other two, which here is the y axis - against four across it.
        assert_eq!(corners_of(&placed.shape)[1], [12.0, 5.0, 0.0]);
        assert_eq!(corners_of(&placed.shape)[0], [0.0, 0.0, 0.0]);
        assert!(placed.flush.is_none(), "a vertex lands on no surface");
        assert_eq!(
            placed.handle_at, None,
            "the shape itself says where the corner is"
        );
        // Snapping applies to how far it has slid along that line, which is what
        // keeps it on the line: per axis, the 4.4 across would have moved it too.
        let snapped = drag
            .shape_at(&over(16.4, 5.6, 0.0), Snap::default())
            .expect("a shape");
        assert_eq!(corners_of(&snapped)[1], [12.0, 6.0, 0.0]);
        drag.mark_changed();
        assert!(drag.has_changed());
    }

    /// How far a corner stands off the edge it faces: the height a latched
    /// altitude drag sets, and what an edge drag has to leave alone.
    fn altitude_of(spec: &ShapeSpec, vertex: usize) -> f64 {
        let points = corners_of(spec);
        let (first, second) = match vertex {
            0 => (points[1], points[2]),
            1 => (points[2], points[0]),
            _ => (points[0], points[1]),
        };
        let edge = difference(second, first);
        let unit = scale(edge, 1.0 / length(edge));
        let offset = difference(points[vertex], first);
        length(difference(offset, scale(unit, inner(offset, unit))))
    }

    /// The number the callout shows beside a vertex drag: the median from the
    /// middle of the edge the corner faces.
    fn median_of(spec: &ShapeSpec, vertex: usize) -> f64 {
        let points = corners_of(spec);
        let (first, second) = match vertex {
            0 => (points[1], points[2]),
            1 => (points[2], points[0]),
            _ => (points[0], points[1]),
        };
        length(difference(points[vertex], midpoint(first, second)))
    }

    /// A corner pulled along the edge it faces shears the triangle: the base
    /// stays where it is, the corner keeps the height it stood at, and it keeps
    /// doing so however far the hand wanders off that line afterwards.
    #[test]
    fn a_vertex_drag_along_the_facing_edge_keeps_the_height_it_started_at() {
        let spec = triangle_spec();
        // The third corner, at (0, 9, 0), faces the edge from a to b - the x
        // axis - and stands 9 mm off it.
        assert!((altitude_of(&spec, 2) - 9.0).abs() < 1e-12);
        let mut drag = drag_from_above(&spec, HandleKind::Vertex(2));

        let slid = drag
            .shape_at(&over(4.0, 9.5, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(corners_of(&slid)[2], [4.0, 9.0, 0.0]);
        assert_eq!(
            [corners_of(&slid)[0], corners_of(&slid)[1]],
            [[0.0; 3], [12.0, 0.0, 0.0]],
            "the base moved"
        );
        assert!(
            (altitude_of(&slid, 2) - 9.0).abs() < 1e-12,
            "the height changed to {}",
            altitude_of(&slid, 2)
        );

        // Ten times as far across the edge as along it, and it is still the
        // slide along it: the classification is made once.
        let wandered = drag
            .shape_at(&over(6.0, 40.0, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(corners_of(&wandered)[2], [6.0, 9.0, 0.0]);
    }

    /// The other dimension of the corner: its height off that edge, at a base
    /// that does not move - and the number beside the drag follows it.
    #[test]
    fn a_vertex_drag_along_its_own_altitude_changes_the_height_and_not_the_base() {
        let spec = triangle_spec();
        let before = median_of(&spec, 2);
        let mut drag = drag_from_above(&spec, HandleKind::Vertex(2));

        let raised = drag
            .shape_at(&over(0.5, 14.0, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(corners_of(&raised)[2], [0.0, 14.0, 0.0]);
        assert!((altitude_of(&raised, 2) - 14.0).abs() < 1e-12);
        assert_eq!(
            [corners_of(&raised)[0], corners_of(&raised)[1]],
            [[0.0; 3], [12.0, 0.0, 0.0]],
            "the base moved"
        );
        assert!(
            median_of(&raised, 2) > before,
            "the number beside the drag did not follow it"
        );

        // As much along the edge as up the altitude is the altitude: the height
        // is the number a corner drag is usually after.
        let mut drag = drag_from_above(&spec, HandleKind::Vertex(2));
        let tied = drag
            .shape_at(&over(3.0, 12.0, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(corners_of(&tied)[2], [0.0, 12.0, 0.0]);
    }

    /// The gesture the latch must not take away: a corner dragged out of its
    /// triangle's own plane is how a prism is aimed, and it stays as free as it
    /// has always been.
    #[test]
    fn a_vertex_drag_out_of_the_triangles_plane_is_free_as_it_always_was() {
        let spec = triangle_spec();
        let handle = handle_of(&spec, HandleKind::Vertex(2));
        // Seen from -y, so the triangle is edge on: its normal is on screen and
        // its altitude is the depth this camera cannot drag in.
        let at = |x: f64, z: f64| ray_from([x, 100.0, z], [x, 9.0, z]);
        let mut drag =
            Drag::start(handle, spec.clone(), &at(0.0, 0.0), [0.0, -1.0, 0.0]).expect("a drag");
        let aimed = drag.shape_at(&at(3.0, 8.0), Snap::OFF).expect("a shape");
        assert_eq!(
            aimed,
            move_vertex(&spec, 2, [3.0, 0.0, 8.0]),
            "an out of plane drag is the free placement it always was"
        );

        // And it keeps that freedom for the rest of the gesture, even where the
        // edge would now dominate.
        let along = drag.shape_at(&at(20.0, 9.0), Snap::OFF).expect("a shape");
        assert_eq!(along, move_vertex(&spec, 2, [20.0, 0.0, 9.0]));
    }

    /// The dead zone, and the collapse guard, on the latched paths: a corner
    /// slid down its own altitude stops a tenth of a millimetre off the line
    /// through the other two, because a triangle of no area is a shape the
    /// configuration refuses - and past it the drag is free again, exactly as
    /// the free path has always been.
    #[test]
    fn a_latched_vertex_drag_is_still_held_off_the_line_it_would_collapse_on() {
        let spec = triangle_spec();
        let mut drag = drag_from_above(&spec, HandleKind::Vertex(2));
        let inside = 0.5 * constants::VIEW_EDIT_RESIZE_LATCH_MM;
        let same = drag
            .shape_at(&over(inside, 9.0 + inside, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(same, spec, "a drag inside the dead zone changed the shape");

        // Down the altitude, to a twentieth of a millimetre off the base.
        let squashed = drag
            .shape_at(&over(0.2, 0.05, 0.0), Snap::OFF)
            .expect("a shape");
        let corner = corners_of(&squashed)[2];
        assert_eq!(corner, [0.0, constants::VIEW_EDIT_MIN_EXTENT_MM, 0.0]);
        assert!(
            squashed.to_shape("test").is_ok(),
            "a latched drag may not leave a shape the configuration refuses"
        );
        // Past the line, the guard is a floor rather than a wall here too.
        let through = drag
            .shape_at(&over(0.2, -11.0, 0.0), Snap::OFF)
            .expect("a shape");
        assert_eq!(corners_of(&through)[2], [0.0, -11.0, 0.0]);
    }

    /// A triangle already flat has no altitude to slide along and no plane to be
    /// taken out of, so its corners are placed freely rather than held to
    /// directions computed from nothing.
    #[test]
    fn a_flat_triangle_places_its_corners_freely_rather_than_latching() {
        let spec = ShapeSpec::Triangle {
            a: [0.0; 3],
            b: [12.0, 0.0, 0.0],
            c: [6.0, 0.0, 0.0],
            thickness: 3.0,
        };
        assert!(
            spec.to_shape("test").is_err(),
            "this fixture is the degenerate one"
        );
        let mut drag = drag_from_above(&spec, HandleKind::Vertex(2));
        let placed = drag
            .shape_at(&over(9.0, 4.0, 0.0), Snap::OFF)
            .expect("a shape");
        let corner = corners_of(&placed)[2];
        assert!(
            corner.iter().all(|c| c.is_finite()),
            "the flat triangle escaped as {corner:?}"
        );
        assert_eq!(placed, move_vertex(&spec, 2, [3.0, 4.0, 0.0]));
    }

    /// The one dimension a triangle's corners do not set: its thickness, moved
    /// from a handle on the face it grows, half of it either side.
    #[test]
    fn a_triangle_thickness_drag_grows_it_from_both_faces() {
        let spec = triangle_spec();
        let handle = handles(&spec, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Thickness)
            .expect("a thickness handle");
        // On the face, at the centroid of the triangle: half a thickness out
        // along the normal, which here is +z.
        assert!(length(difference(handle.position, [4.0, 3.0, 1.5])) < 1e-9);
        assert!(length(difference(handle.axis, [0.0, 0.0, 1.0])) < 1e-9);

        let start = ray_from([4.0, 100.0, 1.5], handle.position);
        let mut drag = Drag::start(handle, spec.clone(), &start, [0.0, -1.0, 0.0]).expect("a drag");
        let to = |z: f64| ray_from([4.0, 100.0, z], [4.0, 3.0, z]);
        // The handle moved 2 mm, so the prism is 4 mm thicker - it grows
        // equally either side, and the face stays under the pointer.
        let placed = drag
            .placed_at(&to(3.5), Snap::OFF, &Surfaces::default())
            .expect("a placement, not the wildcard");
        assert_eq!(thickness_of(&placed.shape), Some(7.0));
        assert_eq!(
            corners_of(&placed.shape),
            corners_of(&spec),
            "a thickness drag never moves a corner"
        );
        let after = handles(&placed.shape, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Thickness)
            .expect("a thickness handle");
        assert!(
            (after.position[2] - 3.5).abs() < 1e-9,
            "{:?}",
            after.position
        );

        // Snapping lands the thickness itself on the increment, which is the
        // number the file holds rather than the half the handle moved.
        let snapped = drag.shape_at(&to(3.2), Snap::default()).expect("a shape");
        assert_eq!(thickness_of(&snapped), Some(6.0));
        // And it can never be dragged through nothing.
        assert_eq!(
            thickness_of(&resize_thickness(&spec, -50.0)),
            Some(constants::VIEW_EDIT_MIN_EXTENT_MM)
        );
        assert_eq!(resize_thickness(&unit_box(), 1.0), unit_box());
        drag.mark_changed();
        assert!(drag.has_changed());
    }

    /// Each radius handle sits on the end of its own semi-axis, in the
    /// ellipsoid's frame, and slides along that axis - which is what makes a
    /// drag of it change that radius and nothing else.
    #[test]
    fn an_ellipsoid_carries_a_radius_handle_on_each_of_its_own_axes() {
        let radii = [6.0, 3.0, 2.0];
        let centre = [1.0, 2.0, 3.0];
        let plain = handles(&ellipsoid_spec(None), 8.0);
        for d in 0..3 {
            let handle = plain
                .iter()
                .find(|h| h.kind == HandleKind::Radius(d))
                .unwrap_or_else(|| panic!("a radius handle for axis {d}"));
            let mut expected = centre;
            expected[d] += radii[d];
            assert!(
                length(difference(handle.position, expected)) < 1e-9,
                "axis {d}: {:?} rather than {expected:?}",
                handle.position
            );
            assert!((handle.axis[d] - 1.0).abs() < 1e-9, "{:?}", handle.axis);
        }

        // Turned a quarter turn about z, the local x axis is world +y: the
        // handle of radius 0 is 6 mm along y, not along x.
        let turned = handles(&ellipsoid_spec(Some([0.0, 0.0, 90.0])), 8.0);
        let handle = turned
            .iter()
            .find(|h| h.kind == HandleKind::Radius(0))
            .expect("a radius handle");
        assert!(handle.axis[1] > 0.99 && handle.axis[0].abs() < 1e-9);
        assert!((handle.position[1] - (centre[1] + 6.0)).abs() < 1e-9);
        assert!((handle.position[0] - centre[0]).abs() < 1e-9);
        // And the handle of radius 1 is 3 mm along -x.
        let handle = turned
            .iter()
            .find(|h| h.kind == HandleKind::Radius(1))
            .expect("a radius handle");
        assert!((handle.position[0] - (centre[0] - 3.0)).abs() < 1e-9);
    }

    /// The dimension drags all land on the increment; an ellipsoid's radius is
    /// one of them, and it is the radius under the pointer that lands.
    #[test]
    fn a_snapped_ellipsoid_radius_drag_lands_that_radius_on_the_increment() {
        let spec = ShapeSpec::Ellipsoid {
            center: [0.0; 3],
            radii: [2.3, 5.0, 7.0],
            rotation_deg: None,
        };
        let handle = handles(&spec, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Radius(0))
            .expect("a radius handle");
        let start = ray_from(
            [handle.position[0], handle.position[1], 100.0],
            handle.position,
        );
        let mut drag = Drag::start(handle, spec.clone(), &start, [0.0, 0.0, -1.0]).expect("a drag");
        let target = [handle.position[0] + 1.4, handle.position[1], 100.0];
        let moved = ray_from(target, [target[0], target[1], handle.position[2]]);

        let free = drag.shape_at(&moved, Snap::OFF).expect("a shape");
        assert!((radius_of(&free, 0).unwrap() - 3.7).abs() < 1e-9);
        let snapped = drag.shape_at(&moved, Snap::default()).expect("a shape");
        assert!((radius_of(&snapped, 0).unwrap() - 4.0).abs() < 1e-9);
        // The other two are exactly what they were: one drag, one radius.
        let ShapeSpec::Ellipsoid { center, radii, .. } = snapped else {
            panic!("an ellipsoid");
        };
        assert_eq!([radii[1], radii[2]], [5.0, 7.0]);
        assert_eq!(center, [0.0; 3], "a resize never moves a shape");
    }

    /// A turned ellipsoid is turned exactly as a box is: the arc drives the
    /// `rotation_deg` component it is drawn for, and nothing else moves.
    #[test]
    fn an_ellipsoid_turns_by_recording_the_angle_a_box_would_record() {
        let start = [30.0, 0.0, 0.0];
        let spec = ellipsoid_spec(Some(start));
        for component in 0..3 {
            let axis = euler_axis(start, component);
            let turned = turn(&spec, component, axis, 15.0, Snap::OFF);
            let ShapeSpec::Ellipsoid {
                center,
                radii,
                rotation_deg,
            } = turned
            else {
                panic!("an ellipsoid");
            };
            let mut expected = start;
            expected[component] += 15.0;
            assert_eq!(rotation_deg, Some(expected));
            assert_eq!(center, [1.0, 2.0, 3.0], "a turn never moves a shape");
            assert_eq!(radii, [6.0, 3.0, 2.0], "nor resizes it");
        }
        // Snapping is of the resulting angle, as a box's is.
        let ShapeSpec::Ellipsoid { rotation_deg, .. } = turn(
            &ellipsoid_spec(None),
            2,
            [0.0, 0.0, 1.0],
            40.0,
            Snap::default(),
        ) else {
            panic!("an ellipsoid");
        };
        assert!((rotation_deg.expect("a rotation")[2] - 45.0).abs() < 1e-9);
    }

    /// A turned box wears its gizmo the way it is turned: the arrows point
    /// along its own edges and the face handles sit on its own faces.
    #[test]
    fn a_rotated_box_carries_its_handles_in_its_own_frame() {
        let turned = turned_box(90.0);
        let set = handles(&turned, 8.0);
        let centre = anchor(&turned);
        assert_eq!(centre, [5.0, 5.0, 5.0]);

        // A quarter turn about z takes the local x axis onto world +y.
        let x_arrow = set
            .iter()
            .find(|h| h.kind == HandleKind::Translate(0))
            .expect("an x arrow");
        assert!(x_arrow.axis[1] > 0.99 && x_arrow.axis[0].abs() < 1e-9);
        assert!((x_arrow.position[1] - (centre[1] + 8.0)).abs() < 1e-9);

        // So the +x face handle - the middle of the face at local max x - sits
        // on the +y side of the box.
        let face = set
            .iter()
            .find(|h| h.kind == HandleKind::Face(0, true))
            .expect("a face handle");
        assert!(
            (face.position[1] - 10.0).abs() < 1e-9,
            "{:?}",
            face.position
        );
        assert!((face.position[0] - 5.0).abs() < 1e-9);

        // And every corner handle sits on a corner of the box as it is turned.
        let corners = Shape::box_corners([0.0; 3], [10.0; 3], [0.0, 0.0, 90.0]);
        for handle in set.iter() {
            let HandleKind::Corner(_) = handle.kind else {
                continue;
            };
            assert!(
                corners
                    .iter()
                    .any(|c| length(difference(*c, handle.position)) < 1e-9),
                "{:?} is not a corner of the turned box",
                handle.position
            );
        }
    }

    /// Rotating a rotated box's face outward has to leave the opposite face
    /// where the user can see it is - which is a different sum from the axis
    /// aligned one, because the centre the box turns about moves too.
    #[test]
    fn resizing_a_rotated_box_keeps_its_opposite_face_where_it_was() {
        let turned = turned_box(90.0);
        let before = Shape::box_corners([0.0; 3], [10.0; 3], [0.0, 0.0, 90.0]);
        let grown = move_face(&turned, 0, true, 6.0);
        let ShapeSpec::Box {
            min,
            max,
            rotation_deg,
        } = grown
        else {
            panic!("a box");
        };
        assert_eq!(
            rotation_deg,
            Some([0.0, 0.0, 90.0]),
            "a resize is not a turn"
        );
        assert!(
            (max[0] - min[0] - 16.0).abs() < 1e-9,
            "the box grew by 6 mm"
        );
        assert!((max[1] - min[1] - 10.0).abs() < 1e-9, "on one axis only");
        // The four corners on the local -x face are exactly where they were.
        let after = Shape::box_corners(min, max, [0.0, 0.0, 90.0]);
        for k in 0..8 {
            if k & 1 != 0 {
                continue;
            }
            assert!(
                length(difference(before[k], after[k])) < 1e-9,
                "corner {k} moved from {:?} to {:?}",
                before[k],
                after[k]
            );
        }
    }

    #[test]
    fn a_snapped_drag_lands_on_the_increment_and_a_free_one_does_not() {
        let shape = ShapeSpec::Box {
            min: [0.3, 0.0, 0.0],
            max: [10.3, 10.0, 10.0],
            rotation_deg: None,
        };
        let handle = handles(&shape, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Translate(0))
            .expect("an x arrow");
        let start = ray_from(
            [handle.position[0], handle.position[1], 100.0],
            handle.position,
        );
        let mut drag =
            Drag::start(handle, shape.clone(), &start, [0.0, 0.0, -1.0]).expect("a drag");
        let to = |x: f64| {
            let target = [handle.position[0] + x, handle.position[1], 100.0];
            ray_from(target, [target[0], target[1], handle.position[2]])
        };

        // The centre started at 5.3 and the pointer covered 7.4 mm, so the
        // free drag lands on 12.7 and the snapped one on the whole millimetre.
        let free = drag.shape_at(&to(7.4), Snap::OFF).expect("a shape");
        assert!((anchor(&free)[0] - 12.7).abs() < 1e-9);
        let snapped = drag.shape_at(&to(7.4), Snap::default()).expect("a shape");
        assert!(
            (anchor(&snapped)[0] - 13.0).abs() < 1e-9,
            "landed on {}",
            anchor(&snapped)[0]
        );
        // Only the axis being dragged snaps: the other two keep the offsets the
        // shape had.
        assert!((anchor(&snapped)[1] - 5.0).abs() < 1e-12);
        let (min, max) = box_of(&snapped);
        assert!(
            (max[0] - min[0] - 10.0).abs() < 1e-12,
            "a move is not a resize"
        );

        // A coarser increment lands further out, and an increment of zero is
        // the free drag again.
        let coarse = Snap {
            millimetres: 5.0,
            degrees: 0.0,
        };
        assert!(
            (anchor(&drag.shape_at(&to(7.4), coarse).expect("a shape"))[0] - 15.0).abs() < 1e-9
        );
        let off = Snap {
            millimetres: 0.0,
            degrees: 0.0,
        };
        assert!((anchor(&drag.shape_at(&to(7.4), off).expect("a shape"))[0] - 12.7).abs() < 1e-9);
    }

    /// A dragged region offered surfaces lands flush on one, in preference to
    /// the millimetre grid, and only within reach of it.
    #[test]
    fn a_drag_lands_flush_on_a_surface_in_preference_to_the_increment() {
        // A 4 mm cube of a region, and a pad whose top face is at z = 20.3 -
        // deliberately off the grid, so that landing on it and landing on the
        // increment are different answers.
        let region = ShapeSpec::Box {
            min: [0.0, 0.0, 0.0],
            max: [4.0, 4.0, 4.0],
            rotation_deg: None,
        };
        let mut surfaces = Surfaces::default();
        surfaces.push_bounds(
            &crate::geometry::Aabb {
                min: [-10.0, -10.0, 0.0],
                max: [10.0, 10.0, 20.3],
            },
            crate::viewer::editor::snap::SurfaceKind::Keepin(0),
        );

        let handle = handles(&region, 4.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Translate(2))
            .expect("a z arrow");
        let start = ray_from([100.0, 0.0, handle.position[2]], handle.position);
        let mut drag =
            Drag::start(handle, region.clone(), &start, [-1.0, 0.0, 0.0]).expect("a drag");
        let to = |z: f64| {
            let target = [
                handle.position[0],
                handle.position[1],
                handle.position[2] + z,
            ];
            ray_from([100.0, 0.0, target[2]], target)
        };

        // Dragged up by 19.2 mm the region's underside is at 19.2, which is
        // 1.1 mm under the pad: within reach, so it lands *on* it.
        let placed = drag
            .placed_at(&to(19.2), Snap::default(), &surfaces)
            .expect("a shape");
        let (min, max) = box_of(&placed.shape);
        assert!(
            (min[2] - 20.3).abs() < 1e-9,
            "the region landed at {} rather than on the face at 20.3",
            min[2]
        );
        assert!((max[2] - 24.3).abs() < 1e-9, "a landing is not a resize");
        let flush = placed.flush.expect("the landing must be reported");
        assert!((flush.plane.coordinate - 20.3).abs() < 1e-12);
        assert_eq!(
            flush.plane.what,
            crate::viewer::editor::snap::SurfaceKind::Keepin(0)
        );

        // Out of reach of the face, the millimetre grid has it as usual.
        let placed = drag
            .placed_at(&to(10.4), Snap::default(), &surfaces)
            .expect("a shape");
        let (min, _) = box_of(&placed.shape);
        assert!((min[2] - 10.0).abs() < 1e-9, "landed at {}", min[2]);
        assert!(placed.flush.is_none());

        // The bypass switches off both: the region goes exactly where the
        // pointer put it, face or no face.
        let placed = drag
            .placed_at(&to(19.2), Snap::OFF, &surfaces)
            .expect("a shape");
        let (min, _) = box_of(&placed.shape);
        assert!((min[2] - 19.2).abs() < 1e-9, "landed at {}", min[2]);
        assert!(placed.flush.is_none());

        // And with no surfaces offered at all - which is what a keepout or a
        // keepin drag gets - it is the grid again.
        let placed = drag
            .placed_at(&to(19.2), Snap::default(), &Surfaces::default())
            .expect("a shape");
        let (min, _) = box_of(&placed.shape);
        assert!((min[2] - 19.0).abs() < 1e-9, "landed at {}", min[2]);
        assert!(placed.flush.is_none());
    }

    /// The candidate planes are axis aligned, so a drag that runs across all
    /// three of them at once is not a placement against any of them.
    #[test]
    fn an_oblique_drag_lands_on_the_increment_rather_than_on_a_plane() {
        let region = ShapeSpec::Box {
            min: [0.0, 0.0, 0.0],
            max: [4.0, 4.0, 4.0],
            rotation_deg: Some([0.0, 45.0, 0.0]),
        };
        let mut surfaces = Surfaces::default();
        surfaces.push_bounds(
            &crate::geometry::Aabb {
                min: [-10.0, -10.0, 0.0],
                max: [10.0, 10.0, 20.3],
            },
            crate::viewer::editor::snap::SurfaceKind::Keepin(0),
        );
        // A 45 degree turn about y leaves the local z axis exactly halfway
        // between world x and world z, which is the alignment floor.
        let handle = handles(&region, 4.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Translate(2))
            .expect("a z arrow");
        assert!(
            (handle.axis[2] - 0.5_f64.sqrt()).abs() < 1e-9,
            "this test no longer describes the geometry it was written for"
        );
        let start = ray_from([0.0, 100.0, handle.position[2]], handle.position);
        let mut drag = Drag::start(handle, region, &start, [0.0, -1.0, 0.0]).expect("a drag");
        let target = sum(handle.position, scale(handle.axis, 19.0));
        let moved = ray_from([0.0, 100.0, target[2]], target);
        // Above the floor it still places; the floor is where it stops, and
        // either way the drag produces a shape rather than nothing.
        let placed = drag
            .placed_at(&moved, Snap::default(), &surfaces)
            .expect("a shape");
        assert!(placed.shape != *drag.original());
    }

    #[test]
    fn a_snapped_resize_lands_the_dimension_on_the_increment() {
        let shape = ShapeSpec::Box {
            min: [0.0, 0.0, 0.0],
            max: [10.3, 10.0, 10.0],
            rotation_deg: None,
        };
        let handle = handles(&shape, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Face(0, true))
            .expect("a +x face handle");
        let start = ray_from(
            [handle.position[0], handle.position[1], 100.0],
            handle.position,
        );
        let mut drag =
            Drag::start(handle, shape.clone(), &start, [0.0, 0.0, -1.0]).expect("a drag");
        let target = [handle.position[0] + 3.4, handle.position[1], 100.0];
        let moved = ray_from(target, [target[0], target[1], handle.position[2]]);

        // 10.3 wide, dragged 3.4 further: 13.7 free, 14 snapped - and it is the
        // *dimension* that lands on the increment, which is what the callout
        // shows.
        let free = drag.shape_at(&moved, Snap::OFF).expect("a shape");
        let (min, max) = box_of(&free);
        assert!((max[0] - min[0] - 13.7).abs() < 1e-9);
        let snapped = drag.shape_at(&moved, Snap::default()).expect("a shape");
        let (min, max) = box_of(&snapped);
        assert!(
            (max[0] - min[0] - 14.0).abs() < 1e-9,
            "{:?}",
            max[0] - min[0]
        );
        assert!(min[0].abs() < 1e-12, "the opposite face stayed put");
        assert!((max[1] - min[1] - 10.0).abs() < 1e-12, "one axis only");
    }

    #[test]
    fn a_snapped_radius_drag_lands_the_radius_on_the_increment() {
        let sphere = ShapeSpec::Sphere {
            center: [0.0; 3],
            radius: 2.3,
        };
        let handle = handles(&sphere, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Radius(0))
            .expect("a radius handle");
        let start = ray_from(
            [handle.position[0], handle.position[1], 100.0],
            handle.position,
        );
        let mut drag = Drag::start(handle, sphere, &start, [0.0, 0.0, -1.0]).expect("a drag");
        let target = [handle.position[0] + 1.4, handle.position[1], 100.0];
        let moved = ray_from(target, [target[0], target[1], handle.position[2]]);
        let free = drag.shape_at(&moved, Snap::OFF).expect("a shape");
        assert!((radius_of(&free, 0).unwrap() - 3.7).abs() < 1e-9);
        let snapped = drag.shape_at(&moved, Snap::default()).expect("a shape");
        assert!((radius_of(&snapped, 0).unwrap() - 4.0).abs() < 1e-9);
    }

    /// The arc drag: the angle comes from where the ray crosses the arc's own
    /// plane, the sweep is measured from the grab, and the result lands on the
    /// rotation increment.
    #[test]
    fn a_rotation_arc_turns_by_the_angle_the_pointer_swept() {
        let shape = unit_box();
        let handle = handles(&shape, 8.0)
            .into_iter()
            .find(|h| h.kind == HandleKind::Rotate(2))
            .expect("a z arc");
        let centre = anchor(&shape);
        // The z arc lies in the plane z = 5, in the basis that plane is
        // measured in; a ray straight down onto a point of it crosses it
        // there, so the angle the drag reads is that point's own.
        let (u, v) = tessellate::basis([0.0, 0.0, 1.0]);
        let at = |degrees: f64| {
            let (sin, cos) = degrees.to_radians().sin_cos();
            let point = sum(centre, scale(sum(scale(u, cos), scale(v, sin)), 10.0));
            ray_from([point[0], point[1], 100.0], point)
        };
        let start_angle = arc_angle(&at(0.0), centre, [0.0, 0.0, 1.0]).expect("an angle");
        assert!(start_angle.abs() < 1e-9);
        let quarter = arc_angle(&at(90.0), centre, [0.0, 0.0, 1.0]).expect("an angle");
        assert!((quarter - 90.0).abs() < 1e-9);

        let mut drag =
            Drag::start(handle, shape.clone(), &at(0.0), [0.0, 0.0, -1.0]).expect("a drag");
        // Swept 40 degrees, snapped to the nearest 22.5 step.
        let snapped = drag.shape_at(&at(40.0), Snap::default()).expect("a shape");
        let ShapeSpec::Box { rotation_deg, .. } = snapped else {
            panic!("a box");
        };
        let rotation = rotation_deg.expect("a rotation");
        assert!((rotation[2] - 45.0).abs() < 1e-9, "{rotation:?}");
        assert_eq!([rotation[0], rotation[1]], [0.0, 0.0], "one axis only");
        // Alt: the same drag, free.
        let free = drag.shape_at(&at(40.0), Snap::OFF).expect("a shape");
        let ShapeSpec::Box { rotation_deg, .. } = free else {
            panic!("a box");
        };
        assert!((rotation_deg.expect("a rotation")[2] - 40.0).abs() < 1e-9);
        // A box does not change size when it is turned.
        assert_eq!(box_of(&snapped), box_of(&shape));
    }

    /// A box already turned is turned further about the axis its own Euler
    /// component drives, so the number the gizmo produces is the number the
    /// file holds.
    #[test]
    fn a_rotation_arc_drives_the_component_it_is_drawn_for() {
        let start = [30.0, 0.0, 0.0];
        let spec = ShapeSpec::Box {
            min: [0.0; 3],
            max: [10.0, 4.0, 2.0],
            rotation_deg: Some(start),
        };
        for component in 0..3 {
            let axis = euler_axis(start, component);
            let turned = turn(&spec, component, axis, 15.0, Snap::OFF);
            let ShapeSpec::Box { rotation_deg, .. } = turned else {
                panic!("a box");
            };
            let mut expected = start;
            expected[component] += 15.0;
            assert_eq!(rotation_deg, Some(expected));

            // And that really is a rotation of 15 degrees about that axis: the
            // corners of the new box are the old ones turned about it.
            let before = Shape::box_corners([0.0; 3], [10.0, 4.0, 2.0], start);
            let after = Shape::box_corners([0.0; 3], [10.0, 4.0, 2.0], expected);
            let centre = anchor(&spec);
            let matrix = axis_rotation_matrix(axis, 15.0);
            for k in 0..8 {
                let mapped = sum(centre, rotate(&matrix, difference(before[k], centre)));
                assert!(
                    length(difference(mapped, after[k])) < 1e-9,
                    "component {component}, corner {k}: {mapped:?} against {:?}",
                    after[k]
                );
            }
        }
    }

    #[test]
    fn a_cylinder_turns_its_caps_about_its_centre_and_keeps_its_length() {
        let cylinder = ShapeSpec::Cylinder {
            p1: [0.0, 0.0, 0.0],
            p2: [10.0, 0.0, 0.0],
            radius: 2.0,
        };
        let before = anchor(&cylinder);
        for degrees in [22.5, 45.0, 90.0, -30.0] {
            let turned = turn(&cylinder, 2, [0.0, 0.0, 1.0], degrees, Snap::OFF);
            let ShapeSpec::Cylinder { p1, p2, radius } = turned else {
                panic!("a cylinder");
            };
            assert!(
                (length(difference(p2, p1)) - 10.0).abs() < 1e-9,
                "{degrees}"
            );
            assert!(
                length(difference(anchor(&turned), before)) < 1e-9,
                "{degrees}: the centre moved"
            );
            assert!((radius - 2.0).abs() < 1e-12);
        }
        // A quarter turn about z takes an x aligned cylinder onto y.
        let ShapeSpec::Cylinder { p1, p2, .. } =
            turn(&cylinder, 2, [0.0, 0.0, 1.0], 90.0, Snap::OFF)
        else {
            panic!("a cylinder");
        };
        assert!((p1[1] + 5.0).abs() < 1e-9 && (p2[1] - 5.0).abs() < 1e-9);
        assert!((p1[0] - 5.0).abs() < 1e-9 && (p2[0] - 5.0).abs() < 1e-9);
        // Snapping applies to the sweep, because a cylinder has no angle of its
        // own to land on a multiple.
        let ShapeSpec::Cylinder { p1, p2, .. } =
            turn(&cylinder, 2, [0.0, 0.0, 1.0], 40.0, Snap::default())
        else {
            panic!("a cylinder");
        };
        let swept = (p2[1] - p1[1]).atan2(p2[0] - p1[0]).to_degrees();
        assert!((swept - 45.0).abs() < 1e-9, "swept {swept}");
        // A sphere has nothing to turn.
        let sphere = ShapeSpec::Sphere {
            center: [1.0, 2.0, 3.0],
            radius: 4.0,
        };
        assert_eq!(
            turn(&sphere, 2, [0.0, 0.0, 1.0], 45.0, Snap::OFF),
            sphere.clone()
        );
    }

    /// The rings sit outside everything else, so from many angles one of them
    /// is in front of an arrow. A press aimed at the arrow must still get it.
    #[test]
    fn a_rotation_arc_never_takes_a_press_aimed_at_an_arrow() {
        let shape = unit_box();
        let set = handles(&shape, 8.0);
        let arrow = set
            .iter()
            .find(|h| h.kind == HandleKind::Translate(0))
            .expect("an x arrow");
        let ray = ray_from(
            [arrow.position[0], arrow.position[1], 100.0],
            arrow.position,
        );
        // The y ring really is in the way: on depth alone it would win.
        let arc = set
            .iter()
            .find(|h| h.kind == HandleKind::Rotate(1))
            .expect("a y arc");
        let arc_t = crate::viewer::editor::pick::hit_shape(&ray, &arc.volume);
        let arrow_t = crate::viewer::editor::pick::hit_shape(&ray, &arrow.volume);
        assert!(
            arc_t.is_some_and(|arc| arrow_t.is_some_and(|arrow| arc < arrow)),
            "this test no longer describes the geometry it was written for"
        );
        assert_eq!(
            grab(&ray, &set).map(|h| h.kind),
            Some(HandleKind::Translate(0))
        );
        // And an arc is still grabbable where nothing else is.
        let onto = ray_from(
            [arc.position[0], arc.position[1] + 100.0, arc.position[2]],
            arc.position,
        );
        assert_eq!(
            grab(&onto, &set).map(|h| h.kind),
            Some(HandleKind::Rotate(1))
        );
    }

    /// The user's report: *"the resize boxes and the move arrows are overlapping
    /// causing the boxes to not be able to be dragged or selected"*.
    ///
    /// On anything near cubic a face handle sits *inside* the shaft of the arrow
    /// along its own axis, at the same depth to the bit, so nothing but the tier
    /// rule can hand the press to the smaller target.
    #[test]
    fn a_resize_handle_inside_an_arrow_takes_the_press_from_it() {
        let shape = unit_box();
        let set = handles(&shape, 8.0);
        let face = set
            .iter()
            .find(|h| h.kind == HandleKind::Face(0, true))
            .expect("a +x face handle");
        let arrow = set
            .iter()
            .find(|h| h.kind == HandleKind::Translate(0))
            .expect("an x arrow");
        // The face handle really is buried in the arrow: on the axis, short of
        // the tip.
        assert!(
            crate::viewer::editor::pick::hit_shape(
                &ray_from([face.position[0], face.position[1], 100.0], face.position),
                &arrow.volume
            )
            .is_some(),
            "this test no longer describes the geometry it was written for"
        );

        let ray = ray_from([face.position[0], face.position[1], 100.0], face.position);
        let face_t = crate::viewer::editor::pick::hit_shape(&ray, &face.volume).expect("a hit");
        let arrow_t = crate::viewer::editor::pick::hit_shape(&ray, &arrow.volume).expect("a hit");
        assert!(
            arrow_t <= face_t + 1e-9,
            "depth alone would already favour the face handle, so this proves nothing: \
             {arrow_t} against {face_t}"
        );
        assert_eq!(
            grab(&ray, &set).map(|h| h.kind),
            Some(HandleKind::Face(0, true))
        );

        // And the arrow is still the arrow everywhere no resize handle is: past
        // the face handle, along the rest of the shaft.
        let along = sum(anchor(&shape), scale(arrow.axis, 7.0));
        let ray = ray_from([along[0], along[1], 100.0], along);
        assert_eq!(
            grab(&ray, &set).map(|h| h.kind),
            Some(HandleKind::Translate(0))
        );

        // A resize handle the ray merely passes *behind* is a different thing
        // from one buried in the shaft, and takes nothing: this ray is aimed
        // squarely at the arrow's tip and goes on to a corner of the box.
        let corner = set
            .iter()
            .find(|h| h.kind == HandleKind::Corner([true, true, true]))
            .expect("a corner handle");
        let direction = difference(corner.position, arrow.position);
        let ray = Ray {
            origin: difference(arrow.position, scale(direction, 100.0 / length(direction))),
            direction: scale(direction, 1.0 / length(direction)),
        };
        let corner_t =
            crate::viewer::editor::pick::hit_shape(&ray, &corner.volume).expect("a corner hit");
        let arrow_t =
            crate::viewer::editor::pick::hit_shape(&ray, &arrow.volume).expect("an arrow hit");
        assert!(
            arrow_t < corner_t,
            "this test no longer describes the geometry it was written for"
        );
        assert!(
            !arrow.volume.contains(corner.position),
            "the corner is off the shaft, which is why the arrow keeps the press"
        );
        assert_eq!(
            grab(&ray, &set).map(|h| h.kind),
            Some(HandleKind::Translate(0)),
            "a corner handle behind the arrow's tip took a press aimed at the tip"
        );
    }

    /// The same for the radius handles, which is where this was first noticed:
    /// a sphere's single handle stands at its radius, and an ellipsoid's short
    /// semi-axes stand well inside the arrows.
    #[test]
    fn a_radius_handle_inside_an_arrow_is_reachable() {
        let sphere = ShapeSpec::Sphere {
            center: [0.0; 3],
            radius: 2.0,
        };
        let set = handles(&sphere, 8.0);
        let handle = set
            .iter()
            .find(|h| h.kind == HandleKind::Radius(0))
            .expect("a radius handle");
        let ray = ray_from(
            [handle.position[0], handle.position[1], 100.0],
            handle.position,
        );
        assert_eq!(
            grab(&ray, &set).map(|h| h.kind),
            Some(HandleKind::Radius(0)),
            "a radius shorter than the gizmo is unreachable"
        );

        // The anisotropic case: radii of 6, 3 and 2 against a gizmo of 8, so the
        // two short semi-axes are inside their own arrows.
        let set = handles(&ellipsoid_spec(None), 8.0);
        for component in [1, 2] {
            let handle = set
                .iter()
                .find(|h| h.kind == HandleKind::Radius(component))
                .expect("a radius handle");
            let ray = ray_from(
                [
                    handle.position[0],
                    handle.position[1] + 100.0,
                    handle.position[2],
                ],
                handle.position,
            );
            assert_eq!(
                grab(&ray, &set).map(|h| h.kind),
                Some(HandleKind::Radius(component)),
                "radius {component} is unreachable"
            );
        }
    }

    /// The tiers, as a rule rather than as three examples of one: every cube
    /// beats an arrow, an arrow beats an arc, and within a tier it is the
    /// nearest hit that wins.
    #[test]
    fn the_grab_tiers_run_cubes_then_arrows_then_arcs() {
        for kind in [
            HandleKind::Face(0, true),
            HandleKind::Corner([true; 3]),
            HandleKind::Endpoint(0),
            HandleKind::Radius(1),
            HandleKind::TranslateFree,
        ] {
            assert_eq!(kind.grab_rank(), 0, "{kind:?} is drawn as a cube");
        }
        assert_eq!(HandleKind::Translate(2).grab_rank(), 1);
        assert_eq!(HandleKind::Rotate(0).grab_rank(), 2);
        assert!(
            HandleKind::Face(0, true).grab_rank() < HandleKind::Translate(0).grab_rank()
                && HandleKind::Translate(0).grab_rank() < HandleKind::Rotate(0).grab_rank()
        );
        // Only the arcs are outside everything, which is what keeps them from
        // taking a press aimed at what they ring.
        assert!(HandleKind::Rotate(1).rings_the_gizmo());
        for kind in [
            HandleKind::Translate(0),
            HandleKind::TranslateFree,
            HandleKind::Face(2, false),
        ] {
            assert!(!kind.rings_the_gizmo(), "{kind:?}");
        }
    }

    /// The centre cube is drawn as one of the same markers the resize handles
    /// are, and it stands where all three shafts begin, so it needs the rule
    /// they got: a press on it is a press on it.
    #[test]
    fn the_centre_cube_takes_the_press_from_the_arrows_that_start_on_it() {
        // A sphere, whose only resize handle is out at its radius: on a box the
        // face cube nearest the camera is in front of the centre one, and being
        // the nearer cube of the same tier it is the one a press there means.
        let shape = ShapeSpec::Sphere {
            center: [0.0; 3],
            radius: 2.0,
        };
        let set = handles(&shape, 8.0);
        let centre = set
            .iter()
            .find(|h| h.kind == HandleKind::TranslateFree)
            .expect("a centre handle");
        for d in 0..3 {
            let arrow = set
                .iter()
                .find(|h| h.kind == HandleKind::Translate(d))
                .expect("an arrow");
            assert!(
                arrow.volume.contains(centre.position),
                "axis {d}: the arrow no longer starts on the centre handle"
            );
        }
        let ray = ray_from(
            [centre.position[0], centre.position[1], 100.0],
            centre.position,
        );
        assert_eq!(
            grab(&ray, &set).map(|h| h.kind),
            Some(HandleKind::TranslateFree)
        );
    }

    #[test]
    fn the_gizmo_is_drawable_and_colour_codes_its_axes() {
        let shape = unit_box();
        let length = gizmo_length(shape_radius(&shape.to_shape("test").unwrap()), 100.0);
        let handles = handles(&shape, length);
        let layer = mesh(&handles, length, anchor(&shape), None);
        assert!(layer.triangles() > 0);
        assert!(layer.vertices.iter().all(|v| {
            v.position
                .iter()
                .chain(v.normal.iter())
                .all(|value| value.is_finite())
        }));
        let seen = colors(&layer);
        for axis in constants::VIEW_COLOR_GIZMO_AXES {
            assert!(seen.contains(&axis), "an axis arrow lost its colour");
        }
        assert!(seen.contains(&constants::VIEW_COLOR_GIZMO_HANDLE));
    }

    /// The handle under the pointer says so by brightening, and only that one
    /// handle does.
    #[test]
    fn the_hovered_handle_is_drawn_brighter_than_the_rest() {
        let shape = unit_box();
        let length = gizmo_length(shape_radius(&shape.to_shape("test").unwrap()), 100.0);
        let handles = handles(&shape, length);
        let hovered = HandleKind::Translate(0);
        let plain = mesh(&handles, length, anchor(&shape), None);
        let lit = mesh(&handles, length, anchor(&shape), Some(hovered));
        assert_eq!(plain.triangles(), lit.triangles(), "only the colour moves");

        let base = constants::VIEW_COLOR_GIZMO_AXES[0];
        let bright = brightened(base);
        assert!(colors(&plain).contains(&base));
        assert!(!colors(&plain).contains(&bright));
        assert!(
            colors(&lit).contains(&bright),
            "the hovered arrow is not lit"
        );
        assert!(
            !colors(&lit).contains(&base),
            "and is no longer its plain self"
        );
        // The other two arrows are untouched.
        for axis in [1, 2] {
            assert!(colors(&lit).contains(&constants::VIEW_COLOR_GIZMO_AXES[axis]));
        }
        // Brightening moves towards white and leaves the alpha alone.
        for channel in 0..3 {
            assert!(bright[channel] >= base[channel] && bright[channel] <= 1.0);
        }
        assert_eq!(bright[3], base[3]);
    }

    /// The hover outline is thinner than the selection shell for every shape
    /// kind, which is what keeps the two apart when both are on screen.
    #[test]
    fn the_hover_outline_is_thinner_than_the_selection_shell() {
        for shape in [
            Shape::axis_aligned_box([0.0; 3], [10.0, 4.0, 4.0]),
            Shape::Sphere {
                center: [1.0, 2.0, 3.0],
                radius: 4.0,
            },
            Shape::Cylinder {
                p1: [0.0; 3],
                p2: [0.0, 0.0, 12.0],
                radius: 2.0,
            },
            Shape::Ellipsoid {
                center: [1.0, 2.0, 3.0],
                radii: [6.0, 3.0, 2.0],
                rotation_deg: [0.0, 0.0, 30.0],
            },
            Shape::Tube {
                p1: [0.0; 3],
                p2: [20.0, 0.0, 0.0],
                bend: None,
                radius: 2.0,
            },
            Shape::Tube {
                p1: [0.0; 3],
                p2: [20.0, 0.0, 0.0],
                bend: Some([10.0, 6.0, 0.0]),
                radius: 2.0,
            },
            Shape::Cone {
                p1: [0.0; 3],
                p2: [0.0, 0.0, 10.0],
                radius1: 4.0,
                radius2: 1.0,
            },
            Shape::Cone {
                p1: [0.0; 3],
                p2: [0.0, 0.0, 10.0],
                radius1: 4.0,
                radius2: 0.0,
            },
            Shape::Triangle {
                a: [0.0; 3],
                b: [12.0, 0.0, 0.0],
                c: [0.0, 9.0, 0.0],
                thickness: 3.0,
            },
        ] {
            let length = gizmo_length(shape_radius(&shape), 100.0);
            let hover = hover_margin(&shape, length);
            let selection = selection_margin(&shape, length);
            assert!(hover > 0.0, "{shape:?}: the outline has to be visible");
            assert!(
                hover < selection,
                "{shape:?}: hover margin {hover} is not thinner than {selection}"
            );
        }
    }

    #[test]
    fn the_gizmo_scales_with_the_object_and_never_vanishes() {
        let big = gizmo_length(50.0, 100.0);
        let small = gizmo_length(0.0001, 100.0);
        assert!((big - 50.0 * constants::VIEW_EDIT_GIZMO_LENGTH_RADIUS_FRACTION).abs() < 1e-12);
        assert!(
            (small - 100.0 * constants::VIEW_EDIT_GIZMO_MIN_SCENE_FRACTION).abs() < 1e-12,
            "a tiny object still needs grabbable handles, got {small}"
        );
        // Even with no scene at all the gizmo has a size.
        assert!(gizmo_length(0.0, 0.0) >= constants::VIEW_EDIT_MIN_EXTENT_MM);
    }

    #[test]
    fn the_selection_shell_encloses_the_shape_it_highlights() {
        for shape in [
            Shape::axis_aligned_box([0.0; 3], [10.0, 4.0, 4.0]),
            Shape::Box {
                min: [0.0; 3],
                max: [10.0, 4.0, 4.0],
                rotation_deg: [0.0, 0.0, 30.0],
            },
            Shape::Sphere {
                center: [1.0, 2.0, 3.0],
                radius: 4.0,
            },
            Shape::Cylinder {
                p1: [0.0; 3],
                p2: [0.0, 0.0, 8.0],
                radius: 2.0,
            },
            Shape::Ellipsoid {
                center: [1.0, 2.0, 3.0],
                radii: [6.0, 3.0, 2.0],
                rotation_deg: [0.0; 3],
            },
            Shape::Ellipsoid {
                center: [1.0, 2.0, 3.0],
                radii: [6.0, 3.0, 2.0],
                rotation_deg: [15.0, -35.0, 62.0],
            },
            Shape::Tube {
                p1: [0.0; 3],
                p2: [20.0, 0.0, 0.0],
                bend: None,
                radius: 2.0,
            },
            Shape::Tube {
                p1: [0.0; 3],
                p2: [20.0, 0.0, 0.0],
                bend: Some([10.0, 6.0, 0.0]),
                radius: 2.0,
            },
            Shape::Cone {
                p1: [0.0; 3],
                p2: [0.0, 0.0, 8.0],
                radius1: 4.0,
                radius2: 1.0,
            },
            Shape::Cone {
                p1: [1.0, 2.0, 3.0],
                p2: [7.0, 6.0, 9.0],
                radius1: 1.0,
                radius2: 0.0,
            },
            Shape::Triangle {
                a: [0.0; 3],
                b: [12.0, 0.0, 0.0],
                c: [0.0, 9.0, 0.0],
                thickness: 3.0,
            },
            Shape::Triangle {
                a: [1.0, 2.0, 3.0],
                b: [7.0, 3.0, 9.0],
                c: [2.0, 9.0, 5.0],
                thickness: 2.0,
            },
        ] {
            let margin = selection_margin(&shape, 8.0);
            assert!(margin > 0.0);
            let shell = inflated(&shape, margin);
            let inner = shape.bounds();
            let outer = shell.bounds();
            for d in 0..3 {
                assert!(
                    outer.min[d] < inner.min[d] && outer.max[d] > inner.max[d],
                    "axis {d}: {outer:?} does not enclose {inner:?}"
                );
            }
            // The shell has to clear the shape *itself* rather than its box - a
            // slanted wall and a turned face are exactly where a margin taken
            // as a box would cut through the shape it is drawn around - and it
            // has to clear it by a *bounded* fraction of the margin rather than
            // by anything positive at all. Only the second catches a
            // construction drifting towards zero, which is the way a shell
            // fails: it starts z-fighting long before it goes inside out.
            //
            // A whole margin is what an inflation that is a plain offset gives
            // - the box, the sphere, the cylinder, the tube, the prism - and
            // the floor is what the two that are not can hold. A cone's sloped
            // wall reads two thirds of a margin at its wide rim, for the reason
            // [`inflated`] gives.
            let floor = match shape {
                // Not a property of the shell at all: an ellipsoid's field is a
                // 1-Lipschitz *bound* on the distance rather than the distance
                // (see [`crate::geometry::Shape::Ellipsoid`]), so what can be
                // read here is that bound. It under-states by exactly the ratio
                // of the smallest grown radius to the largest, which is
                // (2 + m) / (6 + m) = 0.35 on this fixture; the geometry itself
                // clears a whole margin at every axis.
                Shape::Ellipsoid { .. } => 0.25,
                _ => 0.5,
            };
            for vertex in &crate::viewer::tessellate::shape(&shape).vertices {
                let clearance = -shell.signed_distance(*vertex);
                assert!(
                    clearance >= floor * margin,
                    "{shape:?}: the surface point {vertex:?} stands {clearance} inside its own \
                     shell, under the {floor} of a {margin} mm margin this pins"
                );
            }
        }
    }

    /// Corners of a box shape, for the assertions above.
    fn box_of(spec: &ShapeSpec) -> (Vec3, Vec3) {
        match *spec {
            ShapeSpec::Box { min, max, .. } => (min, max),
            _ => panic!("expected a box"),
        }
    }
}
