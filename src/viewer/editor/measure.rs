//! In-viewport measurement: what a drag is producing, in numbers.
//!
//! Every drag has one value that is the point of it - how far the object moved,
//! how wide it now is, how far it has been turned - and this module is that
//! value: derived from the shape the drag started on and the shape it is at
//! now, drawn in the viewport as a dimension line, shown as a floating number
//! box beside it, and applied back to the shape when the user types an exact
//! one into that box instead.
//!
//! Nothing here draws or parses. The state machine below is pure - a drag
//! raises a callout, a release starts its linger, a click turns it into a text
//! field, a value commits or is cancelled - so all of it is exercised without a
//! window; [`crate::viewer::editor::ui`] is the only thing that knows about
//! egui, and it only ever asks this module what to show.

use std::time::{Duration, Instant};

use crate::config::ShapeSpec;
use crate::constants;
use crate::geometry::{
    Aabb, Vec3, cross, difference, inner, length, normalize_degrees, scale, sum,
};
use crate::mesh::Mesh;
use crate::viewer::editor::gizmo::{self, Handle, HandleKind};
use crate::viewer::editor::snap::Snap;
use crate::viewer::editor::state::Selection;
use crate::viewer::scene::{LayerMesh, Shading};
use crate::viewer::tessellate;

/// What a callout measures, and therefore what typing a number into it sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasureKind {
    /// Distance the object has moved along the drag's own axis, signed. This
    /// is the number Tinkercad shows while something is being moved: how far,
    /// not where.
    Offset,
    /// One edge length of a box, measured along the box's own frame.
    Extent,
    /// A distance between two points of a shape that the handle being dragged
    /// moves one of: a cylinder's, a tube's or a cone's two ends, the middle of
    /// a tube and the point its bend has been pulled out to, or a triangle's
    /// vertex and the middle of the edge it faces.
    Length,
    /// A radius.
    Radius,
    /// The thickness of a triangular prism, measured through it from face to
    /// face.
    Thickness,
    /// Degrees about the axis the arc turns.
    Angle,
}

impl MeasureKind {
    /// Unit the value is in, for the callout's own label.
    pub fn unit(self) -> &'static str {
        match self {
            MeasureKind::Angle => "deg",
            _ => "mm",
        }
    }
}

/// One live measurement of a drag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measure {
    /// What is being measured.
    pub kind: MeasureKind,
    /// Its value, in millimetres or degrees.
    pub value: f64,
    /// Unit direction the measurement runs along; the rotation axis for an
    /// angle.
    pub axis: Vec3,
    /// Which of a box's three axes, which radius of an ellipsoid, or which
    /// rotation component, this is.
    pub component: usize,
    /// One end of the dimension line.
    pub from: Vec3,
    /// The other end.
    pub to: Vec3,
}

impl Measure {
    /// World point the number box is anchored to: the middle of what it
    /// measures.
    pub fn anchor(&self) -> Vec3 {
        scale(sum(self.from, self.to), 0.5)
    }

    /// The number as the callout shows it.
    pub fn label(&self) -> String {
        format!(
            "{:.*} {}",
            constants::VIEW_EDIT_CALLOUT_DECIMALS,
            self.value,
            self.kind.unit()
        )
    }

    /// The number as it is offered for editing: no unit, so what comes back is
    /// what was typed.
    pub fn field_text(&self) -> String {
        format!("{:.*}", constants::VIEW_EDIT_CALLOUT_DECIMALS, self.value)
    }
}

/// The measurement a drag from `original` to `current` is producing, or `None`
/// when this handle measures nothing (a shape that is not the kind the handle
/// belongs to).
pub fn measure(handle: &Handle, original: &ShapeSpec, current: &ShapeSpec) -> Option<Measure> {
    let start = gizmo::anchor(original);
    let now = gizmo::anchor(current);
    match handle.kind {
        HandleKind::Translate(component) => {
            let axis = handle.axis;
            let value = inner(difference(now, start), axis);
            Some(Measure {
                kind: MeasureKind::Offset,
                value,
                axis,
                component,
                from: start,
                to: sum(start, scale(axis, value)),
            })
        }
        HandleKind::TranslateFree => {
            let moved = difference(now, start);
            let value = length(moved);
            let axis = if value > 0.0 {
                scale(moved, 1.0 / value)
            } else {
                [0.0, 0.0, 1.0]
            };
            Some(Measure {
                kind: MeasureKind::Offset,
                value,
                axis,
                component: 0,
                from: start,
                to: now,
            })
        }
        HandleKind::Face(component, positive) => extent_measure(current, component, positive),
        HandleKind::Corner(mask) => {
            // Three dimensions change at once; the callout is the one that
            // changed most, which is the one the drag was mostly about.
            let before = gizmo::box_extent(original)?;
            let after = gizmo::box_extent(current)?;
            let component = (0..3)
                .max_by(|a, b| {
                    (after[*a] - before[*a])
                        .abs()
                        .total_cmp(&(after[*b] - before[*b]).abs())
                })
                .unwrap_or(0);
            extent_measure(current, component, mask[component])
        }
        // What an end handle is setting is how far apart the two ends are. On a
        // bent tube that is the chord it spans rather than the length of its
        // curve, which is what its end handles move it along.
        HandleKind::Endpoint(_) => {
            let (p1, p2) = gizmo::endpoints(current)?;
            Some(span_measure(p1, p2))
        }
        // And what the bend handle is setting is how far the middle of the tube
        // has been pulled off the line between its ends: zero is straight, and
        // the number grows with the curve.
        //
        // Measured to where the **handle** is rather than to the bend the shape
        // kept, because for one frame in the middle of the gesture those are
        // different points: a bend dragged back onto that line is stored as no
        // bend at all, and the number would jump to zero from wherever along
        // the tube the pointer really is. The handle is what the editor keeps
        // under the pointer ([`gizmo::Placed::handle_at`]), and away from a
        // drag it is the bend or the middle, so this is the same number
        // everywhere else.
        HandleKind::Bend => {
            // A tube, and nothing else, has a middle to bend.
            gizmo::bend_of(current)?;
            let (p1, p2) = gizmo::endpoints(current)?;
            Some(span_measure(scale(sum(p1, p2), 0.5), handle.position))
        }
        // What a vertex handle is setting is how far that corner stands off the
        // edge between the other two: the endpoint's own callout, with the far
        // end replaced by the middle of the edge the vertex faces. It falls to
        // nothing exactly as the triangle goes flat, which is the one thing the
        // drag will not let it do.
        HandleKind::Vertex(vertex) => {
            let points = gizmo::vertices(current)?;
            Some(span_measure(
                opposite_midpoint(&points, vertex)?,
                *points.get(vertex)?,
            ))
        }
        // Measured through the prism from face to face, which is the number the
        // file holds - not the half of it the handle itself moves.
        HandleKind::Thickness => {
            let value = gizmo::thickness_of(current)?;
            let axis = handle.axis;
            let centre = gizmo::anchor(current);
            let half = scale(axis, 0.5 * value);
            Some(Measure {
                kind: MeasureKind::Thickness,
                value,
                axis,
                component: 0,
                from: difference(centre, half),
                to: sum(centre, half),
            })
        }
        HandleKind::Radius(component) => {
            let value = gizmo::radius_of(current, component)?;
            let axis = handle.axis;
            // Where the radius is measured *from*, which is the centre for
            // every kind built about one, the middle of the curve for a tube -
            // so the line is drawn across the tube rather than across the chord
            // it spans - and the cap this radius belongs to for a cone.
            let origin = gizmo::radius_origin(current, component);
            Some(Measure {
                kind: MeasureKind::Radius,
                value,
                axis,
                // Which radius is being dragged, so an ellipsoid's callout
                // names the one under the pointer and a number typed into it
                // sets that one.
                component,
                from: origin,
                to: sum(origin, scale(axis, value)),
            })
        }
        HandleKind::Rotate(component) => {
            let value = match *current {
                // A box and an ellipsoid record the angle they are at, so that
                // is what is shown.
                ShapeSpec::Box { rotation_deg, .. } | ShapeSpec::Ellipsoid { rotation_deg, .. } => {
                    normalize_degrees(rotation_deg.unwrap_or_default()[component])
                }
                // A cylinder, a tube and a cone have nowhere to record one, so
                // the sweep is recovered from how far the line between their
                // ends has come round.
                ShapeSpec::Cylinder { .. } | ShapeSpec::Tube { .. } | ShapeSpec::Cone { .. } => {
                    swept_between(original, current, handle.axis)?
                }
                // Neither carries a rotation gizmo at all; see
                // [`gizmo::is_rotatable`].
                ShapeSpec::Sphere { .. } | ShapeSpec::Triangle { .. } => return None,
            };
            let (u, _) = tessellate::basis(handle.axis);
            let radius = length(difference(handle.position, now)).max(f64::MIN_POSITIVE);
            Some(Measure {
                kind: MeasureKind::Angle,
                value,
                axis: handle.axis,
                component,
                from: now,
                to: sum(now, scale(u, radius)),
            })
        }
    }
}

/// The dimension of a box along one axis of its own frame, measured between the
/// centres of the two faces that bound it.
fn extent_measure(current: &ShapeSpec, component: usize, positive: bool) -> Option<Measure> {
    let extent = gizmo::box_extent(current)?[component];
    let centre = gizmo::anchor(current);
    let axis = gizmo::local_axes(current)[component];
    let half = scale(axis, 0.5 * extent);
    Some(Measure {
        kind: MeasureKind::Extent,
        value: extent,
        axis,
        // Which face was dragged decides which one a typed value moves, so it
        // is carried rather than re-derived.
        component: component + if positive { 0 } else { 3 },
        from: difference(centre, half),
        to: sum(centre, half),
    })
}

/// The distance between two points of a shape, as the callout measures it: the
/// number, the direction it runs in, and the line it is drawn along.
fn span_measure(from: Vec3, to: Vec3) -> Measure {
    let along = difference(to, from);
    let value = length(along);
    Measure {
        kind: MeasureKind::Length,
        value,
        axis: if value > 0.0 {
            scale(along, 1.0 / value)
        } else {
            [0.0, 0.0, 1.0]
        },
        component: 0,
        from,
        to,
    }
}

/// The middle of the edge a triangle's vertex faces: the far end of the median
/// through it, which is what its callout measures along.
fn opposite_midpoint(points: &[Vec3; 3], vertex: usize) -> Option<Vec3> {
    let (first, second) = match vertex {
        0 => (points[1], points[2]),
        1 => (points[2], points[0]),
        2 => (points[0], points[1]),
        _ => return None,
    };
    Some(scale(sum(first, second), 0.5))
}

/// Signed angle, in degrees, about `axis` between the lines joining the ends of
/// two shapes that have ends - a cylinder, a tube or a cone.
fn swept_between(original: &ShapeSpec, current: &ShapeSpec, axis: Vec3) -> Option<f64> {
    let (a1, a2) = gizmo::endpoints(original)?;
    let (b1, b2) = gizmo::endpoints(current)?;
    let before = difference(a2, a1);
    let after = difference(b2, b1);
    let sine = inner(cross(before, after), axis);
    let cosine = inner(before, after);
    if sine == 0.0 && cosine == 0.0 {
        return None;
    }
    Some(sine.atan2(cosine).to_degrees())
}

/// The component of a box's own frame an [`MeasureKind::Extent`] measurement
/// belongs to, and whether the face that was dragged is the upper one.
fn extent_face(component: usize) -> (usize, bool) {
    (component % 3, component < 3)
}

/// The shape that measurement would have if its value were `value`.
///
/// This is the typed-number path, and it goes through the very operations a
/// drag goes through, so a value typed into a callout lands exactly where
/// dragging to it would have.
pub fn applied(handle: &Handle, original: &ShapeSpec, at: &Measure, value: f64) -> ShapeSpec {
    match at.kind {
        MeasureKind::Offset => gizmo::translate(original, scale(at.axis, value)),
        MeasureKind::Extent => {
            let (component, positive) = extent_face(at.component);
            let Some(extent) = gizmo::box_extent(original) else {
                return original.clone();
            };
            let sign = if positive { 1.0 } else { -1.0 };
            gizmo::move_face(
                original,
                component,
                positive,
                sign * (value - extent[component]),
            )
        }
        // Every length drag moves one point away from another along the line it
        // is already on, so all of them are the same sum: how much further out
        // the typed number puts it.
        MeasureKind::Length => {
            let (moving, fixed) = match handle.kind {
                HandleKind::Vertex(vertex) => {
                    let Some(points) = gizmo::vertices(original) else {
                        return original.clone();
                    };
                    match (points.get(vertex), opposite_midpoint(&points, vertex)) {
                        (Some(corner), Some(middle)) => (*corner, middle),
                        _ => return original.clone(),
                    }
                }
                _ => {
                    let Some((p1, p2)) = gizmo::endpoints(original) else {
                        return original.clone();
                    };
                    match handle.kind {
                        HandleKind::Endpoint(0) => (p1, p2),
                        HandleKind::Endpoint(_) => (p2, p1),
                        HandleKind::Bend => match gizmo::bend_of(original) {
                            Some(bend) => (bend, scale(sum(p1, p2), 0.5)),
                            None => return original.clone(),
                        },
                        _ => return original.clone(),
                    }
                }
            };
            let along = difference(moving, fixed);
            let current = length(along);
            if current <= 0.0 {
                // A bend that has not been pulled out yet, or an end on top of
                // the other one: there is no direction to put the number on.
                return original.clone();
            }
            let step = scale(along, (value - current) / current);
            match handle.kind {
                HandleKind::Bend => gizmo::move_bend(original, step),
                HandleKind::Endpoint(end) => gizmo::move_endpoint(original, end, step),
                HandleKind::Vertex(vertex) => gizmo::move_vertex(original, vertex, step),
                _ => original.clone(),
            }
        }
        MeasureKind::Radius => {
            let Some(radius) = gizmo::radius_of(original, at.component) else {
                return original.clone();
            };
            gizmo::resize_radius(original, at.component, value - radius)
        }
        MeasureKind::Thickness => {
            let Some(thickness) = gizmo::thickness_of(original) else {
                return original.clone();
            };
            gizmo::resize_thickness(original, value - thickness)
        }
        MeasureKind::Angle => {
            let swept = match *original {
                // The box's and the ellipsoid's callouts show where they are,
                // so a typed number is where they should be.
                ShapeSpec::Box { rotation_deg, .. } | ShapeSpec::Ellipsoid { rotation_deg, .. } => {
                    value - rotation_deg.unwrap_or_default()[at.component]
                }
                // The cylinder's shows how far it has come, so a typed number
                // is how far it should come from where the drag began.
                _ => value,
            };
            gizmo::turn(original, at.component, at.axis, swept, Snap::OFF)
        }
    }
}

/// Where a callout is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// The drag that raised it is still held.
    Dragging,
    /// The drag is over and the number is readable for a while yet.
    Lingering,
    /// It has been clicked and is a text field.
    Typing,
}

/// A dimension callout: the number a drag is producing, and the box that number
/// can be typed into instead.
#[derive(Debug, Clone)]
pub struct Callout {
    /// The object it measures.
    pub selection: Selection,
    /// The handle whose drag raised it.
    pub handle: Handle,
    /// The shape that drag started on, which a typed value is applied to.
    pub original: ShapeSpec,
    /// The live measurement.
    pub at: Measure,
    /// When the drag was released; `None` while it is still held.
    released: Option<Instant>,
    /// The text being typed, once the callout has been clicked.
    editing: Option<String>,
}

impl Callout {
    /// Raise a callout for a drag that has just started moving.
    pub fn new(selection: Selection, handle: Handle, original: ShapeSpec, at: Measure) -> Callout {
        Callout {
            selection,
            handle,
            original,
            at,
            released: None,
            editing: None,
        }
    }

    /// Follow the drag.
    pub fn update(&mut self, at: Measure) {
        self.at = at;
        // A drag that was released and grabbed again is dragging once more.
        self.released = None;
    }

    /// The drag let go; the number stays up for a while.
    pub fn release(&mut self, now: Instant) {
        if self.released.is_none() {
            self.released = Some(now);
        }
    }

    /// Where this callout is in its life.
    pub fn phase(&self) -> Phase {
        match (&self.editing, self.released) {
            (Some(_), _) => Phase::Typing,
            (None, Some(_)) => Phase::Lingering,
            (None, None) => Phase::Dragging,
        }
    }

    /// True once the linger has run out and nothing is being typed into it.
    ///
    /// A callout being typed into never expires: the user is in the middle of
    /// telling it a number.
    pub fn is_expired(&self, now: Instant, linger: Duration) -> bool {
        match (self.editing.is_some(), self.released) {
            (true, _) => false,
            (false, Some(at)) => now.saturating_duration_since(at) >= linger,
            (false, None) => false,
        }
    }

    /// Turn the callout into a text field holding the value it is showing.
    pub fn begin_typing(&mut self) {
        if self.editing.is_none() {
            self.editing = Some(self.at.field_text());
        }
    }

    /// The text being typed, for the field to edit in place.
    pub fn text_mut(&mut self) -> Option<&mut String> {
        self.editing.as_mut()
    }

    /// The text being typed, if any.
    pub fn text(&self) -> Option<&str> {
        self.editing.as_deref()
    }

    /// Stop typing, keeping the value the drag left.
    pub fn cancel_typing(&mut self) {
        self.editing = None;
    }

    /// The shape a typed value would produce, or `None` when what was typed is
    /// not a number the configuration could hold.
    pub fn committed(&self, text: &str) -> Option<ShapeSpec> {
        let value = parse_number(text)?;
        Some(applied(&self.handle, &self.original, &self.at, value))
    }
}

/// The same number parsing the properties panel does: whitespace anywhere is
/// ignored, a typographic minus is a hyphen, and anything that is not finite is
/// not a number a configuration may hold.
pub fn parse_number(text: &str) -> Option<f64> {
    let cleaned: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| if c == '\u{2212}' { '-' } else { c })
        .collect();
    let value: f64 = cleaned.parse().ok()?;
    value.is_finite().then_some(value)
}

/// The dimension line of a measurement: a thin tube with an arrow head at each
/// end, running between the two points the number is measured across.
///
/// An angle has no line - what it measures is a turn, and the arc handle is
/// already drawn round the axis it turns about - so it comes back empty.
pub fn dimension_mesh(at: &Measure, gizmo_length: f64) -> Mesh {
    if at.kind == MeasureKind::Angle {
        return Mesh::default();
    }
    tessellate::dimension_line(
        at.from,
        at.to,
        gizmo_length * constants::VIEW_EDIT_MEASURE_TUBE_FRACTION,
        gizmo_length * constants::VIEW_EDIT_MEASURE_HEAD_LENGTH_FRACTION,
        gizmo_length * constants::VIEW_EDIT_MEASURE_HEAD_RADIUS_FRACTION,
    )
}

/// The gap between the bottom of `bounds` and the floor of the domain, and the
/// line that shows it.
///
/// Drawn only for a drag that is mostly vertical, which is when that distance
/// is the thing being set; a sideways drag does not change it.
pub fn floor_measure(bounds: &Aabb, floor_z: f64, drag_axis: Vec3, free: bool) -> Option<Measure> {
    if !free && drag_axis[2].abs() < constants::VIEW_EDIT_FLOOR_INDICATOR_MIN_Z {
        return None;
    }
    if bounds.is_empty() || !floor_z.is_finite() {
        return None;
    }
    let x = 0.5 * (bounds.min[0] + bounds.max[0]);
    let y = 0.5 * (bounds.min[1] + bounds.max[1]);
    Some(Measure {
        kind: MeasureKind::Offset,
        value: bounds.min[2] - floor_z,
        axis: [0.0, 0.0, 1.0],
        component: 2,
        from: [x, y, floor_z],
        to: [x, y, bounds.min[2]],
    })
}

/// The whole measurement overlay of one drag: its dimension line, and the
/// floor-distance line when there is one.
pub fn overlay(at: &Measure, floor: Option<&Measure>, gizmo_length: f64) -> Option<LayerMesh> {
    let mut out = LayerMesh::default();
    add(
        &mut out,
        &dimension_mesh(at, gizmo_length),
        constants::VIEW_COLOR_MEASURE,
    );
    if let Some(floor) = floor {
        add(
            &mut out,
            &dimension_mesh(floor, gizmo_length),
            constants::VIEW_COLOR_MEASURE_FLOOR,
        );
    }
    (!out.is_empty()).then_some(out)
}

/// Append a mesh to a layer in one colour.
fn add(out: &mut LayerMesh, mesh: &Mesh, color: [f32; 4]) {
    let part = LayerMesh::from_mesh(mesh, color, Shading::Rounded);
    if part.is_empty() {
        return;
    }
    out.vertices.extend(part.vertices.iter().copied());
    out.bounds = out.bounds.union(&part.bounds);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::editor::gizmo::handles;

    fn box_spec() -> ShapeSpec {
        ShapeSpec::Box {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 4.0, 6.0],
            rotation_deg: None,
        }
    }

    fn handle_of(spec: &ShapeSpec, kind: HandleKind) -> Handle {
        handles(spec, 8.0)
            .into_iter()
            .find(|h| h.kind == kind)
            .unwrap_or_else(|| panic!("no {kind:?} handle"))
    }

    #[test]
    fn a_translation_measures_the_distance_it_covered_along_its_own_axis() {
        let spec = box_spec();
        let handle = handle_of(&spec, HandleKind::Translate(0));
        let moved = gizmo::translate(&spec, [7.0, 0.0, 0.0]);
        let at = measure(&handle, &spec, &moved).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Offset);
        assert!((at.value - 7.0).abs() < 1e-12);
        assert_eq!(at.label(), "7.00 mm");
        // Signed: back the other way is a negative number, not a distance.
        let back = gizmo::translate(&spec, [-2.5, 0.0, 0.0]);
        let at = measure(&handle, &spec, &back).expect("a measurement");
        assert!((at.value + 2.5).abs() < 1e-12);
        // And typing the number in lands the shape where dragging to it would.
        let typed = applied(&handle, &spec, &at, 7.0);
        assert_eq!(gizmo::anchor(&typed), gizmo::anchor(&moved));
    }

    #[test]
    fn a_face_drag_measures_the_dimension_it_is_changing() {
        let spec = box_spec();
        let handle = handle_of(&spec, HandleKind::Face(0, true));
        let wider = gizmo::move_face(&spec, 0, true, 3.0);
        let at = measure(&handle, &spec, &wider).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Extent);
        assert!((at.value - 13.0).abs() < 1e-12);
        // The line runs across the dimension it names.
        assert!((length(difference(at.to, at.from)) - 13.0).abs() < 1e-9);
        // Typing an exact width moves the same face and keeps the other one.
        let typed = applied(&handle, &spec, &at, 20.0);
        let ShapeSpec::Box { min, max, .. } = typed else {
            panic!("a box");
        };
        assert!((max[0] - 20.0).abs() < 1e-12 && min[0].abs() < 1e-12);

        // The negative face is measured and typed the same way, from the other
        // side: the opposite face is what stays put.
        let handle = handle_of(&spec, HandleKind::Face(0, false));
        let at = measure(&handle, &spec, &spec).expect("a measurement");
        let typed = applied(&handle, &spec, &at, 20.0);
        let ShapeSpec::Box { min, max, .. } = typed else {
            panic!("a box");
        };
        assert!((min[0] + 10.0).abs() < 1e-12 && (max[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn a_corner_drag_measures_the_dimension_that_moved_most() {
        let spec = box_spec();
        let mask = [true, true, true];
        let handle = handle_of(&spec, HandleKind::Corner(mask));
        let dragged = gizmo::move_corner(&spec, mask, [1.0, 9.0, 2.0]);
        let at = measure(&handle, &spec, &dragged).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Extent);
        assert_eq!(extent_face(at.component), (1, true));
        assert!((at.value - 13.0).abs() < 1e-12);
    }

    #[test]
    fn a_radius_and_a_cylinder_length_measure_themselves() {
        let cylinder = ShapeSpec::Cylinder {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            radius: 2.0,
        };
        let handle = handle_of(&cylinder, HandleKind::Radius(0));
        let at = measure(&handle, &cylinder, &cylinder).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Radius);
        assert!((at.value - 2.0).abs() < 1e-12);
        let typed = applied(&handle, &cylinder, &at, 5.0);
        assert!((gizmo::radius_of(&typed, 0).unwrap() - 5.0).abs() < 1e-12);

        let handle = handle_of(&cylinder, HandleKind::Endpoint(1));
        let moved = gizmo::move_endpoint(&cylinder, 1, [0.0, 0.0, 4.0]);
        let at = measure(&handle, &cylinder, &moved).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Length);
        assert!((at.value - 14.0).abs() < 1e-12);
        // Typed back to 10 along the same direction, the cylinder is what it
        // was.
        let typed = applied(&handle, &cylinder, &at, 10.0);
        let ShapeSpec::Cylinder { p1, p2, .. } = typed else {
            panic!("a cylinder");
        };
        assert_eq!(p1, [0.0, 0.0, 0.0]);
        assert!((p2[2] - 10.0).abs() < 1e-9);
    }

    /// A tube's end handles measure the distance between its two ends, bent or
    /// not, and its middle handle measures how far the bend has been pulled off
    /// the line between them - which is the number that says how curved it is.
    #[test]
    fn a_tube_measures_the_span_of_its_ends_and_the_pull_of_its_bend() {
        let tube = ShapeSpec::Tube {
            p1: [0.0, 0.0, 0.0],
            p2: [20.0, 0.0, 0.0],
            bend: None,
            radius: 2.0,
        };
        let handle = handle_of(&tube, HandleKind::Endpoint(1));
        let moved = gizmo::move_endpoint(&tube, 1, [4.0, 0.0, 0.0]);
        let at = measure(&handle, &tube, &moved).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Length);
        assert!((at.value - 24.0).abs() < 1e-12);
        // Typed back to 20 along the same direction, the tube is what it was.
        let typed = applied(&handle, &tube, &at, 20.0);
        assert_eq!(gizmo::endpoints(&typed), gizmo::endpoints(&tube));

        // The radius reads as a cylinder's does.
        let handle = handle_of(&tube, HandleKind::Radius(0));
        let at = measure(&handle, &tube, &tube).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Radius);
        assert!((at.value - 2.0).abs() < 1e-12);
        assert_eq!(at.from, [10.0, 0.0, 0.0], "measured from the middle");
        let typed = applied(&handle, &tube, &at, 5.0);
        assert!((gizmo::radius_of(&typed, 0).unwrap() - 5.0).abs() < 1e-12);

        // On a bent tube it is measured from the middle of the curve, so the
        // line is drawn across the tube rather than across the chord: the
        // dimension line starts on the surface's own centre line.
        let hook = ShapeSpec::Tube {
            p1: [0.0, 0.0, 0.0],
            p2: [2.0, 1.0, 0.0],
            bend: Some([-6.0, 8.0, 0.0]),
            radius: 1.0,
        };
        let handle = handle_of(&hook, HandleKind::Radius(0));
        let at = measure(&handle, &hook, &hook).expect("a measurement");
        let shape = hook.to_shape("test").expect("a shape");
        assert!(
            shape.signed_distance(at.from).abs() > 0.0 && shape.contains(at.from),
            "the line starts off the tube at {:?}",
            at.from
        );
        assert!(
            (length(difference(at.to, at.from)) - 1.0).abs() < 1e-9,
            "the line runs across the radius it names"
        );
        assert!(
            !shape.contains([1.0, 0.5, 0.0]),
            "the middle of the chord, which is nowhere near this tube"
        );

        // The bend handle: zero while the tube is straight, and the distance it
        // has been pulled once it is not.
        let handle = handle_of(&tube, HandleKind::Bend);
        let at = measure(&handle, &tube, &tube).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Length);
        assert!(at.value.abs() < 1e-12, "a straight tube is pulled 0 mm");
        // With no direction to put it on, a typed number leaves it straight.
        assert_eq!(applied(&handle, &tube, &at, 5.0), tube);

        // Pulled out: measured to where the handle now is, which is the bend
        // itself while the shape has one. (The editor keeps the handle under
        // the pointer; the test builds it from the shape, which is the same
        // point.)
        let bent = gizmo::move_bend(&tube, [0.0, 6.0, 0.0]);
        let handle = handle_of(&bent, HandleKind::Bend);
        let at = measure(&handle, &tube, &bent).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Length);
        assert!((at.value - 6.0).abs() < 1e-12, "{}", at.value);
        assert_eq!(at.from, [10.0, 0.0, 0.0], "measured from the middle");
        assert_eq!(at.to, [10.0, 6.0, 0.0]);
        assert_eq!(at.label(), "6.00 mm");
        // Typed, the bend goes exactly that far out along the way it was
        // pulled - and typing zero straightens the tube.
        let typed = applied(&handle, &bent, &at, 15.0);
        let ShapeSpec::Tube { bend, .. } = typed else {
            panic!("a tube");
        };
        assert!(length(difference(bend.expect("a bend"), [10.0, 15.0, 0.0])) < 1e-9);
        let ShapeSpec::Tube { bend, .. } = applied(&handle, &bent, &at, 0.0) else {
            panic!("a tube");
        };
        assert_eq!(bend, None, "typing zero is straightening it");
    }

    /// A cone measures each of its radii from the cap that radius belongs to,
    /// and a number typed into either callout sets that one alone.
    #[test]
    fn a_cone_measures_each_radius_from_its_own_cap() {
        let cone = ShapeSpec::Cone {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            radius1: 4.0,
            radius2: 2.0,
        };
        for (component, origin, radius) in [(0, [0.0, 0.0, 0.0], 4.0), (1, [0.0, 0.0, 10.0], 2.0)] {
            let handle = handle_of(&cone, HandleKind::Radius(component));
            let at = measure(&handle, &cone, &cone).expect("a measurement");
            assert_eq!(at.kind, MeasureKind::Radius);
            assert!((at.value - radius).abs() < 1e-12);
            assert_eq!(at.from, origin, "measured from its own cap");
            assert!((length(difference(at.to, at.from)) - radius).abs() < 1e-9);
            // Typed, it sets that radius and leaves the other where it is.
            let typed = applied(&handle, &cone, &at, 6.0);
            assert!((gizmo::radius_of(&typed, component).unwrap() - 6.0).abs() < 1e-12);
            let other = 1 - component;
            assert_eq!(
                gizmo::radius_of(&typed, other),
                gizmo::radius_of(&cone, other)
            );
            // And zero is the apex on the narrow end, the floor on the wide.
            let pointed = applied(&handle, &cone, &at, 0.0);
            let expected = if component == 0 {
                constants::VIEW_EDIT_MIN_EXTENT_MM
            } else {
                0.0
            };
            assert_eq!(gizmo::radius_of(&pointed, component), Some(expected));
        }
        // Its end handles measure the span between its caps, as a cylinder's
        // do, and its rotation callout reads the sweep the same way.
        let handle = handle_of(&cone, HandleKind::Endpoint(1));
        let moved = gizmo::move_endpoint(&cone, 1, [0.0, 0.0, 4.0]);
        let at = measure(&handle, &cone, &moved).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Length);
        assert!((at.value - 14.0).abs() < 1e-12);
        // About x, because a cone that runs along z is not moved by a turn
        // about z - there would be no sweep to recover.
        let handle = handle_of(&cone, HandleKind::Rotate(0));
        let turned = gizmo::turn(&cone, 0, handle.axis, 30.0, Snap::OFF);
        let at = measure(&handle, &cone, &turned).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Angle);
        assert!((at.value - 30.0).abs() < 1e-9, "{}", at.value);
    }

    /// A triangle's corner callout is the endpoint's, with the far end
    /// replaced by the middle of the edge that corner faces; its thickness
    /// callout is the number the file holds rather than the half a drag moves.
    #[test]
    fn a_triangle_measures_its_corners_and_its_thickness() {
        let triangle = ShapeSpec::Triangle {
            a: [0.0, 0.0, 0.0],
            b: [12.0, 0.0, 0.0],
            c: [6.0, 8.0, 0.0],
            thickness: 3.0,
        };
        // The third corner stands 8 mm off the middle of the edge between the
        // other two, which is the median this measures.
        let handle = handle_of(&triangle, HandleKind::Vertex(2));
        let at = measure(&handle, &triangle, &triangle).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Length);
        assert!((at.value - 8.0).abs() < 1e-12);
        assert_eq!(at.from, [6.0, 0.0, 0.0], "the middle of the opposite edge");
        assert_eq!(at.to, [6.0, 8.0, 0.0]);
        assert_eq!(at.label(), "8.00 mm");
        // Typed, the corner goes exactly that far out along the line it is on,
        // and the other two stay where they are.
        let typed = applied(&handle, &triangle, &at, 20.0);
        let corners = gizmo::vertices(&typed).expect("a triangle");
        assert!(length(difference(corners[2], [6.0, 20.0, 0.0])) < 1e-9);
        assert_eq!(corners[0], [0.0, 0.0, 0.0]);
        assert_eq!(corners[1], [12.0, 0.0, 0.0]);
        // Dragging the corner moves the number with it.
        let dragged = gizmo::move_vertex(&triangle, 2, [0.0, 4.0, 0.0]);
        let at = measure(&handle, &triangle, &dragged).expect("a measurement");
        assert!((at.value - 12.0).abs() < 1e-12);

        // The thickness, measured through the prism from face to face.
        let handle = handle_of(&triangle, HandleKind::Thickness);
        let at = measure(&handle, &triangle, &triangle).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Thickness);
        assert_eq!(at.kind.unit(), "mm");
        assert!((at.value - 3.0).abs() < 1e-12);
        assert!((length(difference(at.to, at.from)) - 3.0).abs() < 1e-9);
        let typed = applied(&handle, &triangle, &at, 7.0);
        assert_eq!(gizmo::thickness_of(&typed), Some(7.0));
        assert_eq!(
            gizmo::vertices(&typed),
            gizmo::vertices(&triangle),
            "a thickness never moves a corner"
        );
        // And a triangle has no rotation callout at all, because it has no
        // rotation gizmo to raise one.
        assert!(!gizmo::is_rotatable(&triangle));
    }

    /// The number follows the handle rather than the shape's stored bend, so it
    /// stays continuous through the moment a drag straightens the tube: a bend
    /// pulled back onto the line off-centre is *stored* as none at all, and the
    /// number must not fall to zero from wherever along the tube the pointer
    /// really is.
    #[test]
    fn a_bend_callout_follows_the_handle_through_the_straightening() {
        let tube = ShapeSpec::Tube {
            p1: [0.0, 0.0, 0.0],
            p2: [20.0, 0.0, 0.0],
            bend: None,
            radius: 2.0,
        };
        let mut handle = handle_of(&tube, HandleKind::Bend);
        // Five millimetres short of the middle, coming down onto the line: the
        // number closes on 5, which is where the handle is, rather than on 0.
        let mut readings = Vec::new();
        for y in [2.0, 1.0, 0.25, 0.0] {
            handle.position = [5.0, y, 0.0];
            let current = if y > 0.0 {
                gizmo::move_bend(&tube, [-5.0, y, 0.0])
            } else {
                // The straightened shape, which is the tube it started as.
                tube.clone()
            };
            let at = measure(&handle, &tube, &current).expect("a measurement");
            readings.push(at.value);
            assert_eq!(at.to, handle.position, "the line ends at the handle");
        }
        for (a, b) in readings.iter().zip(readings.iter().skip(1)) {
            assert!(
                (a - b).abs() < 0.5,
                "the number jumped from {a} to {b} instead of interpolating"
            );
        }
        let last = *readings.last().expect("a reading");
        assert!(
            (last - 5.0).abs() < 1e-12,
            "straightened off-centre it read {last} rather than the 5 mm the handle is at"
        );
        // Zero is what it reads when the handle really is in the middle, which
        // is the only time it should.
        handle.position = [10.0, 0.0, 0.0];
        let at = measure(&handle, &tube, &tube).expect("a measurement");
        assert!(at.value.abs() < 1e-12);
    }

    /// The callout of an ellipsoid's radius drag names the radius that is
    /// being dragged, and a number typed into it sets that one.
    #[test]
    fn an_ellipsoid_radius_callout_measures_the_axis_it_was_dragged_on() {
        let spec = ShapeSpec::Ellipsoid {
            center: [1.0, 2.0, 3.0],
            radii: [6.0, 3.0, 2.0],
            rotation_deg: Some([0.0, 0.0, 90.0]),
        };
        for component in 0..3 {
            let handle = handle_of(&spec, HandleKind::Radius(component));
            let at = measure(&handle, &spec, &spec).expect("a measurement");
            assert_eq!(at.kind, MeasureKind::Radius);
            assert_eq!(at.component, component);
            let was = [6.0, 3.0, 2.0][component];
            assert!((at.value - was).abs() < 1e-12, "{component}: {}", at.value);
            // The dimension line runs along that semi-axis, from the centre.
            assert!((length(difference(at.to, at.from)) - was).abs() < 1e-9);
            // Typed, it is that radius that moves and only that one.
            let typed = applied(&handle, &spec, &at, 9.0);
            let ShapeSpec::Ellipsoid { radii, .. } = typed else {
                panic!("an ellipsoid");
            };
            for d in 0..3 {
                let expected = if d == component {
                    9.0
                } else {
                    [6.0, 3.0, 2.0][d]
                };
                assert!((radii[d] - expected).abs() < 1e-12, "{radii:?}");
            }
        }
    }

    #[test]
    fn an_angle_measures_where_a_box_is_and_how_far_a_cylinder_has_come() {
        let spec = box_spec();
        let handle = handle_of(&spec, HandleKind::Rotate(2));
        let turned = gizmo::turn(&spec, 2, handle.axis, 45.0, Snap::OFF);
        let at = measure(&handle, &spec, &turned).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Angle);
        assert!((at.value - 45.0).abs() < 1e-9);
        assert_eq!(at.label(), "45.00 deg");
        // Typed: an absolute angle for a box.
        let typed = applied(&handle, &spec, &at, 90.0);
        let ShapeSpec::Box { rotation_deg, .. } = typed else {
            panic!("a box");
        };
        assert!((rotation_deg.expect("a rotation")[2] - 90.0).abs() < 1e-9);

        // An ellipsoid records one exactly as a box does.
        let ellipsoid = ShapeSpec::Ellipsoid {
            center: [0.0; 3],
            radii: [6.0, 3.0, 2.0],
            rotation_deg: None,
        };
        let handle = handle_of(&ellipsoid, HandleKind::Rotate(2));
        let turned = gizmo::turn(&ellipsoid, 2, handle.axis, 45.0, Snap::OFF);
        let at = measure(&handle, &ellipsoid, &turned).expect("a measurement");
        assert_eq!(at.kind, MeasureKind::Angle);
        assert!((at.value - 45.0).abs() < 1e-9);
        let typed = applied(&handle, &ellipsoid, &at, 90.0);
        let ShapeSpec::Ellipsoid { rotation_deg, .. } = typed else {
            panic!("an ellipsoid");
        };
        assert!((rotation_deg.expect("a rotation")[2] - 90.0).abs() < 1e-9);

        // A cylinder has no angle of its own, so what is shown is the sweep.
        let cylinder = ShapeSpec::Cylinder {
            p1: [-5.0, 0.0, 0.0],
            p2: [5.0, 0.0, 0.0],
            radius: 1.0,
        };
        let handle = handle_of(&cylinder, HandleKind::Rotate(2));
        let turned = gizmo::turn(&cylinder, 2, handle.axis, 30.0, Snap::OFF);
        let at = measure(&handle, &cylinder, &turned).expect("a measurement");
        assert!((at.value - 30.0).abs() < 1e-9, "swept {}", at.value);
        let typed = applied(&handle, &cylinder, &at, 90.0);
        let ShapeSpec::Cylinder { p1, p2, .. } = typed else {
            panic!("a cylinder");
        };
        // A quarter turn about z takes an x aligned cylinder onto y.
        assert!(p1[0].abs() < 1e-9 && (p2[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn the_floor_indicator_measures_the_gap_below_a_vertical_drag_only() {
        let bounds = Aabb {
            min: [0.0, 0.0, 12.0],
            max: [10.0, 10.0, 20.0],
        };
        let at = floor_measure(&bounds, 2.0, [0.0, 0.0, 1.0], false).expect("a floor measurement");
        assert!((at.value - 10.0).abs() < 1e-12);
        assert_eq!(at.from[2], 2.0);
        assert_eq!(at.to[2], 12.0);
        // A sideways drag is not setting the height, so it says nothing.
        assert!(floor_measure(&bounds, 2.0, [1.0, 0.0, 0.0], false).is_none());
        // A free drag can move it up, so it does.
        assert!(floor_measure(&bounds, 2.0, [1.0, 0.0, 0.0], true).is_some());
        assert!(floor_measure(&Aabb::empty(), 2.0, [0.0, 0.0, 1.0], true).is_none());
    }

    #[test]
    fn the_overlay_draws_a_line_for_a_measurement_and_nothing_for_an_angle() {
        let spec = box_spec();
        let handle = handle_of(&spec, HandleKind::Translate(0));
        let moved = gizmo::translate(&spec, [7.0, 0.0, 0.0]);
        let at = measure(&handle, &spec, &moved).expect("a measurement");
        let layer = overlay(&at, None, 8.0).expect("a line");
        assert!(layer.triangles() > 0);

        // A drag that has not moved has nothing to measure across.
        let still = measure(&handle, &spec, &spec).expect("a measurement");
        assert!(overlay(&still, None, 8.0).is_none());

        // An angle's picture is the arc handle itself.
        let handle = handle_of(&spec, HandleKind::Rotate(2));
        let turned = gizmo::turn(&spec, 2, handle.axis, 45.0, Snap::OFF);
        let angle = measure(&handle, &spec, &turned).expect("a measurement");
        assert!(overlay(&angle, None, 8.0).is_none());
        // But with a floor line beside it there is something to draw.
        let floor = floor_measure(
            &Aabb {
                min: [0.0, 0.0, 12.0],
                max: [10.0, 10.0, 20.0],
            },
            0.0,
            [0.0, 0.0, 1.0],
            false,
        )
        .expect("a floor measurement");
        assert!(overlay(&angle, Some(&floor), 8.0).is_some());
    }

    /// The whole life of a callout: raised by a drag, lingering after it,
    /// clicked into a field, typed into, committed - and cancelled instead.
    #[test]
    fn a_callout_lingers_then_takes_a_typed_value_or_lets_it_go() {
        let linger = Duration::from_secs_f64(constants::VIEW_EDIT_CALLOUT_LINGER_S);
        let spec = box_spec();
        let handle = handle_of(&spec, HandleKind::Translate(0));
        let moved = gizmo::translate(&spec, [7.0, 0.0, 0.0]);
        let at = measure(&handle, &spec, &moved).expect("a measurement");
        let mut callout = Callout::new(Selection::Keepout(0), handle, spec.clone(), at);

        let start = Instant::now();
        assert_eq!(callout.phase(), Phase::Dragging);
        assert!(
            !callout.is_expired(start + linger * 10, linger),
            "a drag that is still held never expires"
        );

        callout.release(start);
        assert_eq!(callout.phase(), Phase::Lingering);
        assert!(!callout.is_expired(start, linger));
        assert!(callout.is_expired(start + linger, linger));

        // Clicked: it becomes a field holding what it was showing.
        callout.begin_typing();
        assert_eq!(callout.phase(), Phase::Typing);
        assert_eq!(callout.text(), Some("7.00"));
        assert!(
            !callout.is_expired(start + linger * 10, linger),
            "a callout being typed into must not vanish under the pointer"
        );

        // A typed value is applied to the shape the drag started on, so it is
        // an absolute answer rather than a second drag on top of the first.
        *callout.text_mut().expect("a field") = "12.5".to_string();
        let committed = callout.committed("12.5").expect("a shape");
        assert_eq!(gizmo::anchor(&committed)[0], gizmo::anchor(&spec)[0] + 12.5);
        // What is not a number is refused, and the callout keeps what it had.
        for text in ["", "nan", "inf", "twelve"] {
            assert!(callout.committed(text).is_none(), "{text} was accepted");
        }

        // Cancelling puts it back to lingering and changes nothing.
        callout.cancel_typing();
        assert_eq!(callout.phase(), Phase::Lingering);
        assert!(callout.is_expired(start + linger, linger));
    }

    #[test]
    fn a_callout_that_is_dragged_again_stops_lingering() {
        let spec = box_spec();
        let handle = handle_of(&spec, HandleKind::Translate(0));
        let at = measure(&handle, &spec, &spec).expect("a measurement");
        let mut callout = Callout::new(Selection::Keepout(0), handle, spec.clone(), at);
        callout.release(Instant::now());
        assert_eq!(callout.phase(), Phase::Lingering);
        let moved = gizmo::translate(&spec, [1.0, 0.0, 0.0]);
        callout.update(measure(&handle, &spec, &moved).expect("a measurement"));
        assert_eq!(callout.phase(), Phase::Dragging);
        assert!((callout.at.value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_number_parser_is_the_one_the_properties_panel_uses() {
        assert_eq!(parse_number("12.5"), Some(12.5));
        assert_eq!(parse_number(" 1 000.5 "), Some(1000.5));
        assert_eq!(parse_number("\u{2212}3"), Some(-3.0));
        for text in ["nan", "inf", "-inf", "hello", ""] {
            assert_eq!(parse_number(text), None, "{text} was accepted");
        }
    }
}
