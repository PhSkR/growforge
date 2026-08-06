//! Placing a shape by clicking its points in the viewport.
//!
//! The tube's own creation flow, and the reason it exists: "click two points to
//! make a line then drag the middle to make a curve". The add row's `tube`
//! button opens the mode this module describes instead of dropping a default
//! tube at the centre of the domain; the two clicks that follow are the tube's
//! two ends, and the bend handle the second click leaves selected is what curves
//! it.
//!
//! Three pieces, separated the way the floor grid separates its own:
//!
//! * [`landing`] is where a click lands - pure arithmetic over a ray, the
//!   configuration's geometry and the floor plane, which is what the tests aim
//!   at;
//! * [`Placing`] is the state machine's whole state: what is being placed, and
//!   the points it has so far;
//! * [`Placing::overlay`] is the only part that knows about triangles.
//!
//! The landing rule is one sentence: a click lands on the nearest surface its
//! ray meets, and on the floor grid's plane when it meets none. A click that
//! meets neither - at the sky, or past the ruled floor - lands nowhere at all
//! and leaves the placement exactly as it was, because the alternative is a
//! point at an arbitrary distance along a ray that was aimed at nothing.

use crate::config::{Config, ShapeSpec};
use crate::constants;
use crate::geometry::{Shape, Vec3, difference, length};
use crate::mesh::Mesh;
use crate::viewer::editor::gizmo;
use crate::viewer::editor::grid::FloorGrid;
use crate::viewer::editor::pick::{self, Ray};
use crate::viewer::editor::snap::Snap;
use crate::viewer::editor::state::{self, NewObject};
use crate::viewer::scene::{LayerMesh, Shading};
use crate::viewer::tessellate;

/// A placement in progress: the object the clicks are making, and the points
/// they have landed on so far.
///
/// One click moves it from no first point to one; the second click ends it. It
/// owns the viewport's clicks for as long as it lives, which is what suspends
/// selecting objects by clicking them - a click during a placement is a point,
/// not a selection.
#[derive(Debug, Clone, PartialEq)]
pub struct Placing {
    /// What the finished placement adds, and to which list. Carried from the
    /// add row that started it, so the tube lands where the button that was
    /// clicked would have put it.
    pub what: NewObject,
    /// The first point, once it has been clicked.
    pub first: Option<Vec3>,
    /// Where the next click would land, as the pointer last reported it, and
    /// `None` while the pointer is over nothing placeable. What the preview is
    /// drawn from.
    pub at: Option<Vec3>,
}

impl Placing {
    /// A placement of `what` that has not taken a point yet.
    pub fn new(what: NewObject) -> Placing {
        Placing {
            what,
            first: None,
            at: None,
        }
    }

    /// What the panel says to do next.
    pub fn hint(&self) -> &'static str {
        match self.first {
            None => constants::VIEW_EDIT_PLACE_HINT_FIRST,
            Some(_) => constants::VIEW_EDIT_PLACE_HINT_SECOND,
        }
    }

    /// The preview: a marker on every point the placement has, and - once there
    /// is a first point and somewhere for the second one to go - the tube those
    /// two points would make.
    ///
    /// `radius` is the radius the placed tube would have and `marker_edge` the
    /// edge of the point markers, both decided by the caller against the scene.
    /// `None` when there is nothing to draw, which is a placement whose pointer
    /// is over nothing and which has taken no point yet.
    pub fn overlay(&self, radius: f64, marker_edge: f64) -> Option<LayerMesh> {
        let mut out = LayerMesh::default();
        if let (Some(first), Some(at)) = (self.first, self.at) {
            let ghost = tessellate::shape(&Shape::Tube {
                p1: first,
                p2: at,
                bend: None,
                radius,
            });
            append(&mut out, &ghost, constants::VIEW_COLOR_PLACE_GHOST);
        }
        for point in [self.first, self.at].into_iter().flatten() {
            append(
                &mut out,
                &tessellate::marker_cube(point, marker_edge),
                constants::VIEW_COLOR_PLACE_MARKER,
            );
        }
        (!out.is_empty()).then_some(out)
    }
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

/// Where a placement click along `ray` lands, snapped, or `None` when it lands
/// nowhere.
///
/// The surface comes first: a ray that meets any of the configuration's
/// geometry - the design space included, see [`state::placement_targets`] -
/// lands on the nearest of it, so a point is placed *on* the model the user is
/// looking at. Meeting none of it, the click falls through to the ruled floor,
/// which is the plane everything else is drawn standing on.
///
/// Both are then snapped, every coordinate of them, to the increment the panel
/// is set to - the increment the grid on that floor is ruled at - so two points
/// clicked at the same grid line really are on it. That moves a point off the
/// surface it landed on by less than half an increment, which is the same trade
/// every snapped drag makes and the reason the bypass key exists.
pub fn landing(ray: &Ray, config: &Config, floor: Option<&FloorGrid>, snap: Snap) -> Option<Vec3> {
    let point =
        surface_point(ray, config).or_else(|| floor.and_then(|grid| floor_point(ray, grid)))?;
    let snapped = [
        snap.length(point[0]),
        snap.length(point[1]),
        snap.length(point[2]),
    ];
    snapped
        .iter()
        .all(|value| value.is_finite())
        .then_some(snapped)
}

/// The nearest point on the configuration's own geometry along `ray`.
fn surface_point(ray: &Ray, config: &Config) -> Option<Vec3> {
    let mut nearest: Option<f64> = None;
    for target in state::placement_targets(config) {
        let Some(t) = pick::hit_shape(ray, &target.shape) else {
            continue;
        };
        if nearest.is_none_or(|best| t < best) {
            nearest = Some(t);
        }
    }
    nearest.map(|t| ray.at(t))
}

/// Where `ray` crosses the plane the floor grid is ruled on, inside the
/// footprint it rules.
///
/// Bounded by that footprint rather than taken as an infinite plane, because
/// the plane is only there in as much as it is drawn: a ray that passes just
/// under the horizon crosses an infinite floor kilometres away, and a point out
/// there is not what the click meant. Outside the ruling there is nothing to
/// land on, and the click does nothing.
fn floor_point(ray: &Ray, grid: &FloorGrid) -> Option<Vec3> {
    let plane = [0.0, 0.0, grid.bounds.min[2]];
    let point = gizmo::plane_point(ray, plane, [0.0, 0.0, 1.0])?;
    let inside = (0..2).all(|d| point[d] >= grid.bounds.min[d] && point[d] <= grid.bounds.max[d]);
    inside.then_some(point)
}

/// True when two clicked points are too near one another to be two points.
///
/// The increment is the resolution the clicks themselves have: two of them
/// inside it are the same point clicked twice, and the tube they would make is
/// a degenerate one the placement refuses rather than writes. With snapping off
/// or bypassed there is no such resolution, and
/// [`constants::VIEW_EDIT_PLACE_MIN_LENGTH_MM`] stands in for it.
pub fn coincident(a: Vec3, b: Vec3, snap: Snap) -> bool {
    let tolerance = if snap.snaps_lengths() {
        snap.millimetres
    } else {
        constants::VIEW_EDIT_PLACE_MIN_LENGTH_MM
    };
    length(difference(b, a)) < tolerance
}

/// The straight tube two placed points make.
///
/// No bend: a placed tube is a line, and the curve is the drag that comes after
/// it. The file says so by carrying no `bend` key at all.
pub fn tube_between(p1: Vec3, p2: Vec3, radius: f64) -> ShapeSpec {
    ShapeSpec::Tube {
        p1,
        p2,
        bend: None,
        radius,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;
    use crate::viewer::editor::state::ShapeKind;

    /// The fixture's configuration, parsed.
    fn config() -> Config {
        Config::parse(crate::viewer::editor::tests::fixture()).expect("the fixture parses")
    }

    /// A ray straight down from high above.
    fn down(x: f64, y: f64) -> Ray {
        Ray {
            origin: [x, y, 300.0],
            direction: [0.0, 0.0, -1.0],
        }
    }

    fn floor() -> FloorGrid {
        FloorGrid::derive(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [64.0, 32.0, 32.0],
            },
            1.0,
        )
        .expect("a grid")
    }

    #[test]
    fn a_click_lands_on_the_nearest_surface_and_falls_through_to_the_floor() {
        let config = config();
        let grid = floor();
        let snap = Snap::default();

        // Straight down onto the lid of the domain, which is at z = 32: the
        // design space is a landing surface even though it is never selectable.
        let on_domain = landing(&down(30.0, 20.0), &config, Some(&grid), snap).expect("a landing");
        assert_eq!(on_domain, [30.0, 20.0, 32.0]);

        // Down the axis of the fixture's cylinder keepout, which stands proud of
        // the domain's lid at z = 33: the nearer surface is the one that wins.
        let on_keepout = landing(&down(16.0, 16.0), &config, Some(&grid), snap).expect("a landing");
        assert_eq!(on_keepout, [16.0, 16.0, 33.0]);

        // Beside the domain but over the ruled floor, which reaches a tenth of
        // the footprint past it: nothing is hit, so the plane answers.
        let on_floor = landing(&down(-4.0, 16.0), &config, Some(&grid), snap).expect("a landing");
        assert_eq!(on_floor, [-4.0, 16.0, 0.0]);

        // Past the ruling there is no plane to land on, and none at all when
        // the domain has no footprint to rule.
        assert!(landing(&down(-40.0, 16.0), &config, Some(&grid), snap).is_none());
        assert!(landing(&down(-4.0, 16.0), &config, None, snap).is_none());

        // And a ray aimed away from the floor - at the sky - meets neither.
        let up = Ray {
            origin: [-4.0, 16.0, 10.0],
            direction: [0.0, 0.0, 1.0],
        };
        assert!(landing(&up, &config, Some(&grid), snap).is_none());
    }

    #[test]
    fn a_landing_lands_on_the_increment_unless_it_is_bypassed() {
        let config = config();
        let grid = floor();
        let coarse = Snap {
            millimetres: 5.0,
            ..Snap::default()
        };
        let at = landing(&down(-4.4, 16.6), &config, Some(&grid), coarse).expect("a landing");
        assert_eq!(at, [-5.0, 15.0, 0.0]);
        // With no increment at all the point is exactly where the ray crossed.
        let free = landing(&down(-4.4, 16.6), &config, Some(&grid), Snap::OFF).expect("a landing");
        assert!((free[0] + 4.4).abs() < 1e-12 && (free[1] - 16.6).abs() < 1e-12);
    }

    #[test]
    fn a_subtracted_domain_entry_is_not_a_surface_to_land_on() {
        let mut config = config();
        // A second domain entry, subtracting a block out of the lid. Nothing
        // draws it, so nothing may land on it either - the click falls through
        // to the additive entry underneath.
        config.domain.push(crate::config::DomainEntry::Box {
            op: crate::config::CsgOpSpec::Subtract,
            min: [20.0, 10.0, 30.0],
            max: [40.0, 24.0, 60.0],
            rotation_deg: None,
        });
        let at = landing(&down(30.0, 20.0), &config, Some(&floor()), Snap::default())
            .expect("a landing");
        assert_eq!(at, [30.0, 20.0, 32.0]);
    }

    #[test]
    fn two_points_inside_the_increment_are_one_point() {
        let snap = Snap::default();
        assert!(coincident([10.0, 0.0, 0.0], [10.0, 0.0, 0.0], snap));
        assert!(coincident([10.0, 0.0, 0.0], [10.4, 0.0, 0.0], snap));
        // Two neighbouring grid points are exactly an increment apart, which is
        // the shortest tube the grid can describe and a legal one.
        assert!(!coincident([10.0, 0.0, 0.0], [11.0, 0.0, 0.0], snap));
        // With snapping off the floor is the placement minimum instead.
        let inside = constants::VIEW_EDIT_PLACE_MIN_LENGTH_MM * 0.5;
        assert!(coincident([0.0, 0.0, 0.0], [inside, 0.0, 0.0], Snap::OFF));
        assert!(!coincident(
            [0.0, 0.0, 0.0],
            [constants::VIEW_EDIT_PLACE_MIN_LENGTH_MM * 2.0, 0.0, 0.0],
            Snap::OFF
        ));
    }

    #[test]
    fn the_preview_draws_the_points_it_has_and_the_tube_between_them() {
        let mut placing = Placing::new(NewObject::Keepout(ShapeKind::Tube));
        assert_eq!(placing.hint(), constants::VIEW_EDIT_PLACE_HINT_FIRST);
        // Nothing at all before the pointer has landed anywhere.
        assert!(placing.overlay(2.0, 1.0).is_none());

        // The marker under the pointer, on its own.
        placing.at = Some([10.0, 0.0, 0.0]);
        let marker = placing.overlay(2.0, 1.0).expect("a marker");
        assert_eq!(colors(&marker), vec![constants::VIEW_COLOR_PLACE_MARKER]);

        // And with a first point down, the band as well - and the hint moves on.
        placing.first = Some([0.0, 0.0, 0.0]);
        assert_eq!(placing.hint(), constants::VIEW_EDIT_PLACE_HINT_SECOND);
        let band = placing.overlay(2.0, 1.0).expect("a band");
        assert!(band.triangles() > marker.triangles());
        let colors = colors(&band);
        assert!(colors.contains(&constants::VIEW_COLOR_PLACE_GHOST));
        assert!(colors.contains(&constants::VIEW_COLOR_PLACE_MARKER));
        // The band spans the two points, plus the radius its rounded ends add.
        assert!(band.bounds.min[0] <= -2.0 && band.bounds.max[0] >= 12.0);
    }

    /// The distinct colours a preview carries.
    fn colors(layer: &LayerMesh) -> Vec<[f32; 4]> {
        let mut seen: Vec<[f32; 4]> = Vec::new();
        for vertex in &layer.vertices {
            if !seen.contains(&vertex.color) {
                seen.push(vertex.color);
            }
        }
        seen
    }
}
