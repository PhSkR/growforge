//! Projection of the exported surface's vertices back onto the analytic
//! boundaries the configuration described.
//!
//! Everything upstream of the export works on a voxel field. A cell is
//! classified by its **centre** ([`crate::grid::Grid::classify`]), so a cell
//! whose centre lies a hair outside a keepout is material for its whole width
//! and the modelled solid reaches up to half a voxel into the forbidden region;
//! marching cubes then puts vertices on the node lattice, which is half a voxel
//! further out again, and Taubin smoothing moves them once more. The exported
//! STL therefore cuts a fraction of a voxel into a keepout and bulges the same
//! distance out of the domain. On a pin bore that is not a cosmetic error: a
//! 2.75 mm bore meshed on a 1.5 mm grid comes out under diameter and the pin
//! does not fit.
//!
//! **Supersampling does not fix it.** The refined lattice is the trilinear
//! interpolant of the same coarse node field ([`crate::mesh::ScalarField::refined`]);
//! it describes the same encroaching surface with more triangles.
//!
//! What fixes it is knowing where the boundary really is, which is what
//! [`crate::geometry::Boundaries`] carries on the problem. Under the default
//! `[output] boundaries = "exact"` this pass takes every exported vertex that
//! violates one - inside a keepout, or outside the domain - and moves it onto
//! the analytic surface it violates, plus
//! [`constants::BOUNDARY_CLAMP_EPS_MM`] on the legal side so a containment test
//! agrees. Where the part meets a bore, the bore becomes the cylinder that was
//! asked for.
//!
//! **The scatter goes both ways, and so does the correction.** The sampling
//! above does not put a wall's vertices reliably outside the surface: it puts
//! them on *both* sides of it, and the smoothing that rounds the staircase pulls
//! the corners inward. Correcting only the vertices that are proud of a
//! boundary, which are the ones legality can see, leaves the inward half as
//! dimples: a third to a half of a voxel, measured, and visible as scalloping on
//! what was supposed to be a cone. So a vertex that is legal but sits within
//! [`constants::BOUNDARY_CLAMP_CAPTURE_VOXELS`] of the boundary it rests on is
//! seated onto it as well, by the same projection onto the same legal side. A
//! vertex further away than that rests on nothing - it is an
//! optimizer's free surface through the middle of the domain - and is left
//! exactly where the smoothing put it.
//!
//! Three things bound it, because a projection that is allowed to do anything is
//! a projection that can wreck a surface:
//!
//! * **Member by member.** A vertex is pushed out of the keepout it is deepest
//!   inside, onto *that member's* own surface - closed form for the box, the
//!   sphere, the capped cylinder and the tube, bounded Newton steps for the
//!   ellipsoid, which has no closed form (see
//!   [`crate::geometry::Shape::nearest_surface_point`]). The corrected position
//!   is then tested again, because overlapping keepouts can hand a vertex to one
//!   another and because leaving one can leave the domain. At most
//!   [`constants::BOUNDARY_CLAMP_MAX_PASSES`] rounds.
//! * **A displacement cap.** The encroachment this exists for is sub-voxel by
//!   construction, so a vertex that would have to move further than
//!   [`constants::BOUNDARY_CLAMP_MAX_DISPLACEMENT_VOXELS`] voxels is not this
//!   defect and is left alone.
//! * **An honest give-up.** A vertex that is still illegal when the budget runs
//!   out, or whose correction is past the cap, keeps the position it had and is
//!   counted in [`ClampReport::gave_up`]. Nothing here loops forever and nothing
//!   here claims a surface is clean when it is not.
//!
//! **And what it leaves alone is counted.** Leaving a vertex where it is, is the
//! right answer for a free surface and the wrong one for a face that was drawn
//! against a shape and came out short of it by more than the capture band. This
//! pass cannot tell those two apart - the difference is the engine's, not the
//! geometry's - so it measures rather than guesses: a vertex that comes to rest
//! further from its nearest boundary than a clamped vertex's own offset, and
//! nearer to it than [`constants::BOUNDARY_ADRIFT_WINDOW_VOXELS`], is counted in
//! [`ClampReport::adrift`], with the worst distance beside it. Nothing moves
//! because of it, and the count is taken on every run; what the run *says* about
//! it is [`ClampReport::notes`]'s question, and depends on whether the part was
//! drawn from shapes or designed between them. It exists because a face that
//! shipped a fraction of a millimetre off the surface it belonged on read as a
//! perfectly clean clamp line, and the slicer found the defect the tool had not
//! mentioned.
//!
//! The pass runs **after** the island cull, so no work is spent on fragments
//! that are about to be discarded and the culling verdicts see exactly the
//! geometry they always did, and **before** validation, so what the validator
//! accepts is the file that ships. Collapsing two vertices onto one point is the
//! real risk of moving them, and [`constants::MIN_TRIANGLE_AREA_MM2`] stays the
//! arbiter of that: a clamp that produces a degenerate triangle fails the export
//! rather than shipping one.
//!
//! The live preview never reaches here - `viewer::scene::preview_surface` runs
//! marching cubes alone, skipping smoothing, culling and validation - so a
//! preview is unclamped by construction and needs no guard. (Written as a plain
//! name rather than a link: the viewer is behind a feature gate and this module
//! is not.)

use rayon::prelude::*;

use crate::constants;
use crate::geometry::{Boundaries, Csg, Shape, Vec3, difference, length, scale, sum};
use crate::mesh::Mesh;

/// What the boundary clamp found and did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClampReport {
    /// Vertices that were moved onto a boundary: the ones that were proud of
    /// one, and the ones that were resting a sampling error short of one.
    pub vertices_moved: usize,
    /// The furthest any one of them travelled, in millimetres. Zero when none
    /// moved.
    pub max_displacement_mm: f64,
    /// Vertices left where they were because no legal position was reached
    /// inside the pass budget, or because the one that was lay past the
    /// displacement cap. The exported surface still crosses a boundary at each
    /// of them.
    pub gave_up: usize,
    /// Vertices that came to rest near an analytic boundary without sitting on
    /// it: further out than a clamped vertex's own offset, and no further than
    /// [`constants::BOUNDARY_ADRIFT_WINDOW_VOXELS`] from it. Measured after the
    /// corrections above were applied, so a vertex the pass seated is not one of
    /// these and a vertex it could not reach is.
    ///
    /// Nothing was done about them: this is the count of what the pass left, not
    /// of what it failed at. Whether that is a defect, a pass falling short or
    /// nothing at all is the run's question and not the geometry's, so the count
    /// is always taken and not always spoken - see [`ClampReport::notes`].
    pub adrift: usize,
    /// How far the furthest of them rests from the boundary it is nearest to, in
    /// millimetres. Zero when none are.
    pub max_adrift_mm: f64,
}

impl ClampReport {
    /// What the run says about this pass, on the console and in the editor's
    /// panel alike.
    ///
    /// One line per sentence, an outcome that needs saying loudly opening with
    /// the word "warning", so a caller can print them straight or lay them out
    /// in a panel without parsing anything back out - the same contract
    /// [`crate::trim::TrimReport::notes`] keeps.
    ///
    /// `solid` is [`crate::problem::Problem::is_solid`] and `flushing` is
    /// [`crate::problem::Problem::is_flushing`]. Between them they decide what -
    /// and whether - this says about the vertices the pass left resting off a
    /// surface, because what an adrift vertex *means* is the run's answer rather
    /// than the mesh's:
    ///
    /// * **Drawn.** Under the solid engine the part is the shapes it was drawn
    ///   from, so every exported surface belongs to a domain or keepout boundary
    ///   and a vertex resting off one is a defect about to ship: a warning,
    ///   always.
    /// * **Designed, and flushed out to the shapes.** An engine that designs the
    ///   material in between makes free surfaces everywhere, and one running near
    ///   a boundary is not by itself wrong. But a run that asked for
    ///   `[output] flush` asked for its walls to reach the shapes they rest
    ///   against, and a vertex still short of one is that pass falling short:
    ///   said plainly, with the key that reaches further.
    /// * **Designed, and not flushed.** Nothing is said at all. Measured on a
    ///   stock optimized part, roughly a third of its vertices lie within the
    ///   window of the domain box - they are the part's own free surfaces near a
    ///   wall, and a line that appears on every run of every design says nothing
    ///   about the one run that is wrong. The count stays on the report for
    ///   whatever asks it; `[output] flush` is how a user opts into being told.
    ///
    /// Both are taken as arguments rather than carried on the report because
    /// every caller that composes these lines holds the problem already, and the
    /// compiler is then what keeps the console and the editor's panel saying the
    /// same thing.
    pub fn notes(&self, solid: bool, flushing: bool) -> Vec<String> {
        let mut notes = Vec::new();
        if self.vertices_moved == 0 {
            notes.push(
                "every exported vertex was already inside the domain and clear of the keepouts"
                    .to_string(),
            );
        } else {
            notes.push(format!(
                "clamped {} vertices onto the analytic surfaces, moving them {:.4} mm at most",
                self.vertices_moved, self.max_displacement_mm
            ));
        }
        if self.gave_up > 0 {
            let left = if self.gave_up == 1 {
                "1 vertex was".to_string()
            } else {
                format!("{} vertices were", self.gave_up)
            };
            notes.push(format!(
                "warning: {left} left unmoved and the surface still crosses a boundary there: no \
                 legal position was reached in {} passes, or the correction needed was further \
                 than the {} voxel a sampling artefact can be",
                constants::BOUNDARY_CLAMP_MAX_PASSES,
                constants::BOUNDARY_CLAMP_MAX_DISPLACEMENT_VOXELS
            ));
        }
        if self.adrift > 0 {
            let resting = if self.adrift == 1 {
                "1 vertex rests".to_string()
            } else {
                format!("{} vertices rest", self.adrift)
            };
            match (solid, flushing) {
                (true, _) => notes.push(format!(
                    "warning: {resting} up to {:.4} mm off the surface they belong to: this part \
                     is the shapes it was drawn from, so every exported surface is a domain or \
                     keepout surface and one standing off it ships as a face in the wrong place",
                    self.max_adrift_mm
                )),
                (false, true) => notes.push(format!(
                    "[output] flush ran and {resting} up to {:.4} mm off the surface they belong \
                     to: where that is a wall meant to meet the shape it rests against, the pass \
                     did not reach it and a larger flush_depth_mm does; where it is a free \
                     surface running past a boundary, it rests on nothing and nothing is wrong",
                    self.max_adrift_mm
                )),
                // Every optimized part has free surfaces near its boundaries;
                // see the note on this function for why silence is the honest
                // answer when nothing asked for them to be brought out.
                (false, false) => {}
            }
        }
        notes
    }
}

/// What the clamp decided about one vertex.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Correction {
    /// Inside the domain and clear of every keepout already.
    Legal,
    /// Illegal, and this is where it belongs.
    Moved(Vec3),
    /// Illegal, and left where it is; see [`ClampReport::gave_up`].
    GaveUp,
}

/// Move every vertex of `mesh` onto the boundary it belongs on: the ones that
/// violate one, and the ones resting a sampling error short of one.
///
/// `voxel_mm` is the grid spacing the field was sampled on, which is what the
/// displacement cap, the capture band and the adrift window are all measured in.
///
/// What comes back describes both halves of the pass: what it moved, and - in
/// [`ClampReport::adrift`] - what it left resting near a boundary without
/// seating, measured on the positions this function leaves behind.
pub fn resolve(mesh: &mut Mesh, boundaries: &Boundaries, voxel_mm: f64) -> ClampReport {
    let cap = constants::BOUNDARY_CLAMP_MAX_DISPLACEMENT_VOXELS * voxel_mm;
    let capture = constants::BOUNDARY_CLAMP_CAPTURE_VOXELS * voxel_mm;
    // Decided in parallel and applied in order, so the report counts the same
    // vertices in the same order however many threads ran.
    let corrections: Vec<Correction> = mesh
        .vertices
        .par_iter()
        .map(|vertex| corrected(*vertex, boundaries, cap, capture))
        .collect();

    let mut report = ClampReport {
        vertices_moved: 0,
        max_displacement_mm: 0.0,
        gave_up: 0,
        adrift: 0,
        max_adrift_mm: 0.0,
    };
    for (vertex, correction) in mesh.vertices.iter_mut().zip(corrections) {
        match correction {
            Correction::Legal => {}
            Correction::Moved(to) => {
                report.vertices_moved += 1;
                report.max_displacement_mm = report
                    .max_displacement_mm
                    .max(length(difference(to, *vertex)));
                *vertex = to;
            }
            Correction::GaveUp => report.gave_up += 1,
        }
    }

    // Measured on the corrected positions, which is the surface that ships, and
    // reduced rather than collected: a count and a maximum are the same whatever
    // order the threads finished in.
    let window = constants::BOUNDARY_ADRIFT_WINDOW_VOXELS * voxel_mm;
    let (adrift, worst) = mesh
        .vertices
        .par_iter()
        .filter_map(|vertex| adrift_by(*vertex, boundaries, window))
        .map(|distance| (1usize, distance))
        .reduce(|| (0, 0.0), |a, b| (a.0 + b.0, a.1.max(b.1)));
    report.adrift = adrift;
    report.max_adrift_mm = worst;
    report
}

/// How far `vertex` rests from the boundary it is nearest to, when that is far
/// enough to be worth saying and near enough to be about that boundary at all;
/// `None` otherwise.
///
/// The lower end is where a *corrected* vertex sits: the clamp lands one
/// [`constants::BOUNDARY_CLAMP_EPS_MM`] off the surface it seats onto, and the
/// iterated projections arrive within their own tolerance of that, so twice the
/// offset is the width of "already on it" - the same figure the tests measure a
/// seated vertex against. The upper end is
/// [`constants::BOUNDARY_ADRIFT_WINDOW_VOXELS`], past which a vertex rests on
/// nothing and belongs to no surface to be off.
///
/// It reads the same nearest boundary [`seated`] does, from the same function,
/// so what is counted here is exactly what that decided not to seat.
fn adrift_by(vertex: Vec3, boundaries: &Boundaries, window: f64) -> Option<f64> {
    let (distance, _) = nearest_boundary(vertex, boundaries)?;
    let seat = 2.0 * constants::BOUNDARY_CLAMP_EPS_MM;
    (distance.is_finite() && distance > seat && distance <= window).then_some(distance)
}

/// The analytic boundary `p` is nearest to, whichever side of it `p` sits on: how
/// far away it is, and which surface it is - a keepout member, or `None` for the
/// domain as a whole, which is a composite rather than a member.
///
/// `None` when the caller described no boundary at all, which is a mesh held to
/// nothing.
fn nearest_boundary(p: Vec3, boundaries: &Boundaries) -> Option<(f64, Option<&Shape>)> {
    let mut nearest: Option<(f64, Option<&Shape>)> = None;
    for shape in boundaries.keepout.shapes() {
        let distance = shape.signed_distance(p).abs();
        if nearest.is_none_or(|(best, _)| distance < best) {
            nearest = Some((distance, Some(shape)));
        }
    }
    if !boundaries.domain.is_empty() {
        let distance = boundaries.domain.signed_distance(p).abs();
        if nearest.is_none_or(|(best, _)| distance < best) {
            nearest = Some((distance, None));
        }
    }
    nearest
}

/// True when `p` is inside the domain and outside every keepout.
///
/// An empty half of `boundaries` constrains nothing: a caller with no keepouts,
/// or none that carried a domain in, is asking about the other one alone.
fn legal(p: Vec3, boundaries: &Boundaries) -> bool {
    if !boundaries.keepout.is_empty() && boundaries.keepout.signed_distance(p) < 0.0 {
        return false;
    }
    if !boundaries.domain.is_empty() && boundaries.domain.signed_distance(p) > 0.0 {
        return false;
    }
    true
}

/// Where one vertex belongs, or that it has to be left alone.
fn corrected(vertex: Vec3, boundaries: &Boundaries, cap: f64, capture: f64) -> Correction {
    if legal(vertex, boundaries) {
        return seated(vertex, boundaries, cap, capture);
    }
    let mut at = vertex;
    for _ in 0..constants::BOUNDARY_CLAMP_MAX_PASSES {
        let Some(next) = one_pass(at, boundaries) else {
            return Correction::GaveUp;
        };
        at = next;
        if legal(at, boundaries) {
            let displacement = length(difference(at, vertex));
            if !displacement.is_finite() || displacement > cap {
                return Correction::GaveUp;
            }
            return Correction::Moved(at);
        }
    }
    Correction::GaveUp
}

/// The other half of the same artefact: a vertex that is already legal, but is
/// resting a sampling error short of the surface it belongs on.
///
/// A wall's vertices come out of marching cubes and Taubin smoothing scattered
/// to *both* sides of the surface the wall really is. The proud ones are
/// corrected by [`corrected`] because they are illegal; these are not illegal at
/// all - they are a dimple, and left alone they are what makes an exported cone
/// scallop instead of being a cone.
///
/// Two things bound it, and between them they are why this cannot deform a
/// surface. **The capture band**: only a vertex within
/// [`constants::BOUNDARY_CLAMP_CAPTURE_VOXELS`] of a boundary is seated on it,
/// which is the scale a cell-centre classification can be wrong by and nothing
/// larger; an optimizer's free surface running through the middle of the domain
/// is nowhere near one and is never touched. **Legality of the result**: the
/// seated position is tested exactly as an incoming vertex is, so a correction
/// that would put a vertex proud of another boundary - the seam where a keepout
/// meets the domain wall - is dropped and the vertex stays where it legally
/// already was. Nothing here can produce a vertex that is proud of anything, and
/// a seat that cannot be made exactly is not made at all: `GaveUp` means "still
/// crosses a boundary", which is never true of these.
fn seated(vertex: Vec3, boundaries: &Boundaries, cap: f64, capture: f64) -> Correction {
    // The boundary this vertex is nearest to, whichever side of it the vertex
    // sits on: a keepout member, or the domain as a whole. The same selection
    // [`adrift_by`] reads, so that what is counted as resting off a surface is
    // measured against the surface this would have seated it onto.
    let Some((distance, which)) = nearest_boundary(vertex, boundaries) else {
        return Correction::Legal;
    };
    // Too far away to be a sampling artefact, or already on the surface: the
    // offset a correction lands on is the width of "already there".
    if !(distance.is_finite() && distance <= capture && distance > constants::BOUNDARY_CLAMP_EPS_MM)
    {
        return Correction::Legal;
    }
    let target = match which {
        Some(shape) => onto(vertex, shape),
        None => pull_in(vertex, &boundaries.domain),
    };
    match target {
        Some(to)
            if legal(to, boundaries)
                && length(difference(to, vertex)).is_finite()
                && length(difference(to, vertex)) <= cap =>
        {
            Correction::Moved(to)
        }
        _ => Correction::Legal,
    }
}

/// One correction: out of the keepout the point is deepest inside, or - when it
/// is clear of them all - back inside the domain.
///
/// Deepest first because that is the correction with the furthest to go, and
/// every other member is re-examined against the position it produces rather
/// than against the one it started from.
fn one_pass(at: Vec3, boundaries: &Boundaries) -> Option<Vec3> {
    let mut deepest: Option<(f64, &Shape)> = None;
    for shape in boundaries.keepout.shapes() {
        let distance = shape.signed_distance(at);
        if distance < 0.0 && deepest.is_none_or(|(best, _)| distance < best) {
            deepest = Some((distance, shape));
        }
    }
    if let Some((_, shape)) = deepest {
        return onto(at, shape);
    }
    if !boundaries.domain.is_empty() && boundaries.domain.signed_distance(at) > 0.0 {
        return pull_in(at, &boundaries.domain);
    }
    Some(at)
}

/// The point on `shape`'s surface that `at` belongs on, a hair outside it.
///
/// Outside is the legal side of a keepout, so that is the side the offset of
/// [`constants::BOUNDARY_CLAMP_EPS_MM`] is taken on whichever side `at` came
/// from: the direction of travel for a vertex being pushed out of the shape,
/// and the reverse of it for one being seated onto the surface from outside.
/// The sign is read from the field rather than from the caller, so one statement
/// of where a vertex belongs serves both.
fn onto(at: Vec3, shape: &Shape) -> Option<Vec3> {
    let surface = shape.nearest_surface_point(at)?;
    let travelled = difference(surface, at);
    let reach = length(travelled);
    if reach <= 0.0 || !reach.is_finite() {
        return None;
    }
    let outward = match shape.signed_distance(at) < 0.0 {
        true => travelled,
        false => difference(at, surface),
    };
    Some(sum(
        surface,
        scale(outward, constants::BOUNDARY_CLAMP_EPS_MM / reach),
    ))
}

/// The point just inside the domain that `at` belongs on, from either side of
/// it: the descent below is signed, so it seats a vertex that fell short of the
/// surface exactly as it pulls in one that overshot it.
///
/// The domain is an ordered CSG tree, whose field is a composition of minima and
/// maxima with non-differentiable seams and no nearest-point formula, so this is
/// a bounded descent onto the level set rather than a projection.
fn pull_in(at: Vec3, domain: &Csg) -> Option<Vec3> {
    descend(at, -constants::BOUNDARY_CLAMP_EPS_MM, |p| {
        domain.signed_distance(p)
    })
}

/// Move `start` onto the level set `field == target` by bounded steps along the
/// field's own gradient, read by central differences.
///
/// The step is the residual carried along the **unit** gradient direction, which
/// is Newton's own step for a field whose gradient is a unit vector - what a
/// signed distance is, wherever it is differentiable - and which cannot blow up
/// at a seam where two branches meet and the difference quotient collapses
/// towards zero. `None` is a descent that did not arrive inside
/// [`constants::BOUNDARY_CLAMP_MAX_STEPS`], or one that found no direction to
/// move in at all.
fn descend(start: Vec3, target: f64, field: impl Fn(Vec3) -> f64) -> Option<Vec3> {
    let half_step = constants::BOUNDARY_CLAMP_GRADIENT_STEP_MM;
    let mut at = start;
    for _ in 0..constants::BOUNDARY_CLAMP_MAX_STEPS {
        let residual = field(at) - target;
        if !residual.is_finite() {
            return None;
        }
        if residual.abs() <= constants::BOUNDARY_CLAMP_TOLERANCE_MM {
            return Some(at);
        }
        let gradient: Vec3 = std::array::from_fn(|d| {
            let mut low = at;
            let mut high = at;
            low[d] -= half_step;
            high[d] += half_step;
            (field(high) - field(low)) / (2.0 * half_step)
        });
        let slope = length(gradient);
        if slope <= 0.0 || !slope.is_finite() {
            return None;
        }
        at = difference(at, scale(gradient, residual / slope));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{CsgOp, ShapeUnion};

    /// A mesh of loose vertices: this pass reads no triangle, so a fixture does
    /// not need to be a surface to exercise it.
    fn loose(vertices: Vec<Vec3>) -> Mesh {
        Mesh {
            vertices,
            triangles: Vec::new(),
        }
    }

    /// Boundaries made of keepouts alone.
    fn forbidden(shapes: Vec<Shape>) -> Boundaries {
        Boundaries {
            domain: Csg::default(),
            keepout: ShapeUnion::new(shapes),
        }
    }

    /// The one voxel size the unit fixtures are measured against; large enough
    /// that the displacement cap is never what a projection test is asserting.
    const VOXEL_MM: f64 = 10.0;

    /// The four shapes whose nearest surface point is a closed form, each with a
    /// point inside it that lands exactly on the surface it names.
    #[test]
    fn a_point_inside_a_closed_form_shape_is_projected_onto_its_surface() {
        let cases: Vec<(&str, Shape, Vec3)> = vec![
            (
                "box",
                Shape::axis_aligned_box([0.0, 0.0, 0.0], [10.0, 20.0, 30.0]),
                [9.5, 12.0, 17.0],
            ),
            (
                "turned box",
                Shape::Box {
                    min: [0.0, 0.0, 0.0],
                    max: [10.0, 20.0, 30.0],
                    rotation_deg: [0.0, 0.0, 30.0],
                },
                [9.5, 12.0, 17.0],
            ),
            (
                "sphere",
                Shape::Sphere {
                    center: [1.0, 2.0, 3.0],
                    radius: 5.0,
                },
                [1.0, 2.0, 7.6],
            ),
            (
                "cylinder wall",
                Shape::Cylinder {
                    p1: [0.0, 0.0, 0.0],
                    p2: [0.0, 0.0, 20.0],
                    radius: 4.0,
                },
                [3.7, 0.4, 10.0],
            ),
            (
                "cylinder cap",
                Shape::Cylinder {
                    p1: [0.0, 0.0, 0.0],
                    p2: [0.0, 0.0, 20.0],
                    radius: 4.0,
                },
                [1.0, 0.5, 0.2],
            ),
            (
                "tube",
                Shape::Tube {
                    p1: [0.0, 0.0, 0.0],
                    p2: [20.0, 0.0, 0.0],
                    bend: None,
                    radius: 3.0,
                },
                [10.0, 2.6, 0.4],
            ),
            (
                "bent tube",
                Shape::Tube {
                    p1: [0.0, 0.0, 0.0],
                    p2: [20.0, 0.0, 0.0],
                    bend: Some([10.0, 8.0, 0.0]),
                    radius: 3.0,
                },
                [10.0, 6.5, 0.7],
            ),
        ];

        for (what, shape, inside) in cases {
            assert!(
                shape.signed_distance(inside) < 0.0,
                "{what}: the fixture point is not inside it"
            );
            let surface = shape
                .nearest_surface_point(inside)
                .unwrap_or_else(|| panic!("{what}: no projection"));
            assert!(
                shape.signed_distance(surface).abs() < 1e-9,
                "{what}: the projection landed at {} off the surface",
                shape.signed_distance(surface)
            );
            // And it is the *nearest* point: no closer one exists, so the step
            // it took is the depth the field reported.
            let travelled = length(difference(surface, inside));
            assert!(
                (travelled + shape.signed_distance(inside)).abs() < 1e-9,
                "{what}: moved {travelled} for a depth of {}",
                -shape.signed_distance(inside)
            );

            // Through the pass itself: the vertex comes out legal, and clear of
            // the surface rather than on it.
            let mut mesh = loose(vec![inside]);
            let report = resolve(&mut mesh, &forbidden(vec![shape]), VOXEL_MM);
            assert_eq!(report.vertices_moved, 1, "{what}");
            assert_eq!(report.gave_up, 0, "{what}");
            assert!(
                shape.signed_distance(mesh.vertices[0]) > 0.0,
                "{what}: the clamped vertex is still inside"
            );
            assert!(
                shape.signed_distance(mesh.vertices[0]) < 2.0 * constants::BOUNDARY_CLAMP_EPS_MM,
                "{what}: the clamp overshot the surface"
            );
        }
    }

    /// A point already outside a shape is never touched, whichever shape it is.
    ///
    /// Past the adrift window as well as past the capture band, so this is the
    /// vertex the pass has no business with at all: nothing moved, and nothing
    /// said, under either engine.
    #[test]
    fn a_vertex_that_violates_nothing_is_left_alone() {
        let shape = Shape::Sphere {
            center: [0.0; 3],
            radius: 4.0,
        };
        let window = constants::BOUNDARY_ADRIFT_WINDOW_VOXELS * VOXEL_MM;
        let outside = [4.0 + 2.0 * window, 1.0, -2.0];
        let mut mesh = loose(vec![outside]);
        let report = resolve(&mut mesh, &forbidden(vec![shape]), VOXEL_MM);
        assert_eq!(report.vertices_moved, 0);
        assert_eq!(report.gave_up, 0);
        assert_eq!(report.max_displacement_mm, 0.0);
        assert_eq!(report.adrift, 0, "{report:?}");
        assert_eq!(report.max_adrift_mm, 0.0, "{report:?}");
        assert_eq!(mesh.vertices[0], outside);
        for (solid, flushing) in [(false, false), (false, true), (true, false), (true, true)] {
            assert_eq!(
                report.notes(solid, flushing),
                vec![
                    "every exported vertex was already inside the domain and clear of the keepouts"
                        .to_string()
                ]
            );
        }
    }

    /// What the pass leaves alone, it now counts: a vertex resting a whole voxel
    /// off the surface it is nearest to is past the capture band, so the clamp
    /// does not touch it, and inside the adrift window, so the report says how
    /// many and how far.
    ///
    /// This is the reading the incident that asked for it needed. A part shipped
    /// with its bottom face resting 0.88 of a voxel above the plate; the clamp
    /// was right not to move it - that far out is not a sampling artefact - and
    /// the run report said nothing whatever about it.
    #[test]
    fn a_vertex_resting_off_a_surface_is_counted_and_measured() {
        let sphere = Shape::Sphere {
            center: [0.0; 3],
            radius: 20.0,
        };
        let boundaries = forbidden(vec![sphere]);
        let capture = constants::BOUNDARY_CLAMP_CAPTURE_VOXELS * VOXEL_MM;
        let window = constants::BOUNDARY_ADRIFT_WINDOW_VOXELS * VOXEL_MM;

        // One voxel out: twice the capture band, a tenth of the way through the
        // window, and exactly where it started when the pass is done.
        let off = [20.0 + VOXEL_MM, 0.0, 0.0];
        let mut mesh = loose(vec![off]);
        let report = resolve(&mut mesh, &boundaries, VOXEL_MM);
        assert_eq!(report.vertices_moved, 0, "{report:?}");
        assert_eq!(report.gave_up, 0, "{report:?}");
        assert_eq!(mesh.vertices[0], off, "the measurement moved a vertex");
        assert_eq!(report.adrift, 1, "{report:?}");
        assert!(
            (report.max_adrift_mm - VOXEL_MM).abs() < 1e-9,
            "measured {} mm for a vertex one {VOXEL_MM} mm voxel out",
            report.max_adrift_mm
        );

        // A seated vertex is on the surface, so it is not off it: the same
        // fixture the dimple test uses, counted after the seat rather than
        // before it.
        let mut mesh = loose(vec![[20.0 + 0.4 * capture, 0.0, 0.0]]);
        let report = resolve(&mut mesh, &boundaries, VOXEL_MM);
        assert_eq!(report.vertices_moved, 1, "{report:?}");
        assert_eq!(report.adrift, 0, "{report:?}");
        assert_eq!(report.max_adrift_mm, 0.0, "{report:?}");

        // And the worst of several is what the report carries, over both kinds
        // of boundary: the count is every one of them.
        let mut mesh = loose(vec![
            [20.0 + VOXEL_MM, 0.0, 0.0],
            [0.0, 20.0 + 0.75 * window, 0.0],
            [0.0, 0.0, 20.0 + 2.0 * window],
        ]);
        let report = resolve(&mut mesh, &boundaries, VOXEL_MM);
        assert_eq!(report.adrift, 2, "{report:?}");
        assert!(
            (report.max_adrift_mm - 0.75 * window).abs() < 1e-9,
            "{report:?}"
        );

        // What the run says about it, from the report the pass really produced,
        // in each of the three things a run can be. Loud on a part that is its
        // shapes, whether or not it was flushed:
        for flushing in [false, true] {
            let loud = report.notes(true, flushing);
            assert_eq!(loud.len(), 2, "{loud:?}");
            assert!(
                loud[1].starts_with("warning: 2 vertices rest up to"),
                "{loud:?}"
            );
        }
        // A designed part whose walls were asked to reach the shapes: the pass
        // fell short, said plainly and with the key that reaches further.
        let flushed = report.notes(false, true);
        assert_eq!(flushed.len(), 2, "{flushed:?}");
        assert!(!flushed[1].starts_with("warning"), "{flushed:?}");
        assert!(
            flushed[1].starts_with("[output] flush ran and"),
            "{flushed:?}"
        );
        assert!(flushed[1].contains("flush_depth_mm"), "{flushed:?}");
        // And a designed part that asked for nothing: silence. Every optimized
        // part has free surfaces near its boundaries, and a line on every run
        // is a line nobody reads on the run that matters.
        let quiet = report.notes(false, false);
        assert_eq!(quiet.len(), 1, "{quiet:?}");
        assert!(
            !quiet.iter().any(|note| note.contains("off the surface")),
            "an unflushed design was told about its free surfaces: {quiet:?}"
        );
        // The count itself is taken either way: what changes is what is said.
        assert_eq!(report.adrift, 2, "{report:?}");
    }

    /// Both ends of the window, on the measurement itself.
    ///
    /// The lower one is the width of "already on the surface": a clamped vertex
    /// is left one offset clear of what it was seated onto, so nothing inside
    /// twice that is off anything. The upper one is where a surface stops
    /// belonging to a boundary at all - past it lies an optimizer's free surface
    /// through the middle of the domain, which rests on nothing and is nobody's
    /// defect.
    #[test]
    fn the_adrift_window_is_bounded_at_both_ends() {
        let sphere = Shape::Sphere {
            center: [0.0; 3],
            radius: 20.0,
        };
        let boundaries = forbidden(vec![sphere]);
        let window = constants::BOUNDARY_ADRIFT_WINDOW_VOXELS * VOXEL_MM;
        let seat = 2.0 * constants::BOUNDARY_CLAMP_EPS_MM;
        let out = |distance: f64| [20.0 + distance, 0.0, 0.0];

        // On the surface, and within the offset a correction lands on: nothing
        // to say about either.
        assert_eq!(adrift_by(out(0.0), &boundaries, window), None);
        assert_eq!(adrift_by(out(0.5 * seat), &boundaries, window), None);
        // Clear of it: measured, however small the gap.
        let just_off = adrift_by(out(10.0 * seat), &boundaries, window).expect("counted");
        assert!((just_off - 10.0 * seat).abs() < 1e-12, "{just_off}");
        // The far end is inclusive, and a hair past it is nothing.
        let edge = adrift_by(out(window), &boundaries, window).expect("counted");
        assert!((edge - window).abs() < 1e-12, "{edge}");
        assert_eq!(adrift_by(out(window + 1e-6), &boundaries, window), None);

        // A mesh held to nothing rests off nothing.
        assert_eq!(
            adrift_by(out(0.5 * window), &Boundaries::default(), window),
            None
        );
    }

    /// The other half of the sampling artefact: a vertex that violates nothing
    /// but sits a fraction of a voxel short of the surface it belongs on is
    /// seated onto it, from either side and against either kind of boundary.
    ///
    /// This is the dimple. Marching cubes and the smoothing that follows it
    /// scatter a wall's vertices to both sides of the surface; correcting only
    /// the illegal side leaves the legal side scalloped, which is what an
    /// exported cone came out as before this existed.
    #[test]
    fn a_vertex_resting_short_of_a_surface_is_seated_onto_it() {
        let eps = constants::BOUNDARY_CLAMP_EPS_MM;
        let band = constants::BOUNDARY_CLAMP_CAPTURE_VOXELS * VOXEL_MM;
        let sphere = Shape::Sphere {
            center: [0.0; 3],
            radius: 20.0,
        };

        // A keepout, from the legal side: 2 mm clear of a surface it should be
        // on, which at this voxel size is well inside the band.
        let short = [22.0, 0.0, 0.0];
        let mut mesh = loose(vec![short]);
        let report = resolve(&mut mesh, &forbidden(vec![sphere]), VOXEL_MM);
        assert_eq!(report.vertices_moved, 1, "{report:?}");
        assert_eq!(report.gave_up, 0, "{report:?}");
        let seated = mesh.vertices[0];
        let distance = sphere.signed_distance(seated);
        assert!(
            distance > 0.0 && distance <= 2.0 * eps,
            "seated at {distance} mm from the surface, on the wrong side or short of it"
        );

        // And the domain, from inside it: the same gap, the same seat, and the
        // result is still strictly inside.
        let domain = Csg::new(vec![(CsgOp::Add, sphere)]);
        let boundaries = Boundaries {
            domain,
            keepout: ShapeUnion::new(Vec::new()),
        };
        let mut mesh = loose(vec![[18.0, 0.0, 0.0]]);
        let report = resolve(&mut mesh, &boundaries, VOXEL_MM);
        assert_eq!(report.vertices_moved, 1, "{report:?}");
        let distance = boundaries.domain.signed_distance(mesh.vertices[0]);
        assert!(
            distance < 0.0 && distance.abs() <= 2.0 * eps,
            "seated at {distance} mm inside the domain surface"
        );

        // Past the band nothing is touched: a free surface running through the
        // middle of the domain is not a sampling artefact, and the smoothing
        // that shaped it is left alone. Left alone and counted, which is the
        // whole of what the adrift reading adds: the geometry is the same.
        let far = [20.0 + 2.0 * band, 0.0, 0.0];
        let mut mesh = loose(vec![far]);
        let report = resolve(&mut mesh, &forbidden(vec![sphere]), VOXEL_MM);
        assert_eq!(report.vertices_moved, 0, "{report:?}");
        assert_eq!(mesh.vertices[0], far);
        assert_eq!(report.adrift, 1, "{report:?}");
    }

    /// The ellipsoid has no closed form, so its projection is iterated - and it
    /// has to arrive, on the turned shape as well as the plain one.
    ///
    /// The interior points are written in the ellipsoid's **own** coordinates
    /// and carried through the rotation, so the same offsets are the same
    /// fractions of the same semi-axes whichever way the shape is turned.
    #[test]
    fn an_ellipsoid_projection_converges_inside_its_budget() {
        use crate::geometry::{rotate, rotation_matrix};

        let center = [1.0, -2.0, 4.0];
        let radii = [6.0, 3.0, 2.0];
        for rotation in [[0.0; 3], [0.0, 0.0, 30.0], [15.0, -35.0, 62.0]] {
            let shape = Shape::Ellipsoid {
                center,
                radii,
                rotation_deg: rotation,
            };
            let matrix = rotation_matrix(rotation);
            // Points all over the interior: a hair inside the surface on each
            // axis, an oblique one, and one a hair off the centre.
            for local in [
                [5.9, 0.0, 0.0],
                [0.0, 2.9, 0.0],
                [0.0, 0.0, 1.9],
                [2.0, -1.0, 1.0],
                [-4.5, 0.5, -0.5],
                [1e-6, 0.0, 0.0],
            ] {
                let inside = sum(center, rotate(&matrix, local));
                assert!(
                    shape.signed_distance(inside) < 0.0,
                    "{rotation:?} {local:?} is not inside the fixture"
                );
                let surface = shape
                    .nearest_surface_point(inside)
                    .unwrap_or_else(|| panic!("{rotation:?} {local:?}: no projection"));
                // Arrived, to the tolerance the budget promises.
                assert!(
                    shape.signed_distance(surface).abs()
                        <= constants::BOUNDARY_CLAMP_TOLERANCE_MM + 1e-12,
                    "{rotation:?} {local:?}: landed {} off the surface",
                    shape.signed_distance(surface)
                );
                // And it left along the outward normal rather than wandering:
                // the step it took is no shorter than the shape's own field
                // says the surface is, which is that field's Lipschitz bound.
                let travelled = length(difference(surface, inside));
                assert!(
                    travelled >= -shape.signed_distance(inside) - 1e-9,
                    "{rotation:?} {local:?}: moved {travelled} for a depth bound of {}",
                    -shape.signed_distance(inside)
                );
            }
            // The centre has no single nearest point, and says so rather than
            // inventing one.
            assert_eq!(shape.nearest_surface_point(center), None);

            // Through the pass, from a vertex a fraction of a voxel inside.
            let just_inside = sum(center, rotate(&matrix, [0.0, 2.99, 0.0]));
            let mut mesh = loose(vec![just_inside]);
            let report = resolve(&mut mesh, &forbidden(vec![shape]), VOXEL_MM);
            assert_eq!(report.vertices_moved, 1, "{rotation:?}");
            assert_eq!(report.gave_up, 0, "{rotation:?}");
            assert!(
                shape.signed_distance(mesh.vertices[0]) > 0.0,
                "{rotation:?}: the clamped vertex is still inside"
            );
        }
    }

    /// Two overlapping keepouts: leaving one puts the vertex inside the other,
    /// so a single projection is not enough and the pass has to go round again.
    #[test]
    fn overlapping_keepouts_are_left_by_more_than_one_projection() {
        let first = Shape::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 5.0,
        };
        let second = Shape::Sphere {
            center: [6.0, 0.0, 0.0],
            radius: 5.0,
        };
        let boundaries = forbidden(vec![first, second]);
        // Deep inside the first and inside the lens the two share, so the first
        // projection - out of the deeper one - lands inside the second.
        let start = [2.0, 0.5, 0.0];
        assert!(first.signed_distance(start) < 0.0 && second.signed_distance(start) < 0.0);
        let once = onto(start, &first).expect("a projection");
        assert!(
            second.signed_distance(once) < 0.0,
            "the fixture no longer needs a second pass"
        );

        let mut mesh = loose(vec![start]);
        let report = resolve(&mut mesh, &boundaries, VOXEL_MM);
        assert_eq!(report.gave_up, 0, "{report:?}");
        assert_eq!(report.vertices_moved, 1);
        let at = mesh.vertices[0];
        assert!(
            first.signed_distance(at) > 0.0 && second.signed_distance(at) > 0.0,
            "the vertex is still inside one of them: {at:?}"
        );
    }

    /// A tube thicker than the bend it follows overlaps itself, and the offset
    /// construction that is exact for every other tube is not a projection
    /// there: a radius out from the nearest centre-line point crosses the arc's
    /// own centre and lands back **inside** the solid.
    ///
    /// The half circle below has a centre line of radius 2.6 mm. At a tube
    /// radius of 3.0 the shape is self-intersecting and the projection is
    /// refused; at 0.99 of the arc radius it is not, and the projection is exact.
    /// The refusal is what lets the clamp reach its honest give-up on the
    /// **first** pass rather than spending its whole budget rediscovering the
    /// same illegal point, which is asserted directly on `one_pass`: `corrected`
    /// returns `GaveUp` the moment that answers `None`.
    #[test]
    fn a_self_intersecting_tube_refuses_to_name_a_surface_point() {
        use crate::geometry::{tube_closest_point, tube_self_intersects};

        let arc_radius = 2.6;
        let (p1, bend, p2) = (
            [arc_radius, 0.0, 0.0],
            [0.0, arc_radius, 0.0],
            [-arc_radius, 0.0, 0.0],
        );
        let tube = |radius| Shape::Tube {
            p1,
            p2,
            bend: Some(bend),
            radius,
        };

        // The point the assertions below are made at: near the arc's first end,
        // where the outward ray sweeps right across the disc.
        let inside = [2.2, 0.2, 0.0];

        // Just under the threshold: an ordinary tube, projected exactly - even
        // here, where the projection passes within a hair of the arc's own
        // centre. Below the threshold it cannot pass *through* it, which is what
        // keeps the nearest point of the curve the same point and the
        // construction exact.
        let sound = tube(0.99 * arc_radius);
        assert!(!sound.self_intersects());
        assert!(!tube_self_intersects(p1, p2, Some(bend), 0.99 * arc_radius));
        assert!(sound.signed_distance(inside) < 0.0);
        let surface = sound.nearest_surface_point(inside).expect("a projection");
        assert!(
            sound.signed_distance(surface).abs() < 1e-9,
            "landed {} off the surface",
            sound.signed_distance(surface)
        );

        // Past it: refused rather than answered wrongly. The threshold itself is
        // probed from either side rather than exactly on it - the circumcircle
        // is solved in floating point, so "radius exactly the arc radius" is not
        // a case a test can name.
        let overlapping = tube(3.0);
        assert!(overlapping.self_intersects());
        assert!(tube_self_intersects(p1, p2, Some(bend), 3.0));
        assert!(tube_self_intersects(p1, p2, Some(bend), 1.001 * arc_radius));
        assert!(overlapping.signed_distance(inside) < 0.0);
        assert_eq!(overlapping.nearest_surface_point(inside), None);

        // And this is why: the construction it would otherwise have used carries
        // the vertex across the disc and lands it back inside the solid. Swept
        // over several interior points rather than asserted at one, because a
        // single point can land on the surface by coincidence - the answer is
        // wrong in this regime, not wrong everywhere in it.
        let worst = [inside, [0.6, 2.0, 0.0], [0.0, 2.0, 0.0]]
            .into_iter()
            .map(|p| {
                assert_eq!(overlapping.nearest_surface_point(p), None, "{p:?}");
                let on_line = tube_closest_point(p, p1, p2, Some(bend));
                let outward = difference(p, on_line);
                let naive = sum(on_line, scale(outward, 3.0 / length(outward)));
                overlapping.signed_distance(naive)
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            worst < -0.5,
            "the fixture no longer reproduces the defect: worst is {worst}"
        );

        // Through the pass: the vertex is left exactly where it was, counted,
        // and given up on in one round rather than eight.
        let boundaries = forbidden(vec![overlapping]);
        assert_eq!(onto(inside, &overlapping), None);
        assert_eq!(
            one_pass(inside, &boundaries),
            None,
            "the pass has to give up on the first round rather than burn its budget"
        );
        let mut mesh = loose(vec![inside, [50.0, 50.0, 50.0]]);
        let report = resolve(&mut mesh, &boundaries, VOXEL_MM);
        assert_eq!(report.vertices_moved, 0, "{report:?}");
        assert_eq!(report.gave_up, 1, "{report:?}");
        assert_eq!(mesh.vertices[0], inside, "the vertex was moved");
        assert_eq!(mesh.vertices[1], [50.0, 50.0, 50.0]);

        // A straight tube cannot self-intersect however thick it is, which is
        // the same arithmetic as a bend that never became an arc.
        for bend in [
            None,
            Some([arc_radius, 0.0, 0.5 * constants::TUBE_COLLINEAR_EPS_MM]),
        ] {
            let straight = Shape::Tube {
                p1,
                p2,
                bend,
                radius: 1e6,
            };
            assert!(!straight.self_intersects(), "{bend:?}");
        }
    }

    /// The other way a tube overlaps itself, and the one the fold bound alone
    /// cannot see: an arc bent nearly the whole way round, whose two **ends**
    /// have closed on each other across the gap. Its radius is well under the
    /// arc's, so nothing about its bend is wrong; what is wrong is that a point
    /// beyond one end projects to a surface point inside the other.
    ///
    /// These are the numbers the review found it at, at printable scale.
    #[test]
    fn a_tube_whose_ends_have_closed_across_the_gap_is_refused_too() {
        use crate::geometry::{tube_arc, tube_closest_point};

        let (p1, p2, bend) = ([5.909, 0.522, 0.0], [5.909, -0.522, 0.0], [-6.0, 0.0, 0.0]);
        let radius = 1.8;
        let shape = Shape::Tube {
            p1,
            p2,
            bend: Some(bend),
            radius,
        };
        let arc = tube_arc(p1, p2, Some(bend)).expect("an arc");

        // The fold bound is nowhere near: this tube is less than a third of its
        // own arc radius, and more than half a turn round it.
        assert!(
            radius < 0.35 * arc.radius,
            "arc radius {} against tube radius {radius}",
            arc.radius
        );
        assert!(arc.span > std::f64::consts::PI);
        assert!(arc.reach() < radius, "reach {}", arc.reach());
        assert!(shape.self_intersects(), "the gap mode was not detected");

        // Points in the gap, just inside one end: refused, and the construction
        // that would otherwise have been used lands well inside the other end.
        let worst = [[5.909, 0.3, 0.0], [6.2, 0.0, 0.0], [5.7, -0.3, 0.0]]
            .into_iter()
            .map(|p| {
                assert!(shape.signed_distance(p) < 0.0, "{p:?} is not inside");
                assert_eq!(shape.nearest_surface_point(p), None, "{p:?}");
                let on_line = tube_closest_point(p, p1, p2, Some(bend));
                let outward = difference(p, on_line);
                let naive = sum(on_line, scale(outward, radius / length(outward)));
                shape.signed_distance(naive)
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            worst < -0.5,
            "the fixture no longer reproduces the defect: worst is {worst}"
        );

        // And through the pass: one round to the give-up, vertex untouched.
        let boundaries = forbidden(vec![shape]);
        let vertex = [5.909, 0.3, 0.0];
        assert_eq!(onto(vertex, &shape), None);
        assert_eq!(one_pass(vertex, &boundaries), None);
        let mut mesh = loose(vec![vertex]);
        let report = resolve(&mut mesh, &boundaries, VOXEL_MM);
        assert_eq!(report.vertices_moved, 0, "{report:?}");
        assert_eq!(report.gave_up, 1, "{report:?}");
        assert_eq!(mesh.vertices[0], vertex);

        // Open the same gap out and the tube is ordinary again: the ends are
        // then further apart than the tube is thick, and the projection is
        // exact. Nothing here is a blanket refusal of a nearly closed arc.
        let open = Shape::Tube {
            p1: [3.0, 5.196, 0.0],
            p2: [3.0, -5.196, 0.0],
            bend: Some(bend),
            radius,
        };
        assert!(!open.self_intersects());
        // Probed at the same place the closed one fails: just inside one end,
        // facing the other across the gap.
        for inside in [[3.0, 4.696, 0.0], [3.4, 5.0, 0.2], [-5.0, 0.0, 0.3]] {
            assert!(open.signed_distance(inside) < 0.0, "{inside:?}");
            let surface = open.nearest_surface_point(inside).expect("a projection");
            assert!(
                open.signed_distance(surface).abs() < 1e-9,
                "{inside:?} landed {} off the surface",
                open.signed_distance(surface)
            );
        }
    }

    /// The give-up path: a vertex with no single nearest surface point is left
    /// exactly where it was, and counted.
    #[test]
    fn a_vertex_the_projection_cannot_name_a_target_for_is_left_and_counted() {
        // The centre of a sphere is equidistant from all of it.
        let shape = Shape::Sphere {
            center: [3.0, 4.0, 5.0],
            radius: 2.0,
        };
        let mut mesh = loose(vec![[3.0, 4.0, 5.0]]);
        let report = resolve(&mut mesh, &forbidden(vec![shape]), VOXEL_MM);
        assert_eq!(report.vertices_moved, 0);
        assert_eq!(report.gave_up, 1);
        assert_eq!(mesh.vertices[0], [3.0, 4.0, 5.0], "the vertex was moved");
        let notes = report.notes(false, true);
        assert_eq!(notes.len(), 3, "{notes:?}");
        assert!(notes[1].starts_with("warning: 1 vertex was"), "{notes:?}");
        // And the same vertex in the adrift count, which is not a double
        // report of one thing but two statements about it: it crosses the
        // boundary, and it rests 2 mm from the surface it belongs on. The
        // count is positional and asks nothing about how a vertex got there.
        assert_eq!(report.adrift, 1, "{report:?}");
        assert!((report.max_adrift_mm - 2.0).abs() < 1e-9, "{report:?}");
        assert!(notes[2].contains("1 vertex rests"), "{notes:?}");
        // The give-up line is the pass's own and is said whatever the run was;
        // only the adrift line is the one a design with no flush is spared.
        let unflushed = report.notes(false, false);
        assert_eq!(unflushed.len(), 2, "{unflushed:?}");
        assert!(
            unflushed[1].starts_with("warning: 1 vertex was"),
            "{unflushed:?}"
        );
    }

    /// The sanity cap: a vertex a whole voxel and more inside a keepout is not
    /// the sub-voxel artefact this pass corrects, so it is left alone.
    #[test]
    fn a_correction_past_the_displacement_cap_is_refused() {
        let shape = Shape::Sphere {
            center: [0.0; 3],
            radius: 10.0,
        };
        let voxel = 1.0;
        let cap = constants::BOUNDARY_CLAMP_MAX_DISPLACEMENT_VOXELS * voxel;
        // Just inside the cap, and just outside it.
        let near = [0.0, 0.0, 10.0 - 0.5 * cap];
        let deep = [0.0, 0.0, 10.0 - 2.0 * cap];

        let mut mesh = loose(vec![near, deep]);
        let report = resolve(&mut mesh, &forbidden(vec![shape]), voxel);
        assert_eq!(report.vertices_moved, 1, "{report:?}");
        assert_eq!(report.gave_up, 1, "{report:?}");
        assert!(report.max_displacement_mm <= cap);
        assert_ne!(mesh.vertices[0], near, "the shallow vertex was not moved");
        assert_eq!(mesh.vertices[1], deep, "the deep vertex was moved anyway");
    }

    /// The other half of the pass: a vertex outside the domain is pulled back
    /// onto it, including through a subtraction, where the field is a composite
    /// with a seam rather than a single shape.
    #[test]
    fn a_vertex_outside_the_domain_is_pulled_back_onto_it() {
        let domain = Csg::new(vec![
            (
                CsgOp::Add,
                Shape::axis_aligned_box([0.0, 0.0, 0.0], [20.0, 20.0, 20.0]),
            ),
            (
                CsgOp::Subtract,
                Shape::Sphere {
                    center: [20.0, 10.0, 10.0],
                    radius: 6.0,
                },
            ),
        ]);
        let boundaries = Boundaries {
            domain,
            keepout: ShapeUnion::default(),
        };
        // Past the far face, and inside the bite the sphere takes out of it.
        let past_face = [20.4, 4.0, 4.0];
        let in_the_bite = [17.0, 10.0, 10.0];
        assert!(boundaries.domain.signed_distance(past_face) > 0.0);
        assert!(boundaries.domain.signed_distance(in_the_bite) > 0.0);

        let mut mesh = loose(vec![past_face, in_the_bite]);
        let report = resolve(&mut mesh, &boundaries, VOXEL_MM);
        assert_eq!(report.gave_up, 0, "{report:?}");
        assert_eq!(report.vertices_moved, 2);
        for vertex in &mesh.vertices {
            let distance = boundaries.domain.signed_distance(*vertex);
            assert!(distance < 0.0, "{vertex:?} is still outside: {distance}");
            assert!(
                distance > -2.0 * constants::BOUNDARY_CLAMP_EPS_MM,
                "{vertex:?} was pushed {distance} deep rather than onto the surface"
            );
        }
        assert!(report.max_displacement_mm > 0.0);
        assert!(report.notes(false, true)[0].starts_with("clamped 2 vertices"));
        // Both were pulled onto the surface, so neither rests off one and the
        // clamp's line is the only line, whatever the run was.
        assert_eq!(report.adrift, 0, "{report:?}");
        for (solid, flushing) in [(false, false), (false, true), (true, false), (true, true)] {
            let notes = report.notes(solid, flushing);
            assert_eq!(notes.len(), 1, "{notes:?}");
        }
    }

    /// The descent underneath that: it arrives on the level set it was asked
    /// for, and gives up rather than spinning when there is no direction to go.
    #[test]
    fn the_descent_arrives_on_its_level_set_or_says_it_did_not() {
        let sphere = Shape::Sphere {
            center: [0.0; 3],
            radius: 7.0,
        };
        let field = |p: Vec3| sphere.signed_distance(p);
        for target in [0.0, -0.5, 1.5] {
            let landed = descend([13.0, 2.0, -4.0], target, field).expect("a descent");
            assert!(
                (field(landed) - target).abs() <= constants::BOUNDARY_CLAMP_TOLERANCE_MM,
                "target {target}: landed at {}",
                field(landed)
            );
        }
        // A field with no gradient anywhere has nowhere to send the point.
        assert_eq!(descend([1.0, 2.0, 3.0], 0.0, |_| 5.0), None);
        // And one that is not a number is refused rather than followed.
        assert_eq!(descend([1.0, 2.0, 3.0], 0.0, |_| f64::NAN), None);
    }

    /// Empty boundaries constrain nothing, which is what lets a caller hold a
    /// mesh to its keepouts alone - or to nothing at all.
    #[test]
    fn empty_boundaries_move_nothing() {
        let vertices = vec![[0.0, 0.0, 0.0], [1e6, -1e6, 1e6]];
        let mut mesh = loose(vertices.clone());
        let report = resolve(&mut mesh, &Boundaries::default(), VOXEL_MM);
        assert_eq!(report.vertices_moved, 0);
        assert_eq!(report.gave_up, 0);
        assert_eq!(mesh.vertices, vertices);
    }
}
