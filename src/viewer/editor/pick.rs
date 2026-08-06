//! Turning a pointer position into a ray, and that ray into the object it hits.
//!
//! The ray is built from the camera the frame was drawn with, so what is picked
//! is what is under the cursor whatever the viewport's aspect or the panel's
//! width. Every intersection is analytic against the same signed distance
//! primitives the configuration is written in - no depth buffer is read back
//! and nothing is rasterized for picking.

use crate::constants;
use crate::geometry::{
    Shape, Vec3, cross, difference, inner, is_unrotated, length, rotate_inverse, rotation_matrix,
    scale, sum, triangle_normal,
};
use crate::viewer::camera::OrbitCamera;
use crate::viewer::editor::state::{Selection, Target};

/// A ray in world space.
///
/// Compared for equality by the editor's hover path, which re-picks only when
/// the ray under the pointer has actually moved - which is either the pointer
/// or the camera having moved, and neither costs a pick when neither did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// Where the ray starts, in millimetres.
    pub origin: Vec3,
    /// Unit direction.
    pub direction: Vec3,
}

impl Ray {
    /// The point at parameter `t`.
    pub fn at(&self, t: f64) -> Vec3 {
        sum(self.origin, scale(self.direction, t))
    }
}

/// The ray through a pointer position, in physical pixels from the top left of
/// a `width` x `height` viewport.
///
/// The projection is the camera's own: a vertical field of view of
/// [`constants::VIEW_FOV_DEGREES`] and a horizontal one widened by the aspect
/// ratio, which is exactly what [`OrbitCamera::view_projection`] builds.
pub fn ray_through(camera: &OrbitCamera, x: f64, y: f64, width: f64, height: f64) -> Ray {
    let (forward, side, up) = camera.view_basis();
    let aspect = if height > 0.0 { width / height } else { 1.0 };
    let tangent = (0.5 * constants::VIEW_FOV_DEGREES.to_radians()).tan();
    // Pixel centres, so the middle of the viewport is the middle of the frame.
    let ndc_x = if width > 0.0 {
        2.0 * (x + 0.5) / width - 1.0
    } else {
        0.0
    };
    let ndc_y = if height > 0.0 {
        1.0 - 2.0 * (y + 0.5) / height
    } else {
        0.0
    };
    let direction = [
        forward[0] + ndc_x * aspect * tangent * side[0] + ndc_y * tangent * up[0],
        forward[1] + ndc_x * aspect * tangent * side[1] + ndc_y * tangent * up[1],
        forward[2] + ndc_x * aspect * tangent * side[2] + ndc_y * tangent * up[2],
    ];
    let len = length(direction);
    Ray {
        origin: camera.eye(),
        direction: if len > 0.0 {
            scale(direction, 1.0 / len)
        } else {
            forward
        },
    }
}

/// Nearest non-negative hit of an axis aligned box, by the slab method.
pub fn hit_box(ray: &Ray, min: Vec3, max: Vec3) -> Option<f64> {
    nearest_ahead(box_span(ray, min, max)?)
}

/// Where `ray` enters and leaves an axis aligned box, by the slab method;
/// `None` when it misses it altogether. Either parameter can be negative, which
/// is a box the ray starts inside of or behind.
fn box_span(ray: &Ray, min: Vec3, max: Vec3) -> Option<[f64; 2]> {
    let mut near = f64::NEG_INFINITY;
    let mut far = f64::INFINITY;
    for d in 0..3 {
        if ray.direction[d].abs() <= 0.0 {
            if ray.origin[d] < min[d] || ray.origin[d] > max[d] {
                return None;
            }
            continue;
        }
        let inverse = 1.0 / ray.direction[d];
        let mut t0 = (min[d] - ray.origin[d]) * inverse;
        let mut t1 = (max[d] - ray.origin[d]) * inverse;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        near = near.max(t0);
        far = far.min(t1);
        if near > far {
            return None;
        }
    }
    Some([near, far])
}

/// Nearest non-negative hit of a box turned about its own centre.
///
/// The ray is taken into the box's own frame - inverse-rotate both its origin
/// and its direction about the centre - and the axis aligned slab test above
/// answers there. A rotation preserves lengths and the direction stays a unit
/// vector, so the parameter that comes back is the same `t` in world space:
/// this is exact, and it costs one matrix per shape rather than a march.
pub fn hit_rotated_box(ray: &Ray, min: Vec3, max: Vec3, rotation_deg: Vec3) -> Option<f64> {
    if is_unrotated(rotation_deg) {
        return hit_box(ray, min, max);
    }
    let center = Shape::box_center(min, max);
    let matrix = rotation_matrix(rotation_deg);
    let local = Ray {
        origin: sum(
            center,
            rotate_inverse(&matrix, difference(ray.origin, center)),
        ),
        direction: rotate_inverse(&matrix, ray.direction),
    };
    hit_box(&local, min, max)
}

/// Nearest non-negative hit of a sphere.
pub fn hit_sphere(ray: &Ray, center: Vec3, radius: f64) -> Option<f64> {
    if radius <= 0.0 {
        return None;
    }
    let offset = difference(ray.origin, center);
    // The direction is a unit vector, so the quadratic's leading term is one.
    let half_b = inner(offset, ray.direction);
    let c = inner(offset, offset) - radius * radius;
    let discriminant = half_b * half_b - c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    nearest_ahead([-half_b - root, -half_b + root])
}

/// Nearest non-negative hit of an ellipsoid turned about its own centre.
///
/// Exact, and analytic like the rest of them. Two changes of coordinates take
/// it to a unit sphere, and the second one is not rigid, which is the only
/// thing to be careful about:
///
/// 1. **Into the ellipsoid's frame.** `a0 = R^T (o - c)` and `ad = R^T d`. A
///    rotation preserves lengths, so `ad` is still a unit vector and `t` still
///    measures millimetres - this is what [`hit_rotated_box`] does.
/// 2. **Into the unit sphere's frame.** Divide both by the radii component-wise:
///    `a = a0 / r`, `b = ad / r`. This *is* a scaling, so `b` is not a unit
///    vector and lengths in this frame are not millimetres. What survives is the
///    parameter itself: the map is linear, so the scaled ray is `a + t * b` with
///    the **same** `t` that walks the original ray. Solving there and returning
///    `t` unaltered therefore returns millimetres along the world ray, and no
///    correction is needed - only the quadratic's leading coefficient `b . b`,
///    which for a sphere would have been one, has to be carried.
///
/// The surface is `|a + t b| = 1`, so
/// `(b.b) t^2 + 2 (a.b) t + (a.a - 1) = 0`.
pub fn hit_ellipsoid(ray: &Ray, center: Vec3, radii: Vec3, rotation_deg: Vec3) -> Option<f64> {
    if radii.iter().any(|r| *r <= 0.0) {
        return None;
    }
    let offset = difference(ray.origin, center);
    let (local_origin, local_direction) = if is_unrotated(rotation_deg) {
        (offset, ray.direction)
    } else {
        let matrix = rotation_matrix(rotation_deg);
        (
            rotate_inverse(&matrix, offset),
            rotate_inverse(&matrix, ray.direction),
        )
    };
    let a = [
        local_origin[0] / radii[0],
        local_origin[1] / radii[1],
        local_origin[2] / radii[2],
    ];
    let b = [
        local_direction[0] / radii[0],
        local_direction[1] / radii[1],
        local_direction[2] / radii[2],
    ];
    let leading = inner(b, b);
    if leading <= 0.0 {
        return None;
    }
    let half_b = inner(a, b);
    let c = inner(a, a) - 1.0;
    let discriminant = half_b * half_b - leading * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    nearest_ahead([(-half_b - root) / leading, (-half_b + root) / leading])
}

/// Nearest non-negative hit of a cylinder capped by the planes through `p1`
/// and `p2`.
pub fn hit_cylinder(ray: &Ray, p1: Vec3, p2: Vec3, radius: f64) -> Option<f64> {
    let axis = difference(p2, p1);
    let axis_length = length(axis);
    if radius <= 0.0 || axis_length <= 0.0 {
        return None;
    }
    let unit = scale(axis, 1.0 / axis_length);
    let offset = difference(ray.origin, p1);
    // Split the ray into the part along the axis and the part across it; the
    // radial part is what the infinite cylinder's quadratic is written in.
    let d_along = inner(ray.direction, unit);
    let o_along = inner(offset, unit);
    let d_radial = difference(ray.direction, scale(unit, d_along));
    let o_radial = difference(offset, scale(unit, o_along));
    let a = inner(d_radial, d_radial);
    let half_b = inner(o_radial, d_radial);
    let c = inner(o_radial, o_radial) - radius * radius;

    let mut best: Option<f64> = None;
    let mut keep = |t: f64| {
        if t >= 0.0 && best.is_none_or(|current| t < current) {
            best = Some(t);
        }
    };
    if a > 0.0 {
        let discriminant = half_b * half_b - a * c;
        if discriminant >= 0.0 {
            let root = discriminant.sqrt();
            for t in [(-half_b - root) / a, (-half_b + root) / a] {
                let height = o_along + t * d_along;
                if (0.0..=axis_length).contains(&height) {
                    keep(t);
                }
            }
        }
    }
    if d_along.abs() > 0.0 {
        for cap in [0.0, axis_length] {
            let t = (cap - o_along) / d_along;
            let radial = sum(o_radial, scale(d_radial, t));
            if inner(radial, radial) <= radius * radius {
                keep(t);
            }
        }
    }
    best
}

/// Nearest non-negative hit of a capped cone, in closed form.
///
/// The cylinder's own split - the ray taken apart into the piece along the axis
/// and the piece across it - with the constant radius replaced by the one that
/// walks linearly up the axis. The lateral surface is then the quadric
/// `|radial(t)|^2 = r(h(t))^2`, whose roots are kept where they fall between
/// the two caps, and each cap is the disc its plane cuts. A frustum is convex,
/// so the nearest of those roots is the hit - there is no second entry to
/// miss, which is what makes this a formula rather than the tube's march.
///
/// A ray running parallel to the slanted wall leaves no quadratic at all - the
/// leading coefficient vanishes - and is solved as the linear equation it is,
/// which is the one crossing such a ray really has.
pub fn hit_cone(ray: &Ray, p1: Vec3, p2: Vec3, radius1: f64, radius2: f64) -> Option<f64> {
    let axis = difference(p2, p1);
    let axis_length = length(axis);
    if radius1 <= 0.0 || radius2 < 0.0 || axis_length <= 0.0 {
        return None;
    }
    let unit = scale(axis, 1.0 / axis_length);
    let offset = difference(ray.origin, p1);
    let d_along = inner(ray.direction, unit);
    let o_along = inner(offset, unit);
    let d_radial = difference(ray.direction, scale(unit, d_along));
    let o_radial = difference(offset, scale(unit, o_along));
    // The radius at height h is `taper * h + base`, and the wall is where the
    // radial offset reaches it.
    let taper = (radius2 - radius1) / axis_length;
    let base = radius1 + taper * o_along;
    let a = inner(d_radial, d_radial) - taper * taper * d_along * d_along;
    let half_b = inner(o_radial, d_radial) - taper * d_along * base;
    let c = inner(o_radial, o_radial) - base * base;

    let mut best: Option<f64> = None;
    let mut keep = |t: f64| {
        if t >= 0.0 && best.is_none_or(|current| t < current) {
            best = Some(t);
        }
    };
    // The height a wall root has to fall between the two caps at, with the
    // tangency band's own slack either side: a root that lands exactly on a rim
    // - or on the apex, which every ray down a true cone's axis does - is one
    // rounding away from falling outside the range it is the boundary of.
    let slack = constants::VIEW_EDIT_PICK_TANGENT_EPS * axis_length;
    let mut on_wall = |t: f64| {
        let height = o_along + t * d_along;
        if (-slack..=axis_length + slack).contains(&height) {
            keep(t);
        }
    };
    if a.abs() > 0.0 {
        let discriminant = half_b * half_b - a * c;
        // A tangency is a discriminant of zero, and zero is what this
        // subtraction loses first: see
        // [`constants::VIEW_EDIT_PICK_TANGENT_EPS`], which is the band a ray
        // grazing the wall along its whole length falls in.
        let tangent = constants::VIEW_EDIT_PICK_TANGENT_EPS * (half_b * half_b + (a * c).abs());
        if discriminant >= -tangent {
            let root = discriminant.max(0.0).sqrt();
            for t in [(-half_b - root) / a, (-half_b + root) / a] {
                on_wall(t);
            }
        }
    } else if half_b != 0.0 {
        on_wall(-0.5 * c / half_b);
    }
    if d_along.abs() > 0.0 {
        // The apex of a true cone is a cap of no radius: a point, and one the
        // wall above already answers for.
        for (cap, radius) in [(0.0, radius1), (axis_length, radius2)] {
            if radius <= 0.0 {
                continue;
            }
            let t = (cap - o_along) / d_along;
            let radial = sum(o_radial, scale(d_radial, t));
            if inner(radial, radial) <= radius * radius {
                keep(t);
            }
        }
    }
    best
}

/// Nearest non-negative hit of a triangular prism, in closed form.
///
/// The box's slab method, on the five planes a prism is bounded by rather than
/// on three pairs: the two faces half a thickness either side of the triangle's
/// plane, and one side plane per edge, each facing away from the vertex it does
/// not touch. A prism is convex, so clipping the ray against every half space
/// in turn leaves the one interval it is inside for - and the near end of that
/// interval is where it entered.
pub fn hit_triangle(ray: &Ray, a: Vec3, b: Vec3, c: Vec3, thickness: f64) -> Option<f64> {
    let normal = triangle_normal(a, b, c)?;
    if thickness <= 0.0 {
        return None;
    }
    let half = 0.5 * thickness;
    let along = inner(a, normal);
    // One side plane per edge, facing away from the vertex it does not touch,
    // which is the direction the edge crossed with the normal points.
    let mut sides = [([0.0; 3], 0.0); 3];
    for (slot, (from, to)) in sides.iter_mut().zip([(a, b), (b, c), (c, a)]) {
        let side = cross(difference(to, from), normal);
        let len = length(side);
        if len <= 0.0 {
            return None;
        }
        let side = scale(side, 1.0 / len);
        *slot = (side, inner(from, side));
    }
    let planes = [
        (normal, along + half),
        (scale(normal, -1.0), half - along),
        sides[0],
        sides[1],
        sides[2],
    ];

    let mut near = f64::NEG_INFINITY;
    let mut far = f64::INFINITY;
    for (plane, limit) in planes {
        let denominator = inner(ray.direction, plane);
        let distance = limit - inner(ray.origin, plane);
        if denominator == 0.0 {
            // Parallel to this face: inside its slab for every `t`, or for
            // none at all.
            if distance < 0.0 {
                return None;
            }
            continue;
        }
        let t = distance / denominator;
        if denominator > 0.0 {
            far = far.min(t);
        } else {
            near = near.max(t);
        }
        if near > far {
            return None;
        }
    }
    nearest_ahead([near, far])
}

/// Nearest non-negative hit of a tube, by sphere tracing its own exact field.
///
/// The only primitive here that is not solved in closed form, and the reason is
/// the shape rather than the effort: a tube bent into an arc is **not convex**,
/// so a ray can enter it, leave it, cross the gap the curve encloses and enter
/// it again. There is no quadratic whose roots those are (the general
/// ray-torus problem is a quartic, and this is a piece of a torus with two
/// spherical ends), and a wrong root here is a click that selects the object
/// behind the one under the pointer.
///
/// So it is traced instead, which the tube is unusually well suited to because
/// its field is the **true** distance rather than a bound (see
/// [`crate::geometry::Shape::Tube`]):
///
/// 1. The ray is clipped to the tube's exact bounding box, so the march starts
///    at the surface of the box and can never run past the far side of it.
/// 2. From there it steps by the distance to the surface, which is the largest
///    step that provably cannot cross it - inside or outside, since stepping by
///    the distance from within is just as safe.
/// 3. It stops when it is within [`constants::VIEW_EDIT_TUBE_PICK_EPS_MM`] of
///    the surface, or after [`constants::VIEW_EDIT_TUBE_PICK_MAX_STEPS`] steps.
///
/// Every part of that is deterministic and bounded: the same ray and the same
/// tube give the same answer, on any machine, in a bounded number of distance
/// evaluations. A ray that grazes the surface, where the steps become
/// arbitrarily short, runs the step count out and reports a miss.
pub fn hit_tube(ray: &Ray, p1: Vec3, p2: Vec3, bend: Option<Vec3>, radius: f64) -> Option<f64> {
    if radius <= 0.0 {
        return None;
    }
    let shape = Shape::Tube {
        p1,
        p2,
        bend,
        radius,
    };
    let bounds = shape.bounds();
    let [near, far] = box_span(ray, bounds.min, bounds.max)?;
    if far < 0.0 {
        return None;
    }
    let mut t = near.max(0.0);
    for _ in 0..constants::VIEW_EDIT_TUBE_PICK_MAX_STEPS {
        let distance = shape.signed_distance(ray.at(t));
        if distance.abs() <= constants::VIEW_EDIT_TUBE_PICK_EPS_MM {
            return Some(t);
        }
        t += distance.abs();
        if t > far {
            return None;
        }
    }
    None
}

/// Nearest non-negative hit of a configuration shape.
pub fn hit_shape(ray: &Ray, shape: &Shape) -> Option<f64> {
    match *shape {
        Shape::Box {
            min,
            max,
            rotation_deg,
        } => hit_rotated_box(ray, min, max, rotation_deg),
        Shape::Cylinder { p1, p2, radius } => hit_cylinder(ray, p1, p2, radius),
        Shape::Sphere { center, radius } => hit_sphere(ray, center, radius),
        Shape::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } => hit_ellipsoid(ray, center, radii, rotation_deg),
        Shape::Tube {
            p1,
            p2,
            bend,
            radius,
        } => hit_tube(ray, p1, p2, bend, radius),
        Shape::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => hit_cone(ray, p1, p2, radius1, radius2),
        Shape::Triangle { a, b, c, thickness } => hit_triangle(ray, a, b, c, thickness),
    }
}

/// The object a click picks, or `None` for a click on empty space.
///
/// Objects are ranked by [`Selection::pick_rank`] first and by depth second,
/// so the shell everything else lives in never swallows the click, and ties
/// inside one rank are broken by list order - the order the tree shows.
pub fn nearest(ray: &Ray, targets: &[Target]) -> Option<Selection> {
    let mut best: Option<(u8, f64, Selection)> = None;
    for target in targets {
        let Some(t) = hit_shape(ray, &target.shape) else {
            continue;
        };
        let rank = target.selection.pick_rank();
        if best.is_none_or(|(current_rank, current_t, _)| (rank, t) < (current_rank, current_t)) {
            best = Some((rank, t, target.selection));
        }
    }
    best.map(|(_, _, selection)| selection)
}

/// The nearer of two roots that lies ahead of the ray's origin.
fn nearest_ahead(roots: [f64; 2]) -> Option<f64> {
    let [near, far] = roots;
    if near >= 0.0 {
        Some(near)
    } else if far >= 0.0 {
        Some(far)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;

    fn along(origin: Vec3, direction: Vec3) -> Ray {
        let len = length(direction);
        Ray {
            origin,
            direction: scale(direction, 1.0 / len),
        }
    }

    #[test]
    fn a_box_is_hit_on_its_near_face_and_missed_beside_it() {
        let (min, max) = ([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let ray = along([-5.0, 5.0, 5.0], [1.0, 0.0, 0.0]);
        let t = hit_box(&ray, min, max).expect("a hit");
        assert!((t - 5.0).abs() < 1e-9, "entered at {t}");
        assert!(hit_box(&along([-5.0, 25.0, 5.0], [1.0, 0.0, 0.0]), min, max).is_none());
        // A ray that starts inside leaves through the far face.
        let inside = hit_box(&along([5.0, 5.0, 5.0], [1.0, 0.0, 0.0]), min, max).expect("a hit");
        assert!((inside - 5.0).abs() < 1e-9);
        // A box behind the camera is not a hit at all.
        assert!(hit_box(&along([-5.0, 5.0, 5.0], [-1.0, 0.0, 0.0]), min, max).is_none());
        // And a ray parallel to a face outside the slab misses.
        assert!(hit_box(&along([-5.0, -1.0, 5.0], [1.0, 0.0, 0.0]), min, max).is_none());
    }

    /// The analytic hit against a turned box, checked against a brute force
    /// march along the ray of the box's own signed distance function: two
    /// completely different routes to the same surface.
    #[test]
    fn a_rotated_box_is_hit_where_a_march_of_its_own_field_finds_it() {
        let (min, max) = ([-6.0, -2.0, -3.0], [6.0, 2.0, 3.0]);
        for rotation in [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 45.0],
            [90.0, 0.0, 0.0],
            [15.0, -35.0, 62.0],
            [180.0, 180.0, 180.0],
        ] {
            let shape = Shape::Box {
                min,
                max,
                rotation_deg: rotation,
            };
            for direction in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.7, -0.4],
                [-0.3, -1.0, 0.6],
            ] {
                let origin = scale(direction, -40.0 / length(direction));
                let ray = along(origin, direction);
                let analytic = hit_shape(&ray, &shape);
                let marched = march(&ray, &shape);
                match (analytic, marched) {
                    (Some(a), Some(m)) => assert!(
                        (a - m).abs() < 1e-2,
                        "{rotation:?} along {direction:?}: {a} against a marched {m}"
                    ),
                    (None, None) => {}
                    other => panic!("{rotation:?} along {direction:?}: {other:?}"),
                }
            }

            // A ray aimed past the box misses whichever way it is turned; the
            // offset is beyond the half diagonal, so no rotation can reach it.
            let past = along([0.0, 40.0, 0.0], [0.0, -1.0, 0.0]);
            let wide = Ray {
                origin: [20.0, past.origin[1], 0.0],
                direction: past.direction,
            };
            assert!(hit_shape(&wide, &shape).is_none(), "{rotation:?}");
        }
    }

    /// Step of the brute force march below, and the distance it walks.
    const MARCH_STEP: f64 = 0.005;
    const MARCH_FAR: f64 = 80.0;

    /// The lowest value the shape's field takes along `ray`, which says how
    /// far inside the ray goes - or, when it is positive, how far it stayed
    /// outside.
    fn deepest(ray: &Ray, shape: &Shape) -> f64 {
        let mut lowest = f64::INFINITY;
        let mut t = 0.0;
        while t < MARCH_FAR {
            lowest = lowest.min(shape.signed_distance(ray.at(t)));
            t += MARCH_STEP;
        }
        lowest
    }

    /// The first point along `ray` at which the shape's own field goes
    /// negative, found by stepping and then bisecting. Slow and dumb on
    /// purpose: it shares no arithmetic with the analytic intersection.
    fn march(ray: &Ray, shape: &Shape) -> Option<f64> {
        let (step, far) = (MARCH_STEP, MARCH_FAR);
        let mut previous = 0.0;
        let mut t = 0.0;
        while t < far {
            if shape.signed_distance(ray.at(t)) <= 0.0 {
                let (mut low, mut high) = (previous, t);
                for _ in 0..60 {
                    let mid = 0.5 * (low + high);
                    if shape.signed_distance(ray.at(mid)) <= 0.0 {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }
                return Some(0.5 * (low + high));
            }
            previous = t;
            t += step;
        }
        None
    }

    /// Every handle volume, and every pick target, goes through `hit_shape`, so
    /// a turned box has to be reachable through the dispatcher too.
    #[test]
    fn picking_prefers_the_nearer_of_two_turned_boxes() {
        let targets = vec![
            Target {
                selection: Selection::Keepout(0),
                shape: Shape::Box {
                    min: [-2.0, -2.0, -2.0],
                    max: [2.0, 2.0, 2.0],
                    rotation_deg: [0.0, 0.0, 45.0],
                },
            },
            Target {
                selection: Selection::Keepin(0),
                shape: Shape::Box {
                    min: [-2.0, -2.0, 18.0],
                    max: [2.0, 2.0, 22.0],
                    rotation_deg: [0.0, 45.0, 0.0],
                },
            },
        ];
        let up = along([0.0, 0.0, -30.0], [0.0, 0.0, 1.0]);
        assert_eq!(nearest(&up, &targets), Some(Selection::Keepout(0)));
        let down = along([0.0, 0.0, 40.0], [0.0, 0.0, -1.0]);
        assert_eq!(nearest(&down, &targets), Some(Selection::Keepin(0)));
        // A turn puts a corner where a face used to be: a ray that would miss
        // the axis aligned box along the diagonal hits the turned one.
        let corner = along([2.5, 2.5, -30.0], [0.0, 0.0, 1.0]);
        assert!(hit_box(&corner, [-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]).is_none());
        let diagonal = along([2.5, 0.0, -30.0], [0.0, 0.0, 1.0]);
        assert!(hit_shape(&diagonal, &targets[0].shape).is_some());
    }

    /// The analytic ray-ellipsoid intersection against a brute force march of
    /// the ellipsoid's own field: two completely different routes to the same
    /// surface, over anisotropic radii, every rotation and grazing rays.
    #[test]
    fn an_ellipsoid_is_hit_where_a_march_of_its_own_field_finds_it() {
        let (center, radii) = ([1.0, -2.0, 3.0], [6.0, 2.0, 3.0]);
        for rotation in [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 45.0],
            [90.0, 0.0, 0.0],
            [15.0, -35.0, 62.0],
            [180.0, 180.0, 180.0],
        ] {
            let shape = Shape::Ellipsoid {
                center,
                radii,
                rotation_deg: rotation,
            };
            for direction in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.7, -0.4],
                [-0.3, -1.0, 0.6],
            ] {
                // From 40 mm out, aimed at the centre, and offset sideways in
                // steps that walk from the middle of the shape to well past it:
                // the last of them graze the surface and miss.
                let (u, v) = crate::viewer::tessellate::basis(direction);
                for offset in [0.0, 1.0, 1.9, 2.0, 2.05, 3.0, 6.5, 12.0] {
                    let from = sum(center, scale(direction, -40.0 / length(direction)));
                    let origin = sum(from, scale(sum(u, scale(v, 0.5)), offset));
                    let ray = along(origin, direction);
                    let analytic = hit_shape(&ray, &shape);
                    let what = format!("{rotation:?} along {direction:?} offset {offset}");
                    // How deep the ray goes decides what may be asserted. A ray
                    // that clearly enters must be found by both routes; one
                    // that clearly misses by neither; and a grazing one - which
                    // is where a marched sample can fall either side of a chord
                    // shorter than its own step - only has to come back on the
                    // surface.
                    let clear = 10.0 * MARCH_STEP;
                    let depth = deepest(&ray, &shape);
                    if depth < -clear {
                        let a = analytic.unwrap_or_else(|| panic!("{what}: no analytic hit"));
                        let m = march(&ray, &shape).unwrap_or_else(|| panic!("{what}: no march"));
                        assert!((a - m).abs() < 1e-2, "{what}: {a} against a marched {m}");
                    } else if depth > clear {
                        assert!(analytic.is_none(), "{what}: {analytic:?} on a clear miss");
                        assert!(march(&ray, &shape).is_none(), "{what}: the march hit");
                    }
                    // Whatever it reports, a hit is on the surface and ahead.
                    if let Some(t) = analytic {
                        assert!(t >= 0.0, "{what}: a hit behind the origin");
                        assert!(
                            shape.signed_distance(ray.at(t)).abs() < 1e-9,
                            "{what}: the hit is not on the surface"
                        );
                    }
                }
            }
        }
    }

    /// The analytic ray-cone intersection against a brute force march of the
    /// cone's own field: two completely different routes to the same surface,
    /// over a frustum, a true cone and a widening one, and rays that graze.
    #[test]
    fn a_cone_is_hit_where_a_march_of_its_own_field_finds_it() {
        let (p1, p2) = ([1.0, -2.0, 0.0], [1.0, -2.0, 12.0]);
        for (radius1, radius2) in [(5.0, 2.0), (5.0, 0.0), (2.0, 5.0), (3.0, 3.0)] {
            let shape = Shape::Cone {
                p1,
                p2,
                radius1,
                radius2,
            };
            let centre = [1.0, -2.0, 6.0];
            for direction in [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.7, -0.4],
                [-0.3, -1.0, 0.6],
            ] {
                let (u, v) = crate::viewer::tessellate::basis(direction);
                for offset in [0.0, 1.0, 2.5, 4.0, 4.9, 5.0, 5.1, 9.0] {
                    let from = sum(centre, scale(direction, -40.0 / length(direction)));
                    let origin = sum(from, scale(sum(u, scale(v, 0.5)), offset));
                    let ray = along(origin, direction);
                    let analytic = hit_shape(&ray, &shape);
                    let what = format!("({radius1}, {radius2}) along {direction:?} at {offset}");
                    let clear = 10.0 * MARCH_STEP;
                    let depth = deepest(&ray, &shape);
                    if depth < -clear {
                        let a = analytic.unwrap_or_else(|| panic!("{what}: no analytic hit"));
                        let m = march(&ray, &shape).unwrap_or_else(|| panic!("{what}: no march"));
                        assert!((a - m).abs() < 1e-2, "{what}: {a} against a marched {m}");
                    } else if depth > clear {
                        assert!(analytic.is_none(), "{what}: {analytic:?} on a clear miss");
                        assert!(march(&ray, &shape).is_none(), "{what}: the march hit");
                    }
                    if let Some(t) = analytic {
                        assert!(t >= 0.0, "{what}: a hit behind the origin");
                        assert!(
                            shape.signed_distance(ray.at(t)).abs() < 1e-9,
                            "{what}: the hit is not on the surface"
                        );
                    }
                }
            }
        }
    }

    /// The cases a taper has that a cylinder does not: the narrow end is
    /// narrow, the apex is a point, and a ray running parallel to the slanted
    /// wall leaves a linear equation rather than a quadratic.
    #[test]
    fn a_cone_is_hit_at_its_own_width_rather_than_a_cylinders() {
        let (p1, p2) = ([0.0, 0.0, 0.0], [0.0, 0.0, 10.0]);
        let (radius1, radius2) = (4.0, 0.0);
        // Straight down the axis onto the apex.
        let ray = along([0.0, 0.0, 20.0], [0.0, 0.0, -1.0]);
        let t = hit_cone(&ray, p1, p2, radius1, radius2).expect("a hit");
        assert!((t - 10.0).abs() < 1e-9, "hit at {t}");
        // Across it at half height, where it is half as wide: a cylinder of
        // radius 4 would be hit 3 mm sooner.
        let ray = along([-20.0, 0.0, 5.0], [1.0, 0.0, 0.0]);
        let t = hit_cone(&ray, p1, p2, radius1, radius2).expect("a hit");
        assert!((t - 18.0).abs() < 1e-9, "hit at {t}");
        assert!(hit_cylinder(&ray, p1, p2, 4.0).expect("a hit") < t);
        // And past the wall at that height, where the cylinder still hits and
        // the cone does not.
        let ray = along([-20.0, 3.0, 5.0], [1.0, 0.0, 0.0]);
        assert!(hit_cone(&ray, p1, p2, radius1, radius2).is_none());
        assert!(hit_cylinder(&ray, p1, p2, 4.0).is_some());
        // The base cap, from below.
        let ray = along([1.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
        assert!((hit_cone(&ray, p1, p2, radius1, radius2).expect("a hit") - 5.0).abs() < 1e-9);
        // A ray parallel to the slanted wall: no quadratic at all, and the cap
        // is what it crosses. The march agrees.
        let slant = along([2.0, 0.0, -5.0], [-4.0, 0.0, 10.0]);
        let t = hit_cone(&slant, p1, p2, radius1, radius2).expect("a hit");
        let shape = Shape::Cone {
            p1,
            p2,
            radius1,
            radius2,
        };
        let marched = march(&slant, &shape).expect("a march");
        assert!(
            (t - marched).abs() < 1e-2,
            "{t} against a marched {marched}"
        );
        // A degenerate cone is not pickable at all.
        assert!(hit_cone(&ray, p1, p1, radius1, radius2).is_none());
        assert!(hit_cone(&ray, p1, p2, 0.0, 1.0).is_none());
        assert!(hit_cone(&ray, p1, p2, 1.0, -1.0).is_none());
    }

    /// The analytic ray-prism intersection against a brute force march of the
    /// prism's own field, over a flat triangle and one turned out of every
    /// axis, with rays that pass beside its edges.
    #[test]
    fn a_triangular_prism_is_hit_where_a_march_of_its_own_field_finds_it() {
        for (a, b, c) in [
            ([0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [3.0, 8.0, 0.0]),
            ([1.0, 2.0, 3.0], [7.0, 3.0, 9.0], [2.0, 9.0, 5.0]),
            // The other winding, which must not turn the prism inside out.
            ([0.0, 0.0, 0.0], [3.0, 8.0, 0.0], [12.0, 0.0, 0.0]),
        ] {
            let shape = Shape::Triangle {
                a,
                b,
                c,
                thickness: 4.0,
            };
            let centre = crate::viewer::tessellate::centroid(&shape);
            for direction in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.7, -0.4],
                [-0.3, -1.0, 0.6],
            ] {
                let (u, v) = crate::viewer::tessellate::basis(direction);
                for offset in [0.0, 1.0, 2.0, 3.5, 4.0, 5.0, 8.0, 14.0] {
                    let from = sum(centre, scale(direction, -40.0 / length(direction)));
                    let origin = sum(from, scale(sum(u, scale(v, 0.5)), offset));
                    let ray = along(origin, direction);
                    let analytic = hit_shape(&ray, &shape);
                    let what = format!("{a:?} along {direction:?} at {offset}");
                    let clear = 10.0 * MARCH_STEP;
                    let depth = deepest(&ray, &shape);
                    if depth < -clear {
                        let hit = analytic.unwrap_or_else(|| panic!("{what}: no analytic hit"));
                        let m = march(&ray, &shape).unwrap_or_else(|| panic!("{what}: no march"));
                        assert!(
                            (hit - m).abs() < 1e-2,
                            "{what}: {hit} against a marched {m}"
                        );
                    } else if depth > clear {
                        assert!(analytic.is_none(), "{what}: {analytic:?} on a clear miss");
                        assert!(march(&ray, &shape).is_none(), "{what}: the march hit");
                    }
                    if let Some(t) = analytic {
                        assert!(t >= 0.0, "{what}: a hit behind the origin");
                        assert!(
                            shape.signed_distance(ray.at(t)).abs() < 1e-9,
                            "{what}: the hit is not on the surface"
                        );
                    }
                }
            }
        }
    }

    /// A prism is hit on its own five faces and missed beside every one of
    /// them, which is what the slab method has to get right for a shape whose
    /// sides are not axis aligned.
    #[test]
    fn a_triangular_prism_is_hit_on_its_faces_and_missed_beside_them() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [0.0, 9.0, 0.0]);
        let thickness = 4.0;
        // Down onto the top face, over a point that is inside the triangle.
        let ray = along([2.0, 2.0, 20.0], [0.0, 0.0, -1.0]);
        let t = hit_triangle(&ray, a, b, c, thickness).expect("a hit");
        assert!((t - 18.0).abs() < 1e-9, "hit at {t}");
        // Over a point outside the hypotenuse, which the bounding box contains
        // and the triangle does not.
        let ray = along([10.0, 8.0, 20.0], [0.0, 0.0, -1.0]);
        assert!(hit_triangle(&ray, a, b, c, thickness).is_none());
        assert!(hit_box(&ray, [0.0, 0.0, -2.0], [12.0, 9.0, 2.0]).is_some());
        // Through a side face, and past the end of it.
        let ray = along([-20.0, 4.0, 0.0], [1.0, 0.0, 0.0]);
        assert!((hit_triangle(&ray, a, b, c, thickness).expect("a hit") - 20.0).abs() < 1e-9);
        assert!(
            hit_triangle(
                &along([-20.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
                a,
                b,
                c,
                thickness
            )
            .is_none()
        );
        // From inside, the exit comes back.
        let inside = hit_triangle(&along([1.0, 1.0, 0.0], [0.0, 0.0, 1.0]), a, b, c, thickness)
            .expect("a hit");
        assert!((inside - 2.0).abs() < 1e-9, "left at {inside}");
        // Behind the ray is not a hit, and neither is a degenerate prism.
        assert!(
            hit_triangle(
                &along([2.0, 2.0, 20.0], [0.0, 0.0, 1.0]),
                a,
                b,
                c,
                thickness
            )
            .is_none()
        );
        assert!(hit_triangle(&ray, a, b, [24.0, 0.0, 0.0], thickness).is_none());
        assert!(hit_triangle(&ray, a, b, c, 0.0).is_none());
    }

    /// The scaling that takes the ellipsoid to a unit sphere is not rigid, so
    /// the parameter that comes back has to be checked against millimetres
    /// rather than against the scaled frame's own units.
    #[test]
    fn an_ellipsoid_hit_is_in_millimetres_along_the_original_ray() {
        let (center, radii) = ([0.0, 0.0, 0.0], [10.0, 2.0, 5.0]);
        // Straight down the long axis: the surface is 10 mm from the centre, so
        // a ray starting 30 mm out enters at 20. A frame scaled by the radii
        // would answer 2.
        let ray = along([-30.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let t = hit_ellipsoid(&ray, center, radii, [0.0; 3]).expect("a hit");
        assert!((t - 20.0).abs() < 1e-9, "entered at {t}");
        // And down the short one, where the same scaling would answer 15.
        let ray = along([0.0, -30.0, 0.0], [0.0, 1.0, 0.0]);
        let t = hit_ellipsoid(&ray, center, radii, [0.0; 3]).expect("a hit");
        assert!((t - 28.0).abs() < 1e-9, "entered at {t}");
        // From inside, the exit comes back, in millimetres too.
        let inside = hit_ellipsoid(&along([0.0; 3], [0.0, 0.0, 1.0]), center, radii, [0.0; 3])
            .expect("a hit");
        assert!((inside - 5.0).abs() < 1e-9, "left at {inside}");
        // A turn of 90 degrees about z swaps which axis is which.
        let ray = along([-30.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let t = hit_ellipsoid(&ray, center, radii, [0.0, 0.0, 90.0]).expect("a hit");
        assert!((t - 28.0).abs() < 1e-9, "entered at {t}");
        // Behind the ray is not a hit, and neither is a degenerate ellipsoid.
        assert!(
            hit_ellipsoid(
                &along([-30.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
                center,
                radii,
                [0.0; 3]
            )
            .is_none()
        );
        assert!(hit_ellipsoid(&ray, center, [10.0, 0.0, 5.0], [0.0; 3]).is_none());
        // Past the short axis, where a sphere of the long radius would be hit.
        assert!(
            hit_ellipsoid(
                &along([0.0, 4.0, 0.0], [1.0, 0.0, 0.0]),
                center,
                radii,
                [0.0; 3]
            )
            .is_none()
        );
    }

    /// A turned ellipsoid has to be reachable through the dispatcher and the
    /// ranking, like every other pick target.
    #[test]
    fn picking_reaches_a_turned_ellipsoid_through_the_dispatcher() {
        let targets = vec![
            Target {
                selection: Selection::Keepout(0),
                shape: Shape::Ellipsoid {
                    center: [0.0; 3],
                    radii: [6.0, 1.5, 1.5],
                    rotation_deg: [0.0, 0.0, 90.0],
                },
            },
            Target {
                selection: Selection::Keepin(0),
                shape: Shape::Ellipsoid {
                    center: [0.0, 0.0, 20.0],
                    radii: [2.0; 3],
                    rotation_deg: [0.0; 3],
                },
            },
        ];
        let up = along([0.0, 0.0, -30.0], [0.0, 0.0, 1.0]);
        assert_eq!(nearest(&up, &targets), Some(Selection::Keepout(0)));
        let down = along([0.0, 0.0, 40.0], [0.0, 0.0, -1.0]);
        assert_eq!(nearest(&down, &targets), Some(Selection::Keepin(0)));
        // The turn put the long axis on y: a ray 4 mm out along y hits, and the
        // same ray along x, where the ellipsoid is now 1.5 mm wide, misses.
        let on_y = along([0.0, 4.0, -30.0], [0.0, 0.0, 1.0]);
        assert!(hit_shape(&on_y, &targets[0].shape).is_some());
        let on_x = along([4.0, 0.0, -30.0], [0.0, 0.0, 1.0]);
        assert!(hit_shape(&on_x, &targets[0].shape).is_none());
    }

    /// The traced ray-tube intersection against a brute force march of the
    /// tube's own field: two routes to the same surface, straight and bent,
    /// from every direction and past every side of it.
    #[test]
    fn a_tube_is_hit_where_a_march_of_its_own_field_finds_it() {
        for bend in [None, Some([10.0, 6.0, 2.0]), Some([10.0, -8.0, 0.0])] {
            let shape = Shape::Tube {
                p1: [0.0, 0.0, 0.0],
                p2: [20.0, 0.0, 0.0],
                bend,
                radius: 2.5,
            };
            let centre = [10.0, 0.0, 0.0];
            for direction in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.7, -0.4],
                [-0.3, -1.0, 0.6],
            ] {
                let (u, v) = crate::viewer::tessellate::basis(direction);
                for offset in [0.0, 1.0, 2.4, 2.6, 4.0, 8.0, 15.0] {
                    let from = sum(centre, scale(direction, -40.0 / length(direction)));
                    let origin = sum(from, scale(sum(u, scale(v, 0.5)), offset));
                    let ray = along(origin, direction);
                    let traced = hit_shape(&ray, &shape);
                    let what = format!("{bend:?} along {direction:?} offset {offset}");
                    // A ray that clearly enters must be found both ways; one
                    // that clearly misses by neither. A grazing ray is the case
                    // the trace is allowed to give up on, so it only has to
                    // agree that whatever it did report is on the surface.
                    let clear = 10.0 * MARCH_STEP;
                    let depth = deepest(&ray, &shape);
                    if depth < -clear {
                        let t = traced.unwrap_or_else(|| panic!("{what}: no traced hit"));
                        let m = march(&ray, &shape).unwrap_or_else(|| panic!("{what}: no march"));
                        assert!((t - m).abs() < 1e-2, "{what}: {t} against a marched {m}");
                    } else if depth > clear {
                        assert!(traced.is_none(), "{what}: {traced:?} on a clear miss");
                        assert!(march(&ray, &shape).is_none(), "{what}: the march hit");
                    }
                    if let Some(t) = traced {
                        assert!(t >= 0.0, "{what}: a hit behind the origin");
                        assert!(
                            shape.signed_distance(ray.at(t)).abs()
                                <= constants::VIEW_EDIT_TUBE_PICK_EPS_MM,
                            "{what}: the hit is not on the surface"
                        );
                    }
                }
            }
        }
    }

    /// The case a closed form would have to be careful about and a trace simply
    /// is not: a bent tube is **not convex**, and a ray through the gap its arc
    /// encloses misses it - twice over, entering and leaving the box that holds
    /// it without ever touching the tube.
    #[test]
    fn a_ray_through_the_gap_a_bent_tube_encloses_misses_it() {
        // A half turn: the ends at either side, the arc bulging up over a gap
        // three radii deep in the middle.
        let shape = Shape::Tube {
            p1: [-10.0, 0.0, 0.0],
            p2: [10.0, 0.0, 0.0],
            bend: Some([0.0, 0.0, 10.0]),
            radius: 2.0,
        };
        let bounds = shape.bounds();
        // The gap is inside the bounding box, so the box is no answer at all.
        let through = along([0.0, -40.0, 5.0], [0.0, 1.0, 0.0]);
        assert!(hit_box(&through, bounds.min, bounds.max).is_some());
        assert!(
            hit_shape(&through, &shape).is_none(),
            "the gap under the arc is not the tube"
        );
        assert!(
            march(&through, &shape).is_none(),
            "nor does a march find it"
        );

        // Aimed at the arc over that gap, from the same direction, it does hit -
        // and the hit is on the surface rather than at the box.
        let over = along([0.0, -40.0, 9.5], [0.0, 1.0, 0.0]);
        let t = hit_shape(&over, &shape).expect("the top of the arc");
        assert!(shape.signed_distance(over.at(t)).abs() <= constants::VIEW_EDIT_TUBE_PICK_EPS_MM);
        let marched = march(&over, &shape).expect("a march");
        assert!(
            (t - marched).abs() < 1e-2,
            "{t} against a marched {marched}"
        );

        // Straight down the middle from above: the arc first, then the gap
        // below it, then nothing - a convex primitive could not do this.
        let down = along([0.0, 0.0, 40.0], [0.0, 0.0, -1.0]);
        let t = hit_shape(&down, &shape).expect("the top of the arc");
        assert!((t - 28.0).abs() < 1e-3, "entered at {t} rather than 28");
        // And past the ends of the arc there is nothing to hit at all.
        let beside = along([0.0, 0.0, 40.0], [0.0, 0.0, -1.0]);
        let wide = Ray {
            origin: [0.0, 5.0, 40.0],
            direction: beside.direction,
        };
        assert!(hit_shape(&wide, &shape).is_none());
        // A degenerate tube is not pickable.
        assert!(hit_tube(&down, [-10.0, 0.0, 0.0], [10.0, 0.0, 0.0], None, 0.0).is_none());
        // Nor is one behind the pointer.
        let behind = along([0.0, 0.0, 40.0], [0.0, 0.0, 1.0]);
        assert!(hit_shape(&behind, &shape).is_none());
    }

    /// A tube has to be reachable through the dispatcher and the ranking, like
    /// every other pick target.
    #[test]
    fn picking_reaches_a_bent_tube_through_the_dispatcher() {
        let targets = vec![
            Target {
                selection: Selection::Keepout(0),
                shape: Shape::Tube {
                    p1: [-6.0, 0.0, 0.0],
                    p2: [6.0, 0.0, 0.0],
                    bend: Some([0.0, 0.0, 3.0]),
                    radius: 1.5,
                },
            },
            Target {
                selection: Selection::Keepin(0),
                shape: Shape::Tube {
                    p1: [-6.0, 0.0, 20.0],
                    p2: [6.0, 0.0, 20.0],
                    bend: None,
                    radius: 1.5,
                },
            },
        ];
        let up = along([0.0, 0.0, -30.0], [0.0, 0.0, 1.0]);
        assert_eq!(nearest(&up, &targets), Some(Selection::Keepout(0)));
        let down = along([0.0, 0.0, 40.0], [0.0, 0.0, -1.0]);
        assert_eq!(nearest(&down, &targets), Some(Selection::Keepin(0)));
    }

    #[test]
    fn a_sphere_is_hit_at_its_surface() {
        let ray = along([0.0, 0.0, -20.0], [0.0, 0.0, 1.0]);
        let t = hit_sphere(&ray, [0.0, 0.0, 0.0], 4.0).expect("a hit");
        assert!((t - 16.0).abs() < 1e-9, "entered at {t}");
        // Tangent misses by a hair, and a miss is a miss.
        assert!(hit_sphere(&ray, [4.001, 0.0, 0.0], 4.0).is_none());
        assert!(hit_sphere(&ray, [0.0, 0.0, 0.0], 0.0).is_none());
        // From inside, the exit point comes back.
        let inside = hit_sphere(&along([0.0; 3], [1.0, 0.0, 0.0]), [0.0; 3], 4.0).expect("a hit");
        assert!((inside - 4.0).abs() < 1e-9);
    }

    #[test]
    fn a_capped_cylinder_is_hit_on_its_side_and_on_its_caps() {
        let (p1, p2, radius) = ([0.0, 0.0, 0.0], [0.0, 0.0, 10.0], 3.0);
        // Straight at the side.
        let side = hit_cylinder(&along([-10.0, 0.0, 5.0], [1.0, 0.0, 0.0]), p1, p2, radius)
            .expect("a hit");
        assert!((side - 7.0).abs() < 1e-9, "entered at {side}");
        // Down the axis, onto the far cap.
        let cap =
            hit_cylinder(&along([0.0, 0.0, -5.0], [0.0, 0.0, 1.0]), p1, p2, radius).expect("a hit");
        assert!((cap - 5.0).abs() < 1e-9, "entered at {cap}");
        // Past the end of the axis: the infinite cylinder would be hit here,
        // the capped one is not.
        assert!(
            hit_cylinder(&along([-10.0, 0.0, 15.0], [1.0, 0.0, 0.0]), p1, p2, radius).is_none()
        );
        // Outside the radius, and a degenerate cylinder.
        assert!(hit_cylinder(&along([-10.0, 4.0, 5.0], [1.0, 0.0, 0.0]), p1, p2, radius).is_none());
        assert!(hit_cylinder(&along([-10.0, 0.0, 5.0], [1.0, 0.0, 0.0]), p1, p1, radius).is_none());
    }

    #[test]
    fn the_nearest_object_wins_and_empty_space_selects_nothing() {
        let targets = vec![
            Target {
                selection: Selection::Keepout(0),
                shape: Shape::Sphere {
                    center: [0.0, 0.0, 0.0],
                    radius: 2.0,
                },
            },
            Target {
                selection: Selection::Keepin(0),
                shape: Shape::Sphere {
                    center: [0.0, 0.0, 20.0],
                    radius: 2.0,
                },
            },
        ];
        // From below, the first sphere is in front.
        let up = along([0.0, 0.0, -20.0], [0.0, 0.0, 1.0]);
        assert_eq!(nearest(&up, &targets), Some(Selection::Keepout(0)));
        // From above, the order reverses: it is depth that decides, not the
        // order of the list.
        let down = along([0.0, 0.0, 40.0], [0.0, 0.0, -1.0]);
        assert_eq!(nearest(&down, &targets), Some(Selection::Keepin(0)));
        // A ray that misses everything picks nothing.
        let past = along([50.0, 0.0, -20.0], [0.0, 0.0, 1.0]);
        assert_eq!(nearest(&past, &targets), None);
        assert_eq!(nearest(&up, &[]), None);
    }

    #[test]
    fn an_enclosing_shell_never_swallows_the_click() {
        let targets = vec![
            Target {
                selection: Selection::Domain(0),
                shape: Shape::axis_aligned_box([-50.0; 3], [50.0; 3]),
            },
            Target {
                selection: Selection::Keepout(0),
                shape: Shape::Sphere {
                    center: [0.0; 3],
                    radius: 2.0,
                },
            },
        ];
        // The domain is entered first by depth, and picked second by rank.
        let through = along([0.0, 0.0, -100.0], [0.0, 0.0, 1.0]);
        assert_eq!(nearest(&through, &targets), Some(Selection::Keepout(0)));
        // Where nothing else is, the domain is what the click means.
        let aside = along([40.0, 40.0, -100.0], [0.0, 0.0, 1.0]);
        assert_eq!(nearest(&aside, &targets), Some(Selection::Domain(0)));
    }

    #[test]
    fn the_pointer_ray_agrees_with_the_camera_it_was_drawn_with() {
        let mut camera = OrbitCamera::default();
        camera.fit(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [40.0, 20.0, 20.0],
            },
            1000.0 / 600.0,
        );
        let (width, height) = (1000.0, 600.0);

        // The centre pixel looks straight at the target.
        let centre = ray_through(
            &camera,
            width / 2.0 - 0.5,
            height / 2.0 - 0.5,
            width,
            height,
        );
        let to_target = difference(camera.target, camera.eye());
        let unit = scale(to_target, 1.0 / length(to_target));
        for d in 0..3 {
            assert!(
                (centre.direction[d] - unit[d]).abs() < 1e-9,
                "the centre pixel is not the view direction: {:?} against {unit:?}",
                centre.direction
            );
        }
        assert_eq!(centre.origin, camera.eye());

        // A ray cast at the projected position of a known point comes back to
        // that point: this is the round trip picking depends on.
        let point = [30.0, 5.0, 15.0];
        let matrix = camera.view_projection(width / height);
        let clip = crate::viewer::camera::transform_point(&matrix, point);
        let (ndc_x, ndc_y) = (clip[0] / clip[3], clip[1] / clip[3]);
        let px = (ndc_x + 1.0) * 0.5 * width - 0.5;
        let py = (1.0 - ndc_y) * 0.5 * height - 0.5;
        let ray = ray_through(&camera, px, py, width, height);
        let distance = length(difference(point, camera.eye()));
        let landed = ray.at(distance);
        for d in 0..3 {
            assert!(
                (landed[d] - point[d]).abs() < 1e-6,
                "the ray through {point:?} landed on {landed:?}"
            );
        }
        // And a tiny sphere at that point is what a click there picks.
        let targets = vec![Target {
            selection: Selection::Domain(0),
            shape: Shape::Sphere {
                center: point,
                radius: 0.5,
            },
        }];
        assert_eq!(nearest(&ray, &targets), Some(Selection::Domain(0)));
    }
}
