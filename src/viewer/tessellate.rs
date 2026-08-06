//! Analytic tessellation of the shapes the viewer draws as overlays.
//!
//! Keepout, keepin and support regions are signed distance primitives, so an
//! exact triangulation is cheaper and sharper than running marching cubes over
//! them. Loads have no geometry of their own and get procedural indicators: an
//! arrow for a force, a circular arc arrow for a torque.
//!
//! Every mesh here comes out closed and wound counter-clockwise seen from
//! outside, which is what the shaded renderer and the mesh validator expect.

use std::f64::consts::{PI, TAU};

use crate::constants;
use crate::geometry::{Shape, Vec3, cross, difference, length, scale, sum};
use crate::mesh::Mesh;

/// Append `source` to `target`, shifting its triangle indices into place.
pub fn append(target: &mut Mesh, source: &Mesh) {
    let offset = target.vertices.len() as u32;
    target.vertices.extend_from_slice(&source.vertices);
    target.triangles.extend(
        source
            .triangles
            .iter()
            .map(|t| [t[0] + offset, t[1] + offset, t[2] + offset]),
    );
}

fn normalize(v: Vec3) -> Vec3 {
    let len = length(v);
    if len > 0.0 { scale(v, 1.0 / len) } else { v }
}

/// Two unit vectors completing `axis` into a right handed basis
/// `(u, v, axis)`.
pub fn basis(axis: Vec3) -> (Vec3, Vec3) {
    let axis = normalize(axis);
    // Cross with whichever cardinal axis is least aligned with `axis`, so the
    // product never collapses.
    let (mut smallest, mut smallest_value) = (0, f64::INFINITY);
    for (d, value) in axis.iter().enumerate() {
        if value.abs() < smallest_value {
            smallest = d;
            smallest_value = value.abs();
        }
    }
    let mut helper = [0.0; 3];
    helper[smallest] = 1.0;
    let u = normalize(cross(helper, axis));
    let v = cross(axis, u);
    (u, v)
}

/// Axis aligned box.
pub fn box_mesh(min: Vec3, max: Vec3) -> Mesh {
    rotated_box_mesh(min, max, [0.0; 3])
}

/// Box turned about its own centre; see [`crate::geometry::rotation_matrix`].
///
/// A rotation is proper, so the winding of every face survives it and the mesh
/// comes out closed and outward facing exactly as the axis aligned one does.
pub fn rotated_box_mesh(min: Vec3, max: Vec3, rotation_deg: Vec3) -> Mesh {
    let vertices = if crate::geometry::is_unrotated(rotation_deg) {
        vec![
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
        ]
    } else {
        // `box_corners` is indexed by an (x, y, z) bit mask; the winding table
        // below walks the base face anticlockwise, so the order is remapped.
        let corners = Shape::box_corners(min, max, rotation_deg);
        [0, 1, 3, 2, 4, 5, 7, 6]
            .into_iter()
            .map(|k: usize| corners[k])
            .collect()
    };
    let triangles = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    Mesh {
        vertices,
        triangles,
    }
}

/// Cylinder capped by discs through `p1` and `p2`.
pub fn cylinder(p1: Vec3, p2: Vec3, radius: f64, segments: usize) -> Mesh {
    let mut mesh = Mesh::default();
    let axis = difference(p2, p1);
    if length(axis) <= 0.0 || radius <= 0.0 || segments < 3 {
        return mesh;
    }
    let (u, v) = basis(axis);
    for s in 0..segments {
        let angle = TAU * s as f64 / segments as f64;
        let (sin, cos) = angle.sin_cos();
        let offset = [
            radius * (cos * u[0] + sin * v[0]),
            radius * (cos * u[1] + sin * v[1]),
            radius * (cos * u[2] + sin * v[2]),
        ];
        mesh.vertices
            .push([p1[0] + offset[0], p1[1] + offset[1], p1[2] + offset[2]]);
        mesh.vertices
            .push([p2[0] + offset[0], p2[1] + offset[1], p2[2] + offset[2]]);
    }
    let low_center = mesh.vertices.len() as u32;
    mesh.vertices.push(p1);
    let high_center = mesh.vertices.len() as u32;
    mesh.vertices.push(p2);

    for s in 0..segments {
        let next = (s + 1) % segments;
        let (a, b) = (2 * s as u32, 2 * s as u32 + 1);
        let (c, d) = (2 * next as u32, 2 * next as u32 + 1);
        // Side quad, counter-clockwise seen from outside.
        mesh.triangles.push([a, c, d]);
        mesh.triangles.push([a, d, b]);
        // Caps: the far one faces along the axis, the near one against it.
        mesh.triangles.push([high_center, b, d]);
        mesh.triangles.push([low_center, c, a]);
    }
    mesh
}

/// Capped cone - the [`crate::geometry::Shape::Cone`] primitive - swept between
/// the discs of `radius1` at `p1` and `radius2` at `p2`.
///
/// The cylinder's construction with a radius per end, so a frustum whose two
/// radii are equal is that cylinder's own mesh. A `radius2` of zero is the true
/// cone: the second ring degenerates to the single apex vertex the side
/// triangles fan into, and the cap that would have closed it is dropped rather
/// than written as a ring of coincident points, so the surface stays closed and
/// carries no degenerate triangle either way.
///
/// Not to be confused with [`cone`] below, which is the **arrowhead** the
/// overlays are drawn with: a base, a direction and a height rather than a
/// configuration shape.
pub fn frustum(p1: Vec3, p2: Vec3, radius1: f64, radius2: f64, segments: usize) -> Mesh {
    let mut mesh = Mesh::default();
    let axis = difference(p2, p1);
    if length(axis) <= 0.0 || radius1 <= 0.0 || radius2 < 0.0 || segments < 3 {
        return mesh;
    }
    let (u, v) = basis(axis);
    // One ring per end, or the single point an end of no radius is. Returns
    // where it starts in the vertex list and how many points it has, which is
    // what lets one stitching rule cover both.
    let ring = |mesh: &mut Mesh, centre: Vec3, radius: f64| -> (u32, usize) {
        let start = mesh.vertices.len() as u32;
        if radius <= 0.0 {
            mesh.vertices.push(centre);
            return (start, 1);
        }
        for s in 0..segments {
            let angle = TAU * s as f64 / segments as f64;
            let (sin, cos) = angle.sin_cos();
            mesh.vertices.push([
                centre[0] + radius * (cos * u[0] + sin * v[0]),
                centre[1] + radius * (cos * u[1] + sin * v[1]),
                centre[2] + radius * (cos * u[2] + sin * v[2]),
            ]);
        }
        (start, segments)
    };
    let low = ring(&mut mesh, p1, radius1);
    let high = ring(&mut mesh, p2, radius2);
    let low_center = mesh.vertices.len() as u32;
    mesh.vertices.push(p1);
    let high_center = (radius2 > 0.0).then(|| {
        let index = mesh.vertices.len() as u32;
        mesh.vertices.push(p2);
        index
    });

    let slot = |(start, count): (u32, usize), s: usize| -> u32 {
        if count == 1 { start } else { start + s as u32 }
    };
    for s in 0..segments {
        let next = (s + 1) % segments;
        let (a, c) = (slot(low, s), slot(low, next));
        let (b, d) = (slot(high, s), slot(high, next));
        // Side quad, counter-clockwise seen from outside; an apex leaves one
        // triangle of the two, which is the fan.
        if a != c {
            mesh.triangles.push([a, c, d]);
        }
        if b != d {
            mesh.triangles.push([a, d, b]);
        }
        // The near cap faces against the axis, the far one along it.
        mesh.triangles.push([low_center, c, a]);
        if let Some(high_center) = high_center {
            mesh.triangles.push([high_center, b, d]);
        }
    }
    mesh
}

/// Triangular prism - the [`crate::geometry::Shape::Triangle`] primitive - the
/// triangle through `a`, `b` and `c` extruded `thickness` millimetres
/// symmetrically about its own plane.
///
/// Two triangular caps and a quad per edge, six vertices in all. The caps carry
/// the winding the triangle's own normal is defined by, one of them reversed,
/// and every side quad is wound to face away from the third vertex, so the
/// surface is closed and outward facing whichever way round the three points
/// were written.
pub fn triangle_prism(a: Vec3, b: Vec3, c: Vec3, thickness: f64) -> Mesh {
    let mut mesh = Mesh::default();
    let Some(normal) = crate::geometry::triangle_normal(a, b, c) else {
        return mesh;
    };
    if thickness <= 0.0 {
        return mesh;
    }
    let offset = scale(normal, 0.5 * thickness);
    for corner in [a, b, c] {
        mesh.vertices.push(difference(corner, offset));
    }
    for corner in [a, b, c] {
        mesh.vertices.push(sum(corner, offset));
    }
    // The cap on the normal's side keeps the triangle's own winding; the one
    // behind it is that winding reversed, because it faces the other way.
    mesh.triangles.push([0, 2, 1]);
    mesh.triangles.push([3, 4, 5]);
    for s in 0..3u32 {
        let next = (s + 1) % 3;
        mesh.triangles.push([s, next, next + 3]);
        mesh.triangles.push([s, next + 3, s + 3]);
    }
    mesh
}

/// Cone with its base disc at `base` and its apex `height` along `axis`.
///
/// The **arrowhead** the overlays are drawn with - a load vector's head, the
/// tip of a gizmo arrow - and not the [`crate::geometry::Shape::Cone`]
/// primitive, which is [`frustum`] above.
pub fn cone(base: Vec3, axis: Vec3, height: f64, radius: f64, segments: usize) -> Mesh {
    let mut mesh = Mesh::default();
    if height <= 0.0 || radius <= 0.0 || segments < 3 || length(axis) <= 0.0 {
        return mesh;
    }
    let axis = normalize(axis);
    let (u, v) = basis(axis);
    for s in 0..segments {
        let angle = TAU * s as f64 / segments as f64;
        let (sin, cos) = angle.sin_cos();
        mesh.vertices.push([
            base[0] + radius * (cos * u[0] + sin * v[0]),
            base[1] + radius * (cos * u[1] + sin * v[1]),
            base[2] + radius * (cos * u[2] + sin * v[2]),
        ]);
    }
    let center = mesh.vertices.len() as u32;
    mesh.vertices.push(base);
    let apex = mesh.vertices.len() as u32;
    mesh.vertices.push([
        base[0] + height * axis[0],
        base[1] + height * axis[1],
        base[2] + height * axis[2],
    ]);

    for s in 0..segments {
        let a = s as u32;
        let b = ((s + 1) % segments) as u32;
        mesh.triangles.push([a, b, apex]);
        mesh.triangles.push([center, b, a]);
    }
    mesh
}

/// Latitude/longitude sphere.
pub fn sphere(center: Vec3, radius: f64, segments: usize, rings: usize) -> Mesh {
    let mut mesh = Mesh::default();
    if radius <= 0.0 || segments < 3 || rings < 2 {
        return mesh;
    }
    // Index 0 is the north pole, the last index the south pole, and ring `i`
    // (1 <= i < rings) occupies the `segments` slots after them.
    mesh.vertices
        .push([center[0], center[1], center[2] + radius]);
    for i in 1..rings {
        let polar = std::f64::consts::PI * i as f64 / rings as f64;
        let (sin_polar, cos_polar) = polar.sin_cos();
        for j in 0..segments {
            let azimuth = TAU * j as f64 / segments as f64;
            let (sin_az, cos_az) = azimuth.sin_cos();
            mesh.vertices.push([
                center[0] + radius * sin_polar * cos_az,
                center[1] + radius * sin_polar * sin_az,
                center[2] + radius * cos_polar,
            ]);
        }
    }
    let south = mesh.vertices.len() as u32;
    mesh.vertices
        .push([center[0], center[1], center[2] - radius]);

    let index = |i: usize, j: usize| -> u32 {
        if i == 0 {
            0
        } else if i == rings {
            south
        } else {
            1 + ((i - 1) * segments + j % segments) as u32
        }
    };
    for i in 0..rings {
        for j in 0..segments {
            let a = index(i, j);
            let b = index(i + 1, j);
            let c = index(i + 1, j + 1);
            let d = index(i, j + 1);
            if b != c {
                mesh.triangles.push([a, b, c]);
            }
            if a != d {
                mesh.triangles.push([a, c, d]);
            }
        }
    }
    mesh
}

/// Ellipsoid: the latitude/longitude sphere above, scaled by the radii along
/// its own axes and then turned about its centre.
///
/// The unit sphere is what is tessellated, so the ellipsoid inherits its
/// topology exactly - closed, and one quad band per ring. Scaling by three
/// positive radii and a proper rotation both preserve orientation, so every
/// face still faces out and the mesh is as closed as the sphere it came from.
pub fn ellipsoid(
    center: Vec3,
    radii: Vec3,
    rotation_deg: Vec3,
    segments: usize,
    rings: usize,
) -> Mesh {
    if radii.iter().any(|r| *r <= 0.0) {
        return Mesh::default();
    }
    let mut mesh = sphere([0.0; 3], 1.0, segments, rings);
    let turned = !crate::geometry::is_unrotated(rotation_deg);
    let matrix = crate::geometry::rotation_matrix(rotation_deg);
    for vertex in &mut mesh.vertices {
        let stretched = [
            vertex[0] * radii[0],
            vertex[1] * radii[1],
            vertex[2] * radii[2],
        ];
        let placed = if turned {
            crate::geometry::rotate(&matrix, stretched)
        } else {
            stretched
        };
        *vertex = sum(center, placed);
    }
    mesh
}

/// One ring of a tube's surface: the circle of points a radius out from
/// `centre`, tilted by `alpha` towards the tangent, which is what sweeps the
/// hemispheres of the end caps out of the same construction as the barrel.
///
/// `plane` is the ring's zero angle and is the same vector at every ring of one
/// tube, so consecutive rings cannot twist against each other. Returns where
/// the ring starts in the vertex list and how many points it has.
fn tube_ring(
    mesh: &mut Mesh,
    centre: Vec3,
    plane: Vec3,
    tangent: Vec3,
    radius: f64,
    alpha: f64,
    segments: usize,
) -> (u32, usize) {
    let start = mesh.vertices.len() as u32;
    let (sin_alpha, cos_alpha) = alpha.sin_cos();
    let side = cross(tangent, plane);
    let axial = scale(tangent, -radius * sin_alpha);
    for s in 0..segments {
        let angle = TAU * s as f64 / segments as f64;
        let (sin, cos) = angle.sin_cos();
        let radial = sum(scale(plane, cos), scale(side, sin));
        mesh.vertices
            .push(sum(centre, sum(scale(radial, radius * cos_alpha), axial)));
    }
    (start, segments)
}

/// One end point of a tube's centre line, pushed out by a radius along the
/// curve: the pole its cap closes on.
fn tube_pole(mesh: &mut Mesh, centre: Vec3, offset: Vec3) -> (u32, usize) {
    let start = mesh.vertices.len() as u32;
    mesh.vertices.push(sum(centre, offset));
    (start, 1)
}

/// Tube: everything within `radius` of the centre line through `p1`, `p2` and
/// an optional `bend`.
///
/// One closed surface rather than a barrel with two spheres dropped on the
/// ends: the rings of the caps are built in the same frame the barrel's are, at
/// a tilt that walks from the equator to the pole, so the whole thing is a
/// single sweep and every vertex of it is exactly a radius from the curve -
/// which is to say exactly on the surface the field describes.
///
/// A tube fatter than the arc it bends round folds through itself on the inside
/// of the bend. The field is still exact - it is a distance to a curve, and
/// nothing about that cares - so what is voxelized, picked and exported is
/// right; it is only this overlay that reads as a knot.
pub fn tube(
    p1: Vec3,
    p2: Vec3,
    bend: Option<Vec3>,
    radius: f64,
    segments: usize,
    arc_segments: usize,
    cap_rings: usize,
) -> Mesh {
    let mut mesh = Mesh::default();
    if radius <= 0.0 || segments < 3 || arc_segments < 1 || cap_rings < 1 {
        return mesh;
    }
    // The centre line, sampled, with a unit tangent at each sample. `plane` is
    // the normal of the plane the arc bends in, which is perpendicular to every
    // one of those tangents - a straight tube has no such plane and takes any
    // perpendicular.
    let (centres, tangents, plane) = match crate::geometry::tube_arc(p1, p2, bend) {
        Some(arc) => {
            let mut centres = Vec::with_capacity(arc_segments + 1);
            let mut tangents = Vec::with_capacity(arc_segments + 1);
            for i in 0..=arc_segments {
                let angle = arc.span * i as f64 / arc_segments as f64;
                let (sin, cos) = angle.sin_cos();
                centres.push(arc.point(angle));
                tangents.push(normalize(sum(scale(arc.start, -sin), scale(arc.turn, cos))));
            }
            (centres, tangents, arc.normal)
        }
        None => {
            let axis = difference(p2, p1);
            if length(axis) <= 0.0 {
                // A capsule of no length is the ball its two caps make, and a
                // sphere is what that is: two caps' worth of rings.
                return sphere(p1, radius, segments, 2 * cap_rings);
            }
            let unit = normalize(axis);
            (vec![p1, p2], vec![unit, unit], basis(axis).0)
        }
    };

    let last = centres.len() - 1;
    let mut rings: Vec<(u32, usize)> = Vec::with_capacity(2 * cap_rings + centres.len());
    rings.push(tube_pole(
        &mut mesh,
        centres[0],
        scale(tangents[0], -radius),
    ));
    for j in 1..cap_rings {
        let alpha = 0.5 * PI * (1.0 - j as f64 / cap_rings as f64);
        rings.push(tube_ring(
            &mut mesh,
            centres[0],
            plane,
            tangents[0],
            radius,
            alpha,
            segments,
        ));
    }
    for (centre, tangent) in centres.iter().zip(tangents.iter()) {
        rings.push(tube_ring(
            &mut mesh, *centre, plane, *tangent, radius, 0.0, segments,
        ));
    }
    for j in 1..cap_rings {
        let alpha = -0.5 * PI * (j as f64 / cap_rings as f64);
        rings.push(tube_ring(
            &mut mesh,
            centres[last],
            plane,
            tangents[last],
            radius,
            alpha,
            segments,
        ));
    }
    rings.push(tube_pole(
        &mut mesh,
        centres[last],
        scale(tangents[last], radius),
    ));

    // Every ring is parameterized the same way round and they are in order
    // along the sweep, so one rule stitches the barrel and both caps; a pole is
    // the ring of one point that turns its own band into a fan.
    let slot = |(start, count): (u32, usize), s: usize| -> u32 {
        if count == 1 { start } else { start + s as u32 }
    };
    for pair in rings.windows(2) {
        for s in 0..segments {
            let next = (s + 1) % segments;
            let (lower, lower_next) = (slot(pair[0], s), slot(pair[0], next));
            let (upper, upper_next) = (slot(pair[1], s), slot(pair[1], next));
            if lower != lower_next {
                mesh.triangles.push([lower, lower_next, upper_next]);
            }
            if upper_next != upper {
                mesh.triangles.push([lower, upper_next, upper]);
            }
        }
    }
    mesh
}

/// Tessellate a signed distance primitive at the viewer's standard resolution.
pub fn shape(shape: &Shape) -> Mesh {
    match *shape {
        Shape::Box {
            min,
            max,
            rotation_deg,
        } => rotated_box_mesh(min, max, rotation_deg),
        Shape::Cylinder { p1, p2, radius } => {
            cylinder(p1, p2, radius, constants::VIEW_CYLINDER_SEGMENTS)
        }
        Shape::Sphere { center, radius } => sphere(
            center,
            radius,
            constants::VIEW_SPHERE_SEGMENTS,
            constants::VIEW_SPHERE_RINGS,
        ),
        // The counts a sphere gets: it is the same surface, stretched.
        Shape::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } => ellipsoid(
            center,
            radii,
            rotation_deg,
            constants::VIEW_SPHERE_SEGMENTS,
            constants::VIEW_SPHERE_RINGS,
        ),
        // The cylinder's count around it, because it is the same barrel, and
        // half a sphere's rings per cap, because that is what a cap is.
        Shape::Tube {
            p1,
            p2,
            bend,
            radius,
        } => tube(
            p1,
            p2,
            bend,
            radius,
            constants::VIEW_CYLINDER_SEGMENTS,
            constants::VIEW_TUBE_ARC_SEGMENTS,
            constants::VIEW_TUBE_CAP_RINGS,
        ),
        // The cylinder's count around it, because it is the same barrel with a
        // radius per end.
        Shape::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => frustum(p1, p2, radius1, radius2, constants::VIEW_CYLINDER_SEGMENTS),
        // Flat faces all the way round: eight triangles, and no count to pick.
        Shape::Triangle { a, b, c, thickness } => triangle_prism(a, b, c, thickness),
    }
}

/// Geometric centre of a primitive, where a load indicator is anchored.
pub fn centroid(shape: &Shape) -> Vec3 {
    match *shape {
        // The centre is the point a box turns about, so it is where it is
        // whatever rotation the box carries.
        Shape::Box { min, max, .. } => Shape::box_center(min, max),
        Shape::Cylinder { p1, p2, .. } => [
            0.5 * (p1[0] + p2[0]),
            0.5 * (p1[1] + p2[1]),
            0.5 * (p1[2] + p2[2]),
        ],
        Shape::Sphere { center, .. } => center,
        // As with a box, the centre is the point it turns about.
        Shape::Ellipsoid { center, .. } => center,
        // The middle of the *curve*, which on a bent tube is not the middle of
        // the line between its ends: an indicator hangs off the tube rather
        // than off the chord it spans.
        Shape::Tube { p1, p2, bend, .. } => match crate::geometry::tube_arc(p1, p2, bend) {
            Some(arc) => arc.point(0.5 * arc.span),
            None => [
                0.5 * (p1[0] + p2[0]),
                0.5 * (p1[1] + p2[1]),
                0.5 * (p1[2] + p2[2]),
            ],
        },
        // The middle of the axis, which is the cylinder's answer and the point
        // the gizmo anchors a cone on. Deliberately not the centre of *mass*,
        // which a taper pulls towards the wide end: an indicator that hung
        // there would hang somewhere neither the shape's own numbers nor its
        // handles name.
        Shape::Cone { p1, p2, .. } => [
            0.5 * (p1[0] + p2[0]),
            0.5 * (p1[1] + p2[1]),
            0.5 * (p1[2] + p2[2]),
        ],
        // The centroid of the face, which the symmetric extrusion makes the
        // centroid of the prism as well.
        Shape::Triangle { a, b, c, .. } => [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ],
    }
}

/// Arrow of total length `length_mm` whose tip sits at `tip` and which points
/// along `direction`.
pub fn arrow(tip: Vec3, direction: Vec3, length_mm: f64) -> Mesh {
    let mut mesh = Mesh::default();
    if length_mm <= 0.0 || length(direction) <= 0.0 {
        return mesh;
    }
    let axis = normalize(direction);
    let head_length = length_mm * constants::VIEW_ARROW_HEAD_LENGTH_FRACTION;
    let head_radius = length_mm * constants::VIEW_ARROW_HEAD_RADIUS_FRACTION;
    let shaft_radius = length_mm * constants::VIEW_ARROW_SHAFT_RADIUS_FRACTION;
    let along = |distance: f64| -> Vec3 {
        [
            tip[0] - distance * axis[0],
            tip[1] - distance * axis[1],
            tip[2] - distance * axis[2],
        ]
    };
    let tail = along(length_mm);
    let head_base = along(head_length);
    append(
        &mut mesh,
        &cylinder(
            tail,
            head_base,
            shaft_radius,
            constants::VIEW_CYLINDER_SEGMENTS,
        ),
    );
    append(
        &mut mesh,
        &cone(
            head_base,
            axis,
            head_length,
            head_radius,
            constants::VIEW_CYLINDER_SEGMENTS,
        ),
    );
    mesh
}

/// Circular arc arrow of radius `radius_mm` around `axis_dir` through
/// `axis_point`. The sweep follows the right hand rule when `sense` is
/// positive and runs the other way when it is negative.
pub fn torque_arc(axis_point: Vec3, axis_dir: Vec3, radius_mm: f64, sense: f64) -> Mesh {
    arc(
        axis_point,
        axis_dir,
        radius_mm,
        constants::VIEW_TORQUE_ARC_SWEEP_DEGREES * sense.signum(),
        radius_mm * constants::VIEW_TORQUE_TUBE_RADIUS_FRACTION,
        radius_mm * constants::VIEW_TORQUE_HEAD_LENGTH_FRACTION,
        radius_mm * constants::VIEW_TORQUE_HEAD_RADIUS_FRACTION,
        if sense == 0.0 {
            0
        } else {
            constants::VIEW_TORQUE_ARC_SEGMENTS
        },
    )
}

/// A curved arrow of `sweep_deg` degrees about `axis` through `center`, for the
/// editor's rotation handles: the same construction as a torque arc, sized by
/// its caller rather than by the load overlay's own constants.
pub fn arc_arrow(
    center: Vec3,
    axis: Vec3,
    radius_mm: f64,
    sweep_deg: f64,
    tube_radius: f64,
) -> Mesh {
    arc(
        center,
        axis,
        radius_mm,
        sweep_deg,
        tube_radius,
        constants::VIEW_EDIT_ROTATE_ARC_HEAD_LENGTH_FACTOR * tube_radius,
        constants::VIEW_EDIT_ROTATE_ARC_HEAD_RADIUS_FACTOR * tube_radius,
        constants::VIEW_EDIT_ROTATE_ARC_SEGMENTS,
    )
}

/// A circular arc arrow: short tubes along the arc and a cone at its end.
#[allow(clippy::too_many_arguments)]
fn arc(
    axis_point: Vec3,
    axis_dir: Vec3,
    radius_mm: f64,
    sweep_deg: f64,
    tube_radius: f64,
    head_length: f64,
    head_radius: f64,
    segments: usize,
) -> Mesh {
    let mut mesh = Mesh::default();
    if radius_mm <= 0.0 || sweep_deg == 0.0 || segments < 2 || length(axis_dir) <= 0.0 {
        return mesh;
    }
    let (u, v) = basis(axis_dir);
    let sweep = sweep_deg.to_radians();
    let point = |angle: f64| -> Vec3 {
        let (sin, cos) = angle.sin_cos();
        [
            axis_point[0] + radius_mm * (cos * u[0] + sin * v[0]),
            axis_point[1] + radius_mm * (cos * u[1] + sin * v[1]),
            axis_point[2] + radius_mm * (cos * u[2] + sin * v[2]),
        ]
    };
    let mut previous = point(0.0);
    for s in 1..=segments {
        let current = point(sweep * s as f64 / segments as f64);
        append(
            &mut mesh,
            &cylinder(
                previous,
                current,
                tube_radius,
                constants::VIEW_ARC_TUBE_SEGMENTS,
            ),
        );
        previous = current;
    }
    let last = point(sweep * (segments - 1) as f64 / segments as f64);
    append(
        &mut mesh,
        &cone(
            previous,
            difference(previous, last),
            head_length,
            head_radius,
            constants::VIEW_CYLINDER_SEGMENTS,
        ),
    );
    mesh
}

/// A dimension line: a thin tube from `from` to `to` with an arrow head at each
/// end, which is how a measurement reads as one rather than as a strut.
///
/// Comes back empty when the two ends are too close for the heads to fit, so a
/// drag that has not moved yet draws nothing instead of a blob.
pub fn dimension_line(
    from: Vec3,
    to: Vec3,
    tube_radius: f64,
    head_length: f64,
    head_radius: f64,
) -> Mesh {
    let mut mesh = Mesh::default();
    let along = difference(to, from);
    let span = length(along);
    if span <= head_length * constants::VIEW_EDIT_MEASURE_MIN_LENGTH_HEADS
        || tube_radius <= 0.0
        || head_radius <= 0.0
    {
        return mesh;
    }
    let axis = normalize(along);
    append(
        &mut mesh,
        &cylinder(from, to, tube_radius, constants::VIEW_ARC_TUBE_SEGMENTS),
    );
    // Both heads point outwards, so the line reads as measuring the gap between
    // its ends rather than as pointing from one to the other.
    append(
        &mut mesh,
        &cone(
            sum(from, scale(axis, head_length)),
            scale(axis, -1.0),
            head_length,
            head_radius,
            constants::VIEW_CYLINDER_SEGMENTS,
        ),
    );
    append(
        &mut mesh,
        &cone(
            difference(to, scale(axis, head_length)),
            axis,
            head_length,
            head_radius,
            constants::VIEW_CYLINDER_SEGMENTS,
        ),
    );
    mesh
}

/// Axis aligned cube of edge `edge` centred on `center`.
pub fn marker_cube(center: Vec3, edge: f64) -> Mesh {
    let half = 0.5 * edge;
    box_mesh(
        [center[0] - half, center[1] - half, center[2] - half],
        [center[0] + half, center[1] + half, center[2] + half],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::inner;
    use crate::mesh::validate;

    fn assert_sane(mesh: &Mesh, what: &str) {
        assert!(!mesh.triangles.is_empty(), "{what} produced no triangles");
        for (index, v) in mesh.vertices.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    v[d].is_finite(),
                    "{what} vertex {index} has a non-finite coordinate {v:?}"
                );
            }
        }
        for (index, t) in mesh.triangles.iter().enumerate() {
            let a = mesh.vertices[t[0] as usize];
            let b = mesh.vertices[t[1] as usize];
            let c = mesh.vertices[t[2] as usize];
            let normal = cross(difference(b, a), difference(c, a));
            let area = 0.5 * length(normal);
            assert!(
                area > constants::MIN_TRIANGLE_AREA_MM2,
                "{what} triangle {index} is degenerate (area {area:e})"
            );
        }
    }

    #[test]
    fn a_box_is_a_closed_solid_of_the_right_volume() {
        let mesh = box_mesh([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        assert_sane(&mesh, "box");
        assert_eq!(mesh.vertices.len(), 8);
        assert_eq!(mesh.triangles.len(), 12);
        let stats = validate::validate(&mesh).expect("closed box");
        assert!((stats.volume_mm3 - 24.0).abs() < 1e-9);
    }

    #[test]
    fn a_cylinder_is_closed_and_approaches_its_analytic_volume() {
        let segments = 64;
        let mesh = cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 10.0], 3.0, segments);
        assert_sane(&mesh, "cylinder");
        assert_eq!(mesh.vertices.len(), 2 * segments + 2);
        assert_eq!(mesh.triangles.len(), 4 * segments);
        let stats = validate::validate(&mesh).expect("closed cylinder");
        let expected = std::f64::consts::PI * 9.0 * 10.0;
        assert!(
            (stats.volume_mm3 - expected).abs() / expected < 0.01,
            "cylinder volume {} differs from {expected}",
            stats.volume_mm3
        );
    }

    #[test]
    fn a_cylinder_along_every_axis_stays_closed() {
        for axis in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ] {
            let mesh = cylinder([0.0, 0.0, 0.0], axis, 0.4, 16);
            assert_sane(&mesh, "cylinder");
            validate::validate(&mesh).expect("closed cylinder");
        }
    }

    #[test]
    fn a_cone_is_closed_and_approaches_its_analytic_volume() {
        let segments = 64;
        let mesh = cone([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 6.0, 2.0, segments);
        assert_sane(&mesh, "cone");
        assert_eq!(mesh.vertices.len(), segments + 2);
        assert_eq!(mesh.triangles.len(), 2 * segments);
        let stats = validate::validate(&mesh).expect("closed cone");
        let expected = std::f64::consts::PI * 4.0 * 6.0 / 3.0;
        assert!(
            (stats.volume_mm3 - expected).abs() / expected < 0.01,
            "cone volume {} differs from {expected}",
            stats.volume_mm3
        );
    }

    /// The configuration shape's own mesh: closed at every taper, the
    /// cylinder's mesh when its two radii are equal, and the arrowhead's
    /// topology when the second one is nothing at all.
    #[test]
    fn a_frustum_is_closed_from_its_cylinder_to_its_apex() {
        let segments = 64;
        let (p1, p2) = ([0.0, 0.0, 0.0], [0.0, 0.0, 10.0]);
        let mesh = frustum(p1, p2, 4.0, 2.0, segments);
        assert_sane(&mesh, "frustum");
        assert_eq!(mesh.vertices.len(), 2 * segments + 2);
        assert_eq!(mesh.triangles.len(), 4 * segments);
        let stats = validate::validate(&mesh).expect("closed frustum");
        // pi h (r1^2 + r1 r2 + r2^2) / 3, approached from inside as any
        // inscribed mesh is.
        let expected = std::f64::consts::PI * 10.0 * (16.0 + 8.0 + 4.0) / 3.0;
        assert!(
            (stats.volume_mm3 - expected).abs() / expected < 0.01,
            "frustum volume {} differs from {expected}",
            stats.volume_mm3
        );

        // The apex: one vertex rather than a ring of coincident ones, the cap
        // that would have closed it dropped, and still a closed solid.
        let pointed = frustum(p1, p2, 4.0, 0.0, segments);
        assert_sane(&pointed, "cone");
        assert_eq!(pointed.vertices.len(), segments + 2);
        assert_eq!(pointed.triangles.len(), 2 * segments);
        let stats = validate::validate(&pointed).expect("closed cone");
        let expected = std::f64::consts::PI * 16.0 * 10.0 / 3.0;
        assert!(
            (stats.volume_mm3 - expected).abs() / expected < 0.01,
            "cone volume {} differs from {expected}",
            stats.volume_mm3
        );

        // Equal radii are the cylinder: the same barrel, vertex for vertex, and
        // the same closed solid. Compared as geometry rather than as indices -
        // a frustum writes its two rings one after the other, because an apex
        // is a ring of one point, where the cylinder interleaves them.
        let round = frustum(p1, p2, 3.0, 3.0, segments);
        let plain = cylinder(p1, p2, 3.0, segments);
        assert_eq!(round.vertices.len(), plain.vertices.len());
        assert_eq!(round.triangles.len(), plain.triangles.len());
        for v in &round.vertices {
            assert!(
                plain
                    .vertices
                    .iter()
                    .any(|w| length(difference(*v, *w)) < 1e-12),
                "{v:?} is not a vertex of the cylinder"
            );
        }
        let barrel = validate::validate(&round).expect("closed frustum");
        let reference = validate::validate(&plain).expect("closed cylinder");
        assert!((barrel.volume_mm3 - reference.volume_mm3).abs() < 1e-9);
        assert_eq!(barrel.bounds.min, reference.bounds.min);
        assert_eq!(barrel.bounds.max, reference.bounds.max);

        // Every vertex is on the surface the field describes, whichever way it
        // is pointed - which is what makes the overlay the shape rather than an
        // outline of it - and inside the bounds the editor frames against.
        for (p1, p2, radius1, radius2) in [
            ([1.0, 2.0, 3.0], [7.0, 6.0, 9.0], 3.0, 1.0),
            ([1.0, 2.0, 3.0], [7.0, 6.0, 9.0], 3.0, 0.0),
            ([0.0; 3], [4.0, 0.0, 0.0], 1.0, 2.0),
        ] {
            let shape = Shape::Cone {
                p1,
                p2,
                radius1,
                radius2,
            };
            let mesh = super::shape(&shape);
            assert_sane(&mesh, "cone");
            let stats = validate::validate(&mesh).expect("closed cone");
            let bounds = shape.bounds();
            for v in &mesh.vertices {
                assert!(
                    shape.signed_distance(*v).abs() < 1e-9,
                    "{v:?} is off the surface of {shape:?}"
                );
            }
            for d in 0..3 {
                assert!(stats.bounds.min[d] >= bounds.min[d] - 1e-9);
                assert!(stats.bounds.max[d] <= bounds.max[d] + 1e-9);
            }
        }

        // Nothing to draw is an empty mesh rather than a broken one.
        for (radius1, radius2) in [(0.0, 1.0), (1.0, -1.0)] {
            assert!(
                frustum(p1, p2, radius1, radius2, segments)
                    .triangles
                    .is_empty()
            );
        }
        assert!(frustum(p1, p1, 4.0, 2.0, segments).triangles.is_empty());
        assert!(frustum(p1, p2, 4.0, 2.0, 2).triangles.is_empty());
    }

    /// The prism's mesh: two caps, three quads, and every face outward however
    /// the three points were wound.
    #[test]
    fn a_triangular_prism_is_closed_and_faces_outward_either_way_round() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [0.0, 9.0, 0.0]);
        for (a, b, c) in [(a, b, c), (a, c, b)] {
            let mesh = triangle_prism(a, b, c, 4.0);
            assert_sane(&mesh, "prism");
            assert_eq!(mesh.vertices.len(), 6);
            assert_eq!(mesh.triangles.len(), 8, "two caps and a quad per edge");
            let stats = validate::validate(&mesh).expect("closed prism");
            // Exactly its own volume: the mesh is the solid rather than an
            // approximation of it, so this is an equality and not a bound.
            assert!(
                (stats.volume_mm3 - 0.5 * 12.0 * 9.0 * 4.0).abs() < 1e-9,
                "prism volume {}",
                stats.volume_mm3
            );
            let shape = Shape::Triangle {
                a,
                b,
                c,
                thickness: 4.0,
            };
            for v in &mesh.vertices {
                assert!(shape.signed_distance(*v).abs() < 1e-12, "{v:?}");
            }
            assert_eq!(super::shape(&shape).triangles, mesh.triangles);
        }

        // Turned out of the axes, and still the solid its field describes.
        let shape = Shape::Triangle {
            a: [1.0, 2.0, 3.0],
            b: [7.0, 3.0, 9.0],
            c: [2.0, 9.0, 5.0],
            thickness: 3.0,
        };
        let mesh = super::shape(&shape);
        assert_sane(&mesh, "turned prism");
        let stats = validate::validate(&mesh).expect("closed prism");
        assert_eq!(stats.bounds.min, shape.bounds().min);
        assert_eq!(stats.bounds.max, shape.bounds().max);
        for v in &mesh.vertices {
            assert!(shape.signed_distance(*v).abs() < 1e-9, "{v:?}");
        }

        // A degenerate triangle draws nothing at all.
        assert!(
            triangle_prism(a, b, [24.0, 0.0, 0.0], 4.0)
                .triangles
                .is_empty()
        );
        assert!(triangle_prism(a, b, c, 0.0).triangles.is_empty());
    }

    /// The centroid a load indicator hangs from, for the two new kinds.
    #[test]
    fn a_cone_and_a_prism_are_anchored_where_their_handles_are() {
        let cone = Shape::Cone {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            radius1: 4.0,
            radius2: 1.0,
        };
        // The middle of the axis, which is where the gizmo anchors it - not
        // the centre of mass, which the taper would pull towards the wide end.
        assert_eq!(centroid(&cone), [0.0, 0.0, 5.0]);
        let prism = Shape::Triangle {
            a: [0.0, 0.0, 0.0],
            b: [12.0, 0.0, 0.0],
            c: [0.0, 9.0, 0.0],
            thickness: 4.0,
        };
        assert_eq!(centroid(&prism), [4.0, 3.0, 0.0]);
    }

    #[test]
    fn a_sphere_is_closed_and_approaches_its_analytic_volume() {
        let (segments, rings) = (48, 24);
        let mesh = sphere([1.0, -2.0, 3.0], 5.0, segments, rings);
        assert_sane(&mesh, "sphere");
        assert_eq!(mesh.vertices.len(), (rings - 1) * segments + 2);
        // Two poles contribute one triangle per segment, the inner bands two.
        assert_eq!(mesh.triangles.len(), 2 * segments * (rings - 1));
        let stats = validate::validate(&mesh).expect("closed sphere");
        let expected = 4.0 / 3.0 * std::f64::consts::PI * 125.0;
        assert!(
            (stats.volume_mm3 - expected).abs() / expected < 0.01,
            "sphere volume {} differs from {expected}",
            stats.volume_mm3
        );
    }

    /// The ellipsoid is the sphere's own mesh, stretched and turned: same
    /// topology, same closure, and every vertex on the surface it describes.
    #[test]
    fn an_ellipsoid_is_the_sphere_mesh_stretched_onto_its_own_surface() {
        let (segments, rings) = (48, 24);
        let (center, radii) = ([1.0, -2.0, 3.0], [6.0, 3.0, 2.0]);
        for rotation in [[0.0; 3], [0.0, 0.0, 45.0], [15.0, -35.0, 62.0]] {
            let mesh = ellipsoid(center, radii, rotation, segments, rings);
            assert_sane(&mesh, "ellipsoid");
            assert_eq!(mesh.vertices.len(), (rings - 1) * segments + 2);
            assert_eq!(mesh.triangles.len(), 2 * segments * (rings - 1));
            let stats = validate::validate(&mesh).expect("closed ellipsoid");
            // 4/3 pi abc, approached from inside as any inscribed mesh is.
            let expected = 4.0 / 3.0 * std::f64::consts::PI * radii[0] * radii[1] * radii[2];
            assert!(
                (stats.volume_mm3 - expected).abs() / expected < 0.01,
                "{rotation:?}: volume {} differs from {expected}",
                stats.volume_mm3
            );
            // Every vertex is on the surface the field describes, which is what
            // makes the overlay the shape rather than an outline of it.
            let shape = Shape::Ellipsoid {
                center,
                radii,
                rotation_deg: rotation,
            };
            for v in &mesh.vertices {
                assert!(
                    shape.signed_distance(*v).abs() < 1e-9,
                    "{rotation:?}: {v:?} is off the surface"
                );
            }
            // And it sits inside its own bounds, which is what the editor
            // clamps and frames against.
            let bounds = shape.bounds();
            for d in 0..3 {
                assert!(stats.bounds.min[d] >= bounds.min[d] - 1e-9);
                assert!(stats.bounds.max[d] <= bounds.max[d] + 1e-9);
            }
        }
        // Equal radii are the sphere, to the vertex.
        let round = ellipsoid([0.0; 3], [4.0; 3], [0.0; 3], segments, rings);
        let sphere = sphere([0.0; 3], 4.0, segments, rings);
        assert_eq!(round.triangles, sphere.triangles);
        for (a, b) in round.vertices.iter().zip(sphere.vertices.iter()) {
            assert!(length(difference(*a, *b)) < 1e-12, "{a:?} against {b:?}");
        }
    }

    /// The tube's mesh is one closed surface with every vertex exactly on the
    /// field's own zero level set - straight and bent - which is what makes the
    /// overlay the shape rather than an outline of it.
    #[test]
    fn a_tube_is_a_closed_capsule_whose_vertices_lie_on_its_own_surface() {
        let (segments, arc_segments, cap_rings) = (32, 24, 8);
        for (bend, bands) in [
            (None, 1),
            (Some([10.0, 6.0, 0.0]), arc_segments),
            (Some([10.0, 3.0, 4.0]), arc_segments),
        ] {
            let (p1, p2, radius) = ([0.0, 0.0, 0.0], [20.0, 0.0, 0.0], 2.0);
            let mesh = tube(p1, p2, bend, radius, segments, arc_segments, cap_rings);
            assert_sane(&mesh, "tube");
            // Two poles, and a ring per band boundary and per intermediate cap
            // ring either end.
            assert_eq!(
                mesh.vertices.len(),
                2 + segments * (2 * (cap_rings - 1) + bands + 1)
            );
            let stats = validate::validate(&mesh).expect("closed tube");
            let shape = Shape::Tube {
                p1,
                p2,
                bend,
                radius,
            };
            for v in &mesh.vertices {
                assert!(
                    shape.signed_distance(*v).abs() < 1e-9,
                    "{bend:?}: {v:?} is off the surface by {}",
                    shape.signed_distance(*v)
                );
            }
            // It is inside its own bounds, which is what the editor frames and
            // clamps against, and it holds a sensible volume: a barrel about
            // the centre line plus the ball its two caps make, approached from
            // inside as any inscribed mesh is.
            let bounds = shape.bounds();
            for d in 0..3 {
                assert!(stats.bounds.min[d] >= bounds.min[d] - 1e-9);
                assert!(stats.bounds.max[d] <= bounds.max[d] + 1e-9);
            }
            let curve = match crate::geometry::tube_arc(p1, p2, bend) {
                Some(arc) => arc.radius * arc.span,
                None => length(difference(p2, p1)),
            };
            let expected = std::f64::consts::PI * radius * radius * curve
                + 4.0 / 3.0 * std::f64::consts::PI * radius * radius * radius;
            assert!(
                (stats.volume_mm3 - expected).abs() / expected < 0.05,
                "{bend:?}: volume {} differs from {expected}",
                stats.volume_mm3
            );
        }
        // A straight tube is the capsule its ends describe, and one of no
        // length at all is the ball those ends share rather than nothing.
        let ball = tube(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            None,
            4.0,
            segments,
            1,
            cap_rings,
        );
        let stats = validate::validate(&ball).expect("closed ball");
        let expected = 4.0 / 3.0 * std::f64::consts::PI * 64.0;
        assert!(
            (stats.volume_mm3 - expected).abs() / expected < 0.03,
            "the ball holds {} rather than {expected}",
            stats.volume_mm3
        );
        // Degenerate requests come back empty rather than as bad triangles.
        assert!(tube([0.0; 3], [1.0, 0.0, 0.0], None, 0.0, segments, 1, cap_rings).is_empty());
        assert!(tube([0.0; 3], [1.0, 0.0, 0.0], None, 1.0, 2, 1, cap_rings).is_empty());
        assert!(tube([0.0; 3], [1.0, 0.0, 0.0], None, 1.0, segments, 0, cap_rings).is_empty());
        assert!(tube([0.0; 3], [1.0, 0.0, 0.0], None, 1.0, segments, 1, 0).is_empty());
    }

    /// The load indicator of a bent tube hangs off the middle of its curve,
    /// which is a point on the tube - not the middle of the line between its
    /// ends, which need not be anywhere near it.
    #[test]
    fn a_bent_tube_is_centred_on_its_own_curve() {
        let straight = Shape::Tube {
            p1: [0.0, 0.0, 0.0],
            p2: [20.0, 0.0, 0.0],
            bend: None,
            radius: 2.0,
        };
        assert_eq!(centroid(&straight), [10.0, 0.0, 0.0]);
        let bent = Shape::Tube {
            p1: [0.0, 0.0, 0.0],
            p2: [20.0, 0.0, 0.0],
            bend: Some([10.0, 8.0, 0.0]),
            radius: 2.0,
        };
        let middle = centroid(&bent);
        assert!(bent.contains(middle), "the centroid is not on the tube");
        assert!(
            (middle[1] - 8.0).abs() < 1e-9,
            "the apex of this arc is its bend point: {middle:?}"
        );
        // Most of a circle, where the chord's midpoint is nowhere near it.
        let hook = Shape::Tube {
            p1: [0.0, 0.0, 0.0],
            p2: [2.0, 1.0, 0.0],
            bend: Some([-6.0, 8.0, 0.0]),
            radius: 1.0,
        };
        assert!(hook.contains(centroid(&hook)));
        assert!(!hook.contains([1.0, 0.5, 0.0]), "the chord's own midpoint");
    }

    #[test]
    fn shapes_dispatch_on_their_own_variant() {
        for primitive in [
            Shape::axis_aligned_box([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
            Shape::Box {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 2.0, 3.0],
                rotation_deg: [15.0, -30.0, 45.0],
            },
            Shape::Cylinder {
                p1: [0.0, 0.0, 0.0],
                p2: [4.0, 1.0, 2.0],
                radius: 1.5,
            },
            Shape::Sphere {
                center: [2.0, 2.0, 2.0],
                radius: 2.0,
            },
            Shape::Ellipsoid {
                center: [2.0, 2.0, 2.0],
                radii: [3.0, 2.0, 1.0],
                rotation_deg: [0.0; 3],
            },
            Shape::Ellipsoid {
                center: [2.0, 2.0, 2.0],
                radii: [3.0, 2.0, 1.0],
                rotation_deg: [15.0, -30.0, 45.0],
            },
            Shape::Tube {
                p1: [0.0, 0.0, 0.0],
                p2: [8.0, 2.0, 1.0],
                bend: None,
                radius: 1.0,
            },
            Shape::Tube {
                p1: [0.0, 0.0, 0.0],
                p2: [8.0, 2.0, 1.0],
                bend: Some([4.0, 4.0, 3.0]),
                radius: 1.0,
            },
            Shape::Cone {
                p1: [0.0, 0.0, 0.0],
                p2: [4.0, 1.0, 2.0],
                radius1: 1.5,
                radius2: 0.5,
            },
            Shape::Cone {
                p1: [0.0, 0.0, 0.0],
                p2: [4.0, 1.0, 2.0],
                radius1: 1.5,
                radius2: 0.0,
            },
            Shape::Triangle {
                a: [0.0, 0.0, 0.0],
                b: [4.0, 1.0, 2.0],
                c: [1.0, 4.0, 0.0],
                thickness: 1.0,
            },
        ] {
            let mesh = shape(&primitive);
            assert_sane(&mesh, "shape");
            validate::validate(&mesh).expect("closed shape");
            let c = centroid(&primitive);
            assert!(c.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn an_arrow_ends_at_its_anchor_and_trails_back_along_the_force() {
        let tip = [10.0, 0.0, 0.0];
        let direction = [0.0, 0.0, -1.0];
        let length_mm = 20.0;
        let mesh = arrow(tip, direction, length_mm);
        assert_sane(&mesh, "arrow");
        // Shaft and head are each closed, so the pair validates as a solid.
        let stats = validate::validate(&mesh).expect("closed arrow");
        // The tip sits on the anchor and the body trails away against the
        // force, so a downward load points down at the region it acts on.
        assert!(
            (stats.bounds.min[2] - tip[2]).abs() < 1e-9,
            "the arrow tip is at {}",
            stats.bounds.min[2]
        );
        assert!(
            (stats.bounds.max[2] - (tip[2] + length_mm)).abs() < 1e-9,
            "the arrow tail is at {}",
            stats.bounds.max[2]
        );
        // Nothing sticks out sideways beyond the head radius.
        let radius = length_mm * constants::VIEW_ARROW_HEAD_RADIUS_FRACTION;
        for d in [0, 1] {
            assert!(stats.bounds.max[d] - tip[d] <= radius + 1e-9);
            assert!(tip[d] - stats.bounds.min[d] <= radius + 1e-9);
        }
        // A zero length arrow is skipped rather than built degenerate.
        assert!(arrow(tip, direction, 0.0).is_empty());
    }

    #[test]
    fn a_torque_arc_wraps_its_axis_and_reverses_with_the_sign() {
        let axis_point = [5.0, 5.0, 5.0];
        let axis = [0.0, 0.0, 1.0];
        let radius = 8.0;
        let positive = torque_arc(axis_point, axis, radius, 1.0);
        let negative = torque_arc(axis_point, axis, radius, -1.0);
        assert_sane(&positive, "torque arc");
        assert_sane(&negative, "torque arc");
        validate::validate(&positive).expect("closed torque arc");
        let stats = validate::validate(&positive).expect("stats");
        // The tube stays in the plane of the arc, one tube radius either side.
        let slack = radius * constants::VIEW_TORQUE_HEAD_RADIUS_FRACTION * 2.0;
        assert!((stats.bounds.max[2] - axis_point[2]).abs() < slack);
        assert!((stats.bounds.min[2] - axis_point[2]).abs() < slack);
        // The two senses mirror each other, so they cannot be identical.
        assert_eq!(positive.vertices.len(), negative.vertices.len());
        assert!(
            positive
                .vertices
                .iter()
                .zip(negative.vertices.iter())
                .any(|(a, b)| a != b)
        );
    }

    #[test]
    fn degenerate_requests_produce_empty_meshes_rather_than_bad_triangles() {
        assert!(cylinder([0.0; 3], [0.0; 3], 1.0, 16).is_empty());
        assert!(cylinder([0.0; 3], [1.0, 0.0, 0.0], 0.0, 16).is_empty());
        assert!(cylinder([0.0; 3], [1.0, 0.0, 0.0], 1.0, 2).is_empty());
        assert!(cone([0.0; 3], [0.0, 0.0, 1.0], 0.0, 1.0, 16).is_empty());
        assert!(sphere([0.0; 3], 0.0, 16, 8).is_empty());
        assert!(ellipsoid([0.0; 3], [1.0, 0.0, 1.0], [0.0; 3], 16, 8).is_empty());
        assert!(ellipsoid([0.0; 3], [1.0, -1.0, 1.0], [0.0; 3], 16, 8).is_empty());
        assert!(ellipsoid([0.0; 3], [1.0; 3], [0.0; 3], 2, 8).is_empty());
        assert!(arrow([0.0; 3], [0.0; 3], 5.0).is_empty());
        assert!(torque_arc([0.0; 3], [0.0, 0.0, 1.0], 5.0, 0.0).is_empty());
    }

    #[test]
    fn a_marker_cube_is_centred_on_its_point() {
        let mesh = marker_cube([2.0, 3.0, 4.0], 1.0);
        assert_sane(&mesh, "marker");
        let stats = validate::validate(&mesh).expect("closed cube");
        assert!((stats.volume_mm3 - 1.0).abs() < 1e-9);
        for d in 0..3 {
            let mid = 0.5 * (stats.bounds.min[d] + stats.bounds.max[d]);
            assert!((mid - [2.0, 3.0, 4.0][d]).abs() < 1e-9);
        }
    }

    #[test]
    fn appending_keeps_both_meshes_intact() {
        let mut target = box_mesh([0.0; 3], [1.0, 1.0, 1.0]);
        let source = box_mesh([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);
        append(&mut target, &source);
        assert_eq!(target.vertices.len(), 16);
        assert_eq!(target.triangles.len(), 24);
        let stats = validate::validate(&target).expect("two closed cubes");
        assert!((stats.volume_mm3 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn the_basis_is_orthonormal_and_right_handed() {
        for axis in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-3.0, 0.2, 7.0],
        ] {
            let (u, v) = basis(axis);
            let unit = normalize(axis);
            assert!((length(u) - 1.0).abs() < 1e-12);
            assert!((length(v) - 1.0).abs() < 1e-12);
            assert!(inner(u, v).abs() < 1e-12);
            assert!(inner(u, unit).abs() < 1e-12);
            let handed = cross(u, v);
            for d in 0..3 {
                assert!((handed[d] - unit[d]).abs() < 1e-12);
            }
        }
    }
}
