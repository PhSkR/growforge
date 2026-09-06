//! Signed distance primitives and the ordered CSG evaluation used to define a
//! design domain.
//!
//! All distances are millimetres. The sign convention is the usual one:
//! negative inside, zero on the surface, positive outside.

use std::f64::consts::{PI, TAU};

use crate::constants;

/// A point or vector in millimetre world coordinates.
pub type Vec3 = [f64; 3];

/// Axis aligned bounding box in millimetres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Lower corner.
    pub min: Vec3,
    /// Upper corner.
    pub max: Vec3,
}

impl Aabb {
    /// Box that contains nothing; `union` with it is the identity.
    pub fn empty() -> Self {
        Aabb {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        }
    }

    /// True when the box has no interior on some axis.
    pub fn is_empty(&self) -> bool {
        (0..3).any(|d| self.max[d] < self.min[d])
    }

    /// Smallest box containing both inputs.
    pub fn union(&self, other: &Aabb) -> Aabb {
        let mut out = Aabb::empty();
        for d in 0..3 {
            out.min[d] = self.min[d].min(other.min[d]);
            out.max[d] = self.max[d].max(other.max[d]);
        }
        out
    }

    /// Edge lengths along x, y, z.
    pub fn extent(&self) -> Vec3 {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Enclosed volume in cubic millimetres.
    pub fn volume(&self) -> f64 {
        let e = self.extent();
        e[0] * e[1] * e[2]
    }

    /// Grow the box by `margin` on every side.
    pub fn expanded(&self, margin: f64) -> Aabb {
        Aabb {
            min: [
                self.min[0] - margin,
                self.min[1] - margin,
                self.min[2] - margin,
            ],
            max: [
                self.max[0] + margin,
                self.max[1] + margin,
                self.max[2] + margin,
            ],
        }
    }
}

/// Boolean operation applied by a domain entry, in list order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsgOp {
    /// Union the shape into the domain.
    Add,
    /// Cut the shape out of the domain.
    Subtract,
}

/// A 3x3 matrix, row major: `m[r][c]`.
pub type Mat3 = [[f64; 3]; 3];

/// Rotation matrix of a triple of degrees about the **fixed world axes**, in
/// x, y, z order (extrinsic XYZ): `R = Rz(rz) * Ry(ry) * Rx(rx)`.
///
/// Extrinsic means every angle is measured about a world axis, not about an
/// axis the previous rotation already moved; the composition order above is
/// what makes that true. Positive angles follow the right hand rule.
pub fn rotation_matrix(deg: Vec3) -> Mat3 {
    let (sx, cx) = deg[0].to_radians().sin_cos();
    let (sy, cy) = deg[1].to_radians().sin_cos();
    let (sz, cz) = deg[2].to_radians().sin_cos();
    [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
}

/// True when a rotation triple leaves everything where it is, which is what
/// the unrotated fast paths test for.
///
/// Compared against zero rather than against a tolerance: the point is that a
/// shape carrying no rotation takes exactly the arithmetic it took before
/// rotation existed, so that every recorded result of an unrotated model is
/// reproduced bit for bit.
pub fn is_unrotated(deg: Vec3) -> bool {
    deg[0] == 0.0 && deg[1] == 0.0 && deg[2] == 0.0
}

/// `m * v`.
pub fn rotate(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// `m^T * v`, which for a rotation matrix is the inverse rotation.
pub fn rotate_inverse(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// The axis that one component of an extrinsic XYZ triple turns about, given
/// the rotation the shape already carries.
///
/// The triple composes as `Rz * Ry * Rx`, so adding `d` degrees to one of the
/// three is a rotation of `d` degrees about a definite axis - and only the z
/// component turns about a world axis. The others turn about the axis the
/// components outside them have already moved:
///
/// | component | axis                    |
/// | --------- | ----------------------- |
/// | 0 (x)     | `Rz * Ry * x`           |
/// | 1 (y)     | `Rz * y`                |
/// | 2 (z)     | world z                 |
///
/// This is what lets a rotation gizmo drive the very number the file holds:
/// turning about the axis this returns *is* incrementing that component.
pub fn euler_axis(rotation_deg: Vec3, component: usize) -> Vec3 {
    match component {
        0 => rotate(
            &rotation_matrix([0.0, rotation_deg[1], rotation_deg[2]]),
            [1.0, 0.0, 0.0],
        ),
        1 => rotate(
            &rotation_matrix([0.0, 0.0, rotation_deg[2]]),
            [0.0, 1.0, 0.0],
        ),
        _ => [0.0, 0.0, 1.0],
    }
}

/// An angle folded into `[-180, 180]`, for display. The stored value is never
/// normalized: what the user typed is what the file holds.
pub fn normalize_degrees(deg: f64) -> f64 {
    if !deg.is_finite() {
        return deg;
    }
    let wrapped = (deg + 180.0).rem_euclid(360.0) - 180.0;
    // `rem_euclid` maps exactly 180 to -180; the positive half turn reads
    // better and is what a rotation of 180 degrees was typed as.
    if wrapped == -180.0 { 180.0 } else { wrapped }
}

/// A signed distance primitive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// Rectangular box, optionally rotated about its own centre.
    Box {
        /// Lower corner of the unrotated box in mm.
        min: Vec3,
        /// Upper corner of the unrotated box in mm.
        max: Vec3,
        /// Rotation about the box centre in degrees, extrinsic XYZ.
        /// `[0, 0, 0]` is an axis aligned box.
        rotation_deg: Vec3,
    },
    /// Cylinder capped by the planes through `p1` and `p2`.
    Cylinder {
        /// Centre of the first cap.
        p1: Vec3,
        /// Centre of the second cap.
        p2: Vec3,
        /// Cylinder radius in mm.
        radius: f64,
    },
    /// Sphere.
    Sphere {
        /// Centre in mm.
        center: Vec3,
        /// Radius in mm.
        radius: f64,
    },
    /// Ellipsoid with a semi-axis per direction, optionally rotated about its
    /// own centre.
    ///
    /// Its field is the scaled-space one: the sample is centred, inverse
    /// rotated, divided component-wise by the radii, and
    /// `(|q / r| - 1) * min(r)` is returned. Two things about that are worth
    /// knowing, and everything downstream rests on the first:
    ///
    /// * **The zero level set is exact.** `|q / r| = 1` *is* the ellipsoid, so
    ///   the sign is the exact answer to "is this point inside", at every point
    ///   in space. That is the only thing growforge asks a shape: cell
    ///   classification samples cell centres ([`crate::grid::Grid::classify`]),
    ///   support and load region selection samples nodes
    ///   ([`crate::problem::Problem::build`]), [`Shape::contains`], [`Csg`] and
    ///   [`ShapeUnion`] threshold the sign, and the editor's hover, pick and
    ///   containment read the bounds. None of them reads a magnitude.
    /// * **The magnitude away from the surface is a bound, not the distance.**
    ///   It is the exact distance in the scaled space carried back with the
    ///   smallest radius, which under-states the true distance and never
    ///   over-states it, and it is 1-Lipschitz - so a sphere trace of it
    ///   converges rather than stepping through the surface. It is exact at the
    ///   centre, where it is `-min(r)`, and it is the sphere's own exact field
    ///   when the three radii are equal. (The true distance to an ellipsoid has
    ///   no closed form; it is the root of a sextic.)
    Ellipsoid {
        /// Centre in mm, which is the point it rotates about.
        center: Vec3,
        /// Semi-axes in mm along the ellipsoid's own x, y and z, before any
        /// rotation. Positive by construction: [`crate::config::ShapeSpec`]
        /// rejects anything else.
        radii: Vec3,
        /// Rotation about the centre in degrees, extrinsic XYZ - the same
        /// semantics, and the same helpers, as [`Shape::Box`]. `[0, 0, 0]` is
        /// an axis aligned ellipsoid.
        rotation_deg: Vec3,
    },
    /// Everything within `radius` of a curve: a **capsule** between `p1` and
    /// `p2`, or, when it carries a `bend` point, the circular **arc** through
    /// the three.
    ///
    /// It is the round-ended sibling of [`Shape::Cylinder`], not a variant of
    /// it: a cylinder is capped by the flat planes through its two points,
    /// while a tube's ends are hemispheres centred on them. That is what lets
    /// two tubes meeting at a point join without a seam, and it is why a bent
    /// one can be a single primitive at all.
    ///
    /// **Its field is the exact signed distance, everywhere.** The tube is the
    /// set of points within `radius` of its centre line, so its signed distance
    /// is the distance to that curve minus the radius - and the distance to a
    /// segment and the distance to a circular arc both have closed forms. That
    /// is a stronger guarantee than the ellipsoid's, whose magnitude away from
    /// the surface is only a bound (see [`Shape::Ellipsoid`]): a sphere trace of
    /// a tube converges in a handful of steps because every step is the real
    /// distance, which is what the editor's ray picking relies on - a bent tube
    /// is not convex and has no closed form hit.
    Tube {
        /// Centre of the first end cap.
        p1: Vec3,
        /// Centre of the second end cap.
        p2: Vec3,
        /// A point the centre line is bent through. The curve is then the arc
        /// of the circle through `p1`, `bend` and `p2` that passes through
        /// `bend`. Absent - or within [`constants::TUBE_COLLINEAR_EPS_MM`] of
        /// the line through the two ends, which is the same tube to well under
        /// a voxel - is the straight capsule; see [`tube_arc`].
        bend: Option<Vec3>,
        /// Radius in mm.
        radius: f64,
    },
    /// Capped cone - a **frustum** - between the flat discs through `p1` and
    /// `p2`, carrying a radius at each.
    ///
    /// The straight-walled sibling of [`Shape::Cylinder`], which is exactly the
    /// case where its two radii are equal, and capped the same way: by the
    /// planes through its two points rather than by anything rounded.
    /// `radius2` may be **zero**, which is the true cone - the second disc
    /// shrinks to an apex - while `radius1` is the one that has to be positive,
    /// so that the wide end is always a real disc.
    ///
    /// **Its field is the exact signed distance, everywhere.** A frustum is a
    /// surface of revolution, so the nearest surface point to a sample lies in
    /// that sample's own meridian half plane - every other meridian is further
    /// by the arc between them - and the whole of the geometry is therefore two
    /// dimensional: the distance from `(radial offset, height along the axis)`
    /// to the trapezoid the profile is, which is the nearest of three segments,
    /// the two caps and the slanted wall. That is a stronger guarantee than the
    /// ellipsoid's, whose magnitude away from the surface is only a bound (see
    /// [`Shape::Ellipsoid`]), and a frustum is **convex**, so the editor picks
    /// one in closed form as it does a cylinder.
    Cone {
        /// Centre of the first cap, where the shape is `radius1` wide.
        p1: Vec3,
        /// Centre of the second cap, where it is `radius2` wide.
        p2: Vec3,
        /// Radius at `p1` in mm. Positive by construction:
        /// [`crate::config::ShapeSpec`] rejects anything else.
        radius1: f64,
        /// Radius at `p2` in mm. Zero is legal and is the apex of a true cone;
        /// negative is not.
        radius2: f64,
    },
    /// Triangular prism: the triangle `a`, `b`, `c` extruded by `thickness`
    /// **symmetrically about its own plane**, half of it either side.
    ///
    /// The one primitive whose orientation is given entirely by the points it
    /// is written in - three of them fix a plane and a shape in it, which is
    /// complete pose control - so it carries no rotation, as a sphere carries
    /// none for the opposite reason.
    ///
    /// **Its field is the exact signed distance, everywhere.** The prism is the
    /// product of the triangle and the slab, so a sample's distance to it
    /// splits the same way a box's does: the exact distance within the plane -
    /// the nearest of the three edges, each clamped to its own ends, which is
    /// what puts a sample in an edge region or a vertex region without a case
    /// analysis of its own - combined with the distance out of the slab in the
    /// squared-space form the [`Shape::Cylinder`] arm combines its wall and its
    /// caps in. Convex, so it too is picked in closed form.
    Triangle {
        /// First vertex in mm.
        a: Vec3,
        /// Second vertex in mm.
        b: Vec3,
        /// Third vertex in mm.
        c: Vec3,
        /// Depth of the extrusion in mm, half of it either side of the plane
        /// through the three vertices. Positive by construction.
        thickness: f64,
    },
}

/// The circular arc a bent [`Shape::Tube`] follows, once the circumcircle
/// through its three points has been solved.
///
/// The arc is parameterized by an angle from `start` towards `turn`, so angle
/// zero is `p1`, angle [`Arc::span`] is `p2`, and the bend point is somewhere
/// strictly between the two - which is the orientation [`tube_arc`] picks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc {
    /// Centre of the circle.
    pub center: Vec3,
    /// Its radius in mm.
    pub radius: f64,
    /// Unit vector from the centre to the first end: the arc's zero angle.
    pub start: Vec3,
    /// Unit vector at a right angle to it in the circle's plane, in the
    /// direction the arc runs.
    pub turn: Vec3,
    /// Unit normal of the circle's plane, `start x turn`.
    pub normal: Vec3,
    /// Angle of the second end, in radians. Strictly between 0 and a whole
    /// turn.
    pub span: f64,
}

impl Arc {
    /// The point at `angle` radians round the circle.
    pub fn point(&self, angle: f64) -> Vec3 {
        let (sin, cos) = angle.sin_cos();
        sum(
            self.center,
            sum(
                scale(self.start, self.radius * cos),
                scale(self.turn, self.radius * sin),
            ),
        )
    }

    /// The first end of the arc, which is the tube's `p1`.
    pub fn first(&self) -> Vec3 {
        self.point(0.0)
    }

    /// The second end, which is the tube's `p2`.
    pub fn last(&self) -> Vec3 {
        self.point(self.span)
    }

    /// The largest tube radius this arc can carry before the tube around it
    /// overlaps itself, in millimetres: the arc's **reach**.
    ///
    /// A tube is its centre line grown by a radius, and that construction stays
    /// invertible - every surface point is one curve point carried out along one
    /// normal, so the nearest surface point to anything is reached by walking a
    /// radius out from the nearest curve point - only while the radius stays
    /// under this. Two separate things bound it, and the reach is the smaller.
    ///
    /// * **The bend.** Walking a radius inward from a curve point passes the
    ///   circle's own centre once the radius reaches [`Arc::radius`], and past
    ///   the centre the walk comes back *down* towards the far side of the arc.
    ///   This is the bound a full circle has, and it is the only one a **minor**
    ///   arc has.
    /// * **The gap.** An arc that is not closed has two ends, and a tube's ends
    ///   are hemispheres centred on them. Once those two balls overlap, a point
    ///   beyond one end projects to a surface point that is inside the other -
    ///   and the balls overlap when the chord between the ends,
    ///   `2 R sin(span / 2)`, is under two tube radii.
    ///
    /// The gap bound applies **only to a major arc**, `span >= PI`, and that is
    /// the whole subtlety of it: a short arc also has its two ends close
    /// together, but the arc itself is what lies between them, not empty space.
    /// Formally, the ends bound the reach only when the segment joining them
    /// lies in the normal cone at each - and with `t` the tangent pointing from
    /// an end into the arc, that segment's inner product with `t` is
    /// `R sin(span)` at *both* ends, which is non-positive exactly when
    /// `span >= PI`. Below a half turn the ends face each other *through* the
    /// arc, the chord is a shortcut rather than a bottleneck, and a fat stubby
    /// arc projects exactly however near its ends are. The two bounds agree at
    /// `span = PI`, where the chord is the diameter, so this is a continuous
    /// tightening of the first rather than a second rule bolted beside it.
    ///
    /// See [`tube_self_intersects`], which is what asks this question of a tube.
    pub fn reach(&self) -> f64 {
        if self.span < PI {
            return self.radius;
        }
        // Half the chord across the gap, which past a half turn is never more
        // than the radius and falls to nothing as the arc closes on itself.
        self.radius * (0.5 * self.span).sin()
    }

    /// Exact distance from `q` to the arc.
    ///
    /// Split into the circle's own plane and the offset out of it. The nearest
    /// point of the *whole* circle is the one at `q`'s own angle, so when that
    /// angle lies within the arc it is the nearest point of the arc too, at
    /// `sqrt((|in-plane offset| - R)^2 + out-of-plane^2)`. When it does not,
    /// the distance to a circle point falls with the cosine of the angle
    /// between them, so the nearest point of the arc is whichever end is
    /// nearer.
    ///
    /// A `q` on the axis through the centre has no angle at all; every point of
    /// the circle is then equidistant, which is what the formula returns for
    /// the zero angle it falls back on.
    pub fn distance(&self, q: Vec3) -> f64 {
        let offset = sub(q, self.center);
        let x = dot(offset, self.start);
        let y = dot(offset, self.turn);
        let height = dot(offset, self.normal);
        let angle = y.atan2(x).rem_euclid(TAU);
        if angle <= self.span {
            let radial = (x * x + y * y).sqrt() - self.radius;
            (radial * radial + height * height).sqrt()
        } else {
            norm(sub(q, self.first())).min(norm(sub(q, self.last())))
        }
    }

    /// The point of the arc nearest to `q`, which is where [`Arc::distance`]
    /// measures to.
    ///
    /// The same case split, carried through to the point rather than to the
    /// length: the point at `q`'s own angle when the arc reaches that angle, and
    /// the nearer end when it does not. A `q` on the axis through the centre has
    /// no angle of its own and every point of the circle is equidistant; the
    /// zero angle `atan2` falls back on is one of them, so the answer is still a
    /// nearest point and still the one the distance was measured to.
    pub fn closest_point(&self, q: Vec3) -> Vec3 {
        let offset = sub(q, self.center);
        let x = dot(offset, self.start);
        let y = dot(offset, self.turn);
        let angle = y.atan2(x).rem_euclid(TAU);
        if angle <= self.span {
            self.point(angle)
        } else {
            let (first, last) = (self.first(), self.last());
            if norm(sub(q, first)) <= norm(sub(q, last)) {
                first
            } else {
                last
            }
        }
    }
}

/// The arc a tube's three points describe, or `None` when it is straight.
///
/// Straight is a tube with no bend point, and a tube whose bend point lies
/// within [`constants::TUBE_COLLINEAR_EPS_MM`] of the infinite line through its
/// two ends - which includes a bend that has drifted onto that line past an
/// end, where there is no circle through the three at all. Such a tube takes
/// the segment's own arithmetic, bit for bit the arithmetic a tube carrying no
/// bend takes.
///
/// The circumcentre is the standard closed form: with `u = bend - p1` and
/// `v = p2 - p1`,
///
/// ```text
/// c = p1 + ((|u|^2 v - |v|^2 u) x (u x v)) / (2 |u x v|^2)
/// ```
///
/// and `|u x v|` is the chord length times the bend point's distance from the
/// chord, which is what the collinearity test above is written in.
pub fn tube_arc(p1: Vec3, p2: Vec3, bend: Option<Vec3>) -> Option<Arc> {
    let bend = bend?;
    let u = sub(bend, p1);
    let v = sub(p2, p1);
    let chord = norm(v);
    let plane = cross(u, v);
    let twice_area = norm(plane);
    if chord <= 0.0 || twice_area <= constants::TUBE_COLLINEAR_EPS_MM * chord {
        return None;
    }
    let offset = scale(
        cross(sub(scale(v, dot(u, u)), scale(u, dot(v, v))), plane),
        1.0 / (2.0 * twice_area * twice_area),
    );
    let center = sum(p1, offset);
    let radius = norm(offset);
    // `offset` runs from the first end to the centre, so the arc's own zero
    // angle - the unit vector from the centre out to that end - is its reverse.
    let start = scale(offset, -1.0 / radius);
    let mut normal = scale(plane, 1.0 / twice_area);
    let mut turn = cross(normal, start);
    // Of the two arcs between the ends, the tube is the one through the bend
    // point. Turning the plane over reverses the angles, so it is the choice
    // between the two that puts the bend inside the span rather than outside.
    let angle_of = |p: Vec3, turn: Vec3| -> f64 {
        let offset = sub(p, center);
        dot(offset, turn).atan2(dot(offset, start)).rem_euclid(TAU)
    };
    let mut span = angle_of(p2, turn);
    if angle_of(bend, turn) > span {
        normal = scale(normal, -1.0);
        turn = scale(turn, -1.0);
        span = TAU - span;
    }
    Some(Arc {
        center,
        radius,
        start,
        turn,
        normal,
        span,
    })
}

/// The point of the segment between `a` and `b` nearest to `q`.
fn segment_closest_point(q: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let along = sub(b, a);
    let offset = sub(q, a);
    let squared = dot(along, along);
    let t = if squared > 0.0 {
        (dot(offset, along) / squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    sum(a, scale(along, t))
}

/// Exact distance from `q` to the segment between `a` and `b`.
fn segment_distance(q: Vec3, a: Vec3, b: Vec3) -> f64 {
    norm(sub(q, segment_closest_point(q, a, b)))
}

/// Exact distance from `q` to a tube's centre line: the arc through its three
/// points, or the segment between its two ends when it has no usable bend.
pub fn tube_distance(q: Vec3, p1: Vec3, p2: Vec3, bend: Option<Vec3>) -> f64 {
    match tube_arc(p1, p2, bend) {
        Some(arc) => arc.distance(q),
        None => segment_distance(q, p1, p2),
    }
}

/// The point of a tube's centre line nearest to `q`, which is where
/// [`tube_distance`] measures to.
pub fn tube_closest_point(q: Vec3, p1: Vec3, p2: Vec3, bend: Option<Vec3>) -> Vec3 {
    match tube_arc(p1, p2, bend) {
        Some(arc) => arc.closest_point(q),
        None => segment_closest_point(q, p1, p2),
    }
}

/// True when a tube overlaps itself: its radius is at or past the reach of the
/// arc its centre line follows ([`Arc::reach`]).
///
/// There are two ways for that to happen and the reach is the smaller of them -
/// the tube **folds** through the centre of its own bend, or its two ends
/// **close** on each other across the gap the arc leaves open. The second is not
/// a corollary of the first: a shallow arc of large radius, bent almost the
/// whole way round, self-intersects at a radius a small fraction of the arc's.
///
/// Such a tube is still a perfectly good solid. It is the set of points within
/// `radius` of its centre line whatever that radius is, it voxelizes like any
/// other shape, and its signed distance ([`tube_distance`] minus the radius)
/// stays **exact**, because that set is the definition rather than an
/// approximation of it.
///
/// What it stops being is *offsettable*: the surface is no longer the centre
/// line carried out by a radius, so walking a radius out from the nearest
/// centre-line point lands on a point that is still **inside** the solid.
/// [`Shape::nearest_surface_point`] refuses such a tube for that reason rather
/// than answering wrongly, and [`crate::config::Config::static_warnings`] says
/// so where it has a consequence.
///
/// A straight tube can never self-intersect, and neither can one whose bend is
/// within [`constants::TUBE_COLLINEAR_EPS_MM`] of the line through its ends:
/// both have no arc at all, and [`tube_arc`] is the one place that is decided.
pub fn tube_self_intersects(p1: Vec3, p2: Vec3, bend: Option<Vec3>, radius: f64) -> bool {
    tube_arc(p1, p2, bend).is_some_and(|arc| radius >= arc.reach())
}

/// A sample's place in a cone's own **meridian half plane**: the frame every
/// piece of [`Shape::Cone`]'s geometry is written in.
///
/// `None` when the cone has no axis to measure along, which is a shape
/// [`crate::config::ShapeSpec::to_shape`] refuses.
struct ConeFrame {
    /// Unit vector from `p1` towards `p2`.
    unit: Vec3,
    /// Distance between the two cap centres, in mm.
    len: f64,
    /// The part of `p - p1` across the axis.
    radial: Vec3,
    /// Its length, which is the half plane's first coordinate.
    rho: f64,
    /// Height of the sample along the axis from `p1`, the second coordinate.
    along: f64,
}

/// Take `p` into a cone's meridian half plane.
fn cone_frame(p: Vec3, p1: Vec3, p2: Vec3) -> Option<ConeFrame> {
    let axis = sub(p2, p1);
    let len = norm(axis);
    if len <= 0.0 {
        return None;
    }
    let unit = scale(axis, 1.0 / len);
    let offset = sub(p, p1);
    let along = dot(offset, unit);
    let radial = sub(offset, scale(unit, along));
    Some(ConeFrame {
        unit,
        len,
        rho: norm(radial),
        radial,
        along,
    })
}

/// The boundary of a capped cone in that half plane: the cap at `p1`, the
/// slanted wall, and the cap at `p2`, each as a pair of `[radial, along]`
/// points.
///
/// A `radius2` of zero degenerates the second cap to the apex the wall already
/// ends on, which costs one repeated point rather than a case of its own - the
/// distance to a segment of no length is the distance to that point.
fn cone_profile(radius1: f64, radius2: f64, len: f64) -> [([f64; 2], [f64; 2]); 3] {
    let base = [radius1, 0.0];
    let top = [radius2, len];
    [([0.0, 0.0], base), (base, top), ([0.0, len], top)]
}

/// The point of the 2D segment between `a` and `b` nearest to `q`; the half
/// plane's version of [`segment_closest_point`].
fn segment_closest_point_2d(q: [f64; 2], a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let along = [b[0] - a[0], b[1] - a[1]];
    let offset = [q[0] - a[0], q[1] - a[1]];
    let squared = along[0] * along[0] + along[1] * along[1];
    let t = if squared > 0.0 {
        ((offset[0] * along[0] + offset[1] * along[1]) / squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    [a[0] + along[0] * t, a[1] + along[1] * t]
}

/// Distance between two points of the half plane.
fn distance_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    let (x, y) = (a[0] - b[0], a[1] - b[1]);
    (x * x + y * y).sqrt()
}

/// The point of a cone's profile nearest `q`, and how far it is.
fn cone_profile_closest_point(
    q: [f64; 2],
    radius1: f64,
    radius2: f64,
    len: f64,
) -> ([f64; 2], f64) {
    let mut best = q;
    let mut distance = f64::INFINITY;
    for (from, to) in cone_profile(radius1, radius2, len) {
        let point = segment_closest_point_2d(q, from, to);
        let candidate = distance_2d(q, point);
        if candidate < distance {
            distance = candidate;
            best = point;
        }
    }
    (best, distance)
}

/// True when a point of the half plane is inside the cone: within the two caps,
/// and no further out than the radius interpolated linearly between them.
fn cone_contains_profile(q: [f64; 2], radius1: f64, radius2: f64, len: f64) -> bool {
    q[1] >= 0.0 && q[1] <= len && q[0] <= radius1 + (radius2 - radius1) * q[1] / len
}

/// Unit normal of the plane through three points, or `None` when they are
/// collinear and there is no such plane.
pub fn triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Option<Vec3> {
    let normal = cross(sub(b, a), sub(c, a));
    let len = norm(normal);
    (len > 0.0).then(|| scale(normal, 1.0 / len))
}

/// Area of the triangle through three points, in square millimetres.
pub fn triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    0.5 * norm(cross(sub(b, a), sub(c, a)))
}

/// The shortest of a triangle's three altitudes, in millimetres: twice its area
/// over its longest side.
///
/// The measurement that says how **flat** a triangle is rather than how small
/// it is - it is zero exactly when the three points are collinear, whatever
/// their spread - which is what makes it the one
/// [`crate::config::ShapeSpec::to_shape`] rejects a triangle by.
pub fn triangle_shortest_altitude(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let longest = norm(sub(b, a)).max(norm(sub(c, b))).max(norm(sub(a, c)));
    if longest <= 0.0 {
        return 0.0;
    }
    2.0 * triangle_area(a, b, c) / longest
}

/// Radius of the circle inscribed in a triangle: `2 * Area / perimeter`.
///
/// The radius of the largest ball that fits inside it, which is the vocabulary
/// the minimum feature size is written in, and half of what a triangular prism
/// answers [`Shape::min_extent`] with.
pub fn triangle_inradius(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let perimeter = norm(sub(b, a)) + norm(sub(c, b)) + norm(sub(a, c));
    if perimeter <= 0.0 {
        return 0.0;
    }
    2.0 * triangle_area(a, b, c) / perimeter
}

/// The point of a triangle's **boundary** - the closed polyline through its
/// three edges - nearest to `q`.
///
/// The edge and vertex regions of the usual decomposition are exactly what the
/// clamp inside [`segment_closest_point`] applies to each edge's own parameter,
/// so the nearest of the three answers is the nearest point of the boundary
/// with no case analysis of its own.
fn triangle_edge_closest_point(q: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let mut best = segment_closest_point(q, a, b);
    let mut distance = norm(sub(q, best));
    for (from, to) in [(b, c), (c, a)] {
        let point = segment_closest_point(q, from, to);
        let candidate = norm(sub(q, point));
        if candidate < distance {
            distance = candidate;
            best = point;
        }
    }
    best
}

/// True when a point **on a triangle's own plane** lies inside it, edges
/// included.
///
/// One sign test per edge: the three edge cross products of a triangle all
/// point the way its normal does, so a point on the inner side of all three is
/// inside it.
fn triangle_contains_coplanar(q: Vec3, a: Vec3, b: Vec3, c: Vec3, normal: Vec3) -> bool {
    [(a, b), (b, c), (c, a)]
        .into_iter()
        .all(|(from, to)| dot(cross(sub(to, from), sub(q, from)), normal) >= 0.0)
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// Cross product of two vectors.
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Length of a vector.
pub fn length(a: Vec3) -> f64 {
    norm(a)
}

/// Vector difference `a - b`.
pub fn difference(a: Vec3, b: Vec3) -> Vec3 {
    sub(a, b)
}

/// Vector sum `a + b`.
pub fn sum(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scale a vector by a scalar.
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Inner product of two vectors.
pub fn inner(a: Vec3, b: Vec3) -> f64 {
    dot(a, b)
}

impl Shape {
    /// An axis aligned box, which is a box carrying no rotation.
    pub fn axis_aligned_box(min: Vec3, max: Vec3) -> Shape {
        Shape::Box {
            min,
            max,
            rotation_deg: [0.0; 3],
        }
    }

    /// Centre of a box's `min`/`max` pair, which is the point it rotates about.
    pub fn box_center(min: Vec3, max: Vec3) -> Vec3 {
        [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ]
    }

    /// The eight corners of a box, rotated about its centre.
    pub fn box_corners(min: Vec3, max: Vec3, rotation_deg: Vec3) -> [Vec3; 8] {
        let center = Shape::box_center(min, max);
        let matrix = rotation_matrix(rotation_deg);
        let mut out = [[0.0; 3]; 8];
        for (k, corner) in out.iter_mut().enumerate() {
            let local = [
                if k & 1 == 0 { min[0] } else { max[0] } - center[0],
                if k & 2 == 0 { min[1] } else { max[1] } - center[1],
                if k & 4 == 0 { min[2] } else { max[2] } - center[2],
            ];
            *corner = sum(center, rotate(&matrix, local));
        }
        out
    }

    /// Local coordinates of `p` about a shape centred on `center` and turned by
    /// `rotation_deg`: the offset from the centre, inverse-rotated.
    ///
    /// An unrotated shape takes `p - centre` itself, which is the very
    /// arithmetic it took before rotation existed.
    fn to_local(p: Vec3, center: Vec3, rotation_deg: Vec3) -> Vec3 {
        let offset = sub(p, center);
        if is_unrotated(rotation_deg) {
            offset
        } else {
            rotate_inverse(&rotation_matrix(rotation_deg), offset)
        }
    }

    /// The inverse of [`Shape::to_local`]: a local offset carried back into
    /// world coordinates.
    fn from_local(local: Vec3, center: Vec3, rotation_deg: Vec3) -> Vec3 {
        if is_unrotated(rotation_deg) {
            sum(center, local)
        } else {
            sum(center, rotate(&rotation_matrix(rotation_deg), local))
        }
    }

    /// Signed distance from `p` to the shape's surface, negative inside.
    ///
    /// Exact for the box, the capped cylinder, the sphere, the tube, the cone
    /// and the triangular prism. For the ellipsoid the zero level set is exact
    /// and the magnitude away from it is a bound; see [`Shape::Ellipsoid`],
    /// which lists what reads it.
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        match *self {
            Shape::Box {
                min,
                max,
                rotation_deg,
            } => {
                let center = Shape::box_center(min, max);
                let half = [
                    0.5 * (max[0] - min[0]),
                    0.5 * (max[1] - min[1]),
                    0.5 * (max[2] - min[2]),
                ];
                // A rotation is rigid, so the exact distance to the rotated box
                // is the distance from the inverse-rotated sample point to the
                // axis aligned one.
                let local = Shape::to_local(p, center, rotation_deg);
                let q = [
                    local[0].abs() - half[0],
                    local[1].abs() - half[1],
                    local[2].abs() - half[2],
                ];
                let outside = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
                let inside = q[0].max(q[1]).max(q[2]).min(0.0);
                norm(outside) + inside
            }
            Shape::Cylinder { p1, p2, radius } => {
                // Exact signed distance to a capped cylinder (the standard
                // squared-space formulation, which stays well conditioned for
                // very short or very long axes).
                let ba = sub(p2, p1);
                let pa = sub(p, p1);
                let baba = dot(ba, ba);
                let paba = dot(pa, ba);
                let radial = [
                    pa[0] * baba - ba[0] * paba,
                    pa[1] * baba - ba[1] * paba,
                    pa[2] * baba - ba[2] * paba,
                ];
                let x = norm(radial) - radius * baba;
                let y = (paba - baba * 0.5).abs() - baba * 0.5;
                let x2 = x * x;
                let y2 = y * y * baba;
                let d = if x.max(y) < 0.0 {
                    -x2.min(y2)
                } else {
                    (if x > 0.0 { x2 } else { 0.0 }) + (if y > 0.0 { y2 } else { 0.0 })
                };
                d.signum() * d.abs().sqrt() / baba
            }
            Shape::Sphere { center, radius } => norm(sub(p, center)) - radius,
            Shape::Ellipsoid {
                center,
                radii,
                rotation_deg,
            } => {
                // The scaled-space field: exact on the surface, a 1-Lipschitz
                // bound elsewhere. See the variant's own documentation.
                let local = Shape::to_local(p, center, rotation_deg);
                let scaled = [
                    local[0] / radii[0],
                    local[1] / radii[1],
                    local[2] / radii[2],
                ];
                (norm(scaled) - 1.0) * radii[0].min(radii[1]).min(radii[2])
            }
            // The tube is everything within `radius` of its centre line, so
            // this is the whole of it - and it is the true distance, not a
            // bound. See [`Shape::Tube`].
            Shape::Tube {
                p1,
                p2,
                bend,
                radius,
            } => tube_distance(p, p1, p2, bend) - radius,
            // Exact, and two dimensional: the distance from the sample's place
            // in the cone's own meridian half plane to the trapezoid its
            // profile is. See [`Shape::Cone`].
            Shape::Cone {
                p1,
                p2,
                radius1,
                radius2,
            } => {
                let Some(frame) = cone_frame(p, p1, p2) else {
                    // No axis, so no solid: a cone of no length encloses
                    // nothing, and everything is infinitely far outside it.
                    return f64::INFINITY;
                };
                let q = [frame.rho, frame.along];
                let (_, distance) = cone_profile_closest_point(q, radius1, radius2, frame.len);
                if cone_contains_profile(q, radius1, radius2, frame.len) {
                    -distance
                } else {
                    distance
                }
            }
            // Exact: the in-plane distance to the triangle and the distance out
            // of the slab, combined the way the cylinder combines its wall and
            // its caps. See [`Shape::Triangle`].
            Shape::Triangle { a, b, c, thickness } => {
                let Some(normal) = triangle_normal(a, b, c) else {
                    // Three collinear points enclose nothing at any thickness,
                    // which is why a configuration may not hold them.
                    return f64::INFINITY;
                };
                let axial = dot(sub(p, a), normal);
                let projected = sub(p, scale(normal, axial));
                let across = norm(sub(
                    projected,
                    triangle_edge_closest_point(projected, a, b, c),
                ));
                let face = if triangle_contains_coplanar(projected, a, b, c, normal) {
                    -across
                } else {
                    across
                };
                let slab = axial.abs() - 0.5 * thickness;
                let outside = [face.max(0.0), slab.max(0.0)];
                let inside = face.max(slab).min(0.0);
                (outside[0] * outside[0] + outside[1] * outside[1]).sqrt() + inside
            }
        }
    }

    /// True when `p` lies inside or on the shape.
    pub fn contains(&self, p: Vec3) -> bool {
        self.signed_distance(p) <= constants::INSIDE_SDF_THRESHOLD_MM
    }

    /// The point of the shape's **surface** nearest to `p`, or `None` when
    /// there is no single such point to name.
    ///
    /// This is what the export's boundary clamp moves a vertex onto (see
    /// [`crate::mesh::clamp`]): the isosurface is extracted from a field
    /// sampled on a lattice, so a vertex can end up a fraction of a voxel inside
    /// a keepout, and projecting it here puts it exactly on the analytic surface
    /// the configuration described rather than on the voxel field's idea of it.
    ///
    /// Closed form for the box, the sphere, the capped cylinder, the tube, the
    /// cone and the triangular prism - the same shapes whose signed distance is
    /// exact. The **ellipsoid** has
    /// none: its nearest surface point is the root of a sextic, so it is reached
    /// by Newton steps on the shape's own field, which is exactly zero on the
    /// surface and smooth everywhere off the centre. The iteration is bounded by
    /// [`constants::BOUNDARY_CLAMP_MAX_STEPS`] and converges to
    /// [`constants::BOUNDARY_CLAMP_TOLERANCE_MM`]; it follows the field's
    /// gradient rather than solving for the true minimum distance, so what comes
    /// back is a point *on* the surface reached along its normal - which for the
    /// sub-voxel corrections this exists for differs from the nearest point by
    /// second order.
    ///
    /// `None` means the question has no answer here rather than that the shape
    /// has no surface: `p` at the centre of a sphere or an ellipsoid, on the
    /// centre line of a tube, or on the axis of a cylinder or a cone nearer its
    /// wall than its caps, is equidistant from a whole circle or sphere of
    /// surface points.
    /// A degenerate shape - a zero radius, a zero length - answers `None` too.
    /// The clamp leaves such a vertex where it is and says so.
    ///
    /// So does a **self-intersecting tube** - one that folds through the centre
    /// of its own bend, or whose two ends close on each other across the gap.
    /// There the surface is no longer the centre line carried out by a radius,
    /// and the construction this uses would hand back a point that is still
    /// inside the solid. It is refused rather than answered wrongly, so the
    /// clamp reaches its give-up in one pass instead of spending its whole
    /// budget rediscovering the same illegal point; see [`Arc::reach`] for where
    /// the line is and [`tube_self_intersects`], which is also what warns about
    /// such a shape when a configuration holds one.
    pub fn nearest_surface_point(&self, p: Vec3) -> Option<Vec3> {
        match *self {
            Shape::Box {
                min,
                max,
                rotation_deg,
            } => {
                let center = Shape::box_center(min, max);
                let half = [
                    0.5 * (max[0] - min[0]),
                    0.5 * (max[1] - min[1]),
                    0.5 * (max[2] - min[2]),
                ];
                if half.iter().any(|h| *h <= 0.0) {
                    return None;
                }
                let local = Shape::to_local(p, center, rotation_deg);
                let clamped: Vec3 = std::array::from_fn(|d| local[d].clamp(-half[d], half[d]));
                let surface = if clamped == local {
                    // Inside: the nearest surface is the face it is least deep
                    // under, and the point is that face's own plane.
                    let mut face = 0;
                    for d in 1..3 {
                        if half[d] - local[d].abs() < half[face] - local[face].abs() {
                            face = d;
                        }
                    }
                    let mut out = local;
                    out[face] = if local[face] < 0.0 {
                        -half[face]
                    } else {
                        half[face]
                    };
                    out
                } else {
                    // Outside: the clamp into the box *is* the nearest point of
                    // the solid, and a point outside can only be nearest to a
                    // point of its surface.
                    clamped
                };
                Some(Shape::from_local(surface, center, rotation_deg))
            }
            Shape::Cylinder { p1, p2, radius } => {
                let axis = sub(p2, p1);
                let len = norm(axis);
                if len <= 0.0 || radius <= 0.0 {
                    return None;
                }
                let unit_axis = scale(axis, 1.0 / len);
                let offset = sub(p, p1);
                let along = dot(offset, unit_axis);
                let radial = sub(offset, scale(unit_axis, along));
                let rho = norm(radial);

                // The lateral wall, at the height clamped into the cylinder and
                // a radius out from the axis.
                let overshoot = along - along.clamp(0.0, len);
                let lateral_distance = (overshoot * overshoot + (rho - radius).powi(2)).sqrt();
                // The nearer cap disc, at the same radial offset capped by the
                // radius - which is the cap rim for anything further out.
                let cap = if along <= 0.5 * len { 0.0 } else { len };
                let inward = if rho > radius {
                    scale(radial, radius / rho)
                } else {
                    radial
                };
                let cap_point = sum(p1, sum(scale(unit_axis, cap), inward));
                let cap_distance = norm(sub(cap_point, p));

                if rho <= 0.0 {
                    // On the axis: the wall is a whole circle of equally near
                    // points, so it can only be answered when a cap is nearer.
                    return (cap_distance <= lateral_distance).then_some(cap_point);
                }
                if lateral_distance < cap_distance {
                    let wall = sum(
                        p1,
                        sum(
                            scale(unit_axis, along.clamp(0.0, len)),
                            scale(radial, radius / rho),
                        ),
                    );
                    return Some(wall);
                }
                Some(cap_point)
            }
            Shape::Sphere { center, radius } => {
                let direction = sub(p, center);
                let distance = norm(direction);
                if distance <= 0.0 || radius <= 0.0 {
                    return None;
                }
                Some(sum(center, scale(direction, radius / distance)))
            }
            Shape::Ellipsoid {
                center,
                radii,
                rotation_deg,
            } => {
                if radii.iter().any(|r| *r <= 0.0) {
                    return None;
                }
                let smallest = radii[0].min(radii[1]).min(radii[2]);
                let mut local = Shape::to_local(p, center, rotation_deg);
                for _ in 0..constants::BOUNDARY_CLAMP_MAX_STEPS {
                    let scaled: Vec3 = std::array::from_fn(|d| local[d] / radii[d]);
                    let reach = norm(scaled);
                    if reach <= 0.0 {
                        // The centre, where every surface point is as near as
                        // every other.
                        return None;
                    }
                    let value = (reach - 1.0) * smallest;
                    if value.abs() <= constants::BOUNDARY_CLAMP_TOLERANCE_MM {
                        return Some(Shape::from_local(local, center, rotation_deg));
                    }
                    // The field's own gradient, which the scaled form gives in
                    // closed form: `d/dx (|x / r| - 1) * min(r)`.
                    let gradient: Vec3 = std::array::from_fn(|d| {
                        smallest * local[d] / (radii[d] * radii[d] * reach)
                    });
                    let slope = norm(gradient);
                    if slope <= 0.0 || !slope.is_finite() {
                        return None;
                    }
                    local = sub(local, scale(gradient, value / (slope * slope)));
                }
                None
            }
            Shape::Tube {
                p1,
                p2,
                bend,
                radius,
            } => {
                // A tube that overlaps itself - through its own bend, or across
                // the gap between its ends - has swallowed part of its inside,
                // and the offset below stops being a projection there: it would
                // answer with a point that is still inside the solid.
                if radius <= 0.0 || tube_self_intersects(p1, p2, bend, radius) {
                    return None;
                }
                // The tube is its centre line grown by the radius, so the
                // nearest surface point is a radius out from the nearest point
                // of that line - exactly, inside and out, because the distance
                // to the line is 1-Lipschitz and this attains the bound.
                let on_line = tube_closest_point(p, p1, p2, bend);
                let direction = sub(p, on_line);
                let distance = norm(direction);
                if distance <= 0.0 {
                    return None;
                }
                Some(sum(on_line, scale(direction, radius / distance)))
            }
            Shape::Cone {
                p1,
                p2,
                radius1,
                radius2,
            } => {
                if radius1 <= 0.0 || radius2 < 0.0 {
                    return None;
                }
                let frame = cone_frame(p, p1, p2)?;
                let q = [frame.rho, frame.along];
                let (surface, _) = cone_profile_closest_point(q, radius1, radius2, frame.len);
                // Out of the half plane again: the height along the axis, and
                // the profile's radius carried out along the sample's own
                // radial direction.
                let outward = if frame.rho > 0.0 {
                    scale(frame.radial, 1.0 / frame.rho)
                } else if surface[0] > 0.0 {
                    // On the axis, nearest to a whole circle of the wall or of
                    // a cap rim: there is no single point to name.
                    return None;
                } else {
                    // On the axis and nearest a cap centre or the apex, which
                    // is one point and is on the axis too.
                    [0.0; 3]
                };
                Some(sum(
                    p1,
                    sum(scale(frame.unit, surface[1]), scale(outward, surface[0])),
                ))
            }
            Shape::Triangle { a, b, c, thickness } => {
                if thickness <= 0.0 {
                    return None;
                }
                let normal = triangle_normal(a, b, c)?;
                let half = 0.5 * thickness;
                let axial = dot(sub(p, a), normal);
                let projected = sub(p, scale(normal, axial));
                let on_face = triangle_contains_coplanar(projected, a, b, c, normal);
                let on_edge = triangle_edge_closest_point(projected, a, b, c);
                if !on_face || axial.abs() > half {
                    // Outside: the sample clamped into the solid - into the
                    // triangle across the plane, into the slab along the
                    // normal - and a point outside a solid can only be nearest
                    // to a point of its surface.
                    let across = if on_face { projected } else { on_edge };
                    return Some(sum(across, scale(normal, axial.clamp(-half, half))));
                }
                // Inside: the nearest surface is whichever it is least deep
                // under - a side wall at its own height, or the nearer face.
                if norm(sub(projected, on_edge)) <= half - axial.abs() {
                    Some(sum(on_edge, scale(normal, axial)))
                } else {
                    Some(sum(
                        projected,
                        scale(normal, if axial < 0.0 { -half } else { half }),
                    ))
                }
            }
        }
    }

    /// Half extents of a rotated ellipsoid, which are exact.
    ///
    /// The ellipsoid is the image of the unit ball,
    /// `E = { c + R * diag(r) * u : |u| <= 1 }`, so its extent along the world
    /// axis `e_i` is its support function there:
    ///
    /// ```text
    /// max_{|u| <= 1} e_i . (R diag(r) u) = |(R diag(r))^T e_i|
    ///                                    = |row i of R diag(r)|
    ///                                    = sqrt(sum_j (R[i][j] * r_j)^2)
    /// ```
    ///
    /// which is attained by the `u` pointing along that row - a real point of
    /// the surface, so the box touches it and cannot be shrunk. Unrotated, `R`
    /// is the identity and the row collapses to `r_i`: the centre +- radii box
    /// the ellipsoid was always bounded by, to the bit.
    pub fn ellipsoid_half_extents(radii: Vec3, rotation_deg: Vec3) -> Vec3 {
        if is_unrotated(rotation_deg) {
            return radii;
        }
        let matrix = rotation_matrix(rotation_deg);
        std::array::from_fn(|i| {
            (0..3)
                .map(|j| {
                    let component = matrix[i][j] * radii[j];
                    component * component
                })
                .sum::<f64>()
                .sqrt()
        })
    }

    /// Tight axis aligned bounds of the shape.
    ///
    /// A rotated box is bounded by the extremes of its own eight corners, which
    /// is exact: a box is the convex hull of them, as a triangular prism is of
    /// its six. A rotated ellipsoid has no
    /// corners and is bounded by its support function instead, which is exact
    /// for the same reason; see [`Shape::ellipsoid_half_extents`], and a cone
    /// is bounded by the support function of the two cap discs it is the hull
    /// of. A tube is
    /// bounded by the extremes of its centre line grown by its radius, which is
    /// exact because the tube is exactly that curve grown by that radius: the
    /// point that reaches furthest along an axis is the curve's own furthest
    /// point, offset a radius along it.
    pub fn bounds(&self) -> Aabb {
        match *self {
            Shape::Box {
                min,
                max,
                rotation_deg,
            } => {
                if is_unrotated(rotation_deg) {
                    return Aabb { min, max };
                }
                let mut bb = Aabb::empty();
                for corner in Shape::box_corners(min, max, rotation_deg) {
                    for (d, value) in corner.iter().enumerate() {
                        bb.min[d] = bb.min[d].min(*value);
                        bb.max[d] = bb.max[d].max(*value);
                    }
                }
                bb
            }
            Shape::Cylinder { p1, p2, radius } => {
                // The cap discs are perpendicular to the axis; their extent on
                // axis d is radius * sqrt(1 - (axis_d/|axis|)^2).
                let axis = sub(p2, p1);
                let len = norm(axis);
                let mut bb = Aabb::empty();
                for d in 0..3 {
                    let unit = if len > 0.0 { axis[d] / len } else { 0.0 };
                    let spread = radius * (1.0 - unit * unit).max(0.0).sqrt();
                    bb.min[d] = p1[d].min(p2[d]) - spread;
                    bb.max[d] = p1[d].max(p2[d]) + spread;
                }
                bb
            }
            Shape::Sphere { center, radius } => Aabb {
                min: [center[0] - radius, center[1] - radius, center[2] - radius],
                max: [center[0] + radius, center[1] + radius, center[2] + radius],
            },
            Shape::Ellipsoid {
                center,
                radii,
                rotation_deg,
            } => {
                let half = Shape::ellipsoid_half_extents(radii, rotation_deg);
                Aabb {
                    min: sub(center, half),
                    max: sum(center, half),
                }
            }
            Shape::Tube {
                p1,
                p2,
                bend,
                radius,
            } => {
                // The extremes of an arc on one axis are its two ends and the
                // points where its tangent is at a right angle to that axis -
                // where `d/dphi (start_d cos phi + turn_d sin phi)` vanishes,
                // which is a pair of angles half a turn apart. Only the ones
                // the arc actually reaches count.
                let arc = tube_arc(p1, p2, bend);
                let mut bb = Aabb::empty();
                for d in 0..3 {
                    let mut low = p1[d].min(p2[d]);
                    let mut high = p1[d].max(p2[d]);
                    if let Some(arc) = arc {
                        let stationary = arc.turn[d].atan2(arc.start[d]);
                        for half_turn in 0..2 {
                            let angle = (stationary + half_turn as f64 * PI).rem_euclid(TAU);
                            if angle <= arc.span {
                                let reach = arc.point(angle)[d];
                                low = low.min(reach);
                                high = high.max(reach);
                            }
                        }
                    }
                    bb.min[d] = low - radius;
                    bb.max[d] = high + radius;
                }
                bb
            }
            Shape::Cone {
                p1,
                p2,
                radius1,
                radius2,
            } => {
                // A frustum is the convex hull of its two cap discs, so its
                // extreme along an axis is the further of the two discs' own -
                // and a disc perpendicular to the axis spreads
                // `radius * sqrt(1 - (axis_d/|axis|)^2)` along it, which is the
                // cylinder's rule with a radius per end.
                let axis = sub(p2, p1);
                let len = norm(axis);
                let mut bb = Aabb::empty();
                for d in 0..3 {
                    let unit = if len > 0.0 { axis[d] / len } else { 0.0 };
                    let spread = (1.0 - unit * unit).max(0.0).sqrt();
                    let (first, second) = (radius1 * spread, radius2 * spread);
                    bb.min[d] = (p1[d] - first).min(p2[d] - second);
                    bb.max[d] = (p1[d] + first).max(p2[d] + second);
                }
                bb
            }
            Shape::Triangle { a, b, c, thickness } => {
                // The prism is the convex hull of its six corners - the three
                // vertices, half a thickness either side of their plane - so
                // their extremes are its bounds exactly, with nothing to
                // inflate by afterwards.
                let offset = match triangle_normal(a, b, c) {
                    Some(normal) => scale(normal, 0.5 * thickness),
                    None => [0.0; 3],
                };
                let mut bb = Aabb::empty();
                for corner in [a, b, c] {
                    for side in [-1.0, 1.0] {
                        let point = sum(corner, scale(offset, side));
                        for (d, value) in point.iter().enumerate() {
                            bb.min[d] = bb.min[d].min(*value);
                            bb.max[d] = bb.max[d].max(*value);
                        }
                    }
                }
                bb
            }
        }
    }

    /// True when the shape overlaps itself, which only a tube can do.
    ///
    /// Asked of a [`Shape`] rather than of a tube's four numbers so that a
    /// caller holding a shape list - the `[[keepout]]` union, say - can ask each
    /// member without knowing which variant it is. See
    /// [`tube_self_intersects`] for what it means and what it costs.
    pub fn self_intersects(&self) -> bool {
        match *self {
            Shape::Tube {
                p1,
                p2,
                bend,
                radius,
            } => tube_self_intersects(p1, p2, bend, radius),
            // Written out rather than left to the catch-all below, because the
            // answer is a property of the shape rather than an absence of one:
            // a frustum is the convex hull of its two cap discs and a prism the
            // hull of its six corners, and no part of a convex solid can reach
            // any other part of it.
            Shape::Cone { .. } | Shape::Triangle { .. } => false,
            _ => false,
        }
    }

    /// Smallest extent of the shape, used to reject degenerate inputs.
    ///
    /// A box's edge lengths and an ellipsoid's semi-axes are what they are
    /// whichever way the shape is turned, so rotation does not enter here. An
    /// ellipsoid answers with its smallest *radius*, as a sphere does, and so
    /// does a tube: a capsule of no length is a ball rather than a degenerate
    /// solid, so its radius is the only measurement of it that can vanish.
    /// That is what makes it unlike a cylinder, which answers with the smaller
    /// of its radius and its length because a cylinder of no length is a disc.
    ///
    /// A **cone** answers with the smaller of its length and its *first*
    /// radius. The second is deliberately left out: zero is the legal apex of a
    /// true cone rather than a degenerate shape, and measuring it here would
    /// reject every cone that comes to a point. What can still make a cone
    /// nothing is a wide end of no width or an axis of no length, and those are
    /// the two this answers with.
    ///
    /// A **triangular prism** answers with the smaller of its thickness and the
    /// diameter of its inscribed circle - the largest ball that fits inside its
    /// triangle, which is the "smallest ball that fits" the minimum feature
    /// size is written in. A prism whose triangle is a sliver is thin however
    /// far apart its vertices are, and this is what says so.
    pub fn min_extent(&self) -> f64 {
        match *self {
            Shape::Box { min, max, .. } => (0..3)
                .map(|d| max[d] - min[d])
                .fold(f64::INFINITY, f64::min),
            Shape::Cylinder { p1, p2, radius } => norm(sub(p2, p1)).min(radius),
            Shape::Sphere { radius, .. } => radius,
            Shape::Ellipsoid { radii, .. } => radii.iter().copied().fold(f64::INFINITY, f64::min),
            Shape::Tube { radius, .. } => radius,
            Shape::Cone {
                p1, p2, radius1, ..
            } => norm(sub(p2, p1)).min(radius1),
            Shape::Triangle { a, b, c, thickness } => {
                thickness.min(2.0 * triangle_inradius(a, b, c))
            }
        }
    }
}

/// An ordered list of add/subtract shape operations.
#[derive(Debug, Clone, Default)]
pub struct Csg {
    entries: Vec<(CsgOp, Shape)>,
}

impl Csg {
    /// Build a CSG tree from an ordered list of operations.
    pub fn new(entries: Vec<(CsgOp, Shape)>) -> Self {
        Csg { entries }
    }

    /// True when the tree has no operations at all.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Signed distance of the combined solid, evaluated in list order:
    /// `add` unions (min), `subtract` cuts (max with the negated distance).
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        let mut d = f64::INFINITY;
        for (op, shape) in &self.entries {
            let ds = shape.signed_distance(p);
            d = match op {
                CsgOp::Add => d.min(ds),
                CsgOp::Subtract => d.max(-ds),
            };
        }
        d
    }

    /// True when `p` lies inside the combined solid.
    pub fn contains(&self, p: Vec3) -> bool {
        self.signed_distance(p) <= constants::INSIDE_SDF_THRESHOLD_MM
    }

    /// Union of the bounds of the additive entries. Subtractions can only
    /// shrink the solid, so they do not enlarge the bounding box.
    pub fn bounds(&self) -> Aabb {
        let mut bb = Aabb::empty();
        for (op, shape) in &self.entries {
            if *op == CsgOp::Add {
                bb = bb.union(&shape.bounds());
            }
        }
        bb
    }
}

/// A union of shapes, used for keepout, keepin and boundary condition regions
/// where no subtraction is possible.
#[derive(Debug, Clone, Default)]
pub struct ShapeUnion {
    shapes: Vec<Shape>,
}

impl ShapeUnion {
    /// Build a union from a shape list.
    pub fn new(shapes: Vec<Shape>) -> Self {
        ShapeUnion { shapes }
    }

    /// True when the union holds no shapes.
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// The member shapes, in configuration order.
    ///
    /// Anything that has to treat the union as the several *places* it was
    /// written as reads this rather than the combined field: the export's
    /// boundary clamp projects a vertex onto the surface of the member it
    /// violates, and which member that is, is a question only the members can
    /// answer.
    pub fn shapes(&self) -> &[Shape] {
        &self.shapes
    }

    /// Signed distance of the combined solid: the minimum over the members.
    ///
    /// An empty union is infinitely far away, which is the identity of the
    /// minimum and the honest reading of "nothing is forbidden here".
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        self.shapes
            .iter()
            .map(|s| s.signed_distance(p))
            .fold(f64::INFINITY, f64::min)
    }

    /// True when `p` lies inside any member shape.
    pub fn contains(&self, p: Vec3) -> bool {
        self.shapes.iter().any(|s| s.contains(p))
    }

    /// Union of the member bounds.
    pub fn bounds(&self) -> Aabb {
        let mut bb = Aabb::empty();
        for s in &self.shapes {
            bb = bb.union(&s.bounds());
        }
        bb
    }
}

/// The analytic surfaces a part's geometry has to respect: the solid it must
/// stay inside, and the union of regions it must stay out of.
///
/// The set the classifier voxelizes, kept as the shapes the configuration wrote
/// rather than as the cells they produced. Classification samples a cell centre
/// and can therefore resolve a boundary no finer than half a voxel; these are
/// what the export's boundary clamp measures the extracted surface against so
/// the exported bore is the cylinder that was asked for. Any of them may be
/// empty - a configuration with no `[[keepout]]`, or a caller that has no domain
/// to hold the mesh to - and an empty one constrains nothing.
///
/// The solid is the domain **union** the keepins: a keepin takes precedence over
/// the domain ([`crate::grid::Grid::classify`]), so material inside one is
/// material even where the keepin sticks out of the domain, and the keepin's own
/// outer skin is the surface there. Keepins only ever enlarge the solid, so
/// adding one can never make a position the clamp accepted illegal.
#[derive(Debug, Clone, Default)]
pub struct Boundaries {
    /// The design domain, as the ordered `[[domain]]` CSG tree.
    pub domain: Csg,
    /// The union of the `[[keepout]]` regions.
    pub keepout: ShapeUnion,
    /// The union of the `[[keepin]]` regions.
    pub keepin: ShapeUnion,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two unit vectors at right angles to `axis` and to each other, for
    /// walking round a tube's surface. The viewer has one of these, but it is
    /// behind a feature gate and this module is not.
    fn perpendicular_basis(axis: Vec3) -> (Vec3, Vec3) {
        let axis = scale(axis, 1.0 / norm(axis));
        let smallest = (0..3)
            .min_by(|a, b| axis[*a].abs().total_cmp(&axis[*b].abs()))
            .expect("three axes");
        let mut helper = [0.0; 3];
        helper[smallest] = 1.0;
        let u = cross(helper, axis);
        let u = scale(u, 1.0 / norm(u));
        (u, cross(axis, u))
    }

    #[test]
    fn box_sdf_signs_and_distances() {
        let b = Shape::axis_aligned_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        assert!(b.signed_distance([5.0, 5.0, 5.0]) < 0.0);
        assert!((b.signed_distance([5.0, 5.0, 5.0]) + 5.0).abs() < 1e-12);
        assert!((b.signed_distance([15.0, 5.0, 5.0]) - 5.0).abs() < 1e-12);
        assert!(b.contains([0.0, 0.0, 0.0]));
        assert!(!b.contains([10.0001, 5.0, 5.0]));
    }

    #[test]
    fn sphere_sdf_matches_radial_distance() {
        let s = Shape::Sphere {
            center: [1.0, 2.0, 3.0],
            radius: 4.0,
        };
        assert!((s.signed_distance([1.0, 2.0, 3.0]) + 4.0).abs() < 1e-12);
        assert!((s.signed_distance([1.0 + 6.0, 2.0, 3.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cylinder_sdf_respects_caps_and_radius() {
        let c = Shape::Cylinder {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            radius: 3.0,
        };
        // Centre of the axis: distance to the nearest surface is the radius.
        assert!((c.signed_distance([0.0, 0.0, 5.0]) + 3.0).abs() < 1e-9);
        // Radially outside.
        assert!((c.signed_distance([5.0, 0.0, 5.0]) - 2.0).abs() < 1e-9);
        // Beyond a cap.
        assert!((c.signed_distance([0.0, 0.0, 14.0]) - 4.0).abs() < 1e-9);
        assert!(!c.contains([0.0, 0.0, -0.001]));
        assert!(c.contains([0.0, 0.0, 0.0]));
    }

    #[test]
    fn cylinder_bounds_contain_the_caps() {
        let c = Shape::Cylinder {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            radius: 3.0,
        };
        let bb = c.bounds();
        assert!((bb.min[0] + 3.0).abs() < 1e-12);
        assert!((bb.max[2] - 10.0).abs() < 1e-12);
        assert!(bb.min[2].abs() < 1e-12);
    }

    #[test]
    fn csg_order_is_respected() {
        let csg = Csg::new(vec![
            (
                CsgOp::Add,
                Shape::axis_aligned_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
            ),
            (
                CsgOp::Subtract,
                Shape::Sphere {
                    center: [5.0, 5.0, 5.0],
                    radius: 2.0,
                },
            ),
        ]);
        assert!(csg.contains([1.0, 1.0, 1.0]));
        assert!(!csg.contains([5.0, 5.0, 5.0]));
        let bb = csg.bounds();
        assert_eq!(bb.min, [0.0, 0.0, 0.0]);
        assert_eq!(bb.max, [10.0, 10.0, 10.0]);
    }

    /// A tube on a circular arc of `radius` spanning `span` radians, written as
    /// the three points a configuration would hold.
    ///
    /// The arc is laid in the z = 0 plane starting on +x, so `tube_arc` recovers
    /// exactly this circle: the bend point is placed at half the span, which is
    /// what puts it strictly inside the arc rather than on the other one.
    fn arc_tube(arc_radius: f64, span: f64, tube_radius: f64) -> Shape {
        let at = |angle: f64| {
            let (sin, cos) = angle.sin_cos();
            [arc_radius * cos, arc_radius * sin, 0.0]
        };
        Shape::Tube {
            p1: at(0.0),
            p2: at(span),
            bend: Some(at(0.5 * span)),
            radius: tube_radius,
        }
    }

    /// Brute force: does the offset construction - a radius out from the nearest
    /// centre-line point - land on the surface everywhere inside `shape`?
    ///
    /// This is the question [`Shape::nearest_surface_point`] answers for a tube,
    /// asked of the *field* instead of the formula, so the predicate that gates
    /// it can be checked against the geometry rather than against its own
    /// derivation. Returns how far inside the solid the worst answer landed;
    /// zero means the construction is sound everywhere this sampled.
    fn worst_offset_error(shape: &Shape) -> f64 {
        let Shape::Tube {
            p1,
            p2,
            bend,
            radius,
        } = *shape
        else {
            panic!("this probe is about tubes");
        };
        let bb = shape.bounds();
        let steps = 36;
        let mut worst: f64 = 0.0;
        for i in 0..=steps {
            for j in 0..=steps {
                for k in 0..=steps {
                    let p = [
                        bb.min[0] + (bb.max[0] - bb.min[0]) * i as f64 / steps as f64,
                        bb.min[1] + (bb.max[1] - bb.min[1]) * j as f64 / steps as f64,
                        bb.min[2] + (bb.max[2] - bb.min[2]) * k as f64 / steps as f64,
                    ];
                    if shape.signed_distance(p) >= 0.0 {
                        continue;
                    }
                    let on_line = tube_closest_point(p, p1, p2, bend);
                    let outward = sub(p, on_line);
                    let reach = norm(outward);
                    if reach <= 0.0 {
                        continue;
                    }
                    let offset = sum(on_line, scale(outward, radius / reach));
                    // Negative means the "surface" point is inside the solid.
                    worst = worst.max(-shape.signed_distance(offset));
                }
            }
        }
        worst
    }

    /// The predicate that gates the tube projection, checked against the field
    /// it is a claim about rather than against the derivation behind it.
    ///
    /// Two modes make a tube overlap itself and the reach is the smaller of
    /// them: the **fold** through the centre of its own bend, and the **gap**
    /// between two ends that have closed on each other. The second is the one a
    /// naive "chord shorter than the diameter" test gets wrong in both
    /// directions, so every claim below is probed:
    ///
    /// * a major arc whose ends have closed really does fold, and the predicate
    ///   flags it;
    /// * a **minor** arc whose chord is under the diameter - stubby, fat, ends
    ///   almost touching - really does **not**, and the predicate stays silent;
    /// * the boundary is exact from both sides at a hundredth of the radius.
    #[test]
    fn the_reach_of_an_arc_is_where_its_tube_starts_to_overlap_itself() {
        // A 90 degree arc whose chord is a hair under two tube radii: the case
        // the naive chord test condemns and the geometry acquits. The ends are
        // 1.90 mm apart with a tube 1.00 mm thick, and the projection is still
        // exact everywhere inside it, because what lies between those ends is
        // the arc rather than empty space.
        let stubby_arc_radius = 1.9 / 2.0_f64.sqrt();
        let stubby = arc_tube(stubby_arc_radius, 0.5 * PI, 1.0);
        let Shape::Tube { p1, p2, .. } = stubby else {
            unreachable!()
        };
        let chord = norm(sub(p2, p1));
        assert!((chord - 1.9).abs() < 1e-9, "chord {chord}");
        assert!(chord < 2.0, "the fixture must be inside the naive test");
        assert!(!stubby.self_intersects(), "a minor arc was condemned");
        let error = worst_offset_error(&stubby);
        assert!(
            error < 1e-9,
            "a stubby minor arc does fold after all, by {error} mm - the simple chord test would \
             have been right and this predicate is wrong"
        );

        // The same span at a radius that does fold it: the bend bound.
        assert!(arc_tube(stubby_arc_radius, 0.5 * PI, stubby_arc_radius * 1.01).self_intersects());
        assert!(!arc_tube(stubby_arc_radius, 0.5 * PI, stubby_arc_radius * 0.99).self_intersects());

        // Major arcs, swept over the gap the reviewer swept: 5 to 40 degrees of
        // opening on an arc of radius 6, with a tube of radius 1.8 - well under
        // the fold bound, so that alone would call every one of them sound.
        let arc_radius = 6.0;
        for gap_deg in [5.0_f64, 10.0, 20.0, 30.0, 40.0] {
            let span = TAU - gap_deg.to_radians();
            let half_chord = arc_radius * (0.5 * span).sin();
            let shape = arc_tube(arc_radius, span, 1.8);
            let flagged = shape.self_intersects();
            let error = worst_offset_error(&shape);
            assert_eq!(
                flagged,
                1.8 >= half_chord,
                "gap {gap_deg} deg: half chord {half_chord}"
            );
            if flagged {
                assert!(
                    error > 0.1,
                    "gap {gap_deg} deg was flagged but the construction is sound: {error}"
                );
            } else {
                assert!(
                    error < 1e-9,
                    "gap {gap_deg} deg was passed but the construction is wrong by {error} mm"
                );
            }
        }

        // The boundary itself, from both sides, on a major arc where the gap
        // bound is what binds: a hundredth of the reach either way.
        let span = TAU - 20.0_f64.to_radians();
        let reach = arc_radius * (0.5 * span).sin();
        assert!(reach < arc_radius, "the gap must be the binding bound here");
        let under = arc_tube(arc_radius, span, 0.99 * reach);
        let over = arc_tube(arc_radius, span, 1.01 * reach);
        assert!(!under.self_intersects() && over.self_intersects());
        assert!(
            worst_offset_error(&under) < 1e-9,
            "just under the reach the construction must still be exact"
        );
        assert!(
            worst_offset_error(&over) > 0.0,
            "just over the reach it must not be"
        );

        // And the reach itself is the smaller of the two bounds, continuous
        // through the half turn where they meet.
        let arc_of = |span: f64| {
            let Shape::Tube { p1, p2, bend, .. } = arc_tube(arc_radius, span, 1.0) else {
                unreachable!()
            };
            tube_arc(p1, p2, bend).expect("an arc")
        };
        for span in [0.25 * PI, 0.9 * PI, PI] {
            assert!((arc_of(span).reach() - arc_radius).abs() < 1e-9, "{span}");
        }
        for span in [1.1 * PI, 1.5 * PI, 1.95 * PI] {
            let arc = arc_of(span);
            assert!(arc.reach() < arc_radius, "{span}");
            assert!((arc.reach() - arc_radius * (0.5 * span).sin()).abs() < 1e-9);
        }
    }

    /// Only a tube can overlap itself, and a shape is asked rather than told:
    /// a caller holding a list of them names no variant.
    #[test]
    fn only_a_tube_can_overlap_itself_and_only_past_its_reach() {
        for shape in [
            Shape::axis_aligned_box([0.0; 3], [1.0; 3]),
            Shape::Box {
                min: [0.0; 3],
                max: [1.0; 3],
                rotation_deg: [10.0, 20.0, 30.0],
            },
            Shape::Sphere {
                center: [0.0; 3],
                radius: 1e6,
            },
            Shape::Cylinder {
                p1: [0.0; 3],
                p2: [0.0, 0.0, 1e-3],
                radius: 1e6,
            },
            Shape::Ellipsoid {
                center: [0.0; 3],
                radii: [1e6, 1e-3, 1e-3],
                rotation_deg: [0.0; 3],
            },
        ] {
            assert!(!shape.self_intersects(), "{shape:?}");
        }

        // A half circle of radius 5, and tubes either side of that radius.
        let (p1, bend, p2) = ([5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-5.0, 0.0, 0.0]);
        let arc = tube_arc(p1, p2, Some(bend)).expect("an arc");
        assert!((arc.radius - 5.0).abs() < 1e-12);
        assert!(!tube_self_intersects(p1, p2, Some(bend), 4.9));
        assert!(tube_self_intersects(p1, p2, Some(bend), 5.1));
        // Whichever way the same tube is written: the arc is the circumcircle
        // through the three points, so the rule does not depend on the order.
        assert!(tube_self_intersects(p2, p1, Some(bend), 5.1));
        // A tube with no arc has no bend to overlap, however thick it is - and
        // a bend inside the collinear epsilon has no arc either.
        assert!(!tube_self_intersects(p1, p2, None, 1e6));
        assert!(!tube_self_intersects(
            p1,
            p2,
            Some([0.0, 0.5 * constants::TUBE_COLLINEAR_EPS_MM, 0.0]),
            1e6
        ));
    }

    /// A union's field is the minimum over its members, everywhere and by
    /// construction, which is what makes "is this point in *any* keepout" and
    /// "how deep is it in the nearest one" the same question.
    #[test]
    fn a_union_measures_from_the_nearest_of_its_members() {
        let shapes = vec![
            Shape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 4.0,
            },
            Shape::axis_aligned_box([10.0, -2.0, -2.0], [14.0, 2.0, 2.0]),
            Shape::Cylinder {
                p1: [0.0, 20.0, 0.0],
                p2: [0.0, 20.0, 6.0],
                radius: 1.5,
            },
        ];
        let union = ShapeUnion::new(shapes.clone());
        assert_eq!(union.shapes(), shapes.as_slice());
        for p in [
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [0.0, 20.0, 3.0],
            [-30.0, 5.0, 11.0],
            [9.0, 1.9, 1.9],
        ] {
            let expected = shapes
                .iter()
                .map(|s| s.signed_distance(p))
                .fold(f64::INFINITY, f64::min);
            assert_eq!(union.signed_distance(p), expected, "at {p:?}");
            // And the sign is what `contains` has always answered.
            assert_eq!(
                union.contains(p),
                union.signed_distance(p) <= constants::INSIDE_SDF_THRESHOLD_MM,
                "at {p:?}"
            );
        }
        // Nothing forbidden is infinitely far away, which is the identity of
        // the minimum and what lets a caller skip the union entirely.
        let empty = ShapeUnion::default();
        assert_eq!(empty.signed_distance([1.0, 2.0, 3.0]), f64::INFINITY);
        assert!(empty.shapes().is_empty());
    }

    /// The composition order is what makes the angles mean anything, so it is
    /// pinned against hand-computed images rather than against itself.
    #[test]
    fn the_rotation_matrix_is_extrinsic_xyz_about_the_world_axes() {
        // A quarter turn about z takes +x to +y.
        let z90 = rotation_matrix([0.0, 0.0, 90.0]);
        let turned = rotate(&z90, [1.0, 0.0, 0.0]);
        assert!((turned[0]).abs() < 1e-12 && (turned[1] - 1.0).abs() < 1e-12);
        // About y it takes +z to +x.
        let y90 = rotation_matrix([0.0, 90.0, 0.0]);
        let turned = rotate(&y90, [0.0, 0.0, 1.0]);
        assert!((turned[0] - 1.0).abs() < 1e-12 && turned[2].abs() < 1e-12);
        // About x it takes +y to +z.
        let x90 = rotation_matrix([90.0, 0.0, 0.0]);
        let turned = rotate(&x90, [0.0, 1.0, 0.0]);
        assert!((turned[1]).abs() < 1e-12 && (turned[2] - 1.0).abs() < 1e-12);

        // Extrinsic XYZ is R = Rz * Ry * Rx: x first, about the *world* axes.
        let combined = rotation_matrix([90.0, 90.0, 0.0]);
        let expected = rotate(&y90, rotate(&x90, [0.0, 1.0, 0.0]));
        let got = rotate(&combined, [0.0, 1.0, 0.0]);
        for d in 0..3 {
            assert!((got[d] - expected[d]).abs() < 1e-12, "{got:?} {expected:?}");
        }

        // The inverse really is the transpose, and the matrix is orthonormal.
        let m = rotation_matrix([13.0, -47.0, 91.5]);
        let v = [3.0, -1.0, 2.0];
        let round_trip = rotate_inverse(&m, rotate(&m, v));
        for d in 0..3 {
            assert!((round_trip[d] - v[d]).abs() < 1e-12);
        }
        assert!((length(rotate(&m, v)) - length(v)).abs() < 1e-12);

        assert!(is_unrotated([0.0, -0.0, 0.0]));
        assert!(!is_unrotated([0.0, 0.0, 1e-9]));
    }

    #[test]
    fn degrees_are_normalized_into_a_half_turn_either_way() {
        for (raw, expected) in [
            (0.0, 0.0),
            (45.0, 45.0),
            (180.0, 180.0),
            (-180.0, 180.0),
            (270.0, -90.0),
            (360.0, 0.0),
            (720.0 + 22.5, 22.5),
            (-450.0, -90.0),
        ] {
            assert!(
                (normalize_degrees(raw) - expected).abs() < 1e-9,
                "{raw} normalized to {}",
                normalize_degrees(raw)
            );
        }
    }

    /// A 45 degree turn about z of a square prism: every face and corner is
    /// somewhere hand-computable, and that is where the distances are checked.
    #[test]
    fn a_rotated_box_measures_from_its_turned_faces() {
        let half = 5.0;
        let b = Shape::Box {
            min: [-half, -half, -half],
            max: [half, half, half],
            rotation_deg: [0.0, 0.0, 45.0],
        };
        // The centre is the centre whatever the rotation.
        assert!((b.signed_distance([0.0; 3]) + half).abs() < 1e-12);
        // A corner of the unrotated cube now lies on the +y axis, at the half
        // diagonal; a point that far out on +y is exactly on the surface.
        let diagonal = half * 2.0_f64.sqrt();
        assert!(b.signed_distance([0.0, diagonal, 0.0]).abs() < 1e-12);
        // The point that used to sit on the +x face is now inside: that face
        // has swung away, and the nearest one is `half * cos(45)` off.
        let sunk = half - half * 0.5_f64.sqrt();
        assert!((b.signed_distance([half, 0.0, 0.0]) + sunk).abs() < 1e-12);
        // Straight out along the turned face normal.
        let normal = [0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0];
        let point = scale(normal, half + 3.0);
        assert!((b.signed_distance(point) - 3.0).abs() < 1e-12);
        // z is untouched by a turn about z.
        assert!((b.signed_distance([0.0, 0.0, half + 2.0]) - 2.0).abs() < 1e-12);

        // The bounds are the extremes of the eight corners, exactly.
        let bb = b.bounds();
        for d in 0..2 {
            assert!((bb.min[d] + diagonal).abs() < 1e-12, "{bb:?}");
            assert!((bb.max[d] - diagonal).abs() < 1e-12, "{bb:?}");
        }
        assert!((bb.min[2] + half).abs() < 1e-12 && (bb.max[2] - half).abs() < 1e-12);
        // Turning it does not change what it is made of.
        assert!((b.min_extent() - 2.0 * half).abs() < 1e-12);
    }

    /// A whole turn of every axis is the identity, and the corners of a box
    /// rotated by 90 degrees are the corners of the box the other way round.
    #[test]
    fn a_rotated_box_matches_the_box_the_rotation_maps_it_onto() {
        let rotated = Shape::Box {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 4.0, 2.0],
            rotation_deg: [0.0, 0.0, 90.0],
        };
        // Centre [5, 2, 1]; a quarter turn about z swaps the 10 mm x extent and
        // the 4 mm y one about that centre.
        let expected = Shape::axis_aligned_box([3.0, -3.0, 0.0], [7.0, 7.0, 2.0]);
        for p in [
            [5.0, 2.0, 1.0],
            [3.0, 8.0, 1.0],
            [0.0, 0.0, 0.0],
            [-4.0, 2.0, 7.0],
            [8.5, 2.0, 1.0],
        ] {
            assert!(
                (rotated.signed_distance(p) - expected.signed_distance(p)).abs() < 1e-9,
                "at {p:?}: {} against {}",
                rotated.signed_distance(p),
                expected.signed_distance(p)
            );
        }
        let (a, b) = (rotated.bounds(), expected.bounds());
        for d in 0..3 {
            assert!((a.min[d] - b.min[d]).abs() < 1e-9 && (a.max[d] - b.max[d]).abs() < 1e-9);
        }
    }

    /// The zero level set is what every consumer reads, so it is checked
    /// against points whose position on the surface is known by hand: the three
    /// axis points, an oblique one, and the centre.
    #[test]
    fn an_ellipsoid_is_zero_exactly_on_its_own_surface() {
        let radii = [6.0, 3.0, 2.0];
        let center = [1.0, -2.0, 4.0];
        let e = Shape::Ellipsoid {
            center,
            radii,
            rotation_deg: [0.0; 3],
        };
        // The six axis points sit at exactly a radius from the centre.
        for d in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut p = center;
                p[d] += sign * radii[d];
                assert!(e.signed_distance(p).abs() < 1e-12, "axis {d}: {p:?}");
                assert!(e.contains(p));
                // A hair further out is outside, a hair in is inside.
                let mut out = p;
                out[d] += sign * 1e-6;
                assert!(e.signed_distance(out) > 0.0 && !e.contains(out));
                let mut inside = p;
                inside[d] -= sign * 1e-6;
                assert!(e.signed_distance(inside) < 0.0 && e.contains(inside));
            }
        }
        // An oblique surface point: (x/6)^2 + (y/3)^2 + (z/2)^2 = 1 with each
        // term a third, so every offset is its radius over sqrt(3).
        let third = 1.0 / 3.0_f64.sqrt();
        let oblique = [
            center[0] + radii[0] * third,
            center[1] - radii[1] * third,
            center[2] + radii[2] * third,
        ];
        assert!(e.signed_distance(oblique).abs() < 1e-12, "{oblique:?}");
        // At the centre the nearest surface is the smallest radius away, and
        // that much of the field is exact too.
        assert!((e.signed_distance(center) + 2.0).abs() < 1e-12);
        // The magnitude is a bound: never further than the true distance, which
        // straight out along an axis is known exactly.
        let far = [center[0] + radii[0] + 5.0, center[1], center[2]];
        let bound = e.signed_distance(far);
        assert!(bound > 0.0 && bound <= 5.0 + 1e-12, "{bound} over-states 5");
        assert!((e.min_extent() - 2.0).abs() < 1e-12);
    }

    /// Three equal radii are a sphere, and the field says so: this is what makes
    /// the ellipsoid's approximation harmless where it degenerates.
    #[test]
    fn an_ellipsoid_with_equal_radii_is_the_sphere_it_looks_like() {
        let center = [2.0, 5.0, -1.0];
        let sphere = Shape::Sphere {
            center,
            radius: 4.0,
        };
        let e = Shape::Ellipsoid {
            center,
            radii: [4.0; 3],
            rotation_deg: [0.0; 3],
        };
        for p in [
            [0.0, 0.0, 0.0],
            [2.0, 5.0, -1.0],
            [12.0, 5.0, -1.0],
            [2.0, 5.0, 3.0],
            [-7.0, 13.0, 6.0],
            [2.5, 5.5, -0.5],
        ] {
            let (a, b) = (sphere.signed_distance(p), e.signed_distance(p));
            assert!((a - b).abs() < 1e-12, "at {p:?}: {a} against {b}");
        }
        assert_eq!(sphere.bounds(), e.bounds());
    }

    /// A turned ellipsoid measures from the surface as it is turned: its own
    /// surface points, carried through the rotation, are still on it.
    #[test]
    fn a_rotated_ellipsoid_keeps_its_surface_under_the_turn() {
        let radii = [6.0, 3.0, 2.0];
        let center = [1.0, -2.0, 4.0];
        for rotation in [
            [0.0, 0.0, 45.0],
            [90.0, 0.0, 0.0],
            [15.0, -35.0, 62.0],
            [180.0, 180.0, 180.0],
        ] {
            let e = Shape::Ellipsoid {
                center,
                radii,
                rotation_deg: rotation,
            };
            let matrix = rotation_matrix(rotation);
            // The centre is the centre whatever the rotation, and the field
            // there is the smallest radius deep.
            assert!(
                (e.signed_distance(center) + 2.0).abs() < 1e-12,
                "{rotation:?}"
            );
            // Every point of the unrotated surface, turned about the centre, is
            // on the turned surface.
            for (a, b) in [(0.3_f64, 0.7_f64), (1.1, 2.4), (0.0, 0.0), (2.0, 5.5)] {
                let (sin_polar, cos_polar) = a.sin_cos();
                let (sin_az, cos_az) = b.sin_cos();
                let local = [
                    radii[0] * sin_polar * cos_az,
                    radii[1] * sin_polar * sin_az,
                    radii[2] * cos_polar,
                ];
                let p = sum(center, rotate(&matrix, local));
                assert!(
                    e.signed_distance(p).abs() < 1e-12,
                    "{rotation:?} at {p:?}: {}",
                    e.signed_distance(p)
                );
                // Scaled in and out about the centre, the sign follows.
                let inner = sum(center, rotate(&matrix, scale(local, 0.9)));
                let outer = sum(center, rotate(&matrix, scale(local, 1.1)));
                assert!(e.signed_distance(inner) < 0.0 && e.signed_distance(outer) > 0.0);
            }
            // A turn moves no material: the smallest extent is what it was.
            assert!((e.min_extent() - 2.0).abs() < 1e-12);
        }
    }

    /// The bounds are exact both ways: nothing on the surface reaches past
    /// them, and something on the surface touches each of the six faces.
    #[test]
    fn a_rotated_ellipsoid_is_bounded_exactly_by_its_support_function() {
        let radii = [6.0, 3.0, 2.0];
        let center = [1.0, -2.0, 4.0];
        // Hand computed: an ellipsoid of semi-axes 3 and 1 turned 45 degrees
        // about z reaches sqrt(3^2 cos^2 45 + 1^2 sin^2 45) = sqrt(5) on x.
        let flat = Shape::Ellipsoid {
            center: [0.0; 3],
            radii: [3.0, 1.0, 1.0],
            rotation_deg: [0.0, 0.0, 45.0],
        };
        let bb = flat.bounds();
        assert!((bb.max[0] - 5.0_f64.sqrt()).abs() < 1e-12, "{bb:?}");
        assert!((bb.max[1] - 5.0_f64.sqrt()).abs() < 1e-12, "{bb:?}");
        assert!(
            (bb.max[2] - 1.0).abs() < 1e-12,
            "z is untouched by a z turn"
        );

        for rotation in [
            [0.0; 3],
            [0.0, 0.0, 45.0],
            [90.0, 0.0, 0.0],
            [15.0, -35.0, 62.0],
        ] {
            let e = Shape::Ellipsoid {
                center,
                radii,
                rotation_deg: rotation,
            };
            let bb = e.bounds();
            let matrix = rotation_matrix(rotation);
            // Brute force: a fine sweep of the surface, which has no corners to
            // shortcut through. Nothing on it may leave the box.
            let (polar_steps, azimuth_steps) = (180, 360);
            let mut reach = [0.0_f64; 3];
            for i in 0..=polar_steps {
                let polar = std::f64::consts::PI * i as f64 / polar_steps as f64;
                let (sin_polar, cos_polar) = polar.sin_cos();
                for j in 0..azimuth_steps {
                    let azimuth = 2.0 * std::f64::consts::PI * j as f64 / azimuth_steps as f64;
                    let (sin_az, cos_az) = azimuth.sin_cos();
                    let point = sum(
                        center,
                        rotate(
                            &matrix,
                            [
                                radii[0] * sin_polar * cos_az,
                                radii[1] * sin_polar * sin_az,
                                radii[2] * cos_polar,
                            ],
                        ),
                    );
                    for d in 0..3 {
                        assert!(
                            point[d] >= bb.min[d] - 1e-9 && point[d] <= bb.max[d] + 1e-9,
                            "{rotation:?}: {point:?} leaves {bb:?} on axis {d}"
                        );
                        reach[d] = reach[d].max((point[d] - center[d]).abs());
                    }
                }
            }
            // And the box is no larger than it has to be: the sampled reach
            // comes up to it, so no face has slack a tighter box could take.
            for d in 0..3 {
                let half = bb.max[d] - center[d];
                assert!(
                    (half - reach[d]) / half < 1e-4,
                    "{rotation:?}: axis {d} half extent {half} against a reach of {}",
                    reach[d]
                );
                assert!((center[d] - bb.min[d] - half).abs() < 1e-12, "{bb:?}");
            }
            // The face really is touched: the support direction of axis d is
            // the d-th row of R * diag(r), and the point it picks is on the
            // surface and exactly on the face.
            let half = Shape::ellipsoid_half_extents(radii, rotation);
            for d in 0..3 {
                let row: Vec3 = std::array::from_fn(|j| matrix[d][j] * radii[j]);
                let unit = scale(row, 1.0 / length(row));
                let local: Vec3 = std::array::from_fn(|j| radii[j] * unit[j]);
                let touch = sum(center, rotate(&matrix, local));
                assert!(e.signed_distance(touch).abs() < 1e-9, "{rotation:?} {d}");
                assert!(
                    (touch[d] - (center[d] + half[d])).abs() < 1e-9,
                    "{rotation:?}: axis {d} touches at {} rather than {}",
                    touch[d],
                    center[d] + half[d]
                );
            }
        }
    }

    /// An unrotated ellipsoid is the centre +- radii box, exactly, and carrying
    /// a rotation of zero changes nothing measurable.
    #[test]
    fn an_unrotated_ellipsoid_is_bounded_by_its_own_radii() {
        let (center, radii) = ([1.5, -2.25, 0.75], [6.0, 3.0, 2.0]);
        let plain = Shape::Ellipsoid {
            center,
            radii,
            rotation_deg: [0.0; 3],
        };
        assert_eq!(
            plain.bounds(),
            Aabb {
                min: [
                    center[0] - radii[0],
                    center[1] - radii[1],
                    center[2] - radii[2]
                ],
                max: [
                    center[0] + radii[0],
                    center[1] + radii[1],
                    center[2] + radii[2]
                ],
            }
        );
        assert_eq!(
            Shape::ellipsoid_half_extents(radii, [0.0, -0.0, 0.0]),
            radii
        );
        let explicit = Shape::Ellipsoid {
            center,
            radii,
            rotation_deg: [360.0, 0.0, 0.0],
        };
        for p in [center, [0.0; 3], [8.0, 1.0, 1.0], [1.5, 0.75, 0.75]] {
            assert!((plain.signed_distance(p) - explicit.signed_distance(p)).abs() < 1e-12);
        }
    }

    /// A tube with no bend is the capsule between its two ends: round caps,
    /// which is exactly what tells it apart from the cylinder it would
    /// otherwise be.
    #[test]
    fn a_straight_tube_is_the_capsule_its_two_ends_describe() {
        let t = Shape::Tube {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            bend: None,
            radius: 3.0,
        };
        // On the axis: a radius deep, all the way along it.
        for z in [0.0, 2.5, 5.0, 10.0] {
            assert!(
                (t.signed_distance([0.0, 0.0, z]) + 3.0).abs() < 1e-12,
                "{z}"
            );
        }
        // Radially outside, and radially inside.
        assert!((t.signed_distance([5.0, 0.0, 5.0]) - 2.0).abs() < 1e-12);
        assert!((t.signed_distance([3.0, 0.0, 5.0])).abs() < 1e-12);
        // Beyond an end the cap is a hemisphere, so the distance is measured
        // from the end *point*: 4 mm past it is 4 - 3 = 1 mm outside, where the
        // cylinder's flat cap would have put the surface at the cap plane.
        assert!((t.signed_distance([0.0, 0.0, 14.0]) - 1.0).abs() < 1e-12);
        let flat = Shape::Cylinder {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            radius: 3.0,
        };
        assert!((flat.signed_distance([0.0, 0.0, 14.0]) - 4.0).abs() < 1e-9);
        assert!(t.contains([0.0, 0.0, 12.9]) && !flat.contains([0.0, 0.0, 12.9]));
        // Off the end and off the axis, which is the corner the cylinder gets
        // wrong: 3-4-5 from the end point.
        assert!((t.signed_distance([4.0, 0.0, 13.0]) - 2.0).abs() < 1e-12);
        // The bounds are the segment grown by the radius on every side.
        assert_eq!(
            t.bounds(),
            Aabb {
                min: [-3.0, -3.0, -3.0],
                max: [3.0, 3.0, 13.0],
            }
        );
        assert!((t.min_extent() - 3.0).abs() < 1e-12);
    }

    /// A bend that lies on the line through the two ends is no bend at all, and
    /// the tube is then the straight one down to the last bit: the collinear
    /// case takes the very same arithmetic rather than a circle of enormous
    /// radius.
    #[test]
    fn a_tube_bent_along_its_own_axis_is_the_straight_one_bit_for_bit() {
        let (p1, p2, radius) = ([1.0, -2.0, 0.5], [7.0, 4.0, 6.5], 2.0);
        let straight = Shape::Tube {
            p1,
            p2,
            bend: None,
            radius,
        };
        // On the segment, past either end, and a hair off the line - all of
        // them inside the collinearity epsilon.
        let (u, _) = perpendicular_basis(sub(p2, p1));
        for along in [0.5, 0.0, 1.0, -3.0, 4.0] {
            for sideways in [0.0, 0.5 * constants::TUBE_COLLINEAR_EPS_MM] {
                let bend = sum(sum(p1, scale(sub(p2, p1), along)), scale(u, sideways));
                let collinear = Shape::Tube {
                    p1,
                    p2,
                    bend: Some(bend),
                    radius,
                };
                assert!(
                    tube_arc(p1, p2, Some(bend)).is_none(),
                    "{along} {sideways}: a bend this shallow is not an arc"
                );
                for p in [
                    [0.0, 0.0, 0.0],
                    [4.0, 1.0, 3.5],
                    [1.0, -2.0, 0.5],
                    [-5.0, 6.0, 2.0],
                    [7.0, 4.0, 9.0],
                ] {
                    let (a, b) = (straight.signed_distance(p), collinear.signed_distance(p));
                    assert_eq!(a.to_bits(), b.to_bits(), "at {p:?}: {a} against {b}");
                }
                assert_eq!(straight.bounds(), collinear.bounds());
            }
        }
    }

    /// The circle a bend resolves to is the circumcircle of the three points,
    /// and the arc of it the tube follows is the one that goes *through* the
    /// bend.
    #[test]
    fn the_arc_of_a_bent_tube_is_the_circumcircle_through_its_three_points() {
        // Hand computed: the three points of the unit circle in the xy plane.
        let arc =
            tube_arc([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], Some([0.0, 1.0, 0.0])).expect("an arc");
        assert!(length(arc.center) < 1e-12, "{:?}", arc.center);
        assert!((arc.radius - 1.0).abs() < 1e-12);
        assert!((arc.span - std::f64::consts::PI).abs() < 1e-12);
        assert!(
            length(difference(
                arc.point(0.5 * std::f64::consts::PI),
                [0.0, 1.0, 0.0]
            )) < 1e-12
        );

        for (p1, bend, p2) in [
            ([0.0, 0.0, 0.0], [5.0, 3.0, 0.0], [10.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [5.0, 3.0, 2.0], [10.0, 1.0, -4.0]),
            // A bend behind the first end: the arc is the long way round.
            ([0.0, 0.0, 0.0], [-4.0, 2.0, 0.0], [10.0, 0.0, 0.0]),
            ([2.0, -3.0, 4.0], [3.0, -1.0, 9.0], [-6.0, 2.0, 5.0]),
        ] {
            let arc = tube_arc(p1, p2, Some(bend)).expect("an arc");
            // Every one of the three is on the circle, and the ends are where
            // the parameterization says.
            for p in [p1, bend, p2] {
                assert!(
                    (length(difference(p, arc.center)) - arc.radius).abs() < 1e-9,
                    "{p:?} is not on the circle"
                );
                assert!(
                    inner(difference(p, arc.center), arc.normal).abs() < 1e-9,
                    "{p:?} is not in its plane"
                );
                assert!(arc.distance(p).abs() < 1e-9, "{p:?} is not on the arc");
            }
            assert!(length(difference(arc.first(), p1)) < 1e-9);
            assert!(length(difference(arc.last(), p2)) < 1e-9);
            assert!(arc.span > 0.0 && arc.span < TAU);
            // The frame is orthonormal and right handed.
            assert!((length(arc.start) - 1.0).abs() < 1e-12);
            assert!((length(arc.turn) - 1.0).abs() < 1e-12);
            assert!(inner(arc.start, arc.turn).abs() < 1e-12);
            assert!(length(difference(cross(arc.start, arc.turn), arc.normal)) < 1e-9);
        }
    }

    /// The field of a bent tube is the true distance to its arc, everywhere -
    /// which is what a sphere trace of it relies on. Checked against a dense
    /// sampling of the arc, which shares none of the closed form's arithmetic.
    #[test]
    fn a_bent_tube_carries_the_true_distance_to_its_own_arc() {
        let (p1, bend, p2) = ([0.0, 0.0, 0.0], [10.0, 6.0, 2.0], [20.0, 0.0, -3.0]);
        let radius = 2.5;
        let shape = Shape::Tube {
            p1,
            p2,
            bend: Some(bend),
            radius,
        };
        let arc = tube_arc(p1, p2, Some(bend)).expect("an arc");
        // The reference: the nearest of ten thousand points along the arc. Its
        // error is at most half the spacing, which is well under the tolerance.
        let samples = 10_000;
        let nearest = |q: Vec3| -> f64 {
            (0..=samples)
                .map(|i| {
                    let angle = arc.span * i as f64 / samples as f64;
                    length(difference(q, arc.point(angle)))
                })
                .fold(f64::INFINITY, f64::min)
        };
        let mut checked = 0;
        for i in 0..11 {
            for j in 0..7 {
                for k in 0..5 {
                    let q = [
                        -8.0 + 3.7 * i as f64,
                        -6.0 + 3.1 * j as f64,
                        -9.0 + 4.3 * k as f64,
                    ];
                    let exact = shape.signed_distance(q) + radius;
                    let sampled = nearest(q);
                    assert!(
                        exact <= sampled + 1e-9,
                        "at {q:?}: {exact} over-states the sampled {sampled}"
                    );
                    assert!(
                        (exact - sampled).abs() < 1e-2,
                        "at {q:?}: {exact} against a sampled {sampled}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 385);
        // And the zero level set is exactly the surface: a radius out from any
        // point of the arc, in any direction at a right angle to it.
        for i in 0..=8 {
            let angle = arc.span * i as f64 / 8.0;
            let on = arc.point(angle);
            let tangent = difference(arc.point(angle + 1e-6), on);
            let (u, v) = perpendicular_basis(tangent);
            for (a, b) in [(1.0, 0.0), (0.0, 1.0), (-0.6, 0.8), (0.6, -0.8)] {
                let out = sum(on, scale(sum(scale(u, a), scale(v, b)), radius));
                assert!(
                    shape.signed_distance(out).abs() < 1e-6,
                    "{out:?} is off the surface by {}",
                    shape.signed_distance(out)
                );
                let inside = sum(on, scale(sum(scale(u, a), scale(v, b)), 0.9 * radius));
                let outside = sum(on, scale(sum(scale(u, a), scale(v, b)), 1.1 * radius));
                assert!(shape.signed_distance(inside) < 0.0);
                assert!(shape.signed_distance(outside) > 0.0);
            }
        }
        // Off the end of the arc - along the tangent there, away from the
        // sweep - the cap is a hemisphere about that end, as the straight
        // tube's is: 4 mm out is 4 mm from the curve.
        let past = sum(p1, scale(arc.turn, -4.0));
        assert!(
            (shape.signed_distance(past) - (4.0 - radius)).abs() < 1e-9,
            "{past:?}: {}",
            shape.signed_distance(past)
        );
        assert!((shape.min_extent() - radius).abs() < 1e-12);
    }

    /// The bounds of a bent tube are exact both ways: nothing on its surface
    /// reaches past them, and something on it touches each of the six faces.
    #[test]
    fn a_bent_tube_is_bounded_exactly_by_its_own_arc() {
        let radius = 2.0;
        for (p1, bend, p2) in [
            ([0.0, 0.0, 0.0], [10.0, 6.0, 0.0], [20.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [10.0, 6.0, 3.0], [20.0, -2.0, -4.0]),
            // Most of a circle: the arc reaches past both of its ends on every
            // axis, which is what the stationary angles are there for.
            ([0.0, 0.0, 0.0], [-6.0, 8.0, 0.0], [2.0, 1.0, 0.0]),
            ([1.0, -2.0, 3.0], [4.0, 8.0, -1.0], [-5.0, 3.0, 6.0]),
        ] {
            let shape = Shape::Tube {
                p1,
                p2,
                bend: Some(bend),
                radius,
            };
            let arc = tube_arc(p1, p2, Some(bend)).expect("an arc");
            let bb = shape.bounds();
            // Brute force: rings of points a radius out from a fine sampling of
            // the arc, and whole spheres about the two ends, which between them
            // cover the surface.
            let (along_steps, round_steps) = (2000, 360);
            let mut reach = [f64::NEG_INFINITY; 3];
            let mut low = [f64::INFINITY; 3];
            let mut sample = |point: Vec3| {
                for d in 0..3 {
                    assert!(
                        point[d] >= bb.min[d] - 1e-9 && point[d] <= bb.max[d] + 1e-9,
                        "{point:?} leaves {bb:?} on axis {d}"
                    );
                    reach[d] = reach[d].max(point[d]);
                    low[d] = low[d].min(point[d]);
                }
            };
            for i in 0..=along_steps {
                let angle = arc.span * i as f64 / along_steps as f64;
                let on = arc.point(angle);
                let tangent = difference(arc.point(angle + 1e-6), on);
                let (u, v) = perpendicular_basis(tangent);
                for j in 0..round_steps {
                    let round = TAU * j as f64 / round_steps as f64;
                    let (sin, cos) = round.sin_cos();
                    sample(sum(on, scale(sum(scale(u, cos), scale(v, sin)), radius)));
                }
            }
            for end in [p1, p2] {
                for i in 0..=round_steps / 2 {
                    let polar = std::f64::consts::PI * i as f64 / (round_steps / 2) as f64;
                    let (sin_polar, cos_polar) = polar.sin_cos();
                    for j in 0..round_steps {
                        let azimuth = TAU * j as f64 / round_steps as f64;
                        let (sin_az, cos_az) = azimuth.sin_cos();
                        sample(sum(
                            end,
                            scale([sin_polar * cos_az, sin_polar * sin_az, cos_polar], radius),
                        ));
                    }
                }
            }
            // No face has slack a tighter box could take.
            for d in 0..3 {
                let span = bb.max[d] - bb.min[d];
                assert!(
                    (bb.max[d] - reach[d]) / span < 1e-3 && (low[d] - bb.min[d]) / span < 1e-3,
                    "axis {d}: {bb:?} against a reach of {} .. {}",
                    low[d],
                    reach[d]
                );
            }
        }
    }

    /// The promise the whole extension rests on: a configuration that carries
    /// no rotation is arithmetically the configuration that existed before it
    /// could, down to the last bit. Every recorded trajectory and every
    /// determinism hash depends on this.
    #[test]
    fn an_unrotated_box_is_bit_for_bit_the_axis_aligned_box() {
        let (min, max) = ([-3.25, 1.5, -0.75], [7.125, 11.0, 4.5]);
        let plain = Shape::axis_aligned_box(min, max);
        let explicit = Shape::Box {
            min,
            max,
            rotation_deg: [0.0, -0.0, 0.0],
        };
        let mut samples = Vec::new();
        for i in 0..13 {
            for j in 0..7 {
                for k in 0..5 {
                    samples.push([
                        -8.0 + 1.37 * i as f64,
                        -4.0 + 2.9 * j as f64,
                        -6.0 + 3.1 * k as f64,
                    ]);
                }
            }
        }
        for p in samples {
            let a = plain.signed_distance(p);
            let b = explicit.signed_distance(p);
            assert_eq!(a.to_bits(), b.to_bits(), "at {p:?}: {a} against {b}");
        }
        assert_eq!(plain.bounds(), explicit.bounds());
    }

    /// A lattice of sample points over a shape's bounds, grown by `margin` so
    /// that the sweep covers the outside of it as well as the inside.
    fn lattice(shape: &Shape, margin: f64, steps: usize) -> Vec<Vec3> {
        let bb = shape.bounds().expanded(margin);
        let mut out = Vec::with_capacity((steps + 1).pow(3));
        for i in 0..=steps {
            for j in 0..=steps {
                for k in 0..=steps {
                    out.push([
                        bb.min[0] + (bb.max[0] - bb.min[0]) * i as f64 / steps as f64,
                        bb.min[1] + (bb.max[1] - bb.min[1]) * j as f64 / steps as f64,
                        bb.min[2] + (bb.max[2] - bb.min[2]) * k as f64 / steps as f64,
                    ]);
                }
            }
        }
        out
    }

    /// The claim behind every exact field here, asked of the field itself: the
    /// projection lands **on** the surface, and it lands exactly as far away as
    /// the field said the surface was.
    fn assert_projection_is_exact(shape: &Shape, what: &str, margin: f64, steps: usize) {
        let mut checked = 0;
        for p in lattice(shape, margin, steps) {
            let Some(surface) = shape.nearest_surface_point(p) else {
                continue;
            };
            let residual = shape.signed_distance(surface);
            assert!(
                residual.abs() < 1e-9,
                "{what}: the projection of {p:?} landed {residual} off the surface"
            );
            let travelled = norm(sub(surface, p));
            assert!(
                (travelled - shape.signed_distance(p).abs()).abs() < 1e-9,
                "{what}: {p:?} moved {travelled} for a distance of {}",
                shape.signed_distance(p)
            );
            checked += 1;
        }
        assert!(
            checked > 100,
            "{what}: only {checked} points were projected"
        );
    }

    /// Not one point the field calls inside lies outside the bounds - swept
    /// over a lattice that covers the whole of the box and a margin round it.
    fn assert_bounds_contain_the_solid(shape: &Shape, what: &str, steps: usize) {
        let bb = shape.bounds();
        let mut inside = 0;
        for p in lattice(shape, 0.25 * length(bb.extent()), steps) {
            if shape.signed_distance(p) > 0.0 {
                continue;
            }
            inside += 1;
            for d in 0..3 {
                assert!(
                    p[d] >= bb.min[d] - 1e-9 && p[d] <= bb.max[d] + 1e-9,
                    "{what}: {p:?} is inside the shape and outside {bb:?}"
                );
            }
        }
        assert!(
            inside > 100,
            "{what}: the sweep found only {inside} points inside"
        );
    }

    /// Every face of the box is touched by the surface it is derived from, so
    /// no tighter box contains the shape.
    ///
    /// Handed a sweep of real surface points, which is what a bound can be
    /// checked against without re-deriving it: a lattice would only ever come
    /// as near a slanted or curved extreme as its own spacing.
    fn assert_bounds_are_tight(shape: &Shape, what: &str, surface: &[Vec3]) {
        let bb = shape.bounds();
        for p in surface {
            assert!(
                shape.signed_distance(*p).abs() < 1e-9,
                "{what}: the sweep point {p:?} is not on the surface"
            );
            for d in 0..3 {
                assert!(
                    p[d] >= bb.min[d] - 1e-9 && p[d] <= bb.max[d] + 1e-9,
                    "{what}: the surface point {p:?} is outside {bb:?}"
                );
            }
        }
        for d in 0..3 {
            let span = bb.max[d] - bb.min[d];
            let reach = surface.iter().fold(f64::NEG_INFINITY, |m, p| m.max(p[d]));
            let low = surface.iter().fold(f64::INFINITY, |m, p| m.min(p[d]));
            assert!(
                (bb.max[d] - reach) / span < 1e-3 && (low - bb.min[d]) / span < 1e-3,
                "{what} axis {d}: {bb:?} against a reach of {low} .. {reach}"
            );
        }
    }

    /// The two cap circles of a cone, densely: the points its bounding box is
    /// the hull of, whichever way it is pointed.
    fn cone_rim(shape: &Shape) -> Vec<Vec3> {
        let Shape::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } = *shape
        else {
            panic!("this sweep is about cones");
        };
        let (u, v) = perpendicular_basis(sub(p2, p1));
        let mut out = Vec::new();
        for (centre, radius) in [(p1, radius1), (p2, radius2)] {
            for i in 0..720 {
                let angle = TAU * i as f64 / 720.0;
                let (sin, cos) = angle.sin_cos();
                out.push(sum(
                    centre,
                    sum(scale(u, radius * cos), scale(v, radius * sin)),
                ));
            }
        }
        out
    }

    /// The other half of the exactness claim, which the projection alone
    /// cannot make: the field never says the surface is **further** than it is.
    ///
    /// The projection lands on the surface at exactly the distance the field
    /// reported, which proves the field never understates. This walks a sweep
    /// of the surface itself and asks whether any point of it is nearer than
    /// the field allowed - within the sweep's own spacing, which is what a
    /// finite sample of a curved surface can say.
    fn assert_the_field_never_overstates(
        shape: &Shape,
        what: &str,
        surface: &[Vec3],
        spacing: f64,
    ) {
        for p in lattice(shape, 2.0, 6) {
            let nearest = surface
                .iter()
                .fold(f64::INFINITY, |best, s| best.min(norm(sub(*s, p))));
            let field = shape.signed_distance(p).abs();
            assert!(
                field <= nearest + spacing,
                "{what}: the field puts the surface {field} from {p:?}, and a point of it is at \
                 {nearest}"
            );
        }
    }

    /// A sweep of a cone's whole surface: the wall, and both cap discs.
    fn cone_surface(shape: &Shape) -> Vec<Vec3> {
        let Shape::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } = *shape
        else {
            panic!("this sweep is about cones");
        };
        let (u, v) = perpendicular_basis(sub(p2, p1));
        let steps = 60;
        let mut out = Vec::new();
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let centre = sum(p1, scale(sub(p2, p1), t));
            let wall = radius1 + (radius2 - radius1) * t;
            for j in 0..steps {
                let angle = TAU * j as f64 / steps as f64;
                let (sin, cos) = angle.sin_cos();
                let radial = sum(scale(u, cos), scale(v, sin));
                out.push(sum(centre, scale(radial, wall)));
                // The cap discs, swept out to their own rims.
                if i == 0 || i == steps {
                    for k in 1..steps {
                        out.push(sum(centre, scale(radial, wall * k as f64 / steps as f64)));
                    }
                }
            }
        }
        out
    }

    /// A sweep of a triangular prism's whole surface: both faces, and the three
    /// side quads.
    fn prism_surface(shape: &Shape) -> Vec<Vec3> {
        let Shape::Triangle { a, b, c, thickness } = *shape else {
            panic!("this sweep is about prisms");
        };
        let normal = triangle_normal(a, b, c).expect("a normal");
        let steps = 40;
        let mut out = Vec::new();
        for i in 0..=steps {
            for j in 0..=steps - i {
                let (u, v) = (i as f64 / steps as f64, j as f64 / steps as f64);
                let point = sum(a, sum(scale(sub(b, a), u), scale(sub(c, a), v)));
                for side in [-0.5 * thickness, 0.5 * thickness] {
                    out.push(sum(point, scale(normal, side)));
                }
            }
        }
        for (from, to) in [(a, b), (b, c), (c, a)] {
            for i in 0..=steps {
                let along = sum(from, scale(sub(to, from), i as f64 / steps as f64));
                for j in 0..=steps {
                    let side = thickness * (j as f64 / steps as f64 - 0.5);
                    out.push(sum(along, scale(normal, side)));
                }
            }
        }
        out
    }

    /// A cone whose two radii are equal is a cylinder, and says so with the
    /// same numbers at every point of space: the two fields are the same field,
    /// which is what makes the cone a generalization rather than a second
    /// answer to the same question.
    #[test]
    fn a_cone_of_one_radius_is_the_cylinder_it_generalizes() {
        let (p1, p2, radius) = ([1.0, 2.0, 3.0], [1.0, 9.0, 3.0], 2.5);
        let cylinder = Shape::Cylinder { p1, p2, radius };
        let cone = Shape::Cone {
            p1,
            p2,
            radius1: radius,
            radius2: radius,
        };
        for p in lattice(&cylinder, 4.0, 20) {
            let (a, b) = (cylinder.signed_distance(p), cone.signed_distance(p));
            assert!(
                (a - b).abs() < 1e-9,
                "at {p:?}: the cylinder says {a} and the cone {b}"
            );
        }
        assert_eq!(cone.bounds(), cylinder.bounds());
        assert!((cone.min_extent() - cylinder.min_extent()).abs() < 1e-12);
        assert!(!cone.self_intersects());
    }

    /// The zero level set is the surface, exactly: every point of a frustum's
    /// wall, of its two caps and of a true cone's apex reads as zero, and the
    /// projection of anything else lands on that surface at the distance the
    /// field reported.
    #[test]
    fn a_cone_carries_its_exact_distance_from_its_wall_to_its_apex() {
        let (p1, p2) = ([0.0, 0.0, 0.0], [0.0, 0.0, 10.0]);
        for (what, radius1, radius2) in [
            ("frustum", 4.0, 1.0),
            ("true cone", 4.0, 0.0),
            ("widening", 1.0, 4.0),
        ] {
            let cone = Shape::Cone {
                p1,
                p2,
                radius1,
                radius2,
            };
            // The wall, walked round and along: every one of those points is
            // on the surface the field describes.
            for i in 0..=10 {
                let t = i as f64 / 10.0;
                let radius = radius1 + (radius2 - radius1) * t;
                for j in 0..8 {
                    let angle = TAU * j as f64 / 8.0;
                    let (sin, cos) = angle.sin_cos();
                    let p = [radius * cos, radius * sin, 10.0 * t];
                    assert!(
                        cone.signed_distance(p).abs() < 1e-12,
                        "{what}: the wall point {p:?} is {} off the surface",
                        cone.signed_distance(p)
                    );
                }
            }
            // The caps, and the apex where the second one has no radius.
            assert!(cone.signed_distance([0.5 * radius1, 0.0, 0.0]).abs() < 1e-12);
            assert!(cone.signed_distance([0.0, 0.0, 10.0]).abs() < 1e-12);
            // Inside, and the depth is the true one: a point on the axis is
            // its own radius from the wall along the perpendicular, and no
            // further than the nearer cap.
            let middle = [0.0, 0.0, 5.0];
            assert!(cone.signed_distance(middle) < 0.0);
            assert_projection_is_exact(&cone, what, 3.0, 16);
            assert_bounds_contain_the_solid(&cone, what, 24);
            assert_bounds_are_tight(&cone, what, &cone_rim(&cone));
            // The box is the hull of the two discs: the wider radius across,
            // and the two cap planes along.
            let widest = radius1.max(radius2);
            assert_eq!(cone.bounds().min, [-widest, -widest, 0.0]);
            assert_eq!(cone.bounds().max, [widest, widest, 10.0]);
            assert!(!cone.self_intersects());
        }

        // Pointed off the axes, where the caps are ellipses to the world and
        // the support function is what bounds them.
        let tilted = Shape::Cone {
            p1: [1.0, 2.0, 3.0],
            p2: [7.0, 6.0, 9.0],
            radius1: 3.0,
            radius2: 1.0,
        };
        assert_projection_is_exact(&tilted, "tilted cone", 2.0, 16);
        assert_bounds_contain_the_solid(&tilted, "tilted cone", 24);
        assert_bounds_are_tight(&tilted, "tilted cone", &cone_rim(&tilted));
        // And the distance is the distance to the surface itself, measured
        // against a sweep of that surface rather than against the formula: the
        // spacing of the sweep is 3 mm of wall over 60 steps.
        assert_the_field_never_overstates(&tilted, "tilted cone", &cone_surface(&tilted), 0.2);

        // A true cone's apex is a legal shape rather than a degenerate one, and
        // `min_extent` is what says so: it measures the wide end and the
        // length, never the point.
        let apex = Shape::Cone {
            p1,
            p2,
            radius1: 4.0,
            radius2: 0.0,
        };
        assert!((apex.min_extent() - 4.0).abs() < 1e-12);
        assert!(apex.contains([0.0, 0.0, 9.999]));
        assert!(!apex.contains([0.5, 0.0, 9.9]));
        // And it is the cone it looks like: half way up, half as wide.
        assert!(apex.contains([1.99, 0.0, 5.0]) && !apex.contains([2.01, 0.0, 5.0]));
    }

    /// Off the axis of revolution there is one nearest surface point and it is
    /// named; on the axis there is a whole circle of them and it is not - the
    /// cylinder's own rule, with the apex and the cap centres as the points
    /// that stay answerable because they lie on the axis themselves.
    #[test]
    fn a_cone_answers_on_its_axis_only_where_one_point_is_nearest() {
        let cone = Shape::Cone {
            p1: [0.0, 0.0, 0.0],
            p2: [0.0, 0.0, 10.0],
            radius1: 4.0,
            radius2: 0.0,
        };
        // Just inside the base, where the cap is what is nearest: a point.
        let base = cone
            .nearest_surface_point([0.0, 0.0, 0.1])
            .expect("a point");
        assert!(norm(sub(base, [0.0, 0.0, 0.0])) < 1e-12);
        // Beyond the apex, which is one point too - and the only surface point
        // anywhere near it.
        let tip = cone
            .nearest_surface_point([0.0, 0.0, 11.0])
            .expect("a point");
        assert!(norm(sub(tip, [0.0, 0.0, 10.0])) < 1e-9, "{tip:?}");
        assert!((cone.signed_distance([0.0, 0.0, 11.0]) - 1.0).abs() < 1e-12);
        // In the middle, where the wall is nearest and it is a whole circle -
        // and just under the apex, where it still is: the wall closes on the
        // axis there faster than the point does.
        assert_eq!(cone.nearest_surface_point([0.0, 0.0, 5.0]), None);
        assert_eq!(cone.nearest_surface_point([0.0, 0.0, 9.9]), None);
        // A cone with no length or no wide end has no surface to project onto.
        assert_eq!(
            Shape::Cone {
                p1: [0.0; 3],
                p2: [0.0; 3],
                radius1: 1.0,
                radius2: 1.0,
            }
            .nearest_surface_point([1.0, 0.0, 0.0]),
            None
        );
    }

    /// The prism's field, from its faces to its edges to its corners: the zero
    /// set is the surface, and the projection lands on it at the distance the
    /// field reported - through every region of the decomposition, which is
    /// what a lattice over the whole bounding box walks through.
    #[test]
    fn a_triangular_prism_carries_its_exact_distance_through_every_region() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [3.0, 8.0, 0.0]);
        let prism = Shape::Triangle {
            a,
            b,
            c,
            thickness: 4.0,
        };
        // The two faces, at half the thickness either side of the plane.
        for point in [a, b, c, [5.0, 2.0, 0.0]] {
            for side in [-2.0, 2.0] {
                let p = [point[0], point[1], side];
                assert!(
                    prism.signed_distance(p).abs() < 1e-12,
                    "{p:?} is {} off the face",
                    prism.signed_distance(p)
                );
            }
        }
        // A point out beyond one face is that far out.
        assert!((prism.signed_distance([5.0, 2.0, 5.0]) - 3.0).abs() < 1e-12);
        // Out beyond an edge, in the plane: the perpendicular distance to it.
        assert!((prism.signed_distance([5.0, -3.0, 0.0]) - 3.0).abs() < 1e-12);
        // And beyond a corner, where both terms bite: the distance to the
        // corner's own edge of the solid.
        let corner = prism.signed_distance([-3.0, -4.0, 4.0]);
        assert!(
            (corner - (9.0f64 + 16.0 + 4.0).sqrt()).abs() < 1e-12,
            "{corner}"
        );
        // Inside, the depth is the nearest surface, which here is the face.
        assert!((prism.signed_distance([5.0, 2.0, 0.0]) + 2.0).abs() < 1e-12);

        assert_projection_is_exact(&prism, "prism", 3.0, 16);
        assert_bounds_contain_the_solid(&prism, "prism", 24);
        assert!(!prism.self_intersects());

        // The bounds are the six corners and nothing more: a prism in the
        // z = 0 plane spans exactly its own triangle across and its own
        // thickness through.
        let bb = prism.bounds();
        assert_eq!(bb.min, [0.0, 0.0, -2.0]);
        assert_eq!(bb.max, [12.0, 8.0, 2.0]);
    }

    /// The six corners of a triangular prism, which are what its bounding box
    /// is the hull of.
    fn prism_corners(shape: &Shape) -> Vec<Vec3> {
        let Shape::Triangle { a, b, c, thickness } = *shape else {
            panic!("this sweep is about prisms");
        };
        let normal = triangle_normal(a, b, c).expect("a normal");
        let mut out = Vec::new();
        for corner in [a, b, c] {
            for side in [-0.5 * thickness, 0.5 * thickness] {
                out.push(sum(corner, scale(normal, side)));
            }
        }
        out
    }

    /// A prism turned out of the axes is the same solid turned: its field is
    /// still exact, and its bounds are still the extremes of its own six
    /// corners rather than a box around a box.
    #[test]
    fn a_triangular_prism_off_the_axes_keeps_its_exact_field_and_its_hull() {
        let prism = Shape::Triangle {
            a: [1.0, 2.0, 3.0],
            b: [7.0, 3.0, 9.0],
            c: [2.0, 9.0, 5.0],
            thickness: 3.0,
        };
        assert_projection_is_exact(&prism, "turned prism", 2.0, 16);
        assert_bounds_contain_the_solid(&prism, "turned prism", 24);
        // Exactly the hull of its own six corners, every one of them on the
        // surface and every face of the box touched by one.
        assert_bounds_are_tight(&prism, "turned prism", &prism_corners(&prism));
        // And the distance is the distance to the surface itself, against a
        // sweep of that surface rather than against the formula.
        assert_the_field_never_overstates(&prism, "turned prism", &prism_surface(&prism), 0.3);
    }

    /// What makes a prism thin is the ball that fits inside its triangle, not
    /// how far apart its corners are: a sliver a metre long is a sliver.
    #[test]
    fn a_prisms_smallest_extent_is_the_ball_that_fits_in_its_triangle() {
        // A right triangle of legs 3 and 4: area 6, perimeter 12, inradius 1.
        let sliver = Shape::Triangle {
            a: [0.0; 3],
            b: [3.0, 0.0, 0.0],
            c: [0.0, 4.0, 0.0],
            thickness: 10.0,
        };
        assert!(
            (sliver.min_extent() - 2.0).abs() < 1e-12,
            "the inscribed circle is 2 mm across"
        );
        // Thinner through than across, and the thickness is the answer.
        let plate = Shape::Triangle {
            a: [0.0; 3],
            b: [3.0, 0.0, 0.0],
            c: [0.0, 4.0, 0.0],
            thickness: 0.5,
        };
        assert!((plate.min_extent() - 0.5).abs() < 1e-12);
        // A long thin triangle is thin however far its corners are apart.
        let long = Shape::Triangle {
            a: [0.0; 3],
            b: [1000.0, 0.0, 0.0],
            c: [500.0, 0.02, 0.0],
            thickness: 10.0,
        };
        assert!(long.min_extent() < 0.05, "{}", long.min_extent());
        assert!(triangle_shortest_altitude([0.0; 3], [1000.0, 0.0, 0.0], [500.0, 0.02, 0.0]) > 0.0);
        // Three points on one line have no altitude at all, at any spread.
        assert_eq!(
            triangle_shortest_altitude([0.0; 3], [10.0, 0.0, 0.0], [4.0, 0.0, 0.0]),
            0.0
        );
        assert_eq!(
            triangle_normal([0.0; 3], [10.0, 0.0, 0.0], [4.0, 0.0, 0.0]),
            None
        );
    }
}
