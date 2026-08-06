//! Orbit camera and the 4x4 matrix algebra it needs.
//!
//! Matrices are column major, `m[column][row]`, which is the layout WGSL
//! expects for a `mat4x4<f32>`, and they are built in f64 so the fit maths can
//! be tested to full precision before being narrowed for the uniform buffer.
//!
//! The world is z-up: growforge configurations put gravity and tip loads along
//! z, so an orbit around the world z axis is the natural one.

use crate::constants;
use crate::geometry::{Aabb, Vec3, cross, difference, inner, length, scale};

/// A column major 4x4 matrix, `m[column][row]`.
pub type Mat4 = [[f64; 4]; 4];

/// The identity matrix.
pub fn identity() -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for (c, column) in m.iter_mut().enumerate() {
        column[c] = 1.0;
    }
    m
}

/// Matrix product `a * b`, applied to a column vector as `a * (b * v)`.
pub fn multiply(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (c, column) in out.iter_mut().enumerate() {
        for (r, value) in column.iter_mut().enumerate() {
            *value = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

/// Transform the point `p` (implicit w = 1) and return the homogeneous result.
pub fn transform_point(m: &Mat4, p: Vec3) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (r, value) in out.iter_mut().enumerate() {
        *value = m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r];
    }
    out
}

/// Narrow a matrix for the uniform buffer.
pub fn to_f32(m: &Mat4) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for (c, column) in out.iter_mut().enumerate() {
        for (r, value) in column.iter_mut().enumerate() {
            *value = m[c][r] as f32;
        }
    }
    out
}

fn normalize(v: Vec3) -> Vec3 {
    let len = length(v);
    if len > 0.0 { scale(v, 1.0 / len) } else { v }
}

/// Right handed view matrix looking from `eye` towards `target`.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let forward = normalize(difference(target, eye));
    let mut side = cross(forward, up);
    if length(side) == 0.0 {
        // The view direction is parallel to `up`; any perpendicular will do.
        side = cross(forward, [1.0, 0.0, 0.0]);
    }
    let side = normalize(side);
    let upward = cross(side, forward);
    [
        [side[0], upward[0], -forward[0], 0.0],
        [side[1], upward[1], -forward[1], 0.0],
        [side[2], upward[2], -forward[2], 0.0],
        [
            -inner(side, eye),
            -inner(upward, eye),
            inner(forward, eye),
            1.0,
        ],
    ]
}

/// Right handed perspective projection onto the `[0, 1]` depth range wgpu uses.
pub fn perspective(fov_y_radians: f64, aspect: f64, near: f64, far: f64) -> Mat4 {
    let focal = 1.0 / (0.5 * fov_y_radians).tan();
    let depth = near - far;
    [
        [focal / aspect, 0.0, 0.0, 0.0],
        [0.0, focal, 0.0, 0.0],
        [0.0, 0.0, far / depth, -1.0],
        [0.0, 0.0, near * far / depth, 0.0],
    ]
}

/// A camera orbiting a target point.
#[derive(Debug, Clone, Copy)]
pub struct OrbitCamera {
    /// Point the camera looks at, in millimetres.
    pub target: Vec3,
    /// Distance from the target, in millimetres.
    pub distance: f64,
    /// Azimuth about world +z, in radians.
    pub yaw: f64,
    /// Elevation above the world xy plane, in radians.
    pub pitch: f64,
    /// Radius of the scene's bounding sphere; sets the clip planes and the
    /// zoom limits.
    pub scene_radius: f64,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        OrbitCamera {
            target: [0.0, 0.0, 0.0],
            distance: 1.0,
            yaw: constants::VIEW_INITIAL_YAW_DEGREES.to_radians(),
            pitch: constants::VIEW_INITIAL_PITCH_DEGREES.to_radians(),
            scene_radius: constants::VIEW_MIN_SCENE_RADIUS_MM,
        }
    }
}

/// Distance at which a bounding sphere of `radius` fills the frustum, leaving
/// [`constants::VIEW_FIT_MARGIN`] of slack.
pub fn fit_distance(radius: f64, aspect: f64) -> f64 {
    let half_fov_y = 0.5 * constants::VIEW_FOV_DEGREES.to_radians();
    let half_fov_x = (half_fov_y.tan() * aspect.max(f64::MIN_POSITIVE)).atan();
    let half_fov = half_fov_y.min(half_fov_x);
    radius * constants::VIEW_FIT_MARGIN / half_fov.sin()
}

impl OrbitCamera {
    /// Frame `bounds` from the default direction.
    pub fn fit(&mut self, bounds: &Aabb, aspect: f64) {
        let (center, radius) = bounding_sphere(bounds);
        self.target = center;
        self.scene_radius = radius;
        self.distance = fit_distance(radius, aspect);
        self.yaw = constants::VIEW_INITIAL_YAW_DEGREES.to_radians();
        self.pitch = constants::VIEW_INITIAL_PITCH_DEGREES.to_radians();
    }

    /// Position of the camera in world coordinates.
    pub fn eye(&self) -> Vec3 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let direction = [cos_pitch * cos_yaw, cos_pitch * sin_yaw, sin_pitch];
        [
            self.target[0] + self.distance * direction[0],
            self.target[1] + self.distance * direction[1],
            self.target[2] + self.distance * direction[2],
        ]
    }

    /// Orthonormal view basis `(forward, side, up)` in world coordinates.
    ///
    /// The same basis [`look_at`] builds from the eye and the target, exposed
    /// because the editor's ray casting has to unproject a pointer position
    /// through exactly the camera the frame was drawn with.
    pub fn view_basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = normalize(difference(self.target, self.eye()));
        let mut side = cross(forward, [0.0, 0.0, 1.0]);
        if length(side) == 0.0 {
            side = cross(forward, [1.0, 0.0, 0.0]);
        }
        let side = normalize(side);
        (forward, side, cross(side, forward))
    }

    /// Near and far clip planes for the current distance.
    pub fn clip_planes(&self) -> (f64, f64) {
        let radius = self.scene_radius.max(constants::VIEW_MIN_SCENE_RADIUS_MM);
        let near =
            (self.distance - radius).max(radius * constants::VIEW_NEAR_PLANE_RADIUS_FRACTION);
        let far = self.distance + radius * constants::VIEW_FAR_PLANE_RADIUS_MARGIN;
        (near, far.max(near * 2.0))
    }

    /// Combined view-projection matrix for a viewport of the given aspect.
    pub fn view_projection(&self, aspect: f64) -> Mat4 {
        let (near, far) = self.clip_planes();
        let view = look_at(self.eye(), self.target, [0.0, 0.0, 1.0]);
        let projection = perspective(
            constants::VIEW_FOV_DEGREES.to_radians(),
            aspect.max(f64::MIN_POSITIVE),
            near,
            far,
        );
        multiply(&projection, &view)
    }

    /// Orbit by a pointer drag of `dx`, `dy` physical pixels.
    pub fn orbit(&mut self, dx: f64, dy: f64) {
        let step = constants::VIEW_ORBIT_DEGREES_PER_PIXEL.to_radians();
        let limit = constants::VIEW_PITCH_LIMIT_DEGREES.to_radians();
        self.yaw -= dx * step;
        self.pitch = (self.pitch + dy * step).clamp(-limit, limit);
    }

    /// Pan by a pointer drag of `dx`, `dy` physical pixels across a viewport
    /// `viewport_height` pixels tall.
    pub fn pan(&mut self, dx: f64, dy: f64, viewport_height: f64) {
        if viewport_height <= 0.0 {
            return;
        }
        // One pixel spans this much world space in the plane through the
        // target, which keeps the grabbed point under the pointer.
        let half_fov = 0.5 * constants::VIEW_FOV_DEGREES.to_radians();
        let per_pixel = 2.0 * self.distance * half_fov.tan() / viewport_height;
        let forward = normalize(difference(self.target, self.eye()));
        let side = normalize(cross(forward, [0.0, 0.0, 1.0]));
        let upward = cross(side, forward);
        for d in 0..3 {
            self.target[d] += (-dx * side[d] + dy * upward[d]) * per_pixel;
        }
    }

    /// Zoom by `lines` of scroll travel; positive zooms in.
    pub fn zoom(&mut self, lines: f64) {
        let radius = self.scene_radius.max(constants::VIEW_MIN_SCENE_RADIUS_MM);
        let factor = (1.0 - constants::VIEW_ZOOM_PER_SCROLL_LINE).powf(lines);
        self.distance = (self.distance * factor).clamp(
            radius * constants::VIEW_MIN_DISTANCE_RADIUS_FRACTION,
            radius * constants::VIEW_MAX_DISTANCE_RADIUS_FACTOR,
        );
    }
}

/// Centre and radius of the sphere circumscribing `bounds`.
pub fn bounding_sphere(bounds: &Aabb) -> (Vec3, f64) {
    if bounds.is_empty() {
        return ([0.0, 0.0, 0.0], constants::VIEW_MIN_SCENE_RADIUS_MM);
    }
    let center = [
        0.5 * (bounds.min[0] + bounds.max[0]),
        0.5 * (bounds.min[1] + bounds.max[1]),
        0.5 * (bounds.min[2] + bounds.max[2]),
    ];
    let radius = 0.5 * length(bounds.extent());
    (center, radius.max(constants::VIEW_MIN_SCENE_RADIUS_MM))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corners(bounds: &Aabb) -> Vec<Vec3> {
        let mut out = Vec::with_capacity(8);
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    out.push([
                        if i == 0 { bounds.min[0] } else { bounds.max[0] },
                        if j == 0 { bounds.min[1] } else { bounds.max[1] },
                        if k == 0 { bounds.min[2] } else { bounds.max[2] },
                    ]);
                }
            }
        }
        out
    }

    fn box_bounds() -> Aabb {
        Aabb {
            min: [10.0, -5.0, 0.0],
            max: [130.0, 35.0, 40.0],
        }
    }

    #[test]
    fn identity_and_multiplication_agree() {
        let m = look_at([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let product = multiply(&identity(), &m);
        for c in 0..4 {
            for r in 0..4 {
                assert!((product[c][r] - m[c][r]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn look_at_places_the_target_on_the_view_axis() {
        let eye = [30.0, -20.0, 15.0];
        let target = [1.0, 2.0, 3.0];
        let view = look_at(eye, target, [0.0, 0.0, 1.0]);
        let p = transform_point(&view, target);
        assert!(p[0].abs() < 1e-9 && p[1].abs() < 1e-9);
        // The target sits straight ahead, so its view space z is minus the
        // distance from the eye.
        let distance = length(difference(target, eye));
        assert!((p[2] + distance).abs() < 1e-9, "view z is {}", p[2]);
        let origin = transform_point(&view, eye);
        assert!(origin[0].abs() < 1e-9 && origin[1].abs() < 1e-9 && origin[2].abs() < 1e-9);
    }

    #[test]
    fn perspective_maps_the_clip_planes_to_zero_and_one() {
        let m = perspective(45f64.to_radians(), 1.5, 0.5, 200.0);
        for (view_z, expected) in [(-0.5, 0.0), (-200.0, 1.0)] {
            let p = transform_point(&m, [0.0, 0.0, view_z]);
            let ndc_z = p[2] / p[3];
            assert!(
                (ndc_z - expected).abs() < 1e-9,
                "view z {view_z} mapped to {ndc_z}, expected {expected}"
            );
        }
    }

    #[test]
    fn a_fitted_camera_frames_the_whole_scene() {
        let bounds = box_bounds();
        for aspect in [0.5, 1.0, 16.0 / 9.0, 3.0] {
            let mut camera = OrbitCamera::default();
            camera.fit(&bounds, aspect);
            let matrix = camera.view_projection(aspect);
            let mut widest: f64 = 0.0;
            for corner in corners(&bounds) {
                let p = transform_point(&matrix, corner);
                assert!(p[3] > 0.0, "corner {corner:?} landed behind the camera");
                let (x, y, z) = (p[0] / p[3], p[1] / p[3], p[2] / p[3]);
                assert!(
                    x.abs() <= 1.0 && y.abs() <= 1.0,
                    "corner {corner:?} projects to ({x}, {y}) at aspect {aspect}"
                );
                assert!(
                    (0.0..=1.0).contains(&z),
                    "corner {corner:?} projects to depth {z}"
                );
                widest = widest.max(x.abs()).max(y.abs());
            }
            assert!(
                widest > 0.4,
                "the fit is far too loose at aspect {aspect}: widest corner is at {widest}"
            );
        }
    }

    #[test]
    fn fit_distance_grows_with_the_scene_and_shrinks_with_a_wider_view() {
        let near = fit_distance(10.0, 1.0);
        let far = fit_distance(20.0, 1.0);
        assert!((far - 2.0 * near).abs() < 1e-9);
        // A wider viewport is limited by the vertical field of view, so the
        // distance stops shrinking once the aspect passes one.
        assert!(fit_distance(10.0, 0.5) > near);
        assert!((fit_distance(10.0, 4.0) - near).abs() < 1e-9);
    }

    #[test]
    fn bounding_sphere_of_a_degenerate_scene_is_usable() {
        let point = Aabb {
            min: [3.0, 3.0, 3.0],
            max: [3.0, 3.0, 3.0],
        };
        let (center, radius) = bounding_sphere(&point);
        assert_eq!(center, [3.0, 3.0, 3.0]);
        assert!(radius >= constants::VIEW_MIN_SCENE_RADIUS_MM);
        let (_, empty_radius) = bounding_sphere(&Aabb::empty());
        assert!(empty_radius.is_finite() && empty_radius > 0.0);
    }

    #[test]
    fn zoom_is_clamped_to_the_scene_scale() {
        let mut camera = OrbitCamera::default();
        camera.fit(&box_bounds(), 1.0);
        let radius = camera.scene_radius;
        for _ in 0..200 {
            camera.zoom(1.0);
        }
        assert!(camera.distance >= radius * constants::VIEW_MIN_DISTANCE_RADIUS_FRACTION - 1e-12);
        for _ in 0..400 {
            camera.zoom(-1.0);
        }
        assert!(camera.distance <= radius * constants::VIEW_MAX_DISTANCE_RADIUS_FACTOR + 1e-12);
    }

    #[test]
    fn pitch_never_reaches_the_pole() {
        let mut camera = OrbitCamera::default();
        camera.fit(&box_bounds(), 1.0);
        let limit = constants::VIEW_PITCH_LIMIT_DEGREES.to_radians();
        for _ in 0..1000 {
            camera.orbit(0.0, 100.0);
        }
        assert!(camera.pitch <= limit + 1e-12);
        for _ in 0..2000 {
            camera.orbit(0.0, -100.0);
        }
        assert!(camera.pitch >= -limit - 1e-12);
    }

    #[test]
    fn panning_moves_the_target_perpendicular_to_the_view() {
        let mut camera = OrbitCamera::default();
        camera.fit(&box_bounds(), 1.0);
        let before = camera.target;
        let eye_before = camera.eye();
        camera.pan(25.0, -10.0, 800.0);
        let moved = difference(camera.target, before);
        let view_direction = normalize(difference(before, eye_before));
        assert!(length(moved) > 0.0);
        assert!(
            inner(moved, view_direction).abs() < 1e-9,
            "panning must not change the distance along the view axis"
        );
    }
}
