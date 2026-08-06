//! The increments a drag lands on.
//!
//! One rule, applied to the value a handle is actually changing: the position
//! along the axis being dragged, the dimension being resized, the angle being
//! turned. Nothing else about the shape is touched, so a snapped x drag never
//! moves y onto a multiple of anything.
//!
//! Snapping is *absolute*: the value lands on a multiple of the increment
//! rather than moving by one, which is what makes a dragged object line up with
//! another one that was dragged there earlier. Holding the bypass key
//! (see [`crate::viewer::editor::Editor::set_snap_bypass`]) turns it off for
//! the duration of the drag.

use crate::constants;
use crate::geometry::Aabb;

/// The increments a drag lands on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snap {
    /// Increment lengths land on, in millimetres. Zero, negative or non-finite
    /// means no snapping at all.
    pub millimetres: f64,
    /// Increment angles land on, in degrees, under the same rule.
    pub degrees: f64,
}

impl Default for Snap {
    fn default() -> Self {
        Snap {
            millimetres: constants::VIEW_EDIT_SNAP_MM,
            degrees: constants::VIEW_EDIT_SNAP_DEGREES,
        }
    }
}

impl Snap {
    /// Snapping switched off entirely, which is what the bypass key selects.
    pub const OFF: Snap = Snap {
        millimetres: 0.0,
        degrees: 0.0,
    };

    /// `value` in millimetres, on the nearest increment.
    pub fn length(self, value: f64) -> f64 {
        to_multiple(value, self.millimetres)
    }

    /// `value` in degrees, on the nearest increment.
    pub fn angle(self, value: f64) -> f64 {
        to_multiple(value, self.degrees)
    }

    /// True while lengths are being snapped, which is what the panel reports.
    pub fn snaps_lengths(self) -> bool {
        usable(self.millimetres)
    }
}

/// `value` rounded to the nearest multiple of `step`.
///
/// A step that is not a usable positive number leaves the value exactly as it
/// was - the same value, not a rounded one - so switching snapping off costs
/// no precision at all.
pub fn to_multiple(value: f64, step: f64) -> f64 {
    if !usable(step) || !value.is_finite() {
        return value;
    }
    (value / step).round() * step
}

/// Whether an increment is a number that can be snapped to.
fn usable(step: f64) -> bool {
    step.is_finite() && step >= constants::VIEW_EDIT_SNAP_MIN_MM
}

/// What a candidate plane belongs to, for the callout to name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceKind {
    /// The outside of the design domain.
    Domain,
    /// A face of the `[[keepin]]` entry with this zero based index.
    Keepin(usize),
}

impl SurfaceKind {
    /// What the callout says when a drag has landed on this surface.
    pub fn label(self) -> String {
        match self {
            SurfaceKind::Domain => "flush on the domain".to_string(),
            SurfaceKind::Keepin(index) => format!("flush on keepin {}", index + 1),
        }
    }
}

/// One candidate plane: a coordinate on one axis, and what it belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    /// Where the plane sits on its axis, in millimetres.
    pub coordinate: f64,
    /// What surface it is a face of.
    pub what: SurfaceKind,
}

/// A drag that has landed flush on a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flush {
    /// How much further the shape has to move to sit exactly on the plane.
    pub correction: f64,
    /// The plane it landed on.
    pub plane: Plane,
}

/// The surfaces a dragged region can land flush against: the faces of the
/// keepins, and the outside of the domain.
///
/// Planes rather than shapes, and axis aligned ones: what a support or a load
/// region is being placed against is a *face* - the top of a pad, the wall of
/// the design space - and a face is where its bounding box ends. A rotated box
/// is matched by the bounds it really occupies, so it lands flush by its own
/// silhouette rather than by an edge that happens to point that way; matching
/// the oriented plane of a turned face against another turned face is a
/// different feature and is not this one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Surfaces {
    planes: [Vec<Plane>; 3],
}

impl Surfaces {
    /// Take the faces of `bounds` as candidate planes on every axis.
    pub fn push_bounds(&mut self, bounds: &Aabb, what: SurfaceKind) {
        if bounds.is_empty() {
            return;
        }
        for (axis, planes) in self.planes.iter_mut().enumerate() {
            for coordinate in [bounds.min[axis], bounds.max[axis]] {
                if coordinate.is_finite() {
                    planes.push(Plane { coordinate, what });
                }
            }
        }
    }

    /// True when there is nothing to land on.
    pub fn is_empty(&self) -> bool {
        self.planes.iter().all(|planes| planes.is_empty())
    }

    /// The candidate planes on one axis.
    pub fn on(&self, axis: usize) -> &[Plane] {
        &self.planes[axis]
    }

    /// The correction that puts the nearer face of `bounds` exactly on a
    /// candidate plane of `axis`, when one is within `distance`.
    ///
    /// Both faces are offered to every plane and the smallest correction wins,
    /// so a region dragged towards a pad lands on whichever of its own faces
    /// reaches the pad first - its underside on the top of the pad, or its top
    /// on the underside.
    pub fn flush(&self, axis: usize, bounds: &Aabb, distance: f64) -> Option<Flush> {
        if bounds.is_empty() || !distance.is_finite() || distance <= 0.0 {
            return None;
        }
        let mut best: Option<Flush> = None;
        for plane in self.on(axis) {
            for face in [bounds.min[axis], bounds.max[axis]] {
                let correction = plane.coordinate - face;
                if correction.abs() > distance {
                    continue;
                }
                if best.is_none_or(|current| correction.abs() < current.correction.abs()) {
                    best = Some(Flush {
                        correction,
                        plane: *plane,
                    });
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_lands_on_the_nearest_multiple_of_the_increment() {
        assert!((to_multiple(4.4, 1.0) - 4.0).abs() < 1e-12);
        assert!((to_multiple(4.6, 1.0) - 5.0).abs() < 1e-12);
        assert!((to_multiple(-4.6, 1.0) + 5.0).abs() < 1e-12);
        assert!((to_multiple(4.4, 0.5) - 4.5).abs() < 1e-12);
        assert!((to_multiple(4.4, 2.0) - 4.0).abs() < 1e-12);
        assert!(
            (to_multiple(3.0, 2.0) - 4.0).abs() < 1e-12,
            "halves round out"
        );
        assert!((to_multiple(-0.4, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn an_unusable_increment_leaves_the_value_exactly_alone() {
        for step in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-9] {
            assert_eq!(to_multiple(4.4, step).to_bits(), 4.4_f64.to_bits());
        }
        assert!(to_multiple(f64::NAN, 1.0).is_nan());
        assert!(!Snap::OFF.snaps_lengths());
        assert!(Snap::default().snaps_lengths());
    }

    /// The flush rule: the *nearest* face of the dragged region to the nearest
    /// candidate plane, and nothing at all beyond the snap distance.
    #[test]
    fn a_face_lands_on_the_nearest_plane_within_reach() {
        let mut surfaces = Surfaces::default();
        assert!(surfaces.is_empty());
        surfaces.push_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [100.0, 60.0, 40.0],
            },
            SurfaceKind::Domain,
        );
        surfaces.push_bounds(
            &Aabb {
                min: [40.0, 10.0, 20.0],
                max: [60.0, 50.0, 30.0],
            },
            SurfaceKind::Keepin(1),
        );
        assert!(!surfaces.is_empty());
        assert_eq!(surfaces.on(0).len(), 4, "two objects, two faces each");

        // A region whose upper face is 0.4 mm under the top of the keepin: it
        // is that face, and that plane, that meet.
        let region = Aabb {
            min: [45.0, 20.0, 24.0],
            max: [50.0, 25.0, 29.6],
        };
        let flush = surfaces.flush(2, &region, 2.5).expect("a landing");
        assert!((flush.correction - 0.4).abs() < 1e-12);
        assert_eq!(flush.plane.what, SurfaceKind::Keepin(1));
        assert!((flush.plane.coordinate - 30.0).abs() < 1e-12);
        assert_eq!(flush.plane.what.label(), "flush on keepin 2");
        // Applying the correction really does land it on the plane.
        assert!((region.max[2] + flush.correction - 30.0).abs() < 1e-12);

        // The other way round: a region just above that face reaches down for
        // it with its underside.
        let region = Aabb {
            min: [45.0, 20.0, 30.7],
            max: [50.0, 25.0, 35.0],
        };
        let flush = surfaces.flush(2, &region, 2.5).expect("a landing");
        assert!((flush.correction + 0.7).abs() < 1e-12);

        // Out of reach is out of reach.
        let far = Aabb {
            min: [45.0, 20.0, 34.0],
            max: [50.0, 25.0, 36.0],
        };
        assert!(surfaces.flush(2, &far, 2.5).is_none());
        // And so is a distance of nothing, which is what a bypassed drag has.
        assert!(surfaces.flush(2, &region, 0.0).is_none());
        assert!(Surfaces::default().flush(2, &region, 2.5).is_none());
        assert!(surfaces.flush(2, &Aabb::empty(), 2.5).is_none());

        // The domain's own walls are candidates too, and the nearest plane wins
        // when two are in reach.
        let against_the_wall = Aabb {
            min: [98.4, 20.0, 5.0],
            max: [99.9, 25.0, 10.0],
        };
        let flush = surfaces
            .flush(0, &against_the_wall, 2.5)
            .expect("a landing");
        assert_eq!(flush.plane.what, SurfaceKind::Domain);
        assert!((flush.correction - 0.1).abs() < 1e-12);
        assert_eq!(SurfaceKind::Domain.label(), "flush on the domain");
    }

    #[test]
    fn the_defaults_come_from_the_constants_and_apply_per_unit() {
        let snap = Snap::default();
        assert_eq!(snap.millimetres, constants::VIEW_EDIT_SNAP_MM);
        assert_eq!(snap.degrees, constants::VIEW_EDIT_SNAP_DEGREES);
        assert!((snap.length(3.4) - 3.0).abs() < 1e-12);
        assert!((snap.angle(20.0) - 22.5).abs() < 1e-12);
        assert!((snap.angle(50.0) - 45.0).abs() < 1e-12);
        // Off means off, in both units.
        assert!((Snap::OFF.length(3.4) - 3.4).abs() < 1e-12);
        assert!((Snap::OFF.angle(20.0) - 20.0).abs() < 1e-12);
    }
}
