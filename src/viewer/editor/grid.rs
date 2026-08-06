//! The ruled plane on the floor of the domain.
//!
//! Snapping is invisible until there is something to see it against. The grid
//! is that: minor lines at the increment the panel is set to, a major line every
//! [`constants::VIEW_EDIT_GRID_MAJOR_EVERY`]-th of them, over the domain's
//! footprint plus a margin, drawn on the plane `z = domain bottom`.
//!
//! Derivation is separated from drawing on purpose. [`FloorGrid::derive`] is
//! pure arithmetic over an axis aligned box and an increment - which is what the
//! line counts, the extents and the cap rule are tested against - and
//! [`FloorGrid::mesh`] is the only part that knows about triangles.
//!
//! The cap is the one rule with judgement in it. A 400 mm domain at a 0.1 mm
//! increment asks for eight thousand lines, which is neither drawable at any
//! useful frame rate nor legible at any zoom. Past
//! [`constants::VIEW_EDIT_GRID_MAX_LINES`] the spacing is multiplied by the
//! smallest *whole* number that brings the count back inside it - whole, so that
//! every line still lands on a multiple of the increment the drags land on - and
//! the panel says the spacing it ended up with rather than quietly ruling
//! something else.

use crate::constants;
use crate::geometry::{Aabb, Vec3};
use crate::mesh::Mesh;
use crate::viewer::scene::{LayerMesh, Shading};
use crate::viewer::tessellate;

/// One ruled line of the grid, in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridLine {
    /// Where the line starts.
    pub from: Vec3,
    /// Where it ends.
    pub to: Vec3,
    /// True for the heavier lines, every
    /// [`constants::VIEW_EDIT_GRID_MAJOR_EVERY`]-th one.
    pub major: bool,
}

/// The floor grid of one domain at one increment.
#[derive(Debug, Clone, PartialEq)]
pub struct FloorGrid {
    /// Spacing the minor lines really ended up at, in millimetres.
    pub spacing_mm: f64,
    /// Spacing that was asked for, which differs from `spacing_mm` only when the
    /// line cap coarsened it.
    pub requested_mm: f64,
    /// Whole number `spacing_mm / requested_mm` the cap rule applied. One when
    /// nothing was coarsened.
    pub coarsening: usize,
    /// Footprint the lines span, at the floor height on z.
    pub bounds: Aabb,
    /// Every line, x-running ones first.
    pub lines: Vec<GridLine>,
}

impl FloorGrid {
    /// The grid for a domain and a snap increment, or `None` when the domain has
    /// no usable footprint to rule.
    ///
    /// `increment_mm` is the panel's snap setting. Snapping switched off leaves
    /// the grid at [`constants::VIEW_EDIT_SNAP_MM`]: the plane is a ruler as
    /// well as a preview of the increment, and a ruler with no divisions is not
    /// one.
    pub fn derive(domain: &Aabb, increment_mm: f64) -> Option<FloorGrid> {
        if domain.is_empty() {
            return None;
        }
        let requested_mm =
            if increment_mm.is_finite() && increment_mm >= constants::VIEW_EDIT_SNAP_MIN_MM {
                increment_mm
            } else {
                constants::VIEW_EDIT_SNAP_MM
            };
        let extent = domain.extent();
        if !extent[0].is_finite() || !extent[1].is_finite() || extent[0] <= 0.0 || extent[1] <= 0.0
        {
            return None;
        }
        let margin = extent[0].max(extent[1]) * constants::VIEW_EDIT_GRID_MARGIN_FRACTION;
        let bounds = Aabb {
            min: [
                domain.min[0] - margin,
                domain.min[1] - margin,
                domain.min[2],
            ],
            max: [
                domain.max[0] + margin,
                domain.max[1] + margin,
                domain.min[2],
            ],
        };
        let span = [bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1]];

        // How many lines the requested spacing wants, and the whole factor that
        // brings that inside the cap.
        let wanted = lines_for(span, requested_mm);
        let cap = constants::VIEW_EDIT_GRID_MAX_LINES.max(1);
        let mut coarsening = 1usize;
        while lines_for(span, requested_mm * coarsening as f64) > cap {
            coarsening += 1;
            // A domain so large, or an increment so fine, that no whole factor
            // helps cannot happen: the count falls off as 1/coarsening, so the
            // loop terminates at coarsening <= wanted. The guard is against a
            // degenerate span that makes the count constant.
            if coarsening > wanted.max(1) {
                return None;
            }
        }
        let spacing_mm = requested_mm * coarsening as f64;

        let mut lines = Vec::new();
        for axis in 0..2 {
            for (index, coordinate) in multiples(bounds.min[axis], bounds.max[axis], spacing_mm) {
                let major = index.rem_euclid(constants::VIEW_EDIT_GRID_MAJOR_EVERY as i64) == 0;
                let (from, to) = if axis == 0 {
                    // A line at this x, running the whole way along y.
                    (
                        [coordinate, bounds.min[1], bounds.min[2]],
                        [coordinate, bounds.max[1], bounds.min[2]],
                    )
                } else {
                    (
                        [bounds.min[0], coordinate, bounds.min[2]],
                        [bounds.max[0], coordinate, bounds.min[2]],
                    )
                };
                lines.push(GridLine { from, to, major });
            }
        }
        Some(FloorGrid {
            spacing_mm,
            requested_mm,
            coarsening,
            bounds,
            lines,
        })
    }

    /// How many lines there are.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// How many of them are the heavier ones.
    pub fn major_count(&self) -> usize {
        self.lines.iter().filter(|line| line.major).count()
    }

    /// What the panel says under the snap control, when the cap coarsened the
    /// spacing. `None` when the grid is ruled at exactly the increment asked
    /// for.
    pub fn note(&self) -> Option<String> {
        (self.coarsening > 1).then(|| {
            format!(
                "floor grid ruled at {} mm, not {} mm: a finer grid would need more than {} lines",
                self.spacing_mm,
                self.requested_mm,
                constants::VIEW_EDIT_GRID_MAX_LINES
            )
        })
    }

    /// The drawable layer: one thin square prism per line, minor and major in
    /// their own colours.
    pub fn mesh(&self) -> Option<LayerMesh> {
        let span = [
            self.bounds.max[0] - self.bounds.min[0],
            self.bounds.max[1] - self.bounds.min[1],
        ];
        let diagonal = (span[0] * span[0] + span[1] * span[1]).sqrt();
        let weights = [
            diagonal * constants::VIEW_EDIT_GRID_MINOR_WEIGHT_FRACTION,
            diagonal * constants::VIEW_EDIT_GRID_MAJOR_WEIGHT_FRACTION,
        ];
        let mut out = LayerMesh::default();
        for major in [false, true] {
            let mut mesh = Mesh::default();
            for line in self.lines.iter().filter(|line| line.major == major) {
                tessellate::append(
                    &mut mesh,
                    &tessellate::cylinder(
                        line.from,
                        line.to,
                        weights[usize::from(major)],
                        constants::VIEW_EDIT_GRID_TUBE_SEGMENTS,
                    ),
                );
            }
            let color = if major {
                constants::VIEW_COLOR_GRID_MAJOR
            } else {
                constants::VIEW_COLOR_GRID_MINOR
            };
            let part = LayerMesh::from_mesh(&mesh, color, Shading::Flat);
            if part.is_empty() {
                continue;
            }
            out.bounds = out.bounds.union(&part.bounds);
            out.vertices.extend(part.vertices);
        }
        (!out.is_empty()).then_some(out)
    }
}

/// Lines a footprint of `span` needs at `spacing`, both axes together.
fn lines_for(span: [f64; 2], spacing: f64) -> usize {
    if spacing <= 0.0 || !spacing.is_finite() {
        return usize::MAX;
    }
    span.iter()
        .map(|extent| (extent / spacing).floor() as usize + 1)
        .fold(0usize, |total, count| total.saturating_add(count))
}

/// The multiples of `step` inside `[low, high]`, each with the whole number it
/// is a multiple of.
///
/// The index is what decides which lines are major, and it counts from the
/// world origin rather than from the edge of the footprint, so the heavy lines
/// sit on round coordinates and stay where they are when the domain is resized.
fn multiples(low: f64, high: f64, step: f64) -> Vec<(i64, f64)> {
    let mut out = Vec::new();
    if step <= 0.0 || !step.is_finite() || !low.is_finite() || !high.is_finite() || high < low {
        return out;
    }
    let first = (low / step).ceil() as i64;
    let last = (high / step).floor() as i64;
    for index in first..=last {
        out.push((index, index as f64 * step));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(x: f64, y: f64) -> Aabb {
        Aabb {
            min: [0.0, 0.0, -5.0],
            max: [x, y, 20.0],
        }
    }

    #[test]
    fn the_lines_span_the_footprint_plus_the_margin_at_the_increment() {
        let grid = FloorGrid::derive(&domain(60.0, 30.0), 1.0).expect("a grid");
        assert_eq!(grid.spacing_mm, 1.0);
        assert_eq!(grid.coarsening, 1);
        assert!(grid.note().is_none());

        // The margin is a tenth of the larger side, on every side of both axes.
        let margin = 6.0;
        assert!((grid.bounds.min[0] + margin).abs() < 1e-9);
        assert!((grid.bounds.max[0] - (60.0 + margin)).abs() < 1e-9);
        assert!((grid.bounds.min[1] + margin).abs() < 1e-9);
        assert!((grid.bounds.max[1] - (30.0 + margin)).abs() < 1e-9);
        // Flat on the floor of the domain, not on its lid.
        assert_eq!(grid.bounds.min[2], -5.0);
        assert_eq!(grid.bounds.max[2], -5.0);
        for line in &grid.lines {
            assert_eq!(line.from[2], -5.0);
            assert_eq!(line.to[2], -5.0);
        }

        // 72 mm of x at 1 mm is the 73 whole millimetres from -6 to 66, and
        // 42 mm of y is 43 of them.
        assert_eq!(grid.line_count(), 73 + 43);
        // Every tenth line counted from the world origin is a major one, so they
        // sit on round coordinates rather than on the edge of the footprint:
        // x = 0, 10 .. 60 is seven of them over -6 .. 66, and y = 0, 10 .. 30 is
        // four over -6 .. 36.
        assert_eq!(grid.major_count(), 7 + 4);
        for line in grid.lines.iter().filter(|line| line.major) {
            let coordinate = if line.from[0] == line.to[0] {
                line.from[0]
            } else {
                line.from[1]
            };
            assert!(
                (coordinate / 10.0 - (coordinate / 10.0).round()).abs() < 1e-9,
                "a major line at {coordinate} is not on a round coordinate"
            );
        }
        // And the lines really do run the other way across the footprint.
        let along_y = grid
            .lines
            .iter()
            .find(|l| l.from[1] != l.to[1])
            .expect("one");
        assert!((along_y.from[1] - grid.bounds.min[1]).abs() < 1e-9);
        assert!((along_y.to[1] - grid.bounds.max[1]).abs() < 1e-9);
    }

    #[test]
    fn the_spacing_follows_the_snap_increment() {
        let footprint = domain(60.0, 30.0);
        let fine = FloorGrid::derive(&footprint, 0.5).expect("a grid");
        let coarse = FloorGrid::derive(&footprint, 5.0).expect("a grid");
        assert_eq!(fine.spacing_mm, 0.5);
        assert_eq!(coarse.spacing_mm, 5.0);
        assert!(
            fine.line_count() > coarse.line_count(),
            "{} against {}",
            fine.line_count(),
            coarse.line_count()
        );
        // Snapping switched off still rules a plane, at the default increment.
        for off in [0.0, -1.0, f64::NAN, 1e-9] {
            let grid = FloorGrid::derive(&footprint, off).expect("a grid");
            assert_eq!(grid.requested_mm, constants::VIEW_EDIT_SNAP_MM);
        }
    }

    #[test]
    fn the_cap_coarsens_the_spacing_by_a_whole_factor_and_says_so() {
        // 400 x 400 mm at 0.1 mm asks for about eight thousand lines.
        let grid = FloorGrid::derive(&domain(400.0, 400.0), 0.1).expect("a grid");
        assert!(
            grid.line_count() <= constants::VIEW_EDIT_GRID_MAX_LINES,
            "the cap let {} lines through",
            grid.line_count()
        );
        assert!(grid.coarsening > 1);
        assert_eq!(grid.requested_mm, 0.1);
        // Whole, so every line is still on a multiple of the increment.
        let ratio = grid.spacing_mm / grid.requested_mm;
        assert!((ratio - ratio.round()).abs() < 1e-9, "ratio {ratio}");
        let note = grid.note().expect("a coarsened grid says so");
        assert!(note.contains("floor grid ruled at"), "{note}");

        // One line under the cap is left exactly as asked.
        let inside = FloorGrid::derive(&domain(60.0, 30.0), 1.0).expect("a grid");
        assert_eq!(inside.coarsening, 1);
        assert!(inside.note().is_none());
    }

    #[test]
    fn a_domain_with_no_footprint_rules_nothing() {
        assert!(FloorGrid::derive(&Aabb::empty(), 1.0).is_none());
        let flat = Aabb {
            min: [0.0, 5.0, 0.0],
            max: [0.0, 5.0, 10.0],
        };
        assert!(FloorGrid::derive(&flat, 1.0).is_none());
    }

    #[test]
    fn the_mesh_carries_both_weights_and_both_colours() {
        let grid = FloorGrid::derive(&domain(60.0, 30.0), 1.0).expect("a grid");
        let mesh = grid.mesh().expect("geometry");
        assert!(mesh.triangles() > 0);
        let mut colors: Vec<[f32; 4]> = Vec::new();
        for vertex in &mesh.vertices {
            if !colors.contains(&vertex.color) {
                colors.push(vertex.color);
            }
        }
        assert_eq!(colors.len(), 2, "minor and major must be told apart");
        assert!(colors.contains(&constants::VIEW_COLOR_GRID_MINOR));
        assert!(colors.contains(&constants::VIEW_COLOR_GRID_MAJOR));
        // The drawn geometry stays on the floor plane, less the line weight.
        let weight = 0.01 * (grid.bounds.max[0] - grid.bounds.min[0]);
        assert!((mesh.bounds.min[2] - grid.bounds.min[2]).abs() < weight);
        assert!((mesh.bounds.max[2] - grid.bounds.min[2]).abs() < weight);
    }
}
