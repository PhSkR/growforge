//! The voxel grid seen as a region a strut may occupy.
//!
//! A cell is *allowed* when it carries material or may come to carry it, which
//! is every cell the classifier did not mark [`CellKind::Void`]: design cells
//! and forced solid cells. Keepouts and the space outside the domain are both
//! void, so both are impassable here, and the space outside the grid block is
//! impassable too.
//!
//! A symmetric run confines the growth further, to the fundamental domain of
//! its symmetry (see [`crate::engine::growth::symmetry`]): a cell outside the
//! sector is as impassable as a keepout, so every path, every step and every
//! attraction point stays inside the sector without any stage of the engine
//! having to know that it is one. What a region *is* stays a question about
//! material alone - see [`GrowthDomain::cells_touching_nodes`].
//!
//! Two containment tests live here and they are deliberately different. The
//! centre line of a segment is checked *exactly*, by walking the voxels the
//! segment passes through, because "a branch never enters a forbidden cell" is
//! an invariant of the engine. A capsule around that centre line is checked by
//! sampling, because it is only used to decide whether a shortcut is safe, and
//! the sampled test is conservative where it matters: the exact centre line
//! test runs alongside it.

use crate::constants;
use crate::engine::growth::symmetry::Symmetry;
use crate::geometry::{self, Vec3};
use crate::grid::{CellKind, Grid};

/// The allowed subset of a grid.
#[derive(Debug)]
pub struct GrowthDomain<'a> {
    grid: &'a Grid,
    allowed: Vec<bool>,
    confined: bool,
}

impl<'a> GrowthDomain<'a> {
    /// Classify a grid into allowed and forbidden cells.
    pub fn new(grid: &'a Grid) -> GrowthDomain<'a> {
        GrowthDomain {
            allowed: grid.cells.iter().map(|c| *c != CellKind::Void).collect(),
            grid,
            confined: false,
        }
    }

    /// The same, confined to the fundamental domain of `symmetry` when there is
    /// one.
    ///
    /// A cell whose centre lies outside the sector is forbidden exactly as a
    /// keepout cell is, which is what keeps the growth inside the sector; the
    /// copies the symmetry makes cover the rest of the domain.
    pub fn confined(grid: &'a Grid, symmetry: Option<&Symmetry>) -> GrowthDomain<'a> {
        let mut domain = GrowthDomain::new(grid);
        let Some(symmetry) = symmetry else {
            return domain;
        };
        for (cell, allowed) in domain.allowed.iter_mut().enumerate() {
            let (i, j, k) = grid.cell_ijk(cell);
            *allowed &= symmetry.contains(grid.cell_center(i, j, k));
        }
        domain.confined = true;
        domain
    }

    /// True when this view is confined to a symmetry's fundamental domain, so
    /// that an error message can say that the space outside it is impassable.
    pub fn is_confined(&self) -> bool {
        self.confined
    }

    /// The grid this view was built on.
    pub fn grid(&self) -> &Grid {
        self.grid
    }

    /// True when cell `e` may hold material.
    pub fn allows_cell(&self, cell: usize) -> bool {
        self.allowed[cell]
    }

    /// Grid coordinates of the cell containing `p`, or `None` outside the grid.
    fn coords_at(&self, p: Vec3) -> Option<(usize, usize, usize)> {
        let g = self.grid;
        let counts = [g.nx, g.ny, g.nz];
        let mut ijk = [0usize; 3];
        for d in 0..3 {
            let local = (p[d] - g.origin[d]) / g.h;
            if !local.is_finite() || local < 0.0 {
                return None;
            }
            let index = local.floor() as usize;
            if index >= counts[d] {
                return None;
            }
            ijk[d] = index;
        }
        Some((ijk[0], ijk[1], ijk[2]))
    }

    /// Index of the cell containing `p`, or `None` outside the grid.
    pub fn cell_at(&self, p: Vec3) -> Option<usize> {
        let (i, j, k) = self.coords_at(p)?;
        Some(self.grid.cell_index(i, j, k))
    }

    /// True when `p` lies inside an allowed cell.
    pub fn allows_point(&self, p: Vec3) -> bool {
        self.cell_at(p).is_some_and(|e| self.allowed[e])
    }

    /// True when every cell the ball of `radius` about `p` reaches into is
    /// allowed. A zero radius degenerates to [`GrowthDomain::allows_point`].
    pub fn allows_ball(&self, p: Vec3, radius: f64) -> bool {
        let g = self.grid;
        let counts = [g.nx, g.ny, g.nz];
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for d in 0..3 {
            let low = (p[d] - radius - g.origin[d]) / g.h;
            let high = (p[d] + radius - g.origin[d]) / g.h;
            if !low.is_finite() || !high.is_finite() {
                return false;
            }
            // A ball whose extreme point on this axis is outside the grid has
            // left the allowed region: outside the block is not material.
            if low < 0.0 || high >= counts[d] as f64 {
                return false;
            }
            lo[d] = low.floor() as usize;
            hi[d] = (high.floor() as usize).min(counts[d] - 1);
        }
        let radius_sq = radius * radius;
        for k in lo[2]..=hi[2] {
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    // Only cells the ball really reaches into matter; the index
                    // ranges above cover the ball's bounding box, which sticks
                    // out at the corners.
                    let mut distance_sq = 0.0;
                    for (d, index) in [i, j, k].into_iter().enumerate() {
                        let min = g.origin[d] + index as f64 * g.h;
                        let gap = (min - p[d]).max(p[d] - (min + g.h)).max(0.0);
                        distance_sq += gap * gap;
                    }
                    if distance_sq <= radius_sq && !self.allowed[g.cell_index(i, j, k)] {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// True when the straight segment from `a` to `b` only ever passes through
    /// allowed cells.
    ///
    /// Exact: the cells are enumerated by a 3D digital differential analyser
    /// (Amanatides and Woo's voxel traversal), so a segment that clips the
    /// corner of a keepout for a fraction of a voxel is still rejected.
    pub fn segment_inside(&self, a: Vec3, b: Vec3) -> bool {
        let g = self.grid;
        if !b.iter().all(|v| v.is_finite()) {
            return false;
        }
        let Some((i0, j0, k0)) = self.coords_at(a) else {
            return false;
        };
        if !self.allowed[g.cell_index(i0, j0, k0)] {
            return false;
        }
        let direction = geometry::difference(b, a);
        let mut current = [i0 as i64, j0 as i64, k0 as i64];
        let counts = [g.nx as i64, g.ny as i64, g.nz as i64];

        // Everything is parametrized by u in [0, 1] along the segment, so the
        // walk ends exactly at b whatever the direction's length is.
        let mut step = [0i64; 3];
        let mut next_crossing = [f64::INFINITY; 3];
        let mut crossing_delta = [f64::INFINITY; 3];
        for d in 0..3 {
            if direction[d] == 0.0 {
                continue;
            }
            step[d] = if direction[d] > 0.0 { 1 } else { -1 };
            let boundary_index = current[d] + i64::from(direction[d] > 0.0);
            let boundary = g.origin[d] + boundary_index as f64 * g.h;
            next_crossing[d] = (boundary - a[d]) / direction[d];
            crossing_delta[d] = g.h / direction[d].abs();
        }

        // One cell per crossing, and a segment can cross at most the grid's
        // full extent on each axis.
        let limit = counts[0] + counts[1] + counts[2] + 3;
        for _ in 0..limit {
            let axis = (0..3)
                .min_by(|x, y| next_crossing[*x].total_cmp(&next_crossing[*y]))
                .expect("three axes");
            // Nothing crosses before the far end: b lies in the cell the walk
            // is standing in, which was checked on entry. The endpoints are
            // finite, so a crossing parameter is never a NaN.
            if next_crossing[axis] > 1.0 {
                return true;
            }
            current[axis] += step[axis];
            if current[axis] < 0 || current[axis] >= counts[axis] {
                return false;
            }
            let cell = g.cell_index(
                current[0] as usize,
                current[1] as usize,
                current[2] as usize,
            );
            if !self.allowed[cell] {
                return false;
            }
            next_crossing[axis] += crossing_delta[axis];
        }
        // Unreachable for a segment inside the block; refusing is the safe end.
        false
    }

    /// True when the capsule of `radius` around the segment `a`..`b` stays
    /// inside the allowed cells.
    ///
    /// The centre line is exact (see [`GrowthDomain::segment_inside`]); the
    /// radius is tested on samples half a voxel apart, whose balls overlap, so
    /// the only region the test can miss is a scallop between two samples far
    /// thinner than a voxel.
    pub fn capsule_inside(&self, a: Vec3, b: Vec3, radius: f64) -> bool {
        if !self.segment_inside(a, b) {
            return false;
        }
        if radius <= 0.0 {
            return true;
        }
        let spacing = constants::GROWTH_SEGMENT_SAMPLE_VOXELS * self.grid.h;
        let length = geometry::length(geometry::difference(b, a));
        let steps = (length / spacing).ceil().max(1.0) as usize;
        for s in 0..=steps {
            let u = s as f64 / steps as f64;
            let p = geometry::sum(a, geometry::scale(geometry::difference(b, a), u));
            if !self.allows_ball(p, radius) {
                return false;
            }
        }
        true
    }

    /// The **material** cells touching any of `nodes`, ascending and
    /// deduplicated.
    ///
    /// A load or support region is a set of nodes, and a skeleton lives on cell
    /// centres; this is the bridge between the two.
    ///
    /// Deliberately *not* confined to the fundamental domain of a symmetry: a
    /// region is a piece of the problem, and half a region is not one. A
    /// support patch that straddles a mirror plane holds material on both sides
    /// of it, and a branch that ends on the far half has still ended on a
    /// support. Which regions a symmetric run routes a backbone to, and which
    /// of their cells it may route through, are separate questions answered in
    /// [`crate::engine::growth`] and by [`GrowthDomain::allows_cell`].
    pub fn cells_touching_nodes(&self, nodes: &[usize]) -> Vec<usize> {
        self.grid.cells_touching_nodes(nodes)
    }

    /// Centre of cell `e` in millimetres.
    pub fn cell_center(&self, cell: usize) -> Vec3 {
        let (i, j, k) = self.grid.cell_ijk(cell);
        self.grid.cell_center(i, j, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Axis, SymmetryParams};
    use crate::engine::growth::fixtures::design_grid;

    #[test]
    fn a_confined_domain_stops_at_the_symmetry_plane_but_still_knows_its_regions() {
        // Ten cells of 1 mm from the origin, so the block centre is x = 5 and
        // the fundamental half is the five cells i = 0 .. 4.
        let grid = design_grid(10);
        let symmetry = Symmetry::new(
            SymmetryParams::Mirror {
                first: Axis::X,
                second: None,
            },
            &grid.bounds(),
            constants::GROWTH_SYMMETRY_BOUNDARY_EPSILON_VOXELS * grid.h,
        );
        let domain = GrowthDomain::confined(&grid, Some(&symmetry));
        assert!(domain.is_confined());
        assert!(!GrowthDomain::confined(&grid, None).is_confined());

        for i in 0..10 {
            let cell = grid.cell_index(i, 3, 3);
            assert_eq!(
                domain.allows_cell(cell),
                i < 5,
                "cell {i} on the wrong side of the plane"
            );
        }
        assert!(domain.allows_point([4.5, 3.5, 3.5]));
        assert!(!domain.allows_point([5.5, 3.5, 3.5]));
        // A step across the plane is refused exactly as a step into a keepout
        // is, so the growth cannot leave its sector.
        assert!(domain.segment_inside([1.5, 3.5, 3.5], [4.5, 3.5, 3.5]));
        assert!(!domain.segment_inside([4.5, 3.5, 3.5], [5.5, 3.5, 3.5]));
        assert!(!domain.allows_ball([4.5, 3.5, 3.5], 1.5));

        // But a region is still the whole region: the node in the middle of the
        // block touches cells on both sides, and all eight of them come back.
        let node = grid.node_index(5, 5, 5);
        let cells = domain.cells_touching_nodes(&[node]);
        assert_eq!(cells.len(), 8);
        assert!(cells.contains(&grid.cell_index(5, 4, 4)), "{cells:?}");
        assert_eq!(
            cells,
            GrowthDomain::new(&grid).cells_touching_nodes(&[node])
        );
    }

    #[test]
    fn points_outside_the_block_and_inside_a_keepout_are_both_forbidden() {
        let mut grid = design_grid(4);
        let blocked = grid.cell_index(2, 2, 2);
        grid.cells[blocked] = CellKind::Void;
        let domain = GrowthDomain::new(&grid);

        assert!(domain.allows_point([0.5, 0.5, 0.5]));
        assert!(!domain.allows_point([2.5, 2.5, 2.5]));
        assert!(!domain.allows_point([-0.1, 0.5, 0.5]));
        assert!(!domain.allows_point([4.5, 0.5, 0.5]));
        assert_eq!(domain.cell_at([2.5, 2.5, 2.5]), Some(blocked));
        assert_eq!(domain.cell_at([100.0, 0.0, 0.0]), None);
    }

    #[test]
    fn the_voxel_walk_catches_a_wall_the_segment_only_clips() {
        let mut grid = design_grid(8);
        // A wall at i = 4 with a single hole at (4, 4, 4).
        for k in 0..8 {
            for j in 0..8 {
                let cell = grid.cell_index(4, j, k);
                grid.cells[cell] = CellKind::Void;
            }
        }
        let hole = grid.cell_index(4, 4, 4);
        grid.cells[hole] = CellKind::Design;
        let domain = GrowthDomain::new(&grid);

        // Straight through the hole.
        assert!(domain.segment_inside([0.5, 4.5, 4.5], [7.5, 4.5, 4.5]));
        // Straight into the wall.
        assert!(!domain.segment_inside([0.5, 1.5, 4.5], [7.5, 1.5, 4.5]));
        // A shallow diagonal aimed at the hole, which crosses into the
        // forbidden cell (4, 5, 4) at x = 4.9 and is inside it for a tenth of a
        // voxel. Sampling would step over that; the voxel walk cannot.
        assert!(!domain.segment_inside([0.5, 4.912, 4.5], [7.5, 5.052, 4.5]));
        // A segment that leaves the block is refused as well.
        assert!(!domain.segment_inside([0.5, 4.5, 4.5], [12.0, 4.5, 4.5]));
        // A degenerate segment is just its own cell.
        assert!(domain.segment_inside([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]));
    }

    #[test]
    fn the_capsule_test_needs_room_the_centre_line_does_not() {
        let mut grid = design_grid(9);
        // A slab of allowed cells three voxels thick in k, forbidden elsewhere.
        for k in 0..9 {
            for j in 0..9 {
                for i in 0..9 {
                    if !(3..6).contains(&k) {
                        let cell = grid.cell_index(i, j, k);
                        grid.cells[cell] = CellKind::Void;
                    }
                }
            }
        }
        let domain = GrowthDomain::new(&grid);
        let a = [0.5, 4.5, 4.5];
        let b = [8.5, 4.5, 4.5];
        assert!(domain.segment_inside(a, b));
        // Half a voxel of clearance either side of the centre line.
        assert!(domain.capsule_inside(a, b, 0.4));
        // More than the slab can hold.
        assert!(!domain.capsule_inside(a, b, 1.6));
        // A zero radius is the centre line.
        assert!(domain.capsule_inside(a, b, 0.0));
    }

    #[test]
    fn a_ball_that_reaches_out_of_the_block_is_refused() {
        let grid = design_grid(4);
        let domain = GrowthDomain::new(&grid);
        assert!(domain.allows_ball([2.0, 2.0, 2.0], 1.5));
        assert!(!domain.allows_ball([2.0, 2.0, 2.0], 2.5));
        assert!(!domain.allows_ball([0.2, 2.0, 2.0], 0.5));
    }

    #[test]
    fn nodes_map_to_the_allowed_cells_that_touch_them() {
        let mut grid = design_grid(3);
        let blocked = grid.cell_index(0, 0, 0);
        grid.cells[blocked] = CellKind::Void;
        let domain = GrowthDomain::new(&grid);

        // The interior node (1, 1, 1) touches eight cells, one of them void.
        let node = grid.node_index(1, 1, 1);
        let cells = domain.cells_touching_nodes(&[node]);
        assert_eq!(cells.len(), 7);
        assert!(!cells.contains(&blocked));
        assert!(cells.windows(2).all(|w| w[0] < w[1]), "must be sorted");

        // A corner node touches one cell, and the same node twice is one cell.
        let corner = grid.node_index(3, 3, 3);
        assert_eq!(
            domain.cells_touching_nodes(&[corner, corner]),
            vec![grid.cell_index(2, 2, 2)]
        );
    }
}
