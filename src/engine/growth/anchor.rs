//! Where a branch is allowed to end, and the pass that removes the ones that
//! end anywhere else.
//!
//! A branch that stops in mid air because the attraction points around it ran
//! out is not a structure. It carries no load, which makes it dead mass in a
//! tool whose whole purpose is weight; it is an unsupported overhang, which
//! makes it unprintable; and it reads as a model that stopped half way, which is
//! how this was found. Growforge therefore treats a free branch tip as a defect
//! rather than as decoration.
//!
//! An **anchor cell** is a cell the finite element model already treats as real,
//! independently of anything the growth engine does:
//!
//! * a forced solid ([`CellKind::Solid`]) cell, which carries material at
//!   density one whatever grows near it;
//! * a cell of a support region, whose nodes are constrained, so material there
//!   is held;
//! * a cell of a load region, whose nodes carry the applied force, so material
//!   there is loaded directly.
//!
//! A branch tip within the fusion tolerance of one of those has grown into
//! something: the capsule around the tip reaches into the cell and the two come
//! out of the isosurface as a single solid. Every other tip is free, and
//! [`prune`] removes it together with the whole dead-end chain behind it, back
//! to the last junction that still leads somewhere.

use crate::engine::growth::skeleton::Skeleton;
use crate::geometry::Vec3;
use crate::grid::{CellKind, Grid};

/// The cells a branch may legitimately end on.
#[derive(Debug, Clone)]
pub struct AnchorSet {
    cells: Vec<bool>,
    count: usize,
}

impl AnchorSet {
    /// An empty set over `grid`.
    pub fn empty(grid: &Grid) -> AnchorSet {
        AnchorSet {
            cells: vec![false; grid.n_cells()],
            count: 0,
        }
    }

    /// The set of every forced solid cell, which is material by construction.
    pub fn solid(grid: &Grid) -> AnchorSet {
        let mut set = AnchorSet::empty(grid);
        for (cell, kind) in grid.cells.iter().enumerate() {
            if *kind == CellKind::Solid {
                set.insert(cell);
            }
        }
        set
    }

    /// Add one cell.
    pub fn insert(&mut self, cell: usize) {
        if !self.cells[cell] {
            self.cells[cell] = true;
            self.count += 1;
        }
    }

    /// Add every cell of `cells`.
    pub fn extend(&mut self, cells: &[usize]) {
        for &cell in cells {
            self.insert(cell);
        }
    }

    /// True when `cell` is an anchor.
    pub fn contains(&self, cell: usize) -> bool {
        self.cells[cell]
    }

    /// How many cells the set holds.
    pub fn len(&self) -> usize {
        self.count
    }

    /// True when nothing anchors anything.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// True when any anchor cell comes within `tolerance` of `p`.
    ///
    /// The test is against the cell's box rather than its centre, so a tip that
    /// has grown *into* an anchor cell is at distance zero from it however far
    /// it is from the centre.
    pub fn near(&self, grid: &Grid, p: Vec3, tolerance: f64) -> bool {
        let counts = [grid.nx, grid.ny, grid.nz];
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for d in 0..3 {
            let low = (p[d] - tolerance - grid.origin[d]) / grid.h;
            let high = (p[d] + tolerance - grid.origin[d]) / grid.h;
            if !low.is_finite() || !high.is_finite() {
                return false;
            }
            if high < 0.0 || low >= counts[d] as f64 {
                return false;
            }
            lo[d] = low.max(0.0) as usize;
            hi[d] = high.min(counts[d] as f64 - 1.0) as usize;
        }
        let tolerance_sq = tolerance * tolerance;
        for k in lo[2]..=hi[2] {
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    let cell = grid.cell_index(i, j, k);
                    if !self.cells[cell] {
                        continue;
                    }
                    let mut distance_sq = 0.0;
                    for (d, index) in [i, j, k].into_iter().enumerate() {
                        let min = grid.origin[d] + index as f64 * grid.h;
                        let gap = (min - p[d]).max(p[d] - (min + grid.h)).max(0.0);
                        distance_sq += gap * gap;
                    }
                    if distance_sq <= tolerance_sq {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Anchor cell centres thinned to a minimum separation of `spacing`.
    ///
    /// Growth needs somewhere to aim, and aiming at every anchor cell of a
    /// tabletop would ask for one branch per voxel. Cells are visited in index
    /// order and kept only when no already kept point is within `spacing`, which
    /// is a deterministic Poisson-disk thinning: what comes back is one point
    /// per `spacing`-sized patch of structural surface, and index order means
    /// the layer nearest the grid origin - the underside of a plate - is the one
    /// that keeps them.
    pub fn thinned(&self, grid: &Grid, spacing: f64) -> Vec<Vec3> {
        let mut kept: Vec<Vec3> = Vec::new();
        let spacing_sq = spacing * spacing;
        for cell in 0..self.cells.len() {
            if !self.cells[cell] {
                continue;
            }
            let (i, j, k) = grid.cell_ijk(cell);
            let centre = grid.cell_center(i, j, k);
            let crowded = kept.iter().any(|p| {
                let d = [centre[0] - p[0], centre[1] - p[1], centre[2] - p[2]];
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < spacing_sq
            });
            if !crowded {
                kept.push(centre);
            }
        }
        kept
    }
}

/// The test that decides whether a point has fused to structural material.
///
/// Bundled because the three parts of it always travel together: the cells that
/// count, the grid they are indexed on, and how close is close enough.
#[derive(Debug, Clone, Copy)]
pub struct Fusion<'a> {
    /// Cells a branch may end on.
    pub anchors: &'a AnchorSet,
    /// Grid the anchor set is indexed on.
    pub grid: &'a Grid,
    /// How close a point has to come, in millimetres.
    pub tolerance_mm: f64,
}

impl Fusion<'_> {
    /// True when `p` has fused to structural material.
    pub fn fused(&self, p: Vec3) -> bool {
        self.anchors.near(self.grid, p, self.tolerance_mm)
    }
}

/// Every forced solid cell reachable from the solid cells of `seeds` through
/// face connected forced solid cells.
///
/// A load applied on a rigid body is carried by everything that fuses to that
/// body, not only by what fuses within a voxel of where the arrow was drawn. A
/// load region that sits on a keepin pad therefore extends over the whole pad,
/// which is what lets a branch reaching the underside of a tabletop count as
/// carrying the tabletop.
pub fn solid_body(grid: &Grid, seeds: &[usize]) -> Vec<usize> {
    let mut seen = vec![false; grid.n_cells()];
    let mut stack: Vec<usize> = seeds
        .iter()
        .copied()
        .filter(|&c| grid.cells[c] == CellKind::Solid)
        .collect();
    for &cell in &stack {
        seen[cell] = true;
    }
    let mut body = stack.clone();
    while let Some(cell) = stack.pop() {
        let (i, j, k) = grid.cell_ijk(cell);
        for (di, dj, dk) in [
            (-1i32, 0i32, 0i32),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ] {
            let (ni, nj, nk) = (i as i32 + di, j as i32 + dj, k as i32 + dk);
            if ni < 0
                || nj < 0
                || nk < 0
                || ni >= grid.nx as i32
                || nj >= grid.ny as i32
                || nk >= grid.nz as i32
            {
                continue;
            }
            let next = grid.cell_index(ni as usize, nj as usize, nk as usize);
            if !seen[next] && grid.cells[next] == CellKind::Solid {
                seen[next] = true;
                body.push(next);
                stack.push(next);
            }
        }
    }
    body.sort_unstable();
    body
}

/// Skeleton leaves that end within `tolerance` of an anchor cell.
pub fn fused_leaves(
    skeleton: &Skeleton,
    anchors: &AnchorSet,
    grid: &Grid,
    tolerance: f64,
) -> Vec<usize> {
    let leaf = skeleton.leaves();
    (0..skeleton.len())
        .filter(|&n| leaf[n] && anchors.near(grid, skeleton.position(n), tolerance))
        .collect()
}

/// Skeleton leaves that end on nothing at all.
///
/// With pruning on this is empty by construction, and the tests hold it to that.
pub fn free_leaves(
    skeleton: &Skeleton,
    anchors: &AnchorSet,
    grid: &Grid,
    tolerance: f64,
) -> Vec<usize> {
    let leaf = skeleton.leaves();
    (0..skeleton.len())
        .filter(|&n| leaf[n] && !anchors.near(grid, skeleton.position(n), tolerance))
        .collect()
}

/// What a pruning pass removed.
#[derive(Debug, Clone)]
pub struct Pruning {
    /// Nodes removed.
    pub removed: usize,
    /// Old node index to new node index; `None` for a node that was removed.
    pub map: Vec<Option<usize>>,
}

/// Remove every branch that does not end on an anchor.
///
/// A node is kept when it is fused to an anchor itself, or when any node below
/// it is: one reverse pass suffices, because a node's parent always has a
/// smaller index, so every child has been decided before its parent is. What
/// survives is exactly the union of the paths from the roots to the fused tips -
/// dead-end chains vanish back to the last junction that still leads somewhere,
/// however long they are, and no iteration to a fixed point is needed.
///
/// The backbones are never at risk: their roots sit on support cells and their
/// tips on load region cells, both of which are anchors.
pub fn prune(skeleton: &mut Skeleton, anchors: &AnchorSet, grid: &Grid, tolerance: f64) -> Pruning {
    let mut keep: Vec<bool> = (0..skeleton.len())
        .map(|n| anchors.near(grid, skeleton.position(n), tolerance))
        .collect();
    for index in (0..skeleton.len()).rev() {
        if keep[index]
            && let Some(parent) = skeleton.nodes[index].parent
        {
            keep[parent] = true;
        }
    }
    let removed = keep.iter().filter(|k| !**k).count();
    let map = skeleton.retain(&keep);
    Pruning { removed, map }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use crate::engine::growth::fixtures::design_grid;

    /// A block with a forced solid plate over the top two layers.
    fn plated(n: usize) -> Grid {
        let mut grid = design_grid(n);
        for k in n - 2..n {
            for j in 0..n {
                for i in 0..n {
                    let cell = grid.cell_index(i, j, k);
                    grid.cells[cell] = CellKind::Solid;
                }
            }
        }
        grid
    }

    #[test]
    fn the_solid_cells_are_anchors_and_the_design_cells_are_not() {
        let grid = plated(8);
        let anchors = AnchorSet::solid(&grid);
        assert_eq!(anchors.len(), 8 * 8 * 2);
        assert!(!anchors.is_empty());
        assert!(anchors.contains(grid.cell_index(3, 3, 7)));
        assert!(!anchors.contains(grid.cell_index(3, 3, 3)));

        // A point inside the plate is at distance zero from it.
        assert!(anchors.near(&grid, [3.5, 3.5, 6.5], 0.0));
        // The plate's underside is the plane z = 6, so the centre of the cell
        // below it is half a voxel away.
        assert!(!anchors.near(&grid, [3.5, 3.5, 5.5], 0.4));
        assert!(anchors.near(&grid, [3.5, 3.5, 5.5], 0.6));
        // Two voxels below, it is out of reach of anything a strut could bridge.
        assert!(!anchors.near(&grid, [3.5, 3.5, 4.5], 1.4));
        // And a point far outside the block anchors nothing.
        assert!(!anchors.near(&grid, [100.0, 0.0, 0.0], 5.0));
        assert!(!anchors.near(&grid, [f64::NAN, 0.0, 0.0], 5.0));
    }

    /// The fusion tolerance may never accept a tip that only *touches* its
    /// anchor.
    ///
    /// A strut of radius `r` whose tip sits exactly `r` from the anchor has the
    /// anchor's nearest point exactly on its capsule surface: signed distance
    /// zero, density exactly the iso level, tangent rather than overlapping.
    /// Marching cubes can extract that as two separate shells, and the mesh
    /// validation checks manifoldness rather than connectedness, so it would
    /// ship as a floating chunk. The tolerance is therefore strictly inside the
    /// zero crossing, and this test stands on the exact boundary to prove it.
    #[test]
    fn a_tip_that_only_touches_its_anchor_does_not_count_as_fused() {
        let grid = plated(8);
        let anchors = AnchorSet::solid(&grid);
        // The plate's underside is the plane z = 6.
        let min_radius = 1.0;
        let tolerance = constants::GROWTH_FUSION_RADII * min_radius;
        // A tolerance of a whole radius is tangency, not fusion.
        const { assert!(constants::GROWTH_FUSION_RADII < 1.0) };

        // Exactly one strut radius below the plate: the capsule would touch it
        // at a single point and no more.
        let tangent: Vec3 = [3.5, 3.5, 6.0 - min_radius];
        assert!(
            !anchors.near(&grid, tangent, tolerance),
            "a tip that only touches its anchor was accepted as fused"
        );
        // The old, knife-edge tolerance is what accepted it.
        assert!(anchors.near(&grid, tangent, min_radius));

        // Just inside the new threshold, where the capsule really does overlap.
        let overlapping: Vec3 = [3.5, 3.5, 6.0 - tolerance + 1e-9];
        assert!(
            anchors.near(&grid, overlapping, tolerance),
            "a tip inside the threshold was not accepted"
        );
        // The penetration such a tip buys is strictly positive, and it is the
        // margin the constant's arithmetic is written against.
        let penetration = min_radius - tolerance;
        assert!(penetration > 0.0);
        assert!((penetration - 0.2 * min_radius).abs() < 1e-12);

        // The worst case the doc comment quotes: with the tip that far out, a
        // cell centre on the way to the anchor still sits strictly inside the
        // capsule, so its density is strictly above the iso level. The grid here
        // has h = 1, and the guidance is min_feature >= 3 h, so r >= 1.5 h.
        let r = 1.5 * grid.h;
        let worst = (constants::GROWTH_FUSION_RADII * r).hypot(grid.h * 3.0f64.sqrt() / 2.0);
        assert!(
            worst < r,
            "a cell on the way to the anchor sits {worst} out, past the {r} capsule"
        );
    }

    #[test]
    fn extra_regions_join_the_set_without_duplicating_it() {
        let grid = plated(6);
        let mut anchors = AnchorSet::solid(&grid);
        let before = anchors.len();
        let floor: Vec<usize> = (0..6).map(|i| grid.cell_index(i, 0, 0)).collect();
        anchors.extend(&floor);
        assert_eq!(anchors.len(), before + floor.len());
        // Re-adding the plate changes nothing.
        anchors.extend(&[grid.cell_index(0, 0, 5)]);
        assert_eq!(anchors.len(), before + floor.len());
        assert!(anchors.contains(floor[3]));
    }

    #[test]
    fn thinning_keeps_one_point_per_patch_starting_at_the_underside() {
        let grid = plated(12);
        let anchors = AnchorSet::solid(&grid);
        let spacing = 4.0;
        let points = anchors.thinned(&grid, spacing);

        assert!(points.len() > 1, "the plate needs several aim points");
        assert!(
            points.len() < anchors.len() / 4,
            "{} points is barely a thinning of {}",
            points.len(),
            anchors.len()
        );
        // No two kept points are closer than the spacing.
        for (a, first) in points.iter().enumerate() {
            for second in &points[a + 1..] {
                let d = crate::geometry::length(crate::geometry::difference(*first, *second));
                assert!(d >= spacing - 1e-12, "two points {d} apart");
            }
        }
        // Index order visits the lower layer of the plate first, so that is the
        // one the points end up on: the underside, which is where a branch can
        // reach them.
        assert!(
            points.iter().all(|p| p[2] < 11.0),
            "a point landed on the top layer of the plate"
        );
        // Every point is inside an anchor cell.
        for p in &points {
            let cell = grid.cell_index(p[0] as usize, p[1] as usize, p[2] as usize);
            assert!(anchors.contains(cell), "point {p:?} is not on an anchor");
        }
    }

    #[test]
    fn a_solid_body_is_the_whole_connected_plate() {
        let mut grid = plated(8);
        // A second, separate solid block in the corner.
        let island = grid.cell_index(0, 0, 0);
        grid.cells[island] = CellKind::Solid;

        let body = solid_body(&grid, &[grid.cell_index(4, 4, 7)]);
        assert_eq!(body.len(), 8 * 8 * 2, "the plate is one body");
        assert!(!body.contains(&island), "the island is a different body");
        assert_eq!(solid_body(&grid, &[island]), vec![island]);
        // A seed that is not solid at all contributes nothing.
        assert!(solid_body(&grid, &[grid.cell_index(4, 4, 1)]).is_empty());
    }

    /// A backbone from the floor to the plate, plus a three node side branch
    /// that ends in mid air.
    fn skeleton_with_a_dangling_branch(grid: &Grid) -> (Skeleton, usize) {
        let mut skeleton = Skeleton::new();
        let root = skeleton.add_root([4.5, 4.5, 0.5]);
        let mid = skeleton.add_child(root, [4.5, 4.5, 3.5]);
        let tip = skeleton.add_child(mid, [4.5, 4.5, 6.5]);
        // The dangling chain, growing sideways off the junction and stopping.
        let mut free = skeleton.add_child(mid, [5.5, 4.5, 3.5]);
        for step in 0..2 {
            free = skeleton.add_child(free, [6.5 + step as f64, 4.5, 3.5]);
        }
        let _ = grid;
        (skeleton, tip)
    }

    #[test]
    fn pruning_removes_a_dead_end_chain_back_to_its_junction() {
        let grid = plated(8);
        let mut anchors = AnchorSet::solid(&grid);
        anchors.extend(&[grid.cell_index(4, 4, 0)]);
        let (mut skeleton, tip) = skeleton_with_a_dangling_branch(&grid);
        assert_eq!(skeleton.len(), 6);
        assert_eq!(free_leaves(&skeleton, &anchors, &grid, 1.0).len(), 1);

        let pruning = prune(&mut skeleton, &anchors, &grid, 1.0);

        assert_eq!(pruning.removed, 3, "the whole dead-end chain has to go");
        assert_eq!(skeleton.len(), 3, "the backbone has to survive intact");
        assert!(
            free_leaves(&skeleton, &anchors, &grid, 1.0).is_empty(),
            "a free tip survived the pruning"
        );
        // The backbone's nodes are remapped, in order, and still a chain.
        assert_eq!(pruning.map[0], Some(0));
        assert_eq!(pruning.map[tip], Some(2));
        assert_eq!(pruning.map[3], None);
        assert_eq!(skeleton.nodes[2].parent, Some(1));
        assert_eq!(skeleton.nodes[0].parent, None);
    }

    #[test]
    fn a_chain_that_reaches_the_plate_is_kept_whole() {
        let grid = plated(8);
        let mut anchors = AnchorSet::solid(&grid);
        anchors.extend(&[grid.cell_index(4, 4, 0)]);
        let (mut skeleton, _) = skeleton_with_a_dangling_branch(&grid);
        // Extend the dangling chain up into the plate: now it ends on something.
        let free = skeleton.len() - 1;
        skeleton.add_child(free, [7.5, 4.5, 6.5]);

        let before = skeleton.len();
        let pruning = prune(&mut skeleton, &anchors, &grid, 1.0);
        assert_eq!(pruning.removed, 0, "nothing dead ends any more");
        assert_eq!(skeleton.len(), before);
        assert_eq!(fused_leaves(&skeleton, &anchors, &grid, 1.0).len(), 2);
    }

    #[test]
    fn pruning_a_skeleton_that_ends_on_nothing_at_all_empties_it() {
        // Degenerate but reachable: no anchors, so nothing is structural.
        let grid = design_grid(8);
        let anchors = AnchorSet::empty(&grid);
        let (mut skeleton, _) = skeleton_with_a_dangling_branch(&grid);
        let pruning = prune(&mut skeleton, &anchors, &grid, 1.0);
        assert_eq!(pruning.removed, 6);
        assert!(skeleton.is_empty());
        assert!(pruning.map.iter().all(|m| m.is_none()));
    }
}
