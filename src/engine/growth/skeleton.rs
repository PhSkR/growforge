//! The strut skeleton: a rooted forest of nodes, and the shortcut pass that
//! turns a voxel path into a polyline.
//!
//! Roots sit on the supports and every other node has exactly one parent, so a
//! segment is a node together with its parent and the load a branch carries
//! accumulates by walking the nodes backwards. A node's parent always has a
//! smaller index than the node itself, because a child can only be created
//! after its parent, which is what makes that walk a single reverse pass.

use crate::constants;
use crate::engine::growth::domain::GrowthDomain;
use crate::geometry::{self, Vec3};

/// One point of the skeleton.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node {
    /// Position in millimetres.
    pub position: Vec3,
    /// Parent node, or `None` for a root sitting on a support.
    pub parent: Option<usize>,
}

/// A rooted forest of strut nodes.
#[derive(Debug, Clone, Default)]
pub struct Skeleton {
    /// Nodes in creation order; every parent index is smaller than its child's.
    pub nodes: Vec<Node>,
}

impl Skeleton {
    /// An empty skeleton.
    pub fn new() -> Skeleton {
        Skeleton::default()
    }

    /// Add a root at `position` and return its index.
    pub fn add_root(&mut self, position: Vec3) -> usize {
        self.nodes.push(Node {
            position,
            parent: None,
        });
        self.nodes.len() - 1
    }

    /// Add a child of `parent` at `position` and return its index.
    pub fn add_child(&mut self, parent: usize, position: Vec3) -> usize {
        debug_assert!(
            parent < self.nodes.len(),
            "a child needs an existing parent"
        );
        self.nodes.push(Node {
            position,
            parent: Some(parent),
        });
        self.nodes.len() - 1
    }

    /// Append `points` as a chain, the first one a root, and return the index of
    /// the last node.
    pub fn add_chain(&mut self, points: &[Vec3]) -> usize {
        let mut previous = self.add_root(points[0]);
        for point in &points[1..] {
            previous = self.add_child(previous, *point);
        }
        previous
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when nothing has been grown yet.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of segments, one per node that has a parent.
    pub fn segment_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.parent.is_some()).count()
    }

    /// Position of node `index`.
    pub fn position(&self, index: usize) -> Vec3 {
        self.nodes[index].position
    }

    /// Every node that no other node claims as its parent.
    pub fn leaves(&self) -> Vec<bool> {
        let mut leaf = vec![true; self.nodes.len()];
        for node in &self.nodes {
            if let Some(parent) = node.parent {
                leaf[parent] = false;
            }
        }
        leaf
    }

    /// Total length of all segments in millimetres.
    pub fn total_length_mm(&self) -> f64 {
        self.nodes
            .iter()
            .filter_map(|n| n.parent.map(|p| (n.position, self.nodes[p].position)))
            .map(|(a, b)| geometry::length(geometry::difference(a, b)))
            .sum()
    }

    /// Drop every node not flagged in `keep`, returning the old index to new
    /// index map.
    ///
    /// The kept set has to be closed upwards - keeping a node means keeping its
    /// parent - or the survivors would not be a forest any more. Creation order
    /// is preserved among the survivors, which is what keeps a pruned skeleton
    /// as reproducible as the one it came from.
    pub fn retain(&mut self, keep: &[bool]) -> Vec<Option<usize>> {
        debug_assert_eq!(keep.len(), self.nodes.len(), "one flag per node");
        let mut map = vec![None; self.nodes.len()];
        let mut kept = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            if !keep[index] {
                continue;
            }
            let parent = match node.parent {
                Some(parent) => {
                    debug_assert!(
                        keep[parent],
                        "node {index} was kept but its parent {parent} was not"
                    );
                    map[parent]
                }
                None => None,
            };
            map[index] = Some(kept.len());
            kept.push(Node {
                position: node.position,
                parent,
            });
        }
        self.nodes = kept;
        map
    }
}

/// The largest line-of-sight radius, from a halving ladder starting at
/// `radius`, that the unsimplified path itself already satisfies.
///
/// A shortcut has to keep a strut of the minimum radius inside the allowed
/// cells, but in a domain thinner than that radius no segment could ever pass
/// that test, not even the voxel path the shortcut would replace. Backing the
/// radius off until the original path clears it keeps the rule meaningful in a
/// thin domain and guarantees a shortcut is never worse than what it replaces.
/// The ladder ends at zero, the bare centre line, which every voxel path
/// satisfies by construction.
pub fn shortcut_radius(domain: &GrowthDomain, points: &[Vec3], radius: f64) -> f64 {
    let mut candidate = radius;
    for _ in 0..constants::GROWTH_SIMPLIFY_RADIUS_BACKOFFS {
        if points
            .windows(2)
            .all(|pair| domain.capsule_inside(pair[0], pair[1], candidate))
        {
            return candidate;
        }
        candidate *= 0.5;
    }
    0.0
}

/// Put a node every `spacing` along a polyline, endpoints included.
///
/// The shortcut pass collapses a voxel path into as few points as the domain
/// allows, which is what takes the staircase out of it; that leaves a trunk of
/// one or two enormous segments. Resampling it puts the nodes back without
/// putting the staircase back - every inserted point lies on the straight line
/// between two points the shortcut already proved clear - and a trunk made of
/// nodes is a trunk the canopy can sprout from and Murray's law can taper along.
pub fn resample(points: &[Vec3], spacing: f64) -> Vec<Vec3> {
    if points.len() < 2 || spacing <= 0.0 || spacing.is_nan() {
        return points.to_vec();
    }
    let mut resampled = vec![points[0]];
    for pair in points.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        let span = geometry::difference(to, from);
        let length = geometry::length(span);
        let steps = (length / spacing).floor() as usize;
        for step in 1..=steps {
            let along = step as f64 * spacing;
            // The last insertion is skipped when it would land on top of the
            // end point, which the next iteration adds anyway.
            if length - along < 0.5 * spacing {
                break;
            }
            resampled.push(geometry::sum(from, geometry::scale(span, along / length)));
        }
        resampled.push(to);
    }
    resampled
}

/// Greedy line-of-sight shortcutting of a voxel path.
///
/// Walks forward from the current anchor for as long as the straight segment
/// from the anchor to the candidate keeps a capsule of `radius` inside the
/// allowed cells, and keeps the last candidate that did. What comes out is a
/// polyline through a subset of the original points, never through anything
/// else, so it cannot route through a cell the path itself avoided.
pub fn shortcut(domain: &GrowthDomain, points: &[Vec3], radius: f64) -> Vec<Vec3> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut simplified = vec![points[0]];
    let mut anchor = 0;
    while anchor + 1 < points.len() {
        let mut reach = anchor + 1;
        for candidate in anchor + 2..points.len() {
            if !domain.capsule_inside(points[anchor], points[candidate], radius) {
                break;
            }
            reach = candidate;
        }
        simplified.push(points[reach]);
        anchor = reach;
    }
    simplified
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::growth::fixtures::design_grid;
    use crate::grid::CellKind;

    #[test]
    fn a_chain_is_a_root_followed_by_children() {
        let mut skeleton = Skeleton::new();
        assert!(skeleton.is_empty());
        let tip = skeleton.add_chain(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        assert_eq!(tip, 2);
        assert_eq!(skeleton.len(), 3);
        assert_eq!(skeleton.segment_count(), 2);
        assert_eq!(skeleton.nodes[0].parent, None);
        assert_eq!(skeleton.nodes[2].parent, Some(1));
        assert!((skeleton.total_length_mm() - 2.0).abs() < 1e-12);

        // A branch off the middle leaves two leaves and no more roots.
        skeleton.add_child(1, [1.0, 1.0, 0.0]);
        let leaves = skeleton.leaves();
        assert_eq!(leaves, vec![false, false, true, true]);
        assert_eq!(
            skeleton.nodes.iter().filter(|n| n.parent.is_none()).count(),
            1
        );
        // Every parent index is smaller than its child's, which the flow pass
        // relies on.
        for (index, node) in skeleton.nodes.iter().enumerate() {
            assert!(node.parent.is_none_or(|p| p < index));
        }
    }

    /// A zigzag corridor: allowed cells only along a staircase, everything else
    /// forbidden. Any shortcut that cuts a corner leaves the corridor.
    fn zigzag() -> crate::grid::Grid {
        let n = 12;
        let mut grid = design_grid(n);
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Void;
        }
        let mut open = |i: usize, j: usize| {
            for k in 0..n {
                let cell = grid.cell_index(i, j, k);
                grid.cells[cell] = CellKind::Design;
            }
        };
        // Alternating runs along i and j, forming a staircase in the xy plane.
        for i in 0..6 {
            open(i, 0);
        }
        for j in 0..6 {
            open(5, j);
        }
        for i in 5..11 {
            open(i, 5);
        }
        grid
    }

    #[test]
    fn resampling_puts_nodes_back_without_leaving_the_line() {
        let line = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 6.0, 0.0]];
        let resampled = resample(&line, 2.0);

        // The corners survive and nothing moved off the polyline.
        assert_eq!(resampled.first(), Some(&line[0]));
        assert_eq!(resampled.last(), Some(&line[2]));
        assert!(resampled.contains(&line[1]));
        assert!(resampled.len() > line.len(), "{resampled:?}");
        for point in &resampled {
            let on_first = point[1] == 0.0 && (0.0..=10.0).contains(&point[0]);
            let on_second = point[0] == 10.0 && (0.0..=6.0).contains(&point[1]);
            assert!(on_first || on_second, "{point:?} left the polyline");
            assert_eq!(point[2], 0.0);
        }
        // No two consecutive nodes are further apart than the spacing.
        for pair in resampled.windows(2) {
            let gap = geometry::length(geometry::difference(pair[1], pair[0]));
            assert!(gap <= 2.0 + 1e-12, "a {gap} mm gap survived resampling");
        }

        // Degenerate inputs come back untouched rather than spinning.
        assert_eq!(resample(&line, 0.0), line.to_vec());
        assert_eq!(resample(&line, f64::NAN), line.to_vec());
        assert_eq!(resample(&line[..1], 2.0), line[..1].to_vec());
        // A spacing longer than the whole polyline keeps only its corners.
        assert_eq!(resample(&line, 100.0), line.to_vec());
    }

    #[test]
    fn the_shortcut_pass_never_leaves_the_corridor() {
        let grid = zigzag();
        let domain = GrowthDomain::new(&grid);
        let path: Vec<Vec3> = [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (6, 5),
            (7, 5),
            (8, 5),
            (9, 5),
            (10, 5),
        ]
        .into_iter()
        .map(|(i, j)| domain.cell_center(grid.cell_index(i, j, 6)))
        .collect();

        for radius in [0.0, 0.4] {
            let simplified = shortcut(&domain, &path, radius);
            assert!(simplified.len() >= 3, "the corners cannot be cut away");
            assert!(simplified.len() < path.len(), "the straight runs must fuse");
            assert_eq!(simplified.first(), path.first());
            assert_eq!(simplified.last(), path.last());
            // Every surviving segment stays inside the corridor.
            for pair in simplified.windows(2) {
                assert!(
                    domain.capsule_inside(pair[0], pair[1], radius),
                    "a shortcut left the allowed cells"
                );
            }
            // And every kept point is one of the originals.
            for point in &simplified {
                assert!(path.contains(point), "the pass invented a point");
            }
        }
    }

    #[test]
    fn the_shortcut_radius_backs_off_until_the_original_path_clears_it() {
        let grid = zigzag();
        let domain = GrowthDomain::new(&grid);
        let path: Vec<Vec3> = (0..6)
            .map(|i| domain.cell_center(grid.cell_index(i, 0, 6)))
            .collect();

        // The corridor is one voxel wide, so a radius of two voxels is
        // impossible: the ladder has to walk down to one the path itself
        // clears, and what comes back has to be such a radius.
        let backed_off = shortcut_radius(&domain, &path, 2.0);
        assert!(backed_off < 2.0, "the ladder never moved: {backed_off}");
        assert!(
            path.windows(2)
                .all(|pair| domain.capsule_inside(pair[0], pair[1], backed_off)),
            "the backed off radius {backed_off} still does not fit the corridor"
        );
        // A radius the corridor can hold survives untouched.
        let fits = shortcut_radius(&domain, &path, 0.4);
        assert!((fits - 0.4).abs() < 1e-12, "radius {fits} was backed off");
        // And a radius no rung of the ladder can reach ends at the bare centre
        // line, which every voxel path satisfies by construction.
        assert_eq!(shortcut_radius(&domain, &path, 16.0), 0.0);
        // A path of one or two points is returned as it is.
        assert_eq!(shortcut(&domain, &path[..1], 0.0), path[..1].to_vec());
        assert_eq!(shortcut(&domain, &path[..2], 0.0), path[..2].to_vec());
    }
}
