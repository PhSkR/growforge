//! Shortest voxel path from a load region to a support region.
//!
//! A* over the allowed cells with the 26 neighbours of a cell and Euclidean
//! edge costs (`h`, `h*sqrt(2)`, `h*sqrt(3)`), so a diagonal step is never
//! cheaper than the distance it covers and the path is a geometric shortest
//! path rather than a Manhattan one.
//!
//! The heuristic is the distance from a cell centre to the bounding box of the
//! target cells. Distance to a box is never longer than the distance to any
//! point inside it, so the heuristic never over-estimates and the first time the
//! search settles a target it has settled the optimum.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::engine::growth::domain::GrowthDomain;
use crate::geometry::{Aabb, Vec3};

/// A queue entry ordered by `f = g + heuristic`, smallest first.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    estimate: f64,
    cell: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed, because `BinaryHeap` is a max-heap. Ties break on the cell
        // index so the queue order, and therefore the path, is deterministic.
        other
            .estimate
            .total_cmp(&self.estimate)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Distance from `p` to the closest point of `box`, zero inside it.
fn distance_to_box(p: Vec3, bounds: &Aabb) -> f64 {
    let sum: f64 = p
        .iter()
        .zip(bounds.min.iter().zip(bounds.max.iter()))
        .map(|(component, (min, max))| {
            let gap = (min - component).max(component - max).max(0.0);
            gap * gap
        })
        .sum();
    sum.sqrt()
}

/// The 26 neighbour offsets of a cell with their step lengths in voxels.
fn neighbour_offsets() -> Vec<(i64, i64, i64, f64)> {
    let mut offsets = Vec::with_capacity(26);
    for dk in -1i64..=1 {
        for dj in -1i64..=1 {
            for di in -1i64..=1 {
                if (di, dj, dk) == (0, 0, 0) {
                    continue;
                }
                let squares = (di * di + dj * dj + dk * dk) as f64;
                offsets.push((di, dj, dk, squares.sqrt()));
            }
        }
    }
    offsets
}

/// Shortest path through allowed cells from any cell of `from` to any cell of
/// `to`, as a list of cell indices starting at a `from` cell.
///
/// Returns `None` when no such path exists.
pub fn shortest_path(domain: &GrowthDomain, from: &[usize], to: &[usize]) -> Option<Vec<usize>> {
    let grid = domain.grid();
    let n = grid.n_cells();
    if from.is_empty() || to.is_empty() {
        return None;
    }

    let mut is_target = vec![false; n];
    let mut bounds = Aabb::empty();
    for &cell in to {
        if !domain.allows_cell(cell) {
            continue;
        }
        is_target[cell] = true;
        let centre = domain.cell_center(cell);
        let half = 0.5 * grid.h;
        bounds = bounds.union(&Aabb {
            min: [centre[0] - half, centre[1] - half, centre[2] - half],
            max: [centre[0] + half, centre[1] + half, centre[2] + half],
        });
    }
    if bounds.is_empty() {
        return None;
    }

    let mut cost = vec![f64::INFINITY; n];
    let mut came_from = vec![usize::MAX; n];
    let mut settled = vec![false; n];
    let mut queue: BinaryHeap<Candidate> = BinaryHeap::new();

    // Multi-source: every cell of the load region starts the search at zero
    // cost, so what comes out is the shortest path between the two regions.
    let mut sources: Vec<usize> = from
        .iter()
        .copied()
        .filter(|&c| domain.allows_cell(c))
        .collect();
    sources.sort_unstable();
    sources.dedup();
    for &cell in &sources {
        cost[cell] = 0.0;
        queue.push(Candidate {
            estimate: distance_to_box(domain.cell_center(cell), &bounds),
            cell,
        });
    }

    let offsets = neighbour_offsets();
    let counts = [grid.nx as i64, grid.ny as i64, grid.nz as i64];
    while let Some(Candidate { cell, .. }) = queue.pop() {
        if settled[cell] {
            continue;
        }
        settled[cell] = true;
        if is_target[cell] {
            let mut path = vec![cell];
            let mut walk = cell;
            while came_from[walk] != usize::MAX {
                walk = came_from[walk];
                path.push(walk);
            }
            path.reverse();
            return Some(path);
        }
        let (i, j, k) = grid.cell_ijk(cell);
        let base = [i as i64, j as i64, k as i64];
        for (di, dj, dk, length) in &offsets {
            let next = [base[0] + di, base[1] + dj, base[2] + dk];
            if (0..3).any(|d| next[d] < 0 || next[d] >= counts[d]) {
                continue;
            }
            let neighbour = grid.cell_index(next[0] as usize, next[1] as usize, next[2] as usize);
            if settled[neighbour] || !domain.allows_cell(neighbour) {
                continue;
            }
            let candidate = cost[cell] + length * grid.h;
            if candidate < cost[neighbour] {
                cost[neighbour] = candidate;
                came_from[neighbour] = cell;
                queue.push(Candidate {
                    estimate: candidate + distance_to_box(domain.cell_center(neighbour), &bounds),
                    cell: neighbour,
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::growth::fixtures::design_grid;
    use crate::grid::CellKind;

    /// A block with a wall at `i = 4` pierced by a single hole.
    fn walled(n: usize, hole: Option<(usize, usize)>) -> crate::grid::Grid {
        let mut grid = design_grid(n);
        for k in 0..n {
            for j in 0..n {
                let cell = grid.cell_index(4, j, k);
                grid.cells[cell] = CellKind::Void;
            }
        }
        if let Some((j, k)) = hole {
            let cell = grid.cell_index(4, j, k);
            grid.cells[cell] = CellKind::Design;
        }
        grid
    }

    #[test]
    fn a_path_routes_through_the_only_gap_in_a_wall() {
        let n = 9;
        let grid = walled(n, Some((7, 1)));
        let domain = GrowthDomain::new(&grid);
        let from = vec![grid.cell_index(0, 0, 0)];
        let to = vec![grid.cell_index(8, 0, 0)];
        let path = shortest_path(&domain, &from, &to).expect("a path through the gap");

        assert_eq!(path.first(), Some(&from[0]));
        assert_eq!(path.last(), Some(&to[0]));
        // It has to go through the hole, and every cell on it is allowed.
        assert!(path.contains(&grid.cell_index(4, 7, 1)));
        for &cell in &path {
            assert!(
                domain.allows_cell(cell),
                "the path entered a forbidden cell"
            );
        }
        // Consecutive cells are 26-neighbours.
        for pair in path.windows(2) {
            let (a, b) = (grid.cell_ijk(pair[0]), grid.cell_ijk(pair[1]));
            let step = [a.0.abs_diff(b.0), a.1.abs_diff(b.1), a.2.abs_diff(b.2)];
            assert!(step.iter().all(|s| *s <= 1) && step.iter().sum::<usize>() > 0);
        }
    }

    #[test]
    fn a_walled_off_target_has_no_path_at_all() {
        let grid = walled(9, None);
        let domain = GrowthDomain::new(&grid);
        assert!(
            shortest_path(
                &domain,
                &[grid.cell_index(0, 0, 0)],
                &[grid.cell_index(8, 0, 0)]
            )
            .is_none()
        );
    }

    #[test]
    fn the_path_is_a_geometric_shortest_path() {
        // An open block: the diagonal has to come out as the straight line, not
        // as a staircase, which is what the 26 neighbourhood and the Euclidean
        // edge costs buy.
        let grid = design_grid(9);
        let domain = GrowthDomain::new(&grid);
        let start = grid.cell_index(0, 0, 0);
        let end = grid.cell_index(8, 8, 8);
        let path = shortest_path(&domain, &[start], &[end]).expect("a path");
        assert_eq!(path.len(), 9, "the diagonal is eight steps: {}", path.len());

        let length: f64 = path
            .windows(2)
            .map(|pair| {
                let a = domain.cell_center(pair[0]);
                let b = domain.cell_center(pair[1]);
                crate::geometry::length(crate::geometry::difference(b, a))
            })
            .sum();
        let straight = crate::geometry::length(crate::geometry::difference(
            domain.cell_center(end),
            domain.cell_center(start),
        ));
        assert!(
            (length - straight).abs() < 1e-9,
            "path length {length} is not the straight line {straight}"
        );
    }

    #[test]
    fn empty_and_forbidden_endpoints_are_refused() {
        let mut grid = design_grid(4);
        let blocked = grid.cell_index(3, 3, 3);
        grid.cells[blocked] = CellKind::Void;
        let domain = GrowthDomain::new(&grid);
        assert!(shortest_path(&domain, &[], &[0]).is_none());
        assert!(shortest_path(&domain, &[0], &[]).is_none());
        assert!(shortest_path(&domain, &[0], &[blocked]).is_none());
        // A target that is also a source is a one cell path.
        assert_eq!(shortest_path(&domain, &[0], &[0]), Some(vec![0]));
    }
}
