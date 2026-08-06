//! Topology of the density field the surface is about to be extracted from:
//! enclosed cavities in the empty space, and disconnected bodies in the
//! material. Both are flood fills over the same thresholded field.
//!
//! A cavity is a connected group of below-threshold cells that cannot reach the
//! grid boundary. Everything the mesh pipeline sees is padded with a layer of
//! empty space, so "reaches the boundary" is exactly "is connected to the
//! outside world" and an unreachable group comes out of marching cubes as a
//! sealed internal void: unprintable without trapped support, and unnoticeable
//! in a slicer preview.
//!
//! A *body* is a connected group of at-or-above-threshold cells. growforge
//! designs one part, so anything past the first body is material that is not
//! joined to the rest of the part: it prints as a separate loose object, and it
//! is load bearing in no sense at all. Neither the mesh validation nor the
//! cavity pass can see it - a floating chunk is perfectly watertight, manifold,
//! consistently wound and encloses no cavity - which is exactly why it is
//! checked for here.
//!
//! Connectivity is face connectivity (six neighbours) for both. A cavity joined
//! to the outside only through a shared edge or corner therefore counts as
//! enclosed, and two bodies joined only across an edge or a corner count as two;
//! both readings are the conservative one, because an escape path one cell wide
//! diagonally is not a drain hole and a diagonal touch is not a joint.
//!
//! Everything here describes the **density field**, and the cavity policy acts
//! on it before marching cubes runs, so the cavity report describes the surface
//! that was written. The body count does not: the surface is extracted from the
//! node averages of this field, and the two disagree in both directions - a lone
//! dense cell is a body here and makes no surface at all, and a clump joined to
//! the part only through a one-cell-wide bridge is one body here and two shells
//! there. [`crate::mesh::islands`] is the reading of the mesh itself, and the
//! run prints both, each naming its own object.

use rayon::prelude::*;

use crate::config::VoidPolicy;
use crate::constants;
use crate::engine::cell_density;
use crate::geometry::Vec3;
use crate::grid::{CellKind, Grid};

/// One cavity the exported surface would fully enclose.
#[derive(Debug, Clone, PartialEq)]
pub struct EnclosedVoid {
    /// Number of cells in the cavity.
    pub cells: usize,
    /// Volume in cubic millimetres.
    pub volume_mm3: f64,
    /// Centroid of the cavity's cells, in millimetres.
    pub centroid: Vec3,
    /// True when every cell of the cavity may be turned into material. A cavity
    /// that overlaps a keepout, or the space outside the domain, may not be:
    /// filling it would break a constraint the user asked for.
    pub fillable: bool,
}

/// What the enclosed cavity pass found and did.
#[derive(Debug, Clone, PartialEq)]
pub struct VoidReport {
    /// The policy that was in force.
    pub policy: VoidPolicy,
    /// Every cavity found, largest first.
    pub voids: Vec<EnclosedVoid>,
    /// Cells that were turned into material.
    pub filled_cells: usize,
    /// Volume that was turned into material, in cubic millimetres.
    pub filled_volume_mm3: f64,
    /// Mass that filling added, in grams.
    pub added_mass_g: f64,
}

impl VoidReport {
    /// Cavities that remain in the exported surface.
    pub fn remaining(&self) -> usize {
        self.voids.iter().filter(|v| !self.was_filled(v)).count()
    }

    /// True when this cavity was filled rather than exported.
    pub fn was_filled(&self, cavity: &EnclosedVoid) -> bool {
        self.policy == VoidPolicy::Fill && cavity.fillable
    }
}

/// True when a cell touches the boundary of the grid block.
fn on_boundary(grid: &Grid, cell: usize) -> bool {
    let (i, j, k) = grid.cell_ijk(cell);
    i == 0 || j == 0 || k == 0 || i + 1 == grid.nx || j + 1 == grid.ny || k + 1 == grid.nz
}

/// The six face neighbours of a cell.
fn neighbours(grid: &Grid, cell: usize) -> impl Iterator<Item = usize> + '_ {
    let (i, j, k) = grid.cell_ijk(cell);
    [
        (-1i32, 0i32, 0i32),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]
    .into_iter()
    .filter_map(move |(di, dj, dk)| {
        let (ni, nj, nk) = (i as i32 + di, j as i32 + dj, k as i32 + dk);
        if ni < 0
            || nj < 0
            || nk < 0
            || ni >= grid.nx as i32
            || nj >= grid.ny as i32
            || nk >= grid.nz as i32
        {
            None
        } else {
            Some(grid.cell_index(ni as usize, nj as usize, nk as usize))
        }
    })
}

/// Group the below-threshold cells that the grid boundary cannot reach.
///
/// Returns one cell list per cavity, in scan order.
fn enclosed_groups(grid: &Grid, densities: &[f64], iso: f64) -> Vec<Vec<usize>> {
    let n = grid.n_cells();
    let empty: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|e| cell_density(grid, densities, e) < iso)
        .collect();

    // Everything the outside world can reach through empty cells.
    let mut reached = vec![false; n];
    let mut stack: Vec<usize> = (0..n)
        .filter(|&e| empty[e] && on_boundary(grid, e))
        .collect();
    for &e in &stack {
        reached[e] = true;
    }
    while let Some(cell) = stack.pop() {
        for next in neighbours(grid, cell) {
            if empty[next] && !reached[next] {
                reached[next] = true;
                stack.push(next);
            }
        }
    }

    // What is left forms the cavities.
    let mut seen = vec![false; n];
    let mut groups = Vec::new();
    for seed in 0..n {
        if !empty[seed] || reached[seed] || seen[seed] {
            continue;
        }
        seen[seed] = true;
        let mut group = vec![seed];
        let mut queue = vec![seed];
        while let Some(cell) = queue.pop() {
            for next in neighbours(grid, cell) {
                if empty[next] && !reached[next] && !seen[next] {
                    seen[next] = true;
                    group.push(next);
                    queue.push(next);
                }
            }
        }
        groups.push(group);
    }
    groups
}

/// Statistics of one cavity.
fn describe(grid: &Grid, group: &[usize]) -> EnclosedVoid {
    let mut centroid = [0.0; 3];
    for &cell in group {
        let (i, j, k) = grid.cell_ijk(cell);
        let center = grid.cell_center(i, j, k);
        for (slot, value) in centroid.iter_mut().zip(center.iter()) {
            *slot += value;
        }
    }
    let count = group.len() as f64;
    for slot in centroid.iter_mut() {
        *slot /= count;
    }
    EnclosedVoid {
        cells: group.len(),
        volume_mm3: count * grid.cell_volume(),
        centroid,
        fillable: group.iter().all(|&e| grid.cells[e] != CellKind::Void),
    }
}

/// One connected body of material in the field the surface came from.
#[derive(Debug, Clone, PartialEq)]
pub struct SolidBody {
    /// Number of cells in the body.
    pub cells: usize,
    /// Volume in cubic millimetres.
    pub volume_mm3: f64,
    /// Centroid of the body's cells, in millimetres.
    pub centroid: Vec3,
}

/// Which connected body of material every cell belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BodyLabels {
    /// Body index of every cell in grid element order, `None` where the cell
    /// holds no material at the threshold the labelling used.
    pub of_cell: Vec<Option<usize>>,
    /// Number of bodies found. Labels run from zero to one below it, in the
    /// scan order of their first cell.
    pub bodies: usize,
}

/// Label every cell with the connected body of material it belongs to.
///
/// The flood fill both [`solid_bodies`] and the load path pre-check of
/// [`crate::stress::broken_load_paths`] are built on. `threshold` is the
/// physical density a cell needs to count as material at all, which is the iso
/// level for the exported bodies and something far lower for "is there any
/// stiffness on this path".
pub fn body_labels(grid: &Grid, densities: &[f64], threshold: f64) -> BodyLabels {
    let n = grid.n_cells();
    let material: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|e| cell_density(grid, densities, e) >= threshold)
        .collect();

    let mut of_cell: Vec<Option<usize>> = vec![None; n];
    let mut bodies = 0;
    for seed in 0..n {
        if !material[seed] || of_cell[seed].is_some() {
            continue;
        }
        let label = bodies;
        bodies += 1;
        of_cell[seed] = Some(label);
        let mut queue = vec![seed];
        while let Some(cell) = queue.pop() {
            for next in neighbours(grid, cell) {
                if material[next] && of_cell[next].is_none() {
                    of_cell[next] = Some(label);
                    queue.push(next);
                }
            }
        }
    }
    BodyLabels { of_cell, bodies }
}

/// Every connected body of material at `iso`, largest first.
///
/// One body is a part. Two or more means the field carries at least one body
/// that is joined to none of the others, which a slicer will happily lay down as
/// a separate loose object. It is invisible to every other check on the field,
/// so it is reported rather than assumed away.
///
/// What it is *not* is a verdict on the file: what ships is the mesh, whose own
/// components [`crate::mesh::islands`] counts and judges by the purposes the
/// configuration declared. A second body here may be debris, and it may equally
/// be a keepin boss that was asked for; the two are told apart there.
///
/// The same flood fill as the cavity pass, over the complementary set of cells.
pub fn solid_bodies(grid: &Grid, densities: &[f64], iso: f64) -> Vec<SolidBody> {
    let labels = body_labels(grid, densities, iso);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); labels.bodies];
    for (cell, label) in labels.of_cell.iter().enumerate() {
        if let Some(label) = label {
            groups[*label].push(cell);
        }
    }
    let mut bodies: Vec<SolidBody> = groups
        .iter()
        .map(|group| {
            let described = describe(grid, group);
            SolidBody {
                cells: described.cells,
                volume_mm3: described.volume_mm3,
                centroid: described.centroid,
            }
        })
        .collect();
    bodies.sort_by(|a, b| b.volume_mm3.total_cmp(&a.volume_mm3));
    bodies
}

/// Find every cavity the surface at `iso` would enclose, largest first.
pub fn detect(grid: &Grid, densities: &[f64], iso: f64) -> Vec<EnclosedVoid> {
    let mut voids: Vec<EnclosedVoid> = enclosed_groups(grid, densities, iso)
        .iter()
        .map(|group| describe(grid, group))
        .collect();
    voids.sort_by(|a, b| b.volume_mm3.total_cmp(&a.volume_mm3));
    voids
}

/// Apply the configured policy to `densities`, reporting what was found either
/// way.
///
/// Under [`VoidPolicy::Fill`] every fillable cavity is turned into solid
/// material before the surface is extracted, so the exported STL has no trapped
/// cavity left. A cavity that overlaps a keepout, or the space outside the
/// domain, is never filled: the constraint wins, and the report says the cavity
/// is still there.
pub fn resolve(
    grid: &Grid,
    densities: &mut [f64],
    iso: f64,
    policy: VoidPolicy,
    density_g_cm3: f64,
) -> VoidReport {
    let groups = enclosed_groups(grid, densities, iso);
    let mut voids: Vec<EnclosedVoid> = groups.iter().map(|group| describe(grid, group)).collect();

    let mut filled_cells = 0;
    if policy == VoidPolicy::Fill {
        for (group, cavity) in groups.iter().zip(voids.iter()) {
            if !cavity.fillable {
                continue;
            }
            for &cell in group {
                densities[cell] = constants::DENSITY_MAX;
            }
            filled_cells += group.len();
        }
    }

    voids.sort_by(|a, b| b.volume_mm3.total_cmp(&a.volume_mm3));
    let filled_volume_mm3 = filled_cells as f64 * grid.cell_volume();
    VoidReport {
        policy,
        voids,
        filled_cells,
        filled_volume_mm3,
        added_mass_g: filled_volume_mm3 * density_g_cm3 / constants::MM3_PER_CM3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;

    /// A cube of design cells, `n` on a side.
    fn design_grid(n: usize) -> Grid {
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [n as f64, n as f64, n as f64],
            },
            1.0,
        );
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Design;
        }
        grid
    }

    /// A solid block with a hollow core of side `hole`, centred.
    fn hollow(n: usize, hole: usize) -> (Grid, Vec<f64>) {
        let grid = design_grid(n);
        let mut densities = vec![1.0; grid.n_cells()];
        let lo = (n - hole) / 2;
        for k in lo..lo + hole {
            for j in lo..lo + hole {
                for i in lo..lo + hole {
                    densities[grid.cell_index(i, j, k)] = 0.0;
                }
            }
        }
        (grid, densities)
    }

    #[test]
    fn a_hollow_cube_has_exactly_one_enclosed_void() {
        let (grid, densities) = hollow(7, 3);
        let voids = detect(&grid, &densities, 0.5);
        assert_eq!(voids.len(), 1);
        assert_eq!(voids[0].cells, 27);
        assert!((voids[0].volume_mm3 - 27.0).abs() < 1e-12);
        // The hole is centred on the block, so its centroid is the centre.
        for component in voids[0].centroid {
            assert!((component - 3.5).abs() < 1e-12, "centroid {component}");
        }
        assert!(voids[0].fillable);
    }

    #[test]
    fn an_open_channel_encloses_nothing() {
        let (grid, mut densities) = hollow(7, 3);
        // Drill the cavity out to the face along +x.
        for i in 5..7 {
            densities[grid.cell_index(i, 3, 3)] = 0.0;
        }
        assert!(detect(&grid, &densities, 0.5).is_empty());
    }

    #[test]
    fn a_diagonal_leak_is_not_an_escape_path() {
        let (grid, mut densities) = hollow(7, 3);
        // An empty channel out to the +x face whose inner end only shares an
        // edge with the cavity at (4, 4, 3). With face connectivity that is not
        // a drain, which is the conservative reading.
        densities[grid.cell_index(5, 5, 3)] = 0.0;
        densities[grid.cell_index(6, 5, 3)] = 0.0;
        let voids = detect(&grid, &densities, 0.5);
        assert_eq!(voids.len(), 1, "the channel drained the cavity");
        assert_eq!(voids[0].cells, 27);
    }

    #[test]
    fn fill_mode_fills_exactly_the_enclosed_cells() {
        let (grid, densities) = hollow(7, 3);
        let before = densities.clone();
        let mut filled = densities;
        let report = resolve(&grid, &mut filled, 0.5, VoidPolicy::Fill, 2.0);
        assert_eq!(report.filled_cells, 27);
        assert!((report.filled_volume_mm3 - 27.0).abs() < 1e-12);
        // 27 mm3 at 2 g/cm3 is 54 mg.
        assert!((report.added_mass_g - 0.054).abs() < 1e-12);
        assert_eq!(report.remaining(), 0);

        let changed: Vec<usize> = (0..grid.n_cells())
            .filter(|&e| before[e] != filled[e])
            .collect();
        assert_eq!(changed.len(), 27);
        for e in changed {
            assert_eq!(filled[e], 1.0);
            assert_eq!(before[e], 0.0);
        }
        // Filling closed the cavity for good.
        assert!(detect(&grid, &filled, 0.5).is_empty());
    }

    #[test]
    fn warn_mode_reports_without_touching_the_field() {
        let (grid, densities) = hollow(7, 3);
        let before = densities.clone();
        let mut kept = densities;
        let report = resolve(&grid, &mut kept, 0.5, VoidPolicy::Warn, 2.0);
        assert_eq!(kept, before);
        assert_eq!(report.voids.len(), 1);
        assert_eq!(report.filled_cells, 0);
        assert_eq!(report.added_mass_g, 0.0);
        assert_eq!(report.remaining(), 1);
    }

    #[test]
    fn a_cavity_inside_a_keepout_is_reported_but_never_filled() {
        let (mut grid, densities) = hollow(7, 3);
        // Declare the hole a keepout, as a deliberate internal cavity would be.
        for k in 2..5 {
            for j in 2..5 {
                for i in 2..5 {
                    let cell = grid.cell_index(i, j, k);
                    grid.cells[cell] = CellKind::Void;
                }
            }
        }
        let before = densities.clone();
        let mut field = densities;
        let report = resolve(&grid, &mut field, 0.5, VoidPolicy::Fill, 1.0);
        assert_eq!(report.voids.len(), 1);
        assert!(!report.voids[0].fillable);
        assert_eq!(report.filled_cells, 0);
        assert_eq!(field, before, "a keepout must survive the fill policy");
        assert_eq!(report.remaining(), 1);
    }

    #[test]
    fn two_separate_cavities_are_reported_largest_first() {
        let grid = design_grid(9);
        let mut densities = vec![1.0; grid.n_cells()];
        densities[grid.cell_index(2, 2, 2)] = 0.0;
        for k in 5..7 {
            for j in 5..7 {
                for i in 5..7 {
                    densities[grid.cell_index(i, j, k)] = 0.0;
                }
            }
        }
        let voids = detect(&grid, &densities, 0.5);
        assert_eq!(voids.len(), 2);
        assert_eq!(voids[0].cells, 8);
        assert_eq!(voids[1].cells, 1);
    }

    #[test]
    fn a_solid_block_and_an_empty_block_both_enclose_nothing() {
        let grid = design_grid(5);
        assert!(detect(&grid, &vec![1.0; grid.n_cells()], 0.5).is_empty());
        assert!(detect(&grid, &vec![0.0; grid.n_cells()], 0.5).is_empty());
    }

    #[test]
    fn one_part_is_one_body_and_an_empty_field_is_none() {
        let grid = design_grid(5);
        let whole = solid_bodies(&grid, &vec![1.0; grid.n_cells()], 0.5);
        assert_eq!(whole.len(), 1);
        assert_eq!(whole[0].cells, grid.n_cells());
        assert!((whole[0].volume_mm3 - grid.n_cells() as f64).abs() < 1e-12);
        for component in whole[0].centroid {
            assert!((component - 2.5).abs() < 1e-12, "centroid {component}");
        }
        assert!(solid_bodies(&grid, &vec![0.0; grid.n_cells()], 0.5).is_empty());
    }

    #[test]
    fn a_floating_island_is_a_second_body_and_is_reported_after_the_part() {
        let grid = design_grid(9);
        let mut densities = vec![0.0; grid.n_cells()];
        // A bar of eight cells, and a single cell floating clear of it.
        for i in 0..8 {
            densities[grid.cell_index(i, 0, 0)] = 1.0;
        }
        let island = grid.cell_index(4, 4, 4);
        densities[island] = 1.0;

        let bodies = solid_bodies(&grid, &densities, 0.5);
        assert_eq!(bodies.len(), 2, "the island has to come out separate");
        assert_eq!(bodies[0].cells, 8, "largest first");
        assert_eq!(bodies[1].cells, 1);
        assert_eq!(bodies[1].centroid, [4.5, 4.5, 4.5]);

        // Nothing else in the pipeline sees this: the island encloses no cavity.
        assert!(detect(&grid, &densities, 0.5).is_empty());
    }

    #[test]
    fn a_diagonal_touch_is_not_a_joint() {
        // Two cells sharing only a corner are two bodies, which is the same
        // conservative reading of connectivity the cavity pass uses.
        let grid = design_grid(4);
        let mut densities = vec![0.0; grid.n_cells()];
        densities[grid.cell_index(1, 1, 1)] = 1.0;
        densities[grid.cell_index(2, 2, 2)] = 1.0;
        assert_eq!(solid_bodies(&grid, &densities, 0.5).len(), 2);

        // Sharing a face makes them one.
        densities[grid.cell_index(2, 2, 2)] = 0.0;
        densities[grid.cell_index(2, 1, 1)] = 1.0;
        assert_eq!(solid_bodies(&grid, &densities, 0.5).len(), 1);
    }

    #[test]
    fn void_and_solid_cells_count_as_the_classifier_says_whatever_the_field_holds() {
        // A keepout cell is never material and a keepin cell always is, exactly
        // as `cell_density` pins them for every other stage.
        let mut grid = design_grid(5);
        let forced_void = grid.cell_index(0, 0, 0);
        let forced_solid = grid.cell_index(4, 4, 4);
        grid.cells[forced_void] = CellKind::Void;
        grid.cells[forced_solid] = CellKind::Solid;

        let mut densities = vec![0.0; grid.n_cells()];
        densities[forced_void] = 1.0;
        let bodies = solid_bodies(&grid, &densities, 0.5);
        assert_eq!(bodies.len(), 1, "only the keepin cell counts");
        assert_eq!(bodies[0].cells, 1);
        assert_eq!(bodies[0].centroid, [4.5, 4.5, 4.5]);
    }
}
