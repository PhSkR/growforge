//! Self-weight: the design dependent body force of the structure itself.
//!
//! The weight of an element is `rho_e * rho_material * V * g` newtons, spread
//! equally over its eight nodes along the gravity direction. It scales linearly
//! with the physical density (not with the penalized stiffness), so it has to be
//! reassembled whenever the design changes, and it adds a load term to the
//! compliance sensitivity.
//!
//! Units: see [`constants::TONNE_PER_MM3_PER_G_PER_CM3`] for the dimensional
//! analysis behind the density conversion. `acceleration` is the gravitational
//! acceleration vector in mm/s^2, already carrying its direction.

use rayon::prelude::*;

use crate::constants::{self, DOF_PER_NODE, NODES_PER_ELEMENT};
use crate::geometry::Vec3;
use crate::grid::{CellKind, Grid};

/// Newtons of weight one node picks up per unit of physical density of one
/// adjacent element, per unit of acceleration.
///
/// `t/mm^3 * mm^3 = t`, and one eighth of it lands on each node; multiplying by
/// an acceleration in mm/s^2 gives newtons.
fn nodal_mass(grid: &Grid, density_g_cm3: f64) -> f64 {
    constants::TONNE_PER_MM3_PER_G_PER_CM3 * density_g_cm3 * grid.cell_volume()
        / NODES_PER_ELEMENT as f64
}

/// Physical density of a cell as the model sees it: pinned for void and forced
/// solid cells, from the array for design cells.
fn cell_density(grid: &Grid, densities: &[f64], cell: usize) -> f64 {
    match grid.cells[cell] {
        CellKind::Void => constants::DENSITY_MIN,
        CellKind::Solid => constants::DENSITY_MAX,
        CellKind::Design => densities[cell],
    }
}

/// Sum of the physical densities of the (at most eight) cells around a node.
fn adjacent_density_sum(grid: &Grid, densities: &[f64], node: usize) -> f64 {
    let (i, j, k) = grid.node_ijk(node);
    let mut sum = 0.0;
    for dk in 0..2 {
        for dj in 0..2 {
            for di in 0..2 {
                let (ci, cj, ck) = (
                    i as isize + di - 1,
                    j as isize + dj - 1,
                    k as isize + dk - 1,
                );
                if ci < 0
                    || cj < 0
                    || ck < 0
                    || ci >= grid.nx as isize
                    || cj >= grid.ny as isize
                    || ck >= grid.nz as isize
                {
                    continue;
                }
                let cell = grid.cell_index(ci as usize, cj as usize, ck as usize);
                sum += cell_density(grid, densities, cell);
            }
        }
    }
    sum
}

/// Add the self weight of `densities` to a nodal force vector, in newtons.
///
/// The scatter is written as a gather over nodes, so it parallelises without
/// the colouring the stiffness operator needs.
pub fn accumulate(
    grid: &Grid,
    densities: &[f64],
    acceleration: Vec3,
    density_g_cm3: f64,
    forces: &mut [f64],
) {
    debug_assert_eq!(densities.len(), grid.n_cells());
    debug_assert_eq!(forces.len(), grid.n_dof());
    let mass = nodal_mass(grid, density_g_cm3);
    forces
        .par_chunks_mut(DOF_PER_NODE)
        .enumerate()
        .for_each(|(node, slot)| {
            let share = mass * adjacent_density_sum(grid, densities, node);
            for (d, value) in slot.iter_mut().enumerate() {
                *value += share * acceleration[d];
            }
        });
}

/// Accumulate the load term `weight * 2 u^T df/drho_e` of the compliance
/// sensitivity, element by element.
///
/// This is the part of `dC/dx_e = 2 u^T (df/dx_e) - u^T (dK/dx_e) u` that a
/// design independent load does not have. It is the term that can push a
/// sensitivity positive.
pub fn accumulate_sensitivity(
    grid: &Grid,
    displacements: &[f64],
    acceleration: Vec3,
    density_g_cm3: f64,
    weight: f64,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), grid.n_cells());
    debug_assert_eq!(displacements.len(), grid.n_dof());
    let factor = 2.0 * weight * nodal_mass(grid, density_g_cm3);
    out.par_iter_mut().enumerate().for_each(|(cell, slot)| {
        let (i, j, k) = grid.cell_ijk(cell);
        let mut work = 0.0;
        for node in grid.element_nodes(i, j, k) {
            let base = node * DOF_PER_NODE;
            for (d, a) in acceleration.iter().enumerate() {
                work += a * displacements[base + d];
            }
        }
        *slot += factor * work;
    });
}

/// Total self weight of a density field in newtons, for reports.
pub fn total_weight_n(
    grid: &Grid,
    densities: &[f64],
    acceleration: Vec3,
    density_g_cm3: f64,
) -> f64 {
    let mass = constants::TONNE_PER_MM3_PER_G_PER_CM3 * density_g_cm3 * grid.cell_volume();
    let total: f64 = (0..grid.n_cells())
        .into_par_iter()
        .map(|cell| cell_density(grid, densities, cell))
        .sum();
    mass * total * crate::geometry::length(acceleration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;

    fn single_element(h: f64) -> Grid {
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [h, h, h],
            },
            h,
        );
        grid.cells[0] = CellKind::Design;
        grid
    }

    #[test]
    fn the_density_conversion_matches_its_dimensional_analysis() {
        // One cubic centimetre of PLA weighs rho * V * g:
        // 1.24 g = 1.24e-3 kg, times 9.81 m/s^2, is 12.16 mN.
        let grid = single_element(10.0);
        let densities = vec![1.0];
        let weight = total_weight_n(&grid, &densities, [0.0, 0.0, -9810.0], 1.24);
        let expected = 1.24e-3 * 9.81;
        assert!(
            (weight - expected).abs() < 1e-12,
            "self weight {weight} N is not {expected} N"
        );
        assert!((constants::TONNE_PER_MM3_PER_G_PER_CM3 - 1e-9).abs() < 1e-24);
    }

    #[test]
    fn a_single_element_splits_its_weight_over_eight_nodes() {
        let h = 4.0;
        let grid = single_element(h);
        let density_g_cm3 = 2.5;
        let acceleration = [0.0, 0.0, -9810.0];
        let densities = vec![1.0];
        let mut forces = vec![0.0; grid.n_dof()];
        accumulate(&grid, &densities, acceleration, density_g_cm3, &mut forces);

        let expected =
            constants::TONNE_PER_MM3_PER_G_PER_CM3 * density_g_cm3 * h.powi(3) * acceleration[2];
        let total: f64 = forces.iter().skip(2).step_by(DOF_PER_NODE).sum();
        assert!(
            (total - expected).abs() < 1e-15 * expected.abs(),
            "total z force {total} is not {expected}"
        );
        // Nothing sideways, and every node carries exactly one eighth.
        for node in 0..grid.n_nodes() {
            let base = node * DOF_PER_NODE;
            assert_eq!(forces[base], 0.0);
            assert_eq!(forces[base + 1], 0.0);
            assert!((forces[base + 2] - expected / 8.0).abs() < 1e-18);
        }
        assert!(expected < 0.0, "gravity must pull downwards");
    }

    #[test]
    fn weight_scales_with_density_and_stays_on_the_direction() {
        let grid = single_element(2.0);
        let acceleration = [3.0, -4.0, 0.0];
        let mut half = vec![0.0; grid.n_dof()];
        accumulate(&grid, &[0.5], acceleration, 1.0, &mut half);
        let mut full = vec![0.0; grid.n_dof()];
        accumulate(&grid, &[1.0], acceleration, 1.0, &mut full);
        for (a, b) in half.iter().zip(full.iter()) {
            assert!((2.0 * a - b).abs() < 1e-18);
        }
        // The nodal force is parallel to the acceleration.
        assert!((full[0] / full[1] - acceleration[0] / acceleration[1]).abs() < 1e-12);
        assert_eq!(full[2], 0.0);
    }

    #[test]
    fn void_cells_are_weightless_and_solid_cells_are_full_weight() {
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [3.0, 1.0, 1.0],
            },
            1.0,
        );
        grid.cells[0] = CellKind::Void;
        grid.cells[1] = CellKind::Solid;
        grid.cells[2] = CellKind::Design;
        // The array claims full density everywhere; only the design cell may
        // believe it.
        let densities = vec![1.0; 3];
        let total = total_weight_n(&grid, &densities, [0.0, 0.0, -1000.0], 1.0);
        let per_cell = constants::TONNE_PER_MM3_PER_G_PER_CM3 * 1.0 * 1.0 * 1000.0;
        assert!((total - 2.0 * per_cell).abs() < 1e-18);
    }

    #[test]
    fn the_sensitivity_is_twice_the_work_of_the_body_force() {
        let h = 2.0;
        let grid = single_element(h);
        let acceleration = [0.0, 0.0, -9810.0];
        let density_g_cm3 = 1.2;
        // A uniform downward displacement of every node.
        let mut u = vec![0.0; grid.n_dof()];
        for node in 0..grid.n_nodes() {
            u[node * DOF_PER_NODE + 2] = -0.5;
        }
        let mut sensitivity = vec![0.0; grid.n_cells()];
        accumulate_sensitivity(
            &grid,
            &u,
            acceleration,
            density_g_cm3,
            1.0,
            &mut sensitivity,
        );

        // df/drho is the element weight spread over its nodes, so
        // 2 u^T df/drho is 2 * (total weight) * (-0.5).
        let total =
            constants::TONNE_PER_MM3_PER_G_PER_CM3 * density_g_cm3 * h.powi(3) * acceleration[2];
        let expected = 2.0 * total * -0.5;
        assert!(
            (sensitivity[0] - expected).abs() < 1e-15 * expected.abs(),
            "sensitivity {} is not {expected}",
            sensitivity[0]
        );
        assert!(
            expected > 0.0,
            "self weight can make a sensitivity positive"
        );
    }
}
