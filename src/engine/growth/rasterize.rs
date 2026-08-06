//! Turning the skeleton into a density field.
//!
//! Every segment is a capsule: the set of points within its radius of the
//! segment between two nodes. The capsules are unioned with a polynomial smooth
//! minimum, which rounds the junctions into fillets instead of leaving a crease
//! a printer would have to bridge, and the resulting signed distance is mapped
//! through a smoothstep one voxel wide into a density in `[0, 1]`.
//!
//! Void cells stay at zero and forced solid cells at one whatever the skeleton
//! does: a keepout is a constraint, not a suggestion, and a keepin pad is
//! material the user already asked for.
//!
//! The fold is parallelised over slabs of constant `k` rather than over
//! segments, because the smooth minimum is not associative: every cell has to
//! see the segments in the same order for the field to be reproducible, and a
//! slab owns its cells exclusively.

use rayon::prelude::*;

use crate::constants;
use crate::engine::growth::skeleton::Skeleton;
use crate::geometry::{self, Vec3};
use crate::grid::{CellKind, Grid};

/// Polynomial smooth minimum with blend width `k`.
///
/// `smin(a, b, k) <= min(a, b)`, with equality once the two are more than `k`
/// apart, so unioning two distant struts is the plain union and unioning two
/// that meet rounds the joint over a width of `k`.
pub fn smooth_min(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    // A non-finite input (the +inf the field starts at) leaves `overlap` at
    // zero through `max`, which ignores NaN, so this degenerates to `min`.
    let overlap = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - overlap * overlap * k * 0.25
}

/// Signed distance from `p` to the capsule of `radius` around the segment
/// `a`..`b`.
pub fn capsule_distance(p: Vec3, a: Vec3, b: Vec3, radius: f64) -> f64 {
    let axis = geometry::difference(b, a);
    let offset = geometry::difference(p, a);
    let length_sq = geometry::inner(axis, axis);
    let along = if length_sq > 0.0 {
        (geometry::inner(offset, axis) / length_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = geometry::scale(axis, along);
    geometry::length(geometry::difference(offset, closest)) - radius
}

/// Cell index range a ball of `reach` about the segment `a`..`b` can touch.
fn segment_box(grid: &Grid, a: Vec3, b: Vec3, reach: f64) -> Option<[(usize, usize); 3]> {
    let mut ranges = [(0usize, 0usize); 3];
    let counts = [grid.nx, grid.ny, grid.nz];
    for d in 0..3 {
        let low = a[d].min(b[d]) - reach;
        let high = a[d].max(b[d]) + reach;
        if !low.is_finite() || !high.is_finite() {
            return None;
        }
        let first = ((low - grid.origin[d]) / grid.h).floor();
        let last = ((high - grid.origin[d]) / grid.h).floor();
        if last < 0.0 || first >= counts[d] as f64 {
            return None;
        }
        ranges[d] = (
            first.max(0.0) as usize,
            (last.min(counts[d] as f64 - 1.0)) as usize,
        );
    }
    Some(ranges)
}

/// One strut ready to be rasterized: its two ends, its radius, and the cell
/// index range it can reach into.
#[derive(Debug, Clone, Copy)]
struct Strut {
    from: Vec3,
    to: Vec3,
    radius: f64,
    ranges: [(usize, usize); 3],
}

/// Signed distance from every cell centre to the smooth union of the struts.
///
/// `radii[c]` is the radius of the segment that joins node `c` to its parent;
/// roots have no segment and their entry is ignored.
pub fn distance_field(grid: &Grid, skeleton: &Skeleton, radii: &[f64], blend: f64) -> Vec<f64> {
    // The band that matters is the radius plus the blend width plus the voxel
    // the smoothstep spans; beyond it a cell's density is zero either way.
    let margin = blend + constants::GROWTH_SURFACE_WIDTH_VOXELS * grid.h;
    let struts: Vec<Strut> = skeleton
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(child, node)| {
            let parent = node.parent?;
            let from = skeleton.position(parent);
            let to = node.position;
            let radius = radii[child];
            Some(Strut {
                from,
                to,
                radius,
                ranges: segment_box(grid, from, to, radius + margin)?,
            })
        })
        .collect();

    let slab = grid.nx * grid.ny;
    let mut distance = vec![f64::INFINITY; grid.n_cells()];
    distance
        .par_chunks_mut(slab)
        .enumerate()
        .for_each(|(k, cells)| {
            for strut in &struts {
                if k < strut.ranges[2].0 || k > strut.ranges[2].1 {
                    continue;
                }
                for j in strut.ranges[1].0..=strut.ranges[1].1 {
                    for i in strut.ranges[0].0..=strut.ranges[0].1 {
                        let centre = grid.cell_center(i, j, k);
                        let d = capsule_distance(centre, strut.from, strut.to, strut.radius);
                        let slot = &mut cells[i + grid.nx * j];
                        *slot = smooth_min(*slot, d, blend);
                    }
                }
            }
        });
    distance
}

/// Density of one cell from its signed distance to the structure.
fn density_of(distance: f64, width: f64) -> f64 {
    let t = (0.5 - distance / width).clamp(constants::DENSITY_MIN, constants::DENSITY_MAX);
    // The classic smoothstep, so the surface band has zero slope at both ends
    // and marching cubes does not see a crease where the ramp starts.
    t * t * (3.0 - 2.0 * t)
}

/// Densities of every cell from the skeleton, with the cell classification
/// applied on top.
pub fn densities(grid: &Grid, skeleton: &Skeleton, radii: &[f64]) -> Vec<f64> {
    let blend = constants::GROWTH_SMOOTH_UNION_VOXELS * grid.h;
    let width = constants::GROWTH_SURFACE_WIDTH_VOXELS * grid.h;
    let distance = distance_field(grid, skeleton, radii, blend);
    distance
        .par_iter()
        .zip(grid.cells.par_iter())
        .map(|(d, kind)| match kind {
            CellKind::Void => constants::DENSITY_MIN,
            CellKind::Solid => constants::DENSITY_MAX,
            CellKind::Design => density_of(*d, width),
        })
        .collect()
}

/// Mean density over the design cells, which is what `mass_fraction` measures.
pub fn design_volume_fraction(grid: &Grid, densities: &[f64]) -> f64 {
    let (total, count) = grid
        .cells
        .par_iter()
        .zip(densities.par_iter())
        .filter(|(kind, _)| **kind == CellKind::Design)
        .map(|(_, d)| (*d, 1usize))
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::growth::fixtures::design_grid;

    #[test]
    fn the_smooth_union_never_exceeds_the_plain_one_and_is_monotone() {
        let k = 0.5;
        for a in [-2.0, -0.4, 0.0, 0.3, 1.5] {
            for b in [-1.7, -0.2, 0.0, 0.9, 2.2] {
                let smooth = smooth_min(a, b, k);
                assert!(
                    smooth <= a.min(b) + 1e-15,
                    "smin({a}, {b}) = {smooth} is above min"
                );
                assert!(
                    smooth >= a.min(b) - 0.25 * k - 1e-15,
                    "smin({a}, {b}) = {smooth} dipped more than the blend allows"
                );
                // Symmetric, and adding material to either input can only bring
                // the union nearer: monotone decreasing in each argument.
                assert!((smooth - smooth_min(b, a, k)).abs() < 1e-15);
                assert!(smooth_min(a - 0.3, b, k) <= smooth + 1e-15);
                assert!(smooth_min(a, b - 0.3, k) <= smooth + 1e-15);
            }
        }
        // Far apart it is the plain minimum, and a zero blend always is.
        assert!((smooth_min(-5.0, 5.0, k) - -5.0).abs() < 1e-15);
        assert_eq!(smooth_min(1.0, 2.0, 0.0), 1.0);
        // The +inf the field starts at unions to the other operand exactly.
        assert_eq!(smooth_min(f64::INFINITY, -0.5, k), -0.5);
        assert_eq!(smooth_min(f64::INFINITY, f64::INFINITY, k), f64::INFINITY);
    }

    #[test]
    fn a_single_capsule_rasterizes_to_its_analytic_volume() {
        // A 1 mm grid, a 30 mm long strut of radius 4 mm along x, well clear of
        // every boundary.
        let grid = design_grid(48);
        let mut skeleton = Skeleton::new();
        skeleton.add_chain(&[[9.0, 24.0, 24.0], [39.0, 24.0, 24.0]]);
        let radius: f64 = 4.0;
        let radii = vec![0.0, radius];

        let field = densities(&grid, &skeleton, &radii);
        let volume: f64 = field.iter().sum::<f64>() * grid.cell_volume();
        // A capsule is a cylinder plus a sphere.
        let expected = std::f64::consts::PI * radius * radius * 30.0
            + 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        let error = (volume - expected).abs() / expected;
        assert!(
            error < 0.03,
            "rasterized {volume:.2} mm3 against an analytic {expected:.2} mm3 ({:.2}%)",
            error * 100.0
        );

        // The axis is solid, the far corner is empty, and the surface sits
        // where the signed distance vanishes.
        let solid = grid.cell_index(24, 24, 24);
        assert!((field[solid] - 1.0).abs() < 1e-9);
        assert_eq!(field[grid.cell_index(1, 1, 1)], 0.0);
        assert!((density_of(0.0, grid.h) - 0.5).abs() < 1e-12);
        assert_eq!(density_of(10.0, grid.h), 0.0);
        assert_eq!(density_of(-10.0, grid.h), 1.0);
    }

    #[test]
    fn keepouts_stay_empty_and_keepins_stay_solid_whatever_grows() {
        let mut grid = design_grid(12);
        let void = grid.cell_index(6, 6, 6);
        let solid = grid.cell_index(1, 1, 1);
        grid.cells[void] = CellKind::Void;
        grid.cells[solid] = CellKind::Solid;

        let mut skeleton = Skeleton::new();
        // Straight through the void cell, and nowhere near the solid one.
        skeleton.add_chain(&[[6.5, 0.5, 6.5], [6.5, 11.5, 6.5]]);
        let field = densities(&grid, &skeleton, &[0.0, 3.0]);

        assert_eq!(field[void], 0.0, "a strut wrote into a keepout");
        assert_eq!(field[solid], 1.0, "a keepin lost its material");
        assert!(
            field[grid.cell_index(6, 5, 6)] > 0.9,
            "the strut is missing"
        );
        assert!(field.iter().all(|d| (0.0..=1.0).contains(d)));
    }

    #[test]
    fn the_volume_fraction_is_the_mean_over_the_design_cells_alone() {
        let mut grid = design_grid(4);
        let void = grid.cell_index(0, 0, 0);
        let solid = grid.cell_index(1, 0, 0);
        grid.cells[void] = CellKind::Void;
        grid.cells[solid] = CellKind::Solid;
        let mut field = vec![0.0; grid.n_cells()];
        field[void] = 0.0;
        field[solid] = 1.0;
        // Half of the 62 design cells solid.
        for slot in field.iter_mut().skip(2).take(31) {
            *slot = 1.0;
        }
        let fraction = design_volume_fraction(&grid, &field);
        assert!((fraction - 31.0 / 62.0).abs() < 1e-12, "got {fraction}");
    }

    #[test]
    fn thicker_struts_never_produce_less_material() {
        let grid = design_grid(24);
        let mut skeleton = Skeleton::new();
        skeleton.add_chain(&[[6.0, 12.0, 12.0], [18.0, 12.0, 12.0]]);
        let mut previous = 0.0;
        for radius in [1.0, 2.0, 3.0, 4.0] {
            let fraction =
                design_volume_fraction(&grid, &densities(&grid, &skeleton, &[0.0, radius]));
            assert!(
                fraction >= previous,
                "radius {radius} gave {fraction} after {previous}"
            );
            previous = fraction;
        }
        assert!(previous > 0.0);
    }
}
