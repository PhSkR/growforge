//! Isosurface extraction, smoothing, validation and STL export.

pub mod clamp;
pub mod islands;
pub mod marching_cubes;
pub mod smooth;
pub mod stl;
pub mod tables;
pub mod validate;

use anyhow::Result;
use rayon::prelude::*;

use crate::config::IslandPolicy;
use crate::constants;
use crate::geometry::{Boundaries, Vec3, cross, difference, length, scale};
use crate::grid::{CellKind, Grid};
use clamp::ClampReport;
use islands::IslandReport;
use validate::MeshStats;

/// An indexed triangle mesh in millimetre world coordinates.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    /// Vertex positions.
    pub vertices: Vec<Vec3>,
    /// Triangles as vertex index triples, wound counter-clockwise seen from
    /// outside the solid.
    pub triangles: Vec<[u32; 3]>,
}

impl Mesh {
    /// True when the mesh holds no triangles.
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }
}

/// A scalar field sampled on a uniform point lattice.
#[derive(Debug, Clone)]
pub struct ScalarField {
    /// Sample count along x.
    pub nx: usize,
    /// Sample count along y.
    pub ny: usize,
    /// Sample count along z.
    pub nz: usize,
    /// World position of sample (0, 0, 0).
    pub origin: Vec3,
    /// Sample spacing in millimetres.
    pub spacing: f64,
    /// Sample values in x-fastest order.
    pub values: Vec<f64>,
}

impl ScalarField {
    /// Allocate a zeroed field.
    pub fn new(nx: usize, ny: usize, nz: usize, origin: Vec3, spacing: f64) -> ScalarField {
        ScalarField {
            nx,
            ny,
            nz,
            origin,
            spacing,
            values: vec![0.0; nx * ny * nz],
        }
    }

    /// Linear index of sample `(i, j, k)`.
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// Value at sample `(i, j, k)`.
    pub fn at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.values[self.index(i, j, k)]
    }

    /// World position of sample `(i, j, k)`.
    pub fn position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        [
            self.origin[0] + i as f64 * self.spacing,
            self.origin[1] + j as f64 * self.spacing,
            self.origin[2] + k as f64 * self.spacing,
        ]
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when the field holds no samples.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Trilinear value at a world position, clamped to the sampled block.
    ///
    /// Positions outside the lattice read its boundary rather than zero, so a
    /// vertex that smoothing nudged past the padding still gets the value of the
    /// material it came from.
    pub fn sample(&self, position: Vec3) -> f64 {
        let counts = [self.nx, self.ny, self.nz];
        if counts.contains(&0) {
            return 0.0;
        }
        let mut base = [0usize; 3];
        let mut frac = [0.0f64; 3];
        for d in 0..3 {
            let last = counts[d] - 1;
            let local = (position[d] - self.origin[d]) / self.spacing;
            let clamped = if local.is_finite() {
                local.clamp(0.0, last as f64)
            } else {
                0.0
            };
            let cell = (clamped.floor() as usize).min(last.saturating_sub(1));
            base[d] = cell;
            frac[d] = (clamped - cell as f64).clamp(0.0, 1.0);
        }
        let mut value = 0.0;
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let weight = [(di, 0), (dj, 1), (dk, 2)]
                        .into_iter()
                        .map(|(step, d)| if step == 0 { 1.0 - frac[d] } else { frac[d] })
                        .product::<f64>();
                    if weight == 0.0 {
                        continue;
                    }
                    let i = (base[0] + di).min(counts[0] - 1);
                    let j = (base[1] + dj).min(counts[1] - 1);
                    let k = (base[2] + dk).min(counts[2] - 1);
                    value += weight * self.at(i, j, k);
                }
            }
        }
        value
    }

    /// Resample onto a lattice `factor` times finer along every axis.
    ///
    /// The refined lattice covers exactly the same block: same origin, spacing
    /// divided by `factor`, and a sample landing on every sample of the source,
    /// whose value it copies rather than interpolates. Everything in between is
    /// trilinear, so the refined field *is* this field's trilinear interpolant
    /// read finely - the same interpolant [`ScalarField::sample`] evaluates -
    /// and the zero padding stays zero, which is what still closes the
    /// isosurface inside the block.
    ///
    /// A `factor` of one hands the field back untouched, so the default export
    /// is the one that existed before supersampling, bit for bit.
    pub fn refined(self, factor: usize) -> ScalarField {
        if factor <= 1 || self.nx < 2 || self.ny < 2 || self.nz < 2 {
            return self;
        }
        // Lower source sample and interpolation weight of every refined index,
        // built once per axis: index arithmetic rather than world positions, so
        // a refined sample sitting on a source sample reads it exactly.
        let axis = |count: usize| -> Vec<(usize, f64)> {
            (0..(count - 1) * factor + 1)
                .map(|index| {
                    let cell = (index / factor).min(count - 2);
                    (cell, (index - cell * factor) as f64 / factor as f64)
                })
                .collect()
        };
        let (xs, ys, zs) = (axis(self.nx), axis(self.ny), axis(self.nz));
        let mut out = ScalarField::new(
            xs.len(),
            ys.len(),
            zs.len(),
            self.origin,
            self.spacing / factor as f64,
        );
        let slab = out.nx * out.ny;
        out.values
            .par_chunks_mut(slab)
            .enumerate()
            .for_each(|(pk, values)| {
                let (k, wk) = zs[pk];
                for (pj, row) in values.chunks_mut(xs.len()).enumerate() {
                    let (j, wj) = ys[pj];
                    for (slot, &(i, wi)) in row.iter_mut().zip(xs.iter()) {
                        let mut value = 0.0;
                        for (dk, fk) in [(0, 1.0 - wk), (1, wk)] {
                            for (dj, fj) in [(0, 1.0 - wj), (1, wj)] {
                                for (di, fi) in [(0, 1.0 - wi), (1, wi)] {
                                    let weight = fi * fj * fk;
                                    if weight == 0.0 {
                                        continue;
                                    }
                                    value += weight * self.at(i + di, j + dj, k + dk);
                                }
                            }
                        }
                        *slot = value;
                    }
                }
            });
        out
    }

    /// Push samples that sit exactly on `iso` to the material side.
    ///
    /// Sampling a flat solid boundary produces node values exactly equal to the
    /// iso level, which would place several marching cubes vertices on the same
    /// lattice point and emit zero-area triangles. Nudging the value keeps
    /// every vertex strictly inside its edge; the displacement is
    /// `ISO_SNAP_EPS` in density units, far below the resolution of the
    /// optimization itself.
    pub fn snap_away_from(&mut self, iso: f64) {
        for v in self.values.iter_mut() {
            if (*v - iso).abs() < constants::ISO_SNAP_EPS {
                *v = iso + constants::ISO_SNAP_EPS;
            }
        }
    }
}

/// Sample the physical densities of a grid onto its node lattice.
///
/// A node value is the mean of the densities of its eight adjacent cells; cells
/// outside the grid count as zero. The lattice is wrapped in
/// `FIELD_PADDING_CELLS` layers of zeros so the isosurface always closes inside
/// the sampled block.
pub fn node_density_field(grid: &Grid, densities: &[f64], iso: f64) -> ScalarField {
    assert_eq!(
        densities.len(),
        grid.n_cells(),
        "density array must have one entry per cell"
    );
    let pad = constants::FIELD_PADDING_CELLS;
    let mut field = ScalarField::new(
        grid.nnx() + 2 * pad,
        grid.nny() + 2 * pad,
        grid.nnz() + 2 * pad,
        [
            grid.origin[0] - pad as f64 * grid.h,
            grid.origin[1] - pad as f64 * grid.h,
            grid.origin[2] - pad as f64 * grid.h,
        ],
        grid.h,
    );

    let corners_per_node = 1 << 3;
    for pk in 0..field.nz {
        for pj in 0..field.ny {
            for pi in 0..field.nx {
                let (i, j, k) = (
                    pi as isize - pad as isize,
                    pj as isize - pad as isize,
                    pk as isize - pad as isize,
                );
                if i < 0
                    || j < 0
                    || k < 0
                    || i > grid.nx as isize
                    || j > grid.ny as isize
                    || k > grid.nz as isize
                {
                    continue;
                }
                let mut sum = 0.0;
                for dk in 0..2 {
                    for dj in 0..2 {
                        for di in 0..2 {
                            let ci = i - 1 + di;
                            let cj = j - 1 + dj;
                            let ck = k - 1 + dk;
                            if ci < 0
                                || cj < 0
                                || ck < 0
                                || ci >= grid.nx as isize
                                || cj >= grid.ny as isize
                                || ck >= grid.nz as isize
                            {
                                continue;
                            }
                            let e = grid.cell_index(ci as usize, cj as usize, ck as usize);
                            sum += match grid.cells[e] {
                                CellKind::Void => 0.0,
                                CellKind::Solid => 1.0,
                                CellKind::Design => densities[e],
                            };
                        }
                    }
                }
                let index = field.index(pi, pj, pk);
                field.values[index] = sum / corners_per_node as f64;
            }
        }
    }
    field.snap_away_from(iso);
    field
}

/// Number of samples the export lattice of `grid` holds at supersampling
/// `factor`: the padded node lattice, refined.
///
/// Both the refined field and the marching cubes edge table are sized against
/// it, so it is what the memory of an export scales with. The arithmetic
/// saturates, so an absurd configuration reports `usize::MAX` rather than
/// wrapping into a small number.
pub fn supersampled_node_count(grid: &Grid, factor: usize) -> usize {
    let pad = 2 * constants::FIELD_PADDING_CELLS;
    let factor = factor.max(1);
    [grid.nnx() + pad, grid.nny() + pad, grid.nnz() + pad]
        .into_iter()
        .map(|n| (n - 1).saturating_mul(factor).saturating_add(1))
        .fold(1usize, usize::saturating_mul)
}

/// Area weighted vertex normals of an indexed mesh, one per vertex.
///
/// Every triangle contributes its raw cross product, whose length is twice its
/// area, so a wide triangle pulls harder than a sliver without an explicit
/// weight; the sum is normalized at the end. Triangles carrying a non-finite
/// corner contribute nothing, and a vertex left without a usable direction
/// comes back as the zero vector, which is how a caller knows to fall back on
/// the face normal.
///
/// This is shading data: the exported STL carries per-triangle facet normals,
/// as its format requires, and never these.
pub fn vertex_normals(mesh: &Mesh) -> Vec<Vec3> {
    let mut normals = vec![[0.0f64; 3]; mesh.vertices.len()];
    for triangle in &mesh.triangles {
        let corners = triangle.map(|index| mesh.vertices[index as usize]);
        if corners
            .iter()
            .any(|c| c.iter().any(|value| !value.is_finite()))
        {
            continue;
        }
        let raw = cross(
            difference(corners[1], corners[0]),
            difference(corners[2], corners[0]),
        );
        for &index in triangle.iter() {
            let slot = &mut normals[index as usize];
            for d in 0..3 {
                slot[d] += raw[d];
            }
        }
    }
    for normal in normals.iter_mut() {
        let len = length(*normal);
        *normal = if len > 0.0 && len.is_finite() {
            scale(*normal, 1.0 / len)
        } else {
            [0.0; 3]
        };
    }
    normals
}

/// A surface that is ready to be written, and what the passes over it found.
#[derive(Debug, Clone)]
pub struct SurfaceExport {
    /// The surface itself, exactly as it will be written.
    pub mesh: Mesh,
    /// Statistics of that surface.
    pub stats: MeshStats,
    /// What the floating fragment pass found and did.
    pub islands: IslandReport,
    /// What the boundary clamp moved; `None` when the caller handed in no
    /// analytic boundaries to hold the surface to, which is what
    /// `[output] boundaries = "voxel"` does.
    pub clamp: Option<ClampReport>,
}

/// Full export pipeline: sample, refine, extract, smooth, cull, clamp and
/// validate.
///
/// `supersample` is the `[output]` factor: one meshes the node lattice itself,
/// and anything higher runs marching cubes on a lattice that much finer,
/// interpolated from the same node field. Smoothing and validation see only the
/// mesh that comes out, so neither knows the difference.
///
/// `islands` is the `[output]` floating fragment policy and `anchors` the
/// declared purposes it judges a component by (see
/// [`crate::problem::Problem::anchors`]); both are applied to the smoothed mesh
/// and therefore compose with supersampling for free, since the components are
/// whatever the refined surface came out in. Culling happens *before*
/// validation, so what the validator accepts is the file that ships and not a
/// superset of it.
///
/// `boundaries` are the analytic domain, keepout and keepin surfaces the field
/// is a voxelization of, or `None` to export the isosurface exactly as that
/// field produced it - which is `[output] boundaries = "voxel"`, and what a
/// caller that has no shapes to hold the mesh to passes. With them, every vertex
/// that ends up inside a keepout or outside the solid - the domain union the
/// keepins - is projected back onto the surface it violates; see [`clamp`]. The clamp runs after the cull, so no work
/// goes into fragments that are about to be discarded and the culling verdicts
/// see the geometry they always did, and before validation, so the positions
/// that are validated are the positions that are written.
#[allow(clippy::too_many_arguments)]
pub fn surface_from_densities(
    grid: &Grid,
    densities: &[f64],
    iso: f64,
    smoothing_iterations: usize,
    supersample: usize,
    islands: IslandPolicy,
    anchors: &islands::Anchors,
    boundaries: Option<&Boundaries>,
) -> Result<SurfaceExport> {
    let mut field = node_density_field(grid, densities, iso);
    if supersample > 1 {
        field = field.refined(supersample);
        // Interpolating between a snapped node and its neighbour can land a
        // refined sample back exactly on the iso level, which is the same
        // zero-area triangle the node sampling nudges away from.
        field.snap_away_from(iso);
    }
    let mut mesh = marching_cubes::extract(&field, iso)?;
    smooth::taubin(&mut mesh, smoothing_iterations);
    let report = islands::resolve(&mut mesh, islands, anchors);
    // The voxel size the field was sampled on is what bounds a correction: the
    // encroachment being corrected is an artefact of that sampling.
    let clamped = boundaries.map(|boundaries| clamp::resolve(&mut mesh, boundaries, grid.h));
    let stats = validate::validate(&mesh)?;
    Ok(SurfaceExport {
        mesh,
        stats,
        islands: report,
        clamp: clamped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;

    fn solid_grid(n: usize, h: f64) -> Grid {
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [n as f64 * h, n as f64 * h, n as f64 * h],
            },
            h,
        );
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Solid;
        }
        grid
    }

    #[test]
    fn node_field_is_padded_with_zeros() {
        let grid = solid_grid(3, 1.0);
        let densities = vec![1.0; grid.n_cells()];
        let field = node_density_field(&grid, &densities, 0.5);
        assert_eq!(field.nx, grid.nnx() + 2);
        for k in 0..field.nz {
            for j in 0..field.ny {
                assert_eq!(field.at(0, j, k), 0.0);
                assert_eq!(field.at(field.nx - 1, j, k), 0.0);
            }
        }
        // The centre of a fully solid block is one.
        assert!((field.at(3, 3, 3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn trilinear_sampling_hits_the_lattice_and_clamps_outside_it() {
        let mut field = ScalarField::new(3, 3, 3, [0.0, 0.0, 0.0], 2.0);
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    let index = field.index(i, j, k);
                    field.values[index] = (i + 10 * j + 100 * k) as f64;
                }
            }
        }
        // Exactly on a sample.
        assert!((field.sample([2.0, 4.0, 0.0]) - 21.0).abs() < 1e-12);
        // Halfway along one edge.
        assert!((field.sample([1.0, 0.0, 0.0]) - 0.5).abs() < 1e-12);
        // The centre of a cell is the mean of its eight corners.
        let expected = (0..8)
            .map(|c| ((c & 1) + 10 * ((c >> 1) & 1) + 100 * ((c >> 2) & 1)) as f64)
            .sum::<f64>()
            / 8.0;
        assert!((field.sample([1.0, 1.0, 1.0]) - expected).abs() < 1e-12);
        // Outside the block reads the boundary, never zero.
        assert!((field.sample([-50.0, -50.0, -50.0]) - 0.0).abs() < 1e-12);
        assert!((field.sample([50.0, 50.0, 50.0]) - 222.0).abs() < 1e-12);
        assert_eq!(field.sample([f64::NAN, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn boundary_nodes_are_nudged_off_the_iso_level() {
        let grid = solid_grid(3, 1.0);
        let densities = vec![1.0; grid.n_cells()];
        let field = node_density_field(&grid, &densities, 0.5);
        // A node on the outer face of a solid block averages four solid cells
        // and four cells outside the grid.
        let v = field.at(1, 2, 2);
        assert!((v - 0.5).abs() > 0.0, "value {v} was left on the iso level");
        assert!((v - 0.5).abs() <= constants::ISO_SNAP_EPS + 1e-15);
        assert!(
            v > 0.5,
            "the nudge must keep the boundary inside the surface"
        );
    }

    #[test]
    fn solid_block_exports_a_watertight_box_of_the_right_size() {
        // Averaging eight cells per node puts the faces of a solid block
        // exactly on the domain boundary but drops the node values along its
        // outer edges and corners below the iso level, so the exported solid
        // carries a one voxel wide 45 degree chamfer there. The chamfer costs
        // about 12 * h^2 / 2 * (L - 2h) plus the corner pieces, which is
        // roughly 1.4 % of this block.
        let cells = 20;
        let h = 1.0;
        let grid = solid_grid(cells, h);
        let densities = vec![1.0; grid.n_cells()];
        let SurfaceExport { mesh, stats, .. } = surface_from_densities(
            &grid,
            &densities,
            0.5,
            0,
            1,
            IslandPolicy::Cull,
            &islands::Anchors::none(),
            None,
        )
        .expect("surface");
        assert!(!mesh.is_empty());

        let side = cells as f64 * h;
        let expected = side.powi(3);
        assert!(
            stats.volume_mm3 < expected,
            "the chamfer must remove material, got {}",
            stats.volume_mm3
        );
        assert!(
            (expected - stats.volume_mm3) / expected < 0.02,
            "volume {} is more than 2 % below {expected}",
            stats.volume_mm3
        );
        // The flat faces stay on the domain boundary, up to the sub-micron
        // offset the iso nudge introduces (ISO_SNAP_EPS / iso of a voxel).
        let tolerance = h * constants::ISO_SNAP_EPS / 0.5 * 2.0;
        for d in 0..3 {
            assert!(
                stats.bounds.min[d].abs() < tolerance,
                "face min {} is off the domain boundary",
                stats.bounds.min[d]
            );
            assert!(
                (stats.bounds.max[d] - side).abs() < tolerance,
                "face max {} is off the domain boundary",
                stats.bounds.max[d]
            );
        }
    }

    /// A grid of design cells whose densities describe an off-centre ball, so
    /// the surface is curved, asymmetric and crosses the iso level everywhere
    /// rather than sitting on the domain boundary.
    fn blob_grid(n: usize, h: f64) -> (Grid, Vec<f64>) {
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [n as f64 * h, n as f64 * h, n as f64 * h],
            },
            h,
        );
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Design;
        }
        let side = n as f64 * h;
        let centre = [0.46 * side, 0.52 * side, 0.5 * side];
        let radius = 0.31 * side;
        let densities = (0..grid.n_cells())
            .map(|e| {
                let (i, j, k) = grid.cell_ijk(e);
                let p = [
                    grid.origin[0] + (i as f64 + 0.5) * h,
                    grid.origin[1] + (j as f64 + 0.5) * h,
                    grid.origin[2] + (k as f64 + 0.5) * h,
                ];
                // A smooth ramp across the ball's shell, squeezed on one axis
                // so the shape has no symmetry to hide a welding fault behind.
                let d = ((p[0] - centre[0]).powi(2)
                    + 1.7 * (p[1] - centre[1]).powi(2)
                    + (p[2] - centre[2]).powi(2))
                .sqrt();
                ((radius - d) / h * 0.5 + 0.5).clamp(0.0, 1.0)
            })
            .collect();
        (grid, densities)
    }

    /// A sampled sphere signed distance field, the fixture the analytic volume
    /// is checked against.
    fn sphere_field(radius: f64, spacing: f64) -> ScalarField {
        let half = radius * 1.5;
        let n = (2.0 * half / spacing) as usize + 1;
        let mut field = ScalarField::new(n, n, n, [-half, -half, -half], spacing);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = field.position(i, j, k);
                    let index = field.index(i, j, k);
                    field.values[index] = radius - (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                }
            }
        }
        field.snap_away_from(0.0);
        field
    }

    /// Extract a field at `factor`, exactly as [`surface_from_densities`] does.
    fn extract_at(field: &ScalarField, iso: f64, factor: usize) -> Mesh {
        let mut refined = field.clone().refined(factor);
        if factor > 1 {
            refined.snap_away_from(iso);
        }
        marching_cubes::extract(&refined, iso).expect("extraction")
    }

    #[test]
    fn supersample_one_writes_the_same_stl_as_the_unrefined_pipeline() {
        // The regression this pins: refining must be an addition, not a
        // refactor. At factor one the export has to be the sample, extract,
        // smooth, validate, write sequence it always was, byte for byte.
        let (grid, densities) = blob_grid(12, 1.5);
        let iso = 0.5;

        let field = node_density_field(&grid, &densities, iso);
        let mut reference = marching_cubes::extract(&field, iso).expect("extraction");
        smooth::taubin(&mut reference, 6);
        let reference_stats = validate::validate(&reference).expect("watertight");

        let SurfaceExport {
            mesh,
            stats,
            islands,
            clamp,
        } = surface_from_densities(
            &grid,
            &densities,
            iso,
            6,
            1,
            IslandPolicy::Cull,
            &islands::Anchors::none(),
            None,
        )
        .expect("surface");
        assert_eq!(stats, reference_stats);
        assert_eq!(mesh.vertices, reference.vertices);
        assert_eq!(mesh.triangles, reference.triangles);
        // A blob is one component, so the cull is a no-op on it and the bytes
        // below are the ones the pipeline wrote before the pass existed. No
        // boundaries were handed in either, so no vertex was clamped.
        assert_eq!(islands.components, 1);
        assert_eq!(islands.culled_triangles, 0);
        assert_eq!(clamp, None);

        let directory = std::env::temp_dir();
        let a = directory.join("growforge_supersample_reference.stl");
        let b = directory.join("growforge_supersample_one.stl");
        stl::write(&a, &reference).expect("write");
        stl::write(&b, &mesh).expect("write");
        assert_eq!(
            std::fs::read(&a).expect("reference STL"),
            std::fs::read(&b).expect("supersample 1 STL"),
            "supersample = 1 must write the same bytes as the unrefined pipeline"
        );
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();

        // And the refinement itself is the identity there, allocation aside.
        let untouched = field.clone().refined(1);
        assert_eq!(untouched.values, field.values);
        assert_eq!(untouched.spacing, field.spacing);
        assert_eq!(
            (untouched.nx, untouched.ny, untouched.nz),
            (field.nx, field.ny, field.nz)
        );
    }

    /// A keepin *buried* inside the domain - further from the part's surface
    /// than the capture band - adds nothing to the solid the clamp holds the
    /// mesh to, and so changes nothing about the file that ships.
    ///
    /// Buried is the whole of the claim. A keepin's skin is a seat target and a
    /// flush surface wherever it lies, so an interior keepin a surface already
    /// rests within half a voxel of, or a `flush = "walls"` run, can change the
    /// export; what this pins is the case the change was not supposed to reach,
    /// which is most keepins.
    #[test]
    fn a_keepin_inside_the_domain_writes_the_same_stl() {
        use crate::geometry::{Csg, CsgOp, Shape, ShapeUnion};

        let h = 1.5;
        let (grid, densities) = blob_grid(12, h);
        let side = 12.0 * h;
        let domain = Csg::new(vec![(
            CsgOp::Add,
            Shape::axis_aligned_box([0.0, 0.0, 0.0], [side, side, side]),
        )]);
        // Deep inside the blob's material: further from its surface than the
        // capture band and than the adrift window, so nothing seats onto it and
        // nothing is counted against it either.
        let buried = Shape::Sphere {
            center: [0.46 * side, 0.52 * side, 0.5 * side],
            radius: 0.5,
        };
        let held = Boundaries {
            domain: domain.clone(),
            keepout: ShapeUnion::default(),
            keepin: ShapeUnion::new(vec![buried]),
        };
        let plain = Boundaries {
            domain,
            keepout: ShapeUnion::default(),
            keepin: ShapeUnion::default(),
        };

        let export = |boundaries: &Boundaries| {
            surface_from_densities(
                &grid,
                &densities,
                0.5,
                6,
                1,
                IslandPolicy::Cull,
                &islands::Anchors::none(),
                Some(boundaries),
            )
            .expect("surface")
        };
        let reference = export(&plain);
        let with_keepin = export(&held);
        assert!(!reference.mesh.triangles.is_empty());
        assert_eq!(with_keepin.clamp, reference.clamp);
        assert_eq!(with_keepin.stats, reference.stats);

        let directory = std::env::temp_dir();
        let a = directory.join("growforge_interior_keepin_reference.stl");
        let b = directory.join("growforge_interior_keepin.stl");
        stl::write(&a, &reference.mesh).expect("write");
        stl::write(&b, &with_keepin.mesh).expect("write");
        assert_eq!(
            std::fs::read(&a).expect("reference STL"),
            std::fs::read(&b).expect("keepin STL"),
            "a buried keepin must not change a single byte of the export"
        );
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();
    }

    #[test]
    fn a_refined_lattice_keeps_its_source_samples_and_interpolates_between_them() {
        let mut field = ScalarField::new(3, 3, 3, [-1.0, 2.0, 0.5], 2.0);
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    let index = field.index(i, j, k);
                    field.values[index] = (i + 10 * j + 100 * k) as f64;
                }
            }
        }
        let source = field.clone();
        let refined = field.refined(2);

        assert_eq!((refined.nx, refined.ny, refined.nz), (5, 5, 5));
        assert_eq!(refined.spacing, 1.0);
        assert_eq!(refined.origin, source.origin);
        // Every source sample survives exactly, including the last one on each
        // axis, which is the clamped end of the interpolation.
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    assert_eq!(
                        refined.at(2 * i, 2 * j, 2 * k),
                        source.at(i, j, k),
                        "sample ({i}, {j}, {k}) was not copied"
                    );
                }
            }
        }
        // Halfway along an edge is the mean of its ends, and the centre of a
        // cell the mean of its eight corners: the source's own interpolant.
        assert!((refined.at(1, 0, 0) - 0.5).abs() < 1e-12);
        assert!((refined.at(0, 1, 0) - 5.0).abs() < 1e-12);
        assert!((refined.at(1, 1, 1) - source.sample([0.0, 3.0, 1.5])).abs() < 1e-12);
        // The refined lattice covers the same block, so a refined sample and a
        // world-space lookup of the source agree everywhere.
        for k in 0..refined.nz {
            for j in 0..refined.ny {
                for i in 0..refined.nx {
                    let expected = source.sample(refined.position(i, j, k));
                    assert!(
                        (refined.at(i, j, k) - expected).abs() < 1e-9,
                        "refined ({i}, {j}, {k}) is {} not {expected}",
                        refined.at(i, j, k)
                    );
                }
            }
        }
    }

    #[test]
    fn supersampling_a_sphere_closes_on_the_analytic_volume() {
        let radius = 10.0;
        let field = sphere_field(radius, 1.5);
        let expected = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);

        let coarse = extract_at(&field, 0.0, 1);
        let fine = extract_at(&field, 0.0, 2);
        let coarse_stats = validate::validate(&coarse).expect("closed sphere");
        let fine_stats = validate::validate(&fine).expect("closed refined sphere");

        let coarse_error = (coarse_stats.volume_mm3 - expected).abs() / expected;
        let fine_error = (fine_stats.volume_mm3 - expected).abs() / expected;
        assert!(
            fine_error < coarse_error,
            "supersample 2 is {:.4}% off against {:.4}% at supersample 1",
            fine_error * 100.0,
            coarse_error * 100.0
        );

        // A halved edge length is four times the triangles on the same surface.
        let ratio = fine_stats.triangles as f64 / coarse_stats.triangles as f64;
        assert!(
            (3.4..4.6).contains(&ratio),
            "supersample 2 gave {} triangles against {}, a ratio of {ratio:.2}",
            fine_stats.triangles,
            coarse_stats.triangles
        );
        // Every vertex still sits on the sphere, now within the finer spacing.
        for v in &fine.vertices {
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((r - radius).abs() < 0.75, "vertex at radius {r}");
        }
    }

    #[test]
    fn an_asymmetric_field_stays_watertight_at_supersample_three() {
        // Welding keys on the refined lattice's own edge identities, so a crack
        // between two refined cubes shows up as an edge used once instead of
        // twice, which is exactly what the validator refuses.
        let (grid, densities) = blob_grid(9, 2.0);
        let iso = 0.5;
        let SurfaceExport { mesh, stats, .. } = surface_from_densities(
            &grid,
            &densities,
            iso,
            4,
            3,
            IslandPolicy::Cull,
            &islands::Anchors::none(),
            None,
        )
        .expect("surface");
        assert!(stats.triangles > 0 && stats.volume_mm3 > 0.0);
        validate::validate(&mesh).expect("the refined surface must be closed and manifold");

        let plain = surface_from_densities(
            &grid,
            &densities,
            iso,
            4,
            1,
            IslandPolicy::Cull,
            &islands::Anchors::none(),
            None,
        )
        .expect("surface");
        let ratio = stats.triangles as f64 / plain.stats.triangles as f64;
        assert!(
            (7.0..11.0).contains(&ratio),
            "supersample 3 gave a triangle ratio of {ratio:.2}"
        );
        // The refined surface stays inside the block the coarse one closed in.
        for d in 0..3 {
            assert!(stats.bounds.min[d] >= plain.stats.bounds.min[d] - grid.h);
            assert!(stats.bounds.max[d] <= plain.stats.bounds.max[d] + grid.h);
        }
    }

    /// A design grid of `n` cells a side, one millimetre each.
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

    /// The field the user's run came out with, reduced to what makes it happen:
    /// a solid part, a two-cell-wide clump of intermediate density floating
    /// clear of it, and a one-cell-wide bridge of the same density joining them.
    ///
    /// Every cell of the bridge and the clump is at or above the iso level, so
    /// the cell flood fill of [`crate::voids::solid_bodies`] walks straight
    /// across the bridge and reports **one** body. The node lattice cannot see
    /// the bridge at all - a node touching two of its cells averages them with
    /// six empty ones and lands far below the iso level - while the clump has
    /// one node with eight of its own cells around it, which is above. So the
    /// surface comes out in two pieces where the field says one, and the piece
    /// that ships is a skin welded to nothing.
    fn island_field(density: f64) -> (Grid, Vec<f64>) {
        let grid = design_grid(12);
        let mut densities = vec![0.0; grid.n_cells()];
        for k in 4..8 {
            for j in 4..8 {
                for i in 0..5 {
                    densities[grid.cell_index(i, j, k)] = 1.0;
                }
            }
        }
        for i in 5..8 {
            densities[grid.cell_index(i, 5, 5)] = density;
        }
        for k in 5..7 {
            for j in 5..7 {
                for i in 8..10 {
                    densities[grid.cell_index(i, j, k)] = density;
                }
            }
        }
        (grid, densities)
    }

    /// A support region over the face the part of [`island_field`] stands on,
    /// which is what makes that part the declared geometry and the clump debris.
    fn island_anchors(grid: &Grid) -> islands::Anchors {
        let mut nodes = Vec::new();
        for k in 0..grid.nnz() {
            for j in 0..grid.nny() {
                nodes.push(grid.node_index(0, j, k));
            }
        }
        islands::Anchors::new(
            grid,
            vec![islands::AnchorRegion {
                kind: islands::AnchorKind::Support,
                label: "support 1".to_string(),
                nodes,
                probes: Vec::new(),
                shape: None,
            }],
        )
    }

    #[test]
    fn a_skin_the_cell_flood_fill_cannot_see_is_culled_from_the_export() {
        let (grid, densities) = island_field(0.6);
        let iso = 0.5;
        let anchors = island_anchors(&grid);
        // The defect this pass exists for: the field reports one body.
        assert_eq!(crate::voids::solid_bodies(&grid, &densities, iso).len(), 1);

        let kept = surface_from_densities(
            &grid,
            &densities,
            iso,
            4,
            1,
            IslandPolicy::Keep,
            &anchors,
            None,
        )
        .expect("surface");
        assert_eq!(kept.islands.components, 2, "the surface came out in one");
        assert_eq!(kept.islands.bodies.len(), 2);
        assert_eq!(kept.islands.culled_triangles, 0);
        assert_eq!(kept.islands.fragments.len(), 1);
        assert_eq!(islands::label_components(&kept.mesh).components, 2);
        validate::validate(&kept.mesh).expect("both pieces are closed and manifold");
        let fragment = kept.islands.fragments[0].clone();
        assert!(fragment.volume_mm3 > 0.0 && fragment.volume_mm3 < 1.0);
        assert!(
            fragment.centroid[0] > 8.0,
            "the fragment sits where the clump is: {:?}",
            fragment.centroid
        );

        let culled = surface_from_densities(
            &grid,
            &densities,
            iso,
            4,
            1,
            IslandPolicy::Cull,
            &anchors,
            None,
        )
        .expect("surface");
        assert_eq!(culled.islands.components, 2);
        assert_eq!(culled.islands.bodies.len(), 1);
        assert!(
            culled.islands.bodies[0]
                .anchors
                .holds(islands::AnchorKind::Support),
            "the part is what the support holds"
        );
        assert!(culled.islands.unserved.is_empty());
        assert_eq!(culled.islands.cavity_shells, 0);
        assert_eq!(culled.islands.culled_fragments, 1);
        assert_eq!(culled.islands.fragments, vec![fragment.clone()]);
        assert_eq!(culled.islands.culled_triangles, fragment.triangles);
        assert!((culled.islands.culled_volume_mm3 - fragment.volume_mm3).abs() < 1e-12);
        assert_eq!(
            culled.islands.largest_fragment_mm3(),
            Some(fragment.volume_mm3)
        );

        // What ships is one piece, and it is the part rather than the skin.
        assert_eq!(islands::label_components(&culled.mesh).components, 1);
        validate::validate(&culled.mesh).expect("the culled surface must still validate");
        assert_eq!(
            culled.stats.triangles,
            kept.stats.triangles - fragment.triangles
        );
        assert!(
            (kept.stats.volume_mm3 - culled.stats.volume_mm3 - fragment.volume_mm3).abs() < 1e-9
        );
        assert!(
            culled.stats.bounds.max[0] < 6.0 && kept.stats.bounds.max[0] > 8.0,
            "culling has to take the skin's bounds with it: {:?}",
            culled.stats.bounds
        );
        // The vertices the skin used left with it rather than being orphaned,
        // and every index still addresses a vertex that is there.
        assert_eq!(culled.stats.vertices, culled.mesh.vertices.len());
        assert!(culled.mesh.vertices.len() < kept.mesh.vertices.len());
        assert!(
            culled
                .mesh
                .triangles
                .iter()
                .flatten()
                .all(|&v| (v as usize) < culled.mesh.vertices.len())
        );
    }

    #[test]
    fn culling_composes_with_supersampling() {
        // The components are whatever the refined surface came out in, so the
        // pass reads the same field at more places rather than a different one.
        let (grid, densities) = island_field(0.6);
        let iso = 0.5;
        let anchors = island_anchors(&grid);
        for factor in [1, 2, 3] {
            let export = surface_from_densities(
                &grid,
                &densities,
                iso,
                4,
                factor,
                IslandPolicy::Cull,
                &anchors,
                None,
            )
            .expect("surface");
            assert_eq!(
                export.islands.components, 2,
                "supersample {factor} found {} components",
                export.islands.components
            );
            assert_eq!(export.islands.bodies.len(), 1);
            assert_eq!(export.islands.culled_fragments, 1);
            assert_eq!(islands::label_components(&export.mesh).components, 1);
            validate::validate(&export.mesh).expect("the culled surface must validate");
        }
    }

    #[test]
    fn the_cavity_policy_decides_the_cavity_and_the_island_policy_never_argues() {
        // A part with a sealed cavity, and the same floating skin as above.
        let (grid, mut densities) = island_field(0.6);
        for k in 5..7 {
            for j in 5..7 {
                for i in 1..3 {
                    densities[grid.cell_index(i, j, k)] = 0.0;
                }
            }
        }
        let iso = 0.5;
        let anchors = island_anchors(&grid);
        assert_eq!(crate::voids::detect(&grid, &densities, iso).len(), 1);

        // voids = "warn" keeps the cavity, so its inner shell has to survive
        // the cull: the two policies describe different things.
        let mut warned = densities.clone();
        let report = crate::voids::resolve(
            &grid,
            &mut warned,
            iso,
            crate::config::VoidPolicy::Warn,
            1.0,
        );
        assert_eq!(report.remaining(), 1);
        let export = surface_from_densities(
            &grid,
            &warned,
            iso,
            4,
            1,
            IslandPolicy::Cull,
            &anchors,
            None,
        )
        .expect("surface");
        assert_eq!(export.islands.components, 3);
        assert_eq!(export.islands.bodies.len(), 1);
        assert_eq!(
            export.islands.cavity_shells, 1,
            "the cavity shell was culled"
        );
        assert_eq!(export.islands.culled_fragments, 1);
        assert_eq!(islands::label_components(&export.mesh).components, 2);
        validate::validate(&export.mesh).expect("the culled surface must validate");

        // voids = "fill" removes the cavity before the surface exists, so there
        // is no inner shell to argue about and only the skin is culled.
        let mut filled = densities.clone();
        let report = crate::voids::resolve(
            &grid,
            &mut filled,
            iso,
            crate::config::VoidPolicy::Fill,
            1.0,
        );
        assert_eq!(report.remaining(), 0);
        let export = surface_from_densities(
            &grid,
            &filled,
            iso,
            4,
            1,
            IslandPolicy::Cull,
            &anchors,
            None,
        )
        .expect("surface");
        assert_eq!(export.islands.components, 2);
        assert_eq!(export.islands.bodies.len(), 1);
        assert_eq!(export.islands.cavity_shells, 0);
        assert_eq!(export.islands.culled_fragments, 1);
        assert_eq!(islands::label_components(&export.mesh).components, 1);
        validate::validate(&export.mesh).expect("the culled surface must validate");
    }

    #[test]
    fn a_culled_export_writes_the_same_bytes_as_the_field_without_the_skin() {
        // The cull is not a rewrite of the surface: what ships is exactly what
        // the same pipeline writes for a field that never had the skin in it.
        let iso = 0.5;
        let (grid, with_skin) = island_field(0.6);
        let (_, mut without_skin) = island_field(0.6);
        for k in 5..7 {
            for j in 5..7 {
                for i in 8..10 {
                    without_skin[grid.cell_index(i, j, k)] = 0.0;
                }
            }
        }

        let anchors = island_anchors(&grid);
        let culled = surface_from_densities(
            &grid,
            &with_skin,
            iso,
            4,
            1,
            IslandPolicy::Cull,
            &anchors,
            None,
        )
        .expect("surface");
        let clean = surface_from_densities(
            &grid,
            &without_skin,
            iso,
            4,
            1,
            IslandPolicy::Cull,
            &anchors,
            None,
        )
        .expect("surface");
        assert_eq!(clean.islands.components, 1);
        assert_eq!(culled.mesh.vertices, clean.mesh.vertices);
        assert_eq!(culled.mesh.triangles, clean.mesh.triangles);
        assert_eq!(culled.stats, clean.stats);

        let directory = std::env::temp_dir();
        let a = directory.join("growforge_island_culled.stl");
        let b = directory.join("growforge_island_clean.stl");
        stl::write(&a, &culled.mesh).expect("write");
        stl::write(&b, &clean.mesh).expect("write");
        assert_eq!(
            std::fs::read(&a).expect("culled STL"),
            std::fs::read(&b).expect("clean STL")
        );
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();
    }

    #[test]
    fn the_export_lattice_count_follows_the_cube_of_the_factor() {
        let grid = solid_grid(4, 1.0);
        let pad = 2 * constants::FIELD_PADDING_CELLS;
        let side = grid.nnx() + pad;
        assert_eq!(supersampled_node_count(&grid, 1), side.pow(3));
        assert_eq!(
            supersampled_node_count(&grid, 2),
            ((side - 1) * 2 + 1).pow(3)
        );
        // A factor of zero is the unrefined lattice, never an empty one.
        assert_eq!(supersampled_node_count(&grid, 0), side.pow(3));
        // Nothing wraps, however absurd the request.
        assert_eq!(supersampled_node_count(&grid, usize::MAX), usize::MAX);
    }

    /// A closed box, two triangles per face, wound counter-clockwise seen from
    /// outside.
    fn box_mesh(min: Vec3, max: Vec3) -> Mesh {
        let corner = |c: usize| {
            [
                if c & 1 == 0 { min[0] } else { max[0] },
                if c & 2 == 0 { min[1] } else { max[1] },
                if c & 4 == 0 { min[2] } else { max[2] },
            ]
        };
        Mesh {
            vertices: (0..8).map(corner).collect(),
            triangles: vec![
                [0, 3, 1],
                [0, 2, 3],
                [4, 5, 7],
                [4, 7, 6],
                [0, 1, 5],
                [0, 5, 4],
                [2, 7, 3],
                [2, 6, 7],
                [0, 4, 6],
                [0, 6, 2],
                [1, 3, 7],
                [1, 7, 5],
            ],
        }
    }

    #[test]
    fn vertex_normals_average_the_faces_that_meet_at_a_corner() {
        for (min, max) in [
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            // Stretched, so the faces meeting at a corner carry very different
            // areas and weighting by area is not weighting by triangle count.
            ([0.0, 0.0, 0.0], [8.0, 1.0, 1.0]),
        ] {
            let mesh = box_mesh(min, max);
            validate::validate(&mesh).expect("the fixture must be a closed box");
            let normals = vertex_normals(&mesh);
            assert_eq!(normals.len(), mesh.vertices.len());

            // Independently: the unit normal of each incident triangle, scaled
            // by that triangle's area, summed and normalized.
            let mut expected = vec![[0.0f64; 3]; mesh.vertices.len()];
            for triangle in &mesh.triangles {
                let c = triangle.map(|index| mesh.vertices[index as usize]);
                let raw = cross(difference(c[1], c[0]), difference(c[2], c[0]));
                let area = 0.5 * length(raw);
                let unit = scale(raw, 1.0 / length(raw));
                for &index in triangle.iter() {
                    for d in 0..3 {
                        expected[index as usize][d] += area * unit[d];
                    }
                }
            }

            for (index, vertex) in mesh.vertices.iter().enumerate() {
                let normal = normals[index];
                assert!(
                    (length(normal) - 1.0).abs() < 1e-12,
                    "normal {normal:?} is not normalized"
                );
                let want = scale(expected[index], 1.0 / length(expected[index]));
                for d in 0..3 {
                    assert!(
                        (normal[d] - want[d]).abs() < 1e-12,
                        "corner {vertex:?} got {normal:?} against {want:?}"
                    );
                    // And whatever the weights, a corner normal points out of
                    // the octant that corner sits in.
                    let outward = if vertex[d] > 0.5 * (min[d] + max[d]) {
                        1.0
                    } else {
                        -1.0
                    };
                    assert!(
                        normal[d] * outward > 0.0,
                        "corner {vertex:?} got {normal:?}, which points inwards on axis {d}"
                    );
                }
            }

            // The corner at the origin is the diagonal vertex of all three of
            // its faces, so it collects them in equal proportion: for the cube
            // that is the body diagonal exactly.
            if min == [0.0; 3] && max == [1.0; 3] {
                let unit = -1.0 / 3.0f64.sqrt();
                for d in 0..3 {
                    assert!((normals[0][d] - unit).abs() < 1e-12, "{:?}", normals[0]);
                }
            } else {
                // On the stretched box the same corner leans towards the two
                // large faces, which is the area weighting doing its job: a
                // count weighted average would have stayed on the diagonal.
                assert!(
                    normals[0][0].abs() < 0.2 && normals[0][1] < -0.6,
                    "{:?} does not lean towards the large faces",
                    normals[0]
                );
            }
        }
    }

    #[test]
    fn vertex_normals_of_a_sphere_point_outwards() {
        let radius = 10.0;
        let mut mesh = extract_at(&sphere_field(radius, 1.0), 0.0, 1);
        smooth::taubin(&mut mesh, 4);
        let normals = vertex_normals(&mesh);
        let mut worst = 1.0f64;
        for (vertex, normal) in mesh.vertices.iter().zip(normals.iter()) {
            let r = length(*vertex);
            let radial = scale(*vertex, 1.0 / r);
            let alignment = (0..3).map(|d| radial[d] * normal[d]).sum::<f64>();
            worst = worst.min(alignment);
        }
        assert!(
            worst > 0.97,
            "the least outward vertex normal only reaches {worst:.4}"
        );

        // A degenerate mesh leaves zero vectors rather than NaNs, which is what
        // tells a renderer to fall back on the face normal.
        let flat = Mesh {
            vertices: vec![[0.0; 3], [1.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]],
            triangles: vec![[0, 1, 2]],
        };
        for normal in vertex_normals(&flat) {
            assert_eq!(normal, [0.0; 3]);
        }
    }
}
