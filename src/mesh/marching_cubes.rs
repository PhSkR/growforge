//! Marching cubes isosurface extraction with exact vertex welding.
//!
//! Vertices are keyed by the global lattice edge they sit on, never by their
//! floating point position, so the two cubes that share an edge always share
//! the identical vertex index and the surface comes out closed by construction.

use anyhow::{Result, bail};

use crate::constants;
use crate::mesh::tables::{CORNER_OFFSET, EDGE_CORNERS, TRI_TABLE};
use crate::mesh::{Mesh, ScalarField};

/// Number of lattice edges that start at a given lattice point.
const EDGES_PER_POINT: usize = 3;

/// Identity of a cube edge in the global lattice: the axis it runs along and
/// the corner offset of its lower end.
#[derive(Debug, Clone, Copy)]
struct EdgeIdentity {
    axis: usize,
    lower: [usize; 3],
}

/// Derive the lattice identity of each of the twelve cube edges from the corner
/// and edge tables, so the two tables stay the single source of truth.
fn edge_identities() -> [EdgeIdentity; 12] {
    let mut out = [EdgeIdentity {
        axis: 0,
        lower: [0, 0, 0],
    }; 12];
    for (e, slot) in out.iter_mut().enumerate() {
        let [a, b] = EDGE_CORNERS[e];
        let oa = CORNER_OFFSET[a];
        let ob = CORNER_OFFSET[b];
        let axis = (0..3)
            .find(|&d| oa[d] != ob[d])
            .expect("a cube edge must differ on exactly one axis");
        *slot = EdgeIdentity {
            axis,
            lower: if oa[axis] < ob[axis] { oa } else { ob },
        };
    }
    out
}

/// Extract the `iso` level set of `field`.
///
/// Fails only when the surface would need more vertices than a `u32` index can
/// address, which no realistic grid reaches.
pub fn extract(field: &ScalarField, iso: f64) -> Result<Mesh> {
    let mut mesh = Mesh::default();
    if field.nx < 2 || field.ny < 2 || field.nz < 2 {
        return Ok(mesh);
    }
    let identities = edge_identities();
    let mut vertex_of_edge = vec![u32::MAX; EDGES_PER_POINT * field.len()];

    for k in 0..field.nz - 1 {
        for j in 0..field.ny - 1 {
            for i in 0..field.nx - 1 {
                let mut values = [0.0f64; 8];
                let mut cube = 0usize;
                for (c, offset) in CORNER_OFFSET.iter().enumerate() {
                    let v = field.at(i + offset[0], j + offset[1], k + offset[2]);
                    values[c] = v;
                    // Bourke's convention: a corner below the iso level is
                    // outside the material and sets its bit.
                    if v < iso {
                        cube |= 1 << c;
                    }
                }

                let row = &TRI_TABLE[cube];
                let mut t = 0;
                while t + 2 < row.len() && row[t] >= 0 {
                    let mut triangle = [0u32; 3];
                    for (m, slot) in triangle.iter_mut().enumerate() {
                        let edge = row[t + m] as usize;
                        *slot = vertex_for_edge(
                            field,
                            iso,
                            &identities,
                            &values,
                            (i, j, k),
                            edge,
                            &mut vertex_of_edge,
                            &mut mesh.vertices,
                        )?;
                    }
                    mesh.triangles.push(triangle);
                    t += 3;
                }
            }
        }
    }

    Ok(mesh)
}

/// Look up, or create, the vertex sitting on cube edge `edge`.
#[allow(clippy::too_many_arguments)]
fn vertex_for_edge(
    field: &ScalarField,
    iso: f64,
    identities: &[EdgeIdentity; 12],
    values: &[f64; 8],
    cell: (usize, usize, usize),
    edge: usize,
    vertex_of_edge: &mut [u32],
    vertices: &mut Vec<[f64; 3]>,
) -> Result<u32> {
    let (i, j, k) = cell;
    let identity = identities[edge];
    let point = field.index(
        i + identity.lower[0],
        j + identity.lower[1],
        k + identity.lower[2],
    );
    let key = EDGES_PER_POINT * point + identity.axis;
    if vertex_of_edge[key] != u32::MAX {
        return Ok(vertex_of_edge[key]);
    }
    if vertices.len() >= constants::MAX_MESH_VERTICES {
        bail!(
            "the isosurface needs more than {} vertices, which a 32 bit vertex index cannot address; use a coarser voxel_size_mm",
            constants::MAX_MESH_VERTICES
        );
    }

    let [a, b] = EDGE_CORNERS[edge];
    let oa = CORNER_OFFSET[a];
    let ob = CORNER_OFFSET[b];
    let pa = field.position(i + oa[0], j + oa[1], k + oa[2]);
    let pb = field.position(i + ob[0], j + ob[1], k + ob[2]);
    let va = values[a];
    let vb = values[b];
    let denominator = vb - va;
    let t = if denominator == 0.0 {
        0.5
    } else {
        ((iso - va) / denominator).clamp(0.0, 1.0)
    };
    let position = [
        pa[0] + t * (pb[0] - pa[0]),
        pa[1] + t * (pb[1] - pa[1]),
        pa[2] + t * (pb[2] - pa[2]),
    ];

    let index = vertices.len() as u32;
    vertices.push(position);
    vertex_of_edge[key] = index;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::validate;
    use std::collections::HashSet;
    use std::f64::consts::PI;

    /// Deterministic xorshift generator, so the tests never depend on a crate
    /// or on the platform's entropy.
    struct Rng(u64);

    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn cube_index(field: &ScalarField, i: usize, j: usize, k: usize, iso: f64) -> usize {
        let mut cube = 0usize;
        for (c, offset) in CORNER_OFFSET.iter().enumerate() {
            if field.at(i + offset[0], j + offset[1], k + offset[2]) < iso {
                cube |= 1 << c;
            }
        }
        cube
    }

    #[test]
    fn table_is_watertight_over_all_256_cases() {
        let n = 22;
        let iso = 0.5;
        let mut field = ScalarField::new(n, n, n, [0.0, 0.0, 0.0], 1.0);
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let border =
                        i == 0 || j == 0 || k == 0 || i == n - 1 || j == n - 1 || k == n - 1;
                    let index = field.index(i, j, k);
                    field.values[index] = if border { 0.0 } else { rng.next_f64() };
                }
            }
        }
        field.snap_away_from(iso);

        let mut seen = HashSet::new();
        for k in 0..n - 1 {
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    seen.insert(cube_index(&field, i, j, k, iso));
                }
            }
        }
        assert_eq!(
            seen.len(),
            256,
            "the random field only exercised {} of the 256 cases",
            seen.len()
        );

        let mesh = extract(&field, iso).expect("extraction");
        assert!(!mesh.is_empty());
        validate::validate(&mesh).expect("the extracted surface must be closed and manifold");
    }

    #[test]
    fn sphere_volume_matches_the_analytic_value() {
        let radius = 10.0;
        let spacing = 0.5;
        let half = 15.0;
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

        let mesh = extract(&field, 0.0).expect("extraction");
        let stats = validate::validate(&mesh).expect("closed sphere");
        let expected = 4.0 / 3.0 * PI * radius.powi(3);
        let error = (stats.volume_mm3 - expected).abs() / expected;
        assert!(
            error < 0.01,
            "sphere volume {} differs from {expected} by {:.3}%",
            stats.volume_mm3,
            error * 100.0
        );
        for v in &mesh.vertices {
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(
                (r - radius).abs() < spacing,
                "vertex at radius {r} is off the sphere"
            );
        }
    }

    #[test]
    fn vertices_are_welded_by_edge_identity() {
        // Two neighbouring cubes crossed by the same planar surface share the
        // vertices on their common face.
        let mut field = ScalarField::new(4, 3, 3, [0.0, 0.0, 0.0], 1.0);
        for k in 0..field.nz {
            for j in 0..field.ny {
                for i in 0..field.nx {
                    let index = field.index(i, j, k);
                    // A ramp along z: the 0.5 level sits between k = 0 and 1.
                    field.values[index] = if k == 0 { 0.0 } else { 1.0 };
                }
            }
        }
        let mesh = extract(&field, 0.5).expect("extraction");
        assert!(!mesh.triangles.is_empty());
        // One vertex per vertical lattice edge crossed by the plane.
        assert_eq!(mesh.vertices.len(), field.nx * field.ny);
    }

    #[test]
    fn a_field_entirely_above_or_below_the_iso_level_has_no_surface() {
        let mut field = ScalarField::new(5, 5, 5, [0.0, 0.0, 0.0], 1.0);
        field.values.iter_mut().for_each(|v| *v = 1.0);
        assert!(extract(&field, 0.5).expect("extraction").is_empty());
        field.values.iter_mut().for_each(|v| *v = 0.0);
        assert!(extract(&field, 0.5).expect("extraction").is_empty());
    }
}
