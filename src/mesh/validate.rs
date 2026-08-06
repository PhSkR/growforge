//! Mesh validation and statistics.
//!
//! A mesh is accepted for export only when it is watertight (every edge is
//! shared by exactly two triangles), manifold and consistently wound (those two
//! triangles traverse the edge in opposite directions), free of degenerate
//! triangles and free of non-finite coordinates.

use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::constants;
use crate::geometry::{Aabb, cross, difference, inner};
use crate::mesh::Mesh;

/// Statistics of a validated mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshStats {
    /// Triangle count.
    pub triangles: usize,
    /// Vertex count after welding.
    pub vertices: usize,
    /// Axis aligned bounds in millimetres.
    pub bounds: Aabb,
    /// Enclosed volume in cubic millimetres, from the divergence theorem.
    pub volume_mm3: f64,
}

/// Signed volume enclosed by a closed triangle mesh, positive when the
/// triangles are wound counter-clockwise seen from outside.
pub fn enclosed_volume(mesh: &Mesh) -> f64 {
    let mut total = 0.0;
    for t in &mesh.triangles {
        let a = mesh.vertices[t[0] as usize];
        let b = mesh.vertices[t[1] as usize];
        let c = mesh.vertices[t[2] as usize];
        total += inner(a, cross(b, c));
    }
    total / 6.0
}

/// Area of a triangle in square millimetres.
fn triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let n = cross(difference(b, a), difference(c, a));
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

/// Validate a mesh and compute its statistics.
pub fn validate(mesh: &Mesh) -> Result<MeshStats> {
    if mesh.triangles.is_empty() {
        bail!(
            "the extracted surface is empty: no cell density crosses the iso level, so there is nothing to export"
        );
    }

    if mesh.vertices.len() > constants::MAX_MESH_VERTICES {
        bail!(
            "the mesh holds {} vertices, more than a 32 bit vertex index can address ({})",
            mesh.vertices.len(),
            constants::MAX_MESH_VERTICES
        );
    }
    let n_vertices = mesh.vertices.len() as u32;
    let mut bounds = Aabb::empty();
    for (index, v) in mesh.vertices.iter().enumerate() {
        for d in 0..3 {
            if !v[d].is_finite() {
                bail!("vertex {index} has a non-finite coordinate ({:?})", v);
            }
            bounds.min[d] = bounds.min[d].min(v[d]);
            bounds.max[d] = bounds.max[d].max(v[d]);
        }
    }

    // Every undirected edge must be used exactly twice, once in each direction.
    let mut edges: HashMap<(u32, u32), (u32, i32)> =
        HashMap::with_capacity(mesh.triangles.len() * 3);
    for (index, t) in mesh.triangles.iter().enumerate() {
        for &v in t.iter() {
            if v >= n_vertices {
                bail!("triangle {index} references vertex {v}, which does not exist");
            }
        }
        if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
            bail!("triangle {index} repeats a vertex index: {t:?}");
        }
        let area = triangle_area(
            mesh.vertices[t[0] as usize],
            mesh.vertices[t[1] as usize],
            mesh.vertices[t[2] as usize],
        );
        if area < constants::MIN_TRIANGLE_AREA_MM2 {
            bail!(
                "triangle {index} is degenerate: area {area:e} mm2 is below {:e} mm2",
                constants::MIN_TRIANGLE_AREA_MM2
            );
        }
        for m in 0..3 {
            let a = t[m];
            let b = t[(m + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            let orientation = if a < b { 1 } else { -1 };
            let entry = edges.entry(key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += orientation;
        }
    }

    for (&(a, b), &(count, orientation)) in edges.iter() {
        if count != 2 {
            bail!(
                "the mesh is not watertight: edge ({a}, {b}) is shared by {count} triangles instead of 2"
            );
        }
        if orientation != 0 {
            bail!(
                "the mesh is not consistently wound: edge ({a}, {b}) is traversed twice in the same direction"
            );
        }
    }

    let volume = enclosed_volume(mesh);
    if volume <= 0.0 || volume.is_nan() {
        bail!(
            "the mesh encloses a non-positive volume ({volume} mm3), which means the triangles are wound inside out"
        );
    }

    Ok(MeshStats {
        triangles: mesh.triangles.len(),
        vertices: mesh.vertices.len(),
        bounds,
        volume_mm3: volume,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube centred on the origin, wound counter-clockwise from outside.
    fn cube(size: f64) -> Mesh {
        let h = size / 2.0;
        let vertices = vec![
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ];
        let triangles = vec![
            // -z
            [0, 2, 1],
            [0, 3, 2],
            // +z
            [4, 5, 6],
            [4, 6, 7],
            // -y
            [0, 1, 5],
            [0, 5, 4],
            // +y
            [3, 6, 2],
            [3, 7, 6],
            // -x
            [0, 4, 7],
            [0, 7, 3],
            // +x
            [1, 2, 6],
            [1, 6, 5],
        ];
        Mesh {
            vertices,
            triangles,
        }
    }

    #[test]
    fn a_closed_cube_validates_with_the_right_volume() {
        let mesh = cube(2.0);
        let stats = validate(&mesh).expect("cube");
        assert_eq!(stats.triangles, 12);
        assert!((stats.volume_mm3 - 8.0).abs() < 1e-12);
        assert_eq!(stats.bounds.min, [-1.0, -1.0, -1.0]);
    }

    #[test]
    fn an_open_mesh_is_rejected() {
        let mut mesh = cube(2.0);
        mesh.triangles.pop();
        let err = validate(&mesh).unwrap_err().to_string();
        assert!(err.contains("watertight"), "unexpected error: {err}");
    }

    #[test]
    fn inconsistent_winding_is_rejected() {
        let mut mesh = cube(2.0);
        let last = mesh.triangles.len() - 1;
        mesh.triangles[last].swap(0, 1);
        let err = validate(&mesh).unwrap_err().to_string();
        assert!(err.contains("wound"), "unexpected error: {err}");
    }

    #[test]
    fn inside_out_meshes_are_rejected() {
        let mut mesh = cube(2.0);
        for t in mesh.triangles.iter_mut() {
            t.swap(0, 1);
        }
        let err = validate(&mesh).unwrap_err().to_string();
        assert!(
            err.contains("non-positive volume"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn degenerate_triangles_are_rejected() {
        let mut mesh = cube(2.0);
        mesh.vertices[1] = mesh.vertices[0];
        let err = validate(&mesh).unwrap_err().to_string();
        assert!(err.contains("degenerate"), "unexpected error: {err}");
    }

    #[test]
    fn non_finite_vertices_are_rejected() {
        let mut mesh = cube(2.0);
        mesh.vertices[3][1] = f64::NAN;
        let err = validate(&mesh).unwrap_err().to_string();
        assert!(err.contains("non-finite"), "unexpected error: {err}");
    }

    #[test]
    fn an_empty_mesh_is_rejected() {
        let err = validate(&Mesh::default()).unwrap_err().to_string();
        assert!(err.contains("empty"), "unexpected error: {err}");
    }
}
