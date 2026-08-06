//! Taubin lambda/mu mesh smoothing.
//!
//! One pass is a Laplacian shrinking step with factor `lambda` followed by an
//! expanding step with factor `mu < -lambda`. The pair acts as a low pass
//! filter on the surface signal, so features are smoothed without the volume
//! loss that repeated plain Laplacian smoothing produces. Connectivity is never
//! touched, so a watertight mesh stays watertight.

use rayon::prelude::*;

use crate::constants;
use crate::mesh::Mesh;

/// Neighbour lists for every vertex, built from the triangle edges.
fn adjacency(mesh: &Mesh) -> Vec<Vec<u32>> {
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); mesh.vertices.len()];
    for t in &mesh.triangles {
        for m in 0..3 {
            let a = t[m] as usize;
            let b = t[(m + 1) % 3];
            adjacency[a].push(b);
            adjacency[b as usize].push(t[m]);
        }
    }
    for list in adjacency.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    adjacency
}

/// One umbrella-operator step: move every vertex a fraction `factor` towards
/// the centroid of its neighbours.
fn laplacian_step(vertices: &mut Vec<[f64; 3]>, adjacency: &[Vec<u32>], factor: f64) {
    let updated: Vec<[f64; 3]> = vertices
        .par_iter()
        .enumerate()
        .map(|(index, v)| {
            let neighbours = &adjacency[index];
            if neighbours.is_empty() {
                return *v;
            }
            let mut centroid = [0.0f64; 3];
            for &n in neighbours {
                let p = vertices[n as usize];
                for d in 0..3 {
                    centroid[d] += p[d];
                }
            }
            let inverse = 1.0 / neighbours.len() as f64;
            let mut out = *v;
            for d in 0..3 {
                out[d] += factor * (centroid[d] * inverse - v[d]);
            }
            out
        })
        .collect();
    *vertices = updated;
}

/// Apply `iterations` Taubin passes in place.
pub fn taubin(mesh: &mut Mesh, iterations: usize) {
    if iterations == 0 || mesh.vertices.is_empty() {
        return;
    }
    let adjacency = adjacency(mesh);
    for _ in 0..iterations {
        laplacian_step(&mut mesh.vertices, &adjacency, constants::TAUBIN_LAMBDA);
        laplacian_step(&mut mesh.vertices, &adjacency, constants::TAUBIN_MU);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ScalarField, marching_cubes, validate};

    fn sphere_mesh(radius: f64, spacing: f64) -> Mesh {
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
        marching_cubes::extract(&field, 0.0).expect("extraction")
    }

    #[test]
    fn smoothing_preserves_validity_and_volume() {
        let mesh = sphere_mesh(10.0, 1.0);
        let before = validate::validate(&mesh).expect("sphere");
        let mut smoothed = mesh.clone();
        taubin(&mut smoothed, 10);
        let after = validate::validate(&smoothed).expect("smoothed sphere");
        assert_eq!(before.triangles, after.triangles);
        let shrink = (before.volume_mm3 - after.volume_mm3).abs() / before.volume_mm3;
        assert!(shrink < 0.03, "volume changed by {:.2}%", shrink * 100.0);
    }

    /// Mean squared umbrella operator magnitude: the discrete measure of
    /// surface roughness that Taubin smoothing is designed to shrink.
    fn roughness(mesh: &Mesh) -> f64 {
        let adjacency = adjacency(mesh);
        let mut total = 0.0;
        for (index, v) in mesh.vertices.iter().enumerate() {
            let neighbours = &adjacency[index];
            if neighbours.is_empty() {
                continue;
            }
            let mut centroid = [0.0f64; 3];
            for &n in neighbours {
                let p = mesh.vertices[n as usize];
                for d in 0..3 {
                    centroid[d] += p[d];
                }
            }
            let inverse = 1.0 / neighbours.len() as f64;
            for d in 0..3 {
                let delta = centroid[d] * inverse - v[d];
                total += delta * delta;
            }
        }
        total / mesh.vertices.len() as f64
    }

    #[test]
    fn smoothing_reduces_surface_roughness() {
        let mesh = sphere_mesh(10.0, 1.0);
        let mut smoothed = mesh.clone();
        taubin(&mut smoothed, 10);
        let before = roughness(&mesh);
        let after = roughness(&smoothed);
        assert!(
            after < before,
            "roughness went from {before:e} to {after:e}"
        );
    }

    #[test]
    fn zero_iterations_is_a_no_op() {
        let mesh = sphere_mesh(6.0, 1.5);
        let mut copy = mesh.clone();
        taubin(&mut copy, 0);
        assert_eq!(mesh.vertices, copy.vertices);
    }
}
