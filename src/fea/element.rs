//! Trilinear 8-node hexahedral element for linear isotropic elasticity.
//!
//! The element stiffness is integrated with the full 2x2x2 Gauss rule over a
//! cube of edge `h`. Because the geometry is a cube, the Jacobian is the
//! constant `h/2` times the identity, but the integration is written in the
//! general form so that a future non-uniform grid only has to supply node
//! coordinates.
//!
//! Local node `l` sits at offset `(l & 1, (l >> 1) & 1, (l >> 2) & 1)` of the
//! cell, matching [`crate::grid::Grid::element_nodes`]. Local degree of freedom
//! `3 * l + d` is the displacement of node `l` along axis `d`.

use crate::constants::{
    DOF_PER_ELEMENT, GAUSS_POINTS_PER_AXIS, NODES_PER_ELEMENT, STRESS_COMPONENTS,
};

/// A 24x24 element stiffness matrix.
pub type Ke = [[f64; DOF_PER_ELEMENT]; DOF_PER_ELEMENT];

/// The 6x24 matrix mapping element displacements to the stress at the element
/// centroid, for a unit Young's modulus.
pub type StressMatrix = [[f64; DOF_PER_ELEMENT]; STRESS_COMPONENTS];

/// Number of independent strain components in 3D.
const STRAIN_COMPONENTS: usize = STRESS_COMPONENTS;

/// Gauss point coordinates and weights of the two point rule on [-1, 1].
fn gauss_rule() -> ([f64; GAUSS_POINTS_PER_AXIS], [f64; GAUSS_POINTS_PER_AXIS]) {
    let a = 1.0 / 3.0f64.sqrt();
    ([-a, a], [1.0, 1.0])
}

/// Natural coordinates of local node `l`, each entry -1 or +1.
fn node_natural_coords(l: usize) -> [f64; 3] {
    [
        2.0 * (l & 1) as f64 - 1.0,
        2.0 * ((l >> 1) & 1) as f64 - 1.0,
        2.0 * ((l >> 2) & 1) as f64 - 1.0,
    ]
}

/// Isotropic elasticity matrix for unit Young's modulus.
///
/// Strain and stress are ordered `[xx, yy, zz, xy, yz, zx]`, with engineering
/// shear strains.
fn elasticity_matrix(nu: f64) -> [[f64; STRAIN_COMPONENTS]; STRAIN_COMPONENTS] {
    let mut d = [[0.0; STRAIN_COMPONENTS]; STRAIN_COMPONENTS];
    let c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let lambda = c * nu;
    let diag = c * (1.0 - nu);
    let shear = c * (1.0 - 2.0 * nu) / 2.0;
    for (i, row) in d.iter_mut().enumerate().take(3) {
        for (j, entry) in row.iter_mut().enumerate().take(3) {
            *entry = if i == j { diag } else { lambda };
        }
    }
    for i in 0..3 {
        d[3 + i][3 + i] = shear;
    }
    d
}

/// Strain-displacement matrix at a natural coordinate, for a cube of edge `h`.
fn strain_displacement(
    xi: f64,
    eta: f64,
    zeta: f64,
    h: f64,
) -> [[f64; DOF_PER_ELEMENT]; STRAIN_COMPONENTS] {
    // The Jacobian of the map from natural to physical coordinates is
    // (h/2) * I, so d/dx = (2/h) d/dxi.
    let inv_j = 2.0 / h;
    let mut b = [[0.0; DOF_PER_ELEMENT]; STRAIN_COMPONENTS];
    for l in 0..NODES_PER_ELEMENT {
        let n = node_natural_coords(l);
        let dn = [
            0.125 * n[0] * (1.0 + n[1] * eta) * (1.0 + n[2] * zeta) * inv_j,
            0.125 * (1.0 + n[0] * xi) * n[1] * (1.0 + n[2] * zeta) * inv_j,
            0.125 * (1.0 + n[0] * xi) * (1.0 + n[1] * eta) * n[2] * inv_j,
        ];
        let cx = 3 * l;
        b[0][cx] = dn[0];
        b[1][cx + 1] = dn[1];
        b[2][cx + 2] = dn[2];
        // gamma_xy
        b[3][cx] = dn[1];
        b[3][cx + 1] = dn[0];
        // gamma_yz
        b[4][cx + 1] = dn[2];
        b[4][cx + 2] = dn[1];
        // gamma_zx
        b[5][cx] = dn[2];
        b[5][cx + 2] = dn[0];
    }
    b
}

/// Stiffness matrix of a cubic 8-node hexahedron of edge `h` for unit Young's
/// modulus and Poisson's ratio `nu`.
///
/// Scaling: `B` carries a factor `1/h` and the volume element a factor `h^3`,
/// so the result is linear in `h`.
pub fn hex8_stiffness(nu: f64, h: f64) -> Ke {
    let d = elasticity_matrix(nu);
    let (points, weights) = gauss_rule();
    let det_j = (h / 2.0).powi(3);
    let mut ke = [[0.0; DOF_PER_ELEMENT]; DOF_PER_ELEMENT];

    for (gx, &xi) in points.iter().enumerate() {
        for (gy, &eta) in points.iter().enumerate() {
            for (gz, &zeta) in points.iter().enumerate() {
                let w = weights[gx] * weights[gy] * weights[gz] * det_j;
                let b = strain_displacement(xi, eta, zeta, h);
                // db = D * B, then ke += B^T * db * w.
                let mut db = [[0.0; DOF_PER_ELEMENT]; STRAIN_COMPONENTS];
                for r in 0..STRAIN_COMPONENTS {
                    for c in 0..DOF_PER_ELEMENT {
                        let mut acc = 0.0;
                        for s in 0..STRAIN_COMPONENTS {
                            acc += d[r][s] * b[s][c];
                        }
                        db[r][c] = acc;
                    }
                }
                for i in 0..DOF_PER_ELEMENT {
                    for j in 0..DOF_PER_ELEMENT {
                        let mut acc = 0.0;
                        for s in 0..STRAIN_COMPONENTS {
                            acc += b[s][i] * db[s][j];
                        }
                        ke[i][j] += w * acc;
                    }
                }
            }
        }
    }

    // The integration is symmetric analytically; average the two triangles so
    // it is symmetric to the last bit numerically as well.
    for i in 0..DOF_PER_ELEMENT {
        let (head, tail) = ke.split_at_mut(i + 1);
        let row = &mut head[i];
        for (offset, mirrored) in tail.iter_mut().enumerate() {
            let j = i + 1 + offset;
            let mean = 0.5 * (row[j] + mirrored[i]);
            row[j] = mean;
            mirrored[i] = mean;
        }
    }
    ke
}

/// Matrix mapping the 24 element displacements to the stress at the element
/// centroid, for unit Young's modulus and Poisson's ratio `nu`.
///
/// It is `D B` evaluated at the natural origin, so multiplying by a modulus in
/// MPa and by displacements in mm gives a stress in MPa. Components come out in
/// the usual `[xx, yy, zz, xy, yz, zx]` order.
pub fn hex8_centroid_stress(nu: f64, h: f64) -> StressMatrix {
    let d = elasticity_matrix(nu);
    let b = strain_displacement(0.0, 0.0, 0.0, h);
    let mut out = [[0.0; DOF_PER_ELEMENT]; STRESS_COMPONENTS];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, entry) in row.iter_mut().enumerate() {
            *entry = (0..STRAIN_COMPONENTS).map(|s| d[r][s] * b[s][c]).sum();
        }
    }
    out
}

/// Stress at an element centroid in MPa, for a Young's modulus of `modulus`.
pub fn element_stress(
    matrix: &StressMatrix,
    modulus: f64,
    u: &[f64; DOF_PER_ELEMENT],
) -> [f64; STRESS_COMPONENTS] {
    let mut out = [0.0; STRESS_COMPONENTS];
    for (row, slot) in matrix.iter().zip(out.iter_mut()) {
        *slot = modulus * row.iter().zip(u.iter()).map(|(m, d)| m * d).sum::<f64>();
    }
    out
}

/// Von Mises equivalent stress of a `[xx, yy, zz, xy, yz, zx]` stress vector.
pub fn von_mises(stress: &[f64; STRESS_COMPONENTS]) -> f64 {
    let (sx, sy, sz) = (stress[0], stress[1], stress[2]);
    let (txy, tyz, tzx) = (stress[3], stress[4], stress[5]);
    (0.5 * ((sx - sy).powi(2) + (sy - sz).powi(2) + (sz - sx).powi(2))
        + 3.0 * (txy * txy + tyz * tyz + tzx * tzx))
        .sqrt()
}

/// Strain energy density factor `u^T ke u` for one element.
pub fn element_energy(ke: &Ke, u: &[f64; DOF_PER_ELEMENT]) -> f64 {
    let mut acc = 0.0;
    for i in 0..DOF_PER_ELEMENT {
        let mut row = 0.0;
        for j in 0..DOF_PER_ELEMENT {
            row += ke[i][j] * u[j];
        }
        acc += u[i] * row;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec(ke: &Ke, v: &[f64; DOF_PER_ELEMENT]) -> [f64; DOF_PER_ELEMENT] {
        let mut out = [0.0; DOF_PER_ELEMENT];
        for (row, slot) in ke.iter().zip(out.iter_mut()) {
            *slot = row.iter().zip(v.iter()).map(|(k, x)| k * x).sum();
        }
        out
    }

    fn max_abs(v: &[f64; DOF_PER_ELEMENT]) -> f64 {
        v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
    }

    fn scale_of(ke: &Ke) -> f64 {
        let mut m: f64 = 0.0;
        for row in ke.iter() {
            for x in row.iter() {
                m = m.max(x.abs());
            }
        }
        m
    }

    #[test]
    fn stiffness_is_symmetric_with_a_positive_diagonal() {
        let ke = hex8_stiffness(0.3, 2.0);
        for (i, row) in ke.iter().enumerate() {
            assert!(row[i] > 0.0, "diagonal entry {i} is not positive");
            for (j, entry) in row.iter().enumerate() {
                assert_eq!(*entry, ke[j][i], "asymmetry at ({i}, {j})");
            }
        }
    }

    #[test]
    fn rigid_body_modes_are_in_the_null_space() {
        let h = 2.0;
        let ke = hex8_stiffness(0.3, h);
        let tol = 1e-12 * scale_of(&ke);

        // Three translations.
        for axis in 0..3 {
            let mut v = [0.0; DOF_PER_ELEMENT];
            for l in 0..NODES_PER_ELEMENT {
                v[3 * l + axis] = 1.0;
            }
            assert!(
                max_abs(&matvec(&ke, &v)) < tol,
                "translation along axis {axis} is not a rigid body mode"
            );
        }

        // Three linearised rotations about the element centre.
        let coords: Vec<[f64; 3]> = (0..NODES_PER_ELEMENT)
            .map(|l| {
                let n = node_natural_coords(l);
                [n[0] * h / 2.0, n[1] * h / 2.0, n[2] * h / 2.0]
            })
            .collect();
        for axis in 0..3 {
            let mut omega = [0.0; 3];
            omega[axis] = 1.0;
            let mut v = [0.0; DOF_PER_ELEMENT];
            for (l, c) in coords.iter().enumerate() {
                let d = crate::geometry::cross(omega, *c);
                v[3 * l] = d[0];
                v[3 * l + 1] = d[1];
                v[3 * l + 2] = d[2];
            }
            assert!(
                max_abs(&matvec(&ke, &v)) < tol,
                "rotation about axis {axis} is not a rigid body mode"
            );
        }
    }

    #[test]
    fn stiffness_scales_linearly_with_element_size() {
        let a = hex8_stiffness(0.3, 1.0);
        let b = hex8_stiffness(0.3, 3.0);
        for i in 0..DOF_PER_ELEMENT {
            for j in 0..DOF_PER_ELEMENT {
                assert!(
                    (b[i][j] - 3.0 * a[i][j]).abs() < 1e-10 * scale_of(&b).max(1.0),
                    "entry ({i}, {j}) does not scale linearly with h"
                );
            }
        }
    }

    #[test]
    fn centroid_stress_reproduces_a_uniform_uniaxial_state() {
        let h = 2.0;
        let nu = 0.3;
        let modulus = 2100.0;
        let strain = 1e-3;
        let matrix = hex8_centroid_stress(nu, h);

        // A uniform axial stretch with the matching lateral contraction is a
        // constant strain state, which a trilinear hex represents exactly.
        let mut u = [0.0; DOF_PER_ELEMENT];
        for l in 0..NODES_PER_ELEMENT {
            let n = node_natural_coords(l);
            let position = [n[0] * h / 2.0, n[1] * h / 2.0, n[2] * h / 2.0];
            u[3 * l] = strain * position[0];
            u[3 * l + 1] = -nu * strain * position[1];
            u[3 * l + 2] = -nu * strain * position[2];
        }
        let stress = element_stress(&matrix, modulus, &u);
        let expected = modulus * strain;
        assert!(
            (stress[0] - expected).abs() < 1e-9 * expected,
            "axial stress {} is not {expected}",
            stress[0]
        );
        for component in stress.iter().skip(1) {
            assert!(component.abs() < 1e-9 * expected, "spurious {component}");
        }
        // Uniaxial stress: the von Mises equivalent is the axial stress itself.
        assert!((von_mises(&stress) - expected).abs() < 1e-9 * expected);
    }

    #[test]
    fn von_mises_vanishes_under_hydrostatic_pressure_and_scales_with_shear() {
        assert_eq!(von_mises(&[7.0, 7.0, 7.0, 0.0, 0.0, 0.0]), 0.0);
        // Pure shear tau has a von Mises equivalent of sqrt(3) tau.
        let tau = 5.0;
        let value = von_mises(&[0.0, 0.0, 0.0, tau, 0.0, 0.0]);
        assert!((value - 3.0f64.sqrt() * tau).abs() < 1e-12);
        assert_eq!(von_mises(&[0.0; STRESS_COMPONENTS]), 0.0);
    }

    #[test]
    fn stiffness_is_positive_semidefinite_on_random_vectors() {
        let ke = hex8_stiffness(0.33, 1.5);
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..64 {
            let mut v = [0.0; DOF_PER_ELEMENT];
            for entry in v.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *entry = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            }
            assert!(element_energy(&ke, &v) >= -1e-12 * scale_of(&ke));
        }
    }
}
