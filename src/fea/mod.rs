//! Finite element model: element stiffness, the matrix-free global operator
//! and the iterative solver.

pub mod backend;
pub mod cancel;
pub mod element;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod gravity;
pub mod solver;

use std::marker::PhantomData;

use rayon::prelude::*;

use crate::constants::{DOF_PER_ELEMENT, DOF_PER_NODE, ELEMENT_COLORS, NODES_PER_ELEMENT};
use crate::grid::Grid;

pub use backend::{BoundSolver, LinearSolver};
pub use cancel::{CancelProbe, SolveLimits};
pub use element::{
    Ke, StressMatrix, element_energy, element_stress, hex8_centroid_stress, hex8_stiffness,
    von_mises,
};
pub use solver::{CgSolution, Solve, norm, project};

/// Shared mutable view of a scatter target.
///
/// The colouring below guarantees that, inside one colour, no two elements
/// touch the same node, so the concurrent read-modify-write of distinct indices
/// is data-race free. Rust cannot express that invariant in the type system, so
/// it is enforced by construction here and asserted by
/// `grid::tests::colors_never_share_nodes`.
struct ScatterView<'a> {
    ptr: *mut f64,
    len: usize,
    _marker: PhantomData<&'a mut [f64]>,
}

// SAFETY: every concurrent user of a `ScatterView` writes to a disjoint set of
// indices; see the type documentation.
unsafe impl Send for ScatterView<'_> {}
// SAFETY: see above.
unsafe impl Sync for ScatterView<'_> {}

impl<'a> ScatterView<'a> {
    fn new(target: &'a mut [f64]) -> Self {
        ScatterView {
            ptr: target.as_mut_ptr(),
            len: target.len(),
            _marker: PhantomData,
        }
    }

    /// Add `value` to entry `index`.
    ///
    /// # Safety
    /// No two threads may call this concurrently with the same `index`.
    unsafe fn add(&self, index: usize, value: f64) {
        debug_assert!(index < self.len);
        // SAFETY: `index` is in bounds and unique to the calling thread within
        // the current colour.
        unsafe {
            *self.ptr.add(index) += value;
        }
    }
}

/// Run `body(i, j, k, element_index)` over every element, colour by colour.
///
/// The eight colours are the parity classes of `(i, j, k)`. Colours run
/// sequentially, elements inside a colour run in parallel.
fn colored_element_loop<F>(grid: &Grid, body: F)
where
    F: Fn(usize, usize, usize, usize) + Sync + Send,
{
    for color in 0..ELEMENT_COLORS {
        let ci = color & 1;
        let cj = (color >> 1) & 1;
        let ck = (color >> 2) & 1;
        if ci >= grid.nx || cj >= grid.ny || ck >= grid.nz {
            continue;
        }
        let slices: Vec<usize> = (ck..grid.nz).step_by(2).collect();
        slices.into_par_iter().for_each(|k| {
            for j in (cj..grid.ny).step_by(2) {
                for i in (ci..grid.nx).step_by(2) {
                    body(i, j, k, grid.cell_index(i, j, k));
                }
            }
        });
    }
}

/// Gather the 24 element displacements of one element out of a global vector.
pub fn gather(u: &[f64], nodes: &[usize; NODES_PER_ELEMENT]) -> [f64; DOF_PER_ELEMENT] {
    let mut ue = [0.0; DOF_PER_ELEMENT];
    for (l, &n) in nodes.iter().enumerate() {
        let base = n * DOF_PER_NODE;
        for d in 0..DOF_PER_NODE {
            ue[l * DOF_PER_NODE + d] = u[base + d];
        }
    }
    ue
}

/// The global stiffness operator, applied without ever assembling a matrix.
///
/// `moduli[e]` is the effective Young's modulus of element `e` in MPa; void
/// elements carry zero and are skipped.
pub struct StiffnessOperator<'a> {
    grid: &'a Grid,
    ke0: &'a Ke,
    moduli: &'a [f64],
}

impl<'a> StiffnessOperator<'a> {
    /// Bind an operator to a grid, a unit stiffness matrix and per-element
    /// moduli.
    pub fn new(grid: &'a Grid, ke0: &'a Ke, moduli: &'a [f64]) -> Self {
        assert_eq!(
            moduli.len(),
            grid.n_cells(),
            "modulus array must have one entry per cell"
        );
        StiffnessOperator { grid, ke0, moduli }
    }

    /// Compute `out = K u`.
    pub fn apply(&self, u: &[f64], out: &mut [f64]) {
        debug_assert_eq!(u.len(), self.grid.n_dof());
        debug_assert_eq!(out.len(), self.grid.n_dof());
        out.fill(0.0);
        let view = ScatterView::new(out);
        colored_element_loop(self.grid, |i, j, k, e| {
            let modulus = self.moduli[e];
            if modulus == 0.0 {
                return;
            }
            let nodes = self.grid.element_nodes(i, j, k);
            let ue = gather(u, &nodes);
            for (l, &n) in nodes.iter().enumerate() {
                let base = n * DOF_PER_NODE;
                for d in 0..DOF_PER_NODE {
                    let row = &self.ke0[l * DOF_PER_NODE + d];
                    let mut acc = 0.0;
                    for c in 0..DOF_PER_ELEMENT {
                        acc += row[c] * ue[c];
                    }
                    // SAFETY: elements of one colour touch disjoint nodes.
                    unsafe { view.add(base + d, modulus * acc) };
                }
            }
        });
    }

    /// Diagonal of `K`, used by the Jacobi preconditioner.
    pub fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0; self.grid.n_dof()];
        {
            let view = ScatterView::new(&mut diag);
            colored_element_loop(self.grid, |i, j, k, e| {
                let modulus = self.moduli[e];
                if modulus == 0.0 {
                    return;
                }
                let nodes = self.grid.element_nodes(i, j, k);
                for (l, &n) in nodes.iter().enumerate() {
                    let base = n * DOF_PER_NODE;
                    for d in 0..DOF_PER_NODE {
                        let local = l * DOF_PER_NODE + d;
                        // SAFETY: elements of one colour touch disjoint nodes.
                        unsafe { view.add(base + d, modulus * self.ke0[local][local]) };
                    }
                }
            });
        }
        diag
    }

    /// Per-element `u^T ke0 u` for the unit stiffness matrix, which is the
    /// strain energy of the element divided by its modulus.
    pub fn unit_strain_energies(&self, u: &[f64], out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.grid.n_cells());
        let grid = self.grid;
        let ke0 = self.ke0;
        out.par_iter_mut().enumerate().for_each(|(e, slot)| {
            let (i, j, k) = grid.cell_ijk(e);
            let nodes = grid.element_nodes(i, j, k);
            let ue = gather(u, &nodes);
            *slot = element_energy(ke0, &ue);
        });
    }

    /// Number of degrees of freedom of the model.
    pub fn n_dof(&self) -> usize {
        self.grid.n_dof()
    }

    /// The grid the operator is bound to.
    pub fn grid(&self) -> &Grid {
        self.grid
    }

    /// The unit stiffness matrix of one element.
    pub fn ke0(&self) -> &Ke {
        self.ke0
    }

    /// Effective Young's modulus of every cell in MPa, in grid element order.
    ///
    /// The design dependent half of the operator: a backend that keeps the
    /// operator somewhere other than this process's memory re-uploads exactly
    /// this when the design moves.
    pub fn moduli(&self) -> &[f64] {
        self.moduli
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;
    use crate::grid::CellKind;

    fn single_element_grid(h: f64) -> Grid {
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [h, h, h],
            },
            h,
        );
        grid.cells[0] = CellKind::Solid;
        grid
    }

    #[test]
    fn operator_reproduces_the_element_matrix() {
        let h = 2.0;
        let grid = single_element_grid(h);
        let ke0 = hex8_stiffness(0.3, h);
        let moduli = vec![1.0];
        let op = StiffnessOperator::new(&grid, &ke0, &moduli);

        // Column j of K equals K applied to the j-th unit vector; for a single
        // element the global and local orderings coincide.
        for j in 0..DOF_PER_ELEMENT {
            let mut u = vec![0.0; grid.n_dof()];
            u[j] = 1.0;
            let mut out = vec![0.0; grid.n_dof()];
            op.apply(&u, &mut out);
            for i in 0..DOF_PER_ELEMENT {
                assert!((out[i] - ke0[i][j]).abs() < 1e-12, "entry ({i}, {j})");
            }
        }
    }

    #[test]
    fn diagonal_matches_the_operator() {
        let h = 1.5;
        let grid = single_element_grid(h);
        let ke0 = hex8_stiffness(0.3, h);
        let moduli = vec![2.5];
        let op = StiffnessOperator::new(&grid, &ke0, &moduli);
        let diag = op.diagonal();
        for i in 0..DOF_PER_ELEMENT {
            let mut u = vec![0.0; grid.n_dof()];
            u[i] = 1.0;
            let mut out = vec![0.0; grid.n_dof()];
            op.apply(&u, &mut out);
            assert!((diag[i] - out[i]).abs() < 1e-12, "diagonal entry {i}");
        }
    }

    #[test]
    fn multi_element_operator_is_symmetric() {
        let h = 1.0;
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [3.0, 2.0, 2.0],
            },
            h,
        );
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Solid;
        }
        let ke0 = hex8_stiffness(0.3, h);
        let moduli: Vec<f64> = (0..grid.n_cells()).map(|e| 1.0 + e as f64).collect();
        let op = StiffnessOperator::new(&grid, &ke0, &moduli);

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let a: Vec<f64> = (0..grid.n_dof()).map(|_| rand()).collect();
        let b: Vec<f64> = (0..grid.n_dof()).map(|_| rand()).collect();
        let mut ka = vec![0.0; grid.n_dof()];
        let mut kb = vec![0.0; grid.n_dof()];
        op.apply(&a, &mut ka);
        op.apply(&b, &mut kb);
        let lhs: f64 = b.iter().zip(ka.iter()).map(|(x, y)| x * y).sum();
        let rhs: f64 = a.iter().zip(kb.iter()).map(|(x, y)| x * y).sum();
        assert!((lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0));
    }

    #[test]
    fn void_elements_contribute_nothing() {
        let h = 1.0;
        let mut grid = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [2.0, 1.0, 1.0],
            },
            h,
        );
        grid.cells[0] = CellKind::Solid;
        grid.cells[1] = CellKind::Void;
        let ke0 = hex8_stiffness(0.3, h);
        let moduli = vec![1.0, 0.0];
        let op = StiffnessOperator::new(&grid, &ke0, &moduli);
        let diag = op.diagonal();
        // Nodes at i = 2 belong only to the void element.
        for j in 0..grid.nny() {
            for k in 0..grid.nnz() {
                let n = grid.node_index(2, j, k);
                for d in 0..DOF_PER_NODE {
                    assert_eq!(diag[n * DOF_PER_NODE + d], 0.0);
                }
            }
        }
    }
}
