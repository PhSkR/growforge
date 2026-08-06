//! Matrix-free preconditioned conjugate gradient solver with Jacobi
//! preconditioning and Dirichlet boundary conditions applied by projection.
//!
//! The constrained degrees of freedom are never eliminated from the system.
//! Instead every vector that enters the Krylov space is projected onto the free
//! subspace (fixed components set to zero), which keeps the operator symmetric
//! positive definite on that subspace and leaves the index arithmetic of the
//! grid untouched.

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::constants;
use crate::fea::SolveLimits;

/// Outcome of one linear solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgSolution {
    /// Number of conjugate gradient iterations performed.
    pub iterations: usize,
    /// Final relative residual ||r|| / ||b||.
    pub relative_residual: f64,
}

/// What one linear solve came back with.
///
/// The three things a solve can do are kept apart deliberately. Meeting the
/// tolerance is `Ok(Solve::Converged)`; failing to - breaking down, or running
/// out of iterations - is `Err`, because the caller asked a question this
/// solver could not answer. Being *called off* is neither: nobody wants the
/// answer any more, which is not a failure of anything and must never be
/// reported as one. See [`crate::fea::CancelProbe`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Solve {
    /// The solve reached the caller's tolerance.
    Converged(CgSolution),
    /// The caller's cancel probe went true part way through.
    ///
    /// The solution vector is left on an iterate nobody asked for: on the CPU
    /// the check runs before the iteration's work, so it is the last *completed*
    /// iteration's, and on the GPU the correction a cancelled refinement pass
    /// was building is dropped rather than folded, so it is the last completed
    /// pass's. Either way it is not a solution of anything the caller asked for
    /// and it is not to be read as one - which is why there are no statistics
    /// attached to say how good it is.
    Cancelled,
}

impl Solve {
    /// The statistics of a solve that converged, or `None` for a cancelled one.
    pub fn converged(self) -> Option<CgSolution> {
        match self {
            Solve::Converged(solution) => Some(solution),
            Solve::Cancelled => None,
        }
    }

    /// True when the caller called this solve off.
    pub fn is_cancelled(self) -> bool {
        matches!(self, Solve::Cancelled)
    }
}

/// Zero the constrained components of `v`.
pub fn project(v: &mut [f64], fixed: &[bool]) {
    v.par_iter_mut().zip(fixed.par_iter()).for_each(|(x, &f)| {
        if f {
            *x = 0.0;
        }
    });
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm of `a`.
///
/// Public because every backend has to measure its residual against the same
/// yardstick as this one, or "converged to 1e-8" would mean different things
/// depending on where the solve ran.
pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Build the Jacobi preconditioner `1 / diag`, guarding rows whose diagonal is
/// not positive (constrained or unattached degrees of freedom).
fn inverse_diagonal(diag: &[f64], fixed: &[bool]) -> Vec<f64> {
    diag.par_iter()
        .zip(fixed.par_iter())
        .map(|(&d, &f)| {
            if f || d <= constants::JACOBI_DIAGONAL_EPS {
                1.0 / constants::JACOBI_DIAGONAL_FLOOR
            } else {
                1.0 / d
            }
        })
        .collect()
}

/// Solve `K u = f` on the free subspace.
///
/// `apply` must compute `K v` for an arbitrary vector, `diag` the diagonal of
/// `K`, and `fixed` flags the constrained degrees of freedom. `x` carries the
/// initial guess in and the solution out, which lets the caller warm-start from
/// the previous optimization iteration.
///
/// `limits` carries the tolerance, the iteration cap and - for a caller that
/// may change its mind, which on the command line is nobody - the cancellation
/// probe the loop below checks every
/// [`crate::constants::CG_CANCEL_CHECK_INTERVAL`] iterations.
pub fn solve<F>(
    apply: F,
    diag: &[f64],
    fixed: &[bool],
    b: &[f64],
    x: &mut [f64],
    limits: SolveLimits<'_>,
) -> Result<Solve>
where
    F: Fn(&[f64], &mut [f64]),
{
    let SolveLimits {
        tolerance,
        max_iterations,
        cancel,
    } = limits;
    let n = b.len();
    assert_eq!(diag.len(), n, "diagonal length must match the system size");
    assert_eq!(
        fixed.len(),
        n,
        "constraint mask length must match the system size"
    );
    assert_eq!(x.len(), n, "solution length must match the system size");

    let mut rhs = b.to_vec();
    project(&mut rhs, fixed);
    let b_norm = norm(&rhs);
    if b_norm == 0.0 {
        // Nothing drives the system: the exact solution is zero.
        x.fill(0.0);
        return Ok(Solve::Converged(CgSolution {
            iterations: 0,
            relative_residual: 0.0,
        }));
    }

    let minv = inverse_diagonal(diag, fixed);
    project(x, fixed);

    let mut r = vec![0.0; n];
    apply(x, &mut r);
    project(&mut r, fixed);
    r.par_iter_mut()
        .zip(rhs.par_iter())
        .for_each(|(ri, &bi)| *ri = bi - *ri);

    let mut z: Vec<f64> = r
        .par_iter()
        .zip(minv.par_iter())
        .map(|(&ri, &mi)| ri * mi)
        .collect();
    project(&mut z, fixed);

    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![0.0; n];
    let mut relative_residual = norm(&r) / b_norm;

    if relative_residual < tolerance {
        return Ok(Solve::Converged(CgSolution {
            iterations: 0,
            relative_residual,
        }));
    }

    for iteration in 1..=max_iterations {
        // Before the work of the iteration rather than after it, so a stop
        // costs at most the checkpoint interval and never one more matrix
        // vector product than that.
        if cancel.cancelled_at(iteration) {
            return Ok(Solve::Cancelled);
        }
        apply(&p, &mut ap);
        project(&mut ap, fixed);
        let pap = dot(&p, &ap);
        if pap <= 0.0 || pap.is_nan() {
            bail!(
                "conjugate gradient breakdown: the operator is not positive definite on the free subspace (p^T K p = {pap})"
            );
        }
        let alpha = rz / pap;

        x.par_iter_mut()
            .zip(p.par_iter())
            .for_each(|(xi, &pi)| *xi += alpha * pi);
        r.par_iter_mut()
            .zip(ap.par_iter())
            .for_each(|(ri, &api)| *ri -= alpha * api);

        relative_residual = norm(&r) / b_norm;
        if relative_residual < tolerance {
            return Ok(Solve::Converged(CgSolution {
                iterations: iteration,
                relative_residual,
            }));
        }

        z.par_iter_mut()
            .zip(r.par_iter())
            .zip(minv.par_iter())
            .for_each(|((zi, &ri), &mi)| *zi = ri * mi);
        project(&mut z, fixed);

        let rz_next = dot(&r, &z);
        let beta = rz_next / rz;
        rz = rz_next;
        p.par_iter_mut()
            .zip(z.par_iter())
            .for_each(|(pi, &zi)| *pi = zi + beta * *pi);
    }

    bail!(
        "conjugate gradient did not converge in {max_iterations} iterations (relative residual {relative_residual:.3e}, target {tolerance:.3e})"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// The limits every test below solves to unless it says otherwise.
    fn limits() -> SolveLimits<'static> {
        SolveLimits::new(
            constants::CG_RELATIVE_TOLERANCE,
            constants::CG_MAX_ITERATIONS,
        )
    }

    /// Dense symmetric positive definite matrix used by the solver tests.
    fn dense() -> Vec<Vec<f64>> {
        vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 5.0, 2.0],
            vec![0.0, 0.0, 2.0, 6.0],
        ]
    }

    fn dense_apply(a: &[Vec<f64>], v: &[f64], out: &mut [f64]) {
        for (i, row) in a.iter().enumerate() {
            out[i] = row.iter().zip(v.iter()).map(|(m, x)| m * x).sum();
        }
    }

    /// Reference solution by Gaussian elimination with partial pivoting.
    fn gauss_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut m: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut r = row.clone();
                r.push(b[i]);
                r
            })
            .collect();
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&x, &y| m[x][col].abs().partial_cmp(&m[y][col].abs()).unwrap())
                .unwrap();
            m.swap(col, pivot);
            let d = m[col][col];
            for v in m[col].iter_mut() {
                *v /= d;
            }
            let pivot_row = m[col].clone();
            for (row, equation) in m.iter_mut().enumerate() {
                if row == col {
                    continue;
                }
                let f = equation[col];
                for (c, value) in equation.iter_mut().enumerate().skip(col) {
                    *value -= f * pivot_row[c];
                }
            }
        }
        (0..n).map(|i| m[i][n]).collect()
    }

    #[test]
    fn pcg_matches_a_direct_solve() {
        let a = dense();
        let b = vec![1.0, -2.0, 3.0, 0.5];
        let diag: Vec<f64> = (0..4).map(|i| a[i][i]).collect();
        let fixed = vec![false; 4];
        let mut x = vec![0.0; 4];
        let result = solve(
            |v, out| dense_apply(&a, v, out),
            &diag,
            &fixed,
            &b,
            &mut x,
            limits(),
        )
        .expect("solve")
        .converged()
        .expect("nothing asked this solve to stop");
        assert!(result.relative_residual < constants::CG_RELATIVE_TOLERANCE);

        let reference = gauss_solve(&a, &b);
        for i in 0..4 {
            assert!(
                (x[i] - reference[i]).abs() < 1e-9,
                "component {i}: got {}, expected {}",
                x[i],
                reference[i]
            );
        }
    }

    #[test]
    fn pcg_honours_dirichlet_projection() {
        let a = dense();
        let b = vec![1.0, -2.0, 3.0, 0.5];
        let diag: Vec<f64> = (0..4).map(|i| a[i][i]).collect();
        let fixed = vec![true, false, false, false];
        let mut x = vec![0.0; 4];
        solve(
            |v, out| dense_apply(&a, v, out),
            &diag,
            &fixed,
            &b,
            &mut x,
            limits(),
        )
        .expect("solve");
        assert_eq!(x[0], 0.0);

        // The remaining rows must satisfy the reduced system exactly.
        let reduced: Vec<Vec<f64>> = (1..4).map(|i| (1..4).map(|j| a[i][j]).collect()).collect();
        let reference = gauss_solve(&reduced, &b[1..]);
        for i in 0..3 {
            assert!((x[i + 1] - reference[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn pcg_warm_start_converges_immediately() {
        let a = dense();
        let b = vec![1.0, -2.0, 3.0, 0.5];
        let diag: Vec<f64> = (0..4).map(|i| a[i][i]).collect();
        let fixed = vec![false; 4];
        let mut x = gauss_solve(&a, &b);
        let result = solve(
            |v, out| dense_apply(&a, v, out),
            &diag,
            &fixed,
            &b,
            &mut x,
            limits(),
        )
        .expect("solve")
        .converged()
        .expect("converged");
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn pcg_reports_non_convergence() {
        let a = dense();
        let b = vec![1.0, -2.0, 3.0, 0.5];
        let diag: Vec<f64> = (0..4).map(|i| a[i][i]).collect();
        let fixed = vec![false; 4];
        let mut x = vec![0.0; 4];
        let err = solve(
            |v, out| dense_apply(&a, v, out),
            &diag,
            &fixed,
            &b,
            &mut x,
            SolveLimits::new(1e-16, 1),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("did not converge"), "unexpected error: {err}");
    }

    #[test]
    fn zero_right_hand_side_solves_to_zero() {
        let a = dense();
        let b = vec![0.0; 4];
        let diag: Vec<f64> = (0..4).map(|i| a[i][i]).collect();
        let fixed = vec![false; 4];
        let mut x = vec![1.0; 4];
        let result = solve(
            |v, out| dense_apply(&a, v, out),
            &diag,
            &fixed,
            &b,
            &mut x,
            limits(),
        )
        .expect("solve")
        .converged()
        .expect("converged");
        assert_eq!(result.iterations, 0);
        assert!(x.iter().all(|v| *v == 0.0));
    }

    /// A system that needs many iterations, stopped part way through: the solve
    /// comes back cancelled rather than converged or failed, and it does so
    /// within one checkpoint interval of the moment it was asked.
    #[test]
    fn a_cancelled_solve_stops_inside_the_checkpoint_bound() {
        // A long thin chain, which is the worst case a Jacobi preconditioner
        // has: 400 unknowns take hundreds of iterations at 1e-12.
        let n = 400;
        let apply_count = Cell::new(0usize);
        let apply = |v: &[f64], out: &mut [f64]| {
            apply_count.set(apply_count.get() + 1);
            for i in 0..n {
                let left = if i > 0 { v[i - 1] } else { 0.0 };
                let right = if i + 1 < n { v[i + 1] } else { 0.0 };
                out[i] = 2.0 * v[i] - left - right;
            }
        };
        let diag = vec![2.0; n];
        let fixed = vec![false; n];
        let b = vec![1.0; n];

        // Uncancelled, this really does run long enough for the test to mean
        // something.
        let mut x = vec![0.0; n];
        let reference = solve(
            apply,
            &diag,
            &fixed,
            &b,
            &mut x,
            SolveLimits::new(1e-12, 5000),
        )
        .expect("solve")
        .converged()
        .expect("converged");
        let interval = constants::CG_CANCEL_CHECK_INTERVAL;
        assert!(
            reference.iterations > 4 * interval,
            "the fixture converges in {} iterations, too few to cancel inside",
            reference.iterations
        );

        // Now stop it after a fixed amount of work.
        let stop_after = 2 * interval;
        apply_count.set(0);
        let stop = || apply_count.get() >= stop_after;
        let mut x = vec![0.0; n];
        let outcome = solve(
            apply,
            &diag,
            &fixed,
            &b,
            &mut x,
            SolveLimits::new(1e-12, 5000).watching(&stop),
        )
        .expect("a cancelled solve is not an error");
        assert_eq!(outcome, Solve::Cancelled);
        assert!(outcome.is_cancelled() && outcome.converged().is_none());
        // The first matrix-vector product happens before the loop, so the
        // iteration count trails the product count by one.
        assert!(
            apply_count.get() <= stop_after + interval + 1,
            "the stop took {} matrix-vector products to take effect, past the \
             bound of {}",
            apply_count.get(),
            stop_after + interval + 1
        );
    }
}
