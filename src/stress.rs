//! Post-run stress recovery.
//!
//! Once the optimization has converged the final density field is analysed once
//! more, and the von Mises stress is recovered at every element centroid from
//! those displacements. Nothing here can move the optimization: it runs after
//! the last iteration, on the field that is about to be exported.
//!
//! Two caveats the numbers come with, both inherent to density based topology
//! optimization rather than to this implementation:
//!
//! * **Intermediate densities.** Stress is recovered with the *full* modulus
//!   `E0`, which is standard SIMP practice, so a half dense element is reported
//!   as if it were solid. Elements below
//!   [`constants::STRESS_DENSITY_THRESHOLD`] are left out entirely; the ones
//!   just above it are the least trustworthy numbers in the table.
//! * **Discretization.** A single trilinear hexahedron per voxel underestimates
//!   peaks at re-entrant corners and at the staircase edges of the voxelized
//!   boundary. Treat the maximum as a screening figure, not as a certificate.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;

use crate::constants;
use crate::fea::{
    self, CancelProbe, LinearSolver, SolveLimits, StiffnessOperator, element_stress,
    hex8_centroid_stress, hex8_stiffness, von_mises,
};
use crate::json;
use crate::problem::Problem;

/// Iteration budget and tolerance of the post-run stress solve.
///
/// Separate from the optimization path's, and separately parameterised, because
/// the two solves are not the same problem: the optimization solves a warm
/// started sequence and can absorb one bad iteration, while this one runs once,
/// cold, on the most ill-conditioned field the run ever produced. The defaults
/// are [`constants::STRESS_CG_TOLERANCE`] and
/// [`constants::STRESS_CG_MAX_ITERATIONS`]; a caller passes its own to make the
/// solve fail on purpose and exercise the degraded report path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressLimits {
    /// Relative residual the solve is taken to.
    pub tolerance: f64,
    /// Hard iteration cap.
    pub max_iterations: usize,
}

impl Default for StressLimits {
    fn default() -> StressLimits {
        StressLimits {
            tolerance: constants::STRESS_CG_TOLERANCE,
            max_iterations: constants::STRESS_CG_MAX_ITERATIONS,
        }
    }
}

/// The stress report of a run, or the reason there is none.
///
/// A stress solve that will not converge is a degraded report, not a failed
/// run: the density field is already final, the STL is already implied by it,
/// and refusing to write the part because the *check* on it did not converge
/// would throw away everything the run computed. The absence travels with the
/// outcome so every consumer - console, JSON, viewer - can say so rather than
/// quietly showing zeros.
#[derive(Debug, Clone)]
pub enum StressOutcome {
    /// The solve converged and the table below is what it found.
    Available(StressReport),
    /// The solve failed; the string is the formatted error chain.
    Unavailable(String),
}

impl StressOutcome {
    /// The report, when there is one.
    pub fn report(&self) -> Option<&StressReport> {
        match self {
            StressOutcome::Available(report) => Some(report),
            StressOutcome::Unavailable(_) => None,
        }
    }

    /// Why there is no report, when there is none.
    pub fn reason(&self) -> Option<&str> {
        match self {
            StressOutcome::Available(_) => None,
            StressOutcome::Unavailable(reason) => Some(reason),
        }
    }

    /// True when a report is available.
    pub fn is_available(&self) -> bool {
        matches!(self, StressOutcome::Available(_))
    }
}

/// Stress statistics of one load case.
#[derive(Debug, Clone)]
pub struct CaseStress {
    /// Load case name.
    pub name: String,
    /// Largest von Mises stress over the evaluated elements, in MPa.
    pub max_mpa: f64,
    /// The [`constants::STRESS_PERCENTILE`] percentile, in MPa. Far less
    /// sensitive to a single staircase corner than the maximum.
    pub percentile_mpa: f64,
    /// Mean over the highest [`constants::STRESS_TOP_FRACTION`] of the
    /// evaluated elements, in MPa.
    pub top_fraction_mean_mpa: f64,
    /// `yield_strength_mpa / max_mpa`, when the material declares a yield
    /// strength and the structure carries any stress at all.
    pub safety_factor: Option<f64>,
    /// Per element von Mises stress in MPa, zero where the element was below
    /// the density threshold. In grid element order.
    pub von_mises: Vec<f64>,
}

/// The stress report of a whole run.
#[derive(Debug, Clone)]
pub struct StressReport {
    /// One entry per load case, in configuration order.
    pub cases: Vec<CaseStress>,
    /// Material yield strength in MPa, when one was configured.
    pub yield_strength_mpa: Option<f64>,
    /// Physical density an element needed to be evaluated at all.
    pub density_threshold: f64,
    /// Number of elements the report covers.
    pub evaluated_cells: usize,
}

impl StressReport {
    /// The worst safety factor over every load case.
    pub fn worst_safety_factor(&self) -> Option<f64> {
        self.cases
            .iter()
            .filter_map(|c| c.safety_factor)
            .fold(None, |worst: Option<f64>, factor| {
                Some(worst.map_or(factor, |w| w.min(factor)))
            })
    }

    /// The largest von Mises stress over every load case, in MPa.
    pub fn max_mpa(&self) -> f64 {
        self.cases.iter().map(|c| c.max_mpa).fold(0.0, f64::max)
    }

    /// The report in the few lines a reader wants: the safety factor, the peak
    /// each load case reached, and - when the export came out in pieces - the
    /// warning that says what the factor is really a factor of.
    ///
    /// Both numbers are the report's own - [`StressReport::worst_safety_factor`]
    /// over [`StressReport::max_mpa`] - rather than a second derivation of them,
    /// so this can only ever quote what the console table quotes.
    ///
    /// `bodies` is how many separate bodies the exported surface holds, which is
    /// not something a stress report can know: it is a property of the mesh, and
    /// it is passed in from the island report of the surface that was written -
    /// [`crate::mesh::islands::IslandReport::bodies`], after the cull. See
    /// [`disconnected_bodies_note`] for what it buys.
    pub fn summary(&self, bodies: usize) -> StressSummary {
        let peak = self.max_mpa();
        let headline = match (self.worst_safety_factor(), self.yield_strength_mpa) {
            (Some(factor), Some(strength)) => {
                format!("safety factor {factor:.2} (peak {peak:.4} MPa vs yield {strength} MPa)")
            }
            // A yield strength and no factor is a structure carrying nothing at
            // all: there is no peak to divide into.
            (None, Some(strength)) => format!(
                "safety factor n/a (peak {peak:.4} MPa vs yield {strength} MPa; nothing is \
                 carrying load)"
            ),
            (_, None) => format!(
                "safety factor n/a (peak {peak:.4} MPa; no yield_strength_mpa in [material])"
            ),
        };
        StressSummary {
            warning: disconnected_bodies_note(bodies).map(|note| format!("warning: {note}")),
            headline,
            cases: self
                .cases
                .iter()
                .map(|case| format!("{} peak {:.4} MPa", case.name, case.max_mpa))
                .collect(),
        }
    }
}

/// What a safety factor means when the exported surface came out in more than
/// one piece, or `None` when it came out as one connected part.
///
/// A model whose pieces are each held by their own supports is solved correctly
/// and reported correctly, every piece in equilibrium against what holds it, and
/// the number that comes out of it is still not the number a user asked for:
/// a part meant to *join* two things was never analysed as one. The analysis
/// cannot tell that the supports are fictitious; the placed regions are what the
/// configuration declared and it has nothing to check them against. What it can
/// tell is that the surface it is describing is in pieces, and saying so beside
/// the factor is the difference between a screening figure and a claim about a
/// part that is held together by air.
///
/// `bodies` is the count of the surface that was *written*: the bodies the island
/// cull kept, post-trim and post-cull, because every deliverable of a run
/// describes the export rather than some field on the way to it. Two or more is
/// always plural, so there is one sentence rather than two.
///
/// One function behind every surface that says this - the editor's panel, the
/// console line the editor echoes there, and `growforge run`'s own stress block -
/// each of which decorates it its own way and none of which may word it
/// differently.
pub fn disconnected_bodies_note(bodies: usize) -> Option<String> {
    (bodies > 1).then(|| {
        format!(
            "the export is {bodies} separate bodies - this safety factor describes each piece \
             against its own supports, not one connected part"
        )
    })
}

/// The stress report of a run as plain text.
///
/// One formatter behind every consumer that reads the numbers rather than
/// tabulating or colouring them: the editor's panel and the console line the
/// editor's session echoes there, which therefore cannot disagree. The console
/// prints [`StressSummary::lines`] in order; the panel draws the warning and the
/// headline on their own, because the safety factor is the number the part is
/// judged by and the warning is what that number is worth.
///
/// Built from the analysis that describes the exported part - the one after a
/// trim's re-analysis, never the one before it - because that is what
/// [`crate::complete`] hands its caller.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StressSummary {
    /// What the safety factor below is really a factor of, when the exported
    /// surface came out in more than one piece; `None` for the single body every
    /// other line here assumes. Worded by [`disconnected_bodies_note`] and
    /// prefixed with "warning", which is what a panel colours on.
    pub warning: Option<String>,
    /// The safety factor, the peak it was taken from and the yield strength it
    /// was measured against; or, when there is no factor, which of the two is
    /// missing.
    pub headline: String,
    /// One line per load case, in configuration order: its name and its peak
    /// von Mises stress.
    pub cases: Vec<String>,
}

impl StressSummary {
    /// Every line of the summary: the warning when there is one, then the
    /// headline, then the load cases.
    ///
    /// The warning comes first because it is the caveat on everything under it,
    /// and a reader who stops after one line has to have read it rather than the
    /// factor it qualifies.
    pub fn lines(&self) -> Vec<String> {
        let mut lines =
            Vec::with_capacity(usize::from(self.warning.is_some()) + 1 + self.cases.len());
        lines.extend(self.warning.iter().cloned());
        lines.push(self.headline.clone());
        lines.extend(self.cases.iter().cloned());
        lines
    }
}

/// Order statistics of an ascending, non-empty sample.
fn statistics(sorted: &[f64]) -> (f64, f64, f64) {
    let n = sorted.len();
    let max = sorted[n - 1];
    // Nearest rank: the smallest value at or above the requested fraction.
    let rank = ((constants::STRESS_PERCENTILE * n as f64).ceil() as usize).clamp(1, n);
    let percentile = sorted[rank - 1];
    let top = ((constants::STRESS_TOP_FRACTION * n as f64).ceil() as usize).clamp(1, n);
    let mean = sorted[n - top..].iter().sum::<f64>() / top as f64;
    (max, percentile, mean)
}

/// Solve every load case on `densities` and recover the von Mises stresses.
///
/// Any failure at all - including a solve that will not converge - is an error
/// here. Callers that want the degraded report instead want [`analyse_with`],
/// which is what tells the two kinds of failure apart.
pub fn analyse(problem: &Problem, densities: &[f64]) -> Result<StressReport> {
    let outcome = analyse_with(
        problem,
        densities,
        StressLimits::default(),
        CancelProbe::NONE,
    )?
    .expect("an unwatched stress pass is never cancelled");
    match outcome {
        StressOutcome::Available(report) => Ok(report),
        StressOutcome::Unavailable(reason) => Err(anyhow!(reason)),
    }
}

/// Whether every load of every case can reach a support through material.
///
/// The near-singular system this is about is the one where a load enters a
/// structure that is joined to nothing: the only path from the loaded nodes to
/// the constrained ones runs through cells at the SIMP stiffness floor, a factor
/// of [`constants::SIMP_EMIN_FRACTION`] down, and no conjugate gradient will
/// resolve it. That is not something the solve can discover cheaply - it
/// discovers it by grinding for tens of thousands of iterations and then failing
/// - but it is one flood fill to answer here.
///
/// The fill is [`crate::voids::solid_bodies`]'s own, at
/// [`constants::STRESS_LOAD_PATH_DENSITY`] rather than at the iso level: the
/// question is whether there is *any* stiffness on the path, not whether the
/// path is part of the printed part. A node counts as standing on a body when
/// any of the up-to-eight cells around it belongs to it.
///
/// Returns one message per load that cannot reach a support, in configuration
/// order, and an empty vector for a structure that is properly held.
pub fn broken_load_paths(problem: &Problem, densities: &[f64]) -> Vec<String> {
    let grid = &problem.grid;
    let labels = crate::voids::body_labels(grid, densities, constants::STRESS_LOAD_PATH_DENSITY);
    // Which bodies the supports stand on. A support that stands on nothing at
    // all contributes nothing, which is itself a reason a load cannot reach one.
    let mut held: Vec<bool> = vec![false; labels.bodies];
    for support in &problem.supports {
        for body in bodies_under(grid, &labels.of_cell, &support.nodes) {
            held[body] = true;
        }
    }

    let mut broken = Vec::new();
    for case in &problem.load_cases {
        for (index, load) in case.loads.iter().enumerate() {
            // A gravity load acts on every element there is, so it is carried
            // wherever the material is; only a placed region can be stranded.
            if load.nodes.is_empty() {
                continue;
            }
            let standing = bodies_under(grid, &labels.of_cell, &load.nodes);
            let what = format!("load {} of load case \"{}\"", index + 1, case.name);
            if standing.is_empty() {
                broken.push(format!(
                    "{what} acts on nodes with no material around them at all, so the system it \
                     drives is near-singular"
                ));
            } else if !standing.iter().any(|&body| held[body]) {
                broken.push(format!(
                    "{what} is not connected to any support through material, so the system it \
                     drives is near-singular"
                ));
            }
        }
    }
    broken
}

/// The connected bodies of the up-to-eight cells around each of `nodes`,
/// without repeats.
fn bodies_under(
    grid: &crate::grid::Grid,
    of_cell: &[Option<usize>],
    nodes: &[usize],
) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for &node in nodes {
        let (i, j, k) = grid.node_ijk(node);
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    // A node at coordinate `i` is a corner of cells `i - 1` and
                    // `i`, of which either may be off the grid at a face.
                    let (ci, cj, ck) = (i + di, j + dj, k + dk);
                    if ci == 0 || cj == 0 || ck == 0 {
                        continue;
                    }
                    let (ci, cj, ck) = (ci - 1, cj - 1, ck - 1);
                    if ci >= grid.nx || cj >= grid.ny || ck >= grid.nz {
                        continue;
                    }
                    let Some(body) = of_cell[grid.cell_index(ci, cj, ck)] else {
                        continue;
                    };
                    if !out.contains(&body) {
                        out.push(body);
                    }
                }
            }
        }
    }
    out
}

/// [`analyse`] with an explicit iteration budget and tolerance, classifying the
/// two kinds of failure the stress pass can suffer.
///
/// `Err` is a *setup* failure: the configured solver backend could not be
/// opened, or the design could not be bound to it. Nothing was computed and
/// nothing about the run is salvageable by carrying on, so it propagates.
///
/// `Ok(Some(StressOutcome::Unavailable))` is a *solve* failure of an
/// already-opened, already-bound solver: it did not converge in its budget, it
/// broke down, or the pre-check below found it could not have converged. The
/// density field is unaffected and the part is still exportable, so the run
/// keeps its STL and loses only the table.
///
/// `Ok(None)` is neither: `cancel` said the caller no longer wants the report.
pub fn analyse_with(
    problem: &Problem,
    densities: &[f64],
    limits: StressLimits,
    cancel: CancelProbe<'_>,
) -> Result<Option<StressOutcome>> {
    let mut solver =
        LinearSolver::new(problem).context("opening the linear solver for the stress report")?;
    analyse_with_solver(problem, densities, limits, &mut solver, cancel)
}

/// [`analyse_with`] against a solver the caller has already opened.
///
/// Binding still happens here, and still fails hard: it is part of setting the
/// solver up, not part of solving with it.
pub fn analyse_with_solver(
    problem: &Problem,
    densities: &[f64],
    limits: StressLimits,
    solver: &mut LinearSolver,
    cancel: CancelProbe<'_>,
) -> Result<Option<StressOutcome>> {
    // Cheap enough to do before anything is bound, and the one failure mode a
    // solve cannot report in less than its whole iteration budget: a load that
    // reaches no support through material drives a system a thousand million
    // times worse conditioned than the one this tolerance assumes.
    let broken = broken_load_paths(problem, densities);
    if !broken.is_empty() {
        return Ok(Some(StressOutcome::Unavailable(format!(
            "{}; solving it would only spend the iteration budget to say so",
            broken.join("; ")
        ))));
    }

    let grid = &problem.grid;
    let e0 = problem.material.youngs_modulus_mpa;
    let threshold = constants::STRESS_DENSITY_THRESHOLD;

    let mut moduli = vec![0.0; grid.n_cells()];
    crate::engine::simp_moduli(
        grid,
        densities,
        e0,
        problem.optimization.penalty,
        &mut moduli,
    );
    let ke0 = hex8_stiffness(problem.material.poisson_ratio, grid.h);
    let operator = StiffnessOperator::new(grid, &ke0, &moduli);
    let diagonal = operator.diagonal();
    let stress_matrix = hex8_centroid_stress(problem.material.poisson_ratio, grid.h);

    let evaluated: Vec<usize> = (0..grid.n_cells())
        .filter(|&e| crate::engine::cell_density(grid, densities, e) >= threshold)
        .collect();

    let mut bound = solver
        .bind(&operator, &diagonal, &problem.fixed)
        .context("binding the final design to the linear solver")?;

    let mut forces = vec![0.0; grid.n_dof()];
    let mut cases = Vec::with_capacity(problem.load_cases.len());
    for (index, case) in problem.load_cases.iter().enumerate() {
        problem.case_forces(index, densities, &mut forces);
        let mut u = vec![0.0; grid.n_dof()];
        // The one degradable step. Everything above it had to work for the
        // question to even be asked; this is the one that is allowed to answer
        // "not in the budget you gave me".
        let solved = bound.solve(
            &forces,
            &mut u,
            SolveLimits {
                tolerance: limits.tolerance,
                max_iterations: limits.max_iterations,
                cancel,
            },
        );
        match solved {
            Err(error) => {
                return Ok(Some(StressOutcome::Unavailable(format!(
                    "solving load case \"{}\" for the stress report: {error:#}",
                    case.name
                ))));
            }
            // Called off: not a degraded report, because nobody is going to read
            // one. The caller unwinds and exports nothing.
            Ok(fea::Solve::Cancelled) => return Ok(None),
            Ok(fea::Solve::Converged(_)) => {}
        }

        // Recovery uses the full solid modulus, not the penalized one: see the
        // module caveats.
        let mut per_element = vec![0.0; grid.n_cells()];
        let stresses: Vec<(usize, f64)> = evaluated
            .par_iter()
            .map(|&e| {
                let (i, j, k) = grid.cell_ijk(e);
                let ue = fea::gather(&u, &grid.element_nodes(i, j, k));
                (e, von_mises(&element_stress(&stress_matrix, e0, &ue)))
            })
            .collect();
        let mut sorted = Vec::with_capacity(stresses.len());
        for (e, value) in stresses {
            per_element[e] = value;
            sorted.push(value);
        }
        sorted.sort_by(|a, b| a.total_cmp(b));

        let (max_mpa, percentile_mpa, top_fraction_mean_mpa) = if sorted.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            statistics(&sorted)
        };
        let safety_factor = problem
            .material
            .yield_strength_mpa
            .filter(|_| max_mpa > 0.0)
            .map(|y| y / max_mpa);
        cases.push(CaseStress {
            name: case.name.clone(),
            max_mpa,
            percentile_mpa,
            top_fraction_mean_mpa,
            safety_factor,
            von_mises: per_element,
        });
    }

    Ok(Some(StressOutcome::Available(StressReport {
        cases,
        yield_strength_mpa: problem.material.yield_strength_mpa,
        density_threshold: threshold,
        evaluated_cells: evaluated.len(),
    })))
}

/// Serialize the report as JSON. Stresses are in MPa throughout.
pub fn to_json(problem: &Problem, report: &StressReport) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!(
        "  \"project\": {},\n",
        json::string(&problem.name)
    ));
    out.push_str(&format!(
        "  \"generator\": {},\n",
        json::string(&format!(
            "{} {}",
            constants::PROGRAM_NAME,
            constants::VERSION
        ))
    ));
    out.push_str("  \"units\": \"MPa\",\n");
    out.push_str(&format!(
        "  \"yield_strength_mpa\": {},\n",
        json::optional_number(report.yield_strength_mpa)
    ));
    out.push_str(&format!(
        "  \"density_threshold\": {},\n",
        json::number(report.density_threshold)
    ));
    out.push_str(&format!(
        "  \"evaluated_cells\": {},\n",
        report.evaluated_cells
    ));
    out.push_str(&format!(
        "  \"percentile\": {},\n",
        json::number(constants::STRESS_PERCENTILE)
    ));
    out.push_str(&format!(
        "  \"top_fraction\": {},\n",
        json::number(constants::STRESS_TOP_FRACTION)
    ));
    out.push_str("  \"loadcases\": [\n");
    for (index, case) in report.cases.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"name\": {},\n", json::string(&case.name)));
        out.push_str(&format!(
            "      \"max_von_mises_mpa\": {},\n",
            json::number(case.max_mpa)
        ));
        out.push_str(&format!(
            "      \"percentile_von_mises_mpa\": {},\n",
            json::number(case.percentile_mpa)
        ));
        out.push_str(&format!(
            "      \"top_fraction_mean_mpa\": {},\n",
            json::number(case.top_fraction_mean_mpa)
        ));
        out.push_str(&format!(
            "      \"safety_factor\": {}\n",
            json::optional_number(case.safety_factor)
        ));
        out.push_str(if index + 1 == report.cases.len() {
            "    }\n"
        } else {
            "    },\n"
        });
    }
    out.push_str("  ]\n}\n");
    out
}

/// Write the JSON report to `path`, creating the parent directory if needed.
pub fn write_json(path: &Path, problem: &Problem, report: &StressReport) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating the directory {}", parent.display()))?;
    }
    fs::write(path, to_json(problem, report))
        .with_context(|| format!("writing the stress report to {}", path.display()))
}

/// Sample a per-element stress field onto the node lattice the exported surface
/// was extracted from, using the same eight-cell averaging as the densities.
///
/// Only elements that were evaluated contribute; a node with no evaluated
/// neighbour reads zero. The lattice carries the same
/// [`constants::FIELD_PADDING_CELLS`] as the density field, so a value can be
/// looked up at any point of the exported mesh.
pub fn node_stress_field(
    problem: &Problem,
    densities: &[f64],
    von_mises: &[f64],
) -> crate::mesh::ScalarField {
    let grid = &problem.grid;
    let pad = constants::FIELD_PADDING_CELLS;
    let mut field = crate::mesh::ScalarField::new(
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
    let threshold = constants::STRESS_DENSITY_THRESHOLD;
    for pk in 0..field.nz {
        for pj in 0..field.ny {
            for pi in 0..field.nx {
                let (i, j, k) = (
                    pi as isize - pad as isize,
                    pj as isize - pad as isize,
                    pk as isize - pad as isize,
                );
                let mut sum = 0.0;
                let mut count = 0usize;
                for dk in 0..2 {
                    for dj in 0..2 {
                        for di in 0..2 {
                            let (ci, cj, ck) = (i - 1 + di, j - 1 + dj, k - 1 + dk);
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
                            if crate::engine::cell_density(grid, densities, e) < threshold {
                                continue;
                            }
                            sum += von_mises[e];
                            count += 1;
                        }
                    }
                }
                let index = field.index(pi, pj, pk);
                field.values[index] = if count > 0 { sum / count as f64 } else { 0.0 };
            }
        }
    }
    field
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    /// A single solid element in pure axial tension, restrained on three
    /// perpendicular faces so it may contract laterally: the von Mises stress
    /// has to come out as `F / A`.
    const BAR: &str = r#"
[project]
name = "bar"

[resolution]
voxel_size_mm = 10.0

[material]
youngs_modulus_mpa = 2100.0
poisson_ratio = 0.35
density_g_cm3 = 1.27
yield_strength_mpa = 47.0

# The reproducible backend: this fixture's numbers are compared, and the default
# backend is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.5
min_feature_mm = 20.0

[output]
stl_path = "bar.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [10.0, 10.0, 10.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 10.5, 10.5] }
directions = ["x"]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [10.5, 0.5, 10.5] }
directions = ["y"]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [10.5, 10.5, 0.5] }
directions = ["z"]

[[loadcases]]
name = "pull"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [9.5, -0.5, -0.5], max = [10.5, 10.5, 10.5] }
vector = [1000.0, 0.0, 0.0]
"#;

    fn bar() -> Problem {
        let config = Config::parse(BAR).expect("parse");
        Problem::build(&config, &PathBuf::from(".")).expect("build")
    }

    #[test]
    fn a_uniaxial_bar_reports_force_over_area() {
        let problem = bar();
        let densities = vec![1.0; problem.grid.n_cells()];
        let report = analyse(&problem, &densities).expect("analyse");
        assert_eq!(report.cases.len(), 1);
        assert_eq!(report.evaluated_cells, 1);

        let expected = 1000.0 / (10.0 * 10.0);
        let case = &report.cases[0];
        let error = (case.max_mpa - expected).abs() / expected;
        assert!(
            error < 0.03,
            "von Mises {} differs from F/A = {expected} by {:.2}%",
            case.max_mpa,
            error * 100.0
        );
        // A single element makes all three statistics the same number.
        assert!((case.percentile_mpa - case.max_mpa).abs() < 1e-12);
        assert!((case.top_fraction_mean_mpa - case.max_mpa).abs() < 1e-12);
    }

    #[test]
    fn the_safety_factor_is_yield_over_the_peak() {
        let problem = bar();
        let densities = vec![1.0; problem.grid.n_cells()];
        let report = analyse(&problem, &densities).expect("analyse");
        let case = &report.cases[0];
        let factor = case.safety_factor.expect("a safety factor");
        assert!((factor - 47.0 / case.max_mpa).abs() < 1e-12);
        assert_eq!(report.worst_safety_factor(), Some(factor));
        assert!((report.max_mpa() - case.max_mpa).abs() < 1e-12);
    }

    /// A report of two load cases with a known yield strength, as text.
    ///
    /// The headline is the *worst* case's factor, and it is character for
    /// character the number the console table prints for that case - both are
    /// `{:.2}` of the same `safety_factor`, which is what keeps the panel, the
    /// editor's console and `print_stress_report` from ever quoting three
    /// different numbers for one part.
    #[test]
    fn the_summary_quotes_the_worst_safety_factor_and_every_peak() {
        let report = two_case_report();

        // One body: every line is what it was before the export could be in
        // pieces at all, and there is no warning to read.
        let summary = report.summary(1);
        assert_eq!(summary.warning, None);
        assert_eq!(
            summary.headline,
            "safety factor 14.69 (peak 3.2000 MPa vs yield 47 MPa)"
        );
        assert_eq!(
            summary.cases,
            vec![
                "hoist peak 1.2500 MPa".to_string(),
                "side peak 3.2000 MPa".to_string(),
            ]
        );
        assert_eq!(summary.lines(), {
            let mut expected = vec![summary.headline.clone()];
            expected.extend(summary.cases.clone());
            expected
        });
        let worst = report.cases[1].safety_factor.expect("a safety factor");
        assert_eq!(report.worst_safety_factor(), Some(worst));
        assert!(
            summary.headline.contains(&format!("{worst:.2}")),
            "the headline is not the table's own number: {}",
            summary.headline
        );

        // No yield strength: the peak is still said, and the key that would give
        // a factor is named rather than the line simply reading "n/a".
        let mut without = report.clone();
        without.yield_strength_mpa = None;
        for case in &mut without.cases {
            case.safety_factor = None;
        }
        assert_eq!(
            without.summary(1).headline,
            "safety factor n/a (peak 3.2000 MPa; no yield_strength_mpa in [material])"
        );
        assert_eq!(without.summary(1).cases, summary.cases);

        // A yield strength and a structure carrying nothing: there is no peak to
        // divide into, and saying so is not the same as saying the key is
        // missing.
        let mut idle = report.clone();
        for case in &mut idle.cases {
            case.max_mpa = 0.0;
            case.safety_factor = None;
        }
        assert_eq!(
            idle.summary(1).headline,
            "safety factor n/a (peak 0.0000 MPa vs yield 47 MPa; nothing is carrying load)"
        );
    }

    /// Two load cases with a known yield strength, for the text the summary
    /// makes of them.
    fn two_case_report() -> StressReport {
        let case = |name: &str, max_mpa: f64| CaseStress {
            name: name.to_string(),
            max_mpa,
            percentile_mpa: max_mpa * 0.8,
            top_fraction_mean_mpa: max_mpa * 0.9,
            safety_factor: Some(47.0 / max_mpa),
            von_mises: Vec::new(),
        };
        StressReport {
            cases: vec![case("hoist", 1.25), case("side", 3.2)],
            yield_strength_mpa: Some(47.0),
            density_threshold: constants::STRESS_DENSITY_THRESHOLD,
            evaluated_cells: 512,
        }
    }

    /// An export in pieces says so above its safety factor, and an export in one
    /// piece is untouched.
    ///
    /// The incident: two rods to be linked by one part came out as two bodies,
    /// each load group shunting into its own local support anchor, and the
    /// summary reported "safety factor 5.07" of a thing held together by air.
    /// The analysis is truthful about the model it was given and cannot know the
    /// supports are fictitious - but it does know the surface came out in pieces,
    /// and that is what this says.
    #[test]
    fn an_export_in_several_bodies_warns_above_its_safety_factor() {
        let report = two_case_report();
        let one = report.summary(1);

        for bodies in [2, 5] {
            let summary = report.summary(bodies);
            let warning = summary
                .warning
                .clone()
                .expect("an export in pieces must say so");
            assert_eq!(
                warning,
                format!(
                    "warning: the export is {bodies} separate bodies - this safety factor \
                     describes each piece against its own supports, not one connected part"
                )
            );
            // The colouring rule the panel and the pass notes share.
            assert!(warning.starts_with("warning"), "{warning}");
            // The count is the one it was given, and no other number in the line
            // can be mistaken for it.
            assert!(
                warning.contains(&format!("is {bodies} separate")),
                "{warning}"
            );

            // The warning is read first, and nothing under it moved.
            assert_eq!(summary.lines()[0], warning);
            assert_eq!(summary.lines().len(), one.lines().len() + 1);
            assert_eq!(summary.headline, one.headline);
            assert_eq!(summary.cases, one.cases);
        }

        // Zero bodies is an export with nothing in it, which is not a part in
        // pieces; one body is the case every other line assumes.
        assert_eq!(disconnected_bodies_note(0), None);
        assert_eq!(disconnected_bodies_note(1), None);
        assert_eq!(report.summary(0), one);
    }

    #[test]
    fn a_material_without_a_yield_strength_reports_no_safety_factor() {
        let text = BAR.replace("yield_strength_mpa = 47.0", "");
        let config = Config::parse(&text).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let densities = vec![1.0; problem.grid.n_cells()];
        let report = analyse(&problem, &densities).expect("analyse");
        assert!(report.cases[0].safety_factor.is_none());
        assert_eq!(report.worst_safety_factor(), None);
        assert!(to_json(&problem, &report).contains("\"safety_factor\": null"));
    }

    #[test]
    fn near_void_elements_are_left_out_of_the_report() {
        let problem = bar();
        let below = vec![constants::STRESS_DENSITY_THRESHOLD * 0.5; problem.grid.n_cells()];
        let report = analyse(&problem, &below).expect("analyse");
        assert_eq!(report.evaluated_cells, 0);
        assert_eq!(report.cases[0].max_mpa, 0.0);
        assert!(report.cases[0].safety_factor.is_none());
    }

    #[test]
    fn the_statistics_pick_the_documented_ranks() {
        let sorted: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        let (max, percentile, top) = statistics(&sorted);
        assert_eq!(max, 100.0);
        assert_eq!(percentile, 99.0);
        // The top decile of a hundred samples is the last ten.
        assert!((top - 95.5).abs() < 1e-12);
        // A single sample is its own everything.
        assert_eq!(statistics(&[7.0]), (7.0, 7.0, 7.0));
    }

    #[test]
    fn the_json_report_carries_every_case() {
        let problem = bar();
        let densities = vec![1.0; problem.grid.n_cells()];
        let report = analyse(&problem, &densities).expect("analyse");
        let text = to_json(&problem, &report);
        assert!(text.contains("\"project\": \"bar\""));
        assert!(text.contains("\"name\": \"pull\""));
        assert!(text.contains("\"max_von_mises_mpa\""));
        assert!(text.contains("\"yield_strength_mpa\": 47"));
        // Balanced braces and brackets are a cheap structural check.
        assert_eq!(
            text.matches('{').count(),
            text.matches('}').count(),
            "unbalanced braces in {text}"
        );
        assert_eq!(text.matches('[').count(), text.matches(']').count());
        assert!(!text.contains(",\n  ]"), "trailing comma in {text}");
    }

    /// A structure with enough elements for the recovery to have something to
    /// disagree about, solved on both backends.
    const BRACKET: &str = r#"
[project]
name = "bracket"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "petg"

# The reproducible backend: this fixture's numbers are compared, and the default
# backend is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0

[output]
stl_path = "bracket.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [40.0, 12.0, 12.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 12.5, 12.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [40.0, 6.0, 6.0], radius = 3.0 }
vector = [0.0, 0.0, -120.0]
"#;

    /// Opening a backend is setup, not solving. A machine or a build that
    /// cannot provide the configured one has been asked for something it cannot
    /// do, which says nothing about the structure and must not be softened into
    /// "stress report unavailable" - if it were, a `growth` run with
    /// `backend = "gpu"` on an adapterless machine would exit zero having never
    /// opened a backend at all, because growth performs no solves of its own and
    /// this is the first place one is asked for.
    #[test]
    fn a_backend_that_cannot_be_opened_is_a_hard_error_not_a_degraded_report() {
        let mut problem = bar();
        problem.solver.backend = crate::config::SolverBackend::Gpu;
        let densities = vec![1.0; problem.grid.n_cells()];
        match analyse_with(
            &problem,
            &densities,
            StressLimits::default(),
            CancelProbe::NONE,
        ) {
            // This build and machine can open a compute backend. The single
            // element bar is trivial, so the report has to be there: whatever
            // else happens, `Ok(Unavailable)` is not a legal answer to a backend
            // question.
            Ok(outcome) => {
                let outcome = outcome.expect("nothing asked this analysis to stop");
                assert!(
                    outcome.is_available(),
                    "a backend that opened must not yield a degraded report: {:?}",
                    outcome.reason()
                );
            }
            // It cannot, and says so as an error rather than a warning.
            Err(error) => {
                let text = format!("{error:#}");
                assert!(
                    text.contains("opening the linear solver"),
                    "unexpected error: {text}"
                );
            }
        }
    }

    /// The other half of the same contract: binding is setup too.
    #[test]
    fn a_design_that_cannot_be_bound_is_a_hard_error_not_a_degraded_report() {
        let what = "a_design_that_cannot_be_bound_is_a_hard_error_not_a_degraded_report";
        // A solver's buffers are sized for one grid, so handing it another
        // grid's operator is a real, reachable bind failure - no test-only hook
        // needed. Only the GPU backend has buffers to mis-size; the CPU one
        // cannot fail to bind at all.
        let other = Problem::build(&Config::parse(BRACKET).expect("parse"), &PathBuf::from("."))
            .expect("build");
        let Some(mut solver) = crate::fea::backend::gpu_or_skip(what, &other.grid, &other.fixed)
        else {
            return;
        };

        let problem = bar();
        assert_ne!(problem.grid.n_cells(), other.grid.n_cells());
        let densities = vec![1.0; problem.grid.n_cells()];
        let error = analyse_with_solver(
            &problem,
            &densities,
            StressLimits::default(),
            &mut solver,
            CancelProbe::NONE,
        )
        .expect_err("binding a mismatched design must fail");
        let text = format!("{error:#}");
        assert!(
            text.contains("binding the final design"),
            "unexpected error: {text}"
        );
    }

    #[test]
    fn the_gpu_backend_recovers_the_same_stresses() {
        let what = "the_gpu_backend_recovers_the_same_stresses";
        let config = Config::parse(BRACKET).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        if crate::fea::backend::gpu_or_skip(what, &problem.grid, &problem.fixed).is_none() {
            return;
        }

        // A partly resolved design, so the recovery sees the SIMP stiffness
        // contrast rather than a uniform block.
        let densities: Vec<f64> = (0..problem.grid.n_cells())
            .map(|e| {
                let (_, j, k) = problem.grid.cell_ijk(e);
                if j == k || j + k == problem.grid.ny - 1 {
                    1.0
                } else {
                    0.1
                }
            })
            .collect();

        let reference = analyse(&problem, &densities).expect("cpu stress");
        let mut accelerated = problem.clone();
        accelerated.solver.backend = crate::config::SolverBackend::Gpu;
        let candidate = analyse(&accelerated, &densities).expect("gpu stress");

        assert_eq!(candidate.evaluated_cells, reference.evaluated_cells);
        assert!(reference.max_mpa() > 0.0);
        let error = (candidate.max_mpa() - reference.max_mpa()).abs() / reference.max_mpa();
        println!(
            "{what}: cpu {:.6} MPa, gpu {:.6} MPa, relative difference {error:.3e}",
            reference.max_mpa(),
            candidate.max_mpa()
        );
        assert!(
            error < 1e-3,
            "peak von Mises {} differs from the CPU reference {} by {error:.3e}",
            candidate.max_mpa(),
            reference.max_mpa()
        );
    }

    /// A structure cut in two by a wall of keepout, with the load on the far
    /// side of the cut from the supports. This is the shape of the incident the
    /// pre-check exists for: the system it drives is near-singular, and finding
    /// that out by solving costs the whole iteration budget.
    const STRANDED: &str = r#"
[project]
name = "stranded"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "pla"

[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.5
min_feature_mm = 16.0
max_iterations = 1

[output]
stl_path = "stranded.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [64.0, 16.0, 16.0]

[[keepout]]
shape = "box"
min = [28.0, -1.0, -1.0]
max = [36.0, 17.0, 17.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 16.5, 16.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [64.0, 8.0, 8.0], radius = 6.0 }
vector = [0.0, 0.0, -50.0]
"#;

    fn stranded() -> Problem {
        let config = Config::parse(STRANDED).expect("parse");
        Problem::build(&config, &PathBuf::from(".")).expect("build")
    }

    /// The same fixture with the wall taken out, so the load reaches the
    /// supports through solid material.
    fn joined() -> Problem {
        let text = STRANDED.replace(
            "[[keepout]]\nshape = \"box\"\nmin = [28.0, -1.0, -1.0]\nmax = [36.0, 17.0, 17.0]\n",
            "",
        );
        assert!(!text.contains("keepout"), "the wall was not removed");
        let config = Config::parse(&text).expect("parse");
        Problem::build(&config, &PathBuf::from(".")).expect("build")
    }

    #[test]
    fn a_load_cut_off_from_the_supports_is_found_before_anything_is_solved() {
        let problem = stranded();
        let densities = vec![1.0; problem.grid.n_cells()];
        let broken = broken_load_paths(&problem, &densities);
        assert_eq!(broken.len(), 1, "{broken:?}");
        assert!(
            broken[0].contains("not connected to any support")
                && broken[0].contains("load case \"tip\""),
            "{}",
            broken[0]
        );

        // And that is what the stress pass reports, as a degraded report rather
        // than as an error: the field is still exportable.
        let outcome = analyse_with(
            &problem,
            &densities,
            StressLimits::default(),
            CancelProbe::NONE,
        )
        .expect("the analysis itself must succeed")
        .expect("nothing asked it to stop");
        let reason = outcome.reason().expect("a reason");
        assert!(!outcome.is_available());
        assert!(
            reason.contains("not connected to any support"),
            "unexpected reason: {reason}"
        );
    }

    #[test]
    fn a_properly_held_structure_reports_no_broken_load_path() {
        let problem = joined();
        let densities = vec![1.0; problem.grid.n_cells()];
        assert!(broken_load_paths(&problem, &densities).is_empty());

        // And a field with nothing in it at all is reported as such rather than
        // solved: no material means no path.
        let empty = vec![0.0; problem.grid.n_cells()];
        let broken = broken_load_paths(&problem, &empty);
        assert_eq!(broken.len(), 1, "{broken:?}");
        assert!(
            broken[0].contains("no material around them"),
            "{}",
            broken[0]
        );
    }

    /// A path through soft material is still a path. The threshold is about
    /// whether there is any stiffness at all, not about what the printed part
    /// is, so a bridge at a tenth density must not be called a break.
    #[test]
    fn a_soft_but_present_load_path_is_not_a_broken_one() {
        let problem = stranded();
        let open = joined();
        let soft = vec![0.1; open.grid.n_cells()];
        assert!(broken_load_paths(&open, &soft).is_empty());
        // Below the threshold there is nothing to carry anything.
        let vanishing = vec![constants::STRESS_LOAD_PATH_DENSITY * 0.5; open.grid.n_cells()];
        assert!(!broken_load_paths(&open, &vanishing).is_empty());
        // A gravity load acts on every element and can never be stranded.
        assert_eq!(problem.load_cases.len(), 1);
    }

    /// The stress pass answers a stop with "no report at all" rather than with a
    /// degraded one: nobody is going to read either.
    ///
    /// The bracket rather than the single element bar, because a solve that
    /// finishes before its first cancellation checkpoint has finished, and the
    /// bar's does.
    #[test]
    fn a_stopped_stress_pass_produces_nothing() {
        let config = Config::parse(BRACKET).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let densities = vec![1.0; problem.grid.n_cells()];
        let stop = || true;
        let outcome = analyse_with(
            &problem,
            &densities,
            StressLimits::default(),
            CancelProbe::watching(&stop),
        )
        .expect("a stop is not an error");
        assert!(outcome.is_none());
    }

    #[test]
    fn the_node_field_averages_the_evaluated_neighbours_only() {
        let problem = bar();
        let densities = vec![1.0; problem.grid.n_cells()];
        let report = analyse(&problem, &densities).expect("analyse");
        let field = node_stress_field(&problem, &densities, &report.cases[0].von_mises);
        let pad = constants::FIELD_PADDING_CELLS;
        assert_eq!(field.nx, problem.grid.nnx() + 2 * pad);
        // Every corner of the single element sees exactly that element.
        let value = field.at(pad, pad, pad);
        assert!((value - report.cases[0].max_mpa).abs() < 1e-9);
        // The padding ring touches nothing.
        assert_eq!(field.at(0, 0, 0), 0.0);
    }
}
