//! Removal of the material that carries nothing: the prongs that reach to
//! nothing and the gossamer wisps between members.
//!
//! A density based optimization converges on a *field*, not on a structure, and
//! the field it converges on keeps a little material wherever removing it cost
//! the objective nothing to leave. Thresholded at the iso level that material
//! becomes geometry: thin spurs standing off the load path, hairs bridging two
//! members, lobes that end in mid air. None of it is load bearing, all of it is
//! printed, and most of it needs support to print at all. This pass removes it
//! after the run, under `[output] trim`, which is `off` unless a configuration
//! asks for it.
//!
//! **The criterion is the stress envelope.** Every load case is already solved
//! once for the run's own stress report, so the pass costs no extra solve: the
//! envelope is the per-cell maximum von Mises stress over every case, and a part
//! cell is a candidate for removal when its envelope stress is below
//! [`constants::TRIM_STRESS_FRACTION`] of the envelope's own peak. The envelope
//! rather than any single case, because a member that is idle in one case and
//! working in another is working; a fraction of the peak rather than an absolute
//! stress, because the number has to mean the same thing for a bracket at 2 MPa
//! and a mount at 200.
//!
//! The pass is stress *informed*, which means it only ever removes material the
//! stress pass actually measured. An element below
//! [`constants::STRESS_DENSITY_THRESHOLD`] is left out of the stress report
//! entirely and carries a zero in every case, and that zero is an absence of
//! evidence rather than a measurement of none; condemning material on it would
//! be exactly the thing this pass must not do. So a cell is a candidate only if
//! the report covered it. The honest limitation that buys: below an `iso_level`
//! of 0.5 the part includes cells the report does not cover, and the wisps among
//! them survive the trim. At the default iso level the two thresholds are the
//! same number and there are no such cells at all.
//!
//! Two things are never removed however little stress they carry:
//!
//! * **Purposes.** The cells of every support region, every `[[keepin]]` entry
//!   and every non-gravity load region - the same set, assembled the same way,
//!   that the guide wireframe runs its wires between (see
//!   [`crate::engine::wireframe`]). A support pad is unstressed *because* it is
//!   held; a keepin is material the configuration asked for by name. Gravity
//!   declares no region - it acts on every element there is - so it protects
//!   nothing in particular.
//! * **Anything whose removal would take the structure apart.** Before and after
//!   the tentative removal, the purposes above are located on the connected
//!   bodies of the field, and the pairs of them that share a body are compared.
//!   If one pair that was joined no longer is, the *whole pass is refused*: not
//!   the offending cells, the pass. The field the caller holds is the field it
//!   handed in, the run exports untrimmed, and the report names the pair that
//!   would have been separated. A partial trim would be a different structure
//!   from either the one the optimizer produced or the one the user asked for,
//!   and nothing here is allowed to quietly ship one.
//!
//! Connectivity is face connectivity - six neighbours - which is
//! [`crate::voids`]'s reading for every structural question in growforge: a
//! diagonal touch is not a joint. Two members meeting only along an edge are two
//! members, so a removal that leaves them touching that way still counts as
//! having separated them, which is the conservative direction.
//!
//! What runs afterwards is the whole analysis again. Cavities are re-resolved
//! and the stresses re-solved on the trimmed field, so the stress table, the
//! safety factor, the void report and the STL all describe the geometry that
//! ships; the pre-trim stresses are the criterion and nothing else, and they
//! survive only as the two numbers in the trim note. The island culling of
//! [`crate::mesh::islands`] then runs where it always did and sweeps up any
//! fragment the removal orphaned.
//!
//! One consequence of running between the two: under `voids = "fill"` the pass
//! sees the cavity policy's own material, which carries no load by construction
//! and can therefore be condemned. The re-analysis fills the cavity again -
//! there is one cavity policy and it is applied to the field that ships - so the
//! exported part is right either way; what the trim's *count* reports is what it
//! removed, refilled cells included.
//!
//! Two things this pass is deliberately not:
//!
//! * **A printability check.** The overhang filter shapes the design *during*
//!   the optimization; trimming happens after it, and the cut faces it leaves
//!   are not re-checked against the build direction. That is the same standing
//!   smoothing and island culling have had since they were added - all three
//!   change the surface after the constraint has done its work - and the reason
//!   the material it removes is the material at the far end of the stress
//!   distribution rather than anything a slicer would have leant on.
//! * **`[growth] prune`.** That removes branches of a skeleton while the growth
//!   engine is still building one, by whether they reached anything; this
//!   removes cells of a finished density field, by the stress in them. The
//!   growth engine's own pruning means a grown design rarely has anything left
//!   for this to find, but the pass is engine agnostic and legal on either.

use rayon::prelude::*;

use crate::config::TrimPolicy;
use crate::constants;
use crate::engine::cell_density;
use crate::grid::Grid;
use crate::problem::Problem;
use crate::stress::{StressOutcome, StressReport};
use crate::voids;

/// One place the configuration declared a purpose for material.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Purpose {
    /// How the reports name it.
    pub label: String,
    /// The cells it covers, in ascending order. Empty when a keepout swallowed
    /// the region, which protects nothing and joins nothing.
    pub cells: Vec<usize>,
}

/// How the trim pass ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrimEnding {
    /// The cells were removed and the caller's field is the trimmed one.
    Removed,
    /// Nothing was below the threshold, so there was nothing to remove.
    NothingBelowThreshold,
    /// Removing them would have separated two purposes, so nothing was removed.
    /// The two strings are their labels.
    Refused {
        /// The first of the pair, in the order [`purposes`] assembles them.
        first: String,
        /// The second of the pair.
        second: String,
    },
    /// The run has no stress report, so the criterion does not exist. The string
    /// is why there is none.
    NoStressReport(String),
}

/// What the trim pass found and did.
#[derive(Debug, Clone, PartialEq)]
pub struct TrimReport {
    /// The policy that was in force.
    pub policy: TrimPolicy,
    /// Fraction of the envelope peak that was called negligible.
    pub fraction: f64,
    /// Peak of the envelope von Mises stress over the part, in MPa.
    pub peak_mpa: f64,
    /// Envelope stress a part cell had to stay below to be a candidate, in MPa.
    pub threshold_mpa: f64,
    /// Cells the part held before the pass.
    pub part_cells: usize,
    /// Cells that were removed - or, under [`TrimEnding::Refused`], that would
    /// have been.
    pub cells: usize,
    /// Volume those cells hold, in cubic millimetres.
    pub volume_mm3: f64,
    /// How the pass ended.
    pub ending: TrimEnding,
}

impl TrimReport {
    /// True when the field the caller handed in was changed, and therefore has
    /// to be analysed again before anything describes it.
    pub fn changed_the_field(&self) -> bool {
        self.ending == TrimEnding::Removed
    }

    /// Fraction of the part's cells the pass removed, or would have.
    pub fn removed_fraction(&self) -> f64 {
        if self.part_cells == 0 {
            return 0.0;
        }
        self.cells as f64 / self.part_cells as f64
    }

    /// What the run says about this pass, on the console and in the editor's
    /// panel alike.
    ///
    /// One line per sentence, an outcome that needs saying loudly opening with
    /// the word "warning", so a caller can print them straight or lay them out
    /// in a panel without parsing anything back out.
    pub fn notes(&self) -> Vec<String> {
        match &self.ending {
            TrimEnding::Removed => vec![format!(
                "removed {} cells, {:.3} cm3, {:.1}% of the part, below {:.4} MPa ({:.1}% of the \
                 {:.4} MPa envelope peak)",
                self.cells,
                self.volume_mm3 / constants::MM3_PER_CM3,
                100.0 * self.removed_fraction(),
                self.threshold_mpa,
                100.0 * self.fraction,
                self.peak_mpa
            )],
            TrimEnding::NothingBelowThreshold => vec![format!(
                "nothing below {:.4} MPa ({:.1}% of the {:.4} MPa envelope peak); the field is \
                 unchanged",
                self.threshold_mpa,
                100.0 * self.fraction,
                self.peak_mpa
            )],
            TrimEnding::Refused { first, second } => vec![
                format!(
                    "warning: refused, and the part was exported untrimmed: removing the {} cells \
                     below {:.4} MPa would have left {} joined to {} through no material at all",
                    self.cells, self.threshold_mpa, first, second
                ),
                "the design depends on material that carries almost no load, which is worth \
                 looking at before trimming it: a lower trim_stress_fraction removes less, and \
                 trim = \"off\" is what the run did anyway"
                    .to_string(),
            ],
            TrimEnding::NoStressReport(reason) => vec![format!(
                "warning: trim = \"stress\" has no stresses to judge by, so the part was exported \
                 untrimmed ({reason})"
            )],
        }
    }
}

/// Every place the configuration declared a purpose for material, in the order
/// the guide wireframe wires them: the support regions, the keepin entries, then
/// the load regions, each in configuration order.
///
/// Assembled exactly as [`crate::engine::wireframe`]'s terminals are, and for
/// the same reason: what a region *is*, as a set of cells, has to be one answer
/// throughout a run. A gravity load selects no region and contributes none.
pub fn purposes(problem: &Problem) -> Vec<Purpose> {
    let grid = &problem.grid;
    let mut purposes = Vec::new();
    for support in &problem.supports {
        purposes.push(Purpose {
            label: format!("support {}", support.index),
            cells: grid.cells_touching_nodes(&support.nodes),
        });
    }
    for keepin in &problem.keepins {
        // Already cell level, and already the cells the classifier pinned solid
        // for this entry.
        purposes.push(Purpose {
            label: format!("keepin {}", keepin.index),
            cells: keepin.cells.clone(),
        });
    }
    for case in &problem.load_cases {
        for (index, load) in case.loads.iter().enumerate() {
            if load.nodes.is_empty() {
                // Gravity: it acts on every element, so it names no place.
                continue;
            }
            purposes.push(Purpose {
                label: format!("load {} of case \"{}\"", index + 1, case.name),
                cells: grid.cells_touching_nodes(&load.nodes),
            });
        }
    }
    purposes
}

/// The per-cell maximum von Mises stress over every load case, in MPa.
///
/// Cells the stress pass left out, which are the ones below
/// [`constants::STRESS_DENSITY_THRESHOLD`], carry zero in every case and
/// therefore zero here. That zero is an absence of evidence and not a
/// measurement of none, which is why [`trim`] takes the same threshold into its
/// candidate test rather than reading a zero here as "carries nothing"; see the
/// module header.
pub fn envelope(report: &StressReport, cells: usize) -> Vec<f64> {
    let mut envelope = vec![0.0; cells];
    for case in &report.cases {
        for (slot, value) in envelope.iter_mut().zip(case.von_mises.iter()) {
            *slot = f64::max(*slot, *value);
        }
    }
    envelope
}

/// Which bodies each purpose stands on, and which pairs of purposes therefore
/// share one.
///
/// A purpose that holds no material at all stands on nothing and pairs with
/// nothing: it is not connected to anything now, so a removal cannot disconnect
/// it, and the island report is what says a declared region is unreached.
fn joined_pairs(
    grid: &Grid,
    densities: &[f64],
    iso: f64,
    purposes: &[Purpose],
) -> Vec<(usize, usize)> {
    let labels = voids::body_labels(grid, densities, iso);
    let bodies: Vec<Vec<usize>> = purposes
        .iter()
        .map(|purpose| {
            let mut bodies: Vec<usize> = purpose
                .cells
                .iter()
                .filter_map(|&cell| labels.of_cell[cell])
                .collect();
            bodies.sort_unstable();
            bodies.dedup();
            bodies
        })
        .collect();
    let mut pairs = Vec::new();
    for first in 0..bodies.len() {
        for second in first + 1..bodies.len() {
            if bodies[first].iter().any(|b| bodies[second].contains(b)) {
                pairs.push((first, second));
            }
        }
    }
    pairs
}

/// Trim `densities` against a stress envelope, or refuse to.
///
/// The core of the pass, taking the criterion and the places to protect rather
/// than reading them off a problem, so both are testable against a field written
/// by hand. [`resolve`] is what a run calls.
///
/// `iso` is the level the part is read at, and the level the connectivity guard
/// asks its question at: what may not come apart is the printed structure.
pub fn trim(
    grid: &Grid,
    densities: &mut [f64],
    iso: f64,
    envelope: &[f64],
    purposes: &[Purpose],
    fraction: f64,
) -> TrimReport {
    let n = grid.n_cells();
    let part: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|e| cell_density(grid, densities, e) >= iso)
        .collect();
    // Whether the stress pass evaluated the cell at all. Below its own
    // threshold it recorded a zero rather than a measurement, and an absence of
    // evidence may not condemn material; see the module header.
    let measured: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|e| cell_density(grid, densities, e) >= constants::STRESS_DENSITY_THRESHOLD)
        .collect();
    let part_cells = part.iter().filter(|&&is_part| is_part).count();
    let peak_mpa = (0..n)
        .filter(|&e| part[e])
        .map(|e| envelope[e])
        .fold(0.0, f64::max);
    let threshold_mpa = fraction * peak_mpa;

    let mut protected = vec![false; n];
    for purpose in purposes {
        for &cell in &purpose.cells {
            protected[cell] = true;
        }
    }
    let candidates: Vec<usize> = (0..n)
        .filter(|&e| part[e] && measured[e] && !protected[e] && envelope[e] < threshold_mpa)
        .collect();

    let report = |cells: usize, ending| TrimReport {
        policy: TrimPolicy::Stress,
        fraction,
        peak_mpa,
        threshold_mpa,
        part_cells,
        cells,
        volume_mm3: cells as f64 * grid.cell_volume(),
        ending,
    };
    if candidates.is_empty() {
        return report(0, TrimEnding::NothingBelowThreshold);
    }

    // The guard, on a copy: nothing the caller holds moves until the removal is
    // known to be one the structure survives.
    let before = joined_pairs(grid, densities, iso, purposes);
    let mut trimmed = densities.to_vec();
    for &cell in &candidates {
        trimmed[cell] = constants::DENSITY_MIN;
    }
    let after = joined_pairs(grid, &trimmed, iso, purposes);
    if let Some(&(first, second)) = before.iter().find(|pair| !after.contains(pair)) {
        return report(
            candidates.len(),
            TrimEnding::Refused {
                first: purposes[first].label.clone(),
                second: purposes[second].label.clone(),
            },
        );
    }

    densities.copy_from_slice(&trimmed);
    report(candidates.len(), TrimEnding::Removed)
}

/// Apply the configured policy to `densities`, reporting what it did.
///
/// `None` is `trim = "off"`: the pass did not run, and there is nothing to say
/// about it. Everything else comes back as a report, including the two endings
/// that leave the field alone - a removal the connectivity guard refused, and a
/// run whose stress solve produced no table to judge by.
pub fn resolve(
    problem: &Problem,
    densities: &mut [f64],
    stress: &StressOutcome,
) -> Option<TrimReport> {
    if problem.output.trim == TrimPolicy::Off {
        return None;
    }
    let fraction = problem.output.trim_stress_fraction;
    let Some(report) = stress.report() else {
        return Some(TrimReport {
            policy: problem.output.trim,
            fraction,
            peak_mpa: 0.0,
            threshold_mpa: 0.0,
            part_cells: 0,
            cells: 0,
            volume_mm3: 0.0,
            ending: TrimEnding::NoStressReport(
                stress
                    .reason()
                    .unwrap_or("the stress solve produced no report")
                    .to_string(),
            ),
        });
    };
    let envelope = envelope(report, problem.grid.n_cells());
    Some(trim(
        &problem.grid,
        densities,
        problem.output.iso_level,
        &envelope,
        &purposes(problem),
        fraction,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;
    use crate::grid::CellKind;

    /// A cube of design cells, `n` on a side, one millimetre each.
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

    /// A bar of material along x at `(j, k) = (0, 0)`, held at one end and
    /// loaded at the other, with a two-cell spur hanging off the middle of it.
    ///
    /// ```text
    ///   spur                 s
    ///                        s
    ///   bar    H b b b b b b b b b L
    /// ```
    ///
    /// Every bar cell carries stress, the spur carries none, and the two ends
    /// are the purposes. `n` is 12, so the indices below are readable.
    struct Bar {
        grid: Grid,
        densities: Vec<f64>,
        envelope: Vec<f64>,
        purposes: Vec<Purpose>,
        spur: Vec<usize>,
    }

    fn bar() -> Bar {
        let grid = design_grid(12);
        let mut densities = vec![0.0; grid.n_cells()];
        let mut envelope = vec![0.0; grid.n_cells()];
        for i in 0..12 {
            let cell = grid.cell_index(i, 0, 0);
            densities[cell] = 1.0;
            envelope[cell] = 10.0;
        }
        // The spur: material, joined to the bar by a face, carrying nothing.
        let spur: Vec<usize> = (1..3).map(|j| grid.cell_index(6, j, 0)).collect();
        for &cell in &spur {
            densities[cell] = 1.0;
        }
        let purposes = vec![
            Purpose {
                label: "support 1".to_string(),
                cells: vec![grid.cell_index(0, 0, 0)],
            },
            Purpose {
                label: "load 1 of case \"tip\"".to_string(),
                cells: vec![grid.cell_index(11, 0, 0)],
            },
        ];
        Bar {
            grid,
            densities,
            envelope,
            purposes,
            spur,
        }
    }

    #[test]
    fn a_spur_that_carries_nothing_is_removed_and_reported() {
        let Bar {
            grid,
            mut densities,
            envelope,
            purposes,
            spur,
        } = bar();
        let before = densities.clone();
        let report = trim(
            &grid,
            &mut densities,
            0.5,
            &envelope,
            &purposes,
            constants::TRIM_STRESS_FRACTION,
        );

        assert_eq!(report.ending, TrimEnding::Removed);
        assert!(report.changed_the_field());
        assert_eq!(report.cells, spur.len());
        assert_eq!(report.part_cells, 12 + spur.len());
        assert!((report.volume_mm3 - spur.len() as f64).abs() < 1e-12);
        assert!((report.peak_mpa - 10.0).abs() < 1e-12);
        assert!((report.threshold_mpa - 0.1).abs() < 1e-12);
        assert!((report.removed_fraction() - 2.0 / 14.0).abs() < 1e-12);

        // Exactly the spur left, and the bar is untouched.
        let changed: Vec<usize> = (0..grid.n_cells())
            .filter(|&e| before[e] != densities[e])
            .collect();
        assert_eq!(changed, spur);
        for &cell in &changed {
            assert_eq!(densities[cell], constants::DENSITY_MIN);
        }

        // And the note carries the numbers the pre-trim field is the only source
        // of, because that field is about to be thrown away.
        let note = report.notes().join(" ");
        assert!(note.contains("removed 2 cells"), "{note}");
        assert!(note.contains("14.3% of the part"), "{note}");
        assert!(note.contains("10.0000 MPa envelope peak"), "{note}");
    }

    #[test]
    fn a_purpose_cell_below_the_threshold_is_kept() {
        let Bar {
            grid,
            mut densities,
            mut envelope,
            purposes,
            spur,
        } = bar();
        // The support pad is unstressed because it is held, which is exactly the
        // reading that must not cost it its material.
        let pad = grid.cell_index(0, 0, 0);
        envelope[pad] = 0.0;
        let report = trim(
            &grid,
            &mut densities,
            0.5,
            &envelope,
            &purposes,
            constants::TRIM_STRESS_FRACTION,
        );
        assert_eq!(report.ending, TrimEnding::Removed);
        assert_eq!(report.cells, spur.len());
        assert_eq!(densities[pad], 1.0, "a declared region lost its material");
    }

    #[test]
    fn a_low_stress_bridge_whose_removal_would_disconnect_two_purposes_aborts_the_pass() {
        let grid = design_grid(12);
        let mut densities = vec![0.0; grid.n_cells()];
        let mut envelope = vec![0.0; grid.n_cells()];
        // The same bar, but the middle two cells carry nothing: the load path
        // runs through material the criterion calls negligible.
        for i in 0..12 {
            let cell = grid.cell_index(i, 0, 0);
            densities[cell] = 1.0;
            envelope[cell] = if (5..7).contains(&i) { 0.0 } else { 10.0 };
        }
        let purposes = vec![
            Purpose {
                label: "support 1".to_string(),
                cells: vec![grid.cell_index(0, 0, 0)],
            },
            Purpose {
                label: "load 1 of case \"tip\"".to_string(),
                cells: vec![grid.cell_index(11, 0, 0)],
            },
        ];
        let before = densities.clone();
        let report = trim(
            &grid,
            &mut densities,
            0.5,
            &envelope,
            &purposes,
            constants::TRIM_STRESS_FRACTION,
        );

        assert_eq!(
            report.ending,
            TrimEnding::Refused {
                first: "support 1".to_string(),
                second: "load 1 of case \"tip\"".to_string(),
            }
        );
        assert!(!report.changed_the_field());
        assert_eq!(report.cells, 2, "the count is what it would have removed");
        assert_eq!(densities, before, "a refused pass may not touch the field");

        // Loudly, and naming both ends of what it would have broken.
        let note = report.notes().join(" ");
        assert!(note.starts_with("warning: refused"), "{note}");
        assert!(note.contains("support 1"), "{note}");
        assert!(note.contains("load 1 of case \"tip\""), "{note}");
    }

    /// The envelope is the maximum over the cases, so a cell that is idle in
    /// every case but one is working material.
    #[test]
    fn a_cell_hot_in_one_load_case_only_survives_the_envelope() {
        let Bar {
            grid,
            mut densities,
            purposes,
            spur,
            ..
        } = bar();
        let cells = grid.n_cells();
        let mut quiet = vec![0.0; cells];
        let mut loaded = vec![0.0; cells];
        for i in 0..12 {
            let cell = grid.cell_index(i, 0, 0);
            quiet[cell] = 10.0;
            loaded[cell] = 10.0;
        }
        // The spur is hot in the second case alone.
        for &cell in &spur {
            loaded[cell] = 4.0;
        }
        let report = StressReport {
            cases: vec![
                crate::stress::CaseStress {
                    name: "quiet".to_string(),
                    max_mpa: 10.0,
                    percentile_mpa: 10.0,
                    top_fraction_mean_mpa: 10.0,
                    safety_factor: None,
                    von_mises: quiet,
                },
                crate::stress::CaseStress {
                    name: "loaded".to_string(),
                    max_mpa: 10.0,
                    percentile_mpa: 10.0,
                    top_fraction_mean_mpa: 10.0,
                    safety_factor: None,
                    von_mises: loaded,
                },
            ],
            yield_strength_mpa: None,
            density_threshold: constants::STRESS_DENSITY_THRESHOLD,
            evaluated_cells: 14,
        };
        let envelope = envelope(&report, cells);
        for &cell in &spur {
            assert_eq!(envelope[cell], 4.0, "the envelope lost a case");
        }
        let before = densities.clone();
        let trimmed = trim(
            &grid,
            &mut densities,
            0.5,
            &envelope,
            &purposes,
            constants::TRIM_STRESS_FRACTION,
        );
        assert_eq!(trimmed.ending, TrimEnding::NothingBelowThreshold);
        assert_eq!(trimmed.cells, 0);
        assert_eq!(densities, before);
    }

    /// The fraction is the knob, and it is honoured: at four tenths of the peak
    /// the spur of the previous test is negligible after all.
    #[test]
    fn the_configured_fraction_decides_what_counts_as_negligible() {
        let Bar {
            grid,
            mut densities,
            mut envelope,
            purposes,
            spur,
        } = bar();
        for &cell in &spur {
            envelope[cell] = 4.0;
        }
        let kept = trim(
            &grid,
            &mut densities.clone(),
            0.5,
            &envelope,
            &purposes,
            0.3,
        );
        assert_eq!(kept.ending, TrimEnding::NothingBelowThreshold);

        let report = trim(&grid, &mut densities, 0.5, &envelope, &purposes, 0.5);
        assert_eq!(report.ending, TrimEnding::Removed);
        assert_eq!(report.cells, spur.len());
        assert!((report.threshold_mpa - 5.0).abs() < 1e-12);
    }

    /// A part with nothing below the threshold is left exactly as it was, and
    /// says so rather than reporting a removal of nothing.
    #[test]
    fn a_part_that_is_all_load_path_is_untouched() {
        let Bar {
            grid,
            mut densities,
            mut envelope,
            purposes,
            spur,
        } = bar();
        for &cell in &spur {
            envelope[cell] = 10.0;
        }
        let before = densities.clone();
        let report = trim(
            &grid,
            &mut densities,
            0.5,
            &envelope,
            &purposes,
            constants::TRIM_STRESS_FRACTION,
        );
        assert_eq!(report.ending, TrimEnding::NothingBelowThreshold);
        assert!(!report.changed_the_field());
        assert_eq!(report.cells, 0);
        assert_eq!(report.volume_mm3, 0.0);
        assert_eq!(densities, before);
        assert!(
            report.notes()[0].starts_with("nothing below"),
            "{:?}",
            report.notes()
        );
    }

    /// Material the stress report never covered is never removed, however low
    /// the zero it carries here looks.
    ///
    /// The only way to have such a cell is an `iso_level` below
    /// [`constants::STRESS_DENSITY_THRESHOLD`], where the part includes cells
    /// the report leaves out. Their zero is an absence of evidence rather than a
    /// measurement of none, and this pass condemns nothing on it - so a wisp
    /// down there survives the trim, which is the limitation the rule buys.
    #[test]
    fn a_part_cell_the_stress_pass_never_evaluated_is_not_a_candidate() {
        let Bar {
            grid,
            mut densities,
            envelope,
            purposes,
            spur,
        } = bar();
        // Half of the spur sits between a low iso level and the stress pass's
        // own threshold: it is part of the part, and it is not in the report.
        let unmeasured = spur[1];
        densities[unmeasured] = 0.4;
        let iso = 0.3;
        assert!(iso < constants::STRESS_DENSITY_THRESHOLD);
        let report = trim(
            &grid,
            &mut densities,
            iso,
            &envelope,
            &purposes,
            constants::TRIM_STRESS_FRACTION,
        );

        assert_eq!(report.ending, TrimEnding::Removed);
        assert_eq!(
            report.part_cells,
            12 + spur.len(),
            "the unevaluated cell is still part of the part"
        );
        assert_eq!(report.cells, 1, "only the evaluated spur cell may go");
        assert_eq!(densities[spur[0]], constants::DENSITY_MIN);
        assert_eq!(
            densities[unmeasured], 0.4,
            "material the report never covered was condemned on its own absence"
        );
    }

    /// A cell joined to the part only across an edge is joined to nothing: the
    /// guard reads connectivity the way every other structural pass in
    /// growforge does.
    #[test]
    fn the_guard_counts_a_diagonal_touch_as_no_joint() {
        let grid = design_grid(6);
        let mut densities = vec![0.0; grid.n_cells()];
        let mut envelope = vec![0.0; grid.n_cells()];
        // Two blocks meeting only at a corner, one purpose in each, and the
        // corner cells carrying nothing.
        let left = grid.cell_index(1, 1, 1);
        let right = grid.cell_index(2, 2, 2);
        for &cell in &[left, right] {
            densities[cell] = 1.0;
            envelope[cell] = 10.0;
        }
        let idle = grid.cell_index(3, 2, 2);
        densities[idle] = 1.0;
        let purposes = vec![
            Purpose {
                label: "support 1".to_string(),
                cells: vec![left],
            },
            Purpose {
                label: "load 1 of case \"tip\"".to_string(),
                cells: vec![right],
            },
        ];
        // The two purposes were never joined, so removing the idle cell beside
        // one of them separates nothing and the pass goes ahead.
        let report = trim(
            &grid,
            &mut densities,
            0.5,
            &envelope,
            &purposes,
            constants::TRIM_STRESS_FRACTION,
        );
        assert_eq!(report.ending, TrimEnding::Removed);
        assert_eq!(report.cells, 1);
        assert_eq!(densities[idle], constants::DENSITY_MIN);
    }

    /// The two policy paths of [`resolve`], on a problem rather than a
    /// hand-written field: `off` does nothing at all, and a run with no stress
    /// table says so instead of trimming on no evidence.
    #[test]
    fn the_policy_decides_whether_the_pass_runs_at_all() {
        use crate::config::Config;

        const TINY: &str = r#"
[project]
name = "trim_policy"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 16.0

[output]
stl_path = "trim_policy.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [32.0, 16.0, 16.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 16.5, 16.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [32.0, 8.0, 8.0], radius = 4.0 }
vector = [0.0, 0.0, -20.0]
"#;

        let config = Config::parse(TINY).expect("parse");
        let mut problem = Problem::build(&config, &std::env::temp_dir()).expect("build");
        let mut densities = vec![1.0; problem.grid.n_cells()];
        let before = densities.clone();
        let unavailable = StressOutcome::Unavailable("the solve did not converge".to_string());

        assert_eq!(problem.output.trim, TrimPolicy::Off);
        assert!(
            resolve(&problem, &mut densities, &unavailable).is_none(),
            "the pass ran with trim = \"off\""
        );

        problem.output.trim = TrimPolicy::Stress;
        let report = resolve(&problem, &mut densities, &unavailable).expect("a report");
        assert_eq!(
            report.ending,
            TrimEnding::NoStressReport("the solve did not converge".to_string())
        );
        assert!(!report.changed_the_field());
        assert_eq!(densities, before);
        let note = report.notes().join(" ");
        assert!(note.starts_with("warning: trim"), "{note}");
        assert!(note.contains("did not converge"), "{note}");

        // And the purposes it would have protected are the regions the
        // configuration declared, named as every other report names them.
        let labels: Vec<String> = purposes(&problem)
            .into_iter()
            .map(|purpose| purpose.label)
            .collect();
        assert_eq!(labels, ["support 1", "load 1 of case \"tip\""]);
        for purpose in purposes(&problem) {
            assert!(
                !purpose.cells.is_empty(),
                "{} selected nothing",
                purpose.label
            );
        }
    }
}
