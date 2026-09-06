//! SIMP topology optimization: solid isotropic material with penalization,
//! density filtering and a volume constrained design variable update.
//!
//! One iteration is
//! 1. push the design variables through the density chain into the printed
//!    physical densities,
//! 2. interpolate `E(x) = Emin + x^p (E0 - Emin)`,
//! 3. solve `K u = f` for every load case (warm started, self weight included),
//! 4. accumulate the weighted compliance and its sensitivities,
//! 5. push the sensitivities back through the chain transposes,
//! 6. take one update step under the volume constraint, with the scheme
//!    `[optimization] update` selected.
//!
//! With `[optimization.wireframe]` set, the run starts from a seeded guide
//! instead of a uniform field and step 6 is followed by a density floor along
//! that guide for the first `hold_iterations` iterations. Nothing else changes -
//! except that a design under the floor is not free, so neither of the two
//! verdicts about a settling design is asked until the floor comes off: see
//! [`crate::engine::wireframe`].
//!
//! With `[optimization.reduce]` set the loop above is one *stage* of a
//! schedule: the design starts at `mass_fraction` - solid unless the file says
//! otherwise - and each stage re-converges it at a lower volume target,
//! `ratio` of the one before, until the design no longer holds
//! `target_safety_factor`. Everything the schedule needs stays inside this one
//! call - one solver, one workspace, one design carried from stage to stage as
//! its warm start - and what comes out is the lightest stage that held the
//! target. See [`SimpEngine::optimize`]'s stage section.
//!
//! The loop stops for one of three reasons, reported apart because they mean
//! different things about the design that comes out: the convergence tolerance
//! (an answer), the stall criterion of [`crate::engine::stall`] (an iterate the
//! problem will not improve on), or the iteration cap (whatever the budget ended
//! on). All three export. The first two are unreachable while a guide's floor is
//! held, so a run that ends inside the hold window ended on the cap or on a
//! cancellation, and says that its design carries the wire as forced material.

use std::time::Instant;

use anyhow::{Result, bail};

use crate::config::{ReduceMethod, ReduceMethodParams};
use crate::constants;
use crate::engine::chain::DensityChain;
use crate::engine::local_volume::LocalVolume;
use crate::engine::objective::{Objective, Workspace};
use crate::engine::stall::{Sample, StallWatch, stall_note};
use crate::engine::update::{Buffers, Constraint, LocalConstraint, Updater, design_mean};
use crate::engine::wireframe;
use crate::engine::{
    DensityField, Engine, OverhangResidual, ReduceStage, ReduceSummary, StopReason,
};
use crate::fea::{CancelProbe, LinearSolver};
use crate::grid::CellKind;
use crate::problem::Problem;
use crate::report::{
    IterationStats, ReduceProgress, Reporter, reduce_stage_note, reduce_summary_note,
};
use crate::stress::{self, StressLimits, StressOutcome};

/// The SIMP engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpEngine;

/// The design of a reduction stage that held the target safety factor, kept
/// whole so that the schedule can export it after later stages have moved on.
///
/// Both ends of the density chain travel together: the printed field is the
/// design, and the filtered one is what the overhang residual is measured
/// against, so a design exported out of an earlier stage is described by that
/// stage's numbers throughout.
struct Kept {
    /// Printed physical densities of the stage's design.
    printed: Vec<f64>,
    /// The chain's intermediate stage for the same design.
    filtered: Vec<f64>,
    /// Mean printed density over the design cells.
    volume_fraction: f64,
    /// Weighted compliance of the stage's last analysed design.
    compliance: f64,
    /// Largest design variable change of the stage's last iteration.
    max_change: f64,
    /// The record the summary reports this stage by.
    stage: ReduceStage,
}

impl Engine for SimpEngine {
    fn name(&self) -> &'static str {
        constants::DEFAULT_ENGINE
    }

    fn optimize(&self, problem: &Problem, reporter: &dyn Reporter) -> Result<DensityField> {
        let grid = &problem.grid;
        let n_cells = grid.n_cells();
        let params = problem.optimization;
        let reduce = params.reduce;
        // The evolutionary method shares the stage schedule below but not the
        // update rule, and that rule is not in this build. Refused here rather
        // than quietly run under the other one: a run that reported a method it
        // did not use would be worse than no run.
        if let Some(reduce) = reduce
            && matches!(reduce.method, ReduceMethodParams::Beso { .. })
        {
            bail!(
                "[optimization.reduce] method = \"{}\" is not implemented in this build; \
                 method = \"{}\" is the one it has",
                reduce.method.kind().label(),
                ReduceMethod::Continuation.label()
            );
        }

        let design_cells: Vec<usize> = (0..n_cells)
            .filter(|&e| grid.cells[e] == CellKind::Design)
            .collect();
        let chain = DensityChain::new(grid, params.filter_radius_mm, params.overhang);
        let objective = Objective::new(problem, &chain);
        // One device, one set of buffers, for the whole run: the design is
        // re-uploaded per iteration, everything else stays put.
        let mut solver = LinearSolver::new(problem)?;
        if solver.kind() != crate::config::SolverBackend::Cpu {
            reporter.note(&format!("linear solves on {}", solver.description()));
        }
        // The same question the loop asks between iterations, carried down into
        // the solves themselves so that a stop pressed during one of them takes
        // effect inside it rather than after it. A reporter that never asks for
        // a stop - every console one - makes this a load of a constant `false`
        // once per checkpoint.
        let stop = || reporter.cancelled();
        let cancel = CancelProbe::watching(&stop);

        // Design variables: void cells stay at zero, forced solid cells at one.
        let mut x = vec![0.0; n_cells];
        for (e, cell) in grid.cells.iter().enumerate() {
            x[e] = match cell {
                CellKind::Void => 0.0,
                CellKind::Solid => 1.0,
                CellKind::Design => params.mass_fraction,
            };
        }
        // The guide wireframe, when one was asked for: a thin network joining
        // every load, support and keepin region, seeded into the initial design
        // and held under it for the first iterations. A guide, not a constraint -
        // see [`wireframe`].
        let wireframe = wireframe::Wireframe::build(problem, reporter);
        if let Some(wire) = &wireframe {
            wire.seed(&mut x);
        }

        let mut work = Workspace::new(problem);
        let mut x_new = vec![0.0; n_cells];
        let mut dc = vec![0.0; n_cells];
        let mut dv = vec![0.0; n_cells];
        let mut updater = Updater::new(params.update, n_cells);
        // The local volume cap, when one was asked for: a second constraint, its
        // own kernel and its own gradient. Nothing is allocated for it when the
        // table is absent, which is what keeps a plain run's working set the one
        // it always had.
        let mut local = params.local_volume.map(|p| LocalVolume::new(grid, p));
        let mut dg = match local {
            Some(_) => vec![0.0; n_cells],
            None => Vec::new(),
        };
        if let Some(cap) = &local {
            reporter.note(&format!(
                "local volume: no neighbourhood of radius {:.3} mm ({} stencil taps) may hold \
                 more than {:.3} of its material",
                params
                    .local_volume
                    .expect("the cap was built from these params")
                    .radius_mm,
                cap.stencil_size(),
                cap.max_fraction()
            ));
        }

        let start = Instant::now();
        let mut compliance = 0.0;
        let mut initial_compliance = 0.0;
        let mut volume_fraction = 0.0;
        let mut max_change = f64::INFINITY;
        let mut iterations = 0;
        let mut shift_reported = false;
        // Why the stage below stopped. Assigned at the top of every stage, so a
        // schedule reads the reason of the stage it is in rather than the one
        // before it.
        let mut stop;

        // The reduction schedule of `[optimization.reduce]`. Without the table
        // there is one stage, its volume target is `mass_fraction`, and
        // everything below the stage bookkeeping is the loop this engine always
        // ran - which is what keeps a plain run's trajectory the one it had.
        // With the table, `mass_fraction` is where the schedule starts instead.
        let mut stage_target = params.mass_fraction;
        let mut stage_index = 0;
        let mut stage_refine = false;
        let mut stages: Vec<ReduceStage> = Vec::new();
        // The lightest design that held the target safety factor, kept whole
        // because the stages after it are free to be worse than it.
        let mut kept: Option<Kept> = None;
        // The bracket the refinement bisects: the lowest volume target that held
        // the target safety factor, and the highest that did not. The schedule
        // descends, so the failing one is the lower of the two.
        let (mut passing_target, mut failing_target) = (None, None);
        let mut refinements_left = reduce.map_or(0, |r| r.refine_stages);

        loop {
            stage_index += 1;
            let stage_start = iterations;
            let mut stage_iterations = 0;
            let progress = reduce.map(|_| ReduceProgress {
                stage: stage_index,
                target_fraction: stage_target,
                completed: stages.len(),
            });
            // Which stage a note belongs to, and nothing at all on a run without
            // a schedule - whose notes are then the ones it always printed.
            let stage_prefix = match reduce {
                Some(_) => format!("reduce stage {stage_index}: "),
                None => String::new(),
            };
            // A design variable change is read against how far the configured
            // scheme was allowed to move one, so the watch is told which scheme
            // is running. It is built per stage: settling is a question about
            // one volume target, and the window that answers it holds the
            // iterations taken at that target alone.
            let mut watch = StallWatch::new(params.update.move_limit());
            // Nothing but running out of budget stops a loop that runs to its
            // end, so that is what the reason is until something else happens.
            stop = StopReason::IterationCap;

            for iteration in 1..=params.max_iterations {
                // A stop inside the solve unwinds the whole iteration: it produced
                // no compliance and no sensitivities, so it never happened. The
                // design stays the one the last completed iteration left, which is
                // what `work.printed` still holds.
                let Some(value) =
                    objective.evaluate(&x, &mut work, &mut dc, &mut solver, cancel)?
                else {
                    stop = StopReason::Cancelled;
                    break;
                };
                stage_iterations = iteration;
                iterations = stage_start + iteration;
                compliance = value.compliance;

                // The volume sensitivity is the chain transpose of a unit field. It
                // only moves with the design when the chain is nonlinear, which is
                // what the overhang filter makes it.
                if iteration == 1 || !chain.is_linear() {
                    objective.volume_sensitivity(&mut work, &design_cells, &mut dv);
                }
                // The local cap's is recomputed every iteration unconditionally: the
                // p-mean that aggregates it is nonlinear in the design itself, so it
                // moves even where the chain above does not.
                let mut worst_local_fraction = None;
                let mut local_residual = None;
                if let Some(cap) = local.as_mut() {
                    worst_local_fraction = Some(cap.measure(&work.printed, &design_cells).worst);
                    cap.sensitivity(
                        &chain,
                        &work.filtered,
                        &work.printed,
                        &design_cells,
                        &mut dg,
                    );
                    local_residual = Some(cap.residual());
                }

                let constraint = Constraint {
                    design_cells: &design_cells,
                    target_volume_fraction: stage_target,
                    chain: &chain,
                    local: local_residual.map(|value| LocalConstraint {
                        value,
                        gradient: &dg,
                        current_volume_fraction: design_mean(&work.printed, &design_cells),
                    }),
                };

                let mut step = updater.update(
                    &x,
                    &dc,
                    &dv,
                    &constraint,
                    Buffers {
                        x_new: &mut x_new,
                        filtered: &mut work.filtered,
                        printed: &mut work.printed,
                    },
                )?;
                // The guide's floor goes on here, before anything reads the step: it
                // is the design the next iteration will analyse, so it is the design
                // the reporter, the live density field and the stall watch have to
                // see. Outside the hold window this is the step exactly as it came.
                if let Some(wire) = &wireframe {
                    step = wire.hold(
                        iteration,
                        &x,
                        step,
                        &constraint,
                        Buffers {
                            x_new: &mut x_new,
                            filtered: &mut work.filtered,
                            printed: &mut work.printed,
                        },
                        reporter,
                    );
                }
                max_change = step.max_change;
                volume_fraction = step.volume_fraction;
                // The first iteration of the run, not of the stage: what the
                // compliance of the design the run was handed was.
                if iterations == 1 {
                    initial_compliance = compliance;
                }
                if step.shift > 0.0 && !shift_reported {
                    shift_reported = true;
                    reporter.note(&format!(
                        "iteration {iteration}: a design dependent load made some sensitivities \
                         positive; the optimality criteria step is running with the self-weight \
                         sensitivity shift from here on"
                    ));
                }

                let stats = IterationStats {
                    iteration,
                    compliance,
                    volume_fraction,
                    worst_local_fraction,
                    max_change,
                    cg_iterations: value.cg_iterations,
                    elapsed_s: start.elapsed().as_secs_f64(),
                    growth: None,
                    reduce: progress,
                };
                reporter.iteration(&stats);
                // `work.printed` holds the physical densities of the trial design
                // the update step just accepted, which is exactly what a live
                // preview should show.
                reporter.densities(&stats, &work.printed);

                x.copy_from_slice(&x_new);
                // Checked here rather than at the top of the loop so that a
                // cancelled run always has at least one finished iteration behind
                // it, and therefore a physical density field that means something.
                if reporter.cancelled() {
                    stop = StopReason::Cancelled;
                    break;
                }
                // Both verdicts about a settling design wait for the guide's floor to
                // come off. Under the floor the design is not free: the change an
                // iteration reports is partly the floor pushing its cells back up, so
                // a change below the tolerance is not the design arriving, and a
                // window of held iterations would be a verdict about the guide rather
                // than about the problem. The watch is not fed either, so its window
                // holds free iterations alone and starts filling at the release -
                // which keeps the window meaning what its constant says it means even
                // when the hold is longer than one. Nothing but the iteration cap and
                // a cancellation can therefore end a run while the wire is forced,
                // and both of those say so below.
                if wireframe.as_ref().is_none_or(|wire| !wire.holds(iteration)) {
                    if max_change < params.convergence_tol {
                        stop = StopReason::Converged;
                        reporter.note(&format!(
                            "{stage_prefix}converged after {iteration} iterations (max change \
                             {max_change:.5} < {:.5})",
                            params.convergence_tol
                        ));
                        break;
                    }
                    // Asked after the convergence test, never before it: settling is
                    // the good outcome and a run that has just settled is not
                    // stalled, whatever its window looks like.
                    if watch.observe(Sample {
                        compliance,
                        max_change,
                    }) {
                        stop = StopReason::Stalled;
                        reporter.note(&format!(
                            "{stage_prefix}{}",
                            stall_note(
                                iteration,
                                max_change,
                                params.convergence_tol,
                                params.update,
                                params.local_volume.is_some(),
                            )
                        ));
                        break;
                    }
                }
            }

            match stop {
                StopReason::Cancelled => {
                    // Zero completed iterations is reachable now that a stop is
                    // honoured inside a solve: there is no max change to quote
                    // for an iteration that never produced one.
                    reporter.note(&match iterations {
                        0 => {
                            "cancelled inside the first iteration; nothing was computed".to_string()
                        }
                        n => format!("cancelled after {n} iterations (max change {max_change:.5})"),
                    });
                }
                StopReason::IterationCap => reporter.note(&format!(
                    "{stage_prefix}stopped at the iteration cap of {} (max change \
                     {max_change:.5})",
                    params.max_iterations
                )),
                // Both of these announced themselves where they happened, with
                // the iteration they happened on, and a schedule that ran out of
                // stages says what it exported in its own summary.
                StopReason::Converged | StopReason::Stalled | StopReason::ReduceComplete => {}
            }

            // Everything below is the schedule's. A run without one is the run
            // that just finished.
            let Some(reduce) = reduce else { break };
            if stop == StopReason::Cancelled {
                break;
            }

            // The stage's verdict. The connectivity gate goes first: a design
            // whose loads reach no support through material drives a system no
            // conjugate gradient will resolve, and the flood fill answers that
            // for the price of a pass over the grid rather than of a solve.
            let broken = stress::broken_load_paths(problem, &work.printed);
            let safety_factor = if broken.is_empty() {
                match stress::analyse_with_solver(
                    problem,
                    &work.printed,
                    StressLimits::default(),
                    &mut solver,
                    cancel,
                )? {
                    Some(StressOutcome::Available(report)) => {
                        let factor = report.worst_safety_factor();
                        // A report with no factor in it: this design has no peak
                        // von Mises stress to divide the yield strength by. The
                        // stage fails like any other unmeasured one, and says
                        // which way it was unmeasurable rather than leaving an
                        // `n/a` to be read as a solve that went wrong.
                        if factor.is_none() {
                            reporter.note(&format!(
                                "{stage_prefix}no safety factor for this stage: {} of its \
                                 elements reach the stress report's density threshold of {:.2}, \
                                 so it has no peak stress to measure",
                                report.evaluated_cells, report.density_threshold
                            ));
                        }
                        factor
                    }
                    // A stage nobody could measure did not hold the target: the
                    // schedule is a claim about the design it exports, and an
                    // unmeasured design cannot carry one.
                    Some(StressOutcome::Unavailable(reason)) => {
                        reporter.note(&format!(
                            "{stage_prefix}no safety factor for this stage: {reason}"
                        ));
                        None
                    }
                    // Called off inside one of the stage's load case solves.
                    // Nothing is exported from a cancelled run, so there is no
                    // schedule to finish either.
                    None => {
                        stop = StopReason::Cancelled;
                        reporter.note(&format!(
                            "{stage_prefix}cancelled inside the stress report of the stage"
                        ));
                        break;
                    }
                }
            } else {
                reporter.note(&format!("{stage_prefix}{}", broken.join("; ")));
                None
            };
            let passed = safety_factor.is_some_and(|f| f >= reduce.target_safety_factor);

            let record = ReduceStage {
                index: stage_index,
                target_fraction: stage_target,
                achieved_fraction: volume_fraction,
                iterations: stage_iterations,
                safety_factor,
                passed,
                refine: stage_refine,
            };
            let mut note = reduce_stage_note(&record);
            // A stage that failed while it was still moving may have run out of
            // budget rather than out of material, and the two are worth telling
            // apart before the schedule's verdict is believed.
            if !passed && stop == StopReason::IterationCap {
                note.push_str(
                    "; the stage stopped at the iteration cap, so raising max_iterations may be \
                     what it needs",
                );
            }
            reporter.note(&note);
            stages.push(record);

            if passed {
                // Every stage the schedule runs after this one is lighter, so
                // this can only ever be an improvement; comparing anyway is what
                // makes "the lightest design that held" a property of the code
                // rather than of the schedule that fed it.
                if kept
                    .as_ref()
                    .is_none_or(|held| volume_fraction < held.volume_fraction)
                {
                    kept = Some(Kept {
                        printed: work.printed.clone(),
                        filtered: work.filtered.clone(),
                        volume_fraction,
                        compliance,
                        max_change,
                        stage: record,
                    });
                }
                passing_target = Some(stage_target);
            } else {
                failing_target = Some(stage_target);
                // The first stage is the fraction the run started from, solid
                // unless the file said otherwise. Nothing lighter can hold a
                // target that one missed, so there is no bracket to bisect and
                // nothing left to look for: the schedule stops, exports the
                // design it started from, and warns in its summary that it did.
                if passing_target.is_none() {
                    break;
                }
            }

            match (passing_target, failing_target) {
                // The bracket is closed, so every stage from here bisects it -
                // as many of them as `refine_stages` asked for and no more,
                // because each one is a whole re-converged design and a whole
                // stress report.
                (Some(pass), Some(fail)) => {
                    if refinements_left == 0 {
                        break;
                    }
                    refinements_left -= 1;
                    stage_refine = true;
                    stage_target = (pass + fail) / 2.0;
                }
                // Still descending. The floor is a stage of its own - it is
                // asked for like any other - and a schedule that reaches it and
                // still holds the target has nowhere left to go.
                _ => {
                    if stage_target <= reduce.min_mass_fraction {
                        break;
                    }
                    stage_target = (stage_target * reduce.ratio).max(reduce.min_mass_fraction);
                }
            }
        }

        // What the schedule decided, and the design it decided on. A cancelled
        // run has neither: nothing is exported from one, so there is no stage to
        // report as the one that shipped.
        let mut reduce_summary = None;
        if let Some(reduce) = reduce
            && stop != StopReason::Cancelled
        {
            let exported = match kept {
                // The design in the file is the lightest stage that held the
                // target, and the numbers that describe it are that stage's
                // rather than those of the last stage the loop happened to run.
                // The filtered field travels with the printed one so that the
                // overhang residual below is measured on the design that
                // shipped, not across two stages.
                Some(held) => {
                    work.printed = held.printed;
                    work.filtered = held.filtered;
                    volume_fraction = held.volume_fraction;
                    compliance = held.compliance;
                    max_change = held.max_change;
                    held.stage
                }
                // No stage held the target, which can only be the first one: the
                // run exports the design it started from, which is what the
                // workspace still holds.
                None => *stages.first().expect("a finished schedule ran a stage"),
            };
            stop = StopReason::ReduceComplete;
            let summary = ReduceSummary {
                method: reduce.method.kind(),
                target_safety_factor: reduce.target_safety_factor,
                exported,
                stages,
                // Measured by `crate::complete`, on the field the `[output]`
                // passes leave: the schedule ends here, and those passes have
                // not run yet.
                finished_safety_factor: None,
            };
            reporter.note(&reduce_summary_note(&summary));
            reduce_summary = Some(summary);
        }

        // A run that ended inside the hold window ended on a design the optimizer
        // was not free to settle: the wire is still forced material in it. Only
        // the iteration cap and a cancellation can get here - the two verdicts
        // wait for the release - and neither of them is a reason to keep quiet
        // about what the design carries. A cancelled run that never finished an
        // iteration has no design to say it of.
        if let Some(wire) = &wireframe
            && iterations > 0
            && wire.holds(iterations)
        {
            reporter.note(&format!(
                "warning: [optimization.wireframe]: the run stopped after {iterations} iterations \
                 with the guide's floor still active (hold_iterations = {}), so the design it \
                 ended on carries the wire as forced material rather than as a guide; raise \
                 max_iterations past hold_iterations, or lower hold_iterations, to let the \
                 optimizer settle it",
                wire.hold_iterations()
            ));
        }

        // The printed densities of the final design variables are already in
        // the workspace, because every update step pushes its own trial point
        // through the chain.
        let overhang_residual = chain.overhang().map(|_| {
            let gaps = design_cells
                .iter()
                .map(|&e| (work.printed[e] - work.filtered[e]).abs());
            let (max, total) = gaps.fold((0.0f64, 0.0f64), |(max, total), gap| {
                (max.max(gap), total + gap)
            });
            OverhangResidual {
                max,
                mean: total / design_cells.len().max(1) as f64,
            }
        });
        Ok(DensityField {
            densities: work.printed,
            iterations,
            compliance,
            initial_compliance,
            volume_fraction,
            max_change,
            stop,
            overhang_residual,
            growth: None,
            reduce: reduce_summary,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, UpdateScheme};
    use crate::report::SilentReporter;
    use std::path::PathBuf;

    fn tiny(extra: &str) -> String {
        format!(
            r#"
[project]
name = "tiny"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

# The recorded trajectories and the run-to-run comparisons below are bit-for-bit
# claims, so every fixture here names the reproducible backend rather than
# taking the default one, which is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0
max_iterations = 5
{extra}

[output]
stl_path = "tiny.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [24.0, 8.0, 8.0]

[[supports]]
region = {{ shape = "box", min = [0.0, 0.0, 0.0], max = [0.5, 8.0, 8.0] }}

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = {{ shape = "box", min = [23.5, 0.0, 0.0], max = [24.5, 8.0, 8.0] }}
vector = [0.0, 0.0, -20.0]
"#
        )
    }

    fn run(text: &str) -> crate::engine::DensityField {
        let config = Config::parse(text).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        SimpEngine
            .optimize(&problem, &SilentReporter)
            .expect("optimize")
    }

    #[test]
    fn simp_reduces_compliance_and_meets_the_volume_target() {
        let field = run(&tiny(""));
        assert_eq!(field.iterations, 5);
        assert!(
            (field.volume_fraction - 0.4).abs() < 1e-3,
            "volume fraction {} missed the target",
            field.volume_fraction
        );
        assert!(
            field.compliance < field.initial_compliance,
            "compliance {} did not improve on {}",
            field.compliance,
            field.initial_compliance
        );
        assert!(field.densities.iter().all(|d| (0.0..=1.0).contains(d)));
        assert!(field.overhang_residual.is_none());
    }

    /// The compliance trajectory this problem produced before phase 3, as the
    /// 0.2.0 binary printed it. It pins the plain path against every refactor
    /// the overhang, self-weight and stress work brought with it.
    const PLAIN_TRAJECTORY: [f64; 5] = [
        3.654_772e1,
        2.924_280e1,
        2.590_055e1,
        2.391_551e1,
        2.193_148e1,
    ];

    #[test]
    fn the_plain_path_reproduces_its_pre_phase_3_trajectory() {
        use crate::report::IterationStats;
        use std::sync::Mutex;

        #[derive(Default)]
        struct Trace(Mutex<Vec<f64>>);
        impl Reporter for Trace {
            fn iteration(&self, stats: &IterationStats) {
                self.0.lock().expect("trace").push(stats.compliance);
            }
            fn note(&self, _message: &str) {}
        }

        let config = Config::parse(&tiny("")).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let trace = Trace::default();
        SimpEngine.optimize(&problem, &trace).expect("optimize");
        let seen = trace.0.into_inner().expect("trace");
        assert_eq!(seen.len(), PLAIN_TRAJECTORY.len());
        for (iteration, (got, expected)) in seen.iter().zip(PLAIN_TRAJECTORY.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6 * expected,
                "iteration {} compliance {got} left the recorded {expected}",
                iteration + 1
            );
        }
    }

    #[test]
    fn the_overhang_filter_leaves_a_printable_design() {
        let field = run(&tiny("[optimization.overhang]\nbuild_direction = \"z+\""));
        assert!(
            (field.volume_fraction - 0.4).abs() < 1e-3,
            "volume fraction {} missed the target",
            field.volume_fraction
        );
        let residual = field.overhang_residual.expect("an overhang residual");
        assert!(residual.max.is_finite() && residual.mean.is_finite());
        assert!(residual.mean <= residual.max);
        // Every printed density has to be reachable from the layer below.
        assert!(field.densities.iter().all(|d| *d >= -1e-12));
    }

    /// A capped run of the same problem on both backends. The optimality
    /// criteria step, the filters and the volume constraint are CPU code that
    /// neither backend touches, so the volume target has to be met identically;
    /// only the compliance, which the displacements feed, may move at all.
    #[test]
    fn a_gpu_run_reaches_the_same_compliance_as_the_cpu_run() {
        let what = "a_gpu_run_reaches_the_same_compliance_as_the_cpu_run";
        let config = Config::parse(&tiny("")).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        if crate::fea::backend::gpu_or_skip(what, &problem.grid, &problem.fixed).is_none() {
            return;
        }

        let reference = SimpEngine
            .optimize(&problem, &SilentReporter)
            .expect("cpu run");

        let mut accelerated = problem.clone();
        accelerated.solver.backend = crate::config::SolverBackend::Gpu;
        let candidate = SimpEngine
            .optimize(&accelerated, &SilentReporter)
            .expect("gpu run");

        assert_eq!(candidate.iterations, reference.iterations);
        let error =
            (candidate.compliance - reference.compliance).abs() / reference.compliance.abs();
        println!(
            "{what}: cpu {:.9e}, gpu {:.9e}, relative difference {error:.3e}",
            reference.compliance, candidate.compliance
        );
        assert!(
            error < 1e-3,
            "final compliance {} differs from the CPU reference {} by {error:.3e}",
            candidate.compliance,
            reference.compliance
        );
        assert!(
            (candidate.volume_fraction - 0.4).abs() < 1e-3,
            "volume fraction {} missed the target",
            candidate.volume_fraction
        );
    }

    /// A design variable update scheme selected from the configuration, so the
    /// tests below exercise the same seam a user does.
    fn with_update(scheme: UpdateScheme) -> String {
        match scheme {
            UpdateScheme::Oc => tiny(""),
            UpdateScheme::Mma => tiny("update = \"mma\""),
        }
    }

    /// The tiny cantilever run long enough for either scheme to have settled.
    fn settled(scheme: UpdateScheme) -> crate::engine::DensityField {
        run(&with_update(scheme).replace("max_iterations = 5", "max_iterations = 30"))
    }

    /// MMA has to be a usable alternative on a problem the optimality criteria
    /// step already handles: the same volume constraint to the same tolerance,
    /// and a compliance in the same place.
    ///
    /// The bound on the compliance is deliberately loose (5 %). The two schemes
    /// minimize different approximations of the same objective and land on
    /// different local designs; what is being gated here is that neither is
    /// *degraded*, not that they agree.
    #[test]
    fn mma_matches_the_optimality_criteria_step_on_the_plain_cantilever() {
        let reference = settled(UpdateScheme::Oc);
        let candidate = settled(UpdateScheme::Mma);
        assert!(
            (candidate.volume_fraction - 0.4).abs() < 1e-3,
            "volume fraction {} missed the target",
            candidate.volume_fraction
        );
        let error =
            (candidate.compliance - reference.compliance).abs() / reference.compliance.abs();
        println!(
            "plain cantilever: oc {:.6e}, mma {:.6e}, relative difference {error:.3e}",
            reference.compliance, candidate.compliance
        );
        assert!(
            error < 0.05,
            "mma compliance {} is more than 5 % from the optimality criteria result {}",
            candidate.compliance,
            reference.compliance
        );
        assert!(candidate.compliance < candidate.initial_compliance);
        assert!(candidate.densities.iter().all(|d| (0.0..=1.0).contains(d)));
    }

    /// `update = "mma"` has to reach the loop, not just the configuration. Two
    /// observables say it did: the design trajectory is not the one the
    /// optimality criteria step produces, and the self-weight sensitivity shift
    /// that only that step needs is never announced under MMA, even on a problem
    /// that makes it engage.
    #[test]
    fn the_configured_update_scheme_reaches_the_loop() {
        use crate::report::IterationStats;
        use std::sync::Mutex;

        #[derive(Default)]
        struct Trace {
            compliances: Mutex<Vec<f64>>,
            notes: Mutex<Vec<String>>,
        }
        impl Reporter for Trace {
            fn iteration(&self, stats: &IterationStats) {
                self.compliances
                    .lock()
                    .expect("trace")
                    .push(stats.compliance);
            }
            fn note(&self, message: &str) {
                self.notes.lock().expect("trace").push(message.to_string());
            }
        }

        // A dominant self weight, so the optimality criteria step has to shift
        // its sensitivities and says so.
        let heavy = |extra: &str| {
            tiny(extra).replace(
                "vector = [0.0, 0.0, -20.0]",
                "vector = [0.0, 0.0, -20.0]\n[[loadcases.loads]]\ntype = \"gravity\"\ng_mm_s2 = 9.81e6",
            )
        };
        let trace_of = |text: String| {
            let config = Config::parse(&text).expect("parse");
            let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
            let trace = Trace::default();
            SimpEngine.optimize(&problem, &trace).expect("optimize");
            (
                trace.compliances.into_inner().expect("trace"),
                trace.notes.into_inner().expect("trace"),
            )
        };

        let (oc, oc_notes) = trace_of(heavy(""));
        let (mma, mma_notes) = trace_of(heavy("update = \"mma\""));
        assert_eq!(oc.len(), mma.len());
        // The first iteration is the same design analysed the same way, so the
        // two agree there to the spread of a parallel floating point reduction.
        // Every later one is the update scheme's own work, and they part.
        assert!(
            (oc[0] - mma[0]).abs() < 1e-9 * oc[0].abs(),
            "the two schemes started from different designs: {} against {}",
            oc[0],
            mma[0]
        );
        assert!(
            oc.iter()
                .zip(mma.iter())
                .skip(1)
                .any(|(a, b)| (a - b).abs() > 1e-3 * a.abs()),
            "the two schemes produced the same trajectory: {oc:?} vs {mma:?}"
        );
        assert!(
            oc_notes.iter().any(|n| n.contains("sensitivity shift")),
            "this problem is supposed to engage the shift: {oc_notes:?}"
        );
        assert!(
            !mma_notes.iter().any(|n| n.contains("sensitivity shift")),
            "MMA represents a positive sensitivity itself and must not shift: {mma_notes:?}"
        );
    }

    /// Self weight is a design dependent load, and MMA takes it without the
    /// relabelling the optimality criteria step needs. Both the gravity and the
    /// plain case have to meet the volume target.
    #[test]
    fn mma_runs_with_and_without_gravity() {
        let plain = run(&tiny("update = \"mma\""));
        let heavy = run(&tiny("update = \"mma\"").replace(
            "vector = [0.0, 0.0, -20.0]",
            "vector = [0.0, 0.0, -20.0]\n[[loadcases.loads]]\ntype = \"gravity\"",
        ));
        for field in [&plain, &heavy] {
            assert!(
                (field.volume_fraction - 0.4).abs() < 1e-3,
                "volume fraction {} missed the target",
                field.volume_fraction
            );
            assert!(field.compliance.is_finite() && field.compliance > 0.0);
            assert!(field.densities.iter().all(|d| (0.0..=1.0).contains(d)));
        }
        assert!(
            heavy.compliance > plain.compliance,
            "self weight has to add compliance: {} against {}",
            heavy.compliance,
            plain.compliance
        );
    }

    /// A miniature of the shipped `shelf_bracket.toml`: a wall flange, a loaded
    /// shelf pad, self weight and the self-supporting filter, at 8 x 2 x 12
    /// cells. It is the configuration that makes a design variable a
    /// near-discrete "does this column print or not" decision, which is the
    /// whole reason MMA is here.
    ///
    /// The convergence tolerance is set below anything either scheme reaches, so
    /// both runs spend the same iterations and their final design variable
    /// change is comparable.
    fn mini_bracket(extra: &str, iterations: usize) -> String {
        format!(
            r#"
[project]
name = "mini_bracket"

[resolution]
voxel_size_mm = 3.0

[material]
preset = "petg"

# The reproducible backend, for the same reason as the fixture above.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.16
min_feature_mm = 9.0
max_iterations = {iterations}
convergence_tol = 1e-9
{extra}

  [optimization.overhang]
  build_direction = "z+"

[output]
stl_path = "mini_bracket.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [24.0, 6.0, 36.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 0.0]
max = [3.0, 6.0, 36.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 33.0]
max = [24.0, 6.0, 36.0]

[[supports]]
region = {{ shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 6.5, 36.5] }}
directions = ["x", "y", "z"]

[[loadcases]]
name = "shelf-load"
[[loadcases.loads]]
type = "force"
region = {{ shape = "box", min = [-0.5, -0.5, 34.5], max = [24.5, 6.5, 36.5] }}
vector = [0.0, 0.0, -400.0]
[[loadcases.loads]]
type = "gravity"
"#
        )
    }

    /// The compliance trajectory of the miniature bracket under the optimality
    /// criteria step, recorded from this build.
    ///
    /// Its counterpart [`PLAIN_TRAJECTORY`] pins the path with no filter but the
    /// density one; this pins the path *through the self-supporting filter's
    /// adjoint*, which is what decides every overhang number the project
    /// reports and which a refactor can break silently - the forward sweep keeps
    /// producing a plausible printable design while the sensitivities it hands
    /// back are wrong. `oc` is the scheme recorded here because it is the
    /// default and the reference; the trajectory is a property of the chain, not
    /// of the update.
    ///
    /// Sensitivity is not theoretical: an off-by-one in the exponent of the
    /// smooth maximum's adjoint moves the second iteration by 0.6 %, four
    /// decades above the tolerance below.
    const OVERHANG_TRAJECTORY: [f64; 8] = [
        8.994_248_727e2,
        5.687_695_745e2,
        3.483_205_053e2,
        2.808_433_724e2,
        2.533_640_850e2,
        2.298_964_598e2,
        2.091_603_919e2,
        1.923_859_412e2,
    ];

    #[test]
    fn the_overhang_path_reproduces_its_recorded_trajectory() {
        use crate::report::IterationStats;
        use std::sync::Mutex;

        #[derive(Default)]
        struct Trace(Mutex<Vec<f64>>);
        impl Reporter for Trace {
            fn iteration(&self, stats: &IterationStats) {
                self.0.lock().expect("trace").push(stats.compliance);
            }
            fn note(&self, _message: &str) {}
        }

        let config = Config::parse(&mini_bracket("", OVERHANG_TRAJECTORY.len())).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        assert!(problem.optimization.overhang.is_some());
        assert_eq!(problem.optimization.update, UpdateScheme::Oc);
        let trace = Trace::default();
        SimpEngine.optimize(&problem, &trace).expect("optimize");
        let seen = trace.0.into_inner().expect("trace");
        assert_eq!(seen.len(), OVERHANG_TRAJECTORY.len());
        for (iteration, (got, expected)) in seen.iter().zip(OVERHANG_TRAJECTORY.iter()).enumerate()
        {
            assert!(
                (got - expected).abs() < 1e-6 * expected,
                "iteration {} compliance {got} left the recorded {expected}",
                iteration + 1
            );
        }
    }

    /// The reason MMA exists here: with the self-supporting filter in the chain
    /// the optimality criteria step cannot settle the near-discrete "does this
    /// column print" decisions and leaves a larger design variable change behind
    /// at the same iteration count.
    ///
    /// The claim is encoded conservatively - MMA's final change may not be
    /// *larger* - because how much better it is depends on the problem. Measured
    /// on this fixture at 40 iterations the two are 0.0002 (mma) against 0.0033
    /// (oc); on the shipped `shelf_bracket.toml` at its own 120 they are 0.017
    /// against 0.025, the second of which is above the 0.01 the example asks
    /// for.
    #[test]
    fn mma_leaves_a_smaller_change_than_the_optimality_criteria_step_under_the_overhang_filter() {
        let reference = run(&mini_bracket("", 40));
        let candidate = run(&mini_bracket("update = \"mma\"", 40));
        println!(
            "overhang fixture: oc max change {:.5} at compliance {:.6e}, mma max change {:.5} at \
             compliance {:.6e}",
            reference.max_change, reference.compliance, candidate.max_change, candidate.compliance
        );
        assert_eq!(reference.iterations, 40);
        assert_eq!(candidate.iterations, 40);
        assert!(
            candidate.max_change <= reference.max_change,
            "mma left {} against the optimality criteria step's {}",
            candidate.max_change,
            reference.max_change
        );
        // Both designs are still on target and printable; a smaller change that
        // came from missing the volume constraint would prove nothing.
        for field in [&reference, &candidate] {
            assert!(
                (field.volume_fraction - 0.16).abs() < 1e-3,
                "volume fraction {} missed the target",
                field.volume_fraction
            );
            let residual = field.overhang_residual.expect("an overhang residual");
            assert!(residual.max.is_finite() && residual.mean <= residual.max);
        }
    }

    /// The cooperative cancellation seam the editor's auto-regrow drives: a
    /// reporter that asks for a stop gets one between iterations, with a usable
    /// density field and an honest iteration count.
    #[test]
    fn a_cancelling_reporter_stops_the_loop_between_iterations() {
        use crate::report::IterationStats;
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Default)]
        struct StopAfter {
            after: usize,
            seen: AtomicUsize,
            notes: Mutex<Vec<String>>,
        }
        impl Reporter for StopAfter {
            fn iteration(&self, _stats: &IterationStats) {
                self.seen.fetch_add(1, Ordering::Relaxed);
            }
            fn note(&self, message: &str) {
                self.notes.lock().expect("notes").push(message.to_string());
            }
            fn cancelled(&self) -> bool {
                self.seen.load(Ordering::Relaxed) >= self.after
            }
        }

        let text = tiny("").replace("max_iterations = 5", "max_iterations = 30");
        let config = Config::parse(&text).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let reporter = StopAfter {
            after: 2,
            ..StopAfter::default()
        };
        let field = SimpEngine.optimize(&problem, &reporter).expect("optimize");

        assert_eq!(
            field.iterations, 2,
            "the cancel must land between iterations"
        );
        assert_eq!(
            field.stop,
            StopReason::Cancelled,
            "a cancelled run has neither converged nor spent its budget"
        );
        assert_eq!(field.densities.len(), problem.grid.n_cells());
        assert!(field.densities.iter().all(|d| (0.0..=1.0).contains(d)));
        let notes = reporter.notes.into_inner().expect("notes");
        assert!(
            notes.iter().any(|n| n.contains("cancelled")),
            "a cancelled run must not report an iteration cap: {notes:?}"
        );
        // And the default reporter never cancels, so every other run is
        // untouched: the same problem runs its full budget.
        let full = SimpEngine
            .optimize(&problem, &SilentReporter)
            .expect("optimize");
        assert!(full.iterations > 2);
    }

    /// A miniature of the *deep* bracket - the shipped one reaching two and a
    /// half times as far out from the wall - which is the documented problem the
    /// optimality criteria step cannot do at all: 15 x 2 x 7 cells, and every
    /// iteration of it takes a step of exactly the 0.2 move limit while the
    /// compliance goes nowhere.
    ///
    /// This is `mini_bracket`'s arrangement with the shelf reaching 60 mm out
    /// from the wall instead of 24, and with the *default* convergence tolerance
    /// rather than the unreachable one that fixture uses: nothing here is
    /// contrived to defeat the tolerance, the problem simply never settles. The
    /// same configuration is `lib`'s `STUBBORN`, which takes it through the
    /// export as well.
    fn deep_bracket(iterations: usize) -> String {
        format!(
            r#"
[project]
name = "deep_bracket"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "petg"

# The reproducible backend, for the same reason as the fixtures above.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.16
min_feature_mm = 12.0
max_iterations = {iterations}
convergence_tol = 0.01

  [optimization.overhang]
  build_direction = "z+"

[output]
stl_path = "deep_bracket.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [60.0, 8.0, 28.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 0.0]
max = [4.0, 8.0, 28.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 24.0]
max = [60.0, 8.0, 28.0]

[[supports]]
region = {{ shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 8.5, 28.5] }}
directions = ["x", "y", "z"]

[[loadcases]]
name = "shelf-load"
[[loadcases.loads]]
type = "force"
region = {{ shape = "box", min = [-0.5, -0.5, 24.5], max = [60.5, 8.5, 28.5] }}
vector = [0.0, 0.0, -400.0]
[[loadcases.loads]]
type = "gravity"
"#
        )
    }

    /// The case the stall criterion exists for, in the loop: a problem that will
    /// not settle stops itself, says so with the iteration it stopped on, and
    /// names the scheme that could not do it.
    ///
    /// The budget is 250 rather than the default thousand purely so that a test
    /// which is *supposed* to stop early cannot cost minutes if it ever stops
    /// doing so; the stall lands at 123, well clear of it, so the assertion that
    /// the cap was not what stopped the run means something.
    #[test]
    fn a_problem_that_will_not_settle_stops_itself_and_says_which_scheme_did_not() {
        use crate::report::IterationStats;
        use std::sync::Mutex;

        #[derive(Default)]
        struct Notes(Mutex<Vec<String>>);
        impl Reporter for Notes {
            fn iteration(&self, _stats: &IterationStats) {}
            fn note(&self, message: &str) {
                self.0.lock().expect("notes").push(message.to_string());
            }
        }

        let budget = 250;
        let config = Config::parse(&deep_bracket(budget)).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        assert_eq!(problem.optimization.update, UpdateScheme::Oc);
        let notes = Notes::default();
        let field = SimpEngine.optimize(&problem, &notes).expect("optimize");
        let notes = notes.0.into_inner().expect("notes");

        assert_eq!(field.stop, StopReason::Stalled);
        assert!(
            field.iterations >= constants::STALL_WINDOW_ITERATIONS && field.iterations < budget,
            "a stall may not be called before a full window or after the cap: {} iterations",
            field.iterations
        );
        // It stopped because it was traversing, not because it had arrived: the
        // design was still crossing the box at the move limit.
        assert!(
            field.max_change >= constants::STALL_MOVE_LIMIT_FRACTION * constants::OC_MOVE_LIMIT,
            "the stall has to be a traversing design: max change {}",
            field.max_change
        );
        // What it says, and what it must not say: the cap did not stop this run.
        let stall = notes
            .iter()
            .find(|note| note.contains("still traversing"))
            .unwrap_or_else(|| panic!("no stall note in {notes:?}"));
        assert!(stall.contains(&format!("stopped after {} iterations", field.iterations)));
        assert!(stall.contains("update = \"mma\""), "{stall}");
        assert!(
            !notes.iter().any(|note| note.contains("iteration cap")),
            "{notes:?}"
        );
        // And it is a finished design, not a broken one: on target, printable,
        // and with the overhang residual every run of this fixture reports.
        assert!(
            (field.volume_fraction - 0.16).abs() < 1e-3,
            "volume fraction {} missed the target",
            field.volume_fraction
        );
        assert!(field.densities.iter().all(|d| (0.0..=1.0).contains(d)));
        assert!(field.overhang_residual.is_some());
    }

    /// The other half of the claim, and the one that costs a design if it is
    /// wrong: a run that is still settling is never cut off, however long it
    /// takes. The plain cantilever asked for `convergence_tol = 0.001` works its
    /// change down the whole way and converges at iteration 134 - a third of a
    /// window past the point where the stall test is armed and watching it.
    #[test]
    fn a_run_that_is_still_settling_is_left_alone_past_the_stall_window() {
        let field =
            run(&tiny("convergence_tol = 0.001")
                .replace("max_iterations = 5", "max_iterations = 250"));
        assert_eq!(field.stop, StopReason::Converged);
        assert!(
            field.iterations > constants::STALL_WINDOW_ITERATIONS,
            "this fixture is only evidence while it converges after a full \
             window: {} iterations",
            field.iterations
        );
        assert!(field.max_change < 0.001);
    }

    /// The guide wireframe through the loop it is a stage of: the seed is in the
    /// design the first iteration analyses, the floor holds the volume above the
    /// target while it is on, and the iteration after the release is back on
    /// target - which is the whole claim that the wire is a guide and not a
    /// constraint.
    #[test]
    fn the_guide_wireframe_seeds_the_run_holds_its_floor_and_lets_go() {
        use crate::report::IterationStats;
        use std::sync::Mutex;

        #[derive(Default)]
        struct Trace {
            volumes: Mutex<Vec<f64>>,
            notes: Mutex<Vec<String>>,
        }
        impl Reporter for Trace {
            fn iteration(&self, stats: &IterationStats) {
                self.volumes
                    .lock()
                    .expect("trace")
                    .push(stats.volume_fraction);
            }
            fn note(&self, message: &str) {
                self.notes.lock().expect("trace").push(message.to_string());
            }
        }

        let hold = 2;
        let text = tiny(&format!(
            "[optimization.wireframe]\nhold_iterations = {hold}"
        ))
        .replace("max_iterations = 5", "max_iterations = 6");
        let config = Config::parse(&text).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let params = problem.optimization.wireframe.expect("a wireframe");
        assert_eq!(params.hold_iterations, hold);
        // The guide it seeds, so the assertions below are against a wire that is
        // really there.
        let wire =
            crate::engine::wireframe::Wireframe::build(&problem, &SilentReporter).expect("a guide");
        assert!(!wire.cells().is_empty());

        let trace = Trace::default();
        let field = SimpEngine.optimize(&problem, &trace).expect("optimize");
        let volumes = trace.volumes.into_inner().expect("trace");
        let notes = trace.notes.into_inner().expect("trace");
        assert_eq!(volumes.len(), 6);

        let target = problem.optimization.mass_fraction;
        for (iteration, volume) in volumes.iter().enumerate().take(hold) {
            assert!(
                *volume > target + 1e-3,
                "iteration {} is inside the hold and has to run heavy: {volume} against {target}",
                iteration + 1
            );
        }
        for (iteration, volume) in volumes.iter().enumerate().skip(hold) {
            assert!(
                (volume - target).abs() < 1e-3,
                "iteration {} is past the release and has to be on target: {volume}",
                iteration + 1
            );
        }
        assert!(
            (field.volume_fraction - target).abs() < 1e-3,
            "the exported design has to meet the volume target: {}",
            field.volume_fraction
        );
        assert!(notes.iter().any(|n| n.contains("wireframe:")), "{notes:?}");
        assert!(
            notes
                .iter()
                .any(|n| n.contains("wireframe:") && n.contains("released")),
            "the release has to be announced: {notes:?}"
        );

        // And the same problem without the table runs its own way: the first
        // iteration of the guided run analysed a different design.
        let plain = run(&tiny("").replace("max_iterations = 5", "max_iterations = 6"));
        assert!(
            (plain.volume_fraction - target).abs() < 1e-3,
            "the unguided run is the reference and has to be on target"
        );
        assert!(
            plain
                .densities
                .iter()
                .zip(field.densities.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "the guide changed nothing at all, so it never reached the loop"
        );
    }

    /// A reporter that keeps the notes and the per-iteration numbers, which is
    /// what the guide's claims are made of.
    #[derive(Default)]
    struct Guided {
        volumes: std::sync::Mutex<Vec<f64>>,
        notes: std::sync::Mutex<Vec<String>>,
    }

    impl Reporter for Guided {
        fn iteration(&self, stats: &IterationStats) {
            self.volumes
                .lock()
                .expect("trace")
                .push(stats.volume_fraction);
        }
        fn note(&self, message: &str) {
            self.notes.lock().expect("trace").push(message.to_string());
        }
    }

    impl Guided {
        fn taken(self) -> (Vec<f64>, Vec<String>) {
            (
                self.volumes.into_inner().expect("trace"),
                self.notes.into_inner().expect("trace"),
            )
        }
    }

    /// **A run cannot settle while the guide's floor is on**, which is what keeps
    /// a forced wire out of a part nobody was warned about.
    ///
    /// The shape here is the one that exported a forced wire silently: a
    /// convergence tolerance above the update scheme's move limit, so *every*
    /// iteration's change is below it and the very first one looks settled. Under
    /// the floor that reading is wrong - the change an iteration reports is partly
    /// the floor pushing its cells back up - so the verdict waits for the release
    /// and the run converges on the first free iteration instead, at the volume
    /// target rather than above it.
    ///
    /// The wireframe is left on its defaults, tolerance apart, because the default
    /// hold is what a user gets.
    #[test]
    fn a_guided_run_cannot_settle_while_its_floor_is_still_on() {
        let tolerance = constants::OC_MOVE_LIMIT + 0.1;
        let hold = constants::WIREFRAME_HOLD_ITERATIONS;
        let budget = hold + 20;
        let guided = tiny(&format!(
            "convergence_tol = {tolerance}\n\n[optimization.wireframe]"
        ))
        .replace("max_iterations = 5", &format!("max_iterations = {budget}"));
        let config = Config::parse(&guided).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        assert_eq!(
            problem
                .optimization
                .wireframe
                .expect("a wireframe")
                .hold_iterations,
            hold,
            "this test is about the default hold"
        );
        let trace = Guided::default();
        let field = SimpEngine.optimize(&problem, &trace).expect("optimize");
        let (volumes, notes) = trace.taken();

        assert_eq!(field.stop, StopReason::Converged);
        assert_eq!(
            field.iterations,
            hold + 1,
            "the settle verdict has to land on the first iteration after the release"
        );
        assert_eq!(volumes.len(), hold + 1);
        let target = problem.optimization.mass_fraction;
        assert!(
            (field.volume_fraction - target).abs() < 1e-3,
            "the design that ships has to be the optimizer's, on target: {}",
            field.volume_fraction
        );
        // Every held iteration ran heavy, and the one it converged on did not.
        assert!(
            volumes[..hold].iter().all(|v| *v > target + 1e-3),
            "the floor was not on: {volumes:?}"
        );
        assert!((volumes[hold] - target).abs() < 1e-3, "{volumes:?}");
        assert!(
            notes.iter().any(|n| n.contains("released")),
            "the release has to be announced: {notes:?}"
        );
        assert!(
            !notes.iter().any(|n| n.contains("forced material")),
            "a run that reached the release carries no forced wire: {notes:?}"
        );

        // The counter-case, and the evidence that the tolerance really is one
        // every iteration meets: the same problem with no guide settles on the
        // first iteration.
        let plain = run(&tiny(&format!("convergence_tol = {tolerance}"))
            .replace("max_iterations = 5", &format!("max_iterations = {budget}")));
        assert_eq!(plain.stop, StopReason::Converged);
        assert_eq!(plain.iterations, 1);
    }

    /// A hold longer than the whole stall window, which is where the two verdicts
    /// have to be seen waiting rather than merely deferred by an iteration or two.
    ///
    /// The stall criterion is not fed at all while the floor is on, so its window
    /// starts filling at the release and the earliest a stall can be called is
    /// `hold_iterations` plus a full window. With a budget below that, no stall is
    /// reachable however the held iterations looked - which is what this asserts,
    /// together with the positive control that they *did* look settled: the change
    /// this fixture reports under the floor dips below its own `convergence_tol`,
    /// so an ungated loop would have called this run converged in the middle of the
    /// hold and exported the wire as forced material. It runs to its cap instead.
    #[test]
    fn a_hold_longer_than_the_stall_window_defers_both_verdicts() {
        use crate::engine::stall::{Sample, is_stalled};

        let hold = constants::STALL_WINDOW_ITERATIONS + 5;
        let budget = hold + 5;
        // The cantilever at a coarser voxel size than the fixtures above use: a
        // hundred and ten iterations is a hundred and ten linear solves, and what
        // is under test is the loop's bookkeeping rather than any one design.
        let text = tiny(&format!(
            "[optimization.wireframe]\nhold_iterations = {hold}"
        ))
        .replace("max_iterations = 5", &format!("max_iterations = {budget}"))
        .replace("voxel_size_mm = 2.0", "voxel_size_mm = 4.0");
        let config = Config::parse(&text).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let tolerance = problem.optimization.convergence_tol;

        #[derive(Default)]
        struct Samples {
            samples: std::sync::Mutex<Vec<Sample>>,
            notes: std::sync::Mutex<Vec<String>>,
        }
        impl Reporter for Samples {
            fn iteration(&self, stats: &IterationStats) {
                self.samples.lock().expect("trace").push(Sample {
                    compliance: stats.compliance,
                    max_change: stats.max_change,
                });
            }
            fn note(&self, message: &str) {
                self.notes.lock().expect("trace").push(message.to_string());
            }
        }

        let trace = Samples::default();
        let field = SimpEngine.optimize(&problem, &trace).expect("optimize");
        let samples = trace.samples.into_inner().expect("trace");
        let notes = trace.notes.into_inner().expect("trace");

        assert_eq!(
            field.stop,
            StopReason::IterationCap,
            "no verdict was reachable inside this budget, so the cap had to end it"
        );
        assert_eq!(field.iterations, budget);
        assert_eq!(samples.len(), budget);
        assert!(
            !notes.iter().any(|n| n.contains("still traversing")),
            "a stall was called on a window that cannot have been full: {notes:?}"
        );
        assert!(notes.iter().any(|n| n.contains("released")), "{notes:?}");

        // The positive control: under the floor this run looked settled, over and
        // over, by the very test that would have ended it.
        let held = &samples[..hold];
        assert!(
            held.iter().filter(|s| s.max_change < tolerance).count() > 1,
            "the fixture has to look settled under the floor for this to prove \
             anything: smallest held change {:?}",
            held.iter()
                .map(|s| s.max_change)
                .fold(f64::INFINITY, f64::min)
        );
        // And it was not a stall by the criterion's own reading, so nothing here
        // rests on a suppressed stall verdict: what is asserted about the stall
        // criterion is that its window cannot fill before the release.
        assert!(!is_stalled(
            &held[..constants::STALL_WINDOW_ITERATIONS],
            problem.optimization.update.move_limit()
        ));
        assert!(budget < hold + constants::STALL_WINDOW_ITERATIONS);
    }

    /// The one way left to export a forced wire is to run out of budget inside the
    /// hold window, and it is reported twice: once by the guide when it is built,
    /// where the misconfiguration is visible before any time is spent, and once at
    /// the end, about the design that actually came out.
    #[test]
    fn a_run_capped_inside_the_hold_window_says_the_wire_is_forced() {
        let budget = 3;
        let text = tiny("[optimization.wireframe]\nhold_iterations = 10")
            .replace("max_iterations = 5", &format!("max_iterations = {budget}"));
        let config = Config::parse(&text).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let trace = Guided::default();
        let field = SimpEngine.optimize(&problem, &trace).expect("optimize");
        let (volumes, notes) = trace.taken();

        assert_eq!(field.stop, StopReason::IterationCap);
        assert_eq!(field.iterations, budget);
        // The floor was on for every one of them, so the design that came out is
        // above the volume target.
        assert!(
            volumes
                .iter()
                .all(|v| *v > problem.optimization.mass_fraction + 1e-3),
            "{volumes:?}"
        );
        assert!(field.volume_fraction > problem.optimization.mass_fraction + 1e-3);
        let forced: Vec<&String> = notes
            .iter()
            .filter(|n| n.contains("warning") && n.contains("forced material"))
            .collect();
        assert_eq!(
            forced.len(),
            1,
            "the design that came out has to be reported once: {notes:?}"
        );
        assert!(
            forced[0].contains(&format!("after {budget} iterations")),
            "{forced:?}"
        );
        assert!(
            notes
                .iter()
                .any(|n| n.contains("warning") && n.contains("never released")),
            "the misconfiguration has to be reported at setup as well: {notes:?}"
        );
        assert!(
            !notes.iter().any(|n| n.contains("released after")),
            "nothing was released: {notes:?}"
        );
    }

    /// A block wide enough to hold either one thick load path or several thin
    /// ones: supported at both bottom corners, loaded down at the middle of its
    /// top face, and coarse enough that a local neighbourhood is a few voxels
    /// rather than the whole part.
    ///
    /// Two voxels to a feature rather than the three the density filter wants,
    /// which is what keeps two converging runs of it inside a unit test - so it
    /// carries the resolution warning, and the test asserts that this is the
    /// only one it carries. What is under test is the constraint, not the
    /// surface the design would be exported into; `tests/integration.rs` runs
    /// the same arrangement at a resolution that warns about nothing, end to
    /// end.
    fn porous_block(extra: &str, iterations: usize) -> String {
        format!(
            r#"
[project]
name = "porous_block"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

# The reproducible backend, for the same reason as the fixtures above.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.3
min_feature_mm = 4.0
max_iterations = {iterations}
convergence_tol = 1e-4
update = "mma"
{extra}

[output]
stl_path = "porous_block.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [40.0, 6.0, 16.0]

[[supports]]
region = {{ shape = "box", min = [-0.5, -0.5, -0.5], max = [4.5, 6.5, 0.5] }}

[[supports]]
region = {{ shape = "box", min = [35.5, -0.5, -0.5], max = [40.5, 6.5, 0.5] }}

[[loadcases]]
name = "middle"
[[loadcases.loads]]
type = "force"
region = {{ shape = "box", min = [16.0, -0.5, 15.5], max = [24.0, 6.5, 16.5] }}
vector = [0.0, 0.0, -200.0]
"#
        )
    }

    /// What a finished design really holds locally, measured the way the run
    /// measures it: the aggregate the constraint is stated on and the worst
    /// neighbourhood behind it.
    fn local_measurement(
        problem: &Problem,
        densities: &[f64],
        params: crate::config::LocalVolumeParams,
    ) -> crate::engine::local_volume::Measurement {
        let design_cells: Vec<usize> = (0..problem.grid.n_cells())
            .filter(|&e| problem.grid.cells[e] == CellKind::Design)
            .collect();
        crate::engine::local_volume::LocalVolume::new(&problem.grid, params)
            .measure(densities, &design_cells)
    }

    /// **The reason the local volume constraint exists.** The same problem, once
    /// under the global volume target alone and once with the local cap on top
    /// of it: the first spends its material on a few members packed solid, and
    /// the second cannot.
    ///
    /// Every claim here is a measured number. On this fixture, both runs
    /// converging:
    ///
    /// ```text
    ///                  worst local   aggregate   volume   compliance
    ///   no cap            0.9128       0.7603     0.4000   1.1874e1
    ///   max_fraction 0.6  0.7161       0.6000     0.3239   1.4547e1
    /// ```
    ///
    /// The aggregate the constraint is *stated* on lands exactly on the cap. The
    /// worst neighbourhood behind it lands 0.116 above, which is the p-mean's
    /// documented under-estimate and not a failure of the solve - see
    /// [`crate::engine::local_volume`]. And the volume comes out under its
    /// target, because with the cap binding the global constraint is the slack
    /// one: material the design is allowed is material it has nowhere to put.
    #[test]
    fn a_local_cap_breaks_up_the_solid_members_a_global_target_leaves() {
        let mass = 0.4;
        let budget = 120;
        let text = porous_block("", budget)
            .replace("mass_fraction = 0.3", &format!("mass_fraction = {mass}"));
        let config = Config::parse(&text).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        // The fixture's deliberate coarseness, and nothing else.
        assert_eq!(problem.warnings.len(), 1, "{:?}", problem.warnings);
        assert!(
            problem.warnings[0].contains("min_feature_mm"),
            "{:?}",
            problem.warnings
        );
        let free = SimpEngine
            .optimize(&problem, &SilentReporter)
            .expect("optimize");

        // The cap on its defaults, which is what a user gets from the bare table.
        let capped_text = text.replace(
            "update = \"mma\"",
            "update = \"mma\"\n\n  [optimization.local_volume]",
        );
        let capped_config = Config::parse(&capped_text).expect("parse");
        let capped_problem = Problem::build(&capped_config, &PathBuf::from(".")).expect("build");
        let params = capped_problem.optimization.local_volume.expect("a cap");
        assert_eq!(params.max_fraction, constants::LOCAL_VOLUME_MAX_FRACTION);
        assert_eq!(
            params.radius_mm,
            constants::LOCAL_VOLUME_RADIUS_FACTOR * problem.optimization.filter_radius_mm
        );
        let capped = SimpEngine
            .optimize(&capped_problem, &SilentReporter)
            .expect("optimize");

        let cap = params.max_fraction;
        let before = local_measurement(&problem, &free.densities, params);
        let after = local_measurement(&capped_problem, &capped.densities, params);
        println!(
            "porous block under a {cap} cap: worst local {:.4} -> {:.4}, aggregate {:.4} -> \
             {:.4}, volume {:.4} -> {:.4}, compliance {:.6e} -> {:.6e}, {:?} at {} against {:?} \
             at {} iterations",
            before.worst,
            after.worst,
            before.aggregate,
            after.aggregate,
            free.volume_fraction,
            capped.volume_fraction,
            free.compliance,
            capped.compliance,
            free.stop,
            free.iterations,
            capped.stop,
            capped.iterations
        );

        // Both runs have to be finished designs, or none of the comparison
        // means anything.
        assert_eq!(free.stop, StopReason::Converged);
        assert_eq!(capped.stop, StopReason::Converged);
        // The uncapped run really does pack its material: a neighbourhood of it
        // holds nine tenths of what it could.
        assert!(
            before.worst > 0.85,
            "this fixture is only evidence while the free run packs solid: {before:?}"
        );
        assert!(
            (free.volume_fraction - mass).abs() < 1e-3,
            "the free run has to meet its volume target: {}",
            free.volume_fraction
        );
        // The constraint as the run states it is met: the aggregate is on the
        // cap, not above it.
        assert!(
            after.aggregate <= cap + constants::LOCAL_VOLUME_DUAL_TOLERANCE.max(1e-3),
            "the aggregate the cap is stated on was left at {}",
            after.aggregate
        );
        // And the true worst neighbourhood follows it down, to within the
        // p-mean's under-estimate. 0.12 is the measured 0.116 with a little
        // room; the aggregate assertion above is the tight one.
        assert!(
            after.worst <= cap + 0.12,
            "the capped run left a neighbourhood at {}, above the {cap} cap",
            after.worst
        );
        assert!(
            after.worst < before.worst - 0.15,
            "the cap has to change the design: {:.4} against {:.4}",
            after.worst,
            before.worst
        );
        // The volume target is an upper bound the cap may leave unreached, and
        // the run says what it really used.
        assert!(
            capped.volume_fraction <= mass + 1e-3,
            "the capped run went over its volume target: {}",
            capped.volume_fraction
        );
        assert!(capped.compliance < capped.initial_compliance);
        assert!(capped.densities.iter().all(|d| (0.0..=1.0).contains(d)));
        // Spreading material over more members costs stiffness. That is what is
        // being bought, so it may not be silently absent.
        assert!(
            capped.compliance > free.compliance,
            "a capped design cannot be stiffer than a free one: {} against {}",
            capped.compliance,
            free.compliance
        );
    }

    /// The per-iteration line carries the true worst neighbourhood while a cap
    /// is active, and nothing at all when there is none.
    #[test]
    fn the_worst_local_fraction_is_reported_only_while_a_cap_is_active() {
        use std::sync::Mutex;

        #[derive(Default)]
        struct Trace(Mutex<Vec<Option<f64>>>);
        impl Reporter for Trace {
            fn iteration(&self, stats: &IterationStats) {
                self.0
                    .lock()
                    .expect("trace")
                    .push(stats.worst_local_fraction);
            }
            fn note(&self, _message: &str) {}
        }

        let trace_of = |text: String| {
            let config = Config::parse(&text).expect("parse");
            let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
            let trace = Trace::default();
            SimpEngine.optimize(&problem, &trace).expect("optimize");
            trace.0.into_inner().expect("trace")
        };

        let plain = trace_of(porous_block("", 3));
        assert!(plain.iter().all(|v| v.is_none()), "{plain:?}");

        let capped = trace_of(porous_block("\n  [optimization.local_volume]", 3));
        assert_eq!(capped.len(), 3);
        for (iteration, reported) in capped.iter().enumerate() {
            let worst =
                reported.unwrap_or_else(|| panic!("iteration {} reported nothing", iteration + 1));
            assert!(
                (0.0..=1.0).contains(&worst),
                "iteration {} reported {worst}",
                iteration + 1
            );
        }
        // The design starts uniform at `mass_fraction`, so the first line's
        // worst neighbourhood is that fraction and nothing else.
        assert!(
            (capped[0].expect("a reading") - 0.3).abs() < 1e-9,
            "{capped:?}"
        );
        // And the run moves it: a uniform start is not what it stays.
        assert!(capped[2].expect("a reading") > capped[0].expect("a reading"));
    }

    #[test]
    fn self_weight_runs_and_still_meets_the_volume_target() {
        let field = run(&tiny("").replace(
            "vector = [0.0, 0.0, -20.0]",
            "vector = [0.0, 0.0, -20.0]\n[[loadcases.loads]]\ntype = \"gravity\"",
        ));
        assert!(
            (field.volume_fraction - 0.4).abs() < 1e-3,
            "volume fraction {} missed the target",
            field.volume_fraction
        );
        assert!(field.compliance.is_finite() && field.compliance > 0.0);
    }

    /// The tiny cantilever under a reduction schedule. `mass_fraction` is gone
    /// from it, so the run starts solid and the schedule alone decides what
    /// fraction comes out.
    fn reducing(keys: &str) -> String {
        tiny(&format!("[optimization.reduce]\n{keys}")).replace("mass_fraction = 0.4\n", "")
    }

    /// A reporter that keeps the notes and which stage every progress line said
    /// it belonged to.
    #[derive(Default)]
    struct Staged {
        stages: std::sync::Mutex<Vec<Option<ReduceProgress>>>,
        notes: std::sync::Mutex<Vec<String>>,
    }

    impl Reporter for Staged {
        fn iteration(&self, stats: &IterationStats) {
            self.stages.lock().expect("trace").push(stats.reduce);
        }
        fn note(&self, message: &str) {
            self.notes.lock().expect("trace").push(message.to_string());
        }
    }

    /// Run a reduction schedule and take back the field it exported, the stage
    /// every progress line belonged to, and the notes it said.
    fn reduce_run(keys: &str) -> (DensityField, Vec<Option<ReduceProgress>>, Vec<String>) {
        let config = Config::parse(&reducing(keys)).expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let reporter = Staged::default();
        let field = SimpEngine.optimize(&problem, &reporter).expect("optimize");
        (
            field,
            reporter.stages.into_inner().expect("trace"),
            reporter.notes.into_inner().expect("trace"),
        )
    }

    /// The schedule a table asks for: the fraction the run starts from, then
    /// `ratio` of the stage before, clamped at the floor - and a stage that
    /// reaches the floor still holding the target is the last one, because there
    /// is nowhere below it to go.
    #[test]
    fn a_reduction_descends_by_the_ratio_and_stops_at_the_floor() {
        let (field, lines, notes) = reduce_run(
            "target_safety_factor = 1.01\nratio = 0.8\nmin_mass_fraction = 0.6\n\
             refine_stages = 0\n",
        );
        assert_eq!(field.stop, StopReason::ReduceComplete);
        let summary = field.reduce.expect("a schedule");
        let targets: Vec<f64> = summary.stages.iter().map(|s| s.target_fraction).collect();
        // 1.0, then 0.8 of each in turn, and 0.512 clamped up to the floor.
        let expected = [1.0, 0.8, 0.64, 0.6];
        assert_eq!(targets.len(), expected.len(), "{targets:?}");
        for (got, want) in targets.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "{targets:?}");
        }
        assert!(
            summary.stages.iter().all(|s| s.passed && !s.refine),
            "nothing here should fail or refine: {:?}",
            summary.stages
        );
        assert_eq!(summary.exported.index, summary.stages.len());
        assert!(!summary.missed_the_target());
        // Every progress line says which stage it belongs to and what that stage
        // is driving to.
        let seen: Vec<ReduceProgress> = lines
            .iter()
            .map(|line| line.expect("a stage on every line"))
            .collect();
        for line in &seen {
            let record = summary.stages[line.stage - 1];
            assert_eq!(record.index, line.stage);
            assert!((line.target_fraction - record.target_fraction).abs() < 1e-12);
            assert_eq!(line.completed, line.stage - 1);
        }
        assert_eq!(seen.first().expect("a line").stage, 1);
        assert_eq!(seen.last().expect("a line").stage, summary.stages.len());
        assert!(
            notes
                .iter()
                .any(|n| n.starts_with("reduce stage 4: target")),
            "{notes:?}"
        );
    }

    /// A target no design of this domain can hold fails the fraction the run
    /// started from - solid here - and that is what it exports, under a warning
    /// that does not tell a solid design to find more material.
    #[test]
    fn an_unreachable_target_exports_the_solid_start_with_a_warning() {
        let (field, _, notes) = reduce_run("target_safety_factor = 1e6\n");
        assert_eq!(field.stop, StopReason::ReduceComplete);
        assert_eq!(field.volume_fraction, 1.0);
        assert!(field.densities.iter().all(|d| *d == 1.0), "not solid");
        let summary = field.reduce.expect("a schedule");
        assert_eq!(summary.stages.len(), 1, "{:?}", summary.stages);
        assert_eq!(summary.exported.index, 1);
        assert!(!summary.exported.passed && summary.missed_the_target());
        assert_eq!(summary.exported.achieved_fraction, 1.0);
        // The stage was measured, so the schedule failed it on the number rather
        // than on the geometry.
        assert!(summary.exported.safety_factor.is_some());
        let warning = notes
            .iter()
            .find(|n| n.starts_with("warning: [optimization.reduce]"))
            .expect("a warning");
        assert!(warning.contains("no stage held the target"), "{warning}");
        assert!(warning.contains("the design is already solid"), "{warning}");
    }

    /// The ordinary outcome: a stage holds the target, a lighter one does not,
    /// the bracket between them is bisected, and what comes out is the lightest
    /// design that held - lighter than the fraction the run started from.
    #[test]
    fn a_reachable_target_exports_a_design_lighter_than_the_start() {
        let (field, _, notes) = reduce_run(
            "target_safety_factor = 3.0\nratio = 0.5\nmin_mass_fraction = 0.1\n\
             refine_stages = 1\n",
        );
        assert_eq!(field.stop, StopReason::ReduceComplete);
        let summary = field.reduce.expect("a schedule");
        let exported = summary.exported;
        assert!(exported.passed && !summary.missed_the_target());
        assert!(
            exported.achieved_fraction < 1.0,
            "the schedule exported the start: {exported:?}"
        );
        assert!(exported.safety_factor.expect("a factor") >= summary.target_safety_factor);
        // The field is the exported stage's, not the last stage the loop ran.
        assert!((field.volume_fraction - exported.achieved_fraction).abs() < 1e-12);
        assert!(
            field.iterations
                == summary
                    .stages
                    .iter()
                    .map(|stage| stage.iterations)
                    .sum::<usize>(),
            "the run's iterations are the schedule's: {:?}",
            summary.stages
        );
        // The stage after the first failure is a bisection of the bracket around
        // it, and it is recorded as one.
        let first_fail = summary
            .stages
            .iter()
            .position(|stage| !stage.passed)
            .expect("a stage that failed");
        let refined = summary.stages[first_fail + 1];
        assert!(refined.refine, "{refined:?}");
        let bracket = (
            summary.stages[first_fail - 1].target_fraction,
            summary.stages[first_fail].target_fraction,
        );
        assert!(
            (refined.target_fraction - (bracket.0 + bracket.1) / 2.0).abs() < 1e-12,
            "{refined:?} does not bisect {bracket:?}"
        );
        assert!(
            notes.iter().any(|n| n.contains("(refinement)")),
            "{notes:?}"
        );
        assert!(
            notes
                .iter()
                .any(|n| n.starts_with("reduce: exported stage") && n.contains(">=")),
            "{notes:?}"
        );
    }

    /// A stage that failed on the budget rather than on the material says which
    /// it was, and every verdict a stage reaches names the stage it belongs to.
    #[test]
    fn a_capped_failing_stage_names_the_cap_and_every_verdict_names_its_stage() {
        let (_, _, notes) = reduce_run(
            "target_safety_factor = 3.0\nratio = 0.5\nmin_mass_fraction = 0.1\n\
             refine_stages = 0\n",
        );
        // No verdict of a stage is said without the stage it belongs to.
        for note in &notes {
            if note.contains("converged after") || note.contains("stopped at the iteration cap of")
            {
                assert!(note.starts_with("reduce stage "), "{note}");
            }
        }
        const CAP: &str = "the stage stopped at the iteration cap, so raising max_iterations may \
                           be what it needs";
        let capped: Vec<&String> = notes.iter().filter(|note| note.contains(CAP)).collect();
        assert!(!capped.is_empty(), "no capped stage failed: {notes:?}");
        // Only a stage that failed carries it: a stage that held the target was
        // asked for nothing more, whatever stopped it.
        for note in capped {
            assert!(note.contains(", fail"), "{note}");
        }
        assert!(
            notes
                .iter()
                .filter(|note| note.contains(", pass"))
                .all(|note| !note.contains(CAP)),
            "{notes:?}"
        );
    }

    /// The stage records reach the report the run writes beside the part.
    #[test]
    fn the_schedule_reaches_the_stress_report_json() {
        let config = Config::parse(&reducing(
            "target_safety_factor = 3.0\nratio = 0.5\nrefine_stages = 0\n",
        ))
        .expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let field = SimpEngine
            .optimize(&problem, &SilentReporter)
            .expect("optimize");
        let summary = field.reduce.expect("a schedule");
        let report = crate::stress::analyse(&problem, &field.densities).expect("a stress report");
        let json = crate::stress::to_json(&problem, &report, Some(&summary));
        assert!(json.contains("\"reduce\": {"), "{json}");
        assert!(
            json.contains(&format!("\"exported_stage\": {}", summary.exported.index)),
            "{json}"
        );
        assert_eq!(
            json.matches("\"target_fraction\"").count(),
            summary.stages.len()
        );
        assert!(json.contains("\"method\": \"continuation\""), "{json}");
    }

    /// The evolutionary method shares the schedule but not the update rule, and
    /// that rule is not in this build: the run refuses rather than reporting a
    /// method it did not use.
    #[test]
    fn the_evolutionary_method_is_refused_by_the_engine() {
        let config = Config::parse(&reducing("target_safety_factor = 2.0\nmethod = \"beso\"\n"))
            .expect("parse");
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let error = SimpEngine
            .optimize(&problem, &SilentReporter)
            .expect_err("the evolutionary method is not built");
        let text = format!("{error:#}");
        assert!(
            text.contains("beso") && text.contains("not implemented in this build"),
            "{text}"
        );
    }
}
