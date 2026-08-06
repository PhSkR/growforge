//! Detecting a SIMP run that has stopped making progress.
//!
//! The optimization loop has always had one way to stop early - the design
//! variable change falling below `[optimization] convergence_tol` - and one way
//! to stop late, the iteration cap. Between them sits the run that will never do
//! either: every iteration takes a step of the full move limit, the compliance
//! wanders inside a band, and neither changes for as long as there is budget.
//! That run is what this module recognises, so it can be stopped honestly rather
//! than at the far end of a budget that is now
//! [`constants::DEFAULT_MAX_ITERATIONS`] long.
//!
//! Two things are asked of a window of iterations, and the first is what
//! separates the cases:
//!
//! 1. **The design is still traversing.** Every iteration in the window took a
//!    step of at least [`constants::STALL_MOVE_LIMIT_FRACTION`] of the update
//!    scheme's move limit - the optimizer asking, iteration after iteration, to
//!    move further than it is allowed - and the smallest such step of the
//!    window's second half is no more than [`constants::STALL_CHANGE_DECAY`]
//!    below the first half's.
//! 2. **And it is buying nothing.** The best compliance of the second half
//!    betters the best of the first half by less than
//!    [`constants::STALL_COMPLIANCE_GAIN`].
//!
//! The measurements behind every threshold, the six traced runs they come from
//! and the asymmetric bias between a missed stall and a false one are documented
//! on the constants themselves.

use std::collections::VecDeque;

use crate::config::UpdateScheme;
use crate::constants;

/// One iteration as the stall test sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Weighted compliance objective of the iteration, in N*mm.
    pub compliance: f64,
    /// Largest absolute design variable change the iteration's update took.
    pub max_change: f64,
}

/// The smallest value of an iterator, or `None` when anything in it is not a
/// usable number.
///
/// A window holding a NaN or an infinity says nothing either way, and "says
/// nothing" has to read as "not stalled": see the bias documented on
/// [`constants::STALL_WINDOW_ITERATIONS`].
fn finite_min(mut values: impl Iterator<Item = f64>) -> Option<f64> {
    values.try_fold(f64::INFINITY, |best, value| {
        value.is_finite().then(|| best.min(value))
    })
}

/// Relative improvement of `candidate` on `reference`, both of which are
/// positive.
///
/// Negative when the candidate is worse, which is a legitimate reading here: a
/// window whose second half is worse than its first half has bought nothing, and
/// falls below every threshold this module compares against.
fn gain(reference: f64, candidate: f64) -> f64 {
    (reference - candidate) / reference
}

/// True when `window` shows a run that has stopped making progress.
///
/// `window` is the run's trailing history in iteration order; only its last
/// [`constants::STALL_WINDOW_ITERATIONS`] entries are read, and a shorter
/// history is never stalled - there is not yet enough of it to say so.
/// `move_limit` is the furthest one step of the update scheme that produced
/// these samples may move a design variable, which is what a change is read
/// against: see [`UpdateScheme::move_limit`].
///
/// The three conditions, all of which must hold:
///
/// 1. **Traversing.** Every change in the window is at least
///    [`constants::STALL_MOVE_LIMIT_FRACTION`] of `move_limit`.
/// 2. **Not settling.** The smallest change of the window's second half improves
///    on the smallest of its first half by less than
///    [`constants::STALL_CHANGE_DECAY`].
/// 3. **Buying nothing.** The best compliance of the second half improves on the
///    best of the first half by less than
///    [`constants::STALL_COMPLIANCE_GAIN`].
///
/// The halves are compared on their *best* value rather than their endpoints,
/// because the trajectories this has to separate oscillate: a descending run
/// keeps setting new lows through its oscillation and a wandering one stops
/// setting them, while an endpoint comparison on either would be reading the
/// phase of the wander.
pub fn is_stalled(window: &[Sample], move_limit: f64) -> bool {
    let length = constants::STALL_WINDOW_ITERATIONS;
    if window.len() < length {
        return false;
    }
    let window = &window[window.len() - length..];
    let (first, second) = window.split_at(length / 2);

    // Non-finite or non-positive values make every ratio below meaningless.
    let Some(change_floor) = finite_min(window.iter().map(|s| s.max_change)) else {
        return false;
    };
    if change_floor < constants::STALL_MOVE_LIMIT_FRACTION * move_limit {
        return false;
    }
    let Some(compliance_floor) = finite_min(window.iter().map(|s| s.compliance)) else {
        return false;
    };
    if compliance_floor <= 0.0 {
        return false;
    }

    let best =
        |half: &[Sample], of: fn(&Sample) -> f64| half.iter().map(of).fold(f64::INFINITY, f64::min);
    let change = |sample: &Sample| sample.max_change;
    let compliance = |sample: &Sample| sample.compliance;

    gain(best(first, change), best(second, change)) < constants::STALL_CHANGE_DECAY
        && gain(best(first, compliance), best(second, compliance))
            < constants::STALL_COMPLIANCE_GAIN
}

/// The trailing window [`is_stalled`] is asked about, kept by the optimization
/// loop.
///
/// It holds [`constants::STALL_WINDOW_ITERATIONS`] samples and nothing more, so
/// a run of any length costs the same fixed handful of kilobytes and the same
/// fixed comparison per iteration - nothing next to one finite element solve.
#[derive(Debug, Clone)]
pub struct StallWatch {
    move_limit: f64,
    window: VecDeque<Sample>,
}

impl StallWatch {
    /// A watch for a run whose update scheme may move a design variable at most
    /// `move_limit` per step.
    pub fn new(move_limit: f64) -> StallWatch {
        StallWatch {
            move_limit,
            window: VecDeque::with_capacity(constants::STALL_WINDOW_ITERATIONS),
        }
    }

    /// Record one finished iteration and answer whether the run has stalled.
    pub fn observe(&mut self, sample: Sample) -> bool {
        if self.window.len() == constants::STALL_WINDOW_ITERATIONS {
            self.window.pop_front();
        }
        self.window.push_back(sample);
        is_stalled(self.window.make_contiguous(), self.move_limit)
    }
}

/// What the run says when it stops on the stall criterion.
///
/// Scheme aware, because the remedy is: the optimality criteria step has a
/// documented alternative that settles problems it cannot, and MMA - which is
/// that alternative - does not. The message never claims the design is bad: it
/// is a real iterate at the volume target, it exports, and the stress report
/// below it says what it carries.
///
/// `local_volume` says whether an `[optimization.local_volume]` cap was active,
/// because it changes what a design still traversing at the move limit *means*.
/// Under a local cap the material has nowhere concentrated to go: the optimizer
/// trades it between neighbourhoods, and doing that at the move limit for a
/// hundred iterations is what a converged bone-like design looks like from the
/// inside rather than a failure to settle.
pub fn stall_note(
    iterations: usize,
    max_change: f64,
    convergence_tol: f64,
    update: UpdateScheme,
    local_volume: bool,
) -> String {
    let advice = if local_volume {
        "under [optimization.local_volume] this is often the converged character of the design \
         rather than a failure: a local cap leaves the optimizer trading material between \
         neighbourhoods that are all equally full, and that trade never quite stops. Take this \
         iterate - it holds the cap and the stress report says what it carries - or raise \
         max_fraction if you want fewer, thicker members"
    } else {
        match update {
            UpdateScheme::Oc => {
                "this problem does not settle under update = \"oc\"; consider update = \"mma\", \
                 which damps exactly the variables that keep crossing the box"
            }
            UpdateScheme::Mma => {
                "this problem does not settle under update = \"mma\" either; raise max_iterations \
                 if you want it to keep looking, or take this iterate - it meets the volume \
                 constraint, and the stress report says what it carries"
            }
        }
    };
    // The tolerance is printed as it was written rather than to a fixed number
    // of places: a run asking for 1e-9 would otherwise be told it is above
    // "0.00000".
    format!(
        "stopped after {iterations} iterations: the compliance has established no better design \
         for {} iterations and the design is still traversing at the {} move limit (max change \
         {max_change:.5} against the {convergence_tol} asked for) - {advice}",
        constants::STALL_WINDOW_ITERATIONS,
        update.label()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The move limit both shipped update schemes take.
    const LIMIT: f64 = constants::OC_MOVE_LIMIT;

    /// Half the window, which is what the two comparisons are made over.
    const HALF: usize = constants::STALL_WINDOW_ITERATIONS / 2;

    /// The unambiguous stall: a compliance that does not move at all while every
    /// step is clipped by the move limit. Both comparisons read exactly zero, so
    /// it is what the boundary tests hold everything else against.
    fn pinned(count: usize) -> Vec<Sample> {
        vec![
            Sample {
                compliance: 4.313e1,
                max_change: LIMIT,
            };
            count
        ]
    }

    /// The documented never-settler, as it was traced: the 120 mm deep variant
    /// of the shipped shelf bracket under `oc` takes a step of exactly the 0.2
    /// move limit on every one of its 400 iterations while the compliance
    /// wanders inside a 2.7 % band around 4.3e1, establishing no better design.
    ///
    /// The wander here is deterministic and deliberately ugly - two
    /// incommensurable periods, so no window is a repeat of the one before it.
    fn wandering_at_the_move_limit(count: usize) -> Vec<Sample> {
        (0..count)
            .map(|i| {
                let phase = (i as f64 * 0.7).sin() + (i as f64 * 0.31).cos();
                Sample {
                    compliance: 4.313e1 * (1.0 + 0.0068 * phase),
                    max_change: LIMIT,
                }
            })
            .collect()
    }

    /// A run whose compliance falls by `per_iteration` of itself each step and
    /// whose change decays by `change_decay` of itself each step, starting from
    /// the move limit.
    fn trajectory(count: usize, per_iteration: f64, change_decay: f64) -> Vec<Sample> {
        let mut compliance = 4.313e1;
        let mut max_change = LIMIT;
        (0..count)
            .map(|_| {
                let sample = Sample {
                    compliance,
                    max_change,
                };
                compliance *= 1.0 - per_iteration;
                max_change *= 1.0 - change_decay;
                sample
            })
            .collect()
    }

    #[test]
    fn a_window_that_is_not_full_yet_is_never_stalled() {
        let short = pinned(constants::STALL_WINDOW_ITERATIONS - 1);
        assert!(!is_stalled(&short, LIMIT));
        // One more sample is the whole difference: the boundary is the window
        // length itself, not a sample either side of it.
        let full = pinned(constants::STALL_WINDOW_ITERATIONS);
        assert!(is_stalled(&full, LIMIT));
        assert!(!is_stalled(&[], LIMIT));
        // Only the trailing window is read, so a long healthy run that has just
        // gone flat is judged on the flat part alone.
        let mut history = trajectory(500, 0.01, 0.0);
        history.extend(pinned(constants::STALL_WINDOW_ITERATIONS));
        assert!(is_stalled(&history, LIMIT));
    }

    #[test]
    fn the_documented_never_settler_is_called_stalled() {
        let stuck = wandering_at_the_move_limit(400);
        let mut watch = StallWatch::new(LIMIT);
        let fired = stuck
            .iter()
            .position(|sample| watch.observe(*sample))
            .expect("400 iterations at the move limit with no better design is a stall")
            + 1;
        // Never before a full window, and long before the 400 iterations the
        // traced case spent waiting for a cap.
        assert!(
            (constants::STALL_WINDOW_ITERATIONS..=2 * constants::STALL_WINDOW_ITERATIONS)
                .contains(&fired),
            "the wandering case was called at iteration {fired}"
        );
    }

    /// The traced slow converger: the same deep bracket under `mma` spends its
    /// first ~130 iterations pinned at the move limit too, and what tells it
    /// apart is that they buy 2.9 % to 12.8 % of compliance per window against
    /// the non-settler's few tenths. It converges at 267 and may never be cut
    /// off on the way.
    #[test]
    fn a_run_still_buying_compliance_at_the_move_limit_is_never_called_stalled() {
        // 0.03 % per iteration is 2.9 % per window: the slowest window measured
        // on that run while it was pinned, and still six times the threshold.
        let descending = trajectory(400, 0.000_3, 0.0);
        let mut watch = StallWatch::new(LIMIT);
        assert!(
            descending.iter().all(|sample| !watch.observe(*sample)),
            "a run buying 2.9 % of compliance per window was called stalled"
        );
    }

    /// The other half of the same claim, and the one every converging run
    /// measured relies on: a change that comes off the move limit is a design
    /// working towards an answer, and it keeps its budget however flat the
    /// compliance has gone.
    #[test]
    fn a_change_that_has_come_off_the_move_limit_is_never_called_stalled() {
        // The closest a converging run came: the deep bracket under `mma` held
        // 57 % of the move limit through a window whose compliance bought 0.4 %.
        let converging: Vec<Sample> = (0..400)
            .map(|i| Sample {
                compliance: 4.313e1 * (1.0 - 4e-5 * i as f64),
                max_change: 0.57 * LIMIT,
            })
            .collect();
        let mut watch = StallWatch::new(LIMIT);
        assert!(converging.iter().all(|sample| !watch.observe(*sample)));

        // And the shipped examples, whose worst windows sit at 14 % to 29 % of
        // the move limit with the compliance just as flat.
        for fraction in [0.14, 0.29, 0.53] {
            let flat: Vec<Sample> = vec![
                Sample {
                    compliance: 4.313e1,
                    max_change: fraction * LIMIT,
                };
                constants::STALL_WINDOW_ITERATIONS
            ];
            assert!(
                !is_stalled(&flat, LIMIT),
                "a design at {fraction} of the move limit is not traversing"
            );
        }
    }

    /// The thresholds are boundaries, and a test that does not stand on them
    /// would not notice one being moved.
    #[test]
    fn the_thresholds_are_where_the_constants_say_they_are() {
        // The traversing leg: one iteration below the fraction is enough to
        // spare the whole window, because the test is over every one of them.
        // The dip goes in the *first* half, where the change leg reads it as the
        // run getting no better rather than as progress; a dip in the second
        // half is a change coming down, and the change leg spares that window
        // before this one is reached.
        let mut dipped = pinned(constants::STALL_WINDOW_ITERATIONS);
        dipped[HALF - 3].max_change = constants::STALL_MOVE_LIMIT_FRACTION * LIMIT * 0.99;
        assert!(!is_stalled(&dipped, LIMIT));
        dipped[HALF - 3].max_change = constants::STALL_MOVE_LIMIT_FRACTION * LIMIT * 1.01;
        assert!(is_stalled(&dipped, LIMIT));
        let mut coming_down = pinned(constants::STALL_WINDOW_ITERATIONS);
        coming_down[HALF + 3].max_change = constants::STALL_MOVE_LIMIT_FRACTION * LIMIT * 1.01;
        assert!(!is_stalled(&coming_down, LIMIT));

        // The compliance leg, with the change pinned.
        let bought = |gain: f64| {
            let mut samples = pinned(constants::STALL_WINDOW_ITERATIONS);
            for sample in samples.iter_mut().skip(HALF) {
                sample.compliance *= 1.0 - gain;
            }
            samples
        };
        assert!(!is_stalled(
            &bought(constants::STALL_COMPLIANCE_GAIN * 1.01),
            LIMIT
        ));
        assert!(is_stalled(
            &bought(constants::STALL_COMPLIANCE_GAIN * 0.99),
            LIMIT
        ));

        // The change leg, with the compliance flat and everything still above
        // the traversing fraction.
        let settling = |decay: f64| {
            let mut samples = pinned(constants::STALL_WINDOW_ITERATIONS);
            for sample in samples.iter_mut().skip(HALF) {
                sample.max_change *= 1.0 - decay;
            }
            samples
        };
        let headroom = 1.0 - constants::STALL_MOVE_LIMIT_FRACTION;
        assert!(
            constants::STALL_CHANGE_DECAY < headroom,
            "a decay the traversing leg would reject first proves nothing"
        );
        assert!(!is_stalled(
            &settling(constants::STALL_CHANGE_DECAY * 1.01),
            LIMIT
        ));
        assert!(is_stalled(
            &settling(constants::STALL_CHANGE_DECAY * 0.99),
            LIMIT
        ));
    }

    /// The change is read against the scheme's own move limit, so a scheme that
    /// was allowed to travel further is not called stalled for taking the same
    /// step.
    #[test]
    fn the_move_limit_the_steps_are_read_against_is_the_schemes_own() {
        let window = pinned(constants::STALL_WINDOW_ITERATIONS);
        assert!(is_stalled(&window, LIMIT));
        assert!(!is_stalled(&window, LIMIT * 2.0));
        assert_eq!(UpdateScheme::Oc.move_limit(), constants::OC_MOVE_LIMIT);
        assert_eq!(UpdateScheme::Mma.move_limit(), constants::MMA_MOVE_LIMIT);
    }

    /// Nothing the test cannot read may be called a stall.
    #[test]
    fn an_unreadable_window_is_not_a_stall() {
        let broken = |compliance: f64, max_change: f64| {
            let mut samples = pinned(constants::STALL_WINDOW_ITERATIONS);
            assert!(
                is_stalled(&samples, LIMIT),
                "the baseline has to be a stall"
            );
            samples[HALF] = Sample {
                compliance,
                max_change,
            };
            samples
        };
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!is_stalled(&broken(bad, LIMIT), LIMIT));
            assert!(!is_stalled(&broken(4.313e1, bad), LIMIT));
        }
        // A compliance of zero (or below) has no relative improvement to speak
        // of, so it is not read as a plateau either.
        assert!(!is_stalled(&broken(0.0, LIMIT), LIMIT));
    }

    #[test]
    fn the_stall_note_names_the_scheme_and_its_remedy() {
        let tol = constants::DEFAULT_CONVERGENCE_TOL;
        let oc = stall_note(437, LIMIT, tol, UpdateScheme::Oc, false);
        assert!(oc.contains("stopped after 437 iterations"), "{oc}");
        assert!(oc.contains("still traversing at the oc move limit"), "{oc}");
        assert!(oc.contains("update = \"mma\""), "{oc}");
        let mma = stall_note(
            500,
            constants::MMA_MOVE_LIMIT,
            tol,
            UpdateScheme::Mma,
            false,
        );
        assert!(mma.contains("max_iterations"), "{mma}");
        assert!(mma.contains("take this iterate"), "{mma}");
        // Neither may read as a failure: a stalled run is a finished run.
        for note in [oc, mma] {
            assert!(
                !note.contains("error") && !note.contains("failed"),
                "{note}"
            );
        }
    }

    /// Under a local volume cap the same reading means something else: the
    /// design trading material between neighbourhoods at the move limit is what
    /// a converged bone-like structure does, so the note says so instead of
    /// sending the user to the scheme they are already on.
    #[test]
    fn the_stall_note_reads_a_capped_run_as_its_converged_character() {
        let tol = constants::DEFAULT_CONVERGENCE_TOL;
        let capped = stall_note(311, constants::MMA_MOVE_LIMIT, tol, UpdateScheme::Mma, true);
        assert!(capped.contains("stopped after 311 iterations"), "{capped}");
        assert!(capped.contains("[optimization.local_volume]"), "{capped}");
        assert!(capped.contains("converged character"), "{capped}");
        assert!(capped.contains("max_fraction"), "{capped}");
        // The remedy for a run that will not settle is not the remedy here, and
        // offering it would send a user to the scheme they are already on.
        assert!(!capped.contains("does not settle"), "{capped}");
        assert!(
            !capped.contains("error") && !capped.contains("failed"),
            "{capped}"
        );
    }
}
