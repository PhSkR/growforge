//! What bounds one linear solve, and who may call it off part way through.
//!
//! A solve is the longest uninterruptible thing growforge does: one late SIMP
//! iteration on a marginal structure can spend thousands of conjugate gradient
//! iterations, and a stress pass on the final field can spend tens of
//! thousands. Cancellation used to reach only as far as the *loop around* the
//! solves - [`crate::report::Reporter::cancelled`], asked between optimization
//! iterations and at the viewer's stage boundaries - so a stop pressed during
//! one long solve did nothing at all until that solve had finished.
//!
//! [`CancelProbe`] carries the question down to the iteration itself. It is
//! optional and off by default: every command line path builds its limits with
//! [`SolveLimits::new`], passes no probe, and runs exactly the arithmetic it
//! always did, checkpoint branch included - the check is one predictable
//! `Option` test per [`crate::constants::CG_CANCEL_CHECK_INTERVAL`] iterations.
//!
//! A cancelled solve is neither an error nor a non-convergence: nobody wants
//! the answer any more. See [`crate::fea::Solve`].

use crate::constants;

/// Whether the caller still wants the answer.
///
/// Cloneable and copyable so it can be handed to a nested solve without
/// ceremony, and cheap to ask: a probe that is not there costs one `Option`
/// test, and one that is costs whatever the closure behind it does - in
/// practice one relaxed atomic load.
#[derive(Clone, Copy, Default)]
pub struct CancelProbe<'a> {
    ask: Option<&'a dyn Fn() -> bool>,
}

impl std::fmt::Debug for CancelProbe<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CancelProbe")
            .field("watched", &self.ask.is_some())
            .finish()
    }
}

impl<'a> CancelProbe<'a> {
    /// A probe nobody answers: the solve runs to its own conclusion.
    pub const NONE: CancelProbe<'a> = CancelProbe { ask: None };

    /// A probe that puts the question to `ask`.
    pub fn watching(ask: &'a dyn Fn() -> bool) -> CancelProbe<'a> {
        CancelProbe { ask: Some(ask) }
    }

    /// True when someone is watching this solve at all.
    pub fn is_watched(&self) -> bool {
        self.ask.is_some()
    }

    /// Whether the solve should stop, asked on checkpoint iterations only.
    ///
    /// `iteration` is one-based and counts whatever the caller calls an
    /// iteration; the checkpoint is every
    /// [`constants::CG_CANCEL_CHECK_INTERVAL`]-th of them. Off a checkpoint the
    /// closure is not called at all, which is what makes the seam free for the
    /// solves nobody is watching *and* for the ones that are.
    pub fn cancelled_at(&self, iteration: usize) -> bool {
        iteration.is_multiple_of(constants::CG_CANCEL_CHECK_INTERVAL) && self.cancelled()
    }

    /// Whether the solve should stop, asked now.
    ///
    /// Used where the natural checkpoint is already an expensive boundary - a
    /// device readback, the start of a refinement pass - and counting
    /// iterations to it would only make the answer later.
    pub fn cancelled(&self) -> bool {
        self.ask.is_some_and(|ask| ask())
    }
}

/// How far one linear solve has to go, how long it may take, and who may call
/// it off.
#[derive(Clone, Copy, Debug)]
pub struct SolveLimits<'a> {
    /// Relative residual `||r|| / ||b||` the solve stops at.
    pub tolerance: f64,
    /// Hard iteration cap; overrunning it is a non-convergence error.
    pub max_iterations: usize,
    /// Who may call the solve off. [`CancelProbe::NONE`] by default.
    pub cancel: CancelProbe<'a>,
}

impl SolveLimits<'_> {
    /// Limits nobody is watching, which is what every command line path uses.
    pub fn new(tolerance: f64, max_iterations: usize) -> SolveLimits<'static> {
        SolveLimits {
            tolerance,
            max_iterations,
            cancel: CancelProbe::NONE,
        }
    }
}

impl<'a> SolveLimits<'a> {
    /// The same limits, with `ask` able to stop the solve.
    pub fn watching(self, ask: &'a dyn Fn() -> bool) -> SolveLimits<'a> {
        SolveLimits {
            cancel: CancelProbe::watching(ask),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn an_unwatched_probe_never_stops_anything() {
        let probe = CancelProbe::NONE;
        assert!(!probe.is_watched());
        for iteration in 0..1000 {
            assert!(!probe.cancelled_at(iteration));
        }
        assert!(!probe.cancelled());
        assert!(format!("{probe:?}").contains("false"));
    }

    #[test]
    fn a_watched_probe_is_asked_only_on_the_checkpoint_iterations() {
        let asked = Cell::new(0usize);
        let ask = || {
            asked.set(asked.get() + 1);
            false
        };
        let probe = SolveLimits::new(1e-8, 10).watching(&ask).cancel;
        assert!(probe.is_watched());
        let interval = constants::CG_CANCEL_CHECK_INTERVAL;
        for iteration in 1..=(4 * interval) {
            probe.cancelled_at(iteration);
        }
        assert_eq!(
            asked.get(),
            4,
            "the probe must be asked once per {interval} iterations"
        );
        // And the unconditional form is asked every time it is called.
        probe.cancelled();
        assert_eq!(asked.get(), 5);
    }

    #[test]
    fn a_probe_that_says_stop_is_believed_at_the_next_checkpoint() {
        let stop = Cell::new(false);
        let ask = || stop.get();
        let limits = SolveLimits::new(1e-8, 10).watching(&ask);
        let interval = constants::CG_CANCEL_CHECK_INTERVAL;
        assert!(!limits.cancel.cancelled_at(interval));
        stop.set(true);
        // Not until the next checkpoint, and then certainly.
        assert!(!limits.cancel.cancelled_at(interval + 1));
        assert!(limits.cancel.cancelled_at(2 * interval));
        // The limits themselves are carried through untouched.
        assert_eq!(limits.tolerance, 1e-8);
        assert_eq!(limits.max_iterations, 10);
    }
}
