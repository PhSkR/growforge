//! The bi-directional evolutionary update: a design that is cells rather than
//! densities, re-cut every iteration.
//!
//! `[optimization.reduce] method = "beso"` replaces the design variable step of
//! [`super::oc`] and [`super::mma`] with the soft-kill evolutionary rule of
//! Huang and Xie. One iteration is
//!
//! 1. rank every design cell by its compliance sensitivity,
//! 2. smooth the ranking over the density filter's own neighbourhood and average
//!    it with the previous iteration's,
//! 3. lower the volume target by `evolution_rate` of itself, floored at the
//!    stage's target,
//! 4. cut the ranking where the cells above it fill that volume - letting at
//!    most `add_ratio` of the current volume back in from below the cut - and
//!    set every cell above it solid and every cell below it to
//!    [`constants::DENSITY_MIN`].
//!
//! Everything else about the run is unchanged: the same objective evaluation,
//! the same density chain, the same stage schedule of
//! [`super::simp::SimpEngine::optimize`] around it, and the same stress report
//! deciding what a stage was worth.
//!
//! **Why the SIMP sensitivity.** The ranking is `-dC/dx` at the current design,
//! which is what [`super::objective`] already computes at `[optimization]
//! penalty` for the update schemes. A design that is 0/1 in its design
//! variables is not 0/1 where the penalization reads it: the density chain
//! spreads every cell over its neighbourhood first, so `-dC/dx` is the chain
//! transpose of the penalized elemental strain energies of the *printed* field:
//! the classic filtered sensitivity, which is the quantity the evolutionary
//! method ranks by, carried back onto the cells that made it. That is the
//! soft-kill
//! reading: the removed cells keep the stiffness `[optimization]
//! stiffness_floor` gives them, they are therefore still in the model, and their
//! sensitivity says what putting them back would buy. It is also why the low
//! value here is [`constants::DENSITY_MIN`] rather than an invented small
//! density: the softness of the kill is the stiffness floor's, where every other
//! part of this crate already reads it, and a cell at zero density still ranks
//! because the density filter carries its neighbours' material into it.
//!
//! **The smoother.** The ranking is filtered through the density filter's
//! transpose, normalized so that its rows sum to one - the weighted
//! neighbourhood average the method's mesh-independence rests on, at exactly the
//! radius the density filter runs at (`min_feature_mm / 2`). The transpose
//! rather than the forward filter because the forward one reads a forced solid
//! neighbour as a density of one, which is a statement about material and not
//! about sensitivity. This pass is on top of the chain transpose the objective
//! already took: that one is the gradient's, and this one is the length scale's.
//!
//! **The overhang filter, when one is on**, needs nothing here and is not
//! refused. It is a stage of the density chain, so it shapes the field the
//! analysis sees exactly as it does under SIMP, and a cell it erases has a zero
//! row in the chain: its sensitivity is zero, it ranks last, and the cut removes
//! it - the same fate [`super::oc`] gives such a cell for the same reason. Should
//! material grow underneath it later, the row stops being zero, the sensitivity
//! comes back, and so may the cell.
//!
//! **Never removed:** the design cells of every support and load region, the
//! purposes of [`crate::trim`]. A support pad carries no strain energy *because*
//! it is held, so the ranking puts it last, and a design that has thrown its
//! supports away is not a lighter design but a broken one. Keepins need no
//! protection here - they are forced solid cells and never design variables.

use std::collections::VecDeque;

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::constants;
use crate::engine::chain::DensityChain;
use crate::engine::update::{Buffers, Constraint, Step, design_mean};
use crate::grid::CellKind;
use crate::problem::Problem;

/// What the evolutionary update's own settling test says about the stage it is
/// running.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Settling {
    /// The design is still moving: either the volume has not reached the stage's
    /// target yet, or the compliance is still changing over the window.
    Running,
    /// The volume is at the stage's target and the compliance has stopped
    /// moving. The number is the relative change the window measured.
    Settled(f64),
    /// The volume is at the stage's target and the last cut flipped no cell at
    /// all: the ranking asks for the design it already has.
    Unmoved,
}

/// The evolutionary update and everything it carries between iterations.
pub struct State {
    /// Share of the current volume one iteration takes away.
    evolution_rate: f64,
    /// Largest share of the current volume one iteration may let back.
    add_ratio: f64,
    /// Design cells a support or a load region covers, which no cut removes.
    protected: Vec<bool>,
    /// How many of them there are: the volume no target can go below.
    protected_cells: usize,
    /// Row sums of the sensitivity smoother, so its rows can be normalized to
    /// one. Fixed for the run: it is a property of the grid and the radius.
    smoother_norm: Vec<f64>,
    /// The volume target the last iteration was cut to, as a fraction of the
    /// design cells. `None` until the first cut, which is where the field the
    /// schedule handed over becomes a 0/1 design.
    target: Option<f64>,
    /// The averaged sensitivity fields of the last
    /// [`constants::BESO_SENSITIVITY_HISTORY`] iterations, oldest first.
    history: VecDeque<Vec<f64>>,
    /// The compliance of the last `2 *`
    /// [`constants::BESO_CONVERGENCE_WINDOW`] iterations, oldest first.
    compliances: VecDeque<f64>,
    /// Whether the last cut landed the design on the stage's own target.
    at_target: bool,
    /// Whether the last cut flipped no cell at all.
    unmoved: bool,
    /// `-dC/dx` over the design cells, zero everywhere else.
    raw: Vec<f64>,
    /// The same after the smoother and the history average.
    ranking: Vec<f64>,
    /// The design cells in ranking order, protected ones first.
    order: Vec<usize>,
    /// Which cells the last cut kept.
    keep: Vec<bool>,
}

impl State {
    /// Build the update for `problem`, ranked through `chain`'s density filter.
    pub fn new(
        problem: &Problem,
        chain: &DensityChain,
        evolution_rate: f64,
        add_ratio: f64,
    ) -> State {
        let grid = &problem.grid;
        let n_cells = grid.n_cells();
        let mut protected = vec![false; n_cells];
        for purpose in crate::trim::purposes(problem) {
            for &e in &purpose.cells {
                if grid.cells[e] == CellKind::Design {
                    protected[e] = true;
                }
            }
        }
        let protected_cells = protected.iter().filter(|held| **held).count();
        // The smoother is the filter transpose applied to a unit field over the
        // design cells: every design cell's row sum, taken once.
        let unit: Vec<f64> = grid
            .cells
            .iter()
            .map(|kind| f64::from(*kind == CellKind::Design))
            .collect();
        let mut smoother_norm = vec![0.0; n_cells];
        chain.filter().apply_transpose(&unit, &mut smoother_norm);
        State {
            evolution_rate,
            add_ratio,
            protected,
            protected_cells,
            smoother_norm,
            target: None,
            history: VecDeque::new(),
            compliances: VecDeque::new(),
            at_target: false,
            unmoved: false,
            raw: vec![0.0; n_cells],
            ranking: vec![0.0; n_cells],
            order: Vec::new(),
            keep: vec![false; n_cells],
        }
    }

    /// How many design cells a support or a load region protects from every cut.
    pub fn protected_cells(&self) -> usize {
        self.protected_cells
    }

    /// Put the update back onto a design it did not itself produce: `x`.
    ///
    /// The stage schedule restarts a refinement from the design that held the
    /// target safety factor rather than from the lighter one that did not, and
    /// everything carried between iterations describes the trajectory that was
    /// abandoned to go back to it. The running volume target becomes the one
    /// this design is at - so the next cut descends from it rather than from
    /// wherever the abandoned stage got to - and the sensitivity history, the
    /// compliance window and the settling flags start again on it.
    pub fn restart(&mut self, x: &[f64], design_cells: &[usize]) {
        self.target = Some(design_mean(x, design_cells));
        self.history.clear();
        self.compliances.clear();
        self.at_target = false;
        self.unmoved = false;
    }

    /// Take one evolutionary step from `x` under `dc = dC/dx`.
    ///
    /// The volume sensitivity is not read: the volume of a 0/1 design is the
    /// number of cells above the cut, and the cut is where the constraint is
    /// met, so there is no multiplier to price it against.
    pub fn update(
        &mut self,
        x: &[f64],
        dc: &[f64],
        constraint: &Constraint<'_>,
        buffers: Buffers<'_>,
    ) -> Result<Step> {
        let design = constraint.design_cells;
        let cells = design.len();
        let Buffers {
            x_new,
            filtered,
            printed,
        } = buffers;

        for &e in design {
            self.raw[e] = -dc[e];
        }
        constraint
            .chain
            .filter()
            .apply_transpose(&self.raw, &mut self.ranking);
        let (norm, ranking) = (&self.smoother_norm, &mut self.ranking);
        for &e in design {
            if norm[e] > 0.0 {
                ranking[e] /= norm[e];
            }
        }
        if let Some(&bad) = design.iter().find(|&&e| !self.ranking[e].is_finite()) {
            bail!(
                "the evolutionary update read a non-finite compliance sensitivity at cell {bad}; \
                 the design it was taken at cannot be ranked"
            );
        }
        // The history of one: the field this cut ranks by is the average of the
        // current sensitivities and the averaged field the last cut used, which
        // is what the method's convergence rests on.
        let divisor = (self.history.len() + 1) as f64;
        for past in &self.history {
            for &e in design {
                self.ranking[e] += past[e];
            }
        }
        for &e in design {
            self.ranking[e] /= divisor;
        }
        if constants::BESO_SENSITIVITY_HISTORY > 0 {
            let mut slot = match self.history.len() >= constants::BESO_SENSITIVITY_HISTORY {
                true => self.history.pop_front().expect("a non-empty history"),
                false => vec![0.0; self.ranking.len()],
            };
            slot.copy_from_slice(&self.ranking);
            self.history.push_back(slot);
        }

        // Everything the cut is about: how much volume this iteration is for,
        // how much of it there is now, and how much of it may be new.
        let low = constants::DENSITY_MIN;
        let high = constants::DENSITY_MAX;
        let solid_now = |e: usize| x[e] > 0.5 * (low + high);
        let solid_count = design.iter().filter(|&&e| solid_now(e)).count();
        let seeding = self.target.is_none();
        let previous = self.target.unwrap_or_else(|| design_mean(x, design));
        let target =
            (previous * (1.0 - self.evolution_rate)).max(constraint.target_volume_fraction);
        let wanted = ((target * cells as f64).round() as usize).clamp(self.protected_cells, cells);
        // The cap governs how far one iteration may move a 0/1 design. The first
        // one has none to move: it is where the field the schedule handed over
        // becomes a 0/1 design in the first place, so nothing but the volume
        // target caps it.
        let cap = match (seeding, self.add_ratio > 0.0) {
            (true, _) => cells,
            (false, false) => 0,
            (false, true) => ((self.add_ratio * solid_count as f64).round() as usize)
                .max(constants::BESO_ADDITION_FLOOR_CELLS),
        };

        self.order.clear();
        self.order.extend_from_slice(design);
        let (protected, ranking) = (&self.protected, &self.ranking);
        self.order.par_sort_unstable_by(|&a, &b| {
            protected[b]
                .cmp(&protected[a])
                .then(ranking[b].total_cmp(&ranking[a]))
                // Cells of equal rank are cut in grid order, so the same design
                // and the same sensitivities always produce the same cut.
                .then(a.cmp(&b))
        });
        // Walk the ranking and keep cells until the volume is filled, taking
        // every cell that is already material and new ones only while the
        // addition budget lasts. Skipping a capped addition is what lowers the
        // removal threshold: the walk goes on into material the unrestricted cut
        // would have removed, and that material stays.
        for &e in design {
            self.keep[e] = false;
        }
        let (mut kept, mut added, mut threshold) = (0, 0, f64::INFINITY);
        for &e in &self.order {
            if kept == wanted {
                break;
            }
            if !solid_now(e) {
                if added == cap {
                    continue;
                }
                added += 1;
            }
            self.keep[e] = true;
            threshold = self.ranking[e];
            kept += 1;
        }

        x_new.copy_from_slice(x);
        let mut flips = 0;
        for &e in design {
            if self.keep[e] != solid_now(e) {
                flips += 1;
            }
            x_new[e] = if self.keep[e] { high } else { low };
        }
        constraint.chain.apply(x_new, filtered, printed);

        self.target = Some(target);
        // At the stage's target when the evolution has descended to it and the
        // cut actually filled it - which it cannot have done while the addition
        // cap is still walking the design up towards a refinement's target.
        self.at_target = target <= constraint.target_volume_fraction && kept == wanted;
        self.unmoved = flips == 0;
        Ok(Step {
            max_change: flips as f64 / cells.max(1) as f64,
            volume_fraction: design_mean(printed, design),
            lambda: threshold,
            shift: 0.0,
        })
    }

    /// Feed the iteration's compliance to the settling test and read the
    /// verdict.
    ///
    /// The stage has settled when its volume is at the stage's target and
    /// either the last cut flipped no cell at all - the ranking asking for the
    /// design it already has, which the next iteration would ask for again - or
    /// the compliance summed over the last [`constants::BESO_CONVERGENCE_WINDOW`]
    /// iterations is within `tolerance` of the sum over the window before it.
    /// That is the method's own test, and it is asked here rather than of
    /// [`super::stall`] because the design variable change of a 0/1 design is a
    /// count of flips: it does not decay towards a tolerance, and a window of it
    /// says nothing about a design that is oscillating between two cuts of the
    /// same volume.
    pub fn observe(&mut self, compliance: f64, tolerance: f64) -> Settling {
        let window = constants::BESO_CONVERGENCE_WINDOW;
        self.compliances.push_back(compliance);
        while self.compliances.len() > 2 * window {
            self.compliances.pop_front();
        }
        if !self.at_target {
            return Settling::Running;
        }
        if self.unmoved {
            return Settling::Unmoved;
        }
        if self.compliances.len() < 2 * window {
            return Settling::Running;
        }
        let recent: f64 = self.compliances.iter().skip(window).sum();
        let earlier: f64 = self.compliances.iter().take(window).sum();
        if recent == 0.0 {
            return Settling::Running;
        }
        let change = ((recent - earlier) / recent).abs();
        match change < tolerance {
            true => Settling::Settled(change),
            false => Settling::Running,
        }
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("beso::State")
            .field("evolution_rate", &self.evolution_rate)
            .field("add_ratio", &self.add_ratio)
            .field("protected_cells", &self.protected_cells)
            .field("target", &self.target)
            .field("at_target", &self.at_target)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    /// The tiny cantilever of the SIMP tests: 12 x 4 x 4 design cells, a support
    /// face at one end and a load face at the other.
    ///
    /// `mass_fraction` is the schema's business and not this module's - the
    /// cuts below start from the solid design the schedule's first stage hands
    /// over, whatever a file says.
    fn problem() -> Problem {
        let text = r#"
[project]
name = "beso"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0
max_iterations = 5

[output]
stl_path = "beso.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [24.0, 8.0, 8.0]

[[supports]]
region = { shape = "box", min = [0.0, 0.0, 0.0], max = [0.5, 8.0, 8.0] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [23.5, 0.0, 0.0], max = [24.5, 8.0, 8.0] }
vector = [0.0, 0.0, -20.0]
"#;
        let config = Config::parse(text).expect("parse");
        Problem::build(&config, &PathBuf::from(".")).expect("build")
    }

    /// Everything one cut needs: the problem, its chain, its design cells and
    /// the buffers a step writes into.
    struct Fixture {
        problem: Problem,
        chain: DensityChain,
        design: Vec<usize>,
        x: Vec<f64>,
        x_new: Vec<f64>,
        filtered: Vec<f64>,
        printed: Vec<f64>,
    }

    impl Fixture {
        fn new() -> Fixture {
            let problem = problem();
            let cells = problem.grid.n_cells();
            let chain = DensityChain::new(
                &problem.grid,
                problem.optimization.filter_radius_mm,
                problem.optimization.overhang,
            );
            let design: Vec<usize> = (0..cells)
                .filter(|&e| problem.grid.cells[e] == CellKind::Design)
                .collect();
            Fixture {
                problem,
                chain,
                design,
                x: vec![constants::DENSITY_MAX; cells],
                x_new: vec![0.0; cells],
                filtered: vec![0.0; cells],
                printed: vec![0.0; cells],
            }
        }

        fn state(&self, evolution_rate: f64, add_ratio: f64) -> State {
            State::new(&self.problem, &self.chain, evolution_rate, add_ratio)
        }

        /// One cut at `target`, with the design of the previous one as its
        /// starting point.
        fn cut(&mut self, state: &mut State, dc: &[f64], target: f64) -> Step {
            let step = state
                .update(
                    &self.x,
                    dc,
                    &Constraint {
                        design_cells: &self.design,
                        target_volume_fraction: target,
                        chain: &self.chain,
                        local: None,
                    },
                    Buffers {
                        x_new: &mut self.x_new,
                        filtered: &mut self.filtered,
                        printed: &mut self.printed,
                    },
                )
                .expect("a cut");
            self.x.copy_from_slice(&self.x_new);
            step
        }

        fn solid(&self) -> usize {
            self.design
                .iter()
                .filter(|&&e| self.x[e] == constants::DENSITY_MAX)
                .count()
        }
    }

    /// A ranking that puts the cells with the lowest index last, so a cut is
    /// predictable without a solve behind it.
    fn by_index(fixture: &Fixture) -> Vec<f64> {
        let mut dc = vec![0.0; fixture.problem.grid.n_cells()];
        for &e in &fixture.design {
            dc[e] = -(e as f64);
        }
        dc
    }

    /// The volume falls by `evolution_rate` of itself an iteration and stops at
    /// the stage's target rather than below it.
    #[test]
    fn the_volume_descends_at_the_evolution_rate_to_the_stage_target() {
        let mut fixture = Fixture::new();
        let mut state = fixture.state(0.1, 0.0);
        let dc = by_index(&fixture);
        let cells = fixture.design.len() as f64;
        let mut expected: f64 = 1.0;
        for _ in 0..12 {
            expected = (expected * (1.0 - 0.1)).max(0.5);
            fixture.cut(&mut state, &dc, 0.5);
            assert_eq!(
                fixture.solid(),
                (expected * cells).round() as usize,
                "the cut missed the iteration's volume target of {expected}"
            );
        }
        // Seven steps of 0.9 reach the target, and nothing goes below it.
        assert_eq!(fixture.solid(), (0.5 * cells).round() as usize);
    }

    /// No more than `add_ratio` of the current volume may come back in one
    /// iteration, however much of the ranking asks to.
    #[test]
    fn the_addition_cap_binds_when_the_ranking_reverses() {
        let mut fixture = Fixture::new();
        let mut state = fixture.state(0.5, 0.05);
        let dc = by_index(&fixture);
        // The first cut is the seeding one: half the design, ranked by index.
        fixture.cut(&mut state, &dc, 0.5);
        let before: Vec<bool> = (0..fixture.x.len())
            .map(|e| fixture.x[e] == constants::DENSITY_MAX)
            .collect();
        let solid = fixture.solid();
        let cap = ((0.05 * solid as f64).round() as usize).max(1);
        // Now rank the design the other way round: every cell the first cut
        // removed is now the most valuable one there is.
        let reversed: Vec<f64> = dc.iter().map(|value| -value).collect();
        fixture.cut(&mut state, &reversed, 0.5);
        let added = fixture
            .design
            .iter()
            .filter(|&&e| fixture.x[e] == constants::DENSITY_MAX && !before[e])
            .count();
        assert!(added > 0, "a reversed ranking added nothing at all");
        assert!(
            added <= cap,
            "{added} cells came back, past the cap of {cap}"
        );
        // The volume is still the target's: the walk went further down the
        // ranking for the material the cap would not let it replace.
        assert_eq!(fixture.solid(), solid);
    }

    /// The cells of the supports and the loads are never cut away, even when the
    /// ranking puts them last and the target is below what they alone fill.
    #[test]
    fn the_cells_of_a_support_or_a_load_survive_every_cut() {
        let mut fixture = Fixture::new();
        let mut state = fixture.state(0.5, 0.0);
        let protected: Vec<usize> = crate::trim::purposes(&fixture.problem)
            .iter()
            .flat_map(|purpose| purpose.cells.clone())
            .filter(|&e| fixture.problem.grid.cells[e] == CellKind::Design)
            .collect();
        assert!(!protected.is_empty(), "the fixture protects nothing");
        assert_eq!(state.protected_cells(), {
            let mut unique = protected.clone();
            unique.sort_unstable();
            unique.dedup();
            unique.len()
        });
        // The worst possible ranking: every protected cell is the least valuable
        // cell in the design, and the target is below what they fill.
        let mut dc = vec![0.0; fixture.problem.grid.n_cells()];
        for &e in &fixture.design {
            dc[e] = -1.0;
        }
        for &e in &protected {
            dc[e] = 0.0;
        }
        // Half the volume a cut, down to a target far below what the protected
        // cells alone fill.
        for _ in 0..8 {
            fixture.cut(&mut state, &dc, 0.01);
        }
        for &e in &protected {
            assert_eq!(
                fixture.x[e],
                constants::DENSITY_MAX,
                "cell {e} of a support or a load was cut away"
            );
        }
        // A target below the protected volume is the protected volume.
        assert_eq!(fixture.solid(), state.protected_cells());
    }

    /// A run of cuts on the sensitivities of real solves leaves a structure and
    /// not a checkerboard: no cell of it stands on its own with nothing but
    /// removed material touching it.
    ///
    /// The smoother is what buys that, and it is asked of it here rather than of
    /// a hand-written alternating field - a linear filter with a self weight
    /// damps such a field, it does not invert it, and no published sensitivity
    /// filter would either. What it has to do is keep the ranking of a *solved*
    /// design smooth enough that the cut follows the structure.
    #[test]
    fn cuts_on_solved_sensitivities_leave_no_cell_standing_alone() {
        use crate::engine::objective::{Objective, Workspace};
        use crate::fea::{CancelProbe, LinearSolver};

        let problem = problem();
        let grid = &problem.grid;
        let cells = grid.n_cells();
        let chain = DensityChain::new(
            grid,
            problem.optimization.filter_radius_mm,
            problem.optimization.overhang,
        );
        let design: Vec<usize> = (0..cells)
            .filter(|&e| grid.cells[e] == CellKind::Design)
            .collect();
        let mut state = State::new(&problem, &chain, 0.1, 0.02);
        let objective = Objective::new(&problem, &chain);
        let mut work = Workspace::new(&problem);
        let mut solver = LinearSolver::new(&problem).expect("a solver");
        let mut x = vec![constants::DENSITY_MAX; cells];
        let mut previous = x.clone();
        let mut x_new = vec![0.0; cells];
        let mut dc = vec![0.0; cells];
        let running = || false;
        let cancel = CancelProbe::watching(&running);

        for _ in 0..15 {
            objective
                .evaluate(&x, &mut work, &mut dc, &mut solver, cancel)
                .expect("an evaluation")
                .expect("a value");
            state
                .update(
                    &x,
                    &dc,
                    &Constraint {
                        design_cells: &design,
                        target_volume_fraction: 0.4,
                        chain: &chain,
                        local: None,
                    },
                    Buffers {
                        x_new: &mut x_new,
                        filtered: &mut work.filtered,
                        printed: &mut work.printed,
                    },
                )
                .expect("a cut");
            previous.copy_from_slice(&x);
            x.copy_from_slice(&x_new);
        }
        let kept = design
            .iter()
            .filter(|&&e| x[e] == constants::DENSITY_MAX)
            .count();
        assert_eq!(kept, (0.4 * design.len() as f64).round() as usize);

        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let solid =
            |i: usize, j: usize, k: usize| x[i + nx * (j + ny * k)] == constants::DENSITY_MAX;
        let mut alone = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if !solid(i, j, k) {
                        continue;
                    }
                    let touching = [
                        i > 0 && solid(i - 1, j, k),
                        i + 1 < nx && solid(i + 1, j, k),
                        j > 0 && solid(i, j - 1, k),
                        j + 1 < ny && solid(i, j + 1, k),
                        k > 0 && solid(i, j, k - 1),
                        k + 1 < nz && solid(i, j, k + 1),
                    ];
                    if !touching.iter().any(|joined| *joined) {
                        alone.push((i, j, k));
                    }
                }
            }
        }
        // A checkerboard is half the design standing alone. What a smoothed
        // ranking leaves is a stray of it at worst - a cell one cut has just
        // added to find out whether material belongs there, or one the same cut
        // stranded and the next one takes - so the claim is a share far under
        // anything a checkerboard could be and far over what a cut strands.
        assert!(
            20 * alone.len() <= kept,
            "{} of the {kept} cells kept stand alone: {alone:?}",
            alone.len()
        );
        assert_ne!(x, previous, "the last cut moved nothing at all");
    }

    /// The settling test waits for the volume to be at the stage's target, and
    /// then for a cut that flips nothing - or, while the cuts keep moving cells,
    /// for both halves of its window.
    #[test]
    fn the_settling_test_needs_the_target_and_then_a_design_that_stops_moving() {
        let mut fixture = Fixture::new();
        let dc = by_index(&fixture);
        // Still descending - the slowest rate there is against a distant target
        // - so a compliance that does not move at all is still not a settled
        // stage.
        let mut descending = fixture.state(0.005, 0.0);
        for _ in 0..2 * constants::BESO_CONVERGENCE_WINDOW {
            fixture.cut(&mut descending, &dc, 0.5);
            assert_eq!(descending.observe(10.0, 1e-3), Settling::Running);
        }

        // At the target from the first cut, and over the moment a cut asks for
        // the design the one before it made: the iteration after would be that
        // one again, whatever a window of compliances has left to say.
        let mut fixture = Fixture::new();
        let mut arrived = fixture.state(0.5, 0.0);
        fixture.cut(&mut arrived, &dc, 0.5);
        assert_eq!(arrived.observe(10.0, 1e-3), Settling::Running);
        fixture.cut(&mut arrived, &dc, 0.5);
        assert_eq!(arrived.observe(10.0, 1e-3), Settling::Unmoved);

        // A design at the target that keeps flipping cells - the ranking here
        // reverses every iteration - is not still, and that one waits for the
        // window to fill.
        let mut fixture = Fixture::new();
        let mut oscillating = fixture.state(0.5, 1.0);
        let reversed: Vec<f64> = dc.iter().map(|value| -value).collect();
        let mut settled = None;
        for iteration in 1..=4 * constants::BESO_CONVERGENCE_WINDOW {
            let ranking = match iteration % 2 {
                0 => &reversed,
                _ => &dc,
            };
            fixture.cut(&mut oscillating, ranking, 0.5);
            match oscillating.observe(10.0, 1e-3) {
                Settling::Running => {}
                Settling::Unmoved => panic!("the cut of iteration {iteration} flipped nothing"),
                Settling::Settled(change) => {
                    settled = Some((iteration, change));
                    break;
                }
            }
        }
        let (iteration, change) = settled.expect("a settled stage");
        assert_eq!(iteration, 2 * constants::BESO_CONVERGENCE_WINDOW);
        assert!(change < 1e-3, "{change}");
    }
}
