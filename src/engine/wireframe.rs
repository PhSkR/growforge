//! The guide wireframe: a thin connected network of the places a design has to
//! join, seeded into the initial density field and held under it for a while.
//!
//! `[optimization.wireframe]` switches it on. At setup the engine routes a
//! shortest keepout-avoiding path between every load region, every support region
//! and every `[[keepin]]` entry until all of them hang off one network, rasterizes
//! that network as a thin wire, seeds the wire into the initial design variables
//! and applies it as a **density floor** after each update step for the first
//! `hold_iterations` iterations. Then it lets go.
//!
//! **The wire is a guide and never a constraint.** Once the floor is released the
//! optimizer may keep it, thicken it, move it or dissolve it entirely, and nothing
//! downstream knows it was ever there: no cell is pinned and no sensitivity is
//! touched. What the guide buys is the *first* few iterations - a design that
//! starts with a load path already joining the regions the user declared, instead
//! of a uniform field the analysis has to find one in.
//!
//! What holds that up is where the run's verdicts are asked. A design under the
//! floor is not free, so [`crate::engine::simp`] asks neither the convergence test
//! nor the stall criterion while the floor is on: **a run cannot settle inside the
//! hold window**, and only the iteration cap and a cancellation can end one there.
//! Both of those are reported as what they are - a design that carries the wire as
//! forced material - so the one way to export a forced wire is to be told that is
//! what you are doing. A run that reaches the release settles the wire itself.
//!
//! ## What it is made of
//!
//! Everything geometric here is the growth engine's, used from the outside:
//! [`GrowthDomain`] for what a wire may pass through (every cell the classifier
//! did not mark [`CellKind::Void`], so keepouts and the space outside the domain
//! are impassable), [`astar::shortest_path`] for the routing - multi-source and
//! multi-target, with its tie-breaking already deterministic - and
//! [`skeleton`]/[`rasterize`] for turning the voxel paths into capsules and the
//! capsules into densities. Nothing about the growth heuristic itself is involved:
//! there is no canopy, no thickening and no attraction points.
//!
//! ## Terminals
//!
//! One terminal per support region, one per `[[keepin]]` **entry** and one per
//! load region, in that order, each in configuration order. Keepin entries are
//! taken apart rather than as the merged union the classifier uses (see
//! [`crate::problem::KeepinSummary`]): two disjoint pads are two places to reach,
//! and wiring their union would leave one of them unwired. A gravity load selects
//! no region and is skipped, exactly as the growth engine skips it. A terminal
//! that resolves to no material cell, or that no path can reach, is named in a
//! warning and left out; the rest of the network is built regardless.
//!
//! ## What the hold costs
//!
//! The floor is applied *after* the update step, which has already bisected its
//! multiplier to land on `mass_fraction`, so **during the hold the realized volume
//! fraction runs above `mass_fraction`** by whatever the wire adds. That is
//! reported honestly rather than hidden - the step the reporter sees is restated on
//! the floored design - and it self-corrects in the first iteration after the
//! release, when the update is once again the last word on the design. A run that
//! never gets there (its cap inside the hold window, or a cancellation) keeps the
//! excess, and says so.
//!
//! An `[optimization.local_volume]` cap is overshot the same way, for the same
//! reason and with the same remedy: the floor lands on top of a step that had
//! already priced it, so a neighbourhood along the wire can sit above
//! `max_fraction` while the hold is on. It is the volume overshoot above in a
//! second currency and takes no special code - see
//! [`crate::engine::local_volume`].
//!
//! ## Two properties worth knowing
//!
//! * **The wire is the cells whose centre lies inside it.** At a radius below
//!   half a voxel diagonal the seed can therefore come out dotted rather than
//!   continuous where a path runs diagonally. It is a guide, and the density
//!   filter smears a dotted seed back into a continuous one, so this costs
//!   sharpness rather than correctness; a radius of about a voxel or more avoids
//!   it entirely.
//! * **There is no viewer overlay, and none is needed.** The seeded wire is part
//!   of the density field the engine publishes every iteration, so the live
//!   surface in the viewer shows it, thickening or dissolving, as it happens.

use rayon::prelude::*;

use crate::config::WireframeParams;
use crate::constants;
use crate::engine::growth::domain::GrowthDomain;
use crate::engine::growth::skeleton::Skeleton;
use crate::engine::growth::{astar, rasterize, skeleton};
use crate::engine::update::{Buffers, Constraint, Step, design_mean};
use crate::geometry::Vec3;
use crate::grid::CellKind;
use crate::problem::Problem;
use crate::report::Reporter;

/// One place the wire has to reach: what to call it, and the cells that stand
/// for it.
#[derive(Debug, Clone)]
struct Terminal {
    /// Name used in the warnings about this terminal, spelled as the rest of the
    /// crate spells it.
    label: String,
    /// Material cells the region covers, ascending and deduplicated.
    cells: Vec<usize>,
}

/// A built guide wireframe: the design cells the wire covers and the controls it
/// was built with.
#[derive(Debug, Clone)]
pub struct Wireframe {
    /// Design cells the rasterized wire covers, ascending. Never empty: a wire
    /// that covers no design cell is no guide, and [`Wireframe::build`] returns
    /// `None` for it.
    cells: Vec<usize>,
    /// Paths routed between terminals.
    paths: usize,
    /// Terminals that resolved to at least one cell.
    terminals: usize,
    /// The resolved controls.
    params: WireframeParams,
}

impl Wireframe {
    /// Build the guide for `problem`, or `None` when there is none to build.
    ///
    /// `None` covers four cases, three of which are reported: the feature is off
    /// (`[optimization.wireframe]` absent, and nothing is reported for a feature
    /// nobody asked for), fewer than two terminals resolved, every terminal
    /// already touched another one so no path was needed, and a wire that covers
    /// no design cell.
    pub fn build(problem: &Problem, reporter: &dyn Reporter) -> Option<Wireframe> {
        let params = problem.optimization.wireframe?;
        let grid = &problem.grid;
        let domain = GrowthDomain::new(grid);
        let terminals = terminals(problem, &domain, reporter);
        if terminals.len() < 2 {
            reporter.note(&format!(
                "warning: [optimization.wireframe]: {} of this problem's regions resolved to \
                 material cells, so there is nothing to wire together; the guide is off for this \
                 run",
                terminals.len()
            ));
            return None;
        }

        // The network in construction: the cells it holds, and a mask of them so
        // that "is this terminal already on it" is a lookup rather than a scan.
        // Every terminal that joins contributes its whole cell set, because a
        // region is one place - the same view the growth engine takes of a
        // support patch or a load pad.
        let mut member = vec![false; grid.n_cells()];
        let mut network: Vec<usize> = Vec::new();
        extend(&mut network, &mut member, &terminals[0].cells);
        let mut wire = Skeleton::new();
        let mut paths = 0;
        for terminal in &terminals[1..] {
            if terminal.cells.iter().any(|&cell| member[cell]) {
                continue;
            }
            let Some(path) = astar::shortest_path(&domain, &network, &terminal.cells) else {
                reporter.note(&format!(
                    "warning: [optimization.wireframe]: no path from the guide network to {} \
                     avoids the keepouts, so it is left unwired and the rest of the network is \
                     built without it (keepouts and the space outside the domain are impassable)",
                    terminal.label
                ));
                continue;
            };
            wire.add_chain(&polyline(&domain, &path, params.radius_mm));
            extend(&mut network, &mut member, &path);
            extend(&mut network, &mut member, &terminal.cells);
            paths += 1;
        }
        if wire.is_empty() {
            reporter.note(
                "warning: [optimization.wireframe]: every region of this problem already touches \
                 another one, so no wire was needed; the guide is off for this run",
            );
            return None;
        }

        // One rasterization of the whole network, so the wire is one field rather
        // than a union of per-path ones, and the classification is applied for
        // free: void cells come back empty and forced solid ones full.
        let radii = vec![params.radius_mm; wire.len()];
        let densities = rasterize::densities(grid, &wire, &radii);
        let cells: Vec<usize> = (0..grid.n_cells())
            .filter(|&cell| {
                grid.cells[cell] == CellKind::Design
                    && densities[cell] > constants::WIREFRAME_CELL_THRESHOLD
            })
            .collect();
        if cells.is_empty() {
            reporter.note(
                "warning: [optimization.wireframe]: the wire covers no design cell - every cell it \
                 runs through is already forced solid - so there is nothing to seed; the guide is \
                 off for this run",
            );
            return None;
        }

        reporter.note(&format!(
            "wireframe: {paths} path{} joining {} regions, {} design cells seeded at density {:.3} \
             and held as a floor for {} iterations",
            if paths == 1 { "" } else { "s" },
            terminals.len(),
            cells.len(),
            params.seed_density,
            params.hold_iterations
        ));
        if params.hold_iterations >= problem.optimization.max_iterations {
            reporter.note(&format!(
                "warning: [optimization.wireframe]: hold_iterations {} is not below \
                 max_iterations {}, so the floor is never released and the wire is forced into the \
                 design this run exports; lower hold_iterations, or raise max_iterations",
                params.hold_iterations, problem.optimization.max_iterations
            ));
        }
        Some(Wireframe {
            cells,
            paths,
            terminals: terminals.len(),
            params,
        })
    }

    /// The design cells the wire covers, ascending.
    pub fn cells(&self) -> &[usize] {
        &self.cells
    }

    /// Paths routed between terminals, one per terminal that was not already on
    /// the network.
    pub fn paths(&self) -> usize {
        self.paths
    }

    /// Terminals that resolved to material cells.
    pub fn terminals(&self) -> usize {
        self.terminals
    }

    /// Density the wire is seeded and held at.
    pub fn seed_density(&self) -> f64 {
        self.params.seed_density
    }

    /// Iterations the floor is held for.
    pub fn hold_iterations(&self) -> usize {
        self.params.hold_iterations
    }

    /// True while `iteration` is still inside the hold window.
    pub fn holds(&self, iteration: usize) -> bool {
        iteration <= self.params.hold_iterations
    }

    /// Raise the wire's design cells to the seed density in `x`.
    ///
    /// Upwards only, and only on the design cells the wire covers: a cell the
    /// caller already had above the seed keeps its own value, and no void or
    /// forced solid cell is a wire cell in the first place.
    pub fn seed(&self, x: &mut [f64]) {
        for &cell in &self.cells {
            x[cell] = x[cell].max(self.params.seed_density);
        }
    }

    /// Apply the floor to the trial design in `buffers`, and restate the `step`
    /// that produced it.
    ///
    /// A no-op once the hold window has passed, which is announced the first time
    /// it is. Inside the window the floor is applied to `x_new` and the step is
    /// restated on the floored design: the largest change against `x` and the
    /// volume fraction of the printed densities the floored design maps to. Those
    /// are what the caller reports, and the floored design is the one the next
    /// iteration analyses, so restating them is the difference between reporting
    /// the run and reporting the proposal. What the caller must *not* do with a
    /// restated step is read a verdict out of it - a change held down by the floor
    /// is not a design settling - which is why [`crate::engine::simp`] asks
    /// neither the convergence test nor the stall criterion while [`Wireframe::holds`]
    /// is true.
    pub fn hold(
        &self,
        iteration: usize,
        x: &[f64],
        step: Step,
        constraint: &Constraint<'_>,
        buffers: Buffers<'_>,
        reporter: &dyn Reporter,
    ) -> Step {
        if !self.holds(iteration) {
            if self.params.hold_iterations > 0 && iteration == self.params.hold_iterations + 1 {
                reporter.note(&format!(
                    "wireframe: the density floor was released after {} iterations; the guide is \
                     the optimizer's to keep, thicken or dissolve from here",
                    self.params.hold_iterations
                ));
            }
            return step;
        }
        let Buffers {
            x_new,
            filtered,
            printed,
        } = buffers;
        let mut raised = false;
        for &cell in &self.cells {
            if x_new[cell] < self.params.seed_density {
                x_new[cell] = self.params.seed_density;
                raised = true;
            }
        }
        // Nothing was below the floor, so the step already describes the design
        // that is there - and `printed` already belongs to it.
        if !raised {
            return step;
        }
        // The volume constraint is stated on the printed densities, so the
        // floored design has to go back through the chain before its volume
        // fraction means anything.
        constraint.chain.apply(x_new, filtered, printed);
        Step {
            max_change: constraint
                .design_cells
                .par_iter()
                .map(|&e| (x_new[e] - x[e]).abs())
                .reduce(|| 0.0, f64::max),
            volume_fraction: design_mean(printed, constraint.design_cells),
            ..step
        }
    }
}

/// Add `cells` to the network, keeping its membership mask in step.
fn extend(network: &mut Vec<usize>, member: &mut [bool], cells: &[usize]) {
    for &cell in cells {
        if !member[cell] {
            member[cell] = true;
            network.push(cell);
        }
    }
}

/// The polyline one voxel path becomes: cell centres, shortcut into as few
/// segments as the domain allows, then resampled.
///
/// The shortcut radius is backed off until the unsimplified path itself clears it
/// (see [`skeleton::shortcut_radius`]), so a wire thicker than the corridor it
/// runs through still simplifies and still cannot leave the allowed cells - the
/// same rule the growth engine's backbones follow.
fn polyline(domain: &GrowthDomain, path: &[usize], radius_mm: f64) -> Vec<Vec3> {
    let points: Vec<Vec3> = path.iter().map(|&cell| domain.cell_center(cell)).collect();
    let clearance = skeleton::shortcut_radius(domain, &points, radius_mm);
    let simplified = skeleton::shortcut(domain, &points, clearance);
    skeleton::resample(&simplified, constants::WIREFRAME_RESAMPLE_RADII * radius_mm)
}

/// Every place the wire has to reach, in the order it is wired: the support
/// regions, the keepin entries, then the load regions, each in configuration
/// order.
///
/// The order is fixed rather than chosen so that the network - and therefore the
/// seed - is a function of the configuration alone. A terminal that selected no
/// material cell is named and dropped: it cannot be reached, and a silent drop
/// would leave the user with a guide that quietly misses the region they drew.
fn terminals(problem: &Problem, domain: &GrowthDomain, reporter: &dyn Reporter) -> Vec<Terminal> {
    let mut terminals = Vec::new();
    let mut add = |label: String, cells: Vec<usize>| {
        if cells.is_empty() {
            reporter.note(&format!(
                "warning: [optimization.wireframe]: {label} holds no material cell, so the guide \
                 skips it; a region a keepout swallowed is void, and keepout wins"
            ));
            return;
        }
        terminals.push(Terminal { label, cells });
    };

    for support in &problem.supports {
        add(
            format!("support {}", support.index),
            domain.cells_touching_nodes(&support.nodes),
        );
    }
    for keepin in &problem.keepins {
        // Keepin cells are forced solid, which the domain treats as passable, so
        // a path may start or end on one.
        add(format!("keepin {}", keepin.index), keepin.cells.clone());
    }
    for case in &problem.load_cases {
        for (index, load) in case.loads.iter().enumerate() {
            if load.nodes.is_empty() {
                // Gravity: no region of its own, it acts on everything.
                continue;
            }
            add(
                format!("load {} of case \"{}\"", index + 1, case.name),
                domain.cells_touching_nodes(&load.nodes),
            );
        }
    }
    terminals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::engine::chain::DensityChain;
    use crate::report::{IterationStats, SilentReporter};
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// A reporter that keeps its notes, so a test can read the warnings.
    #[derive(Default)]
    struct Notes(Mutex<Vec<String>>);

    impl Reporter for Notes {
        fn iteration(&self, _stats: &IterationStats) {}
        fn note(&self, message: &str) {
            self.0.lock().expect("notes").push(message.to_string());
        }
    }

    impl Notes {
        fn taken(self) -> Vec<String> {
            self.0.into_inner().expect("notes")
        }
    }

    /// A bar with a keepout wall across it, between the support at one end and
    /// the load high up at the other. With `gap` the wall stops short of the top,
    /// leaving three layers of cells for a wire to pass through; without it the
    /// wall is the full height and the far half of the bar is unreachable.
    ///
    /// 40 x 10 x 20 cells of 2 mm, which is 8000 cells: small enough for a unit
    /// test, fine enough that the default wire radius is wider than half a voxel
    /// diagonal and therefore covers every cell its paths run through.
    ///
    /// The iteration budget is longer than the default hold so that the fixture's
    /// own configuration is one the guide has nothing to warn about; the test that
    /// is about an outlasting hold pins one.
    fn walled(gap: bool, wireframe: &str, extra: &str) -> String {
        format!(
            r#"
[project]
name = "walled"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.3
min_feature_mm = 12.0
max_iterations = 50

[optimization.wireframe]
{wireframe}

[output]
stl_path = "walled.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [80.0, 20.0, 40.0]

[[keepout]]
shape = "box"
min = [36.0, -1.0, -1.0]
max = [44.0, 21.0, {wall_top:?}]

[[supports]]
region = {{ shape = "box", min = [-1.0, -1.0, -1.0], max = [1.0, 21.0, 41.0] }}

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = {{ shape = "box", min = [79.0, -1.0, 35.0], max = [81.0, 21.0, 41.0] }}
vector = [0.0, 0.0, -200.0]
{extra}
"#,
            wall_top = if gap { 30.0 } else { 41.0 }
        )
    }

    /// Two keepin pads on opposite sides of the wall, so the second one can only
    /// be reached through the gap.
    const TWO_PADS: &str = r#"
[[keepin]]
shape = "box"
min = [8.0, 6.0, 6.0]
max = [16.0, 14.0, 14.0]

[[keepin]]
shape = "box"
min = [60.0, 6.0, 6.0]
max = [68.0, 14.0, 14.0]
"#;

    fn build(text: &str) -> Problem {
        let config = Config::parse(text).expect("parse");
        Problem::build(&config, &PathBuf::from(".")).expect("build")
    }

    /// Cells of the wire together with every terminal cell: what has to come out
    /// as one connected body if the guide joined anything.
    fn network_of(problem: &Problem, wire: &Wireframe) -> Vec<usize> {
        let domain = GrowthDomain::new(&problem.grid);
        let mut cells = wire.cells().to_vec();
        for terminal in terminals(problem, &domain, &SilentReporter) {
            cells.extend(terminal.cells);
        }
        cells.sort_unstable();
        cells.dedup();
        cells
    }

    /// Number of 26-connected components of `cells`.
    fn components(problem: &Problem, cells: &[usize]) -> usize {
        let grid = &problem.grid;
        let mut member = vec![false; grid.n_cells()];
        for &cell in cells {
            member[cell] = true;
        }
        let mut seen = vec![false; grid.n_cells()];
        let mut components = 0;
        for &start in cells {
            if seen[start] {
                continue;
            }
            components += 1;
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(cell) = stack.pop() {
                let (i, j, k) = grid.cell_ijk(cell);
                for dk in -1i64..=1 {
                    for dj in -1i64..=1 {
                        for di in -1i64..=1 {
                            let at = [i as i64 + di, j as i64 + dj, k as i64 + dk];
                            let counts = [grid.nx as i64, grid.ny as i64, grid.nz as i64];
                            if (0..3).any(|d| at[d] < 0 || at[d] >= counts[d]) {
                                continue;
                            }
                            let next =
                                grid.cell_index(at[0] as usize, at[1] as usize, at[2] as usize);
                            if member[next] && !seen[next] {
                                seen[next] = true;
                                stack.push(next);
                            }
                        }
                    }
                }
            }
        }
        components
    }

    /// True when some cell of `cells` is a 26-neighbour of a wire cell, or is one
    /// itself: a wire really arrives here.
    fn wire_reaches(problem: &Problem, wire: &Wireframe, cells: &[usize]) -> bool {
        let grid = &problem.grid;
        let mut on_wire = vec![false; grid.n_cells()];
        for &cell in wire.cells() {
            on_wire[cell] = true;
        }
        cells.iter().any(|&cell| {
            let (i, j, k) = grid.cell_ijk(cell);
            (-1i64..=1).any(|dk| {
                (-1i64..=1).any(|dj| {
                    (-1i64..=1).any(|di| {
                        let at = [i as i64 + di, j as i64 + dj, k as i64 + dk];
                        let counts = [grid.nx as i64, grid.ny as i64, grid.nz as i64];
                        if (0..3).any(|d| at[d] < 0 || at[d] >= counts[d]) {
                            return false;
                        }
                        on_wire[grid.cell_index(at[0] as usize, at[1] as usize, at[2] as usize)]
                    })
                })
            })
        })
    }

    /// The whole point of the guide: one connected network through the wall's gap,
    /// joining the regions the configuration declared, never entering a keepout.
    #[test]
    fn the_wire_routes_around_a_wall_and_joins_every_region() {
        let problem = build(&walled(true, "", ""));
        let notes = Notes::default();
        let wire = Wireframe::build(&problem, &notes).expect("a guide");
        let notes = notes.taken();
        assert!(!wire.cells().is_empty());
        assert_eq!(wire.terminals(), 2, "a support and a load");
        assert_eq!(wire.paths(), 1);
        assert!(
            notes.iter().any(|n| n.contains("wireframe: 1 path")),
            "the run has to say what it built: {notes:?}"
        );
        assert!(
            !notes.iter().any(|n| n.contains("warning")),
            "nothing about this problem is a warning: {notes:?}"
        );

        // Every wire cell is a design cell of the grid, so no keepout cell and no
        // cell outside the domain is on it.
        let grid = &problem.grid;
        for &cell in wire.cells() {
            assert_eq!(
                grid.cells[cell],
                CellKind::Design,
                "the wire claimed a cell that is not a design variable"
            );
        }
        // The wall is cells i = 18 .. 21; the gap is k >= 15. The wire has to be
        // in the gap, and cannot be anywhere else in the wall's columns.
        let through: Vec<(usize, usize, usize)> = wire
            .cells()
            .iter()
            .map(|&cell| grid.cell_ijk(cell))
            .filter(|(i, _, _)| (18..=21).contains(i))
            .collect();
        assert!(
            !through.is_empty(),
            "the wire never crossed the wall's columns"
        );
        assert!(
            through.iter().all(|(_, _, k)| *k >= 15),
            "the wire crossed the wall below its gap: {through:?}"
        );

        // And it is one body together with the regions it joins.
        let network = network_of(&problem, &wire);
        assert_eq!(
            components(&problem, &network),
            1,
            "the guide left the regions in more than one piece"
        );
    }

    /// The same configuration built twice is the same wire, cell for cell: the
    /// seed a run starts from is a function of the configuration alone.
    #[test]
    fn building_the_same_guide_twice_gives_the_same_cells() {
        let text = walled(true, "", TWO_PADS);
        let first = build(&text);
        let second = build(&text);
        let one = Wireframe::build(&first, &SilentReporter).expect("a guide");
        let two = Wireframe::build(&second, &SilentReporter).expect("a guide");
        assert_eq!(one.cells(), two.cells());
        assert_eq!(one.paths(), two.paths());
        assert_eq!(one.terminals(), two.terminals());
    }

    /// The reason keepin entries are resolved one at a time: two disjoint pads
    /// are two places, and both of them get a wire. Wiring their merged union
    /// would reach whichever one the search settled first and leave the other
    /// alone.
    #[test]
    fn every_keepin_entry_gets_its_own_wire() {
        let problem = build(&walled(true, "", TWO_PADS));
        assert_eq!(problem.keepins.len(), 2, "the fixture needs two entries");
        let wire = Wireframe::build(&problem, &SilentReporter).expect("a guide");
        assert_eq!(wire.terminals(), 4, "a support, two pads and a load");
        assert_eq!(wire.paths(), 3);
        for keepin in &problem.keepins {
            assert!(!keepin.cells.is_empty());
            assert!(
                wire_reaches(&problem, &wire, &keepin.cells),
                "no wire arrives at keepin {}",
                keepin.index
            );
        }
        // Including the pad on the far side of the wall, which is only reachable
        // through the gap, so the whole thing is still one body.
        assert_eq!(components(&problem, &network_of(&problem, &wire)), 1);
    }

    /// A region with no material in it is named and skipped too. A keepin entry
    /// inside a keepout is the case that reaches this: keepout wins, so the cells
    /// the entry asked for are void and there is nothing to reach.
    #[test]
    fn a_region_a_keepout_swallowed_warns_and_the_rest_still_wire_up() {
        let swallowed = r#"
[[keepin]]
shape = "box"
min = [37.0, 8.0, 8.0]
max = [43.0, 12.0, 12.0]
"#;
        let problem = build(&walled(true, "", swallowed));
        assert_eq!(problem.keepins.len(), 1);
        assert!(
            problem.keepins[0].cells.is_empty(),
            "the fixture has to bury its pad in the wall"
        );
        let notes = Notes::default();
        let wire = Wireframe::build(&problem, &notes).expect("a guide");
        let notes = notes.taken();
        assert!(
            notes
                .iter()
                .any(|n| n.contains("warning") && n.contains("keepin 1")),
            "the buried pad had to be named: {notes:?}"
        );
        // The support and the load are still wired to each other.
        assert_eq!(wire.terminals(), 2);
        assert_eq!(wire.paths(), 1);
        assert_eq!(components(&problem, &network_of(&problem, &wire)), 1);
    }

    /// A terminal nothing can reach is named and skipped, and everything that can
    /// be wired still is.
    #[test]
    fn an_unreachable_terminal_warns_and_the_rest_still_wire_up() {
        let problem = build(&walled(false, "", TWO_PADS));
        let notes = Notes::default();
        let wire = Wireframe::build(&problem, &notes).expect("a guide");
        let notes = notes.taken();

        // The far pad and the load are both behind a wall with no gap.
        for unreachable in ["keepin 2", "load 1 of case \"tip\""] {
            assert!(
                notes
                    .iter()
                    .any(|n| n.contains("warning") && n.contains(unreachable)),
                "{unreachable} had to be reported unreachable: {notes:?}"
            );
        }
        // And the near pad is wired anyway.
        assert_eq!(wire.paths(), 1);
        assert!(!wire.cells().is_empty());
        assert!(wire_reaches(&problem, &wire, &problem.keepins[0].cells));
        assert!(
            wire.cells()
                .iter()
                .all(|&cell| problem.grid.cell_ijk(cell).0 < 18),
            "a wire crossed a solid wall"
        );
    }

    /// The seed raises the wire's design cells and touches nothing else, ever
    /// upwards: a void cell stays void, a forced solid cell stays solid, and a
    /// design cell already above the seed keeps what it had.
    #[test]
    fn the_seed_raises_wire_design_cells_and_only_upwards() {
        let problem = build(&walled(true, "", TWO_PADS));
        let wire = Wireframe::build(&problem, &SilentReporter).expect("a guide");
        let grid = &problem.grid;

        // The field the SIMP engine starts from, with one wire cell already above
        // the seed density so that "only upwards" can be observed.
        let mut x: Vec<f64> = grid
            .cells
            .iter()
            .map(|kind| match kind {
                CellKind::Void => constants::DENSITY_MIN,
                CellKind::Solid => constants::DENSITY_MAX,
                CellKind::Design => problem.optimization.mass_fraction,
            })
            .collect();
        let above = wire.cells()[0];
        let untouched = constants::DENSITY_MAX;
        x[above] = untouched;
        let before = x.clone();
        wire.seed(&mut x);

        assert!(wire.seed_density() > problem.optimization.mass_fraction);
        for (cell, (new, old)) in x.iter().zip(before.iter()).enumerate() {
            let expected = if cell == above {
                untouched
            } else if wire.cells().contains(&cell) {
                wire.seed_density()
            } else {
                *old
            };
            assert_eq!(*new, expected, "cell {cell}");
            assert!(*new >= *old, "cell {cell} went down");
        }
        // Nothing outside the design cells moved at all.
        for (cell, kind) in grid.cells.iter().enumerate() {
            if *kind != CellKind::Design {
                assert_eq!(x[cell], before[cell], "cell {cell} is {kind:?}");
            }
        }
    }

    /// The floor inside the window, released after it - and the step restated on
    /// the floored design rather than on the one the update proposed.
    #[test]
    fn the_floor_is_held_then_released_and_the_step_it_reports_is_the_floored_one() {
        let problem = build(&walled(true, "hold_iterations = 2", TWO_PADS));
        let wire = Wireframe::build(&problem, &SilentReporter).expect("a guide");
        assert_eq!(wire.hold_iterations(), 2);
        let grid = &problem.grid;
        let n = grid.n_cells();
        let design_cells: Vec<usize> = (0..n)
            .filter(|&cell| grid.cells[cell] == CellKind::Design)
            .collect();
        let chain = DensityChain::new(grid, problem.optimization.filter_radius_mm, None);
        let constraint = Constraint {
            design_cells: &design_cells,
            target_volume_fraction: problem.optimization.mass_fraction,
            chain: &chain,
            local: None,
        };

        // A design at the target everywhere, and a step that moved nothing: what
        // the floor does to it is all that is left in the numbers.
        let x: Vec<f64> = grid
            .cells
            .iter()
            .map(|kind| match kind {
                CellKind::Void => constants::DENSITY_MIN,
                CellKind::Solid => constants::DENSITY_MAX,
                CellKind::Design => problem.optimization.mass_fraction,
            })
            .collect();
        let mut x_new = x.clone();
        let mut filtered = vec![0.0; n];
        let mut printed = vec![0.0; n];
        chain.apply(&x_new, &mut filtered, &mut printed);
        let proposed = Step {
            max_change: 0.0,
            volume_fraction: design_mean(&printed, &design_cells),
            lambda: 1.25,
            shift: 0.5,
        };

        // What the floored design is, worked out here rather than asked for.
        let mut floored = x.clone();
        for &cell in wire.cells() {
            floored[cell] = floored[cell].max(wire.seed_density());
        }
        let mut expected_filtered = vec![0.0; n];
        let mut expected_printed = vec![0.0; n];
        chain.apply(&floored, &mut expected_filtered, &mut expected_printed);
        let expected_volume = design_mean(&expected_printed, &design_cells);
        let expected_change = design_cells
            .iter()
            .map(|&e| (floored[e] - x[e]).abs())
            .fold(0.0, f64::max);

        let notes = Notes::default();
        for iteration in 1..=wire.hold_iterations() {
            x_new.copy_from_slice(&x);
            assert!(wire.holds(iteration));
            let held = wire.hold(
                iteration,
                &x,
                proposed,
                &constraint,
                Buffers {
                    x_new: &mut x_new,
                    filtered: &mut filtered,
                    printed: &mut printed,
                },
                &notes,
            );
            assert_eq!(x_new, floored, "iteration {iteration} was not floored");
            assert!(
                (held.volume_fraction - expected_volume).abs() < 1e-12,
                "iteration {iteration}: volume fraction {} is not the floored design's {}",
                held.volume_fraction,
                expected_volume
            );
            assert!(
                (held.max_change - expected_change).abs() < 1e-12,
                "iteration {iteration}: change {} is not the floored design's {}",
                held.max_change,
                expected_change
            );
            // The floor added material, so the volume ran above what the update
            // step had settled on - which is the documented cost of the hold.
            assert!(held.volume_fraction > proposed.volume_fraction);
            assert!(held.max_change > proposed.max_change);
            // And the multiplier and the shift are the update's own, untouched.
            assert_eq!(held.lambda, proposed.lambda);
            assert_eq!(held.shift, proposed.shift);
            // The printed densities are the floored design's too, which is what
            // the caller publishes as the live field.
            assert_eq!(printed, expected_printed);
        }

        // One past the window: nothing is floored, the step is handed back as it
        // came, and the release is announced once.
        for iteration in [wire.hold_iterations() + 1, wire.hold_iterations() + 2] {
            x_new.copy_from_slice(&x);
            assert!(!wire.holds(iteration));
            let released = wire.hold(
                iteration,
                &x,
                proposed,
                &constraint,
                Buffers {
                    x_new: &mut x_new,
                    filtered: &mut filtered,
                    printed: &mut printed,
                },
                &notes,
            );
            assert_eq!(x_new, x, "iteration {iteration} was still floored");
            assert_eq!(released.volume_fraction, proposed.volume_fraction);
            assert_eq!(released.max_change, proposed.max_change);
        }
        let notes = notes.taken();
        assert_eq!(
            notes.iter().filter(|n| n.contains("released")).count(),
            1,
            "the release is announced exactly once: {notes:?}"
        );

        // A trial design already above the floor is left exactly as it is, and
        // its step with it.
        x_new.copy_from_slice(&floored);
        let untouched = wire.hold(
            1,
            &x,
            proposed,
            &constraint,
            Buffers {
                x_new: &mut x_new,
                filtered: &mut filtered,
                printed: &mut printed,
            },
            &SilentReporter,
        );
        assert_eq!(x_new, floored);
        assert_eq!(untouched.volume_fraction, proposed.volume_fraction);
        assert_eq!(untouched.max_change, proposed.max_change);
    }

    /// A hold that outlasts the run is a wire forced into the exported part,
    /// which is the one thing the guide promises not to be, so it is reported.
    #[test]
    fn a_hold_that_outlasts_the_run_is_reported() {
        let budget = 50;
        let problem = build(&walled(true, &format!("hold_iterations = {budget}"), ""));
        assert_eq!(problem.optimization.max_iterations, budget);
        let notes = Notes::default();
        Wireframe::build(&problem, &notes).expect("a guide");
        let notes = notes.taken();
        assert!(
            notes
                .iter()
                .any(|n| n.contains("warning") && n.contains("never released")),
            "a floor that never lets go has to be reported: {notes:?}"
        );

        // One iteration shorter, and there is nothing to report.
        let problem = build(&walled(
            true,
            &format!("hold_iterations = {}", budget - 1),
            "",
        ));
        let notes = Notes::default();
        Wireframe::build(&problem, &notes).expect("a guide");
        assert!(
            !notes.taken().iter().any(|n| n.contains("never released")),
            "a floor that releases inside the budget is not a warning"
        );
    }

    /// A problem with one region and nothing to join it to gets no guide, and is
    /// told why rather than left wondering.
    #[test]
    fn a_problem_with_nothing_to_join_gets_no_guide() {
        // A bar two cells long whose load region reaches back to the middle node,
        // so the cells the load touches include the one the support touches: the
        // two terminals overlap and no path is needed to join them.
        let text = r#"
[project]
name = "stub"

[resolution]
voxel_size_mm = 8.0

[material]
preset = "pla"

[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.3
min_feature_mm = 16.0
max_iterations = 2

[optimization.wireframe]

[output]
stl_path = "stub.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [16.0, 8.0, 8.0]

[[supports]]
region = { shape = "box", min = [-1.0, -1.0, -1.0], max = [1.0, 9.0, 9.0] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [7.0, -1.0, -1.0], max = [17.0, 9.0, 9.0] }
vector = [0.0, 0.0, -100.0]
"#;
        let problem = build(text);
        let notes = Notes::default();
        assert!(Wireframe::build(&problem, &notes).is_none());
        let notes = notes.taken();
        assert!(
            notes
                .iter()
                .any(|n| n.contains("warning") && n.contains("already touches another")),
            "the reason has to be reported: {notes:?}"
        );

        // And with the table absent there is neither a guide nor a word about it.
        let mut config = Config::parse(text).expect("parse");
        config.optimization.wireframe = None;
        let problem = Problem::build(&config, &PathBuf::from(".")).expect("build");
        let notes = Notes::default();
        assert!(Wireframe::build(&problem, &notes).is_none());
        assert!(
            notes.taken().is_empty(),
            "a feature nobody asked for says nothing"
        );
    }
}
