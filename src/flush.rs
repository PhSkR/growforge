//! Walls held flush against the surfaces they rest on: the field is filled out
//! to the analytic domain and keepout boundaries wherever a wall already stands
//! against one.
//!
//! A density optimizer converges on a *field*, and the last cells of a wall
//! pressed against the edge of the domain are the cells it has the least reason
//! to resolve: they come out at 0.4, 0.6, 0.8 in a band that should be solid,
//! because the compliance barely notices and the density filter smooths across
//! the boundary as if there were design space on the other side of it. Read at
//! the iso level that band is not a wall standing on a surface. It is a wall
//! that dips inwards by a fraction of a voxel here and reaches the surface
//! there, and what the exported STL shows is a rippled face where a flat one was
//! drawn.
//!
//! **The mesh side of this is already solved, and it is not enough.**
//! [`crate::mesh::clamp`] seats every exported vertex that rests within
//! [`constants::BOUNDARY_CLAMP_CAPTURE_VOXELS`] of a boundary exactly onto it,
//! in both directions. That half voxel is the scale of the *classification*
//! error, and it is deliberately not larger: a vertex further from a boundary
//! than that rests on nothing, and moving it would drag an optimizer's free
//! surface onto a wall it was never near. So where the field dips deeper than
//! the capture band the clamp is right to leave the vertex alone, and the ripple
//! survives into the file. What is missing is material, and only the field can
//! be given it. This pass is that, under `[output] flush`, which is `off` unless
//! a configuration asks for it.
//!
//! **The predicate is a band, grown from the wall inside it.** A design cell is
//! raised to [`constants::DENSITY_MAX`] when all three hold:
//!
//! * it is not already there - a cell at full density is material the wall
//!   already had, and raising it changes nothing;
//! * its centre lies within `flush_depth_mm` of a constraint surface, which is
//!   the domain's own boundary ([`crate::geometry::Csg::signed_distance`]) or
//!   the wall of the nearest keepout
//!   ([`crate::geometry::ShapeUnion::signed_distance`]), whichever is nearer,
//!   taken unsigned - the *band*;
//! * the part's own material, **inside that band**, comes within
//!   `flush_depth_mm` of it, measured by the exact Euclidean transform of
//!   [`crate::reinforce::distances_to_seeds`] seeded on those cells.
//!
//! The third condition is what makes this a correction rather than a coat of
//! paint. A boundary region with no wall on it - the open face of a bracket,
//! the mouth of a bore the design never reached - has no material in the band
//! anywhere near it and comes through untouched, which is the whole difference
//! between filling the ripples of a wall and skinning the domain. And it is the
//! material *in the band* rather than the part's material anywhere, because a
//! member running two voxels clear of a face is not a wall resting on it: seeded
//! on the part at large, a bar through the middle of a domain would grow a
//! detached plate on every face it passed within `flush_depth_mm` of, joined to
//! nothing and culled as an island. Material in the band is the definition of
//! "resting against".
//!
//! **The gate is [`crate::grid::CellKind::Design`]**, and it is the whole of the
//! safety story, exactly as it is for [`crate::reinforce`]: a keepout, the space
//! outside the domain and a cavity the void policy owns are all
//! [`crate::grid::CellKind::Void`], forced solid cells are already full, and
//! nothing here has to know which of those it is looking at to leave it alone.
//! The pass can only ever add material to the design space, so it can only ever
//! strengthen the part - there is no removal to refuse and no connectivity to
//! take apart - and the analysis that runs afterwards over the filled field is
//! what keeps the safety factor honest about the material it added.
//!
//! **The caveat, which is inherent and is why the pass is opt-in.** The fill
//! cannot tell a pockmark in a wall from the end of one. Material that stops
//! inside the band gets joined to the surface it stopped near, and a wall's edge
//! grows a fringe of up to `flush_depth_mm` along the boundary it rests on.
//! That is the same reach that fills the ripple, seen from the other side, and
//! no local rule separates the two: a ripple is a hole in a wall and a fringe is
//! the ground beside one, and within `flush_depth_mm` they look the same. The
//! note the pass writes says so, and the depth is the knob.
//!
//! The pass is therefore applied **once**, to the field an engine produced, and
//! the pipeline applies it exactly once ([`crate::complete`]; the editor's
//! generate-stl button works on a copy of the retained field for the same
//! reason). A second application over its own output would extend that fringe
//! again, by up to `flush_depth_mm` more, wherever the first one left material
//! at the edge of the band; where the first one filled the band it reached, a
//! second raises nothing.
//!
//! **Where it sits in the pipeline: trim, then flush, then reinforce.** The trim
//! judges material by the stresses of the field the engine produced and has to
//! see that field. The flush then puts the walls back to the geometry they were
//! meant to have, *before* the reinforcement measures how thin anything is: a
//! rippled wall reads as a row of thin places that the reinforcement would spend
//! ball after ball on, and the fill it lays against a boundary is exactly the
//! case its own warning is about - see `opened_by_the_floor` and the regression
//! it pins. Flushed first, the reinforcement measures a wall of even thickness
//! and spends its material on the arms that are really thin. All three share the
//! one re-analysis afterwards.

use rayon::prelude::*;

use crate::config::FlushPolicy;
use crate::constants;
use crate::engine::cell_density;
use crate::geometry::Boundaries;
use crate::grid::{CellKind, Grid};
use crate::problem::Problem;
use crate::reinforce::distances_to_seeds;

/// What the flush pass found and did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlushReport {
    /// Cells the pass raised to [`constants::DENSITY_MAX`]. Only the ones it
    /// really changed: a design cell already at full density is material the
    /// wall already had.
    pub cells_raised: usize,
    /// Volume those cells hold, in cubic millimetres.
    pub volume_added_mm3: f64,
    /// How deep the fill reached, in millimetres: `[output] flush_depth_mm`,
    /// which absent is [`constants::FLUSH_DEPTH_VOXELS`] of the voxel size.
    pub depth_mm: f64,
    /// Whether the problem described any surface for a wall to rest against at
    /// all. False only for a caller that built a problem with no domain and no
    /// keepout, where the pass can do nothing whatever the field holds.
    ///
    /// Carried on the report because [`FlushReport::notes`] is the whole
    /// contract a caller reads - it takes no arguments, and a pass that could
    /// not run has to say why rather than reporting an unchanged field.
    pub any_surface: bool,
    /// Whether `[optimization.local_volume]` was in force, for the one sentence
    /// the cap earns; the same reading
    /// [`crate::reinforce::ReinforceReport::local_volume_capped`] carries.
    pub local_volume_capped: bool,
}

impl FlushReport {
    /// True when the field the caller handed in was changed, and therefore has
    /// to be analysed again before anything describes it.
    pub fn changed_the_field(&self) -> bool {
        self.cells_raised > 0
    }

    /// What the run says about this pass, on the console and in the editor's
    /// panel alike.
    ///
    /// One line per sentence, an outcome that needs saying loudly opening with
    /// the word "warning", so a caller can print them straight or lay them out
    /// in a panel without parsing anything back out - the same contract
    /// [`crate::trim::TrimReport::notes`] keeps.
    ///
    /// The caveat gets a line of its own whenever the pass spent material, and
    /// deliberately not a warning: joining material that stopped inside the band
    /// to the surface beside it is what was asked for, not a defect in the
    /// result. The warning is kept for the pass that could not run at all.
    pub fn notes(&self) -> Vec<String> {
        let mut notes = Vec::new();
        if !self.any_surface {
            notes.push(
                "warning: this problem describes no domain or keepout surface, so there is nothing \
                 for a wall to be flush with and the field is unchanged"
                    .to_string(),
            );
            return notes;
        }
        if self.cells_raised == 0 {
            notes.push(format!(
                "no wall stood short of a surface within {:.3} mm; the field is unchanged",
                self.depth_mm
            ));
        } else {
            notes.push(format!(
                "filled {} cells, {:.3} cm3, out to the surfaces the walls rest on, {:.3} mm deep",
                self.cells_raised,
                self.volume_added_mm3 / constants::MM3_PER_CM3,
                self.depth_mm
            ));
            notes.push(format!(
                "material that stopped within {:.3} mm of one of those surfaces is now joined to \
                 it: within that depth the fill cannot tell a pockmark in a wall from the end of \
                 one",
                self.depth_mm
            ));
        }
        if self.local_volume_capped && self.cells_raised > 0 {
            notes.push(
                "the [optimization.local_volume] cap is a constraint on the optimizer, and the \
                 filled cells may take a neighbourhood past it in the exported field; the wall the \
                 configuration drew outranks the cap"
                    .to_string(),
            );
        }
        notes
    }
}

/// How deep the pass fills, in millimetres: what `[output] flush_depth_mm` names
/// when a configuration names one, and otherwise
/// [`constants::FLUSH_DEPTH_VOXELS`] of the voxel size.
///
/// The one statement of that derivation. It lives here rather than in
/// [`crate::config::Config::output_params`] because the voxel size is a property
/// of the grid and not of the `[output]` table, and both the pass and the range
/// check in [`crate::problem::Problem::build`] have to read the same number.
pub fn depth_mm(h: f64, asked: Option<f64>) -> f64 {
    asked.unwrap_or(constants::FLUSH_DEPTH_VOXELS * h)
}

/// The cells the part is made of, as every pass in growforge reads it: physical
/// density at or above the iso level the surface is extracted at.
fn part_mask(grid: &Grid, densities: &[f64], iso: f64) -> Vec<bool> {
    (0..grid.n_cells())
        .into_par_iter()
        .map(|e| cell_density(grid, densities, e) >= iso)
        .collect()
}

/// Distance from the centre of every cell to the nearest constraint surface, in
/// millimetres, unsigned.
///
/// The nearer of the domain's own boundary and the wall of the nearest keepout.
/// Unsigned because what matters is proximity to the surface rather than which
/// side of it a point is on: which side a *cell* is on is already settled by its
/// [`CellKind`], and the fill only ever writes to design cells, which are inside
/// the domain and outside every keepout by construction.
///
/// An empty half is infinitely far away, which is
/// [`crate::geometry::ShapeUnion::signed_distance`]'s own reading of "nothing is
/// forbidden here" and comes through the minimum unchanged.
fn surface_distances_mm(grid: &Grid, boundaries: &Boundaries) -> Vec<f64> {
    (0..grid.n_cells())
        .into_par_iter()
        .map(|e| {
            let (i, j, k) = grid.cell_ijk(e);
            let centre = grid.cell_center(i, j, k);
            let domain = boundaries.domain.signed_distance(centre).abs();
            let keepout = boundaries.keepout.signed_distance(centre).abs();
            domain.min(keepout)
        })
        .collect()
}

/// Fill `densities` out to the surfaces its walls rest on, reporting what that
/// took.
///
/// The core of the pass, taking the depth and the level the part is read at
/// rather than reading them off a problem, so both are testable against a field
/// written by hand. [`resolve`] is what a run calls.
///
/// `local_volume_capped` is only carried into the report; nothing here is
/// measured against the cap. See the module header.
pub fn flush(
    grid: &Grid,
    boundaries: &Boundaries,
    densities: &mut [f64],
    iso: f64,
    depth_mm: f64,
    local_volume_capped: bool,
) -> FlushReport {
    let any_surface = !boundaries.domain.is_empty() || !boundaries.keepout.is_empty();
    let report = |cells_raised: usize| FlushReport {
        cells_raised,
        volume_added_mm3: cells_raised as f64 * grid.cell_volume(),
        depth_mm,
        any_surface,
        local_volume_capped,
    };
    if !any_surface {
        return report(0);
    }

    let surface = surface_distances_mm(grid, boundaries);
    let band: Vec<bool> = surface
        .par_iter()
        .map(|&distance| distance <= depth_mm)
        .collect();
    // The wall itself: the part's material inside the band. What the fill
    // spreads from, and the reason an open stretch of boundary stays open.
    let part = part_mask(grid, densities, iso);
    let wall: Vec<bool> = (0..grid.n_cells())
        .into_par_iter()
        .map(|e| part[e] && band[e])
        .collect();
    let reach = distances_to_seeds(grid, &wall);

    let raise: Vec<bool> = (0..grid.n_cells())
        .into_par_iter()
        .map(|cell| {
            // The design gate is the whole of what keeps this out of a keepout,
            // a filled cavity and the space outside the domain; a forced solid
            // cell is already at full density and is not a change.
            if !band[cell] || grid.cells[cell] != CellKind::Design {
                return false;
            }
            if densities[cell] >= constants::DENSITY_MAX {
                return false;
            }
            reach[cell].sqrt() * grid.h <= depth_mm
        })
        .collect();

    let mut cells_raised = 0;
    for cell in 0..grid.n_cells() {
        if !raise[cell] {
            continue;
        }
        densities[cell] = constants::DENSITY_MAX;
        cells_raised += 1;
    }
    report(cells_raised)
}

/// Apply the configured policy to `densities`, reporting what it did.
///
/// `None` is `flush = "off"`: the pass did not run, and there is nothing to say
/// about it. Anything else comes back as a report, including the run that found
/// no wall standing short of anything.
///
/// Like [`crate::reinforce::resolve`] and unlike [`crate::trim::resolve`] this
/// needs no stress report: where a surface is, is geometry, and the answer does
/// not depend on the load.
pub fn resolve(problem: &Problem, densities: &mut [f64]) -> Option<FlushReport> {
    if problem.output.flush == FlushPolicy::Off {
        return None;
    }
    Some(flush(
        &problem.grid,
        &problem.boundaries,
        densities,
        problem.output.iso_level,
        depth_mm(problem.grid.h, problem.output.flush_depth_mm),
        problem.optimization.local_volume.is_some(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Aabb, Csg, CsgOp, Shape, ShapeUnion};

    /// The side of the test grids, in cells of one millimetre.
    const N: usize = 12;

    /// A cube of design cells `N` on a side, one millimetre each, with the
    /// analytic box it is the voxelization of.
    ///
    /// The domain is the grid's own bounds, so every cell centre is inside it
    /// and the band is the outer shell of the block: cell centres sit at 0.5,
    /// 1.5, ... from a face, so a depth of two millimetres is the two outermost
    /// cells of every axis.
    fn walled_grid() -> (Grid, Boundaries) {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [N as f64, N as f64, N as f64],
        };
        let mut grid = Grid::from_bounds(&bounds, 1.0);
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Design;
        }
        let boundaries = Boundaries {
            domain: Csg::new(vec![(
                CsgOp::Add,
                Shape::axis_aligned_box(bounds.min, bounds.max),
            )]),
            keepout: ShapeUnion::default(),
        };
        (grid, boundaries)
    }

    /// The unsigned distance from a cell centre to the nearest constraint
    /// surface, computed by hand from the analytic box: the smallest gap between
    /// the centre and any of the six faces.
    ///
    /// The reference [`surface_distances_mm`] is checked against, and
    /// deliberately not the same expression: one asks the CSG tree, this one
    /// knows what the tree is.
    fn brute_surface_mm(grid: &Grid, cell: usize) -> f64 {
        let (i, j, k) = grid.cell_ijk(cell);
        let centre = grid.cell_center(i, j, k);
        centre
            .iter()
            .map(|&x| x.min(N as f64 - x))
            .fold(f64::INFINITY, f64::min)
    }

    /// The predicate, computed by hand over every pair of cells: which cells a
    /// fill of `depth` would raise, given the field it starts from.
    ///
    /// Quadratic in the cells, which is affordable on a fixture and nowhere
    /// else, and it is what proves the two transforms in [`flush`] compute what
    /// the module header says they do.
    fn brute_raise(grid: &Grid, densities: &[f64], iso: f64, depth: f64) -> Vec<bool> {
        let band: Vec<bool> = (0..grid.n_cells())
            .map(|e| brute_surface_mm(grid, e) <= depth)
            .collect();
        let wall: Vec<usize> = (0..grid.n_cells())
            .filter(|&e| band[e] && cell_density(grid, densities, e) >= iso)
            .collect();
        (0..grid.n_cells())
            .map(|cell| {
                if !band[cell] || grid.cells[cell] != CellKind::Design {
                    return false;
                }
                if densities[cell] >= constants::DENSITY_MAX {
                    return false;
                }
                let (i, j, k) = grid.cell_ijk(cell);
                wall.iter().any(|&other| {
                    let (oi, oj, ok) = grid.cell_ijk(other);
                    let (di, dj, dk) = (
                        (i as f64 - oi as f64) * grid.h,
                        (j as f64 - oj as f64) * grid.h,
                        (k as f64 - ok as f64) * grid.h,
                    );
                    (di * di + dj * dj + dk * dk).sqrt() <= depth
                })
            })
            .collect()
    }

    /// A wall two cells thick lying against the `k = 0` face over the `i` range
    /// given, at `density`, in a field that is otherwise empty.
    ///
    /// The `i` range is what leaves the rest of that face open: a stretch of
    /// boundary with no material anywhere near it, which is the case the pass
    /// has to leave alone.
    fn wall_against_the_floor(
        grid: &Grid,
        densities: &mut [f64],
        columns: std::ops::Range<usize>,
        density: f64,
    ) {
        for i in columns {
            for j in 0..N {
                for k in 0..2 {
                    densities[grid.cell_index(i, j, k)] = density;
                }
            }
        }
    }

    /// The analytic distances the band is cut at are the ones a hand
    /// computation gives, on the tree the configuration wrote.
    #[test]
    fn the_surface_distance_is_the_distance_to_the_nearest_face() {
        let (grid, boundaries) = walled_grid();
        let measured = surface_distances_mm(&grid, &boundaries);
        for (cell, distance) in measured.iter().enumerate() {
            let expected = brute_surface_mm(&grid, cell);
            assert!(
                (distance - expected).abs() < 1e-12,
                "cell {cell} measured {distance} against {expected}"
            );
        }
    }

    /// The pass on the case it exists for: a wall whose outermost cells came out
    /// below the iso level is filled out to the face it rests on, and the result
    /// is exactly the hand-computed predicate.
    #[test]
    fn a_wall_that_came_out_short_of_its_face_is_filled_out_to_it() {
        let (grid, boundaries) = walled_grid();
        let mut densities = vec![0.0; grid.n_cells()];
        // The wall: solid one cell in, and a rippled outer layer that is under
        // the iso level in every other column - the field an optimizer leaves
        // against a face it has no reason to resolve.
        wall_against_the_floor(&grid, &mut densities, 0..8, 1.0);
        for i in (0..8).step_by(2) {
            for j in 0..N {
                densities[grid.cell_index(i, j, 0)] = 0.35;
            }
        }
        let before = densities.clone();
        let depth = 2.0;
        let expected = brute_raise(&grid, &densities, 0.5, depth);

        let report = flush(&grid, &boundaries, &mut densities, 0.5, depth, false);
        assert!(report.changed_the_field());
        assert!(report.any_surface);
        assert_eq!(report.depth_mm, depth);
        assert_eq!(
            report.cells_raised,
            expected.iter().filter(|&&raise| raise).count()
        );
        assert!(
            (report.volume_added_mm3 - report.cells_raised as f64 * grid.cell_volume()).abs()
                < 1e-12
        );
        for cell in 0..grid.n_cells() {
            let want = if expected[cell] {
                constants::DENSITY_MAX
            } else {
                before[cell]
            };
            assert_eq!(densities[cell], want, "cell {cell}");
        }
        // Every rippled cell of the wall is now solid, which is the claim: the
        // face it rests on has material against it all the way along.
        for i in 0..8 {
            for j in 0..N {
                assert_eq!(
                    densities[grid.cell_index(i, j, 0)],
                    constants::DENSITY_MAX,
                    "the wall is still short of the face at ({i}, {j}, 0)"
                );
            }
        }
        // And nothing lost material.
        for cell in 0..grid.n_cells() {
            assert!(densities[cell] >= before[cell], "cell {cell} lost material");
        }
        let notes = report.notes();
        assert!(notes[0].starts_with("filled"), "{notes:?}");
        assert!(
            notes[1].contains("joined to it"),
            "the caveat is not stated: {notes:?}"
        );
        assert!(
            notes.iter().all(|note| !note.starts_with("warning")),
            "{notes:?}"
        );
    }

    /// The other half of the predicate: a stretch of the same face with no wall
    /// on it grows nothing at all.
    #[test]
    fn an_open_stretch_of_boundary_grows_no_skin() {
        let (grid, boundaries) = walled_grid();
        let mut densities = vec![0.0; grid.n_cells()];
        wall_against_the_floor(&grid, &mut densities, 0..5, 1.0);
        let depth = 2.0;
        let report = flush(&grid, &boundaries, &mut densities, 0.5, depth, false);
        assert!(report.changed_the_field(), "{report:?}");

        // Beyond the fringe the fill is allowed - `depth` past the last wall
        // column, which is the documented caveat - the floor is bare.
        for i in 8..N {
            for j in 0..N {
                for k in 0..2 {
                    assert_eq!(
                        densities[grid.cell_index(i, j, k)],
                        0.0,
                        "the open floor grew a skin at ({i}, {j}, {k})"
                    );
                }
            }
        }
    }

    /// A member running clear of every surface is not a wall, and the band it
    /// passes near is not filled: the fill spreads from the material *in* the
    /// band and from nothing else.
    ///
    /// The regression this pins is a detached plate. Seeded on the part at
    /// large, a bar two voxels clear of a face would put a one-cell skin on that
    /// face, joined to nothing, which the island cull then removes from the mesh
    /// while the field still carries it.
    #[test]
    fn a_member_clear_of_a_surface_grows_nothing_on_it() {
        let (grid, boundaries) = walled_grid();
        let mut densities = vec![0.0; grid.n_cells()];
        // A bar along x through the middle of the block, three cells off the
        // floor and stopping clear of the end faces: its nearest band cell is
        // one cell away, well inside a two millimetre reach, and it is still not
        // a wall.
        for i in 3..N - 3 {
            for j in 5..7 {
                for k in 3..5 {
                    densities[grid.cell_index(i, j, k)] = 1.0;
                }
            }
        }
        let before = densities.clone();
        let report = flush(&grid, &boundaries, &mut densities, 0.5, 2.0, false);

        assert_eq!(report.cells_raised, 0, "{report:?}");
        assert!(!report.changed_the_field());
        assert_eq!(densities, before, "a bar in free space filled a band");
        assert!(
            report.notes()[0].starts_with("no wall stood short"),
            "{:?}",
            report.notes()
        );
    }

    /// The interior of the part is never touched, however dense or thin it is:
    /// a cell further than the depth from every surface is not in the band.
    #[test]
    fn the_interior_of_the_part_is_left_exactly_as_it_was() {
        let (grid, boundaries) = walled_grid();
        let mut densities = vec![0.6; grid.n_cells()];
        let before = densities.clone();
        let depth = 2.0;
        flush(&grid, &boundaries, &mut densities, 0.5, depth, false);

        for cell in 0..grid.n_cells() {
            if brute_surface_mm(&grid, cell) <= depth {
                assert_eq!(densities[cell], constants::DENSITY_MAX, "cell {cell}");
            } else {
                assert_eq!(densities[cell], before[cell], "cell {cell} was written");
            }
        }
    }

    /// Only design cells are ever written: a keepout, the space outside the
    /// domain and a forced solid pad all come through byte for byte, and the
    /// keepout's own wall is a surface the fill reaches out to.
    #[test]
    fn only_design_cells_are_ever_raised() {
        let (mut grid, mut boundaries) = walled_grid();
        // A keepout slab through the middle of the block and the cells it
        // voids; a forced solid pad against the far face; and a strip outside
        // the domain.
        let keepout =
            Shape::axis_aligned_box([-1.0, -1.0, 7.0], [N as f64 + 1.0, N as f64 + 1.0, 9.0]);
        boundaries.keepout = ShapeUnion::new(vec![keepout]);
        for i in 0..N {
            for j in 0..N {
                for (k, kind) in [
                    (7, CellKind::Void),
                    (8, CellKind::Void),
                    (10, CellKind::Solid),
                    (N - 1, CellKind::Void),
                ] {
                    let cell = grid.cell_index(i, j, k);
                    grid.cells[cell] = kind;
                }
            }
        }
        let mut densities = vec![0.0; grid.n_cells()];
        // A wall standing under the keepout's lower face, one cell short of it.
        for i in 0..N {
            for j in 0..N {
                densities[grid.cell_index(i, j, 5)] = 1.0;
            }
        }
        let before = densities.clone();
        let report = flush(&grid, &boundaries, &mut densities, 0.5, 2.0, false);
        assert!(report.cells_raised > 0, "{report:?}");

        for cell in 0..grid.n_cells() {
            if grid.cells[cell] != CellKind::Design {
                assert_eq!(
                    densities[cell], before[cell],
                    "a {:?} cell was written",
                    grid.cells[cell]
                );
            }
        }
        // The design cell between that wall and the keepout wall is filled: the
        // surface a wall rests on is the keepout's as much as the domain's.
        assert_eq!(
            densities[grid.cell_index(3, 3, 6)],
            constants::DENSITY_MAX,
            "the cell against the keepout wall was not filled"
        );
    }

    /// The depth is what the caller asked for, and it is what decides how far
    /// the fill reaches: the same field, filled twice, differs by exactly the
    /// cells the deeper band holds.
    #[test]
    fn the_depth_decides_how_far_the_fill_reaches() {
        let (grid, boundaries) = walled_grid();
        let mut densities = vec![0.0; grid.n_cells()];
        wall_against_the_floor(&grid, &mut densities, 0..N, 1.0);
        // A single column of the wall missing, all the way through: how deep the
        // hole is filled is how deep the band goes.
        for k in 0..N {
            densities[grid.cell_index(4, 4, k)] = 0.0;
        }

        let mut shallow = densities.clone();
        let shallow_report = flush(&grid, &boundaries, &mut shallow, 0.5, 1.0, false);
        let mut deep = densities.clone();
        let deep_report = flush(&grid, &boundaries, &mut deep, 0.5, 3.0, false);

        assert_eq!(shallow_report.depth_mm, 1.0);
        assert_eq!(deep_report.depth_mm, 3.0);
        assert!(
            deep_report.cells_raised > shallow_report.cells_raised,
            "{deep_report:?} against {shallow_report:?}"
        );
        // The hole through the wall: filled to one cell at a depth of one
        // millimetre, and to three at three.
        assert_eq!(shallow[grid.cell_index(4, 4, 0)], constants::DENSITY_MAX);
        assert_eq!(shallow[grid.cell_index(4, 4, 1)], 0.0);
        assert_eq!(deep[grid.cell_index(4, 4, 2)], constants::DENSITY_MAX);
        assert_eq!(deep[grid.cell_index(4, 4, 3)], 0.0);
    }

    /// A pass over a band it has already filled raises nothing.
    ///
    /// The claim the pipeline needs, and exactly the claim it can make: where
    /// the first pass filled the band it reached, a second finds every cell of
    /// it at full density and every cell beyond it further than the depth from
    /// any wall. Where a wall *ends* inside the band the fringe of the module
    /// header is what a second pass would extend, which is why the pipeline
    /// applies this once, to the field an engine produced. The fixture is a
    /// shell of material with pockmarks in it: the pass's own case, and one with
    /// no fringe to extend.
    #[test]
    fn a_second_pass_over_a_filled_band_raises_nothing() {
        let (grid, boundaries) = walled_grid();
        let depth = 2.0;
        // Every cell of the band is material, except the pockmarks.
        let mut densities: Vec<f64> = (0..grid.n_cells())
            .map(|cell| {
                if brute_surface_mm(&grid, cell) <= depth {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        for (i, j, k) in [
            (0usize, 4usize, 0usize),
            (5, 0, 1),
            (3, 3, 0),
            (N - 1, 6, 1),
        ] {
            densities[grid.cell_index(i, j, k)] = 0.2;
        }

        let first = flush(&grid, &boundaries, &mut densities, 0.5, depth, false);
        assert_eq!(first.cells_raised, 4, "{first:?}");
        let filled = densities.clone();
        let second = flush(&grid, &boundaries, &mut densities, 0.5, depth, false);
        assert_eq!(second.cells_raised, 0, "{second:?}");
        assert_eq!(densities, filled, "a second pass moved the field");
        assert!(!second.changed_the_field());
        assert!(
            second.notes()[0].starts_with("no wall stood short"),
            "{:?}",
            second.notes()
        );
    }

    /// A problem with no analytic surfaces at all cannot flush anything, and
    /// says so loudly rather than reporting an unchanged field.
    #[test]
    fn a_problem_with_no_surfaces_says_so_rather_than_reporting_nothing() {
        let (grid, _) = walled_grid();
        let mut densities = vec![1.0; grid.n_cells()];
        let before = densities.clone();
        let report = flush(
            &grid,
            &Boundaries::default(),
            &mut densities,
            0.5,
            2.0,
            false,
        );

        assert!(!report.any_surface);
        assert_eq!(report.cells_raised, 0);
        assert_eq!(densities, before);
        let notes = report.notes();
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].starts_with("warning:"), "{notes:?}");
    }

    /// The local volume note is informational, appears only when the cap was
    /// configured *and* the pass spent material, and never opens with "warning".
    #[test]
    fn the_local_volume_note_is_only_added_when_the_cap_and_the_pass_both_ran() {
        let (grid, boundaries) = walled_grid();
        let mut densities = vec![0.0; grid.n_cells()];
        wall_against_the_floor(&grid, &mut densities, 0..N, 0.6);

        let mut capped_field = densities.clone();
        let capped = flush(&grid, &boundaries, &mut capped_field, 0.5, 2.0, true);
        assert!(capped.changed_the_field());
        let note = capped
            .notes()
            .into_iter()
            .find(|note| note.contains("local_volume"))
            .expect("the cap earns a note");
        assert!(!note.starts_with("warning"), "{note}");

        let mut uncapped_field = densities.clone();
        let uncapped = flush(&grid, &boundaries, &mut uncapped_field, 0.5, 2.0, false);
        assert!(
            !uncapped.notes().iter().any(|n| n.contains("local_volume")),
            "{:?}",
            uncapped.notes()
        );

        let mut empty = vec![0.0; grid.n_cells()];
        let unchanged = flush(&grid, &boundaries, &mut empty, 0.5, 2.0, true);
        assert!(!unchanged.changed_the_field());
        assert!(
            !unchanged.notes().iter().any(|n| n.contains("local_volume")),
            "a pass that spent nothing cannot have exceeded a cap: {:?}",
            unchanged.notes()
        );
    }

    /// The two policy paths of [`resolve`], on a problem rather than a
    /// hand-written field, and the depth both of its ways: derived from the
    /// voxel size when the file names none, and taken as written when it does.
    #[test]
    fn the_policy_decides_whether_the_pass_runs_at_all() {
        use crate::config::Config;

        const TINY: &str = r#"
[project]
name = "flush_policy"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0

[output]
stl_path = "flush_policy.stl"

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
        assert_eq!(problem.output.flush, FlushPolicy::Off);
        assert_eq!(problem.output.flush_depth_mm, None);
        assert_eq!(
            depth_mm(problem.grid.h, problem.output.flush_depth_mm),
            constants::FLUSH_DEPTH_VOXELS * problem.grid.h
        );

        // A wall lying against the floor of the domain, one layer of it under
        // the iso level.
        let grid = &problem.grid;
        let mut densities = vec![0.0; grid.n_cells()];
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                densities[grid.cell_index(i, j, 0)] = 0.3;
                densities[grid.cell_index(i, j, 1)] = 1.0;
            }
        }
        let before = densities.clone();
        assert!(
            resolve(&problem, &mut densities).is_none(),
            "the pass ran with flush = \"off\""
        );
        assert_eq!(densities, before);

        problem.output.flush = FlushPolicy::Walls;
        let report = resolve(&problem, &mut densities).expect("a report");
        assert_eq!(report.depth_mm, constants::FLUSH_DEPTH_VOXELS * grid.h);
        assert!(report.changed_the_field());
        assert!(!report.local_volume_capped);
        assert!(report.notes()[0].starts_with("filled"), "{report:?}");

        // And the written depth, which is used exactly as it stands.
        let mut asked = before.clone();
        problem.output.flush_depth_mm = Some(1.5);
        let report = resolve(&problem, &mut asked).expect("a report");
        assert_eq!(report.depth_mm, 1.5);
    }
}
