//! Minimum printable thickness, enforced on the field that ships: the arms that
//! came out too thin to print are thickened until they are not.
//!
//! An optimizer spends its material budget where the compliance pays for it, and
//! at the far end of that trade it builds members a hundredth of a millimetre
//! wide if the objective asks for them. Thresholded at the iso level those
//! members become geometry, and a slicer meets a wall thinner than one extrusion:
//! it lays a single bead where a member was meant to be, or it drops the feature
//! entirely. Nothing upstream prevents it. `min_feature_mm` is the radius of a
//! *density filter*, which is a smoothing length and not a floor - it stops the
//! design changing faster than that, it does not stop it converging on something
//! thinner - and the growth engine's Murray sizing is a rule about a skeleton
//! rather than a promise about the rasterized result. This pass is the promise,
//! made after the run, under `[output] reinforce`, which is `off` unless a
//! configuration asks for it.
//!
//! It is also the other half of a trade. [`crate::trim`] frees the mass in the
//! prongs that carry nothing; this spends mass on the arms that carry something
//! and cannot be printed. Run together they shift material out of the parts of
//! the design that are not structure and into the parts that are, which is the
//! exchange the two notes are written to read as.
//!
//! **The measure is the inscribed ball.** A Euclidean distance transform over
//! the part gives every cell `d`, the distance from its centre to the nearest
//! cell that is not part of the part, and the *surface* lies half a voxel inside
//! that cell ([`constants::REINFORCE_SURFACE_OFFSET_VOXELS`]), so a member `w`
//! cells across measures `w` cells thick. Local thickness is then `2 d` - but
//! only where `d` is a local maximum. That is the whole reason this pass looks at
//! the **spine**: at a local maximum of the distance field the ball inscribed at
//! the cell touches the boundary on opposite sides, so its diameter is the
//! thickness of the member there; anywhere else the ball touches the near
//! boundary alone and `2 d` reads as the distance to the nearest surface rather
//! than as a thickness. The pass therefore measures thickness where the
//! measurement means something, at a cost of one distance transform, and the
//! exact sphere-fitting reading - the maximum over every ball that *covers* a
//! cell - is what the tests check the result against.
//!
//! Spine cells are found plateau tolerantly - no neighbour strictly further from
//! the surface - over the block of twenty-six cells around them. The tolerance is
//! what finds the ridge of a slab of even width, where two cells carry the same
//! distance and neither is strictly the larger; the diagonals are what keep it
//! from finding ridges that are not there, because on the six faces alone every
//! convex edge of every part is a plateau and would come back as a thin member.
//! Off-grid neighbours carry zero: the block the exporter wraps in a layer of
//! zeros ends at the grid, and the part's surface there is a real surface.
//!
//! **The operation is a ball fill.** Every spine whose diameter falls short of
//! the floor has every *design* cell within `thickness / 2` of it raised to
//! [`constants::DENSITY_MAX`] - the fill the cavity policy of [`crate::voids`]
//! uses, on the cells [`crate::grid::CellKind::Design`] leaves free. That gate is
//! the whole of the safety story: a keepout, a cell outside the domain and a
//! cavity the void policy owns are all [`crate::grid::CellKind::Void`], forced
//! solid cells are already full, and nothing here has to know which of those it
//! is looking at to leave it alone. The radius is half a voxel longer than half
//! the floor, because the material a filled cell contributes reaches to its own
//! face: filling to `thickness / 2` measured centre to centre would leave the
//! surface half a voxel short of the floor, which is the same half voxel the
//! measure takes off.
//!
//! **What cannot be done is reported, not refused.** A member pressed against a
//! keepout wall or the edge of the domain has nowhere to grow into, and its ball
//! comes back clipped. The pass runs the distance transform a second time over
//! the field it has just changed and re-reads every place it called thin; the
//! ones a ball of the floor's diameter still does not reach are counted in
//! [`ReinforceReport::still_thin`] and named in a warning. That second reading is
//! the *opening* of the part by that ball rather than the spine measure again,
//! because the fill moves the spine of everything it touches - see
//! `opened_by_the_floor`. Every other member is reinforced regardless: the
//! count and its cause are what a run says about it, the way
//! [`crate::mesh::clamp::ClampReport::gave_up`] reports the vertices the boundary
//! clamp could not move. There is no pass-level refusal here and no silence.
//!
//! **Nothing is trimmed afterwards, by design.** Reinforcement material is
//! deliberately unloaded: it is there so the member can be printed, not so it can
//! carry anything, and the stress solve that runs over the reinforced field will
//! duly find almost nothing in it. Trimming it would remove it again, which is
//! why the pipeline runs the trim *first* - on the stresses of the field the
//! engine produced - and reinforces after it, once. See [`crate::complete`].
//!
//! One interplay worth stating: under `[optimization.local_volume]` the cap is a
//! constraint on the *optimizer*, and cells this pass thickens may push a
//! neighbourhood past it in the exported field. That is the intended order of
//! precedence - a part that cannot be printed is not a part - and the report says
//! so rather than leaving the two numbers to disagree in silence.

use rayon::prelude::*;

use crate::config::ReinforcePolicy;
use crate::constants;
use crate::engine::cell_density;
use crate::grid::{CellKind, Grid};
use crate::problem::Problem;

/// What the reinforcement pass found and did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReinforceReport {
    /// Cells the pass raised to [`constants::DENSITY_MAX`]. Only the ones it
    /// really changed: a design cell already at full density is material the
    /// member already had.
    pub cells_raised: usize,
    /// Volume those cells hold, in cubic millimetres.
    pub volume_added_mm3: f64,
    /// Thin places the fill could not bring up to the floor, because the ball it
    /// would have taken ran into a keepout, a cavity or the edge of the domain.
    /// The exported part is still under the floor there.
    pub still_thin: usize,
    /// The floor itself, in millimetres: `[output] reinforce_thickness_mm`,
    /// which absent is `[optimization] min_feature_mm`.
    pub thickness_mm: f64,
    /// Whether `[optimization.local_volume]` was in force. Carried on the report
    /// because [`ReinforceReport::notes`] is the whole contract a caller reads -
    /// it takes no arguments, and the one sentence the cap earns has to come
    /// from somewhere.
    pub local_volume_capped: bool,
}

impl ReinforceReport {
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
    /// [`crate::trim::TrimReport::notes`] keeps. The headline is phrased as
    /// spending, because beside the trim's own note in the same block the two
    /// are the two halves of one exchange.
    pub fn notes(&self) -> Vec<String> {
        let mut notes = Vec::new();
        if self.cells_raised == 0 {
            notes.push(format!(
                "nothing thinner than {:.3} mm; the field is unchanged",
                self.thickness_mm
            ));
        } else {
            notes.push(format!(
                "reinforced {} cells, {:.3} cm3, onto arms below {:.3} mm",
                self.cells_raised,
                self.volume_added_mm3 / constants::MM3_PER_CM3,
                self.thickness_mm
            ));
        }
        if self.still_thin > 0 {
            let places = if self.still_thin == 1 {
                "1 place is".to_string()
            } else {
                format!("{} places are", self.still_thin)
            };
            notes.push(format!(
                "warning: {places} still thinner than {:.3} mm and the exported part is thin \
                 there: the material is pressed against a keepout or the domain edge, and the \
                 floor cannot be met without growing into it",
                self.thickness_mm
            ));
        }
        if self.local_volume_capped && self.cells_raised > 0 {
            notes.push(
                "the [optimization.local_volume] cap is a constraint on the optimizer, and the \
                 reinforced cells may take a neighbourhood past it in the exported field; \
                 printability outranks the cap"
                    .to_string(),
            );
        }
        notes
    }
}

/// The cells the part is made of, as every pass in growforge reads it: physical
/// density at or above the iso level the surface is extracted at.
fn part_mask(grid: &Grid, densities: &[f64], iso: f64) -> Vec<bool> {
    (0..grid.n_cells())
        .into_par_iter()
        .map(|e| cell_density(grid, densities, e) >= iso)
        .collect()
}

/// Distance from a cell's centre to the surface of the part, in millimetres,
/// given its squared distance to the nearest cell that is not part of it.
///
/// The half voxel is the whole difference between a cell count and a thickness;
/// see [`constants::REINFORCE_SURFACE_OFFSET_VOXELS`].
fn surface_distance_mm(squared_voxels: f64, h: f64) -> f64 {
    (squared_voxels.sqrt() - constants::REINFORCE_SURFACE_OFFSET_VOXELS) * h
}

/// Squared distance, in voxels, from every cell of the grid to the nearest cell
/// that is **not** part of `part`. Zero on those cells themselves.
///
/// The grid is padded by one cell of open space on every side. Anything past the
/// block is not part of the part: the exporter wraps the node field in a layer of
/// zeros and the surface closes there, so a member reaching the edge of the
/// domain is bounded by a real surface rather than continuing forever.
pub fn squared_distances(grid: &Grid, part: &[bool]) -> Vec<f64> {
    distances_to(grid, |cell| !part[cell], true)
}

/// Squared distance, in voxels, from every cell of the grid to the nearest cell
/// `seed` holds. Zero on those cells themselves.
///
/// The mirror of [`squared_distances`], and what the second reading of the pass
/// asks: how far the nearest place that *does* meet the floor is. Nothing past
/// the edge of the block is a seed here - there is no part out there to inscribe
/// a ball in - which is the one thing that differs between the two.
fn distances_to_seeds(grid: &Grid, seed: &[bool]) -> Vec<f64> {
    distances_to(grid, |cell| seed[cell], false)
}

/// The transform both readings share: squared distance, in voxels, from every
/// cell to the nearest cell `seed` holds, with `outside_is_seed` saying what lies
/// past the edge of the grid.
///
/// The exact Euclidean transform, not a chamfer approximation: three separable
/// passes of the lower envelope of parabolas (Felzenszwalb and Huttenlocher),
/// each linear in the cells it sweeps, and each exact given the pass before it -
/// which is why the answer does not depend on the order the axes are taken in.
/// Squared rather than rooted because that is what the passes compose in and
/// what a comparison between two distances can be made in without a square root
/// changing a tie into a difference.
fn distances_to(
    grid: &Grid,
    seed: impl Fn(usize) -> bool + Sync,
    outside_is_seed: bool,
) -> Vec<f64> {
    let dims = [grid.nx + 2, grid.ny + 2, grid.nz + 2];
    let (px, py) = (dims[0], dims[1]);
    // Larger than any squared distance the padded block can hold, so a cell that
    // is not a seed never wins a comparison against one that is.
    let far = (dims[0] * dims[0] + dims[1] * dims[1] + dims[2] * dims[2]) as f64;
    let mut field = vec![0.0; dims[0] * dims[1] * dims[2]];
    field.par_iter_mut().enumerate().for_each(|(p, slot)| {
        let i = p % px;
        let j = (p / px) % py;
        let k = p / (px * py);
        let inside = i > 0 && j > 0 && k > 0 && i <= grid.nx && j <= grid.ny && k <= grid.nz;
        let is_seed = if inside {
            seed(grid.cell_index(i - 1, j - 1, k - 1))
        } else {
            outside_is_seed
        };
        *slot = if is_seed { 0.0 } else { far };
    });

    for axis in 0..3 {
        transform_along(&mut field, dims, axis);
    }

    (0..grid.n_cells())
        .into_par_iter()
        .map(|e| {
            let (i, j, k) = grid.cell_ijk(e);
            field[(i + 1) + px * ((j + 1) + py * (k + 1))]
        })
        .collect()
}

/// One separable pass of the distance transform, along `axis` of a block of
/// `dims` cells.
///
/// Lines along the first axis are contiguous and are swept where they lie.
/// Lines along the other two are gathered into a contiguous buffer, swept there
/// and scattered back, which keeps every sweep on a slice rayon can hand out
/// without an aliasing argument and costs one buffer of the block's size.
fn transform_along(field: &mut [f64], dims: [usize; 3], axis: usize) {
    let n = dims[axis];
    let stride: usize = dims[..axis].iter().product();
    if stride == 1 {
        field.par_chunks_mut(n).for_each(envelope);
        return;
    }
    let mut packed = vec![0.0; field.len()];
    packed
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(line, values)| {
            let start = (line % stride) + (line / stride) * stride * n;
            for (t, slot) in values.iter_mut().enumerate() {
                *slot = field[start + t * stride];
            }
            envelope(values);
        });
    field.par_iter_mut().enumerate().for_each(|(e, slot)| {
        let t = (e / stride) % n;
        let line = (e % stride) + (e / (stride * n)) * stride;
        *slot = packed[line * n + t];
    });
}

/// The lower envelope of the parabolas `f[q] + (x - q)^2` over one line,
/// in place.
///
/// Every intersection is finite - `f` holds finite values and the parabolas are
/// taken in increasing `q` - so the walk back down the hull can never step below
/// the sentinel at `z[0]`, which is what keeps `k` in range without a bound
/// check the algorithm does not have.
fn envelope(f: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    // Hull vertices, the boundaries between them, and the result.
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut d = vec![0.0f64; n];
    let intersection = |f: &[f64], q: usize, p: usize| {
        ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * (q as f64 - p as f64))
    };
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = intersection(f, q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = intersection(f, q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, slot) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        *slot = (q as f64 - p as f64).powi(2) + f[p];
    }
    f.copy_from_slice(&d);
}

/// True when a part cell's inscribed ball is thinner than the floor.
fn is_thin(squared_voxels: f64, h: f64, thickness_mm: f64) -> bool {
    let eps = constants::REINFORCE_THICKNESS_EPS_VOXELS * h;
    2.0 * surface_distance_mm(squared_voxels, h) < thickness_mm - eps
}

/// Which cells a ball of the floor's diameter both fits inside the part at and
/// reaches: the part **opened** by that ball.
///
/// The reading the second pass over the field needs, and the one thing the spine
/// measure cannot be asked for there. Filling around a thin spine moves the
/// spine - a member lying against the domain edge grows inwards, and its ridge
/// travels with the material - so `2 d` at the cell that *used* to be the spine
/// is a distance to the surface afterwards rather than a thickness, and reading
/// it as one condemns members that came out exactly right.
///
/// What the floor really asks is the morphological question: is there a ball of
/// radius `thickness / 2` that fits inside the part and covers this cell. Its
/// centres are the cells the distance field already puts at or above that radius,
/// and "covers" is one more transform - the distance to the nearest such centre,
/// which has to be no further than the radius plus the half voxel by which a ball
/// reaches into the cell its surface passes through. Two thresholds and one
/// transform, exact, and linear in the cells.
fn opened_by_the_floor(
    grid: &Grid,
    part: &[bool],
    squared: &[f64],
    thickness_mm: f64,
) -> Vec<bool> {
    let centres: Vec<bool> = (0..grid.n_cells())
        .into_par_iter()
        .map(|e| part[e] && !is_thin(squared[e], grid.h, thickness_mm))
        .collect();
    let reach = distances_to_seeds(grid, &centres);
    let eps = constants::REINFORCE_THICKNESS_EPS_VOXELS * grid.h;
    let radius_mm = 0.5 * thickness_mm + constants::REINFORCE_SURFACE_OFFSET_VOXELS * grid.h + eps;
    reach
        .into_par_iter()
        .map(|squared| squared.sqrt() * grid.h <= radius_mm)
        .collect()
}

/// The spine cells of the part that are thinner than `thickness_mm`, ascending.
///
/// A spine cell is a local maximum of the distance field over the block of
/// twenty-six cells around it, plateau tolerantly: no neighbour strictly further
/// from the surface, rather than every neighbour strictly nearer. Plateau
/// tolerance is what finds the ridge of a member of even width, where two cells
/// carry the same distance and neither is strictly the larger; the diagonals are
/// what keep that tolerance from finding ridges that are not there. A cell on the
/// convex edge of a solid block sits one voxel from two of its faces and so do
/// all six of its face neighbours - the edge is a plateau, and read on the faces
/// alone every convex edge of every part comes back as a thin member. The cell
/// diagonally inside it is two voxels from the surface and says plainly that it
/// does not. This is deliberately *not* the face connectivity every structural
/// question in growforge is asked at: which cells are joined is one question, and
/// which cell is the ridge of a distance field is another.
///
/// The comparison is on the squared distances, which are exact integers in a
/// `f64` at any grid size this crate will build, so a plateau is a plateau and
/// not a rounding.
///
/// What this measures, and does not: a member whose distance rises all the way
/// along it - a taper - has its only maximum where it is thickest, so a thin
/// taper is reinforced at its root rather than along its length. The spine is
/// where a thickness is a thickness; see the module header.
fn thin_spines(grid: &Grid, part: &[bool], squared: &[f64], thickness_mm: f64) -> Vec<usize> {
    (0..grid.n_cells())
        .into_par_iter()
        .filter(|&e| {
            if !part[e] {
                return false;
            }
            let (i, j, k) = grid.cell_ijk(e);
            for dk in -1i64..=1 {
                for dj in -1i64..=1 {
                    for di in -1i64..=1 {
                        let (ni, nj, nk) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                        // Off-grid neighbours are open space and carry nothing,
                        // which never outranks a cell of the part.
                        if ni < 0
                            || nj < 0
                            || nk < 0
                            || ni >= grid.nx as i64
                            || nj >= grid.ny as i64
                            || nk >= grid.nz as i64
                        {
                            continue;
                        }
                        let neighbour = grid.cell_index(ni as usize, nj as usize, nk as usize);
                        if squared[neighbour] > squared[e] {
                            return false;
                        }
                    }
                }
            }
            is_thin(squared[e], grid.h, thickness_mm)
        })
        .collect()
}

/// Raise every design cell within `radius_mm` of `centre` to full density,
/// marking what it changed in `fill`.
fn fill_ball(grid: &Grid, centre: usize, radius_mm: f64, fill: &mut [bool]) {
    let radius = radius_mm / grid.h;
    let reach = radius.ceil() as usize;
    let (ci, cj, ck) = grid.cell_ijk(centre);
    let lo = |c: usize| c.saturating_sub(reach);
    for k in lo(ck)..=(ck + reach).min(grid.nz - 1) {
        for j in lo(cj)..=(cj + reach).min(grid.ny - 1) {
            for i in lo(ci)..=(ci + reach).min(grid.nx - 1) {
                let (di, dj, dk) = (
                    i as f64 - ci as f64,
                    j as f64 - cj as f64,
                    k as f64 - ck as f64,
                );
                if di * di + dj * dj + dk * dk > radius * radius {
                    continue;
                }
                fill[grid.cell_index(i, j, k)] = true;
            }
        }
    }
}

/// Hold `densities` to a minimum thickness, reporting what that took.
///
/// The core of the pass, taking the floor and the level the part is read at
/// rather than reading them off a problem, so both are testable against a field
/// written by hand. [`resolve`] is what a run calls.
///
/// `local_volume_capped` is only carried into the report; nothing here is
/// measured against the cap. See the module header for why the printability
/// floor outranks it.
pub fn reinforce(
    grid: &Grid,
    densities: &mut [f64],
    iso: f64,
    thickness_mm: f64,
    local_volume_capped: bool,
) -> ReinforceReport {
    let report = |cells_raised: usize, still_thin: usize| ReinforceReport {
        cells_raised,
        volume_added_mm3: cells_raised as f64 * grid.cell_volume(),
        still_thin,
        thickness_mm,
        local_volume_capped,
    };

    let part = part_mask(grid, densities, iso);
    let squared = squared_distances(grid, &part);
    let thin = thin_spines(grid, &part, &squared, thickness_mm);
    if thin.is_empty() {
        return report(0, 0);
    }

    // Half the floor to reach the surface, plus the half voxel between the last
    // filled centre and the face it puts material out to.
    let radius_mm = 0.5 * thickness_mm + constants::REINFORCE_SURFACE_OFFSET_VOXELS * grid.h;
    let mut fill = vec![false; grid.n_cells()];
    for &centre in &thin {
        fill_ball(grid, centre, radius_mm, &mut fill);
    }
    let mut cells_raised = 0;
    for cell in 0..grid.n_cells() {
        // The design gate is the whole of what keeps this out of a keepout, a
        // filled cavity and the space outside the domain; a forced solid cell is
        // already at full density and is not a change.
        if !fill[cell] || grid.cells[cell] != CellKind::Design {
            continue;
        }
        if densities[cell] >= constants::DENSITY_MAX {
            continue;
        }
        densities[cell] = constants::DENSITY_MAX;
        cells_raised += 1;
    }
    // The second reading, over the field as it now stands: a ball clipped by a
    // keepout or by the edge of the domain leaves the member it was meant to
    // thicken short of the floor, and that is what the count reports. Asked as
    // the opening rather than as the spine measure, because the fill has moved
    // the spine of everything it touched; see [`opened_by_the_floor`]. A pass
    // that raised nothing - every ball landed on material that was already
    // there - is measured on the field it was handed, which is the same field.
    let (part, squared) = if cells_raised == 0 {
        (part, squared)
    } else {
        let part = part_mask(grid, densities, iso);
        let squared = squared_distances(grid, &part);
        (part, squared)
    };
    let opened = opened_by_the_floor(grid, &part, &squared, thickness_mm);
    let still_thin = thin.iter().filter(|&&e| !opened[e]).count();
    report(cells_raised, still_thin)
}

/// Apply the configured policy to `densities`, reporting what it did.
///
/// `None` is `reinforce = "off"`: the pass did not run, and there is nothing to
/// say about it. Anything else comes back as a report, including the run that
/// found nothing to thicken.
///
/// Unlike [`crate::trim::resolve`] this needs no stress report: how thin a member
/// is, is geometry, and the answer does not depend on the load. That is what lets
/// it run after the trim on the same field without a solve between them.
pub fn resolve(problem: &Problem, densities: &mut [f64]) -> Option<ReinforceReport> {
    if problem.output.reinforce == ReinforcePolicy::Off {
        return None;
    }
    Some(reinforce(
        &problem.grid,
        densities,
        problem.output.iso_level,
        problem.output.reinforce_thickness_mm,
        problem.optimization.local_volume.is_some(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;

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

    /// A bar of `width` by `width` cells running along x through the middle of
    /// an `n` cell grid, centred on the other two axes.
    ///
    /// It stops three cells short of each face rather than running off the grid,
    /// so its own ends are free surfaces inside the design space: a member that
    /// leaves the domain is pinned against the edge of it by construction, and
    /// that is the fixture of
    /// [`a_member_pinned_against_a_keepout_is_counted_and_warned_about`] rather
    /// than of the ordinary case.
    fn bar(n: usize, width: usize) -> (Grid, Vec<f64>) {
        let grid = design_grid(n);
        let mut densities = vec![0.0; grid.n_cells()];
        let lo = (n - width) / 2;
        for i in 3..n - 3 {
            for j in lo..lo + width {
                for k in lo..lo + width {
                    densities[grid.cell_index(i, j, k)] = 1.0;
                }
            }
        }
        (grid, densities)
    }

    /// The **exact** local thickness of every part cell, in millimetres: twice
    /// the largest inscribed radius over every ball that covers the cell.
    ///
    /// This is the sphere-fitting definition - the distance transform of the
    /// distance transform - and it is deliberately not what the pass computes.
    /// The pass reads `2 d` at the spine, where the two readings agree because
    /// the ball inscribed at a local maximum touches the boundary on both sides;
    /// this brute force is what proves that they do, at a cost quadratic in the
    /// cells, which is affordable in a test on a fixture and nowhere else.
    ///
    /// A cell is covered by the ball at `c` when its centre lies within a half
    /// voxel of that ball, because a cell whose centre the surface passes within
    /// half a voxel of is a cell the ball reaches into.
    ///
    /// Two places the reading is *not* a thickness, and no pass can make it one:
    /// the free tip of a member and the convex corner of a block, where no
    /// inscribed ball of the structure's own thickness contains the point at all.
    /// Assertions below are made on the spine, which is where it is one.
    fn exact_thickness_mm(grid: &Grid, part: &[bool]) -> Vec<f64> {
        let squared = squared_distances(grid, part);
        let centres: Vec<(usize, f64)> = (0..grid.n_cells())
            .filter(|&e| part[e])
            .map(|e| (e, surface_distance_mm(squared[e], grid.h)))
            .collect();
        (0..grid.n_cells())
            .map(|e| {
                if !part[e] {
                    return 0.0;
                }
                let (i, j, k) = grid.cell_ijk(e);
                let mut best = 0.0f64;
                for &(c, radius) in &centres {
                    let (ci, cj, ck) = grid.cell_ijk(c);
                    let (di, dj, dk) = (
                        (i as f64 - ci as f64) * grid.h,
                        (j as f64 - cj as f64) * grid.h,
                        (k as f64 - ck as f64) * grid.h,
                    );
                    let distance = (di * di + dj * dj + dk * dk).sqrt();
                    if distance
                        <= radius + constants::REINFORCE_SURFACE_OFFSET_VOXELS * grid.h + 1e-9
                    {
                        best = best.max(2.0 * radius);
                    }
                }
                best
            })
            .collect()
    }

    /// The exact reading at every spine cell of a field, which is where a
    /// thickness is a thickness. `(cell, thickness_mm)` pairs, ascending.
    fn spine_thicknesses(grid: &Grid, densities: &[f64], iso: f64) -> Vec<(usize, f64)> {
        let part = part_mask(grid, densities, iso);
        let squared = squared_distances(grid, &part);
        let exact = exact_thickness_mm(grid, &part);
        // Every spine there is, which `thin_spines` with an unreachable floor
        // is exactly: a floor no member can meet leaves the thinness test always
        // true, so what comes back is the spine itself.
        thin_spines(grid, &part, &squared, f64::MAX)
            .into_iter()
            .map(|cell| (cell, exact[cell]))
            .collect()
    }

    /// A single cell of material: the distance to open space is one voxel from
    /// it and grows by the lattice's own metric away from it.
    #[test]
    fn the_distance_transform_is_exact_on_a_single_cell() {
        let grid = design_grid(7);
        let mut part = vec![false; grid.n_cells()];
        part[grid.cell_index(3, 3, 3)] = true;
        let squared = squared_distances(&grid, &part);

        assert_eq!(squared[grid.cell_index(3, 3, 3)], 1.0);
        // Every cell that is not part of the part is at distance zero from one.
        for (cell, &value) in squared.iter().enumerate() {
            if !part[cell] {
                assert_eq!(value, 0.0, "cell {cell}");
            }
        }
    }

    /// A slab of known width, where the answer can be written down: the cell `t`
    /// cells in from a face is `min(t + 1, width - t)` from open space, and the
    /// transform gives the square of it.
    #[test]
    fn the_distance_transform_is_exact_on_a_slab_of_known_width() {
        for width in 1..6usize {
            let n = 9;
            let grid = design_grid(n);
            let mut part = vec![false; grid.n_cells()];
            for i in 0..width {
                for j in 0..n {
                    for k in 0..n {
                        part[grid.cell_index(i, j, k)] = true;
                    }
                }
            }
            let squared = squared_distances(&grid, &part);
            for t in 0..width {
                let expected = ((t + 1).min(width - t) as f64).powi(2);
                let cell = grid.cell_index(t, n / 2, n / 2);
                assert_eq!(
                    squared[cell], expected,
                    "slab of width {width}, cell {t} in from the face"
                );
                // And the thickness the pass reads off the spine of it is the
                // width itself, in millimetres.
                if (t + 1).min(width - t) == width.div_ceil(2) {
                    let thickness = 2.0 * surface_distance_mm(squared[cell], grid.h);
                    let expected = if width % 2 == 0 {
                        // An even slab has no cell on its mid plane, so the
                        // ridge sits half a voxel off it and reads one voxel
                        // short - in the direction that thickens rather than
                        // the one that leaves a thin member alone.
                        width as f64 - 1.0
                    } else {
                        width as f64
                    };
                    assert!(
                        (thickness - expected).abs() < 1e-12,
                        "slab of width {width} read as {thickness} mm thick"
                    );
                }
            }
        }
    }

    /// An L of two arms, where the answer is the lattice distance to the nearest
    /// cell outside the L and the inside corner is what an axis-by-axis
    /// approximation would get wrong.
    #[test]
    fn the_distance_transform_is_exact_on_an_l_shape() {
        let grid = design_grid(9);
        let mut part = vec![false; grid.n_cells()];
        // Two arms two cells thick meeting at (2, 2), in the plane k = 4.
        for i in 2..8 {
            for j in 2..4 {
                part[grid.cell_index(i, j, 4)] = true;
            }
        }
        for j in 2..8 {
            for i in 2..4 {
                part[grid.cell_index(i, j, 4)] = true;
            }
        }
        let squared = squared_distances(&grid, &part);

        // Brute force, over every cell that is not part of the part.
        let empty: Vec<(f64, f64, f64)> = (0..grid.n_cells())
            .filter(|&e| !part[e])
            .map(|e| {
                let (i, j, k) = grid.cell_ijk(e);
                (i as f64, j as f64, k as f64)
            })
            .collect();
        for cell in 0..grid.n_cells() {
            if !part[cell] {
                continue;
            }
            let (i, j, k) = grid.cell_ijk(cell);
            let expected = empty
                .iter()
                .map(|&(x, y, z)| {
                    (i as f64 - x).powi(2) + (j as f64 - y).powi(2) + (k as f64 - z).powi(2)
                })
                .fold(f64::INFINITY, f64::min);
            assert_eq!(squared[cell], expected, "cell ({i}, {j}, {k})");
        }
    }

    /// The three passes compose in any order, which is what "separable" means
    /// and what a transform that got its strides wrong would not survive.
    #[test]
    fn the_transform_does_not_depend_on_the_order_of_its_axes() {
        let grid = design_grid(6);
        let mut part = vec![false; grid.n_cells()];
        for i in 1..5 {
            part[grid.cell_index(i, 2, 2)] = true;
        }
        part[grid.cell_index(3, 3, 2)] = true;
        part[grid.cell_index(3, 4, 2)] = true;
        let reference = squared_distances(&grid, &part);

        let dims = [grid.nx + 2, grid.ny + 2, grid.nz + 2];
        for order in [[2usize, 1, 0], [1, 0, 2], [0, 2, 1]] {
            let far = (dims[0] * dims[0] + dims[1] * dims[1] + dims[2] * dims[2]) as f64;
            let mut field = vec![0.0; dims[0] * dims[1] * dims[2]];
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        if part[grid.cell_index(i, j, k)] {
                            field[(i + 1) + dims[0] * ((j + 1) + dims[1] * (k + 1))] = far;
                        }
                    }
                }
            }
            for axis in order {
                transform_along(&mut field, dims, axis);
            }
            for (cell, expected) in reference.iter().enumerate() {
                let (i, j, k) = grid.cell_ijk(cell);
                assert_eq!(
                    field[(i + 1) + dims[0] * ((j + 1) + dims[1] * (k + 1))],
                    *expected,
                    "axis order {order:?} disagrees at cell {cell}"
                );
            }
        }
    }

    /// The pass itself, on the case it exists for: a bar three voxels across
    /// with a floor of five is thin, and comes out at the floor or above it
    /// everywhere along its spine.
    #[test]
    fn a_bar_below_the_floor_is_thickened_up_to_it() {
        let (grid, mut densities) = bar(15, 3);
        let before = densities.clone();
        let floor = 5.0;
        assert!(
            spine_thicknesses(&grid, &densities, 0.5)
                .iter()
                .any(|&(_, thickness)| thickness < floor),
            "the fixture is not thin to begin with"
        );

        let report = reinforce(&grid, &mut densities, 0.5, floor, false);
        assert!(report.changed_the_field());
        assert!(report.cells_raised > 0);
        assert_eq!(report.still_thin, 0, "{report:?}");
        assert_eq!(report.thickness_mm, floor);
        assert!(
            (report.volume_added_mm3 - report.cells_raised as f64 * grid.cell_volume()).abs()
                < 1e-12
        );

        // The exact reading, at every spine of the field that came out: the
        // guarantee the pass makes, checked by the definition it does not use.
        for (cell, thickness) in spine_thicknesses(&grid, &densities, 0.5) {
            assert!(
                thickness >= floor - 1e-9,
                "spine cell {cell} came out {thickness} mm thick"
            );
        }
        // And every cell the bar started with is now inside a member that meets
        // the floor: the pass grew around the material rather than beside it.
        let part = part_mask(&grid, &densities, 0.5);
        let exact = exact_thickness_mm(&grid, &part);
        for cell in 0..grid.n_cells() {
            if before[cell] >= 0.5 {
                assert!(
                    exact[cell] >= floor - 1e-9,
                    "cell {cell} of the original bar is {} mm thick",
                    exact[cell]
                );
            }
        }
        // Material was added and none was taken away.
        for cell in 0..grid.n_cells() {
            assert!(densities[cell] >= before[cell], "cell {cell} lost material");
        }
    }

    /// A bar already at the floor is left exactly as it was, and says so rather
    /// than reporting a reinforcement of nothing.
    #[test]
    fn a_bar_already_at_the_floor_is_untouched() {
        let (grid, mut densities) = bar(15, 5);
        let before = densities.clone();
        let report = reinforce(&grid, &mut densities, 0.5, 5.0, false);

        assert!(!report.changed_the_field());
        assert_eq!(report.cells_raised, 0);
        assert_eq!(report.still_thin, 0);
        assert_eq!(report.volume_added_mm3, 0.0);
        assert_eq!(
            densities, before,
            "a pass with nothing to do moved the field"
        );
        assert!(
            report.notes()[0].starts_with("nothing thinner"),
            "{:?}",
            report.notes()
        );
    }

    /// Only design cells are ever written: a keepout, the space outside the
    /// domain and a forced solid pad all come through byte for byte.
    #[test]
    fn only_design_cells_are_ever_raised() {
        let n = 15;
        let (mut grid, mut densities) = bar(n, 3);
        // A keepout wall on one side of the bar, a forced solid pad on the
        // other, and a strip outside the domain further out - all three inside
        // the reach of the fill, and all three the same `Void` or `Solid` the
        // classifier writes.
        for i in 0..n {
            for k in 0..n {
                for (j, kind) in [
                    (10, CellKind::Void),
                    (4, CellKind::Solid),
                    (14, CellKind::Void),
                ] {
                    let cell = grid.cell_index(i, j, k);
                    grid.cells[cell] = kind;
                }
            }
        }
        let before = densities.clone();
        let report = reinforce(&grid, &mut densities, 0.5, 5.0, false);
        assert!(report.cells_raised > 0);

        for cell in 0..grid.n_cells() {
            if grid.cells[cell] != CellKind::Design {
                assert_eq!(
                    densities[cell], before[cell],
                    "a {:?} cell was written",
                    grid.cells[cell]
                );
            }
        }
    }

    /// A member pressed against a keepout wall cannot be brought up to the
    /// floor, and the pass says so: it counts the place, warns about it, and
    /// reinforces everything else it can reach.
    #[test]
    fn a_member_pinned_against_a_keepout_is_counted_and_warned_about() {
        let n = 17;
        let mut grid = design_grid(n);
        let mut densities = vec![0.0; grid.n_cells()];
        // A slot two cells wide walled in by keepout on both sides, thicker
        // than the fill can reach across: nothing can be thickened into it.
        for i in 0..n {
            for k in 0..n {
                for j in (0..3).chain(5..9) {
                    let cell = grid.cell_index(i, j, k);
                    grid.cells[cell] = CellKind::Void;
                }
            }
        }
        for i in 0..n {
            for j in 3..5 {
                for k in 6..9 {
                    densities[grid.cell_index(i, j, k)] = 1.0;
                }
            }
        }
        // And a second member, in open design space, that can be.
        for i in 0..n {
            for j in 12..15 {
                for k in 8..11 {
                    densities[grid.cell_index(i, j, k)] = 1.0;
                }
            }
        }
        let free = grid.cell_index(8, 13, 9);
        let report = reinforce(&grid, &mut densities, 0.5, 5.0, false);

        assert!(report.still_thin > 0, "{report:?}");
        assert!(
            report.cells_raised > 0,
            "the member that could be reinforced was not"
        );
        let notes = report.notes();
        assert!(notes[0].starts_with("reinforced"), "{notes:?}");
        assert!(
            notes.iter().any(|note| note.starts_with("warning:")
                && note.contains("keepout")
                && note.contains("5.000 mm")),
            "{notes:?}"
        );
        // The pass completed: the free member really is at the floor now.
        let part = part_mask(&grid, &densities, 0.5);
        let exact = exact_thickness_mm(&grid, &part);
        assert!(
            exact[free] >= 5.0 - 1e-9,
            "the free member came out {} mm thick",
            exact[free]
        );
        // And nothing crossed the wall.
        for (cell, density) in densities.iter().enumerate() {
            if grid.cells[cell] == CellKind::Void {
                assert_eq!(*density, 0.0);
            }
        }
    }

    /// A plate that can only grow one way is brought up to the floor and is
    /// **not** warned about: the fill moved its spine, and the cell that used to
    /// be the spine is no longer where the thickness is read.
    ///
    /// The regression this pins is a warning that was not true. Re-reading `2 d`
    /// at the old spine after the fill measures the distance from it to the near
    /// surface of a member that grew away from it, which on the plainest of
    /// parts - a cantilever whose chords lie along the faces of its own domain -
    /// condemned eight hundred places that had come out exactly right. What the
    /// second reading asks now is whether a ball of the floor's diameter fits
    /// and reaches; see [`opened_by_the_floor`].
    ///
    /// The rim of the plate is not part of the claim, and no pass could make it
    /// one: the outer corner of any body holds no inscribed ball of the body's
    /// own thickness. What is asserted is the plate's interior, a fill radius in
    /// from its edge.
    #[test]
    fn a_plate_that_grew_away_from_the_domain_edge_is_not_warned_about() {
        let n = 15;
        let grid = design_grid(n);
        let mut densities = vec![0.0; grid.n_cells()];
        // A plate two cells thick lying flat against the edge of the domain,
        // with design space above it: it can only grow one way, and growing
        // that way is enough. Every cell of it is a spine - the whole plate is
        // one plateau at a distance of one voxel - so the fill lays a slab
        // rather than a row of balls.
        let (lo, hi) = (3, n - 3);
        for i in lo..hi {
            for j in 0..2 {
                for k in lo..hi {
                    densities[grid.cell_index(i, j, k)] = 1.0;
                }
            }
        }
        let floor = 5.0;
        let before = densities.clone();
        let report = reinforce(&grid, &mut densities, 0.5, floor, false);
        assert!(report.cells_raised > 0);

        let part = part_mask(&grid, &densities, 0.5);
        let squared = squared_distances(&grid, &part);
        let opened = opened_by_the_floor(&grid, &part, &squared, floor);
        let exact = exact_thickness_mm(&grid, &part);
        // The interior of the plate: far enough in from its rim that the fill
        // around it was not cut short by the plate's own edge.
        let reach = (0.5 * floor / grid.h).ceil() as usize + 1;
        let mut interior = 0;
        let mut would_have_been_condemned = 0;
        for cell in 0..grid.n_cells() {
            let (i, j, k) = grid.cell_ijk(cell);
            if before[cell] < 0.5
                || i < lo + reach
                || i + reach >= hi
                || k < lo + reach
                || k + reach >= hi
            {
                continue;
            }
            interior += 1;
            if is_thin(squared[cell], grid.h, floor) {
                // The naive reading, at a cell the fill left a voxel from the
                // edge of the domain: this is what used to be warned about.
                would_have_been_condemned += 1;
            }
            assert!(
                opened[cell],
                "cell ({i}, {j}, {k}) of a plate that reached the floor was called thin"
            );
            assert!(
                exact[cell] >= floor - 1e-9,
                "cell ({i}, {j}, {k}) came out {} mm thick",
                exact[cell]
            );
        }
        assert!(interior > 0, "the fixture has no interior to assert on");
        assert!(
            would_have_been_condemned > 0,
            "the fixture no longer moves the spine, so it pins nothing"
        );
    }

    /// The local volume note is informational, appears only when the cap was
    /// configured *and* the pass spent material, and never opens with "warning".
    #[test]
    fn the_local_volume_note_is_only_added_when_the_cap_and_the_pass_both_ran() {
        let (grid, mut thin) = bar(15, 3);
        let capped = reinforce(&grid, &mut thin, 0.5, 5.0, true);
        assert!(capped.changed_the_field());
        let note = capped
            .notes()
            .into_iter()
            .find(|note| note.contains("local_volume"))
            .expect("the cap earns a note");
        assert!(!note.starts_with("warning"), "{note}");

        let (grid, mut thin) = bar(15, 3);
        let uncapped = reinforce(&grid, &mut thin, 0.5, 5.0, false);
        assert!(
            !uncapped.notes().iter().any(|n| n.contains("local_volume")),
            "{:?}",
            uncapped.notes()
        );

        let (grid, mut thick) = bar(15, 5);
        let unchanged = reinforce(&grid, &mut thick, 0.5, 5.0, true);
        assert!(!unchanged.changed_the_field());
        assert!(
            !unchanged.notes().iter().any(|n| n.contains("local_volume")),
            "a pass that spent nothing cannot have exceeded a cap: {:?}",
            unchanged.notes()
        );
    }

    /// The two policy paths of [`resolve`], on a problem rather than a
    /// hand-written field: `off` does nothing at all, and the floor a run with no
    /// `reinforce_thickness_mm` holds to is `min_feature_mm`.
    #[test]
    fn the_policy_decides_whether_the_pass_runs_at_all() {
        use crate::config::Config;

        const TINY: &str = r#"
[project]
name = "reinforce_policy"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0

[output]
stl_path = "reinforce_policy.stl"

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
        assert_eq!(problem.output.reinforce, ReinforcePolicy::Off);
        assert_eq!(
            problem.output.reinforce_thickness_mm,
            problem.optimization.min_feature_mm
        );

        // A single row of cells along x: three voxels of six millimetres against
        // a floor of eight.
        let mut densities = vec![0.0; problem.grid.n_cells()];
        for i in 0..problem.grid.nx {
            densities[problem.grid.cell_index(i, 4, 4)] = 1.0;
        }
        let before = densities.clone();
        assert!(
            resolve(&problem, &mut densities).is_none(),
            "the pass ran with reinforce = \"off\""
        );
        assert_eq!(densities, before);

        problem.output.reinforce = ReinforcePolicy::MinThickness;
        let report = resolve(&problem, &mut densities).expect("a report");
        assert_eq!(report.thickness_mm, 8.0);
        assert!(report.changed_the_field());
        assert!(!report.local_volume_capped);
        assert!(report.notes()[0].starts_with("reinforced"), "{report:?}");
    }
}
