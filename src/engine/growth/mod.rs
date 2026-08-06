//! The growth engine: a fast, deterministic, organic alternative to SIMP.
//!
//! Where the SIMP engine asks a finite element solve what to keep, this one
//! grows a structure and then lets the post-run stress report say how good it
//! is. In five stages:
//!
//! 1. **Backbone.** Every load region is routed to every support region it can
//!    reach by a shortest voxel path (A*, 26 neighbours), which is then
//!    shortcut into a polyline. A load region that reaches no support at all
//!    fails the run: without a load path there is nothing to grow onto.
//! 2. **Branching.** Space colonization fills the free space with an organic
//!    canopy hanging off the backbones. Attraction points are seeded both
//!    through the interior, which is what makes the routing organic, and on the
//!    structural surfaces themselves, which is what gives the branches somewhere
//!    to arrive: a surface point is only consumed once a branch has fused to the
//!    surface it sits on.
//! 3. **Pruning.** Every branch that still ends on nothing is removed, back to
//!    the last junction that leads somewhere. `[growth] prune = false` keeps
//!    them.
//! 4. **Thickening.** Each load region pushes its magnitude into the skeleton,
//!    split over *every* place a branch fuses to it rather than over the
//!    backbones alone, the flow accumulates towards the roots, and Murray's law
//!    turns it into a radius per segment. One global scale is bisected until the
//!    result hits `mass_fraction`.
//! 5. **Rasterize.** The struts are unioned as capsules and sampled into the
//!    density field the rest of growforge already knows how to handle.
//!
//! **The strength of the result is a heuristic, not a calculation.** Nothing
//! here solves an equilibrium; the load path exists by construction and the
//! thicknesses follow a branching law, which is a good prior and no more than
//! that. The stress report that runs afterwards on the exported field is the
//! honest answer, and it is the number to read before printing.
//!
//! Determinism is a contract: the same configuration produces a byte identical
//! STL. Everything random comes from the in-crate [`prng`], seeded from the
//! configuration alone, and the one parallel loop in [`rasterize`] is
//! partitioned so that every cell still sees the segments in the same order.
//!
//! **Symmetry.** With `[growth.symmetry]` set, stages one to four run inside
//! one fundamental domain of the declared symmetry and stage five rasterizes
//! the union of its copies, so a symmetric problem produces a part whose
//! sectors are identical rather than four differently scattered legs. See
//! [`symmetry`] for what the fundamental domain is and how a region is decided
//! to belong to it.

pub mod anchor;
pub mod astar;
pub mod colonize;
pub mod domain;
pub mod prng;
pub mod rasterize;
pub mod skeleton;
pub mod symmetry;
pub mod thicken;

use std::borrow::Cow;
use std::time::Instant;

use anyhow::{Result, bail};

use crate::config::GrowthParams;
use crate::constants;
use crate::engine::growth::anchor::AnchorSet;
use crate::engine::growth::colonize::{Colonizer, GrowthStop};
use crate::engine::growth::domain::GrowthDomain;
use crate::engine::growth::prng::Pcg32;
use crate::engine::growth::skeleton::Skeleton;
use crate::engine::growth::symmetry::Symmetry;
use crate::engine::{DensityField, Engine, GrowthSummary, GrowthSymmetry, StopReason};
use crate::geometry::{self, Vec3};
use crate::grid::{CellKind, Grid};
use crate::problem::Problem;
use crate::report::{GrowthPhase, GrowthProgress, IterationStats, Reporter};

/// The growth heuristic engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct GrowthEngine;

/// One place a load enters the structure.
///
/// A region is the node set of a single force or torque load. The same node set
/// appearing in two load cases is one region carrying the sum of both, which is
/// what "the load this point has to survive" means.
#[derive(Debug, Clone)]
struct LoadRegion {
    /// Load case the region was first seen in, for error messages.
    case: String,
    /// One-based index of the load inside that case.
    load: usize,
    /// Cells the region's nodes touch.
    cells: Vec<usize>,
    /// Everything a branch may fuse to and count as carrying this load: the
    /// region's own cells plus every forced solid body they sit on.
    ///
    /// A load applied to a rigid pad travels through that pad, so a branch that
    /// reaches the underside of a tabletop is carrying the tabletop even though
    /// the force arrows were drawn on its top face.
    footprint: AnchorSet,
    /// Total load magnitude in newtons.
    ///
    /// A symmetric run keeps a region's **whole declared magnitude** in the
    /// sector that owns it, including a region that straddles the sector
    /// boundary and is therefore only partly inside it - a centred load pad, a
    /// support in the middle. There is no geometric `1 / sectors` share: the
    /// region is owned whole by the sector its centre is in, and it pushes its
    /// full magnitude into that sector's skeleton. With one load region, or with
    /// regions that all straddle or all do not, this changes nothing at all,
    /// because the flow only decides the *relative* thickness of the struts and
    /// the volume target sets the absolute scale. Where it does show is a
    /// problem mixing the two: a straddling region is weighted against a
    /// non-straddling one as if all of it were in the sector, so its struts come
    /// out thicker than their share of the load - in every copy, since the
    /// copies are copies. The stress report at the end runs on the whole
    /// replicated structure with the real loads and is the honest verdict either
    /// way.
    magnitude_n: f64,
    /// Skeleton nodes the backbones attached to this region, one per support
    /// region it reached. Remapped by pruning.
    backbone_tips: Vec<usize>,
    /// Every skeleton node the load enters the structure through: the backbone
    /// tips plus every branch tip that fused to the footprint.
    attachments: Vec<usize>,
    /// Nodes the region selected, used only to recognise it again.
    nodes: Vec<usize>,
}

/// Magnitude a load region has to carry in one load case, in newtons.
///
/// Taken from the assembled nodal forces rather than from the configuration, so
/// every load type is handled by the same rule: a force load sums back to its
/// own magnitude, and a torque load sums to `M / r_eff` with
/// `r_eff = sum(|r_i|^2) / sum(|r_i|)`, the lever the tangential forces were
/// actually built on. Two loads whose regions overlap both count the shared
/// nodes, which errs towards a thicker strut where two loads meet.
fn region_magnitude(forces: &[f64], nodes: &[usize]) -> f64 {
    nodes
        .iter()
        .map(|&node| {
            let base = node * constants::DOF_PER_NODE;
            geometry::length([forces[base], forces[base + 1], forces[base + 2]])
        })
        .sum()
}

/// Self weight a load case carries at the target volume, in newtons.
///
/// The structure does not exist yet, so its weight is estimated from the volume
/// the run is aiming at: every forced solid cell plus `mass_fraction` of the
/// design cells.
fn case_gravity_n(problem: &Problem, acceleration: Vec3) -> f64 {
    let volume_mm3 = (problem.counts.solid as f64
        + problem.counts.design as f64 * problem.optimization.mass_fraction)
        * problem.cell_volume_mm3();
    problem.material.density_g_cm3
        * constants::TONNE_PER_MM3_PER_G_PER_CM3
        * volume_mm3
        * geometry::length(acceleration)
}

/// Collect the load regions of the whole problem, summing the magnitude of a
/// region that appears in several load cases and sharing each case's self
/// weight equally over the regions in it.
fn load_regions(problem: &Problem, domain: &GrowthDomain) -> Vec<LoadRegion> {
    let mut regions: Vec<LoadRegion> = Vec::new();
    for case in &problem.load_cases {
        let mut in_this_case = Vec::new();
        for (index, load) in case.loads.iter().enumerate() {
            if load.nodes.is_empty() {
                // Gravity: no region of its own, it acts on everything.
                continue;
            }
            let magnitude = region_magnitude(&case.forces, &load.nodes);
            let existing = regions.iter().position(|r| r.nodes == load.nodes);
            let slot = match existing {
                Some(slot) => {
                    regions[slot].magnitude_n += magnitude;
                    slot
                }
                None => {
                    let cells = domain.cells_touching_nodes(&load.nodes);
                    let mut footprint = AnchorSet::empty(domain.grid());
                    footprint.extend(&cells);
                    footprint.extend(&anchor::solid_body(domain.grid(), &cells));
                    regions.push(LoadRegion {
                        case: case.name.clone(),
                        load: index + 1,
                        cells,
                        footprint,
                        magnitude_n: magnitude,
                        backbone_tips: Vec::new(),
                        attachments: Vec::new(),
                        nodes: load.nodes.clone(),
                    });
                    regions.len() - 1
                }
            };
            in_this_case.push(slot);
        }
        if let Some(acceleration) = case.gravity
            && !in_this_case.is_empty()
        {
            let share = case_gravity_n(problem, acceleration) / in_this_case.len() as f64;
            for slot in in_this_case {
                regions[slot].magnitude_n += share;
            }
        }
    }
    regions
}

/// Common centre of a set of cells, in millimetres, or `None` for an empty set.
///
/// It is where a region *is*, which two separate decisions rest on: where a
/// backbone plants its foot (see [`interior_cell`]) and, for a symmetric run,
/// which sector a region belongs to.
fn centroid(domain: &GrowthDomain, cells: &[usize]) -> Option<Vec3> {
    if cells.is_empty() {
        return None;
    }
    let mut centroid = [0.0f64; 3];
    for &cell in cells {
        let centre = domain.cell_center(cell);
        for (slot, value) in centroid.iter_mut().zip(centre.iter()) {
            *slot += value / cells.len() as f64;
        }
    }
    Some(centroid)
}

/// The cell of `cells` nearest to their common centroid.
///
/// This is where a backbone plants its foot: **one support region, one foot, in
/// the middle of it.** Aiming at the region as a whole instead lands the foot
/// wherever the search happens to meet it, which for a region that is uniformly
/// reachable - a patch of floor under a tabletop, say, every cell of it the same
/// distance below the load - is decided by nothing more than the queue's tie
/// break, and comes out at a corner. The foot then stands on the rim of its own
/// footprint and the thickened leg hangs off one side of it, which is what a user
/// saw and reported.
///
/// A foot in the middle overflows its patch on every side when the leg is
/// thicker than the patch is wide, which is honest - the leg really is thicker
/// than what holds it - where hanging off one side is just wrong.
///
/// Ties go to the lowest cell index, which only ever picks between cells the
/// same distance from the centroid, and keeps the choice reproducible.
fn interior_cell(domain: &GrowthDomain, cells: &[usize]) -> Option<usize> {
    let centroid = centroid(domain, cells)?;
    cells.iter().copied().min_by(|a, b| {
        let da = geometry::length(geometry::difference(domain.cell_center(*a), centroid));
        let db = geometry::length(geometry::difference(domain.cell_center(*b), centroid));
        da.total_cmp(&db).then(a.cmp(b))
    })
}

/// Route one guaranteed load path per (load region, support region) pair.
///
/// Every load region has to reach at least one support, or the run fails: a
/// load with no path to ground is a modelling mistake the engine cannot grow
/// its way out of. A support a given load cannot reach is skipped silently -
/// another load may well use it, and a keepout between the two is a legitimate
/// thing to model.
///
/// The two ends of a path are deliberately not treated alike. The search leaves
/// the load region from whichever of its cells is nearest, because a distributed
/// load enters the structure everywhere and the shortest way out of it is the
/// one that spends the least material; it arrives at the *middle* of the support
/// region, because that is the one place this leg is grounded. See
/// [`interior_cell`].
fn grow_backbones(
    problem: &Problem,
    domain: &GrowthDomain,
    support_cells: &[&[usize]],
    problem_step_mm: f64,
    regions: &mut [LoadRegion],
    skeleton: &mut Skeleton,
) -> Result<usize> {
    let min_radius = 0.5 * problem.optimization.min_feature_mm;

    let mut backbones = 0;
    for region in regions.iter_mut() {
        for cells in support_cells {
            // Aim for the middle of the region; fall back to the region as a
            // whole when its middle is walled off, so nothing that used to find
            // a path stops finding one.
            let path = interior_cell(domain, cells)
                .and_then(|goal| astar::shortest_path(domain, &region.cells, &[goal]))
                .or_else(|| astar::shortest_path(domain, &region.cells, cells));
            let Some(path) = path else {
                continue;
            };
            // The path comes back load first; the skeleton is rooted on the
            // supports, so it goes in the other way round.
            let mut points: Vec<Vec3> = path.iter().map(|&c| domain.cell_center(c)).collect();
            points.reverse();
            let radius = skeleton::shortcut_radius(domain, &points, min_radius);
            let simplified = skeleton::shortcut(domain, &points, radius);
            // The shortcut leaves a trunk of very few, very long segments. The
            // canopy sprouts from skeleton nodes and Murray's law tapers between
            // them, so the trunk gets its nodes back at the growth step length -
            // on the same straight lines, so it stays as direct as the shortcut
            // made it.
            let trunk = skeleton::resample(&simplified, problem_step_mm);
            region.backbone_tips.push(skeleton.add_chain(&trunk));
            backbones += 1;
        }
        if region.backbone_tips.is_empty() {
            bail!(
                "load case \"{}\", load {}: no path from the load region to any support region \
                 through the design domain, so there is nothing to grow a load path along \
                 (keepouts and the space outside the domain are impassable{}; check that the \
                 region is not walled off, or run this problem with engine = \"{}\")",
                region.case,
                region.load,
                if domain.is_confined() {
                    ", and so is everything outside the fundamental domain of \
                     [growth.symmetry]"
                } else {
                    ""
                },
                constants::DEFAULT_ENGINE
            );
        }
    }
    Ok(backbones)
}

/// Work out where each load region actually enters the structure.
///
/// The backbone tips are the guaranteed entry points; every branch tip that
/// fused to the region's footprint is another one, and it is a real load path -
/// the tip is joined to material the finite element model applies the force to.
/// Spreading the region's magnitude over all of them, rather than over the
/// backbones alone, is what turns a grown branch from decoration into structure:
/// it carries load, so Murray's law gives it a thickness for carrying it.
fn collect_attachments(
    regions: &mut [LoadRegion],
    skeleton: &Skeleton,
    grid: &Grid,
    tolerance: f64,
) {
    let leaf = skeleton.leaves();
    for region in regions.iter_mut() {
        let mut attachments = region.backbone_tips.clone();
        for (node, is_leaf) in leaf.iter().enumerate() {
            if !is_leaf || attachments.contains(&node) {
                continue;
            }
            if region
                .footprint
                .near(grid, skeleton.position(node), tolerance)
            {
                attachments.push(node);
            }
        }
        attachments.sort_unstable();
        attachments.dedup();
        region.attachments = attachments;
    }
}

/// The whole structure a fundamental skeleton stands for: itself when the run
/// is not symmetric, and the union of its copies when it is.
///
/// Borrowed in the asymmetric case, so the default path allocates nothing and
/// rasterizes exactly the skeleton it grew.
fn replicated<'a>(symmetry: Option<&Symmetry>, skeleton: &'a Skeleton) -> Cow<'a, Skeleton> {
    match symmetry {
        Some(symmetry) => Cow::Owned(symmetry.replicate(skeleton)),
        None => Cow::Borrowed(skeleton),
    }
}

/// Per-node values repeated to match [`replicated`].
fn replicated_radii<'a>(symmetry: Option<&Symmetry>, radii: &'a [f64]) -> Cow<'a, [f64]> {
    match symmetry {
        Some(symmetry) => Cow::Owned(symmetry.repeat(radii)),
        None => Cow::Borrowed(radii),
    }
}

/// Everything a growth run needs to report a progress line.
struct Progress<'a> {
    reporter: &'a dyn Reporter,
    problem: &'a Problem,
    /// The symmetry the reported structure is replicated with, if any: what a
    /// watcher is shown is the whole part, not the sector it is grown from.
    symmetry: Option<&'a Symmetry>,
    start: Instant,
    line: usize,
}

impl Progress<'_> {
    /// Publish one line and, through the observer hook, the density field the
    /// skeleton stands for at this moment.
    fn emit(&mut self, phase: GrowthPhase, skeleton: &Skeleton, remaining: usize, radii: &[f64]) {
        self.line += 1;
        let skeleton = replicated(self.symmetry, skeleton);
        let radii = replicated_radii(self.symmetry, radii);
        let densities = rasterize::densities(&self.problem.grid, &skeleton, &radii);
        let stats = IterationStats {
            iteration: self.line,
            compliance: 0.0,
            volume_fraction: rasterize::design_volume_fraction(&self.problem.grid, &densities),
            // The local volume cap is a SIMP constraint; the growth engine
            // rejects the table outright.
            worst_local_fraction: None,
            max_change: 0.0,
            cg_iterations: Vec::new(),
            elapsed_s: self.start.elapsed().as_secs_f64(),
            growth: Some(GrowthProgress {
                phase,
                segments: skeleton.segment_count(),
                attractors_remaining: remaining,
            }),
        };
        self.reporter.iteration(&stats);
        self.reporter.densities(&stats, &densities);
    }
}

/// Warn when the problem is not symmetric under the transforms it declared.
///
/// Growth symmetry replicates **geometry**, whatever the loads do, and checking
/// that a whole problem really is symmetric is neither cheap nor robust - a
/// keepout rotated by a hair, a load vector that is symmetric only about a
/// different centre. What is cheap is the case that catches the real mistake:
/// every region's centre should be mapped onto another region of the same kind.
/// A region that is not is named, and the run carries on, because the stress
/// report at the end runs on the whole replicated structure with the real loads
/// and is the honest verdict either way.
fn warn_asymmetric_regions(
    symmetry: &Symmetry,
    tolerance_mm: f64,
    kind: &str,
    regions: &[(String, Vec3)],
    reporter: &dyn Reporter,
) {
    for (name, centre) in regions {
        for copy in 1..symmetry.sectors() {
            let image = symmetry.image(copy, *centre);
            let nearest = regions
                .iter()
                .map(|(_, other)| geometry::length(geometry::difference(*other, image)))
                .fold(f64::INFINITY, f64::min);
            if nearest <= tolerance_mm {
                continue;
            }
            reporter.note(&format!(
                "warning: [growth.symmetry] {} takes {kind} {name} to [{:.1}, {:.1}, {:.1}] mm, \
                 where this problem has no {kind}: the nearest one is {nearest:.1} mm away, \
                 against a tolerance of {tolerance_mm:.1} mm. The part will be symmetric anyway - \
                 symmetry replicates geometry, not loads - so read the stress report, which runs \
                 on the whole replicated structure with the real loads",
                symmetry.params().describe(),
                image[0],
                image[1],
                image[2]
            ));
            break;
        }
    }
}

/// The skeleton a growth run grew, before any of it was thickened.
///
/// Stages one to three of the engine produce this; the thickening and the
/// rasterization are functions of it and of the load magnitudes. It is public
/// because the invariant that makes the result a structure rather than a
/// decoration - with pruning on, no branch ends on nothing - is a property of
/// the skeleton, and a property nobody outside the crate can check is a promise
/// nobody has to keep.
#[derive(Debug)]
pub struct Growth {
    /// The strut skeleton, pruned unless `[growth] prune` is off.
    pub skeleton: Skeleton,
    /// The cells a branch is allowed to end on.
    pub anchors: AnchorSet,
    /// How close a tip has to come to an anchor to count as fused, in mm.
    pub fusion_tolerance_mm: f64,
    /// Guaranteed load paths routed from a load region to a support region.
    pub backbones: usize,
    /// Space colonization iterations performed.
    pub steps: usize,
    /// Why the colonization stopped.
    pub stop: GrowthStop,
    /// Attraction points scattered in total.
    pub scattered: usize,
    /// Of those, the ones seeded on a structural surface.
    pub surface_targets: usize,
    /// Surface targets no branch ever fused to.
    pub unreached_surfaces: usize,
    /// Attraction points a branch reached.
    pub consumed: usize,
    /// Nodes pruning removed.
    pub pruned_nodes: usize,
    /// The symmetry the skeleton is replicated by, or `None` when the run grew
    /// the whole domain. The skeleton above is always the **fundamental** one.
    pub symmetry: Option<Symmetry>,
    /// Design cells inside the fundamental domain: what the growth had to fill,
    /// and about `1 / sectors` of the problem's design cells - exactly that
    /// only when no cell centre lies on a sector boundary; see
    /// [`crate::engine::GrowthSymmetry::fundamental_design_cells`] for the
    /// odd-axis case.
    pub fundamental_design_cells: usize,
    regions: Vec<LoadRegion>,
    lines: usize,
    started: Instant,
}

impl Growth {
    /// Branch tips that end on nothing at all.
    ///
    /// Empty by construction when `[growth] prune` is on: a tip that ends in mid
    /// air carries no load, cannot be printed without support underneath it, and
    /// reads as a model that stopped half way.
    pub fn free_leaves(&self, grid: &Grid) -> Vec<usize> {
        anchor::free_leaves(
            &self.skeleton,
            &self.anchors,
            grid,
            self.fusion_tolerance_mm,
        )
    }
}

/// Resolve the growth controls a problem carries.
fn growth_params(problem: &Problem) -> Result<GrowthParams> {
    problem.growth.ok_or_else(|| {
        anyhow::anyhow!(
            "the growth engine was selected without resolved [growth] parameters; this is a \
             growforge bug, the problem builder resolves them for engine = \"{}\"",
            constants::GROWTH_ENGINE
        )
    })
}

/// Stages one to three: route the guaranteed load paths, grow the canopy onto
/// the structural surfaces, and remove whatever still ends on nothing.
///
/// With `[growth.symmetry]` set, all three run inside one fundamental domain
/// (see [`symmetry`]) and the skeleton that comes back is that sector's alone;
/// [`GrowthEngine::optimize`] replicates it. Three things about that are worth
/// having in one place:
///
/// * **A region belongs to the sector its centre is in, whole.** A load or
///   support region straddling the boundary is grown for once rather than
///   half-grown twice, and it carries its **full declared magnitude** in that
///   sector - there is no `1 / sectors` geometric share. That is invisible in a
///   problem with one load region, or one whose regions all straddle or all do
///   not, because the flow decides only the *relative* thickness of the struts
///   and the volume target sets the absolute scale; it shows in a problem
///   mixing the two, where the straddling region is weighted as if all of it
///   were in the sector and its struts come out thicker than their share of the
///   load, in every copy. The stress report runs on the whole replicated
///   structure with the real loads regardless.
/// * **Anchors are not clipped.** What a branch may end on is every region of
///   the problem, in whatever sector: a support patch that straddles the
///   boundary holds material on both sides of it. Only the attraction points
///   are confined, because a surface across the boundary is one this sector's
///   branches can never reach.
/// * **The skeleton is exact; the rasterized field may not be.** See
///   [`symmetry::Symmetry::maps_cell_centres`].
pub fn grow(problem: &Problem, reporter: &dyn Reporter) -> Result<Growth> {
    {
        let params = growth_params(problem)?;
        let grid = &problem.grid;
        // The fundamental domain of the declared symmetry, or the whole grid.
        // Everything that grows is confined to it; what a *region* is stays a
        // question about material, so the anchors below still cover the part
        // the copies will occupy.
        let symmetry = params.symmetry.map(|symmetry| {
            Symmetry::new(
                symmetry,
                &grid.bounds(),
                constants::GROWTH_SYMMETRY_BOUNDARY_EPSILON_VOXELS * grid.h,
            )
        });
        let domain = GrowthDomain::confined(grid, symmetry.as_ref());
        let min_radius = 0.5 * problem.optimization.min_feature_mm;
        // Every strut is at least this thick, so a tip this close to structural
        // material rasterizes into it. It is the one length that decides what
        // counts as fused, from the surface attraction points through the
        // pruning to the load attachments.
        let fusion_tolerance = constants::GROWTH_FUSION_RADII * min_radius;
        let mut progress = Progress {
            reporter,
            problem,
            symmetry: symmetry.as_ref(),
            start: Instant::now(),
            line: 0,
        };

        // 1. Backbones: the guaranteed load paths.
        let mut regions = load_regions(problem, &domain);
        if regions.is_empty() {
            bail!(
                "the growth engine needs at least one force or torque load to grow towards, and \
                 this problem has only gravity loads; add a load with a region, or run it with \
                 engine = \"{}\"",
                constants::DEFAULT_ENGINE
            );
        }
        let support_cells: Vec<Vec<usize>> = problem
            .supports
            .iter()
            .map(|s| domain.cells_touching_nodes(&s.nodes))
            .collect();

        // Everything a branch is allowed to end on: material by construction,
        // held by a support, or loaded directly. Built from every region, not
        // only the ones this sector grows for: a support patch that straddles
        // the boundary holds material on both sides of it, and a tip that ends
        // on the far half has still ended on a support.
        let mut anchors = AnchorSet::solid(grid);
        for cells in &support_cells {
            anchors.extend(cells);
        }
        for region in &regions {
            anchors.extend(&region.cells);
        }
        let fusion = anchor::Fusion {
            anchors: &anchors,
            grid,
            tolerance_mm: fusion_tolerance,
        };

        // A symmetric run routes backbones only for the regions its own sector
        // owns, which is the ones whose *centre* lies in the fundamental
        // domain: a region straddling the boundary belongs to the sector its
        // middle is in, and the copies carry the rest. The load a dropped
        // region applies is carried by the copy that owns it.
        //
        // A region that is owned is owned *whole*, and carries its whole
        // declared magnitude here even when most of it lies in another sector.
        // See `LoadRegion::magnitude_n` for what that costs and when.
        let routed_supports: Vec<&[usize]> = support_cells
            .iter()
            .filter(|cells| owned(symmetry.as_ref(), &domain, cells))
            .map(Vec::as_slice)
            .collect();
        if let Some(symmetry) = &symmetry {
            let tolerance = constants::GROWTH_SYMMETRY_REGION_TOLERANCE_VOXELS * grid.h;
            let supports = named_centres(
                &domain,
                problem.supports.iter().map(|s| s.index.to_string()),
                &support_cells,
            );
            warn_asymmetric_regions(symmetry, tolerance, "support", &supports, reporter);
            let load_cells: Vec<Vec<usize>> = regions.iter().map(|r| r.cells.clone()).collect();
            let loads = named_centres(
                &domain,
                regions
                    .iter()
                    .map(|r| format!("\"{}\" load {}", r.case, r.load)),
                &load_cells,
            );
            warn_asymmetric_regions(symmetry, tolerance, "load region", &loads, reporter);
            regions.retain(|region| owned(Some(symmetry), &domain, &region.cells));
            if regions.is_empty() {
                bail!(
                    "[growth.symmetry] {}: no load region has its centre in the fundamental \
                     domain, so the sector this run grows carries nothing. Every load is in a \
                     sector the copies replicate rather than in the one that is grown; move a \
                     load, change the planes, or drop the symmetry",
                    symmetry.params().describe()
                );
            }
            if routed_supports.is_empty() {
                bail!(
                    "[growth.symmetry] {}: no support region has its centre in the fundamental \
                     domain, so there is nowhere in the grown sector to plant a foot. Move a \
                     support, change the planes, or drop the symmetry",
                    symmetry.params().describe()
                );
            }
        }

        let mut skeleton = Skeleton::new();
        let backbones = grow_backbones(
            problem,
            &domain,
            &routed_supports,
            params.step_mm,
            &mut regions,
            &mut skeleton,
        )?;
        let uniform = |skeleton: &Skeleton| vec![min_radius; skeleton.len()];
        progress.emit(GrowthPhase::Backbone, &skeleton, 0, &uniform(&skeleton));

        // 2. Branching: the organic canopy, aimed at the structural surfaces.
        let mut rng = Pcg32::new(params.seed);
        let attractors = colonize::scatter(&domain, &fusion, &params, &mut rng);
        let surfaces = attractors.iter().filter(|a| a.on_surface).count();
        let scattered = attractors.len();
        let mut colonizer = Colonizer::new(attractors, &skeleton, &params, &fusion);
        let mut steps = 0;
        let mut stop = GrowthStop::Exhausted;
        while colonizer.remaining() > 0 {
            if steps >= params.max_steps {
                stop = GrowthStop::StepCap;
                break;
            }
            if colonizer.step(&mut skeleton, &domain, &params, &fusion) == 0 {
                stop = GrowthStop::Stalled;
                break;
            }
            steps += 1;
            if steps.is_multiple_of(constants::GROWTH_REPORT_INTERVAL_STEPS) {
                progress.emit(
                    GrowthPhase::Branching,
                    &skeleton,
                    colonizer.remaining(),
                    &uniform(&skeleton),
                );
            }
        }
        match stop {
            GrowthStop::Exhausted => reporter.note(&format!(
                "growth: every attraction point was reached or given up after {steps} steps ({} \
                 reached, {} unreachable)",
                colonizer.consumed(),
                colonizer.abandoned()
            )),
            GrowthStop::Stalled => reporter.note(&format!(
                "growth: no branch could advance after {steps} steps, {} attraction points out of \
                 reach",
                colonizer.remaining()
            )),
            GrowthStop::StepCap => reporter.note(&format!(
                "growth: stopped at the step cap of {}, {} attraction points left",
                params.max_steps,
                colonizer.remaining()
            )),
        }
        let unreached = colonizer.unreached_surfaces();
        if unreached > 0 {
            reporter.note(&format!(
                "growth: {unreached} of the {surfaces} structural surface targets were never \
                 reached; branches that were heading for them end on nothing and are {}",
                if params.prune {
                    "pruned"
                } else {
                    "kept, because [growth] prune is off"
                }
            ));
        }

        // 3. Pruning: everything that ends on nothing goes.
        let mut pruned_nodes = 0;
        if params.prune {
            let pruning = anchor::prune(&mut skeleton, &anchors, grid, fusion_tolerance);
            pruned_nodes = pruning.removed;
            for region in regions.iter_mut() {
                // A backbone tip is anchored by construction, so it always
                // survives; the map only moves it.
                region.backbone_tips = region
                    .backbone_tips
                    .iter()
                    .filter_map(|&tip| pruning.map[tip])
                    .collect();
            }
            debug_assert!(
                regions.iter().all(|r| !r.backbone_tips.is_empty()),
                "pruning removed a backbone tip, which is anchored on a load region"
            );
            progress.emit(
                GrowthPhase::Pruning,
                &skeleton,
                colonizer.remaining(),
                &uniform(&skeleton),
            );
            reporter.note(&format!(
                "growth: pruned {pruned_nodes} branch nodes that ended on nothing"
            ));
        }

        // What the growth had to fill. With no symmetry it is every design cell
        // of the problem; with one it is the sector's share of them, which is
        // the arithmetic behind `mass_fraction` still meaning the same thing -
        // see `GrowthEngine::optimize`.
        let fundamental_design_cells = grid
            .cells
            .iter()
            .enumerate()
            .filter(|(cell, kind)| **kind == CellKind::Design && domain.allows_cell(*cell))
            .count();
        // Read out before the symmetry moves into the result, which is what
        // the progress reporter borrowed it for.
        let (lines, started) = (progress.line, progress.start);

        Ok(Growth {
            skeleton,
            anchors,
            fusion_tolerance_mm: fusion_tolerance,
            backbones,
            steps,
            stop,
            scattered,
            surface_targets: surfaces,
            unreached_surfaces: unreached,
            consumed: colonizer.consumed(),
            pruned_nodes,
            symmetry,
            fundamental_design_cells,
            regions,
            lines,
            started,
        })
    }
}

/// True when the fundamental domain owns a region, which is what decides
/// whether a symmetric run routes a backbone for it.
///
/// The region's **centre** decides, so a region that straddles the boundary
/// belongs to the sector its middle is in and is grown for once rather than
/// half-grown twice. Without a symmetry every region is owned.
fn owned(symmetry: Option<&Symmetry>, domain: &GrowthDomain, cells: &[usize]) -> bool {
    match symmetry {
        None => true,
        Some(symmetry) => centroid(domain, cells).is_some_and(|centre| symmetry.contains(centre)),
    }
}

/// Pair up region names with region centres, dropping any region that selected
/// no material cell at all.
fn named_centres(
    domain: &GrowthDomain,
    names: impl Iterator<Item = String>,
    cells: &[Vec<usize>],
) -> Vec<(String, Vec3)> {
    names
        .zip(cells.iter())
        .filter_map(|(name, cells)| centroid(domain, cells).map(|centre| (name, centre)))
        .collect()
}

impl Engine for GrowthEngine {
    fn name(&self) -> &'static str {
        constants::GROWTH_ENGINE
    }

    fn optimize(&self, problem: &Problem, reporter: &dyn Reporter) -> Result<DensityField> {
        let params = growth_params(problem)?;
        let grid = &problem.grid;
        let min_radius = 0.5 * problem.optimization.min_feature_mm;

        let grown = grow(problem, reporter)?;
        // The four owned parts move out; everything else the summary needs is
        // a plain number still readable through `grown`.
        let Growth {
            skeleton,
            fusion_tolerance_mm,
            mut regions,
            symmetry,
            ..
        } = grown;
        // The progress lines carry on where the growth stages left off, so a
        // console or a viewer sees one continuous run.
        let mut progress = Progress {
            reporter,
            problem,
            symmetry: symmetry.as_ref(),
            start: grown.started,
            line: grown.lines,
        };

        // 4. Thickening: what each branch carries, and how thick that makes it.
        collect_attachments(&mut regions, &skeleton, grid, fusion_tolerance_mm);
        let fused_tips: usize = regions
            .iter()
            .map(|r| r.attachments.len().saturating_sub(r.backbone_tips.len()))
            .sum();
        let mut region_loads = Vec::new();
        for region in &regions {
            let share = region.magnitude_n / region.attachments.len() as f64;
            for attachment in &region.attachments {
                region_loads.push((*attachment, share));
            }
        }
        let total_load: f64 = regions.iter().map(|r| r.magnitude_n).sum();
        let loads = thicken::node_loads(
            &skeleton,
            &region_loads,
            constants::GROWTH_BRANCH_TIP_LOAD_FRACTION * total_load,
        );
        let flow = thicken::accumulate_flow(&skeleton, &loads);
        // The structure the volume target is measured on is the **whole** one:
        // the fundamental skeleton unioned with its copies. That is what makes
        // `mass_fraction` mean the same thing symmetric or not - the mean
        // density over every design cell of the problem. Measuring the sector
        // alone against the same fraction would come to the same number when
        // the sectors are equal (a fraction f of a volume V/N in each of N
        // sectors is a fraction f of V), but only the whole structure is
        // measured on exactly the cells `mass_fraction` is defined over, so
        // that is what is measured. One scale is bisected for all of the
        // copies, which is why they stay identical.
        let export = replicated(symmetry.as_ref(), &skeleton);
        let scaling = thicken::resolve_scale(
            &flow,
            params.murray_exponent,
            min_radius,
            params.max_radius_mm,
            problem.optimization.mass_fraction,
            |radii| {
                let radii = replicated_radii(symmetry.as_ref(), radii);
                rasterize::design_volume_fraction(
                    grid,
                    &rasterize::densities(grid, &export, &radii),
                )
            },
        );
        if let Some(achievable) = scaling.clamped {
            reporter.note(&format!(
                "warning: the strut radius clamps [{:.3}, {:.3}] mm cannot reach mass_fraction \
                 {:.4}; the design settles at {:.4} instead. Lower min_feature_mm to go thinner, \
                 raise [growth] max_radius_mm or attractor_per_cm3 to go heavier",
                min_radius, params.max_radius_mm, problem.optimization.mass_fraction, achievable
            ));
        }

        // 5. Rasterize the final structure.
        let radii = thicken::radii(
            &flow,
            scaling.scale,
            params.murray_exponent,
            min_radius,
            params.max_radius_mm,
        );
        let densities =
            rasterize::densities(grid, &export, &replicated_radii(symmetry.as_ref(), &radii));
        let volume_fraction = rasterize::design_volume_fraction(grid, &densities);
        progress.emit(GrowthPhase::Thickening, &skeleton, 0, &radii);

        // Roots carry no segment, so their radius is not part of the range. A
        // skeleton with no segment at all is possible in principle - a load
        // region that already touches its support routes a path one cell long -
        // and reports an empty range rather than an infinite one.
        let range = skeleton
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.parent.is_some())
            .map(|(index, _)| radii[index])
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), r| {
                (lo.min(r), hi.max(r))
            });
        let range = if range.0.is_finite() {
            range
        } else {
            (0.0, 0.0)
        };
        Ok(DensityField {
            densities,
            iterations: grown.steps,
            compliance: 0.0,
            initial_compliance: 0.0,
            volume_fraction,
            max_change: 0.0,
            // Growth has two ways of ending of its own accord - every attraction
            // point consumed, or nothing able to grow any further - and both of
            // them are "there was nothing left to do", which is what
            // [`StopReason::Converged`] means for an engine with no design
            // variables to settle. Only the step cap is stopping short.
            stop: match grown.stop.is_natural() {
                true => StopReason::Converged,
                false => StopReason::IterationCap,
            },
            overhang_residual: None,
            growth: Some(GrowthSummary {
                backbones: grown.backbones,
                // The whole structure's segments, copies included: it is what
                // was exported, and on a symmetric run it is `sectors` times
                // what the fundamental domain grew.
                segments: export.segment_count(),
                attractors: grown.scattered,
                consumed: grown.consumed,
                radius_range_mm: range,
                clamped_volume_fraction: scaling.clamped,
                surface_targets: grown.surface_targets,
                unreached_surfaces: grown.unreached_surfaces,
                pruned_nodes: grown.pruned_nodes,
                fused_tips,
                symmetry: symmetry.as_ref().map(|symmetry| GrowthSymmetry {
                    params: symmetry.params(),
                    fundamental_design_cells: grown.fundamental_design_cells,
                    exact_on_the_voxel_lattice: symmetry.maps_cell_centres(grid),
                }),
            }),
        })
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    //! Shared fixtures for the growth engine's own tests.

    use crate::config::GrowthParams;
    use crate::constants;
    use crate::geometry::Aabb;
    use crate::grid::{CellKind, Grid};

    /// A cube of design cells, `n` on a side, one millimetre per cell.
    pub fn design_grid(n: usize) -> Grid {
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

    /// The resolved growth controls a configuration with this `min_feature_mm`,
    /// this `mass_fraction` and no `[growth]` table would produce.
    pub fn growth_params(min_feature_mm: f64, mass_fraction: f64) -> GrowthParams {
        let kill = 0.5 * min_feature_mm * constants::GROWTH_KILL_MASS_FRACTION_MARGIN
            / mass_fraction.sqrt();
        GrowthParams {
            seed: constants::DEFAULT_GROWTH_SEED,
            attractor_per_cm3: constants::DEFAULT_ATTRACTOR_PER_CM3,
            attraction_radius_mm: constants::GROWTH_ATTRACTION_PER_KILL * kill,
            kill_radius_mm: kill,
            step_mm: constants::GROWTH_STEP_PER_MIN_FEATURE * min_feature_mm,
            murray_exponent: constants::DEFAULT_MURRAY_EXPONENT,
            max_radius_mm: constants::GROWTH_MAX_RADIUS_PER_MIN_FEATURE * min_feature_mm,
            max_steps: constants::DEFAULT_GROWTH_MAX_STEPS,
            prune: constants::DEFAULT_GROWTH_PRUNE,
            symmetry: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Axis, Config, SymmetryParams};
    use crate::report::SilentReporter;
    use crate::voids;
    use std::path::PathBuf;

    /// A block on four feet with a keepout column through the middle: small
    /// enough to run in a unit test, and still exercising every stage.
    fn canopy(extra: &str) -> String {
        format!(
            r#"
engine = "growth"

[project]
name = "canopy"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "petg"

[optimization]
mass_fraction = 0.15
min_feature_mm = 12.0

[output]
stl_path = "canopy.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [80.0, 80.0, 60.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 52.0]
max = [80.0, 80.0, 60.0]

[[keepout]]
shape = "cylinder"
p1 = [40.0, 40.0, -4.0]
p2 = [40.0, 40.0, 48.0]
radius = 14.0

[[supports]]
region = {{ shape = "box", min = [-1.0, -1.0, -1.0], max = [12.0, 12.0, 1.0] }}

[[supports]]
region = {{ shape = "box", min = [68.0, -1.0, -1.0], max = [81.0, 12.0, 1.0] }}

[[loadcases]]
name = "top"
[[loadcases.loads]]
type = "force"
region = {{ shape = "box", min = [-1.0, -1.0, 55.0], max = [81.0, 81.0, 61.0] }}
vector = [0.0, 0.0, -400.0]
{extra}
"#
        )
    }

    /// The same block on **four** feet, one per corner, which is the four-fold
    /// symmetric problem a user reported growing four different legs. The
    /// keepout column and the load are centred, so every region either sits on
    /// both mirror planes or has its three images among the other regions.
    fn four_legged(extra: &str) -> String {
        format!(
            r#"
engine = "growth"

[project]
name = "four_legged"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "petg"

[optimization]
mass_fraction = 0.15
min_feature_mm = 12.0

[output]
stl_path = "four_legged.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [80.0, 80.0, 60.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 52.0]
max = [80.0, 80.0, 60.0]

[[keepout]]
shape = "cylinder"
p1 = [40.0, 40.0, -4.0]
p2 = [40.0, 40.0, 48.0]
radius = 14.0

[[supports]]
region = {{ shape = "box", min = [-1.0, -1.0, -1.0], max = [12.0, 12.0, 1.0] }}

[[supports]]
region = {{ shape = "box", min = [68.0, -1.0, -1.0], max = [81.0, 12.0, 1.0] }}

[[supports]]
region = {{ shape = "box", min = [-1.0, 68.0, -1.0], max = [12.0, 81.0, 1.0] }}

[[supports]]
region = {{ shape = "box", min = [68.0, 68.0, -1.0], max = [81.0, 81.0, 1.0] }}

[[loadcases]]
name = "top"
[[loadcases.loads]]
type = "force"
region = {{ shape = "box", min = [-1.0, -1.0, 55.0], max = [81.0, 81.0, 61.0] }}
vector = [0.0, 0.0, -400.0]
{extra}
"#
        )
    }

    /// The four-fold mirror table, which is what the feature was asked for.
    const FOUR_FOLD: &str = "\n[growth.symmetry]\nkind = \"mirror\"\nplanes = [\"x\", \"y\"]\n";

    fn build(text: &str) -> Problem {
        let config = Config::parse(text).expect("parse");
        Problem::build(&config, &PathBuf::from(".")).expect("build")
    }

    fn run(text: &str) -> DensityField {
        GrowthEngine
            .optimize(&build(text), &SilentReporter)
            .expect("growth")
    }

    /// A load high up at one end, and a wide support patch on the floor whose
    /// near edge is much closer to it than its middle is. The shortest path to
    /// the patch *as a whole* therefore ends on the patch's rim; the shortest
    /// path to its middle does not.
    fn offset_support() -> String {
        r#"
engine = "growth"

[project]
name = "offset"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "petg"

[optimization]
mass_fraction = 0.2
min_feature_mm = 12.0

[output]
stl_path = "offset.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [80.0, 20.0, 40.0]

[[supports]]
region = { shape = "box", min = [-1.0, -1.0, -1.0], max = [33.0, 21.0, 1.0] }

[[loadcases]]
name = "corner"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [71.0, -1.0, 35.0], max = [81.0, 21.0, 41.0] }
vector = [0.0, 0.0, -200.0]
"#
        .to_string()
    }

    /// The defect a user reported from a top-down screenshot: the legs did not
    /// sit on their supports but hung off to one side.
    ///
    /// A support region that is uniformly reachable has every cell at the same
    /// cost, so which one the search settles is decided by the queue's tie
    /// break - a corner - and the thickened foot then overhangs its own
    /// footprint. The fixture here is the harder case: the patch's near edge is
    /// genuinely cheaper than its middle, and the foot still has to plant in the
    /// middle.
    #[test]
    fn a_backbone_plants_its_foot_in_the_middle_of_its_support_not_on_its_rim() {
        let problem = build(&offset_support());
        let domain = GrowthDomain::new(&problem.grid);
        let cells = domain.cells_touching_nodes(&problem.supports[0].nodes);
        assert!(cells.len() > 8, "the fixture needs a patch, not a point");

        // The centroid of the patch, worked out here rather than asked for.
        let mut centroid = [0.0f64; 3];
        for &cell in &cells {
            let centre = domain.cell_center(cell);
            for (slot, value) in centroid.iter_mut().zip(centre.iter()) {
                *slot += value / cells.len() as f64;
            }
        }

        let grown = grow(&problem, &SilentReporter).expect("grow");
        let root = grown
            .skeleton
            .nodes
            .iter()
            .find(|n| n.parent.is_none())
            .expect("a backbone root");
        let offset = (root.position[0] - centroid[0]).hypot(root.position[1] - centroid[1]);
        assert!(
            offset <= 0.5 * problem.grid.h,
            "the foot planted {offset:.2} mm from the middle of its support"
        );

        // It is emphatically not the cell the plain search would have settled:
        // the one nearest the load, on the patch's rim.
        let load_cells = domain.cells_touching_nodes(&problem.load_cases[0].loads[0].nodes);
        let load_centre = domain.cell_center(load_cells[0]);
        let nearest_to_load = cells
            .iter()
            .copied()
            .min_by(|a, b| {
                let da =
                    geometry::length(geometry::difference(domain.cell_center(*a), load_centre));
                let db =
                    geometry::length(geometry::difference(domain.cell_center(*b), load_centre));
                da.total_cmp(&db)
            })
            .expect("a nearest cell");
        let foot = domain
            .cell_at(root.position)
            .expect("the foot is in the grid");
        assert_ne!(
            foot, nearest_to_load,
            "the foot planted on the rim cell nearest the load"
        );

        // And it is an interior cell of the patch: every lateral neighbour of it
        // belongs to the patch too, which is what "not on the rim" means for a
        // footprint one cell thick.
        let (i, j, k) = problem.grid.cell_ijk(foot);
        for (di, dj) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let neighbour =
                problem
                    .grid
                    .cell_index((i as i32 + di) as usize, (j as i32 + dj) as usize, k);
            assert!(
                cells.contains(&neighbour),
                "the foot cell has a lateral neighbour outside the support patch"
            );
        }
    }

    /// The defect this engine was fixed for: a branch that stops in mid air.
    ///
    /// It is what the user saw as antler-like stubs reaching inward from the
    /// legs and ending on nothing, and it is a defect rather than a style,
    /// because such a branch carries no load, cannot be printed without support
    /// under it, and reads as a model that stopped half way.
    #[test]
    fn pruning_leaves_no_branch_ending_in_mid_air() {
        let problem = build(&canopy(""));
        let grown = grow(&problem, &SilentReporter).expect("growth");

        assert!(grown.pruned_nodes > 0, "the fixture grew nothing to prune");
        assert!(
            grown.free_leaves(&problem.grid).is_empty(),
            "{} branch tips still end on nothing",
            grown.free_leaves(&problem.grid).len()
        );
        // Every leaf ends on structural material: a support, a keepin surface
        // or a load region.
        let leaf = grown.skeleton.leaves();
        let fused = anchor::fused_leaves(
            &grown.skeleton,
            &grown.anchors,
            &problem.grid,
            grown.fusion_tolerance_mm,
        );
        assert_eq!(fused.len(), leaf.iter().filter(|l| **l).count());
        assert!(!fused.is_empty());
        // And the backbones are untouched by all of it.
        assert_eq!(grown.backbones, 2);
    }

    #[test]
    fn prune_false_keeps_the_free_tips_that_prune_true_removes() {
        let problem = build(&canopy(""));
        let decorative = build(&canopy("\n[growth]\nprune = false\n"));
        assert!(!decorative.growth.expect("params").prune);

        let kept = grow(&decorative, &SilentReporter).expect("growth");
        let pruned = grow(&problem, &SilentReporter).expect("growth");

        assert_eq!(kept.pruned_nodes, 0, "prune = false must remove nothing");
        assert!(
            !kept.free_leaves(&decorative.grid).is_empty(),
            "the decorative run has to keep the tips that end on nothing"
        );
        assert!(pruned.free_leaves(&problem.grid).is_empty());
        assert!(
            kept.skeleton.len() > pruned.skeleton.len(),
            "pruning did not make the skeleton smaller: {} against {}",
            kept.skeleton.len(),
            pruned.skeleton.len()
        );
        // Pruning is what lets the run hit its volume target here at all: the
        // dead branches are made of the same material as the live ones, and on
        // this fixture they alone already overrun the budget at the minimum
        // strut radius. Keeping them is a choice that costs mass, and the run
        // says so rather than quietly overfilling.
        let field = run(&canopy(""));
        assert_eq!(field.growth.expect("summary").clamped_volume_fraction, None);
        assert!((field.volume_fraction - 0.15).abs() <= 0.01 * 0.15);

        let decorative = run(&canopy("\n[growth]\nprune = false\n"));
        let clamped = decorative
            .growth
            .expect("summary")
            .clamped_volume_fraction
            .expect("the dead branches have to be reported as costing the budget");
        assert!(clamped > 0.15, "the extra branches cost nothing?");
        assert!((decorative.volume_fraction - clamped).abs() < 1e-12);
    }

    #[test]
    fn a_fused_branch_tip_takes_a_share_of_the_load_off_the_backbone() {
        let problem = build(&canopy(""));
        let grown = grow(&problem, &SilentReporter).expect("growth");
        let mut regions = grown.regions.clone();
        let skeleton = &grown.skeleton;

        // Backbone tips alone, which is what the engine used to share the load
        // over, against every place a branch actually fused to the load.
        let backbone_only: usize = regions.iter().map(|r| r.backbone_tips.len()).sum();
        collect_attachments(
            &mut regions,
            skeleton,
            &problem.grid,
            grown.fusion_tolerance_mm,
        );
        let attachments: usize = regions.iter().map(|r| r.attachments.len()).sum();
        assert!(
            attachments > backbone_only,
            "no branch tip fused to the load region: {attachments} attachments"
        );

        // The load a region carries is split over all of them, so the flow the
        // backbone tip has to carry is measurably smaller than it would be with
        // the backbones alone. Both fields are compared at the same node.
        let flow_with = |loads: &[(usize, f64)]| {
            thicken::accumulate_flow(skeleton, &thicken::node_loads(skeleton, loads, 0.0))
        };
        let region = &regions[0];
        let tip = region.backbone_tips[0];
        let shared: Vec<(usize, f64)> = region
            .attachments
            .iter()
            .map(|&n| (n, region.magnitude_n / region.attachments.len() as f64))
            .collect();
        let backbone: Vec<(usize, f64)> = region
            .backbone_tips
            .iter()
            .map(|&n| (n, region.magnitude_n / region.backbone_tips.len() as f64))
            .collect();
        assert!(
            flow_with(&shared)[tip] < flow_with(&backbone)[tip],
            "the fused tips took no load off the backbone"
        );
        // Nothing is lost: the roots still carry the whole region either way.
        let roots: f64 = (0..skeleton.len())
            .filter(|&n| skeleton.nodes[n].parent.is_none())
            .map(|n| flow_with(&shared)[n])
            .sum();
        assert!(
            (roots - region.magnitude_n).abs() < 1e-9 * region.magnitude_n,
            "the roots carry {roots} of a {} N region",
            region.magnitude_n
        );
    }

    #[test]
    fn a_grown_field_hits_its_volume_target_and_stays_out_of_the_keepout() {
        let problem = build(&canopy(""));
        let field = run(&canopy(""));
        let summary = field.growth.expect("a growth summary");

        assert_eq!(summary.backbones, 2, "one path per load and support pair");
        assert!(
            summary.segments > summary.backbones,
            "nothing branched: {summary:?}"
        );
        assert!(summary.attractors > 0 && summary.consumed > 0);
        assert_eq!(summary.clamped_volume_fraction, None);
        assert!(
            (field.volume_fraction - 0.15).abs() <= 0.01 * 0.15,
            "volume fraction {} missed the 0.15 target",
            field.volume_fraction
        );
        assert!(field.densities.iter().all(|d| (0.0..=1.0).contains(d)));
        assert!(field.compliance == 0.0 && field.overhang_residual.is_none());

        // The keepout column is empty and the keepin plate is solid.
        for (cell, kind) in problem.grid.cells.iter().enumerate() {
            match kind {
                crate::grid::CellKind::Void => assert_eq!(field.densities[cell], 0.0),
                crate::grid::CellKind::Solid => assert_eq!(field.densities[cell], 1.0),
                crate::grid::CellKind::Design => {}
            }
        }
        // Radii respect the clamps.
        let (low, high) = summary.radius_range_mm;
        assert!(low >= 6.0 - 1e-9 && high <= 36.0 + 1e-9, "{low} .. {high}");
    }

    #[test]
    fn the_same_seed_grows_the_same_field_and_a_different_one_does_not() {
        let baseline = run(&canopy(""));
        let again = run(&canopy(""));
        assert_eq!(
            baseline.densities, again.densities,
            "the same configuration must grow a bit-identical field"
        );

        let reseeded = run(&canopy("\n[growth]\nseed = 12345\n"));
        assert_eq!(reseeded.densities.len(), baseline.densities.len());
        assert_ne!(
            reseeded.densities, baseline.densities,
            "a different seed must grow a different structure"
        );
        // Both still meet the same target, which is what makes the seed a
        // stylistic choice rather than a structural one.
        assert!((reseeded.volume_fraction - 0.15).abs() <= 0.01 * 0.15);
    }

    #[test]
    fn every_load_region_stays_connected_to_a_support() {
        let problem = build(&canopy(""));
        let field = run(&canopy(""));
        let iso = problem.output.iso_level;
        let grid = &problem.grid;

        // Flood fill the material from the support cells with the same six
        // connectivity the cavity pass uses, and the load region has to be in
        // what it reaches.
        let solid = |cell: usize| crate::engine::cell_density(grid, &field.densities, cell) >= iso;
        let domain = GrowthDomain::new(grid);
        let mut reached = vec![false; grid.n_cells()];
        let mut stack: Vec<usize> = problem
            .supports
            .iter()
            .flat_map(|s| domain.cells_touching_nodes(&s.nodes))
            .filter(|&c| solid(c))
            .collect();
        assert!(!stack.is_empty(), "no support cell carries material");
        for &cell in &stack {
            reached[cell] = true;
        }
        while let Some(cell) = stack.pop() {
            let (i, j, k) = grid.cell_ijk(cell);
            for (di, dj, dk) in [
                (-1i32, 0i32, 0i32),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ] {
                let (ni, nj, nk) = (i as i32 + di, j as i32 + dj, k as i32 + dk);
                if ni < 0
                    || nj < 0
                    || nk < 0
                    || ni >= grid.nx as i32
                    || nj >= grid.ny as i32
                    || nk >= grid.nz as i32
                {
                    continue;
                }
                let next = grid.cell_index(ni as usize, nj as usize, nk as usize);
                if !reached[next] && solid(next) {
                    reached[next] = true;
                    stack.push(next);
                }
            }
        }

        for case in &problem.load_cases {
            for load in &case.loads {
                if load.nodes.is_empty() {
                    continue;
                }
                let cells = domain.cells_touching_nodes(&load.nodes);
                assert!(
                    cells.iter().any(|&c| reached[c]),
                    "the load region of \"{}\" is not connected to any support",
                    case.name
                );
            }
        }
        // And the field the pipeline would export encloses nothing unprintable
        // beyond what the void policy reports.
        assert!(voids::detect(grid, &field.densities, iso).len() < grid.n_cells());
    }

    #[test]
    fn a_load_walled_off_from_every_support_names_itself_in_the_error() {
        // A keepout slab across the whole domain between the load and the feet.
        let text = canopy("").replace(
            r#"[[keepout]]
shape = "cylinder"
p1 = [40.0, 40.0, -4.0]
p2 = [40.0, 40.0, 48.0]
radius = 14.0"#,
            r#"[[keepout]]
shape = "box"
min = [-1.0, -1.0, 20.0]
max = [81.0, 81.0, 32.0]"#,
        );
        let problem = build(&text);
        let error = GrowthEngine
            .optimize(&problem, &SilentReporter)
            .unwrap_err()
            .to_string();
        assert!(error.contains("\"top\""), "the case must be named: {error}");
        assert!(error.contains("load 1"), "the load must be named: {error}");
        assert!(error.contains("no path"), "unexpected error: {error}");
    }

    #[test]
    fn a_problem_with_only_gravity_is_refused_with_a_way_out() {
        let text = canopy("").replace(
            r#"[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [-1.0, -1.0, 55.0], max = [81.0, 81.0, 61.0] }
vector = [0.0, 0.0, -400.0]"#,
            "[[loadcases.loads]]\ntype = \"gravity\"",
        );
        let error = GrowthEngine
            .optimize(&build(&text), &SilentReporter)
            .unwrap_err()
            .to_string();
        assert!(error.contains("gravity"), "unexpected error: {error}");
        assert!(
            error.contains(constants::DEFAULT_ENGINE),
            "the error has to offer a way out: {error}"
        );
    }

    #[test]
    fn self_weight_thickens_the_structure_without_changing_the_target() {
        let plain = run(&canopy(""));
        let heavy = run(&canopy("[[loadcases.loads]]\ntype = \"gravity\"\n"));
        // Gravity adds to every region's magnitude, so the flow pattern moves,
        // but the volume target is met either way.
        assert!((heavy.volume_fraction - 0.15).abs() <= 0.01 * 0.15);
        assert_eq!(heavy.densities.len(), plain.densities.len());
    }

    /// A reporter that keeps every one-off note the engine produced.
    #[derive(Default)]
    struct Notes(std::sync::Mutex<Vec<String>>);

    impl Reporter for Notes {
        fn iteration(&self, _stats: &IterationStats) {}
        fn note(&self, message: &str) {
            self.0.lock().expect("notes").push(message.to_string());
        }
    }

    #[test]
    fn a_volume_target_the_clamps_cannot_reach_warns_and_carries_on() {
        // Struts 12 mm thick cannot be arranged into 2 % of this domain, so the
        // run has to say what it can reach and produce a design anyway.
        let text = canopy("").replace("mass_fraction = 0.15", "mass_fraction = 0.02");
        let notes = Notes::default();
        let field = GrowthEngine
            .optimize(&build(&text), &notes)
            .expect("a clamped run still has to produce a design");

        let achievable = field
            .growth
            .expect("a growth summary")
            .clamped_volume_fraction
            .expect("the clamp has to be reported");
        assert!(
            achievable > 0.02,
            "the clamp report {achievable} is not above the target it missed"
        );
        assert!((field.volume_fraction - achievable).abs() < 1e-12);

        let notes = notes.0.into_inner().expect("notes");
        let warning = notes
            .iter()
            .find(|n| n.starts_with("warning:"))
            .unwrap_or_else(|| panic!("no warning among {notes:?}"));
        assert!(warning.contains("mass_fraction"), "{warning}");
        assert!(warning.contains(&format!("{achievable:.4}")), "{warning}");
        assert!(warning.contains("min_feature_mm"), "{warning}");
    }

    #[test]
    fn a_progress_line_is_emitted_for_every_phase() {
        use crate::report::GrowthPhase;
        use std::sync::Mutex;

        #[derive(Default)]
        struct Trace(Mutex<Vec<(GrowthPhase, usize)>>);
        impl Reporter for Trace {
            fn iteration(&self, stats: &IterationStats) {
                let growth = stats.growth.expect("a growth run reports growth stats");
                assert_eq!(stats.compliance, 0.0);
                assert!(stats.cg_iterations.is_empty());
                self.0
                    .lock()
                    .expect("trace")
                    .push((growth.phase, growth.segments));
            }
            fn note(&self, _message: &str) {}
        }

        let trace = Trace::default();
        GrowthEngine
            .optimize(&build(&canopy("")), &trace)
            .expect("growth");
        let seen = trace.0.into_inner().expect("trace");
        assert_eq!(seen.first().map(|s| s.0), Some(GrowthPhase::Backbone));
        assert_eq!(seen.last().map(|s| s.0), Some(GrowthPhase::Thickening));
        for phase in [GrowthPhase::Branching, GrowthPhase::Pruning] {
            assert!(
                seen.iter().any(|s| s.0 == phase),
                "the {} phase never reported",
                phase.label()
            );
        }

        // The skeleton only ever grows until it is pruned, which is the one
        // stage that takes segments away; thickening changes no topology.
        let pruned_at = seen
            .iter()
            .position(|s| s.0 == GrowthPhase::Pruning)
            .expect("a pruning line");
        assert!(
            seen[..=pruned_at - 1].windows(2).all(|w| w[0].1 <= w[1].1),
            "the skeleton shrank before it was pruned: {seen:?}"
        );
        assert!(
            seen[pruned_at].1 < seen[pruned_at - 1].1,
            "pruning removed nothing from {:?}",
            seen[pruned_at - 1]
        );
        assert!(
            seen[pruned_at..].windows(2).all(|w| w[0].1 == w[1].1),
            "the skeleton changed after it was pruned: {seen:?}"
        );
    }

    /// Distance from a cell centre to the nearer of the two mirror planes of
    /// the four-fold fixture, whose domain centre is (40, 40).
    fn seam_distance(grid: &Grid, cell: usize) -> f64 {
        let (i, j, k) = grid.cell_ijk(cell);
        let centre = grid.cell_center(i, j, k);
        (centre[0] - 40.0).abs().min((centre[1] - 40.0).abs())
    }

    #[test]
    fn growth_stays_inside_the_fundamental_domain_and_the_copies_cover_the_rest() {
        let problem = build(&four_legged(FOUR_FOLD));
        let grown = grow(&problem, &SilentReporter).expect("growth");
        let symmetry = grown.symmetry.as_ref().expect("a resolved symmetry");
        assert_eq!(symmetry.sectors(), 4);

        // Nothing that grew left the quarter: every colonization step is
        // clamped at the boundary exactly as it is at a keepout.
        for (index, node) in grown.skeleton.nodes.iter().enumerate() {
            assert!(
                symmetry.contains(node.position),
                "node {index} grew to {:?}, outside the fundamental domain",
                node.position
            );
        }
        assert!(grown.skeleton.len() > 20, "the fixture grew almost nothing");

        // One backbone, not four: only the corner support whose centre is in
        // the quarter is routed to, and the copies stand on the other three.
        assert_eq!(grown.backbones, 1);
        assert_eq!(
            grow(&build(&four_legged("")), &SilentReporter)
                .expect("growth")
                .backbones,
            4,
            "the same problem without symmetry routes one backbone per foot"
        );

        let replicated = symmetry.replicate(&grown.skeleton);
        assert_eq!(replicated.len(), 4 * grown.skeleton.len());
        let domain = GrowthDomain::new(&problem.grid);
        for support in &problem.supports {
            let cells = domain.cells_touching_nodes(&support.nodes);
            assert!(
                replicated.nodes.iter().any(|node| domain
                    .cell_at(node.position)
                    .is_some_and(|cell| cells.contains(&cell))),
                "no strut of the replicated structure stands on support {}",
                support.index
            );
        }

        // The quarter really is a quarter of what has to be filled.
        assert_eq!(
            grown.fundamental_design_cells * 4,
            problem.counts.design,
            "the fundamental domain holds {} of {} design cells",
            grown.fundamental_design_cells,
            problem.counts.design
        );
    }

    /// The strongest statement the feature makes: the exported field **is**
    /// symmetric.
    ///
    /// The two bands are counted apart because only one of them could ever
    /// differ by more than the arithmetic. A cell further from a plane than any
    /// strut can reach sees only struts of its own copy, in the same order its
    /// image sees the images of them, so its density is its image's to the last
    /// bit. A cell within reach of a plane sees struts of two copies, and the
    /// smooth minimum that unions them is commutative but **not associative**,
    /// so the same distances arriving in a different order could come out a
    /// hair apart - bounded by the blend width, and measured at zero here. A
    /// failure in the seam band is that, and not a broken symmetry.
    #[test]
    fn a_symmetric_run_grows_a_symmetric_field() {
        let problem = build(&four_legged(FOUR_FOLD));
        let field = run(&four_legged(FOUR_FOLD));
        let grid = &problem.grid;
        let summary = field.growth.expect("a growth summary");
        assert!(summary.symmetry.is_some());

        // Beyond this distance from a plane, no strut of another copy is within
        // reach of a cell at all, so the fold order there cannot differ.
        let reach = summary.radius_range_mm.1
            + constants::GROWTH_SMOOTH_UNION_VOXELS * grid.h
            + constants::GROWTH_SURFACE_WIDTH_VOXELS * grid.h;
        let mut worst_seam = 0.0f64;
        let mut worst_far = 0.0f64;
        let mut far_cells = 0;
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let cell = grid.cell_index(i, j, k);
                    let far = seam_distance(grid, cell) > reach;
                    far_cells += usize::from(far);
                    for image in [
                        grid.cell_index(grid.nx - 1 - i, j, k),
                        grid.cell_index(i, grid.ny - 1 - j, k),
                        grid.cell_index(grid.nx - 1 - i, grid.ny - 1 - j, k),
                    ] {
                        let difference = (field.densities[cell] - field.densities[image]).abs();
                        if far {
                            worst_far = worst_far.max(difference);
                        } else {
                            worst_seam = worst_seam.max(difference);
                        }
                    }
                }
            }
        }
        assert!(far_cells > grid.n_cells() / 8, "only {far_cells} far cells");
        assert!(
            worst_far <= 1e-12,
            "a cell out of reach of the seam differs from its mirror by {worst_far}"
        );
        assert!(
            worst_seam <= 1e-12,
            "a cell in the seam band differs from its mirror by {worst_seam}"
        );
        // No cell of the union is over-full: two copies of the same strut are
        // still one strut, whatever they add up to before the clamp.
        assert!(field.densities.iter().all(|d| (0.0..=1.0).contains(d)));

        // The asymmetric run of the same problem is *not* symmetric, which is
        // the defect this feature exists for.
        let plain = run(&four_legged(""));
        let plain_worst = (0..grid.n_cells())
            .map(|cell| {
                let (i, j, k) = grid.cell_ijk(cell);
                let image = grid.cell_index(grid.nx - 1 - i, j, k);
                (plain.densities[cell] - plain.densities[image]).abs()
            })
            .fold(0.0f64, f64::max);
        assert!(
            plain_worst > 0.5,
            "the asymmetric run came out symmetric anyway ({plain_worst})"
        );
    }

    #[test]
    fn the_volume_target_is_met_over_the_whole_replicated_structure() {
        let problem = build(&four_legged(FOUR_FOLD));
        let field = run(&four_legged(FOUR_FOLD));
        let grid = &problem.grid;
        assert!(
            (field.volume_fraction - 0.15).abs() <= 0.01 * 0.15,
            "the replicated structure came out at {}",
            field.volume_fraction
        );

        // And the arithmetic behind it: the quarter that was grown is itself at
        // the target fraction *of its own quarter of the design cells*, which
        // is why growing one sector to `mass_fraction` fills the whole to
        // `mass_fraction`.
        let symmetry = Symmetry::new(
            problem.growth.expect("params").symmetry.expect("symmetry"),
            &grid.bounds(),
            constants::GROWTH_SYMMETRY_BOUNDARY_EPSILON_VOXELS * grid.h,
        );
        let mut total = 0.0;
        let mut cells = 0usize;
        for cell in 0..grid.n_cells() {
            let (i, j, k) = grid.cell_ijk(cell);
            if grid.cells[cell] != CellKind::Design || !symmetry.contains(grid.cell_center(i, j, k))
            {
                continue;
            }
            total += field.densities[cell];
            cells += 1;
        }
        let sector_fraction = total / cells as f64;
        assert_eq!(cells * 4, problem.counts.design);
        assert!(
            (sector_fraction - field.volume_fraction).abs() < 1e-4,
            "the sector is at {sector_fraction} where the whole is at {}",
            field.volume_fraction
        );
    }

    #[test]
    fn the_same_symmetric_configuration_grows_the_same_field_twice() {
        let baseline = run(&four_legged(FOUR_FOLD));
        let again = run(&four_legged(FOUR_FOLD));
        assert_eq!(
            baseline.densities, again.densities,
            "a symmetric configuration has to grow a bit-identical field"
        );
        // A different symmetry is a different structure, and no symmetry at all
        // is the default this feature may not have changed.
        let halved = run(&four_legged(
            "\n[growth.symmetry]\nkind = \"mirror\"\nplanes = [\"x\"]\n",
        ));
        assert_ne!(halved.densities, baseline.densities);
        assert_ne!(run(&four_legged("")).densities, baseline.densities);
    }

    /// A strut that ends **on** a mirror plane meets its own reflection there:
    /// the two capsules share the endpoint exactly, so what comes out is one
    /// body rather than two that touch.
    #[test]
    fn a_strut_that_ends_on_the_symmetry_plane_meets_its_twin_as_one_body() {
        let grid = fixtures::design_grid(20);
        let symmetry = Symmetry::new(
            SymmetryParams::Mirror {
                first: Axis::X,
                second: None,
            },
            &grid.bounds(),
            constants::GROWTH_SYMMETRY_BOUNDARY_EPSILON_VOXELS * grid.h,
        );
        assert_eq!(symmetry.center()[0], 10.0);

        let mut skeleton = Skeleton::new();
        let root = skeleton.add_root([2.5, 10.5, 2.5]);
        // Ending exactly on the plane, which is where the twin is waiting.
        skeleton.add_child(root, [10.0, 10.5, 10.5]);
        let replicated = symmetry.replicate(&skeleton);
        let densities = rasterize::densities(&grid, &replicated, &symmetry.repeat(&[0.0, 2.0]));

        assert!(
            densities.iter().all(|d| (0.0..=1.0).contains(d)),
            "the union of a strut and its twin exceeded a full cell"
        );
        let bodies = voids::solid_bodies(&grid, &densities, 0.5);
        assert_eq!(
            bodies.len(),
            1,
            "the strut and its reflection came out as {} separate bodies",
            bodies.len()
        );
        // The meeting point is solid, and so is a cell on each side of it.
        for x in [9.5, 10.5] {
            let cell = grid.cell_index(x as usize, 10, 10).min(grid.n_cells() - 1);
            assert!(densities[cell] > 0.9, "the seam is open at x = {x}");
        }
    }

    #[test]
    fn a_lopsided_problem_warns_under_the_symmetry_it_declares() {
        // One foot moved off its corner, so the images of the other three land
        // where no support is.
        let lopsided = four_legged(FOUR_FOLD).replace(
            r#"region = { shape = "box", min = [68.0, 68.0, -1.0], max = [81.0, 81.0, 1.0] }"#,
            r#"region = { shape = "box", min = [40.0, 68.0, -1.0], max = [53.0, 81.0, 1.0] }"#,
        );
        let notes = Notes::default();
        GrowthEngine
            .optimize(&build(&lopsided), &notes)
            .expect("a lopsided problem still grows");
        let notes = notes.0.into_inner().expect("notes");
        let warning = notes
            .iter()
            .find(|n| n.contains("[growth.symmetry]"))
            .unwrap_or_else(|| panic!("no symmetry warning among {notes:?}"));
        assert!(warning.starts_with("warning:"), "{warning}");
        assert!(
            warning.contains("support"),
            "the region must be named: {warning}"
        );
        assert!(warning.contains("stress report"), "{warning}");

        // And the symmetric problem says nothing at all.
        let quiet = Notes::default();
        GrowthEngine
            .optimize(&build(&four_legged(FOUR_FOLD)), &quiet)
            .expect("growth");
        let quiet = quiet.0.into_inner().expect("notes");
        assert!(
            !quiet.iter().any(|n| n.contains("[growth.symmetry]")),
            "a symmetric problem warned anyway: {quiet:?}"
        );
    }

    #[test]
    fn a_symmetry_no_load_or_support_lies_under_is_refused_with_a_reason() {
        // Both feet moved to the far side of the plane, so the half that is
        // grown has nowhere to plant one.
        let text = canopy("\n[growth.symmetry]\nkind = \"mirror\"\nplanes = [\"y\"]\n")
            .replace(
                "min = [-1.0, -1.0, -1.0], max = [12.0, 12.0, 1.0]",
                "min = [-1.0, 68.0, -1.0], max = [12.0, 81.0, 1.0]",
            )
            .replace(
                "min = [68.0, -1.0, -1.0], max = [81.0, 12.0, 1.0]",
                "min = [68.0, 68.0, -1.0], max = [81.0, 81.0, 1.0]",
            );
        let error = GrowthEngine
            .optimize(&build(&text), &SilentReporter)
            .unwrap_err()
            .to_string();
        assert!(error.contains("[growth.symmetry]"), "{error}");
        assert!(error.contains("support"), "{error}");
        assert!(error.contains("mirror across y"), "{error}");
    }
}
