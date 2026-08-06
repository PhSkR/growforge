//! Space colonization: the organic half of the skeleton.
//!
//! Runions et al., "Modeling Trees with a Space Colonization Algorithm",
//! Eurographics Workshop on Natural Phenomena, 2007. Attraction points are
//! scattered through the free space; every one of them pulls on the branch node
//! nearest to it, each pulled node steps towards the average of the directions
//! pulling on it, and a point a branch has reached is consumed. The result
//! looks grown rather than drawn because the branching pattern comes out of the
//! space that was there to fill, not out of a rule about angles.
//!
//! Two properties of the growth are what make this cheap. Nodes are only ever
//! added, never moved or removed, so the distance from an attractor to its
//! nearest node can only fall: it is computed once against the initial skeleton
//! and then updated against new nodes alone. And attraction points never move,
//! so one spatial hash built at the start serves the whole run.

use crate::config::GrowthParams;
use crate::constants;
use crate::engine::growth::anchor::Fusion;
use crate::engine::growth::domain::GrowthDomain;
use crate::engine::growth::prng::Pcg32;
use crate::engine::growth::skeleton::Skeleton;
use crate::geometry::{self, Vec3};
use crate::grid::CellKind;

/// One attraction point, and how close a branch has to come before it counts as
/// reached.
///
/// The two kinds of point differ only in that distance, and the difference is
/// the whole of what makes growth *connect* rather than wander. An interior
/// point is consumed from a whole kill radius away: it has done its job once a
/// branch is heading through its neighbourhood, and holding on any longer would
/// only drag every branch onto every point. A point seeded on a structural
/// surface is consumed only when a branch has actually fused to that surface, so
/// a branch aimed at one keeps growing until it arrives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attractor {
    /// Where the point sits, in millimetres.
    pub position: Vec3,
    /// Distance at which a branch node consumes it.
    pub kill_radius_mm: f64,
    /// True when the point was seeded on a structural surface.
    pub on_surface: bool,
}

/// Scatter attraction points through the domain.
///
/// Two groups come back, in this order:
///
/// * **Surface points**, one per patch of the fusion test's anchor set: the
///   keepin cells, support region cells and load region cells a branch is
///   allowed to end on. Their kill radius is the fusion tolerance, so they pull
///   a branch all the way in.
/// * **Interior points**, scattered through the *design* cells by rejection
///   sampling, which is exact whatever shape the domain has. A keepout never
///   holds one, so nothing pulls a branch at a wall. Their kill radius is the
///   configured one, and they are what makes the routing organic rather than
///   radial.
///
/// Both kinds are confined to whatever `domain` allows, which for a symmetric
/// run is the fundamental domain: a surface a branch could never reach across
/// the sector boundary would otherwise pull one at it for the whole step
/// budget, and interior points outside the sector would be attractors for a
/// structure that is grown elsewhere. The interior count is taken from the
/// design cells the domain allows rather than from all of them, so
/// `attractor_per_cm3` means the same density of points in a sector as it does
/// in a whole domain - which is what makes the replicated structure come out at
/// the same density as an asymmetric one.
///
/// A domain that is a vanishing fraction of its own bounding box simply ends up
/// with fewer interior points than asked for, after
/// [`constants::GROWTH_ATTRACTOR_PLACEMENT_ATTEMPTS`] tries each; a sector is a
/// mild case of exactly that.
pub fn scatter(
    domain: &GrowthDomain,
    fusion: &Fusion,
    params: &GrowthParams,
    rng: &mut Pcg32,
) -> Vec<Attractor> {
    let grid = domain.grid();
    let spacing = constants::GROWTH_SURFACE_ATTRACTOR_SPACING_KILLS * params.kill_radius_mm;
    let mut points: Vec<Attractor> = fusion
        .anchors
        .thinned(grid, spacing)
        .into_iter()
        .filter(|position| domain.allows_point(*position))
        .map(|position| Attractor {
            position,
            kill_radius_mm: fusion.tolerance_mm,
            on_surface: true,
        })
        .collect();

    let design_cells = grid
        .cells
        .iter()
        .enumerate()
        .filter(|(cell, kind)| **kind == CellKind::Design && domain.allows_cell(*cell))
        .count();
    let design_volume_cm3 = design_cells as f64 * grid.cell_volume() / constants::MM3_PER_CM3;
    let wanted = (params.attractor_per_cm3 * design_volume_cm3).round() as usize;

    let bounds = grid.bounds();
    points.reserve(wanted);
    for _ in 0..wanted {
        for _ in 0..constants::GROWTH_ATTRACTOR_PLACEMENT_ATTEMPTS {
            let candidate = [
                rng.range(bounds.min[0], bounds.max[0]),
                rng.range(bounds.min[1], bounds.max[1]),
                rng.range(bounds.min[2], bounds.max[2]),
            ];
            if domain.cell_at(candidate).is_some_and(|cell| {
                grid.cells[cell] == CellKind::Design && domain.allows_cell(cell)
            }) {
                points.push(Attractor {
                    position: candidate,
                    kill_radius_mm: params.kill_radius_mm,
                    on_surface: false,
                });
                break;
            }
        }
    }
    points
}

/// A uniform spatial hash over the attraction points, bucketed at the
/// attraction radius so the points within reach of a node are in the 27 buckets
/// around it.
#[derive(Debug)]
struct AttractorHash {
    origin: Vec3,
    cell: f64,
    counts: [i64; 3],
    buckets: Vec<Vec<usize>>,
}

impl AttractorHash {
    fn new(points: &[Attractor], cell: f64) -> AttractorHash {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for point in points {
            for d in 0..3 {
                min[d] = min[d].min(point.position[d]);
                max[d] = max[d].max(point.position[d]);
            }
        }
        let mut counts = [1i64; 3];
        for d in 0..3 {
            if min[d].is_finite() {
                counts[d] = (((max[d] - min[d]) / cell).floor() as i64 + 1).max(1);
            } else {
                min[d] = 0.0;
            }
        }
        let mut hash = AttractorHash {
            origin: min,
            cell,
            counts,
            buckets: vec![Vec::new(); (counts[0] * counts[1] * counts[2]) as usize],
        };
        for (index, point) in points.iter().enumerate() {
            let bucket = hash.bucket_of(point.position);
            hash.buckets[bucket].push(index);
        }
        hash
    }

    fn coords_of(&self, p: Vec3) -> [i64; 3] {
        let mut ijk = [0i64; 3];
        for d in 0..3 {
            let local = ((p[d] - self.origin[d]) / self.cell).floor() as i64;
            ijk[d] = local.clamp(0, self.counts[d] - 1);
        }
        ijk
    }

    fn bucket_of(&self, p: Vec3) -> usize {
        let ijk = self.coords_of(p);
        (ijk[0] + self.counts[0] * (ijk[1] + self.counts[1] * ijk[2])) as usize
    }

    /// Call `visit` on every attractor within one bucket of `p`.
    fn near(&self, p: Vec3, mut visit: impl FnMut(usize)) {
        let ijk = self.coords_of(p);
        for dk in -1i64..=1 {
            for dj in -1i64..=1 {
                for di in -1i64..=1 {
                    let (i, j, k) = (ijk[0] + di, ijk[1] + dj, ijk[2] + dk);
                    if i < 0
                        || j < 0
                        || k < 0
                        || i >= self.counts[0]
                        || j >= self.counts[1]
                        || k >= self.counts[2]
                    {
                        continue;
                    }
                    let bucket = (i + self.counts[0] * (j + self.counts[1] * k)) as usize;
                    for &attractor in &self.buckets[bucket] {
                        visit(attractor);
                    }
                }
            }
        }
    }

    /// Drop retired attractors so later passes stop walking over them.
    fn compact(&mut self, live: impl Fn(usize) -> bool) {
        for bucket in self.buckets.iter_mut() {
            bucket.retain(|&a| live(a));
        }
    }
}

/// Why space colonization stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStop {
    /// Every attraction point was reached and consumed.
    Exhausted,
    /// Nothing could grow any further: no live point is within reach of a node
    /// that can still step, or every remaining step is blocked.
    Stalled,
    /// `[growth] max_steps` was hit.
    StepCap,
}

impl GrowthStop {
    /// True when growth ended of its own accord rather than on the cap.
    pub fn is_natural(self) -> bool {
        self != GrowthStop::StepCap
    }
}

/// What became of an attraction point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PointState {
    /// Still pulling on a branch.
    Live,
    /// A branch came within its kill radius.
    Reached,
    /// No branch got any closer to it for
    /// [`patience_steps`] steps in a row, so it was given up as unreachable.
    Abandoned,
}

/// The growing front: attraction points, what they are nearest to, and which
/// nodes are still allowed to step.
#[derive(Debug)]
pub struct Colonizer {
    /// The scattered attraction points.
    pub points: Vec<Attractor>,
    state: Vec<PointState>,
    /// Nearest node that can still grow, which is the one the point pulls on.
    puller_node: Vec<usize>,
    /// How far away that node is.
    puller_distance: Vec<f64>,
    stagnant_steps: Vec<usize>,
    consecutive_failures: Vec<usize>,
    /// Nodes that have fused to structural material. A branch that has arrived
    /// has done its job and stops: letting it grow on would only send it
    /// crawling along the surface it just reached, adding members that connect
    /// nothing to nothing.
    arrived: Vec<bool>,
    hash: AttractorHash,
    remaining: usize,
}

impl Colonizer {
    /// Bind a set of attraction points to a starting skeleton.
    pub fn new(
        points: Vec<Attractor>,
        skeleton: &Skeleton,
        params: &GrowthParams,
        fusion: &Fusion,
    ) -> Colonizer {
        let hash = AttractorHash::new(&points, params.attraction_radius_mm);
        let count = points.len();
        let arrived: Vec<bool> = (0..skeleton.len())
            .map(|n| fusion.fused(skeleton.position(n)))
            .collect();
        let mut colonizer = Colonizer {
            state: vec![PointState::Live; count],
            puller_node: vec![usize::MAX; count],
            puller_distance: vec![f64::INFINITY; count],
            stagnant_steps: vec![0; count],
            consecutive_failures: vec![0; skeleton.len()],
            arrived,
            remaining: count,
            points,
            hash,
        };
        // The starting skeleton is the backbones; every attractor has to see all
        // of it before the first step is taken. A surface point the backbone
        // already fused to is reached here and never pulls on anything.
        for node in 0..skeleton.len() {
            let growable = !colonizer.arrived[node];
            colonizer.observe_node(node, skeleton.position(node), growable);
        }
        colonizer.compact();
        colonizer
    }

    /// Attraction points still pulling on a branch.
    pub fn remaining(&self) -> usize {
        self.remaining
    }

    /// Attraction points a branch actually reached.
    pub fn consumed(&self) -> usize {
        self.state
            .iter()
            .filter(|s| **s == PointState::Reached)
            .count()
    }

    /// Attraction points given up as unreachable.
    pub fn abandoned(&self) -> usize {
        self.state
            .iter()
            .filter(|s| **s == PointState::Abandoned)
            .count()
    }

    /// Surface points no branch ever fused to.
    ///
    /// Each of them is a structural surface the growth could not connect to, so
    /// the count is what the engine reports when it explains a stall.
    pub fn unreached_surfaces(&self) -> usize {
        self.points
            .iter()
            .zip(self.state.iter())
            .filter(|(point, state)| point.on_surface && **state != PointState::Reached)
            .count()
    }

    /// Show one node to the attractors near it.
    ///
    /// Two separate things happen here, and keeping them apart is what stops a
    /// branch that has already arrived from capturing a point that some other
    /// branch could still grow to:
    ///
    /// * **Reaching.** *Every* node retires a point it comes within the kill
    ///   radius of, arrived or not - the structure is there, whether or not it
    ///   can grow any further.
    /// * **Pulling.** Only a node that can still grow competes to be the one a
    ///   point pulls on. A point whose nearest node has arrived would otherwise
    ///   hang off that node for ever, pulling on something that will never move,
    ///   and the surface it sits on would never get its branch.
    fn observe_node(&mut self, node: usize, position: Vec3, growable: bool) {
        let points = &self.points;
        let state = &mut self.state;
        let puller_node = &mut self.puller_node;
        let puller_distance = &mut self.puller_distance;
        let stagnant_steps = &mut self.stagnant_steps;
        let remaining = &mut self.remaining;
        self.hash.near(position, |attractor| {
            if state[attractor] != PointState::Live {
                return;
            }
            let distance =
                geometry::length(geometry::difference(points[attractor].position, position));
            if distance <= points[attractor].kill_radius_mm {
                state[attractor] = PointState::Reached;
                *remaining -= 1;
                return;
            }
            if growable && distance < puller_distance[attractor] {
                puller_distance[attractor] = distance;
                puller_node[attractor] = node;
                // Something got closer, so the point is not stuck after all.
                stagnant_steps[attractor] = 0;
            }
        });
    }

    /// Give up on every attractor no branch has been getting closer to.
    fn abandon_stalled(&mut self, params: &GrowthParams) {
        let patience = patience_steps(params);
        for attractor in 0..self.points.len() {
            if self.state[attractor] != PointState::Live {
                continue;
            }
            self.stagnant_steps[attractor] += 1;
            if self.stagnant_steps[attractor] >= patience {
                self.state[attractor] = PointState::Abandoned;
                self.remaining -= 1;
            }
        }
        self.compact();
    }

    /// Drop retired attractors so later passes stop walking over them.
    fn compact(&mut self) {
        let state = &self.state;
        self.hash
            .compact(|attractor| state[attractor] == PointState::Live);
    }

    /// Advance every node an attractor is pulling on by one step.
    ///
    /// Returns the number of nodes that were added, which is zero exactly when
    /// growth has stalled.
    pub fn step(
        &mut self,
        skeleton: &mut Skeleton,
        domain: &GrowthDomain,
        params: &GrowthParams,
        fusion: &Fusion,
    ) -> usize {
        let node_count = skeleton.len();
        self.consecutive_failures.resize(node_count, 0);

        // Accumulate the pull on each node. An attractor only ever pulls on the
        // node nearest to it, and only from within the attraction radius.
        let mut pull = vec![[0.0f64; 3]; node_count];
        let mut pulled = vec![false; node_count];
        for attractor in 0..self.points.len() {
            if self.state[attractor] != PointState::Live
                || self.puller_distance[attractor] > params.attraction_radius_mm
            {
                continue;
            }
            let node = self.puller_node[attractor];
            if node == usize::MAX
                || self.consecutive_failures[node] >= constants::GROWTH_MAX_FAILED_STEPS
            {
                continue;
            }
            let offset =
                geometry::difference(self.points[attractor].position, skeleton.position(node));
            let length = geometry::length(offset);
            if length == 0.0 {
                continue;
            }
            let unit = geometry::scale(offset, 1.0 / length);
            for (slot, component) in pull[node].iter_mut().zip(unit.iter()) {
                *slot += component;
            }
            pulled[node] = true;
        }

        let mut grown = Vec::new();
        for node in 0..node_count {
            if !pulled[node] {
                continue;
            }
            let direction = pull[node];
            let magnitude = geometry::length(direction);
            if magnitude == 0.0 {
                // Attractors pulling in exactly opposing directions: the node
                // has nowhere to go, which counts as a blocked step.
                self.consecutive_failures[node] += 1;
                continue;
            }
            let from = skeleton.position(node);
            let unit = geometry::scale(direction, 1.0 / magnitude);
            match first_valid_step(domain, from, unit, params.step_mm) {
                Some(position) => {
                    self.consecutive_failures[node] = 0;
                    let child = skeleton.add_child(node, position);
                    self.consecutive_failures.push(0);
                    self.arrived.push(fusion.fused(position));
                    grown.push((child, position));
                }
                None => self.consecutive_failures[node] += 1,
            }
        }

        for (node, position) in &grown {
            let growable = !self.arrived[*node];
            self.observe_node(*node, *position, growable);
        }
        // Order matters: a point something got closer to this step has had its
        // patience reset by `observe_node`, and a point a branch arrived at was
        // retired there too, before it can be counted as stalled.
        self.abandon_stalled(params);
        grown.len()
    }
}

/// Steps an attraction point waits for something to come nearer before it is
/// given up: the time a branch needs to cross the whole attraction radius, so a
/// point at the far edge of its own reach is never abandoned while a branch is
/// still on its way to it. Never below one, whatever the configuration.
fn patience_steps(params: &GrowthParams) -> usize {
    let travel = params.attraction_radius_mm / params.step_mm;
    (constants::GROWTH_ATTRACTOR_PATIENCE_TRAVELS * travel)
        .ceil()
        .max(1.0) as usize
}

/// The first of the candidate step directions whose segment stays inside the
/// allowed cells.
///
/// The preferred direction is tried first; if it is blocked the step is
/// deflected by projecting it onto a coordinate plane, dropping the largest
/// component first, which is the component most likely to be the one running
/// into the obstacle. A step that survives none of the four is blocked, and
/// [`constants::GROWTH_MAX_FAILED_STEPS`] of those in a row end the branch.
fn first_valid_step(domain: &GrowthDomain, from: Vec3, direction: Vec3, step: f64) -> Option<Vec3> {
    let mut order = [0usize, 1, 2];
    order.sort_by(|a, b| direction[*b].abs().total_cmp(&direction[*a].abs()));

    for attempt in 0..4 {
        let mut candidate = direction;
        if attempt > 0 {
            candidate[order[attempt - 1]] = 0.0;
        }
        let magnitude = geometry::length(candidate);
        if magnitude == 0.0 {
            continue;
        }
        let target = geometry::sum(from, geometry::scale(candidate, step / magnitude));
        if domain.segment_inside(from, target) {
            return Some(target);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::growth::anchor::AnchorSet;
    use crate::engine::growth::fixtures::{design_grid, growth_params};
    use crate::geometry::Vec3;

    /// Defaults for a part with this feature size at a mass fraction of 0.25,
    /// which is where the fixtures below sit.
    fn params_for(min_feature: f64) -> GrowthParams {
        growth_params(min_feature, 0.25)
    }

    /// Interior points only, for the tests that are about the stepping rather
    /// than about where the points came from.
    fn interior(points: Vec<Vec3>, kill_radius_mm: f64) -> Vec<Attractor> {
        points
            .into_iter()
            .map(|position| Attractor {
                position,
                kill_radius_mm,
                on_surface: false,
            })
            .collect()
    }

    #[test]
    fn interior_attractors_land_only_in_design_cells_and_surface_ones_only_on_anchors() {
        let mut grid = design_grid(10);
        // A keepout slab and a forced solid slab. The keepout may never hold a
        // point of either kind; the forced solid one is a structural surface, so
        // it holds surface points and no interior ones.
        for k in 0..10 {
            for j in 0..10 {
                for i in 0..3 {
                    let cell = grid.cell_index(i, j, k);
                    grid.cells[cell] = CellKind::Void;
                }
                for i in 7..10 {
                    let cell = grid.cell_index(i, j, k);
                    grid.cells[cell] = CellKind::Solid;
                }
            }
        }
        let domain = GrowthDomain::new(&grid);
        let anchors = AnchorSet::solid(&grid);
        let fusion = Fusion {
            anchors: &anchors,
            grid: &grid,
            tolerance_mm: 1.0,
        };
        let mut params = params_for(2.0);
        params.attractor_per_cm3 = 5000.0;
        let mut rng = Pcg32::new(1);
        let points = scatter(&domain, &fusion, &params, &mut rng);

        assert!(points.iter().any(|p| p.on_surface), "no surface targets");
        assert!(points.iter().any(|p| !p.on_surface), "no interior points");
        for point in &points {
            let cell = domain.cell_at(point.position).expect("inside the block");
            assert_ne!(
                grid.cells[cell],
                CellKind::Void,
                "a point landed in the keepout at {:?}",
                point.position
            );
            if point.on_surface {
                assert!(
                    anchors.contains(cell),
                    "a surface point landed off the anchors at {:?}",
                    point.position
                );
                assert_eq!(point.kill_radius_mm, 1.0, "surface points fuse, not pass");
            } else {
                assert_eq!(grid.cells[cell], CellKind::Design);
                assert_eq!(point.kill_radius_mm, params.kill_radius_mm);
                assert!(
                    point.position[0] >= 3.0 && point.position[0] < 7.0,
                    "point {:?} left the design slab",
                    point.position
                );
            }
        }

        // The same seed scatters the same points; a different one does not.
        let mut again = Pcg32::new(1);
        assert_eq!(scatter(&domain, &fusion, &params, &mut again), points);
        let mut other = Pcg32::new(2);
        assert_ne!(scatter(&domain, &fusion, &params, &mut other), points);
    }

    /// A symmetric run may only scatter inside its own sector: an attractor on
    /// the far side of the plane is a target for the copy, not for what is
    /// being grown.
    #[test]
    fn a_confined_domain_scatters_only_inside_its_own_sector() {
        use crate::config::{Axis, SymmetryParams};
        use crate::engine::growth::symmetry::Symmetry;

        let mut grid = design_grid(10);
        // A solid cell in each half, so there is a surface target on both
        // sides of the plane at x = 5.
        for i in [1, 8] {
            let cell = grid.cell_index(i, 5, 5);
            grid.cells[cell] = CellKind::Solid;
        }
        let anchors = AnchorSet::solid(&grid);
        let fusion = Fusion {
            anchors: &anchors,
            grid: &grid,
            tolerance_mm: 0.5,
        };
        let mut params = params_for(2.0);
        params.attractor_per_cm3 = 4000.0;
        let symmetry = Symmetry::new(
            SymmetryParams::Mirror {
                first: Axis::X,
                second: None,
            },
            &grid.bounds(),
            constants::GROWTH_SYMMETRY_BOUNDARY_EPSILON_VOXELS * grid.h,
        );

        let whole = scatter(
            &GrowthDomain::new(&grid),
            &fusion,
            &params,
            &mut Pcg32::new(11),
        );
        let sector = scatter(
            &GrowthDomain::confined(&grid, Some(&symmetry)),
            &fusion,
            &params,
            &mut Pcg32::new(11),
        );

        assert_eq!(whole.iter().filter(|p| p.on_surface).count(), 2);
        assert_eq!(
            sector.iter().filter(|p| p.on_surface).count(),
            1,
            "the surface target across the plane has to be dropped"
        );
        for point in &sector {
            assert!(
                point.position[0] <= 5.0,
                "an attractor landed at {:?}, outside the sector",
                point.position
            );
        }
        // The point *density* is what `attractor_per_cm3` means, so half the
        // domain asks for about half the points rather than all of them.
        let interior = |points: &[Attractor]| points.iter().filter(|p| !p.on_surface).count();
        let ratio = interior(&sector) as f64 / interior(&whole) as f64;
        assert!(
            (0.45..0.55).contains(&ratio),
            "the sector scattered {} interior points against {} in the whole domain",
            interior(&sector),
            interior(&whole)
        );
    }

    #[test]
    fn a_surface_point_holds_on_until_a_branch_has_actually_arrived() {
        // One open corridor with a forced solid cell at the far end. The branch
        // has to travel the whole way: a surface point is only consumed at the
        // fusion tolerance, not at the kill radius.
        let mut grid = design_grid(12);
        let target = grid.cell_index(11, 0, 0);
        grid.cells[target] = CellKind::Solid;
        let domain = GrowthDomain::new(&grid);
        let anchors = AnchorSet::solid(&grid);
        let fusion = Fusion {
            anchors: &anchors,
            grid: &grid,
            tolerance_mm: 0.6,
        };
        let mut params = params_for(2.0);
        params.attractor_per_cm3 = 0.0;
        let mut rng = Pcg32::new(5);
        let points = scatter(&domain, &fusion, &params, &mut rng);
        assert_eq!(points.len(), 1, "one anchor cell, one surface target");
        assert!(
            params.kill_radius_mm > 3.0,
            "the kill radius has to be the loose one"
        );

        let mut skeleton = Skeleton::new();
        skeleton.add_root(domain.cell_center(grid.cell_index(0, 0, 0)));
        let mut colonizer = Colonizer::new(points, &skeleton, &params, &fusion);
        assert_eq!(colonizer.remaining(), 1, "consumed before anything grew");
        assert_eq!(colonizer.unreached_surfaces(), 1);

        let mut steps = 0;
        while steps < params.max_steps && colonizer.remaining() > 0 {
            if colonizer.step(&mut skeleton, &domain, &params, &fusion) == 0 {
                break;
            }
            steps += 1;
        }
        assert_eq!(colonizer.remaining(), 0, "the branch never arrived");
        assert_eq!(colonizer.unreached_surfaces(), 0);
        // And it really is touching: the last node sits within the fusion
        // tolerance of the anchor cell.
        let tip = skeleton.len() - 1;
        assert!(
            anchors.near(&grid, skeleton.position(tip), 0.6),
            "the point was consumed without the branch reaching the surface"
        );
    }

    #[test]
    fn branches_never_enter_forbidden_cells_and_consume_what_they_reach() {
        let mut grid = design_grid(16);
        // A keepout pillar the growth has to route around.
        for k in 0..16 {
            for j in 6..10 {
                for i in 6..10 {
                    let cell = grid.cell_index(i, j, k);
                    grid.cells[cell] = CellKind::Void;
                }
            }
        }
        let domain = GrowthDomain::new(&grid);
        let anchors = AnchorSet::empty(&grid);
        let fusion = Fusion {
            anchors: &anchors,
            grid: &grid,
            tolerance_mm: 1.0,
        };
        let mut params = params_for(2.0);
        params.attractor_per_cm3 = 50.0;
        let mut rng = Pcg32::new(7);
        let points = scatter(&domain, &fusion, &params, &mut rng);
        assert!(points.len() > 50, "only {} attractors", points.len());

        let mut skeleton = Skeleton::new();
        skeleton.add_root(domain.cell_center(grid.cell_index(0, 0, 0)));
        let mut colonizer = Colonizer::new(points, &skeleton, &params, &fusion);
        let scattered = colonizer.points.len();

        let mut steps = 0;
        while steps < params.max_steps && colonizer.remaining() > 0 {
            if colonizer.step(&mut skeleton, &domain, &params, &fusion) == 0 {
                break;
            }
            steps += 1;
        }

        assert!(skeleton.len() > 1, "nothing grew");
        assert!(colonizer.consumed() > 0, "no attractor was reached");
        assert_eq!(colonizer.consumed() + colonizer.remaining(), scattered);

        // Every segment, and therefore every node, stays inside allowed cells.
        for node in &skeleton.nodes {
            let Some(parent) = node.parent else { continue };
            assert!(
                domain.segment_inside(skeleton.position(parent), node.position),
                "a branch left the allowed cells"
            );
        }
        // And no node sits inside the pillar.
        for node in &skeleton.nodes {
            let cell = domain.cell_at(node.position).expect("inside the block");
            assert!(domain.allows_cell(cell));
        }
    }

    #[test]
    fn the_step_cap_stops_growth_before_the_attractors_run_out() {
        let grid = design_grid(20);
        let domain = GrowthDomain::new(&grid);
        let anchors = AnchorSet::empty(&grid);
        let fusion = Fusion {
            anchors: &anchors,
            grid: &grid,
            tolerance_mm: 1.0,
        };
        let mut params = params_for(2.0);
        params.max_steps = 3;
        params.attractor_per_cm3 = 200.0;
        let mut rng = Pcg32::new(3);
        let points = scatter(&domain, &fusion, &params, &mut rng);

        let mut skeleton = Skeleton::new();
        skeleton.add_root(domain.cell_center(grid.cell_index(0, 0, 0)));
        let mut colonizer = Colonizer::new(points, &skeleton, &params, &fusion);

        let mut steps = 0;
        while steps < params.max_steps {
            if colonizer.step(&mut skeleton, &domain, &params, &fusion) == 0 {
                break;
            }
            steps += 1;
        }
        assert_eq!(steps, params.max_steps, "the cap did not stop the growth");
        assert!(
            colonizer.remaining() > 0,
            "the cap has to bite before the attractors are exhausted"
        );
    }

    #[test]
    fn a_boxed_in_node_gives_up_after_a_few_blocked_steps() {
        // One allowed cell, an attractor pulling from outside it: every step
        // leaves the cell and is refused, so the node has to stop trying.
        let mut grid = design_grid(5);
        for cell in grid.cells.iter_mut() {
            *cell = CellKind::Void;
        }
        let open = grid.cell_index(2, 2, 2);
        grid.cells[open] = CellKind::Design;
        let domain = GrowthDomain::new(&grid);
        let anchors = AnchorSet::empty(&grid);
        let fusion = Fusion {
            anchors: &anchors,
            grid: &grid,
            tolerance_mm: 0.5,
        };
        let mut params = params_for(1.0);
        // A kill radius below the two millimetres out to the attractor, so it
        // survives to keep pulling on a node that can never reach it.
        params.kill_radius_mm = 0.5;
        params.attraction_radius_mm = 5.0;
        params.step_mm = 0.75;

        let mut skeleton = Skeleton::new();
        skeleton.add_root(domain.cell_center(open));
        let outside: Vec3 = [4.5, 2.5, 2.5];
        let mut colonizer = Colonizer::new(
            interior(vec![outside], params.kill_radius_mm),
            &skeleton,
            &params,
            &fusion,
        );

        // Blocked steps end the branch, and then the point it could never reach
        // is given up rather than pulling for the rest of the run.
        for _ in 0..constants::GROWTH_MAX_FAILED_STEPS + 2 {
            assert_eq!(colonizer.step(&mut skeleton, &domain, &params, &fusion), 0);
        }
        assert_eq!(skeleton.len(), 1, "a branch escaped the single open cell");
        for _ in 0..patience_steps(&params) {
            colonizer.step(&mut skeleton, &domain, &params, &fusion);
        }
        assert_eq!(colonizer.remaining(), 0, "the point was never given up");
        assert_eq!(colonizer.abandoned(), 1);
        assert_eq!(colonizer.consumed(), 0, "nothing was ever reached");
    }
}
