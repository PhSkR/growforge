//! Connected components of the extracted surface, and the floating fragments
//! culled from it.
//!
//! [`crate::voids::solid_bodies`] answers the same question one stage earlier
//! and on a different object: it flood fills the *cell* grid at the iso level,
//! so it describes the density field. The surface that ships is extracted from
//! the *node* field, whose values are eight-cell averages, optionally refined
//! and then smoothed, and the two readings disagree in both directions. A lone
//! dense cell is a body in the field and peaks at an eighth of the iso level in
//! the node lattice, so it makes no surface at all. A clump of at-or-above-iso
//! cells joined to the part only through a one-cell-wide bridge is *one* body in
//! the field - the fill walks straight across the bridge - while a node touching
//! two cells of that bridge averages them with six empty ones and lands far
//! below the iso level, so the surface comes out in two pieces and the second is
//! a skin welded to nothing: watertight, manifold, enclosing no cavity, and
//! invisible to every check that came before. Intermediate density clumps and
//! one-voxel connectors are both routine SIMP artifacts at low volume fractions,
//! which is why this pass looks at the mesh itself, after smoothing and before
//! validation, and why the run reports both readings side by side rather than
//! letting one stand for the other.
//!
//! Connectivity is shared vertices. Marching cubes welds by lattice edge, so two
//! triangles share a vertex index exactly when they share a corner of the
//! surface, and a union-find over the triangle list in triangle order labels
//! every component deterministically.
//!
//! A component is *outward* or *inward* by the sign of the volume it encloses,
//! which is meaningful precisely because [`crate::mesh::validate`] enforces
//! outward consistent winding on the whole mesh: an outward shell of material
//! encloses a positive volume, the inner shell of a cavity a negative one.
//!
//! What decides whether an outward shell is kept is **purpose, never size**. A
//! shell is culled only when it touches no support region, no load region and no
//! keepin cell: geometry that serves nothing the configuration declared is
//! debris, and geometry that serves something declared is deliberate however
//! small it is and whatever else is in the file. Ranking by volume was tried and
//! is wrong - a keepin boss rides on top of `mass_fraction` and can outweigh the
//! structure, so the largest shell is not the part in the general case, and
//! ranking shipped the boss while culling the load path. Volume now orders the
//! report and decides nothing.
//!
//! An inward shell has no purpose of its own: it is the inside of a cavity, and
//! it belongs to whichever outward shell contains it - see [`resolve`].

use crate::config::IslandPolicy;
use crate::constants;
use crate::geometry::{Aabb, Shape, Vec3, cross, difference, inner, length, scale};
use crate::grid::Grid;
use crate::mesh::Mesh;

/// What a region of the problem declares the material inside it is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorKind {
    /// A `[[supports]]` region: the part is held here.
    Support,
    /// A `[[loadcases.loads]]` region: load enters the part here.
    Load,
    /// A `[[keepin]]` region: material was asked for here.
    Keepin,
}

impl AnchorKind {
    /// Every kind, in report order.
    pub const ALL: [AnchorKind; 3] = [AnchorKind::Support, AnchorKind::Load, AnchorKind::Keepin];

    /// Short label for reports.
    pub fn label(self) -> &'static str {
        match self {
            AnchorKind::Support => "support",
            AnchorKind::Load => "load",
            AnchorKind::Keepin => "keepin",
        }
    }

    /// Bit this kind occupies in an [`AnchorSet`].
    fn bit(self) -> u8 {
        match self {
            AnchorKind::Support => 1,
            AnchorKind::Load => 2,
            AnchorKind::Keepin => 4,
        }
    }
}

/// The kinds of declared region a component touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AnchorSet(u8);

impl AnchorSet {
    /// The set that touches nothing declared.
    pub fn none() -> AnchorSet {
        AnchorSet(0)
    }

    /// True when the component touches nothing the configuration declared, which
    /// is what makes it debris.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// True when the component touches a region of this kind.
    pub fn holds(self, kind: AnchorKind) -> bool {
        self.0 & kind.bit() != 0
    }

    /// The kinds it holds, in report order.
    pub fn kinds(self) -> impl Iterator<Item = AnchorKind> {
        AnchorKind::ALL.into_iter().filter(move |k| self.holds(*k))
    }

    /// Add a kind.
    pub fn insert(&mut self, kind: AnchorKind) {
        self.0 |= kind.bit();
    }

    /// Add everything in `other`.
    fn union(&mut self, other: AnchorSet) {
        self.0 |= other.0;
    }
}

/// One region of the problem that declares a purpose for the material in it.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorRegion {
    /// What the region declares the material is for.
    pub kind: AnchorKind,
    /// How the region is named in reports, e.g. `support 1`.
    pub label: String,
    /// The grid nodes it covers, in ascending order.
    pub nodes: Vec<usize>,
    /// A bounded sample of points inside the material the region selected, in
    /// millimetres. A component that *encloses* one of these holds the region
    /// even when its surface runs nowhere near the region's nodes, which is the
    /// case of a load region buried in the middle of a keepin pad.
    pub probes: Vec<Vec3>,
    /// The region's own shape, when it has one. Neither the nodes nor the probes
    /// can say whether a *culled* fragment was inside the region: the nodes are
    /// a lattice sample and the probes are a bounded one, and a region spanning
    /// two disconnected lobes can have both of them land in the lobe that
    /// survives. This is asked of every fragment that leaves, so material a
    /// declared region asked for is never removed silently.
    pub shape: Option<Shape>,
}

/// Every declared purpose of a problem, on the node lattice of its grid.
///
/// A component holds a region when either of two things is true, because the two
/// extremes of a thin part and a thick one need different questions asked:
///
/// * a vertex of the component lies in a lattice cell one of the region's nodes
///   is a corner of. Vertices came off that lattice, the node stencil is the
///   same eight-cell average the density field is sampled with, and matching an
///   index is exact where matching a point against a shape has to pick a
///   tolerance. The lattice cell is half a voxel of slack, which is what
///   smoothing and the iso nudge can move a vertex by, and no more.
/// * the component *encloses* one of the region's probes - points inside the
///   material the region selected. A load region in the middle of a keepin pad
///   is nowhere near the pad's surface, and this is what sees it.
///
/// Neither test alone is enough: material a voxel thick shrinks to a surface
/// that may enclose no cell centre at all, and material ten voxels thick has no
/// vertex within a voxel of its interior. Where the probes come from, and the
/// one shape of region they can still under-cover, is
/// [`crate::problem::Problem`]'s to say.
#[derive(Debug, Clone, Default)]
pub struct Anchors {
    origin: Vec3,
    spacing: f64,
    counts: [usize; 3],
    regions: Vec<AnchorRegion>,
    kind_of_node: Vec<u8>,
}

impl Anchors {
    /// Build the lattice mask of `regions` over the node lattice of `grid`.
    ///
    /// Node indices outside the lattice are ignored rather than rejected: a
    /// region list is a report of what a problem selected, and this is a
    /// classifier, not a validator.
    pub fn new(grid: &Grid, regions: Vec<AnchorRegion>) -> Anchors {
        let mut kind_of_node = vec![0u8; grid.n_nodes()];
        for region in &regions {
            for &node in &region.nodes {
                if let Some(slot) = kind_of_node.get_mut(node) {
                    *slot |= region.kind.bit();
                }
            }
        }
        Anchors {
            origin: grid.origin,
            spacing: grid.h,
            counts: [grid.nnx(), grid.nny(), grid.nnz()],
            regions,
            kind_of_node,
        }
    }

    /// The anchors of a problem that declared nothing at all.
    ///
    /// Nothing is anchored, so nothing is culled: with no declared purpose
    /// anywhere there is no evidence about what the part is, and removing
    /// geometry on no evidence is the failure this rule exists to prevent.
    pub fn none() -> Anchors {
        Anchors::default()
    }

    /// The declared regions, in the order they were given.
    pub fn regions(&self) -> &[AnchorRegion] {
        &self.regions
    }

    /// True when nothing at all was declared.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// The nodes of the lattice cell a world position lies in, clamped to the
    /// lattice.
    fn nodes_around(&self, point: Vec3) -> impl Iterator<Item = usize> + '_ {
        let mut base = [0usize; 3];
        let mut inside = true;
        for d in 0..3 {
            let last = self.counts[d].saturating_sub(1);
            let local = (point[d] - self.origin[d]) / self.spacing;
            if !local.is_finite() || self.counts[d] == 0 {
                inside = false;
                break;
            }
            base[d] = (local.floor().clamp(0.0, last as f64)) as usize;
        }
        let counts = self.counts;
        (0..8).filter_map(move |corner| {
            if !inside {
                return None;
            }
            let mut index = 0;
            let mut stride = 1;
            for d in 0..3 {
                let step = (corner >> d) & 1;
                let coordinate = (base[d] + step).min(counts[d].saturating_sub(1));
                index += coordinate * stride;
                stride *= counts[d];
            }
            Some(index)
        })
    }

    /// The kinds of region the lattice cell holding `point` touches.
    fn kinds_at(&self, point: Vec3) -> AnchorSet {
        let mut set = AnchorSet::none();
        for node in self.nodes_around(point) {
            let bits = self.kind_of_node[node];
            for kind in AnchorKind::ALL {
                if bits & kind.bit() != 0 {
                    set.insert(kind);
                }
            }
        }
        set
    }
}

/// One connected component of an extracted surface.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceComponent {
    /// Triangle count.
    pub triangles: usize,
    /// Volume the component encloses, in cubic millimetres, signed: positive
    /// for an outward shell of material, negative for the inner shell of a
    /// cavity.
    pub volume_mm3: f64,
    /// Area weighted centroid of the component's triangles, in millimetres.
    /// The surface centroid rather than the volume one, because it stays where
    /// the shell is even when the shell encloses almost nothing.
    pub centroid: Vec3,
    /// The kinds of declared region this component touches. Empty is what makes
    /// an outward shell a floating fragment.
    pub anchors: AnchorSet,
}

/// Which connected component every triangle of a mesh belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentLabels {
    /// Component index of every triangle, in triangle order.
    pub of_triangle: Vec<usize>,
    /// Number of components found. Labels run from zero to one below it, in the
    /// order of each component's first triangle.
    pub components: usize,
}

/// What the island pass found and did.
#[derive(Debug, Clone, PartialEq)]
pub struct IslandReport {
    /// The policy that was in force.
    pub policy: IslandPolicy,
    /// Connected components the extracted surface held, before any culling.
    pub components: usize,
    /// The outward shells of the exported mesh, largest first, each with the
    /// kinds of declared region that anchor it.
    pub bodies: Vec<SurfaceComponent>,
    /// Inner cavity shells in the exported mesh.
    pub cavity_shells: usize,
    /// Every outward shell that touches nothing the configuration declared,
    /// largest first. They are still in the exported mesh under
    /// [`IslandPolicy::Keep`], and under [`IslandPolicy::Cull`] when nothing at
    /// all was declared.
    pub fragments: Vec<SurfaceComponent>,
    /// How many of them were removed.
    pub culled_fragments: usize,
    /// Triangles removed, fragments and their own inner shells together.
    pub culled_triangles: usize,
    /// Absolute volume removed, in cubic millimetres.
    pub culled_volume_mm3: f64,
    /// Labels of the support and load regions that no exported body reaches.
    /// Empty is the normal case; anything in it means the part that ships does
    /// not connect something the configuration asked for.
    pub unserved: Vec<String>,
    /// Culled fragments that lie inside a declared support or load region, as
    /// the fragment's index in [`IslandReport::fragments`] and the region's
    /// label. Empty is the normal case: debris is debris. Anything in it is
    /// material a declared region asked for that came out disconnected and was
    /// removed - the partial loss [`IslandReport::unserved`] structurally cannot
    /// see, because the rest of that same region is still served by what ships.
    pub culled_inside: Vec<(usize, String)>,
}

impl IslandReport {
    /// Volume of the largest floating fragment found, in cubic millimetres.
    pub fn largest_fragment_mm3(&self) -> Option<f64> {
        self.fragments.first().map(|f| f.volume_mm3.abs())
    }

    /// Total volume of every floating fragment found, in cubic millimetres,
    /// whether it was culled or kept.
    pub fn fragment_volume_mm3(&self) -> f64 {
        self.fragments.iter().map(|f| f.volume_mm3.abs()).sum()
    }
}

/// Disjoint set over vertex indices, with union by size and path halving.
struct Dsu {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl Dsu {
    fn new(n: usize) -> Dsu {
        Dsu {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut v: u32) -> u32 {
        while self.parent[v as usize] != v {
            let grand = self.parent[self.parent[v as usize] as usize];
            self.parent[v as usize] = grand;
            v = grand;
        }
        v
    }

    fn union(&mut self, a: u32, b: u32) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
    }
}

/// Label every triangle with the connected component it belongs to.
///
/// Two triangles are connected when they share a vertex index. Vertices are
/// welded by lattice edge upstream, so that is exactly "share a corner of the
/// surface"; a mesh assembled some other way is read the same way, which is
/// what makes the labelling a property of the index buffer alone. Component
/// indices follow the order of each component's first triangle, so the same
/// mesh always labels the same way whatever the union order inside it.
pub fn label_components(mesh: &Mesh) -> ComponentLabels {
    let mut dsu = Dsu::new(mesh.vertices.len());
    for t in &mesh.triangles {
        dsu.union(t[0], t[1]);
        dsu.union(t[0], t[2]);
    }
    // Roots in first-triangle order, so the dense labels are deterministic.
    let mut label_of_root: Vec<Option<usize>> = vec![None; mesh.vertices.len()];
    let mut of_triangle = Vec::with_capacity(mesh.triangles.len());
    let mut components = 0;
    for t in &mesh.triangles {
        let root = dsu.find(t[0]) as usize;
        let label = *label_of_root[root].get_or_insert_with(|| {
            let next = components;
            components += 1;
            next
        });
        of_triangle.push(label);
    }
    ComponentLabels {
        of_triangle,
        components,
    }
}

/// Signed volume of a subset of a mesh's triangles, in cubic millimetres.
///
/// The same divergence theorem sum [`crate::mesh::validate::enclosed_volume`]
/// applies to a whole mesh, over a component of one.
pub fn component_volume(mesh: &Mesh, triangles: &[usize]) -> f64 {
    let mut total = 0.0;
    for &index in triangles {
        let t = mesh.triangles[index];
        let a = mesh.vertices[t[0] as usize];
        let b = mesh.vertices[t[1] as usize];
        let c = mesh.vertices[t[2] as usize];
        total += inner(a, cross(b, c));
    }
    total / 6.0
}

/// Axis aligned bounds of the vertices a subset of triangles uses.
fn component_bounds(mesh: &Mesh, triangles: &[usize]) -> Aabb {
    let mut bounds = Aabb::empty();
    for &index in triangles {
        for v in mesh.triangles[index] {
            let vertex = mesh.vertices[v as usize];
            for (slot, value) in bounds.min.iter_mut().zip(vertex.iter()) {
                *slot = slot.min(*value);
            }
            for (slot, value) in bounds.max.iter_mut().zip(vertex.iter()) {
                *slot = slot.max(*value);
            }
        }
    }
    bounds
}

/// Triangle count, signed volume, area weighted centroid and declared purposes
/// of a component.
fn describe(mesh: &Mesh, triangles: &[usize], anchors: &Anchors) -> SurfaceComponent {
    let mut centroid = [0.0f64; 3];
    let mut area_sum = 0.0;
    let mut touched = AnchorSet::none();
    for &index in triangles {
        let t = mesh.triangles[index];
        let corners = t.map(|v| mesh.vertices[v as usize]);
        let area = 0.5
            * length(cross(
                difference(corners[1], corners[0]),
                difference(corners[2], corners[0]),
            ));
        area_sum += area;
        for d in 0..3 {
            centroid[d] += area * (corners[0][d] + corners[1][d] + corners[2][d]) / 3.0;
        }
        for corner in corners {
            touched.union(anchors.kinds_at(corner));
        }
    }
    if area_sum > 0.0 {
        centroid = scale(centroid, 1.0 / area_sum);
    }
    SurfaceComponent {
        triangles: triangles.len(),
        volume_mm3: component_volume(mesh, triangles),
        centroid,
        anchors: touched,
    }
}

/// True when `point` is inside the axis aligned bounds.
fn within(bounds: &Aabb, point: Vec3) -> bool {
    (0..3).all(|d| point[d] >= bounds.min[d] && point[d] <= bounds.max[d])
}

/// The labels of the support and load regions that no kept component reaches.
///
/// The same two questions [`Anchors`] answers for a component, asked of the
/// surviving ones: a region is served when one of its nodes is a node of the
/// lattice cell some surviving vertex lies in, or when a surviving outward shell
/// encloses one of its probes. Keepin regions are left out - a keepin is
/// material that was asked for, not a connection that has to be made - and so is
/// a gravity load, which selects no region at all.
fn unserved_regions(
    mesh: &Mesh,
    anchors: &Anchors,
    groups: &[Vec<usize>],
    bounds: &[Aabb],
    kept_bodies: &[usize],
) -> Vec<String> {
    if anchors.is_empty() {
        return Vec::new();
    }
    let mut covered = vec![false; anchors.kind_of_node.len()];
    for &component in kept_bodies {
        for &index in &groups[component] {
            for v in mesh.triangles[index] {
                for node in anchors.nodes_around(mesh.vertices[v as usize]) {
                    covered[node] = true;
                }
            }
        }
    }
    anchors
        .regions()
        .iter()
        .filter(|region| region.kind != AnchorKind::Keepin && !region.nodes.is_empty())
        .filter(|region| {
            !region.nodes.iter().any(|&n| covered[n])
                && !region.probes.iter().any(|&probe| {
                    kept_bodies.iter().any(|&component| {
                        within(&bounds[component], probe)
                            && encloses(mesh, &groups[component], probe)
                    })
                })
        })
        .map(|region| region.label.clone())
        .collect()
}

/// True when a closed component of `mesh` encloses `point`.
///
/// The solid angle the component subtends at the point, by the Van Oosterom and
/// Strackee formula, summed over its triangles: `4 * PI` in magnitude when the
/// point is inside a closed shell and zero when it is outside, whatever the
/// shell's orientation. A ray cast would be cheaper and would have to pick a
/// direction that no triangle edge or plane is degenerate against; the solid
/// angle has no such direction to pick and no parity to lose, which is what
/// makes it the reading a cull decision can be made on.
pub fn encloses(mesh: &Mesh, triangles: &[usize], point: Vec3) -> bool {
    let mut solid_angle = 0.0;
    for &index in triangles {
        let t = mesh.triangles[index];
        let [a, b, c] = t.map(|v| difference(mesh.vertices[v as usize], point));
        let (la, lb, lc) = (length(a), length(b), length(c));
        let denominator = la * lb * lc + inner(a, b) * lc + inner(a, c) * lb + inner(b, c) * la;
        solid_angle += 2.0 * inner(a, cross(b, c)).atan2(denominator);
    }
    // Half a full turn: the sum is 4 * PI, -4 * PI or 0 up to rounding, so
    // anything in between is noise rather than a decision.
    solid_angle.abs() > constants::ISLAND_SOLID_ANGLE_INSIDE
}

/// Classify the components of `mesh` against the purposes `anchors` declares
/// and, under [`IslandPolicy::Cull`], remove the ones that serve none.
///
/// An outward shell is kept when it touches a support region, a load region or a
/// keepin cell, and culled when it touches none of the three. Size is not a
/// criterion: a keepin boss is paid for out of the same `mass_fraction` as the
/// structure and can be the larger of the two, so "the biggest shell is the
/// part" is false in ordinary use and ranking by it shipped the boss while
/// culling the load path. What survives is everything the configuration asked
/// for, however many pieces that is; what leaves is what nothing asked for.
///
/// Two floors keep the rule from ever removing geometry on no evidence: nothing
/// is culled when no shell is anchored at all - a surface that reaches nothing
/// declared says nothing about which of its pieces is the part - and nothing is
/// culled under [`IslandPolicy::Keep`]. Both are reported rather than silent.
///
/// An inward shell is the inside of a cavity, and which solid it belongs to is
/// not something its winding can say - it is the part's own cavity when the part
/// contains it, and a fragment's cavity when a fragment does. So each one is
/// attributed to the innermost outward shell that [`encloses`] it and survives
/// exactly when that shell survives. This is what keeps the cull from
/// contradicting the cavity policy: a cavity kept by `voids = "warn"` keeps its
/// inner shell, and a cavity inside a fragment leaves with the fragment rather
/// than being orphaned in the file. The attribution costs a solid angle sum per
/// shell per candidate whose bounds hold the point at all, and is only reached
/// when something is really being culled.
///
/// Nothing is removed when there is nothing to remove: a mesh in one piece, or
/// a mesh whose components are anchored solids and their cavities, is handed
/// back untouched rather than rebuilt, so an export with no fragment in it is
/// the export it always was, byte for byte.
pub fn resolve(mesh: &mut Mesh, policy: IslandPolicy, anchors: &Anchors) -> IslandReport {
    let labels = label_components(mesh);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); labels.components];
    for (triangle, &label) in labels.of_triangle.iter().enumerate() {
        groups[label].push(triangle);
    }
    let mut described: Vec<SurfaceComponent> =
        groups.iter().map(|g| describe(mesh, g, anchors)).collect();
    let outward: Vec<usize> = (0..described.len())
        .filter(|&c| described[c].volume_mm3 > 0.0)
        .collect();
    // A point outside a shell's bounds is outside the shell, and the box is
    // exact about that, so every solid angle sum below is only paid where it can
    // change the answer.
    let bounds: Vec<Aabb> = groups.iter().map(|g| component_bounds(mesh, g)).collect();

    // A shell whose surface runs nowhere near a declared node may still hold the
    // region deep inside itself: a load region in the middle of a keepin pad is
    // several voxels from the pad's surface. Asked only of the shells the
    // lattice match left with nothing, so a clean export never pays for it.
    for &component in &outward {
        if !described[component].anchors.is_empty() {
            continue;
        }
        for region in anchors.regions() {
            if region.probes.iter().any(|&probe| {
                within(&bounds[component], probe) && encloses(mesh, &groups[component], probe)
            }) {
                described[component].anchors.insert(region.kind);
            }
        }
    }

    // Debris: an outward shell serving nothing that was declared. Volume orders
    // the list and decides nothing in it.
    let purposeless: Vec<usize> = outward
        .iter()
        .copied()
        .filter(|&c| described[c].anchors.is_empty())
        .collect();
    let mut ranked: Vec<(usize, SurfaceComponent)> = purposeless
        .iter()
        .map(|&c| (c, described[c].clone()))
        .collect();
    ranked.sort_by(|a, b| b.1.volume_mm3.abs().total_cmp(&a.1.volume_mm3.abs()));
    let fragments: Vec<SurfaceComponent> = ranked.iter().map(|(_, f)| f.clone()).collect();

    let anchored = outward.len() - purposeless.len();
    let cull = policy == IslandPolicy::Cull && anchored > 0 && !purposeless.is_empty();

    let mut keep = vec![true; described.len()];
    if cull {
        for &component in &purposeless {
            keep[component] = false;
        }
        // Candidate containers, innermost first, so the first hit is the shell
        // the cavity belongs to.
        let mut containers = outward.clone();
        containers.sort_by(|&a, &b| {
            described[a]
                .volume_mm3
                .total_cmp(&described[b].volume_mm3)
                .then(a.cmp(&b))
        });
        for component in 0..described.len() {
            if described[component].volume_mm3 > 0.0 {
                continue;
            }
            let point = mesh.vertices[mesh.triangles[groups[component][0]][0] as usize];
            // A cavity with no container at all is an inside-out shell floating
            // on its own; it leaves with the fragments.
            keep[component] = containers
                .iter()
                .find(|&&c| within(&bounds[c], point) && encloses(mesh, &groups[c], point))
                .is_some_and(|&c| keep[c]);
        }
    }

    let mut culled_fragments = 0;
    let mut culled_triangles = 0;
    let mut culled_volume_mm3 = 0.0;
    let mut bodies: Vec<SurfaceComponent> = Vec::new();
    let mut cavity_shells = 0;
    let mut kept_bodies: Vec<usize> = Vec::new();
    for (component, description) in described.iter().enumerate() {
        if !keep[component] {
            culled_triangles += description.triangles;
            culled_volume_mm3 += description.volume_mm3.abs();
            if description.volume_mm3 > 0.0 {
                culled_fragments += 1;
            }
            continue;
        }
        if description.volume_mm3 > 0.0 {
            kept_bodies.push(component);
            bodies.push(description.clone());
        } else {
            cavity_shells += 1;
        }
    }
    bodies.sort_by(|a, b| b.volume_mm3.total_cmp(&a.volume_mm3));
    let unserved = unserved_regions(mesh, anchors, &groups, &bounds, &kept_bodies);
    let culled_inside = culled_inside_regions(mesh, anchors, &groups, &ranked, &keep);
    if culled_triangles > 0 {
        retain_components(mesh, &labels, &keep);
    }

    IslandReport {
        policy,
        components: labels.components,
        bodies,
        cavity_shells,
        fragments,
        culled_fragments,
        culled_triangles,
        culled_volume_mm3,
        unserved,
        culled_inside,
    }
}

/// Culled fragments that lie inside a declared support or load region, as
/// `(index into the ranked fragment list, region label)`.
///
/// The check the other two cannot make. The lattice match and the probes both
/// ask whether a component *holds* a region, and both are samples: a region
/// spanning two disconnected lobes can have every node and every probe of it
/// land in the lobe that survives, and then the lobe that leaves takes material
/// the region asked for with it while
/// [`unserved_regions`] reads the region as served, because it is - by the other
/// lobe. So every fragment that is about to leave is asked the question
/// directly, against the region's own shape rather than any sampling of it: is
/// any of your surface inside it. A handful of fragments against a handful of
/// regions, only on the path where something is really being removed.
fn culled_inside_regions(
    mesh: &Mesh,
    anchors: &Anchors,
    groups: &[Vec<usize>],
    ranked: &[(usize, SurfaceComponent)],
    keep: &[bool],
) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    for (index, (component, _)) in ranked.iter().enumerate() {
        if keep[*component] {
            continue;
        }
        for region in anchors.regions() {
            let Some(shape) = &region.shape else {
                continue;
            };
            let inside = groups[*component].iter().any(|&triangle| {
                mesh.triangles[triangle]
                    .iter()
                    .any(|&v| shape.contains(mesh.vertices[v as usize]))
            });
            if inside {
                found.push((index, region.label.clone()));
            }
        }
    }
    found
}

/// Keep only the triangles whose component is marked, and the vertices they
/// still use.
///
/// Triangles keep their relative order and vertices their relative index order,
/// so the surviving mesh is the one that would have been extracted had the
/// culled material never been there - no reordering, and nothing for the
/// per-vertex stress colours or the STL writer to re-key.
fn retain_components(mesh: &mut Mesh, labels: &ComponentLabels, keep: &[bool]) {
    let mut used = vec![false; mesh.vertices.len()];
    let mut triangles = Vec::with_capacity(mesh.triangles.len());
    for (index, &label) in labels.of_triangle.iter().enumerate() {
        if !keep[label] {
            continue;
        }
        let t = mesh.triangles[index];
        for &v in t.iter() {
            used[v as usize] = true;
        }
        triangles.push(t);
    }

    let mut remap = vec![0u32; mesh.vertices.len()];
    let mut vertices = Vec::with_capacity(mesh.vertices.len());
    for (index, &keep_vertex) in used.iter().enumerate() {
        if !keep_vertex {
            continue;
        }
        remap[index] = vertices.len() as u32;
        vertices.push(mesh.vertices[index]);
    }
    for t in triangles.iter_mut() {
        for v in t.iter_mut() {
            *v = remap[*v as usize];
        }
    }
    mesh.vertices = vertices;
    mesh.triangles = triangles;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Aabb;
    use crate::mesh::validate;

    /// A closed box, two triangles per face, wound counter-clockwise seen from
    /// outside when `outward`, and inside out when not.
    fn box_mesh(min: Vec3, max: Vec3, outward: bool) -> Mesh {
        let corner = |c: usize| {
            [
                if c & 1 == 0 { min[0] } else { max[0] },
                if c & 2 == 0 { min[1] } else { max[1] },
                if c & 4 == 0 { min[2] } else { max[2] },
            ]
        };
        let mut triangles = vec![
            [0, 3, 1],
            [0, 2, 3],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 7, 3],
            [2, 6, 7],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
        ];
        if !outward {
            for t in triangles.iter_mut() {
                t.swap(0, 1);
            }
        }
        Mesh {
            vertices: (0..8).map(corner).collect(),
            triangles,
        }
    }

    /// Append `other` to `mesh`, shifting its indices.
    fn append(mesh: &mut Mesh, other: &Mesh) {
        let base = mesh.vertices.len() as u32;
        mesh.vertices.extend_from_slice(&other.vertices);
        mesh.triangles
            .extend(other.triangles.iter().map(|t| t.map(|v| v + base)));
    }

    /// A cube of side `side` at the origin holding a centred cubic cavity of
    /// side `hole`.
    fn hollow_cube(side: f64, hole: f64) -> Mesh {
        let mut mesh = box_mesh([0.0; 3], [side; 3], true);
        let lo = 0.5 * (side - hole);
        // The inner shell is wound inside out relative to itself, which is
        // outward relative to the material around it.
        append(&mut mesh, &box_mesh([lo; 3], [lo + hole; 3], false));
        mesh
    }

    /// The lattice every fixture in here lives on: one millimetre cells over
    /// the block the boxes above are placed in.
    fn fixture_grid() -> crate::grid::Grid {
        crate::grid::Grid::from_bounds(
            &Aabb {
                min: [-4.0; 3],
                max: [32.0; 3],
            },
            1.0,
        )
    }

    /// Anchors declaring each `(kind, min, max)` box over the nodes inside it.
    fn anchors_over(regions: &[(AnchorKind, Vec3, Vec3)]) -> Anchors {
        let grid = fixture_grid();
        let regions = regions
            .iter()
            .enumerate()
            .map(|(index, &(kind, min, max))| {
                let mut nodes = Vec::new();
                for k in 0..grid.nnz() {
                    for j in 0..grid.nny() {
                        for i in 0..grid.nnx() {
                            let point = [
                                grid.origin[0] + i as f64 * grid.h,
                                grid.origin[1] + j as f64 * grid.h,
                                grid.origin[2] + k as f64 * grid.h,
                            ];
                            if (0..3).all(|d| point[d] >= min[d] && point[d] <= max[d]) {
                                nodes.push(grid.node_index(i, j, k));
                            }
                        }
                    }
                }
                assert!(!nodes.is_empty(), "region {} selected no node", index + 1);
                AnchorRegion {
                    kind,
                    label: format!("{} region {}", kind.label(), index + 1),
                    nodes,
                    probes: Vec::new(),
                    shape: Some(Shape::axis_aligned_box(min, max)),
                }
            })
            .collect();
        Anchors::new(&grid, regions)
    }

    /// The same regions, each carrying one probe point at `probe`.
    fn anchors_probing(regions: &[(AnchorKind, Vec3, Vec3)], probe: Vec3) -> Anchors {
        let grid = fixture_grid();
        let mut anchors = anchors_over(regions);
        let regions: Vec<AnchorRegion> = anchors
            .regions()
            .iter()
            .map(|region| AnchorRegion {
                probes: vec![probe],
                ..region.clone()
            })
            .collect();
        anchors = Anchors::new(&grid, regions);
        anchors
    }

    /// A support region under the ten millimetre cube every fixture starts with.
    fn anchored_part() -> Anchors {
        anchors_over(&[(AnchorKind::Support, [0.0, 0.0, 0.0], [10.0, 10.0, 0.0])])
    }

    /// One region whose node and probe samples both landed in the first lobe of
    /// a part that came out in two, while its shape spans both of them.
    fn region_over_two_lobes(shape: Shape) -> Anchors {
        let grid = fixture_grid();
        let mut nodes = Vec::new();
        for j in 0..grid.nny() {
            for i in 0..grid.nnx() {
                let point = [
                    grid.origin[0] + i as f64 * grid.h,
                    grid.origin[1] + j as f64 * grid.h,
                ];
                if (0..2).all(|d| point[d] >= 0.0 && point[d] <= 10.0) {
                    nodes.push(grid.node_index(i, j, 4));
                }
            }
        }
        assert!(!nodes.is_empty());
        Anchors::new(
            &grid,
            vec![AnchorRegion {
                kind: AnchorKind::Load,
                label: "load 1 of case \"tip\"".to_string(),
                nodes,
                probes: vec![[5.0, 5.0, 5.0]],
                shape: Some(shape),
            }],
        )
    }

    #[test]
    fn a_two_fragment_mesh_labels_two_components() {
        let mut mesh = box_mesh([0.0; 3], [10.0; 3], true);
        append(&mut mesh, &box_mesh([20.0; 3], [22.0; 3], true));
        let labels = label_components(&mesh);
        assert_eq!(labels.components, 2);
        assert_eq!(labels.of_triangle.len(), mesh.triangles.len());
        // Twelve triangles each, in the order the two shells were appended.
        assert!(labels.of_triangle[..12].iter().all(|&l| l == 0));
        assert!(labels.of_triangle[12..].iter().all(|&l| l == 1));
        // Labelling is a pure function of the mesh.
        assert_eq!(label_components(&mesh), labels);
    }

    #[test]
    fn labels_follow_triangle_order_however_the_vertices_are_numbered() {
        // The same two shells, with the second one's triangles listed first:
        // the labels have to follow the triangles rather than the vertices.
        let mut mesh = box_mesh([0.0; 3], [10.0; 3], true);
        append(&mut mesh, &box_mesh([20.0; 3], [22.0; 3], true));
        mesh.triangles.rotate_left(12);
        let labels = label_components(&mesh);
        assert_eq!(labels.components, 2);
        assert!(labels.of_triangle[..12].iter().all(|&l| l == 0));
        assert!(labels.of_triangle[12..].iter().all(|&l| l == 1));
        // And the first component is now the small shell.
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); 2];
        for (index, &label) in labels.of_triangle.iter().enumerate() {
            groups[label].push(index);
        }
        assert!(component_volume(&mesh, &groups[0]) < component_volume(&mesh, &groups[1]));
    }

    #[test]
    fn a_body_with_a_cavity_is_two_components_and_neither_is_culled() {
        let mut mesh = hollow_cube(10.0, 4.0);
        let before = mesh.clone();
        validate::validate(&mesh).expect("a hollow cube is closed and wound outwards");

        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchored_part());
        assert_eq!(report.components, 2);
        assert_eq!(report.bodies.len(), 1);
        assert!(report.bodies[0].anchors.holds(AnchorKind::Support));
        assert_eq!(report.cavity_shells, 1);
        assert!(report.fragments.is_empty());
        assert_eq!(report.culled_triangles, 0);
        assert_eq!(report.culled_volume_mm3, 0.0);
        assert!(report.unserved.is_empty());
        // Untouched, not rebuilt.
        assert_eq!(mesh.vertices, before.vertices);
        assert_eq!(mesh.triangles, before.triangles);
        // 1000 mm3 of material minus the 64 mm3 cavity.
        let stats = validate::validate(&mesh).expect("still closed");
        assert!((stats.volume_mm3 - 936.0).abs() < 1e-9);
    }

    #[test]
    fn a_shell_that_serves_nothing_declared_is_culled_and_reported() {
        let mut mesh = box_mesh([0.0; 3], [10.0; 3], true);
        append(
            &mut mesh,
            &box_mesh([20.0, 0.0, 0.0], [22.0, 2.0, 2.0], true),
        );
        append(
            &mut mesh,
            &box_mesh([-3.0, 0.0, 0.0], [-2.0, 1.0, 1.0], true),
        );
        validate::validate(&mesh).expect("three closed shells are a valid mesh");

        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchored_part());
        assert_eq!(report.components, 3);
        assert_eq!(report.bodies.len(), 1);
        assert_eq!(report.cavity_shells, 0);
        assert_eq!(report.culled_fragments, 2);
        assert_eq!(report.culled_triangles, 24);
        assert!((report.culled_volume_mm3 - 9.0).abs() < 1e-9);
        // Largest first: the 2 mm cube, then the 1 mm one.
        assert_eq!(report.fragments.len(), 2);
        assert!((report.fragments[0].volume_mm3 - 8.0).abs() < 1e-9);
        assert!((report.fragments[1].volume_mm3 - 1.0).abs() < 1e-9);
        assert!(report.fragments.iter().all(|f| f.anchors.is_empty()));
        assert_eq!(report.largest_fragment_mm3(), Some(8.0));
        assert!((report.fragment_volume_mm3() - 9.0).abs() < 1e-9);
        for (component, expected) in report.fragments[0].centroid.iter().zip([21.0, 1.0, 1.0]) {
            assert!(
                (component - expected).abs() < 1e-9,
                "centroid {:?}",
                report.fragments[0].centroid
            );
        }

        // What is left is the part alone, still closed, with nothing but its
        // own vertices in it.
        assert_eq!(mesh.triangles.len(), 12);
        assert_eq!(mesh.vertices.len(), 8);
        let stats = validate::validate(&mesh).expect("the culled mesh must validate");
        assert!((stats.volume_mm3 - 1000.0).abs() < 1e-9);
        assert_eq!(stats.bounds.min, [0.0; 3]);
        assert_eq!(stats.bounds.max, [10.0; 3]);
    }

    /// The rule this pass turns on, and the one ranking by volume got wrong: a
    /// keepin boss is paid for out of the same volume target as the structure
    /// and can be the larger of the two. Both are deliberate, both ship, and
    /// only the shell that serves nothing leaves.
    #[test]
    fn a_keepin_boss_larger_than_the_structure_ships_with_it() {
        // A slender structural bar from the support to the load, a fat boss in
        // a lobe of its own, and a speck of debris between them.
        let mut mesh = box_mesh([0.0, 0.0, 0.0], [12.0, 2.0, 2.0], true);
        append(&mut mesh, &box_mesh([20.0; 3], [28.0; 3], true));
        append(
            &mut mesh,
            &box_mesh([16.0, 0.0, 0.0], [17.0, 1.0, 1.0], true),
        );
        validate::validate(&mesh).expect("the fixture must be a valid mesh");

        let anchors = anchors_over(&[
            (AnchorKind::Support, [0.0, 0.0, 0.0], [0.0, 2.0, 2.0]),
            (AnchorKind::Load, [12.0, 0.0, 0.0], [12.0, 2.0, 2.0]),
            (AnchorKind::Keepin, [20.0; 3], [28.0; 3]),
        ]);
        let structure = component_volume(&mesh, &(0..12).collect::<Vec<_>>());
        let boss = component_volume(&mesh, &(12..24).collect::<Vec<_>>());
        assert!(
            boss > structure,
            "the fixture must have the boss outweigh the structure: {boss} against {structure}"
        );

        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchors);
        assert_eq!(report.components, 3);
        assert_eq!(report.bodies.len(), 2, "both lobes have to ship");
        // Largest first in the report, which is the boss, and each says what
        // keeps it.
        assert!(report.bodies[0].anchors.holds(AnchorKind::Keepin));
        assert!(!report.bodies[0].anchors.holds(AnchorKind::Support));
        assert!(report.bodies[1].anchors.holds(AnchorKind::Support));
        assert!(report.bodies[1].anchors.holds(AnchorKind::Load));
        assert_eq!(report.culled_fragments, 1);
        assert_eq!(report.culled_triangles, 12);
        assert!((report.culled_volume_mm3 - 1.0).abs() < 1e-9);
        assert!(report.unserved.is_empty(), "{:?}", report.unserved);

        // The mesh that ships holds both lobes and not the speck.
        assert_eq!(mesh.triangles.len(), 24);
        let stats = validate::validate(&mesh).expect("the culled mesh must validate");
        assert!((stats.volume_mm3 - (structure + boss)).abs() < 1e-9);
        assert_eq!(label_components(&mesh).components, 2);
    }

    /// The shipped cantilever's own shape: a load region buried in the middle of
    /// a keepin pad, several voxels from the nearest surface. The lattice match
    /// cannot see it, and a component that *encloses* the region's material
    /// holds it all the same - without which every such run would be culled
    /// against nothing and warned about for a load it does connect.
    #[test]
    fn a_region_buried_inside_a_body_is_held_by_it_all_the_same() {
        let mut mesh = box_mesh([0.0; 3], [10.0; 3], true);
        append(&mut mesh, &box_mesh([20.0; 3], [22.0; 3], true));
        // The only declared region sits at the centre of the big cube, five
        // millimetres from any of its faces.
        let anchors = anchors_probing(
            &[(AnchorKind::Load, [4.0, 4.0, 4.0], [6.0, 6.0, 6.0])],
            [5.0, 5.0, 5.0],
        );
        assert!(
            anchors.kinds_at([0.0, 5.0, 5.0]).is_empty(),
            "the fixture must have no vertex near the region"
        );

        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchors);
        assert_eq!(report.bodies.len(), 1);
        assert!(report.bodies[0].anchors.holds(AnchorKind::Load));
        assert_eq!(report.culled_fragments, 1, "the small shell holds nothing");
        assert!(report.unserved.is_empty(), "{:?}", report.unserved);
        let stats = validate::validate(&mesh).expect("the culled mesh must validate");
        assert!((stats.volume_mm3 - 1000.0).abs() < 1e-9);
    }

    /// The compound case neither sampling can see: one declared region spanning
    /// two disconnected lobes, with every node and every probe of it in the lobe
    /// that survives. The lobe that leaves takes material the region asked for
    /// with it, and the region reads as *served* - it is, by the other lobe - so
    /// the unserved warning stays silent and cannot be what says so. Every
    /// fragment that leaves is therefore asked directly whether it was inside a
    /// declared region.
    #[test]
    fn a_culled_fragment_inside_a_declared_region_is_named() {
        let mut mesh = box_mesh([0.0; 3], [10.0; 3], true);
        append(&mut mesh, &box_mesh([14.0; 3], [24.0; 3], true));
        validate::validate(&mesh).expect("two closed shells are a valid mesh");
        let anchors = region_over_two_lobes(Shape::axis_aligned_box([0.0; 3], [24.0; 3]));

        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchors);
        assert_eq!(report.bodies.len(), 1, "the first lobe is what is anchored");
        assert_eq!(report.culled_fragments, 1);
        assert!(
            report.unserved.is_empty(),
            "the region is served by the lobe that stayed, which is why the \
             unserved check cannot be the one that reports this: {:?}",
            report.unserved
        );
        assert_eq!(
            report.culled_inside,
            vec![(0, "load 1 of case \"tip\"".to_string())],
            "the fragment that left was inside a declared region and nothing said so"
        );
        validate::validate(&mesh).expect("the culled mesh must validate");

        // The same two lobes with a region that really does stop at the first:
        // debris is debris, and nothing is claimed about it.
        let mut mesh = box_mesh([0.0; 3], [10.0; 3], true);
        append(&mut mesh, &box_mesh([14.0; 3], [24.0; 3], true));
        let anchors = region_over_two_lobes(Shape::axis_aligned_box([0.0; 3], [10.0; 3]));
        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchors);
        assert_eq!(report.culled_fragments, 1);
        assert!(report.culled_inside.is_empty());
        assert!(report.unserved.is_empty());
    }

    #[test]
    fn a_region_no_exported_body_reaches_is_named() {
        // The structure detached from its load: the bar reaches the support and
        // stops half way, and the load region is left on nothing.
        let mut mesh = box_mesh([0.0, 0.0, 0.0], [6.0, 2.0, 2.0], true);
        append(
            &mut mesh,
            &box_mesh([16.0, 0.0, 0.0], [17.0, 1.0, 1.0], true),
        );
        let anchors = anchors_over(&[
            (AnchorKind::Support, [0.0, 0.0, 0.0], [0.0, 2.0, 2.0]),
            (AnchorKind::Load, [12.0, 0.0, 0.0], [12.0, 2.0, 2.0]),
        ]);

        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchors);
        assert_eq!(report.bodies.len(), 1);
        assert_eq!(report.culled_fragments, 1);
        assert_eq!(report.unserved, vec!["load region 2".to_string()]);
        validate::validate(&mesh).expect("the culled mesh must validate");
    }

    #[test]
    fn nothing_is_culled_when_nothing_declared_is_reached() {
        // No shell touches anything declared, so there is no evidence about
        // which of them is the part and none of them is removed.
        let mut mesh = box_mesh([0.0; 3], [10.0; 3], true);
        append(&mut mesh, &box_mesh([20.0; 3], [22.0; 3], true));
        let before = mesh.clone();
        let anchors =
            anchors_over(&[(AnchorKind::Support, [30.0, 30.0, 30.0], [32.0, 32.0, 32.0])]);

        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchors);
        assert_eq!(report.bodies.len(), 2);
        assert_eq!(report.fragments.len(), 2, "both are still purposeless");
        assert_eq!(report.culled_fragments, 0);
        assert_eq!(report.culled_triangles, 0);
        assert_eq!(report.unserved, vec!["support region 1".to_string()]);
        assert_eq!(mesh.vertices, before.vertices);
        assert_eq!(mesh.triangles, before.triangles);

        // And the same with no declared region at all.
        let report = resolve(&mut mesh, IslandPolicy::Cull, &Anchors::none());
        assert_eq!(report.culled_triangles, 0);
        assert!(report.unserved.is_empty());
        assert_eq!(mesh.triangles, before.triangles);
    }

    #[test]
    fn keeping_islands_leaves_the_mesh_exactly_as_it_was() {
        let mut mesh = box_mesh([0.0; 3], [10.0; 3], true);
        append(&mut mesh, &box_mesh([20.0; 3], [22.0; 3], true));
        let before = mesh.clone();

        let report = resolve(&mut mesh, IslandPolicy::Keep, &anchored_part());
        assert_eq!(report.policy, IslandPolicy::Keep);
        assert_eq!(report.components, 2);
        assert_eq!(report.bodies.len(), 2);
        assert_eq!(report.culled_fragments, 0);
        assert_eq!(report.culled_triangles, 0);
        assert_eq!(report.fragments.len(), 1);
        assert_eq!(mesh.vertices, before.vertices);
        assert_eq!(mesh.triangles, before.triangles);
        validate::validate(&mesh).expect("both shells are still closed");
    }

    #[test]
    fn a_fragments_own_cavity_leaves_with_the_fragment() {
        // A part with a cavity, and a hollow fragment floating clear of it: the
        // two inner shells are both negative and only one of them belongs to
        // the part.
        let mut mesh = hollow_cube(10.0, 4.0);
        let mut fragment = box_mesh([20.0; 3], [26.0; 3], true);
        append(&mut fragment, &box_mesh([22.0; 3], [24.0; 3], false));
        append(&mut mesh, &fragment);
        validate::validate(&mesh).expect("two hollow solids are a valid mesh");

        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchored_part());
        assert_eq!(report.components, 4);
        assert_eq!(report.bodies.len(), 1);
        assert_eq!(report.cavity_shells, 1, "the part keeps its own cavity");
        assert_eq!(report.culled_fragments, 1);
        assert_eq!(report.culled_triangles, 24, "the shell and its cavity");
        // 216 mm3 of fragment plus the 8 mm3 its cavity encloses.
        assert!((report.culled_volume_mm3 - 224.0).abs() < 1e-9);
        assert_eq!(mesh.triangles.len(), 24);
        let stats = validate::validate(&mesh).expect("the culled mesh must validate");
        assert!((stats.volume_mm3 - 936.0).abs() < 1e-9);
    }

    #[test]
    fn a_fragment_rattling_inside_a_cavity_is_culled_with_its_own_cavity() {
        // The hard case for a containment test that only asks the part: the
        // fragment sits *inside* the part's cavity, so the part encloses its
        // inner shell too, and only the innermost container is the right answer.
        let mut mesh = hollow_cube(20.0, 12.0);
        let mut rattle = box_mesh([9.0; 3], [11.0; 3], true);
        append(&mut rattle, &box_mesh([9.5; 3], [10.5; 3], false));
        append(&mut mesh, &rattle);
        validate::validate(&mesh).expect("a rattle in a cavity is still a valid mesh");

        let anchors = anchors_over(&[(AnchorKind::Support, [0.0, 0.0, 0.0], [20.0, 20.0, 0.0])]);
        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchors);
        assert_eq!(report.components, 4);
        assert_eq!(report.bodies.len(), 1);
        assert_eq!(report.cavity_shells, 1);
        assert_eq!(report.culled_fragments, 1);
        assert_eq!(report.culled_triangles, 24);
        assert_eq!(mesh.triangles.len(), 24);
        let stats = validate::validate(&mesh).expect("the culled mesh must validate");
        // 8000 mm3 of material less the 1728 mm3 cavity, and nothing rattling.
        assert!((stats.volume_mm3 - 6272.0).abs() < 1e-9);
    }

    #[test]
    fn culling_is_the_same_whatever_order_the_components_come_in() {
        // Triangle order fixes the labels, so the same three shells listed in a
        // different order have to cull to the same part.
        let mut reference = box_mesh([0.0; 3], [10.0; 3], true);
        append(&mut reference, &box_mesh([20.0; 3], [22.0; 3], true));
        let mut rotated = reference.clone();
        rotated.triangles.rotate_left(12);

        let anchors = anchored_part();
        let a = resolve(&mut reference, IslandPolicy::Cull, &anchors);
        let b = resolve(&mut rotated, IslandPolicy::Cull, &anchors);
        assert_eq!(a, b);
        assert_eq!(reference.vertices, rotated.vertices);
        assert_eq!(reference.triangles, rotated.triangles);
    }

    #[test]
    fn an_inside_out_mesh_is_left_for_the_validator() {
        let mut mesh = box_mesh([0.0; 3], [10.0; 3], false);
        append(&mut mesh, &box_mesh([20.0; 3], [22.0; 3], false));
        let before = mesh.clone();
        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchored_part());
        assert!(report.bodies.is_empty(), "an inverted shell is no body");
        assert_eq!(report.cavity_shells, 2);
        assert_eq!(report.culled_triangles, 0);
        assert_eq!(mesh.triangles, before.triangles);
        assert!(
            validate::validate(&mesh)
                .unwrap_err()
                .to_string()
                .contains("non-positive volume")
        );
    }

    #[test]
    fn an_empty_mesh_has_no_components() {
        let mut mesh = Mesh::default();
        let report = resolve(&mut mesh, IslandPolicy::Cull, &anchored_part());
        assert_eq!(report.components, 0);
        assert!(report.bodies.is_empty());
        assert_eq!(report.largest_fragment_mm3(), None);
        assert_eq!(report.fragment_volume_mm3(), 0.0);
        assert!(mesh.is_empty());
    }

    #[test]
    fn a_closed_shell_encloses_its_inside_and_nothing_else() {
        let mesh = box_mesh([0.0; 3], [10.0; 3], true);
        let all: Vec<usize> = (0..mesh.triangles.len()).collect();
        assert!(encloses(&mesh, &all, [5.0, 5.0, 5.0]));
        // Just inside a face, and just outside it.
        assert!(encloses(&mesh, &all, [5.0, 5.0, 9.999]));
        assert!(!encloses(&mesh, &all, [5.0, 5.0, 10.001]));
        assert!(!encloses(&mesh, &all, [-1.0, 5.0, 5.0]));
        // Winding does not change what a shell contains.
        let flipped = box_mesh([0.0; 3], [10.0; 3], false);
        assert!(encloses(&flipped, &all, [5.0, 5.0, 5.0]));
        assert!(!encloses(&flipped, &all, [50.0, 5.0, 5.0]));
        // A point on the axis of a face, where a ray cast would have to choose
        // a direction, reads the same.
        assert!(encloses(&mesh, &all, [5.0, 5.0, 0.001]));
    }

    #[test]
    fn a_vertex_is_matched_to_the_lattice_cell_it_lies_in() {
        // Half a voxel of slack around a declared region, and nothing beyond it:
        // enough for the iso nudge and for smoothing, not enough to adopt a
        // shell a voxel away.
        let anchors = anchors_over(&[(AnchorKind::Load, [5.0, 5.0, 5.0], [5.0, 5.0, 5.0])]);
        assert!(anchors.kinds_at([5.0, 5.0, 5.0]).holds(AnchorKind::Load));
        assert!(anchors.kinds_at([4.4, 5.2, 5.9]).holds(AnchorKind::Load));
        assert!(!anchors.kinds_at([3.5, 5.0, 5.0]).holds(AnchorKind::Load));
        assert!(!anchors.kinds_at([5.0, 5.0, 6.5]).holds(AnchorKind::Load));
        // Outside the lattice altogether, and not a number, are both nothing.
        assert!(anchors.kinds_at([1e9, 0.0, 0.0]).is_empty());
        assert!(anchors.kinds_at([f64::NAN, 5.0, 5.0]).is_empty());
        // An empty declaration anchors nothing anywhere.
        assert!(Anchors::none().kinds_at([5.0, 5.0, 5.0]).is_empty());
        assert!(Anchors::none().is_empty());
    }

    #[test]
    fn an_anchor_set_holds_the_kinds_that_were_put_in_it() {
        let mut set = AnchorSet::none();
        assert!(set.is_empty() && set.kinds().count() == 0);
        set.insert(AnchorKind::Keepin);
        let mut other = AnchorSet::none();
        other.insert(AnchorKind::Support);
        set.union(other);
        assert!(!set.is_empty());
        assert!(set.holds(AnchorKind::Support) && set.holds(AnchorKind::Keepin));
        assert!(!set.holds(AnchorKind::Load));
        // Report order, whatever order they went in.
        assert_eq!(
            set.kinds().map(|k| k.label()).collect::<Vec<_>>(),
            vec!["support", "keepin"]
        );
    }
}
