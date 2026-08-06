//! Growth symmetry: the fundamental domain a symmetric run grows in, and the
//! transforms that replicate what it grew.
//!
//! Space colonization is stochastic, so a four-fold problem - a square table on
//! four identical corner feet - grows four *different* legs. They are all
//! sound and none of them is the same, which is organic and is exactly what a
//! user reported wanting an alternative to. `[growth.symmetry]` is that
//! alternative: **grow one sector of the domain and replicate exact copies of
//! it**, so a symmetric problem can produce a symmetric part.
//!
//! ## What "symmetric" is exact about, and what it is not
//!
//! The **skeleton** is exact under every symmetry this module supports: the
//! copies are the images of one node set under matrices that are themselves
//! exact, so a node's image is a node to the last bit.
//!
//! The **rasterized field** is a different question, because it is the skeleton
//! sampled at cell centres. It is exact only when the transform maps cell
//! centres onto cell centres, which is the case for every mirror, for a half
//! turn, and for a quarter turn whose two plane axes hold cell counts of the
//! same parity (a square footprint always does) - see
//! [`Symmetry::maps_cell_centres`]. Any other angle takes a cell centre to a
//! point *inside* another cell rather than to its centre, so the two are
//! sampled a fraction of a voxel apart. The field is then symmetric only to
//! within the voxel size: cells more than a voxel from any surface still match
//! exactly, and cells in the surface band can differ by as much as a whole
//! density, because that is what the smoothstep spans over one voxel. A finer
//! `voxel_size_mm` is the remedy, and the exported surface is smooth through it
//! either way.
//!
//! ## The fundamental domain
//!
//! Everything is measured about the **centre of the grid block**, which is the
//! domain bounding box padded symmetrically out to whole voxels, so it is the
//! domain's own bounding box centre. A mirror plane through that centre maps
//! cell centres onto cell centres exactly: cell `i` on an axis of `n` cells maps
//! to cell `n - 1 - i`.
//!
//! * **Mirror.** One plane normal to `x` keeps the half `x <= centre.x`; a
//!   second plane keeps the quarter that also satisfies `y <= centre.y`. The
//!   half-spaces are **closed**: an odd number of cells on an axis puts a whole
//!   layer of cell centres exactly on the plane, and a layer that belonged to
//!   neither half would be a one-voxel gap through the middle of the part. A
//!   layer that belongs to both is unioned with its own image instead, which
//!   the smooth minimum absorbs.
//! * **Rotational.** The sector is `0 <= atan2(v, u) < 2 pi / order` in the
//!   plane perpendicular to the axis, `u` and `v` being the two axes after it in
//!   cyclic order. Half-open, because the copies of a rotation *partition* the
//!   turn: what the sector's upper edge excludes, the next copy covers.
//!
//! ## The transforms
//!
//! Copy 0 is the fundamental domain itself and is **not transformed at all**,
//! so nothing the run grew is moved by a rounding error. Every other copy is
//! `p -> centre + M (p - centre)`, with `M` a reflection (a diagonal of `+-1`,
//! exact) or a rotation. Rotations by a whole quarter turn - every copy of a
//! two-fold or four-fold symmetry, and the quarter-turn copies of an eight- or
//! twelve-fold one - take their sine and cosine from a table rather than from
//! the trigonometric functions, so those are exact too and a four-fold part is
//! symmetric to the last bit of its skeleton.
//!
//! Replication order is fixed: the identity, then the transforms in the order
//! they are built here. That is what keeps a symmetric run as deterministic as
//! an asymmetric one, because the smooth minimum that unions the struts is not
//! associative and therefore not indifferent to the order they arrive in.

use crate::config::{Axis, SymmetryParams};
use crate::engine::growth::skeleton::{Node, Skeleton};
use crate::geometry::{self, Aabb, Mat3, Vec3};
use crate::grid::Grid;

/// The identity, which is always copy 0.
const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Cosines of the four quarter turns, so a quarter turn is exact.
const QUARTER_COS: [f64; 4] = [1.0, 0.0, -1.0, 0.0];
/// Sines of the four quarter turns.
const QUARTER_SIN: [f64; 4] = [0.0, 1.0, 0.0, -1.0];

/// A symmetry group acting on the design domain.
#[derive(Debug, Clone)]
pub struct Symmetry {
    params: SymmetryParams,
    center: Vec3,
    tolerance_mm: f64,
    transforms: Vec<Mat3>,
}

/// Diagonal reflection matrix that negates every axis in `axes`.
///
/// Exact: every entry is `0`, `1` or `-1`.
fn reflection(axes: &[Axis]) -> Mat3 {
    let mut matrix = IDENTITY;
    for axis in axes {
        matrix[axis.dof()][axis.dof()] = -1.0;
    }
    matrix
}

/// Rotation by `step` of `order` whole turns about `axis`.
///
/// The right hand rule about the positive axis. A step that lands on a whole
/// quarter turn takes its sine and cosine from [`QUARTER_COS`] and
/// [`QUARTER_SIN`], because `(std::f64::consts::TAU / 4.0).cos()` is `6.1e-17`
/// rather than zero and a structure that is symmetric only to within that is
/// not what this module promises.
fn rotation(axis: Axis, step: usize, order: usize) -> Mat3 {
    let (sin, cos) = if (4 * step).is_multiple_of(order) {
        let quarter = (4 * step / order) % 4;
        (QUARTER_SIN[quarter], QUARTER_COS[quarter])
    } else {
        (std::f64::consts::TAU * step as f64 / order as f64).sin_cos()
    };
    let (w, u, v) = plane_of(axis);
    let mut matrix = [[0.0f64; 3]; 3];
    matrix[w][w] = 1.0;
    matrix[u][u] = cos;
    matrix[u][v] = -sin;
    matrix[v][u] = sin;
    matrix[v][v] = cos;
    matrix
}

/// An axis and the two axes of the plane it turns, in cyclic order.
fn plane_of(axis: Axis) -> (usize, usize, usize) {
    let w = axis.dof();
    (w, (w + 1) % 3, (w + 2) % 3)
}

impl Symmetry {
    /// Build the symmetry `params` describes about the centre of `bounds`.
    ///
    /// `tolerance_mm` is how far past the boundary of the fundamental domain a
    /// point may sit and still be counted inside it: a sliver that exists to
    /// settle the sign of a rounding error on a point that is *on* the
    /// boundary, and that decides nothing else. See
    /// [`crate::constants::GROWTH_SYMMETRY_BOUNDARY_EPSILON_VOXELS`].
    pub fn new(params: SymmetryParams, bounds: &Aabb, tolerance_mm: f64) -> Symmetry {
        let mut center = [0.0f64; 3];
        for (d, slot) in center.iter_mut().enumerate() {
            *slot = 0.5 * (bounds.min[d] + bounds.max[d]);
        }
        let transforms = match params {
            SymmetryParams::Mirror {
                first,
                second: None,
            } => vec![IDENTITY, reflection(&[first])],
            SymmetryParams::Mirror {
                first,
                second: Some(second),
            } => vec![
                IDENTITY,
                reflection(&[first]),
                reflection(&[second]),
                reflection(&[first, second]),
            ],
            SymmetryParams::Rotational { order, axis } => {
                (0..order).map(|step| rotation(axis, step, order)).collect()
            }
        };
        debug_assert_eq!(
            transforms.len(),
            params.sectors(),
            "the transform list has to be the sector count the configuration resolved"
        );
        Symmetry {
            params,
            center,
            tolerance_mm: tolerance_mm.max(0.0),
            transforms,
        }
    }

    /// The resolved controls this was built from.
    pub fn params(&self) -> SymmetryParams {
        self.params
    }

    /// The point everything is measured about: the centre of the grid block.
    pub fn center(&self) -> Vec3 {
        self.center
    }

    /// How many copies of the fundamental domain make up the whole structure,
    /// the fundamental one included.
    pub fn sectors(&self) -> usize {
        self.transforms.len()
    }

    /// The linear part of copy `index`, which for copy 0 is the identity.
    pub fn matrix(&self, index: usize) -> Mat3 {
        self.transforms[index]
    }

    /// True when every copy takes a cell centre of `grid` to a cell centre, so
    /// the **rasterized** field is symmetric to the bit rather than to within a
    /// voxel.
    ///
    /// The skeleton is exact either way; this is about where it is sampled.
    ///
    /// * A reflection about the block centre takes cell `i` of an axis of `n`
    ///   cells to cell `n - 1 - i`, whatever `n` is, so every mirror qualifies.
    /// * A half turn is two such reflections at once, so `order = 2` does too.
    /// * A quarter turn swaps the two axes of its plane: cell `j` of the one
    ///   goes to cell `(n_u + n_v) / 2 - j - 1` of the other, which is a cell
    ///   index exactly when `n_u + n_v` is even. A square footprint, where the
    ///   two counts are equal, always is.
    /// * Every other angle lands inside a cell rather than on its centre, so
    ///   the field is resampled and cells within about a voxel of a surface can
    ///   differ by as much as the smoothstep spans there.
    pub fn maps_cell_centres(&self, grid: &Grid) -> bool {
        match self.params {
            SymmetryParams::Mirror { .. } => true,
            SymmetryParams::Rotational { order, axis } => {
                let (_, u, v) = plane_of(axis);
                let counts = [grid.nx, grid.ny, grid.nz];
                match order {
                    2 => true,
                    4 => (counts[u] + counts[v]).is_multiple_of(2),
                    _ => false,
                }
            }
        }
    }

    /// Where copy `index` puts the point `p`.
    ///
    /// Copy 0 is the fundamental domain itself and is returned untouched, so
    /// the structure the run grew is in the result exactly as it was grown -
    /// `centre + (p - centre)` is not `p` in floating point.
    pub fn image(&self, index: usize, p: Vec3) -> Vec3 {
        if index == 0 {
            return p;
        }
        geometry::sum(
            self.center,
            geometry::rotate(
                &self.transforms[index],
                geometry::difference(p, self.center),
            ),
        )
    }

    /// True when `p` lies in the fundamental domain.
    ///
    /// See the module documentation for why the mirror half-spaces are closed
    /// and the rotational sector is half-open. The tolerance the symmetry was
    /// built with is added to the closed side of each: past the mirror plane,
    /// and below the sector's lower edge, which are the two places a point that
    /// is *on* the boundary can be rounded to.
    pub fn contains(&self, p: Vec3) -> bool {
        match self.params {
            SymmetryParams::Mirror { first, second } => {
                self.in_half(first, p) && second.is_none_or(|axis| self.in_half(axis, p))
            }
            SymmetryParams::Rotational { order, axis } => {
                let (_, u, v) = plane_of(axis);
                let (du, dv) = (p[u] - self.center[u], p[v] - self.center[v]);
                let radius = du.hypot(dv);
                // The axis itself is shared by every sector, and no angle there
                // means anything.
                if radius <= self.tolerance_mm {
                    return true;
                }
                let slack = (self.tolerance_mm / radius).atan();
                let mut angle = dv.atan2(du);
                if angle < -slack {
                    angle += std::f64::consts::TAU;
                }
                angle >= -slack && angle < std::f64::consts::TAU / order as f64
            }
        }
    }

    /// True on the keeping side of one mirror plane, the plane itself included.
    fn in_half(&self, axis: Axis, p: Vec3) -> bool {
        p[axis.dof()] <= self.center[axis.dof()] + self.tolerance_mm
    }

    /// The whole structure: `skeleton` followed by one transformed copy of it
    /// per further sector.
    ///
    /// Copy `c` occupies nodes `c * skeleton.len() ..`, so every parent index
    /// still comes before its child's and the flow accumulation, the pruning
    /// and the rasterizer all go on working on the result unchanged.
    pub fn replicate(&self, skeleton: &Skeleton) -> Skeleton {
        let mut replicated = Skeleton {
            nodes: Vec::with_capacity(skeleton.len() * self.sectors()),
        };
        for index in 0..self.sectors() {
            let offset = index * skeleton.len();
            for node in &skeleton.nodes {
                replicated.nodes.push(Node {
                    position: self.image(index, node.position),
                    parent: node.parent.map(|parent| parent + offset),
                });
            }
        }
        replicated
    }

    /// Per-node values of `skeleton` repeated once per sector, to line up with
    /// [`Symmetry::replicate`].
    pub fn repeat(&self, values: &[f64]) -> Vec<f64> {
        values.repeat(self.sectors())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use crate::geometry::Aabb;

    /// A block from the origin to (80, 40, 20), whose centre is (40, 20, 10).
    fn bounds() -> Aabb {
        Aabb {
            min: [0.0, 0.0, 0.0],
            max: [80.0, 40.0, 20.0],
        }
    }

    /// The boundary slack a 1 mm grid would give it, which is what the engine
    /// passes and is far below anything these fixtures measure.
    const SLACK: f64 = constants::GROWTH_SYMMETRY_BOUNDARY_EPSILON_VOXELS;

    fn mirror(first: Axis, second: Option<Axis>) -> Symmetry {
        Symmetry::new(SymmetryParams::Mirror { first, second }, &bounds(), SLACK)
    }

    fn rotational(order: usize, axis: Axis) -> Symmetry {
        Symmetry::new(SymmetryParams::Rotational { order, axis }, &bounds(), SLACK)
    }

    #[test]
    fn a_mirror_reflects_exactly_and_leaves_the_fundamental_copy_alone() {
        let symmetry = mirror(Axis::X, None);
        assert_eq!(symmetry.center(), [40.0, 20.0, 10.0]);
        assert_eq!(symmetry.sectors(), 2);
        assert_eq!(symmetry.matrix(0), IDENTITY);
        assert_eq!(
            symmetry.matrix(1),
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );

        // Hand computed: 12 mm to the left of the centre goes 12 mm to the
        // right of it, and nothing else moves.
        let p = [28.0, 5.0, 3.0];
        assert_eq!(
            symmetry.image(0, p),
            p,
            "copy 0 may not move a point at all"
        );
        assert_eq!(symmetry.image(1, p), [52.0, 5.0, 3.0]);
        // A point on the plane is its own image, exactly.
        assert_eq!(symmetry.image(1, [40.0, 5.0, 3.0]), [40.0, 5.0, 3.0]);
        // And reflecting twice is the identity, exactly.
        assert_eq!(symmetry.image(1, symmetry.image(1, p)), p);
    }

    #[test]
    fn two_planes_quarter_the_domain_in_a_fixed_order() {
        let symmetry = mirror(Axis::X, Some(Axis::Y));
        assert_eq!(symmetry.sectors(), 4);
        let p = [28.0, 8.0, 3.0];
        assert_eq!(
            (0..4).map(|c| symmetry.image(c, p)).collect::<Vec<_>>(),
            vec![
                [28.0, 8.0, 3.0],
                [52.0, 8.0, 3.0],
                [28.0, 32.0, 3.0],
                [52.0, 32.0, 3.0]
            ]
        );
        // The quarter is the low corner, and the planes belong to it.
        assert!(symmetry.contains([28.0, 8.0, 3.0]));
        assert!(
            symmetry.contains([40.0, 20.0, 3.0]),
            "the planes are closed"
        );
        assert!(!symmetry.contains([52.0, 8.0, 3.0]));
        assert!(!symmetry.contains([28.0, 32.0, 3.0]));
        assert!(!symmetry.contains([52.0, 32.0, 3.0]));
        // Every point off the planes is brought into the fundamental domain by
        // exactly one copy, which is what makes the four copies a replication
        // of it rather than an overlap or a gap.
        for x in [1.0, 39.0, 41.0, 79.0] {
            for y in [1.0, 19.0, 21.0, 39.0] {
                let p = [x, y, 5.0];
                let owners = (0..4)
                    .filter(|&c| symmetry.contains(symmetry.image(c, p)))
                    .count();
                assert_eq!(owners, 1, "{p:?} is claimed by {owners} sectors");
            }
        }
    }

    #[test]
    fn quarter_turns_are_exact_and_a_full_turn_comes_back_to_the_start() {
        let symmetry = rotational(4, Axis::Z);
        assert_eq!(symmetry.sectors(), 4);
        assert_eq!(symmetry.matrix(0), IDENTITY);
        assert_eq!(
            symmetry.matrix(1),
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            "a quarter turn about z has to be exact"
        );
        assert_eq!(
            symmetry.matrix(2),
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        );

        // Hand computed about (40, 20): 10 mm along +x from the centre turns
        // into 10 mm along +y from it, exactly.
        let p = [50.0, 20.0, 7.0];
        assert_eq!(symmetry.image(1, p), [40.0, 30.0, 7.0]);
        assert_eq!(symmetry.image(2, p), [30.0, 20.0, 7.0]);
        assert_eq!(symmetry.image(3, p), [40.0, 10.0, 7.0]);
        // Four quarter turns are the identity, to the last bit.
        let mut walked = p;
        for _ in 0..4 {
            walked = symmetry.image(1, walked);
        }
        assert_eq!(walked, p);

        // Two-fold is a half turn, also exact; six-fold is not a quarter turn
        // and only has to be a rotation.
        assert_eq!(
            rotational(2, Axis::Z).matrix(1),
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        );
        let six = rotational(6, Axis::Z);
        let m = six.matrix(1);
        let angle = std::f64::consts::TAU / 6.0;
        assert!((m[0][0] - angle.cos()).abs() < 1e-15);
        assert!((m[1][0] - angle.sin()).abs() < 1e-15);
        // Orthogonal: every row is a unit vector.
        for (index, row) in m.iter().enumerate() {
            let norm: f64 = row.iter().map(|value| value * value).sum();
            assert!(
                (norm - 1.0).abs() < 1e-15,
                "row {index} is not a unit vector"
            );
        }
        // And `order` of them come back to the start.
        let mut walked = p;
        for _ in 0..6 {
            walked = six.image(1, walked);
        }
        for d in 0..3 {
            assert!((walked[d] - p[d]).abs() < 1e-12, "{walked:?} against {p:?}");
        }
    }

    #[test]
    fn a_rotational_sector_is_half_open_and_covers_the_turn_once() {
        let symmetry = rotational(4, Axis::Z);
        let centre = symmetry.center();
        // Angles measured about the centre in the xy plane.
        let at = |degrees: f64| {
            let radians = degrees.to_radians();
            [
                centre[0] + 10.0 * radians.cos(),
                centre[1] + 10.0 * radians.sin(),
                5.0,
            ]
        };
        assert!(symmetry.contains(at(0.0)), "the lower edge belongs to it");
        assert!(symmetry.contains(at(45.0)));
        assert!(!symmetry.contains(at(90.0)), "the upper edge does not");
        assert!(!symmetry.contains(at(180.0)));
        assert!(!symmetry.contains(at(-45.0)));
        // The axis itself is in the sector: it is shared by all of them.
        assert!(symmetry.contains(centre));
        // Every direction is brought into the sector by exactly one copy, so
        // the copies partition the turn.
        for degrees in (0..360).step_by(7) {
            let p = at(degrees as f64);
            let owners = (0..4)
                .filter(|&c| symmetry.contains(symmetry.image(c, p)))
                .count();
            assert_eq!(owners, 1, "{degrees} degrees is claimed {owners} times");
        }
        // An axis other than z turns the plane perpendicular to it.
        let about_x = rotational(4, Axis::X);
        assert_eq!(about_x.image(1, [10.0, 30.0, 10.0]), [10.0, 20.0, 20.0]);
    }

    /// A point *on* the boundary is the common case, not the corner case: a
    /// centred load, a central support, and - whenever an axis has an odd
    /// number of cells - a whole layer of cell centres. It has to belong to the
    /// fundamental domain whichever way its last bit rounds.
    #[test]
    fn a_point_on_the_boundary_belongs_to_the_fundamental_domain_either_way_it_rounds() {
        let symmetry = mirror(Axis::X, None);
        assert!(symmetry.contains([40.0, 5.0, 3.0]));
        assert!(symmetry.contains([40.0 + 1e-12, 5.0, 3.0]));
        // The slack settles a rounding error and nothing else: a hair past the
        // plane is still a hair, but a hundredth of a millimetre is geometry.
        assert!(!symmetry.contains([40.01, 5.0, 3.0]));

        // The same at a rotational sector's lower edge, where an angle of
        // -1e-16 would otherwise wrap to nearly a whole turn.
        let symmetry = rotational(4, Axis::Z);
        let c = symmetry.center();
        assert!(symmetry.contains([c[0] + 10.0, c[1], 5.0]));
        assert!(symmetry.contains([c[0] + 10.0, c[1] - 1e-12, 5.0]));
        assert!(!symmetry.contains([c[0] + 10.0, c[1] - 0.01, 5.0]));
        // And the axis belongs to it, whatever the angle there rounds to.
        assert!(symmetry.contains([c[0], c[1], 5.0]));
        assert!(symmetry.contains([c[0], c[1], -1000.0]));
    }

    /// Which symmetries the *voxel lattice* is exact under, which is a
    /// different and weaker question than which ones the skeleton is exact
    /// under - the skeleton is exact under all of them.
    #[test]
    fn only_the_transforms_that_land_on_cell_centres_are_exact_on_the_lattice() {
        use crate::engine::growth::fixtures::design_grid;

        let square = design_grid(8);
        assert!(mirror(Axis::X, None).maps_cell_centres(&square));
        assert!(mirror(Axis::X, Some(Axis::Y)).maps_cell_centres(&square));
        assert!(rotational(2, Axis::Z).maps_cell_centres(&square));
        assert!(rotational(4, Axis::Z).maps_cell_centres(&square));
        // A sixth or a third of a turn takes a cell centre inside a cell, not
        // to its centre; so does a twelfth, even though three of its copies are
        // quarter turns.
        for order in [3, 5, 6, 7, 8, 12] {
            assert!(
                !rotational(order, Axis::Z).maps_cell_centres(&square),
                "{order}-fold was claimed exact on the lattice"
            );
        }
        // An odd number of cells on one axis of the turned plane breaks the
        // quarter turn: the swap lands half a voxel out.
        let mut oblong = design_grid(8);
        oblong.nz = 7;
        oblong.cells.truncate(oblong.nx * oblong.ny * oblong.nz);
        assert!(!rotational(4, Axis::X).maps_cell_centres(&oblong));
        assert!(rotational(2, Axis::X).maps_cell_centres(&oblong));
        // A mirror never cares.
        assert!(mirror(Axis::Z, None).maps_cell_centres(&oblong));
    }

    #[test]
    fn replication_copies_the_skeleton_once_per_sector_and_keeps_the_forest() {
        let mut skeleton = Skeleton::new();
        let root = skeleton.add_root([10.0, 5.0, 0.0]);
        let mid = skeleton.add_child(root, [20.0, 5.0, 5.0]);
        skeleton.add_child(mid, [30.0, 5.0, 10.0]);

        let symmetry = mirror(Axis::X, None);
        let replicated = symmetry.replicate(&skeleton);
        assert_eq!(replicated.len(), 6);
        assert_eq!(replicated.segment_count(), 4);
        // The fundamental copy is untouched, node for node.
        for node in 0..skeleton.len() {
            assert_eq!(replicated.nodes[node], skeleton.nodes[node]);
        }
        // The mirrored copy is the hand computed reflection, and its parents
        // point inside itself.
        assert_eq!(replicated.position(3), [70.0, 5.0, 0.0]);
        assert_eq!(replicated.position(4), [60.0, 5.0, 5.0]);
        assert_eq!(replicated.position(5), [50.0, 5.0, 10.0]);
        assert_eq!(replicated.nodes[3].parent, None);
        assert_eq!(replicated.nodes[4].parent, Some(3));
        assert_eq!(replicated.nodes[5].parent, Some(4));
        // Every parent still comes before its child, which is what the flow
        // pass and the pruning rely on.
        for (index, node) in replicated.nodes.iter().enumerate() {
            assert!(node.parent.is_none_or(|p| p < index));
        }
        // Per-node values line up with the copies.
        assert_eq!(
            symmetry.repeat(&[1.0, 2.0, 3.0]),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );

        // And the replicated node set is invariant under the transform: the
        // image of every node is a node.
        for index in 0..replicated.len() {
            let image = symmetry.image(1, replicated.position(index));
            assert!(
                (0..replicated.len()).any(|other| {
                    geometry::length(geometry::difference(replicated.position(other), image))
                        < 1e-12
                }),
                "the image of node {index} is not in the skeleton"
            );
        }
    }

    #[test]
    fn an_empty_skeleton_replicates_to_an_empty_one() {
        let symmetry = rotational(3, Axis::Z);
        let replicated = symmetry.replicate(&Skeleton::new());
        assert!(replicated.is_empty());
        assert!(symmetry.repeat(&[]).is_empty());
    }
}
