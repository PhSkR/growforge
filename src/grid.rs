//! Axis aligned voxel grid: construction from a bounding box, cell
//! classification against the domain CSG, and index arithmetic shared by the
//! finite element model and the mesh exporter.
//!
//! Index conventions used everywhere in growforge:
//! * element `(i, j, k)` has linear index `i + nx * (j + ny * k)`
//! * node `(i, j, k)` has linear index `i + (nx+1) * (j + (ny+1) * k)`
//! * degree of freedom `d` of node `n` has index `3 * n + d`
//! * local element node `l` sits at offset `(l & 1, (l >> 1) & 1, (l >> 2) & 1)`

use rayon::prelude::*;

use crate::constants;
use crate::geometry::{Aabb, Csg, ShapeUnion, Vec3};

/// Role of a voxel in the optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKind {
    /// Outside the domain or inside a keepout: carries no stiffness.
    Void,
    /// A design variable: density is free in [0, 1].
    Design,
    /// Forced solid: density is pinned at 1 and is not a design variable.
    Solid,
}

/// Cell counts by kind, reported by `growforge check`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CellCounts {
    /// Number of void cells.
    pub void: usize,
    /// Number of design cells.
    pub design: usize,
    /// Number of forced solid cells.
    pub solid: usize,
}

impl CellCounts {
    /// Cells that carry stiffness (design plus solid).
    pub fn active(&self) -> usize {
        self.design + self.solid
    }
}

/// A uniform cubic voxel grid covering the design space.
#[derive(Debug, Clone)]
pub struct Grid {
    /// World position in millimetres of node (0, 0, 0).
    pub origin: Vec3,
    /// Edge length of a cell in millimetres.
    pub h: f64,
    /// Element count along x.
    pub nx: usize,
    /// Element count along y.
    pub ny: usize,
    /// Element count along z.
    pub nz: usize,
    /// Classification of every cell, in element index order.
    pub cells: Vec<CellKind>,
}

/// Cells [`Grid::from_bounds`] lays along one axis of length `extent` at a
/// voxel size of `h`.
///
/// In `f64` rather than `usize` because the caller may be asking about a grid
/// that cannot be built: a voxel size small enough, or a cell target large
/// enough, derives counts whose product is far past `usize`, and a saturating
/// cast would hide exactly the case worth reporting. A degenerate ratio comes
/// back as it is, for the same reason.
pub fn axis_cells(extent: f64, h: f64) -> f64 {
    let count = (extent / h).ceil();
    if count.is_nan() {
        return count;
    }
    count.max(constants::MIN_CELLS_PER_AXIS as f64)
}

/// Cells [`Grid::from_bounds`] would lay over `bounds` at a voxel size of `h`.
///
/// This is what decides whether the grid is built at all; see
/// [`constants::MAX_GRID_CELLS`] and `Problem::build`.
pub fn cell_count_for(bounds: &Aabb, h: f64) -> f64 {
    let extent = bounds.extent();
    axis_cells(extent[0], h) * axis_cells(extent[1], h) * axis_cells(extent[2], h)
}

impl Grid {
    /// Lay a grid of cubic cells of size `h` over `bounds`, padding the box
    /// symmetrically so that it fits inside a whole number of cells.
    ///
    /// The caller is expected to have checked the size of the grid first:
    /// laying one out allocates it. `Problem::build` does that against
    /// [`constants::MAX_GRID_CELLS`].
    pub fn from_bounds(bounds: &Aabb, h: f64) -> Grid {
        let extent = bounds.extent();
        let mut n = [0usize; 3];
        let mut origin = [0.0f64; 3];
        for d in 0..3 {
            // The second floor catches a degenerate extent or voxel size, whose
            // ratio is not a number and casts to zero.
            n[d] = (axis_cells(extent[d], h) as usize).max(constants::MIN_CELLS_PER_AXIS);
            let covered = n[d] as f64 * h;
            origin[d] = bounds.min[d] - 0.5 * (covered - extent[d]);
        }
        Grid {
            origin,
            h,
            nx: n[0],
            ny: n[1],
            nz: n[2],
            cells: vec![CellKind::Void; n[0] * n[1] * n[2]],
        }
    }

    /// Classify every cell by its centre point.
    ///
    /// Precedence is keepout, then keepin, then domain: a centre inside a
    /// keepout is always void, a centre inside a keepin is always solid, and
    /// everything else inside the domain becomes a design variable. Returns
    /// the number of cells that fell inside both a keepout and a keepin, which
    /// the caller turns into an overlap warning.
    pub fn classify(&mut self, domain: &Csg, keepout: &ShapeUnion, keepin: &ShapeUnion) -> usize {
        let nx = self.nx;
        let ny = self.ny;
        let h = self.h;
        let origin = self.origin;
        let overlaps: usize = self
            .cells
            .par_iter_mut()
            .enumerate()
            .map(|(e, cell)| {
                let i = e % nx;
                let j = (e / nx) % ny;
                let k = e / (nx * ny);
                let c = [
                    origin[0] + (i as f64 + 0.5) * h,
                    origin[1] + (j as f64 + 0.5) * h,
                    origin[2] + (k as f64 + 0.5) * h,
                ];
                let in_keepout = keepout.contains(c);
                let in_keepin = keepin.contains(c);
                *cell = if in_keepout {
                    CellKind::Void
                } else if in_keepin {
                    CellKind::Solid
                } else if domain.contains(c) {
                    CellKind::Design
                } else {
                    CellKind::Void
                };
                usize::from(in_keepout && in_keepin)
            })
            .sum();
        overlaps
    }

    /// Total number of cells.
    pub fn n_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Node count along x.
    pub fn nnx(&self) -> usize {
        self.nx + 1
    }

    /// Node count along y.
    pub fn nny(&self) -> usize {
        self.ny + 1
    }

    /// Node count along z.
    pub fn nnz(&self) -> usize {
        self.nz + 1
    }

    /// Total number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nnx() * self.nny() * self.nnz()
    }

    /// Total number of displacement degrees of freedom.
    pub fn n_dof(&self) -> usize {
        self.n_nodes() * constants::DOF_PER_NODE
    }

    /// Linear index of element `(i, j, k)`.
    pub fn cell_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// Grid coordinates of element `e`.
    pub fn cell_ijk(&self, e: usize) -> (usize, usize, usize) {
        let i = e % self.nx;
        let j = (e / self.nx) % self.ny;
        let k = e / (self.nx * self.ny);
        (i, j, k)
    }

    /// Linear index of node `(i, j, k)`.
    pub fn node_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nnx() * (j + self.nny() * k)
    }

    /// Grid coordinates of node `n`.
    pub fn node_ijk(&self, n: usize) -> (usize, usize, usize) {
        let i = n % self.nnx();
        let j = (n / self.nnx()) % self.nny();
        let k = n / (self.nnx() * self.nny());
        (i, j, k)
    }

    /// World position of the centre of element `(i, j, k)`.
    pub fn cell_center(&self, i: usize, j: usize, k: usize) -> Vec3 {
        [
            self.origin[0] + (i as f64 + 0.5) * self.h,
            self.origin[1] + (j as f64 + 0.5) * self.h,
            self.origin[2] + (k as f64 + 0.5) * self.h,
        ]
    }

    /// World position of node `(i, j, k)`.
    pub fn node_pos(&self, i: usize, j: usize, k: usize) -> Vec3 {
        [
            self.origin[0] + i as f64 * self.h,
            self.origin[1] + j as f64 * self.h,
            self.origin[2] + k as f64 * self.h,
        ]
    }

    /// The eight node indices of element `(i, j, k)` in local node order.
    pub fn element_nodes(&self, i: usize, j: usize, k: usize) -> [usize; 8] {
        let mut nodes = [0usize; 8];
        for (l, node) in nodes.iter_mut().enumerate() {
            let di = l & 1;
            let dj = (l >> 1) & 1;
            let dk = (l >> 2) & 1;
            *node = self.node_index(i + di, j + dj, k + dk);
        }
        nodes
    }

    /// The **material** cells touching any of `nodes`, ascending and
    /// deduplicated.
    ///
    /// A support or load region is a set of nodes and everything structural is
    /// a set of cells, so this is the bridge between the two: a node is a corner
    /// of up to eight cells, and the ones a keepout made void are left out
    /// because no pass may put material there.
    ///
    /// One answer to "which cells is this region" for every consumer - the
    /// growth engine's terminals, the guide wireframe's, and the purposes the
    /// trim pass protects - so a region can never mean two different sets of
    /// cells in one run.
    pub fn cells_touching_nodes(&self, nodes: &[usize]) -> Vec<usize> {
        let mut cells = Vec::new();
        for &node in nodes {
            let (ni, nj, nk) = self.node_ijk(node);
            for dk in 0..2 {
                for dj in 0..2 {
                    for di in 0..2 {
                        let (i, j, k) = (ni + di, nj + dj, nk + dk);
                        // A node at coordinate `i` is a corner of cells `i - 1`
                        // and `i`, of which either may be off the grid at a
                        // face.
                        if i == 0 || j == 0 || k == 0 {
                            continue;
                        }
                        let (i, j, k) = (i - 1, j - 1, k - 1);
                        if i >= self.nx || j >= self.ny || k >= self.nz {
                            continue;
                        }
                        let cell = self.cell_index(i, j, k);
                        if self.cells[cell] != CellKind::Void {
                            cells.push(cell);
                        }
                    }
                }
            }
        }
        cells.sort_unstable();
        cells.dedup();
        cells
    }

    /// Cell counts by kind.
    pub fn counts(&self) -> CellCounts {
        let mut c = CellCounts::default();
        for cell in &self.cells {
            match cell {
                CellKind::Void => c.void += 1,
                CellKind::Design => c.design += 1,
                CellKind::Solid => c.solid += 1,
            }
        }
        c
    }

    /// Bounding box of the grid block itself.
    pub fn bounds(&self) -> Aabb {
        Aabb {
            min: self.origin,
            max: [
                self.origin[0] + self.nx as f64 * self.h,
                self.origin[1] + self.ny as f64 * self.h,
                self.origin[2] + self.nz as f64 * self.h,
            ],
        }
    }

    /// Volume of a single cell in cubic millimetres.
    pub fn cell_volume(&self) -> f64 {
        self.h * self.h * self.h
    }

    /// Flags every node that touches at least one cell carrying stiffness.
    ///
    /// Nodes that touch only void cells have an all-zero row in the global
    /// stiffness matrix; the caller pins them so the system stays solvable.
    pub fn active_nodes(&self) -> Vec<bool> {
        let mut active = vec![false; self.n_nodes()];
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    if self.cells[self.cell_index(i, j, k)] == CellKind::Void {
                        continue;
                    }
                    for n in self.element_nodes(i, j, k) {
                        active[n] = true;
                    }
                }
            }
        }
        active
    }

    /// Element indices grouped into the eight parity colours
    /// `(i % 2, j % 2, k % 2)`. Two elements in the same colour differ by at
    /// least two on some axis and therefore share no node, so scatter-adds
    /// inside one colour cannot race.
    pub fn color_of(i: usize, j: usize, k: usize) -> usize {
        (i & 1) | ((j & 1) << 1) | ((k & 1) << 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{CsgOp, Shape};

    fn unit_domain() -> Csg {
        Csg::new(vec![(
            CsgOp::Add,
            Shape::axis_aligned_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
        )])
    }

    #[test]
    fn grid_covers_bounds_exactly_when_divisible() {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 20.0, 5.0],
        };
        let g = Grid::from_bounds(&bounds, 2.5);
        assert_eq!((g.nx, g.ny, g.nz), (4, 8, 2));
        assert!((g.origin[0] - 0.0).abs() < 1e-12);
        assert_eq!(g.n_cells(), 64);
        assert_eq!(g.n_nodes(), 5 * 9 * 3);
    }

    /// The count the budget check reads has to be the count the grid would
    /// really lay out, and it has to keep meaning something for the grids that
    /// cannot be laid out at all - which is the only reason it exists.
    #[test]
    fn the_cell_count_agrees_with_the_grid_and_survives_a_runaway_request() {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 20.0, 5.0],
        };
        for h in [0.5, 1.0, 2.5, 3.0, 7.0] {
            let grid = Grid::from_bounds(&bounds, h);
            assert_eq!(
                cell_count_for(&bounds, h),
                grid.n_cells() as f64,
                "the count and the grid disagree at h = {h}"
            );
        }
        // An axis shorter than one cell still gets one, as the grid does.
        assert_eq!(axis_cells(0.1, 5.0), 1.0);
        assert_eq!(axis_cells(0.0, 5.0), 1.0);
        // And a voxel size small enough to overflow every integer in the crate
        // comes back as a number the caller can still compare and report.
        let huge = cell_count_for(&bounds, 1e-9);
        assert!(huge.is_finite() && huge > usize::MAX as f64 / 1e3, "{huge}");
        assert!(cell_count_for(&bounds, f64::MIN_POSITIVE).is_infinite());
        assert!(axis_cells(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn grid_pads_symmetrically_when_not_divisible() {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 10.0, 10.0],
        };
        let g = Grid::from_bounds(&bounds, 3.0);
        assert_eq!((g.nx, g.ny, g.nz), (4, 4, 4));
        // 4 cells of 3 mm cover 12 mm, so 1 mm of pad on each side.
        assert!((g.origin[0] + 1.0).abs() < 1e-12);
        let b = g.bounds();
        assert!(b.min[0] <= 0.0 && b.max[0] >= 10.0);
    }

    #[test]
    fn index_round_trips() {
        let g = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [3.0, 4.0, 5.0],
            },
            1.0,
        );
        for e in 0..g.n_cells() {
            let (i, j, k) = g.cell_ijk(e);
            assert_eq!(g.cell_index(i, j, k), e);
        }
        for n in 0..g.n_nodes() {
            let (i, j, k) = g.node_ijk(n);
            assert_eq!(g.node_index(i, j, k), n);
        }
    }

    #[test]
    fn element_nodes_follow_local_ordering() {
        let g = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [2.0, 2.0, 2.0],
            },
            1.0,
        );
        let nodes = g.element_nodes(0, 0, 0);
        assert_eq!(nodes[0], g.node_index(0, 0, 0));
        assert_eq!(nodes[1], g.node_index(1, 0, 0));
        assert_eq!(nodes[2], g.node_index(0, 1, 0));
        assert_eq!(nodes[4], g.node_index(0, 0, 1));
        assert_eq!(nodes[7], g.node_index(1, 1, 1));
    }

    #[test]
    fn colors_never_share_nodes() {
        let g = Grid::from_bounds(
            &Aabb {
                min: [0.0, 0.0, 0.0],
                max: [4.0, 4.0, 4.0],
            },
            1.0,
        );
        let mut per_color: Vec<Vec<usize>> = vec![Vec::new(); constants::ELEMENT_COLORS];
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    per_color[Grid::color_of(i, j, k)].extend(g.element_nodes(i, j, k));
                }
            }
        }
        for nodes in per_color {
            let mut sorted = nodes.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), nodes.len(), "a colour reused a node");
        }
    }

    #[test]
    fn keepout_beats_keepin_beats_domain() {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 10.0, 10.0],
        };
        let mut g = Grid::from_bounds(&bounds, 1.0);
        let keepin = ShapeUnion::new(vec![Shape::axis_aligned_box(
            [0.0, 0.0, 0.0],
            [4.0, 10.0, 10.0],
        )]);
        let keepout = ShapeUnion::new(vec![Shape::axis_aligned_box(
            [0.0, 0.0, 0.0],
            [2.0, 10.0, 10.0],
        )]);
        let overlaps = g.classify(&unit_domain(), &keepout, &keepin);
        assert_eq!(overlaps, 2 * 10 * 10);
        // x in [0,2): keepout wins -> void. x in [2,4): keepin -> solid.
        assert_eq!(g.cells[g.cell_index(0, 5, 5)], CellKind::Void);
        assert_eq!(g.cells[g.cell_index(1, 5, 5)], CellKind::Void);
        assert_eq!(g.cells[g.cell_index(2, 5, 5)], CellKind::Solid);
        assert_eq!(g.cells[g.cell_index(3, 5, 5)], CellKind::Solid);
        assert_eq!(g.cells[g.cell_index(5, 5, 5)], CellKind::Design);
    }

    /// An ellipsoid keepout carves exactly the cells whose centres are inside
    /// it, which for a 10 mm cube at 1 mm cells is a set that can be written
    /// out by hand from `(x/rx)^2 + (y/ry)^2 + (z/rz)^2 <= 1`.
    #[test]
    fn an_ellipsoid_keepout_carves_the_cells_its_own_field_covers() {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 10.0, 10.0],
        };
        let mut g = Grid::from_bounds(&bounds, 1.0);
        // Centred on the middle of the grid, 4 mm along x and 1.5 mm along the
        // other two: a long thin shape whose extent per axis is different, so a
        // radius used on the wrong axis shows.
        let keepout = ShapeUnion::new(vec![Shape::Ellipsoid {
            center: [5.0, 5.0, 5.0],
            radii: [4.0, 1.5, 1.5],
            rotation_deg: [0.0; 3],
        }]);
        g.classify(&unit_domain(), &keepout, &ShapeUnion::default());

        // Cell centres are at 0.5, 1.5, ... 9.5, so the middle row runs from
        // x = 1.5 (3.5 mm out, inside) to x = 8.5, and y and z reach one cell
        // either side of the centre (1.0 mm out of 1.5) and no further.
        let mut void = 0;
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let centre = [i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5];
                    let scaled = (centre[0] - 5.0) * (centre[0] - 5.0) / 16.0
                        + (centre[1] - 5.0) * (centre[1] - 5.0) / 2.25
                        + (centre[2] - 5.0) * (centre[2] - 5.0) / 2.25;
                    let expected = if scaled <= 1.0 {
                        void += 1;
                        CellKind::Void
                    } else {
                        CellKind::Design
                    };
                    assert_eq!(
                        g.cells[g.cell_index(i, j, k)],
                        expected,
                        "cell ({i}, {j}, {k}) at {centre:?}"
                    );
                }
            }
        }
        // Hand counted: the four rows of y, z in {4, 5} span x = 1 .. 8 (eight
        // cells each), and the eight cells with exactly one of y, z at 3 or 6
        // sit at (1 + 6.25/2.25) > 1 and are outside. So 4 * 8 = 32.
        assert_eq!(void, 32, "the carved set is not the hand counted one");
        assert_eq!(g.cells[g.cell_index(5, 5, 5)], CellKind::Void);
        assert_eq!(g.cells[g.cell_index(0, 5, 5)], CellKind::Design, "x = 0.5");
        assert_eq!(g.cells[g.cell_index(5, 3, 5)], CellKind::Design, "y = 3.5");

        // Turned a quarter turn about z, the long axis is y and the carved set
        // is the same one transposed.
        let mut turned_grid = Grid::from_bounds(&bounds, 1.0);
        let turned = ShapeUnion::new(vec![Shape::Ellipsoid {
            center: [5.0, 5.0, 5.0],
            radii: [4.0, 1.5, 1.5],
            rotation_deg: [0.0, 0.0, 90.0],
        }]);
        turned_grid.classify(&unit_domain(), &turned, &ShapeUnion::default());
        for k in 0..turned_grid.nz {
            for j in 0..turned_grid.ny {
                for i in 0..turned_grid.nx {
                    assert_eq!(
                        turned_grid.cells[turned_grid.cell_index(i, j, k)],
                        g.cells[g.cell_index(j, i, k)],
                        "cell ({i}, {j}, {k}) does not mirror the unturned one"
                    );
                }
            }
        }
    }

    #[test]
    fn active_nodes_exclude_fully_void_neighbourhoods() {
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 10.0, 10.0],
        };
        let mut g = Grid::from_bounds(&bounds, 1.0);
        let keepout = ShapeUnion::new(vec![Shape::axis_aligned_box(
            [0.0, 0.0, 0.0],
            [5.0, 10.0, 10.0],
        )]);
        g.classify(&unit_domain(), &keepout, &ShapeUnion::default());
        let active = g.active_nodes();
        assert!(!active[g.node_index(0, 5, 5)]);
        assert!(active[g.node_index(5, 5, 5)]);
        assert!(active[g.node_index(10, 5, 5)]);
    }
}
