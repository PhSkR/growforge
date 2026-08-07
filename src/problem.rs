//! Turns a validated [`Config`] into the discrete problem the engines consume:
//! a classified voxel grid, a constraint mask and one load vector per load case.

use std::path::Path;

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::config::{
    Axis, Config, GrowthParams, LoadSpec, Material, OptimizationParams, OutputParams, SolverParams,
};
use crate::constants;
use crate::geometry::{self, Boundaries, Shape, Vec3};
use crate::grid::{CellCounts, Grid};

/// One assembled load case.
#[derive(Debug, Clone)]
pub struct LoadCase {
    /// Name from the config, used in reports.
    pub name: String,
    /// Weight in the compliance objective.
    pub weight: f64,
    /// Design independent nodal force vector, one entry per degree of freedom,
    /// in newtons. Self-weight is *not* in here: it depends on the densities and
    /// is recomputed every iteration, see [`LoadCase::gravity`].
    pub forces: Vec<f64>,
    /// Net gravitational acceleration of this case in mm/s^2, when it carries
    /// one or more gravity loads. Several gravity loads add up.
    pub gravity: Option<Vec3>,
    /// Per-load node selection counts, for the problem summary.
    pub loads: Vec<LoadSummary>,
}

/// Node selection result for one load, reported by `growforge check`.
#[derive(Debug, Clone)]
pub struct LoadSummary {
    /// `force`, `torque` or `gravity`.
    pub kind: &'static str,
    /// Number of load-carrying nodes the load acts on.
    pub node_count: usize,
    /// Extra description shown after the node count; empty when there is none.
    pub detail: String,
    /// The load-carrying nodes the region selected, in ascending index order.
    ///
    /// Empty for a gravity load, which selects no region and acts on every
    /// element; `node_count` still reports how many nodes that is. Engines that
    /// need to know *where* a load enters the structure, rather than only the
    /// assembled force vector, read this: the growth engine routes a strut from
    /// every load region to the supports.
    pub nodes: Vec<usize>,
    /// The region shape itself, as the configuration wrote it; `None` for a
    /// gravity load, which has none. The nodes above are what it selected on the
    /// lattice, which is not the same question as "is this point inside it" -
    /// the export's island pass asks the second of a culled fragment.
    pub region: Option<crate::geometry::Shape>,
}

/// Node selection result for one support region.
#[derive(Debug, Clone)]
pub struct SupportSummary {
    /// One-based index of the `[[supports]]` entry.
    pub index: usize,
    /// Number of load-carrying nodes selected by the region.
    pub node_count: usize,
    /// Constrained axes.
    pub directions: Vec<Axis>,
    /// The load-carrying nodes the region selected, in ascending index order.
    ///
    /// The global `fixed` mask cannot be taken apart again into the regions that
    /// wrote it, and the growth engine grows one anchor per region, so the
    /// selection is kept.
    pub nodes: Vec<usize>,
    /// The region shape itself, as the configuration wrote it. See
    /// [`LoadSummary::region`].
    pub region: Option<crate::geometry::Shape>,
}

/// Cell selection result for one `[[keepin]]` entry.
///
/// The classifier merges every keepin entry into one union before it pins cells
/// solid, which is the right thing for classification - a cell is material or it
/// is not, whichever entry asked for it - but loses which entry asked. Anything
/// that has to treat the entries as separate *places* needs them apart: the guide
/// wireframe of [`crate::engine::wireframe`] runs a wire to each of them, and two
/// disjoint pads merged into one union would leave one of them unwired.
#[derive(Debug, Clone)]
pub struct KeepinSummary {
    /// One-based index of the `[[keepin]]` entry.
    pub index: usize,
    /// The forced solid cells this entry accounts for, in ascending index order.
    ///
    /// Empty when a keepout swallowed the entry whole - keepout wins, and the
    /// cells are void - or when it selected no cell centre at all.
    pub cells: Vec<usize>,
}

/// The discrete optimization problem.
#[derive(Debug, Clone)]
pub struct Problem {
    /// Project name.
    pub name: String,
    /// Engine key selected by the config.
    pub engine: String,
    /// Classified voxel grid.
    pub grid: Grid,
    /// Resolved material properties.
    pub material: Material,
    /// Resolved optimization controls.
    pub optimization: OptimizationParams,
    /// Resolved growth engine controls; `None` unless `engine = "growth"`.
    pub growth: Option<GrowthParams>,
    /// Resolved linear solver controls.
    pub solver: SolverParams,
    /// Resolved output controls.
    pub output: OutputParams,
    /// The analytic surfaces the classified grid is a voxelization of: the
    /// domain CSG tree and the keepout union, as the configuration wrote them.
    ///
    /// Kept past classification because the *export* needs them: a cell is
    /// classified by its centre and the extracted surface is then smoothed, so
    /// an exported vertex can sit a fraction of a voxel inside a keepout, and
    /// putting it back on the boundary needs the boundary itself
    /// (see [`crate::mesh::clamp`]). They ride on the problem rather than being
    /// threaded from the configuration at export time because the editor's
    /// generate-stl button exports a **retained** problem: boundaries taken from
    /// the configuration as it stands could describe geometry the retained
    /// problem was never built from, and carrying them here forecloses that
    /// mismatch entirely. The same reason the per-entry shapes ride along in
    /// [`SupportSummary::region`] and [`LoadSummary::region`].
    pub boundaries: Boundaries,
    /// Constrained degrees of freedom, one flag per DOF.
    pub fixed: Vec<bool>,
    /// Assembled load cases.
    pub load_cases: Vec<LoadCase>,
    /// Per-support-region node counts.
    pub supports: Vec<SupportSummary>,
    /// Per-keepin-entry cell selections, in configuration order.
    pub keepins: Vec<KeepinSummary>,
    /// Cell counts by kind.
    pub counts: CellCounts,
    /// Non-fatal problems found while building.
    pub warnings: Vec<String>,
}

/// Reject a grid that cannot be built before the attempt to build it takes the
/// process down.
///
/// Both resolution keys reach here: `voxel_size_mm` is used as given, and
/// `target_cells` has already been turned into a voxel size, so what is checked
/// is the grid either of them derives rather than the number one of them wrote.
/// The count is worked out in floating point, because the whole point is that
/// it may not fit in a `usize` - see [`constants::MAX_GRID_CELLS`].
fn check_grid_budget(config: &Config, bounds: &crate::geometry::Aabb, h: f64) -> Result<()> {
    let cells = crate::grid::cell_count_for(bounds, h);
    let budget = constants::MAX_GRID_CELLS;
    // Written as a negated comparison so that a count that is not a number -
    // a degenerate extent or voxel size - is rejected rather than accepted.
    if cells <= budget as f64 {
        return Ok(());
    }
    let extent = bounds.extent();
    let axes = [
        crate::grid::axis_cells(extent[0], h),
        crate::grid::axis_cells(extent[1], h),
        crate::grid::axis_cells(extent[2], h),
    ];
    let (asked, remedy) = match (
        config.resolution.voxel_size_mm,
        config.resolution.target_cells,
    ) {
        (Some(size), _) => (format!("voxel_size_mm = {size}"), "raise voxel_size_mm"),
        (None, Some(target)) => (
            format!("target_cells = {target}, which derives a voxel size of {h:.6} mm,"),
            "lower target_cells",
        ),
        // `Config::voxel_size` has already rejected having neither.
        (None, None) => ("this resolution".to_string(), "coarsen the resolution"),
    };
    bail!(
        "[resolution]: {asked} lays out {:.0} x {:.0} x {:.0} = {cells:.3e} cells over the \
         {:.3} x {:.3} x {:.3} mm domain, above the budget of {budget} cells; {remedy}, or \
         shrink the domain",
        axes[0],
        axes[1],
        axes[2],
        extent[0],
        extent[1],
        extent[2]
    )
}

/// Collect the forced solid cells whose centre lies inside `shape`.
///
/// The classifier pins a cell solid by its centre point (see
/// [`Grid::classify`]), so this asks the same question of a single keepin entry
/// that the classification asks of their union: which cells are solid *because of
/// this entry*. A cell the entry covers but a keepout swallowed is not among them
/// - keepout wins and the cell is void.
fn select_solid_cells(grid: &Grid, shape: &Shape) -> Vec<usize> {
    (0..grid.n_cells())
        .into_par_iter()
        .filter(|&e| {
            if grid.cells[e] != crate::grid::CellKind::Solid {
                return false;
            }
            let (i, j, k) = grid.cell_ijk(e);
            shape.contains(grid.cell_center(i, j, k))
        })
        .collect()
}

/// Collect the load-carrying nodes whose position lies inside `shape`.
fn select_nodes(grid: &Grid, active: &[bool], shape: &Shape) -> Vec<usize> {
    (0..grid.n_nodes())
        .into_par_iter()
        .filter(|&n| {
            if !active[n] {
                return false;
            }
            let (i, j, k) = grid.node_ijk(n);
            shape.contains(grid.node_pos(i, j, k))
        })
        .collect()
}

impl Problem {
    /// Build the discrete problem. `config_dir` anchors relative output paths.
    pub fn build(config: &Config, config_dir: &Path) -> Result<Problem> {
        config.validate_static()?;

        let engine = config.engine_name()?.to_string();
        crate::engine::ensure_registered(&engine)?;
        let material = config.material()?;
        let optimization = config.optimization_params()?;
        let solver = config.solver_params()?;
        let growth = config.growth_params()?;
        let output = config.output_params(config_dir)?;
        let domain = config.domain_csg()?;
        let keepout = config.keepout_union()?;
        let keepin = config.keepin_union()?;

        // Keepin regions take precedence over the domain, so they are part of
        // the modelled solid even where they stick out of it and must be inside
        // the grid.
        let mut bounds = domain.bounds();
        if !keepin.is_empty() {
            bounds = bounds.union(&keepin.bounds());
        }
        if bounds.is_empty() || bounds.volume() <= 0.0 || bounds.volume().is_nan() {
            bail!("the domain bounding box is empty or degenerate");
        }

        let h = config.voxel_size(bounds.volume())?;
        // Before the grid is laid out, because laying it out allocates it.
        check_grid_budget(config, &bounds, h)?;
        let mut grid = Grid::from_bounds(&bounds, h);
        let overlap_cells = grid.classify(&domain, &keepout, &keepin);
        let counts = grid.counts();

        // The ones the configuration alone can raise, first, so they read in the
        // order the file does before anything the grid found is added.
        let mut warnings = config.static_warnings();
        if overlap_cells > 0 {
            warnings.push(format!(
                "{overlap_cells} cells lie inside both a keepout and a keepin region; keepout wins and they are void"
            ));
        }
        // The voxel size is only known here, so the one growth control that is
        // measured against it is checked here rather than in the config.
        if let Some(growth) = growth
            && growth.step_mm < constants::GROWTH_MIN_STEP_VOXELS * h
        {
            bail!(
                "[growth]: step_mm = {:.4} mm is only {:.3} voxels at a voxel size of {h:.4} mm; a \
                 step shorter than {} voxels cannot leave the cell it started in",
                growth.step_mm,
                growth.step_mm / h,
                constants::GROWTH_MIN_STEP_VOXELS
            );
        }
        // Supersampling refines the export lattice by its cube, so the factor
        // that is harmless on a coarse grid is what runs a fine one out of
        // memory. The check belongs here because only the voxelization knows
        // how large the lattice really is.
        let export_nodes = crate::mesh::supersampled_node_count(&grid, output.supersample);
        if export_nodes > constants::SUPERSAMPLE_NODE_BUDGET_WARN {
            warnings.push(format!(
                "[output] supersample = {} refines the export lattice to {export_nodes} samples \
                 (about {:.0} MiB), above the {} sample budget; the export may exhaust memory, so \
                 lower supersample or use a coarser voxel_size_mm",
                output.supersample,
                (export_nodes as f64) * constants::SUPERSAMPLE_BYTES_PER_NODE as f64
                    / constants::BYTES_PER_MIB,
                constants::SUPERSAMPLE_NODE_BUDGET_WARN
            ));
        }
        // The local volume kernel is a second density filter, and its cost per
        // iteration is the cell count times its stencil. How many taps that is
        // is only known here, where the voxel size is.
        if let Some(cap) = optimization.local_volume {
            let reach = cap.radius_mm / h;
            let taps = (4.0 / 3.0) * std::f64::consts::PI * reach * reach * reach;
            if taps > constants::LOCAL_VOLUME_STENCIL_TAPS_WARN as f64 {
                warnings.push(format!(
                    "[optimization.local_volume] radius_mm = {:.3} mm is {reach:.1} voxels at a \
                     voxel size of {h:.3} mm, so its kernel is about {taps:.0} taps per cell - \
                     above the {} the run is budgeted for. Every iteration pays that twice, and \
                     it will dominate the analysis; lower radius_mm, or use a coarser \
                     voxel_size_mm",
                    cap.radius_mm,
                    constants::LOCAL_VOLUME_STENCIL_TAPS_WARN
                ));
            }
        }
        let min_feature_voxels = optimization.min_feature_mm / h;
        if min_feature_voxels < constants::MIN_FEATURE_VOXELS_WARN {
            warnings.push(format!(
                "min_feature_mm = {:.3} mm is only {:.2} voxels at a voxel size of {:.3} mm; the density filter cannot resolve features below about {:.1} voxels",
                optimization.min_feature_mm,
                min_feature_voxels,
                h,
                constants::MIN_FEATURE_VOXELS_WARN
            ));
        }

        if counts.active() == 0 {
            bail!(
                "the domain is empty after voxelization: no cell centre of the {}x{}x{} grid falls inside the domain (try a smaller voxel_size_mm)",
                grid.nx,
                grid.ny,
                grid.nz
            );
        }
        if counts.design == 0 {
            bail!(
                "the domain contains no design cells after voxelization: every cell is void or forced solid, so there is nothing to optimize"
            );
        }

        let active = grid.active_nodes();
        let mut fixed = vec![false; grid.n_dof()];
        // Nodes that touch no load-carrying cell have an empty row in the
        // stiffness matrix; pin them so the operator stays positive definite.
        for (n, is_active) in active.iter().enumerate() {
            if !is_active {
                for d in 0..constants::DOF_PER_NODE {
                    fixed[n * constants::DOF_PER_NODE + d] = true;
                }
            }
        }

        let mut supports = Vec::with_capacity(config.supports.len());
        for (idx, spec) in config.supports.iter().enumerate() {
            let what = format!("[[supports]] entry {}", idx + 1);
            let shape = spec.region.to_shape(&what)?;
            let nodes = select_nodes(&grid, &active, &shape);
            if nodes.is_empty() {
                bail!(
                    "{what}: the region selects no node that touches material; move or enlarge it (nodes sit on the grid lattice with a spacing of {h:.3} mm)"
                );
            }
            let directions = spec
                .directions
                .clone()
                .unwrap_or_else(|| vec![Axis::X, Axis::Y, Axis::Z]);
            for n in &nodes {
                for axis in &directions {
                    fixed[n * constants::DOF_PER_NODE + axis.dof()] = true;
                }
            }
            supports.push(SupportSummary {
                index: idx + 1,
                node_count: nodes.len(),
                directions,
                nodes,
                region: Some(shape),
            });
        }

        // One entry at a time, next to the merged union the classifier already
        // applied: which entry a solid cell came from is a question only the
        // shapes can answer, and the grid no longer holds it.
        let mut keepins = Vec::with_capacity(config.keepin.len());
        for (idx, spec) in config.keepin.iter().enumerate() {
            let shape = spec.to_shape(&format!("[[keepin]] entry {}", idx + 1))?;
            keepins.push(KeepinSummary {
                index: idx + 1,
                cells: select_solid_cells(&grid, &shape),
            });
        }

        let active_node_count = active.iter().filter(|a| **a).count();
        let mut load_cases = Vec::with_capacity(config.loadcases.len());
        for (ci, case) in config.loadcases.iter().enumerate() {
            let what = format!("[[loadcases]] \"{}\" (entry {})", case.name, ci + 1);
            let mut forces = vec![0.0; grid.n_dof()];
            let mut gravity: Option<Vec3> = None;
            // Largest single gravity contribution, which is what the sum below
            // has to be judged degenerate against.
            let mut gravity_scale = 0.0f64;
            let mut summaries = Vec::with_capacity(case.loads.len());
            for (li, load) in case.loads.iter().enumerate() {
                let lwhat = format!("{what}, load {}", li + 1);
                if let Some(acceleration) = load.gravity_acceleration(&lwhat)? {
                    // Self-weight scales with the densities, so nothing can be
                    // assembled yet; only the acceleration is resolved here.
                    let total = gravity.unwrap_or([0.0; 3]);
                    gravity = Some(geometry::sum(total, acceleration));
                    gravity_scale = gravity_scale.max(geometry::length(acceleration));
                    summaries.push(LoadSummary {
                        kind: load.kind(),
                        node_count: active_node_count,
                        detail: format!(
                            "{:.1} mm/s2 along [{:.3}, {:.3}, {:.3}]",
                            geometry::length(acceleration),
                            acceleration[0] / geometry::length(acceleration),
                            acceleration[1] / geometry::length(acceleration),
                            acceleration[2] / geometry::length(acceleration)
                        ),
                        nodes: Vec::new(),
                        region: None,
                    });
                    continue;
                }
                let shape = load
                    .region()
                    .expect("only gravity loads have no region")
                    .to_shape(&lwhat)?;
                let nodes = select_nodes(&grid, &active, &shape);
                if nodes.is_empty() {
                    bail!(
                        "{lwhat}: the region selects no node that touches material; move or enlarge it (nodes sit on the grid lattice with a spacing of {h:.3} mm)"
                    );
                }
                apply_load(&grid, load, &nodes, &mut forces, &lwhat)?;
                summaries.push(LoadSummary {
                    kind: load.kind(),
                    node_count: nodes.len(),
                    detail: String::new(),
                    nodes,
                    region: Some(shape),
                });
            }

            // Each gravity load is separately non-zero, but two of them can
            // still oppose and leave a case claiming a self weight it does not
            // carry. Everything downstream then divides by that magnitude.
            if let Some(total) = gravity {
                let net = geometry::length(total);
                if net.is_nan() || net <= constants::MIN_NET_GRAVITY_FRACTION * gravity_scale {
                    bail!(
                        "{what}: its gravity loads cancel out (net acceleration {net:.3e} mm/s2 \
                         against a largest single contribution of {gravity_scale:.3e} mm/s2); \
                         drop them, or give them directions that do not oppose"
                    );
                }
            }

            let free_magnitude: f64 = forces
                .iter()
                .zip(fixed.iter())
                .map(|(f, &is_fixed)| if is_fixed { 0.0 } else { f.abs() })
                .fold(0.0, f64::max);
            let gravity_can_move = gravity.is_some_and(|a| {
                (0..grid.n_nodes()).any(|n| {
                    active[n]
                        && (0..constants::DOF_PER_NODE)
                            .any(|d| a[d] != 0.0 && !fixed[n * constants::DOF_PER_NODE + d])
                })
            });
            if free_magnitude == 0.0 && !gravity_can_move {
                bail!(
                    "{what}: every loaded degree of freedom is constrained by a support, so the load case cannot deform the structure"
                );
            }

            load_cases.push(LoadCase {
                name: case.name.clone(),
                weight: case.weight.unwrap_or(constants::DEFAULT_LOADCASE_WEIGHT),
                forces,
                gravity,
                loads: summaries,
            });
        }

        Ok(Problem {
            name: config.project.name.clone(),
            engine,
            grid,
            material,
            optimization,
            growth,
            solver,
            output,
            boundaries: Boundaries { domain, keepout },
            fixed,
            load_cases,
            supports,
            keepins,
            counts,
            warnings,
        })
    }

    /// Every region of this problem that declares a purpose for the material in
    /// it: the support regions, the load regions and the keepin cells.
    ///
    /// This is what the export's island pass keeps geometry for. The regions are
    /// built from the node selections the problem already made rather than from
    /// the shapes again, so what counts as "inside a support region" is one
    /// answer throughout the run: the same nodes the solver constrained. Each one
    /// also carries a few probe points, taken from the cells of the region that
    /// hold material in `densities`, for the components whose surface runs
    /// nowhere near those nodes.
    ///
    /// A gravity load selects no region - it acts on every element - so it
    /// contributes no anchor and is never reported as unreached.
    pub fn anchors(&self, densities: &[f64]) -> crate::mesh::islands::Anchors {
        use crate::mesh::islands::{AnchorKind, AnchorRegion};
        let region = |kind, label: String, nodes: Vec<usize>, shape: Option<Shape>| AnchorRegion {
            probes: self.material_probes(&nodes, densities),
            kind,
            label,
            nodes,
            shape,
        };
        let mut regions = Vec::new();
        for support in &self.supports {
            regions.push(region(
                AnchorKind::Support,
                format!("support {}", support.index),
                support.nodes.clone(),
                support.region,
            ));
        }
        for case in &self.load_cases {
            for (index, load) in case.loads.iter().enumerate() {
                if load.nodes.is_empty() {
                    continue;
                }
                regions.push(region(
                    AnchorKind::Load,
                    format!("load {} of case \"{}\"", index + 1, case.name),
                    load.nodes.clone(),
                    load.region,
                ));
            }
        }
        // Keepin cells are pinned solid by the classifier, so the grid is where
        // they are, shapes and all.
        let keepin: Vec<usize> = (0..self.grid.n_cells())
            .filter(|&cell| self.grid.cells[cell] == crate::grid::CellKind::Solid)
            .flat_map(|cell| {
                let (i, j, k) = self.grid.cell_ijk(cell);
                (0..8).map(move |corner| {
                    self.grid.node_index(
                        i + (corner & 1),
                        j + ((corner >> 1) & 1),
                        k + ((corner >> 2) & 1),
                    )
                })
            })
            .collect();
        if !keepin.is_empty() {
            // The keepin union is many shapes and the cells already carry it, so
            // this region is the cells alone: the shape below is what a *culled*
            // fragment is tested against, and a keepin cell is solid by
            // construction - no fragment is made of one that the mesh does not
            // hold as material.
            regions.push(region(
                AnchorKind::Keepin,
                "keepin".to_string(),
                keepin,
                None,
            ));
        }
        crate::mesh::islands::Anchors::new(&self.grid, regions)
    }

    /// Centres of up to [`constants::ISLAND_ANCHOR_PROBES_PER_REGION`] cells
    /// that touch `nodes` and hold material at the export iso level.
    ///
    /// Spread over the region in **space**, one probe per occupied box of a
    /// coarse partition of its own bounding box. Every axis with more than one
    /// cell of extent is split before any axis is split twice, so no structure
    /// along one axis can take the whole budget; what is left of it then goes to
    /// the axis with the widest boxes.
    ///
    /// Both halves of that are defects this walked into. Taking every n-th cell
    /// of the list is aliasing: the list is in raster order, so a pad split left
    /// and right by a gap, 21 cells to a row and 168 of them, makes the stride an
    /// exact multiple of the row and puts all eight probes at the same offset in
    /// every row - one patch, never the other. Growing the partition widest-first
    /// alone is starvation: a pad 100 cells wide and 12 tall spends the whole
    /// budget on x and probes a single row of y.
    ///
    /// What this does **not** promise is that every gap is straddled. A gap
    /// narrower than one box can sit inside one, and no finite partition can
    /// promise otherwise; a region split by such a gap may be probed on one side
    /// only, and the component holding the other side can then be culled. That
    /// case is *reported* rather than prevented, and by a check that does not
    /// depend on this sampling at all: every fragment that leaves is tested
    /// against the region shapes themselves
    /// ([`crate::mesh::islands::IslandReport::culled_inside`]), so material a
    /// declared region asked for cannot be removed silently even when the rest
    /// of that region is still served by what ships and the unserved-region
    /// warning therefore stays quiet.
    ///
    /// A region whose material was optimized away has no probe at all, which is
    /// the honest answer: there is nothing there to be part of anything.
    fn material_probes(&self, nodes: &[usize], densities: &[f64]) -> Vec<crate::geometry::Vec3> {
        let grid = &self.grid;
        let mut seen = vec![false; grid.n_cells()];
        let mut cells = Vec::new();
        for &node in nodes {
            let (i, j, k) = grid.node_ijk(node);
            for corner in 0..8 {
                let (ci, cj, ck) = (
                    i as isize - (corner & 1),
                    j as isize - ((corner >> 1) & 1),
                    k as isize - ((corner >> 2) & 1),
                );
                if ci < 0 || cj < 0 || ck < 0 {
                    continue;
                }
                let (ci, cj, ck) = (ci as usize, cj as usize, ck as usize);
                if ci >= grid.nx || cj >= grid.ny || ck >= grid.nz {
                    continue;
                }
                let cell = grid.cell_index(ci, cj, ck);
                if seen[cell] {
                    continue;
                }
                seen[cell] = true;
                if crate::engine::cell_density(grid, densities, cell) >= self.output.iso_level {
                    cells.push(cell);
                }
            }
        }

        let wanted = constants::ISLAND_ANCHOR_PROBES_PER_REGION;
        let centre = |cell: usize| {
            let (i, j, k) = grid.cell_ijk(cell);
            grid.cell_center(i, j, k)
        };
        if cells.len() <= wanted {
            return cells.into_iter().map(centre).collect();
        }

        // The region's own extent in cells, and a partition of it into at most
        // `wanted` boxes.
        let ijk = |cell: usize| {
            let (i, j, k) = grid.cell_ijk(cell);
            [i, j, k]
        };
        let mut low = ijk(cells[0]);
        let mut high = low;
        for &cell in &cells {
            let at = ijk(cell);
            for d in 0..3 {
                low[d] = low[d].min(at[d]);
                high[d] = high[d].max(at[d]);
            }
        }
        let extent = [0, 1, 2].map(|d| high[d] - low[d] + 1);
        let mut divisions = [1usize; 3];
        let boxes = |divisions: [usize; 3]| divisions[0] * divisions[1] * divisions[2];

        // Every axis that has an extent at all is split once first, widest axis
        // first so that the axes a small budget cannot reach are the narrow
        // ones. Without this the greedy growth below starves an axis: a pad 100
        // cells wide and 12 tall spends all eight boxes on x and probes one row.
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| extent[b].cmp(&extent[a]).then(a.cmp(&b)));
        for d in order {
            if extent[d] > 1 && boxes(divisions) * 2 <= wanted {
                divisions[d] = 2;
            }
        }

        // Then the remaining budget goes to the axis with the widest boxes, so a
        // long thin region is cut along its length and a cube into octants.
        loop {
            let mut widest: Option<usize> = None;
            for d in 0..3 {
                let mut candidate = divisions;
                candidate[d] += 1;
                if boxes(candidate) > wanted {
                    continue;
                }
                // Widths compared as extent / divisions, cross multiplied so the
                // comparison is exact; ties go to the earlier axis.
                if widest.is_none_or(|b| extent[d] * divisions[b] > extent[b] * divisions[d]) {
                    widest = Some(d);
                }
            }
            match widest {
                Some(d) => divisions[d] += 1,
                None => break,
            }
        }

        // One probe per occupied box, the first cell of it in scan order.
        let mut probe_of_box = vec![None; boxes(divisions)];
        for &cell in &cells {
            let at = ijk(cell);
            let bin = [0, 1, 2]
                .map(|d| ((at[d] - low[d]) * divisions[d] / extent[d]).min(divisions[d] - 1));
            let index = bin[0] + divisions[0] * (bin[1] + divisions[1] * bin[2]);
            probe_of_box[index].get_or_insert(cell);
        }
        probe_of_box
            .into_iter()
            .flatten()
            .take(wanted)
            .map(centre)
            .collect()
    }

    /// Rough peak working set of the optimization, in bytes.
    ///
    /// The moving asymptotes update carries state the optimality criteria step
    /// does not - its asymptotes and the two previous design points - and a
    /// local volume cap carries a second kernel and a second gradient, so the
    /// count of live cell arrays depends on what the configuration asked for.
    pub fn estimated_memory_bytes(&self) -> usize {
        let cells = self.grid.n_cells();
        let dof = self.grid.n_dof();
        let arrays = constants::ESTIMATED_CELL_ARRAYS
            + match self.optimization.update {
                crate::config::UpdateScheme::Oc => 0,
                crate::config::UpdateScheme::Mma => constants::ESTIMATED_MMA_CELL_ARRAYS,
            }
            + match self.optimization.local_volume {
                None => 0,
                Some(_) => constants::ESTIMATED_LOCAL_VOLUME_CELL_ARRAYS,
            };
        let cell_arrays = arrays * cells * constants::BYTES_PER_F64;
        let solver_vectors = (constants::ESTIMATED_SOLVER_VECTORS
            + constants::ESTIMATED_VECTORS_PER_LOADCASE * self.load_cases.len())
            * dof
            * constants::BYTES_PER_F64;
        let classification = cells * size_of::<crate::grid::CellKind>();
        cell_arrays + solver_vectors + classification
    }

    /// Volume of one voxel in cubic millimetres.
    pub fn cell_volume_mm3(&self) -> f64 {
        self.grid.cell_volume()
    }

    /// True when the solid engine is the one that will run this problem.
    ///
    /// The engine identity rather than anything about the field, which is what
    /// every report that has to describe a solid run keys on: a field of all
    /// ones is a legitimate answer from any engine, and "no iterations" is not a
    /// property a summary may infer an engine from. A growth run is recognised
    /// the same way, through [`Problem::growth`] being resolved.
    pub fn is_solid(&self) -> bool {
        self.engine == constants::SOLID_ENGINE
    }

    /// True when any load case carries self weight.
    pub fn has_gravity(&self) -> bool {
        self.load_cases.iter().any(|c| c.gravity.is_some())
    }

    /// Assemble the nodal force vector of one load case at the given physical
    /// densities, self weight included.
    ///
    /// Self weight is design dependent, so this has to be called again whenever
    /// the densities move. Everything that solves a load case goes through here,
    /// which is what keeps the optimizer and the post-run stress analysis on the
    /// same right hand side.
    pub fn case_forces(&self, case: usize, densities: &[f64], forces: &mut [f64]) {
        let case = &self.load_cases[case];
        forces.copy_from_slice(&case.forces);
        if let Some(acceleration) = case.gravity {
            crate::fea::gravity::accumulate(
                &self.grid,
                densities,
                acceleration,
                self.material.density_g_cm3,
                forces,
            );
        }
    }
}

/// Scatter one design independent load onto the nodal force vector.
///
/// Gravity never reaches this function: its magnitude follows the physical
/// densities, so it is assembled per iteration by [`crate::fea::gravity`].
fn apply_load(
    grid: &Grid,
    load: &LoadSpec,
    nodes: &[usize],
    forces: &mut [f64],
    what: &str,
) -> Result<()> {
    match load {
        LoadSpec::Gravity { .. } => unreachable!("gravity is assembled from the densities"),
        LoadSpec::Force { vector, .. } => {
            let share = 1.0 / nodes.len() as f64;
            for &n in nodes {
                let base = n * constants::DOF_PER_NODE;
                for d in 0..constants::DOF_PER_NODE {
                    forces[base + d] += vector[d] * share;
                }
            }
        }
        LoadSpec::Torque {
            axis_point,
            axis_dir,
            magnitude_nmm,
            ..
        } => {
            let len = geometry::length(*axis_dir);
            let axis = geometry::scale(*axis_dir, 1.0 / len);
            // Perpendicular offset of every selected node from the axis.
            let offsets: Vec<Vec3> = nodes
                .iter()
                .map(|&n| {
                    let (i, j, k) = grid.node_ijk(n);
                    let v = geometry::difference(grid.node_pos(i, j, k), *axis_point);
                    let along = geometry::inner(v, axis);
                    geometry::difference(v, geometry::scale(axis, along))
                })
                .collect();
            let sum_sq: f64 = offsets.iter().map(|r| geometry::inner(*r, *r)).sum();
            if sum_sq == 0.0 {
                bail!(
                    "{what}: every selected node lies on the torque axis, so no tangential force can be built"
                );
            }
            // f_i = c * (axis x r_i) has magnitude c |r_i|, is tangential, and
            // sums to zero net force while delivering the requested torque.
            let c = magnitude_nmm / sum_sq;
            for (&n, r) in nodes.iter().zip(offsets.iter()) {
                let f = geometry::scale(geometry::cross(axis, *r), c);
                let base = n * constants::DOF_PER_NODE;
                for d in 0..constants::DOF_PER_NODE {
                    forces[base + d] += f[d];
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn cantilever_config(extra: &str) -> String {
        format!(
            r#"
[project]
name = "t"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0

[output]
stl_path = "t.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [40.0, 16.0, 16.0]

[[supports]]
region = {{ shape = "box", min = [0.0, 0.0, 0.0], max = [0.5, 16.0, 16.0] }}

[[loadcases]]
name = "main"
[[loadcases.loads]]
type = "force"
region = {{ shape = "sphere", center = [40.0, 8.0, 8.0], radius = 3.0 }}
vector = [0.0, 0.0, -50.0]
{extra}
"#
        )
    }

    fn build(text: &str) -> Result<Problem> {
        let cfg = Config::parse(text)?;
        Problem::build(&cfg, &PathBuf::from("."))
    }

    /// A slab exactly as wide as the reviewer's aliasing case: a flat pad of
    /// eight rows, each row eleven material cells, a one-cell gap, then ten
    /// more. 168 material cells in 21-cell rows, which is what makes
    /// `168 / 8 = 21` an exact multiple of the row and put every probe of the
    /// old index stride at the same offset in every row - all eight in the left
    /// patch, none in the right.
    fn split_pad() -> (Problem, Vec<f64>, Vec<usize>, f64) {
        let text = r#"
[project]
name = "pad"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0

[output]
stl_path = "pad.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [44.0, 16.0, 4.0]

[[supports]]
region = { shape = "box", min = [0.0, 0.0, 0.0], max = [0.5, 16.0, 4.0] }

[[loadcases]]
name = "main"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [44.0, 8.0, 2.0], radius = 3.0 }
vector = [0.0, 0.0, -50.0]
"#;
        let problem = build(text).expect("build");
        assert_eq!(
            (problem.grid.nx, problem.grid.ny, problem.grid.nz),
            (22, 8, 2)
        );

        let mut densities = vec![0.0; problem.grid.n_cells()];
        let gap = 11;
        for j in 0..8 {
            for i in 0..22 {
                if i != gap {
                    densities[problem.grid.cell_index(i, j, 0)] = 1.0;
                }
            }
        }
        assert_eq!(
            densities.iter().filter(|d| **d > 0.5).count(),
            168,
            "the fixture must be the reviewer's 168 material cells"
        );

        // The region covers the whole slab: the material cells of the pad are
        // the candidates, in raster order.
        let mut nodes = Vec::new();
        for k in 0..problem.grid.nnz() {
            for j in 0..problem.grid.nny() {
                for i in 0..problem.grid.nnx() {
                    nodes.push(problem.grid.node_index(i, j, k));
                }
            }
        }
        let gap_x = problem.grid.origin[0] + (gap as f64 + 0.5) * problem.grid.h;
        (problem, densities, nodes, gap_x)
    }

    #[test]
    fn probes_are_spread_through_a_region_that_a_gap_splits_in_two() {
        let (problem, densities, nodes, gap_x) = split_pad();
        let probes = problem.material_probes(&nodes, &densities);
        assert_eq!(probes.len(), constants::ISLAND_ANCHOR_PROBES_PER_REGION);

        let left = probes.iter().filter(|p| p[0] < gap_x).count();
        let right = probes.iter().filter(|p| p[0] > gap_x).count();
        assert!(
            left > 0 && right > 0,
            "every probe landed on one side of the gap: {probes:?}"
        );
        assert_eq!(left + right, probes.len(), "a probe landed in the gap");
        // Both axes of the pad are covered, not just its length.
        let rows: std::collections::BTreeSet<i64> =
            probes.iter().map(|p| (p[1] * 1e6) as i64).collect();
        assert!(rows.len() > 1, "every probe landed in the same row");

        // Every probe is a cell centre holding material, and the answer is a
        // pure function of the field.
        let grid = &problem.grid;
        for probe in &probes {
            let at = [0, 1, 2].map(|d| ((probe[d] - grid.origin[d]) / grid.h - 0.5).round());
            assert!(
                at.iter().all(|c| *c >= 0.0),
                "probe {probe:?} is outside the grid"
            );
            let cell = grid.cell_index(at[0] as usize, at[1] as usize, at[2] as usize);
            assert!(densities[cell] > 0.5, "probe {probe:?} is not on material");
        }
        assert_eq!(probes, problem.material_probes(&nodes, &densities));
    }

    #[test]
    fn a_region_with_no_material_left_in_it_has_no_probes() {
        let (problem, _, nodes, _) = split_pad();
        let empty = vec![0.0; problem.grid.n_cells()];
        assert!(problem.material_probes(&nodes, &empty).is_empty());
    }

    /// The other way a partition can miss a patch: one axis so much longer than
    /// the others that a widest-first split spends every box on it. A hundred
    /// cells of x against twelve of y, with the y gap in the middle: growing the
    /// partition greedily gives eight columns and one row, and every probe lands
    /// in the low patch.
    #[test]
    fn probes_are_spread_across_the_short_axis_of_a_very_long_region() {
        let text = r#"
[project]
name = "long"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0

[output]
stl_path = "long.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [200.0, 24.0, 2.0]

[[supports]]
region = { shape = "box", min = [0.0, 0.0, 0.0], max = [0.5, 24.0, 2.0] }

[[loadcases]]
name = "main"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [200.0, 12.0, 1.0], radius = 3.0 }
vector = [0.0, 0.0, -50.0]
"#;
        let problem = build(text).expect("build");
        let grid = &problem.grid;
        assert_eq!((grid.nx, grid.ny, grid.nz), (100, 12, 1));

        // Material below the gap and above it, the gap four cells deep.
        let mut densities = vec![0.0; grid.n_cells()];
        for j in (0..4).chain(8..12) {
            for i in 0..100 {
                densities[grid.cell_index(i, j, 0)] = 1.0;
            }
        }
        let mut nodes = Vec::new();
        for k in 0..grid.nnz() {
            for j in 0..grid.nny() {
                for i in 0..grid.nnx() {
                    nodes.push(grid.node_index(i, j, k));
                }
            }
        }

        let probes = problem.material_probes(&nodes, &densities);
        assert_eq!(probes.len(), constants::ISLAND_ANCHOR_PROBES_PER_REGION);
        let low = probes.iter().filter(|p| p[1] < 8.0).count();
        let high = probes.iter().filter(|p| p[1] > 16.0).count();
        assert!(
            low > 0 && high > 0,
            "the long axis took every probe: {probes:?}"
        );
        assert_eq!(low + high, probes.len(), "a probe landed in the gap");
        // The length is still covered: starving x for y would be the same
        // defect the other way round.
        let columns: std::collections::BTreeSet<i64> =
            probes.iter().map(|p| (p[0] * 1e6) as i64).collect();
        assert!(columns.len() > 1, "every probe landed in the same column");
        assert_eq!(probes, problem.material_probes(&nodes, &densities));
    }

    #[test]
    fn cantilever_problem_builds() {
        let p = build(&cantilever_config("")).expect("build");
        assert_eq!((p.grid.nx, p.grid.ny, p.grid.nz), (20, 8, 8));
        assert_eq!(p.counts.design, p.grid.n_cells());
        assert_eq!(p.supports.len(), 1);
        // The support plane holds 9 x 9 nodes.
        assert_eq!(p.supports[0].node_count, 81);
        assert_eq!(p.load_cases.len(), 1);
        let total: f64 = p.load_cases[0].forces.iter().skip(2).step_by(3).sum();
        assert!((total + 50.0).abs() < 1e-9, "total z force is {total}");
    }

    /// Node selection reads the region's signed distance function, so a turned
    /// support box selects the nodes it really covers - and this is the set
    /// worked out by hand on a grid small enough to write down.
    ///
    /// The grid has 2 mm nodes at x = 0, 2, ..., 40 and y, z = 0, 2, ..., 16.
    /// The region is a 4 mm cube about the node at (8, 8, 8), turned 45 degrees
    /// about z: a square of half diagonal 2 sqrt(2) = 2.83 mm in the x-y plane,
    /// so it reaches the four nodes 2 mm away along x and y, and no further -
    /// the diagonal neighbours at (10, 10) are 2.83 mm out, exactly on the
    /// corner, and the region's own faces cut them off.
    #[test]
    fn a_turned_support_box_selects_the_nodes_it_covers() {
        let region = "region = { shape = \"box\", min = [6.0, 6.0, 6.0], max = [10.0, 10.0, 10.0]";
        let text = cantilever_config("").replace(
            "region = { shape = \"box\", min = [0.0, 0.0, 0.0], max = [0.5, 16.0, 16.0] }",
            &format!("{region}, rotation_deg = [0.0, 0.0, 45.0] }}"),
        );
        let problem = build(&text).expect("build");
        let grid = &problem.grid;
        let selected: std::collections::HashSet<usize> =
            problem.supports[0].nodes.iter().copied().collect();

        // The five nodes of the turned square's own cross, at each of the three
        // z planes the 4 mm tall region spans.
        let mut expected = std::collections::HashSet::new();
        for k in [3usize, 4, 5] {
            for (i, j) in [(4usize, 4usize), (3, 4), (5, 4), (4, 3), (4, 5)] {
                expected.insert(grid.node_index(i, j, k));
            }
        }
        assert_eq!(
            selected, expected,
            "the turned region selected a different node set"
        );

        // The same box left axis aligned covers its corners as well, so the
        // turn really is what changed the answer.
        let square = cantilever_config("").replace(
            "region = { shape = \"box\", min = [0.0, 0.0, 0.0], max = [0.5, 16.0, 16.0] }",
            &format!("{region} }}"),
        );
        let plain = build(&square).expect("build");
        assert_eq!(plain.supports[0].node_count, 3 * 3 * 3);
        assert!(problem.supports[0].node_count < plain.supports[0].node_count);
    }

    /// Region selection is a node-by-node question of the shape's own field, so
    /// an ellipsoid region has to select the nodes inside the ellipsoid rather
    /// than the ones inside the box around it. This is that set, worked out by
    /// hand on a grid small enough to write down.
    ///
    /// The grid has 2 mm nodes, so a region centred on the node at (8, 8, 8)
    /// with semi-axes 5, 3, 3 mm covers the node `(2i, 2j, 2k)` when
    /// `(di/2.5)^2 + (dj/1.5)^2 + (dk/1.5)^2 <= 1` in node steps: five nodes
    /// along the long axis, three where one of the short offsets is spent, and
    /// only the centre one where both are.
    #[test]
    fn an_ellipsoid_support_region_selects_the_nodes_inside_it() {
        let axis_aligned = "region = { shape = \"box\", min = [0.0, 0.0, 0.0], \
                            max = [0.5, 16.0, 16.0] }";
        let ellipsoid = "region = { shape = \"ellipsoid\", center = [8.0, 8.0, 8.0], \
                         radii = [5.0, 3.0, 3.0]";
        let text = cantilever_config("").replace(axis_aligned, &format!("{ellipsoid} }}"));
        let problem = build(&text).expect("build");
        let grid = &problem.grid;
        let selected: std::collections::HashSet<usize> =
            problem.supports[0].nodes.iter().copied().collect();

        let node = |di: i32, dj: i32, dk: i32| {
            grid.node_index((4 + di) as usize, (4 + dj) as usize, (4 + dk) as usize)
        };
        let mut expected = std::collections::HashSet::new();
        for di in -2..=2 {
            expected.insert(node(di, 0, 0));
        }
        for (dj, dk) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            for di in -1..=1 {
                expected.insert(node(di, dj, dk));
            }
        }
        for (dj, dk) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            expected.insert(node(0, dj, dk));
        }
        assert_eq!(expected.len(), 5 + 12 + 4);
        assert_eq!(
            selected, expected,
            "the ellipsoid region selected a different node set"
        );

        // The box around it selects its corners as well, so the shape of the
        // field really is what decided the answer.
        let boxed = cantilever_config("").replace(
            axis_aligned,
            "region = { shape = \"box\", min = [3.0, 5.0, 5.0], max = [13.0, 11.0, 11.0] }",
        );
        assert_eq!(
            build(&boxed).expect("build").supports[0].node_count,
            5 * 3 * 3
        );

        // Turned a quarter turn about z, the long axis is y: the same set, with
        // its first two indices swapped.
        let turned = cantilever_config("").replace(
            axis_aligned,
            &format!("{ellipsoid}, rotation_deg = [0.0, 0.0, 90.0] }}"),
        );
        let turned = build(&turned).expect("build");
        let selected: std::collections::HashSet<usize> =
            turned.supports[0].nodes.iter().copied().collect();
        let mut expected = std::collections::HashSet::new();
        for di in -2..=2 {
            expected.insert(node(0, di, 0));
        }
        for (di, dk) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            for dj in -1..=1 {
                expected.insert(node(di, dj, dk));
            }
        }
        for (di, dk) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            expected.insert(node(di, 0, dk));
        }
        assert_eq!(selected, expected, "the turned region did not turn with it");
    }

    /// The moving asymptotes update keeps four more cell arrays alive than the
    /// optimality criteria step, and the estimate `growforge check` prints has to
    /// say so rather than under-reporting the run the user asked for.
    #[test]
    fn the_memory_estimate_accounts_for_the_update_scheme() {
        let reference = build(&cantilever_config("")).expect("build");
        let text = cantilever_config("").replace(
            "min_feature_mm = 8.0",
            "min_feature_mm = 8.0\nupdate = \"mma\"",
        );
        let candidate = build(&text).expect("build");
        let expected = constants::ESTIMATED_MMA_CELL_ARRAYS
            * reference.grid.n_cells()
            * constants::BYTES_PER_F64;
        assert_eq!(
            candidate.estimated_memory_bytes() - reference.estimated_memory_bytes(),
            expected
        );
    }

    /// A local volume cap carries a kernel, a field and a gradient of its own,
    /// and the estimate has to say so as well - measured against the moving
    /// asymptotes run it is only ever added to, since the cap needs that scheme.
    #[test]
    fn the_memory_estimate_accounts_for_the_local_volume_cap() {
        let mma = cantilever_config("").replace(
            "min_feature_mm = 8.0",
            "min_feature_mm = 8.0\nupdate = \"mma\"",
        );
        let reference = build(&mma).expect("build");
        let candidate = build(&format!("{mma}\n[optimization.local_volume]\n")).expect("build");
        assert!(candidate.optimization.local_volume.is_some());
        let expected = constants::ESTIMATED_LOCAL_VOLUME_CELL_ARRAYS
            * reference.grid.n_cells()
            * constants::BYTES_PER_F64;
        assert_eq!(
            candidate.estimated_memory_bytes() - reference.estimated_memory_bytes(),
            expected
        );
    }

    /// A neighbourhood so large that measuring it would cost more than the
    /// analysis is warned about - and only the voxelization can know that, which
    /// is why the check lives here rather than in the configuration.
    #[test]
    fn an_enormous_local_volume_kernel_is_warned_about() {
        let base = cantilever_config("").replace(
            "min_feature_mm = 8.0",
            "min_feature_mm = 8.0\nupdate = \"mma\"",
        );
        let sane = build(&format!("{base}\n[optimization.local_volume]\n")).expect("build");
        assert!(sane.warnings.is_empty(), "{:?}", sane.warnings);

        // 2 mm voxels and a 40 mm neighbourhood: twenty voxels of reach, some
        // 33000 taps per cell.
        let huge = build(&format!(
            "{base}\n[optimization.local_volume]\nradius_mm = 40.0\n"
        ))
        .expect("build");
        assert_eq!(huge.warnings.len(), 1, "{:?}", huge.warnings);
        assert!(
            huge.warnings[0].contains("[optimization.local_volume]")
                && huge.warnings[0].contains("taps per cell")
                && huge.warnings[0].contains("lower radius_mm"),
            "{:?}",
            huge.warnings
        );
    }

    /// A grid no machine can hold used to abort the process on the allocation
    /// ("capacity overflow"), whatever command asked for it. Both keys can
    /// produce one, so both are checked against the budget.
    #[test]
    fn a_grid_past_the_budget_is_rejected_instead_of_allocated() {
        // The route that was reported: a legal `usize` cell target.
        let absurd = cantilever_config("")
            .replace("voxel_size_mm = 2.0", "target_cells = 12345678901234567890");
        let error = build(&absurd).unwrap_err().to_string();
        assert!(error.contains("target_cells"), "unexpected error: {error}");
        assert!(
            error.contains(&constants::MAX_GRID_CELLS.to_string()),
            "the error must quote the budget: {error}"
        );
        assert!(
            error.contains("derives a voxel size"),
            "the error must say what the key turned into: {error}"
        );
        assert!(error.contains("cells"), "unexpected error: {error}");
        assert!(
            error.contains("lower target_cells"),
            "the remedy must name the key that was used: {error}"
        );

        // The other route to the same runaway grid: a voxel size small enough
        // that an ordinary domain needs more cells than the budget.
        let fine = cantilever_config("").replace("voxel_size_mm = 2.0", "voxel_size_mm = 0.001");
        let error = build(&fine).unwrap_err().to_string();
        assert!(error.contains("voxel_size_mm"), "unexpected error: {error}");
        assert!(
            error.contains(&constants::MAX_GRID_CELLS.to_string()),
            "the error must quote the budget: {error}"
        );
        // 40 x 16 x 16 mm at a micron is 1.02e13 cells, and the error says so
        // rather than reporting a wrapped or saturated count.
        assert!(error.contains("1.024e13"), "the count is wrong: {error}");
        assert!(
            error.contains("raise voxel_size_mm"),
            "the remedy must name the key that was used: {error}"
        );
    }

    /// The counting the budget check is made of, at both sides of the line.
    ///
    /// The boundary is exercised here rather than by building a configuration
    /// that sits just under it: at the budget the model allocates something
    /// like sixteen gigabytes, which is the point of the budget and not
    /// something a test should ask a machine for.
    #[test]
    fn the_budget_admits_the_grid_below_it_and_refuses_the_one_above() {
        let config = Config::parse(&cantilever_config("")).expect("parse");
        let side = 400.0;
        let bounds = crate::geometry::Aabb {
            min: [0.0; 3],
            max: [side; 3],
        };
        // A cube of exactly the budget, and one cell more per axis.
        let cells_per_axis = (constants::MAX_GRID_CELLS as f64).cbrt().floor();
        let at_budget = side / cells_per_axis;
        assert!(check_grid_budget(&config, &bounds, at_budget).is_ok());
        let over_budget = side / (cells_per_axis + 1.0);
        assert!(check_grid_budget(&config, &bounds, over_budget).is_err());
        // A degenerate voxel size is refused rather than rounded into one cell.
        assert!(check_grid_budget(&config, &bounds, f64::NAN).is_err());
    }

    /// A big grid is still an ordinary grid: the guard may not stand in the way
    /// of a real part at a real resolution.
    #[test]
    fn a_large_but_sane_configuration_still_builds() {
        // A 200 mm cube at 2 mm voxels: a million cells, a hundredth of the
        // budget, and the size of a genuine print bed part.
        let text = cantilever_config("")
            .replace("max = [40.0, 16.0, 16.0]", "max = [200.0, 200.0, 200.0]")
            .replace("max = [0.5, 16.0, 16.0]", "max = [0.5, 200.0, 200.0]")
            .replace(
                "center = [40.0, 8.0, 8.0]",
                "center = [200.0, 100.0, 100.0]",
            );
        let problem = build(&text).expect("a large but ordinary grid must build");
        assert_eq!(problem.grid.n_cells(), 100 * 100 * 100);
        assert!(problem.grid.n_cells() < constants::MAX_GRID_CELLS);
    }

    #[test]
    fn support_region_selecting_nothing_is_rejected() {
        let text = cantilever_config("").replace(
            "min = [0.0, 0.0, 0.0], max = [0.5, 16.0, 16.0]",
            "min = [-10.0, -10.0, -10.0], max = [-5.0, -5.0, -5.0]",
        );
        let err = build(&text).unwrap_err().to_string();
        assert!(err.contains("selects no node"), "unexpected error: {err}");
    }

    #[test]
    fn load_region_selecting_nothing_is_rejected() {
        let text = cantilever_config("").replace(
            "center = [40.0, 8.0, 8.0], radius = 3.0",
            "center = [400.0, 8.0, 8.0], radius = 3.0",
        );
        let err = build(&text).unwrap_err().to_string();
        assert!(err.contains("selects no node"), "unexpected error: {err}");
    }

    /// The configuration's own non-fatal findings travel on the same
    /// `warnings` vector the voxelization's do, which is the one channel
    /// `growforge check`, every run header and the editor's validity line all
    /// read. A self-overlapping keepout tube is the first of them.
    #[test]
    fn a_configuration_warning_reaches_the_problem_beside_the_grids_own() {
        let text = cantilever_config(
            r#"
[[keepout]]
shape = "tube"
p1 = [22.6, 8.0, 8.0]
p2 = [17.4, 8.0, 8.0]
bend = [20.0, 10.6, 8.0]
radius = 3.0
"#,
        );
        let problem = build(&text).expect("a self-overlapping tube must still build");
        assert_eq!(problem.warnings.len(), 1, "{:?}", problem.warnings);
        assert!(
            problem.warnings[0].starts_with("[[keepout]] entry 1:"),
            "{:?}",
            problem.warnings
        );
        assert!(
            problem.warnings[0].contains("overlaps itself"),
            "{:?}",
            problem.warnings
        );
        // It really is that keepout, and it really did carve the grid: a
        // warning about a shape that did nothing would be noise.
        assert!(problem.counts.void > 0);
        assert!(problem.boundaries.keepout.shapes()[0].self_intersects());

        // The same tube thinner than its own bend says nothing at all.
        let sound = build(&text.replace("radius = 3.0", "radius = 2.5")).expect("build");
        assert!(sound.warnings.is_empty(), "{:?}", sound.warnings);
    }

    #[test]
    fn keepout_that_empties_the_domain_is_rejected() {
        let text = cantilever_config(
            r#"
[[keepout]]
shape = "box"
min = [-1.0, -1.0, -1.0]
max = [41.0, 17.0, 17.0]
"#,
        );
        let err = build(&text).unwrap_err().to_string();
        assert!(
            err.contains("empty after voxelization"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn keepin_cells_are_not_design_variables() {
        let text = cantilever_config(
            r#"
[[keepin]]
shape = "box"
min = [36.0, 4.0, 4.0]
max = [40.0, 12.0, 12.0]
"#,
        );
        let p = build(&text).expect("build");
        assert_eq!(p.counts.solid, 2 * 4 * 4);
        assert_eq!(p.counts.design, p.grid.n_cells() - p.counts.solid);
    }

    /// Every keepin entry keeps its own cells, which is what lets anything
    /// downstream treat two disjoint pads as two places rather than as one union.
    #[test]
    fn each_keepin_entry_resolves_its_own_cells() {
        let text = cantilever_config(
            r#"
[[keepin]]
shape = "box"
min = [4.0, 4.0, 4.0]
max = [8.0, 12.0, 12.0]

[[keepin]]
shape = "box"
min = [32.0, 4.0, 4.0]
max = [36.0, 12.0, 12.0]
"#,
        );
        let p = build(&text).expect("build");
        assert_eq!(p.keepins.len(), 2);
        for entry in &p.keepins {
            assert_eq!(entry.cells.len(), 2 * 4 * 4, "entry {}", entry.index);
            assert!(
                entry
                    .cells
                    .iter()
                    .all(|&c| p.grid.cells[c] == crate::grid::CellKind::Solid),
                "entry {} claimed a cell that is not solid",
                entry.index
            );
            assert!(
                entry.cells.windows(2).all(|w| w[0] < w[1]),
                "must be sorted"
            );
        }
        // Disjoint entries, and together they are every solid cell.
        assert_eq!(p.keepins[0].index, 1);
        assert_eq!(p.keepins[1].index, 2);
        assert!(
            p.keepins[0]
                .cells
                .iter()
                .all(|c| !p.keepins[1].cells.contains(c))
        );
        assert_eq!(
            p.keepins[0].cells.len() + p.keepins[1].cells.len(),
            p.counts.solid
        );

        // An entry a keepout swallows whole resolves to nothing: keepout wins.
        let swallowed = cantilever_config(
            r#"
[[keepin]]
shape = "box"
min = [4.0, 4.0, 4.0]
max = [8.0, 12.0, 12.0]

[[keepout]]
shape = "box"
min = [3.0, 3.0, 3.0]
max = [9.0, 13.0, 13.0]
"#,
        );
        let p = build(&swallowed).expect("build");
        assert_eq!(p.keepins.len(), 1);
        assert!(
            p.keepins[0].cells.is_empty(),
            "a swallowed entry has no cells: {:?}",
            p.keepins[0].cells
        );
        // And a configuration with no keepin at all has no entries.
        assert!(
            build(&cantilever_config(""))
                .expect("build")
                .keepins
                .is_empty()
        );
    }

    #[test]
    fn overlapping_keepout_and_keepin_warns() {
        let text = cantilever_config(
            r#"
[[keepin]]
shape = "box"
min = [36.0, 4.0, 4.0]
max = [40.0, 12.0, 12.0]

[[keepout]]
shape = "box"
min = [38.0, 4.0, 4.0]
max = [40.0, 12.0, 12.0]
"#,
        );
        let p = build(&text).expect("build");
        assert!(
            p.warnings.iter().any(|w| w.contains("keepout")),
            "warnings: {:?}",
            p.warnings
        );
    }

    #[test]
    fn a_supersampled_export_lattice_over_the_budget_warns_but_builds() {
        // Contrived on purpose: a 160 mm cube at 2 mm voxels is an 80 x 80 x 80
        // grid, whose padded node lattice refined four times is 329^3 samples,
        // about 6.7 GiB of triangles' worth of lattice past the budget.
        let cube = |factor: usize| {
            cantilever_config("")
                .replace("max = [40.0, 16.0, 16.0]", "max = [160.0, 160.0, 160.0]")
                .replace("max = [0.5, 16.0, 16.0]", "max = [0.5, 160.0, 160.0]")
                .replace("center = [40.0, 8.0, 8.0]", "center = [160.0, 80.0, 80.0]")
                .replace(
                    "stl_path = \"t.stl\"",
                    &format!("stl_path = \"t.stl\"\nsupersample = {factor}"),
                )
        };

        let plain = build(&cube(1)).expect("build");
        assert_eq!((plain.grid.nx, plain.grid.ny, plain.grid.nz), (80, 80, 80));
        assert!(
            !plain.warnings.iter().any(|w| w.contains("supersample")),
            "the unrefined lattice must not warn: {:?}",
            plain.warnings
        );
        assert!(
            crate::mesh::supersampled_node_count(&plain.grid, 1)
                <= constants::SUPERSAMPLE_NODE_BUDGET_WARN
        );

        let refined = build(&cube(constants::MAX_SUPERSAMPLE)).expect("build");
        let warning = refined
            .warnings
            .iter()
            .find(|w| w.contains("supersample"))
            .unwrap_or_else(|| panic!("no budget warning: {:?}", refined.warnings));
        assert!(
            warning.contains("MiB") && warning.contains("budget"),
            "unhelpful warning: {warning}"
        );
        // Warned, not refused: the run is still built and still exportable.
        assert_eq!(refined.output.supersample, constants::MAX_SUPERSAMPLE);
        assert!(
            crate::mesh::supersampled_node_count(&refined.grid, constants::MAX_SUPERSAMPLE)
                > constants::SUPERSAMPLE_NODE_BUDGET_WARN
        );
    }

    #[test]
    fn small_min_feature_warns_but_builds() {
        let text = cantilever_config("").replace("min_feature_mm = 8.0", "min_feature_mm = 2.0");
        let p = build(&text).expect("build");
        assert!(
            p.warnings.iter().any(|w| w.contains("min_feature_mm")),
            "warnings: {:?}",
            p.warnings
        );
    }

    #[test]
    fn torque_load_has_zero_net_force_and_matching_moment() {
        let text = cantilever_config("").replace(
            r#"type = "force"
region = { shape = "sphere", center = [40.0, 8.0, 8.0], radius = 3.0 }
vector = [0.0, 0.0, -50.0]"#,
            r#"type = "torque"
region = { shape = "box", min = [39.0, -1.0, -1.0], max = [41.0, 17.0, 17.0] }
axis_point = [40.0, 8.0, 8.0]
axis_dir = [1.0, 0.0, 0.0]
magnitude_nmm = 500.0"#,
        );
        let p = build(&text).expect("build");
        let f = &p.load_cases[0].forces;

        let mut net = [0.0f64; 3];
        let mut moment = [0.0f64; 3];
        for n in 0..p.grid.n_nodes() {
            let base = n * constants::DOF_PER_NODE;
            let force = [f[base], f[base + 1], f[base + 2]];
            if force == [0.0, 0.0, 0.0] {
                continue;
            }
            let (i, j, k) = p.grid.node_ijk(n);
            let r = geometry::difference(p.grid.node_pos(i, j, k), [40.0, 8.0, 8.0]);
            let m = geometry::cross(r, force);
            for ((total, moment_total), (f, mi)) in net
                .iter_mut()
                .zip(moment.iter_mut())
                .zip(force.iter().zip(m.iter()))
            {
                *total += f;
                *moment_total += mi;
            }
        }
        for (d, component) in net.iter().enumerate() {
            assert!(
                component.abs() < 1e-9,
                "net force component {d} is {component}"
            );
        }
        assert!(
            (moment[0] - 500.0).abs() < 1e-6,
            "moment about x is {}",
            moment[0]
        );
        assert!(moment[1].abs() < 1e-6 && moment[2].abs() < 1e-6);
    }

    #[test]
    fn torque_about_a_line_of_nodes_is_rejected() {
        let text = cantilever_config("").replace(
            r#"type = "force"
region = { shape = "sphere", center = [40.0, 8.0, 8.0], radius = 3.0 }
vector = [0.0, 0.0, -50.0]"#,
            r#"type = "torque"
region = { shape = "sphere", center = [40.0, 8.0, 8.0], radius = 0.5 }
axis_point = [40.0, 8.0, 8.0]
axis_dir = [1.0, 0.0, 0.0]
magnitude_nmm = 500.0"#,
        );
        let err = build(&text).unwrap_err().to_string();
        assert!(err.contains("torque axis"), "unexpected error: {err}");
    }

    #[test]
    fn unknown_engine_is_rejected_before_any_work() {
        let text = format!(
            "engine = \"grow\"
{}",
            cantilever_config("")
        );
        let err = build(&text).unwrap_err().to_string();
        assert!(err.contains("unknown engine"), "unexpected error: {err}");
    }

    /// Two gravity loads that are individually valid but oppose each other. The
    /// force load is there so the case still has something driving it, which is
    /// what let this build before the sum was checked.
    fn opposed_gravity(second_g: &str) -> String {
        cantilever_config(&format!(
            r#"
[[loadcases.loads]]
type = "gravity"
direction = [0.0, 0.0, -1.0]
g_mm_s2 = 9810.0

[[loadcases.loads]]
type = "gravity"
direction = [0.0, 0.0, 1.0]
g_mm_s2 = {second_g}
"#
        ))
    }

    #[test]
    fn cancelling_gravity_loads_are_rejected() {
        let err = build(&opposed_gravity("9810.0")).unwrap_err().to_string();
        assert!(
            err.contains("cancel out"),
            "the error must say what went wrong: {err}"
        );
        assert!(
            err.contains("\"main\""),
            "the error must name the load case: {err}"
        );
        // Individually both loads are perfectly legal, so the config itself
        // still parses and validates; only the assembled case can see this.
        Config::parse(&opposed_gravity("9810.0"))
            .expect("parse")
            .validate_static()
            .expect("each load on its own is valid");
    }

    #[test]
    fn near_cancelling_gravity_loads_still_build_and_stay_finite() {
        // A net of 1e-4 mm/s^2 against 9810, a decade clear of the rejection
        // threshold: contrived, but it is a real load and must survive.
        let problem = build(&opposed_gravity("9809.9999")).expect("build");
        let acceleration = problem.load_cases[0].gravity.expect("self weight");
        let net = geometry::length(acceleration);
        assert!(
            net > constants::MIN_NET_GRAVITY_FRACTION * 9810.0,
            "net {net} is not above the rejection threshold"
        );
        assert!((net - 1e-4).abs() < 1e-12, "net acceleration {net}");
        assert!(acceleration[2] < 0.0, "the stronger load must win");

        // The self weight the reporter would print stays finite: the mass falls
        // out of the division rather than the acceleration cancelling into it.
        let densities = vec![1.0; problem.grid.n_cells()];
        let newtons = crate::fea::gravity::total_weight_n(
            &problem.grid,
            &densities,
            acceleration,
            problem.material.density_g_cm3,
        );
        let grams = newtons / net * constants::GRAMS_PER_TONNE;
        assert!(newtons.is_finite() && newtons > 0.0, "weight {newtons} N");
        assert!(grams.is_finite() && grams > 0.0, "mass {grams} g");
    }

    #[test]
    fn a_single_gravity_load_is_never_treated_as_cancelling() {
        let problem = build(&cantilever_config(
            "\n[[loadcases.loads]]\ntype = \"gravity\"\n",
        ))
        .expect("build");
        let acceleration = problem.load_cases[0].gravity.expect("self weight");
        assert!((geometry::length(acceleration) - constants::DEFAULT_GRAVITY_MM_S2).abs() < 1e-9);
    }

    #[test]
    fn the_growth_engine_resolves_its_controls_and_keeps_the_region_node_lists() {
        let text = format!("engine = \"growth\"\n{}", cantilever_config(""));
        let problem = build(&text).expect("build");
        let growth = problem.growth.expect("resolved growth controls");
        assert!(growth.step_mm > 0.0 && growth.kill_radius_mm < growth.attraction_radius_mm);

        // Both region kinds keep the nodes they selected, which is what the
        // growth engine routes between.
        assert_eq!(
            problem.supports[0].nodes.len(),
            problem.supports[0].node_count
        );
        let load = &problem.load_cases[0].loads[0];
        assert_eq!(load.nodes.len(), load.node_count);
        assert!(load.nodes.windows(2).all(|w| w[0] < w[1]), "must be sorted");
        // Every selected node really carries part of the load.
        let forces = &problem.load_cases[0].forces;
        for &n in &load.nodes {
            let base = n * constants::DOF_PER_NODE;
            assert!(forces[base + 2] < 0.0, "node {n} carries no load");
        }
    }

    #[test]
    fn a_growth_step_shorter_than_a_voxel_fraction_is_rejected() {
        let text = format!(
            "engine = \"growth\"\n{}\n[growth]\nstep_mm = 0.01\n",
            cantilever_config("")
        );
        let err = build(&text).unwrap_err().to_string();
        assert!(err.contains("step_mm"), "unexpected error: {err}");
        assert!(err.contains("voxel"), "unexpected error: {err}");
    }

    #[test]
    fn a_gravity_load_keeps_its_node_count_without_a_node_list() {
        let problem = build(&cantilever_config(
            "\n[[loadcases.loads]]\ntype = \"gravity\"\n",
        ))
        .expect("build");
        let gravity = &problem.load_cases[0].loads[1];
        assert_eq!(gravity.kind, "gravity");
        assert!(gravity.node_count > 0);
        assert!(
            gravity.nodes.is_empty(),
            "gravity selects no region, so it has no node list"
        );
    }

    #[test]
    fn fully_constrained_load_case_is_rejected() {
        let text = cantilever_config("").replace(
            "min = [0.0, 0.0, 0.0], max = [0.5, 16.0, 16.0]",
            "min = [-1.0, -1.0, -1.0], max = [41.0, 17.0, 17.0]",
        );
        let err = build(&text).unwrap_err().to_string();
        assert!(
            err.contains("constrained by a support"),
            "unexpected error: {err}"
        );
    }
}
