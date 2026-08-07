//! TOML problem definition: serde types, resolution of presets and defaults,
//! and all validation that can be done before the domain is voxelized.
//!
//! Every table rejects unknown keys, so a typo fails loudly instead of being
//! silently ignored.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

use crate::constants;
use crate::geometry::{Csg, CsgOp, Shape, ShapeUnion, Vec3};

/// A parsed `growforge` problem definition.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Project identity.
    pub project: ProjectConfig,
    /// Optimization engine key. May also be given as `project.engine`; see
    /// [`Config::engine_name`].
    pub engine: Option<String>,
    /// Voxel resolution request.
    pub resolution: ResolutionConfig,
    /// Material properties.
    pub material: MaterialConfig,
    /// Optimization controls.
    pub optimization: OptimizationConfig,
    /// Linear solver controls. Absent means the CPU backend.
    pub solver: Option<SolverConfig>,
    /// Growth engine controls. Only legal when `engine = "growth"`.
    pub growth: Option<GrowthConfig>,
    /// Mesh output controls.
    pub output: OutputConfig,
    /// Ordered CSG list defining the design space.
    #[serde(default)]
    pub domain: Vec<DomainEntry>,
    /// Regions that must stay empty.
    #[serde(default)]
    pub keepout: Vec<ShapeSpec>,
    /// Regions that are forced solid and excluded from the design variables.
    #[serde(default)]
    pub keepin: Vec<ShapeSpec>,
    /// Displacement boundary conditions.
    #[serde(default)]
    pub supports: Vec<SupportSpec>,
    /// Load cases combined into the weighted compliance objective.
    #[serde(default)]
    pub loadcases: Vec<LoadCaseSpec>,
}

/// `[project]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectConfig {
    /// Human readable project name, echoed in reports.
    pub name: String,
    /// Optimization engine key. Equivalent to the top level `engine` key; TOML
    /// binds a bare key written after `[project]` to this table, so both
    /// spellings are accepted (but not both at once).
    pub engine: Option<String>,
}

/// `[resolution]`: exactly one key must be present.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolutionConfig {
    /// Cubic voxel edge length in millimetres.
    pub voxel_size_mm: Option<f64>,
    /// Alternative: derive the voxel size from the domain bounding box so the
    /// grid holds roughly this many cells.
    pub target_cells: Option<usize>,
}

/// `[material]`: either a preset key or a complete set of custom values.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialConfig {
    /// Preset key, one of [`constants::MATERIAL_PRESETS`].
    pub preset: Option<String>,
    /// Young's modulus in MPa.
    pub youngs_modulus_mpa: Option<f64>,
    /// Poisson's ratio.
    pub poisson_ratio: Option<f64>,
    /// Mass density in g/cm^3.
    pub density_g_cm3: Option<f64>,
    /// Tensile yield strength in MPa; stored for later phases.
    pub yield_strength_mpa: Option<f64>,
}

/// `[optimization]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizationConfig {
    /// Fraction of the design cells to keep, in the open interval (0, 1).
    ///
    /// Required by every engine that has a mass target to meet, and **refused**
    /// by the one that has none: `engine = "solid"` fills the domain completely,
    /// so a fraction of it is not a thing that configuration can ask for. See
    /// [`Config::check_solid`].
    pub mass_fraction: Option<f64>,
    /// Smallest structural feature the result should contain, in millimetres.
    pub min_feature_mm: f64,
    /// SIMP penalization exponent. Defaults to [`constants::SIMP_PENALTY`].
    pub penalty: Option<f64>,
    /// Stiffness a design cell at zero density keeps, as a fraction of the solid
    /// modulus - the `Emin` of `E(x) = Emin + x^p (E0 - Emin)`. Defaults to
    /// [`constants::SIMP_EMIN_FRACTION`] and must lie between
    /// [`constants::STIFFNESS_FLOOR_MIN`] and
    /// [`constants::STIFFNESS_FLOOR_MAX`] inclusive.
    ///
    /// The default is nine decades of stiffness contrast across the load path,
    /// and that contrast is what the conjugate gradient's iteration count grows
    /// with: an ill-conditioned model can exhaust its budget still short of the
    /// residual it was asked for. Raising the floor trades void stiffness the
    /// design never bought for a system that solves - negligible while the
    /// floor stays far below what the thinnest real member carries, and no
    /// longer negligible above that.
    ///
    /// Forced void cells are untouched: they carry a literal zero and their
    /// nodes are pinned out of the system, whatever this says.
    pub stiffness_floor: Option<f64>,
    /// Iteration cap.
    pub max_iterations: Option<usize>,
    /// Stop once the largest absolute design variable change drops below this.
    pub convergence_tol: Option<f64>,
    /// Design variable update scheme. Defaults to [`UpdateScheme::Oc`].
    pub update: Option<UpdateScheme>,
    /// Self-supporting (overhang) constraint. Absent means no constraint.
    pub overhang: Option<OverhangConfig>,
    /// Guide wireframe. Absent means no guide.
    pub wireframe: Option<WireframeConfig>,
    /// Local volume constraint. Absent means the global volume target alone.
    pub local_volume: Option<LocalVolumeConfig>,
}

/// How the SIMP loop moves the design variables under the volume constraint.
///
/// The optimality criteria step is the default and the reference: it is what
/// the recorded regression trajectories and the shipped examples were written
/// against, and switching schemes changes every number a run prints. The method
/// of moving asymptotes is the alternative for problems the multiplicative
/// optimality criteria ratio cannot settle - in practice the ones where the
/// self-supporting filter turns a design variable into a near-discrete "does
/// this column print or not" decision. See the crate README.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateScheme {
    /// Optimality criteria: the classical multiplicative update with a move
    /// limit, bisected on the volume Lagrange multiplier.
    #[default]
    Oc,
    /// Svanberg's method of moving asymptotes, specialized to one volume
    /// constraint.
    Mma,
}

impl UpdateScheme {
    /// Config spelling, for reports and error messages.
    pub fn label(self) -> &'static str {
        match self {
            UpdateScheme::Oc => "oc",
            UpdateScheme::Mma => "mma",
        }
    }

    /// Furthest one step of this scheme may move a design variable.
    ///
    /// The two schemes carry their own figure (they happen to be equal, so that
    /// switching schemes does not silently change how far an iteration may
    /// travel), and anything reading a design variable change against "as far as
    /// this iteration was allowed to go" - the stall criterion of
    /// [`crate::engine::stall`] - has to ask the scheme that took the step
    /// rather than assume one of them.
    pub fn move_limit(self) -> f64 {
        match self {
            UpdateScheme::Oc => constants::OC_MOVE_LIMIT,
            UpdateScheme::Mma => constants::MMA_MOVE_LIMIT,
        }
    }
}

/// `[optimization.overhang]`.
///
/// Present means the additive manufacturing filter is chained after the density
/// filter, so the design the analysis and the volume constraint see is already
/// printable. The self-supporting angle is fixed at 45 degrees by the filter's
/// supporting stencil; there is deliberately no angle key.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverhangConfig {
    /// Direction the part is built in: layers stack towards this axis sign, so
    /// the build plate is the face the direction points away from.
    pub build_direction: BuildDirection,
}

/// `[optimization.wireframe]`.
///
/// Present switches the guide wireframe on: a thin keepout-avoiding network
/// joining every load region, every support region and every `[[keepin]]` entry
/// is seeded into the initial density field and held as a density floor for the
/// first iterations, then released. It is a guide, not a constraint - see
/// [`crate::engine::wireframe`] - and SIMP only, like the overhang table.
///
/// Every key is optional; the defaults live in [`crate::constants`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireframeConfig {
    /// Radius of the guide wire in millimetres. Defaults to
    /// [`constants::WIREFRAME_RADIUS_MM`].
    pub radius_mm: Option<f64>,
    /// Iterations the wire is held as a density floor for, counted from the
    /// first. Defaults to [`constants::WIREFRAME_HOLD_ITERATIONS`]; zero seeds
    /// the wire and holds nothing.
    pub hold_iterations: Option<usize>,
    /// Density the wire cells are seeded and held at. Defaults to
    /// [`constants::WIREFRAME_SEED_DENSITY`].
    pub seed_density: Option<f64>,
}

/// `[optimization.local_volume]`.
///
/// Present caps how much material any *neighbourhood* of the design may hold, on
/// top of the global `mass_fraction` target: the optimizer can no longer spend
/// its budget on a few thick members and flush surfaces, and what it builds
/// instead is a network of many thinner ones with porous interiors. See
/// [`crate::engine::local_volume`].
///
/// Every key is optional; the defaults live in [`crate::constants`]. The
/// constraint is SIMP only and `update = "mma"` only - it is a second constraint,
/// and the optimality criteria step has room for exactly one.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalVolumeConfig {
    /// Largest share of material a neighbourhood may hold, in the open interval
    /// (0, 1) and above `mass_fraction`. Defaults to
    /// [`constants::LOCAL_VOLUME_MAX_FRACTION`].
    pub max_fraction: Option<f64>,
    /// Radius of the neighbourhood the share is measured over, in millimetres.
    /// Defaults to [`constants::LOCAL_VOLUME_RADIUS_FACTOR`] times the density
    /// filter radius, which is itself `min_feature_mm / FILTER_RADIUS_DIVISOR`.
    pub radius_mm: Option<f64>,
}

/// `[solver]`: where the finite element linear solves run, and how far they are
/// taken.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverConfig {
    /// Backend the conjugate gradient runs on. Absent - the key or the whole
    /// table - resolves to [`constants::DEFAULT_SOLVER_BACKEND`], the compute
    /// device, and that default is *soft*: [`Config::solver_params`] falls back
    /// to the CPU when this build or this machine has no device, where a
    /// backend named here is an instruction and fails instead.
    ///
    /// Not to be confused with the derived [`Default`] of [`SolverBackend`],
    /// which is the CPU and is a different thing on purpose: it is the
    /// *reference* backend, the one the determinism promises and the recorded
    /// trajectories are written against. Resolving an absent key through that
    /// impl rather than through [`Config::solver_params`] reads as "cpu" over a
    /// configuration every run of which asks for the device.
    pub backend: Option<SolverBackend>,
    /// Relative residual `||r|| / ||b||` the *optimization* loop's solves are
    /// taken to, on either backend. Defaults to
    /// [`constants::CG_RELATIVE_TOLERANCE`] and must lie between
    /// [`constants::CG_TOLERANCE_MIN`] and [`constants::CG_TOLERANCE_MAX`]
    /// inclusive.
    ///
    /// The default is tight and the target is hard: a solve that reaches the
    /// iteration cap short of it fails the run, and a model can sit a factor of
    /// 1.5 away from a number no printed part could tell apart. Loosening it
    /// buys iterations back at the price of displacement precision, which is
    /// no price at all well below 1e-6.
    ///
    /// Stress recovery is deliberately not covered: it runs at its own
    /// [`constants::STRESS_CG_TOLERANCE`], a property of the recovery rather
    /// than of the part.
    pub tolerance: Option<f64>,
}

/// Where the conjugate gradient runs.
///
/// The CPU backend is the default and the reference: it is the one the
/// determinism promises and the recorded regression trajectories are written
/// against, because it is the only one whose arithmetic is fully specified by
/// this crate. The GPU backend is an explicit opt-in and trades that guarantee
/// for speed - see the crate README.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverBackend {
    /// Double precision conjugate gradient on the CPU. Bit-for-bit reproducible
    /// for a given build and thread count.
    #[default]
    Cpu,
    /// Single precision conjugate gradient on a compute device, wrapped in
    /// double precision iterative refinement on the host. Reproducible on one
    /// machine and driver, not guaranteed across machines.
    Gpu,
}

impl SolverBackend {
    /// Config spelling, for reports and error messages.
    pub fn label(self) -> &'static str {
        match self {
            SolverBackend::Cpu => "cpu",
            SolverBackend::Gpu => "gpu",
        }
    }
}

/// Why a run is not on the backend the default asked for.
///
/// Only the *default* ever falls back. A configuration that named its backend
/// gets it or gets an error, because a named backend is an instruction rather
/// than a preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverFallback {
    /// This build was compiled without the `gpu` feature.
    NoFeature,
    /// The feature is there but the machine offered no usable adapter.
    NoAdapter,
}

impl SolverFallback {
    /// One line for the summary, saying what happened and why.
    pub fn note(self) -> &'static str {
        match self {
            SolverFallback::NoFeature => {
                "no `gpu` feature in this build, so the default backend fell back to the cpu"
            }
            SolverFallback::NoAdapter => {
                "no compute adapter on this machine, so the default backend fell back to the cpu"
            }
        }
    }
}

/// Resolved linear solver controls with defaults applied.
///
/// No `Eq`: the tolerance is an `f64`, and the trait it would need is the one
/// floating point does not have. `PartialEq` is what the comparisons here ever
/// wanted - the resolved controls of two configurations, in tests and in the
/// editor's preview check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverParams {
    /// Backend the conjugate gradient runs on, after any fallback.
    pub backend: SolverBackend,
    /// True when the configuration named the backend itself, which is what
    /// makes it an instruction rather than a preference: a named backend that
    /// cannot be provided is an error, and is never quietly softened.
    pub explicit: bool,
    /// Set when the default asked for a backend this build or this machine
    /// could not provide, so the summary can say what is running instead.
    pub fell_back: Option<SolverFallback>,
    /// Relative residual the optimization loop's solves are taken to, already
    /// defaulted and range checked by [`Config::solver_params`].
    pub tolerance: f64,
}

/// Written out rather than derived, because a derived one would hand out a
/// tolerance of `0.0`: a target no residual reaches, so every solve under it
/// would run the whole iteration cap and then fail.
///
/// This is the *reference* configuration - the CPU backend the determinism
/// promises are written against, at the tolerance the recorded trajectories
/// were recorded at - and it is for a `Problem` assembled in memory rather than
/// parsed. A configuration on disk is resolved by [`Config::solver_params`],
/// which is the only path that reads the file's own keys.
impl Default for SolverParams {
    fn default() -> SolverParams {
        SolverParams {
            backend: SolverBackend::default(),
            explicit: false,
            fell_back: None,
            tolerance: constants::CG_RELATIVE_TOLERANCE,
        }
    }
}

/// `[growth]`: controls of the growth heuristic engine.
///
/// Every key is optional; the defaults live in [`crate::constants`]. The four
/// lengths default to multiples of `[optimization] min_feature_mm`, which is the
/// one number in the configuration that states the scale of the part's features.
///
/// Two `[optimization]` keys are deliberately reused rather than duplicated
/// here: `mass_fraction` is the volume target the strut radii are normalized
/// against, and `min_feature_mm` is the smallest strut *diameter* (so the
/// smallest radius is half of it).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrowthConfig {
    /// Seed of the in-crate PCG32 generator. The same seed and the same
    /// configuration produce a byte identical STL.
    pub seed: Option<u64>,
    /// Attraction points scattered per cubic centimetre of design domain.
    pub attractor_per_cm3: Option<f64>,
    /// How far a branch tip can see an attractor, in millimetres.
    pub attraction_radius_mm: Option<f64>,
    /// Distance at which a branch consumes an attractor, in millimetres.
    pub kill_radius_mm: Option<f64>,
    /// Length of one space colonization step, in millimetres.
    pub step_mm: Option<f64>,
    /// Exponent `n` of Murray's law `r_parent^n = sum(r_child^n)`.
    pub murray_exponent: Option<f64>,
    /// Cap on a single strut's radius, in millimetres.
    pub max_radius_mm: Option<f64>,
    /// Cap on space colonization iterations.
    pub max_steps: Option<usize>,
    /// Remove branches that end on nothing. Default
    /// [`constants::DEFAULT_GROWTH_PRUNE`].
    pub prune: Option<bool>,
    /// Grow one fundamental domain and replicate it. Absent means the whole
    /// domain is grown, which is the default and the asymmetric result.
    pub symmetry: Option<SymmetryConfig>,
}

/// `[growth.symmetry]`: grow a fraction of the domain and replicate it.
///
/// Which keys are legal depends on `kind`, and the ones that belong to the
/// other kind are rejected rather than ignored - a `planes` under a rotational
/// symmetry is a misunderstanding worth reporting, not a no-op.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SymmetryConfig {
    /// Which symmetry the structure has.
    pub kind: SymmetryKind,
    /// `kind = "mirror"`: one or two mirror plane **normals**, so `"x"` is the
    /// plane `x = centre`. Required for that kind, at most
    /// [`constants::GROWTH_SYMMETRY_MAX_PLANES`] of them.
    pub planes: Option<Vec<Axis>>,
    /// `kind = "rotational"`: how many sectors the full turn is divided into.
    /// Required for that kind.
    pub order: Option<usize>,
    /// `kind = "rotational"`: axis the rotation turns about, through the domain
    /// centre. Defaults to [`constants::DEFAULT_SYMMETRY_AXIS`].
    pub axis: Option<Axis>,
}

/// Which symmetry a grown structure is replicated with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymmetryKind {
    /// Reflection in one or two axis-aligned planes through the domain centre.
    Mirror,
    /// Rotation by whole `1 / order` turns about an axis through the domain
    /// centre.
    Rotational,
}

impl SymmetryKind {
    /// Config spelling, for reports and for writing a configuration back out.
    pub fn label(self) -> &'static str {
        match self {
            SymmetryKind::Mirror => "mirror",
            SymmetryKind::Rotational => "rotational",
        }
    }
}

/// Resolved growth symmetry.
///
/// The two variants carry exactly what their kind needs, so nothing downstream
/// has to ask whether an `order` belongs to a mirror. Copyable, because
/// [`GrowthParams`] is: a resolved growth control is a plain value that travels
/// with the problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryParams {
    /// One or two mirror planes through the domain centre, named by their
    /// normals. Two planes give four sectors.
    Mirror {
        /// Normal of the first plane.
        first: Axis,
        /// Normal of the second plane, when there is one.
        second: Option<Axis>,
    },
    /// `order`-fold rotation about the domain centre axis `axis`.
    Rotational {
        /// Sectors the full turn is divided into.
        order: usize,
        /// Axis the rotation turns about.
        axis: Axis,
    },
}

impl SymmetryParams {
    /// Kind this resolved symmetry came from.
    pub fn kind(self) -> SymmetryKind {
        match self {
            SymmetryParams::Mirror { .. } => SymmetryKind::Mirror,
            SymmetryParams::Rotational { .. } => SymmetryKind::Rotational,
        }
    }

    /// How many copies of the fundamental domain make up the whole structure,
    /// the identity copy included.
    pub fn sectors(self) -> usize {
        match self {
            SymmetryParams::Mirror { second, .. } => {
                if second.is_some() {
                    4
                } else {
                    2
                }
            }
            SymmetryParams::Rotational { order, .. } => order,
        }
    }

    /// One line naming the symmetry, for reports and warnings.
    pub fn describe(self) -> String {
        match self {
            SymmetryParams::Mirror {
                first,
                second: None,
            } => format!("mirror across {}", first.label()),
            SymmetryParams::Mirror {
                first,
                second: Some(second),
            } => format!("mirror across {} and {}", first.label(), second.label()),
            SymmetryParams::Rotational { order, axis } => {
                format!("{order}-fold rotation about {}", axis.label())
            }
        }
    }
}

/// Resolved growth engine controls with defaults applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrowthParams {
    /// Seed of the deterministic generator.
    pub seed: u64,
    /// Attraction points per cubic centimetre of design domain.
    pub attractor_per_cm3: f64,
    /// Attraction radius in millimetres.
    pub attraction_radius_mm: f64,
    /// Kill radius in millimetres.
    pub kill_radius_mm: f64,
    /// Space colonization step length in millimetres.
    pub step_mm: f64,
    /// Exponent of Murray's law.
    pub murray_exponent: f64,
    /// Largest strut radius in millimetres.
    pub max_radius_mm: f64,
    /// Cap on space colonization iterations.
    pub max_steps: usize,
    /// Remove branches that do not end on a structural surface.
    pub prune: bool,
    /// Symmetry the structure is grown and replicated with, or `None` for a
    /// run that grows the whole domain.
    pub symmetry: Option<SymmetryParams>,
}

/// Direction layers stack in during printing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum BuildDirection {
    /// Layers stack towards +x; the build plate is the x minimum face.
    #[serde(rename = "x+")]
    XPlus,
    /// Layers stack towards -x; the build plate is the x maximum face.
    #[serde(rename = "x-")]
    XMinus,
    /// Layers stack towards +y; the build plate is the y minimum face.
    #[serde(rename = "y+")]
    YPlus,
    /// Layers stack towards -y; the build plate is the y maximum face.
    #[serde(rename = "y-")]
    YMinus,
    /// Layers stack towards +z; the build plate is the z minimum face.
    #[serde(rename = "z+")]
    ZPlus,
    /// Layers stack towards -z; the build plate is the z maximum face.
    #[serde(rename = "z-")]
    ZMinus,
}

impl BuildDirection {
    /// Axis the layers stack along.
    pub fn axis(self) -> usize {
        match self {
            BuildDirection::XPlus | BuildDirection::XMinus => 0,
            BuildDirection::YPlus | BuildDirection::YMinus => 1,
            BuildDirection::ZPlus | BuildDirection::ZMinus => 2,
        }
    }

    /// True when layer indices grow with the axis coordinate.
    pub fn is_positive(self) -> bool {
        matches!(
            self,
            BuildDirection::XPlus | BuildDirection::YPlus | BuildDirection::ZPlus
        )
    }

    /// Config spelling, for reports.
    pub fn label(self) -> &'static str {
        match self {
            BuildDirection::XPlus => "x+",
            BuildDirection::XMinus => "x-",
            BuildDirection::YPlus => "y+",
            BuildDirection::YMinus => "y-",
            BuildDirection::ZPlus => "z+",
            BuildDirection::ZMinus => "z-",
        }
    }
}

/// What to do with cavities that the exported surface would fully enclose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoidPolicy {
    /// Report them and export them as they are.
    #[default]
    Warn,
    /// Fill them with material before meshing, and report what was filled.
    Fill,
}

impl VoidPolicy {
    /// Config spelling, for reports and for writing a configuration back out.
    pub fn label(self) -> &'static str {
        match self {
            VoidPolicy::Warn => "warn",
            VoidPolicy::Fill => "fill",
        }
    }
}

/// What to do with the floating fragments of the extracted surface: shells of
/// material that came out welded to nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IslandPolicy {
    /// Remove them before the mesh is validated and written, and report what
    /// was removed.
    #[default]
    Cull,
    /// Export the surface exactly as it came out of marching cubes, fragments
    /// and all, and report them.
    Keep,
}

impl IslandPolicy {
    /// Config spelling, for reports and for writing a configuration back out.
    pub fn label(self) -> &'static str {
        match self {
            IslandPolicy::Cull => "cull",
            IslandPolicy::Keep => "keep",
        }
    }
}

/// How closely the exported surface has to follow the analytic boundaries the
/// configuration described.
///
/// The isosurface is extracted from a density field sampled on a lattice and
/// then smoothed, and neither step knows the shapes: cells are classified by
/// their centres, so material may reach half a voxel past the boundary that
/// carved it, and Taubin smoothing moves vertices again. The result is a surface
/// that can cut a fraction of a voxel into a keepout - a pin bore comes out
/// under its nominal diameter - or bulge the same distance out of the domain.
/// Supersampling does not help: the refined lattice interpolates the same coarse
/// field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryFidelity {
    /// Project every exported vertex that violates a keepout, or that leaves the
    /// domain, back onto the analytic surface it violates. Where the part meets
    /// a bore, the bore is the cylinder the configuration wrote.
    #[default]
    Exact,
    /// Export the isosurface exactly as the voxel field produced it. The legacy
    /// behaviour, and the escape hatch for a run that has to reproduce a file
    /// written before the clamp existed.
    Voxel,
}

impl BoundaryFidelity {
    /// Config spelling, for reports and for writing a configuration back out.
    pub fn label(self) -> &'static str {
        match self {
            BoundaryFidelity::Exact => "exact",
            BoundaryFidelity::Voxel => "voxel",
        }
    }
}

/// What to do with the material of the final field that carries no load: the
/// prongs that reach to nothing and the gossamer wisps between members.
///
/// Unrelated to `[growth] prune`, which removes branches of a *skeleton* while
/// the growth engine is still building one. This runs after every engine, on the
/// density field itself, and judges material by the stress in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrimPolicy {
    /// Export the field the engine produced, wisps and all.
    #[default]
    Off,
    /// Remove the part cells whose envelope von Mises stress is a negligible
    /// fraction of the part's peak, unless removing them would disconnect two
    /// of the places the configuration declared a purpose for.
    Stress,
}

impl TrimPolicy {
    /// Config spelling, for reports and for writing a configuration back out.
    pub fn label(self) -> &'static str {
        match self {
            TrimPolicy::Off => "off",
            TrimPolicy::Stress => "stress",
        }
    }
}

/// What to do about the material of the final field that is too thin to print:
/// the arms a slicer would lay down as a single bead, or miss entirely.
///
/// Unrelated to the growth engine's own strut sizing (`engine::growth::thicken`),
/// which gives a *skeleton* its radii from the flow through it while the design
/// is still being built. This runs after every engine, on the density field
/// itself, and measures the geometry that is about to be exported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReinforcePolicy {
    /// Export the field the engine produced, however thin its arms came out.
    #[default]
    Off,
    /// Thicken every member thinner than `reinforce_thickness_mm` up to it,
    /// into the design space around it - and never into a keepout, a cavity the
    /// void policy owns or the space outside the domain.
    MinThickness,
}

impl ReinforcePolicy {
    /// Config spelling, for reports and for writing a configuration back out.
    pub fn label(self) -> &'static str {
        match self {
            ReinforcePolicy::Off => "off",
            ReinforcePolicy::MinThickness => "min_thickness",
        }
    }
}

/// What to do about the walls of the final field that stand against a surface
/// the configuration drew but come out short of it.
///
/// Unrelated to `[output] boundaries`, which corrects the same defect on the
/// *mesh* and can only reach what the field put within half a voxel of the
/// surface. This runs after every engine, on the density field itself, and is
/// what puts the material there for the clamp to seat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlushPolicy {
    /// Export the field the engine produced, however far short of its surfaces
    /// the walls came out.
    #[default]
    Off,
    /// Fill the design cells within `flush_depth_mm` of a domain or keepout
    /// surface that the part's own material already reaches into, so a wall
    /// resting against that surface reaches it.
    Walls,
}

impl FlushPolicy {
    /// Config spelling, for reports and for writing a configuration back out.
    pub fn label(self) -> &'static str {
        match self {
            FlushPolicy::Off => "off",
            FlushPolicy::Walls => "walls",
        }
    }
}

/// `[output]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    /// Destination STL path; relative paths resolve against the config file.
    pub stl_path: String,
    /// Isosurface level in density units.
    pub iso_level: Option<f64>,
    /// Number of Taubin smoothing passes.
    pub smoothing_iterations: Option<usize>,
    /// Isosurface lattice refinement factor: 1 meshes the node lattice itself,
    /// N meshes a lattice N times finer in every axis.
    pub supersample: Option<usize>,
    /// Enclosed cavity policy.
    pub voids: Option<VoidPolicy>,
    /// Floating fragment policy of the extracted surface.
    pub islands: Option<IslandPolicy>,
    /// How closely the exported surface has to follow the analytic domain and
    /// keepout surfaces. Absent is [`BoundaryFidelity::Exact`].
    pub boundaries: Option<BoundaryFidelity>,
    /// Unloaded material policy of the final density field.
    pub trim: Option<TrimPolicy>,
    /// Fraction of the envelope von Mises peak the trim pass calls negligible.
    /// Absent is [`constants::TRIM_STRESS_FRACTION`].
    pub trim_stress_fraction: Option<f64>,
    /// Minimum printable thickness policy of the final density field.
    pub reinforce: Option<ReinforcePolicy>,
    /// Thickness floor the reinforcement pass holds the part to, in
    /// millimetres. Absent is `[optimization] min_feature_mm`, the smallest
    /// feature the run was asked to resolve.
    pub reinforce_thickness_mm: Option<f64>,
    /// Flush wall policy of the final density field.
    pub flush: Option<FlushPolicy>,
    /// How deep the flush pass fills against a domain or keepout surface, in
    /// millimetres. Absent is [`constants::FLUSH_DEPTH_VOXELS`] of the voxel
    /// size, which is a property of the grid rather than of this table; see
    /// [`crate::flush::depth_mm`].
    pub flush_depth_mm: Option<f64>,
    /// Destination of the machine readable stress report; relative paths
    /// resolve against the config file. Absent writes no file.
    pub stress_json: Option<String>,
}

/// A shape, tagged by its `shape` key.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case", deny_unknown_fields)]
pub enum ShapeSpec {
    /// Box, axis aligned unless it carries a rotation.
    Box {
        /// Lower corner in mm, before any rotation.
        min: Vec3,
        /// Upper corner in mm, before any rotation.
        max: Vec3,
        /// Rotation about the box **centre** in degrees, applied about the
        /// fixed world axes in x, y, z order (extrinsic XYZ). Absent is the
        /// same as `[0, 0, 0]`: an axis aligned box.
        rotation_deg: Option<Vec3>,
    },
    /// Cylinder capped at `p1` and `p2`.
    Cylinder {
        /// First cap centre.
        p1: Vec3,
        /// Second cap centre.
        p2: Vec3,
        /// Radius in mm.
        radius: f64,
    },
    /// Sphere.
    Sphere {
        /// Centre in mm.
        center: Vec3,
        /// Radius in mm.
        radius: f64,
    },
    /// Ellipsoid, axis aligned unless it carries a rotation.
    Ellipsoid {
        /// Centre in mm.
        center: Vec3,
        /// Semi-axes in mm along the ellipsoid's own x, y and z, before any
        /// rotation. All three must be positive.
        radii: Vec3,
        /// Rotation about the ellipsoid **centre** in degrees, applied about
        /// the fixed world axes in x, y, z order (extrinsic XYZ) - the same
        /// semantics as [`ShapeSpec::Box`]. Absent is the same as `[0, 0, 0]`.
        rotation_deg: Option<Vec3>,
    },
    /// Tube: a round-ended capsule between `p1` and `p2`, curved into a
    /// circular arc when it carries a `bend`.
    Tube {
        /// Centre of the first end cap.
        p1: Vec3,
        /// Centre of the second end cap.
        p2: Vec3,
        /// A point the tube's centre line is bent through. Absent is the
        /// straight capsule, and so is a point on the line through the two
        /// ends; see [`crate::geometry::Shape::Tube`].
        bend: Option<Vec3>,
        /// Radius in mm. Must be positive.
        radius: f64,
    },
    /// Cone: a capped frustum between `p1` and `p2`, carrying a radius at each.
    Cone {
        /// Centre of the wide cap.
        p1: Vec3,
        /// Centre of the other cap.
        p2: Vec3,
        /// Radius at `p1` in mm. Must be positive.
        radius1: f64,
        /// Radius at `p2` in mm. Must not be negative; zero is the apex of a
        /// true cone, and `radius1 == radius2` is the cylinder.
        radius2: f64,
    },
    /// Triangular prism: the triangle through `a`, `b` and `c`, extruded
    /// symmetrically about its own plane.
    Triangle {
        /// First vertex in mm.
        a: Vec3,
        /// Second vertex in mm.
        b: Vec3,
        /// Third vertex in mm.
        c: Vec3,
        /// Depth of the extrusion in mm, half either side of the plane through
        /// the three vertices. Must be positive.
        thickness: f64,
    },
}

impl ShapeSpec {
    /// Config spelling of the `shape` key, for reports and for writing a
    /// configuration back out.
    pub fn kind(&self) -> &'static str {
        match self {
            ShapeSpec::Box { .. } => "box",
            ShapeSpec::Cylinder { .. } => "cylinder",
            ShapeSpec::Sphere { .. } => "sphere",
            ShapeSpec::Ellipsoid { .. } => "ellipsoid",
            ShapeSpec::Tube { .. } => "tube",
            ShapeSpec::Cone { .. } => "cone",
            ShapeSpec::Triangle { .. } => "triangle",
        }
    }

    /// Rotation this shape carries, if it is a kind that can carry one.
    ///
    /// `Some([0, 0, 0])` and `None` mean the same geometry; they differ only in
    /// whether the key is written to the file.
    pub fn rotation_deg(&self) -> Option<Vec3> {
        match *self {
            ShapeSpec::Box { rotation_deg, .. } | ShapeSpec::Ellipsoid { rotation_deg, .. } => {
                rotation_deg
            }
            // A cylinder, a tube and a cone are already pointed by the points
            // they run between, a triangle by the three its vertices are, and a
            // sphere looks the same whichever way it is turned.
            ShapeSpec::Cylinder { .. }
            | ShapeSpec::Sphere { .. }
            | ShapeSpec::Tube { .. }
            | ShapeSpec::Cone { .. }
            | ShapeSpec::Triangle { .. } => None,
        }
    }

    /// Convert to the geometry primitive, rejecting degenerate inputs.
    pub fn to_shape(&self, what: &str) -> Result<Shape> {
        let shape = match *self {
            ShapeSpec::Box {
                min,
                max,
                rotation_deg,
            } => {
                for d in 0..3 {
                    if max[d] <= min[d] {
                        bail!(
                            "{what}: box has non-positive extent on axis {d} (min = {}, max = {})",
                            min[d],
                            max[d]
                        );
                    }
                }
                Shape::Box {
                    min,
                    max,
                    rotation_deg: rotation_deg.unwrap_or_default(),
                }
            }
            ShapeSpec::Cylinder { p1, p2, radius } => Shape::Cylinder { p1, p2, radius },
            ShapeSpec::Sphere { center, radius } => Shape::Sphere { center, radius },
            ShapeSpec::Ellipsoid {
                center,
                radii,
                rotation_deg,
            } => {
                // Named per axis rather than left to the degeneracy guard
                // below: the message says which of the three is wrong, and the
                // field divides by every one of them.
                for (d, radius) in radii.iter().enumerate() {
                    if *radius <= 0.0 || !radius.is_finite() {
                        bail!(
                            "{what}: ellipsoid radii must be positive and finite \
                             (radii[{d}] = {radius})"
                        );
                    }
                }
                Shape::Ellipsoid {
                    center,
                    radii,
                    rotation_deg: rotation_deg.unwrap_or_default(),
                }
            }
            ShapeSpec::Tube {
                p1,
                p2,
                bend,
                radius,
            } => {
                // Named rather than left to the degeneracy guard below, which a
                // tube reaches with its radius alone: a capsule of no length is
                // a ball, so the radius is the one measurement that can make it
                // nothing. A bend on the line through the two ends is not an
                // error - it is the straight tube - so nothing here rejects it.
                if radius <= 0.0 || !radius.is_finite() {
                    bail!("{what}: tube radius must be positive and finite (radius = {radius})");
                }
                Shape::Tube {
                    p1,
                    p2,
                    bend,
                    radius,
                }
            }
            ShapeSpec::Cone {
                p1,
                p2,
                radius1,
                radius2,
            } => {
                // Named per radius rather than left to the degeneracy guard
                // below, because the two are not the same measurement. The wide
                // end has to be a real disc, so `radius1` is checked as every
                // other radius in the schema is; the narrow one is allowed to
                // be *zero*, which is the apex of a true cone, and only a
                // negative one is a shape nothing can sample. See
                // [`Shape::min_extent`], which is what leaves the apex alone.
                if radius1 <= 0.0 || !radius1.is_finite() {
                    bail!("{what}: cone radius1 must be positive and finite (radius1 = {radius1})");
                }
                if radius2 < 0.0 || !radius2.is_finite() {
                    bail!(
                        "{what}: cone radius2 must be a finite number that is not negative \
                         (radius2 = {radius2}); zero is the apex of a true cone"
                    );
                }
                Shape::Cone {
                    p1,
                    p2,
                    radius1,
                    radius2,
                }
            }
            ShapeSpec::Triangle { a, b, c, thickness } => {
                if thickness <= 0.0 || !thickness.is_finite() {
                    bail!(
                        "{what}: triangle thickness must be positive and finite \
                         (thickness = {thickness})"
                    );
                }
                // Three collinear points are refused outright rather than
                // treated as something else, which is what makes this unlike a
                // tube's bend landing on the line through its ends: a bend
                // there is still a tube, while a triangle of no area encloses
                // nothing at any thickness. The shortest altitude is the
                // measurement that says so - it is zero exactly when the three
                // points are collinear, however far apart they are.
                let altitude = crate::geometry::triangle_shortest_altitude(a, b, c);
                if altitude <= constants::MIN_SHAPE_EXTENT_MM || altitude.is_nan() {
                    bail!(
                        "{what}: triangle has no area (its shortest altitude is {altitude} mm, at \
                         or under the {} mm degeneracy guard): the three points are collinear, and \
                         no thickness gives them a solid to extrude",
                        constants::MIN_SHAPE_EXTENT_MM
                    );
                }
                Shape::Triangle { a, b, c, thickness }
            }
        };
        let extent = shape.min_extent();
        if extent <= constants::MIN_SHAPE_EXTENT_MM || extent.is_nan() {
            bail!("{what}: shape is degenerate (smallest extent is {extent})");
        }
        Ok(shape)
    }
}

/// One entry of the ordered domain CSG list.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case", deny_unknown_fields)]
pub enum DomainEntry {
    /// Box, axis aligned unless it carries a rotation.
    Box {
        /// Boolean operation.
        op: CsgOpSpec,
        /// Lower corner in mm, before any rotation.
        min: Vec3,
        /// Upper corner in mm, before any rotation.
        max: Vec3,
        /// Rotation about the box centre in degrees, extrinsic XYZ; see
        /// [`ShapeSpec::Box`].
        rotation_deg: Option<Vec3>,
    },
    /// Cylinder capped at `p1` and `p2`.
    Cylinder {
        /// Boolean operation.
        op: CsgOpSpec,
        /// First cap centre.
        p1: Vec3,
        /// Second cap centre.
        p2: Vec3,
        /// Radius in mm.
        radius: f64,
    },
    /// Sphere.
    Sphere {
        /// Boolean operation.
        op: CsgOpSpec,
        /// Centre in mm.
        center: Vec3,
        /// Radius in mm.
        radius: f64,
    },
    /// Ellipsoid, axis aligned unless it carries a rotation.
    Ellipsoid {
        /// Boolean operation.
        op: CsgOpSpec,
        /// Centre in mm.
        center: Vec3,
        /// Semi-axes in mm, before any rotation; all three must be positive.
        radii: Vec3,
        /// Rotation about the ellipsoid centre in degrees, extrinsic XYZ; see
        /// [`ShapeSpec::Ellipsoid`].
        rotation_deg: Option<Vec3>,
    },
    /// Tube, straight unless it carries a bend point.
    Tube {
        /// Boolean operation.
        op: CsgOpSpec,
        /// Centre of the first end cap.
        p1: Vec3,
        /// Centre of the second end cap.
        p2: Vec3,
        /// A point the centre line is bent through; see [`ShapeSpec::Tube`].
        bend: Option<Vec3>,
        /// Radius in mm.
        radius: f64,
    },
    /// Cone: a capped frustum; see [`ShapeSpec::Cone`].
    Cone {
        /// Boolean operation.
        op: CsgOpSpec,
        /// Centre of the wide cap.
        p1: Vec3,
        /// Centre of the other cap.
        p2: Vec3,
        /// Radius at `p1` in mm; must be positive.
        radius1: f64,
        /// Radius at `p2` in mm; zero is the apex of a true cone.
        radius2: f64,
    },
    /// Triangular prism; see [`ShapeSpec::Triangle`].
    Triangle {
        /// Boolean operation.
        op: CsgOpSpec,
        /// First vertex in mm.
        a: Vec3,
        /// Second vertex in mm.
        b: Vec3,
        /// Third vertex in mm.
        c: Vec3,
        /// Depth of the extrusion in mm, half either side of the plane.
        thickness: f64,
    },
}

impl DomainEntry {
    /// Boolean operation of this entry.
    pub fn op(&self) -> CsgOp {
        let spec = match *self {
            DomainEntry::Box { op, .. } => op,
            DomainEntry::Cylinder { op, .. } => op,
            DomainEntry::Sphere { op, .. } => op,
            DomainEntry::Ellipsoid { op, .. } => op,
            DomainEntry::Tube { op, .. } => op,
            DomainEntry::Cone { op, .. } => op,
            DomainEntry::Triangle { op, .. } => op,
        };
        match spec {
            CsgOpSpec::Add => CsgOp::Add,
            CsgOpSpec::Subtract => CsgOp::Subtract,
        }
    }

    /// Shape of this entry, without the operation.
    pub fn shape_spec(&self) -> ShapeSpec {
        match *self {
            DomainEntry::Box {
                min,
                max,
                rotation_deg,
                ..
            } => ShapeSpec::Box {
                min,
                max,
                rotation_deg,
            },
            DomainEntry::Cylinder { p1, p2, radius, .. } => ShapeSpec::Cylinder { p1, p2, radius },
            DomainEntry::Sphere { center, radius, .. } => ShapeSpec::Sphere { center, radius },
            DomainEntry::Ellipsoid {
                center,
                radii,
                rotation_deg,
                ..
            } => ShapeSpec::Ellipsoid {
                center,
                radii,
                rotation_deg,
            },
            DomainEntry::Tube {
                p1,
                p2,
                bend,
                radius,
                ..
            } => ShapeSpec::Tube {
                p1,
                p2,
                bend,
                radius,
            },
            DomainEntry::Cone {
                p1,
                p2,
                radius1,
                radius2,
                ..
            } => ShapeSpec::Cone {
                p1,
                p2,
                radius1,
                radius2,
            },
            DomainEntry::Triangle {
                a, b, c, thickness, ..
            } => ShapeSpec::Triangle { a, b, c, thickness },
        }
    }
}

/// `op` values accepted by a domain entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CsgOpSpec {
    /// Union this shape into the domain.
    Add,
    /// Cut this shape out of the domain.
    Subtract,
}

impl CsgOpSpec {
    /// Config spelling, for reports and for writing a configuration back out.
    pub fn label(self) -> &'static str {
        match self {
            CsgOpSpec::Add => "add",
            CsgOpSpec::Subtract => "subtract",
        }
    }
}

/// A coordinate axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    /// Global x.
    X,
    /// Global y.
    Y,
    /// Global z.
    Z,
}

impl Axis {
    /// Every axis, in the order a support constrains them when it names none.
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    /// Config spelling, for reports and for writing a configuration back out.
    pub fn label(self) -> &'static str {
        match self {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        }
    }

    /// Degree of freedom offset within a node.
    pub fn dof(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// `[[supports]]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportSpec {
    /// Nodes inside this region are constrained.
    pub region: ShapeSpec,
    /// Constrained directions; all three when omitted.
    pub directions: Option<Vec<Axis>>,
}

/// `[[loadcases]]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadCaseSpec {
    /// Name used in reports.
    pub name: String,
    /// Weight in the compliance objective.
    pub weight: Option<f64>,
    /// Loads applied simultaneously in this case.
    #[serde(default)]
    pub loads: Vec<LoadSpec>,
}

/// One load inside a load case, tagged by its `type` key.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum LoadSpec {
    /// A total force in newtons spread evenly over the selected nodes.
    Force {
        /// Node selection region.
        region: ShapeSpec,
        /// Total force vector in N.
        vector: Vec3,
    },
    /// A total torque in newton-millimetres about an axis.
    Torque {
        /// Node selection region.
        region: ShapeSpec,
        /// A point on the torque axis.
        axis_point: Vec3,
        /// Direction of the torque axis; the sense follows the right hand rule.
        axis_dir: Vec3,
        /// Total torque magnitude in N*mm.
        magnitude_nmm: f64,
    },
    /// The self-weight of the structure itself: a body force proportional to
    /// the physical density of every element, so it changes as the design does.
    Gravity {
        /// Direction gravity pulls in; normalized on use. Defaults to
        /// [`constants::DEFAULT_GRAVITY_DIRECTION`].
        direction: Option<Vec3>,
        /// Gravitational acceleration in mm/s^2. Defaults to
        /// [`constants::DEFAULT_GRAVITY_MM_S2`].
        g_mm_s2: Option<f64>,
    },
}

impl LoadSpec {
    /// Node selection region of this load, if it has one. Gravity acts on every
    /// element and selects nothing.
    pub fn region(&self) -> Option<&ShapeSpec> {
        match self {
            LoadSpec::Force { region, .. } => Some(region),
            LoadSpec::Torque { region, .. } => Some(region),
            LoadSpec::Gravity { .. } => None,
        }
    }

    /// Short label used in reports.
    pub fn kind(&self) -> &'static str {
        match self {
            LoadSpec::Force { .. } => "force",
            LoadSpec::Torque { .. } => "torque",
            LoadSpec::Gravity { .. } => "gravity",
        }
    }

    /// Gravitational acceleration vector of a gravity load in mm/s^2, with the
    /// defaults applied. Errors on a degenerate direction or acceleration.
    pub fn gravity_acceleration(&self, what: &str) -> Result<Option<Vec3>> {
        let LoadSpec::Gravity { direction, g_mm_s2 } = self else {
            return Ok(None);
        };
        let direction = direction.unwrap_or(constants::DEFAULT_GRAVITY_DIRECTION);
        let g = g_mm_s2.unwrap_or(constants::DEFAULT_GRAVITY_MM_S2);
        let len = crate::geometry::length(direction);
        if len == 0.0 || !len.is_finite() {
            bail!("{what}: gravity direction is the zero vector");
        }
        if g <= 0.0 || !g.is_finite() {
            bail!(
                "{what}: gravity g_mm_s2 must be positive (got {g}); flip `direction` to point \
                 gravity the other way"
            );
        }
        Ok(Some(crate::geometry::scale(direction, g / len)))
    }
}

/// Fully resolved isotropic material properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Young's modulus in MPa.
    pub youngs_modulus_mpa: f64,
    /// Poisson's ratio.
    pub poisson_ratio: f64,
    /// Mass density in g/cm^3.
    pub density_g_cm3: f64,
    /// Tensile yield strength in MPa; unused by the phase 1 objective.
    pub yield_strength_mpa: Option<f64>,
}

/// Resolved optimization controls with defaults applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizationParams {
    /// Target fraction of design cells kept.
    pub mass_fraction: f64,
    /// Smallest feature size in millimetres.
    pub min_feature_mm: f64,
    /// Density filter radius in millimetres.
    pub filter_radius_mm: f64,
    /// SIMP penalization exponent `p` of `E(x) = Emin + x^p (E0 - Emin)`.
    pub penalty: f64,
    /// Stiffness floor `Emin` of that interpolation, as a fraction of the solid
    /// modulus. Every stage that interpolates a modulus or differentiates one
    /// reads it from here, so the forward solve and the sensitivity can never
    /// be taken at two different floors.
    pub stiffness_floor: f64,
    /// Iteration cap.
    pub max_iterations: usize,
    /// Design variable change threshold for convergence.
    pub convergence_tol: f64,
    /// Update scheme the SIMP loop moves the design variables with.
    pub update: UpdateScheme,
    /// Build direction of the self-supporting constraint; `None` switches the
    /// additive manufacturing filter off entirely.
    pub overhang: Option<BuildDirection>,
    /// Guide wireframe controls; `None` switches the guide off entirely.
    pub wireframe: Option<WireframeParams>,
    /// Local volume constraint controls; `None` leaves the global volume target
    /// as the only constraint.
    pub local_volume: Option<LocalVolumeParams>,
}

/// Resolved guide wireframe controls with defaults applied.
///
/// Copyable, because [`OptimizationParams`] is: a resolved control is a plain
/// value that travels with the problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WireframeParams {
    /// Radius of the guide wire in millimetres.
    pub radius_mm: f64,
    /// Iterations the wire is held as a density floor for.
    pub hold_iterations: usize,
    /// Density the wire cells are seeded and held at.
    pub seed_density: f64,
}

/// Resolved local volume constraint controls with defaults applied.
///
/// Copyable for the same reason [`WireframeParams`] is: a resolved control is a
/// plain value that travels with the problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalVolumeParams {
    /// Largest share of material a neighbourhood may hold.
    pub max_fraction: f64,
    /// Radius of that neighbourhood in millimetres.
    pub radius_mm: f64,
}

/// Resolved mesh output controls with defaults applied.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputParams {
    /// Absolute destination path of the STL file.
    pub stl_path: PathBuf,
    /// Isosurface level in density units.
    pub iso_level: f64,
    /// Number of Taubin smoothing passes.
    pub smoothing_iterations: usize,
    /// Isosurface lattice refinement factor; 1 is the node lattice itself.
    pub supersample: usize,
    /// What to do with cavities the exported surface would fully enclose.
    pub voids: VoidPolicy,
    /// What to do with the floating fragments of the extracted surface.
    pub islands: IslandPolicy,
    /// How closely the exported surface has to follow the analytic domain and
    /// keepout surfaces.
    pub boundaries: BoundaryFidelity,
    /// What to do with the material of the final field that carries no load.
    pub trim: TrimPolicy,
    /// Fraction of the envelope von Mises peak the trim pass calls negligible.
    pub trim_stress_fraction: f64,
    /// What to do with the material of the final field that is too thin to
    /// print.
    pub reinforce: ReinforcePolicy,
    /// Thickness floor the reinforcement pass holds the part to, in
    /// millimetres, already defaulted to `[optimization] min_feature_mm`. The
    /// one resolved output control that is derived from another table; see
    /// [`Config::output_params`].
    pub reinforce_thickness_mm: f64,
    /// What to do with the walls of the final field that stand short of the
    /// surface they rest against.
    pub flush: FlushPolicy,
    /// How deep the flush pass fills, in millimetres, exactly as the file asked
    /// - the one resolved output control that is **not** defaulted here.
    ///
    /// Its default is [`constants::FLUSH_DEPTH_VOXELS`] of the voxel size, and
    /// the voxel size is a property of the grid: a `[resolution] target_cells`
    /// derives one from the domain bounds, which this method does not have and
    /// would have to build the whole CSG tree to get. [`crate::flush::depth_mm`]
    /// is the one place the derivation happens, and both the pass and the
    /// range check in `Problem::build` read it from there.
    pub flush_depth_mm: Option<f64>,
    /// Absolute destination of the JSON stress report, when one was asked for.
    pub stress_json: Option<PathBuf>,
}

/// Make a configured path absolute against the directory holding the config.
fn resolve_path(raw: &str, config_dir: &Path) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        config_dir.join(path)
    }
}

/// The text of a starter configuration for a file that does not exist yet.
///
/// A block on the floor with a load pressing on a pad at its top: the smallest
/// arrangement that is a *problem* rather than a fragment, so the editor opens
/// on a model that validates, grows and exports as it stands. Every number in
/// it comes from [`constants`] and every one of them is meant to be dragged.
///
/// `name` is what the project is called and what the STL is named after,
/// normally the file's own stem.
pub fn starter_config(name: &str) -> String {
    let side = constants::STARTER_DOMAIN_MM;
    let pad = 0.5 * side * constants::STARTER_PAD_FRACTION;
    let (low, high) = (0.5 * side - pad, 0.5 * side + pad);
    let thickness = constants::STARTER_PAD_MM;
    let top = side - thickness;
    format!(
        "# growforge starter configuration, written by `growforge edit`.\n\
         #\n\
         # A {side:.0} mm block standing on the floor, carrying {load:.0} N on a pad at the\n\
         # top. Everything here is a starting point: drag the objects in the viewport,\n\
         # type exact numbers in the panel, and save when it is the part you want.\n\
         #\n\
         # Units: millimetres, newtons, MPa, N*mm, g/cm^3.\n\
         \n\
         [project]\n\
         name = \"{name}\"\n\
         engine = \"{engine}\"\n\
         \n\
         [resolution]\n\
         voxel_size_mm = {voxel:?}\n\
         \n\
         [material]\n\
         preset = \"{material}\"\n\
         \n\
         [optimization]\n\
         # The fraction of the block to keep, and the smallest feature to resolve.\n\
         mass_fraction = {mass:?}\n\
         min_feature_mm = {feature:?}\n\
         \n\
         [output]\n\
         stl_path = \"{name}.stl\"\n\
         \n\
         # The space the part may occupy.\n\
         [[domain]]\n\
         op = \"add\"\n\
         shape = \"box\"\n\
         min = [0.0, 0.0, 0.0]\n\
         max = [{side:?}, {side:?}, {side:?}]\n\
         \n\
         # A solid pad at the top for the load to push on.\n\
         [[keepin]]\n\
         shape = \"box\"\n\
         min = [{low:?}, {low:?}, {top:?}]\n\
         max = [{high:?}, {high:?}, {side:?}]\n\
         \n\
         # The floor the part stands on.\n\
         [[supports]]\n\
         region = {{ shape = \"box\", min = [-1.0, -1.0, -1.0], max = [{side:?}, {side:?}, {floor:?}] }}\n\
         \n\
         [[loadcases]]\n\
         name = \"top-load\"\n\
         \n\
         # {load:.0} N pressing straight down on the pad.\n\
         [[loadcases.loads]]\n\
         type = \"force\"\n\
         region = {{ shape = \"box\", min = [{low:?}, {low:?}, {top:?}], max = [{high:?}, {high:?}, {side:?}] }}\n\
         vector = [0.0, 0.0, -{load:?}]\n",
        side = side,
        low = low,
        high = high,
        top = top,
        floor = thickness,
        voxel = constants::STARTER_VOXEL_MM,
        mass = constants::STARTER_MASS_FRACTION,
        feature = constants::STARTER_MIN_FEATURE_MM,
        load = constants::STARTER_LOAD_N,
        engine = constants::STARTER_ENGINE,
        material = constants::MATERIAL_PRESETS[0].key,
        name = name,
    )
}

/// Resolve and validate a `[growth.symmetry]` table.
///
/// Each kind is checked for the keys it needs and refuses the ones that belong
/// to the other, so a `planes` written under a rotational symmetry fails loudly
/// instead of being silently dropped - a symmetry that is not the one that was
/// asked for produces a part that is wrong in a way nothing later can catch.
fn symmetry_params(spec: &SymmetryConfig) -> Result<SymmetryParams> {
    let what = "[growth.symmetry]";
    match spec.kind {
        SymmetryKind::Mirror => {
            if spec.order.is_some() || spec.axis.is_some() {
                bail!(
                    "{what}: order and axis belong to kind = \"{}\"; a mirror symmetry is given \
                     by its planes",
                    SymmetryKind::Rotational.label()
                );
            }
            let planes = spec.planes.as_deref().unwrap_or_default();
            if planes.is_empty() {
                bail!(
                    "{what}: kind = \"{}\" needs planes, one or two of \"x\", \"y\", \"z\", naming \
                     each plane by its normal (planes = [\"x\", \"y\"] is the four-fold quartering \
                     of the domain)",
                    SymmetryKind::Mirror.label()
                );
            }
            if planes.len() > constants::GROWTH_SYMMETRY_MAX_PLANES {
                bail!(
                    "{what}: at most {} mirror planes (got {}); three would leave an eighth of \
                     the domain to grow in",
                    constants::GROWTH_SYMMETRY_MAX_PLANES,
                    planes.len()
                );
            }
            if planes.len() == 2 && planes[0] == planes[1] {
                bail!(
                    "{what}: planes lists \"{}\" twice; two mirror planes have to be different \
                     axes",
                    planes[0].label()
                );
            }
            Ok(SymmetryParams::Mirror {
                first: planes[0],
                second: planes.get(1).copied(),
            })
        }
        SymmetryKind::Rotational => {
            if spec.planes.is_some() {
                bail!(
                    "{what}: planes belongs to kind = \"{}\"; a rotational symmetry is given by \
                     its order and axis",
                    SymmetryKind::Mirror.label()
                );
            }
            let Some(order) = spec.order else {
                bail!(
                    "{what}: kind = \"{}\" needs order, the number of sectors the full turn is \
                     divided into",
                    SymmetryKind::Rotational.label()
                );
            };
            if !(constants::GROWTH_SYMMETRY_MIN_ORDER..=constants::GROWTH_SYMMETRY_MAX_ORDER)
                .contains(&order)
            {
                bail!(
                    "{what}: order must lie in [{}, {}] (got {order})",
                    constants::GROWTH_SYMMETRY_MIN_ORDER,
                    constants::GROWTH_SYMMETRY_MAX_ORDER
                );
            }
            Ok(SymmetryParams::Rotational {
                order,
                axis: spec.axis.unwrap_or(constants::DEFAULT_SYMMETRY_AXIS),
            })
        }
    }
}

/// Resolve and validate an `[optimization.wireframe]` table.
///
/// The hold length is not checked against anything here: zero is a legal request
/// (seed the wire, hold nothing), and how it compares with `max_iterations` is a
/// question about the run rather than about the table - the engine reports on it
/// where it can see both.
fn wireframe_params(spec: &WireframeConfig) -> Result<WireframeParams> {
    let what = "[optimization.wireframe]";
    let radius_mm = spec.radius_mm.unwrap_or(constants::WIREFRAME_RADIUS_MM);
    if radius_mm <= 0.0 || !radius_mm.is_finite() {
        bail!("{what}: radius_mm must be positive and finite (got {radius_mm})");
    }
    let seed_density = spec
        .seed_density
        .unwrap_or(constants::WIREFRAME_SEED_DENSITY);
    if !(seed_density > constants::DENSITY_MIN && seed_density <= constants::DENSITY_MAX) {
        bail!(
            "{what}: seed_density must lie in ({}, {}] (got {seed_density}); a wire at zero \
             density is no wire at all",
            constants::DENSITY_MIN,
            constants::DENSITY_MAX
        );
    }
    Ok(WireframeParams {
        radius_mm,
        hold_iterations: spec
            .hold_iterations
            .unwrap_or(constants::WIREFRAME_HOLD_ITERATIONS),
        seed_density,
    })
}

/// Resolve and validate an `[optimization.local_volume]` table.
///
/// `mass_fraction` and `filter_radius_mm` come from the same `[optimization]`
/// table and both are checked against: a cap at or below the global target is
/// infeasible - the mean of the local fractions *is* the global one, and a
/// p-mean is never below a mean - and a neighbourhood no larger than the density
/// filter's own would price a member's cross section rather than how the members
/// are spread.
fn local_volume_params(
    spec: &LocalVolumeConfig,
    mass_fraction: f64,
    filter_radius_mm: f64,
) -> Result<LocalVolumeParams> {
    let what = "[optimization.local_volume]";
    let max_fraction = spec
        .max_fraction
        .unwrap_or(constants::LOCAL_VOLUME_MAX_FRACTION);
    if !(max_fraction > 0.0 && max_fraction < 1.0) {
        bail!(
            "{what}: max_fraction must lie strictly between 0 and 1 (got {max_fraction}); at 1 \
             every neighbourhood may be solid, which is no constraint at all"
        );
    }
    if max_fraction <= mass_fraction {
        bail!(
            "{what}: max_fraction ({max_fraction}) must stay above [optimization] mass_fraction \
             ({mass_fraction}): the average of the local fractions is the global one, so a cap at \
             or below the global target cannot be met by any design; raise max_fraction, or lower \
             mass_fraction"
        );
    }
    let radius_mm = spec
        .radius_mm
        .unwrap_or(constants::LOCAL_VOLUME_RADIUS_FACTOR * filter_radius_mm);
    if !radius_mm.is_finite() || radius_mm <= filter_radius_mm {
        bail!(
            "{what}: radius_mm must be a finite length above the density filter radius \
             ({filter_radius_mm:.4} mm, which is min_feature_mm / {}) (got {radius_mm}); a \
             neighbourhood that small measures one member's own cross section rather than how \
             the members are spread, and only a grey design could satisfy it",
            constants::FILTER_RADIUS_DIVISOR
        );
    }
    Ok(LocalVolumeParams {
        max_fraction,
        radius_mm,
    })
}

/// Reject a number that is not finite.
///
/// TOML has `nan`, `inf` and `-inf` literals and serde accepts all three into
/// an `f64`, so without this every length, load and tolerance in the schema can
/// arrive as a value no arithmetic downstream can do anything with: a NaN
/// spreads silently through the finite element assembly, and an infinity turns
/// a bounding box into a grid nobody can lay out.
fn finite(what: &str, key: &str, value: f64) -> Result<()> {
    if value.is_finite() {
        return Ok(());
    }
    bail!("{what}: {key} must be a finite number (got {value})")
}

/// The same for each component of a vector.
fn finite_vec3(what: &str, key: &str, value: Vec3) -> Result<()> {
    for (d, component) in value.iter().enumerate() {
        finite(what, &format!("{key}[{d}]"), *component)?;
    }
    Ok(())
}

/// The same for every number of a shape, whichever kind it is.
fn finite_shape(what: &str, spec: &ShapeSpec) -> Result<()> {
    match *spec {
        ShapeSpec::Box {
            min,
            max,
            rotation_deg,
        } => {
            finite_vec3(what, "min", min)?;
            finite_vec3(what, "max", max)?;
            // Absent is [0, 0, 0], which is finite; only a written key can hold
            // a `nan`, and a rotation matrix built from one turns every sample
            // of the shape into a number nothing downstream can use.
            if let Some(rotation) = rotation_deg {
                finite_vec3(what, "rotation_deg", rotation)?;
            }
        }
        ShapeSpec::Cylinder { p1, p2, radius } => {
            finite_vec3(what, "p1", p1)?;
            finite_vec3(what, "p2", p2)?;
            finite(what, "radius", radius)?;
        }
        ShapeSpec::Sphere { center, radius } => {
            finite_vec3(what, "center", center)?;
            finite(what, "radius", radius)?;
        }
        ShapeSpec::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } => {
            finite_vec3(what, "center", center)?;
            finite_vec3(what, "radii", radii)?;
            if let Some(rotation) = rotation_deg {
                finite_vec3(what, "rotation_deg", rotation)?;
            }
        }
        ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        } => {
            finite_vec3(what, "p1", p1)?;
            finite_vec3(what, "p2", p2)?;
            // Only a written key can hold a `nan`, and one in the bend point
            // decides whether the tube is an arc at all.
            if let Some(bend) = bend {
                finite_vec3(what, "bend", bend)?;
            }
            finite(what, "radius", radius)?;
        }
        ShapeSpec::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => {
            finite_vec3(what, "p1", p1)?;
            finite_vec3(what, "p2", p2)?;
            finite(what, "radius1", radius1)?;
            finite(what, "radius2", radius2)?;
        }
        ShapeSpec::Triangle { a, b, c, thickness } => {
            finite_vec3(what, "a", a)?;
            finite_vec3(what, "b", b)?;
            finite_vec3(what, "c", c)?;
            finite(what, "thickness", thickness)?;
        }
    }
    Ok(())
}

impl Config {
    /// Read and parse a config file.
    pub fn load(path: &Path) -> Result<Config> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        Config::parse(&text).with_context(|| format!("parsing config file {}", path.display()))
    }

    /// Parse a config from TOML text.
    ///
    /// Finiteness is checked here rather than only in
    /// [`Config::validate_static`], because a configuration holding a NaN is
    /// not something to carry around and reject later: it is one that cannot
    /// be compared with itself, and everything that holds a configuration -
    /// the editor's modified marker above all - is entitled to assume the
    /// numbers in it are numbers.
    pub fn parse(text: &str) -> Result<Config> {
        let config: Config = toml::from_str(text).map_err(|e| anyhow!("{e}"))?;
        config.check_finite()?;
        Ok(config)
    }

    /// Every floating point number in the configuration, of every table, list,
    /// shape and load, must be a finite number.
    ///
    /// One pass over the whole tree rather than a test per key, so a field
    /// added later is covered by construction: what it costs to forget is a
    /// `nan` reaching the solver.
    pub fn check_finite(&self) -> Result<()> {
        finite(
            "[resolution]",
            "voxel_size_mm",
            self.resolution.voxel_size_mm.unwrap_or_default(),
        )?;

        let m = &self.material;
        for (key, value) in [
            ("youngs_modulus_mpa", m.youngs_modulus_mpa),
            ("poisson_ratio", m.poisson_ratio),
            ("density_g_cm3", m.density_g_cm3),
            ("yield_strength_mpa", m.yield_strength_mpa),
        ] {
            finite("[material]", key, value.unwrap_or_default())?;
        }

        let o = &self.optimization;
        finite(
            "[optimization]",
            "mass_fraction",
            o.mass_fraction.unwrap_or_default(),
        )?;
        finite("[optimization]", "min_feature_mm", o.min_feature_mm)?;
        finite("[optimization]", "penalty", o.penalty.unwrap_or_default())?;
        finite(
            "[optimization]",
            "stiffness_floor",
            o.stiffness_floor.unwrap_or_default(),
        )?;
        finite(
            "[optimization]",
            "convergence_tol",
            o.convergence_tol.unwrap_or_default(),
        )?;
        if let Some(w) = &o.wireframe {
            for (key, value) in [("radius_mm", w.radius_mm), ("seed_density", w.seed_density)] {
                finite("[optimization.wireframe]", key, value.unwrap_or_default())?;
            }
        }
        if let Some(l) = &o.local_volume {
            for (key, value) in [("max_fraction", l.max_fraction), ("radius_mm", l.radius_mm)] {
                finite(
                    "[optimization.local_volume]",
                    key,
                    value.unwrap_or_default(),
                )?;
            }
        }

        if let Some(s) = &self.solver {
            finite("[solver]", "tolerance", s.tolerance.unwrap_or_default())?;
        }

        if let Some(g) = &self.growth {
            for (key, value) in [
                ("attractor_per_cm3", g.attractor_per_cm3),
                ("attraction_radius_mm", g.attraction_radius_mm),
                ("kill_radius_mm", g.kill_radius_mm),
                ("step_mm", g.step_mm),
                ("murray_exponent", g.murray_exponent),
                ("max_radius_mm", g.max_radius_mm),
            ] {
                finite("[growth]", key, value.unwrap_or_default())?;
            }
        }

        finite(
            "[output]",
            "iso_level",
            self.output.iso_level.unwrap_or_default(),
        )?;

        for (idx, entry) in self.domain.iter().enumerate() {
            finite_shape(
                &format!("[[domain]] entry {}", idx + 1),
                &entry.shape_spec(),
            )?;
        }
        for (list, specs) in [("[[keepout]]", &self.keepout), ("[[keepin]]", &self.keepin)] {
            for (idx, spec) in specs.iter().enumerate() {
                finite_shape(&format!("{list} entry {}", idx + 1), spec)?;
            }
        }
        for (idx, support) in self.supports.iter().enumerate() {
            finite_shape(&format!("[[supports]] entry {}", idx + 1), &support.region)?;
        }

        for (ci, case) in self.loadcases.iter().enumerate() {
            let what = format!("[[loadcases]] \"{}\" (entry {})", case.name, ci + 1);
            finite(&what, "weight", case.weight.unwrap_or_default())?;
            for (li, load) in case.loads.iter().enumerate() {
                let lwhat = format!("{what}, load {}", li + 1);
                if let Some(region) = load.region() {
                    finite_shape(&lwhat, region)?;
                }
                match load {
                    LoadSpec::Force { vector, .. } => finite_vec3(&lwhat, "vector", *vector)?,
                    LoadSpec::Torque {
                        axis_point,
                        axis_dir,
                        magnitude_nmm,
                        ..
                    } => {
                        finite_vec3(&lwhat, "axis_point", *axis_point)?;
                        finite_vec3(&lwhat, "axis_dir", *axis_dir)?;
                        finite(&lwhat, "magnitude_nmm", *magnitude_nmm)?;
                    }
                    LoadSpec::Gravity { direction, g_mm_s2 } => {
                        finite_vec3(&lwhat, "direction", direction.unwrap_or_default())?;
                        finite(&lwhat, "g_mm_s2", g_mm_s2.unwrap_or_default())?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Engine key, accepting either the top level `engine` or `project.engine`.
    pub fn engine_name(&self) -> Result<&str> {
        match (&self.engine, &self.project.engine) {
            (Some(_), Some(_)) => bail!(
                "engine is set both at the top level and inside [project]; keep only one of them"
            ),
            (Some(e), None) | (None, Some(e)) => Ok(e.as_str()),
            (None, None) => Ok(constants::DEFAULT_ENGINE),
        }
    }

    /// Resolve the material, applying a preset or the custom values.
    pub fn material(&self) -> Result<Material> {
        let m = &self.material;
        let has_custom = m.youngs_modulus_mpa.is_some()
            || m.poisson_ratio.is_some()
            || m.density_g_cm3.is_some()
            || m.yield_strength_mpa.is_some();
        let resolved = match (&m.preset, has_custom) {
            (Some(_), true) => {
                bail!("[material]: give either preset or explicit properties, not both")
            }
            (Some(key), false) => {
                let preset = constants::MATERIAL_PRESETS
                    .iter()
                    .find(|p| p.key == key)
                    .ok_or_else(|| {
                        let known: Vec<&str> =
                            constants::MATERIAL_PRESETS.iter().map(|p| p.key).collect();
                        anyhow!(
                            "[material]: unknown preset \"{key}\"; known presets are {}",
                            known.join(", ")
                        )
                    })?;
                Material {
                    youngs_modulus_mpa: preset.youngs_modulus_mpa,
                    poisson_ratio: preset.poisson_ratio,
                    density_g_cm3: preset.density_g_cm3,
                    yield_strength_mpa: Some(preset.yield_strength_mpa),
                }
            }
            (None, _) => Material {
                youngs_modulus_mpa: m.youngs_modulus_mpa.ok_or_else(|| {
                    anyhow!("[material]: youngs_modulus_mpa is required when no preset is given")
                })?,
                poisson_ratio: m.poisson_ratio.ok_or_else(|| {
                    anyhow!("[material]: poisson_ratio is required when no preset is given")
                })?,
                density_g_cm3: m.density_g_cm3.ok_or_else(|| {
                    anyhow!("[material]: density_g_cm3 is required when no preset is given")
                })?,
                yield_strength_mpa: m.yield_strength_mpa,
            },
        };
        if resolved.youngs_modulus_mpa <= 0.0 || resolved.youngs_modulus_mpa.is_nan() {
            bail!(
                "[material]: youngs_modulus_mpa must be positive (got {})",
                resolved.youngs_modulus_mpa
            );
        }
        let nu = resolved.poisson_ratio;
        if nu.is_nan() || nu <= constants::POISSON_RATIO_MIN || nu >= constants::POISSON_RATIO_MAX {
            bail!(
                "[material]: poisson_ratio must lie in ({}, {}) (got {})",
                constants::POISSON_RATIO_MIN,
                constants::POISSON_RATIO_MAX,
                resolved.poisson_ratio
            );
        }
        if resolved.density_g_cm3 <= 0.0 || resolved.density_g_cm3.is_nan() {
            bail!(
                "[material]: density_g_cm3 must be positive (got {})",
                resolved.density_g_cm3
            );
        }
        if let Some(y) = resolved.yield_strength_mpa
            && (y <= 0.0 || y.is_nan())
        {
            bail!("[material]: yield_strength_mpa must be positive (got {y})");
        }
        Ok(resolved)
    }

    /// Resolve optimization controls and apply defaults.
    ///
    /// Two of the keys are resolved against the engine rather than on their own,
    /// because the solid engine has no use for either and refuses both by name
    /// in [`Config::check_solid`]: `mass_fraction`, which it fills the domain
    /// rather than targeting a share of, and `[optimization.local_volume]`,
    /// which prices a density field it does not have. Resolving them here anyway
    /// would fail first and say something else - a range, or an infeasible cap -
    /// in place of the message that says why the key is not this engine's to
    /// set.
    pub fn optimization_params(&self) -> Result<OptimizationParams> {
        let o = &self.optimization;
        let solid = self.is_solid();
        let mass_fraction = match solid {
            // Every design cell is filled, so the fraction of them that ends up
            // as material is one. It is what the summary lines, the self weight
            // estimate and the stress pass read, and it is the truth about the
            // field the engine returns rather than a stand-in for a missing key.
            true => constants::DENSITY_MAX,
            false => {
                let engine = self.engine_name()?;
                let mass = o.mass_fraction.ok_or_else(|| {
                    anyhow!(
                        "[optimization]: mass_fraction is required with engine = \"{engine}\": it \
                         is the fraction of the design cells that ends up as material, strictly \
                         between 0 and 1. Only engine = \"{}\", which fills the domain completely, \
                         does without it",
                        constants::SOLID_ENGINE
                    )
                })?;
                if !(mass > 0.0 && mass < 1.0) {
                    bail!(
                        "[optimization]: mass_fraction must lie strictly between 0 and 1 (got \
                         {mass})"
                    );
                }
                mass
            }
        };
        if o.min_feature_mm <= 0.0 || o.min_feature_mm.is_nan() {
            bail!(
                "[optimization]: min_feature_mm must be positive (got {})",
                o.min_feature_mm
            );
        }
        let penalty = o.penalty.unwrap_or(constants::SIMP_PENALTY);
        if !penalty.is_finite() || penalty < constants::SIMP_PENALTY_MIN {
            bail!(
                "[optimization]: penalty must be a finite number of at least {} (got {penalty}); \
                 it is the exponent p of E(x) = Emin + x^p (E0 - Emin), and 2 to 5 is the useful \
                 range - {} is the default",
                constants::SIMP_PENALTY_MIN,
                constants::SIMP_PENALTY
            );
        }
        let stiffness_floor = o.stiffness_floor.unwrap_or(constants::SIMP_EMIN_FRACTION);
        if !(stiffness_floor.is_finite()
            && (constants::STIFFNESS_FLOOR_MIN..=constants::STIFFNESS_FLOOR_MAX)
                .contains(&stiffness_floor))
        {
            bail!(
                "[optimization]: stiffness_floor must lie between {:e} and {:e} inclusive (got \
                 {stiffness_floor:e}); it is the Emin of E(x) = Emin + x^p (E0 - Emin), what an \
                 emptied design cell still carries as a fraction of the solid modulus, and below \
                 the lower bound the stiffness contrast is past what the arithmetic resolves while \
                 above the upper the void carries the part rather than the design - {:e} is the \
                 default",
                constants::STIFFNESS_FLOOR_MIN,
                constants::STIFFNESS_FLOOR_MAX,
                constants::SIMP_EMIN_FRACTION
            );
        }
        let max_iterations = o
            .max_iterations
            .unwrap_or(constants::DEFAULT_MAX_ITERATIONS);
        if max_iterations == 0 {
            bail!("[optimization]: max_iterations must be at least 1");
        }
        let convergence_tol = o
            .convergence_tol
            .unwrap_or(constants::DEFAULT_CONVERGENCE_TOL);
        if convergence_tol <= 0.0 || convergence_tol.is_nan() {
            bail!("[optimization]: convergence_tol must be positive (got {convergence_tol})");
        }
        let wireframe = o.wireframe.as_ref().map(wireframe_params).transpose()?;
        let filter_radius_mm = o.min_feature_mm / constants::FILTER_RADIUS_DIVISOR;
        let update = o.update.unwrap_or_default();
        // The scheme gate. It is asked here, where `update` is resolved, and
        // skipped for the two engines that have no update scheme at all - the
        // growth engine and the solid one - both of which reject the table
        // outright with a message of their own, where "set update = \"mma\""
        // would be advice to nowhere.
        if o.local_volume.is_some() && update != UpdateScheme::Mma && !self.is_growth() && !solid {
            bail!(
                "[optimization.local_volume] needs update = \"{}\" (this configuration uses \
                 \"{}\"): the local cap is a second constraint, and the optimality criteria step \
                 prices exactly one - its multiplicative ratio is bisected on a single \
                 multiplier. Set update = \"{}\", or drop the table",
                UpdateScheme::Mma.label(),
                update.label(),
                UpdateScheme::Mma.label()
            );
        }
        let local_volume = match solid {
            // Refused by name in `check_solid`; a cap priced against a mass
            // fraction of one is infeasible by construction and would fail here
            // with that arithmetic instead of with the reason.
            true => None,
            false => o
                .local_volume
                .as_ref()
                .map(|spec| local_volume_params(spec, mass_fraction, filter_radius_mm))
                .transpose()?,
        };
        Ok(OptimizationParams {
            mass_fraction,
            min_feature_mm: o.min_feature_mm,
            filter_radius_mm,
            penalty,
            stiffness_floor,
            max_iterations,
            convergence_tol,
            update,
            overhang: o.overhang.as_ref().map(|o| o.build_direction),
            wireframe,
            local_volume,
        })
    }

    /// Resolve linear solver controls and apply defaults.
    ///
    /// The default is [`constants::DEFAULT_SOLVER_BACKEND`], and it is a *soft*
    /// one: a build with no `gpu` feature resolves it to the CPU here, and a
    /// machine with no adapter resolves it to the CPU when the solver is opened
    /// (see [`crate::fea::backend::LinearSolver::new`]). A backend the
    /// configuration asked for **by name** is not softened at all - the request
    /// is an instruction, and `gpu` on a build that cannot provide it is
    /// rejected here so that `growforge check` says so before any work is done.
    ///
    /// `tolerance` defaults to [`constants::CG_RELATIVE_TOLERANCE`] and is
    /// range checked here, so `growforge check` refuses a target no solve could
    /// meet before any work is done. It reaches the optimization loop's solves
    /// only: stress recovery keeps its own constants.
    ///
    /// Nothing in here opens a device: this runs on every keystroke in the
    /// editor, and asking the driver for an adapter per keystroke is not what a
    /// validation pass may cost.
    pub fn solver_params(&self) -> Result<SolverParams> {
        let asked = self.solver.as_ref().and_then(|s| s.backend);
        #[cfg(not(feature = "gpu"))]
        if asked == Some(SolverBackend::Gpu) {
            bail!(
                "[solver]: backend = \"{}\" needs the `gpu` cargo feature, which this build was \
                 compiled without; rebuild with the default features (cargo build --release), or \
                 set backend = \"{}\"",
                SolverBackend::Gpu.label(),
                SolverBackend::Cpu.label()
            );
        }
        let tolerance = self
            .solver
            .as_ref()
            .and_then(|s| s.tolerance)
            .unwrap_or(constants::CG_RELATIVE_TOLERANCE);
        if !(tolerance.is_finite()
            && (constants::CG_TOLERANCE_MIN..=constants::CG_TOLERANCE_MAX).contains(&tolerance))
        {
            bail!(
                "[solver]: tolerance must lie between {:e} and {:e} inclusive (got \
                 {tolerance:e}); it is the relative residual the optimization's solves stop at, \
                 and below the lower bound the arithmetic cannot hold one while above the upper \
                 the compliance carries more of the solver's error than of the design's",
                constants::CG_TOLERANCE_MIN,
                constants::CG_TOLERANCE_MAX
            );
        }
        let backend = asked.unwrap_or(constants::DEFAULT_SOLVER_BACKEND);
        let mut params = SolverParams {
            backend,
            explicit: asked.is_some(),
            fell_back: None,
            tolerance,
        };
        if !cfg!(feature = "gpu") && !params.explicit && params.backend == SolverBackend::Gpu {
            params.backend = SolverBackend::Cpu;
            params.fell_back = Some(SolverFallback::NoFeature);
        }
        Ok(params)
    }

    /// True when the configuration selected the growth engine.
    pub fn is_growth(&self) -> bool {
        matches!(self.engine_name(), Ok(name) if name == constants::GROWTH_ENGINE)
    }

    /// True when the configuration selected the solid engine.
    pub fn is_solid(&self) -> bool {
        matches!(self.engine_name(), Ok(name) if name == constants::SOLID_ENGINE)
    }

    /// Reject everything `engine = "solid"` cannot mean.
    ///
    /// The rejecting half of a `*_params` pass with nothing to resolve: the
    /// solid engine has no controls of its own, so what it needs from the
    /// configuration is that nothing in it describes an optimization that is not
    /// going to happen. Six things do - a mass target, a build direction, a
    /// guide, a porosity cap, and either of the two passes over the finished
    /// field - and each is refused by name rather than ignored, because a
    /// configuration that asks for one is a configuration whose author expects
    /// something this engine does not do.
    ///
    /// The keys that are merely *unused* are left alone, exactly as the growth
    /// engine leaves them: `penalty`, `stiffness_floor`, `min_feature_mm`,
    /// `max_iterations`, `convergence_tol` and `update` resolve to their
    /// defaults and cost nothing, and `[[supports]]` and `[[loadcases]]` are
    /// still required - the post-run stress report is what says whether the part
    /// holds.
    pub fn check_solid(&self) -> Result<()> {
        if !self.is_solid() {
            return Ok(());
        }
        if self.optimization.mass_fraction.is_some() {
            bail!(
                "engine = \"{}\" cannot be combined with [optimization] mass_fraction: the solid \
                 engine fills the domain completely, so there is no fraction of it to target - the \
                 part is the design space itself. Remove the key, or run this problem with engine \
                 = \"{}\"",
                constants::SOLID_ENGINE,
                constants::DEFAULT_ENGINE
            );
        }
        if self.optimization.overhang.is_some() {
            bail!(
                "engine = \"{}\" cannot be combined with [optimization.overhang]: the \
                 self-supporting filter is a stage of the SIMP density chain, and the solid engine \
                 has no density chain to chain it onto. There is no solid equivalent either - what \
                 a fully dense domain needs to print is supports under it, which is the slicer's \
                 business rather than the design's. Drop the table, or run this problem with engine \
                 = \"{}\"",
                constants::SOLID_ENGINE,
                constants::DEFAULT_ENGINE
            );
        }
        if self.optimization.wireframe.is_some() {
            bail!(
                "engine = \"{}\" cannot be combined with [optimization.wireframe]: the guide \
                 wireframe seeds a density field and holds a floor under the SIMP update, and the \
                 solid engine runs no update to hold anything under. There is no solid equivalent \
                 either, and none is needed: every cell the guide would route through is already \
                 full. Drop the table, or run this problem with engine = \"{}\"",
                constants::SOLID_ENGINE,
                constants::DEFAULT_ENGINE
            );
        }
        if self.optimization.local_volume.is_some() {
            bail!(
                "engine = \"{}\" cannot be combined with [optimization.local_volume]: the local cap \
                 forbids material from concentrating, and a fully dense part is that concentration \
                 everywhere - no design satisfies a cap below one, and the solid engine has no \
                 design variable to move towards one anyway. Drop the table, or run this problem \
                 with engine = \"{}\"",
                constants::SOLID_ENGINE,
                constants::DEFAULT_ENGINE
            );
        }
        // The three passes over the finished field. All are legal for every other
        // engine and all would take this one's whole point away: what the solid
        // engine exports is the domain, exactly, and a pass that removes or adds
        // material to it exports something else. The flush is refused for the
        // same reason and not because it would be wrong: a fully dense domain
        // already reaches its own surfaces, so the pass has nothing to correct
        // there and the boundary clamp alone puts the mesh on them.
        let trim = self.output.trim.unwrap_or_default();
        let reinforce = self.output.reinforce.unwrap_or_default();
        let flush = self.output.flush.unwrap_or_default();
        for (key, asked, label, off) in [
            (
                "trim",
                trim != TrimPolicy::Off,
                trim.label(),
                TrimPolicy::Off.label(),
            ),
            (
                "reinforce",
                reinforce != ReinforcePolicy::Off,
                reinforce.label(),
                ReinforcePolicy::Off.label(),
            ),
            (
                "flush",
                flush != FlushPolicy::Off,
                flush.label(),
                FlushPolicy::Off.label(),
            ),
        ] {
            if asked {
                bail!(
                    "engine = \"{}\" cannot be combined with [output] {key} = \"{label}\": the \
                     solid engine exports exactly the domain that was described, and this pass \
                     would alter it - {key} is for a part an optimizer produced, where some of the \
                     material carries nothing, came out too thin, or came out short of a surface \
                     it rests on. Set {key} = \"{off}\", or run this problem with engine = \"{}\"",
                    constants::SOLID_ENGINE,
                    constants::DEFAULT_ENGINE
                );
            }
        }
        Ok(())
    }

    /// Resolve growth controls and apply defaults.
    ///
    /// `min_feature_mm` comes from `[optimization]` and sets the scale of every
    /// length default. Returns `None` when the growth engine is not selected,
    /// which is also when a `[growth]` table is rejected.
    pub fn growth_params(&self) -> Result<Option<GrowthParams>> {
        if !self.is_growth() {
            // The symmetry table gets its own message: it is the one growth
            // control a SIMP user is likely to reach for by name, and "remove
            // [growth]" would not say why the thing they asked for is missing.
            if self.growth.as_ref().is_some_and(|g| g.symmetry.is_some()) {
                bail!(
                    "[growth.symmetry] is only valid with engine = \"{}\" (this configuration \
                     selects \"{}\"): symmetry replicates a grown skeleton, and the SIMP engine \
                     has no skeleton to replicate. There is no SIMP equivalent either - a \
                     symmetry constraint on the density field is not available. Drop the table, \
                     or run this problem with engine = \"{}\"",
                    constants::GROWTH_ENGINE,
                    self.engine_name()?,
                    constants::GROWTH_ENGINE
                );
            }
            if self.growth.is_some() {
                bail!(
                    "[growth] is only valid with engine = \"{}\" (this configuration selects \
                     \"{}\"); remove the table, or switch engines",
                    constants::GROWTH_ENGINE,
                    self.engine_name()?
                );
            }
            return Ok(None);
        }
        if self.optimization.overhang.is_some() {
            bail!(
                "engine = \"{}\" cannot be combined with [optimization.overhang]: the \
                 self-supporting filter is a stage of the SIMP density chain, and the growth \
                 engine has no density chain to chain it onto. There is no growth equivalent \
                 either - orienting a grown structure by build direction is not available. Drop \
                 the table, or run this problem with engine = \"{}\"",
                constants::GROWTH_ENGINE,
                constants::DEFAULT_ENGINE
            );
        }
        if self.optimization.wireframe.is_some() {
            bail!(
                "engine = \"{}\" cannot be combined with [optimization.wireframe]: the guide \
                 wireframe seeds a density field and holds a floor under the SIMP update, and the \
                 growth engine has no density field to seed. There is no growth equivalent either \
                 - a growth run already routes a strut from every load region to the supports, \
                 which is what the guide is a stand-in for. Drop the table, or run this problem \
                 with engine = \"{}\"",
                constants::GROWTH_ENGINE,
                constants::DEFAULT_ENGINE
            );
        }
        if self.optimization.local_volume.is_some() {
            bail!(
                "engine = \"{}\" cannot be combined with [optimization.local_volume]: the local \
                 cap is a second constraint the moving asymptotes step prices against a density \
                 field, and the growth engine has neither - it places struts and sizes them by \
                 Murray's law, so there is no density to cap. There is no growth equivalent \
                 either, and none is needed: a grown canopy is already a network of many thin \
                 members rather than a solid block. Drop the table, or run this problem with \
                 engine = \"{}\"",
                constants::GROWTH_ENGINE,
                constants::DEFAULT_ENGINE
            );
        }

        // Validated separately, and the length defaults are derived from them.
        let optimization = self.optimization_params()?;
        let min_feature_mm = optimization.min_feature_mm;
        // The kill radius decides how much skeleton there will be, so its
        // default is the one that leaves `mass_fraction` reachable; see
        // `constants::GROWTH_KILL_MASS_FRACTION_MARGIN`.
        let default_kill = 0.5 * min_feature_mm * constants::GROWTH_KILL_MASS_FRACTION_MARGIN
            / optimization.mass_fraction.sqrt();
        let g = self.growth.as_ref();
        let positive = |value: Option<f64>, key: &str, fallback: f64| -> Result<f64> {
            let resolved = value.unwrap_or(fallback);
            if resolved <= 0.0 || !resolved.is_finite() {
                bail!("[growth]: {key} must be positive and finite (got {resolved})");
            }
            Ok(resolved)
        };

        let params = GrowthParams {
            seed: g
                .and_then(|g| g.seed)
                .unwrap_or(constants::DEFAULT_GROWTH_SEED),
            attractor_per_cm3: positive(
                g.and_then(|g| g.attractor_per_cm3),
                "attractor_per_cm3",
                constants::DEFAULT_ATTRACTOR_PER_CM3,
            )?,
            attraction_radius_mm: positive(
                g.and_then(|g| g.attraction_radius_mm),
                "attraction_radius_mm",
                constants::GROWTH_ATTRACTION_PER_KILL * default_kill,
            )?,
            kill_radius_mm: positive(
                g.and_then(|g| g.kill_radius_mm),
                "kill_radius_mm",
                default_kill,
            )?,
            step_mm: positive(
                g.and_then(|g| g.step_mm),
                "step_mm",
                constants::GROWTH_STEP_PER_MIN_FEATURE * min_feature_mm,
            )?,
            murray_exponent: g
                .and_then(|g| g.murray_exponent)
                .unwrap_or(constants::DEFAULT_MURRAY_EXPONENT),
            max_radius_mm: positive(
                g.and_then(|g| g.max_radius_mm),
                "max_radius_mm",
                constants::GROWTH_MAX_RADIUS_PER_MIN_FEATURE * min_feature_mm,
            )?,
            max_steps: g
                .and_then(|g| g.max_steps)
                .unwrap_or(constants::DEFAULT_GROWTH_MAX_STEPS),
            prune: g
                .and_then(|g| g.prune)
                .unwrap_or(constants::DEFAULT_GROWTH_PRUNE),
            symmetry: g
                .and_then(|g| g.symmetry.as_ref())
                .map(symmetry_params)
                .transpose()?,
        };

        if params.kill_radius_mm >= params.attraction_radius_mm {
            bail!(
                "[growth]: kill_radius_mm ({:.3}) must stay below attraction_radius_mm ({:.3}), \
                 or every attractor is consumed before it can steer a branch",
                params.kill_radius_mm,
                params.attraction_radius_mm
            );
        }
        if params.murray_exponent < constants::MURRAY_EXPONENT_MIN
            || params.murray_exponent > constants::MURRAY_EXPONENT_MAX
            || params.murray_exponent.is_nan()
        {
            bail!(
                "[growth]: murray_exponent must lie in [{}, {}] (got {})",
                constants::MURRAY_EXPONENT_MIN,
                constants::MURRAY_EXPONENT_MAX,
                params.murray_exponent
            );
        }
        if params.max_radius_mm < 0.5 * min_feature_mm {
            bail!(
                "[growth]: max_radius_mm ({:.3}) is below the smallest strut radius \
                 min_feature_mm / 2 ({:.3}), so no radius could satisfy both",
                params.max_radius_mm,
                0.5 * min_feature_mm
            );
        }
        if params.max_steps == 0 {
            bail!("[growth]: max_steps must be at least 1");
        }
        Ok(Some(params))
    }

    /// Resolve output controls, applying defaults and making `stl_path`
    /// absolute relative to the directory holding the config file.
    ///
    /// One of those defaults is not the `[output]` table's own:
    /// `reinforce_thickness_mm` falls back to `[optimization] min_feature_mm`,
    /// the smallest feature the run was asked to resolve, which is the smallest
    /// one it can be held to. The seam is here rather than in
    /// [`Problem::build`](crate::problem::Problem::build) because this is a
    /// method of the whole configuration and both tables are already in hand -
    /// the same reason `[optimization.local_volume]`'s radius derives from the
    /// density filter radius inside [`Config::optimization_params`]. Resolving
    /// it downstream would leave `OutputParams` carrying an unresolved option
    /// that every consumer had to default again.
    pub fn output_params(&self, config_dir: &Path) -> Result<OutputParams> {
        let out = &self.output;
        if out.stl_path.trim().is_empty() {
            bail!("[output]: stl_path must not be empty");
        }
        let stl_path = resolve_path(&out.stl_path, config_dir);
        let iso_level = out.iso_level.unwrap_or(constants::DEFAULT_ISO_LEVEL);
        if !(iso_level > constants::DENSITY_MIN && iso_level < constants::DENSITY_MAX) {
            bail!(
                "[output]: iso_level must lie strictly between {} and {} (got {iso_level})",
                constants::DENSITY_MIN,
                constants::DENSITY_MAX
            );
        }
        let supersample = out.supersample.unwrap_or(constants::DEFAULT_SUPERSAMPLE);
        if !(1..=constants::MAX_SUPERSAMPLE).contains(&supersample) {
            bail!(
                "[output]: supersample must lie in [1, {}] (got {supersample}); it multiplies the \
                 triangle count and the file size by about its square",
                constants::MAX_SUPERSAMPLE
            );
        }
        let trim_stress_fraction = out
            .trim_stress_fraction
            .unwrap_or(constants::TRIM_STRESS_FRACTION);
        // Checked whichever policy is in force: a value the file cannot mean is
        // a mistake in the file, and switching the pass on later must not be
        // what discovers it. Both ends are excluded - at zero the pass can
        // never remove anything, and at one it would call the peak itself
        // negligible.
        if !(trim_stress_fraction > 0.0 && trim_stress_fraction < 1.0) {
            bail!(
                "[output]: trim_stress_fraction must lie strictly between 0 and 1 (got \
                 {trim_stress_fraction}); it is the fraction of the part's peak von Mises stress \
                 below which material counts as carrying nothing"
            );
        }
        let reinforce_thickness_mm = out
            .reinforce_thickness_mm
            .unwrap_or(self.optimization.min_feature_mm);
        // Checked whichever policy is in force, for the reason the fraction
        // above is: a value the file cannot mean is a mistake in the file, and
        // switching the pass on later must not be what discovers it.
        if !reinforce_thickness_mm.is_finite() || reinforce_thickness_mm <= 0.0 {
            bail!(
                "[output]: reinforce_thickness_mm must be a positive length in millimetres (got \
                 {reinforce_thickness_mm}); it is the thickness the reinforcement pass holds every \
                 member of the part to, and absent it is [optimization] min_feature_mm"
            );
        }
        // Only that it is a length at all. How deep a flush may sensibly be is
        // measured in voxels of the grid the run is solved on, and the voxel
        // size is not known here; `Problem::build` holds that end of it, the
        // way it holds `[growth] step_mm`'s. Checked whichever policy is in
        // force, for the reason the two above are.
        if let Some(depth) = out.flush_depth_mm
            && (!depth.is_finite() || depth <= 0.0)
        {
            bail!(
                "[output]: flush_depth_mm must be a positive length in millimetres (got {depth}); \
                 it is how deep the flush pass fills against a domain or keepout surface, and \
                 absent it is {} voxels of the grid",
                constants::FLUSH_DEPTH_VOXELS
            );
        }
        let stress_json = match &out.stress_json {
            Some(path) if path.trim().is_empty() => {
                bail!("[output]: stress_json must not be empty; remove the key to write no file")
            }
            Some(path) => Some(resolve_path(path, config_dir)),
            None => None,
        };
        Ok(OutputParams {
            stl_path,
            iso_level,
            smoothing_iterations: out
                .smoothing_iterations
                .unwrap_or(constants::DEFAULT_SMOOTHING_ITERATIONS),
            supersample,
            voids: out.voids.unwrap_or_default(),
            islands: out.islands.unwrap_or_default(),
            boundaries: out.boundaries.unwrap_or_default(),
            trim: out.trim.unwrap_or_default(),
            trim_stress_fraction,
            reinforce: out.reinforce.unwrap_or_default(),
            reinforce_thickness_mm,
            flush: out.flush.unwrap_or_default(),
            flush_depth_mm: out.flush_depth_mm,
            stress_json,
        })
    }

    /// Voxel edge length in millimetres, derived from `target_cells` when the
    /// resolution is expressed that way.
    pub fn voxel_size(&self, domain_bounds_volume_mm3: f64) -> Result<f64> {
        let r = &self.resolution;
        match (r.voxel_size_mm, r.target_cells) {
            (Some(_), Some(_)) => {
                bail!("[resolution]: set exactly one of voxel_size_mm and target_cells, not both")
            }
            (None, None) => {
                bail!("[resolution]: set exactly one of voxel_size_mm and target_cells")
            }
            (Some(h), None) => {
                if h <= 0.0 || h.is_nan() {
                    bail!("[resolution]: voxel_size_mm must be positive (got {h})");
                }
                Ok(h)
            }
            (None, Some(cells)) => {
                if cells < constants::MIN_TARGET_CELLS {
                    bail!(
                        "[resolution]: target_cells must be at least {} (got {cells})",
                        constants::MIN_TARGET_CELLS
                    );
                }
                if domain_bounds_volume_mm3 <= 0.0 || domain_bounds_volume_mm3.is_nan() {
                    bail!(
                        "[resolution]: cannot derive a voxel size from an empty domain bounding box"
                    );
                }
                Ok((domain_bounds_volume_mm3 / cells as f64).cbrt())
            }
        }
    }

    /// Build the domain CSG tree.
    pub fn domain_csg(&self) -> Result<Csg> {
        if self.domain.is_empty() {
            bail!("no [[domain]] entries: the design space is empty");
        }
        let mut entries = Vec::with_capacity(self.domain.len());
        let mut has_add = false;
        for (idx, entry) in self.domain.iter().enumerate() {
            let shape = entry
                .shape_spec()
                .to_shape(&format!("[[domain]] entry {}", idx + 1))?;
            let op = entry.op();
            has_add |= op == CsgOp::Add;
            entries.push((op, shape));
        }
        if !has_add {
            bail!("[[domain]] contains no \"add\" entry: the design space is empty");
        }
        Ok(Csg::new(entries))
    }

    /// Build the keepout union.
    pub fn keepout_union(&self) -> Result<ShapeUnion> {
        let mut shapes = Vec::with_capacity(self.keepout.len());
        for (idx, s) in self.keepout.iter().enumerate() {
            shapes.push(s.to_shape(&format!("[[keepout]] entry {}", idx + 1))?);
        }
        Ok(ShapeUnion::new(shapes))
    }

    /// Build the keepin union.
    pub fn keepin_union(&self) -> Result<ShapeUnion> {
        let mut shapes = Vec::with_capacity(self.keepin.len());
        for (idx, s) in self.keepin.iter().enumerate() {
            shapes.push(s.to_shape(&format!("[[keepin]] entry {}", idx + 1))?);
        }
        Ok(ShapeUnion::new(shapes))
    }

    /// The non-fatal counterpart of [`Config::validate_static`]: what is worth
    /// saying about a configuration that is legal, builds, and runs.
    ///
    /// [`crate::problem::Problem::build`] collects these beside the warnings only
    /// the voxelization can raise, so they reach `growforge check`, every run's
    /// header and the editor's own validity line through the one channel all of
    /// them already use.
    ///
    /// Scoped to the `[[keepout]]` entries, because that is where a
    /// self-overlapping tube has a consequence. The export's boundary clamp
    /// projects a vertex onto the surface of the *keepout member* it violates
    /// ([`crate::mesh::clamp`]), and a tube that has swallowed its own inside has
    /// no such projection to offer. The domain is held to its composite signed
    /// distance rather than shape by shape, which stays exact for such a tube, and
    /// a support, load or keepin region is never projected onto at all - warning
    /// about those would name a consequence they do not have.
    pub fn static_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        for (idx, spec) in self.keepout.iter().enumerate() {
            let what = format!("[[keepout]] entry {}", idx + 1);
            // A shape that will not convert at all is `validate_static`'s to
            // report, not this function's to guess about.
            let Ok(shape) = spec.to_shape(&what) else {
                continue;
            };
            if shape.self_intersects() {
                warnings.push(format!(
                    "{what}: this tube overlaps itself - its radius either reaches past the centre \
                     of its own bend, or is more than half the distance between its two ends, \
                     which have closed on each other across the gap the arc leaves open. Its \
                     surface cannot be projected onto exactly, so [output] boundaries = \"exact\" \
                     will leave violations of it in the exported surface instead of clamping them \
                     (reduce radius, open the bend out, or shorten the span)"
                ));
            }
        }
        warnings
    }

    /// Validate everything that does not depend on the voxelization.
    ///
    /// Grid dependent checks (empty domain, regions selecting no nodes) live in
    /// [`crate::problem::Problem::build`]; the non-fatal ones live in
    /// [`Config::static_warnings`].
    pub fn validate_static(&self) -> Result<()> {
        // First, because everything below reasons about the numbers: a
        // configuration built in memory rather than parsed reaches here without
        // having passed the check in `Config::parse`.
        self.check_finite()?;
        self.engine_name()?;
        self.material()?;
        self.optimization_params()?;
        self.solver_params()?;
        self.growth_params()?;
        self.check_solid()?;
        self.domain_csg()?;
        self.keepout_union()?;
        self.keepin_union()?;

        if self.supports.is_empty() {
            bail!("no [[supports]]: the structure would be unconstrained");
        }
        for (idx, s) in self.supports.iter().enumerate() {
            let what = format!("[[supports]] entry {}", idx + 1);
            s.region.to_shape(&what)?;
            if let Some(dirs) = &s.directions {
                if dirs.is_empty() {
                    bail!("{what}: directions is empty; omit it to constrain all three axes");
                }
                for (a, dir) in dirs.iter().enumerate() {
                    if dirs[..a].contains(dir) {
                        bail!("{what}: direction {dir:?} listed twice");
                    }
                }
            }
        }

        if self.loadcases.is_empty() {
            bail!("no [[loadcases]]: nothing would load the structure");
        }
        for (ci, case) in self.loadcases.iter().enumerate() {
            let what = format!("[[loadcases]] \"{}\" (entry {})", case.name, ci + 1);
            let weight = case.weight.unwrap_or(constants::DEFAULT_LOADCASE_WEIGHT);
            if weight <= 0.0 || weight.is_nan() {
                bail!("{what}: weight must be positive (got {weight})");
            }
            if case.loads.is_empty() {
                bail!("{what}: has no [[loadcases.loads]] entries");
            }
            for (li, load) in case.loads.iter().enumerate() {
                let lwhat = format!("{what}, load {}", li + 1);
                if let Some(region) = load.region() {
                    region.to_shape(&lwhat)?;
                }
                match load {
                    LoadSpec::Force { vector, .. } => {
                        if crate::geometry::length(*vector) == 0.0 {
                            bail!("{lwhat}: force vector is zero");
                        }
                    }
                    LoadSpec::Torque {
                        axis_dir,
                        magnitude_nmm,
                        ..
                    } => {
                        if crate::geometry::length(*axis_dir) == 0.0 {
                            bail!("{lwhat}: torque axis_dir is the zero vector");
                        }
                        if *magnitude_nmm == 0.0 {
                            bail!("{lwhat}: torque magnitude_nmm is zero");
                        }
                    }
                    LoadSpec::Gravity { .. } => {
                        load.gravity_acceleration(&lwhat)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
[project]
name = "unit"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.3
min_feature_mm = 6.0

[output]
stl_path = "out.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [20.0, 10.0, 10.0]

[[supports]]
region = { shape = "box", min = [0.0, 0.0, 0.0], max = [1.0, 10.0, 10.0] }

[[loadcases]]
name = "main"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [20.0, 5.0, 5.0], radius = 3.0 }
vector = [0.0, 0.0, -10.0]
"#;

    #[test]
    fn minimal_config_parses_and_validates() {
        let cfg = Config::parse(MINIMAL).expect("parse");
        cfg.validate_static().expect("validate");
        assert_eq!(cfg.engine_name().unwrap(), constants::DEFAULT_ENGINE);
        let m = cfg.material().unwrap();
        assert_eq!(m.youngs_modulus_mpa, 2300.0);
        let o = cfg.optimization_params().unwrap();
        assert_eq!(o.max_iterations, constants::DEFAULT_MAX_ITERATIONS);
        assert_eq!(o.filter_radius_mm, 3.0);
        let out = cfg.output_params(Path::new("/tmp/cfg")).unwrap();
        assert_eq!(out.iso_level, constants::DEFAULT_ISO_LEVEL);
        assert!(out.stl_path.ends_with("out.stl"));
    }

    /// The iteration budget a configuration that says nothing about it gets.
    ///
    /// Pinned as a literal rather than against the constant, because the whole
    /// point of the value is what it is: a run with no `max_iterations` runs to
    /// convergence (or to the stall criterion) rather than to a wall clock
    /// compromise, and an explicit budget is still obeyed to the iteration.
    #[test]
    fn a_configuration_that_asks_for_no_budget_gets_the_one_that_runs_to_convergence() {
        let default = Config::parse(MINIMAL)
            .expect("parse")
            .optimization_params()
            .expect("params");
        assert_eq!(default.max_iterations, 1000);
        assert_eq!(default.max_iterations, constants::DEFAULT_MAX_ITERATIONS);

        let explicit = Config::parse(&MINIMAL.replace(
            "mass_fraction = 0.3",
            "mass_fraction = 0.3\nmax_iterations = 150",
        ))
        .expect("parse")
        .optimization_params()
        .expect("params");
        assert_eq!(explicit.max_iterations, 150);
    }

    /// The penalization exponent: the classical default when the file says
    /// nothing, the file's own number when it does, and a floor of one either
    /// way - below it intermediate density would be rewarded over solid.
    #[test]
    fn the_penalty_defaults_to_the_classical_exponent_and_is_bounded_below() {
        let params = |text: &str| Config::parse(text).expect("parse").optimization_params();
        let default = params(MINIMAL).expect("params");
        assert_eq!(default.penalty, constants::SIMP_PENALTY);
        assert_eq!(default.penalty, 3.0);

        let pinned = MINIMAL.replace("mass_fraction = 0.3", "mass_fraction = 0.3\npenalty = 4.5");
        assert_eq!(params(&pinned).expect("params").penalty, 4.5);
        // The lower bound itself is legal: a linear interpolation penalizes
        // nothing, which is a degenerate problem rather than an inverted one.
        let linear = MINIMAL.replace("mass_fraction = 0.3", "mass_fraction = 0.3\npenalty = 1.0");
        assert_eq!(
            params(&linear).expect("params").penalty,
            constants::SIMP_PENALTY_MIN
        );

        for bad in ["0.9", "0.0", "-3.0", "nan", "inf"] {
            let text = MINIMAL.replace(
                "mass_fraction = 0.3",
                &format!("mass_fraction = 0.3\npenalty = {bad}"),
            );
            // A non-finite literal is refused by the parse itself, everything
            // else by the resolution; both have to name the key.
            let error = match Config::parse(&text) {
                Ok(config) => config.validate_static().unwrap_err().to_string(),
                Err(error) => error.to_string(),
            };
            assert!(
                error.contains("penalty"),
                "unexpected error for {bad}: {error}"
            );
        }
    }

    /// `[optimization] stiffness_floor`: absent is the constant, present is what
    /// the file says, and the bounds are refused at both ends.
    ///
    /// The key exists because the fixed floor was one: nine decades of stiffness
    /// contrast across a one-sided anchoring's load path exhausted the whole
    /// device iteration budget at a relative residual of 3.5e-6 against a target
    /// of 3e-8, and an honest model had nowhere to say that the void it is
    /// solved through need not be quite that empty.
    #[test]
    fn the_stiffness_floor_defaults_to_the_constant_and_is_bounded_at_both_ends() {
        let with = |body: &str| {
            MINIMAL.replace(
                "mass_fraction = 0.3",
                &format!("mass_fraction = 0.3\n{body}"),
            )
        };
        let params = |text: &str| Config::parse(text).expect("parse").optimization_params();

        // Absent, every configuration written before the key runs at exactly
        // what it always ran at.
        let default = params(MINIMAL).expect("params");
        assert_eq!(default.stiffness_floor, constants::SIMP_EMIN_FRACTION);
        assert_eq!(default.stiffness_floor, 1e-9);

        // Named, it is what runs, and it rides beside the exponent rather than
        // in place of it.
        let named = Config::parse(&with("penalty = 4.0\nstiffness_floor = 1e-6")).expect("parse");
        named.validate_static().expect("validate");
        assert_eq!(
            named.optimization.stiffness_floor,
            Some(1e-6),
            "the schema field"
        );
        let resolved = named.optimization_params().expect("params");
        assert_eq!(resolved.stiffness_floor, 1e-6);
        assert_eq!(resolved.penalty, 4.0);

        // Both bounds are reachable: they are the range, not the outside of it.
        for edge in [
            constants::STIFFNESS_FLOOR_MIN,
            constants::STIFFNESS_FLOOR_MAX,
        ] {
            let text = with(&format!("stiffness_floor = {edge:e}"));
            assert_eq!(params(&text).expect("params").stiffness_floor, edge);
        }

        // And outside them nothing is softened. Zero is refused with the rest:
        // a design cell at no stiffness at all is a singular row in a system
        // nothing pins, which is a run that fails late instead of here.
        for bad in ["1e-13", "1e-2", "0.0", "-1e-9", "nan", "inf", "-inf"] {
            let text = with(&format!("stiffness_floor = {bad}"));
            // A non-finite literal is stopped by the finite pass at parse, the
            // rest by the range check at resolution; either way the message
            // names the section and the key and nothing reaches a solver.
            let error = match Config::parse(&text) {
                Ok(config) => config
                    .optimization_params()
                    .expect_err(&format!("{bad} was accepted"))
                    .to_string(),
                Err(error) => error.to_string(),
            };
            assert!(error.contains("stiffness_floor"), "for {bad}: {error}");
            assert!(error.contains("[optimization]"), "for {bad}: {error}");
        }
        // The bounds are named, so the message says what the range is.
        let error = params(&with("stiffness_floor = 1e-2"))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!("{:e}", constants::STIFFNESS_FLOOR_MIN))
                && error.contains(&format!("{:e}", constants::STIFFNESS_FLOOR_MAX)),
            "the message has to name the bounds: {error}"
        );
    }

    /// `[optimization.local_volume]`: every key optional, the defaults derived
    /// from the same `[optimization]` numbers the rest of the run is, and both
    /// bounds enforced against them.
    #[test]
    fn the_local_volume_table_defaults_from_the_optimization_it_sits_in() {
        let text = |extra: &str| {
            MINIMAL.replace(
                "min_feature_mm = 6.0",
                &format!(
                    "min_feature_mm = 6.0\nupdate = \"mma\"\n\n[optimization.local_volume]\n{extra}"
                ),
            )
        };
        let params = |extra: &str| {
            Config::parse(&text(extra))
                .expect("parse")
                .optimization_params()
        };

        let default = params("").expect("params").local_volume.expect("a cap");
        assert_eq!(default.max_fraction, constants::LOCAL_VOLUME_MAX_FRACTION);
        // Three times the density filter radius, which is min_feature_mm / 2.
        assert_eq!(
            default.radius_mm,
            constants::LOCAL_VOLUME_RADIUS_FACTOR * 3.0
        );

        let pinned = params("max_fraction = 0.45\nradius_mm = 12.0")
            .expect("params")
            .local_volume
            .expect("a cap");
        assert_eq!(pinned.max_fraction, 0.45);
        assert_eq!(pinned.radius_mm, 12.0);

        // The cap has to leave room above the global target - the mean of the
        // local fractions is that target - and has to be measured over more than
        // one filter radius.
        let infeasible = params("max_fraction = 0.3").unwrap_err().to_string();
        assert!(infeasible.contains("mass_fraction"), "{infeasible}");
        for bad in ["0.0", "1.0", "-0.5", "2.0"] {
            let error = params(&format!("max_fraction = {bad}"))
                .unwrap_err()
                .to_string();
            assert!(error.contains("max_fraction"), "for {bad}: {error}");
        }
        // The filter radius itself, and anything below it, is refused.
        for bad in ["3.0", "1.0", "0.0", "-4.0"] {
            let error = params(&format!("radius_mm = {bad}"))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("radius_mm") && error.contains("density filter radius"),
                "for {bad}: {error}"
            );
        }
        // And a non-finite one is refused by the parse, like every other number.
        for bad in ["nan", "inf"] {
            let error = Config::parse(&text(&format!("radius_mm = {bad}")))
                .unwrap_err()
                .to_string();
            assert!(error.contains("radius_mm"), "for {bad}: {error}");
        }
    }

    /// The scheme gate: a second constraint needs a second multiplier, so the
    /// cap is refused under the optimality criteria step - by name, with the
    /// remedy in the message.
    #[test]
    fn a_local_volume_cap_is_refused_under_the_optimality_criteria_step() {
        let with = |extra: &str| {
            MINIMAL.replace(
                "min_feature_mm = 6.0",
                &format!("min_feature_mm = 6.0{extra}\n\n[optimization.local_volume]\n"),
            )
        };
        // The default scheme, and the same scheme written out.
        for text in [with(""), with("\nupdate = \"oc\"")] {
            let config = Config::parse(&text).expect("parse");
            let error = config.validate_static().unwrap_err().to_string();
            assert!(
                error.contains("[optimization.local_volume]") && error.contains("update = \"mma\""),
                "the remedy has to be named: {error}"
            );
        }
        // And accepted under the one that can price it.
        let good = Config::parse(&with("\nupdate = \"mma\"")).expect("parse");
        good.validate_static().expect("mma takes the cap");
    }

    /// The engine gate, on the pattern the overhang and wireframe tables set:
    /// the growth engine has no density field to cap, and says so rather than
    /// sending the user to an update scheme it does not have.
    #[test]
    fn a_local_volume_cap_is_refused_by_the_growth_engine() {
        let text = MINIMAL
            .replace("name = \"unit\"", "name = \"unit\"\nengine = \"growth\"")
            .replace(
                "min_feature_mm = 6.0",
                "min_feature_mm = 6.0\n\n[optimization.local_volume]\n",
            );
        let error = Config::parse(&text)
            .expect("parse")
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("[optimization.local_volume]") && error.contains("growth"),
            "{error}"
        );
        assert!(
            !error.contains("update = \"mma\""),
            "the growth engine has no update scheme to switch to: {error}"
        );
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let text = MINIMAL.replace(
            "mass_fraction = 0.3",
            "mass_fraction = 0.3\nmas_fraction = 0.4",
        );
        let err = Config::parse(&text).unwrap_err().to_string();
        assert!(err.contains("mas_fraction"), "unexpected error: {err}");
    }

    #[test]
    fn unknown_keys_inside_a_tagged_shape_are_rejected() {
        let text = MINIMAL.replace(
            "region = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 3.0 }",
            "region = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 3.0, rad = 1.0 }",
        );
        let err = Config::parse(&text).unwrap_err().to_string();
        assert!(err.contains("rad"), "unexpected error: {err}");
    }

    #[test]
    fn unknown_shape_is_rejected() {
        let text = MINIMAL.replace("shape = \"box\"", "shape = \"pyramid\"");
        let err = Config::parse(&text).unwrap_err().to_string();
        assert!(err.contains("pyramid"), "unexpected error: {err}");
    }

    #[test]
    fn engine_accepts_either_placement_but_not_both() {
        let top = format!("engine = \"simp\"\n{MINIMAL}");
        assert_eq!(Config::parse(&top).unwrap().engine_name().unwrap(), "simp");

        let nested = MINIMAL.replace("name = \"unit\"", "name = \"unit\"\nengine = \"simp\"");
        assert_eq!(
            Config::parse(&nested).unwrap().engine_name().unwrap(),
            "simp"
        );

        let both = format!(
            "engine = \"simp\"\n{}",
            MINIMAL.replace("name = \"unit\"", "name = \"unit\"\nengine = \"simp\"")
        );
        assert!(Config::parse(&both).unwrap().engine_name().is_err());
    }

    #[test]
    fn mass_fraction_bounds_are_enforced() {
        for bad in ["0.0", "1.0", "-0.2", "1.5"] {
            let text = MINIMAL.replace("mass_fraction = 0.3", &format!("mass_fraction = {bad}"));
            let cfg = Config::parse(&text).unwrap();
            let err = cfg.validate_static().unwrap_err().to_string();
            assert!(err.contains("mass_fraction"), "unexpected error: {err}");
        }
    }

    #[test]
    fn resolution_requires_exactly_one_key() {
        let both = MINIMAL.replace(
            "voxel_size_mm = 2.0",
            "voxel_size_mm = 2.0\ntarget_cells = 1000",
        );
        let err = Config::parse(&both).unwrap().voxel_size(1.0).unwrap_err();
        assert!(err.to_string().contains("exactly one"));

        let neither = MINIMAL.replace("voxel_size_mm = 2.0", "");
        let err = Config::parse(&neither)
            .unwrap()
            .voxel_size(1.0)
            .unwrap_err();
        assert!(err.to_string().contains("exactly one"));
    }

    #[test]
    fn target_cells_derives_voxel_size() {
        let text = MINIMAL.replace("voxel_size_mm = 2.0", "target_cells = 1000");
        let cfg = Config::parse(&text).unwrap();
        // 1000 mm^3 over 1000 cells is a 1 mm voxel.
        assert!((cfg.voxel_size(1000.0).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn missing_supports_and_loads_are_rejected() {
        let no_support = MINIMAL.replace(
            "[[supports]]\nregion = { shape = \"box\", min = [0.0, 0.0, 0.0], max = [1.0, 10.0, 10.0] }",
            "",
        );
        let err = Config::parse(&no_support)
            .unwrap()
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(err.contains("supports"), "unexpected error: {err}");

        let no_loads = MINIMAL.split("[[loadcases]]").next().unwrap().to_string();
        let err = Config::parse(&no_loads)
            .unwrap()
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(err.contains("loadcases"), "unexpected error: {err}");
    }

    #[test]
    fn unknown_engine_is_rejected_by_the_registry() {
        let text = format!("engine = \"grow\"\n{MINIMAL}");
        let cfg = Config::parse(&text).unwrap();
        assert_eq!(cfg.engine_name().unwrap(), "grow");
        assert!(crate::engine::create(cfg.engine_name().unwrap()).is_err());
    }

    #[test]
    fn zero_force_vector_is_rejected() {
        let text = MINIMAL.replace("vector = [0.0, 0.0, -10.0]", "vector = [0.0, 0.0, 0.0]");
        let err = Config::parse(&text)
            .unwrap()
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(err.contains("zero"), "unexpected error: {err}");
    }

    /// `rotation_deg` is optional everywhere a box is legal, absent means an
    /// axis aligned box, and it is the box's *centre* that turns.
    #[test]
    fn a_box_rotation_is_optional_and_reaches_the_geometry() {
        let plain = Config::parse(MINIMAL).expect("parse");
        assert_eq!(plain.domain[0].shape_spec().rotation_deg(), None);
        let ShapeSpec::Box { rotation_deg, .. } = plain.domain[0].shape_spec() else {
            panic!("a box");
        };
        assert!(rotation_deg.is_none(), "an absent key stays absent");
        // Absent is the same geometry as none at all.
        let shape = plain.domain[0].shape_spec().to_shape("test").unwrap();
        assert_eq!(
            shape,
            Shape::axis_aligned_box([0.0, 0.0, 0.0], [20.0, 10.0, 10.0])
        );

        let text = MINIMAL.replace(
            "max = [20.0, 10.0, 10.0]",
            "max = [20.0, 10.0, 10.0]\nrotation_deg = [0.0, 0.0, 45.0]",
        );
        let turned = Config::parse(&text).expect("parse");
        turned.validate_static().expect("validate");
        assert_eq!(
            turned.domain[0].shape_spec().rotation_deg(),
            Some([0.0, 0.0, 45.0])
        );
        let shape = turned.domain[0].shape_spec().to_shape("test").unwrap();
        let Shape::Box { rotation_deg, .. } = shape else {
            panic!("a box");
        };
        assert_eq!(rotation_deg, [0.0, 0.0, 45.0]);
        // The centre is what it turns about, so the bounds grow either side of
        // it rather than sweeping the box off its own corner.
        let bounds = shape.bounds();
        assert!((0.5 * (bounds.min[0] + bounds.max[0]) - 10.0).abs() < 1e-9);
        assert!((0.5 * (bounds.min[1] + bounds.max[1]) - 5.0).abs() < 1e-9);

        // Every list a box can appear in takes it, and none of the other kinds
        // do: `rotation_deg` on a sphere is an unknown key.
        for replaced in [
            (
                "region = { shape = \"box\", min = [0.0, 0.0, 0.0], max = [1.0, 10.0, 10.0] }",
                "region = { shape = \"box\", min = [0.0, 0.0, 0.0], max = [1.0, 10.0, 10.0], \
                 rotation_deg = [10.0, 0.0, 0.0] }",
            ),
            (
                "[[loadcases]]",
                "[[keepout]]\nshape = \"box\"\nmin = [2.0, 2.0, 2.0]\nmax = [4.0, 4.0, 4.0]\n\
                 rotation_deg = [0.0, 30.0, 0.0]\n\n[[keepin]]\nshape = \"box\"\n\
                 min = [12.0, 2.0, 2.0]\nmax = [14.0, 4.0, 4.0]\nrotation_deg = [5.0, 5.0, 5.0]\n\n\
                 [[loadcases]]",
            ),
            (
                "region = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 3.0 }",
                "region = { shape = \"box\", min = [18.0, 3.0, 3.0], max = [20.0, 7.0, 7.0], \
                 rotation_deg = [0.0, 0.0, 20.0] }",
            ),
        ] {
            let text = MINIMAL.replace(replaced.0, replaced.1);
            let config = Config::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", replaced.1));
            config.validate_static().expect("validate");
        }
        let text = MINIMAL.replace(
            "region = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 3.0 }",
            "region = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 3.0, \
             rotation_deg = [0.0, 0.0, 20.0] }",
        );
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(error.contains("rotation_deg"), "unexpected error: {error}");
    }

    /// The universal finiteness pass covers the new key too: a rotation matrix
    /// built from a `nan` turns every sample of the shape into one.
    #[test]
    fn a_rotation_that_is_not_a_number_is_rejected_by_the_finiteness_pass() {
        for value in ["nan", "inf", "-inf"] {
            let text = MINIMAL.replace(
                "max = [20.0, 10.0, 10.0]",
                &format!("max = [20.0, 10.0, 10.0]\nrotation_deg = [0.0, {value}, 0.0]"),
            );
            let error = Config::parse(&text).unwrap_err().to_string();
            assert!(
                error.contains("rotation_deg[1]"),
                "{value}: unexpected error: {error}"
            );
        }
        // Any magnitude is legal: it is an angle, not a bound.
        let text = MINIMAL.replace(
            "max = [20.0, 10.0, 10.0]",
            "max = [20.0, 10.0, 10.0]\nrotation_deg = [720.0, -900.0, 1e6]",
        );
        Config::parse(&text)
            .expect("parse")
            .validate_static()
            .expect("validate");
    }

    /// An ellipsoid is legal wherever a shape is, carries the box's optional
    /// rotation with the box's semantics, and reaches the geometry as itself.
    #[test]
    fn an_ellipsoid_is_legal_in_every_shape_position() {
        const DOMAIN: &str = "shape = \"box\"\nmin = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]";
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"ellipsoid\"\ncenter = [10.0, 5.0, 5.0]\nradii = [10.0, 5.0, 5.0]",
        );
        let config = Config::parse(&text).expect("parse");
        config.validate_static().expect("validate");
        let spec = config.domain[0].shape_spec();
        assert_eq!(spec.kind(), "ellipsoid");
        assert_eq!(spec.rotation_deg(), None, "an absent key stays absent");
        assert_eq!(
            spec.to_shape("test").expect("a shape"),
            Shape::Ellipsoid {
                center: [10.0, 5.0, 5.0],
                radii: [10.0, 5.0, 5.0],
                rotation_deg: [0.0; 3],
            }
        );

        // Turned, with the box's own semantics: about the centre, extrinsic
        // XYZ, and the bounds grow either side of that centre.
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"ellipsoid\"\ncenter = [10.0, 5.0, 5.0]\nradii = [10.0, 5.0, 5.0]\n\
             rotation_deg = [0.0, 0.0, 45.0]",
        );
        let turned = Config::parse(&text).expect("parse");
        turned.validate_static().expect("validate");
        let spec = turned.domain[0].shape_spec();
        assert_eq!(spec.rotation_deg(), Some([0.0, 0.0, 45.0]));
        let shape = spec.to_shape("test").expect("a shape");
        let Shape::Ellipsoid { rotation_deg, .. } = shape else {
            panic!("an ellipsoid");
        };
        assert_eq!(rotation_deg, [0.0, 0.0, 45.0]);
        let bounds = shape.bounds();
        assert!((0.5 * (bounds.min[0] + bounds.max[0]) - 10.0).abs() < 1e-9);
        // sqrt(10^2 cos^2 45 + 5^2 sin^2 45) either side on x and y.
        let reach = (100.0_f64 * 0.5 + 25.0 * 0.5).sqrt();
        assert!((bounds.max[0] - (10.0 + reach)).abs() < 1e-9, "{bounds:?}");
        assert!((bounds.max[2] - 10.0).abs() < 1e-9, "z is untouched");

        // Every other list an ellipsoid can appear in.
        for replaced in [
            (
                "region = { shape = \"box\", min = [0.0, 0.0, 0.0], max = [1.0, 10.0, 10.0] }",
                "region = { shape = \"ellipsoid\", center = [0.5, 5.0, 5.0], \
                 radii = [0.5, 5.0, 5.0], rotation_deg = [10.0, 0.0, 0.0] }",
            ),
            (
                "[[loadcases]]",
                "[[keepout]]\nshape = \"ellipsoid\"\ncenter = [3.0, 3.0, 3.0]\n\
                 radii = [1.0, 2.0, 3.0]\nrotation_deg = [0.0, 30.0, 0.0]\n\n\
                 [[keepin]]\nshape = \"ellipsoid\"\ncenter = [13.0, 3.0, 3.0]\n\
                 radii = [1.0, 1.0, 1.0]\n\n[[loadcases]]",
            ),
            (
                "region = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 3.0 }",
                "region = { shape = \"ellipsoid\", center = [20.0, 5.0, 5.0], \
                 radii = [3.0, 4.0, 5.0] }",
            ),
        ] {
            let text = MINIMAL.replace(replaced.0, replaced.1);
            let config = Config::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", replaced.1));
            config.validate_static().expect("validate");
        }
    }

    /// A tube is legal wherever a shape is, its bend is optional, and it
    /// reaches the geometry as itself - straight or curved.
    #[test]
    fn a_tube_is_legal_in_every_shape_position() {
        const DOMAIN: &str = "shape = \"box\"\nmin = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]";
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"tube\"\np1 = [2.0, 5.0, 5.0]\np2 = [18.0, 5.0, 5.0]\nradius = 4.0",
        );
        let config = Config::parse(&text).expect("parse");
        config.validate_static().expect("validate");
        let spec = config.domain[0].shape_spec();
        assert_eq!(spec.kind(), "tube");
        assert_eq!(spec.rotation_deg(), None, "a tube carries no rotation");
        assert_eq!(
            spec.to_shape("test").expect("a shape"),
            Shape::Tube {
                p1: [2.0, 5.0, 5.0],
                p2: [18.0, 5.0, 5.0],
                bend: None,
                radius: 4.0,
            }
        );
        // Straight, the bounds are the segment grown by the radius: rounded
        // ends, so the tube reaches a radius past each of its own points.
        let bounds = spec.to_shape("test").expect("a shape").bounds();
        assert_eq!(bounds.min, [-2.0, 1.0, 1.0]);
        assert_eq!(bounds.max, [22.0, 9.0, 9.0]);

        // Bent, the key is carried through and the shape is the arc.
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"tube\"\np1 = [2.0, 5.0, 5.0]\np2 = [18.0, 5.0, 5.0]\n\
             bend = [10.0, 5.0, 9.0]\nradius = 4.0",
        );
        let bent = Config::parse(&text).expect("parse");
        bent.validate_static().expect("validate");
        let shape = bent.domain[0]
            .shape_spec()
            .to_shape("test")
            .expect("a shape");
        let Shape::Tube { bend, .. } = shape else {
            panic!("a tube");
        };
        assert_eq!(bend, Some([10.0, 5.0, 9.0]));
        // Hand computed: a chord of 16 with a sagitta of 4 is a circle of
        // radius (8^2 + 4^2) / (2 * 4) = 10, centred 1 mm below the ends, so
        // the arc tops out at the bend point itself and the tube a radius above
        // it. The bend is what makes it taller than the straight one.
        assert_eq!(shape.bounds().min, [-2.0, 1.0, 1.0]);
        assert_eq!(shape.bounds().max, [22.0, 9.0, 13.0]);
        assert!(shape.contains([10.0, 5.0, 9.0]));
        assert!(!shape.contains([10.0, 5.0, 3.0]), "the arc bends upwards");
        assert!(shape.contains([10.0, 5.0, 12.9]) && !shape.contains([10.0, 5.0, 13.1]));

        // A bend on the line through the two ends is legal and is the straight
        // tube; it is not an error, and the file keeps saying it.
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"tube\"\np1 = [2.0, 5.0, 5.0]\np2 = [18.0, 5.0, 5.0]\n\
             bend = [10.0, 5.0, 5.0]\nradius = 4.0",
        );
        let flat = Config::parse(&text).expect("parse");
        flat.validate_static().expect("validate");
        let shape = flat.domain[0]
            .shape_spec()
            .to_shape("test")
            .expect("a shape");
        assert_eq!(shape.bounds().min, [-2.0, 1.0, 1.0]);
        assert_eq!(shape.bounds().max, [22.0, 9.0, 9.0]);

        // Every other list a tube can appear in.
        for replaced in [
            (
                "region = { shape = \"box\", min = [0.0, 0.0, 0.0], max = [1.0, 10.0, 10.0] }",
                "region = { shape = \"tube\", p1 = [0.0, 2.0, 5.0], p2 = [0.0, 8.0, 5.0], \
                 radius = 2.0 }",
            ),
            (
                "[[loadcases]]",
                "[[keepout]]\nshape = \"tube\"\np1 = [4.0, 3.0, 3.0]\np2 = [12.0, 3.0, 3.0]\n\
                 bend = [8.0, 6.0, 3.0]\nradius = 1.0\n\n\
                 [[keepin]]\nshape = \"tube\"\np1 = [14.0, 3.0, 3.0]\np2 = [16.0, 3.0, 3.0]\n\
                 radius = 1.0\n\n[[loadcases]]",
            ),
            (
                "region = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 3.0 }",
                "region = { shape = \"tube\", p1 = [20.0, 3.0, 5.0], p2 = [20.0, 7.0, 5.0], \
                 bend = [20.0, 5.0, 7.0], radius = 2.0 }",
            ),
        ] {
            let text = MINIMAL.replace(replaced.0, replaced.1);
            let config = Config::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", replaced.1));
            config.validate_static().expect("validate");
        }
    }

    /// A tube's radius is the one measurement of it that can make it nothing,
    /// so it is the one that is named; and a `nan` anywhere in it, the optional
    /// bend included, is caught by the finiteness pass.
    #[test]
    fn a_tube_with_a_radius_that_is_not_positive_is_rejected() {
        const DOMAIN: &str = "shape = \"box\"\nmin = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]";
        for radius in ["0.0", "-3.0"] {
            let text = MINIMAL.replace(
                DOMAIN,
                &format!(
                    "shape = \"tube\"\np1 = [2.0, 5.0, 5.0]\np2 = [18.0, 5.0, 5.0]\n\
                     radius = {radius}"
                ),
            );
            let error = Config::parse(&text)
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("tube radius must be positive"),
                "{radius}: unexpected error: {error}"
            );
        }
        // A `nan` anywhere in it, the optional bend included, is caught by the
        // finiteness pass and named.
        const BENT: &str = "shape = \"tube\"\np1 = [2.0, 5.0, 5.0]\np2 = [18.0, 5.0, 5.0]\n\
                            bend = [10.0, 5.0, 9.0]\nradius = 4.0";
        for (key, spelled, named) in [
            ("p1", "p1 = [2.0, nan, 5.0]", "p1[1]"),
            ("bend", "bend = [10.0, nan, 9.0]", "bend[1]"),
            ("radius", "radius = inf", "radius"),
        ] {
            let broken = BENT
                .lines()
                .map(|line| {
                    if line.starts_with(&format!("{key} ")) {
                        spelled
                    } else {
                        line
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            let error = Config::parse(&MINIMAL.replace(DOMAIN, &broken))
                .unwrap_err()
                .to_string();
            assert!(error.contains(named), "{key}: unexpected error: {error}");
        }
        // A tube of no length is a ball rather than a degenerate shape, so it
        // is not rejected: the radius is what it is made of.
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"tube\"\np1 = [10.0, 5.0, 5.0]\np2 = [10.0, 5.0, 5.0]\nradius = 5.0",
        );
        let config = Config::parse(&text).expect("parse");
        config.validate_static().expect("validate");
        let shape = config.domain[0]
            .shape_spec()
            .to_shape("test")
            .expect("a shape");
        assert!((shape.min_extent() - 5.0).abs() < 1e-12);
        assert!(shape.contains([10.0, 5.0, 9.0]) && !shape.contains([10.0, 5.0, 10.1]));
    }

    /// A cone is legal wherever a shape is, carries no rotation key, and
    /// reaches the geometry as itself - frustum, cylinder or true cone.
    #[test]
    fn a_cone_is_legal_in_every_shape_position() {
        const DOMAIN: &str = "shape = \"box\"\nmin = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]";
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"cone\"\np1 = [2.0, 5.0, 5.0]\np2 = [18.0, 5.0, 5.0]\n\
             radius1 = 4.0\nradius2 = 2.0",
        );
        let config = Config::parse(&text).expect("parse");
        config.validate_static().expect("validate");
        let spec = config.domain[0].shape_spec();
        assert_eq!(spec.kind(), "cone");
        assert_eq!(spec.rotation_deg(), None, "a cone carries no rotation");
        assert_eq!(
            spec.to_shape("test").expect("a shape"),
            Shape::Cone {
                p1: [2.0, 5.0, 5.0],
                p2: [18.0, 5.0, 5.0],
                radius1: 4.0,
                radius2: 2.0,
            }
        );
        // The bounds are the hull of the two cap discs: flat caps, so it
        // reaches no further along its axis than its own points.
        let shape = spec.to_shape("test").expect("a shape");
        assert_eq!(shape.bounds().min, [2.0, 1.0, 1.0]);
        assert_eq!(shape.bounds().max, [18.0, 9.0, 9.0]);

        // The apex case, which is a legal shape rather than a degenerate one.
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"cone\"\np1 = [2.0, 5.0, 5.0]\np2 = [18.0, 5.0, 5.0]\n\
             radius1 = 4.0\nradius2 = 0.0",
        );
        let apex = Config::parse(&text).expect("parse");
        apex.validate_static().expect("validate");
        let shape = apex.domain[0]
            .shape_spec()
            .to_shape("test")
            .expect("a shape");
        assert!(
            (shape.min_extent() - 4.0).abs() < 1e-12,
            "the wide end is what it is measured by"
        );
        assert!(shape.contains([17.9, 5.0, 5.0]) && !shape.contains([18.1, 5.0, 5.0]));

        // Every other list a cone can appear in.
        for replaced in [
            (
                "region = { shape = \"box\", min = [0.0, 0.0, 0.0], max = [1.0, 10.0, 10.0] }",
                "region = { shape = \"cone\", p1 = [0.0, 5.0, 5.0], p2 = [1.0, 5.0, 5.0], \
                 radius1 = 4.0, radius2 = 2.0 }",
            ),
            (
                "[[loadcases]]",
                "[[keepout]]\nshape = \"cone\"\np1 = [4.0, 3.0, 3.0]\np2 = [12.0, 3.0, 3.0]\n\
                 radius1 = 2.0\nradius2 = 0.0\n\n\
                 [[keepin]]\nshape = \"cone\"\np1 = [14.0, 3.0, 3.0]\np2 = [16.0, 3.0, 3.0]\n\
                 radius1 = 1.0\nradius2 = 1.0\n\n[[loadcases]]",
            ),
            (
                "region = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 3.0 }",
                "region = { shape = \"cone\", p1 = [20.0, 5.0, 5.0], p2 = [18.0, 5.0, 5.0], \
                 radius1 = 3.0, radius2 = 1.0 }",
            ),
        ] {
            let text = MINIMAL.replace(replaced.0, replaced.1);
            let config = Config::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", replaced.1));
            config.validate_static().expect("validate");
        }
    }

    /// A cone's two radii are not the same measurement and are not refused the
    /// same way: the wide end has to be a real disc, and the narrow one is
    /// allowed to be the apex.
    #[test]
    fn a_cone_with_a_radius_that_is_not_usable_is_rejected() {
        const DOMAIN: &str = "shape = \"box\"\nmin = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]";
        let cone = |radius1: &str, radius2: &str| {
            MINIMAL.replace(
                DOMAIN,
                &format!(
                    "shape = \"cone\"\np1 = [2.0, 5.0, 5.0]\np2 = [18.0, 5.0, 5.0]\n\
                     radius1 = {radius1}\nradius2 = {radius2}"
                ),
            )
        };
        for radius1 in ["0.0", "-3.0"] {
            let error = Config::parse(&cone(radius1, "2.0"))
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("cone radius1 must be positive"),
                "{radius1}: unexpected error: {error}"
            );
        }
        let error = Config::parse(&cone("4.0", "-1.0"))
            .expect("parse")
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(error.contains("cone radius2"), "unexpected error: {error}");

        // A cone of no length is a disc rather than a solid, and the shared
        // degeneracy guard is what says so - as it does for a cylinder.
        let flat = MINIMAL.replace(
            DOMAIN,
            "shape = \"cone\"\np1 = [10.0, 5.0, 5.0]\np2 = [10.0, 5.0, 5.0]\n\
             radius1 = 4.0\nradius2 = 2.0",
        );
        let error = Config::parse(&flat)
            .expect("parse")
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(error.contains("degenerate"), "unexpected error: {error}");

        // A `nan` anywhere in it is caught by the finiteness pass and named.
        for (key, spelled, named) in [
            ("p2", "p2 = [18.0, nan, 5.0]", "p2[1]"),
            ("radius1", "radius1 = inf", "radius1"),
            ("radius2", "radius2 = nan", "radius2"),
        ] {
            let broken = cone("4.0", "2.0")
                .lines()
                .map(|line| {
                    if line.starts_with(&format!("{key} ")) {
                        spelled
                    } else {
                        line
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            let error = Config::parse(&broken).unwrap_err().to_string();
            assert!(error.contains(named), "{key}: unexpected error: {error}");
        }
    }

    /// A triangle is legal wherever a shape is, carries no rotation key - its
    /// three vertices are what point it - and reaches the geometry as itself.
    #[test]
    fn a_triangle_is_legal_in_every_shape_position() {
        const DOMAIN: &str = "shape = \"box\"\nmin = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]";
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"triangle\"\na = [2.0, 2.0, 5.0]\nb = [18.0, 2.0, 5.0]\n\
             c = [10.0, 8.0, 5.0]\nthickness = 4.0",
        );
        let config = Config::parse(&text).expect("parse");
        config.validate_static().expect("validate");
        let spec = config.domain[0].shape_spec();
        assert_eq!(spec.kind(), "triangle");
        assert_eq!(spec.rotation_deg(), None, "a triangle carries no rotation");
        assert_eq!(
            spec.to_shape("test").expect("a shape"),
            Shape::Triangle {
                a: [2.0, 2.0, 5.0],
                b: [18.0, 2.0, 5.0],
                c: [10.0, 8.0, 5.0],
                thickness: 4.0,
            }
        );
        // The bounds are its six corners: the triangle across, and half the
        // thickness either side of the plane it lies in.
        let shape = spec.to_shape("test").expect("a shape");
        assert_eq!(shape.bounds().min, [2.0, 2.0, 3.0]);
        assert_eq!(shape.bounds().max, [18.0, 8.0, 7.0]);
        assert!(shape.contains([10.0, 3.0, 5.0]) && !shape.contains([10.0, 1.0, 5.0]));

        // Every other list a triangle can appear in.
        for replaced in [
            (
                "region = { shape = \"box\", min = [0.0, 0.0, 0.0], max = [1.0, 10.0, 10.0] }",
                "region = { shape = \"triangle\", a = [0.0, 2.0, 2.0], b = [0.0, 8.0, 2.0], \
                 c = [0.0, 5.0, 8.0], thickness = 1.0 }",
            ),
            (
                "[[loadcases]]",
                "[[keepout]]\nshape = \"triangle\"\na = [4.0, 3.0, 3.0]\nb = [12.0, 3.0, 3.0]\n\
                 c = [8.0, 7.0, 3.0]\nthickness = 2.0\n\n\
                 [[keepin]]\nshape = \"triangle\"\na = [14.0, 3.0, 3.0]\nb = [16.0, 3.0, 3.0]\n\
                 c = [15.0, 5.0, 3.0]\nthickness = 1.0\n\n[[loadcases]]",
            ),
            (
                "region = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 3.0 }",
                "region = { shape = \"triangle\", a = [19.0, 3.0, 5.0], b = [19.0, 7.0, 5.0], \
                 c = [20.0, 5.0, 8.0], thickness = 2.0 }",
            ),
        ] {
            let text = MINIMAL.replace(replaced.0, replaced.1);
            let config = Config::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", replaced.1));
            config.validate_static().expect("validate");
        }
    }

    /// Three points on one line are refused outright, however far apart they
    /// are: unlike a tube's bend landing on the line through its ends, which is
    /// still a tube, a triangle of no area encloses nothing at any thickness.
    #[test]
    fn a_triangle_with_no_area_or_no_thickness_is_rejected() {
        const DOMAIN: &str = "shape = \"box\"\nmin = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]";
        let triangle = |c: &str, thickness: &str| {
            MINIMAL.replace(
                DOMAIN,
                &format!(
                    "shape = \"triangle\"\na = [2.0, 2.0, 5.0]\nb = [18.0, 2.0, 5.0]\n\
                     c = {c}\nthickness = {thickness}"
                ),
            )
        };
        // Collinear, spread right across the domain, and on top of one of the
        // other two: every one of them has no area.
        for c in ["[10.0, 2.0, 5.0]", "[100.0, 2.0, 5.0]", "[2.0, 2.0, 5.0]"] {
            let error = Config::parse(&triangle(c, "4.0"))
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("triangle has no area"),
                "{c}: unexpected error: {error}"
            );
        }
        for thickness in ["0.0", "-2.0"] {
            let error = Config::parse(&triangle("[10.0, 8.0, 5.0]", thickness))
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("triangle thickness must be positive"),
                "{thickness}: unexpected error: {error}"
            );
        }
        // A sliver with real area is legal until the ball that fits inside it
        // is what the degeneracy guard measures.
        let thin = Config::parse(&triangle("[10.0, 2.001, 5.0]", "4.0")).expect("parse");
        thin.validate_static()
            .expect("a thin triangle is still a triangle");
        // A `nan` anywhere in it is caught by the finiteness pass and named.
        for (key, spelled, named) in [
            ("a", "a = [2.0, nan, 5.0]", "a[1]"),
            ("c", "c = [10.0, 8.0, inf]", "c[2]"),
            ("thickness", "thickness = nan", "thickness"),
        ] {
            let broken = triangle("[10.0, 8.0, 5.0]", "4.0")
                .lines()
                .map(|line| {
                    if line.starts_with(&format!("{key} ")) {
                        spelled
                    } else {
                        line
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            let error = Config::parse(&broken).unwrap_err().to_string();
            assert!(error.contains(named), "{key}: unexpected error: {error}");
        }
    }

    /// A radius that is not a positive number is a shape nothing can sample:
    /// the field divides by every one of the three.
    #[test]
    fn an_ellipsoid_with_a_radius_that_is_not_positive_is_rejected() {
        const DOMAIN: &str = "shape = \"box\"\nmin = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]";
        for (radii, expected) in [
            ("[0.0, 5.0, 5.0]", "radii[0]"),
            ("[10.0, -5.0, 5.0]", "radii[1]"),
            ("[10.0, 5.0, -0.0]", "radii[2]"),
        ] {
            let text = MINIMAL.replace(
                DOMAIN,
                &format!("shape = \"ellipsoid\"\ncenter = [10.0, 5.0, 5.0]\nradii = {radii}"),
            );
            let error = Config::parse(&text)
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(expected),
                "{radii}: unexpected error: {error}"
            );
            assert!(error.contains("positive"), "{radii}: {error}");
        }
        // And the finiteness pass names the component before anything divides.
        for value in ["nan", "inf", "-inf"] {
            let text = MINIMAL.replace(
                DOMAIN,
                &format!(
                    "shape = \"ellipsoid\"\ncenter = [10.0, 5.0, 5.0]\nradii = [10.0, {value}, 5.0]"
                ),
            );
            let error = Config::parse(&text).unwrap_err().to_string();
            assert!(error.contains("radii[1]"), "{value}: {error}");
        }
        // A rotation that is not a number is caught by the same pass.
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"ellipsoid\"\ncenter = [10.0, 5.0, 5.0]\nradii = [10.0, 5.0, 5.0]\n\
             rotation_deg = [0.0, nan, 0.0]",
        );
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(error.contains("rotation_deg[1]"), "unexpected: {error}");
        // A `radius` key is not an ellipsoid's, and an unknown key fails loudly.
        let text = MINIMAL.replace(
            DOMAIN,
            "shape = \"ellipsoid\"\ncenter = [10.0, 5.0, 5.0]\nradius = 5.0",
        );
        assert!(Config::parse(&text).is_err());
    }

    #[test]
    fn degenerate_domain_box_is_rejected() {
        let text = MINIMAL.replace("max = [20.0, 10.0, 10.0]", "max = [0.0, 10.0, 10.0]");
        let err = Config::parse(&text)
            .unwrap()
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(err.contains("extent"), "unexpected error: {err}");
    }

    #[test]
    fn material_preset_and_custom_values_cannot_be_mixed() {
        let text = MINIMAL.replace(
            "preset = \"pla\"",
            "preset = \"pla\"\nyoungs_modulus_mpa = 1000.0",
        );
        let err = Config::parse(&text)
            .unwrap()
            .material()
            .unwrap_err()
            .to_string();
        assert!(err.contains("not both"), "unexpected error: {err}");
    }

    #[test]
    fn the_overhang_table_is_optional_and_resolves_to_a_build_direction() {
        let off = Config::parse(MINIMAL).unwrap();
        assert!(off.optimization_params().unwrap().overhang.is_none());

        for (spelling, expected, axis, positive) in [
            ("x+", BuildDirection::XPlus, 0, true),
            ("x-", BuildDirection::XMinus, 0, false),
            ("y+", BuildDirection::YPlus, 1, true),
            ("y-", BuildDirection::YMinus, 1, false),
            ("z+", BuildDirection::ZPlus, 2, true),
            ("z-", BuildDirection::ZMinus, 2, false),
        ] {
            let text = MINIMAL.replace(
                "min_feature_mm = 6.0",
                &format!(
                    "min_feature_mm = 6.0\n\n[optimization.overhang]\nbuild_direction = \"{spelling}\""
                ),
            );
            let cfg = Config::parse(&text).unwrap_or_else(|e| panic!("{spelling}: {e}"));
            cfg.validate_static().expect("validate");
            let resolved = cfg
                .optimization_params()
                .unwrap()
                .overhang
                .expect("a direction");
            assert_eq!(resolved, expected);
            assert_eq!(resolved.axis(), axis);
            assert_eq!(resolved.is_positive(), positive);
            assert_eq!(resolved.label(), spelling);
        }
    }

    /// The guide wireframe table: optional, every key inside it optional too,
    /// and present is what switches the guide on.
    #[test]
    fn the_wireframe_table_is_optional_and_every_key_in_it_defaults() {
        let off = Config::parse(MINIMAL).expect("parse");
        assert!(off.optimization_params().unwrap().wireframe.is_none());

        // The bare table: on, with every default resolved.
        let bare = MINIMAL.replace(
            "min_feature_mm = 6.0",
            "min_feature_mm = 6.0\n\n[optimization.wireframe]",
        );
        let cfg = Config::parse(&bare).expect("parse");
        cfg.validate_static().expect("validate");
        let resolved = cfg
            .optimization_params()
            .unwrap()
            .wireframe
            .expect("a wireframe");
        assert_eq!(resolved.radius_mm, constants::WIREFRAME_RADIUS_MM);
        assert_eq!(
            resolved.hold_iterations,
            constants::WIREFRAME_HOLD_ITERATIONS
        );
        assert_eq!(resolved.seed_density, constants::WIREFRAME_SEED_DENSITY);

        // Every key pinned, and every pinned key arriving.
        let pinned = MINIMAL.replace(
            "min_feature_mm = 6.0",
            "min_feature_mm = 6.0\n\n[optimization.wireframe]\nradius_mm = 1.5\n\
             hold_iterations = 7\nseed_density = 0.6",
        );
        let cfg = Config::parse(&pinned).expect("parse");
        cfg.validate_static().expect("validate");
        let resolved = cfg
            .optimization_params()
            .unwrap()
            .wireframe
            .expect("a wireframe");
        assert_eq!(resolved.radius_mm, 1.5);
        assert_eq!(resolved.hold_iterations, 7);
        assert_eq!(resolved.seed_density, 0.6);
        // A zero hold is a legal request: seed the wire, hold nothing.
        let none_held = MINIMAL.replace(
            "min_feature_mm = 6.0",
            "min_feature_mm = 6.0\n\n[optimization.wireframe]\nhold_iterations = 0",
        );
        assert_eq!(
            Config::parse(&none_held)
                .expect("parse")
                .optimization_params()
                .unwrap()
                .wireframe
                .expect("a wireframe")
                .hold_iterations,
            0
        );
    }

    #[test]
    fn degenerate_wireframe_controls_are_rejected() {
        for (body, needle) in [
            ("radius_mm = 0.0", "radius_mm"),
            ("radius_mm = -2.0", "radius_mm"),
            ("seed_density = 0.0", "seed_density"),
            ("seed_density = 1.5", "seed_density"),
            ("seed_density = -0.1", "seed_density"),
        ] {
            let text = MINIMAL.replace(
                "min_feature_mm = 6.0",
                &format!("min_feature_mm = 6.0\n\n[optimization.wireframe]\n{body}"),
            );
            let err = Config::parse(&text)
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{body}: unexpected error: {err}");
        }
        // A number that is not one is caught by the finiteness pass, which names
        // the key before anything divides by it.
        for value in ["nan", "inf"] {
            let text = MINIMAL.replace(
                "min_feature_mm = 6.0",
                &format!("min_feature_mm = 6.0\n\n[optimization.wireframe]\nradius_mm = {value}"),
            );
            let err = Config::parse(&text).unwrap_err().to_string();
            assert!(err.contains("radius_mm"), "{value}: {err}");
        }
        // And an unknown key inside the table fails loudly rather than silently.
        let text = MINIMAL.replace(
            "min_feature_mm = 6.0",
            "min_feature_mm = 6.0\n\n[optimization.wireframe]\nhold_steps = 3",
        );
        let err = Config::parse(&text).unwrap_err().to_string();
        assert!(err.contains("hold_steps"), "unexpected error: {err}");
    }

    #[test]
    fn the_update_scheme_defaults_to_the_optimality_criteria_step() {
        let cfg = Config::parse(MINIMAL).expect("parse");
        assert_eq!(cfg.optimization_params().unwrap().update, UpdateScheme::Oc);

        for (spelling, expected) in [("oc", UpdateScheme::Oc), ("mma", UpdateScheme::Mma)] {
            let text = MINIMAL.replace(
                "min_feature_mm = 6.0",
                &format!("min_feature_mm = 6.0\nupdate = \"{spelling}\""),
            );
            let cfg = Config::parse(&text).unwrap_or_else(|e| panic!("{spelling}: {e}"));
            cfg.validate_static().expect("validate");
            let resolved = cfg.optimization_params().unwrap().update;
            assert_eq!(resolved, expected);
            assert_eq!(resolved.label(), spelling);
        }
    }

    #[test]
    fn an_unknown_update_scheme_is_rejected_and_names_the_alternatives() {
        let text = MINIMAL.replace(
            "min_feature_mm = 6.0",
            "min_feature_mm = 6.0\nupdate = \"newton\"",
        );
        let err = Config::parse(&text).unwrap_err().to_string();
        assert!(err.contains("newton"), "unexpected error: {err}");
        assert!(
            err.contains(UpdateScheme::Oc.label()) && err.contains(UpdateScheme::Mma.label()),
            "the error must name the options: {err}"
        );
    }

    #[test]
    fn an_unknown_build_direction_is_rejected() {
        let text = MINIMAL.replace(
            "min_feature_mm = 6.0",
            "min_feature_mm = 6.0\n\n[optimization.overhang]\nbuild_direction = \"up\"",
        );
        let err = Config::parse(&text).unwrap_err().to_string();
        assert!(err.contains("up"), "unexpected error: {err}");

        // And so is an angle knob: the stencil fixes the angle at 45 degrees.
        let text = MINIMAL.replace(
            "min_feature_mm = 6.0",
            "min_feature_mm = 6.0\n\n[optimization.overhang]\nbuild_direction = \"z+\"\nangle = 30.0",
        );
        let err = Config::parse(&text).unwrap_err().to_string();
        assert!(err.contains("angle"), "unexpected error: {err}");
    }

    fn with_gravity(body: &str) -> String {
        format!("{MINIMAL}\n[[loadcases.loads]]\ntype = \"gravity\"\n{body}")
    }

    #[test]
    fn a_gravity_load_defaults_to_standard_gravity_pulling_down() {
        let cfg = Config::parse(&with_gravity("")).expect("parse");
        cfg.validate_static().expect("validate");
        let load = &cfg.loadcases[0].loads[1];
        assert_eq!(load.kind(), "gravity");
        assert!(load.region().is_none(), "gravity selects no region");
        let acceleration = load.gravity_acceleration("test").unwrap().expect("gravity");
        assert_eq!(acceleration, [0.0, 0.0, -constants::DEFAULT_GRAVITY_MM_S2]);
    }

    #[test]
    fn a_gravity_direction_is_normalized_before_it_is_scaled() {
        let cfg = Config::parse(&with_gravity(
            "direction = [0.0, 3.0, -4.0]\ng_mm_s2 = 1000.0",
        ))
        .expect("parse");
        let acceleration = cfg.loadcases[0].loads[1]
            .gravity_acceleration("test")
            .unwrap()
            .expect("gravity");
        assert!((crate::geometry::length(acceleration) - 1000.0).abs() < 1e-9);
        assert!((acceleration[1] - 600.0).abs() < 1e-9);
        assert!((acceleration[2] + 800.0).abs() < 1e-9);
    }

    #[test]
    fn a_degenerate_gravity_load_is_rejected() {
        for (body, needle) in [
            ("direction = [0.0, 0.0, 0.0]", "zero vector"),
            ("g_mm_s2 = 0.0", "positive"),
            ("g_mm_s2 = -9810.0", "positive"),
        ] {
            let err = Config::parse(&with_gravity(body))
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{body}: unexpected error: {err}");
        }
        // A region key belongs to the other load types, not to this one.
        let err = Config::parse(&with_gravity(
            "region = { shape = \"sphere\", center = [0.0, 0.0, 0.0], radius = 1.0 }",
        ))
        .unwrap_err()
        .to_string();
        assert!(err.contains("region"), "unexpected error: {err}");
    }

    #[test]
    fn the_output_table_resolves_the_void_policy_and_the_stress_report_path() {
        let default = Config::parse(MINIMAL)
            .unwrap()
            .output_params(Path::new("/cfg"))
            .unwrap();
        assert_eq!(default.voids, VoidPolicy::Warn);
        assert!(default.stress_json.is_none());

        let text = MINIMAL.replace(
            "stl_path = \"out.stl\"",
            "stl_path = \"out.stl\"\nvoids = \"fill\"\nstress_json = \"reports/stress.json\"",
        );
        let resolved = Config::parse(&text)
            .unwrap()
            .output_params(Path::new("/cfg"))
            .unwrap();
        assert_eq!(resolved.voids, VoidPolicy::Fill);
        let path = resolved.stress_json.expect("a path");
        assert!(
            path.starts_with("/cfg"),
            "{} is not anchored on the config directory",
            path.display()
        );
        assert!(path.ends_with("stress.json"));

        // An absolute path is left alone.
        let absolute = std::env::temp_dir().join("growforge_stress.json");
        let text = MINIMAL.replace(
            "stl_path = \"out.stl\"",
            &format!(
                "stl_path = \"out.stl\"\nstress_json = {}",
                crate::json::string(&absolute.display().to_string())
            ),
        );
        let resolved = Config::parse(&text)
            .unwrap()
            .output_params(Path::new("/cfg"))
            .unwrap();
        assert_eq!(resolved.stress_json.unwrap(), absolute);
    }

    #[test]
    fn the_output_table_resolves_the_island_policy() {
        let default = Config::parse(MINIMAL)
            .unwrap()
            .output_params(Path::new("/cfg"))
            .unwrap();
        assert_eq!(default.islands, IslandPolicy::Cull);

        for policy in [IslandPolicy::Cull, IslandPolicy::Keep] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\nislands = \"{}\"", policy.label()),
            );
            let resolved = Config::parse(&text)
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap();
            assert_eq!(resolved.islands, policy);
        }

        // An unknown policy and a misspelled key are both rejected, rather than
        // silently exporting whatever the default happens to be.
        let text = MINIMAL.replace(
            "stl_path = \"out.stl\"",
            "stl_path = \"out.stl\"\nislands = \"prune\"",
        );
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(error.contains("prune"), "unexpected error: {error}");

        let text = MINIMAL.replace(
            "stl_path = \"out.stl\"",
            "stl_path = \"out.stl\"\nisland = \"keep\"",
        );
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(error.contains("island"), "unexpected error: {error}");
    }

    /// A keepout tube thicker than the bend it follows overlaps itself. That is
    /// a legal solid and it voxelizes like any other, so it is a **warning**
    /// rather than a rejection - but the export's boundary clamp cannot project
    /// onto its surface, so the run has to say so instead of leaving the user
    /// with an undiagnosable count of vertices it gave up on.
    #[test]
    fn a_self_overlapping_keepout_tube_is_warned_about_and_a_sound_one_is_not() {
        let keepout = |body: &str| {
            MINIMAL.replace(
                "[[supports]]",
                &format!("[[keepout]]\nshape = \"tube\"\n{body}\n\n[[supports]]"),
            )
        };
        // The **fold**: a half circle of radius 2.6 mm as the centre line, and a
        // tube that reaches past the centre of that circle.
        let folded = |radius: f64| {
            keepout(&format!(
                "p1 = [12.6, 5.0, 5.0]\np2 = [7.4, 5.0, 5.0]\nbend = [10.0, 7.6, 5.0]\n\
                 radius = {radius}"
            ))
        };
        // The **gap**: an arc of radius 6 mm bent nearly the whole way round,
        // whose two ends are 1.044 mm apart, inside a tube of 1.8 - well under
        // the arc's own radius, so nothing about its bend is what is wrong.
        let closed = |radius: f64| {
            keepout(&format!(
                "p1 = [15.909, 5.0, 5.522]\np2 = [15.909, 5.0, 4.478]\nbend = [4.0, 5.0, 5.0]\n\
                 radius = {radius}"
            ))
        };

        for text in [folded(3.0), closed(1.8)] {
            let overlapping = Config::parse(&text).expect("parse");
            // Legal, and still legal: the shape is refused by nothing.
            overlapping
                .validate_static()
                .expect("a self-overlapping tube is a solid, not an error");
            let warnings = overlapping.static_warnings();
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert!(
                warnings[0].starts_with("[[keepout]] entry 1:"),
                "{warnings:?}"
            );
            assert!(warnings[0].contains("overlaps itself"), "{warnings:?}");
            // One message covers both causes, so it has to name both: a reader
            // whose tube is a thin hoop must not be told to look at its bend.
            assert!(
                warnings[0].contains("centre of its own bend"),
                "{warnings:?}"
            );
            assert!(warnings[0].contains("closed on each other"), "{warnings:?}");
            assert!(warnings[0].contains("boundaries"), "{warnings:?}");
            assert!(
                warnings[0].contains("reduce radius"),
                "the warning must say what to do about it: {warnings:?}"
            );
        }

        // A hair under either threshold is an ordinary tube and is not
        // mentioned. The hoop's ends are 1.044 mm apart, so its own threshold is
        // half of that rather than anything to do with its 6 mm arc.
        assert!(
            Config::parse(&folded(0.99 * 2.6))
                .expect("parse")
                .static_warnings()
                .is_empty()
        );
        assert!(
            Config::parse(&closed(0.99 * 0.522))
                .expect("parse")
                .static_warnings()
                .is_empty()
        );
        // Neither is a straight one, however thick, nor a configuration with no
        // keepout at all.
        let straight = Config::parse(&keepout(
            "p1 = [12.6, 5.0, 5.0]\np2 = [7.4, 5.0, 5.0]\nradius = 50.0",
        ))
        .expect("parse");
        assert!(straight.static_warnings().is_empty());
        assert!(Config::parse(MINIMAL).unwrap().static_warnings().is_empty());
    }

    /// The boundary fidelity, whose default is the one `[output]` policy that
    /// is *on* when the key is absent: a configuration that says nothing gets
    /// an exported surface held to the shapes it described.
    #[test]
    fn the_output_table_resolves_the_boundary_fidelity() {
        let default = Config::parse(MINIMAL)
            .unwrap()
            .output_params(Path::new("/cfg"))
            .unwrap();
        assert_eq!(
            default.boundaries,
            BoundaryFidelity::Exact,
            "an absent key has to mean the clamp is on"
        );

        for fidelity in [BoundaryFidelity::Exact, BoundaryFidelity::Voxel] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!(
                    "stl_path = \"out.stl\"\nboundaries = \"{}\"",
                    fidelity.label()
                ),
            );
            let resolved = Config::parse(&text)
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap();
            assert_eq!(resolved.boundaries, fidelity);
        }

        // An unknown value and a misspelled key are both rejected, rather than
        // silently exporting under whichever the default happens to be.
        let text = MINIMAL.replace(
            "stl_path = \"out.stl\"",
            "stl_path = \"out.stl\"\nboundaries = \"analytic\"",
        );
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(error.contains("analytic"), "unexpected error: {error}");

        let text = MINIMAL.replace(
            "stl_path = \"out.stl\"",
            "stl_path = \"out.stl\"\nboundary = \"voxel\"",
        );
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(error.contains("boundary"), "unexpected error: {error}");
    }

    /// The trim policy and its threshold, which is the one `[output]` number
    /// that has to be a fraction rather than a length or a count.
    #[test]
    fn the_output_table_resolves_the_trim_policy_and_its_fraction() {
        let default = Config::parse(MINIMAL)
            .unwrap()
            .output_params(Path::new("/cfg"))
            .unwrap();
        assert_eq!(default.trim, TrimPolicy::Off, "trim must be off by default");
        assert_eq!(
            default.trim_stress_fraction,
            constants::TRIM_STRESS_FRACTION
        );

        for policy in [TrimPolicy::Off, TrimPolicy::Stress] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!(
                    "stl_path = \"out.stl\"\ntrim = \"{}\"\ntrim_stress_fraction = 0.02",
                    policy.label()
                ),
            );
            let resolved = Config::parse(&text)
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap();
            assert_eq!(resolved.trim, policy);
            assert_eq!(resolved.trim_stress_fraction, 0.02);
        }

        // The fraction is an open interval: neither end is a threshold that
        // means anything, and both are checked whatever the policy is - a value
        // the file cannot mean is a mistake in the file.
        for value in ["0.0", "1.0", "-0.1", "1.5"] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\ntrim_stress_fraction = {value}"),
            );
            let error = Config::parse(&text)
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("trim_stress_fraction") && error.contains("between 0 and 1"),
                "unexpected error for {value}: {error}"
            );
        }

        // An unknown policy and a misspelled key are both rejected. "prune" in
        // particular: it is the growth engine's own key, and a configuration
        // that reached for it here has to be told rather than quietly exporting
        // the default.
        for (key, value) in [("trim", "prune"), ("trimming", "stress")] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\n{key} = \"{value}\""),
            );
            let error = Config::parse(&text).unwrap_err().to_string();
            assert!(error.contains(key), "unexpected error: {error}");
        }
    }

    /// The reinforcement policy and its floor, which is the one `[output]`
    /// number whose default comes from another table.
    #[test]
    fn the_output_table_resolves_the_reinforce_policy_and_its_thickness() {
        let default = Config::parse(MINIMAL)
            .unwrap()
            .output_params(Path::new("/cfg"))
            .unwrap();
        assert_eq!(
            default.reinforce,
            ReinforcePolicy::Off,
            "reinforce must be off by default"
        );
        assert_eq!(
            default.reinforce_thickness_mm, 6.0,
            "the floor must default to [optimization] min_feature_mm"
        );

        // Which really is read from that table rather than from a constant.
        let coarser =
            Config::parse(&MINIMAL.replace("min_feature_mm = 6.0", "min_feature_mm = 9.0"))
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap();
        assert_eq!(coarser.reinforce_thickness_mm, 9.0);

        for policy in [ReinforcePolicy::Off, ReinforcePolicy::MinThickness] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!(
                    "stl_path = \"out.stl\"\nreinforce = \"{}\"\nreinforce_thickness_mm = 2.5",
                    policy.label()
                ),
            );
            let resolved = Config::parse(&text)
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap();
            assert_eq!(resolved.reinforce, policy);
            assert_eq!(resolved.reinforce_thickness_mm, 2.5);
        }

        // A thickness is a length: finite and above zero, checked whatever the
        // policy is.
        for value in ["0.0", "-1.0", "nan", "inf"] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\nreinforce_thickness_mm = {value}"),
            );
            let error = Config::parse(&text)
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("reinforce_thickness_mm") && error.contains("positive length"),
                "unexpected error for {value}: {error}"
            );
        }

        // An unknown policy and a misspelled key are both rejected. "thicken"
        // in particular: it is what the growth engine calls its own strut
        // sizing, and a configuration that reached for it here has to be told.
        for (key, value) in [("reinforce", "thicken"), ("reinforcement", "min_thickness")] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\n{key} = \"{value}\""),
            );
            let error = Config::parse(&text).unwrap_err().to_string();
            assert!(error.contains(key), "unexpected error: {error}");
        }
    }

    /// The flush policy and its depth, which is the one `[output]` key this
    /// method deliberately does **not** default: how deep two voxels is, is a
    /// property of the grid, and `Problem::build` is where that is known.
    #[test]
    fn the_output_table_resolves_the_flush_policy_and_carries_its_depth_as_asked() {
        let default = Config::parse(MINIMAL)
            .unwrap()
            .output_params(Path::new("/cfg"))
            .unwrap();
        assert_eq!(
            default.flush,
            FlushPolicy::Off,
            "flush must be off by default"
        );
        assert_eq!(
            default.flush_depth_mm, None,
            "an absent depth stays absent here; the grid is what resolves it"
        );

        for policy in [FlushPolicy::Off, FlushPolicy::Walls] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!(
                    "stl_path = \"out.stl\"\nflush = \"{}\"\nflush_depth_mm = 2.5",
                    policy.label()
                ),
            );
            let resolved = Config::parse(&text)
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap();
            assert_eq!(resolved.flush, policy);
            assert_eq!(resolved.flush_depth_mm, Some(2.5));
        }

        // A depth is a length: finite and above zero, checked whatever the
        // policy is. How many voxels of it is sane is checked where the voxel
        // size is; see `Problem::build`.
        for value in ["0.0", "-1.0", "nan", "inf"] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\nflush_depth_mm = {value}"),
            );
            let error = Config::parse(&text)
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("flush_depth_mm") && error.contains("positive length"),
                "unexpected error for {value}: {error}"
            );
        }

        // An unknown policy and a misspelled key are both rejected. "exact" in
        // particular: it is what `[output] boundaries` calls the mesh side of
        // the same defect, and a configuration that reached for it here has to
        // be told.
        for (key, value) in [("flush", "exact"), ("flush_walls", "walls")] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\n{key} = \"{value}\""),
            );
            let error = Config::parse(&text).unwrap_err().to_string();
            assert!(error.contains(key), "unexpected error: {error}");
        }
    }

    #[test]
    fn the_supersample_factor_defaults_to_one_and_is_bounded() {
        let default = Config::parse(MINIMAL)
            .unwrap()
            .output_params(Path::new("/cfg"))
            .unwrap();
        assert_eq!(default.supersample, constants::DEFAULT_SUPERSAMPLE);
        assert_eq!(default.supersample, 1);

        let with = |value: &str| {
            Config::parse(&MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\nsupersample = {value}"),
            ))
        };
        assert_eq!(
            with(&constants::MAX_SUPERSAMPLE.to_string())
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap()
                .supersample,
            constants::MAX_SUPERSAMPLE
        );

        // Zero would mesh nothing, and past the cap the triangles describe the
        // interpolant rather than the design.
        for value in ["0", &(constants::MAX_SUPERSAMPLE + 1).to_string()] {
            let err = with(value)
                .unwrap()
                .output_params(Path::new("/cfg"))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("supersample") && err.contains("[1, 4]"),
                "{value}: unexpected error: {err}"
            );
        }
        // A negative or fractional factor is not a count at all.
        assert!(with("-1").is_err());
        assert!(with("2.5").is_err());
    }

    #[test]
    fn an_unknown_void_policy_or_an_empty_stress_path_is_rejected() {
        let text = MINIMAL.replace(
            "stl_path = \"out.stl\"",
            "stl_path = \"out.stl\"\nvoids = \"drill\"",
        );
        let err = Config::parse(&text).unwrap_err().to_string();
        assert!(err.contains("drill"), "unexpected error: {err}");

        let text = MINIMAL.replace(
            "stl_path = \"out.stl\"",
            "stl_path = \"out.stl\"\nstress_json = \"\"",
        );
        let err = Config::parse(&text)
            .unwrap()
            .output_params(Path::new("/cfg"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("stress_json"), "unexpected error: {err}");
    }

    /// The minimal configuration with the growth engine selected.
    fn growth(extra: &str) -> String {
        format!(
            "engine = \"{}\"\n{MINIMAL}{extra}",
            constants::GROWTH_ENGINE
        )
    }

    #[test]
    fn growth_controls_default_to_the_scale_of_the_part() {
        let cfg = Config::parse(&growth("")).expect("parse");
        cfg.validate_static().expect("validate");
        assert!(cfg.is_growth());
        let g = cfg.growth_params().unwrap().expect("growth params");

        // min_feature_mm = 6, mass_fraction = 0.3.
        assert_eq!(g.seed, constants::DEFAULT_GROWTH_SEED);
        assert_eq!(g.attractor_per_cm3, constants::DEFAULT_ATTRACTOR_PER_CM3);
        assert_eq!(g.murray_exponent, constants::DEFAULT_MURRAY_EXPONENT);
        assert_eq!(g.max_steps, constants::DEFAULT_GROWTH_MAX_STEPS);
        assert!((g.step_mm - 0.75 * 6.0).abs() < 1e-12);
        assert!((g.max_radius_mm - 3.0 * 6.0).abs() < 1e-12);
        let expected_kill = 3.0 * constants::GROWTH_KILL_MASS_FRACTION_MARGIN / 0.3_f64.sqrt();
        assert!((g.kill_radius_mm - expected_kill).abs() < 1e-12);
        assert!(
            (g.attraction_radius_mm - constants::GROWTH_ATTRACTION_PER_KILL * expected_kill).abs()
                < 1e-12
        );
        // The defaults have to be self consistent, whatever the part's scale.
        assert!(g.kill_radius_mm < g.attraction_radius_mm);

        // Doubling the feature size doubles every length default.
        let bigger =
            Config::parse(&growth("").replace("min_feature_mm = 6.0", "min_feature_mm = 12.0"))
                .unwrap()
                .growth_params()
                .unwrap()
                .expect("growth params");
        assert!((bigger.step_mm - 2.0 * g.step_mm).abs() < 1e-12);
        assert!((bigger.kill_radius_mm - 2.0 * g.kill_radius_mm).abs() < 1e-12);
    }

    #[test]
    fn every_growth_key_can_be_overridden() {
        let cfg = Config::parse(&growth(
            r#"
[growth]
seed = 7
attractor_per_cm3 = 2.5
attraction_radius_mm = 40.0
kill_radius_mm = 9.0
step_mm = 3.0
murray_exponent = 3.0
max_radius_mm = 15.0
max_steps = 25
"#,
        ))
        .expect("parse");
        cfg.validate_static().expect("validate");
        let g = cfg.growth_params().unwrap().expect("growth params");
        assert_eq!(g.seed, 7);
        assert_eq!(g.attractor_per_cm3, 2.5);
        assert_eq!(g.attraction_radius_mm, 40.0);
        assert_eq!(g.kill_radius_mm, 9.0);
        assert_eq!(g.step_mm, 3.0);
        assert_eq!(g.murray_exponent, 3.0);
        assert_eq!(g.max_radius_mm, 15.0);
        assert_eq!(g.max_steps, 25);
    }

    #[test]
    fn a_growth_table_without_the_growth_engine_is_rejected() {
        let text = format!("{MINIMAL}\n[growth]\nseed = 1\n");
        let err = Config::parse(&text)
            .unwrap()
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(err.contains("[growth]"), "unexpected error: {err}");
        assert!(err.contains(constants::GROWTH_ENGINE), "unexpected: {err}");
        // And the engine on its own, without the table, is perfectly fine.
        Config::parse(&growth(""))
            .unwrap()
            .validate_static()
            .expect("the table is optional");
    }

    #[test]
    fn the_growth_engine_rejects_an_overhang_constraint() {
        let text = growth("").replace(
            "min_feature_mm = 6.0",
            "min_feature_mm = 6.0\n\n[optimization.overhang]\nbuild_direction = \"z+\"",
        );
        let err = Config::parse(&text)
            .unwrap()
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(err.contains("overhang"), "unexpected error: {err}");
        assert!(
            err.contains("build direction is not available"),
            "the error has to say there is no growth equivalent: {err}"
        );
        assert!(
            err.contains(constants::DEFAULT_ENGINE),
            "the error has to offer the engine that does support it: {err}"
        );
    }

    #[test]
    fn the_growth_engine_rejects_a_guide_wireframe() {
        let text = growth("").replace(
            "min_feature_mm = 6.0",
            "min_feature_mm = 6.0\n\n[optimization.wireframe]\nradius_mm = 2.0",
        );
        let err = Config::parse(&text)
            .unwrap()
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(err.contains("wireframe"), "unexpected error: {err}");
        assert!(
            err.contains("no growth equivalent"),
            "the error has to say there is no growth equivalent: {err}"
        );
        assert!(
            err.contains(constants::DEFAULT_ENGINE),
            "the error has to offer the engine that does support it: {err}"
        );
    }

    /// The minimal configuration with the solid engine selected, which is the
    /// minimal one *without* its mass fraction: that engine refuses the key.
    fn solid(extra: &str) -> String {
        format!(
            "engine = \"{}\"\n{}{extra}",
            constants::SOLID_ENGINE,
            MINIMAL.replace("mass_fraction = 0.3\n", "")
        )
    }

    #[test]
    fn the_solid_engine_needs_no_mass_fraction_and_every_other_engine_still_does() {
        let cfg = Config::parse(&solid("")).expect("parse");
        cfg.validate_static().expect("validate");
        assert!(cfg.is_solid() && !cfg.is_growth());
        // Resolved to a full domain, which is what the field really comes out
        // at and what every summary line reads.
        assert_eq!(
            cfg.optimization_params().unwrap().mass_fraction,
            constants::DENSITY_MAX
        );

        // The requirement is lifted for that engine and for no other.
        for engine in [constants::DEFAULT_ENGINE, constants::GROWTH_ENGINE] {
            let text = format!(
                "engine = \"{engine}\"\n{}",
                MINIMAL.replace("mass_fraction = 0.3\n", "")
            );
            let err = Config::parse(&text)
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("mass_fraction") && err.contains("required"),
                "{engine}: unexpected error: {err}"
            );
            assert!(
                err.contains(constants::SOLID_ENGINE),
                "{engine}: the error has to name the engine that does without it: {err}"
            );
        }
    }

    #[test]
    fn the_solid_engine_rejects_a_mass_fraction_that_is_set() {
        let text = format!("engine = \"{}\"\n{MINIMAL}", constants::SOLID_ENGINE);
        let err = Config::parse(&text)
            .expect("parse")
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(err.contains("mass_fraction"), "unexpected error: {err}");
        assert!(
            err.contains("Remove the key"),
            "the error has to say what to do about it: {err}"
        );
        assert!(
            err.contains(constants::SOLID_ENGINE) && err.contains(constants::DEFAULT_ENGINE),
            "the error has to name both engines: {err}"
        );
        // A value outside the legal range is refused for the same reason rather
        // than for its range: the key is not this engine's to set at all.
        let out_of_range = format!(
            "engine = \"{}\"\n{}",
            constants::SOLID_ENGINE,
            MINIMAL.replace("mass_fraction = 0.3", "mass_fraction = 1.5")
        );
        let err = Config::parse(&out_of_range)
            .expect("parse")
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(err.contains("Remove the key"), "unexpected error: {err}");
    }

    #[test]
    fn the_solid_engine_rejects_every_table_that_prices_a_density_field() {
        for (body, needle) in [
            (
                "\n[optimization.overhang]\nbuild_direction = \"z+\"\n",
                "overhang",
            ),
            ("\n[optimization.wireframe]\nradius_mm = 2.0\n", "wireframe"),
            (
                "\n[optimization.local_volume]\nmax_fraction = 0.6\n",
                "local_volume",
            ),
        ] {
            let err = Config::parse(&solid(body))
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{body}: unexpected error: {err}");
            assert!(
                err.contains("no solid equivalent") || err.contains("solid engine"),
                "{body}: the error has to say why: {err}"
            );
            assert!(
                err.contains(constants::DEFAULT_ENGINE),
                "{body}: the error has to offer the engine that does support it: {err}"
            );
        }
        // The local cap is refused by the engine rather than by the update
        // scheme gate, which would otherwise fire first and give advice to
        // nowhere.
        let err = Config::parse(&solid(
            "\n[optimization.local_volume]\nmax_fraction = 0.6\n",
        ))
        .expect("parse")
        .validate_static()
        .unwrap_err()
        .to_string();
        assert!(
            !err.contains("needs update"),
            "the update scheme gate answered for the solid engine: {err}"
        );
    }

    #[test]
    fn the_solid_engine_rejects_a_pass_that_would_alter_the_domain() {
        for (body, needle) in [
            ("trim = \"stress\"", "trim"),
            ("reinforce = \"min_thickness\"", "reinforce"),
            ("flush = \"walls\"", "flush"),
        ] {
            let text = solid("").replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\n{body}"),
            );
            let err = Config::parse(&text)
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{body}: unexpected error: {err}");
            assert!(
                err.contains("exports exactly the domain"),
                "{body}: the error has to say why: {err}"
            );
        }
        // All three explicitly off is what the engine asks for, and is accepted.
        let off = solid("").replace(
            "stl_path = \"out.stl\"",
            "stl_path = \"out.stl\"\ntrim = \"off\"\nreinforce = \"off\"\nflush = \"off\"",
        );
        Config::parse(&off)
            .expect("parse")
            .validate_static()
            .expect("off is what the solid engine wants");
    }

    #[test]
    fn the_solid_engine_leaves_the_simp_knobs_it_never_reads_alone() {
        // Legal but unused, exactly as they are under the growth engine: a
        // configuration that carries them still builds, and they resolve to
        // whatever they say.
        let text = solid("").replace(
            "min_feature_mm = 6.0",
            "min_feature_mm = 6.0\npenalty = 4.0\nstiffness_floor = 1e-6\nmax_iterations = 7\n\
             convergence_tol = 0.05\nupdate = \"mma\"",
        );
        let cfg = Config::parse(&text).expect("parse");
        cfg.validate_static().expect("validate");
        let params = cfg.optimization_params().expect("params");
        assert_eq!(params.penalty, 4.0);
        assert_eq!(params.stiffness_floor, 1e-6);
        assert_eq!(params.max_iterations, 7);
        assert_eq!(params.update, UpdateScheme::Mma);
    }

    #[test]
    fn degenerate_growth_controls_are_rejected() {
        for (body, needle) in [
            (
                "kill_radius_mm = 40.0\nattraction_radius_mm = 20.0",
                "kill_radius_mm",
            ),
            (
                "kill_radius_mm = 20.0\nattraction_radius_mm = 20.0",
                "kill_radius_mm",
            ),
            ("step_mm = 0.0", "step_mm"),
            ("step_mm = -1.0", "step_mm"),
            ("attractor_per_cm3 = 0.0", "attractor_per_cm3"),
            ("murray_exponent = 0.5", "murray_exponent"),
            ("murray_exponent = 100.0", "murray_exponent"),
            ("max_radius_mm = 1.0", "max_radius_mm"),
            ("max_steps = 0", "max_steps"),
        ] {
            let err = Config::parse(&growth(&format!("\n[growth]\n{body}\n")))
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{body}: unexpected error: {err}");
        }
    }

    /// `[growth.symmetry]` under `[growth]`, resolved into the variant its kind
    /// describes.
    fn symmetry(body: &str) -> String {
        growth(&format!("\n[growth]\n[growth.symmetry]\n{body}\n"))
    }

    #[test]
    fn a_symmetry_table_resolves_into_the_variant_its_kind_names() {
        let one = Config::parse(&symmetry("kind = \"mirror\"\nplanes = [\"x\"]"))
            .expect("parse")
            .growth_params()
            .expect("resolve")
            .expect("growth params")
            .symmetry
            .expect("a symmetry");
        assert_eq!(
            one,
            SymmetryParams::Mirror {
                first: Axis::X,
                second: None
            }
        );
        assert_eq!(one.sectors(), 2);
        assert_eq!(one.kind(), SymmetryKind::Mirror);
        assert_eq!(one.describe(), "mirror across x");

        let four = Config::parse(&symmetry("kind = \"mirror\"\nplanes = [\"x\", \"y\"]"))
            .expect("parse")
            .growth_params()
            .expect("resolve")
            .expect("growth params")
            .symmetry
            .expect("a symmetry");
        assert_eq!(
            four,
            SymmetryParams::Mirror {
                first: Axis::X,
                second: Some(Axis::Y)
            }
        );
        assert_eq!(four.sectors(), 4);
        assert_eq!(four.describe(), "mirror across x and y");

        // The axis of a rotation defaults rather than being required, because
        // z is up in every other default growforge has.
        let turned = Config::parse(&symmetry("kind = \"rotational\"\norder = 6"))
            .expect("parse")
            .growth_params()
            .expect("resolve")
            .expect("growth params")
            .symmetry
            .expect("a symmetry");
        assert_eq!(
            turned,
            SymmetryParams::Rotational {
                order: 6,
                axis: constants::DEFAULT_SYMMETRY_AXIS
            }
        );
        assert_eq!(turned.sectors(), 6);
        assert_eq!(turned.kind(), SymmetryKind::Rotational);
        assert_eq!(turned.describe(), "6-fold rotation about z");
        assert_eq!(
            Config::parse(&symmetry("kind = \"rotational\"\norder = 3\naxis = \"x\""))
                .expect("parse")
                .growth_params()
                .expect("resolve")
                .expect("growth params")
                .symmetry,
            Some(SymmetryParams::Rotational {
                order: 3,
                axis: Axis::X
            })
        );

        // And no table at all is the default: the whole domain is grown.
        assert_eq!(
            Config::parse(&growth(""))
                .expect("parse")
                .growth_params()
                .expect("resolve")
                .expect("growth params")
                .symmetry,
            None
        );
    }

    #[test]
    fn degenerate_symmetry_controls_are_rejected() {
        for (body, needle) in [
            ("kind = \"mirror\"", "needs planes"),
            ("kind = \"mirror\"\nplanes = []", "needs planes"),
            (
                "kind = \"mirror\"\nplanes = [\"x\", \"y\", \"z\"]",
                "at most 2 mirror planes",
            ),
            (
                "kind = \"mirror\"\nplanes = [\"y\", \"y\"]",
                "lists \"y\" twice",
            ),
            ("kind = \"mirror\"\nplanes = [\"x\"]\norder = 4", "order"),
            ("kind = \"mirror\"\nplanes = [\"x\"]\naxis = \"z\"", "axis"),
            ("kind = \"rotational\"", "needs order"),
            ("kind = \"rotational\"\norder = 1", "order must lie in"),
            ("kind = \"rotational\"\norder = 0", "order must lie in"),
            ("kind = \"rotational\"\norder = 13", "order must lie in"),
            (
                "kind = \"rotational\"\norder = 4\nplanes = [\"x\"]",
                "planes belongs to",
            ),
        ] {
            let err = Config::parse(&symmetry(body))
                .expect("parse")
                .validate_static()
                .unwrap_err()
                .to_string();
            assert!(err.contains("[growth.symmetry]"), "{body}: {err}");
            assert!(err.contains(needle), "{body}: unexpected error: {err}");
        }
        // The bounds are the constants, not numbers written twice.
        assert!(
            Config::parse(&symmetry(&format!(
                "kind = \"rotational\"\norder = {}",
                constants::GROWTH_SYMMETRY_MAX_ORDER
            )))
            .expect("parse")
            .validate_static()
            .is_ok()
        );
        // An unknown kind is a parse error, like every other tagged key.
        assert!(Config::parse(&symmetry("kind = \"helical\"")).is_err());
    }

    #[test]
    fn a_symmetry_table_without_the_growth_engine_says_so_in_its_own_words() {
        let text = format!("{MINIMAL}\n[growth.symmetry]\nkind = \"mirror\"\nplanes = [\"x\"]\n");
        let err = Config::parse(&text)
            .unwrap()
            .validate_static()
            .unwrap_err()
            .to_string();
        assert!(err.contains("[growth.symmetry]"), "unexpected error: {err}");
        assert!(
            err.contains("no skeleton to replicate"),
            "the error has to say why SIMP cannot have it: {err}"
        );
        assert!(err.contains(constants::GROWTH_ENGINE), "unexpected: {err}");
        // The same table with the engine that owns it is fine.
        Config::parse(&symmetry("kind = \"mirror\"\nplanes = [\"x\"]"))
            .unwrap()
            .validate_static()
            .expect("the growth engine accepts it");
    }

    /// The default backend is the compute device, and it is *soft*: a build
    /// that has no `gpu` feature resolves it to the CPU here rather than
    /// refusing the configuration, and says which it did.
    #[test]
    fn the_solver_table_is_optional_and_defaults_to_the_compute_backend() {
        let default = Config::parse(MINIMAL).unwrap();
        assert!(default.solver.is_none());
        default
            .validate_static()
            .expect("a soft default never fails");
        let resolved = default.solver_params().unwrap();
        assert!(
            !resolved.explicit,
            "a backend nobody named is not an instruction"
        );
        if cfg!(feature = "gpu") {
            assert_eq!(resolved.backend, constants::DEFAULT_SOLVER_BACKEND);
            assert_eq!(resolved.fell_back, None);
        } else {
            assert_eq!(
                resolved.backend,
                SolverBackend::Cpu,
                "a build with no compute backend must fall back rather than fail"
            );
            assert_eq!(resolved.fell_back, Some(SolverFallback::NoFeature));
            assert!(resolved.fell_back.unwrap().note().contains("cpu"));
        }

        // Named explicitly, it is an instruction: it is what runs, and it is
        // never softened into something else.
        let explicit = MINIMAL.replace("[output]", "[solver]\nbackend = \"cpu\"\n\n[output]");
        let parsed = Config::parse(&explicit).expect("parse");
        parsed.validate_static().expect("validate");
        let resolved = parsed.solver_params().unwrap();
        assert_eq!(resolved.backend, SolverBackend::Cpu);
        assert!(resolved.explicit && resolved.fell_back.is_none());
        assert_eq!(SolverBackend::Cpu.label(), "cpu");
        assert_eq!(SolverBackend::Gpu.label(), "gpu");
        // The type's own default is still the reference backend, so a problem
        // built in memory rather than parsed solves reproducibly.
        assert_eq!(SolverParams::default().backend, SolverBackend::Cpu);
        assert!(!SolverParams::default().explicit);
    }

    /// `[solver] tolerance`: absent is the constant, present is what it says,
    /// and the bounds are refused at both ends.
    ///
    /// The key exists because the fixed target was one: an honest model reached
    /// the iteration cap at a relative residual of 1.558e-8 against 1e-8, a
    /// factor of 1.5 from a number no printed part could tell apart, and had
    /// nowhere to say so.
    #[test]
    fn the_solver_tolerance_defaults_to_the_constant_and_is_bounded_at_both_ends() {
        let with =
            |body: &str| MINIMAL.replace("[output]", &format!("[solver]\n{body}\n\n[output]"));

        let absent = Config::parse(MINIMAL).expect("parse");
        assert_eq!(
            absent.solver_params().unwrap().tolerance,
            constants::CG_RELATIVE_TOLERANCE
        );
        // A table that names only the backend is the same absence.
        let keyless = Config::parse(&with("backend = \"cpu\"")).expect("parse");
        assert_eq!(
            keyless.solver_params().unwrap().tolerance,
            constants::CG_RELATIVE_TOLERANCE
        );

        // Named, it is what runs, and it rides beside the backend rather than
        // in place of it.
        let named = Config::parse(&with("backend = \"cpu\"\ntolerance = 3e-8")).expect("parse");
        assert_eq!(named.solver.as_ref().unwrap().tolerance, Some(3e-8));
        named.validate_static().expect("validate");
        let resolved = named.solver_params().unwrap();
        assert_eq!(resolved.tolerance, 3e-8);
        assert_eq!(resolved.backend, SolverBackend::Cpu);
        // Both bounds are reachable: they are the range, not the outside of it.
        for edge in [constants::CG_TOLERANCE_MIN, constants::CG_TOLERANCE_MAX] {
            let config = Config::parse(&with(&format!("tolerance = {edge:e}"))).expect("parse");
            assert_eq!(config.solver_params().unwrap().tolerance, edge);
        }

        // And outside them nothing is softened: a target no solve could meet is
        // refused before a run spends the iteration budget proving it.
        for bad in ["1e-11", "1e-3", "0.0", "-1e-8", "nan", "inf", "-inf"] {
            let text = with(&format!("tolerance = {bad}"));
            // `nan` and the infinities are stopped by the finite pass at parse,
            // the rest by the range check at resolution; either way the message
            // names the key and nothing reaches a solver.
            let error = match Config::parse(&text) {
                Ok(config) => config
                    .solver_params()
                    .expect_err(&format!("{bad} was accepted"))
                    .to_string(),
                Err(error) => error.to_string(),
            };
            assert!(error.contains("tolerance"), "for {bad}: {error}");
            assert!(error.contains("[solver]"), "for {bad}: {error}");
        }
        // The bounds are named, so the message says what the range is.
        let error = Config::parse(&with("tolerance = 1e-3"))
            .unwrap()
            .solver_params()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!("{:e}", constants::CG_TOLERANCE_MIN))
                && error.contains(&format!("{:e}", constants::CG_TOLERANCE_MAX)),
            "the message has to name the bounds: {error}"
        );
    }

    /// A `SolverParams` nobody resolved still carries limits a solve can meet.
    ///
    /// The [`Default`] impl is written out rather than derived for exactly this:
    /// a derived one would hand out `tolerance: 0.0`, a target no residual
    /// reaches, so every solve under it would spend the whole iteration cap and
    /// then fail. Nothing in the crate builds one for a run - `Problem` gets
    /// its controls from [`Config::solver_params`] - but the type is public and
    /// the trap would be silent.
    #[test]
    fn the_solver_params_default_is_the_reference_configuration() {
        let default = SolverParams::default();
        assert_eq!(default.tolerance, constants::CG_RELATIVE_TOLERANCE);
        assert!(default.tolerance > 0.0, "a target no solve could ever meet");
        assert_eq!(default.backend, SolverBackend::Cpu);
        assert!(!default.explicit);
        assert_eq!(default.fell_back, None);
        // And it is the same thing a configuration that says nothing resolves
        // to, but for the backend, whose default is soft and is the compute
        // device rather than the reference one.
        let resolved = Config::parse(MINIMAL).unwrap().solver_params().unwrap();
        assert_eq!(resolved.tolerance, default.tolerance);

        // `PartialEq` is what the `Eq` an f64 cannot have left behind, and it
        // compares the whole struct: the tolerance is part of what makes two
        // resolved sets of controls the same set.
        assert_eq!(default, SolverParams::default());
        assert_ne!(
            default,
            SolverParams {
                tolerance: 3e-8,
                ..default
            }
        );
    }

    #[test]
    fn the_gpu_backend_parses_and_is_accepted_only_by_a_build_that_has_it() {
        let text = MINIMAL.replace("[output]", "[solver]\nbackend = \"gpu\"\n\n[output]");
        let parsed = Config::parse(&text).expect("parse");
        assert_eq!(
            parsed.solver.as_ref().unwrap().backend,
            Some(SolverBackend::Gpu)
        );
        let resolved = parsed.solver_params();
        if cfg!(feature = "gpu") {
            assert_eq!(resolved.unwrap().backend, SolverBackend::Gpu);
            parsed.validate_static().expect("validate");
        } else {
            let error = resolved.unwrap_err().to_string();
            assert!(error.contains("gpu"), "unexpected error: {error}");
            assert!(
                error.contains("cargo feature"),
                "the error must say what is missing: {error}"
            );
        }
    }

    #[test]
    fn an_unknown_solver_backend_or_key_is_rejected() {
        let text = MINIMAL.replace("[output]", "[solver]\nbackend = \"cuda\"\n\n[output]");
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(error.contains("cuda"), "unexpected error: {error}");

        let text = MINIMAL.replace("[output]", "[solver]\nbackends = \"cpu\"\n\n[output]");
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(error.contains("backends"), "unexpected error: {error}");
    }

    /// Every enum's `label` has to be the spelling serde accepts, because the
    /// editor writes a configuration back out through them: a label that drifts
    /// from the deserializer produces a file growforge itself would reject.
    #[test]
    fn every_config_enum_labels_itself_the_way_it_is_spelled() {
        for policy in [VoidPolicy::Warn, VoidPolicy::Fill] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\nvoids = \"{}\"", policy.label()),
            );
            let parsed = Config::parse(&text).expect("parse");
            assert_eq!(parsed.output.voids, Some(policy));
        }
        for policy in [IslandPolicy::Cull, IslandPolicy::Keep] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\nislands = \"{}\"", policy.label()),
            );
            let parsed = Config::parse(&text).expect("parse");
            assert_eq!(parsed.output.islands, Some(policy));
        }
        for fidelity in [BoundaryFidelity::Exact, BoundaryFidelity::Voxel] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!(
                    "stl_path = \"out.stl\"\nboundaries = \"{}\"",
                    fidelity.label()
                ),
            );
            let parsed = Config::parse(&text).expect("parse");
            assert_eq!(parsed.output.boundaries, Some(fidelity));
        }
        for policy in [TrimPolicy::Off, TrimPolicy::Stress] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\ntrim = \"{}\"", policy.label()),
            );
            let parsed = Config::parse(&text).expect("parse");
            assert_eq!(parsed.output.trim, Some(policy));
        }
        for policy in [ReinforcePolicy::Off, ReinforcePolicy::MinThickness] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\nreinforce = \"{}\"", policy.label()),
            );
            let parsed = Config::parse(&text).expect("parse");
            assert_eq!(parsed.output.reinforce, Some(policy));
        }
        for policy in [FlushPolicy::Off, FlushPolicy::Walls] {
            let text = MINIMAL.replace(
                "stl_path = \"out.stl\"",
                &format!("stl_path = \"out.stl\"\nflush = \"{}\"", policy.label()),
            );
            let parsed = Config::parse(&text).expect("parse");
            assert_eq!(parsed.output.flush, Some(policy));
        }
        for op in [CsgOpSpec::Add, CsgOpSpec::Subtract] {
            let text = MINIMAL.replace("op = \"add\"", &format!("op = \"{}\"", op.label()));
            let parsed = Config::parse(&text).expect("parse");
            let expected = match op {
                CsgOpSpec::Add => CsgOp::Add,
                CsgOpSpec::Subtract => CsgOp::Subtract,
            };
            assert_eq!(parsed.domain[0].op(), expected);
        }
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let text = MINIMAL.replace(
                "max = [1.0, 10.0, 10.0] }",
                &format!(
                    "max = [1.0, 10.0, 10.0] }}\ndirections = [\"{}\"]",
                    axis.label()
                ),
            );
            let parsed = Config::parse(&text).expect("parse");
            assert_eq!(parsed.supports[0].directions.as_deref(), Some(&[axis][..]));
        }
        for shape in [
            ShapeSpec::Box {
                min: [0.0; 3],
                max: [1.0; 3],
                rotation_deg: None,
            },
            ShapeSpec::Cylinder {
                p1: [0.0; 3],
                p2: [1.0; 3],
                radius: 1.0,
            },
            ShapeSpec::Sphere {
                center: [0.0; 3],
                radius: 1.0,
            },
            ShapeSpec::Ellipsoid {
                center: [0.0; 3],
                radii: [1.0; 3],
                rotation_deg: None,
            },
            ShapeSpec::Tube {
                p1: [0.0; 3],
                p2: [1.0; 3],
                bend: None,
                radius: 1.0,
            },
            ShapeSpec::Cone {
                p1: [0.0; 3],
                p2: [1.0; 3],
                radius1: 1.0,
                radius2: 0.0,
            },
            ShapeSpec::Triangle {
                a: [0.0; 3],
                b: [1.0, 0.0, 0.0],
                c: [0.0, 1.0, 0.0],
                thickness: 1.0,
            },
        ] {
            // The `shape` key of the minimal config is a box; swapping the
            // spelling in has to keep it parseable as the same variant.
            let body = match shape {
                ShapeSpec::Box { .. } => "min = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]",
                ShapeSpec::Cylinder { .. } => {
                    "p1 = [0.0, 0.0, 0.0]\np2 = [20.0, 0.0, 0.0]\nradius = 5.0"
                }
                ShapeSpec::Sphere { .. } => "center = [0.0, 0.0, 0.0]\nradius = 5.0",
                ShapeSpec::Ellipsoid { .. } => {
                    "center = [10.0, 5.0, 5.0]\nradii = [10.0, 5.0, 5.0]"
                }
                ShapeSpec::Tube { .. } => {
                    "p1 = [2.0, 5.0, 5.0]\np2 = [18.0, 5.0, 5.0]\n\
                     bend = [10.0, 8.0, 5.0]\nradius = 4.0"
                }
                ShapeSpec::Cone { .. } => {
                    "p1 = [2.0, 5.0, 5.0]\np2 = [18.0, 5.0, 5.0]\nradius1 = 5.0\nradius2 = 2.0"
                }
                ShapeSpec::Triangle { .. } => {
                    "a = [2.0, 2.0, 5.0]\nb = [18.0, 2.0, 5.0]\n\
                     c = [10.0, 8.0, 5.0]\nthickness = 4.0"
                }
            };
            let text = MINIMAL.replace(
                "shape = \"box\"\nmin = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]",
                &format!("shape = \"{}\"\n{body}", shape.kind()),
            );
            let parsed = Config::parse(&text).expect("parse");
            assert_eq!(parsed.domain[0].shape_spec().kind(), shape.kind());
        }
    }

    /// TOML has `nan` and `inf` literals and serde takes them into an `f64`
    /// happily. Every number in the schema is checked, in one pass, so that
    /// neither can reach the grid, the solver or anything that compares a
    /// configuration with itself.
    #[test]
    fn a_number_that_is_not_a_number_is_rejected_wherever_it_is() {
        let cases = [
            (
                "mass_fraction = 0.3",
                "mass_fraction = nan",
                "mass_fraction",
            ),
            (
                "voxel_size_mm = 2.0",
                "voxel_size_mm = inf",
                "voxel_size_mm",
            ),
            (
                "min_feature_mm = 6.0",
                "min_feature_mm = -inf",
                "min_feature_mm",
            ),
            (
                "vector = [0.0, 0.0, -10.0]",
                "vector = [0.0, nan, -10.0]",
                "vector[1]",
            ),
            (
                "center = [20.0, 5.0, 5.0], radius = 3.0",
                "center = [20.0, 5.0, 5.0], radius = nan",
                "radius",
            ),
            (
                "max = [20.0, 10.0, 10.0]",
                "max = [20.0, inf, 10.0]",
                "max[1]",
            ),
        ];
        for (from, to, key) in cases {
            let text = MINIMAL.replace(from, to);
            assert_ne!(text, MINIMAL, "the fixture no longer holds `{from}`");
            let error = match Config::parse(&text) {
                Ok(_) => panic!("{key}: parsing must fail, not succeed"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains(key), "{key}: unexpected error: {error}");
            assert!(error.contains("finite"), "{key}: unexpected error: {error}");
        }

        // A configuration built in memory rather than parsed is caught by the
        // validation instead, which is the same pass.
        let mut config = Config::parse(MINIMAL).expect("parse");
        config.optimization.min_feature_mm = f64::NAN;
        let error = config.validate_static().unwrap_err().to_string();
        assert!(error.contains("min_feature_mm") && error.contains("finite"));

        // And the checks cover the optional tables too.
        let text = format!(
            "engine = \"{}\"\n{MINIMAL}\n[growth]\nstep_mm = nan\n",
            constants::GROWTH_ENGINE
        );
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(
            error.contains("step_mm") && error.contains("finite"),
            "{error}"
        );

        let text = format!("{MINIMAL}\n[[loadcases.loads]]\ntype = \"gravity\"\ng_mm_s2 = inf\n");
        let error = Config::parse(&text).unwrap_err().to_string();
        assert!(
            error.contains("g_mm_s2") && error.contains("finite"),
            "{error}"
        );
    }

    /// The starter configuration `growforge edit` writes for a name that is
    /// not there yet has to be a *problem*: something that validates, builds
    /// and runs as written, with no warnings to explain away.
    #[test]
    fn the_starter_configuration_is_a_valid_problem_as_written() {
        let text = starter_config("brand_new_part");
        let config = Config::parse(&text).expect("the starter must parse");
        config.validate_static().expect("and validate");
        let problem = crate::problem::Problem::build(&config, Path::new("."))
            .expect("and build a discrete problem");
        assert!(
            problem.warnings.is_empty(),
            "the starter configuration warns: {:?}",
            problem.warnings
        );
        assert_eq!(config.project.name, "brand_new_part");
        assert_eq!(config.output.stl_path, "brand_new_part.stl");
        assert_eq!(config.engine_name().unwrap(), constants::STARTER_ENGINE);
        assert!(problem.counts.design > 0 && problem.counts.solid > 0);
        assert_eq!(problem.load_cases.len(), 1);
        // The load pushes down on the pad, and the floor holds the part up.
        let total: f64 = problem.load_cases[0]
            .forces
            .iter()
            .skip(2)
            .step_by(constants::DOF_PER_NODE)
            .sum();
        assert!(
            (total + constants::STARTER_LOAD_N).abs() < 1e-6,
            "the starter's load came out as {total} N"
        );
        assert!(problem.supports[0].node_count > 0);
        // And it is small enough to grow while someone watches.
        assert!(problem.grid.n_cells() < constants::VIEW_EDIT_PREVIEW_CELL_BUDGET);
    }

    #[test]
    fn unknown_material_preset_lists_the_known_ones() {
        let text = MINIMAL.replace("preset = \"pla\"", "preset = \"nylon\"");
        let err = Config::parse(&text)
            .unwrap()
            .material()
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("nylon") && err.contains("petg"),
            "unexpected error: {err}"
        );
    }
}
