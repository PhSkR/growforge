//! Every default, tolerance and tunable used by growforge.
//!
//! Nothing outside this module may hardcode a numeric policy value: a number in
//! the rest of the crate either comes from the user's configuration or from
//! here. Values that are mathematical identities (a factor of 2 in a midpoint,
//! the 8 corners of a hexahedron) are not policy and stay where they are used.
//!
//! Unit system used throughout growforge:
//! lengths in millimetres, forces in newtons, stresses in megapascals
//! (N/mm^2), torques in newton-millimetres, mass densities in g/cm^3.

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// The *machine* name: the crate, the executable, the command that is typed,
/// the folder the configurations live in. It is `CARGO_PKG_NAME` spelled out,
/// and it is what anything a machine matches on uses.
///
/// The name a user is *shown* is [`DISPLAY_NAME`]. The two are deliberately
/// different strings for one reason: a rename of the product is not a rename of
/// the command, the install directory or the release file, and holding them
/// apart is what lets the first happen without the second.
pub const PROGRAM_NAME: &str = "growforge";

/// The *display* name: what every surface a user reads calls this program -
/// window titles, both panel headings, the console banner, `--version`, the
/// header stamped into an exported STL, and the prose of a message that names
/// the program.
///
/// A deliberate literal, not a derivation. The brand carries something the
/// package name cannot - what the program makes - so it cannot be assembled
/// from `CARGO_PKG_NAME`, and a build that renamed the crate must not silently
/// rename the product.
///
/// The coupling rule, pinned by `the_display_name_extends_the_machine_name` in
/// this module's tests: **the brand may extend the machine name, never
/// contradict it.** A user who reads `growforge 3D` on a window has to be able
/// to type `growforge` at a prompt and reach it, so the displayed name starts
/// with the name the executable, the command and the install directory carry.
pub const DISPLAY_NAME: &str = "growforge 3D";

/// Crate version, used everywhere a build names itself.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Machine name and version as one string - `growforge 0.40.0`.
///
/// Kept for the surfaces that name the *binary* rather than the product; every
/// surface a user reads uses [`DISPLAY_NAME_AND_VERSION`] instead.
///
/// Assembled from the manifest at compile time. `concat!` takes literals only,
/// which is why the two halves come from the manifest rather than from
/// [`PROGRAM_NAME`] and [`VERSION`]; the strings are the same, and these cannot
/// drift from the binary.
pub const NAME_AND_VERSION: &str = concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION"));

/// Display name and version as one string - `growforge 3D 0.40.0` - and the
/// single derivation of that fragment.
///
/// Everything that names the build to a user is this constant: the line the
/// command line prints before anything else, `--version`, the prefix of both
/// window titles - which each panel draws as its own heading, so the panels name
/// the build without a second line to say it - and the header of an exported
/// STL. One constant because they have to agree: a screenshot, a pasted console
/// log and a part opened in a slicer all have to name the same build, and a run
/// cannot then be attributed to a version it was not built from.
///
/// `concat!` takes literals only, so the brand is spelled again here rather than
/// read from [`DISPLAY_NAME`]; the two are pinned to each other by the same test
/// that pins the coupling rule.
pub const DISPLAY_NAME_AND_VERSION: &str = concat!("growforge 3D", " ", env!("CARGO_PKG_VERSION"));

/// Name of the default optimization engine.
pub const DEFAULT_ENGINE: &str = "simp";

/// Name of the growth heuristic engine.
pub const GROWTH_ENGINE: &str = "growth";

/// Name of the engine that exports the domain itself.
///
/// It optimizes nothing: every design cell is filled and the part that ships is
/// the design space the configuration described, held to its analytic surfaces
/// by the same export the other engines go through. What it is for is the part
/// that was never a topology problem - a plug, a housing, a fixture drawn as
/// CSG - which before it had to be faked through SIMP at a mass fraction just
/// under one, where the optimizer redistributes material it was not asked to
/// move and the density filter erodes the boundary cells.
pub const SOLID_ENGINE: &str = "solid";

/// Backend the linear solves run on when `[solver] backend` is absent.
///
/// The compute device, because that is what the machine in front of the user
/// has and what makes a finite element run take seconds rather than minutes.
/// This default is *soft* in a way an explicit `backend = "gpu"` is not: a
/// build without the `gpu` feature, or a machine with no adapter, quietly runs
/// on the CPU instead and says so in the summary, where a configuration that
/// asked for the GPU by name still fails loudly.
///
/// What it costs is the strongest of the reproducibility promises: a default
/// run is reproducible on one machine and driver, and `backend = "cpu"` is what
/// makes it bit-for-bit reproducible everywhere, for a given build and thread
/// count. Everything in this crate that records a trajectory or a hash
/// therefore names the CPU explicitly.
pub const DEFAULT_SOLVER_BACKEND: crate::config::SolverBackend = crate::config::SolverBackend::Gpu;

// ---------------------------------------------------------------------------
// Material presets
// ---------------------------------------------------------------------------

/// A named set of isotropic material properties.
#[derive(Debug, Clone, Copy)]
pub struct MaterialPreset {
    /// Key used in `[material] preset = "..."`.
    pub key: &'static str,
    /// Young's modulus in MPa.
    pub youngs_modulus_mpa: f64,
    /// Poisson's ratio (dimensionless).
    pub poisson_ratio: f64,
    /// Mass density in g/cm^3.
    pub density_g_cm3: f64,
    /// Tensile yield strength in MPa. Stored for later phases; the phase 1
    /// compliance objective does not use it.
    pub yield_strength_mpa: f64,
}

/// Built-in filament presets. Values are typical printed-part datasheet
/// figures, not certified minima; override them per project when a supplier
/// datasheet is available.
pub const MATERIAL_PRESETS: &[MaterialPreset] = &[
    MaterialPreset {
        key: "pla",
        youngs_modulus_mpa: 2300.0,
        poisson_ratio: 0.36,
        density_g_cm3: 1.24,
        yield_strength_mpa: 50.0,
    },
    MaterialPreset {
        key: "petg",
        youngs_modulus_mpa: 2100.0,
        poisson_ratio: 0.37,
        density_g_cm3: 1.27,
        yield_strength_mpa: 47.0,
    },
    MaterialPreset {
        key: "abs",
        youngs_modulus_mpa: 2000.0,
        poisson_ratio: 0.35,
        density_g_cm3: 1.04,
        yield_strength_mpa: 40.0,
    },
];

// ---------------------------------------------------------------------------
// Configuration defaults
// ---------------------------------------------------------------------------

/// Default `[optimization] max_iterations`: enough budget to run to
/// convergence rather than a budget to be spent.
///
/// A cap is not a target. The old default of 150 was a wall clock compromise
/// from before the compute backend existed, and in the skeletal regime
/// (`mass_fraction` near 0.10, which is what this tool is for) it stopped runs
/// that were still descending: the design that came out was the 150th iterate
/// rather than the answer, and nothing said so beyond a `max change` above the
/// tolerance in the summary. A default the user has to raise on every real
/// problem is the wrong default.
///
/// What makes a thousand affordable is the device. On an RTX 3080 the shipped
/// `cantilever.toml` spends about 10 s on its 150 iterations, so a full budget is
/// a minute or two; the 1.5 mm skeletal cantilever this default was raised for
/// (58 000 cells) converges at iteration 682 in 2 min 52 s; and the expensive end
/// measured on this project, a 192 000 cell run at 1.0 mm, is about 3 min per 150
/// iterations, so a full budget there is some twenty minutes. All of them are
/// runs you start and come back to, not runs you cannot afford, and a converged
/// design at 682 iterations is worth more than a truncated one at 150.
///
/// It is a *budget*, and two things keep it from being spent for nothing. An
/// explicit `[optimization] max_iterations` still wins outright - a
/// configuration that asks for 150 gets exactly 150, which is what leaves every
/// shipped example and every recorded trajectory untouched - and a run that has
/// stopped making progress stops itself, see [`STALL_WINDOW_ITERATIONS`].
pub const DEFAULT_MAX_ITERATIONS: usize = 1000;

/// Default `[optimization] convergence_tol`: stop once the largest absolute
/// change of any design variable between iterations falls below this.
pub const DEFAULT_CONVERGENCE_TOL: f64 = 0.01;

/// Default `[output] iso_level` for the marching cubes isosurface.
pub const DEFAULT_ISO_LEVEL: f64 = 0.5;

/// Default `[output] smoothing_iterations` for Taubin smoothing.
pub const DEFAULT_SMOOTHING_ITERATIONS: usize = 10;

/// Default `[[loadcases]] weight` in the weighted compliance objective.
pub const DEFAULT_LOADCASE_WEIGHT: f64 = 1.0;

/// Default `[[loadcases.loads]] direction` of a gravity load: straight down.
pub const DEFAULT_GRAVITY_DIRECTION: [f64; 3] = [0.0, 0.0, -1.0];

/// Default `[[loadcases.loads]] g_mm_s2` of a gravity load: standard gravity in
/// millimetres per second squared (9.80665 m/s^2 rounded to the precision the
/// rest of the model deserves).
pub const DEFAULT_GRAVITY_MM_S2: f64 = 9810.0;

/// Several gravity loads in one load case add up as acceleration vectors. Their
/// sum counts as degenerate, and the load case is rejected, once it falls to
/// this fraction of the largest single contribution: the loads have cancelled,
/// and the case would claim a self weight it does not carry.
///
/// The test is relative rather than absolute so that it means the same thing for
/// a centrifuge and for micro-gravity; each individual load is separately
/// required to be non-zero when it is parsed.
///
/// The comparison is made on the naive Euclidean norm of the summed
/// acceleration, which overflows to infinity at a component magnitude of about
/// 1e155 mm/s^2. That is some 140 orders of magnitude beyond any physical use of
/// growforge, and the check is accepted as inheriting that limit rather than
/// paying for a scaled norm everywhere.
pub const MIN_NET_GRAVITY_FRACTION: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Configuration validation limits
// ---------------------------------------------------------------------------

/// Poisson's ratio must lie strictly inside this open interval for the
/// isotropic stiffness tensor to stay positive definite.
pub const POISSON_RATIO_MIN: f64 = -1.0;
/// Upper open bound on Poisson's ratio.
pub const POISSON_RATIO_MAX: f64 = 0.5;

/// A `min_feature_mm` below this many voxels is not resolvable by the density
/// filter; growforge warns (it does not fail) when the ratio is smaller.
pub const MIN_FEATURE_VOXELS_WARN: f64 = 3.0;

/// Smallest number of elements allowed along any grid axis.
pub const MIN_CELLS_PER_AXIS: usize = 1;

/// `target_cells` must ask for at least this many cells.
pub const MIN_TARGET_CELLS: usize = 1;

/// Largest voxel grid growforge will lay out, in cells.
///
/// Both resolution keys can ask for a grid no machine could hold:
/// `[resolution] target_cells` is a `usize`, so `1e19` of them is a legal
/// request, and `voxel_size_mm` is an arbitrary positive length, so a micron
/// over a 200 mm part is another. The grid is allocated the moment it is laid
/// out, and an allocation that large aborts the process - "capacity overflow",
/// no error message, no exit code worth reading - so the count is checked
/// against this budget first and rejected with an explanation.
///
/// The figure comes from the same accounting `Problem::estimated_memory_bytes`
/// does. A run keeps about [`ESTIMATED_CELL_ARRAYS`] `f64` arrays over the
/// cells, and about `ESTIMATED_SOLVER_VECTORS + ESTIMATED_VECTORS_PER_LOADCASE`
/// per load case over the degrees of freedom, which are three per node and so
/// about three per cell on a large grid; with the classification that is
/// roughly 250 bytes per cell. At this budget a single load case therefore
/// needs about 16 GiB - a large desktop's entire memory, which is what makes it
/// a ceiling rather than a recommendation. For scale: a 256 mm cube at 0.63 mm
/// voxels sits at the budget, the same cube at 1 mm voxels uses a quarter of
/// it, and the shipped examples are four to five orders of magnitude below.
pub const MAX_GRID_CELLS: usize = 64_000_000;

// ---------------------------------------------------------------------------
// Geometry / voxelization
// ---------------------------------------------------------------------------

/// A cell centre is treated as inside a signed distance field when its signed
/// distance is at or below this value (millimetres). Zero keeps the classic
/// "negative is inside" convention.
pub const INSIDE_SDF_THRESHOLD_MM: f64 = 0.0;

/// Degeneracy guard for geometry inputs: shapes thinner than this in any
/// dimension (millimetres) are rejected as degenerate.
pub const MIN_SHAPE_EXTENT_MM: f64 = 1e-9;

/// How far off the line through its two ends a tube's bend point has to be,
/// in millimetres, before the tube is a circular arc rather than a straight
/// capsule.
///
/// It is a *sagitta* threshold, so it says exactly what it costs: the arc
/// through three points whose middle one lies this far off the chord departs
/// from that chord by at most this much, anywhere along it. A micron is three
/// orders of magnitude below the finest voxel any run lays out, so a bend this
/// shallow is a tube nothing downstream could tell from the straight one -
/// while the circumcircle it would otherwise be solved into has a radius of
/// millions of millimetres and a plane normal made of rounding error. Below it
/// the tube takes the segment's own arithmetic, which is exact, cheap, and the
/// same arithmetic a tube carrying no bend at all takes.
pub const TUBE_COLLINEAR_EPS_MM: f64 = 1e-3;

// ---------------------------------------------------------------------------
// SIMP material interpolation
// ---------------------------------------------------------------------------

/// Default SIMP penalization exponent p in E(x) = Emin + x^p (E0 - Emin), used
/// when `[optimization] penalty` says nothing.
///
/// Three is the classical value and the one every recorded trajectory and every
/// shipped example was written against. Raising it prices intermediate density
/// harder and drives the field towards a black-and-white design at the cost of
/// more local minima; lowering it towards one leaves a greyer, easier problem.
pub const SIMP_PENALTY: f64 = 3.0;

/// Smallest `[optimization] penalty` a configuration may ask for.
///
/// At `p = 1` the interpolation is linear and there is no penalization at all -
/// the optimizer is free to fill the domain with intermediate density, which is
/// a valid (if useless) continuous problem. Below one, intermediate density
/// would be *rewarded* over solid material per unit volume, so the whole premise
/// of the method is inverted; that is rejected rather than run.
pub const SIMP_PENALTY_MIN: f64 = 1.0;

/// Stiffness floor of a design cell as a fraction of the solid modulus, used
/// when `[optimization] stiffness_floor` says nothing.
///
/// It is the `Emin` of `E(x) = Emin + x^p (E0 - Emin)`: what a cell the
/// optimizer has emptied still carries, and what keeps the global stiffness
/// matrix positive definite where density has collapsed to zero. Nine decades
/// below the solid modulus is the classical value and the one every recorded
/// trajectory and every shipped example was written against. Forced void cells
/// are not interpolated at all and carry a literal zero - their nodes are
/// pinned out of the system entirely - so this is a design cell's floor and
/// only a design cell's.
pub const SIMP_EMIN_FRACTION: f64 = 1e-9;

/// Lowest `[optimization] stiffness_floor` a configuration may ask for.
///
/// Three decades below the default. The stiffness contrast across the grid is
/// what the conjugate gradient's iteration count grows with, and at 1e-12 the
/// assembly already spans twelve decades against the 2.2e-16 of relative
/// precision an f64 sum carries: the system is numerically singular rather than
/// merely stiff, and a solve asked to resolve it spends its budget on rounding
/// error. Nothing is bought below here that a lower floor does not cost twice.
pub const STIFFNESS_FLOOR_MIN: f64 = 1e-12;

/// Highest `[optimization] stiffness_floor` a configuration may ask for.
///
/// Six decades above the default, and the point where the floor stops being
/// bookkeeping and starts being structure. The floor is parasitic stiffness -
/// modulus the design never bought, holding load the optimizer should have had
/// to route through material - and 1e-3 of the solid modulus is exactly what a
/// cell at density 0.1 carries at the default penalty, which is the weakest
/// path the crate already calls solvable rather than near-singular (see
/// [`STRESS_LOAD_PATH_DENSITY`]). Above it the void genuinely carries the part:
/// the compliance stops being a property of the design, and the optimizer is
/// being paid to leave holes.
pub const STIFFNESS_FLOOR_MAX: f64 = 1e-3;

// ---------------------------------------------------------------------------
// Density filter
// ---------------------------------------------------------------------------

/// The filter radius is `min_feature_mm / FILTER_RADIUS_DIVISOR`: a cone
/// filter of radius R cannot produce members thinner than about 2R.
pub const FILTER_RADIUS_DIVISOR: f64 = 2.0;

// ---------------------------------------------------------------------------
// Additive manufacturing (overhang) filter
// ---------------------------------------------------------------------------
//
// Langelaar's filter sweeps the grid layer by layer along the build direction
// and sets the printed density of an element to
// `smin(blueprint, smax(printed densities of its supporting region))`.
// The supporting region on a cubic grid is the element directly below plus its
// four lateral face neighbours in the layer below, which is what fixes the
// self-supporting angle at 45 degrees. There is no angle knob: a different
// angle needs a different stencil.
//
// M. Langelaar, "An additive manufacturing filter for topology optimization of
// print-ready designs", Struct Multidisc Optim 55 (2017) 871-883.

/// Elements in the supporting region of one element: the one directly below
/// plus its four lateral face neighbours in the layer below.
pub const AM_SUPPORT_COUNT: usize = 5;

/// Exponent `P` of the P-norm smooth maximum over the supporting region. Larger
/// is a sharper (more accurate) maximum but a stiffer function to optimize.
///
/// An integer, because the filter runs once per bisection step of the volume
/// constraint and an integer power is a handful of multiplications rather than a
/// call to `powf`.
pub const AM_SMOOTH_MAX_EXPONENT: i32 = 40;

/// Density `xi_0` at which the smooth maximum is made exact for a supporting
/// region whose members are all equal. It sets the corrected exponent
/// `Q = P + ln(n) / ln(xi_0)` used in place of `P` when the P-norm is taken, so
/// that `smax(v, ..., v) = v` holds at `v = xi_0` instead of overestimating it
/// by `n^(1/P)`. Must lie strictly between 0 and 1.
pub const AM_SMOOTH_MAX_ANCHOR: f64 = 0.5;

/// Smoothing parameter `eps` of the smooth minimum
/// `smin(a, b) = ((a + b) - sqrt((a - b)^2 + eps) + sqrt(eps)) / 2`.
/// The approximation overestimates the true minimum by at most
/// `sqrt(eps) / 2` in density units.
pub const AM_SMOOTH_MIN_EPS: f64 = 1e-4;

// ---------------------------------------------------------------------------
// Optimality criteria update
// ---------------------------------------------------------------------------

/// Maximum change of a design variable in one OC step.
pub const OC_MOVE_LIMIT: f64 = 0.2;

/// Damping exponent eta on the OC multiplicative update.
pub const OC_DAMPING: f64 = 0.5;

/// Lower bracket for the bisection on the volume Lagrange multiplier.
pub const OC_LAMBDA_LOWER: f64 = 0.0;

/// Upper bracket for the bisection on the volume Lagrange multiplier.
pub const OC_LAMBDA_UPPER: f64 = 1e9;

/// Bisection stops once the bracket's relative width drops below this.
pub const OC_BISECTION_TOL: f64 = 1e-6;

/// Bisection stops once the mean physical density over the design cells is
/// this close to the requested mass fraction.
pub const OC_VOLUME_TOLERANCE: f64 = 1e-6;

/// Hard cap on bisection steps, so a pathological objective cannot spin.
pub const OC_BISECTION_MAX_STEPS: usize = 200;

/// Lower bound of the design variable box constraint.
pub const DENSITY_MIN: f64 = 0.0;

/// Upper bound of the design variable box constraint.
pub const DENSITY_MAX: f64 = 1.0;

/// Safety margin of the self-weight sensitivity shift, as a fraction of the
/// largest `|dC/dx_e| / dV/dx_e` over the design cells.
///
/// A design-dependent load makes `dC/dx_e` positive for some elements, which the
/// multiplicative optimality criteria step cannot represent. The shift replaces
/// `dC/dx` by `dC/dx - mu dV/dx` with `mu` large enough to make every shifted
/// sensitivity strictly negative. Because the shift is a multiple of the volume
/// gradient it only renames the Lagrange multiplier (`lambda -> lambda + mu`)
/// and leaves the stationarity conditions of the volume constrained problem
/// unchanged; it damps the step rather than moving the optimum. This margin
/// keeps the largest shifted sensitivity away from zero, where the update ratio
/// would collapse the corresponding element in a single step.
pub const OC_SELF_WEIGHT_SHIFT_MARGIN: f64 = 0.05;

// ---------------------------------------------------------------------------
// Method of moving asymptotes (MMA) update
// ---------------------------------------------------------------------------
//
// The alternative to the optimality criteria step, selected with
// `[optimization] update = "mma"`. Every design variable gets a lower and an
// upper asymptote `L_e < x_e < U_e`; the objective and the volume constraint are
// replaced by the separable convex approximations
//
//   f(x) ~ f(x^k) + sum_e [ p_e / (U_e - x_e) + q_e / (x_e - L_e) ]
//   p_e = (U_e - x_e)^2 max(df/dx_e, 0),  q_e = (x_e - L_e)^2 max(-df/dx_e, 0)
//
// and the subproblem is solved through its dual, which is scalar here because
// there is exactly one constraint. Moving the asymptotes towards the current
// point damps a variable that oscillates and moving them away lets a variable
// that is making monotone progress take a longer step - which is what settles
// the near-discrete "does this column print or not" decisions that the
// self-supporting filter creates and the multiplicative optimality criteria
// ratio cannot represent.
//
// K. Svanberg, "The method of moving asymptotes - a new method for structural
// optimization", Int J Numer Meth Engng 24 (1987) 359-373.

/// Maximum change of a design variable in one MMA step.
///
/// The asymptotes already bound the step, so this is a second, absolute belt on
/// the early iterations where they are at their widest. The same figure as
/// [`OC_MOVE_LIMIT`], so switching schemes does not silently change how far one
/// iteration may travel.
pub const MMA_MOVE_LIMIT: f64 = 0.2;

/// Iterations that take the symmetric initial asymptotes rather than the
/// adaptive rule, which needs two previous design points to see a trend.
pub const MMA_INITIAL_ASYMPTOTE_STEPS: usize = 2;

/// Initial distance from the design variable to either asymptote, as a fraction
/// of the box `[DENSITY_MIN, DENSITY_MAX]`.
pub const MMA_ASYMPTOTE_INIT: f64 = 0.5;

/// Factor the asymptote distance is multiplied by when a design variable
/// reverses direction (`(x^k - x^k-1)(x^k-1 - x^k-2) < 0`). Below one, so an
/// oscillating variable is progressively damped.
pub const MMA_ASYMPTOTE_SHRINK: f64 = 0.7;

/// Factor the asymptote distance is multiplied by when a design variable keeps
/// moving the same way. Above one, so monotone progress is allowed to
/// accelerate.
pub const MMA_ASYMPTOTE_EXPAND: f64 = 1.2;

/// Smallest distance from a design variable to either asymptote, as a fraction
/// of the box. It caps how far the shrinking can damp a variable, so an
/// oscillation can never freeze one in place.
pub const MMA_ASYMPTOTE_MIN: f64 = 0.01;

/// Largest distance from a design variable to either asymptote, as a fraction
/// of the box. Far asymptotes make the approximation nearly linear, which is
/// where MMA stops being conservative.
pub const MMA_ASYMPTOTE_MAX: f64 = 10.0;

/// Fraction of the gap to an asymptote that the subproblem's own bounds keep
/// clear of it (Svanberg's `albefa`).
///
/// The approximation has a pole at each asymptote, so the trial point is never
/// allowed to reach one: the subproblem is solved over
/// `[L_e + albefa (x_e - L_e), U_e - albefa (U_e - x_e)]` intersected with the
/// box and the move limit.
pub const MMA_BOUND_ASYMPTOTE_FRACTION: f64 = 0.1;

/// Lower bracket for the bisection on the volume multiplier of the MMA
/// subproblem's dual. Zero is the unconstrained minimizer of the objective
/// approximation, which is the heaviest design the step can produce.
pub const MMA_LAMBDA_LOWER: f64 = 0.0;

/// Upper bracket for the same bisection.
///
/// Larger than [`OC_LAMBDA_UPPER`] because the multiplier means something else
/// here: it is the exchange rate between the compliance and the volume
/// fraction, so it carries the magnitude of the compliance itself (N*mm) rather
/// than a dimensionless update ratio.
pub const MMA_LAMBDA_UPPER: f64 = 1e12;

/// Bisection stops once the bracket's relative width drops below this.
pub const MMA_BISECTION_TOL: f64 = 1e-6;

/// Bisection stops once the mean physical density over the design cells is this
/// close to the requested mass fraction. The same figure as
/// [`OC_VOLUME_TOLERANCE`], so both update schemes meet the volume constraint to
/// the same accuracy.
pub const MMA_VOLUME_TOLERANCE: f64 = 1e-6;

/// Hard cap on bisection steps, so a pathological subproblem cannot spin.
pub const MMA_BISECTION_MAX_STEPS: usize = 200;

// ---------------------------------------------------------------------------
// Local volume constraint
// ---------------------------------------------------------------------------
//
// `[optimization.local_volume]`: an upper bound on the material inside every
// neighbourhood of the design, on top of the global volume target. A global
// target alone is met most cheaply by a few thick members and flush surfaces; a
// *local* cap forbids that concentration, and what comes out instead is a
// network of many thinner members with porous interiors - the bone-like
// structures the infill optimization literature produces.
//
// J. Wu, N. Aage, R. Westermann, O. Sigmund, "Infill optimization for additive
// manufacturing - approaching bone-like porous structures", IEEE Trans Vis
// Comput Graph 24 (2018) 1127-1140.

/// Default `[optimization.local_volume] max_fraction`: the largest share of
/// material any neighbourhood may hold.
///
/// Well above a typical `mass_fraction` (which the cap has to leave room for -
/// the mean of the local averages is the global one, so a cap at or below the
/// global target is infeasible by construction and rejected) and well below one,
/// which is what leaves the optimizer a porous interior to build rather than a
/// solid one.
pub const LOCAL_VOLUME_MAX_FRACTION: f64 = 0.6;

/// Default `[optimization.local_volume] radius_mm`, as a multiple of the density
/// filter radius `min_feature_mm / FILTER_RADIUS_DIVISOR`.
///
/// The cap has to be measured over a neighbourhood *larger* than the smallest
/// feature, or it would price a single member's own cross section rather than
/// how the members are spread: at one filter radius every solid member violates
/// it and the only feasible design is grey. Three radii holds a few members and
/// the gaps between them, which is the scale the constraint is about.
pub const LOCAL_VOLUME_RADIUS_FACTOR: f64 = 3.0;

/// Exponent of the p-mean that aggregates the per-cell local fractions into the
/// one scalar the constraint is stated on.
///
/// One constraint rather than one per cell is what keeps the subproblem a dual
/// in two multipliers instead of thousands. The p-mean
/// `((1/N) sum_e f_e^P)^(1/P)` approaches the worst cell from *below* as `P`
/// grows, so the aggregate is an under-estimate of the true worst local
/// fraction and the design can exceed `max_fraction` locally by the gap: at this
/// exponent a field where a tenth of the cells sit at the peak reads about 13 %
/// low, and one where a third do about 7 %. Raising `P` closes that gap and
/// sharpens the aggregate into a maximum the gradient can only see one cell of,
/// which is what makes the constraint jump between cells and the run oscillate;
/// sixteen is the usual compromise, and the run reports the true worst fraction
/// beside the aggregate so the gap is never hidden.
pub const LOCAL_VOLUME_AGGREGATION_EXPONENT: f64 = 16.0;

/// Residual either constraint of the two-constraint MMA dual is solved to.
///
/// Both are measured in the same units - a fraction of a neighbourhood, a
/// fraction of the design space - so one figure serves both, and it is the same
/// one the single-constraint bisection uses ([`MMA_VOLUME_TOLERANCE`]).
pub const LOCAL_VOLUME_DUAL_TOLERANCE: f64 = 1e-6;

/// Cap on the alternating sweeps of the two-constraint dual.
///
/// The dual is maximized by exact coordinate ascent: one multiplier is bisected
/// with the other held, then the other, until a sweep moves neither. Each
/// coordinate bisection is the single-constraint one and runs under the same
/// [`MMA_BISECTION_MAX_STEPS`] and [`MMA_BISECTION_TOL`] budget. Fifty sweeps is
/// far past what a two variable concave problem needs - the measured runs settle
/// in two or three - and exists so that a pathological subproblem cannot spin.
pub const LOCAL_VOLUME_DUAL_MAX_SWEEPS: usize = 50;

/// Stencil taps above which the local kernel is warned about.
///
/// The kernel is a second density filter at `radius_mm`, and its cost per
/// iteration is the cell count times this. At the default three filter radii it
/// is about 27 times the density filter's own taps, which is affordable; a
/// radius a few times larger than that is not, and the run would spend its time
/// in the constraint rather than in the analysis. Five thousand taps is about a
/// ten voxel reach.
pub const LOCAL_VOLUME_STENCIL_TAPS_WARN: usize = 5000;

// ---------------------------------------------------------------------------
// Stall detection
// ---------------------------------------------------------------------------
//
// Some problems never settle. The documented one is the 120 mm deep variant of
// the shipped shelf bracket under `oc` (see the update scheme section of the
// README): every one of its 400 iterations takes a step of exactly the 0.2 move
// limit while the compliance wanders inside a band, and it would do the same for
// 4000. Nothing about a single iteration tells that run from one that is
// converging slowly - both are pinned at the move limit - so the test below is
// over a *window* of iterations, and it asks two things of every one of them.
//
// The measurements the thresholds come from are six runs, all of them traced
// iteration by iteration with the test switched off. Each row is that run's
// *worst* 100-iteration window, meaning the one that came closest to being
// called a stall - the smallest design variable change it ever held for a whole
// window, as a fraction of the move limit, and what the compliance bought over
// that window:
//
//   run                                    outcome            worst window
//   deep bracket, oc  (the non-settler)    400 = the cap      100 %, +0.2 %
//   deep bracket, mma                      converged at 267    57 %, +0.4 %
//   skeletal cantilever 1.5 mm, oc, 0.10   converged at 682    53 %, +3.3 %
//   mbb_bridge.toml                        150 = its cap       29 %, +0.5 %
//   cantilever.toml                        150 = its cap       15 %, +0.3 %
//   shelf_bracket.toml, mma                converged at 132    14 %, +0.3 %
//
// The separation is in the first column, not the second: a run that will not
// settle spends *every* iteration clipped by the move limit, and a run that is
// working towards an answer - even one taking hundreds more iterations to get
// there - lets the change off the limit long before it arrives. The compliance
// column is the second leg rather than the first because it cannot separate
// these on its own: a converging run's last hundred iterations buy just as
// little as a stalled run's.
//
// The compliance leg is asked as "did the second half of the window better the
// first half's best", which is what makes it immune to the wandering: a
// descending run keeps setting new lows through its oscillation, and a wandering
// one stops setting them.
//
// The bias is deliberately asymmetric. A stall that is *not* caught spends the
// rest of the budget and stops at the cap, which is exactly what growforge did
// before this existed; a converging run falsely called stalled loses the design
// it was on its way to. So the thresholds sit between the two columns above with
// room on the converging side, and anything the test cannot read - a non-finite
// number, a window that is not full - reads as "not stalled".

/// Iterations the stall test looks back over.
///
/// A hundred consecutive iterations of no progress is the claim being made, and
/// it has to be long enough that nothing a converging run does inside one can
/// fill it. Measured on the traces above it is: at 100 the test fires on the
/// non-settler at iteration 255 and on nothing else, at 50 it fires at 208 (a
/// weaker claim for 47 iterations of budget), and at 150 not until 394, which on
/// a problem whose whole point is that it never ends is most of a run.
///
/// It is also what a stall costs before it is called, which at the default
/// [`DEFAULT_MAX_ITERATIONS`] is a tenth of the budget.
///
/// The window is split into halves of `STALL_WINDOW_ITERATIONS / 2`, so an even
/// number keeps the two comparable.
pub const STALL_WINDOW_ITERATIONS: usize = 100;

/// Fraction of the update scheme's move limit that every iteration of the window
/// has to have reached for the design to count as still traversing.
///
/// This is the leg that separates the cases. A step clipped by the move limit is
/// the optimizer asking to move further than it is allowed; a hundred of them in
/// a row, with nothing to show for it, is a run that cannot represent the
/// decision it is being asked to make - which is exactly what the
/// self-supporting filter's near-discrete "does this column print" decisions do
/// to the multiplicative optimality criteria ratio.
///
/// Four fifths sits between the two documented regimes with room on the side
/// that matters: the non-settlers hold 100 % of the move limit for hundreds of
/// iterations, while the closest any converging run measured came was 57 % - the
/// deep bracket under `mma` around iteration 188, on its way to converging at
/// 267. Raising this costs nothing but missed stalls (which are the old
/// behaviour); lowering it below about 0.6 would have cut that run off.
pub const STALL_MOVE_LIMIT_FRACTION: f64 = 0.8;

/// Relative compliance improvement the second half of the window has to make on
/// the first half's best for the run to count as still buying something.
///
/// Half a percent over a hundred iterations. What it has to be above is the
/// *noise* of a wandering trajectory: the non-settler's compliance moves inside a
/// 2.7 % band with no trend, and the best-of-fifty statistic that comes out of
/// that drifts by 0.2 % to 1.0 % per window in either direction, so a threshold
/// under 0.3 % would never fire at all and one over 2 % would fire while the
/// wander was still real. What it has to be below is what a converging run buys
/// while it is pinned at the move limit, and that is 2.9 % per window at its
/// slowest and 12.8 % at its fastest - a factor of six of headroom.
pub const STALL_COMPLIANCE_GAIN: f64 = 0.005;

/// Relative decrease of the design variable change that the second half of the
/// window has to make on the first half's smallest for the design to count as
/// still settling.
///
/// The second safeguard on the traversing leg, for the run that is pinned near
/// the move limit but working its way off it: a change falling by more than a
/// twentieth per window is going somewhere, and that run keeps its budget. It
/// cannot be much tighter than this and still mean anything - a change decaying
/// this slowly would need `ln(1/20) / ln(1 - 0.05)` windows, some 5800
/// iterations, to reach the default `convergence_tol` from the move limit. The
/// documented non-settlers sit at exactly 0.0: pinned, iteration after
/// iteration.
pub const STALL_CHANGE_DECAY: f64 = 0.05;

// ---------------------------------------------------------------------------
// Guide wireframe
// ---------------------------------------------------------------------------
//
// The optional `[optimization.wireframe]` seeds a SIMP run with a thin connected
// network of struts joining every load region, every support region and every
// `[[keepin]]` entry, and holds that network as a density floor for the first
// iterations of the run. It is a guide and nothing else: once the hold is
// released the optimizer may keep, thicken, move or dissolve the wire, and
// nothing pins it into the exported part. The numbers below are therefore about
// making the guide *visible* to the optimizer - thick enough for the density
// filter to resolve, held long enough for a load path to form along it - rather
// than about the geometry of the part.

/// Default `[optimization.wireframe] radius_mm`: radius of the capsule the guide
/// is rasterized as, in millimetres.
///
/// A 5 mm wire, which is the order of a typical `min_feature_mm`, so the guide is
/// a feature the density filter can resolve rather than one it smears away in
/// the first iteration. A wire much thinner than the filter radius arrives at the
/// analysis as a faint smudge that carries nothing.
pub const WIREFRAME_RADIUS_MM: f64 = 2.5;

/// Default `[optimization.wireframe] hold_iterations`: iterations the wire is
/// applied as a density floor for, counted from the first.
///
/// Long enough for the compliance to have made use of the load path the wire
/// offers: past that the guide is either being built on or being competed with,
/// and either way it has said what it had to say. Zero is legal and seeds the
/// wire without holding it.
///
/// What the length costs is the delay it puts in front of every verdict about a
/// settling design, because neither of them is asked while the floor is on (see
/// [`crate::engine::wireframe`]): a stall cannot be called before
/// `hold_iterations + STALL_WINDOW_ITERATIONS`. At well under
/// [`STALL_WINDOW_ITERATIONS`] that delay is a fraction of the window it delays,
/// and both stay a small part of [`DEFAULT_MAX_ITERATIONS`].
pub const WIREFRAME_HOLD_ITERATIONS: usize = 40;

/// Default `[optimization.wireframe] seed_density`: the density the wire cells
/// are seeded at and held at.
///
/// Solid, [`DENSITY_MAX`]. The wire is a load path offered to the optimizer, and
/// at half density it offers an eighth of the stiffness - the SIMP penalty is a
/// cube - which is not an offer.
pub const WIREFRAME_SEED_DENSITY: f64 = DENSITY_MAX;

/// Spacing of the nodes put back along a simplified wire path, in wire radii.
///
/// The shortcut pass collapses a voxel path into as few segments as the domain
/// allows, and the rasterizer scans the bounding box of every segment it draws,
/// so one segment across the whole domain makes every cell of that box a
/// candidate. Nodes two radii apart keep the scan local for the cost of a few
/// nodes, and every inserted node lies on a straight line the shortcut already
/// proved clear, so the path itself does not move.
pub const WIREFRAME_RESAMPLE_RADII: f64 = 2.0;

/// Rasterized density above which a cell counts as part of the wire.
///
/// The smoothstep the capsule distance is mapped through takes exactly this value
/// at the capsule surface, so the cells above it are the ones whose centre lies
/// inside the wire.
pub const WIREFRAME_CELL_THRESHOLD: f64 = 0.5;

// ---------------------------------------------------------------------------
// Growth engine
// ---------------------------------------------------------------------------
//
// The growth engine is a heuristic: it routes struts from the loads to the
// supports, hangs an organic canopy off them, thickens everything by the load
// each branch carries and rasterizes the result. It never evaluates a
// compliance. The lengths below are expressed as multiples of the user's own
// `min_feature_mm`, because that is the one length in the configuration that
// states the scale of the part's features; an absolute millimetre default would
// be meaningless on a part ten times larger or smaller. All of them can be
// overridden in absolute millimetres from the `[growth]` table.

/// Default `[growth] seed`. Any fixed value does; this one is the fractional
/// part of the golden ratio in 64 bits, a conventional non-degenerate constant.
pub const DEFAULT_GROWTH_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Multiplier of the 64 bit LCG behind PCG32, from O'Neill's reference
/// implementation.
pub const PCG32_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// Sequence selector ("stream") the growth engine's PCG32 runs on. Two
/// generators with the same seed but different sequences produce unrelated
/// streams; growforge only ever needs one, so it is fixed here rather than
/// exposed.
pub const PCG32_STREAM: u64 = 54;

/// Divisor that turns a PCG32 draw into a `f64` in `[0, 1)`: 2^32.
pub const PCG32_F64_DIVISOR: f64 = 4_294_967_296.0;

/// Default `[growth] attractor_per_cm3`: how many space colonization attraction
/// points are scattered per cubic centimetre of design domain. About one per
/// kill-radius cube is what makes a branch pattern rather than a bush.
pub const DEFAULT_ATTRACTOR_PER_CM3: f64 = 5.0;

/// Default `[growth] step_mm`, as a multiple of `min_feature_mm`: the distance
/// one space colonization step advances a branch tip.
pub const GROWTH_STEP_PER_MIN_FEATURE: f64 = 0.75;

/// Safety margin on the default `[growth] kill_radius_mm`.
///
/// The kill radius is what decides how much skeleton there will be, and
/// therefore how heavy the result is before any thickening. Branches grow until
/// every attraction point has been consumed, which is when the tube of the kill
/// radius around them has swept the design domain: a domain of volume `V` ends
/// up with about `L = V / (pi * k^2)` of skeleton, occupying
/// `(r_min / k)^2` of the domain at the minimum strut radius. Reading that
/// backwards, `k = r_min / sqrt(mass_fraction)` is the kill radius whose bare
/// skeleton weighs exactly what was asked for, and this margin on top of it
/// leaves the thickening room to taper rather than starting already clamped.
/// The default kill radius is therefore
/// `min_feature_mm / 2 * MARGIN / sqrt(mass_fraction)`.
///
/// The margin is generous because the estimate above is optimistic on two
/// counts: the smoothstep at the surface adds about half a voxel to every
/// radius, and branches that pass close to each other sweep the same space
/// twice, so the skeleton comes out longer than one pass over the domain.
/// Measured on the shipped example, the two together cost a factor of about
/// three in volume, and this margin leaves the bare skeleton near a third of
/// the requested mass fraction.
pub const GROWTH_KILL_MASS_FRACTION_MARGIN: f64 = 2.5;

/// Default `[growth] attraction_radius_mm`, as a multiple of the kill radius:
/// how far a branch tip can see an attractor. Must stay above one, or every
/// attractor is consumed before it can steer anything.
pub const GROWTH_ATTRACTION_PER_KILL: f64 = 3.0;

/// Default `[growth] max_radius_mm`, as a multiple of `min_feature_mm`: the
/// cap on a single strut's radius.
pub const GROWTH_MAX_RADIUS_PER_MIN_FEATURE: f64 = 3.0;

/// Default `[growth] murray_exponent`. Murray's law `r_parent^n = sum r_child^n`
/// is `n = 3` for laminar flow in pipes; measured biological and structural
/// branchings sit lower, and 2.6 is the usual compromise for load-carrying
/// (rather than fluid-carrying) branches.
pub const DEFAULT_MURRAY_EXPONENT: f64 = 2.6;

/// Legal range of `[growth] murray_exponent`. Below 1 a parent would be thinner
/// than its children; above this the taper is numerically indistinguishable
/// from a step.
pub const MURRAY_EXPONENT_MIN: f64 = 1.0;
/// Upper bound of the legal `[growth] murray_exponent` range.
pub const MURRAY_EXPONENT_MAX: f64 = 8.0;

/// Default `[growth] max_steps`: cap on space colonization iterations. One
/// iteration advances every branch tip that can see an attractor by one step.
pub const DEFAULT_GROWTH_MAX_STEPS: usize = 400;

/// A `[growth] step_mm` shorter than this many voxels cannot leave the cell it
/// started in and would spin without growing, so it is rejected.
pub const GROWTH_MIN_STEP_VOXELS: f64 = 0.25;

/// Spacing, in voxels, of the samples along a candidate segment when it is
/// tested against the allowed cells. Half a voxel cannot skip over a cell.
pub const GROWTH_SEGMENT_SAMPLE_VOXELS: f64 = 0.5;

/// Consecutive blocked steps after which a branch tip stops growing for good.
pub const GROWTH_MAX_FAILED_STEPS: usize = 3;

/// Rejection sampling attempts per attraction point before the scatter gives up
/// on a domain whose design cells are a vanishing fraction of its bounding box.
pub const GROWTH_ATTRACTOR_PLACEMENT_ATTEMPTS: usize = 64;

/// How many times the shortcut pass halves its line-of-sight radius, looking for
/// one the unsimplified voxel path itself already satisfies.
///
/// The pass asks that a shortcut keep a strut of the minimum radius inside the
/// allowed cells. In a domain thinner than that radius no shortcut could ever
/// pass, so the radius is backed off until the original path clears it; at the
/// end of the ladder the test is the bare centre line, which every voxel path
/// satisfies by construction. A shortcut is therefore never worse than the path
/// it replaces.
pub const GROWTH_SIMPLIFY_RADIUS_BACKOFFS: usize = 6;

/// Blend radius of the smooth minimum that unions the strut capsules, in
/// voxels. It rounds the junctions instead of creating a crease; larger values
/// add material at every junction.
pub const GROWTH_SMOOTH_UNION_VOXELS: f64 = 0.35;

/// Width, in voxels, of the smoothstep that turns the skeleton's signed distance
/// into a density. One voxel puts the whole transition inside the band marching
/// cubes interpolates across anyway.
pub const GROWTH_SURFACE_WIDTH_VOXELS: f64 = 1.0;

/// Share of the total load magnitude that the grown canopy carries collectively,
/// spread equally over the branch tips that are not themselves load attachment
/// points.
///
/// Such a tip ends on a support or on a keepin that carries no load of its own.
/// It still has to hold its own branch up, and it still takes whatever the loads
/// spread into it, so giving it a share of the real load is what makes the
/// canopy taper like the backbone instead of collapsing to the minimum feature
/// everywhere.
pub const GROWTH_BRANCH_TIP_LOAD_FRACTION: f64 = 0.25;

/// Default `[growth] prune`: remove branches that end on nothing.
///
/// A space colonization branch stops where the attraction points run out, which
/// leaves a free tip hanging in mid air. Such a tip carries no load - it is dead
/// mass in a tool whose whole point is weight - reads as an unfinished model,
/// and is an unsupported overhang no printer can lay down. Pruning them is
/// therefore the default; `prune = false` keeps them for anyone who wants the
/// decorative growth for its own sake.
pub const DEFAULT_GROWTH_PRUNE: bool = true;

/// How close a branch tip has to come to a structural surface to count as fused
/// to it, as a multiple of the smallest strut radius `r = min_feature_mm / 2`.
///
/// It is **deliberately below one**, and the arithmetic is the whole point.
///
/// A tip accepted at exactly `r` would put the anchor's nearest point exactly on
/// the capsule's surface, where the signed distance is zero and the density is
/// exactly the default iso level. That is tangency, not overlap: the two solids
/// touch at a single point that the cell-centre sampling need not even land on,
/// and what comes out of marching cubes can be two separate watertight shells - a
/// "fused" tip shipping as a floating chunk, which the mesh validation (which
/// checks manifoldness, not connectedness) would not catch.
///
/// At `f < 1` the capsule reaches `(1 - f) * r` *past* the anchor's boundary, so
/// the penetration depth is strictly positive. The intervening cells are above
/// the iso level too: a cell whose interior the line from the tip to the anchor
/// crosses has its centre at most `h * sqrt(3) / 2` off that line and at most
/// `f * r` along it, so it lies within `sqrt((f r)^2 + 3 h^2 / 4)` of the tip,
/// which stays strictly inside the capsule whenever
/// `r > h * sqrt(3) / 2 / sqrt(1 - f^2)`. At `f = 0.8` that is `r > 1.44 h`,
/// which any configuration `growforge` does not already warn about satisfies:
/// [`MIN_FEATURE_VOXELS_WARN`] asks for `min_feature_mm >= 3 h`, and therefore
/// for `r >= 1.5 h`.
///
/// Anything further away than `f * r` is a branch that ends on nothing, whatever
/// it looks like from a distance.
pub const GROWTH_FUSION_RADII: f64 = 0.8;

/// How long an attraction point waits for something to come nearer before it is
/// abandoned as unreachable, as a multiple of the time a branch would take to
/// cross the whole attraction radius (`attraction_radius_mm / step_mm` steps).
///
/// A point behind a wall, or on a surface no branch can get at, would otherwise
/// drag a branch after it for the entire step budget: the branch never arrives,
/// so the point is never consumed, and the growth spends itself on a member that
/// ends up pruned. The distance from a point to the nearest branch that can
/// still grow only ever falls, so "nothing came any closer for this long" is
/// exactly "nothing is coming".
///
/// The budget is expressed as travel rather than as a step count because that is
/// what it has to cover: a point at the far edge of its own attraction radius is
/// legitimately that many steps away, and a fixed count would give up on it
/// while the branch was still on its way.
pub const GROWTH_ATTRACTOR_PATIENCE_TRAVELS: f64 = 1.0;

/// Spacing of the attraction points seeded on the structural surfaces, as a
/// multiple of the kill radius.
///
/// These are the points that turn growth into *connecting*: unlike an interior
/// point, one of them is only consumed when a branch has actually reached the
/// surface it sits on, so a branch keeps growing until it fuses. One per kill
/// ball footprint matches the spacing the interior growth settles at anyway,
/// and it is what bounds how many branches have to make the whole journey.
pub const GROWTH_SURFACE_ATTRACTOR_SPACING_KILLS: f64 = 1.0;

/// Relative tolerance on the rasterized volume fraction when the radius scale is
/// bisected against the requested `mass_fraction`.
pub const GROWTH_VOLUME_TOLERANCE: f64 = 1e-3;

/// Hard cap on bisection steps of the radius scale.
pub const GROWTH_VOLUME_BISECTION_MAX_STEPS: usize = 40;

/// Colonization iterations between two progress lines and density snapshots.
/// Bounds what a live viewer costs a run that nobody is watching.
pub const GROWTH_REPORT_INTERVAL_STEPS: usize = 4;

// ---------------------------------------------------------------------------
// Growth symmetry
// ---------------------------------------------------------------------------
//
// `[growth.symmetry]` grows one fundamental domain and replicates exact copies
// of it, so a symmetric problem can produce a perfectly symmetric part instead
// of four differently scattered legs. Everything below bounds what may be
// asked for; the semantics live in `engine::growth::symmetry`.

/// Mirror planes `[growth.symmetry] planes` may name: one halves the domain,
/// two quarter it. A third would leave an eighth of the domain and a part whose
/// every octant is the same, which is a lattice rather than a structure, and it
/// is the point at which the fundamental domain stops holding enough of the
/// problem to route a load path through.
pub const GROWTH_SYMMETRY_MAX_PLANES: usize = 2;

/// Smallest `[growth.symmetry] order`: two-fold, the rotational counterpart of
/// a single mirror plane. One would be no symmetry at all.
pub const GROWTH_SYMMETRY_MIN_ORDER: usize = 2;

/// Largest `[growth.symmetry] order`. A twelfth of the domain is already a
/// sector too thin for a canopy at any usable voxel size, and the sector's
/// volume - and with it the part of the volume budget the fundamental domain
/// gets to spend - falls as `1 / order`.
pub const GROWTH_SYMMETRY_MAX_ORDER: usize = 12;

/// Axis `[growth.symmetry] axis` defaults to for a rotational symmetry: z, the
/// up axis every other default in growforge is written against.
pub const DEFAULT_SYMMETRY_AXIS: crate::config::Axis = crate::config::Axis::Z;

/// How far past the boundary of the fundamental domain a point may sit and
/// still count as inside it, in voxels.
///
/// It decides nothing but the sign of a rounding error, and it has to exist.
/// The two points this is asked about most are a cell centre lying *on* a mirror
/// plane - which happens to a whole layer whenever an axis has an odd number of
/// cells - and the centre of a region straddling the plane, which is where a
/// centred load or a central support sits. Both are exactly on the boundary in
/// arithmetic and a few parts in 1e13 to either side of it in floating point,
/// and a middle layer that belonged to neither half would be a one-voxel gap
/// through the part while a centred load that belonged to no sector would fail
/// the run outright.
///
/// A millionth of a voxel is four orders of magnitude above the rounding it
/// absorbs (a centroid over thousands of cells is accurate to about 1e-9 mm)
/// and six below anything a user can express, so it can never decide a question
/// about geometry: two *distinct* regions would have to sit this close to the
/// boundary on either side of it to be owned by the same sector.
pub const GROWTH_SYMMETRY_BOUNDARY_EPSILON_VOXELS: f64 = 1e-6;

/// How far a region's centre may sit from the image of another region's centre,
/// in voxels, before the run warns that the problem is not symmetric under the
/// declared transforms.
///
/// Two voxels rather than one because the centre compared is the centroid of the
/// *cells* a region selects, and voxelization can move that by a fraction of a
/// voxel on each side of a transform that does not map cell centres onto cell
/// centres - which is every rotation that is not a quarter turn on a square
/// grid. A real asymmetry is a region in the wrong place, which is voxels or
/// tens of voxels out, not fractions of one.
pub const GROWTH_SYMMETRY_REGION_TOLERANCE_VOXELS: f64 = 2.0;

// ---------------------------------------------------------------------------
// Linear solver
// ---------------------------------------------------------------------------

/// Conjugate gradient stops at this relative residual ||r|| / ||b||.
///
/// The default of `[solver] tolerance`, which is what the optimization loop's
/// solves are taken to when the configuration says nothing. Tight on purpose:
/// it is the value every recorded trajectory and every determinism hash was
/// written against, so a run that does not name one reproduces them.
pub const CG_RELATIVE_TOLERANCE: f64 = 1e-8;

/// Tightest `[solver] tolerance` a configuration may ask for.
///
/// The residual is formed in f64, whose relative precision is 2.2e-16, and a
/// conjugate gradient's *recursive* residual drifts from the true one by orders
/// of magnitude more than that over the thousands of iterations a stiff system
/// takes. Two decades below the default is as far as the arithmetic reaches
/// before the iteration cap becomes the thing that ends the solve; the finite
/// difference gradient check goes tighter still, and does it in code where a
/// solve that fails is the test failing.
pub const CG_TOLERANCE_MIN: f64 = 1e-10;

/// Loosest `[solver] tolerance` a configuration may ask for.
///
/// Four decades above the default, and two above the point
/// ([`STRESS_CG_TOLERANCE`]) where the displacement error is already far below
/// what one trilinear hexahedron per voxel resolves. Past this the compliance
/// history itself starts to carry the solver's error rather than the design's,
/// and an optimizer reading noise for a gradient is not a cheaper run - it is a
/// different problem.
pub const CG_TOLERANCE_MAX: f64 = 1e-4;

/// Hard iteration cap for the conjugate gradient solver.
///
/// A guardrail rather than a knob, which is why it is not a configuration key:
/// what a run needs from it is that a solve which has stopped converging ends.
pub const CG_MAX_ITERATIONS: usize = 10_000;

/// Diagonal entries at or below this are replaced by `JACOBI_DIAGONAL_FLOOR`
/// so the Jacobi preconditioner can never divide by zero.
pub const JACOBI_DIAGONAL_EPS: f64 = 0.0;

/// Replacement value for non-positive diagonal entries (an identity row).
pub const JACOBI_DIAGONAL_FLOOR: f64 = 1.0;

/// Number of element colours in the race-free scatter partition: elements are
/// grouped by the parity triple (i%2, j%2, k%2), so elements in one colour
/// share no node.
pub const ELEMENT_COLORS: usize = 8;

/// Conjugate gradient iterations between two cancellation checks.
///
/// The arithmetic, on the smallest grid worth solving at all (a thousand
/// elements): one iteration is one matrix-vector product, and the matrix-free
/// operator spends `DOF_PER_ELEMENT^2 = 576` multiply-adds per element on it,
/// so an iteration costs at least 6e5 floating point operations - a hundred
/// microseconds of one core, and far more on a real grid. A check is one relaxed
/// atomic load, about a nanosecond. Asking every 32 iterations therefore costs
/// on the order of 1e-10 of the solve while bounding the latency of a stop at 32
/// iterations, which is milliseconds on that smallest grid and well under a
/// second on the largest one an editor previews. Both halves of the trade are
/// decades clear, so the value is not sensitive: anything from 8 to 256 would
/// do, and 32 is the middle of that range.
pub const CG_CANCEL_CHECK_INTERVAL: usize = 32;

// ---------------------------------------------------------------------------
// GPU conjugate gradient
// ---------------------------------------------------------------------------
//
// Consumed by the `gpu` feature only, and by the shader it drives. WGSL has no
// f64, so the device-side Krylov recurrence runs in f32 and is wrapped in a
// double precision iterative refinement loop on the host: the residual is
// always formed with the same f64 operator the CPU backend uses, and the GPU
// only ever solves for a correction. That is what lets a single precision inner
// solve reach the same relative residual the CPU path promises.
//
// The values below live here rather than in the shader so that the single "no
// policy number outside constants.rs" rule keeps holding. The handful the
// shader has to know are mirrored by WGSL `const` declarations, and
// `fea::gpu::tests::the_shader_mirrors_the_configured_constants` fails if the
// two ever drift apart.

/// Threads per compute workgroup. 64 is a whole number of warps on every
/// vendor and keeps the per-thread register pressure of the element gather low
/// enough to stay resident.
pub const GPU_WORKGROUP_SIZE: u32 = 64;

/// Workgroups dispatched by every grid-strided kernel.
///
/// Fixed rather than derived from the problem size, because it is also the
/// number of partial sums a reduction produces: holding it constant is what
/// makes the reductions bit-reproducible on a given device for a given problem.
pub const GPU_DISPATCH_WORKGROUPS: u32 = 1024;

/// Slot of `r^T z` in the shader's scalar buffer.
pub const GPU_SCALAR_RZ: usize = 0;
/// Slot of `p^T K p`.
pub const GPU_SCALAR_PAP: usize = 1;
/// Slot of the conjugate gradient step length.
pub const GPU_SCALAR_ALPHA: usize = 2;
/// Slot of the search direction update factor.
pub const GPU_SCALAR_BETA: usize = 3;
/// Slot of `||r||^2`, the value the host reads back to test convergence.
pub const GPU_SCALAR_RESIDUAL_SQ: usize = 4;
/// Slot of the breakdown flag, set by the shader when `p^T K p` is not
/// positive. Non-zero means the device-side iteration has stopped advancing.
pub const GPU_SCALAR_BREAKDOWN: usize = 5;
/// Number of slots in the shader's scalar buffer.
pub const GPU_SCALAR_COUNT: usize = 6;

/// Tightest relative residual a single precision inner solve is ever asked for.
///
/// Two decades above the f32 epsilon of 1.2e-7, so the inner solve stops where
/// rounding rather than the Krylov space limits it. Tightening it past the floor
/// only burns iterations that no longer reduce the error. A pass that needs less
/// than this asks for less: see [`GPU_REFINEMENT_TARGET_MARGIN`].
pub const GPU_INNER_TOLERANCE: f32 = 1e-5;

/// Safety factor on the accuracy one refinement pass asks the device for.
///
/// The right hand side of an inner solve has unit length, so the factor the
/// outer residual still has to be multiplied by is also the inner residual that
/// would achieve it. Asking for exactly that would leave a pass landing on its
/// target and needing another one, so it is asked for a decade tighter - and no
/// tighter than [`GPU_INNER_TOLERANCE`], which is all single precision can
/// deliver. This is what stops a late pass from grinding on long after its work
/// is done: the residual an inner solve leaves is a hard right hand side for the
/// next one, and its attainable floor rises pass by pass.
pub const GPU_REFINEMENT_TARGET_MARGIN: f64 = 0.1;

/// Hard iteration cap on one single precision inner solve.
pub const GPU_INNER_MAX_ITERATIONS: usize = 20_000;

/// The caller's iteration cap is multiplied by this to bound the device
/// iterations of one solve.
///
/// A refinement pass restarts the Krylov space, so the device spends two to
/// three times the iterations a single double precision conjugate gradient
/// would to reach the same residual - iterations that are an order of magnitude
/// cheaper each. Scaling the cap is what makes "at most `CG_MAX_ITERATIONS`"
/// express the same amount of patience on both backends rather than cutting the
/// GPU off part way through work the CPU would have been allowed to finish.
pub const GPU_ITERATION_BUDGET_FACTOR: usize = 4;

/// Device-side iterations between two residual readbacks.
///
/// A readback is a pipeline stall, so it is amortised over a batch of
/// iterations recorded into one command buffer; the cost of overshooting the
/// target by up to this many iterations is far below the cost of stalling every
/// iteration.
pub const GPU_RESIDUAL_CHECK_INTERVAL: usize = 25;

/// A readback interval counts as progress only when it cuts the best residual
/// seen so far by at least this factor.
///
/// Without a margin, the jitter of a solve that has already hit the single
/// precision floor keeps setting marginal new minima and the stall test below
/// never fires.
pub const GPU_INNER_STALL_GAIN: f64 = 0.95;

/// The stall test above is only armed while the residual is below this fraction
/// of the one the solve started from (which is 1 by construction, the right hand
/// side being normalised).
///
/// A Jacobi preconditioned elasticity system spends its first hundreds of
/// iterations in a plateau where the 2-norm residual sits *above* where it
/// began - measured on the 857k degree of freedom cantilever it peaks at 7.5
/// times its starting value around iteration 400 and only engages near
/// iteration 500. A solve up there is in its plateau, not at its floor, and
/// there is nothing to call a stall against; until it comes back down the
/// iteration budget is the only limit. A genuine single precision floor is
/// decades below this, so the guard can never hide one.
pub const GPU_INNER_ENGAGED_RESIDUAL: f64 = 0.5;

/// Consecutive armed readback intervals without progress before an inner solve
/// gives up and hands what it has to the refinement loop.
///
/// Generous, because even below [`GPU_INNER_ENGAGED_RESIDUAL`] the residual
/// keeps bouncing: measured on the cantilever above, a converging solve goes as
/// many as four armed intervals without a new best. Eight leaves a factor of two
/// of headroom while still capping the waste of a genuinely stalled solve at
/// `GPU_INNER_STALL_INTERVALS * GPU_RESIDUAL_CHECK_INTERVAL` iterations.
pub const GPU_INNER_STALL_INTERVALS: usize = 8;

/// Hard cap on double precision refinement passes per linear solve.
pub const GPU_REFINEMENT_MAX_PASSES: usize = 30;

/// A refinement pass must cut the residual to at most this fraction of what it
/// started with. A pass that does not has hit the single precision floor, and
/// repeating it would spin rather than converge.
pub const GPU_REFINEMENT_MIN_GAIN: f64 = 0.5;

/// Storage buffers the shader binds across its two bind groups. An adapter that
/// allows fewer per shader stage cannot run it.
pub const GPU_STORAGE_BUFFERS: u32 = 12;

/// The multiplicative identity, handed to the shader through its uniform block.
///
/// Not a tunable and not a fudge factor: it is exactly one, and `pcg.wgsl`
/// multiplies by it in the one place a compensated sum has to survive the
/// shader compiler - the solution accumulator. A compensation term is
/// algebraically zero and stays non-zero only because floating point addition
/// does not associate, so a compiler that reassociates deletes it, which the
/// Vulkan compiler on an RTX 3080 does: before this existed the accumulator's
/// second limb came back exactly zero for every degree of freedom. Reading the
/// factor from a uniform is what stops the compiler proving the identity, at
/// the cost of a multiplication that cannot change a bit.
pub const GPU_CARRY_GUARD: f32 = 1.0;

// ---------------------------------------------------------------------------
// Finite element model
// ---------------------------------------------------------------------------

/// Nodes per hexahedral element.
pub const NODES_PER_ELEMENT: usize = 8;

/// Displacement degrees of freedom per node.
pub const DOF_PER_NODE: usize = 3;

/// Degrees of freedom per element.
pub const DOF_PER_ELEMENT: usize = NODES_PER_ELEMENT * DOF_PER_NODE;

/// Gauss points per axis for the element stiffness integration (2x2x2 rule).
pub const GAUSS_POINTS_PER_AXIS: usize = 2;

/// Independent components of a symmetric stress or strain tensor in 3D, ordered
/// `[xx, yy, zz, xy, yz, zx]` with engineering shear strains.
pub const STRESS_COMPONENTS: usize = 6;

// ---------------------------------------------------------------------------
// Self-weight loads
// ---------------------------------------------------------------------------

/// Mass density conversion from the `g/cm^3` of `[material]` to the tonnes per
/// cubic millimetre that the millimetre-newton-second unit system needs.
///
/// Dimensional analysis, with `[L] = mm`, `[T] = s` and force in newtons:
/// `1 N = 1 kg m/s^2 = 1 kg * 1000 mm/s^2`, so the consistent mass unit is the
/// tonne (`1 t * 1 mm/s^2 = 1000 kg * 1e-3 m/s^2 = 1 N`). Converting the
/// density itself,
/// `1 g/cm^3 = 1 g / 1000 mm^3 = 1e-6 t / 1000 mm^3 = 1e-9 t/mm^3`.
/// The self-weight of an element is therefore
/// `rho [g/cm^3] * TONNE_PER_MM3_PER_G_PER_CM3 * V [mm^3] * g [mm/s^2]`
/// newtons: `t/mm^3 * mm^3 * mm/s^2 = t mm/s^2 = N`.
pub const TONNE_PER_MM3_PER_G_PER_CM3: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Stress recovery
// ---------------------------------------------------------------------------

/// Elements whose physical density is below this are left out of the stress
/// report. Recovering a stress from a near-void element says nothing about the
/// printed part, and dividing the tiny SIMP modulus back out amplifies it into
/// meaningless peaks.
pub const STRESS_DENSITY_THRESHOLD: f64 = 0.5;

/// Percentile reported next to the peak von Mises stress, as a fraction.
pub const STRESS_PERCENTILE: f64 = 0.99;

/// Fraction of the evaluated elements averaged into the "top decile mean".
pub const STRESS_TOP_FRACTION: f64 = 0.10;

/// Hard iteration cap for the post-run stress solve, separate from
/// [`CG_MAX_ITERATIONS`] and deliberately higher.
///
/// The stress solve runs once, at the end, on the final near-0/1 density field
/// whose stiffness contrast is the worst the run ever produced, and it has no
/// warm start and no later iteration to recover in. The optimization loop could
/// not afford this budget every iteration; the one solve that decides whether
/// there is a stress table at all can.
pub const STRESS_CG_MAX_ITERATIONS: usize = 50_000;

/// Relative residual the post-run stress solve is taken to, relaxed against
/// [`CG_RELATIVE_TOLERANCE`].
///
/// Stress recovery differentiates the displacement field, so the error it
/// inherits is first order in the residual: at 1e-6 the recovered von Mises
/// values are stable to parts per million, which is four orders below the
/// percent-level discretization error a single trilinear hexahedron per voxel
/// carries at a staircase boundary (see the module caveats in `stress`). The
/// two extra decades the optimization path asks for buy nothing this model can
/// resolve, and they are exactly what makes the late-iteration system expensive
/// to solve.
pub const STRESS_CG_TOLERANCE: f64 = 1e-6;

/// Physical density a cell needs before the load path pre-check counts it as
/// material joining one node to the next.
///
/// Deliberately far below [`STRESS_DENSITY_THRESHOLD`] and below any iso level:
/// the question this threshold answers is not "is this part of the printed
/// part" but "is there any stiffness here at all". A path through cells at 0.1
/// carries `0.1^p = 1e-3` of the solid modulus at the default penalty and solves
/// perfectly well; a path that exists only through cells at the SIMP floor carries
/// `[optimization] stiffness_floor`, a factor of 1e9 down at its
/// [`SIMP_EMIN_FRACTION`] default, and is what turns the system near-singular.
/// Only the second is reported.
pub const STRESS_LOAD_PATH_DENSITY: f64 = 0.05;

// ---------------------------------------------------------------------------
// Trim pass
// ---------------------------------------------------------------------------

/// Fraction of the envelope von Mises peak a part cell has to stay below for
/// `[output] trim = "stress"` to consider removing it.
///
/// One percent of the worst stress the part carries anywhere, over every load
/// case: a member below that is not a load path. It carries a hundredth of what
/// the structure's own hot spot does while occupying material, printing time
/// and, for the prongs and wisps this pass exists for, overhang support, so
/// removing it changes nothing about how the part behaves.
///
/// Conservative in the direction that matters. A genuine load path carries
/// stress by definition, and the ratio between the peak of a structure and the
/// stress in a member that is actually working is a small number, not two orders
/// of magnitude; every one percent that this cut-off is raised by starts asking
/// whether lightly loaded material is still load bearing, which is a question
/// the optimizer has already answered and this pass has no business re-opening.
/// Overriding it is `[output] trim_stress_fraction`, and the connectivity guard
/// of [`crate::trim`] holds whatever it is set to.
pub const TRIM_STRESS_FRACTION: f64 = 0.01;

// ---------------------------------------------------------------------------
// Reinforcement pass
// ---------------------------------------------------------------------------

/// Distance from the centre of a cell to its own face, in voxels.
///
/// Lattice geometry rather than a tunable, and named because
/// [`crate::reinforce`] leans on it twice and reading a bare `0.5` in either
/// place would be reading two different halves. The distance transform measures
/// centre to centre, so the *surface* of the part lies half a voxel inside the
/// nearest cell that is not part of it: a member of `w` cells has a spine
/// distance of `(w + 1) / 2` cells and a thickness of `w`. The same half voxel
/// is what the fill radius adds, so that the material a fill leaves reaches the
/// floor rather than stopping a half cell short of it.
pub const REINFORCE_SURFACE_OFFSET_VOXELS: f64 = 0.5;

/// Slack, as a fraction of a voxel, by which a member has to fall short of
/// `[output] reinforce_thickness_mm` before it counts as thin.
///
/// A member built exactly on the floor - the common case, since a run whose
/// `min_feature_mm` is the floor resolves features at it - measures the floor
/// back out to the last bit, and rounding either way would either thicken a part
/// that is already right or report it as impossible to fix. Small enough that
/// nothing a user could mean by a thickness lands inside it.
pub const REINFORCE_THICKNESS_EPS_VOXELS: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Flush fill pass (`[output] flush`)
// ---------------------------------------------------------------------------

/// How deep `[output] flush = "walls"` fills, in voxels, when a configuration
/// names no `flush_depth_mm`.
///
/// The defect the pass exists for is a sampling one, and its scale is the voxel:
/// a cell is classified by its centre, so the modelled boundary can be wrong by
/// half a cell ([`BOUNDARY_CLAMP_CAPTURE_VOXELS`]), and the intermediate
/// densities an optimizer leaves in the last cells of a wall pull the extracted
/// isosurface a further fraction of a voxel inwards. Two voxels is that scale
/// with a margin: deep enough to reach the material of a wall whose outermost
/// cells came out below the iso level and to fill the cells between it and the
/// surface, and shallow enough that what it fills is still the wall rather than
/// the part behind it. It is also twice the deepest correction the boundary
/// clamp will make to a single vertex
/// ([`BOUNDARY_CLAMP_MAX_DISPLACEMENT_VOXELS`]), which is the mesh-side reading
/// of the same artefact.
///
/// A run that wants another depth writes `[output] flush_depth_mm`, which is a
/// length rather than a count of voxels because that is what the rest of the
/// table is; the two are related by the grid, and [`crate::flush::depth_mm`] is
/// the one place that relation is stated.
pub const FLUSH_DEPTH_VOXELS: f64 = 2.0;

/// Shallowest depth `[output] flush_depth_mm` may name, in voxels.
///
/// Half a voxel is the distance from a cell centre to its own face, and the
/// resolution a cell-centre classification has: below it the pass could only
/// ever select cells the boundary already passes through, which is less than the
/// artefact it is being asked to correct. Excluded rather than allowed, because a
/// depth at exactly that scale is already a pass that cannot do its job, and a
/// configuration that names one has misunderstood the key rather than tuned it.
pub const FLUSH_DEPTH_MIN_VOXELS: f64 = 0.5;

/// Deepest depth `[output] flush_depth_mm` may name, in voxels.
///
/// Four times [`FLUSH_DEPTH_VOXELS`], which is generous: the artefact is about
/// one voxel deep and the default already carries a margin over it. Past this
/// the pass stops seating the walls that rest on a surface and starts laying a
/// solid skin over every one of them, taking the mass, the local volume cap and
/// the design the optimizer produced with it - a different operation, asked for
/// with the same key. Excluded at the end for the reason the shallow end is.
pub const FLUSH_DEPTH_MAX_VOXELS: f64 = 8.0;

// ---------------------------------------------------------------------------
// Mesh extraction
// ---------------------------------------------------------------------------

/// Thickness, in cells, of the zero padding wrapped around the node field so
/// the isosurface always closes inside the sampled block.
pub const FIELD_PADDING_CELLS: usize = 1;

/// Node values within this distance of the iso level are nudged to
/// `iso + ISO_SNAP_EPS`. Sampling a flat solid boundary lands node values
/// exactly on the iso level, which would place several marching cubes vertices
/// on the same node and produce zero-area triangles; the nudge keeps every
/// vertex strictly inside its edge at a geometrically negligible cost.
pub const ISO_SNAP_EPS: f64 = 1e-4;

/// Largest number of vertices a mesh may hold. Vertices are addressed by
/// `u32`, and `u32::MAX` is reserved as the "no vertex on this edge yet"
/// marker of the marching cubes edge table, so the last index is unusable.
pub const MAX_MESH_VERTICES: usize = u32::MAX as usize;

/// Default `[output] supersample`: mesh on the node lattice itself.
pub const DEFAULT_SUPERSAMPLE: usize = 1;

/// Largest `[output] supersample` factor.
///
/// Triangle count and file size grow with about the square of the factor and
/// the lattice with its cube, so 4 already turns a 200 000 triangle export into
/// a three million triangle one. Beyond that the extra triangles describe the
/// trilinear interpolant of the density field rather than anything the
/// optimization resolved.
pub const MAX_SUPERSAMPLE: usize = 4;

/// Bytes the export spends per refined lattice sample: eight for the `f64`
/// value and twelve for the three `u32` edge slots marching cubes keys its
/// vertex welding on.
pub const SUPERSAMPLE_BYTES_PER_NODE: usize = 20;

/// Refined lattice size above which `[output] supersample` is warned about.
///
/// At [`SUPERSAMPLE_BYTES_PER_NODE`] this is about 610 MiB of transient export
/// memory, on top of whatever the grid and the solver already hold. The run is
/// not stopped: the figure is a machine-dependent judgement, and a workstation
/// with the memory for it is entitled to ask.
pub const SUPERSAMPLE_NODE_BUDGET_WARN: usize = 32_000_000;

/// Largest number of triangles a binary STL can declare in its `u32` count.
pub const MAX_STL_TRIANGLES: usize = u32::MAX as usize;

/// Solid angle, in steradians, above which a closed shell counts as enclosing a
/// point.
///
/// A closed surface subtends `4 * PI` at every point inside it and zero at
/// every point outside it, whatever its winding, so the decision line is the
/// midpoint of the two and everything near it is rounding. Only a point lying
/// exactly *on* a shell is genuinely ambiguous, and it reads as outside, which
/// is the reading that never removes geometry on a coin toss; see
/// [`crate::mesh::islands::encloses`].
pub const ISLAND_SOLID_ANGLE_INSIDE: f64 = std::f64::consts::TAU;

/// Floating fragments listed individually in the run summary before the rest
/// are counted in one line. A skeletal design can mint dozens of small skins,
/// and the volume of the largest few is what says whether they matter.
pub const ISLAND_REPORT_MAX_LINES: usize = 8;

/// Points per declared region that the island pass tests components against for
/// containment, spread over the material the region selected by taking one from
/// each occupied box of a partition of the region into at most this many.
///
/// The second of the two questions [`crate::mesh::islands::Anchors`] asks, and
/// the expensive one: a solid angle sum per probe per candidate shell. One point
/// would answer it for a region whose material is one piece; eight is a
/// two-by-two-by-two partition of a compact region, or four by two of a flat
/// one, and is still nothing next to the surface it is asked about.
///
/// What the count buys is precisely this: every axis of the region that has an
/// extent is divided (see [`crate::problem::Problem`]'s probe selection), so no
/// periodic row structure and no dominant axis can take every probe. It does not
/// buy every gap being straddled - a gap narrower than one box can lie inside
/// one, at any finite count - which is why containment is the *second* leg of
/// the anchor test rather than the only one.
pub const ISLAND_ANCHOR_PROBES_PER_REGION: usize = 8;

/// A kept body enclosing less than this many spheres of diameter
/// `[optimization] min_feature_mm` is reported as tiny.
///
/// One such sphere is the smallest lump the density filter can resolve at all,
/// so a separate body below it is not a feature of the design: it is a stub the
/// optimization left behind, and it ships as a loose piece because a declared
/// region touches it and geometry a declared region asked for is never removed.
/// Saying so is the whole of what this threshold does - nothing is culled by it.
pub const ISLAND_TINY_BODY_SPHERES: f64 = 1.0;

/// Taubin smoothing shrink factor.
pub const TAUBIN_LAMBDA: f64 = 0.5;

/// Taubin smoothing unshrink factor; must satisfy mu < -lambda.
pub const TAUBIN_MU: f64 = -0.53;

/// Triangles with an area below this (mm^2) are reported as degenerate.
pub const MIN_TRIANGLE_AREA_MM2: f64 = 1e-12;

/// Cubic millimetres per cubic centimetre, for the mass estimate.
pub const MM3_PER_CM3: f64 = 1000.0;

/// Grams per tonne, for turning a self weight in newtons back into the mass
/// behind it. The millimetre-newton-second system's mass unit is the tonne (see
/// [`TONNE_PER_MM3_PER_G_PER_CM3`]), so `W [N] / a [mm/s^2]` comes out in
/// tonnes and needs this to be read as grams.
pub const GRAMS_PER_TONNE: f64 = 1e6;

// ---------------------------------------------------------------------------
// Boundary clamp (`[output] boundaries`)
// ---------------------------------------------------------------------------
//
// The export samples the density field on a lattice and Taubin-smooths what
// marching cubes gives back, so an exported vertex can sit a fraction of a voxel
// inside a keepout or outside the domain: neither pass knows the analytic
// shapes, and cell-centre classification cannot resolve a boundary finer than
// half a cell. Under the default `boundaries = "exact"` the vertices that do are
// projected back onto the analytic surface they violate, which is what makes a
// pin bore in the exported STL the cylinder the configuration asked for. These
// control that projection.

/// How far the clamp may move a single vertex, in voxels.
///
/// The encroachment this pass corrects is **sub-voxel by construction**: a cell
/// is classified by its centre, so material may reach at most half a voxel past
/// the boundary that carved it, and smoothing moves a vertex by a fraction of
/// the spacing between vertices, which is the voxel size. A vertex that would
/// have to move further than one whole voxel to become legal is therefore not
/// this defect - it is a configuration, a field or a mesh that is wrong in some
/// other way - and moving it that far would deform the surface rather than
/// correct it. Such a vertex is left where it is and counted in the report.
pub const BOUNDARY_CLAMP_MAX_DISPLACEMENT_VOXELS: f64 = 1.0;

/// How many times the clamp re-examines a vertex it has already moved.
///
/// Projecting a vertex out of one keepout can push it into an overlapping one,
/// or out of the domain; the corrected position is therefore tested again. Two
/// or three rounds settle every arrangement that terminates at all, and the
/// ceiling is what keeps a pair of shapes whose surfaces bounce a point between
/// them from spinning. A vertex still illegal at the end is left where it was
/// and counted.
pub const BOUNDARY_CLAMP_MAX_PASSES: usize = 8;

/// How far outside a keepout surface - and inside the domain surface - a clamped
/// vertex is placed, in millimetres.
///
/// The projection lands on the zero level set, where a containment test that
/// reads `signed_distance <= 0` would still call the vertex inside. This offset
/// puts it strictly on the legal side. Ten nanometres: five orders of magnitude
/// below the finest voxel any run lays out and four below a slicer's resolution,
/// so it is invisible in the part and large enough to survive the rounding of a
/// millimetre-scale coordinate.
pub const BOUNDARY_CLAMP_EPS_MM: f64 = 1e-5;

/// How near its target level set an iterated projection counts as arrived, in
/// millimetres.
///
/// Only the two projections that have no closed form use it: the ellipsoid,
/// whose nearest surface point is the root of a sextic, and the domain, which is
/// an ordered CSG composite with non-differentiable seams. Well below
/// [`BOUNDARY_CLAMP_EPS_MM`], so a converged projection is on the legal side of
/// the surface by that offset and not by rounding.
pub const BOUNDARY_CLAMP_TOLERANCE_MM: f64 = 1e-7;

/// Largest number of steps an iterated projection takes before giving up.
///
/// A ceiling rather than a working number: a Newton step on a field whose
/// gradient is a unit vector converges quadratically and arrives in a handful.
/// What this bounds is the seam of a CSG composite, where the gradient jumps and
/// the iteration can oscillate instead of settling; a vertex that reaches the
/// ceiling is left where it was and counted.
pub const BOUNDARY_CLAMP_MAX_STEPS: usize = 64;

/// How near an analytic surface a **legal** vertex has to sit, in voxels, for
/// the gap to count as a sampling artefact and be closed.
///
/// The clamp's first job is the vertex that is *proud* of a boundary, and that
/// one is decided by legality: it is on the wrong side, so it is moved. This
/// decides the other half of the same artefact. Marching cubes puts a vertex
/// within about half a voxel of the crossing it interpolates, Taubin smoothing
/// then rounds the staircase that sampling produces, and the two together
/// scatter a wall's vertices to *both* sides of the surface the wall really is.
/// Correcting only the outward half leaves the inward half as dimples - measured
/// at up to a third of a voxel on a 0.5 mm plug, and visible as scalloping on a
/// surface that was supposed to be a cone.
///
/// Half a voxel is the artefact's own scale, and it is the figure the module
/// header already names: a cell is classified by its **centre**, so the modelled
/// solid and the analytic one differ by at most half a voxel anywhere. Nothing
/// further away than that is a sampling error, and nothing further away is
/// touched - an optimizer's free surface running through the middle of the
/// domain is not near a boundary at all, and the smoothing that shaped it is
/// left alone.
///
/// What it costs, stated plainly: a free surface that runs *within* half a voxel
/// of a boundary is treated as resting on it and is seated onto it. That is a
/// gap the voxel field cannot resolve anyway - a void channel that thin is below
/// what one cell centre can carry - and the correction is always onto the legal
/// side, so it can add a sliver of material inside the domain and can never put
/// any outside it or inside a keepout.
pub const BOUNDARY_CLAMP_CAPTURE_VOXELS: f64 = 0.5;

/// Half-step of the central difference the domain projection reads its gradient
/// from, in millimetres.
///
/// The domain is an ordered CSG tree of minima and maxima, which has no analytic
/// gradient to hand out; the step is small enough that no real feature hides
/// inside it and large enough that the difference of two millimetre-scale
/// distances keeps its significant digits.
pub const BOUNDARY_CLAMP_GRADIENT_STEP_MM: f64 = 1e-5;

/// How far from the nearest analytic boundary a vertex may come to rest, in
/// voxels, and still be **counted** as adrift from it rather than read as a free
/// surface through the middle of the domain.
///
/// This measures; it moves nothing. The clamp seats what it can reach
/// ([`BOUNDARY_CLAMP_CAPTURE_VOXELS`], half a voxel) and deliberately leaves
/// everything further out alone, and until this existed it left it *silently*:
/// a face that came out a whole voxel off the surface it was drawn against read
/// as an ordinary free surface, the clamp line said nothing about it, and the
/// first thing to notice was the slicer. So the pass counts what it did not
/// seat, and the report says so.
///
/// Two voxels is that band. Four times the capture, so everything the seat
/// should have caught and did not is comfortably inside it and the reading is
/// not a hair away from the correction; it also covers the incident that named
/// it, a bottom face resting 0.88 of its run's voxel above the plate, with room
/// on either side. It is the flush pass's own depth ([`FLUSH_DEPTH_VOXELS`]),
/// which is not a coincidence: what that pass reaches out to fill is exactly
/// what this one reports as resting short - and the definition below derives
/// from it so the two cannot drift apart. Wider than this and an optimizer's
/// free surface running past a wall - which rests on nothing and is nobody's
/// defect - starts being counted; the count would then be noise on every
/// organic part and would say nothing on the one that is wrong.
pub const BOUNDARY_ADRIFT_WINDOW_VOXELS: f64 = FLUSH_DEPTH_VOXELS;

// ---------------------------------------------------------------------------
// STL output
// ---------------------------------------------------------------------------

/// Binary STL header length in bytes.
pub const STL_HEADER_BYTES: usize = 80;

/// Bytes per triangle record in a binary STL file.
pub const STL_TRIANGLE_BYTES: usize = 50;

/// Value written into the per-triangle attribute byte count field.
pub const STL_ATTRIBUTE_BYTE_COUNT: u16 = 0;

// ---------------------------------------------------------------------------
// Memory estimate (reported by `growforge check`)
// ---------------------------------------------------------------------------

/// Number of full-length f64 arrays over cells that the SIMP loop keeps live
/// (x, x_tilde, x_new, dc, dv, element moduli, filter normalisation).
pub const ESTIMATED_CELL_ARRAYS: usize = 7;

/// Further full-length f64 arrays over cells that `update = "mma"` adds: the two
/// asymptotes and the two previous design points it needs to see a trend.
pub const ESTIMATED_MMA_CELL_ARRAYS: usize = 4;

/// Further full-length f64 arrays over cells that
/// `[optimization.local_volume]` adds.
///
/// Six of its own - the local kernel's normalisation, the local field, the
/// aggregate's seed, the two arrays its chain transpose passes through, and the
/// design space gradient the update reads - and four in the moving asymptotes
/// subproblem, which carries a second pair of separable coefficients and the two
/// anchors the two-constraint dual evaluates its approximations against. The
/// last four are per *design* cell rather than per cell, so counting them whole
/// runs the estimate high on a problem with forced material, which is the safe
/// direction for a budget.
pub const ESTIMATED_LOCAL_VOLUME_CELL_ARRAYS: usize = 10;

/// Number of full-length f64 arrays over degrees of freedom that the solver
/// keeps live per solve (residual, search direction, preconditioned residual,
/// matrix-vector product, diagonal).
pub const ESTIMATED_SOLVER_VECTORS: usize = 5;

/// Number of full-length f64 arrays over degrees of freedom kept per load case
/// (the load vector and the warm-started displacement).
pub const ESTIMATED_VECTORS_PER_LOADCASE: usize = 2;

/// Bytes per stored floating point value.
pub const BYTES_PER_F64: usize = 8;

/// Bytes per mebibyte, for human readable memory reporting.
pub const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

// ---------------------------------------------------------------------------
// Solver benchmark (`growforge bench`)
// ---------------------------------------------------------------------------

/// Timed solves per backend. Enough to see the run-to-run spread, few enough
/// that a multi-million degree of freedom problem still finishes.
pub const BENCH_SOLVES: usize = 3;

/// Relative residual the benchmark solves are taken to.
///
/// Two decades looser than [`CG_RELATIVE_TOLERANCE`]: far enough into the
/// ill-conditioned part of the spectrum that the comparison is about the solver
/// rather than about the first few easy iterations, and near enough that a CPU
/// reference on the largest configuration finishes in minutes rather than
/// hours. The same tolerance is used for every backend, so the ratio is a like
/// for like speedup.
pub const BENCH_CG_TOLERANCE: f64 = 1e-6;

/// Iteration cap of one benchmark solve. A run that hits it is reported as
/// capped rather than silently compared against a converged one.
pub const BENCH_CG_MAX_ITERATIONS: usize = 20_000;

/// Milliseconds per second, for the benchmark table.
pub const MS_PER_S: f64 = 1000.0;

// ---------------------------------------------------------------------------
// Viewer: window and testing hooks
// ---------------------------------------------------------------------------
//
// Everything below is consumed by the `viewer` feature only. The values live
// here rather than in the viewer module so that the single "no policy number
// outside constants.rs" rule keeps holding, and so a build without the feature
// still documents them.

/// Window title prefix; the mode and the document are appended, so a title
/// reads `growforge 3D 0.40.0 - <project>` in the run and setup windows and
/// `growforge 3D 0.40.0 edit - <file>` in the editor.
///
/// The prefix carries the brand and the version - [`DISPLAY_NAME_AND_VERSION`] -
/// because the title bar and the taskbar entry are what a user, and a screenshot
/// of a window, see first. Each panel draws this same title as its own heading,
/// which is what a screenshot of the panel alone carries and what makes a
/// second, weaker version line under it a repetition rather than a claim of its
/// own.
pub const VIEW_WINDOW_TITLE: &str = DISPLAY_NAME_AND_VERSION;

/// Initial window width in logical pixels.
pub const VIEW_WINDOW_WIDTH: u32 = 1280;

/// Initial window height in logical pixels.
pub const VIEW_WINDOW_HEIGHT: u32 = 800;

/// Environment variable that closes the viewer automatically this many seconds
/// after the first rendered frame. Unset or unparsable means "never", so it
/// cannot affect an interactive session by accident. Testing and CI hook.
pub const VIEW_AUTOCLOSE_ENV: &str = "GROWFORGE_VIEW_AUTOCLOSE_S";

/// Background clear colour, RGBA in the surface's own encoding.
pub const VIEW_BACKGROUND_COLOR: [f64; 4] = [0.07, 0.08, 0.10, 1.0];

/// Consecutive frames the drawing surface may reject before the viewer gives up
/// on its device.
///
/// A rejected frame is what a driver reset - a Windows TDR, which a long compute
/// job on the same physical GPU can trip - looks like from the window's side:
/// the device the frames come from is gone for a moment. Reconfiguring the
/// surface is the renderer's only recovery, and a surface whose device comes
/// back does so within a few frames, so this is the grace a reset needs rather
/// than a limit on anything healthy. Thirty frames is about a fifth of a second
/// at 144 Hz: generous for a reset, short enough that a device which never comes
/// back is reported instead of leaving a dead session drawing nothing.
pub const VIEW_SURFACE_REJECTION_LIMIT: u32 = 30;

// ---------------------------------------------------------------------------
// Viewer: camera
// ---------------------------------------------------------------------------

/// Vertical field of view in degrees.
pub const VIEW_FOV_DEGREES: f64 = 45.0;

/// The fitted view leaves this much slack around the scene's bounding sphere.
pub const VIEW_FIT_MARGIN: f64 = 1.15;

/// Near plane, as a fraction of the scene's bounding sphere radius.
pub const VIEW_NEAR_PLANE_RADIUS_FRACTION: f64 = 1e-3;

/// Far plane, as a multiple of the scene radius added behind the target.
pub const VIEW_FAR_PLANE_RADIUS_MARGIN: f64 = 8.0;

/// Azimuth of the initial camera direction, in degrees about world +z.
pub const VIEW_INITIAL_YAW_DEGREES: f64 = 35.0;

/// Elevation of the initial camera direction, in degrees above the xy plane.
pub const VIEW_INITIAL_PITCH_DEGREES: f64 = 25.0;

/// The camera may not pass within this many degrees of either pole, which
/// would make the up vector ambiguous.
pub const VIEW_PITCH_LIMIT_DEGREES: f64 = 89.0;

/// Orbit speed in degrees of rotation per pixel of pointer travel.
pub const VIEW_ORBIT_DEGREES_PER_PIXEL: f64 = 0.4;

/// Multiplicative zoom step applied per scroll line.
pub const VIEW_ZOOM_PER_SCROLL_LINE: f64 = 0.12;

/// Pixels of scroll travel that count as one line, for devices that report
/// pixel deltas.
pub const VIEW_SCROLL_PIXELS_PER_LINE: f64 = 40.0;

/// Closest the camera may come to its target, as a fraction of the scene
/// radius.
pub const VIEW_MIN_DISTANCE_RADIUS_FRACTION: f64 = 0.05;

/// Furthest the camera may retreat, as a multiple of the scene radius.
pub const VIEW_MAX_DISTANCE_RADIUS_FACTOR: f64 = 50.0;

/// Scene radius substituted when the scene is degenerate (a single point), so
/// the projection never divides by zero.
pub const VIEW_MIN_SCENE_RADIUS_MM: f64 = 1e-3;

// ---------------------------------------------------------------------------
// Viewer: shading and colours
// ---------------------------------------------------------------------------

/// Direction the key light travels in world space; it is normalized on use.
pub const VIEW_LIGHT_DIRECTION: [f32; 3] = [-0.45, -0.75, -0.55];

/// Fraction of the surface colour that survives with no light on it.
pub const VIEW_AMBIENT_FRACTION: f32 = 0.32;

/// Weight of the light hitting a surface from behind, which keeps the inside
/// of an open preview surface readable.
pub const VIEW_BACKLIGHT_FRACTION: f32 = 0.45;

/// Whether the density surface starts out flat shaded.
///
/// False: the isosurface is drawn with area weighted per-vertex normals, so a
/// marching cubes surface reads as the curved solid it approximates instead of
/// as its facets. The side panel's "flat shading" switch turns the per-triangle
/// normals back on. Overlays are diagrammatic and are always flat.
pub const VIEW_DEFAULT_FLAT_SHADING: bool = false;

/// Colour of the evolving density isosurface and of the final mesh.
pub const VIEW_COLOR_DENSITY: [f32; 4] = [0.85, 0.80, 0.70, 1.0];

/// Colour of the voxelized design domain surface (translucent).
pub const VIEW_COLOR_DOMAIN: [f32; 4] = [0.42, 0.58, 0.78, 0.14];

/// Colour of the keepout regions (translucent).
pub const VIEW_COLOR_KEEPOUT: [f32; 4] = [0.88, 0.28, 0.26, 0.26];

/// Colour of the keepin regions (translucent).
pub const VIEW_COLOR_KEEPIN: [f32; 4] = [0.30, 0.78, 0.46, 0.26];

/// Colour of the support markers.
pub const VIEW_COLOR_SUPPORT: [f32; 4] = [0.24, 0.50, 0.95, 1.0];

/// Colour of the load arrows and torque arcs.
pub const VIEW_COLOR_LOAD: [f32; 4] = [0.96, 0.62, 0.12, 1.0];

/// Anchor colours of the von Mises stress ramp, cold to hot, interpolated
/// linearly in between. A blue-teal-yellow-red ramp keeps a monotone perceived
/// lightness, so the reading survives a greyscale screenshot.
pub const VIEW_STRESS_COLORMAP: &[[f32; 3]] = &[
    [0.19, 0.25, 0.62],
    [0.24, 0.52, 0.83],
    [0.42, 0.76, 0.71],
    [0.86, 0.83, 0.36],
    [0.93, 0.53, 0.20],
    [0.71, 0.10, 0.14],
];

/// Opacity of the stress coloured surface; it replaces the opaque density
/// surface, so it is opaque too.
pub const VIEW_STRESS_ALPHA: f32 = 1.0;

// ---------------------------------------------------------------------------
// Viewer: overlay geometry
// ---------------------------------------------------------------------------

/// Angle between two facets, in degrees, above which they are drawn as an edge
/// rather than as one curved surface. See [`crate::viewer::scene::Shading`].
///
/// Comfortably above the angle between the facets of anything tessellated
/// here (a 48 segment barrel turns 7.5 degrees per facet, a 24 ring sphere the
/// same), and comfortably below the shallowest real edge any of them has, which
/// is the 90 degrees where a cylinder's barrel meets its cap.
pub const VIEW_CREASE_ANGLE_DEGREES: f64 = 40.0;

/// Number of segments around a tessellated cylinder or cone.
///
/// 48 rather than 24: a keepout cylinder is one of the largest things on
/// screen, and at 24 the silhouette of one reads as a polygon however it is
/// shaded. The cost is proportional and small - a cylinder is 4 triangles per
/// segment, so 192 instead of 96 - and none of it reaches an exported mesh,
/// which is extracted from the density field rather than tessellated.
pub const VIEW_CYLINDER_SEGMENTS: usize = 48;

/// Number of segments around the tube of a torque arc; the arc is made of many
/// short tubes, so it needs fewer than a standalone cylinder.
pub const VIEW_ARC_TUBE_SEGMENTS: usize = 12;

/// Longitude divisions of a tessellated sphere.
pub const VIEW_SPHERE_SEGMENTS: usize = 48;

/// Latitude divisions of a tessellated sphere, poles included.
///
/// Half the longitude count, which is what makes the quads square-ish; a
/// sphere is then 2 * 48 * 23 = 2208 triangles, against 528 before.
pub const VIEW_SPHERE_RINGS: usize = 24;

/// Number of straight steps a bent tube's centre line is drawn in.
///
/// The sweep of a tube is at most a whole turn, so at this count a facet spans
/// no more than 11 degrees of the arc - the same order as the 7.5 degrees
/// between two facets around a cylinder, and inside the crease angle, so a
/// bend reads as one curved surface rather than as a chain of pipes.
pub const VIEW_TUBE_ARC_SEGMENTS: usize = 32;

/// Latitude divisions of a tube's rounded end cap, pole included. Half a
/// sphere's, because a cap is half a sphere.
pub const VIEW_TUBE_CAP_RINGS: usize = VIEW_SPHERE_RINGS / 2;

/// Edge length of a support marker cube, as a fraction of the voxel size.
pub const VIEW_SUPPORT_MARKER_VOXEL_FRACTION: f64 = 0.34;

/// Largest number of support marker cubes drawn; denser support regions are
/// subsampled with a constant stride so the overlay stays bounded.
pub const VIEW_MAX_SUPPORT_MARKERS: usize = 20_000;

/// Total length of a load arrow, as a fraction of the scene radius.
pub const VIEW_ARROW_LENGTH_RADIUS_FRACTION: f64 = 0.30;

/// Arrow shaft radius, as a fraction of the arrow length.
pub const VIEW_ARROW_SHAFT_RADIUS_FRACTION: f64 = 0.035;

/// Arrow head length, as a fraction of the arrow length.
pub const VIEW_ARROW_HEAD_LENGTH_FRACTION: f64 = 0.28;

/// Arrow head base radius, as a fraction of the arrow length.
pub const VIEW_ARROW_HEAD_RADIUS_FRACTION: f64 = 0.10;

/// Radius of a torque arc, as a fraction of the scene radius.
pub const VIEW_TORQUE_ARC_RADIUS_FRACTION: f64 = 0.22;

/// Sweep of a torque arc in degrees; less than a full turn so the head is
/// visible next to the tail.
pub const VIEW_TORQUE_ARC_SWEEP_DEGREES: f64 = 280.0;

/// Number of straight tube segments approximating a torque arc.
pub const VIEW_TORQUE_ARC_SEGMENTS: usize = 48;

/// Tube radius of a torque arc, as a fraction of the arc radius.
pub const VIEW_TORQUE_TUBE_RADIUS_FRACTION: f64 = 0.045;

/// Torque arrow head length, as a fraction of the arc radius.
pub const VIEW_TORQUE_HEAD_LENGTH_FRACTION: f64 = 0.16;

/// Torque arrow head base radius, as a fraction of the arc radius.
pub const VIEW_TORQUE_HEAD_RADIUS_FRACTION: f64 = 0.055;

// ---------------------------------------------------------------------------
// Viewer: live preview
// ---------------------------------------------------------------------------

/// Shortest wall clock gap between two preview isosurface extractions. The
/// snapshot slot always keeps the newest design, so raising this throttles the
/// preview without ever showing a stale one.
pub const VIEW_PREVIEW_MIN_INTERVAL_S: f64 = 0.20;

/// Width of the egui side panel in points.
pub const VIEW_PANEL_WIDTH_POINTS: f32 = 260.0;

/// What the block of layer visibility switches is called, in both windows.
///
/// Shared because the block is: the run panel writes it as the block's own
/// heading, and the editor's panel writes it on the collapsing header the block
/// sits under, where it is the header's persistent id as well as its label.
pub const VIEW_LAYER_SWITCHES_LABEL: &str = "show";

/// How often the viewer wakes to service its window message queue after the
/// window has been torn down but the run behind it is still going.
///
/// A process that owns window state and stops servicing its message queue for
/// more than a few seconds is declared hung by Windows and terminated, which
/// would kill a detached run part way through. The event loop therefore stays
/// alive until the run ends; it simply has nothing left to draw.
pub const VIEW_DETACHED_POLL_INTERVAL_S: f64 = 0.25;

// ---------------------------------------------------------------------------
// Viewer: visual editor (`growforge edit`)
// ---------------------------------------------------------------------------
//
// The editor opens the same window in a mode where the configuration itself is
// the document: objects are picked and dragged in the viewport, every value is
// editable numerically, and the setup overlays are rebuilt from the edited
// configuration. Everything below is consumed by that mode only.

/// Width of the editor's side panel in points.
///
/// Wider than [`VIEW_PANEL_WIDTH_POINTS`], because the editor's properties
/// panel puts three numeric fields on a row where the run panel puts one line
/// of text. The 3D viewport is sized against whichever of the two is showing.
pub const VIEW_EDIT_PANEL_WIDTH_POINTS: f32 = 360.0;

/// Label of the object tree's block.
///
/// Every labelled block of the panel sits under a collapsing header, so a
/// session can put away whatever it is not working on. egui derives a header's
/// persistent id from its label text, which makes each of these labels the name
/// of a piece of session state as well as what the block says on screen - and
/// is why the lists whose own labels carry a live count are given one of these
/// as an explicit id salt instead.
pub const VIEW_EDIT_BLOCK_OBJECTS: &str = "objects";

/// Label of the block holding the properties of the selection.
pub const VIEW_EDIT_BLOCK_PROPERTIES: &str = "properties";

/// Label of the block holding the safety factor of the run on screen.
pub const VIEW_EDIT_BLOCK_STRESS: &str = "stress";

/// Label of the block holding the live problem summary.
pub const VIEW_EDIT_BLOCK_PROBLEM: &str = "problem";

/// Label of the block holding the control legend.
pub const VIEW_EDIT_BLOCK_CONTROLS: &str = "controls";

/// Name of the load case list in the object tree.
///
/// Its header reads `load cases (n)`, so the name is also given to the header
/// as an explicit id salt: a label that carries a count is a label that changes,
/// and a header identified by a changing label loses whatever the user did to
/// it every time an object is added or deleted. The four region lists are
/// salted the same way with their own names, which they already hold.
pub const VIEW_EDIT_LOAD_CASE_LIST: &str = "load cases";

/// Undo steps the editor keeps. Older steps fall off the bottom.
pub const VIEW_EDIT_UNDO_DEPTH: usize = 100;

/// Quiet time after the last committed edit before the setup is re-voxelized
/// and a re-run is started.
///
/// A gizmo drag commits once, on release, but a numeric field being dragged
/// commits on every value it passes through; without this the editor would
/// re-voxelize the whole domain tens of times per second.
pub const VIEW_EDIT_REFRESH_DEBOUNCE_S: f64 = 0.15;

/// Pointer travel, in physical pixels, below which a left press and release
/// counts as a click rather than as an orbit.
///
/// This is what lets the left button both select and orbit: the camera sees
/// every drag it always saw, and only a press that went nowhere picks.
pub const VIEW_EDIT_CLICK_SLOP_PIXELS: f64 = 4.0;

/// Iteration cap of a SIMP fast preview, whatever `[optimization]
/// max_iterations` says. A preview exists to show where the material is going,
/// not to converge.
pub const VIEW_EDIT_PREVIEW_MAX_ITERATIONS: usize = 20;

/// Cell budget of a SIMP fast preview. A grid above this is coarsened by
/// scaling the voxel size up by the cube root of the excess, which is the
/// factor that brings the cell count back to the budget.
pub const VIEW_EDIT_PREVIEW_CELL_BUDGET: usize = 40_000;

/// Whether auto-regrow starts switched on for `engine = "growth"`. The growth
/// engine grows a design in milliseconds, so a re-run per edit is free.
pub const VIEW_EDIT_AUTO_REGROW_GROWTH: bool = true;

/// Whether auto-regrow starts switched on for the SIMP engine. Off: even a
/// coarsened, iteration-capped preview is seconds of finite element solving.
pub const VIEW_EDIT_AUTO_REGROW_SIMP: bool = false;

/// How much later than the moment a full run recorded its own output file the
/// file's mtime may still be, and count as that run's write rather than
/// somebody else's, in seconds.
///
/// The recorded time is read back off the file after the write, so an untouched
/// file needs no tolerance at all; this is for the filesystems that quantize a
/// timestamp more coarsely than that read (FAT resolves two seconds) or settle
/// it a moment after the handle closes. It is also the window in which another
/// program's overwrite is mistaken for ours, so it stays as small as those
/// filesystems allow.
pub const VIEW_EDIT_OUTPUT_MTIME_TOLERANCE_S: f64 = 2.0;

/// Smallest extent a drag may leave a shape with, in millimetres. Well above
/// [`MIN_SHAPE_EXTENT_MM`], which is a degeneracy guard rather than a usable
/// size.
pub const VIEW_EDIT_MIN_EXTENT_MM: f64 = 0.1;

/// Edge length of a newly added object, as a fraction of the domain bounding
/// box's smallest edge. New objects land at the centre of the domain.
pub const VIEW_EDIT_NEW_OBJECT_SIZE_FRACTION: f64 = 0.25;

/// Radius a new tube is given, and the half-length of its straight segment, as
/// a fraction of [`VIEW_EDIT_NEW_OBJECT_SIZE_FRACTION`]'s edge length.
///
/// Half the sphere's radius, which makes a default tube twice as long as it is
/// wide - a tube rather than a ball, with room either side of the middle handle
/// for the bend it is dragged into. The one place the size of a tube is decided,
/// so an added one and a placed one are the same tube.
pub const VIEW_EDIT_NEW_TUBE_RADIUS_FRACTION: f64 = 0.25;

/// The second radius of a new cone, as a fraction of its first.
///
/// Under one, so a cone added from the panel *tapers*: at one it would be the
/// cylinder it already has a button for, and an add that produced a shape
/// indistinguishable from another kind's would read as a failed add. Half is far
/// enough to be unmistakable at a glance and still leaves the narrow end wide
/// enough to carry a grabbable radius handle.
pub const VIEW_EDIT_NEW_CONE_TAPER_FRACTION: f64 = 0.5;

/// Thickness of a new triangular prism, as a fraction of its side length.
///
/// A quarter, which reads as a plate rather than as a block: the three vertex
/// handles are what shape it, and a default deep enough to hide them behind
/// itself would be one nobody could grab.
pub const VIEW_EDIT_NEW_TRIANGLE_THICKNESS_FRACTION: f64 = 0.25;

/// Edge length of the cube new objects are sized against while the
/// configuration has no usable domain to measure, in millimetres.
pub const VIEW_EDIT_FALLBACK_EXTENT_MM: f64 = 100.0;

/// Name prefix of a newly added load case; the case's number is appended.
pub const VIEW_EDIT_NEW_CASE_PREFIX: &str = "case-";

/// Extension the editor's file dialogs filter on and give a name that was typed
/// without one. A growforge 3D configuration is a TOML document.
pub const VIEW_EDIT_FILE_EXTENSION: &str = "toml";

/// What the file dialogs call that kind of file in their filter dropdown.
pub const VIEW_EDIT_FILE_FILTER: &str = "growforge 3D configuration";

/// Extension of the copy an editing session writes beside its file when the
/// viewer dies with unsaved changes in it.
///
/// A second extension rather than a suffix on the stem, so `rod_bracket.toml`
/// is recovered as `rod_bracket.recovered.toml`: a name that says what it is,
/// still a `.toml` the editor and `growforge run` open, and never the name of
/// the file the session was editing.
pub const VIEW_EDIT_RECOVERY_EXTENSION: &str = "recovered.toml";

/// Title of the dialog the "open" button raises.
pub const VIEW_EDIT_OPEN_DIALOG_TITLE: &str = "open a growforge 3D configuration";

/// Title of the dialog the "new" button raises.
pub const VIEW_EDIT_NEW_DIALOG_TITLE: &str = "new growforge 3D configuration";

/// Name the "new" dialog opens with, which is also what the scaffolded file's
/// project and STL are named after when it is accepted as it stands: the file
/// stem is what [`crate::viewer::editor::state::EditorState::create`] names
/// everything from.
pub const VIEW_EDIT_NEW_FILE_NAME: &str = "new_part.toml";

/// What the status line says when "new" was pointed at a file that is already
/// there. A "new" is never a way to open a file and never a way to overwrite
/// one, so it says so rather than quietly doing either.
pub const VIEW_EDIT_NEW_EXISTS_NOTE: &str = "already exists - use open instead; a new file is only scaffolded for a name that is not \
     there yet";

/// Folder, under the platform's Documents folder, that is the canonical home of
/// a user's growforge configurations.
///
/// What `growforge edit` with no path looks in, and what the installed Start
/// Menu shortcut therefore lands in: an installed copy has no working directory
/// worth defaulting to - the program sits in Program Files, where nothing may be
/// written - so the one place it can offer is the user's own. Nothing else reads
/// it: a path given on the command line is always taken as it stands, and the
/// folder is only created when a file is actually saved into it.
pub const VIEW_EDIT_CONFIGS_HOME_DIR: &str = "growforge";

/// What the console says when the dialog a pathless `growforge edit` raised was
/// cancelled.
///
/// A cancelled dialog is an answer, not a failure: nothing was asked for after
/// all, so the command says what happened in one line and exits successfully.
pub const VIEW_EDIT_NOTHING_OPENED: &str = "nothing opened";

// ---------------------------------------------------------------------------
// Starter configuration (`growforge edit` on a path that is not there yet)
// ---------------------------------------------------------------------------
//
// The smallest configuration that is a real problem: a block standing on the
// floor with something pressing on a pad at the top. Every number below is a
// starting point to be dragged around in the editor, not a recommendation; what
// they have to be is *valid*, so the window opens on a model rather than on an
// error, and coarse, so it grows in a moment.

/// Edge length of the starter block, in millimetres. A common print bed's
/// worth of part.
pub const STARTER_DOMAIN_MM: f64 = 120.0;

/// Voxel size of the starter configuration. Thirty cells per axis: fine enough
/// to see a shape, coarse enough to regrow between keystrokes, and well inside
/// the preview budget whichever engine the file is later switched to.
pub const STARTER_VOXEL_MM: f64 = 4.0;

/// Smallest feature of the starter configuration, in millimetres. Four voxels,
/// which is clear of the resolution warning.
pub const STARTER_MIN_FEATURE_MM: f64 = 16.0;

/// Mass fraction the starter configuration keeps.
pub const STARTER_MASS_FRACTION: f64 = 0.15;

/// Side of the starter's loaded pad, as a fraction of the block.
pub const STARTER_PAD_FRACTION: f64 = 0.4;

/// Thickness of that pad and of the support slab, in millimetres.
pub const STARTER_PAD_MM: f64 = 9.0;

/// Load pressing down on the pad, in newtons.
pub const STARTER_LOAD_N: f64 = 400.0;

/// Engine the starter configuration selects.
///
/// The topology optimizer, which is what growforge is for and what the compute
/// backend a new file also defaults to exists to accelerate. It is not the
/// cheapest choice - the growth heuristic grows a design in milliseconds where
/// this one solves - which is why auto-regrow starts *off* for it
/// (see [`VIEW_EDIT_AUTO_REGROW_SIMP`]): a new file opens on its setup, and the
/// first run is one the user asks for.
pub const STARTER_ENGINE: &str = DEFAULT_ENGINE;

/// How many integer literals above `i64::MAX` one configuration file may hold
/// and still round-trip through the editor.
///
/// The format preserving TOML library stores every integer as an `i64` and
/// carries larger ones as text (see the editor's `toml_io` module).
/// `[growth] seed` is the only `u64` in the schema, so one is the realistic
/// number; the budget exists so that a pathological file fails with an
/// explanation instead of spinning.
pub const VIEW_EDIT_MAX_BIG_INTEGERS: usize = 8;

/// Magnitude of a newly added force load, in newtons.
pub const VIEW_EDIT_NEW_FORCE_N: f64 = 100.0;

/// Magnitude of a newly added torque load, in newton-millimetres.
pub const VIEW_EDIT_NEW_TORQUE_NMM: f64 = 1000.0;

/// Millimetres a numeric length field moves per pixel of drag.
pub const VIEW_EDIT_DRAG_SPEED_MM: f64 = 0.05;

/// Fraction of a value a dimensionless numeric field moves per pixel of drag.
pub const VIEW_EDIT_DRAG_SPEED_FRACTION: f64 = 0.002;

/// Fraction of *itself* a field written in exponent notation moves per pixel of
/// drag.
///
/// The fields whose useful range spans decades rather than a factor of a few -
/// `[solver] tolerance` ([`CG_TOLERANCE_MIN`] to [`CG_TOLERANCE_MAX`]) and
/// `[optimization] stiffness_floor` ([`STIFFNESS_FLOOR_MIN`] to
/// [`STIFFNESS_FLOOR_MAX`]) - so the drag is proportional and the pixels buy
/// decades: at a fiftieth of the value per pixel a decade is about 115 points
/// of travel and a range of six about 690, which is the same order as a length
/// field crossing a part.
pub const VIEW_EDIT_SCIENTIFIC_DRAG_FRACTION: f64 = 0.02;

/// Degrees a numeric rotation field moves per pixel of drag. Coarser than the
/// length fields: a degree is a much smaller change to a part than a
/// millimetre, and the snapped gizmo arcs are where round angles come from.
pub const VIEW_EDIT_ROTATION_DRAG_SPEED_DEG: f64 = 0.25;

/// Decimal places a numeric field shows. Millimetres to a micron, which is a
/// decade finer than any printer resolves and two below the voxel sizes these
/// problems are solved on.
pub const VIEW_EDIT_DECIMALS: usize = 3;

/// Voxel size a configuration is given when the editor switches it from a cell
/// target to an explicit size, in millimetres.
pub const VIEW_EDIT_DEFAULT_VOXEL_MM: f64 = 2.0;

/// Cell count a configuration is given when the editor switches it the other
/// way.
pub const VIEW_EDIT_DEFAULT_TARGET_CELLS: usize = 500_000;

/// Build direction a newly enabled overhang constraint starts with: layers
/// stacking up, which is what a printer standing on its plate does.
pub const VIEW_EDIT_DEFAULT_BUILD_DIRECTION: crate::config::BuildDirection =
    crate::config::BuildDirection::ZPlus;

/// Mirror plane a newly enabled `[growth.symmetry]` starts with: one plane
/// normal to x, the smallest symmetry there is, so the editor adds a halved
/// problem rather than deciding a four-fold one on the user's behalf.
pub const VIEW_EDIT_DEFAULT_SYMMETRY_PLANE: crate::config::Axis = crate::config::Axis::X;

/// Order a `[growth.symmetry]` switched to `rotational` in the editor starts
/// with: four-fold, the symmetry of the corner-supported problems this feature
/// was asked for.
pub const VIEW_EDIT_DEFAULT_SYMMETRY_ORDER: usize = 4;

/// Label of the button each scalar section of the editor's panel carries to put
/// its own keys back to the defaults.
pub const VIEW_EDIT_RESET_LABEL: &str = "reset";

/// What that button says while its section is already at its defaults, which is
/// when it is drawn disabled rather than taken away: a control that vanishes is
/// one the user has to go looking for again.
pub const VIEW_EDIT_RESET_AT_DEFAULTS_HINT: &str = "this section is at its defaults";

/// Voxels a reset of the optimization section puts `[optimization]
/// min_feature_mm` at, as a multiple of the voxel size the configuration is
/// solved on.
///
/// The one key of that section with no default of its own: it is a length, and
/// what a length means depends entirely on the grid, so there is no number to
/// return it to - only a rule. The rule is the one the run's own warning
/// teaches, [`MIN_FEATURE_VOXELS_WARN`], and it is held to that very constant
/// rather than to a second copy of the figure: a reset lands exactly on the
/// smallest feature the density filter can resolve, and so can never leave a
/// configuration the next run warns about.
pub const VIEW_EDIT_RESET_MIN_FEATURE_VOXELS: f64 = MIN_FEATURE_VOXELS_WARN;

/// Smallest `sin^2` of the angle between a pointer ray and the line or plane a
/// handle is dragged in, below which the drag is ignored for that frame.
///
/// A ray that looks straight down a translation arrow has no well defined point
/// on it, and one that grazes a drag plane crosses it arbitrarily far away.
/// Both are geometry, not policy failures: the handle simply holds still until
/// the camera or the pointer gives it something to work with.
pub const VIEW_EDIT_MIN_DRAG_ANGLE_SINE_SQUARED: f64 = 1e-6;

/// How far a resize drag has to travel across the plane it is dragged in before
/// it latches on to the dimension it is changing, in millimetres.
///
/// A resize handle sets one dimension, and which one is read off the *start* of
/// the gesture: until the pointer has covered this much of the drag plane there
/// is no direction to read it from, so the drag changes nothing at all rather
/// than resizing on the strength of a few microns of shake. Small enough to be
/// over before a deliberate drag has begun - a fraction of a handle's own width,
/// and well under [`VIEW_EDIT_SNAP_MM`] - and far larger than the jitter of a
/// press.
pub const VIEW_EDIT_RESIZE_LATCH_MM: f64 = 0.6;

/// Colour of the selected object's highlight shell (translucent).
pub const VIEW_COLOR_SELECTION: [f32; 4] = [1.0, 0.95, 0.35, 0.30];

/// Colours of the three gizmo translation arrows, in x, y, z order.
pub const VIEW_COLOR_GIZMO_AXES: [[f32; 4]; 3] = [
    [0.92, 0.26, 0.28, 1.0],
    [0.36, 0.84, 0.36, 1.0],
    [0.30, 0.52, 0.98, 1.0],
];

/// Colour of the camera-plane translation handle at the gizmo's centre.
pub const VIEW_COLOR_GIZMO_PLANE: [f32; 4] = [0.95, 0.95, 0.95, 1.0];

/// Colour of the resize handles (box corners and faces, radii, cylinder
/// endpoints).
pub const VIEW_COLOR_GIZMO_HANDLE: [f32; 4] = [1.0, 0.78, 0.20, 1.0];

/// Colour of a tube's bend handle. Its own hue rather than the resize amber,
/// because on a straight tube it sits exactly where the centre handle of every
/// other shape is and it does something else entirely: it curves the tube.
pub const VIEW_COLOR_GIZMO_BEND: [f32; 4] = [0.80, 0.40, 0.95, 1.0];

/// Length of a translation arrow, as a fraction of the selected object's
/// bounding sphere radius.
///
/// Object relative, so the handles keep their proportion to whatever they move,
/// with a floor against the scene so a very small object is still grabbable.
/// Below one on purpose: a box's bounding sphere is its half *diagonal*, so
/// this puts each arrow's tip a little outside the face it points through -
/// reachable, in proportion, and inside the view that framed the object rather
/// than out past the edge of it.
pub const VIEW_EDIT_GIZMO_LENGTH_RADIUS_FRACTION: f64 = 0.75;

/// Floor on the gizmo length, as a fraction of the whole scene's bounding
/// sphere radius, so a tiny object still gets grabbable handles.
pub const VIEW_EDIT_GIZMO_MIN_SCENE_FRACTION: f64 = 0.06;

/// Edge length of a cubic handle, as a fraction of the gizmo length.
pub const VIEW_EDIT_HANDLE_SIZE_FRACTION: f64 = 0.13;

/// Radius of a handle's pick sphere, as a multiple of half its edge length.
/// Above one, so a handle is easier to hit than to see.
pub const VIEW_EDIT_HANDLE_PICK_FACTOR: f64 = 1.7;

/// How much larger a tube's bend handle is drawn, and grabbed, than the other
/// cubic handles.
///
/// On a straight tube the bend handle sits exactly on the camera-plane
/// translation handle - the middle of the tube is both the point that moves it
/// and the point that curves it - and two markers in one place are one click
/// with one winner. Above one, so the bend is the volume the ray enters first
/// and the cube the user sees, which is what makes "drag the middle to make a
/// curve" the thing that happens. Bend the tube and the handle leaves the
/// centre, and the translation handle is exposed again.
pub const VIEW_EDIT_BEND_HANDLE_FACTOR: f64 = 1.25;

/// Largest number of sphere-tracing steps a ray-tube intersection takes.
///
/// A tube bent into an arc is not convex and has no closed form hit, so it is
/// traced against its own exact distance field instead: from the entry point of
/// its bounding box, step by the distance to the surface, which can never step
/// through it. The count is a ceiling rather than a working number - a ray
/// aimed at the tube converges in a handful of steps - and it is what keeps a
/// ray that grazes the surface, where the steps become arbitrarily short, from
/// spinning. Such a ray reports a miss.
pub const VIEW_EDIT_TUBE_PICK_MAX_STEPS: usize = 128;

/// How near the surface a sphere-traced ray counts as having hit it, in
/// millimetres. A tenth of a micron: a hit is on the surface as far as anything
/// that reads a pick position is concerned, and it is coarse enough that the
/// trace terminates rather than halving its way down to the last bit.
pub const VIEW_EDIT_TUBE_PICK_EPS_MM: f64 = 1e-4;

/// How near zero a ray-quadric discriminant counts as the tangency it is,
/// relative to the terms it was subtracted from.
///
/// A ray that touches a cone's slanted wall along its whole length - which is
/// exactly what a ray down the axis of a true cone does, grazing the wall at the
/// apex - has a discriminant of zero, and zero is what a subtraction of two
/// nearly equal numbers loses first. One ulp of cancellation turns that entry
/// into a miss and hands the press the far side of the shape. This is the band
/// in which the tangency is taken as real: at 1e-9 relative, seven orders of
/// magnitude above the rounding it exists for, a ray has to pass within a
/// fraction of a nanometre of the surface to fall inside it - and the point it
/// then reports is on that surface to the same accuracy.
pub const VIEW_EDIT_PICK_TANGENT_EPS: f64 = 1e-9;

/// Margin the selection highlight is inflated by, as a fraction of the selected
/// shape's smallest extent, so it never z-fights with the shape itself.
pub const VIEW_EDIT_SELECTION_MARGIN_FRACTION: f64 = 0.02;

/// Floor on that margin, as a fraction of the gizmo length, so a very thin
/// shape still gets a visible shell.
pub const VIEW_EDIT_SELECTION_MARGIN_GIZMO_FRACTION: f64 = 0.02;

// ---------------------------------------------------------------------------
// Viewer: editor hover highlight
// ---------------------------------------------------------------------------
//
// A preview of what a click would take: the object under the pointer gets a
// second, thinner shell, drawn by the same machinery as the selection one and
// deliberately a different colour, so "what is selected" and "what would be
// selected" never read as the same thing.

/// Colour of the hover outline (translucent).
///
/// Cool against the selection shell's warm yellow, and weaker, because the
/// hover is a prediction and the selection is a fact.
pub const VIEW_COLOR_HOVER: [f32; 4] = [0.45, 0.88, 1.0, 0.22];

/// Margin the hover outline is inflated by, as a fraction of the hovered
/// shape's smallest extent. A third of
/// [`VIEW_EDIT_SELECTION_MARGIN_FRACTION`], which is what makes it read as the
/// thinner of the two when both are on screen.
pub const VIEW_EDIT_HOVER_MARGIN_FRACTION: f64 = 0.0067;

/// Floor on that margin, as a fraction of the gizmo length, in the same
/// proportion to [`VIEW_EDIT_SELECTION_MARGIN_GIZMO_FRACTION`].
pub const VIEW_EDIT_HOVER_MARGIN_GIZMO_FRACTION: f64 = 0.0067;

/// How far a hovered gizmo handle's colour is moved towards white.
///
/// The handle under the pointer brightens instead of growing a shell of its
/// own: a handle is already a small solid marker, and a second surface around
/// it would be larger than the thing it describes.
pub const VIEW_EDIT_HANDLE_HOVER_BRIGHTEN: f32 = 0.45;

// ---------------------------------------------------------------------------
// Viewer: editor floor grid
// ---------------------------------------------------------------------------
//
// A ruled plane on the floor of the domain, at the increment drags land on, so
// the snapping has something to be seen against.

/// Colour of the minor grid lines, at the snap increment.
///
/// Halved in 0.26.1, on the user's "dim the grid 50%": the ruling is what the
/// model is seen against, and at its old weight it competed with the model. The
/// halving is on the RGB rather than on the alpha because the grid is an opaque
/// layer - it is drawn in the pass built with no blend state, where the alpha a
/// vertex carries never reaches a blend - so alpha is not a brightness lever
/// here. Was `[0.55, 0.60, 0.68, 0.55]`.
pub const VIEW_COLOR_GRID_MINOR: [f32; 4] = [0.275, 0.30, 0.34, 0.55];

/// Colour of every [`VIEW_EDIT_GRID_MAJOR_EVERY`]-th line.
///
/// Dimmed by the same factor as the minor lines, so the emphasis keeps the
/// contrast against them that it had. Was `[0.78, 0.84, 0.92, 0.85]`.
pub const VIEW_COLOR_GRID_MAJOR: [f32; 4] = [0.39, 0.42, 0.46, 0.85];

/// One line in this many is a major one.
pub const VIEW_EDIT_GRID_MAJOR_EVERY: usize = 10;

/// How far the grid runs past the domain's footprint, as a fraction of the
/// larger footprint side.
pub const VIEW_EDIT_GRID_MARGIN_FRACTION: f64 = 0.10;

/// Radius of a minor grid line's tube, as a fraction of the footprint's
/// diagonal, so the weight of the ruling follows the size of the model rather
/// than being an absolute width that vanishes on a large one.
pub const VIEW_EDIT_GRID_MINOR_WEIGHT_FRACTION: f64 = 0.0008;

/// The same for a major line, which is drawn heavier.
pub const VIEW_EDIT_GRID_MAJOR_WEIGHT_FRACTION: f64 = 0.0018;

/// Sides of the prism a grid line is drawn as. Four, because a grid line is a
/// hairline on screen and nothing about it reads as round.
pub const VIEW_EDIT_GRID_TUBE_SEGMENTS: usize = 4;

/// Largest number of lines the floor grid may draw, both axes together.
///
/// A large domain at a fine increment asks for millions of lines, which is
/// neither drawable nor legible. Past this the minor spacing is multiplied by
/// the smallest whole number that brings the count back inside the cap - whole,
/// so the major lines still land on multiples of the increment - and the panel
/// says the spacing it ended up with.
pub const VIEW_EDIT_GRID_MAX_LINES: usize = 400;

/// Whether the floor grid starts switched on. Editor only: the layer has
/// geometry only in edit mode, so `view` and `run --view` are unchanged.
pub const VIEW_EDIT_GRID_DEFAULT_ON: bool = true;

// ---------------------------------------------------------------------------
// Viewer: editor precision interaction (snapping, callouts, indicators)
// ---------------------------------------------------------------------------
//
// What turns dragging into *placing*: a drag lands on a round number, the
// number is shown where the drag is happening, and it can be typed instead of
// dragged. Everything below is the editor's own interaction state - none of it
// reaches the configuration file.

/// Millimetre increment a translation or a resize snaps to, until the panel is
/// told otherwise.
pub const VIEW_EDIT_SNAP_MM: f64 = 1.0;

/// Increments the panel's snap dropdown offers, in millimetres. Anything else
/// is typed into the field beside it.
pub const VIEW_EDIT_SNAP_CHOICES_MM: [f64; 7] = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];

/// Smallest snap increment the panel accepts, in millimetres. At or below it
/// snapping is off, which is also what a field typed to zero means.
pub const VIEW_EDIT_SNAP_MIN_MM: f64 = 1e-4;

/// Degree increment a rotation drag snaps to. Sixteen steps to a full turn,
/// which holds 45 and 90 degrees as well as the halves between them.
pub const VIEW_EDIT_SNAP_DEGREES: f64 = 22.5;

/// How near a surface a dragged load or support region has to come, in
/// millimetres, before it lands flush against it.
///
/// Larger than [`VIEW_EDIT_SNAP_MM`] on purpose: within this distance the
/// surface is what the user is aiming at, and the millimetre grid would
/// otherwise park the region one increment short of the face it is meant to sit
/// on. Small enough that it only catches a region that is already there.
pub const VIEW_EDIT_SURFACE_SNAP_MM: f64 = 2.5;

/// How much of a drag axis has to point along a world axis before a flush
/// landing on that axis' planes means anything.
///
/// The candidate planes are axis aligned; a drag along the local axis of a
/// turned box runs across all three of them at once, and correcting such a drag
/// onto one plane moves it along the other two by more than the correction
/// itself. Above a half - which every unrotated drag is, exactly - the axis the
/// drag mostly runs along is the one it is placing against.
pub const VIEW_EDIT_SURFACE_SNAP_MIN_ALIGNMENT: f64 = 0.5;

/// Seconds a dimension callout stays on screen after the drag that raised it
/// was released, so the number can be read - and clicked to be typed over.
pub const VIEW_EDIT_CALLOUT_LINGER_S: f64 = 6.0;

/// Decimal places a callout shows. One is what a millimetre readout wants; the
/// exact value is always available in the properties panel.
pub const VIEW_EDIT_CALLOUT_DECIMALS: usize = 2;

/// Points a callout is offset from the point it is anchored to, so the number
/// sits beside the geometry rather than on top of it.
pub const VIEW_EDIT_CALLOUT_OFFSET_POINTS: f32 = 14.0;

/// Width of a callout's text field in points, when it has been clicked to type
/// an exact value into.
pub const VIEW_EDIT_CALLOUT_FIELD_WIDTH_POINTS: f32 = 64.0;

/// Points kept clear at the right and bottom edges of the 3D view, so a callout
/// anchored to an object at the edge of it is still fully visible and can never
/// slide under the side panel.
pub const VIEW_EDIT_CALLOUT_RESERVE_POINTS: [f32; 2] = [120.0, 40.0];

/// Radius of a rotation arc handle's ring, as a fraction of the gizmo length.
/// Above one, so the rings sit outside the translation arrows.
pub const VIEW_EDIT_ROTATE_RING_RADIUS_FRACTION: f64 = 1.35;

/// Sweep of a rotation arc in degrees.
///
/// Shallow on purpose: the arc is grabbed by a tube around its own *chord*, and
/// at this sweep the arc bulges only `1 - cos(sweep/2)` of its radius away from
/// that chord - well inside [`VIEW_EDIT_ROTATE_HANDLE_PICK_FRACTION`] - so the
/// whole drawn arrow is grabbable rather than one point on it.
pub const VIEW_EDIT_ROTATE_ARC_SWEEP_DEGREES: f64 = 45.0;

/// Number of straight tube segments approximating a rotation arc.
pub const VIEW_EDIT_ROTATE_ARC_SEGMENTS: usize = 12;

/// Tube radius of a rotation arc, as a fraction of the ring radius.
pub const VIEW_EDIT_ROTATE_ARC_TUBE_FRACTION: f64 = 0.035;

/// Length of a rotation arc's arrow head, as a multiple of its tube radius.
pub const VIEW_EDIT_ROTATE_ARC_HEAD_LENGTH_FACTOR: f64 = 4.0;

/// Base radius of that head, as a multiple of the tube radius.
pub const VIEW_EDIT_ROTATE_ARC_HEAD_RADIUS_FACTOR: f64 = 2.2;

/// Radius of a rotation handle's pick volume, as a fraction of the ring radius.
pub const VIEW_EDIT_ROTATE_HANDLE_PICK_FRACTION: f64 = 0.16;

/// Colour of the rotation arcs, one per axis, in x, y, z order. The axis
/// colours of the translation arrows, dimmed, so a ring reads as belonging to
/// the arrow it turns about.
pub const VIEW_COLOR_GIZMO_ARCS: [[f32; 4]; 3] = [
    [0.78, 0.30, 0.42, 1.0],
    [0.36, 0.72, 0.46, 1.0],
    [0.36, 0.48, 0.86, 1.0],
];

/// Colour of the dimension lines drawn during a drag.
pub const VIEW_COLOR_MEASURE: [f32; 4] = [1.0, 0.93, 0.55, 1.0];

/// Colour of the floor-distance indicator, which measures something else - the
/// gap to the bottom of the domain - and says so by being another colour.
pub const VIEW_COLOR_MEASURE_FLOOR: [f32; 4] = [0.55, 0.85, 1.0, 1.0];

/// Radius of a dimension line's tube, as a fraction of the gizmo length.
pub const VIEW_EDIT_MEASURE_TUBE_FRACTION: f64 = 0.018;

/// Length of a dimension line's arrow head, as a fraction of the gizmo length.
pub const VIEW_EDIT_MEASURE_HEAD_LENGTH_FRACTION: f64 = 0.11;

/// Base radius of that head, as a fraction of the gizmo length.
pub const VIEW_EDIT_MEASURE_HEAD_RADIUS_FRACTION: f64 = 0.045;

/// Shortest dimension line that is drawn at all, as a multiple of its own head
/// length: below it the two heads would meet and there is nothing to read.
pub const VIEW_EDIT_MEASURE_MIN_LENGTH_HEADS: f64 = 2.2;

/// Smallest `|z|` of a translation axis that counts as moving vertically, which
/// is when the floor-distance indicator is drawn. Above a half turn's worth of
/// tilt the drag is mostly sideways and the distance to the floor is not what
/// is being set.
pub const VIEW_EDIT_FLOOR_INDICATOR_MIN_Z: f64 = 0.5;

/// Whether new editor sessions keep non-domain objects inside the domain.
pub const VIEW_EDIT_CONTAINMENT_DEFAULT: bool = true;

/// What the panel says when a commit was moved to keep it inside the domain.
pub const VIEW_EDIT_CONTAINMENT_NOTE: &str =
    "kept inside the domain; untick \"keep inside domain\" to place it outside";

/// Seconds that note stays up.
pub const VIEW_EDIT_CONTAINMENT_NOTE_S: f64 = 4.0;

/// Whether the viewport starts out showing nothing but the part.
///
/// Off: a session opens on the model it is there to edit - the domain, the
/// regions, the gizmo - and the switch beside this one's siblings in the
/// precision block is what puts the run's own surface on screen alone. Editor
/// only, like the floor grid's switch: it is the one panel that draws it.
pub const VIEW_EDIT_ISOLATE_PART_DEFAULT: bool = false;

/// What the precision block says for as long as the part is shown on its own.
///
/// The viewport stops taking edits while it is - every handle and outline a
/// gesture would aim at is hidden - so the panel says why nothing under the
/// pointer answers, for as long as that is true. The same promise the placement
/// hint keeps.
pub const VIEW_EDIT_ISOLATE_PART_NOTE: &str =
    "showing the part alone: viewport editing is suspended, and the camera and the panel are not";

// ---------------------------------------------------------------------------
// Viewer: editor two-click placement
// ---------------------------------------------------------------------------
//
// The tube's own creation flow: the add row's "tube" button opens a placement
// mode instead of dropping one at the centre of the domain, and the two clicks
// that follow are the tube's two ends. Everything here describes what that mode
// draws while it waits for them.

/// Colour of the rubber band drawn between the first clicked point and where
/// the second click would land (translucent).
///
/// The colour of the selection shell the placed tube is about to get, at the
/// hover shell's weight: what is on screen is a preview of an object that does
/// not exist yet, so it reads as provisional rather than as geometry.
pub const VIEW_COLOR_PLACE_GHOST: [f32; 4] = [1.0, 0.95, 0.35, 0.22];

/// Colour of the markers on the points themselves: the one already clicked, and
/// the one under the pointer.
///
/// Opaque and warm against the band, because a point is the thing being aimed,
/// and it is small enough that a translucent one would be lost against the
/// surface it sits on.
pub const VIEW_COLOR_PLACE_MARKER: [f32; 4] = [1.0, 0.78, 0.20, 1.0];

/// Edge length of those markers, as a fraction of the scene's bounding sphere
/// radius.
///
/// Scene relative rather than object relative, unlike a gizmo handle: there is
/// no object yet to take a proportion of. Sized to read as the gizmo's cubes do
/// on a middling object, so the marker that becomes an end handle does not
/// change size when the tube appears under it.
pub const VIEW_EDIT_PLACE_MARKER_SCENE_FRACTION: f64 = 0.012;

/// Shortest tube two placement clicks may make, in millimetres, when there is no
/// snap increment to measure them against.
///
/// With snapping on, the increment is the resolution the clicks themselves have
/// and two inside it are the same point clicked twice. With it off - or with the
/// bypass key held - there is no such resolution, and this is what keeps a
/// second click on the first one from writing a tube of no length.
pub const VIEW_EDIT_PLACE_MIN_LENGTH_MM: f64 = 0.1;

/// What the panel says while the first point of a tube is being placed.
pub const VIEW_EDIT_PLACE_HINT_FIRST: &str = "click the first point in the viewport; Esc cancels";

/// And while the second is.
pub const VIEW_EDIT_PLACE_HINT_SECOND: &str =
    "click the second point; then drag the middle to curve it. Esc cancels";

#[cfg(test)]
mod tests {
    use super::*;

    /// The brand may extend the machine name, never contradict it.
    ///
    /// [`DISPLAY_NAME`] is a literal, because what the product is called is not
    /// derivable from what the package is called, and this is the coupling that
    /// literal is still held to: the name on a window has to begin with the name
    /// typed at a prompt, so a user reading `growforge 3D` knows what to run,
    /// what the executable is called and which folder the setup installed into.
    /// Read from the manifest rather than from [`PROGRAM_NAME`], so a crate
    /// rename that left the brand behind - either half moving without the other
    /// - fails here rather than on a user's machine.
    #[test]
    fn the_display_name_extends_the_machine_name() {
        assert!(
            DISPLAY_NAME.starts_with(env!("CARGO_PKG_NAME")),
            "the display name {DISPLAY_NAME:?} does not start with the package name {:?}: a brand \
             that contradicts the command, the executable and the install directory leaves a user \
             with nothing to type",
            env!("CARGO_PKG_NAME")
        );
        assert_eq!(
            PROGRAM_NAME,
            env!("CARGO_PKG_NAME"),
            "the machine name is the package name spelled out; they have drifted"
        );
    }

    /// The two display constants are one string and its extension by the
    /// version, which `concat!` cannot express because it takes literals only.
    #[test]
    fn the_display_version_string_is_the_display_name_and_the_version() {
        assert_eq!(
            DISPLAY_NAME_AND_VERSION,
            format!("{DISPLAY_NAME} {VERSION}"),
            "the brand is spelled twice in this module and the two spellings have drifted"
        );
        assert_eq!(
            NAME_AND_VERSION,
            format!("{PROGRAM_NAME} {VERSION}"),
            "the machine name and version string no longer matches its two halves"
        );
    }
}
