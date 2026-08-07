//! End-to-end checks: a single element against the analytic axial answer, a
//! small cantilever optimization through the whole pipeline, and the shipped
//! example configurations.

use std::path::{Path, PathBuf};

use growforge::config::{BoundaryFidelity, Config};
use growforge::constants;
use growforge::fea::{StiffnessOperator, hex8_stiffness, solver};
use growforge::geometry::Aabb;
use growforge::grid::{CellKind, Grid};
use growforge::mesh;
use growforge::problem::Problem;
use growforge::report::SilentReporter;
use growforge::{load_problem, optimize_and_export};

/// A fixture text as `engine` needs to be given it.
///
/// One engine takes a key out rather than adding one: the solid engine refuses
/// `[optimization] mass_fraction`, because it fills the domain instead of
/// targeting a share of it, so a fixture written for the optimizing engines is
/// handed to it without that line. Everything else - the geometry, the regions,
/// the loads, the resolution - is the same problem, which is the point of
/// running one fixture on more than one engine at all.
fn for_engine(engine: &str, fixture: &str) -> String {
    let body: String = match engine == constants::SOLID_ENGINE {
        true => fixture
            .lines()
            .filter(|line| !line.trim_start().starts_with("mass_fraction"))
            .collect::<Vec<_>>()
            .join("\n"),
        false => fixture.to_string(),
    };
    format!("engine = \"{engine}\"\n{body}")
}

/// One hexahedron under uniform axial traction, restrained on three
/// perpendicular faces so it may contract laterally. A trilinear hex reproduces
/// a uniform strain state exactly, so the tip displacement must match the
/// analytic `F L / (E A)`.
#[test]
fn single_element_axial_displacement_matches_the_analytic_answer() {
    let h = 10.0;
    let youngs_modulus = 2300.0;
    let poisson_ratio = 0.3;
    let total_force = 100.0;

    let mut grid = Grid::from_bounds(
        &Aabb {
            min: [0.0, 0.0, 0.0],
            max: [h, h, h],
        },
        h,
    );
    grid.cells[0] = CellKind::Solid;

    let ke0 = hex8_stiffness(poisson_ratio, h);
    let moduli = vec![youngs_modulus];
    let operator = StiffnessOperator::new(&grid, &ke0, &moduli);

    // Roller restraints on the three faces through the origin.
    let mut fixed = vec![false; grid.n_dof()];
    for a in 0..2 {
        for b in 0..2 {
            fixed[grid.node_index(0, a, b) * constants::DOF_PER_NODE] = true;
            fixed[grid.node_index(a, 0, b) * constants::DOF_PER_NODE + 1] = true;
            fixed[grid.node_index(a, b, 0) * constants::DOF_PER_NODE + 2] = true;
        }
    }

    // Uniform traction on the far face, as consistent nodal loads.
    let mut forces = vec![0.0; grid.n_dof()];
    let loaded: Vec<usize> = (0..2)
        .flat_map(|b| (0..2).map(move |a| (a, b)))
        .map(|(a, b)| grid.node_index(1, a, b))
        .collect();
    for &n in &loaded {
        forces[n * constants::DOF_PER_NODE] += total_force / loaded.len() as f64;
    }

    let diagonal = operator.diagonal();
    let mut u = vec![0.0; grid.n_dof()];
    solver::solve(
        |v, out| operator.apply(v, out),
        &diagonal,
        &fixed,
        &forces,
        &mut u,
        growforge::fea::SolveLimits::new(
            constants::CG_RELATIVE_TOLERANCE,
            constants::CG_MAX_ITERATIONS,
        ),
    )
    .expect("solve");

    // F L / (E A) with L = h and A = h^2.
    let expected = total_force * h / (youngs_modulus * h * h);
    for &n in &loaded {
        let ux = u[n * constants::DOF_PER_NODE];
        let error = (ux - expected).abs() / expected;
        assert!(
            error < 1e-3,
            "axial displacement {ux} differs from {expected} by {:.4}%",
            error * 100.0
        );
    }

    // Poisson contraction of the free faces, also exact for a uniform strain.
    let stress = total_force / (h * h);
    let expected_lateral = -poisson_ratio * stress / youngs_modulus * h;
    let lateral = u[grid.node_index(1, 1, 1) * constants::DOF_PER_NODE + 1];
    assert!(
        (lateral - expected_lateral).abs() / expected_lateral.abs() < 1e-3,
        "lateral displacement {lateral} differs from {expected_lateral}"
    );
}

const SMALL_CANTILEVER: &str = r#"
[project]
name = "integration_cantilever"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

# The reproducible backend: this fixture's numbers are compared, and the default
# backend is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.3
min_feature_mm = 8.0
max_iterations = 20
convergence_tol = 0.01

[output]
stl_path = "growforge_integration_cantilever.stl"
iso_level = 0.5
smoothing_iterations = 4

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [80.0, 32.0, 32.0]

[[keepin]]
shape = "box"
min = [72.0, 12.0, 12.0]
max = [80.0, 20.0, 20.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 32.5, 32.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [76.0, 16.0, 16.0], radius = 4.0 }
vector = [0.0, 0.0, -50.0]
"#;

#[test]
fn small_cantilever_runs_end_to_end() {
    let directory = std::env::temp_dir();
    let config = Config::parse(SMALL_CANTILEVER).expect("parse");
    let problem = Problem::build(&config, &directory).expect("build");

    assert_eq!(
        (problem.grid.nx, problem.grid.ny, problem.grid.nz),
        (40, 16, 16)
    );
    assert!(
        problem.warnings.is_empty(),
        "warnings: {:?}",
        problem.warnings
    );

    let outcome = optimize_and_export(&problem, &SilentReporter).expect("run");

    assert_eq!(outcome.field.iterations, 20);
    assert!(
        (outcome.field.volume_fraction - 0.3).abs() < 0.01 * 0.3,
        "final volume fraction {} is not within 1% of the 0.3 target",
        outcome.field.volume_fraction
    );
    assert!(
        outcome.field.compliance < outcome.field.initial_compliance,
        "compliance {} did not improve on the first iteration value {}",
        outcome.field.compliance,
        outcome.field.initial_compliance
    );

    // The exported surface must be watertight; re-validating proves it. The
    // sequence below is the export pass by pass - sample, extract, smooth,
    // clamp, validate - so a resolved cantilever, which comes out in one piece,
    // has to export through the island pass byte for byte the same.
    let field = mesh::node_density_field(
        &problem.grid,
        &outcome.field.densities,
        problem.output.iso_level,
    );
    let mut extracted =
        mesh::marching_cubes::extract(&field, problem.output.iso_level).expect("extraction");
    mesh::smooth::taubin(&mut extracted, problem.output.smoothing_iterations);
    let clamped = mesh::clamp::resolve(&mut extracted, &problem.boundaries, problem.grid.h);
    let stats = mesh::validate::validate(&extracted).expect("watertight surface");
    // The default fidelity is "exact", so the clamp really did run and really
    // did have something to do: sampling on a lattice puts the flat faces of a
    // solid domain up to half a voxel outside it.
    assert_eq!(problem.output.boundaries, BoundaryFidelity::Exact);
    assert_eq!(Some(clamped), outcome.clamp);
    assert!(clamped.vertices_moved > 0, "{clamped:?}");
    assert_eq!(clamped.gave_up, 0, "{clamped:?}");
    assert!(clamped.max_displacement_mm <= problem.grid.h, "{clamped:?}");
    assert_eq!(stats.triangles, outcome.stats.triangles);
    assert!(stats.volume_mm3 > 0.0);

    assert_eq!(outcome.islands.components, 1, "a cantilever is one piece");
    assert_eq!(outcome.islands.bodies.len(), 1);
    assert!(
        outcome.islands.unserved.is_empty(),
        "{:?}",
        outcome.islands.unserved
    );
    assert_eq!(outcome.islands.culled_triangles, 0);
    assert_eq!(stats, outcome.stats);
    assert_eq!(extracted.vertices, outcome.mesh.vertices);
    assert_eq!(extracted.triangles, outcome.mesh.triangles);
    let unculled = directory.join("growforge_integration_cantilever_unculled.stl");
    mesh::stl::write(&unculled, &extracted).expect("write the pre-cull surface");
    assert_eq!(
        std::fs::read(&unculled).expect("pre-cull STL"),
        std::fs::read(&outcome.stl_path).expect("exported STL"),
        "the island pass changed the bytes of a single component export"
    );
    std::fs::remove_file(&unculled).ok();

    // The mesh must stay inside the padded grid block - and, the clamp having
    // run, inside the domain itself: every vertex of it, not just its bounds.
    let bounds = problem.grid.bounds();
    for d in 0..3 {
        assert!(stats.bounds.min[d] >= bounds.min[d] - problem.grid.h - 1e-9);
        assert!(stats.bounds.max[d] <= bounds.max[d] + problem.grid.h + 1e-9);
    }
    for vertex in &outcome.mesh.vertices {
        let outside = problem.boundaries.domain.signed_distance(*vertex);
        assert!(
            outside <= constants::BOUNDARY_CLAMP_EPS_MM,
            "vertex {vertex:?} sits {outside} mm outside the domain"
        );
    }

    // And the file that was written must parse back.
    let (header, triangles) = mesh::stl::read(&outcome.stl_path).expect("read back the STL");
    assert!(String::from_utf8_lossy(&header).starts_with(constants::PROGRAM_NAME));
    assert_eq!(triangles.len(), outcome.stats.triangles);
    std::fs::remove_file(&outcome.stl_path).ok();
}

/// A cantilever carrying the load, and a keepin boss in a lobe of its own with
/// a support and no load. The boss is the larger closed volume of the two.
const DISCONNECTED_BOSS: &str = r#"
[project]
name = "integration_boss"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

# The reproducible backend: this fixture's numbers are compared, and the default
# backend is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.3
min_feature_mm = 8.0
max_iterations = 30

[output]
stl_path = "growforge_integration_boss.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [60.0, 20.0, 20.0]

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 40.0]
max = [24.0, 24.0, 64.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 40.0]
max = [24.0, 24.0, 64.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 20.5, 20.5] }

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, 39.5], max = [0.5, 24.5, 64.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [60.0, 10.0, 10.0], radius = 5.0 }
vector = [0.0, 0.0, -40.0]
"#;

/// Culling keeps everything the configuration declared a purpose for, however
/// many pieces that is and whatever their sizes are.
///
/// The regression this pins is a defect ranking by volume had: the keepin boss
/// encloses more than the load path does, so "the largest component is the part"
/// shipped an STL holding the boss alone, with the whole load-carrying structure
/// culled as a fragment, and the run exited zero with a stress table describing
/// material that was not in the file.
#[test]
fn a_disconnected_keepin_boss_ships_with_the_structure_that_carries_the_load() {
    use growforge::mesh::islands::AnchorKind;

    let directory = std::env::temp_dir();
    let config = Config::parse(DISCONNECTED_BOSS).expect("parse");
    let mut problem = Problem::build(&config, &directory).expect("build");
    problem.output.stl_path = directory.join("growforge_integration_boss.stl");

    let outcome = optimize_and_export(&problem, &SilentReporter).expect("run");
    let islands = &outcome.islands;

    // Two lobes, both declared, both exported.
    assert_eq!(
        islands.bodies.len(),
        2,
        "the export came out as {:?}",
        islands.bodies
    );
    assert_eq!(islands.culled_fragments, 0);
    assert_eq!(islands.culled_triangles, 0);
    assert!(islands.unserved.is_empty(), "{:?}", islands.unserved);

    // The boss is the bigger of the two and holds no load; the structure is the
    // smaller one and holds both the support and the load.
    let boss = &islands.bodies[0];
    let structure = &islands.bodies[1];
    assert!(
        boss.volume_mm3 > structure.volume_mm3,
        "the fixture must have the boss outweigh the structure"
    );
    assert!(boss.anchors.holds(AnchorKind::Keepin));
    assert!(!boss.anchors.holds(AnchorKind::Load));
    assert!(structure.anchors.holds(AnchorKind::Support));
    assert!(
        structure.anchors.holds(AnchorKind::Load),
        "the load path has to be the one that carries the load"
    );

    // Which is to say the file holds both: the load path reaches the tip at
    // x = 60 and the boss stands to z = 64.
    assert_eq!(
        growforge::mesh::islands::label_components(&outcome.mesh).components,
        2 + islands.cavity_shells
    );
    assert!(
        outcome.stats.bounds.max[0] > 55.0,
        "the load path is missing from the export: {:?}",
        outcome.stats.bounds
    );
    assert!(
        outcome.stats.bounds.max[2] > 60.0,
        "the boss is missing from the export: {:?}",
        outcome.stats.bounds
    );
    mesh::validate::validate(&outcome.mesh).expect("watertight surface");
    let (_, triangles) = mesh::stl::read(&outcome.stl_path).expect("read back the STL");
    assert_eq!(triangles.len(), outcome.stats.triangles);
    std::fs::remove_file(&outcome.stl_path).ok();
}

/// Two lumps in domain lobes of their own, each holding a support and a load of
/// its own and joined to the other by nothing.
const TWO_ANCHORED_LUMPS: &str = r#"
[project]
name = "integration_two_lumps"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "pla"

# The reproducible backend: this fixture's numbers are read, and the default
# backend is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.5
min_feature_mm = 16.0

[output]
stl_path = "growforge_integration_two_lumps.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [24.0, 16.0, 16.0]

[[domain]]
op = "add"
shape = "box"
min = [40.0, 0.0, 0.0]
max = [64.0, 16.0, 16.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 16.5, 16.5] }

[[supports]]
region = { shape = "box", min = [63.5, -0.5, -0.5], max = [64.5, 16.5, 16.5] }

[[loadcases]]
name = "pull"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [19.5, -0.5, -0.5], max = [24.5, 16.5, 16.5] }
vector = [0.0, 0.0, -30.0]
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [39.5, -0.5, -0.5], max = [44.5, 16.5, 16.5] }
vector = [0.0, 0.0, 30.0]
"#;

/// A pipeline whose export comes out in two pieces reports a safety factor with
/// the warning that says what it is a factor of.
///
/// The incident: a part meant to link two rods was exported as two separate
/// bodies - each load group shunting into its own local support anchor - and
/// every surface quoted "safety factor 5.07" for a thing whose two halves are
/// held together by air. Nothing here can know that the supports are fictitious:
/// they are what the configuration declared, every load reaches one through
/// material, and the solve is a correct reading of that model. What the run does
/// know is that the surface it wrote is in pieces, and that is what it now says
/// beside the number.
///
/// The whole post-run pipeline over a saturated field, which is the sequence
/// `growforge run` executes once its engine has handed a design over: the same
/// analysis, the same trim, the same export, and the same `Completed` the command
/// line prints its stress block from.
#[test]
fn an_export_in_two_bodies_reports_its_safety_factor_with_a_warning() {
    let directory = std::env::temp_dir();
    let config = Config::parse(TWO_ANCHORED_LUMPS).expect("parse");
    let mut problem = Problem::build(&config, &directory).expect("build");
    problem.output.stl_path = directory.join("growforge_integration_two_lumps.stl");

    let mut densities = vec![1.0; problem.grid.n_cells()];
    let completed = growforge::complete(
        &problem,
        &mut densities,
        growforge::stress::StressLimits::default(),
        &growforge::Unwatched,
    )
    .expect("the pipeline")
    .expect("an unwatched pipeline is never stopped");

    // The fixture is the case it was written for: two bodies, neither of them
    // debris, and a converged stress report to be warned about.
    let bodies = completed.islands.bodies.len();
    assert_eq!(bodies, 2, "the export came out as {bodies} bodies");
    assert_eq!(completed.islands.culled_fragments, 0);
    assert!(
        completed.islands.unserved.is_empty(),
        "{:?}",
        completed.islands.unserved
    );
    let report = completed
        .stress
        .report()
        .unwrap_or_else(|| panic!("no stress report: {:?}", completed.stress.reason()));

    // The summary the panel and the editor's console read, from the count the
    // island report of the written surface carries.
    let summary = report.summary(bodies);
    assert_eq!(
        summary.warning.as_deref(),
        Some(
            "warning: the export is 2 separate bodies - this safety factor describes each piece \
             against its own supports, not one connected part"
        )
    );
    assert_eq!(
        summary.lines()[0],
        summary.warning.clone().expect("warning")
    );
    assert!(
        summary.headline.starts_with("safety factor ") && !summary.headline.contains("n/a"),
        "the fixture stopped producing a factor to warn about: {}",
        summary.headline
    );
    // The one-body reading of the same report is the summary this always gave,
    // warning and all its absence.
    assert_eq!(report.summary(1).warning, None);
    assert_eq!(report.summary(1).headline, summary.headline);

    // And the console block `growforge run` prints, from the same two values it
    // is handed in `main`.
    growforge::report::print_stress_report(&completed.stress, bodies);
    growforge::report::print_island_report(&completed.islands, problem.optimization.min_feature_mm);

    mesh::validate::validate(&completed.mesh).expect("watertight surface");
    std::fs::remove_file(&problem.output.stl_path).ok();
}

/// The same cantilever, printed standing on its z = 0 face and carrying its own
/// weight. Everything phase 3 added is exercised at once and has to leave a
/// design that is watertight, self supporting and reported on.
const PRINTABLE_CANTILEVER: &str = r#"
[project]
name = "integration_printable"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "petg"

# The reproducible backend: this fixture's numbers are compared, and the default
# backend is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.3
min_feature_mm = 12.0
max_iterations = 30
convergence_tol = 0.01

  [optimization.overhang]
  build_direction = "z+"

[output]
stl_path = "growforge_integration_printable.stl"
iso_level = 0.5
smoothing_iterations = 4
voids = "fill"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [64.0, 24.0, 64.0]

[[keepin]]
shape = "box"
min = [56.0, 8.0, 52.0]
max = [64.0, 16.0, 60.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 24.5, 64.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "sphere", center = [60.0, 12.0, 56.0], radius = 5.0 }
vector = [0.0, 0.0, -80.0]
[[loadcases.loads]]
type = "gravity"
"#;

#[test]
fn overhang_gravity_voids_and_stress_run_end_to_end() {
    let directory = std::env::temp_dir();
    let config = Config::parse(PRINTABLE_CANTILEVER).expect("parse");
    let problem = Problem::build(&config, &directory).expect("build");
    assert!(problem.has_gravity());
    assert!(problem.optimization.overhang.is_some());
    assert_eq!(problem.output.voids, growforge::config::VoidPolicy::Fill);

    let outcome = optimize_and_export(&problem, &SilentReporter).expect("run");

    // The overhang filter is part of the chain, so the exported field is the
    // printed one and the residual is finite and bounded by the density range.
    let residual = outcome
        .field
        .overhang_residual
        .expect("an overhang residual");
    assert!(residual.max.is_finite() && residual.max <= 1.0 + 1e-9);
    assert!(residual.mean >= 0.0 && residual.mean <= residual.max);

    // Every printed density has to be reachable from the layer below: no cell
    // may carry material that its supporting region cannot hold up.
    let grid = &problem.grid;
    let iso = problem.output.iso_level;
    for k in 1..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let cell = grid.cell_index(i, j, k);
                if grid.cells[cell] != CellKind::Design || outcome.field.densities[cell] < iso {
                    continue;
                }
                let supported = [(0i32, 0i32), (-1, 0), (1, 0), (0, -1), (0, 1)]
                    .into_iter()
                    .filter_map(|(di, dj)| {
                        let (si, sj) = (i as i32 + di, j as i32 + dj);
                        (si >= 0 && sj >= 0 && si < grid.nx as i32 && sj < grid.ny as i32)
                            .then(|| grid.cell_index(si as usize, sj as usize, k - 1))
                    })
                    .any(|below| match grid.cells[below] {
                        CellKind::Void => false,
                        CellKind::Solid => true,
                        CellKind::Design => outcome.field.densities[below] >= iso,
                    });
                assert!(
                    supported,
                    "cell ({i}, {j}, {k}) prints at {} with nothing under it",
                    outcome.field.densities[cell]
                );
            }
        }
    }

    // Filling leaves no trapped cavity behind in the exported field.
    assert_eq!(
        growforge::voids::detect(grid, &outcome.field.densities, iso)
            .iter()
            .filter(|v| v.fillable)
            .count(),
        0
    );
    assert_eq!(outcome.voids.policy, growforge::config::VoidPolicy::Fill);

    // The stress report covers the one load case and yields a safety factor
    // against the PETG preset's yield strength.
    let stress = outcome
        .stress
        .report()
        .unwrap_or_else(|| panic!("no stress report: {:?}", outcome.stress.reason()));
    assert_eq!(stress.cases.len(), 1);
    let case = &stress.cases[0];
    assert!(case.max_mpa > 0.0 && case.max_mpa.is_finite());
    assert!(case.percentile_mpa <= case.max_mpa);
    assert!(case.top_fraction_mean_mpa <= case.max_mpa);
    let factor = case.safety_factor.expect("a safety factor");
    assert!((factor - 47.0 / case.max_mpa).abs() < 1e-9);
    assert_eq!(case.von_mises.len(), grid.n_cells());

    // And the surface written from that field is watertight.
    let stats = mesh::validate::validate(&outcome.mesh).expect("watertight surface");
    assert_eq!(stats.triangles, outcome.stats.triangles);
    assert!(stats.volume_mm3 > 0.0);
    std::fs::remove_file(&outcome.stl_path).ok();
}

/// A block a global volume target alone would answer with a couple of thick
/// members, under a local volume cap and the moving asymptotes step the cap
/// needs. Small enough to run its whole pipeline in a test, and porous enough
/// that the cap is doing something.
const CAPPED_BLOCK: &str = r#"
[project]
name = "integration_capped_block"

# The reproducible backend: the numbers below are compared, and the default
# backend is the machine's compute device.
[solver]
backend = "cpu"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[optimization]
mass_fraction = 0.4
min_feature_mm = 6.0
max_iterations = 40
convergence_tol = 1e-4
update = "mma"

  [optimization.local_volume]
  max_fraction = 0.6

[output]
stl_path = "growforge_integration_capped_block.stl"
iso_level = 0.5
smoothing_iterations = 2

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [60.0, 6.0, 24.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [6.5, 6.5, 0.5] }

[[supports]]
region = { shape = "box", min = [53.5, -0.5, -0.5], max = [60.5, 6.5, 0.5] }

[[loadcases]]
name = "middle"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [24.0, -0.5, 23.5], max = [36.0, 6.5, 24.5] }
vector = [0.0, 0.0, -200.0]
"#;

/// The local volume cap end to end: a run that completes, a design that really
/// holds the cap it was given, and a watertight export of it.
#[test]
fn a_local_volume_capped_run_completes_and_exports() {
    use growforge::config::UpdateScheme;
    use growforge::engine::local_volume::LocalVolume;

    let directory = std::env::temp_dir();
    let config = Config::parse(CAPPED_BLOCK).expect("parse");
    let problem = Problem::build(&config, &directory).expect("build");
    let cap = problem.optimization.local_volume.expect("a cap");
    assert_eq!(problem.optimization.update, UpdateScheme::Mma);
    assert_eq!(cap.max_fraction, 0.6);
    // Three density filter radii, which is three halves of min_feature_mm.
    assert_eq!(
        cap.radius_mm,
        constants::LOCAL_VOLUME_RADIUS_FACTOR * problem.optimization.filter_radius_mm
    );
    assert!(
        problem.warnings.is_empty(),
        "warnings: {:?}",
        problem.warnings
    );

    let outcome = optimize_and_export(&problem, &SilentReporter).expect("run");
    assert!(outcome.field.iterations > 1);

    // What the design really holds, measured the way the run measures it.
    let design_cells: Vec<usize> = (0..problem.grid.n_cells())
        .filter(|&e| problem.grid.cells[e] == CellKind::Design)
        .collect();
    let measured =
        LocalVolume::new(&problem.grid, cap).measure(&outcome.field.densities, &design_cells);
    println!(
        "capped block: aggregate {:.4} against the {:.2} cap, worst neighbourhood {:.4}, volume \
         {:.4} against the 0.4 target, {} iterations",
        measured.aggregate,
        cap.max_fraction,
        measured.worst,
        outcome.field.volume_fraction,
        outcome.field.iterations
    );
    // The constraint as the run states it, met.
    assert!(
        measured.aggregate <= cap.max_fraction + 1e-3,
        "the aggregate the cap is stated on was left at {}",
        measured.aggregate
    );
    // And the true worst neighbourhood behind it, within the p-mean's
    // documented under-estimate.
    assert!(
        measured.worst <= cap.max_fraction + 0.12,
        "a neighbourhood was left at {}",
        measured.worst
    );
    // The volume target is an upper bound under a cap: the design may use less
    // than it is allowed, and never more.
    assert!(
        outcome.field.volume_fraction <= 0.4 + 1e-3,
        "the run went over its volume target: {}",
        outcome.field.volume_fraction
    );

    // And the design exports like any other: watertight, in one piece, with
    // every declared region still served.
    let stats = mesh::validate::validate(&outcome.mesh).expect("watertight surface");
    assert_eq!(stats.triangles, outcome.stats.triangles);
    assert!(stats.volume_mm3 > 0.0);
    assert!(
        outcome.islands.unserved.is_empty(),
        "{:?}",
        outcome.islands.unserved
    );
    let (header, triangles) = mesh::stl::read(&outcome.stl_path).expect("read back the STL");
    assert!(String::from_utf8_lossy(&header).starts_with(constants::PROGRAM_NAME));
    assert_eq!(triangles.len(), outcome.stats.triangles);
    std::fs::remove_file(&outcome.stl_path).ok();
}

fn example_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

#[test]
fn the_shelf_bracket_example_exercises_every_phase_three_feature() {
    let path = example_path("shelf_bracket.toml");
    let (config, problem) = growforge::load_config_and_problem(&path).expect("build");
    assert!(problem.warnings.is_empty(), "{:?}", problem.warnings);
    assert_eq!(problem.optimization.overhang.unwrap().label(), "z+");
    // The one shipped example that selects the moving asymptotes update: the
    // optimality criteria step cannot settle this geometry, and the example is
    // what the README's guidance points at. Pinned so the choice cannot be
    // dropped by accident.
    assert_eq!(
        problem.optimization.update,
        growforge::config::UpdateScheme::Mma
    );
    assert_eq!(problem.optimization.max_iterations, 150);
    assert_eq!(problem.output.voids, growforge::config::VoidPolicy::Warn);
    assert!(problem.output.stress_json.is_some());
    assert_eq!(problem.material.yield_strength_mpa, Some(47.0));

    // One load case that mixes a distributed force with self weight.
    assert_eq!(problem.load_cases.len(), 1);
    let case = &problem.load_cases[0];
    assert_eq!(case.loads.len(), 2);
    assert_eq!(case.loads[0].kind, "force");
    assert_eq!(case.loads[1].kind, "gravity");
    let acceleration = case.gravity.expect("self weight");
    assert!((growforge::geometry::length(acceleration) - 9810.0).abs() < 1e-9);
    assert!(acceleration[2] < 0.0, "gravity must pull downwards");
    assert!(config.loadcases[0].loads[1].region().is_none());
}

/// Cells that carry material at the iso level and are reachable from the
/// support regions through face connected material, which is the same
/// connectivity the enclosed cavity pass uses.
fn reachable_from_supports(problem: &Problem, densities: &[f64]) -> Vec<bool> {
    let grid = &problem.grid;
    let iso = problem.output.iso_level;
    let solid = |cell: usize| growforge::engine::cell_density(grid, densities, cell) >= iso;

    let mut reached = vec![false; grid.n_cells()];
    let mut stack: Vec<usize> = problem
        .supports
        .iter()
        .flat_map(|s| cells_touching(grid, &s.nodes))
        .filter(|&c| solid(c))
        .collect();
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
    reached
}

/// The cells a set of nodes touches, whatever their classification.
fn cells_touching(grid: &Grid, nodes: &[usize]) -> Vec<usize> {
    let mut cells = Vec::new();
    for &node in nodes {
        let (ni, nj, nk) = grid.node_ijk(node);
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let (i, j, k) = (ni + di, nj + dj, nk + dk);
                    if i == 0 || j == 0 || k == 0 {
                        continue;
                    }
                    let (i, j, k) = (i - 1, j - 1, k - 1);
                    if i < grid.nx && j < grid.ny && k < grid.nz {
                        cells.push(grid.cell_index(i, j, k));
                    }
                }
            }
        }
    }
    cells.sort_unstable();
    cells.dedup();
    cells
}

/// The growth engine end to end on the shipped example: a load path to every
/// foot, a watertight STL, a stress report, and the same bytes every time.
#[test]
fn the_growth_canopy_example_runs_end_to_end() {
    let directory = std::env::temp_dir();
    let path = example_path("growth_canopy.toml");
    let (_, mut problem) = growforge::load_config_and_problem(&path).expect("build");
    // Never write into the source tree from a test.
    problem.output.stl_path = directory.join("growforge_growth_canopy.stl");
    problem.output.stress_json = None;
    assert_eq!(problem.engine, "growth");
    assert!(problem.growth.is_some());
    assert!(problem.warnings.is_empty(), "{:?}", problem.warnings);

    let outcome = optimize_and_export(&problem, &SilentReporter).expect("run");
    let summary = outcome.field.growth.expect("a growth summary");

    // One guaranteed load path per (load region, support region) pair: the one
    // tabletop load reaching all four feet.
    assert_eq!(summary.backbones, 4);
    assert!(summary.segments > 50, "only {} segments", summary.segments);
    assert!(summary.consumed > 0 && summary.consumed <= summary.attractors);
    assert_eq!(summary.clamped_volume_fraction, None);
    assert!(
        (outcome.field.volume_fraction - 0.12).abs() <= 0.01 * 0.12,
        "volume fraction {} missed the 0.12 target",
        outcome.field.volume_fraction
    );
    assert_eq!(outcome.field.compliance, 0.0, "growth solves nothing");
    assert!(summary.pruned_nodes > 0, "nothing was pruned at all");
    assert!(
        summary.fused_tips > 0,
        "no branch tip fused to the tabletop, so nothing but the backbones carries it"
    );

    // The whole part is one connected body. A branch that fused to nothing but
    // itself would come out as a floating island, which is watertight, manifold
    // and encloses no cavity, so this is the only check in the pipeline that
    // can see it.
    assert_eq!(
        outcome.solids.len(),
        1,
        "the field came out in {} pieces: {:?}",
        outcome.solids.len(),
        outcome.solids.iter().skip(1).collect::<Vec<_>>()
    );
    // And so is the surface that ships, which is the reading the field's own
    // flood fill cannot make. A shipped example that culled anything would be a
    // shipped example whose recorded output had changed.
    assert_eq!(
        outcome.islands.components, 1,
        "the exported surface came out in {} components: {:?}",
        outcome.islands.components, outcome.islands.fragments
    );
    assert_eq!(outcome.islands.bodies.len(), 1);
    assert_eq!(outcome.islands.culled_triangles, 0);
    assert_eq!(outcome.islands.culled_volume_mm3, 0.0);
    assert!(
        outcome.islands.unserved.is_empty(),
        "{:?}",
        outcome.islands.unserved
    );

    // No branch may end in mid air. A free tip carries no load, cannot be
    // printed without support under it, and is what a user reported as the
    // model looking unfinished.
    let grown = growforge::engine::growth::grow(&problem, &SilentReporter).expect("grow");
    let free = grown.free_leaves(&problem.grid);
    assert!(
        free.is_empty(),
        "{} of the {} skeleton nodes end on nothing",
        free.len(),
        grown.skeleton.len()
    );
    assert!(grown.skeleton.len() > 50);

    // Every load region is connected to a support through material.
    let reached = reachable_from_supports(&problem, &outcome.field.densities);
    for case in &problem.load_cases {
        for load in &case.loads {
            if load.nodes.is_empty() {
                continue;
            }
            let cells = cells_touching(&problem.grid, &load.nodes);
            assert!(
                cells.iter().any(|&c| reached[c]),
                "the load region of \"{}\" never reaches a support",
                case.name
            );
        }
    }

    // The exported surface is watertight, and the run reported a stress.
    let stats = mesh::validate::validate(&outcome.mesh).expect("watertight surface");
    assert_eq!(stats.triangles, outcome.stats.triangles);
    assert!(stats.volume_mm3 > 0.0);
    let stress = outcome
        .stress
        .report()
        .unwrap_or_else(|| panic!("no stress report: {:?}", outcome.stress.reason()));
    assert_eq!(stress.cases.len(), 1);
    let case = &stress.cases[0];
    assert!(case.max_mpa > 0.0 && case.max_mpa.is_finite());
    assert!(case.safety_factor.expect("a safety factor") > 0.0);
    // Filling left no trapped cavity behind.
    assert_eq!(
        growforge::voids::detect(
            &problem.grid,
            &outcome.field.densities,
            problem.output.iso_level
        )
        .iter()
        .filter(|v| v.fillable)
        .count(),
        0
    );

    // The same configuration grows the same file, byte for byte. Both halves
    // are grown and exported the same way, so what is compared is the engine
    // and the exporter rather than the post-processing in between.
    let mut first = problem.clone();
    first.output.stl_path = directory.join("growforge_growth_canopy_a.stl");
    let mut second = problem.clone();
    second.output.stl_path = directory.join("growforge_growth_canopy_b.stl");
    let grown_a = growforge::optimize(&first, &SilentReporter).expect("regrow");
    let grown_b = growforge::optimize(&second, &SilentReporter).expect("regrow");
    assert_eq!(
        grown_a.densities, grown_b.densities,
        "the same configuration grew two different density fields"
    );
    growforge::export(&first, &grown_a.densities).expect("export");
    growforge::export(&second, &grown_b.densities).expect("export");
    assert_eq!(
        std::fs::read(&first.output.stl_path).expect("first STL"),
        std::fs::read(&second.output.stl_path).expect("second STL"),
        "the same configuration produced two different STL files"
    );

    std::fs::remove_file(&outcome.stl_path).ok();
    std::fs::remove_file(&first.output.stl_path).ok();
    std::fs::remove_file(&second.output.stl_path).ok();
}

/// The symmetric shipped example: one quarter grown, four identical legs
/// exported, and a field that really is symmetric about both declared planes.
#[test]
fn the_symmetric_growth_canopy_example_grows_four_identical_legs() {
    let directory = std::env::temp_dir();
    let path = example_path("growth_canopy_symmetric.toml");
    let (_, mut problem) = growforge::load_config_and_problem(&path).expect("build");
    // Never write into the source tree from a test.
    problem.output.stl_path = directory.join("growforge_growth_canopy_symmetric.stl");
    problem.output.stress_json = None;
    assert_eq!(problem.engine, "growth");
    assert!(problem.warnings.is_empty(), "{:?}", problem.warnings);

    let outcome = optimize_and_export(&problem, &SilentReporter).expect("run");
    let summary = outcome.field.growth.expect("a growth summary");
    let symmetry = summary.symmetry.expect("a symmetry summary");
    assert_eq!(symmetry.params.sectors(), 4);
    assert_eq!(
        symmetry.fundamental_design_cells * 4,
        problem.counts.design,
        "the fundamental domain is a quarter of the design cells"
    );

    // One backbone: the one foot whose centre is in the grown quarter. The
    // plain example routes four, one per foot.
    assert_eq!(summary.backbones, 1);
    assert!(summary.segments >= 4 && summary.segments.is_multiple_of(4));
    assert_eq!(summary.clamped_volume_fraction, None);
    assert!(
        (outcome.field.volume_fraction - 0.12).abs() <= 0.01 * 0.12,
        "volume fraction {} missed the 0.12 target over the whole structure",
        outcome.field.volume_fraction
    );

    // The whole part is one connected body: every strut that reached a mirror
    // plane met its own reflection there. The exported surface too, so this
    // example's recorded output is what it was before the island pass existed.
    assert_eq!(
        outcome.solids.len(),
        1,
        "the field came out in {} pieces",
        outcome.solids.len()
    );
    assert_eq!(
        outcome.islands.components, 1,
        "the exported surface came out in {} components: {:?}",
        outcome.islands.components, outcome.islands.fragments
    );
    assert_eq!(outcome.islands.culled_triangles, 0);

    // And it is symmetric. Both planes pass through the centre of the grid
    // block, whose axes hold an even number of cells, so cell (i, j, k) mirrors
    // onto (nx - 1 - i, j, k) and (i, ny - 1 - j, k) exactly.
    let grid = &problem.grid;
    let densities = &outcome.field.densities;
    let mut worst = 0.0f64;
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let here = densities[grid.cell_index(i, j, k)];
                for image in [
                    grid.cell_index(grid.nx - 1 - i, j, k),
                    grid.cell_index(i, grid.ny - 1 - j, k),
                    grid.cell_index(grid.nx - 1 - i, grid.ny - 1 - j, k),
                ] {
                    worst = worst.max((here - densities[image]).abs());
                }
            }
        }
    }
    assert!(
        worst <= 1e-12,
        "the exported field differs from its own mirror by {worst}"
    );
    assert!(densities.iter().all(|d| (0.0..=1.0).contains(d)));

    // The exported surface is watertight and the run reported a stress.
    let stats = mesh::validate::validate(&outcome.mesh).expect("watertight surface");
    assert_eq!(stats.triangles, outcome.stats.triangles);
    assert!(stats.volume_mm3 > 0.0);
    let stress = outcome
        .stress
        .report()
        .unwrap_or_else(|| panic!("no stress report: {:?}", outcome.stress.reason()));
    assert_eq!(stress.cases.len(), 1);
    assert!(stress.cases[0].safety_factor.expect("a safety factor") > 1.0);

    // The same configuration grows the same file, byte for byte, symmetry and
    // all: the transforms are exact arithmetic and the copies are unioned in a
    // fixed order.
    let mut first = problem.clone();
    first.output.stl_path = directory.join("growforge_symmetric_canopy_a.stl");
    let mut second = problem.clone();
    second.output.stl_path = directory.join("growforge_symmetric_canopy_b.stl");
    let grown_a = growforge::optimize(&first, &SilentReporter).expect("regrow");
    let grown_b = growforge::optimize(&second, &SilentReporter).expect("regrow");
    assert_eq!(grown_a.densities, grown_b.densities);
    growforge::export(&first, &grown_a.densities).expect("export");
    growforge::export(&second, &grown_b.densities).expect("export");
    assert_eq!(
        stl_hash(&first.output.stl_path),
        stl_hash(&second.output.stl_path),
        "two symmetric runs of the same configuration hashed differently"
    );

    std::fs::remove_file(&outcome.stl_path).ok();
    std::fs::remove_file(&first.output.stl_path).ok();
    std::fs::remove_file(&second.output.stl_path).ok();
}

/// A six-fold problem: a round column carrying a load on a cap plate and
/// standing on the floor. Every region is a body of revolution, so the problem
/// is symmetric under any rotation about z - and a sixth of a turn is one of the
/// angles the voxel lattice is *not* symmetric under, which is the point of it.
const SIX_FOLD_COLUMN: &str = r#"
[project]
name = "integration_six_fold"
engine = "growth"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "petg"

# The reproducible backend: this fixture's numbers are compared, and the default
# backend is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.15
min_feature_mm = 12.0

[growth]
seed = 4242

  [growth.symmetry]
  kind = "rotational"
  order = 6
  axis = "z"

[output]
stl_path = "growforge_integration_six_fold.stl"
voids = "fill"

[[domain]]
op = "add"
shape = "cylinder"
p1 = [0.0, 0.0, 0.0]
p2 = [0.0, 0.0, 80.0]
radius = 60.0

[[keepin]]
shape = "cylinder"
p1 = [0.0, 0.0, 72.0]
p2 = [0.0, 0.0, 80.0]
radius = 60.0

[[supports]]
region = { shape = "cylinder", p1 = [0.0, 0.0, -1.0], p2 = [0.0, 0.0, 1.0], radius = 60.0 }

[[loadcases]]
name = "cap"
[[loadcases.loads]]
type = "force"
region = { shape = "cylinder", p1 = [0.0, 0.0, 76.0], p2 = [0.0, 0.0, 81.0], radius = 60.0 }
vector = [0.0, 0.0, -400.0]
"#;

/// Cell holding `p`, or `None` outside the block.
fn cell_holding(grid: &Grid, p: [f64; 3]) -> Option<usize> {
    let counts = [grid.nx, grid.ny, grid.nz];
    let mut ijk = [0usize; 3];
    for d in 0..3 {
        let local = (p[d] - grid.origin[d]) / grid.h;
        if !local.is_finite() || local < 0.0 || local >= counts[d] as f64 {
            return None;
        }
        ijk[d] = local.floor() as usize;
    }
    Some(grid.cell_index(ijk[0], ijk[1], ijk[2]))
}

/// True when a cell and all 26 of its neighbours carry the same saturated
/// density, so it lies more than a voxel from every surface and from every
/// keepout or keepin boundary.
fn deep_inside_or_outside(grid: &Grid, densities: &[f64], cell: usize) -> bool {
    let here = densities[cell];
    if here > 1e-9 && here < 1.0 - 1e-9 {
        return false;
    }
    let (i, j, k) = grid.cell_ijk(cell);
    for dk in -1i32..=1 {
        for dj in -1i32..=1 {
            for di in -1i32..=1 {
                let (ni, nj, nk) = (i as i32 + di, j as i32 + dj, k as i32 + dk);
                if ni < 0
                    || nj < 0
                    || nk < 0
                    || ni >= grid.nx as i32
                    || nj >= grid.ny as i32
                    || nk >= grid.nz as i32
                {
                    return false;
                }
                let neighbour = grid.cell_index(ni as usize, nj as usize, nk as usize);
                if (densities[neighbour] - here).abs() > 1e-9 {
                    return false;
                }
            }
        }
    }
    true
}

/// A rotation the voxel lattice is not symmetric under still produces an exact
/// skeleton, and a field whose asymmetry is **bounded by the sampling**.
///
/// The skeleton's copies are images of one node set under an exact matrix, so
/// the *continuous* structure is symmetric to the last bit. The field is that
/// structure sampled at cell centres, and a sixth of a turn takes a cell centre
/// to a point inside another cell rather than to its centre - up to half a
/// diagonal, `h sqrt(3) / 2`, away from where that cell is sampled. The density
/// is a smoothstep one voxel wide over a one-Lipschitz distance, so two samples
/// that far apart can differ by as much as a whole density **wherever the field
/// varies at all**, and not at all where it does not. That is exactly what this
/// asserts: every cell whose own 26-neighbourhood carries one saturated value -
/// more than a voxel from any surface, keepout or keepin boundary - matches its
/// image exactly, and the cells that differ are all in the band between. The
/// fraction of them is capped as a regression guard, with margin over what this
/// fixture measures.
#[test]
fn a_rotation_off_the_voxel_lattice_stays_symmetric_to_within_a_voxel() {
    use growforge::engine::growth::symmetry::Symmetry;

    let directory = std::env::temp_dir();
    let config = Config::parse(SIX_FOLD_COLUMN).expect("parse");
    let mut problem = Problem::build(&config, &directory).expect("build");
    problem.output.stl_path = directory.join("growforge_integration_six_fold.stl");
    assert!(problem.warnings.is_empty(), "{:?}", problem.warnings);

    let outcome = optimize_and_export(&problem, &SilentReporter).expect("run");
    let summary = outcome.field.growth.expect("a growth summary");
    let symmetry_summary = summary.symmetry.expect("a symmetry summary");
    assert_eq!(symmetry_summary.params.sectors(), 6);
    assert!(
        !symmetry_summary.exact_on_the_voxel_lattice,
        "a sixth of a turn does not land on cell centres and must not claim to"
    );
    assert!(summary.segments.is_multiple_of(6) && summary.segments >= 6);
    assert_eq!(
        outcome.solids.len(),
        1,
        "the field came out in {} pieces",
        outcome.solids.len()
    );
    mesh::validate::validate(&outcome.mesh).expect("watertight surface");

    let grid = &problem.grid;
    let densities = &outcome.field.densities;
    let symmetry = Symmetry::new(
        problem.growth.expect("params").symmetry.expect("symmetry"),
        &grid.bounds(),
        constants::GROWTH_SYMMETRY_BOUNDARY_EPSILON_VOXELS * grid.h,
    );

    // The fixture has to have grown something off the axis, or the comparison
    // below would be about a column that is trivially symmetric.
    let off_axis = (0..grid.n_cells())
        .filter(|&cell| {
            let (i, j, k) = grid.cell_ijk(cell);
            let centre = grid.cell_center(i, j, k);
            densities[cell] > 0.5 && centre[0].hypot(centre[1]) > 20.0
        })
        .count();
    assert!(
        off_axis > 100,
        "only {off_axis} cells of canopy off the axis"
    );

    let mut compared = 0usize;
    let mut differing = 0usize;
    let mut over_threshold = 0usize;
    let mut worst = 0.0f64;
    for cell in 0..grid.n_cells() {
        let (i, j, k) = grid.cell_ijk(cell);
        let Some(image) = cell_holding(grid, symmetry.image(1, grid.cell_center(i, j, k))) else {
            continue;
        };
        compared += 1;
        let difference = (densities[cell] - densities[image]).abs();
        if difference <= 1e-9 {
            continue;
        }
        differing += 1;
        worst = worst.max(difference);
        if difference > 0.1 {
            over_threshold += 1;
        }
        assert!(
            !deep_inside_or_outside(grid, densities, cell)
                || !deep_inside_or_outside(grid, densities, image),
            "cell {cell} differs from its image by {difference} with both of them more than a \
             voxel from any surface, which sampling cannot explain"
        );
    }
    assert!(compared > grid.n_cells() / 2, "only {compared} comparisons");
    // A regression guard rather than a physical bound: the cells that can
    // differ are the surface band, which is a small share of the block, and
    // this fixture measures 6.7 % of them over a tenth of a density (worst
    // case a full one, which is what a half-diagonal of resampling across a
    // one-voxel smoothstep is worth). Ten per cent leaves margin for the band
    // moving with the canopy without letting a real loss of symmetry through.
    let share = over_threshold as f64 / compared as f64;
    assert!(
        share <= 0.10,
        "{:.1} % of cells differ from their image by more than 0.1 ({differing} differ at all, \
         worst {worst:.3})",
        share * 100.0
    );

    // And it is still deterministic: the same configuration exports the same
    // bytes, whatever the sampling does.
    let mut first = problem.clone();
    first.output.stl_path = directory.join("growforge_six_fold_a.stl");
    let mut second = problem.clone();
    second.output.stl_path = directory.join("growforge_six_fold_b.stl");
    let grown_a = growforge::optimize(&first, &SilentReporter).expect("regrow");
    let grown_b = growforge::optimize(&second, &SilentReporter).expect("regrow");
    assert_eq!(grown_a.densities, grown_b.densities);
    growforge::export(&first, &grown_a.densities).expect("export");
    growforge::export(&second, &grown_b.densities).expect("export");
    assert_eq!(
        stl_hash(&first.output.stl_path),
        stl_hash(&second.output.stl_path),
        "two runs of the same six-fold configuration hashed differently"
    );

    std::fs::remove_file(&outcome.stl_path).ok();
    std::fs::remove_file(&first.output.stl_path).ok();
    std::fs::remove_file(&second.output.stl_path).ok();
}

/// Hash of a file's bytes with the standard library's own hasher, which is
/// seeded from fixed keys and so gives the same value for the same bytes
/// throughout a run.
fn stl_hash(path: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Determinism has to survive the refined lattice: the same grown field
/// supersampled twice must still export the same file. The refinement is a pure
/// function of the node field and the extraction welds on the refined lattice's
/// own edge identities, so nothing in it may depend on iteration or thread
/// order.
#[test]
fn a_supersampled_growth_run_exports_the_same_file_twice() {
    let directory = std::env::temp_dir();
    let path = example_path("growth_canopy.toml");
    let (_, problem) = growforge::load_config_and_problem(&path).expect("build");
    assert_eq!(problem.engine, "growth");

    let mut first = problem.clone();
    first.output.supersample = 2;
    first.output.stress_json = None;
    first.output.stl_path = directory.join("growforge_supersample_canopy_a.stl");
    let mut second = first.clone();
    second.output.stl_path = directory.join("growforge_supersample_canopy_b.stl");

    let grown_a = growforge::optimize(&first, &SilentReporter).expect("grow");
    let grown_b = growforge::optimize(&second, &SilentReporter).expect("grow");
    let stats_a = growforge::export(&first, &grown_a.densities)
        .expect("export")
        .stats;
    let stats_b = growforge::export(&second, &grown_b.densities)
        .expect("export")
        .stats;

    assert_eq!(stats_a, stats_b);
    assert_eq!(
        stl_hash(&first.output.stl_path),
        stl_hash(&second.output.stl_path),
        "two supersampled runs of the same configuration hashed differently"
    );
    assert_eq!(
        std::fs::read(&first.output.stl_path).expect("first STL"),
        std::fs::read(&second.output.stl_path).expect("second STL"),
        "two supersampled runs of the same configuration wrote different bytes"
    );

    // And the refinement really did something: four times the triangles of the
    // same field exported unrefined, still watertight.
    let mut plain = first.clone();
    plain.output.supersample = 1;
    plain.output.stl_path = directory.join("growforge_supersample_canopy_plain.stl");
    let plain_export = growforge::export(&plain, &grown_a.densities).expect("export");
    mesh::validate::validate(&plain_export.mesh).expect("watertight surface");
    let ratio = stats_a.triangles as f64 / plain_export.stats.triangles as f64;
    assert!(
        (3.0..5.0).contains(&ratio),
        "supersample 2 gave {} triangles against {}, a ratio of {ratio:.2}",
        stats_a.triangles,
        plain_export.stats.triangles
    );

    for problem in [&first, &second, &plain] {
        std::fs::remove_file(&problem.output.stl_path).ok();
    }
}

/// A cantilever with a turned ellipsoid keepout through the middle of it, held
/// by an ellipsoid support region and loaded through an ellipsoid one: the
/// shape in every position that has one, on a problem small enough to run.
const ELLIPSOID_KEEPOUT: &str = r#"
[project]
name = "integration_ellipsoid"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

# The reproducible backend: what is compared here is this fixture's geometry,
# and the default backend is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.3
min_feature_mm = 8.0
max_iterations = 15

[output]
stl_path = "growforge_integration_ellipsoid.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [64.0, 32.0, 32.0]

# The part this exists for: a keepout between the two ends, turned off the
# world axes so that its bounding box is not its shape.
[[keepout]]
shape = "ellipsoid"
center = [32.0, 16.0, 16.0]
radii = [14.0, 6.0, 6.0]
rotation_deg = [0.0, 0.0, 30.0]

[[keepin]]
shape = "box"
min = [58.0, 12.0, 12.0]
max = [64.0, 20.0, 20.0]

[[supports]]
region = { shape = "ellipsoid", center = [0.0, 16.0, 16.0], radii = [1.0, 18.0, 18.0] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "ellipsoid", center = [61.0, 16.0, 16.0], radii = [3.0, 4.0, 4.0] }
vector = [0.0, 0.0, -50.0]
"#;

/// Every engine has to route around an ellipsoid keepout and export a
/// watertight part, and none may leave a scrap of material inside it.
///
/// The solid engine runs it too, because what is asserted here is engine
/// agnostic: a carved cell is void for every engine, and the part that comes out
/// of one is watertight and spans the domain whoever built it. What is *not*
/// asserted of it is anything about optimization - it fills the design cells the
/// other two spend.
///
/// The keepout is turned, so a run that read its bounding box instead of its
/// field would carve a visibly different volume - and one that read the field
/// on the wrong axis would carve the wrong one.
#[test]
fn an_ellipsoid_keepout_runs_on_every_engine_and_is_left_empty() {
    use growforge::geometry::Shape;

    let directory = std::env::temp_dir();
    let keepout = Shape::Ellipsoid {
        center: [32.0, 16.0, 16.0],
        radii: [14.0, 6.0, 6.0],
        rotation_deg: [0.0, 0.0, 30.0],
    };
    for engine in ["simp", "growth", constants::SOLID_ENGINE] {
        let text = for_engine(engine, ELLIPSOID_KEEPOUT);
        let config = Config::parse(&text).unwrap_or_else(|e| panic!("{engine}: {e:#}"));
        let mut problem = Problem::build(&config, &directory)
            .unwrap_or_else(|e| panic!("{engine} failed to build: {e:#}"));
        problem.output.stl_path = directory.join(format!("growforge_integration_{engine}.stl"));
        assert_eq!(problem.engine, engine);
        assert!(
            problem.warnings.is_empty(),
            "{engine}: {:?}",
            problem.warnings
        );

        // The support and the load both select through an ellipsoid region, and
        // both have to find nodes with material on them or the run is refused.
        assert!(problem.supports[0].node_count > 0);
        assert!(problem.load_cases[0].loads[0].node_count > 0);

        let outcome = optimize_and_export(&problem, &SilentReporter)
            .unwrap_or_else(|e| panic!("{engine} failed to run: {e:#}"));

        // Not one cell whose centre lies inside the keepout carries material,
        // and there are plenty of such cells to be wrong about.
        let grid = &problem.grid;
        let bounds = keepout.bounds();
        let (mut carved, mut spared) = (0, 0);
        for e in 0..grid.n_cells() {
            let (i, j, k) = grid.cell_ijk(e);
            let centre = grid.cell_center(i, j, k);
            if keepout.contains(centre) {
                carved += 1;
                assert_eq!(
                    grid.cells[e],
                    CellKind::Void,
                    "{engine}: the cell at {centre:?} is inside the keepout and was not carved"
                );
                assert_eq!(
                    outcome.field.densities[e], 0.0,
                    "{engine}: the cell at {centre:?} is inside the keepout and holds material"
                );
            } else if (0..3).all(|d| centre[d] > bounds.min[d] && centre[d] < bounds.max[d]) {
                // Inside the box around the keepout but outside the keepout
                // itself: a run that carved the bounding box would take these,
                // and they are design cells like any other.
                spared += 1;
                assert_eq!(
                    grid.cells[e],
                    CellKind::Design,
                    "{engine}: the cell at {centre:?} is outside the keepout and was carved"
                );
            }
        }
        assert!(carved > 100, "{engine}: only {carved} cells were carved");
        assert!(
            spared > carved / 4,
            "{engine}: only {spared} cells separate the shape from its box, so this fixture \
             cannot tell them apart"
        );

        // The part that came out is watertight, and the file holds it.
        mesh::validate::validate(&outcome.mesh)
            .unwrap_or_else(|e| panic!("{engine} exported a surface that is not closed: {e:#}"));
        assert!(outcome.stats.triangles > 0);
        assert!(outcome.stats.volume_mm3 > 0.0);
        let (_, triangles) = mesh::stl::read(&outcome.stl_path).expect("read back the STL");
        assert_eq!(triangles.len(), outcome.stats.triangles);
        // It spans the gap: the load end and the support end are both in it.
        assert!(
            outcome.stats.bounds.min[0] < 4.0 && outcome.stats.bounds.max[0] > 60.0,
            "{engine}: the structure does not reach both ends: {:?}",
            outcome.stats.bounds
        );
        std::fs::remove_file(&outcome.stl_path).ok();
    }
}

/// The simp run of that fixture is anchored by its ellipsoid support region:
/// the analysis is well posed, and the optimization improves on where it
/// started rather than sitting on a singular solve.
#[test]
fn an_ellipsoid_support_region_anchors_a_simp_run() {
    let directory = std::env::temp_dir();
    let config = Config::parse(ELLIPSOID_KEEPOUT).expect("parse");
    let mut problem = Problem::build(&config, &directory).expect("build");
    problem.output.stl_path = directory.join("growforge_integration_ellipsoid_anchor.stl");
    assert_eq!(problem.engine, "simp");

    // The rounded patch the region selects: every node at x = 0 within 18 mm of
    // the middle of that face, which is what holds the structure up.
    let held = problem.supports[0].node_count;
    assert!(held > 40, "the ellipsoid region held only {held} nodes");
    for &n in &problem.supports[0].nodes {
        let (i, j, k) = problem.grid.node_ijk(n);
        let p = problem.grid.node_pos(i, j, k);
        assert!(
            p[0].abs() < 1e-9,
            "a held node is off the support face: {p:?}"
        );
        let offset = (p[1] - 16.0) * (p[1] - 16.0) + (p[2] - 16.0) * (p[2] - 16.0);
        assert!(offset <= 18.0 * 18.0 + 1e-9, "{p:?} is outside the region");
    }

    let outcome = optimize_and_export(&problem, &SilentReporter).expect("run");
    assert!(
        outcome.field.compliance.is_finite() && outcome.field.compliance > 0.0,
        "compliance {} says the solve was not held",
        outcome.field.compliance
    );
    assert!(
        outcome.field.compliance < outcome.field.initial_compliance,
        "compliance {} did not improve on the first iteration value {}",
        outcome.field.compliance,
        outcome.field.initial_compliance
    );
    assert!(
        (outcome.field.volume_fraction - 0.3).abs() < 0.02 * 0.3,
        "final volume fraction {} missed the 0.3 target",
        outcome.field.volume_fraction
    );
    mesh::validate::validate(&outcome.mesh).expect("watertight surface");
    std::fs::remove_file(&outcome.stl_path).ok();
}

/// The same cantilever with a **bent tube** through the middle of it, held by a
/// tube support region and loaded through one: the shape in every position that
/// has one, curved, on a problem small enough to run.
const TUBE_KEEPOUT: &str = r#"
[project]
name = "integration_tube"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

# The reproducible backend, for the reason the ellipsoid fixture gives.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.3
min_feature_mm = 8.0
max_iterations = 15

[output]
stl_path = "growforge_integration_tube.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [64.0, 32.0, 32.0]

# The part this exists for: a keepout that is a curve rather than a straight
# run, so a bounding box, a chord or a cylinder would all carve something else.
[[keepout]]
shape = "tube"
p1 = [20.0, 16.0, 4.0]
p2 = [44.0, 16.0, 4.0]
bend = [32.0, 16.0, 20.0]
radius = 5.0

[[keepin]]
shape = "box"
min = [58.0, 12.0, 12.0]
max = [64.0, 20.0, 20.0]

[[supports]]
region = { shape = "tube", p1 = [0.0, 4.0, 16.0], p2 = [0.0, 28.0, 16.0], radius = 3.0 }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "tube", p1 = [61.0, 14.0, 16.0], p2 = [61.0, 18.0, 16.0], bend = [61.0, 16.0, 18.0], radius = 3.0 }
vector = [0.0, 0.0, -50.0]
"#;

/// Every engine has to route around a bent tube keepout and export a
/// watertight part, and none may leave a scrap of material inside it.
///
/// The keepout curves, so a run that read its bounding box - or mistook it for
/// the cylinder between its two ends - would carve a visibly different volume,
/// and this fixture has enough cells either side of the difference to say so.
#[test]
fn a_tube_keepout_runs_on_every_engine_and_is_left_empty() {
    use growforge::geometry::Shape;

    let directory = std::env::temp_dir();
    let keepout = Shape::Tube {
        p1: [20.0, 16.0, 4.0],
        p2: [44.0, 16.0, 4.0],
        bend: Some([32.0, 16.0, 20.0]),
        radius: 5.0,
    };
    for engine in ["simp", "growth", constants::SOLID_ENGINE] {
        let text = for_engine(engine, TUBE_KEEPOUT);
        let config = Config::parse(&text).unwrap_or_else(|e| panic!("{engine}: {e:#}"));
        let mut problem = Problem::build(&config, &directory)
            .unwrap_or_else(|e| panic!("{engine} failed to build: {e:#}"));
        problem.output.stl_path =
            directory.join(format!("growforge_integration_tube_{engine}.stl"));
        assert_eq!(problem.engine, engine);
        assert!(
            problem.warnings.is_empty(),
            "{engine}: {:?}",
            problem.warnings
        );

        // The support and the load both select through a tube region.
        assert!(problem.supports[0].node_count > 0);
        assert!(problem.load_cases[0].loads[0].node_count > 0);

        let outcome = optimize_and_export(&problem, &SilentReporter)
            .unwrap_or_else(|e| panic!("{engine} failed to run: {e:#}"));

        // Not one cell whose centre lies inside the tube carries material, and
        // the cells the curve leaves *behind* - inside its box, outside the
        // tube - are design cells like any other.
        let grid = &problem.grid;
        let bounds = keepout.bounds();
        let (mut carved, mut spared) = (0, 0);
        for e in 0..grid.n_cells() {
            let (i, j, k) = grid.cell_ijk(e);
            let centre = grid.cell_center(i, j, k);
            if keepout.contains(centre) {
                carved += 1;
                assert_eq!(
                    grid.cells[e],
                    CellKind::Void,
                    "{engine}: the cell at {centre:?} is inside the keepout and was not carved"
                );
                assert_eq!(
                    outcome.field.densities[e], 0.0,
                    "{engine}: the cell at {centre:?} is inside the keepout and holds material"
                );
            } else if (0..3).all(|d| centre[d] > bounds.min[d] && centre[d] < bounds.max[d]) {
                spared += 1;
                assert_eq!(
                    grid.cells[e],
                    CellKind::Design,
                    "{engine}: the cell at {centre:?} is outside the keepout and was carved"
                );
            }
        }
        assert!(carved > 100, "{engine}: only {carved} cells were carved");
        assert!(
            spared > carved / 4,
            "{engine}: only {spared} cells separate the curve from its box, so this fixture \
             cannot tell them apart"
        );

        mesh::validate::validate(&outcome.mesh)
            .unwrap_or_else(|e| panic!("{engine} exported a surface that is not closed: {e:#}"));
        assert!(outcome.stats.triangles > 0);
        assert!(outcome.stats.volume_mm3 > 0.0);
        let (_, triangles) = mesh::stl::read(&outcome.stl_path).expect("read back the STL");
        assert_eq!(triangles.len(), outcome.stats.triangles);
        assert!(
            outcome.stats.bounds.min[0] < 4.0 && outcome.stats.bounds.max[0] > 60.0,
            "{engine}: the structure does not reach both ends: {:?}",
            outcome.stats.bounds
        );
        std::fs::remove_file(&outcome.stl_path).ok();
    }
}

/// The simp run of that fixture is anchored by its tube support region: the
/// analysis is well posed, and the optimization improves on where it started
/// rather than sitting on a singular solve.
#[test]
fn a_tube_support_region_anchors_a_simp_run() {
    let directory = std::env::temp_dir();
    let config = Config::parse(TUBE_KEEPOUT).expect("parse");
    let mut problem = Problem::build(&config, &directory).expect("build");
    problem.output.stl_path = directory.join("growforge_integration_tube_anchor.stl");
    assert_eq!(problem.engine, "simp");

    // The capsule the region selects: every held node is within 3 mm of the
    // segment it runs along - which, unlike a slab of an ellipsoid, is a
    // rounded volume that reaches into the part as far as it reaches across
    // it, and the run has to be held by exactly that.
    let held = problem.supports[0].node_count;
    assert!(held > 10, "the tube region held only {held} nodes");
    let mut on_the_face = 0;
    for &n in &problem.supports[0].nodes {
        let (i, j, k) = problem.grid.node_ijk(n);
        let p = problem.grid.node_pos(i, j, k);
        let along = p[1].clamp(4.0, 28.0);
        let offset = p[0] * p[0] + (p[1] - along) * (p[1] - along) + (p[2] - 16.0) * (p[2] - 16.0);
        assert!(
            offset <= 9.0 + 1e-9,
            "{p:?} is {} mm off the region's own centre line",
            offset.sqrt()
        );
        if p[0].abs() < 1e-9 {
            on_the_face += 1;
        }
    }
    assert!(
        on_the_face > 5,
        "only {on_the_face} of the held nodes are on the supported face"
    );

    let outcome = optimize_and_export(&problem, &SilentReporter).expect("run");
    assert!(
        outcome.field.compliance.is_finite() && outcome.field.compliance > 0.0,
        "compliance {} says the solve was not held",
        outcome.field.compliance
    );
    assert!(
        outcome.field.compliance < outcome.field.initial_compliance,
        "compliance {} did not improve on the first iteration value {}",
        outcome.field.compliance,
        outcome.field.initial_compliance
    );
    mesh::validate::validate(&outcome.mesh).expect("watertight surface");
    std::fs::remove_file(&outcome.stl_path).ok();
}

/// The same cantilever with a **cone** through the middle of it, held by a cone
/// support region and loaded through one: the shape in every position that has
/// one, tapering to a point, on a problem small enough to run.
const CONE_KEEPOUT: &str = r#"
[project]
name = "integration_cone"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

# The reproducible backend, for the reason the ellipsoid fixture gives.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.3
min_feature_mm = 8.0
max_iterations = 15

[output]
stl_path = "growforge_integration_cone.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [64.0, 32.0, 32.0]

# The part this exists for: a keepout that narrows to a point, so a bounding
# box - or the cylinder between its two caps - would carve a visibly different
# volume.
[[keepout]]
shape = "cone"
p1 = [20.0, 16.0, 16.0]
p2 = [44.0, 16.0, 16.0]
radius1 = 8.0
radius2 = 0.0

[[keepin]]
shape = "box"
min = [58.0, 12.0, 12.0]
max = [64.0, 20.0, 20.0]

[[supports]]
region = { shape = "cone", p1 = [0.0, 16.0, 16.0], p2 = [4.0, 16.0, 16.0], radius1 = 8.0, radius2 = 4.0 }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "cone", p1 = [58.0, 16.0, 16.0], p2 = [63.0, 16.0, 16.0], radius1 = 5.0, radius2 = 2.0 }
vector = [0.0, 0.0, -50.0]
"#;

/// The same cantilever with a **triangular prism** through the middle of it,
/// held by a prism support region and loaded through one.
const TRIANGLE_KEEPOUT: &str = r#"
[project]
name = "integration_triangle"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "pla"

[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.3
min_feature_mm = 8.0
max_iterations = 15

[output]
stl_path = "growforge_integration_triangle.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [64.0, 32.0, 32.0]

# The part this exists for: a keepout with sloping sides, so a bounding box
# would carve away the two corners the prism leaves standing.
[[keepout]]
shape = "triangle"
a = [20.0, 16.0, 4.0]
b = [44.0, 16.0, 4.0]
c = [32.0, 16.0, 26.0]
thickness = 8.0

[[keepin]]
shape = "box"
min = [58.0, 12.0, 12.0]
max = [64.0, 20.0, 20.0]

[[supports]]
region = { shape = "triangle", a = [0.0, 4.0, 16.0], b = [0.0, 28.0, 16.0], c = [0.0, 16.0, 28.0], thickness = 4.0 }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "triangle", a = [61.0, 12.0, 16.0], b = [61.0, 20.0, 16.0], c = [61.0, 16.0, 22.0], thickness = 3.0 }
vector = [0.0, 0.0, -50.0]
"#;

/// Every engine has to route around each of the two new shapes and export a
/// watertight part, and none may leave a scrap of material inside one.
///
/// The keepouts taper and slope, so a run that read a bounding box - or, for
/// the cone, mistook it for the cylinder between its two caps - would carve a
/// visibly different volume, and both fixtures have enough cells either side of
/// the difference to say so. The support and the load select through a region
/// of the same kind, so every position a shape can appear in is exercised on a
/// problem that really runs.
#[test]
fn a_cone_and_a_triangle_keepout_run_on_every_engine_and_are_left_empty() {
    use growforge::geometry::Shape;

    let directory = std::env::temp_dir();
    let cases: [(&str, &str, Shape); 2] = [
        (
            "cone",
            CONE_KEEPOUT,
            Shape::Cone {
                p1: [20.0, 16.0, 16.0],
                p2: [44.0, 16.0, 16.0],
                radius1: 8.0,
                radius2: 0.0,
            },
        ),
        (
            "triangle",
            TRIANGLE_KEEPOUT,
            Shape::Triangle {
                a: [20.0, 16.0, 4.0],
                b: [44.0, 16.0, 4.0],
                c: [32.0, 16.0, 26.0],
                thickness: 8.0,
            },
        ),
    ];
    for (name, fixture, keepout) in cases {
        for engine in ["simp", "growth", constants::SOLID_ENGINE] {
            let what = format!("{name}/{engine}");
            let text = for_engine(engine, fixture);
            let config = Config::parse(&text).unwrap_or_else(|e| panic!("{what}: {e:#}"));
            let mut problem = Problem::build(&config, &directory)
                .unwrap_or_else(|e| panic!("{what} failed to build: {e:#}"));
            problem.output.stl_path =
                directory.join(format!("growforge_integration_{name}_{engine}.stl"));
            assert_eq!(problem.engine, engine);
            assert!(
                problem.warnings.is_empty(),
                "{what}: {:?}",
                problem.warnings
            );

            // The support and the load both select through a region of the
            // same kind as the keepout.
            assert!(problem.supports[0].node_count > 0, "{what}: nothing held");
            assert!(problem.load_cases[0].loads[0].node_count > 0, "{what}");

            let outcome = optimize_and_export(&problem, &SilentReporter)
                .unwrap_or_else(|e| panic!("{what} failed to run: {e:#}"));

            // Not one cell whose centre lies inside the shape carries
            // material, and the cells its slope leaves *behind* - inside its
            // box, outside the shape - are design cells like any other.
            let grid = &problem.grid;
            let bounds = keepout.bounds();
            let (mut carved, mut spared) = (0, 0);
            for e in 0..grid.n_cells() {
                let (i, j, k) = grid.cell_ijk(e);
                let centre = grid.cell_center(i, j, k);
                if keepout.contains(centre) {
                    carved += 1;
                    assert_eq!(
                        grid.cells[e],
                        CellKind::Void,
                        "{what}: the cell at {centre:?} is inside the keepout and was not carved"
                    );
                    assert_eq!(
                        outcome.field.densities[e], 0.0,
                        "{what}: the cell at {centre:?} is inside the keepout and holds material"
                    );
                } else if (0..3).all(|d| centre[d] > bounds.min[d] && centre[d] < bounds.max[d]) {
                    spared += 1;
                    assert_eq!(
                        grid.cells[e],
                        CellKind::Design,
                        "{what}: the cell at {centre:?} is outside the keepout and was carved"
                    );
                }
            }
            assert!(carved > 100, "{what}: only {carved} cells were carved");
            assert!(
                spared > carved / 4,
                "{what}: only {spared} cells separate the shape from its box, so this fixture \
                 cannot tell them apart"
            );

            mesh::validate::validate(&outcome.mesh)
                .unwrap_or_else(|e| panic!("{what} exported a surface that is not closed: {e:#}"));
            assert!(outcome.stats.triangles > 0);
            assert!(outcome.stats.volume_mm3 > 0.0);
            let (_, triangles) = mesh::stl::read(&outcome.stl_path).expect("read back the STL");
            assert_eq!(triangles.len(), outcome.stats.triangles);
            assert!(
                outcome.stats.bounds.min[0] < 4.0 && outcome.stats.bounds.max[0] > 60.0,
                "{what}: the structure does not reach both ends: {:?}",
                outcome.stats.bounds
            );
            std::fs::remove_file(&outcome.stl_path).ok();
        }
    }
}

/// The simp runs of those two fixtures are anchored by their own regions: the
/// analyses are well posed, and each optimization improves on where it started
/// rather than sitting on a singular solve.
#[test]
fn a_cone_and_a_triangle_support_region_each_anchor_a_simp_run() {
    let directory = std::env::temp_dir();
    for (name, fixture) in [("cone", CONE_KEEPOUT), ("triangle", TRIANGLE_KEEPOUT)] {
        let config = Config::parse(fixture).expect("parse");
        let mut problem = Problem::build(&config, &directory).expect("build");
        problem.output.stl_path =
            directory.join(format!("growforge_integration_{name}_anchor.stl"));
        assert_eq!(problem.engine, "simp");

        // Every held node is one the region really contains, and enough of
        // them are on the supported face to hold the part rather than a
        // corner of it.
        let region = match name {
            "cone" => growforge::geometry::Shape::Cone {
                p1: [0.0, 16.0, 16.0],
                p2: [4.0, 16.0, 16.0],
                radius1: 8.0,
                radius2: 4.0,
            },
            _ => growforge::geometry::Shape::Triangle {
                a: [0.0, 4.0, 16.0],
                b: [0.0, 28.0, 16.0],
                c: [0.0, 16.0, 28.0],
                thickness: 4.0,
            },
        };
        let held = problem.supports[0].node_count;
        assert!(held > 10, "{name}: the region held only {held} nodes");
        let mut on_the_face = 0;
        for &n in &problem.supports[0].nodes {
            let (i, j, k) = problem.grid.node_ijk(n);
            let p = problem.grid.node_pos(i, j, k);
            assert!(region.contains(p), "{name}: {p:?} is outside the region");
            if p[0].abs() < 1e-9 {
                on_the_face += 1;
            }
        }
        assert!(
            on_the_face > 5,
            "{name}: only {on_the_face} of the held nodes are on the supported face"
        );

        let outcome = optimize_and_export(&problem, &SilentReporter).expect("run");
        assert!(
            outcome.field.compliance.is_finite() && outcome.field.compliance > 0.0,
            "{name}: compliance {} says the solve was not held",
            outcome.field.compliance
        );
        assert!(
            outcome.field.compliance < outcome.field.initial_compliance,
            "{name}: compliance {} did not improve on the first iteration value {}",
            outcome.field.compliance,
            outcome.field.initial_compliance
        );
        mesh::validate::validate(&outcome.mesh).expect("watertight surface");
        std::fs::remove_file(&outcome.stl_path).ok();
    }
}

#[test]
fn shipped_examples_build_without_warnings() {
    for name in [
        "cantilever.toml",
        "mbb_bridge.toml",
        "shelf_bracket.toml",
        "growth_canopy.toml",
        "growth_canopy_symmetric.toml",
    ] {
        let problem = load_problem(&example_path(name))
            .unwrap_or_else(|e| panic!("{name} failed to build: {e:#}"));
        assert!(
            problem.warnings.is_empty(),
            "{name} produced warnings: {:?}",
            problem.warnings
        );
        assert!(problem.counts.design > 0);
        assert!(!problem.load_cases.is_empty());
        for support in &problem.supports {
            assert!(support.node_count > 0);
        }
        for case in &problem.load_cases {
            for load in &case.loads {
                assert!(load.node_count > 0);
            }
        }
    }
}

/// A grown canopy with `[growth] prune` switched off, so the branches that
/// ended on nothing are still in the field: the prongs that reach to nothing,
/// deliberately left in for the trim pass to find.
///
/// Growth rather than SIMP because the shape has to be *there* to be removed. A
/// density optimization erases material that carries nothing all by itself -
/// that is what a compliance objective does - so the only unloaded geometry a
/// SIMP run leaves is geometry something forced it to keep, and everything that
/// can force it is either a declared region or a seed the trim pass protects.
/// Growth builds a skeleton instead, and with its own pruning off it builds one
/// with stubs on it.
const UNPRUNED_CANOPY: &str = r#"
[project]
name = "integration_trim"
engine = "growth"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "petg"

[optimization]
mass_fraction = 0.15
min_feature_mm = 12.0

[growth]
seed = 20260730
attractor_per_cm3 = 6.0
max_steps = 300
# The point of the fixture: branches that ended on nothing are kept.
prune = false

[output]
stl_path = "growforge_integration_trim.stl"
iso_level = 0.5
smoothing_iterations = 4
trim = "stress"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [120.0, 60.0, 80.0]

[[keepin]]
shape = "box"
min = [0.0, 0.0, 72.0]
max = [120.0, 60.0, 80.0]

[[supports]]
region = { shape = "box", min = [-1.0, -1.0, -1.0], max = [26.0, 61.0, 1.0] }

[[supports]]
region = { shape = "box", min = [94.0, -1.0, -1.0], max = [121.0, 61.0, 1.0] }

[[loadcases]]
name = "top"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [-1.0, -1.0, 78.0], max = [121.0, 61.0, 81.0] }
vector = [0.0, 0.0, -400.0]
"#;

/// The trim pass end to end: the same grown design exported twice, once as it
/// was grown and once with `[output] trim = "stress"`, and the second file is
/// the first one with the stubs gone.
#[test]
fn trimming_removes_the_branches_that_ended_on_nothing_from_the_exported_surface() {
    use growforge::config::TrimPolicy;
    use growforge::trim::TrimEnding;

    let directory = std::env::temp_dir();
    let config = Config::parse(UNPRUNED_CANOPY).expect("parse");

    let mut untrimmed = Problem::build(&config, &directory).expect("build");
    untrimmed.output.trim = TrimPolicy::Off;
    untrimmed.output.stl_path = directory.join("growforge_integration_untrimmed.stl");
    let before = optimize_and_export(&untrimmed, &SilentReporter).expect("the untrimmed run");
    assert!(before.trim.is_none(), "the pass ran with trim = \"off\"");
    assert_eq!(
        before.field.growth.expect("a growth summary").pruned_nodes,
        0,
        "the fixture pruned its own stubs, so there is nothing here to trim"
    );

    let mut problem = Problem::build(&config, &directory).expect("build");
    problem.output.stl_path = directory.join("growforge_integration_trim.stl");
    assert_eq!(problem.output.trim, TrimPolicy::Stress);
    let after = optimize_and_export(&problem, &SilentReporter).expect("the trimmed run");

    // The same design went in: growth is deterministic, so the two runs differ
    // in the pass and in nothing else.
    assert_eq!(
        before.field.iterations, after.field.iterations,
        "the two runs did not grow the same design"
    );

    let report = after.trim.as_ref().expect("a trim report");
    assert_eq!(report.ending, TrimEnding::Removed, "{report:?}");
    assert!(report.cells > 0 && report.volume_mm3 > 0.0);
    assert!(
        report.removed_fraction() > 0.0 && report.removed_fraction() < 0.5,
        "the pass removed {:.1}% of the part, which is not a trim",
        100.0 * report.removed_fraction()
    );
    assert!(
        report.notes()[0].starts_with("removed"),
        "{:?}",
        report.notes()
    );

    // The surface really lost the stubs: less material, fewer triangles, and a
    // different file. Nothing was gained anywhere.
    assert!(
        after.stats.volume_mm3 < before.stats.volume_mm3,
        "the trimmed surface encloses {:.1} mm3 against the untrimmed {:.1}",
        after.stats.volume_mm3,
        before.stats.volume_mm3
    );
    assert_ne!(
        std::fs::read(&before.stl_path).expect("the untrimmed STL"),
        std::fs::read(&after.stl_path).expect("the trimmed STL")
    );

    // And it is still a part: watertight, one piece, held by everything the
    // configuration declared, and stress analysed on the geometry that shipped.
    mesh::validate::validate(&after.mesh).expect("watertight surface");
    assert_eq!(
        after.solids.len(),
        1,
        "the trim broke the field into {} pieces",
        after.solids.len()
    );
    assert_eq!(after.islands.bodies.len(), 1);
    assert!(
        after.islands.unserved.is_empty(),
        "{:?}",
        after.islands.unserved
    );
    let table = after.stress.report().expect("a stress table");
    let evaluated = (0..problem.grid.n_cells())
        .filter(|&e| {
            growforge::engine::cell_density(&problem.grid, &after.field.densities, e)
                >= constants::STRESS_DENSITY_THRESHOLD
        })
        .count();
    assert_eq!(
        table.evaluated_cells, evaluated,
        "the stress table describes a different field from the one that was written"
    );

    std::fs::remove_file(&before.stl_path).ok();
    std::fs::remove_file(&after.stl_path).ok();
}

/// A grown structure whose members are deliberately thinner than the floor the
/// reinforcement pass will be asked for, with a keepout cylinder through the
/// middle of the space they grow in.
///
/// `min_feature_mm` sizes the struts at 8 mm, two voxels across, and the run
/// below asks for 16 mm: every member of the design is under the floor, and the
/// bore is there for the pass to be pressed against and refuse to cross.
const THIN_MEMBERS: &str = r#"
[project]
name = "integration_reinforce"
engine = "growth"

[resolution]
voxel_size_mm = 4.0

[material]
preset = "petg"

[optimization]
mass_fraction = 0.12
min_feature_mm = 8.0

[growth]
seed = 20260804
attractor_per_cm3 = 5.0
max_steps = 250

[output]
stl_path = "growforge_integration_reinforce.stl"
iso_level = 0.5
smoothing_iterations = 4
reinforce = "min_thickness"
reinforce_thickness_mm = 16.0

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [120.0, 60.0, 80.0]

[[keepout]]
shape = "cylinder"
p1 = [60.0, -1.0, 40.0]
p2 = [60.0, 61.0, 40.0]
radius = 10.0

[[keepin]]
shape = "box"
min = [0.0, 0.0, 72.0]
max = [120.0, 60.0, 80.0]

[[supports]]
region = { shape = "box", min = [-1.0, -1.0, -1.0], max = [26.0, 61.0, 1.0] }

[[supports]]
region = { shape = "box", min = [94.0, -1.0, -1.0], max = [121.0, 61.0, 1.0] }

[[loadcases]]
name = "top"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [-1.0, -1.0, 78.0], max = [121.0, 61.0, 81.0] }
vector = [0.0, 0.0, -400.0]
"#;

/// The reinforcement pass end to end: the same grown design exported twice,
/// once as it was grown and once held to a 16 mm floor, and the second file is
/// the first one with its arms thickened.
///
/// The thickness assertions are made with the **exact** sphere-fitting reading -
/// twice the largest inscribed ball that covers a cell - computed here from the
/// distance transform alone. The pass measures at the spine, where the two agree
/// because the ball inscribed at a local maximum touches the boundary on both
/// sides; that they agree is what this proves rather than assumes. It is checked
/// where the pass promised something: at the places that were thin before the
/// pass and whose ball fitted inside the design space. A member pressed against
/// the bore or the edge of the domain is the count the report warns with, and a
/// free tip or a convex corner is not a thickness at all.
#[test]
fn reinforcement_thickens_the_members_that_were_below_the_printable_floor() {
    use growforge::config::ReinforcePolicy;
    use growforge::grid::Grid;
    use growforge::reinforce::squared_distances;

    let floor = 16.0;
    let directory = std::env::temp_dir();
    let config = Config::parse(THIN_MEMBERS).expect("parse");

    let mut plain = Problem::build(&config, &directory).expect("build");
    plain.output.reinforce = ReinforcePolicy::Off;
    plain.output.stl_path = directory.join("growforge_integration_thin.stl");
    let before = optimize_and_export(&plain, &SilentReporter).expect("the unreinforced run");
    assert!(
        before.reinforce.is_none(),
        "the pass ran with reinforce = \"off\""
    );

    let mut problem = Problem::build(&config, &directory).expect("build");
    problem.output.stl_path = directory.join("growforge_integration_reinforce.stl");
    assert_eq!(problem.output.reinforce, ReinforcePolicy::MinThickness);
    assert_eq!(problem.output.reinforce_thickness_mm, floor);
    let after = optimize_and_export(&problem, &SilentReporter).expect("the reinforced run");

    // The same design went in: growth is deterministic, so the two runs differ
    // in the pass and in nothing else.
    assert_eq!(
        before.field.iterations, after.field.iterations,
        "the two runs did not grow the same design"
    );

    let report = after.reinforce.as_ref().expect("a reinforce report");
    assert!(
        report.cells_raised > 0 && report.volume_added_mm3 > 0.0,
        "{report:?}"
    );
    assert_eq!(report.thickness_mm, floor);
    let notes = report.notes();
    assert!(notes[0].starts_with("reinforced"), "{notes:?}");
    // The keepin slab lies against the top of the domain and cannot be
    // thickened upwards, so this fixture is also the warning's own case.
    assert!(report.still_thin > 0, "{report:?}");
    assert!(
        notes.iter().any(|note| note.starts_with("warning:")),
        "a place that could not be reinforced was not warned about: {notes:?}"
    );

    // The surface really gained the material: more of it, and a different file.
    assert!(
        after.stats.volume_mm3 > before.stats.volume_mm3,
        "the reinforced surface encloses {:.1} mm3 against the plain {:.1}",
        after.stats.volume_mm3,
        before.stats.volume_mm3
    );
    assert_ne!(
        std::fs::read(&before.stl_path).expect("the plain STL"),
        std::fs::read(&after.stl_path).expect("the reinforced STL")
    );
    // And both are still parts.
    mesh::validate::validate(&before.mesh).expect("watertight plain surface");
    mesh::validate::validate(&after.mesh).expect("watertight reinforced surface");

    // Nothing but design cells was written: the bore, the space outside the
    // domain and the forced solid slab come through the pass unchanged.
    let grid = &problem.grid;
    for cell in 0..grid.n_cells() {
        if grid.cells[cell] != CellKind::Design {
            assert_eq!(
                after.field.densities[cell], before.field.densities[cell],
                "a {:?} cell was written",
                grid.cells[cell]
            );
        }
    }

    // The exact reading, from the distance transform alone.
    let iso = problem.output.iso_level;
    let part = |densities: &[f64]| -> Vec<bool> {
        (0..grid.n_cells())
            .map(|e| growforge::engine::cell_density(grid, densities, e) >= iso)
            .collect()
    };
    let surface_mm =
        |squared: f64| (squared.sqrt() - constants::REINFORCE_SURFACE_OFFSET_VOXELS) * grid.h;
    // Twice the largest inscribed ball covering `cell`, over the whole part.
    let exact_mm = |part: &[bool], squared: &[f64], cell: usize| {
        let (i, j, k) = grid.cell_ijk(cell);
        let mut best: f64 = 0.0;
        for other in 0..grid.n_cells() {
            if !part[other] {
                continue;
            }
            let radius = surface_mm(squared[other]);
            if 2.0 * radius <= best {
                continue;
            }
            let (ci, cj, ck) = grid.cell_ijk(other);
            let (di, dj, dk) = (
                (i as f64 - ci as f64) * grid.h,
                (j as f64 - cj as f64) * grid.h,
                (k as f64 - ck as f64) * grid.h,
            );
            let distance = (di * di + dj * dj + dk * dk).sqrt();
            if distance <= radius + constants::REINFORCE_SURFACE_OFFSET_VOXELS * grid.h + 1e-9 {
                best = 2.0 * radius;
            }
        }
        best
    };
    // Whether the ball the fill would have taken fitted: every cell it covers
    // inside the grid and clear of the void. Where it did, the floor is a
    // promise; where it did not, the report's warning is.
    let unclipped = |grid: &Grid, cell: usize| {
        let radius = 0.5 * floor + constants::REINFORCE_SURFACE_OFFSET_VOXELS * grid.h;
        let reach = (radius / grid.h).ceil() as i64;
        let (ci, cj, ck) = grid.cell_ijk(cell);
        for dk in -reach..=reach {
            for dj in -reach..=reach {
                for di in -reach..=reach {
                    let squared = ((di * di + dj * dj + dk * dk) as f64) * grid.h * grid.h;
                    if squared > radius * radius {
                        continue;
                    }
                    let (i, j, k) = (ci as i64 + di, cj as i64 + dj, ck as i64 + dk);
                    if i < 0
                        || j < 0
                        || k < 0
                        || i >= grid.nx as i64
                        || j >= grid.ny as i64
                        || k >= grid.nz as i64
                    {
                        return false;
                    }
                    if grid.cells[grid.cell_index(i as usize, j as usize, k as usize)]
                        == CellKind::Void
                    {
                        return false;
                    }
                }
            }
        }
        true
    };

    let plain_part = part(&before.field.densities);
    let plain_squared = squared_distances(grid, &plain_part);
    let final_part = part(&after.field.densities);
    let final_squared = squared_distances(grid, &final_part);

    // Every place that was thin before the pass and had the room to be fixed
    // is at the floor after it - measured by a definition the pass never used.
    let mut checked = 0;
    for cell in 0..grid.n_cells() {
        if !plain_part[cell] || 2.0 * surface_mm(plain_squared[cell]) >= floor {
            continue;
        }
        let (i, j, k) = grid.cell_ijk(cell);
        let is_spine = (-1i64..=1).all(|dk| {
            (-1i64..=1).all(|dj| {
                (-1i64..=1).all(|di| {
                    let (ni, nj, nk) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                    if ni < 0
                        || nj < 0
                        || nk < 0
                        || ni >= grid.nx as i64
                        || nj >= grid.ny as i64
                        || nk >= grid.nz as i64
                    {
                        return true;
                    }
                    plain_squared[grid.cell_index(ni as usize, nj as usize, nk as usize)]
                        <= plain_squared[cell]
                })
            })
        });
        if !is_spine || !unclipped(grid, cell) {
            continue;
        }
        let thickness = exact_mm(&final_part, &final_squared, cell);
        assert!(
            thickness >= floor - 1e-9,
            "the member at cell {cell} came out {thickness:.3} mm thick against a {floor} mm floor"
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "the fixture has no thin member with room to be thickened, so it proves nothing"
    );

    std::fs::remove_file(&before.stl_path).ok();
    std::fs::remove_file(&after.stl_path).ok();
}

/// A solid block with a pin bore straight through it, at the user's own
/// numbers: a 2.75 mm radius bore meshed on a 1.5 mm grid.
///
/// The keepin box is what makes it a bore rather than an optimization result -
/// there is material right up against the cylinder whatever the engine does
/// with the rest of the block, so the exported surface has a bore wall to get
/// right or wrong.
const BORE: &str = r#"
[project]
name = "integration_bore"

[resolution]
voxel_size_mm = 1.5

[material]
preset = "pla"

# The reproducible backend: the two runs below are compared against each other,
# and the default backend is the machine's compute device.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.6
min_feature_mm = 4.5
max_iterations = 3

[output]
stl_path = "growforge_integration_bore.stl"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [24.0, 24.0, 12.0]

# The pin bore. Cell centres are classified, so the cells the classifier leaves
# solid reach a fraction of a voxel into it.
[[keepout]]
shape = "cylinder"
p1 = [12.0, 12.0, -1.0]
p2 = [12.0, 12.0, 13.0]
radius = 2.75

[[keepin]]
shape = "box"
min = [4.5, 4.5, 0.0]
max = [19.5, 19.5, 12.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [24.5, 24.5, 0.5] }

[[loadcases]]
name = "press"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [-0.5, -0.5, 11.5], max = [24.5, 24.5, 12.5] }
vector = [0.0, 0.0, -100.0]
"#;

/// The defect the user reported, and the pass that fixes it: the exported STL
/// clipping into a pin bore.
///
/// **One** design is optimized and then exported twice, once under each
/// `[output] boundaries` setting, so the only difference between the two files
/// is the setting. Under `"voxel"` the surface really does cut into the analytic
/// cylinder - that is the fixture earning its keep - and under the default
/// `"exact"` every exported vertex is outside the keepout and inside the domain,
/// which is a class of check no other test in this suite makes: it asks the
/// *mesh* about the *shapes*, rather than the cells about either.
///
/// Nothing here needs to guard the live preview: `scene::preview_surface` runs
/// marching cubes alone and never enters the export pipeline, so a preview is
/// unclamped by construction.
#[test]
fn a_bore_is_exported_as_the_cylinder_that_was_asked_for() {
    use growforge::geometry::Shape;

    let directory = std::env::temp_dir();
    let config = Config::parse(BORE).expect("parse");
    let bore = Shape::Cylinder {
        p1: [12.0, 12.0, -1.0],
        p2: [12.0, 12.0, 13.0],
        radius: 2.75,
    };

    let mut exact = Problem::build(&config, &directory).expect("build");
    exact.output.stl_path = directory.join("growforge_integration_bore_exact.stl");
    assert_eq!(
        exact.output.boundaries,
        BoundaryFidelity::Exact,
        "the clamp has to be what a configuration gets without asking"
    );
    let mut voxel = Problem::build(&config, &directory).expect("build");
    voxel.output.boundaries = BoundaryFidelity::Voxel;
    voxel.output.stl_path = directory.join("growforge_integration_bore_voxel.stl");

    let clamped = optimize_and_export(&exact, &SilentReporter).expect("the clamped run");
    // The same field, through the same export, with the clamp switched off.
    let raw = growforge::export(&voxel, &clamped.field.densities).expect("the unclamped export");

    // How far the worst vertex of a mesh reaches into the bore, and how far the
    // worst one reaches out of the domain. Both are positive when violated.
    let worst = |vertices: &[[f64; 3]]| {
        let mut into_bore: f64 = 0.0;
        let mut out_of_domain: f64 = 0.0;
        for vertex in vertices {
            into_bore = into_bore.max(-exact.boundaries.keepout.signed_distance(*vertex));
            out_of_domain = out_of_domain.max(exact.boundaries.domain.signed_distance(*vertex));
        }
        (into_bore, out_of_domain)
    };

    // The defect, measured: without the clamp the surface is inside the bore and
    // outside the block, both by a fraction of a voxel.
    let (raw_bore, raw_domain) = worst(&raw.mesh.vertices);
    assert!(raw.clamp.is_none(), "the pass ran under boundaries = voxel");
    assert!(
        raw_bore > 0.01,
        "the unclamped surface only reaches {raw_bore} mm into the bore, so this fixture cannot \
         show what the clamp is for"
    );
    assert!(
        raw_bore < voxel.grid.h,
        "{raw_bore} is not a sampling error"
    );
    assert!(
        raw_domain > 0.01 && raw_domain < voxel.grid.h,
        "the unclamped surface leaves the domain by {raw_domain} mm"
    );

    // And the fix: every vertex of the exported surface, not merely its bounds.
    let (clamped_bore, clamped_domain) = worst(&clamped.mesh.vertices);
    let eps = constants::BOUNDARY_CLAMP_EPS_MM;
    assert!(
        clamped_bore <= eps,
        "the clamped surface still reaches {clamped_bore} mm into the bore"
    );
    assert!(
        clamped_domain <= eps,
        "the clamped surface still leaves the domain by {clamped_domain} mm"
    );

    let report = clamped.clamp.expect("a clamp report");
    assert!(report.vertices_moved > 0, "{report:?}");
    assert_eq!(report.gave_up, 0, "{report:?}");
    assert!(
        report.max_displacement_mm <= exact.grid.h,
        "{report:?} moved a vertex further than a voxel"
    );
    let notes = report.notes(exact.is_solid(), exact.is_flushing());
    assert!(notes[0].starts_with("clamped"), "{notes:?}");

    // Both files are parts: closed, manifold, one piece, and on disk.
    for (mesh_out, islands, stats, path) in [
        (
            &clamped.mesh,
            &clamped.islands,
            clamped.stats,
            &exact.output.stl_path,
        ),
        (&raw.mesh, &raw.islands, raw.stats, &voxel.output.stl_path),
    ] {
        mesh::validate::validate(mesh_out).expect("watertight surface");
        assert_eq!(islands.bodies.len(), 1);
        assert!(islands.unserved.is_empty(), "{:?}", islands.unserved);
        let (_, triangles) = mesh::stl::read(path).expect("read back the STL");
        assert_eq!(triangles.len(), stats.triangles);
    }
    // The clamp moves the surface onto the boundaries in **both** directions,
    // which is what "exact" has to mean: out of the bore and off the outside of
    // the block where the sampling overshot, and back out onto the block's faces
    // and the bore wall where it fell short. Correcting only the overshoot would
    // leave the undershoot as dimples - the same sub-voxel scatter, on the legal
    // side, where nothing about legality can see it.
    //
    // Stated as the defect and its absence, over the vertices that rest on a
    // boundary at all: an optimizer's free surface through the middle of the
    // domain rests on nothing and is left where the smoothing put it.
    let capture = constants::BOUNDARY_CLAMP_CAPTURE_VOXELS * exact.grid.h;
    let resting = |vertices: &[[f64; 3]]| {
        let mut worst: f64 = 0.0;
        let mut counted = 0usize;
        for vertex in vertices {
            // The nearest boundary, whichever it is: this part rests on the
            // block's faces and on the bore wall, and the clamp seats a vertex
            // onto the one it is nearest to.
            let distance = exact
                .boundaries
                .domain
                .signed_distance(*vertex)
                .abs()
                .min(exact.boundaries.keepout.signed_distance(*vertex).abs());
            if distance <= capture {
                counted += 1;
                worst = worst.max(distance);
            }
        }
        (counted, worst)
    };
    let (raw_resting, raw_worst) = resting(&raw.mesh.vertices);
    let (clamped_resting, clamped_worst) = resting(&clamped.mesh.vertices);
    assert!(
        raw_resting > 0 && raw_worst > 2.0 * eps,
        "the unclamped surface has no vertex resting off a boundary ({raw_resting} within the          band, worst {raw_worst}), so this fixture cannot show what the seating is for"
    );
    assert!(
        clamped_resting > 0,
        "no clamped vertex rests on a boundary, so the claim below is vacuous"
    );
    assert!(
        clamped_worst <= 2.0 * eps,
        "the clamped surface still sits {clamped_worst} mm off the boundary it rests on"
    );
    assert_ne!(
        std::fs::read(&exact.output.stl_path).expect("the clamped STL"),
        std::fs::read(&voxel.output.stl_path).expect("the unclamped STL"),
        "the two settings wrote the same file"
    );
    // The bore in the file is the analytic cylinder rather than a voxel-quantized
    // approximation of one: the vertices that sit on its wall are on it exactly.
    let on_the_bore: Vec<[f64; 3]> = clamped
        .mesh
        .vertices
        .iter()
        .copied()
        .filter(|v| bore.signed_distance(*v) < 0.5 * exact.grid.h)
        .collect();
    assert!(
        !on_the_bore.is_empty(),
        "no exported vertex is anywhere near the bore wall"
    );
    let on_the_wall = on_the_bore
        .iter()
        .filter(|v| bore.signed_distance(**v) <= 2.0 * eps)
        .count();
    assert!(
        on_the_wall > 0,
        "{} vertices are near the bore and not one is on it",
        on_the_bore.len()
    );

    std::fs::remove_file(&exact.output.stl_path).ok();
    std::fs::remove_file(&voxel.output.stl_path).ok();
}

/// The user's bathtub plug, in the shape it was drawn: a tapered cone shell
/// hollowed out from above and a handle standing out of the mouth.
///
/// Nothing here is a topology problem. There is no mass fraction, because the
/// engine that runs it fills the domain: what should come out of the export is
/// the CSG tree itself, held to the analytic cone, the analytic cavity and the
/// analytic cylinder.
const PLUG: &str = r#"
engine = "solid"

[project]
name = "integration_plug"

[resolution]
voxel_size_mm = 1.5

[material]
preset = "petg"

# The reproducible backend: the stress solve of a fully dense part is what says
# whether the handle holds, and the default backend is the machine's device.
[solver]
backend = "cpu"

[optimization]
min_feature_mm = 4.5

[output]
stl_path = "growforge_integration_plug.stl"

# The plug body: a frustum, wider at the bottom so it seats in the drain.
[[domain]]
op = "add"
shape = "cone"
p1 = [24.0, 24.0, 0.0]
p2 = [24.0, 24.0, 20.0]
radius1 = 14.0
radius2 = 11.0

# Hollowed out from above: a wall and a floor, open at the top.
[[domain]]
op = "subtract"
shape = "cone"
p1 = [24.0, 24.0, 4.0]
p2 = [24.0, 24.0, 26.0]
radius1 = 9.0
radius2 = 7.0

# The handle, rooted in that floor and standing out of the mouth.
[[domain]]
op = "add"
shape = "cylinder"
p1 = [24.0, 24.0, 2.0]
p2 = [24.0, 24.0, 26.0]
radius = 3.5

[[supports]]
region = { shape = "box", min = [9.0, 9.0, -0.5], max = [39.0, 39.0, 0.5] }

[[loadcases]]
name = "pull"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [19.0, 19.0, 24.0], max = [29.0, 29.0, 26.5] }
vector = [0.0, 0.0, 30.0]
"#;

/// What the solid engine is for, end to end: a part that was drawn rather than
/// optimized comes out of the pipeline as the shapes it was drawn from.
///
/// The claim is the one the SIMP workaround it replaces cannot make. At
/// `mass_fraction` just under one the optimizer still redistributes material and
/// the density filter still erodes the boundary cells, so the walls of this plug
/// ripple by a fraction of a voxel; here every design cell is filled and the only
/// thing between the CSG tree and the file is the isosurface and the clamp that
/// holds it to those same shapes. So this test asks the *mesh* about the
/// *shapes*, the way `a_bore_is_exported_as_the_cylinder_that_was_asked_for`
/// does, and to the same tolerance: no exported vertex is proud of the domain,
/// and the vertices that sit on a surface are on it exactly.
///
/// It is also the whole pipeline over a zero-iteration field: the cavity pass,
/// the stress solve, the island cull, the clamp, the validation and the STL all
/// run exactly as they do after an optimization.
#[test]
fn the_solid_engine_exports_the_domain_it_was_given() {
    let directory = std::env::temp_dir();
    let config = Config::parse(PLUG).expect("parse");
    let mut problem = Problem::build(&config, &directory).expect("build");
    problem.output.stl_path = directory.join("growforge_integration_plug_solid.stl");
    assert_eq!(problem.engine, constants::SOLID_ENGINE);
    assert!(problem.warnings.is_empty(), "{:?}", problem.warnings);
    assert_eq!(
        problem.output.boundaries,
        BoundaryFidelity::Exact,
        "the clamp has to be what a configuration gets without asking"
    );

    let outcome = optimize_and_export(&problem, &SilentReporter).expect("the solid run");

    // The field is the classification: nothing was optimized, and nothing
    // iterated.
    assert_eq!(outcome.field.iterations, 0);
    assert_eq!(outcome.field.volume_fraction, 1.0);
    for (e, kind) in problem.grid.cells.iter().enumerate() {
        let expected = match kind {
            CellKind::Void => 0.0,
            _ => 1.0,
        };
        assert_eq!(
            outcome.field.densities[e], expected,
            "cell {e} of kind {kind:?} came out at {}",
            outcome.field.densities[e]
        );
    }

    // One part, in one piece, with nothing floating and no cavity sealed inside
    // it: a cup that is open at the top has none to find.
    mesh::validate::validate(&outcome.mesh).expect("watertight surface");
    assert_eq!(outcome.islands.bodies.len(), 1);
    assert_eq!(outcome.islands.culled_fragments, 0);
    assert!(
        outcome.islands.unserved.is_empty(),
        "{:?}",
        outcome.islands.unserved
    );
    assert_eq!(
        outcome.voids.voids.len(),
        0,
        "an open cup has no enclosed cavity: {:?}",
        outcome.voids.voids
    );
    assert_eq!(outcome.voids.filled_cells, 0);

    // The exactness claim, vertex by vertex against the analytic tree.
    let eps = constants::BOUNDARY_CLAMP_EPS_MM;
    let domain = &problem.boundaries.domain;
    let worst = outcome
        .mesh
        .vertices
        .iter()
        .fold(f64::NEG_INFINITY, |worst, v| {
            worst.max(domain.signed_distance(*v))
        });
    assert!(
        worst <= eps,
        "the exported surface stands {worst} mm proud of the domain it was drawn from"
    );
    // And the other half of the same claim, which is the one that says the wall
    // is a cone rather than merely not proud of one: no vertex is *short* of the
    // surface either. Marching cubes and the smoothing after it scatter a wall's
    // vertices to both sides; correcting only the proud ones leaves inward
    // dimples of a third to a half of a voxel, which is what scallops an
    // exported cone. Measured on this fixture before the clamp seated them: the
    // worst vertex sat 0.7018 mm - 0.47 voxels - inside the wall.
    //
    // Every vertex of this part rests on a boundary, because the part *is* the
    // domain, so the claim is made over all of them rather than over a band: the
    // furthest any vertex sits from the tree is the offset a correction lands on.
    let off_the_surface = outcome.mesh.vertices.iter().fold(0.0f64, |worst, v| {
        worst.max(domain.signed_distance(*v).abs())
    });
    assert!(
        off_the_surface <= 2.0 * eps,
        "the exported surface dimples {off_the_surface} mm inside the domain it was drawn from"
    );

    let report = outcome.clamp.expect("a clamp report");
    assert!(report.vertices_moved > 0, "{report:?}");
    assert_eq!(report.gave_up, 0, "{report:?}");
    assert!(
        report.max_displacement_mm <= problem.grid.h,
        "{report:?} moved a vertex further than a voxel"
    );
    // And the same claim as the pass itself makes it. This test used to count
    // the vertices resting off the surface with a helper of its own; the clamp
    // now counts them - it has to, because a run that ships one has to say so -
    // and there is one definition of the count rather than two that can drift.
    // On this part every surface is a domain surface, so the answer is none.
    assert_eq!(
        report.adrift,
        0,
        "{} of {} vertices rest off the domain they were drawn from, the worst by {} mm",
        report.adrift,
        outcome.mesh.vertices.len(),
        report.max_adrift_mm
    );
    assert_eq!(report.max_adrift_mm, 0.0, "{report:?}");
    // Which is the whole of what a clean solid run says: no warning line, and
    // nothing about anything adrift.
    let notes = report.notes(problem.is_solid(), problem.is_flushing());
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(
        !notes.iter().any(|note| note.starts_with("warning")),
        "{notes:?}"
    );

    // And the vertices that sit on a surface sit on it exactly, on each of the
    // three shapes the part was drawn from - the outer cone, the cavity that was
    // cut out of it and the handle - rather than on a voxel quantization of one.
    use growforge::geometry::Shape;
    for (what, shape) in [
        (
            "the outer cone",
            Shape::Cone {
                p1: [24.0, 24.0, 0.0],
                p2: [24.0, 24.0, 20.0],
                radius1: 14.0,
                radius2: 11.0,
            },
        ),
        (
            "the cavity",
            Shape::Cone {
                p1: [24.0, 24.0, 4.0],
                p2: [24.0, 24.0, 26.0],
                radius1: 9.0,
                radius2: 7.0,
            },
        ),
        (
            "the handle",
            Shape::Cylinder {
                p1: [24.0, 24.0, 2.0],
                p2: [24.0, 24.0, 26.0],
                radius: 3.5,
            },
        ),
    ] {
        let near: Vec<[f64; 3]> = outcome
            .mesh
            .vertices
            .iter()
            .copied()
            .filter(|v| shape.signed_distance(*v).abs() < 0.5 * problem.grid.h)
            .collect();
        assert!(
            !near.is_empty(),
            "no exported vertex is anywhere near {what}"
        );
        let on_it = near
            .iter()
            .filter(|v| shape.signed_distance(**v).abs() <= 2.0 * eps)
            .count();
        assert!(
            on_it > 0,
            "{} vertices are near {what} and not one is on it",
            near.len()
        );
    }

    // The part holds together well enough to be analysed: the stress pass ran
    // over the field that shipped, as it does after any other engine.
    assert!(
        outcome.stress.is_available(),
        "the stress pass said nothing"
    );
    assert!(outcome.stats.triangles > 0 && outcome.stats.volume_mm3 > 0.0);
    let (_, triangles) = mesh::stl::read(&outcome.stl_path).expect("read back the STL");
    assert_eq!(triangles.len(), outcome.stats.triangles);
    std::fs::remove_file(&outcome.stl_path).ok();
}

/// A wall lying on the floor of its own domain, at the shape a density
/// optimizer leaves one in.
///
/// Half the footprint of the block, held at one end and pushed at the other, so
/// the field is a load path rather than a slab in mid air; the other half of
/// that floor is deliberately bare. What the flush pass is measured on is the
/// outermost layer of the wall, which the field below leaves under the iso
/// level: the erosion a density filter produces against a boundary it has no
/// design space on the far side of.
const FLOOR_WALL: &str = r#"
[project]
name = "integration_flush"

[resolution]
voxel_size_mm = 2.0

[material]
preset = "petg"

# The reproducible backend: this fixture's deviations are compared against the
# boundary clamp's own offset.
[solver]
backend = "cpu"

[optimization]
mass_fraction = 0.4
min_feature_mm = 8.0
max_iterations = 1

[output]
stl_path = "growforge_integration_flush.stl"
smoothing_iterations = 4
flush = "walls"

[[domain]]
op = "add"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [48.0, 24.0, 24.0]

[[supports]]
region = { shape = "box", min = [-0.5, -0.5, -0.5], max = [0.5, 24.5, 6.5] }

[[loadcases]]
name = "tip"
[[loadcases.loads]]
type = "force"
region = { shape = "box", min = [21.5, -0.5, -0.5], max = [24.5, 24.5, 6.5] }
vector = [0.0, 0.0, -50.0]
"#;

/// The flush pass end to end: the same field exported twice, once as it stands
/// and once with `flush = "walls"`, and the second file is the first one with
/// its wall standing on the floor it was drawn against.
///
/// The field is written by hand rather than optimized, for the reason the
/// pipeline fixtures in `lib.rs` are: what is under test is what the pass and
/// the boundary clamp do with a *shape* - a wall whose outermost cells came out
/// under the iso level - and not whether some engine converges on one. The
/// densities below are what a run really leaves against a boundary: solid two
/// cells in, eroded in the last one, and eroded by different amounts across the
/// face, which is what makes the exported floor ripple rather than merely sit
/// low.
///
/// Three claims, in the order they matter:
///
/// * with the pass on, every vertex of that floor is on the analytic face, to
///   the same `2 x BOUNDARY_CLAMP_EPS_MM` the plug and the bore are held to -
///   measured at 1.0e-5 mm, which is the clamp's own offset;
/// * with it off, the same fixture dimples 1.6214 mm - 0.81 of a voxel, past
///   the 1.0 mm the clamp captures - which is why the mesh side cannot fix this
///   alone;
/// * the bare half of the same floor grows nothing, a fringe of the fill's own
///   depth past the end of the wall aside.
#[test]
fn a_wall_that_came_out_short_of_the_floor_is_exported_flush_with_it() {
    use growforge::config::FlushPolicy;
    use growforge::stress::StressLimits;
    use growforge::{Unwatched, complete};

    let directory = std::env::temp_dir();
    let config = Config::parse(FLOOR_WALL).expect("parse");

    let problem = |name: &str, flush: FlushPolicy| {
        let mut problem = Problem::build(&config, &directory).expect("build");
        problem.output.flush = flush;
        problem.output.stl_path = directory.join(format!("growforge_integration_{name}.stl"));
        problem
    };
    let flushed = problem("flush_on", FlushPolicy::Walls);
    let plain = problem("flush_off", FlushPolicy::Off);
    let grid = &flushed.grid;
    assert_eq!((grid.nx, grid.ny, grid.nz), (24, 12, 12));
    assert_eq!(flushed.output.flush, FlushPolicy::Walls);
    assert_eq!(
        flushed.output.flush_depth_mm, None,
        "the fixture names no depth, so the pass runs at the derived default"
    );
    let depth = constants::FLUSH_DEPTH_VOXELS * grid.h;

    // The field: a wall over the near half of the floor, solid from the second
    // cell up and under the iso level in the first. The two levels in that
    // layer are what make the exported face a ripple - each dips the isosurface
    // inwards by a different fraction of a voxel - and both are below the level
    // the surface is drawn at, so neither is part of the part.
    let wall_columns = grid.nx / 2;
    let mut densities = vec![0.1; grid.n_cells()];
    for i in 0..wall_columns {
        for j in 0..grid.ny {
            densities[grid.cell_index(i, j, 0)] = if j < grid.ny / 2 { 0.2 } else { 0.45 };
            for k in 1..3 {
                densities[grid.cell_index(i, j, k)] = 1.0;
            }
        }
    }
    let before = densities.clone();

    let mut flushed_field = densities.clone();
    let with_flush = complete(
        &flushed,
        &mut flushed_field,
        StressLimits::default(),
        &Unwatched,
    )
    .expect("the flushed pipeline")
    .expect("nothing asked it to stop");
    let mut plain_field = densities.clone();
    let without_flush = complete(
        &plain,
        &mut plain_field,
        StressLimits::default(),
        &Unwatched,
    )
    .expect("the plain pipeline")
    .expect("nothing asked it to stop");
    assert!(
        without_flush.flush.is_none(),
        "the pass ran with flush = \"off\""
    );
    assert_eq!(plain_field, before, "flush = \"off\" moved the field");

    let report = with_flush.flush.expect("a flush report");
    assert!(report.changed_the_field(), "{report:?}");
    assert_eq!(report.depth_mm, depth);
    assert!(report.any_surface);
    assert!(
        (report.volume_added_mm3 - report.cells_raised as f64 * grid.cell_volume()).abs() < 1e-12
    );
    let notes = report.notes();
    assert!(notes[0].starts_with("filled"), "{notes:?}");
    assert!(
        notes[1].contains("joined to it"),
        "the caveat is not stated: {notes:?}"
    );

    // The wall's outermost layer really is solid now, and nothing that was
    // already material lost any.
    for i in 0..wall_columns {
        for j in 0..grid.ny {
            let cell = grid.cell_index(i, j, 0);
            assert_eq!(
                flushed_field[cell], 1.0,
                "the wall is still short of the floor at ({i}, {j}, 0)"
            );
        }
    }
    for cell in 0..grid.n_cells() {
        assert!(
            flushed_field[cell] >= before[cell],
            "cell {cell} lost material"
        );
    }

    // The bare half of the same floor. Everything past the fringe the fill is
    // allowed - `flush_depth_mm` beyond the last material, which is the pass's
    // own documented caveat - is untouched, at every height and not merely at
    // the floor.
    let bare = wall_columns + (depth / grid.h).ceil() as usize + 1;
    let mut checked = 0;
    for i in bare..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let cell = grid.cell_index(i, j, k);
                assert_eq!(
                    flushed_field[cell], before[cell],
                    "the bare floor grew a skin at ({i}, {j}, {k})"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "the fixture leaves no bare floor to assert on");

    // How far the exported floor sits from the face it was drawn against, over
    // the vertices above the middle of the wall. The window is geometric rather
    // than a distance to the boundary, deliberately: selecting the vertices that
    // are already *near* the face would leave out exactly the ones the defect
    // produces, which is the case this fixture exists to measure. Inside it the
    // nearest face of the domain is the floor and nothing else, so the distance
    // to the analytic surface is the height above it.
    let domain = &flushed.boundaries.domain;
    let floor_band = |vertices: &[[f64; 3]]| {
        let mut worst: f64 = 0.0;
        let mut counted = 0usize;
        for vertex in vertices {
            if !(6.0..=18.0).contains(&vertex[0])
                || !(6.0..=18.0).contains(&vertex[1])
                || vertex[2] > 3.0
            {
                continue;
            }
            counted += 1;
            worst = worst.max(domain.signed_distance(*vertex).abs());
        }
        (counted, worst)
    };
    let eps = constants::BOUNDARY_CLAMP_EPS_MM;
    let capture = constants::BOUNDARY_CLAMP_CAPTURE_VOXELS * grid.h;
    let (plain_count, plain_worst) = floor_band(&without_flush.mesh.vertices);
    let (flushed_count, flushed_worst) = floor_band(&with_flush.mesh.vertices);

    // The defect, measured: the unflushed wall stands off its own floor by more
    // than the clamp can reach, which is why the field is what has to be filled.
    assert!(
        plain_count > 0,
        "the unflushed export has no vertex over the wall at all"
    );
    assert!(
        plain_worst > capture,
        "the unflushed floor is only {plain_worst} mm off the face, inside the {capture} mm the \
         clamp seats by itself, so this fixture cannot show what the pass is for"
    );
    assert!(
        plain_worst < grid.h,
        "{plain_worst} mm is not a sampling artefact"
    );
    // And the fix, vertex by vertex against the analytic box.
    assert!(
        flushed_count > 0,
        "the flushed export has no vertex over the wall, so the claim below is vacuous"
    );
    assert!(
        flushed_worst <= 2.0 * eps,
        "the flushed floor still stands {flushed_worst} mm off the face it rests on (the same \
         fixture measures {plain_worst} mm unflushed)"
    );

    // Both files are parts, and they are not the same file.
    for completed in [&with_flush, &without_flush] {
        mesh::validate::validate(&completed.mesh).expect("watertight surface");
        assert_eq!(completed.islands.bodies.len(), 1);
    }
    assert!(
        with_flush.stats.volume_mm3 > without_flush.stats.volume_mm3,
        "the flushed surface encloses {:.1} mm3 against the plain {:.1}",
        with_flush.stats.volume_mm3,
        without_flush.stats.volume_mm3
    );
    assert_ne!(
        std::fs::read(&flushed.output.stl_path).expect("the flushed STL"),
        std::fs::read(&plain.output.stl_path).expect("the plain STL"),
        "the two settings wrote the same file"
    );
    std::fs::remove_file(&flushed.output.stl_path).ok();
    std::fs::remove_file(&plain.output.stl_path).ok();
}
