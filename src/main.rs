//! growforge command line interface.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};

use growforge::config::Config;
use growforge::constants;
use growforge::problem::Problem;
use growforge::report::{
    ConsoleReporter, print_bench_report, print_clamp_report, print_flush_report,
    print_island_report, print_mesh_stats, print_problem_summary, print_reduce_finish,
    print_reinforce_report, print_self_weight, print_solid_report, print_stress_report,
    print_trim_report, print_void_report, print_warnings,
};
use growforge::{RunOutcome, load_config_and_problem};

#[derive(Parser)]
// `name` is the command: what is typed, and what the usage line and every error
// message spell. `display_name` is the product: what `--version` answers with.
// They are deliberately different strings - the brand extends the command name
// rather than replacing it - so a rebrand never changes what a user types.
#[command(
    name = constants::PROGRAM_NAME,
    display_name = constants::DISPLAY_NAME,
    version,
    about = "Grows strong, weight-optimized 3D structures with FEA based topology optimization",
    long_about = "growforge 3D reads a TOML problem definition, voxelizes the design domain, runs \
                  SIMP topology optimization with a real finite element solve, the fast growth \
                  heuristic, or no optimization at all when the part was drawn rather than \
                  optimized, reports the von Mises stresses of the result and exports a \
                  watertight binary STL. Units are millimetres, newtons, MPa and N*mm."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate a configuration and print the problem summary without optimizing.
    Check {
        /// Path to the TOML problem definition.
        config: PathBuf,
    },
    /// Show the problem setup in a 3D window without optimizing.
    View {
        /// Path to the TOML problem definition.
        config: PathBuf,
    },
    /// Edit the problem definition visually: drag the geometry, change every
    /// value numerically, re-run on the spot and save the file back.
    Edit {
        /// Path to the TOML problem definition. It is written only when you
        /// save. Left out, the file dialog asks for one, starting in the
        /// growforge folder of your Documents.
        config: Option<PathBuf>,
    },
    /// Time the linear solve of a configuration on every available backend.
    Bench {
        /// Path to the TOML problem definition.
        config: PathBuf,
    },
    /// Optimize the structure and write the STL.
    Run {
        /// Path to the TOML problem definition.
        config: PathBuf,
        /// Suppress the per-iteration progress lines.
        #[arg(long)]
        quiet: bool,
        /// Watch the density surface evolve in a 3D window. Closing the window
        /// detaches the viewer; the run finishes and writes its STL either way.
        #[arg(long)]
        view: bool,
    },
}

fn main() -> ExitCode {
    match dispatch() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch() -> Result<()> {
    let command = Cli::parse().command;
    // Every invocation names the build before it says anything else, so a
    // console log, a pasted transcript or a terminal left open beside a window
    // is attributable to a binary. Printed here rather than per subcommand, and
    // after the parse rather than before it: `--version` and `--help` are
    // answered by clap inside `parse`, which exits, so neither is given this
    // line twice.
    println!("{}", constants::DISPLAY_NAME_AND_VERSION);
    match command {
        Command::Check { config } => {
            let (_, problem) = load_config_and_problem(&config)?;
            print_warnings(&problem.warnings);
            print_problem_summary(&problem);
            println!("configuration is valid");
            Ok(())
        }
        Command::Bench { config } => {
            let (_, problem) = load_config_and_problem(&config)?;
            print_warnings(&problem.warnings);
            print_bench_report(&growforge::bench::run(&problem)?);
            Ok(())
        }
        Command::View { config } => {
            let (config, problem) = load_config_and_problem(&config)?;
            print_warnings(&problem.warnings);
            print_problem_summary(&problem);
            println!();
            open_setup_view(&config, &problem)
        }
        // The editor loads the file itself: it has to open a configuration that
        // does not build yet, which is exactly the one you need to edit, and it
        // keeps the file's own text so a save preserves it.
        //
        // It is also the one command that can be started without a path -
        // a Start Menu shortcut has no terminal to type one into - and then it
        // asks for the file in the platform's own dialog before any window
        // exists. Every other command needs its path: there is nothing sensible
        // to check, view, time or run without one.
        Command::Edit { config } => match config {
            Some(config) => open_editor(&config),
            None => open_editor_without_a_path(),
        },
        Command::Run {
            config,
            quiet,
            view,
        } => {
            let (config, problem) = load_config_and_problem(&config)?;
            print_warnings(&problem.warnings);
            print_problem_summary(&problem);
            println!();

            let reporter = ConsoleReporter::new(quiet);
            let outcome = if view {
                run_with_view(&config, &problem, &reporter)?
            } else {
                growforge::optimize_and_export(&problem, &reporter)?
            };

            println!();
            if problem.is_solid() {
                // No iterations, no compliance, and neither is printed as a
                // zero: what a solid run has to say is what it filled. The
                // stress report further down is what says how good it is.
                println!(
                    "solid          {} design cells filled, {} already solid; nothing was \
                     optimized",
                    problem.counts.design, problem.counts.solid
                );
            } else {
                match &outcome.field.growth {
                    // A growth run has no compliance to report; what it grew, and
                    // the stress report further down, are what describe it.
                    Some(growth) => {
                        println!(
                            "growth steps   {} ({})",
                            outcome.field.iterations,
                            if outcome.field.stop.converged() {
                                "growth complete"
                            } else {
                                "step cap"
                            }
                        );
                        println!(
                            "skeleton       {} backbones, {} segments, radius {:.2} .. {:.2} mm",
                            growth.backbones,
                            growth.segments,
                            growth.radius_range_mm.0,
                            growth.radius_range_mm.1
                        );
                        println!(
                            "attractors     {} scattered, {} consumed",
                            growth.attractors, growth.consumed
                        );
                        println!(
                            "connections    {} surface targets, {} unreached, {} fused branch tips \
                         carrying load",
                            growth.surface_targets, growth.unreached_surfaces, growth.fused_tips
                        );
                        if let Some(symmetry) = growth.symmetry {
                            println!(
                                "symmetry       {}, {} sectors; grown in {} of the {} design cells \
                             and copied{}",
                                symmetry.params.describe(),
                                symmetry.params.sectors(),
                                symmetry.fundamental_design_cells,
                                problem.counts.design,
                                // A transform that does not land on cell centres
                                // gives an exact skeleton and a field resampled a
                                // fraction of a voxel off, so the run says so
                                // rather than letting "symmetric" be read as more
                                // than it is.
                                if symmetry.exact_on_the_voxel_lattice {
                                    ""
                                } else {
                                    " (skeleton exact, rasterized surface approximate to within a \
                                 voxel)"
                                }
                            );
                        }
                        if growth.pruned_nodes > 0 {
                            println!(
                                "pruned         {} branch nodes that ended on nothing",
                                growth.pruned_nodes
                            );
                        }
                        if let Some(achievable) = growth.clamped_volume_fraction {
                            println!(
                                "volume clamp   the radius limits allow {achievable:.4}, not the \
                             requested {:.4}",
                                problem.optimization.mass_fraction
                            );
                        }
                    }
                    None => {
                        // Four outcomes, named apart: an answer, an iterate the
                        // problem will not improve on, whatever the budget ended
                        // on, and the stage an [optimization.reduce] schedule
                        // kept. The engine has already printed the sentence that
                        // says what to do about the second, and the schedule's
                        // own stage lines and summary about the fourth.
                        println!(
                            "iterations     {} ({})",
                            outcome.field.iterations,
                            outcome.field.stop.label()
                        );
                        println!(
                            "compliance     {:.6e} N*mm (first iteration {:.6e})",
                            outcome.field.compliance, outcome.field.initial_compliance
                        );
                    }
                }
            }
            println!("volume frac    {:.4}", outcome.field.volume_fraction);
            if let Some(residual) = outcome.field.overhang_residual {
                println!(
                    "overhang res.  {:.5} max, {:.5} mean |printed - designed| over the design cells",
                    residual.max, residual.mean
                );
            }
            print_trim_report(outcome.trim.as_ref());
            // Beside the trim, in the order the pipeline ran them: what was
            // freed, what was put back out to the surfaces the walls rest on,
            // then what was spent.
            print_flush_report(outcome.flush.as_ref());
            print_reinforce_report(outcome.reinforce.as_ref());
            print_void_report(&outcome.voids);
            print_solid_report(&outcome.solids);
            print_island_report(&outcome.islands, problem.optimization.min_feature_mm);
            // After the island line because that is the order the export ran
            // them in: the clamp only ever moves vertices of the components the
            // cull kept. What the run was goes with it, because what a vertex
            // resting off a boundary means is the run's answer: a defect on a
            // part that was drawn, a flush that fell short on one that asked for
            // its walls to reach, and nothing worth a line on one that did not.
            print_clamp_report(
                outcome.clamp.as_ref(),
                problem.is_solid(),
                problem.is_flushing(),
            );
            // The count is the island report's, printed just above: the table
            // and the warning on it describe the same exported surface.
            print_stress_report(&outcome.stress, outcome.islands.bodies.len());
            // Beside that table because the factor it quotes is the table's: a
            // reduction schedule chose its design before the passes above ran,
            // and this is what the part they left measures against the target it
            // was held to.
            print_reduce_finish(
                outcome.field.reduce.as_ref(),
                outcome.trim.as_ref(),
                outcome.flush.as_ref(),
                outcome.reinforce.as_ref(),
            );
            if let Some(path) = &problem.output.stress_json
                && outcome.stress.is_available()
            {
                println!("stress json    {}", path.display());
            }
            print_mesh_stats(
                &outcome.stats,
                problem.material.density_g_cm3,
                problem.output.supersample,
            );
            print_self_weight(&problem, &outcome.field.densities);
            println!(
                "time           {:.2} s optimizing, {:.2} s analysing, {:.2} s exporting",
                outcome.optimize_s, outcome.analysis_s, outcome.export_s
            );
            println!("wrote          {}", outcome.stl_path.display());
            Ok(())
        }
    }
}

#[cfg(feature = "viewer")]
fn open_setup_view(config: &Config, problem: &Problem) -> Result<()> {
    growforge::viewer::view_setup(config, problem)
}

#[cfg(feature = "viewer")]
fn open_editor(config: &std::path::Path) -> Result<()> {
    growforge::viewer::edit(config)
}

#[cfg(feature = "viewer")]
fn open_editor_without_a_path() -> Result<()> {
    growforge::viewer::edit_without_a_path()
}

#[cfg(feature = "viewer")]
fn run_with_view(
    config: &Config,
    problem: &Problem,
    reporter: &ConsoleReporter,
) -> Result<RunOutcome> {
    growforge::viewer::run_with_view(config, problem, reporter)
}

/// Message used by every viewer entry point when the feature is compiled out.
#[cfg(not(feature = "viewer"))]
const NO_VIEWER: &str = "this build of growforge 3D was compiled without the `viewer` feature, \
                         so it has no window; rebuild with the default features (cargo build \
                         --release) to use `view`, `edit` and `run --view`";

#[cfg(not(feature = "viewer"))]
fn open_setup_view(_config: &Config, _problem: &Problem) -> Result<()> {
    anyhow::bail!(NO_VIEWER)
}

#[cfg(not(feature = "viewer"))]
fn open_editor(_config: &std::path::Path) -> Result<()> {
    anyhow::bail!(NO_VIEWER)
}

#[cfg(not(feature = "viewer"))]
fn open_editor_without_a_path() -> Result<()> {
    anyhow::bail!(NO_VIEWER)
}

#[cfg(not(feature = "viewer"))]
fn run_with_view(
    _config: &Config,
    _problem: &Problem,
    _reporter: &ConsoleReporter,
) -> Result<RunOutcome> {
    anyhow::bail!(NO_VIEWER)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a command line as the binary's own would be parsed, without the
    /// program name a real one starts with.
    fn parse(arguments: &[&str]) -> Result<Command, clap::Error> {
        let mut line = vec![constants::PROGRAM_NAME];
        line.extend_from_slice(arguments);
        Cli::try_parse_from(line).map(|cli| cli.command)
    }

    #[test]
    fn edit_takes_the_path_it_was_given() {
        let Ok(Command::Edit { config }) = parse(&["edit", "parts/bracket.toml"]) else {
            panic!("`edit <path>` did not parse as an edit");
        };
        assert_eq!(config, Some(PathBuf::from("parts/bracket.toml")));
    }

    #[test]
    fn edit_on_its_own_is_a_command_with_no_path_in_it() {
        let Ok(Command::Edit { config }) = parse(&["edit"]) else {
            panic!("`edit` with no path was rejected");
        };
        assert_eq!(config, None, "a path appeared where none was typed");
    }

    #[test]
    fn every_other_command_still_needs_its_path() {
        // Nothing to check, view, time or run without one, so the parser
        // refuses rather than inventing a file.
        for command in ["check", "view", "bench", "run"] {
            let Err(error) = parse(&[command]) else {
                panic!("`{command}` parsed with no path");
            };
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::MissingRequiredArgument,
                "`{command}` was refused for the wrong reason: {error}"
            );
        }
    }

    #[test]
    fn a_path_given_to_another_command_is_the_one_it_gets() {
        let Ok(Command::Run { config, .. }) = parse(&["run", "parts/bracket.toml"]) else {
            panic!("`run <path>` did not parse as a run");
        };
        assert_eq!(config, PathBuf::from("parts/bracket.toml"));
    }
}
