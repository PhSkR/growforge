//! The editor's file layer: a `toml_edit` document kept alongside the typed
//! [`Config`], so that saving rewrites the values that changed and nothing else.
//!
//! [`Document::sync`] projects the configuration into the document key by key.
//! A key whose value already matches is not touched at all, so its spelling
//! (`1e3`, `1_000`, an aligned column, a trailing comment) survives; a key that
//! changed keeps its surrounding whitespace and comment and gets a new value;
//! a key the configuration no longer has is removed. Object lists are kept in
//! step structurally by [`Document::push_object`] and
//! [`Document::remove_object`], which the editor calls together with the
//! matching change to the configuration, so the table that describes an object
//! goes on describing that object however many objects come and go before the
//! file is written.
//!
//! ## Integers beyond the signed 64 bit range
//!
//! `toml_edit` stores every integer as an `i64` and refuses to parse a document
//! containing a larger literal, while `[growth] seed` is a `u64` and the shipped
//! `growth_canopy.toml` uses a seed above `i64::MAX`. Such a literal is
//! therefore replaced, in the text handed to the parser, by a sentinel integer
//! that no other value in the document uses, and swapped back for the original
//! text when the document is rendered. The typed configuration is unaffected -
//! it is parsed from the original text by the `toml` crate, which reads `u64`
//! natively - so the seed the editor shows, runs and writes is always the real
//! one.

use anyhow::{Result, bail};
use toml_edit::{Array, ArrayOfTables, DocumentMut, InlineTable, Item, Table, TableLike, Value};

use crate::config::{Config, LoadSpec, ShapeSpec};
use crate::constants;
use crate::geometry::Vec3;
use crate::viewer::editor::state::domain_op;

/// Which object list a structural edit applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum List {
    /// `[[domain]]`.
    Domain,
    /// `[[keepout]]`.
    Keepout,
    /// `[[keepin]]`.
    Keepin,
    /// `[[supports]]`.
    Supports,
    /// `[[loadcases]]`.
    LoadCases,
    /// `[[loadcases.loads]]` of the load case with this index.
    Loads(usize),
}

impl List {
    /// Key this list lives under.
    fn key(self) -> &'static str {
        match self {
            List::Domain => "domain",
            List::Keepout => "keepout",
            List::Keepin => "keepin",
            List::Supports => "supports",
            List::LoadCases => "loadcases",
            List::Loads(_) => "loads",
        }
    }
}

/// An integer literal too large for `toml_edit`, carried as text.
#[derive(Debug, Clone)]
struct BigInteger {
    /// Stand-in stored in the document.
    sentinel: i64,
    /// The literal exactly as the file spells it, restored on render.
    raw: String,
    /// What that literal means, so an unchanged value can be recognised.
    value: u64,
}

/// A configuration file's text, edited in place.
#[derive(Debug, Clone)]
pub struct Document {
    doc: DocumentMut,
    big: Vec<BigInteger>,
    /// The file's own line ending. `toml_edit` renders every newline as a bare
    /// line feed whatever it parsed, and rewriting every line of a Windows
    /// authored file is not what a save of one value means.
    crlf: bool,
}

impl Document {
    /// Parse a configuration file's text.
    pub fn parse(text: &str) -> Result<Document> {
        let crlf = text.contains("\r\n");
        let mut current = text.to_string();
        let mut big: Vec<BigInteger> = Vec::new();
        loop {
            match current.parse::<DocumentMut>() {
                Ok(doc) => return Ok(Document { doc, big, crlf }),
                Err(error) => {
                    let masked = error
                        .span()
                        .and_then(|span| current.get(span.clone()).map(|raw| (span, raw)))
                        .and_then(|(span, raw)| {
                            big_literal(raw).map(|value| (span, raw.to_string(), value))
                        });
                    let Some((span, raw, value)) = masked else {
                        bail!("{error}");
                    };
                    if big.len() >= constants::VIEW_EDIT_MAX_BIG_INTEGERS {
                        bail!(
                            "this configuration holds more than {} integers above {}, which is \
                             more than the editor can carry through a round trip; edit it as text",
                            constants::VIEW_EDIT_MAX_BIG_INTEGERS,
                            i64::MAX
                        );
                    }
                    let Some(sentinel) = free_sentinel(&current, &big) else {
                        bail!("{error}");
                    };
                    current.replace_range(span, &sentinel.to_string());
                    big.push(BigInteger {
                        sentinel,
                        raw,
                        value,
                    });
                }
            }
        }
    }

    /// The document as text, ready to be written.
    pub fn render(&self) -> String {
        let mut text = self.doc.to_string();
        for big in &self.big {
            text = replace_token(&text, &big.sentinel.to_string(), &big.raw);
        }
        if self.crlf {
            // Normalized first, so a line that already ends in one is not given
            // a second carriage return.
            text = text.replace("\r\n", "\n").replace('\n', "\r\n");
        }
        text
    }

    /// Append an empty table to an object list, for an object that has just
    /// been added to the configuration. [`Document::sync`] fills it in.
    pub fn push_object(&mut self, list: List) {
        if let Some(mut slot) = self.array(list, true) {
            slot.push_empty();
        }
    }

    /// Remove the table of an object that has just been deleted from the
    /// configuration, taking its comments with it.
    pub fn remove_object(&mut self, list: List, index: usize) {
        if let Some(mut slot) = self.array(list, false)
            && index < slot.len()
        {
            slot.remove(index);
        }
    }

    /// Resolve an object list, optionally creating it.
    fn array(&mut self, list: List, create: bool) -> Option<ArraySlot<'_>> {
        let root = self.doc.as_table_mut();
        match list {
            List::Loads(case) => {
                let cases = array_slot(root, List::LoadCases.key(), create)?;
                let table = cases.into_table(case)?;
                array_slot(table, list.key(), create)
            }
            _ => array_slot(root, list.key(), create),
        }
    }

    /// Write every value of `config` into the document.
    ///
    /// Fails only when a number cannot be written at all: an unsigned value
    /// past the signed range for which this document spells every stand-in
    /// already. The document keeps its old value in that case, and the caller
    /// must not write it out - a save that reports success while the file keeps
    /// the previous number is the one outcome an editor may never produce.
    pub fn sync(&mut self, config: &Config) -> Result<()> {
        // Rendered before anything is written, so a sentinel can be chosen that
        // no value in the document already spells.
        let text = self.doc.to_string();
        // Anything a structural edit removed since the last projection - a
        // deleted table can take a stand-in with it - is given back before the
        // pool is drawn from.
        self.prune();
        let Document { doc, big, .. } = self;
        let mut sink = Sink {
            big,
            text: &text,
            refused: Vec::new(),
        };
        let root = doc.as_table_mut();

        set_opt_str(root, "engine", config.engine.as_deref());

        let project = sub_table(root, "project", Style::Table);
        set_str(project, "name", &config.project.name);
        set_opt_str(project, "engine", config.project.engine.as_deref());

        let resolution = sub_table(root, "resolution", Style::Table);
        set_opt_float(resolution, "voxel_size_mm", config.resolution.voxel_size_mm);
        sink.count(resolution, "target_cells", config.resolution.target_cells);

        let material = sub_table(root, "material", Style::Table);
        set_opt_str(material, "preset", config.material.preset.as_deref());
        for (key, value) in [
            ("youngs_modulus_mpa", config.material.youngs_modulus_mpa),
            ("poisson_ratio", config.material.poisson_ratio),
            ("density_g_cm3", config.material.density_g_cm3),
            ("yield_strength_mpa", config.material.yield_strength_mpa),
        ] {
            set_opt_float(material, key, value);
        }

        let optimization = sub_table(root, "optimization", Style::Table);
        // Optional in the schema because one engine refuses it, so the key
        // leaves the file when the solid engine is chosen and comes back with
        // its value when it is left.
        set_opt_float(
            optimization,
            "mass_fraction",
            config.optimization.mass_fraction,
        );
        set_float(
            optimization,
            "min_feature_mm",
            config.optimization.min_feature_mm,
        );
        sink.count(
            optimization,
            "max_iterations",
            config.optimization.max_iterations,
        );
        set_opt_float(
            optimization,
            "convergence_tol",
            config.optimization.convergence_tol,
        );
        set_opt_float(optimization, "penalty", config.optimization.penalty);
        // Exponent notation, as the `[solver]` tolerance is written and for its
        // reason: a floor of 1e-9 spelled out is a decimal point and eight
        // zeros, and the panel, the README and the run's own messages all write
        // these keys the short way.
        set_opt_scientific_float(
            optimization,
            "stiffness_floor",
            config.optimization.stiffness_floor,
        );
        set_opt_str(
            optimization,
            "update",
            config.optimization.update.map(|u| u.label()),
        );
        match &config.optimization.overhang {
            Some(overhang) => {
                let table = sub_table(optimization, "overhang", Style::Table);
                set_str(table, "build_direction", overhang.build_direction.label());
            }
            None => drop(optimization.remove("overhang")),
        }
        match &config.optimization.wireframe {
            Some(wireframe) => {
                let table = sub_table(optimization, "wireframe", Style::Table);
                set_opt_float(table, "radius_mm", wireframe.radius_mm);
                sink.count(table, "hold_iterations", wireframe.hold_iterations);
                set_opt_float(table, "seed_density", wireframe.seed_density);
            }
            None => drop(optimization.remove("wireframe")),
        }
        match &config.optimization.local_volume {
            Some(local) => {
                let table = sub_table(optimization, "local_volume", Style::Table);
                set_opt_float(table, "max_fraction", local.max_fraction);
                set_opt_float(table, "radius_mm", local.radius_mm);
            }
            None => drop(optimization.remove("local_volume")),
        }
        match &config.optimization.reduce {
            Some(reduce) => {
                let table = sub_table(optimization, "reduce", Style::Table);
                set_float(table, "target_safety_factor", reduce.target_safety_factor);
                set_opt_str(table, "method", reduce.method.map(|m| m.label()));
                set_opt_float(table, "ratio", reduce.ratio);
                sink.count(table, "refine_stages", reduce.refine_stages);
                set_opt_float(table, "min_mass_fraction", reduce.min_mass_fraction);
                set_opt_float(table, "evolution_rate", reduce.evolution_rate);
                set_opt_float(table, "add_ratio", reduce.add_ratio);
            }
            None => drop(optimization.remove("reduce")),
        }

        match &config.solver {
            Some(solver) => {
                let table = sub_table(root, "solver", Style::Table);
                set_opt_str(table, "backend", solver.backend.map(|b| b.label()));
                set_opt_scientific_float(table, "tolerance", solver.tolerance);
            }
            None => drop(root.remove("solver")),
        }

        match &config.growth {
            Some(growth) => {
                let table = sub_table(root, "growth", Style::Table);
                sink.unsigned(table, "seed", growth.seed);
                for (key, value) in [
                    ("attractor_per_cm3", growth.attractor_per_cm3),
                    ("attraction_radius_mm", growth.attraction_radius_mm),
                    ("kill_radius_mm", growth.kill_radius_mm),
                    ("step_mm", growth.step_mm),
                    ("murray_exponent", growth.murray_exponent),
                    ("max_radius_mm", growth.max_radius_mm),
                ] {
                    set_opt_float(table, key, value);
                }
                sink.count(table, "max_steps", growth.max_steps);
                set_opt_bool(table, "prune", growth.prune);
                match &growth.symmetry {
                    Some(symmetry) => {
                        let table = sub_table(table, "symmetry", Style::Table);
                        set_str(table, "kind", symmetry.kind.label());
                        match &symmetry.planes {
                            Some(planes) => {
                                let mut array = Array::new();
                                for axis in planes {
                                    array.push(axis.label());
                                }
                                array.fmt();
                                set_if_changed(table, "planes", Value::Array(array));
                            }
                            None => drop(table.remove("planes")),
                        }
                        sink.count(table, "order", symmetry.order);
                        set_opt_str(table, "axis", symmetry.axis.map(|a| a.label()));
                    }
                    None => drop(table.remove("symmetry")),
                }
            }
            None => drop(root.remove("growth")),
        }

        let output = sub_table(root, "output", Style::Table);
        set_str(output, "stl_path", &config.output.stl_path);
        set_opt_float(output, "iso_level", config.output.iso_level);
        sink.count(
            output,
            "smoothing_iterations",
            config.output.smoothing_iterations,
        );
        sink.count(output, "supersample", config.output.supersample);
        set_opt_str(output, "voids", config.output.voids.map(|v| v.label()));
        set_opt_str(output, "islands", config.output.islands.map(|i| i.label()));
        set_opt_str(
            output,
            "boundaries",
            config.output.boundaries.map(|b| b.label()),
        );
        set_opt_str(output, "trim", config.output.trim.map(|t| t.label()));
        set_opt_float(
            output,
            "trim_stress_fraction",
            config.output.trim_stress_fraction,
        );
        set_opt_str(
            output,
            "reinforce",
            config.output.reinforce.map(|r| r.label()),
        );
        set_opt_float(
            output,
            "reinforce_thickness_mm",
            config.output.reinforce_thickness_mm,
        );
        set_opt_str(output, "flush", config.output.flush.map(|f| f.label()));
        set_opt_float(output, "flush_depth_mm", config.output.flush_depth_mm);
        set_opt_str(output, "stress_json", config.output.stress_json.as_deref());

        sync_array(root, List::Domain.key(), &config.domain, |table, entry| {
            set_str(table, "op", domain_op(entry).label());
            write_shape(table, &entry.shape_spec());
        });
        sync_array(root, List::Keepout.key(), &config.keepout, write_shape);
        sync_array(root, List::Keepin.key(), &config.keepin, write_shape);
        sync_array(
            root,
            List::Supports.key(),
            &config.supports,
            |table, spec| {
                let region = sub_table(table, "region", Style::Inline);
                write_shape(region, &spec.region);
                match &spec.directions {
                    Some(axes) => {
                        let mut array = Array::new();
                        for axis in axes {
                            array.push(axis.label());
                        }
                        array.fmt();
                        set_if_changed(table, "directions", Value::Array(array));
                    }
                    None => drop(table.remove("directions")),
                }
            },
        );
        sync_array(
            root,
            List::LoadCases.key(),
            &config.loadcases,
            |table, case| {
                set_str(table, "name", &case.name);
                set_opt_float(table, "weight", case.weight);
                // Every case spells its own load list the same way, so the index
                // this key is asked for does not matter.
                sync_array(table, List::Loads(0).key(), &case.loads, write_load);
            },
        );

        sink.finish()?;
        // Pruned again now that the projection has replaced what it replaced,
        // so what the pool holds afterwards is exactly the stand-ins the file
        // holds: an edit from one huge number to another gives the first one's
        // stand-in straight back.
        self.prune();
        Ok(())
    }

    /// Drop the stand-ins the document no longer holds.
    ///
    /// "Still in the rendered text" is exactly the condition under which
    /// [`Document::render`] would swap one back, so this can never drop a live
    /// one; and because a stand-in is only ever chosen from values the document
    /// did not already spell, an occurrence in that text is always one this
    /// module wrote.
    fn prune(&mut self) {
        let text = self.doc.to_string();
        self.big
            .retain(|entry| find_token(&text, &entry.sentinel.to_string()).is_some());
    }
}

/// What the unsigned projections need beyond the table they write into: the
/// pool of stand-ins, the text a new one must not collide with, and the keys
/// that could not be written at all.
///
/// The last of those is the point. There is no number the pool cannot hold once
/// stale entries are pruned - the eight unsigned keys of the schema can need at
/// most eight stand-ins, and the pool offers sixteen - but if it ever did run
/// out, the alternative to reporting it is leaving the file's old value in place
/// while telling the user their edit was saved. A save that cannot be made must
/// fail.
struct Sink<'a> {
    big: &'a mut Vec<BigInteger>,
    text: &'a str,
    refused: Vec<&'static str>,
}

impl Sink<'_> {
    /// Write an optional count.
    ///
    /// `usize` is at most 64 bits on every platform this crate builds for, so
    /// the widening to `u64` cannot lose anything; the narrowing that can - to
    /// the `i64` a TOML document holds - is what [`Sink::unsigned`] guards.
    fn count(&mut self, table: &mut dyn TableLike, key: &'static str, value: Option<usize>) {
        self.unsigned(table, key, value.map(|value| value as u64));
    }

    /// Write an optional unsigned integer, which the document can only hold
    /// directly up to `i64::MAX`.
    ///
    /// Every unsigned key in the schema goes through here - the counts as well
    /// as `[growth] seed` - because any of them can legally hold a number the
    /// document cannot: a `usize` is 64 bits wide and TOML's integers are
    /// signed. A value already carried by a stand-in and still unchanged is left
    /// exactly as the file spells it; a larger one, new or changed, takes a
    /// fresh stand-in and is restored as text when the document is rendered. The
    /// conversion is checked, so no configuration integer can ever be written as
    /// its wrapped two's-complement self - which would produce a file growforge
    /// itself would refuse to read.
    fn unsigned(&mut self, table: &mut dyn TableLike, key: &'static str, value: Option<u64>) {
        let Some(value) = value else {
            table.remove(key);
            return;
        };
        let carried = table
            .get(key)
            .and_then(Item::as_integer)
            .and_then(|current| self.big.iter().find(|entry| entry.sentinel == current));
        if carried.is_some_and(|entry| entry.value == value) {
            return;
        }
        if let Ok(signed) = i64::try_from(value) {
            set_if_changed(table, key, Value::from(signed));
            return;
        }
        let Some(sentinel) = free_sentinel(self.text, self.big) else {
            // The key keeps whatever it had, and the save this belongs to is
            // refused rather than silently writing the old number back.
            self.refused.push(key);
            return;
        };
        self.big.push(BigInteger {
            sentinel,
            raw: value.to_string(),
            value,
        });
        set_if_changed(table, key, Value::from(sentinel));
    }

    /// Fail the projection if anything could not be written.
    fn finish(self) -> Result<()> {
        if self.refused.is_empty() {
            return Ok(());
        }
        bail!(
            "this configuration's {} cannot be written: it is above {}, and this file already \
             spells every stand-in the editor has for such a number. Lower it, or edit the file \
             as text",
            self.refused.join(", "),
            i64::MAX
        )
    }
}

/// Write one load's own keys.
fn write_load(table: &mut dyn TableLike, load: &LoadSpec) {
    set_str(table, "type", load.kind());
    let mine: &[&str] = match load {
        LoadSpec::Force { region, vector } => {
            write_shape(sub_table(table, "region", Style::Inline), region);
            set_vec3(table, "vector", *vector);
            &["region", "vector"]
        }
        LoadSpec::Torque {
            region,
            axis_point,
            axis_dir,
            magnitude_nmm,
        } => {
            write_shape(sub_table(table, "region", Style::Inline), region);
            set_vec3(table, "axis_point", *axis_point);
            set_vec3(table, "axis_dir", *axis_dir);
            set_float(table, "magnitude_nmm", *magnitude_nmm);
            &["region", "axis_point", "axis_dir", "magnitude_nmm"]
        }
        LoadSpec::Gravity { direction, g_mm_s2 } => {
            match direction {
                Some(direction) => drop(set_vec3(table, "direction", *direction)),
                None => drop(table.remove("direction")),
            }
            set_opt_float(table, "g_mm_s2", *g_mm_s2);
            &["direction", "g_mm_s2"]
        }
    };
    remove_other(table, &LOAD_KEYS, mine);
}

/// Every geometry key any shape kind uses. A kind writes its own and drops the
/// ones only the other kinds have, so a shape that changed kind leaves nothing
/// of its old self behind.
const SHAPE_KEYS: [&str; 15] = [
    "min",
    "max",
    "rotation_deg",
    "p1",
    "p2",
    "bend",
    "center",
    "radius",
    "radii",
    "radius1",
    "radius2",
    "a",
    "b",
    "c",
    "thickness",
];

/// Every key any load kind uses beyond its `type`, for the same reason.
const LOAD_KEYS: [&str; 7] = [
    "region",
    "vector",
    "axis_point",
    "axis_dir",
    "magnitude_nmm",
    "direction",
    "g_mm_s2",
];

/// Drop the keys of `all` that this variant does not write.
fn remove_other(table: &mut dyn TableLike, all: &[&str], mine: &[&str]) {
    for key in all.iter().filter(|key| !mine.contains(key)) {
        table.remove(key);
    }
}

/// Write a shape's `shape` key and its geometry.
fn write_shape(table: &mut dyn TableLike, spec: &ShapeSpec) {
    let mut inserted = set_str(table, "shape", spec.kind());
    let mine: &[&str] = match *spec {
        ShapeSpec::Box {
            min,
            max,
            rotation_deg,
        } => {
            inserted |= set_vec3(table, "min", min);
            inserted |= set_vec3(table, "max", max);
            // Absent is the same geometry as [0, 0, 0], so a box that carries
            // no rotation writes no key: a shape nobody turned comes back out
            // of the editor exactly as it went in.
            match rotation_deg {
                Some(rotation) => inserted |= set_vec3(table, "rotation_deg", rotation),
                None => drop(table.remove("rotation_deg")),
            }
            &["min", "max", "rotation_deg"]
        }
        ShapeSpec::Cylinder { p1, p2, radius } => {
            inserted |= set_vec3(table, "p1", p1);
            inserted |= set_vec3(table, "p2", p2);
            inserted |= set_float(table, "radius", radius);
            &["p1", "p2", "radius"]
        }
        ShapeSpec::Sphere { center, radius } => {
            inserted |= set_vec3(table, "center", center);
            inserted |= set_float(table, "radius", radius);
            &["center", "radius"]
        }
        ShapeSpec::Ellipsoid {
            center,
            radii,
            rotation_deg,
        } => {
            inserted |= set_vec3(table, "center", center);
            inserted |= set_vec3(table, "radii", radii);
            // Absent stays absent, exactly as a box's does.
            match rotation_deg {
                Some(rotation) => inserted |= set_vec3(table, "rotation_deg", rotation),
                None => drop(table.remove("rotation_deg")),
            }
            &["center", "radii", "rotation_deg"]
        }
        ShapeSpec::Tube {
            p1,
            p2,
            bend,
            radius,
        } => {
            inserted |= set_vec3(table, "p1", p1);
            inserted |= set_vec3(table, "p2", p2);
            // A tube nobody bent carries no bend key, exactly as a box nobody
            // turned carries no rotation - and straightening one takes the key
            // away again.
            match bend {
                Some(bend) => inserted |= set_vec3(table, "bend", bend),
                None => drop(table.remove("bend")),
            }
            inserted |= set_float(table, "radius", radius);
            &["p1", "p2", "bend", "radius"]
        }
        ShapeSpec::Cone {
            p1,
            p2,
            radius1,
            radius2,
        } => {
            inserted |= set_vec3(table, "p1", p1);
            inserted |= set_vec3(table, "p2", p2);
            inserted |= set_float(table, "radius1", radius1);
            // Written even when it is zero: the apex of a true cone is a value
            // the shape has, not a key it does without.
            inserted |= set_float(table, "radius2", radius2);
            &["p1", "p2", "radius1", "radius2"]
        }
        ShapeSpec::Triangle { a, b, c, thickness } => {
            inserted |= set_vec3(table, "a", a);
            inserted |= set_vec3(table, "b", b);
            inserted |= set_vec3(table, "c", c);
            inserted |= set_float(table, "thickness", thickness);
            &["a", "b", "c", "thickness"]
        }
    };
    remove_other(table, &SHAPE_KEYS, mine);
    // A key added to an inline table would otherwise land against its
    // neighbours with no spacing; a table nothing was added to is left exactly
    // as the file wrote it.
    if inserted {
        table.fmt();
    }
}

/// Whether a sub-table is written as its own `[header]` or inline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Style {
    /// A `[header]` table.
    Table,
    /// An inline `{ ... }` table, which is how the shipped examples write the
    /// region of a support or a load.
    Inline,
}

/// Get a sub-table, creating it in the requested style when it is missing.
fn sub_table<'a>(parent: &'a mut dyn TableLike, key: &str, style: Style) -> &'a mut dyn TableLike {
    if !parent.get(key).is_some_and(Item::is_table_like) {
        let item = match style {
            Style::Table => Item::Table(Table::new()),
            Style::Inline => Item::Value(Value::InlineTable(InlineTable::new())),
        };
        parent.insert(key, item);
    }
    parent
        .get_mut(key)
        .and_then(Item::as_table_like_mut)
        .expect("the table was just created if it was missing")
}

/// Bring an object list to the length `items` has and write every entry.
fn sync_array<T>(
    parent: &mut dyn TableLike,
    key: &str,
    items: &[T],
    mut write: impl FnMut(&mut dyn TableLike, &T),
) {
    let create = !items.is_empty();
    let Some(mut slot) = array_slot(parent, key, create) else {
        return;
    };
    while slot.len() > items.len() {
        slot.remove(slot.len() - 1);
    }
    while slot.len() < items.len() {
        slot.push_empty();
    }
    for (index, item) in items.iter().enumerate() {
        if let Some(table) = slot.table(index) {
            write(table, item);
        }
    }
    if items.is_empty() {
        parent.remove(key);
    }
}

/// An object list, however the file spells it.
enum ArraySlot<'a> {
    /// The `[[key]]` form.
    Tables(&'a mut ArrayOfTables),
    /// An inline `key = [{ ... }]` array, which is legal TOML for the same
    /// structure and which a hand-written file may use.
    Inline(&'a mut Array),
}

/// Resolve an object list under `key`, creating an empty `[[key]]` array when
/// there is none and `create` is set.
fn array_slot<'a>(parent: &'a mut dyn TableLike, key: &str, create: bool) -> Option<ArraySlot<'a>> {
    let present = parent
        .get(key)
        .is_some_and(|item| item.is_array_of_tables() || item.is_array());
    if !present {
        if !create {
            return None;
        }
        parent.insert(key, Item::ArrayOfTables(ArrayOfTables::new()));
    }
    match parent.get_mut(key)? {
        Item::ArrayOfTables(tables) => Some(ArraySlot::Tables(tables)),
        Item::Value(Value::Array(array)) => Some(ArraySlot::Inline(array)),
        _ => None,
    }
}

impl<'a> ArraySlot<'a> {
    fn len(&self) -> usize {
        match self {
            ArraySlot::Tables(tables) => tables.len(),
            ArraySlot::Inline(array) => array.len(),
        }
    }

    fn push_empty(&mut self) {
        match self {
            ArraySlot::Tables(tables) => tables.push(Table::new()),
            ArraySlot::Inline(array) => array.push(Value::InlineTable(InlineTable::new())),
        }
    }

    fn remove(&mut self, index: usize) {
        match self {
            ArraySlot::Tables(tables) => drop(tables.remove(index)),
            ArraySlot::Inline(array) => drop(array.remove(index)),
        }
    }

    fn table(&mut self, index: usize) -> Option<&mut dyn TableLike> {
        match self {
            ArraySlot::Tables(tables) => tables.get_mut(index).map(|t| t as &mut dyn TableLike),
            ArraySlot::Inline(array) => array
                .get_mut(index)
                .and_then(Value::as_inline_table_mut)
                .map(|t| t as &mut dyn TableLike),
        }
    }

    /// As [`ArraySlot::table`], but handing the borrow on to the caller so a
    /// nested list can be resolved inside the entry.
    fn into_table(self, index: usize) -> Option<&'a mut dyn TableLike> {
        match self {
            ArraySlot::Tables(tables) => tables.get_mut(index).map(|t| t as &mut dyn TableLike),
            ArraySlot::Inline(array) => array
                .get_mut(index)
                .and_then(Value::as_inline_table_mut)
                .map(|t| t as &mut dyn TableLike),
        }
    }
}

/// A float value with default formatting.
fn float_value(value: f64) -> Value {
    Value::from(value)
}

/// A float value in exponent notation.
///
/// `Value::from` spells an `f64` the way `Display` does, which is a decimal
/// point and ten zeros for a solver tolerance of 1e-10. The exponent form is
/// how the README, the panel and the solver's own messages all write these, so
/// it is how a save writes them too. A value the file already holds is left
/// exactly as the user spelled it: [`set_if_changed`] compares numbers, not
/// their spelling.
///
/// Parsed rather than constructed because `Formatted::set_repr_unchecked` is
/// private to `toml_edit`. Every finite `f64` formats to a valid TOML float
/// this way - `{:e}` always emits an exponent - so the fallback is unreachable
/// rather than a silent second spelling.
fn scientific_float_value(value: f64) -> Value {
    format!("{value:e}")
        .parse::<Value>()
        .unwrap_or_else(|_| float_value(value))
}

/// A three component array with default formatting.
fn vec3_value(value: Vec3) -> Value {
    let mut array = Array::new();
    for component in value {
        array.push(component);
    }
    array.fmt();
    Value::Array(array)
}

/// Write `value` unless the document already holds it, keeping everything
/// around it: the key's own decor, which is where the comment block above a
/// line lives, and the value's, which is its spacing and any trailing comment.
///
/// The item is replaced in place rather than re-inserted under the same name,
/// because inserting formats the key afresh and a re-formatted key is a
/// comment block deleted.
///
/// Returns true when the key was not there at all, which is what tells an
/// inline table that it needs reformatting.
fn set_if_changed(table: &mut dyn TableLike, key: &str, value: Value) -> bool {
    match table.get_mut(key) {
        Some(item) if item.is_value() => {
            let current = item.as_value().expect("the item is a value");
            if same_value(current, &value) {
                return false;
            }
            let mut value = value;
            *value.decor_mut() = current.decor().clone();
            *item = Item::Value(value);
            false
        }
        _ => {
            table.insert(key, Item::Value(value));
            true
        }
    }
}

/// Whether two values mean the same thing, ignoring how they are spelled.
fn same_value(current: &Value, new: &Value) -> bool {
    match (current, new) {
        (Value::String(a), Value::String(b)) => a.value() == b.value(),
        (Value::Boolean(a), Value::Boolean(b)) => a.value() == b.value(),
        (Value::Integer(a), Value::Integer(b)) => a.value() == b.value(),
        // An integer in a float field means the same number; the reverse is a
        // real change, because the file would have to be rewritten to hold it.
        (Value::Float(a), Value::Float(b)) => a.value() == b.value(),
        (Value::Integer(a), Value::Float(b)) => (*a.value() as f64) == *b.value(),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| same_value(x, y))
        }
        _ => false,
    }
}

fn set_str(table: &mut dyn TableLike, key: &str, value: &str) -> bool {
    set_if_changed(table, key, Value::from(value))
}

fn set_float(table: &mut dyn TableLike, key: &str, value: f64) -> bool {
    set_if_changed(table, key, float_value(value))
}

fn set_vec3(table: &mut dyn TableLike, key: &str, value: Vec3) -> bool {
    set_if_changed(table, key, vec3_value(value))
}

fn set_opt_str(table: &mut dyn TableLike, key: &str, value: Option<&str>) {
    match value {
        Some(value) => drop(set_str(table, key, value)),
        None => drop(table.remove(key)),
    }
}

fn set_opt_float(table: &mut dyn TableLike, key: &str, value: Option<f64>) {
    match value {
        Some(value) => drop(set_float(table, key, value)),
        None => drop(table.remove(key)),
    }
}

/// The same for the keys written in exponent notation.
fn set_opt_scientific_float(table: &mut dyn TableLike, key: &str, value: Option<f64>) {
    match value {
        Some(value) => drop(set_if_changed(table, key, scientific_float_value(value))),
        None => drop(table.remove(key)),
    }
}

fn set_opt_bool(table: &mut dyn TableLike, key: &str, value: Option<bool>) {
    match value {
        Some(value) => drop(set_if_changed(table, key, Value::from(value))),
        None => drop(table.remove(key)),
    }
}

/// A literal `toml_edit` cannot hold but `u64` can, with TOML's digit
/// separators removed.
fn big_literal(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    if !trimmed.bytes().all(|b| b.is_ascii_digit() || b == b'_') {
        return None;
    }
    let digits: String = trimmed.chars().filter(|c| *c != '_').collect();
    match digits.parse::<u64>() {
        Ok(value) if value > i64::MAX as u64 => Some(value),
        _ => None,
    }
}

/// A stand-in integer that neither the document nor another stand-in uses.
fn free_sentinel(text: &str, big: &[BigInteger]) -> Option<i64> {
    (0..constants::VIEW_EDIT_MAX_BIG_INTEGERS as i64 * 2)
        .map(|offset| i64::MAX - offset)
        .find(|candidate| {
            !big.iter().any(|entry| entry.sentinel == *candidate)
                && find_token(text, &candidate.to_string()).is_none()
        })
}

/// Offset of `token` in `text` where it is not part of a longer number or word.
fn find_token(text: &str, token: &str) -> Option<usize> {
    let bounded = |c: Option<char>| !c.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.');
    let mut from = 0;
    while let Some(offset) = text[from..].find(token) {
        let start = from + offset;
        let end = start + token.len();
        let before = text[..start].chars().next_back();
        let after = text[end..].chars().next();
        if bounded(before) && bounded(after) {
            return Some(start);
        }
        from = end;
    }
    None
}

/// Replace every whole-token occurrence of `token`.
fn replace_token(text: &str, token: &str, replacement: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(offset) = find_token(rest, token) {
        out.push_str(&rest[..offset]);
        out.push_str(replacement);
        rest = &rest[offset + token.len()..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::editor::tests::fixture;

    /// Parse a configuration and its document together, the way the editor
    /// holds them.
    fn open(text: &str) -> (Config, Document) {
        (
            Config::parse(text).expect("parse config"),
            Document::parse(text).expect("parse document"),
        )
    }

    #[test]
    fn a_document_nothing_changed_in_round_trips_byte_for_byte() {
        let text = fixture();
        let (config, mut document) = open(text);
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text, "an untouched save must be a no-op");
    }

    #[test]
    fn every_shipped_example_round_trips_byte_for_byte() {
        for name in [
            "cantilever.toml",
            "mbb_bridge.toml",
            "shelf_bracket.toml",
            "growth_canopy.toml",
            "growth_canopy_symmetric.toml",
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples")
                .join(name);
            let text = std::fs::read_to_string(&path).expect("read example");
            let (config, mut document) = open(&text);
            document.sync(&config).expect("sync");
            assert_eq!(document.render(), text, "{name} did not survive a save");
        }
    }

    /// The symmetry table is a sub-table of an optional table, and both of its
    /// kinds have keys the other rejects, so a save has to write exactly the
    /// ones the configuration holds and take the rest away with it.
    #[test]
    fn the_symmetry_table_appears_switches_kind_and_disappears_again() {
        use crate::config::{Axis, SymmetryConfig, SymmetryKind};
        use crate::viewer::editor::tests::growth_fixture;

        let text = format!("{}\n[growth]\nseed = 1\n", growth_fixture());
        let (mut config, mut document) = open(&text);
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text, "an untouched save must be a no-op");

        // Switched on in the panel: the table appears with its planes.
        config.growth.as_mut().expect("growth").symmetry = Some(SymmetryConfig {
            kind: SymmetryKind::Mirror,
            planes: Some(vec![Axis::X, Axis::Y]),
            order: None,
            axis: None,
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("[growth.symmetry]"), "{saved}");
        assert!(saved.contains("kind = \"mirror\""), "{saved}");
        assert!(saved.contains("planes = [\"x\", \"y\"]"), "{saved}");
        assert!(!saved.contains("order"), "{saved}");
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(
            reparsed
                .growth_params()
                .expect("resolve")
                .expect("growth params")
                .symmetry,
            Some(crate::config::SymmetryParams::Mirror {
                first: Axis::X,
                second: Some(Axis::Y)
            })
        );

        // Switched to a rotation: the mirror's key goes, the rotation's arrive.
        config.growth.as_mut().expect("growth").symmetry = Some(SymmetryConfig {
            kind: SymmetryKind::Rotational,
            planes: None,
            order: Some(5),
            axis: Some(Axis::Z),
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("kind = \"rotational\""), "{saved}");
        assert!(
            saved.contains("order = 5") && saved.contains("axis = \"z\""),
            "{saved}"
        );
        assert!(
            !saved.contains("planes"),
            "the mirror's planes outlived the mirror: {saved}"
        );
        Config::parse(&saved)
            .expect("reparse")
            .validate_static()
            .expect("what was written has to be readable");

        // And switched off, the table goes away and the file is what it was.
        config.growth.as_mut().expect("growth").symmetry = None;
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text, "the table outlived the symmetry");
    }

    /// The guide wireframe table is optional and so is every key in it, so a save
    /// has to write only the keys the configuration pins, and take the whole table
    /// away again when the checkbox goes off.
    #[test]
    fn the_wireframe_table_appears_with_its_pinned_keys_and_disappears_again() {
        use crate::config::WireframeConfig;

        let text = fixture();
        let (mut config, mut document) = open(text);
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text, "an untouched save must be a no-op");

        // Switched on with nothing pinned: the table appears and no key does.
        config.optimization.wireframe = Some(WireframeConfig {
            radius_mm: None,
            hold_iterations: None,
            seed_density: None,
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("[optimization.wireframe]"), "{saved}");
        for key in ["radius_mm", "hold_iterations", "seed_density"] {
            assert!(!saved.contains(key), "{key} was written unpinned: {saved}");
        }
        // And the file it was written into is otherwise untouched: the comment
        // rich lines around the table it was added to are still there.
        assert!(
            saved.contains("mass_fraction = 0.3  # keep this comment"),
            "adding the table rewrote its neighbours: {saved}"
        );
        assert!(
            Config::parse(&saved)
                .expect("reparse")
                .optimization_params()
                .expect("params")
                .wireframe
                .is_some()
        );

        // Keys pinned in the panel: each one arrives, and reads back as itself.
        config.optimization.wireframe = Some(WireframeConfig {
            radius_mm: Some(3.5),
            hold_iterations: Some(12),
            seed_density: Some(0.75),
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("radius_mm = 3.5"), "{saved}");
        assert!(saved.contains("hold_iterations = 12"), "{saved}");
        assert!(saved.contains("seed_density = 0.75"), "{saved}");
        let resolved = Config::parse(&saved)
            .expect("reparse")
            .optimization_params()
            .expect("params")
            .wireframe
            .expect("a wireframe");
        assert_eq!(resolved.radius_mm, 3.5);
        assert_eq!(resolved.hold_iterations, 12);
        assert_eq!(resolved.seed_density, 0.75);

        // A key unpinned again is removed rather than left behind.
        config.optimization.wireframe = Some(WireframeConfig {
            radius_mm: Some(3.5),
            hold_iterations: None,
            seed_density: None,
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("radius_mm = 3.5"), "{saved}");
        assert!(!saved.contains("hold_iterations"), "{saved}");
        assert!(!saved.contains("seed_density"), "{saved}");

        // And switched off, the table goes and the file is what it was.
        config.optimization.wireframe = None;
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text, "the table outlived the guide");
    }

    /// The local volume table on the same terms, beside the flat `penalty` key
    /// it shares its section with: both optional, both written only when pinned,
    /// and both gone again when they are not.
    #[test]
    fn the_local_volume_table_and_the_penalty_key_round_trip() {
        use crate::config::{LocalVolumeConfig, UpdateScheme};

        let text = fixture();
        let (mut config, mut document) = open(text);

        // The cap needs the moving asymptotes scheme, so the configuration that
        // carries it is the one that would really be saved.
        config.optimization.update = Some(UpdateScheme::Mma);
        config.optimization.penalty = Some(4.0);
        config.optimization.local_volume = Some(LocalVolumeConfig {
            max_fraction: None,
            radius_mm: None,
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("penalty = 4.0"), "{saved}");
        assert!(saved.contains("[optimization.local_volume]"), "{saved}");
        for key in ["max_fraction", "radius_mm"] {
            assert!(!saved.contains(key), "{key} was written unpinned: {saved}");
        }
        let reparsed = Config::parse(&saved).expect("reparse");
        reparsed
            .validate_static()
            .expect("the saved file has to run");
        let resolved = reparsed.optimization_params().expect("params");
        assert_eq!(resolved.penalty, 4.0);
        let cap = resolved.local_volume.expect("a cap");
        assert_eq!(cap.max_fraction, constants::LOCAL_VOLUME_MAX_FRACTION);

        // Pinned keys arrive and read back as themselves.
        config.optimization.local_volume = Some(LocalVolumeConfig {
            max_fraction: Some(0.55),
            radius_mm: Some(9.0),
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("max_fraction = 0.55"), "{saved}");
        assert!(saved.contains("radius_mm = 9.0"), "{saved}");
        let cap = Config::parse(&saved)
            .expect("reparse")
            .optimization_params()
            .expect("params")
            .local_volume
            .expect("a cap");
        assert_eq!(cap.max_fraction, 0.55);
        assert_eq!(cap.radius_mm, 9.0);

        // And unset, all three go: the file is byte for byte what it was.
        config.optimization.local_volume = None;
        config.optimization.penalty = None;
        config.optimization.update = None;
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text, "the table outlived the cap");
    }

    /// The reduction table: one required key written, the optional ones only
    /// when they are pinned, and the whole table out of the file again when the
    /// configuration no longer has it.
    #[test]
    fn the_reduce_table_round_trips_with_the_keys_that_were_pinned() {
        use crate::config::{ReduceConfig, ReduceMethod, ReduceMethodParams};

        let text = fixture();
        let (mut config, mut document) = open(text);

        config.optimization.reduce = Some(ReduceConfig {
            target_safety_factor: 2.5,
            method: None,
            ratio: None,
            refine_stages: None,
            min_mass_fraction: None,
            evolution_rate: None,
            add_ratio: None,
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("[optimization.reduce]"), "{saved}");
        assert!(saved.contains("target_safety_factor = 2.5"), "{saved}");
        // "ratio" on its own is a substring of max_iterations, so it is asked
        // for the way it would be written.
        for key in [
            "method",
            "\nratio",
            "refine_stages",
            "min_mass_fraction",
            "evolution_rate",
            "add_ratio",
        ] {
            assert!(!saved.contains(key), "{key} was written unpinned: {saved}");
        }
        let reparsed = Config::parse(&saved).expect("reparse");
        reparsed
            .validate_static()
            .expect("the saved file has to run");
        let reduce = reparsed
            .optimization_params()
            .expect("params")
            .reduce
            .expect("a reduction");
        assert_eq!(reduce.target_safety_factor, 2.5);
        assert_eq!(reduce.method, ReduceMethodParams::Continuation);
        // The mass fraction the fixture already carries is where it starts.
        assert_eq!(
            reparsed
                .optimization_params()
                .expect("params")
                .mass_fraction,
            0.3
        );

        // Pinned keys, the two evolutionary ones included, arrive and read back
        // as themselves.
        config.optimization.reduce = Some(ReduceConfig {
            target_safety_factor: 4.25,
            method: Some(ReduceMethod::Beso),
            ratio: Some(0.9),
            refine_stages: Some(3),
            min_mass_fraction: Some(0.05),
            evolution_rate: Some(0.03),
            add_ratio: Some(0.0),
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        // Every key spelled out in the file, which is what the panel's rows
        // pin one at a time: a key that reads back through a default rather
        // than off the line it wrote would round-trip and still be missing.
        for line in [
            "target_safety_factor = 4.25",
            "method = \"beso\"",
            "ratio = 0.9",
            "refine_stages = 3",
            "min_mass_fraction = 0.05",
            "evolution_rate = 0.03",
            "add_ratio = 0.0",
        ] {
            assert!(saved.contains(line), "{line} is not in the file: {saved}");
        }
        let reduce = Config::parse(&saved)
            .expect("reparse")
            .optimization_params()
            .expect("params")
            .reduce
            .expect("a reduction");
        assert_eq!(reduce.target_safety_factor, 4.25);
        assert_eq!(reduce.ratio, 0.9);
        assert_eq!(reduce.refine_stages, 3);
        assert_eq!(reduce.min_mass_fraction, 0.05);
        assert_eq!(
            reduce.method,
            ReduceMethodParams::Beso {
                evolution_rate: 0.03,
                add_ratio: 0.0,
            }
        );

        // The other method, written out rather than left to the default, and
        // the two evolutionary keys gone with it - which is the state the
        // panel's method combo leaves behind.
        config.optimization.reduce = Some(ReduceConfig {
            method: Some(ReduceMethod::Continuation),
            evolution_rate: None,
            add_ratio: None,
            ..config.optimization.reduce.clone().expect("the table")
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("method = \"continuation\""), "{saved}");
        for key in ["evolution_rate", "add_ratio"] {
            assert!(
                !saved.contains(key),
                "{key} outlived the method that owns it: {saved}"
            );
        }
        assert_eq!(
            Config::parse(&saved)
                .expect("reparse")
                .optimization_params()
                .expect("params")
                .reduce
                .expect("a reduction")
                .method,
            ReduceMethodParams::Continuation
        );

        // And unset, the whole table goes: the file is what it was.
        config.optimization.reduce = None;
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text, "the table outlived the reduction");
    }

    /// `[optimization] stiffness_floor` is written in exponent notation, and
    /// what is read back is the number that was held.
    ///
    /// The shared float writer spells an `f64` the way `Display` does, which for
    /// this key is a decimal point and eight zeros at its default and a decimal
    /// point and eleven at its lower bound - a file nobody would recognise and,
    /// re-read through a decimal-rounding widget, a floor of zero. The key
    /// therefore takes the same writer `[solver] tolerance` does.
    #[test]
    fn the_stiffness_floor_round_trips_in_exponent_notation() {
        let text = fixture();
        let (mut config, mut document) = open(text);

        for floor in [
            1e-6,
            constants::STIFFNESS_FLOOR_MIN,
            constants::STIFFNESS_FLOOR_MAX,
        ] {
            config.optimization.stiffness_floor = Some(floor);
            document.sync(&config).expect("sync");
            let saved = document.render();
            assert!(
                saved.contains(&format!("stiffness_floor = {floor:e}")),
                "{floor:e} was not written as an exponent: {saved}"
            );
            let reparsed = Config::parse(&saved).expect("reparse");
            reparsed
                .validate_static()
                .expect("the saved file has to run");
            assert_eq!(
                reparsed
                    .optimization_params()
                    .expect("params")
                    .stiffness_floor,
                floor
            );
        }

        // Unpinned again the key goes, the resolution is the constant once more,
        // and the file is byte for byte what it was.
        config.optimization.stiffness_floor = None;
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(!saved.contains("stiffness_floor"), "{saved}");
        assert_eq!(
            Config::parse(&saved)
                .expect("reparse")
                .optimization_params()
                .expect("params")
                .stiffness_floor,
            constants::SIMP_EMIN_FRACTION
        );
        assert_eq!(document.render(), text, "the key outlived the floor");
    }

    /// Both keys of the `[solver]` table, together and apart.
    ///
    /// The table is written whole from one struct, so the risk is a key
    /// disappearing under its neighbour: what is saved has to be exactly what
    /// the editor holds, and has to read back as itself.
    #[test]
    fn the_solver_table_round_trips_both_of_its_keys() {
        use crate::config::{SolverBackend, SolverConfig};

        let text = fixture();
        let (mut config, mut document) = open(text);
        assert!(config.solver.is_none());

        // The tolerance alone: the table appears with one key in it.
        config.solver = Some(SolverConfig {
            backend: None,
            tolerance: Some(3e-8),
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("[solver]"), "{saved}");
        assert!(saved.contains("tolerance = 3e-8"), "{saved}");
        assert!(!saved.contains("backend"), "{saved}");
        let reparsed = Config::parse(&saved).expect("reparse");
        reparsed
            .validate_static()
            .expect("the saved file has to run");
        assert_eq!(reparsed.solver_params().expect("resolve").tolerance, 3e-8);

        // And beside a backend: neither key writes the other's.
        config.solver = Some(SolverConfig {
            backend: Some(SolverBackend::Cpu),
            tolerance: Some(3e-8),
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("backend = \"cpu\""), "{saved}");
        assert!(saved.contains("tolerance = 3e-8"), "{saved}");
        let resolved = Config::parse(&saved)
            .expect("reparse")
            .solver_params()
            .expect("resolve");
        assert_eq!(resolved.backend, SolverBackend::Cpu);
        assert_eq!(resolved.tolerance, 3e-8);

        // The tolerance unpinned again is removed, and the backend stays.
        config.solver = Some(SolverConfig {
            backend: Some(SolverBackend::Cpu),
            tolerance: None,
        });
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("backend = \"cpu\""), "{saved}");
        assert!(!saved.contains("tolerance"), "{saved}");
        assert_eq!(
            Config::parse(&saved)
                .expect("reparse")
                .solver_params()
                .expect("resolve")
                .tolerance,
            constants::CG_RELATIVE_TOLERANCE
        );

        // And the whole table off, the file is byte for byte what it was.
        config.solver = None;
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text, "the table outlived the keys");
    }

    #[test]
    fn editing_one_value_leaves_every_other_byte_alone() {
        let text = fixture();
        let (mut config, mut document) = open(text);
        config.optimization.mass_fraction = Some(0.42);
        document.sync(&config).expect("sync");
        let saved = document.render();

        // The edited line carries the new number and keeps its own comment.
        assert!(saved.contains("mass_fraction = 0.42"), "{saved}");
        assert!(
            saved.contains("mass_fraction = 0.42  # keep this comment"),
            "the trailing comment of an edited value must survive: {saved}"
        );
        assert!(
            saved.contains("# belongs to the key, not to the value"),
            "the comment block above an edited key must survive: {saved}"
        );
        // And everything else is byte identical.
        let expected = text.replace("mass_fraction = 0.3", "mass_fraction = 0.42");
        assert_eq!(saved, expected);
        // What it reparses to is what the editor was holding.
        let reparsed = Config::parse(&saved).expect("reparse");
        assert!((reparsed.optimization.mass_fraction.expect("a mass target") - 0.42).abs() < 1e-12);
    }

    /// The rotation key is optional, and an editor may not add noise to a file:
    /// a box nobody turned comes back without one, and a box somebody did comes
    /// back with a clean array that reparses to what the editor was holding.
    #[test]
    fn a_rotation_appears_only_once_a_box_has_been_turned() {
        let text = fixture();
        let (mut config, mut document) = open(text);
        // Every box in the fixture starts unrotated, and a save leaves the file
        // byte for byte as it was - the key never appears.
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text);
        assert!(!document.render().contains("rotation_deg"));

        // Turn the keepin box, the way a rotation arc does.
        let ShapeSpec::Box { rotation_deg, .. } = &mut config.keepin[0] else {
            panic!("the fixture's keepin is a box");
        };
        *rotation_deg = Some([0.0, 0.0, 45.0]);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(
            saved.contains("rotation_deg = [0.0, 0.0, 45.0]"),
            "the turn did not reach the file: {saved}"
        );
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(
            reparsed.keepin[0].rotation_deg(),
            Some([0.0, 0.0, 45.0]),
            "the file did not come back as what was written"
        );
        // And nothing else moved: the comment rich fixture is untouched.
        assert_eq!(
            saved,
            text.replace(
                "min = [56.0, 8.0, 8.0]\nmax = [64.0, 24.0, 24.0]",
                "min = [56.0, 8.0, 8.0]\nmax = [64.0, 24.0, 24.0]\nrotation_deg = [0.0, 0.0, 45.0]"
            )
        );
        assert!(saved.contains("# keep this comment"), "{saved}");
        assert!(saved.contains("# the first keepout"), "{saved}");

        // Turned back to nothing, the key goes away again with it.
        let ShapeSpec::Box { rotation_deg, .. } = &mut config.keepin[0] else {
            panic!("a box");
        };
        *rotation_deg = None;
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text, "the key outlived the rotation");

        // A rotation of exactly zero is a key the user asked for, so it is
        // written: absent and [0, 0, 0] are the same geometry, and the file
        // says which one the user chose.
        let ShapeSpec::Box { rotation_deg, .. } = &mut config.keepin[0] else {
            panic!("a box");
        };
        *rotation_deg = Some([0.0; 3]);
        document.sync(&config).expect("sync");
        assert!(document.render().contains("rotation_deg = [0.0, 0.0, 0.0]"));
    }

    /// An ellipsoid writes its own keys and no others, its rotation stays
    /// optional the way a box's does, and the rest of a comment rich file is
    /// untouched by any of it.
    #[test]
    fn an_ellipsoid_round_trips_with_its_rotation_optional() {
        let text = fixture();
        let (mut config, mut document) = open(text);

        // The fixture's second keepout, a sphere, becomes an ellipsoid.
        config.keepout[1] = ShapeSpec::Ellipsoid {
            center: [50.0, 16.0, 16.0],
            radii: [8.0, 4.0, 4.0],
            rotation_deg: None,
        };
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("shape = \"ellipsoid\""), "{saved}");
        assert!(saved.contains("radii = [8.0, 4.0, 4.0]"), "{saved}");
        assert!(
            !saved.contains("rotation_deg"),
            "an ellipsoid nobody turned may not grow the key: {saved}"
        );
        assert!(
            !saved.contains("radius = 4.0"),
            "the sphere's own key outlived it: {saved}"
        );
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(reparsed.keepout[1], config.keepout[1]);
        // The comments of everything around it survived.
        assert!(saved.contains("# the second keepout"), "{saved}");
        assert!(saved.contains("# keep this comment"), "{saved}");

        // Turned, the key appears; turned back to nothing, it goes away again.
        let ShapeSpec::Ellipsoid { rotation_deg, .. } = &mut config.keepout[1] else {
            panic!("an ellipsoid");
        };
        *rotation_deg = Some([0.0, 0.0, 45.0]);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("rotation_deg = [0.0, 0.0, 45.0]"), "{saved}");
        assert_eq!(
            Config::parse(&saved).expect("reparse").keepout[1].rotation_deg(),
            Some([0.0, 0.0, 45.0])
        );
        let ShapeSpec::Ellipsoid { rotation_deg, .. } = &mut config.keepout[1] else {
            panic!("an ellipsoid");
        };
        *rotation_deg = None;
        document.sync(&config).expect("sync");
        assert!(!document.render().contains("rotation_deg"));

        // And back to the sphere it started as: the file is what it was, byte
        // for byte, `radii` and all gone with it.
        config.keepout[1] = ShapeSpec::Sphere {
            center: [50.0, 16.0, 16.0],
            radius: 4.0,
        };
        document.sync(&config).expect("sync");
        assert_eq!(
            document.render(),
            text,
            "the ellipsoid's keys outlived the ellipsoid"
        );
    }

    /// A tube writes its own keys and no others, its bend comes and goes with
    /// the key, and straightening one takes that key away again.
    #[test]
    fn a_tube_round_trips_with_its_bend_optional() {
        let text = fixture();
        let (mut config, mut document) = open(text);

        // The fixture's second keepout, a sphere, becomes a straight tube.
        config.keepout[1] = ShapeSpec::Tube {
            p1: [46.0, 16.0, 16.0],
            p2: [54.0, 16.0, 16.0],
            bend: None,
            radius: 3.0,
        };
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("shape = \"tube\""), "{saved}");
        assert!(saved.contains("p1 = [46.0, 16.0, 16.0]"), "{saved}");
        assert!(saved.contains("radius = 3.0"), "{saved}");
        assert!(
            !saved.contains("bend"),
            "a tube nobody bent may not grow the key: {saved}"
        );
        assert!(
            !saved.contains("center = [50.0, 16.0, 16.0]"),
            "the sphere's own key outlived it: {saved}"
        );
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(reparsed.keepout[1], config.keepout[1]);
        assert!(saved.contains("# the second keepout"), "{saved}");

        // Bent, the key appears and comes back as what was written.
        let ShapeSpec::Tube { bend, .. } = &mut config.keepout[1] else {
            panic!("a tube");
        };
        *bend = Some([50.0, 16.0, 22.0]);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("bend = [50.0, 16.0, 22.0]"), "{saved}");
        assert_eq!(
            Config::parse(&saved).expect("reparse").keepout[1],
            config.keepout[1]
        );

        // Straightened - which is what the gizmo and the panel both do by
        // clearing the bend - the key goes with it.
        let ShapeSpec::Tube { bend, .. } = &mut config.keepout[1] else {
            panic!("a tube");
        };
        *bend = None;
        document.sync(&config).expect("sync");
        assert!(
            !document.render().contains("bend"),
            "the key outlived the bend: {}",
            document.render()
        );

        // And back to the sphere it started as: every key only a tube has goes
        // with it, and the file still parses as the shape it now is. The
        // *order* of the keys it kept is the file's own - `radius` is a key
        // both kinds spell, so it stays where the tube left it and `center`
        // comes back after it - which is why this is the shape rather than the
        // bytes.
        config.keepout[1] = ShapeSpec::Sphere {
            center: [50.0, 16.0, 16.0],
            radius: 4.0,
        };
        document.sync(&config).expect("sync");
        let saved = document.render();
        // The fixture's other keepout is a cylinder, so it is this tube's own
        // points that must be gone rather than the keys' spellings.
        for key in ["p1 = [46.0", "p2 = [54.0", "bend"] {
            assert!(
                !saved.contains(key),
                "the tube's {key} outlived the tube: {saved}"
            );
        }
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(reparsed.keepout[1], config.keepout[1]);
        assert_eq!(reparsed.keepout[0], config.keepout[0], "and its neighbour");
        assert!(saved.contains("# the second keepout"), "{saved}");
        assert!(saved.contains("# keep this comment"), "{saved}");
    }

    /// A cone writes its own four keys and no others, its apex is written as
    /// the zero it is, and changing kind takes every one of them away again.
    #[test]
    fn a_cone_round_trips_with_both_of_its_radii() {
        let text = fixture();
        let (mut config, mut document) = open(text);

        // The fixture's second keepout, a sphere, becomes a frustum.
        config.keepout[1] = ShapeSpec::Cone {
            p1: [46.0, 16.0, 16.0],
            p2: [54.0, 16.0, 16.0],
            radius1: 4.0,
            radius2: 2.0,
        };
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("shape = \"cone\""), "{saved}");
        assert!(saved.contains("p1 = [46.0, 16.0, 16.0]"), "{saved}");
        assert!(saved.contains("radius1 = 4.0"), "{saved}");
        assert!(saved.contains("radius2 = 2.0"), "{saved}");
        assert!(
            !saved.contains("center = [50.0, 16.0, 16.0]"),
            "the sphere's own key outlived it: {saved}"
        );
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(reparsed.keepout[1], config.keepout[1]);
        assert!(saved.contains("# the second keepout"), "{saved}");

        // Dragged to its apex, the second radius is written as zero rather
        // than dropped: it is a value the shape has.
        let ShapeSpec::Cone { radius2, .. } = &mut config.keepout[1] else {
            panic!("a cone");
        };
        *radius2 = 0.0;
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("radius2 = 0.0"), "{saved}");
        assert_eq!(
            Config::parse(&saved).expect("reparse").keepout[1],
            config.keepout[1]
        );

        // And back to the sphere it started as: every key only a cone has goes
        // with it, and the file still parses as the shape it now is.
        config.keepout[1] = ShapeSpec::Sphere {
            center: [50.0, 16.0, 16.0],
            radius: 4.0,
        };
        document.sync(&config).expect("sync");
        let saved = document.render();
        for key in ["p1 = [46.0", "p2 = [54.0", "radius1", "radius2"] {
            assert!(
                !saved.contains(key),
                "the cone's {key} outlived the cone: {saved}"
            );
        }
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(reparsed.keepout[1], config.keepout[1]);
        assert_eq!(reparsed.keepout[0], config.keepout[0], "and its neighbour");
        assert!(saved.contains("# keep this comment"), "{saved}");
    }

    /// A triangle writes its three vertices and its thickness and no other
    /// key, and changing kind takes all four away again.
    #[test]
    fn a_triangle_round_trips_with_its_three_vertices() {
        let text = fixture();
        let (mut config, mut document) = open(text);

        config.keepout[1] = ShapeSpec::Triangle {
            a: [46.0, 12.0, 16.0],
            b: [54.0, 12.0, 16.0],
            c: [50.0, 20.0, 16.0],
            thickness: 3.0,
        };
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("shape = \"triangle\""), "{saved}");
        for key in [
            "a = [46.0, 12.0, 16.0]",
            "b = [54.0, 12.0, 16.0]",
            "c = [50.0, 20.0, 16.0]",
            "thickness = 3.0",
        ] {
            assert!(saved.contains(key), "{key} is missing from {saved}");
        }
        assert!(
            !saved.contains("center = [50.0, 16.0, 16.0]"),
            "the sphere's own key outlived it: {saved}"
        );
        assert!(
            !saved.contains("rotation_deg"),
            "a triangle has no rotation to write: {saved}"
        );
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(reparsed.keepout[1], config.keepout[1]);
        assert!(saved.contains("# the second keepout"), "{saved}");

        // A vertex dragged in the viewport reaches the file as itself.
        let ShapeSpec::Triangle { c, .. } = &mut config.keepout[1] else {
            panic!("a triangle");
        };
        *c = [50.0, 24.0, 18.0];
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("c = [50.0, 24.0, 18.0]"), "{saved}");
        assert_eq!(
            Config::parse(&saved).expect("reparse").keepout[1],
            config.keepout[1]
        );

        // And back to the sphere: the four keys go with the triangle.
        config.keepout[1] = ShapeSpec::Sphere {
            center: [50.0, 16.0, 16.0],
            radius: 4.0,
        };
        document.sync(&config).expect("sync");
        let saved = document.render();
        for key in ["a = [", "b = [", "c = [", "thickness"] {
            assert!(
                !saved.contains(key),
                "the triangle's {key} outlived the triangle: {saved}"
            );
        }
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(reparsed.keepout[1], config.keepout[1]);
        assert_eq!(reparsed.keepout[0], config.keepout[0], "and its neighbour");
        assert!(saved.contains("# keep this comment"), "{saved}");
    }

    /// A box that changes kind must not leave its rotation behind on a shape
    /// that has no such key.
    #[test]
    fn a_shape_that_stops_being_a_box_drops_its_rotation() {
        let text = fixture();
        let (mut config, mut document) = open(text);
        let ShapeSpec::Box { rotation_deg, .. } = &mut config.keepin[0] else {
            panic!("a box");
        };
        *rotation_deg = Some([10.0, 20.0, 30.0]);
        document.sync(&config).expect("sync");
        assert!(document.render().contains("rotation_deg"));

        config.keepin[0] = ShapeSpec::Sphere {
            center: [60.0, 16.0, 16.0],
            radius: 4.0,
        };
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(
            !saved.contains("rotation_deg"),
            "a sphere kept the box's rotation: {saved}"
        );
        Config::parse(&saved).expect("the save must still be readable");
    }

    #[test]
    fn an_added_object_appends_valid_toml_that_reparses_to_it() {
        let text = fixture();
        let (mut config, mut document) = open(text);
        let added = ShapeSpec::Cylinder {
            p1: [1.0, 2.0, 3.0],
            p2: [1.0, 2.0, 9.0],
            radius: 2.5,
        };
        config.keepout.push(added.clone());
        document.push_object(List::Keepout);
        document.sync(&config).expect("sync");
        let saved = document.render();

        let reparsed = Config::parse(&saved).expect("the addition must be valid TOML");
        assert_eq!(reparsed.keepout.len(), config.keepout.len());
        // It is appended to its own section rather than to the end of the
        // file: the new table lands after the last keepout and before whatever
        // section came next.
        let appended = saved.find("radius = 2.5").expect("the new object");
        let next_section = saved.find("[[keepin]]").expect("the next section");
        assert!(
            appended < next_section,
            "the new object was not appended to its own section: {saved}"
        );
        let ShapeSpec::Cylinder { p1, p2, radius } = &reparsed.keepout[reparsed.keepout.len() - 1]
        else {
            panic!("the appended object came back as another shape: {saved}");
        };
        assert_eq!(*p1, [1.0, 2.0, 3.0]);
        assert_eq!(*p2, [1.0, 2.0, 9.0]);
        assert!((*radius - 2.5).abs() < 1e-12);
        // The comments of everything that was already there are untouched.
        assert!(saved.contains("# the design space"), "{saved}");
        assert!(saved.contains("# keep this comment"), "{saved}");
    }

    #[test]
    fn a_deleted_object_takes_its_own_table_and_comments_with_it() {
        let text = fixture();
        let (mut config, mut document) = open(text);
        assert_eq!(config.keepout.len(), 2);
        // Delete the first of the two, which is the one carrying a comment.
        config.keepout.remove(0);
        document.remove_object(List::Keepout, 0);
        document.sync(&config).expect("sync");
        let saved = document.render();

        assert!(
            !saved.contains("# the first keepout"),
            "the deleted object's comment stayed behind: {saved}"
        );
        assert!(
            saved.contains("# the second keepout"),
            "the surviving object lost its comment: {saved}"
        );
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(reparsed.keepout.len(), 1);
        let ShapeSpec::Sphere { radius, .. } = &reparsed.keepout[0] else {
            panic!("the wrong object survived: {saved}");
        };
        assert!((*radius - 4.0).abs() < 1e-12);
    }

    #[test]
    fn a_seed_beyond_the_signed_range_survives_a_round_trip_and_can_be_changed() {
        let text = format!("{}\n[growth]\nseed = 11396317718348371989\n", fixture());
        let (mut config, mut document) = open(&text);
        assert_eq!(
            config.growth.as_ref().and_then(|g| g.seed),
            Some(11396317718348371989)
        );
        document.sync(&config).expect("sync");
        assert_eq!(
            document.render(),
            text,
            "an untouched huge seed must not move"
        );

        // Changed to another value past i64::MAX, then to a small one.
        config.growth.as_mut().expect("growth").seed = Some(u64::MAX);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains(&u64::MAX.to_string()), "{saved}");
        assert_eq!(
            Config::parse(&saved)
                .expect("reparse")
                .growth
                .and_then(|g| g.seed),
            Some(u64::MAX)
        );

        config.growth.as_mut().expect("growth").seed = Some(7);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("seed = 7"), "{saved}");
        assert_eq!(
            Config::parse(&saved)
                .expect("reparse")
                .growth
                .and_then(|g| g.seed),
            Some(7)
        );
    }

    #[test]
    fn a_windows_authored_file_keeps_its_line_endings() {
        // The format preserving parser renders every newline as a bare line
        // feed, so a file written on Windows would come back with every one of
        // its lines rewritten - a diff of the whole file for a save of one
        // value.
        let text = fixture().replace('\n', "\r\n");
        let (mut config, mut document) = open(&text);
        document.sync(&config).expect("sync");
        assert_eq!(document.render(), text);

        config.optimization.mass_fraction = Some(0.42);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert_eq!(saved.matches("\r\n").count(), text.matches("\r\n").count());
        assert!(!saved.contains("\r\r"), "a carriage return was doubled");
        assert_eq!(
            saved,
            text.replace("mass_fraction = 0.3", "mass_fraction = 0.42")
        );
    }

    /// Every unsigned key in the schema can legally hold a number larger than
    /// the document's signed integers, not just `[growth] seed`: a `usize` is
    /// 64 bits wide. Writing one back as its wrapped negative would produce a
    /// file growforge itself refuses to read - and `sync` runs on every save,
    /// so it would happen to a field nobody touched.
    #[test]
    fn a_count_beyond_the_signed_range_survives_a_save_of_a_file_nobody_edited() {
        let huge: u64 = 12345678901234567890;
        for (key, replaced) in [
            ("target_cells", "voxel_size_mm = 4.0"),
            ("max_iterations", "max_iterations = 4"),
            ("smoothing_iterations", "iso_level = 0.5"),
        ] {
            let text = fixture().replace(replaced, &format!("{key} = {huge}"));
            let (config, mut document) = open(&text);
            document.sync(&config).expect("sync");
            let saved = document.render();
            assert_eq!(saved, text, "{key}: an untouched save rewrote the file");

            // And what came back is still a configuration growforge can read,
            // holding the number the file always held.
            let reparsed = Config::parse(&saved).unwrap_or_else(|e| panic!("{key}: {e}"));
            let held = match key {
                "target_cells" => reparsed.resolution.target_cells,
                "max_iterations" => reparsed.optimization.max_iterations,
                _ => reparsed.output.smoothing_iterations,
            };
            assert_eq!(held, Some(huge as usize), "{key} came back wrong");
            assert!(saved.contains(&huge.to_string()), "{key}: {saved}");
            let wrapped = (huge as i64).to_string();
            assert!(
                !saved.contains(&wrapped),
                "{key}: written as its wrapped self {wrapped}"
            );
        }
    }

    #[test]
    fn a_count_edited_past_the_signed_range_is_written_as_the_number_it_is() {
        let text = fixture().replace("iso_level = 0.5", "supersample = 2");
        let (mut config, mut document) = open(&text);
        config.output.supersample = Some(usize::MAX);
        config.optimization.max_iterations = Some(u64::MAX as usize - 1);
        document.sync(&config).expect("sync");
        let saved = document.render();

        let reparsed = Config::parse(&saved).expect("the save must still be readable");
        assert_eq!(reparsed.output.supersample, Some(usize::MAX));
        assert_eq!(
            reparsed.optimization.max_iterations,
            Some(u64::MAX as usize - 1)
        );
        // Back down to an ordinary number, and the file says an ordinary
        // number.
        config.output.supersample = Some(2);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("supersample = 2"), "{saved}");
        assert_eq!(
            Config::parse(&saved).expect("reparse").output.supersample,
            Some(2)
        );
    }

    /// A session of edits, each to another number past the signed range. The
    /// stand-ins the document no longer holds have to be given back, or the
    /// pool runs out and an edit silently stops reaching the file.
    #[test]
    fn a_session_of_huge_edits_keeps_reaching_the_file() {
        let text = format!("{}\n[growth]\nseed = 1\n", fixture());
        let (mut config, mut document) = open(&text);
        for step in 0..20_u64 {
            let seed = u64::MAX - step;
            config.growth.as_mut().expect("growth").seed = Some(seed);
            document.sync(&config).expect("sync");
            let saved = document.render();
            let reparsed = Config::parse(&saved)
                .unwrap_or_else(|e| panic!("edit {step} produced an unreadable file: {e}"));
            assert_eq!(
                reparsed.growth.and_then(|g| g.seed),
                Some(seed),
                "edit {step} did not reach the file: {saved}"
            );
            // One live value means one stand-in, however many have been used.
            assert_eq!(document.big.len(), 1, "the pool grew at edit {step}");
        }

        // And a key that goes back below the range gives its stand-in up.
        config.growth.as_mut().expect("growth").seed = Some(5);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("seed = 5"), "{saved}");
        assert!(document.big.is_empty(), "a stand-in outlived its value");
    }

    #[test]
    fn the_pool_keeps_the_stand_ins_the_document_holds_and_drops_the_rest() {
        let text = format!("{}\n[growth]\nseed = 11396317718348371989\n", fixture());
        let (mut config, mut document) = open(&text);
        assert_eq!(document.big.len(), 1, "the seed is carried");

        // A stale entry, of the kind an edit from one huge value to another
        // used to leave behind for good.
        document.big.push(BigInteger {
            sentinel: i64::MAX - 7,
            raw: "18446744073709551000".to_string(),
            value: 18446744073709551000,
        });
        assert_eq!(document.big.len(), 2);
        document.sync(&config).expect("sync");
        assert_eq!(document.big.len(), 1, "the stale entry survived");
        // The live one is untouched, and still spells what the file spells.
        assert_eq!(document.render(), text);

        // Dropping the key drops its stand-in with it.
        config.growth = None;
        document.sync(&config).expect("sync");
        assert!(document.big.is_empty());
    }

    /// The defense behind the pruning: if the pool ever did run out, the save
    /// has to fail loudly rather than leave the old number in the file and
    /// report success.
    #[test]
    fn a_value_that_cannot_be_written_fails_the_save_instead_of_going_missing() {
        // A file that already spells every stand-in the editor can choose, in a
        // comment. Contrived, and exactly the condition the guard is for.
        let spelled: Vec<String> = (0..constants::VIEW_EDIT_MAX_BIG_INTEGERS as i64 * 2)
            .map(|offset| (i64::MAX - offset).to_string())
            .collect();
        let text = format!(
            "# reserved: {}\n{}\n[growth]\nseed = 1\n",
            spelled.join(" "),
            fixture()
        );
        let (mut config, mut document) = open(&text);
        config.growth.as_mut().expect("growth").seed = Some(u64::MAX);

        let error = document
            .sync(&config)
            .expect_err("a value that cannot be written must fail the save")
            .to_string();
        assert!(
            error.contains("seed"),
            "the error must name the key: {error}"
        );
        assert!(error.contains(&i64::MAX.to_string()), "unexpected: {error}");
        // And the file is untouched: the old value is still the old value, and
        // it was never reported as saved.
        assert_eq!(document.render(), text);
    }

    #[test]
    fn an_optional_key_that_was_dropped_is_removed_from_the_file() {
        let text = fixture();
        let (mut config, mut document) = open(text);
        assert!(config.output.iso_level.is_some());
        config.output.iso_level = None;
        config.optimization.max_iterations = None;
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(!saved.contains("iso_level"), "{saved}");
        assert!(!saved.contains("max_iterations"), "{saved}");
        Config::parse(&saved).expect("still valid TOML");
    }

    /// The island policy is a key like any other: absent until it is chosen,
    /// written with the spelling the parser accepts, and gone again when it is
    /// cleared.
    #[test]
    fn the_island_policy_round_trips_through_a_save() {
        let (mut config, mut document) = open(fixture());
        assert_eq!(config.output.islands, None);

        config.output.islands = Some(crate::config::IslandPolicy::Keep);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("islands = \"keep\""), "{saved}");
        assert_eq!(
            Config::parse(&saved).expect("reparse").output.islands,
            Some(crate::config::IslandPolicy::Keep)
        );

        config.output.islands = None;
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(!saved.contains("islands"), "{saved}");
        Config::parse(&saved).expect("still valid TOML");
    }

    /// The same for the boundary fidelity, whose default is the *on* value: a
    /// configuration that never mentions it clamps, so the key has to stay
    /// absent until the editor is asked for one - including when the value it
    /// is set to is the default.
    #[test]
    fn the_boundary_fidelity_round_trips_through_a_save() {
        let (mut config, mut document) = open(fixture());
        assert_eq!(config.output.boundaries, None);

        config.output.boundaries = Some(crate::config::BoundaryFidelity::Voxel);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("boundaries = \"voxel\""), "{saved}");
        assert_eq!(
            Config::parse(&saved).expect("reparse").output.boundaries,
            Some(crate::config::BoundaryFidelity::Voxel)
        );

        config.output.boundaries = Some(crate::config::BoundaryFidelity::Exact);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("boundaries = \"exact\""), "{saved}");

        config.output.boundaries = None;
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(!saved.contains("boundaries"), "{saved}");
        // And an absent key is still the clamp: the default is not "off".
        let reparsed = Config::parse(&saved).expect("still valid TOML");
        assert_eq!(reparsed.output.boundaries, None);
        assert_eq!(
            reparsed
                .output_params(std::path::Path::new("."))
                .expect("resolve")
                .boundaries,
            crate::config::BoundaryFidelity::Exact
        );
    }

    /// The same for the trim pass, which is two keys rather than one: the
    /// policy and the threshold it judges by, each absent until it is set.
    #[test]
    fn the_trim_policy_and_its_fraction_round_trip_through_a_save() {
        let (mut config, mut document) = open(fixture());
        assert_eq!(config.output.trim, None);
        assert_eq!(config.output.trim_stress_fraction, None);

        config.output.trim = Some(crate::config::TrimPolicy::Stress);
        config.output.trim_stress_fraction = Some(0.02);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("trim = \"stress\""), "{saved}");
        assert!(saved.contains("trim_stress_fraction = 0.02"), "{saved}");
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(
            reparsed.output.trim,
            Some(crate::config::TrimPolicy::Stress)
        );
        assert_eq!(reparsed.output.trim_stress_fraction, Some(0.02));

        config.output.trim = None;
        config.output.trim_stress_fraction = None;
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(!saved.contains("trim"), "{saved}");
        Config::parse(&saved).expect("still valid TOML");
    }

    /// And the same for the reinforcement pass, whose two keys behave the same
    /// way: an absent thickness is not a written one, and what it resolves to
    /// comes from `[optimization]` rather than from this table.
    #[test]
    fn the_reinforce_policy_and_its_thickness_round_trip_through_a_save() {
        let (mut config, mut document) = open(fixture());
        assert_eq!(config.output.reinforce, None);
        assert_eq!(config.output.reinforce_thickness_mm, None);

        config.output.reinforce = Some(crate::config::ReinforcePolicy::MinThickness);
        config.output.reinforce_thickness_mm = Some(2.5);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("reinforce = \"min_thickness\""), "{saved}");
        assert!(saved.contains("reinforce_thickness_mm = 2.5"), "{saved}");
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(
            reparsed.output.reinforce,
            Some(crate::config::ReinforcePolicy::MinThickness)
        );
        assert_eq!(reparsed.output.reinforce_thickness_mm, Some(2.5));

        config.output.reinforce = None;
        config.output.reinforce_thickness_mm = None;
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(!saved.contains("reinforce"), "{saved}");
        let reparsed = Config::parse(&saved).expect("still valid TOML");
        assert_eq!(
            reparsed
                .output_params(std::path::Path::new("."))
                .expect("resolve")
                .reinforce_thickness_mm,
            reparsed.optimization.min_feature_mm
        );
    }

    /// And the same for the flush pass. Its depth is the one of the three whose
    /// absent value is not a constant and not another table's key either: it is
    /// two voxels of the grid, so what a save has to preserve is the *absence*.
    #[test]
    fn the_flush_policy_and_its_depth_round_trip_through_a_save() {
        let (mut config, mut document) = open(fixture());
        assert_eq!(config.output.flush, None);
        assert_eq!(config.output.flush_depth_mm, None);

        config.output.flush = Some(crate::config::FlushPolicy::Walls);
        config.output.flush_depth_mm = Some(2.5);
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("flush = \"walls\""), "{saved}");
        assert!(saved.contains("flush_depth_mm = 2.5"), "{saved}");
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(
            reparsed.output.flush,
            Some(crate::config::FlushPolicy::Walls)
        );
        assert_eq!(reparsed.output.flush_depth_mm, Some(2.5));

        // Switched off through the panel, which takes the depth with it: the
        // file keeps the policy it was told and loses the control over a pass
        // that is no longer running. See `ui::write_flush_policy`.
        config.output.flush = Some(crate::config::FlushPolicy::Off);
        config.output.flush_depth_mm = None;
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(saved.contains("flush = \"off\""), "{saved}");
        assert!(
            !saved.contains("flush_depth_mm"),
            "a depth outlived the pass it belongs to: {saved}"
        );
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(reparsed.output.flush, Some(crate::config::FlushPolicy::Off));
        assert_eq!(reparsed.output.flush_depth_mm, None);

        config.output.flush = None;
        config.output.flush_depth_mm = None;
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(!saved.contains("flush"), "{saved}");
        let reparsed = Config::parse(&saved).expect("still valid TOML");
        assert_eq!(
            reparsed
                .output_params(std::path::Path::new("."))
                .expect("resolve")
                .flush_depth_mm,
            None,
            "an absent depth has to stay absent through the resolver as well"
        );
    }

    #[test]
    fn a_shape_that_is_edited_keeps_its_inline_region_formatting() {
        let text = fixture();
        let (mut config, mut document) = open(text);
        let LoadSpec::Force { region, .. } = &mut config.loadcases[0].loads[0] else {
            panic!("the fixture's first load is a force");
        };
        let ShapeSpec::Sphere { center, radius } = region else {
            panic!("the fixture's first load region is a sphere");
        };
        *center = [11.0, 12.0, 13.0];
        let kept = *radius;
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert!(
            saved.contains("region = { shape = \"sphere\", center = [11.0, 12.0, 13.0], radius ="),
            "an inline region must stay inline: {saved}"
        );
        let reparsed = Config::parse(&saved).expect("reparse");
        let LoadSpec::Force { region, .. } = &reparsed.loadcases[0].loads[0] else {
            panic!("the load type changed");
        };
        let ShapeSpec::Sphere { center, radius } = region else {
            panic!("the region shape changed");
        };
        assert_eq!(*center, [11.0, 12.0, 13.0]);
        assert!((*radius - kept).abs() < 1e-12);
    }

    #[test]
    fn an_object_list_written_as_an_inline_array_is_edited_in_place() {
        // Legal TOML for the same structure, and something a hand written file
        // may use; the editor has to keep working on it rather than doubling it.
        let text = concat!(
            "keepout = [{ shape = \"sphere\", center = [1.0, 1.0, 1.0], radius = 1.0 }]\n\n",
            "[project]\nname = \"inline\"\n\n[resolution]\nvoxel_size_mm = 2.0\n\n",
            "[material]\npreset = \"pla\"\n\n",
            "[optimization]\nmass_fraction = 0.3\nmin_feature_mm = 6.0\n\n",
            "[output]\nstl_path = \"out.stl\"\n\n",
            "[[domain]]\nop = \"add\"\nshape = \"box\"\nmin = [0.0, 0.0, 0.0]\nmax = [20.0, 10.0, 10.0]\n\n",
            "[[supports]]\nregion = { shape = \"box\", min = [0.0, 0.0, 0.0], max = [1.0, 10.0, 10.0] }\n\n",
            "[[loadcases]]\nname = \"main\"\n[[loadcases.loads]]\ntype = \"force\"\n",
            "region = { shape = \"sphere\", center = [20.0, 5.0, 5.0], radius = 3.0 }\n",
            "vector = [0.0, 0.0, -10.0]\n",
        );
        let (mut config, mut document) = open(text);
        assert_eq!(config.keepout.len(), 1);
        let ShapeSpec::Sphere { center, .. } = &mut config.keepout[0] else {
            panic!("a sphere");
        };
        *center = [2.0, 2.0, 2.0];
        document.sync(&config).expect("sync");
        let saved = document.render();
        assert_eq!(
            saved.matches("keepout").count(),
            1,
            "the inline array must be edited, not duplicated: {saved}"
        );
        let reparsed = Config::parse(&saved).expect("reparse");
        assert_eq!(reparsed.keepout.len(), 1);
        let ShapeSpec::Sphere { center, .. } = &reparsed.keepout[0] else {
            panic!("a sphere");
        };
        assert_eq!(*center, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn a_file_that_is_not_toml_at_all_is_refused_with_its_own_error() {
        let error = Document::parse("[project\nname = ")
            .unwrap_err()
            .to_string();
        assert!(!error.is_empty());
        // An integer past u64 is not something the sentinel trick can carry
        // either, so it comes back as the parser's own complaint.
        assert!(Document::parse("a = 111111111111111111111111111111\n").is_err());
    }

    #[test]
    fn whole_token_replacement_never_touches_a_longer_number() {
        assert_eq!(
            replace_token("a = 12, b = 123", "12", "7"),
            "a = 7, b = 123"
        );
        assert_eq!(replace_token("x12 = 12", "12", "7"), "x12 = 7");
        assert_eq!(replace_token("a = 1.12", "12", "7"), "a = 1.12");
        assert_eq!(replace_token("no match", "12", "7"), "no match");
    }
}
