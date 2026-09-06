# Changelog

## 2026-09-05 - 0.43.0 - (title pending)

- **`[optimization.reduce]`** is read, validated and echoed by `check`: a
  required `target_safety_factor`, a `method` of `"continuation"` or `"beso"`,
  and the schedule around them (`ratio`, `refine_stages`, `min_mass_fraction`,
  and `evolution_rate` / `add_ratio` for the evolutionary method alone). With
  the table present `mass_fraction` is optional and means the fraction the run
  *starts* from - solid when it is left out. Refused by the solid and growth
  engines, beside `[optimization.wireframe]` or `[optimization.local_volume]`,
  beside an `update` scheme under `method = "beso"`, and with a `[material]`
  that declares no `yield_strength_mpa` - the safety factor is measured against
  it. Nothing removes material yet; the stage loop is the next change.
- **Switching to the growth engine** in the editor now takes the four
  `[optimization]` tables that engine refuses with it (`overhang`, `wireframe`,
  `local_volume`, `reduce`), as the switch to `solid` already did.

## 2026-08-17 - 0.42.0 - Copy and paste an object

Building a part out of four of the same bracket meant drawing it four times.
**`Ctrl+C` copies the selected object and `Ctrl+V` pastes it**, for every kind
of object the tree addresses.

- **`Ctrl+C` / `Ctrl+V`** on a domain entry, a keepout, a keepin, a support, a
  load or a whole load case with its loads. The clone lands on the coordinates
  it was copied from - two objects in one place, told apart in the tree list,
  the answer two adds already get - and becomes the selection. A pasted load
  case takes `constants::VIEW_EDIT_PASTE_NAME_SUFFIX` on its name, the one
  free-text name in the model; nothing is made unique, because names here are
  the user's own labels.
- **A copied load goes into the case the selection addresses** - the case
  itself, or the case a selected load is in - and the selection is read at the
  moment of the paste rather than remembered from the copy, so it can never name
  a case that has been deleted or renumbered since. Copying a load leaves it
  selected, so pasting straight away puts it back into its own case; selecting
  another case first is how the same load is copied into that one. With nothing
  that names a case selected there is nowhere to put it, and a paste with
  nowhere to go is no edit at all rather than half of one - or a guess.
- **One paste is one undo step**, restoring the selection the paste was made
  with; a copy is no step at all - nothing edited, nothing to save. The
  configuration and the document are changed together, as an add does, so a save
  writes the pasted object out with its own table and the file reads back as the
  session holds it.
- **The clipboard holds data, not a selection**: what was copied survives the
  original being dragged, deleted or undone away, and one copy pastes as many
  times as it is asked to. Session-scoped: never the platform's clipboard, and
  gone with the window.
- Both spellings of the two presses are read. The winit backend translates
  `Ctrl+C` and `Ctrl+V` into clipboard *events* and emits no key press for
  either, so reading key presses alone is a binding that works in a test and
  nowhere else; both are read in the one shortcut block, which is what keeps the
  modal guard and the text-field guard over them.
- **The paste press has a third route, because the backend answers `Ctrl+V` out
  of the *platform's* clipboard**: it delivers nothing at all when there is no
  text there - an empty clipboard, or one holding a screenshot - and this
  clipboard holds objects rather than text. So the window notes that press
  itself (`Editor::note_paste_key`, beside the `F` it already reads) and the
  shortcut block answers it in the same frame, folded into the one decision:
  one press is one paste however many ways it arrives. The note is taken
  *before* either guard can return, so a `Ctrl+V` typed into a text field or
  aimed at the modal is refused with its own frame rather than landing an object
  when the field is left or the question answered. Which key that is is decided
  by the letter the layout produced, falling back to the position on a layout
  that has no Latin letters - the rule the egui backend uses for the bindings it
  answers itself.
- **A paste that will do nothing does nothing at all**, the modes around it
  included: whether it can happen is resolved before a placement in progress is
  cancelled, through the very target resolution the paste itself uses, so the
  one refusable paste cannot cost the user the two clicks they are halfway
  through. A paste that will happen still cancels it first, as every structural
  edit does.
- Tests: every kind copied, pasted, compared against the original and found
  selected; the undo step and the copy that is not one; a paste after the
  original is deleted; the empty clipboard, the empty selection and the refused
  load, none of which disturbs a placement in progress; load targeting into its
  own case, into another, and refused after a delete has renumbered the list or
  put the selection on a shape; all three routes a press arrives by, each
  counter-verified by
  removing it; a refused note that must not arrive later, counter-verified by
  taking it after the guards; which key the window takes for a paste, by letter
  and by position; the file a paste is saved to; and `Ctrl+C` / `Ctrl+V` in the
  modal guard's own test, which now watches the clipboard as well.
- Version 0.42.0.

## 2026-08-13 - 0.41.0 - Show nothing but the part

A finished part is looked at through the model that produced it: the domain
shell, the regions, the grid and the gizmo are all still on screen. **One box in
the editor's precision block puts the part on screen alone** - and takes the
viewport's editing with it, because the handles a click would find are among
what it hides.

- **`show nothing but the part`**, beside `keep inside domain` and `floor grid`.
  Ticked, only the density surface is drawn: every other layer is left out
  whatever its own switch says, and unticking gives each of them back exactly as
  it was - the mode overrides the per-layer flags rather than writing them. The
  stress colouring and flat shading still apply, since both are the same layer.
- **The viewport stops taking edits while it is ticked**: no picking, no grab, no
  drag, no hover outline, no placement click. Whatever a gesture was holding when
  the box went on is put down once, by the editor
  (`Editor::suspend_interaction`) - a drag ends as letting go ends it, a
  placement cancels as Escape cancels it. **The camera and the whole panel are
  untouched**: orbit, pan, zoom and every widget go on working, which is what the
  mode is for. Decided at the two roots of the editor's pointer input rather than
  inside the gestures, so no path can be added that forgets to ask.
- **It cannot be ticked before a run**, and says why on hover in the same words
  every empty layer's switch uses. A part that goes away under a ticked box -
  an edit that invalidates the run - clears the switch with it, so the viewport
  is never left showing nothing and taking nothing. Cleared in `Scene::set`
  itself, whoever empties the layer, so no future caller can leave the flag
  standing behind a part that is gone and have the next preview re-isolate the
  window nobody asked to isolate.
- Nothing is persisted: it is a session switch like every other layer control.
  The run window has no precision block and gets no box.
- `constants::VIEW_EDIT_ISOLATE_PART_DEFAULT` (off) and
  `VIEW_EDIT_ISOLATE_PART_NOTE`, the line the panel shows for as long as the
  viewport is suspended.
- Tests: the box is in the precision block and not in `show`, and a click on it
  there is what changes the scene; every other layer hidden and all of them
  restored; the disabled box's hover text before a run, and the click it refuses;
  and the whole input path through the window - hover, grab, drag, click, orbit -
  with the box on, off, and with the part taken away under it. Counter-verified
  by removing the gate: hover, grab and the flag-clearing all fail.
- Version 0.41.0.

## 2026-08-10 - 0.40.0 - It is called growforge 3D

The product has a name of its own now. **Every surface a user reads says
`growforge 3D`; every name a machine matches on stays `growforge`** - that is
the rule the rebrand is built on, and the two are separate constants so the
first can move again without the second ever having to.

- **Displayed:** both window titles and therefore both panel headings
  (`growforge 3D 0.40.0 - <project>`, `growforge 3D 0.40.0 edit - <file>`), the
  console banner every command prints first, `--version`, the 80 byte header
  stamped into every exported STL and the `generator` of the stress JSON, the
  editor's file dialogs, the installer's `AppName` and its shortcuts, and the
  prose of a message that names the program.
- **Machine, unchanged:** the crate, the library, the executable, the command
  that is typed (`growforge edit ...` parses exactly as before), the GitHub
  slug, `Documents\growforge`, the installer's `AppId` and `{autopf}\growforge`
  install directory - so this release upgrades the last one in place - and
  `growforge-setup-<version>.exe`.
- `constants::DISPLAY_NAME` and `DISPLAY_NAME_AND_VERSION` are the new
  derivation of the displayed name; `PROGRAM_NAME` and `NAME_AND_VERSION` stay
  the machine one. `DISPLAY_NAME` is a deliberate literal - what the product is
  called cannot be assembled from what the package is called - held to one rule
  by a test: **the brand may extend the machine name, never contradict it**, so
  what is read on a window can always be typed at a prompt.
- **An upgrade cleans up the shortcut names it changed.** An `[InstallDelete]`
  block removes 0.39.0's `growforge` and `Uninstall growforge` icons, which an
  upgrade otherwise leaves sitting beside the new ones until an uninstall. It
  names those two files literally, inside the group this setup created itself:
  no wildcards, and nothing under `{autodesktop}` ever - a shortcut a user made
  is theirs.
- **Upgrading works at all now.** The shipped samples are installed read-only,
  and Setup refuses to replace a read-only file without `overwritereadonly`, so
  every upgrade over an existing install aborted on the first example and rolled
  itself back. Present since the installer was added in 0.39.0 and found by the
  first upgrade anyone ran; the samples still land read-only.
- Tests: the drift guard on that rule and on the two spellings of the brand, the
  titles, the panel headings, the console banner and the STL header re-pinned to
  the brand spelled out, and `--version` answering with the product while the
  usage line still names the command. Counter-verified by rolling the brand back
  to the bare machine name: seven display assertions fail, and the drift guard -
  correctly - does not. The setup is verified by a real 0.39.0 -> 0.40.0 upgrade
  in a sandbox: old icons gone, new ones there, an unrelated shortcut left in the
  same folder untouched, then an uninstall that leaves nothing.
- Version 0.40.0.

## 2026-08-09 - 0.39.0 - It installs, and it opens without being told where

growforge has been a binary you build and a path you type. Neither survives
being handed to someone: there is nothing to double click, and a shortcut to
`growforge.exe` with no arguments printed a usage error. This release is the
setup that installs it and the one command that can be started without a path.

- **A Windows installer.** `tools\installer.iss` compiled by
  `tools\build_installer.ps1`: `growforge.exe` into `Program Files\growforge`
  with the shipped `examples\*.toml` beside it read-only, a Start Menu shortcut
  (a desktop one offered unchecked), an uninstaller, and `CloseApplications` so
  an upgrade over a running editor asks to close it rather than failing on a
  locked file. The `AppId` GUID is fixed and commented as such: it is the
  identity that makes the next setup an upgrade rather than a second parallel
  install. The version is never written in the script - it is read off the built
  executable and passed in with `/DAppVersion` - and the build script publishes
  nothing: it builds, compiles the setup into the ignored `target\installer\`,
  and says where it landed.
- **`growforge edit` takes no path at all now**, which is what makes that
  shortcut sane. It asks for the file in the platform's own dialog before any
  window exists, starting in `Documents\growforge`, the canonical home of a
  user's configurations: `open` when that folder already holds `.toml` files,
  save-as when it is empty or not there yet, and from what comes back this is
  `growforge edit <that file>` down to the same call - an existing file opens, a
  typed name is scaffolded. Cancelling is an answer: one line, `nothing opened`,
  and a successful exit. **Nothing is created on the way to the question**; the
  folder comes into existence when a file is saved into it, and until it does
  the dialog opens in `Documents` itself rather than at a path the shell cannot
  show. `edit <file>` is
  untouched, and `check`, `view`, `bench` and `run` still require their path -
  there is nothing sensible to do without one.
- **The executable carries its version.** A new `build.rs` stamps the Windows
  VersionInfo resource - `FileVersion`, `ProductVersion`, `ProductName`,
  `FileDescription` - from the `CARGO_PKG_*` values Cargo hands a build script,
  so the Properties tab of a copied binary names the build and the manifest
  stays the only place a version is written. It reported 0.0.0.0 until now. The
  script does nothing at all when the target is not Windows.
- Tests: the dialog decision over a real folder - missing, empty, other files
  only, a directory that merely ends in `.toml`, an actual configuration, and an
  uppercase `.TOML` - plus the configurations home resolving under the platform's
  Documents folder, and the command line parsing `edit` with a path, `edit`
  without one, and the four commands that still refuse to run without theirs.
  The version resource and the installer are verified where they exist: a
  probe of the built executable's `VersionInfo`, and a silent install of the
  setup into a sandbox directory, `--version` from the installed copy, and a
  silent uninstall that leaves nothing behind - which is what caught the
  read-only samples surviving their own uninstaller.
- New dependencies: `dirs` (under the `viewer` feature) for the platform's
  Documents folder, asked for by known folder because one redirected onto
  OneDrive is not under `%USERPROFILE%` at all; `winresource` as a build
  dependency for the version resource. `tools/sweep.log` is now ignored.
- Version 0.39.0.

## 2026-08-08 - 0.38.1 - The grid switch moves to the controls it belongs to

- **The `floor grid` switch is in the precision block**, under `keep inside
  domain` and beside the `snap mm` it is ruled at, instead of down among the
  overlay switches in `show`: the grid is what makes that increment visible, so
  it reads as a workspace control rather than as one more layer. Same checkbox,
  same hover text, same scene state - only its place in the panel changed.
- The layer table now says where each switch is drawn (`SwitchHome`), so the
  shared `show` block leaves out the ones a block of its own owns rather than
  naming a layer. `view` and `run --view` are untouched: the floor grid is an
  editor layer and neither ever listed it.
- Tests: the precision block draws the switch, the `show` block no longer does,
  and a real click on the relocated row - laid out, hit tested and released over
  four headless frames - flips the same visibility it always did.
  Counter-verified by clicking beside it.
- Version 0.38.1.

## 2026-08-08 - 0.38.0 - The title bar and the terminal say it too

0.37.0 put the build in small weak text under both panels' headings, where it is
easy to miss. The two places a user actually looks first are the window's title
bar and the console the command was typed into, so the version moves into them.

- **Both window titles carry the version.** `growforge 0.38.0 - <project>` in the
  run and setup windows, `growforge 0.38.0 edit - <file>` in the editor, with the
  unsaved asterisk still on the end of it. Each panel draws that same title as
  its heading, so a screenshot of a panel alone names the build too - in the
  heading rather than in a footnote under it.
- **0.37.0's version line is gone from both panels**, superseded by the heading
  above it: one prominent statement instead of a subtle one repeated. What is
  left is one constant, `NAME_AND_VERSION`, behind the titles and the console
  line alike, so nothing that names the build can drift.
- **Every command prints the version first.** One line, `growforge 0.38.0`, at
  the dispatch and ahead of any other output, for `check`, `run`, `bench`, `view`
  and `edit` alike; `--version` and `--help` are answered by the parser and are
  unchanged.
- Tests: both title builders asserted whole - clean, dirty and saved for the
  editor - the first stdout line of the real binary asserted end to end, and each
  panel's painted heading asserted whole in its own headless frame.
  Counter-verified by breaking the constant.
- **New maintenance script `tools/sweep.ps1`** (run weekly by a Windows
  scheduled task registered outside the repo): deletes regenerable build
  caches - incremental, docs, an oversized debug profile - and stale session
  scratchpads, after a hundred gigabytes of cargo artifacts accumulated in a
  week. It refuses to run mid-build and every deletion roots in a validated
  absolute path.
- Version 0.38.0.

## 2026-08-07 - 0.37.0 - The window says which build it is

Tracing which build produced an artifact cost a week's forensics; the window
that drew it should have said so on its face.

- **Both panels name the build under their heading.** The run panel and the
  editor's panel each draw `growforge <version>` in the same weak small text as
  the adapter line beside it, from one constant assembled out of the manifest at
  compile time - so a screenshot identifies the binary and cannot name a version
  the running program is not. Window titles are untouched.
- Tests: each panel's own frame is drawn headlessly and the painted rows are read
  back for the string, counter-verified by breaking the constant, which fails
  both.
- Version 0.37.0.

## 2026-08-07 - 0.36.2 - The clamp report learns to see a floated face

A user's print failed on a part whose bottom face floated 0.44 mm above the
plate. Every code path at HEAD was hunted and exonerated - the command line, the
editor's full run and its generate-stl all produce a flush part - but the hunt
found something worse than the bug it was looking for: had any of them produced
that face, **the run report would have read perfectly clean**. The boundary
clamp seats a vertex resting up to half a voxel short of a surface and silently
leaves everything further out alone, counting nothing, so a face 0.88 of a voxel
off the shape it was drawn against passed as an ordinary free surface. The
slicer and the printer found the defect. The tool has to find it first.

- **The clamp counts what it did not seat.** After the corrections are applied,
  every vertex whose distance to its nearest analytic boundary - the same
  nearest-boundary selection the seat itself makes, now one shared function - is
  more than twice the offset a correction lands on and no more than
  `BOUNDARY_ADRIFT_WINDOW_VOXELS` (two voxels, four times the capture band and
  the flush pass's own depth) is counted in `ClampReport::adrift`, with the worst
  distance beside it. **No geometry changed**: nothing is moved that was not
  moved before, and a run's STL is byte for byte the file 0.36.1 wrote.
- **What it means is the run's answer, so what is said about it is too.** Under
  `engine = "solid"` the part *is* the shapes it was drawn from, every exported
  surface belongs to a domain or keepout boundary, and a vertex resting off one
  is a face about to ship in the wrong place: the line opens with `warning` and
  is drawn in the panel's warning colour. Under an optimizing or growing engine
  most of the surface belongs to nothing, and a free surface running near a
  boundary is what the engine decided - so the count is spoken only when
  `[output] flush` was asked for, where a vertex still short of a shape is that
  pass falling short, and the line says so plainly and names `flush_depth_mm` as
  what reaches further. With `flush = "off"` **nothing is said at all**: measured
  on the shipped cantilever, 1749 of 5956 exported vertices lie within the window
  of the part's own domain box, and a line on every run of every design says
  nothing about the one run that is wrong. The count stays on the report either
  way. `ClampReport::notes` takes both facts as arguments - `Problem::is_solid`
  and the new `Problem::is_flushing` - so the compiler is what keeps the console
  and the editor's panel saying the same thing.
- **Silent when there is nothing to say.** A report with nothing adrift produces
  no line at all, in any of the four runs it can belong to, which is asserted
  directly rather than assumed.
- Tests: the counted vertex, the vertex past the window, the seated vertex and
  both ends of the window arithmetic, in the clamp module; all three voices, the
  unflushed silence, the singular wording and the absence counter-check, in the
  report module and again through the panel's own drawing rule. The silence
  assertions were counter-verified by restoring the unconditional line, which
  fails four of them. The solid engine's plug fixture keeps asserting that
  nothing is adrift, now reading the pass's count instead of a helper of its own
  - one definition, which cannot drift from the production one.
- Version 0.36.2.

## 2026-08-07 - 0.36.1 - Two editor run-completion defects

Both reported by the user, on a session of the solid engine: "generate stl"
greyed out after a run that had just written its file, and the toolbar still
saying the run was going long after it was over.

- **A completed run keeps its design, whatever its engine reported.** Retention
  was fed by the per-iteration callback alone, so an engine that reports no
  iteration - the solid one reports none, by design - left nothing behind: a full
  run wrote its STL and "generate stl" stayed disabled for the rest of the
  session, and a solid preview left the button disabled too. `run_worker` now
  hands the field to a `keep` callback the moment the engine hands it over, and
  the editor's full run keeps it there; a preview keeps the field it ends on the
  same way. The engine's own reporting is untouched: it still reports nothing,
  and the panel still shows no per-iteration block for it.
- **Before the deliverable passes, never after.** What is kept is the field *as
  the engine produced it*, taken before `finish` resolves it in place, because
  "generate stl" puts whatever is kept through that same pipeline: a field kept
  after it would come back through the cavity policy and the `[output]` passes
  twice, and the flush's fringe reaches further out on every application. Pinned
  by the file - a full run with `flush = "walls"` and a generation from its kept
  design are compared byte for byte, and keeping the post-`finish` field makes
  that test fail with a part 8 kB larger than the one the run wrote.
- **A run that finishes says so.** The panel had a transition for a run that was
  stopped and one for a run that failed, and none at all for one that ended on
  its own, so "running the full pipeline; it will write ..." stayed on the status
  line for the rest of the session - a generic defect that the solid engine's
  one-frame run merely made impossible to miss. `note_run_success` sits beside
  `note_run_failure` in the same pump, keyed on `RunStatus::Finished`, with a
  dedup flag of its own reset at the three run-start sites: a full run says what
  it wrote and where, a generation says what it generated, and a preview - which
  writes nothing and says nothing when it starts - says nothing here either.
- Tests: a solid full run through the worker keeping its design, generating from
  it, and producing a file identical to the run's own, with the engine's silence
  asserted beside it; the flush comparison above; the toolbar offering the button
  after a solid preview; and, through the editor, the status line moving off the
  running message when a full run finishes, saying it once, and going back to the
  running message for the next run - with the generation's own line asserted where
  that path is already driven.
- Version 0.36.1.

## 2026-08-06 - 0.36.0 - The wall reaches the shape it was drawn against

From the user's ask: a smoothing pass that fills in material on the outer walls
of an optimized part, out to the exact dimensions of the shape those walls rest
against. 0.35.0 fixed the *mesh* side of this - the boundary clamp now seats a
vertex resting up to half a voxel short of a surface onto it - and half a voxel
is deliberately all it will reach, because a vertex further out than that rests
on nothing and moving it would drag a free surface onto a wall it was never
near. What is left is the field: an optimizer has little reason to resolve the
last cells against a boundary, and it leaves them at 0.4, 0.6, 0.8 in a band
that should be solid. Read at the iso level the wall dips inwards further than
the clamp can capture, and the exported face ripples.

- **`[output] flush = "walls"`**, a third pass over the finished density field,
  `off` unless a configuration asks for it. A design cell is raised to full
  density when it is not already there, its centre lies within `flush_depth_mm`
  of a domain or keepout surface, and the part's own material *inside that band*
  comes within `flush_depth_mm` of it. The third condition is what makes it a
  correction rather than a coat of paint: a stretch of boundary with no wall on
  it stays bare, and a member running clear of a face grows no detached plate on
  it. The `Design` gate is the whole of the safety story, as it is for the
  reinforcement - a keepout, a filled cavity and the space outside the domain are
  all void cells - so the pass can only ever add material to the design space.
- **`[output] flush_depth_mm`**, optional, default `FLUSH_DEPTH_VOXELS` (two) of
  the run's own voxel size: the artefact is a sampling one and the voxel is its
  scale. A written depth is checked for being a length at all with the rest of
  the `[output]` table, and for being between `FLUSH_DEPTH_MIN_VOXELS` (half a
  voxel, below which it cannot reach past the cells the surface already passes
  through) and `FLUSH_DEPTH_MAX_VOXELS` (eight, above which it lays a skin over
  every wall instead of seating the ones that rest on a surface) in
  `Problem::build`, which is where the voxel size is known - the same seam
  `[growth] step_mm` is checked at.
- **The caveat, stated in the note the pass writes and in the README.** Within
  that depth the fill cannot tell a pockmark in a wall from the end of one, so
  material that stopped near a surface is joined to it and a wall's edge grows a
  fringe of up to `flush_depth_mm`. That is the same reach that fills the ripple,
  seen from the other side. The pass is opt-in for it.
- **The order is trim, then flush, then reinforce**, and all three share the one
  re-analysis. The flush goes before the reinforcement on purpose: a rippled wall
  reads as a row of thin places, and the reinforcement would spend ball after
  ball on them and then warn that it could not reach the floor against a
  boundary. Flushed first, it measures a wall of even thickness. Everything the
  run reports afterwards - the stress table, the safety factor, the JSON, the STL
  - comes from the analysis over the filled field, so the fill is never free.
- **Reported, never refused.** A fill-only pass cannot disconnect anything, so
  there is no abort machinery here as there is in the trim: the report says how
  many cells were filled and what that joined, and the only warning it has is for
  a problem that described no surface to be flush with at all. The lines reach
  the console and the editor's panel alike, between the trim's and the
  reinforcement's.
- **The editor.** A combo and a conditional depth row in the output section, the
  section reset covering both keys, a format-preserving round trip through save,
  and the switch to `engine = "solid"` clearing them - that engine exports
  exactly the domain it was given, and refuses this pass by name as it refuses
  the other two. **The depth goes wherever the policy goes**, unlike the trim's
  fraction and the reinforcement's floor: switching the combo back to `off`
  takes it out as the engine switch and the section reset do, in one undo step.
  It is the one of the three whose legal range is measured in voxels, the row
  that edits it exists only while the pass does, and a leftover under `off` could
  be taken out of range by a later resolution edit and refuse the build with no
  row on screen naming the key.
- Tests: the predicate against a hand-computed brute force reference (the rippled
  wall filled, the open stretch bare, a member clear of a surface growing
  nothing, the interior untouched, void and solid cells never written, the depth
  deciding the reach, a second pass over a filled band raising nothing, and the
  no-surface warning); the config matrix both ways, the voxel range at both ends,
  the solid rejection, the editor's rows, reset, engine switch, the combo's
  switch to `off` taking the depth out of the config and the file in one undo
  step, panel notes and file round trip; the shared re-analysis, now over five
  runs; and end to end, a
  wall lying on the floor of its own domain with its outermost layer under the
  iso level - flushed, every vertex over it is on the analytic face to 1.0e-5 mm,
  and unflushed the same fixture dimples **1.6214 mm**, 0.81 of a voxel, past the
  1.0 mm the clamp captures.
- Version 0.36.0.

## 2026-08-06 - 0.35.0 - The part that was drawn, not optimized

From the user's bathtub plug: a cone shell with a cavity cut out of it and a
handle on top. Nothing about it is a topology problem, and until now the only way
to get it out of growforge was to fake one - `mass_fraction = 0.99` through SIMP,
where the optimizer still redistributes material it was not asked to move and the
density filter still erodes the cells at the boundary, so the walls come out
rippled by a fraction of a voxel.

- **`engine = "solid"`**, a third registered engine that optimizes nothing:
  every design cell is filled, `Reporter::iteration` is never called, and the
  field handed over is the cell classification read as a density. Everything
  downstream is untouched and runs exactly as it does after the other two - the
  cavity pass, the stress solve, the island cull, the boundary clamp, the STL -
  so the surface that ships is the configuration's own CSG held to its analytic
  shapes - to `2e-5 mm` on the shipped fixture, which is the boundary clamp's own
  offset rather than anything about the voxel size.
- **What the engine refuses, and why.** `[optimization] mass_fraction` is not
  required by it and is **rejected** when set (there is no share of a domain it
  fills completely); `[optimization.overhang]`, `.wireframe` and `.local_volume`
  are rejected in the style `growth_params` set, each saying why a fully dense
  part has no such concept; and `[output] trim` and `reinforce` are rejected
  unless off - both alter the very domain this engine exports. `penalty`,
  `stiffness_floor`, `min_feature_mm`, `max_iterations`, `convergence_tol` and
  `update` stay legal and unused, as they are under `growth`. Supports and load
  cases are still required: the stress report is what says whether the part
  holds.
- **The boundary clamp now corrects the artefact in both directions**, which is
  what makes "the domain itself" true rather than nearly true, and it is engine
  agnostic - a SIMP or growth wall standing against a keepout or a domain face
  gains exactly the same. Marching cubes and the Taubin smoothing after it
  scatter a wall's vertices to *both* sides of the surface; the pass only ever
  corrected the side legality could see, so the proud vertices were pulled back
  exactly onto the analytic surface and the ones that fell short stayed where
  they were. Measured on the plug fixture: worst deviation +0.0000 mm outward,
  **-0.70 mm inward** - half a voxel of dimple, which is the scalloping the
  render showed. A legal vertex within `BOUNDARY_CLAMP_CAPTURE_VOXELS` (half a
  voxel, the scale a cell-centre classification can be wrong by) of the boundary
  it rests on is now seated onto it by the same projection onto the same legal
  side; anything further away rests on nothing - an optimizer's free surface
  through the middle of the domain - and is left where the smoothing put it, and
  a seat that would land proud of another boundary is dropped. After it, every
  vertex of that fixture is on an analytic surface to within
  `2 x BOUNDARY_CLAMP_EPS_MM` (2e-5 mm), and no vertex is proud of one.
- **The reports say what ran.** Keyed on the engine rather than on the field, so
  no summary infers "solid" from a field of ones: `growforge check` prints
  *"solid          every design cell fully dense; nothing is optimized"* in place
  of the filter, overhang, update, local volume and wireframe rows, and names the
  domain volume as the target rather than inventing a mass fraction; the run
  summary prints what was filled instead of a zero iteration count and a zero
  compliance; and the viewer's panel says a solid run has no progress to show
  rather than waiting for a first iteration that is never coming.
- **The editor switches valid by construction.** Choosing `solid` takes the mass
  fraction, the three SIMP-only tables and the two output passes out; choosing
  another engine gives the mass fraction back at `STARTER_MASS_FRACTION`, and so
  does the engine section's reset. The mass fraction row exists exactly while the
  key does, previews run at the real resolution and configuration (a coarsened
  preview of this engine would be a preview of a different part), and the
  optimization tooltips now name both engines that ignore or reject each key.
- `OptimizationConfig::mass_fraction` is `Option<f64>` in the schema, resolved by
  `Config::optimization_params` to the value it always had for every other engine
  and to `DENSITY_MAX` for this one; a file that omits it under `simp` or
  `growth` is refused by name, where before serde said "missing field".
- Tests: the registry and `available()`; the config matrix in both directions
  (absent, present, out of range, each rejected table, each output pass, the
  unused knobs); the field's shape and its zero progress lines; the editor's
  switch, reset, row, preview and file round trip; and end to end, the plug
  itself - one body, no enclosed cavity, and every exported vertex on the cone,
  the cavity or the handle to `2 x BOUNDARY_CLAMP_EPS_MM`, from *both* sides
  (that assertion fails at 0.70 mm without the seating above). The bore fixture
  makes the same two-sided claim on a SIMP export, in place of a "the clamp only
  ever removes material" assertion that was true only of the one-sided pass. The
  three keepout fixtures now run on all three engines.
- Version 0.35.0.

## 2026-08-06 - 0.34.0 - A rejected frame is not the end of the session

From an incident: 40 000 GPU compute iterations on the physical GPU the window
was drawing on almost certainly tripped a driver reset, the next frame came back
as a validation error, and the viewer died on the spot - `error: the GPU rejected
the next frame of the viewer`, exit 1. A reset that lasts a moment took the
window with it, and anything unsaved in an editing session would have gone too.

- **A rejected frame is retried rather than fatal.** `Gpu::render` reads what the
  surface answered as a `SurfaceStatus` and asks `SurfaceHealth` what to do: a
  rejection reconfigures the surface and skips the frame, exactly as a stale one
  does, and only `VIEW_SURFACE_REJECTION_LIMIT` (30) consecutive rejections give
  up - about a fifth of a second at 144 Hz. **An acquired frame is what clears
  the streak**, so recovery is proven by drawing; a timeout, an occluded window
  and a stale surface neither count towards the limit nor forgive what came
  before them. One console line on the first rejection of a streak and none after
  it, so a reset that passes costs a note rather than a wall of them.
- **The exit semantics are unchanged.** A device that never comes back still
  ends the window with an error and the process with a failure, now saying what
  happened: *"the GPU rejected 30 consecutive frames of the viewer; the device
  was likely reset and did not come back"*.
- **An editing session's unsaved document survives the window.** Any fatal frame
  error now goes through `ViewerApp::fail`, which writes a dirty session's
  configuration to `<name>.recovered.toml` beside the file it came from before
  the teardown, through the same format preserving projection a save uses - the
  recovered file is byte for byte what `Ctrl+S` would have written. **The file
  being edited is never touched**, and the session is not marked saved: a rescue
  is not a save. `view` and `run --view` windows have no document and do
  nothing. Both outcomes are reported on the console under the `editor rescue`
  prefix, the write that failed included.
- Tests: the streak policy below the limit, at it, cleared by a drawn frame, and
  unmoved by skipped and stale frames; the rescue writing what the save it stood
  in for produces, leaving the original bytes and the unsaved marker alone,
  replacing a previous recovery, naming the file in a write that could not be
  made, and writing nothing when the projection itself fails; the fatal path
  rescuing a dirty session, keeping its error, and writing nothing for a clean
  session or a window with no document.
- Version 0.34.0.

## 2026-08-06 - 0.33.0 - The nine decades under the load path

The tension case again, and the same class of incident as 0.28.0 one layer down.
A 1.2 M degree of freedom run at `mass_fraction = 0.15` over a 204 mm span, held
on one side, exhausted 40 000 device iterations at a relative residual of 3.5e-6
against a target of 3e-8 - still falling, not stalled - and then 10 000 CPU ones
at 4.3e-6. Nothing was wrong with the model: the stiffness contrast it hands the
conjugate gradient was hardwired at 1e9.

- **`[optimization] stiffness_floor`**, the `Emin` of
  `E(x) = Emin + x^p (E0 - Emin)` as a fraction of the solid modulus - what an
  emptied *design* cell still carries. Optional, defaulting to
  `constants::SIMP_EMIN_FRACTION` (1e-9), bounded by `STIFFNESS_FLOOR_MIN`
  (1e-12) and `STIFFNESS_FLOOR_MAX` (1e-3) inclusive and refused outside them by
  `Config::optimization_params`, so `growforge check` says so before any work is
  done. Every existing configuration resolves to the value it always ran at.
- **What it buys and what it costs.** The contrast across the load path is what
  a conjugate gradient's iteration count grows with, and nine decades of it is
  what an ill-conditioned honest model spends its whole budget on. A higher
  floor is parasitic stiffness the design never bought; at 1e-6 an emptied cell
  is still a thousandth of one at density 0.1, and at the 1e-3 the key stops at
  the void is carrying the structure rather than helping it be solved for.
- **Forced void cells are untouched at any floor**: they carry a literal zero
  and their nodes are pinned out of the system entirely. The floor is a design
  cell's, and only a design cell's.
- **One resolved value reaches every mirror of the interpolation.**
  `engine::simp_moduli` takes it beside `penalty` - the optimizer, the benchmark
  and the post-run stress analysis all call it - and `Objective` forms its `emin`
  from `problem.optimization.stiffness_floor` rather than the constant. A
  forward sweep and an adjoint taken at two different floors is a gradient
  nothing reports as wrong, so the finite difference check is run at three of
  them.
- **Scope held deliberately narrow**, as 0.28.0 held it: the iteration caps stay
  compile-time constants - a cap is the guardrail that ends a solve which has
  stopped converging, not a knob - and the GPU precision fixtures stay pinned to
  the default floor, because what they measure is the worst contrast a
  configuration can hand the backends.
- **The editor grew the row beside `penalty`**, drawn by the exponent-notation
  drag value 0.28.0 added for the tolerance: `tolerance_widget` is now
  `scientific_widget` over the range its caller names, and
  `VIEW_EDIT_TOLERANCE_DRAG_FRACTION` is `VIEW_EDIT_SCIENTIFIC_DRAG_FRACTION`
  with two fields under it. The row resets with its section, and the save writes
  `stiffness_floor = 1e-6` rather than a decimal point and five zeros.
- Tests: the key absent, named, at both bounds and refused past either (plus
  zero, a negative, `nan` and both infinities), the message naming the section,
  the key and the range; the default pinned to the constant *and* to 1e-9; the
  moduli of a design cell at density 0 and 1 read off the array the solve is
  assembled from; the finite difference gradient at 1e-12, 1e-6 and 1e-3, all
  within 8e-10 relative; the row drawn pinned and unpinned and unchanged by the
  frame that drew it, the section reset clearing it, the panel width guard with
  it pinned, and the key round-tripped through a save at three values and gone
  again from a file that never had it.
- Version 0.33.0.

## 2026-08-05 - 0.32.0 - Collapsible panel blocks

Asked for by the user: the object lists closed to begin with, and a dropdown on
every other labelled block of the editor's panel.

- **Every labelled block of the panel is now a collapsing header**: `objects`,
  `properties`, `show`, `stress`, `problem` and `controls`, beside the six
  scalar sections that already had one. The first four open, the last two
  closed. The toolbar, the precision block and the validation block are
  deliberately not collapsible: warnings may not be hideable.
- **The five object lists start closed**, which is what the panel opens on: the
  tree, folded, with the properties of whatever is picked under it.
- **Each list's header carries an explicit `id_salt`.** Their labels hold a live
  count, and egui identifies a header by its label text - so without it,
  `keepout (2)` and `keepout (3)` are two different headers, and a list the user
  had opened would shut itself the moment an object was added to it.
- The `delete` button moved from the `properties` heading to the first row
  inside that block. Same button, same undo step, same `Delete` key beside it.
- The shared layer switches take their heading from the panel that draws them:
  the run panel writes it as before, the editor's header carries it instead of
  saying the same word twice. The run panel is otherwise untouched.
- Tests: every block's opening state, list by list and section by section, and a
  list that stays open across the add that renames it - which fails without the
  salt. The width guard re-measures every row at its new indent depth.
- Version 0.32.0.

## 2026-08-05 - 0.31.1 - The stray line down the editor panel

Reported by the user: a full height vertical line drawn across the side panel,
about 65 points in from its left edge, over the top of everything.

- **The `add` row of each object list now wraps.** It put one button per shape
  kind on a single `ui.horizontal`, which never wraps, so the two kinds 0.30.0
  added took it 62.875 points past the panel's content width. A row that
  overflows is not merely clipped: egui finds the frame wider than
  `VIEW_EDIT_PANEL_WIDTH_POINTS`, clamps the panel rect back by sliding its
  *left* edge inwards, and paints the panel's separator at the edge it has just
  moved - which is the line, 62.875 points in from where the panel really
  starts. `horizontal_wrapped` puts the row on as many lines as the schema needs.
- **`triangle` was off the window entirely** - laid out at x 1282 to 1335 of a
  1280 wide window, so it painted nothing and its interact rect was empty. In
  all four lists it could not be seen or clicked; `cone` ended 2 points into the
  panel's margin. Both are back inside the panel.
- Audited every other row of the panel by measurement rather than by eye: the
  widest are the wrapped notes at 336 and the summary lines at 328, against a
  budget of 344. The pass notes and the stress block are gated behind a finished
  run and so are outside a headless measurement; they are single wrapping
  labels, which cannot overflow, and now say so where they are drawn. Nothing
  else changed.
- `ui::panel` returns how far its content spilled past the width it declares,
  measured inside the frame where egui decides it rather than after the clamp
  has hidden it.
- Tests: the whole panel laid out headlessly in the widest state it has - both
  engines, every object kind in the tree, every shape and every load as the
  selection, every optional table pinned, every section expanded - asserting
  zero overflow; it fails at 62.875 points with the wrap reverted.
- Version 0.31.1.

## 2026-08-05 - 0.31.0 - Axis-snap resize drags

Reported by the user: dragging a cylinder's end handle *"veers into another
axis"*, because a free camera-plane drag changes the length **and** tilts the
shape with every pixel that is not exactly along it - and then *"make sure all
shapes resize box snaps to the drag axis as well"*.

- **A resize drag now latches on to the dimension it set off in.** The
  classification is made once, as soon as the pointer has covered
  `VIEW_EDIT_RESIZE_LATCH_MM` of the drag plane, from the accumulated
  displacement rather than the frame's, and holds until the button comes up
  (`gizmo::Latch`). Inside that dead zone the drag changes nothing at all, so a
  press can no longer nudge what it grabbed. Every direction is compared as it is
  *seen*, in the drag plane, and an axis turned towards the camera is no
  candidate at all - the existing near-parallel guard, read as the sine squared
  it is.
- **An end handle** - a cylinder's, a tube's, a cone's - pulled along the line
  through the two ends sets the **length**: the end slides on that line, so the
  shape keeps its direction whatever the hand does afterwards, and it is the
  *span* that lands on the snap increment rather than each coordinate separately,
  which is what would take the end off the line. Floored at the smallest usable
  length, so an end dragged through the far one stops rather than turning the
  shape around. Pulled across the axis instead, the same handle still places its
  end freely in the plane - the only way to point a `p1`/`p2` shape somewhere
  else - and keeps that freedom for the rest of the drag.
- **A box corner** now grows the one edge the gesture started along, in the box's
  own frame, and leaves the other two dimensions exactly as they were. It grew
  all three at once, which is what made a corner impossible to aim.
- **A triangle's vertex** latches the same way, on the three directions a corner
  really has: along the edge it faces, which shears the triangle at a constant
  height; along its own **height** off that edge; or, when the gesture sets off
  out of the triangle's plane, free - which is the prism's only pose control and
  had to survive. The out-of-plane drag has to *beat* both in-plane directions to
  win, so a tie goes to the dimension, and a triangle already flat has neither a
  height nor a plane and is placed freely. The collapse guard still holds a
  latched corner a tenth of a millimetre off the line through the other two.
- Unlatched on purpose: the move handles and a tube's bend, which are placements
  through and through; the axis handles are one dimension by construction.
- **A drag frame that asks for the shape it started on now does nothing at all**,
  where it used to commit that shape and open an undo interaction for it. The
  dead zone made that an everyday gesture - a click on a resize handle with an
  unsteady hand - and it recorded an undo step that undid nothing *and cleared
  the redo stack*, so merely touching a handle cost the user their redo. It also
  leaves the containment clamp to the first real frame, so an object a file holds
  outside the domain is no longer pulled in by a click, with no undo step over
  the move. The same applies to any drag the increment rounds back to where it
  began; once a drag has changed something, every frame of it is committed, so a
  drag that returns to its start still puts the shape back.
- `Drag::placed_at` and `Drag::shape_at` take `&mut self` for the latch; new
  `gizmo::place_endpoint` and `gizmo::place_vertex` put a point *at* a place,
  which is what keeps a latched drag exactly on its line. New tunable:
  `VIEW_EDIT_RESIZE_LATCH_MM`.
- Tests: dead zone, latch and its persistence, ties resolving to the resize,
  the camera looking down the axis, scalar snapping, the minimum-length floor
  without a flip, tube and cone ends; the corner cases - dominant axis, the axis
  facing the camera excluded, a turned box latching in its own frame; and the
  vertex cases - edge slide at a constant height, height at a fixed base, the
  free out-of-plane drag, the collapse guard under a latch, and the flat
  triangle. Through the editor's own paths: a press and a shake inside the dead
  zone leaving the undo *and* redo stacks alone and the clamp uncommitted, and
  the drag that does leave it still one step measured from where it started.

## 2026-08-05 - 0.30.0 - Two more shapes to draw with

"also add cones and triangles to the shapes". Both are now primitives in their
own right, legal wherever a shape is and drawn, picked, dragged and saved like
any other.

- **`shape = "cone"`** - a capped frustum between `p1` and `p2` with a radius at
  each. `radius1 == radius2` is the cylinder it generalizes, to the last bit of
  arithmetic, and **`radius2 = 0` is the apex of a true cone** rather than a
  degenerate shape: the wide end and the length are what `min_extent` measures,
  so a cone that comes to a point is never rejected for coming to a point. Its
  field is the **exact** signed distance everywhere - a frustum is a surface of
  revolution, so the whole of its geometry is the distance in the sample's own
  meridian half plane to the trapezoid its profile is - and its bounds are the
  support function of the two cap discs it is the hull of.
- **`shape = "triangle"`** - the triangle through `a`, `b` and `c`, extruded
  `thickness` millimetres symmetrically about its own plane. Its field is exact
  too: the in-plane distance to the triangle, whose edge and vertex regions are
  the clamp each edge applies to its own parameter, combined with the slab the
  way a cylinder combines its wall and its caps. Its bounds are its six corners
  and nothing more, and its smallest extent is the **diameter of its inscribed
  circle** - the smallest ball that fits, which is the vocabulary the minimum
  feature size is already written in - so a sliver a metre long is a sliver.
- **A triangle of no area is refused outright**, with its shortest altitude
  named. Deliberately not the tube's precedent, where a bend on the line through
  the two ends is still a tube: three collinear points enclose nothing at any
  thickness, so there is no shape to fall back on.
- **Both are picked in closed form**, because both are convex: the cone by its
  lateral quadric and its two cap discs - with the linear case a ray parallel to
  the slanted wall really has, and a documented tangency band, without which a
  ray straight down a true cone's axis loses its entry to one ulp of cancellation
  and picks the far side - and the prism by the box's slab method over the five
  planes it is bounded by.
- **Editor handles.** A cone gets an end handle and a radius handle at each of
  its two ends, and turns on the cylinder's own arcs. Its narrow radius drags all
  the way to **zero**, where every other radius in the editor stops at a tenth of
  a millimetre, because that zero is the shape the drag is aiming at; at the apex
  that handle stands on the end handle, and the press goes to the radius - the
  one that can bring the cone back off its point - by the rule a straight tube's
  bend handle gets. A triangle gets a handle on each corner and one for its
  thickness, and **no rotation arcs**: three free vertices are already complete
  pose control, which is the sphere's reason turned inside out. A corner is held
  a tenth of a millimetre off the line through the other two, and the thickness
  grows twice as fast as the handle that moves it, because the extrusion is
  symmetric.
- New tunables: `VIEW_EDIT_NEW_CONE_TAPER_FRACTION` (a new cone tapers visibly -
  one that looked like a cylinder would read as a failed add),
  `VIEW_EDIT_NEW_TRIANGLE_THICKNESS_FRACTION`, and `VIEW_EDIT_PICK_TANGENT_EPS`.
- **The selection shell's contract is strictly positive clearance**, said in
  those words rather than as an exactness the sloped case does not have: a
  cone's shell clears its wall by exactly the margin at the apex and in the
  cylinder-equivalent case, and by two thirds of one at the wide rim of a steep
  taper, because translating the caps while growing the radii tilts the shell's
  wall. The test now pins a bounded fraction of the margin at every surface
  point of every kind rather than merely a positive one, which is what catches a
  construction drifting back towards zero.
- The viewer's tessellation is `frustum`; `tessellate::cone` was already the
  **arrowhead** every overlay is drawn with and keeps that name, with both doc
  comments and the README now saying which is which.
- Tests: the equal-radii cone against the cylinder pointwise; both fields against
  a sweep of their own surface, and every projection landing on that surface at
  exactly the distance reported; the bounds against the cap circles and the six
  corners; the nearest surface point refused where a whole circle is nearest and
  answered at the cap centres and past the apex; both analytic hits against a
  march of the shape's own field, plus the apex down the axis and the ray
  parallel to the slant; the rejections, the TOML round trips, the handle
  inventories, the apex drag, the vertex drag and the thickness drag - the arms
  the drag dispatcher's wildcard would otherwise swallow - the callouts typed
  both ways, the meshes closed at every taper and either winding, the panel drawn
  for both kinds, the defaults sized like every other, and **end-to-end runs of a
  cone keepout and a triangular prism keepout on both engines**, each with a
  support and a load region of the same kind.
- Version 0.30.0.

## 2026-08-05 - 0.29.0 - One click back to the defaults

"can we add a reset to defaults button for properties". Every scalar section of
the editor's panel now carries one, on a row inside its own header.

- **A `reset` button in each of the six sections** - engine (the `[solver]`
  table with it), resolution, material, optimization, growth and output - which
  puts that section's keys back to what a configuration that never mentioned
  them would run at, and touches nothing else. The objects deliberately have
  none: a shape is geometry the user drew, and there is no default position for
  a load to return to.
- **One click is one undo step.** The interaction is opened and closed in the
  same gesture, the way the tube's `straighten` button does it, so one undo puts
  every key of the section back exactly as it was.
- **A reset lands a configuration that still builds.** Every coupling the
  interactive controls honour is honoured here: leaving the growth engine takes
  the `[growth]` table with it exactly as the engine combo does, and the
  resolution goes on carrying exactly one of its two keys. Switching a section's
  optional sub-tables off is part of the reset - overhang, wireframe, local
  volume, `[growth.symmetry]` - because a table that is not there is a feature
  that is not running, and the tooltips say so before the click.
- **`[optimization] mass_fraction` keeps its value.** How heavy the part may be
  is design intent, not a default anyone can invent, and a button that quietly
  changed the weight target would be one nobody could trust with an hour's work.
- **`[optimization] min_feature_mm` is derived rather than fixed**: it is a
  length, and what a length means depends on the grid, so it comes back to
  `VIEW_EDIT_RESET_MIN_FEATURE_VOXELS` times the voxel size this configuration
  is solved on. That constant *is* `MIN_FEATURE_VOXELS_WARN` rather than a second
  copy of the figure, so a reset lands exactly on the smallest feature the
  density filter resolves and can never leave a configuration the next run warns
  about. The voxel size is the key when there is one, and the grid the last build
  derived when the resolution is a cell target.
- **Both `[output]` paths keep their values**: where a run writes is a decision
  about the project and not about the part - the rule that section already drew
  them under - so `stl_path` (the one required key of the schema, and the one
  with no default at all) and `stress_json` alike are carried across. A reset
  puts properties back; it does not move deliverables. The rest of the table
  goes, policies included, so `boundaries` is `exact` again, `trim` and
  `reinforce` `off`, `voids` `warn` and `islands` `cull`.
- **The button is disabled while its section is already at its defaults**, with
  "this section is at its defaults" on hover. The predicate is the reset itself
  applied to a copy of the configuration - a section is at its defaults exactly
  when resetting it would change nothing - so there is no second statement of
  what those defaults are for the first to drift from.
- **Comment loss on removed keys is amplified, and is said out loud**: the file
  layer has always taken a key's comment with the key (its module doc says so),
  and a reset removes up to a dozen keys at once. What a reset does not touch
  keeps its comments, and one undo brings the removed keys back.
- A table whose keys are all optional is emptied rather than removed, which is
  the rule the tolerance checkbox already followed: an empty `[solver]` or
  `[growth]` resolves to precisely what no table at all does, and removing it
  would take a comment block with no key left to hang on. The engine's reset is
  the exception, because a `[growth]` table under any other engine is illegal.
- Tests: every section's reset asserted key by key, including the mass fraction
  left alone, the minimum feature derived from both of its voxel sources - and
  left alone under a cell target that has never built, where there is no grid to
  derive one from - both output paths carried across, and the `[growth]` table
  the engine's reset clears; one undo restoring the whole configuration, for
  each of the six; the predicate true, false and true again, with the rows drawn
  in both states; every section reset out of two adversarial states - the growth
  engine with a full table under it, and the local volume cap under the update
  scheme it requires - still building; and two saves read back as text: the
  removed keys and their comments gone with the untouched ones intact, and the
  emptied `[solver]` still carrying its own comment block where the `[growth]`
  table is gone outright.
- Version 0.29.0.

## 2026-08-04 - 0.28.0 - A target nobody was allowed to argue with

The tension case killed a run at iteration 99 with the solver 1.5x short of a
fixed target: the CPU conjugate gradient reached the 10 000 iteration cap at a
relative residual of 1.558e-8 against the hardwired 1e-8, and an honest model
lost its run to a number no printed part could tell apart.

- **`[solver] tolerance`**, the relative residual the *optimization* loop's
  solves are taken to, on either backend. Optional, defaulting to
  `constants::CG_RELATIVE_TOLERANCE` (1e-8), bounded by `CG_TOLERANCE_MIN` (1e-10)
  and `CG_TOLERANCE_MAX` (1e-4) inclusive and refused outside them by
  `Config::solver_params`, so `growforge check` says so before any work is done.
- **Scope held deliberately narrow.** Stress recovery keeps its own
  `STRESS_CG_*` pair - a property of the recovery, not of the part - and the
  iteration caps stay constants: a cap is the guardrail that ends a solve which
  has stopped converging, not a knob.
- **`SolverParams` lost its derived `Eq`** (an `f64` has no such trait) and
  gained a written-out `Default`: a derived one would hand out `tolerance: 0.0`,
  a target no residual reaches, so a `Problem` assembled in memory would spend
  the whole iteration cap and then fail. The impl is the reference
  configuration - CPU, at the tolerance every recorded trajectory was recorded
  at.
- **The `solver` summary line names a loosened target**, beside the fallback
  note and for the same reason: the line says what a run's solves will really
  do. `report::solver_line` is built rather than printed so the wording is
  asserted on.
- **And the CPU's reproducibility promise carries its conditions**, in the
  summary line, the panel's backend tooltip, the README's table and
  `DEFAULT_SOLVER_BACKEND`'s docs alike: bit-for-bit *for a given build and
  thread count*, the qualifier `SolverBackend::Cpu`'s own docs always had. The
  host reductions - the conjugate gradient's dot products, the compliance sum -
  are `par_iter().sum()`, which rayon orders by the size of the pool, so the last
  bits move with it exactly as the GPU's host half does. Measured at about 1e-9
  relative on one solve; the README's reproducibility section now says so rather
  than promising "everywhere" unqualified.
- `Objective` takes its `cg_tolerance` from `problem.solver.tolerance` rather
  than the constant; `with_cg_tolerance` still overrides it, which is how the
  finite difference harness keeps a target tighter than any configuration may
  name. With that, `fea::CG_TOLERANCE` had no readers left and is **removed**:
  a re-export that named a target nothing takes any more is a trap, and
  `constants::CG_RELATIVE_TOLERANCE` through `Config::solver_params` is the one
  source.
- **The editor grew the row beside the backend combo**, and a widget to draw it
  with: the shared numeric field rounds to three decimals, a floor of 5e-4 that
  every tolerance sits below, so it would have shown 1e-8 as `0` and set it to
  `0` on the first pixel of a drag. `tolerance_widget` drags by a fraction of
  the value itself, is bounded by the range the configuration accepts, and reads
  and writes exponent notation - as does the save-back, so a saved tolerance is
  `3e-8` and not a decimal point and eight zeros. Each of the two widgets over
  the one-struct `[solver]` table carries the other's key across.
- Tests: the key absent, named, at both bounds and refused past either (plus
  `nan`, the infinities, zero and a negative); the `Default` pinned; the
  objective's source and its override; a looser target measured to cost fewer
  iterations for the same compliance (63 to 44 iterations over four decades, the
  compliance moving by an order of 1e-9 to 1e-10 - the digit itself varies with
  the thread count, so the test asserts the order and not the digit); the
  summary line in both states and beside a fallback;
  the row drawn in both states; and both keys of the table round-tripped through
  a save, together and apart.
- Version 0.28.0.

## 2026-08-04 - 0.27.0 - Shifting the mass into the arms

The user asked for two things that turned out to be one: "needs an arm
thickening feature too as these are not printable as is", and "maybe a mass
shifting... shift some of the mass into the arms and will actually make the part
stronger". The trim pass already frees the mass of what carries nothing; there
was nowhere for it to go.

- **`[output] reinforce = "min_thickness"`**, a printable thickness floor held on
  the exported field, with `reinforce_thickness_mm` defaulting to
  `[optimization] min_feature_mm` - the one `[output]` control whose default
  comes from another table, resolved in `Config::output_params`, where both are
  in hand. `min_feature_mm` is a density filter radius, a smoothing length; it
  never was a floor on what the design converges to.
- **`src/reinforce.rs`**, a sibling of `trim` and `voids`: an exact Euclidean
  distance transform over the part (three separable lower-envelope passes,
  Felzenszwalb and Huttenlocher, linear per axis, the block padded with open
  space so a member reaching the domain edge is bounded by a real surface), the
  **spine** of every member as the local maxima of it, and twice the distance at
  those maxima as the local thickness - which is what an inscribed ball touching
  both sides of a member is, and is not what twice the distance means anywhere
  else. The surface sits half a voxel inside the nearest cell that is not part,
  so a member `w` cells across measures `w` cells thick.
- **The operation is a ball fill** on `CellKind::Design` cells alone, which is
  the whole of the safety story: keepouts, the space outside the domain and
  filled cavities are all void cells, and forced solid cells are already full.
  The radius carries the same half voxel the measure takes off, so a reinforced
  member meets the floor rather than stopping a half cell short of it.
- **Impossibility is counted, never refused.** A member pressed against a keepout
  or the edge of the domain has nowhere to grow; the pass re-runs the transform
  over the field it just changed and warns with the count of places a ball of the
  floor's diameter still does not reach - the `ClampReport::gave_up` precedent.
  Everything it can reach is reinforced regardless. That second reading is the
  morphological **opening** of the part by that ball, not the spine measure
  again: filling moves a member's spine, and re-reading the old one condemned
  eight hundred places on a plain cantilever that had come out exactly right.
- **One re-analysis for both passes.** `complete` runs the trim on the stresses
  of the field the engine produced, reinforces after it, and if *either* changed
  the field runs the cavity pass and the stress solve once more over the result,
  so the table, the safety factor, the JSON and the STL describe what shipped.
  Nothing is trimmed afterwards, by design: reinforcement material is
  deliberately unloaded, and a second trim would remove exactly what was added.
- **Named `reinforce`, not `thicken`**: `engine::growth::thicken` sizes a
  *skeleton*'s struts from the flow through them while a design is being built.
  This measures a *finished field* about to be exported. The README says so in
  both places.
- **Reported everywhere the trim is**, and beside it, so the exchange reads as
  one: `print_reinforce_report` on the console, `reinforce` in the editor's pass
  notes block, the two `[output]` rows in the panel with their tooltips, the
  save-back keys, and one informational note when
  `[optimization.local_volume]` was capped - the cap is a constraint on the
  optimizer, and printability outranks it at export.
- Tests: the distance transform against hand-written fields (single cell, slabs
  of every width, an L against brute force) and invariant under the order of its
  axes; a bar under the floor thickened and a bar on it left byte-identical; the
  **exact** sphere-fitting thickness - twice the largest inscribed ball covering
  a cell - implemented independently in the tests and asserted on the result, so
  what the spine measure claims is proved rather than assumed; a member pinned
  against a keepout counted, warned about and the free one beside it still fixed;
  a plate that grew away from the domain edge *not* warned about; only design
  cells ever written; the local volume note in all three states; the
  consolidated re-analysis counted through a scripted `Stages` for both passes,
  each alone and neither; config, save-back and panel round trips; and an
  end-to-end grown fixture with a bore, exported with the pass off and on.
- Version 0.27.0.

## 2026-08-04 - 0.26.2 - A safety factor of two pieces held apart by air

The user's report: "its saying there is a safety factor of 5.07 where all thats
holding the part together is air". The export was **two disconnected bodies** -
each rod's load group shunting into its own local support anchor - so the solve
was a correct reading of the model it was handed, and the number it produced was
meaningless for a part meant to link the two. Nothing can tell the report that
the supports are fictitious; it can tell that the mesh came out in pieces, and it
now says so beside the number.

- **`stress::disconnected_bodies_note(bodies)`**, one sentence for every surface
  the summary reaches: "the export is N separate bodies - this safety factor
  describes each piece against its own supports, not one connected part".
  `StressSummary` carries it as `warning`, prefixed with `warning:` on the
  convention the trim and clamp notes set, and `lines()` puts it *first* - a
  reader who stops after one line has to have read the caveat, not the factor.
- **All three surfaces**: the editor panel draws it above the headline in the
  warning colour (the pass-notes rule), the editor console echoes it as
  `editor stress warning: ...`, and `print_stress_report` prints it as
  `  warning      ...` above the table. One sentence, three layouts, from the one
  formatter 0.23.0 established.
- **The count is the export's**: `IslandReport::bodies.len()` of the surface that
  was *written*, post-trim and post-cull, threaded in as a parameter -
  `StressReport::summary(bodies)` and `print_stress_report(outcome, bodies)` - so
  the warning and the `mesh bodies` line above it can never disagree. A stress
  report has no way to know this on its own; it is a property of the mesh.
- **A single-body export is byte for byte what it was**, warning `None` and every
  other line untouched, which the tests pin.
- Tests: the summary at 1, 2 and 5 bodies; the console block's line and every
  branch of the block printing; the panel block drawn in both states; a two-lump
  fixture - two lobes, each with its own support and load, joined by nothing -
  through the editor's export and through the whole post-run pipeline, asserting
  the warning reaches the panel, the console echo and the `Completed` the command
  line prints from.
- Version 0.26.2.

## 2026-08-03 - 0.26.1 - The floor grid at half brightness

The user asked to "dim the grid 50% and add the option to disable it". The
option was already there - the **show** panel's `floor grid` switch, tooltipped
since 0.25.0 - and stays the way to turn the ruling off outright; this is the
dimming half.

- **`VIEW_COLOR_GRID_MINOR` and `VIEW_COLOR_GRID_MAJOR` halved**, `[0.55, 0.60,
  0.68]` to `[0.275, 0.30, 0.34]` and `[0.78, 0.84, 0.92]` to `[0.39, 0.42,
  0.46]`, alphas untouched. The same factor on both, so the major lines keep the
  contrast against the minor ones that they had.
- **On the RGB, not the alpha**: the grid is an opaque layer, drawn by the
  pipeline built with no blend state, so the alpha a grid vertex carries never
  reaches a blend and halving it would have changed nothing on screen. The
  constants' doc comments say so, and carry the previous values.
- Editor only, as the layer always was: `view` and `run --view` draw no grid, so
  nothing outside `growforge edit` changes. No other layer's colour is touched.
- Version 0.26.1.

## 2026-08-01 - 0.26.0 - Many thin members instead of one thick one

The user asked: "is there a way to make the algorithm prefer many structural
supports rather than just a flush surface?" There was not. A global
`mass_fraction` is met most cheaply by concentrating material, and the optimizer
did exactly that.

- **`[optimization.local_volume]`**, a cap on how much material any
  *neighbourhood* may hold, on top of the global target - Wu/Aage style infill
  optimization, and the bone-like networks that come with it. `max_fraction`
  (default 0.6, must sit above `mass_fraction`) and `radius_mm` (default three
  density filter radii, must sit above one). The local field is a second
  `DensityFilter` over the printed densities; the per-cell fractions are
  aggregated into one constraint by a p-mean, and its gradient travels back
  through the kernel's transpose and the density chain's, recomputed every
  iteration because the p-mean is nonlinear in the design even where the chain
  is not.
- **MMA takes a second constraint.** `Variable` carries a second pair of
  separable coefficients and the anchors an *evaluated* approximation needs, and
  the subproblem's dual is maximized over both multipliers by exact coordinate
  ascent - each coordinate the same bisection as before, the sweep bounded and
  the joint KKT conditions measured. The single-constraint path is untouched and
  takes the arithmetic it always did. Documented exception to the module's
  measure-the-real-one rule: with two constraints the dual is solved on both
  approximations, because re-measuring a kernel of 27 times the density filter's
  taps at every trial multiplier would cost more than the analysis.
- **Gating**, on the precedent the overhang and wireframe tables set:
  `update = "oc"` is refused with `update = "mma"` named as the remedy, and the
  growth engine refuses the table outright.
- **`[optimization] penalty`**, the SIMP exponent `p` of
  `E(x) = Emin + x^p (E0 - Emin)`, exposed as a flat key defaulting to the
  `SIMP_PENALTY` it was hard-coded to. At least 1, 2 to 5 documented as useful.
  `engine::simp_moduli` takes it as a parameter, so the optimizer, the post-run
  stress analysis and the benchmark cannot describe different structures.
- **Reporting**: the per-iteration line gains the *true* worst neighbourhood
  while a cap is active (the p-mean under-estimates it, and the run says so
  rather than hiding it), the problem summary a `local volume` row, the memory
  estimate the ten cell arrays the cap adds, and the stall note the reading a
  capped run needs - traversing at the move limit under a local cap is usually
  the converged character of these designs.
- Tests: finite differences of the constraint's value and gradient through the
  plain, overhang and forced-cell chains (1.5e-8 worst) and of the compliance at
  `penalty = 2` and `4` (1.2e-9), the dual solver slack and binding and
  deterministic, the behavioural claim measured - a block whose worst
  neighbourhood is 0.913 uncapped holds 0.716 under a 0.6 cap, with the
  aggregate exactly on it - the config round trips and both gates, the editor
  rows and the save-back, and one end-to-end capped run through the export.
- Version 0.26.0.

## 2026-07-31 - 0.25.0 - Every control in the panel says what it does

The user asked to "add tooltips to all of the properties section". The editor put
sixty-odd controls in front of the user and explained three of them; what
`kill mm` or `boundaries` did was in the README or nowhere.

- **Hover text on every editable control** of the side panel: the engine,
  resolution, material, optimization, growth and output sections, the properties
  of all five shape kinds, the domain `op`, a support's fixed axes, a load case's
  name and weight, every load's own fields, the snap row and `keep inside
  domain`. Each is condensed from the README's documentation of that key, units
  and defaults included.
- **The helpers require it.** `vec3_row`, `length_row`, `number_row`,
  `integer_row`, `optional_row`, `optional_integer_row`, `combo`, `combo_str` and
  `small_button` take the text as a non-optional parameter, so a row that
  explains nothing does not compile. `combo`/`combo_str` now keep the
  `InnerResponse` `show_ui` hands back, which is where the text hangs.
- **The layer switches carry theirs in the layer table**: `LayerInfo::help`,
  beside the label, so a layer cannot be added without saying what it draws - and
  `show` explains itself in the plain viewer as well as in the editor. A switch
  whose layer is empty says so on hover instead.
- The strings are literals beside the rows they belong to, as the existing
  `generate stl` and callout hints already were: the constants file is for
  tunable values, not for UI copy.
- **Fixed while writing the solver tooltip, and wrong since 0.10.0: the solver
  combo showed the wrong default.** It resolved an absent `[solver]` table
  through `Option::unwrap_or_default`, and `SolverBackend`'s derived default is
  the CPU - the *reference* backend the determinism promises are written
  against - while `solver_params()` resolves an absent table to
  `DEFAULT_SOLVER_BACKEND`, the compute device with a soft fall back. So a
  configuration that named no backend read `cpu` in the panel while every run
  under it asked for the GPU. Nothing was ever written by the display itself,
  but clicking `gpu` to correct it wrote the key - and a backend named in the
  file is an instruction that forfeits exactly that soft fall back, which is
  what "my GPU default got changed back somehow" looks like from the inside. The
  combo now shows what a run would use (`shown_backend`), and `SolverConfig`'s
  stale doc comment, which still claimed the CPU default, says what the two
  defaults are and why they differ.
- Tests: the panel and output-section smoke draws exercise every reshaped call
  site, a new scene test asserts every layer says what it draws, and a new panel
  test pins the solver combo to the resolved default for a keyless
  configuration, for a table with no key in it, and to the file's own value for
  a named backend.
- Version 0.25.0.

## 2026-07-31 - 0.24.0 - The editor is not stuck on the file it was launched with

The user asked for "a shortcut to the editor with file selection so i can open
any file or start new ones". Until now `edit` was bound to its argument for the
life of the window: another file meant closing it and relaunching.

- **`open` and `new` in the toolbar**, beside `save`, and `Ctrl+O` / `Ctrl+N`.
  Both raise the platform's own file dialog (`rfd`, unparented, filtered to
  `.toml`, starting in the current file's directory) and switch the session to
  what comes back. The dialog blocks the window and nothing else: the run behind
  it keeps going and drops the preview frames nobody collects.
- **One guard, three intents.** `CloseDecision`/`decide` now answer an `Intent` -
  `CloseWindow | OpenFile | NewFile` - so a switch asks the same modal, with the
  same save/discard/cancel, on the same computed `is_dirty()`. `close_modal`
  became `guard_modal` and names the file being moved to.
- **The swap is a whole new session**, built through the same constructors the
  command line uses (`Editor::open`, the new `Editor::create`). The old session
  is stopped *and joined* before the new one is constructed - never two workers
  behind one window - and the window then re-opens on it: new scene, camera
  refitted, title, directory, empty undo history, nothing retained.
- **`new` never becomes an open and never overwrites.** A path that is already
  there is refused where it is picked, with the reason on the status line, and
  refused again by the scaffold underneath.
- **One question at a time.** While the modal is up the pending intent is
  immutable - `request()` refuses everything, repeats included - and
  `shortcuts()` returns early for *every* binding, not a list of the dangerous
  ones: egui routes clicks by layer but keys by focus, so `Ctrl+O` used to reach
  past the modal and quietly turn "close?" into "open?", with the user's
  "discard" then answering a question they never read.
- **Structural fix behind all of it:** `ViewerApp` no longer watches a
  `RunProbe` captured when the window was built. `should_keep_pumping` asks the
  session the window holds *now*, so a swapped-in worker is the one the closed
  window outlives; `watching()` stays for `run --view`, whose run outlives the
  window by design. `edit()` takes no closure at all any more.
- Tests: the switch mid-run (old run stopped and joined, new session on the new
  path, nothing retained, keep-pumping following the new worker), the modal
  carrying the right intent through save/discard/cancel, the scaffold's refusal
  and its second gate, `Ctrl+O`/`Ctrl+N` reaching the buttons' own methods, the
  modal drawing for all three intents, the legend listing both bindings, and a
  switch to a file that will not parse leaving the document alone.
- `rfd 0.17.2` under the `viewer` feature; it takes the same `raw-window-handle
  0.6.2` the winit and wgpu stack already pins.
- Version 0.24.0.

## 2026-07-30 - 0.23.0 - The safety factor, in words

The user asked "what is the safety factor?" of the part in front of them and the
editor had no answer anywhere in text. The number was computed after every run,
painted into the von Mises layer, printed by `growforge run` and written to the
JSON - but an editor session could only look at colours.

- **Editor panel: a stress block**, under the pass notes and its own block rather
  than one more note. The safety factor is the headline -
  `safety factor 6.23 (peak 8.0269 MPa vs yield 50 MPa)` - with the peak of every
  load case under it. It describes the run the window is showing: absent until
  one has analysed the field it exported, absent again the moment the next run
  starts, and absent for a run that produced no report, so it can never be stale.
- **Editor console: the same lines**, `editor stress ...`, beside the
  `editor wrote ...` line - after a full run and after "generate stl" alike.
- **One formatter**, `StressReport::summary`, over the report's own
  `worst_safety_factor` and `max_mpa`: the panel and the console cannot quote
  different numbers, and neither can quote a number the console table does not.
  `print_stress_report` keeps its table.
- The summary describes the **exported** part: it is built from the analysis
  `complete()` hands back, which after a trim is the re-analysis.
- `Progress` gained `stress`, set by the viewer's `finish` through
  `ViewLink::set_stress_summary` - the path the trim and clamp notes take.
  `full` and `export_retained` now write their console lines to the session's
  console, which is stdout in a running editor and a buffer under test.
- Tests: the formatter over a two-case report, both no-factor branches and the
  headline pinned to the table's own `{:.2}`; the panel showing a finished run's
  factor, showing nothing for a run stopped before its analysis, showing the
  generation's afterwards, and dropping it when the next run starts; the console
  echo asserted out of `full` and `export_retained`, and silent when a generation
  is stopped.
- Version 0.23.0.

## 2026-07-30 - 0.22.0 - The bore in the file is the bore that was asked for

The user's report: the exported mesh clips **into** a keepout - a pin bore came
out under diameter. Measured on their own numbers, a 2.75 mm bore on a 1.5 mm
grid: the surface reaches 0.087 mm inside the cylinder. Cells are classified by
their centres and the extracted surface is then smoothed, so neither pass ever
sees the shape; supersampling cannot help, because the refined lattice only
interpolates the same coarse field.

- **`[output] boundaries = "exact"` (the default).** After the island cull and
  before validation, every exported vertex that lies inside a keepout or outside
  the domain is projected onto the analytic surface it violates, plus
  `BOUNDARY_CLAMP_EPS_MM` (10 nm) on the legal side. Where the part meets a bore,
  the bore is the cylinder the configuration wrote. `"voxel"` is the legacy
  escape hatch.
- **This changes the STL of an existing configuration**, deliberately: the same
  config exports a part whose outer faces sit on the domain instead of half a
  voxel past it, and whose keepout walls are the shapes rather than approximations
  of them. `boundaries = "voxel"` reproduces the old bytes.
- **Closed form where there is one.** Box, sphere, capped cylinder and tube
  (segment or arc) project onto their nearest surface point exactly;
  `Shape::nearest_surface_point` is where it lives, beside the signed distance it
  agrees with. The **ellipsoid** has no closed form - the nearest point is the
  root of a sextic - so it takes bounded Newton steps on its own field, which is
  exact on the surface. The **domain** is an ordered CSG composite with
  non-differentiable seams and takes a bounded descent onto its level set.
- **Bounded, and honest when it cannot.** Overlapping keepouts hand a vertex to
  one another, so the corrected position is re-examined up to
  `BOUNDARY_CLAMP_MAX_PASSES` times; a correction further than
  `BOUNDARY_CLAMP_MAX_DISPLACEMENT_VOXELS` (one voxel) is not the sub-voxel
  artefact this exists for and is refused. A vertex that is still illegal keeps
  the position it had and is counted in the report - never a silent claim, never
  a loop that does not end.
- **A self-overlapping tube is named rather than guessed at.** Such a tube is a
  legal solid with an exact signed distance, but its surface is no longer its
  centre line carried out by a radius, so walking a radius out from the nearest
  centre-line point lands *inside* it. There are two ways in, and `Arc::reach` is
  the smaller of them: the tube **folds** through the centre of its own bend
  (`radius >= arc radius`), or its two **ends close** on each other across the
  gap a major arc leaves open (`radius >= half the chord`, and only for a span of
  half a turn or more - below that the ends face each other *through* the arc,
  and a fat stubby arc with near-touching ends projects exactly, which is what a
  naive chord test gets wrong). `nearest_surface_point` refuses both instead of
  answering with a point that is still inside, so the clamp reaches its give-up
  in one pass; and a `[[keepout]]` holding one raises a configuration **warning**
  naming both causes, through the same channel every other warning uses, so that
  give-up count is diagnosable rather than mysterious.
- **`Problem` carries its boundaries.** `Problem::build` already built the domain
  CSG and the keepout union and dropped them after classification; it keeps them
  now. `export` reads them off the problem rather than off a configuration read
  again, because the editor's generate-stl button exports a **retained** problem
  and a second reading could describe geometry that problem was never built from.
  No signature of `export` or `complete` grew a geometry parameter.
- **Said in both places**, like the trim: `print_clamp_report` on the console
  beside the island line, and the editor panel's notes block, which now renders
  both passes under the same warning-colour rule.
- Editor: `boundaries` combo in the output section, saved back through the format
  preserving writer. `ShapeUnion` gained `signed_distance` (the minimum over its
  members) and `shapes()`.
- **The shipped `examples/` artifacts were regenerated** from this tree, because
  the default changes them and a repository must not ship a file its own binary
  would not produce. Every STL kept its exact triangle count - the clamp moves
  vertices and adds none - and every one now spans exactly its domain instead of
  a fraction of a voxel past it; the two stress JSONs moved only in the generator
  string and the last digits of the solver's own arithmetic.
- Tests: the projection of a point inside each closed-form shape, the ellipsoid's
  iteration converging inside its budget on three rotations, overlapping keepouts
  taking more than one pass, the give-up path leaving the vertex where it was, the
  displacement cap, the domain descent through a subtraction; a union measuring
  from its nearest member; end to end, a bore through a block whose every exported
  vertex is outside the keepout and inside the domain, against the same field
  under `"voxel"`, which demonstrably is not.
- Version 0.22.0.

## 2026-07-30 - 0.21.0 - Trim: the prongs that reach to nothing

The user's ask, in their words: "prongs that reach to nothing... gossamer
wisps... a pass that removes these after generation completes." It is
`[output] trim`, it is off unless a configuration asks for it, and it is the
same pass on both engines.

- **`trim = "stress"`.** After the run's own stress analysis and before the mesh
  is extracted, the part cells whose **envelope** von Mises stress - the maximum
  over every load case, already computed, so no extra solve - is below
  `TRIM_STRESS_FRACTION` (1 %) of the envelope's peak are removed.
  `[output] trim_stress_fraction` overrides the fraction, in the open interval
  (0, 1). Only material the report **measured** is ever removed: an element the
  stress pass left out carries a zero that is an absence of evidence, not a
  measurement of none, so below an `iso_level` of 0.5 the wisps the report does
  not cover survive - stated in the README rather than papered over.
- **Nothing declared is ever removed**: the cells of every support region, every
  `[[keepin]]` entry and every non-gravity load region, assembled exactly as the
  guide wireframe assembles its terminals.
- **All or nothing.** The purposes above are located on the field's connected
  bodies - six-face connectivity, as everywhere else - before and after the
  tentative removal. If a pair that shared a body no longer does, the whole pass
  is refused: the field is untouched, the run exports untrimmed, and the warning
  names both ends of what would have been separated. There is no partial trim.
- **The trimmed part is the one that is reported.** A pass that removed anything
  re-runs the cavity pass and the stress solve on the trimmed field, so the STL,
  the stress table, the safety factor, the JSON and the void report all describe
  what was written. The pre-trim stresses survive only as the two numbers in the
  note.
- **Said in both places.** The note - or the refusal - reaches the console beside
  the other reports (`print_trim_report`) and the editor's panel, under the
  layer switches, in the warning colour when it is one. The panel had no
  structural report line before this.
- **One post-run pipeline.** `optimize_and_export` and the viewer's `finish` now
  run the same `complete()`: analyse, trim, analyse again, export. The stage
  hooks and the stop question are a `Stages` trait, `Unwatched` for the command
  line and the window's status line for the viewer. The stress JSON is written
  once, past the last boundary a stop is honoured at, so "a stopped run leaves
  no file behind" covers the machine readable report as well as the STL - it did
  not while the write sat above that check.
- `Grid::cells_touching_nodes` is the one answer to "which cells is this region"
  for the growth domain, the wireframe and the trim's protected set.
- Editor: `trim` combo and the fraction row in the output section, both saved
  back through the format preserving writer.
- Not `[growth] prune`, which removes skeleton branches mid-run by whether they
  reached anything; the README says so in both sections.
- Tests: the spur removed and reported, a protected region below the threshold
  kept, a low-stress bridge refusing the pass, the multi-case envelope, the
  fraction override, a diagonal touch as no joint, both policy paths; the
  pipeline reporting the part it exported rather than the one it judged, and a
  refusal writing the untrimmed file byte for byte; the note reaching the window
  and the outcome alike; the panel's rows and its note; and an unpruned grown
  canopy losing its stubs end to end.
- Version 0.21.0.

## 2026-07-30 - 0.20.0 - Click two points to make a line

The other half of the user's sentence. 0.19.0 shipped the shape and the handle
that curves it - "drag the middle to make a curve" - and said that placing one by
clicking its two points was the next step. This is that step, and the flow now
reads as it was asked for: **click the `tube` button, click where it starts,
click where it ends, drag the middle.**

- **The `tube` button opens a mode rather than adding an object.** In all four
  add rows - domain, keepout, keepin, supports - it now arms a placement for
  *that* row; every other kind is dropped at the centre of the domain exactly as
  before. The two clicks that follow are the tube's ends, and what is left behind
  is a straight tube in that row's list, selected, with its bend handle in the
  middle of it. Clicking the same button again cancels; another row's button
  starts that row's placement instead.
- **A click lands on what it was aimed at.** The click's ray takes the nearest
  surface it meets - any object's, the design space's own included, which is the
  surface most of a model is drawn against - and falls through to the ruled floor
  plane when it meets none. Both land on the snap increment, and `Alt` frees them
  the way it frees a drag. A click that meets neither, at the sky or past the
  ruling, does nothing at all rather than putting a point at an arbitrary
  distance along a ray that was aimed at nothing.
- **The mode owns the viewport's clicks while it has them.** Selecting by
  clicking is suspended, no gizmo handle can be grabbed, and nothing of the
  selection is drawn - a handle that cannot be grabbed is a handle that lies
  about what a press would do. The selection itself is kept: `Esc` at either
  stage leaves the mode and hands its overlays straight back. Clicks are confined
  to the 3D view like every other click, and a second click on the first point is
  ignored rather than written as a tube of no length.
- **A structural change of the document leaves the mode too.** Deleting the
  selected object, undoing or redoing while points are being clicked cancels the
  placement first: the selection the mode promised to hand back may not be there
  any more, and re-entering costs one button click. A number typed into the
  properties panel does not - a clicked point is a position in the world rather
  than a reference to an object, and nothing typed can invalidate it. The panel's
  delete button and the `Delete` key are now one path for exactly this reason.
- **What it draws while it waits.** A marker where the next click would land, and
  once the first point is down, the tube those two points would make, at the very
  radius the placed one will have. The panel says which list is being placed into
  and which click is next.
- **One size rule for a new tube.** `VIEW_EDIT_NEW_TUBE_RADIUS_FRACTION` is now
  the single answer to how thick a new tube is, whether it was added from the row
  or placed by two clicks, and a placement is committed through the same
  containment gate as a drag or a typed number - so a tube placed on the lid of
  the domain comes back inside it, and the panel says so.
- Tests: the landing against surfaces, the floor and the sky, snapped and
  bypassed, with a subtracted domain entry excluded; the whole flow from each of
  the four add rows, into the right list with the right radius; `Esc` at both
  stages and the button as a toggle; a click on an object placing rather than
  selecting, with the same click outside the mode selecting; the clicks that are
  ignored - nowhere, coincident, outside the view; containment on and off; the
  panel and the preview drawn at both stages. And the gap round A's review left:
  a **bend** drag pushed through the lid of the domain with containment on, whose
  handle and callout still read the 14 mm it committed rather than the 15 it was
  asked for.
- Version 0.20.0.

## 2026-07-30 - 0.19.0 - The tube

Asked for by the user, who wanted curved domains and curved elements and put it
as plainly as it can be put: "click two points to make a line then drag the
middle to make a curve". Everything growforge could describe until now was
straight or round-about-a-centre; a curved run of material had to be faked out
of a chain of cylinders, each with its own flat caps and its own seam. There is
now a **tube**: two end points, a radius, and an optional bend point. Unbent it
is a capsule - the round-ended sibling of the cylinder, not a variant of it -
and with a bend it is the circular arc through the three points. It is legal
wherever a shape is: a `[[domain]]` entry either way, a `[[keepout]]`, a
`[[keepin]]`, a support region, a load region.

This is the shape and the handle that curves it. **Placing one by clicking two
points is the next step** and is not in this release; a tube is added from the
add rows like every other kind and then dragged into place.

- **The field is the exact signed distance, everywhere.** A tube is everything
  within its radius of its centre line, so its field is the distance to that
  curve minus the radius - and the distance to a segment and to a circular arc
  both have closed forms. That is a stronger guarantee than the ellipsoid's,
  whose magnitude away from the surface is only a bound: the arc's distance
  splits into the circle's own plane and the offset out of it, takes the point
  at the sample's own angle when that angle lies within the arc, and the nearer
  end when it does not. The bounds are exact for the same reason - the extremes
  of the curve, which are its ends and the points where its tangent turns
  perpendicular to an axis, grown by the radius.
- **The bend is solved as a circumcircle, and a shallow one is no bend at all.**
  A bend within `TUBE_COLLINEAR_EPS_MM` (a micron) of the line through the two
  ends is *legal* and is the straight tube, bit for bit the arithmetic a tube
  carrying no bend takes: at that depth the arc departs from the chord by less
  than a micron anywhere along it, three orders of magnitude under the finest
  voxel, while the circle it would otherwise be solved into has a radius of
  millions of millimetres and a plane normal made of rounding error.
- **The middle handle is the feature.** The gizmo of a tube carries both end
  handles, a radius handle on the middle of its curve, and a **bend handle** in
  the middle: on a straight tube it sits exactly where every other shape's
  camera-plane handle sits, and dragging it curves the tube. Two markers in one
  place is one click with one winner, so the bend is drawn and grabbed a quarter
  larger than the other cubes and in its own colour - it is the volume the ray
  enters first and the cube that can be seen. Bend the tube and the handle
  leaves the centre, and the translation handle is the centre's again. Dragging
  the bend back onto the line between the two ends **straightens** the tube and
  drops the key; the properties panel has a **straighten** button for doing it
  without aiming.
- **Picking is a bounded sphere trace rather than a closed form.** A bent tube is
  not convex - a ray can enter it, leave it, cross the gap its arc encloses and
  enter it again - so it is traced against its own exact field: clipped to its
  bounding box, stepped by the distance to the surface, which can never step
  through it, and stopped at a tenth of a micron or after
  `VIEW_EDIT_TUBE_PICK_MAX_STEPS`. Deterministic and bounded, with no wall clock
  and no randomness in it.
- **The overlay is one closed surface.** The rings of the end caps are built in
  the same frame the barrel's are, at a tilt that walks from the equator to the
  pole, so a tube is a single sweep rather than a barrel with two spheres
  dropped on it - and every vertex of it is exactly on the surface the field
  describes. A load indicator on one hangs off the middle of its *curve*, which
  on a bent tube is not the middle of the line between its ends.
- **`min_extent` is its radius**, as a sphere's and an ellipsoid's are: a capsule
  of no length is a ball rather than a degenerate solid, so the radius is the
  only measurement of it that can vanish. That is what makes it unlike the
  cylinder, which answers with the smaller of its radius and its length because
  a cylinder of no length is a disc.
- Tests: the capsule's own distances against the cylinder's flat caps; a bend on
  the axis reproducing the straight tube to the bit; the circumcircle through
  three points, ends and bend on the arc, over four triples including one that
  goes the long way round; the field against ten thousand sampled arc points at
  385 places; the exact bounds against a brute-force sweep of the surface; a tube
  legal in every shape position with its bend optional; a radius that is not
  positive named, and a `nan` in the bend caught; a round trip through the
  format-preserving writer, bend appearing, straightening and gone; the bend
  drag itself - which is the arm the drag dispatcher's wildcard would otherwise
  swallow - the middle of a straight tube grabbed as its bend, an end drag
  keeping the bend, a turn carrying it round; the callout for the pull of a bend
  and the span of the ends, typed both ways; the traced hit against a marched
  one over three shapes, five directions and seven offsets, and a ray through the
  concave gap missing it both ways; the mesh closed with every vertex on the
  surface; and both engines routing round a bent tube keepout, held by a tube
  support region.
- Version 0.19.0.

## 2026-07-29 - 0.18.1 - Carry-path re-measurement

Documentation only. The 0.13.0 entry recorded that forcing the shader's remaining
compensation carries "made the pass count vary between runs", which has been read
since as a latent defect in the reduction carry path. It was a confounded
measurement: the host arithmetic around the device solve was never controlled
for. Re-measured on the same machine, one cell at a time: 12 repeats each on the
two skeletal fixtures, 8 in the floor regime.

- **The device is a pure function of its input bits.** With
  `RAYON_NUM_THREADS=1` the shipped shader and an all-guarded one are both
  bit-stable over 12 repeats of the same solve - same device iterations, same
  pass count, same solution. 448 of 448 recorded refinement passes and 96 of 96
  device-only probes on a fixed right hand side reproduced exactly, both ways.
- **The variance is the host's parallel `norm`.** With rayon free *both* variants
  vary: the shipped shader spends 3750 to 4900 device iterations over 4 or 5
  passes on the 192 000 cell cantilever, because `solver::norm` sums in a
  different order and the narrowed right hand side of the third pass onwards
  differs in its last bits. The README's reproducibility section already said as
  much; the carry attribution was wrong.
- **The 16 % cost did not reproduce.** All-guarded was 6 to 29 % *cheaper* in
  device iterations on the two skeletal fixtures measured. The real price of the
  live carries is roughly 2 to 4 % of the wall time of a device iteration - the
  kernel is memory bound - plus, in the 1e9 contrast floor regime, twenty times
  more wasted device work before the CPU fallback takes over.
- **No shader change.** The all-guarded variant does not ship: it loses in the
  floor regime, and what a shader compiler does to a compensated sum is a
  property of the driver. `pcg.wgsl`, `fea::gpu` and the README carry the
  measured position instead of the old claim.
- Version 0.18.1.

## 2026-07-29 - 0.18.0 - "generate stl": the design on screen, on demand

Asked for by the user after stopping a run at iteration 527 and finding that
stopping it wrote nothing at all: the design was on screen, hours of solving were
behind it, and there was no way to have it as a file. There is now a **generate
stl** button in the editor toolbar. Stopping still writes nothing on its own -
that policy is untouched - and asking is what writes.

- **Every run keeps its newest design.** The worker holds one slot: the
  full-resolution field of the last iteration any run reported, beside the
  `Problem` that run was built from. It is fed by the very per-iteration callback
  the preview mesher is fed by, so it needs nothing the run was not already
  reporting: it costs one copy of the field per reported iteration - beside the one
  the preview snapshot already makes - and one field held, both of which are lost
  in the noise of the solve they sit next to. It is replaced whole when the next run
  starts, previews included, because what the window is showing is what the button
  has to export.
- **A superseded run cannot write into its replacement's slot.** A run is
  superseded by cancelling it and then clearing the slot, and cancellation is
  cooperative, so a cancelled run may still have an iteration in flight. Retention
  therefore answers "is this run still wanted" *inside the slot's own lock*, the
  way `LatestSlot::push` answers "is this channel still open": a check made before
  taking the lock could pass and then write behind the clear, leaving a design of
  the previous configuration in the slot the new run is repopulating - and the
  panel offering it.
- **The pair is what is exported, never a rebuilt problem.** Grid, material,
  policies and output path all come from the kept problem, so a configuration
  edited since the run - even one edited until it no longer builds - cannot export
  a field against a grid it does not belong to. The button stays offered through
  any edit for the same reason.
- **The same deliverables a run produces, because it is the same call.**
  `run_worker`'s tail is now `viewer::finish`: the cavity pass, the stress solve,
  the culled mesh, the STL, the final frame and the `Finished` status. Both the
  full pipeline and the on-demand export go through it, so the file, the surface
  in the viewport and the stress colouring are the ones the run would have
  produced from that field, and the write is recorded the same way - the editor
  goes on telling its own output from somebody else's.
- **An iteration that finished after the stop is not kept.** The snapshot channel
  closes on the same click, so the last design the window was *sent* is the last
  design that is kept: what is exported is what was on screen when the stop was
  pressed, not what the thread computed on its way to its next checkpoint.
- **One writer at a time, and never a button that does nothing.** A full run and
  a generation both own the output file, so each disables the other, an
  auto-regrow preview is deferred rather than allowed to take the worker from
  either, and every state in which clicking would do nothing renders the button
  disabled instead. The generation registers with `RunProbe` like every other run
  thread, so a window cannot close over a write in flight, and **stop** ends it
  through the same cooperative seam - before anything is written.
- Tests: a finished run's design generated after its own file was deleted, giving
  back the same triangle count in the frame, the status and the file, with the
  stress layer the panel's switch hangs off; the user's own case - a 400-iteration
  run stopped mid-flight, writing nothing, then generated into a real part of
  plausible volume; a preview's design generated, which is what makes the button
  what is on screen; nothing kept being a disabled button and a guarded no-op; a
  generation and a full run never owning the worker at once, with a preview taking
  it from neither; a generation refused as a kind of run started from a
  configuration; the retention guard refusing a cancelled run's field and leaving a
  cleared slot cleared; the toolbar drawn in both states; and the editor path end to
  end, status line, frame label, stress layer and file.
- Version 0.18.0.

## 2026-07-29 - 0.17.0 - The guide wireframe

Asked for by the user, for a part whose keepins the optimizer kept leaving
stranded: *"a thin wireframe between the keepins that the simp can follow ...
avoids keepouts and draws the quickest possible path to each as a guide only ...
toggleable"*. A SIMP run starts from a uniform field, so the first iterations are
spent discovering a load path the configuration already implies; where the regions
are far apart or separated by a keepout, what it discovers instead is a design that
serves some of them and abandons the rest.

- **`[optimization.wireframe]`** switches it on, and its presence is the switch,
  like `[optimization.overhang]`. `radius_mm`, `hold_iterations` and
  `seed_density` are all optional, all defaulted from `constants.rs`. SIMP only:
  `engine = "growth"` rejects the table with the reason, exactly as it rejects an
  overhang constraint, because a growth run already routes a strut from every load
  to the supports.
- **One network through every region the problem declares.** At setup a shortest
  keepout-avoiding path is routed between every support region, every `[[keepin]]`
  **entry** and every load region until all of them hang off one wire. The routing
  is the growth engine's own, used from the outside - the same A* over the 26
  neighbours with Euclidean edge costs, the same shortcut pass, the same capsule
  rasterizer - so keepouts and the space outside the domain are impassable and the
  wire goes around them or not at all. The order is fixed (supports, keepins,
  loads, each in configuration order), so the guide is a function of the
  configuration alone.
- **Keepin entries are resolved one at a time**, next to the merged union the
  classifier already pins solid: `Problem::keepins` carries the cells of each
  entry. Two disjoint pads are two places to reach, and wiring their union would
  have left one of them unwired - which is the defect that prompted the feature.
- **Seeded, held, released.** The wire's design cells are raised to `seed_density`
  in the initial design variables (upwards only, and never a void or forced solid
  cell), and for `hold_iterations` iterations the same cells are floored again
  after each update step. Then the floor comes off and is announced. Nothing is
  pinned and no sensitivity is touched, so from there the optimizer may keep the
  wire, thicken it, move it or dissolve it.
- **A run cannot decide it has finished while the floor is on.** A floored design
  is not free - the change an iteration reports is partly the floor pushing its
  cells back up - so the loop asks neither the convergence test nor the stall
  criterion until the release, and the stall window fills with free iterations
  alone. Without that, a tolerance above the update's move limit made *every*
  iteration look settled and the run exported a forced wire on iteration one; now
  it converges on the first free iteration instead, at the volume target. Only the
  iteration cap and a cancellation can end a run inside the hold window, and both
  say what the design carries: *"the run stopped after N iterations with the
  guide's floor still active ... the design it ended on carries the wire as forced
  material rather than as a guide"*. A `hold_iterations` that cannot release inside
  `max_iterations` is warned about at setup as well, where the misconfiguration is
  visible before any time is spent.
- **The floored step is the reported step.** The floor goes on after the update
  has bisected its multiplier onto `mass_fraction`, so during the hold the
  realized volume fraction runs above the target by whatever the wire adds. That
  is restated rather than hidden: the volume fraction and the design variable
  change the iteration line shows are recomputed on the floored design, which is
  the one the next iteration analyses. It self-corrects in the first iteration
  after the release.
- **A region that cannot be reached is named and skipped**, and the rest of the
  network is built without it; so is one that holds no material at all, and a
  problem with nothing to wire says so and runs unguided. No configuration is
  rejected for the shape of its geometry.
- **The editor carries it**: a *guide wireframe* checkbox in the optimization
  section with the three keys under it, each showing its default until pinned, and
  a format-preserving save that adds and removes the table without touching
  anything around it.
- Tests: the wire routed through the one gap in a wall, joining its regions into
  one 26-connected body and entering no keepout; two disjoint keepin pads each
  reached, which is what per-entry resolution exists for; an unreachable pad and an
  unreachable load both reported with the rest still wired, and the same for a pad
  buried in a keepout, which holds no material to reach; the same
  configuration built twice, cell for cell identical; the seed raising only design
  cells and only upwards; the floor held and then released, with the volume
  fraction, the change and the printed densities all restated on the floored
  design and the multiplier left alone; a hold that outlasts its budget reported;
  through the loop, a guided run heavy for its hold window, on target from the
  release, and a different design from the unguided one; a tolerance above the move
  limit refusing to settle until the release and then settling on target, where the
  same problem unguided settles on iteration one; a hold longer than the whole
  stall window deferring both verdicts, with the held iterations shown to have
  looked settled while it did; a run capped inside its hold window reporting the
  forced wire it ended on; and the table's defaults, its rejections and its round
  trip through a save.
- Version 0.17.0.

## 2026-07-29 - 0.16.1 - Two editor interaction defects

Both reported by the user, mid-design on a real part: *"the resize boxes and the
move arrows are overlapping causing the boxes to not be able to be dragged or
selected"* and *"im trying to move it past the edge via the editor but it stops
when it hits the edge"* - the second with **keep inside domain** already unticked.

- **A cube buried in an arrow now takes the press.** `gizmo::grab` ranks in two
  steps: what the press landed on (nearest hit, arcs excluded from that contest as
  before), then a handle of a lower `grab_rank` whose grab point lies *inside*
  that volume. Tier 0 is every marker cube - the face, corner, cap and radius
  handles and the centre handle - against the arrows at 1 and the arcs at 2. All
  of them were unreachable where they overlap: a box's `+x` face handle sits at
  half its width against an arrow reaching three quarters of its half diagonal,
  and the centre cube is where all three shafts begin, so the ray always entered
  an arrow first. Burial is a property of the two handles rather than of the ray,
  so a press aimed at an arrow's tip still keeps the arrow when a corner cube of
  the object lies further along the same ray, and half way along a shaft is still
  the arrow.
- **A drag now follows the pointer off the 3D view.** Containment was never what
  stopped the second one: the view is fitted to the model, so the pointer position
  that asks for an object outside the domain is usually outside the viewport too -
  over the side panel or off the window - and every such `CursorMoved` was
  dropped, freezing the drag at about the wall whether containment was on or off.
  A drag in progress owns the pointer and is fed a ray wherever it is; a press, a
  release and the hover are still confined to the 3D view, where there is
  something to aim at.
- **The camera's delta reference now follows the pointer through a gizmo drag.** A
  drag claims every pointer move it sees, so the camera path never ran for any of
  them and the point its next orbit or pan measures from stood still: a camera
  gesture around a drag snapped by everything that drag had covered. Every position
  a drag claims is now written to `Pointer::last` as it passes - unless a camera
  gesture is itself live, the one case where the camera path still runs and needs
  the reference it has not consumed yet. That single write is what holds the
  invariant up, before *and* after the drag; the catch-ups at the grab and the
  release are belt and braces on it. No camera arithmetic changed.
- Tests: the buried face, centre, sphere-radius and ellipsoid short-axis cases and
  the behind-the-tip counter-case in `gizmo`; through the whole input path at
  three scale factors, a face handle inside an arrow resizing rather than moving,
  the centre cube grabbed where the shafts meet with the arrows still grabbed
  along them, an ellipsoid's short semi-axis dragged, and every shape kind leaving
  the domain with containment off (asserting the pointer really does leave the
  view) against the same gestures still clamped with it on; and the same four
  pixel pan moving the camera the same distance on a clean window as on one where
  a gizmo drag has just ended and on one where a gizmo drag is still held.
- Version 0.16.1.

## 2026-07-29 - 0.16.0 - The ellipsoid

Asked for by the user, for a part in hand: a central ellipsoid keepout between
two rod clamps, approximated with a box until now.

- **`shape = "ellipsoid"`**, legal wherever a shape is - `[[domain]]`,
  `[[keepout]]`, `[[keepin]]`, support and load regions - with
  `center = [x, y, z]`, `radii = [rx, ry, rz]` (semi-axes, all positive and
  finite, rejected by component when they are not) and the box's own optional
  `rotation_deg`: about the shape's centre, extrinsic XYZ, the same helpers.
- **Its field is the scaled-space one**, `(|q / r| - 1) * min(r)`. The zero
  level set is exact, which is what every consumer reads - classification samples
  cell centres, region selection samples nodes, the CSG and containment read the
  sign - and the magnitude away from the surface is a lower bound rather than the
  distance. Documented on the variant, in the README, and exact for a sphere.
- **The bounding box of a turned ellipsoid is exact**, by its support function:
  a half extent of `sqrt(sum_j (R[i][j] * r_j)^2)` per axis. Unrotated it is
  `center +- radii`, to the bit.
- **The editor treats it as a first class shape**: an exact analytic
  ray-ellipsoid pick (solved in the scaled frame, the parameter returned in
  millimetres along the world ray), a radius handle per semi-axis in the shape's
  own frame, the rotation arcs, centre/radii/rotation fields, hover and selection
  shells from the tessellation, containment by the turned bounding box, snapping
  and callouts on the dragged radius, and the add menu.
- **Overlays** tessellate as the sphere's own lat/long mesh, stretched by the
  radii and turned, at the sphere's segment and ring counts.
- **Files round trip.** `radii` is written where the sphere's `radius` was, an
  absent rotation stays absent, a shape that stops being an ellipsoid drops both,
  and the shipped examples are byte identical.
- Version 0.16.0. `HandleKind::Radius` and `gizmo::radius_of`/`resize_radius`
  now carry which radius they mean, which is 0 for a sphere and a cylinder.

## 2026-07-29 - 0.15.0 review closeout

- stall.rs module doc bullet stated the change-decay condition with zero
  tolerance; aligned with the function doc and constants (the second half's
  smallest step may sit up to `STALL_CHANGE_DECAY` below the first half's).

## 2026-07-29 - 0.15.0 - Run to convergence, and stop when it will not come

Asked for by the user: *"set the default iteration limit to always run to
convergence - 150 is too low"*. In the skeletal regime this tool is for
(`mass_fraction` near 0.10) 150 is not a budget, it is a truncation - their
1.5 mm cantilever is still descending at 150 and only settles at **iteration
682**, on a compliance 14 % below the one the old default handed them (2.194e1
against 2.562e1), in 2 min 52 s on the device. But this project also has
documented proof that some problems *never* settle, so a raw unbounded default
would hang exactly those runs forever. The budget is therefore raised **and** the
loop learns to recognise the runs that would never spend it.

- **`[optimization] max_iterations` now defaults to 1000** (was 150). It is a
  budget rather than a target: what makes it affordable is the compute backend.
  The shipped `cantilever.toml` spends ~10 s on its 150 iterations; the user's
  1.5 mm skeletal cantilever (58 000 cells) spends 2 min 52 s on all 682 it
  needed, where 150 would have been 38 s of it; their 1.0 mm run at 192 000 cells
  is the expensive end, at ~3 min per 150 iterations (0.13.0), so a full budget
  there is some twenty minutes. An explicit `max_iterations` still wins to the
  iteration, which is
  what leaves every shipped example and every recorded trajectory untouched, and
  the editor's fast preview keeps its own cap of 20.
- **A run that has stopped making progress now stops itself.** The criterion is
  over a window of 100 iterations and asks two things of every one of them:
  *every* step in the window was clipped to at least 80 % of the update scheme's
  move limit (and the second half's smallest step is no more than 5 % below the
  first half's), **and** the best compliance of the second half betters the best
  of the first half by less than 0.5 %. Best-of-half rather than endpoints,
  because a descending run keeps setting new lows through its own oscillation and
  a wandering one stops setting them.
- **The thresholds are measured, not guessed.** Six runs were traced iteration by
  iteration with the test switched off; each figure is that run's *worst*
  100-iteration window, the one that came closest to being called a stall:

  | run | outcome | smallest change held for a whole window | what it bought |
  | --- | --- | --- | --- |
  | 120 mm deep shelf bracket, `oc` | 400 = the cap | **100 %** of the move limit | +0.2 % |
  | the same bracket, `mma` | converged at 267 | 57 % | +0.4 % |
  | 1.5 mm skeletal cantilever, `oc` | converged at 682 | 53 % | +3.3 % |
  | `mbb_bridge.toml` | 150 = its cap | 29 % | +0.5 % |
  | `cantilever.toml` | 150 = its cap | 15 % | +0.3 % |
  | `shelf_bracket.toml`, `mma` | converged at 132 | 14 % | +0.3 % |

  The separation is the move limit column, not the compliance column: a run that
  will not settle spends every iteration clipped, and a run working towards an
  answer lets the change off the limit long before it arrives. A compliance-only
  test was written first and is wrong - it stopped `cantilever.toml` at 140 and,
  worse, `shelf_bracket.toml` at 115, seventeen iterations before it converges.
  Replayed over the traces the shipped criterion fires on the `oc` deep bracket
  alone (at 255); run live it stopped that bracket at 361 of its 400 and left the
  other five alone.
- **The bias is asymmetric on purpose.** A missed stall costs the rest of the
  budget - which is exactly what growforge did before this existed - while a
  false stall costs the design the run was on its way to. Anything the test
  cannot read clearly (a window that is not full, a non-finite number) reads as
  "not stalled".
- **Three stop reasons, reported apart.** `DensityField::converged: bool` becomes
  `DensityField::stop: StopReason` - `Converged | Stalled | IterationCap |
  Cancelled` - and the run summary names which: `iterations 361 (stalled)`. The
  engine prints the sentence that says what to do about it, scheme aware: under
  `oc` it points at `update = "mma"`, which damps exactly the variables that keep
  crossing the box, and under `mma` it says to raise `max_iterations` or to take
  the iterate. `Cancelled` is not a fourth outcome, it is the editor's stop
  button, which exports nothing; the growth engine maps its own natural end to
  `Converged` and its step cap to `IterationCap`, so its summary line is
  unchanged.
- **A stalled run is a finished run.** It exports, is stress analysed, writes its
  STL and exits zero, identically to a capped one. The only difference is the
  sentence printed about it, and that sentence reaches the console, the run
  summary, the `run --view` panel and - new - the editor's own panel, which now
  shows the engine's last word under its run line.
- **Nothing that was pinned moved.** Every shipped example sets `max_iterations`
  explicitly and re-runs to its recorded numbers: `cantilever.toml` 150 at
  7.850071e0, `mbb_bridge.toml` 150 at 1.858260e2, `shelf_bracket.toml` converged
  at 132 on 1.879277e1, `growth_canopy.toml` 24 steps. The recorded trajectory
  tests, the growth determinism hashes and the editor's preview cap are untouched.
- New `engine::stall` module (the criterion as a pure function over a window, the
  rolling watch the loop keeps, and the message), `UpdateScheme::move_limit`, and
  four constants with the measurements behind them. 13 new tests: the criterion's
  legs and boundaries, the traced never-settler and the traced slow converger, a
  loop that stalls on a miniature of the deep bracket at iteration 123 and one
  that converges at 134 with the test armed and watching, all three stop reasons
  through `RunOutcome`, the default budget flip, and the stall note reaching the
  panel.
- README: new "When a run stops" section (the three reasons, the criterion, the
  evidence table), and the timing table, tuning advice, editor panel table and
  configuration reference updated for the new default.

## 2026-07-29 - 0.14.0 - mesh level island culling

Reported by the user with screenshots: a converged SIMP run at
`mass_fraction = 0.12` printed `solid bodies 1 connected body` while the
exported STL visibly carried small floating shells. The report was not lying, it
was describing a different object: the body count flood fills the *cell* grid at
the iso level, while the surface is extracted from the *node* averages of that
field. The two disagree in both directions - a lone dense cell makes no surface
at all, and a clump joined to the part only through a one-cell-wide bridge makes
a surface the bridge cannot reach - so the connectivity claim never covered what
shipped.

- **`[output] islands = "cull" | "keep"`,** default `cull`. After extraction,
  welding and smoothing, and *before* validation and the STL write, the mesh is
  partitioned into connected components by shared vertices (union-find in
  triangle order, so the labelling is deterministic). `keep` exports the
  extracted surface, fragments and all.
- **A component is kept for its purpose, never its size.** It survives when it
  touches a support region, a load region or a keepin cell, and is culled when it
  touches none of the three: geometry serving nothing the configuration declared
  is debris, and geometry serving something declared is deliberate however small
  it is. Ranking by volume was written first and is wrong - a keepin boss rides on
  top of `mass_fraction` and can outweigh the structure, and on a two-lobe
  fixture the largest-wins rule shipped an STL holding the boss with the entire
  load-carrying structure culled out of it, exit code zero and a stress table for
  material that was not in the file. Volume now orders the report and decides
  nothing. Two floors: nothing is culled when no component reaches anything
  declared, and nothing under `keep`.
- **"Touches" is asked two ways,** because neither alone covers both a thin part
  and a thick one: a vertex of the component lies in a lattice cell one of the
  region's nodes is a corner of, or the component *encloses* one of the region's
  probes (cell centres inside the material it selected). The shipped
  `cantilever.toml` needs the second - its load region sits in the middle of a
  keepin pad, several voxels from the nearest surface - and reported a false
  unreached warning with only the first. The probes are distributed through the
  region in space, one per occupied box of a partition of its bounding box, with
  every axis that has an extent split before any axis is split twice. Both
  halves of that are defects walked into and now guarded: taking every n-th cell
  of the raster-ordered list aliases against regular rows (a pad split left and
  right by a gap, 21 cells per row and 168 of them, puts all eight probes in the
  left patch), and growing the partition widest-first alone starves an axis (a
  pad 100 cells wide and 12 tall spends every box on x and probes one row of y).
  Either would leave a patch of a region unprobed and, if nothing else anchored
  the component holding it, culled. Stated limitation, in the same register as
  the tiny-body note: a gap narrower than one box can still lie inside one, at
  any finite probe count, which is why containment is the second leg of the
  anchor test and why the unserved-region warning exists.
- **A body that is anchored but tiny is named.** Any exported body smaller than
  the sphere of diameter `min_feature_mm` (scaled by `ISLAND_TINY_BODY_SPHERES`),
  when more than one body ships, prints a warning and the design-level remedies.
  It is still exported: a declared region asked for the material in it, and size
  culls nothing. What it prevents is finding a loose stub in a slicer instead.
- **A region no exported body reaches is named.** After the cull, every support
  and load region that nothing in the shipped surface touches gets a warning line
  of its own - the "your structure does not reach its load" case, which nothing
  said out loud before.
- **And every culled fragment that was inside a declared region is named too.**
  The line above cannot cover the partial loss: a region spanning two
  disconnected lobes stays *served* by the lobe that survives, so it never reads
  as unserved while the lobe that leaves takes material the region asked for with
  it. Each fragment about to be removed is therefore tested against the region
  shapes themselves - the SDFs the configuration is written in - and names both
  itself and the region; the fragment note drops its "nothing declared asked for
  it" wording when it does, because there it would be false.
- **Cavity shells are attributed, not guessed.** An inward shell is the inside of
  a cavity; which solid it belongs to is decided by the innermost outward shell
  whose solid angle sum encloses it, and it survives exactly when that shell
  does. So `voids = "warn"` keeps its inner shell, a fragment's own cavity leaves
  with the fragment, and the two policies cannot contradict each other.
- **Two lines, each naming its object.** `field bodies` is the cell level flood
  fill of the density field; `mesh bodies` is the connected components of the
  surface that was written, each body saying what anchors it, with the cavity
  shells and the culled fragments and their volumes beside it. The field level
  warning no longer claims a second body carries no load: it says which line
  decides that.
- **An export with nothing to remove is untouched, byte for byte.** Nothing is
  rebuilt when every component is anchored, so both shipped growth examples still
  write the bytes they did (canopy 401 segments / 22668 triangles / safety 14.54,
  canopy_symmetric 552 / 16.62 - verified by comparing the files), and the
  cantilever's STL is byte identical to the pre-culling pipeline's.
- Measured on a skeletal multi-load arm at `mass_fraction = 0.08`: the surface
  came out in 43 components; 38 that reach nothing declared were culled and the
  five that hold the support, the load or the keepin pad shipped.
- Viewer and editor: the final frame is the culled mesh, because it is the mesh
  `export` returns; the editor's output panel gains the `islands` dropdown and
  saves the key back. 35 new tests.

## 2026-07-29 - 0.13.0 review closeout

- README: one sentence in "What single precision cannot do" noting the per-solve
  CPU finish does not conflict with the named-backend-is-an-instruction rule
  (the device was opened and runs the run; one solve inside it completed at
  higher precision).

## 2026-07-29 - 0.13.0 - GPU solver in the skeletal regime

Reported by the user: at `mass_fraction = 0.10` the GPU backend stopped being
usable and their runs were taking 70 minutes on the CPU instead of ~1 on the
device. Two distinct failures, one root cause each.

- **The solution accumulator now carries a compensation limb.** The device's `x`
  was a plain f32 running sum of hundreds to thousands of conjugate gradient
  steps, and its accumulated rounding, not the residual, is what bounds how good
  a correction the host can be handed: measured on a 192k cell cantilever at
  `mass_fraction = 0.10`, the device reported a converged inner residual of
  6.7e-6 while the correction it returned left a *true* relative residual of
  1.388 - above the 1.0 of the zero start, so the refinement loop was being
  pushed backwards and gave up. A Neumaier second limb (`xlo`, read back and
  folded in f64) takes that to 4.5e-3 per pass. The user's 1.0 mm run now
  finishes all 150 iterations on the device in **3 min 4 s**.
- **A compensated sum has to be hidden from the shader compiler.** The carry is
  algebraically zero, and the Vulkan compiler on an RTX 3080 duly proved it zero
  and deleted it - from *every* compensation in `pcg.wgsl`, silently, since they
  were written. The accumulator reads its sum back through `GPU_CARRY_GUARD`, a
  uniform that is exactly 1.0, which the compiler cannot see the value of. The
  element gather and the reductions are left as the driver takes them: forcing
  their carries too was measured to cost 16 % more device iterations and to make
  the pass count vary between runs, and the centred gather and the fixed
  reduction shape already carry those sums.
- **A refinement pass is a trial.** The fold is undone again if the f64 residual
  says the correction was not one, so a solve can no longer return - or fall back
  from - an `x` worse than the one it was given.
- **A solve single precision cannot do is finished on the CPU, loudly.** After an
  aggressive first `mma` step a third of the cells sit at exactly `Emin`, whole
  regions are held together by nothing but the stiffness floor, and the exact
  solution *rounded to f32* leaves a residual 200 times the right hand side;
  device side modulus floors, symmetric diagonal scaling, an extended precision
  matrix-vector product, residual replacement and an outer flexible CG were all
  measured and none of them solves it. Such a solve now completes on the CPU with
  a printed reason naming what happened, counted by
  `LinearSolver::cpu_fallbacks`, instead of failing the run. On the user's 1.5 mm
  `mma` configuration only iterations 2 and 3 fall back; the run completes.
- **Regression pinned both ways.** `a_skeletal_design_solves_on_the_gpu_...`
  (slender bar at 0.10, asserts zero fallbacks) and
  `a_design_at_the_density_floor_...` (asserts a converged answer either way);
  both verified to fail on the previous shader and solver.
- Benchmark, same machine, against the same measurements without the limb:
  24 000 cells 2 % slower on the same 750 iterations, 271 350 cells 14 % slower
  on 10 % more, 3 000 000 cells 1 % faster on 10 % fewer.
- README: the precision section gains the accumulator, the carry guard and what
  single precision cannot do; the reproducibility claim is corrected - two GPU
  runs agree on every reported number and can still write different STL bytes.

## 2026-07-29 - 0.12.0 review closeout

- The sibling doc comment on `Growth::fundamental_design_cells` still carried
  the unqualified `1 / sectors` claim its twin in `engine::mod` had already
  softened; aligned with a pointer to the fuller odd-axis explanation.

## 2026-07-28 - 0.12.0 - Growth symmetry: grow one sector, replicate the rest

From user feedback on 0.11.0: a four-fold problem (square table, four identical
corner supports, central load) grew four *different* legs, because the attractor
scatter is stochastic. The result was accepted, and an option for symmetric
output was asked for. Asymmetric growth stays the default.

- **`[growth.symmetry]`.** `kind = "mirror"` with one or two `planes` (named by
  their normals, two of them quartering the domain), or `kind = "rotational"`
  with `order` in 2..12 about `axis` (default z). Everything is measured about
  the domain bounding box centre. Growth runs entirely inside the fundamental
  domain - attractors scattered only there, colonization steps clamped at the
  boundary exactly as at a keepout, backbones routed only for the load and
  support regions whose *centre* is in the sector - and the pruned, thickened
  skeleton is then replicated by the symmetry transforms, with the union of
  every copy rasterized. `mass_fraction` still measures the whole replicated
  structure. Only legal with `engine = "growth"`, with its own error under simp.
- **Symmetry replicates geometry, not loads,** and the README says so. Verifying
  that a whole problem is symmetric is brittle, so growforge checks the cheap
  half instead: each load and support region's centre should land on a region of
  the same kind under every declared transform, and one that does not is named
  in a warning while the run carries on. The stress report runs on the whole
  replicated structure with the real loads either way.
- **Exact arithmetic.** Reflections are diagonal `+-1`; quarter turns take their
  sine and cosine from a table, because `cos(pi/2)` is `6.1e-17`; the
  fundamental copy is not transformed at all; the replication order is fixed.
  Determinism holds for every kind and order: the same configuration exports
  byte-identical STL.
- **What is exact, stated precisely.** The *skeleton* is exact under every
  symmetry. The *rasterized field* is exact only where the transform maps cell
  centres onto cell centres: every mirror, `order = 2`, and `order = 4` on axes
  of matching parity. Every other order lands inside a cell instead, so the
  field is resampled up to half a voxel diagonal away and cells in the surface
  band can differ by as much as a whole density - 6.7 % of them over 0.1 on the
  six-fold test fixture, every cell more than a voxel from a surface exact. A
  finer `voxel_size_mm` is the remedy, the README carries the table, and the run
  line says "(skeleton exact, rasterized surface approximate to within a voxel)"
  for the orders it applies to.
- **A straddling region is owned whole.** A load or support region whose centre
  is in the sector is grown for there and carries its **full declared
  magnitude**, with no `1 / sectors` geometric share for the part of it in
  another sector. Invisible with one region or with regions that all straddle or
  all do not; visible in a problem mixing the two, where the straddling region's
  struts come out thicker than their share of the load in every copy. Documented
  in the README and on `grow` and `LoadRegion::magnitude_n`; the stress report
  runs on the whole part with the real loads regardless.
- **Reported and editable.** `check` names the symmetry, its sector count and the
  domain centre; a run adds the sectors and how many design cells the grown
  sector held. The editor's growth section gains the table (kind dropdown, planes
  or order and axis), and auto-regrow picks it up.
- **`examples/growth_canopy_symmetric.toml`**: the canopy problem quartered. One
  backbone and 138 segments grown, 552 exported, four identical legs, one
  connected body, volume fraction 0.1200 against 0.12, safety **16.6** (the
  asymmetric original: 4 backbones, 401 segments, 14.5).
- 509 lib + 9 integration tests (298 + 9 with `--no-default-features`), five
  clippy feature sets clean, docs clean. `growth_canopy` is untouched: 401
  segments, safety 14.54, 22 668 triangles; the shipped examples still round trip
  byte for byte.

## 2026-07-28 - 0.11.0 - Editor: a session that survives its runs

Four items, all from live user testing of 0.10.0.

- **A run that fails no longer reads as a dead program.** Reported: a full run
  started from the editor failed its linear solve on a marginally connected
  structure and the failure looked like the process exiting. Traced: the failing
  solve was the *optimization* one (`solving load case "tip"`, at the
  optimization tolerance of 1e-8), which is outside the degradation contract in
  both the CLI and the editor by design - there is no design field to export.
  The stress pass degrades identically on both paths and always did. What was
  wrong was the *consequence*: the editor's worker printed the failure with the
  same `error: ` prefix `main` uses for a fatal error, and the panel went on
  claiming the run was in flight. Now the panel carries the solver's own message
  - which names the fix - plus that the session is unaffected, the console line
  is run-scoped, `run full` is offered again at once, and nothing is written. An
  editor session that ends normally exits zero however its runs went.
- **A near-singular load path is found before it is solved for.** A load whose
  region reaches no support through material drives a system a factor of 1e9
  worse conditioned than the tolerance assumes; it used to be discovered by
  spending the whole 50 000 iteration stress budget. The analysis already flood
  fills the field for the cavity and solid-body reports, so the same fill now
  answers it first and the stress report degrades immediately with what to do
  about it. The threshold is 0.05, far below any iso level: the question is
  whether there is any stiffness on the path, not whether the path is part of
  the printed part.
- **Stop takes effect inside a solve.** Reported: "the generation is not
  stopping when the stop button is pressed." Two causes. First, `ViewReporter` -
  the wrapper the engine is actually handed for a full run - never forwarded
  `Reporter::cancelled`, so a full run's SIMP loop never saw a stop at all and it
  reached the run only at the stage boundary after the whole optimization.
  Second, even with that fixed, cancellation was checked only *between*
  iterations, and one cold solve is thousands of conjugate gradient iterations.
  The probe now reaches the solver: the CPU loop asks every 32 iterations
  (`CG_CANCEL_CHECK_INTERVAL`, a relaxed atomic load against an iteration
  costing at least 6e5 flops) and the GPU driver once per refinement pass and
  once per device readback batch. A cancelled solve is a distinct outcome -
  `Solve::Cancelled`, neither an error nor a non-convergence - which unwinds the
  objective evaluation, the SIMP iteration, the stress pass and the run in turn,
  exporting nothing and leaving the design of the last completed iteration. On a
  27 648 element problem the stop went from 35.6 s to 0.06 s. Command line paths
  pass no probe and run exactly the arithmetic they did; detaching a
  `run --view` window is still not a stop.
- **Hover highlight.** The object under the pointer gets a thin cyan outline,
  because overlapping elements were hard to tell apart. It is picked with
  exactly the ray and the rank rule a click uses, so it previews the click
  rather than approximating it, and it is drawn by the selection-shell machinery
  at a thinner margin on a layer of its own. Suppressed during any drag or
  orbit, over the side panel, off the window, over the object already selected,
  and over a gizmo handle - where the handle brightens instead. The domain is
  excluded, for the reason it is not clickable. Re-picked only when the ray
  under the pointer has moved.
- **Floor grid at the snap increment.** Minor lines at the panel's `snap mm`
  (1 mm default), a major line every tenth, on the plane `z = the bottom of the
  domain` over its footprint plus a tenth of it as margin, re-ruled when the
  increment changes. Major lines are counted from the world origin, so they sit
  on round coordinates. Capped at 400 lines: past that the spacing is multiplied
  by the smallest whole number that fits - whole, so every line is still on a
  multiple of the increment - and the panel says the spacing it ended up with.
  A layer switch like any other, on by default in edit mode only.
- 488 lib + 7 integration tests (278 + 7 with `--no-default-features`), all four
  feature sets clippy clean, docs clean. `growth_canopy` still exports 401
  segments, safety 14.54, 22668 triangles; `run --view` still detaches and
  finishes headless; the shipped examples still round trip byte for byte.

## 2026-07-28 - 0.10.0 - Editor: precision interaction

Dragging becomes *placing*: a drag lands on a round number, the number is shown
where the drag is happening and can be typed instead, boxes can be turned, and
objects stay inside the domain.

- **Snapping.** Translations, resizes and radii land on a millimetre increment
  (default 1 mm, panel dropdown plus free entry); rotations land on 22.5
  degrees. Snapping is absolute - the *value* lands on a multiple, not the
  movement - and applies only to what the handle is changing, so an x drag never
  moves y. Holding **Alt** frees the drag.
- **Dimension callouts.** A drag raises a floating number box beside the
  geometry it measures: the signed distance moved along the drag axis, the
  dimension being resized, the radius, the cylinder length, the angle turned.
  It lingers 6 s after the release and can be **clicked and typed over** with an
  exact value, applied to the shape the drag started on as one undo step;
  Escape or a click elsewhere cancels. Anchored to the projected 3D point and
  kept inside the 3D view, so it never lands under the side panel.
- **Rotation, in the schema and in the gizmo.** Box shapes take an optional
  `rotation_deg = [rx, ry, rz]`, applied about the box **centre** in extrinsic
  XYZ order, legal in every list a box is legal in. Absent is the axis aligned
  box it always was, bit for bit: the unrotated SDF path is the very arithmetic
  it was, and the key is only written once a box has been turned. The geometry
  is exact throughout - the SDF inverse-rotates the sample point, the bounding
  box is the extremes of the eight rotated corners, and picking inverse-rotates
  the *ray* rather than marching. Box objects get one curved arrow per axis;
  cylinders get the same arcs, which turn their caps about the segment centre;
  spheres get none.
- **Directional indicators.** A dimension line with an arrow head at each end is
  drawn across whatever the callout is measuring, on a new `Measure` overlay
  layer, plus a floor-distance line to the bottom of the domain while an object
  is moved vertically. Both are visible only while there is a number on screen.
- **Domain containment (on by default).** Every commit of a non-domain object -
  gizmo drag, numeric field, typed callout - is clamped so its bounding box
  stays inside the domain's; a rotated box is clamped by its rotated bounds, an
  object too large for an axis is centred on it, and a clamped commit leaves a
  transient note in the panel. The panel toggle switches it off for the
  documented case of a keepin that deliberately sticks out.
- **Load and support regions land flush on faces.** These two are placed against
  a surface rather than at a coordinate - the top of a load pad, the wall of the
  design space - so while one is dragged, the faces of the keepins and the
  outside of the domain are candidates: a face of the region within 2.5 mm of
  one lands exactly on it, and the callout says which. The surface wins over the
  millimetre grid where both apply, Alt switches off both, and the other object
  kinds are placed by their own numbers and get the grid alone. A rotated box
  lands by the bounds it really occupies against those axis aligned planes;
  matching a turned face against a turned face is a different feature.
- **Curved overlays are smooth.** The overlay shapes are shaded by a 40 degree
  crease rule: normals meeting at a point are averaged only across facets within
  that angle, so a cylinder's barrel and a sphere read as round, the rim where a
  barrel meets its cap stays an edge, and a box is exactly as flat as it was.
  The tessellation went to 48 segments round a cylinder and 48 x 24 over a
  sphere. The voxelized domain deliberately stays flat - its facets are the
  model the solver has - and no exported mesh is touched, the STL being
  extracted rather than tessellated.
- **The default solver backend is now the GPU, softly.** `[solver] backend`
  absent means the compute device; a build without the `gpu` feature resolves it
  to the CPU while the configuration is read, and a machine with no adapter
  resolves it when the device is asked for, each saying so in one line. A
  backend *named* in the file is unchanged and still an instruction: `"gpu"`
  without the feature or without an adapter is an error. What the default costs
  is cross-machine reproducibility - results stay deterministic per machine and
  driver - so every test in this crate that pins a recorded trajectory, a
  gradient check or a hash now names `backend = "cpu"` in its own fixture. The
  shipped examples are untouched and simply take the new default: on this
  machine `cantilever` optimizes in ~10 s where the CPU takes ~170, at the same
  compliance of 7.850071e0. The README timing table was re-measured.
- **A new file scaffolds onto `simp`.** The starter configuration switches from
  the growth heuristic to the topology optimizer, which is what the new default
  backend exists to accelerate; auto-regrow stays off for `simp`, so a new file
  opens on its setup and the first run is one the user asks for.
- **`RunGuard::drop` comment corrected.** It claimed two `Relaxed` stores
  enforced a cross-thread order they do not; it now says what actually makes the
  pair safe - nothing reads them across a race, because everything that reads
  them does so after a thread join.
- 460 lib + 7 integration tests (266 + 7 with `--no-default-features`), all four
  feature sets clippy clean, docs clean. `growth_canopy` still exports 401
  segments, safety 14.54, 22668 triangles - on the new default backend; the
  shipped examples still round trip byte for byte, without a `rotation_deg` key.

## 2026-07-28 - 0.9.0 - Editor: review round

Four review findings on the hands-on round, the first of them the Windows
AppHang contract reintroduced for edit mode.

- **An editor window watches its own runs again.** Its `ViewerApp` was built
  with no `watching` callback, so the loop exited on the first tick after
  teardown and the join for a worker mid-stage - a SIMP iteration is about a
  second, an analysis fifteen - happened outside it, with the message queue
  unserviced: the hang class Phase 2 fixed. The probe behind the callback is a
  count of live run threads kept by the threads themselves, so it covers a
  preview, a full run and a run stopped and winding down to its next checkpoint
  alike (the deleted `full_run_flag` covered only the last), and a panicking run
  still counts itself out. Semantics are unchanged: the close still stops
  everything and a stopped run still writes nothing.
- **The overwrite warning no longer accuses the session of its own file.**
  `wrote_output` was declared and never assigned, so every full run after the
  first warned that the file it was about to overwrite had been written
  elsewhere. The write is recorded where it is known to have happened - by the
  run thread, from the mtime the file turns out to have - and compared with the
  tolerance a coarse filesystem timestamp needs.
- **`Worker::detach` removed.** Dead since the close began stopping runs, and its
  semantics - leave a full run going headless - were the opposite of the
  editor's contract.
- **The stale-pointer regression test now discriminates.** It moved the cursor
  between the drag and the press it was pinning, which would have healed the bug
  it names; it now presses with no move at all after the drag, and fails on the
  old code.
- **A run thread that panics no longer wedges the panel.** Both end-of-run
  stores sat at the foot of the thread's closure, which an unwind skips: the run
  stayed "in flight" for the rest of the session, showing as running and making
  `run full` a silently disabled no-op. One guard now ends the run - the live
  count, and the run's own done flag with it - so an unwinding thread ends as
  completely as a returning one.
- **A stopped full run hands the worker back at once.** `is_running_full` was a
  latched flag only the thread could clear, so it stayed true through the whole
  winding-down window and disagreed with `is_running`; it is now derived from
  the current run, which a stop releases immediately. A stopped run cannot reach
  its export, so it owns neither the output file nor the button - the thread it
  leaves behind is the window's business, not the panel's.
- 401 lib + 7 integration tests (257 + 7 with `--no-default-features`), all four
  feature sets clippy clean, docs clean.

## 2026-07-28 - 0.9.0 - Editor: hands-on round

Six items from the first hands-on pass, the first of them a defect the headless
tests could not have caught because nothing synthesized a pointer.

- **Gizmo arrows could not be dragged, and clicking one selected the domain.**
  Two root causes, both found by reading the input path (neither was the display
  scale: every value in it - the cursor, the surface, the panel's share - is
  already in physical pixels, and the ray is built against the very viewport the
  frame was drawn with). First, the ray for a *press* was cast from a pointer
  position that only the camera path maintained, and the camera never sees the
  moves a gizmo drag takes - so after any drag the next press was cast from
  wherever that drag began: it missed the handle, fell through to picking, and
  selected the domain shell that encloses everything; when it did hit, the drag's
  reference was captured at the wrong ray and the first real move jumped by the
  distance between them ("blows them up"). The pointer position is now tracked
  once, on every event, before anything decides not to look at it. Second, a
  handle's grab volume was a small sphere at the arrow's *tip*, while the arrow
  the user aims at is the whole shaft: handles now carry a grab *shape* - a
  capped cylinder along the arrow, a sphere for the cube handles - tested with
  the same analytic intersections the picking uses. Arrows are also shorter
  relative to what they move (0.75 of the bounding radius, so the tip sits just
  outside the face it points through rather than out past the edge of the view).
- **A deterministic input-simulation harness.** Synthesized winit pointer events
  are driven through the real window event handler - no window, no GPU - against
  a known camera and surface, at scale factors 1.0, 1.25 and 1.5. It asserts end
  to end that a click on an arrow's shaft grabs that handle and does not change
  the selection, that dragging it moves the object by the distance the pointer
  covered on that axis and no other, that a click selects, that empty space
  deselects, that a press that travelled orbits instead, and that a press after
  a drag casts from where the pointer is now. This class of bug no longer
  depends on human hands.
- **The selection shell and the handles track the object.** They are rebuilt
  whenever they no longer describe the selected shape - compared against the
  shape itself rather than against a flag someone has to remember to set - so a
  resize that ends smaller than it started, a number typed into a field, an undo
  and an add all leave overlays of the right size on the next frame.
- **The domain is selected from the tree, never by a click.** Everything else
  lives inside it, so a click that would only hit the domain now counts as empty
  space. Selected from the tree it drags and resizes like any other object: the
  selecting is restricted, the editing is not.
- **A stop button.** It ends the preview an edit started or the full pipeline,
  through the same cooperative seam - `Reporter::cancelled` between iterations,
  and again at each stage boundary of the pipeline - so a stopped run reaches no
  export and writes **no file at all**. Idempotent, honest status line, and the
  editor is immediately usable again.
- **Closing an editor window ends its session.** Whatever was running is stopped
  and nothing is written; the unsaved-changes modal says so when a run is in
  flight. `run --view` still detaches and finishes headless - that run was asked
  for on the command line and has a file to write - but a run asked for inside a
  window has nowhere to report once the window is gone, and one left growing
  invisibly is what made two concurrently open editors appear to interfere.
  Instance audit: two growforge processes share no lock, socket, pipe, temp file
  or environment channel, and nothing raises or focuses a window after startup;
  the only thing two sessions can share is a file they are both configured to
  write, and a full run now says so before it overwrites one that something else
  wrote.
- **`growforge edit` on a path that does not exist scaffolds a starter
  configuration** and opens it: a block on the floor with a load on a pad at the
  top, growth engine, PLA, every number from `constants`. It validates, builds
  without warnings and grows as written. An existing file is never overwritten
  by that path.
- 395 lib + 7 integration tests (257 + 7 with `--no-default-features`), all four
  feature sets clippy clean, docs clean.

## 2026-07-28 - 0.9.0 - Visual editor: `growforge edit`

The problem definition itself becomes the document. `growforge edit
<config.toml>` opens the viewer's window in editor mode: pick objects in the
viewport or the tree, drag them, type exact numbers, watch the setup
re-voxelize, regrow on the spot, save the file back.

- **Sidebar object editor.** Every object of the configuration in a tree -
  domain entries with their `op`, keepout, keepin, supports with their fixed
  axes, load cases and their force, torque and gravity loads - each with add and
  delete buttons and a properties panel of exact numeric fields. Every scalar
  section is editable too, with an optional key's own default shown next to the
  checkbox that adds it. Live validation runs the real `Problem::build` on every
  committed edit and shows what it says, with the live problem summary (grid,
  cells, nodes, per-region node counts, memory) beside it.
- **Viewport interaction.** Analytic ray picking against the same signed
  distance primitives the configuration is written in, ranked so that what sits
  inside something else is picked before what contains it. The selection gets a
  translucent shell and a gizmo: three axis arrows and a centre handle to
  translate, eight corners and six faces for a box, both caps and a radius for a
  cylinder, a radius for a sphere. Left click selects and left drag still
  orbits - a click is a press that went nowhere - and a whole drag is one undo
  step, applied to the shape it started on rather than accumulated.
- **Undo/redo** over the configuration, the document and the selection
  together, bounded at 100 steps, with `Ctrl+Z` / `Ctrl+Y` / `Delete` /
  `Ctrl+S`.
- **Instant refresh and auto-regrow.** A committed edit re-voxelizes and rebuilds
  the setup overlays through `scene::build`, the same path `view` uses, after a
  150 ms debounce. Auto-regrow (on for `growth`, off for `simp`) then re-runs the
  engine in the background: growth at its real resolution, SIMP as a fast preview
  capped at 20 iterations on a grid coarsened to about 40 000 cells and labelled
  `preview` in the panel. A newer edit cancels the run in flight through a new
  cooperative `Reporter::cancelled` hook the SIMP loop checks between
  iterations - the default is false, so every existing run is untouched. `run
  full` is the real pipeline on `viewer::run_worker`, stress report and STL
  included; previews write nothing at all.
- **Format preserving save.** New `toml_edit` dependency, gated behind the
  `viewer` feature. Comments, key order, blank lines, alignment, number spellings
  and line endings of everything untouched come back byte for byte on all four
  shipped examples; only changed values are rewritten, in place, keeping their
  trailing comments. Added objects append cleanly and a deleted object takes its
  own comments with it, because the document is edited structurally alongside the
  configuration rather than reconstructed at save time. A modified marker in the
  title bar tracks the dirty state and closing with unsaved changes asks (save /
  discard / cancel) before the teardown path starts.
- `toml_edit` holds every integer in an `i64` and refuses to parse a document
  with a larger literal, while `examples/growth_canopy.toml` pins a `u64` seed
  above `i64::MAX`. Such a literal is now carried as text through a sentinel that
  no other value in the document uses, so the seed the editor shows, runs and
  writes is the real one and the file round trips untouched.
- **Pre-existing defect, found while testing the editor and fixed here:** a
  resolution that derives a runaway grid took the process down on the
  allocation. `target_cells` is a `usize`, so `12345678901234567890` is a legal
  request, and `voxel_size_mm = 0.002` over a 120 mm part is another route to
  the same place; both ended in `capacity overflow` with no message and exit
  code 101, on `check`, `view` and `run` alike - and, once the editor existed,
  on a number typed into a field. The derived cell count is now checked against
  `constants::MAX_GRID_CELLS` (64 million, about 16 GiB by the memory estimate's
  own accounting, four to five orders above every shipped example) *before* the
  grid is laid out, and rejected with a line naming the key, the grid it
  derives, the count and the budget. The count is worked out in floating point,
  because the number being reported is precisely one that does not fit in a
  `usize`. In the editor it lands in the validation panel like any other
  rejection, with the last model that built still on screen.
- **Every number in a configuration must be a finite number.** TOML has `nan`,
  `inf` and `-inf` literals and serde takes all three into an `f64`, so a load
  vector's component, a radius, a tolerance or a voxel size could arrive as
  something no arithmetic downstream can do anything with - and a `nan` force
  reached the finite element assembly unchecked. One pass over the whole tree
  (every table, list, shape, region and load) now runs in `Config::parse` and in
  `validate_static`, naming the key it rejects. It also removes the last way to
  poison the editor's modified marker, which compares a configuration with
  itself and could never match one holding a `nan`; the editor's numeric fields
  refuse to accept one being typed in either.
- A second round of review fixes on the round trip: the stand-ins that carry
  integers past the signed range are **pruned to the ones the document actually
  holds** on every projection, so a session of edits to successive huge values
  gives each stand-in straight back instead of exhausting the pool - and if the
  pool ever did run out, the save now **fails loudly** with the key named
  instead of leaving the old number in the file and reporting success.
- Review fixes, all three found before release: **every** unsigned key now goes
  through that same sentinel path, not just the seed - `target_cells`,
  `max_iterations`, `max_steps`, `smoothing_iterations` and `supersample` are
  `usize`, a save projects all of them unconditionally, and a plain `as i64`
  wrote a legal 64 bit count back as its wrapped negative, producing a file
  growforge itself would refuse to read; the conversion is checked now, so no
  configuration integer can wrap. The modified marker is **computed** from the
  configuration against the one the file holds instead of latched, so an edit
  undone back to the saved state stops claiming unsaved changes and stops
  raising the modal, while undoing *past* a save correctly still does.
  `[output] stress_json` is shown next to `stl_path`, read-only like it.
- New `Scene` layer roles: setup layers are what an edit rebuilds, and the
  editor's own selection and gizmo overlays are listed by its panel alone, so
  `view` and `run --view` are unchanged down to the checkbox list. `run --view`
  on `growth_canopy.toml` still reports 401 segments, a 14.54 safety factor and
  22 668 triangles.
- 383 lib + 7 integration tests (256 + 7 with `--no-default-features`), all four
  feature sets clippy clean, docs clean. The whole panel is drawn headlessly in a
  test - every selection, every section, the modal - because a window smoke test
  only exercises what nobody clicked on. Interactive polish (gizmo feel, modal
  flow) is covered by unit tests of its maths and by hand.

## 2026-07-28 - 0.8.1 doc fixes from the full-system verification sweep

- README's Surface quality table still quoted growth_canopy's pre-0.8.1 mesh
  (28 100 / 109 176 triangles); refreshed to the current 22 668 / 89 520.
- README's "single error line" claim softened: TOML parse failures print the
  toml crate's multi-line diagnostic by design; semantic rejections stay one
  line.

## 2026-07-28 - 0.8.1 - Growth: feet plant in the middle of their supports

Reported by a user from a top-down view of the running preview: "the posts of
the table looking object are not sitting directly on the supports but hanging
off slightly."

- Measured on `examples/growth_canopy.toml`: every one of the four legs planted
  on the extreme corner cell of its 7 x 7 support patch, **16.97 mm** - over four
  voxels - from the patch centre, with a foot radius of up to 19.8 mm hanging off
  two sides of it. The cause was not the shortcut pass and not the load side: the
  tabletop spans the whole plate, so all 49 cells of a corner patch are reachable
  at *exactly* equal cost, the search stops at the first target it settles, and
  among equal-cost candidates the deterministic tie break is the lowest cell
  index - a corner.
- A backbone now aims at the **centre** of the support region rather than at the
  region as a whole: one support region, one foot, planted in its middle. The
  two ends of a path are deliberately different. The path still leaves the load
  region from whichever cell is nearest, because a distributed load enters
  everywhere and the shortest way out spends the least material; it arrives at
  the support's centre, because that is the one place this leg is grounded. A
  region whose centre is walled off falls back to the old behaviour, so nothing
  that used to find a path stops finding one.
- All four offsets are now **0.00 mm** and each trunk rises vertically out of its
  patch. It was a structural defect and not only a visual one: a foot on the rim
  transfers its load through the constrained nodes it half covers, and centring
  it lifts the safety factor from **10.23 to 14.54** at the same 0.1199 volume
  fraction (peak von Mises 4.59 to 3.23 MPa), with 401 segments instead of 332.
  The exported STL changes accordingly; same-seed reproducibility is unchanged
  and still tested byte for byte.
- The fusion tolerance is untouched: how close two solids must come to merge and
  where a path should terminate are different questions, and a foot planted in a
  region's middle is inside the region proper by construction.

## 2026-07-28 - 0.8.0 - Growth: branches that end on nothing are a defect

Reported by a user running `examples/growth_canopy.toml`: the canopy grew
antler-like stubs that reached inward from the legs and stopped in mid air -
"doesn't look quite right, I think it stopped before it was done". The run had
finished as designed; the design was wrong. A free branch tip carries no load,
which makes it dead mass in a tool whose whole purpose is weight, it is an
unsupported overhang no printer can lay down, and it reads as an unfinished
model. Structurally purposeless geometry is now treated as a defect rather than
as decoration.

- **Growth aims at the structure.** Attraction points now come in two kinds. The
  interior ones are as before and keep the routing organic; new *surface* points
  are seeded one per patch of keepin, support region and load region cells, and
  are consumed only once a branch has actually fused to the surface they sit on,
  so a branch aimed at one keeps growing until it arrives. A branch that has
  arrived stops growing - letting it continue only sent it crawling along the
  surface it had just reached.
- **Pruning.** New `[growth] prune`, default true: every branch that still ends
  on nothing is removed together with the dead-end chain behind it, back to the
  last junction that leads somewhere. One reverse pass over the skeleton, which
  is exactly the union of the paths from the roots to the fused tips. `false`
  keeps the free tips for anyone who wants the decorative growth, at the price of
  their share of `mass_fraction`.
- **Fused tips carry load.** A load region's magnitude is now split over every
  place a branch fuses to it - the backbone tips *and* every canopy tip that grew
  into it - rather than over the backbones alone. A region's footprint extends
  over the whole connected keepin body its cells sit on, so a branch reaching the
  underside of a tabletop is carrying the tabletop.
- **An attraction point nobody is approaching is given up**, after the time a
  branch would need to cross the attraction radius. Without it, a point behind a
  wall dragged a branch after it for the entire step budget and the growth spent
  itself on a member that was then pruned.
- Backbones are resampled at the growth step length after the shortcut pass. The
  shortcut left a trunk of one or two enormous segments, which gave the canopy
  nowhere to sprout from and Murray's law nothing to taper along; the resampled
  nodes sit on the same straight lines, so the trunk stays as direct as the
  shortcut made it.
- **Fusion is overlap, not tangency.** A tip is fused once it is within *0.8* of
  the smallest strut radius of an anchor, never a whole radius. At a whole radius
  the anchor sits exactly on the capsule's surface, where the signed distance is
  zero and the density is exactly the iso level; marching cubes can extract that
  as two separate watertight shells, so a "fused" tip could ship as a floating
  chunk. The shipped example only escaped it by a parameter accident - a step
  length half again the radius, which overshot the boundary.
  `constants::GROWTH_FUSION_RADII` carries the worst-case penetration arithmetic,
  and a test stands on the exact old boundary and asserts it is refused.
- **The analysis pass counts the connected bodies of material**, the same flood
  fill the cavity pass runs over the complementary cells. More than one body is a
  floating island: joined to nothing, carrying no load, printed as a separate
  loose object - and invisible to every other check, since an island is perfectly
  watertight, manifold and encloses no cavity. Reported next to the cavity report
  as a `solid bodies` line and a warning, never a failure, mirroring
  `voids = "warn"`. `RunOutcome` gains `solids`; `analyse` and `analyse_with`
  return it.
- On the shipped example, at the same 0.1200 volume fraction: 332 segments
  instead of 224, **no free tips at all and one connected body** (279 branch
  nodes pruned, 198 canopy tips fused into the tabletop), and a safety factor of
  **10.23 against the previous 8.89** - the peak von Mises falls from 5.29 to
  4.59 MPa. The stress solve also got faster (about 8 s against 19 s) because the
  field it is given is cleaner. The exported STL is a different file from
  0.7.0's, which is expected: the algorithm changed. Same-seed reproducibility is
  unchanged and still tested byte for byte.
- New `engine::growth::anchor` module (anchor set, fusion test, solid body flood
  fill, the pruning pass) and a public `engine::growth::grow`, which returns the
  skeleton so that "no branch ends on nothing" is a property anything outside the
  crate can check rather than a promise nobody has to keep. New
  `GrowthPhase::Pruning` progress line, four new `GrowthSummary` counters, and
  `connections` and `solid bodies` lines in the run summary. 17 new tests.

## 2026-07-28 - 0.7.0 - MMA: a second update scheme for runs that will not settle

- New `[optimization] update = "oc" | "mma"` (default `"oc"`). An unknown value
  is rejected and names the two. `oc` stays the default and the reference, so
  every shipped example, recorded trajectory and reproducibility promise is
  untouched; the recorded pre-phase-3 compliance trajectory still reproduces.
- `mma` is Svanberg's method of moving asymptotes (1987), specialized to this
  problem shape: one volume constraint, box `[0, 1]`, move limit. Separable
  convex approximations with a lower and an upper asymptote per design variable,
  a closed form primal `x(lambda) = (L sqrt(p) + U sqrt(q)) / (sqrt(p) +
  sqrt(q))`, and the single-constraint dual solved by the same bisection `oc`
  uses - on the same chain-aware volume, measured on the printed densities.
  Asymptotes start symmetric and then shrink by 0.7 on an oscillation, widen by
  1.2 on monotone progress, clamped to 0.01 .. 10 of the box.
- It is for overhang runs the optimality criteria step cannot settle. On the
  shipped `shelf_bracket.toml`: `oc` ends at the 120 iteration cap with a design
  variable change of 0.0254 against the 0.01 asked for, `mma` at 0.0167 and a
  2.5 % lower compliance, and converges at iteration 132. On a 120 mm deep
  variant of the same bracket `oc` sits on its 0.2 move limit for 400 iterations
  with the compliance still inside a 1.6 % band at the end of them (4.220e1);
  `mma` converges at iteration 214 on 2.689e1, 36 % lower. Expect to raise
  `max_iterations`: MMA is more conservative per step and uses the extra
  iterations to actually converge.
- `examples/shelf_bracket.toml`, the overhang showcase, now ships with
  `update = "mma"` and `max_iterations = 150`: it is the example the README's own
  guidance is about, so it practices it, and it converges at iteration 132 on a
  compliance of 1.879277e1 instead of ending on the cap. Under `oc` over the same
  150 iterations it never converges (0.12882 at the cap, 1.926071e1). Every other
  shipped example is unchanged and stays on `oc`.
- Review fix: the README's update-scheme comparison quoted the OC overhang
  residual from the older 120-iteration control run (0.013); refreshed to the
  150-iteration control's 0.012 so both sides of the comparison come from the
  same budget.
- MMA needs no self-weight sensitivity shift: a positive `dC/dx` lands in the
  upper asymptote term and pushes that variable down, so the shift is never
  applied and never announced. Tested with gravity on and off.
- Adjoint fix in the self-supporting filter, which MMA exposed: the smooth
  maximum's derivative was formed as `smax / sum_i v_i^P`, and a supporting
  region of densities around 1e-8 drives that sum into the subnormals while the
  derivative is still of order one. The ratio overflowed to `inf` and met the
  `0` of `v_i^(P-1)` further down the chain, so the whole sensitivity field
  became `NaN` within four iterations. It is now factored through the region's
  largest member, where every term is bounded. `oc` could reach the same state;
  it simply never did on the shipped examples. The fix moves an overhang run's
  arithmetic in its last digits: the bracket's `oc` baseline goes from
  1.927318e1 to 1.927065e1 over 120 iterations.
- The seam: `engine::update` holds the `Updater` enum, the shared `Step`,
  `Buffers` and `Constraint`, and the volume measurement both schemes share;
  `engine::mma` holds the new scheme. `engine::oc` keeps its own step untouched.
- `growforge check` and the run banner gain an `update` line next to the filter
  and overhang lines. The per-iteration line and the stats panel are unchanged.
- Housekeeping: `SUPERSAMPLE_NODE_BUDGET_WARN` said "about 640 MiB" for what is
  32e6 x 20 B = 610.35 MiB. The runtime message always computed it correctly.
- 18 new lib tests: the closed form against a hand computed case and against the
  stationarity condition it solves, the dual bisection on three volume targets,
  the asymptote rules and both clamps, the subproblem bounds, mixed sign
  sensitivities without a shift, design-cells-only, both schemes meeting the same
  target, config parsing and rejection, the scheme reaching the loop, MMA within
  5 % of `oc` on the plain cantilever, a miniature shelf bracket where MMA leaves
  a smaller final change than `oc` at equal iterations, the memory estimate
  accounting for MMA's four extra cell arrays, the self-supporting filter's
  adjoint staying finite on a supporting region that underflows, and a recorded
  compliance trajectory through that adjoint (the overhang counterpart of the
  plain one, so a silent regression in the transpose fails a test rather than a
  paragraph).

## 2026-07-28 - 0.6.0 - Surface quality: supersampled meshing and smooth shading

- New `[output] supersample = N` (integer, default 1, capped at 4): marching
  cubes runs on a lattice N times finer in every axis, resampled from the same
  node density field by trilinear interpolation. Triangles and file size grow
  with about N^2, the lattice with N^3. Measured: `cantilever.toml` 15 352 ->
  60 744 triangles (0.73 -> 2.90 MiB, export 0.01 -> 0.02 s),
  `growth_canopy.toml` 28 100 -> 109 176 (1.34 -> 5.21 MiB, 0.01 -> 0.04 s).
- `supersample = 1` writes the byte-identical STL of the pipeline before this
  change: the refinement returns the field untouched at factor 1, so the default
  export literally takes the old path, and a test compares the bytes.
- The refined lattice copies every source sample exactly (index arithmetic, not
  world positions), keeps the zero padding zero so the surface still closes, and
  welds vertices on its own global edge ids, so watertightness comes from the
  same construction. Taubin smoothing, validation and the void pass are
  unchanged and never see the difference.
- `growforge check` warns when the projected export lattice passes
  `SUPERSAMPLE_NODE_BUDGET_WARN` (32 million samples, about 640 MiB at 20 bytes
  per sample). It warns rather than refuses. The mesh stats line names the
  factor only when it is above 1.
- The viewer draws the density surface smooth: area weighted per-vertex normals,
  computed on the mesher and export threads and never on the render thread, for
  previews and for the final mesh, plain and stress coloured alike. A new
  `flat shading` panel switch restores the per-triangle look by deriving the
  face normals from the layer's own positions at upload time, so nothing extra
  is kept alive for a switch that is usually never flipped. Overlays stay flat.
- The shader was already normal-agnostic and is unchanged: shading mode is a
  property of the vertex data, which keeps flat mode pixel-exact instead of
  approximating it from screen space derivatives.
- Stress colouring is unaffected by either: it samples the node stress field
  trilinearly at each vertex, so a supersampled surface reads the same field at
  more places. The STL is unaffected too - facet normals stay per-triangle, as
  the format requires.
- Previews are never supersampled; the window switches to the real refined
  surface when the run finishes.
- 14 new lib tests and 1 new integration test, including the byte-identical
  gate, watertightness at factor 3 on an asymmetric field, and a growth run
  exporting the same bytes twice at factor 2.

## 2026-07-28 - 0.5.0 - Phase 5, GPU compute and a stress report that can fail

- New `[solver] backend = "cpu" | "gpu"` behind a new `gpu` cargo feature (on by
  default, independent of `viewer`, sharing its wgpu dependency).
  `--no-default-features` is still a pure CPU build. CPU stays the default and
  the reference: it is what the determinism promises and the recorded
  trajectories are written against.
- GPU conjugate gradient in WGSL: the whole Krylov recurrence lives on the
  device (five dispatches per iteration, batched into one command buffer, only
  the residual read back), wrapped in double precision iterative refinement on
  the host. WGSL has no f64, so the device solves for a *correction* while the
  residual and the convergence test stay f64 on the same operator the CPU
  backend uses - which is what lets a single precision inner solve deliver the
  same 1e-8 the CPU path promises. Node-centric gather (no atomics, fixed
  order), Neumaier compensated element accumulation and reductions, and each
  refinement pass asks the device only for the accuracy the outer loop still
  needs.
- The gather is centred on the node's own displacement, which is exact (the
  rigid body translations are in the element matrix's null space) and removes
  the near-total cancellation of the raw 24-term row dot. It is what decides
  whether the backend works on a fine mesh at all: without it a 3 million
  element problem stalls at a relative residual of 5e-1 instead of reaching
  1e-8, and it doubled the speedup on every size.
- Measured on an RTX 3080: 12.4x on the 80 703 DOF cantilever, 9.7x at 856 980
  DOF, 7.3x at 9 211 503 DOF (3 million elements), and
  `examples/cantilever.toml` at its full 150 iterations in 11.1 s against about
  170 s, landing on the same final compliance of `7.850071e0`, the same volume
  fraction and the same stress table. Parity against the CPU reference is 1e-9
  or better on displacements, compliance and peak von Mises, five decades inside
  the gates.
- New `growforge bench <config.toml>`: times three cold solves of the real
  assembled problem on every available backend to one tolerance and prints DOF,
  per-solve milliseconds and speedup. No synthetic matrices.
- A stress solve that will not converge is now a warning, not a failed run. The
  STL is still written, the cavity report still printed, the exit code is still
  zero, and the summary says `stress report unavailable: <reason>`. No
  `stress_json` is written and the viewer's stress switch stays disabled.
  `RunOutcome::stress` became a `StressOutcome`.
- The line between the two is drawn at *solving*, and drawn in one place:
  `stress::analyse_with` returns `Err` for a setup failure (a backend that
  cannot be opened, a design that cannot be bound) and `Ok(Unavailable)` only for
  a solve failure of an already bound solver. Opening a backend used to be
  inside the degraded region, so `engine = "growth"` with `backend = "gpu"` on an
  adapterless machine would have exited zero having never opened one - growth
  performs no solves, so the stress pass is the first place a backend is asked
  for. New `stress::analyse_with_solver` for a caller that owns the solver, which
  is also how the bind failure is forced in a test without a hook.
- A conjugate gradient breakdown is now tested for after convergence rather than
  before. Device iterations run in batches between readbacks, so a system small
  enough to converge inside one batch kept iterating on rounding noise until
  `p^T K p` collapsed and reported a breakdown for a solve that had already
  finished.
- The compute solver builds its wgpu instance from the environment, so
  `WGPU_BACKEND` really pins the graphics API it runs on - which the
  reproducibility notes had claimed before it was true.
- The post-run stress solve got its own budget, separate from the optimization
  path's and unchanged there: `STRESS_CG_MAX_ITERATIONS` (50 000) and
  `STRESS_CG_TOLERANCE` (1e-6, justified by the recovery's first-order error
  against the discretization error it already carries).
- New `LinearSolver` / `BoundSolver` seam in `fea`, split so the design is
  uploaded once per optimization iteration and the right hand side once per load
  case; `Objective::evaluate` takes the solver, `stress::analyse` builds its own.
  The CPU path through the seam is bit-identical to the bare solver, and a test
  asserts it.
- 16 new lib tests, including the three GPU parity gates. Every test that needs a
  compute adapter skips with a printed note, so an adapterless machine still
  passes.

## 2026-07-28 - 0.4.0 - Phase 4, the growth engine

- Second engine behind the existing `Engine` trait: `engine = "growth"` grows a
  structure instead of optimizing one. A* load paths (26 neighbours, Euclidean
  costs) from every load region to every support region it can reach, shortcut
  into polylines; space colonization for the organic canopy; Murray's law
  `r_parent^n = sum(r_child^n)` over the accumulated load flow for the radii,
  with one global scale bisected against `mass_fraction`; capsules unioned with
  a smooth minimum and sampled into the density field. No finite element solve
  anywhere in it: 0.03 s where SIMP takes minutes.
- Grow then verify: the field goes into the unchanged cavity, stress, meshing
  and STL pipeline, and the von Mises table is what says whether the heuristic
  did well. The README says so in as many words.
- Deterministic by construction: an in-crate PCG32 (no new dependency, no clock
  or entropy seeding) pinned to the published `pcg32-demo` reference vector, and
  a rasterizer parallelised over slabs so the non-associative smooth minimum
  still folds the struts in one order. Same config, byte identical STL.
- New `[growth]` table, legal only with `engine = "growth"` and rejected
  alongside `[optimization.overhang]` with a message saying there is no growth
  equivalent. Every key optional; the length defaults are derived from
  `min_feature_mm` and `mass_fraction` rather than being absolute millimetres,
  so they mean the same thing on a part ten times larger. `mass_fraction` and
  `min_feature_mm` are reused as the volume target and the smallest strut
  diameter.
- A load region with no path to any support fails the run with the case and load
  named; a problem whose only loads are gravity is refused with a pointer to
  `simp`. Radius clamps that cannot reach `mass_fraction` warn with the fraction
  that is achievable and carry on.
- `IterationStats` gained an optional `growth` block (phase, segments,
  attractors left) that the console, the run summary and the viewer panel switch
  on; the SIMP output format is untouched. Density snapshots go out through the
  existing observer hook every `GROWTH_REPORT_INTERVAL_STEPS` steps, so
  `run --view` animates the growth through the existing viewer plumbing.
- `LoadSummary` and `SupportSummary` keep the nodes their region selected, which
  is what the router needs and what the assembled force vector and constraint
  mask can no longer be taken apart into.
- New `examples/growth_canopy.toml`: a 200 mm tabletop on four feet with a
  service column through the middle, 4 backbones and 224 segments in 0.03 s,
  watertight, safety factor 8.9. New README section, 48 new unit tests and one
  new integration test.
- Doc only: `MIN_NET_GRAVITY_FRACTION` records that its comparison inherits the
  naive norm's overflow at about 1e155 mm/s^2, and the README's self-weight
  section mentions that cancelling gravity loads are rejected at build time.

## 2026-07-28 - 0.3.0 - Phase 3, printability, self-weight and stress

- Overhang constraint: optional `[optimization.overhang] build_direction`
  chains Langelaar's additive manufacturing filter after the density filter, so
  the analysis, the volume constraint and the STL all see a printable design.
  The 45 degree self-supporting angle is fixed by the supporting stencil, not by
  a knob. Sensitivities travel back through the reverse layer sweep; the run
  reports the max and mean `|printed - designed|` residual.
- Self-weight: `type = "gravity"` inside a load case, with optional `direction`
  and `g_mm_s2`. The load is design dependent, so it is reassembled per
  iteration and the compliance sensitivity gains its `2 u^T df/dx` term. Density
  is converted from g/cm^3 to tonne/mm^3 so the force comes out in newtons.
- Optimality criteria step shifts every sensitivity by a multiple of the volume
  gradient when a design dependent load makes one positive; the shift only
  renames the Lagrange multiplier, so the optimum is untouched. It engages only
  when needed, leaving the plain path byte for byte as it was.
- The run summary gains a `self weight` line, in newtons and in grams, for every
  load case that carries gravity, next to the existing mass estimate. A run with
  no gravity load prints nothing extra.
- A load case whose gravity loads cancel each other out is rejected while the
  problem is built. Each load is separately required to be non-zero when it is
  parsed, but two opposing ones used to sum to a zero acceleration that the case
  still claimed as a self weight, which the summary then divided by. New
  `constants::MIN_NET_GRAVITY_FRACTION`.
- Enclosed cavities: `[output] voids = "warn" | "fill"`, six-connected flood
  fill from the grid boundary, run before marching cubes so the report always
  describes the exported file. A cavity overlapping a keepout is never filled.
- Stress report: one extra solve per load case on the final field, von Mises at
  every element centroid, max / p99 / top decile / safety factor per load case
  in the run summary and optionally as JSON via `[output] stress_json`.
- Viewer: the finished mesh can be coloured by von Mises stress from a side
  panel toggle, normalized against the material's yield strength; plain shading
  stays the default. A gravity load draws no arrow.
- OC trial fix: a design variable with a zero volume sensitivity (which only the
  self-supporting filter can produce) used to be pushed to its upper move limit
  every iteration, accumulating blueprint material that ambushed the design when
  support later grew under it. Such a variable now decays.
- New modules: `engine::am_filter`, `engine::chain`, `engine::objective`,
  `fea::gravity`, `stress`, `voids`, `json`. `RunOutcome` carries the cavity and
  stress reports; `ScalarField` gained trilinear sampling.
- New `examples/shelf_bracket.toml` exercising all of it; README sections for
  each feature with their units and caveats. 47 new unit tests, including a
  finite difference validation of the whole sensitivity chain for the plain,
  overhang and self-weight configurations, and two new integration tests.

## 2026-07-27 - Phase 2 fix: detached runs were killed as hung applications

- Closing the viewer during a long run could get the whole process terminated
  by Windows part way through, losing the STL (silent death, no panic, exit
  -1). The window was dropped as the event loop was already terminating, so
  winit's `DestroyWindow` was never serviced: the window survived, unpumped,
  for the rest of the run, and Windows eventually declared the process hung
  (`AppHangB1`). Teardown now happens inside the loop, and the loop keeps
  servicing its message queue until the run finishes.
- Every path out of the viewer now goes through one teardown and one loop exit,
  so a failed frame or a failed GPU init cannot abandon the loop either.
- New `constants::VIEW_DETACHED_POLL_INTERVAL_S`.

## 2026-07-27 - Phase 2 review fixes

- A panicking optimization worker now marks the run failed in the viewer's
  stats block instead of leaving the window on stale "optimizing" numbers; the
  panic still propagates and fails the process.
- Fitting the view with `F` before the first frame no longer suppresses the
  opening auto-fit, and uses the panel-confined viewport rather than the whole
  surface.

## 2026-07-27 - 0.2.0 - Phase 2, native viewer

- `growforge view <config>` opens a 3D window on the problem setup, and
  `growforge run <config> --view` shows the density isosurface evolving over a
  run and then the exported mesh. Closing the window detaches the viewer; the
  run finishes headless and still writes its STL.
- Viewer stack behind the default-on `viewer` cargo feature: wgpu 29, winit
  0.30, egui 0.35 (+ egui-wgpu, egui-winit), bytemuck, pollster.
  `--no-default-features` builds the solver alone.
- Observer seam: `Reporter` gained a default no-op `densities` hook, which the
  SIMP loop calls once per iteration with the physical densities.
- New `viewer` module: orbit camera and matrix maths, analytic tessellation of
  boxes/cylinders/spheres plus procedural force arrows and torque arcs,
  overlay scene assembly, a latest-only one-slot snapshot channel, the wgpu
  renderer and the egui side panel.
- `lib.rs` split into `optimize` and `export` behind the unchanged
  `optimize_and_export`; `RunOutcome` now carries the exported mesh and
  `load_config_and_problem` returns the configuration alongside the problem.
- `GROWFORGE_VIEW_AUTOCLOSE_S` closes the window N seconds after the first
  frame, for smoke tests and CI.
- All viewer colours, sizes, camera parameters and throttles in
  `src/constants.rs`; 38 new unit tests, 36 of them behind the feature.

## 2026-07-27 - Phase 1 review fixes

- Mesh pipeline rejects meshes that would overflow a 32 bit vertex or triangle
  index instead of truncating the cast; `marching_cubes::extract` now returns a
  `Result`. Limits live in `constants::MAX_MESH_VERTICES` and
  `MAX_STL_TRIANGLES`.
- README documents the measured wall times of both examples.

## 2026-07-27 - 0.1.0 - Phase 1

- Initial growforge crate: `check` and `run` CLI over a TOML problem definition.
- SDF geometry (box, capped cylinder, sphere) with ordered CSG, voxel grid with
  keepout/keepin/domain classification.
- Hexahedral linear elasticity FEA: 2x2x2 Gauss element stiffness, matrix-free
  Jacobi PCG with Dirichlet projection and an eight-colour parallel element
  loop, force and torque load assembly, multiple weighted load cases.
- SIMP engine behind an `Engine` trait: cone density filter with transpose
  sensitivities, optimality criteria update with bisected volume constraint,
  per-iteration reporting and a `--quiet` flag.
- Mesh export: marching cubes with edge-keyed welding, Taubin smoothing,
  watertight/manifold/winding validation with volume and mass statistics, binary
  STL writer and reader.
- All tunables in `src/constants.rs`; examples `cantilever.toml` and
  `mbb_bridge.toml`; 81 unit and 3 integration tests.
