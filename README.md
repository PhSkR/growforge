# growforge

growforge grows strong, weight-optimized 3D structures. It reads a TOML problem
definition, voxelizes the design domain, runs one of three engines over it, and
exports a watertight binary STL ready for slicing.

| engine     | what it does                                                    | cost           |
| ---------- | --------------------------------------------------------------- | -------------- |
| `simp`     | topology optimization driven by a real finite element solve      | minutes        |
| `growth`   | a deterministic growth heuristic: route, branch, thicken         | a fraction of a second |
| `solid`    | no optimization at all: fills the domain and exports what was drawn | instant     |

Scope so far: all three engines, static linear elasticity, compliance minimization
under a volume constraint, a 45 degree self-supporting (overhang) constraint, an
optional guide wireframe that seeds a SIMP run with a load path joining every
region it has to reach and then lets go of it,
self-weight loads, enclosed cavity detection, a post-run von Mises stress report
that runs on whichever engine's result, marching cubes export with optional
lattice supersampling for a finer surface and floating fragments culled from the
surface that ships, an optional GPU compute backend for
the finite element solve, a native 3D viewer, smooth shaded, for checking a
setup and watching a run, and a visual editor that edits the problem definition
itself - drag the geometry, change any value numerically, regrow on the spot,
save the file back.

## Units

Millimetres, newtons, megapascals (MPa = N/mm^2), newton-millimetres for torque,
g/cm^3 for mass density and mm/s^2 for gravitational acceleration. There is no
unit key anywhere in the config; every number is in these units. The one
conversion growforge does internally is the mass density: `1 g/cm^3 = 1e-9
tonne/mm^3`, which is what makes `rho * V * g` come out in newtons (see
`constants::TONNE_PER_MM3_PER_G_PER_CM3` for the dimensional analysis).

## Build and run

```sh
cargo build --release

cargo run --release -- check examples/cantilever.toml
cargo run --release -- view  examples/cantilever.toml
cargo run --release -- edit  examples/cantilever.toml
cargo run --release -- bench examples/cantilever.toml
cargo run --release -- run   examples/cantilever.toml
cargo run --release -- run   examples/cantilever.toml --view
cargo run --release -- run   examples/mbb_bridge.toml --quiet
cargo run --release -- run   examples/shelf_bracket.toml
cargo run --release -- run   examples/growth_canopy.toml
cargo run --release -- run   examples/growth_canopy_symmetric.toml
```

Always use `--release`. The finite element solve is the whole workload and a
debug build runs it one to two orders of magnitude slower.

Measured wall times for the shipped examples on the machine this was developed
on (a multi-core x86-64 desktop with an RTX 3080), with `--release` and the
**default solver backend**, which is the compute device. Every shipped SIMP
example sets `max_iterations = 150` itself, so these times are budgets rather
than the default one (which is 1000 - see [When a run
stops](#when-a-run-stops)); only `shelf_bracket.toml` reaches its convergence
tolerance, at iteration 132 of its 150:

| example               | engine   | grid                | iterations | optimize | analyse | export  |
| --------------------- | -------- | ------------------- | ---------- | -------- | ------- | ------- |
| `cantilever.toml`     | `simp`   | 60 x 20 x 20 cells  | 150        | ~10 s    | ~0.4 s  | ~0.01 s |
| `mbb_bridge.toml`     | `simp`   | 96 x 16 x 16 cells  | 150        | ~24 s    | ~0.5 s  | ~0.01 s |
| `shelf_bracket.toml`  | `simp`   | 30 x 9 x 47 cells   | 132 of 150 | ~14 s    | ~0.4 s  | ~0.01 s |
| `growth_canopy.toml`  | `growth` | 50 x 50 x 30 cells  | 24 steps   | ~0.04 s  | ~0.9 s  | ~0.01 s |
| `growth_canopy_symmetric.toml` | `growth` | 50 x 50 x 30 cells | 23 steps | ~0.03 s | ~1 s | ~0.01 s |

On the CPU backend the same runs take about ~170 s, ~500 s, ~45 s and ~8 s: the
optimize column is the conjugate gradient and the analyse column is the one
extra solve per load case the stress report needs, and both of them are what the
device accelerates. The designs are the same either way - `cantilever.toml`
converges to a compliance of 7.850071e0 on both - see
[Solver backend](#solver-backend) for what "the same" is and is not promised to
mean.

For the `simp` runs almost all of that is the conjugate gradient solve, which
runs a few hundred iterations per optimization step at the default 1e-8 relative
residual. Set `max_iterations` low while you iterate on a design - a hundred is
plenty to see where the material is going - and leave it out of the final run,
where the default budget of 1000 lets the design converge and the stall
criterion stops it if it never will. The "analyse" column is the cavity pass plus the one extra finite
element solve per load case the stress report needs; it does not scale with
`max_iterations`. The "export" column is the default `supersample = 1`; see
[Surface quality](#surface-quality) for what refining the lattice costs.
`check` is instant.

The `growth` row is the whole point of that engine: it grows a 75 000 cell
design in thirty milliseconds, and then spends twenty seconds *checking* it with
the same finite element solve. On a growth run the analysis is the run.

The self-supporting constraint costs roughly two to three times the wall time
per iteration, because the volume constraint's bisection has to push every trial
design through the whole filter chain, not just the density filter.

`check` validates the configuration and prints the problem summary (grid size,
cell and node counts, how many nodes each support and load region selected,
whether the overhang constraint is on or, for a growth run, the resolved growth
controls, the solver backend, the cavity policy, estimated memory) without
optimizing. `view` prints the same summary and opens a 3D window on the
setup. `edit` opens the same window on the same setup and lets you change it -
see [Editor](#editor). `run` does the optimization, prints the cavity and stress
reports and writes the STL, with `--view` adding a live window over it.
`--quiet` suppresses the per-iteration progress lines. `bench` times the linear
solve of a configuration on every available backend and writes nothing; see
[Solver backend](#solver-backend).

Every command exits non-zero with a clear `error: ...` message when the problem
is rejected. Semantic rejections (a bad mass fraction, cancelling gravity loads)
are a single line; TOML parse failures (a typo'd key, an unknown enum value)
carry the toml crate's multi-line diagnostic with the offending line and a
caret, which is worth knowing if a script parses stderr.

## Growth engine

```toml
engine = "growth"
```

An alternative to SIMP that grows a structure instead of optimizing one. It is
fast (a fraction of a second where SIMP takes minutes), deterministic, and it
comes out looking organic rather than like a stress field, which is often what a
visible part wants. In five stages:

1. **Backbone.** Every load region is routed to every support region it can
   reach: a shortest path over the voxel grid (A*, 26 neighbours, Euclidean edge
   costs) through cells that are not keepouts and not outside the domain, then
   shortcut into a polyline so it is not a voxel staircase. This is the
   *guarantee* in the engine: a load with no path to any support fails the run
   with the load case and the load named, rather than growing something that is
   not connected to the ground.

   **One support region, one foot, planted in the middle of it.** The path
   leaves the load region from whichever of its cells is nearest - a distributed
   load enters the structure everywhere, and the shortest way out of it spends
   the least material - but it arrives at the support region's *centre*, not
   wherever the search first meets it. A support region every cell of which is
   equally reachable, such as a patch of floor under a tabletop, would otherwise
   be entered at whatever corner the search settled first, and the thickened leg
   would stand on the rim of its own footprint and hang off one side of it. A
   centred foot overflows its patch on every side when the leg is thicker than
   the patch is wide, which is honest; hanging off one side is not. If you want a
   leg somewhere specific, declare a support region there.
2. **Branching.** Space colonization (Runions et al.) fills the free space:
   attraction points are scattered through the design cells, each pulls on the
   branch node **that can still grow** nearest to it, every pulled node steps
   `step_mm` towards the average of its pulls, and a point a branch reaches is
   consumed. A step that would leave the allowed cells is deflected onto a
   coordinate plane, and a branch that is blocked a few times in a row stops.

   The points come in two kinds, and the difference is what makes the growth
   *connect* rather than wander. **Interior points**, scattered through the
   design cells, are consumed from a whole `kill_radius_mm` away and are what
   makes the routing organic. **Surface points**, seeded one per patch of
   structural surface - keepin cells, support region cells, load region cells -
   are consumed only when a branch has actually fused to the surface they sit
   on, so a branch aimed at one keeps growing until it arrives. A branch that
   arrives stops: it has done its job, and letting it grow on would only send it
   crawling along the surface it just reached. A point nothing has come any
   nearer to for as long as it would take a branch to cross the attraction
   radius is given up as unreachable, so no branch spends the run chasing
   something behind a wall.
3. **Pruning.** Every branch that still ends on nothing is removed, together
   with the dead-end chain behind it, back to the last junction that leads
   somewhere. See below.
4. **Thickening.** Not the same pass as
   [`[output] reinforce`](#reinforcement-minimum-printable-thickness), which
   holds a *finished field* to a printable floor: this one sizes the struts of
   the skeleton from the flow through them, while the design is still being
   built. Each load region pushes its magnitude into the skeleton,
   split over **every** place a branch fuses to it - the backbone tips *and*
   every canopy tip that grew into it - every remaining branch tip gets a share
   of the canopy load, and the flow accumulates towards the roots. Radii follow
   Murray's law `r_parent^n = sum(r_child^n)`, and one global scale is bisected
   until the rasterized field hits `mass_fraction`.
5. **Rasterize.** The struts are capsules, unioned with a smooth minimum so
   junctions come out filleted, sampled at cell centres and mapped through a
   smoothstep one voxel wide into densities. Keepouts stay empty and keepin
   regions stay solid whatever grew nearby.

From there it is the same pipeline as SIMP: the cavity policy, the stress
report, marching cubes, smoothing, validation and the STL.

**Read the stress report before printing.** This is a heuristic and it says so:
nothing in it solves an equilibrium. The load path exists by construction and
the thicknesses follow a branching law that is a good prior for a structure in
tension and compression, but no part of the growth knows what the stresses
actually are. The von Mises table that the run prints afterwards is computed by
the same finite element solve SIMP optimizes against, on the exact field that
was exported, and it is the honest answer. Grow, then verify. A design whose
safety factor comes out too low wants a larger `mass_fraction`, a larger
`min_feature_mm`, or the `simp` engine.

Two things follow from being a heuristic and are worth knowing:

* **The backbones are the verified paths.** A backbone is a shortest path from a
  load to a support and exists by construction. A fused canopy tip is a real
  connection - the tip is joined to material the finite element model applies
  the force to, and it takes a share of the load in the thickening - but the
  path it takes to ground was grown, not solved for.
* **A problem with only gravity loads is refused.** Growth needs somewhere to
  grow towards, and a self-weight load has no region. Add a force load, or use
  `simp`.

### Pruning: no branch may end on nothing

```toml
[growth]
prune = true    # optional, default true
```

A space colonization branch stops where the attraction points around it run out,
which leaves a tip hanging in mid air. Growforge treats that as a **defect, not
a style**, and removes it. A free tip:

* carries no load, which makes it dead mass in a tool whose entire purpose is
  weight;
* is an unsupported overhang, which no printer can lay down;
* and reads as a model that stopped half way, which is how it was first
  reported.

The pass keeps a node when it is fused to a structural surface itself, or when
any node below it is; what survives is exactly the union of the paths from the
roots to the fused tips. Dead-end chains vanish back to the last junction that
still leads somewhere, however long they are. The backbones are never at risk:
their roots sit on support cells and their tips on load region cells.

A tip counts as **fused** when it comes within **0.8** of the smallest strut
radius (`min_feature_mm / 2`) of a keepin cell, a support region cell or a load
region cell. Every strut is at least that thick, so such a tip rasterizes *into*
that cell and the two come out of the isosurface as one solid.

The 0.8 is not a fudge factor. A tip accepted at a whole radius would put the
anchor exactly on its capsule's surface, where the signed distance is zero and
the density is exactly the iso level: tangent, not overlapping, and marching
cubes can turn that into two separate watertight shells. Staying strictly inside
the zero crossing buys a penetration depth of `0.2 * min_feature_mm / 2` and
keeps every cell between the tip and the anchor strictly above the iso level;
`constants::GROWTH_FUSION_RADII` carries the arithmetic.

And because the failure it guards against is invisible to everything else in the
pipeline - a floating chunk is perfectly watertight, manifold and encloses no
cavity - the analysis pass also **counts the connected bodies of material** in
the exported field and reports them next to the cavity report:

```text
field bodies   1 connected body in the density field
mesh bodies    1 in the exported surface
```

Anything past the first body is material joined to none of the rest, which a
slicer lays down as a separate loose object. For a growth run that is always a
defect, so the run says so plainly - and, like the cavity report, warns rather
than failing. Whether such a body is debris or something the configuration asked
for is decided on the mesh rather than here, by the rule in
[Islands](#islands).

The two lines describe **two different objects**, and reading one as the other is
what let a defect through: see [Islands](#islands) below.

The reclaimed mass does not vanish: the volume target is met by the thickening
that follows, so it goes into the branches that do work. On the shipped example
that is worth a **64 % better safety factor at the same volume fraction** (14.5
against 8.9), and the run says what it did:

```text
connections    29 surface targets, 21 unreached, 212 fused branch tips carrying load
pruned         250 branch nodes that ended on nothing
```

`prune = false` keeps the free tips, for anyone who wants the decorative growth
for its own sake. Be aware of the trade: those branches are made of the same
material as the working ones and are paid for out of the same `mass_fraction`,
so on a design where they are a large share of the skeleton the volume target
becomes unreachable at the minimum strut radius and the run reports the clamp.

`[growth] prune` and `[output] trim` are **different passes and not
alternatives**: this one removes branches of a *skeleton*, while the growth
engine is still building one, by whether they reached anything; [Trimming
unloaded material](#trimming-unloaded-material) removes cells of a *finished
density field*, after any engine, by the stress in them.

### Determinism

The same configuration produces a byte identical STL. The only randomness is a
PCG32 generator written out in `src/engine/growth/prng.rs` and seeded from
`[growth] seed` alone: nothing reads the clock or the operating system's
entropy, and the one parallel loop in the rasterizer is partitioned so that
every cell sees the struts in the same order whatever the thread count. The
generator is pinned by a test against the published `pcg32-demo` reference
vector, so the stream cannot drift under a refactor. Changing the seed changes
the canopy and nothing else: the volume target, the load paths and the clamps
are all met either way. The seed only moves the parts that survive pruning, so a
design whose canopy is entirely determined by where the structural surfaces are
can come out the same under two seeds; that is the growth doing its job, not the
seed being ignored.

### Symmetry: grow one sector, replicate the rest

```toml
[growth.symmetry]
kind = "mirror"               # "mirror" | "rotational"
planes = ["x", "y"]           # mirror: one or two plane normals -> 2 or 4 sectors
# kind = "rotational"
# order = 4                   # rotational: 2 .. 12 sectors of the full turn
# axis = "z"                  # rotational: default "z"
```

Space colonization is stochastic, so a four-fold problem - a square table on
four identical corner feet, loaded in the middle - grows **four different
legs**. All of them are sound and none of them is the same, which is organic and
is exactly what one user asked for an alternative to. This is that alternative:
growforge grows **one fundamental domain** and replicates exact copies of it, so
a symmetric problem can produce a symmetric part. Leaving the table out is the
default and changes nothing.

**The fundamental domain** is measured about the **centre of the domain's
bounding box** (which is the centre of the grid block, the bounding box padded
symmetrically out to whole voxels):

* `kind = "mirror"` with `planes = ["x"]` keeps the half `x <= centre.x`. A
  second plane keeps the quarter that also satisfies `y <= centre.y`. Planes are
  named by their **normals**, so `"x"` is the plane `x = centre.x`, and two of
  them give four sectors.
* `kind = "rotational"` with `order = N` keeps the sector spanning the first
  `1 / N` of a turn about the `axis` through the centre, measured from the axis
  after it in cyclic order (`x` for a rotation about `z`).

Inside that sector, everything runs exactly as it always did: attraction points
are scattered only there, a colonization step that would cross the boundary is
refused exactly as a step into a keepout is, and the backbones are routed only
for the load and support regions the sector owns. **A region belongs to the
sector its centre is in**, so a support patch or a load pad that straddles a
plane - a load applied to the whole tabletop, a support in the middle - belongs
to the sector its middle is in and is grown for once, rather than half-grown
twice. The regions that are left out are covered by the copies. After pruning
and thickening, the skeleton is replicated by the symmetry transforms and the
**union of every copy** is rasterized, so a strut that reaches a plane meets its
own reflection there and the smooth minimum fillets the joint like any other.

**A straddling region is owned whole, and carries its whole load there.** The
sector that owns it pushes the region's **full declared magnitude** into the
skeleton it grows - there is no `1 / sectors` geometric share for the part of the
region that lies in another sector. Where that is invisible: a problem with one
load region, or one whose regions all straddle, or one whose regions all do not,
because the flow only decides the *relative* thickness of the struts and the
volume target sets the absolute scale. Where it shows: a problem mixing the two,
where the straddling region is weighted as if all of it were in the sector, so
its struts come out thicker than their share of the load - and, since the copies
are copies, in every sector. The stress report is computed on the whole
replicated part with the real loads regardless, so it still tells you what the
structure actually does.

`mass_fraction` still means what it always did: the mean density over **every**
design cell of the problem. The arithmetic works out because the sector is about
`1 / N` of the design volume, so filling it to a fraction `f` of its own volume
fills the whole to `f` as well; growforge measures the whole replicated
structure rather than the sector, so the two agree even where the boundary cuts
through cells. (The sector is *exactly* `1 / N` of the design cells only when no
cell centre lands on the boundary. An axis with an odd number of cells puts a
whole layer of centres on a mirror plane, and that layer belongs to the
fundamental domain - a layer belonging to neither half would be a one-voxel gap
through the part - so five cells across a plane are owned three to two. The run
prints the count it actually grew in.) One radius scale is bisected for all the
copies, which is what keeps them identical.

**What is exact, and what is exact to within a voxel.** The *skeleton* is exact
under every symmetry here: the copies are images of one node set under matrices
that are themselves exact, so a node's image is a node to the last bit. The
*rasterized field* is that skeleton sampled at cell centres, and it is exact only
when the transform maps cell centres onto cell centres:

| symmetry                                       | skeleton | rasterized field |
| ---------------------------------------------- | -------- | ---------------- |
| any `mirror`                                    | exact    | exact            |
| `rotational` `order = 2`                        | exact    | exact            |
| `rotational` `order = 4`                        | exact    | exact when the two axes it turns hold cell counts of the same parity - a square footprint always does |
| every other `order` (3, 5, 6, 7, 8, 9, 10, 11, 12) | exact | **approximate, bounded by the voxel size** |

A sixth of a turn takes a cell centre to a point *inside* another cell rather
than to its centre, up to half a voxel diagonal from where that cell is sampled.
The density is a smoothstep one voxel wide, so two samples that far apart agree
exactly wherever the field is flat - every cell more than a voxel from a surface
matches its image exactly - and can differ by as much as a whole density in the
surface band. On the six-fold fixture in the test suite, 6.7 % of cells differ by
more than 0.1, all of them within a voxel of a surface. **Choose a finer
`voxel_size_mm` if seam-level symmetry matters**; the exported surface is smooth
through the band either way, and the run says which case it is in:

```text
symmetry       6-fold rotation about z, 6 sectors; grown in 2160 of the 12888
               design cells and copied (skeleton exact, rasterized surface
               approximate to within a voxel)
```

**Symmetry replicates geometry, not loads.** Growforge does *not* try to verify
that the problem is symmetric - checking a whole problem is neither cheap nor
robust - so a lopsided load on a symmetric structure will happily produce a
symmetric part that is wrong for it. What it does do is check the cheap half:
every load and support region's centre should land on another region of the same
kind under each declared transform, and a region that does not is **named in a
warning** while the run carries on:

```text
warning: [growth.symmetry] mirror across x and y takes support 1 to
[72.0, 72.0, 2.0] mm, where this problem has no support: the nearest one is
28.3 mm away, against a tolerance of 8.0 mm. ...
```

The stress report at the end runs on the whole replicated structure with the
real loads, so an asymmetric load on a symmetric part still gets an honest
verdict - read it, as always, before printing.

Two smaller consequences worth knowing. A run whose fundamental domain contains
no load region, or no support region, **fails** with that as the reason: the
sector would carry nothing, or have nowhere to plant a foot. And a strut lying
*in* a mirror plane is unioned with its own image, which the smooth minimum
fillets very slightly thicker than a single strut; the volume target absorbs it.

Determinism is unchanged for every kind and order: the same configuration
produces a byte identical STL. The reflection matrices are exact, quarter turns
take their sine and cosine from a table rather than from `cos` (which gives
`6.1e-17` for a right angle), the fundamental copy is not transformed at all,
and the copies are unioned in a fixed order.

`examples/growth_canopy_symmetric.toml` is the shipped example: the canopy
problem with `planes = ["x", "y"]`, which grows one quarter (1 backbone instead
of 4, 138 segments replicated to 552) and comes out as **four identical legs**,
one connected body, at a volume fraction of 0.1200 against the 0.12 asked for,
with a safety factor of **16.6**.

### Configuration

```toml
[growth]                      # only legal with engine = "growth"
seed = 11396317718348371989   # optional; any u64
attractor_per_cm3 = 5.0       # optional; attraction points per cm3 of design domain
attraction_radius_mm = 130.0  # optional; how far a branch tip sees an attractor
kill_radius_mm = 43.3         # optional; when a branch has reached one
step_mm = 9.0                 # optional; one space colonization step
murray_exponent = 2.6         # optional; n in r_parent^n = sum(r_child^n)
max_radius_mm = 27.0          # optional; cap on a single strut radius
max_steps = 400               # optional; cap on colonization iterations
prune = true                  # optional; remove branches that end on nothing
```

Two `[optimization]` keys are **reused rather than duplicated**, and this is the
part to internalise:

* **`mass_fraction`** is the volume target the strut radii are normalized
  against, measured exactly as it is for SIMP: the mean density over the design
  cells.
* **`min_feature_mm`** is the smallest strut **diameter**, so the smallest strut
  *radius* is half of it. It is also the scale every length default is derived
  from.

The defaults are scale aware rather than absolute millimetres, because a fixed
default would be meaningless on a part ten times larger:

| key                    | default                                                      |
| ---------------------- | ------------------------------------------------------------ |
| `step_mm`              | `0.75 * min_feature_mm`                                       |
| `kill_radius_mm`       | `min_feature_mm / 2 * 2.5 / sqrt(mass_fraction)`              |
| `attraction_radius_mm` | `3 * kill_radius_mm`                                          |
| `max_radius_mm`        | `3 * min_feature_mm`                                          |
| `murray_exponent`      | `2.6`                                                         |
| `attractor_per_cm3`    | `5.0`                                                         |
| `max_steps`            | `400`                                                         |

The kill radius default is the one with real content in it. Branches grow until
every attraction point has been consumed, which is when the tube of the kill
radius around them has swept the design domain, so a domain of volume `V` ends
up with about `L = V / (pi k^2)` of skeleton, weighing `(r_min / k)^2` of the
domain at the minimum radius. Reading that backwards,
`k = r_min / sqrt(mass_fraction)` is the kill radius whose bare skeleton weighs
exactly what was asked for; the factor of 2.5 on top leaves the thickening room
to taper instead of starting already clamped. **A smaller `kill_radius_mm` gives
a finer, denser canopy of thinner branches; a larger one gives fewer, fatter
ones.** That is the knob to turn for looks.

Rejected: a `[growth]` table without `engine = "growth"`, `engine = "growth"`
together with `[optimization.overhang]` (the self-supporting filter is a stage
of the SIMP density chain and there is no growth equivalent - orienting a grown
structure by build direction is not available), a `kill_radius_mm` at or above
`attraction_radius_mm`, a `max_radius_mm` below `min_feature_mm / 2`, a
`murray_exponent` outside `[1, 8]`, a non-positive length, a zero `max_steps`,
and a `step_mm` shorter than a quarter of a voxel.

If the radius clamps cannot reach `mass_fraction` - the skeleton is already too
heavy at the minimum radius, or too light at the maximum - the run **warns with
the fraction that is achievable and carries on** rather than failing. A design
that is a little heavy is still a design.

### Sizing a growth problem

The engine grows a skeleton whose length is set by the kill radius, so the
number of branches you get at a given `mass_fraction` is roughly proportional to
the cell count. A visibly rich canopy therefore wants a fine grid, and the cost
of a fine grid is **not** the growth (which stays in the tens of milliseconds)
but the stress solve afterwards: past a few hundred thousand elements the
conjugate gradient solve can exhaust its iteration cap on a mostly empty field
and fail the run. `examples/growth_canopy.toml` sits at 75 000 cells, which
grows in 0.05 s and verifies in about 8 s.

### Example

`examples/growth_canopy.toml` is a 200 x 200 mm tabletop carrying 600 N spread
over its surface plus its own weight, standing on four feet, with a 60 mm
service column through the middle that the canopy has to route around. It grows
4 backbones and 401 segments, plants each of its four feet dead centre on its
support patch, prunes 250 branch nodes that ended on nothing, fuses 212 branch
tips into the underside of the tabletop, comes out as **one connected body**,
lands on a volume fraction of 0.1199 against the 0.12 asked for, and reports a
safety factor of **14.5** against PETG's yield strength.

`examples/growth_canopy_symmetric.toml` is the same problem with
`[growth.symmetry] planes = ["x", "y"]`: one quarter grown (1 backbone, 138
segments) and mirrored into **four identical legs** (552 segments), still one
connected body, at a volume fraction of 0.1200 and a safety factor of **16.6**.
See [Symmetry](#symmetry-grow-one-sector-replicate-the-rest).

## Overhang constraint (printability)

```toml
[optimization.overhang]
build_direction = "z+"     # one of x+, x-, y+, y-, z+, z-
```

Adding the table switches on Langelaar's additive manufacturing filter, so the
design the analysis, the volume constraint and the exported STL all see is
already printable. Leaving it out is the default and changes nothing.

The filter sweeps the grid layer by layer along the build direction. The first
layer sits on the build plate and prints exactly as designed; every later layer's
printed density is `smin(designed, smax(printed densities of its supporting
region))`, where the supporting region is the element directly below plus its
four lateral face neighbours in that layer. **That stencil is what fixes the
self-supporting angle at 45 degrees, so there is deliberately no angle key**: a
different angle would need a different stencil, not a different number.

`build_direction` names the direction layers stack in, so `z+` means the build
plate is the `z` minimum face of the grid and overhangs open upwards.

What this does and does not promise:

* Forced solid (`[[keepin]]`) cells count as printable support at density one and
  void cells as zero, so a keepin pad anchors whatever grows on top of it even
  where the pad itself floats.
* The smoothing is approximate. The smooth minimum can leave a printed density
  about 0.5 % above one where the design is fully solid, and the P-norm smooth
  maximum is exact at density 0.5 and over-estimates elsewhere. Both are the
  published scheme; the exponents live in `constants.rs`.
* The run reports an **overhang residual**, the largest and mean
  `|printed - designed|` over the design cells. The mean is the number to watch:
  a small mean says the filter is no longer fighting the design and the printed
  result is what the optimizer actually asked for. The maximum is a single worst
  cell and is usually one variable the filter is still erasing.

## Guide wireframe

```toml
[optimization.wireframe]
radius_mm = 2.5            # optional, default 2.5; radius of the guide wire
hold_iterations = 40       # optional, default 40; iterations the floor is held
seed_density = 1.0         # optional, default 1.0; density it is seeded at
```

Adding the table starts a SIMP run from a **guide** instead of a uniform field. At
setup a shortest keepout-avoiding path is routed between every load region, every
support region and every `[[keepin]]` entry until all of them hang off one
network; the network is rasterized as a thin wire, seeded into the initial density
field, and applied as a **density floor** after each update step for the first
`hold_iterations` iterations. Then it is released, and the run says so.

**It is a guide and never a constraint.** After the release the optimizer may keep
the wire, thicken it, move it or dissolve it entirely; no cell is pinned and no
sensitivity is touched. What the guide buys is the first few iterations - a design
that already joins the regions the configuration declared, rather than a uniform
field the analysis has to find a load path in.

What holds that up is that **the run cannot decide it has finished while the floor
is on**. A design under the floor is not free - the change an iteration reports is
partly the floor pushing its cells back up - so neither the convergence test nor
the stall criterion is asked until the release, and the stall window only ever
holds iterations the design was free in. A run therefore cannot settle inside the
hold window; only the iteration cap and a cancellation can end one there, and both
say so: *"the run stopped after N iterations with the guide's floor still active
... the design it ended on carries the wire as forced material rather than as a
guide"*. So the only way to export a forced wire is to be told that is what you
are doing.

The routing is the growth engine's, used from the outside: the same A* over the
26 neighbours of a cell with Euclidean edge costs, the same shortcut pass, and the
same capsule rasterizer. Keepouts and the space outside the domain are impassable,
so the wire goes around them or not at all. Nothing of the growth *heuristic* is
involved: there is no canopy and no thickening.

* **Every `[[keepin]]` entry is a terminal of its own**, not the merged union the
  classifier pins solid, so two disjoint pads both get a wire. Support and load
  regions are terminals as declared; a gravity load selects no region and is
  skipped. The order is fixed - supports, keepins, loads, each in configuration
  order - so the guide is a function of the configuration alone and a run is as
  reproducible as it was.
* **A region the wire cannot reach is named and skipped**, and the rest of the
  network is built without it. So is a region that holds no material at all - one
  a keepout swallowed, for instance, since keepout wins. Fewer than two regions
  with material, or a problem whose regions already touch each other, means there
  is nothing to wire and the guide reports that it is off.
* **During the hold the volume fraction runs above `mass_fraction`.** The floor is
  applied after the update step has already bisected its multiplier onto the
  target, so the wire's material is on top of it. That is reported honestly rather
  than hidden - the iteration line shows the floored design - and it self-corrects
  in the first iteration after the release. A
  [local volume cap](#local-volume-constraint-bone-like-structures) is
  overshot the same way and for the same reason, and recovers the same way. A `hold_iterations` that is not below
  `max_iterations` never releases at all, and is warned about twice: once when the
  guide is built, where the misconfiguration is visible before any time is spent,
  and once at the end, about the design that actually came out.
* **There is no separate overlay in the viewer, and none is needed**: the seeded
  wire is part of the density field every iteration publishes, so the live surface
  shows it being used or dissolved as it happens.

## Local volume constraint (bone-like structures)

```toml
[optimization]
update = "mma"             # required: the cap is a second constraint

[optimization.local_volume]
max_fraction = 0.6         # optional, default 0.6; cap on any neighbourhood
radius_mm = 6.0            # optional, default 3 x the density filter radius
```

Adding the table caps how much material any **neighbourhood** of the design may
hold, on top of the global `mass_fraction` target. Without it, the cheapest way
to meet a global target is to concentrate: a few thick members, flush surfaces, a
solid interior. With it, that is not available, and what the optimizer builds
instead is a network of many thinner members with porous interiors - the
structure bone has, and the answer to *"is there a way to make the algorithm
prefer many structural supports rather than just a flush surface?"*.

The local share of a cell is a second cone filter - the same kernel the density
filter is, at `radius_mm` - applied to the printed densities. Forced solid
(`[[keepin]]`) neighbours count as one and void neighbours are excluded from the
average, so a cell packed against a keepin pad reads the pad as material and a
cell at the domain boundary is not diluted by the space outside it.

The default radius is **three density filter radii**, which is
`3 x min_feature_mm / 2`. It has to be larger than one: a neighbourhood the size
of the smallest feature would price a single member's own cross section rather
than how the members are spread, and only a grey design could satisfy it.
`max_fraction` has to be above `mass_fraction` for the same kind of reason - the
average of the local fractions *is* the global one, so a cap at or below the
global target cannot be met by any design, and the configuration is rejected
rather than run.

Measured on a 40 x 6 x 16 mm block supported at both bottom corners and loaded in
the middle of its top face, at `mass_fraction = 0.4`, both runs converging:

| | worst neighbourhood | aggregate | volume | compliance |
| --- | --- | --- | --- | --- |
| no cap | 0.913 | 0.760 | 0.400 | 1.187e1 |
| `max_fraction = 0.6` | 0.716 | 0.600 | 0.324 | 1.455e1 |

What this does and does not promise:

* **`update = "mma"` is required.** The cap is a second constraint, and the
  optimality criteria step prices exactly one - its multiplicative ratio is
  bisected on a single multiplier. A configuration that asks for the cap under
  `oc` is rejected with that remedy named. The growth engine rejects the table
  outright: it places struts rather than densities, and a grown canopy is
  already a network of thin members.
* **The cap is on an aggregate, and the worst neighbourhood runs above it.** The
  per-cell fractions are aggregated into one constraint by a p-mean, which
  approaches the worst cell from below; the run therefore reports the **true**
  worst neighbourhood every iteration next to the volume fraction, and it is that
  number, not the aggregate, that says what the design holds. On the block above
  the aggregate lands exactly on the 0.6 asked for and the worst neighbourhood at
  0.716. The exponent is in `constants.rs` with what raising it costs.
* **The volume target becomes an upper bound.** With the cap binding, the global
  constraint is the slack one: material the design is allowed is material it has
  nowhere to put, and the block above converges at 0.324 of a 0.4 target. The run
  reports what it really used. While a design is still moving, the volume can
  also sit a little under the target for a second reason - the two-constraint
  subproblem meets the volume through its own separable approximation rather than
  by bisecting on the measured value, and that error follows the step size to
  zero as the design settles (0.0007 on the block above).
* **It costs iterations and stiffness.** Spreading the same material over more
  members is less stiff than concentrating it - that is what is being bought -
  and the block above takes 43 iterations against 16. A capped run that stops on
  the [stall criterion](#when-a-run-stops) is usually not a failure: material
  being traded between neighbourhoods that are all equally full is the converged
  character of these designs, and the stall note says so.
* **The kernel is not free.** It is a second filter at three times the density
  filter's radius, so about 27 times its taps, applied twice per iteration. A
  much larger `radius_mm` than the default is warned about with the tap count it
  would cost.
* **A guide wireframe's hold may violate it, transiently.** The floor is applied
  after the update step, exactly as it is applied over the volume target (see
  [Guide wireframe](#guide-wireframe)), so while the wire is held a
  neighbourhood along it can sit above the cap. It self-corrects in the first
  iteration after the release, and neither verdict about a settling design is
  asked before then.

## Self-weight (gravity)

```toml
[[loadcases.loads]]
type = "gravity"
direction = [0.0, 0.0, -1.0]   # optional, default [0, 0, -1]; normalized on use
g_mm_s2 = 9810.0               # optional, default 9810 mm/s^2
```

A gravity load has no `region`: it acts on every element. Its force is
`physical_density * material_density * element_volume * g`, spread equally over
the element's eight nodes along `direction`, which comes out in newtons because
the mass density is converted from g/cm^3 to tonne/mm^3 first. Several gravity
loads in one case add up as acceleration vectors, and a case may freely mix
gravity with force and torque loads.

Self-weight is a **design dependent** load: `f` moves with the densities, so it
is reassembled every iteration and the compliance sensitivity gains a load term,

```text
dC/dx_e = 2 u^T (df/dx_e) - u^T (dK/dx_e) u
```

both halves chained back through the filter transposes. That first term can make
a sensitivity positive, which the multiplicative optimality criteria update
cannot represent. When it does, growforge shifts every sensitivity by a multiple
of the volume gradient until they are all negative again and prints a one-off
note saying so. Because the shift is a multiple of the *volume* gradient it only
renames the Lagrange multiplier and leaves the stationarity conditions untouched;
it damps the step rather than moving the optimum.

The run summary reports the self weight it carried, in newtons and in grams, for
every load case that has one:

```text
estimated mass 104.51 g at 1.27 g/cm3
self weight    1.0645 N (108.51 g) in load case "shelf-load"
```

The two masses are independent readings of the same part, one from the enclosed
volume of the exported surface and one from the voxel density field the solver
actually loaded, so they agree to within the resolution of the isosurface rather
than exactly.

A gravity load with a zero direction or a non-positive `g_mm_s2` is rejected;
flip `direction` to pull the other way. So is a load case whose gravity loads
cancel out to a (near) zero net acceleration: each load is separately legal, but
the case would then claim a self weight it does not carry, and everything
downstream divides by that magnitude. That one is caught while the problem is
built rather than while the config is parsed, because only the sum can see it.

## Update scheme

```toml
[optimization]
update = "oc"      # "oc" (default) | "mma"
```

Which scheme moves the design variables under the volume constraint. Both meet
that constraint to the same tolerance, measured the same way (the mean *printed*
density at the far end of the density chain), and both respect the same box
`[0, 1]`, the same move limit and the same `[[keepin]]` / `[[keepout]]`
classification. Only design cells ever move.

**`oc`, optimality criteria (the default).** The classical multiplicative
update `x <- x (-dC/dx / (lambda dV/dx))^0.5` clipped to a move limit, with
`lambda` bisected until the volume lands on `mass_fraction`. Cheap, robust and
the reference: **it is what the shipped examples and the recorded regression
trajectories were written against, so leaving this alone is what keeps a rerun
of an old configuration reproducible.** Switching schemes changes every number a
run prints.

**`mma`, method of moving asymptotes.** Svanberg's scheme, specialized here to
one volume constraint. Each design variable carries a lower and an upper
asymptote; objective and constraint are replaced by separable convex
approximations with a pole at each asymptote, and the subproblem's dual is a
single scalar the volume multiplier is bisected on, exactly as `oc` bisects its
own. The asymptotes then move: a variable that reversed direction has its range
shrunk (damping it), one that kept going has its range widened (letting it
accelerate).

Pick `mma` when a run **will not settle**, which in practice means an
`[optimization.overhang]` run on an aggressive geometry. The self-supporting
filter turns a design variable into a near-discrete "does this column print or
not" decision, and the multiplicative `oc` ratio cannot represent a decision like
that: it drives the variable across the box every iteration and the run ends at
the iteration cap with a design variable change well above `convergence_tol`.
Damping exactly those variables is what the moving asymptotes are for.

The shipped `examples/shelf_bracket.toml` is that case, which is why it is the
one example that sets `update = "mma"` itself. Both schemes over its own budget
(150 iterations, `convergence_tol = 0.01`):

| | converges | final max change | compliance |
| --- | --- | --- | --- |
| `oc` | never | 0.12882 at the cap (0.02542 at iteration 120) | 1.926071e1 |
| `mma` | **at iteration 132** | 0.00470 | 1.879277e1 |

and on a 120 mm deep variant of the same bracket, which is the case `oc` cannot
do at all: `oc` sits on its 0.2 move limit for 400 iterations with the compliance
still wandering inside a 1.6 % band at the end of them, while `mma` converges at
iteration **214** on a compliance of 2.689e1 - some 36 % below the 4.220e1 `oc`
was still moving around - having spent its last 30 iterations inside 0.1 %. Those
two trajectories are also what the stall criterion was written against: it stops
the first and never touches the second. See [When a run
stops](#when-a-run-stops).

Two things to expect when you switch:

* **More iterations, not fewer.** MMA is more conservative per step: it needs
  about 10 % more iterations than `oc` spends reaching its *cap*, and it uses
  them to actually converge. Raise `max_iterations` - the shipped bracket asks
  for 150 to converge at 132.
* **A slightly larger overhang residual.** MMA settles on designs whose printed
  and blueprint densities differ a little more (0.020 against 0.012 mean on the
  shipped bracket): it keeps blueprint material the filter is still trimming
  where that buys stiffness. The exported surface is the printed field either
  way, so it stays printable.

On a problem `oc` already handles - no overhang filter, no design dependent load
- there is nothing to gain. The two land in the same place:
`examples/cantilever.toml` at its 150 iterations gives 7.850071e0 under `oc` and
7.817574e0 under `mma`, 0.4 % apart. `mma` is not worse there, but it is not
better either, and it leaves a larger design variable change behind (0.113
against 0.017) because it is still trading material between two equally good
arrangements. `oc` remains the right default.

Self-weight needs no special handling under MMA. A design dependent load can
make `dC/dx_e` positive, which is what the `oc` step's sensitivity shift exists
to repair; MMA represents a positive sensitivity directly (it lands in the upper
asymptote term and pushes that variable down), so no shift is applied and none is
announced.

`mma` is also what a second constraint needs: the
[local volume cap](#local-volume-constraint-bone-like-structures) is priced by a
dual in two multipliers, and the `oc` ratio is bisected on one. A configuration
that asks for the cap under `oc` is rejected rather than run.

## When a run stops

```toml
[optimization]
max_iterations = 1000   # optional, default 1000
convergence_tol = 0.01  # optional, default 0.01
```

A SIMP run ends for one of three reasons, and the summary line names which:

```
iterations     682 (converged)
iterations     361 (stalled)
iterations    1000 (iteration cap)
```

**`converged`** - the largest design variable change of an iteration fell below
`convergence_tol`. The design settled; this is the answer.

**`stalled`** - the run stopped making progress and said so. For 100 iterations
in a row every step was clipped by the update scheme's move limit - the optimizer
asking to move further than it is allowed - and the compliance established no
better design in all of them. The design is a real one: it meets the volume
constraint, it is exported, it is stress analysed, and the run exits zero exactly
as a converged one does. It is simply the iterate the problem will not improve on
rather than a settled answer, and the run says what to do about it:

```
stopped after 361 iterations: the compliance has established no better design
for 100 iterations and the design is still traversing at the oc move limit (max
change 0.20000 against the 0.01 asked for) - this problem does not settle under
update = "oc"; consider update = "mma", which damps exactly the variables that
keep crossing the box
```

and under `mma`, which has no third scheme to point at, it says to raise
`max_iterations` or to take the iterate.

**`iteration cap`** - `max_iterations` ran out with the design still moving and
still improving. The design is whatever the budget ended on, and more budget
would have bought a better one.

The default budget is **1000**, not a target: it exists so that a run converges
instead of stopping at a wall clock compromise. In the skeletal regime this tool
is for (`mass_fraction` near 0.10) the old default of 150 routinely cut runs off
while they were still descending: a 1.5 mm cantilever at `mass_fraction = 0.10`
converges at iteration **682** (2 min 52 s on an RTX 3080), and the design it was
handed at 150 carried a compliance 17 % higher than that. A budget you set
explicitly is still obeyed to the iteration - every shipped example asks for 150
and gets exactly 150 - and `growforge edit`'s fast previews keep their own much
smaller cap.

### What the stall criterion is, and what it is not

Some problems never settle. The 120 mm deep variant of the shipped shelf bracket
under `oc` is the documented one: every one of its 400 iterations takes a step of
exactly the 0.2 move limit while the compliance wanders inside a band, and it
would do the same for 4000. Nothing about a single iteration tells that run from
one that is converging slowly - both are pinned at the move limit - so the test
is over a **window of 100 iterations**, and it asks two things of every one of
them:

1. **Is the design still traversing?** Every change in the window is at least
   **80 %** of the move limit, and the smallest change of the window's second
   half is no more than 5 % below the first half's.
2. **And is it buying anything?** The best compliance of the second half betters
   the best of the first half by less than **0.5 %**. (Best, not last: a
   descending run keeps setting new lows through its own oscillation, and a
   wandering one stops setting them.)

The thresholds come from six runs traced iteration by iteration with the test
switched off. Each row is that run's *worst* 100-iteration window - the one that
came closest to being called a stall:

| run | outcome | smallest change held for a whole window | what that window bought |
| --- | --- | --- | --- |
| deep bracket, `oc` | 400 = the cap | **100 %** of the move limit | +0.2 % |
| deep bracket, `mma` | converged at 267 | 57 % | +0.4 % |
| 1.5 mm skeletal cantilever, `oc`, 0.10 | converged at 682 | 53 % | +3.3 % |
| `mbb_bridge.toml` | 150 = its cap | 29 % | +0.5 % |
| `cantilever.toml` | 150 = its cap | 15 % | +0.3 % |
| `shelf_bracket.toml`, `mma` | converged at 132 | 14 % | +0.3 % |

The separation is in the third column, not the fourth: a run that will not settle
spends *every* iteration clipped by the move limit, and a run that is working
towards an answer - even one taking hundreds more iterations to get there - lets
the change off the limit long before it arrives. The compliance leg cannot
separate these on its own, which is why it is the second question rather than the
first: a converging run's last hundred iterations buy just as little as a stalled
run's. Replayed over these traces the criterion fires on the `oc` deep bracket at
iteration 255 and on nothing else; run live it stopped that bracket at 361. Which
iteration it lands on follows the wander and moves from run to run - what does
not is that it is the only one of the six it stops.

The bias is deliberately asymmetric, because the two mistakes are not: a stall
that is missed costs the rest of the budget, which is exactly what growforge did
before this existed, while a converging run falsely called stalled loses the
design it was on its way to. Anything the test cannot read clearly - a window
that is not full, a non-finite number - reads as "not stalled". Every threshold
lives in `src/constants.rs` with the measurement behind it.

## Enclosed voids

```toml
[output]
voids = "warn"     # "warn" (default) | "fill"
```

After the last iteration, the final density field is thresholded at
`output.iso_level` and the below-threshold cells are flood filled from the grid
boundary with six-connectivity. Anything the boundary cannot reach is a cavity
the exported surface would seal: unprintable without trapped support, and easy to
miss in a slicer preview. Face connectivity is the conservative reading, so a
group joined to the outside only across an edge or a corner still counts as
enclosed.

Both modes always report every cavity with its cell count, volume and centroid.
`fill` additionally turns them into material before the surface is extracted and
reports the volume and mass that added. The pass runs before marching cubes, so
the report always describes the file that was written.

One exception, and it is deliberate: a cavity that overlaps a `[[keepout]]` or
the space outside the domain is **never** filled, because filling it would break
a constraint that was asked for. Such a cavity is reported as still present with
the reason attached.

## Islands

```toml
[output]
islands = "cull"   # "cull" (default) | "keep"
```

A run reports connectivity **twice**, because there are two objects to be
connected and they are not the same one:

```text
field bodies   1 connected body in the density field
mesh bodies    2 in the exported surface (+1 cavity shell); culled 3 floating
               fragments (1.940 mm3 in total, largest 0.447 mm3)
  body 1        13480.759 mm3 in 1976 triangles, holds support, keepin
  body 2         6040.655 mm3 in 2200 triangles, holds support, load
```

`field bodies` is the cell level reading: the density field thresholded at
`iso_level` and flood filled with face connectivity. It describes the object the
stress solve and the mass figures describe.

`mesh bodies` is the surface that was written: its triangles partitioned into
connected components by shared vertices. Marching cubes welds by lattice edge, so
two triangles share a vertex index exactly when they share a corner of the
surface, and the labelling is a union-find in triangle order - deterministic, and
unchanged by supersampling, which only means the components are found on the
refined surface.

**The two genuinely disagree, in both directions.** A node value is the mean of
its eight surrounding cells, so a lone dense cell peaks at 0.125 and makes no
surface at all, while a clump of cells joined to the part only through a
one-cell-wide bridge makes a surface that the bridge, averaged with its six empty
neighbours, cannot reach. The first reads as bodies the file does not have; the
second - reported by a user at `mass_fraction = 0.12`, with the run printing one
connected body over an STL carrying floating shells - as bodies the file has and
the field does not. Which is why both lines are printed and each names its object.

Culling acts on the mesh, after smoothing and **before validation**, so what the
validator accepts is the file that ships rather than a superset of it. What
decides it is **purpose, never size**:

* an **outward** wound shell is kept when it touches a `[[supports]]` region, a
  `[[loadcases.loads]]` region or a `[[keepin]]` cell, and culled when it touches
  none of the three. Geometry that serves nothing the configuration declared is
  debris; geometry that serves something declared is deliberate, however small it
  is and whatever else is in the file. Every kept body says what holds it, and
  every culled fragment is reported with its volume, triangle count and centroid;
* an **inward** wound shell is the inside of a cavity, and has no purpose of its
  own. It belongs to the innermost outward shell that encloses it - decided by
  the solid angle that shell subtends at one of its vertices, which needs no ray
  direction to be chosen and has no parity to lose - and survives exactly when
  that shell survives. So a cavity kept by `voids = "warn"` keeps its inner
  shell, and a cavity inside a fragment leaves with the fragment rather than
  being orphaned in the file.

**Volume decides nothing.** Ranking by it was tried and is wrong: a `[[keepin]]`
boss is paid for out of the same `mass_fraction` as the structure and can be the
larger of the two, so "the biggest component is the part" holds an STL that is
the boss with the load path culled out of it. A part is as many pieces as the
configuration asked for.

"Touches" is two questions, because a part one voxel thick and a part ten voxels
thick need different ones asked:

* **on the lattice.** A vertex counts as touching every node of the lattice cell
  it lies in - the surface came off that lattice, so matching an index is exact
  where matching a point against a shape would have to pick a tolerance, and the
  lattice cell is the half voxel of slack smoothing and the iso nudge can move a
  vertex by;
* **by containment.** A component that *encloses* one of a region's probes - cell
  centres inside the material that region selected - holds that region too. The
  shipped `cantilever.toml` needs this: its load region sits in the middle of a
  keepin pad, several voxels from the nearest surface, so no vertex of the part
  is anywhere near it.

Neither alone is enough. Material a voxel thick shrinks to a surface that may
enclose no cell centre at all, and material ten voxels thick has no vertex within
a voxel of its interior.

The probes are spread over the region in **space**: the region's own bounding box
is partitioned into at most `ISLAND_ANCHOR_PROBES_PER_REGION` boxes and one probe
is taken from each occupied one. Every axis with an extent is split before any
axis is split twice, and what is left of the budget goes to the axis with the
widest boxes. Both halves of that rule are defects this walked into and now
guards:

* taking every n-th cell of the region's cell list is **aliasing** - the list is
  in raster order, so a pad split left and right by a gap, 21 cells to a row and
  168 of them, makes the stride an exact multiple of the row and lands all eight
  probes in one patch;
* growing the partition widest-first alone is **starvation** - a pad 100 cells
  wide and 12 tall spends the whole budget on x and probes a single row of y.

What this buys, precisely: no periodic row structure and no dominant axis can
capture every probe. What it does not buy, and no finite partition can: a gap
**narrower than one box** can lie inside one, so a region split by a thin gap may
be probed on one side only, and the component holding the other side can then be
culled. That case is *reported* rather than prevented, by the check below - which
is deliberately not built on any sampling of the region.

After the cull, two things are named. A support or load region that **no exported
body reaches**:

```text
  warning      load 1 of case "tip" is reached by nothing in the exported
               surface: the part that ships does not connect it
```

and, per culled fragment, any declared region it was **inside**:

```text
  warning      culled fragment 2 (41.880 mm3) lies inside load 1 of case "tip":
               material a declared region asked for was disconnected and removed,
               so the load path may be incomplete
```

The second exists because the first structurally cannot cover the partial case: a
region spanning two disconnected lobes is still *served* by the lobe that
survives, so it never appears unserved, while the lobe that leaves takes material
the region asked for with it. Every fragment that is about to be removed is
therefore tested against the region shapes themselves - the same SDFs the
configuration is written in, a handful of fragments against a handful of regions -
and the fragment note drops its "nothing declared asked for it" wording whenever
this fires, because there it would be false.

Two floors keep the rule from removing geometry on no evidence: nothing is culled
when no body reaches anything declared at all, and nothing is culled under
`islands = "keep"`. Both are said in the report rather than assumed.

A body that is anchored but **tiny** ships and is named:

```text
  warning      body 2 holds keepin and is kept for it, but encloses 122.567 mm3
               against the 268.083 mm3 of a sphere of min_feature_mm (8.000 mm):
               it ships as a separate loose piece
  remedy       a body that small is a design that did not resolve rather than a
               feature; more iterations, a coarser min_feature_mm or a higher
               mass_fraction join it to the part. It is not removed: a declared
               region asked for the material in it
```

The threshold is the sphere of diameter `min_feature_mm` - the smallest lump the
density filter can resolve at all - scaled by `ISLAND_TINY_BODY_SPHERES`, and it
is only ever a *warning*: a stub the optimization left behind is still material a
declared region asked for, and size culls nothing. It is not said for a single
body, because a small part is a part rather than a loose piece.

Nothing is rebuilt when there is nothing to remove: a surface whose components
are all anchored, cavities and all, is handed to the validator and the writer
untouched. An export with no fragment in it is therefore byte for byte the export
it was before this pass existed, which is what keeps the shipped examples'
recorded outputs where they are.

`islands = "keep"` exports the extracted surface, fragments and all, and still
reports them. It is the setting for looking at what the optimizer really
produced.

## Exact boundaries

Everything before the export works on a voxel field, and a voxel field does not
know where a shape is. A cell is classified by its **centre**, so a cell whose
centre lies a hair outside a keepout stays material for its whole width and the
modelled solid reaches up to half a voxel into the forbidden region; marching
cubes then puts vertices on the node lattice, and Taubin smoothing moves them
again. The result is an exported surface that cuts into a keepout and bulges out
of the domain, by a fraction of a voxel each. On a pin bore that is not cosmetic:
a 2.75 mm bore meshed on a 1.5 mm grid comes out **0.087 mm under radius** on
this project's own test fixture, and the pin does not fit.

Supersampling does not help. The refined lattice is the trilinear interpolant of
the *same* coarse node field, so it describes the same encroaching surface with
more triangles.

```toml
[output]
boundaries = "exact"       # optional, default "exact"; "voxel" is the old behaviour
```

Under `"exact"`, after the island cull and before validation, every exported
vertex that lies inside a keepout or outside the domain is projected onto the
analytic surface it violates - the surface the configuration described, not the
grid's idea of it - and left a documented ten nanometres on the legal side so a
containment test agrees. Where the part meets a bore, the bore **is** the
cylinder.

The scatter goes both ways, so the correction does too. The sampling does not put
a wall's vertices reliably *outside* the surface - it puts them on both sides of
it, and the smoothing that rounds the staircase pulls the corners inward - so
correcting only the vertices that are proud of a boundary leaves the inward half
as **dimples**: measured at 0.70 mm, half a voxel, on this project's plug
fixture, and visible as scalloping on what was drawn as a cone. A vertex that
violates nothing but rests within `BOUNDARY_CLAMP_CAPTURE_VOXELS` (half a voxel,
the scale a cell-centre classification can be wrong by) of a boundary is
therefore seated onto it as well, by the same projection onto the same legal
side. Anything further away rests on nothing - an optimizer's free surface
through the middle of the domain - and is left exactly where the smoothing put
it, and a seat that would land a vertex proud of another boundary is dropped.
What that costs is stated plainly: a free surface running *within* half a voxel
of a boundary is treated as resting on it, which is a gap the voxel field cannot
resolve anyway, and the correction is always onto the legal side.

| shape                          | how it is projected                         |
| ------------------------------ | ------------------------------------------- |
| box, sphere, capped cylinder    | closed form, exactly the nearest surface point |
| cone (frustum or true cone)     | closed form: the shape is a surface of revolution, so the nearest point lies in the sample's own meridian half plane, and there it is the nearest of three segments - the two caps and the slanted wall |
| triangular prism                | closed form: the sample clamped into the triangle across its plane and into the slab along its normal, which is the box's rule on a shape whose sides are not axis aligned |
| tube (straight or bent)         | closed form: the nearest point of its centre line, a radius out. A tube that overlaps itself - because it **folds** through the centre of its own bend, or because its two **ends have closed** on each other across the gap a bend of more than half a turn leaves open - has swallowed part of its own inside and is no longer offsettable that way; it is refused rather than answered wrongly, and a `[[keepout]]` holding one is **warned about** by `growforge check`, by every run and by the editor's validity line |
| ellipsoid                       | bounded Newton steps on its own field, which is exact on the surface (the nearest point is the root of a sextic and has no closed form) |
| the domain                      | bounded descent onto the level set of the ordered CSG composite, which has seams and no nearest-point formula |

Three things bound it, because a projection allowed to do anything can wreck a
surface:

* **Overlapping keepouts** hand a vertex to one another, and leaving a keepout
  can leave the domain, so a corrected position is examined again - up to
  `BOUNDARY_CLAMP_MAX_PASSES` rounds.
* **A displacement cap.** The encroachment this exists for is sub-voxel by
  construction, so a vertex that would have to move further than
  `BOUNDARY_CLAMP_MAX_DISPLACEMENT_VOXELS` (one voxel) is not this defect and is
  left where it is.
* **An honest give-up.** A vertex still illegal when the budget runs out, or past
  the cap, keeps its position and is counted. The run says so:

```text
boundaries     clamped 412 vertices onto the analytic surfaces, moving them 0.0874 mm at most
```

and, when there were any it could not correct, a `warning` line naming how many.
Both reach the console and the editor's panel. The clamp runs **after** the
island cull - no work on fragments that are about to be discarded, and the
culling verdicts see the geometry they always did - and **before** validation, so
the positions that are validated are the positions that are written; a clamp that
collapsed a triangle fails the export rather than shipping one.

`boundaries = "voxel"` exports the isosurface exactly as the field produced it.
That is what growforge did before 0.22.0, and it is the setting for reproducing a
file written by an older version - **the default changes the STL of an existing
configuration**, which is the point of it.

The live preview inside the viewer is never clamped: it runs marching cubes alone
and never enters the export pipeline. The surface the window switches to when a
run finishes is the exported one, clamp included.

## Stress report

After the last iteration, the final field is solved once more per load case and
the von Mises stress is recovered at every element centroid. The run prints

```text
stress         3063 elements at density >= 0.50, recovered with the full solid modulus
  loadcase                  max MPa      p99 MPa  top 10% MPa     safety
  shelf-load                 1.6079       1.2674       1.1195      29.23
```

the maximum, the 99th percentile, the mean of the top decile, and
`yield_strength_mpa / max`, which needs a yield strength in `[material]` (every
preset has one) and reads `n/a` without it. Adding

```toml
[output]
stress_json = "shelf_bracket_stress.json"   # relative to the config file
```

writes the same numbers as JSON.

The editor says it too, and in text: the panel's **stress** block puts the safety
factor first, with the peak of every load case under it, and the same lines are
echoed to the console the session was launched from beside the `editor wrote`
line - after a full run and after `generate stl` alike.

```text
editor wrote  bracket.stl
editor stress safety factor 6.23 (peak 8.0269 MPa vs yield 50 MPa)
editor stress tip peak 8.0269 MPa
```

Until 0.23.0 an editor session could only look at the colours: the number was
computed, painted and printed by `growforge run`, and readable nowhere in the
window. Panel and console share one formatter with the table's own numbers, so
the three cannot disagree, and the block describes the run on screen - it is
absent while the next one runs rather than left behind.

### An export that came out in pieces

A safety factor is a factor of the thing that was analysed, and when the
exported surface holds more than one body that thing is not one part. Every
surface the summary reaches says so, above the number:

```text
stress         512 elements at density >= 0.50, recovered with the full solid modulus
  warning      the export is 2 separate bodies - this safety factor describes each piece against its own supports, not one connected part
```

```text
editor stress warning: the export is 2 separate bodies - this safety factor describes each piece against its own supports, not one connected part
editor stress safety factor 5.07 (peak 9.2700 MPa vs yield 47 MPa)
```

and the editor's panel draws the same line in the warning colour above its
headline. The count is the **mesh bodies** line's own - the bodies the island
cull kept in the file that was written, so this and that line can never
disagree.

Nothing here can tell you that a support is fictitious. A part meant to link two
rods whose load groups each shunt into their own local support anchor is a model
that solves correctly, converges, and reports a healthy factor for two pieces
held apart by air - and the placed regions are what the configuration declared,
so there is nothing to check them against. What a run does know is that its
surface came out in pieces, which is what this says. Several bodies is not
always wrong (a keepin boss beside a bracket is two of them, and both were asked
for), so it is a warning rather than a refusal - but if the pieces were meant to
be one part, the factor above it is not a reading of anything.

Read the numbers with these caveats:

* **Intermediate densities.** Elements below a physical density of 0.5 are left
  out, and the ones above it are recovered with the *full* solid modulus `E0`,
  which is standard SIMP stress-recovery practice. A half dense element is
  therefore reported as if it were solid, so the elements just above the
  threshold are the least trustworthy numbers in the table.
* **Discretization.** One trilinear hexahedron per voxel under-estimates peaks at
  re-entrant corners and along the staircase edges of the voxelized boundary. The
  percentile and top-decile columns are there because the maximum is easily one
  such corner. Treat the table as a screening figure, not a certificate.
* **The objective is stiffness, not strength.** Compliance minimization has no
  stress constraint in it, so a converged design usually lands with a generous
  safety factor and the report is how you find out when it does not.

The stress pass cannot influence the optimization: it runs after convergence, on
the field that is about to be exported.

### When the stress solve does not converge

The stress solve is the hardest linear system a run ever poses: it is cold (no
warm start), it runs on the final near-0/1 field where the stiffness contrast is
at its worst, and unlike an optimization iteration it gets no second chance. It
therefore has its own budget, separate from the optimization path's, and it is
allowed to fail:

```text
stress         report unavailable: solving load case "tip-down" for the stress
               report: conjugate gradient did not converge in 50000 iterations
               (relative residual 3.482e-6, target 1.000e-6)
               the density field, the STL and the cavity report above are unaffected
```

**A stress solve that will not converge degrades the report, it does not fail the
run.** The density field is final either way and the part is what was asked for,
so the STL is still written, the cavity report is still printed, the run still
exits zero, and only the stress table is missing. No `stress_json` is written
when there is nothing to write, and the viewer's `colour by von Mises stress`
switch stays disabled.

The line is drawn precisely at *solving*. What degrades is a solve, by a solver
that was successfully opened and successfully bound to the design, that did not
converge in its budget or that broke down. Everything else is a hard error and
exits non-zero, including the two nearest neighbours of that case: a solver
backend that cannot be **opened** (`backend = "gpu"` with no adapter, a device
that refuses, a problem past the adapter's buffer limits) and a design that
cannot be **bound** to one. Neither of those says anything about the structure -
both mean the run was asked for something this machine cannot do - and failing
them softly would be at its worst under `engine = "growth"` with
`backend = "gpu"`, where growth performs no solves of its own and the stress pass
is the *first* place a backend is opened at all.

One failure mode is caught before the solve rather than by it. A load whose
region reaches no support **through material** drives a system whose only path
from the loaded nodes to the constrained ones runs through cells at the SIMP
stiffness floor - a factor of 1e9 down at its default - and no conjugate gradient
resolves that;
it finds out by spending its whole 50 000 iteration budget and then saying so.
The analysis already flood fills the field for the enclosed-cavity and
solid-body reports, so the same fill answers the question first, and the report
degrades immediately with what to do about it:

```text
stress         report unavailable: load 1 of load case "tip" is not connected to
               any support through material, so the system it drives is
               near-singular; solving it would only spend the iteration budget
               to say so
```

The threshold for "material" here is deliberately far below any iso level
(`STRESS_LOAD_PATH_DENSITY`, 0.05): the question is whether there is *any*
stiffness on the path, not whether the path is part of the printed part. A
bridge at a tenth density carries 1e-3 of the solid modulus and solves perfectly
well, and is not reported.

The two knobs behind that budget are compile-time constants rather than config
keys, because they are properties of the recovery rather than of the part:
`STRESS_CG_MAX_ITERATIONS` (50 000, five times the optimization cap) and
`STRESS_CG_TOLERANCE` (1e-6, two decades looser than the optimization's default
1e-8, and unaffected by `[solver] tolerance`).
Stress recovery differentiates the displacement field, so its error is first
order in the residual; at 1e-6 the recovered von Mises values are stable to parts
per million, four orders below the percent-level discretization error a single
trilinear hexahedron per voxel carries at a staircase boundary. The two extra
decades the optimization asks for buy nothing this model can resolve, and they
are exactly what makes the final system expensive.

## Trimming unloaded material

```toml
[output]
trim = "off"                 # "off" (default) | "stress"
trim_stress_fraction = 0.01  # optional, default 0.01, open interval (0, 1)
```

The prongs that reach to nothing and the gossamer wisps between members. A
density optimization converges on a *field*, not on a structure, and it keeps a
little material wherever removing it would have cost the objective nothing;
thresholded at `iso_level` that material becomes geometry - thin spurs standing
off the load path, hairs bridging two members, lobes that end in mid air. None of
it is load bearing, all of it is printed, and most of it needs support to print
at all. `trim = "stress"` removes it after the run.

**The criterion is the stress envelope**, which the run has already computed: the
pass costs no extra solve. Every load case is solved once for the stress report,
the envelope is the per-cell maximum von Mises stress over all of them, and a
cell of the part is a candidate when its envelope stress is below
`trim_stress_fraction` of the envelope's own peak. The envelope rather than any
one case, because a member that is idle in one case and working in another is
working. A fraction of the peak rather than an absolute stress, because the
number has to mean the same thing for a bracket at 2 MPa and a mount at 200. One
percent is deliberately far out on the tail: a genuine load path carries stress
by definition, and the ratio between a structure's peak and the stress in a
member that is actually working is a small number, not two orders of magnitude.

The pass only ever removes material the report **measured**. Elements below a
physical density of 0.5 are left out of the stress report entirely (see the
stress section) and carry a zero, and that zero is an absence of evidence rather
than a measurement of none - so a cell the report did not cover is never a
candidate. The limitation that buys, stated plainly: with `iso_level` below 0.5
the part includes cells the report leaves out, and the wisps among them survive
the trim. At the default iso level the two thresholds are the same number and
there are no such cells.

Two things are never removed, however little stress they carry:

* **Everything declared.** The cells of every `[[supports]]` region, every
  `[[keepin]]` entry and every non-gravity load region. A support pad is
  unstressed *because* it is held; a keepin is material the configuration asked
  for by name. A gravity load declares no region - it acts on every element -
  so it protects nothing in particular.
* **Anything whose removal would take the structure apart.** Before and after
  the tentative removal, those regions are located on the connected bodies of
  the field and the pairs of them that share a body are compared. If a pair that
  was joined no longer is, the **whole pass is refused** - not the offending
  cells, the pass - the field is left exactly as it was, the run exports
  untrimmed, and the warning names both ends of what would have been separated:

```text
trim           warning: refused, and the part was exported untrimmed: removing
               the 412 cells below 0.0461 MPa would have left support 1 joined
               to load 1 of case "tip" through no material at all
```

Connectivity is face connectivity, six neighbours, which is the reading every
structural question in growforge uses: a diagonal touch is not a joint. There is
no partial outcome by design - a half-applied trim is a structure neither the
optimizer produced nor the user asked for.

A pass that removed something runs **the whole analysis again**: cavities are
re-resolved and the stresses re-solved on the trimmed field, so the stress table,
the safety factor, the void report, the JSON and the STL all describe the
geometry that shipped. The pre-trim stresses are the criterion and nothing else,
and they survive only as the two numbers in the note:

```text
trim           removed 2387 cells, 152.768 cm3, 18.6% of the part, below
               0.0461 MPa (1.0% of the 4.6066 MPa envelope peak)
```

The note reaches the console and the editor's panel alike. Any fragment the
removal orphans is swept up by the island pass downstream under the usual
[`islands`](#islands) policy.

Two things the pass is deliberately not:

* **A printability check.** The overhang filter shapes the design *during* the
  optimization; this happens after it, and the cut faces it leaves are not
  re-checked against the build direction. That is the same standing smoothing
  and island culling have - all three change the surface after the constraint
  has done its work - and it is why what is removed is the far tail of the
  stress distribution rather than anything a slicer would have leant on.
* **[`[growth] prune`](#pruning-no-branch-may-end-on-nothing).** That removes
  branches of a skeleton while the growth engine is still building one, by
  whether they reached anything. This removes cells of a finished density field,
  by the stress in them. The growth engine's own pruning means a grown design
  rarely has much left for this to find, but the pass is engine agnostic and
  legal on either.

A run whose stress solve did not converge has no criterion, so it trims nothing
and says so rather than removing material on no evidence.

## Reinforcement (minimum printable thickness)

```toml
[output]
reinforce = "off"            # "off" (default) | "min_thickness"
reinforce_thickness_mm = 3.0 # optional, default [optimization] min_feature_mm
```

The arms that came out too thin to print. An optimizer spends its material where
the compliance pays for it, and at the far end of that trade it builds members a
fraction of a millimetre wide because the objective asked for them. A slicer
meets a wall thinner than one extrusion and lays a single bead where a member was
meant to be, or drops the feature altogether. Nothing upstream prevents it:
`min_feature_mm` is the radius of a *density filter*, a smoothing length that
stops the design changing faster than that, and it is not a floor on what the
design converges to. `reinforce = "min_thickness"` is the floor, applied to the
field after the run.

**The measure is the inscribed ball.** A Euclidean distance transform over the
part gives every cell the distance from its centre to the nearest cell that is
not part of it, and the surface lies half a voxel inside that cell - so a member
`w` voxels across measures `w` voxels thick. Twice that distance is the local
thickness, but only where the distance is a **local maximum**: on that spine the
inscribed ball touches the boundary on opposite sides, so its diameter is the
thickness of the member; anywhere else it touches the near surface alone and
twice the distance is a distance to the surface rather than a thickness. So the
pass measures thickness where the measurement means something, at the cost of one
distance transform and no solve at all - how thin a member is, is geometry.

**The operation is a ball fill.** Every spine below the floor has every *design*
cell within half the floor of it raised to full density. That gate is the whole
of the safety story: keepouts, the space outside the domain and the cavities the
`voids` policy owns are all void cells, forced solid cells are already full, and
the fill writes to neither.

**What cannot be done is reported, not refused.** A member pressed against a
keepout wall or the edge of the domain has nowhere to grow into. The pass runs
the distance transform a second time over the field it just changed and re-reads
every place it called thin - asking there whether a ball of the floor's diameter
now fits inside the part and reaches the place, which is the part *opened* by
that ball. Filling moves a member's spine, so the same spine reading a second
time would condemn members that came out exactly right; what is counted are the
places a ball of the floor's diameter still does not reach:

```text
reinforce      reinforced 3184 cells, 12.736 cm3, onto arms below 3.000 mm
               warning: 2 places are still thinner than 3.000 mm and the
               exported part is thin there: the material is pressed against a
               keepout or the domain edge, and the floor cannot be met without
               growing into it
```

Everything it *can* reach is reinforced regardless - there is no pass-level
refusal here, unlike the trim's connectivity guard, because thickening one member
cannot take the structure apart. Both lines reach the console and the editor's
panel alike.

**The mass shift.** Run with `trim = "stress"` the two passes are the two halves
of one exchange: the trim frees the mass of the prongs that carry nothing, and
this spends mass on the arms that carry something and cannot be printed. The
notes sit beside each other so the trade reads directly. The order is trim,
[flush](#flush-fill), then reinforce, and **nothing is trimmed afterwards**:
reinforcement material is
deliberately unloaded - it is there so the member can be printed, not so it can
carry anything - and a second trim would remove exactly what was just added. A
run in which either pass changed the field re-runs the cavity pass and the stress
solve **once** over the result, so every number the run reports, the JSON and the
STL included, describes the part that shipped.

**Not to be confused with the growth engine's thickening.** Step 4 of
[the growth engine](#growth-engine) sizes the struts of a *skeleton* from
Murray's law while the design is still being built, and it is about how much
material a branch deserves. This measures the *finished density field* about to
be exported, under any engine, and is about what a printer can lay down. The one
is a structural rule, the other a manufacturing floor.

Three smaller things worth knowing:

* **Where the fill fits, the floor is met or exceeded, never approximated
  down.** The fill radius is half a voxel longer than half the floor, because a
  filled cell puts material out to its own face; a member reinforced against a
  3 mm floor comes out at 3 mm or a little over. Where it did not fit, the
  warning above is what says so.
* **The free tip of a member is thin by construction**, and so is the sharp
  corner of a block: no inscribed ball of the part's thickness contains either
  point. The floor is a statement about members, which is what the spine is.
* **`[optimization.local_volume]`** is a constraint on the *optimizer*. Cells
  this pass thickens may take a neighbourhood past the cap in the exported field,
  and the report says so: printability outranks the cap.

## Flush fill

```toml
[output]
flush = "off"        # "off" (default) | "walls"
flush_depth_mm = 3.0 # optional, default 2 voxels of the grid
```

The walls that rest against a shape the configuration drew, but come out short
of it. An optimizer has little reason to resolve the last cells of a wall
pressed against the edge of the domain: they come out at 0.4, 0.6, 0.8 in a band
that should be solid, because the compliance barely notices and the density
filter smooths across the boundary as if there were design space on the far side
of it. Read at `iso_level` that band is not a wall standing on a surface - it is
a wall that dips inwards here and reaches the surface there, and the exported
face ripples where a flat one was drawn.

**The mesh side of this is already solved, and it is not enough.**
[`boundaries = "exact"`](#exact-boundaries) seats every exported vertex resting
within half a voxel of a surface exactly onto it. Half a voxel is the scale a
cell-centre classification can be wrong by, and it is deliberately all the clamp
will reach: a vertex further away than that rests on nothing - it is an
optimizer's free surface through the middle of the domain - and moving it would
drag that surface onto a wall it was never near. So where the field dips deeper,
the clamp is right to leave the vertex alone and the ripple survives into the
file. What is missing is material, and only the field can be given it.

**The predicate is a band, grown from the wall inside it.** A design cell is
raised to full density when all three hold:

* it is not already there;
* its centre lies within `flush_depth_mm` of a constraint surface - the domain's
  own boundary or the wall of the nearest keepout, whichever is nearer - the
  *band*;
* the part's own material, **inside that band**, comes within `flush_depth_mm`
  of it, by an exact Euclidean distance transform seeded on those cells.

The third condition is what makes this a correction rather than a coat of paint.
A stretch of boundary with no wall on it - the open face of a bracket, the mouth
of a bore the design never reached - has no material in the band anywhere near
it and comes through untouched. And it is the material *in the band* rather than
the part's material anywhere, because a member running two voxels clear of a
face is not a wall resting on it: seeded on the part at large, a bar through the
middle of a domain would grow a detached plate on every face it passed near.

Only design cells are ever written, which is the whole of the safety story: a
keepout, the space outside the domain and a cavity the `voids` policy owns are
all void cells, forced solid cells are already full. The pass can only ever add
material, so there is nothing it can disconnect and no pass-level refusal here.

**The caveat, which is inherent and is why the pass is opt-in.** Within
`flush_depth_mm` the fill cannot tell a pockmark in a wall from the end of one.
Material that stops inside the band gets joined to the surface it stopped near,
and a wall's edge grows a fringe of up to `flush_depth_mm` along the boundary it
rests on. That is the same reach that fills the ripple, seen from the other
side. The note says so, and the depth is the knob:

```text
flush          filled 288 cells, 2.304 cm3, out to the surfaces the walls rest
               on, 4.000 mm deep
               material that stopped within 4.000 mm of one of those surfaces is
               now joined to it: within that depth the fill cannot tell a
               pockmark in a wall from the end of one
```

`flush_depth_mm` is a length, and absent it is two voxels of the grid the run is
solved on - the artefact is a sampling one, and the voxel is its scale. Written,
it has to stay between half a voxel (below which it cannot reach past the cells
the surface already passes through) and eight (above which the pass lays a skin
over every wall instead of seating the ones that rest on a surface), and the run
is refused with both figures if it does not.

**Where it sits.** The order is [trim](#trimming-unloaded-material), then flush,
then [reinforce](#reinforcement-minimum-printable-thickness), and all three
share one re-analysis. The flush goes before the reinforcement on purpose: a
rippled wall reads as a row of thin places, and the reinforcement would spend
ball after ball on them and then warn that it could not reach the floor against
a boundary. Flushed first, it measures a wall of even thickness. Everything the
run reports afterwards - the stress table, the safety factor, the JSON, the STL
- comes from the analysis over the filled field, so the material the fill added
is in the safety factor rather than beside it.

The pass is applied **once**, to the field an engine produced, which is what the
pipeline does and what the editor's generate button does (it works on a copy of
the design it kept). A second application over its own output would extend that
fringe again wherever the first one left material at the edge of the band.

Measured on the shipped fixture - a wall lying on the floor of its own domain
with its outermost layer under the iso level - the exported floor stands
**1.6214 mm** off the face it was drawn against, 0.81 of a voxel and past the
1.0 mm the clamp captures by itself; with `flush = "walls"` every vertex of it
is on that face to 1.0e-5 mm, which is the clamp's own offset.

## Solver backend

```toml
[solver]
backend = "gpu"            # "gpu" (default) | "cpu"
tolerance = 3e-8           # optional; 1e-8 (default), 1e-10 .. 1e-4
```

The finite element solve is the whole workload of a `simp` run, and it runs on
the GPU unless you say otherwise.

| backend | arithmetic | reproducibility | speed |
| ------- | ---------- | --------------- | ----- |
| `gpu` (default) | f32 on the device, f64 refinement on the host | bit-for-bit on one machine and driver only | 7 to 12 times faster, see below |
| `cpu` | f64 throughout | bit-for-bit everywhere, for a given build and thread count | baseline |

**The default is the compute device, and it is a soft default.** A build without
the `gpu` cargo feature, or a machine with no usable adapter, runs on the CPU
instead and says so - in the `check` and `run` summary for the first, and as a
one-line notice the moment the device is asked for and refused for the second. A
run is never refused for want of hardware it never asked for by name.

**A backend you name is an instruction.** `backend = "gpu"` on a build compiled
without the feature, or on a machine with no adapter, is an error and always has
been: you asked for a device, and silently running somewhere else would make the
summary a lie. Only the default softens.

**What the default costs you is reproducibility.** The device arithmetic is
deterministic - fixed gather order, fixed reduction shape - but the run around it
is not quite: measured on `examples/cantilever.toml`, three GPU runs of the same
input land on the same compliance to every digit printed and on three different
STL files. Bit-for-bit reproducibility - the
promise the recorded regression trajectories and the shipped determinism hashes
are written against - needs `backend = "cpu"`, which is what every such test in
this crate names explicitly. Set it in your own configuration when a number has
to be reproduced somewhere else; the answers agree to far better than the
modelling error either way (see [Precision](#precision) below for the measured
differences).

Only the linear solve moves. The SIMP outer loop, the density and overhang
filters, the optimality criteria step, the sensitivities, the growth engine and
the meshing all stay on the CPU, and the volume constraint is therefore met
identically on both backends.

### Tolerance

`tolerance` is the relative residual `||r|| / ||b||` every solve of the
optimization loop is taken to, on either backend. It defaults to `1e-8` and has
to lie between `1e-10` and `1e-4`.

**The default is tight, and the target is hard.** A solve that runs into the
10 000 iteration cap short of it does not return a slightly worse answer; it
fails the run. A one-sided-anchor model here died at optimization iteration 99
with the CPU conjugate gradient capped at a relative residual of 1.558e-8 - a
factor of 1.5 from a target that was never a physical requirement - and had
nowhere to say so. `tolerance = 3e-8` carries that run to the end and changes
nothing a printed part could tell apart.

**Looser is faster and less precise, in that order.** The iteration count of a
conjugate gradient grows with `ln(1/tolerance)`, so decades are cheap to buy and
cheap to give back: measured on a small cantilever, four of them (1e-8 to 1e-4)
took one solve from 63 iterations to 44 and moved the compliance by an order of
1e-9 to 1e-10 relative - five orders and more below the loosest tolerance the
key accepts. The exact digit is not a property of the pair and is not quoted as
one: it moves with the thread count, because the host-side reductions the
compliance is summed by are ordered by it, for the reason
[Reproducibility](#reproducibility) documents. The displacement field is
accurate to about the residual, and well below 1e-6 that is orders finer than
the percent-level discretization error one trilinear hexahedron per voxel
already carries - anything from 1e-8 to 1e-6 is the same physics at a different
price. Above 1e-6 you are trading real accuracy, and the compliance history
starts to carry the solver's error rather than the design's. Tighten it instead
when a number has to be reproduced against another solver.

A target you loosened is named in the `solver` line of the `check` and `run`
summary - `solver cpu backend, bit-for-bit reproducible for this build and
thread count, tolerance 3e-8` - for the same reason the fallback is: the line
says what the run will really do. A run at the default says nothing about it.

**It is the optimization's tolerance alone.** Stress recovery runs at its own
`STRESS_CG_TOLERANCE`, a property of the recovery rather than of the part (see
[Stress report](#stress-report)), and the iteration caps
stay compile-time constants in both places: a cap is the guardrail that ends a
solve which has stopped converging, not a knob to be turned.

**And a target you cannot reach is not always the thing to move.** A solve whose
residual is still falling when the cap ends it - two and three decades short,
rather than the factor of 1.5 above - is not being asked for too much precision;
it is being handed a system too ill-conditioned to give it. What that costs is
set by the *stiffness contrast* across the load path, and the contrast is
`[optimization] stiffness_floor`: 1e-9 of the solid modulus by default, which is
nine decades for the conjugate gradient to cross. The measured case is a
one-sided anchor over a 204 mm span at 1.2 M degrees of freedom, which exhausted
40 000 device iterations at 3.5e-6 against a target of 3e-8 and then 10 000 CPU
ones at 4.3e-6. Raising the floor buys conditioning back for parasitic void
stiffness the design never asked for, and the trade is cheap while the floor
stays far below what the thinnest real member carries: at 1e-6 an emptied cell
is a thousandth of the stiffness of one at density 0.1, and nothing the part
does is measurably different. At the 1e-3 the key stops at, the void is carrying
the structure - see the SIMP interpolation in the
[configuration reference](#configuration-reference).

### Precision

WGSL has no `f64`, and these are the worst systems to hand single precision: late
in a run the stiffness contrast across the grid is 1e9 (the SIMP `Emin` floor at
its default `[optimization] stiffness_floor` - lower configured floors widen it
further, up to 1e12 at the bound) while the CPU backend promises a relative
residual of 1e-8. An f32 conjugate
gradient cannot reach that - f32 carries about 1.2e-7 of relative precision, so
its residual stops falling a decade or two above the target however long it
runs.

So the device never solves the problem. It solves for a *correction*, inside a
double precision iterative refinement loop on the host:

```text
r = b - K x                    f64, the same operator the CPU backend uses
y ~= K^-1 (r / ||r||)          f32, on the device
x <- x + ||r|| y               f64
```

repeated until `||r|| / ||b||` meets the tolerance the caller asked for. The
convergence test is made on the f64 residual of the f64 operator, so a GPU
solution satisfies the *same* criterion as a CPU one rather than a single
precision imitation of it. Normalising the right hand side to unit length before
narrowing it to f32 keeps the device arithmetic in the middle of its exponent
range however small the outer residual has become, and each pass asks the device
for only the accuracy the outer loop still needs.

Inside the device solve, four things defend the arithmetic. The operator is
applied node-centric - each thread owns one node and gathers its up to eight
adjacent elements in a fixed order, so there are no atomics and no scatter. The
eight per-element contributions, which is where the 1e9 stiffness contrast
lands, are accumulated with Neumaier compensation, and every reduction is a
compensated two stage tree over a fixed workgroup count. The gather is
*centred*: displacements are gathered relative to the node's own, which is exact
because the three rigid body translations are in the null space of the element
matrix, and which removes the near-total cancellation of the raw 24-term row
dot. That one decides whether the backend works at all on a fine mesh - without
it the attainable residual degrades with refinement, and a 3 million element
problem stalls at 5e-1 instead of reaching 1e-8. And the *solution accumulator*
carries a compensation limb of its own, which decides whether the backend works
at all on a **compliant** design; both limbs come home and are folded in f64.

That last one is the fix for a class of failure the low mass fraction workflow
lives in. What the host folds is only as good as the residual it leaves, and one
f32 rounding of the solution already leaves `eps * | |K| |x| |` of it; a plain
f32 running sum over the thousands of steps a slender bar at `mass_fraction =
0.10` needs leaves that many times over. Measured on a 192 000 cell cantilever
at 1 mm voxels, the device used to report a converged inner residual of 6.7e-6
while the correction it handed back left a relative residual of **1.388** -
above the 1.0 of the zero start it was correcting, so the refinement loop was
being actively pushed backwards. With the limb the same solve runs to 1e-8 on
the device and the whole 150 iteration run takes 3 min 4 s.

A compensated sum is algebraically zero and stays non-zero only because floating
point addition does not associate, so a shader compiler that reassociates can
prove the carry zero and delete it - and the Vulkan compiler on an RTX 3080
does, silently, to every carry in the shader. The solution accumulator therefore
reads its sum back through a uniform that is exactly 1.0 (`GPU_CARRY_GUARD`),
which breaks the identity the compiler needs without changing a bit. The other
two are left as the driver takes them: the centred gather and the fixed
reduction shape already carry those sums, and forcing their carries buys no
accuracy for roughly 2 to 4 % of the wall time of every device iteration. It
changes no reproducibility property either - on one device and driver the kernel
returns the same bits for the same input whichever way it is compiled. What does
vary run to run is the *host* arithmetic around the device solve, the parallel
`norm` the refinement loop measures its residual with; see
[Reproducibility](#reproducibility).

### What single precision cannot do

The correction crosses the bus as two f32 limbs, so the best a pass can leave is
bounded by how well those limbs describe `K^-1 r`. Past a point no arrangement
of f32 helps. The measured case: a 58 320 cell cantilever whose design an
aggressive first `mma` step has driven to the density floor, so a third of the
cells sit at exactly `Emin` and whole regions are held together by nothing but
the stiffness floor. Its condition number is nine decades above the structure's
own, the CPU needs 6 075 double precision iterations, `p^T K p` swings over
twelve decades along the way, and the *exact* solution rounded to f32 leaves a
residual 200 times the right hand side. Symmetric diagonal scaling, a device
side modulus floor, an extended precision matrix-vector product, exact residual
replacement and an outer flexible CG preconditioned by the device solve were all
measured on it; none of them solves it.

So the solver says so and finishes that solve on the CPU:

```text
solver         this solve was finished on the CPU (GPU conjugate gradient breakdown
               after 1375 device iterations ...). The answer meets the same tolerance
               the GPU one would have, at CPU speed; set [solver] backend = "cpu" to
               stop the device trying.
```

This per-solve finish does not conflict with "a backend you name is an
instruction": the device was opened and is doing the run - one solve inside it
was completed at higher precision, which no backend choice can forbid.
It is never silent - the notice names the iteration and the reason, and
`LinearSolver::cpu_fallbacks` counts them - and it is never a *worse* answer: a
refinement pass is a trial, undone again if the f64 residual says the correction
was not one, so the CPU takes over from the best iterate the device reached and
returns a solution at the caller's tolerance. It is also rare and transient: on
the 1.5 mm `mma` cantilever above, iterations 2 and 3 fall back and every one
after them runs on the device.

Measured against the CPU reference (the parity tests in `fea::backend`,
`engine::simp` and `stress`):

| check | gate | measured |
| ----- | ---- | -------- |
| displacement, relative L2, uniform solid | < 1e-4 | 3.0e-11 |
| displacement, relative L2, graded field (3e2 contrast) | < 1e-4 | 5.0e-11 |
| displacement, relative L2, late-SIMP field (1e9 contrast) | < 1e-4 | 1.5e-14 |
| displacement, relative L2, skeletal bar at `mass_fraction = 0.10` | < 1e-4 | 2.1e-10 |
| displacement, relative L2, design at the density floor (CPU fallback) | < 1e-4 | 1.3e-6 |
| final compliance of a capped cantilever run | < 1e-3 | 8.1e-12 |
| peak von Mises of a partly resolved bracket | < 1e-3 | 8.8e-9 |
| `examples/cantilever.toml`, 150 iterations | < 1e-3 | 7.850071e0 on both |

### Reproducibility

The gather order and the reduction shape are fixed, so a device solve returns the
same bits for the same input on the same device and driver. That is *not* the
same as a reproducible run: the host arithmetic around it is parallel, the
refinement loop takes its decisions on thresholds, and a last-bit difference is
enough to change how many passes a solve spends. Two GPU runs of the same
configuration agree on every number a report prints and can still write different
STL bytes. **And nothing guarantees anything across machines, drivers, vendors or
graphics APIs** - a different fused-multiply-add schedule is enough to move the
last bits, and the same machine offers several APIs onto the same card. The
compute solver builds its wgpu instance from the environment, so `WGPU_BACKEND`
really does pin which one it takes:

```sh
WGPU_BACKEND=vulkan growforge run part.toml    # rather than whichever wgpu ranks first
```

The run prints the API and driver it chose on its `linear solves on ...` line, so
a result can be quoted with the thing that makes it reproducible attached. If you
need a result that reproduces on another machine, use the CPU backend.

**The CPU backend is bit-for-bit for a given build *and thread count*** - the
qualifier the rustdoc on `SolverBackend::Cpu` has always carried, and it is worth
spelling out here because it is the same mechanism as the GPU's host half, not a
different one. The reductions are parallel: the conjugate gradient's dot products
and the compliance sum are `par_iter().sum()`, and rayon splits a sum by the size
of the pool, so a 24-thread run and a single-threaded one add the same numbers in
a different order. Only the last bits move. Measured on a small cantilever, one
solve's compliance differs by about 1e-9 relative between the two - decades below
the loosest tolerance this solver accepts, and orders below anything a voxelized
model resolves. That is why the recorded compliance trajectories in this crate
are pinned to a relative tolerance rather than to bits, and why the determinism
tests compare two runs of one process rather than a stored hash. Pin
`RAYON_NUM_THREADS` beside the build when a number has to come back identical to
its last digit; nothing about the *part* depends on it.

The GPU rows above carry no thread-count clause, and that is a decision rather
than an omission: the clause sharpens a promise of exactness, and the GPU tier
makes none - two runs can differ with everything pinned, because the device
schedules itself. Attaching "and thread count" there would imply that pinning
the pool buys back a reproducibility the device never offered.

### Benchmark

`growforge bench <config.toml>` times the linear solve of a real assembled
problem - the operator the first SIMP iteration of that configuration would
build, and the first load case as it is actually assembled - on every backend
this build can reach. Every backend runs three cold solves to the same relative
residual of 1e-6, so the ratio is a like for like speedup:

```sh
growforge bench examples/cantilever.toml
```

Measured on the machine this was developed on (Ryzen-class 24-thread desktop,
NVIDIA RTX 3080 on Vulkan, driver 591.86, `--release`), on
`examples/cantilever.toml` and two finer variants of the same 120 x 40 x 40 mm
block:

| voxel | cells | degrees of freedom | cpu best | cpu iters | gpu best | gpu iters | speedup |
| ----- | ----- | ------------------ | -------- | --------- | -------- | --------- | ------- |
| 2.0 mm | 24 000 | 80 703 | 691 ms | 364 | 61 ms | 750 | **11.3x** |
| 0.9 mm | 271 350 | 856 980 | 5 092 ms | 820 | 672 ms | 1 725 | **7.6x** |
| 0.4 mm | 3 000 000 | 9 211 503 | 118 622 ms | 1 857 | 16 874 ms | 4 725 | **7.0x** |

The GPU spends two to three times as many conjugate gradient iterations as the
CPU, because every refinement pass restarts the Krylov space, and still wins by
seven to eleven times: an iteration on the device is roughly twenty times
cheaper. The advantage narrows on the largest problem, where the extra
iterations and the per-pass round trip of the correction across the bus both
cost more.

The second limb on the solution accumulator is what these numbers cost: against
the same measurements without it, the 24 000 cell solve is 2 % slower on the same
750 iterations, the 271 350 cell one 14 % slower on 10 % more of them, and the 3
million cell one 1 % *faster* on 10 % fewer. That is the price of the correction
being worth what it says it is worth, and it is what the skeletal regime below
is bought with.

End to end, `examples/cantilever.toml` at its full 150 iterations takes **11.1 s**
of optimization on the GPU against about **170 s** on the CPU, and lands on the
same final compliance of `7.850071e0` with the same volume fraction and the same
stress table. That is better than the cold-solve table suggests, because from the
second optimization iteration on the warm start leaves a residual small enough
for a single refinement pass to finish.

These are one machine's numbers on one driver. Run `growforge bench` on yours.

## Surface quality

Two independent knobs decide how the exported surface looks, and a third decides
how it is drawn. They are not alternatives to each other:

| control                          | what it changes                                    |
| -------------------------------- | -------------------------------------------------- |
| `[output] smoothing_iterations`   | how much Taubin smoothing rounds the triangles it was given |
| `[output] supersample`            | how many triangles there are to smooth              |
| viewer's `flat shading` switch    | nothing about the mesh; only how the window shades it |

Smoothing moves vertices; it never adds any. On a marching cubes surface the
vertex spacing is the voxel size, so no amount of smoothing can describe a
feature finer than one voxel, and past about ten passes it only rounds corners
the design meant to keep. Supersampling is the other half: it re-runs marching
cubes on a lattice `N` times finer in every axis, interpolated trilinearly from
the same node densities, so the surface follows the same field with `N` times
the resolution.

```toml
[output]
supersample = 2            # optional, default 1; 1 .. 4
```

Before and after, on the shipped examples (same density field, exported twice):

| example                   | `supersample` | triangles | STL size | export time |
| ------------------------- | ------------- | --------- | -------- | ----------- |
| `cantilever.toml`         | 1             | 15 352    | 0.73 MiB | 0.01 s      |
| `cantilever.toml`         | 2             | 60 744    | 2.90 MiB | 0.02 s      |
| `growth_canopy.toml`      | 1             | 22 668    | 1.08 MiB | 0.01 s      |
| `growth_canopy.toml`      | 2             | 89 520    | 4.27 MiB | 0.04 s      |

At `supersample = 1` the export is exactly what it always was, byte for byte;
that is a test, not a claim. (A surface that came out in more than one piece is
the one thing that moves it, and only because the floating pieces are culled from
the file - see [Islands](#islands).) What the finer lattice buys is the staircase: a
strut that came out of a 4 mm grid as a chain of visible voxel facets comes out
at `2` as a rounded strut of the same volume, because the trilinear interpolant
between the nodes is smooth and the coarse lattice was simply not sampling it
often enough. The volume moves by a fraction of a percent (the canopy above
gains 0.6 %, the cantilever loses 0.08 %) and the enclosed shape is the same
solid, so mass estimates and stress readings are unaffected.

What it costs:

* **Triangles and file size grow with about `N^2`** - the surface area is fixed
  and the edge length is `h/N` - so `2` is roughly four times the file and `4`
  roughly sixteen.
* **Memory grows with `N^3`**, because the lattice does. The export holds about
  20 bytes per lattice sample (the value and the marching cubes edge table), and
  `growforge check` warns before a run when the projected lattice passes
  `SUPERSAMPLE_NODE_BUDGET_WARN` (32 million samples, about 640 MiB). It warns
  rather than refuses: the machine that has the memory is entitled to ask.
* **Time**, both in the extraction and in every downstream tool. Slicing a
  three million triangle STL is not free.

Supersampling is a finishing control, not a resolution control. It resolves the
*field the optimizer produced* more finely; it cannot invent structure the voxel
grid never had. A part that comes out blocky because the grid is too coarse
wants a smaller `voxel_size_mm`, which costs the solve; a part that is right but
facetted wants `supersample`, which costs only the export. Previews inside the
viewer are never supersampled - they exist for a fifth of a second each - and the
window switches to the real, refined surface when the run finishes.

## Viewer

`growforge view <config.toml>` opens a native window on the problem as the
optimizer will see it and optimizes nothing; `growforge edit <config.toml>` opens
the same window on the same setup and lets you change it, see
[Editor](#editor). `growforge run <config.toml>
--view` opens the same window over a real run: the density isosurface is
re-extracted as the optimization iterates, and when the run finishes the window
switches to the smoothed, validated surface that was written to the STL.

Overlays, each with its own checkbox in the side panel:

| overlay              | what it shows                                                 |
| -------------------- | ------------------------------------------------------------- |
| `density surface`    | the evolving isosurface, then the exported mesh (opaque)      |
| `supports`           | a cube on every node a `[[supports]]` region constrains       |
| `loads`              | an arrow per force, a circular arc arrow per torque           |
| `domain (voxelized)` | the design domain as voxelized, chamfers and all (translucent)|
| `keepout`            | the `[[keepout]]` shapes (translucent)                        |
| `keepin`             | the `[[keepin]]` shapes (translucent)                         |

Below them sits **`colour by von Mises stress`**, which is enabled once a run has
finished and produced a stress report. It swaps the plain shading of the exported
mesh for the stress ramp: blue is unstressed, red is the material's yield
strength, and the load case with the highest peak is the one shown. Element
stresses reach the mesh through the same eight-cell averaging the densities use,
sampled trilinearly at each vertex, so a supersampled surface reads the same
field at more places rather than a different one. Plain shading stays the
default, and the setup view never has the option.

Under it, **`flat shading`**. The density surface - preview and final, plain and
stress coloured - is drawn smooth by default: each vertex carries the area
weighted average of the normals of the triangles meeting there, computed on the
mesher thread, so a marching cubes isosurface reads as the curved solid it
approximates instead of as its facets. Ticking the box hands the renderer the
per-triangle normals instead and the old faceted look comes back, which is the
honest view of what the mesh really is. Neither setting touches the geometry:
the same vertices are drawn either way, and the STL is unaffected - a binary STL
carries one facet normal per triangle by definition, and always has. The switch
applies to the density surface alone; the overlays have a shading rule of their
own and keep it whatever it says.

That rule is a **crease angle**. The overlay shapes are tessellated analytically,
so the facets across a cylinder's barrel or a sphere are a few degrees apart
while the rim where that barrel meets its cap is a right angle: normals meeting
at a point are averaged only with the ones within 40 degrees of their own facet.
A barrel and a sphere therefore read as round, every real edge stays an edge, and
a box comes out exactly as flat as it was - one rule, and no shape has to declare
which kind it is. The tessellation itself is 48 segments round a cylinder and
48 x 24 over a sphere, which is a few hundred to a couple of thousand triangles
per shape and costs nothing at these sizes. **The one exception is the voxelized
domain**, which stays flat on purpose: its facets are the answer, and rounding
them would draw a domain the solver does not have. None of this reaches an
exported mesh - the STL is extracted from the density field, not tessellated.

The domain overlay is the voxelized surface rather than the ideal CSG solid, so
what you check is what the solver actually got. Support markers are read back
out of the assembled constraint mask, so they show the nodes really pinned;
nodes the model builder pins for conditioning because they touch no material are
not drawn. A force arrow has its tip on the centroid of its region and points
along the force vector; a torque arc wraps `axis_dir` through `axis_point`
following the right hand rule, reversed for a negative magnitude. Both are sized
relative to the scene, not in absolute millimetres. A gravity load gets no
indicator: it acts on the whole structure, and the density surface is already the
picture of where it acts.

Controls:

| input                    | action    |
| ------------------------ | --------- |
| left drag                | orbit     |
| right drag, middle drag  | pan       |
| scroll wheel             | zoom      |
| `F`                      | fit view  |

The view is fitted when the window opens and again on `F`.

**Closing the window during a run detaches the viewer, it does not cancel the
run.** The optimization carries on headless at full speed, keeps printing its
per-iteration lines, and still writes the STL. Ctrl+C in the console is the way
to abort. The viewer can never change what a run produces: the same code writes
the STL with and without `--view`.

The window is cheap: it renders on its own thread budget while the solver runs,
and a run measured with and without it stayed inside its own run-to-run spread
(8.4 to 8.8 s for the same 40 iterations). A snapshot the mesher has not picked
up is overwritten rather than queued, so a slow preview costs skipped frames and
never memory or solver time.

The viewer lives behind the `viewer` cargo feature, which is on by default and
gates every graphics dependency:

```sh
cargo build --release --no-default-features                  # pure CPU solver
cargo build --release --no-default-features --features gpu   # headless GPU solver
```

In a build without `viewer`, `view` and `--view` still parse but fail with a
message telling you to rebuild. If the machine has no usable GPU adapter, both
fail immediately, before any optimization work, with a message telling you to run
without `--view`.

**A frame the GPU rejects is retried, not fatal.** A driver reset - a Windows
TDR, which a long compute job on the same physical GPU can trip - shows up at the
window as the surface refusing to hand over the next frame. The renderer
reconfigures the surface and skips that frame, exactly as it does for a surface
that has gone stale, and prints one line the first time it happens:

```
viewer surface the GPU rejected a frame; reconfiguring and retrying (the device may be resetting)
```

Nothing more is printed while it lasts. A frame that is actually drawn clears the
streak; frames skipped because the compositor was busy or the window is not
visible are not rejections and count towards nothing. Only **30 consecutive
rejected frames** - about a fifth of a second at 144 Hz - is a device that is not
coming back, and then the window ends with `the GPU rejected 30 consecutive
frames of the viewer; the device was likely reset and did not come back` and the
process exits failure. A `growforge edit` session with unsaved changes writes
them out first: see [Saving](#saving).

The `gpu` feature is independent of `viewer`: a headless build can still solve on
the GPU, and a viewer build can still be told `backend = "cpu"`. They share the
wgpu dependency but not a device - the compute solver opens its own, so
`growforge run --view` with `backend = "gpu"` draws and solves at the same time.

### Testing hook

Setting `GROWFORGE_VIEW_AUTOCLOSE_S` to a number of seconds makes the window
close itself that long after its first frame, which is how the viewer is smoke
tested without a human at the keyboard:

```sh
GROWFORGE_VIEW_AUTOCLOSE_S=5 growforge view examples/cantilever.toml
```

An unset, empty or unparsable value means "never", so it cannot cut an
interactive session short. On exit the viewer prints the frame count and the
average frame rate. It closes the editor's window too, unsaved changes and all:
a smoke test has nobody to answer the modal, and it is a testing hook rather
than a way to close a window you are using.

## Editor

```sh
growforge edit examples/cantilever.toml
growforge edit parts/new_bracket.toml   # a path that is not there yet is scaffolded
```

`edit` opens the viewer's window in editor mode: the same setup overlays, the
same camera, and the problem definition itself as the document. You pick objects
in the viewport or in the tree, drag them, type exact numbers into every field,
watch the setup re-voxelize as you go, re-run the engine on the spot, and save
the file back when you are happy.

**The file on disk is the source of truth.** `edit` reads it once and writes it
only when you ask. Nothing else is ever written - the background re-runs export
nothing at all, and an STL appears only where you ask for one: `run full`, or
`generate stl` on the design already on screen.

Given a path that does not exist, `edit` writes a **starter configuration**
there and opens it: a block standing on the floor with a load pressing on a pad
at the top, on the `simp` engine, in PLA. It validates, runs and exports as it
stands, and every number in it is meant to be dragged. An existing file is never
scaffolded over. A new file therefore defaults to topology optimization on the
compute backend - and, because that is seconds of solving rather than
milliseconds of growth, auto-regrow starts *off* for it: the window opens on the
setup, and the first run is one you ask for.

**You are not stuck with the file you launched on.** The toolbar's `open` and
`new` buttons - `Ctrl+O` and `Ctrl+N` - raise the platform's own file dialog and
switch the session to whatever comes back, without closing the window: see
[Opening and starting files](#opening-and-starting-files).

### Panels

| panel               | what is in it                                                              |
| ------------------- | -------------------------------------------------------------------------- |
| toolbar             | save, `open`, `new`, undo, redo, the auto-regrow switch, `run full`, `stop`, `generate stl`, and the last message |
| validation          | why the configuration is not runnable, or the warnings `check` would print, then what the run behind the window is doing and the engine's last word on why it ended |
| objects             | every `[[domain]]`, `[[keepout]]`, `[[keepin]]`, `[[supports]]` and load case, with add and delete; the `tube` button places one by two clicks instead of adding it |
| properties          | exact numeric fields for whatever is selected, and its own extras (a domain entry's `op`, a support's fixed axes, a load's vector or torque) |
| engine / resolution / material / optimization / growth / output | every scalar the configuration holds, each with its default shown when the key is absent; each section carries a `reset` button that puts its own keys back to those defaults |
| problem             | grid size, cell, node and degree of freedom counts, per-region node counts and the memory estimate, refreshed on every edit |
| show                | the same layer switches the viewer has, plus the editor's own selection, hover, gizmo, dimension, floor grid and placement preview overlays; under them, what the [trim](#trimming-unloaded-material) pass removed from the design on screen, or the warning that it was refused, what the [flush fill](#flush-fill) pass put back out to the surfaces the walls rest on, what the [reinforcement](#reinforcement-minimum-printable-thickness) pass spent on its thin arms, or the warning that a member could not be thickened, and what the [boundary clamp](#exact-boundaries) moved onto the shapes |
| stress              | the [safety factor](#stress-report) of the part that was just written, first, with the peak von Mises stress of every load case under it; present after a full run and after `generate stl`, absent while one is running and absent when the stress solve produced no report |

**Every labelled block above folds away**, and stays however you left it for the
session. The window opens on the ones a session works in - `objects`,
`properties`, `show` and `stress` - with `problem` and `controls` folded, and
with **the five object lists themselves closed**: click `keepout (3)` to see
what is in it. `material` and `output` open closed as they always have. The
toolbar, the snapping controls and the validation block are not foldable: a
warning you cannot see is a warning that was never printed.

An optional key has a checkbox in front of it: unticked means the key is not in
the file at all and growforge's own default applies, and the default is shown
next to it. Ticking it writes the key with that default as its starting value.

**Every control in the panel explains itself on hover** - what the key does, in
what units, and what its default is - so the reference below is for reading and
the panel is for working; the `show` switches carry the same text in the viewer's
window.

The optimization section carries the three optional `[optimization]` sub-tables
that way too: `overhang constraint`, `guide wireframe` and `local volume` are
checkboxes, and ticking one adds the table with every key inside it still on its
own default, so the rows appear with their defaults showing and nothing is
written into the file that was not asked for.

The growth section carries `[growth.symmetry]` the same way: a checkbox adds the
table with a single mirror plane, a dropdown switches the kind, and each kind
offers only its own keys - the planes for a mirror (the second one cannot be set
to the first), the order and axis for a rotation. Auto-regrow picks it up like
any other edit, so the part becomes symmetric while you watch.

**Each of those six sections carries a `reset` button**, on a row inside its own
header. It puts that section's keys back to what a configuration that never
mentioned them would run at, and touches nothing else; **one click is one undo
step**, so one undo puts the whole section back exactly as it was. The button is
greyed out while its section is already at its defaults, and says so on hover.
The objects have none: a shape is geometry you drew, and there is no default
position for a load to return to.

Resetting a section switches its optional sub-tables off with it - the overhang
constraint, the guide wireframe, the local volume cap, `[growth.symmetry]` -
because a table that is not there is a feature that is not running, and
**resetting the engine takes a `[growth]` table with it**, exactly as switching
the engine by hand does: what a reset lands on still builds. Four keys are
deliberately not put back to a default, and the tooltips say which:

| key                             | what a reset does with it                    |
| ------------------------------- | -------------------------------------------- |
| `[optimization] mass_fraction`  | keeps its value - how heavy the part may be is design intent, and there is no default to invent for it |
| `[optimization] min_feature_mm` | returns to three voxels at the voxel size this configuration is solved on - the smallest feature the density filter can resolve, rather than a fixed length, so a reset never lands on a value the run then warns about |
| `[output] stl_path`             | keeps its value - where a run writes is a decision about the project, not about the part, and it is the one required key of the schema besides |
| `[output] stress_json`          | keeps its value, for that same reason: a reset puts properties back, it does not move deliverables |

A key a reset removes takes its comment with it, the way [saving](#saving)
always removes a key you switched off; the keys it does not touch keep theirs,
and one undo brings the removed ones back.

### Controls

| input                    | action                                       |
| ------------------------ | -------------------------------------------- |
| move the pointer         | outline whatever a click would select        |
| left click               | select the object under the pointer; empty space deselects |
| `tube` in an add row     | start placing a tube in that list: two clicks are its ends |
| left click while placing | put down that point - the mode owns the click, so nothing is selected by it |
| `Esc`                    | leave the placement, discarding a point already put down |
| left click on a handle   | grab it - the whole arrow is grabbable, not just its tip; a cube standing in an arrow's shaft wins the press |
| left drag                | orbit (unchanged - a click is a press that went nowhere) |
| drag a handle            | move or resize the selected object, on the snap increment; the drag follows the pointer off the 3D view, so an object can be taken out of the framed domain |
| drag an arc              | rotate the object about that axis, on the angle increment |
| `Alt` + drag             | the same drag with no snapping at all        |
| click the number box     | type an exact value for what the drag was changing |
| `Enter` / `Esc`          | apply that value / leave the drag's own      |
| right drag, middle drag  | pan                                          |
| scroll wheel             | zoom                                         |
| `F`                      | fit view                                     |
| `Delete`                 | delete the selected object                   |
| `Ctrl+Z` / `Ctrl+Y`      | undo / redo (`Ctrl+Shift+Z` redoes too)      |
| `Ctrl+S`                 | save                                         |
| `Ctrl+O`                 | open another configuration in this window    |
| `Ctrl+N`                 | scaffold a new one and edit that             |

**The object under the pointer gets a thin outline** before you click it, which
is how overlapping objects are told apart: a load region sitting inside a keepin
pad is two shapes in the same place, and which of them a click means is a rule
rather than a guess. The outline is that rule's answer, drawn with the very ray
and the very ranking a click would use, so it previews the click exactly. It is
cyan and thinner where the selection shell is yellow and thicker, so "what is
selected" and "what would be selected" never read as the same thing; hovering
what is already selected draws nothing extra. Nothing is outlined while a drag
or an orbit is in progress, while the pointer is over the side panel or off the
window, or over a gizmo handle - a click there grabs the handle, and the handle
says so by brightening instead. The domain is not outlined either, for the reason
it is not clickable: everything else lives inside it.

The selected object gets a translucent shell and a gizmo: three axis arrows and
a centre handle that translate it, three curved arrows that turn it, and the
resize handles its own shape has - eight corners and six faces for a box, both
cap centres and a radius for a cylinder, a radius for a sphere, one radius
handle per semi-axis for an ellipsoid, both end centres, a radius and a **bend**
handle for a tube, both end centres and **a radius at each of them** for a cone,
and a handle on each of the three **corners** plus one for the **thickness** of a
triangle. An axis handle slides along its own line; the
centre, corner, endpoint, vertex and bend handles move in the plane facing the
camera; an arc handle turns the object about the axis it rings. On a **rotated
box or ellipsoid** the arrows, faces, corners and radius handles follow the
shape's own frame, so the handles sit on the faces you can see and a drag along
"x" runs along the shape's own x. The numeric fields follow the drag live, and
the whole drag is **one** undo step, recorded when you let go.

**A resize drag snaps to the dimension you set off in.** A box corner, the end
handles of a cylinder, a tube or a cone, and a triangle's corners read which way
the gesture started - once, as soon as it has travelled far enough to mean
something - and hold that answer until you let go. A box corner grows the one
edge you pulled along and leaves the other two exactly as they were; an end
handle pulled *along* the shape sets its **length**, sliding the end on the line
through both of them so the shape cannot veer off while you pull it. Pulled
*across* the shape instead, the same handle places its end anywhere in the plane,
which is how a cylinder, a tube or a cone is pointed somewhere else - and a
triangle's corner has the same pair of gestures, described with the triangle
below. Until the drag has covered that first fraction of a millimetre it changes
nothing at all, so a press cannot nudge what it grabbed. The move and bend
handles are placements rather than resizes and stay free in every direction.

**A cone is dragged to its point.** Its second radius handle goes all the way to
zero - which is the apex of a true cone, the shape that drag is aiming at - where
every other radius in the editor stops at a tenth of a millimetre. At zero that
handle stands exactly on the second end handle, and two markers in one place are
one press with one winner: it goes to the radius, which is the handle that can
bring the cone back off its point, and moving it a hair separates the two again.
The wide end keeps the usual floor, because a cone with no wide end is nothing at
all.

**A triangle is dragged by its corners.** Between them its three vertex handles
set where the prism is and which way it faces - which is why it has no rotation
arcs, exactly as a sphere has none for the opposite reason - and each of them
latches like every other resize. Drag a corner **in the triangle's own plane**
and it sizes it: along the edge that corner faces, which slides it sideways at
the height it stood at, or along that height itself, at a base that does not
move. Drag it **out of the plane** and it is the pose gesture, free across the
camera's plane and snapping per axis exactly as it always did; that drag has to
be the bigger one to win, so a gesture that splits the difference sizes rather
than aims. A corner is held a tenth of a millimetre off the line through the
other two whichever way it was dragged there: a triangle on one line has no area,
and no thickness would give it a solid.
The thickness handle sits on the middle of the face and slides along its normal;
the extrusion is symmetric, so the face stays under the pointer while the prism
grows twice as fast either side of its own plane, and the callout shows the
thickness the file holds rather than the half the handle moved.

**Drag the middle of a tube to curve it.** A tube's bend handle sits in the
middle of it - on the bend it carries, or halfway between its two ends while it
is straight - and dragging it is what gives the tube its curve, through the very
point you drag it to. On a straight tube that is exactly where every other
shape's centre handle is, so the bend is drawn a little larger and in its own
colour: two markers in one place is one press with one winner, and on a tube the
winner is the bend. Curve it and the centre is the translation handle's again.
Drag the bend back onto the line between the two ends and the tube **straightens**
- the key goes with it - and the properties panel has a **straighten** button for
doing that without aiming.

**Click two points to make a line.** The `tube` button in an add row does not
drop one at the centre of the domain the way every other kind does: it arms a
**placement** for that row, and the next two clicks in the viewport are the
tube's two ends. What is left behind is a straight tube in that list, selected,
with its bend handle waiting in the middle of it - which is the whole gesture the
tube exists for: click, click, drag the middle.

A click lands on the nearest surface its ray meets - any object's, and the design
space's own as well, which is the surface most of a model is drawn against - and
falls through to the ruled floor plane when it meets none. Either way it lands on
the snap increment, and `Alt` frees it exactly as it frees a drag. A click that
meets neither surface nor floor - at the sky, or past the ruled floor - does
nothing at all and leaves the placement where it was, rather than putting a point
at some arbitrary distance along a ray aimed at nothing; so does a second click
that lands on the first one, which would be a tube of no length.

While the mode is armed the panel says which list is being placed into and which
click is next, a marker shows where the next click would land, and once the first
point is down the tube those two points would make is drawn between them at the
radius the placed one will have. **The mode owns the clicks**: nothing is
selected by them, no handle can be grabbed, and nothing of the selection is drawn
- a handle that cannot be grabbed would be lying about what a press does. What
was selected is kept, though: `Esc` at either stage leaves the mode and hands its
overlays straight back, and so does clicking that same `tube` button again.
Clicking another row's `tube` starts that row's placement instead. A placed tube
goes through the containment rule like any other commit, so one placed on the lid
of the domain comes back inside it and the panel says so.

Deleting the selected object, undoing or redoing while a placement is armed
leaves the mode as well - what it promised to hand back may no longer be there,
and re-entering it is one click. Typing numbers into the properties panel does
not: a clicked point is a position in the world rather than a reference to an
object, so nothing typed can invalidate it.

**The gizmo's grab volumes overlap, so a press is ranked rather than measured.**
An arrow's whole shaft is grabbable, and every cube stands inside one: the centre
handle is where all three shafts begin, and on anything near cubic the resize
cubes are on the shaft along their own axis - a box's `+x` face handle sits at
half its width while the arrow runs to three quarters of its half diagonal, and a
radius shorter than the gizmo is buried the same way. Depth alone would hand
every one of those presses to the arrow, so:

* a cube **buried inside** an arrow's volume takes a press that hits both: it is
  the small, precise target and it can never be the nearer hit;
* a cube the ray merely passes *behind* takes nothing - a press aimed at an
  arrow's tip keeps the arrow, however many corner cubes lie further along that
  ray, and half way along a shaft is still the arrow;
* the rotation arcs ring the gizmo from outside it, so one of them is often
  between the pointer and everything else, and a press that hits an arc and
  anything else always gets the inner one.

Which is the rule picking uses on objects: what is inside is taken before what
encloses it.

### Snapping, and the number box

Drags **snap**: what the handle is changing lands on a multiple of the
increment - the position along the drag axis, the dimension being resized, the
radius, the angle. Only that value: an x drag never moves y onto anything, and
a resize never moves the face it is not dragging. Snapping is *absolute* rather
than relative, so two objects dragged onto the same grid line really meet
whatever they started off by.

| control            | what it sets                                                    |
| ------------------ | ---------------------------------------------------------------- |
| `snap mm`          | the length increment: a dropdown of the usual ones, `off`, or any number typed in beside it (default 1 mm) |
| held `Alt`         | no snapping at all, for as long as it is held                    |
| (fixed)            | rotations snap to 22.5 degrees, which holds 45 and 90            |

**A floor grid is ruled at that increment**, on the plane `z = the bottom of the
domain`, over the domain's footprint plus a tenth of it as margin, with a heavier
line every tenth one. It is what makes the increment visible: change `snap mm`
and the grid is re-ruled to match on the next frame. The major lines are counted
from the world origin rather than from the edge of the footprint, so they sit on
round coordinates and stay put when the domain is resized. The lines are drawn
dim - half the brightness they had before 0.26.1 - so the ruling is read against
the model rather than competing with it. It is a layer like any other and can be
switched off entirely under **show**; it is drawn in edit mode alone, so `view`
and `run --view` are unchanged.

Fine increments over large domains are capped. A 400 mm domain at a 0.1 mm
increment asks for eight thousand lines, which is neither drawable nor legible,
so past 400 lines (`VIEW_EDIT_GRID_MAX_LINES`) the spacing is multiplied by the
smallest **whole** number that brings the count back inside the cap - whole, so
every line still lands on a multiple of the increment drags land on - and the
panel says so rather than quietly ruling something else:

```text
floor grid ruled at 3 mm, not 0.1 mm: a finer grid would need more than 400 lines
```

**Support and load regions land flush on faces.** These two are not placed
against a coordinate but against a *surface* - the top of a load pad, the wall
of the design space - and half a millimetre off that surface is a region that
selects a different set of nodes. So while one of them is dragged, the faces of
the `[[keepin]]` entries and the outside of the domain are candidates: bring a
face of the region within 2.5 mm of one and it lands exactly on it, and the
callout says which (`flush on keepin 1`). The surface wins over the millimetre
grid where both are in range - within that distance the face is what you are
aiming at - and `Alt` switches off both. Keepouts, keepins and domain entries
are pieces of the model in their own right, are placed by their own numbers, and
get the grid alone.

A rotated box lands by the bounding box it really occupies, matched against the
axis aligned faces above; matching a turned face against another turned face is
a different feature and is not this one. A drag running diagonally across all
three axes at once - which is what an arrow of a turned box does - lands on the
grid instead, because no one axis is the one it is being placed against.

While a handle is held, a **number box** floats beside what it is measuring:
how far the object has moved along the drag axis (signed), how wide it now is,
its radius, its thickness, the distance between a cylinder's, a tube's or a
cone's two ends, how far a tube's bend has been pulled off the line between them,
how far a triangle's corner stands off the edge it faces, or the angle it has
been turned to. A thin
dimension line with an arrow head at each end is drawn across the same thing, and
a second line measures the gap to the floor of the domain while an object is
being moved vertically. Both stay for a few seconds after you let go.

**Click the number** and it becomes a text field: type an exact value, press
Enter, and the object goes exactly there - applied to the shape the drag started
on, as one undo step rather than a second one stacked on the drag. Escape, or a
click anywhere else, leaves the drag's own value alone. It is the same edit the
properties panel makes, spelled where you are looking.

### Keeping objects inside the domain

**keep inside domain** (on by default) clamps every commit of a keepout, keepin,
support or load region so that its bounding box stays inside the domain's - a
drag stops at the wall, a number typed into the panel or into a callout is
brought back, and the panel says so for a few seconds when it had to, naming the
switch that did it. Only the *position* is ever changed: an object is never
resized to fit, and one too large for the domain along some axis is centred on it
instead. A rotated box or ellipsoid is clamped by the room it really takes up,
which is its rotated bounding box. The domain entries themselves are exempt -
they are what everything else is kept inside of.

Switch it off to place something deliberately outside: a keepin that protrudes
is a legitimate model - keepin regions are forced solid *and* part of the grid
even where they stick out of the domain, which is how a mounting boss or a load
pad that overhangs the design space is described. With it off a drag really does
leave the domain: the view is fitted to the model, so the pointer usually has to
go outside the 3D view - over the side panel, or off the window - to ask for a
position outside it, and a drag in progress owns the pointer wherever it goes.
Only the press and the hover are confined to the 3D view, where there is
something to aim at.

Picking is analytic - the ray is intersected with the very shapes the
configuration is written in - and it prefers what is *inside* something else:
loads and supports first, keepout and keepin next. **The domain is picked from
the tree, never by a click**, so it can never steal a click from the objects
inside it; once selected there it drags and resizes in the viewport like
anything else. Every pointer position, and every ray cast through one, is in
physical pixels against the same viewport the frame was drawn with, so the
gizmos land where they look whatever the display's scale factor is.

### Auto-regrow, previews and the full run

**Auto-regrow** re-runs the engine after every committed edit, so the part grows
again as you change it. It starts on for `engine = "growth"` and off for `simp`,
because what a re-run costs is not the same thing on the two engines:

| engine   | what a re-run is                                                                                      |
| -------- | ----------------------------------------------------------------------------------------------------- |
| `growth` | the real engine at the real resolution: it grows a design in milliseconds                             |
| `simp`   | a **fast preview**: at most 20 iterations on a grid coarsened to about 40 000 cells, labelled `preview` |

A preview is not a result. The panel says `preview` next to the surface it is
showing, and the numbers a full run prints - stresses, cavities, mass - are not
computed for one at all. The solver backend is left exactly as the configuration
asks, so a preview of a `backend = "gpu"` problem solves on the GPU too.

Edits are debounced: the re-voxelization and the re-run wait until you have
stopped moving, rather than firing on every value a drag passes through. An edit
that lands while a preview is running cancels that preview and starts one of the
newer configuration - the SIMP loop checks for it between iterations, so it
stops with the design it had reached and never mid-solve.

**Stop** ends whatever is running - the preview an edit started, the full
pipeline, an stl generation - through the same cooperative seam an edit uses. The
question is asked in three places, and all three matter:

* inside the **linear solve**, every 32 conjugate gradient iterations
  (`CG_CANCEL_CHECK_INTERVAL`), and on the GPU once per device readback batch.
  This is what makes a stop prompt: one cold solve on a real grid is thousands of
  iterations and tens of seconds, and a stop that had to wait for it did not feel
  like a stop at all. The check is a relaxed atomic load against an iteration
  that costs at least 6e5 floating point operations, so it is free by ten orders
  of magnitude, and the latency of a stop is bounded by those 32 iterations - on
  a 27 000 element problem, 60 ms measured against the 36 s that solve takes;
* between **engine iterations**, where a stopped run keeps the design the last
  completed iteration left;
* at each **pipeline stage boundary**, which is what guarantees nothing is
  exported.

A cancelled solve is a third outcome alongside converged and failed: it is not an
error, is never reported as one, and no partial answer is ever folded back into
the design. A stopped run **writes no file of its own accord**, not even a partial
one; what is already on screen stays there, and the editor is immediately usable
again. Immediately means what it says: a stopped run can no longer reach its
export, so it owns neither the output file nor the buttons, and `run full` comes
back the moment you stop - whatever stage its thread is still leaving.

What a stop leaves behind is the design itself, and `generate stl` is how you ask
for it as a file. Stopping remains a thing that writes nothing; asking is what
writes.

Nothing on the command line asks for a stop, so `growforge run` and
`growforge run --view` solve exactly the arithmetic they always did; closing a
`run --view` window detaches the run, which is still not a stop.

**A run that fails does not end the session.** A structure that is barely
connected can fail its linear solve outright - the operator loses definiteness,
or single precision cannot resolve it - and that is the run's failure, not the
window's. The panel says so in the solver's own words, which are the actionable
ones:

```text
full run failed and wrote nothing: solving load case "tip": the GPU conjugate
gradient stopped improving at a relative residual of 1.412e1 against a target of
1.000e-8 ... Set [solver] backend = "cpu" for this problem - the session is
unaffected; change the configuration and run again
```

Nothing is written, `run full` is offered again at once, and the window is still
there to fix the problem in. Only a failure of the *session* - a configuration
that cannot be read when the editor opens, a window that will not come up - ends
the process with a non-zero status; an editor session that you close exits zero
however its runs went.

**`run full`** is `growforge run` on the current configuration: the full
iteration budget, the cavity pass, the stress report, and the STL written to
`[output] stl_path`. Progress appears in the window exactly as with
`run --view`, the console prints the same per-iteration lines, and the run ends
on the exported mesh with the stress colouring available.

**`generate stl`** writes the deliverables of **the design already on screen**:
the enclosed cavity pass, the stress report, the mesh and the STL, without running
an engine at all. It is the same code a full run's own export is - the file, the
surface in the viewport and the stress colouring are the ones `run full` would
have produced from that field - and it is the answer to having stopped a long run
and wanted the part it had reached.

Every run keeps its newest design for it: the full-resolution field of the last
iteration it reported, together with **the problem that run was built from**. Both
matter:

* **The field is whatever the window last showed.** A converged run, a run that
  spent its budget, a run you stopped at iteration 527, a coarsened `simp`
  preview: all of them are designs, and any of them can be exported. The frame
  label above the panel says which you are looking at, so a `preview` is never
  mistaken for a converged result - it will be written at the resolution the
  preview ran, because that is the design that is on screen.
* **The problem is the one that design was computed on**, not one rebuilt from the
  configuration as it stands now. Carry on editing after the run - change the
  resolution, move the loads, break the configuration outright - and the button
  still exports the design you are looking at, to the path that run resolved,
  rather than a field reinterpreted against a grid it never belonged to. `generate
  stl` is therefore *not* disabled by an invalid configuration, where `run full`
  is.

It is offered whenever there is a design on screen and nothing is already writing,
and disabled otherwise: a full run and a generation both own `[output] stl_path`,
so each disables the other, and an auto-regrow preview waits rather than taking
the worker from either. There is no state in which the button is there and does
nothing. **Stop** ends a generation like any other run, at the same checkpoints,
which are all before the write; the overwrite warning applies to it exactly as it
does to a full run. Starting a new run of any kind replaces the kept design, so
what is exportable is always what the window is showing and never the run before
it.

**Closing an editor window ends its session.** Whatever is running behind it -
a preview, a full run, an stl generation - is stopped, nothing is written, and the
process exits;
the unsaved-changes modal says so when a run is in flight. This is deliberately
not what `run --view` does, where closing the window detaches a run that was
asked for on the command line and lets it finish and write its STL headless.
There the run outlives the window by design; here the window *is* the session,
and a growth left running invisibly is exactly what makes two editors fight over
one output path.

Stopping is cooperative, so the run ends at its next checkpoint rather than the
instant you click the close button. The window goes at once; the process stays
until that checkpoint with its event loop still turning, exactly as it does for a
detached `run --view`, because a process that owns window state and stops
servicing its message queue is one Windows declares hung and kills.

Two editors can of course be pointed at two configurations that name the same
`[output] stl_path`. Nothing in growforge arbitrates that - the path is your own
instruction - but before a full run overwrites a file that something else wrote,
the panel says so. "Something else" means what it says: a file this session's own
last run wrote - by `run full` or by `generate stl` - is recognized as ours, mtime
and all, so a second write never accuses you of your own work.

`run full` is disabled while the configuration is not runnable, and auto-regrow
starts nothing until it builds. `generate stl` is not: it exports a design that
was already computed, together with the problem it was computed on, so it cannot
be invalidated by an edit. Saving is never disabled either: it is your file, and a
half-finished problem definition is a perfectly good thing to save - the
validation panel is what tells you it is not runnable yet.

### Saving

`Ctrl+S` or the save button writes the configuration back to the file it came
from. The write is a **format preserving round trip**: comments, key order,
blank lines, alignment, the spelling of every number and every value you did not
change come back byte for byte, including the file's own line endings. Only the
values you changed are rewritten, in place, keeping the comment block above the
key and the comment after the value; a key you switched off is removed, and an
object you added is appended to its own section - after the last of its
siblings, not at the end of the file. A deleted object takes its own comments
with it and no others.

The title bar carries an asterisk while there are unsaved changes, and closing
the window with any asks first: **save**, **discard** or **cancel**. Nothing is
ever silently written or silently thrown away. "Unsaved" is measured against
what the file holds rather than latched by the first edit, so undoing your way
back to the saved state clears the marker and closes without a question.

**If the window dies with unsaved changes in it, they are written out beside the
file.** A viewer error nothing can be drawn through - a GPU device that never
came back, above all - puts the configuration as it stands in
`<name>.recovered.toml`, in the directory the file being edited lives in, and
says so on the console:

```
editor rescue the viewer stopped with unsaved changes; they are in parts/bracket.recovered.toml
editor rescue parts/bracket.toml itself is unchanged
```

It is the same format preserving round trip a save is, so the recovered file is
byte for byte what `Ctrl+S` would have written and is a configuration `edit` and
`run` open as they are: read it, and rename it over the original if you want it.
**The file you were editing is never written to by this path**, an earlier
recovered file is replaced rather than added to, and a write that could not be
made says so instead of being lost. The window still ends in failure - a rescue
is not a recovery of the session, only of the document.

### Opening and starting files

The toolbar's **`open`** and **`new`** buttons, and `Ctrl+O` and `Ctrl+N`, move
the session to another file without closing the window. Both raise the
platform's own file dialog - the one every other program on the machine raises -
starting in the directory the current file came from and filtered to `.toml`.
`open` picks a file that is there; `new` is save-as shaped, so you type a name.
The window is frozen while the dialog is up, which is what a modal dialog is;
the run behind it keeps going and simply drops the preview frames nobody is
collecting.

The switch goes through **the same question a close does**. Unsaved edits put up
the same modal - **save**, **discard**, **cancel** - naming the file you are
going to; cancel leaves you exactly where you were, with the edits intact.
Whatever is running behind the window is **stopped and collected before the new
session exists**, so two engines can never be running at one window, and a
stopped run writes nothing as always. What opens is a whole new session: the new
file's setup on screen, the camera framed on it as if the window had just opened
there, the title, the directory relative paths resolve against, an empty undo
history and no design left over from the run you were watching.

**`new` refuses a file that is already there.** It is not a second way to open
one and it is never a way to overwrite one: the status line says so - *"...
already exists - use open instead"* - and nothing about the session changes. It
is the same refusal `growforge edit` makes on the command line, and it is made
twice: once when you pick the name, and once by the scaffold itself, so a file
that appears in between - or a save dialog that offered to replace one - still
cannot be written over.

A file that will not open - a configuration that does not parse - leaves the
session on the document it was on, with the reason on the status line.

## Configuration reference

Every table rejects unknown keys, so a typo fails loudly instead of being
silently ignored.

```toml
[project]
name = "cantilever"        # required, used in reports

engine = "simp"            # optional, default "simp"; "simp" | "growth" |
                           # "solid". May also be written as project.engine (a
                           # bare key after [project] belongs to that table in
                           # TOML), but not in both places

[resolution]               # exactly one of the two keys must be present
voxel_size_mm = 2.0        # cubic voxel edge length
# target_cells = 2000000   # derive the voxel size from the domain bounding box

[material]                 # either a preset, or all custom values, never both
preset = "pla"             # "pla" | "petg" | "abs"
# youngs_modulus_mpa = 2300.0
# poisson_ratio = 0.36
# density_g_cm3 = 1.24       # drives the self-weight loads and the mass estimate
# yield_strength_mpa = 50.0  # optional; the stress report's safety factor needs it

[optimization]
mass_fraction = 0.3        # required except under engine = "solid", which
                           # REJECTS it; open interval (0, 1); fraction of the
                           # DESIGN cells kept, measured on the PRINTED densities.
                           # The growth engine normalizes its strut radii to it
min_feature_mm = 4.0       # required; the density filter radius is half of this.
                           # For the growth engine it is the smallest strut
                           # DIAMETER, and the scale of every [growth] default
penalty = 3.0              # optional, default 3.0; simp only (the growth and
                           # solid engines never read it). The exponent p of
                           # E(x) = Emin + x^p (E0 - Emin). At least 1, and 2 to
                           # 5 is the useful range: higher prices intermediate
                           # density harder and drives the field black and white,
                           # at the cost of more local minima. Every recorded
                           # trajectory and every shipped example is at 3
stiffness_floor = 1e-9     # optional, default 1e-9, 1e-12 .. 1e-3; simp only.
                           # The growth and solid engines never read it.
                           # The Emin of that formula: what an emptied DESIGN
                           # cell still carries, as a fraction of E0. The
                           # default is nine decades of stiffness contrast
                           # across the load path, which is what the solver's
                           # iteration count grows with - raise it when a run
                           # exhausts its budget short of the tolerance. Forced
                           # void cells carry a literal zero either way
max_iterations = 150       # optional, default 1000; simp only - growth has its
                           # own step cap and solid runs no loop. A budget, not a
                           # target: a run stops on convergence, or on the stall
                           # criterion, long before it here. See "When a run
                           # stops" above
convergence_tol = 0.01     # optional, default 0.01; simp only, stop once the
                           # largest design variable change falls below it.
                           # Neither other engine has a design variable
update = "oc"              # optional, default "oc"; "oc" | "mma". simp only:
                           # neither other engine moves a design variable.
                           # "mma" is the method of moving asymptotes, for runs
                           # the optimality criteria step cannot settle. See the
                           # update scheme section above

# [optimization.overhang]  # optional; absent means no printability constraint
# build_direction = "z+"   # required inside the table: x+ | x- | y+ | y- | z+ | z-
                           # the 45 degree angle is fixed by the filter stencil.
                           # simp only: engine = "growth" and engine = "solid"
                           # reject this table

# [optimization.wireframe] # optional; absent means no guide. Present seeds a
# radius_mm = 2.5          # thin wire through every load, support and keepin
# hold_iterations = 40     # region and holds it as a density floor for the first
# seed_density = 1.0       # iterations, then lets go. Every key optional; simp
                           # only: engine = "growth" and engine = "solid" reject
                           # this table. See the guide wireframe section above

# [optimization.local_volume] # optional; absent means the global volume target
# max_fraction = 0.6       # alone. Present caps the material any NEIGHBOURHOOD
# radius_mm = 6.0          # may hold, which is what turns a few thick members
                           # into a network of many thin ones. Every key
                           # optional: max_fraction defaults to 0.6 and must sit
                           # above mass_fraction, radius_mm to three density
                           # filter radii (3 x min_feature_mm / 2) and must sit
                           # above one. NEEDS update = "mma"; engine = "growth"
                           # and engine = "solid" reject this table. See the
                           # local volume section

# [solver]                 # optional; absent means the compute backend, falling
# backend = "gpu"          # back to the cpu when this build or this machine has
                           # none. "gpu" (default) | "cpu". Named explicitly,
                           # "gpu" needs the `gpu` cargo feature and an adapter
                           # and fails without them; "cpu" is what makes a run
                           # reproducible across machines. See the solver
                           # backend section above
# tolerance = 3e-8         # relative residual the optimization's solves stop
                           # at, 1e-10 .. 1e-4, default 1e-8. A hard target: a
                           # solve that reaches the iteration cap short of it
                           # fails the run. Looser is faster and, below 1e-6,
                           # indistinguishable in the part

# [growth]                 # optional, and only legal with engine = "growth";
# seed = 1                 # every key inside it is optional too. See the growth
                           # engine section above for the defaults and what they
                           # are derived from

# [growth.symmetry]        # optional; grow one sector and replicate it
# kind = "mirror"          # "mirror" | "rotational"
# planes = ["x", "y"]      # mirror only; one or two plane NORMALS through the
                           # domain centre. Two of them quarter the domain
# order = 4                # rotational only; sectors of the full turn, 2 .. 12.
                           # 2 and 4 are exact on the voxel lattice, the others
                           # are exact in the skeleton and approximate in the
                           # rasterized surface
# axis = "z"               # rotational only, default "z". See the symmetry
                           # section above, including what it does *not* check

[output]
stl_path = "cantilever.stl"  # required; relative paths resolve against the
                             # directory holding the config file
iso_level = 0.5              # optional, default 0.5
smoothing_iterations = 10    # optional, default 10; Taubin passes over the
                             # extracted surface. Moves vertices, never adds any
supersample = 1              # optional, default 1; integer 1 .. 4. Meshes a
                             # lattice this many times finer in every axis.
                             # Triangles and file size grow with about its
                             # square, memory with its cube. See the surface
                             # quality section above
voids = "warn"               # optional, default "warn"; "fill" seals enclosed
                             # cavities with material before meshing
islands = "cull"             # optional, default "cull"; removes the components
                             # of the exported surface that touch no support,
                             # load or keepin region, before it is validated and
                             # written. "keep" exports the extracted surface,
                             # fragments and all. See the islands section above
boundaries = "exact"         # optional, default "exact"; puts every exported
                             # vertex that lies inside a keepout or outside the
                             # domain back onto the analytic surface it violates,
                             # and seats the ones resting up to half a voxel
                             # short of a boundary onto it too, so a bore in the
                             # STL is the cylinder that was asked for and a wall
                             # does not scallop. "voxel" exports the isosurface exactly
                             # as the voxel field produced it - the behaviour
                             # before 0.22.0. See the exact boundaries section
trim = "off"                 # optional, default "off"; "stress" removes the
                             # cells of the part whose von Mises stress, over
                             # every load case, is a negligible fraction of the
                             # part's peak - and refuses the whole pass if that
                             # would disconnect two declared regions. Nothing to
                             # do with [growth] prune, and rejected outright by
                             # engine = "solid", which exports the domain exactly.
                             # See the trimming section
trim_stress_fraction = 0.01  # optional, default 0.01; open interval (0, 1). The
                             # fraction of the peak that counts as negligible
reinforce = "off"            # optional, default "off"; "min_thickness" thickens
                             # every member of the part that came out thinner
                             # than reinforce_thickness_mm up to it, into the
                             # design space around it and never into a keepout
                             # or out of the domain. Nothing to do with the
                             # growth engine's own thickening, and rejected
                             # outright by engine = "solid" for the reason trim
                             # is. See the reinforcement section
reinforce_thickness_mm = 3.0 # optional, default [optimization] min_feature_mm;
                             # a positive length in millimetres. The floor the
                             # pass above holds every member to
flush = "off"                # optional, default "off"; "walls" fills the design
                             # cells within flush_depth_mm of a domain or keepout
                             # surface that the part's own material already
                             # reaches into, so a wall resting against a shape
                             # the configuration drew comes out flush with it
                             # rather than rippled. Material that stopped within
                             # that depth of a surface is joined to it. Rejected
                             # outright by engine = "solid" for the reason trim
                             # is. See the flush fill section
flush_depth_mm = 3.0         # optional, default two voxels of the grid; between
                             # half a voxel and eight of them. How deep the pass
                             # above fills
# stress_json = "stress.json" # optional; writes the stress table as JSON,
                             # relative to the config file

[[domain]]                 # ordered CSG: union of the adds minus the subtracts
op = "add"                 # "add" | "subtract"
shape = "box"
min = [0.0, 0.0, 0.0]
max = [120.0, 40.0, 40.0]
# rotation_deg = [0.0, 0.0, 45.0]   # optional, box and ellipsoid; see below
# shape = "cylinder":  p1 = [x, y, z], p2 = [x, y, z], radius   (capped cylinder)
# shape = "sphere":    center = [x, y, z], radius
# shape = "ellipsoid": center = [x, y, z], radii = [rx, ry, rz], and the same
#                      optional rotation_deg. All three radii must be positive
# shape = "tube":      p1 = [x, y, z], p2 = [x, y, z], radius, and an optional
#                      bend = [x, y, z]. A round-ended capsule, curved into a
#                      circular arc through the three points when it has one
# shape = "cone":      p1 = [x, y, z], p2 = [x, y, z], radius1, radius2. A
#                      capped frustum: radius1 at p1 must be positive, radius2
#                      at p2 may be zero, which is the apex of a true cone
# shape = "triangle":  a = [x, y, z], b = [x, y, z], c = [x, y, z], thickness.
#                      The triangle through the three points, extruded half a
#                      thickness either side of its own plane

[[keepout]]                # must stay empty; same shape keys, no op. Under the
shape = "cylinder"         # default [output] boundaries = "exact" the exported
p1 = [60.0, 20.0, 0.0]     # surface honours this shape exactly rather than to
p2 = [60.0, 20.0, 40.0]    # the voxel grid: see the exact boundaries section
radius = 8.0

[[keepin]]                 # forced solid and not a design variable, for load
shape = "box"              # pads and anchor bosses
min = [110.0, 15.0, 15.0]
max = [120.0, 25.0, 25.0]

[[supports]]               # nodes inside the region get their listed degrees
region = { shape = "box", min = [0.0, 0.0, 0.0], max = [0.5, 40.0, 40.0] }
directions = ["x", "y", "z"]   # optional, default all three

[[loadcases]]
name = "main"
weight = 1.0               # optional, default 1.0; the objective is
                           # sum_i weight_i * compliance_i

  [[loadcases.loads]]
  type = "force"           # total force in N, spread evenly over the nodes the
  region = { shape = "sphere", center = [115.0, 20.0, 20.0], radius = 4.0 }
  vector = [0.0, 0.0, -50.0]

  # [[loadcases.loads]]
  # type = "torque"        # total torque in N*mm about an axis; the nodal
  # region = { ... }       # forces are tangential, scale with the distance from
  # axis_point = [0.0, 0.0, 0.0]   # the axis and cancel as a net force
  # axis_dir = [1.0, 0.0, 0.0]
  # magnitude_nmm = 500.0

  # [[loadcases.loads]]
  # type = "gravity"       # the structure's own weight; no region, it acts on
  # direction = [0.0, 0.0, -1.0]   # every element. Optional, default [0, 0, -1]
  # g_mm_s2 = 9810.0       # optional, default 9810 mm/s^2
```

### Tubes

A `tube` is everything within a radius of a curve. With two points it is a
**capsule** - a barrel with a hemisphere on each end - and with a third, the
`bend`, it is the **circular arc** through the three. It is legal wherever a
shape is: a `[[domain]]` entry, a `[[keepout]]`, a `[[keepin]]`, a support
region, a load region. In the editor it is the one shape that is **drawn rather
than dropped**: the `tube` button takes two clicks for its ends, and the middle
handle then curves it.

```toml
[[keepout]]
shape = "tube"
p1 = [20.0, 16.0, 4.0]
p2 = [44.0, 16.0, 4.0]
bend = [32.0, 16.0, 20.0]    # optional; absent is a straight capsule
radius = 5.0
```

* `p1` and `p2` are the centres of the two **rounded ends**, so the tube reaches
  a radius past each of them: a straight tube spans `|p2 - p1| + 2 * radius`.
  That is the difference from a `cylinder`, which is capped by the flat planes
  through its two points and stops there. Two tubes that meet at a point join
  without a seam; two cylinders do not.
* `bend` is a point the centre line passes through. The curve is then the arc of
  the circle through `p1`, `bend` and `p2` that **goes through the bend** - which
  can be the long way round, so a bend beside or behind an end is a hook rather
  than an error.
* The radius must be a positive, finite number, and it is what the degeneracy
  guard measures: a tube of no length is a ball rather than a degenerate shape,
  so `p1 == p2` is legal and is that ball.
* **A bend on the line through the two ends is the straight tube**, not a
  rejection. Within `TUBE_COLLINEAR_EPS_MM` - a micron - of that line, the arc
  would depart from the chord by less than a micron anywhere along it, three
  orders of magnitude below the finest voxel, and the circle through the three
  points has a radius of millions of millimetres. Such a tube takes the
  segment's arithmetic, bit for bit the arithmetic a tube with no `bend` key
  takes. The editor uses the same rule the other way: drag the bend back onto
  that line and the key is dropped.
* A tube carries no `rotation_deg`. Its points are what point it, as a
  cylinder's are; the editor's rotation arcs turn all three of them together.

**Its distance field is exact, everywhere** - which is more than the ellipsoid's
promises. A tube is the set of points within `radius` of its centre line, so its
signed distance is the distance to that curve minus the radius, and both curves
have a closed form: the segment clamps the projection, and the arc splits the
sample into the circle's own plane and the offset out of it, taking the point at
the sample's own angle when that angle lies inside the arc and the nearer end
when it does not. Its bounding box is exact for the same reason: the extremes of
the curve - its two ends, and the points where the tangent turns perpendicular
to an axis - grown by the radius.

**Picking one is a sphere trace rather than a formula.** A bent tube is not
convex: a ray can enter it, leave it, cross the gap the arc encloses and enter
it again, and there is no quadratic whose roots those are. The editor clips the
ray to the tube's exact bounding box and steps by the distance to the surface,
which - the field being the true distance - can never step through it, for a
bounded number of steps. A ray that grazes the surface reports a miss.

### Cones

A `cone` is a **capped frustum**: the solid between the flat discs through `p1`
and `p2`, carrying a radius at each. It is legal wherever a shape is - a
`[[domain]]` entry, a `[[keepout]]`, a `[[keepin]]`, a support region, a load
region - and it is added from the editor's rows like any other shape.

```toml
[[keepout]]
shape = "cone"
p1 = [20.0, 16.0, 16.0]
p2 = [44.0, 16.0, 16.0]
radius1 = 8.0            # at p1; must be positive
radius2 = 0.0            # at p2; zero is the apex of a true cone
```

* It is the **straight-walled sibling of a `cylinder`**, capped the same way: by
  the planes through its two points, not by anything rounded. `radius1 ==
  radius2` is that cylinder exactly - the same field, at every point of space -
  so a cone is a generalization of one rather than a second answer to the same
  question.
* **`radius2` may be zero**, and that is the apex of a true cone rather than a
  degenerate shape. `radius1` is the one that has to be a positive, finite
  number: the wide end has to be a real disc, and it is what the degeneracy
  guard measures, together with the length of the axis. A negative `radius2` is
  refused by name.
* The narrow end is legal at either end of the taper: `radius2 > radius1` is a
  cone that opens out, and nothing about it is a special case.
* A cone carries no `rotation_deg`. Its points are what point it, as a
  cylinder's are; the editor's rotation arcs turn both of them together.
* Its bounding box is exact. A frustum is the convex hull of its two cap discs,
  so its extent along a world axis is the further of the two discs' own -
  `radius * sqrt(1 - (axis_d / |axis|)^2)` either side of each cap centre.

One thing to know if you read the source: the viewer's tessellation of this
shape is `tessellate::frustum`, not `tessellate::cone`. The latter is older and
is the **arrowhead** every overlay is drawn with - the head of a load vector, the
tip of a gizmo arrow - taken as a base, a direction and a height rather than as a
configuration shape, and it was left where it was rather than renamed under the
code that draws with it.

**Its distance field is exact, everywhere.** A frustum is a surface of
revolution, so the nearest surface point to a sample lies in that sample's own
meridian half plane - every other meridian is further by the arc between them -
and the whole of the geometry is therefore two dimensional: the distance from
`(radial offset, height along the axis)` to the trapezoid the profile is, which
is the nearest of three segments, the two caps and the slanted wall. **Picking
one is a formula rather than a march**, because a frustum is convex: the lateral
quadric's roots between the two caps, and the cap discs, with the nearest of them
the hit. A ray that runs parallel to the slanted wall leaves a linear equation
rather than a quadratic and is solved as one.

### Triangles

A `triangle` is a **triangular prism**: the triangle through `a`, `b` and `c`,
extruded `thickness` millimetres symmetrically about its own plane, half either
side. It is legal wherever a shape is, and is added from the editor's rows like
any other shape.

```toml
[[keepout]]
shape = "triangle"
a = [20.0, 16.0, 4.0]
b = [44.0, 16.0, 4.0]
c = [32.0, 16.0, 26.0]
thickness = 8.0          # half of it either side of the plane through a, b, c
```

* The three points fix a plane and a shape in it, so they **point the prism as
  well as shape it**: there is no `rotation_deg` key, for the sphere's reason
  turned inside out. Three free vertices are already complete control of where
  the shape is and which way it faces.
* `thickness` must be a positive, finite number, and it is centred on the plane:
  the prism reaches half of it either side, so moving the three points never
  moves the solid off them.
* **Three points on one line are refused**, however far apart they are, with
  their shortest altitude named. This is deliberately unlike a tube's `bend`
  landing on the line through its two ends, which is still a tube: a triangle of
  no area encloses nothing at any thickness, so there is no shape to fall back
  on.
* What makes a prism thin is the **ball that fits inside its triangle**, not how
  far apart its corners are. Its smallest extent is the smaller of the thickness
  and the diameter of the inscribed circle (`2 * Area / perimeter`), which is the
  same "smallest ball that fits" the minimum feature size is written in - so a
  sliver a metre long is a sliver.
* Its bounding box is exact and needs no inflation: the prism is the convex hull
  of its six corners - the three vertices, half a thickness either side of the
  plane - and their extremes are the box.

**Its distance field is exact, everywhere.** The prism is the product of the
triangle and the slab, so a sample's distance to it splits the way a box's does:
the exact distance within the plane - the nearest of the three edges, each
clamped to its own ends, which is what puts a sample in an edge region or a
vertex region without a case analysis of its own - combined with the distance out
of the slab. **Picking one is a formula too**: the box's slab method over the
five planes a prism is bounded by, two faces and one per edge.

### Ellipsoids

An `ellipsoid` is a sphere with a radius per axis. It is legal wherever a shape
is - a `[[domain]]` entry, a `[[keepout]]`, a `[[keepin]]`, a support region, a
load region - and takes the same optional rotation a box does:

```toml
[[keepout]]
shape = "ellipsoid"
center = [60.0, 20.0, 20.0]
radii = [24.0, 8.0, 8.0]          # semi-axes along its own x, y, z; all > 0
rotation_deg = [0.0, 0.0, 30.0]   # optional; absent is the same as [0, 0, 0]
```

* `center` is the middle of the shape and the point it turns about. `radii` are
  **semi-axes**, not diameters: the shape spans `2 * rx` along its own x. Equal
  radii are a sphere, and behave as one.
* Every radius must be a positive, finite number; a zero or negative one is
  refused by name (`radii[1]`), because it is a shape nothing can sample.
* The rotation is the box's, exactly: degrees about **x, y and z** applied about
  the **fixed world axes** in that order (extrinsic XYZ,
  `R = Rz(rz) * Ry(ry) * Rx(rx)`), about the shape's own centre, right hand
  rule, any magnitude, and absent is not the same as written - see the section
  below, all of which applies here too.
* The bounding box of a turned ellipsoid is exact. It has no corners to take the
  extremes of, so the half extent along each world axis is that axis's support
  function, `sqrt(sum_j (R[i][j] * r_j)^2)`; unrotated that is `center +- radii`.

**One thing to know about its distance field.** A box, a capped cylinder, a
sphere and a tube carry their exact signed distance. An ellipsoid's has no closed form (the
nearest point on it is the root of a sextic), so growforge uses the scaled-space
field: `(|q / r| - 1) * min(r)` on the centred, inverse-rotated sample. **Its
zero level set is exact** - the surface is exactly where the field says it is -
so every question growforge asks a shape is answered exactly: cell
classification samples cell centres, support and load region selection samples
nodes, keepout/keepin precedence and the domain CSG read the sign, and the
editor's picking, hovering and containment use the exact intersection and the
exact bounds. Away from the surface the magnitude is a lower bound on the true
distance rather than the distance itself - never larger, and never changing by
more than one millimetre per millimetre travelled. Nothing in growforge reads
it, so nothing is approximate because of it; it is written down here because the
field is public and a future consumer that wants a true distance would need to
know.

### Rotated boxes

A `box` - in a `[[domain]]` entry, a `[[keepout]]`, a `[[keepin]]`, a support
region or a load region - takes an optional rotation:

```toml
[[keepout]]
shape = "box"
min = [40.0, 10.0, 10.0]
max = [60.0, 30.0, 30.0]
rotation_deg = [0.0, 0.0, 45.0]   # optional; absent is the same as [0, 0, 0]
```

* The three angles are degrees about **x, y and z**, applied about the **fixed
  world axes** in that order - extrinsic XYZ, which composes as
  `R = Rz(rz) * Ry(ry) * Rx(rx)`. Every angle is measured about a world axis,
  not about an axis a previous rotation already moved.
* The box turns about its **own centre**, which is the midpoint of `min` and
  `max`. Those two corners describe the box before it is turned and keep meaning
  what they meant: the size of the box is `max - min` whichever way it faces.
* Positive angles follow the right hand rule: a quarter turn about z takes +x
  onto +y.
* Any magnitude is legal - it is an angle, not a bound - and what you wrote is
  what the file keeps. The editor's fields normalize what they *show* into
  [-180, 180].
* **Absent is not the same as written.** A box with no `rotation_deg` key is the
  axis aligned box it always was, down to the last bit of arithmetic, and a file
  that never had the key never grows one.

Boxes and ellipsoids take it. Spheres do not need one, a cylinder, a tube and a
cone are already described by the points they run between, a triangle by the
three its vertices are, and `rotation_deg` on any of them is an unknown key. Everything downstream reads the shape through its signed
distance function and its bounding box, so a rotated box or ellipsoid voxelizes,
selects nodes, exports and picks exactly like any other shape.

### Precedence and classification

Each cell is classified by its centre point: inside a keepout it is **void**,
otherwise inside a keepin it is **solid** (density pinned at 1, not a design
variable), otherwise inside the domain it is a **design** cell, otherwise void.
Keepout therefore beats keepin, which beats the domain; growforge warns when a
keepout and a keepin overlap.

### What is rejected, and what only warns

Rejected with an explanatory message: any number that is not finite - TOML's
`nan`, `inf` and `-inf` are legal literals and serde takes them, so every float
in every table, shape, region and load is checked and the offending key is
named - an empty or degenerate domain, a domain
with no design cells left, a triangle whose three points lie on one line (it has
no area, and no thickness gives it a solid), a cone whose wide end has no radius,
no supports, no load cases, a load case with no
loads, a support or load region that selects no node touching material, a load
case whose loaded degrees of freedom are all constrained, a missing
`mass_fraction` (except under `engine = "solid"`) or one outside (0, 1), both or
neither resolution key, a resolution that would lay out
more than `constants::MAX_GRID_CELLS` cells (either key can ask for one: a huge
`target_cells` or a tiny `voxel_size_mm`; the message names the key, the grid it
derives and the budget), a torque whose selected nodes all lie on its axis, a
gravity load with a zero direction or a non-positive `g_mm_s2`, a load case
whose gravity loads cancel out, an unknown build direction, an unknown void
policy, an unknown material preset, an unknown engine, an unknown key anywhere.

Rejected for a growth run specifically: a `[growth]` table without
`engine = "growth"`, `engine = "growth"` with `[optimization.overhang]` or with
`[optimization.wireframe]`, a degenerate growth control (see the growth engine
section), a problem whose only loads are gravity, and a load region with no path
to any support. Nothing about pruning is ever a rejection: a surface no branch
could reach is reported, and the branches that were heading for it are removed.

Rejected for a solid run specifically: `engine = "solid"` with
`[optimization] mass_fraction` set (it fills the domain, so there is no share of
it to target), with `[optimization.overhang]`, `[optimization.wireframe]` or
`[optimization.local_volume]`, or with `[output] trim`, `reinforce` or `flush`
set to anything but `"off"` - every one of those passes alters the very domain
that engine exports.
`mass_fraction` is required for every other engine and refused for this one, so
a file switching between them gains and loses the key.

Rejected for a symmetric growth run: a `[growth.symmetry]` table without
`engine = "growth"`, a kind carrying the other kind's keys, no `planes` or more
than two of them, the same plane twice, a missing `order` or one outside
`[2, 12]`, and a fundamental domain holding no load region or no support region.

Also rejected, whatever the engine: `[output] flush_depth_mm` outside half a
voxel to eight voxels of the grid the run is solved on, which is a range only
the voxel size can be measured against and so is checked once the grid is known.

Warned about but allowed: `min_feature_mm` smaller than about three voxels (the
density filter cannot resolve it), keepout/keepin overlap, an enclosed cavity
under `voids = "warn"`, an exported field holding more than one connected body of
material - always a defect for a tool that designs one part, never a reason to
throw the part away - a surface that came out in more than one piece, whose
fragments are culled under the default `islands` policy and reported either way,
a load or support region that the declared symmetry maps onto nothing, and - for
a guide wireframe - a region it cannot reach, a region that holds no material at
all, and a `hold_iterations` that outlasts `max_iterations`.

## How it works

1. **Voxelize.** Signed distance functions for box, capped cylinder, sphere,
   ellipsoid, tube, cone and triangular prism, combined in list order (`add` is a
   union, `subtract`
   cuts). A box or an ellipsoid may carry a rotation, which is applied by
   inverse-rotating the sample point about the shape's centre; the box's
   distance field stays exact and the ellipsoid's stays exact on its surface,
   which is where classification reads it (see the ellipsoid section). A tube's
   is exact everywhere, being a distance to its own centre line; a cone's is a
   two dimensional distance in its own meridian half plane, and a prism's is its
   triangle and its slab combined. The grid is
   axis aligned with cubic cells covering the domain bounding box.
2. **Analyse.** 8-node hexahedral elements with full 2x2x2 Gauss integration,
   `E(x) = Emin + x^p (E0 - Emin)` with `p` from `[optimization] penalty`,
   3 by default, and `Emin` from `[optimization] stiffness_floor` times `E0`,
   1e-9 of it by default. `K u = f` is solved
   matrix-free by Jacobi preconditioned conjugate gradients; supports are
   applied by projection, and the element loop is parallelised over eight
   parity colours so scatter-adds never race. Displacements warm-start from the
   previous iteration. Self-weight, being design dependent, is reassembled into
   `f` every iteration from the current physical densities.
3. **Optimize** (`engine = "simp"`). Design variables live on the design cells only. A density
   chain maps them to the physical densities that drive the analysis: a linear
   cone filter of radius `min_feature_mm / 2` first, then the self-supporting
   filter when one is configured. Sensitivities travel back through both
   transposes, the second one being the reverse layer sweep. The configured
   update scheme then enforces the volume constraint by bisection on its
   Lagrange multiplier, measuring the volume on the printed densities at the far
   end of the chain: either the optimality criteria step with a move limit of
   0.2 and damping 0.5, or the method of moving asymptotes, whose subproblem has
   a closed form primal per variable and the same scalar dual. See the update
   scheme section. With `[optimization.local_volume]` set there is a second
   constraint - a cone average of the printed densities over a larger
   neighbourhood, aggregated by a p-mean - and the moving asymptotes subproblem
   becomes a dual in two multipliers, maximized by exact coordinate ascent. See
   the local volume section.

   **Or grow** (`engine = "growth"`). Steps 2 and 3 are replaced entirely: A*
   load paths from every load region to every support region it can reach,
   space colonization for the canopy, Murray's law over the accumulated flow for
   the radii, and capsules unioned with a smooth minimum for the field. No
   finite element solve is involved at all. See the growth engine section.

   **Or hold solid** (`engine = "solid"`). Steps 2 and 3 do not happen at all:
   every design cell is filled and the field handed on is the cell
   classification itself, so the part that ships is the design space that was
   drawn. There is no mass fraction to set (the key is rejected), no progress
   line to print and no compliance to quote; step 4 onwards runs exactly as it
   does after either optimizer, and the boundary clamp of step 5 is what makes
   the exported surface the analytic shapes rather than a voxelization of them.
   For the part that was never a topology problem - a plug, a housing, a
   fixture - which before this had to be faked at a mass fraction just under
   one, where the optimizer moves material it was not asked to move and the
   density filter erodes the boundary cells.
4. **Post-process.** Enclosed cavities are resolved under the `[output] voids`
   policy, and the resulting field is solved once more per load case to recover
   the von Mises stresses. Neither can move the optimization, which is over.
   Under `[output] trim` the stress envelope those solves produce is then used
   once more, to remove the material of the part that carries none of the load -
   unless doing so would disconnect two declared regions, in which case the
   whole pass is refused and the run exports untrimmed. Under
   `[output] flush` the design cells within a depth of a domain or keepout
   surface that the part's own material already reaches into are then filled, so
   a wall standing against a shape the configuration drew reaches it. And under
   `[output] reinforce` a Euclidean distance transform over the part finds the
   spine of every member, and the ones whose inscribed ball is thinner than the
   floor are thickened into the design cells around them; a place the fill could
   not reach is counted and warned about rather than refusing the pass. If any
   of the three changed the field, the cavity pass and the stress solve run
   *again* over the result - once, for all of them - so every number the run
   reports describes what was written. See the trimming, flush fill and
   reinforcement sections.
5. **Export.** Node values are the mean of the eight surrounding cell
   densities, wrapped in a layer of zeros. With `supersample = N > 1` that
   lattice is then resampled `N` times finer in every axis from its own
   trilinear interpolant, which copies the original samples exactly and keeps
   the zero padding zero. Marching cubes extracts the `iso_level` surface,
   welding vertices by global lattice edge - the refined lattice's own edges
   when there is one, which is what keeps a supersampled surface closed by the
   same construction - and Taubin smoothing (lambda 0.5, mu -0.53) rounds it
   without shrinking it. The surface is then partitioned into connected
   components and its floating fragments culled under the `[output] islands`
   policy. Under the default `[output] boundaries = "exact"` every vertex that
   still lies inside a keepout or outside the domain is then projected onto the
   analytic surface it violates - closed form for the box, sphere, capped
   cylinder, tube, cone and triangular prism, a bounded iteration for the
   ellipsoid and for the domain composite - so the shapes the configuration described survive the voxel grid.
   Only then is the mesh checked for watertightness, manifold edges, consistent
   winding and degenerate triangles and written as a binary STL: what is
   validated is what ships.

6. **Show (optional).** With `--view`, the engine also hands each iteration's
   physical densities to the reporter through a default no-op hook. The viewer's
   reporter copies them into a one-slot channel; a mesher thread takes the
   newest and runs marching cubes alone on it, skipping smoothing, supersampling
   and validation, expands it into a triangle soup with area weighted per-vertex
   normals, and pushes the result into a second one-slot channel the window
   drains. Every normal a frame carries is computed there, never on the thread
   that draws. A snapshot that nobody has taken is simply overwritten, so the
   preview rate adapts to whichever of the solver and the mesher is slower and
   the link can never hold more than one design.

One consequence of the export worth knowing: because a node value averages eight
cells, the outer edges of a fully solid domain come out with a one-voxel 45
degree chamfer. Supersampling does not change it - it samples the same field more
finely, so the chamfer is described by more triangles rather than removed. The
other consequence, a surface that drifted a fraction of a voxel outside the
domain or into a keepout, is what `[output] boundaries` now corrects.

## Tuning

Every default and tolerance a configuration does *not* carry lives in
`src/constants.rs` with a doc comment - the defaults of the optional keys
included, so the file is where "what happens if I leave this out" is answered:
the default SIMP penalty and its lower bound, the default design-cell stiffness
floor and the bounds of the key that carries it, filter
radius divisor, OC move limit,
damping and self-weight shift margin, the MMA move limit, asymptote
initialization, shrink and expand factors and their clamps, the local volume
cap's default fraction and radius factor, its aggregation exponent and its dual's
tolerance and sweep budget, the default iteration
budget and the stall criterion's window, compliance and change thresholds and its
traversing guard, the self-supporting
filter's smooth minimum and maximum parameters, the gravity defaults and the g/cm^3 to
tonne/mm^3 conversion, the stress density threshold, percentiles and its own
solve budget, the voxel grid's cell budget, conjugate gradient tolerance and
iteration cap, the GPU solver's
workgroup shape, refinement targets and stall thresholds, the benchmark's solve
count and tolerance, Taubin coefficients, the supersampling cap and its lattice
budget, material presets, the
mesh validation thresholds, the growth engine's PCG32 constants, length defaults,
canopy load share, smooth union and surface widths and volume bisection limits,
and every viewer colour, size, camera parameter and throttle interval, including
the editor's panel width, undo depth, refresh debounce, click slop, gizmo and
handle proportions, new-object defaults and fast preview budgets.

## Development

```sh
cargo fmt
cargo clippy --all-targets
cargo clippy --all-targets --no-default-features
cargo clippy --all-targets --no-default-features --features gpu
cargo clippy --all-targets --no-default-features --features viewer
cargo doc --no-deps --all-features
cargo test --release
cargo test --release --no-default-features
```

Tests that need a compute adapter skip with a printed note rather than failing,
so the suite passes unchanged on an adapterless machine and in a build without
the `gpu` feature.

The suite covers the element stiffness (symmetry, rigid body modes, linear
scaling in `h`), voxel classification, the density filter and its transpose,
the conjugate gradient solver against a direct solve, a single element against
the analytic `F L / (E A)` answer, the marching cubes table over all 256 corner
configurations, a sphere against its analytic volume, the STL round trip, and a
small cantilever through the whole pipeline.

The phase 3 work adds: the self-supporting filter (a floating blob is erased, a
45 degree ramp and a standing column survive, the build plate layer prints its
blueprint, the build direction picks the plate) and its adjoint against a
directional finite difference; **a finite difference check of the complete
`dC/dx` chain on a 6 x 6 x 6 grid for three configurations — plain, overhang on
and self-weight on — all three agreeing with central differences to about 1e-9
relative**, which is what gates the rest; the self-weight force assembly against
`rho V g` and the dimensional analysis of the density conversion; cavity
detection on a hollow cube, an open channel and a keepout; the von Mises stress
of a uniaxial bar against `F / A`; and a compliance trajectory recorded from the
0.2.0 binary, which pins the plain path against the refactor phase 3 needed.

The viewer adds the overlay tessellation (each primitive against its analytic
volume, all closed and free of degenerate triangles), the camera fit and
projection maths, the latest-only snapshot channel, the observer hook's no-op
default, the stress ramp and the plain/stress shading toggle.

The phase 4 growth engine adds: the PCG32 stream against the published
`pcg32-demo` reference vector; the voxel walk catching a segment that clips a
keepout for a tenth of a voxel, which sampling would step over; A* routing
through the one gap in a wall, refusing a walled-off target and reproducing the
straight line on an open block; the shortcut pass on an adversarial zigzag
corridor, which may not cut a single corner; interior attraction points landing
only in design cells and surface ones only on anchors, a surface point holding on
until a branch has actually arrived, branches never entering a forbidden cell,
and a boxed-in branch giving up and then being given up on; Murray's law at
every unloaded junction and the volume bisection against both its clamps; a
single capsule against its analytic volume, the smooth union's monotonicity, and
keepout/keepin pinning; **the same seed growing a bit-identical field and a
different seed growing a different one**; **pruning taking a dead-end chain back
to its junction, keeping one that reaches the plate, `prune = false` keeping what
`prune = true` removes, a fused tip measurably taking load off the backbone,
**a tip placed exactly on the old knife-edge tolerance being refused as merely
tangent**, and **a foot planting in the middle of its support patch rather than
on the rim nearest the load**; the solid body flood fill on a floating island, on a diagonal touch
that is not a joint, and on the classifier's pinned cells; and the whole example
end to end with a connectivity flood fill from the supports, **an assertion that
not one skeleton leaf ends free and that the part is one connected body**, and a
byte-for-byte comparison of two independently grown STL files. The whole suite
runs in about half a minute in release mode, most of it the stress solve of that
last one. Nothing in it opens a window; the window itself is smoke tested with
`GROWFORGE_VIEW_AUTOCLOSE_S`.

Growth symmetry adds: the reflection and quarter-turn matrices being **exact**
(`-1` and `0`, not `6.1e-17`), a replicated point set against hand-computed
positions, the copies covering every point of the block exactly once, a point
*on* the boundary belonging to the fundamental domain whichever way it rounds,
and which transforms land on cell centres (every mirror, a half turn, a quarter
turn on axes of matching parity - and not a third, a fifth, a sixth or a
twelfth); the confinement - attraction points scattered only in the sector and at
the same density as in a whole domain, a step across the plane refused like a
step into a keepout, one backbone routed where an asymmetric run of the same
problem routes four, and the copies still standing on all four supports; the
volume arithmetic, where the quarter that was grown is at the same fraction of
its own design cells as the whole structure is of all of them; a strut ending on
the plane meeting its twin as **one** connected body with no cell over-full;
**the exported field being identical to its own mirror in both planes, to 1e-12,
where the asymmetric run of the same problem differs by more than half a
density**; **a six-fold column through the whole pipeline, where every cell more
than a voxel from a surface still matches its rotated image exactly, the cells
that differ are capped as a fraction, and two runs are byte identical** - the
honest statement of what a rotation off the voxel lattice costs; the region
warning firing on a lopsided problem, naming the region, and staying silent on
the symmetric one; the configuration bounds and the `[growth.symmetry]` round
trip through the editor's file layer; and the shipped symmetric example end to
end, hashed twice.

The surface quality work adds: **`supersample = 1` writing the same STL bytes as
the pipeline that has no supersampling in it at all**, which is what pins the
refinement as an addition rather than a refactor; the refined lattice copying
every source sample exactly and agreeing with the source's own trilinear lookup
everywhere between them; a sampled sphere landing closer to its analytic volume
at `2` than at `1` with about four times the triangles; an asymmetric field
staying watertight and crack free at `3`, which is the welding keying on the
refined lattice's edge identities; the lattice budget warning firing on a
contrived 80^3 grid at `4` and staying silent at `1`; the configuration bounds;
area weighted vertex normals against an independently weighted sum on a cube and
a stretched box, and pointing outwards on a sphere; a supersampled surface
reading the same trilinear stress field its coarse counterpart does; and **the
same growth configuration exporting the same bytes twice at `supersample = 2`**.

The editor adds, all without opening a window: every object type added, edited
and deleted through the same validation path a run uses, and each addition on its
own leaving a configuration that still builds; the undo stack's semantics
(bounded depth keeping the newest steps, one interaction being one step however
many values it passes through, redo cleared by a new edit, the selection restored
with the configuration, and a selection that no longer exists being no
selection); the ray casting (box, sphere and capped cylinder each hit on their
near face, on their caps and from inside, missed beside them and behind the
camera, the nearest hit winning, and a ray cast at the projected position of a
known point coming back to that point through the very camera the frame was
drawn with); **an enclosing shell never swallowing a click**, which is what makes
the domain pickable without making it the only thing pickable; the drag maths
(axis constrained translation moving on that axis and no other, a plane drag
following the pointer, faces and corners resizing without ever inverting a box,
radii and cylinder caps floored, a drag applied to the shape it started on rather
than accumulated, and a ray parallel to a handle giving no motion rather than a
runaway); the debounce collapsing a burst of edits into one refresh; **the SIMP
fast preview's arithmetic** (a grid inside the budget untouched, eight times the
budget coarsened by exactly two, the iteration cap never raising a configuration's
own smaller one, and a growth preview left at the real resolution); and the TOML
round trip on a comment rich fixture and on **all four shipped examples byte for
byte** - one value edited leaving every other byte alone and keeping its trailing
comment, an added object reparsing to what was added, a deleted object taking its
own comments and no others, an optional key that was switched off disappearing,
an inline `region = { ... }` staying inline, an object list written as an inline
array being edited rather than duplicated, a Windows authored file keeping its
line endings, and **an integer larger than the format preserving parser can hold
at all - `growth_canopy.toml`'s `u64` seed, and any of the five counts, which are
`usize` and so just as wide - surviving an untouched save, an edit past the
signed range, and an edit back below it**, which is what keeps a save from ever
writing a number as its wrapped negative; twenty successive edits to such
values each reaching the file, the stand-ins that carry them pruned to the ones
the document holds, and a save that cannot write one failing loudly rather than
leaving the old number behind.

Island culling adds: component labelling on a two-fragment mesh, on a body with a
cavity and under a permuted triangle order, which is what pins the labels to the
triangles rather than to the vertices; the purpose rule (**a keepin boss larger
than the structure shipping beside it while the speck between them is culled**,
which is the defect ranking by volume had; a shell that serves nothing culled and
reported with its volume; `islands = "keep"` leaving the mesh **exactly** as it
was; nothing culled at all when no shell reaches anything declared, with and
without a declared region to reach); the cavity attribution (a fragment's own
cavity leaving with it, and **a fragment rattling inside the part's cavity
leaving without taking the part's cavity shell with it**, which is the case a
containment test that only asks the part gets wrong); an inside-out mesh left for
the validator to refuse rather than culled into silence; the lattice match, half
a voxel of slack and no more, outside the lattice and on a `nan`; **a declared
region buried in the middle of a body being held by it through containment**,
which is the shipped cantilever's own shape; **the probes of a pad split by a gap
landing in both patches** on the 168-cell, 21-per-row fixture that makes an index
stride alias exactly and on the 100-by-12 one that starves a widest-first
partition of its short axis, and a region with no material left having none at
all; **a culled fragment inside a declared region being named while the region
itself reads as served**, which is the partial loss the unserved check cannot
see, together with the fragment note dropping its "nothing declared asked for it"
wording exactly there; a tiny anchored body warned about beside another and a
single one of any size left alone; the closed shell containment test
just inside and just outside a face and on the axis of one, where a ray cast would
have to choose a direction; and, through the whole export, **a field engineered to
mint the user's defect - a clump joined to the part by a
one-cell-wide bridge, which the cell flood fill walks across and the node lattice
cannot see - coming out as two components that the cell level reading calls one
body**, culled to a single validated body whose bytes are those of the same field
without the clump in it, kept intact under `islands = "keep"`, composing with
`supersample` 1, 2 and 3, and leaving a `voids = "warn"` cavity shell alone while
`voids = "fill"` leaves nothing to leave alone; the summary lines' own formats for
every combination they can print; the configuration round trip and rejection;
**a disconnected keepin boss and the load path it does not touch both surviving a
real run end to end**; and **the shipped growth examples and the cantilever
exporting byte-identical files through the new path**, the cantilever's against
the pre-culling pipeline run inline in the test.

The MMA update adds: the closed form primal against a case worked out by hand and
against the stationarity condition it solves, plus its monotonicity in the
multiplier, which is what makes the dual bisectable; the dual bisection landing
on three different volume targets; the asymptote rules (an oscillation shrinks
the range by 0.7, monotone progress widens it by 1.2, standing still leaves it,
and both clamps hold under repeated shrinking); the subproblem bounds against the
box, the move limit and the asymptote margin; a mixed sign sensitivity field
needing no shift and staying continuous across `dC/dx = 0`; only design cells
moving; **the recorded pre-phase-3 compliance trajectory still reproducing
exactly, because `oc` is untouched and remains the default**, and a second
recorded trajectory through the self-supporting filter's adjoint, which is what
every overhang number depends on and what a refactor can break while the forward
sweep keeps producing a plausible design; MMA reaching the
same compliance as `oc` within 5 % on the plain cantilever; the configured scheme
demonstrably reaching the loop (a different trajectory, and no self-weight shift
announced); and **a miniature of the shipped shelf bracket - flange, loaded pad,
self weight, self-supporting filter - where MMA leaves a smaller final design
variable change than `oc` at the same iteration count**.

The ellipsoid adds: the zero level set against hand-computed surface points (the
six axis points at exactly their own radius, an oblique one with each term a
third, the sign flipping either side of all of them, the centre at exactly minus
the smallest radius, and the whole field agreeing with a sphere's when the three
radii are equal); the same under every rotation, where the unrotated surface
carried through the turn is still on the surface; **the bounding box against a
brute force sweep of the surface, which nothing may leave and which comes up to
every one of the six faces**, plus the support point of each axis landing exactly
on its face, and a hand-computed `sqrt(5)` case; classification carving exactly
the cells whose centres are inside it, hand counted on a 10 mm cube of 1 mm
cells and transposed by a quarter turn; region selection picking the hand-listed
node set on a 2 mm lattice, which is a strict subset of what the box around it
would select; **the analytic ray intersection against a march of the shape's own
field** over anisotropic radii, five rotations and rays walked out from the axis
to well past the surface - with the parameter checked against millimetres along
the world ray, which is what the non-rigid scaling into the unit sphere's frame
would get wrong; the per-axis radius handles sitting on the ends of their own
semi-axes in the shape's own frame and each drag changing that radius alone; the
turn recording the same `rotation_deg` component a box's does; the containment
clamp using the turned bounding box; the tessellation being the sphere's own
mesh with every vertex on the surface it describes and its volume within 1 % of
`4/3 pi abc`; the TOML round trip, `radii` and an optional rotation appearing
and disappearing without touching a byte of anything else; the rejections (a
zero, negative or non-finite radius, named by component); a whole editor session
- pick, drag a semi-axis, callout, save, reopen - and the same through the
window's own input path at three display scale factors; and **an end-to-end run
of a turned ellipsoid keepout on both engines, where every cell inside it is
carved, every cell inside its bounding box but outside it is not, and both parts
export watertight**, together with an ellipsoid support region anchoring a simp
run.

The tube adds: the capsule's own distances, checked where a cylinder's flat cap
would answer differently and at the corner off the end where it would answer
wrongly; **a bend on the line through the two ends reproducing the straight tube
bit for bit**, on and past the segment and a half-epsilon off it; the arc being
the circumcircle of the three points, with the ends and the bend on it, over four
triples including one whose arc goes the long way round; **the field against ten
thousand sampled points of the arc at 385 places, where the closed form may never
over-state what the sampling found**, plus the surface a radius out in every
direction at right angles to the curve; **the exact bounds against a brute force
sweep of the surface, which nothing may leave and which comes up to every one of
the six faces** on four arcs including most of a circle; a tube legal in every
shape position, straight and bent, with a hand-computed circle behind its
bounding box; the rejections (a radius that is not positive, named; a `nan` in
the optional bend); the TOML round trip, `bend` appearing, straightening and
gone; **the bend drag itself, which is the arm the drag dispatcher's wildcard
would otherwise swallow silently** - the middle of a straight tube grabbed as its
bend and the centre handle taking it back once it is curved, an end drag keeping
the bend, a turn carrying it round; the callout for the pull of a bend and the
span of the ends, typed both ways, where typing zero straightens it; **the traced
ray intersection against a march of the shape's own field** over three tubes,
five directions and seven offsets, with **a ray through the concave gap the arc
encloses missing it both ways** while the bounding box is hit; the mesh coming
out closed with every vertex on the zero level set and its volume within 5 % of a
barrel plus a ball; and **an end-to-end run of a bent tube keepout on both
engines**, where every cell inside the curve is carved and every cell inside its
box but outside the curve is not, together with a tube support region anchoring a
simp run.

The cone adds: **the field of a cone whose two radii are equal against a
cylinder's, pointwise over a lattice that covers both of them and a margin
round** - the same numbers, not merely the same surface; the zero level set on
the wall, the caps and the apex of a frustum, a true cone and one that opens out;
**the projection landing on the surface at exactly the distance the field
reported**, over a lattice through every region, and **the field checked against
a sweep of the surface itself**, which is the half of exactness a projection
cannot state; the nearest surface point refused where a whole circle is equally
near - on the axis, including just under an apex, where the wall closes on the
axis faster than the point does - and answered at the cap centres and beyond the
tip, which are single points; **the bounds against the two cap circles, which
nothing inside may leave and which come up to every one of the six faces**,
tilted as well as axis aligned; a cone legal in every shape position with
hand-computed bounds; the rejections (a wide end that is not positive, a negative
narrow end, a `nan` anywhere, a cone of no length); the TOML round trip with
`radius2` written as the zero it is; a radius handle at each end measured from
its own cap, **the apex reachable by drag where every other radius stops at a
tenth of a millimetre**, and the apex grabbed as the radius that can undo it;
**the analytic ray intersection against a march of the shape's own field** over
four tapers, five directions and eight offsets, plus the taper hit at its own
width where a cylinder's would be 3 mm sooner, the apex hit down the axis, and a
ray parallel to the slanted wall, which leaves no quadratic at all; the mesh
closed at every taper, the cylinder's own barrel when the radii are equal, and
the apex a single vertex rather than a ring of coincident ones; and **an
end-to-end run of a cone keepout on both engines** with a cone support region
anchoring a simp run.

The triangle adds: the zero level set on both faces, past an edge and past a
corner where both terms bite; **the projection landing on the surface at exactly
the distance the field reported** through the face, edge and vertex regions of a
lattice, and **the field checked against a sweep of the surface itself**; the
bounds as the hull of the six corners, exactly, turned out of the axes as well as
flat; the smallest extent as the diameter of the inscribed circle, which is what
makes a sliver a metre long a sliver; a triangle legal in every shape position;
the rejections (three points on one line - spread wide, collinear in the middle
and one on top of another - a thickness that is not positive, a `nan` anywhere);
the TOML round trip of three vertices and a thickness; **the vertex drag and the
thickness drag, which are the arms the drag dispatcher's wildcard would otherwise
swallow silently** - a corner moved with the other two and the thickness left
alone, a corner held off the line through the other two and free again past it,
and the thickness growing twice as fast as the handle that moves it; the callouts
for the median through a corner and for the thickness, typed both ways; **the
analytic ray intersection against a march of the shape's own field** over both
windings and a turned prism, plus hits on its five faces and misses beside every
one of them where the bounding box is hit; the mesh closed and outward facing
either way round with exactly its own volume; and **an end-to-end run of a
triangular prism keepout on both engines** with a prism support region anchoring
a simp run.

Placing one by two clicks adds: the landing itself - the nearest surface, the
design space included and a subtracted domain entry excluded, the ruled floor
underneath it, the sky and the space past the ruling landing nowhere, snapped and
bypassed; the whole flow from each of the four add rows, into that row's list
with the shared new-tube radius, selected and carrying its bend handle; `Esc` at
both stages, the button as a toggle and another row's button as a restart; **a
click on an object placing a point rather than selecting it**, with the same
click outside the mode selecting it; the clicks a placement ignores - one that
lands nowhere, one on the point already taken, and one outside the 3D view; the
containment rule applied to what two clicks committed, on and off; and the panel
and the preview drawn at both stages. Alongside them, **a bend drag pushed
through the lid of the domain with containment on**, whose handle and callout
read the 14 mm it committed rather than the 15 it was asked for.
