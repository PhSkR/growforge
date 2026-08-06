// growforge: matrix-free preconditioned conjugate gradient in single precision.
//
// The whole Krylov recurrence lives on the device: one iteration is five
// dispatches and touches no host memory, so a batch of them is recorded into a
// single command buffer and only the residual is ever read back. The host wraps
// this in double precision iterative refinement (see `fea::gpu`), which is what
// lets an f32 inner solve deliver the f64 residual the CPU backend promises.
//
// Determinism, on one device and driver:
//
//   * The operator is applied node-centric. Each thread owns one node, gathers
//     the contributions of its up to eight adjacent elements in a fixed corner
//     order and writes its own three degrees of freedom. No atomics, no scatter,
//     so the sum order does not depend on how the scheduler interleaves.
//   * Every reduction is two stage with a fixed workgroup count: a grid-strided
//     per-thread accumulation, a binary tree inside the workgroup, then one
//     workgroup folding the fixed-size partial array. Both stages carry a
//     Neumaier compensation term where the shader compiler honours one - see
//     `compensated_add` - and the shape of the tree is fixed either way.
//   * Every kernel loops a uniform number of times and masks the tail, so the
//     workgroup barriers sit in uniform control flow.
//
// Index conventions match `grid`: element (i, j, k) is `i + nx * (j + ny * k)`,
// node (i, j, k) is `i + (nx+1) * (j + (ny+1) * k)`, degree of freedom `d` of
// node `n` is `3 * n + d`, and local element node `l` sits at offset
// `(l & 1, (l >> 1) & 1, (l >> 2) & 1)`.

// Mirrored by `constants::GPU_WORKGROUP_SIZE`; the test
// `fea::gpu::tests::the_shader_mirrors_the_configured_constants` fails if the
// two ever drift apart. The same holds for every `const` below.
const WORKGROUP_SIZE: u32 = 64u;

// Slots of the `scalars` buffer, mirrored by the `GPU_SCALAR_*` constants.
const SLOT_RZ: u32 = 0u;
const SLOT_PAP: u32 = 1u;
const SLOT_ALPHA: u32 = 2u;
const SLOT_BETA: u32 = 3u;
const SLOT_RESIDUAL_SQ: u32 = 4u;
const SLOT_BREAKDOWN: u32 = 5u;

// Mirrored by `constants::NODES_PER_ELEMENT`, `DOF_PER_NODE`,
// `DOF_PER_ELEMENT`. Identities of the discretization, not tunables.
const NODES_PER_ELEMENT: u32 = 8u;
const DOF_PER_NODE: u32 = 3u;
const DOF_PER_ELEMENT: u32 = 24u;

// Bits in one word of the constraint mask.
const BITS_PER_WORD: u32 = 32u;

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    n_nodes: u32,
    n_dof: u32,
    n_cells: u32,
    n_partials: u32,
    // Always exactly 1.0. Multiplying by it changes no bit, and that is the
    // point: it is the only thing standing between the solution accumulator's
    // compensation limb and a shader compiler that would otherwise prove it
    // zero. See `accumulate_solution`.
    carry_guard: f32,
};

// Group 0 holds everything that changes at most once per design.
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ke0: array<f32>;
@group(0) @binding(2) var<storage, read> moduli: array<f32>;
@group(0) @binding(3) var<storage, read> minv: array<f32>;
@group(0) @binding(4) var<storage, read> free_bits: array<u32>;

// Group 1 holds the Krylov state.
@group(1) @binding(0) var<storage, read_write> xv: array<f32>;
@group(1) @binding(1) var<storage, read_write> rv: array<f32>;
@group(1) @binding(2) var<storage, read_write> zv: array<f32>;
@group(1) @binding(3) var<storage, read_write> pv: array<f32>;
@group(1) @binding(4) var<storage, read_write> apv: array<f32>;
@group(1) @binding(5) var<storage, read_write> partials: array<f32>;
@group(1) @binding(6) var<storage, read_write> scalars: array<f32>;
// The compensation limb of the solution accumulator. `xv` alone is a plain f32
// running sum of hundreds of steps, and its rounding is what decides how good a
// correction the device can hand back at all: see the note on `update`.
@group(1) @binding(7) var<storage, read_write> xlo: array<f32>;

var<workgroup> share_sum: array<f32, WORKGROUP_SIZE>;
var<workgroup> share_carry: array<f32, WORKGROUP_SIZE>;

/// 1.0 for a free degree of freedom, 0.0 for a constrained one.
fn free_of(index: u32) -> f32 {
    let word = free_bits[index / BITS_PER_WORD];
    return f32((word >> (index % BITS_PER_WORD)) & 1u);
}

/// Neumaier compensated addition: `acc.x` carries the running sum and `acc.y`
/// the rounding error dropped so far, which is added back once at the end.
///
/// Best effort, and deliberately so. The carry below is *algebraically* zero -
/// `(a - (a + b)) + b` - and survives only because floating point addition does
/// not associate, so a shader compiler that is allowed to reassociate proves it
/// zero and deletes it. The Vulkan compiler on an RTX 3080 does exactly that,
/// to every call of this function; on a driver that does not, the compensation
/// is there. Both are correct answers to the same question - the sums it guards
/// are the element gather, which the centred gather already defends, and the
/// reductions, which are two stage trees over a fixed shape.
///
/// Forcing these carries too, through `params.carry_guard`, is affordable and
/// provably harmless: measured 2026-07-29 on one device and driver, it costs
/// roughly 2 to 4 % of the wall time of a device iteration and changes no
/// determinism property at all - 96 device solves on a fixed right hand side
/// returned identical iteration counts, residual bits and correction bits,
/// all-guarded and shipped alike. It is left off because it buys no accuracy
/// the centred gather and the fixed tree do not already provide, and because a
/// design past the single precision floor then grinds twenty times longer
/// before conceding to the CPU. Where a carry is *load bearing*, it is taken
/// through [`accumulate_solution`] instead, which the compiler cannot elide.
fn compensated_add(acc: vec2<f32>, value: f32) -> vec2<f32> {
    let total = acc.x + value;
    var carry = acc.y;
    if (abs(acc.x) >= abs(value)) {
        carry = carry + ((acc.x - total) + value);
    } else {
        carry = carry + ((value - total) + acc.x);
    }
    return vec2<f32>(total, carry);
}

/// One conjugate gradient step into the two limb solution accumulator:
/// `acc + step`, keeping the rounding error the high limb cannot hold.
///
/// This is the one compensation in the shader that has to survive the compiler,
/// because it is the one that decides whether the device can hand back a useful
/// correction at all. What the host folds is only as good as the residual it
/// leaves, `||b - K x||`, and one f32 rounding of `x` already leaves
/// `eps * | |K| |x| |` of it; a plain f32 running sum over the thousands of
/// steps a compliant design needs leaves that many times over. Measured on a
/// 192k cell cantilever at mass fraction 0.10: the plain sum bottoms out at a
/// true relative residual of 1.4, above the 1.0 of the zero start it was
/// correcting, while this reaches 4.5e-3 and lets the refinement loop finish.
///
/// `params.carry_guard` is exactly 1.0, so multiplying by it changes no bit. It
/// is read from the uniform block, which is what stops the compiler proving the
/// carry zero: before it was there the limb came back exactly 0.0 for all
/// 610203 degrees of freedom of that cantilever.
fn accumulate_solution(acc: vec2<f32>, step: f32) -> vec2<f32> {
    let total = acc.x + step;
    let rounded = total * params.carry_guard;
    var carry = acc.y;
    if (abs(acc.x) >= abs(step)) {
        carry = carry + ((acc.x - rounded) + step);
    } else {
        carry = carry + ((step - rounded) + acc.x);
    }
    return vec2<f32>(total, carry);
}

/// Fold one compensated accumulator per thread into one value per workgroup.
///
/// Every invocation of the workgroup must call this, and in the same order when
/// it is called more than once: the barriers are what make the tree well
/// defined.
fn reduce_workgroup(lane: u32, acc: vec2<f32>) -> f32 {
    // The previous reduction's result is read by every lane, so the shared
    // arrays may not be overwritten until all of them are past it.
    workgroupBarrier();
    share_sum[lane] = acc.x;
    share_carry[lane] = acc.y;
    workgroupBarrier();
    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            var merged = compensated_add(
                vec2<f32>(share_sum[lane], share_carry[lane]),
                share_sum[lane + stride]
            );
            merged.y = merged.y + share_carry[lane + stride];
            share_sum[lane] = merged.x;
            share_carry[lane] = merged.y;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return share_sum[0] + share_carry[0];
}

/// Number of grid-strided passes that cover `count` items.
fn pass_count(count: u32, stride: u32) -> u32 {
    return (count + stride - 1u) / stride;
}

/// Fold the fixed-size partial array `partials[offset .. offset + n_partials]`.
fn reduce_partials(lane: u32, offset: u32) -> f32 {
    var acc = vec2<f32>(0.0, 0.0);
    let passes = pass_count(params.n_partials, WORKGROUP_SIZE);
    var index = lane;
    for (var sweep = 0u; sweep < passes; sweep = sweep + 1u) {
        if (index < params.n_partials) {
            acc = compensated_add(acc, partials[offset + index]);
        }
        index = index + WORKGROUP_SIZE;
    }
    return reduce_workgroup(lane, acc);
}

/// Start a solve from the right hand side the host wrote into `rv`.
///
/// Leaves `x = 0` in both limbs, `p = 0`, `r` projected onto the free subspace, `z = M r`, and
/// the partial sums of `r^T z` and `r^T r`. `finish_update` then publishes them,
/// and because the host zeroed the scalar buffer it derives `beta = 0`, so the
/// following `update_p` sets `p = z`.
@compute @workgroup_size(WORKGROUP_SIZE)
fn init(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = WORKGROUP_SIZE * groups.x;
    let passes = pass_count(params.n_dof, stride);
    var rz = vec2<f32>(0.0, 0.0);
    var rr = vec2<f32>(0.0, 0.0);
    var index = gid.x;
    for (var sweep = 0u; sweep < passes; sweep = sweep + 1u) {
        if (index < params.n_dof) {
            let residual = rv[index] * free_of(index);
            let preconditioned = minv[index] * residual;
            rv[index] = residual;
            zv[index] = preconditioned;
            xv[index] = 0.0;
            xlo[index] = 0.0;
            pv[index] = 0.0;
            rz = compensated_add(rz, residual * preconditioned);
            rr = compensated_add(rr, residual * residual);
        }
        index = index + stride;
    }
    let folded_rz = reduce_workgroup(lid.x, rz);
    let folded_rr = reduce_workgroup(lid.x, rr);
    if (lid.x == 0u) {
        partials[wid.x] = folded_rz;
        partials[params.n_partials + wid.x] = folded_rr;
    }
}

/// `ap = P K p` on the free subspace, plus the partial sums of `p^T ap`.
@compute @workgroup_size(WORKGROUP_SIZE)
fn matvec(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let nnx = params.nx + 1u;
    let nny = params.ny + 1u;
    let stride = WORKGROUP_SIZE * groups.x;
    let passes = pass_count(params.n_nodes, stride);
    var pap = vec2<f32>(0.0, 0.0);
    var node = gid.x;
    for (var sweep = 0u; sweep < passes; sweep = sweep + 1u) {
        if (node < params.n_nodes) {
            let ni = node % nnx;
            let nj = (node / nnx) % nny;
            let nk = node / (nnx * nny);
            let own = DOF_PER_NODE * node;
            // The three rigid body translations are in the null space of the
            // element matrix, so every row of it sums to zero over the eight
            // nodes of one axis. Gathering displacements *relative to this
            // node's own* therefore changes nothing mathematically, and is what
            // makes the row dot below survivable in single precision: the raw
            // gather is a near total cancellation whose surviving digits shrink
            // with the square of the mesh refinement, and differences of
            // neighbouring displacements have no such cancellation in them.
            let centre = vec3<f32>(pv[own], pv[own + 1u], pv[own + 2u]);
            var accumulated = array<vec2<f32>, 3>(
                vec2<f32>(0.0, 0.0),
                vec2<f32>(0.0, 0.0),
                vec2<f32>(0.0, 0.0),
            );
            // The eight elements this node is a corner of, in a fixed order.
            for (var corner = 0u; corner < NODES_PER_ELEMENT; corner = corner + 1u) {
                let di = corner & 1u;
                let dj = (corner >> 1u) & 1u;
                let dk = (corner >> 2u) & 1u;
                let ei = i32(ni) + i32(di) - 1;
                let ej = i32(nj) + i32(dj) - 1;
                let ek = i32(nk) + i32(dk) - 1;
                if (ei < 0 || ej < 0 || ek < 0
                    || ei >= i32(params.nx) || ej >= i32(params.ny) || ek >= i32(params.nz)) {
                    continue;
                }
                let cell = u32(ei) + params.nx * (u32(ej) + params.ny * u32(ek));
                let modulus = moduli[cell];
                if (modulus == 0.0) {
                    continue;
                }
                // Local index of this node inside that element: the element sits
                // one cell back along every axis the corner offset advanced.
                let local = (1u - di) + 2u * (1u - dj) + 4u * (1u - dk);
                let row = DOF_PER_NODE * local * DOF_PER_ELEMENT;
                var block = vec3<f32>(0.0, 0.0, 0.0);
                for (var other = 0u; other < NODES_PER_ELEMENT; other = other + 1u) {
                    let oi = u32(ei) + (other & 1u);
                    let oj = u32(ej) + ((other >> 1u) & 1u);
                    let ok = u32(ek) + ((other >> 2u) & 1u);
                    let base = DOF_PER_NODE * (oi + nnx * (oj + nny * ok));
                    for (var axis = 0u; axis < DOF_PER_NODE; axis = axis + 1u) {
                        let value = pv[base + axis] - centre[axis];
                        let column = DOF_PER_NODE * other + axis;
                        block.x = block.x + ke0[row + column] * value;
                        block.y = block.y + ke0[row + DOF_PER_ELEMENT + column] * value;
                        block.z = block.z + ke0[row + 2u * DOF_PER_ELEMENT + column] * value;
                    }
                }
                // The element contributions span the full SIMP stiffness
                // contrast, so it is their sum that needs the compensation.
                accumulated[0] = compensated_add(accumulated[0], modulus * block.x);
                accumulated[1] = compensated_add(accumulated[1], modulus * block.y);
                accumulated[2] = compensated_add(accumulated[2], modulus * block.z);
            }
            for (var axis = 0u; axis < DOF_PER_NODE; axis = axis + 1u) {
                let slot = own + axis;
                let value = (accumulated[axis].x + accumulated[axis].y) * free_of(slot);
                apv[slot] = value;
                pap = compensated_add(pap, pv[slot] * value);
            }
        }
        node = node + stride;
    }
    let folded = reduce_workgroup(lid.x, pap);
    if (lid.x == 0u) {
        partials[wid.x] = folded;
    }
}

/// Publish `p^T K p` and the step length `alpha = r^T z / p^T K p`.
@compute @workgroup_size(WORKGROUP_SIZE)
fn finish_pap(@builtin(local_invocation_id) lid: vec3<u32>) {
    let pap = reduce_partials(lid.x, 0u);
    if (lid.x == 0u) {
        scalars[SLOT_PAP] = pap;
        // `!(pap > 0.0)` also catches a NaN, which is what a diverged or
        // indefinite operator produces here.
        if (scalars[SLOT_BREAKDOWN] != 0.0 || !(pap > 0.0)) {
            scalars[SLOT_BREAKDOWN] = 1.0;
            scalars[SLOT_ALPHA] = 0.0;
        } else {
            scalars[SLOT_ALPHA] = scalars[SLOT_RZ] / pap;
        }
    }
}

/// Take the step: `x += alpha p`, `r -= alpha ap`, `z = M r`, and the partial
/// sums of the two inner products the next step needs.
///
/// `x` is accumulated with a Neumaier compensation limb, and that is not a
/// refinement - it is what makes the device solve useful at all on a compliant
/// design. The correction the host folds is only as good as the residual it
/// leaves, `||b - K x||`, and one f32 rounding of `x` already leaves
/// `eps * | |K| |x| |` of it; a plain f32 running sum over the hundreds of steps
/// such a design needs leaves that many times over. Measured on a 192k cell
/// cantilever at mass fraction 0.10 the plain sum bottoms out at a true residual
/// of 1.4 - above the unit right hand side it started from, so the correction
/// makes the solution worse - while the compensated one reaches 4.5e-3. The
/// carry costs three more f32 operations and one more vector.
@compute @workgroup_size(WORKGROUP_SIZE)
fn update(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let alpha = scalars[SLOT_ALPHA];
    let stride = WORKGROUP_SIZE * groups.x;
    let passes = pass_count(params.n_dof, stride);
    var rz = vec2<f32>(0.0, 0.0);
    var rr = vec2<f32>(0.0, 0.0);
    var index = gid.x;
    for (var sweep = 0u; sweep < passes; sweep = sweep + 1u) {
        if (index < params.n_dof) {
            let stepped = accumulate_solution(
                vec2<f32>(xv[index], xlo[index]),
                alpha * pv[index]
            );
            xv[index] = stepped.x;
            xlo[index] = stepped.y;
            let residual = rv[index] - alpha * apv[index];
            let preconditioned = minv[index] * residual;
            rv[index] = residual;
            zv[index] = preconditioned;
            rz = compensated_add(rz, residual * preconditioned);
            rr = compensated_add(rr, residual * residual);
        }
        index = index + stride;
    }
    let folded_rz = reduce_workgroup(lid.x, rz);
    let folded_rr = reduce_workgroup(lid.x, rr);
    if (lid.x == 0u) {
        partials[wid.x] = folded_rz;
        partials[params.n_partials + wid.x] = folded_rr;
    }
}

/// Publish `||r||^2` and the direction update factor `beta`.
@compute @workgroup_size(WORKGROUP_SIZE)
fn finish_update(@builtin(local_invocation_id) lid: vec3<u32>) {
    let rz = reduce_partials(lid.x, 0u);
    let rr = reduce_partials(lid.x, params.n_partials);
    if (lid.x == 0u) {
        scalars[SLOT_RESIDUAL_SQ] = rr;
        let previous = scalars[SLOT_RZ];
        // `rz >= 0.0` is false for a NaN, so a poisoned recurrence stops
        // advancing instead of spreading.
        var beta = 0.0;
        if (scalars[SLOT_BREAKDOWN] == 0.0 && previous > 0.0 && rz >= 0.0) {
            beta = rz / previous;
        }
        scalars[SLOT_BETA] = beta;
        scalars[SLOT_RZ] = rz;
    }
}

/// `p = z + beta p`.
@compute @workgroup_size(WORKGROUP_SIZE)
fn update_p(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let beta = scalars[SLOT_BETA];
    let stride = WORKGROUP_SIZE * groups.x;
    let passes = pass_count(params.n_dof, stride);
    var index = gid.x;
    for (var sweep = 0u; sweep < passes; sweep = sweep + 1u) {
        if (index < params.n_dof) {
            pv[index] = zv[index] + beta * pv[index];
        }
        index = index + stride;
    }
}
