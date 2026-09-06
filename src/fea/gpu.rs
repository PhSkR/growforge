//! Preconditioned conjugate gradient on a compute device, with double
//! precision iterative refinement on the host.
//!
//! # Precision architecture
//!
//! WGSL has no `f64`, and the systems this crate solves are the worst case for
//! single precision: late in a SIMP run the stiffness contrast across the grid
//! is `[optimization] stiffness_floor`, a factor of 1e9 at its default of
//! [`constants::SIMP_EMIN_FRACTION`], while the CPU backend
//! promises the relative residual it was asked for, which is
//! [`constants::CG_RELATIVE_TOLERANCE`], 1e-8, unless `[solver] tolerance`
//! names a looser one.
//! An f32 conjugate gradient cannot get there on its own: f32 has about 1.2e-7
//! of relative precision, so its residual stops falling one to two decades
//! above the target no matter how many iterations it is given.
//!
//! So the device never solves the problem. It solves for a *correction*:
//!
//! ```text
//!   r = b - K x                     (f64, the same CPU operator the CPU backend uses)
//!   y ~= K^-1 (r / ||r||)           (f32, on the device)
//!   x <- x + ||r|| y                (f64)
//! ```
//!
//! repeated until `||r|| / ||b||` meets the caller's tolerance. Each pass costs
//! one CPU matrix-vector product and one device solve, and multiplies the
//! residual by whatever relative accuracy the device solve reached, so two to
//! five passes cover the whole range from a cold start to 1e-8. Each pass asks
//! the device for only the accuracy the outer loop still needs, floored at
//! [`constants::GPU_INNER_TOLERANCE`]. The convergence test is made on the f64
//! residual of the f64 operator, which is why the answer satisfies the same
//! criterion as the CPU one rather than a single precision imitation of it.
//! Normalising the right hand side to unit length before it is narrowed to f32
//! is what keeps the device arithmetic inside its exponent range however small
//! the outer residual has become. A pass is a *trial*: the fold is undone again
//! if the f64 residual says the correction was not one, so the solution this
//! loop hands back can never be worse than the one it was given.
//!
//! Inside the device solve, precision is defended in four places: the gather is
//! centred on the node's own displacement, which is exact and removes the near
//! total cancellation of the raw row dot; the eight per-element contributions to
//! a node are accumulated with Neumaier compensation, because that is where the
//! 1e9 contrast lands; every reduction is a compensated two stage tree over a
//! fixed workgroup count; and the solution accumulator carries a compensation
//! limb of its own, which is what decides whether the correction is worth
//! anything at all on a compliant design.
//!
//! Only the last of those is *forced*. A compensation term is algebraically
//! zero, so a shader compiler that reassociates floating point adds can prove it
//! zero and delete it - which the Vulkan compiler on an RTX 3080 does, to every
//! one of them. The solution accumulator therefore reads its sum back through
//! [`constants::GPU_CARRY_GUARD`], which the compiler cannot see the value of;
//! the other two are left as the driver takes them, because the centred gather
//! and the fixed reduction shape already carry those sums and forcing the
//! carries there costs roughly 2 to 4 % of the wall time of a device iteration
//! (measured 2026-07-29) while changing neither the accuracy nor the
//! reproducibility of a solve. See `pcg.wgsl`.
//!
//! # What single precision cannot do
//!
//! The correction crosses the bus as two f32 limbs, so the best residual a pass
//! can leave is bounded by how well those limbs can describe `K^-1 r` - roughly
//! `eps * | |K| |x| | / ||b||`, the componentwise conditioning of the system.
//! Measured on the 58k cell cantilever whose design a first MMA step has driven
//! to the density floor, that is 8.0e10 against a unit right hand side: the
//! *exact* solution rounded to f32 leaves a residual of 2.0e2, and the device's
//! own iteration breaks down long before reaching it, because the conjugate
//! gradient there needs 6075 double precision iterations and swings `p^T K p`
//! over twelve decades. No arrangement of single precision solves that system.
//! A solve that runs into this is not failed: it is finished on the CPU by
//! [`crate::fea::BoundSolver::solve`], loudly, with the best iterate the device
//! reached as its warm start. See [`SinglePrecisionLimit`].
//!
//! # Reproducibility
//!
//! The operator is applied node-centric with no atomics and a fixed gather
//! order, and the reductions have a fixed shape, so a given device and driver
//! return the same bits for the same input. Nothing guarantees that across
//! machines, drivers or vendors: f32 rounding of a different fused-multiply-add
//! schedule is enough to move the last bits, and - as
//! [`constants::GPU_CARRY_GUARD`] records - what a shader compiler is willing to
//! do to a compensated sum differs between them too. The CPU backend stays the
//! default and the reference for exactly that reason.

use std::sync::mpsc;

use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::constants::{
    self, DOF_PER_ELEMENT, GPU_DISPATCH_WORKGROUPS, GPU_SCALAR_BREAKDOWN, GPU_SCALAR_COUNT,
    GPU_SCALAR_RESIDUAL_SQ,
};
use crate::device::{Scopes, refused_with};
use crate::fea::{CancelProbe, CgSolution, Solve, SolveLimits, StiffnessOperator, solver};
use crate::grid::Grid;

/// The shader driving every device side kernel.
const SHADER: &str = include_str!("pcg.wgsl");

/// Bits in one word of the constraint mask, mirrored by the shader.
const BITS_PER_MASK_WORD: usize = 32;

/// Bytes of the shader's scalar buffer.
const SCALAR_BYTES: u64 = (GPU_SCALAR_COUNT * size_of::<f32>()) as u64;

/// Uniform block shared by every kernel. Mirrors `Params` in `pcg.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    n_nodes: u32,
    n_dof: u32,
    n_cells: u32,
    n_partials: u32,
    /// Always [`constants::GPU_CARRY_GUARD`]; see `opaque` in `pcg.wgsl`.
    carry_guard: f32,
}

/// The compiled kernels, in the order one conjugate gradient iteration runs
/// them.
struct Pipelines {
    init: wgpu::ComputePipeline,
    matvec: wgpu::ComputePipeline,
    finish_pap: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    finish_update: wgpu::ComputePipeline,
    update_p: wgpu::ComputePipeline,
}

/// Device buffers that outlive a single solve.
struct Buffers {
    ke0: wgpu::Buffer,
    moduli: wgpu::Buffer,
    minv: wgpu::Buffer,
    solution: wgpu::Buffer,
    solution_low: wgpu::Buffer,
    residual: wgpu::Buffer,
    scalars: wgpu::Buffer,
    scalar_readback: wgpu::Buffer,
    vector_readback: wgpu::Buffer,
}

/// A solve the device could not finish, because of the system it was given
/// rather than because of anything wrong with the device or this code.
///
/// It is a type rather than a message so that the one caller allowed to act on
/// it - [`crate::fea::BoundSolver::solve`], which finishes such a solve on the
/// CPU - can tell it apart from a device that has gone away, a buffer that
/// could not be mapped or a design that does not fit. Those are still errors:
/// they say the machine cannot do the work, not that single precision cannot
/// resolve this particular stiffness matrix.
#[derive(Debug)]
pub struct SinglePrecisionLimit {
    reason: String,
}

impl std::fmt::Display for SinglePrecisionLimit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl std::error::Error for SinglePrecisionLimit {}

/// Wrap a reason as the outcome the CPU fallback watches for.
pub(crate) fn defeated(reason: String) -> anyhow::Error {
    anyhow::Error::new(SinglePrecisionLimit { reason })
}

/// What one device side solve achieved.
struct InnerOutcome {
    iterations: usize,
    /// Relative residual, against a right hand side of unit length.
    residual: f64,
    /// Set when the caller called the solve off between two readback batches
    /// rather than the batch loop ending on its own terms.
    cancelled: bool,
}

/// A conjugate gradient solver bound to one grid, running on a compute device.
///
/// The buffers are allocated once and reused for every load case and every
/// optimization iteration; only the density derived data (`bind`) and the right
/// hand side of a solve cross the bus.
pub struct GpuSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    description: String,
    pipelines: Pipelines,
    problem_group: wgpu::BindGroup,
    vector_group: wgpu::BindGroup,
    buffers: Buffers,
    n_dof: usize,
    n_cells: usize,
    /// Host staging for the narrowing and widening conversions.
    narrowed: Vec<f32>,
    /// Host staging for the compensation limb of a correction, read back
    /// alongside [`GpuSolver::narrowed`] and added into it in f64.
    correction_low: Vec<f32>,
    /// Host scratch for the f64 residual of the refinement loop.
    residual: Vec<f64>,
    /// The solution as it stood before the current refinement pass folded its
    /// correction, so a pass that turns out to have made things worse can be
    /// put back.
    previous: Vec<f64>,
    /// Set once `bind` has uploaded a design.
    bound: bool,
}

impl std::fmt::Debug for GpuSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuSolver")
            .field("description", &self.description)
            .field("n_dof", &self.n_dof)
            .field("n_cells", &self.n_cells)
            .finish_non_exhaustive()
    }
}

/// Narrow to f32 without ever producing an infinity: a value outside the f32
/// range is clamped, so a degenerate modulus or preconditioner entry degrades
/// the convergence rate instead of poisoning the whole vector with NaN.
fn narrow(value: f64) -> f32 {
    if value.is_nan() {
        return 0.0;
    }
    value.clamp(-(f32::MAX as f64), f32::MAX as f64) as f32
}

impl GpuSolver {
    /// Open a compute device and allocate the buffers for `grid`.
    ///
    /// `fixed` is the constraint mask; it is static for the life of a problem
    /// and is packed into a bitset on the way to the device.
    pub fn new(grid: &Grid, fixed: &[bool]) -> Result<GpuSolver> {
        let n_dof = grid.n_dof();
        let n_cells = grid.n_cells();
        assert_eq!(
            fixed.len(),
            n_dof,
            "constraint mask length must match the system size"
        );

        // From the environment, so `WGPU_BACKEND` really does pin the graphics
        // API the solve runs on. That matters here in a way it does not for the
        // viewer: results are only reproducible within one API and driver, so
        // being able to say which one is part of the reproducibility contract.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        // Device loss (a driver reset, an unplugged eGPU) is not recovered from:
        // it surfaces as an error from the readback below, which fails the solve
        // with a message rather than returning a wrong answer.
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|error| {
            anyhow!(
                "no GPU adapter is available for the compute solver ({error}); set [solver] \
                 backend = \"cpu\" to solve on the CPU"
            )
        })?;
        let info = adapter.get_info();
        let description = format!(
            "{} ({:?}, {:?}) driver {} {}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );
        let limits = adapter.limits();
        check_limits(&limits, &description, n_dof, n_cells)?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(constants::PROGRAM_NAME),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|error| {
            anyhow!(
                "the GPU adapter \"{}\" refused a compute device ({error}); set [solver] backend \
                 = \"cpu\" to solve on the CPU",
                info.name
            )
        })?;

        // Everything below is asked of the device, and a device with no memory
        // left refuses - which wgpu answers with a panic unless somebody is
        // reading. The vectors here are one f32 per degree of freedom each and
        // there are eight of them, so this is the allocation another program's
        // hold on the card is felt in first. Read at the end, where it becomes
        // the error this constructor has always been able to return and
        // [`crate::fea::backend::LinearSolver::new`] falls back from.
        let scopes = Scopes::open(&device);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("growforge_pcg"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("growforge_pcg_params"),
            contents: bytemuck::bytes_of(&Params {
                nx: grid.nx as u32,
                ny: grid.ny as u32,
                nz: grid.nz as u32,
                n_nodes: grid.n_nodes() as u32,
                n_dof: n_dof as u32,
                n_cells: n_cells as u32,
                n_partials: GPU_DISPATCH_WORKGROUPS,
                carry_guard: constants::GPU_CARRY_GUARD,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let free_bits = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("growforge_pcg_free"),
            contents: bytemuck::cast_slice(&pack_free_mask(fixed)),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let storage = |label: &str, elements: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (elements * size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | usage,
                mapped_at_creation: false,
            })
        };
        let ke0 = storage(
            "growforge_pcg_ke0",
            DOF_PER_ELEMENT * DOF_PER_ELEMENT,
            wgpu::BufferUsages::COPY_DST,
        );
        let moduli = storage(
            "growforge_pcg_moduli",
            n_cells,
            wgpu::BufferUsages::COPY_DST,
        );
        let minv = storage("growforge_pcg_minv", n_dof, wgpu::BufferUsages::COPY_DST);
        let solution = storage(
            "growforge_pcg_x",
            n_dof,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let solution_low = storage("growforge_pcg_x_low", n_dof, wgpu::BufferUsages::COPY_SRC);
        let residual = storage("growforge_pcg_r", n_dof, wgpu::BufferUsages::COPY_DST);
        let preconditioned = storage("growforge_pcg_z", n_dof, wgpu::BufferUsages::empty());
        let direction = storage("growforge_pcg_p", n_dof, wgpu::BufferUsages::empty());
        let applied = storage("growforge_pcg_ap", n_dof, wgpu::BufferUsages::empty());
        let partials = storage(
            "growforge_pcg_partials",
            2 * GPU_DISPATCH_WORKGROUPS as usize,
            wgpu::BufferUsages::empty(),
        );
        let scalars = storage(
            "growforge_pcg_scalars",
            GPU_SCALAR_COUNT,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let scalar_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("growforge_pcg_scalar_readback"),
            size: SCALAR_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vector_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("growforge_pcg_vector_readback"),
            size: (n_dof * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let problem_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("growforge_pcg_problem"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
            ],
        });
        let vector_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("growforge_pcg_vectors"),
            entries: &(0..8).map(|b| storage_entry(b, false)).collect::<Vec<_>>(),
        });
        let problem_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("growforge_pcg_problem"),
            layout: &problem_layout,
            entries: &[
                binding(0, &params),
                binding(1, &ke0),
                binding(2, &moduli),
                binding(3, &minv),
                binding(4, &free_bits),
            ],
        });
        let vector_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("growforge_pcg_vectors"),
            layout: &vector_layout,
            entries: &[
                binding(0, &solution),
                binding(1, &residual),
                binding(2, &preconditioned),
                binding(3, &direction),
                binding(4, &applied),
                binding(5, &partials),
                binding(6, &scalars),
                binding(7, &solution_low),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("growforge_pcg"),
            bind_group_layouts: &[Some(&problem_layout), Some(&vector_layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            init: pipeline("init"),
            matvec: pipeline("matvec"),
            finish_pap: pipeline("finish_pap"),
            update: pipeline("update"),
            finish_update: pipeline("finish_update"),
            update_p: pipeline("update_p"),
        };
        if let Some(refusal) = scopes.close() {
            return Err(refused_with(
                &format!("give the compute solver on \"{}\" its buffers", info.name),
                &refusal,
                "set [solver] backend = \"cpu\" to solve on the CPU",
            ));
        }

        Ok(GpuSolver {
            device,
            queue,
            description,
            pipelines,
            problem_group,
            vector_group,
            buffers: Buffers {
                ke0,
                moduli,
                minv,
                solution,
                solution_low,
                residual,
                scalars,
                scalar_readback,
                vector_readback,
            },
            n_dof,
            n_cells,
            narrowed: vec![0.0; n_dof],
            correction_low: vec![0.0; n_dof],
            residual: vec![0.0; n_dof],
            previous: vec![0.0; n_dof],
            bound: false,
        })
    }

    /// Human readable description of the device the solves run on.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Upload the design dependent half of the operator.
    ///
    /// Call once per design, before the load cases that share it. Only the unit
    /// stiffness matrix, the per-element moduli and the Jacobi preconditioner
    /// cross the bus; the grid, the constraint mask and every Krylov vector stay
    /// where they were allocated.
    pub fn bind(
        &mut self,
        operator: &StiffnessOperator,
        diagonal: &[f64],
        fixed: &[bool],
    ) -> Result<()> {
        let moduli = operator.moduli();
        if moduli.len() != self.n_cells || diagonal.len() != self.n_dof {
            bail!(
                "the GPU solver was built for a {} cell, {} degree of freedom grid and was handed \
                 {} cells and {} degrees of freedom",
                self.n_cells,
                self.n_dof,
                moduli.len(),
                diagonal.len()
            );
        }

        // Everything the design costs the device, under one reading: a card
        // with no memory left refuses a write to a buffer it has invalidated,
        // and the alternative to reading that is the process dying of it.
        let scopes = Scopes::open(&self.device);
        let mut flat = vec![0.0f32; DOF_PER_ELEMENT * DOF_PER_ELEMENT];
        for (row, values) in operator.ke0().iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                flat[row * DOF_PER_ELEMENT + column] = narrow(*value);
            }
        }
        self.queue
            .write_buffer(&self.buffers.ke0, 0, bytemuck::cast_slice(&flat));

        let narrowed: Vec<f32> = moduli.par_iter().map(|m| narrow(*m)).collect();
        self.queue
            .write_buffer(&self.buffers.moduli, 0, bytemuck::cast_slice(&narrowed));

        // The same guarded reciprocal the CPU backend uses. Constrained rows get
        // the floor rather than zero, exactly as on the CPU; their residual is
        // zero by projection, so the value never reaches the iteration.
        let inverse: Vec<f32> = diagonal
            .par_iter()
            .zip(fixed.par_iter())
            .map(|(&d, &is_fixed)| {
                narrow(if is_fixed || d <= constants::JACOBI_DIAGONAL_EPS {
                    1.0 / constants::JACOBI_DIAGONAL_FLOOR
                } else {
                    1.0 / d
                })
            })
            .collect();
        self.queue
            .write_buffer(&self.buffers.minv, 0, bytemuck::cast_slice(&inverse));
        if let Some(refusal) = scopes.close() {
            // Half a design is on the device, so nothing may solve against it
            // until a later bind replaces it: the moduli left there are another
            // design's, and a solve reading them would answer the wrong
            // question rather than fail.
            self.bound = false;
            return Err(refused_with(
                "give the compute solver this design",
                &refusal,
                "set [solver] backend = \"cpu\" to solve on the CPU",
            ));
        }
        self.bound = true;
        Ok(())
    }

    /// Solve `K u = f` on the free subspace to `limits.tolerance`, refining in
    /// f64.
    ///
    /// `x` carries the initial guess in and the solution out, so the warm starts
    /// the optimization loop relies on work exactly as they do on the CPU.
    ///
    /// `limits.cancel` is asked at both boundaries of the refinement structure:
    /// once per pass, and once per device-side readback batch. Both are already
    /// pipeline stalls of their own - a batch ends by mapping a buffer and
    /// waiting for the queue - so the probe costs nothing measurable there, and
    /// the latency of a stop is one batch of
    /// [`constants::GPU_RESIDUAL_CHECK_INTERVAL`] device iterations.
    pub fn solve(
        &mut self,
        operator: &StiffnessOperator,
        fixed: &[bool],
        b: &[f64],
        x: &mut [f64],
        limits: SolveLimits<'_>,
    ) -> Result<Solve> {
        // One reading for the whole solve: the encoders, the submissions and
        // the readback buffers of every pass are asked for inside it, and a
        // device that refuses any of them is why this solve ended - whatever
        // the refinement below made of what came back afterwards.
        let scopes = Scopes::open(&self.device);
        let solved = self.refine(operator, fixed, b, x, limits);
        match scopes.close() {
            None => solved,
            Some(refusal) => Err(refused_with(
                "run a solve on the compute device",
                &refusal,
                "set [solver] backend = \"cpu\" to solve on the CPU",
            )),
        }
    }

    /// The mixed precision refinement itself, inside [`GpuSolver::solve`]'s
    /// reading of what the device refused.
    ///
    /// Split out for that reading alone: the loop returns from a dozen places,
    /// and a scope has to be closed on every one of them.
    fn refine(
        &mut self,
        operator: &StiffnessOperator,
        fixed: &[bool],
        b: &[f64],
        x: &mut [f64],
        limits: SolveLimits<'_>,
    ) -> Result<Solve> {
        assert!(self.bound, "bind must be called before a GPU solve");
        assert_eq!(b.len(), self.n_dof, "right hand side length");
        assert_eq!(x.len(), self.n_dof, "solution length");
        let SolveLimits {
            tolerance,
            max_iterations,
            cancel,
        } = limits;

        let mut rhs = b.to_vec();
        solver::project(&mut rhs, fixed);
        let rhs_norm = solver::norm(&rhs);
        if rhs_norm == 0.0 {
            x.fill(0.0);
            return Ok(Solve::Converged(CgSolution {
                iterations: 0,
                relative_residual: 0.0,
            }));
        }
        solver::project(x, fixed);

        let mut residual_norm = self.refresh_residual(operator, fixed, &rhs, x);
        let mut relative = residual_norm / rhs_norm;
        if relative < tolerance {
            return Ok(Solve::Converged(CgSolution {
                iterations: 0,
                relative_residual: relative,
            }));
        }

        let mut iterations = 0;
        let total_budget = max_iterations.saturating_mul(constants::GPU_ITERATION_BUDGET_FACTOR);
        for _ in 0..constants::GPU_REFINEMENT_MAX_PASSES {
            if cancel.cancelled() {
                return Ok(Solve::Cancelled);
            }
            let budget = total_budget
                .saturating_sub(iterations)
                .min(constants::GPU_INNER_MAX_ITERATIONS);
            if budget == 0 {
                break;
            }

            // Unit length in, so the device solve works in the middle of the f32
            // exponent range whatever decade the outer residual has reached.
            let scale = 1.0 / residual_norm;
            let source = &self.residual;
            self.narrowed
                .par_iter_mut()
                .zip(source.par_iter())
                .for_each(|(slot, &value)| *slot = narrow(value * scale));
            self.queue.write_buffer(
                &self.buffers.residual,
                0,
                bytemuck::cast_slice(&self.narrowed),
            );

            // Ask the device for exactly the accuracy the outer loop still
            // needs, and no more. The right hand side went in at unit length, so
            // the factor the outer residual has to be multiplied by is also the
            // inner residual that would achieve it; a decade of margin on top
            // keeps a pass that lands on its target from needing another one.
            // The floor is what single precision can be expected to deliver at
            // all, so a request tighter than that is never made.
            let needed =
                tolerance * rhs_norm / residual_norm * constants::GPU_REFINEMENT_TARGET_MARGIN;
            let inner_tolerance = (needed as f32).max(constants::GPU_INNER_TOLERANCE);
            let inner = self.inner_solve(inner_tolerance, budget, cancel)?;
            if inner.cancelled {
                // The correction still on the device is part way through a
                // Krylov recurrence, so it is dropped rather than folded into
                // `x`: nobody wants this answer, and half of one is not one.
                return Ok(Solve::Cancelled);
            }
            iterations += inner.iterations;
            self.download_correction()?;
            // The fold is a trial until the f64 residual says it was one. A
            // correction the device could not compute accurately enough moves
            // the solution *away* from the answer, and a solve that keeps such
            // a step hands its caller - and the fallback below - something
            // worse than what it already had. Measured before the compensated
            // accumulator existed: a first pass on a 192k cell cantilever at
            // mass fraction 0.10 left a relative residual of 1.388, against the
            // 1.0 it started from.
            self.previous.copy_from_slice(x);
            let (high, low) = (&self.narrowed, &self.correction_low);
            x.par_iter_mut()
                .zip(high.par_iter())
                .zip(low.par_iter())
                .for_each(|((slot, &high), &low)| {
                    *slot += residual_norm * (f64::from(high) + f64::from(low));
                });

            let updated = self.refresh_residual(operator, fixed, &rhs, x);
            if !updated.is_finite() || updated >= residual_norm {
                x.copy_from_slice(&self.previous);
                return Err(defeated(format!(
                    "the correction the GPU conjugate gradient produced after {} device \
                     iterations left a relative residual of {:.3e}, no better than the {:.3e} it \
                     started the pass from (inner residual {:.3e})",
                    inner.iterations,
                    updated / rhs_norm,
                    relative,
                    inner.residual,
                )));
            }
            relative = updated / rhs_norm;
            if relative < tolerance {
                return Ok(Solve::Converged(CgSolution {
                    iterations,
                    relative_residual: relative,
                }));
            }
            // A pass that did not earn its keep has hit the single precision
            // floor - but only if it stopped of its own accord. One the
            // iteration budget cut short has proved nothing, and falls through
            // to the cap message below.
            let stalled = updated >= constants::GPU_REFINEMENT_MIN_GAIN * residual_norm;
            if stalled && inner.iterations < budget {
                return Err(defeated(format!(
                    "the GPU conjugate gradient stopped improving at a relative residual of \
                     {relative:.3e} against a target of {tolerance:.3e} (inner residual {:.3e} \
                     after {} device iterations)",
                    inner.residual, inner.iterations,
                )));
            }
            residual_norm = updated;
        }

        Err(defeated(format!(
            "the GPU conjugate gradient did not converge in {iterations} device iterations \
             (relative residual {relative:.3e}, target {tolerance:.3e})"
        )))
    }

    /// Recompute `r = b - K x` in f64 and return `||r||`.
    fn refresh_residual(
        &mut self,
        operator: &StiffnessOperator,
        fixed: &[bool],
        rhs: &[f64],
        x: &[f64],
    ) -> f64 {
        operator.apply(x, &mut self.residual);
        solver::project(&mut self.residual, fixed);
        self.residual
            .par_iter_mut()
            .zip(rhs.par_iter())
            .for_each(|(slot, &value)| *slot = value - *slot);
        solver::norm(&self.residual)
    }

    /// Run the device side conjugate gradient on the right hand side already in
    /// the residual buffer, leaving the correction in the solution buffer.
    ///
    /// The right hand side has unit length, so `tolerance` is both the relative
    /// and the absolute residual to stop at.
    fn inner_solve(
        &mut self,
        tolerance: f32,
        max_iterations: usize,
        cancel: CancelProbe<'_>,
    ) -> Result<InnerOutcome> {
        self.queue.write_buffer(
            &self.buffers.scalars,
            0,
            &[0u8; GPU_SCALAR_COUNT * size_of::<f32>()],
        );

        let mut encoder = self.encoder();
        {
            let mut pass = self.pass(&mut encoder);
            dispatch(&mut pass, &self.pipelines.init, GPU_DISPATCH_WORKGROUPS);
            dispatch(&mut pass, &self.pipelines.finish_update, 1);
            dispatch(&mut pass, &self.pipelines.update_p, GPU_DISPATCH_WORKGROUPS);
        }
        self.queue.submit([encoder.finish()]);

        let tolerance = f64::from(tolerance);
        let mut iterations = 0;
        let mut residual = 1.0;
        let mut best = f64::INFINITY;
        let mut stalled = 0;
        while iterations < max_iterations {
            // Before a batch is recorded rather than after it has run: a batch
            // is the granularity of a stop, and asking first is what keeps that
            // granularity at one batch instead of two.
            if cancel.cancelled() {
                return Ok(InnerOutcome {
                    iterations,
                    residual,
                    cancelled: true,
                });
            }
            let batch = constants::GPU_RESIDUAL_CHECK_INTERVAL.min(max_iterations - iterations);
            let mut encoder = self.encoder();
            {
                let mut pass = self.pass(&mut encoder);
                for _ in 0..batch {
                    dispatch(&mut pass, &self.pipelines.matvec, GPU_DISPATCH_WORKGROUPS);
                    dispatch(&mut pass, &self.pipelines.finish_pap, 1);
                    dispatch(&mut pass, &self.pipelines.update, GPU_DISPATCH_WORKGROUPS);
                    dispatch(&mut pass, &self.pipelines.finish_update, 1);
                    dispatch(&mut pass, &self.pipelines.update_p, GPU_DISPATCH_WORKGROUPS);
                }
            }
            encoder.copy_buffer_to_buffer(
                &self.buffers.scalars,
                0,
                &self.buffers.scalar_readback,
                0,
                SCALAR_BYTES,
            );
            self.queue.submit([encoder.finish()]);
            iterations += batch;

            let scalars = self.read_scalars()?;
            residual = f64::from(scalars[GPU_SCALAR_RESIDUAL_SQ]).max(0.0).sqrt();
            if !residual.is_finite() {
                return Err(defeated(format!(
                    "the GPU conjugate gradient diverged after {iterations} device iterations \
                     (residual {residual})"
                )));
            }
            // Convergence is tested before breakdown, and deliberately. A batch
            // of iterations runs to completion before anything is read back, so
            // a system small enough to converge inside one batch keeps iterating
            // on a residual that is already pure rounding noise, and `p^T K p`
            // duly collapses to zero. That is the solve finishing, not the
            // operator losing definiteness. The shader freezes the iteration on
            // breakdown (alpha and beta go to zero), so the residual read back
            // here is the one at the moment it happened either way.
            if residual < tolerance {
                break;
            }
            if scalars[GPU_SCALAR_BREAKDOWN] != 0.0 {
                return Err(defeated(format!(
                    "GPU conjugate gradient breakdown after {iterations} device iterations at a \
                     relative residual of {residual:.3e}: the operator is not positive definite \
                     on the free subspace in single precision"
                )));
            }
            // The 2-norm residual of a conjugate gradient is not monotone; only
            // the energy norm of the error is. Progress is therefore measured
            // against the best reading so far rather than the last one, and a
            // solve whose residual currently sits above where it started is
            // inside its plateau, not at its floor, so it is not eligible to be
            // called stalled at all.
            let improved = residual < best * constants::GPU_INNER_STALL_GAIN;
            best = best.min(residual);
            if improved || residual > constants::GPU_INNER_ENGAGED_RESIDUAL {
                stalled = 0;
            } else {
                stalled += 1;
                if stalled >= constants::GPU_INNER_STALL_INTERVALS {
                    break;
                }
            }
        }
        Ok(InnerOutcome {
            iterations,
            residual,
            cancelled: false,
        })
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("growforge_pcg"),
            })
    }

    fn pass<'a>(&'a self, encoder: &'a mut wgpu::CommandEncoder) -> wgpu::ComputePass<'a> {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("growforge_pcg"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.problem_group, &[]);
        pass.set_bind_group(1, &self.vector_group, &[]);
        pass
    }

    /// Read the shader's scalar block back.
    fn read_scalars(&self) -> Result<[f32; GPU_SCALAR_COUNT]> {
        let mut values = [0.0f32; GPU_SCALAR_COUNT];
        self.map(&self.buffers.scalar_readback, |bytes| {
            values.copy_from_slice(bytemuck::cast_slice(bytes));
        })?;
        Ok(values)
    }

    /// Read the two limbs of the device's solution accumulator back into
    /// [`GpuSolver::narrowed`] and [`GpuSolver::correction_low`].
    ///
    /// Both limbs, because the whole point of carrying the second one is that
    /// the correction is worth more than an f32 can say; folding only the high
    /// limb would throw the accuracy away on the way home.
    fn download_correction(&mut self) -> Result<()> {
        self.download(self.buffers.solution.clone())?;
        // `download` always lands in `narrowed`, so the high limb is parked in
        // the low slot while the low one is fetched, and the two are put back.
        std::mem::swap(&mut self.narrowed, &mut self.correction_low);
        let result = self.download(self.buffers.solution_low.clone());
        std::mem::swap(&mut self.narrowed, &mut self.correction_low);
        result
    }

    /// Copy a device vector into [`GpuSolver::narrowed`].
    fn download(&mut self, source: wgpu::Buffer) -> Result<()> {
        let mut encoder = self.encoder();
        encoder.copy_buffer_to_buffer(
            &source,
            0,
            &self.buffers.vector_readback,
            0,
            (self.n_dof * size_of::<f32>()) as u64,
        );
        self.queue.submit([encoder.finish()]);
        let mut narrowed = std::mem::take(&mut self.narrowed);
        let result = self.map(&self.buffers.vector_readback, |bytes| {
            narrowed.copy_from_slice(bytemuck::cast_slice(bytes));
        });
        self.narrowed = narrowed;
        result
    }

    /// Wait for the queue to drain, map `buffer` and hand its bytes to `read`.
    fn map(&self, buffer: &wgpu::Buffer, read: impl FnOnce(&[u8])) -> Result<()> {
        let (sender, receiver) = mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |done| {
                let _ = sender.send(done);
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| anyhow!("waiting for the GPU solver to finish: {error}"))?;
        receiver
            .recv()
            .context("the GPU solver's readback never completed")?
            .context("mapping a GPU solver readback buffer")?;
        read(&buffer.slice(..).get_mapped_range());
        buffer.unmap();
        Ok(())
    }
}

/// Pack a constraint mask into the bitset the shader reads: one bit per degree
/// of freedom, set when the degree of freedom is *free*.
fn pack_free_mask(fixed: &[bool]) -> Vec<u32> {
    let mut words = vec![0u32; fixed.len().div_ceil(BITS_PER_MASK_WORD)];
    for (index, &is_fixed) in fixed.iter().enumerate() {
        if !is_fixed {
            words[index / BITS_PER_MASK_WORD] |= 1 << (index % BITS_PER_MASK_WORD);
        }
    }
    words
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn binding(index: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
    }
}

fn dispatch(pass: &mut wgpu::ComputePass<'_>, pipeline: &wgpu::ComputePipeline, groups: u32) {
    pass.set_pipeline(pipeline);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Refuse an adapter that cannot hold the problem, with a message that says
/// which limit it is rather than letting the driver fail the allocation.
fn check_limits(
    limits: &wgpu::Limits,
    description: &str,
    n_dof: usize,
    n_cells: usize,
) -> Result<()> {
    if limits.max_storage_buffers_per_shader_stage < constants::GPU_STORAGE_BUFFERS {
        bail!(
            "the GPU adapter \"{description}\" allows {} storage buffers per shader stage and the \
             solver needs {}; set [solver] backend = \"cpu\"",
            limits.max_storage_buffers_per_shader_stage,
            constants::GPU_STORAGE_BUFFERS
        );
    }
    if limits.max_compute_workgroups_per_dimension < GPU_DISPATCH_WORKGROUPS {
        bail!(
            "the GPU adapter \"{description}\" allows {} workgroups per dispatch and the solver \
             needs {}; set [solver] backend = \"cpu\"",
            limits.max_compute_workgroups_per_dimension,
            GPU_DISPATCH_WORKGROUPS
        );
    }
    if limits.max_compute_invocations_per_workgroup < constants::GPU_WORKGROUP_SIZE {
        bail!(
            "the GPU adapter \"{description}\" allows {} invocations per workgroup and the solver \
             needs {}; set [solver] backend = \"cpu\"",
            limits.max_compute_invocations_per_workgroup,
            constants::GPU_WORKGROUP_SIZE
        );
    }
    let largest = (n_dof.max(n_cells) * size_of::<f32>()) as u64;
    let allowed = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    if largest > allowed {
        bail!(
            "this problem needs a {:.1} MiB GPU buffer and the adapter \"{description}\" allows \
             {:.1} MiB; solve it on a coarser grid, or set [solver] backend = \"cpu\"",
            largest as f64 / constants::BYTES_PER_MIB,
            allowed as f64 / constants::BYTES_PER_MIB
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every value the shader mirrors from `constants`, as the WGSL declaration
    /// it must appear as. A drift here is silent corruption at run time, so it
    /// is a compile-time-shaped test instead.
    #[test]
    fn the_shader_mirrors_the_configured_constants() {
        let declarations = [
            format!(
                "const WORKGROUP_SIZE: u32 = {}u;",
                constants::GPU_WORKGROUP_SIZE
            ),
            format!("const SLOT_RZ: u32 = {}u;", constants::GPU_SCALAR_RZ),
            format!("const SLOT_PAP: u32 = {}u;", constants::GPU_SCALAR_PAP),
            format!("const SLOT_ALPHA: u32 = {}u;", constants::GPU_SCALAR_ALPHA),
            format!("const SLOT_BETA: u32 = {}u;", constants::GPU_SCALAR_BETA),
            format!(
                "const SLOT_RESIDUAL_SQ: u32 = {}u;",
                constants::GPU_SCALAR_RESIDUAL_SQ
            ),
            format!(
                "const SLOT_BREAKDOWN: u32 = {}u;",
                constants::GPU_SCALAR_BREAKDOWN
            ),
            format!(
                "const NODES_PER_ELEMENT: u32 = {}u;",
                constants::NODES_PER_ELEMENT
            ),
            format!("const DOF_PER_NODE: u32 = {}u;", constants::DOF_PER_NODE),
            format!("const DOF_PER_ELEMENT: u32 = {DOF_PER_ELEMENT}u;"),
            format!("const BITS_PER_WORD: u32 = {BITS_PER_MASK_WORD}u;"),
        ];
        // The compensation guard is not a `const` - it comes down in the uniform
        // block, which is the whole point of it - so it is checked as the two
        // lines that have to exist for any carry in the shader to survive a
        // reassociating compiler. Deleting either is silent: every compensated
        // sum goes on compiling and quietly stops compensating.
        assert!(
            SHADER.contains("carry_guard: f32,")
                && SHADER.contains("let rounded = total * params.carry_guard;"),
            "pcg.wgsl no longer routes the solution accumulator through the carry guard"
        );
        assert_eq!(constants::GPU_CARRY_GUARD, 1.0);
        for declaration in declarations {
            assert!(
                SHADER.contains(&declaration),
                "pcg.wgsl no longer declares `{declaration}`"
            );
        }
        // The scalar slots have to cover the buffer exactly.
        assert_eq!(
            constants::GPU_SCALAR_COUNT,
            constants::GPU_SCALAR_BREAKDOWN + 1
        );
    }

    #[test]
    fn the_free_mask_packs_one_bit_per_degree_of_freedom() {
        let mut fixed = vec![false; 70];
        fixed[0] = true;
        fixed[31] = true;
        fixed[32] = true;
        fixed[69] = true;
        let words = pack_free_mask(&fixed);
        assert_eq!(words.len(), 3);
        for (index, &is_fixed) in fixed.iter().enumerate() {
            let bit = (words[index / BITS_PER_MASK_WORD] >> (index % BITS_PER_MASK_WORD)) & 1;
            assert_eq!(bit == 1, !is_fixed, "degree of freedom {index}");
        }
        // Nothing outside the mask may read as free.
        assert_eq!(words[2] >> (70 - 2 * BITS_PER_MASK_WORD), 0);
    }

    #[test]
    fn narrowing_never_produces_an_infinity_or_a_nan() {
        assert_eq!(narrow(0.0), 0.0);
        assert_eq!(narrow(1.5), 1.5);
        assert_eq!(narrow(f64::NAN), 0.0);
        assert_eq!(narrow(1e300), f32::MAX);
        assert_eq!(narrow(-1e300), -f32::MAX);
        assert!(narrow(f64::INFINITY).is_finite());
        assert!(narrow(f64::NEG_INFINITY).is_finite());
    }
}
