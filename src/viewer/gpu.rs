//! wgpu device setup and the two-pass renderer: the shaded scene first, the
//! egui overlay on top.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::constants;
use crate::viewer::camera::{self, Mat4};
use crate::viewer::scene::{LAYERS, Layer, LayerMesh, Scene, Vertex};

/// Depth format of the 3D pass. Not a tunable: it is the one format every
/// backend wgpu supports offers, and the scene needs no stencil.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_projection: [[f32; 4]; 4],
    light: [f32; 4],
    shading: [f32; 4],
}

/// One layer's vertex buffer.
#[derive(Debug)]
struct LayerBuffer {
    buffer: wgpu::Buffer,
    vertices: u32,
}

/// What acquiring the next surface frame reported.
///
/// The renderer's own reading of `wgpu::CurrentSurfaceTexture`, so the recovery
/// policy below is decided - and tested - without a device to acquire from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceStatus {
    /// A frame came back and can be drawn into.
    Acquired,
    /// No frame this time, and nothing is wrong with the surface: the
    /// compositor was busy, or the window is not visible.
    Skipped,
    /// The surface no longer matches the window and has to be reconfigured.
    Stale,
    /// The device refused to hand a frame over at all.
    Rejected,
}

impl SurfaceStatus {
    /// Read what wgpu answered.
    fn of(acquired: &wgpu::CurrentSurfaceTexture) -> SurfaceStatus {
        match acquired {
            wgpu::CurrentSurfaceTexture::Success(_)
            | wgpu::CurrentSurfaceTexture::Suboptimal(_) => SurfaceStatus::Acquired,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                SurfaceStatus::Skipped
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                SurfaceStatus::Stale
            }
            wgpu::CurrentSurfaceTexture::Validation => SurfaceStatus::Rejected,
        }
    }
}

/// What the renderer does about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceAction {
    /// Draw into the frame that came back.
    Draw,
    /// Skip this frame and leave the surface alone.
    Skip,
    /// Reconfigure the surface and skip this frame.
    Reconfigure,
    /// The same, after a rejection: `streak` counts the consecutive rejected
    /// frames including this one, so the first of a streak can say so once.
    Retry { streak: u32 },
    /// Give up on the device: `streak` frames in a row were rejected and it has
    /// not come back.
    GiveUp { streak: u32 },
}

/// How the drawing surface is behaving, frame by frame.
///
/// A rejected frame used to end the viewer where it stood. It is instead a
/// device that may be resetting: the surface is reconfigured and the frame
/// skipped, exactly as a stale surface is, and only a rejection that goes on
/// for [`constants::VIEW_SURFACE_REJECTION_LIMIT`] consecutive frames is a
/// device that is not coming back. A frame that *is* acquired is what clears
/// the streak - the recovery is proven by drawing, not by any other answer the
/// surface gives - so a skipped or stale frame neither counts towards the limit
/// nor forgives what came before it.
#[derive(Debug, Clone, Copy, Default)]
pub struct SurfaceHealth {
    rejections: u32,
}

impl SurfaceHealth {
    /// Record what one acquisition reported and say what to do about it.
    pub fn observe(&mut self, status: SurfaceStatus) -> SurfaceAction {
        match status {
            SurfaceStatus::Acquired => {
                self.rejections = 0;
                SurfaceAction::Draw
            }
            SurfaceStatus::Skipped => SurfaceAction::Skip,
            SurfaceStatus::Stale => SurfaceAction::Reconfigure,
            SurfaceStatus::Rejected => {
                self.rejections = self.rejections.saturating_add(1);
                if self.rejections >= constants::VIEW_SURFACE_REJECTION_LIMIT {
                    SurfaceAction::GiveUp {
                        streak: self.rejections,
                    }
                } else {
                    SurfaceAction::Retry {
                        streak: self.rejections,
                    }
                }
            }
        }
    }
}

/// Everything egui needs painted in a frame.
pub struct EguiFrame {
    /// Tessellated egui primitives.
    pub primitives: Vec<egui::ClippedPrimitive>,
    /// Textures egui created or freed this frame.
    pub textures_delta: egui::TexturesDelta,
    /// Points per physical pixel.
    pub pixels_per_point: f32,
}

/// The GPU side of the viewer.
pub struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    opaque_pipeline: wgpu::RenderPipeline,
    translucent_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    buffers: Vec<Option<LayerBuffer>>,
    egui_renderer: egui_wgpu::Renderer,
    /// Consecutive rejected frames, which is what tells a device resetting
    /// apart from one that is gone.
    health: SurfaceHealth,
    /// Human readable description of the adapter the viewer picked.
    pub adapter_description: String,
}

impl Gpu {
    /// Create a surface, device and pipelines for `window`.
    pub fn new(window: Arc<Window>) -> Result<Gpu> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(Arc::clone(&window))
            .context("creating a drawing surface for the viewer window")?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .map_err(|error| {
            anyhow!(
                "no GPU adapter can draw to this window ({error}); run the same command without \
                 --view (or use `growforge run`) to work headless"
            )
        })?;

        let info = adapter.get_info();
        let adapter_description = format!(
            "{} ({:?}, {:?}) driver {} {}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(constants::PROGRAM_NAME),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|error| {
            anyhow!(
                "the GPU adapter \"{}\" refused a rendering device ({error}); run without --view",
                info.name
            )
        })?;

        let capabilities = surface.get_capabilities(&adapter);
        // egui paints gamma encoded colours and expects a non-sRGB target; our
        // own colour constants are authored the same way.
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| capabilities.formats.first().copied())
            .ok_or_else(|| {
                anyhow!(
                    "the GPU adapter offers no colour format for this window; run without --view"
                )
            })?;
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| {
                anyhow!("the GPU adapter cannot present to this window; run without --view")
            })?;
        config.format = format;
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("growforge_scene"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("growforge_uniforms"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_projection: camera::to_f32(&camera::identity()),
                light: [0.0, 0.0, -1.0, constants::VIEW_AMBIENT_FRACTION],
                shading: [constants::VIEW_BACKLIGHT_FRACTION, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("growforge_uniform_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("growforge_uniforms"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("growforge_scene"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let opaque_pipeline =
            build_pipeline(&device, &pipeline_layout, &shader, config.format, false);
        let translucent_pipeline =
            build_pipeline(&device, &pipeline_layout, &shader, config.format, true);
        let depth_view = create_depth_view(&device, &config);
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            config.format,
            egui_wgpu::RendererOptions::default(),
        );

        Ok(Gpu {
            surface,
            device,
            queue,
            config,
            depth_view,
            opaque_pipeline,
            translucent_pipeline,
            uniform_buffer,
            bind_group,
            buffers: (0..LAYERS.len()).map(|_| None).collect(),
            egui_renderer,
            health: SurfaceHealth::default(),
            adapter_description,
        })
    }

    /// Current drawing surface size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigure the surface after a resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth_view(&self.device, &self.config);
    }

    /// Upload (or clear) the vertex buffer of one layer.
    pub fn upload(&mut self, layer: Layer, mesh: Option<&LayerMesh>) {
        let slot = layer.slot();
        self.buffers[slot] = match mesh {
            Some(mesh) if !mesh.is_empty() => Some(LayerBuffer {
                buffer: self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("growforge_layer"),
                        contents: bytemuck::cast_slice(&mesh.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                vertices: mesh.vertices.len() as u32,
            }),
            _ => None,
        };
    }

    /// Draw the scene and the egui overlay into the next surface frame.
    ///
    /// `scene_width` is how many pixels of the surface the 3D view owns; the
    /// rest is where egui puts its panel, and drawing geometry under it would
    /// only ever be hidden.
    pub fn render(
        &mut self,
        view_projection: &Mat4,
        scene: &Scene,
        scene_width: u32,
        egui: EguiFrame,
    ) -> Result<()> {
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_projection: camera::to_f32(view_projection),
                light: [
                    constants::VIEW_LIGHT_DIRECTION[0],
                    constants::VIEW_LIGHT_DIRECTION[1],
                    constants::VIEW_LIGHT_DIRECTION[2],
                    constants::VIEW_AMBIENT_FRACTION,
                ],
                shading: [constants::VIEW_BACKLIGHT_FRACTION, 0.0, 0.0, 0.0],
            }),
        );

        let acquired = self.surface.get_current_texture();
        match self.health.observe(SurfaceStatus::of(&acquired)) {
            SurfaceAction::Draw => {}
            SurfaceAction::Skip => return Ok(()),
            SurfaceAction::Reconfigure => {
                let (width, height) = self.size();
                self.resize(width, height);
                return Ok(());
            }
            SurfaceAction::Retry { streak } => {
                // Once per streak, on the frame that opened it. A device that
                // is resetting rejects frames as fast as they are asked for,
                // and a line each would be a wall of them for what is usually
                // over before it can be read.
                if streak == 1 {
                    println!(
                        "viewer surface the GPU rejected a frame; reconfiguring and retrying (the \
                         device may be resetting)"
                    );
                }
                let (width, height) = self.size();
                self.resize(width, height);
                return Ok(());
            }
            SurfaceAction::GiveUp { streak } => {
                return Err(anyhow!(
                    "the GPU rejected {streak} consecutive frames of the viewer; the device was \
                     likely reset and did not come back"
                ));
            }
        }
        let (wgpu::CurrentSurfaceTexture::Success(frame)
        | wgpu::CurrentSurfaceTexture::Suboptimal(frame)) = acquired
        else {
            // Those two are the whole of `Draw`; this arm is the compiler's
            // share of the destructuring, not a case.
            return Ok(());
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("growforge_frame"),
            });

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: egui.pixels_per_point,
        };
        for (id, delta) in &egui.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let user_buffers = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &egui.primitives,
            &screen,
        );

        {
            let background = constants::VIEW_BACKGROUND_COLOR;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("growforge_scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background[0],
                            g: background[1],
                            b: background[2],
                            a: background[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let width = scene_width.clamp(1, self.config.width.max(1));
            let height = self.config.height.max(1);
            pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, width, height);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // Opaque first so the translucent overlays can test against a
            // finished depth buffer without writing to it.
            for translucent in [false, true] {
                pass.set_pipeline(if translucent {
                    &self.translucent_pipeline
                } else {
                    &self.opaque_pipeline
                });
                for info in LAYERS.iter().filter(|i| i.translucent == translucent) {
                    if !scene.is_visible(info.layer) {
                        continue;
                    }
                    if let Some(buffer) = &self.buffers[info.layer.slot()] {
                        pass.set_vertex_buffer(0, buffer.buffer.slice(..));
                        pass.draw(0..buffer.vertices, 0..1);
                    }
                }
            }
        }

        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("growforge_egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer
                .render(&mut pass, &egui.primitives, &screen);
        }

        self.queue
            .submit(user_buffers.into_iter().chain([encoder.finish()]));
        frame.present();
        for id in &egui.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        Ok(())
    }
}

/// Check that some GPU adapter exists before any window is opened or any work
/// is started, and describe it.
///
/// Failing here is the difference between a clear "run without --view" message
/// and a half-built window, so both viewer entry points call it first.
pub fn probe_adapter() -> Result<String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .map_err(|error| {
        anyhow!(
            "no GPU adapter is available on this machine ({error}); run the same command without \
             --view, or use `growforge run` and `growforge check`, to work headless"
        )
    })?;
    let info = adapter.get_info();
    Ok(format!("{} ({:?})", info.name, info.backend))
}

fn create_depth_view(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("growforge_depth"),
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    translucent: bool,
) -> wgpu::RenderPipeline {
    let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(if translucent {
            "growforge_translucent"
        } else {
            "growforge_opaque"
        }),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attributes,
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: translucent.then_some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            // Both sides are drawn: a preview isosurface is open where it meets
            // the padding, and the overlays are seen from inside as often as
            // from outside.
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(!translucent),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The streak policy alone, which is the whole of the decision: what wgpu
    /// answered cannot be produced without a window and a device, so
    /// [`SurfaceStatus::of`] is the one line that stays a reading of the four
    /// cases rather than a decision about them.
    ///
    /// Rejections short of the limit are retried, and each says how far the
    /// streak has got so the first of one can be the only one reported.
    #[test]
    fn a_rejected_frame_is_retried_until_the_limit_and_then_given_up_on() {
        let mut health = SurfaceHealth::default();
        for streak in 1..constants::VIEW_SURFACE_REJECTION_LIMIT {
            assert_eq!(
                health.observe(SurfaceStatus::Rejected),
                SurfaceAction::Retry { streak },
                "rejection {streak} of {} is still a retry",
                constants::VIEW_SURFACE_REJECTION_LIMIT
            );
        }
        let streak = constants::VIEW_SURFACE_REJECTION_LIMIT;
        assert_eq!(
            health.observe(SurfaceStatus::Rejected),
            SurfaceAction::GiveUp { streak },
            "the limit is what the viewer gives up at"
        );
    }

    /// A frame that was acquired is the recovery, and it is the only thing that
    /// counts as one: the streak starts again from nothing after it.
    #[test]
    fn an_acquired_frame_clears_the_streak() {
        let mut health = SurfaceHealth::default();
        for _ in 0..constants::VIEW_SURFACE_REJECTION_LIMIT - 1 {
            health.observe(SurfaceStatus::Rejected);
        }
        assert_eq!(health.observe(SurfaceStatus::Acquired), SurfaceAction::Draw);
        assert_eq!(
            health.observe(SurfaceStatus::Rejected),
            SurfaceAction::Retry { streak: 1 },
            "a drawn frame leaves nothing behind to give up on"
        );
    }

    /// A busy compositor and a resized window are not a device in trouble.
    /// Neither counts towards the limit - a session that skipped a thousand
    /// frames behind another window may not die of it - and neither forgives a
    /// rejection either, because neither proves the device drew anything.
    #[test]
    fn a_skipped_or_stale_frame_is_not_a_rejection_and_does_not_clear_one() {
        let mut health = SurfaceHealth::default();
        for _ in 0..constants::VIEW_SURFACE_REJECTION_LIMIT * 2 {
            assert_eq!(health.observe(SurfaceStatus::Skipped), SurfaceAction::Skip);
            assert_eq!(
                health.observe(SurfaceStatus::Stale),
                SurfaceAction::Reconfigure
            );
        }

        assert_eq!(
            health.observe(SurfaceStatus::Rejected),
            SurfaceAction::Retry { streak: 1 }
        );
        health.observe(SurfaceStatus::Skipped);
        health.observe(SurfaceStatus::Stale);
        assert_eq!(
            health.observe(SurfaceStatus::Rejected),
            SurfaceAction::Retry { streak: 2 },
            "the streak carries across the answers that are not about the device"
        );
    }
}
