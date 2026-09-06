//! wgpu device setup and the two-pass renderer: the shaded scene first, the
//! egui overlay on top.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::constants;
use crate::device::{Scopes, ask, refused, refused_with};
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
    /// A frame came back and was drawn into.
    Acquired,
    /// No frame this time, and nothing is wrong with the surface: the
    /// compositor was busy, or the window is not visible.
    Skipped,
    /// The surface no longer matches the window and has to be reconfigured.
    Stale,
    /// The device refused: it would not hand a frame over, or it would not
    /// give a frame it had handed over what drawing into it needs - the depth
    /// texture, a layer's vertex buffer, egui's own. Both are the renderer
    /// asking the device for something and being told no.
    Rejected,
}

impl SurfaceStatus {
    /// Read what wgpu answered.
    ///
    /// Only the acquisition: whether an acquired frame was really drawn is the
    /// device's next answer, and [`Gpu::render`] observes that one instead.
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
/// device that may be resetting, or busy with somebody else's memory: the
/// surface is reconfigured and the frame skipped, exactly as a stale surface
/// is, and only a rejection that goes on for
/// [`constants::VIEW_SURFACE_REJECTION_LIMIT`] consecutive frames is a device
/// that is not coming back. A frame that was *drawn* is what clears the streak,
/// because the recovery is proven by drawing and by no other answer the surface
/// gives: a skipped or stale frame neither counts towards the limit nor forgives
/// what came before it, and neither does a frame that was handed over and then
/// refused what it needed.
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
    /// What the device last refused outside a frame - a layer's vertex buffer -
    /// waiting for the next frame to count it.
    ///
    /// An upload is not a frame and reconfiguring the surface would not retry
    /// it, so it is not the streak's business where it happens. It is the next
    /// frame's: the device that would not take a layer is the one that frame is
    /// about to ask for more, and the frame loop is where the bounded retry
    /// lives.
    refused: Option<String>,
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
        // Everything the device is asked to make for this window, under one
        // reading: a machine whose video memory is already spoken for refuses
        // here as readily as it does mid-session, and wgpu's answer to a refusal
        // nobody caught is to panic. What comes back instead is the same "run
        // without --view" this function has always answered a device it could
        // not use with.
        let scopes = Scopes::open(&device);
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
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            config.format,
            egui_wgpu::RendererOptions::default(),
        );
        if let Some(refusal) = scopes.close() {
            return Err(refused_with(
                "set the viewer's renderer up",
                &refusal,
                "close what else is using the card, or run the same command without --view",
            ));
        }
        let depth_view = create_depth_view(&device, &config)?;

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
            refused: None,
            adapter_description,
        })
    }

    /// Current drawing surface size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigure the surface after a resize.
    ///
    /// A device that refuses the surface or the depth texture the new size needs
    /// has rejected this frame exactly as one that refuses a frame has: the
    /// streak counts it, the next frame tries again, and the error only comes
    /// back once the streak has run out - see [`SurfaceHealth`].
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.config.width = width;
        self.config.height = height;
        match self.reconfigure() {
            None => Ok(()),
            Some(failure) => self.frame_over(SurfaceStatus::Rejected, Some(failure)),
        }
    }

    /// Upload (or clear) the vertex buffer of one layer.
    ///
    /// A device that refuses the buffer leaves the layer empty instead of taking
    /// the process down with it, and what it said waits for the next frame: see
    /// [`Gpu::refused`].
    pub fn upload(&mut self, layer: Layer, mesh: Option<&LayerMesh>) {
        let slot = layer.slot();
        let Some(mesh) = mesh.filter(|mesh| !mesh.is_empty()) else {
            self.buffers[slot] = None;
            return;
        };
        let uploaded = ask(&self.device, "take a layer of the scene", || {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("growforge_layer"),
                    contents: bytemuck::cast_slice(&mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
        match uploaded {
            Ok(buffer) => {
                self.buffers[slot] = Some(LayerBuffer {
                    buffer,
                    vertices: mesh.vertices.len() as u32,
                });
            }
            Err(error) => {
                self.buffers[slot] = None;
                self.refused = Some(format!("{error:#}"));
            }
        }
    }

    /// Configure the surface for the size in the current configuration and make
    /// the depth texture that goes with it. Answers what the device refused
    /// with, if it refused.
    ///
    /// What it had before is left in place: the frame that asked is over either
    /// way, and a depth texture that is no longer valid is a frame the device
    /// will refuse again, which is what the streak is counting.
    fn reconfigure(&mut self) -> Option<String> {
        if let Err(error) = ask(&self.device, "reconfigure the drawing surface", || {
            self.surface.configure(&self.device, &self.config)
        }) {
            return Some(format!("{error:#}"));
        }
        match create_depth_view(&self.device, &self.config) {
            Ok(view) => {
                self.depth_view = view;
                None
            }
            Err(error) => Some(format!("{error:#}")),
        }
    }

    /// What one frame came to, and what the renderer does about it.
    ///
    /// The one place the streak is advanced, so a frame counts once however it
    /// ended. `failure` is what the device said when it refused, when it said
    /// anything: a frame the surface would not hand over says nothing beyond
    /// having been refused.
    fn frame_over(&mut self, status: SurfaceStatus, failure: Option<String>) -> Result<()> {
        match self.health.observe(status) {
            SurfaceAction::Draw | SurfaceAction::Skip => Ok(()),
            // A surface that has to be reconfigured and cannot be is a device
            // refusing this frame rather than a window that moved, and it is
            // counted as one: a device answering "outdated" for ever, with
            // every reconfigure refused, would otherwise be retried for ever
            // and never reach the give-up the rescue hangs off. One level deep
            // only - a rejected frame reconfigures too, and what *that* refuses
            // is the next frame's verdict.
            SurfaceAction::Reconfigure => match self.reconfigure() {
                None => Ok(()),
                Some(failure) => self.frame_over(SurfaceStatus::Rejected, Some(failure)),
            },
            SurfaceAction::Retry { streak } => {
                // Once per streak, on the frame that opened it. A device that
                // is resetting rejects frames as fast as they are asked for,
                // and a line each would be a wall of them for what is usually
                // over before it can be read.
                if streak == 1 {
                    match &failure {
                        Some(failure) => {
                            println!("viewer surface {failure}; reconfiguring and retrying")
                        }
                        None => println!(
                            "viewer surface the GPU rejected a frame; reconfiguring and retrying \
                             (the device may be resetting)"
                        ),
                    }
                }
                let _ = self.reconfigure();
                Ok(())
            }
            SurfaceAction::GiveUp { streak } => Err(match failure {
                Some(failure) => anyhow!(
                    "{failure}, and {streak} consecutive frames of the viewer went the same way; \
                     the device did not come back"
                ),
                None => anyhow!(
                    "the GPU rejected {streak} consecutive frames of the viewer; the device was \
                     likely reset and did not come back"
                ),
            }),
        }
    }

    /// Draw the scene and the egui overlay into the next surface frame.
    ///
    /// `scene_width` is how many pixels of the surface the 3D view owns; the
    /// rest is where egui puts its panel, and drawing geometry under it would
    /// only ever be hidden.
    ///
    /// One request stays outside every scope here: dropping a frame that was
    /// not presented discards its texture, and wgpu reports a failed discard
    /// fatally whatever scope is open. A device that refuses *that* still ends
    /// the process, as every refusal used to.
    pub fn render(
        &mut self,
        view_projection: &Mat4,
        scene: &Scene,
        scene_width: u32,
        egui: EguiFrame,
    ) -> Result<()> {
        // A layer the device would not take is this frame's verdict before the
        // frame is even asked for: it is the same device, and it is about to be
        // asked for more.
        if let Some(failure) = self.refused.take() {
            return self.frame_over(SurfaceStatus::Rejected, Some(failure));
        }
        // Acquiring is a request like any other: a surface whose configuration
        // the device refused has no presentation left to hand a frame from, and
        // answers that through the handler rather than in the value below.
        let acquired = match ask(&self.device, "hand a frame over", || {
            self.surface.get_current_texture()
        }) {
            Ok(acquired) => acquired,
            Err(error) => {
                return self.frame_over(SurfaceStatus::Rejected, Some(format!("{error:#}")));
            }
        };
        let status = SurfaceStatus::of(&acquired);
        if status != SurfaceStatus::Acquired {
            return self.frame_over(status, None);
        }
        let (wgpu::CurrentSurfaceTexture::Success(frame)
        | wgpu::CurrentSurfaceTexture::Suboptimal(frame)) = acquired
        else {
            // Those two are the whole of an acquisition; this arm is the
            // compiler's share of the destructuring, not a case.
            return Ok(());
        };
        match self.draw(&frame, view_projection, scene, scene_width, egui) {
            // Presenting is the last thing the device is asked for and is
            // refused like the rest: a frame drawn into a surface that went
            // invalid in the meantime is a frame that never reached the screen.
            Ok(()) => match ask(&self.device, "present the frame it drew", || {
                frame.present()
            }) {
                Ok(()) => self.frame_over(SurfaceStatus::Acquired, None),
                Err(error) => self.frame_over(SurfaceStatus::Rejected, Some(format!("{error:#}"))),
            },
            // Nothing was drawn, so nothing is presented: the frame is dropped
            // and the streak decides whether the device gets another.
            Err(error) => self.frame_over(SurfaceStatus::Rejected, Some(format!("{error:#}"))),
        }
    }

    /// Draw one acquired frame, with everything the device is asked for inside
    /// one reading of its errors.
    ///
    /// The frame's own view, egui's textures and buffers, the encoder, the
    /// passes and the submission are all in here, so one scope covers every
    /// allocation a device under memory pressure can refuse - and refusing any
    /// of them is a frame that was not drawn rather than a process that is over.
    fn draw(
        &mut self,
        frame: &wgpu::SurfaceTexture,
        view_projection: &Mat4,
        scene: &Scene,
        scene_width: u32,
        egui: EguiFrame,
    ) -> Result<()> {
        let scopes = Scopes::open(&self.device);
        self.paint(frame, view_projection, scene, scene_width, egui);
        match scopes.close() {
            None => Ok(()),
            Some(refusal) => Err(refused("draw a frame", &refusal)),
        }
    }

    /// The drawing itself.
    ///
    /// Nothing here reports a failure, because nothing here can: what the device
    /// refuses it answers for through the error scope [`Gpu::draw`] holds open
    /// over the whole of it, and an invalid resource is drawn with rather than
    /// panicked on until that scope is read.
    fn paint(
        &mut self,
        frame: &wgpu::SurfaceTexture,
        view_projection: &Mat4,
        scene: &Scene,
        scene_width: u32,
        egui: EguiFrame,
    ) {
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
        for id in &egui.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
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

/// The depth buffer the 3D pass writes, sized to the configured surface.
///
/// Both calls are read together, because the second is only ever refused when
/// the first was: a texture the device had no memory for comes back invalid, and
/// making a view of it is the validation error the session used to die of.
fn create_depth_view(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> Result<wgpu::TextureView> {
    ask(device, "make the viewer's depth texture", || {
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
    })
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

    /// A frame the device handed over and then refused what it needed - the
    /// depth texture, a layer's vertex buffer, egui's own - is a rejected frame
    /// like one it never handed over, and [`Gpu::render`] observes it as one.
    ///
    /// Which is the whole reason the acquisition is not what clears the streak:
    /// a device that keeps handing over frames it will not let anyone draw into
    /// would otherwise start the count again every frame and never be given up
    /// on, however long it went on refusing.
    #[test]
    fn a_frame_that_was_acquired_and_then_refused_counts_towards_the_same_limit() {
        let mut health = SurfaceHealth::default();
        // The surface would not hand this one over - either by answering that
        // it could not, or by refusing the acquisition outright, which is what
        // a surface whose reconfigure was refused does.
        assert_eq!(
            health.observe(SurfaceStatus::Rejected),
            SurfaceAction::Retry { streak: 1 }
        );
        // ... and the device would not give this one its depth texture.
        assert_eq!(
            health.observe(SurfaceStatus::Rejected),
            SurfaceAction::Retry { streak: 2 },
            "where the refusal happened is not what the streak counts"
        );
        // A frame that really drew is what clears them both.
        assert_eq!(health.observe(SurfaceStatus::Acquired), SurfaceAction::Draw);
        assert_eq!(
            health.observe(SurfaceStatus::Rejected),
            SurfaceAction::Retry { streak: 1 }
        );
    }

    /// A stale surface the device will not reconfigure is a rejected frame, and
    /// the streak is what ends it.
    ///
    /// The case is a device that has stopped working while still answering:
    /// every frame comes back outdated, every reconfigure is refused, and
    /// nothing is ever drawn. Counting the refused reconfigure - which is what
    /// [`Gpu::frame_over`] does with one - is what gets such a session to the
    /// give-up, and from there to the teardown that writes its work out.
    #[test]
    fn a_reconfigure_the_device_refuses_ends_a_surface_that_is_always_stale() {
        let mut health = SurfaceHealth::default();
        let mut last = SurfaceAction::Skip;
        for _ in 0..constants::VIEW_SURFACE_REJECTION_LIMIT {
            assert_eq!(
                health.observe(SurfaceStatus::Stale),
                SurfaceAction::Reconfigure
            );
            // And the reconfigure it asks for is refused.
            last = health.observe(SurfaceStatus::Rejected);
        }
        assert_eq!(
            last,
            SurfaceAction::GiveUp {
                streak: constants::VIEW_SURFACE_REJECTION_LIMIT
            },
            "a surface nobody can reconfigure would otherwise be retried for ever"
        );
    }

    /// The depth texture on this machine's own device: an extent the device
    /// cannot make comes back as an error rather than as the panic that ended a
    /// session mid-run.
    ///
    /// The real trigger - a card with no memory left for a texture this size -
    /// cannot be staged in a test, but it arrives through the same handler as
    /// this one, and what is on trial is that somebody is reading it.
    #[test]
    fn a_depth_texture_the_device_cannot_make_is_an_error_rather_than_a_panic() {
        let Some(device) = crate::device::device_or_skip("the depth texture test") else {
            return;
        };
        let mut config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8Unorm,
            width: device.limits().max_texture_dimension_2d.saturating_add(1),
            height: 1,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: Vec::new(),
        };
        let error = create_depth_view(&device, &config)
            .expect_err("the device made a texture wider than it says it can");
        println!("the device refused as expected: {error:#}");

        // And the window sizes it really gets are unaffected by the reading.
        config.width = constants::VIEW_WINDOW_WIDTH;
        config.height = constants::VIEW_WINDOW_HEIGHT;
        create_depth_view(&device, &config).expect("a depth texture of a window's own size");
    }
}
