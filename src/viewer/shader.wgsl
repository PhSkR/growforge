// Shaded triangle soup for the growforge viewer.
//
// Vertices carry their own colour, so one pipeline pair (opaque and
// translucent) draws every layer without rebinding anything but the vertex
// buffer.
//
// Shading mode is a property of the vertex data, not of this shader: a normal
// is interpolated across the triangle and renormalized per fragment, so the
// same code shades smoothly when the vertices carry averaged vertex normals
// and flat when all three carry the face normal. That is what keeps the flat
// mode pixel-exact rather than approximated from screen space derivatives.

struct Uniforms {
    view_projection: mat4x4<f32>,
    // xyz: direction the key light travels, w: ambient fraction.
    light: vec4<f32>,
    // x: backlight fraction, yzw: padding to the 16 byte alignment.
    shading: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_projection * vec4<f32>(position, 1.0);
    out.normal = normal;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.normal);
    let to_light = normalize(-uniforms.light.xyz);
    // Back faces are lit too: an unsmoothed preview surface and the inside of
    // a translucent overlay would otherwise read as flat black.
    let front = max(dot(normal, to_light), 0.0);
    let back = max(dot(-normal, to_light), 0.0) * uniforms.shading.x;
    let ambient = uniforms.light.w;
    let shade = ambient + (1.0 - ambient) * max(front, back);
    return vec4<f32>(in.color.rgb * shade, in.color.a);
}
