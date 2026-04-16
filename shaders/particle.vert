#version 450

struct Particle {
    vec2 pos;
    vec2 vel;
    uint gen;
    uint _p0;
    uint _p1;
    uint _p2;
};

layout(std430, binding = 0) readonly buffer Particles {
    Particle data[];
};

layout(push_constant) uniform PC {
    vec4 view;    // view_min.xy, view_max.xy
    vec4 params;  // half_size, base_hue, hue_step, _
} pc;

layout(location = 0) out vec3 v_color;

const vec2 corners[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0)
);

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 view_min = pc.view.xy;
    vec2 view_max = pc.view.zw;
    float half_size = pc.params.x;
    float base_hue  = pc.params.y;
    float hue_step  = pc.params.z;

    Particle p = data[gl_InstanceIndex];
    vec2 world = p.pos + corners[gl_VertexIndex] * half_size;

    vec2 t = (world - view_min) / (view_max - view_min);
    vec2 ndc = t * 2.0 - 1.0;
    gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);

    float hue = fract(base_hue + float(p.gen) * hue_step);
    v_color = hsv2rgb(vec3(hue, 0.82, 0.95));
}
