#version 460

// ============================================================================
// particle.frag — point-sprite fragment shader.
//
// Renders each particle as a soft round dot. Uses gl_PointCoord (the
// fragment's [0, 1] coordinate inside the point sprite) to discard pixels
// outside a unit circle, with a smooth alpha falloff at the edge.
// ============================================================================

layout(location = 0) in  vec3 in_color;
layout(location = 0) out vec4 out_color;

void main() {
    // Distance from the sprite center, in [0, 0.5√2].
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) {
        discard;
    }
    // Smooth fade in the outer 20% to anti-alias the edge.
    float alpha = smoothstep(0.25, 0.18, r2);
    out_color = vec4(in_color, alpha);
}
