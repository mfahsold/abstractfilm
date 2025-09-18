#version 330
uniform sampler2D state_tex;
uniform sampler2D drive_tex;
uniform float diffusion;
uniform float feedback;
uniform float time_step;
in vec2 v_uv;
out vec4 fragColor;

vec4 sample_state(vec2 offset) {
    vec2 texel = 1.0 / vec2(textureSize(state_tex, 0));
    return texture(state_tex, v_uv + offset * texel);
}

void main() {
    vec4 center = texture(state_tex, v_uv);
    vec4 north = sample_state(vec2(0.0, 1.0));
    vec4 south = sample_state(vec2(0.0, -1.0));
    vec4 east = sample_state(vec2(1.0, 0.0));
    vec4 west = sample_state(vec2(-1.0, 0.0));
    vec4 laplace = (north + south + east + west) - 4.0 * center;
    vec4 drive = texture(drive_tex, v_uv);
    vec4 updated = center + time_step * (diffusion * laplace + feedback * (drive - center));
    fragColor = clamp(updated, 0.0, 1.0);
}
