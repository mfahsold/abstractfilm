#version 330

uniform sampler2D state_tex;
uniform sampler2D drive_tex;
uniform sampler2D flow_tex;
uniform float diffusion_u;
uniform float diffusion_v;
uniform float feed;
uniform float kill;
uniform float advection;
uniform float drive_influence;
uniform float time_step;

in vec2 v_uv;
out vec4 fragColor;

vec2 sample_state(vec2 uv)
{
    return texture(state_tex, clamp(uv, 0.0, 1.0)).rg;
}

void main()
{
    vec2 texel = 1.0 / vec2(textureSize(state_tex, 0));
    vec2 flow = texture(flow_tex, v_uv).rg * advection;
    vec2 advected_uv = clamp(v_uv - flow, 0.0, 1.0);

    vec2 center = sample_state(advected_uv);
    vec2 north = sample_state(advected_uv + vec2(0.0, texel.y));
    vec2 south = sample_state(advected_uv - vec2(0.0, texel.y));
    vec2 east  = sample_state(advected_uv + vec2(texel.x, 0.0));
    vec2 west  = sample_state(advected_uv - vec2(texel.x, 0.0));
    vec2 northeast = sample_state(advected_uv + vec2(texel.x, texel.y));
    vec2 northwest = sample_state(advected_uv + vec2(-texel.x, texel.y));
    vec2 southeast = sample_state(advected_uv + vec2(texel.x, -texel.y));
    vec2 southwest = sample_state(advected_uv + vec2(-texel.x, -texel.y));

    float lap_u = (north.r + south.r + east.r + west.r) * 0.2 + (northeast.r + northwest.r + southeast.r + southwest.r) * 0.05 - center.r * 0.8;
    float lap_v = (north.g + south.g + east.g + west.g) * 0.2 + (northeast.g + northwest.g + southeast.g + southwest.g) * 0.05 - center.g * 0.8;

    float u = center.r;
    float v = center.g;

    float drive_luma = dot(texture(drive_tex, v_uv).rgb, vec3(0.299, 0.587, 0.114));
    float feed_mod = clamp(feed + drive_influence * (drive_luma - 0.5), 0.0, 0.2);
    float kill_mod = clamp(kill, 0.0, 0.25);

    float uvv = u * v * v;
    float du = diffusion_u * lap_u - uvv + feed_mod * (1.0 - u);
    float dv = diffusion_v * lap_v + uvv - (kill_mod + feed_mod) * v;

    u = clamp(u + time_step * du, 0.0, 1.0);
    v = clamp(v + time_step * dv, 0.0, 1.0);

    fragColor = vec4(u, v, drive_luma, 1.0);
}
