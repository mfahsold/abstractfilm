#version 330
uniform sampler2D input_tex;
uniform vec3 shadow_tint;
uniform vec3 highlight_tint;
uniform float saturation;
uniform float gamma;
in vec2 v_uv;
out vec4 fragColor;

vec3 apply_saturation(vec3 color, float sat) {
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    return mix(vec3(gray), color, sat);
}

vec3 apply_gamma(vec3 color, float g) {
    return pow(color, vec3(1.0 / max(g, 1e-3)));
}

void main() {
    vec4 input_color = texture(input_tex, v_uv);
    vec3 c = apply_gamma(input_color.rgb, gamma);
    float luminance = dot(c, vec3(0.299, 0.587, 0.114));
    vec3 tint = mix(shadow_tint, highlight_tint, smoothstep(0.2, 0.8, luminance));
    c = apply_saturation(c, saturation);
    c = clamp(c * tint, 0.0, 1.0);
    fragColor = vec4(c, input_color.a);
}
