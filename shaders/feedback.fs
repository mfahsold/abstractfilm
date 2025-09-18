#version 330
uniform sampler2D rd_tex;
uniform sampler2D video_tex;
uniform float rd_mix;
uniform float glow_strength;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec4 rd = texture(rd_tex, v_uv);
    vec4 vid = texture(video_tex, v_uv);
    float luminance = dot(vid.rgb, vec3(0.299, 0.587, 0.114));
    vec3 glow = smoothstep(0.7, 1.0, vec3(luminance)) * vid.rgb * glow_strength;
    vec3 combined = mix(vid.rgb, rd.rgb, rd_mix) + glow;
    fragColor = vec4(clamp(combined, 0.0, 1.0), 1.0);
}
