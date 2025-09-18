#version 330

uniform sampler2D rd_tex;
uniform sampler2D video_tex;
uniform float rd_mix;
uniform float glow_strength;
uniform vec3 glow_color;
uniform sampler2D flow_tex;
uniform float displacement;
uniform float height_scale;
uniform vec3 light_dir;

in vec2 v_uv;
out vec4 fragColor;

vec3 palette(float t)
{
    vec3 a = vec3(0.18, 0.05, 0.35);
    vec3 b = vec3(0.95, 0.35, 0.80);
    vec3 c = vec3(0.20, 0.45, 0.65);
    vec3 d = vec3(0.95, 0.75, 1.05);
    return a + b * sin(6.28318 * (c * t + d));
}

void main()
{
    vec4 rd_sample = texture(rd_tex, v_uv);
    vec2 flow = texture(flow_tex, v_uv).rg;
    vec2 texel = 1.0 / vec2(textureSize(rd_tex, 0));

    float u = rd_sample.r;
    float v = rd_sample.g;
    float drive = rd_sample.b;
    float pattern = clamp(v - u + drive * 0.75, 0.0, 1.0);

    vec2 displacement_uv = v_uv + flow * displacement + (pattern - 0.5) * displacement * vec2(0.45, 0.75);
    displacement_uv = clamp(displacement_uv, 0.0, 1.0);
    vec3 video_sample = texture(video_tex, displacement_uv).rgb;

    vec3 rd_color = palette(pattern);
    rd_color = clamp(rd_color, 0.0, 1.5);
    rd_color *= 0.55 + 0.65 * drive;

    float height = pattern;
    float hx = texture(rd_tex, v_uv + vec2(texel.x, 0.0)).g - height;
    float hy = texture(rd_tex, v_uv + vec2(0.0, texel.y)).g - height;
    vec3 normal = normalize(vec3(-hx * height_scale, -hy * height_scale, 1.0));
    float lighting = max(0.0, dot(normal, normalize(light_dir)));
    float ambient = 0.35;
    float shade = clamp(ambient + (1.0 - ambient) * lighting, 0.0, 1.5);

    vec3 combined = mix(video_sample, rd_color, rd_mix);
    combined *= shade;

    float glow_mask = smoothstep(0.55, 0.95, drive) + smoothstep(0.45, 0.85, pattern) * 0.5;
    vec3 glow = glow_strength * glow_color * glow_mask;

    combined = clamp(combined + glow, 0.0, 1.0);
    fragColor = vec4(combined, 1.0);
}
