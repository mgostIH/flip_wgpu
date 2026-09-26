struct Params {
    width: u32,
    height: u32,
    spatial_radius: u32,
    feature_radius: u32,
    cmax: f32,
    pccmax: f32,
    gqc: f32,
    gpt: f32,
    gqf: f32,
    exposure: f32,
    tone_mapper: u32,
    padding1: f32,
};

@group(0) @binding(0) var<storage, read> reference_pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> test_pixels: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> colors: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn linear_rgb_to_xyz(rgb: vec3<f32>) -> vec3<f32> {
    let x = (10135552.0 / 24577794.0) * rgb.x
        + (8788810.0 / 24577794.0) * rgb.y
        + (4435075.0 / 24577794.0) * rgb.z;
    let y = (2613072.0 / 12288897.0) * rgb.x
        + (8788810.0 / 12288897.0) * rgb.y
        + (887015.0 / 12288897.0) * rgb.z;
    let z = (1425312.0 / 73733382.0) * rgb.x
        + (8788810.0 / 73733382.0) * rgb.y
        + (70074185.0 / 73733382.0) * rgb.z;
    return vec3<f32>(x, y, z);
}

fn tone_map(rgb: vec3<f32>) -> vec3<f32> {
    let exposed = rgb * exp2(params.exposure);
    if (params.tone_mapper == 0u) {
        let luminance = dot(exposed, vec3<f32>(0.2126, 0.7152, 0.0722));
        return clamp(exposed / (1.0 + luminance), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    var a = 0.9036;
    var b = 0.018;
    var d = 0.8748;
    var e = 0.354;
    var f = 0.14;
    if (params.tone_mapper == 2u) {
        a = 0.231683;
        b = 0.013791;
        d = 0.18;
        e = 0.3;
        f = 0.018;
    }
    let square = exposed * exposed;
    let numerator = square * a + exposed * b;
    let denominator = square * d + exposed * e + f;
    return clamp(numerator / denominator, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_rgb_to_ycxc_z(rgb: vec3<f32>) -> vec3<f32> {
    let xyz = linear_rgb_to_xyz(rgb) * vec3<f32>(
        1.052156925,
        1.0,
        0.918357670,
    );
    return vec3<f32>(
        116.0 * xyz.y - 16.0,
        500.0 * (xyz.x - xyz.y),
        200.0 * (xyz.y - xyz.z),
    );
}

@compute @workgroup_size(16, 8)
fn convert_hdr(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let index = id.y * params.width + id.x;
    let ref_rgb = tone_map(reference_pixels[index].xyz);
    let test_rgb = tone_map(test_pixels[index].xyz);
    colors[index * 2u] = vec4<f32>(linear_rgb_to_ycxc_z(ref_rgb), 0.0);
    colors[index * 2u + 1u] = vec4<f32>(linear_rgb_to_ycxc_z(test_rgb), 0.0);
}
