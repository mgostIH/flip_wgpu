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
    padding0: f32,
    padding1: f32,
};

@group(0) @binding(0) var<storage, read> reference_pixels: array<u32>;
@group(0) @binding(1) var<storage, read> test_pixels: array<u32>;
@group(0) @binding(2) var<storage, read_write> colors: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn srgb_to_linear(channel: f32) -> f32 {
    if (channel <= 0.04045) {
        return channel / 12.92;
    }
    return pow((channel + 0.055) / 1.055, 2.4);
}

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

fn srgb_word_to_ycxc_z(pixel: u32) -> vec3<f32> {
    let srgb = vec3<f32>(
        f32(pixel & 255u),
        f32((pixel >> 8u) & 255u),
        f32((pixel >> 16u) & 255u),
    ) / 255.0;
    let linear_rgb = vec3<f32>(
        srgb_to_linear(srgb.x),
        srgb_to_linear(srgb.y),
        srgb_to_linear(srgb.z),
    );
    let xyz = linear_rgb_to_xyz(linear_rgb) * vec3<f32>(
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

@compute @workgroup_size(8, 8)
fn convert(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let index = id.y * params.width + id.x;
    colors[index * 2u] = vec4<f32>(srgb_word_to_ycxc_z(reference_pixels[index]), 0.0);
    colors[index * 2u + 1u] = vec4<f32>(srgb_word_to_ycxc_z(test_pixels[index]), 0.0);
}
