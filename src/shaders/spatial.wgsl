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

@group(0) @binding(0) var<storage, read> source: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> filters: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> destination: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> error_destination: array<f32>;

fn clamp_x(x: i32) -> u32 {
    return u32(clamp(x, 0, i32(params.width) - 1));
}

fn clamp_y(y: i32) -> u32 {
    return u32(clamp(y, 0, i32(params.height) - 1));
}

fn xyz_to_linear_rgb(xyz: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        3.241003275 * xyz.x - 1.537398934 * xyz.y - 0.498615861 * xyz.z,
        -0.969224334 * xyz.x + 1.875930071 * xyz.y + 0.041554224 * xyz.z,
        0.055639423 * xyz.x - 0.204011202 * xyz.y + 1.057148933 * xyz.z,
    );
}

fn linear_rgb_to_xyz(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        (10135552.0 / 24577794.0) * rgb.x
            + (8788810.0 / 24577794.0) * rgb.y
            + (4435075.0 / 24577794.0) * rgb.z,
        (2613072.0 / 12288897.0) * rgb.x
            + (8788810.0 / 12288897.0) * rgb.y
            + (887015.0 / 12288897.0) * rgb.z,
        (1425312.0 / 73733382.0) * rgb.x
            + (8788810.0 / 73733382.0) * rgb.y
            + (70074185.0 / 73733382.0) * rgb.z,
    );
}

fn xyz_to_lab(input_xyz: vec3<f32>) -> vec3<f32> {
    let xyz = input_xyz * vec3<f32>(1.052156925, 1.0, 0.918357670);
    let delta = 6.0 / 29.0;
    let delta_cube = delta * delta * delta;
    let factor = 1.0 / (3.0 * delta * delta);
    let term = 4.0 / 29.0;
    var f = xyz;
    if (xyz.x > delta_cube) {
        f.x = pow(xyz.x, 1.0 / 3.0);
    } else {
        f.x = factor * xyz.x + term;
    }
    if (xyz.y > delta_cube) {
        f.y = pow(xyz.y, 1.0 / 3.0);
    } else {
        f.y = factor * xyz.y + term;
    }
    if (xyz.z > delta_cube) {
        f.z = pow(xyz.z, 1.0 / 3.0);
    } else {
        f.z = factor * xyz.z + term;
    }
    return vec3<f32>(
        116.0 * f.y - 16.0,
        500.0 * (f.x - f.y),
        200.0 * (f.y - f.z),
    );
}

fn to_lab_hunt(ycxc_z: vec3<f32>) -> vec3<f32> {
    let y_prime = (ycxc_z.x + 16.0) / 116.0;
    let cx = ycxc_z.y / 500.0;
    let cz = ycxc_z.z / 200.0;
    let xyz = vec3<f32>(y_prime + cx, y_prime, y_prime - cz)
        * vec3<f32>(0.950428545, 1.0, 1.088900371);
    let linear_rgb = clamp(xyz_to_linear_rgb(xyz), vec3<f32>(0.0), vec3<f32>(1.0));
    var lab = xyz_to_lab(linear_rgb_to_xyz(linear_rgb));
    lab.y = 0.01 * lab.x * lab.y;
    lab.z = 0.01 * lab.x * lab.z;
    return lab;
}

fn hyab(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a.yz - b.yz;
    return abs(a.x - b.x) + sqrt(d.x * d.x + d.y * d.y);
}

@compute @workgroup_size(16, 8)
fn spatial_x(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let pixel_index = id.y * params.width + id.x;
    var ref_y = 0.0;
    var ref_cx = 0.0;
    var ref_cz1 = 0.0;
    var ref_cz2 = 0.0;
    var test_y = 0.0;
    var test_cx = 0.0;
    var test_cz1 = 0.0;
    var test_cz2 = 0.0;
    for (var offset = -i32(params.spatial_radius); offset <= i32(params.spatial_radius); offset += 1) {
        let x = clamp_x(i32(id.x) + offset);
        let sample_index = id.y * params.width + x;
        let weight = filters[u32(offset + i32(params.spatial_radius))];
        let ref_color = source[sample_index * 2u];
        let test_color = source[sample_index * 2u + 1u];
        ref_y += weight.x * ref_color.x;
        ref_cx += weight.y * ref_color.y;
        ref_cz1 += weight.z * ref_color.z;
        ref_cz2 += weight.w * ref_color.z;
        test_y += weight.x * test_color.x;
        test_cx += weight.y * test_color.y;
        test_cz1 += weight.z * test_color.z;
        test_cz2 += weight.w * test_color.z;
    }
    let out_index = pixel_index * 2u;
    destination[out_index] = vec4<f32>(ref_y, ref_cx, ref_cz1, ref_cz2);
    destination[out_index + 1u] = vec4<f32>(test_y, test_cx, test_cz1, test_cz2);
}

@compute @workgroup_size(16, 8)
fn spatial_y(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let pixel_index = id.y * params.width + id.x;
    var ref_y = 0.0;
    var ref_cx = 0.0;
    var ref_cz1 = 0.0;
    var ref_cz2 = 0.0;
    var test_y = 0.0;
    var test_cx = 0.0;
    var test_cz1 = 0.0;
    var test_cz2 = 0.0;
    for (var offset = -i32(params.spatial_radius); offset <= i32(params.spatial_radius); offset += 1) {
        let y = clamp_y(i32(id.y) + offset);
        let sample_index = (y * params.width + id.x) * 2u;
        let weight = filters[u32(offset + i32(params.spatial_radius))];
        let ref_color = source[sample_index];
        let test_color = source[sample_index + 1u];
        ref_y += weight.x * ref_color.x;
        ref_cx += weight.y * ref_color.y;
        ref_cz1 += weight.z * ref_color.z;
        ref_cz2 += weight.w * ref_color.w;
        test_y += weight.x * test_color.x;
        test_cx += weight.y * test_color.y;
        test_cz1 += weight.z * test_color.z;
        test_cz2 += weight.w * test_color.w;
    }

    let ref_lab = to_lab_hunt(vec3<f32>(ref_y, ref_cx, ref_cz1 + ref_cz2));
    let test_lab = to_lab_hunt(vec3<f32>(test_y, test_cx, test_cz1 + test_cz2));
    let color_difference = pow(hyab(ref_lab, test_lab), params.gqc);
    var mapped_difference: f32;
    if (color_difference < params.pccmax) {
        mapped_difference = color_difference * (params.gpt / params.pccmax);
    } else {
        mapped_difference = params.gpt
            + ((color_difference - params.pccmax) / (params.cmax - params.pccmax))
                * (1.0 - params.gpt);
    }
    error_destination[pixel_index] = mapped_difference;
}
