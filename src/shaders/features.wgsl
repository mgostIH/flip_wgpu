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
@group(0) @binding(2) var<storage, read> color_difference: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> destination: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

fn clamp_x(x: i32) -> u32 {
    return u32(clamp(x, 0, i32(params.width) - 1));
}

fn clamp_y(y: i32) -> u32 {
    return u32(clamp(y, 0, i32(params.height) - 1));
}

@compute @workgroup_size(8, 8)
fn features_x(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let pixel_index = id.y * params.width + id.x;
    var ref_dx = 0.0;
    var ref_ddx = 0.0;
    var ref_gaussian = 0.0;
    var test_dx = 0.0;
    var test_ddx = 0.0;
    var test_gaussian = 0.0;
    for (var offset = -i32(params.feature_radius); offset <= i32(params.feature_radius); offset += 1) {
        let x = clamp_x(i32(id.x) + offset);
        let sample_index = (id.y * params.width + x) * 2u;
        let weight = filters[u32(offset + i32(params.feature_radius))];
        let ref_y = source[sample_index].x * (1.0 / 116.0) + (16.0 / 116.0);
        let test_y = source[sample_index + 1u].x * (1.0 / 116.0) + (16.0 / 116.0);
        ref_dx += weight.y * ref_y;
        ref_ddx += weight.z * ref_y;
        ref_gaussian += weight.x * ref_y;
        test_dx += weight.y * test_y;
        test_ddx += weight.z * test_y;
        test_gaussian += weight.x * test_y;
    }
    destination[pixel_index * 2u] = vec4<f32>(ref_dx, ref_ddx, ref_gaussian, 0.0);
    destination[pixel_index * 2u + 1u] = vec4<f32>(test_dx, test_ddx, test_gaussian, 0.0);
}

@compute @workgroup_size(8, 8)
fn features_y_and_error(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let pixel_index = id.y * params.width + id.x;
    var ref_dx = 0.0;
    var ref_ddx = 0.0;
    var ref_dy = 0.0;
    var ref_ddy = 0.0;
    var test_dx = 0.0;
    var test_ddx = 0.0;
    var test_dy = 0.0;
    var test_ddy = 0.0;
    for (var offset = -i32(params.feature_radius); offset <= i32(params.feature_radius); offset += 1) {
        let y = clamp_y(i32(id.y) + offset);
        let sample_index = (y * params.width + id.x) * 2u;
        let weight = filters[u32(offset + i32(params.feature_radius))];
        let ref_features = source[sample_index];
        let test_features = source[sample_index + 1u];
        ref_dx += weight.x * ref_features.x;
        ref_ddx += weight.x * ref_features.y;
        ref_dy += weight.y * ref_features.z;
        ref_ddy += weight.z * ref_features.z;
        test_dx += weight.x * test_features.x;
        test_ddx += weight.x * test_features.y;
        test_dy += weight.y * test_features.z;
        test_ddy += weight.z * test_features.z;
    }
    let edge_ref = sqrt(ref_dx * ref_dx + ref_dy * ref_dy);
    let edge_test = sqrt(test_dx * test_dx + test_dy * test_dy);
    let point_ref = sqrt(ref_ddx * ref_ddx + ref_ddy * ref_ddy);
    let point_test = sqrt(test_ddx * test_ddx + test_ddy * test_ddy);
    let edge_difference = abs(edge_ref - edge_test);
    let point_difference = abs(point_ref - point_test);
    let feature_difference = pow(
        (1.0 / sqrt(2.0)) * max(edge_difference, point_difference),
        params.gqf,
    );
    let error = pow(color_difference[pixel_index].x, 1.0 - feature_difference);
    destination[pixel_index] = vec4<f32>(error, 0.0, 0.0, 0.0);
}
