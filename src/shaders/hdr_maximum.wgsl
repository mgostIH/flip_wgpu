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
    exposure_fraction: f32,
};

@group(0) @binding(0) var<storage, read> current_errors: array<f32>;
@group(0) @binding(1) var<storage, read_write> maximum_errors: array<f32>;
@group(0) @binding(2) var<storage, read_write> winning_exposures: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 8)
fn reduce_hdr(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let index = id.y * params.width + id.x;
    let error = current_errors[index];
    if (error > maximum_errors[index]) {
        maximum_errors[index] = error;
        winning_exposures[index] = params.exposure_fraction;
    }
}
