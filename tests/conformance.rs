use std::{error::Error, fs::File, io::BufReader, path::Path};

use flip_wgpu::{
    DEFAULT_PIXELS_PER_DEGREE, FlipEvaluator, FlipOptions, HdrOptions, ToneMapper, palettes,
};
use wgpu::util::DeviceExt;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const LDR_MEAN: f32 = 0.159691;
const HDR_MEAN: f32 = 0.283478;

#[test]
fn bundled_ldr_and_hdr_match_flip_reference_results() {
    let result = pollster::block_on(run_conformance());
    if let Err(error) = result {
        panic!("FLIP conformance check failed: {error}");
    }
}

async fn run_conformance() -> Result<(), Box<dyn Error>> {
    let instance = wgpu::Instance::default();
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
    {
        Ok(adapter) => adapter,
        Err(_) => match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                force_fallback_adapter: true,
                ..Default::default()
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(error) => {
                eprintln!("skipping GPU conformance test: no wgpu adapter is available ({error})");
                return Ok(());
            }
        },
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    let (reference_rgb, test_rgb) = (
        load_rgb8(&fixtures.join("reference.png"))?,
        load_rgb8(&fixtures.join("test.png"))?,
    );
    let expected_ldr = load_rgb8(&fixtures.join("correct_ldrflip_cpp.png"))?;
    let mut ldr = FlipEvaluator::new(&device, &queue, WIDTH, HEIGHT, FlipOptions::default())?;
    let ldr_result = ldr.evaluate_ldr(&reference_rgb, &test_rgb)?;
    assert_mean("LDR", &ldr_result.errors, LDR_MEAN);
    assert_error_map("LDR", &ldr_result.errors, &expected_ldr);

    let (reference_hdr, test_hdr) = (
        load_rgba32f(&fixtures.join("reference.exr"))?,
        load_rgba32f(&fixtures.join("test.exr"))?,
    );
    let expected_hdr = load_rgb8(&fixtures.join("correct_hdrflip_cpp.png"))?;
    let mut hdr = FlipEvaluator::new(
        &device,
        &queue,
        WIDTH,
        HEIGHT,
        FlipOptions {
            pixels_per_degree: DEFAULT_PIXELS_PER_DEGREE,
            tone_mapper: ToneMapper::Aces,
        },
    )?;
    let hdr_result = hdr.evaluate_hdr(&reference_hdr, &test_hdr, HdrOptions::default())?;
    assert_mean("HDR", &hdr_result.errors, HDR_MEAN);
    assert_error_map("HDR", &hdr_result.errors, &expected_hdr);
    assert_eq!(hdr_result.exposures.len(), (WIDTH * HEIGHT) as usize);

    let small_reference = [[32, 96, 180], [240, 220, 180], [40, 50, 60], [255, 200, 10]];
    let small_test = [[32, 96, 180], [220, 205, 170], [40, 48, 57], [245, 185, 12]];
    let reference_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("conformance RGB reference"),
        contents: &pack_rgb8(&small_reference),
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let test_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("conformance RGB test"),
        contents: &pack_rgb8(&small_test),
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let mut buffer_ldr = FlipEvaluator::new(&device, &queue, 2, 2, FlipOptions::default())?;
    let slice_ldr = buffer_ldr.evaluate_ldr(&small_reference, &small_test)?;
    let gpu_ldr = buffer_ldr.evaluate_ldr_buffers(&reference_buffer, &test_buffer)?;
    assert_eq!(slice_ldr.errors, gpu_ldr.errors);
    let reference_texture = create_rgba8_texture(&device, &queue, &small_reference);
    let test_texture = create_rgba8_texture(&device, &queue, &small_test);
    let texture_ldr = buffer_ldr.evaluate_ldr_textures(&reference_texture, &test_texture)?;
    assert_eq!(slice_ldr.errors, texture_ldr.errors);

    let small_hdr_reference = [
        [0.1, 0.2, 0.3, 1.0],
        [1.0, 0.8, 0.4, 1.0],
        [0.4, 0.5, 0.6, 1.0],
        [2.0, 1.3, 0.8, 1.0],
    ];
    let small_hdr_test = [
        [0.1, 0.2, 0.3, 1.0],
        [0.9, 0.7, 0.35, 1.0],
        [0.38, 0.48, 0.59, 1.0],
        [1.8, 1.2, 0.75, 1.0],
    ];
    let reference_hdr_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("conformance HDR reference"),
        contents: &pack_rgba32f(&small_hdr_reference),
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let test_hdr_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("conformance HDR test"),
        contents: &pack_rgba32f(&small_hdr_test),
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let explicit_exposures = HdrOptions {
        start_exposure: Some(-1.0),
        stop_exposure: Some(1.0),
        exposure_count: Some(3),
    };
    let mut buffer_hdr = FlipEvaluator::new(&device, &queue, 2, 2, FlipOptions::default())?;
    let slice_hdr =
        buffer_hdr.evaluate_hdr(&small_hdr_reference, &small_hdr_test, explicit_exposures)?;
    let gpu_hdr = buffer_hdr.evaluate_hdr_buffers(
        &reference_hdr_buffer,
        &test_hdr_buffer,
        explicit_exposures,
    )?;
    assert_eq!(slice_hdr.errors, gpu_hdr.errors);
    assert_eq!(slice_hdr.exposures, gpu_hdr.exposures);
    let reference_hdr_texture = create_rgba32f_texture(&device, &queue, &small_hdr_reference);
    let test_hdr_texture = create_rgba32f_texture(&device, &queue, &small_hdr_test);
    let texture_hdr = buffer_hdr.evaluate_hdr_textures(
        &reference_hdr_texture,
        &test_hdr_texture,
        explicit_exposures,
    )?;
    assert_eq!(slice_hdr.errors, texture_hdr.errors);
    assert_eq!(slice_hdr.exposures, texture_hdr.exposures);
    Ok(())
}

fn create_rgba8_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[[u8; 3]],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("conformance sRGB texture"),
        size: wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let rgba: Vec<u8> = pixels
        .iter()
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(2),
        },
        wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
    );
    texture
}

fn create_rgba32f_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[[f32; 4]],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("conformance linear HDR texture"),
        size: wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pack_rgba32f(pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(32),
            rows_per_image: Some(2),
        },
        wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
    );
    texture
}

fn pack_rgb8(pixels: &[[u8; 3]]) -> Vec<u8> {
    pixels
        .iter()
        .map(|pixel| u32::from(pixel[0]) | (u32::from(pixel[1]) << 8) | (u32::from(pixel[2]) << 16))
        .flat_map(u32::to_ne_bytes)
        .collect()
}

fn pack_rgba32f(pixels: &[[f32; 4]]) -> Vec<u8> {
    pixels
        .iter()
        .flat_map(|pixel| pixel.iter().flat_map(|component| component.to_ne_bytes()))
        .collect()
}

fn assert_mean(label: &str, errors: &[f32], expected: f32) {
    let mean = errors.iter().copied().sum::<f32>() / errors.len() as f32;
    assert!(
        (mean - expected).abs() <= 0.00001,
        "{label} mean was {mean:.6}, expected {expected:.6}"
    );
}

fn assert_error_map(label: &str, errors: &[f32], expected: &[[u8; 3]]) {
    assert_eq!(errors.len(), expected.len());
    let mut max_channel_delta = 0_u8;
    for (&error, expected_rgb) in errors.iter().zip(expected) {
        let palette_index = (error.clamp(0.0, 1.0) * 255.0 + 0.5) as usize;
        for (channel, expected_channel) in
            palettes::MAGMA[palette_index].into_iter().zip(expected_rgb)
        {
            let actual = (channel * 255.0 + 0.5) as u8;
            max_channel_delta = max_channel_delta.max(actual.abs_diff(*expected_channel));
        }
    }
    assert!(
        max_channel_delta <= 3,
        "{label} map's maximum channel difference was {max_channel_delta}, expected at most 3"
    );
}

fn load_rgb8(path: &Path) -> Result<Vec<[u8; 3]>, Box<dyn Error>> {
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path)?));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let mut decoded = vec![0; reader.output_buffer_size().ok_or("PNG is too large")?];
    let info = reader.next_frame(&mut decoded)?;
    let bytes = &decoded[..info.buffer_size()];
    let pixels = match info.color_type {
        png::ColorType::Rgb => bytes
            .chunks_exact(3)
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect(),
        png::ColorType::Rgba => bytes
            .chunks_exact(4)
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect(),
        png::ColorType::Grayscale => bytes.iter().map(|&v| [v, v, v]).collect(),
        png::ColorType::GrayscaleAlpha => {
            bytes.chunks_exact(2).map(|pixel| [pixel[0]; 3]).collect()
        }
        png::ColorType::Indexed => return Err("indexed PNG was not expanded".into()),
    };
    Ok(pixels)
}

struct ExrPixels {
    width: usize,
    pixels: Vec<[f32; 4]>,
}

fn load_rgba32f(path: &Path) -> Result<Vec<[f32; 4]>, Box<dyn Error>> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| ExrPixels {
            width: resolution.width(),
            pixels: vec![[0.0; 4]; resolution.area()],
        },
        |pixels, position, (red, green, blue, alpha): (f32, f32, f32, f32)| {
            let index = position.y() * pixels.width + position.x();
            pixels.pixels[index] = [red, green, blue, alpha];
        },
    )?;
    Ok(image.layer_data.channel_data.pixels.pixels)
}
