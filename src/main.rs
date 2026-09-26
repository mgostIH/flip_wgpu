mod magma;
mod viridis;

use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    time::{Duration, Instant},
};

use wgpu::util::DeviceExt;

const FLIP_CONSTANTS: FlipConstants = FlipConstants {
    gqc: 0.7,
    gpc: 0.4,
    gpt: 0.95,
    gw: 0.082,
    gqf: 0.5,
};

const DEFAULT_PPD: f32 = 0.7_f32 * (3840.0_f32 / 0.7_f32) * (std::f32::consts::PI / 180.0_f32);

#[derive(Clone, Copy, Debug)]
enum ToneMapper {
    Reinhard,
    Aces,
    Hable,
}

impl ToneMapper {
    fn id(self) -> u32 {
        match self {
            Self::Reinhard => 0,
            Self::Aces => 1,
            Self::Hable => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Reinhard => "reinhard",
            Self::Aces => "aces",
            Self::Hable => "hable",
        }
    }
}

struct CliOptions {
    reference: PathBuf,
    tests: Vec<PathBuf>,
    ppd: f32,
    tone_mapper: ToneMapper,
    start_exposure: Option<f32>,
    stop_exposure: Option<f32>,
    exposure_count: Option<u32>,
    verbosity: u8,
    force_fallback_adapter: bool,
    profile_gpu: bool,
    directory: PathBuf,
    basename: Option<String>,
    explicit_output: Option<PathBuf>,
    magma: bool,
    error_map: bool,
    exposure_map: bool,
    textfile: bool,
    csv: Option<PathBuf>,
}

#[derive(Clone, Copy)]
struct FlipConstants {
    gqc: f32,
    gpc: f32,
    gpt: f32,
    gw: f32,
    gqf: f32,
}

struct RgbImage {
    width: u32,
    height: u32,
    pixels: Vec<[u8; 3]>,
}

struct HdrImage {
    width: u32,
    height: u32,
    pixels: Vec<[f32; 4]>,
}

struct ExrPixels {
    width: usize,
    pixels: Vec<[f32; 4]>,
}

struct FrameResult {
    errors: Vec<f32>,
    elapsed: Duration,
}

struct HdrResult {
    errors: Vec<f32>,
    exposures: Vec<f32>,
    elapsed: Duration,
}

struct GpuProfile {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readback: wgpu::Buffer,
}

struct ErrorStatistics {
    mean: f32,
    weighted_median: f32,
    first_weighted_quartile: f32,
    third_weighted_quartile: f32,
    min: f32,
    max: f32,
}

fn error_statistics(errors: &[f32]) -> ErrorStatistics {
    let sum = errors
        .iter()
        .copied()
        .fold(0.0_f32, |sum, value| sum + value);
    let mut sorted = errors.to_vec();
    sorted.sort_by(f32::total_cmp);

    fn weighted_percentile(sorted: &[f32], sum: f32, percentile: f32) -> f32 {
        let target = percentile * sum;
        let mut running_sum = 0.0_f32;
        for &value in sorted {
            running_sum += value;
            if running_sum > target {
                return value;
            }
        }
        0.0
    }

    ErrorStatistics {
        mean: sum / errors.len() as f32,
        weighted_median: weighted_percentile(&sorted, sum, 0.5),
        first_weighted_quartile: weighted_percentile(&sorted, sum, 0.25),
        third_weighted_quartile: weighted_percentile(&sorted, sum, 0.75),
        min: sorted.first().copied().unwrap_or(0.0),
        max: sorted.last().copied().unwrap_or(0.0),
    }
}

struct FlipGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    spatial_radius: u32,
    feature_radius: u32,
    cmax: f32,
    tone_mapper: ToneMapper,
    gpu_profile: Option<GpuProfile>,
    params: wgpu::Buffer,
    reference_ldr: wgpu::Buffer,
    test_ldr: wgpu::Buffer,
    reference_hdr: wgpu::Buffer,
    test_hdr: wgpu::Buffer,
    error_map: wgpu::Buffer,
    readback: wgpu::Buffer,
    hdr_maximum: wgpu::Buffer,
    hdr_winning_exposure: wgpu::Buffer,
    hdr_readback: wgpu::Buffer,
    ldr_convert_pipeline: wgpu::ComputePipeline,
    hdr_convert_pipeline: wgpu::ComputePipeline,
    spatial_x_pipeline: wgpu::ComputePipeline,
    spatial_y_pipeline: wgpu::ComputePipeline,
    features_x_pipeline: wgpu::ComputePipeline,
    features_y_pipeline: wgpu::ComputePipeline,
    hdr_maximum_pipeline: wgpu::ComputePipeline,
    ldr_convert_group: wgpu::BindGroup,
    hdr_convert_group: wgpu::BindGroup,
    spatial_x_group: wgpu::BindGroup,
    spatial_y_group: wgpu::BindGroup,
    features_x_group: wgpu::BindGroup,
    features_y_group: wgpu::BindGroup,
    hdr_maximum_group: wgpu::BindGroup,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        return;
    }
    let options = match parse_cli(&args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}\n");
            print_usage();
            std::process::exit(2);
        }
    };

    if let Err(error) = pollster::block_on(run(options)) {
        eprintln!("wgpu FLIP failed: {error}");
        std::process::exit(1);
    }
}

fn print_usage() {
    println!(
        "{}",
        format!(
            r#"wgpu FLIP: compare PNG (LDR) or EXR (HDR) images

Usage:
  wgpu_test <reference.png|reference.exr> <test> <output.png>
  wgpu_test --reference REF --test TEST [TEST ...] [options]

Options:
  -h, --help                     Show this help
  -r, --reference PATH           Reference image
  -t, --test PATH [PATH ...]     Test image(s), repeatable
      --pixels-per-degree N      Viewing scale (default: {ppd:.6})
      --viewing-conditions D W PX  Monitor distance, width, horizontal pixels
      --tone-mapper NAME         HDR: aces (default), hable, or reinhard
      --start-exposure N         Override automatic HDR start exposure
      --stop-exposure N          Override automatic HDR stop exposure
      --num-exposures N          HDR exposure count (minimum 2)
  -d, --directory PATH           Output directory (default: .)
  -b, --basename NAME            Output name; supports one test image
      --output PATH              Explicit error-map path (one test only)
  -v, --verbosity 0|1|2          0 quiet, 1 mean, 2 full summary (default: 2)
      --force-fallback-adapter   Prefer a software/fallback adapter
      --profile-gpu              Print GPU time for each compute pass
      --no-error-map             Do not write the FLIP map
      --no-exposure-map          Do not write the HDR exposure map
      --no-magma                 Use grayscale for the FLIP map
      --textfile                 Write pooled statistics beside the FLIP map
      --csv PATH                 Append pooled statistics to a CSV file

HDR defaults to an automatic exposure range based on the reference image."#,
            ppd = DEFAULT_PPD
        )
    );
}

fn parse_cli(args: &[String]) -> Result<CliOptions, String> {
    if args.len() == 3 && !args[0].starts_with('-') {
        return Ok(CliOptions {
            reference: PathBuf::from(&args[0]),
            tests: vec![PathBuf::from(&args[1])],
            ppd: DEFAULT_PPD,
            tone_mapper: ToneMapper::Aces,
            start_exposure: None,
            stop_exposure: None,
            exposure_count: None,
            verbosity: 2,
            force_fallback_adapter: false,
            profile_gpu: false,
            directory: PathBuf::from("."),
            basename: None,
            explicit_output: Some(PathBuf::from(&args[2])),
            magma: true,
            error_map: true,
            exposure_map: true,
            textfile: false,
            csv: None,
        });
    }

    let mut reference = None;
    let mut tests = Vec::new();
    let mut ppd = DEFAULT_PPD;
    let mut tone_mapper = ToneMapper::Aces;
    let mut start_exposure = None;
    let mut stop_exposure = None;
    let mut exposure_count = None;
    let mut verbosity = 2;
    let mut force_fallback_adapter = false;
    let mut profile_gpu = false;
    let mut directory = PathBuf::from(".");
    let mut basename = None;
    let mut explicit_output = None;
    let mut magma = true;
    let mut error_map = true;
    let mut exposure_map = true;
    let mut textfile = false;
    let mut csv = None;
    let mut index = 0;

    while index < args.len() {
        let option = args[index].as_str();
        index += 1;
        match option {
            "-r" | "--reference" => {
                reference = Some(PathBuf::from(cli_value(args, &mut index, option)?));
            }
            "-t" | "--test" => {
                let first = index;
                while index < args.len() && !args[index].starts_with('-') {
                    tests.push(PathBuf::from(&args[index]));
                    index += 1;
                }
                if index == first {
                    return Err("--test requires at least one image path".into());
                }
            }
            "-ppd" | "--pixels-per-degree" => {
                ppd = parse_number(cli_value(args, &mut index, option)?, option)?;
            }
            "-vc" | "--viewing-conditions" => {
                let distance: f32 = parse_number(cli_value(args, &mut index, option)?, option)?;
                let width: f32 = parse_number(cli_value(args, &mut index, option)?, option)?;
                let pixels: f32 = parse_number(cli_value(args, &mut index, option)?, option)?;
                if distance <= 0.0 || width <= 0.0 || pixels <= 0.0 {
                    return Err("viewing-condition values must all be positive".into());
                }
                ppd = distance * (pixels / width) * (std::f32::consts::PI / 180.0);
            }
            "-tm" | "--tone-mapper" => {
                let value = cli_value(args, &mut index, option)?.to_ascii_lowercase();
                tone_mapper = match value.as_str() {
                    "reinhard" => ToneMapper::Reinhard,
                    "aces" => ToneMapper::Aces,
                    "hable" => ToneMapper::Hable,
                    _ => {
                        return Err(format!(
                            "unknown tone mapper `{value}` (choose aces, hable, or reinhard)"
                        ));
                    }
                };
            }
            "-n" | "--num-exposures" => {
                let count: u32 = parse_number(cli_value(args, &mut index, option)?, option)?;
                if count < 2 {
                    return Err("--num-exposures must be at least 2".into());
                }
                exposure_count = Some(count);
            }
            "-cstart" | "--start-exposure" => {
                start_exposure = Some(parse_number(cli_value(args, &mut index, option)?, option)?);
            }
            "-cstop" | "--stop-exposure" => {
                stop_exposure = Some(parse_number(cli_value(args, &mut index, option)?, option)?);
            }
            "-v" | "--verbosity" => {
                verbosity = parse_number(cli_value(args, &mut index, option)?, option)?;
                if verbosity > 2 {
                    return Err("--verbosity must be 0, 1, or 2".into());
                }
            }
            "--force-fallback-adapter" => force_fallback_adapter = true,
            "--profile-gpu" => profile_gpu = true,
            "-d" | "--directory" => {
                directory = PathBuf::from(cli_value(args, &mut index, option)?);
            }
            "-b" | "--basename" => {
                basename = Some(cli_value(args, &mut index, option)?.to_owned());
            }
            "--output" => {
                explicit_output = Some(PathBuf::from(cli_value(args, &mut index, option)?));
            }
            "--no-magma" | "-nm" => magma = false,
            "--no-error-map" | "-nerm" => error_map = false,
            "--no-exposure-map" | "-nexm" => exposure_map = false,
            "--textfile" | "-txt" => textfile = true,
            "--csv" | "-c" => csv = Some(PathBuf::from(cli_value(args, &mut index, option)?)),
            _ => return Err(format!("unknown option or argument `{option}`")),
        }
    }

    if !ppd.is_finite() || ppd <= 0.0 {
        return Err("pixels per degree must be a finite positive number".into());
    }
    if reference.is_none() || tests.is_empty() {
        return Err("both --reference and at least one --test image are required".into());
    }
    if tests.len() > 1 && (basename.is_some() || explicit_output.is_some()) {
        return Err("--basename and --output require exactly one test image".into());
    }

    Ok(CliOptions {
        reference: reference.expect("checked above"),
        tests,
        ppd,
        tone_mapper,
        start_exposure,
        stop_exposure,
        exposure_count,
        verbosity,
        force_fallback_adapter,
        profile_gpu,
        directory,
        basename,
        explicit_output,
        magma,
        error_map,
        exposure_map,
        textfile,
        csv,
    })
}

fn cli_value<'a>(args: &'a [String], index: &mut usize, option: &str) -> Result<&'a str, String> {
    let value = args
        .get(*index)
        .ok_or_else(|| format!("{option} requires a value"))?;
    *index += 1;
    Ok(value)
}

fn parse_number<T: std::str::FromStr>(value: &str, option: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value `{value}` for {option}"))
}

async fn run(options: CliOptions) -> Result<(), Box<dyn Error>> {
    for test_path in &options.tests {
        run_pair(&options, test_path).await?;
    }
    Ok(())
}

async fn run_pair(options: &CliOptions, test_path: &Path) -> Result<(), Box<dyn Error>> {
    let reference_path = &options.reference;
    let hdr = path_has_extension(reference_path, "exr");
    if !hdr && !path_has_extension(reference_path, "png") {
        return Err(format!(
            "unsupported reference format: {} (expected PNG or EXR)",
            reference_path.display()
        )
        .into());
    }
    if path_has_extension(test_path, "exr") != hdr
        || (!hdr && !path_has_extension(test_path, "png"))
    {
        return Err(format!(
            "test image format must match the reference: {}",
            test_path.display()
        )
        .into());
    }
    let reference_path_text = reference_path.to_string_lossy();
    let test_path_text = test_path.to_string_lossy();

    let (width, height, errors, exposure_values, evaluation_time, hdr_range) = if hdr {
        let reference = load_exr(&reference_path_text)?;
        let test = load_exr(&test_path_text)?;
        ensure_same_size(reference.width, reference.height, test.width, test.height)?;
        let gpu = FlipGpu::new(
            reference.width,
            reference.height,
            options.ppd,
            options.tone_mapper,
            options.verbosity,
            options.force_fallback_adapter,
            options.profile_gpu,
        )
        .await?;
        gpu.upload_hdr(&reference, &test);
        let (automatic_start, automatic_stop, _) = hdr_exposures(&reference, options.tone_mapper);
        let start = options.start_exposure.unwrap_or(automatic_start);
        let stop = options.stop_exposure.unwrap_or(automatic_stop);
        if !start.is_finite() || !stop.is_finite() || start > stop {
            return Err(
                "HDR start exposure must be finite and no greater than stop exposure".into(),
            );
        }
        let exposure_count = options
            .exposure_count
            .unwrap_or_else(|| ((stop - start).ceil() as u32).max(2));
        if options.verbosity >= 2 {
            println!("HDR tone mapper: {}", options.tone_mapper.name());
            println!("HDR exposures: {exposure_count} from {start:.6} to {stop:.6} stops");
        }
        let result = gpu.compute_hdr_range(start, stop, exposure_count)?;
        (
            reference.width,
            reference.height,
            result.errors,
            Some(result.exposures),
            result.elapsed,
            Some((start, stop, exposure_count)),
        )
    } else {
        let reference = load_png(&reference_path_text)?;
        let test = load_png(&test_path_text)?;
        ensure_same_size(reference.width, reference.height, test.width, test.height)?;
        let gpu = FlipGpu::new(
            reference.width,
            reference.height,
            options.ppd,
            options.tone_mapper,
            options.verbosity,
            options.force_fallback_adapter,
            options.profile_gpu,
        )
        .await?;
        gpu.upload_ldr(&reference, &test);
        let frame = gpu.compute_ldr()?;
        (
            reference.width,
            reference.height,
            frame.errors,
            None,
            frame.elapsed,
            None,
        )
    };

    let stats = error_statistics(&errors);
    let output_path = error_map_path(options, test_path, hdr_range);
    if options.error_map {
        if let Some(parent) = output_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        save_error_png(&output_path, width, height, &errors, options.magma)?;
    }
    if hdr && options.exposure_map {
        let exposure_path = exposure_map_path(&output_path);
        if let Some(values) = exposure_values.as_deref() {
            if let Some(parent) = exposure_path
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
            {
                fs::create_dir_all(parent)?;
            }
            save_viridis_png(&exposure_path, width, height, values)?;
        }
    }
    if options.textfile {
        let stats_path = output_path.with_extension("txt");
        if let Some(parent) = stats_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        write_text_stats(&stats_path, &stats, evaluation_time, options.ppd, hdr_range)?;
    }
    if let Some(csv_path) = &options.csv {
        if let Some(parent) = csv_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        append_csv_stats(csv_path, reference_path, test_path, &stats, evaluation_time)?;
    }

    if options.verbosity >= 1 {
        println!("Mean: {:.6}", stats.mean);
    }
    if options.verbosity >= 2 {
        println!("Test: {}", test_path.display());
        println!("Weighted median: {:.6}", stats.weighted_median);
        println!(
            "1st weighted quartile: {:.6}",
            stats.first_weighted_quartile
        );
        println!(
            "3rd weighted quartile: {:.6}",
            stats.third_weighted_quartile
        );
        println!("Min: {:.6}", stats.min);
        println!("Max: {:.6}", stats.max);
        println!(
            "Evaluation time: {:.4} seconds",
            evaluation_time.as_secs_f32()
        );
        if options.error_map {
            println!("Error map: {}", output_path.display());
        }
        if hdr && options.exposure_map {
            println!(
                "Exposure map: {}",
                exposure_map_path(&output_path).display()
            );
        }
    }
    Ok(())
}

fn path_has_extension(path: &Path, extension: &str) -> bool {
    path.extension()
        .is_some_and(|value| value.eq_ignore_ascii_case(extension))
}

fn error_map_path(options: &CliOptions, test_path: &Path, hdr: Option<(f32, f32, u32)>) -> PathBuf {
    if let Some(path) = &options.explicit_output {
        return path.clone();
    }
    if let Some(name) = &options.basename {
        return options.directory.join(format!("{name}.png"));
    }
    let reference_stem = options
        .reference
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let test_stem = test_path.file_stem().unwrap_or_default().to_string_lossy();
    let mode = if let Some((start, stop, count)) = hdr {
        format!(
            "hdr.{}.{}.to.{}.{}",
            options.tone_mapper.name(),
            filename_number(start),
            filename_number(stop),
            count
        )
    } else {
        "ldr".to_owned()
    };
    options.directory.join(format!(
        "flip.{reference_stem}.{test_stem}.{:.0}ppd.{mode}.png",
        options.ppd
    ))
}

fn filename_number(value: f32) -> String {
    format!("{value:.4}").replace('-', "m").replace('.', "p")
}

fn exposure_map_path(error_map_path: &Path) -> PathBuf {
    let stem = error_map_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    error_map_path.with_file_name(format!("{stem}.exposure_map.png"))
}

fn write_text_stats(
    path: &Path,
    stats: &ErrorStatistics,
    evaluation_time: Duration,
    ppd: f32,
    hdr_range: Option<(f32, f32, u32)>,
) -> Result<(), Box<dyn Error>> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "Mean: {:.6}", stats.mean)?;
    writeln!(file, "Weighted median: {:.6}", stats.weighted_median)?;
    writeln!(
        file,
        "1st weighted quartile: {:.6}",
        stats.first_weighted_quartile
    )?;
    writeln!(
        file,
        "3rd weighted quartile: {:.6}",
        stats.third_weighted_quartile
    )?;
    writeln!(file, "Min: {:.6}", stats.min)?;
    writeln!(file, "Max: {:.6}", stats.max)?;
    writeln!(file, "Pixels per degree: {ppd:.6}")?;
    if let Some((start, stop, count)) = hdr_range {
        writeln!(file, "Start exposure: {start:.6}")?;
        writeln!(file, "Stop exposure: {stop:.6}")?;
        writeln!(file, "Number of exposures: {count}")?;
    }
    writeln!(
        file,
        "Evaluation time: {:.4} seconds",
        evaluation_time.as_secs_f32()
    )?;
    Ok(())
}

fn append_csv_stats(
    path: &Path,
    reference: &Path,
    test: &Path,
    stats: &ErrorStatistics,
    evaluation_time: Duration,
) -> Result<(), Box<dyn Error>> {
    let needs_header = !path.exists() || path.metadata()?.len() == 0;
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if needs_header {
        writeln!(
            file,
            "Reference,Test,Mean,Weighted median,1st weighted quartile,3rd weighted quartile,Min,Max,Evaluation time"
        )?;
    }
    writeln!(
        file,
        "\"{}\",\"{}\",{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4}",
        csv_escape(&reference.display().to_string()),
        csv_escape(&test.display().to_string()),
        stats.mean,
        stats.weighted_median,
        stats.first_weighted_quartile,
        stats.third_weighted_quartile,
        stats.min,
        stats.max,
        evaluation_time.as_secs_f32()
    )?;
    Ok(())
}

fn csv_escape(value: &str) -> String {
    value.replace('"', "\"\"")
}

impl FlipGpu {
    async fn new(
        width: u32,
        height: u32,
        ppd: f32,
        tone_mapper: ToneMapper,
        verbosity: u8,
        force_fallback_adapter: bool,
        profile_gpu: bool,
    ) -> Result<Self, Box<dyn Error>> {
        if width == 0 || height == 0 {
            return Err("image dimensions must be nonzero".into());
        }
        let pixel_count = width as usize * height as usize;
        let spatial_radius = spatial_filter_radius(ppd);
        let feature_radius = feature_filter_radius(ppd);
        let cmax = compute_max_distance(FLIP_CONSTANTS.gqc);
        let params_bytes = params_bytes(
            width,
            height,
            spatial_radius,
            feature_radius,
            cmax,
            0.0,
            tone_mapper,
            0.0,
        );

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                force_fallback_adapter,
                ..Default::default()
            })
            .await?;
        let adapter_info = adapter.get_info();
        if profile_gpu && !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err("this adapter does not support GPU timestamp queries".into());
        }
        if verbosity >= 2 {
            println!(
                "Adapter: {} ({:?}, {:?})",
                adapter_info.name, adapter_info.device_type, adapter_info.backend
            );
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: if profile_gpu {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                ..Default::default()
            })
            .await?;
        let gpu_profile = if profile_gpu {
            Some(GpuProfile {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("FLIP pass timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 12,
                }),
                resolved: empty_buffer(
                    &device,
                    "FLIP resolved timestamps",
                    96,
                    wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                ),
                readback: empty_buffer(
                    &device,
                    "FLIP timestamp readback",
                    96,
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                ),
            })
        } else {
            None
        };

        let reference_ldr = empty_buffer(
            &device,
            "reference PNG pixels",
            pixel_count * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let test_ldr = empty_buffer(
            &device,
            "test PNG pixels",
            pixel_count * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let reference_hdr = empty_buffer(
            &device,
            "reference EXR pixels",
            pixel_count * 16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let test_hdr = empty_buffer(
            &device,
            "test EXR pixels",
            pixel_count * 16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let source_colors = empty_buffer(
            &device,
            "YCxCz input images",
            pixel_count * 2 * 16,
            wgpu::BufferUsages::STORAGE,
        );
        let spatial_intermediate = empty_buffer(
            &device,
            "spatial filter intermediate",
            pixel_count * 2 * 16,
            wgpu::BufferUsages::STORAGE,
        );
        let color_difference = empty_buffer(
            &device,
            "color difference",
            pixel_count * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let feature_intermediate = empty_buffer(
            &device,
            "feature filter intermediate",
            pixel_count * 2 * 16,
            wgpu::BufferUsages::STORAGE,
        );
        let error_map = empty_buffer(
            &device,
            "FLIP error map",
            pixel_count * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = empty_buffer(
            &device,
            "FLIP error map readback",
            pixel_count * 4,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let hdr_maximum = empty_buffer(
            &device,
            "HDR maximum errors",
            pixel_count * 4,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let hdr_winning_exposure = empty_buffer(
            &device,
            "HDR winning exposure",
            pixel_count * 4,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let hdr_readback = empty_buffer(
            &device,
            "HDR result readback",
            pixel_count * 8,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FLIP parameters"),
            contents: &params_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let spatial_filter_buffer = storage_buffer(
            &device,
            "FLIP spatial filters",
            &vec4_bytes(&spatial_filter(ppd, spatial_radius)),
        );
        let feature_filter_buffer = storage_buffer(
            &device,
            "FLIP feature filter",
            &vec4_bytes(&feature_filter(ppd, feature_radius)),
        );

        let ldr_convert_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FLIP LDR input conversion shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/convert.wgsl").into()),
        });
        let hdr_convert_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FLIP HDR exposure and input conversion shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/convert_hdr.wgsl").into()),
        });
        let spatial_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FLIP spatial difference shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/spatial.wgsl").into()),
        });
        let features_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FLIP feature difference shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/features.wgsl").into()),
        });
        let hdr_maximum_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FLIP HDR maximum shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hdr_maximum.wgsl").into()),
        });

        let ldr_convert_pipeline = compute_pipeline(
            &device,
            &ldr_convert_module,
            "convert",
            "FLIP LDR input conversion",
        );
        let hdr_convert_pipeline = compute_pipeline(
            &device,
            &hdr_convert_module,
            "convert_hdr",
            "FLIP HDR tone mapping and input conversion",
        );
        let spatial_x_pipeline = compute_pipeline(
            &device,
            &spatial_module,
            "spatial_x",
            "FLIP spatial X filter",
        );
        let spatial_y_pipeline = compute_pipeline(
            &device,
            &spatial_module,
            "spatial_y",
            "FLIP spatial Y filter and color difference",
        );
        let features_x_pipeline = compute_pipeline(
            &device,
            &features_module,
            "features_x",
            "FLIP feature X filter",
        );
        let features_y_pipeline = compute_pipeline(
            &device,
            &features_module,
            "features_y_and_error",
            "FLIP feature Y filter and final error",
        );
        let hdr_maximum_pipeline = compute_pipeline(
            &device,
            &hdr_maximum_module,
            "reduce_hdr",
            "FLIP HDR maximum error and exposure",
        );

        let ldr_convert_group = bind_group(
            &device,
            &ldr_convert_pipeline,
            &[
                (0, &reference_ldr),
                (1, &test_ldr),
                (2, &source_colors),
                (3, &params),
            ],
        );
        let hdr_convert_group = bind_group(
            &device,
            &hdr_convert_pipeline,
            &[
                (0, &reference_hdr),
                (1, &test_hdr),
                (2, &source_colors),
                (3, &params),
            ],
        );
        let spatial_x_group = bind_group(
            &device,
            &spatial_x_pipeline,
            &[
                (0, &source_colors),
                (1, &spatial_filter_buffer),
                (2, &spatial_intermediate),
                (3, &params),
            ],
        );
        let spatial_y_group = bind_group(
            &device,
            &spatial_y_pipeline,
            &[
                (0, &spatial_intermediate),
                (1, &spatial_filter_buffer),
                (4, &color_difference),
                (3, &params),
            ],
        );
        let features_x_group = bind_group(
            &device,
            &features_x_pipeline,
            &[
                (0, &source_colors),
                (1, &feature_filter_buffer),
                (3, &feature_intermediate),
                (4, &params),
            ],
        );
        let features_y_group = bind_group(
            &device,
            &features_y_pipeline,
            &[
                (0, &feature_intermediate),
                (1, &feature_filter_buffer),
                (2, &color_difference),
                (5, &error_map),
                (4, &params),
            ],
        );
        let hdr_maximum_group = bind_group(
            &device,
            &hdr_maximum_pipeline,
            &[
                (0, &error_map),
                (1, &hdr_maximum),
                (2, &hdr_winning_exposure),
                (3, &params),
            ],
        );

        if verbosity >= 2 {
            println!(
                "PPD: {ppd:.6} (spatial radius {spatial_radius}, feature radius {feature_radius})"
            );
        }

        Ok(Self {
            device,
            queue,
            width,
            height,
            spatial_radius,
            feature_radius,
            cmax,
            tone_mapper,
            gpu_profile,
            params,
            reference_ldr,
            test_ldr,
            reference_hdr,
            test_hdr,
            error_map,
            readback,
            hdr_maximum,
            hdr_winning_exposure,
            hdr_readback,
            ldr_convert_pipeline,
            hdr_convert_pipeline,
            spatial_x_pipeline,
            spatial_y_pipeline,
            features_x_pipeline,
            features_y_pipeline,
            hdr_maximum_pipeline,
            ldr_convert_group,
            hdr_convert_group,
            spatial_x_group,
            spatial_y_group,
            features_x_group,
            features_y_group,
            hdr_maximum_group,
        })
    }

    fn upload_ldr(&self, reference: &RgbImage, test: &RgbImage) {
        self.queue.write_buffer(
            &self.reference_ldr,
            0,
            &u32_bytes(&pack_pixels(&reference.pixels)),
        );
        self.queue
            .write_buffer(&self.test_ldr, 0, &u32_bytes(&pack_pixels(&test.pixels)));
    }

    fn upload_hdr(&self, reference: &HdrImage, test: &HdrImage) {
        self.queue
            .write_buffer(&self.reference_hdr, 0, &vec4_bytes(&reference.pixels));
        self.queue
            .write_buffer(&self.test_hdr, 0, &vec4_bytes(&test.pixels));
    }

    fn compute_ldr(&self) -> Result<FrameResult, Box<dyn Error>> {
        self.compute_frame()
    }

    fn compute_hdr_range(
        &self,
        start: f32,
        stop: f32,
        count: u32,
    ) -> Result<HdrResult, Box<dyn Error>> {
        let started = Instant::now();
        let pixel_bytes = self.width as u64 * self.height as u64 * 4;
        let step = (stop - start) / (count - 1) as f32;
        for index in 0..count {
            let exposure = start + index as f32 * step;
            let fraction = index as f32 / (count - 1) as f32;
            self.queue.write_buffer(
                &self.params,
                0,
                &params_bytes(
                    self.width,
                    self.height,
                    self.spatial_radius,
                    self.feature_radius,
                    self.cmax,
                    exposure,
                    self.tone_mapper,
                    fraction,
                ),
            );

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("FLIP HDR exposure encoder"),
                });
            if index == 0 {
                encoder.clear_buffer(&self.hdr_maximum, 0, None);
                encoder.clear_buffer(&self.hdr_winning_exposure, 0, None);
            }
            let profile_last_exposure = index + 1 == count;
            self.encode_frame_passes(&mut encoder, true, profile_last_exposure);
            dispatch(
                &mut encoder,
                &self.hdr_maximum_pipeline,
                &self.hdr_maximum_group,
                self.width.div_ceil(16),
                self.height.div_ceil(8),
                if profile_last_exposure {
                    self.gpu_profile
                        .as_ref()
                        .map(|profile| (&profile.queries, 10))
                } else {
                    None
                },
            );
            if profile_last_exposure {
                if let Some(profile) = &self.gpu_profile {
                    encoder.resolve_query_set(&profile.queries, 0..12, &profile.resolved, 0);
                    encoder.copy_buffer_to_buffer(&profile.resolved, 0, &profile.readback, 0, 96);
                }
                encoder.copy_buffer_to_buffer(
                    &self.hdr_maximum,
                    0,
                    &self.hdr_readback,
                    0,
                    pixel_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    &self.hdr_winning_exposure,
                    0,
                    &self.hdr_readback,
                    pixel_bytes,
                    pixel_bytes,
                );
            }
            self.queue.submit([encoder.finish()]);
        }

        let (sender, receiver) = mpsc::channel();
        self.hdr_readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result);
            });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv()??;
        let mapped = self.hdr_readback.get_mapped_range(..)?;
        let (errors_bytes, exposures_bytes) = mapped.split_at(pixel_bytes as usize);
        let errors = errors_bytes
            .chunks_exact(4)
            .map(|pixel| f32::from_ne_bytes(pixel.try_into().expect("f32 bytes")))
            .collect();
        let exposures = exposures_bytes
            .chunks_exact(4)
            .map(|pixel| f32::from_ne_bytes(pixel.try_into().expect("f32 bytes")))
            .collect();
        drop(mapped);
        self.hdr_readback.unmap();
        self.print_gpu_profile(true)?;
        Ok(HdrResult {
            errors,
            exposures,
            elapsed: started.elapsed(),
        })
    }

    fn compute_frame(&self) -> Result<FrameResult, Box<dyn Error>> {
        let started = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FLIP compute encoder"),
            });
        self.encode_frame_passes(&mut encoder, false, true);
        if let Some(profile) = &self.gpu_profile {
            encoder.resolve_query_set(&profile.queries, 0..10, &profile.resolved, 0);
            encoder.copy_buffer_to_buffer(&profile.resolved, 0, &profile.readback, 0, 80);
        }
        let buffer_size = (self.width as u64) * (self.height as u64) * 4;
        encoder.copy_buffer_to_buffer(&self.error_map, 0, &self.readback, 0, buffer_size);
        self.queue.submit([encoder.finish()]);

        let (sender, receiver) = mpsc::channel();
        self.readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result);
            });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv()??;

        let mapped = self.readback.get_mapped_range(..)?;
        let errors = mapped
            .chunks_exact(4)
            .map(|pixel| f32::from_ne_bytes(pixel[0..4].try_into().expect("f32 bytes")))
            .collect();
        drop(mapped);
        self.readback.unmap();
        self.print_gpu_profile(false)?;
        Ok(FrameResult {
            errors,
            elapsed: started.elapsed(),
        })
    }

    fn print_gpu_profile(&self, hdr: bool) -> Result<(), Box<dyn Error>> {
        if let Some(profile) = &self.gpu_profile {
            let (sender, receiver) = mpsc::channel();
            profile
                .readback
                .map_async(wgpu::MapMode::Read, .., move |result| {
                    let _ = sender.send(result);
                });
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
            receiver.recv()??;
            let timestamps = profile.readback.get_mapped_range(..)?;
            let ticks: Vec<u64> = timestamps
                .chunks_exact(8)
                .map(|bytes| u64::from_ne_bytes(bytes.try_into().expect("timestamp bytes")))
                .collect();
            drop(timestamps);
            profile.readback.unmap();
            let period = self.queue.get_timestamp_period();
            let mut names = vec![
                "convert",
                "spatial X",
                "spatial Y",
                "features X",
                "features Y",
            ];
            if hdr {
                names.push("HDR maximum");
                println!("GPU pass times for the final HDR exposure:");
            }
            let mut total_ms = 0.0;
            for (index, name) in names.iter().enumerate() {
                let elapsed_ms = ticks[index * 2 + 1].saturating_sub(ticks[index * 2]) as f32
                    * period
                    / 1_000_000.0;
                total_ms += elapsed_ms;
                println!("GPU pass {name}: {elapsed_ms:.3} ms");
            }
            println!("GPU pass sum: {total_ms:.3} ms");
        }
        Ok(())
    }

    fn encode_frame_passes(&self, encoder: &mut wgpu::CommandEncoder, hdr: bool, profile: bool) {
        let (convert_pipeline, convert_group) = if hdr {
            (&self.hdr_convert_pipeline, &self.hdr_convert_group)
        } else {
            (&self.ldr_convert_pipeline, &self.ldr_convert_group)
        };
        let workgroups_x = self.width.div_ceil(16);
        let workgroups_y = self.height.div_ceil(8);
        for (index, (pipeline, group)) in [
            (convert_pipeline, convert_group),
            (&self.spatial_x_pipeline, &self.spatial_x_group),
            (&self.spatial_y_pipeline, &self.spatial_y_group),
            (&self.features_x_pipeline, &self.features_x_group),
            (&self.features_y_pipeline, &self.features_y_group),
        ]
        .into_iter()
        .enumerate()
        {
            dispatch(
                encoder,
                pipeline,
                group,
                workgroups_x,
                workgroups_y,
                if profile {
                    self.gpu_profile
                        .as_ref()
                        .map(|profile| (&profile.queries, index as u32 * 2))
                } else {
                    None
                },
            );
        }
    }
}

fn load_png(path: &str) -> Result<RgbImage, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let mut buffer = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG decoded image is too large")?
    ];
    let info = reader.next_frame(&mut buffer)?;
    let bytes = &buffer[..info.buffer_size()];
    let mut pixels = Vec::with_capacity((info.width * info.height) as usize);
    match info.color_type {
        png::ColorType::Rgb => {
            pixels.extend(bytes.chunks_exact(3).map(|px| [px[0], px[1], px[2]]));
        }
        png::ColorType::Rgba => {
            pixels.extend(bytes.chunks_exact(4).map(|px| [px[0], px[1], px[2]]));
        }
        png::ColorType::Grayscale => {
            pixels.extend(bytes.iter().map(|&gray| [gray, gray, gray]));
        }
        png::ColorType::GrayscaleAlpha => {
            pixels.extend(bytes.chunks_exact(2).map(|px| [px[0], px[0], px[0]]));
        }
        png::ColorType::Indexed => return Err("indexed PNG was not expanded by decoder".into()),
    }
    Ok(RgbImage {
        width: info.width,
        height: info.height,
        pixels,
    })
}

fn load_exr(path: &str) -> Result<HdrImage, Box<dyn Error>> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| ExrPixels {
            width: resolution.width(),
            pixels: vec![[0.0; 4]; resolution.area()],
        },
        |pixels, position, (red, green, blue, _alpha): (f32, f32, f32, f32)| {
            let index = position.y() * pixels.width + position.x();
            pixels.pixels[index] = [red, green, blue, 0.0];
        },
    )?;
    let width = image.layer_data.size.width() as u32;
    let height = image.layer_data.size.height() as u32;
    Ok(HdrImage {
        width,
        height,
        pixels: image.layer_data.channel_data.pixels.pixels,
    })
}

fn ensure_same_size(
    reference_width: u32,
    reference_height: u32,
    test_width: u32,
    test_height: u32,
) -> Result<(), Box<dyn Error>> {
    if reference_width != test_width || reference_height != test_height {
        return Err(format!(
            "image sizes differ: reference is {reference_width}x{reference_height}, test is {test_width}x{test_height}"
        )
        .into());
    }
    Ok(())
}

fn hdr_exposures(reference: &HdrImage, tone_mapper: ToneMapper) -> (f32, f32, u32) {
    let [tc0, tc1, tc2, tc3, tc4, tc5] = tone_mapping_coefficients(tone_mapper);
    let t = 0.85_f32;
    let a = tc0 - t * tc3;
    let b = tc1 - t * tc4;
    let c = tc2 - t * tc5;
    let x_max = if a.abs() < f32::EPSILON {
        -c / b
    } else {
        let discriminant = (b * b - 4.0 * a * c).max(0.0).sqrt();
        let first_root = (-b + discriminant) / (2.0 * a);
        let second_root = (-b - discriminant) / (2.0 * a);
        first_root.max(second_root)
    };

    let mut luminances: Vec<f32> = reference
        .pixels
        .iter()
        .map(|pixel| 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2])
        .collect();
    let max_luminance = luminances
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(f32::EPSILON);
    let median_index = luminances.len() / 2;
    luminances.select_nth_unstable_by(median_index, f32::total_cmp);
    let median_luminance = luminances[median_index].max(f32::EPSILON);
    let start = (x_max / max_luminance).log2();
    let stop = (x_max / median_luminance).log2();
    let count = ((stop - start).ceil() as u32).max(2);
    (start, stop, count)
}

fn tone_mapping_coefficients(tone_mapper: ToneMapper) -> [f32; 6] {
    match tone_mapper {
        ToneMapper::Reinhard => [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ToneMapper::Aces => [
            0.6 * 0.6 * 2.51,
            0.6 * 0.03,
            0.0,
            0.6 * 0.6 * 2.43,
            0.6 * 0.59,
            0.14,
        ],
        ToneMapper::Hable => [0.231683, 0.013791, 0.0, 0.18, 0.3, 0.018],
    }
}

fn pack_pixels(pixels: &[[u8; 3]]) -> Vec<u32> {
    pixels
        .iter()
        .map(|pixel| u32::from(pixel[0]) | (u32::from(pixel[1]) << 8) | (u32::from(pixel[2]) << 16))
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn vec4_bytes(values: &[[f32; 4]]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.iter().flat_map(|component| component.to_ne_bytes()))
        .collect()
}

fn params_bytes(
    width: u32,
    height: u32,
    spatial_radius: u32,
    feature_radius: u32,
    cmax: f32,
    exposure: f32,
    tone_mapper: ToneMapper,
    exposure_fraction: f32,
) -> Vec<u8> {
    [
        width.to_ne_bytes(),
        height.to_ne_bytes(),
        spatial_radius.to_ne_bytes(),
        feature_radius.to_ne_bytes(),
        cmax.to_ne_bytes(),
        (FLIP_CONSTANTS.gpc * cmax).to_ne_bytes(),
        FLIP_CONSTANTS.gqc.to_ne_bytes(),
        FLIP_CONSTANTS.gpt.to_ne_bytes(),
        FLIP_CONSTANTS.gqf.to_ne_bytes(),
        exposure.to_ne_bytes(),
        tone_mapper.id().to_ne_bytes(),
        exposure_fraction.to_ne_bytes(),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn spatial_filter_radius(ppd: f32) -> u32 {
    let max_scale = 0.04_f32;
    (3.0 * (max_scale / (2.0 * std::f32::consts::PI.powi(2))).sqrt() * ppd).ceil() as u32
}

fn feature_filter_radius(ppd: f32) -> u32 {
    (3.0 * (0.5 * FLIP_CONSTANTS.gw * ppd)).ceil() as u32
}

fn spatial_filter(ppd: f32, radius: u32) -> Vec<[f32; 4]> {
    let mut weights = Vec::with_capacity((radius * 2 + 1) as usize);
    let mut sum_y = 0.0;
    let mut sum_cx = 0.0;
    let mut sum_cz1 = 0.0;
    let mut sum_cz2 = 0.0;
    let delta_x = 1.0 / ppd;

    for x in 0..=(radius * 2) {
        let ix = (x as i32 - radius as i32) as f32 * delta_x;
        let ix2 = ix * ix;
        let weight = [
            gaussian(ix2, 1.0, 0.0047),
            gaussian(ix2, 1.0, 0.0053),
            gaussian_sqrt(ix2, 34.1, 0.04),
            gaussian_sqrt(ix2, 13.5, 0.025),
        ];
        sum_y += weight[0];
        sum_cx += weight[1];
        sum_cz1 += weight[2];
        sum_cz2 += weight[3];
        weights.push(weight);
    }
    let inv_cz_norm = 1.0 / (sum_cz1 * sum_cz1 + sum_cz2 * sum_cz2).sqrt();
    for weight in &mut weights {
        weight[0] /= sum_y;
        weight[1] /= sum_cx;
        weight[2] *= inv_cz_norm;
        weight[3] *= inv_cz_norm;
    }
    weights
}

fn gaussian(x2: f32, a: f32, b: f32) -> f32 {
    let pi = std::f32::consts::PI;
    a * (pi / b).sqrt() * (-(pi * pi) * x2 / b).exp()
}

fn gaussian_sqrt(x2: f32, a: f32, b: f32) -> f32 {
    let pi = std::f32::consts::PI;
    (a * (pi / b).sqrt()).sqrt() * (-(pi * pi) * x2 / b).exp()
}

fn feature_filter(ppd: f32, radius: u32) -> Vec<[f32; 4]> {
    let std_dev = 0.5 * FLIP_CONSTANTS.gw * ppd;
    let mut values = Vec::with_capacity((radius * 2 + 1) as usize);
    let mut gaussian_sum = 0.0;
    let mut dg_sum_negative = 0.0;
    let mut dg_sum_positive = 0.0;
    let mut ddg_sum_negative = 0.0;
    let mut ddg_sum_positive = 0.0;

    for x in 0..=(radius * 2) {
        let xx = (x as i32 - radius as i32) as f32;
        let g = (-(xx * xx) / (2.0 * std_dev * std_dev)).exp();
        gaussian_sum += g;
        let dg = -xx * g;
        if dg > 0.0 {
            dg_sum_positive += dg;
        } else {
            dg_sum_negative -= dg;
        }
        let ddg = (xx * xx / (std_dev * std_dev) - 1.0) * g;
        if ddg > 0.0 {
            ddg_sum_positive += ddg;
        } else {
            ddg_sum_negative -= ddg;
        }
        values.push([g, dg, ddg, 0.0]);
    }

    for value in &mut values {
        value[0] /= gaussian_sum;
        value[1] /= if value[1] > 0.0 {
            dg_sum_positive
        } else {
            dg_sum_negative
        };
        value[2] /= if value[2] > 0.0 {
            ddg_sum_positive
        } else {
            ddg_sum_negative
        };
    }
    values
}

fn linear_rgb_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    [
        (10135552.0 / 24577794.0) * rgb[0]
            + (8788810.0 / 24577794.0) * rgb[1]
            + (4435075.0 / 24577794.0) * rgb[2],
        (2613072.0 / 12288897.0) * rgb[0]
            + (8788810.0 / 12288897.0) * rgb[1]
            + (887015.0 / 12288897.0) * rgb[2],
        (1425312.0 / 73733382.0) * rgb[0]
            + (8788810.0 / 73733382.0) * rgb[1]
            + (70074185.0 / 73733382.0) * rgb[2],
    ]
}

fn xyz_to_lab(mut xyz: [f32; 3]) -> [f32; 3] {
    xyz[0] *= 1.052156925;
    xyz[2] *= 0.918357670;
    let delta = 6.0 / 29.0;
    let delta_square = delta * delta;
    let delta_cube = delta * delta_square;
    let factor = 1.0 / (3.0 * delta_square);
    let term = 4.0 / 29.0;
    for component in &mut xyz {
        *component = if *component > delta_cube {
            component.powf(1.0 / 3.0)
        } else {
            factor * *component + term
        };
    }
    [
        116.0 * xyz[1] - 16.0,
        500.0 * (xyz[0] - xyz[1]),
        200.0 * (xyz[1] - xyz[2]),
    ]
}

fn hunt(luminance: f32, chrominance: f32) -> f32 {
    0.01 * luminance * chrominance
}

fn hyab(a: [f32; 3], b: [f32; 3]) -> f32 {
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    (a[0] - b[0]).abs() + (da * da + db * db).sqrt()
}

fn compute_max_distance(gqc: f32) -> f32 {
    let green_lab = xyz_to_lab(linear_rgb_to_xyz([0.0, 1.0, 0.0]));
    let blue_lab = xyz_to_lab(linear_rgb_to_xyz([0.0, 0.0, 1.0]));
    let green_hunt = [
        green_lab[0],
        hunt(green_lab[0], green_lab[1]),
        hunt(green_lab[0], green_lab[2]),
    ];
    let blue_hunt = [
        blue_lab[0],
        hunt(blue_lab[0], blue_lab[1]),
        hunt(blue_lab[0], blue_lab[2]),
    ];
    hyab(green_hunt, blue_hunt).powf(gqc)
}

fn empty_buffer(
    device: &wgpu::Device,
    label: &str,
    size: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage,
        mapped_at_creation: false,
    })
}

fn storage_buffer(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn compute_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FLIP pass bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups_x: u32,
    workgroups_y: u32,
    timestamp: Option<(&wgpu::QuerySet, u32)>,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("FLIP compute pass"),
        timestamp_writes: timestamp.map(|(query_set, index)| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        }),
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

fn save_error_png(
    path: &Path,
    width: u32,
    height: u32,
    errors: &[f32],
    use_magma: bool,
) -> Result<(), Box<dyn Error>> {
    let mut rgb = Vec::with_capacity(errors.len() * 3);
    for &error in errors {
        let color = if use_magma {
            let lut_index = (error.clamp(0.0, 1.0) * 255.0 + 0.5) as usize;
            magma::MAGMA[lut_index]
        } else {
            [error.clamp(0.0, 1.0); 3]
        };
        for channel in color {
            rgb.push((channel.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    save_rgb_png(path, width, height, &rgb)
}

fn save_viridis_png(
    path: &Path,
    width: u32,
    height: u32,
    values: &[f32],
) -> Result<(), Box<dyn Error>> {
    let mut rgb = Vec::with_capacity(values.len() * 3);
    for &value in values {
        let index = (value.clamp(0.0, 1.0) * 255.0 + 0.5) as usize;
        for channel in viridis::VIRIDIS[index] {
            rgb.push((channel.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    save_rgb_png(path, width, height, &rgb)
}

fn save_rgb_png(path: &Path, width: u32, height: u32, rgb: &[u8]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(rgb)?;
    Ok(())
}
