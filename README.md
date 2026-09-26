# wgpu FLIP

A Rust implementation of NVIDIA's FLIP image comparison metric. The color conversion,
spatial filtering, feature filtering, and per-pixel error calculation run in WGSL compute
shaders through `wgpu`. PNG provides the LDR input path; EXR provides linear HDR input.

## Run

The original three-path form still works:

```sh
cargo run -- reference.png test.png error-map.png
cargo run -- reference.exr test.exr error-map.png
```

Or use FLIP-style named options. `--test` accepts multiple paths:

```sh
cargo run -- --reference reference.png --test test-a.png test-b.png --directory results
cargo run -- --reference reference.exr --test test.exr --tone-mapper hable
```

The default viewing scale is 67.020645 pixels per degree. Set it directly with
`--pixels-per-degree`, or derive it from monitor distance, width, and horizontal pixel
resolution with `--viewing-conditions DISTANCE WIDTH PIXELS`. HDR tone mappers are
`aces` (default), `hable`, and `reinhard`. HDR start/stop exposure and exposure count
can be overridden with `--start-exposure`, `--stop-exposure`, and `--num-exposures`.

By default the program writes a Magma-colored FLIP error map. For HDR it also writes a
Viridis exposure map that shows which exposure produced each pixel's maximum error.
Use `--no-error-map`, `--no-exposure-map`, or `--no-magma` to change those outputs.
`--basename` names outputs, `--directory` selects their folder, `--textfile` saves the
summary beside the error map, and `--csv PATH` appends summary rows to a CSV file.
Run `cargo run -- --help` for all options. Histogram plots, exit-on-threshold, and
exporting all intermediate HDR images are not implemented.
Use `--profile-gpu` to print GPU times for the compute passes when the adapter supports
timestamp queries. For HDR, it profiles the final exposure; the other exposures run
the same passes.

## What the result means

FLIP produces a value between 0 and 1 for every pixel. Zero means that pixel is
perceptually alike in the two inputs; values nearer one indicate a larger perceptual
difference. The PNG is a false-color visualization of this per-pixel error image. It is
not a reconstructed or blended difference image.

The printed mean is the overall FLIP score. The weighted median and weighted quartiles
describe the distribution of error while giving larger errors more weight; min and max
show the endpoints. In HDR mode, FLIP tone maps a sequence of exposures, computes LDR
FLIP at each exposure, and keeps the largest error and its exposure at every pixel.

## Reproduce the comparisons

The sampler compares the Rust implementation with the C++ and CUDA executables from
`../../Github/flip`. It runs engines serially, samples cases reproducibly, saves maps,
and checks both summary values and per-pixel map differences. It has no persistent
daemon; each invocation exits after its selected cases finish.

```sh
uv run scripts/compare_flip.py --fraction 0.5 --seed 20260925
```

By default it discovers the bundled PNG and EXR `reference`/`test` pairs. For a larger
image collection, create a CSV manifest with paths relative to the manifest file:

```csv
name,reference,test
scene-a,images/scene-a-reference.png,images/scene-a-test.png
scene-b,images/scene-b-reference.exr,images/scene-b-test.exr
```

Then choose a random fraction (and optionally an engine subset or maximum case count):

```sh
uv run scripts/compare_flip.py --manifest cases.csv --fraction 0.25 --seed 42
```

To run it in the background and retain console output:

```sh
mkdir -p target/flip-bench
nohup uv run scripts/compare_flip.py \
  --manifest cases.csv --fraction 0.25 --seed 42 \
  --output-dir target/flip-bench \
  > target/flip-bench/console.log 2>&1 &
```

The script incrementally writes `report.json` and `runs.jsonl`, plus `results.csv`,
per-engine logs, and error maps. JSON records include the command, adapter, wall time,
reported evaluation time, all pooled FLIP statistics, exact matching-pixel fraction,
mean channel delta, and maximum channel delta. Its defaults require score deltas no
larger than `1e-5` and map channel deltas no larger than 3. `UV_CACHE_DIR=/tmp/uv-cache`
can be set if the default uv cache is not writable.

## Accuracy against the bundled FLIP cases

On the bundled 1920x1080 PNG and EXR pairs, the Rust mean scores match the FLIP CPU and
CUDA means to six decimal places: `0.159691` for LDR and `0.283478` for HDR/ACES. Compared
with the FLIP C++ and CUDA maps, the GPU wgpu maps have over 99.99% exact pixels and a
largest RGB channel difference of 3. The forced software adapter maps are over 99.98%
exact, also with a maximum channel difference of 3. Hable and Reinhard HDR runs match
the CUDA means to six decimal places; their map differences have the same maximum
channel delta of 3. The HDR exposure map matches CUDA at 99.9988% of pixels. Its 25
differing pixels out of 2,073,600 have a maximum channel delta of 210 and mean channel
delta of 0.000358 across the map. Tiny floating-point changes make adjacent exposures
tie or swap as the location of maximum error at those pixels.

## Performance

The bundled PNG and EXR pairs are 1920x1080; the EXR case evaluates 14 exposures. On
the GTX 750 Ti Vulkan adapter, five LDR and three HDR runs of each Rust binary were
alternated and timed in Release mode. Error and exposure map writing were disabled.
The values are medians. “Evaluation” is the program's algorithm timer; “process” is
wall time for the whole command, including image decoding and GPU setup.

| Rust wgpu version | LDR evaluation | LDR process | HDR evaluation | HDR process |
| --- | ---: | ---: | ---: | ---: |
| Initial implementation | 59.1 ms | 476.3 ms | 787.4 ms | 1.523 s |
| Optimized implementation | 35.1 ms | 393.9 ms | 421.6 ms | 1.151 s |

The evaluation speedups are 1.68x for LDR and 1.87x for HDR on this case. The changes
halve the spatial intermediate buffer, store scalar errors as scalars, use 16x8
workgroups, and keep the HDR per-pixel maximum on the GPU. HDR now reads back both the
error and winning-exposure maps once after all exposures, instead of reading back an
error map after every exposure. The optimized HDR map and mean score match the initial
implementation on the bundled case.

`--profile-gpu` measured about 25.6 ms of compute passes for the final HDR exposure:
1.9 ms for conversion and tone mapping, 6.2 and 6.0 ms for the two spatial passes,
5.6 and 5.5 ms for the two feature passes, and 0.4 ms to update the per-pixel maximum.
Fourteen full passes, command submission, and the final readback account for the roughly
422 ms evaluation. Timestamp profiling adds overhead, so these pass times are only a
guide to where GPU time goes.

The comparison script also ran all backends once with error maps enabled after the
optimization. The following values are illustrative single runs on the same case:

| Backend | LDR evaluation | LDR process | HDR evaluation | HDR process |
| --- | ---: | ---: | ---: | ---: |
| Rust wgpu, Vulkan GPU | 38.1 ms | 0.710 s | 426.4 ms | 1.412 s |
| FLIP CUDA | 111.8 ms | 1.108 s | 1.398 s | 2.437 s |
| Rust wgpu, Vulkan CPU | 816.7 ms | 1.517 s | 6.467 s | 7.518 s |
| FLIP C++ Release | 693.8 ms | 1.500 s | 10.047 s | 10.792 s |

The timer scopes differ: wgpu excludes pipeline/filter setup and automatic HDR
exposure-range selection, then includes compute dispatch and error-map readback.
FLIP's timer surrounds `FLIP::evaluate`, including filter setup and HDR exposure
selection. Whole-command time includes setup for both. These results are specific to
this GTX 750 Ti, the CUDA binary built for `sm_50`, the host, and these images; they
do not establish a general speed ranking between wgpu and CUDA. CPU timings varied
between runs, so use the sampling script for a larger corpus or another machine.
