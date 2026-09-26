# API notes

`FlipEvaluator` binds its compute pipelines to one image width, height, pixels
per degree, tone mapper, and `wgpu` device. It can be reused for more than one
comparison with those settings. Use separate evaluators when dimensions or
settings differ. Evaluations take `&mut self` because each evaluator reuses its
input and intermediate buffers.

## Input conventions

- `evaluate_ldr`: each input is exactly `width * height` `[u8; 3]` pixels in
  row-major order. RGB values are sRGB encoded, as they are in an ordinary PNG.
- `evaluate_hdr`: each input is exactly `width * height` `[f32; 4]` pixels in
  row-major order. RGB is linear-light; alpha is ignored.
- `evaluate_ldr_buffers`: each `COPY_SRC` buffer contains packed RGB8 values as
  numeric `u32`s using `r | (g << 8) | (b << 16)`.
- `evaluate_hdr_buffers`: each `COPY_SRC` buffer contains tightly packed
  row-major RGBA32F values. Explicit start exposure, stop exposure, and sample
  count are required for this path.
- `evaluate_ldr_textures`: `Rgba8UnormSrgb` 2D textures with matching dimensions
  and `COPY_SRC` usage.
- `evaluate_hdr_textures`: `Rgba32Float` 2D textures with matching dimensions
  and `COPY_SRC` usage. Explicit exposure settings are required.

GPU input buffers must belong to the same device used to construct the
`FlipEvaluator`. The result is read back as `Vec<f32>` even when the inputs are
GPU buffers. The synchronous methods call `Device::poll` and block until mapping
finishes, so call them off an application's latency-sensitive render thread or
adapt the implementation for asynchronous readback.

## Wgpu requirements

The crate currently uses the `wgpu` 30 API and requests no optional device
features for regular evaluation. Its compute passes use storage buffers and
workgroups of 16 by 8. When consuming the raw shader constants directly, follow
the bindings and `Params` definitions in the WGSL sources. Those layouts are
internal to `FlipEvaluator` and may change.

## CLI

The `cli` feature enables the convenience binary and its PNG/EXR loading,
output maps, statistics, and CSV/text reporting. Applications that only need
in-memory evaluation can set `default-features = false` on this path dependency.
