# Workspace structure

## Dependency direction

```text
rvc-rs-inference ─┐
                  ├─> rvc-rs-engine ─> rvc-rs-candle ─> pthrs
rvc-rs-cli ───────┘          │                  └──────> Candle
                             ├─> rvc-rs-core
                             ├─> rvc-rs-dsp
                             └─> rvc-rs-audio (when streaming begins)
```

Dependencies flow inward. Libraries must not depend on applications.

## Package rules

### `rvc-rs-core`

Owns stable vocabulary: architecture version, sample rate, compute selection,
generator inputs, shape validation, and the generator trait. It must not depend
on Candle, egui, CPAL, or filesystem formats.

### `rvc-rs-candle`

Owns device resolution, decoded-weight conversion, layer implementations,
checkpoint-to-layer binding, and the forward pass. This is the only package
allowed to mirror the original PyTorch model architecture.

### `rvc-rs-dsp`

Owns audio math independent of devices and neural models. Hot-path functions
accept caller-owned output slices. External resampling/FFT crates can be wrapped
here after their exact usage is known.

### `rvc-rs-audio`

Owns device discovery, stream negotiation, callback adapters, and bounded audio
transport. CPAL is optional until streaming work begins so core CI does not
require platform audio development packages.

### `rvc-rs-engine`

Owns model selection, settings, lifecycle state, offline jobs, future worker
threads, cancellation, progress, and latency metrics. Front ends communicate
through this package and do not call Candle or CPAL directly.

### `rvc-rs-cli`

Owns headless developer operations and automation-friendly output. It contains
no model equations.

### `rvc-rs-inference`

Owns the egui desktop experience. It may select files, edit settings, display
state, and send commands to the engine. It never performs inference in the UI
thread.

