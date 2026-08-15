# Architecture

## Production boundary

The default workspace is standalone. `rvc-rs-candle` consumes `.pth` and
`.index` data through `pthrs`; neither ONNX Runtime nor `vc-rs` participates in
model loading, retrieval, or inference.

```text
.pth -> pthrs validation -> named f32 tensors -> Candle device
.index -> pthrs IVF-Flat -> resident vectors + reusable search workspace
```

Checkpoint and index loading are eager startup operations. They may allocate
and perform file I/O. The real-time loop may not.

The shared HuBERT/ContentVec checkpoint is a mandatory engine-owned asset. It
is resolved from the per-user cache, downloaded once when absent, and verified
against pinned size and SHA-256 metadata. Front ends never supply its path.

## Target streaming flow

```text
input callback -> bounded SPSC queue -> inference worker
inference worker:
  rolling audio -> 16 kHz -> ContentVec + native F0
  features -> index blend -> Candle RVC generator
  waveform -> SOLA/crossfade -> output queue
output callback <- bounded SPSC queue
```

The callback moves fixed-size sample blocks only. Model execution, resampling,
retrieval, logging, file I/O, and allocation stay on the inference worker.

## MMVCServerSIO compatibility points

- retain audio, pitch, and feature history across calls;
- calculate the 100 Hz feature grid from output-rate context;
- include block, crossfade, 12 ms SOLA search, and extra conversion history;
- round generator context up to a 128-sample boundary;
- blend retrieved features before 2x temporal interpolation;
- align consecutive outputs by normalized correlation and crossfade the chosen
  boundary;
- return exactly one device block per cycle.

These are behavioral requirements. Python tensor layouts and padding rules are
recorded as parity fixtures before each Candle block is accepted.

## Model implementation status

Items 1–6 and native v2 ContentVec are implemented. The next work is a
resident chunked pipeline, CPAL devices, then RMVPE and numerical parity.

## Adapter policy

Alternative runtimes belong outside the core dependency graph. A future
adapter should be published as a separate crate; native types must not depend
on it.
