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

The native v2 model path is implemented, but offline numerical parity remains
the release gate. Real-checkpoint testing found and corrected HuBERT
normalization, feature upsampling, source generation, resampling, and F0
failures in 0.4.2-rc.1. Flexible quality controls were added in 0.4.2-rc.2.
Streaming work stays blocked until the corrected
offline path passes reference and listening tests.

## Adapter policy

Alternative runtimes belong outside the core dependency graph. A future
adapter should be published as a separate crate; native types must not depend
on it.
