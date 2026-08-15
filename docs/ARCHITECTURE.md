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

## Target streaming flow

```text
input callback -> bounded SPSC queue -> inference worker
inference worker:
  rolling audio -> 16 kHz -> ContentVec + RMVPE
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

## Model implementation order

1. bind every checkpoint tensor by exact state-dictionary name;
2. speaker/pitch embeddings and content encoder;
3. residual coupling flow;
4. NSF source path;
5. HiFi-GAN decoder;
6. v2/40k/F0 top-level inference;
7. native ContentVec and RMVPE;
8. streaming worker and CPAL devices.

## Adapter policy

Alternative runtimes belong outside the core dependency graph. The detached
`crates/rvc-rs-onnx` prototype may later be renamed and published as a `vc-rs`
adapter. Native types must not depend on it.
