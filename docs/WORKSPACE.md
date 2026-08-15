# Workspace structure

```text
rvc-rs-cli ---------> rvc-rs-engine -----> rvc-rs-candle -----> pthrs
rvc-rs-inference ---/        |                    |
                             v                    v
                       rvc-rs-core          Candle tensors
                             |
                             v
                        rvc-rs-dsp
```

- `rvc-rs-core`: backend-neutral generator inputs and streaming geometry.
- `rvc-rs-candle`: direct checkpoint loading, retrieval state, and model math.
- `rvc-rs-dsp`: rolling buffers, SOLA, crossfade, metering, and channel mixing.
- `rvc-rs-audio`: dependency-light WAV I/O and future device-format helpers.
- `rvc-rs-engine`: preparation, ownership, lifecycle, and worker orchestration.
- applications: thin CLI and non-critical egui development UI.

There is no ONNX or `vc-rs` adapter in the source tree.
