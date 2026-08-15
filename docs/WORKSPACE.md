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
- `rvc-rs-audio`: file codecs and future device-format helpers.
- `rvc-rs-engine`: preparation, ownership, lifecycle, and worker orchestration.
- applications: thin CLI and non-critical egui development UI.

`crates/rvc-rs-onnx` is explicitly excluded. It is retained only as source for
a possible future `vc-rs` adapter and cannot influence native core interfaces.
