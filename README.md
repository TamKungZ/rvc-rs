# rvc-rs

Native Rust implementation of Retrieval-based Voice Conversion focused on
direct RVC `.pth` checkpoints, FAISS `.index` retrieval, and real-time audio.
Python/PyTorch behavior is the compatibility reference, not a runtime
dependency.

## Direction

The production path is:

```text
microphone -> rolling 16 kHz context -> ContentVec + F0
           -> native .index retrieval blend
           -> Candle generator loaded directly from .pth
           -> SOLA/crossfade -> output device
```

The workspace no longer depends on `vc-rs`, `vc-core`, or ONNX Runtime. The old
ONNX experiment is excluded from the workspace under `crates/rvc-rs-onnx` as a
future optional adapter and is not part of the native build.

## Current native checkpoint (0.3.0)

Implemented and tested:

- safe ZIP-based PyTorch `.pth` parsing through `pthrs`;
- extraction and validation of RVC version, sample rate, F0 flag, speaker count,
  feature dimension, and architecture configuration;
- eager transfer of every generator weight to the selected Candle device;
- pure-Rust FAISS `IndexIVFFlat` loading and dimension validation;
- preallocated nearest-neighbor retrieval blending with no per-frame heap
  allocation;
- MMVC-style fixed rolling buffers and 128-sample conversion alignment;
- allocation-free normalized-correlation SOLA selection and raised-cosine
  crossfade;
- CPU/CUDA/Metal device selection boundaries.

Still required before audio can be converted:

- Candle forward pass for the RVC synthesizer;
- native ContentVec/HuBERT inference;
- native F0 extraction (RMVPE first);
- inference worker and CPAL device streams;
- numerical and waveform parity against the Python reference.

The program returns an explicit error instead of passthrough or fake output
while those stages are incomplete.

## Inspect and prepare a real model

Build the CLI:

```bash
cargo build --release -p rvc-rs-cli
```

Load a `.pth`, decode all tensors into Candle, and optionally load its index:

```bash
cargo run -p rvc-rs-cli -- prepare-native voice.pth voice.index cpu
```

Use `-` when no index is available:

```bash
cargo run -p rvc-rs-cli -- prepare-native voice.pth - auto
```

This command is the current real-checkpoint gate. It fails on malformed or
unsupported checkpoints, incompatible index dimensions, missing tensors, and
unavailable devices.

## Workspace

| Package | Responsibility |
|---|---|
| `rvc-rs-core` | Model contracts and MMVC-compatible streaming geometry |
| `rvc-rs-candle` | Direct `.pth` loading, Candle tensors, and native retrieval |
| `rvc-rs-dsp` | Rolling buffers, SOLA, crossfade, meters, and channel mixing |
| `rvc-rs-audio` | File audio I/O and future device-format utilities |
| `rvc-rs-engine` | Native model preparation and pipeline lifecycle |
| `rvc-rs-cli` | Model preparation, validation, and diagnostics |
| `rvc-rs-inference` | Non-critical development UI |

## Validation

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
```

End-to-end parity requires a redistributable or user-supplied RVC checkpoint,
index, ContentVec/RMVPE weights, and recorded Python reference tensors.

## References

- [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [MMVCServerSIO / voice-changer](https://github.com/w-okada/voice-changer)
- [vc-rs](https://github.com/shirohata/vc-rs) — optional adapter/reference only

## License

Licensed under the [Apache License 2.0](LICENSE).
