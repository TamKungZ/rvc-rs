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

The workspace does not depend on `vc-rs`, `vc-core`, ONNX Runtime, Python,
PyTorch, libtorch, or native FAISS.

## Current native checkpoint (0.4.1)

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
- native ContentVec/HuBERT v2 inference;
- in-tree autocorrelation F0 extraction and RVC pitch quantization;
- complete v2 F0 inference (`enc_p`, residual coupling flow, NSF source, and
  HiFi-GAN decoder);
- WAV decode, resample, conversion, and float-WAV output;
- tested conversion with the TITAN 32k `.pth` and matching `.index` from the
  `pthrs` compatibility matrix.
- mandatory managed `hubert_base.pt`: downloaded once into the per-user cache,
  pinned by immutable revision, and verified by size plus SHA-256 before use;
- no ContentVec/HuBERT file argument or picker in the CLI and GUI.

Real-time CPAL device streaming, RMVPE parity, v1 ContentVec, and non-F0 model
support remain in progress.

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

## Convert a WAV now

RVC requires a content encoder in addition to the target voice checkpoint.
`rvc-rs` manages the required RVC `hubert_base.pt` internally: the first
conversion downloads and verifies it, and every later conversion reuses the
same per-user cached asset. It is mandatory and cannot be selected or disabled.
The F0 extractor is built in and does not need an RMVPE/ONNX file.

```bash
cargo run --release -p rvc-rs-cli -- convert \
  voice.pth voice.index input.wav output.wav 0 auto
```

Use `-` instead of `voice.index` to disable retrieval. The last two optional
arguments are pitch shift in semitones and device. The current file decoder is
deliberately WAV-only to avoid pulling a full codec framework into the core
build.

The managed model is stored at the OS cache location:

- Linux: `$XDG_CACHE_HOME/rvc-rs/models/hubert_base.pt`, or
  `$HOME/.cache/rvc-rs/models/hubert_base.pt`;
- macOS: `$HOME/Library/Caches/rvc-rs/models/hubert_base.pt`;
- Windows: `%LOCALAPPDATA%\rvc-rs\models\hubert_base.pt`.

See [Managed model assets](docs/MODEL_ASSETS.md) for pinned provenance and
integrity metadata.

## Workspace

| Package | Responsibility |
|---|---|
| `rvc-rs-core` | Model contracts and MMVC-compatible streaming geometry |
| `rvc-rs-candle` | ContentVec, direct `.pth` model loading, generator, and retrieval |
| `rvc-rs-dsp` | Rolling buffers, SOLA, crossfade, meters, and channel mixing |
| `rvc-rs-audio` | File audio I/O and future device-format utilities |
| `rvc-rs-engine` | Native file-conversion pipeline and lifecycle |
| `rvc-rs-cli` | Conversion, model preparation, validation, and diagnostics |
| `rvc-rs-inference` | Non-critical development UI |

## Validation

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
```

Waveform parity work requires recorded Python reference tensors. Functional
end-to-end tests require a user-supplied RVC checkpoint; ContentVec uses the
same mandatory managed HuBERT asset as normal inference.

## References

- [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [MMVCServerSIO / voice-changer](https://github.com/w-okada/voice-changer)
- [vc-rs](https://github.com/shirohata/vc-rs) — optional adapter/reference only

## License

Licensed under the [Apache License 2.0](LICENSE).
