# rvc-rs

Native Rust building blocks and applications for Retrieval-based Voice
Conversion (RVC), without requiring Python or PyTorch at runtime.

The project uses [`pthrs`](https://github.com/TamKungZ/pthrs) for exported
PyTorch checkpoints and FAISS IVF-Flat retrieval indexes, Candle for the first
native inference backend, and egui/eframe for the desktop application.

> [!IMPORTANT]
> This is a carefully gated implementation foundation, not a working voice
> converter yet. The desktop app and CLI are functional, but generator execution
> remains locked until deterministic Rust output matches a trusted PyTorch
> reference.

## Workspace

| Package | Type | Responsibility |
|---|---|---|
| `rvc-rs-core` | library | Backend-independent model specs, inputs, validation, and inference traits |
| `rvc-rs-candle` | library | Candle devices, `pthrs` weight boundary, and the RVC model implementation |
| `rvc-rs-dsp` | library | Allocation-free channel mixing, meters, and chunk-boundary crossfades |
| `rvc-rs-audio` | library | Platform-independent audio device and stream contracts |
| `rvc-rs-engine` | library | Shared settings, jobs, validation, state, and future inference worker |
| `rvc-rs-cli` | application | Headless validation and backend diagnostics |
| `rvc-rs-inference` | application | egui desktop application for offline and future real-time conversion |

The GUI and CLI call the same engine. Model mathematics never belongs in a
front end.

## Build

```bash
cargo check --workspace
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

Launch the desktop application:

```bash
cargo run -p rvc-rs-inference
```

Run backend diagnostics:

```bash
cargo run -p rvc-rs-cli -- doctor cpu
```

Optional acceleration:

```bash
cargo run -p rvc-rs-inference --features cuda
cargo run -p rvc-rs-inference --features metal
```

CUDA requires a compatible CUDA development environment. Metal is available on
supported Apple platforms.

## First working milestone

Given one fixed v2/40k/F0 checkpoint and fixed generator-ready tensors:

1. load every required weight from `.pth` through `pthrs 0.2.0`;
2. construct the generator with Candle;
3. execute a deterministic CPU `f32` forward pass;
4. produce a mono waveform;
5. match recorded PyTorch intermediate outputs and waveform within a documented
   tolerance.

Only then does the project add ContentVec/HuBERT, F0 extraction, retrieval
blending, file conversion, and real-time audio.

## Design principles

- No Python or PyTorch runtime dependency.
- Offline numerical correctness before microphone streaming.
- Explicit errors instead of placeholder audio.
- Caller-owned reusable buffers on real-time paths.
- Audio callbacks never run inference or perform file I/O.
- CPU is the correctness baseline; accelerators are validated against it.
- Model and fixture licensing is respected; large weights are not committed.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Workspace structure](docs/WORKSPACE.md)
- [Implementation roadmap](docs/ROADMAP.md)
- [Reference testing](docs/REFERENCE_TESTING.md)
- [Desktop application](docs/GUI.md)
- [Handoff for a new chat](docs/HANDOFF.md)
- [Codex project rules](AGENTS.md)

## License

Licensed under the [Apache License 2.0](LICENSE).

