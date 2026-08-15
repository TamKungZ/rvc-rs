# Handoff for a new ChatGPT/Codex thread

Attach the ZIP and use this prompt:

```text
Continue the rvc-rs project in this ZIP.

Context:
- pthrs 0.2.0 is published at https://github.com/TamKungZ/pthrs
- pthrs reads exported RVC .pth tensors and FAISS IndexIVFFlat .index files
- this project must perform inference without Python/PyTorch at runtime
- Candle is the first direct-checkpoint tensor backend
- rvc-rs-inference is the egui desktop app
- GUI and CLI must share rvc-rs-engine
- CPU correctness comes before CUDA, Metal, and real-time streaming

First milestone:
fixed generator inputs + v2/40k/F0 .pth
-> pthrs weight adapter
-> Candle RVC generator
-> deterministic mono waveform matching a PyTorch reference

Read AGENTS.md, README.md, and every file under docs/ before changing code.
Start with Phase 1 and Phase 2 in docs/ROADMAP.md.
Do not start microphone streaming and do not claim inference works until the
waveform parity test passes.
If pthrs needs changes, keep them format/retrieval-generic and send them
separately from rvc-rs.
```

## Current state

- Seven-package workspace and dependency boundaries are established.
- Core input validation and starter DSP functions have unit tests.
- Candle CPU device/tensor smoke testing exists with optional CUDA and Metal.
- Shared engine configuration, path validation, and lifecycle exist.
- CLI and egui application use the shared engine.
- `CandleGenerator` intentionally returns `ModelNotImplemented`.
- No checkpoint API was guessed; inspect the published `pthrs 0.2.0` API first.
- No model forward pass, content encoder, F0 extractor, audio decoder, or stream
  exists yet.

