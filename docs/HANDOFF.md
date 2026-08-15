# Handoff

## Non-negotiable direction

Build direct `.pth` + `.index` real-time RVC in native Rust/Candle. Python RVC
and MMVCServerSIO are behavioral references. Do not replace missing native work
with ONNX or a `vc-rs` dependency.

## Completed in 0.3.0

- removed `vc-rs`, `vc-core`, and ONNX Runtime from the workspace graph;
- direct `.pth` metadata validation and eager state-dictionary transfer;
- native loaded IVF-Flat retrieval with reusable workspace;
- MMVC-compatible streaming geometry and rolling buffers;
- normalized-correlation SOLA plus crossfade;
- `prepare-native` CLI gate.

## Immediate next task

Implement v2/40k/F0 generator parity in `rvc-rs-candle`, starting with exact
weight binding and recorded Python inputs/outputs. The target Python class is
`SynthesizerTrnMs768NSFsid`; port only inference-required blocks.

After generator parity: native ContentVec layer 12, RMVPE, the preallocated
worker, and CPAL device streams. See `ROADMAP.md`.
