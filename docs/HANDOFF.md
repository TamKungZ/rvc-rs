# Handoff

## Non-negotiable direction

Build direct `.pth` + `.index` real-time RVC in native Rust/Candle. Python RVC
and MMVCServerSIO are behavioral references. Do not replace missing native work
with ONNX or a `vc-rs` dependency.

## Completed in 0.4.0

- removed `vc-rs`, `vc-core`, and ONNX Runtime from the workspace graph;
- direct `.pth` metadata validation and eager state-dictionary transfer;
- native loaded IVF-Flat retrieval with reusable workspace;
- MMVC-compatible streaming geometry and rolling buffers;
- normalized-correlation SOLA plus crossfade;
- `prepare-native` CLI gate.
- native v2 ContentVec layer-12 inference;
- native v2/F0 RVC text encoder, reverse flow, NSF, and decoder;
- dependency-light DSP F0 extraction;
- working WAV conversion with optional `.index` retrieval;
- functional test against TITAN 32k and `hubert_base.pt`.

## Immediate next task

Refactor the resident file pipeline into a chunked worker, connect CPAL device
streams, and apply the existing SOLA/crossfade primitives. Then replace or
augment autocorrelation F0 with native RMVPE and record Python parity fixtures.
