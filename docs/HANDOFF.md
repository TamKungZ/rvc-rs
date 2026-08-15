# Handoff

## Non-negotiable direction

Build direct `.pth` + `.index` real-time RVC in native Rust/Candle. Python RVC
and MMVCServerSIO are behavioral references. Do not replace missing native work
with ONNX or a `vc-rs` dependency.

## Implemented through 0.4.2-rc.1

- removed `vc-rs`, `vc-core`, and ONNX Runtime from the workspace graph;
- direct `.pth` metadata validation and eager state-dictionary transfer;
- native loaded IVF-Flat retrieval with reusable workspace;
- MMVC-compatible streaming geometry and rolling buffers;
- normalized-correlation SOLA plus crossfade;
- `prepare-native` CLI gate.
- native v2 ContentVec layer-12 inference, corrected to fairseq's
  `GroupNorm(512, 512)` and post-norm transformer behavior;
- native v2/F0 RVC text encoder, reverse flow, NSF, and decoder;
- dependency-light YIN F0 extraction without the former high-frequency
  autocorrelation spikes;
- WAV conversion with optional `.index` retrieval (release-candidate status;
  waveform parity is not yet established);
- nearest-neighbor feature expansion, band-limited input resampling, and
  RVC-compatible NSF phase/noise behavior.
- removed user-supplied HuBERT paths from CLI, GUI, and engine model identity;
- mandatory HuBERT auto-download into the OS cache with immutable revision,
  exact-size, and SHA-256 enforcement.

## Immediate next task

Build and run 0.4.2-rc.1 against the recorded v2/40k/F0 regression case, first
without retrieval and then with its matching index. Record HuBERT, text
encoder, flow, decoder, and final waveform reference tensors. Do not begin
CPAL streaming until this gate passes. Native RMVPE remains the next front-end
quality milestone after correctness is established.
