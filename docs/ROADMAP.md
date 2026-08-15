# Implementation roadmap

## Phase 0 — foundation

- [x] Umbrella Cargo workspace
- [x] Backend-independent model and input validation
- [x] Candle CPU/CUDA/Metal device selection
- [x] Safe decoded-`f32` to Candle tensor boundary
- [x] Allocation-free starter DSP primitives
- [x] Shared engine settings, jobs, lifecycle, and validation
- [x] Headless CLI
- [x] egui desktop application with persisted form state
- [x] CI for formatting, Clippy, tests, and docs
- [x] Explicit gate preventing fake conversion output

## Phase 1 — trusted fixtures

- [ ] Select one local v2/40k/F0 exported checkpoint
- [ ] Record its hash and non-secret provenance
- [ ] Create deterministic phone, length, pitch, pitchf, and speaker tensors
- [ ] Record PyTorch intermediate outputs and waveform
- [ ] Store shapes, dtypes, hashes, revisions, seeds, and tolerances in a manifest
- [ ] Keep model and array bytes outside Git

Exit condition: the reference bundle is independently reproducible.

## Phase 2 — weight adapter

- [ ] Inspect the exact public `pthrs 0.2.0` API
- [ ] Derive `ModelSpec` from checkpoint config
- [ ] Load one named tensor into Candle on CPU
- [ ] Preserve and verify name, original dtype, shape, and element count
- [ ] Bind every required generator weight exactly once
- [ ] Report complete missing and unexpected weight sets
- [ ] Add checkpoint compatibility tests using existing pthrs fixtures

Exit condition: all v2/40k/F0 weights bind without silent transformations.

## Phase 3 — generator parity

- [ ] Embeddings
- [ ] Content encoder
- [ ] Residual coupling flow
- [ ] NSF source path
- [ ] HiFi-GAN decoder and upsampling blocks
- [ ] Top-level synthesizer
- [ ] Intermediate output comparisons
- [ ] Final waveform comparison
- [ ] NaN/infinity checks

Exit condition: deterministic CPU `f32` output matches the reference tolerance.

## Phase 4 — offline conversion

- [ ] Audio decode and WAV encode
- [ ] ContentVec/HuBERT inference
- [ ] F0 extraction
- [ ] `pthrs` retrieval search and feature blending
- [ ] Resampling and normalization
- [ ] Chunk boundary handling
- [ ] CLI file conversion
- [ ] GUI progress, cancellation, and output reveal
- [ ] Timing and peak-memory report

Exit condition: WAV-to-WAV conversion works without Python/PyTorch at runtime.

## Phase 5 — compatibility

- [ ] v1 / 40 kHz / F0
- [ ] v1 / 48 kHz / F0
- [ ] v2 / 32 kHz / F0
- [ ] non-F0 exported inference model
- [ ] compatibility matrix shared with pthrs documentation

## Phase 6 — real-time

- [ ] CPAL device discovery and stable selection
- [ ] Fixed-size workspaces and bounded SPSC buffers
- [ ] Dedicated inference worker
- [ ] Back-pressure, underrun, and overrun policies
- [ ] Overlap/crossfade artifact tests
- [ ] Input/output clock-drift compensation
- [ ] Latency breakdown in the GUI
- [ ] Device disconnect recovery

## Phase 7 — optimization and packaging

- [ ] Release CPU baseline and allocation audit
- [ ] CUDA parity and profiling
- [ ] Metal parity and profiling
- [ ] Optional lower precision
- [ ] Windows, Linux, and macOS packaging
- [ ] Model compatibility and performance regression suite

