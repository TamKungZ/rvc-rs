# Roadmap

Goal: real-time RVC with MMVCServerSIO behavior and no Python, PyTorch,
libtorch, FAISS, or ONNX Runtime in the default build.

## 1. Model data

- [x] Safe ZIP PyTorch checkpoint reader
- [x] RVC metadata and config extraction
- [x] Eager state-dictionary transfer to Candle
- [x] Pure-Rust FAISS IVF-Flat reader and preallocated search
- [ ] Required/unexpected weight manifest for every supported variant

## 2. Generator

- [x] v2 F0 synthesizer: text encoder, flow, NSF source, HiFi-GAN decoder
- [x] Block and waveform parity with PyTorch

## 3. Front end

- [x] HuBERT v2 layer-12 features
- [ ] HuBERT v1 layer-9 with final projection
- [x] RMVPE with PyTorch parity
- [x] YIN fallback
- [x] Index blend and unvoiced protection
- [x] RVC offline pipeline: high-pass, padding, quiet-point splits, RMS mix
- [x] Managed HuBERT and RMVPE downloads

## 4. Real time

- [x] MMVC context geometry and 128-sample alignment
- [x] Rolling history buffers
- [x] Allocation-free SOLA search and crossfade
- [ ] Preallocated pipeline workspaces
- [ ] Bounded SPSC queues and inference worker
- [ ] CPAL input/output streams
- [ ] Underrun, overrun, and device-loss handling
- [ ] Latency budget and soak test

## 5. Compatibility

- [ ] v1 models
- [ ] Non-F0 models
- [ ] Common weight-norm and export variants
- [ ] IVF-Flat variants found in public RVC models
