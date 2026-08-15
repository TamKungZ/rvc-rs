# Native `.pth` / `.index` roadmap

The completion target is MMVCServerSIO-compatible real-time RVC behavior with
no Python, PyTorch, libtorch, FAISS, vc-rs, vc-core, or ONNX Runtime dependency
in the default build.

## 1. Model data — active

- [x] Safe ZIP PyTorch checkpoint reader
- [x] RVC metadata/config extraction
- [x] Checkpoint consistency validation
- [x] Eager state-dictionary transfer to Candle
- [x] Pure-Rust FAISS IVF-Flat reader
- [x] In-memory, preallocated index search and blending
- [ ] Complete required/unexpected weight manifest for every supported variant
- [ ] Real v2/40k/F0 checkpoint preparation fixture

Exit: a real checkpoint and index load with every generator weight accounted
for exactly once.

## 2. Generator parity — next critical path

- [ ] Speaker and pitch embeddings
- [ ] Text/content encoder
- [ ] Residual coupling flow
- [ ] NSF sine source
- [ ] HiFi-GAN upsampling decoder and residual blocks
- [ ] v2/40k/F0 top-level forward
- [ ] Intermediate tensor comparisons against Python
- [ ] Final waveform tolerance and finite-value checks

Exit: generator-ready tensors produce the same waveform as the Python
`SynthesizerTrnMs768NSFsid.infer` reference.

## 3. Native front end

- [ ] ContentVec/HuBERT architecture and checkpoint adapter
- [ ] v2 layer-12 features
- [ ] v1 layer-9 + final projection
- [ ] RMVPE architecture and checkpoint adapter
- [ ] F0 quantization, transpose, unvoiced interpolation, and protect mask
- [x] Native `.index` retrieval blend
- [ ] Full raw-audio-to-generator-input parity

## 4. Real-time execution

- [x] MMVC-compatible context/feature geometry and 128-sample alignment
- [x] Fixed rolling history buffers
- [x] Allocation-free SOLA search and crossfade
- [ ] Preallocated pipeline workspaces
- [ ] Bounded SPSC input/output queues
- [ ] Dedicated inference worker
- [ ] CPAL input/output streams
- [ ] Back-pressure, underrun, overrun, and device-loss policy
- [ ] Measured latency budget and sustained soak test

## 5. Compatibility

- [ ] v1/v2
- [ ] 32/40/48 kHz
- [ ] F0 and non-F0
- [ ] Multi-speaker checkpoints
- [ ] Common weight-norm/export variants
- [ ] IVF-Flat index variants encountered in public RVC models

## Optional adapters

The previous `vc-rs`/ONNX code is excluded from the workspace. It may become an
explicit adapter after the native pipeline stands on its own; it cannot be the
default implementation or define the core interfaces.
