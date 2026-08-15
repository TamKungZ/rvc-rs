# Native `.pth` / `.index` roadmap

The completion target is MMVCServerSIO-compatible real-time RVC behavior with
no Python, PyTorch, libtorch, FAISS, vc-rs, vc-core, or ONNX Runtime dependency
in the default build.

## 1. Model data — working

- [x] Safe ZIP PyTorch checkpoint reader
- [x] RVC metadata/config extraction
- [x] Checkpoint consistency validation
- [x] Eager state-dictionary transfer to Candle
- [x] Pure-Rust FAISS IVF-Flat reader
- [x] In-memory, preallocated index search and blending
- [ ] Complete required/unexpected weight manifest for every supported variant
- [x] Real v2/32k/F0 checkpoint and matching index fixture

Exit: a real checkpoint and index load with every generator weight accounted
for exactly once.

## 2. Generator inference — working

- [x] Speaker and pitch embeddings
- [x] Text/content encoder
- [x] Residual coupling flow
- [x] NSF sine source
- [x] HiFi-GAN upsampling decoder and residual blocks
- [x] v2 F0 top-level forward implemented
- [ ] Intermediate tensor comparisons against Python
- [ ] Final waveform tolerance and finite-value checks

Exit: generator-ready tensors produce the same waveform as the Python
`SynthesizerTrnMs768NSFsid.infer` reference.

## 3. Native front end

- [x] ContentVec/HuBERT architecture and checkpoint adapter
- [x] v2 layer-12 features
- [ ] v1 layer-9 + final projection
- [ ] RMVPE architecture and checkpoint adapter
- [x] Lightweight YIN F0, transpose, and quantization
- [x] Native `.index` retrieval blend
- [x] Functional WAV-to-generator-input pipeline
- [x] Mandatory managed HuBERT download, cache, and integrity verification
- [ ] RMVPE/protect and Python numerical parity

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

No adapter implementation is kept in this repository. A future `vc-rs`
adapter can be a separate crate after the native pipeline stands on its own.
