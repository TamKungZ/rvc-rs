# Architecture

## Offline data flow

```text
.pth -> pthrs -> named tensors -> Candle weight binding -> RVC generator
.index -> pthrs -> retrieval search --------------------------^
audio -> resample -> content encoder -> F0 -> feature blend ---|
RVC generator -> boundary handling -> WAV output
```

The first milestone bypasses audio decoding, content encoding, F0 extraction,
and retrieval by loading fixed generator-ready tensors. This isolates model
correctness from the rest of the pipeline.

## Model implementation order

1. exact checkpoint-to-Candle tensor adapter;
2. speaker and pitch embeddings;
3. content/text encoder;
4. residual coupling flow;
5. NSF source path;
6. HiFi-GAN decoder and upsampling blocks;
7. top-level synthesizer forward pass.

Every block receives its own fixed inputs and recorded PyTorch output. Do not
wait for final audio to discover a low-level padding or layout mismatch.

## Planned streaming design

```text
high-priority input callback -> bounded SPSC input buffer
bounded input -> inference worker -> bounded SPSC output buffer
bounded output -> high-priority output callback
```

Callbacks only convert sample formats and move bounded slices. They do not:

- allocate;
- log;
- read files;
- acquire a contended mutex;
- resample;
- run neural inference.

Model loading, index loading, device initialization, buffer reservation, and FFT
planning happen before audio streams start.

## Backend policy

Candle is the first direct-checkpoint backend. Do not generalize over several
tensor frameworks before the first generator works. `rvc-rs-core` remains
backend-independent so ONNX Runtime, Burn, or another implementation can be
added later without changing front ends.

