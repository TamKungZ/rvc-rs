# Architecture

## Model loading

```text
.pth   -> pthrs -> named f32 tensors -> Candle device
.index -> pthrs IVF-Flat -> resident vectors + reusable search workspace
```

Loading is eager and happens before inference. HuBERT and RMVPE are managed
assets owned by the engine; front ends never supply their paths.

## Offline conversion

The engine follows RVC WebUI `infer/modules/vc/pipeline.py`:

```text
input -> 16 kHz -> peak limit -> 48 Hz zero-phase high-pass -> 1 s reflect pad
      -> RMVPE (or YIN) over the whole input
      -> split at the quietest point near every 38 s
      -> per segment: HuBERT layer 12 -> index blend -> 2x upsample
                      -> unvoiced protection -> generator -> trim padding
      -> RMS envelope mix -> peak limit -> WAV
```

On CPU the decoder residual blocks use a time-major im2col convolution with
fused leaky ReLU, bias, and residual additions. Other devices use plain Candle
operations.

## Real-time target

```text
input callback -> bounded SPSC queue -> inference worker
inference worker:
  rolling audio -> 16 kHz -> HuBERT + F0
  features -> index blend -> generator
  waveform -> SOLA/crossfade -> output queue
output callback <- bounded SPSC queue
```

Callbacks move fixed-size sample blocks only. Model execution, resampling,
retrieval, logging, file I/O, and allocation stay on the worker.

The block geometry follows MMVCServerSIO:

- audio, pitch, and feature history are kept across calls;
- the feature grid is 100 Hz, derived from output-rate context;
- each call covers block, crossfade, 12 ms SOLA search, and extra history;
- generator context is rounded up to 128 samples;
- retrieval is blended before 2x interpolation;
- consecutive outputs are aligned by normalized correlation and crossfaded;
- each cycle returns exactly one device block.
