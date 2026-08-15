# 0.4.2 real-audio regression

This release candidate is based on a user-supplied RVC v2/40k/F0 checkpoint,
matching IVF-Flat index, and an 8.18-second singing WAV. The binary fixtures
remain local and are not distributed with the source tree.

## Fixture identity

Checkpoint metadata: RVC `v2`, `40k`, F0 enabled, 109 speaker embeddings,
training label `250epoch`.

## 0.4.1 failure signature

- Output was finite and unclipped but perceptually unusable.
- Spectral centroid rose from roughly 0.78 kHz to 1.42 kHz.
- Zero-crossing rate rose from roughly 0.048 to 0.102.
- The autocorrelation F0 path marked 93.9% of frames voiced and produced
  false 840-1067 Hz spikes in breath/silence regions.

## Corrections in 0.4.2-rc.1

- HuBERT feature extractor uses fairseq's `GroupNorm(512, 512)`.
- HuBERT transformer layers use post-norm for `layer_norm_first=false`.
- Content features are doubled with nearest-neighbor expansion.
- Input sample-rate conversion uses a windowed-sinc anti-aliasing filter.
- F0 uses YIN local-minimum selection, voicing rejection, and sub-lag
  interpolation.
- NSF excitation follows RVC's frame-wise phase accumulation and Gaussian
  noise behavior.
- The decoder's final default LeakyReLU slope is `0.01`.

## Required validation

Run the same source first without retrieval, then with the matching index:

```bash
cargo run --release -p rvc-rs-cli -- convert \
  test.pth - test.wav output-no-index.wav 0 auto

cargo run --release -p rvc-rs-cli -- convert \
  test.pth test.index test.wav output-index.wav 0 auto
```

Do not promote the release candidate until both outputs are listened to and
the generator is compared against captured PyTorch intermediate tensors.
