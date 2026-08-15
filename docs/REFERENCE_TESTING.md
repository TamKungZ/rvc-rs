# Reference testing

## Why it is mandatory

A model may produce finite, voice-like audio while still being mathematically
wrong. Listening and shape checks cannot reliably locate convolution padding,
tensor layout, weight normalization, random noise, or flow mistakes.

Python and PyTorch are allowed only to generate trusted development references.
They are not runtime dependencies.

## Suggested local bundle

```text
tests/fixtures/local-v2-40k-f0/
├── manifest.json
├── model.pth
├── input/
│   ├── phone.npy
│   ├── phone_lengths.npy
│   ├── pitch.npy
│   ├── pitchf.npy
│   └── sid.npy
└── expected/
    ├── enc_p.npz
    ├── flow.npz
    ├── decoder-blocks.npz
    └── waveform.npy
```

The binary data is ignored by Git. A redacted manifest may be committed.

## Rules

- Fix Python, PyTorch, NumPy, upstream RVC, and CUDA revisions.
- Fix all random seeds and put the model in evaluation mode.
- Record input and output names, dtypes, shapes, and hashes.
- Disable nondeterministic paths where possible.
- Capture intermediate results before Rust optimization or fusion.
- Compare maximum absolute error, relative error, and RMSE.
- Reject NaN and infinity before numerical comparison.
- Establish CPU `f32` tolerance before accelerator tolerances.
- Document every relaxed tolerance with the responsible operation.

## Provenance

Follow the existing pthrs compatibility convention:

- forgotten user-supplied sources: `Local fixture — original source unknown`;
- public sources: immutable URL or pinned revision;
- corrupt sources: record the rejection and never change correct code to accept
  truncated data.

