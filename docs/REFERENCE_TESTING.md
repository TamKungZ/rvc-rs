# Reference testing

Audio that sounds plausible can still be numerically wrong, so every model
block is compared against PyTorch. Python and PyTorch are used only to create
these references; they are never runtime dependencies.

## Fixtures

Parity tests read `.dev/`, which is ignored by Git, and skip when a fixture is
missing:

```text
.dev/
├── model/<name>/*.pth, *.index
└── reference/
    ├── <name>/synth/   phone, pitch, pitchf, m_p, logs_p, z, wav
    ├── hubert/         audio, layer12
    └── rmvpe/          audio, mel, hidden, f0
```

All arrays are little-endian `.npy` files. HuBERT and RMVPE tests also use the
managed models from the user cache.

- `synth`: `SynthesizerTrnMs768NSFsid` from upstream RVC with the checkpoint's
  config, random phone features, a pitch contour with unvoiced gaps, and every
  noise source set to zero.
- `hubert`: layer-12 output of torchaudio's HuBERT-base loaded with
  `hubert_base.pt`.
- `rmvpe`: upstream `MelSpectrogram` and `E2E` output on a few seconds of
  16 kHz audio, plus the decoded F0 with threshold 0.03.

Upstream RVC code is taken from
`RVC-Project/Retrieval-based-Voice-Conversion-WebUI` at commit
`7ef19867780cf703841ebafb565a4e47d1ea86ff`.

## Whole-file checks

RVC samples noise, so two upstream runs never match sample for sample. Compare
the Rust output with one upstream run and use a second upstream run with
another seed as the noise floor for log-mel distance and F0 agreement.

## Rules

- Pin Python, PyTorch, upstream RVC, and model revisions.
- Fix seeds and use evaluation mode.
- Capture intermediate results before Rust optimization or fusion.
- Reject NaN and infinity before comparing values.
- Establish CPU `f32` tolerances before accelerator tolerances.
- Document every relaxed tolerance with the operation responsible.
