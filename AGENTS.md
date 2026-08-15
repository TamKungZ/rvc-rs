# Project instructions

Read `README.md` and every document under `docs/` before architectural or model
implementation changes.

## Current goal

```text
fixed content + pitch + speaker tensors
-> weights loaded from .pth by pthrs 0.2.0
-> Candle RVC v2/40k/F0 forward pass
-> mono waveform matching a trusted PyTorch reference
```

## Rules

- Python and PyTorch may create development reference fixtures, but must never
  become runtime dependencies.
- Keep checkpoint/index decoding and IVF-Flat retrieval inside `pthrs`.
- Keep model mathematics inside `rvc-rs-candle`.
- Keep backend-neutral contracts inside `rvc-rs-core`.
- Keep orchestration and state inside `rvc-rs-engine`.
- Keep UI code inside `rvc-rs-inference`; it must call the shared engine.
- Do not add microphone streaming before offline waveform parity passes.
- Do not emit placeholder audio or call the scaffold working inference.
- Do not silently reshape, transpose, truncate, rename, or ignore weights.
- Test model blocks against reference outputs before composing them.
- CPU `f32` is the correctness baseline. Acceleration follows correctness.
- Do not commit model weights, indexes, audio, NumPy arrays, or private paths.
- If `pthrs` needs a change, keep it generic and make it separately.
- Preserve the package boundaries documented in `docs/WORKSPACE.md`.

## Validation

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo doc --workspace --no-deps
```

Follow `docs/ROADMAP.md` in order unless a concrete failing test proves the
order must change.

