# rvc-rs

Native Rust voice conversion for RVC models. It loads RVC `.pth` checkpoints
and FAISS `.index` files directly and runs inference with Candle, without
Python, PyTorch, ONNX Runtime, or FAISS.

Output matches the reference RVC WebUI pipeline within its own run-to-run
variation.

## Supported

- RVC v2 models with pitch (F0); 40 and 48 kHz are verified, 32 kHz is
  untested
- optional `.index` retrieval
- WAV input and output
- CPU; the `cuda` and `metal` Cargo features exist but are untested

Not yet supported: real-time microphone conversion, v1 models, and models
without pitch.

## Usage

Download a build from [Releases](https://github.com/TamKungZ/rvc-rs/releases).
The package contains the `rvc-rs` command-line tool and the `rvc-rs-inference`
desktop app.

```bash
rvc-rs convert voice.pth voice.index input.wav output.wav
```

Use `-` in place of the index to disable retrieval. Two optional positional
arguments follow the output path: pitch shift in semitones and device
(`auto`, `cpu`, `cuda:N`, `metal:N`):

```bash
rvc-rs convert voice.pth - input.wav output.wav 12 cpu
```

Common options:

| Option | Meaning |
|---|---|
| `--preset <balanced\|clean\|singing\|identity>` | starting settings; `balanced` matches RVC WebUI defaults |
| `--index-rate <0..1>` | retrieval strength |
| `--protect <0..0.5>` | keep consonants and breaths from the source; `0.5` disables |
| `--rms-mix <0..1>` | `0` follows the source loudness, `1` keeps the model's |
| `--f0-method <rmvpe\|yin>` | pitch extractor; RMVPE is the default |
| `--speaker <id>` | speaker of a multi-speaker model |

Run `rvc-rs --help` for the full list.

## Downloaded models

The first conversion downloads two shared RVC models, `hubert_base.pt` and
`rmvpe.pt`, into the user cache directory (`%LOCALAPPDATA%\rvc-rs\models` on
Windows, `~/Library/Caches/rvc-rs/models` on macOS, `~/.cache/rvc-rs/models`
on Linux). Both are pinned and checked with SHA-256 before use. See
[docs/MODEL_ASSETS.md](docs/MODEL_ASSETS.md).

## License

[Mozilla Public License 2.0](LICENSE). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party portions.
