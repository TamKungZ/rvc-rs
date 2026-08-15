# Managed model assets

## Mandatory HuBERT/ContentVec

Every RVC conversion uses one shared content encoder in addition to the selected
voice `.pth`. `rvc-rs` owns this runtime dependency and does not expose a model
path in the CLI, GUI, or `ModelFiles` API.

On first use the engine downloads the asset to the per-user OS cache through a
blocking Rust HTTPS client. The transfer is written to a process-specific
partial file. The engine verifies the exact byte length and SHA-256 digest
before atomically publishing it as `hubert_base.pt`. Existing cache entries are
also verified before inference; missing, truncated, or modified files are
replaced from the pinned source.

| Field | Value |
|---|---|
| Repository | `lj1995/VoiceConversionWebUI` |
| Revision | `1c75048c96f23f99da4b12909b532b5983290d7d` |
| Filename | `hubert_base.pt` |
| Size | `189507909` bytes |
| SHA-256 | `f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96` |
| Repository license | MIT |

The model bytes are downloaded at runtime and are not committed to, embedded
in, or licensed as part of the Apache-2.0 source tree.

## Dependency budget

Managed downloading adds only two direct crates to the engine:

- `ureq` 3.4.0 with default features disabled and only pure-Rust `rustls` HTTPS;
- `sha2` 0.10.9 for streaming SHA-256 verification.

No async runtime, JSON stack, gzip decoder, Hugging Face SDK, system OpenSSL,
Python, PyTorch, or shell command is used.
