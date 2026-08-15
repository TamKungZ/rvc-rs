# Third-party notices

`rvc-rs` is Apache-2.0 licensed. It uses third-party dependencies under their
respective licenses.

## blazen-audio-vc ContentVec portions

The ContentVec implementation files identified in `LICENSES/MPL-2.0.txt`
contain portions adapted from `blazen-audio-vc` 0.7.0 and remain available
under Mozilla Public License 2.0. The complete license and affected-file list
are included there.

## Managed HuBERT model

On first inference, `rvc-rs` downloads `hubert_base.pt` from the MIT-licensed
`lj1995/VoiceConversionWebUI` model repository at immutable revision
`1c75048c96f23f99da4b12909b532b5983290d7d`. Its SHA-256 is
`f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96`.
The model is cached outside this source tree and is not part of the Apache-2.0
licensed source distribution. See `docs/MODEL_ASSETS.md`.

## User model files

RVC voice, future RMVPE, and retrieval-index files are user-supplied data. They
are not part of this source tree and are not covered by the `rvc-rs` license.
Users are responsible for complying with each model's license and data rights.
