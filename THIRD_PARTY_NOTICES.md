# Third-party notices

`rvc-rs` is licensed under the Mozilla Public License 2.0. It uses third-party
dependencies under their respective licenses.

## blazen-audio-vc ContentVec portions

These files contain portions adapted from `blazen-audio-vc` 0.7.0
(<https://crates.io/crates/blazen-audio-vc/0.7.0>), also licensed under the
Mozilla Public License 2.0:

- `crates/rvc-rs-candle/src/content.rs`
- `crates/rvc-rs-candle/src/feature_extractor.rs`
- `crates/rvc-rs-candle/src/hubert.rs`
- `crates/rvc-rs-candle/src/hubert_encoder.rs`

## Managed models

On first use, `rvc-rs` downloads `hubert_base.pt` and `rmvpe.pt` from the
MIT-licensed `lj1995/VoiceConversionWebUI` model repository at pinned
revisions. They are cached outside this source tree and are not part of the
MPL-2.0 licensed source distribution. See `docs/MODEL_ASSETS.md`.

## User model files

RVC voice models and retrieval indexes are user-supplied data. They are not
part of this source tree and are not covered by the `rvc-rs` license. Users are
responsible for complying with each model's license and data rights.
