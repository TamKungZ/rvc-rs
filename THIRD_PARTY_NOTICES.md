# Third-party notices

`rvc-rs` is Apache-2.0 licensed. It uses third-party dependencies under their
respective licenses.

## blazen-audio-vc ContentVec portions

The ContentVec implementation files identified in `LICENSES/MPL-2.0.txt`
contain portions adapted from `blazen-audio-vc` 0.7.0 and remain available
under Mozilla Public License 2.0. The complete license and affected-file list
are included there.

## Managed models

On first use, `rvc-rs` downloads `hubert_base.pt` and `rmvpe.pt` from the
MIT-licensed `lj1995/VoiceConversionWebUI` model repository at pinned
revisions. They are cached outside this source tree and are not part of the
Apache-2.0 licensed source distribution. See `docs/MODEL_ASSETS.md`.

## User model files

RVC voice models and retrieval indexes are user-supplied data. They are not
part of this source tree and are not covered by the `rvc-rs` license. Users are
responsible for complying with each model's license and data rights.
