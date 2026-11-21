# Managed model assets

Every conversion needs RVC's shared HuBERT content encoder, and RMVPE when it
is the selected F0 method. The engine downloads each file once into the user
cache, writes it to a temporary file, checks the exact size and SHA-256, and
then moves it into place. Cached files are checked again before every
conversion and replaced if they do not match.

| File | Revision of `lj1995/VoiceConversionWebUI` | Size | SHA-256 |
|---|---|---|---|
| `hubert_base.pt` | `1c75048c96f23f99da4b12909b532b5983290d7d` | 189507909 | `f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96` |
| `rmvpe.pt` | `e6d0c1a17da07c33557852f9dfa2bd44cc75737d` | 181184272 | `6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193` |

The repository is MIT licensed. The files are downloaded at runtime and are
not part of this source tree.
