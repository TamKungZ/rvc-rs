# Changelog

## [1.0.0] - 2026-11-21

First stable release of offline RVC v2 conversion with output matching the
reference RVC WebUI pipeline.

### Added

- Native RMVPE pitch extraction, now the default F0 method. The model is
  downloaded once and checked against a pinned SHA-256.
- `--f0-method <rmvpe|yin>` in the CLI and an F0 method selector in the
  desktop app.
- Offline pipeline matching RVC WebUI: 48 Hz zero-phase high-pass, 1 s context
  padding, splits at quiet points every ~38 s, and RVC's RMS envelope mix.
- Parity tests for the generator, HuBERT, and RMVPE against local PyTorch
  references.

### Changed

- License changed from Apache-2.0 to MPL-2.0.
- The `balanced` preset and the default generator noise scale now match RVC
  WebUI defaults.
- Faster decoder on CPU; a 20 s clip converts about twice as fast.
- Managed model downloads are retried up to three times.

### Fixed

- Unnatural, distorted output caused by wrong dilations in the generator's
  flow WaveNet.
- RMVPE output near the end of the input, which now runs the GRU over the
  zero-padded frames like upstream.
- Out-of-memory risk on long inputs; RMVPE and HuBERT now process bounded
  windows and segments.
