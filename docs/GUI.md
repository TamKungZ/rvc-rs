# Desktop app

`rvc-rs-inference` selects a `.pth`, an optional `.index`, and a device, and
converts a WAV file with the same engine as the CLI. Microphone conversion is
not available yet.

Presets (**Balanced**, **Clean speech**, **Singing**, **Strong identity**)
change only quality settings; device, pitch shift, and speaker stay as they
are. Every field remains editable afterwards.

The main panel holds retrieval rate, protection, generator noise, and RMS mix.
The advanced panel holds retrieval K/nprobe, the F0 method, the YIN-only F0
range, threshold, and median radius, and output gain.
