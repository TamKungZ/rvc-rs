# Development UI

The egui application is not the current critical path. It can select a native
RVC `.pth`, optional `.index`, and Candle device, then run **Prepare native
model** to validate metadata and transfer model/index data into resident native
state.

Offline conversion and microphone start remain disabled until the Candle
generator passes waveform parity. UI work must not delay model, retrieval, or
real-time engine work.
