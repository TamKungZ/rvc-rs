# Development UI

The egui application is not the current critical path. It can select a native
RVC `.pth`, optional `.index`, and Candle device, then run **Prepare native
model** to validate metadata and transfer model/index data into resident native
state. ContentVec is deliberately not selectable: the engine downloads,
verifies, and reuses its mandatory managed HuBERT asset.

Offline conversion is enabled. Microphone start remains disabled until the
resident streaming worker and CPAL device path are connected. UI work must not
delay model, retrieval, or real-time engine work.
