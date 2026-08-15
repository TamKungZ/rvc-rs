# Development UI

The egui application is not the current critical path. It can select a native
RVC `.pth`, optional `.index`, and Candle device, then run **Prepare native
model** to validate metadata and transfer model/index data into resident native
state. ContentVec is deliberately not selectable: the engine downloads,
verifies, and reuses its mandatory managed HuBERT asset.

Offline conversion is enabled. Microphone start remains disabled until the
resident streaming worker and CPAL device path are connected. UI work must not
delay model, retrieval, or real-time engine work.

## Quality controls

The UI exposes four starting presets: **Balanced**, **Clean speech**,
**Singing**, and **Strong identity**. Selecting a preset changes only
quality-related fields; device, pitch shift, speaker ID, chunk size, and
crossfade remain untouched. All fields remain editable after applying a preset.

The main panel contains the controls most likely to matter during listening:
retrieval rate, consonant protection, generator noise, and RMS-envelope mix.
The collapsed advanced panel contains retrieval K/nprobe, the YIN F0 range and
threshold, voiced-frame median radius, and final gain. A protect value of `0.5`
disables feature protection; smaller values mix more of the pre-retrieval
ContentVec feature back into unvoiced frames. An RMS mix of `0` follows the
source envelope, while `1` leaves the generated envelope unchanged.
