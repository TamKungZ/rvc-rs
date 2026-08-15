# Desktop application

`rvc-rs-inference` is the user-facing egui/eframe application.

## Included now

- checkpoint and optional index selection;
- offline input and output selection;
- offline and real-time mode surfaces;
- device, pitch, retrieval, speaker, chunk, and crossfade controls;
- persisted form state;
- path and configuration validation;
- Candle backend doctor;
- visible engine state and activity log;
- explicit generator-not-ready failure instead of placeholder output.

## UI thread policy

File dialogs may block briefly because they are direct user actions. Model
loading, inference, audio decoding, file encoding, and device streaming must run
outside the egui thread. The future engine worker reports progress and events
back to the UI through bounded channels.

## Unlock order

1. backend doctor;
2. offline input validation;
3. model preparation progress;
4. offline conversion and cancellation;
5. output preview/reveal;
6. real-time device selection;
7. live levels, latency, underruns, and start/stop.

The real-time button remains disabled until offline waveform parity passes.

