#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Audio-device contracts that keep platform callbacks outside model code.

use thiserror::Error;

/// Direction in which an audio device moves samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceDirection {
    /// Capture samples from a microphone or virtual input.
    Input,
    /// Render samples to speakers or a virtual output.
    Output,
}

/// Stable information displayed by the CLI or GUI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioDeviceInfo {
    /// Backend-specific identifier persisted by the application.
    pub id: String,
    /// Human-readable device name.
    pub name: String,
    /// Whether the device captures or renders samples.
    pub direction: DeviceDirection,
    /// Whether the audio host currently marks this device as default.
    pub is_default: bool,
}

/// Negotiated audio stream format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamFormat {
    /// Samples per second for each channel.
    pub sample_rate: u32,
    /// Number of interleaved channels.
    pub channels: u16,
    /// Requested callback buffer size in frames, if fixed.
    pub buffer_frames: Option<u32>,
}

/// Platform-independent interface implemented by an audio host adapter.
pub trait AudioHost {
    /// Lists capture and playback devices without opening a stream.
    fn devices(&self) -> Result<Vec<AudioDeviceInfo>, AudioError>;
}

/// Audio host and stream setup failures.
#[derive(Debug, Error)]
pub enum AudioError {
    /// No matching input or output device exists.
    #[error("audio device is unavailable: {0}")]
    DeviceUnavailable(String),
    /// The requested stream format is unsupported.
    #[error("unsupported audio stream format: {0}")]
    UnsupportedFormat(String),
    /// A platform backend returned an error.
    #[error("audio backend error: {0}")]
    Backend(String),
}

/// Re-exports CPAL when the optional native backend is enabled.
#[cfg(feature = "cpal-backend")]
pub use cpal;

/// Returns the number of audio hosts reported by CPAL.
#[cfg(feature = "cpal-backend")]
pub fn available_cpal_host_count() -> usize {
    cpal::available_hosts().len()
}

