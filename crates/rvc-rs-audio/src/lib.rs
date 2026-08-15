#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Audio decoding, WAV encoding, and device contracts kept outside model code.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// Fully decoded mono floating-point audio.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAudio {
    /// Interleaved channels mixed to mono in the normalized `-1.0..=1.0` range.
    pub samples: Vec<f32>,
    /// Source samples per second.
    pub sample_rate: u32,
}

/// Decodes PCM or floating-point WAV and mixes all channels to mono.
pub fn decode_audio_mono(path: impl AsRef<Path>) -> Result<DecodedAudio, AudioError> {
    let path = path.as_ref();
    let mut reader = hound::WavReader::open(path).map_err(|source| AudioError::Decode {
        path: path.to_owned(),
        message: source.to_string(),
    })?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 {
        return Err(AudioError::Decode {
            path: path.to_owned(),
            message: "WAV has zero channels".to_owned(),
        });
    }
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| AudioError::Decode {
                path: path.to_owned(),
                message: source.to_string(),
            })?,
        hound::SampleFormat::Int => {
            let peak = ((1_u64 << spec.bits_per_sample.saturating_sub(1)) - 1) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / peak))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| AudioError::Decode {
                    path: path.to_owned(),
                    message: source.to_string(),
                })?
        }
    };
    let mono: Vec<f32> = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();

    if mono.is_empty() {
        return Err(AudioError::Decode {
            path: path.to_owned(),
            message: "the selected track decoded to no samples".to_owned(),
        });
    }
    Ok(DecodedAudio {
        samples: mono,
        sample_rate: spec.sample_rate,
    })
}

/// Writes normalized mono audio as a 32-bit floating-point WAV file.
pub fn write_wav_mono(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), AudioError> {
    let path = path.as_ref();
    let specification = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(path, specification).map_err(|source| AudioError::Encode {
            path: path.to_owned(),
            message: source.to_string(),
        })?;
    for &sample in samples {
        writer
            .write_sample(sample.clamp(-1.0, 1.0))
            .map_err(|source| AudioError::Encode {
                path: path.to_owned(),
                message: source.to_string(),
            })?;
    }
    writer.finalize().map_err(|source| AudioError::Encode {
        path: path.to_owned(),
        message: source.to_string(),
    })
}

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
    /// A source file could not be opened.
    #[error("failed to open audio file {path}: {source}", path = path.display())]
    OpenFile {
        /// Source path.
        path: PathBuf,
        /// Operating-system error.
        #[source]
        source: std::io::Error,
    },
    /// The source container, codec, or decoded stream is invalid.
    #[error("failed to decode audio file {path}: {message}", path = path.display())]
    Decode {
        /// Source path.
        path: PathBuf,
        /// Decoder explanation.
        message: String,
    },
    /// The destination WAV could not be encoded.
    #[error("failed to write WAV file {path}: {message}", path = path.display())]
    Encode {
        /// Destination path.
        path: PathBuf,
        /// Encoder explanation.
        message: String,
    },
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
