#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Audio decoding, WAV encoding, and device contracts kept outside model code.

use std::fs::File;
use std::path::{Path, PathBuf};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use thiserror::Error;

/// Fully decoded mono floating-point audio.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAudio {
    /// Interleaved channels mixed to mono in the normalized `-1.0..=1.0` range.
    pub samples: Vec<f32>,
    /// Source samples per second.
    pub sample_rate: u32,
}

/// Decodes the first audio track recognized by Symphonia and mixes it to mono.
///
/// WAV, FLAC, MP3, Ogg/Vorbis, AAC/MP4 and other formats enabled by Symphonia
/// are accepted. The source sample rate is preserved for the conversion output.
pub fn decode_audio_mono(path: impl AsRef<Path>) -> Result<DecodedAudio, AudioError> {
    let path = path.as_ref();
    let source = File::open(path).map_err(|source| AudioError::OpenFile {
        path: path.to_owned(),
        source,
    })?;
    let stream = MediaSourceStream::new(Box::new(source), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|source| AudioError::Decode {
            path: path.to_owned(),
            message: source.to_string(),
        })?;
    let mut format = probed.format;
    let track = format.default_track().ok_or_else(|| AudioError::Decode {
        path: path.to_owned(),
        message: "no supported audio track was found".to_owned(),
    })?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioError::Decode {
            path: path.to_owned(),
            message: "the source sample rate is unavailable".to_owned(),
        })?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|source| AudioError::Decode {
            path: path.to_owned(),
            message: source.to_string(),
        })?;

    let mut mono = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(source))
                if source.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(AudioError::Decode {
                    path: path.to_owned(),
                    message: "the audio stream changed format while decoding".to_owned(),
                });
            }
            Err(source) => {
                return Err(AudioError::Decode {
                    path: path.to_owned(),
                    message: source.to_string(),
                });
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(source) => {
                return Err(AudioError::Decode {
                    path: path.to_owned(),
                    message: source.to_string(),
                });
            }
        };
        let channels = decoded.spec().channels.count();
        if channels == 0 {
            continue;
        }
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buffer.copy_interleaved_ref(decoded);
        mono.extend(
            buffer
                .samples()
                .chunks(channels)
                .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32),
        );
    }

    if mono.is_empty() {
        return Err(AudioError::Decode {
            path: path.to_owned(),
            message: "the selected track decoded to no samples".to_owned(),
        });
    }
    Ok(DecodedAudio {
        samples: mono,
        sample_rate,
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
