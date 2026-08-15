#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Backend-independent RVC model and inference contracts.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Supported exported RVC architecture generations.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ModelVersion {
    /// RVC v1, normally using 256-dimensional content features.
    V1,
    /// RVC v2, normally using 768-dimensional content features.
    V2,
}

impl ModelVersion {
    /// Returns the standard content feature width for this architecture.
    pub const fn feature_dimension(self) -> usize {
        match self {
            Self::V1 => 256,
            Self::V2 => 768,
        }
    }
}

/// Output sample rates represented by common exported RVC checkpoints.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum SampleRate {
    /// 32,000 samples per second.
    Hz32000,
    /// 40,000 samples per second.
    Hz40000,
    /// 48,000 samples per second.
    Hz48000,
}

impl SampleRate {
    /// Returns the numeric rate in hertz.
    pub const fn hz(self) -> u32 {
        match self {
            Self::Hz32000 => 32_000,
            Self::Hz40000 => 40_000,
            Self::Hz48000 => 48_000,
        }
    }
}

/// Tensor execution target selected for model inference.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum ComputeDevice {
    /// Prefer an available accelerator and fall back to CPU.
    #[default]
    Auto,
    /// Always use CPU inference.
    Cpu,
    /// Use the CUDA device at the supplied zero-based index.
    Cuda(usize),
    /// Use the Metal device at the supplied zero-based index.
    Metal(usize),
}

/// Facts needed to construct and validate one exported RVC model.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelSpec {
    /// Exported RVC architecture version.
    pub version: ModelVersion,
    /// Synthesizer output sample rate.
    pub sample_rate: SampleRate,
    /// Whether coarse and continuous F0 tracks are required.
    pub uses_f0: bool,
    /// Number of speaker embeddings in the checkpoint.
    pub speaker_count: usize,
}

impl ModelSpec {
    /// Returns the content feature width implied by the architecture version.
    pub const fn feature_dimension(self) -> usize {
        self.version.feature_dimension()
    }

    /// Validates model invariants independent of the tensor backend.
    pub fn validate(self) -> Result<(), InputError> {
        if self.speaker_count == 0 {
            return Err(InputError::InvalidModelSpec(
                "speaker_count must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// A row-major content feature matrix with shape `[frames, dimensions]`.
#[derive(Clone, Copy, Debug)]
pub struct FeatureMatrix<'a> {
    /// Contiguous row-major feature values.
    pub values: &'a [f32],
    /// Number of time frames.
    pub frames: usize,
    /// Number of values per frame.
    pub dimensions: usize,
}

/// Pitch values aligned one-to-one with content feature frames.
#[derive(Clone, Copy, Debug)]
pub struct PitchTrack<'a> {
    /// Quantized pitch bins supplied to the pitch embedding.
    pub coarse: &'a [i64],
    /// Continuous fundamental frequency in hertz.
    pub continuous_hz: &'a [f32],
}

/// Validated inputs accepted by a generator forward pass.
#[derive(Clone, Copy, Debug)]
pub struct GeneratorInput<'a> {
    /// Content features from ContentVec or a compatible encoder.
    pub features: FeatureMatrix<'a>,
    /// Pitch tracks for an F0 model, or `None` for a non-F0 model.
    pub pitch: Option<PitchTrack<'a>>,
    /// Zero-based speaker embedding index.
    pub speaker_id: usize,
}

impl GeneratorInput<'_> {
    /// Checks shapes and model-dependent invariants before inference.
    pub fn validate(&self, spec: ModelSpec) -> Result<(), InputError> {
        spec.validate()?;

        if self.features.frames == 0 {
            return Err(InputError::EmptyFeatures);
        }

        let expected_values = self
            .features
            .frames
            .checked_mul(self.features.dimensions)
            .ok_or(InputError::FeatureLengthOverflow)?;

        if self.features.values.len() != expected_values {
            return Err(InputError::FeatureLength {
                expected: expected_values,
                actual: self.features.values.len(),
            });
        }

        let expected_dimensions = spec.feature_dimension();
        if self.features.dimensions != expected_dimensions {
            return Err(InputError::FeatureDimension {
                expected: expected_dimensions,
                actual: self.features.dimensions,
            });
        }

        match (spec.uses_f0, self.pitch) {
            (true, None) => return Err(InputError::MissingPitch),
            (false, Some(_)) => return Err(InputError::UnexpectedPitch),
            (_, _) => {}
        }

        if let Some(pitch) = self.pitch {
            if pitch.coarse.len() != self.features.frames {
                return Err(InputError::PitchLength {
                    field: "coarse",
                    expected: self.features.frames,
                    actual: pitch.coarse.len(),
                });
            }
            if pitch.continuous_hz.len() != self.features.frames {
                return Err(InputError::PitchLength {
                    field: "continuous_hz",
                    expected: self.features.frames,
                    actual: pitch.continuous_hz.len(),
                });
            }
        }

        if self.speaker_id >= spec.speaker_count {
            return Err(InputError::SpeakerOutOfRange {
                speaker_id: self.speaker_id,
                speaker_count: spec.speaker_count,
            });
        }

        Ok(())
    }
}

/// Errors found before a tensor backend is invoked.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum InputError {
    /// The checkpoint-derived model specification is invalid.
    #[error("invalid model spec: {0}")]
    InvalidModelSpec(&'static str),
    /// No content frames were supplied.
    #[error("content features contain no frames")]
    EmptyFeatures,
    /// `frames * dimensions` overflowed `usize`.
    #[error("content feature shape overflowed")]
    FeatureLengthOverflow,
    /// The content slice does not match its declared matrix shape.
    #[error("content feature length mismatch: expected {expected}, got {actual}")]
    FeatureLength {
        /// Required number of values.
        expected: usize,
        /// Supplied number of values.
        actual: usize,
    },
    /// The content feature width is incompatible with the model version.
    #[error("content feature dimension mismatch: expected {expected}, got {actual}")]
    FeatureDimension {
        /// Width required by the model.
        expected: usize,
        /// Width supplied by the caller.
        actual: usize,
    },
    /// An F0 model was called without pitch tracks.
    #[error("F0 model requires pitch tracks")]
    MissingPitch,
    /// A non-F0 model was called with pitch tracks.
    #[error("non-F0 model does not accept pitch tracks")]
    UnexpectedPitch,
    /// A pitch track is not aligned with the content frames.
    #[error("pitch field {field} length mismatch: expected {expected}, got {actual}")]
    PitchLength {
        /// Name of the mismatched pitch field.
        field: &'static str,
        /// Required number of frames.
        expected: usize,
        /// Supplied number of frames.
        actual: usize,
    },
    /// The selected speaker embedding does not exist.
    #[error("speaker id {speaker_id} is outside 0..{speaker_count}")]
    SpeakerOutOfRange {
        /// Requested zero-based speaker index.
        speaker_id: usize,
        /// Number of available speaker embeddings.
        speaker_count: usize,
    },
}

/// A backend capable of executing an initialized RVC generator.
pub trait VoiceGenerator {
    /// Backend-specific failure type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Describes the initialized model.
    fn spec(&self) -> ModelSpec;

    /// Synthesizes mono floating-point audio from generator-ready inputs.
    fn synthesize(&mut self, input: &GeneratorInput<'_>) -> Result<Vec<f32>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v2_f0_spec() -> ModelSpec {
        ModelSpec {
            version: ModelVersion::V2,
            sample_rate: SampleRate::Hz40000,
            uses_f0: true,
            speaker_count: 1,
        }
    }

    #[test]
    fn validates_v2_f0_input() {
        let values = vec![0.0; 2 * 768];
        let coarse = [1, 2];
        let continuous_hz = [110.0, 111.0];
        let input = GeneratorInput {
            features: FeatureMatrix {
                values: &values,
                frames: 2,
                dimensions: 768,
            },
            pitch: Some(PitchTrack {
                coarse: &coarse,
                continuous_hz: &continuous_hz,
            }),
            speaker_id: 0,
        };

        assert_eq!(input.validate(v2_f0_spec()), Ok(()));
    }

    #[test]
    fn rejects_wrong_dimension() {
        let values = vec![0.0; 256];
        let coarse = [1];
        let continuous_hz = [110.0];
        let input = GeneratorInput {
            features: FeatureMatrix {
                values: &values,
                frames: 1,
                dimensions: 256,
            },
            pitch: Some(PitchTrack {
                coarse: &coarse,
                continuous_hz: &continuous_hz,
            }),
            speaker_id: 0,
        };

        assert_eq!(
            input.validate(v2_f0_spec()),
            Err(InputError::FeatureDimension {
                expected: 768,
                actual: 256,
            })
        );
    }

    #[test]
    fn rejects_missing_pitch() {
        let values = vec![0.0; 768];
        let input = GeneratorInput {
            features: FeatureMatrix {
                values: &values,
                frames: 1,
                dimensions: 768,
            },
            pitch: None,
            speaker_id: 0,
        };

        assert_eq!(input.validate(v2_f0_spec()), Err(InputError::MissingPitch));
    }
}

