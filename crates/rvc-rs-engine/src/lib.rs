#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Shared orchestration state for the GUI, CLI, and future streaming worker.

use rvc_rs_candle::{backend_smoke_test, resolve_device};
use rvc_rs_core::ComputeDevice;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// User-adjustable inference settings shared by every front end.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EngineConfig {
    /// Tensor execution target.
    pub device: ComputeDevice,
    /// Pitch transposition in semitones.
    pub pitch_shift: i8,
    /// Retrieval blend ratio in the inclusive range `0.0..=1.0`.
    pub retrieval_rate: f32,
    /// Target streaming chunk duration in milliseconds.
    pub chunk_ms: u16,
    /// Boundary crossfade duration in milliseconds.
    pub crossfade_ms: u16,
    /// Speaker embedding index for multi-speaker checkpoints.
    pub speaker_id: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            device: ComputeDevice::Auto,
            pitch_shift: 0,
            retrieval_rate: 0.75,
            chunk_ms: 160,
            crossfade_ms: 40,
            speaker_id: 0,
        }
    }
}

impl EngineConfig {
    /// Validates values without opening a checkpoint or audio device.
    pub fn validate(&self) -> Result<(), EngineError> {
        if !(-24..=24).contains(&self.pitch_shift) {
            return Err(EngineError::InvalidConfig(
                "pitch_shift must be between -24 and 24 semitones",
            ));
        }
        if !(0.0..=1.0).contains(&self.retrieval_rate) {
            return Err(EngineError::InvalidConfig(
                "retrieval_rate must be between 0.0 and 1.0",
            ));
        }
        if !(20..=2_000).contains(&self.chunk_ms) {
            return Err(EngineError::InvalidConfig(
                "chunk_ms must be between 20 and 2000",
            ));
        }
        if self.crossfade_ms >= self.chunk_ms {
            return Err(EngineError::InvalidConfig(
                "crossfade_ms must be smaller than chunk_ms",
            ));
        }
        Ok(())
    }
}

/// Files that identify one voice model and its optional retrieval index.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelFiles {
    /// Exported RVC inference checkpoint.
    pub checkpoint: PathBuf,
    /// Optional FAISS IVF-Flat index.
    pub index: Option<PathBuf>,
}

impl ModelFiles {
    /// Validates extensions and local file presence.
    pub fn validate(&self) -> Result<(), EngineError> {
        validate_file(&self.checkpoint, "pth", "checkpoint")?;
        if let Some(index) = &self.index {
            validate_file(index, "index", "retrieval index")?;
        }
        Ok(())
    }
}

/// One file-to-file conversion request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OfflineJob {
    /// Source audio to convert.
    pub input_audio: PathBuf,
    /// Destination WAV path.
    pub output_audio: PathBuf,
}

impl OfflineJob {
    /// Validates required paths without decoding audio.
    pub fn validate(&self) -> Result<(), EngineError> {
        if !self.input_audio.is_file() {
            return Err(EngineError::MissingFile {
                field: "input audio",
                path: self.input_audio.clone(),
            });
        }
        if self.output_audio.as_os_str().is_empty() {
            return Err(EngineError::MissingPath("output audio"));
        }
        if self.output_audio.extension().and_then(|value| value.to_str()) != Some("wav") {
            return Err(EngineError::WrongExtension {
                field: "output audio",
                expected: "wav",
                path: self.output_audio.clone(),
            });
        }
        Ok(())
    }
}

/// High-level lifecycle visible to the GUI and CLI.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum EngineState {
    /// No model is selected.
    #[default]
    Empty,
    /// Paths and settings are selected but the model is not loaded.
    Configured,
    /// Model loading and allocation are in progress.
    Preparing,
    /// Model and reusable workspaces are ready.
    Ready,
    /// An offline or real-time job is active.
    Running,
    /// The most recent operation failed.
    Failed(String),
}

impl EngineState {
    /// Returns a short stable label for front ends.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Empty => "No model",
            Self::Configured => "Configured",
            Self::Preparing => "Preparing",
            Self::Ready => "Ready",
            Self::Running => "Running",
            Self::Failed(_) => "Failed",
        }
    }
}

/// Shared engine handle. Model inference will be added behind this API.
#[derive(Debug, Default)]
pub struct Engine {
    config: EngineConfig,
    model: Option<ModelFiles>,
    state: EngineState,
}

impl Engine {
    /// Creates an engine using safe default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns current settings.
    pub const fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Replaces settings after validation.
    pub fn set_config(&mut self, config: EngineConfig) -> Result<(), EngineError> {
        config.validate()?;
        self.config = config;
        self.refresh_state();
        Ok(())
    }

    /// Returns the selected model paths.
    pub fn model(&self) -> Option<&ModelFiles> {
        self.model.as_ref()
    }

    /// Replaces model paths without opening model data.
    pub fn set_model(&mut self, model: ModelFiles) {
        self.model = Some(model);
        self.refresh_state();
    }

    /// Returns the current lifecycle state.
    pub const fn state(&self) -> &EngineState {
        &self.state
    }

    /// Checks a selected device by performing a Candle tensor round trip.
    pub fn doctor(&self) -> Result<Vec<f32>, EngineError> {
        let device = resolve_device(self.config.device)?;
        backend_smoke_test(&device).map_err(EngineError::Inference)
    }

    /// Validates all inputs needed for an offline conversion.
    pub fn validate_offline(&self, job: &OfflineJob) -> Result<(), EngineError> {
        self.config.validate()?;
        self.model.as_ref().ok_or(EngineError::NoModel)?.validate()?;
        job.validate()?;
        Ok(())
    }

    /// Starts an offline job once the generator exists.
    pub fn start_offline(&mut self, job: &OfflineJob) -> Result<(), EngineError> {
        self.validate_offline(job)?;
        let error = EngineError::FeatureUnavailable(
            "RVC generator forward pass has not reached reference parity yet",
        );
        self.state = EngineState::Failed(error.to_string());
        Err(error)
    }

    fn refresh_state(&mut self) {
        self.state = if self.model.is_some() {
            EngineState::Configured
        } else {
            EngineState::Empty
        };
    }
}

fn validate_file(
    path: &Path,
    extension: &'static str,
    field: &'static str,
) -> Result<(), EngineError> {
    if !path.is_file() {
        return Err(EngineError::MissingFile {
            field,
            path: path.to_owned(),
        });
    }
    if path.extension().and_then(|value| value.to_str()) != Some(extension) {
        return Err(EngineError::WrongExtension {
            field,
            expected: extension,
            path: path.to_owned(),
        });
    }
    Ok(())
}

/// Configuration, path, backend, and unfinished-feature errors.
#[derive(Debug, Error)]
pub enum EngineError {
    /// A setting violates a documented range or relationship.
    #[error("invalid engine configuration: {0}")]
    InvalidConfig(&'static str),
    /// A required path was not supplied.
    #[error("missing required path: {0}")]
    MissingPath(&'static str),
    /// No voice model was selected.
    #[error("no voice model is selected")]
    NoModel,
    /// A selected local file does not exist.
    #[error("{field} file does not exist: {path}", path = path.display())]
    MissingFile {
        /// Name shown to the caller.
        field: &'static str,
        /// Missing path.
        path: PathBuf,
    },
    /// A path has an incompatible extension.
    #[error("{field} must use .{expected}: {path}", path = path.display())]
    WrongExtension {
        /// Name shown to the caller.
        field: &'static str,
        /// Required extension without a leading dot.
        expected: &'static str,
        /// Rejected path.
        path: PathBuf,
    },
    /// Candle initialization or inference failed.
    #[error(transparent)]
    Inference(#[from] rvc_rs_candle::InferenceError),
    /// A deliberately gated project milestone is not implemented.
    #[error("feature unavailable: {0}")]
    FeatureUnavailable(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_crossfade_larger_than_chunk() {
        let config = EngineConfig {
            chunk_ms: 40,
            crossfade_ms: 40,
            ..EngineConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(EngineError::InvalidConfig(_))
        ));
    }

    #[test]
    fn selecting_a_model_changes_state_without_claiming_readiness() {
        let mut engine = Engine::new();
        engine.set_model(ModelFiles {
            checkpoint: PathBuf::from("voice.pth"),
            index: None,
        });
        assert_eq!(engine.state(), &EngineState::Configured);
    }
}
