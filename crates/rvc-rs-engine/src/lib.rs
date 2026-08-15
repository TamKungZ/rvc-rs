#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Shared orchestration state for the GUI, CLI, and future streaming worker.

mod assets;

pub use assets::{hubert_cache_path, hubert_is_cached, hubert_is_ready, AssetError};

use candle_core::IndexOp;
use rvc_rs_audio::{decode_audio_mono, write_wav_mono};
use rvc_rs_candle::{
    backend_smoke_test, resolve_device, CandleGenerator, ContentEncoder, RetrievalIndex, RvcVersion,
};
use rvc_rs_core::{
    ComputeDevice, FeatureMatrix, GeneratorInput, ModelVersion, PitchTrack, VoiceGenerator,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
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
            retrieval_rate: 0.0,
            chunk_ms: 1_000,
            crossfade_ms: 85,
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
    /// Native RVC inference checkpoint (`.pth`).
    pub checkpoint: PathBuf,
    /// Optional FAISS IVF-Flat index.
    pub index: Option<PathBuf>,
}

impl ModelFiles {
    /// Validates extensions and local file presence.
    pub fn validate(&self) -> Result<(), EngineError> {
        validate_file(&self.checkpoint, "pth", "voice model")?;
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
        if self
            .output_audio
            .extension()
            .and_then(|value| value.to_str())
            != Some("wav")
        {
            return Err(EngineError::WrongExtension {
                field: "output audio",
                expected: "wav",
                path: self.output_audio.clone(),
            });
        }
        if let Some(parent) = self
            .output_audio
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            if !parent.is_dir() {
                return Err(EngineError::MissingDirectory {
                    path: parent.to_owned(),
                });
            }
        }
        Ok(())
    }
}

/// Result metadata for a completed WAV conversion.
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineReport {
    /// Written destination path.
    pub output_audio: PathBuf,
    /// Number of mono output samples.
    pub samples: usize,
    /// Output sample rate.
    pub sample_rate: u32,
    /// Number of inference chunks.
    pub chunks: usize,
    /// End-to-end decode, inference, and encode time.
    pub elapsed: Duration,
    /// Time reported inside the model pipeline.
    pub inference_time: Duration,
}

/// Validated, owned offline work that can be moved to a background thread.
#[derive(Clone, Debug)]
pub struct OfflineTask {
    config: EngineConfig,
    model: ModelFiles,
    job: OfflineJob,
}

impl OfflineTask {
    /// Decodes, converts, and writes the requested audio file.
    pub fn run(self) -> Result<OfflineReport, EngineError> {
        let started = Instant::now();
        let decoded = decode_audio_mono(&self.job.input_audio)?;
        let samples_16k = linear_resample(&decoded.samples, decoded.sample_rate, 16_000);
        let mut generator = CandleGenerator::load(&self.model.checkpoint, self.config.device)?;
        let spec = generator.spec();
        let contentvec_path = assets::ensure_hubert()?;
        if spec.version != ModelVersion::V2 {
            return Err(EngineError::FeatureUnavailable(
                "the current native ContentVec path supports RVC v2 checkpoints only",
            ));
        }
        let encoder = ContentEncoder::load(&contentvec_path, generator.device(), RvcVersion::V2)?;
        let inference_started = Instant::now();
        let encoded = encoder.encode(&samples_16k)?;
        let (_, content_frames, dimensions) = encoded.dims3()?;
        let mut base_features = encoded.i(0)?.flatten_all()?.to_vec1::<f32>()?;

        if self.config.retrieval_rate > 0.0 {
            let index_path = self
                .model
                .index
                .as_ref()
                .ok_or(EngineError::MissingPath("retrieval index"))?;
            let mut index = RetrievalIndex::load(index_path, dimensions, 8)?;
            index.blend_frames(&mut base_features, self.config.retrieval_rate, 8, 1)?;
        }

        let features = upsample_features_2x(&base_features, content_frames, dimensions);
        let pitch = extract_pitch_autocorrelation(&samples_16k, self.config.pitch_shift);
        let frames = (features.len() / dimensions).min(pitch.len());
        if frames == 0 {
            return Err(EngineError::FeatureUnavailable(
                "source audio is too short to produce inference frames",
            ));
        }
        let features = &features[..frames * dimensions];
        let continuous: Vec<f32> = pitch.into_iter().take(frames).collect();
        let coarse: Vec<i64> = continuous.iter().copied().map(pitch_to_coarse).collect();
        let output = generator.synthesize(&GeneratorInput {
            features: FeatureMatrix {
                values: features,
                frames,
                dimensions,
            },
            pitch: Some(PitchTrack {
                coarse: &coarse,
                continuous_hz: &continuous,
            }),
            speaker_id: self.config.speaker_id,
        })?;
        let inference_time = inference_started.elapsed();
        let sample_rate = spec.sample_rate.hz();
        write_wav_mono(&self.job.output_audio, &output, sample_rate)?;
        Ok(OfflineReport {
            output_audio: self.job.output_audio,
            samples: output.len(),
            sample_rate,
            chunks: 1,
            elapsed: started.elapsed(),
            inference_time,
        })
    }
}

fn linear_resample(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == target_rate {
        return samples.to_vec();
    }
    let length = (samples.len() as u64 * u64::from(target_rate) / u64::from(source_rate)) as usize;
    let step = f64::from(source_rate) / f64::from(target_rate);
    (0..length)
        .map(|i| {
            let position = i as f64 * step;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let amount = (position - left as f64) as f32;
            samples[left] * (1.0 - amount) + samples[right] * amount
        })
        .collect()
}

fn upsample_features_2x(input: &[f32], frames: usize, dimensions: usize) -> Vec<f32> {
    if frames == 0 {
        return Vec::new();
    }
    let mut output = vec![0.0; frames * 2 * dimensions];
    for frame in 0..frames * 2 {
        let position = frame as f32 * 0.5;
        let left = (position.floor() as usize).min(frames - 1);
        let right = (left + 1).min(frames - 1);
        let amount = position - left as f32;
        for d in 0..dimensions {
            output[frame * dimensions + d] = input[left * dimensions + d] * (1.0 - amount)
                + input[right * dimensions + d] * amount;
        }
    }
    output
}

fn extract_pitch_autocorrelation(samples: &[f32], semitones: i8) -> Vec<f32> {
    const RATE: usize = 16_000;
    const HOP: usize = 160;
    const WINDOW: usize = 1_024;
    const MIN_LAG: usize = RATE / 1_100;
    const MAX_LAG: usize = RATE / 50;
    let frames = samples.len() / HOP;
    let shift = 2f32.powf(f32::from(semitones) / 12.0);
    let mut output = Vec::with_capacity(frames);
    for frame in 0..frames {
        let center = frame * HOP;
        let start = center.saturating_sub(WINDOW / 2);
        let end = (start + WINDOW).min(samples.len());
        let slice = &samples[start..end];
        let rms = if slice.is_empty() {
            0.0
        } else {
            (slice.iter().map(|x| x * x).sum::<f32>() / slice.len() as f32).sqrt()
        };
        if rms < 0.005 || slice.len() <= MAX_LAG + 2 {
            output.push(0.0);
            continue;
        }
        let mut best_lag = MIN_LAG;
        let mut best = -1.0_f32;
        for lag in MIN_LAG..=MAX_LAG.min(slice.len() / 2) {
            let mut correlation = 0.0;
            let mut left_energy = 1e-9;
            let mut right_energy = 1e-9;
            for i in (0..slice.len() - lag).step_by(2) {
                let a = slice[i];
                let b = slice[i + lag];
                correlation += a * b;
                left_energy += a * a;
                right_energy += b * b;
            }
            let normalized = correlation / (left_energy * right_energy).sqrt();
            if normalized > best {
                best = normalized;
                best_lag = lag;
            }
        }
        output.push(if best >= 0.35 {
            RATE as f32 / best_lag as f32 * shift
        } else {
            0.0
        });
    }
    output
}

fn pitch_to_coarse(hz: f32) -> i64 {
    if hz <= 0.0 {
        return 1;
    }
    let mel = 1127.0 * (1.0 + hz / 700.0).ln();
    let min = 1127.0_f32 * (1.0_f32 + 50.0_f32 / 700.0_f32).ln();
    let max = 1127.0_f32 * (1.0_f32 + 1_100.0_f32 / 700.0_f32).ln();
    (((mel - min) * 254.0 / (max - min) + 1.0).round() as i64).clamp(1, 255)
}

/// Fully resident native model state intended to be owned by the inference worker.
#[derive(Debug)]
pub struct PreparedNativeModel {
    generator: CandleGenerator,
    content_encoder: ContentEncoder,
    retrieval: Option<RetrievalIndex>,
}

impl PreparedNativeModel {
    /// Loaded RVC generator checkpoint and its Candle tensors.
    pub const fn generator(&self) -> &CandleGenerator {
        &self.generator
    }

    /// Mandatory managed ContentVec/HuBERT encoder.
    pub const fn content_encoder(&self) -> &ContentEncoder {
        &self.content_encoder
    }

    /// Optional in-memory FAISS retrieval index.
    pub const fn retrieval(&self) -> Option<&RetrievalIndex> {
        self.retrieval.as_ref()
    }

    /// Mutable retrieval state for allocation-free frame blending.
    pub fn retrieval_mut(&mut self) -> Option<&mut RetrievalIndex> {
        self.retrieval.as_mut()
    }
}

/// Facts reported after native model preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativePreparationReport {
    /// Number of generator tensors resident on the Candle device.
    pub tensor_count: usize,
    /// RVC content feature width (256 for v1, 768 for v2).
    pub feature_dimension: usize,
    /// Generator output sample rate.
    pub sample_rate: u32,
    /// Number of speaker embeddings.
    pub speaker_count: usize,
    /// Whether the generator consumes F0 tracks.
    pub uses_f0: bool,
    /// Number of vectors loaded from the optional retrieval index.
    pub index_vectors: Option<u64>,
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
    prepared: Option<PreparedNativeModel>,
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
        self.prepared = None;
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
        self.prepared = None;
        self.refresh_state();
    }

    /// Returns the current lifecycle state.
    pub const fn state(&self) -> &EngineState {
        &self.state
    }

    /// Returns native model state after [`Engine::prepare_native`] succeeds.
    pub const fn prepared_native(&self) -> Option<&PreparedNativeModel> {
        self.prepared.as_ref()
    }

    /// Opens `.pth`, transfers all generator weights to Candle, and loads the
    /// optional `.index` into its reusable real-time search workspace.
    pub fn prepare_native(&mut self) -> Result<NativePreparationReport, EngineError> {
        self.config.validate()?;
        let model = self.model.clone().ok_or(EngineError::NoModel)?;
        model.validate()?;
        self.state = EngineState::Preparing;

        let result = (|| {
            let generator = CandleGenerator::load(&model.checkpoint, self.config.device)?;
            let spec = generator.spec();
            if spec.version != ModelVersion::V2 {
                return Err(EngineError::FeatureUnavailable(
                    "the current managed ContentVec path supports RVC v2 checkpoints only",
                ));
            }
            let hubert_path = assets::ensure_hubert()?;
            let content_encoder =
                ContentEncoder::load(&hubert_path, generator.device(), RvcVersion::V2)?;
            let retrieval = model
                .index
                .as_ref()
                .map(|path| RetrievalIndex::load(path, spec.feature_dimension(), 8))
                .transpose()?;
            if self.config.retrieval_rate > 0.0 && retrieval.is_none() {
                return Err(EngineError::MissingPath("retrieval index"));
            }
            let report = NativePreparationReport {
                tensor_count: generator
                    .checkpoint()
                    .expect("loaded generator owns a checkpoint")
                    .tensor_count(),
                feature_dimension: spec.feature_dimension(),
                sample_rate: spec.sample_rate.hz(),
                speaker_count: spec.speaker_count,
                uses_f0: spec.uses_f0,
                index_vectors: retrieval.as_ref().map(RetrievalIndex::len),
            };
            Ok((
                PreparedNativeModel {
                    generator,
                    content_encoder,
                    retrieval,
                },
                report,
            ))
        })();

        match result {
            Ok((prepared, report)) => {
                self.prepared = Some(prepared);
                self.state = EngineState::Ready;
                Ok(report)
            }
            Err(error) => {
                self.prepared = None;
                self.state = EngineState::Failed(error.to_string());
                Err(error)
            }
        }
    }

    /// Checks a selected device by performing a Candle tensor round trip.
    pub fn doctor(&self) -> Result<Vec<f32>, EngineError> {
        let device = resolve_device(self.config.device)?;
        backend_smoke_test(&device).map_err(EngineError::Inference)
    }

    /// Validates all inputs needed for an offline conversion.
    pub fn validate_offline(&self, job: &OfflineJob) -> Result<(), EngineError> {
        self.config.validate()?;
        let model = self.model.as_ref().ok_or(EngineError::NoModel)?;
        model.validate()?;
        job.validate()?;
        Ok(())
    }

    /// Validates and snapshots an offline job for a worker thread.
    pub fn begin_offline(&mut self, job: &OfflineJob) -> Result<OfflineTask, EngineError> {
        self.validate_offline(job)?;
        self.state = EngineState::Running;
        Ok(OfflineTask {
            config: self.config.clone(),
            model: self.model.clone().ok_or(EngineError::NoModel)?,
            job: job.clone(),
        })
    }

    /// Applies a worker result to the shared lifecycle state.
    pub fn finish_offline(&mut self, result: &Result<OfflineReport, String>) {
        self.state = match result {
            Ok(_) => EngineState::Ready,
            Err(error) => EngineState::Failed(error.clone()),
        };
    }

    /// Runs an offline job synchronously. CLI callers normally use this helper.
    pub fn start_offline(&mut self, job: &OfflineJob) -> Result<OfflineReport, EngineError> {
        let task = self.begin_offline(job)?;
        let result = task.run();
        self.state = match &result {
            Ok(_) => EngineState::Ready,
            Err(error) => EngineState::Failed(error.to_string()),
        };
        result
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
    /// The output directory does not exist.
    #[error("output directory does not exist: {path}", path = path.display())]
    MissingDirectory {
        /// Missing parent directory.
        path: PathBuf,
    },
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
    /// Audio decoding or WAV encoding failed.
    #[error(transparent)]
    Audio(#[from] rvc_rs_audio::AudioError),
    /// A mandatory managed runtime model could not be resolved or verified.
    #[error(transparent)]
    Asset(#[from] AssetError),
    /// ContentVec loading or inference failed.
    #[error(transparent)]
    Content(#[from] rvc_rs_candle::ContentError),
    /// A raw Candle operation failed while assembling pipeline tensors.
    #[error("tensor pipeline failed: {0}")]
    Tensor(#[from] candle_core::Error),
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
