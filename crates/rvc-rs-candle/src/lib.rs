#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Candle backend and the integration boundary between `pthrs` tensors and RVC.

use candle_core::{Device, Tensor};
use rvc_rs_core::{ComputeDevice, GeneratorInput, InputError, ModelSpec, VoiceGenerator};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Re-exported checkpoint crate used by the weight adapter implementation.
pub use pthrs as checkpoint;

/// A validated `.pth` checkpoint whose state dictionary is resident on a
/// Candle device.
///
/// Loading is intentionally eager: model startup may allocate, while the
/// eventual real-time inference loop must not perform checkpoint I/O or tensor
/// conversion.
#[derive(Debug)]
pub struct NativeCheckpoint {
    path: PathBuf,
    spec: ModelSpec,
    info: pthrs::VoiceModelInfo,
    weights: BTreeMap<String, Tensor>,
}

impl NativeCheckpoint {
    /// Opens, validates, decodes, and transfers every generator tensor once.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        let mut archive = pthrs::PthArchive::open(path)?;
        let info = archive.checkpoint().voice_model_info()?;
        let validation = info.validate(archive.checkpoint());
        if !validation.errors.is_empty() {
            return Err(InferenceError::InvalidCheckpoint(
                validation.errors.join("; "),
            ));
        }
        let spec = model_spec_from_info(&info)?;
        let names: Vec<String> = archive
            .checkpoint()
            .tensor_names()
            .map(str::to_owned)
            .collect();
        let mut weights = BTreeMap::new();
        for name in names {
            let decoded = archive.read_tensor_f32(&name)?;
            let shape = decoded
                .meta
                .shape
                .iter()
                .map(|&dimension| {
                    usize::try_from(dimension).map_err(|_| InferenceError::TensorDimension {
                        name: name.clone(),
                        dimension,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let tensor = Tensor::from_vec(decoded.values, shape.as_slice(), device)?;
            weights.insert(name, tensor);
        }
        Ok(Self {
            path: path.to_owned(),
            spec,
            info,
            weights,
        })
    }

    /// Source checkpoint path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Model facts derived from checkpoint metadata and tensor shapes.
    pub const fn spec(&self) -> ModelSpec {
        self.spec
    }

    /// Full `pthrs` model information retained for architecture construction.
    pub const fn info(&self) -> &pthrs::VoiceModelInfo {
        &self.info
    }

    /// Number of tensors resident on the selected Candle device.
    pub fn tensor_count(&self) -> usize {
        self.weights.len()
    }

    /// Returns one named checkpoint tensor without copying it.
    pub fn weight(&self, name: &str) -> Result<&Tensor, InferenceError> {
        self.weights
            .get(name)
            .ok_or_else(|| InferenceError::MissingWeight(name.to_owned()))
    }

    /// Iterates every loaded state-dictionary name in stable order.
    pub fn weight_names(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(String::as_str)
    }
}

/// In-memory FAISS IVF-Flat retrieval state for the real-time worker.
///
/// The index and search workspace are allocated at startup. Frame blending
/// reuses one scratch vector and performs no per-frame heap allocation.
#[derive(Debug)]
pub struct RetrievalIndex {
    index: pthrs::LoadedIvfFlatIndex,
    workspace: pthrs::SearchWorkspace,
    output: Vec<f32>,
    max_neighbors: usize,
}

impl RetrievalIndex {
    /// Loads an RVC `.index` and verifies its feature dimension.
    pub fn load(
        path: impl AsRef<Path>,
        feature_dimension: usize,
        max_neighbors: usize,
    ) -> Result<Self, InferenceError> {
        if max_neighbors == 0 {
            return Err(InferenceError::InvalidRetrieval(
                "max_neighbors must be greater than zero",
            ));
        }
        let index = pthrs::FaissIvfFlatIndex::open(path)?.load()?;
        if index.dimension() != feature_dimension {
            return Err(InferenceError::RetrievalDimension {
                expected: feature_dimension,
                actual: index.dimension(),
            });
        }
        let workspace = index.workspace(max_neighbors);
        Ok(Self {
            index,
            workspace,
            output: vec![0.0; feature_dimension],
            max_neighbors,
        })
    }

    /// Feature width stored by the retrieval index.
    pub fn dimension(&self) -> usize {
        self.index.dimension()
    }

    /// Number of searchable vectors.
    pub fn len(&self) -> u64 {
        self.index.len()
    }

    /// Returns whether the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Replaces each row-major feature frame with its retrieval blend.
    pub fn blend_frames(
        &mut self,
        features: &mut [f32],
        rate: f32,
        k: usize,
        nprobe: usize,
    ) -> Result<usize, InferenceError> {
        let dimension = self.dimension();
        if features.is_empty() || !features.len().is_multiple_of(dimension) {
            return Err(InferenceError::FeatureBuffer {
                values: features.len(),
                dimension,
            });
        }
        if k == 0 || nprobe == 0 {
            return Err(InferenceError::InvalidRetrieval(
                "k and nprobe must be greater than zero",
            ));
        }
        if k > self.max_neighbors {
            return Err(InferenceError::InvalidRetrieval(
                "k exceeds the preallocated neighbor capacity",
            ));
        }
        let mut retrieved_neighbors = 0;
        for frame in features.chunks_exact_mut(dimension) {
            retrieved_neighbors += self.index.search_and_blend(
                frame,
                &mut self.output,
                pthrs::SearchOptions { k, nprobe },
                rate,
                &mut self.workspace,
            )?;
            frame.copy_from_slice(&self.output);
        }
        Ok(retrieved_neighbors)
    }
}

fn model_spec_from_info(info: &pthrs::VoiceModelInfo) -> Result<ModelSpec, InferenceError> {
    use rvc_rs_core::{ModelVersion, SampleRate};

    let version = match (
        info.architecture_version.as_deref(),
        info.phone_feature_channels,
    ) {
        (Some("v1"), _) | (_, Some(256)) => ModelVersion::V1,
        (Some("v2"), _) | (_, Some(768)) => ModelVersion::V2,
        (label, channels) => {
            return Err(InferenceError::UnsupportedArchitecture {
                version: label.map(str::to_owned),
                phone_channels: channels,
            })
        }
    };
    let sample_rate = match info.config.sample_rate {
        32_000 => SampleRate::Hz32000,
        40_000 => SampleRate::Hz40000,
        48_000 => SampleRate::Hz48000,
        sample_rate => return Err(InferenceError::UnsupportedSampleRate(sample_rate)),
    };
    let speaker_count = usize::try_from(info.config.speaker_count)
        .map_err(|_| InferenceError::InvalidCheckpoint("speaker count is too large".into()))?;
    let spec = ModelSpec {
        version,
        sample_rate,
        uses_f0: info.pitch_guidance,
        speaker_count,
    };
    spec.validate()?;
    Ok(spec)
}

/// Resolves a requested compute target using compiled Candle backends.
pub fn resolve_device(request: ComputeDevice) -> Result<Device, InferenceError> {
    match request {
        ComputeDevice::Cpu => Ok(Device::Cpu),
        ComputeDevice::Cuda(index) => resolve_cuda(index),
        ComputeDevice::Metal(index) => resolve_metal(index),
        ComputeDevice::Auto => {
            #[cfg(feature = "cuda")]
            if let Ok(device) = Device::new_cuda(0) {
                return Ok(device);
            }

            #[cfg(feature = "metal")]
            if let Ok(device) = Device::new_metal(0) {
                return Ok(device);
            }

            Ok(Device::Cpu)
        }
    }
}

#[cfg(feature = "cuda")]
fn resolve_cuda(index: usize) -> Result<Device, InferenceError> {
    Device::new_cuda(index).map_err(InferenceError::Candle)
}

#[cfg(not(feature = "cuda"))]
fn resolve_cuda(_index: usize) -> Result<Device, InferenceError> {
    Err(InferenceError::BackendNotCompiled("cuda"))
}

#[cfg(feature = "metal")]
fn resolve_metal(index: usize) -> Result<Device, InferenceError> {
    Device::new_metal(index).map_err(InferenceError::Candle)
}

#[cfg(not(feature = "metal"))]
fn resolve_metal(_index: usize) -> Result<Device, InferenceError> {
    Err(InferenceError::BackendNotCompiled("metal"))
}

/// Performs a minimal allocation and device round trip.
///
/// This verifies that the selected Candle backend is usable. It is not a model
/// inference test.
pub fn backend_smoke_test(device: &Device) -> Result<Vec<f32>, InferenceError> {
    let tensor = Tensor::from_slice(&[0.25_f32, -0.5, 1.0, 2.0], 4, device)?;
    tensor.to_vec1().map_err(InferenceError::Candle)
}

/// Converts caller-owned `f32` data to a Candle tensor after exact shape checks.
///
/// The final checkpoint adapter should call this only after `pthrs` has decoded
/// the named tensor and verified its original dtype and shape.
pub fn tensor_from_f32(
    values: &[f32],
    shape: &[usize],
    device: &Device,
) -> Result<Tensor, InferenceError> {
    let expected = shape.iter().try_fold(1_usize, |elements, dimension| {
        elements.checked_mul(*dimension)
    });
    if expected != Some(values.len()) {
        return Err(InferenceError::TensorShape {
            shape: shape.to_vec(),
            values: values.len(),
        });
    }
    Tensor::from_vec(values.to_vec(), shape, device).map_err(InferenceError::Candle)
}

/// Placeholder generator reserving the public shape of the Candle backend.
///
/// It returns an explicit error until the checkpoint adapter and model forward
/// pass are numerically verified.
#[derive(Debug)]
pub struct CandleGenerator {
    spec: ModelSpec,
    device: Device,
    checkpoint: Option<NativeCheckpoint>,
}

impl CandleGenerator {
    /// Creates an uninitialized generator shell after validating its model spec.
    pub fn uninitialized(spec: ModelSpec, request: ComputeDevice) -> Result<Self, InferenceError> {
        spec.validate()?;
        Ok(Self {
            spec,
            device: resolve_device(request)?,
            checkpoint: None,
        })
    }

    /// Loads a real RVC `.pth` checkpoint onto the selected Candle device.
    pub fn load(path: impl AsRef<Path>, request: ComputeDevice) -> Result<Self, InferenceError> {
        let device = resolve_device(request)?;
        let checkpoint = NativeCheckpoint::load(path, &device)?;
        Ok(Self {
            spec: checkpoint.spec(),
            device,
            checkpoint: Some(checkpoint),
        })
    }

    /// Returns the selected Candle device.
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the loaded checkpoint, or `None` for an uninitialized test shell.
    pub const fn checkpoint(&self) -> Option<&NativeCheckpoint> {
        self.checkpoint.as_ref()
    }
}

impl VoiceGenerator for CandleGenerator {
    type Error = InferenceError;

    fn spec(&self) -> ModelSpec {
        self.spec
    }

    fn synthesize(&mut self, input: &GeneratorInput<'_>) -> Result<Vec<f32>, Self::Error> {
        input.validate(self.spec)?;
        Err(InferenceError::ModelNotImplemented)
    }
}

/// Failures produced by the Candle inference layer.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// Input failed backend-independent validation.
    #[error("invalid generator input: {0}")]
    Input(#[from] InputError),
    /// Candle returned an allocation, device, shape, or operation error.
    #[error("Candle backend error: {0}")]
    Candle(#[from] candle_core::Error),
    /// `pthrs` rejected a checkpoint or retrieval index.
    #[error("native model data error: {0}")]
    Checkpoint(#[from] pthrs::Error),
    /// The requested optional backend was not enabled at compile time.
    #[error("{0} support is not compiled; enable the matching Cargo feature")]
    BackendNotCompiled(&'static str),
    /// A decoded tensor cannot be represented by the supplied shape.
    #[error("tensor shape {shape:?} does not contain {values} values")]
    TensorShape {
        /// Caller-supplied dimensions.
        shape: Vec<usize>,
        /// Number of decoded values.
        values: usize,
    },
    /// A checkpoint dimension cannot be represented on this platform.
    #[error("tensor {name} dimension {dimension} does not fit usize")]
    TensorDimension {
        /// Checkpoint tensor name.
        name: String,
        /// Rejected dimension.
        dimension: u64,
    },
    /// Checkpoint metadata or required tensors are inconsistent.
    #[error("invalid RVC checkpoint: {0}")]
    InvalidCheckpoint(String),
    /// Neither the version label nor the phone width identifies v1/v2.
    #[error(
        "unsupported RVC architecture: version={version:?}, phone_channels={phone_channels:?}"
    )]
    UnsupportedArchitecture {
        /// Optional checkpoint version label.
        version: Option<String>,
        /// Optional phone feature width.
        phone_channels: Option<u32>,
    },
    /// Generator output rate is outside the currently mapped RVC variants.
    #[error("unsupported RVC sample rate: {0}")]
    UnsupportedSampleRate(u32),
    /// A required state-dictionary tensor was not loaded.
    #[error("checkpoint weight is missing: {0}")]
    MissingWeight(String),
    /// Retrieval features are not a complete row-major matrix.
    #[error("feature buffer has {values} values, not rows of {dimension}")]
    FeatureBuffer {
        /// Total feature values.
        values: usize,
        /// Required feature width.
        dimension: usize,
    },
    /// Retrieval search settings are invalid.
    #[error("invalid retrieval settings: {0}")]
    InvalidRetrieval(&'static str),
    /// Retrieval index width does not match the checkpoint feature width.
    #[error("retrieval dimension mismatch: expected {expected}, got {actual}")]
    RetrievalDimension {
        /// Width required by the voice checkpoint.
        expected: usize,
        /// Width stored in the FAISS index.
        actual: usize,
    },
    /// The RVC architecture has not been implemented in this scaffold yet.
    #[error("RVC generator forward pass is not implemented; see docs/ROADMAP.md")]
    ModelNotImplemented,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_info(version: &str, phone_channels: u32) -> pthrs::VoiceModelInfo {
        pthrs::VoiceModelInfo {
            config: pthrs::VoiceModelConfig {
                spectrogram_channels: 1025,
                segment_size: 32,
                intermediate_channels: 192,
                hidden_channels: 192,
                filter_channels: 768,
                attention_heads: 2,
                attention_layers: 6,
                kernel_size: 3,
                dropout: 0.0,
                resblock: "1".into(),
                resblock_kernel_sizes: vec![3, 7, 11],
                resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
                upsample_rates: vec![10, 10, 2, 2],
                upsample_initial_channels: 512,
                upsample_kernel_sizes: vec![16, 16, 4, 4],
                speaker_count: 1,
                speaker_embedding_channels: 109,
                sample_rate: 40_000,
            },
            architecture_version: Some(version.into()),
            sample_rate_label: Some("40k".into()),
            pitch_guidance: true,
            training_info: None,
            phone_feature_channels: Some(phone_channels),
        }
    }

    #[test]
    fn cpu_smoke_test_round_trips_values() {
        let values = backend_smoke_test(&Device::Cpu).expect("CPU backend should work");
        assert_eq!(values, vec![0.25, -0.5, 1.0, 2.0]);
    }

    #[test]
    fn rejects_incompatible_tensor_shape() {
        let error = tensor_from_f32(&[1.0, 2.0, 3.0], &[2, 2], &Device::Cpu).unwrap_err();
        assert!(matches!(error, InferenceError::TensorShape { .. }));
    }

    #[test]
    fn derives_v2_f0_spec_from_checkpoint_metadata() {
        let spec = model_spec_from_info(&model_info("v2", 768)).unwrap();
        assert_eq!(spec.version, rvc_rs_core::ModelVersion::V2);
        assert_eq!(spec.sample_rate, rvc_rs_core::SampleRate::Hz40000);
        assert!(spec.uses_f0);
        assert_eq!(spec.speaker_count, 1);
    }

    #[test]
    fn rejects_unknown_generator_sample_rate() {
        let mut info = model_info("v2", 768);
        info.config.sample_rate = 44_100;
        assert!(matches!(
            model_spec_from_info(&info),
            Err(InferenceError::UnsupportedSampleRate(44_100))
        ));
    }
}
