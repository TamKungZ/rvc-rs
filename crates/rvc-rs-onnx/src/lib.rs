#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Production ONNX Runtime backend for complete RVC file conversion.
//!
//! This backend consumes three exported ONNX graphs: the target RVC generator,
//! ContentVec, and RMVPE. It has no Python, PyTorch, or libtorch runtime
//! dependency. The direct `.pth` Candle backend remains separate while it is
//! brought to numerical parity.

use std::path::{Path, PathBuf};
use std::time::Duration;

use thiserror::Error;
use vc_core::dsp;
use vc_core::model_rvc::{
    ChunkConverter, ChunkOutputConfig, F0Config, GpuPriority, NoiseGateShaping,
    OutputDynamicsConfig, RvcPipeline, RvcPipelineConfig,
};
use vc_core::sola::SmoothingKind;
use vc_core::Provider;

/// Files required by the complete ONNX voice-conversion pipeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OnnxModelFiles {
    /// Target RVC generator exported to ONNX.
    pub generator: PathBuf,
    /// ContentVec/Hubert embedder ONNX graph.
    pub contentvec: PathBuf,
    /// RMVPE pitch extractor ONNX graph.
    pub rmvpe: PathBuf,
}

/// Inference target supported by the ONNX backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnnxDevice {
    /// Portable ONNX Runtime CPU execution.
    Cpu,
    /// ONNX Runtime CUDA execution on the selected zero-based GPU.
    Cuda(u32),
}

/// Settings applied to one finite audio conversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnnxConversionConfig {
    /// Inference device.
    pub device: OnnxDevice,
    /// Pitch transposition in semitones.
    pub pitch_shift: f32,
    /// Speaker embedding index.
    pub speaker_id: i64,
    /// Size of each finite processing block.
    pub chunk_ms: u32,
    /// SOLA boundary crossfade duration.
    pub crossfade_ms: u32,
}

/// Timing and size information returned by a completed conversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnnxConversionReport {
    /// Number of input/output samples.
    pub samples: usize,
    /// Input and output sample rate.
    pub sample_rate: u32,
    /// Number of chunks sent through the model.
    pub chunks: usize,
    /// Sum of model inference time reported for all chunks.
    pub inference_time: Duration,
}

/// Converts mono floating-point audio and preserves its duration and rate.
pub fn convert_mono(
    files: &OnnxModelFiles,
    config: OnnxConversionConfig,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(Vec<f32>, OnnxConversionReport), OnnxError> {
    validate_file(&files.generator, "generator")?;
    validate_file(&files.contentvec, "ContentVec")?;
    validate_file(&files.rmvpe, "RMVPE")?;
    validate_config(config, sample_rate)?;

    if samples.is_empty() {
        return Err(OnnxError::InvalidInput("input audio contains no samples"));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(OnnxError::InvalidInput(
            "input audio contains NaN or infinity",
        ));
    }

    let (provider, gpu_device_id) = match config.device {
        OnnxDevice::Cpu => (Provider::Cpu, 0),
        OnnxDevice::Cuda(index) => (Provider::Cuda, index),
    };
    let chunk_samples = dsp::chunk_samples_for_rate(sample_rate, config.chunk_ms);
    let output_extra_ms = config.crossfade_ms.saturating_add(22);
    let pipeline = RvcPipeline::load(RvcPipelineConfig {
        model: &files.generator,
        embedder: &files.contentvec,
        embedder_output: None,
        f0_model: &files.rmvpe,
        provider,
        gpu_priority: GpuPriority::High,
        gpu_device_id,
        sample_rate,
        chunk_samples,
        speaker_id: config.speaker_id,
        pitch_shift: config.pitch_shift,
        f0: F0Config {
            silence_threshold: 0.0,
            ..F0Config::default()
        },
        input_gain: 1.0,
        noise_gate_enabled: false,
        noise_gate_threshold: 0.01,
        noise_gate_shaping: NoiseGateShaping::default(),
        output_extra_ms,
        volume_excluded_ms: config.crossfade_ms,
        extra_convert_ms: 100,
        output_gain: 1.0,
        output_dynamics: OutputDynamicsConfig::default(),
        progress: None,
    })
    .map_err(|error| OnnxError::Pipeline(error.to_string()))?;

    let mut converter = ChunkConverter::new(
        pipeline,
        ChunkOutputConfig {
            kind: SmoothingKind::Sola,
            output_sample_rate: sample_rate,
            output_chunk_samples: chunk_samples,
            crossfade_ms: config.crossfade_ms,
            sola_search_ms: 12,
            tail_discard_ms: 10,
        },
    );

    let preroll = vec![0.0; chunk_samples];
    converter
        .prime(&preroll, sample_rate)
        .map_err(|error| OnnxError::Pipeline(error.to_string()))?;

    let mut output = Vec::with_capacity(samples.len());
    let mut padded = Vec::new();
    let mut converted = Vec::new();
    let mut final_tail = Vec::new();
    let mut inference_time = Duration::ZERO;
    let mut chunks = 0;

    for chunk in samples.chunks(chunk_samples) {
        let model_input = if chunk.len() == chunk_samples {
            chunk
        } else {
            padded.clear();
            padded.extend_from_slice(chunk);
            padded.resize(chunk_samples, 0.0);
            padded.as_slice()
        };
        let stats = converter
            .process_chunk(
                model_input,
                sample_rate,
                Some(&mut final_tail),
                &mut converted,
            )
            .map_err(|error| OnnxError::Pipeline(error.to_string()))?;
        inference_time += stats.inference_time;
        output.extend_from_slice(&converted);
        chunks += 1;
    }

    if output.len() < samples.len() {
        let missing = samples.len() - output.len();
        output.extend_from_slice(&final_tail[..missing.min(final_tail.len())]);
    }
    output.resize(samples.len(), 0.0);
    for sample in &mut output {
        *sample = sample.clamp(-1.0, 1.0);
    }

    Ok((
        output,
        OnnxConversionReport {
            samples: samples.len(),
            sample_rate,
            chunks,
            inference_time,
        },
    ))
}

fn validate_file(path: &Path, role: &'static str) -> Result<(), OnnxError> {
    if !path.is_file() {
        return Err(OnnxError::MissingModel {
            role,
            path: path.to_owned(),
        });
    }
    if !path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("onnx"))
    {
        return Err(OnnxError::WrongExtension {
            role,
            path: path.to_owned(),
        });
    }
    Ok(())
}

fn validate_config(config: OnnxConversionConfig, sample_rate: u32) -> Result<(), OnnxError> {
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(OnnxError::InvalidInput(
            "sample rate must be between 8 kHz and 192 kHz",
        ));
    }
    if !(20..=2_000).contains(&config.chunk_ms) {
        return Err(OnnxError::InvalidInput(
            "chunk duration must be between 20 and 2000 ms",
        ));
    }
    if config.crossfade_ms >= config.chunk_ms {
        return Err(OnnxError::InvalidInput(
            "crossfade duration must be shorter than the chunk",
        ));
    }
    if !config.pitch_shift.is_finite() || !(-24.0..=24.0).contains(&config.pitch_shift) {
        return Err(OnnxError::InvalidInput(
            "pitch shift must be finite and between -24 and 24 semitones",
        ));
    }
    Ok(())
}

/// ONNX model, input, or execution failure.
#[derive(Debug, Error)]
pub enum OnnxError {
    /// A required model file does not exist.
    #[error("{role} ONNX model does not exist: {path}", path = path.display())]
    MissingModel {
        /// Model role shown to the caller.
        role: &'static str,
        /// Missing path.
        path: PathBuf,
    },
    /// A model path is not an ONNX file.
    #[error("{role} model must use .onnx: {path}", path = path.display())]
    WrongExtension {
        /// Model role shown to the caller.
        role: &'static str,
        /// Rejected path.
        path: PathBuf,
    },
    /// Audio or settings are invalid.
    #[error("invalid ONNX conversion input: {0}")]
    InvalidInput(&'static str),
    /// Model loading or inference failed.
    #[error("ONNX RVC pipeline failed: {0}")]
    Pipeline(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_crossfade_equal_to_chunk() {
        let error = validate_config(
            OnnxConversionConfig {
                device: OnnxDevice::Cpu,
                pitch_shift: 0.0,
                speaker_id: 0,
                chunk_ms: 100,
                crossfade_ms: 100,
            },
            48_000,
        )
        .unwrap_err();
        assert!(matches!(error, OnnxError::InvalidInput(_)));
    }
}
