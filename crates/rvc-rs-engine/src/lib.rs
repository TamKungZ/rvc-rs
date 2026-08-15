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

/// Curated starting points for common conversion material.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum QualityPreset {
    /// General-purpose settings with a moderate retrieval blend.
    #[default]
    Balanced,
    /// Favors intelligibility and clean consonants over maximum target identity.
    CleanSpeech,
    /// Keeps a wider F0 range and lighter smoothing for melodic input.
    Singing,
    /// Favors target-speaker identity through stronger, broader retrieval.
    StrongIdentity,
}

impl QualityPreset {
    /// Stable command-line name.
    pub const fn cli_name(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::CleanSpeech => "clean",
            Self::Singing => "singing",
            Self::StrongIdentity => "identity",
        }
    }

    /// Parses a stable command-line name.
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "balanced" => Some(Self::Balanced),
            "clean" | "clean-speech" => Some(Self::CleanSpeech),
            "singing" | "song" => Some(Self::Singing),
            "identity" | "strong-identity" | "character" => Some(Self::StrongIdentity),
            _ => None,
        }
    }

    /// Applies only quality-related fields, preserving device, pitch, speaker,
    /// and future streaming geometry selected by the caller.
    pub fn apply(self, config: &mut EngineConfig) {
        let (
            retrieval_rate,
            retrieval_neighbors,
            retrieval_nprobe,
            protect_rate,
            noise_scale,
            f0_min_hz,
            f0_max_hz,
            f0_yin_threshold,
            f0_filter_radius,
            rms_mix_rate,
        ) = match self {
            Self::Balanced => {
                (0.60, 8, 2, 0.33, 0.50, 50.0, 1_100.0, 0.15, 3, 0.25)
            }
            Self::CleanSpeech => {
                (0.35, 8, 2, 0.15, 0.35, 65.0, 650.0, 0.14, 5, 0.10)
            }
            Self::Singing => {
                (0.65, 12, 4, 0.30, 0.50, 45.0, 1_400.0, 0.12, 2, 0.25)
            }
            Self::StrongIdentity => {
                (0.85, 16, 4, 0.40, 0.60, 50.0, 1_100.0, 0.16, 3, 0.40)
            }
        };
        config.retrieval_rate = retrieval_rate;
        config.retrieval_neighbors = retrieval_neighbors;
        config.retrieval_nprobe = retrieval_nprobe;
        config.protect_rate = protect_rate;
        config.noise_scale = noise_scale;
        config.f0_min_hz = f0_min_hz;
        config.f0_max_hz = f0_max_hz;
        config.f0_yin_threshold = f0_yin_threshold;
        config.f0_filter_radius = f0_filter_radius;
        config.rms_mix_rate = rms_mix_rate;
        config.output_gain_db = 0.0;
    }
}

/// User-adjustable inference settings shared by every front end.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct EngineConfig {
    /// Tensor execution target.
    pub device: ComputeDevice,
    /// Pitch transposition in semitones.
    pub pitch_shift: i8,
    /// Retrieval blend ratio in the inclusive range `0.0..=1.0`.
    pub retrieval_rate: f32,
    /// Number of nearest index vectors blended for each content frame.
    pub retrieval_neighbors: usize,
    /// Number of IVF lists probed for each retrieval query.
    pub retrieval_nprobe: usize,
    /// Retrieval contribution on unvoiced frames; `0.5` disables protection.
    pub protect_rate: f32,
    /// Standard deviation multiplier used when sampling the generator latent.
    pub noise_scale: f32,
    /// Lowest F0 considered by the built-in YIN extractor.
    pub f0_min_hz: f32,
    /// Highest F0 considered by the built-in YIN extractor.
    pub f0_max_hz: f32,
    /// YIN cumulative-mean normalized-difference acceptance threshold.
    pub f0_yin_threshold: f32,
    /// Radius of the voiced-frame median smoother; zero disables it.
    pub f0_filter_radius: usize,
    /// Output-envelope share: zero follows the input RMS, one keeps generated RMS.
    pub rms_mix_rate: f32,
    /// Final output gain in decibels.
    pub output_gain_db: f32,
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
            retrieval_neighbors: 8,
            retrieval_nprobe: 2,
            protect_rate: 0.33,
            noise_scale: 0.50,
            f0_min_hz: 50.0,
            f0_max_hz: 1_100.0,
            f0_yin_threshold: 0.15,
            f0_filter_radius: 3,
            rms_mix_rate: 0.25,
            output_gain_db: 0.0,
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
        if !(1..=32).contains(&self.retrieval_neighbors) {
            return Err(EngineError::InvalidConfig(
                "retrieval_neighbors must be between 1 and 32",
            ));
        }
        if !(1..=64).contains(&self.retrieval_nprobe) {
            return Err(EngineError::InvalidConfig(
                "retrieval_nprobe must be between 1 and 64",
            ));
        }
        if !(0.0..=0.5).contains(&self.protect_rate) {
            return Err(EngineError::InvalidConfig(
                "protect_rate must be between 0.0 and 0.5",
            ));
        }
        if !(0.0..=1.5).contains(&self.noise_scale) {
            return Err(EngineError::InvalidConfig(
                "noise_scale must be between 0.0 and 1.5",
            ));
        }
        if !(40.0..=300.0).contains(&self.f0_min_hz) {
            return Err(EngineError::InvalidConfig(
                "f0_min_hz must be between 40 and 300 Hz",
            ));
        }
        if !(300.0..=1_600.0).contains(&self.f0_max_hz)
            || self.f0_max_hz <= self.f0_min_hz
        {
            return Err(EngineError::InvalidConfig(
                "f0_max_hz must be between 300 and 1600 Hz and greater than f0_min_hz",
            ));
        }
        if !(0.05..=0.40).contains(&self.f0_yin_threshold) {
            return Err(EngineError::InvalidConfig(
                "f0_yin_threshold must be between 0.05 and 0.40",
            ));
        }
        if self.f0_filter_radius > 7 {
            return Err(EngineError::InvalidConfig(
                "f0_filter_radius must be between 0 and 7",
            ));
        }
        if !(0.0..=1.0).contains(&self.rms_mix_rate) {
            return Err(EngineError::InvalidConfig(
                "rms_mix_rate must be between 0.0 and 1.0",
            ));
        }
        if !(-24.0..=12.0).contains(&self.output_gain_db) {
            return Err(EngineError::InvalidConfig(
                "output_gain_db must be between -24 and 12 dB",
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
        let samples_16k = bandlimited_resample(&decoded.samples, decoded.sample_rate, 16_000);
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
        let protected_features = (self.config.retrieval_rate > 0.0
            && self.config.protect_rate < 0.5)
            .then(|| base_features.clone());

        if self.config.retrieval_rate > 0.0 {
            let index_path = self
                .model
                .index
                .as_ref()
                .ok_or(EngineError::MissingPath("retrieval index"))?;
            let mut index = RetrievalIndex::load(
                index_path,
                dimensions,
                self.config.retrieval_neighbors,
            )?;
            index.blend_frames(
                &mut base_features,
                self.config.retrieval_rate,
                self.config.retrieval_neighbors,
                self.config.retrieval_nprobe,
            )?;
        }

        let mut features =
            upsample_features_nearest_2x(&base_features, content_frames, dimensions);
        let protected_features = protected_features.map(|values| {
            upsample_features_nearest_2x(&values, content_frames, dimensions)
        });
        let mut pitch = extract_pitch_yin(&samples_16k, &self.config);
        smooth_voiced_pitch(&mut pitch, self.config.f0_filter_radius);
        let frames = (features.len() / dimensions).min(pitch.len());
        if frames == 0 {
            return Err(EngineError::FeatureUnavailable(
                "source audio is too short to produce inference frames",
            ));
        }
        features.truncate(frames * dimensions);
        let continuous: Vec<f32> = pitch.into_iter().take(frames).collect();
        if let Some(original) = protected_features.as_deref() {
            protect_unvoiced_features(
                &mut features,
                original,
                &continuous,
                dimensions,
                self.config.protect_rate,
            );
        }
        let coarse: Vec<i64> = continuous.iter().copied().map(pitch_to_coarse).collect();
        let mut output = generator.synthesize(&GeneratorInput {
            features: FeatureMatrix {
                values: &features,
                frames,
                dimensions,
            },
            pitch: Some(PitchTrack {
                coarse: &coarse,
                continuous_hz: &continuous,
            }),
            speaker_id: self.config.speaker_id,
            noise_scale: self.config.noise_scale,
        })?;
        let inference_time = inference_started.elapsed();
        let sample_rate = spec.sample_rate.hz();
        if self.config.rms_mix_rate < 1.0 {
            let source_at_output_rate = bandlimited_resample(
                &decoded.samples,
                decoded.sample_rate,
                sample_rate,
            );
            match_rms_envelope(
                &source_at_output_rate,
                &mut output,
                sample_rate,
                self.config.rms_mix_rate,
            );
        }
        apply_output_gain(&mut output, self.config.output_gain_db);
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

fn bandlimited_resample(
    samples: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Vec<f32> {
    if samples.is_empty() || source_rate == target_rate {
        return samples.to_vec();
    }
    let length =
        (samples.len() as u64 * u64::from(target_rate) / u64::from(source_rate)) as usize;
    let step = f64::from(source_rate) / f64::from(target_rate);
    let cutoff = (f64::from(target_rate) / f64::from(source_rate)).min(1.0) * 0.94;
    const RADIUS: isize = 16;

    (0..length)
        .map(|i| {
            let position = i as f64 * step;
            let center = position.floor() as isize;
            let mut weighted = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            for tap in center - RADIUS + 1..=center + RADIUS {
                if tap < 0 || tap >= samples.len() as isize {
                    continue;
                }
                let distance = position - tap as f64;
                let normalized = distance / RADIUS as f64;
                if normalized.abs() >= 1.0 {
                    continue;
                }
                // Windowed-sinc low-pass.  Linear interpolation aliases
                // everything above 8 kHz into HuBERT's 16 kHz input when
                // converting common 44.1/48 kHz WAVs; those aliases corrupt
                // both content features and F0.
                let sinc_argument = cutoff * distance;
                let sinc = if sinc_argument.abs() < 1e-12 {
                    1.0
                } else {
                    (std::f64::consts::PI * sinc_argument).sin()
                        / (std::f64::consts::PI * sinc_argument)
                };
                let window = 0.5 * (1.0 + (std::f64::consts::PI * normalized).cos());
                let weight = cutoff * sinc * window;
                weighted += f64::from(samples[tap as usize]) * weight;
                weight_sum += weight;
            }
            if weight_sum.abs() > 1e-12 {
                (weighted / weight_sum) as f32
            } else {
                samples[center.clamp(0, samples.len() as isize - 1) as usize]
            }
        })
        .collect()
}

fn upsample_features_nearest_2x(
    input: &[f32],
    frames: usize,
    dimensions: usize,
) -> Vec<f32> {
    if frames == 0 {
        return Vec::new();
    }
    let mut output = vec![0.0; frames * 2 * dimensions];
    for frame in 0..frames * 2 {
        let source = frame / 2;
        let input_start = source * dimensions;
        let output_start = frame * dimensions;
        output[output_start..output_start + dimensions]
            .copy_from_slice(&input[input_start..input_start + dimensions]);
    }
    output
}

fn extract_pitch_yin(samples: &[f32], config: &EngineConfig) -> Vec<f32> {
    const RATE: usize = 16_000;
    const HOP: usize = 160;
    const WINDOW: usize = 1_024;
    let min_lag = (RATE as f32 / config.f0_max_hz).floor().max(2.0) as usize;
    let max_lag = (RATE as f32 / config.f0_min_hz).ceil() as usize;
    let fallback_threshold = (config.f0_yin_threshold + 0.07).min(0.47);
    let frames = samples.len() / HOP;
    let shift = 2f32.powf(f32::from(config.pitch_shift) / 12.0);
    let mut output = Vec::with_capacity(frames);
    let mut windowed = vec![0.0_f32; WINDOW];
    let mut difference = vec![0.0_f32; max_lag + 1];
    let mut cmnd = vec![1.0_f32; max_lag + 1];

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
        if rms < 0.005 || slice.len() <= max_lag + 2 {
            output.push(0.0);
            continue;
        }

        let mean = slice.iter().copied().sum::<f32>() / slice.len() as f32;
        let denominator = (slice.len() - 1) as f32;
        for (index, (&sample, destination)) in
            slice.iter().zip(windowed.iter_mut()).enumerate()
        {
            let hann = 0.5
                - 0.5
                    * (2.0 * std::f32::consts::PI * index as f32 / denominator).cos();
            *destination = (sample - mean) * hann;
        }
        let values = &windowed[..slice.len()];

        difference.fill(0.0);
        let frame_max_lag = max_lag.min(values.len() / 2);
        for lag in 1..=frame_max_lag {
            let mut sum = 0.0_f32;
            for index in 0..values.len() - lag {
                let delta = values[index] - values[index + lag];
                sum += delta * delta;
            }
            difference[lag] = sum;
        }

        cmnd.fill(1.0);
        let mut cumulative = 0.0_f32;
        for lag in 1..=frame_max_lag {
            cumulative += difference[lag];
            cmnd[lag] = if cumulative > f32::EPSILON {
                difference[lag] * lag as f32 / cumulative
            } else {
                1.0
            };
        }

        // YIN selects the first sufficiently periodic local minimum rather
        // than the global autocorrelation maximum.  The latter strongly
        // preferred short, high-frequency lags in breath and silence and was
        // the source of 800-1100 Hz spikes in real conversion input.
        let mut selected = None;
        let mut lag = min_lag;
        while lag < frame_max_lag {
            if cmnd[lag] < config.f0_yin_threshold {
                while lag < frame_max_lag && cmnd[lag + 1] < cmnd[lag] {
                    lag += 1;
                }
                selected = Some(lag);
                break;
            }
            lag += 1;
        }

        if selected.is_none() {
            let fallback = (min_lag..=frame_max_lag)
                .min_by(|&left, &right| cmnd[left].total_cmp(&cmnd[right]));
            selected = fallback.filter(|&candidate| cmnd[candidate] <= fallback_threshold);
        }

        let Some(lag) = selected else {
            output.push(0.0);
            continue;
        };
        let refined = if lag > 1 && lag < frame_max_lag {
            let left = cmnd[lag - 1];
            let middle = cmnd[lag];
            let right = cmnd[lag + 1];
            let curvature = left - 2.0 * middle + right;
            if curvature.abs() > 1e-9 {
                lag as f32 + 0.5 * (left - right) / curvature
            } else {
                lag as f32
            }
        } else {
            lag as f32
        };
        output.push(RATE as f32 / refined * shift);
    }
    output
}

fn smooth_voiced_pitch(pitch: &mut [f32], radius: usize) {
    if radius == 0 || pitch.is_empty() {
        return;
    }
    let source = pitch.to_vec();
    let mut neighbors = Vec::with_capacity(radius * 2 + 1);
    for (frame, destination) in pitch.iter_mut().enumerate() {
        if source[frame] <= 0.0 {
            continue;
        }
        neighbors.clear();
        let start = frame.saturating_sub(radius);
        let end = (frame + radius + 1).min(source.len());
        neighbors.extend(source[start..end].iter().copied().filter(|&hz| hz > 0.0));
        neighbors.sort_unstable_by(f32::total_cmp);
        if let Some(&median) = neighbors.get(neighbors.len() / 2) {
            *destination = median;
        }
    }
}

fn protect_unvoiced_features(
    retrieved: &mut [f32],
    original: &[f32],
    pitch: &[f32],
    dimensions: usize,
    protect_rate: f32,
) {
    for (frame, &hz) in pitch.iter().enumerate() {
        if hz > 0.0 {
            continue;
        }
        let start = frame * dimensions;
        let end = start + dimensions;
        if end > retrieved.len() || end > original.len() {
            break;
        }
        for (retrieved_value, &original_value) in
            retrieved[start..end].iter_mut().zip(&original[start..end])
        {
            *retrieved_value =
                *retrieved_value * protect_rate + original_value * (1.0 - protect_rate);
        }
    }
}

fn match_rms_envelope(source: &[f32], output: &mut [f32], sample_rate: u32, mix: f32) {
    if source.is_empty() || output.is_empty() || mix >= 1.0 {
        return;
    }
    let window = (sample_rate as usize / 50).max(1);
    let hop = (sample_rate as usize / 100).max(1);
    let source_energy = energy_prefix(source);
    let output_energy = energy_prefix(output);
    let points = output.len().div_ceil(hop) + 1;
    let mut gains = Vec::with_capacity(points);
    for point in 0..points {
        let output_center = (point * hop).min(output.len().saturating_sub(1));
        let source_center = if output.len() <= 1 {
            0
        } else {
            output_center * source.len().saturating_sub(1) / output.len().saturating_sub(1)
        };
        let source_rms = local_rms(&source_energy, source_center, window, source.len());
        let output_rms = local_rms(&output_energy, output_center, window, output.len());
        let ratio = (source_rms + 1e-4) / (output_rms + 1e-4);
        gains.push(ratio.powf(1.0 - mix).clamp(0.05, 10.0));
    }
    for (index, sample) in output.iter_mut().enumerate() {
        let point = index / hop;
        let fraction = (index % hop) as f32 / hop as f32;
        let left = gains[point];
        let right = gains[(point + 1).min(gains.len() - 1)];
        *sample *= left + (right - left) * fraction;
    }
}

fn energy_prefix(samples: &[f32]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(samples.len() + 1);
    prefix.push(0.0);
    for &sample in samples {
        prefix.push(prefix.last().copied().unwrap_or(0.0) + f64::from(sample * sample));
    }
    prefix
}

fn local_rms(prefix: &[f64], center: usize, window: usize, length: usize) -> f32 {
    let start = center.saturating_sub(window / 2);
    let end = (center + window / 2 + 1).min(length);
    if end <= start {
        return 0.0;
    }
    (((prefix[end] - prefix[start]) / (end - start) as f64).max(0.0).sqrt()) as f32
}

fn apply_output_gain(output: &mut [f32], gain_db: f32) {
    if gain_db == 0.0 {
        return;
    }
    let gain = 10.0_f32.powf(gain_db / 20.0);
    for sample in output {
        *sample *= gain;
    }
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
                .map(|path| {
                    RetrievalIndex::load(
                        path,
                        spec.feature_dimension(),
                        self.config.retrieval_neighbors,
                    )
                })
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
        if self.config.retrieval_rate > 0.0 && model.index.is_none() {
            return Err(EngineError::MissingPath("retrieval index"));
        }
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

    #[test]
    fn yin_tracks_a_clean_sine_without_octave_spikes() {
        let samples: Vec<f32> = (0..32_000)
            .map(|index| {
                (2.0 * std::f32::consts::PI * 220.0 * index as f32 / 16_000.0).sin() * 0.2
            })
            .collect();
        let pitch = extract_pitch_yin(&samples, &EngineConfig::default());
        let voiced: Vec<f32> = pitch.into_iter().filter(|&hz| hz > 0.0).collect();
        assert!(!voiced.is_empty());
        assert!(voiced.iter().all(|&hz| (hz - 220.0).abs() < 3.0));
    }

    #[test]
    fn yin_rejects_silence() {
        let silence = vec![0.0; 16_000];
        let pitch = extract_pitch_yin(&silence, &EngineConfig::default());
        assert!(pitch.iter().all(|&hz| hz == 0.0));
    }

    #[test]
    fn presets_preserve_pitch_device_and_speaker() {
        let mut config = EngineConfig {
            device: ComputeDevice::Cpu,
            pitch_shift: -4,
            speaker_id: 7,
            ..EngineConfig::default()
        };
        QualityPreset::Singing.apply(&mut config);
        assert_eq!(config.device, ComputeDevice::Cpu);
        assert_eq!(config.pitch_shift, -4);
        assert_eq!(config.speaker_id, 7);
        assert_eq!(config.f0_max_hz, 1_400.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn unvoiced_protection_blends_back_original_features() {
        let mut retrieved = vec![10.0, 20.0, 30.0, 40.0];
        let original = [2.0, 4.0, 6.0, 8.0];
        protect_unvoiced_features(&mut retrieved, &original, &[0.0, 220.0], 2, 0.25);
        assert_eq!(retrieved, [4.0, 8.0, 30.0, 40.0]);
    }

    #[test]
    fn voiced_pitch_smoothing_preserves_unvoiced_frames() {
        let mut pitch = [100.0, 102.0, 900.0, 101.0, 0.0];
        smooth_voiced_pitch(&mut pitch, 2);
        assert_eq!(pitch[2], 102.0);
        assert_eq!(pitch[4], 0.0);
    }

    #[test]
    fn rms_matching_moves_output_toward_source_level() {
        let source = vec![0.1_f32; 4_000];
        let mut output = vec![0.4_f32; 4_000];
        match_rms_envelope(&source, &mut output, 40_000, 0.0);
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        assert!((mean - 0.1).abs() < 0.002);
    }

    #[test]
    fn bandlimited_resampler_preserves_duration_and_dc() {
        let source = vec![0.25_f32; 44_100];
        let output = bandlimited_resample(&source, 44_100, 16_000);
        assert_eq!(output.len(), 16_000);
        assert!(output.iter().all(|&sample| (sample - 0.25).abs() < 1e-5));
    }

    #[test]
    fn feature_upsampling_matches_torch_nearest() {
        let output = upsample_features_nearest_2x(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(output, [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }
}
