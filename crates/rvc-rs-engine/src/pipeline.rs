//! Offline conversion pipeline following RVC WebUI `infer/modules/vc/pipeline.py`.
//!
//! The whole input is high-passed, padded, and pitch-tracked once. Long inputs
//! are then split at quiet points and each padded segment is passed through
//! ContentVec, retrieval, protection, and the generator independently.

use crate::{
    extract_pitch_yin, pitch_to_coarse, protect_unvoiced_features, smooth_voiced_pitch,
    upsample_features_nearest_2x, EngineConfig, EngineError, F0Method,
};
use candle_core::IndexOp;
use rvc_rs_candle::{CandleGenerator, ContentEncoder, RetrievalIndex, Rmvpe};
use rvc_rs_core::{FeatureMatrix, GeneratorInput, PitchTrack, VoiceGenerator};

pub(crate) const RATE: usize = 16_000;
pub(crate) const WINDOW: usize = 160;
// RVC's full-precision settings: 1 s context padding, a split point searched
// within ±6 s of every 38 s, and no split for inputs up to 41 s.
const PAD: usize = RATE;
const QUERY: usize = 6 * RATE;
const CENTER: usize = 38 * RATE;
const MAX: usize = 41 * RATE;
const RMVPE_THRESHOLD: f32 = 0.03;

/// Resident models used by one offline conversion.
pub(crate) struct Converter<'a> {
    pub generator: &'a mut CandleGenerator,
    pub encoder: &'a ContentEncoder,
    pub rmvpe: Option<&'a Rmvpe>,
    pub index: Option<&'a mut RetrievalIndex>,
    pub config: &'a EngineConfig,
}

impl Converter<'_> {
    /// Converts high-passed 16 kHz audio into generator-rate audio.
    pub(crate) fn convert(&mut self, audio: &[f32]) -> Result<Vec<f32>, EngineError> {
        let target_rate = self.generator.spec().sample_rate.hz() as usize;
        let pad_target = target_rate * PAD / RATE;
        let splits = split_points(audio);
        let padded = reflect_pad(audio, PAD);
        let frames = padded.len() / WINDOW;
        let mut pitch = self.pitch(&padded)?;
        pitch.resize(frames, 0.0);

        let mut output = Vec::with_capacity(audio.len() * target_rate / RATE);
        let mut start = 0;
        for split in splits {
            let split = split / WINDOW * WINDOW;
            let end = (split + 2 * PAD + WINDOW).min(padded.len());
            let pitch_end = ((split + 2 * PAD) / WINDOW).min(frames);
            let segment = self.segment(&padded[start..end], &pitch[start / WINDOW..pitch_end])?;
            append_trimmed(&mut output, &segment, pad_target);
            start = split;
        }
        let segment = self.segment(&padded[start..], &pitch[start / WINDOW..])?;
        append_trimmed(&mut output, &segment, pad_target);
        Ok(output)
    }

    fn pitch(&self, padded: &[f32]) -> Result<Vec<f32>, EngineError> {
        match (self.config.f0_method, self.rmvpe) {
            (F0Method::Rmvpe, Some(rmvpe)) => {
                let shift = 2_f32.powf(f32::from(self.config.pitch_shift) / 12.0);
                let mut pitch = rmvpe.extract_f0(padded, RMVPE_THRESHOLD)?;
                pitch.iter_mut().for_each(|hz| *hz *= shift);
                Ok(pitch)
            }
            (F0Method::Rmvpe, None) => Err(EngineError::FeatureUnavailable(
                "RMVPE was selected but not loaded",
            )),
            (F0Method::Yin, _) => {
                let mut pitch = extract_pitch_yin(padded, self.config);
                smooth_voiced_pitch(&mut pitch, self.config.f0_filter_radius);
                Ok(pitch)
            }
        }
    }

    fn segment(&mut self, audio: &[f32], pitch: &[f32]) -> Result<Vec<f32>, EngineError> {
        let encoded = self.encoder.encode(audio)?;
        let (_, content_frames, dimensions) = encoded.dims3()?;
        let mut features = encoded.i(0)?.flatten_all()?.to_vec1::<f32>()?;
        let protect = self.config.protect_rate < 0.5;
        let original = protect.then(|| features.clone());

        if self.config.retrieval_rate > 0.0 {
            let index = self
                .index
                .as_deref_mut()
                .ok_or(EngineError::MissingPath("retrieval index"))?;
            index.blend_frames(
                &mut features,
                self.config.retrieval_rate,
                self.config.retrieval_neighbors,
                self.config.retrieval_nprobe,
            )?;
        }

        let mut features = upsample_features_nearest_2x(&features, content_frames, dimensions);
        let frames = (audio.len() / WINDOW)
            .min(content_frames * 2)
            .min(pitch.len());
        if frames == 0 {
            return Ok(Vec::new());
        }
        features.truncate(frames * dimensions);
        let continuous = &pitch[..frames];
        if let Some(original) = original {
            let original = upsample_features_nearest_2x(&original, content_frames, dimensions);
            protect_unvoiced_features(
                &mut features,
                &original,
                continuous,
                dimensions,
                self.config.protect_rate,
            );
        }
        let coarse: Vec<i64> = continuous.iter().copied().map(pitch_to_coarse).collect();
        Ok(self.generator.synthesize(&GeneratorInput {
            features: FeatureMatrix {
                values: &features,
                frames,
                dimensions,
            },
            pitch: Some(PitchTrack {
                coarse: &coarse,
                continuous_hz: continuous,
            }),
            speaker_id: self.config.speaker_id,
            noise_scale: self.config.noise_scale,
        })?)
    }
}

fn append_trimmed(output: &mut Vec<f32>, segment: &[f32], trim: usize) {
    if segment.len() > 2 * trim {
        output.extend_from_slice(&segment[trim..segment.len() - trim]);
    }
}

/// Picks split positions at the quietest 10 ms window near every 38 s mark.
pub(crate) fn split_points(audio: &[f32]) -> Vec<usize> {
    let padded = reflect_pad(audio, WINDOW / 2);
    if padded.len() <= MAX {
        return Vec::new();
    }
    let mut energy = vec![0.0_f64; audio.len()];
    let mut running: f64 = padded[..WINDOW].iter().map(|x| f64::from(x.abs())).sum();
    for (i, value) in energy.iter_mut().enumerate() {
        *value = running;
        running += f64::from(padded[i + WINDOW].abs()) - f64::from(padded[i].abs());
    }
    (CENTER..audio.len())
        .step_by(CENTER)
        .map(|center| {
            let from = center - QUERY;
            let to = (center + QUERY).min(audio.len());
            let mut best = from;
            for i in from..to {
                if energy[i] < energy[best] - 1e-9 {
                    best = i;
                }
            }
            best
        })
        .collect()
}

/// NumPy `mode="reflect"` padding (edge sample not repeated).
pub(crate) fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    if n < 2 {
        let value = samples.first().copied().unwrap_or(0.0);
        return vec![value; n + 2 * pad];
    }
    let period = 2 * (n as isize - 1);
    (-(pad as isize)..(n + pad) as isize)
        .map(|i| {
            let mut j = i.rem_euclid(period);
            if j >= n as isize {
                j = period - j;
            }
            samples[j as usize]
        })
        .collect()
}

/// Scales the input down when its peak exceeds 0.95, as RVC's audio loader does.
pub(crate) fn limit_input_peak(samples: &mut [f32]) {
    let peak = samples.iter().fold(0.0_f32, |peak, x| peak.max(x.abs())) / 0.95;
    if peak > 1.0 {
        samples.iter_mut().for_each(|x| *x /= peak);
    }
}

/// Scales the output down when its peak exceeds 0.99, as RVC does before PCM export.
pub(crate) fn limit_output_peak(samples: &mut [f32]) {
    let peak = samples.iter().fold(0.0_f32, |peak, x| peak.max(x.abs())) / 0.99;
    if peak > 1.0 {
        samples.iter_mut().for_each(|x| *x /= peak);
    }
}

// scipy.signal.butter(N=5, Wn=48, btype="high", fs=16000) and lfilter_zi.
const HIGHPASS_B: [f64; 6] = [
    0.969_960_645_183_844_7,
    -4.849_803_225_919_223,
    9.699_606_451_838_447,
    -9.699_606_451_838_447,
    4.849_803_225_919_223,
    -0.969_960_645_183_844_7,
];
const HIGHPASS_A: [f64; 6] = [
    1.0,
    -4.939_001_819_168_364,
    9.757_863_526_739_543,
    -9.639_544_849_413_458,
    4.761_506_797_356_209,
    -0.940_823_653_205_460_6,
];
const HIGHPASS_ZI: [f64; 5] = [
    -0.969_960_741_370_736_7,
    3.879_842_959_615_721,
    -5.819_764_430_801_29,
    3.879_842_948_235_015,
    -0.969_960_735_678_747_7,
];

/// Zero-phase 48 Hz high-pass equivalent to `scipy.signal.filtfilt(b, a, x)`.
pub(crate) fn highpass_filtfilt(samples: &[f32]) -> Vec<f32> {
    let pad = 3 * HIGHPASS_A.len();
    if samples.len() <= pad {
        return samples.to_vec();
    }
    let n = samples.len();
    let first = f64::from(samples[0]);
    let last = f64::from(samples[n - 1]);
    let mut extended = Vec::with_capacity(n + 2 * pad);
    extended.extend((1..=pad).rev().map(|i| 2.0 * first - f64::from(samples[i])));
    extended.extend(samples.iter().map(|&x| f64::from(x)));
    extended.extend((1..=pad).map(|i| 2.0 * last - f64::from(samples[n - 1 - i])));

    let forward = lfilter(&extended, extended[0]);
    let mut reversed: Vec<f64> = forward.into_iter().rev().collect();
    let initial = reversed[0];
    reversed = lfilter(&reversed, initial);
    reversed.reverse();
    reversed[pad..pad + n].iter().map(|&x| x as f32).collect()
}

fn lfilter(input: &[f64], initial: f64) -> Vec<f64> {
    let mut state: [f64; 5] = HIGHPASS_ZI.map(|z| z * initial);
    input
        .iter()
        .map(|&x| {
            let y = HIGHPASS_B[0] * x + state[0];
            for k in 0..4 {
                state[k] = HIGHPASS_B[k + 1] * x + state[k + 1] - HIGHPASS_A[k + 1] * y;
            }
            state[4] = HIGHPASS_B[5] * x - HIGHPASS_A[5] * y;
            y
        })
        .collect()
}

/// RVC `change_rms`: blends the source loudness envelope into the output.
///
/// `rate` is the output's own share; `1.0` leaves the output unchanged.
pub(crate) fn change_rms(
    source: &[f32],
    source_rate: usize,
    output: &mut [f32],
    output_rate: usize,
    rate: f32,
) {
    if output.is_empty() || source.is_empty() || rate >= 1.0 {
        return;
    }
    let source_rms = interpolate_linear(
        &frame_rms(source, source_rate / 2 * 2, source_rate / 2),
        output.len(),
    );
    let output_rms = interpolate_linear(
        &frame_rms(output, output_rate / 2 * 2, output_rate / 2),
        output.len(),
    );
    for ((sample, &from), &to) in output.iter_mut().zip(&source_rms).zip(&output_rms) {
        let to = to.max(1e-6);
        *sample *= from.powf(1.0 - rate) * to.powf(rate - 1.0);
    }
}

/// `librosa.feature.rms` with `center=True` and zero padding.
fn frame_rms(samples: &[f32], frame: usize, hop: usize) -> Vec<f32> {
    let half = frame / 2;
    let count = 1 + (samples.len() + 2 * half - frame) / hop;
    let mut prefix = Vec::with_capacity(samples.len() + 1);
    prefix.push(0.0_f64);
    for &x in samples {
        prefix.push(prefix.last().copied().unwrap_or(0.0) + f64::from(x) * f64::from(x));
    }
    (0..count)
        .map(|i| {
            let start = (i * hop).saturating_sub(half).min(samples.len());
            let end = (i * hop + frame).saturating_sub(half).min(samples.len());
            ((prefix[end] - prefix[start]) / frame as f64).sqrt() as f32
        })
        .collect()
}

/// `torch.nn.functional.interpolate(mode="linear", align_corners=False)`.
fn interpolate_linear(values: &[f32], length: usize) -> Vec<f32> {
    let scale = values.len() as f64 / length as f64;
    (0..length)
        .map(|i| {
            let position = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
            let left = (position.floor() as usize).min(values.len() - 1);
            let right = (left + 1).min(values.len() - 1);
            let fraction = (position - left as f64) as f32;
            values[left] + (values[right] - values[left]) * fraction
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_matches_numpy() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2),
            [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }

    #[test]
    fn highpass_removes_dc_and_keeps_voice_band() {
        let samples: Vec<f32> = (0..16_000)
            .map(|i| 0.5 + (std::f32::consts::TAU * 440.0 * i as f32 / 16_000.0).sin() * 0.25)
            .collect();
        let filtered = highpass_filtfilt(&samples);
        let middle = &filtered[4_000..12_000];
        let mean = middle.iter().sum::<f32>() / middle.len() as f32;
        let peak = middle.iter().fold(0.0_f32, |p, x| p.max(x.abs()));
        assert!(mean.abs() < 1e-3, "mean {mean}");
        assert!((peak - 0.25).abs() < 0.01, "peak {peak}");
    }

    #[test]
    fn short_inputs_are_not_split() {
        assert!(split_points(&vec![0.1; MAX - WINDOW]).is_empty());
    }

    #[test]
    fn long_inputs_split_at_the_quietest_point() {
        let mut audio = vec![0.5_f32; 80 * RATE];
        let quiet = CENTER + 2 * RATE;
        audio[quiet..quiet + 2 * WINDOW].fill(0.0);
        let splits = split_points(&audio);
        assert_eq!(splits.len(), 2);
        assert!(splits[0] >= quiet && splits[0] <= quiet + WINDOW);
    }

    #[test]
    fn change_rms_follows_source_level_at_zero_rate() {
        let source = vec![0.1_f32; 32_000];
        let mut output = vec![0.4_f32; 80_000];
        change_rms(&source, 16_000, &mut output, 40_000, 0.0);
        let middle = output[20_000..60_000].iter().sum::<f32>() / 40_000.0;
        assert!((middle - 0.1).abs() < 0.005, "{middle}");
    }

    #[test]
    fn linear_interpolation_matches_torch() {
        assert_eq!(interpolate_linear(&[0.0, 1.0], 4), [0.0, 0.25, 0.75, 1.0]);
    }
}
