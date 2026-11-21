//! Native RMVPE pitch estimator.
//!
//! Follows the public RVC `infer/lib/rmvpe.py` implementation: a 128-band log
//! mel spectrogram at 16 kHz/hop 160 feeds a deep U-Net, a bidirectional GRU,
//! and a 360-bin pitch salience head decoded by local weighted averaging.

use crate::InferenceError;
use candle_core::{Device, Tensor};
use candle_nn::ops;
use std::collections::BTreeMap;
use std::path::Path;

/// RMVPE input sample rate.
pub const RMVPE_SAMPLE_RATE: u32 = 16_000;
/// Analysis hop in samples; one F0 value is produced per hop.
pub const RMVPE_HOP: usize = 160;

const N_FFT: usize = 1024;
const N_BINS: usize = N_FFT / 2 + 1;
const N_MELS: usize = 128;
const MEL_FMIN: f64 = 30.0;
const MEL_FMAX: f64 = 8_000.0;
const N_CLASSES: usize = 360;
const GRU_HIDDEN: usize = 256;
const BN_EPS: f64 = 1e-5;
const ENCODER_LAYERS: usize = 5;
const INTERMEDIATE_LAYERS: usize = 4;
const BLOCKS: usize = 4;
// Frames per network pass. Long inputs are split into overlapping windows so
// memory stays bounded; the margins give the convolutional stack and the GRU
// enough context that the stitched salience matches a single pass.
const CHUNK_FRAMES: usize = 2_048;
const CHUNK_MARGIN: usize = 128;

/// Resident RMVPE network and spectral analysis tables.
#[derive(Debug)]
pub struct Rmvpe {
    device: Device,
    dft: Tensor,
    mel_basis: Tensor,
    input_norm: BatchNorm,
    encoders: Vec<Vec<ConvBlockRes>>,
    intermediate: Vec<Vec<ConvBlockRes>>,
    decoders: Vec<DecoderBlock>,
    cnn_weight: Tensor,
    cnn_bias: Tensor,
    gru: [GruDirection; 2],
    fc_weight: Tensor,
    fc_bias: Tensor,
}

impl Rmvpe {
    /// Loads an upstream `rmvpe.pt` state dictionary through `pthrs`.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self, InferenceError> {
        let mut archive = pthrs::PthArchive::open(path.as_ref())?;
        let names: Vec<String> = archive
            .checkpoint()
            .tensor_names()
            .filter(|name| !name.ends_with("num_batches_tracked"))
            .map(str::to_owned)
            .collect();
        let mut tensors = BTreeMap::new();
        for name in names {
            let decoded = archive.read_tensor_f32(&name)?;
            let shape: Vec<usize> = decoded.meta.shape.iter().map(|&d| d as usize).collect();
            tensors.insert(name, (shape, decoded.values));
        }
        let w = RmvpeWeights {
            tensors: &tensors,
            device,
        };

        let mut encoders = Vec::with_capacity(ENCODER_LAYERS);
        for i in 0..ENCODER_LAYERS {
            encoders.push(w.res_blocks(&format!("unet.encoder.layers.{i}.conv"))?);
        }
        let mut intermediate = Vec::with_capacity(INTERMEDIATE_LAYERS);
        for i in 0..INTERMEDIATE_LAYERS {
            intermediate.push(w.res_blocks(&format!("unet.intermediate.layers.{i}.conv"))?);
        }
        let mut decoders = Vec::with_capacity(ENCODER_LAYERS);
        for i in 0..ENCODER_LAYERS {
            let p = format!("unet.decoder.layers.{i}");
            decoders.push(DecoderBlock {
                transpose: w.tensor(&format!("{p}.conv1.0.weight"))?,
                norm: w.batch_norm(&format!("{p}.conv1.1"))?,
                blocks: w.res_blocks(&format!("{p}.conv2"))?,
            });
        }
        let gru = [w.gru("")?, w.gru("_reverse")?];
        let fc_weight = w.tensor("fc.1.weight")?;
        if fc_weight.dims() != [N_CLASSES, 2 * GRU_HIDDEN] {
            return Err(InferenceError::UnsupportedModel(
                "RMVPE output head shape does not match 512 -> 360".into(),
            ));
        }

        Ok(Self {
            device: device.clone(),
            dft: dft_basis(device)?,
            mel_basis: mel_basis(device)?,
            input_norm: w.batch_norm("unet.encoder.bn")?,
            encoders,
            intermediate,
            decoders,
            cnn_weight: w.tensor("cnn.weight")?,
            cnn_bias: w.tensor("cnn.bias")?,
            gru,
            fc_weight,
            fc_bias: w.tensor("fc.1.bias")?,
        })
    }

    /// Estimates F0 in Hz for every 10 ms hop of 16 kHz mono audio.
    ///
    /// Returns `samples.len() / 160 + 1` frames; unvoiced frames are zero.
    pub fn extract_f0(
        &self,
        samples_16k: &[f32],
        threshold: f32,
    ) -> Result<Vec<f32>, InferenceError> {
        let mel = self.log_mel(samples_16k)?;
        let salience = self.salience(&mel)?;
        Ok(decode_salience(&salience, threshold))
    }

    /// Computes the `(128, frames)` log mel spectrogram used by RMVPE.
    pub fn log_mel(&self, samples_16k: &[f32]) -> Result<Tensor, InferenceError> {
        let frames = samples_16k.len() / RMVPE_HOP + 1;
        let padded = reflect_pad(samples_16k, N_FFT / 2);
        let mut framed = vec![0.0_f32; frames * N_FFT];
        for (frame, row) in framed.as_chunks_mut::<N_FFT>().0.iter_mut().enumerate() {
            let start = frame * RMVPE_HOP;
            row.copy_from_slice(&padded[start..start + N_FFT]);
        }
        let framed = Tensor::from_vec(framed, (frames, N_FFT), &self.device)?;
        let spectrum = framed.matmul(&self.dft)?;
        let real = spectrum.narrow(1, 0, N_BINS)?;
        let imag = spectrum.narrow(1, N_BINS, N_BINS)?;
        let magnitude = (real.sqr()? + imag.sqr()?)?.sqrt()?;
        let mel = self.mel_basis.matmul(&magnitude.t()?)?;
        Ok(mel.clamp(1e-5_f32, f32::MAX)?.log()?)
    }

    /// Runs the network on a `(128, frames)` log mel and returns `(frames, 360)`
    /// salience values in row-major order.
    pub fn salience(&self, mel: &Tensor) -> Result<Vec<f32>, InferenceError> {
        let frames = mel.dim(1)?;
        let mut output = vec![0.0_f32; frames * N_CLASSES];
        if frames <= CHUNK_FRAMES {
            let values = self.salience_window(mel)?;
            output.copy_from_slice(&values);
            return Ok(output);
        }
        let mut start = 0;
        while start < frames {
            let keep = CHUNK_FRAMES.min(frames - start);
            let left = start.saturating_sub(CHUNK_MARGIN);
            let right = (start + keep + CHUNK_MARGIN).min(frames);
            let window = mel.narrow(1, left, right - left)?;
            let values = self.salience_window(&window)?;
            let offset = start - left;
            output[start * N_CLASSES..(start + keep) * N_CLASSES]
                .copy_from_slice(&values[offset * N_CLASSES..(offset + keep) * N_CLASSES]);
            start += keep;
        }
        Ok(output)
    }

    fn salience_window(&self, mel: &Tensor) -> Result<Vec<f32>, InferenceError> {
        let frames = mel.dim(1)?;
        let padded_frames = frames.div_ceil(32) * 32;
        let mel = mel.pad_with_zeros(1, 0, padded_frames - frames)?;
        let mut x = mel
            .t()?
            .contiguous()?
            .reshape((1, 1, padded_frames, N_MELS))?;

        x = self.input_norm.forward(&x)?;
        let mut skips = Vec::with_capacity(ENCODER_LAYERS);
        for blocks in &self.encoders {
            for block in blocks {
                x = block.forward(&x)?;
            }
            skips.push(x.clone());
            x = x.avg_pool2d((2, 2))?;
        }
        for blocks in &self.intermediate {
            for block in blocks {
                x = block.forward(&x)?;
            }
        }
        for (decoder, skip) in self.decoders.iter().zip(skips.iter().rev()) {
            x = decoder.forward(&x, skip)?;
        }
        x = x
            .conv2d(&self.cnn_weight, 1, 1, 1, 1)?
            .broadcast_add(&self.cnn_bias.reshape((1, 3, 1, 1))?)?;
        let features = x
            .transpose(1, 2)?
            .contiguous()?
            .reshape((padded_frames, 3 * N_MELS))?;

        // Upstream runs the GRU over the zero-padded frames and slices
        // afterwards, so the backward direction starts inside the padding.
        let forward = self.gru[0].input_projection(&features)?;
        let backward = self.gru[1].input_projection(&features)?;
        let (forward, backward) = std::thread::scope(|scope| {
            let handle = scope.spawn(|| self.gru[0].run(&forward, padded_frames, false));
            let backward = self.gru[1].run(&backward, padded_frames, true);
            (handle.join().expect("GRU worker panicked"), backward)
        });
        let mut hidden = vec![0.0_f32; frames * 2 * GRU_HIDDEN];
        for (frame, row) in hidden
            .as_chunks_mut::<{ 2 * GRU_HIDDEN }>()
            .0
            .iter_mut()
            .enumerate()
        {
            row[..GRU_HIDDEN]
                .copy_from_slice(&forward[frame * GRU_HIDDEN..(frame + 1) * GRU_HIDDEN]);
            row[GRU_HIDDEN..]
                .copy_from_slice(&backward[frame * GRU_HIDDEN..(frame + 1) * GRU_HIDDEN]);
        }
        let hidden = Tensor::from_vec(hidden, (frames, 2 * GRU_HIDDEN), &self.device)?;
        let logits = hidden
            .matmul(&self.fc_weight.t()?)?
            .broadcast_add(&self.fc_bias)?;
        Ok(ops::sigmoid(&logits)?.flatten_all()?.to_vec1::<f32>()?)
    }
}

/// Converts `(frames, 360)` salience into Hz with RVC's local-average decoder.
pub fn decode_salience(salience: &[f32], threshold: f32) -> Vec<f32> {
    salience
        .as_chunks::<N_CLASSES>()
        .0
        .iter()
        .map(|row| {
            let (center, &peak) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap_or((0, &0.0));
            if peak <= threshold {
                return 0.0;
            }
            let start = center.saturating_sub(4);
            let end = (center + 5).min(N_CLASSES);
            let mut weighted = 0.0_f64;
            let mut total = 0.0_f64;
            for (bin, &value) in row.iter().enumerate().take(end).skip(start) {
                weighted += f64::from(value) * (20.0 * bin as f64 + 1_997.379_408_437_619);
                total += f64::from(value);
            }
            (10.0 * 2.0_f64.powf(weighted / total / 1_200.0)) as f32
        })
        .collect()
}

#[derive(Debug)]
struct BatchNorm {
    scale: Tensor,
    shift: Tensor,
}

impl BatchNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)
    }
}

#[derive(Debug)]
struct ConvBlockRes {
    conv1: Tensor,
    norm1: BatchNorm,
    conv2: Tensor,
    norm2: BatchNorm,
    shortcut: Option<(Tensor, Tensor)>,
}

impl ConvBlockRes {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self
            .norm1
            .forward(&x.conv2d(&self.conv1, 1, 1, 1, 1)?)?
            .relu()?;
        let y = self
            .norm2
            .forward(&y.conv2d(&self.conv2, 1, 1, 1, 1)?)?
            .relu()?;
        let residual = match &self.shortcut {
            Some((weight, bias)) => x
                .conv2d(weight, 0, 1, 1, 1)?
                .broadcast_add(&bias.reshape((1, bias.dim(0)?, 1, 1))?)?,
            None => x.clone(),
        };
        y + residual
    }
}

#[derive(Debug)]
struct DecoderBlock {
    transpose: Tensor,
    norm: BatchNorm,
    blocks: Vec<ConvBlockRes>,
}

impl DecoderBlock {
    fn forward(&self, x: &Tensor, skip: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.conv_transpose2d(&self.transpose, 1, 1, 2, 1)?;
        let mut x = self.norm.forward(&x)?.relu()?;
        x = Tensor::cat(&[&x, skip], 1)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

#[derive(Debug)]
struct GruDirection {
    input_weight: Tensor,
    input_bias: Tensor,
    hidden_weight: Vec<f32>,
    hidden_bias: Vec<f32>,
}

impl GruDirection {
    fn input_projection(&self, features: &Tensor) -> candle_core::Result<Vec<f32>> {
        features
            .matmul(&self.input_weight.t()?)?
            .broadcast_add(&self.input_bias)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    fn run(&self, projected: &[f32], frames: usize, reverse: bool) -> Vec<f32> {
        const H: usize = GRU_HIDDEN;
        let mut output = vec![0.0_f32; frames * H];
        let mut state = [0.0_f32; H];
        let mut recurrent = [0.0_f32; 3 * H];
        for step in 0..frames {
            let frame = if reverse { frames - 1 - step } else { step };
            for (row, value) in recurrent.iter_mut().enumerate() {
                let weights = &self.hidden_weight[row * H..(row + 1) * H];
                *value = self.hidden_bias[row]
                    + weights.iter().zip(&state).map(|(w, h)| w * h).sum::<f32>();
            }
            let input = &projected[frame * 3 * H..(frame + 1) * 3 * H];
            for unit in 0..H {
                let reset = sigmoid(input[unit] + recurrent[unit]);
                let update = sigmoid(input[H + unit] + recurrent[H + unit]);
                let candidate = (input[2 * H + unit] + reset * recurrent[2 * H + unit]).tanh();
                state[unit] = (1.0 - update) * candidate + update * state[unit];
            }
            output[frame * H..(frame + 1) * H].copy_from_slice(&state);
        }
        output
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct RmvpeWeights<'a> {
    tensors: &'a BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    device: &'a Device,
}

impl RmvpeWeights<'_> {
    fn raw(&self, name: &str) -> Result<&(Vec<usize>, Vec<f32>), InferenceError> {
        self.tensors
            .get(name)
            .ok_or_else(|| InferenceError::MissingWeight(name.to_owned()))
    }

    fn tensor(&self, name: &str) -> Result<Tensor, InferenceError> {
        let (shape, values) = self.raw(name)?;
        Ok(Tensor::from_slice(values, shape.as_slice(), self.device)?)
    }

    fn batch_norm(&self, prefix: &str) -> Result<BatchNorm, InferenceError> {
        let gamma = &self.raw(&format!("{prefix}.weight"))?.1;
        let beta = &self.raw(&format!("{prefix}.bias"))?.1;
        let mean = &self.raw(&format!("{prefix}.running_mean"))?.1;
        let var = &self.raw(&format!("{prefix}.running_var"))?.1;
        let channels = gamma.len();
        let mut scale = Vec::with_capacity(channels);
        let mut shift = Vec::with_capacity(channels);
        for c in 0..channels {
            let s = gamma[c] / (var[c] + BN_EPS as f32).sqrt();
            scale.push(s);
            shift.push(beta[c] - mean[c] * s);
        }
        Ok(BatchNorm {
            scale: Tensor::from_vec(scale, (1, channels, 1, 1), self.device)?,
            shift: Tensor::from_vec(shift, (1, channels, 1, 1), self.device)?,
        })
    }

    fn res_blocks(&self, prefix: &str) -> Result<Vec<ConvBlockRes>, InferenceError> {
        let mut blocks = Vec::with_capacity(BLOCKS);
        for j in 0..BLOCKS {
            let p = format!("{prefix}.{j}");
            let shortcut = match self.tensors.contains_key(&format!("{p}.shortcut.weight")) {
                true => Some((
                    self.tensor(&format!("{p}.shortcut.weight"))?,
                    self.tensor(&format!("{p}.shortcut.bias"))?,
                )),
                false => None,
            };
            blocks.push(ConvBlockRes {
                conv1: self.tensor(&format!("{p}.conv.0.weight"))?,
                norm1: self.batch_norm(&format!("{p}.conv.1"))?,
                conv2: self.tensor(&format!("{p}.conv.3.weight"))?,
                norm2: self.batch_norm(&format!("{p}.conv.4"))?,
                shortcut,
            });
        }
        Ok(blocks)
    }

    fn gru(&self, suffix: &str) -> Result<GruDirection, InferenceError> {
        let p = "fc.0.gru";
        let hidden_weight = self.raw(&format!("{p}.weight_hh_l0{suffix}"))?;
        if hidden_weight.0 != [3 * GRU_HIDDEN, GRU_HIDDEN] {
            return Err(InferenceError::UnsupportedModel(
                "RMVPE GRU hidden size is not 256".into(),
            ));
        }
        Ok(GruDirection {
            input_weight: self.tensor(&format!("{p}.weight_ih_l0{suffix}"))?,
            input_bias: self.tensor(&format!("{p}.bias_ih_l0{suffix}"))?,
            hidden_weight: hidden_weight.1.clone(),
            hidden_bias: self.raw(&format!("{p}.bias_hh_l0{suffix}"))?.1.clone(),
        })
    }
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    let reflect = |i: isize| -> f32 {
        if n == 1 {
            return samples[0];
        }
        let period = 2 * (n as isize - 1);
        let mut j = i.rem_euclid(period);
        if j >= n as isize {
            j = period - j;
        }
        samples[j as usize]
    };
    for i in -(pad as isize)..(n + pad) as isize {
        out.push(reflect(i));
    }
    out
}

fn dft_basis(device: &Device) -> candle_core::Result<Tensor> {
    let mut basis = vec![0.0_f32; N_FFT * 2 * N_BINS];
    for n in 0..N_FFT {
        let window = 0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / N_FFT as f64).cos();
        for k in 0..N_BINS {
            let angle = std::f64::consts::TAU * ((n * k) % N_FFT) as f64 / N_FFT as f64;
            basis[n * 2 * N_BINS + k] = (window * angle.cos()) as f32;
            basis[n * 2 * N_BINS + N_BINS + k] = (-window * angle.sin()) as f32;
        }
    }
    Tensor::from_vec(basis, (N_FFT, 2 * N_BINS), device)
}

fn mel_basis(device: &Device) -> candle_core::Result<Tensor> {
    let hz_to_mel = |hz: f64| 2_595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2_595.0) - 1.0);
    let low = hz_to_mel(MEL_FMIN);
    let high = hz_to_mel(MEL_FMAX);
    let points: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(low + (high - low) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut basis = vec![0.0_f32; N_MELS * N_BINS];
    for m in 0..N_MELS {
        let lower_width = points[m + 1] - points[m];
        let upper_width = points[m + 2] - points[m + 1];
        let norm = 2.0 / (points[m + 2] - points[m]);
        for k in 0..N_BINS {
            let freq = k as f64 * f64::from(RMVPE_SAMPLE_RATE) / N_FFT as f64;
            let lower = (freq - points[m]) / lower_width;
            let upper = (points[m + 2] - freq) / upper_width;
            basis[m * N_BINS + k] = (lower.min(upper).max(0.0) * norm) as f32;
        }
    }
    Tensor::from_vec(basis, (N_MELS, N_BINS), device)
}

#[cfg(test)]
mod parity {
    use super::*;
    use crate::rvc::parity::{assert_close, read_npy, workspace_root};

    #[test]
    fn matches_upstream_rmvpe() {
        let reference = workspace_root().join(".dev/reference/rmvpe");
        let Some(cache) = std::env::var_os("LOCALAPPDATA")
            .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
            .map(|base| Path::new(&base).join("rvc-rs/models/rmvpe.pt"))
        else {
            return;
        };
        if !cache.exists() || !reference.join("f0.npy").exists() {
            eprintln!("skipping RMVPE parity: fixture missing");
            return;
        }
        let model = Rmvpe::load(&cache, &Device::Cpu).unwrap();
        let audio = read_npy(&reference.join("audio.npy")).f32;
        let mel = model.log_mel(&audio).unwrap();
        let mel_values = mel.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_close(
            "rmvpe.mel",
            &mel_values,
            &read_npy(&reference.join("mel.npy")).f32,
            2e-3,
        );

        let expected_mel = read_npy(&reference.join("mel.npy"));
        let expected_mel = Tensor::from_vec(
            expected_mel.f32,
            (N_MELS, expected_mel.shape[2]),
            &Device::Cpu,
        )
        .unwrap();
        let salience = model.salience(&expected_mel).unwrap();
        assert_close(
            "rmvpe.hidden",
            &salience,
            &read_npy(&reference.join("hidden.npy")).f32,
            2e-3,
        );

        let f0 = decode_salience(&salience, 0.03);
        let expected = read_npy(&reference.join("f0.npy")).f32;
        assert_close("rmvpe.f0", &f0, &expected, 0.5);
    }
}
