//! Native RVC synthesizer implemented with Candle operations.
//!
//! The topology follows the public RVC inference implementation. Checkpoint
//! decoding remains exclusively in `pthrs`; this module only assembles and runs
//! tensors that have already been decoded.

#![allow(missing_docs)]

use crate::{InferenceError, NativeCheckpoint};
use candle_core::{Device, IndexOp, Tensor, D};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, LayerNorm,
    Linear, Module,
};
use rvc_rs_core::GeneratorInput;

const LRELU: f64 = 0.1;
// `torch.nn.functional.leaky_relu(x)` uses 0.01 when no slope is supplied.
// RVC intentionally uses that default only immediately before `conv_post`.
const FINAL_LRELU: f64 = 0.01;

#[derive(Debug)]
pub(crate) struct RvcSynthesizer {
    text: TextEncoder,
    flow: ResidualFlow,
    decoder: NsfDecoder,
    speakers: Embedding,
    device: Device,
    sample_rate: u32,
}

impl RvcSynthesizer {
    pub(crate) fn load(checkpoint: &NativeCheckpoint) -> Result<Self, InferenceError> {
        if !checkpoint.spec().uses_f0 {
            return Err(InferenceError::UnsupportedModel(
                "non-F0 RVC checkpoints are not implemented yet".into(),
            ));
        }
        let c = &checkpoint.info().config;
        let ws = Weights(checkpoint);
        let text = TextEncoder::load(
            &ws,
            checkpoint.spec().feature_dimension(),
            c.intermediate_channels as usize,
            c.hidden_channels as usize,
            c.filter_channels as usize,
            c.attention_heads as usize,
            c.attention_layers as usize,
            c.kernel_size as usize,
        )?;
        let flow = ResidualFlow::load(
            &ws,
            c.intermediate_channels as usize,
            c.hidden_channels as usize,
            c.speaker_embedding_channels as usize,
        )?;
        let decoder = NsfDecoder::load(&ws, c, checkpoint.spec().sample_rate.hz())?;
        let speakers = ws.embedding("emb_g.weight")?;
        Ok(Self {
            text,
            flow,
            decoder,
            speakers,
            device: checkpoint.device().clone(),
            sample_rate: checkpoint.spec().sample_rate.hz(),
        })
    }

    pub(crate) fn forward(&self, input: &GeneratorInput<'_>) -> Result<Vec<f32>, InferenceError> {
        let frames = input.features.frames;
        let phone = Tensor::from_slice(
            input.features.values,
            (1, frames, input.features.dimensions),
            &self.device,
        )?;
        let pitch = input.pitch.ok_or_else(|| {
            InferenceError::UnsupportedModel("F0 checkpoint requires a pitch track".into())
        })?;
        let coarse = Tensor::from_slice(pitch.coarse, (1, frames), &self.device)?;
        let f0 = Tensor::from_slice(pitch.continuous_hz, (1, frames), &self.device)?;
        let sid = Tensor::from_slice(&[input.speaker_id as i64], (1,), &self.device)?;
        let g = self.speakers.forward(&sid)?.unsqueeze(2)?;
        let (mean, logs) = self.text.forward(&phone, &coarse)?;
        let noise = Tensor::randn(0f32, 1f32, mean.shape(), &self.device)?;
        let sampled = ((logs.exp()? * noise)? * input.noise_scale as f64)?;
        let z_p = (&mean + sampled)?;
        let z = self.flow.reverse(&z_p, &g)?;
        self.decoder.forward(&z, &f0, &g, self.sample_rate)
    }
}

struct Weights<'a>(&'a NativeCheckpoint);

impl Weights<'_> {
    fn get(&self, name: &str) -> Result<Tensor, InferenceError> {
        Ok(self.0.weight(name)?.clone())
    }

    fn maybe(&self, name: &str) -> Option<Tensor> {
        self.0.weight(name).ok().cloned()
    }

    fn linear(&self, prefix: &str) -> Result<Linear, InferenceError> {
        let w = self.get(&format!("{prefix}.weight"))?;
        let b = self.maybe(&format!("{prefix}.bias"));
        Ok(Linear::new(w, b))
    }

    fn embedding(&self, name: &str) -> Result<Embedding, InferenceError> {
        let w = self.get(name)?;
        let hidden = w.dim(1)?;
        Ok(Embedding::new(w, hidden))
    }

    fn layer_norm(&self, prefix: &str) -> Result<LayerNorm, InferenceError> {
        Ok(LayerNorm::new(
            self.get(&format!("{prefix}.gamma"))?,
            self.get(&format!("{prefix}.beta"))?,
            1e-5,
        ))
    }

    fn conv(&self, prefix: &str, cfg: Conv1dConfig) -> Result<Conv1d, InferenceError> {
        Ok(Conv1d::new(
            self.get(&format!("{prefix}.weight"))?,
            self.maybe(&format!("{prefix}.bias")),
            cfg,
        ))
    }

    fn norm_conv(&self, prefix: &str, cfg: Conv1dConfig) -> Result<Conv1d, InferenceError> {
        let v = self.get(&format!("{prefix}.weight_v"))?;
        let g = self.get(&format!("{prefix}.weight_g"))?;
        let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
        let w = v.broadcast_mul(&g.broadcast_div(&norm)?)?;
        Ok(Conv1d::new(w, self.maybe(&format!("{prefix}.bias")), cfg))
    }

    fn norm_conv_transpose(
        &self,
        prefix: &str,
        cfg: ConvTranspose1dConfig,
    ) -> Result<ConvTranspose1d, InferenceError> {
        let v = self.get(&format!("{prefix}.weight_v"))?;
        let g = self.get(&format!("{prefix}.weight_g"))?;
        let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
        let w = v.broadcast_mul(&g.broadcast_div(&norm)?)?;
        Ok(ConvTranspose1d::new(
            w,
            self.maybe(&format!("{prefix}.bias")),
            cfg,
        ))
    }
}

#[derive(Debug)]
struct TextEncoder {
    phone: Linear,
    pitch: Embedding,
    encoder: Encoder,
    proj: Conv1d,
    hidden: usize,
    out: usize,
}

impl TextEncoder {
    #[allow(clippy::too_many_arguments)]
    fn load(
        w: &Weights<'_>,
        phone_dim: usize,
        out: usize,
        hidden: usize,
        filter: usize,
        heads: usize,
        layers: usize,
        kernel: usize,
    ) -> Result<Self, InferenceError> {
        let phone = w.linear("enc_p.emb_phone")?;
        if phone.weight().dims2()? != (hidden, phone_dim) {
            return Err(InferenceError::UnsupportedModel(
                "enc_p.emb_phone shape does not match checkpoint metadata".into(),
            ));
        }
        Ok(Self {
            phone,
            pitch: w.embedding("enc_p.emb_pitch.weight")?,
            encoder: Encoder::load(w, hidden, filter, heads, layers, kernel)?,
            proj: w.conv("enc_p.proj", Conv1dConfig::default())?,
            hidden,
            out,
        })
    }

    fn forward(&self, phone: &Tensor, pitch: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let x = (self.phone.forward(phone)? + self.pitch.forward(pitch)?)?;
        let x = ops::leaky_relu(&(x * (self.hidden as f64).sqrt())?, LRELU)?;
        let x = self.encoder.forward(&x.transpose(1, 2)?)?;
        let stats = self.proj.forward(&x)?;
        let parts = stats.chunk(2, 1)?;
        if parts.len() != 2 || parts[0].dim(1)? != self.out {
            candle_core::bail!("invalid enc_p projection output")
        }
        Ok((parts[0].clone(), parts[1].clone()))
    }
}

#[derive(Debug)]
struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn load(
        w: &Weights<'_>,
        hidden: usize,
        filter: usize,
        heads: usize,
        layers: usize,
        kernel: usize,
    ) -> Result<Self, InferenceError> {
        let mut out = Vec::with_capacity(layers);
        for i in 0..layers {
            out.push(EncoderLayer {
                attention: Attention::load(w, i, hidden, heads, 10)?,
                norm1: w.layer_norm(&format!("enc_p.encoder.norm_layers_1.{i}"))?,
                ffn: FeedForward::load(w, i, kernel)?,
                norm2: w.layer_norm(&format!("enc_p.encoder.norm_layers_2.{i}"))?,
                filter,
            });
        }
        Ok(Self { layers: out })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

#[derive(Debug)]
struct EncoderLayer {
    attention: Attention,
    norm1: LayerNorm,
    ffn: FeedForward,
    norm2: LayerNorm,
    filter: usize,
}

impl EncoderLayer {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.attention.forward(x)?;
        let x = channel_norm(&self.norm1, &(x + y)?)?;
        let y = self.ffn.forward(&x)?;
        let x = channel_norm(&self.norm2, &(x + y)?)?;
        let _ = self.filter;
        Ok(x)
    }
}

#[derive(Debug)]
struct FeedForward {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl FeedForward {
    fn load(w: &Weights<'_>, i: usize, kernel: usize) -> Result<Self, InferenceError> {
        let cfg = Conv1dConfig {
            padding: kernel / 2,
            ..Default::default()
        };
        Ok(Self {
            conv1: w.conv(&format!("enc_p.encoder.ffn_layers.{i}.conv_1"), cfg)?,
            conv2: w.conv(&format!("enc_p.encoder.ffn_layers.{i}.conv_2"), cfg)?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.conv2.forward(&self.conv1.forward(x)?.relu()?)
    }
}

#[derive(Debug)]
struct Attention {
    q: Conv1d,
    k: Conv1d,
    v: Conv1d,
    o: Conv1d,
    rel_k: Tensor,
    rel_v: Tensor,
    heads: usize,
    window: usize,
}

impl Attention {
    fn load(
        w: &Weights<'_>,
        i: usize,
        hidden: usize,
        heads: usize,
        window: usize,
    ) -> Result<Self, InferenceError> {
        if !hidden.is_multiple_of(heads) {
            return Err(InferenceError::UnsupportedModel(
                "attention width is not divisible by head count".into(),
            ));
        }
        let p = format!("enc_p.encoder.attn_layers.{i}");
        Ok(Self {
            q: w.conv(&format!("{p}.conv_q"), Conv1dConfig::default())?,
            k: w.conv(&format!("{p}.conv_k"), Conv1dConfig::default())?,
            v: w.conv(&format!("{p}.conv_v"), Conv1dConfig::default())?,
            o: w.conv(&format!("{p}.conv_o"), Conv1dConfig::default())?,
            rel_k: w.get(&format!("{p}.emb_rel_k"))?,
            rel_v: w.get(&format!("{p}.emb_rel_v"))?,
            heads,
            window,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let q = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;
        let (b, d, t) = q.dims3()?;
        let dk = d / self.heads;
        let q = (q.reshape((b, self.heads, dk, t))?.transpose(2, 3)? / (dk as f64).sqrt())?;
        let k = k.reshape((b, self.heads, dk, t))?.transpose(2, 3)?;
        let v = v.reshape((b, self.heads, dk, t))?.transpose(2, 3)?;
        let mut scores = q.matmul(&k.transpose(2, 3)?)?;
        let rk = relative_embeddings(&self.rel_k, t, self.window)?;
        let rel_logits = q.broadcast_matmul(&rk.unsqueeze(0)?.transpose(2, 3)?)?;
        scores = (scores + relative_to_absolute(&rel_logits)?)?;
        let attn = ops::softmax(&scores, D::Minus1)?;
        let mut out = attn.matmul(&v)?;
        let rv = relative_embeddings(&self.rel_v, t, self.window)?;
        out = (out + absolute_to_relative(&attn)?.broadcast_matmul(&rv.unsqueeze(0)?)?)?;
        self.o.forward(&out.transpose(2, 3)?.reshape((b, d, t))?)
    }
}

fn relative_embeddings(x: &Tensor, length: usize, window: usize) -> candle_core::Result<Tensor> {
    let pad = length.saturating_sub(window + 1);
    let x = if pad > 0 {
        x.pad_with_zeros(1, pad, pad)?
    } else {
        x.clone()
    };
    let start = (window + 1).saturating_sub(length);
    x.narrow(1, start, 2 * length - 1)
}

fn relative_to_absolute(x: &Tensor) -> candle_core::Result<Tensor> {
    let (b, h, l, _) = x.dims4()?;
    let x = x.pad_with_zeros(3, 0, 1)?;
    let x = x.reshape((b, h, l * 2 * l))?.pad_with_zeros(2, 0, l - 1)?;
    x.reshape((b, h, l + 1, 2 * l - 1))?
        .narrow(2, 0, l)?
        .narrow(3, l - 1, l)
}

fn absolute_to_relative(x: &Tensor) -> candle_core::Result<Tensor> {
    let (b, h, l, _) = x.dims4()?;
    let x = x.pad_with_zeros(3, 0, l - 1)?;
    let x = x
        .reshape((b, h, l * l + l * (l - 1)))?
        .pad_with_zeros(2, l, 0)?;
    x.reshape((b, h, l, 2 * l))?.narrow(3, 1, 2 * l - 1)
}

fn channel_norm(norm: &LayerNorm, x: &Tensor) -> candle_core::Result<Tensor> {
    norm.forward(&x.transpose(1, 2)?)?.transpose(1, 2)
}

#[derive(Debug)]
struct ResidualFlow {
    layers: Vec<CouplingLayer>,
}

impl ResidualFlow {
    fn load(
        w: &Weights<'_>,
        channels: usize,
        hidden: usize,
        gin: usize,
    ) -> Result<Self, InferenceError> {
        let mut layers = Vec::with_capacity(4);
        for i in 0..4 {
            layers.push(CouplingLayer::load(
                w,
                &format!("flow.flows.{}", i * 2),
                channels,
                hidden,
                gin,
            )?);
        }
        Ok(Self { layers })
    }

    fn reverse(&self, x: &Tensor, g: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for layer in self.layers.iter().rev() {
            x = x.flip(&[1])?;
            x = layer.reverse(&x, g)?;
        }
        Ok(x)
    }
}

#[derive(Debug)]
struct CouplingLayer {
    pre: Conv1d,
    wn: WaveNet,
    post: Conv1d,
    half: usize,
}

impl CouplingLayer {
    fn load(
        w: &Weights<'_>,
        p: &str,
        channels: usize,
        hidden: usize,
        gin: usize,
    ) -> Result<Self, InferenceError> {
        Ok(Self {
            pre: w.conv(&format!("{p}.pre"), Conv1dConfig::default())?,
            wn: WaveNet::load(w, &format!("{p}.enc"), hidden, gin)?,
            post: w.conv(&format!("{p}.post"), Conv1dConfig::default())?,
            half: channels / 2,
        })
    }

    fn reverse(&self, x: &Tensor, g: &Tensor) -> candle_core::Result<Tensor> {
        let x0 = x.narrow(1, 0, self.half)?;
        let x1 = x.narrow(1, self.half, self.half)?;
        let h = self.wn.forward(&self.pre.forward(&x0)?, g)?;
        let x1 = (x1 - self.post.forward(&h)?)?;
        Tensor::cat(&[&x0, &x1], 1)
    }
}

#[derive(Debug)]
struct WaveNet {
    cond: Conv1d,
    input: Vec<Conv1d>,
    residual: Vec<Conv1d>,
    hidden: usize,
}

impl WaveNet {
    fn load(w: &Weights<'_>, p: &str, hidden: usize, _gin: usize) -> Result<Self, InferenceError> {
        let cond = w.norm_conv(&format!("{p}.cond_layer"), Conv1dConfig::default())?;
        let mut input = Vec::with_capacity(3);
        let mut residual = Vec::with_capacity(3);
        for i in 0..3 {
            input.push(w.norm_conv(
                &format!("{p}.in_layers.{i}"),
                Conv1dConfig {
                    padding: (5 * (1usize << i) - (1usize << i)) / 2,
                    dilation: 1usize << i,
                    ..Default::default()
                },
            )?);
            residual
                .push(w.norm_conv(&format!("{p}.res_skip_layers.{i}"), Conv1dConfig::default())?);
        }
        Ok(Self {
            cond,
            input,
            residual,
            hidden,
        })
    }

    fn forward(&self, x: &Tensor, g: &Tensor) -> candle_core::Result<Tensor> {
        let cond = self.cond.forward(g)?;
        let mut x = x.clone();
        let mut output = Tensor::zeros_like(&x)?;
        for i in 0..3 {
            let c = cond.narrow(1, i * 2 * self.hidden, 2 * self.hidden)?;
            let acts = self.input[i].forward(&x)?.broadcast_add(&c)?;
            let a = acts.narrow(1, 0, self.hidden)?.tanh()?;
            let b = ops::sigmoid(&acts.narrow(1, self.hidden, self.hidden)?)?;
            let rs = self.residual[i].forward(&(a * b)?)?;
            if i < 2 {
                x = (x + rs.narrow(1, 0, self.hidden)?)?;
                output = (output + rs.narrow(1, self.hidden, self.hidden)?)?;
            } else {
                output = (output + rs)?;
            }
        }
        Ok(output)
    }
}

#[derive(Debug)]
struct NsfDecoder {
    conv_pre: Conv1d,
    cond: Conv1d,
    source_weight: Tensor,
    source_bias: Tensor,
    ups: Vec<ConvTranspose1d>,
    noises: Vec<Conv1d>,
    resblocks: Vec<Vec<ResBlock>>,
    conv_post: Conv1d,
    rates: Vec<usize>,
}

impl NsfDecoder {
    fn load(
        w: &Weights<'_>,
        c: &pthrs::VoiceModelConfig,
        _sample_rate: u32,
    ) -> Result<Self, InferenceError> {
        let conv_pre = w.conv(
            "dec.conv_pre",
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        )?;
        let cond = w.conv("dec.cond", Conv1dConfig::default())?;
        let source_weight = w.get("dec.m_source.l_linear.weight")?;
        let source_bias = w.get("dec.m_source.l_linear.bias")?;
        let rates: Vec<usize> = c.upsample_rates.iter().map(|&x| x as usize).collect();
        let kernels: Vec<usize> = c
            .upsample_kernel_sizes
            .iter()
            .map(|&x| x as usize)
            .collect();
        let mut ups = Vec::with_capacity(rates.len());
        let mut noises = Vec::with_capacity(rates.len());
        let mut resblocks = Vec::with_capacity(rates.len());
        for i in 0..rates.len() {
            ups.push(w.norm_conv_transpose(
                &format!("dec.ups.{i}"),
                ConvTranspose1dConfig {
                    padding: (kernels[i] - rates[i]) / 2,
                    stride: rates[i],
                    ..Default::default()
                },
            )?);
            let remaining = rates[i + 1..].iter().product::<usize>();
            noises.push(w.conv(
                &format!("dec.noise_convs.{i}"),
                if remaining > 1 {
                    Conv1dConfig {
                        padding: remaining / 2,
                        stride: remaining,
                        ..Default::default()
                    }
                } else {
                    Conv1dConfig::default()
                },
            )?);
            let mut stage = Vec::with_capacity(c.resblock_kernel_sizes.len());
            for j in 0..c.resblock_kernel_sizes.len() {
                let flat = i * c.resblock_kernel_sizes.len() + j;
                let dilations: Vec<usize> = c.resblock_dilation_sizes[j]
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                stage.push(ResBlock::load(
                    w,
                    &format!("dec.resblocks.{flat}"),
                    c.resblock_kernel_sizes[j] as usize,
                    &dilations,
                    c.resblock == "1",
                )?);
            }
            resblocks.push(stage);
        }
        Ok(Self {
            conv_pre,
            cond,
            source_weight,
            source_bias,
            ups,
            noises,
            resblocks,
            conv_post: w.conv(
                "dec.conv_post",
                Conv1dConfig {
                    padding: 3,
                    ..Default::default()
                },
            )?,
            rates,
        })
    }

    fn forward(
        &self,
        z: &Tensor,
        f0: &Tensor,
        g: &Tensor,
        sample_rate: u32,
    ) -> Result<Vec<f32>, InferenceError> {
        let hop = self.rates.iter().product::<usize>();
        let f0v = f0.i(0)?.to_vec1::<f32>()?;
        let source = sine_source(&f0v, hop, sample_rate);
        let mut source = Tensor::from_vec(source, (1, 1, f0v.len() * hop), z.device())?;
        source = source.broadcast_mul(&self.source_weight.reshape((1, 1, 1))?)?;
        source = source
            .broadcast_add(&self.source_bias.reshape((1, 1, 1))?)?
            .tanh()?;
        let mut x = self
            .conv_pre
            .forward(z)?
            .broadcast_add(&self.cond.forward(g)?)?;
        for i in 0..self.ups.len() {
            x = self.ups[i].forward(&ops::leaky_relu(&x, LRELU)?)?;
            let noise = self.noises[i].forward(&source)?;
            let tx = x.dim(2)?;
            let tn = noise.dim(2)?;
            let t = tx.min(tn);
            x = (x.narrow(2, 0, t)? + noise.narrow(2, 0, t)?)?;
            let mut sum: Option<Tensor> = None;
            for block in &self.resblocks[i] {
                let y = block.forward(&x)?;
                sum = Some(match sum {
                    Some(v) => (v + y)?,
                    None => y,
                });
            }
            x = (sum.ok_or_else(|| candle_core::Error::Msg("empty resblock stage".into()))?
                / self.resblocks[i].len() as f64)?;
        }
        let y = self
            .conv_post
            .forward(&ops::leaky_relu(&x, FINAL_LRELU)?)?
            .tanh()?;
        Ok(y.i((0, 0, ..))?.to_vec1::<f32>()?)
    }
}

#[derive(Debug)]
enum ResBlock {
    One(Vec<(Conv1d, Conv1d)>),
    Two(Vec<Conv1d>),
}

impl ResBlock {
    fn load(
        w: &Weights<'_>,
        p: &str,
        kernel: usize,
        dilations: &[usize],
        kind_one: bool,
    ) -> Result<Self, InferenceError> {
        if kind_one {
            let mut pairs = Vec::with_capacity(dilations.len());
            for (i, &d) in dilations.iter().enumerate() {
                let c1 = w.norm_conv(
                    &format!("{p}.convs1.{i}"),
                    Conv1dConfig {
                        padding: (kernel * d - d) / 2,
                        dilation: d,
                        ..Default::default()
                    },
                )?;
                let c2 = w.norm_conv(
                    &format!("{p}.convs2.{i}"),
                    Conv1dConfig {
                        padding: (kernel - 1) / 2,
                        ..Default::default()
                    },
                )?;
                pairs.push((c1, c2));
            }
            Ok(Self::One(pairs))
        } else {
            let mut convs = Vec::with_capacity(dilations.len());
            for (i, &d) in dilations.iter().enumerate() {
                convs.push(w.norm_conv(
                    &format!("{p}.convs.{i}"),
                    Conv1dConfig {
                        padding: (kernel * d - d) / 2,
                        dilation: d,
                        ..Default::default()
                    },
                )?);
            }
            Ok(Self::Two(convs))
        }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        match self {
            Self::One(pairs) => {
                for (a, b) in pairs {
                    let y = a.forward(&ops::leaky_relu(&x, LRELU)?)?;
                    let y = b.forward(&ops::leaky_relu(&y, LRELU)?)?;
                    x = (x + y)?;
                }
            }
            Self::Two(convs) => {
                for c in convs {
                    let y = c.forward(&ops::leaky_relu(&x, LRELU)?)?;
                    x = (x + y)?;
                }
            }
        }
        Ok(x)
    }
}

fn sine_source(f0: &[f32], hop: usize, sample_rate: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(f0.len() * hop);
    // Match RVC SineGen._f02sine: each frame contains the fractional
    // phase for samples 1..=hop, offset by the wrapped cumulative phase
    // of all preceding frames.  Accumulating sample-by-sample looks
    // similar for constant F0 but diverges at frame and voiced/unvoiced
    // boundaries.
    let mut accumulated_phase = 0.0_f64;
    let mut rng = 0x9e37_79b9_u32;
    let mut spare_gaussian = None;
    for &hz in f0 {
        let cycles_per_sample = f64::from(hz) / f64::from(sample_rate);
        for sample in 1..=hop {
            let noise = gaussian(&mut rng, &mut spare_gaussian);
            if hz > 0.0 {
                let phase = accumulated_phase + cycles_per_sample * sample as f64;
                out.push(
                    ((phase * std::f64::consts::TAU).sin() * 0.1 + noise * 0.003) as f32,
                );
            } else {
                out.push((noise * (0.1 / 3.0)) as f32);
            }
        }
        let endpoint = cycles_per_sample * hop as f64;
        let wrapped_endpoint = (endpoint + 0.5) % 1.0 - 0.5;
        accumulated_phase = (accumulated_phase + wrapped_endpoint) % 1.0;
    }
    out
}

fn gaussian(rng: &mut u32, spare: &mut Option<f64>) -> f64 {
    if let Some(value) = spare.take() {
        return value;
    }

    let uniform = |state: &mut u32| {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        // Keep Box-Muller away from ln(0), while retaining a uniform
        // distribution over the representable xorshift output range.
        (f64::from(*state) + 1.0) / (f64::from(u32::MAX) + 2.0)
    };
    let u1 = uniform(rng);
    let u2 = uniform(rng);
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = std::f64::consts::TAU * u2;
    *spare = Some(radius * angle.sin());
    radius * angle.cos()
}
