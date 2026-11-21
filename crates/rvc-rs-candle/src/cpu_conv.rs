//! CPU fast path for the "same"-padded 1-D convolutions in the HiFi-GAN decoder.
//!
//! Candle's CPU conv1d materializes the whole im2col matrix on one thread, which
//! dominates decoder time at 40-48 kHz. This path keeps activations time-major
//! (`length x channels`), so every im2col row is a few contiguous channel
//! copies. Each worker thread processes cache-sized time chunks, applies the
//! preceding leaky ReLU while gathering, and fuses bias and residual additions
//! into the copy-out. The matrix products still run through Candle.

use candle_core::{Device, Tensor};

const CHUNK: usize = 512;

#[derive(Debug)]
pub(crate) struct CpuConv {
    weight: Tensor,
    bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
    padding: usize,
}

impl CpuConv {
    pub(crate) fn new(
        weight: &Tensor,
        bias: Option<&Tensor>,
        dilation: usize,
        padding: usize,
    ) -> candle_core::Result<Self> {
        let (out_channels, in_channels, kernel) = weight.dims3()?;
        if 2 * padding != dilation * (kernel - 1) {
            candle_core::bail!("CpuConv supports only length-preserving padding")
        }
        let bias = match bias {
            Some(bias) => bias.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
            None => vec![0.0; out_channels],
        };
        // (out, in, tap) -> (tap * in + in_index, out) to match the im2col rows.
        let weight = weight
            .to_device(&Device::Cpu)?
            .permute((2, 1, 0))?
            .contiguous()?
            .reshape((kernel * in_channels, out_channels))?;
        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel,
            dilation,
            padding,
        })
    }

    /// Returns `residual + bias + conv(leaky_relu(x, slope))` for a time-major
    /// `(length, in_channels)` input, producing `(length, out_channels)`.
    pub(crate) fn forward(
        &self,
        x: &[f32],
        length: usize,
        slope: f32,
        residual: Option<&[f32]>,
    ) -> candle_core::Result<Vec<f32>> {
        debug_assert_eq!(x.len(), self.in_channels * length);
        let mut output = vec![0.0_f32; self.out_channels * length];
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let mut assignments: Vec<Vec<(usize, &mut [f32])>> =
            (0..threads).map(|_| Vec::new()).collect();
        for (chunk, block) in output.chunks_mut(CHUNK * self.out_channels).enumerate() {
            assignments[chunk % threads].push((chunk * CHUNK, block));
        }
        std::thread::scope(|scope| {
            let workers: Vec<_> = assignments
                .into_iter()
                .map(|blocks| {
                    scope.spawn(move || self.run_blocks(x, length, slope, residual, blocks))
                })
                .collect();
            workers
                .into_iter()
                .try_for_each(|worker| worker.join().expect("convolution worker panicked"))
        })?;
        Ok(output)
    }

    fn run_blocks(
        &self,
        x: &[f32],
        length: usize,
        slope: f32,
        residual: Option<&[f32]>,
        blocks: Vec<(usize, &mut [f32])>,
    ) -> candle_core::Result<()> {
        let (cin, cout) = (self.in_channels, self.out_channels);
        let row_width = self.kernel * cin;
        let mut columns = vec![0.0_f32; CHUNK * row_width];
        for (start, block) in blocks {
            let rows = block.len() / cout;
            for (j, row) in columns.chunks_exact_mut(row_width).take(rows).enumerate() {
                for (tap, destination) in row.chunks_exact_mut(cin).enumerate() {
                    let position =
                        (start + j + tap * self.dilation) as isize - self.padding as isize;
                    if position < 0 || position >= length as isize {
                        destination.fill(0.0);
                        continue;
                    }
                    let source = &x[position as usize * cin..(position as usize + 1) * cin];
                    for (d, &v) in destination.iter_mut().zip(source) {
                        *d = if v < 0.0 { v * slope } else { v };
                    }
                }
            }
            let columns = Tensor::from_slice(
                &columns[..rows * row_width],
                (rows, row_width),
                &Device::Cpu,
            )?;
            let product = columns
                .matmul(&self.weight)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let skip = residual.map(|r| &r[start * cout..(start + rows) * cout]);
            for (j, (out, value)) in block
                .chunks_exact_mut(cout)
                .zip(product.chunks_exact(cout))
                .enumerate()
            {
                match skip {
                    Some(skip) => {
                        let skip = &skip[j * cout..(j + 1) * cout];
                        for (((o, &v), &b), &s) in
                            out.iter_mut().zip(value).zip(&self.bias).zip(skip)
                        {
                            *o = v + b + s;
                        }
                    }
                    None => {
                        for ((o, &v), &b) in out.iter_mut().zip(value).zip(&self.bias) {
                            *o = v + b;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_candle_conv1d_with_leaky_relu_and_residual() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let (channels, length, kernel, dilation) = (5, 9_000, 7, 3);
        let x = Tensor::randn(0f32, 1., (1, channels, length), &device)?;
        let w = Tensor::randn(0f32, 0.2, (channels, channels, kernel), &device)?;
        let b = Tensor::randn(0f32, 0.2, channels, &device)?;
        let padding = dilation * (kernel - 1) / 2;
        let expected = (candle_nn::ops::leaky_relu(&x, 0.1)?
            .conv1d(&w, padding, 1, dilation, 1)?
            .broadcast_add(&b.reshape((1, channels, 1))?)?
            + &x)?
            .squeeze(0)?
            .t()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let input = x.squeeze(0)?.t()?.flatten_all()?.to_vec1::<f32>()?;
        let actual = CpuConv::new(&w, Some(&b), dilation, padding)?.forward(
            &input,
            length,
            0.1,
            Some(&input),
        )?;
        let worst = actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-4, "{worst}");
        Ok(())
    }

    #[test]
    #[ignore = "benchmark"]
    fn bench_decoder_sized_conv() -> candle_core::Result<()> {
        let (c, l, k, dil) = (128_usize, 219_800_usize, 11_usize, 5_usize);
        let x = vec![0.1_f32; c * l];
        let w = Tensor::randn(0f32, 0.01, (c, c, k), &Device::Cpu)?;
        let conv = CpuConv::new(&w, None, dil, (k * dil - dil) / 2)?;
        for _ in 0..2 {
            let started = std::time::Instant::now();
            let _ = conv.forward(&x, l, 0.1, Some(&x))?;
            eprintln!("cpu conv {:?}", started.elapsed());
        }
        Ok(())
    }
}
