#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Allocation-free DSP primitives shared by offline and streaming pipelines.

use std::f32::consts::PI;
use thiserror::Error;

/// Errors caused by incompatible caller-owned audio buffers.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum DspError {
    /// Channel count must be non-zero.
    #[error("channel count must be greater than zero")]
    ZeroChannels,
    /// Interleaved input did not contain complete frames.
    #[error("interleaved sample count {samples} is not divisible by {channels} channels")]
    IncompleteFrame {
        /// Number of supplied samples.
        samples: usize,
        /// Number of channels per frame.
        channels: usize,
    },
    /// Caller-owned output has an unexpected length.
    #[error("output length mismatch: expected {expected}, got {actual}")]
    OutputLength {
        /// Required output length.
        expected: usize,
        /// Supplied output length.
        actual: usize,
    },
    /// Crossfade inputs do not have identical lengths.
    #[error("crossfade length mismatch: tail={tail}, head={head}, output={output}")]
    CrossfadeLength {
        /// Previous chunk tail length.
        tail: usize,
        /// Next chunk head length.
        head: usize,
        /// Output slice length.
        output: usize,
    },
}

/// Mixes interleaved channels into a caller-owned mono buffer.
pub fn interleaved_to_mono(
    input: &[f32],
    channels: usize,
    output: &mut [f32],
) -> Result<(), DspError> {
    if channels == 0 {
        return Err(DspError::ZeroChannels);
    }
    if !input.len().is_multiple_of(channels) {
        return Err(DspError::IncompleteFrame {
            samples: input.len(),
            channels,
        });
    }

    let frames = input.len() / channels;
    if output.len() != frames {
        return Err(DspError::OutputLength {
            expected: frames,
            actual: output.len(),
        });
    }

    let scale = 1.0 / channels as f32;
    for (destination, frame) in output.iter_mut().zip(input.chunks_exact(channels)) {
        *destination = frame.iter().copied().sum::<f32>() * scale;
    }
    Ok(())
}

/// Blends two equally-sized chunk boundaries with a raised-cosine curve.
pub fn raised_cosine_crossfade(
    previous_tail: &[f32],
    next_head: &[f32],
    output: &mut [f32],
) -> Result<(), DspError> {
    if previous_tail.len() != next_head.len() || output.len() != previous_tail.len() {
        return Err(DspError::CrossfadeLength {
            tail: previous_tail.len(),
            head: next_head.len(),
            output: output.len(),
        });
    }

    if output.is_empty() {
        return Ok(());
    }

    let denominator = (output.len() + 1) as f32;
    for (index, ((left, right), destination)) in previous_tail
        .iter()
        .zip(next_head)
        .zip(output)
        .enumerate()
    {
        let phase = (index + 1) as f32 / denominator;
        let right_gain = 0.5 - 0.5 * (PI * phase).cos();
        *destination = left * (1.0 - right_gain) + right * right_gain;
    }
    Ok(())
}

/// Returns the absolute sample peak, or zero for an empty slice.
pub fn peak(samples: &[f32]) -> f32 {
    samples
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0, f32::max)
}

/// Returns the root-mean-square level, or zero for an empty slice.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean_square = samples.iter().map(|sample| sample * sample).sum::<f32>()
        / samples.len() as f32;
    mean_square.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixes_stereo_to_mono_without_allocation() {
        let input = [1.0, -1.0, 0.25, 0.75];
        let mut output = [0.0; 2];
        interleaved_to_mono(&input, 2, &mut output).unwrap();
        assert_eq!(output, [0.0, 0.5]);
    }

    #[test]
    fn crossfade_moves_between_chunks() {
        let mut output = [0.0; 4];
        raised_cosine_crossfade(&[1.0; 4], &[0.0; 4], &mut output).unwrap();
        assert!(output.windows(2).all(|pair| pair[0] > pair[1]));
        assert!(output[0] < 1.0);
        assert!(output[3] > 0.0);
    }

    #[test]
    fn measures_peak_and_rms() {
        assert_eq!(peak(&[-0.25, 0.75, -0.5]), 0.75);
        assert!((rms(&[1.0, -1.0]) - 1.0).abs() < f32::EPSILON);
    }
}

