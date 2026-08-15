#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Allocation-free DSP primitives shared by offline and streaming pipelines.

use std::f32::consts::PI;
use thiserror::Error;

/// Fixed-capacity history used by the real-time worker.
///
/// Storage is allocated once in [`RollingBuffer::new`]. Appending moves values
/// in place and retains exactly the newest `capacity()` elements.
#[derive(Clone, Debug, PartialEq)]
pub struct RollingBuffer<T> {
    values: Vec<T>,
}

impl<T: Copy> RollingBuffer<T> {
    /// Allocates a history buffer and fills it with a startup value.
    pub fn new(capacity: usize, fill: T) -> Self {
        Self {
            values: vec![fill; capacity],
        }
    }

    /// Number of retained elements.
    pub fn capacity(&self) -> usize {
        self.values.len()
    }

    /// Current history from oldest to newest.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    /// Mutable history for in-place feature and F0 updates.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Appends values without allocating, dropping the oldest values.
    pub fn push(&mut self, input: &[T]) {
        let capacity = self.values.len();
        if capacity == 0 || input.is_empty() {
            return;
        }
        if input.len() >= capacity {
            self.values
                .copy_from_slice(&input[input.len() - capacity..]);
            return;
        }
        let retained = capacity - input.len();
        self.values.copy_within(input.len().., 0);
        self.values[retained..].copy_from_slice(input);
    }
}

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
    /// A SOLA candidate does not contain search, block, and tail regions.
    #[error("SOLA candidate is too short: need at least {expected} samples, got {actual}")]
    SolaCandidate {
        /// Minimum required candidate length.
        expected: usize,
        /// Supplied candidate length.
        actual: usize,
    },
    /// SOLA output and next-tail buffers do not match the declared sizes.
    #[error(
        "invalid SOLA buffers: block={block}, previous_tail={previous_tail}, next_tail={next_tail}"
    )]
    SolaBuffers {
        /// Output block length.
        block: usize,
        /// Previous tail length.
        previous_tail: usize,
        /// Next tail length.
        next_tail: usize,
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
    for (index, ((left, right), destination)) in
        previous_tail.iter().zip(next_head).zip(output).enumerate()
    {
        let phase = (index + 1) as f32 / denominator;
        let right_gain = 0.5 - 0.5 * (PI * phase).cos();
        *destination = left * (1.0 - right_gain) + right * right_gain;
    }
    Ok(())
}

/// Aligns and crossfades one generated real-time block using SOLA.
///
/// `candidate` must contain `search_samples + output.len() + next_tail.len()`
/// samples. The first `search_samples + previous_tail.len()` samples are
/// searched using normalized correlation, matching the streaming strategy in
/// MMVCServerSIO. The selected block is written to `output`, and the following
/// overlap is windowed into `next_tail` for the next call.
///
/// All memory is caller-owned; the function performs no heap allocation.
pub fn sola_align_and_crossfade(
    previous_tail: &[f32],
    candidate: &[f32],
    search_samples: usize,
    output: &mut [f32],
    next_tail: &mut [f32],
) -> Result<usize, DspError> {
    if previous_tail.len() != next_tail.len() || output.len() < previous_tail.len() {
        return Err(DspError::SolaBuffers {
            block: output.len(),
            previous_tail: previous_tail.len(),
            next_tail: next_tail.len(),
        });
    }
    let expected = search_samples
        .checked_add(output.len())
        .and_then(|length| length.checked_add(next_tail.len()))
        .unwrap_or(usize::MAX);
    if candidate.len() < expected {
        return Err(DspError::SolaCandidate {
            expected,
            actual: candidate.len(),
        });
    }

    let overlap = previous_tail.len();
    let mut best_offset = 0;
    let mut best_score = f64::NEG_INFINITY;
    for offset in 0..=search_samples {
        let head = &candidate[offset..offset + overlap];
        let mut numerator = 0.0_f64;
        let mut energy = 1e-3_f64;
        for (&left, &right) in head.iter().zip(previous_tail) {
            numerator += f64::from(left) * f64::from(right);
            energy += f64::from(left) * f64::from(left);
        }
        let score = numerator / energy.sqrt();
        if score > best_score {
            best_score = score;
            best_offset = offset;
        }
    }

    output.copy_from_slice(&candidate[best_offset..best_offset + output.len()]);
    if overlap != 0 {
        let denominator = (overlap + 1) as f32;
        for index in 0..overlap {
            let phase = (index + 1) as f32 / denominator;
            let current_gain = 0.5 - 0.5 * (PI * phase).cos();
            output[index] =
                previous_tail[index] * (1.0 - current_gain) + output[index] * current_gain;
        }
        let tail_start = best_offset + output.len();
        for (index, destination) in next_tail.iter_mut().enumerate() {
            let phase = (index + 1) as f32 / denominator;
            let previous_gain = 0.5 + 0.5 * (PI * phase).cos();
            *destination = candidate[tail_start + index] * previous_gain;
        }
    }
    Ok(best_offset)
}

/// Returns the absolute sample peak, or zero for an empty slice.
pub fn peak(samples: &[f32]) -> f32 {
    samples.iter().copied().map(f32::abs).fold(0.0, f32::max)
}

/// Returns the root-mean-square level, or zero for an empty slice.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean_square =
        samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32;
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

    #[test]
    fn sola_finds_matching_offset_without_allocating() {
        let previous = [1.0, 0.5, -0.5, -1.0];
        let candidate = [
            0.0, 0.0, 1.0, 0.5, -0.5, -1.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25,
        ];
        let mut output = [0.0; 6];
        let mut next_tail = [0.0; 4];
        let offset =
            sola_align_and_crossfade(&previous, &candidate, 2, &mut output, &mut next_tail)
                .unwrap();
        assert_eq!(offset, 2);
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(next_tail.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn sola_rejects_short_candidates() {
        let error = sola_align_and_crossfade(&[0.0; 2], &[0.0; 5], 2, &mut [0.0; 3], &mut [0.0; 2])
            .unwrap_err();
        assert!(matches!(error, DspError::SolaCandidate { .. }));
    }

    #[test]
    fn rolling_buffer_retains_newest_values() {
        let mut buffer = RollingBuffer::new(5, 0_i32);
        buffer.push(&[1, 2]);
        assert_eq!(buffer.as_slice(), &[0, 0, 0, 1, 2]);
        buffer.push(&[3, 4, 5, 6]);
        assert_eq!(buffer.as_slice(), &[2, 3, 4, 5, 6]);
        buffer.push(&[7, 8, 9, 10, 11, 12]);
        assert_eq!(buffer.as_slice(), &[8, 9, 10, 11, 12]);
    }
}
