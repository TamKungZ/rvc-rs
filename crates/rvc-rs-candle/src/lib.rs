#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Candle backend and the integration boundary between `pthrs` tensors and RVC.

use candle_core::{Device, Tensor};
use rvc_rs_core::{ComputeDevice, GeneratorInput, InputError, ModelSpec, VoiceGenerator};
use thiserror::Error;

/// Re-exported checkpoint crate used by the weight adapter implementation.
pub use pthrs as checkpoint;

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
}

impl CandleGenerator {
    /// Creates an uninitialized generator shell after validating its model spec.
    pub fn uninitialized(
        spec: ModelSpec,
        request: ComputeDevice,
    ) -> Result<Self, InferenceError> {
        spec.validate()?;
        Ok(Self {
            spec,
            device: resolve_device(request)?,
        })
    }

    /// Returns the selected Candle device.
    pub const fn device(&self) -> &Device {
        &self.device
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
    /// The RVC architecture has not been implemented in this scaffold yet.
    #[error("RVC generator forward pass is not implemented; see docs/ROADMAP.md")]
    ModelNotImplemented,
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

