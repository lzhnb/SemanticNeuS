#pragma once
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

using torch::Tensor;

namespace VolumeAcc {
Tensor spherical_harmonic_forward(const Tensor& dirs, const uint8_t sh_degree);
Tensor spherical_harmonic_backward(
    const Tensor& outputs_grad, const Tensor& dirs, const uint8_t sh_degree);
}  // namespace VolumeAcc