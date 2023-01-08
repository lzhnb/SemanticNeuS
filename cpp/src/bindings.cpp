// Copyright 2023 Zhihao Liang
#include <torch/extension.h>

#include "spherical_harmonics.cuh"

namespace VolumeAcc {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "VolumeAcc: Volume Rendering Acceleration ToolBox";

    // spherical harmonics encoding
    m.def("spherical_harmonic_forward", &spherical_harmonic_forward);
    m.def("spherical_harmonic_backward", &spherical_harmonic_backward);
}

}  // namespace VolumeAcc
