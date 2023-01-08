// Copyright 2022 Gorilla-Lab
#include <torch/extension.h>

namespace GVolume {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "VolumeAcc: Volume Rendering Acceleration ToolBox";
}

}  // namespace GVolume
