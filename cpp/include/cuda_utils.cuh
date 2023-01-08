// Copyright 2023 Zhihao Liang
#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "utils.h"

#define CUDA_GET_THREAD_ID(tid, Q)                             \
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (tid >= Q) return
#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)

#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CUDA_CHECK_THROW(x)                                            \
    do {                                                               \
        cudaError_t result = x;                                        \
        if (result != cudaSuccess)                                     \
            throw std::runtime_error(                                  \
                std::string(FILE_LINE " " #x " failed with error: ") + \
                cudaGetErrorString(result));                           \
    } while (0)

