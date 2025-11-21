// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_kpack/kpack.h"

#include <zstd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Decompression functions will be implemented here

// TODO: Implement NoOp decompression (just read from offset)
// TODO: Implement Zstd per-kernel decompression
// TODO: Build frame index for Zstd archives
