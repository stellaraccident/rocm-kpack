// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_kpack/kpack.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>

// Internal archive structure (opaque to user)
struct kpack_archive {
    std::FILE* file;
    std::string file_path;
    uint32_t version;
    uint64_t toc_offset;

    // TOC data (parsed from MessagePack)
    std::string group_name;
    std::string gfx_arch_family;
    std::vector<std::string> gfx_arches;
    kpack_compression_scheme_t compression_scheme;

    // Compression metadata
    uint64_t blob_offset;
    uint64_t blob_size;

    // Per-binary, per-arch kernel metadata
    // Map: binary_name -> arch -> kernel_metadata
    struct KernelMetadata {
        std::string type;  // "hsaco"
        uint32_t ordinal;
        uint64_t original_size;
    };
    std::map<std::string, std::map<std::string, KernelMetadata>> toc;

    kpack_archive() : file(nullptr), version(0), toc_offset(0),
                      compression_scheme(KPACK_COMPRESSION_NOOP),
                      blob_offset(0), blob_size(0) {}

    ~kpack_archive() {
        if (file) {
            std::fclose(file);
            file = nullptr;
        }
    }
};

// Archive opening will be implemented here
// TODO: Implement in next phase
