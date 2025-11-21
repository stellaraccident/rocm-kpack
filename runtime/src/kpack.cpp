// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_kpack/kpack.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

//
// API Implementation (stubs)
//

kpack_error_t kpack_open(const char* path, kpack_archive_t* archive) {
    if (!path || !archive) {
        return KPACK_ERROR_INVALID_ARGUMENT;
    }

    // TODO: Implement archive opening
    return KPACK_ERROR_NOT_IMPLEMENTED;
}

void kpack_close(kpack_archive_t archive) {
    if (!archive) {
        return;
    }

    // TODO: Implement cleanup
}

kpack_error_t kpack_get_architectures(
    kpack_archive_t archive,
    char*** arches,
    size_t* count
) {
    if (!archive || !arches || !count) {
        return KPACK_ERROR_INVALID_ARGUMENT;
    }

    // TODO: Implement architecture querying
    return KPACK_ERROR_NOT_IMPLEMENTED;
}

kpack_error_t kpack_get_binaries(
    kpack_archive_t archive,
    char*** binaries,
    size_t* count
) {
    if (!archive || !binaries || !count) {
        return KPACK_ERROR_INVALID_ARGUMENT;
    }

    // TODO: Implement binary querying
    return KPACK_ERROR_NOT_IMPLEMENTED;
}

void kpack_free_string_array(char** array, size_t count) {
    if (!array) {
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        free(array[i]);
    }
    free(array);
}

kpack_error_t kpack_get_kernel(
    kpack_archive_t archive,
    const char* binary_name,
    const char* arch,
    void** kernel_data,
    size_t* kernel_size
) {
    if (!archive || !binary_name || !arch || !kernel_data || !kernel_size) {
        return KPACK_ERROR_INVALID_ARGUMENT;
    }

    // TODO: Implement kernel loading
    return KPACK_ERROR_NOT_IMPLEMENTED;
}

void kpack_free_kernel(kpack_archive_t archive, void* kernel_data) {
    if (!kernel_data) {
        return;
    }

    // TODO: Implement kernel freeing
    free(kernel_data);
}
