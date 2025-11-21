// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCM_KPACK_H
#define ROCM_KPACK_H

#include "kpack_types.h"
#include "kpack_export.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Archive Lifecycle
//

// Open a kpack archive for reading
//
// Opens the kpack file at the specified path, validates the header,
// and parses the table of contents (TOC).
//
// Args:
//   path: Path to the .kpack file
//   archive: Output parameter for archive handle
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if path is NULL or archive is NULL
//   KPACK_ERROR_FILE_NOT_FOUND if file doesn't exist
//   KPACK_ERROR_INVALID_FORMAT if file is not a valid kpack
//   KPACK_ERROR_UNSUPPORTED_VERSION if version is not supported
//   KPACK_ERROR_OUT_OF_MEMORY if allocation fails
//   KPACK_ERROR_IO_ERROR on read failures
//   KPACK_ERROR_MSGPACK_PARSE_FAILED if TOC parsing fails
KPACK_API kpack_error_t kpack_open(const char* path, kpack_archive_t* archive);

// Close archive and free all resources
//
// Must not be called while other threads are using the archive.
// After this call, the archive handle is invalid.
//
// Args:
//   archive: Archive handle from kpack_open()
KPACK_API void kpack_close(kpack_archive_t archive);

//
// Discovery and Querying
//

// Get list of architectures available in the archive
//
// Returns a NULL-terminated array of architecture strings.
// Caller must free with kpack_free_string_array().
//
// Args:
//   archive: Archive handle
//   arches: Output parameter for architecture array
//   count: Output parameter for number of architectures
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if any pointer is NULL
//   KPACK_ERROR_OUT_OF_MEMORY if allocation fails
KPACK_API kpack_error_t kpack_get_architectures(
    kpack_archive_t archive,
    char*** arches,
    size_t* count
);

// Get list of binary names that have kernels in the archive
//
// Returns a NULL-terminated array of binary path strings.
// Caller must free with kpack_free_string_array().
//
// Args:
//   archive: Archive handle
//   binaries: Output parameter for binary array
//   count: Output parameter for number of binaries
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if any pointer is NULL
//   KPACK_ERROR_OUT_OF_MEMORY if allocation fails
KPACK_API kpack_error_t kpack_get_binaries(
    kpack_archive_t archive,
    char*** binaries,
    size_t* count
);

// Free string array returned by query functions
//
// Args:
//   array: Array returned by kpack_get_architectures() or kpack_get_binaries()
//   count: Number of strings in array
KPACK_API void kpack_free_string_array(char** array, size_t count);

//
// Kernel Loading
//

// Load kernel for specific binary and architecture
//
// Decompresses and returns the kernel data for the specified binary
// and architecture. Caller must free with kpack_free_kernel().
//
// Args:
//   archive: Archive handle
//   binary_name: Binary path (e.g., "lib/libamdhip64.so.6")
//   arch: Architecture name (e.g., "gfx1100")
//   kernel_data: Output parameter for kernel bytes
//   kernel_size: Output parameter for kernel size in bytes
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if any pointer is NULL
//   KPACK_ERROR_KERNEL_NOT_FOUND if kernel doesn't exist
//   KPACK_ERROR_DECOMPRESSION_FAILED if decompression fails
//   KPACK_ERROR_OUT_OF_MEMORY if allocation fails
KPACK_API kpack_error_t kpack_get_kernel(
    kpack_archive_t archive,
    const char* binary_name,
    const char* arch,
    void** kernel_data,
    size_t* kernel_size
);

// Free kernel data returned by kpack_get_kernel()
//
// Args:
//   archive: Archive handle
//   kernel_data: Pointer returned by kpack_get_kernel()
KPACK_API void kpack_free_kernel(kpack_archive_t archive, void* kernel_data);

#ifdef __cplusplus
}
#endif

#endif // ROCM_KPACK_H
