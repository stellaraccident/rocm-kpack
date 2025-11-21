// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCM_KPACK_H
#define ROCM_KPACK_H

#include "kpack_export.h"
#include "kpack_types.h"

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
// All query functions return pointers to memory owned by the archive.
// Returned pointers are valid until kpack_close() is called.
//

// Get number of architectures available in the archive
//
// Args:
//   archive: Archive handle
//   count: Output parameter for number of architectures
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if any pointer is NULL
KPACK_API kpack_error_t kpack_get_architecture_count(kpack_archive_t archive,
                                                     size_t* count);

// Get architecture name by index
//
// Returns a pointer to an architecture string owned by the archive.
// The pointer is valid until kpack_close() is called.
//
// Args:
//   archive: Archive handle
//   index: Index in range [0, count)
//   arch: Output parameter for architecture string
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if pointers are NULL or index out of range
KPACK_API kpack_error_t kpack_get_architecture(kpack_archive_t archive,
                                               size_t index, const char** arch);

// Get number of binaries that have kernels in the archive
//
// Args:
//   archive: Archive handle
//   count: Output parameter for number of binaries
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if any pointer is NULL
KPACK_API kpack_error_t kpack_get_binary_count(kpack_archive_t archive,
                                               size_t* count);

// Get binary name by index
//
// Returns a pointer to a binary path string owned by the archive.
// The pointer is valid until kpack_close() is called.
//
// Args:
//   archive: Archive handle
//   index: Index in range [0, count)
//   binary: Output parameter for binary path string
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if pointers are NULL or index out of range
KPACK_API kpack_error_t kpack_get_binary(kpack_archive_t archive, size_t index,
                                         const char** binary);

//
// Kernel Loading
//

// Load kernel for specific binary and architecture
//
// Decompresses and returns a pointer to the kernel data for the specified
// binary and architecture. The pointer points to memory owned by the archive.
//
// Memory Management:
// - The returned pointer is valid until the next call to kpack_get_kernel()
//   or kpack_close()
// - The archive maintains a single kernel cache that is overwritten on each
// call
// - For concurrent access, open separate archive handles per thread
//
// Thread Safety:
// - Multiple threads can call this function on DIFFERENT archive handles
// - NOT safe to call concurrently on the SAME archive handle
//
// Args:
//   archive: Archive handle
//   binary_name: Binary path (e.g., "lib/libamdhip64.so.6")
//   arch: Architecture name (e.g., "gfx1100")
//   kernel_data: Output parameter for const pointer to kernel bytes
//   kernel_size: Output parameter for kernel size in bytes
//
// Returns:
//   KPACK_SUCCESS on success
//   KPACK_ERROR_INVALID_ARGUMENT if any pointer is NULL
//   KPACK_ERROR_KERNEL_NOT_FOUND if kernel doesn't exist
//   KPACK_ERROR_DECOMPRESSION_FAILED if decompression fails
//   KPACK_ERROR_OUT_OF_MEMORY if allocation fails
//   KPACK_ERROR_IO_ERROR if file read fails
KPACK_API kpack_error_t kpack_get_kernel(kpack_archive_t archive,
                                         const char* binary_name,
                                         const char* arch,
                                         const void** kernel_data,
                                         size_t* kernel_size);

#ifdef __cplusplus
}
#endif

#endif  // ROCM_KPACK_H
