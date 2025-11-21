// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>

#include "kpack_internal.h"

namespace kpack {

kpack_error_t decompress_noop(kpack_archive* archive, uint32_t ordinal,
                              uint64_t expected_size) {
  if (ordinal >= archive->blobs.size()) {
    return KPACK_ERROR_KERNEL_NOT_FOUND;
  }

  const BlobInfo& blob = archive->blobs[ordinal];

  // Allocate kernel cache
  archive->kernel_cache.resize(blob.size);

  // Seek to blob offset
  if (fseek(archive->file.get(), blob.offset, SEEK_SET) != 0) {
    return KPACK_ERROR_IO_ERROR;
  }

  // Read kernel data
  size_t read =
      fread(archive->kernel_cache.data(), 1, blob.size, archive->file.get());
  if (read != blob.size) {
    return KPACK_ERROR_IO_ERROR;
  }

  return KPACK_SUCCESS;
}

kpack_error_t build_zstd_frame_index(kpack_archive* archive) {
  // POC: Cache entire blob in memory
  archive->zstd_blob.resize(archive->zstd_size);

  // Seek to blob start
  if (fseek(archive->file.get(), archive->zstd_offset, SEEK_SET) != 0) {
    return KPACK_ERROR_IO_ERROR;
  }

  // Read blob
  size_t read = fread(archive->zstd_blob.data(), 1, archive->zstd_size,
                      archive->file.get());
  if (read != archive->zstd_size) {
    return KPACK_ERROR_IO_ERROR;
  }

  // Parse blob header
  const uint8_t* ptr = archive->zstd_blob.data();
  uint32_t num_kernels;
  memcpy(&num_kernels, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);

  uint64_t offset = sizeof(uint32_t);  // Start after num_kernels header
  archive->zstd_frames.reserve(num_kernels);

  // Parse frame headers
  for (uint32_t i = 0; i < num_kernels; ++i) {
    uint32_t frame_size;
    memcpy(&frame_size, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    offset += sizeof(uint32_t);

    FrameInfo frame;
    frame.offset_in_blob = offset;
    frame.compressed_size = frame_size;
    archive->zstd_frames.push_back(frame);

    ptr += frame_size;
    offset += frame_size;
  }

  // Create decompression context
  archive->zstd_ctx.reset(ZSTD_createDCtx());
  if (!archive->zstd_ctx) {
    return KPACK_ERROR_OUT_OF_MEMORY;
  }

  return KPACK_SUCCESS;
}

kpack_error_t decompress_zstd(kpack_archive* archive, uint32_t ordinal,
                              uint64_t expected_size) {
  if (ordinal >= archive->zstd_frames.size()) {
    return KPACK_ERROR_KERNEL_NOT_FOUND;
  }

  const FrameInfo& frame = archive->zstd_frames[ordinal];

  // Get compressed frame from cached blob
  const void* compressed = archive->zstd_blob.data() + frame.offset_in_blob;
  size_t compressed_size = frame.compressed_size;

  // Allocate decompression buffer
  archive->kernel_cache.resize(expected_size);

  // Decompress
  size_t result =
      ZSTD_decompressDCtx(archive->zstd_ctx.get(), archive->kernel_cache.data(),
                          expected_size, compressed, compressed_size);

  if (ZSTD_isError(result)) {
    return KPACK_ERROR_DECOMPRESSION_FAILED;
  }

  if (result != expected_size) {
    return KPACK_ERROR_DECOMPRESSION_FAILED;
  }

  return KPACK_SUCCESS;
}

}  // namespace kpack
