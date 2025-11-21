// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "rocm_kpack/kpack.h"

// Test that library links and basic error handling works
TEST(KpackAPITest, NullArguments) {
  kpack_error_t err;

  // kpack_open with NULL path
  kpack_archive_t archive;
  err = kpack_open(nullptr, &archive);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  // kpack_open with NULL archive
  err = kpack_open("/tmp/test.kpack", nullptr);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  // kpack_get_architecture_count with NULL archive
  size_t count;
  err = kpack_get_architecture_count(nullptr, &count);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);

  // kpack_get_kernel with NULL arguments
  const void* data;
  size_t size;
  err = kpack_get_kernel(nullptr, "test", "gfx1100", &data, &size);
  EXPECT_EQ(err, KPACK_ERROR_INVALID_ARGUMENT);
}

// Test file not found
TEST(KpackAPITest, FileNotFound) {
  kpack_archive_t archive;
  kpack_error_t err = kpack_open("/nonexistent/test.kpack", &archive);
  EXPECT_EQ(err, KPACK_ERROR_FILE_NOT_FOUND);
}

// Test kpack_close with NULL (should not crash)
TEST(KpackAPITest, CloseNull) {
  kpack_close(nullptr);
  // Should not crash
}
