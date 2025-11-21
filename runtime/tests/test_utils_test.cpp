// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_utils.h"

#include <gtest/gtest.h>
#include <filesystem>

// Test that we can access test assets directory
TEST(TestUtilsTest, TestAssetsAccess) {
    // Verify test assets directory exists and is accessible
    EXPECT_NO_THROW({
        auto assets_dir = test_utils::get_test_assets_dir();
        EXPECT_TRUE(std::filesystem::exists(assets_dir));
        EXPECT_TRUE(std::filesystem::is_directory(assets_dir));
    });

    // Verify we can get paths to bundled binaries
    EXPECT_NO_THROW({
        auto test_exe = test_utils::get_test_asset("bundled_binaries/linux/cov5/test_kernel_single.exe");
        EXPECT_TRUE(std::filesystem::is_regular_file(test_exe));
    });

    // Verify we can get paths to code objects
    EXPECT_NO_THROW({
        auto co_file = test_utils::get_test_asset("ccob/ccob_gfx942_sample1.co");
        EXPECT_TRUE(std::filesystem::is_regular_file(co_file));
    });
}
