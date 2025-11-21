import shutil
import subprocess
from pathlib import Path

from rocm_kpack import binutils


def test_toolchain(test_assets_dir: Path, toolchain: binutils.Toolchain):
    bb = binutils.BundledBinary(
        test_assets_dir / "ccob" / "ccob_gfx942_sample1.co", toolchain=toolchain
    )
    with bb.unbundle() as contents:
        for target, filename in contents.target_list:
            if filename.endswith(".hsaco"):
                assert "gfx942" in target
                assert (contents.dest_dir / filename).exists()
                break
        else:
            raise AssertionError("No target hsaco file")


def test_kpack_ref_marker_roundtrip(
    tmp_path: Path, toolchain: binutils.Toolchain, test_assets_dir: Path
):
    """Test adding and reading back a kpack ref marker."""
    # Use a real bundled binary from test assets
    source_binary = (
        test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    )
    marked_binary = tmp_path / "marked_binary.exe"

    # Add marker
    kpack_paths = [".kpack/blas-gfx100X.kpack", ".kpack/torch-gfx100X.kpack"]
    kernel_name = "bin/test_kernel_multi.exe"

    binutils.add_kpack_ref_marker(
        binary_path=source_binary,
        output_path=marked_binary,
        kpack_search_paths=kpack_paths,
        kernel_name=kernel_name,
        toolchain=toolchain,
    )

    # Verify marked binary exists and is executable
    assert marked_binary.exists()
    assert marked_binary.stat().st_size > 0

    # Read marker back
    marker_data = binutils.read_kpack_ref_marker(marked_binary, toolchain=toolchain)
    assert marker_data is not None
    assert marker_data["kpack_search_paths"] == kpack_paths
    assert marker_data["kernel_name"] == kernel_name


def test_kpack_ref_marker_with_host_only_binary(
    tmp_path: Path, toolchain: binutils.Toolchain, test_assets_dir: Path
):
    """Test adding marker to host-only binary (no .hip_fatbin section)."""
    source_binary = test_assets_dir / "bundled_binaries/linux/cov5/host_only.exe"
    marked_binary = tmp_path / "marked_host_only.exe"

    # Add marker to host-only binary
    binutils.add_kpack_ref_marker(
        binary_path=source_binary,
        output_path=marked_binary,
        kpack_search_paths=[".kpack/runtime-gfx1100.kpack"],
        kernel_name="bin/host_only.exe",
        toolchain=toolchain,
    )

    # Read back and verify
    marker_data = binutils.read_kpack_ref_marker(marked_binary, toolchain=toolchain)
    assert marker_data is not None
    assert marker_data["kpack_search_paths"] == [".kpack/runtime-gfx1100.kpack"]
    assert marker_data["kernel_name"] == "bin/host_only.exe"


def test_kpack_ref_marker_with_shared_library(
    tmp_path: Path, toolchain: binutils.Toolchain, test_assets_dir: Path
):
    """Test adding marker to shared library."""
    source_binary = (
        test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_single.so"
    )
    marked_binary = tmp_path / "marked_library.so"

    # Add marker
    binutils.add_kpack_ref_marker(
        binary_path=source_binary,
        output_path=marked_binary,
        kpack_search_paths=[".kpack/hipblas-gfx1100.kpack"],
        kernel_name="lib/libtest_kernel_single.so",
        toolchain=toolchain,
    )

    # Verify
    marker_data = binutils.read_kpack_ref_marker(marked_binary, toolchain=toolchain)
    assert marker_data is not None
    assert marker_data["kernel_name"] == "lib/libtest_kernel_single.so"


def test_kpack_ref_marker_multiple_search_paths(
    tmp_path: Path, toolchain: binutils.Toolchain, test_assets_dir: Path
):
    """Test marker with multiple kpack search paths."""
    source_binary = (
        test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    )
    marked_binary = tmp_path / "multi_search.exe"

    # Multiple search paths for different architecture families
    kpack_paths = [
        ".kpack/blas-gfx1100.kpack",
        ".kpack/blas-gfx100X.kpack",
        ".kpack/blas-gfx90a.kpack",
    ]

    binutils.add_kpack_ref_marker(
        binary_path=source_binary,
        output_path=marked_binary,
        kpack_search_paths=kpack_paths,
        kernel_name="bin/hipcc",
        toolchain=toolchain,
    )

    marker_data = binutils.read_kpack_ref_marker(marked_binary, toolchain=toolchain)
    assert marker_data is not None
    assert len(marker_data["kpack_search_paths"]) == 3
    assert marker_data["kpack_search_paths"] == kpack_paths


def test_read_kpack_ref_marker_no_section(
    tmp_path: Path, toolchain: binutils.Toolchain, test_assets_dir: Path
):
    """Test reading marker from binary without .rocm_kpack_ref section."""
    # Use unmarked binary
    source_binary = (
        test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    )

    # Should return None for binaries without marker
    marker_data = binutils.read_kpack_ref_marker(source_binary, toolchain=toolchain)
    assert marker_data is None


def test_kpack_ref_marker_preserves_sections(
    tmp_path: Path, toolchain: binutils.Toolchain, test_assets_dir: Path
):
    """Test that adding marker preserves existing sections like .hip_fatbin."""
    source_binary = (
        test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    )
    marked_binary = tmp_path / "preserve_sections.exe"

    # Verify source has .hip_fatbin section
    result = subprocess.run(
        [str(toolchain.readelf), "-S", str(source_binary)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ".hip_fatbin" in result.stdout

    # Add marker
    binutils.add_kpack_ref_marker(
        binary_path=source_binary,
        output_path=marked_binary,
        kpack_search_paths=[".kpack/test.kpack"],
        kernel_name="bin/test",
        toolchain=toolchain,
    )

    # Verify marked binary still has .hip_fatbin section
    result = subprocess.run(
        [str(toolchain.readelf), "-S", str(marked_binary)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        ".hip_fatbin" in result.stdout
    ), "Original .hip_fatbin section should be preserved"
    assert (
        ".rocm_kpack_ref" in result.stdout
    ), "New .rocm_kpack_ref section should exist"


def test_kpack_ref_marker_special_characters_in_paths(
    tmp_path: Path, toolchain: binutils.Toolchain, test_assets_dir: Path
):
    """Test marker with special characters in paths."""
    source_binary = (
        test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    )
    marked_binary = tmp_path / "special_chars.exe"

    # Paths with special characters
    kpack_paths = [
        ".kpack/my-lib_v1.0.kpack",
        "../shared/.kpack/common-gfx100X.kpack",
    ]

    binutils.add_kpack_ref_marker(
        binary_path=source_binary,
        output_path=marked_binary,
        kpack_search_paths=kpack_paths,
        kernel_name="bin/my-binary_v2.exe",
        toolchain=toolchain,
    )

    marker_data = binutils.read_kpack_ref_marker(marked_binary, toolchain=toolchain)
    assert marker_data is not None
    assert marker_data["kpack_search_paths"] == kpack_paths
    assert marker_data["kernel_name"] == "bin/my-binary_v2.exe"


def test_kpack_ref_marker_overwrite_output(
    tmp_path: Path, toolchain: binutils.Toolchain, test_assets_dir: Path
):
    """Test that output path can be the same as input (in-place modification)."""
    # Copy source to temp location first
    source_binary = (
        test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    )
    test_binary = tmp_path / "test_inplace.exe"
    shutil.copy2(source_binary, test_binary)

    # Verify no marker initially
    assert binutils.read_kpack_ref_marker(test_binary, toolchain=toolchain) is None

    # Add marker in-place (output_path == input path)
    binutils.add_kpack_ref_marker(
        binary_path=test_binary,
        output_path=test_binary,
        kpack_search_paths=[".kpack/test.kpack"],
        kernel_name="bin/test",
        toolchain=toolchain,
    )

    # Verify marker was added
    marker_data = binutils.read_kpack_ref_marker(test_binary, toolchain=toolchain)
    assert marker_data is not None
    assert marker_data["kernel_name"] == "bin/test"
