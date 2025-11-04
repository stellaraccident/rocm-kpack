"""Tests for ELF fat device neutralizer."""

import os
import subprocess
from pathlib import Path

from rocm_kpack.elf_fat_device_neutralizer import neutralize_binary, ElfFatDeviceNeutralizer
from rocm_kpack import binutils


def test_neutralize_fat_binary(tmp_path: Path, test_assets_dir: Path, toolchain: binutils.Toolchain):
    """Test neutralizing a fat binary with .hip_fatbin section."""
    # Use a multi-arch bundled binary
    input_binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    output_binary = tmp_path / "test_kernel_multi_neutralized.exe"

    # Get original size
    original_size = input_binary.stat().st_size

    # Neutralize
    result = neutralize_binary(input_binary, output_binary, verbose=True)

    # Verify output exists
    assert output_binary.exists()

    # Verify size reduction
    new_size = output_binary.stat().st_size
    assert new_size < original_size
    assert result["removed"] > 0
    assert result["original_size"] == original_size
    assert result["new_size"] == new_size

    # Verify output is a valid ELF file
    readelf_result = subprocess.run(
        [toolchain.readelf, "-h", output_binary],
        capture_output=True,
        text=True
    )
    assert readelf_result.returncode == 0
    assert "ELF" in readelf_result.stdout

    # Verify .hip_fatbin section is gone (should be SHT_NULL or not listed)
    sections_result = subprocess.run(
        [toolchain.readelf, "-S", output_binary],
        capture_output=True,
        text=True
    )
    assert sections_result.returncode == 0
    # Either the section is marked as NULL or doesn't appear with size > 0
    if ".hip_fatbin" in sections_result.stdout:
        # If it appears, it should be NULL type
        assert "NULL" in sections_result.stdout

    # Verify executable permissions are preserved
    assert os.access(output_binary, os.X_OK), "Output binary should be executable"


def test_neutralize_shared_library(tmp_path: Path, test_assets_dir: Path, toolchain: binutils.Toolchain):
    """Test neutralizing a shared library with .hip_fatbin section."""
    input_library = test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_single.so"
    output_library = tmp_path / "libtest_kernel_single_neutralized.so"

    original_size = input_library.stat().st_size

    result = neutralize_binary(input_library, output_library, verbose=True)

    # Verify size reduction
    assert output_library.exists()
    assert output_library.stat().st_size < original_size
    assert result["removed"] > 0

    # Verify it's still a valid shared library
    readelf_result = subprocess.run(
        [toolchain.readelf, "-h", output_library],
        capture_output=True,
        text=True
    )
    assert readelf_result.returncode == 0
    assert "DYN (Shared object file)" in readelf_result.stdout


def test_neutralize_host_only_binary(tmp_path: Path, test_assets_dir: Path):
    """Test neutralizing a binary without .hip_fatbin (should just copy)."""
    input_binary = test_assets_dir / "bundled_binaries/linux/cov5/host_only.exe"
    output_binary = tmp_path / "host_only_neutralized.exe"

    original_size = input_binary.stat().st_size

    result = neutralize_binary(input_binary, output_binary, verbose=True)

    # Should just copy the file
    assert output_binary.exists()
    assert output_binary.stat().st_size == original_size
    assert result["removed"] == 0
    assert result["original_size"] == original_size
    assert result["new_size"] == original_size


def test_neutralizer_has_hip_fatbin(test_assets_dir: Path):
    """Test the has_hip_fatbin() method."""
    # Fat binary should have .hip_fatbin
    fat_binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    neutralizer = ElfFatDeviceNeutralizer(fat_binary)
    assert neutralizer.has_hip_fatbin() is True

    # Host-only binary should not have .hip_fatbin
    host_only = test_assets_dir / "bundled_binaries/linux/cov5/host_only.exe"
    neutralizer = ElfFatDeviceNeutralizer(host_only)
    assert neutralizer.has_hip_fatbin() is False


def test_neutralizer_calculate_removal_plan(test_assets_dir: Path):
    """Test the calculate_removal_plan() method."""
    fat_binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    neutralizer = ElfFatDeviceNeutralizer(fat_binary)

    plan = neutralizer.calculate_removal_plan()

    # Verify plan structure
    assert "removal_size" in plan
    assert "removal_offset" in plan
    assert "removal_vaddr" in plan
    assert "sections_to_shift" in plan
    assert "phdrs_to_update" in plan

    # Verify plan makes sense
    assert plan["removal_size"] > 0
    assert plan["removal_offset"] > 0
    assert isinstance(plan["sections_to_shift"], list)
    assert isinstance(plan["phdrs_to_update"], list)


def test_integration_with_binutils_create_host_only(tmp_path: Path, test_assets_dir: Path, toolchain: binutils.Toolchain):
    """Test that BundledBinary.create_host_only() uses the neutralizer by default."""
    input_binary = test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_multi.so"
    output_binary = tmp_path / "libtest_kernel_multi_host_only.so"

    original_size = input_binary.stat().st_size

    # Create BundledBinary and call create_host_only (should use neutralizer by default)
    bb = binutils.BundledBinary(input_binary, toolchain=toolchain)
    bb.create_host_only(output_binary, use_objcopy=False)

    # Verify size reduction (neutralizer should actually reduce size)
    assert output_binary.exists()
    new_size = output_binary.stat().st_size
    assert new_size < original_size

    # Compare with objcopy path (should have minimal reduction)
    output_binary_objcopy = tmp_path / "libtest_kernel_multi_objcopy.so"
    bb.create_host_only(output_binary_objcopy, use_objcopy=True)

    objcopy_size = output_binary_objcopy.stat().st_size
    # Neutralizer should remove more bytes than objcopy
    neutralizer_reduction = original_size - new_size
    objcopy_reduction = original_size - objcopy_size
    assert neutralizer_reduction > objcopy_reduction


def test_neutralized_binary_compatible_with_objcopy(tmp_path: Path, test_assets_dir: Path, toolchain: binutils.Toolchain):
    """Test that objcopy --add-section works on neutralized binaries."""
    input_binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    output_binary = tmp_path / "test_kernel_multi_neutralized.exe"

    # Neutralize the binary
    neutralize_binary(input_binary, output_binary, verbose=True)

    # Verify neutralized binary exists
    assert output_binary.exists()

    # Create a dummy section content
    marker_file = tmp_path / "test_marker.bin"
    marker_file.write_text("test marker content")

    # Try to add a section using objcopy (this should work on valid ELF files)
    output_with_marker = tmp_path / "test_kernel_multi_with_marker.exe"
    try:
        subprocess.run(
            [
                toolchain.objcopy,
                "--add-section",
                f".test_marker={marker_file}",
                output_binary,
                output_with_marker,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise AssertionError(
            f"objcopy --add-section failed on neutralized binary. "
            f"This indicates the neutralizer produced an invalid ELF file. "
            f"Error: {e.stderr}"
        )

    # Verify the marker was added
    readelf_result = subprocess.run(
        [toolchain.readelf, "-S", output_with_marker],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ".test_marker" in readelf_result.stdout, "Marker section should be present in binary"

    # Verify the binary is still valid
    assert output_with_marker.exists()
    assert output_with_marker.stat().st_size > output_binary.stat().st_size
