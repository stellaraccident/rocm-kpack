#!/usr/bin/env python3
"""
Tests for ELF modification tool (elf_modify_load.py).

This module tests:
1. Zero-page optimization (conservative algorithm for NOBITS conversion)
2. Section mapping to new PT_LOAD segments (with auto-allocation)
3. Pointer setting with relocation updates
4. PIE/shared library relocation requirements
5. Complete workflows combining map-section + set-pointer

Uses C test binaries that validate different scenarios.
"""

import subprocess
import pytest
from pathlib import Path

from rocm_kpack.elf_modify_load import main
from rocm_kpack.binutils import get_section_vaddr


# Test directory containing C sources and built binaries
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent.parent


def is_section_in_pt_load(binary_path: Path, section_name: str) -> bool:
    """
    Check if a section is mapped to memory via a PT_LOAD segment.

    Args:
        binary_path: Path to ELF binary
        section_name: Name of section to check (e.g., ".custom_data")

    Returns:
        True if section is in a PT_LOAD segment, False otherwise
    """
    # Get section virtual address and size
    result = subprocess.run(
        ["readelf", "-S", str(binary_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return False

    section_vaddr = None
    section_size = None
    section_flags = None

    # Section headers span two lines:
    # Line 1: [Nr] Name Type Address Offset
    # Line 2:      Size EntSize Flags Link Info Align
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        if section_name in line:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    section_vaddr = int(parts[3], 16)  # Address column
                    # Next line has size and flags
                    if i + 1 < len(lines):
                        next_parts = lines[i + 1].split()
                        if len(next_parts) >= 3:
                            section_size = int(next_parts[0], 16)  # Size
                            section_flags = next_parts[2]  # Flags column
                    break
                except (ValueError, IndexError):
                    continue

    if section_vaddr is None:
        return False  # Section doesn't exist

    # Non-ALLOC sections are never in PT_LOAD, even if vaddr overlaps
    if section_flags and 'A' not in section_flags:
        return False

    # Get PT_LOAD segments
    result = subprocess.run(
        ["readelf", "-l", str(binary_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return False

    # Parse PT_LOAD segments and check if section falls within any
    # Note: readelf output spans multiple lines:
    #   LOAD           0xoffset 0xvaddr 0xpaddr
    #                  0xfilesz 0xmemsz flags align
    lines = result.stdout.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('LOAD'):
            # First line has: LOAD offset vaddr paddr
            parts1 = line.split()
            if len(parts1) >= 3:
                try:
                    load_vaddr = int(parts1[2], 16)  # Virtual address

                    # Next line has: filesz memsz flags align
                    if i + 1 < len(lines):
                        parts2 = lines[i + 1].split()
                        if len(parts2) >= 2:
                            load_memsz = int(parts2[1], 16)  # Memory size
                            load_end = load_vaddr + load_memsz

                            # Check if section is within this PT_LOAD
                            section_end = section_vaddr + section_size
                            if (load_vaddr <= section_vaddr < load_end or
                                load_vaddr < section_end <= load_end):
                                return True
                except (ValueError, IndexError):
                    pass
        i += 1

    return False


@pytest.fixture(scope="module")
def build_test_binaries():
    """Build all C test binaries before running tests."""
    test_cases = [
        "test_zero_page_aligned",
        "test_zero_page_unaligned_start",
        "test_zero_page_unaligned_size",
        "test_zero_page_both_unaligned",
    ]

    # Build each test binary
    for test_name in test_cases:
        source = TEST_DIR / f"{test_name}.c"
        output = TEST_DIR / test_name

        # Compile with page-aligned section
        result = subprocess.run(
            [
                "gcc",
                "-O0",
                "-g",
                "-o", str(output),
                str(source),
                "-Wl,--section-start=.testdata=0x10000",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.fail(f"Failed to build {test_name}: {result.stderr}")

    yield test_cases

    # Cleanup built binaries and zeroed versions
    for test_name in test_cases:
        (TEST_DIR / test_name).unlink(missing_ok=True)
        (TEST_DIR / f"{test_name}.zeroed").unlink(missing_ok=True)


def run_binary(binary_path: Path) -> tuple[int, str, str]:
    """Run a binary and return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        [str(binary_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def apply_zero_page(input_path: Path, output_path: Path) -> tuple[int, str]:
    """
    Apply zero-page optimization.

    Returns: (exit_code, output)
    """
    import io
    import sys

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = output_buffer = io.StringIO()

    try:
        exit_code = main([
            'zero-page',
            str(input_path),
            str(output_path),
            '--section=.testdata'
        ])
        output = output_buffer.getvalue()
    finally:
        sys.stdout = old_stdout

    return exit_code, output


def test_tool_exists():
    """Verify the elf_modify_load main function can be imported."""
    # If we got here, import succeeded
    assert main is not None


def test_aligned_case(build_test_binaries):
    """
    Test zero-page optimization with fully aligned section.

    Expected: Entire section should be zero-paged.
    """
    test_name = "test_zero_page_aligned"
    input_bin = TEST_DIR / test_name
    output_bin = TEST_DIR / f"{test_name}.zeroed"

    # Verify original binary works
    exit_code, stdout, stderr = run_binary(input_bin)
    assert exit_code != 0, "Original binary should fail (no zero-paging yet)"

    # Apply zero-page optimization
    exit_code, output = apply_zero_page(input_bin, output_bin)
    assert exit_code == 0, f"Zero-page tool failed: {output}"
    assert output_bin.exists(), "Output binary not created"

    # Verify output binary works and detects zero-paging
    exit_code, stdout, stderr = run_binary(output_bin)
    assert exit_code == 0, f"Zeroed binary failed: {stdout}\n{stderr}"
    assert "SUCCESS" in stdout, "Binary should report success"
    assert "zero-paged" in stdout.lower(), "Should confirm zero-paging worked"


def test_unaligned_size_case(build_test_binaries):
    """
    Test zero-page optimization with page-aligned start but unaligned size.

    This is the critical case for .hip_fatbin which has partial pages at the end.
    Expected: Full pages zero-paged, partial page preserved.
    """
    test_name = "test_zero_page_unaligned_size"
    input_bin = TEST_DIR / test_name
    output_bin = TEST_DIR / f"{test_name}.zeroed"

    # Apply zero-page optimization
    exit_code, output = apply_zero_page(input_bin, output_bin)
    assert exit_code == 0, f"Zero-page tool failed: {output}"

    # Check that it reports saving space but keeping partial page
    assert "Pages to zero: 2" in output, "Should zero 2 full pages"
    assert "Suffix kept:" in output, "Should preserve partial page at end"

    # Verify output binary works
    exit_code, stdout, stderr = run_binary(output_bin)
    assert exit_code == 0, f"Zeroed binary failed: {stdout}\n{stderr}"
    assert "SUCCESS" in stdout, "Should report success"
    assert "Full pages zero-paged" in stdout, "Should confirm full pages zeroed"
    assert "Partial page preserved" in stdout, "Should confirm partial page kept"


def test_file_size_reduction(build_test_binaries):
    """Verify that zero-paging actually reduces file size."""
    test_name = "test_zero_page_aligned"
    input_bin = TEST_DIR / test_name
    output_bin = TEST_DIR / f"{test_name}.zeroed"

    # Apply zero-page
    exit_code, output = apply_zero_page(input_bin, output_bin)
    assert exit_code == 0

    # Check file sizes
    original_size = input_bin.stat().st_size
    zeroed_size = output_bin.stat().st_size

    assert zeroed_size < original_size, "Zeroed binary should be smaller"

    # Should save at least 4KB (one page)
    saved = original_size - zeroed_size
    assert saved >= 4096, f"Should save at least 4KB, saved {saved} bytes"


def test_binary_still_executable(build_test_binaries):
    """Verify zeroed binaries are still valid executables."""
    test_name = "test_zero_page_unaligned_size"
    input_bin = TEST_DIR / test_name
    output_bin = TEST_DIR / f"{test_name}.zeroed"

    # Apply zero-page
    exit_code, output = apply_zero_page(input_bin, output_bin)
    assert exit_code == 0

    # Check with readelf that it's still a valid ELF
    result = subprocess.run(
        ["readelf", "-h", str(output_bin)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "readelf should succeed on zeroed binary"
    assert "ELF64" in result.stdout, "Should be a valid ELF64 binary"


def test_tool_reports_savings(build_test_binaries):
    """Verify the tool reports accurate file size savings."""
    test_name = "test_zero_page_aligned"
    input_bin = TEST_DIR / test_name
    output_bin = TEST_DIR / f"{test_name}.zeroed"

    exit_code, output = apply_zero_page(input_bin, output_bin)
    assert exit_code == 0

    # Check that output includes size information
    assert "Original size:" in output
    assert "New size:" in output
    assert "Saved:" in output
    assert "bytes" in output

    # Parse the saved bytes
    import re
    match = re.search(r"Saved: ([\d,]+) bytes", output)
    assert match, "Should report saved bytes"
    saved_bytes = int(match.group(1).replace(",", ""))
    assert saved_bytes == 4096, "Should save exactly 4096 bytes (1 page)"


def test_nonexistent_section(build_test_binaries):
    """Test that tool handles missing section gracefully."""
    # Use one of the test binaries
    test_name = "test_zero_page_aligned"
    input_bin = TEST_DIR / test_name
    output_bin = TEST_DIR / "test_nonexistent.zeroed"

    # Clean up any existing output file from previous runs
    output_bin.unlink(missing_ok=True)

    # Call tool with a section that doesn't exist
    exit_code = main([
        'zero-page',
        str(input_bin),
        str(output_bin),
        '--section=.nonexistent_section',
        '--quiet'
    ])

    # Should fail gracefully
    assert exit_code != 0, "Should fail when section doesn't exist"
    assert not output_bin.exists(), "Should not create output on failure"


def test_all_cases_pass(build_test_binaries):
    """
    Run all test cases to ensure comprehensive coverage.

    Note: Some tests may behave differently than expected due to linker
    behavior (see README.md), but should still execute successfully.
    """
    for test_name in build_test_binaries:
        input_bin = TEST_DIR / test_name
        output_bin = TEST_DIR / f"{test_name}.zeroed"

        # Skip if already tested individually
        if test_name in ["test_zero_page_aligned", "test_zero_page_unaligned_size"]:
            continue

        # Apply zero-page
        exit_code, output = apply_zero_page(input_bin, output_bin)

        # Tool should succeed
        assert exit_code == 0, f"Tool failed for {test_name}: {output}"

        # Binary should be created
        assert output_bin.exists(), f"Output not created for {test_name}"

        # Binary should be smaller
        assert (output_bin.stat().st_size < input_bin.stat().st_size), \
            f"No size reduction for {test_name}"


# ============================================================================
# New tests for map-section and update-relocation functionality
# ============================================================================

@pytest.fixture(scope="module")
def build_mapped_section_test():
    """Build the test_mapped_section binary."""
    source = TEST_DIR / "test_mapped_section.c"
    output = TEST_DIR / "test_mapped_section"
    custom_data_file = TEST_DIR / "custom_data.bin"

    # Create custom data file with test string
    test_string = b"Hello from mapped section!\x00"
    custom_data_file.write_bytes(test_string)

    # Compile as PIE (Position Independent Executable) - the common case
    # Most modern executables are PIE, and all shared libraries are position-independent
    # Tests use auto-allocated addresses which work with PIE's relative addressing
    result = subprocess.run(
        [
            "gcc",
            "-O0",
            "-g",
            "-o", str(output),
            str(source),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to build test_mapped_section: {result.stderr}")

    # Add custom section using objcopy (without ALLOC flag)
    # This creates a section that exists in the ELF file but is NOT mapped to memory
    result = subprocess.run(
        [
            "objcopy",
            "--add-section", f".custom_data={str(custom_data_file)}",
            "--set-section-flags", ".custom_data=contents,readonly",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to add custom section: {result.stderr}")

    yield output

    # Cleanup
    output.unlink(missing_ok=True)
    custom_data_file.unlink(missing_ok=True)
    (TEST_DIR / "test_mapped_section.mapped").unlink(missing_ok=True)


def test_map_section_basic(build_mapped_section_test, toolchain):
    """
    Test basic section mapping to new PT_LOAD with auto-allocation.

    This verifies that a section can be mapped to a new virtual address range
    using auto-allocated addresses (works with PIE binaries).
    """
    import io
    import sys

    input_bin = build_mapped_section_test
    output_bin = TEST_DIR / "test_mapped_section.step1"

    # PRE-CONDITION: Verify .custom_data is NOT in a PT_LOAD before mapping
    assert not is_section_in_pt_load(input_bin, ".custom_data"), \
        ".custom_data should NOT be in PT_LOAD before mapping (section should not have ALLOC flag)"

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = output_buffer = io.StringIO()

    try:
        # Map section with auto-allocated address (no --vaddr specified)
        exit_code = main([
            'map-section',
            str(input_bin),
            str(output_bin),
            '--section=.custom_data',
        ])
        output = output_buffer.getvalue()
    finally:
        sys.stdout = old_stdout

    assert exit_code == 0, f"map-section failed: {output}"
    assert output_bin.exists(), "Output binary not created"

    # Verify output mentions successful mapping
    assert "Successfully mapped" in output

    # POST-CONDITION: Verify .custom_data IS now in a PT_LOAD
    assert is_section_in_pt_load(output_bin, ".custom_data"), \
        ".custom_data should be in PT_LOAD after mapping"

    # Use get_section_vaddr to retrieve the auto-allocated address
    mapped_vaddr = get_section_vaddr(toolchain, output_bin, ".custom_data")
    assert mapped_vaddr is not None, ".custom_data should have a virtual address after mapping"

    # Verify with readelf that new PT_LOAD exists at the mapped address
    result = subprocess.run(
        ["readelf", "-l", str(output_bin)],
        capture_output=True,
        text=True,
    )
    hex_addr = f"{mapped_vaddr:016x}"
    assert hex_addr in result.stdout, f"Should have PT_LOAD at 0x{hex_addr}"

    # Cleanup
    output_bin.unlink(missing_ok=True)


def test_set_pointer_basic(build_mapped_section_test, toolchain):
    """
    Test set-pointer command with auto-allocated section address.

    This tests the complete workflow:
    1. Map section to new PT_LOAD (auto-allocated address)
    2. Set pointer to the mapped section
    3. Verify pointer was written and relocation updated (PIE binaries)
    """
    import io
    import sys
    import struct

    input_bin = build_mapped_section_test
    mapped_bin = TEST_DIR / "test_mapped_section.set_pointer_mapped"
    output_bin = TEST_DIR / "test_mapped_section.set_pointer"

    # Step 1: Map .custom_data section (auto-allocate address)
    exit_code = main([
        'map-section',
        str(input_bin),
        str(mapped_bin),
        '--section=.custom_data',
        '--quiet',
    ])
    assert exit_code == 0, "map-section failed"

    # Get the auto-allocated address
    target_vaddr = get_section_vaddr(toolchain, mapped_bin, ".custom_data")
    assert target_vaddr is not None, ".custom_data should be mapped"

    # Find .test_wrapper section address and file offset
    result = subprocess.run(
        ["readelf", "-S", str(mapped_bin)],
        capture_output=True,
        text=True,
    )

    wrapper_vaddr = None
    wrapper_offset = None
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        if '.test_wrapper' in line:
            parts = line.split()
            if len(parts) >= 5:
                wrapper_vaddr = int(parts[3], 16)  # Address column
                wrapper_offset = int(parts[4], 16)  # Offset column
                break

    if wrapper_vaddr is None:
        pytest.skip(".test_wrapper section not found")

    # The data_ptr field is at offset +8 in the structure
    pointer_vaddr = wrapper_vaddr + 8
    pointer_offset = wrapper_offset + 8

    # Read original pointer value
    mapped_data = mapped_bin.read_bytes()
    original_ptr = struct.unpack_from('<Q', mapped_data, pointer_offset)[0]

    # Step 2: Set pointer to the mapped section
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = output_buffer = io.StringIO()

    try:
        exit_code = main([
            'set-pointer',
            str(mapped_bin),
            str(output_bin),
            f'--at=0x{pointer_vaddr:x}',
            f'--target=0x{target_vaddr:x}',
        ])
        output = output_buffer.getvalue()
    finally:
        sys.stdout = old_stdout

    assert exit_code == 0, f"set-pointer failed: {output}"
    assert output_bin.exists(), "Output binary not created"

    # Verify output mentions success
    assert "Setting pointer" in output
    assert "Successfully set pointer" in output

    # Step 3: Verify the pointer value was actually written to the file
    output_data = output_bin.read_bytes()
    written_ptr = struct.unpack_from('<Q', output_data, pointer_offset)[0]

    assert written_ptr == target_vaddr, \
        f"Pointer not written correctly: expected 0x{target_vaddr:x}, got 0x{written_ptr:x}"

    # Verify the pointer value changed from original
    assert written_ptr != original_ptr, \
        f"Pointer value unchanged: {original_ptr:x}"

    # Check if relocation was mentioned in output (indicates PIE binary with relocations)
    if "Updated relocation" in output:
        # Relocation was found and updated - verify with readelf
        result = subprocess.run(
            ["readelf", "-r", str(output_bin)],
            capture_output=True,
            text=True,
        )
        # Look for relocation at our pointer address
        hex_addr = f"{pointer_vaddr:x}"
        assert hex_addr in result.stdout.lower(), \
            "Relocation should exist at pointer address"
    elif "No relocation found" in output:
        # This is OK for non-PIE binaries
        pass

    # Cleanup
    mapped_bin.unlink(missing_ok=True)
    output_bin.unlink(missing_ok=True)


def test_set_pointer_requires_relocation_for_pie(build_mapped_section_test):
    """
    Test that set-pointer fails for PIE binary without relocation.

    This ensures we catch configuration errors early (at neutralization time)
    rather than creating binaries that will crash at runtime.
    """
    input_bin = build_mapped_section_test
    output_bin = TEST_DIR / "test_should_fail"

    # Try to set pointer at a location that has no relocation
    # Use an address in .rodata or another section without relocations
    # For this test, we'll use an arbitrary address that's valid but has no relocation
    exit_code = main([
        'set-pointer',
        str(input_bin),
        str(output_bin),
        '--at=0x2100',  # Address in .rodata (read-only data, no relocations)
        '--target=0x5000',
        '--quiet',
    ])

    # Should fail for PIE binary without relocation
    assert exit_code != 0, "Should fail when setting pointer without relocation in PIE binary"
    assert not output_bin.exists(), "Should not create output file on failure"

    # Cleanup (in case test failed and file was created)
    output_bin.unlink(missing_ok=True)


def test_full_workflow_map_and_relocate(build_mapped_section_test, toolchain):
    """
    Test complete workflow with auto-allocated addresses (PIE-compatible).

    This simulates the realistic kpack use case:
    1. Map .custom_data section to new PT_LOAD (auto-allocated address)
    2. Set pointer in test_wrapper to auto-allocated address (updates relocation if present)
    3. Verify binary executes and can read from mapped section
    """
    input_bin = build_mapped_section_test
    step1_bin = TEST_DIR / "test_mapped_section.step1"
    final_bin = TEST_DIR / "test_mapped_section.mapped"

    # PRE-CONDITION: Verify .custom_data is NOT in a PT_LOAD before mapping
    assert not is_section_in_pt_load(input_bin, ".custom_data"), \
        ".custom_data should NOT be in PT_LOAD before mapping"

    # Step 1: Map .custom_data to new PT_LOAD (auto-allocate address)
    exit_code = main([
        'map-section',
        str(input_bin),
        str(step1_bin),
        '--section=.custom_data',
        '--quiet',
    ])

    assert exit_code == 0, "map-section failed"

    # POST-CONDITION: Verify .custom_data IS now in a PT_LOAD
    assert is_section_in_pt_load(step1_bin, ".custom_data"), \
        ".custom_data should be in PT_LOAD after mapping"

    # Get the auto-allocated address
    target_vaddr = get_section_vaddr(toolchain, step1_bin, ".custom_data")
    assert target_vaddr is not None, ".custom_data should be mapped"

    # Step 2: Find .test_wrapper address and set pointer to mapped section
    result = subprocess.run(
        ["readelf", "-S", str(step1_bin)],
        capture_output=True,
        text=True,
    )

    # Parse to find .test_wrapper virtual address
    wrapper_vaddr = None
    for line in result.stdout.split('\n'):
        if '.test_wrapper' in line:
            parts = line.split()
            if len(parts) >= 4:
                addr_str = parts[3]
                wrapper_vaddr = int(addr_str, 16)
                break

    if wrapper_vaddr is None:
        pytest.skip(".test_wrapper section not found")

    pointer_vaddr = wrapper_vaddr + 8  # data_ptr field offset

    exit_code = main([
        'set-pointer',
        str(step1_bin),
        str(final_bin),
        f'--at=0x{pointer_vaddr:x}',
        f'--target=0x{target_vaddr:x}',
        '--quiet',
    ])

    assert exit_code == 0, "set-pointer failed"

    # Step 3: Execute final binary
    exit_code, stdout, stderr = run_binary(final_bin)

    if exit_code != 0:
        print(f"Binary output:\n{stdout}")
        print(f"Binary errors:\n{stderr}")

    assert exit_code == 0, f"Final binary failed: {stderr}"
    assert "SUCCESS" in stdout, "Should report success"
    assert "Hello from mapped section!" in stdout, "Should read mapped data"

    # Cleanup
    step1_bin.unlink(missing_ok=True)
    final_bin.unlink(missing_ok=True)
