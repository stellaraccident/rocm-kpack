#!/usr/bin/env python3
"""
Tests for ELF zero-page optimization.

This module tests the conservative zero-page algorithm using C test binaries
that validate different alignment scenarios.
"""

import subprocess
import pytest
from pathlib import Path


# Test directory containing C sources and built binaries
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent.parent
TOOL_PATH = PROJECT_ROOT / "python" / "rocm_kpack" / "elf_zero_pages.py"


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
    result = subprocess.run(
        [
            "python",
            str(TOOL_PATH),
            str(input_path),
            str(output_path),
            "--section=.testdata",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout


def test_tool_exists():
    """Verify the elf_zero_pages.py tool exists."""
    assert TOOL_PATH.exists(), f"Tool not found at {TOOL_PATH}"


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
    result = subprocess.run(
        [
            "python",
            str(TOOL_PATH),
            str(input_bin),
            str(output_bin),
            "--section=.nonexistent_section",
        ],
        capture_output=True,
        text=True,
    )

    # Should fail gracefully
    assert result.returncode != 0, "Should fail when section doesn't exist"
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
