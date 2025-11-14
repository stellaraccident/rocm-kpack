#!/usr/bin/env python3
"""
Test that zero-page optimizer handles program header overflow correctly.

This test forces a program header overflow scenario and verifies that:
1. The binary is still valid after optimization
2. The interpreter is preserved correctly
3. The binary can execute
4. No silent corruption occurs
"""

import subprocess
import pytest
from pathlib import Path
import sys

# Add rocm_kpack to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from rocm_kpack.elf_modify_load import conservative_zero_page


TEST_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def test_binary():
    """Build a simple test binary."""
    test_name = "test_zero_page_aligned"
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

    yield output

    # Cleanup
    output.unlink(missing_ok=True)


def test_forced_overflow_produces_valid_binary(test_binary):
    """
    Test that force_overflow mode produces a valid, executable binary.

    This is the main test - it should FAIL until we implement program
    header relocation properly. Currently it raises AssertionError.
    """
    output = TEST_DIR / "test_overflow_forced.zeroed"

    try:
        # Force overflow condition - should handle it gracefully
        success = conservative_zero_page(
            test_binary,
            output,
            section_name=".testdata",
            verbose=True,
            force_overflow=True
        )

        assert success, "Zero-paging should succeed even with overflow"
        assert output.exists(), "Output file should be created"

        # Verify binary is valid ELF
        result = subprocess.run(
            ["readelf", "-h", str(output)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Output should be valid ELF"
        assert "ELF64" in result.stdout

        # Verify interpreter is preserved
        result = subprocess.run(
            ["readelf", "-l", str(output)],
            capture_output=True,
            text=True,
        )
        assert "/lib64/ld-linux-x86-64.so.2" in result.stdout, "Interpreter should be preserved"

        # Verify binary can execute
        result = subprocess.run(
            [str(output)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Binary should execute successfully: {result.stderr}"
        assert "SUCCESS" in result.stdout, "Binary should report success"

    finally:
        # Cleanup
        output.unlink(missing_ok=True)


def test_overflow_handling_with_padding(test_binary):
    """
    Test that overflow is handled correctly with padding.

    The zero-page optimizer should add padding to ensure proper
    alignment when relocating program headers.
    """
    output = TEST_DIR / "test_overflow_padding.zeroed"

    try:
        # Should succeed with force_overflow
        success = conservative_zero_page(
            test_binary,
            output,
            section_name=".testdata",
            verbose=False,
            force_overflow=True
        )

        assert success, "Zero-paging should succeed with padding"
        assert output.exists(), "Output file should be created"

        # Verify the binary is valid
        result = subprocess.run(
            ["readelf", "-l", str(output)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Should be a valid ELF"

        # Check that program headers are relocated (offset > 1000)
        assert "starting at offset" in result.stdout
        import re
        match = re.search(r"starting at offset (\d+)", result.stdout)
        if match:
            offset = int(match.group(1))
            assert offset > 1000, f"Program headers should be relocated, found at {offset}"

    finally:
        output.unlink(missing_ok=True)
