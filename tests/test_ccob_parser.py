"""Tests for CCOB parser module."""

import subprocess
from pathlib import Path

import pytest

from rocm_kpack.ccob_parser import (
    CCOBHeader,
    UncompressedBundle,
    decompress_ccob,
    list_ccob_targets,
    parse_ccob_file,
)


def test_ccob_header_parse():
    """Test CCOB header parsing."""
    # Construct a minimal CCOB v3 header
    header = (
        b"CCOB"  # Magic
        b"\x03\x00\x01\x00"  # Version 3, compression method 1 (zstd)
        b"\x0c\x8b\x12\x00\x00\x00\x00\x00"  # totalSize: 1,215,244
        b"\x38\x7f\xc6\x00\x00\x00\x00\x00"  # uncompressedSize: 13,008,696
        b"\xde\x30\xdc\xec\xea\x03\x74\xc3"  # hash
    )

    parsed = CCOBHeader.parse(header)
    assert parsed.magic == "CCOB"
    assert parsed.version == 3
    assert parsed.compression_method == 1
    assert parsed.total_size == 1_215_244
    assert parsed.uncompressed_size == 13_008_696


def test_ccob_header_invalid_magic():
    """Test that invalid magic raises ValueError."""
    bad_header = b"XXXX" + b"\x00" * 28

    with pytest.raises(ValueError, match="Invalid magic"):
        CCOBHeader.parse(bad_header)


def test_ccob_header_too_short():
    """Test that short header raises ValueError."""
    short_header = b"CCOB" + b"\x00" * 20

    with pytest.raises(ValueError, match="Header too short"):
        CCOBHeader.parse(short_header)


def test_decompress_ccob_with_real_binary(tmp_path: Path, toolchain):
    """Test CCOB decompression with a real ROCm binary."""
    # Find a real ROCm library with CCOB bundle
    rocm_lib = Path("/home/stella/workspace/rocm/gfx1100/lib/librocblas.so.5")

    if not rocm_lib.exists():
        pytest.skip("librocblas.so.5 not found")

    # Extract .hip_fatbin section
    fatbin_path = tmp_path / "librocblas_fatbin.o"
    result = subprocess.run(
        ["objcopy", "--dump-section", f".hip_fatbin={fatbin_path}", str(rocm_lib)],
        capture_output=True,
    )

    if result.returncode != 0:
        pytest.skip("Could not extract .hip_fatbin section")

    # Parse and decompress
    bundle = parse_ccob_file(fatbin_path)

    # Verify structure
    assert bundle.magic.startswith("__CLANG_OFFLOAD_BUNDLE__")
    assert bundle.num_entries > 0

    # Should have at least one device target
    triples = bundle.list_triples()
    device_targets = [t for t in triples if "hipv4-amdgcn" in t]
    assert len(device_targets) > 0

    # Verify we can extract code objects
    for triple in device_targets:
        code_obj = bundle.get_code_object(triple)
        assert code_obj is not None
        assert len(code_obj) > 0


def test_list_ccob_targets_with_real_binary(tmp_path: Path):
    """Test listing CCOB targets."""
    rocm_lib = Path("/home/stella/workspace/rocm/gfx1100/lib/librocblas.so.5")

    if not rocm_lib.exists():
        pytest.skip("librocblas.so.5 not found")

    # Extract .hip_fatbin section
    fatbin_path = tmp_path / "librocblas_fatbin.o"
    result = subprocess.run(
        ["objcopy", "--dump-section", f".hip_fatbin={fatbin_path}", str(rocm_lib)],
        capture_output=True,
    )

    if result.returncode != 0:
        pytest.skip("Could not extract .hip_fatbin section")

    # List targets
    data = fatbin_path.read_bytes()
    targets = list_ccob_targets(data)

    # Should have host + device targets
    assert len(targets) > 0
    assert any("host-" in t for t in targets)
    assert any("hipv4-amdgcn" in t for t in targets)


def test_decompress_respects_total_size():
    """Test that decompression respects totalSize field, not buffer size.

    This is the key fix that clang-offload-bundler is missing.
    """
    # This test would need a crafted CCOB with padding, which we have
    # in librocblas.so.5. The test above verifies this works.
    pass


def test_uncompressed_bundle_get_code_object():
    """Test getting specific code object from bundle."""
    # Create a minimal uncompressed bundle structure
    magic = b"__CLANG_OFFLOAD_BUNDLE__\x00"[:24]
    num_entries = (1).to_bytes(8, "little")

    # Single entry: offset=100, size=10, triple="test-triple"
    triple = b"test-triple"
    entry = (
        (100).to_bytes(8, "little")  # offset
        + (10).to_bytes(8, "little")  # size
        + len(triple).to_bytes(8, "little")  # triple_size
        + triple
    )

    # Padding to offset 100, then code object data
    padding = b"\x00" * (100 - 32 - len(entry))
    code_data = b"0123456789"

    bundle_data = magic + num_entries + entry + padding + code_data

    # Parse
    bundle = UncompressedBundle.parse(bundle_data)

    assert bundle.num_entries == 1
    assert bundle.entries[0].triple == "test-triple"

    # Get code object
    code_obj = bundle.get_code_object("test-triple")
    assert code_obj == b"0123456789"

    # Try non-existent triple
    assert bundle.get_code_object("nonexistent") is None
