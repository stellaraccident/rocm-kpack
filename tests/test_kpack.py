"""Tests for packed kernel archive format."""

from pathlib import Path

import pytest

from rocm_kpack.kpack import PackedKernelArchive


def test_pack_archive_roundtrip(tmp_path: Path):
    """Test creating, writing, and reading back a pack archive."""
    # Create archive
    archive = PackedKernelArchive(
        group_name="blas",
        gfx_arch_family="gfx100X",
        gfx_arches=["gfx1030", "gfx1001"],
    )

    # Add some kernels
    archive.add_kernel(
        relative_path="bin/hipcc",
        gfx_arch="gfx1030",
        hsaco_data=b"kernel_data_gfx1030",
    )
    archive.add_kernel(
        relative_path="bin/hipcc",
        gfx_arch="gfx1001",
        hsaco_data=b"kernel_data_gfx1001",
    )
    archive.add_kernel(
        relative_path="lib/libhipblas.so",
        gfx_arch="gfx1030",
        hsaco_data=b"hipblas_kernel_gfx1030",
        metadata={"version": "1.0"},
    )

    # Write to file
    pack_file = tmp_path / "blas-gfx100X.kpack"
    archive.write(pack_file)

    # Verify file exists
    assert pack_file.exists()
    assert pack_file.stat().st_size > 0

    # Read back
    loaded = PackedKernelArchive.read(pack_file)

    # Verify metadata
    assert loaded.group_name == "blas"
    assert loaded.gfx_arch_family == "gfx100X"
    assert loaded.gfx_arches == ["gfx1030", "gfx1001"]

    # Verify kernels
    assert loaded.get_kernel("bin/hipcc", "gfx1030") == b"kernel_data_gfx1030"
    assert loaded.get_kernel("bin/hipcc", "gfx1001") == b"kernel_data_gfx1001"
    assert loaded.get_kernel("lib/libhipblas.so", "gfx1030") == b"hipblas_kernel_gfx1030"

    # Verify metadata preserved
    assert loaded.toc["lib/libhipblas.so"]["gfx1030"]["metadata"] == {"version": "1.0"}

    # Verify non-existent kernels
    assert loaded.get_kernel("nonexistent/binary", "gfx1030") is None
    assert loaded.get_kernel("bin/hipcc", "gfx1100") is None


def test_pack_archive_duplicate_kernel_error(tmp_path: Path):
    """Test that adding duplicate kernel raises error."""
    archive = PackedKernelArchive(
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100"],
    )

    archive.add_kernel(
        relative_path="bin/test",
        gfx_arch="gfx1100",
        hsaco_data=b"data",
    )

    with pytest.raises(ValueError, match="Kernel already exists"):
        archive.add_kernel(
            relative_path="bin/test",
            gfx_arch="gfx1100",
            hsaco_data=b"different_data",
        )


def test_pack_archive_path_normalization(tmp_path: Path):
    """Test that paths are normalized consistently."""
    archive = PackedKernelArchive(
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100"],
    )

    # Add with backslashes (Windows-style)
    archive.add_kernel(
        relative_path="bin\\hipcc",
        gfx_arch="gfx1100",
        hsaco_data=b"data",
    )

    pack_file = tmp_path / "test.kpack"
    archive.write(pack_file)
    loaded = PackedKernelArchive.read(pack_file)

    # Retrieve with forward slashes (should work)
    assert loaded.get_kernel("bin/hipcc", "gfx1100") == b"data"


def test_pack_archive_ordinals(tmp_path: Path):
    """Test that ordinals are assigned sequentially."""
    archive = PackedKernelArchive(
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100"],
    )

    archive.add_kernel("bin/a", "gfx1100", b"data_0")
    archive.add_kernel("bin/b", "gfx1100", b"data_1")
    archive.add_kernel("bin/c", "gfx1100", b"data_2")

    # Check ordinals
    assert archive.toc["bin/a"]["gfx1100"]["ordinal"] == 0
    assert archive.toc["bin/b"]["gfx1100"]["ordinal"] == 1
    assert archive.toc["bin/c"]["gfx1100"]["ordinal"] == 2

    # Verify data array
    assert archive.data[0] == b"data_0"
    assert archive.data[1] == b"data_1"
    assert archive.data[2] == b"data_2"


def test_compute_pack_filename():
    """Test pack filename computation."""
    assert PackedKernelArchive.compute_pack_filename("blas", "gfx1100") == "blas-gfx1100.kpack"
    assert PackedKernelArchive.compute_pack_filename("torch", "gfx100X") == "torch-gfx100X.kpack"


def test_pack_archive_repr():
    """Test string representation."""
    archive = PackedKernelArchive(
        group_name="blas",
        gfx_arch_family="gfx100X",
        gfx_arches=["gfx1030", "gfx1001"],
    )
    archive.add_kernel("bin/a", "gfx1030", b"data")
    archive.add_kernel("bin/a", "gfx1001", b"data")
    archive.add_kernel("bin/b", "gfx1030", b"data")

    repr_str = repr(archive)
    assert "blas" in repr_str
    assert "gfx100X" in repr_str
    assert "binaries=2" in repr_str
    assert "kernels=3" in repr_str


def test_pack_archive_streaming_mode(tmp_path: Path):
    """Test streaming write mode for memory efficiency."""
    pack_file = tmp_path / "streaming-test.kpack"

    # Create archive in streaming mode
    archive = PackedKernelArchive(
        group_name="blas",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100"],
        output_path=pack_file,
    )

    # Verify in-memory data array is not used (stays empty)
    assert len(archive.data) == 0

    # Add kernels - should write to disk immediately
    archive.add_kernel("bin/test1", "gfx1100", b"A" * 100)
    assert len(archive.data) == 0  # Still empty in streaming mode

    archive.add_kernel("bin/test2", "gfx1100", b"B" * 200)
    assert len(archive.data) == 0

    # Finalize to write TOC
    archive.finalize()

    # Verify file exists
    assert pack_file.exists()
    assert pack_file.stat().st_size > 0

    # Read back and verify
    loaded = PackedKernelArchive.read(pack_file)
    assert loaded.get_kernel("bin/test1", "gfx1100") == b"A" * 100
    assert loaded.get_kernel("bin/test2", "gfx1100") == b"B" * 200


def test_pack_archive_streaming_mode_errors(tmp_path: Path):
    """Test that streaming mode enforces correct API usage."""
    pack_file = tmp_path / "error-test.kpack"

    # Streaming mode archive
    archive = PackedKernelArchive(
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100"],
        output_path=pack_file,
    )

    # Should not be able to call write() in streaming mode
    with pytest.raises(RuntimeError, match="streaming mode"):
        archive.write(tmp_path / "other.kpack")

    archive.finalize()

    # In-memory mode archive
    archive2 = PackedKernelArchive(
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100"],
    )

    # Should not be able to call finalize() in in-memory mode
    with pytest.raises(RuntimeError, match="Not in streaming mode"):
        archive2.finalize()
