"""Unit tests for database handlers."""

from pathlib import Path

import pytest

from rocm_kpack.database_handlers import (
    RocBLASHandler,
    HipBLASLtHandler,
    AotritonHandler,
    get_database_handlers,
    list_available_handlers,
)


class TestRocBLASHandler:
    """Tests for RocBLASHandler detection logic."""

    @pytest.fixture
    def handler(self):
        return RocBLASHandler()

    @pytest.fixture
    def prefix_root(self, tmp_path):
        """Create a temporary prefix root directory."""
        root = tmp_path / "prefix"
        root.mkdir()
        return root

    def test_name(self, handler):
        """Test handler name."""
        assert handler.name() == "rocblas"

    def test_detect_co_file(self, handler, prefix_root):
        """Test detection of .co file in rocblas/library."""
        file_path = prefix_root / "lib/rocblas/library/TensileLibrary_gfx1100.co"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1100"

    def test_detect_hsaco_file(self, handler, prefix_root):
        """Test detection of .hsaco file in rocblas/library."""
        file_path = prefix_root / "lib/rocblas/library/kernel_gfx1101.hsaco"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1101"

    def test_detect_dat_file(self, handler, prefix_root):
        """Test detection of .dat file in rocblas/library."""
        file_path = prefix_root / "lib/rocblas/library/TensileLibrary_gfx1102.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1102"

    def test_detect_various_architectures(self, handler, prefix_root):
        """Test detection of various gfx architecture formats."""
        test_cases = [
            ("TensileLibrary_gfx90a.dat", "gfx90a"),
            ("TensileLibrary_gfx942.dat", "gfx942"),
            ("kernels_gfx1030.co", "gfx1030"),
        ]

        for filename, expected_arch in test_cases:
            file_path = prefix_root / f"lib/rocblas/library/{filename}"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

            result = handler.detect(file_path, prefix_root)
            assert result == expected_arch, f"Failed for {filename}"

    def test_reject_wrong_directory(self, handler, prefix_root):
        """Test that files not in rocblas/library are rejected."""
        file_path = prefix_root / "lib/other/library/TensileLibrary_gfx1100.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_wrong_extension(self, handler, prefix_root):
        """Test that files with unsupported extensions are rejected."""
        file_path = prefix_root / "lib/rocblas/library/TensileLibrary_gfx1100.txt"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_no_architecture(self, handler, prefix_root):
        """Test that files without architecture suffix are rejected."""
        file_path = prefix_root / "lib/rocblas/library/TensileLibrary.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_file_outside_prefix(self, handler, prefix_root):
        """Test that files outside prefix root are rejected."""
        file_path = Path("/tmp/rocblas/library/TensileLibrary_gfx1100.dat")

        result = handler.detect(file_path, prefix_root)
        assert result is None


class TestHipBLASLtHandler:
    """Tests for HipBLASLtHandler detection logic."""

    @pytest.fixture
    def handler(self):
        return HipBLASLtHandler()

    @pytest.fixture
    def prefix_root(self, tmp_path):
        """Create a temporary prefix root directory."""
        root = tmp_path / "prefix"
        root.mkdir()
        return root

    def test_name(self, handler):
        """Test handler name."""
        assert handler.name() == "hipblaslt"

    def test_detect_co_file(self, handler, prefix_root):
        """Test detection of .co file in hipblaslt/library."""
        file_path = prefix_root / "lib/hipblaslt/library/TensileLibrary_gfx1100.co"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1100"

    def test_detect_hsaco_file(self, handler, prefix_root):
        """Test detection of .hsaco file in hipblaslt/library."""
        file_path = prefix_root / "lib/hipblaslt/library/kernel_gfx1101.hsaco"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1101"

    def test_detect_dat_file(self, handler, prefix_root):
        """Test detection of .dat file in hipblaslt/library."""
        file_path = prefix_root / "lib/hipblaslt/library/TensileLibrary_gfx1102.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1102"

    def test_detect_various_architectures(self, handler, prefix_root):
        """Test detection of various gfx architecture formats."""
        test_cases = [
            ("TensileLibrary_gfx90a.dat", "gfx90a"),
            ("TensileLibrary_gfx942.dat", "gfx942"),
            ("kernels_gfx1030.co", "gfx1030"),
        ]

        for filename, expected_arch in test_cases:
            file_path = prefix_root / f"lib/hipblaslt/library/{filename}"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

            result = handler.detect(file_path, prefix_root)
            assert result == expected_arch, f"Failed for {filename}"

    def test_reject_wrong_directory(self, handler, prefix_root):
        """Test that files not in hipblaslt/library are rejected."""
        file_path = prefix_root / "lib/rocblas/library/TensileLibrary_gfx1100.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_wrong_extension(self, handler, prefix_root):
        """Test that files with unsupported extensions are rejected."""
        file_path = prefix_root / "lib/hipblaslt/library/TensileLibrary_gfx1100.json"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_no_architecture(self, handler, prefix_root):
        """Test that files without architecture suffix are rejected."""
        file_path = prefix_root / "lib/hipblaslt/library/generic.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_file_outside_prefix(self, handler, prefix_root):
        """Test that files outside prefix root are rejected."""
        file_path = Path("/tmp/hipblaslt/library/TensileLibrary_gfx1100.dat")

        result = handler.detect(file_path, prefix_root)
        assert result is None


class TestAotritonHandler:
    """Tests for AotritonHandler detection logic."""

    @pytest.fixture
    def handler(self):
        return AotritonHandler()

    @pytest.fixture
    def prefix_root(self, tmp_path):
        """Create a temporary prefix root directory."""
        root = tmp_path / "prefix"
        root.mkdir()
        return root

    def test_name(self, handler):
        """Test handler name."""
        assert handler.name() == "aotriton"

    def test_detect_file_in_gfx_directory(self, handler, prefix_root):
        """Test detection of file in aotriton/kernels/gfx* directory."""
        file_path = prefix_root / "lib/aotriton/kernels/gfx1100/kernel.hsaco"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1100"

    def test_detect_different_architecture(self, handler, prefix_root):
        """Test detection of different gfx architecture."""
        file_path = prefix_root / "share/aotriton/kernels/gfx1101/kernel.co"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1101"

    def test_detect_nested_path(self, handler, prefix_root):
        """Test detection with nested path before aotriton."""
        file_path = prefix_root / "lib/foo/bar/aotriton/kernels/gfx942/kernel.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx942"

    def test_detect_various_architectures(self, handler, prefix_root):
        """Test detection of various gfx architecture formats."""
        test_cases = [
            ("gfx90a", "gfx90a"),
            ("gfx942", "gfx942"),
            ("gfx1030", "gfx1030"),
            ("gfx1102", "gfx1102"),
        ]

        for arch_dir, expected_arch in test_cases:
            file_path = prefix_root / f"lib/aotriton/kernels/{arch_dir}/kernel.hsaco"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

            result = handler.detect(file_path, prefix_root)
            assert result == expected_arch, f"Failed for {arch_dir}"

    def test_reject_wrong_directory_structure(self, handler, prefix_root):
        """Test that files not in aotriton/kernels are rejected."""
        file_path = prefix_root / "lib/aotriton/other/gfx1100/kernel.hsaco"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_non_gfx_directory(self, handler, prefix_root):
        """Test that files in non-gfx subdirectory are rejected."""
        file_path = prefix_root / "lib/aotriton/kernels/common/kernel.hsaco"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_missing_kernels_directory(self, handler, prefix_root):
        """Test that files not under kernels subdirectory are rejected."""
        file_path = prefix_root / "lib/aotriton/gfx1100/kernel.hsaco"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_file_outside_prefix(self, handler, prefix_root):
        """Test that files outside prefix root are rejected."""
        file_path = Path("/tmp/aotriton/kernels/gfx1100/kernel.hsaco")

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_detect_deeply_nested_file(self, handler, prefix_root):
        """Test detection of file nested multiple levels under architecture directory."""
        file_path = prefix_root / "lib/aotriton/kernels/gfx1100/subdir/deep/kernel.hsaco"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1100"


class TestDatabaseHandlerRegistry:
    """Tests for database handler registry functions."""

    def test_list_available_handlers(self):
        """Test that list_available_handlers returns all registered handlers."""
        handlers = list_available_handlers()
        assert isinstance(handlers, list)
        assert "rocblas" in handlers
        assert "hipblaslt" in handlers
        assert "aotriton" in handlers
        assert len(handlers) == 3

    def test_get_database_handlers_single(self):
        """Test getting a single handler by name."""
        handlers = get_database_handlers(["rocblas"])
        assert len(handlers) == 1
        assert isinstance(handlers[0], RocBLASHandler)

    def test_get_database_handlers_multiple(self):
        """Test getting multiple handlers by name."""
        handlers = get_database_handlers(["rocblas", "hipblaslt"])
        assert len(handlers) == 2
        assert isinstance(handlers[0], RocBLASHandler)
        assert isinstance(handlers[1], HipBLASLtHandler)

    def test_get_database_handlers_all(self):
        """Test getting all handlers."""
        handlers = get_database_handlers(["rocblas", "hipblaslt", "aotriton"])
        assert len(handlers) == 3
        assert isinstance(handlers[0], RocBLASHandler)
        assert isinstance(handlers[1], HipBLASLtHandler)
        assert isinstance(handlers[2], AotritonHandler)

    def test_get_database_handlers_unknown(self):
        """Test that unknown handler name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown database handler: unknown"):
            get_database_handlers(["unknown"])

    def test_get_database_handlers_mixed_valid_invalid(self):
        """Test that partially invalid handler list raises ValueError."""
        with pytest.raises(ValueError, match="Unknown database handler: invalid"):
            get_database_handlers(["rocblas", "invalid"])
