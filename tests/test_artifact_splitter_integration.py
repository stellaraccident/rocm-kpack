"""
Integration tests for the artifact splitter.

These tests simulate real artifact splitting scenarios with mock data.
"""

import shutil
from pathlib import Path

import pytest
import msgpack

from rocm_kpack.artifact_splitter import (
    ArtifactSplitter,
    FileClassificationVisitor,
    ExtractedKernel,
)
from rocm_kpack.artifact_utils import read_artifact_manifest, write_artifact_manifest
from rocm_kpack.database_handlers import RocBLASHandler


class TestArtifactSplitterIntegration:
    """Integration tests for the complete artifact splitting workflow."""

    @pytest.fixture
    def create_test_artifact(self, tmp_path):
        """Create a test artifact directory structure."""
        def _create(prefixes, files_per_prefix=5, include_fat_binaries=False, include_db_files=False):
            artifact_dir = tmp_path / "test_artifact"
            artifact_dir.mkdir()

            # Write artifact manifest
            write_artifact_manifest(artifact_dir, prefixes)

            # Create prefix directories and files
            for prefix in prefixes:
                prefix_dir = artifact_dir / prefix
                prefix_dir.mkdir(parents=True)

                # Create regular files
                lib_dir = prefix_dir / "lib"
                lib_dir.mkdir(parents=True, exist_ok=True)

                for i in range(files_per_prefix):
                    file_path = lib_dir / f"libtest{i}.so"
                    file_path.write_text(f"Mock library content {i}")

                # Optionally create fat binaries (mock)
                if include_fat_binaries:
                    fat_bin = lib_dir / "libfat.so"
                    fat_bin.write_text("Mock fat binary with device code")
                    # Mark it for our tests
                    (lib_dir / ".test_fat_marker").write_text("libfat.so")

                # Optionally create database files
                if include_db_files:
                    db_dir = lib_dir / "rocblas" / "library"
                    db_dir.mkdir(parents=True, exist_ok=True)

                    # Create mock rocBLAS database files
                    (db_dir / "TensileLibrary_gfx1100.dat").write_text("Mock tensor data")
                    (db_dir / "TensileLibrary_gfx1100.co").write_text("Mock code object")
                    (db_dir / "kernels.db").write_text("Mock kernel database")

            return artifact_dir

        return _create

    def test_simple_artifact_split(self, create_test_artifact, toolchain, tmp_path):
        """Test splitting a simple artifact without fat binaries or databases."""
        # Create test artifact with plain text files (not ELF)
        input_dir = create_test_artifact(
            prefixes=["math-libs/BLAS/rocBLAS/stage"],
            files_per_prefix=3
        )

        output_dir = tmp_path / "output"

        # Create splitter with real toolchain
        splitter = ArtifactSplitter(
            component_name="test_lib",
            toolchain=toolchain,
            database_handlers=[],
            verbose=True
        )

        # Run the split - text files won't be detected as fat binaries
        splitter.split(input_dir, output_dir)

        # Verify output structure
        assert output_dir.exists()

        # Should have created generic artifact
        generic_dir = output_dir / "test_lib_generic"
        assert generic_dir.exists()

        # Check manifest was copied
        generic_manifest = read_artifact_manifest(generic_dir)
        assert generic_manifest == ["math-libs/BLAS/rocBLAS/stage"]

        # Check files were copied
        generic_prefix = generic_dir / "math-libs/BLAS/rocBLAS/stage"
        assert generic_prefix.exists()
        assert (generic_prefix / "lib" / "libtest0.so").exists()
        assert (generic_prefix / "lib" / "libtest1.so").exists()
        assert (generic_prefix / "lib" / "libtest2.so").exists()

    def test_artifact_with_fat_binaries(self, test_assets_dir, toolchain, tmp_path):
        """Test splitting artifact with real fat binaries from test assets."""
        # Create test artifact structure
        input_dir = tmp_path / "test_artifact"
        input_dir.mkdir()

        # Create artifact manifest
        prefix = "test/lib/stage"
        write_artifact_manifest(input_dir, [prefix])

        # Create prefix directory
        prefix_dir = input_dir / prefix
        lib_dir = prefix_dir / "lib"
        lib_dir.mkdir(parents=True)

        # Copy real fat binary from test assets
        fat_binary_src = test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_multi.so"
        fat_binary_dest = lib_dir / "libtest.so"
        shutil.copy2(fat_binary_src, fat_binary_dest)

        # Also copy a host-only library
        host_only_src = test_assets_dir / "bundled_binaries/linux/cov5/libhost_only.so"
        host_only_dest = lib_dir / "libhost.so"
        shutil.copy2(host_only_src, host_only_dest)

        output_dir = tmp_path / "output"

        # Create splitter with real toolchain - run everything live
        splitter = ArtifactSplitter(
            component_name="test_lib",
            toolchain=toolchain,
            database_handlers=[],
            verbose=True
        )

        # Run the full split operation
        splitter.split(input_dir, output_dir)

        # Verify generic artifact was created
        generic_dir = output_dir / "test_lib_generic"
        assert generic_dir.exists()

        # Check that both libraries were copied to generic
        generic_lib_dir = generic_dir / prefix / "lib"
        assert (generic_lib_dir / "libtest.so").exists()
        assert (generic_lib_dir / "libhost.so").exists()

        # Verify fat binary was detected and kernels extracted
        # Should have created architecture-specific artifacts
        # The test binary has gfx1100 and gfx1101 kernels
        arch_artifacts = list(output_dir.glob("test_lib_gfx*"))
        assert len(arch_artifacts) >= 1, "Should have created at least one architecture-specific artifact"

        # Check that kpack files were created
        for arch_artifact in arch_artifacts:
            kpack_files = list(arch_artifact.glob("kpack/stage/.kpack/*.kpack"))
            assert len(kpack_files) == 1, f"Should have one kpack file in {arch_artifact}"

        # Check the manifest was created in generic artifact
        manifest_file = generic_dir / prefix / ".kpack" / "test_lib.kpm"
        assert manifest_file.exists(), "Manifest file should exist in generic artifact"

        # Verify manifest content
        with open(manifest_file, 'rb') as f:
            manifest_data = msgpack.unpack(f)

        assert manifest_data["format_version"] == 1
        assert manifest_data["component_name"] == "test_lib"
        assert manifest_data["prefix"] == prefix
        assert len(manifest_data["kpack_files"]) >= 1, "Should have at least one architecture in manifest"

        # Check each architecture entry has required fields
        for arch, info in manifest_data["kpack_files"].items():
            assert "file" in info
            assert "size" in info
            assert "kernel_count" in info
            assert info["size"] > 0
            assert info["kernel_count"] > 0

        # Verify device code was stripped from fat binary in generic artifact
        original_size = fat_binary_src.stat().st_size
        stripped_size = (generic_lib_dir / "libtest.so").stat().st_size
        assert stripped_size < original_size, "Stripped binary should be smaller than original"

    def test_artifact_with_database_files(self, create_test_artifact, toolchain, tmp_path):
        """Test splitting artifact with kernel database files."""
        # Create test artifact with database files
        input_dir = create_test_artifact(
            prefixes=["math-libs/BLAS/rocBLAS/stage"],
            files_per_prefix=2,
            include_db_files=True
        )

        output_dir = tmp_path / "output"

        # Create splitter with rocBLAS handler
        rocblas_handler = RocBLASHandler()
        splitter = ArtifactSplitter(
            component_name="rocblas_lib",
            toolchain=toolchain,
            database_handlers=[rocblas_handler],
            verbose=True
        )

        # Run the split - text files won't be detected as fat binaries
        splitter.split(input_dir, output_dir)

        # Verify generic artifact exists
        generic_dir = output_dir / "rocblas_lib_generic"
        assert generic_dir.exists()

        # Verify database files were moved to architecture-specific artifact
        arch_dir = output_dir / "rocblas_lib_gfx1100"
        assert arch_dir.exists()

        # Check database files in arch-specific artifact
        db_path = arch_dir / "math-libs/BLAS/rocBLAS/stage/lib/rocblas/library"
        assert (db_path / "TensileLibrary_gfx1100.dat").exists()
        assert (db_path / "TensileLibrary_gfx1100.co").exists()

        # Verify database files are NOT in generic artifact
        generic_db_path = generic_dir / "math-libs/BLAS/rocBLAS/stage/lib/rocblas/library"
        if generic_db_path.exists():
            # Directory might exist but should be empty or not have database files
            assert not (generic_db_path / "TensileLibrary_gfx1100.dat").exists()
            assert not (generic_db_path / "TensileLibrary_gfx1100.co").exists()

    def test_multiple_prefixes(self, create_test_artifact, toolchain, tmp_path):
        """Test splitting artifact with multiple prefixes."""
        # Create artifact with multiple prefixes
        prefixes = [
            "math-libs/BLAS/rocBLAS/stage",
            "math-libs/BLAS/hipBLASLt/stage",
            "kpack/stage"
        ]

        input_dir = create_test_artifact(
            prefixes=prefixes,
            files_per_prefix=2
        )

        output_dir = tmp_path / "output"

        # Create splitter with real toolchain
        splitter = ArtifactSplitter(
            component_name="multi_lib",
            toolchain=toolchain,
            database_handlers=[],
            verbose=True
        )

        # Run the split - text files won't be detected as fat binaries
        splitter.split(input_dir, output_dir)

        # Verify generic artifact has all prefixes
        generic_dir = output_dir / "multi_lib_generic"
        assert generic_dir.exists()

        for prefix in prefixes:
            prefix_dir = generic_dir / prefix
            assert prefix_dir.exists(), f"Missing prefix: {prefix}"

    def test_file_classification_visitor(self, test_assets_dir, toolchain, tmp_path):
        """Test the FileClassificationVisitor directly with real files."""
        # Create test directory
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        lib_dir = test_dir / "lib"
        lib_dir.mkdir()

        # Copy real binaries from test assets
        fat_binary_src = test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_multi.so"
        fat_binary = lib_dir / "fat.so"
        shutil.copy2(fat_binary_src, fat_binary)

        host_only_src = test_assets_dir / "bundled_binaries/linux/cov5/libhost_only.so"
        regular_binary = lib_dir / "regular.so"
        shutil.copy2(host_only_src, regular_binary)

        # Create rocBLAS database files
        db_dir = lib_dir / "rocblas" / "library"
        db_dir.mkdir(parents=True)
        (db_dir / "TensileLibrary_gfx1100.dat").write_text("data")

        # Create visitor with handlers
        visitor = FileClassificationVisitor(
            toolchain=toolchain,
            database_handlers=[RocBLASHandler()],
            verbose=True
        )

        # Visit files - now using real ELF analysis
        visitor.visit_file(regular_binary, test_dir)
        visitor.visit_file(fat_binary, test_dir)
        visitor.visit_file(db_dir / "TensileLibrary_gfx1100.dat", test_dir)

        # Check classification results
        assert len(visitor.fat_binaries) == 1
        assert visitor.fat_binaries[0].name == "fat.so"

        assert "gfx1100" in visitor.database_files_by_arch
        assert len(visitor.database_files_by_arch["gfx1100"]) == 1

        assert len(visitor.exclude_from_generic) == 1  # Only database file

    def test_extracted_kernel_dataclass(self):
        """Test the ExtractedKernel dataclass."""
        kernel = ExtractedKernel(
            target_name="hipv4-amdgcn-amd-amdhsa--gfx906",
            kernel_data=b"kernel binary data",
            source_binary_relpath="lib/libtest.so",
            source_prefix="math-libs/BLAS/rocBLAS/stage",
            architecture="gfx906"
        )

        assert kernel.target_name == "hipv4-amdgcn-amd-amdhsa--gfx906"
        assert kernel.kernel_data == b"kernel binary data"
        assert kernel.source_binary_relpath == "lib/libtest.so"
        assert kernel.source_prefix == "math-libs/BLAS/rocBLAS/stage"
        assert kernel.architecture == "gfx906"

    def test_error_handling_missing_input(self, toolchain, tmp_path):
        """Test error handling for missing input directory."""
        splitter = ArtifactSplitter(
            component_name="test",
            toolchain=toolchain,
            database_handlers=[],
            verbose=False
        )

        non_existent = tmp_path / "non_existent"
        output_dir = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            splitter.split(non_existent, output_dir)

    def test_error_handling_missing_manifest(self, toolchain, tmp_path):
        """Test error handling for missing artifact manifest."""
        # Create directory without manifest
        input_dir = tmp_path / "no_manifest"
        input_dir.mkdir()

        output_dir = tmp_path / "output"

        splitter = ArtifactSplitter(
            component_name="test",
            toolchain=toolchain,
            database_handlers=[],
            verbose=False
        )

        with pytest.raises(FileNotFoundError, match="artifact_manifest.txt not found"):
            splitter.split(input_dir, output_dir)