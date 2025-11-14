"""
Artifact combination for the recombination phase.

This module combines generic and architecture-specific artifacts from the
map phase into unified output artifacts organized by package group.
"""

import shutil
from pathlib import Path

from rocm_kpack.artifact_collector import ArtifactCollector, CollectedArtifact
from rocm_kpack.artifact_utils import write_artifact_manifest
from rocm_kpack.manifest_merger import ManifestMerger, PackManifest
from rocm_kpack.packaging_config import ArchitectureGroup


class ArtifactCombiner:
    """
    Combines split artifacts into package-group artifacts.

    This class takes generic and architecture-specific artifacts from the
    map phase and combines them according to package group configuration.
    """

    def __init__(
        self,
        collector: ArtifactCollector,
        manifest_merger: ManifestMerger,
        verbose: bool = False
    ):
        """
        Initialize artifact combiner.

        Args:
            collector: ArtifactCollector with discovered artifacts
            manifest_merger: ManifestMerger for combining .kpm files
            verbose: Enable verbose output
        """
        self.collector = collector
        self.manifest_merger = manifest_merger
        self.verbose = verbose

    def combine_component(
        self,
        component_name: str,
        group_name: str,
        arch_group: ArchitectureGroup,
        output_dir: Path
    ) -> None:
        """
        Combine artifacts for a component and architecture group.

        Args:
            component_name: Component name (e.g., "rocblas_lib")
            group_name: Package group name (e.g., "gfx110X")
            arch_group: Architecture group configuration
            output_dir: Output directory for combined artifact

        Raises:
            ValueError: If required artifacts are missing or invalid
        """
        if self.verbose:
            print(f"\nCombining component '{component_name}' for group '{group_name}'")

        # Get generic artifact
        generic_artifact = self.collector.get_generic_artifact(component_name)
        if generic_artifact is None:
            raise ValueError(f"Generic artifact not found for component '{component_name}'")

        # Check which architectures are available
        availability = self.collector.validate_availability(
            component_name,
            arch_group.architectures,
            require_generic=True
        )

        if self.verbose:
            if availability.available:
                print(f"  Available architectures: {', '.join(availability.available)}")
            if availability.missing:
                print(f"  Missing architectures: {', '.join(availability.missing)}")

        # Create output artifact directory
        output_artifact_dir = output_dir / f"{component_name}_{group_name}"
        output_artifact_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"  Output artifact: {output_artifact_dir}")

        # Copy generic artifact structure
        self._copy_generic_artifact(generic_artifact, output_artifact_dir)

        # Copy architecture-specific content for each available architecture
        for arch in availability.available:
            arch_artifact = self.collector.get_arch_artifact(component_name, arch)
            if arch_artifact is None:
                # Should not happen since validate_availability returned it as available
                raise RuntimeError(f"Architecture artifact {arch} unexpectedly missing")

            self._copy_arch_artifact(arch_artifact, output_artifact_dir)

        # Merge and update .kpm manifests for all prefixes
        self._merge_manifests_for_artifact(
            generic_artifact,
            availability.available,
            component_name,
            output_artifact_dir
        )

        # Write artifact manifest
        write_artifact_manifest(output_artifact_dir, generic_artifact.prefixes)

        if self.verbose:
            print(f"  ✓ Combined artifact created successfully")

    def _copy_generic_artifact(
        self,
        generic_artifact: CollectedArtifact,
        output_dir: Path
    ) -> None:
        """
        Copy generic artifact structure to output directory.

        This copies all files and directories from the generic artifact,
        excluding architecture-specific .kpack files which will be added later.

        Args:
            generic_artifact: Generic artifact to copy
            output_dir: Destination directory
        """
        if self.verbose:
            print(f"  Copying generic artifact from {generic_artifact.path}")

        # Copy each prefix
        for prefix in generic_artifact.prefixes:
            src_prefix = generic_artifact.path / prefix
            dst_prefix = output_dir / prefix

            if not src_prefix.exists():
                if self.verbose:
                    print(f"    Skipping missing prefix: {prefix}")
                continue

            if self.verbose:
                print(f"    Copying prefix: {prefix}")

            # Copy the entire prefix directory tree
            if dst_prefix.exists():
                raise RuntimeError(
                    f"Destination prefix already exists: {dst_prefix}\n"
                    f"This indicates a duplicate copy or previous failed run"
                )

            shutil.copytree(src_prefix, dst_prefix, symlinks=True)

            # Validate copy succeeded
            if not dst_prefix.exists():
                raise RuntimeError(f"Failed to copy generic artifact prefix: {src_prefix} -> {dst_prefix}")

    def _copy_arch_artifact(
        self,
        arch_artifact: CollectedArtifact,
        output_dir: Path
    ) -> None:
        """
        Copy architecture-specific content to output directory.

        This copies kpack files and architecture-specific database files.

        Args:
            arch_artifact: Architecture-specific artifact
            output_dir: Destination directory
        """
        arch = arch_artifact.architecture
        if arch is None:
            raise ValueError("Architecture artifact has no architecture set")

        if self.verbose:
            print(f"    Copying arch-specific content for {arch}")

        # Copy each prefix's architecture-specific content
        for prefix in arch_artifact.prefixes:
            src_prefix = arch_artifact.path / prefix
            dst_prefix = output_dir / prefix

            if not src_prefix.exists():
                continue

            # Copy kpack directory (.kpack/*.kpack files)
            src_kpack_dir = src_prefix / ".kpack"
            if src_kpack_dir.exists():
                dst_kpack_dir = dst_prefix / ".kpack"
                dst_kpack_dir.mkdir(parents=True, exist_ok=True)

                # Copy .kpack files (but not .kpm manifests, we'll regenerate those)
                for kpack_file in src_kpack_dir.glob("*.kpack"):
                    dst_kpack_file = dst_kpack_dir / kpack_file.name
                    if self.verbose:
                        print(f"      Copying {kpack_file.name}")
                    shutil.copy2(kpack_file, dst_kpack_file)

                    # Validate kpack file was copied successfully
                    if not dst_kpack_file.exists():
                        raise RuntimeError(f"Failed to copy kpack file: {kpack_file}")
                    if dst_kpack_file.stat().st_size == 0:
                        raise RuntimeError(f"Kpack file is empty after copy: {dst_kpack_file}")
                    if dst_kpack_file.stat().st_size != kpack_file.stat().st_size:
                        raise RuntimeError(
                            f"Kpack file size mismatch after copy: "
                            f"{kpack_file.stat().st_size} -> {dst_kpack_file.stat().st_size}"
                        )

            # Copy kpack directory structure (for stage/.kpack/*.kpack layout)
            src_kpack_stage = src_prefix / "kpack" / "stage" / ".kpack"
            if src_kpack_stage.exists():
                dst_kpack_stage = dst_prefix / "kpack" / "stage" / ".kpack"
                dst_kpack_stage.mkdir(parents=True, exist_ok=True)

                for kpack_file in src_kpack_stage.glob("*.kpack"):
                    dst_kpack_file = dst_kpack_stage / kpack_file.name
                    if self.verbose:
                        print(f"      Copying {kpack_file.name}")
                    shutil.copy2(kpack_file, dst_kpack_file)

                    # Validate kpack file was copied successfully
                    if not dst_kpack_file.exists():
                        raise RuntimeError(f"Failed to copy kpack file: {kpack_file}")
                    if dst_kpack_file.stat().st_size == 0:
                        raise RuntimeError(f"Kpack file is empty after copy: {dst_kpack_file}")
                    if dst_kpack_file.stat().st_size != kpack_file.stat().st_size:
                        raise RuntimeError(
                            f"Kpack file size mismatch after copy: "
                            f"{kpack_file.stat().st_size} -> {dst_kpack_file.stat().st_size}"
                        )

            # Copy architecture-specific database files
            # These are already organized by architecture in the artifact
            # Just copy any files that aren't in the generic artifact
            self._copy_arch_specific_files(src_prefix, dst_prefix, arch)

    def _should_copy_arch_file(self, file_path: Path, arch: str) -> bool:
        """
        Check if file should be copied for this architecture.

        Args:
            file_path: Path to file to check
            arch: Architecture name

        Returns:
            True if file should be copied, False otherwise
        """
        # Skip .kpack directory (already handled separately)
        if ".kpack" in file_path.parts:
            return False

        # Skip kpack/stage/.kpack (already handled separately)
        if "kpack" in file_path.parts and "stage" in file_path.parts:
            return False

        # Copy files that contain the architecture in their name
        # (e.g., TensileLibrary_gfx1100.dat, Kernels_gfx1101.so)
        return arch in file_path.name

    def _copy_arch_specific_files(
        self,
        src_dir: Path,
        dst_dir: Path,
        arch: str
    ) -> None:
        """
        Copy architecture-specific files (e.g., database files).

        Args:
            src_dir: Source directory
            dst_dir: Destination directory
            arch: Architecture name
        """
        # Walk the source directory
        for src_file in src_dir.rglob("*"):
            if not src_file.is_file():
                continue

            if not self._should_copy_arch_file(src_file, arch):
                continue

            rel_path = src_file.relative_to(src_dir)
            dst_file = dst_dir / rel_path

            # Create parent directories
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            if self.verbose:
                print(f"      Copying {rel_path}")
            shutil.copy2(src_file, dst_file)

            # Validate file was copied successfully
            if not dst_file.exists():
                raise RuntimeError(f"Failed to copy arch-specific file: {src_file}")
            if dst_file.stat().st_size == 0:
                raise RuntimeError(f"Arch-specific file is empty after copy: {dst_file}")
            if dst_file.stat().st_size != src_file.stat().st_size:
                raise RuntimeError(
                    f"Arch-specific file size mismatch after copy: "
                    f"{src_file.stat().st_size} -> {dst_file.stat().st_size}"
                )

    def _merge_manifests_for_artifact(
        self,
        generic_artifact: CollectedArtifact,
        architectures: list[str],
        component_name: str,
        output_dir: Path
    ) -> None:
        """
        Merge .kpm manifests for all prefixes in the artifact.

        Args:
            generic_artifact: Generic artifact with prefix information
            architectures: List of architectures to include
            component_name: Component name
            output_dir: Output artifact directory
        """
        if self.verbose:
            print(f"  Merging manifests for {len(generic_artifact.prefixes)} prefixes")

        for prefix in generic_artifact.prefixes:
            # Find all .kpm manifests in the output directory for this prefix
            kpack_dir = output_dir / prefix / ".kpack"
            if not kpack_dir.exists():
                if self.verbose:
                    print(f"    No .kpack directory in prefix {prefix}, skipping")
                continue

            # Collect all existing manifests for this component
            manifests_to_merge: list[PackManifest] = []

            # The manifests from map phase are single-arch
            # We need to read kpack file info to rebuild the manifest
            kpack_files = list(kpack_dir.glob("*.kpack"))

            if not kpack_files:
                if self.verbose:
                    print(f"    No .kpack files in prefix {prefix}, skipping manifest")
                continue

            # Build manifest entries from kpack files
            from rocm_kpack.manifest_merger import KpackFileEntry

            kpack_entries: dict[str, KpackFileEntry] = {}

            for kpack_file in kpack_files:
                # Extract architecture from filename (e.g., component_gfx1100.kpack)
                name_parts = kpack_file.stem.rsplit("_", 1)
                if len(name_parts) != 2:
                    if self.verbose:
                        print(f"    Skipping kpack file with unexpected name: {kpack_file.name}")
                    continue

                arch = name_parts[1]

                if arch not in architectures:
                    if self.verbose:
                        print(f"    Skipping kpack for architecture {arch} (not in group)")
                    continue

                # Get file size
                size = kpack_file.stat().st_size

                # Kernel count is not required by runtime; set to 0
                # The runtime only uses the filename and architecture fields from the manifest.
                # Parsing kpack files to extract kernel count would add complexity without benefit.
                kernel_count = 0

                kpack_entries[arch] = KpackFileEntry(
                    architecture=arch,
                    filename=kpack_file.name,
                    size=size,
                    kernel_count=kernel_count
                )

            if not kpack_entries:
                if self.verbose:
                    print(f"    No valid kpack entries for prefix {prefix}")
                continue

            # Create merged manifest
            merged_manifest = PackManifest(
                format_version=1,
                component_name=component_name,
                prefix=prefix,
                kpack_files=kpack_entries
            )

            # Write merged manifest
            manifest_path = kpack_dir / f"{component_name}.kpm"
            merged_manifest.to_file(manifest_path)

            if self.verbose:
                print(f"    Created manifest with {len(kpack_entries)} architectures: {prefix}/.kpack/{component_name}.kpm")
