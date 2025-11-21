"""
Manifest merging for the recombination phase.

This module handles reading .kpm manifest files from the map phase and
merging them into unified manifests for package groups.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import msgpack


@dataclass
class KpackFileEntry:
    """Information about a kpack file in a manifest."""

    architecture: str
    filename: str
    size: int
    kernel_count: int


class FoundManifest(NamedTuple):
    """A manifest file found in an artifact."""

    path: Path
    manifest: "PackManifest"


@dataclass
class PackManifest:
    """Represents a .kpm manifest file."""

    format_version: int
    component_name: str
    prefix: str
    kpack_files: dict[str, KpackFileEntry]  # architecture -> entry

    @classmethod
    def from_file(cls, manifest_path: Path) -> "PackManifest":
        """
        Read a .kpm manifest file.

        Args:
            manifest_path: Path to .kpm file

        Returns:
            PackManifest instance

        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ValueError: If manifest format is invalid
        """
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        if manifest_path.stat().st_size == 0:
            raise ValueError(f"Manifest file is empty: {manifest_path}")

        try:
            with open(manifest_path, "rb") as f:
                data = msgpack.unpack(f, raw=False)
        except msgpack.exceptions.UnpackException as e:
            raise ValueError(f"Invalid msgpack format in {manifest_path}: {e}") from e
        except OSError as e:
            raise ValueError(f"Cannot read manifest {manifest_path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Manifest root must be a dict, got {type(data)}")

        # Validate required fields
        if "format_version" not in data:
            raise ValueError("Missing required field: format_version")
        if "component_name" not in data:
            raise ValueError("Missing required field: component_name")
        if "prefix" not in data:
            raise ValueError("Missing required field: prefix")
        if "kpack_files" not in data:
            raise ValueError("Missing required field: kpack_files")

        # Parse kpack file entries
        kpack_entries = {}
        for arch, entry_data in data["kpack_files"].items():
            if not isinstance(entry_data, dict):
                raise ValueError(f"Kpack entry for '{arch}' must be a dict")

            if "file" not in entry_data:
                raise ValueError(f"Kpack entry for '{arch}' missing 'file' field")
            if "size" not in entry_data:
                raise ValueError(f"Kpack entry for '{arch}' missing 'size' field")
            if "kernel_count" not in entry_data:
                raise ValueError(
                    f"Kpack entry for '{arch}' missing 'kernel_count' field"
                )

            kpack_entries[arch] = KpackFileEntry(
                architecture=arch,
                filename=entry_data["file"],
                size=entry_data["size"],
                kernel_count=entry_data["kernel_count"],
            )

        return cls(
            format_version=data["format_version"],
            component_name=data["component_name"],
            prefix=data["prefix"],
            kpack_files=kpack_entries,
        )

    def to_file(self, manifest_path: Path) -> None:
        """
        Write manifest to a .kpm file.

        Args:
            manifest_path: Path where manifest will be written
        """
        # Build kpack_files dict
        kpack_files_dict = {}
        for arch, entry in self.kpack_files.items():
            kpack_files_dict[arch] = {
                "file": entry.filename,
                "size": entry.size,
                "kernel_count": entry.kernel_count,
            }

        data = {
            "format_version": self.format_version,
            "component_name": self.component_name,
            "prefix": self.prefix,
            "kpack_files": kpack_files_dict,
        }

        # Write to file
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "wb") as f:
            msgpack.pack(data, f)

        # Validate output
        if not manifest_path.exists():
            raise RuntimeError(f"Failed to create manifest file: {manifest_path}")
        if manifest_path.stat().st_size == 0:
            raise RuntimeError(f"Created manifest file is empty: {manifest_path}")


class ManifestMerger:
    """
    Merges .kpm manifest files from the map phase.

    This class combines manifests from individual architecture-specific
    artifacts into unified manifests for package groups.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize manifest merger.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def merge_manifests(
        self, manifests: list[PackManifest], component_name: str, prefix: str
    ) -> PackManifest:
        """
        Merge multiple manifests into a single unified manifest.

        Args:
            manifests: List of manifests to merge
            component_name: Component name for the merged manifest
            prefix: Prefix path for the merged manifest

        Returns:
            Merged PackManifest

        Raises:
            ValueError: If manifests have conflicting information
        """
        if not manifests:
            raise ValueError("Cannot merge empty list of manifests")

        if self.verbose:
            print(
                f"  Merging {len(manifests)} manifests for component '{component_name}'"
            )

        # Validate all manifests have same component name
        for manifest in manifests:
            if manifest.component_name != component_name:
                raise ValueError(
                    f"Component name mismatch: expected '{component_name}', "
                    f"got '{manifest.component_name}' in manifest"
                )

        # Merge kpack file entries
        merged_entries: dict[str, KpackFileEntry] = {}

        for manifest in manifests:
            for arch, entry in manifest.kpack_files.items():
                if arch in merged_entries:
                    # Check for conflicts
                    existing = merged_entries[arch]
                    if existing.filename != entry.filename:
                        raise ValueError(
                            f"Conflicting kpack filenames for architecture '{arch}': "
                            f"'{existing.filename}' vs '{entry.filename}'"
                        )
                    if existing.size != entry.size:
                        raise ValueError(
                            f"Conflicting kpack sizes for architecture '{arch}': "
                            f"{existing.size} vs {entry.size}"
                        )
                    if existing.kernel_count != entry.kernel_count:
                        raise ValueError(
                            f"Conflicting kernel counts for architecture '{arch}': "
                            f"{existing.kernel_count} vs {entry.kernel_count}"
                        )
                    # Entries match, skip duplicate
                    continue

                merged_entries[arch] = entry

                if self.verbose:
                    print(
                        f"    Added {arch}: {entry.filename} ({entry.kernel_count} kernels, {entry.size} bytes)"
                    )

        # Create merged manifest
        return PackManifest(
            format_version=1,
            component_name=component_name,
            prefix=prefix,
            kpack_files=merged_entries,
        )

    def find_manifests_in_artifact(
        self, artifact_dir: Path, prefix: str
    ) -> list[FoundManifest]:
        """
        Find all .kpm manifest files in an artifact for a specific prefix.

        Args:
            artifact_dir: Artifact directory to search
            prefix: Prefix path to search in

        Returns:
            List of FoundManifest instances with path and parsed manifest
        """
        results = []

        # .kpm files are located at {artifact}/{prefix}/.kpack/*.kpm
        kpack_dir = artifact_dir / prefix / ".kpack"

        if not kpack_dir.exists():
            return results

        # Find all .kpm files
        for manifest_path in kpack_dir.glob("*.kpm"):
            try:
                manifest = PackManifest.from_file(manifest_path)
                results.append(FoundManifest(path=manifest_path, manifest=manifest))

                if self.verbose:
                    print(
                        f"  Found manifest: {manifest_path.relative_to(artifact_dir)}"
                    )
            except ValueError as e:
                raise RuntimeError(
                    f"Failed to parse manifest {manifest_path}: {e}\n"
                    f"Corrupted manifests indicate incomplete artifacts from map phase"
                ) from e

        return results
