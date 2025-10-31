import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from enum import Enum
from typing import Any

import msgpack


class BinaryType(Enum):
    """Type of bundled binary file."""
    STANDALONE = "standalone"  # .co files - directly in bundler format
    BUNDLED = "bundled"  # Executables/libraries with .hip_fatbin ELF section


class Toolchain:
    """Manages configuration of various toolchain locations."""

    def __init__(
        self,
        *,
        clang_offload_bundler: Path | None = None,
        objcopy: Path | None = None,
        readelf: Path | None = None,
    ):
        self.clang_offload_bundler = self._validate_or_find(
            "clang-offload-bundler", clang_offload_bundler
        )
        self.objcopy = self._validate_or_find("objcopy", objcopy)
        self.readelf = self._validate_or_find("readelf", readelf)

    @staticmethod
    def configure_argparse(p: argparse.ArgumentParser):
        p.add_argument(
            "--clang-offload-bundler", type=Path, help="Path to clang-offload-bundler"
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "Toolchain":
        clang_offload_bundler: Path | None = args.clang_offload_bundler
        return Toolchain(clang_offload_bundler=clang_offload_bundler)

    def _validate_or_find(
        self, tool_file_name: str, explicit_path: Path | None
    ) -> Path:
        if explicit_path is None:
            found_path = shutil.which(tool_file_name)
            if found_path is None:
                raise OSError(f"Could not file tool '{tool_file_name}' on system path")
            explicit_path = Path(found_path)
        if not explicit_path.exists():
            raise OSError(
                f"Tool '{tool_file_name}' at path {explicit_path} does not exist"
            )
        return explicit_path

    def exec_capture_text(self, args: list[str | Path]):
        return subprocess.check_output([str(a) for a in args]).decode()

    def exec(self, args: list[str | Path]):
        subprocess.check_call([str(a) for a in args])


class UnbundledContents:
    """Represents a directory of unbundled contents. This is a context manager that
    will optionally delete the contents on close.
    """

    def __init__(
        self,
        source_binary: "BundledBinary",
        dest_dir: Path,
        delete_on_close: bool,
        target_list: list[tuple[str, str]],
    ):
        self.source_binary = source_binary
        self.dest_dir = dest_dir
        self.delete_on_close = delete_on_close
        self.target_list = target_list

    def __enter__(self) -> "UnbundledContents":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def file_names(self) -> list[str]:
        return [kv[1] for kv in self.target_list]

    def close(self):
        if self.delete_on_close and self.dest_dir.exists():
            shutil.rmtree(self.dest_dir)

    def __repr__(self):
        return f"UnbundledContents(dest_dir={self.dest_dir}, target_list={self.target_list})"


class BundledBinary:
    """Represents a bundled binary at some path.

    Supports two types of bundled binaries:
    1. STANDALONE - Files directly in clang-offload-bundler format (e.g., .co files)
    2. BUNDLED - ELF binaries containing .hip_fatbin section (executables, shared libraries)

    For BUNDLED binaries, the .hip_fatbin section is extracted and treated as
    a bundler-format input.
    """

    def __init__(self, file_path: Path, *, toolchain: Toolchain | None = None):
        self.toolchain = toolchain or Toolchain()
        self.file_path = file_path
        self.binary_type = self._detect_binary_type()
        self._temp_dir: Path | None = None  # For extracted .hip_fatbin sections

    def unbundle(
        self, *, dest_dir: Path | None = None, delete_on_close: bool = True
    ) -> UnbundledContents:
        """Unbundles the binary, returning a context manager which can be used
        to hold the unbundled files open for as long as needed.
        """
        if dest_dir is None:
            dest_dir = Path(tempfile.TemporaryDirectory(delete=False).name)
        target_list = self._list_bundled_targets(self.file_path)
        contents = UnbundledContents(
            self, dest_dir, delete_on_close=delete_on_close, target_list=target_list
        )
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            self._unbundle(
                targets=[kv[0] for kv in target_list],
                outputs=[dest_dir / kv[1] for kv in target_list],
            )
        except:
            contents.close()
            raise
        return contents

    def _detect_binary_type(self) -> BinaryType:
        """Detect if this is a standalone bundler file or bundled ELF binary.

        Uses readelf to check for .hip_fatbin section to determine type.
        Files with .hip_fatbin section are BUNDLED (executables, libraries).
        Files without (or non-ELF files) are STANDALONE (.co files in bundler format).

        Returns:
            BinaryType indicating the file type

        Raises:
            RuntimeError: For unexpected errors during detection
        """
        try:
            result = subprocess.run(
                [str(self.toolchain.readelf), "-S", str(self.file_path.resolve())],
                capture_output=True,
                text=True,
                check=True,  # Raise CalledProcessError on non-zero exit
            )
            # readelf succeeded - this is an ELF file
            # Check for .hip_fatbin section
            if ".hip_fatbin" in result.stdout:
                return BinaryType.BUNDLED
            else:
                # ELF file without .hip_fatbin section
                return BinaryType.STANDALONE

        except subprocess.CalledProcessError:
            # readelf failed - likely not an ELF file
            # Assume STANDALONE (bundler format file like .co)
            return BinaryType.STANDALONE
        except Exception as e:
            # Unexpected error - fail fast
            raise RuntimeError(
                f"Unexpected error detecting binary type for {self.file_path}: {e}"
            )

    def _get_bundler_input(self) -> Path:
        """Get the file path to use as input to clang-offload-bundler.

        For STANDALONE files, returns the file path directly.
        For BUNDLED binaries, extracts the .hip_fatbin section to a temp file.

        Returns:
            Path to file in bundler format
        """
        if self.binary_type == BinaryType.STANDALONE:
            return self.file_path

        # Extract .hip_fatbin section from bundled binary
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp())

        fatbin_path = self._temp_dir / "fatbin.o"
        # Resolve to absolute paths for objcopy
        abs_file_path = self.file_path.resolve()
        abs_fatbin_path = fatbin_path.resolve()

        try:
            self.toolchain.exec(
                [
                    self.toolchain.objcopy,
                    "--dump-section",
                    f".hip_fatbin={abs_fatbin_path}",
                    abs_file_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to extract .hip_fatbin section from {self.file_path}: {e}"
            )

        return fatbin_path

    def _list_bundled_targets(self, file_path: Path) -> list[tuple[str, str]]:
        """Returns a list of (target_name, file_name) for all bundles."""
        bundler_input = self._get_bundler_input()
        lines = (
            self.toolchain.exec_capture_text(
                [
                    self.toolchain.clang_offload_bundler,
                    "--list",
                    "--type=o",
                    f"--input={bundler_input}",
                ]
            )
            .strip()
            .splitlines()
        )

        def _map(target_name: str) -> str:
            if target_name.startswith("host"):
                return f"{target_name}.elf"
            elif target_name.startswith("hip"):
                return f"{target_name}.hsaco"
            raise ValueError(f"Unexpected unbundled target name {target_name}")

        return [
            (target_name, _map(target_name)) for target_name in lines if target_name
        ]

    def _unbundle(self, *, targets: list[str], outputs: list[Path]):
        """Unbundle targets from the binary.

        Args:
            targets: List of target names to unbundle
            outputs: List of output paths (must match length of targets)
        """
        if not targets:
            return

        bundler_input = self._get_bundler_input()
        args = [
            self.toolchain.clang_offload_bundler,
            "--unbundle",
            "--type=o",
            f"--input={bundler_input}",
            f"--targets={','.join(targets)}",
        ]
        for output in outputs:
            args.extend(["--output", output])
        self.toolchain.exec(args)

    def list_bundles(self) -> list[str]:
        """List all architecture bundles in the binary.

        Returns:
            List of architecture strings (e.g., ['gfx1100', 'gfx1101'])
            Only returns GPU architectures, not host bundles.
        """
        target_list = self._list_bundled_targets(self.file_path)
        architectures = []
        for target_name, _ in target_list:
            # Extract architecture from target names like:
            # "hipv4-amdgcn-amd-amdhsa--gfx1100" -> "gfx1100"
            if target_name.startswith("hip"):
                parts = target_name.split("--")
                if len(parts) >= 2:
                    architectures.append(parts[-1])
        return architectures

    def create_host_only(self, output_path: Path) -> None:
        """Create a host-only version of the binary without GPU device code.

        Only supported for BUNDLED binaries (executables/libraries with .hip_fatbin).
        Removes the .hip_fatbin section to create a host-only version.

        Args:
            output_path: Path where host-only binary will be written

        Raises:
            RuntimeError: If operation fails or called on STANDALONE binary
        """
        if self.binary_type == BinaryType.STANDALONE:
            raise RuntimeError(
                f"create_host_only() not supported for STANDALONE binaries. "
                f"File {self.file_path} does not have a .hip_fatbin section to remove. "
                f"If you need to extract the host bundle from a .co file, use unbundle() instead."
            )

        # For BUNDLED binaries, remove .hip_fatbin section
        try:
            self.toolchain.exec(
                [
                    self.toolchain.objcopy,
                    "--remove-section",
                    ".hip_fatbin",
                    self.file_path,
                    output_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to remove .hip_fatbin section from {self.file_path}: {e}"
            )

    def cleanup(self) -> None:
        """Clean up temporary files created during operations."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


def add_kpack_ref_marker(
    binary_path: Path,
    output_path: Path,
    kpack_search_paths: list[str],
    kernel_name: str,
    *,
    toolchain: Toolchain | None = None,
) -> None:
    """Add .rocm_kpack_ref marker section to a binary.

    The marker section contains a MessagePack structure pointing to kpack files
    and identifying the kernel name for TOC lookup.

    Args:
        binary_path: Path to input binary (ELF executable or shared library)
        output_path: Path where marked binary will be written
        kpack_search_paths: List of kpack file paths relative to binary location
        kernel_name: Kernel identifier for TOC lookup in kpack file
        toolchain: Toolchain instance (created if not provided)

    Raises:
        RuntimeError: If objcopy fails to add section
    """
    if toolchain is None:
        toolchain = Toolchain()

    # Create marker structure
    marker_data = {
        "kpack_search_paths": kpack_search_paths,
        "kernel_name": kernel_name,
    }

    # Serialize to MessagePack
    marker_bytes = msgpack.packb(marker_data, use_bin_type=True)

    # Write marker to temporary file and add section
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".marker") as f:
        f.write(marker_bytes)
        f.flush()

        # Add section to binary using objcopy
        # Use absolute paths to avoid objcopy path resolution issues
        abs_binary_path = binary_path.resolve()
        abs_output_path = output_path.resolve()
        abs_marker_file = Path(f.name).resolve()

        try:
            toolchain.exec(
                [
                    toolchain.objcopy,
                    "--add-section",
                    f".rocm_kpack_ref={abs_marker_file}",
                    abs_binary_path,
                    abs_output_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to add .rocm_kpack_ref section to {binary_path}: {e}"
            )


def read_kpack_ref_marker(
    binary_path: Path,
    *,
    toolchain: Toolchain | None = None,
) -> dict[str, Any] | None:
    """Read .rocm_kpack_ref marker section from a binary.

    Args:
        binary_path: Path to binary with marker section
        toolchain: Toolchain instance (created if not provided)

    Returns:
        Marker data dictionary, or None if section doesn't exist

    Raises:
        RuntimeError: If readelf fails or section exists but cannot be read or parsed
    """
    if toolchain is None:
        toolchain = Toolchain()

    # Check if section exists using readelf
    result = subprocess.run(
        [str(toolchain.readelf), "-S", str(binary_path.resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    if ".rocm_kpack_ref" not in result.stdout:
        return None

    # Extract section to temporary file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".marker") as f:
        abs_binary_path = binary_path.resolve()
        abs_marker_file = Path(f.name).resolve()

        try:
            toolchain.exec(
                [
                    toolchain.objcopy,
                    "--dump-section",
                    f".rocm_kpack_ref={abs_marker_file}",
                    abs_binary_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to extract .rocm_kpack_ref section from {binary_path}: {e}"
            )

        # Read and parse MessagePack data
        try:
            with open(abs_marker_file, "rb") as marker_file:
                marker_bytes = marker_file.read()
                marker_data = msgpack.unpackb(marker_bytes, raw=False)
                return marker_data
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse .rocm_kpack_ref marker data from {binary_path}: {e}"
            )
