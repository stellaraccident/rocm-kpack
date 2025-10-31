"""Packed kernel archive format for efficient kernel storage and distribution.

This module provides the PackedKernelArchive class for creating and reading
.kpack files - binary archives with aligned blobs and MessagePack TOC.
"""

import struct
from pathlib import Path
from typing import Any

import msgpack


class PackedKernelArchive:
    """A packed kernel archive containing kernels for a specific architecture family.

    File format optimized for fast reads and memory-mapping:

    [Fixed Binary Header - 16 bytes]
      Magic:      "KPAK" (4 bytes)
      Version:    uint32 (4 bytes)
      TOC Offset: uint64 (8 bytes)

    [Padding to 64-byte boundary]

    [Blob 0 @ 64-byte aligned offset]
    [Padding]
    [Blob 1 @ 64-byte aligned offset]
    ...

    [MessagePack TOC at TOC Offset]
    {
      "group_name": "blas",
      "gfx_arch_family": "gfx100X",
      "gfx_arches": [...],
      "toc": {
        "bin/hipcc": {
          "gfx1030": {
            "type": "hsaco",
            "offset": 64,      # offset from start of file
            "size": 7472       # blob size in bytes
          }
        }
      }
    }
    """

    MAGIC = b"KPAK"
    FORMAT_VERSION = 1
    HEADER_SIZE = 16  # 4 (magic) + 4 (version) + 8 (toc_offset)
    BLOB_ALIGNMENT = 64

    def __init__(
        self,
        group_name: str,
        gfx_arch_family: str,
        gfx_arches: list[str],
        output_path: Path | None = None,
    ):
        """Initialize a new packed kernel archive.

        Args:
            group_name: Name of the build slice (e.g., "blas", "torch")
            gfx_arch_family: Architecture family identifier (e.g., "gfx1100", "gfx100X")
            gfx_arches: List of actual architectures in this family
            output_path: If provided, enables streaming write mode where blobs are
                        written to disk immediately. Call finalize() when done.
                        If None, accumulates in memory until write() is called.
        """
        self.group_name = group_name
        self.gfx_arch_family = gfx_arch_family
        self.gfx_arches = gfx_arches
        # TOC: binary_path -> gfx_arch -> entry (with offset/size)
        self.toc: dict[str, dict[str, dict[str, Any]]] = {}

        # In-memory mode: accumulate blobs (backward compatibility)
        self.data: list[bytes] = []

        # Streaming write mode
        self._output_path: Path | None = output_path
        self._file_handle: Any = None  # Open file handle for streaming writes

        # File path for read-mode archives
        self._file_path: Path | None = None

        # Initialize streaming write mode if output_path provided
        if output_path:
            self._begin_streaming_write()

    def _begin_streaming_write(self) -> None:
        """Initialize streaming write mode by opening file and writing header."""
        assert self._output_path is not None
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        self._file_handle = self._output_path.open("wb")

        # Write header with placeholder TOC offset
        header = struct.pack(
            "<4sIQ",  # little-endian: 4-byte string, uint32, uint64
            self.MAGIC,
            self.FORMAT_VERSION,
            0,  # TOC offset placeholder
        )
        self._file_handle.write(header)

        # Pad to first blob alignment boundary
        current_pos = self._file_handle.tell()
        padding = (self.BLOB_ALIGNMENT - (current_pos % self.BLOB_ALIGNMENT)) % self.BLOB_ALIGNMENT
        self._file_handle.write(b"\x00" * padding)

    def add_kernel(
        self,
        relative_path: str,
        gfx_arch: str,
        hsaco_data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a kernel to the archive.

        In streaming mode (output_path provided to constructor), writes blob
        immediately to disk. Otherwise accumulates in memory.

        Args:
            relative_path: Path to binary relative to install tree root
            gfx_arch: GPU architecture (e.g., "gfx1100")
            hsaco_data: Raw HSACO kernel data
            metadata: Optional metadata dictionary for extensibility

        Raises:
            ValueError: If kernel already exists for (relative_path, gfx_arch)
        """
        # Normalize path to use forward slashes (replace backslashes)
        relative_path = relative_path.replace("\\", "/")

        # Check for duplicates
        if relative_path in self.toc:
            if gfx_arch in self.toc[relative_path]:
                raise ValueError(
                    f"Kernel already exists for {relative_path} @ {gfx_arch}"
                )

        # Add to TOC
        if relative_path not in self.toc:
            self.toc[relative_path] = {}

        entry = {
            "type": "hsaco",
        }
        if metadata:
            entry["metadata"] = metadata

        # Streaming mode: write blob to disk immediately
        if self._file_handle is not None:
            offset = self._file_handle.tell()
            self._file_handle.write(hsaco_data)

            # Update TOC with offset and size
            entry["offset"] = offset
            entry["size"] = len(hsaco_data)

            # Pad to next alignment boundary
            current_pos = self._file_handle.tell()
            padding = (self.BLOB_ALIGNMENT - (current_pos % self.BLOB_ALIGNMENT)) % self.BLOB_ALIGNMENT
            self._file_handle.write(b"\x00" * padding)
        else:
            # In-memory mode: accumulate in data array with ordinal
            ordinal = len(self.data)
            self.data.append(hsaco_data)
            entry["ordinal"] = ordinal

        self.toc[relative_path][gfx_arch] = entry

    def write(self, output_path: Path) -> None:
        """Write the packed archive to a file (in-memory mode only).

        This method is for backward compatibility when output_path is not
        provided to constructor. For streaming mode, use finalize() instead.

        Format:
        1. Write fixed header (magic, version, toc_offset placeholder)
        2. Pad to BLOB_ALIGNMENT boundary
        3. Write each blob aligned to BLOB_ALIGNMENT, tracking offsets
        4. Write MessagePack TOC at end
        5. Seek back and update header with TOC offset

        Args:
            output_path: Path where .kpack file will be written

        Raises:
            RuntimeError: If called in streaming mode (use finalize() instead)
        """
        if self._file_handle is not None:
            raise RuntimeError(
                "Cannot call write() in streaming mode. Use finalize() instead."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as f:
            # Write header with placeholder TOC offset
            header = struct.pack(
                "<4sIQ",  # little-endian: 4-byte string, uint32, uint64
                self.MAGIC,
                self.FORMAT_VERSION,
                0,  # TOC offset placeholder
            )
            f.write(header)

            # Pad to first blob alignment boundary
            current_pos = f.tell()
            padding = (self.BLOB_ALIGNMENT - (current_pos % self.BLOB_ALIGNMENT)) % self.BLOB_ALIGNMENT
            f.write(b"\x00" * padding)

            # Write blobs and update TOC with offsets
            for relative_path, arches in self.toc.items():
                for gfx_arch, entry in arches.items():
                    ordinal = entry["ordinal"]
                    blob_data = self.data[ordinal]

                    # Record offset before writing
                    offset = f.tell()
                    f.write(blob_data)

                    # Update TOC entry with offset and size
                    entry["offset"] = offset
                    entry["size"] = len(blob_data)
                    # Remove ordinal from TOC (no longer needed)
                    del entry["ordinal"]

                    # Pad to next alignment boundary
                    current_pos = f.tell()
                    padding = (self.BLOB_ALIGNMENT - (current_pos % self.BLOB_ALIGNMENT)) % self.BLOB_ALIGNMENT
                    f.write(b"\x00" * padding)

            # Write MessagePack TOC
            toc_offset = f.tell()
            toc_data = {
                "format_version": self.FORMAT_VERSION,
                "group_name": self.group_name,
                "gfx_arch_family": self.gfx_arch_family,
                "gfx_arches": self.gfx_arches,
                "toc": self.toc,
            }
            msgpack.pack(toc_data, f, use_bin_type=True)

            # Backpatch header with TOC offset
            f.seek(8)  # Skip magic (4 bytes) and version (4 bytes)
            f.write(struct.pack("<Q", toc_offset))

    def finalize(self) -> None:
        """Finalize streaming write by writing TOC and closing file.

        Only valid in streaming mode (when output_path provided to constructor).

        Raises:
            RuntimeError: If not in streaming mode
        """
        if self._file_handle is None:
            raise RuntimeError(
                "Not in streaming mode. Use write(output_path) for in-memory mode."
            )

        # Write MessagePack TOC
        toc_offset = self._file_handle.tell()
        toc_data = {
            "format_version": self.FORMAT_VERSION,
            "group_name": self.group_name,
            "gfx_arch_family": self.gfx_arch_family,
            "gfx_arches": self.gfx_arches,
            "toc": self.toc,
        }
        msgpack.pack(toc_data, self._file_handle, use_bin_type=True)

        # Backpatch header with TOC offset
        self._file_handle.seek(8)  # Skip magic (4 bytes) and version (4 bytes)
        self._file_handle.write(struct.pack("<Q", toc_offset))

        # Close file
        self._file_handle.close()
        self._file_handle = None

    @staticmethod
    def read(input_path: Path) -> "PackedKernelArchive":
        """Read a packed archive from a file.

        Reads the binary header, seeks to the TOC, and loads metadata.
        Kernel data remains in the file and is accessed on-demand.

        Args:
            input_path: Path to .kpack file

        Returns:
            PackedKernelArchive instance loaded from file

        Raises:
            ValueError: If magic number or format version is invalid
        """
        with input_path.open("rb") as f:
            # Read and validate header
            header_bytes = f.read(PackedKernelArchive.HEADER_SIZE)
            magic, version, toc_offset = struct.unpack("<4sIQ", header_bytes)

            if magic != PackedKernelArchive.MAGIC:
                raise ValueError(
                    f"Invalid magic number: {magic!r} (expected {PackedKernelArchive.MAGIC!r})"
                )

            if version != PackedKernelArchive.FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported format version: {version} (expected {PackedKernelArchive.FORMAT_VERSION})"
                )

            # Seek to TOC and load it
            f.seek(toc_offset)
            toc_data = msgpack.unpack(f, raw=False)

        # Create archive instance
        archive = PackedKernelArchive(
            group_name=toc_data["group_name"],
            gfx_arch_family=toc_data["gfx_arch_family"],
            gfx_arches=toc_data["gfx_arches"],
        )
        archive.toc = toc_data["toc"]
        archive._file_path = input_path

        return archive

    def get_kernel(self, relative_path: str, gfx_arch: str) -> bytes | None:
        """Retrieve kernel data for a specific binary and architecture.

        Args:
            relative_path: Path to binary relative to install tree root
            gfx_arch: GPU architecture

        Returns:
            Kernel data bytes, or None if not found
        """
        # Normalize path to use forward slashes (replace backslashes)
        relative_path = relative_path.replace("\\", "/")

        if relative_path not in self.toc:
            return None
        if gfx_arch not in self.toc[relative_path]:
            return None

        entry = self.toc[relative_path][gfx_arch]

        # Building mode: use ordinal to access in-memory data array
        if "ordinal" in entry:
            ordinal = entry["ordinal"]
            return self.data[ordinal]

        # Read mode: use offset/size to read from file
        if self._file_path is None:
            raise RuntimeError(
                "Archive is in read mode but no file path is set. "
                "This should not happen - file may have been written but not properly loaded."
            )

        offset = entry["offset"]
        size = entry["size"]

        with self._file_path.open("rb") as f:
            f.seek(offset)
            return f.read(size)

    @staticmethod
    def compute_pack_filename(group_name: str, gfx_arch_family: str) -> str:
        """Compute the standard filename for a pack file.

        Args:
            group_name: Build slice name
            gfx_arch_family: Architecture family

        Returns:
            Filename like "blas-gfx100X.kpack"
        """
        return f"{group_name}-{gfx_arch_family}.kpack"

    def __repr__(self) -> str:
        """String representation of the archive."""
        num_binaries = len(self.toc)
        num_kernels = sum(len(arches) for arches in self.toc.values())
        return (
            f"PackedKernelArchive("
            f"group={self.group_name}, "
            f"family={self.gfx_arch_family}, "
            f"binaries={num_binaries}, "
            f"kernels={num_kernels})"
        )
