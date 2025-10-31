"""CCOB (Clang Code Object Bundle) parser and decompressor.

This module correctly handles compressed CCOB bundles by respecting the
totalSize field in the header, avoiding the bug in clang-offload-bundler
that reads beyond the bundle boundary.

This is a workaround for an incomplete fix in clang/lib/Driver/OffloadBundler.cpp
from commit efda523188c4. The llvm/lib/Object/OffloadBundle.cpp version was fixed
but the clang version was not, causing failures on ROCm libraries like librocblas.

Issue: https://github.com/ROCm/llvm-project/issues/448
Reference: llvm/lib/Object/OffloadBundle.cpp (fixed implementation)
"""

import struct
import zstandard as zstd
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile


@dataclass
class CCOBHeader:
    """Parsed CCOB header.

    Attributes:
        magic: Magic string (should be "CCOB")
        version: Format version (e.g., 0x00010003 for version 1.3)
        compression_method: Compression type (1 = zstd)
        total_size: Total size of this CCOB bundle (header + compressed data)
        uncompressed_size: Size after decompression
        hash: Truncated MD5 hash
    """
    magic: str
    version: int
    compression_method: int
    total_size: int
    uncompressed_size: int
    hash: int

    @classmethod
    def parse(cls, data: bytes) -> "CCOBHeader":
        """Parse CCOB header from bytes.

        Header format (32 bytes for version 1.3):
          Offset | Size | Field
          -------|------|------
          0x00   | 4    | Magic ("CCOB")
          0x04   | 2    | Version (uint16_t)
          0x06   | 2    | Compression method (uint16_t)
          0x08   | 8    | Total size (uint64_t)
          0x10   | 8    | Uncompressed size (uint64_t)
          0x18   | 8    | Hash (uint64_t)

        Args:
            data: At least 32 bytes starting with CCOB header

        Returns:
            Parsed CCOBHeader

        Raises:
            ValueError: If magic is not "CCOB" or header is malformed
        """
        if len(data) < 32:
            raise ValueError(f"Header too short: {len(data)} bytes (need 32)")

        magic = data[0:4].decode('ascii', errors='replace')
        if magic != "CCOB":
            raise ValueError(f"Invalid magic: {magic!r} (expected 'CCOB')")

        # Parse version to determine header format
        version_raw = struct.unpack('<I', data[4:8])[0]

        # Extract version number from lower 16 bits
        version = version_raw & 0xFFFF
        compression_method = (version_raw >> 16) & 0xFFFF

        # Version 1.3 uses 64-bit sizes
        if version == 3:
            total_size = struct.unpack('<Q', data[8:16])[0]
            uncompressed_size = struct.unpack('<Q', data[16:24])[0]
            hash_val = struct.unpack('<Q', data[24:32])[0]
        else:
            # Version 1.2 uses 32-bit sizes
            total_size = struct.unpack('<I', data[8:12])[0]
            uncompressed_size = struct.unpack('<I', data[12:16])[0]
            hash_val = struct.unpack('<Q', data[16:24])[0]

        return cls(
            magic=magic,
            version=version,
            compression_method=compression_method,
            total_size=total_size,
            uncompressed_size=uncompressed_size,
            hash=hash_val,
        )


@dataclass
class BundleEntry:
    """Entry in uncompressed bundle descriptor.

    Attributes:
        offset: Offset in bundle data where this code object starts
        size: Size of this code object in bytes
        triple_size: Size of triple string
        triple: Target triple (e.g., "hipv4-amdgcn-amd-amdhsa--gfx1100")
    """
    offset: int
    size: int
    triple_size: int
    triple: str


@dataclass
class UncompressedBundle:
    """Parsed uncompressed bundle.

    Attributes:
        magic: Magic string (should be "__CLANG_OFFLOAD_BUNDLE__")
        num_entries: Number of bundle entries
        entries: List of bundle entries
        data: Full uncompressed data blob
    """
    magic: str
    num_entries: int
    entries: list[BundleEntry]
    data: bytes

    @classmethod
    def parse(cls, data: bytes) -> "UncompressedBundle":
        """Parse uncompressed bundle format.

        Format:
          Offset | Size | Field
          -------|------|------
          0x00   | 24   | Magic ("__CLANG_OFFLOAD_BUNDLE__")
          0x18   | 8    | Number of entries (uint64_t)
          0x20+  | var  | Array of entry descriptors

        Entry descriptor:
          Offset | Size | Field
          -------|------|------
          0x00   | 8    | Offset (uint64_t)
          0x08   | 8    | Size (uint64_t)
          0x10   | 8    | Triple size (uint64_t)
          0x18   | var  | Triple string (not null-terminated)

        Args:
            data: Decompressed bundle data

        Returns:
            Parsed UncompressedBundle

        Raises:
            ValueError: If magic is wrong or format is invalid
        """
        if len(data) < 32:
            raise ValueError(f"Bundle too short: {len(data)} bytes")

        magic = data[0:24].rstrip(b'\x00').decode('ascii', errors='replace')
        if not magic.startswith("__CLANG_OFFLOAD_BUNDLE__"):
            raise ValueError(f"Invalid bundle magic: {magic!r}")

        num_entries = struct.unpack('<Q', data[24:32])[0]

        entries = []
        pos = 32

        for i in range(num_entries):
            if pos + 24 > len(data):
                raise ValueError(f"Entry {i} header truncated at offset {pos}")

            offset = struct.unpack('<Q', data[pos:pos+8])[0]
            size = struct.unpack('<Q', data[pos+8:pos+16])[0]
            triple_size = struct.unpack('<Q', data[pos+16:pos+24])[0]
            pos += 24

            if pos + triple_size > len(data):
                raise ValueError(f"Entry {i} triple truncated at offset {pos}")

            triple = data[pos:pos+triple_size].decode('ascii', errors='replace')
            pos += triple_size

            entries.append(BundleEntry(
                offset=offset,
                size=size,
                triple_size=triple_size,
                triple=triple,
            ))

        return cls(
            magic=magic,
            num_entries=num_entries,
            entries=entries,
            data=data,
        )

    def get_code_object(self, triple: str) -> bytes | None:
        """Get code object for a specific target triple.

        Args:
            triple: Target triple to search for

        Returns:
            Code object bytes, or None if not found
        """
        for entry in self.entries:
            if entry.triple == triple:
                return self.data[entry.offset:entry.offset + entry.size]
        return None

    def list_triples(self) -> list[str]:
        """Get list of all target triples in this bundle.

        Returns:
            List of triple strings
        """
        return [entry.triple for entry in self.entries]


def decompress_ccob(data: bytes) -> bytes:
    """Decompress CCOB bundle data.

    This correctly respects the totalSize field in the CCOB header,
    avoiding the bug in clang-offload-bundler that reads beyond the
    bundle boundary.

    Args:
        data: CCOB bundle data (header + compressed data)

    Returns:
        Decompressed bundle data

    Raises:
        ValueError: If header is invalid or decompression fails
    """
    header = CCOBHeader.parse(data)

    # Key fix: Only read totalSize bytes, not to end of buffer
    if len(data) < header.total_size:
        raise ValueError(
            f"Data too short: {len(data)} bytes, header says {header.total_size}"
        )

    # Extract compressed data using totalSize (not reading to EOF!)
    header_size = 32
    compressed_data = data[header_size:header.total_size]

    # Decompress using zstd
    if header.compression_method != 1:
        raise ValueError(f"Unsupported compression method: {header.compression_method}")

    dctx = zstd.ZstdDecompressor()
    try:
        decompressed = dctx.decompress(
            compressed_data,
            max_output_size=header.uncompressed_size
        )
    except zstd.ZstdError as e:
        raise ValueError(f"Decompression failed: {e}") from e

    # Verify size
    if len(decompressed) != header.uncompressed_size:
        raise ValueError(
            f"Size mismatch: decompressed {len(decompressed)} bytes, "
            f"expected {header.uncompressed_size}"
        )

    return decompressed


def parse_ccob_file(path: Path) -> UncompressedBundle:
    """Parse and decompress a CCOB bundle file.

    Args:
        path: Path to CCOB bundle file

    Returns:
        Parsed uncompressed bundle

    Raises:
        ValueError: If file is invalid or decompression fails
    """
    data = path.read_bytes()
    decompressed = decompress_ccob(data)
    return UncompressedBundle.parse(decompressed)


def extract_ccob_from_binary(binary_path: Path, output_dir: Path) -> dict[str, Path]:
    """Extract code objects from a binary's .hip_fatbin section.

    This is a convenience function that:
    1. Extracts .hip_fatbin section using objcopy
    2. Decompresses the CCOB bundle
    3. Extracts individual code objects to files

    Args:
        binary_path: Path to binary with .hip_fatbin section
        output_dir: Directory to write extracted code objects

    Returns:
        Dict mapping target triples to extracted file paths

    Raises:
        ValueError: If extraction or parsing fails
        subprocess.CalledProcessError: If objcopy fails
    """
    # Extract .hip_fatbin section
    with tempfile.NamedTemporaryFile(suffix='.fatbin', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            ['objcopy', '--dump-section', f'.hip_fatbin={tmp_path}', str(binary_path)],
            check=True,
            capture_output=True,
        )

        # Parse and extract
        bundle = parse_ccob_file(tmp_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = {}

        for entry in bundle.entries:
            # Create filename from triple
            triple = entry.triple
            # Remove host entries
            if 'host-' in triple:
                continue

            # Extract architecture (e.g., gfx1100 from hipv4-amdgcn-amd-amdhsa--gfx1100)
            parts = triple.split('--')
            if len(parts) >= 2:
                arch = parts[-1]
            else:
                arch = triple.replace('/', '_').replace('-', '_')

            output_file = output_dir / f"{arch}.hsaco"
            code_obj = bundle.get_code_object(triple)
            if code_obj:
                output_file.write_bytes(code_obj)
                extracted[triple] = output_file

        return extracted

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def list_ccob_targets(data: bytes) -> list[str]:
    """List target triples in a CCOB bundle.

    Args:
        data: CCOB bundle data

    Returns:
        List of target triple strings

    Raises:
        ValueError: If parsing fails
    """
    decompressed = decompress_ccob(data)
    bundle = UncompressedBundle.parse(decompressed)
    return bundle.list_triples()
