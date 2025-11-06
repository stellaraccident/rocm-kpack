#!/usr/bin/env python3
"""
Conservative zero-page optimizer for ELF binaries.

This tool applies zero-page optimization to sections while being conservative about
alignment constraints. It only zero-pages fully aligned regions and preserves
unaligned content.

Key principle: It's better to leave some data in the file than to corrupt the binary.
"""

import struct
import sys
from pathlib import Path
from typing import NamedTuple, List, Optional, Tuple


PAGE_SIZE = 0x1000  # 4KB pages


class Elf64_Ehdr(NamedTuple):
    """ELF64 file header (minimal fields we need)"""
    e_phoff: int
    e_shoff: int
    e_phnum: int
    e_shnum: int
    e_shstrndx: int


class Elf64_Phdr(NamedTuple):
    """ELF64 program header"""
    p_type: int
    p_flags: int
    p_offset: int
    p_vaddr: int
    p_paddr: int
    p_filesz: int
    p_memsz: int
    p_align: int


class Elf64_Shdr(NamedTuple):
    """ELF64 section header"""
    sh_name: int
    sh_type: int
    sh_flags: int
    sh_addr: int
    sh_offset: int
    sh_size: int
    sh_link: int
    sh_info: int
    sh_addralign: int
    sh_entsize: int


# ELF constants
PT_LOAD = 1
PT_NOTE = 4
PT_PHDR = 6
PT_GNU_STACK = 0x6474e551
SHT_NOBITS = 8


def read_elf_header(data: bytes) -> Elf64_Ehdr:
    """Read minimal ELF header fields."""
    if data[:4] != b'\x7fELF':
        raise ValueError("Not an ELF file")

    e_phoff = struct.unpack_from('<Q', data, 32)[0]
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_phnum = struct.unpack_from('<H', data, 56)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]

    return Elf64_Ehdr(e_phoff, e_shoff, e_phnum, e_shnum, e_shstrndx)


def read_program_header(data: bytes, offset: int) -> Elf64_Phdr:
    """Read a program header."""
    values = struct.unpack_from('<IIQQQQQQ', data, offset)
    return Elf64_Phdr(*values)


def write_program_header(data: bytearray, offset: int, phdr: Elf64_Phdr):
    """Write a program header."""
    struct.pack_into('<IIQQQQQQ', data, offset,
                     phdr.p_type, phdr.p_flags, phdr.p_offset,
                     phdr.p_vaddr, phdr.p_paddr, phdr.p_filesz,
                     phdr.p_memsz, phdr.p_align)


def read_section_header(data: bytes, offset: int) -> Elf64_Shdr:
    """Read a section header."""
    values = struct.unpack_from('<IIQQQQIIQQ', data, offset)
    return Elf64_Shdr(*values)


def write_section_header(data: bytearray, offset: int, shdr: Elf64_Shdr):
    """Write a section header."""
    struct.pack_into('<IIQQQQIIQQ', data, offset,
                     shdr.sh_name, shdr.sh_type, shdr.sh_flags,
                     shdr.sh_addr, shdr.sh_offset, shdr.sh_size,
                     shdr.sh_link, shdr.sh_info, shdr.sh_addralign,
                     shdr.sh_entsize)


def get_section_name(data: bytes, shstrtab_offset: int, name_idx: int) -> str:
    """Get section name from string table."""
    name_offset = shstrtab_offset + name_idx
    end = data.find(b'\x00', name_offset)
    if end == -1:
        return ""
    return data[name_offset:end].decode('ascii', errors='ignore')


def round_up_to_page(addr: int) -> int:
    """Round address up to next page boundary."""
    return (addr + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)


def round_down_to_page(addr: int) -> int:
    """Round address down to previous page boundary."""
    return addr & ~(PAGE_SIZE - 1)


def calculate_aligned_range(start: int, size: int) -> Tuple[int, int, int]:
    """
    Calculate the page-aligned range within a section.

    Returns: (aligned_start, aligned_size, saved_bytes)
    """
    end = start + size

    # Round up start to next page boundary
    aligned_start = round_up_to_page(start)

    # Round down end to previous page boundary
    aligned_end = round_down_to_page(end)

    if aligned_start >= aligned_end:
        # No full pages to zero
        return (0, 0, 0)

    aligned_size = aligned_end - aligned_start
    return (aligned_start, aligned_size, aligned_size)


def conservative_zero_page(
    input_path: Path,
    output_path: Path,
    section_name: str = ".hip_fatbin",
    verbose: bool = True,
    force_overflow: bool = False
) -> bool:
    """
    Apply conservative zero-page optimization to a section.

    Only zero-pages fully page-aligned regions. Preserves unaligned content.

    Args:
        input_path: Input ELF binary
        output_path: Output ELF binary
        section_name: Section to zero-page (default: .hip_fatbin)
        verbose: Print progress information
        force_overflow: Test mode - artificially reduce available space to force overflow
    """

    # Read input
    data = bytearray(input_path.read_bytes())
    original_size = len(data)

    try:
        ehdr = read_elf_header(data)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False

    if verbose:
        print(f"Processing: {input_path}")
        print(f"  File size: {original_size:,} bytes")

    # Find target section
    shstrtab_shdr = read_section_header(data, ehdr.e_shoff + ehdr.e_shstrndx * 64)
    shstrtab_offset = shstrtab_shdr.sh_offset

    target_section_idx = None
    section_offset = None
    section_vaddr = None
    section_size = None

    for i in range(ehdr.e_shnum):
        shdr = read_section_header(data, ehdr.e_shoff + i * 64)
        name = get_section_name(data, shstrtab_offset, shdr.sh_name)

        if name == section_name:
            target_section_idx = i
            section_offset = shdr.sh_offset
            section_vaddr = shdr.sh_addr
            section_size = shdr.sh_size
            break

    if target_section_idx is None:
        print(f"ERROR: Section '{section_name}' not found", file=sys.stderr)
        return False

    if verbose:
        print(f"\nFound section '{section_name}':")
        print(f"  File offset: 0x{section_offset:x}")
        print(f"  Virtual addr: 0x{section_vaddr:x}")
        print(f"  Size: 0x{section_size:x} ({section_size:,} bytes)")

    # Calculate aligned range
    aligned_vaddr, aligned_size, saved_bytes = calculate_aligned_range(
        section_vaddr, section_size
    )

    if aligned_size == 0:
        print("\nWARNING: Section too small or misaligned - no full pages to zero",
              file=sys.stderr)
        print(f"  Section range: [0x{section_vaddr:x}, 0x{section_vaddr + section_size:x})",
              file=sys.stderr)
        return False

    aligned_offset = section_offset + (aligned_vaddr - section_vaddr)
    section_end_vaddr = section_vaddr + section_size
    aligned_end_vaddr = aligned_vaddr + aligned_size

    if verbose:
        print(f"\nConservative zero-page analysis:")
        print(f"  Total section: [0x{section_vaddr:x}, 0x{section_end_vaddr:x})")
        print(f"  Aligned range: [0x{aligned_vaddr:x}, 0x{aligned_end_vaddr:x})")
        print(f"  Aligned size: 0x{aligned_size:x} ({aligned_size:,} bytes)")
        print(f"  Pages to zero: {aligned_size // PAGE_SIZE}")

        if section_vaddr != aligned_vaddr:
            prefix_size = aligned_vaddr - section_vaddr
            print(f"  Prefix kept: 0x{prefix_size:x} ({prefix_size} bytes)")

        if section_end_vaddr != aligned_end_vaddr:
            suffix_size = section_end_vaddr - aligned_end_vaddr
            print(f"  Suffix kept: 0x{suffix_size:x} ({suffix_size} bytes)")

    # Find the PT_LOAD containing the section
    target_load_idx = None
    target_load = None

    for i in range(ehdr.e_phnum):
        phdr = read_program_header(data, ehdr.e_phoff + i * 56)
        if phdr.p_type == PT_LOAD:
            if (phdr.p_vaddr <= section_vaddr <
                phdr.p_vaddr + phdr.p_memsz):
                target_load_idx = i
                target_load = phdr
                break

    if target_load is None:
        print("ERROR: Could not find PT_LOAD containing section", file=sys.stderr)
        return False

    if verbose:
        print(f"\nFound containing PT_LOAD at index {target_load_idx}:")
        print(f"  vaddr: 0x{target_load.p_vaddr:x}, memsz: 0x{target_load.p_memsz:x}")

    # Remove the aligned bytes from file
    new_data = bytearray(data[:aligned_offset])
    new_data.extend(data[aligned_offset + aligned_size:])

    if verbose:
        print(f"\nRemoving aligned region from file:")
        print(f"  Original size: {original_size:,} bytes")
        print(f"  New size: {len(new_data):,} bytes")
        print(f"  Saved: {saved_bytes:,} bytes ({100*saved_bytes/original_size:.1f}%)")

    # Now we need to split the PT_LOAD into up to 5 segments:
    # 1. Content before section (if any)
    # 2. Prefix of section before aligned region (if unaligned start)
    # 3. Aligned region (zero-page)
    # 4. Suffix of section after aligned region (if unaligned end)
    # 5. Content after section (if any)

    new_phdrs = []

    # Helper to create a PT_LOAD segment
    def make_load(vaddr, vsize, offset, fsize, flags=target_load.p_flags):
        return Elf64_Phdr(
            p_type=PT_LOAD,
            p_flags=flags,
            p_offset=offset,
            p_vaddr=vaddr,
            p_paddr=vaddr,
            p_filesz=fsize,
            p_memsz=vsize,
            p_align=target_load.p_align
        )

    # Calculate all the pieces
    for i in range(ehdr.e_phnum):
        phdr = read_program_header(data, ehdr.e_phoff + i * 56)

        if i != target_load_idx:
            # Not the target load - adjust offset if after removed bytes
            if phdr.p_offset > aligned_offset:
                phdr = Elf64_Phdr(
                    phdr.p_type, phdr.p_flags,
                    phdr.p_offset - aligned_size,
                    phdr.p_vaddr, phdr.p_paddr,
                    phdr.p_filesz, phdr.p_memsz, phdr.p_align
                )
            new_phdrs.append(phdr)
        else:
            # Split the target PT_LOAD
            # Piece 1: Before section (if any)
            if target_load.p_vaddr < section_vaddr:
                pre_vsize = section_vaddr - target_load.p_vaddr
                pre_fsize = min(pre_vsize, target_load.p_filesz)
                new_phdrs.append(make_load(
                    target_load.p_vaddr, pre_vsize,
                    target_load.p_offset, pre_fsize
                ))

            # Piece 2: Section prefix (if unaligned start)
            if section_vaddr < aligned_vaddr:
                prefix_vsize = aligned_vaddr - section_vaddr
                prefix_fsize = prefix_vsize
                new_phdrs.append(make_load(
                    section_vaddr, prefix_vsize,
                    section_offset, prefix_fsize
                ))

            # Piece 3: Aligned region (zero-page)
            new_phdrs.append(make_load(
                aligned_vaddr, aligned_size,
                aligned_offset, 0  # p_filesz=0 for zero-page
            ))

            # Piece 4: Section suffix (if unaligned end)
            if aligned_end_vaddr < section_end_vaddr:
                suffix_vsize = section_end_vaddr - aligned_end_vaddr
                suffix_fsize = suffix_vsize
                # File offset is now at aligned_offset (shifted down)
                new_phdrs.append(make_load(
                    aligned_end_vaddr, suffix_vsize,
                    aligned_offset, suffix_fsize
                ))

            # Piece 5: After section (if any)
            target_load_end = target_load.p_vaddr + target_load.p_memsz
            if section_end_vaddr < target_load_end:
                post_vaddr = section_end_vaddr
                post_vsize = target_load_end - post_vaddr
                # Calculate file offset and size for post-section content
                section_file_end = section_offset + section_size
                post_file_offset = section_file_end - aligned_size  # Shifted down
                post_fsize = (target_load.p_offset + target_load.p_filesz) - section_file_end
                if post_fsize > 0:
                    new_phdrs.append(make_load(
                        post_vaddr, post_vsize,
                        post_file_offset, post_fsize
                    ))

    if verbose:
        print(f"\nProgram headers: {ehdr.e_phnum} -> {len(new_phdrs)}")

    # Check if new program header table will overflow
    old_phdr_size = ehdr.e_phnum * 56
    new_phdr_size = len(new_phdrs) * 56

    # Find the minimum offset of any content after the program header table
    # (sections, segments, interpreter string, etc.)
    min_content_offset = len(new_data)
    for i in range(ehdr.e_shnum):
        shdr = read_section_header(data, ehdr.e_shoff + i * 64)
        if shdr.sh_offset > ehdr.e_phoff + old_phdr_size:
            min_content_offset = min(min_content_offset, shdr.sh_offset)

    available_space = min_content_offset - ehdr.e_phoff

    # Test mode: artificially reduce available space to force overflow handling
    if force_overflow:
        available_space = new_phdr_size - 1
        if verbose:
            print(f"  [TEST MODE] Forcing overflow: available={available_space}, need={new_phdr_size}")

    if new_phdr_size > available_space:
        # Relocate program header table to end of file to avoid overwriting content
        new_phoff = len(new_data)

        if verbose:
            print(f"  Relocating program headers to end of file (insufficient space at offset {ehdr.e_phoff})")
            print(f"  Old size: {old_phdr_size} bytes, new size: {new_phdr_size} bytes")
            print(f"  Available space: {available_space} bytes, need: {new_phdr_size} bytes")
            print(f"  New program header offset: 0x{new_phoff:x}")

        # Find a suitable virtual address for the relocated program headers
        # Place them after all existing segments
        max_vaddr_end = 0
        for phdr in new_phdrs:
            if phdr.p_type == PT_LOAD:
                vaddr_end = phdr.p_vaddr + phdr.p_memsz
                max_vaddr_end = max(max_vaddr_end, vaddr_end)

        # Align to page boundary for the new segment
        phdr_vaddr = round_up_to_page(max_vaddr_end)

        if verbose:
            print(f"  Placing relocated headers at vaddr 0x{phdr_vaddr:x}")

        # CRITICAL: Ensure file offset and vaddr have same page-aligned remainder
        # The kernel requires: (p_offset % PAGE_SIZE) == (p_vaddr % PAGE_SIZE)
        # for all PT_LOAD segments, otherwise mmap will fail
        vaddr_remainder = phdr_vaddr % PAGE_SIZE  # Should be 0 (page-aligned)
        offset_remainder = new_phoff % PAGE_SIZE

        if offset_remainder != vaddr_remainder:
            # Add padding to align file offset with vaddr's page remainder
            padding_needed = (vaddr_remainder - offset_remainder + PAGE_SIZE) % PAGE_SIZE
            if verbose:
                print(f"  Adding {padding_needed} bytes padding for mmap alignment")
                print(f"    File offset: 0x{new_phoff:x} (remainder 0x{offset_remainder:x})")
                print(f"    Target vaddr: 0x{phdr_vaddr:x} (remainder 0x{vaddr_remainder:x})")

            # Pad the file
            new_data.extend(b'\x00' * padding_needed)
            new_phoff = len(new_data)

            if verbose:
                print(f"    Aligned offset: 0x{new_phoff:x}")

        # Update PT_PHDR to point to the new location
        phdr_updated = False

        for i, phdr in enumerate(new_phdrs):
            if phdr.p_type == PT_PHDR:
                # Update PT_PHDR to point to new location
                new_phdrs[i] = Elf64_Phdr(
                    PT_PHDR, phdr.p_flags,
                    new_phoff,      # file offset
                    phdr_vaddr,     # virtual address
                    phdr_vaddr,     # physical address
                    new_phdr_size,  # file size
                    new_phdr_size,  # memory size
                    phdr.p_align
                )
                phdr_updated = True
                if verbose:
                    print(f"  Updated PT_PHDR: vaddr=0x{phdr_vaddr:x}, offset=0x{new_phoff:x}")
                break

        # Add a PT_LOAD segment for the relocated program headers
        # This ensures they get loaded into memory
        # Note: We need to calculate the final size including this PT_LOAD
        final_phdr_count = len(new_phdrs) + 1  # +1 for the PT_LOAD we're adding
        final_phdr_size = final_phdr_count * 56

        phdr_load = Elf64_Phdr(
            PT_LOAD,
            4,  # PF_R (read-only)
            new_phoff,      # file offset
            phdr_vaddr,     # virtual address
            phdr_vaddr,     # physical address
            final_phdr_size,  # file size - must include this PT_LOAD itself!
            final_phdr_size,  # memory size
            0x1000          # page alignment
        )
        new_phdrs.append(phdr_load)

        if verbose:
            print(f"  Added PT_LOAD for relocated headers (final size: {final_phdr_size} bytes)")

        # Also update PT_PHDR with the correct size
        for i, phdr in enumerate(new_phdrs):
            if phdr.p_type == PT_PHDR:
                new_phdrs[i] = Elf64_Phdr(
                    PT_PHDR, phdr.p_flags,
                    new_phoff,      # file offset
                    phdr_vaddr,     # virtual address
                    phdr_vaddr,     # physical address
                    final_phdr_size,  # correct size
                    final_phdr_size,  # correct size
                    phdr.p_align
                )
                break

        # Append program headers to end of file
        for phdr in new_phdrs:
            phdr_bytes = struct.pack('<IIQQQQQQ',
                phdr.p_type, phdr.p_flags, phdr.p_offset,
                phdr.p_vaddr, phdr.p_paddr, phdr.p_filesz,
                phdr.p_memsz, phdr.p_align)
            new_data.extend(phdr_bytes)

        # Update e_phoff in ELF header
        struct.pack_into('<Q', new_data, 32, new_phoff)

        # Update e_phnum in ELF header
        struct.pack_into('<H', new_data, 56, len(new_phdrs))
    else:
        # Write program headers in place
        if verbose:
            print(f"  Writing program headers in place (sufficient space)")

        for i, phdr in enumerate(new_phdrs):
            write_program_header(new_data, ehdr.e_phoff + i * 56, phdr)

        # Update e_phnum in ELF header
        struct.pack_into('<H', new_data, 56, len(new_phdrs))

    # Update section header table offset
    new_shoff = ehdr.e_shoff - aligned_size if ehdr.e_shoff > aligned_offset else ehdr.e_shoff
    struct.pack_into('<Q', new_data, 40, new_shoff)

    # Update section headers
    for i in range(ehdr.e_shnum):
        shdr = read_section_header(data, ehdr.e_shoff + i * 64)

        if i == target_section_idx:
            # Mark as NOBITS
            shdr = Elf64_Shdr(
                shdr.sh_name, SHT_NOBITS, shdr.sh_flags,
                shdr.sh_addr, section_offset, shdr.sh_size,
                shdr.sh_link, shdr.sh_info, shdr.sh_addralign, shdr.sh_entsize
            )
        elif shdr.sh_offset > aligned_offset:
            # Adjust offset
            shdr = Elf64_Shdr(
                shdr.sh_name, shdr.sh_type, shdr.sh_flags,
                shdr.sh_addr, shdr.sh_offset - aligned_size, shdr.sh_size,
                shdr.sh_link, shdr.sh_info, shdr.sh_addralign, shdr.sh_entsize
            )

        write_section_header(new_data, new_shoff + i * 64, shdr)

    # Write output and preserve original permissions
    output_path.write_bytes(new_data)
    original_mode = input_path.stat().st_mode
    output_path.chmod(original_mode)

    if verbose:
        print(f"\n✅ Successfully wrote {output_path}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Conservative zero-page optimizer for ELF binaries"
    )
    parser.add_argument("input", type=Path, help="Input ELF binary")
    parser.add_argument("output", type=Path, help="Output ELF binary")
    parser.add_argument(
        "--section",
        default=".hip_fatbin",
        help="Section name to zero-page (default: .hip_fatbin)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    success = conservative_zero_page(
        args.input,
        args.output,
        section_name=args.section,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)
