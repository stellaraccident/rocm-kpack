"""
ELF Fat Device Neutralizer - Removes .hip_fatbin section and reclaims disk space.

This module neutralizes fat binaries by:
1. Removing .hip_fatbin section content (not just headers)
2. Shifting subsequent sections to fill the gap
3. Updating all ELF structures (program headers, section headers)
4. Actually reclaiming disk space

Unlike objcopy --remove-section, this neutralizer modifies PT_LOAD segments
to eliminate the device code content, creating true host-only binaries.

Note: Currently supports 64-bit little-endian ELF (Linux).
TODO: Windows PE/COFF support will require similar approach with different binary format.
"""

import os
import struct
from pathlib import Path
from typing import NamedTuple


class ElfHeader(NamedTuple):
    """ELF file header (Elf64_Ehdr)"""
    e_ident: bytes  # 16 bytes - magic, class, endianness, etc
    e_type: int
    e_machine: int
    e_version: int
    e_entry: int
    e_phoff: int  # Program header offset
    e_shoff: int  # Section header offset
    e_flags: int
    e_ehsize: int
    e_phentsize: int
    e_phnum: int  # Number of program headers
    e_shentsize: int
    e_shnum: int  # Number of section headers
    e_shstrndx: int  # Section header string table index


class ProgramHeader(NamedTuple):
    """Program header (Elf64_Phdr)"""
    p_type: int
    p_flags: int
    p_offset: int
    p_vaddr: int
    p_paddr: int
    p_filesz: int
    p_memsz: int
    p_align: int


class SectionHeader(NamedTuple):
    """Section header (Elf64_Shdr)"""
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


# Constants
PT_LOAD = 1
PT_DYNAMIC = 2
PT_GNU_EH_FRAME = 0x6474e550

SHT_NULL = 0
SHT_PROGBITS = 1
SHT_NOBITS = 8

SHF_ALLOC = 0x2


class ElfFatDeviceNeutralizer:
    """Neutralizes ELF fat binaries by removing .hip_fatbin section and reclaiming space."""

    def __init__(self, input_path: Path):
        self.input_path = Path(input_path)
        self.data = self.input_path.read_bytes()

        # Save original file permissions for restoration
        self.original_mode = self.input_path.stat().st_mode

        # Parse ELF structures
        self.elf_header = self._parse_elf_header()
        self.program_headers = self._parse_program_headers()
        self.section_headers = self._parse_section_headers()
        self.section_names = self._parse_section_names()

        # Find .hip_fatbin
        self.hip_fatbin_idx = None
        self.hip_fatbin_section = None
        for idx, shdr in enumerate(self.section_headers):
            if self.section_names.get(idx) == ".hip_fatbin":
                self.hip_fatbin_idx = idx
                self.hip_fatbin_section = shdr
                break

    def _parse_elf_header(self) -> ElfHeader:
        """Parse ELF header (assumes 64-bit little-endian)"""
        # Verify ELF magic
        if self.data[:4] != b'\x7fELF':
            raise ValueError("Not an ELF file")

        # Verify 64-bit
        if self.data[4] != 2:
            raise ValueError("Only 64-bit ELF supported")

        # Verify little-endian
        if self.data[5] != 1:
            raise ValueError("Only little-endian ELF supported")

        # Parse header (64 bytes total)
        fmt = '<16sHHIQQQIHHHHHH'  # Little-endian, see Elf64_Ehdr
        fields = struct.unpack_from(fmt, self.data, 0)

        return ElfHeader(*fields)

    def _parse_program_headers(self) -> list[ProgramHeader]:
        """Parse all program headers"""
        headers = []
        offset = self.elf_header.e_phoff
        count = self.elf_header.e_phnum
        size = self.elf_header.e_phentsize

        # Elf64_Phdr format
        fmt = '<IIQQQQQQ'

        for i in range(count):
            fields = struct.unpack_from(fmt, self.data, offset + i * size)
            headers.append(ProgramHeader(*fields))

        return headers

    def _parse_section_headers(self) -> list[SectionHeader]:
        """Parse all section headers"""
        headers = []
        offset = self.elf_header.e_shoff
        count = self.elf_header.e_shnum
        size = self.elf_header.e_shentsize

        # Elf64_Shdr format
        fmt = '<IIQQQQIIQQ'

        for i in range(count):
            fields = struct.unpack_from(fmt, self.data, offset + i * size)
            headers.append(SectionHeader(*fields))

        return headers

    def _parse_section_names(self) -> dict[int, str]:
        """Parse section names from string table"""
        if self.elf_header.e_shstrndx == 0:
            return {}

        shstrtab = self.section_headers[self.elf_header.e_shstrndx]
        strtab_data = self.data[shstrtab.sh_offset:shstrtab.sh_offset + shstrtab.sh_size]

        names = {}
        for idx, shdr in enumerate(self.section_headers):
            # Extract null-terminated string
            name_offset = shdr.sh_name
            if name_offset >= len(strtab_data):
                names[idx] = ""
                continue

            end = strtab_data.find(b'\x00', name_offset)
            if end == -1:
                end = len(strtab_data)

            names[idx] = strtab_data[name_offset:end].decode('utf-8', errors='replace')

        return names

    def has_hip_fatbin(self) -> bool:
        """Check if binary has .hip_fatbin section"""
        return self.hip_fatbin_section is not None

    def calculate_removal_plan(self) -> dict:
        """Calculate how to rebuild the ELF without .hip_fatbin"""
        if not self.has_hip_fatbin():
            raise ValueError("No .hip_fatbin section found")

        shdr = self.hip_fatbin_section
        removal_size = shdr.sh_size
        removal_offset = shdr.sh_offset
        removal_vaddr = shdr.sh_addr

        # Find sections that need to be shifted
        sections_to_shift = []
        for idx, s in enumerate(self.section_headers):
            # Skip the .hip_fatbin itself
            if idx == self.hip_fatbin_idx:
                continue

            # Shift sections that come after .hip_fatbin in the file
            # and are in the same PT_LOAD segment
            if s.sh_offset > removal_offset and s.sh_type != SHT_NULL:
                sections_to_shift.append(idx)

        # Find program headers that need updating
        phdrs_to_update = []
        for idx, phdr in enumerate(self.program_headers):
            # Check if this segment contains .hip_fatbin
            seg_start = phdr.p_offset
            seg_end = phdr.p_offset + phdr.p_filesz

            if removal_offset >= seg_start and removal_offset < seg_end:
                phdrs_to_update.append((idx, 'contains'))
            elif phdr.p_offset > removal_offset:
                phdrs_to_update.append((idx, 'follows'))

        return {
            'removal_size': removal_size,
            'removal_offset': removal_offset,
            'removal_vaddr': removal_vaddr,
            'sections_to_shift': sections_to_shift,
            'phdrs_to_update': phdrs_to_update,
        }

    def rebuild(self, output_path: Path, *, verbose: bool = False) -> dict:
        """Rebuild ELF without .hip_fatbin section"""
        if not self.has_hip_fatbin():
            # No work needed, just copy
            output_path.write_bytes(self.data)
            return {'removed': 0}

        plan = self.calculate_removal_plan()
        removal_size = plan['removal_size']
        removal_offset = plan['removal_offset']

        if verbose:
            print(f"  Removing .hip_fatbin: offset=0x{removal_offset:x}, size=0x{removal_size:x} ({removal_size:,} bytes)")

        # Build new file content
        new_data = bytearray()

        # Copy everything before .hip_fatbin
        new_data.extend(self.data[:removal_offset])

        # Skip .hip_fatbin content
        skip_end = removal_offset + removal_size

        # Copy everything after .hip_fatbin
        new_data.extend(self.data[skip_end:])

        # Now update headers in the new data
        self._update_elf_header(new_data, plan)
        self._update_program_headers(new_data, plan, verbose=verbose)
        self._update_section_headers(new_data, plan, verbose=verbose)

        # Write to output
        output_path.write_bytes(new_data)

        # Restore original file permissions
        os.chmod(output_path, self.original_mode)

        return {
            'removed': removal_size,
            'original_size': len(self.data),
            'new_size': len(new_data),
        }

    def _update_elf_header(self, data: bytearray, plan: dict):
        """Update ELF header in rebuilt file"""
        removal_size = plan['removal_size']
        ehdr = self.elf_header

        # If section header table comes after .hip_fatbin, shift it
        new_shoff = ehdr.e_shoff
        if ehdr.e_shoff > plan['removal_offset']:
            new_shoff -= removal_size

        # Update e_shoff
        struct.pack_into('<Q', data, 40, new_shoff)

    def _update_program_headers(self, data: bytearray, plan: dict, *, verbose: bool = False):
        """Update program headers in rebuilt file"""
        removal_size = plan['removal_size']
        phdr_offset = self.elf_header.e_phoff

        for idx, action in plan['phdrs_to_update']:
            phdr = self.program_headers[idx]
            offset = phdr_offset + idx * self.elf_header.e_phentsize

            if action == 'contains':
                # This segment contains .hip_fatbin - reduce its size
                new_filesz = phdr.p_filesz - removal_size
                new_memsz = phdr.p_memsz - removal_size

                struct.pack_into('<Q', data, offset + 32, new_filesz)
                struct.pack_into('<Q', data, offset + 40, new_memsz)

                if verbose:
                    print(f"  Updated PT_LOAD segment: filesz 0x{phdr.p_filesz:x} -> 0x{new_filesz:x}")

            elif action == 'follows':
                # This segment comes after .hip_fatbin - shift it
                new_offset = phdr.p_offset - removal_size
                new_vaddr = phdr.p_vaddr - removal_size
                new_paddr = phdr.p_paddr - removal_size

                struct.pack_into('<Q', data, offset + 8, new_offset)
                struct.pack_into('<Q', data, offset + 16, new_vaddr)
                struct.pack_into('<Q', data, offset + 24, new_paddr)

    def _update_section_headers(self, data: bytearray, plan: dict, *, verbose: bool = False):
        """Update section headers in rebuilt file"""
        removal_size = plan['removal_size']

        # Section header table might have shifted
        shdr_offset = self.elf_header.e_shoff
        if shdr_offset > plan['removal_offset']:
            shdr_offset -= removal_size

        for idx, shdr in enumerate(self.section_headers):
            offset = shdr_offset + idx * self.elf_header.e_shentsize

            if idx == self.hip_fatbin_idx:
                # Mark .hip_fatbin as NULL
                struct.pack_into('<I', data, offset + 4, SHT_NULL)
                struct.pack_into('<Q', data, offset + 32, 0)  # sh_size = 0
                if verbose:
                    print(f"  Marked .hip_fatbin section as SHT_NULL")

            elif idx in plan['sections_to_shift']:
                # Shift this section
                new_offset = shdr.sh_offset - removal_size

                # Only shift virtual address if section was allocated after .hip_fatbin
                if shdr.sh_addr > 0 and shdr.sh_addr >= plan['removal_vaddr']:
                    new_addr = shdr.sh_addr - removal_size
                    struct.pack_into('<Q', data, offset + 16, new_addr)  # sh_addr at offset+16

                # Always shift file offset
                struct.pack_into('<Q', data, offset + 24, new_offset)  # sh_offset at offset+24

                if verbose:
                    section_name = self.section_names.get(idx, f"section_{idx}")
                    print(f"  Shifted {section_name}: offset 0x{shdr.sh_offset:x} -> 0x{new_offset:x}")


def neutralize_binary(input_path: Path, output_path: Path, *, verbose: bool = False) -> dict:
    """
    Neutralize an ELF fat binary by removing .hip_fatbin section.

    Args:
        input_path: Path to original fat binary
        output_path: Path for neutralized (host-only) binary
        verbose: If True, print detailed progress information

    Returns:
        Dictionary with statistics about the neutralization
    """
    neutralizer = ElfFatDeviceNeutralizer(input_path)

    if not neutralizer.has_hip_fatbin():
        if verbose:
            print(f"  No .hip_fatbin section found, copying as-is")
        output_path.write_bytes(neutralizer.data)
        os.chmod(output_path, neutralizer.original_mode)
        return {'removed': 0, 'original_size': len(neutralizer.data), 'new_size': len(neutralizer.data)}

    return neutralizer.rebuild(output_path, verbose=verbose)
