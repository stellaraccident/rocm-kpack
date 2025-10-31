"""Command-line tool for packing bundled binaries into .kpack archives.

Transforms an install tree containing bundled binaries into:
- Host-only binaries with .rocm_kpack_ref markers
- .kpack archive files with extracted GPU kernels
- Verbatim copies of all other files
"""

import argparse
import sys
import time
from pathlib import Path

from rocm_kpack.artifact_scanner import ArtifactScanner, RecognizerRegistry
from rocm_kpack.binutils import Toolchain
from rocm_kpack.kpack import PackedKernelArchive
from rocm_kpack.packing_visitor import PackingVisitor


def main():
    parser = argparse.ArgumentParser(
        description="Pack bundled binaries into .kpack archives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pack ROCm installation for gfx1100 family
  python -m rocm_kpack.tools.pack_tree \\
    --input /opt/rocm \\
    --output /tmp/rocm-packed \\
    --group-name rocm \\
    --gfx-arch-family gfx1100 \\
    --gfx-arches gfx1100,gfx1101,gfx1102

  # Pack with specific toolchain
  python -m rocm_kpack.tools.pack_tree \\
    --input ./install \\
    --output ./packed \\
    --group-name myapp \\
    --gfx-arch-family gfx100X \\
    --gfx-arches gfx1030,gfx1001 \\
    --clang-offload-bundler /opt/rocm/bin/clang-offload-bundler
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory tree containing bundled binaries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for packed tree",
    )
    parser.add_argument(
        "--group-name",
        type=str,
        required=True,
        help="Group name for this build slice (e.g., 'blas', 'torch', 'rocm')",
    )
    parser.add_argument(
        "--gfx-arch-family",
        type=str,
        required=True,
        help="Architecture family identifier (e.g., 'gfx1100', 'gfx100X')",
    )
    parser.add_argument(
        "--gfx-arches",
        type=str,
        required=True,
        help="Comma-separated list of actual architectures in family",
    )
    Toolchain.configure_argparse(parser)

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}", file=sys.stderr)
        return 1
    if not args.input.is_dir():
        print(f"Error: Input path is not a directory: {args.input}", file=sys.stderr)
        return 1

    # Parse gfx_arches
    gfx_arches = [arch.strip() for arch in args.gfx_arches.split(",")]
    if not gfx_arches:
        print("Error: --gfx-arches must contain at least one architecture", file=sys.stderr)
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Packing install tree:")
    print(f"  Input:            {args.input}")
    print(f"  Output:           {args.output}")
    print(f"  Group:            {args.group_name}")
    print(f"  Arch family:      {args.gfx_arch_family}")
    print(f"  Architectures:    {', '.join(gfx_arches)}")
    print()

    # Initialize toolchain
    try:
        toolchain = Toolchain.from_args(args)
        print(f"Toolchain:")
        print(f"  clang-offload-bundler: {toolchain.clang_offload_bundler}")
        print(f"  objcopy:               {toolchain.objcopy}")
        print(f"  readelf:               {toolchain.readelf}")
        print()
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Create visitor
    visitor = PackingVisitor(
        output_root=args.output,
        group_name=args.group_name,
        gfx_arch_family=args.gfx_arch_family,
        gfx_arches=gfx_arches,
        toolchain=toolchain,
    )

    # Scan and process tree
    print("Scanning tree...")
    start_time = time.time()

    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)

    try:
        scanner.scan_tree(args.input, visitor)
    except Exception as e:
        print(f"\nError during scanning: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    scan_time = time.time() - start_time

    # Finalize
    print("Finalizing .kpack archive...")
    try:
        visitor.finalize()
    except Exception as e:
        print(f"\nError during finalization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    total_time = time.time() - start_time

    # Report statistics
    stats = visitor.get_stats()
    kpack_filename = PackedKernelArchive.compute_pack_filename(
        args.group_name, args.gfx_arch_family
    )
    kpack_path = args.output / ".kpack" / kpack_filename

    print()
    print("=" * 70)
    print("PACKING COMPLETE")
    print("=" * 70)
    print()
    print(f"Statistics:")
    print(f"  Opaque files:       {stats['opaque_files']:>6}")
    print(f"  Bundled binaries:   {stats['bundled_binaries']:>6}")
    print(f"  Kernel databases:   {stats['kernel_databases']:>6}")
    print()
    print(f"Output:")
    print(f"  Packed tree:        {args.output}")
    print(f"  Kpack archive:      {kpack_path}")
    if kpack_path.exists():
        kpack_size_mb = kpack_path.stat().st_size / (1024 * 1024)
        print(f"  Kpack size:         {kpack_size_mb:.2f} MB")
    print()
    print(f"Timing:")
    print(f"  Scan time:          {scan_time:.2f}s")
    print(f"  Total time:         {total_time:.2f}s")
    print()

    # Read back kpack to show TOC summary
    try:
        archive = PackedKernelArchive.read(kpack_path)
        num_binaries = len(archive.toc)
        num_kernels = sum(len(arches) for arches in archive.toc.values())
        print(f"Kpack contents:")
        print(f"  Binaries:           {num_binaries:>6}")
        print(f"  Total kernels:      {num_kernels:>6}")

        if num_binaries > 0 and num_binaries <= 10:
            print()
            print(f"Binaries in archive:")
            for binary_path in sorted(archive.toc.keys()):
                arch_count = len(archive.toc[binary_path])
                print(f"  {binary_path:<50} ({arch_count} arch{'s' if arch_count > 1 else ''})")
    except Exception as e:
        print(f"Warning: Could not read back kpack file: {e}", file=sys.stderr)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
