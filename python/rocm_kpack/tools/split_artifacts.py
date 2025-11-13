#!/usr/bin/env python3

"""
Command-line tool for splitting TheRock artifacts into generic and architecture-specific components.

This tool processes TheRock build artifacts to separate host code from device code,
creating split artifacts suitable for kpack-based distribution.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rocm_kpack.artifact_splitter import ArtifactSplitter
from rocm_kpack.binutils import Toolchain
from rocm_kpack.database_handlers import get_database_handlers, list_available_handlers


def main():
    """Main entry point for the artifact splitter CLI."""
    # Get list of available handlers for help text
    available_handlers = list_available_handlers()

    parser = argparse.ArgumentParser(
        description="Split TheRock artifacts into generic and architecture-specific components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split only fat binaries (no database handling)
  %(prog)s --input-dir artifacts/hip_lib_gfx110X --output-dir output/ --component-name hip_lib

  # Split fat binaries and rocBLAS databases
  %(prog)s --input-dir artifacts/rocblas_lib_gfx110X --output-dir output/ \\
           --component-name rocblas_lib --split-databases rocblas

  # Split with multiple database handlers
  %(prog)s --input-dir artifacts/mixed_lib_gfx110X --output-dir output/ \\
           --component-name mixed_lib --split-databases rocblas hipblaslt aotriton
"""
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input artifact directory containing artifact_manifest.txt"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for split artifacts"
    )

    parser.add_argument(
        "--component-name",
        type=str,
        required=True,
        help="Component name (e.g., 'hip_lib', 'rocblas_lib')"
    )

    parser.add_argument(
        "--split-databases",
        nargs="+",
        choices=available_handlers,
        default=None,
        help=f"Enable database splitting for specified types. Available: {', '.join(available_handlers)}"
    )

    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help=f"Temporary directory for intermediate files (default: {tempfile.gettempdir()})"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--clang-offload-bundler",
        type=Path,
        default=None,
        help="Path to clang-offload-bundler tool (will search PATH if not specified)"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}", file=sys.stderr)
        return 1

    # Check for artifact_manifest.txt
    manifest_path = args.input_dir / "artifact_manifest.txt"
    if not manifest_path.exists():
        print(f"Error: No artifact_manifest.txt found in {args.input_dir}", file=sys.stderr)
        print("       This doesn't appear to be a TheRock artifact directory", file=sys.stderr)
        return 1

    # Set temporary directory environment variable for subprocess tools
    # This is necessary because clang-offload-bundler and other binutils
    # tools respect TMPDIR for their temporary files
    os.environ["TMPDIR"] = str(args.tmp_dir)
    if args.verbose:
        print(f"Using temporary directory: {args.tmp_dir}")

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get database handlers if requested
    database_handlers = []
    if args.split_databases:
        try:
            database_handlers = get_database_handlers(args.split_databases)
            if args.verbose:
                handler_names = ", ".join(h.name() for h in database_handlers)
                print(f"Database handlers enabled: {handler_names}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif args.verbose:
        print("Database splitting disabled (use --split-databases to enable)")

    # Create and run the splitter
    try:
        # Create toolchain with provided clang-offload-bundler path
        toolchain = Toolchain(clang_offload_bundler=args.clang_offload_bundler)

        splitter = ArtifactSplitter(
            component_name=args.component_name,
            toolchain=toolchain,
            database_handlers=database_handlers,
            verbose=args.verbose
        )

        print(f"Splitting artifact: {args.input_dir}")
        print(f"Component name: {args.component_name}")
        print(f"Output directory: {args.output_dir}")

        splitter.split(args.input_dir, args.output_dir)

        print("Splitting complete!")
        return 0

    except Exception as e:
        print(f"Error during splitting: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())