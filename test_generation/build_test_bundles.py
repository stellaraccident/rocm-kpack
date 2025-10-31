#!/usr/bin/env python3
"""Build script for generating test bundled binaries.

Generates various permutations of bundled binaries for testing unbundling
functionality across different platforms and compiler configurations.

Usage:
    python build_test_bundles.py [--rocm-path PATH]
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


class BundleBuilder:
    """Builder for HIP bundled binary test assets."""

    def __init__(self, rocm_path: Path | None = None):
        """Initialize the builder.

        Args:
            rocm_path: Path to ROCm installation (auto-detected if None)
        """
        self.script_dir = Path(__file__).parent
        self.kernel_src = self.script_dir / "simple_kernel.hip"

        # Detect platform first (needed by other methods)
        system = platform.system().lower()
        self.platform = "windows" if system == "windows" else "linux"

        # Find ROCm and tools
        self.rocm_path = self._find_rocm(rocm_path)
        self.hipcc = self._find_hipcc()

        # Detect code object version
        self.code_object_version = self._detect_code_object_version()

        # Setup output directory
        self.output_dir = (
            self.script_dir.parent
            / "test_assets"
            / "bundled_binaries"
            / self.platform
            / f"cov{self.code_object_version}"
        )

    def _find_rocm(self, explicit_path: Path | None) -> Path:
        """Find ROCm installation path.

        Args:
            explicit_path: Explicit ROCm path if provided

        Returns:
            Path to ROCm installation

        Raises:
            RuntimeError: If ROCm cannot be found
        """
        if explicit_path:
            if explicit_path.exists():
                return explicit_path
            raise RuntimeError(f"Specified ROCm path does not exist: {explicit_path}")

        # Check environment variables
        for env_var in ["ROCM_PATH", "HIP_PATH"]:
            env_path = os.environ.get(env_var)
            if env_path:
                path = Path(env_path)
                if path.exists():
                    return path

        raise RuntimeError(
            "Could not find ROCm installation. "
            "Please specify with --rocm-path or set ROCM_PATH/HIP_PATH environment variable."
        )

    def _find_hipcc(self) -> Path:
        """Find hipcc compiler.

        Returns:
            Path to hipcc executable

        Raises:
            RuntimeError: If hipcc cannot be found
        """
        # Try in ROCm bin directory
        hipcc_name = "hipcc.bat" if self.platform == "windows" else "hipcc"
        hipcc = self.rocm_path / "bin" / hipcc_name

        if hipcc.exists():
            return hipcc

        raise RuntimeError(f"Could not find hipcc at {hipcc}")

    def _detect_code_object_version(self) -> str:
        """Detect code object version from compiler.

        Returns:
            Code object version string (e.g., "5", "6")
        """
        try:
            result = subprocess.run(
                [str(self.hipcc), "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            # Try to extract code object version from output
            for line in result.stdout.splitlines():
                if "code object version" in line.lower():
                    # Extract version number
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            return part
            # Default to 5 if not found
            return "5"
        except subprocess.CalledProcessError:
            return "5"

    def _run_hipcc(self, args: list[str], output_file: Path) -> bool:
        """Run hipcc with given arguments.

        Args:
            args: Arguments to pass to hipcc
            output_file: Output file path

        Returns:
            True if successful, False otherwise
        """
        cmd = [str(self.hipcc)] + args
        print(f"  Running: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if output_file.exists():
                size_kb = output_file.stat().st_size / 1024
                print(f"  ✓ Created: {output_file.name} ({size_kb:.1f} KB)")
                return True
            else:
                print(f"  ✗ Failed: Output file not created")
                return False
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            return False

    def build_exe_compressed(self):
        """Build executable with compressed bundles (if supported)."""
        exe_name = "test_kernel_compressed.exe"
        print(f"\nBuilding: {exe_name} (gfx1100,gfx1101 with compression)")
        output = self.output_dir / exe_name

        # Try with compression flag
        success = self._run_hipcc(
            [
                str(self.kernel_src),
                "--offload-arch=gfx1100",
                "--offload-arch=gfx1101",
                "-mllvm",
                "--offload-compress",
                "-o",
                str(output),
                "-fno-gpu-rdc",
                "-DBUILD_EXECUTABLE",
            ],
            output,
        )

        if not success:
            print("  ⚠ Compression may not be supported, trying without flag")
            success = self._run_hipcc(
                [
                    str(self.kernel_src),
                    "--offload-arch=gfx1100",
                    "--offload-arch=gfx1101",
                    "-o",
                    str(output),
                    "-fno-gpu-rdc",
                    "-DBUILD_EXECUTABLE",
                ],
                output,
            )

        return success

    def build_exe_wide_arch(self):
        """Build executable with wide architecture coverage."""
        exe_name = "test_kernel_wide.exe"
        print(f"\nBuilding: {exe_name} (gfx900,gfx906,gfx908,gfx90a,gfx1100)")
        output = self.output_dir / exe_name

        return self._run_hipcc(
            [
                str(self.kernel_src),
                "--offload-arch=gfx900",
                "--offload-arch=gfx906",
                "--offload-arch=gfx908",
                "--offload-arch=gfx90a",
                "--offload-arch=gfx1100",
                "-o",
                str(output),
                "-fno-gpu-rdc",
                "-DBUILD_EXECUTABLE",
            ],
            output,
        )

    def build_executable_single_arch(self):
        """Build executable with single architecture."""
        exe_name = "test_kernel_single.exe"
        print(f"\nBuilding: {exe_name} (gfx1100 executable)")
        output = self.output_dir / exe_name

        return self._run_hipcc(
            [
                str(self.kernel_src),
                "--offload-arch=gfx1100",
                "-o",
                str(output),
                "-fno-gpu-rdc",
                "-DBUILD_EXECUTABLE",
            ],
            output,
        )

    def build_executable_multi_arch(self):
        """Build executable with multiple architectures."""
        exe_name = "test_kernel_multi.exe"
        print(f"\nBuilding: {exe_name} (gfx1100,gfx1101 executable)")
        output = self.output_dir / exe_name

        return self._run_hipcc(
            [
                str(self.kernel_src),
                "--offload-arch=gfx1100",
                "--offload-arch=gfx1101",
                "-o",
                str(output),
                "-fno-gpu-rdc",
                "-DBUILD_EXECUTABLE",
            ],
            output,
        )

    def build_shared_lib_single_arch(self):
        """Build shared library with single architecture."""
        if self.platform == "windows":
            lib_name = "test_kernel_single.dll"
            shared_flag = "-shared"
        else:
            lib_name = "libtest_kernel_single.so"
            shared_flag = "-shared"

        print(f"\nBuilding: {lib_name} (gfx1100 shared library)")
        output = self.output_dir / lib_name

        return self._run_hipcc(
            [
                str(self.kernel_src),
                "--offload-arch=gfx1100",
                "-o",
                str(output),
                "-fno-gpu-rdc",
                shared_flag,
                "-fPIC",
            ],
            output,
        )

    def build_shared_lib_multi_arch(self):
        """Build shared library with multiple architectures."""
        if self.platform == "windows":
            lib_name = "test_kernel_multi.dll"
            shared_flag = "-shared"
        else:
            lib_name = "libtest_kernel_multi.so"
            shared_flag = "-shared"

        print(f"\nBuilding: {lib_name} (gfx1100,gfx1101 shared library)")
        output = self.output_dir / lib_name

        return self._run_hipcc(
            [
                str(self.kernel_src),
                "--offload-arch=gfx1100",
                "--offload-arch=gfx1101",
                "-o",
                str(output),
                "-fno-gpu-rdc",
                shared_flag,
                "-fPIC",
            ],
            output,
        )

    def generate_manifest(self):
        """Generate manifest file documenting the test assets."""
        import datetime

        manifest_path = self.output_dir / "MANIFEST.txt"

        # Get compiler version
        try:
            result = subprocess.run(
                [str(self.hipcc), "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            compiler_version = result.stdout.splitlines()[0]
        except:
            compiler_version = "Unknown"

        exe_ext = ".exe" if self.platform == "windows" else ""
        lib_prefix = "" if self.platform == "windows" else "lib"
        lib_ext = ".dll" if self.platform == "windows" else ".so"

        content = f"""Test Bundled Binaries Generated: {datetime.datetime.now()}
ROCm Path: {self.rocm_path}
Code Object Version: {self.code_object_version}
Platform: {self.platform}
Compiler: {compiler_version}

Executables:
- test_kernel_single{exe_ext}: Single architecture (gfx1100)
- test_kernel_multi{exe_ext}: Multiple architectures (gfx1100, gfx1101)
- test_kernel_compressed{exe_ext}: Multiple architectures with compression (if supported)
- test_kernel_wide{exe_ext}: Wide architecture coverage (gfx900,906,908,90a,1100)

Shared Libraries:
- {lib_prefix}test_kernel_single{lib_ext}: Single architecture (gfx1100)
- {lib_prefix}test_kernel_multi{lib_ext}: Multiple architectures (gfx1100, gfx1101)

Each file contains the same simple HIP kernels (vectorAdd, scalarMultiply).

Source: test_generation/simple_kernel.hip
Build Script: test_generation/build_test_bundles.py

These assets are used to test unbundling functionality across:
- Executables and shared libraries
- Uncompressed vs compressed code objects
- Single vs multiple architectures
- Different code object versions
- ELF (Linux) vs PE/COFF (Windows) formats
"""

        manifest_path.write_text(content)
        print(f"\n✓ Manifest created: {manifest_path}")

    def build_all(self):
        """Build all test bundles."""
        print("=" * 70)
        print("Building Test Bundled Binaries")
        print("=" * 70)
        print(f"ROCm Path: {self.rocm_path}")
        print(f"Compiler: {self.hipcc}")
        print(f"Code Object Version: {self.code_object_version}")
        print(f"Platform: {self.platform}")
        print(f"Output Directory: {self.output_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build all variants
        results = {
            # Executables
            "test_kernel_single (exe)": self.build_executable_single_arch(),
            "test_kernel_multi (exe)": self.build_executable_multi_arch(),
            "test_kernel_compressed (exe)": self.build_exe_compressed(),
            "test_kernel_wide (exe)": self.build_exe_wide_arch(),
            # Shared libraries
            "test_kernel_single (so/dll)": self.build_shared_lib_single_arch(),
            "test_kernel_multi (so/dll)": self.build_shared_lib_multi_arch(),
        }

        # Generate manifest
        self.generate_manifest()

        # Summary
        print("\n" + "=" * 70)
        print("Build Summary")
        print("=" * 70)
        for name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{status}: {name}")

        # List all generated files
        print(f"\nGenerated files in {self.output_dir}:")
        # List all binary files (executables, libraries)
        patterns = ["*.exe", "*.so", "*.dll", "test_kernel_*", "libtest_kernel_*"]
        seen_files = set()
        for pattern in patterns:
            for file in sorted(self.output_dir.glob(pattern)):
                if file.name != "MANIFEST.txt" and file not in seen_files:
                    size_kb = file.stat().st_size / 1024
                    print(f"  {file.name:40} {size_kb:8.1f} KB")
                    seen_files.add(file)

        total_success = sum(results.values())
        total_builds = len(results)
        print(f"\nTotal: {total_success}/{total_builds} successful")

        return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Build test bundled binaries for rocm-kpack testing"
    )
    parser.add_argument(
        "--rocm-path", type=Path, help="Path to ROCm installation (auto-detected if not specified)"
    )

    args = parser.parse_args()

    try:
        builder = BundleBuilder(rocm_path=args.rocm_path)
        success = builder.build_all()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
