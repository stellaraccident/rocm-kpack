import pytest
import pathlib

from rocm_kpack.binutils import Toolchain


@pytest.fixture(scope="session")
def test_assets_dir() -> pathlib.Path:
    """Provides a pathlib.Path to the shared test_assets directory."""
    test_assets_path = pathlib.Path(__file__).parent.parent / "test_assets"
    if not test_assets_path.is_dir():
        raise FileNotFoundError(
            f"test_assets directory not found at: {test_assets_path}"
        )
    return test_assets_path.resolve()


@pytest.fixture(scope="session")
def toolchain() -> Toolchain:
    """Provides a Toolchain, using ROCm installation if available."""
    # Try to find clang-offload-bundler in common ROCm locations
    potential_paths = [
        pathlib.Path("/home/stella/workspace/rocm/gfx1100/lib/llvm/bin/clang-offload-bundler"),
        pathlib.Path("/opt/rocm/llvm/bin/clang-offload-bundler"),
    ]

    for path in potential_paths:
        if path.exists():
            return Toolchain(clang_offload_bundler=path)

    # Fall back to system PATH
    return Toolchain()
