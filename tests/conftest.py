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
    return Toolchain()
