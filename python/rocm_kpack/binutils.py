import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


class Toolchain:
    """Manages configuration of various toolchain locations."""

    def __init__(self, *, clang_offload_bundler: Path | None = None):
        self.clang_offload_bundler = self._validate_or_find(
            "clang-offload-bundler", clang_offload_bundler
        )

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
    """Represents a bundled binary at some path."""

    def __init__(self, file_path: Path, *, toolchain: Toolchain | None = None):
        self.toolchain = toolchain or Toolchain()
        self.file_path = file_path

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

    def _list_bundled_targets(self, file_path: Path) -> list[tuple[str, str]]:
        """Returns a list of (target_name, file_name) for all bundles."""
        lines = (
            self.toolchain.exec_capture_text(
                [
                    self.toolchain.clang_offload_bundler,
                    "--list",
                    "--type=o",
                    f"--input={file_path}",
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
        if not targets:
            return
        args = [
            self.toolchain.clang_offload_bundler,
            "--unbundle",
            "--type=o",
            f"--input={self.file_path}",
            f"--targets={','.join(targets)}",
        ]
        for output in outputs:
            args.extend(["--output", output])
        self.toolchain.exec(args)
