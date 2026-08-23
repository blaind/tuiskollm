#!/usr/bin/env python3

"""Build and verify the downloadable TuiskoLLM server archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import re
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import NoReturn, cast

WORKSPACE = Path(__file__).resolve().parents[1]
BINARY_NAME = "tuiskollm"
ARCHIVE_TARGET = "linux-x86_64-glibc2.35"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
REQUIRED_LIBRARIES = {"libc.so.6", "libcuda.so.1"}
ALLOWED_LIBRARIES = REQUIRED_LIBRARIES | {
    "ld-linux-x86-64.so.2",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
}
BUILD_PATHS = (
    str(WORKSPACE).encode(),
    b"/github/workspace",
    b"/home/runner/work/",
    b"/__w/tuiskollm/",
    b"/root/.cargo",
    b"/root/.rustup",
)


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def workspace_version() -> str:
    with (WORKSPACE / "Cargo.toml").open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    workspace = manifest.get("workspace")
    package = workspace.get("package") if isinstance(workspace, dict) else None
    version = package.get("version") if isinstance(package, dict) else None
    if not isinstance(version, str):
        fail("Cargo.toml does not define a string workspace package version")
    return version


def command_text(arguments: list[str], *, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        fail(f"{' '.join(arguments)} failed with {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def checked_identity(name: str, value: str, pattern: str) -> str:
    if re.fullmatch(pattern, value) is None:
        fail(f"invalid {name} {value!r}")
    return value


def parse_glibc(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)", value)
    if match is None:
        fail(f"invalid glibc version {value!r}")
    return int(match.group(1)), int(match.group(2))


def required_glibc(binary: Path) -> tuple[int, int]:
    versions = {
        (int(major), int(minor))
        for major, minor in re.findall(
            r"GLIBC_(\d+)\.(\d+)", command_text(["readelf", "--version-info", str(binary)])
        )
    }
    if not versions:
        fail(f"{binary}: readelf reported no GLIBC requirements")
    return max(versions)


def dynamic_libraries(binary: Path) -> set[str]:
    libraries = set(
        re.findall(
            r"Shared library: \[([^]]+)\]",
            command_text(["readelf", "--dynamic", str(binary)]),
        )
    )
    missing = REQUIRED_LIBRARIES - libraries
    unexpected = libraries - ALLOWED_LIBRARIES
    if missing or unexpected:
        fail(
            f"{binary}: dynamic libraries missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return libraries


def validate_binary(
    binary: Path,
    expected: str,
    max_glibc: tuple[int, int],
    environment: dict[str, str] | None,
) -> tuple[int, int]:
    header = command_text(["readelf", "--file-header", str(binary)])
    if "Class:                             ELF64" not in header:
        fail(f"{binary}: server is not an ELF64 executable")
    if "Machine:                           Advanced Micro Devices X86-64" not in header:
        fail(f"{binary}: server is not an x86-64 executable")
    dynamic_libraries(binary)

    observed_glibc = required_glibc(binary)
    if observed_glibc > max_glibc:
        fail(
            f"{binary}: requires GLIBC_{observed_glibc[0]}.{observed_glibc[1]}, "
            f"above GLIBC_{max_glibc[0]}.{max_glibc[1]}"
        )

    version = command_text([str(binary), "--version"], environment=environment).strip()
    if version != f"{BINARY_NAME} {expected}":
        fail(f"{binary}: --version returned {version!r}")
    help_text = command_text([str(binary), "--help"], environment=environment)
    if "tuiskollm serve SNAPSHOT [ADDRESS]" not in help_text:
        fail(f"{binary}: --help omitted the serve command")

    binary_bytes = binary.read_bytes()
    leaked = [path.decode() for path in BUILD_PATHS if path in binary_bytes]
    if leaked:
        fail(f"{binary}: executable leaks build paths {leaked}")
    return observed_glibc


def field(text: str, name: str) -> str:
    prefix = f"{name}: "
    value = next(
        (line.removeprefix(prefix) for line in text.splitlines() if line.startswith(prefix)),
        None,
    )
    if value is None:
        fail(f"command output omitted {name!r}")
    return value


def cuda_version() -> tuple[str, str]:
    output = command_text(["ptxas", "--version"])
    match = re.search(r"release ([^,]+), V([^\s]+)", output)
    if match is None:
        fail("ptxas --version omitted the CUDA Toolkit identity")
    return match.group(1), match.group(2)


def build_information(
    binary: Path,
    expected: str,
    observed_glibc: tuple[int, int],
    git_commit: str,
    cuda_oxide_commit: str,
    build_host: str,
) -> bytes:
    rustc = command_text(["rustc", "-vV"])
    cuda_release, cuda_version_text = cuda_version()
    clang = command_text(["clang-21", "--version"]).splitlines()[0]
    values = {
        "artifact": BINARY_NAME,
        "version": expected,
        "archive_target": ARCHIVE_TARGET,
        "rust_target": "x86_64-unknown-linux-gnu",
        "required_glibc": f"{observed_glibc[0]}.{observed_glibc[1]}",
        "git_commit": git_commit,
        "rustc_release": field(rustc, "release"),
        "rustc_commit": field(rustc, "commit-hash"),
        "cuda_toolkit_release": cuda_release,
        "cuda_toolkit_version": cuda_version_text,
        "cuda_oxide_commit": cuda_oxide_commit,
        "clang": clang,
        "build_host": build_host,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
    return "".join(f"{name}={value}\n" for name, value in values.items()).encode()


def tar_entry(name: str, contents: bytes, mode: int, epoch: int) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(contents)
    info.mode = mode
    info.mtime = epoch
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info, contents


def write_archive(
    archive: Path,
    root: str,
    binary: Path,
    build_info: bytes,
    epoch: int,
) -> None:
    entries = [
        tar_entry(f"{root}/BUILD-INFO.txt", build_info, 0o644, epoch),
        tar_entry(
            f"{root}/LICENSE-APACHE",
            (WORKSPACE / "LICENSE-APACHE").read_bytes(),
            0o644,
            epoch,
        ),
        tar_entry(f"{root}/LICENSE-MIT", (WORKSPACE / "LICENSE-MIT").read_bytes(), 0o644, epoch),
        tar_entry(f"{root}/README.md", (WORKSPACE / "README.md").read_bytes(), 0o644, epoch),
        tar_entry(f"{root}/{BINARY_NAME}", binary.read_bytes(), 0o755, epoch),
    ]
    with archive.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for info, contents in entries:
                    tar.addfile(info, io.BytesIO(contents))


def verify_archive(archive: Path, checksum: Path, expected: str) -> None:
    expected_archive = f"{BINARY_NAME}-{expected}-{ARCHIVE_TARGET}.tar.gz"
    if archive.name != expected_archive:
        fail(f"{archive}: expected archive filename {expected_archive!r}")
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        fail(f"{archive}: archive exceeds {MAX_ARCHIVE_BYTES // 1024 // 1024} MiB")

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected_checksum = f"{digest}  {archive.name}\n"
    if checksum.name != f"{archive.name}.sha256" or checksum.read_text() != expected_checksum:
        fail(f"{checksum}: checksum does not describe {archive.name}")

    root = f"{BINARY_NAME}-{expected}-{ARCHIVE_TARGET}"
    required = {
        f"{root}/BUILD-INFO.txt",
        f"{root}/LICENSE-APACHE",
        f"{root}/LICENSE-MIT",
        f"{root}/README.md",
        f"{root}/{BINARY_NAME}",
    }
    with tarfile.open(archive, "r:gz") as packaged:
        members = packaged.getmembers()
        names = {member.name for member in members}
        unsafe = [
            member.name
            for member in members
            if Path(member.name).is_absolute()
            or ".." in Path(member.name).parts
            or member.issym()
            or member.islnk()
        ]
        if unsafe:
            fail(f"{archive}: unsafe members {unsafe}")
        if names != required:
            fail(
                f"{archive}: members differ: missing={sorted(required - names)}, "
                f"extra={sorted(names - required)}"
            )
        non_files = [member.name for member in members if not member.isfile()]
        if non_files:
            fail(f"{archive}: non-file members {non_files}")
        binary_member = packaged.getmember(f"{root}/{BINARY_NAME}")
        if binary_member.mode != 0o755:
            fail(f"{archive}: packaged server mode is {oct(binary_member.mode)}, expected 0o755")
        binary_file = packaged.extractfile(binary_member)
        build_info_file = packaged.extractfile(f"{root}/BUILD-INFO.txt")
        if binary_file is None or build_info_file is None:
            fail(f"{archive}: failed to read packaged release metadata")
        binary_digest = hashlib.sha256(binary_file.read()).hexdigest()
        build_info = dict(
            line.split("=", 1)
            for line in build_info_file.read().decode().splitlines()
            if "=" in line
        )
        if build_info.get("artifact") != BINARY_NAME or build_info.get("version") != expected:
            fail(f"{archive}: BUILD-INFO.txt identifies the wrong artifact")
        if build_info.get("binary_sha256") != binary_digest:
            fail(f"{archive}: BUILD-INFO.txt has the wrong binary digest")

    print(f"{archive}: {archive.stat().st_size / 1024 / 1024:.1f} MiB, sha256 {digest}")


def package(args: argparse.Namespace) -> None:
    binary = cast(Path, args.binary)
    expected = cast(str, args.expected)
    if expected != workspace_version():
        fail(
            f"expected version {expected!r} does not match "
            f"workspace version {workspace_version()!r}"
        )
    if not binary.is_file():
        fail(f"server binary does not exist: {binary}")

    output = cast(Path, args.out)
    output.mkdir(parents=True, exist_ok=True)
    root = f"{BINARY_NAME}-{expected}-{ARCHIVE_TARGET}"
    archive = output / f"{root}.tar.gz"
    checksum = output / f"{archive.name}.sha256"
    max_glibc = parse_glibc(cast(str, args.max_glibc))
    git_commit = checked_identity("Git commit", cast(str, args.git_commit), r"[0-9a-f]{40}")
    cuda_oxide_commit = checked_identity(
        "cuda-oxide commit",
        cast(str, args.cuda_oxide_commit),
        r"[0-9a-f]{40}",
    )
    build_host = checked_identity(
        "build host",
        cast(str, args.build_host),
        r"[A-Za-z0-9][A-Za-z0-9._/-]*",
    )
    source_date_epoch = cast(int, args.source_date_epoch)
    if source_date_epoch < 0:
        fail("source date epoch must not be negative")

    with tempfile.TemporaryDirectory(prefix="tuiskollm-release-", dir=output) as temporary:
        temporary_path = Path(temporary)
        stripped = temporary_path / BINARY_NAME
        command_text(["strip", "--strip-all", "-o", str(stripped), str(binary)])
        stripped.chmod(0o755)
        environment = None
        cuda_stub = cast(Path | None, args.cuda_stub)
        if cuda_stub is not None:
            if not cuda_stub.is_file():
                fail(f"CUDA driver stub does not exist: {cuda_stub}")
            stub_link = temporary_path / "libcuda.so.1"
            stub_link.symlink_to(cuda_stub.resolve())
            environment = os.environ.copy()
            prior_path = environment.get("LD_LIBRARY_PATH")
            environment["LD_LIBRARY_PATH"] = (
                f"{temporary_path}:{prior_path}" if prior_path else str(temporary_path)
            )
        observed_glibc = validate_binary(stripped, expected, max_glibc, environment)
        build_info = build_information(
            stripped,
            expected,
            observed_glibc,
            git_commit,
            cuda_oxide_commit,
            build_host,
        )
        write_archive(archive, root, stripped, build_info, source_date_epoch)

    checksum.write_text(f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n")
    verify_archive(archive, checksum, expected)


def verify(args: argparse.Namespace) -> None:
    verify_archive(cast(Path, args.archive), cast(Path, args.checksum), cast(str, args.expected))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version")

    package_command = commands.add_parser("package")
    package_command.add_argument("--binary", required=True, type=Path)
    package_command.add_argument("--out", required=True, type=Path)
    package_command.add_argument("--expected", required=True)
    package_command.add_argument("--source-date-epoch", required=True, type=int)
    package_command.add_argument("--git-commit", required=True)
    package_command.add_argument("--cuda-oxide-commit", required=True)
    package_command.add_argument("--build-host", required=True)
    package_command.add_argument("--cuda-stub", type=Path)
    package_command.add_argument("--max-glibc", default="2.35")

    verify_command = commands.add_parser("verify")
    verify_command.add_argument("--archive", required=True, type=Path)
    verify_command.add_argument("--checksum", required=True, type=Path)
    verify_command.add_argument("--expected", required=True)

    args = parser.parse_args()
    command = cast(str, args.command)
    if command == "version":
        print(workspace_version())
    elif command == "package":
        package(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
