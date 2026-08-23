#!/usr/bin/env python3

"""Validate TuiskoLLM Python release metadata and distributions."""

from __future__ import annotations

import argparse
import email
import importlib
import importlib.metadata
import subprocess
import tarfile
import tomllib
import zipfile
from email.message import Message
from pathlib import Path
from typing import NoReturn, cast

WORKSPACE = Path(__file__).resolve().parents[1]
DISTRIBUTION = "tuisko-llm"
TAG_PREFIX = "v"
WHEEL_STEM = "tuisko_llm"
WHEEL_TAG = "cp312-abi3-manylinux_2_34_x86_64"
SDIST_MAX_BYTES = 1024 * 1024
REQUIRED_LICENSES = {"LICENSE-APACHE", "LICENSE-MIT"}
REQUIRED_WHEEL_FILES = {
    "tuisko/llm/__init__.py",
    "tuisko/llm/__init__.pyi",
    "tuisko/llm/_native.abi3.so",
    "tuisko/llm/py.typed",
}
BUILD_PATHS = (
    str(WORKSPACE).encode(),
    b"/github/workspace",
    b"/home/runner/work/",
    b"/home/runner/.cargo",
    b"/home/runner/.rustup",
    b"/io/crates/",
    b"/io/python/",
    b"/root/.cargo",
    b"/root/.rustup",
)


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def workspace_version() -> str:
    with (WORKSPACE / "Cargo.toml").open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)

    workspace = manifest.get("workspace")
    if not isinstance(workspace, dict):
        fail("Cargo.toml does not define [workspace]")
    package = workspace.get("package")
    if not isinstance(package, dict):
        fail("Cargo.toml does not define [workspace.package]")
    version = package.get("version")
    if not isinstance(version, str):
        fail("Cargo.toml workspace package version is not a string")
    return version


def check_tag(tag: str) -> None:
    expected = f"{TAG_PREFIX}{workspace_version()}"
    if tag != expected:
        fail(f"tag {tag!r} does not match package version {expected!r}")


def archive_metadata(archive: zipfile.ZipFile, wheel: Path) -> Message:
    metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    if len(metadata_files) != 1:
        fail(f"{wheel}: expected one METADATA file, found {metadata_files}")
    return email.message_from_bytes(archive.read(metadata_files[0]))


def check_wheel(wheel: Path, expected: str, max_mib: int) -> None:
    expected_name = f"{WHEEL_STEM}-{expected}-{WHEEL_TAG}.whl"
    if wheel.name != expected_name:
        fail(f"{wheel}: expected wheel filename {expected_name!r}")

    size = wheel.stat().st_size
    max_bytes = max_mib * 1024 * 1024
    if size > max_bytes:
        fail(f"{wheel}: {size / 1024 / 1024:.1f} MiB exceeds {max_mib} MiB")

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing_files = REQUIRED_WHEEL_FILES - names
        if missing_files:
            fail(f"{wheel}: missing package files {sorted(missing_files)}")
        if any("tests" in Path(name).parts for name in names):
            fail(f"{wheel}: test files must not be packaged")

        metadata = archive_metadata(archive, wheel)
        if metadata["Name"] != DISTRIBUTION or metadata["Version"] != expected:
            fail(f"{wheel}: metadata is Name={metadata['Name']!r}, Version={metadata['Version']!r}")

        wheel_metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(wheel_metadata) != 1:
            fail(f"{wheel}: expected one WHEEL metadata file, found {wheel_metadata}")
        wheel_message = email.message_from_bytes(archive.read(wheel_metadata[0]))
        if wheel_message.get_all("Tag") != [WHEEL_TAG]:
            fail(f"{wheel}: wheel tags are {wheel_message.get_all('Tag')}")

        licenses = {Path(name).name for name in names if ".dist-info/licenses/" in name}
        missing_licenses = REQUIRED_LICENSES - licenses
        if missing_licenses:
            fail(f"{wheel}: missing license files {sorted(missing_licenses)}")

        extension = archive.read("tuisko/llm/_native.abi3.so")
        leaked_paths = [path.decode() for path in BUILD_PATHS if path in extension]
        if leaked_paths:
            fail(f"{wheel}: native extension leaks build paths {leaked_paths}")

    print(f"{wheel}: {size / 1024 / 1024:.1f} MiB, tag {WHEEL_TAG}")


def check_sdist(sdist: Path, expected: str) -> None:
    expected_name = f"{WHEEL_STEM}-{expected}.tar.gz"
    if sdist.name != expected_name:
        fail(f"{sdist}: expected source distribution filename {expected_name!r}")
    if sdist.stat().st_size > SDIST_MAX_BYTES:
        fail(f"{sdist}: source distribution exceeds 1 MiB")

    root = f"{WHEEL_STEM}-{expected}"
    required = {
        f"{root}/Cargo.lock",
        f"{root}/Cargo.toml",
        f"{root}/LICENSE-APACHE",
        f"{root}/LICENSE-MIT",
        f"{root}/docs/python.md",
        f"{root}/pyproject.toml",
        f"{root}/crates/tuisko-frontend/src/lib.rs",
        f"{root}/crates/tuisko-model/src/lib.rs",
        f"{root}/crates/tuisko-python/src/lib.rs",
        f"{root}/python/tuisko/llm/__init__.py",
        f"{root}/python/tuisko/llm/__init__.pyi",
        f"{root}/python/tuisko/llm/py.typed",
    }
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
        unsafe = [name for name in names if Path(name).is_absolute() or ".." in Path(name).parts]
        if unsafe:
            fail(f"{sdist}: unsafe archive paths {unsafe}")
        missing = required - names
        if missing:
            fail(f"{sdist}: missing source files {sorted(missing)}")

    print(f"{sdist}: {sdist.stat().st_size / 1024:.1f} KiB")


def check_artifacts(artifacts: list[Path], expected: str, max_wheel_mib: int) -> None:
    files = [artifact for artifact in artifacts if artifact.is_file()]
    wheels = [artifact for artifact in files if artifact.suffix == ".whl"]
    sdists = [artifact for artifact in files if artifact.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or len(files) != 2:
        fail(f"expected one wheel and one sdist, found {[path.name for path in files]}")
    check_wheel(wheels[0], expected, max_wheel_mib)
    check_sdist(sdists[0], expected)


def smoke_test(expected: str) -> None:
    llm = importlib.import_module("tuisko.llm")
    native = importlib.import_module("tuisko.llm._native")
    installed = importlib.metadata.version(DISTRIBUTION)
    module_version = getattr(llm, "__version__", None)
    if installed != expected or module_version != expected:
        fail(
            f"installed versions do not match {expected}: "
            f"metadata={installed!r}, module={module_version!r}"
        )
    if vars(llm).get("MODEL_ID") != "unsloth/Qwen3.8-27B-NVFP4":
        fail("installed wheel does not identify the exact model target")
    frontend = vars(llm).get("Frontend")
    if not isinstance(frontend, type) or frontend.__module__ != "tuisko.llm._native":
        fail("installed Frontend reports the wrong Python module")

    extension = getattr(native, "__file__", None)
    if not isinstance(extension, str):
        fail("installed native extension has no filesystem path")
    linkage = subprocess.run(
        ["ldd", extension],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.lower()
    forbidden = [
        name for name in ("libcuda", "libnvvm", "libnvidia", "libpython") if name in linkage
    ]
    if forbidden:
        fail(f"installed frontend wheel links forbidden libraries {forbidden}")
    print("installed wheel smoke test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version")

    tag = commands.add_parser("tag")
    tag.add_argument("tag")

    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("--expected", required=True)
    artifacts.add_argument("--max-wheel-mib", required=True, type=int)
    artifacts.add_argument("artifacts", nargs="+", type=Path)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--expected", required=True)

    args = parser.parse_args()
    command = cast(str, args.command)
    if command == "version":
        print(workspace_version())
    elif command == "tag":
        check_tag(cast(str, args.tag))
    elif command == "artifacts":
        check_artifacts(
            cast(list[Path], args.artifacts),
            cast(str, args.expected),
            cast(int, args.max_wheel_mib),
        )
    else:
        smoke_test(cast(str, args.expected))


if __name__ == "__main__":
    main()
