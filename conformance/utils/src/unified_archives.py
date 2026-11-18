# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving engine/version archives for the Unified source store."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
import shutil
import tarfile
import tempfile
import zlib
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import yaml

import execution_manifests
import impls
import capture_sources


DIRECTORY = "versions"
MAX_MEMBER = 32 * 1024 * 1024
MAX_DECODED = 512 * 1024 * 1024
MAX_COMPRESSED = 32 * 1024 * 1024
MAX_EXTENSION_METADATA = 64 * 1024
MAX_EXTENSION_HEADERS = 128
IO_CHUNK = 1024 * 1024


def _member(name: str) -> None:
    parts = name.split("/")
    if (
        "\\" in name or ":" in name
        or any(part in {"", ".", ".."} for part in parts)
        or not name.endswith(".yaml")
        or not (len(parts) == 3 and parts[0] == "families"
                and parts[2] not in {"inputs_and_golden.yaml", "index.yaml"}
                or len(parts) == 2 and parts[0] in {"executions", "receipts"})
    ):
        raise ValueError(f"invalid Unified archive member: {name}")


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _read_archive_snapshot(path: Path) -> tuple[bytes, tuple[int, int, int, int, int]]:
    if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"Unified source must be regular, links forbidden in path ancestors: {path}")
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        data = source.read(MAX_COMPRESSED + 1)
        after = os.fstat(source.fileno())
    current = path.stat()
    if len(data) > MAX_COMPRESSED:
        raise ValueError(f"Unified archive exceeds compressed limit: {path}")
    if (_stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)
            or len(data) != after.st_size):
        raise ValueError(f"Unified archive changed while being read: {path}")
    return data, _stat_identity(after)


def _read_archive(path: Path) -> bytes:
    return _read_archive_snapshot(path)[0]


def paths(root: Path) -> list[Path]:
    directory = root / DIRECTORY
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Unified versions must be a regular directory")
    result = sorted(directory.iterdir())
    for path in result:
        if (path.is_symlink() or not path.is_file()
                or re.fullmatch(r"(?:dynamo|vllm|sglang)-[0-9][a-zA-Z0-9.-]*\.tar\.gz", path.name) is None
                or path.stat().st_size > MAX_COMPRESSED):
            raise ValueError(f"invalid Unified version archive: {path}")
    return result


def _check_tar_declarations(payload, path: Path) -> None:
    # tarfile resolves extensions (and allocates their bodies) before yielding a
    # logical member. Inspect physical headers first without reading those bodies.
    extension_size = 0
    extension_headers = 0
    payload.seek(0, 2)
    payload_size = payload.tell()
    payload.seek(0)
    while header := payload.read(tarfile.BLOCKSIZE):
        if header == bytes(tarfile.BLOCKSIZE):
            break
        try:
            member = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        except tarfile.HeaderError as error:
            raise ValueError(f"invalid Unified tar header: {path}") from error
        if member.size < 0:
            raise ValueError(f"invalid Unified tar size: {path}")
        if member.type in {tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK,
                           tarfile.XHDTYPE, tarfile.XGLTYPE}:
            extension_size += member.size
            extension_headers += 1
            if (extension_size > MAX_EXTENSION_METADATA
                    or extension_headers > MAX_EXTENSION_HEADERS):
                raise ValueError(f"Unified tar extension metadata exceeds limit: {path}")
        else:
            extension_size = 0
            extension_headers = 0
            if member.type == tarfile.GNUTYPE_SPARSE or member.size > MAX_MEMBER:
                raise ValueError(f"invalid Unified archive member: {member.name}")
        next_header = payload.tell() + ((member.size + 511) // 512) * 512
        if next_header > payload_size:
            raise ValueError(f"truncated Unified tar member: {path}")
        payload.seek(next_header)
    payload.seek(0)


def members(path: Path, *, temporary_parent: Path | None = None, compressed_data: bytes | None = None):
    """Read one bounded tar payload from disk, rejecting links and trailing gzip data."""
    seen = set()
    compressed_data = _read_archive(path) if compressed_data is None else compressed_data
    if len(compressed_data) > MAX_COMPRESSED:
        raise ValueError(f"Unified archive exceeds compressed limit: {path}")
    decoder = zlib.decompressobj(wbits=31)
    with tempfile.TemporaryFile(dir=temporary_parent) as payload:
        decoded_size = 0
        with io.BytesIO(compressed_data) as compressed:
            while chunk := compressed.read(IO_CHUNK):
                pending = chunk
                while pending:
                    output = decoder.decompress(
                        pending, min(MAX_DECODED - decoded_size + 1, IO_CHUNK)
                    )
                    pending = decoder.unconsumed_tail
                    decoded_size += len(output)
                    if decoded_size > MAX_DECODED:
                        raise ValueError(f"Unified archive exceeds decoded limit: {path}")
                    payload.write(output)
                    if decoder.unused_data:
                        raise ValueError(f"trailing Unified gzip data: {path}")
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError(f"truncated or trailing Unified gzip data: {path}")
        _check_tar_declarations(payload, path)
        with tarfile.open(fileobj=payload, mode="r:") as archive:
            for member in archive:
                _member(member.name)
                if (not member.isfile() or member.linkname or member.pax_headers
                        or member.name in seen or member.size > MAX_MEMBER):
                    raise ValueError(f"invalid or duplicate Unified archive member: {member.name}")
                seen.add(member.name)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"missing Unified archive member: {member.name}")
                data = stream.read(MAX_MEMBER + 1)
                if len(data) != member.size:
                    raise ValueError(f"truncated Unified archive member: {member.name}")
                yield member.name, data
            if not seen:
                raise ValueError(f"empty Unified archive: {path}")
            payload.seek(0, 2)
            payload_size = payload.tell()
            payload.seek(archive.offset)
            if payload_size < archive.offset + 1024:
                raise ValueError(f"invalid Unified tar end markers: {path}")
            while chunk := payload.read(IO_CHUNK):
                if any(chunk):
                    raise ValueError(f"invalid Unified tar end markers: {path}")


def inventory(root: Path, *, member_callback=None, source_callback=None,
              temporary_parent: Path | None = None) -> list[dict]:
    rows = {}
    for path in sorted([*root.glob("families/*/*.yaml"),
                        *root.glob("executions/*.yaml"), *root.glob("receipts/*.yaml")]):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unified source must be a regular file: {path}")
        data, identity = _read_archive_snapshot(path)
        name = path.relative_to(root).as_posix()
        rows[name] = {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        if source_callback is not None:
            source_callback(name, identity)
        if member_callback is not None:
            member_callback(name, data, None)
    for path in paths(root):
        data, identity = _read_archive_snapshot(path)
        name = path.relative_to(root).as_posix()
        descriptor = {"path": path.relative_to(root).as_posix(),
                      "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        if source_callback is not None:
            source_callback(name, identity)
        for name, data in members(path, temporary_parent=temporary_parent, compressed_data=data):
            if name in rows:
                raise ValueError(f"duplicate Unified source: {name}")
            rows[name] = {"path": name, "sha256": hashlib.sha256(data).hexdigest(),
                          "size": len(data), "archive": descriptor}
            if member_callback is not None:
                member_callback(name, data, descriptor)
    return [rows[name] for name in sorted(rows)]


def read_sources(root: Path, resources: list[dict]):
    """Read selected members, admitting each archive once for the operation."""
    groups = defaultdict(list)
    checkout = root if capture_sources.is_current(root) else root.parent.parent
    current_paths = {}
    if capture_sources.is_current(checkout):
        current_paths = {row["source"]: checkout / row["path"]
                         for row in capture_sources.current(checkout)["captures"] if row["surface"] == "unified"}
    paths_seen = set()
    for resource in resources:
        path = Path(resource["path"])
        if path.is_absolute() or ".." in path.parts or resource["path"] in paths_seen:
            raise ValueError(f"invalid or duplicate source path: {path}")
        paths_seen.add(resource["path"])
        groups[resource.get("archive", {}).get("path")].append(resource)
    for archive_path, selected in groups.items():
        if archive_path is None:
            for resource in selected:
                logical = resource["path"] if root == checkout else capture_sources.STORE + "/" + resource["path"]
                data = _read_archive(current_paths.get(logical, root / resource["path"]))
                if len(data) != resource["size"] or hashlib.sha256(data).hexdigest() != resource["sha256"]:
                    raise ValueError("source member identity mismatch")
                yield resource, data
            continue
        path = Path(archive_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"invalid source archive path: {path}")
        archive = selected[0]["archive"]
        if any(resource["archive"] != archive for resource in selected):
            raise ValueError("inconsistent source archive pins")
        compressed = _read_archive(root / path)
        if len(compressed) != archive["size"] or hashlib.sha256(compressed).hexdigest() != archive["sha256"]:
            raise ValueError("source archive identity mismatch")
        pending = {Path(resource["path"]).relative_to(path.parent.parent).as_posix(): resource for resource in selected}
        for name, data in members(root / path, compressed_data=compressed):
            resource = pending.pop(name, None)
            if resource is None:
                continue
            if len(data) != resource["size"] or hashlib.sha256(data).hexdigest() != resource["sha256"]:
                raise ValueError("source member identity mismatch")
            yield resource, data
        if pending:
            raise ValueError(f"source archive members missing: {sorted(pending)}")


def read_source(root: Path, resource: dict) -> bytes:
    # Exhaust admission before returning even if the selected member is first.
    return list(read_sources(root, [resource]))[0][1]


@contextmanager
def expanded(root: Path, *, temporary_parent: Path | None = None, archive_names: set[str] | None = None):
    """Keep the existing semantic reader/writer operating on ordinary paths."""
    for entry in root.rglob("*"):
        if entry.is_symlink() or not (entry.is_dir() or entry.is_file()):
            raise ValueError(f"Unified source must be regular, links forbidden: {entry}")
    archives = paths(root)
    if archive_names is not None:
        unknown = archive_names - {path.name for path in archives}
        if unknown:
            raise ValueError(f"selected Unified archives missing: {sorted(unknown)}")
        archives = [path for path in archives if path.name in archive_names]
    if not archives:
        yield root
        return
    with tempfile.TemporaryDirectory(prefix="unified-versions-", dir=temporary_parent) as temporary:
        destination = Path(temporary) / root.name
        destination.mkdir()
        for path in root.iterdir():
            if path.name == DIRECTORY:
                continue
            if path.is_symlink():
                raise ValueError(f"Unified source link forbidden: {path}")
            if path.is_dir():
                for entry in path.rglob("*"):
                    if entry.is_symlink() or not (entry.is_dir() or entry.is_file()):
                        raise ValueError(f"Unified source must be regular: {entry}")
                shutil.copytree(path, destination / path.name)
            else:
                shutil.copyfile(path, destination / path.name)
        for archive in archives:
            archive_data = _read_archive(archive)
            for name, data in members(archive, temporary_parent=temporary_parent, compressed_data=archive_data):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(data)
        yield destination


def capture_archive_bytes(documents: dict[str, bytes]) -> bytes:
    """Publish portable gzip headers and stable tar metadata for independent captures."""
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            for name, data in sorted(documents.items()):
                capture_sources.relative(name)
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def engine(implementation: str) -> str:
    source = impls.BATCH_SOURCE_IMPL.get(implementation, implementation)
    return next(spec.engine for spec in impls.IMPL_SPECS if spec.key == source)


def _is_complete_snapshot(path: Path) -> bool:
    """Use the canonical header as a fast format marker, with YAML fallback."""
    with path.open("rb") as source:
        for _ in range(32):
            line = source.readline(4097)
            if not line or len(line) > 4096:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith(b"#") or stripped == b"---":
                continue
            if re.fullmatch(rb"schema_version:\s*2(?:\s+#.*)?", stripped):
                return True
            break
    document = yaml.safe_load(path.read_bytes())
    return isinstance(document, dict) and document.get("schema_version") == 2


def stage_selected(root: Path, staged: Path, changes: dict[str, bytes | None], capture_pattern, version_key) -> None:
    """Stage selected complete snapshots while sharing untouched source inodes."""
    shutil.copytree(root, staged, copy_function=os.link)
    groups = set()
    for name in changes:
        parts = Path(name).parts
        if len(parts) == 3 and parts[0] == "families" and parts[2] == "inputs_and_golden.yaml":
            _member(f"families/{parts[1]}/authored.yaml")
        else:
            _member(name)
        if len(parts) == 3 and parts[0] == "families" and parts[2] != "inputs_and_golden.yaml":
            identity = capture_pattern.fullmatch(Path(name).stem)
            if identity is None:
                raise ValueError(f"invalid complete snapshot identity: {name}")
            groups.add(f"{engine(identity['implementation'])}-{identity['runtime_version']}")
        elif not (len(parts) == 3 and parts[0] == "families" and parts[2] == "inputs_and_golden.yaml"):
            raise ValueError(f"complete snapshot has external metadata: {name}")
    with tempfile.TemporaryDirectory(prefix="selected-versions-", dir=staged.parent) as temporary:
        selected = Path(temporary)
        for group in sorted(groups):
            archive = root / DIRECTORY / f"{group}.tar.gz"
            if archive.exists():
                for name, data in members(archive):
                    if name in changes:
                        continue
                    target = selected / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                (staged / DIRECTORY / archive.name).unlink()
        for name, data in changes.items():
            path = Path(name)
            capture = path.name != "inputs_and_golden.yaml"
            target = (selected if capture else staged) / path
            old = staged / path
            if old.exists():
                old.unlink()
            if data is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        pack(selected, capture_pattern, version_key)
        for archive in paths(selected):
            destination = staged / DIRECTORY / archive.name
            destination.parent.mkdir(exist_ok=True)
            archive.rename(destination)


def pack(root: Path, capture_pattern, version_key) -> None:
    """Pack an unpublished flat generation; the caller owns atomic installation."""
    if paths(root):
        raise ValueError("pack requires an expanded staging generation")
    capture_paths = [path for path in sorted(root.glob("families/*/*.yaml"))
                     if path.name != "inputs_and_golden.yaml"]
    if capture_paths and all(_is_complete_snapshot(path) for path in capture_paths):
        if any((root / directory).exists() for directory in ("executions", "receipts")):
            raise ValueError("complete snapshots cannot use external execution or receipt files")
        groups = defaultdict(list)
        for path in capture_paths:
            identity = capture_pattern.fullmatch(path.stem)
            if identity is None:
                raise ValueError(f"invalid capture filename: {path}")
            implementation, version = identity["implementation"], identity["runtime_version"]
            groups[f"{engine(implementation)}-{version}"].append(path)
        directory = root / DIRECTORY
        directory.mkdir(exist_ok=True)
        for group, sources in sorted(groups.items()):
            destination = directory / f"{group}.tar.gz"
            decoded_size = sum(512 + ((path.stat().st_size + 511) // 512) * 512 for path in sources) + 10240
            if decoded_size > MAX_DECODED:
                raise ValueError(f"version archive exceeds decoded limit: {group}")
            with destination.open("wb") as output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                        for path in sources:
                            data = path.read_bytes()
                            if len(data) > MAX_MEMBER:
                                raise ValueError(f"Unified member exceeds limit: {path}")
                            name = path.relative_to(root).as_posix()
                            _member(name)
                            info = tarfile.TarInfo(name)
                            info.size = len(data)
                            info.mode = 0o644
                            info.uid = 0
                            info.gid = 0
                            info.uname = ""
                            info.gname = ""
                            info.mtime = 0
                            archive.addfile(info, io.BytesIO(data))
            if destination.stat().st_size > MAX_COMPRESSED:
                raise ValueError(f"version archive exceeds compressed limit: {group}")
            for path in sources:
                path.unlink()
        return
    if capture_paths and any(_is_complete_snapshot(path) for path in capture_paths):
        raise ValueError("cannot mix sparse captures and complete snapshots")
    histories = defaultdict(list)
    owners = defaultdict(set)
    live = set()
    manifests = {}
    # Retain metadata belonging to older, unreferenced manifests as evidence too.
    for path in sorted(root.glob("executions/*.yaml")):
        document = yaml.load(path.read_bytes(), Loader=yaml.CSafeLoader)
        name = path.relative_to(root).as_posix()
        group = f"{engine(document['implementation'])}-{document['version']}"
        dependencies = {execution_manifests.source_reference(reference, "receipts")
                        for reference in document.get("receipts", {})}
        manifests[name] = dependencies
        for dependency in dependencies | {name}:
            owners[dependency].add(group)
    for path in sorted(root.glob("families/*/*.yaml")):
        if path.name == "inputs_and_golden.yaml":
            continue
        identity = capture_pattern.fullmatch(path.stem)
        if identity is None:
            raise ValueError(f"invalid capture filename: {path}")
        implementation, version = identity["implementation"], identity["runtime_version"]
        group = f"{engine(implementation)}-{version}"
        name = path.relative_to(root).as_posix()
        document = yaml.load(path.read_bytes(), Loader=yaml.CSafeLoader)
        dependencies = {name}
        if reference := document.get("execution"):
            reference = execution_manifests.source_reference(reference, "executions")
            dependencies.add(reference)
            dependencies.update(manifests[reference])
        histories[path.parent.name, implementation].append((version, dependencies))
        for dependency in dependencies:
            owners[dependency].add(group)
    for history in histories.values():
        for _, dependencies in sorted(history, key=lambda item: version_key(item[0]))[-2:]:
            live.update(dependencies)
    groups = defaultdict(list)
    for path in sorted([*root.glob("families/*/*.yaml"),
                        *root.glob("executions/*.yaml"), *root.glob("receipts/*.yaml")]):
        name = path.relative_to(root).as_posix()
        if path.name == "inputs_and_golden.yaml" or name in live:
            continue
        groups_for_path = owners.get(name)
        if not groups_for_path:
            document = yaml.load(path.read_bytes(), Loader=yaml.CSafeLoader)
            if not document.get("implementation") or not document.get("version"):
                # An unreferenced receipt has no safe engine/version owner.
                # Keep it readable instead of dropping or relabelling evidence.
                continue
            groups_for_path = {f"{engine(document['implementation'])}-{document['version']}"}
        groups[sorted(groups_for_path)[0]].append(path)
    if not groups:
        return
    directory = root / DIRECTORY
    directory.mkdir(exist_ok=True)
    for group, sources in sorted(groups.items()):
        destination = directory / f"{group}.tar.gz"
        decoded_size = sum(512 + ((path.stat().st_size + 511) // 512) * 512 for path in sources) + 10240
        if decoded_size > MAX_DECODED:
            raise ValueError(f"version archive exceeds decoded limit: {group}")
        with destination.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                    for path in sources:
                        data = path.read_bytes()
                        if len(data) > MAX_MEMBER:
                            raise ValueError(f"Unified member exceeds limit: {path}")
                        info = tarfile.TarInfo(path.relative_to(root).as_posix())
                        info.size = len(data)
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(data))
        if destination.stat().st_size > MAX_COMPRESSED:
            raise ValueError(f"version archive exceeds compressed limit: {group}")
        for path in sources:
            path.unlink()
