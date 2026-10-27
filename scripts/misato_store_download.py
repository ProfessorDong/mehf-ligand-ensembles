"""Resumable, checksum-verified download of the MISATO MD.hdf5 archive.

Builds the permanent raw store under ``data/MISATO/raw/``.  Nothing here is
specific to one analysis: the job is to land a byte-exact copy of the
publisher's file and prove it is byte-exact.

Design, and why
---------------
* **Chunked work queue, not one stream per part.**  The file is cut into
  fixed-size chunks (default 512 MiB) that a small pool of worker threads pulls
  from a queue.  Zenodo throttles *per connection* (measured: 1.1 MB/s on one
  stream, 5.1 MB/s aggregate on four, 21.6 MB/s aggregate on eight, all HTTP
  206, no 429), so concurrency is the only lever -- but a static per-worker
  partition would leave the run hostage to its slowest edge server.  A queue
  rebalances automatically.
* **One preallocated sparse target, written with ``os.pwrite``.**  Concatenating
  N part files at the end would need a second 124 GiB of disk on a shared
  filesystem whose free space has been observed to swing 325 GB in an hour.
  Writing every chunk into its final offset in one file needs no copy.
* **The target is ``MD.hdf5.part`` until the MD5 matches**, then it is
  ``os.replace``-d to ``MD.hdf5``.  So the presence of ``MD.hdf5`` is itself a
  checksum-verified statement, and no consumer can ever open a partial file
  under the final name.
* **Progress state is only ever advanced after ``os.fsync``.**  A chunk is
  marked done in ``download_state.json`` after its bytes are on the platter, so
  a hard kill can lose work but can never claim work it did not do.  The state
  file is written ``.tmp`` + ``os.replace``.
* **Stdlib only** (``urllib``), matching ``mdwm.data.http_hdf5``: the MDWM
  environment must not gain dependencies.

Usage
-----
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
    python scripts/misato_store_download.py --workers 8

    # re-verify an existing download without transferring anything
    python scripts/misato_store_download.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ZENODO_RECORD_ID = "7711953"
TARGET_FILE_KEY = "MD.hdf5"

GIB = 1024.0**3
THROTTLE_STATUS = (429, 503)
TRANSIENT_STATUS = (500, 502, 504)

_STOP = threading.Event()
# Set by the monitor when the disk margin is thin.  Workers park between read
# buffers, so a pause really does stop bytes landing on the filesystem instead
# of merely stopping the bookkeeping.
_PAUSE = threading.Event()


def _park_if_paused() -> None:
    while _PAUSE.is_set() and not _STOP.is_set():
        _STOP.wait(2.0)


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


def _install_signal_handlers() -> None:
    def _handler(signum: int, _frame: Any) -> None:
        _log(f"signal {signum} received: draining workers and persisting state")
        _STOP.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # not on the main thread of some hosts
            pass


# --------------------------------------------------------------------------- #
# publisher metadata
# --------------------------------------------------------------------------- #
def fetch_zenodo_file_record(
    record_id: str = ZENODO_RECORD_ID,
    file_key: str = TARGET_FILE_KEY,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """The publisher's own size and MD5 for MD.hdf5, from the Zenodo API.

    A size match is not evidence of a correct download -- a truncated-then-padded
    or a mid-stream-corrupted file can have the right length -- so the MD5 the
    publisher records is fetched here and kept with the data.
    """
    api_url = f"https://zenodo.org/api/records/{record_id}"
    request = urllib.request.Request(
        api_url, headers={"User-Agent": "mdwm-misato-store/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    record = json.loads(payload)
    if snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(snapshot_path, payload)
    for entry in record.get("files", []):
        if entry.get("key") == file_key:
            checksum = str(entry["checksum"])
            algorithm, _, digest = checksum.partition(":")
            if algorithm != "md5":
                raise RuntimeError(f"Zenodo reports a {algorithm} checksum, not md5")
            return {
                "key": entry["key"],
                "size": int(entry["size"]),
                "md5": digest,
                "url": str(entry["links"]["self"]),
                "record_id": record_id,
                "doi": record.get("doi"),
                "record_title": record.get("metadata", {}).get("title"),
                "record_version": record.get("metadata", {}).get("version"),
                "record_published": record.get("metadata", {}).get("publication_date"),
            }
    raise RuntimeError(f"{file_key} is not in Zenodo record {record_id}")


# --------------------------------------------------------------------------- #
# atomic state
# --------------------------------------------------------------------------- #
def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class DownloadState:
    """Chunk ledger for one file, persisted atomically.

    ``done`` is only ever set for a chunk whose bytes have been fsync'd, which
    is what makes a hard kill safe: the ledger under-claims rather than
    over-claims.
    """

    def __init__(self, path: Path, meta: dict[str, Any], chunk_bytes: int) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._dirty = False
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            if (
                int(state.get("size", -1)) != int(meta["size"])
                or str(state.get("md5_expected")) != str(meta["md5"])
                or int(state.get("chunk_bytes", -1)) != int(chunk_bytes)
            ):
                raise RuntimeError(
                    f"{path} describes a different file or chunking; refusing to "
                    "reuse it. Move it aside to start over."
                )
            self.state = state
        else:
            n_chunks = (int(meta["size"]) + chunk_bytes - 1) // chunk_bytes
            self.state = {
                "url": meta["url"],
                "size": int(meta["size"]),
                "md5_expected": meta["md5"],
                "zenodo_record_id": meta["record_id"],
                "zenodo_doi": meta.get("doi"),
                "chunk_bytes": int(chunk_bytes),
                "n_chunks": n_chunks,
                "done": [False] * n_chunks,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "verified_md5": None,
                "completed_utc": None,
                "history": [],
            }
            self.flush(force=True)

    # -- geometry ---------------------------------------------------------- #
    @property
    def size(self) -> int:
        return int(self.state["size"])

    @property
    def chunk_bytes(self) -> int:
        return int(self.state["chunk_bytes"])

    def chunk_span(self, index: int) -> tuple[int, int]:
        start = index * self.chunk_bytes
        return start, min(self.chunk_bytes, self.size - start)

    def pending_indices(self) -> list[int]:
        return [i for i, done in enumerate(self.state["done"]) if not done]

    def bytes_done(self) -> int:
        total = 0
        for index, done in enumerate(self.state["done"]):
            if done:
                total += self.chunk_span(index)[1]
        return total

    # -- mutation ---------------------------------------------------------- #
    def mark_done(self, index: int) -> None:
        with self._lock:
            self.state["done"][index] = True
            self._dirty = True

    def note(self, event: str, **fields: Any) -> None:
        with self._lock:
            self.state["history"].append(
                {
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event": event,
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    **fields,
                }
            )
            self.state["history"] = self.state["history"][-200:]
            self._dirty = True

    def set_verified(self, digest: str) -> None:
        with self._lock:
            self.state["verified_md5"] = digest
            self.state["completed_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            self._dirty = True
        self.flush(force=True)

    def flush(self, *, force: bool = False) -> None:
        with self._lock:
            if not (self._dirty or force):
                return
            self.state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.state["heartbeat_pid"] = os.getpid()
            payload = json.dumps(self.state, indent=2).encode("utf-8")
            self._dirty = False
        _atomic_write_bytes(self.path, payload)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
class DiskGuard:
    """Refuse to start, and pause mid-run, when the disk margin is thin.

    Free space on this filesystem is shared with another user and has been
    observed to move by hundreds of GB within an hour, so the check is both a
    pre-flight and a periodic one, and its failure mode is to pause and then
    exit cleanly -- never to write into a filesystem that is about to fill.
    """

    def __init__(self, path: Path, floor_gib: float, max_pause_minutes: float) -> None:
        self.path = path
        self.floor_bytes = int(floor_gib * GIB)
        self.max_pause_seconds = float(max_pause_minutes) * 60.0

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.path).free

    def preflight(self, remaining_bytes: int) -> None:
        free = self.free_bytes()
        needed = remaining_bytes + self.floor_bytes
        _log(
            f"disk pre-flight: free {free / GIB:.1f} GiB, need "
            f"{remaining_bytes / GIB:.1f} GiB + {self.floor_bytes / GIB:.1f} GiB floor "
            f"= {needed / GIB:.1f} GiB"
        )
        if free < needed:
            raise SystemExit(
                f"ABORT: {free / GIB:.1f} GiB free is below the "
                f"{needed / GIB:.1f} GiB required (remaining download + floor). "
                "Nothing was written."
            )

    def check_or_pause(self, remaining_bytes: int) -> bool:
        """True to keep going; False to stop cleanly.

        A violation parks the workers first (``_PAUSE``), so no further bytes
        land while the margin is thin, and only gives up -- cleanly, with the
        chunk ledger persisted -- if the margin does not recover.
        """
        free = self.free_bytes()
        if free >= remaining_bytes + self.floor_bytes:
            if _PAUSE.is_set():
                _PAUSE.clear()
                _log(f"RESUME: free {free / GIB:.1f} GiB is back above the floor")
            return True
        _PAUSE.set()
        waited = 0.0
        while not _STOP.is_set():
            free = self.free_bytes()
            if free >= remaining_bytes + self.floor_bytes:
                _PAUSE.clear()
                _log(f"RESUME: free {free / GIB:.1f} GiB is back above the floor")
                return True
            if waited >= self.max_pause_seconds:
                _log(
                    f"STOP: disk margin stayed below the floor for "
                    f"{waited / 60:.0f} min (free {free / GIB:.1f} GiB). "
                    "State is persisted; re-run to resume."
                )
                return False
            _log(
                f"PAUSE: free {free / GIB:.1f} GiB < remaining "
                f"{remaining_bytes / GIB:.1f} GiB + floor "
                f"{self.floor_bytes / GIB:.1f} GiB; workers parked, sleeping 60 s"
            )
            _STOP.wait(60.0)
            waited += 60.0
        return False


# --------------------------------------------------------------------------- #
# transfer
# --------------------------------------------------------------------------- #
class ChunkFetcher:
    def __init__(
        self,
        url: str,
        fd: int,
        *,
        read_buffer_bytes: int,
        max_attempts: int,
        timeout_seconds: float,
    ) -> None:
        self.url = url
        self.fd = fd
        self.read_buffer_bytes = int(read_buffer_bytes)
        self.max_attempts = int(max_attempts)
        self.timeout_seconds = float(timeout_seconds)
        self.bytes_transferred = 0
        self._counter_lock = threading.Lock()

    def _add_bytes(self, count: int) -> None:
        with self._counter_lock:
            self.bytes_transferred += count

    def fetch(self, start: int, length: int) -> None:
        """Write ``[start, start+length)`` of the remote file at that offset.

        Resumes within the chunk after a mid-chunk failure, so a dropped
        connection costs the bytes in flight and not the chunk.
        """
        written = 0
        attempt = 0
        while written < length:
            if _STOP.is_set():
                raise InterruptedError("stop requested")
            attempt += 1
            if attempt > self.max_attempts:
                raise RuntimeError(
                    f"chunk at {start} failed after {self.max_attempts} attempts "
                    f"({written}/{length} bytes)"
                )
            first = start + written
            last = start + length - 1
            request = urllib.request.Request(
                self.url,
                headers={
                    "Range": f"bytes={first}-{last}",
                    "User-Agent": "mdwm-misato-store/1.0",
                    "Accept-Encoding": "identity",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"expected 206 for a ranged request, got {response.status}; "
                            "refusing to treat a full-body response as a chunk"
                        )
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {first}-"):
                        raise RuntimeError(
                            f"server returned Content-Range {content_range!r} for "
                            f"bytes {first}-{last}"
                        )
                    offset = first
                    while True:
                        _park_if_paused()
                        if _STOP.is_set():
                            raise InterruptedError("stop requested")
                        block = response.read(self.read_buffer_bytes)
                        if not block:
                            break
                        view = memoryview(block)
                        while view:
                            count = os.pwrite(self.fd, view, offset)
                            view = view[count:]
                            offset += count
                        written = offset - start
                        self._add_bytes(len(block))
                        if written >= length:
                            break
            except InterruptedError:
                raise
            except urllib.error.HTTPError as error:
                if error.code in THROTTLE_STATUS:
                    delay = min(120.0, 10.0 * attempt)
                    _log(f"HTTP {error.code} at offset {first}: backing off {delay:.0f}s")
                    _STOP.wait(delay)
                    continue
                if error.code in TRANSIENT_STATUS:
                    delay = min(30.0, 2.0 * attempt)
                    _log(f"HTTP {error.code} at offset {first}: retrying in {delay:.0f}s")
                    _STOP.wait(delay)
                    continue
                raise
            except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as error:
                delay = min(30.0, 2.0 * attempt)
                _log(f"{type(error).__name__} at offset {first}: retrying in {delay:.0f}s")
                _STOP.wait(delay)
                continue
            if written < length:
                _log(f"short read at offset {first}: {written}/{length}; continuing")
        if written != length:
            raise RuntimeError(f"chunk at {start} wrote {written} of {length} bytes")


def md5_of_file(path: Path, *, buffer_bytes: int = 8 << 20, label: str = "md5") -> str:
    """Streaming MD5 with a fixed buffer, so memory does not scale with the file."""
    digest = hashlib.md5()
    total = path.stat().st_size
    seen = 0
    last = time.monotonic()
    with path.open("rb") as handle:
        while True:
            block = handle.read(buffer_bytes)
            if not block:
                break
            digest.update(block)
            seen += len(block)
            now = time.monotonic()
            if now - last > 60.0:
                _log(f"{label}: {seen / GIB:.1f}/{total / GIB:.1f} GiB hashed")
                last = now
    return digest.hexdigest()


def run_download(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    key = str(args.file_key)
    final_path = raw_dir / key
    part_path = raw_dir / f"{key}.part"
    state_path = raw_dir / (
        "download_state.json" if key == TARGET_FILE_KEY else f"download_state_{key}.json"
    )
    md5_path = raw_dir / f"{key}.md5"

    meta = fetch_zenodo_file_record(
        record_id=str(args.record_id),
        file_key=key,
        snapshot_path=raw_dir / f"zenodo_record_{args.record_id}.json",
    )
    _log(
        f"Zenodo record {meta['record_id']} ({meta['doi']}, v{meta['record_version']}): "
        f"{meta['key']} size={meta['size']} md5={meta['md5']}"
    )

    if final_path.is_file() and not args.force_reverify:
        if final_path.stat().st_size != meta["size"]:
            _log(
                f"ABORT: {final_path} exists with size {final_path.stat().st_size} "
                f"but Zenodo reports {meta['size']}. Not touching it."
            )
            return 2
        _log(f"{final_path} already present; verifying MD5 (this reads 124 GiB)")
        digest = md5_of_file(final_path, label="verify")
        if digest != meta["md5"]:
            _log(f"ABORT: MD5 mismatch on the existing file ({digest} != {meta['md5']})")
            return 2
        _log("MD5 matches the publisher's checksum; nothing to do")
        _write_md5_sidecar(md5_path, meta, digest, final_path)
        return 0

    state = DownloadState(state_path, meta, args.chunk_bytes)
    if not part_path.is_file():
        # Sparse preallocation: reserves the offsets without writing 124 GiB of
        # zeroes, so the disk fills only as real bytes land.
        with part_path.open("wb") as handle:
            handle.truncate(state.size)
        state.note("created_part_file")
    elif part_path.stat().st_size != state.size:
        with part_path.open("r+b") as handle:
            handle.truncate(state.size)
        state.note("retruncated_part_file")

    pending = state.pending_indices()
    remaining = sum(state.chunk_span(i)[1] for i in pending)
    _log(
        f"{len(state.state['done']) - len(pending)}/{state.state['n_chunks']} chunks "
        f"already done ({state.bytes_done() / GIB:.2f} GiB); "
        f"{remaining / GIB:.2f} GiB remaining"
    )
    if args.verify_only:
        _log("--verify-only: not transferring")
        if pending:
            _log(f"cannot verify: {len(pending)} chunks are still missing")
            return 1
        digest = md5_of_file(part_path, label="verify")
        _log(f"md5={digest} expected={meta['md5']} match={digest == meta['md5']}")
        return 0 if digest == meta["md5"] else 2

    guard = DiskGuard(raw_dir, args.disk_floor_gib, args.max_pause_minutes)
    guard.preflight(remaining)

    state.note("run_start", workers=args.workers, pending_chunks=len(pending))
    state.flush(force=True)

    if not pending:
        _log("all chunks present; skipping to verification")
    else:
        work: "queue.Queue[int]" = queue.Queue()
        for index in pending:
            work.put(index)
        errors: list[str] = []
        errors_lock = threading.Lock()

        fd = os.open(part_path, os.O_WRONLY)
        try:
            fetcher = ChunkFetcher(
                meta["url"],
                fd,
                read_buffer_bytes=args.read_buffer_bytes,
                max_attempts=args.max_attempts,
                timeout_seconds=args.timeout_seconds,
            )

            def worker(worker_id: int) -> None:
                while not _STOP.is_set():
                    _park_if_paused()
                    if _STOP.is_set():
                        return
                    try:
                        index = work.get_nowait()
                    except queue.Empty:
                        return
                    start, length = state.chunk_span(index)
                    try:
                        fetcher.fetch(start, length)
                        # fsync BEFORE the ledger admits the chunk: the ledger
                        # must never claim bytes that are not on disk.
                        os.fsync(fd)
                        state.mark_done(index)
                    except InterruptedError:
                        return
                    except Exception as error:  # noqa: BLE001
                        with errors_lock:
                            errors.append(f"chunk {index} @ {start}: {error}")
                        _log(f"worker {worker_id}: chunk {index} failed: {error}")
                        _STOP.set()
                        return
                    finally:
                        work.task_done()

            threads = [
                threading.Thread(target=worker, args=(i,), name=f"dl{i}", daemon=True)
                for i in range(args.workers)
            ]
            started = time.monotonic()
            bytes_at_start = state.bytes_done()
            for thread in threads:
                thread.start()

            last_report = time.monotonic()
            while any(thread.is_alive() for thread in threads):
                _STOP.wait(5.0)
                now = time.monotonic()
                if now - last_report >= args.report_seconds:
                    done_bytes = state.bytes_done()
                    elapsed = max(now - started, 1e-9)
                    rate = (done_bytes - bytes_at_start) / elapsed
                    left = state.size - done_bytes
                    eta_hours = (left / rate / 3600.0) if rate > 0 else float("inf")
                    _log(
                        f"progress {done_bytes / GIB:.2f}/{state.size / GIB:.2f} GiB "
                        f"({100.0 * done_bytes / state.size:.2f}%), "
                        f"{rate / 1e6:.2f} MB/s, ETA {eta_hours:.2f} h, "
                        f"free {guard.free_bytes() / GIB:.1f} GiB"
                    )
                    state.flush(force=True)
                    if not guard.check_or_pause(left):
                        _STOP.set()
                    last_report = now
                else:
                    state.flush()
            for thread in threads:
                thread.join(timeout=60.0)
        finally:
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            state.flush(force=True)

        if errors:
            state.note("run_failed", errors=errors[:10])
            state.flush(force=True)
            _log(f"STOP: {len(errors)} chunk failures; state persisted, re-run to resume")
            return 1
        if _STOP.is_set() and state.pending_indices():
            state.note("run_interrupted", pending=len(state.pending_indices()))
            state.flush(force=True)
            _log(
                f"INTERRUPTED with {len(state.pending_indices())} chunks pending; "
                "state persisted, re-run to resume"
            )
            return 1

    still_pending = state.pending_indices()
    if still_pending:
        _log(f"{len(still_pending)} chunks still pending; not verifying")
        return 1

    _log("all chunks present; computing MD5 of the assembled file")
    digest = md5_of_file(part_path, label="md5")
    if digest != meta["md5"]:
        state.note("md5_mismatch", computed=digest, expected=meta["md5"])
        state.flush(force=True)
        _log(
            f"ABORT: MD5 {digest} does not match the publisher's {meta['md5']}. "
            f"Leaving the file as {part_path} -- it is NOT published as "
            f"{final_path.name}."
        )
        return 2
    os.replace(part_path, final_path)
    state.set_verified(digest)
    _write_md5_sidecar(md5_path, meta, digest, final_path)
    _log(f"VERIFIED md5={digest}; published {final_path}")
    return 0


def _write_md5_sidecar(
    md5_path: Path, meta: dict[str, Any], digest: str, final_path: Path
) -> None:
    payload = {
        "file": str(final_path),
        "size_bytes": int(final_path.stat().st_size),
        "size_bytes_published": int(meta["size"]),
        "md5_published_by_zenodo": meta["md5"],
        "md5_computed_locally": digest,
        "match": digest == meta["md5"],
        "zenodo_record_id": meta["record_id"],
        "zenodo_doi": meta.get("doi"),
        "zenodo_record_version": meta.get("record_version"),
        "zenodo_url": meta["url"],
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_write_bytes(md5_path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    # A plain md5sum-compatible line as well, so `md5sum -c` works.
    _atomic_write_bytes(
        md5_path.with_suffix(".md5.txt"),
        f"{digest}  {final_path.name}\n".encode("utf-8"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/MISATO/raw")
    parser.add_argument("--record-id", default=ZENODO_RECORD_ID)
    parser.add_argument(
        "--file-key",
        default=TARGET_FILE_KEY,
        help="which file of the record to fetch; the smaller files of the same "
        "record (QM.hdf5, train_MD.txt) make this script testable end to end",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="concurrent HTTP streams; Zenodo throttles per connection "
        "(measured 21.6 MB/s aggregate at 8, no 429). Keep it small.",
    )
    parser.add_argument("--chunk-bytes", type=int, default=512 << 20)
    parser.add_argument("--read-buffer-bytes", type=int, default=4 << 20)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--report-seconds", type=float, default=120.0)
    parser.add_argument(
        "--disk-floor-gib",
        type=float,
        default=40.0,
        help="stop rather than let free space fall below this after the "
        "remaining download is accounted for",
    )
    parser.add_argument("--max-pause-minutes", type=float, default=120.0)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force-reverify", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 12:
        raise SystemExit("--workers must be between 1 and 12 (politeness to Zenodo)")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    _install_signal_handlers()
    sys.exit(run_download(args))


if __name__ == "__main__":
    main()
