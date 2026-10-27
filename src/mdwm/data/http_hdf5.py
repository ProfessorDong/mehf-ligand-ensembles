"""Read a remote HDF5 file over HTTP range requests, without downloading it.

The MISATO trajectory archive is 132.8 GB.  Every window this project needs from
it touches a few megabytes per system, so downloading the archive would spend
two orders of magnitude more disk and bandwidth than the task requires -- and the
root filesystem does not have room for it in any case.  HDF5 is a seekable
format, so h5py's ``fileobj`` driver can drive a file-like object that fetches
only the byte ranges the B-tree and the requested chunks actually touch.

Stdlib only, deliberately: the ``MDWM`` environment produced frozen,
hash-chained results and must not gain new dependencies.  ``urllib`` is
sufficient.

Two properties of the Zenodo endpoint shape this implementation:

* It serves ``206 Partial Content`` for range requests but does **not**
  advertise ``Accept-Ranges``.  A reader must therefore verify that ranges are
  honoured rather than trusting the header -- a server that silently ignores
  ``Range`` answers ``200`` with the entire 132.8 GB body, which would be a
  catastrophic accidental download.  :class:`HTTPRangeReader` refuses any
  response that is not ``206`` with a matching ``Content-Range``.
* Enumerating the root group of the MISATO file (13,765 members) is
  pathologically slow over HTTP.  Address groups directly by system id; never
  call ``list(handle.keys())`` on the root.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any

DEFAULT_BLOCK_BYTES = 1 << 20
DEFAULT_CACHE_BLOCKS = 4096


class RangeRequestNotHonored(RuntimeError):
    """The server answered a range request with something other than 206."""


class RateLimited(RuntimeError):
    """The server refused with 429/503 and the retry budget was exhausted."""


# Throttling: the server is telling us to slow down, so back off hard and long.
THROTTLE_STATUS = (429, 503)
# Transient gateway faults: worth retrying on the ordinary short schedule.
# Zenodo returns 504 for range requests that are too large for its proxy, which
# is why block_bytes cannot simply be raised without measuring.
TRANSIENT_STATUS = (500, 502, 504)


class HTTPRangeReader(io.RawIOBase):
    """A seekable, read-only file object backed by HTTP range requests.

    Reads are served from a block cache with least-recently-used eviction, so
    the B-tree nodes that HDF5 revisits constantly are fetched once.  The
    cumulative request count and byte count are exposed for cost reporting.
    """

    def __init__(
        self,
        url: str,
        *,
        size: int | None = None,
        block_bytes: int = DEFAULT_BLOCK_BYTES,
        cache_blocks: int = DEFAULT_CACHE_BLOCKS,
        max_attempts: int = 6,
        timeout_seconds: float = 120.0,
        backoff_seconds: float = 1.0,
        rate_limit_backoff_seconds: float = 10.0,
        min_request_interval_seconds: float = 0.0,
        opener: Any | None = None,
    ) -> None:
        if block_bytes < 1 or cache_blocks < 1 or max_attempts < 1:
            raise ValueError("block size, cache size and attempt count must be positive")
        self.url = str(url)
        self._block = int(block_bytes)
        self._cache_blocks = int(cache_blocks)
        self._max_attempts = int(max_attempts)
        self._timeout = float(timeout_seconds)
        self._backoff = float(backoff_seconds)
        # Zenodo throttles by REQUEST COUNT, not bytes, and answers 429 once the
        # budget is spent.  A 429 is not a transient network blip: retrying it on
        # the same short backoff just spends more of the budget, so it gets its
        # own much longer schedule and honours Retry-After when the server sends
        # one.  The politeness interval bounds this reader's own request rate.
        self._rate_limit_backoff = float(rate_limit_backoff_seconds)
        self._min_interval = float(min_request_interval_seconds)
        self._last_request_at = 0.0
        self.rate_limited_responses = 0
        self._open = opener if opener is not None else urllib.request.urlopen
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._position = 0
        self.bytes_fetched = 0
        self.requests = 0
        self.retries = 0
        self._size = int(size) if size is not None else self._head_size()

    # -- politeness ------------------------------------------------------
    def _await_slot(self) -> None:
        if self._min_interval <= 0.0:
            return
        wait = self._last_request_at + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _retry_after_seconds(error: Any, fallback: float) -> float:
        header = getattr(error, "headers", None)
        value = None if header is None else header.get("Retry-After")
        try:
            return max(float(value), 0.0) if value is not None else fallback
        except (TypeError, ValueError):
            return fallback

    # -- size discovery -------------------------------------------------
    def _head_size(self) -> int:
        """Ask for the file size, with the same retry budget as a data read.

        This is the request every worker issues simultaneously at start-up, so
        leaving it unprotected is exactly how a pool dies on its first 429.
        """
        request = urllib.request.Request(self.url, method="HEAD")
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                self._await_slot()
                response = self._open(request, timeout=self._timeout)
                length = response.headers.get("Content-Length")
                if length is None:
                    raise RuntimeError(f"Remote file does not report a size: {self.url}")
                return int(length)
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code in TRANSIENT_STATUS:
                    if attempt + 1 == self._max_attempts:
                        raise
                    self.retries += 1
                    time.sleep(self._backoff * (2**attempt))
                    continue
                if error.code not in THROTTLE_STATUS or attempt + 1 == self._max_attempts:
                    raise
                self.rate_limited_responses += 1
                self.retries += 1
                time.sleep(
                    self._retry_after_seconds(
                        error, self._rate_limit_backoff * (2**attempt)
                    )
                )
            except (urllib.error.URLError, OSError, TimeoutError) as error:
                last_error = error
                if attempt + 1 == self._max_attempts:
                    break
                self.retries += 1
                time.sleep(self._backoff * (2**attempt))
        raise RuntimeError(
            f"Could not read the remote size after {self._max_attempts} attempts: "
            f"{last_error}"
        ) from last_error

    # -- block fetching -------------------------------------------------
    def _fetch(self, index: int) -> bytes:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        start = index * self._block
        stop = min(start + self._block, self._size) - 1
        if start > stop:
            return b""
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{stop}"}
        )
        expected = stop - start + 1
        data = self._read_with_retries(request, start, stop, expected)
        self.requests += 1
        self.bytes_fetched += len(data)
        self._cache[index] = data
        self._cache.move_to_end(index)
        while len(self._cache) > self._cache_blocks:
            self._cache.popitem(last=False)
        return data

    def _read_with_retries(
        self, request: urllib.request.Request, start: int, stop: int, expected: int
    ) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                self._await_slot()
                response = self._open(request, timeout=self._timeout)
                status = int(getattr(response, "status", 0) or 0)
                if status != 206:
                    # Never retry this: the server ignored Range and is about to
                    # hand back the entire file.  Failing loudly is the only
                    # safe response when the whole point is not to download it.
                    raise RangeRequestNotHonored(
                        f"Expected 206 Partial Content for bytes={start}-{stop}, "
                        f"got {status}; refusing to read a non-ranged body"
                    )
                content_range = response.headers.get("Content-Range")
                if content_range is not None and not content_range.startswith(
                    f"bytes {start}-{stop}/"
                ):
                    raise RangeRequestNotHonored(
                        f"Server returned {content_range!r} for bytes={start}-{stop}"
                    )
                data = response.read()
                if len(data) != expected:
                    raise RangeRequestNotHonored(
                        f"Range bytes={start}-{stop} returned {len(data)} of {expected} bytes"
                    )
                return data
            except RangeRequestNotHonored:
                raise
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code in TRANSIENT_STATUS:
                    if attempt + 1 == self._max_attempts:
                        raise
                    self.retries += 1
                    time.sleep(self._backoff * (2**attempt))
                    continue
                if error.code not in THROTTLE_STATUS:
                    raise
                if attempt + 1 == self._max_attempts:
                    raise RateLimited(
                        f"Range request bytes={start}-{stop} was rate limited "
                        f"after {self._max_attempts} attempts; reduce n_workers "
                        f"or raise block_bytes to spend fewer requests"
                    ) from error
                self.rate_limited_responses += 1
                self.retries += 1
                time.sleep(
                    self._retry_after_seconds(
                        error, self._rate_limit_backoff * (2**attempt)
                    )
                )
            except (urllib.error.URLError, OSError, TimeoutError) as error:
                last_error = error
                if attempt + 1 == self._max_attempts:
                    break
                self.retries += 1
                time.sleep(self._backoff * (2**attempt))
        raise RuntimeError(
            f"Range request bytes={start}-{stop} failed after "
            f"{self._max_attempts} attempts: {last_error}"
        ) from last_error

    # -- file protocol ---------------------------------------------------
    def readinto(self, buffer: Any) -> int:
        wanted = min(len(buffer), self._size - self._position)
        if wanted <= 0:
            return 0
        collected = bytearray()
        while len(collected) < wanted:
            offset = self._position + len(collected)
            block = self._fetch(offset // self._block)
            within = offset % self._block
            chunk = block[within : within + (wanted - len(collected))]
            if not chunk:
                break
            collected += chunk
        buffer[: len(collected)] = collected
        self._position += len(collected)
        return len(collected)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self._position = position
        return self._position

    def tell(self) -> int:
        return self._position

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def size(self) -> int:
        return self._size

    @property
    def cached_blocks(self) -> int:
        return len(self._cache)

    def transfer_statistics(self) -> dict[str, int | float]:
        """Network cost so far, for the run summary."""
        return {
            "requests": int(self.requests),
            "retries": int(self.retries),
            "rate_limited_responses": int(self.rate_limited_responses),
            "bytes_fetched": int(self.bytes_fetched),
            "megabytes_fetched": float(self.bytes_fetched / 1024.0**2),
            "block_bytes": int(self._block),
            "cache_blocks": int(self._cache_blocks),
            "cached_blocks_resident": int(len(self._cache)),
        }


def open_remote_hdf5(url: str, **kwargs: Any):
    """Open a remote HDF5 file for reading; returns ``(h5py.File, reader)``.

    The reader is returned alongside the handle because it carries the transfer
    statistics, and because the caller must keep it alive for the lifetime of
    the handle.
    """
    import h5py

    reader = HTTPRangeReader(url, **kwargs)
    handle = h5py.File(reader, "r")
    return handle, reader
