"""Correctness-gated benchmark for mojo-backports-tarfile.

Every case checks exact agreement with the reference before timing, so a
regression in the Mojo kernels shows up as a correctness failure rather than a
suspiciously good number.

Two kinds of baseline appear here, and they are not interchangeable:

* per-call baselines (`calc_chksums`, `nti`) compare the port's single-call
  API against the equivalent single call in `backports.tarfile`, which is how
  each library is actually used;
* batched NumPy baselines are the fastest reasonable vectorised formulation of
  the same byte arithmetic, and they are reported even when the Mojo path
  loses to them.

Where the Mojo side loses, the table says so.
"""

from __future__ import annotations

import pathlib
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_backports_tarfile as mbt  # noqa: E402
from mojo_backports_tarfile import _lib  # noqa: E402

import backports.tarfile as tarfile

BLOCKSIZE = 512
_PAX_RE = re.compile(br"(\d+) ([^=]+)=")


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _archive(nblocks: int) -> bytes:
    """A run of distinct, well-formed 512-byte header blocks."""
    blocks = []
    for i in range(nblocks):
        name = f"file{i:06d}.bin"
        header = bytearray(BLOCKSIZE)
        header[0:len(name)] = name.encode()
        header[100:108] = mbt.itn(0o644, 8)
        header[108:116] = mbt.itn(1000, 8)
        header[116:124] = mbt.itn(1000, 8)
        header[124:136] = mbt.itn(i * 7, 12)
        header[136:148] = mbt.itn(1600000000 + i, 12)
        header[148:156] = b" " * 8
        header[156:157] = b"0"
        header[257:263] = b"ustar\x00"
        chk = mbt.calc_chksums(bytes(header))[0]
        header[148:156] = ("%06o\x00 " % chk).encode()
        blocks.append(bytes(header))
    return b"".join(blocks)


def bench_scan_headers(nblocks: int = 1 << 14):
    """Per-block null counts, both checksums and number fields, in one call.

    Baseline: a fully vectorised NumPy pass over the same (nblocks, 512) uint8
    array, with the checksum field blanked so the sum is one reduction, plus a
    vectorised ASCII-octal decode of the size field.
    """
    data = _archive(nblocks)
    raw = np.frombuffer(data, dtype=np.uint8).reshape(nblocks, BLOCKSIZE)
    powers = 8 ** np.arange(11, dtype=np.int64)

    got = mbt.scan_headers(data, nblocks, BLOCKSIZE)
    blanked = np.zeros((nblocks, BLOCKSIZE), dtype=np.uint8)
    blanked[:, :148] = raw[:, :148]
    blanked[:, 156:] = raw[:, 156:]
    assert np.array_equal(got[:, 0], (raw == 0).sum(axis=1))
    assert np.array_equal(got[:, 1], blanked.sum(axis=1) + 256)
    digits = np.where(raw[:, 124:135] >= 0x30, raw[:, 124:135] - 0x30, 0)
    assert np.array_equal(got[:, 4], digits[:, ::-1] @ powers)

    def numpy_scan():
        b = np.zeros((nblocks, BLOCKSIZE), dtype=np.uint8)
        b[:, :148] = raw[:, :148]
        b[:, 156:] = raw[:, 156:]
        chk = b.sum(axis=1) + 256
        nulls = (raw == 0).sum(axis=1)
        d = np.where(raw[:, 124:135] >= 0x30, raw[:, 124:135] - 0x30, 0)
        return nulls, chk, d[:, ::-1] @ powers

    return (
        f"scan_headers n={nblocks}",
        _time(numpy_scan, 3),
        _time(lambda: mbt.scan_headers(data, nblocks, BLOCKSIZE), 3),
    )


def bench_calc_chksums(nblocks: int = 1 << 12):
    """One 512-byte checksum per call, against the same call upstream."""
    data = _archive(nblocks)
    got = np.array([mbt.calc_chksums(data[i * 512:(i + 1) * 512])[0]
                    for i in range(nblocks)])
    want = np.array([tarfile.calc_chksums(data[i * 512:(i + 1) * 512])[0]
                     for i in range(nblocks)])
    assert np.array_equal(got, want)

    def mine():
        for i in range(nblocks):
            mbt.calc_chksums(data[i * 512:(i + 1) * 512])

    def theirs():
        for i in range(nblocks):
            tarfile.calc_chksums(data[i * 512:(i + 1) * 512])

    return (
        f"calc_chksums x{nblocks}",
        _time(theirs, 3),
        _time(mine, 3),
    )


def _field_run(nfields: int) -> tuple[bytes, np.ndarray]:
    """`nfields` distinct 12-byte octal number fields, concatenated."""
    values = (np.arange(nfields, dtype=np.int64) * 7919) % (8 ** 11)
    buf = np.empty((nfields, 12), dtype=np.uint8)
    for i, v in enumerate(values):
        buf[i] = np.frombuffer(mbt.itn(int(v), 12), dtype=np.uint8)
    return buf.tobytes(), values


def bench_nti(nfields: int = 1 << 14):
    """One number field per call, against the same call upstream."""
    data, values = _field_run(nfields)
    for i in (0, 1, nfields // 2, nfields - 1):
        assert mbt.nti(data[i * 12:(i + 1) * 12]) == int(values[i])
        assert mbt.nti(data[i * 12:(i + 1) * 12]) == tarfile.nti(
            data[i * 12:(i + 1) * 12]
        )

    def mine():
        for i in range(nfields):
            mbt.nti(data[i * 12:(i + 1) * 12])

    def theirs():
        for i in range(nfields):
            tarfile.nti(data[i * 12:(i + 1) * 12])

    return (
        f"nti per field x{nfields}",
        _time(theirs, 3),
        _time(mine, 3),
    )


def _pax_payload(nrecords: int) -> bytes:
    out = bytearray()
    for i in range(nrecords):
        body = b"path=" + (f"dir{i:05d}/".encode() * 8) + b"\n"
        length = len(body) + 1
        while len(str(length)) + len(body) + 1 != length:
            length = len(str(length)) + len(body) + 1
        out += b"%d %s" % (length, body)
    return bytes(out)


def _regex_frames(payload: bytes):
    out = []
    pos = 0
    while match := _PAX_RE.match(payload, pos):
        length, keyword = match.groups()
        length = int(length)
        if length == 0:
            break
        out.append((keyword, payload[match.end(2) + 1:match.start(1) + length - 1]))
        pos += length
    return out


def bench_pax_framing(nrecords: int = 1 << 12):
    """The record framing scan itself, against the regex loop it replaces."""
    payload = _pax_payload(nrecords)
    raw = np.frombuffer(payload, dtype=np.uint8)
    capacity = len(payload) // 6
    out = np.empty((capacity, 6), dtype=np.int64)
    count = np.zeros(1, dtype=np.int64)
    status = np.zeros(1, dtype=np.int64)

    def kernel_frames():
        _lib.lib.tf_pax_scan(raw.ctypes.data, raw.size, capacity, out.ctypes.data,
                             count.ctypes.data, status.ctypes.data)
        return int(count[0])

    assert kernel_frames() == len(_regex_frames(payload))
    # The ranges the kernel reports must index the same bytes the regex slices.
    _lib.lib.tf_pax_scan(raw.ctypes.data, raw.size, capacity, out.ctypes.data,
                         count.ctypes.data, status.ctypes.data)
    for i, (kw, val) in enumerate(_regex_frames(payload)):
        assert raw[out[i, 2]:out[i, 3]].tobytes() == kw
        assert raw[out[i, 4]:out[i, 5]].tobytes() == val

    return (
        f"pax framing n={nrecords}",
        _time(lambda: _regex_frames(payload), 3),
        _time(kernel_frames, 3),
    )


def bench_pax_records(nrecords: int = 1 << 12):
    """The whole `pax_records` call, including the Python that decodes it."""
    payload = _pax_payload(nrecords)
    want = [(kw.decode(), val) for kw, val in _regex_frames(payload)]
    assert mbt.pax_records(payload) == want

    return (
        f"pax_records (end to end) n={nrecords}",
        _time(lambda: _regex_frames(payload), 3),
        _time(lambda: mbt.pax_records(payload), 3),
    )


def bench_nti_batched(nfields: int = 1 << 14):
    """Decode a run of fixed-width fields, against vectorised NumPy."""
    data, values = _field_run(nfields)
    got, status = mbt.nti_fields(data, nfields, 12)
    assert np.array_equal(got, values)
    assert not status.any()

    powers = 8 ** np.arange(11, dtype=np.int64)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(nfields, 12)

    def numpy_batched():
        d = arr[:, :11] - 0x30
        return d[:, ::-1] @ powers

    return (
        f"nti batched x{nfields}",
        _time(numpy_batched, 3),
        _time(lambda: mbt.nti_fields(data, nfields, 12), 3),
    )


def main():
    print(f"{'case':<38}{'reference':>12}{'mojo-backports-tarfile':>24}{'ratio':>10}")
    print("-" * 84)
    for fn in (
        bench_scan_headers,
        bench_calc_chksums,
        bench_nti,
        bench_pax_framing,
        bench_pax_records,
        bench_nti_batched,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<38}{ref*1e3:>10.2f}ms{got*1e3:>22.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
