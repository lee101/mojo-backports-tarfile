"""Header-codec API for the `backports.tarfile` subset, on Mojo kernels.

The port covers the byte-level arithmetic of the tar header codec: the two
512-byte checksums, the octal and GNU base-256 number fields in both
directions, the fixed-width field padding, and the record framing of a
POSIX.1-2008 pax extended header.

What is deliberately not here: the `TarFile` reader and writer, compression
plumbing, sparse-file handling, extraction filters, and the extraction data
model. Those are IO and policy, and the real `backports.tarfile` is the right
place for them.

Two range limits come from exposing number fields as 64-bit integers:
a base-256 field wider than 63 bits raises `OverflowError` (8-byte fields can
never hit it, since 256**7 is 2**56), and `pax_records` reports at most
`capacity` records, defaulting to `len(payload) // 6`.
"""

from __future__ import annotations

import numpy as np
from backports.tarfile import InvalidHeaderError

from . import _lib
from ._lib import BLOCKSIZE, GNU_FORMAT, calc_chksums, field_pad, itn, nti
from ._lib import nti_fields, pax_records, scan_headers

# Field offsets inside a 512-byte ustar/GNU header block.
OFF_NAME = 0
LEN_NAME = 100
OFF_MODE = 100
LEN_MODE = 8
OFF_UID = 108
LEN_UID = 8
OFF_GID = 116
LEN_GID = 8
OFF_SIZE = 124
LEN_SIZE = 12
OFF_MTIME = 136
LEN_MTIME = 12
OFF_CHKSUM = 148
LEN_CHKSUM = 8
OFF_TYPE = 156
OFF_LINKNAME = 157
OFF_MAGIC = 257
OFF_UNAME = 265
OFF_GNAME = 297
OFF_DEVMAJOR = 329
OFF_DEVMINOR = 337
OFF_PREFIX = 345
LEN_PREFIX = 155


def block(count: int) -> int:
    """`TarInfo._block`: round a byte count up to a whole 512-byte block."""
    blocks, remainder = divmod(count, BLOCKSIZE)
    if remainder:
        blocks += 1
    return blocks * BLOCKSIZE


def pad_payload(payload: bytes) -> bytes:
    """`TarInfo._create_payload`: NUL-pad up to the next 512-byte border."""
    remainder = len(payload) % BLOCKSIZE
    return payload if remainder == 0 else payload + (BLOCKSIZE - remainder) * b"\0"


def header_fields(block512: bytes) -> dict:
    """Decode the numeric fields of one 512-byte header block.

    Returns a dict with the null count, both checksums, the checksum stored in
    the block, and the mode/size/mtime fields. Raises `InvalidHeaderError` for
    an unparsable number field and `ValueError` for a wrong-sized block, the
    same way `TarInfo.frombuf` does.
    """
    if len(block512) != BLOCKSIZE:
        raise ValueError(f"header must be {BLOCKSIZE} bytes, got {len(block512)}")
    rows = scan_headers(block512, 1, BLOCKSIZE)[0]
    nulls, chk_u, chk_s, mode, size, mtime, status = (int(v) for v in rows)
    if nulls == BLOCKSIZE:
        # A block of 512 NULs is the end-of-archive marker; upstream checks
        # that before it ever touches a number field, and an all-NUL field
        # would not parse anyway.
        return {
            "nulls": nulls,
            "chksum": chk_u,
            "chksum_signed": chk_s,
            "chksum_stored": None,
            "chksum_ok": False,
            "mode": None,
            "size": None,
            "mtime": None,
            "is_eof": True,
        }
    if status & 7:
        raise InvalidHeaderError("invalid header")
    stored = nti(block512[OFF_CHKSUM:OFF_CHKSUM + LEN_CHKSUM])
    return {
        "nulls": nulls,
        "chksum": chk_u,
        "chksum_signed": chk_s,
        "chksum_stored": stored,
        "chksum_ok": stored in (chk_u, chk_s),
        "mode": mode,
        "size": size,
        "mtime": mtime,
        "is_eof": False,
    }


def scan_archive(data: bytes) -> list[dict]:
    """Decode every 512-byte block of `data` as a header, in one kernel call.

    This is the arithmetic behind walking an archive's header region without
    paying a Python call per block. It does not follow the payload, so it only
    describes the leading header run; use the real `backports.tarfile` to walk
    members.
    """
    raw = np.frombuffer(data, dtype=np.uint8)
    nblocks = raw.size // BLOCKSIZE
    if nblocks == 0:
        return []
    return scan_headers(raw[: nblocks * BLOCKSIZE], nblocks, BLOCKSIZE)


__all__ = [
    "BLOCKSIZE",
    "GNU_FORMAT",
    "OFF_CHKSUM",
    "OFF_GID",
    "OFF_LINKNAME",
    "OFF_MAGIC",
    "OFF_MODE",
    "OFF_MTIME",
    "OFF_NAME",
    "OFF_SIZE",
    "OFF_TYPE",
    "OFF_UID",
    "block",
    "calc_chksums",
    "field_pad",
    "header_fields",
    "itn",
    "nti",
    "nti_fields",
    "pad_payload",
    "pax_records",
    "scan_archive",
    "scan_headers",
]
