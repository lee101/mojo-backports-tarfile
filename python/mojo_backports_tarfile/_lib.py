"""ctypes bridge to the compiled Mojo header-codec kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

from backports.tarfile import InvalidHeaderError

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-backports-tarfile.so"

_I64 = ctypes.c_int64
_F64 = ctypes.c_double

BLOCKSIZE = 512
GNU_FORMAT = 1


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.tf_calc_chksums.restype = None
    lib.tf_calc_chksums.argtypes = [_I64, _I64]
    lib.tf_scan_headers.restype = None
    lib.tf_scan_headers.argtypes = [_I64, _I64, _I64, _I64]
    lib.tf_itn.restype = None
    lib.tf_itn.argtypes = [_I64, _I64, _I64, _I64, _I64]
    lib.tf_pax_scan.restype = None
    lib.tf_pax_scan.argtypes = [_I64, _I64, _I64, _I64, _I64, _I64]
    lib.tf_field_pad.restype = None
    lib.tf_nti.restype = None
    lib.tf_nti.argtypes = [_I64, _I64, _I64, _I64]
    lib.tf_field_pad.argtypes = [_I64, _I64, _I64, _I64]
    lib.tf_nti_run.restype = None
    lib.tf_nti_run.argtypes = [_I64, _I64, _I64, _I64]
    return lib


lib = _load()


def _u8(buf) -> np.ndarray:
    if isinstance(buf, (bytes, bytearray, memoryview)):
        return np.frombuffer(buf, dtype=np.uint8)
    return np.ascontiguousarray(buf, dtype=np.uint8)


def calc_chksums(header) -> tuple[int, int]:
    """(unsigned, signed) checksum of one 512-byte header block."""
    block = _u8(header)
    if block.size != BLOCKSIZE:
        raise ValueError(f"header must be {BLOCKSIZE} bytes, got {block.size}")
    out = np.empty(2, dtype=np.float64)
    lib.tf_calc_chksums(block.ctypes.data, out.ctypes.data)
    return int(out[0]), int(out[1])


def scan_headers(buf, nheaders: int, stride: int = BLOCKSIZE) -> np.ndarray:
    """Scan `nheaders` consecutive blocks; returns an (nheaders, 7) Int64 array.

    Columns: nulls, unsigned checksum, signed checksum, mode, size, mtime,
    status bitmask (1 mode, 2 size, 4 mtime, 8 chksum field).
    """
    block = _u8(buf)
    if block.size < nheaders * stride:
        raise ValueError("buffer is shorter than nheaders * stride")
    out = np.empty((nheaders, 7), dtype=np.int64)
    lib.tf_scan_headers(
        block.ctypes.data, nheaders, stride, out.ctypes.data
    )
    return out


def nti(field) -> int:
    """Decode one tar number field, raising InvalidHeaderError like upstream."""
    raw = _u8(field)
    out = np.empty(2, dtype=np.int64)
    lib.tf_nti(raw.ctypes.data, 0, raw.size, out.ctypes.data)
    if out[1] == 2:
        raise OverflowError(
            "base-256 tar number field does not fit in int64; the Mojo kernel "
            "exposes number fields as 64-bit integers"
        )
    if out[1] == 1:
        raise InvalidHeaderError("invalid header")
    return int(out[0])


def nti_fields(data, count: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode `count` fixed-width number fields in one kernel call.

    Returns `(values, statuses)` as Int64 arrays; status 0 is success, 1 an
    unparsable field and 2 a base-256 value outside the Int64 range.
    """
    if width < 2 or width > 18:
        raise ValueError("width must be between 2 and 18")
    raw = _u8(data)
    if raw.size < count * width:
        raise ValueError("buffer is shorter than count * width")
    out = np.empty((count, 2), dtype=np.int64)
    lib.tf_nti_run(raw.ctypes.data, count, width, out.ctypes.data)
    return out[:, 0], out[:, 1]


def itn(value: int, digits: int = 8, format: int = 0) -> bytes:
    """Encode one tar number field, byte-for-byte as `backports.tarfile.itn`."""
    if not 2 <= digits <= 18:
        raise ValueError("digits must be between 2 and 18")
    out = np.zeros(digits, dtype=np.uint8)
    status = np.zeros(1, dtype=np.int64)
    lib.tf_itn(int(value), int(digits), int(format), out.ctypes.data,
               status.ctypes.data)
    if status[0]:
        raise ValueError("overflow in number field")
    return out.tobytes()



def pax_records(payload, capacity: int | None = None):
    """Frame a pax extended-header payload.

    Returns a list of `(keyword: str, value: bytes)`, which is what
    `TarInfo._proc_pax` accumulates before decoding.
    """
    raw = _u8(payload)
    n = raw.size
    if capacity is None:
        capacity = max(1, n // 6)
    out = np.empty((capacity, 6), dtype=np.int64)
    count = np.zeros(1, dtype=np.int64)
    status = np.zeros(1, dtype=np.int64)
    lib.tf_pax_scan(raw.ctypes.data, n, capacity, out.ctypes.data,
                    count.ctypes.data, status.ctypes.data)
    if status[0]:
        raise InvalidHeaderError("invalid header")
    found = int(count[0])
    records = []
    for i in range(found):
        kw_start, kw_end = int(out[i, 2]), int(out[i, 3])
        val_start, val_end = int(out[i, 4]), int(out[i, 5])
        records.append(
            (
                raw[kw_start:kw_end].tobytes().decode("utf-8", "surrogateescape"),
                raw[val_start:val_end].tobytes(),
            )
        )
    return records


def field_pad(value, length: int) -> bytes:
    """`backports.tarfile.stn`: copy at most `length` bytes, NUL-pad the rest."""
    src = _u8(value)
    out = np.zeros(length, dtype=np.uint8)
    lib.tf_field_pad(src.ctypes.data, src.size, out.ctypes.data, length)
    return out.tobytes()
