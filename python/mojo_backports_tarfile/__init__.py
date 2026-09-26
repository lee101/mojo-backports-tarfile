"""mojo-backports-tarfile: Mojo kernels for the tar header codec.

Installable alongside the real `backports.tarfile` package, which the tests
compare against for byte-exact parity.
"""

from .core import (
    BLOCKSIZE,
    GNU_FORMAT,
    OFF_CHKSUM,
    OFF_GID,
    OFF_LINKNAME,
    OFF_MAGIC,
    OFF_MODE,
    OFF_MTIME,
    OFF_NAME,
    OFF_SIZE,
    OFF_TYPE,
    OFF_UID,
    block,
    calc_chksums,
    field_pad,
    header_fields,
    itn,
    nti,
    nti_fields,
    pad_payload,
    pax_records,
    scan_archive,
    scan_headers,
)

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
__version__ = "0.1.0"
