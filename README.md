# mojo-backports-tarfile

`mojo-backports-tarfile` is the byte-level subset of
[backports.tarfile](https://pypi.org/project/backports.tarfile/) with the tar
header codec implemented in Mojo and callable from Python.

The Python package is named `mojo_backports_tarfile`, so it installs alongside
the real `backports.tarfile` and the tests compare the two byte for byte.

```python
import io, backports.tarfile as tarfile
import mojo_backports_tarfile as mbt

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tf:
    info = tarfile.TarInfo("hello.txt")
    info.size = 5
    tf.addfile(info, io.BytesIO(b"hello"))
data = buf.getvalue()

mbt.calc_chksums(data[:512])   # (3972, 3972)
mbt.header_fields(data[:512])  # {'chksum': 3972, 'size': 5, 'chksum_ok': True, ...}
mbt.nti(b"00000000005\x00")    # 5
mbt.itn(0o644, 8)              # b'0000644\x00'
```

## Why this subset

A tar implementation is mostly IO: seek, read, decompress, write, extract, and
the extraction filters. None of that is arithmetic. The part that *is* byte
arithmetic is the header codec, and it is not trivial: a 512-byte block carries
two different checksums (unsigned and signed, because Sun and NeXT tars differ),
four number fields in a base-8 text encoding, an optional GNU base-256 binary
encoding including its sign convention, and a POSIX.1-2008 pax extended header
whose records are framed by a self-describing decimal length at the front of
each record. Every one of those is a byte loop over a fixed-size structure, and
every one of them is on the critical path of reading an archive.

That codec is what is ported, and only that.

## Covered subset

| upstream | ported API | what the kernel does |
| --- | --- | --- |
| `calc_chksums` | `calc_chksums` | one 512-byte pass, unsigned and int8 sums, checksum field skipped |
| `nti` | `nti`, `nti_fields` | octal or GNU base-256 decode, `nts` NUL truncation, whitespace strip, sign |
| `itn` | `itn` | octal or base-256 encode, including the negative two's-complement form |
| `TarInfo.frombuf` numeric fields | `scan_headers`, `header_fields` | per block: NUL count, both checksums, mode/size/mtime, status bitmask |
| `TarInfo._proc_pax` record framing | `pax_records` | the `(\d+) ([^=]+)=` scan as a byte walk, reporting keyword and value ranges |
| `stn` | `field_pad` | copy to a fixed width and NUL-pad the tail |
| `TarInfo._block`, `TarInfo._create_payload` | `block`, `pad_payload` | block rounding and payload padding |

### Not implemented

- `TarFile` itself: reading, writing, appending, `addfile`, `extract`,
  `extractall`, `getmembers`, and the offset bookkeeping around them. That is
  IO and state, and the real `backports.tarfile` is the right place for it.
- Compression: gz, bz2, xz and lzma streams.
- Sparse files: `GNUTYPE_SPARSE`, `_proc_sparse`, `_proc_gnusparse_00/01/10`.
- GNU long name/link records (`GNUTYPE_LONGNAME`, `GNUTYPE_LONGLINK`).
- The extraction filters (`data_filter`, `tar_filter`, `fully_trusted_filter`).
- `TarInfo.get_info` and the whole `TarInfo` attribute model beyond the numeric
  fields; `pax_records` returns raw keyword/value pairs rather than a decoded
  `TarInfo`.

### Two documented range limits

Both come from exposing number fields as 64-bit integers, and both are
narrowings rather than behavioural changes:

- a base-256 field whose value does not fit `int64` raises `OverflowError`
  instead of returning a Python bignum. Eight-byte fields can never hit this,
  because `256**7` is `2**56`; only the 12-byte `size` and `mtime` fields can.
- a *negative* base-256 field in a 12-byte field always raises: `-(256**11 - p)`
  is below `-2**63` for every 12-byte payload, whatever `p` is.

## Install

```bash
pixi run build
pixi run test
```

`bash build/build.sh` compiles `src/kernels.mojo` with
`mojo build --emit shared-lib` into `dist/libmojo-backports-tarfile.so`. Set
`PYTHONPATH=python` when using the package outside a Pixi task.

## Performance

Best-of-N wall clock in one process. Every case asserts exact agreement with
the reference before timing. This box is shared, so absolute times move a lot
between runs; the ratios below are from one run and the surrounding runs
agreed to within about 30% except where noted.

| case | reference | mojo-backports-tarfile | result |
| --- | ---: | ---: | ---: |
| `scan_headers` n=16384 (16384 blocks, vs vectorised NumPy) | 40.40 ms | 26.62 ms | 1.52x |
| `calc_chksums` x4096 (vs `tarfile.calc_chksums`) | 116.37 ms | 67.97 ms | 1.71x |
| `nti` one field at a time x16384 (vs `tarfile.nti`) | 36.26 ms | 648.33 ms | **0.06x, a loss** |
| `nti` batched x16384 (vs vectorised NumPy) | 0.92 ms | 0.81 ms | 1.14x, parity |
| pax record framing n=4096 (vs the upstream regex loop) | 8.26 ms | 0.10 ms | 85.55x |
| `pax_records` end to end n=4096 (vs the same regex loop) | 9.83 ms | 13.45 ms | **0.73x, a loss** |

Reading the table honestly:

- The one-field-at-a-time `nti` row is a real loss and it is not fixable by
  making the kernel faster. Decoding twelve bytes costs about a microsecond
  upstream, and a ctypes call plus two small NumPy allocations costs about
  thirty. The kernel is irrelevant at that size; the FFI boundary is the whole
  cost. This is exactly why `nti_fields` exists: batch the work, pay the
  boundary once, and the same decode becomes parity with NumPy.
- `pax_records` end to end also loses, and for a related reason. The framing
  scan itself is two orders of magnitude faster than the regex loop, but the
  shim then walks the returned ranges in Python to build `(str, bytes)` pairs,
  and that Python costs more than the regex ever did. The `pax framing` row is
  the kernel measured on its own, and it is the honest measure of the ported
  work.
- `scan_headers` wins modestly, not dramatically. The vectorised NumPy baseline
  is genuinely good here: a fixed-width byte sum is close to what SIMD wants.
  The Mojo kernel wins because it fuses the null count and both checksums into
  a single pass over each block and adds the field decode, where NumPy needs
  three separate reductions plus a `where` for the octal digits.
- `pax framing` is the standout, and it is not a strawman: the reference is
  the exact regex loop from `TarInfo._proc_pax`, which is what the upstream
  implementation runs.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit. Buffers cross the
C ABI as 64-bit addresses and are reconstructed in Mojo as
`Pointer[UInt8, AnyOrigin[mut=True]]` and `Pointer[Int64, AnyOrigin[mut=True]]`.
A `def ... abi("C")` with no return type is `-> None`, so every status code is
written into a caller-owned `Int64` slot rather than returned.

Every loop is a plain serial loop. These kernels are memory-bound byte walks
over short fixed-size blocks; 1.2.0 cannot pass pointers into a `parallelize`
body, and threading a 512-byte loop would be slower anyway.

There is no FMA in this port. It is all integer and byte arithmetic, so every
parity assertion is `rtol=0, atol=0` — byte-identical or broken.

### One thing worth knowing

The first version of the base-256 decoder checked the range *inside* the
accumulation loop:

```mojo
for i in range(1, length):
    if n > limit:
        return (0, 2)
    n = n * 256 + Int64(buf[unsafe_offset=start + i])
```

That check was measured to be folded away, and an eleven-byte all-`0xFF` field
silently wrapped to `-1` instead of reporting the overflow. The shipped version
does the range check as a separate scan over the payload bytes *before* the
accumulation, where the multiply is not next to a check that can be optimised
away. `tests/test_header_codec.py::test_nti_rejects_a_12_digit_base256_value_beyond_int64`
is the test that catches it.

## Tests

61 tests in `tests/test_header_codec.py` run against the real
`backports.tarfile` 1.2.0 from the test venv. Archives are built by upstream
`tarfile` itself in `USTAR`, `GNU` and `PAX` formats, so the headers being
checked are real ones.

Beyond the direct parity assertions, the tests pin the behaviour that
distinguishes the correct codec from a plausible near-miss: that the signed
checksum diverges from the unsigned one when a name carries a high-bit byte,
that the checksum field is excluded from its own checksum, that the two
base-256 branches are byte-identical to upstream including the truncation in
its negative encoding, that the octal decoder stops at the NUL the way `nts`
does, that the pax scan clamps an over-long declared length the way the
upstream slice does, and that a zero-length pax record is rejected.

## License

MIT
