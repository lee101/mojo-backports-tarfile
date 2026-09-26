"""Byte-level header arithmetic for the `backports.tarfile` codec subset.

The interesting numeric surface in a tar implementation is the header codec:
the two 512-byte checksums, the octal / base-256 number-field encode and
decode, and the record framing of a POSIX.1-2008 pax extended header. Those
are byte loops, and they are what lives here.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

A `def ... abi("C")` with no return type is `-> None`, so status codes are
written into caller-owned Int64 slots rather than returned.
"""

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]

comptime BLOCKSIZE = 512
comptime CHKSUM_START = 148
comptime CHKSUM_LEN = 8

comptime GNU_FORMAT = 1


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def is_space(c: UInt8) -> Bool:
    return c == 0x20 or (c >= 0x09 and c <= 0x0D)


def is_digit(c: UInt8) -> Bool:
    return c >= 0x30 and c <= 0x39


def is_octal_digit(c: UInt8) -> Bool:
    return c >= 0x30 and c <= 0x37


# ---------------------------------------------------------------- nti


def nti_field(buf: BPtr, start: Int, length: Int) -> Tuple[Int64, Int64]:
    """`backports.tarfile.nti` for one number field.

    Returns (value, status); status is 0 on success, 1 for a field that is not a
    valid octal or base-256 number, 2 for a base-256 value that does not fit an
    Int64. Upstream returns a Python int of unbounded width, so the shim
    documents the Int64 range as the one narrowing.
    """
    var s0 = Int64(buf[unsafe_offset=start])
    if s0 == 0o200 or s0 == 0o377:
        var k = length - 1
        # The range is checked by a separate scan over the payload bytes rather
        # than inside the accumulation loop. An in-loop `if n > limit: return`
        # was measured to be folded away at -O3, because the compiler is
        # entitled to assume the signed multiply it sits next to does not
        # overflow, and that silently let the wrap through.
        var fits = True
        if k > 7:
            # Bit 63 lives in the top payload byte, and anything past the
            # eighth payload byte is past bit 63.
            if (buf[unsafe_offset=start + 1] & 0x80) != 0:
                fits = False
            for i in range(8, k):
                if buf[unsafe_offset=start + 1 + i] != 0:
                    fits = False
        if s0 == 0o377 and k > 7:
            # -(256**k - p) is below -2**63 for every payload this wide.
            fits = False
        if not fits:
            return (0, 2)
        var n: Int64 = 0
        for i in range(k):
            n = n * 256 + Int64(buf[unsafe_offset=start + 1 + i])
        if s0 == 0o377:
            var modulus: Int64 = 1
            for _ in range(k):
                modulus = modulus * 256
            n = n - modulus
        return (n, 0)

    # Upstream runs the field through nts(), which truncates at the first NUL,
    # and only then strips whitespace.
    var end = length
    for k in range(length):
        if buf[unsafe_offset=start + k] == 0:
            end = k
            break
    var i = 0
    var j = end
    while i < j and is_space(buf[unsafe_offset=start + i]):
        i += 1
    while j > i and is_space(buf[unsafe_offset=start + j - 1]):
        j -= 1
    if i == j:
        return (0, 0)
    var negative = False
    if buf[unsafe_offset=start + i] == 0x2D:
        negative = True
        i += 1
    elif buf[unsafe_offset=start + i] == 0x2B:
        i += 1
    if i == j:
        return (0, 1)
    var value: Int64 = 0
    for k in range(i, j):
        var c = buf[unsafe_offset=start + k]
        if c > 0x7F or not is_octal_digit(c):
            return (0, 1)
        value = value * 8 + Int64(c - 0x30)
    if negative:
        value = -value
    return (value, 0)


# ------------------------------------------------------------ chksum


@export("tf_nti")
def tf_nti(buf_addr: Int, start: Int, length: Int, out_addr: Int) abi("C"):
    """Decode one tar number field in place.

    out layout (Int64, 2 slots): the decoded value, then the status where 0 is
    success, 1 is an unparsable field and 2 is a base-256 value that does not
    fit an Int64.
    """
    var buf = bp(buf_addr)
    var out = ip(out_addr)
    if length <= 0:
        out[unsafe_offset=0] = 0
        out[unsafe_offset=1] = 0
        return
    var result = nti_field(buf, start, length)
    out[unsafe_offset=0] = result[0]
    out[unsafe_offset=1] = result[1]


@export("tf_nti_run")
def tf_nti_run(
    buf_addr: Int, count: Int64, stride: Int64, out_addr: Int
) abi("C"):
    """Decode `count` fixed-width number fields laid out at `stride` bytes.

    One call for a whole run, so the caller pays the FFI boundary once rather
    than once per 12-byte field. out layout (Int64, 2 slots per field): the
    decoded value, then the status where 0 is success, 1 is an unparsable field
    and 2 is a base-256 value that does not fit an Int64.
    """
    var buf = bp(buf_addr)
    var out = ip(out_addr)
    for i in range(count):
        var result = nti_field(buf, Int(i) * Int(stride), Int(stride))
        var slot = i * 2
        out[unsafe_offset=slot] = result[0]
        out[unsafe_offset=slot + 1] = result[1]




@export("tf_calc_chksums")
def tf_calc_chksums(buf_addr: Int, out_addr: Int) abi("C"):
    """`backports.tarfile.calc_chksums` for one 512-byte header block.

    The eight checksum bytes at offset 148 are skipped entirely, which is
    equivalent to the upstream trick of treating the field as eight spaces
    because that is what a well-formed header holds there. Both the unsigned
    and the signed (int8) sums are produced, because Sun and NeXT tars use the
    signed one. out layout: Float64[0] unsigned, Float64[1] signed.
    """
    var buf = bp(buf_addr)
    var out = fp(out_addr)
    var unsigned_sum: Int64 = 0
    var signed_sum: Int64 = 0
    for i in range(BLOCKSIZE):
        if i >= CHKSUM_START and i < CHKSUM_START + CHKSUM_LEN:
            continue
        var c = Int64(buf[unsafe_offset=i])
        unsigned_sum += c
        signed_sum += c if c < 128 else c - 256
    out[unsafe_offset=0] = Float64(unsigned_sum + 256)
    out[unsafe_offset=1] = Float64(signed_sum + 256)


@export("tf_scan_headers")
def tf_scan_headers(
    buf_addr: Int, nheaders: Int64, stride: Int64, out_addr: Int
) abi("C"):
    """Walk `nheaders` consecutive header blocks and report, per header, the
    integers a sequential tar reader needs.

    This is the inner loop behind `TarFile.next()`: the null-byte count for end
    of archive detection, both checksums for `InvalidHeaderError`, and the
    decoded mode, size and mtime number fields.

    out layout (Int64, 7 slots per header):
        0 NUL count over the whole block (BLOCKSIZE means end of archive)
        1 unsigned checksum
        2 signed checksum
        3 mode   (nti at offset 100, 8 bytes)
        4 size   (nti at offset 124, 12 bytes)
        5 mtime  (nti at offset 136, 12 bytes)
        6 status bitmask: 1 mode, 2 size, 4 mtime, 8 chksum field
    """
    var buf = bp(buf_addr)
    var out = ip(out_addr)
    for h in range(nheaders):
        var base: Int = Int(h) * Int(stride)
        # One pass produces all three byte-wise quantities; separate loops
        # would re-read the same 512 bytes three times.
        var nulls: Int64 = 0
        var unsigned_sum: Int64 = 0
        var signed_sum: Int64 = 0
        for i in range(BLOCKSIZE):
            var c = Int64(buf[unsafe_offset=base + i])
            if c == 0:
                nulls += 1
            if i < CHKSUM_START or i >= CHKSUM_START + CHKSUM_LEN:
                unsigned_sum += c
                signed_sum += c if c < 128 else c - 256

        var mode = nti_field(buf, base + 100, 8)
        var size = nti_field(buf, base + 124, 12)
        var mtime = nti_field(buf, base + 136, 12)
        var chk = nti_field(buf, base + 148, 8)
        var status: Int64 = 0
        if mode[1] != 0:
            status |= 1
        if size[1] != 0:
            status |= 2
        if mtime[1] != 0:
            status |= 4
        if chk[1] != 0:
            status |= 8

        var slot = h * 7
        out[unsafe_offset=slot] = nulls
        out[unsafe_offset=slot + 1] = unsigned_sum + 256
        out[unsafe_offset=slot + 2] = signed_sum + 256
        out[unsafe_offset=slot + 3] = mode[0]
        out[unsafe_offset=slot + 4] = size[0]
        out[unsafe_offset=slot + 5] = mtime[0]
        out[unsafe_offset=slot + 6] = status


# ---------------------------------------------------------------- itn


@export("tf_itn")
def tf_itn(
    value: Int, digits: Int, format: Int, out_addr: Int, status_addr: Int
) abi("C"):
    """`backports.tarfile.itn` for one number field.

    Writes exactly `digits` bytes to out_addr and sets status 0, or 1 if the
    value does not fit the field. For a negative GNU base-256 value the bytes
    are the low `digits - 1` bytes of the two's-complement form over
    `8 * digits` bits, which is byte-for-byte what the upstream
    `256 ** digits + n` shift produces, including its truncation for
    magnitudes above 256**(digits-1).
    """
    var out = bp(out_addr)
    var status = ip(status_addr)
    var n = value
    if n >= 0 and n < 8 ** (digits - 1):
        for i in range(digits - 1):
            var shift = 3 * (digits - 2 - i)
            out[unsafe_offset=i] = UInt8(((n >> shift) & 0x7) + 0x30)
        out[unsafe_offset=digits - 1] = 0
        status[unsafe_offset=0] = 0
        return

    # 256 ** (digits - 1) is 2 ** 56 for the widest field the port accepts, so
    # the range check is only meaningful for digits <= 8; for 12-byte fields
    # every Int64 is in range by construction.
    var in_range = True
    if digits <= 8:
        var bound = 256 ** (digits - 1)
        in_range = n >= -bound and n < bound
    if format == GNU_FORMAT and in_range:
        if n >= 0:
            out[unsafe_offset=0] = 0x80
        else:
            out[unsafe_offset=0] = 0xFF
        for i in range(digits - 1):
            # Upstream inserts each extracted byte at position 1, which lays
            # the payload out most-significant first.
            var slot = digits - 2 - i
            var byte: UInt8 = 0xFF
            if n >= 0 or i < 8:
                byte = UInt8((n >> (8 * i)) & 0xFF)
            out[unsafe_offset=1 + slot] = byte
        status[unsafe_offset=0] = 0
        return

    status[unsafe_offset=0] = 1


# --------------------------------------------------------------- pax


@export("tf_pax_scan")
def tf_pax_scan(
    buf_addr: Int, n: Int, capacity: Int, out_addr: Int, count_addr: Int,
    status_addr: Int
) abi("C"):
    """Frame the records of a POSIX.1-2008 pax extended header payload.

    A record is `"%d %s=%s\\n"`, where the leading decimal is the length of the
    whole record, so the framing is a byte scan rather than a regex. This
    kernel reproduces the `(\\d+) ([^=]+)=` match that
    `TarInfo._proc_pax` runs at each successive position and reports, per
    record, the byte range of the keyword and the byte range of the value.

    out layout (Int64, 6 slots per record):
        0 record start
        1 record length
        2 keyword start
        3 keyword end (exclusive)
        4 value start
        5 value end (exclusive, already clamped to n)
    count_addr receives the record count. status is 0, or 1 when a record
    declares length 0, which upstream rejects.
    """
    var buf = bp(buf_addr)
    var out = ip(out_addr)
    var count = ip(count_addr)
    var status = ip(status_addr)
    var pos: Int = 0
    var found: Int = 0
    status[unsafe_offset=0] = 0
    while pos < n and found < capacity:
        var i = pos
        var k = i
        var length: Int = 0
        while k < n and is_digit(buf[unsafe_offset=k]):
            length = length * 10 + Int(buf[unsafe_offset=k] - 0x30)
            k += 1
        if k == i:
            break
        if length == 0:
            count[unsafe_offset=0] = Int64(found)
            status[unsafe_offset=0] = 1
            return
        if k >= n or buf[unsafe_offset=k] != 0x20:
            break
        var kw_start = k + 1
        var e = kw_start
        while e < n and buf[unsafe_offset=e] != 0x3D:
            e += 1
        if e == kw_start or e >= n:
            break
        var val_start = e + 1
        var val_end = i + length - 1
        if val_end > n:
            val_end = n
        if val_end < val_start:
            val_end = val_start

        var slot = found * 6
        out[unsafe_offset=slot] = Int64(i)
        out[unsafe_offset=slot + 1] = Int64(length)
        out[unsafe_offset=slot + 2] = Int64(kw_start)
        out[unsafe_offset=slot + 3] = Int64(e)
        out[unsafe_offset=slot + 4] = Int64(val_start)
        out[unsafe_offset=slot + 5] = Int64(val_end)
        found += 1
        pos = i + length

    count[unsafe_offset=0] = Int64(found)


# ------------------------------------------------------- field padding


@export("tf_field_pad")
def tf_field_pad(
    src_addr: Int, srclen: Int, out_addr: Int, length: Int
) abi("C"):
    """`backports.tarfile.stn`: copy at most `length` bytes and NUL-pad the
    rest, which is how every fixed-width header field is laid out."""
    var src = bp(src_addr)
    var out = bp(out_addr)
    var n = srclen
    if n > length:
        n = length
    for i in range(n):
        out[unsafe_offset=i] = src[unsafe_offset=i]
    for i in range(n, length):
        out[unsafe_offset=i] = 0
