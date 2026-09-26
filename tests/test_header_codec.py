"""Byte-exact parity tests against the real `backports.tarfile`.

Everything here is integer or byte arithmetic, so the tolerance is
`rtol=0, atol=0` throughout: a tar header either matches byte for byte or the
codec is broken. There is no FMA anywhere in this port.
"""

import io

import numpy as np
import pytest

import mojo_backports_tarfile as mbt

tarfile = pytest.importorskip("backports.tarfile")


def _make_archive(names, fmt=tarfile.USTAR_FORMAT):
    """A real archive built by upstream, returned as raw bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tf:
        for name, body in names:
            data = body.encode() if isinstance(body, str) else body
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 1_600_000_000 + len(name)
            info.mode = 0o644
            info.uid = 1000
            info.gid = 1000
            info.uname = "mojo"
            info.gname = "mojo"
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ------------------------------------------------------------- chksums


def test_chksums_match_upstream_on_a_real_ustar_header():
    header = _make_archive([("a.txt", "hello")])[:512]
    got = mbt.calc_chksums(header)
    want = tarfile.calc_chksums(header)
    assert got == want
    assert got[0] > 0


@pytest.mark.parametrize("fmt", [tarfile.USTAR_FORMAT, tarfile.GNU_FORMAT])
def test_chksums_match_upstream_across_formats(fmt):
    header = _make_archive([("dir/file.bin", b"x" * 17)], fmt=fmt)[:512]
    assert mbt.calc_chksums(header) == tuple(tarfile.calc_chksums(header))


def test_signed_checksum_differs_when_high_bit_bytes_are_present():
    """A name with a non-ASCII byte makes the signed sum differ from unsigned;
    a kernel that dropped the sign branch would pass every ASCII-only test."""
    header = bytearray(_make_archive([("a.txt", "hello")])[:512])
    header[10] = 0xC3
    assert mbt.calc_chksums(bytes(header))[0] != mbt.calc_chksums(bytes(header))[1]
    assert mbt.calc_chksums(bytes(header)) == tuple(
        tarfile.calc_chksums(bytes(header))
    )


def test_chksum_ignores_the_eight_checksum_bytes():
    """Two headers differing only inside the checksum field must agree."""
    a = bytearray(_make_archive([("a.txt", "hello")])[:512])
    b = bytearray(a)
    b[148:156] = b"1234567\x00"
    assert mbt.calc_chksums(bytes(a)) == mbt.calc_chksums(bytes(b))


def test_chksum_rejects_wrong_sized_block():
    with pytest.raises(ValueError, match="512 bytes"):
        mbt.calc_chksums(b"short")


# ------------------------------------------------------------------ nti


@pytest.mark.parametrize(
    "value,digits",
    [
        (0, 8), (1, 8), (0o644, 8), (0o7777, 8), (8**6, 8), (8**7 - 1, 8),
        (0, 12), (8**10, 12), (8**11 - 1, 12),
    ],
)
def test_nti_round_trips_itn_output(value, digits):
    field = tarfile.itn(value, digits)
    assert mbt.nti(field) == value


@pytest.mark.parametrize("value,digits", [(1 << 24, 8), (1 << 40, 12), (2**62, 12)])
def test_nti_round_trips_gnu_base256_output(value, digits):
    field = tarfile.itn(value, digits, tarfile.GNU_FORMAT)
    assert field[0] in (0o200, 0o377)
    assert mbt.nti(field) == value


def test_nti_matches_upstream_on_octal_fields():
    for raw in (b"0000644\x00", b"0000000\x00", b"  1234 \x00", b"\x00" * 8,
                b"77777777777\x00", b"000000001\x00"):
        assert mbt.nti(raw) == tarfile.nti(raw)


def test_nti_matches_upstream_on_base256_fields():
    for raw in (bytes([0o200, 0, 0, 0, 0, 0, 0, 1]),
                bytes([0o200, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD]),
                bytes([0o377, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE])):
        assert mbt.nti(raw) == tarfile.nti(raw)


def test_nti_rejects_an_unparsable_field_like_upstream():
    with pytest.raises(tarfile.InvalidHeaderError):
        mbt.nti(b"00009x9\x00")
    with pytest.raises(tarfile.InvalidHeaderError):
        tarfile.nti(b"00009x9\x00")


def test_nti_empty_field_is_zero():
    assert mbt.nti(b"        ") == 0
    assert tarfile.nti(b"        ") == 0


def test_nti_rejects_a_12_digit_base256_value_beyond_int64():
    """The one documented narrowing: 11 payload bytes can exceed 2**63."""
    raw = bytes([0o200]) + bytes([0xFF] * 11)
    assert tarfile.nti(raw) > 2**63
    with pytest.raises(OverflowError, match="int64"):
        mbt.nti(raw)


def test_nti_fields_matches_per_field_and_upstream():
    values = [0, 1, 0o644, 12345678, 8**11 - 1, 42, 0o7777, 999999999]
    data = b"".join(tarfile.itn(v, 12) for v in values)
    got, status = mbt.nti_fields(data, len(values), 12)
    assert not status.any()
    assert list(got) == values
    for i, v in enumerate(values):
        assert mbt.nti(data[i * 12:(i + 1) * 12]) == tarfile.nti(
            data[i * 12:(i + 1) * 12]
        )


def test_nti_fields_reports_per_field_status():
    data = mbt.itn(7, 12) + b"0000000x9\x00\x00\x00" + mbt.itn(9, 12)
    got, status = mbt.nti_fields(data, 3, 12)
    assert list(got) == [7, 0, 9]
    assert list(status) == [0, 1, 0]


def test_nti_fields_rejects_a_short_buffer():
    with pytest.raises(ValueError, match="shorter than"):
        mbt.nti_fields(b"\0" * 12, 4, 12)


def test_nti_fields_rejects_an_out_of_range_width():
    with pytest.raises(ValueError, match="width"):
        mbt.nti_fields(b"\0" * 12, 1, 24)


# ------------------------------------------------------------------ itn


@pytest.mark.parametrize(
    "value,digits,fmt",
    [
        (0, 8, tarfile.USTAR_FORMAT),
        (0o644, 8, tarfile.USTAR_FORMAT),
        (8**7 - 1, 8, tarfile.USTAR_FORMAT),
        (0, 12, tarfile.USTAR_FORMAT),
        (8**11 - 1, 12, tarfile.USTAR_FORMAT),
        (8**7, 8, tarfile.GNU_FORMAT),
        (1 << 40, 12, tarfile.GNU_FORMAT),
        (-1, 8, tarfile.GNU_FORMAT),
        (-2, 8, tarfile.GNU_FORMAT),
        (-1 << 40, 12, tarfile.GNU_FORMAT),
    ],
)
def test_itn_is_byte_identical_to_upstream(value, digits, fmt):
    got = mbt.itn(value, digits, fmt)
    want = tarfile.itn(value, digits, fmt)
    assert got == bytes(want)
    assert len(got) == digits


def test_itn_rejects_overflow_like_upstream():
    with pytest.raises(ValueError, match="overflow"):
        mbt.itn(8**7, 8, tarfile.USTAR_FORMAT)
    with pytest.raises(ValueError, match="overflow"):
        tarfile.itn(8**7, 8, tarfile.USTAR_FORMAT)


def test_itn_negative_base256_is_a_fixed_point_of_upstream():
    """Upstream's negative base-256 encoding truncates; the port reproduces
    the same bytes, quirk and all, rather than 'fixing' it."""
    for value in (-1, -2, -3, -258, -1 << 40):
        assert mbt.itn(value, 8, tarfile.GNU_FORMAT) == bytes(
            tarfile.itn(value, 8, tarfile.GNU_FORMAT)
        )


# ---------------------------------------------------------- scan_headers


def test_scan_headers_matches_upstream_on_a_real_archive():
    data = _make_archive(
        [("a.txt", "hello"), ("b/c.txt", "world" * 20), ("d.bin", bytes(range(256)))]
    )
    nblocks = len(data) // 512
    rows = mbt.scan_headers(data, nblocks, 512)
    assert rows.shape == (nblocks, 7)

    # Checksums and null counts are defined for every 512-byte block, header
    # or payload, so those columns must match upstream for all of them.
    for h in range(nblocks):
        blk = data[h * 512:(h + 1) * 512]
        want_u, want_s = tarfile.calc_chksums(blk)
        assert rows[h, 1] == want_u
        assert rows[h, 2] == want_s
        assert rows[h, 0] == blk.count(b"\0")

    # The number-field columns are only meaningful where a header really is.
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        offsets = [member.offset for member in tf]
    assert len(offsets) == 3
    for offset in offsets:
        h = offset // 512
        blk = data[h * 512:(h + 1) * 512]
        assert rows[h, 6] == 0
        assert rows[h, 4] == tarfile.nti(blk[124:136])
        assert rows[h, 5] == tarfile.nti(blk[136:148])
        assert rows[h, 3] == tarfile.nti(blk[100:108])


def test_scan_headers_is_not_confused_by_the_payload_between_headers():
    """Reading every 512-byte block, payload included, must still match what
    upstream reports for the blocks that really are headers."""
    data = _make_archive([("a.txt", "hello")])
    nheaders = len(data) // 512
    rows = mbt.scan_headers(data, nheaders, 512)
    for h in range(2):
        blk = data[h * 512:(h + 1) * 512]
        assert rows[h, 1] == tarfile.calc_chksums(blk)[0]


def test_scan_headers_flags_a_corrupt_number_field():
    data = bytearray(_make_archive([("a.txt", "hello")])[:512])
    data[124:136] = b"0000000x9\x00\x00\x00"
    rows = mbt.scan_headers(bytes(data), 1, 512)
    assert rows[0, 6] & 2


def test_scan_headers_handles_a_non_512_stride():
    data = _make_archive([("a.txt", "hello")])
    stride = 1024
    padded = data[:512] + b"\xa5" * 512 + data[512:1024] + b"\xa5" * 512
    rows = mbt.scan_headers(padded, 2, stride)
    assert rows[0, 1] == tarfile.calc_chksums(data[:512])[0]
    assert rows[0, 0] == data[:512].count(b"\0")


# -------------------------------------------------------- header_fields


def test_header_fields_matches_frombuf_on_a_real_header():
    data = _make_archive([("a.txt", "hello")])
    blk = data[:512]
    fields = mbt.header_fields(blk)
    info = tarfile.TarInfo.frombuf(blk, "utf-8", "surrogateescape")
    assert fields["mode"] == info.mode
    assert fields["size"] == info.size
    assert fields["mtime"] == info.mtime
    assert fields["chksum"] == info.chksum
    assert fields["chksum_ok"]
    assert not fields["is_eof"]


def test_header_fields_detects_a_bad_checksum():
    data = bytearray(_make_archive([("a.txt", "hello")])[:512])
    data[149] = ord("7")
    fields = mbt.header_fields(bytes(data))
    assert not fields["chksum_ok"]


def test_header_fields_marks_the_end_of_archive_block():
    assert mbt.header_fields(b"\0" * 512)["is_eof"]


def test_header_fields_rejects_a_truncated_block():
    with pytest.raises(ValueError, match="512 bytes"):
        mbt.header_fields(b"\0" * 100)


# ------------------------------------------------------------- pax scan


def _pax_payload(records):
    out = b""
    for kw, value in records:
        body = value.encode() if isinstance(value, str) else value
        base = len(kw) + len(body) + 3
        length = base + len(str(base))
        while len(str(length)) + base != length:
            length = len(str(length)) + base
        out += b"%d %s=%s\n" % (length, kw.encode(), body)
    return out


def _upstream_pax_records(buf):
    """The record framing from `TarInfo._proc_pax`, verbatim."""
    import re

    regex = re.compile(br"(\d+) ([^=]+)=")
    pos = 0
    found = []
    while match := regex.match(buf, pos):
        length, keyword = match.groups()
        length = int(length)
        if length == 0:
            raise tarfile.InvalidHeaderError("invalid header")
        value = buf[match.end(2) + 1:match.start(1) + length - 1]
        found.append((keyword.decode("utf-8", "surrogateescape"), value))
        pos += length
    return found


def test_pax_records_match_the_upstream_regex_framing():
    payload = _pax_payload(
        [
            ("path", "a/very/long/path/" * 12),
            ("size", "123456"),
            ("mtime", "1600000000.123456"),
            ("uname", "mojo"),
        ]
    )
    assert mbt.pax_records(payload) == _upstream_pax_records(payload)


def test_pax_records_round_trip_a_real_pax_extended_header():
    long_name = "deep/" * 40 + "file.txt"
    data = _make_archive([(long_name, "payload")], fmt=tarfile.PAX_FORMAT)
    # The first block is the pax extended header; its payload follows.
    pax = tarfile.TarInfo.frombuf(data[:512], "utf-8", "surrogateescape")
    assert pax.type == tarfile.XHDTYPE
    payload = data[512:512 + pax.size]
    records = dict(
        (kw, val.decode("utf-8", "surrogateescape"))
        for kw, val in mbt.pax_records(payload)
    )
    assert records["path"] == long_name
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        member = tf.next()
    assert member.name == long_name


def test_pax_records_rejects_a_zero_length_record_like_upstream():
    with pytest.raises(tarfile.InvalidHeaderError, match="invalid header"):
        mbt.pax_records(b"0 path=x\n")
    with pytest.raises(tarfile.InvalidHeaderError, match="invalid header"):
        _upstream_pax_records(b"0 path=x\n")


def test_pax_records_handles_binary_values():
    payload = b"22 comment=\x00\x01\x02\x03binary\n"
    got = mbt.pax_records(payload)
    assert got[0][0] == "comment"
    assert got[0][1] == b"\x00\x01\x02\x03binary"
    assert got == _upstream_pax_records(payload)


def test_pax_records_stops_at_the_first_non_record_like_run():
    payload = b"garbage that is not a record"
    assert mbt.pax_records(payload) == []


def test_pax_records_truncates_an_overlong_declared_length():
    """Upstream slices past the end of the buffer and stops; so does the port."""
    payload = b"40 path=short\n"
    got = mbt.pax_records(payload)
    want = _upstream_pax_records(payload)
    assert got == want


# ------------------------------------------------------------ framing


def test_block_rounds_up_to_the_next_512():
    for count, want in ((0, 0), (1, 512), (512, 512), (513, 1024), (1024, 1024)):
        assert mbt.block(count) == want
        assert mbt.block(count) == tarfile.TarInfo._block(
            tarfile.TarInfo("x"), count
        )


def test_pad_payload_matches_upstream():
    for payload in (b"", b"x", b"x" * 511, b"x" * 512, b"x" * 513):
        got = mbt.pad_payload(payload)
        want = tarfile.TarInfo._create_payload(payload)
        assert got == want
        assert len(got) % 512 == 0


@pytest.mark.parametrize(
    "value,length", [(b"", 8), (b"abc", 8), (b"abcdefghij", 8), (b"x" * 32, 32)]
)
def test_field_pad_matches_upstream(value, length):
    assert mbt.field_pad(value, length) == tarfile.stn(
        value.decode(), length, "ascii", "strict"
    )


def test_field_pad_nul_fills_the_tail():
    assert mbt.field_pad(b"ab", 6) == b"ab\0\0\0\0"


# ------------------------------------------------- assembled header check


def test_round_trip_an_itn_assembled_header_checksum():
    """Assemble the numeric fields with itn and confirm calc_chksums agrees
    with the value the block itself stores, which is what `frombuf` checks."""
    data = _make_archive([("payload.bin", bytes(range(256)) * 4)])
    blk = data[:512]
    fields = mbt.header_fields(blk)
    assert fields["chksum"] == fields["chksum_stored"]
    assert mbt.nti(blk[124:136]) == fields["size"] == 1024
    assert mbt.itn(fields["size"], 12) == blk[124:136]
