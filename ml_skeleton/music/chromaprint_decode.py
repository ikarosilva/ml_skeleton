"""Pure-Python chromaprint fingerprint decoder (no C library).

Implements the chromaprint decompression format so we can decode base64 fingerprints
without calling the C library (which crashes with malloc in some environments).

Format (from acoustid/chromaprint C++ source):
  - Base64 decode -> raw bytes
  - Header: 4 bytes = algorithm (1), size (3 bytes BE)
  - Packed int3 array (values 0-7); 7 = exceptional, use int5 table
  - UnpackBits produces list of subfingerprint integers
  - SimHash over 8 chunks -> 8 uint32s = 256 bits
"""

from __future__ import annotations

import base64
from typing import List, Optional, Tuple

import numpy as np


def _unpack_int3_array(data: bytes) -> List[int]:
    """Unpack bytes to list of 3-bit values (0-7). Matches C++ UnpackInt3Array."""
    out: List[int] = []
    i = 0
    size = len(data)
    while size >= 3:
        s0, s1, s2 = data[i], data[i + 1], data[i + 2]
        i += 3
        size -= 3
        out.append(s0 & 0x07)
        out.append((s0 >> 3) & 0x07)
        out.append((s0 >> 6) | ((s1 & 0x01) << 2))
        out.append((s1 >> 1) & 0x07)
        out.append((s1 >> 4) & 0x07)
        out.append((s1 >> 7) | ((s2 & 0x03) << 1))
        out.append((s2 >> 2) & 0x07)
        out.append((s2 >> 5) & 0x07)
    if size == 2:
        s0, s1 = data[i], data[i + 1]
        out.append(s0 & 0x07)
        out.append((s0 >> 3) & 0x07)
        out.append((s0 >> 6) | ((s1 & 0x01) << 2))
        out.append((s1 >> 1) & 0x07)
        out.append((s1 >> 4) & 0x07)
    elif size == 1:
        s0 = data[i]
        out.append(s0 & 0x07)
        out.append((s0 >> 3) & 0x07)
    return out


def _unpack_int5_array(data: bytes, count: int) -> List[int]:
    """Unpack bytes to list of 5-bit values (0-31)."""
    n = min(count, len(data) * 8 // 5)
    out: List[int] = []
    buf = 0
    bits = 0
    idx = 0
    for byte in data:
        buf |= byte << bits
        bits += 8
        while bits >= 5 and len(out) < n:
            out.append(buf & 0x1F)
            buf >>= 5
            bits -= 5
        if len(out) >= n:
            break
    return out[:n]


def _simhash(data: List[int]) -> int:
    """Reduce list of uint32 to one uint32 (majority bit per position)."""
    if not data:
        return 0
    v = [0] * 32
    for h in data:
        for j in range(32):
            v[j] += (h >> j) & 1
    threshold = len(data) / 2
    out = 0
    for i in range(32):
        if v[i] > threshold:
            out |= 1 << i
    return out & 0xFFFFFFFF


def decompress_fingerprint(raw: bytes) -> Optional[Tuple[List[int], int]]:
    """Decompress chromaprint bytes to (list of subfingerprint uint32s, algorithm)."""
    if len(raw) < 4:
        return None
    algorithm = raw[0]
    size = (raw[1] << 16) | (raw[2] << 8) | raw[3]
    offset = 4
    # Unpack int3 array (max possible size from remaining bytes)
    max_bits = (len(raw) - offset) * 8 // 3
    bits = _unpack_int3_array(raw[offset:])
    # Truncate at (size zeros found)
    found = 0
    num_exceptional = 0
    for i, b in enumerate(bits):
        if b == 0:
            found += 1
            if found == size:
                bits = bits[: i + 1]
                break
        elif b == 7:
            num_exceptional += 1
    if found != size:
        return None
    packed3_size = (len(bits) * 3 + 7) // 8
    offset += packed3_size
    if num_exceptional > 0:
        packed5_size = (num_exceptional * 5 + 7) // 8
        if offset + packed5_size > len(raw):
            return None
        exceptional = _unpack_int5_array(raw[offset:], num_exceptional)
        if len(exceptional) < num_exceptional:
            return None
        j = 0
        for i in range(len(bits)):
            if bits[i] == 7:
                bits[i] = 7 + exceptional[j]
                j += 1
    # UnpackBits: bits -> output list of size integers
    output: List[int] = []
    value = 0
    last_bit = 0
    for b in bits:
        if b == 0:
            output.append(value & 0xFFFFFFFF)
            last_bit = 0
        else:
            last_bit += b
            value ^= 1 << (last_bit - 1)
    if len(output) != size:
        return None
    return (output, algorithm)


def decode_fingerprint_python(fingerprint_b64: str | bytes) -> Optional[Tuple[np.ndarray, int]]:
    """Decode base64 chromaprint to (8 uint32s, algorithm) using pure Python.

    Matches chromaprint.decode_fingerprint() return format for our use:
    (array of 8 uint32 = 256 bits, algorithm). Uses SimHash over 8 chunks.
    """
    try:
        if isinstance(fingerprint_b64, bytes):
            fingerprint_b64 = fingerprint_b64.decode("utf-8")
        b64 = fingerprint_b64.replace("-", "+").replace("_", "/")
        pad = 4 - (len(b64) % 4)
        if pad != 4:
            b64 += "=" * pad
        raw = base64.standard_b64decode(b64)
    except Exception:
        return None
    result = decompress_fingerprint(raw)
    if not result:
        return None
    subfingerprints, algorithm = result
    if len(subfingerprints) < 8:
        return None
    # Split into 8 chunks, SimHash each -> 8 uint32s
    chunk_size = (len(subfingerprints) + 7) // 8
    uint32s = []
    for i in range(8):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(subfingerprints))
        if start >= len(subfingerprints):
            uint32s.append(0)
        else:
            uint32s.append(_simhash(subfingerprints[start:end]))
    return (np.array(uint32s, dtype=np.uint32), algorithm)
