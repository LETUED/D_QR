"""
D_QR 다차원 정보 구조체 (MultiDimData)

여러 독립 데이터 차원을 하나의 직렬화 가능한 컨테이너로.
각 차원은 이름(str) + 데이터(bytes)의 쌍.

사용 예:
    mdd = MultiDimData(
        text   = b"Hello D_QR",
        meta   = b"author:donghwan",
        proof  = hashlib.sha256(b"Hello D_QR").digest(),
    )
    raw = mdd.to_bytes()
    mdd2 = MultiDimData.from_bytes(raw)
    assert mdd2['text'] == b"Hello D_QR"

직렬화 포맷:
    [1B: dim_count]
    per dimension:
        [1B: name_len] [name_bytes] [4B: data_len big-endian] [data_bytes]
"""


class MultiDimData:
    """다차원 데이터 컨테이너."""

    def __init__(self, **dims):
        self._dims = {k: bytes(v) for k, v in dims.items()}

    # ── 직렬화 / 역직렬화 ────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        parts = [bytes([len(self._dims)])]
        for name, data in self._dims.items():
            nb = name.encode('utf-8')
            parts.append(bytes([len(nb)]))
            parts.append(nb)
            parts.append(len(data).to_bytes(4, 'big'))
            parts.append(data)
        return b''.join(parts)

    @classmethod
    def from_bytes(cls, raw: bytes) -> 'MultiDimData':
        idx = 0
        n = raw[idx]; idx += 1
        dims = {}
        for _ in range(n):
            nl = raw[idx]; idx += 1
            name = raw[idx:idx+nl].decode('utf-8'); idx += nl
            dl = int.from_bytes(raw[idx:idx+4], 'big'); idx += 4
            dims[name] = raw[idx:idx+dl]; idx += dl
        return cls(**dims)

    # ── 편의 메서드 ──────────────────────────────────────────────────

    def __getitem__(self, key): return self._dims[key]
    def __len__(self): return len(self._dims)
    def keys(self): return self._dims.keys()
    def items(self): return self._dims.items()
    def dim_names(self): return list(self._dims.keys())

    def summary(self) -> str:
        lines = [f"MultiDimData ({len(self._dims)} 차원):"]
        for name, data in self._dims.items():
            lines.append(f"  [{name}]: {len(data)} bytes")
        return '\n'.join(lines)

    def split_two(self) -> tuple:
        """
        2개 스트림으로 분리 (exp8 DLC 인코딩용).
        차원이 짝수면 절반씩, 홀수면 첫 차원을 stream_a에.
        반환: (stream_a_bytes, stream_b_bytes)
        """
        names = list(self._dims.keys())
        mid = max(1, len(names) // 2)
        a_raw = MultiDimData(**{k: self._dims[k] for k in names[:mid]}).to_bytes()
        b_raw = MultiDimData(**{k: self._dims[k] for k in names[mid:]}).to_bytes()
        return a_raw, b_raw

    @classmethod
    def from_two(cls, a_bytes: bytes, b_bytes: bytes) -> 'MultiDimData':
        """split_two()의 역함수."""
        a = cls.from_bytes(a_bytes)
        b = cls.from_bytes(b_bytes)
        merged = {**a._dims, **b._dims}
        return cls(**merged)
