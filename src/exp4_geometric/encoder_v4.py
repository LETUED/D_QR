"""
D_QR Phase 4 인코더 — 기하 서브삼각형 인코딩 (exp4)

핵심 변경:
  - 색상 채널(c1/c2/c3) 완전 제거
  - 각 육각형 셀 = 6개 정삼각형, 각 삼각형: filled(1) / empty(0)
  - 6비트/셀, 단색 이진 렌더링
  - 조명/색상 불변: 밝기 임계값(128)만으로 디코딩
  - 홀로그램·흑백인쇄·레이저각인 모두 동일 원리 적용 가능

RS ECC: 단일 채널 (색상 분리 불필요)
"""
import math
from reedsolo import RSCodec

from src.core.hex_grid import build_hex_grid, build_reserved, finder_centers
from src.core.bitutils import _max_data_for_budget, _bytes_to_bits, _bits_to_bytes

EXP_TAG = "exp4"
BITS_PER_CELL = 6       # 6 삼각형 × binary
GRID_RADII = [8, 12, 18, 25, 35, 50]


class DQREncoderV4:
    """
    D_QR Phase 4 인코더 — 순수 기하 인코딩.

    색상 채널 없음. 단일 RS ECC.
    기하학적 신뢰성이 색상보다 높으므로 더 낮은 ECC rate 가능.

    사용 예:
        enc = DQREncoderV4(ecc_rate=0.20)
        result = enc.encode(b"Hello, Geometric D_QR!")
    """

    def __init__(self, ecc_rate: float = 0.20):
        if not (0.0 < ecc_rate < 1.0):
            raise ValueError("ecc_rate must be between 0 and 1")
        self.ecc_rate = ecc_rate
        self.nsym = max(2, min(127, int(255 * ecc_rate)))
        self.dpb = 255 - self.nsym

    def _data_cells(self, radius: int) -> list:
        reserved = build_reserved(radius)
        return [c for c in build_hex_grid(radius) if c not in reserved]

    def max_data_bytes(self, radius: int) -> int:
        M = len(self._data_cells(radius))
        total_bytes = (M * BITS_PER_CELL) // 8
        return _max_data_for_budget(total_bytes, self.nsym, self.dpb)

    def _choose_radius(self, n: int) -> int:
        for r in GRID_RADII:
            if self.max_data_bytes(r) >= n:
                return r
        for r in range(GRID_RADII[-1] + 1, 200):
            if self.max_data_bytes(r) >= n:
                return r
        raise ValueError(f"데이터가 너무 큽니다: {n} bytes")

    def encode(self, data: bytes) -> dict:
        """
        Returns dict:
            tensor       : dict[(q,r) → int(0-63)] — 6비트 셀 값
            reserved     : frozenset
            finder_centers : list
            radius       : int
            nsym         : int
            codeword_len : int
            data_len     : int
            exp_tag      : "exp4"
            capacity     : dict
        """
        radius = self._choose_radius(len(data))
        reserved = build_reserved(radius)
        data_cells = self._data_cells(radius)
        M = len(data_cells)

        # RS 인코딩 (단일 채널)
        codeword = bytes(RSCodec(self.nsym).encode(data))
        cw_len = len(codeword)

        # 비트 스트림 → 6비트 셀 값
        bits = _bytes_to_bits(codeword)
        n_cells_needed = math.ceil(len(bits) / BITS_PER_CELL)
        bits += [0] * (n_cells_needed * BITS_PER_CELL - len(bits))

        tensor = {}
        for k, (q, r) in enumerate(data_cells):
            if k < n_cells_needed:
                start = k * BITS_PER_CELL
                val = 0
                for b in range(BITS_PER_CELL):
                    val |= bits[start + b] << (BITS_PER_CELL - 1 - b)
                tensor[(q, r)] = val
            else:
                tensor[(q, r)] = 0

        return {
            'tensor': tensor,
            'reserved': reserved,
            'finder_centers': finder_centers(radius),
            'radius': radius,
            'nsym': self.nsym,
            'codeword_len': cw_len,
            'data_len': len(data),
            'exp_tag': EXP_TAG,
            'capacity': {
                'radius': radius,
                'total_cells': len(build_hex_grid(radius)),
                'data_cells': M,
                'bits_per_cell': BITS_PER_CELL,
                'max_data_bytes': self.max_data_bytes(radius),
                'ecc_rate': self.ecc_rate,
                'nsym': self.nsym,
                'encoding': 'geometric_triangle',
                'color_dependent': False,
            },
        }
