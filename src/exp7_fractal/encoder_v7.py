"""
D_QR Phase 7 인코더 — 프랙탈 중첩 헥사곤 (exp7)

구조:
  외부 격자 (R=outer_radius, hex_size=outer_hex_size)
    └─ 각 외부 데이터 셀 내부에 내부 미니 격자 (R=1, 7셀) 내장
         └─ 각 내부 셀: 6개 서브삼각형 이진 인코딩 (exp4 방식)

용량: 외부_데이터셀 × 7_내부셀 × 6비트 = 42비트/외부셀
      동일 격자 크기에서 exp6(12비트/셀)보다 3.5배 더 많은 비트
      (단, 외부 hex_size가 더 커야 함)

프랙탈 원리:
  동일한 기하 패턴이 두 스케일에서 반복
  해상도 증가 → 더 깊은 레벨 가능 (지수적 용량 성장)
"""
import math
from reedsolo import RSCodec

from src.core.hex_grid import build_hex_grid, build_reserved, finder_centers
from src.core.bitutils import _max_data_for_budget, _bytes_to_bits, _bits_to_bytes

EXP_TAG = "exp7"

# 내부 R=1 격자: 중심 + 6이웃 = 7셀
INNER_CELLS = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
BITS_PER_INNER = 6    # 서브삼각형 6개 × 1비트
BITS_PER_OUTER = len(INNER_CELLS) * BITS_PER_INNER  # = 42

# 기본값: 외부 50px, 내부 14px (내부 격자가 외부 헥사곤 안에 딱 맞음)
DEFAULT_OUTER_HEX_SIZE = 50
DEFAULT_INNER_HEX_SIZE = 14
GRID_RADII = [5, 8, 12, 18, 25]


class DQREncoderV7:
    """
    D_QR Phase 7 프랙탈 인코더.

    사용 예:
        enc = DQREncoderV7(outer_radius=8, ecc_rate=0.20)
        result = enc.encode(b"Fractal D_QR!")
        # result['tensor'][(q1,r1,q2,r2)] = int 0..63 (내부 셀 값)
    """

    def __init__(self,
                 outer_radius: int = 8,
                 outer_hex_size: int = DEFAULT_OUTER_HEX_SIZE,
                 inner_hex_size: int = DEFAULT_INNER_HEX_SIZE,
                 ecc_rate: float = 0.20):
        self.outer_radius = outer_radius
        self.outer_hex_size = outer_hex_size
        self.inner_hex_size = inner_hex_size
        self.ecc_rate = ecc_rate
        self.nsym = max(2, min(127, int(255 * ecc_rate)))
        self.dpb = 255 - self.nsym

    def _outer_data_cells(self):
        reserved = build_reserved(self.outer_radius)
        return [c for c in build_hex_grid(self.outer_radius) if c not in reserved]

    def max_data_bytes(self) -> int:
        M = len(self._outer_data_cells())
        total_bytes = (M * BITS_PER_OUTER) // 8
        return _max_data_for_budget(total_bytes, self.nsym, self.dpb)

    def encode(self, data: bytes) -> dict:
        """
        Returns dict:
            tensor   : dict[(q1,r1,q2,r2) → int(0-63)]
            outer_radius, outer_hex_size, inner_hex_size
            reserved, finder_centers (outer level)
            nsym, codeword_len, data_len, exp_tag, capacity
        """
        outer_cells = self._outer_data_cells()
        M = len(outer_cells)
        reserved = build_reserved(self.outer_radius)

        codeword = bytes(RSCodec(self.nsym).encode(data))
        cw_len = len(codeword)

        bits = _bytes_to_bits(codeword)
        total_bits = M * BITS_PER_OUTER
        bits += [0] * (total_bits - len(bits))

        tensor = {}
        bit_idx = 0
        for q1, r1 in outer_cells:
            for q2, r2 in INNER_CELLS:
                val = 0
                for b in range(BITS_PER_INNER):
                    if bit_idx < len(bits):
                        val |= bits[bit_idx] << (BITS_PER_INNER - 1 - b)
                    bit_idx += 1
                tensor[(q1, r1, q2, r2)] = val

        return {
            'tensor': tensor,
            'outer_radius': self.outer_radius,
            'outer_hex_size': self.outer_hex_size,
            'inner_hex_size': self.inner_hex_size,
            'reserved': reserved,
            'finder_centers': finder_centers(self.outer_radius),
            'outer_cells': outer_cells,
            'nsym': self.nsym,
            'codeword_len': cw_len,
            'data_len': len(data),
            'exp_tag': EXP_TAG,
            'capacity': {
                'outer_radius': self.outer_radius,
                'outer_cells': M,
                'inner_cells_per_outer': len(INNER_CELLS),
                'bits_per_outer': BITS_PER_OUTER,
                'total_inner_cells': M * len(INNER_CELLS),
                'max_data_bytes': self.max_data_bytes(),
                'ecc_rate': self.ecc_rate,
                'fractal_levels': 2,
                'encoding': 'fractal_nested_hex',
            },
        }
