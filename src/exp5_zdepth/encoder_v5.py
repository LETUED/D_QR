"""
D_QR Phase 5 인코더 — Z축 부조 인코딩 (exp5)

핵심 변경:
  - 각 셀 = 8비트/셀 (6비트 삼각형 + 2비트 높이)
  - 8비트 = 1바이트 → RS GF(256)과 자연 정렬
  - 단일 RS 코드워드로 두 채널 동시 보호
  - z-height: 0,1,2,3 (4레벨 = 2비트), 물리적으로는 3D프린터/CNC로 제조

차원 구성:
  bits 7-2: 삼각형 패턴 (표면, exp4 계승)
  bits 1-0: 높이 레벨 (z-depth, 신규)
"""
import math
from reedsolo import RSCodec

from src.core.hex_grid import build_hex_grid, build_reserved, finder_centers
from src.core.bitutils import _max_data_for_budget, _bytes_to_bits, _bits_to_bytes

EXP_TAG = "exp5"
BITS_TRI = 6    # 삼각형 패턴 비트
BITS_Z = 2      # 높이 레벨 비트
BITS_PER_CELL = BITS_TRI + BITS_Z  # = 8
Z_LEVELS = 4    # 0,1,2,3
GRID_RADII = [8, 12, 18, 25, 35, 50]


class DQREncoderV5:
    """
    D_QR Phase 5 인코더 — 3D 부조 + 삼각형 패턴.

    각 셀 = 1바이트:
      bits[7:2] = 삼각형 패턴 (6비트, 표면 기하)
      bits[1:0] = 높이 레벨 (2비트, z-depth)

    사용 예:
        enc = DQREncoderV5(ecc_rate=0.20)
        result = enc.encode(b"Hello, 3D D_QR!")
        # result['heights'][(q,r)] = 0~3 (높이 레벨)
        # result['triangles'][(q,r)] = 0~63 (삼각형 패턴)
    """

    def __init__(self, ecc_rate: float = 0.20):
        if not (0.0 < ecc_rate < 1.0):
            raise ValueError("ecc_rate must be between 0 and 1")
        self.ecc_rate = ecc_rate
        self.nsym = max(2, min(127, int(255 * ecc_rate)))
        self.dpb = 255 - self.nsym

    def _data_cells(self, radius):
        reserved = build_reserved(radius)
        return [c for c in build_hex_grid(radius) if c not in reserved]

    def max_data_bytes(self, radius):
        # 8비트/셀 = 1바이트/셀 → 셀 수가 곧 총 바이트
        M = len(self._data_cells(radius))
        return _max_data_for_budget(M, self.nsym, self.dpb)

    def _choose_radius(self, n):
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
            triangles    : dict[(q,r) → int(0-63)]  — 삼각형 패턴
            heights      : dict[(q,r) → int(0-3)]   — z-depth 레벨
            reserved     : frozenset
            finder_centers : list
            radius       : int
            nsym         : int
            codeword_len : int
            data_len     : int
            exp_tag      : "exp5"
            capacity     : dict
        """
        radius = self._choose_radius(len(data))
        reserved = build_reserved(radius)
        data_cells = self._data_cells(radius)
        M = len(data_cells)

        # RS 인코딩 (단일 채널, 8비트/셀 = 바이트 단위로 자연 정렬)
        codeword = bytes(RSCodec(self.nsym).encode(data))
        cw_len = len(codeword)
        assert cw_len <= M, f"코드워드({cw_len})가 셀 수({M})를 초과"

        # 각 셀에 1바이트 배분, 상위 6비트 = 삼각형, 하위 2비트 = 높이
        triangles = {}
        heights = {}
        for k, (q, r) in enumerate(data_cells):
            if k < cw_len:
                byte_val = codeword[k]
            else:
                byte_val = 0
            triangles[(q, r)] = (byte_val >> BITS_Z) & 0x3F   # bits 7-2
            heights[(q, r)]   = byte_val & 0x03                # bits 1-0

        return {
            'triangles': triangles,
            'heights': heights,
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
                'bits_triangle': BITS_TRI,
                'bits_height': BITS_Z,
                'max_data_bytes': self.max_data_bytes(radius),
                'ecc_rate': self.ecc_rate,
                'nsym': self.nsym,
                'z_levels': Z_LEVELS,
                'encoding': '3d_triangle_relief',
                'color_dependent': False,
            },
        }
