"""
D_QR Phase 3 인코더 — 육각형 격자 기하 코드 (exp3)

Phase 2와의 차이:
  - 사각형 격자(row,col) → 육각형 격자(q,r) axial 좌표계
  - 4이웃 → 6이웃, 2축 → 3축(q/r/s)
  - QR 파인더 패턴(□) → "7-육각 꽃" 마커 × 3개
  - 동일한 채널 독립 RS ECC (Phase 2에서 계승)
"""
import math
from reedsolo import RSCodec

from src.core.constants import BITS_PER_MODULE
from src.core.bitutils import _max_data_for_budget, _bytes_to_bits, _bits_to_bytes
from src.core.constants import BITS_C1, BITS_C2, BITS_C3
from src.core.hex_grid import build_hex_grid, build_reserved, finder_centers

EXP_TAG = "exp3"
GRID_RADII = [8, 12, 18, 25, 35, 50]


class DQREncoderV3:
    """
    D_QR Phase 3 인코더 — 육각형 격자.

    채널 독립 RS ECC (Phase 2 계승) + 육각형 셀 배치.
    ECC 예산: nsym_total = int(255 * ecc_rate), 3:2:2 비율 분배.

    사용 예:
        enc = DQREncoderV3(ecc_rate=0.30)
        result = enc.encode(b"Hello, Hex D_QR!")
    """

    def __init__(self, ecc_rate: float = 0.30):
        if not (0.0 < ecc_rate < 1.0):
            raise ValueError("ecc_rate must be between 0 and 1")
        self.ecc_rate = ecc_rate

        nsym_total = max(6, min(381, int(255 * ecc_rate)))
        self.nsym_c1 = max(2, round(nsym_total * 3 / 7))
        self.nsym_c2 = max(2, round(nsym_total * 2 / 7))
        self.nsym_c3 = max(2, nsym_total - self.nsym_c1 - self.nsym_c2)
        self.dpb_c1 = 255 - self.nsym_c1
        self.dpb_c2 = 255 - self.nsym_c2
        self.dpb_c3 = 255 - self.nsym_c3

    def _data_cells(self, radius: int) -> list:
        reserved = build_reserved(radius)
        return [c for c in build_hex_grid(radius) if c not in reserved]

    def _channel_data_bytes(self, radius: int) -> tuple:
        M = len(self._data_cells(radius))
        return (
            _max_data_for_budget((M * BITS_C1) // 8, self.nsym_c1, self.dpb_c1),
            _max_data_for_budget((M * BITS_C2) // 8, self.nsym_c2, self.dpb_c2),
            _max_data_for_budget((M * BITS_C3) // 8, self.nsym_c3, self.dpb_c3),
        )

    def max_data_bytes(self, radius: int) -> int:
        return sum(self._channel_data_bytes(radius))

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
        데이터를 D_QR Phase 3 육각형 텐서로 인코딩.

        Returns dict:
            tensor           : dict[(q,r) → (c1,c2,c3)] — 데이터 셀
            reserved         : frozenset — 예약 셀 (파인더)
            finder_centers   : list[(q,r)] — 파인더 중심 3개
            radius           : int
            nsym             : (nsym_c1, nsym_c2, nsym_c3)
            codeword_lens    : (c1_cw_len, c2_cw_len, c3_cw_len)
            channel_data_bytes : (c1d, c2d, c3d)
            data_len         : int
            exp_tag          : "exp3"
            capacity         : dict
        """
        radius = self._choose_radius(len(data))
        reserved = build_reserved(radius)
        data_cells = self._data_cells(radius)
        M = len(data_cells)

        c1d, c2d, c3d = self._channel_data_bytes(radius)
        total_data = c1d + c2d + c3d

        # 입력 패딩 후 채널 분할
        padded = data + bytes(max(0, total_data - len(data)))
        c1_raw = padded[:c1d]
        c2_raw = padded[c1d: c1d + c2d]
        c3_raw = padded[c1d + c2d: c1d + c2d + c3d]

        # 채널별 독립 RS 인코딩
        c1_cw = bytes(RSCodec(self.nsym_c1).encode(c1_raw))
        c2_cw = bytes(RSCodec(self.nsym_c2).encode(c2_raw))
        c3_cw = bytes(RSCodec(self.nsym_c3).encode(c3_raw))

        # 코드워드 비트 스트림 (모듈 경계 맞춰 패딩)
        c1_bits = _bytes_to_bits(c1_cw)
        c2_bits = _bytes_to_bits(c2_cw)
        c3_bits = _bytes_to_bits(c3_cw)

        n_c1_m = math.ceil(len(c1_bits) / BITS_C1)
        n_c2_m = math.ceil(len(c2_bits) / BITS_C2)
        n_c3_m = math.ceil(len(c3_bits) / BITS_C3)
        c1_bits += [0] * (n_c1_m * BITS_C1 - len(c1_bits))
        c2_bits += [0] * (n_c2_m * BITS_C2 - len(c2_bits))
        c3_bits += [0] * (n_c3_m * BITS_C3 - len(c3_bits))

        # 텐서: 각 데이터 셀 → (c1, c2, c3)
        tensor = {}
        for idx, (q, r) in enumerate(data_cells):
            k1 = idx * BITS_C1
            k2 = idx * BITS_C2
            k3 = idx * BITS_C3

            c1 = (c1_bits[k1] << 2 | c1_bits[k1+1] << 1 | c1_bits[k1+2]) if k1 < len(c1_bits) else 0
            c2 = (c2_bits[k2] << 1 | c2_bits[k2+1]) if k2 < len(c2_bits) else 0
            c3 = (c3_bits[k3] << 1 | c3_bits[k3+1]) if k3 < len(c3_bits) else 0

            tensor[(q, r)] = (c1, c2, c3)

        return {
            'tensor': tensor,
            'reserved': reserved,
            'finder_centers': finder_centers(radius),
            'radius': radius,
            'nsym': (self.nsym_c1, self.nsym_c2, self.nsym_c3),
            'codeword_lens': (len(c1_cw), len(c2_cw), len(c3_cw)),
            'channel_data_bytes': (c1d, c2d, c3d),
            'data_len': len(data),
            'exp_tag': EXP_TAG,
            'capacity': {
                'radius': radius,
                'total_cells': len(build_hex_grid(radius)),
                'data_cells': M,
                'reserved_cells': len(reserved),
                'max_data_bytes': self.max_data_bytes(radius),
                'bits_per_cell': BITS_PER_MODULE,
                'ecc_rate': self.ecc_rate,
                'nsym_per_channel': (self.nsym_c1, self.nsym_c2, self.nsym_c3),
                'channel_independent_ecc': True,
                'grid_type': 'hexagonal',
            },
        }
