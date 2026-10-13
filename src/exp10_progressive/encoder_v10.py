"""
D_QR Phase 10 인코더 — 패리티 인터리브 dual-grain (exp10)

신뢰도 게이트 점진적 공개(progressive disclosure)의 인코딩 측:
  coarse 셀 ((q+r)%2==0): primary 삼각형 6개 = 6비트/셀,  ECC 30%
                          → stream C (코어 페이로드, 저해상·블러에서도 판독)
  fine   셀 ((q+r)%2==1): 서브삼각형 12개 = 12비트/셀, ECC 20%
                          → stream F (확장 페이로드, 고해상 전용)

패리티 분할인 이유: coarse/fine을 영역으로 나누면 국소 손상이
한 스트림을 통째로 못 읽게 만들 수 있음 → 체스판식 인터리브로 손상을
두 스트림에 고르게 분산.
"""
import math
from reedsolo import RSCodec

from src.core.hex_grid import build_hex_grid, build_reserved, finder_centers
from src.core.bitutils import _bytes_to_bits, _max_data_for_budget
from src.exp9_soft.encoder_v9 import GRID_RADII

EXP_TAG = "exp10"

# grain별 파라미터 — exp9 프로파일 A/B와 동일 값 (블러 생존 경계 실측 근거)
GRAINS = {
    'coarse': {'bits_per_cell': 6,  'ecc_rate': 0.30, 'parity': 0,
               'desc': '코어 페이로드 (저해상·블러 생존)'},
    'fine':   {'bits_per_cell': 12, 'ecc_rate': 0.20, 'parity': 1,
               'desc': '확장 페이로드 (고해상 전용)'},
}


def split_data_cells(radius: int) -> tuple:
    """비예약 셀을 (q+r) 패리티로 (coarse_cells, fine_cells) 분할."""
    reserved = build_reserved(radius)
    coarse, fine = [], []
    for q, r in build_hex_grid(radius):
        if (q, r) in reserved:
            continue
        (coarse if (q + r) % 2 == 0 else fine).append((q, r))
    return coarse, fine


class DQREncoderV10:
    """
    D_QR Phase 10 인코더 — dual-grain 독립 RS 스트림 2개.

    사용 예:
        enc = DQREncoderV10()
        result = enc.encode(coarse_data=b"core", fine_data=b"extended")
    """

    def __init__(self):
        self.nsym_c = max(2, min(127, int(255 * GRAINS['coarse']['ecc_rate'])))
        self.nsym_f = max(2, min(127, int(255 * GRAINS['fine']['ecc_rate'])))
        self.dpb_c = 255 - self.nsym_c
        self.dpb_f = 255 - self.nsym_f

    def max_coarse_bytes(self, radius: int) -> int:
        coarse, _ = split_data_cells(radius)
        budget = (len(coarse) * GRAINS['coarse']['bits_per_cell']) // 8
        return _max_data_for_budget(budget, self.nsym_c, self.dpb_c)

    def max_fine_bytes(self, radius: int) -> int:
        _, fine = split_data_cells(radius)
        budget = (len(fine) * GRAINS['fine']['bits_per_cell']) // 8
        return _max_data_for_budget(budget, self.nsym_f, self.dpb_f)

    def _choose_radius(self, n_coarse: int, n_fine: int) -> int:
        def fits(r):
            return (self.max_coarse_bytes(r) >= n_coarse
                    and self.max_fine_bytes(r) >= n_fine)
        for r in GRID_RADII:
            if fits(r):
                return r
        for r in range(GRID_RADII[-1] + 1, 200):
            if fits(r):
                return r
        raise ValueError(
            f"데이터가 너무 큽니다: coarse {n_coarse}B / fine {n_fine}B")

    def _pack_stream(self, codeword: bytes, cells: list, bpc: int,
                     tensor: dict) -> None:
        """RS 코드워드를 해당 grain 셀들에 bpc비트씩 채움 (잔여 셀은 0)."""
        bits = _bytes_to_bits(codeword)
        n_cells_needed = math.ceil(len(bits) / bpc)
        bits += [0] * (n_cells_needed * bpc - len(bits))

        for k, (q, r) in enumerate(cells):
            if k < n_cells_needed:
                start = k * bpc
                val = 0
                for b in range(bpc):
                    val |= bits[start + b] << (bpc - 1 - b)
                tensor[(q, r)] = val
            else:
                tensor[(q, r)] = 0

    def encode(self, coarse_data: bytes, fine_data: bytes) -> dict:
        radius = self._choose_radius(len(coarse_data), len(fine_data))
        reserved = build_reserved(radius)
        coarse_cells, fine_cells = split_data_cells(radius)

        codeword_c = bytes(RSCodec(self.nsym_c).encode(coarse_data))
        codeword_f = bytes(RSCodec(self.nsym_f).encode(fine_data))

        tensor = {}
        self._pack_stream(codeword_c, coarse_cells,
                          GRAINS['coarse']['bits_per_cell'], tensor)
        self._pack_stream(codeword_f, fine_cells,
                          GRAINS['fine']['bits_per_cell'], tensor)

        return {
            'tensor': tensor,
            'reserved': reserved,
            'finder_centers': finder_centers(radius),
            'radius': radius,
            'nsym_c': self.nsym_c,
            'nsym_f': self.nsym_f,
            'codeword_len_c': len(codeword_c),
            'codeword_len_f': len(codeword_f),
            'coarse_len': len(coarse_data),
            'fine_len': len(fine_data),
            'exp_tag': EXP_TAG,
            'capacity': {
                'radius': radius,
                'total_cells': len(build_hex_grid(radius)),
                'coarse_cells': len(coarse_cells),
                'fine_cells': len(fine_cells),
                'max_coarse_bytes': self.max_coarse_bytes(radius),
                'max_fine_bytes': self.max_fine_bytes(radius),
                'ecc_rate_c': GRAINS['coarse']['ecc_rate'],
                'ecc_rate_f': GRAINS['fine']['ecc_rate'],
                'nsym_c': self.nsym_c,
                'nsym_f': self.nsym_f,
                'encoding': 'dual_grain_parity_interleave',
                'color_dependent': False,
            },
        }
