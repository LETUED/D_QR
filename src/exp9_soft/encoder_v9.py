"""
D_QR Phase 9 인코더 — 가변 프로파일 (exp9)

concept/09_design_consolidation.md 해결방안 1 구현:
  프로파일 A (인쇄·실외, 저 SNR): 6비트/셀,  ECC 30% → 인식률 우선
  프로파일 B (화면·근접, 고 SNR): 12비트/셀, ECC 20% → 밀도 우선

셀 기하:
  A → primary 삼각형 6개 (hex_primary_triangles)
  B → 서브삼각형 12개 (hex_sub_triangles)
"""
import math
from reedsolo import RSCodec

from src.core.hex_grid import build_hex_grid, build_reserved, finder_centers
from src.core.bitutils import _bytes_to_bits, _bits_to_bytes, _max_data_for_budget
from src.core.scramble import xor_scramble

EXP_TAG = "exp9"
GRID_RADII = [8, 12, 18, 25, 35, 50]

PROFILES = {
    'A': {'bits_per_cell': 6,  'ecc_rate': 0.30, 'desc': '인쇄·실외 (인식률 우선)'},
    'B': {'bits_per_cell': 12, 'ecc_rate': 0.20, 'desc': '화면·근접 (밀도 우선)'},
}


class DQREncoderV9:
    """
    D_QR Phase 9 인코더 — 프로파일 선택형.

    사용 예:
        enc = DQREncoderV9(profile='A')
        result = enc.encode("데이터".encode())
    """

    def __init__(self, profile: str = 'A', ecc_rate: float = None,
                 scramble: bool = False):
        # scramble (exp15b, 기본 off — exp9~12 산출물·기존 코드 호환 동결):
        # RS 인코딩 전 메시지에 LFSR 고정 마스크 XOR (src/core/scramble) —
        # 전부-0 근방 페이로드도 비자명 코드워드가 되어 디코더 퇴화 가드와
        # 충돌하지 않는다. CRC 아님(검출 기능 없음, 전단사 변환만 — FTO
        # 제약 3 무관). v9 경로는 포맷 링이 없어 수신 측이 알 수 없으므로
        # 디코더(DQRDecoderV11B descramble=True)와 약속해서 켜야 한다.
        if profile not in PROFILES:
            raise ValueError(f"profile은 {list(PROFILES)} 중 하나여야 함: {profile}")
        self.scramble = scramble
        self.profile = profile
        self.bits_per_cell = PROFILES[profile]['bits_per_cell']
        rate = ecc_rate if ecc_rate is not None else PROFILES[profile]['ecc_rate']
        if not (0.0 < rate < 1.0):
            raise ValueError("ecc_rate must be between 0 and 1")
        self.ecc_rate = rate
        self.nsym = max(2, min(127, int(255 * rate)))
        self.dpb = 255 - self.nsym

    def _data_cells(self, radius):
        reserved = build_reserved(radius)
        return [c for c in build_hex_grid(radius) if c not in reserved]

    def max_data_bytes(self, radius):
        M = len(self._data_cells(radius))
        total_bytes = (M * self.bits_per_cell) // 8
        return _max_data_for_budget(total_bytes, self.nsym, self.dpb)

    def _choose_radius(self, n):
        for r in GRID_RADII:
            if self.max_data_bytes(r) >= n:
                return r
        for r in range(GRID_RADII[-1] + 1, 200):
            if self.max_data_bytes(r) >= n:
                return r
        raise ValueError(f"데이터가 너무 큽니다: {n} bytes")

    def encode(self, data: bytes) -> dict:
        radius = self._choose_radius(len(data))
        reserved = build_reserved(radius)
        data_cells = self._data_cells(radius)
        bpc = self.bits_per_cell

        msg = xor_scramble(data) if self.scramble else data
        codeword = bytes(RSCodec(self.nsym).encode(msg))
        cw_len = len(codeword)

        bits = _bytes_to_bits(codeword)
        n_cells_needed = math.ceil(len(bits) / bpc)
        bits += [0] * (n_cells_needed * bpc - len(bits))

        tensor = {}
        for k, (q, r) in enumerate(data_cells):
            if k < n_cells_needed:
                start = k * bpc
                val = 0
                for b in range(bpc):
                    val |= bits[start + b] << (bpc - 1 - b)
                tensor[(q, r)] = val
            else:
                tensor[(q, r)] = 0

        return {
            'tensor': tensor,
            'reserved': reserved,
            'finder_centers': finder_centers(radius),
            'radius': radius,
            'profile': self.profile,
            'bits_per_cell': bpc,
            'nsym': self.nsym,
            'codeword_len': cw_len,
            'data_len': len(data),
            'scrambled': self.scramble,
            'exp_tag': EXP_TAG,
            'capacity': {
                'radius': radius,
                'total_cells': len(build_hex_grid(radius)),
                'data_cells': len(data_cells),
                'bits_per_cell': bpc,
                'max_data_bytes': self.max_data_bytes(radius),
                'ecc_rate': self.ecc_rate,
                'nsym': self.nsym,
                'profile': self.profile,
                'encoding': f"profile_{self.profile}_{bpc}bit_binary",
                'color_dependent': False,
            },
        }
