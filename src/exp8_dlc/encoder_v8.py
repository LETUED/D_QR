"""
D_QR Phase 8 인코더 — 차원 링크 체인 (DLC, exp8)

동일한 2D 이미지 안에 2개의 완전히 독립된 데이터 스트림:
  Stream A: 짝수 인덱스 셀 (0, 2, 4, ...)
  Stream B: 홀수 인덱스 셀 (1, 3, 5, ...)

각 스트림은 독립적인 RS ECC 보호를 받음:
  - Stream A 손상 → Stream B 영향 없음
  - Stream B 손상 → Stream A 영향 없음

용량: 각 스트림이 exp6의 절반 → 합산 = exp6과 동일
      하지만 두 개의 독립적인 데이터셋을 1장에 담음 (차원 분리)

적용:
  - Stream A: 공개 데이터 (누구나 읽을 수 있는 정보)
  - Stream B: 비공개 데이터 (권한있는 독자만 알고있는 2차 채널)
  - MultiDimData와 연계: 차원별 독립 저장
"""
import math
from reedsolo import RSCodec

from src.core.hex_grid import build_hex_grid, build_reserved, finder_centers
from src.core.bitutils import _max_data_for_budget, _bytes_to_bits, _bits_to_bytes

EXP_TAG = "exp8"
BITS_PER_CELL = 12  # exp6와 동일 (12 서브삼각형 × 1비트)
GRID_RADII = [8, 12, 18, 25, 35, 50]


class DQREncoderV8:
    """
    D_QR Phase 8 DLC 인코더.

    사용 예:
        enc = DQREncoderV8(ecc_rate=0.20)
        result = enc.encode(data_a, data_b)
        # 두 개의 독립 데이터셋을 하나의 코드에 담음
    """

    def __init__(self, ecc_rate: float = 0.20):
        self.ecc_rate = ecc_rate
        self.nsym = max(2, min(127, int(255 * ecc_rate)))
        self.dpb = 255 - self.nsym

    def _data_cells(self, radius):
        reserved = build_reserved(radius)
        return [c for c in build_hex_grid(radius) if c not in reserved]

    def _stream_cells(self, radius):
        cells = self._data_cells(radius)
        return cells[::2], cells[1::2]   # A=짝수, B=홀수

    def max_data_bytes_per_stream(self, radius) -> int:
        a_cells, _ = self._stream_cells(radius)
        total_bytes = (len(a_cells) * BITS_PER_CELL) // 8
        return _max_data_for_budget(total_bytes, self.nsym, self.dpb)

    def _choose_radius(self, n_a: int, n_b: int) -> int:
        for r in GRID_RADII:
            cap = self.max_data_bytes_per_stream(r)
            if cap >= n_a and cap >= n_b:
                return r
        raise ValueError(f"데이터가 너무 큽니다: A={n_a}B, B={n_b}B")

    def encode(self, data_a: bytes, data_b: bytes) -> dict:
        """
        두 독립 데이터를 하나의 헥사곤 그리드에 인코딩.

        Returns dict:
            tensor         : dict[(q,r) → int(0-4095)]
            stream_meta    : {'a': {...}, 'b': {...}}  — 각 스트림의 메타데이터
            stream_a_cells : list[(q,r)]
            stream_b_cells : list[(q,r)]
            reserved, finder_centers, radius
            exp_tag        : "exp8"
        """
        radius = self._choose_radius(len(data_a), len(data_b))
        reserved = build_reserved(radius)
        a_cells, b_cells = self._stream_cells(radius)

        def encode_stream(data, cells):
            cw = bytes(RSCodec(self.nsym).encode(data))
            bits = _bytes_to_bits(cw)
            n_needed = math.ceil(len(bits) / BITS_PER_CELL)
            bits += [0] * (n_needed * BITS_PER_CELL - len(bits))
            result = {}
            for k, (q, r) in enumerate(cells):
                if k < n_needed:
                    start = k * BITS_PER_CELL
                    val = 0
                    for b in range(BITS_PER_CELL):
                        val |= bits[start + b] << (BITS_PER_CELL - 1 - b)
                    result[(q, r)] = val
                else:
                    result[(q, r)] = 0
            return result, len(cw)

        tensor_a, cw_len_a = encode_stream(data_a, a_cells)
        tensor_b, cw_len_b = encode_stream(data_b, b_cells)

        tensor = {**tensor_a, **tensor_b}
        cap = self.max_data_bytes_per_stream(radius)

        return {
            'tensor': tensor,
            'stream_a_cells': a_cells,
            'stream_b_cells': b_cells,
            'stream_meta': {
                'a': {'codeword_len': cw_len_a, 'data_len': len(data_a)},
                'b': {'codeword_len': cw_len_b, 'data_len': len(data_b)},
            },
            'reserved': reserved,
            'finder_centers': finder_centers(radius),
            'radius': radius,
            'nsym': self.nsym,
            'exp_tag': EXP_TAG,
            'capacity': {
                'radius': radius,
                'total_cells': len(build_hex_grid(radius)),
                'stream_a_cells': len(a_cells),
                'stream_b_cells': len(b_cells),
                'bits_per_cell': BITS_PER_CELL,
                'max_bytes_per_stream': cap,
                'ecc_rate': self.ecc_rate,
                'independent_streams': 2,
                'encoding': 'dlc_dual_stream',
            },
        }
