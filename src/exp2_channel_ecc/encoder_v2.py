"""
D_QR Phase 2 인코더 — 채널 독립 RS ECC (exp2)

Phase 1과의 차이:
  - 하나의 통합 RS 코드워드 → 채널(c1/c2/c3)별 독립 RS 코드워드
  - ECC 예산을 채널 비트 수에 비례 배분 (3:2:2)
  - c1 손상이 c2/c3 데이터에 영향 없음 (채널 격리)
  - 동일 ECC rate에서 Phase 1과 유사한 데이터 용량 유지
"""
import math
import numpy as np
from reedsolo import RSCodec

from src.core.constants import BITS_PER_MODULE, GRID_SIZES
from src.core.bitutils import (_build_reserved_mask, _max_data_for_budget,
                               _bytes_to_bits, _bits_to_bytes)

BITS_C1, BITS_C2, BITS_C3 = 3, 2, 2
EXP_TAG = "exp2"


class DQREncoderV2:
    """
    D_QR Phase 2 인코더.

    채널 독립 RS ECC 설계:
      nsym_total = int(255 * ecc_rate) 분배:
        nsym_c1 = round(total * 3/7)   # c1이 3비트/모듈로 가장 큰 몫
        nsym_c2 = round(total * 2/7)
        nsym_c3 = total - c1 - c2

    사용 예:
        enc = DQREncoderV2(ecc_rate=0.30)
        result = enc.encode(b"Hello, D_QR!")
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

    def _n_data_modules(self, S: int) -> int:
        return int(np.sum(~_build_reserved_mask(S)))

    def _channel_data_bytes(self, S: int) -> tuple:
        M = self._n_data_modules(S)
        return (
            _max_data_for_budget((M * BITS_C1) // 8, self.nsym_c1, self.dpb_c1),
            _max_data_for_budget((M * BITS_C2) // 8, self.nsym_c2, self.dpb_c2),
            _max_data_for_budget((M * BITS_C3) // 8, self.nsym_c3, self.dpb_c3),
        )

    def max_data_bytes(self, S: int) -> int:
        return sum(self._channel_data_bytes(S))

    def _choose_grid_size(self, n: int) -> int:
        for S in GRID_SIZES:
            if self.max_data_bytes(S) >= n:
                return S
        for S in range(GRID_SIZES[-1] + 4, 500, 4):
            if self.max_data_bytes(S) >= n:
                return S
        raise ValueError(f"데이터가 너무 큽니다: {n} bytes")

    def encode(self, data: bytes) -> dict:
        """
        데이터를 D_QR Phase 2 텐서로 인코딩.

        Returns dict:
            tensor           : (S, S, 3) uint8
            reserved         : (S, S) bool
            grid_size        : int
            nsym             : (nsym_c1, nsym_c2, nsym_c3)
            codeword_lens    : (c1_cw_len, c2_cw_len, c3_cw_len)
            channel_data_bytes : (c1d, c2d, c3d)
            data_len         : int
            exp_tag          : str — "exp2"
            capacity         : dict
        """
        S = self._choose_grid_size(len(data))
        reserved = _build_reserved_mask(S)
        M = self._n_data_modules(S)

        c1d, c2d, c3d = self._channel_data_bytes(S)
        total_data = c1d + c2d + c3d

        # 입력 패딩 후 채널별 분할
        padded = data + bytes(max(0, total_data - len(data)))
        c1_raw = padded[:c1d]
        c2_raw = padded[c1d:c1d + c2d]
        c3_raw = padded[c1d + c2d:c1d + c2d + c3d]

        # 채널별 독립 RS 인코딩
        c1_cw = bytes(RSCodec(self.nsym_c1).encode(c1_raw))
        c2_cw = bytes(RSCodec(self.nsym_c2).encode(c2_raw))
        c3_cw = bytes(RSCodec(self.nsym_c3).encode(c3_raw))

        # 코드워드 → 비트 스트림 (모듈 경계 맞춰 패딩)
        c1_bits = _bytes_to_bits(c1_cw)
        c2_bits = _bytes_to_bits(c2_cw)
        c3_bits = _bytes_to_bits(c3_cw)

        n_c1_mods = math.ceil(len(c1_bits) / BITS_C1)
        n_c2_mods = math.ceil(len(c2_bits) / BITS_C2)
        n_c3_mods = math.ceil(len(c3_bits) / BITS_C3)
        c1_bits += [0] * (n_c1_mods * BITS_C1 - len(c1_bits))
        c2_bits += [0] * (n_c2_mods * BITS_C2 - len(c2_bits))
        c3_bits += [0] * (n_c3_mods * BITS_C3 - len(c3_bits))

        # 텐서 채우기: 모듈 k → (c1[k], c2[k], c3[k]) 독립
        tensor = np.full((S, S, 3), 255, dtype=np.uint8)
        mod_idx = 0

        for row in range(S):
            for col in range(S):
                if reserved[row, col]:
                    continue

                k1 = mod_idx * BITS_C1
                k2 = mod_idx * BITS_C2
                k3 = mod_idx * BITS_C3

                c1 = (c1_bits[k1] << 2 | c1_bits[k1+1] << 1 | c1_bits[k1+2]) if k1 < len(c1_bits) else 0
                c2 = (c2_bits[k2] << 1 | c2_bits[k2+1]) if k2 < len(c2_bits) else 0
                c3 = (c3_bits[k3] << 1 | c3_bits[k3+1]) if k3 < len(c3_bits) else 0

                tensor[row, col] = [c1, c2, c3]
                mod_idx += 1

        return {
            'tensor': tensor,
            'reserved': reserved,
            'grid_size': S,
            'nsym': (self.nsym_c1, self.nsym_c2, self.nsym_c3),
            'codeword_lens': (len(c1_cw), len(c2_cw), len(c3_cw)),
            'channel_data_bytes': (c1d, c2d, c3d),
            'data_len': len(data),
            'exp_tag': EXP_TAG,
            'capacity': {
                'grid_size': S,
                'data_modules': M,
                'max_data_bytes': self.max_data_bytes(S),
                'bits_per_module': BITS_PER_MODULE,
                'ecc_rate': self.ecc_rate,
                'nsym_per_channel': (self.nsym_c1, self.nsym_c2, self.nsym_c3),
                'channel_independent_ecc': True,
            },
        }
