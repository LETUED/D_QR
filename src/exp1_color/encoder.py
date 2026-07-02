"""
D_QR Phase 1 인코더
bitstream → 텐서 T (H, W, 3)

RS ECC: 블록당 255바이트 (GF(256) 표준), reedsolo가 자동 청킹
"""
import math
import numpy as np
from reedsolo import RSCodec

from src.core.constants import BITS_PER_MODULE, FINDER_SIZE, SEPARATOR, GRID_SIZES
from src.core.bitutils import _build_reserved_mask


class DQREncoder:
    """
    D_QR Phase 1 인코더.

    RS ECC 설계:
      - 블록당 255바이트 (GF(256) 최대)
      - ecc_rate=0.30 → nsym=76, 데이터 179바이트/블록
      - reedsolo가 긴 데이터를 자동으로 청킹 처리

    사용 예:
        enc = DQREncoder(ecc_rate=0.30)
        result = enc.encode(b"Hello, D_QR!")
    """

    def __init__(self, ecc_rate: float = 0.30):
        if not (0.0 < ecc_rate < 1.0):
            raise ValueError("ecc_rate must be between 0 and 1")
        self.ecc_rate = ecc_rate
        # GF(256) 블록당 ECC 심볼 수 (최대 127)
        self.nsym = max(2, min(127, int(255 * ecc_rate)))
        self.data_per_block = 255 - self.nsym
        self._rsc = RSCodec(self.nsym)

    def _n_data_modules(self, S: int) -> int:
        return int(np.sum(~_build_reserved_mask(S)))

    def max_data_bytes(self, S: int) -> int:
        """그리드 크기 S에서 담을 수 있는 최대 데이터 바이트."""
        n_mod = self._n_data_modules(S)
        total_bytes = (n_mod * BITS_PER_MODULE) // 8
        # 총 바이트 중 데이터 비율: data_per_block / 255
        return int(total_bytes * self.data_per_block / 255)

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
        데이터를 D_QR 텐서로 인코딩.

        Returns dict:
            tensor       : (S, S, 3) uint8 — 데이터 모듈은 (c1,c2,c3), reserved는 (255,255,255)
            reserved     : (S, S) bool
            grid_size    : int
            nsym         : int — 블록당 ECC 심볼 수
            codeword_len : int — RS 인코딩 후 총 바이트 수 (디코더에 필요)
            data_len     : int — 원본 데이터 길이
            capacity     : dict
        """
        S = self._choose_grid_size(len(data))
        reserved = _build_reserved_mask(S)

        if len(data) > self.max_data_bytes(S):
            raise ValueError(
                f"데이터 {len(data)} bytes > 그리드 {S}×{S} 최대 {self.max_data_bytes(S)} bytes"
            )

        # Reed-Solomon 인코딩 (reedsolo가 자동 청킹)
        codeword = bytes(self._rsc.encode(data))
        codeword_len = len(codeword)

        # 바이트 → 비트 스트림
        bits = []
        for byte in codeword:
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)

        # 7비트씩 모듈에 배치 (행 우선)
        tensor = np.full((S, S, 3), 255, dtype=np.uint8)
        bit_idx = 0

        for row in range(S):
            for col in range(S):
                if reserved[row, col]:
                    continue
                if bit_idx + BITS_PER_MODULE > len(bits):
                    break
                c1 = (bits[bit_idx] << 2) | (bits[bit_idx + 1] << 1) | bits[bit_idx + 2]
                c2 = (bits[bit_idx + 3] << 1) | bits[bit_idx + 4]
                c3 = (bits[bit_idx + 5] << 1) | bits[bit_idx + 6]
                bit_idx += BITS_PER_MODULE
                tensor[row, col] = [c1, c2, c3]

        return {
            'tensor': tensor,
            'reserved': reserved,
            'grid_size': S,
            'nsym': self.nsym,
            'codeword_len': codeword_len,
            'data_len': len(data),
            'capacity': {
                'grid_size': S,
                'data_modules': self._n_data_modules(S),
                'max_data_bytes': self.max_data_bytes(S),
                'bits_per_module': BITS_PER_MODULE,
                'ecc_rate': self.ecc_rate,
                'nsym_per_block': self.nsym,
            },
        }
