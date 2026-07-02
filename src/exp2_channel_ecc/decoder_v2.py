"""
D_QR Phase 2 디코더 — 채널 독립 RS ECC (exp2)
각 채널을 독립적으로 RS 디코딩하여 원본 데이터 복원.
"""
import math
import numpy as np
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

from src.core.constants import rgb_to_nearest_symbol
from src.core.bitutils import _build_reserved_mask, _bits_to_bytes
from src.exp2_channel_ecc.encoder_v2 import BITS_C1, BITS_C2, BITS_C3

_DEFAULT_QUIET_ZONE = 4


class DQRDecoderV2:
    """
    D_QR Phase 2 디코더.

    사용 예:
        dec = DQRDecoderV2()
        data = dec.decode(img, grid_size=37, module_px=20,
                          nsym=(33,22,21), data_len=L,
                          codeword_lens=(c1l,c2l,c3l),
                          channel_data_bytes=(c1d,c2d,c3d))
    """

    def __init__(self, quiet_zone: int = _DEFAULT_QUIET_ZONE):
        self.quiet_zone = quiet_zone

    def decode(self, image: Image.Image,
               grid_size: int,
               module_px: int,
               nsym: tuple,
               data_len: int,
               codeword_lens: tuple,
               channel_data_bytes: tuple) -> bytes:
        """
        Args:
            image              : PIL.Image.Image (RGB)
            grid_size          : 그리드 크기 S
            module_px          : 모듈당 픽셀 크기
            nsym               : (nsym_c1, nsym_c2, nsym_c3)
            data_len           : 원본 데이터 길이
            codeword_lens      : (c1_cw_len, c2_cw_len, c3_cw_len) — 바이트 수
            channel_data_bytes : (c1_data, c2_data, c3_data) — 채널별 데이터 바이트 수
        """
        S = grid_size
        px = module_px
        qz = self.quiet_zone
        reserved = _build_reserved_mask(S)
        pixels = np.array(image.convert('RGB'))

        c1_syms, c2_syms, c3_syms = [], [], []

        for row in range(S):
            for col in range(S):
                if reserved[row, col]:
                    continue
                cx = (col + qz) * px + px // 2
                cy = (row + qz) * px + px // 2
                r = int(pixels[cy, cx, 0])
                g = int(pixels[cy, cx, 1])
                b = int(pixels[cy, cx, 2])
                c1, c2, c3 = rgb_to_nearest_symbol(r, g, b)
                c1_syms.append(c1)
                c2_syms.append(c2)
                c3_syms.append(c3)

        nsym_c1, nsym_c2, nsym_c3 = nsym
        c1_cw_len, c2_cw_len, c3_cw_len = codeword_lens
        c1_data_len, c2_data_len, c3_data_len = channel_data_bytes

        def syms_to_cw(syms, bits_per_sym, cw_len):
            # 심볼 → 비트 → bytes (cw_len 바이트 정확히)
            bits = []
            for s in syms:
                for shift in range(bits_per_sym - 1, -1, -1):
                    bits.append((s >> shift) & 1)
            return _bits_to_bytes(bits, cw_len)

        n_c1 = math.ceil(c1_cw_len * 8 / BITS_C1)
        n_c2 = math.ceil(c2_cw_len * 8 / BITS_C2)
        n_c3 = math.ceil(c3_cw_len * 8 / BITS_C3)

        c1_cw = syms_to_cw(c1_syms[:n_c1], BITS_C1, c1_cw_len)
        c2_cw = syms_to_cw(c2_syms[:n_c2], BITS_C2, c2_cw_len)
        c3_cw = syms_to_cw(c3_syms[:n_c3], BITS_C3, c3_cw_len)

        try:
            c1_dec, _, _ = RSCodec(nsym_c1).decode(c1_cw)
            c2_dec, _, _ = RSCodec(nsym_c2).decode(c2_cw)
            c3_dec, _, _ = RSCodec(nsym_c3).decode(c3_cw)
        except ReedSolomonError as e:
            raise ReedSolomonError(f"채널 RS 디코딩 실패: {e}")

        result = (bytes(c1_dec)[:c1_data_len]
                  + bytes(c2_dec)[:c2_data_len]
                  + bytes(c3_dec)[:c3_data_len])
        return result[:data_len]
