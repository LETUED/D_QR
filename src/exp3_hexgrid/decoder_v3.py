"""
D_QR Phase 3 디코더 — 육각형 격자 (exp3)

Phase 3 프로토타입 제약:
  - 완벽 정렬 가정 (원근 보정 없음)
  - 렌더러에서 받은 hex_size, origin 사용
  - 각 셀 중앙 픽셀 샘플링
"""
import math
import numpy as np
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

from src.core.constants import rgb_to_nearest_symbol
from src.core.constants import BITS_C1, BITS_C2, BITS_C3
from src.core.bitutils import _bits_to_bytes
from src.core.hex_grid import build_hex_grid, build_reserved, axial_to_pixel


class DQRDecoderV3:
    """
    D_QR Phase 3 디코더.

    사용 예:
        dec = DQRDecoderV3()
        data = dec.decode(img, radius=18, hex_size=18,
                          origin_x=ox, origin_y=oy,
                          nsym=(33,22,21), data_len=L,
                          codeword_lens=(c1l,c2l,c3l),
                          channel_data_bytes=(c1d,c2d,c3d))
    """

    def decode(self, image: Image.Image,
               radius: int,
               hex_size: float,
               origin_x: float,
               origin_y: float,
               nsym: tuple,
               data_len: int,
               codeword_lens: tuple,
               channel_data_bytes: tuple) -> bytes:
        """
        Args:
            image              : PIL.Image.Image (RGB)
            radius             : 격자 반지름
            hex_size           : 렌더러에서 사용한 hex_size
            origin_x, origin_y : 렌더러 원점
            nsym               : (nsym_c1, nsym_c2, nsym_c3)
            data_len           : 원본 데이터 길이
            codeword_lens      : (c1_cw_len, c2_cw_len, c3_cw_len)
            channel_data_bytes : (c1_data, c2_data, c3_data)
        """
        reserved = build_reserved(radius)
        data_cells = [c for c in build_hex_grid(radius) if c not in reserved]

        pixels = np.array(image.convert('RGB'))
        img_h, img_w = pixels.shape[:2]

        c1_syms, c2_syms, c3_syms = [], [], []

        for q, r in data_cells:
            cx, cy = axial_to_pixel(q, r, hex_size, origin_x, origin_y)
            px = int(round(cx))
            py = int(round(cy))

            # 경계 클램핑 (안전장치)
            px = max(0, min(img_w - 1, px))
            py = max(0, min(img_h - 1, py))

            rv = int(pixels[py, px, 0])
            gv = int(pixels[py, px, 1])
            bv = int(pixels[py, px, 2])
            c1, c2, c3 = rgb_to_nearest_symbol(rv, gv, bv)

            c1_syms.append(c1)
            c2_syms.append(c2)
            c3_syms.append(c3)

        nsym_c1, nsym_c2, nsym_c3 = nsym
        c1_cw_len, c2_cw_len, c3_cw_len = codeword_lens
        c1_data_len, c2_data_len, c3_data_len = channel_data_bytes

        def syms_to_cw(syms, bits_per_sym, cw_len):
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
