"""
D_QR Phase 6 디코더 — 12-서브삼각형 이진 (exp6)
각 서브삼각형의 무게중심 밝기 임계값으로 12비트 셀 값 복원.
"""
import math
import numpy as np
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

from src.exp6_subtri.encoder_v6 import BITS_PER_CELL
from src.core.bitutils import _bits_to_bytes
from src.core.hex_grid import build_hex_grid, build_reserved, hex_sub_triangles


class DQRDecoderV6:
    """
    D_QR Phase 6 디코더.

    사용 예:
        dec = DQRDecoderV6()
        data = dec.decode(img, radius=18, hex_size=32,
                          origin_x=ox, origin_y=oy,
                          nsym=51, data_len=L, codeword_len=cw_len)
    """

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def _read_cell(self, pixels, q, r, hex_size, ox, oy):
        """12개 서브삼각형 무게중심 샘플링 → 12비트 정수."""
        img_h, img_w = pixels.shape[:2]
        sub_tris = hex_sub_triangles(q, r, hex_size, ox, oy)

        val = 0
        for idx, (_verts, centroid) in enumerate(sub_tris):
            tx = int(round(centroid[0]))
            ty = int(round(centroid[1]))
            tx = max(0, min(img_w - 1, tx))
            ty = max(0, min(img_h - 1, ty))

            r_ch = int(pixels[ty, tx, 0])
            g_ch = int(pixels[ty, tx, 1])
            b_ch = int(pixels[ty, tx, 2])
            brightness = (r_ch + g_ch + b_ch) // 3

            bit = 1 if brightness < self.threshold else 0
            val |= bit << (BITS_PER_CELL - 1 - idx)

        return val

    def decode(self, image: Image.Image,
               radius: int,
               hex_size: float,
               origin_x: float,
               origin_y: float,
               nsym: int,
               data_len: int,
               codeword_len: int) -> bytes:
        reserved = build_reserved(radius)
        data_cells = [c for c in build_hex_grid(radius) if c not in reserved]

        pixels = np.array(image.convert('RGB'))

        bits = []
        for q, r in data_cells:
            val = self._read_cell(pixels, q, r, hex_size, origin_x, origin_y)
            for b in range(BITS_PER_CELL - 1, -1, -1):
                bits.append((val >> b) & 1)

        codeword = _bits_to_bytes(bits, codeword_len)

        try:
            decoded, _, _ = RSCodec(nsym).decode(codeword)
        except ReedSolomonError as e:
            raise ReedSolomonError(f"RS 디코딩 실패: {e}")

        return bytes(decoded)[:data_len]
