"""
D_QR Phase 7 디코더 — 프랙탈 중첩 헥사곤 (exp7)
외부 셀 중심에서 내부 미니 격자의 서브삼각형 무게중심 샘플링.
"""
import math
import numpy as np
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

from src.exp7_fractal.encoder_v7 import INNER_CELLS, BITS_PER_INNER
from src.core.bitutils import _bits_to_bytes
from src.core.hex_grid import build_hex_grid, build_reserved, axial_to_pixel, hex_primary_triangles


class DQRDecoderV7:

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def _read_inner_cell(self, pixels, q2, r2, inner_hs, outer_cx, outer_cy):
        """내부 셀 (q2,r2)의 6-bit 값 읽기."""
        img_h, img_w = pixels.shape[:2]
        pri_tris = hex_primary_triangles(q2, r2, inner_hs, outer_cx, outer_cy)
        val = 0
        for idx, (_, centroid) in enumerate(pri_tris):
            tx = max(0, min(img_w-1, int(round(centroid[0]))))
            ty = max(0, min(img_h-1, int(round(centroid[1]))))
            brightness = (int(pixels[ty,tx,0]) + int(pixels[ty,tx,1]) + int(pixels[ty,tx,2])) // 3
            bit = 1 if brightness < self.threshold else 0
            val |= bit << (BITS_PER_INNER - 1 - idx)
        return val

    def decode(self, image: Image.Image,
               outer_radius: int,
               outer_hex_size: float,
               inner_hex_size: float,
               origin_x: float,
               origin_y: float,
               nsym: int,
               data_len: int,
               codeword_len: int) -> bytes:

        reserved = build_reserved(outer_radius)
        outer_cells = [c for c in build_hex_grid(outer_radius) if c not in reserved]

        pixels = np.array(image.convert('RGB'))
        bits = []

        for q1, r1 in outer_cells:
            outer_cx, outer_cy = axial_to_pixel(q1, r1, outer_hex_size, origin_x, origin_y)
            for q2, r2 in INNER_CELLS:
                val = self._read_inner_cell(pixels, q2, r2, inner_hex_size, outer_cx, outer_cy)
                for b in range(BITS_PER_INNER - 1, -1, -1):
                    bits.append((val >> b) & 1)

        codeword = _bits_to_bytes(bits, codeword_len)
        try:
            decoded, _, _ = RSCodec(nsym).decode(codeword)
        except ReedSolomonError as e:
            raise ReedSolomonError(f"프랙탈 RS 디코딩 실패: {e}")
        return bytes(decoded)[:data_len]
