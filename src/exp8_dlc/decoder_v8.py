"""
D_QR Phase 8 디코더 — DLC 듀얼 스트림 (exp8)
Stream A(짝수 셀) / Stream B(홀수 셀)를 독립적으로 디코딩.
"""
import math
import numpy as np
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

from src.exp8_dlc.encoder_v8 import BITS_PER_CELL
from src.core.bitutils import _bits_to_bytes
from src.core.hex_grid import build_hex_grid, build_reserved, hex_sub_triangles


class DQRDecoderV8:

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def _read_cell(self, pixels, q, r, hex_size, ox, oy):
        img_h, img_w = pixels.shape[:2]
        sub_tris = hex_sub_triangles(q, r, hex_size, ox, oy)
        val = 0
        for idx, (_, centroid) in enumerate(sub_tris):
            tx = max(0, min(img_w-1, int(round(centroid[0]))))
            ty = max(0, min(img_h-1, int(round(centroid[1]))))
            b = (int(pixels[ty,tx,0]) + int(pixels[ty,tx,1]) + int(pixels[ty,tx,2])) // 3
            bit = 1 if b < self.threshold else 0
            val |= bit << (BITS_PER_CELL - 1 - idx)
        return val

    def _decode_stream(self, pixels, cells, hex_size, ox, oy, nsym, codeword_len, data_len):
        bits = []
        for q, r in cells:
            val = self._read_cell(pixels, q, r, hex_size, ox, oy)
            for b in range(BITS_PER_CELL - 1, -1, -1):
                bits.append((val >> b) & 1)
        cw = _bits_to_bytes(bits, codeword_len)
        decoded, _, _ = RSCodec(nsym).decode(cw)
        return bytes(decoded)[:data_len]

    def decode_a(self, image: Image.Image, encoded: dict, render_info: dict) -> bytes:
        """Stream A (짝수 셀) 디코딩."""
        pixels = np.array(image.convert('RGB'))
        meta = encoded['stream_meta']['a']
        return self._decode_stream(
            pixels, encoded['stream_a_cells'],
            render_info['hex_size'], render_info['origin_x'], render_info['origin_y'],
            encoded['nsym'], meta['codeword_len'], meta['data_len'],
        )

    def decode_b(self, image: Image.Image, encoded: dict, render_info: dict) -> bytes:
        """Stream B (홀수 셀) 디코딩."""
        pixels = np.array(image.convert('RGB'))
        meta = encoded['stream_meta']['b']
        return self._decode_stream(
            pixels, encoded['stream_b_cells'],
            render_info['hex_size'], render_info['origin_x'], render_info['origin_y'],
            encoded['nsym'], meta['codeword_len'], meta['data_len'],
        )

    def decode_both(self, image: Image.Image, encoded: dict, render_info: dict) -> tuple:
        """두 스트림 모두 디코딩. 반환: (data_a, data_b)"""
        return (
            self.decode_a(image, encoded, render_info),
            self.decode_b(image, encoded, render_info),
        )
