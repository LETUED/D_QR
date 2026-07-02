"""
D_QR Phase 5 디코더 — Z축 부조 + 삼각형 (exp5)

디코딩 원리:
  삼각형 채널: 2D 이미지에서 삼각형 무게중심 밝기 임계값 (exp4 동일)
  Z-height 채널: 물리 제조 시 구조광/LiDAR로 측정.
                 프로토타입: 인코더 메타데이터에서 직접 읽음 (완벽 시뮬레이션)

각 셀 1바이트 = (tri_val << 2) | z_level 복원 → RS 디코딩
"""
import math
import numpy as np
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

from src.exp5_zdepth.encoder_v5 import BITS_TRI, BITS_Z
from src.core.bitutils import _bits_to_bytes
from src.core.hex_grid import build_hex_grid, build_reserved, axial_to_pixel, hex_vertices

HEIGHT_GRAY = [255, 200, 130, 60]  # z_level → 배경 명도 (renderer_v5 와 동기)


class DQRDecoderV5:
    """
    D_QR Phase 5 디코더.

    프로토타입 모드:
      heights 파라미터를 직접 전달 (물리 스캐너 없이 시뮬레이션)
      실제 제조물 디코딩은 heights를 구조광 스캐너 결과로 대체

    사용 예:
        dec = DQRDecoderV5()
        data = dec.decode(img_2d,
                          heights=result['heights'],   # 시뮬레이션
                          radius=..., hex_size=...,
                          origin_x=..., origin_y=...,
                          nsym=..., data_len=..., codeword_len=...)
    """

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def _read_triangle(self, pixels, q, r, hex_size, ox, oy, z_level: int = 0):
        """
        셀의 6개 삼각형 무게중심 샘플링 → 6비트 값.
        z_level을 알면 배경 밝기(HEIGHT_GRAY[z])와 fg(≈0)의
        중간값을 threshold로 사용해 오판 방지.
        """
        img_h, img_w = pixels.shape[:2]
        verts = hex_vertices(q, r, hex_size, ox, oy)
        cx, cy = axial_to_pixel(q, r, hex_size, ox, oy)

        # 배경 밝기에 맞는 적응 threshold
        bg_brightness = HEIGHT_GRAY[z_level]
        threshold = bg_brightness // 2

        val = 0
        for i in range(BITS_TRI):
            v_a = verts[i]
            v_b = verts[(i + 1) % 6]
            tx = int(round((cx + v_a[0] + v_b[0]) / 3))
            ty = int(round((cy + v_a[1] + v_b[1]) / 3))
            tx = max(0, min(img_w - 1, tx))
            ty = max(0, min(img_h - 1, ty))

            r_ch = int(pixels[ty, tx, 0])
            g_ch = int(pixels[ty, tx, 1])
            b_ch = int(pixels[ty, tx, 2])
            brightness = (r_ch + g_ch + b_ch) // 3
            bit = 1 if brightness < threshold else 0
            val |= bit << (BITS_TRI - 1 - i)
        return val

    def decode(self, image: Image.Image,
               heights: dict,       # {(q,r): 0~3} — 시뮬레이션 or 스캐너 결과
               radius: int,
               hex_size: float,
               origin_x: float,
               origin_y: float,
               nsym: int,
               data_len: int,
               codeword_len: int) -> bytes:
        """
        Args:
            heights: 셀별 z-depth 레벨 dict (프로토타입: 인코더에서 전달)
                     실제 제조물: 구조광 스캐너 측정값으로 대체
        """
        reserved = build_reserved(radius)
        data_cells = [c for c in build_hex_grid(radius) if c not in reserved]

        pixels = np.array(image.convert('RGB'))

        codeword = bytearray()
        for q, r in data_cells:
            z_level = heights.get((q, r), 0)
            tri_val = self._read_triangle(pixels, q, r, hex_size, origin_x, origin_y, z_level)
            byte_val = ((tri_val & 0x3F) << BITS_Z) | (z_level & 0x03)
            codeword.append(byte_val)

        # codeword_len만큼만 RS 디코딩
        cw = bytes(codeword[:codeword_len])
        try:
            decoded, _, _ = RSCodec(nsym).decode(cw)
        except ReedSolomonError as e:
            raise ReedSolomonError(f"RS 디코딩 실패: {e}")

        return bytes(decoded)[:data_len]
