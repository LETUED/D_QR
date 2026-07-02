"""
D_QR Phase 4 디코더 — 기하 서브삼각형 (exp4)

디코딩 원리:
  - 각 삼각형의 무게중심(centroid) 픽셀을 샘플링
  - 밝기 < threshold → filled(1), 밝기 ≥ threshold → empty(0)
  - 색상 정보 불필요 — 단색 밝기만 사용
  - threshold: fg/bg 중간값 (기본 128)
"""
import math
import numpy as np
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

from src.exp4_geometric.encoder_v4 import BITS_PER_CELL
from src.core.bitutils import _bits_to_bytes
from src.core.hex_grid import build_hex_grid, build_reserved, axial_to_pixel, hex_vertices


class DQRDecoderV4:
    """
    D_QR Phase 4 디코더.

    threshold: fg(어두운 색)와 bg(밝은 색)의 밝기 경계
    반전 색상(bg가 어두운 경우): invert=True

    사용 예:
        dec = DQRDecoderV4()
        data = dec.decode(img, radius=18, hex_size=28,
                          origin_x=ox, origin_y=oy,
                          nsym=51, data_len=L, codeword_len=cw_len)
    """

    def __init__(self, threshold: int = 128, invert: bool = False):
        self.threshold = threshold
        self.invert = invert

    def _read_cell(self, pixels: np.ndarray,
                   q: int, r: int,
                   hex_size: float, ox: float, oy: float) -> int:
        """
        셀 (q,r)의 6개 삼각형 무게중심을 샘플링 → 6비트 값 반환.
        무게중심 = (hex_center + corner_i + corner_(i+1)) / 3
        """
        img_h, img_w = pixels.shape[:2]
        verts = hex_vertices(q, r, hex_size, ox, oy)
        cx, cy = axial_to_pixel(q, r, hex_size, ox, oy)

        val = 0
        for i in range(BITS_PER_CELL):
            v_a = verts[i]
            v_b = verts[(i + 1) % 6]

            # 삼각형 무게중심
            tx = int(round((cx + v_a[0] + v_b[0]) / 3))
            ty = int(round((cy + v_a[1] + v_b[1]) / 3))
            tx = max(0, min(img_w - 1, tx))
            ty = max(0, min(img_h - 1, ty))

            r_ch = int(pixels[ty, tx, 0])
            g_ch = int(pixels[ty, tx, 1])
            b_ch = int(pixels[ty, tx, 2])
            brightness = (r_ch + g_ch + b_ch) // 3

            # dark < threshold → filled(1)
            bit = 1 if brightness < self.threshold else 0
            if self.invert:
                bit = 1 - bit

            val |= bit << (BITS_PER_CELL - 1 - i)

        return val

    def decode(self, image: Image.Image,
               radius: int,
               hex_size: float,
               origin_x: float,
               origin_y: float,
               nsym: int,
               data_len: int,
               codeword_len: int) -> bytes:
        """
        Args:
            image        : PIL.Image.Image
            radius       : 격자 반지름
            hex_size     : 렌더러와 동일한 값
            origin_x/y   : 렌더러 원점
            nsym         : RS ECC 심볼 수
            data_len     : 원본 데이터 길이
            codeword_len : RS 코드워드 총 바이트 수
        """
        reserved = build_reserved(radius)
        data_cells = [c for c in build_hex_grid(radius) if c not in reserved]

        pixels = np.array(image.convert('RGB'))
        img_h, img_w = pixels.shape[:2]

        # 각 셀에서 6비트 값 읽기
        bits = []
        for q, r in data_cells:
            val = self._read_cell(pixels, q, r, hex_size, origin_x, origin_y)
            for b in range(BITS_PER_CELL - 1, -1, -1):
                bits.append((val >> b) & 1)

        # 비트 → 코드워드 바이트
        codeword = _bits_to_bytes(bits, codeword_len)

        # RS 디코딩
        try:
            decoded, _, _ = RSCodec(nsym).decode(codeword)
        except ReedSolomonError as e:
            raise ReedSolomonError(f"RS 디코딩 실패: {e}")

        return bytes(decoded)[:data_len]
