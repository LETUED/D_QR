"""
D_QR Phase 3 렌더러 — 육각형 격자 (exp3)

렌더링 규칙:
  - 데이터 셀: (c1,c2,c3) → HSV → RGB 컬러 정육각형
  - 파인더 중심(1개): 검정
  - 파인더 이웃(6개): 흰색
  - 배경 / quiet zone: 흰색
"""
import math
from PIL import Image, ImageDraw

from src.core.constants import symbol_to_rgb
from src.core.hex_grid import (build_hex_grid, build_reserved, finder_centers,
                        finder_cell_set, hex_vertices, grid_bounding_box,
                        axial_to_pixel)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
QUIET_CELLS = 2  # 격자 가장자리 바깥 여백 (hex 단위)


class DQRRendererV3:
    """
    D_QR Phase 3 이미지 렌더러.

    사용 예:
        ren = DQRRendererV3(hex_size=18)
        img, info = ren.render(encoded)
        img.save("exp3_dqr.png")
        # info: {'hex_size': 18, 'origin_x': ..., 'origin_y': ...}
    """

    def __init__(self, hex_size: int = 18, quiet_cells: int = QUIET_CELLS):
        self.hex_size = hex_size
        self.quiet_cells = quiet_cells

    def _draw_hex(self, draw: ImageDraw.ImageDraw, q: int, r: int,
                  color: tuple, ox: float, oy: float) -> None:
        verts = hex_vertices(q, r, self.hex_size, ox, oy)
        draw.polygon(verts, fill=color)

    def render(self, encoded: dict) -> tuple:
        """
        Args:
            encoded: DQREncoderV3.encode() 반환값

        Returns:
            (PIL.Image.Image, render_info dict)
            render_info: {'hex_size', 'origin_x', 'origin_y'}
        """
        radius = encoded['radius']
        tensor = encoded['tensor']
        reserved = encoded['reserved']
        fcenters = encoded['finder_centers']

        hs = self.hex_size
        qz_px = self.quiet_cells * hs * math.sqrt(3)  # quiet zone 픽셀

        # 이미지 크기 계산 (바운딩 박스 + quiet zone)
        min_x, min_y, max_x, max_y = grid_bounding_box(radius, hs)
        img_w = int(max_x - min_x + 2 * qz_px) + 4
        img_h = int(max_y - min_y + 2 * qz_px) + 4

        # grid origin: 셀 (0,0)의 픽셀 좌표
        ox = -min_x + qz_px + 2
        oy = -min_y + qz_px + 2

        img = Image.new('RGB', (img_w, img_h), WHITE)
        draw = ImageDraw.Draw(img)

        # 파인더 셀 분류
        finder_ring = set()
        finder_core = set(fcenters)
        for cq, cr in fcenters:
            for nq, nr in [(cq+dq, cr+dr) for dq, dr in
                           [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]]:
                finder_ring.add((nq, nr))
        finder_ring -= finder_core

        # ── 데이터 셀 ────────────────────────────────────────────────
        for q, r in build_hex_grid(radius):
            if (q, r) in reserved:
                continue
            c1, c2, c3 = tensor.get((q, r), (0, 0, 0))
            color = symbol_to_rgb(c1, c2, c3)
            self._draw_hex(draw, q, r, color, ox, oy)

        # ── 파인더 패턴: 이웃(흰) → 중심(검) 순으로 그려야 겹침 올바름 ──
        for q, r in finder_ring:
            self._draw_hex(draw, q, r, WHITE, ox, oy)
        for q, r in finder_core:
            self._draw_hex(draw, q, r, BLACK, ox, oy)

        render_info = {'hex_size': hs, 'origin_x': ox, 'origin_y': oy}
        return img, render_info
