"""
D_QR Phase 8 렌더러 — DLC 듀얼 스트림 (exp8)

Stream A 셀과 Stream B 셀은 시각적으로 동일하게 보임 (의도적):
  - 관찰자는 어느 셀이 어느 스트림인지 알 수 없음
  - 디코더만 인덱스 순서를 알고 두 스트림을 분리
  - 마치 하나의 단일 코드처럼 보이지만 내부에 두 차원이 공존
"""
import math
from PIL import Image, ImageDraw

from src.core.hex_grid import (build_hex_grid, build_reserved, finder_centers,
                        hex_vertices, axial_to_pixel, grid_bounding_box,
                        hex_sub_triangles)

QUIET_CELLS = 2
BITS_PER_CELL = 12


class DQRRendererV8:
    """exp8 렌더러. exp6와 거의 동일 — 셀 값만 다름."""

    def __init__(self, hex_size: int = 32,
                 fg_color: tuple = (0, 0, 0),
                 bg_color: tuple = (255, 255, 255),
                 quiet_cells: int = QUIET_CELLS):
        self.hex_size = hex_size
        self.fg = fg_color
        self.bg = bg_color
        self.quiet_cells = quiet_cells

    def _draw_cell(self, draw, q, r, value, ox, oy):
        hs = self.hex_size
        verts = hex_vertices(q, r, hs, ox, oy)
        draw.polygon(verts, fill=self.bg)
        sub_tris = hex_sub_triangles(q, r, hs, ox, oy)
        for idx, (tri_verts, _) in enumerate(sub_tris):
            bit = (value >> (BITS_PER_CELL - 1 - idx)) & 1
            if bit:
                draw.polygon(tri_verts, fill=self.fg)

    def _draw_finder(self, draw, cq, cr, ox, oy):
        hs = self.hex_size
        for dq, dr in [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]:
            draw.polygon(hex_vertices(cq+dq, cr+dr, hs, ox, oy), fill=self.bg)
        draw.polygon(hex_vertices(cq, cr, hs, ox, oy), fill=self.fg)

    def render(self, encoded: dict) -> tuple:
        radius = encoded['radius']
        tensor = encoded['tensor']
        fcenters = encoded['finder_centers']
        reserved = encoded['reserved']

        hs = self.hex_size
        qz_px = self.quiet_cells * hs * math.sqrt(3)
        min_x, min_y, max_x, max_y = grid_bounding_box(radius, hs)
        img_w = int(max_x - min_x + 2 * qz_px) + 4
        img_h = int(max_y - min_y + 2 * qz_px) + 4
        ox = -min_x + qz_px + 2
        oy = -min_y + qz_px + 2

        img = Image.new('RGB', (img_w, img_h), self.bg)
        draw = ImageDraw.Draw(img)

        for q, r in build_hex_grid(radius):
            if (q, r) not in reserved:
                val = tensor.get((q, r), 0)
                self._draw_cell(draw, q, r, val, ox, oy)

        for cq, cr in fcenters:
            self._draw_finder(draw, cq, cr, ox, oy)

        return img, {'hex_size': hs, 'origin_x': ox, 'origin_y': oy}
