"""
D_QR Phase 13 렌더러 — 자기기술 코드 (exp13)

exp9 렌더러와의 차이:
  · 포맷 링 셀: 셀 전체 채움(1)/비움(0)으로 렌더 (삼각형 분할 없음 — 강건)
  · 차등 파인더(NE): 중심 + 이웃 2셀(E·SE) 채움 → 방향 모호성 제거
  · quiet_cells 고정(2): 기하 자기추정 규약의 일부 — 파라미터 제공 안 함
"""
import math

from PIL import Image, ImageDraw

from src.core.hex_grid import (HEX_DIRECTIONS, build_hex_grid,
                               grid_bounding_box, hex_vertices)
from src.exp9_soft.renderer_v9 import DQRRendererV9
from src.exp13_selfdesc.layout import QUIET_CELLS, SPECIAL_DIRS, SPECIAL_FINDER


class DQRRendererV13(DQRRendererV9):
    """
    D_QR Phase 13 렌더러.

    사용 예:
        ren = DQRRendererV13(hex_size=32)
        img, info = ren.render(encoded)   # encoded = DQREncoderV13.encode(...)
    """

    def __init__(self, hex_size: int = 32,
                 fg_color: tuple = (0, 0, 0),
                 bg_color: tuple = (255, 255, 255)):
        # quiet_cells는 기하 자기추정(estimate_geometry)의 전제 — 고정
        super().__init__(hex_size, fg_color, bg_color,
                         quiet_cells=QUIET_CELLS)

    def _draw_finder_v13(self, draw, cq, cr, special, ox, oy):
        hs = self.hex_size
        for i, (dq, dr) in enumerate(HEX_DIRECTIONS):
            fill = self.fg if (special and i in SPECIAL_DIRS) else self.bg
            draw.polygon(hex_vertices(cq + dq, cr + dr, hs, ox, oy), fill=fill)
        draw.polygon(hex_vertices(cq, cr, hs, ox, oy), fill=self.fg)

    def render(self, encoded: dict) -> tuple:
        radius = encoded['radius']
        tensor = encoded['tensor']
        reserved = encoded['reserved']
        bpc = encoded['bits_per_cell']

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
                self._draw_cell(draw, q, r, tensor.get((q, r), 0), bpc, ox, oy)

        for (q, r), bit in encoded['format_cell_bits'].items():
            draw.polygon(hex_vertices(q, r, hs, ox, oy),
                         fill=self.fg if bit else self.bg)

        for i, (cq, cr) in enumerate(encoded['finder_centers']):
            self._draw_finder_v13(draw, cq, cr, i == SPECIAL_FINDER, ox, oy)

        return img, {'hex_size': hs, 'origin_x': ox, 'origin_y': oy}
