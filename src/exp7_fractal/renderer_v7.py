"""
D_QR Phase 7 렌더러 — 프랙탈 중첩 헥사곤 (exp7)

시각 구조:
  외부 헥사곤: 연한 회색 테두리 (구조 가시화)
  내부 미니 헥사곤 7개: 각각 서브삼각형 6개로 인코딩
  외부 파인더: 검정 솔리드 헥사곤
"""
import math
from PIL import Image, ImageDraw

from src.core.hex_grid import (build_hex_grid, build_reserved, finder_centers,
                        hex_vertices, axial_to_pixel, grid_bounding_box,
                        hex_primary_triangles)
from src.exp7_fractal.encoder_v7 import INNER_CELLS, BITS_PER_INNER

QUIET_CELLS = 2


class DQRRendererV7:
    """
    D_QR Phase 7 프랙탈 렌더러.

    사용 예:
        ren = DQRRendererV7()
        img, info = ren.render(encoded)
    """

    def __init__(self,
                 fg_color: tuple = (0, 0, 0),
                 bg_color: tuple = (255, 255, 255),
                 grid_color: tuple = (200, 200, 200),
                 quiet_cells: int = QUIET_CELLS):
        self.fg = fg_color
        self.bg = bg_color
        self.grid = grid_color  # 외부 헥사곤 테두리 색
        self.quiet_cells = quiet_cells

    def _draw_inner_cell(self, draw, q2, r2, val, inner_hs, outer_cx, outer_cy):
        """내부 셀 (q2,r2)을 6-bit val 값으로 그린다."""
        inner_verts = hex_vertices(q2, r2, inner_hs, outer_cx, outer_cy)
        draw.polygon(inner_verts, fill=self.bg)

        pri_tris = hex_primary_triangles(q2, r2, inner_hs, outer_cx, outer_cy)
        for idx, (tri_verts, _) in enumerate(pri_tris):
            bit = (val >> (BITS_PER_INNER - 1 - idx)) & 1
            if bit:
                draw.polygon(tri_verts, fill=self.fg)

    def _draw_finder(self, draw, cq, cr, outer_hs, ox, oy):
        dirs = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for dq, dr in dirs:
            draw.polygon(hex_vertices(cq+dq, cr+dr, outer_hs, ox, oy), fill=self.bg)
        draw.polygon(hex_vertices(cq, cr, outer_hs, ox, oy), fill=self.fg)

    def render(self, encoded: dict) -> tuple:
        outer_radius = encoded['outer_radius']
        outer_hs = encoded['outer_hex_size']
        inner_hs = encoded['inner_hex_size']
        tensor = encoded['tensor']
        outer_cells = encoded['outer_cells']
        reserved = encoded['reserved']
        fcenters = encoded['finder_centers']

        qz_px = self.quiet_cells * outer_hs * math.sqrt(3)
        min_x, min_y, max_x, max_y = grid_bounding_box(outer_radius, outer_hs)
        img_w = int(max_x - min_x + 2 * qz_px) + 4
        img_h = int(max_y - min_y + 2 * qz_px) + 4
        ox = -min_x + qz_px + 2
        oy = -min_y + qz_px + 2

        img = Image.new('RGB', (img_w, img_h), self.bg)
        draw = ImageDraw.Draw(img)

        # ── 외부 파인더 패턴 ────────────────────────────────────────
        for cq, cr in fcenters:
            self._draw_finder(draw, cq, cr, outer_hs, ox, oy)

        # ── 외부 데이터 셀 → 내부 미니 격자 ────────────────────────
        for q1, r1 in outer_cells:
            outer_cx, outer_cy = axial_to_pixel(q1, r1, outer_hs, ox, oy)

            # 외부 헥사곤 테두리 (구조 가시화)
            outer_verts = hex_vertices(q1, r1, outer_hs, ox, oy)
            draw.polygon(outer_verts, fill=(245, 245, 245), outline=self.grid)

            # 7개 내부 셀
            for q2, r2 in INNER_CELLS:
                val = tensor.get((q1, r1, q2, r2), 0)
                self._draw_inner_cell(draw, q2, r2, val, inner_hs, outer_cx, outer_cy)

        render_info = {
            'outer_hex_size': outer_hs,
            'inner_hex_size': inner_hs,
            'origin_x': ox,
            'origin_y': oy,
        }
        return img, render_info
