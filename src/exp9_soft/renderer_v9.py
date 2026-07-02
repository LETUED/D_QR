"""
D_QR Phase 9 렌더러 — 가변 프로파일 단색 (exp9)

프로파일 A: primary 삼각형 6개/셀
프로파일 B: 서브삼각형 12개/셀
"""
import math
from PIL import Image, ImageDraw

from src.core.hex_grid import (build_hex_grid, hex_vertices, grid_bounding_box,
                        hex_primary_triangles, hex_sub_triangles)

QUIET_CELLS = 2


class DQRRendererV9:
    """
    D_QR Phase 9 렌더러.

    사용 예:
        ren = DQRRendererV9(hex_size=32)
        img, info = ren.render(encoded)   # encoded['profile']에 따라 자동 분기
    """

    def __init__(self, hex_size: int = 32,
                 fg_color: tuple = (0, 0, 0),
                 bg_color: tuple = (255, 255, 255),
                 quiet_cells: int = QUIET_CELLS):
        self.hex_size = hex_size
        self.fg = fg_color
        self.bg = bg_color
        self.quiet_cells = quiet_cells

    def _draw_cell(self, draw, q, r, value, bits_per_cell, ox, oy):
        hs = self.hex_size
        draw.polygon(hex_vertices(q, r, hs, ox, oy), fill=self.bg)

        if bits_per_cell == 6:
            tris = hex_primary_triangles(q, r, hs, ox, oy)
        else:
            tris = hex_sub_triangles(q, r, hs, ox, oy)

        for idx, (tri_verts, _centroid) in enumerate(tris):
            bit = (value >> (bits_per_cell - 1 - idx)) & 1
            if bit:
                draw.polygon(tri_verts, fill=self.fg)

    def _draw_finder(self, draw, cq, cr, ox, oy):
        hs = self.hex_size
        dirs = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for dq, dr in dirs:
            draw.polygon(hex_vertices(cq+dq, cr+dr, hs, ox, oy), fill=self.bg)
        draw.polygon(hex_vertices(cq, cr, hs, ox, oy), fill=self.fg)

    def render(self, encoded: dict) -> tuple:
        radius = encoded['radius']
        tensor = encoded['tensor']
        fcenters = encoded['finder_centers']
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

        for cq, cr in fcenters:
            self._draw_finder(draw, cq, cr, ox, oy)

        return img, {'hex_size': hs, 'origin_x': ox, 'origin_y': oy}
