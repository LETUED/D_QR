"""
D_QR Phase 4 렌더러 — 기하 서브삼각형 단색 렌더링 (exp4)

각 육각형 셀을 6개 정삼각형으로 분할,
filled 삼각형 → fg_color, empty → bg_color.
색상은 완전히 파라미터화 — 흑백·홀로그램·레이저 모두 동일 코드.
"""
import math
from PIL import Image, ImageDraw

from src.core.hex_grid import (build_hex_grid, build_reserved, finder_centers,
                        hex_vertices, axial_to_pixel, grid_bounding_box)

QUIET_CELLS = 2
BITS_PER_CELL = 6


class DQRRendererV4:
    """
    D_QR Phase 4 이미지 렌더러 — 단색 기하 패턴.

    fg_color / bg_color 파라미터로 임의 색상 대응:
      - 흑백 인쇄: fg=(0,0,0), bg=(255,255,255)  (기본값)
      - 홀로그램:  fg=(0,255,0), bg=(0,0,0)
      - 레이저:    fg=(255,255,255), bg=(0,0,0)

    사용 예:
        ren = DQRRendererV4(hex_size=28)
        img, info = ren.render(encoded)
    """

    def __init__(self, hex_size: int = 28,
                 fg_color: tuple = (0, 0, 0),
                 bg_color: tuple = (255, 255, 255),
                 quiet_cells: int = QUIET_CELLS):
        self.hex_size = hex_size
        self.fg = fg_color
        self.bg = bg_color
        self.quiet_cells = quiet_cells

    def _draw_cell(self, draw: ImageDraw.ImageDraw,
                   q: int, r: int, value: int,
                   ox: float, oy: float) -> None:
        """
        육각형 셀을 6개 삼각형으로 그린다.
        value: 6비트 정수 (bit5=T0, bit0=T5)
               T_i = 중심과 꼭짓점[i], 꼭짓점[(i+1)%6]로 이루어진 삼각형
        """
        hs = self.hex_size
        verts = hex_vertices(q, r, hs, ox, oy)
        cx, cy = axial_to_pixel(q, r, hs, ox, oy)

        # 셀 배경
        draw.polygon(verts, fill=self.bg)

        # 6개 삼각형
        for i in range(BITS_PER_CELL):
            bit = (value >> (BITS_PER_CELL - 1 - i)) & 1
            if bit:
                v_a = verts[i]
                v_b = verts[(i + 1) % 6]
                draw.polygon([(cx, cy), v_a, v_b], fill=self.fg)

    def _draw_finder(self, draw: ImageDraw.ImageDraw,
                     cq: int, cr: int, ox: float, oy: float) -> None:
        """파인더 마커: 중심=fg (solid), 6이웃=bg."""
        hs = self.hex_size
        directions = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]

        # 이웃(bg) 먼저
        for dq, dr in directions:
            verts = hex_vertices(cq+dq, cr+dr, hs, ox, oy)
            draw.polygon(verts, fill=self.bg)

        # 중심(fg solid) 나중에 — solid hex (6개 삼각형 모두 filled)
        verts = hex_vertices(cq, cr, hs, ox, oy)
        draw.polygon(verts, fill=self.fg)

    def render(self, encoded: dict) -> tuple:
        """
        Returns:
            (PIL.Image.Image, render_info)
            render_info: {'hex_size', 'origin_x', 'origin_y',
                          'fg_color', 'bg_color'}
        """
        radius = encoded['radius']
        tensor = encoded['tensor']
        fcenters = encoded['finder_centers']

        hs = self.hex_size
        qz_px = self.quiet_cells * hs * math.sqrt(3)

        min_x, min_y, max_x, max_y = grid_bounding_box(radius, hs)
        img_w = int(max_x - min_x + 2 * qz_px) + 4
        img_h = int(max_y - min_y + 2 * qz_px) + 4

        ox = -min_x + qz_px + 2
        oy = -min_y + qz_px + 2

        img = Image.new('RGB', (img_w, img_h), self.bg)
        draw = ImageDraw.Draw(img)

        reserved = encoded['reserved']

        # ── 데이터 셀 ────────────────────────────────────────────────
        for q, r in build_hex_grid(radius):
            if (q, r) not in reserved:
                val = tensor.get((q, r), 0)
                self._draw_cell(draw, q, r, val, ox, oy)

        # ── 파인더 패턴 ──────────────────────────────────────────────
        for cq, cr in fcenters:
            self._draw_finder(draw, cq, cr, ox, oy)

        render_info = {
            'hex_size': hs,
            'origin_x': ox,
            'origin_y': oy,
            'fg_color': self.fg,
            'bg_color': self.bg,
        }
        return img, render_info
