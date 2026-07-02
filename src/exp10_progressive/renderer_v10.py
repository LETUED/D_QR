"""
D_QR Phase 10 렌더러 — dual-grain 단색 (exp10)

셀 분할 기하는 v9와 동일(삼각형 6/12분할)이므로 _draw_cell·_draw_finder를
그대로 상속 재사용. render만 패리티에 따라 셀별 비트수를 분기.
"""
import math
from PIL import Image, ImageDraw

from src.core.hex_grid import build_hex_grid, grid_bounding_box
from src.exp9_soft.renderer_v9 import DQRRendererV9
from src.exp10_progressive.encoder_v10 import GRAINS


class DQRRendererV10(DQRRendererV9):
    """
    D_QR Phase 10 렌더러.

    사용 예:
        ren = DQRRendererV10(hex_size=32)
        img, info = ren.render(encoded)   # 셀 패리티로 coarse/fine 자동 분기
    """

    def render(self, encoded: dict) -> tuple:
        radius = encoded['radius']
        tensor = encoded['tensor']
        fcenters = encoded['finder_centers']
        reserved = encoded['reserved']
        bpc_c = GRAINS['coarse']['bits_per_cell']
        bpc_f = GRAINS['fine']['bits_per_cell']

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
            if (q, r) in reserved:
                continue
            bpc = bpc_c if (q + r) % 2 == 0 else bpc_f
            self._draw_cell(draw, q, r, tensor.get((q, r), 0), bpc, ox, oy)

        for cq, cr in fcenters:
            self._draw_finder(draw, cq, cr, ox, oy)

        return img, {'hex_size': hs, 'origin_x': ox, 'origin_y': oy}
