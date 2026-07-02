"""
D_QR Phase 5 렌더러 — 3D 부조 시각화 (exp5)

출력 2종:
  1. 2D PNG  : 높이 레벨을 명도로 표현한 top-down 뷰
               (실제 제조물의 카메라 촬영과 유사)
  2. 3D PNG  : matplotlib 3D 바 차트 — 셀 높이가 z-axis로 표현됨
               인터랙티브 탐색 가능한 구조 직관적 파악용
"""
import math
import numpy as np
from PIL import Image, ImageDraw

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from src.core.hex_grid import (build_hex_grid, build_reserved, finder_centers,
                        hex_vertices, axial_to_pixel, grid_bounding_box)

QUIET_CELLS = 2
BITS_TRI = 6
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 높이 레벨별 명도 (2D top-view용): 0=밝음, 3=어두움
HEIGHT_GRAY = [255, 200, 130, 60]


class DQRRendererV5:
    """
    D_QR Phase 5 렌더러.

    사용 예:
        ren = DQRRendererV5(hex_size=28)
        img_2d, img_3d, info = ren.render(encoded)
        img_2d.save("exp5_dqr_flat.png")
        img_3d.save("exp5_dqr_3d.png")  # None if matplotlib unavailable
    """

    def __init__(self, hex_size: int = 28,
                 fg_color: tuple = BLACK,
                 bg_color: tuple = WHITE,
                 quiet_cells: int = QUIET_CELLS):
        self.hex_size = hex_size
        self.fg = fg_color
        self.bg = bg_color
        self.quiet_cells = quiet_cells

    # ── 2D 렌더링 (exp4와 동일, 높이를 배경 명도로 표현) ────────────

    def _draw_cell_2d(self, draw, q, r, tri_val, z_level, ox, oy):
        hs = self.hex_size
        verts = hex_vertices(q, r, hs, ox, oy)
        cx, cy = axial_to_pixel(q, r, hs, ox, oy)

        # 배경: z-level에 따라 명도 변화 (높을수록 어두움)
        g = HEIGHT_GRAY[z_level]
        draw.polygon(verts, fill=(g, g, g))

        # 삼각형 패턴: fg(검정)으로
        for i in range(BITS_TRI):
            bit = (tri_val >> (BITS_TRI - 1 - i)) & 1
            if bit:
                v_a = verts[i]
                v_b = verts[(i + 1) % 6]
                draw.polygon([(cx, cy), v_a, v_b], fill=self.fg)

    def _draw_finder_2d(self, draw, cq, cr, ox, oy):
        hs = self.hex_size
        dirs = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for dq, dr in dirs:
            draw.polygon(hex_vertices(cq+dq, cr+dr, hs, ox, oy), fill=self.bg)
        draw.polygon(hex_vertices(cq, cr, hs, ox, oy), fill=self.fg)

    def _render_2d(self, encoded: dict) -> tuple:
        radius = encoded['radius']
        triangles = encoded['triangles']
        heights = encoded['heights']
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
                tri = triangles.get((q, r), 0)
                z = heights.get((q, r), 0)
                self._draw_cell_2d(draw, q, r, tri, z, ox, oy)

        for cq, cr in fcenters:
            self._draw_finder_2d(draw, cq, cr, ox, oy)

        return img, {'hex_size': hs, 'origin_x': ox, 'origin_y': oy}

    # ── 3D 렌더링 (matplotlib 3D 바 차트) ────────────────────────────

    def _render_3d(self, encoded: dict):
        if not HAS_MPL:
            return None

        radius = encoded['radius']
        heights = encoded['heights']
        triangles = encoded['triangles']
        reserved = encoded['reserved']

        hs = self.hex_size * 0.5  # 3D 뷰에서는 작게 표시

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        xs, ys, zs, dxs, dys, dzs, colors = [], [], [], [], [], [], []
        dx = hs * math.sqrt(3) * 0.85
        dy = hs * 1.5 * 0.85

        for q, r in build_hex_grid(radius):
            if (q, r) in reserved:
                continue
            cx, cy = axial_to_pixel(q, r, hs)
            z_level = heights.get((q, r), 0)
            tri_val = triangles.get((q, r), 0)

            xs.append(cx - dx/2)
            ys.append(cy - dy/2)
            zs.append(0)
            dxs.append(dx)
            dys.append(dy)
            dzs.append(max(z_level, 0.05))  # 최소 두께

            # 높이 레벨로 색상: 파란색(낮음) → 빨간색(높음)
            t = z_level / 3.0
            colors.append((t, 0.2, 1-t, 0.85))

        ax.bar3d(xs, ys, zs, dxs, dys, dzs, color=colors, shade=True)

        # 파인더 마커 표시 (별도 색)
        for cq, cr in encoded['finder_centers']:
            cx, cy = axial_to_pixel(cq, cr, hs)
            ax.bar3d(cx-dx/2, cy-dy/2, 0, dx, dy, 0.5,
                     color='black', alpha=0.9, shade=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (height level)')
        ax.set_title(f'D_QR Phase 5 — 3D Relief (radius={radius})\n'
                     f'Blue=low z, Red=high z  |  exp5')
        ax.view_init(elev=35, azim=-60)

        fig.tight_layout()

        # PNG로 저장 후 PIL Image로 변환
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def render(self, encoded: dict) -> tuple:
        """
        Returns:
            (img_2d, img_3d, render_info)
            img_2d: 2D top-down 뷰 (높이를 배경 명도로 표현)
            img_3d: matplotlib 3D 바 차트 (None if matplotlib 없음)
            render_info: {'hex_size', 'origin_x', 'origin_y'}
        """
        img_2d, render_info = self._render_2d(encoded)
        img_3d = self._render_3d(encoded)
        return img_2d, img_3d, render_info
