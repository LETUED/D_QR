"""
D_QR Phase 1 렌더러
텐서 T (H, W, 3) → PIL Image

렌더링 규칙:
  - 데이터 모듈: (c1, c2, c3) → HSV → RGB 색상 블록
  - 파인더 패턴: 흑백 (QR과 동일)
  - 타이밍 패턴: 흑백 교대 (파인더 사이)
  - 조용한 영역(quiet zone): 흰색 4모듈 테두리
"""
import numpy as np
from PIL import Image, ImageDraw

from src.core.constants import (
    FINDER_PATTERN, FINDER_SIZE, SEPARATOR,
    symbol_to_rgb,
)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
QUIET_ZONE = 4  # 테두리 흰색 모듈 수


class DQRRenderer:
    """
    D_QR 이미지 렌더러.

    사용 예:
        renderer = DQRRenderer(module_px=20)
        img = renderer.render(encoded)
        img.save("dqr.png")
    """

    def __init__(self, module_px: int = 20, quiet_zone: int = QUIET_ZONE):
        """
        Args:
            module_px  : 모듈 하나의 픽셀 크기 (정사각형)
            quiet_zone : 코드 주변 흰색 테두리 모듈 수
        """
        self.module_px = module_px
        self.quiet_zone = quiet_zone

    def _draw_module(self, draw: ImageDraw.ImageDraw,
                     row: int, col: int, color: tuple, offset: int) -> None:
        """단일 모듈을 module_px × module_px 정사각형으로 그린다."""
        px = self.module_px
        x0 = (col + offset) * px
        y0 = (row + offset) * px
        draw.rectangle([x0, y0, x0 + px - 1, y0 + px - 1], fill=color)

    def _draw_finder(self, draw: ImageDraw.ImageDraw,
                     top_row: int, left_col: int, offset: int) -> None:
        """7×7 파인더 패턴을 지정 위치에 그린다."""
        for r in range(FINDER_SIZE):
            for c in range(FINDER_SIZE):
                color = BLACK if FINDER_PATTERN[r][c] else WHITE
                self._draw_module(draw, top_row + r, left_col + c, color, offset)

    def render(self, encoded: dict) -> Image.Image:
        """
        인코더 출력 dict를 받아 PIL Image로 렌더링.

        Args:
            encoded: DQREncoder.encode()의 반환값

        Returns:
            PIL.Image.Image (RGB 모드)
        """
        tensor: np.ndarray = encoded['tensor']
        reserved: np.ndarray = encoded['reserved']
        S: int = encoded['grid_size']

        px = self.module_px
        qz = self.quiet_zone
        fp = FINDER_SIZE + SEPARATOR  # 8

        # 이미지 크기: 그리드 + quiet zone 양쪽
        img_size = (S + 2 * qz) * px
        img = Image.new('RGB', (img_size, img_size), WHITE)
        draw = ImageDraw.Draw(img)

        # ── 파인더 패턴 3개 ────────────────────────────────────────
        self._draw_finder(draw, 0, 0, qz)            # 좌상
        self._draw_finder(draw, 0, S - FINDER_SIZE, qz)   # 우상
        self._draw_finder(draw, S - FINDER_SIZE, 0, qz)   # 좌하

        # ── 타이밍 패턴 ────────────────────────────────────────────
        for i in range(fp - 1, S - fp + 1):
            color = BLACK if i % 2 == 0 else WHITE
            self._draw_module(draw, 6, i, color, qz)  # 수평
            self._draw_module(draw, i, 6, color, qz)  # 수직

        # ── 데이터 모듈 ────────────────────────────────────────────
        for row in range(S):
            for col in range(S):
                if reserved[row, col]:
                    continue
                c1, c2, c3 = int(tensor[row, col, 0]), int(tensor[row, col, 1]), int(tensor[row, col, 2])
                # tensor가 255,255,255면 아직 데이터 없는 모듈 → 흰색
                if (c1, c2, c3) == (255, 255, 255):
                    color = WHITE
                else:
                    color = symbol_to_rgb(c1, c2, c3)
                self._draw_module(draw, row, col, color, qz)

        return img
