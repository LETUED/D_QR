"""
D_QR Phase 1 상수 정의
- 색상 3채널: c1(색조 8레벨), c2(채도 4레벨), c3(명도 4레벨)
- 모듈당 7비트 (3+2+2)
"""
import colorsys
from typing import Tuple

# ── 채널 정의 ────────────────────────────────────────────────────────
C1_LEVELS = 8   # Hue: 8등분 (0°, 45°, 90°, ..., 315°)
C2_LEVELS = 4   # Saturation: 4레벨
C3_LEVELS = 4   # Value(명도): 4레벨

BITS_C1 = 3     # log2(8)
BITS_C2 = 2     # log2(4)
BITS_C3 = 2     # log2(4)
BITS_PER_MODULE = BITS_C1 + BITS_C2 + BITS_C3  # 7

# ── HSV 팔레트 ───────────────────────────────────────────────────────
# 색조: 360° 를 8등분, 0-1 범위
C1_HUES = [i / C1_LEVELS for i in range(C1_LEVELS)]
# 채도: 0.5~1.0 사이 4단계 (너무 연한 색 제외)
C2_SATS = [0.50, 0.67, 0.83, 1.00]
# 명도: 0.5~1.0 사이 4단계 (너무 어두운 색 제외)
C3_VALS = [0.50, 0.67, 0.83, 1.00]


def _hue_dist(h1: float, h2: float) -> float:
    """색조는 원형(circular) → 0.0과 1.0은 같은 색."""
    d = abs(h1 - h2)
    return min(d, 1.0 - d)


def symbol_to_rgb(c1: int, c2: int, c3: int) -> Tuple[int, int, int]:
    """채널 심볼 → RGB (0-255)."""
    h = C1_HUES[c1]
    s = C2_SATS[c2]
    v = C3_VALS[c3]
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def rgb_to_nearest_symbol(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """RGB 픽셀 → 가장 가까운 (c1, c2, c3) 팔레트 심볼."""
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    c1 = min(range(C1_LEVELS), key=lambda i: _hue_dist(C1_HUES[i], h))
    c2 = min(range(C2_LEVELS), key=lambda i: abs(C2_SATS[i] - s))
    c3 = min(range(C3_LEVELS), key=lambda i: abs(C3_VALS[i] - v))
    return c1, c2, c3


# ── 파인더 패턴 (7×7, QR 동일) ──────────────────────────────────────
FINDER_PATTERN = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

FINDER_SIZE = 7
SEPARATOR = 1   # 파인더 주변 1모듈 여백

# ── 그리드 크기 설정 ─────────────────────────────────────────────────
# QR Version과 대응하는 그리드 크기
GRID_SIZES = [21, 37, 57, 97, 177]  # V1, V5, V10, V20, V40 기준
