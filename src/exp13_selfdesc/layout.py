"""
exp13 자기기술(self-describing) 레이아웃 — 파인더 비대칭화·포맷 링·기하 자기추정.

세 구성요소가 합쳐져 "PNG 메타데이터 없이 이미지만으로 디코딩"을 만든다:
  ① 파인더 비대칭화 — 방향(회전 3 × 거울상 2 = 6가지) 모호성 제거
  ② 포맷 링        — 프로파일·ECC 레벨·반지름 인덱스 등 디코딩 파라미터 내장
  ③ 기하 자기추정  — 렌더링 규약(고정 quiet zone 비율 QUIET_CELLS=2)으로
                     이미지 크기에서 hex_size·origin을 역산
                     (CV 검출 파이프라인 불요, 결정론적)

[파인더 대칭성 — src/core/hex_grid.finder_centers 주석 오류 정정]
세 중심 {(r0,-r0), (-r0,0), (0,r0)} (r0=radius-1)는 '비대칭 배치'가 아니다:
  · 120° 회전 ρ(q,r)=(-q-r, q)로 서로 순환 (NE→SE→W→NE)
  · 반사 (q,r)→(-r,-q)에 집합 불변
즉 대칭군 D3 — 동일 형상 파인더 3개로는 회전 0°/120°/240° × 거울상의
6가지 방향 가설이 구분 불가능하다.

exp13 해법 = 형상 차등 (4번째 마크 대신 기존 마크 1개 변형 — 셀 예산 절약):
NE 파인더(SPECIAL_FINDER=0)만 중심 셀에 더해 이웃 2셀(E·SE 방향)을 채운다.
  · 회전 가설: 차등 파인더가 어느 꼭짓점에 있는지로 즉시 구분
  · 거울상 가설: NE 중심을 고정하는 유일한 반사 (q,r)→(-r,-q)가 채움
    이웃쌍 {E,SE}를 {NW,W}로 보내므로 패턴 불일치 → 구분
6가지 가설 전수 검증: tests/test_exp13_selfdesc.py 테스트 ①.
"""
import math

from src.core.bitutils import _max_data_for_budget
from src.core.hex_grid import (HEX_DIRECTIONS, axial_to_pixel, build_hex_grid,
                               finder_cell_set, finder_centers)
from src.exp9_soft.encoder_v9 import GRID_RADII

EXP_TAG = "exp13"

# 렌더링 규약 — 기하 자기추정의 전제이므로 변경 금지 (renderer_v13이 강제)
QUIET_CELLS = 2

SPECIAL_FINDER = 0        # 형상 차등 파인더 = finder_centers()[0] (NE)
SPECIAL_DIRS = (0, 1)     # 추가 채움 이웃 = HEX_DIRECTIONS[0,1] (E, SE)

FORMAT_BITS = 18          # 포맷 워드 비트 수 (format_ring.py 참조)
FORMAT_RING_DISTS = (2, 4)  # 포맷 셀 배치: 파인더 중심에서 hex 거리 2~4

_SQRT3 = math.sqrt(3)


# ──────────────────────────────────────────────────────────────────
# axial 변환 (방향 가설 6가지의 격자 좌표 대응 — 단위 테스트용)
# ──────────────────────────────────────────────────────────────────

def axial_rot120(q: int, r: int) -> tuple:
    """120° 회전 ρ: cube 좌표 (x,y,z)→(-z,-x,-y)의 axial 표현."""
    return (-q - r, q)


def axial_mirror(q: int, r: int) -> tuple:
    """픽셀 x축 거울상의 axial 대응: x→-x, y 유지 ⇔ (q,r)→(-q-r, r)."""
    return (-q - r, r)


def axial_transform(q: int, r: int, rot_k: int, mirrored: bool) -> tuple:
    """거울상 먼저, 회전 나중 — transform_points(픽셀 변환)와 동일 순서."""
    if mirrored:
        q, r = axial_mirror(q, r)
    for _ in range(rot_k % 3):
        q, r = axial_rot120(q, r)
    return (q, r)


def hex_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    dq, dr = q1 - q2, r1 - r2
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


# ──────────────────────────────────────────────────────────────────
# ① 파인더 비대칭화 — 기대 패턴 (방향 판별의 기준)
# ──────────────────────────────────────────────────────────────────

def expected_finder_pattern(radius: int) -> list:
    """
    파인더 영역 21셀(3마크 × 7셀)의 기대 채움 패턴.

    Returns: [(q, r, bit), ...]
      bit 1 = 채움(어두움): 세 중심 + 차등 파인더의 E·SE 이웃
      bit 0 = 비움(밝음)  : 나머지 이웃 전부
    """
    pattern = []
    for i, (cq, cr) in enumerate(finder_centers(radius)):
        pattern.append((cq, cr, 1))
        for d, (dq, dr) in enumerate(HEX_DIRECTIONS):
            bit = 1 if (i == SPECIAL_FINDER and d in SPECIAL_DIRS) else 0
            pattern.append((cq + dq, cr + dr, bit))
    return pattern


# ──────────────────────────────────────────────────────────────────
# ② 포맷 링 — 셀 배치 (3중 반복: 파인더당 1사본 18셀)
# ──────────────────────────────────────────────────────────────────

def format_cells_per_finder(radius: int) -> list:
    """
    파인더별 포맷 셀 18개 × 3사본.

    배치: 각 파인더 중심에서 hex 거리 2~4의 격자 내 셀을
    (거리, 중심 기준 각도) 순으로 정렬해 앞 18개.
    파인더가 격자 꼭짓점(r0=radius-1)에 있어 거리 2~3만으로는 격자 내
    셀이 18개에 못 미침(radius=8에서 16개) → 거리 4까지 확장.
    결정론적이며 인코더/디코더가 같은 함수를 공유한다.

    Returns: [cells_NE(18), cells_W(18), cells_SE(18)]
    """
    fcells = set()
    for c in finder_centers(radius):
        fcells |= finder_cell_set(*c)

    d_lo, d_hi = FORMAT_RING_DISTS
    grid = build_hex_grid(radius)
    out, used = [], set()
    for cq, cr in finder_centers(radius):
        cands = []
        for q, r in grid:
            if (q, r) in fcells or (q, r) in used:
                continue
            d = hex_distance(q, r, cq, cr)
            if not (d_lo <= d <= d_hi):
                continue
            # 각도는 hex_size=1 단위 기하만 사용 — 렌더 픽셀 크기와 무관
            px, py = axial_to_pixel(q - cq, r - cr, 1.0)
            cands.append((d, math.atan2(py, px), q, r))
        cands.sort()
        if len(cands) < FORMAT_BITS:
            raise ValueError(f"radius={radius}: 포맷 셀 후보 부족 "
                             f"({len(cands)} < {FORMAT_BITS})")
        cells = [(q, r) for _, _, q, r in cands[:FORMAT_BITS]]
        used |= set(cells)
        out.append(cells)
    return out


def build_reserved_v13(radius: int) -> frozenset:
    """exp13 예약 셀 = 파인더 21셀 + 포맷 54셀."""
    reserved = set()
    for c in finder_centers(radius):
        reserved |= finder_cell_set(*c)
    for cells in format_cells_per_finder(radius):
        reserved |= set(cells)
    return frozenset(reserved)


def data_cells_v13(radius: int) -> list:
    """데이터 셀 목록 (q-major 정렬 — v9와 동일 순서 규약)."""
    reserved = build_reserved_v13(radius)
    return [c for c in build_hex_grid(radius) if c not in reserved]


# ──────────────────────────────────────────────────────────────────
# 용량 (페이로드 = 4바이트 길이 프리픽스 + 데이터, 용량까지 0 패딩)
# ──────────────────────────────────────────────────────────────────

def message_capacity(radius: int, bits_per_cell: int, nsym: int) -> int:
    """RS 인코딩 전 메시지(프리픽스+데이터+패딩)의 최대 바이트 수."""
    total_bytes = (len(data_cells_v13(radius)) * bits_per_cell) // 8
    return _max_data_for_budget(total_bytes, nsym, 255 - nsym)


def codeword_length(msg_len: int, nsym: int) -> int:
    """reedsolo 청킹 규칙 그대로의 코드워드 길이 (디코더가 radius에서 유도)."""
    dpb = 255 - nsym
    chunks = (msg_len + dpb - 1) // dpb
    return msg_len + chunks * nsym


# ──────────────────────────────────────────────────────────────────
# ③ 기하 자기추정 — 렌더링 규약 역산
# ──────────────────────────────────────────────────────────────────
#
# renderer_v9/v13 규약 (quiet_cells=2 고정):
#   max_x = (√3·R + 1)·hs,  max_y = (1.5·R + 1)·hs   (격자 점대칭 → min=-max)
#   qz    = 2·√3·hs
#   img_w = int(2·max_x + 2·qz) + 4,  img_h = int(2·max_y + 2·qz) + 4
#   origin = (max_x + qz + 2, max_y + qz + 2)  — 격자 중심 = 이미지 중심
#
# W,H 두 식에 미지수 (R, hs) 둘 → R 후보(GRID_RADII)를 순회하며 W에서 hs를
# 역산하고 H 일치로 검증. int() 절단 오차 분석: |H예측-H실측| ≤ 1 + hc/wc < 2.

def _w_coef(radius: int) -> float:
    return 2.0 * (_SQRT3 * radius + 1 + QUIET_CELLS * _SQRT3)


def _h_coef(radius: int) -> float:
    return 2.0 * (1.5 * radius + 1 + QUIET_CELLS * _SQRT3)


def estimate_geometry(img_w: int, img_h: int,
                      radii: list = None, tol: float = 2.5) -> list:
    """
    이미지 크기 → 기하 후보 목록 [{radius, hex_size, origin_x, origin_y}].

    GRID_RADII 간격(8,12,18,...)에서는 사실상 단일 후보가 나오지만,
    복수 후보도 호출 측(decoder_v13)이 포맷 링 반지름 인덱스 대조로 걸러낸다.
    """
    cands = []
    for radius in (GRID_RADII if radii is None else radii):
        hs = (img_w - 4) / _w_coef(radius)
        if hs < 4:          # 셀이 너무 작으면 (서브)삼각형 표결 자체가 불가
            continue
        if abs(hs * _h_coef(radius) + 4 - img_h) > tol:
            continue
        cands.append({
            'radius': radius,
            'hex_size': hs,
            'origin_x': hs * (_SQRT3 * radius + 1 + QUIET_CELLS * _SQRT3) + 2,
            'origin_y': hs * (1.5 * radius + 1 + QUIET_CELLS * _SQRT3) + 2,
        })
    return cands


# ──────────────────────────────────────────────────────────────────
# 픽셀 변환 (방향 가설 적용) · 셀 단위 샘플점
# ──────────────────────────────────────────────────────────────────

def transform_points(xs, ys, ox: float, oy: float,
                     rot_k: int = 0, mirrored: bool = False) -> tuple:
    """
    격자 중심 (ox,oy) 기준 픽셀 변환: 거울상(x 반전) 먼저, 120°·rot_k 회전 나중.

    가설 (rot_k, mirrored) 6가지가 D3 코드 배치의 가능한 외관 전부를 덮는다.
    삼각형 인덱스 순열을 따로 다루지 않고 표준 샘플점 좌표를 통째로 변환
    → 회전·거울상에서 (서브)삼각형 판독이 자동으로 정합된다.
    """
    import numpy as np
    x = np.asarray(xs, dtype=float) - ox
    y = np.asarray(ys, dtype=float) - oy
    if mirrored:
        x = -x
    k = rot_k % 3
    if k:
        ang = 2.0 * math.pi * k / 3.0
        c, s = math.cos(ang), math.sin(ang)
        x, y = c * x - s * y, s * x + c * y
    return x + ox, y + oy


def hex_cell_sample_points(q: int, r: int, hex_size: float,
                           ox: float, oy: float, t: float = 0.5) -> list:
    """
    셀 전체 채움/비움 판독용 7점 (중심 + 변중점 방향 6점, 반지름 t·hs).

    pointy-top 육각형의 내접원 반지름은 (√3/2)·hs ≈ 0.87·hs → t=0.5면
    1px 수준의 재표본 오차·기하 추정 오차에도 확실히 셀 내부에 머문다.
    파인더·포맷 셀 판독은 이 7점의 다수결 — 이산 표 + 전역 임계 (FTO 준수).
    """
    cx, cy = axial_to_pixel(q, r, hex_size, ox, oy)
    pts = [(cx, cy)]
    for i in range(6):
        ang = math.radians(60 * i)
        pts.append((cx + t * hex_size * math.cos(ang),
                    cy + t * hex_size * math.sin(ang)))
    return pts
