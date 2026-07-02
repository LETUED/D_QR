"""
exp16 2·3단계 — 닮음 가설(2점 대응) + 꼭짓점 스윕 정밀화 → 3점 어파인

FTO: 본 모듈은 위치 추정 전용, 신뢰도·소거 판정과 무관.

좌표 공간:
  표준(canonical) 공간 = hex_size=1, origin=(0,0)의 레이아웃 픽셀 공간
  (src/core/hex_grid.axial_to_pixel 그대로). 추정 어파인 M(2×3)은
  표준 공간 → 이미지 공간 사상이며 회전·스케일·거울상(det<0)·전단을 흡수.
  완만 원근(기울기 ≤15°)은 어파인 근사로 흡수 — v0 범위.

[설계 변경 기록 — 클러스터 블롭 직접 탐지 폐기]
차등 파인더(NE) 3셀 클러스터는 채움 셀(E·SE)의 바깥 이웃이 포맷/데이터
셀이라 흰 분리가 보장되지 않음 → 데이터 영역과 한 연결 성분으로 합쳐져
형상 필터로 못 찾는다(실측). 반면 단일 파인더(W·SE)는 6이웃 전부 비움이
규약이라 항상 고립 솔리드 육각으로 검출된다. 따라서:
  · 측정점 2개(단일 파인더 무게중심 — 육각 대칭이라 무게중심=중심) +
    정삼각형 꼭짓점 구성(닮음 2점 대응의 고유·거울 2해)으로 NE를 구성
  · W↔SE 배정 2 × 거울 2 = 가설 4가지가 방향 모호성 전부를 덮고,
    잘못된 가설은 파인더 21셀 패턴 대조에서 탈락 (exp13 테스트 ① 원리)
  · NE 꼭짓점은 구성점이라 원근에서 오차 → 이산 표 마진 스윕(pipeline)
    으로 정밀화한 뒤 3점 어파인으로 승격
"""
import numpy as np

from src.core.hex_grid import HEX_DIRECTIONS, axial_to_pixel, finder_centers
from src.exp13_selfdesc.layout import SPECIAL_FINDER


def canonical_finder_points(radius: int) -> np.ndarray:
    """표준 공간의 파인더 중심 3개: [NE, W, SE] (finder_centers 순서)."""
    return np.array([axial_to_pixel(q, r, 1.0)
                     for q, r in finder_centers(radius)])


def solve_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    src(N×2, 표준 공간) → dst(N×2, 이미지)의 어파인 M(2×3).
    N=3이면 정확해, N>3이면 최소제곱.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    A = np.column_stack([src, np.ones(len(src))])      # N×3
    try:
        if len(src) == 3:
            sol = np.linalg.solve(A, dst)              # 정확해 (lstsq보다 빠름)
        else:
            sol, _res, rank, _sv = np.linalg.lstsq(A, dst, rcond=None)
            if rank < 3:
                raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        raise ValueError("어파인 추정 실패: 대응점이 퇴화(공선)")
    return sol.T                                       # 2×3


def apply_affine(M: np.ndarray, xs, ys) -> tuple:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return (M[0, 0] * xs + M[0, 1] * ys + M[0, 2],
            M[1, 0] * xs + M[1, 1] * ys + M[1, 2])


def affine_scale(M: np.ndarray) -> float:
    """표준 공간 hex_size=1 기준 → 이미지의 실효 hex_size 추정치."""
    return float(np.sqrt(abs(np.linalg.det(M[:, :2]))))


def is_mirrored(M: np.ndarray) -> bool:
    return bool(np.linalg.det(M[:, :2]) < 0)


def similarity_apex(p_w: np.ndarray, p_se: np.ndarray,
                    canon: np.ndarray, mirrored: bool) -> np.ndarray:
    """
    W·SE 이미지 측정점 2개 → 닮음 변환의 NE 꼭짓점 구성점.

    복소수 표현: 고유 닮음 z′=a·z+b (a=(q2−q1)/(p2−p1)),
    거울 닮음 z′=a·conj(z)+b. canon=[NE,W,SE] 표준 좌표.
    """
    q1, q2 = complex(*p_w), complex(*p_se)
    z_ne, z_w, z_se = (complex(*p) for p in canon)
    if mirrored:
        z_ne, z_w, z_se = (z.conjugate() for z in (z_ne, z_w, z_se))
    a = (q2 - q1) / (z_se - z_w)
    b = q1 - a * z_w
    apex = a * z_ne + b
    return np.array([apex.real, apex.imag])
