"""
표 일치율(vote consensus) 신뢰도 — exp11 공용 모듈.

FTO 핵심 (concept/11_research_roadmap.md §11.1 — Honeywell US 10,235,547 패밀리 회피):
  신뢰도 = 다수표 개수 - 소수표 개수. **이산 표 개수이며, 반사율 간극이 아니다.**
  픽셀 밝기 연속량(밝기 마진, reflectance/contrast/erasure-gap 계열)은
  신뢰도 계산에 일절 사용하지 않는다.

  각 샘플점은 전역 임계값(고정 128 또는 전역 Otsu)으로 하드 이진화만 수행.
  국소 분산·블러 필터·ML 신뢰도 맵은 소거 판정 경로에서 금지.
"""
import numpy as np

# 샘플점 배치 상한: 무게중심 1 + (꼭짓점 3 + 변중점 3) × 보간 링 2
MAX_SAMPLE_POINTS = 13

# 보간 링 반경 프리셋 (무게중심→꼭짓점 보간비 t)
#   DEFAULT_RINGS: exp11 원안 — t=0.5 외곽 링 우선.
#   INNER_RINGS  : exp11b 결함② 해소 — exp11 ablation에서 t=0.5 외곽 샘플이
#                  블러 9.5px에서 이웃 삼각형 번짐을 흡수해 무게중심 표를
#                  다수결로 압도하는 절벽이 관측됨. t≤0.3 내부 우선 배치로
#                  모든 샘플을 번짐 도달 범위 밖(무게중심 근방)에 둔다.
DEFAULT_RINGS = (0.5, 0.25)
INNER_RINGS = (0.3, 0.15)


def triangle_sample_points(verts: list, n: int,
                           rings: tuple = DEFAULT_RINGS) -> list:
    """
    삼각형 내부 샘플점 n개 생성.

    배치 순서: 무게중심 → 무게중심-(꼭짓점·변중점) 보간점.
    보간 링은 rings[0]부터 채우고 모자라면 다음 링을 추가
    (기본 (0.5, 0.25) = exp11 원안, exp11b는 INNER_RINGS=(0.3, 0.15)).
    보간비 t<1이므로 모든 점이 삼각형 내부에 확실히 들어간다.

    verts: 삼각형 꼭짓점 3개 [(x,y), (x,y), (x,y)]
    n    : 샘플점 수 (다수결 동수 방지를 위해 홀수 권장, 1~13)
    rings: 보간 링 반경 t 목록 (각 0<t<1, 링당 6점)
    """
    if not (1 <= n <= 1 + 6 * len(rings)):
        raise ValueError(f"n은 1~{1 + 6 * len(rings)} 범위여야 함: {n}")
    if any(not (0.0 < t < 1.0) for t in rings):
        raise ValueError(f"링 보간비는 0<t<1 이어야 함: {rings}")

    (ax, ay), (bx, by), (cx, cy) = verts
    cen = ((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0)
    mids = [((ax + bx) / 2, (ay + by) / 2),
            ((bx + cx) / 2, (by + cy) / 2),
            ((cx + ax) / 2, (cy + ay) / 2)]
    # 무게중심 둘레에 각도상 고르게 퍼지도록 꼭짓점·변중점을 교차 배열
    ring = [verts[0], mids[0], verts[1], mids[1], verts[2], mids[2]]

    pts = [cen]
    for t in rings:
        for tx, ty in ring:
            if len(pts) >= n:
                return pts
            pts.append((cen[0] + t * (tx - cen[0]),
                        cen[1] + t * (ty - cen[1])))
    return pts


def majority_vote(dark_count: int, n: int) -> tuple:
    """
    다수결 비트 판정.

    Returns: (bit, conf)
      bit  : 1 = 어두움(채움) 다수, 0 = 밝음 다수
      conf : 다수표 - 소수표 = |2·dark_count - n|
             — 이산 표 개수. 반사율 간극(연속 밝기량)이 아님.
    """
    bit = 1 if 2 * dark_count > n else 0
    return bit, abs(2 * dark_count - n)


def global_otsu_threshold(gray: np.ndarray) -> int:
    """
    이미지 전역 Otsu 임계값.

    이미지 전체 히스토그램에서 단일 임계 1개만 산출 —
    국소 적응 임계(블록별·픽셀별)가 아니므로 FTO 제약 2를 준수한다.
    """
    hist = np.bincount(gray.reshape(-1).astype(np.int64),
                       minlength=256)[:256].astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128

    omega = np.cumsum(hist) / total                      # 클래스0 비율
    mu = np.cumsum(hist * np.arange(256)) / total        # 누적 평균
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b = np.where(denom > 0,
                           (mu_t * omega - mu) ** 2 / denom, 0.0)
    return int(np.argmax(sigma_b))
