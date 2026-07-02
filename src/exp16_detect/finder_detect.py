"""
exp16 1단계 — 파인더 후보 탐지 (결정론 CV)

FTO: 본 모듈은 위치 추정 전용, 신뢰도·소거 판정과 무관.
검출 결과는 좌표 가설일 뿐이며, 비트 신뢰도(이산 표 개수)·소거 위치 결정은
전적으로 src/core/softdecode·exp11 계열이 담당한다. 본 모듈은 그 경로를
임포트하지 않는다 (아키텍처 분리 — tests/test_exp16_detect.py가 단언).
ML·학습 모델 불사용 — 전역 Otsu + 연결 성분 + 고정 임계 형상 필터만.

탐지 대상 (exp13 파인더 구조, src/exp13_selfdesc/layout.py 참조):
  · 단일 파인더 ×2 (W·SE) : 솔리드 검정 육각 1셀 + 흰 이웃 6셀 — 규약상
    항상 고립 성분. 차등 파인더(NE 3셀 클러스터)는 채움 셀의 바깥 이웃이
    포맷/데이터 셀이라 데이터 영역과 합쳐질 수 있어 여기서 직접 찾지 않고,
    단일 2개로부터 정삼각형 꼭짓점 구성 + 패턴 대조로 확정한다 (affine 참조).
회전·거울상·스케일·완만 원근에서도 보존되는 성질만 쓴다:
  볼록성(솔리디티), 컴팩트니스(4πA/P²),
  흰 링(단일 파인더 6이웃 전부 비움 → 중심 반지름 1.45·hs 원이 꽃(7셀)
  내부에 머물러 거의 전부 밝음 — 무작위 데이터 셀은 이웃 절반이 어두워 탈락).
"""
import itertools
import math

import cv2
import numpy as np

HEX_AREA_COEF = 3.0 * math.sqrt(3) / 2.0     # 정육각형 면적 = 2.598·hs²

MIN_AREA_PX = 60          # hs≈5px 미만은 (서브)삼각형 표결 자체가 불가
SOLIDITY_MIN = 0.85       # 육각형·3셀 클러스터는 볼록에 가까움, 데이터 블롭 탈락
COMPACT_MIN = 0.45        # 4πA/P²: 정육각형 0.907, 3셀 클러스터 이론값 ≈0.68
RING_RADIUS = 1.45        # 흰 링 샘플 반지름 (hs 단위) — 파인더 꽃 내부 보장
RING_POINTS = 24
WHITE_RING_MIN = 0.85     # 단일 파인더 판정: 링의 밝은 점 비율 하한
PAIR_HS_RATIO = 1.5       # 단일 파인더 두 개의 hs 추정 비율 허용 (원근 여유)
MAX_PAIRS = 24            # 가설 폭주 방지 (격자 스냅·hs 유사도 순 상위만)
PAIR_GRID_TOL = 0.25      # 쌍 거리 역산 r0 ↔ 격자 반경 스냅 상대 오차 허용
                          # (pipeline.RADIUS_REL_TOL=0.22보다 약간 느슨 — 경계
                          #  진짜 쌍을 미리 버리지 않되, 비스냅 가짜 쌍은 제거)


def to_gray(image) -> np.ndarray:
    """PIL 이미지 → 그레이 (decoder_v13._gray_and_thr와 동일 규약: RGB 합//3)."""
    pixels = np.array(image.convert('RGB'), dtype=np.int32)
    return (pixels.sum(axis=2) // 3).astype(np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    """전역 Otsu 임계 — '어두움 = gray < thr' 규약(decoder_v13과 동일).

    cv2의 Otsu 반환값 t는 '어두운 클래스 = gray <= t' 의미라 +1 보정
    (순수 흑백 {0,255} 이미지에서 t=0이 나와 dark<0이 공집합이 되는 함정).
    """
    thr, _ = cv2.threshold(gray.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thr) + 1


def _component_shape_stats(comp_mask: np.ndarray) -> tuple:
    """성분 서브마스크 → (솔리디티, 컴팩트니스). 윤곽 기반."""
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area <= 0:
        return 0.0, 0.0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    perim = cv2.arcLength(cnt, closed=True)
    solidity = area / hull_area if hull_area > 0 else 0.0
    compact = 4.0 * math.pi * area / (perim * perim) if perim > 0 else 0.0
    return solidity, compact


def find_blob_candidates(gray: np.ndarray, thr: int = None) -> tuple:
    """
    어두운 연결 성분 중 파인더가 될 수 있는 블롭 목록.

    Returns: (candidates, labels)
      candidates: [{cx, cy, area, solidity, compact, label}, ...]
      labels    : 연결 성분 라벨 이미지 (클러스터 셀 분해·정밀화에 재사용)
    """
    if thr is None:
        thr = otsu_threshold(gray)
    mask = (gray < thr).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    img_area = gray.shape[0] * gray.shape[1]
    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA_PX or area > 0.25 * img_area:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]),
                      int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]),
                      int(stats[i, cv2.CC_STAT_HEIGHT]))
        # 가늘고 긴 블롭(데이터 줄무늬) 조기 탈락 — 파인더는 종횡비 ≲2
        if max(w, h) > 3.0 * min(w, h):
            continue
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        solidity, compact = _component_shape_stats(comp)
        if solidity < SOLIDITY_MIN or compact < COMPACT_MIN:
            continue
        cands.append({'cx': float(centroids[i][0]),
                      'cy': float(centroids[i][1]),
                      'area': area, 'solidity': solidity,
                      'compact': compact, 'label': i})
    return cands, labels


def ring_dark_fraction(gray: np.ndarray, thr: int, cx: float, cy: float,
                       radius_px: float, n: int = RING_POINTS) -> float:
    """중심 (cx,cy) 반지름 radius_px 원 위 n점의 어두운 비율."""
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    xs = np.clip(np.rint(cx + radius_px * np.cos(ang)),
                 0, gray.shape[1] - 1).astype(np.int64)
    ys = np.clip(np.rint(cy + radius_px * np.sin(ang)),
                 0, gray.shape[0] - 1).astype(np.int64)
    return float((gray[ys, xs] < thr).mean())


def classify_singles(gray: np.ndarray, thr: int, cands: list) -> list:
    """흰 링 검증을 통과한 단일 파인더 후보 (hs 추정치 부여)."""
    singles = []
    for c in cands:
        hs = math.sqrt(c['area'] / HEX_AREA_COEF)
        dark = ring_dark_fraction(gray, thr, c['cx'], c['cy'],
                                  RING_RADIUS * hs)
        if 1.0 - dark >= WHITE_RING_MIN:
            singles.append({**c, 'hs_est': hs})
    return singles


def group_finder_pairs(singles: list, grid_r0: list = None) -> list:
    """
    흰 링을 통과한 단일 파인더 후보의 쌍 — (W, SE) 가설의 재료.

    NE는 차등 파인더(3셀 클러스터)라 여기서 직접 찾지 않고, 쌍에서
    정삼각형 꼭짓점을 구성해(affine.similarity_apex) 패턴 대조로 확정한다.
    쌍 거리 = 정삼각형 변 = 3·hs·r0 (layout 기하) → r0 역산을 동봉.

    grid_r0: 유효 격자 r0 값(=radius-1) 목록. 주어지면 v1 모션 강건 순위
      적용 — 진짜 파인더 쌍의 거리는 이산 격자 반경에 스냅되지만(r0_est가
      어떤 격자 r0에 근접) 모션 번짐이 만든 가짜 쌍은 임의 거리라 안 맞는다.
      이 '격자 스냅 오차'(밝기량 아닌 이산 기하)를 1순위 정렬키로 써서
      가짜 쌍이 MAX_PAIRS 상한에서 진짜 쌍을 밀어내는 현상을 막는다(exp17
      진단: 모션 L≥5에서 가짜 단일 폭증 → 쌍 상한 포화 → 진짜 쌍 탈락).
      어떤 격자에도 안 맞는 쌍은 가설을 0개 생성하므로 드롭(슬롯 낭비 제거).
      None이면 구버전(hs 유사도 순) — 후방 호환.

    Returns: [{'pair': (s1, s2), 'r0_est': float, 'hs_est': float,
               'score': float, 'grid_rel': float|None}, ...]
    """
    pairs = []
    for s1, s2 in itertools.combinations(singles, 2):
        ratio = max(s1['hs_est'], s2['hs_est']) / min(s1['hs_est'],
                                                      s2['hs_est'])
        if ratio > PAIR_HS_RATIO:
            continue
        hs_mean = 0.5 * (s1['hs_est'] + s2['hs_est'])
        d12 = math.hypot(s1['cx'] - s2['cx'], s1['cy'] - s2['cy'])
        r0_est = d12 / (3.0 * hs_mean)
        if r0_est < 3.0:    # GRID_RADII 최소 r0=7 — 한참 못 미치면 기각
            continue
        grid_rel = None
        if grid_r0 is not None:
            grid_rel = min(abs(r0_est - g) / g for g in grid_r0)
            if grid_rel > PAIR_GRID_TOL:   # 비스냅 쌍 = 가설 0개 → 슬롯 낭비 제거
                continue
        pairs.append({'pair': (s1, s2), 'r0_est': r0_est,
                      'hs_est': hs_mean, 'score': ratio, 'grid_rel': grid_rel})
    if grid_r0 is not None:
        # 격자 스냅 오차 1순위(진짜 쌍 ≈0), hs 유사도 2순위
        pairs.sort(key=lambda p: (p['grid_rel'], p['score']))
    else:
        pairs.sort(key=lambda p: p['score'])
    return pairs[:MAX_PAIRS]
