"""
exp16 검출 파이프라인 v0 — 임의 위치·임의 회전·완만 원근의 이미지에서
D_QR을 찾아 exp13 자기기술 디코딩까지 연결.

FTO: 본 모듈은 위치 추정 전용, 신뢰도·소거 판정과 무관.
검출은 디코더에 "좌표 변환(point_map)" 하나만 주입한다 — 표 일치율,
소거 사다리, 공간 스윕, 퇴화 가드 등 소거 판정 경로(decoder_v13 내부)는
한 줄도 건드리지 않는다 (아키텍처 분리). 검출 실패는 명시적 예외
DetectionError — 미검출 오복호 경로 없음 유지. 정밀화 스윕의 목적 함수도
이산 표 마진(정수 합)만 사용, ML·연속 밝기량 불사용.

단계:
  1. 파인더 후보 탐지 (finder_detect — Otsu·연결 성분·형상 필터·흰 링)
     → 단일 파인더(W·SE) 후보와 그 쌍
  2. 기하 가설: 쌍마다 닮음 2점 대응으로 NE 꼭짓점 구성
     (W↔SE 배정 2 × 거울 2 = 4가설) × radius 후보(거리/면적 역산 →
     GRID_RADII 스냅) — affine.similarity_apex
  3. 정밀화 (둘 다 이산 표 마진 최대화, 결정론):
     3a. NE는 구성점이라 잔차를 가짐 → 꼭짓점 격자 스윕으로 3점 어파인 승격
     3b. 완만 원근의 어파인 잔차 → 3점 대응을 고정한 채 사영 자유도 2개
         (g,h)를 격자 스윕 (어파인∘사영 합성 — 격자 외곽에서 2차로 커지는
         원근 오차를 흡수, 기울기 15°까지 커버)
  4. 가설 사전검증: 파인더 21셀 패턴 7점 다수결 '정확 일치' + 콰이어트 존
     안쪽 링(R+1, 렌더링 규약상 전부 비움) 흰 다수결 — 통과 가설만
     decoder_v13(point_map 주입)으로 본 디코딩 (포맷 링 radius 대조·
     3층+가드 그대로). 서로 다른 페이로드가 복수 나오면 모호 — 명시 실패.

v0 범위(사실대로): 스케일 0.5~2×, 회전 0~360°, 원근 기울기 ≤15°,
균일~약간 잡음 배경, 조명 균일. 손촬영·복잡 배경·심한 원근은 범위 밖.
"""
import math

import numpy as np

from src.core.hex_grid import HEX_DIRECTIONS, axial_to_pixel
from src.exp9_soft.encoder_v9 import GRID_RADII
from src.exp13_selfdesc.decoder_v13 import DQRDecoderV13
from src.exp13_selfdesc.layout import expected_finder_pattern
from src.exp16_detect.affine import (affine_scale, apply_affine,
                                     canonical_finder_points, is_mirrored,
                                     similarity_apex, solve_affine)
from src.exp16_detect.finder_detect import (classify_singles,
                                            find_blob_candidates,
                                            group_finder_pairs,
                                            otsu_threshold, to_gray)

RADIUS_REL_TOL = 0.22     # r0 역산 ↔ GRID_RADII 스냅 상대 오차 허용
MAX_HYPOTHESES = 6        # 본 디코딩에 넘길 가설 상한
SWEEP_RANGE = 1.25        # 꼭짓점 스윕 반경 (hex_size_img 단위) — 원근 15° 여유
SWEEP_STEP = 0.125        # 꼭짓점 스윕 격자 간격 (hex_size_img 단위)
PERSP_RANGE = 0.006       # 사영 파라미터 (g,h) 스윕 반경 (표준 공간 단위)
PERSP_STEP = 0.00075      # 사영 스윕 격자 간격
QUIET_WHITE_MIN = 0.90    # 콰이어트 링(R+1) 셀 중 흰 다수결 비율 하한
VERIFY_RINGS = (0.5, 0.7)   # 검증 샘플 링 반경 (hs 단위) — 바깥 링(0.7)은
                            # 셀 경계 근처라 정렬 잔차에 민감 → 표 마진이
                            # 정밀 정렬에서만 만표가 되어 스윕에 기울기 제공
                            # (프로파일 B 서브삼각형의 허용 오차 ~0.2hs 대응)
N_VERIFY = 1 + 6 * len(VERIFY_RINGS)   # 셀당 검증 샘플 수 = 13


class DetectionError(ValueError):
    """검출 실패 — 명시적 예외 (미검출 오복호 경로 없음)."""


def _sorted_grid(rng: float, step: float) -> np.ndarray:
    """|δ| 오름차순 정렬 2D 격자 (모듈 캐시 — 매 가설 재생성 비용 제거)."""
    key = (round(rng, 9), round(step, 9))
    if key not in _GRID_CACHE:
        steps = np.arange(-rng, rng + 1e-9, step)
        _GRID_CACHE[key] = np.array(sorted(
            ((dx, dy) for dx in steps for dy in steps),
            key=lambda d: (d[0] * d[0] + d[1] * d[1], d[0], d[1])))
    return _GRID_CACHE[key]


_GRID_CACHE = {}


def _radius_candidates(r0_est: float) -> list:
    """역산 r0 → GRID_RADII 후보 (상대 오차 오름차순)."""
    scored = []
    for radius in GRID_RADII:
        r0 = radius - 1
        rel = abs(r0_est - r0) / r0
        if rel <= RADIUS_REL_TOL:
            scored.append((rel, radius))
    scored.sort()
    return [radius for _, radius in scored]


def _hex_ring(d: int) -> list:
    """원점 기준 hex 거리 d의 링 셀 6d개 (axial)."""
    q, r = HEX_DIRECTIONS[4][0] * d, HEX_DIRECTIONS[4][1] * d
    cells = []
    for dq, dr in HEX_DIRECTIONS:
        for _ in range(d):
            cells.append((q, r))
            q, r = q + dq, r + dr
    return cells


def _cells_sample_xy(cells: list) -> tuple:
    """
    셀 목록 → 표준 공간 검증 샘플 좌표 (len(cells)·N_VERIFY,).

    셀당 13점 = 중심 + 링 t=0.5 6점 + 링 t=0.7 6점. 디코더의 7점 판독
    (layout.hex_cell_sample_points, t=0.5)과 달리 검출 검증·스윕 전용 —
    바깥 링이 경계 여유 0.166hs라 잔차 0.1hs 수준부터 표가 깎여
    만표 포화 없이 정렬 기울기를 만든다.
    """
    xs, ys = [], []
    for q, r in cells:
        cx, cy = axial_to_pixel(q, r, 1.0)
        xs.append(cx)
        ys.append(cy)
        for t in VERIFY_RINGS:
            for i in range(6):
                ang = math.radians(60 * i)
                xs.append(cx + t * math.cos(ang))
                ys.append(cy + t * math.sin(ang))
    return np.array(xs), np.array(ys)


def _pattern_geometry(radius: int) -> dict:
    """
    가설 검증용 표준 공간 기하 모음: 파인더 21셀 샘플점 + 기대 비트,
    콰이어트 링(R+1, 전부 비움) 샘플점, 대응점 3개.
    """
    pattern = expected_finder_pattern(radius)
    fx, fy = _cells_sample_xy([(q, r) for q, r, _ in pattern])
    expected = np.array([b for _, _, b in pattern])
    quiet_cells = _hex_ring(radius + 1)
    qx, qy = _cells_sample_xy(quiet_cells)
    return {'fx': fx, 'fy': fy, 'expected': expected,
            'qx': qx, 'qy': qy, 'n_quiet': len(quiet_cells),
            'canon': canonical_finder_points(radius)}


def _cell_votes(gray, thr, tx, ty, n_cells):
    """변환 샘플점 → 셀별 어두운 표 (0~N_VERIFY). 이미지 밖이면 None."""
    h, w = gray.shape
    if tx.min() < 0 or ty.min() < 0 or tx.max() > w - 1 or ty.max() > h - 1:
        return None
    xi = np.rint(tx).astype(np.int64)
    yi = np.rint(ty).astype(np.int64)
    return (gray[yi, xi] < thr).reshape(n_cells, N_VERIFY).sum(axis=1)


def _verify_map(gray, thr, geom, map_fn) -> tuple:
    """
    가설 변환 검증 → (통과 여부, 표 마진 점수).
      · 파인더 21셀: 다수결 '정확 일치' 필수, 마진 = 기대 방향 표 합
      · 콰이어트 링(R+1): 흰 다수결 셀 비율 ≥ QUIET_WHITE_MIN
        (렌더링 규약상 전부 비움 — 코드 내부에 들어앉은 가짜 기하를 기각)
    전부 정수 표 연산 (FTO 준수 — 위치 추정 전용).
    """
    fv = _cell_votes(gray, thr, *map_fn(geom['fx'], geom['fy']),
                     len(geom['expected']))
    if fv is None:
        return False, -10 ** 9
    aligned = np.where(geom['expected'] == 1, fv, N_VERIFY - fv)
    if not (aligned * 2 > N_VERIFY).all():
        return False, int(aligned.sum())
    qv = _cell_votes(gray, thr, *map_fn(geom['qx'], geom['qy']),
                     geom['n_quiet'])
    if qv is None:
        return False, -10 ** 9
    if (qv * 2 < N_VERIFY).mean() < QUIET_WHITE_MIN:
        return False, int(aligned.sum())
    return True, int(aligned.sum()) + int((N_VERIFY - qv).sum())


def _affine_point_map(M: np.ndarray):
    """표준 공간(hex_size=1, origin=0) → 이미지 좌표 어파인 변환 클로저."""
    def map_fn(xs, ys):
        return apply_affine(M, xs, ys)
    return map_fn


def _projective_map(M: np.ndarray, g: float, h: float):
    """표준 공간 사영 왜곡(분모 1+g·x+h·y) 후 어파인 M — 합성 변환 클로저."""
    def map_fn(xs, ys):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        d = 1.0 + g * xs + h * ys
        return apply_affine(M, xs / d, ys / d)
    return map_fn


def _warp_gh(xs, ys, g: float, h: float) -> tuple:
    """표준 공간 사영 왜곡 (분모 1+g·x+h·y) — (g,h)=(0,0)이면 항등."""
    if g == 0.0 and h == 0.0:
        return xs, ys
    d = 1.0 + g * xs + h * ys
    return xs / d, ys / d


def _sweep_apex(gray, thr, geom, apex, w_pt, se_pt, hs_img,
                g=0.0, h=0.0, rng=SWEEP_RANGE, step=SWEEP_STEP):
    """
    3a 정밀화 — NE 꼭짓점(구성점)을 ±rng·hs 격자로 이동시키며
    파인더 21셀 패턴 표 마진을 최대화하는 3점 대응을 찾는다 (결정론:
    오프셋을 |δ| 오름차순으로 순회, 마진 갱신은 strict > 만 — 동률이면
    원점에 가까운 해 유지). (g,h)는 고정된 사영 왜곡 아래에서의 재스윕용
    (코스-투-파인 2차 패스 — 프로파일 B의 좁은 정렬 허용 오차 대응).

    어파인은 dst에 선형 → 변환점 = base + δ·(꼭짓점 barycentric 가중)으로
    분해해 오프셋마다 어파인 재추정 없이 평가한다 (사영 왜곡 합성에서도
    성립: 왜곡된 표준 좌표에 대한 barycentric이므로).

    Returns: dst 3×2 (정밀화된 [NE, W, SE] 이미지 대응점) 또는 None.
    """
    cx, cy = _warp_gh(geom['canon'][:, 0], geom['canon'][:, 1], g, h)
    sx, sy = _warp_gh(geom['fx'], geom['fy'], g, h)
    B = np.column_stack([cx, cy, np.ones(3)])          # 3×3
    S = np.column_stack([sx, sy, np.ones(len(sx))])    # N×3
    try:
        W = S @ np.linalg.inv(B)                       # N×3 barycentric 가중
    except np.linalg.LinAlgError:
        return None
    dst0 = np.array([apex, w_pt, se_pt])               # 3×2
    base = W @ dst0                                    # N×2 (δ=0 변환점)
    w_apex = W[:, 0]                                   # 꼭짓점 가중
    expected = geom['expected']
    h_img, w_img = gray.shape

    offsets = _sorted_grid(rng, step) * hs_img                    # K×2

    # 1차 거름: NE 중심 셀(패턴 첫 N_VERIFY샘플)만 — 어두운 다수결이 안
    # 나오는 오프셋은 정확 일치가 불가능하므로 전체 평가에서 제외 (가짜
    # 가설 분기의 지배 비용 절감, 결과 동일: 필요조건만 검사)
    nv = N_VERIFY
    tx0 = base[:nv, 0][None, :] + offsets[:, 0][:, None] * w_apex[:nv][None, :]
    ty0 = base[:nv, 1][None, :] + offsets[:, 1][:, None] * w_apex[:nv][None, :]
    xi0 = np.clip(np.rint(tx0), 0, w_img - 1).astype(np.int64)
    yi0 = np.clip(np.rint(ty0), 0, h_img - 1).astype(np.int64)
    keep = (gray[yi0, xi0] < thr).sum(axis=1) * 2 > nv
    if not keep.any():
        return None
    offsets = offsets[keep]

    # 전 오프셋 일괄 평가 (K×N) — 벤치마크 규모에서 파이썬 루프는 병목
    tx = base[:, 0][None, :] + offsets[:, 0][:, None] * w_apex[None, :]
    ty = base[:, 1][None, :] + offsets[:, 1][:, None] * w_apex[None, :]
    in_img = ((tx.min(axis=1) >= 0) & (ty.min(axis=1) >= 0)
              & (tx.max(axis=1) <= w_img - 1) & (ty.max(axis=1) <= h_img - 1))
    xi = np.clip(np.rint(tx), 0, w_img - 1).astype(np.int64)
    yi = np.clip(np.rint(ty), 0, h_img - 1).astype(np.int64)
    votes = (gray[yi, xi] < thr).reshape(len(offsets), len(expected),
                                         nv).sum(axis=2)           # K×21
    aligned = np.where(expected[None, :] == 1, votes, nv - votes)
    exact = (aligned * 2 > nv).all(axis=1) & in_img
    if not exact.any():
        return None
    margins = np.where(exact, aligned.sum(axis=1), -1)
    k = int(np.argmax(margins))    # 동률이면 |δ| 오름차순 첫 항 = 원점 근접 해
    dst = dst0.copy()
    dst[0] += offsets[k]
    return dst


def _sweep_perspective(gray, thr, geom, dst3, center=(0.0, 0.0),
                       rng=PERSP_RANGE, step=PERSP_STEP):
    """
    3b 정밀화 — 3점 대응(dst3)을 고정한 채 사영 자유도 (g,h)를 격자 스윕
    (center 주변 ±rng — 코스-투-파인 2차 패스에서 재사용).

    각 (g,h)에서 표준 대응점 3개를 사영 왜곡한 뒤 어파인을 재추정하면
    합성 변환이 항상 3점 대응을 정확히 보간 → 남는 2자유도만 움직인다.
    목적 함수 = 파인더 정렬 마진 + 콰이어트 링 흰 마진 (정수 표 합,
    중심 거리 오름차순·strict > 갱신 = 결정론, center 해를 기본 유지).

    Returns: (map_fn, M, score, (g,h)) — 전 후보 부적합이면 None.
    """
    canon = geom['canon']
    score_max = N_VERIFY * (len(geom['expected']) + geom['n_quiet'])  # 만표

    # center 우선 평가 — 이미 만표면 스윕 불필요 (회전·스케일만 있는
    # 다수 사례의 지배 비용 절감, 결과 동일: 만표는 strict > 갱신 규칙상
    # 어떤 오프셋도 이길 수 없음)
    g0, h0 = center
    try:
        wx, wy = _warp_gh(canon[:, 0], canon[:, 1], g0, h0)
        M0 = solve_affine(np.column_stack([wx, wy]), dst3)
        map0 = _projective_map(M0, g0, h0)
        ok0, score0 = _verify_map(gray, thr, geom, map0)
        if ok0 and score0 >= score_max:
            return map0, M0, score0, (g0, h0)
    except ValueError:
        pass

    offsets = _sorted_grid(rng, step) + np.array(center)           # K×2

    # ── 전 (g,h) 일괄 평가 (K개 변환을 배치 선형대수로) ──────────────
    K = len(offsets)
    n_f, n_q = len(geom['expected']), geom['n_quiet']
    sx = np.concatenate([geom['fx'], geom['qx']])      # N (파인더+콰이어트)
    sy = np.concatenate([geom['fy'], geom['qy']])
    g = offsets[:, 0][:, None]
    h = offsets[:, 1][:, None]

    # 대응점 사영 왜곡 → 배치 3×3 해 → 합성 어파인 M_k (K×3×2)
    d_anchor = 1.0 + g * canon[:, 0][None, :] + h * canon[:, 1][None, :]
    valid = (np.abs(d_anchor) > 0.5).all(axis=1)
    d_anchor = np.where(np.abs(d_anchor) < 1e-9, 1.0, d_anchor)
    warped = canon[None, :, :] / d_anchor[:, :, None]            # K×3×2
    B = np.concatenate([warped, np.ones((K, 3, 1))], axis=2)     # K×3×3
    det_ok = np.abs(np.linalg.det(B)) > 1e-9
    valid &= det_ok
    B[~det_ok] = np.eye(3)
    Mk = np.linalg.solve(B, np.broadcast_to(dst3, (K, 3, 2)))    # K×3×2

    # 샘플점 사영 왜곡 후 일괄 변환 (K×N×2)
    d_s = 1.0 + g * sx[None, :] + h * sy[None, :]                # K×N
    Sw = np.stack([sx[None, :] / d_s, sy[None, :] / d_s,
                   np.ones_like(d_s)], axis=2)                   # K×N×3
    T = Sw @ Mk                                                  # K×N×2
    tx, ty = T[:, :, 0], T[:, :, 1]
    h_img, w_img = gray.shape
    in_img = ((tx.min(axis=1) >= 0) & (ty.min(axis=1) >= 0)
              & (tx.max(axis=1) <= w_img - 1)
              & (ty.max(axis=1) <= h_img - 1))
    valid &= in_img
    xi = np.clip(np.rint(tx), 0, w_img - 1).astype(np.int64)
    yi = np.clip(np.rint(ty), 0, h_img - 1).astype(np.int64)
    dark = gray[yi, xi] < thr                                    # K×N

    nv = N_VERIFY
    fv = dark[:, :n_f * nv].reshape(K, n_f, nv).sum(axis=2)      # K×21
    aligned = np.where(geom['expected'][None, :] == 1, fv, nv - fv)
    exact = (aligned * 2 > nv).all(axis=1)
    qv = dark[:, n_f * nv:].reshape(K, n_q, nv).sum(axis=2)      # K×nq
    quiet_ok = (qv * 2 < nv).mean(axis=1) >= QUIET_WHITE_MIN
    ok_mask = valid & exact & quiet_ok
    if not ok_mask.any():
        return None
    scores = aligned.sum(axis=1) + (nv - qv).sum(axis=1)
    scores = np.where(ok_mask, scores, -1)
    k = int(np.argmax(scores))   # 동률 → 중심 거리 오름차순 첫 항 (결정론)
    g_best, h_best = float(offsets[k, 0]), float(offsets[k, 1])
    M = Mk[k].T.copy()
    return (_projective_map(M, g_best, h_best), M, int(scores[k]),
            (g_best, h_best))


def detect_hypotheses(image) -> list:
    """
    이미지 → 사전검증(파인더 패턴 정확 일치 + 콰이어트 링)을 통과한
    기하 가설 목록 (표 마진 점수 내림차순).

    Returns: [{'map_fn': callable, 'M': 2×3 ndarray, 'radius': int,
               'mirrored': bool, 'hex_size_img': float, 'score': int,
               'n_candidates': int}, ...]
    (빈 목록 = 검출 실패. 최종 확정은 detect_and_decode의 본 디코딩이 담당.)
    """
    gray = to_gray(image)
    thr = otsu_threshold(gray)
    cands, _labels = find_blob_candidates(gray, thr)
    singles = classify_singles(gray, thr, cands)
    # v1: 격자 반경 스냅(이산 기하)으로 쌍 순위 — 모션 번짐 가짜 쌍이
    # 상한에서 진짜 쌍을 밀어내는 현상 차단 (exp17 진단·exp18, FTO: 밝기량 아님)
    grid_r0 = [r - 1 for r in GRID_RADII]
    pairs = group_finder_pairs(singles, grid_r0=grid_r0)

    hyps = []
    geom_cache = {}
    for p in pairs:
        s1, s2 = p['pair']
        for radius in _radius_candidates(p['r0_est']):
            if radius not in geom_cache:
                geom_cache[radius] = _pattern_geometry(radius)
            geom = geom_cache[radius]
            for w_cand, se_cand in ((s1, s2), (s2, s1)):
                w_pt = np.array([w_cand['cx'], w_cand['cy']])
                se_pt = np.array([se_cand['cx'], se_cand['cy']])
                for mir in (False, True):
                    apex = similarity_apex(w_pt, se_pt, geom['canon'], mir)
                    dst3 = _sweep_apex(gray, thr, geom, apex,
                                       w_pt, se_pt, p['hs_est'])
                    if dst3 is None:
                        continue
                    refined = _sweep_perspective(gray, thr, geom, dst3)
                    if refined is None:
                        continue
                    map_fn, M, score, gh = refined
                    score_max = N_VERIFY * (len(geom['expected'])
                                            + geom['n_quiet'])
                    if score < score_max:
                        # 2차 패스 (코스-투-파인): 확정된 (g,h) 아래에서
                        # 꼭짓점을 절반 간격으로 재스윕 → 사영도 미세 재스윕.
                        # 프로파일 B(서브삼각형, 허용 오차 ~0.2hs)의 원근
                        # 잔차를 줄인다. 개선 없으면(None) 1차 해 유지.
                        dst3b = _sweep_apex(
                            gray, thr, geom, dst3[0], dst3[1], dst3[2],
                            p['hs_est'], g=gh[0], h=gh[1],
                            rng=SWEEP_STEP * 2, step=SWEEP_STEP / 4)
                        if dst3b is not None:
                            refined2 = _sweep_perspective(
                                gray, thr, geom, dst3b, center=gh,
                                rng=PERSP_STEP * 2, step=PERSP_STEP / 4)
                            if refined2 is not None and refined2[2] > score:
                                map_fn, M, score, gh = refined2
                    hyps.append({'map_fn': map_fn, 'M': M, 'radius': radius,
                                 'mirrored': is_mirrored(M),
                                 'hex_size_img': affine_scale(M),
                                 'score': score, 'n_candidates': len(cands)})
                    if len(hyps) >= MAX_HYPOTHESES:
                        break
                if len(hyps) >= MAX_HYPOTHESES:
                    break
            if len(hyps) >= MAX_HYPOTHESES:
                break
        if len(hyps) >= MAX_HYPOTHESES:
            break
    hyps.sort(key=lambda x: -x['score'])
    return hyps


def detect_and_decode(image, hyps: list = None, **decoder_kwargs) -> dict:
    """
    검출 → 디코딩 전체 파이프라인. 성공 시 decoder_v13의 stats dict에
    'detection' 항목을 더해 반환, 실패 시 DetectionError/복호 예외.

    hyps: detect_hypotheses 결과 재사용 (벤치마크처럼 검출률을 따로 잴 때
    이중 검출 비용 제거). None이면 내부에서 검출.
    decoder_kwargs는 DQRDecoderV13에 그대로 전달 (threshold, use_sweep 등).
    """
    if hyps is None:
        hyps = detect_hypotheses(image)
    if not hyps:
        raise DetectionError(
            "exp16: 파인더 기하 가설 없음 (코드 미존재이거나 v0 범위 밖)")

    # 표 마진 점수 내림차순으로 본 디코딩 — 첫 성공 채택.
    # 틀린 기하 가설의 차단은 디코더 자체의 가드가 담당한다 (포맷 링
    # radius 대조, RS, 길이 프리픽스, 퇴화 코드워드 가드 — exp15b에서
    # 미검출 오복호 0 실측). 검출은 후보 순서만 정하고 판정에 관여하지 않음.
    last_err = None
    for hyp in hyps:
        dec = DQRDecoderV13(point_map=hyp['map_fn'], **decoder_kwargs)
        geometry = {'radius': hyp['radius'], 'hex_size': 1.0,
                    'origin_x': 0.0, 'origin_y': 0.0}
        try:
            stats = dec.decode_with_stats(image, geometry=geometry)
        except Exception as e:           # 가설별 복호 실패 — 다음 가설 시도
            last_err = e
            continue
        stats['hex_size'] = hyp['hex_size_img']   # 표준 1.0 대신 실측 추정치
        stats['detection'] = {
            'affine': hyp['M'].tolist(),
            'mirrored': hyp['mirrored'],
            'hex_size_img': hyp['hex_size_img'],
            'radius': hyp['radius'],
            'score': hyp['score'],
            'n_hypotheses': len(hyps),
        }
        return stats

    raise DetectionError(
        f"exp16: 기하 가설 {len(hyps)}건 전부 복호 실패 "
        f"(마지막 오류: {type(last_err).__name__}: {last_err})")
