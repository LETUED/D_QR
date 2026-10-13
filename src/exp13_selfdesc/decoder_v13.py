"""
D_QR Phase 13 디코더 — 메타데이터 없이 이미지만으로 디코딩 (exp13)

복호 단계 (전부 결정론적, CV 검출 파이프라인 불요):
  1. 기하 자기추정 — 이미지 크기에서 radius 후보 순회, hex_size·origin을
     렌더링 규약(quiet zone 비율 고정) 역산으로 도출 (layout.estimate_geometry)
  2. 방향 판별    — 회전 0°/120°/240° × 거울상 6가지 가설마다 파인더 21셀의
     기대 패턴(차등 파인더 포함)을 7점 다수결로 대조 → 유일 가설 확정
  3. 포맷 링 판독 — 18비트 × 3사본 비트별 다수결 → version·profile·
     ecc_level·radius_idx. radius_idx가 1단계 추정과 일치해야 확정
     (가설이 여러 개 살아남으면 모호 오류 — 침묵 오정정보다 명시 실패)
  4. 페이로드     — exp11b 계열 표 일치율 판독(외곽 링 → 내부 링 폴백)
     + 소거 사다리(budget=True: 1·2층 + errata 예산 + 최종 단계 순수 소거)
     + 사다리 전 패스 실패 시 공간 소거 스윕(exp11b 결함① 경로 —
       데이터 셀 목록을 data_cells_v13으로 넘겨 그룹 유도, 다중 가설
       합의 수락·후보 예산 nsym-2는 v11b와 동일).
     exp15b: 사다리·스윕 양 경로에 퇴화 코드워드 가드(전부-0/상수 메시지
     거부 — 전부-0 코드워드 퇴화 봉쇄) 적용. 포맷 링 scramble 비트가 서면
     복호 메시지를 LFSR 마스크로 원복(디스크램블)한 뒤 길이 프리픽스 검사.
     복호 메시지 선두 4바이트 길이 프리픽스의 무결성 검사 후 데이터 절단
     (사다리·스윕 어느 경로든 동일 검사 — 위반 시 해당 결과 폐기).

FTO 준수: 신뢰도 = 이산 표 개수(다수표-소수표)만 사용, 이진화는 전역
임계(고정 128/전역 Otsu), 밝기 연속량·국소 필터·ML 불사용.
공간 소거 스윕은 위치 가설 시행이라 안전 (spatial.py FTO 위치 참조).
"""
import numpy as np
from reedsolo import ReedSolomonError

from src.core.bitutils import _bits_to_bytes
from src.core.scramble import xor_scramble
from src.core.consensus import (DEFAULT_RINGS, INNER_RINGS,
                                global_otsu_threshold,
                                triangle_sample_points)
from src.core.hex_grid import hex_primary_triangles, hex_sub_triangles
from src.core.softdecode import erasure_ladder_decode
from src.exp11_consensus.spatial import (build_spatial_groups,
                                         spatial_erasure_sweep)
from src.exp9_soft.encoder_v9 import GRID_RADII, PROFILES
from src.exp13_selfdesc.encoder_v13 import LEN_PREFIX
from src.exp13_selfdesc.format_ring import (FORMAT_VERSION, PROFILE_BY_ID,
                                            majority_bits, nsym_for_level,
                                            parse_format)
from src.exp13_selfdesc.layout import (codeword_length, data_cells_v13,
                                       estimate_geometry,
                                       expected_finder_pattern,
                                       format_cells_per_finder,
                                       hex_cell_sample_points,
                                       message_capacity, transform_points)

DEFAULT_SAMPLE_POINTS = 9   # exp11 ablation 채택값 유지


class DQRDecoderV13:
    """
    D_QR Phase 13 디코더 — 자기기술 복호.

    사용 예 (파라미터는 이미지에서 전부 자기추정):
        dec = DQRDecoderV13()
        data = dec.decode(Image.open("code.png"))
    """

    def __init__(self, threshold: int = 128,
                 n_samples: int = DEFAULT_SAMPLE_POINTS,
                 use_otsu: bool = False,
                 rings: tuple = DEFAULT_RINGS,
                 fallback_rings: tuple = INNER_RINGS,
                 use_sweep: bool = True,
                 conf_fill: bool = True,
                 point_map=None):
        if n_samples % 2 == 0:
            raise ValueError(f"n_samples는 홀수여야 함: {n_samples}")
        self.threshold = threshold
        self.n_samples = n_samples
        self.use_otsu = use_otsu
        self.rings = tuple(rings)
        self.fallback_rings = tuple(fallback_rings) if fallback_rings else None
        self.use_sweep = use_sweep
        self.conf_fill = conf_fill
        # exp16 검출 경로 훅: (xs, ys) → 이미지 좌표로 옮기는 외부 변환.
        # 검출 모듈이 추정한 어파인이 회전·거울상·스케일을 전부 흡수하므로
        # point_map이 있으면 내부 방향 가설은 항등 (0, False)만 검증한다.
        # (좌표 변환만 주입 — 신뢰도·소거 판정 경로는 그대로 = FTO 분리 유지)
        self.point_map = point_map
        self._coord_cache = {}
        self._group_cache = {}

    def _map_points(self, xs, ys, ox, oy, rot_k, mirrored):
        """표준 샘플점 → 이미지 좌표. point_map(검출 어파인)이 있으면 우선."""
        if self.point_map is not None:
            return self.point_map(xs, ys)
        return transform_points(xs, ys, ox, oy, rot_k, mirrored)

    # ── 공통: 그레이 변환·전역 임계 (v11과 동일 규약) ────────────────
    def _gray_and_thr(self, image):
        pixels = np.array(image.convert('RGB'), dtype=np.int32)
        gray = pixels.sum(axis=2) // 3
        thr = global_otsu_threshold(gray) if self.use_otsu else self.threshold
        return gray, thr

    def _cell_bits(self, gray, thr, cells, hs, ox, oy, rot_k, mirrored):
        """셀 단위 7점 다수결 비트 — 파인더/포맷 판독용 (이산 표·전역 임계)."""
        xs, ys = [], []
        for q, r in cells:
            for px, py in hex_cell_sample_points(q, r, hs, ox, oy):
                xs.append(px)
                ys.append(py)
        tx, ty = self._map_points(np.array(xs), np.array(ys),
                                  ox, oy, rot_k, mirrored)
        img_h, img_w = gray.shape
        xi = np.clip(np.rint(tx), 0, img_w - 1).astype(np.int64)
        yi = np.clip(np.rint(ty), 0, img_h - 1).astype(np.int64)
        dark = (gray[yi, xi] < thr).reshape(len(cells), 7)
        return (dark.sum(axis=1) * 2 > 7).astype(int)

    # ── 2단계: 방향 판별 ─────────────────────────────────────────────
    def detect_orientation(self, gray, thr, radius, hs, ox, oy) -> list:
        """파인더 패턴이 정확히 일치하는 (rot_k, mirrored) 가설 목록."""
        pattern = expected_finder_pattern(radius)
        cells = [(q, r) for q, r, _ in pattern]
        expected = np.array([b for _, _, b in pattern])
        # point_map(검출 어파인)이 방향을 이미 흡수했으면 항등 가설만 검증
        # (6가설 전부 같은 좌표로 변환되어 모호 판정이 무의미해지기 때문)
        hypotheses = ([(0, False)] if self.point_map is not None
                      else [(k, m) for m in (False, True) for k in (0, 1, 2)])
        matches = []
        for rot_k, mirrored in hypotheses:
            bits = self._cell_bits(gray, thr, cells, hs, ox, oy,
                                   rot_k, mirrored)
            if np.array_equal(bits, expected):
                matches.append((rot_k, mirrored))
        return matches

    # ── 3단계: 포맷 링 판독 ─────────────────────────────────────────
    def read_format(self, gray, thr, radius, hs, ox, oy,
                    rot_k, mirrored) -> dict:
        copies = [list(self._cell_bits(gray, thr, cells, hs, ox, oy,
                                       rot_k, mirrored))
                  for cells in format_cells_per_finder(radius)]
        return parse_format(majority_bits(copies))

    # ── 4단계: 페이로드 표 일치율 판독 ──────────────────────────────
    def _payload_coords(self, radius, hs, bpc, ox, oy, rot_k, mirrored, rings):
        key = (radius, round(hs, 6), bpc, round(ox, 6), round(oy, 6),
               rot_k, mirrored, rings, self.n_samples, id(self.point_map))
        if key in self._coord_cache:
            return self._coord_cache[key]

        cells = data_cells_v13(radius)
        tri_fn = hex_primary_triangles if bpc == 6 else hex_sub_triangles
        xs, ys = [], []
        for q, r in cells:
            for verts, _centroid in tri_fn(q, r, hs, ox, oy):
                for px, py in triangle_sample_points(verts, self.n_samples,
                                                     rings=rings):
                    xs.append(px)
                    ys.append(py)
        tx, ty = self._map_points(np.array(xs), np.array(ys),
                                  ox, oy, rot_k, mirrored)
        n_tris = len(cells) * bpc
        coords = (tx.reshape(n_tris, self.n_samples),
                  ty.reshape(n_tris, self.n_samples))
        self._coord_cache[key] = coords
        return coords

    def _read_codeword(self, gray, thr, radius, hs, bpc, ox, oy,
                       rot_k, mirrored, codeword_len, rings):
        xs, ys = self._payload_coords(radius, hs, bpc, ox, oy,
                                      rot_k, mirrored, rings)
        img_h, img_w = gray.shape
        xi = np.clip(np.rint(xs), 0, img_w - 1).astype(np.int64)
        yi = np.clip(np.rint(ys), 0, img_h - 1).astype(np.int64)

        # v11과 동일: 하드 이진화 후 밝기값은 버려지고 표 개수만 남는다
        dark_votes = (gray[yi, xi] < thr).sum(axis=1)
        n = self.n_samples
        bits = (2 * dark_votes > n).astype(int)
        confs = np.abs(2 * dark_votes - n)

        codeword = _bits_to_bytes(list(bits), codeword_len)
        byte_conf = []
        for i in range(codeword_len):
            chunk = confs[i * 8:(i + 1) * 8]
            byte_conf.append(int(chunk.min()) if chunk.size else 0)
        return codeword, byte_conf

    def _spatial_groups(self, radius, bits_per_cell, codeword_len, nsym):
        # exp13 셀 배치(포맷 링 54셀 추가 예약)로 그룹 유도 — v9 기본과 다름
        key = (radius, bits_per_cell, codeword_len, nsym)
        if key not in self._group_cache:
            self._group_cache[key] = build_spatial_groups(
                radius, bits_per_cell, codeword_len, nsym,
                data_cells=data_cells_v13(radius))
        return self._group_cache[key]

    # ── 전체 파이프라인 ─────────────────────────────────────────────
    def decode_with_stats(self, image, geometry: dict = None) -> dict:
        """
        geometry: 외부(검출 모듈)가 확정한 기하 1건을 주입할 때 사용 —
        {'radius', 'hex_size', 'origin_x', 'origin_y'}. 이때 hex_size·origin은
        point_map 입력(표준 레이아웃) 공간 기준이며, 보통 point_map과 함께
        (hex_size=1, origin=0) 표준 공간으로 쓴다. None이면 기존 자기추정.
        """
        gray, thr = self._gray_and_thr(image)

        if geometry is not None:
            geo_cands = [geometry]
        else:
            geo_cands = estimate_geometry(image.width, image.height)
        if not geo_cands:
            raise ValueError("exp13: 이미지 크기가 렌더링 규약과 불일치 "
                             "(기하 후보 없음)")

        # (기하 × 방향) 가설 중 포맷 링 검증까지 통과한 것만 채택 후보
        viable = []
        for geo in geo_cands:
            R, hs = geo['radius'], geo['hex_size']
            ox, oy = geo['origin_x'], geo['origin_y']
            for rot_k, mirrored in self.detect_orientation(
                    gray, thr, R, hs, ox, oy):
                fmt = self.read_format(gray, thr, R, hs, ox, oy,
                                       rot_k, mirrored)
                if fmt['version'] != FORMAT_VERSION:
                    continue
                if not (0 <= fmt['radius_idx'] < len(GRID_RADII)):
                    continue
                if GRID_RADII[fmt['radius_idx']] != R:   # 반지름 일치 검증
                    continue
                if fmt['profile'] >= len(PROFILE_BY_ID):
                    continue
                viable.append((geo, rot_k, mirrored, fmt))

        if not viable:
            raise ValueError("exp13: 파인더/포맷 링 검출 실패 "
                             "(자기기술 코드가 아니거나 규약 외 이미지)")
        if len(viable) > 1:
            # 침묵 오정정보다 명시적 실패 — exp12 이후 일관 원칙
            raise ValueError(f"exp13: 방향·기하 가설 모호 ({len(viable)}건)")

        geo, rot_k, mirrored, fmt = viable[0]
        R, hs = geo['radius'], geo['hex_size']
        ox, oy = geo['origin_x'], geo['origin_y']
        profile = PROFILE_BY_ID[fmt['profile']]
        bpc = PROFILES[profile]['bits_per_cell']
        nsym = nsym_for_level(fmt['ecc_level'])
        cap = message_capacity(R, bpc, nsym)
        cw_len = codeword_length(cap, nsym)

        ring_passes = [self.rings]
        if self.fallback_rings and self.fallback_rings != self.rings:
            ring_passes.append(self.fallback_rings)

        def finalize(res, rings):
            # 사다리·스윕 공통 마무리 — 길이 프리픽스 무결성 위반 =
            # 오정정 신호 → ReedSolomonError로 해당 결과 폐기
            msg = res['data']
            if fmt['scramble']:
                # exp15b: 인코더가 패딩 포함 메시지에 XOR한 LFSR 마스크 원복
                # (퇴화 가드는 디스크램블 *전* 코드워드 공간에서 이미 적용됨)
                msg = xor_scramble(msg)
            data_len = int.from_bytes(msg[:LEN_PREFIX], 'big')
            if not (0 <= data_len <= cap - LEN_PREFIX):
                raise ReedSolomonError(
                    f"길이 프리픽스 무결성 위반: {data_len}")
            res.update({
                'data': msg[LEN_PREFIX:LEN_PREFIX + data_len],
                'rings_used': rings,
                'radius': R,
                'hex_size': hs,
                'profile': profile,
                'bits_per_cell': bpc,
                'ecc_level': fmt['ecc_level'],
                'nsym': nsym,
                'orientation': {'rot_k': rot_k, 'mirrored': mirrored},
                'format': fmt,
            })
            return res

        last_err = None
        reads = []
        for rings in ring_passes:
            codeword, byte_conf = self._read_codeword(
                gray, thr, R, hs, bpc, ox, oy, rot_k, mirrored, cw_len, rings)
            reads.append((rings, codeword, byte_conf))
            try:
                return finalize(erasure_ladder_decode(
                    codeword, byte_conf, nsym, cap, budget=True,
                    reject_degenerate=True), rings)
            except ReedSolomonError as e:
                last_err = e

        if self.use_sweep:
            # exp11b 결함① 경로 — 만장일치-오류 패치 커버 (스윕 후보 예산은
            # spatial 기본값 nsym-2, 수락은 다중 가설 합의 — v11b와 동일)
            groups = self._spatial_groups(R, bpc, cw_len, nsym)
            for rings, codeword, byte_conf in reads:
                try:
                    return finalize(spatial_erasure_sweep(
                        codeword, byte_conf, nsym, cap, groups,
                        conf_fill=self.conf_fill), rings)
                except ReedSolomonError as e:
                    last_err = e
        raise last_err or ReedSolomonError("exp13 복호 실패")

    def decode(self, image, geometry: dict = None) -> bytes:
        return self.decode_with_stats(image, geometry=geometry)['data']
