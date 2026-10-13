"""
D_QR Phase 11b 디코더 — exp11 중단규칙 3대 결함 해소판 (exp11b)

exp11(decoder_v11)은 보존하고 이 파일에 v11b를 추가한다. exp11 ablation에서
중단규칙을 발동시킨 3대 결함과 각각의 해법:

  결함① 만장일치-오류 (균일 패치, margin 대비 최대 48%p 열세)
    → 공간 소거 스윕 + 합의 수락 (src/exp11_consensus/spatial.py).
      신뢰도 사다리 실패 시 공간 연속 쐐기 그룹(그룹당 바이트 ≤ (nsym-2)/2)
      단위의 위치 가설 소거 시행. 순수 시행은 신뢰도 메트릭 자체를 쓰지
      않으므로 Honeywell 울타리 바깥 + Forney GMD(1966) 선행기술 위.
      수락은 다중 가설 합의(≥3, 2위의 2배, 순수 시행 포함, 후보 예산
      nsym-2)를 통과해야 한다 — 첫 수락 방식의 오복호 경로 차단.

  결함② 블러 절벽 (9.5px에서 t=0.5 외곽 샘플이 이웃 번짐 흡수)
    → 내부 링 폴백 패스 (consensus.INNER_RINGS = (0.3, 0.15)).
      1차 패스는 exp11 원안 배치(패치 부분 덮임의 표 분열 감지에 필수),
      실패 시 2차 패스가 N=9 전 샘플을 무게중심 근방(t≤0.3)에 두고
      재판독해 이웃 번짐 흡수를 차단한다.

  결함③ 청크 errata 예산 (exp12 잔존 오복호 84건 시그니처)
    → 사다리에 budget=True 이중 방어 적용 (src/core/softdecode):
      3-a 청크별 2·오류+소거 ≤ nsym 유일 복호 반경 검사,
      3-b 최종 소거 단계 BM 오류 탐색 금지(only_erasures).
      스윕 후보는 더 엄격한 nsym-2 예산 (위 결함① 참조).

exp15b 추가 — 4층 퇴화 코드워드 가드 (reject_degenerate, 기본 on):
    exp15 기준선 비교의 D_QR 미검출 오복호 7건은 전부 "전부-0 코드워드
    퇴화" 단일 모드 — RS 선형성상 전부-0은 항상 유효 코드워드라 3층
    방어가 전부 무력했다. 복호 결과가 전부-0/상수 메시지면 거부한다
    (softdecode.is_degenerate_message — 사다리·스윕 양 경로 적용).
    descramble 옵션: encoder_v9의 scramble(전부-0 회피 LFSR 마스크,
    기본 off — v9 호환) 사용 시 복호 후 동일 마스크 XOR로 원복.

FTO (exp11과 동일하게 준수): 신뢰도 = 이산 표 개수(다수표-소수표)만 사용,
샘플 이진화는 전역 임계(고정 128/전역 Otsu), 밝기 연속량·국소 필터·ML 금지.
"""
from reedsolo import ReedSolomonError

from src.core.consensus import (DEFAULT_RINGS, INNER_RINGS,
                                triangle_sample_points)
from src.core.hex_grid import (build_hex_grid, build_reserved,
                               hex_primary_triangles, hex_sub_triangles)
from src.core.scramble import xor_scramble
from src.core.softdecode import erasure_ladder_decode
from src.exp11_consensus.decoder_v11 import DQRDecoderV11
from src.exp11_consensus.spatial import (build_spatial_groups,
                                         spatial_erasure_sweep)

import numpy as np

DEFAULT_SAMPLE_POINTS = 9   # exp11 ablation 채택값 유지


class DQRDecoderV11B(DQRDecoderV11):
    """
    D_QR Phase 11b 디코더 — vote consensus + 3대 결함 해소.

    링 배치는 2패스: 1차 외곽 링(exp11 원안 — 패치 부분 덮임의 표 분열
    감지가 erasure 순위에 필수), 실패 시 2차 내부 링(블러 번짐 차단).
    두 패스 모두 동일한 표결·전역 임계 원리이며 어느 쪽도 밝기 연속량을
    쓰지 않는다. 2패스 결과는 사다리(1·2층 + errata 예산)와 공간 소거
    스윕(재인코딩 대조 + 예산)을 통과해야만 채택되므로 패스 추가가
    오수락 경로를 만들지 않는다.

    사용 예 (인터페이스는 v11과 동일):
        dec = DQRDecoderV11B()
        data = dec.decode(img, radius=12, hex_size=32, bits_per_cell=6,
                          origin_x=ox, origin_y=oy,
                          nsym=76, data_len=L, codeword_len=cw_len)
    """

    def __init__(self, threshold: int = 128,
                 n_samples: int = DEFAULT_SAMPLE_POINTS,
                 use_otsu: bool = False,
                 rings: tuple = DEFAULT_RINGS,
                 fallback_rings: tuple = INNER_RINGS,
                 use_sweep: bool = True,
                 conf_fill: bool = True,
                 budget_slack: int = 0,
                 reject_degenerate: bool = True,
                 descramble: bool = False):
        super().__init__(threshold=threshold, n_samples=n_samples,
                         use_otsu=use_otsu)
        for rg in (rings,) + ((fallback_rings,) if fallback_rings else ()):
            if n_samples > 1 + 6 * len(rg):
                raise ValueError(f"n_samples={n_samples} > 링 수용량 "
                                 f"{1 + 6 * len(rg)}: {rg}")
        self.rings = tuple(rings)
        self.fallback_rings = tuple(fallback_rings) if fallback_rings else None
        self.use_sweep = use_sweep
        self.conf_fill = conf_fill
        self.budget_slack = budget_slack
        self.reject_degenerate = reject_degenerate
        self.descramble = descramble
        self._group_cache = {}

    # 결함② — v11의 좌표 생성에 링 배치 파라미터 추가 (캐시 키에 포함)
    def _sample_coords_rings(self, radius, hex_size, bits_per_cell,
                             ox, oy, rings):
        key = (radius, hex_size, bits_per_cell, ox, oy,
               self.n_samples, rings)
        if key in self._coord_cache:
            return self._coord_cache[key]

        reserved = build_reserved(radius)
        data_cells = [c for c in build_hex_grid(radius) if c not in reserved]
        tri_fn = (hex_primary_triangles if bits_per_cell == 6
                  else hex_sub_triangles)

        xs, ys = [], []
        for q, r in data_cells:
            for verts, _centroid in tri_fn(q, r, hex_size, ox, oy):
                for px, py in triangle_sample_points(verts, self.n_samples,
                                                     rings=rings):
                    xs.append(px)
                    ys.append(py)

        n_tris = len(data_cells) * bits_per_cell
        coords = (np.array(xs).reshape(n_tris, self.n_samples),
                  np.array(ys).reshape(n_tris, self.n_samples))
        self._coord_cache[key] = coords
        return coords

    def _sample_coords(self, radius, hex_size, bits_per_cell, ox, oy):
        # v11의 _read_codeword가 호출 — 현재 패스의 링 배치를 사용
        return self._sample_coords_rings(radius, hex_size, bits_per_cell,
                                         ox, oy, self._active_rings)

    def _spatial_groups(self, radius, bits_per_cell, codeword_len, nsym):
        key = (radius, bits_per_cell, codeword_len, nsym)
        if key not in self._group_cache:
            self._group_cache[key] = build_spatial_groups(
                radius, bits_per_cell, codeword_len, nsym)
        return self._group_cache[key]

    def decode_with_stats(self, image, radius, hex_size, bits_per_cell,
                          origin_x, origin_y, nsym, data_len,
                          codeword_len) -> dict:
        """
        복호 순서:
          ① 외곽 링 판독 → 신뢰도 사다리(+errata 예산+퇴화 가드)
          ② 내부 링 판독 → 신뢰도 사다리(+errata 예산+퇴화 가드) (결함② 경로)
          ③ 외곽 링 판독 기준 공간 소거 스윕(+퇴화 가드)         (결함① 경로)
        Returns: v11 키 + {'rings_used'} (+스윕 채택 시 'sweep_trial' 등)
        """
        res = self._decode_core(image, radius, hex_size, bits_per_cell,
                                origin_x, origin_y, nsym, data_len,
                                codeword_len)
        if self.descramble:
            # encoder_v9(scramble=True) 대응 — 마스크는 위치 고정이라
            # data_len 절단 후에 XOR해도 동일하다
            res['data'] = xor_scramble(res['data'])
        return res

    def _decode_core(self, image, radius, hex_size, bits_per_cell,
                     origin_x, origin_y, nsym, data_len, codeword_len):
        ring_passes = [self.rings]
        if self.fallback_rings and self.fallback_rings != self.rings:
            ring_passes.append(self.fallback_rings)

        reads = []
        for rings in ring_passes:
            self._active_rings = rings
            codeword, byte_conf = self._read_codeword(
                image, radius, hex_size, bits_per_cell,
                origin_x, origin_y, codeword_len)
            reads.append((rings, codeword, byte_conf))
            try:
                res = erasure_ladder_decode(
                    codeword, byte_conf, nsym, data_len,
                    budget=True, budget_slack=self.budget_slack,
                    reject_degenerate=self.reject_degenerate)
                res['rings_used'] = rings
                return res
            except ReedSolomonError:
                continue

        if not self.use_sweep:
            raise ReedSolomonError("v11b 사다리 전 패스 실패 (스윕 비활성)")

        # 결함① — 공간 소거 스윕 (만장일치-오류 패치 커버)
        groups = self._spatial_groups(radius, bits_per_cell,
                                      codeword_len, nsym)
        last_err = None
        for rings, codeword, byte_conf in reads:
            try:
                # 스윕 후보 예산은 nsym-2 (spatial 기본값) — 사다리(slack=0)와
                # 달리 가설 시행이라 여력 강제가 오복호 후보만 정확히 걸러냄
                res = spatial_erasure_sweep(
                    codeword, byte_conf, nsym, data_len, groups,
                    conf_fill=self.conf_fill,
                    reject_degenerate=self.reject_degenerate)
                res['rings_used'] = rings
                return res
            except ReedSolomonError as e:
                last_err = e
        raise last_err
