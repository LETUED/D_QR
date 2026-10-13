"""
D_QR exp20 디코더 — 점진적 공개 게이트 FTO 이식 (밝기 마진 → 표 일치율)

[왜 이 모듈이 존재하나 — FTO]
exp10 dual-grain 디코더(decoder_v10)의 점진적 공개 게이트는 전적으로 '밝기
마진'에 의존했다: 셀 판독 신뢰도(byte_conf)와 fine 공개 신뢰도가 모두
픽셀 밝기 연속량이었다. 이는 Honeywell 패밀리(US 10,235,547 등)가 선점한
반사율 기반 신뢰도→RS 소거 메트릭과 동일 구조로, exp11/11b/13이 표 일치율
(vote consensus, 이산 표 개수)로 전면 교체한 바로 그 경로다. 그러나 dual-grain
게이트만 이식에서 누락되어 있었다(2026-06-13 발견, concept/11 §발견).

본 모듈은 게이트를 표 일치율로 이식한다:
  · 셀 비트·신뢰도 = N점 하드 이진화 다수결 + 표차 |2·dark−n| (이산 표 개수)
  · fine 공개 신뢰도(progressive gate) = fine 셀 표차 평균 / N (밝기 아님)
  · 소거 사다리는 exp11b/13과 동일 무오복호 방어(budget+퇴화 가드)
인코딩(encoder_v10)·렌더링(renderer_v10)은 불변 — 디코더 메트릭만 교체.

[점진적 공개 의미 보존]
나쁜 센서(저해상·블러) → fine 셀 표결 합의 약화 → 소거 증가 → fine RS 실패
→ None(coarse만 공개). 좋은 센서 → 강한 합의 → fine 복호. 게이트가 '센서
품질'을 이산 표결로 읽어 점진적 공개를 구현 (밝기 마진 없이).

FTO: 신뢰도·소거 판정에 reflectance/contrast/erasure-gap 계열 양 불사용
     (consensus.majority_vote = 이산 표 개수). 전역 임계만 사용(국소 적응 금지).
"""
import numpy as np
from PIL import Image
from reedsolo import ReedSolomonError

from src.core.bitutils import _bits_to_bytes
from src.core.consensus import global_otsu_threshold, triangle_sample_points
from src.core.hex_grid import hex_primary_triangles, hex_sub_triangles
from src.core.softdecode import erasure_ladder_decode
from src.exp10_progressive.encoder_v10 import GRAINS

DEFAULT_SAMPLE_POINTS = 9   # exp11과 동일 (홀수, 다수결 동수 방지)


class DQRDecoderV20:
    """
    dual-grain 점진적 공개 디코더 (표 일치율 게이트).

    decode_with_stats 시그니처는 decoder_v10과 동일(드롭인 교체) —
    {coarse, fine|None, fine_consensus(0~1), coarse_mode, fine_mode}.
    fine_consensus는 밝기 마진이 아닌 '표차 평균/N'(이산 표결 기반 게이트).
    """

    def __init__(self, threshold: int = 128,
                 n_samples: int = DEFAULT_SAMPLE_POINTS,
                 use_otsu: bool = False,
                 budget_slack: int = 0,
                 reject_degenerate: bool = True):
        if n_samples % 2 == 0:
            raise ValueError(f"n_samples는 홀수여야 함 (다수결 동수 방지): {n_samples}")
        self.threshold = threshold
        self.n_samples = n_samples
        self.use_otsu = use_otsu
        self.budget_slack = budget_slack
        self.reject_degenerate = reject_degenerate
        self._coord_cache = {}

    def _sample_coords(self, cells_key, cells, bpc, hex_size, ox, oy):
        """grain 셀들의 (서브)삼각형 샘플점 좌표 (n_tris, n_samples)."""
        key = (cells_key, hex_size, bpc, ox, oy, self.n_samples)
        if key in self._coord_cache:
            return self._coord_cache[key]
        tri_fn = hex_primary_triangles if bpc == 6 else hex_sub_triangles
        xs, ys = [], []
        for q, r in cells:
            for verts, _centroid in tri_fn(q, r, hex_size, ox, oy):
                for px, py in triangle_sample_points(verts, self.n_samples):
                    xs.append(px)
                    ys.append(py)
        n_tris = len(cells) * bpc
        coords = (np.array(xs).reshape(n_tris, self.n_samples),
                  np.array(ys).reshape(n_tris, self.n_samples))
        self._coord_cache[key] = coords
        return coords

    def _read_stream(self, gray, thr, cells_key, cells, bpc, hex_size,
                     ox, oy, codeword_len):
        """한 grain 판독 → (codeword, byte_conf 이산 표차, 삼각형별 표차 배열).

        밝기값은 하드 이진화 직후 버려지고 표 개수만 남는다 (exp11과 동일).
        """
        img_h, img_w = gray.shape
        xs, ys = self._sample_coords(cells_key, cells, bpc, hex_size, ox, oy)
        xi = np.clip(np.rint(xs), 0, img_w - 1).astype(np.int64)
        yi = np.clip(np.rint(ys), 0, img_h - 1).astype(np.int64)

        dark = gray[yi, xi] < thr                  # (n_tris, n_samples) bool
        dark_votes = dark.sum(axis=1)
        n = self.n_samples
        bits = (2 * dark_votes > n).astype(int)    # 다수결 비트
        confs = np.abs(2 * dark_votes - n)         # 다수표 - 소수표 (이산)

        codeword = _bits_to_bytes(list(bits), codeword_len)
        byte_conf = []
        for i in range(codeword_len):
            chunk = confs[i * 8:(i + 1) * 8]
            byte_conf.append(int(chunk.min()) if chunk.size else 0)
        return codeword, byte_conf, confs

    def decode_with_stats(self, image: Image.Image, radius: int,
                          hex_size: float, origin_x: float, origin_y: float,
                          nsym_c: int, nsym_f: int, coarse_len: int,
                          fine_len: int, codeword_len_c: int,
                          codeword_len_f: int) -> dict:
        """dual-grain 복호 + 표 일치율 게이트 통계."""
        # split_data_cells는 인코더와 동일 분할 — 동결 보존 위해 지연 임포트
        from src.exp10_progressive.encoder_v10 import split_data_cells
        coarse_cells, fine_cells = split_data_cells(radius)

        pixels = np.array(image.convert('RGB'), dtype=np.int32)
        gray = pixels.sum(axis=2) // 3             # v9/v11과 동일 정수 평균
        thr = global_otsu_threshold(gray) if self.use_otsu else self.threshold

        bpc_c = GRAINS['coarse']['bits_per_cell']
        bpc_f = GRAINS['fine']['bits_per_cell']

        # ── stream C: 항상 시도, 실패는 예외 전파 (코어 없이는 무의미) ──
        cw_c, conf_c, _ = self._read_stream(
            gray, thr, ('coarse', radius), coarse_cells, bpc_c,
            hex_size, origin_x, origin_y, codeword_len_c)
        res_c = erasure_ladder_decode(
            cw_c, conf_c, nsym_c, coarse_len,
            budget=True, budget_slack=self.budget_slack,
            reject_degenerate=self.reject_degenerate)

        # ── stream F: 표 일치율 게이트 — 실패 시 None 우아한 강등 ──
        cw_f, conf_f, confs_f = self._read_stream(
            gray, thr, ('fine', radius), fine_cells, bpc_f,
            hex_size, origin_x, origin_y, codeword_len_f)
        # 게이트 신뢰도 = fine 삼각형 표차 평균 / N (이산 표결, 밝기 아님)
        fine_consensus = (float(confs_f.mean()) / self.n_samples
                          if confs_f.size else 0.0)

        try:
            res_f = erasure_ladder_decode(
                cw_f, conf_f, nsym_f, fine_len,
                budget=True, budget_slack=self.budget_slack,
                reject_degenerate=self.reject_degenerate)
            fine_data, fine_mode = res_f['data'], res_f['mode']
        except ReedSolomonError:
            fine_data, fine_mode = None, None

        return {
            'coarse': res_c['data'],
            'fine': fine_data,
            'fine_consensus': fine_consensus,
            'coarse_mode': res_c['mode'],
            'fine_mode': fine_mode,
        }

    def decode(self, image: Image.Image, **kwargs) -> tuple:
        """(coarse, fine|None) 튜플 단축형."""
        stats = self.decode_with_stats(image, **kwargs)
        return stats['coarse'], stats['fine']
