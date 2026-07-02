"""
D_QR Phase 12 디코더 — 침묵 오정정 3층 방어 + 신뢰도 게이트 실시예 (exp12)

3층 방어:
  1층 — 사다리 상한 (nsym → nsym-2)      : src/core/softdecode.ladder_steps
  2층 — 재인코딩 대조 (CRC 아님)          : src/core/softdecode.reencode_mismatch_ok
  3층 — 신뢰도 임계 게이트 (이 파일)      : fine 스트림 평균 바이트 신뢰도가
        임계값 미만이면 RS 시도 자체를 생략하고 fine=None 반환.

3층 게이트의 신뢰도 메트릭 (FTO 제약 1 준수 — Honeywell US 10,235,547 회피):
  exp11 표 일치율(vote consensus) — 서브삼각형당 N개 샘플점을 전역 임계값으로
  하드 이진화한 뒤 (다수표 - 소수표)의 이산 표 개수만 사용.
  픽셀 밝기 연속량(밝기 마진·반사율 간극)은 게이트 판정에 일절 쓰지 않는다.

  주의: 소거 사다리의 '소거 순서' 정렬은 exp10과 동일하게 밝기 마진을
  유지한다 (exp11 중단 규칙 발동 — 메트릭 전환 보류 상태). 게이트는
  RS 시도 여부만 결정하므로 사다리 메트릭과 독립이다.

게이트 임계값 캘리브레이션 (benchmarks/exp12_miscorrection.py 실측):
  exp10 표준 코드(r=12, hex_size=32) 블러 스윕 0~10px, N=9 표결 기준
    마지막 정상 복호(blur 5.5px) vote 신뢰도 = 0.4199
    침묵 오정정 지점(blur 6.0px)  vote 신뢰도 = 0.3911
  → 임계값 = 두 값의 중간 0.405.
  주의: vote 신뢰도는 블러에 단조가 아니다 — 6.5px 이상에서 0.41~0.44
  고원이 형성돼 게이트를 통과한다 (exp11 블러 절벽과 동근원: t=0.5 외곽
  샘플이 이웃 번짐을 흡수해 표가 안정적으로 갈라짐). 이 고원 구간의
  오정정은 1층(사다리 상한)·2층(재인코딩 대조)이 결정론적으로 차단한다
  — 3층 게이트는 단독 방어가 아니라 심층 방어의 첫 관문.
  실측 근거: outputs/exp12/exp12_miscorrection_report.md
"""
import numpy as np
from PIL import Image

from src.core.softdecode import erasure_ladder_decode
from src.core.consensus import triangle_sample_points
from src.core.hex_grid import hex_sub_triangles
from src.exp10_progressive.decoder_v10 import DQRDecoderV10
from src.exp10_progressive.encoder_v10 import GRAINS, split_data_cells

from reedsolo import ReedSolomonError

# 3층 게이트 기본 임계값 — benchmarks/exp12_miscorrection.py 블러 스윕 실측 보정:
# 마지막 정상 복호(5.5px, 0.4199)와 침묵 오정정(6.0px, 0.3911)의 중간값
FINE_GATE_THRESHOLD = 0.405

# 게이트용 표결 샘플점 수 — exp11 ablation에서 채택된 값(N=9)과 동일
GATE_SAMPLE_POINTS = 9


class DQRDecoderV12:
    """
    D_QR Phase 12 디코더 — 3층 방어 점진적 공개.

    사용 예:
        dec = DQRDecoderV12()
        stats = dec.decode_with_stats(img, radius=12, hex_size=32,
                                      origin_x=ox, origin_y=oy,
                                      nsym_c=76, nsym_f=51,
                                      coarse_len=Lc, fine_len=Lf,
                                      codeword_len_c=cwc, codeword_len_f=cwf)
    """

    def __init__(self, threshold: int = 128,
                 gate_threshold: float = FINE_GATE_THRESHOLD,
                 n_samples: int = GATE_SAMPLE_POINTS):
        if n_samples % 2 == 0:
            raise ValueError(f"n_samples는 홀수여야 함 (동수 방지): {n_samples}")
        self.threshold = threshold
        self.gate_threshold = gate_threshold
        self.n_samples = n_samples
        # 셀 판독(사다리용 코드워드·마진)은 v10 경로 그대로 — 합성으로 재사용
        self._v10 = DQRDecoderV10(threshold=threshold)
        # 게이트 샘플점 좌표는 기하에만 의존 → 이미지 간 재사용 캐시
        self._coord_cache = {}

    # ── 3층 게이트: fine 스트림 vote 신뢰도 ────────────────────────────

    def _fine_sample_coords(self, radius, hex_size, ox, oy):
        key = (radius, hex_size, ox, oy, self.n_samples)
        if key in self._coord_cache:
            return self._coord_cache[key]

        _, fine_cells = split_data_cells(radius)
        xs, ys = [], []
        for q, r in fine_cells:
            for verts, _centroid in hex_sub_triangles(q, r, hex_size, ox, oy):
                for px, py in triangle_sample_points(verts, self.n_samples):
                    xs.append(px)
                    ys.append(py)

        n_tris = len(fine_cells) * GRAINS['fine']['bits_per_cell']
        coords = (np.array(xs).reshape(n_tris, self.n_samples),
                  np.array(ys).reshape(n_tris, self.n_samples))
        self._coord_cache[key] = coords
        return coords

    def fine_vote_confidence(self, image, radius, hex_size,
                             origin_x, origin_y, codeword_len_f) -> float:
        """
        fine 스트림 평균 바이트 신뢰도 (0~1) — exp11 표결 메트릭.

        샘플점별 하드 이진화(전역 임계 128) 후 비트 신뢰도 = |다수표-소수표|,
        바이트 신뢰도 = 8비트 표차의 최솟값, 반환값 = 바이트 평균 / N.
        밝기 연속량은 이진화 직후 버려진다 (FTO 제약 1 준수).
        """
        pixels = np.array(image.convert('RGB'), dtype=np.int32)
        gray = pixels.sum(axis=2) // 3
        img_h, img_w = gray.shape

        xs, ys = self._fine_sample_coords(radius, hex_size, origin_x, origin_y)
        xi = np.clip(np.rint(xs), 0, img_w - 1).astype(np.int64)
        yi = np.clip(np.rint(ys), 0, img_h - 1).astype(np.int64)

        dark_votes = (gray[yi, xi] < self.threshold).sum(axis=1)
        confs = np.abs(2 * dark_votes - self.n_samples)  # 이산 표차 (정수)

        # 코드워드를 이루는 바이트만 평가 (잔여 패딩 삼각형 제외)
        byte_conf = []
        for i in range(codeword_len_f):
            chunk = confs[i * 8:(i + 1) * 8]
            if chunk.size == 0:
                break
            byte_conf.append(int(chunk.min()))
        if not byte_conf:
            return 0.0
        return float(sum(byte_conf)) / len(byte_conf) / self.n_samples

    # ── 복호 본체 ──────────────────────────────────────────────────────

    def decode_with_stats(self, image: Image.Image,
                          radius: int,
                          hex_size: float,
                          origin_x: float,
                          origin_y: float,
                          nsym_c: int,
                          nsym_f: int,
                          coarse_len: int,
                          fine_len: int,
                          codeword_len_c: int,
                          codeword_len_f: int) -> dict:
        """
        dual-grain 복호 + 3층 방어 통계.

        Returns:
          {'coarse', 'coarse_mode',
           'fine', 'fine_mode',
           'fine_vote_confidence',  # 게이트 판정에 쓴 표결 신뢰도 (0~1)
           'gate_passed',           # 3층 게이트 통과 여부
           'rs_attempted',          # fine RS 시도 여부 (게이트 차단 시 False)
           'fine_verified'}         # 2층 재인코딩 대조 통과 여부 (None=미시도)
        """
        coarse_cells, fine_cells = split_data_cells(radius)
        pixels = np.array(image.convert('RGB'))

        # ── stream C: 항상 시도 (1·2층 방어는 사다리 기본값으로 적용) ──
        cw_c, conf_c = self._v10._read_stream(
            pixels, coarse_cells, GRAINS['coarse']['bits_per_cell'],
            hex_size, origin_x, origin_y, codeword_len_c)
        res_c = erasure_ladder_decode(cw_c, conf_c, nsym_c, coarse_len)

        # ── stream F: 3층 게이트 → (통과 시) 1·2층 방어 사다리 ──
        vote_conf = self.fine_vote_confidence(
            image, radius, hex_size, origin_x, origin_y, codeword_len_f)

        fine_data, fine_mode, fine_verified = None, None, None
        gate_passed = vote_conf >= self.gate_threshold
        rs_attempted = False

        if gate_passed:
            cw_f, conf_f = self._v10._read_stream(
                pixels, fine_cells, GRAINS['fine']['bits_per_cell'],
                hex_size, origin_x, origin_y, codeword_len_f)
            rs_attempted = True
            try:
                res_f = erasure_ladder_decode(cw_f, conf_f, nsym_f, fine_len)
                fine_data = res_f['data']
                fine_mode = res_f['mode']
                fine_verified = res_f['verified']
            except ReedSolomonError:
                pass  # 사다리 소진 또는 2층 대조 불일치 → 우아한 강등

        return {
            'coarse': res_c['data'],
            'coarse_mode': res_c['mode'],
            'fine': fine_data,
            'fine_mode': fine_mode,
            'fine_vote_confidence': vote_conf,
            'gate_passed': gate_passed,
            'rs_attempted': rs_attempted,
            'fine_verified': fine_verified,
        }

    def decode(self, image: Image.Image, **kwargs) -> tuple:
        """(coarse, fine|None) 튜플 반환 단축형."""
        stats = self.decode_with_stats(image, **kwargs)
        return stats['coarse'], stats['fine']
