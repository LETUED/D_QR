"""
D_QR Phase 11 디코더 — 표 일치율(vote consensus) soft-decision (exp11)

concept/11_research_roadmap.md §11.1 구현 — Honeywell US 10,235,547 패밀리 회피:
  exp9 (decoder_v9): 신뢰도 = |brightness - threshold|
                     → 밝기 마진 = 반사율 간극(erasure gap) 계열 연속량 (FTO 위험)
  exp11 (이 파일)  : 신뢰도 = 다수표 - 소수표
                     → **이산 표 개수이며, 반사율 간극이 아니다.**

  (서브)삼각형당 N개 샘플점(무게중심 + 내부 보간점)을 각각
  전역 임계값(고정 128 또는 전역 Otsu)으로 하드 이진화 → 비트 = 다수결.
  픽셀 밝기 연속량은 신뢰도 계산에 절대 사용하지 않는다.

인코딩 측은 변경 없음 — exp9의 encoder_v9/renderer_v9를 그대로 사용한다.
소거 사다리도 exp9와 동일 구조 (src/core/softdecode.erasure_ladder_decode —
exp12부터 1층 사다리 상한(nsym-2)·2층 재인코딩 대조가 기본 적용됨).
"""
import numpy as np
from PIL import Image

from src.core.bitutils import _bits_to_bytes
from src.core.softdecode import erasure_ladder_decode
from src.core.consensus import triangle_sample_points, global_otsu_threshold
from src.core.hex_grid import (build_hex_grid, build_reserved,
                        hex_primary_triangles, hex_sub_triangles)

# exp11 ablation(benchmarks/exp11_ablation.py) N∈{5,7,9,11,13} 스윕에서
# 총 성공률 최고(3342/4000점)로 채택된 값.
# 단, 중단 규칙 발동(국소 오염·blur 9.5px에서 margin 대비 >10%p 열세)으로
# 파이프라인 기본 디코더 전환은 보류 상태 — outputs/exp11/exp11_ablation_report.md 참조.
DEFAULT_SAMPLE_POINTS = 9


class DQRDecoderV11:
    """
    D_QR Phase 11 디코더 — vote consensus.

    사용 예:
        dec = DQRDecoderV11(n_samples=7)
        data = dec.decode(img, radius=12, hex_size=32, bits_per_cell=6,
                          origin_x=ox, origin_y=oy,
                          nsym=76, data_len=L, codeword_len=cw_len)
    """

    def __init__(self, threshold: int = 128,
                 n_samples: int = DEFAULT_SAMPLE_POINTS,
                 use_otsu: bool = False):
        if n_samples % 2 == 0:
            raise ValueError(f"n_samples는 홀수여야 함 (다수결 동수 방지): {n_samples}")
        self.threshold = threshold
        self.n_samples = n_samples
        self.use_otsu = use_otsu
        # 샘플점 픽셀 좌표는 기하 파라미터에만 의존 → 이미지 간 재사용 캐시
        self._coord_cache = {}

    def _sample_coords(self, radius, hex_size, bits_per_cell, ox, oy):
        """전체 데이터 (서브)삼각형의 샘플점 좌표 배열 (n_tris, n_samples)."""
        key = (radius, hex_size, bits_per_cell, ox, oy, self.n_samples)
        if key in self._coord_cache:
            return self._coord_cache[key]

        reserved = build_reserved(radius)
        data_cells = [c for c in build_hex_grid(radius) if c not in reserved]
        tri_fn = (hex_primary_triangles if bits_per_cell == 6
                  else hex_sub_triangles)

        xs, ys = [], []
        for q, r in data_cells:
            for verts, _centroid in tri_fn(q, r, hex_size, ox, oy):
                for px, py in triangle_sample_points(verts, self.n_samples):
                    xs.append(px)
                    ys.append(py)

        n_tris = len(data_cells) * bits_per_cell
        coords = (np.array(xs).reshape(n_tris, self.n_samples),
                  np.array(ys).reshape(n_tris, self.n_samples))
        self._coord_cache[key] = coords
        return coords

    def _read_codeword(self, image, radius, hex_size, bits_per_cell,
                       origin_x, origin_y, codeword_len):
        """이미지 전체 판독 → (codeword bytes, 바이트별 표차 신뢰도)."""
        pixels = np.array(image.convert('RGB'), dtype=np.int32)
        gray = pixels.sum(axis=2) // 3  # v9와 동일한 정수 평균 밝기
        img_h, img_w = gray.shape

        thr = global_otsu_threshold(gray) if self.use_otsu else self.threshold

        xs, ys = self._sample_coords(radius, hex_size, bits_per_cell,
                                     origin_x, origin_y)
        xi = np.clip(np.rint(xs), 0, img_w - 1).astype(np.int64)
        yi = np.clip(np.rint(ys), 0, img_h - 1).astype(np.int64)

        # 샘플점별 하드 이진화 (전역 임계) — 이 지점 이후 밝기값은 버려지고
        # 표 개수만 남는다. 신뢰도는 이산 표 개수 차이며, 반사율 간극이 아님.
        dark = gray[yi, xi] < thr                  # (n_tris, n_samples) bool
        dark_votes = dark.sum(axis=1)              # 삼각형별 어두움 표 수
        n = self.n_samples
        bits = (2 * dark_votes > n).astype(int)    # 비트 = 다수결
        confs = np.abs(2 * dark_votes - n)         # 다수표 - 소수표 (정수)

        codeword = _bits_to_bytes(list(bits), codeword_len)
        # 바이트 신뢰도 = 그 바이트를 이루는 8비트 표차의 최솟값
        byte_conf = []
        for i in range(codeword_len):
            chunk = confs[i * 8:(i + 1) * 8]
            byte_conf.append(int(chunk.min()) if chunk.size else 0)
        return codeword, byte_conf

    def decode_with_stats(self, image: Image.Image,
                          radius: int,
                          hex_size: float,
                          bits_per_cell: int,
                          origin_x: float,
                          origin_y: float,
                          nsym: int,
                          data_len: int,
                          codeword_len: int) -> dict:
        """
        vote consensus 복호 + 통계.
        Returns: {'data', 'mode', 'erasures_used', 'errata_count'}
        """
        codeword, byte_conf = self._read_codeword(
            image, radius, hex_size, bits_per_cell,
            origin_x, origin_y, codeword_len)

        return erasure_ladder_decode(codeword, byte_conf, nsym, data_len)

    def decode(self, image: Image.Image, **kwargs) -> bytes:
        return self.decode_with_stats(image, **kwargs)['data']
