"""
D_QR Phase 10 디코더 — 신뢰도 게이트 점진적 공개 (exp10)

복호 정책:
  stream C (coarse): 항상 시도. exp9 소거 사다리 그대로 적용,
                     실패는 예외 (코어 페이로드 없이는 코드 자체가 무의미).
  stream F (fine)  : 바이트 신뢰도(8비트 마진 최솟값) 수집 후 사다리 시도,
                     실패 시 예외 대신 None — 우아한 강등(graceful degradation).
                     저화질에서는 코어만, 고화질에서는 전체가 열리는
                     "점진적 공개"가 이 게이트 하나로 구현됨.

셀 판독(soft read)·소거 사다리는 exp9 구현을 import 재사용 — 복사 없음.
"""
import numpy as np
from PIL import Image

from src.core.bitutils import _bits_to_bytes
from src.core.softdecode import erasure_ladder_decode
from src.exp9_soft.decoder_v9 import DQRDecoderV9
from src.exp10_progressive.encoder_v10 import GRAINS, split_data_cells

from reedsolo import ReedSolomonError


class DQRDecoderV10:
    """
    D_QR Phase 10 디코더.

    사용 예:
        dec = DQRDecoderV10()
        stats = dec.decode_with_stats(img, radius=12, hex_size=32,
                                      origin_x=ox, origin_y=oy,
                                      nsym_c=76, nsym_f=51,
                                      coarse_len=Lc, fine_len=Lf,
                                      codeword_len_c=cwc, codeword_len_f=cwf)
    """

    def __init__(self, threshold: int = 128):
        self.threshold = threshold
        # 셀 단위 soft 판독은 v9 구현 그대로 — 합성(composition)으로 재사용
        self._reader = DQRDecoderV9(threshold=threshold)

    def _read_stream(self, pixels, cells, bpc, hex_size, ox, oy,
                     codeword_len):
        """한 grain의 셀들 판독 → (codeword bytes, 바이트별 신뢰도)."""
        bits, margins = [], []
        for q, r in cells:
            cb, cm = self._reader._read_cell_soft(
                pixels, q, r, bpc, hex_size, ox, oy)
            bits.extend(cb)
            margins.extend(cm)

        codeword = _bits_to_bytes(bits, codeword_len)
        # 바이트 신뢰도 = 그 바이트를 이루는 8비트 마진의 최솟값 (exp9와 동일)
        byte_conf = []
        for i in range(codeword_len):
            chunk = margins[i * 8:(i + 1) * 8]
            byte_conf.append(min(chunk) if chunk else 0)
        return codeword, byte_conf

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
        dual-grain 복호 + 통계.

        Returns:
          {'coarse': bytes, 'fine': bytes|None, 'fine_confidence': float,
           'coarse_mode': str, 'fine_mode': str|None}
          fine_confidence: stream F 바이트 신뢰도 평균을 0~1로 정규화
                           (마진 상한 = threshold 기준 128)
        """
        coarse_cells, fine_cells = split_data_cells(radius)
        pixels = np.array(image.convert('RGB'))

        # ── stream C: 항상 시도, 실패는 예외 그대로 전파 ──
        cw_c, conf_c = self._read_stream(
            pixels, coarse_cells, GRAINS['coarse']['bits_per_cell'],
            hex_size, origin_x, origin_y, codeword_len_c)
        res_c = erasure_ladder_decode(cw_c, conf_c, nsym_c, coarse_len)

        # ── stream F: 신뢰도 게이트 — 실패 시 None으로 우아한 강등 ──
        cw_f, conf_f = self._read_stream(
            pixels, fine_cells, GRAINS['fine']['bits_per_cell'],
            hex_size, origin_x, origin_y, codeword_len_f)
        fine_confidence = (sum(conf_f) / len(conf_f) / 128.0) if conf_f else 0.0
        fine_confidence = min(1.0, fine_confidence)

        try:
            res_f = erasure_ladder_decode(cw_f, conf_f, nsym_f, fine_len)
            fine_data, fine_mode = res_f['data'], res_f['mode']
        except ReedSolomonError:
            fine_data, fine_mode = None, None

        return {
            'coarse': res_c['data'],
            'fine': fine_data,
            'fine_confidence': fine_confidence,
            'coarse_mode': res_c['mode'],
            'fine_mode': fine_mode,
        }

    def decode(self, image: Image.Image, **kwargs) -> tuple:
        """(coarse, fine|None) 튜플 반환 단축형."""
        stats = self.decode_with_stats(image, **kwargs)
        return stats['coarse'], stats['fine']
