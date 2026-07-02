"""
D_QR Phase 9 디코더 — soft-decision 소거 복호 (exp9)

concept/09_design_consolidation.md 해결방안 2 구현:
  기존(v6): bit = brightness < threshold  → 밝기 마진을 즉시 버림 (hard decision)
  exp9    : 마진 |brightness - threshold| = 신뢰도로 보존
            마진이 작은(애매한) 바이트를 RS 소거(erasure)로 표시

  RS 정정 한계: 2·오류 + 소거 ≤ nsym
    오류만        → 최대 nsym/2 정정
    위치 아는 소거 → 최대 nsym   정정 (2배)

복호 전략(소거 사다리): hard 시도 → 실패 시 저신뢰 바이트를
nsym/3 → 2nsym/3 → nsym 개씩 소거로 표시하며 재시도.
"""
import numpy as np
from PIL import Image
from reedsolo import RSCodec

from src.core.bitutils import _bits_to_bytes
from src.core.softdecode import erasure_ladder_decode
from src.core.hex_grid import (build_hex_grid, build_reserved,
                        hex_primary_triangles, hex_sub_triangles)


class DQRDecoderV9:
    """
    D_QR Phase 9 디코더 — soft-decision.

    사용 예:
        dec = DQRDecoderV9()
        data = dec.decode(img, radius=18, hex_size=32, bits_per_cell=6,
                          origin_x=ox, origin_y=oy,
                          nsym=76, data_len=L, codeword_len=cw_len)
    """

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def _read_cell_soft(self, pixels, q, r, bits_per_cell, hex_size, ox, oy):
        """셀 판독: (비트 리스트, 비트별 마진 리스트) 반환."""
        img_h, img_w = pixels.shape[:2]
        if bits_per_cell == 6:
            tris = hex_primary_triangles(q, r, hex_size, ox, oy)
        else:
            tris = hex_sub_triangles(q, r, hex_size, ox, oy)

        cell_bits, cell_margins = [], []
        for _verts, centroid in tris:
            tx = max(0, min(img_w - 1, int(round(centroid[0]))))
            ty = max(0, min(img_h - 1, int(round(centroid[1]))))
            brightness = (int(pixels[ty, tx, 0]) + int(pixels[ty, tx, 1])
                          + int(pixels[ty, tx, 2])) // 3
            cell_bits.append(1 if brightness < self.threshold else 0)
            cell_margins.append(abs(brightness - self.threshold))
        return cell_bits, cell_margins

    def _read_codeword(self, image, radius, hex_size, bits_per_cell,
                       origin_x, origin_y, codeword_len):
        """이미지 전체 판독 → (codeword bytes, 바이트별 신뢰도)."""
        reserved = build_reserved(radius)
        data_cells = [c for c in build_hex_grid(radius) if c not in reserved]
        pixels = np.array(image.convert('RGB'))

        bits, margins = [], []
        for q, r in data_cells:
            cb, cm = self._read_cell_soft(pixels, q, r, bits_per_cell,
                                          hex_size, origin_x, origin_y)
            bits.extend(cb)
            margins.extend(cm)

        codeword = _bits_to_bytes(bits, codeword_len)
        # 바이트 신뢰도 = 그 바이트를 이루는 8비트 마진의 최솟값
        byte_conf = []
        for i in range(codeword_len):
            chunk = margins[i * 8:(i + 1) * 8]
            byte_conf.append(min(chunk) if chunk else 0)
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
        soft-decision 복호 + 통계.
        Returns: {'data', 'mode', 'erasures_used', 'errata_count'}
        """
        codeword, byte_conf = self._read_codeword(
            image, radius, hex_size, bits_per_cell,
            origin_x, origin_y, codeword_len)

        # 소거 사다리 로직은 exp10과 공용 → src/core/softdecode로 추출됨
        return erasure_ladder_decode(codeword, byte_conf, nsym, data_len)

    def decode(self, image: Image.Image, **kwargs) -> bytes:
        return self.decode_with_stats(image, **kwargs)['data']

    def decode_hard_only(self, image: Image.Image,
                         radius: int, hex_size: float, bits_per_cell: int,
                         origin_x: float, origin_y: float,
                         nsym: int, data_len: int, codeword_len: int) -> bytes:
        """비교 실험용: 소거 없이 hard decision만 (v6 동작 재현)."""
        codeword, _ = self._read_codeword(
            image, radius, hex_size, bits_per_cell,
            origin_x, origin_y, codeword_len)
        decoded, _, _ = RSCodec(nsym).decode(codeword)
        return bytes(decoded)[:data_len]
