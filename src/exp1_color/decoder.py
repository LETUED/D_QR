"""
D_QR Phase 1 디코더
PIL Image → bytes

Phase 1 제약:
  - 이미지가 완벽하게 정렬됐다고 가정 (원근 보정 없음)
  - 그리드 크기와 모듈 픽셀 크기를 파라미터로 받음
  - 각 모듈의 중앙 픽셀 샘플링 → 팔레트 매핑
"""
import math
import numpy as np
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

from src.core.constants import (
    BITS_PER_MODULE,
    rgb_to_nearest_symbol,
)
from src.core.bitutils import _build_reserved_mask

# quiet zone 기본값 (renderer와 일치해야 함)
_DEFAULT_QUIET_ZONE = 4


class DQRDecoder:
    """
    D_QR Phase 1 디코더.

    사용 예:
        dec = DQRDecoder()
        data = dec.decode(img, grid_size=37, module_px=20, nsym=n, data_len=L)
    """

    def __init__(self, quiet_zone: int = _DEFAULT_QUIET_ZONE):
        self.quiet_zone = quiet_zone

    def decode(self, image: Image.Image,
               grid_size: int,
               module_px: int,
               nsym: int,
               data_len: int,
               codeword_len: int = None) -> bytes:
        """
        D_QR 이미지에서 원본 데이터 복원.

        Args:
            image     : PIL.Image.Image (RGB)
            grid_size : 그리드 크기 S (인코더 반환값)
            module_px : 모듈당 픽셀 크기 (렌더러와 동일해야 함)
            nsym      : RS ECC 심볼 수 (인코더 반환값)
            data_len  : 원본 데이터 길이 (패딩 제거용)

        Returns:
            원본 bytes

        Raises:
            ReedSolomonError: 오류 복원 불가 시
        """
        S = grid_size
        px = module_px
        qz = self.quiet_zone
        reserved = _build_reserved_mask(S)

        img_rgb = image.convert('RGB')
        pixels = np.array(img_rgb)

        bits = []

        for row in range(S):
            for col in range(S):
                if reserved[row, col]:
                    continue

                # 모듈 중앙 픽셀 좌표
                cx = (col + qz) * px + px // 2
                cy = (row + qz) * px + px // 2

                r, g, b = int(pixels[cy, cx, 0]), int(pixels[cy, cx, 1]), int(pixels[cy, cx, 2])
                c1, c2, c3 = rgb_to_nearest_symbol(r, g, b)

                # (c1, c2, c3) → 7비트
                bits.append((c1 >> 2) & 1)
                bits.append((c1 >> 1) & 1)
                bits.append(c1 & 1)
                bits.append((c2 >> 1) & 1)
                bits.append(c2 & 1)
                bits.append((c3 >> 1) & 1)
                bits.append(c3 & 1)

        # 비트 → 바이트 변환
        n_bytes = len(bits) // 8
        codeword = bytearray()
        for i in range(n_bytes):
            byte = 0
            for bit in bits[i * 8: i * 8 + 8]:
                byte = (byte << 1) | bit
            codeword.append(byte)

        # codeword_len이 주어지면 해당 길이만 디코딩 (zero-padding 제외)
        if codeword_len is not None:
            codeword = codeword[:codeword_len]

        # Reed-Solomon 디코딩
        rsc = RSCodec(nsym)
        try:
            decoded, _, _ = rsc.decode(bytes(codeword))
        except ReedSolomonError as e:
            raise ReedSolomonError(f"오류 복원 실패 (손상이 너무 큼): {e}")

        # 원본 길이만큼 잘라서 반환 (zero-padding 제거)
        return bytes(decoded)[:data_len]
