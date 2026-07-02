"""
공유 비트/예약영역 유틸 — exp1~exp9 인코더/디코더가 공통 사용.

기존에 encoder, encoder_v2/v4/v5/v6/v7/v8에 중복 정의되어 있던
헬퍼를 단일 모듈로 추출 (구현 동일, 기능 변경 없음).
"""
import numpy as np

from src.core.constants import FINDER_SIZE, SEPARATOR


def _build_reserved_mask(S: int) -> np.ndarray:
    """파인더 + 구분자 + 타이밍 패턴 위치를 True로 표시 (정사각 그리드)."""
    reserved = np.zeros((S, S), dtype=bool)
    fp = FINDER_SIZE + SEPARATOR  # 8

    reserved[:fp, :fp] = True           # 파인더 좌상
    reserved[:fp, S - fp:] = True       # 파인더 우상
    reserved[S - fp:, :fp] = True       # 파인더 좌하
    reserved[6, fp - 1: S - fp + 1] = True   # 타이밍 수평
    reserved[fp - 1: S - fp + 1, 6] = True   # 타이밍 수직

    return reserved


def _max_data_for_budget(total_bytes: int, nsym: int, data_per_block: int) -> int:
    """total_bytes 예산 내 최대 데이터 바이트 수 (reedsolo 블록 청킹 정확 반영)."""
    best = 0
    for n_blocks in range(1, 500):
        candidate = min(total_bytes - n_blocks * nsym, n_blocks * data_per_block)
        if candidate <= 0:
            break
        if candidate > best:
            best = candidate
        else:
            break
    return best


def _bytes_to_bits(data: bytes) -> list:
    bits = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _bits_to_bytes(bits: list, n_bytes: int) -> bytes:
    result = bytearray(n_bytes)
    for i in range(min(len(bits), n_bytes * 8)):
        if bits[i]:
            result[i // 8] |= (1 << (7 - (i % 8)))
    return bytes(result)
