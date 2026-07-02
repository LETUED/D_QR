"""
exp16 — 검출 파이프라인 v0 (임의 위치·회전·완만 원근 → 자기기술 디코딩).

FTO: 본 패키지는 위치 추정 전용, 신뢰도·소거 판정과 무관 (모듈 docstring 참조).
"""
from src.exp16_detect.pipeline import (DetectionError, detect_and_decode,
                                       detect_hypotheses)

__all__ = ['DetectionError', 'detect_and_decode', 'detect_hypotheses']
