"""
exp17 — 물리 기반 카메라 열화 모델 (인쇄→촬영 간극 합성 근사).

FTO: 입력 장면 생성 전용. 소거·신뢰도 판정 경로와 무관 (degrade 모듈 docstring).
"""
from src.exp17_physical.degrade import (PRESETS, apply_preset, camera_pipeline,
                                        defocus_blur, dot_gain, glare, jpeg,
                                        lens_distortion, motion_blur,
                                        sensor_noise, vignette)

__all__ = ['PRESETS', 'apply_preset', 'camera_pipeline', 'defocus_blur',
           'dot_gain', 'glare', 'jpeg', 'lens_distortion', 'motion_blur',
           'sensor_noise', 'vignette']
