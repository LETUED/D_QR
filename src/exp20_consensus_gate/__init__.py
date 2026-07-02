"""
exp20 — 점진적 공개 게이트 FTO 이식 (밝기 마진 → 표 일치율).

인코딩·렌더링은 exp10 그대로, 디코더 신뢰도·게이트만 이산 표결로 교체.
FTO: Honeywell 반사율 메트릭 회피 (decoder_v20 docstring).
"""
from src.exp20_consensus_gate.decoder_v20 import DQRDecoderV20

__all__ = ['DQRDecoderV20']
