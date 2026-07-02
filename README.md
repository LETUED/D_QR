# D_QR — 다차원 QR 코드 연구

기존 QR의 "모듈당 1비트" 병목을 깨는 시도. 모듈을 스칼라에서 벡터로 확장(T ∈ ℤ^(H×W×D))해 육각 격자·색상 채널·점진 공개(dual-grain) 구조로 용량과 내구성을 같이 끌어올리는 자체 연구다. 2026-05~06 진행.

## 구성

| 경로 | 내용 |
|---|---|
| `dqr_encode.py` / `dqr_decode.py` | CLI 인코더·디코더. 프로파일 A(인쇄·실외)/B(화면·근접), ECC rate 조정 |
| `src/core/` | 육각 격자, RS 소거정정, 신뢰도 연동 soft decode, 스크램블 |
| `src/exp10~exp20/` | 실험별 인코더·디코더 변형 (점진공개, 합의 게이트, 자기서술 포맷링, 홀로그램 시뮬, 검출 파이프라인 등) |
| `benchmarks/` | 실험 스크립트 11종 (블러 스윕, ablation, 오정정률, 물리 열화 스윕 등) |
| `tests/` | 왕복(encode→decode) 테스트 18종 |
| `concept/` | 연구 문서 13편 — `concept/00_INDEX.md`가 읽기 순서 안내 |
| `outputs/` | 실험 산출물 (생성 코드 이미지·벤치마크 결과) |
| `agents/`, `orchestrator.py` | 리서치 자동화 하네스 (부속 도구) |

## 사용

```bash
pip install -r requirements.txt
python dqr_encode.py "hello world" -o outputs/demo.png
python dqr_decode.py outputs/demo.png
python -m pytest tests/          # 왕복 테스트
```

## 연구 문서

`concept/00_INDEX.md`부터. QR 원리 분석 → 설계 철학 → 수식 명세 → 용량 분석(QR vs JAB vs D_QR) → 기술 조사 → 특허 FTO 분석 → 물리 검증 순.
