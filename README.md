# D_QR — 다차원 QR 코드 연구

> 기존 QR의 "모듈당 1비트" 병목을 깨는 자체 연구. 모듈을 벡터로 확장해 용량과 내구성을 같이 끌어올린다. 2026-05~06.

## 무엇인가

모듈을 스칼라에서 벡터로 확장(T ∈ ℤ^(H×W×D))하고, 육각 격자·색상 채널·점진 공개(dual-grain) 구조로 QR의 용량 한계를 넘으려는 연구다. 인코더/디코더 CLI와 실험별 변형, 벤치마크, 그리고 설계·특허 조사 문서로 구성된다.

## 구성

| 경로 | 내용 |
|---|---|
| `dqr_encode.py` / `dqr_decode.py` | CLI 인코더·디코더. 프로파일 A(인쇄·실외)/B(화면·근접), ECC rate 조정 |
| `src/core/` | 육각 격자, RS 소거정정, 신뢰도 연동 soft decode, 스크램블 |
| `src/exp10~exp20/` | 실험별 인코더·디코더 변형 (점진공개, 합의 게이트, 자기서술 포맷링, 홀로그램 시뮬, 검출 파이프라인 등) |
| `benchmarks/` | 실험 스크립트 11종 (블러 스윕, ablation, 오복호율, 물리 열화 스윕 등) |
| `tests/` | 왕복(encode→decode) 테스트 18종 |
| `concept/` | 연구 문서 13편 — `concept/00_INDEX.md`가 읽기 순서 안내 |
| `outputs/` | 실험 산출물 (생성 코드 이미지·벤치마크 결과) |
| `agents/`, `orchestrator.py` | 리서치 자동화 하네스 (부속 도구) |

용어: **오복호**는 RS 디코더가 오류를 잘못 고쳐 틀린 데이터를 "성공"으로 돌려주는 현상(miscorrection)이다. 문서의 "미검출 오복호"는 이를 검출하지 못한 경우를 뜻한다.

연구 문서는 `concept/00_INDEX.md`부터: QR 원리 분석 → 설계 철학 → 수식 명세 → 용량 분석(QR vs JAB vs D_QR) → 기술 조사 → 특허 FTO 분석 → 물리 검증 순.

## 실행

```bash
pip install -r requirements.txt
python dqr_encode.py "hello world" -o outputs/demo.png
python dqr_decode.py outputs/demo.png
python -m pytest tests/          # 왕복 테스트
```

## 상태

활발히 실험 중인 연구 단계 (2026-06 기준 exp1~exp20 산출물 생성, 왕복 테스트 18종). 특허 FTO 분석(`concept/10_patent_fto.md`)을 코드 설계 제약에 반영해 진행 중.
