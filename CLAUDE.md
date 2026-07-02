# D_QR — Research Multi-Agent System

## 빠른 시작

```bash
copy .env.example .env   # ANTHROPIC_API_KEY 입력
py -m pip install -r requirements.txt
py main.py "연구 질문"
```

## 아키텍처

```
사용자 질문
    │
    ▼
ResearchOrchestrator          orchestrator.py
  model: sonnet-4-6
  thinking: adaptive
  tool: delegate_to_agent
    │
    ├──▶ SearchAgent           agents/search.py
    │      web_search (서버사이드)
    │
    ├──▶ AnalystAgent          agents/analyst.py
    │      도구 없음 (순수 추론)
    │
    ├──▶ CoderAgent            agents/coder.py
    │      code_execution (서버사이드)
    │
    └──▶ WriterAgent           agents/writer.py
           save_report → outputs/*.md
```

## 연구 문서 (`concept/`)

읽기 순서: `00_INDEX.md` → `01` → `02` → `03` → `04` → `05` → `06`

| 파일 | 내용 |
|------|------|
| `00_INDEX.md` | 마스터 인덱스, 논문 목록, 미해결 문제 통합 |
| `01_qr_fundamentals.md` | QR/바코드 작동 원리 심층 분석 |
| `02_design_philosophy.md` | D_QR 4대 설계 철학 |
| `03_dqr_specification.md` | 수식 정의 (텐서 표현, E/R/D 함수) |
| `04_capacity_analysis.md` | QR vs JAB vs D_QR 용량 비교 |
| `05_technology_survey.md` | 기술 전수 조사 (메타서피스~양자) |
| `06_commercialization.md` | 수익화 전략 (특허, SDK, 수직 SaaS) |

## Claude Code 에이전트 (`.claude/agents/`)

Claude Code 세션 내에서 `/agent` 명령 또는 Agent 도구로 호출 가능:

| 에이전트 | 파일 | 용도 |
|---------|------|------|
| `researcher` | `.claude/agents/researcher.md` | 웹 검색·문헌 수집 |
| `analyst` | `.claude/agents/analyst.md` | 분석·패턴 인식 |
| `coder` | `.claude/agents/coder.md` | Python 코드 작성 |
| `writer` | `.claude/agents/writer.md` | 보고서 작성·저장 |

## 모델 설정

| 구분 | 모델 | 설정 |
|------|------|------|
| Orchestrator | `claude-sonnet-4-6` | thinking: adaptive, effort: high |
| 서브에이전트 | `claude-sonnet-4-6` | 기본 설정 |

`config.py`에서 변경 가능.

## 디렉토리

```
D_QR/
├── main.py              진입점
├── orchestrator.py      오케스트레이터
├── config.py            모델/경로 설정
├── dqr_encode.py        D_QR 인코더 CLI (exp9)
├── dqr_decode.py        D_QR 디코더 CLI
├── agents/
│   ├── base.py          공통 도구 루프
│   ├── search.py        웹 검색
│   ├── analyst.py       분석
│   ├── coder.py         코드 실행
│   └── writer.py        보고서 저장
├── src/
│   ├── core/            hex_grid.py, multidim.py, constants.py, bitutils.py (공유)
│   ├── exp1_color/      encoder.py, decoder.py, renderer.py
│   ├── exp2_channel_ecc/  *_v2 (채널 독립 RS ECC)
│   ├── exp3_hexgrid/      *_v3 (헥사 격자)
│   ├── exp4_geometric/    *_v4 (기하 방향)
│   ├── exp5_zdepth/       *_v5 (Z-깊이)
│   ├── exp6_subtri/       *_v6 (서브 삼각형)
│   ├── exp7_fractal/      *_v7 (프랙탈)
│   ├── exp8_dlc/          *_v8 (듀얼 스트림)
│   └── exp9_soft/         *_v9 (soft-decision)
├── tests/               test_*.py (왕복 테스트, 루트에서 실행)
├── benchmarks/          exp9_benchmark.py
├── .claude/
│   ├── agents/          Claude Code 서브에이전트 정의
│   └── settings.local.json
├── outputs/             exp1/ ~ exp9/, misc/ 하위 폴더별 산출물
└── .env                 ← git에 올리지 말 것
```

## 주의사항

- `.env` → `.gitignore`에 추가 필수
- `outputs/` 폴더에 마크다운 보고서 저장됨
- 코드 실행 도구는 Anthropic 서버 샌드박스에서 실행 (로컬 아님)
