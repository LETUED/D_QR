# D_QR 연구 문서 인덱스
## Research Document Index

**프로젝트:** D_QR — 다차원 QR 코드  
**연구자:** 동환  
**최종 업데이트:** 2026년 5월 28일

---

## 연구 진행 순서 (권장 읽기 순서)

```
[1단계: 기초 이해]
  01_qr_fundamentals.md
  → QR/바코드가 실제로 어떻게 작동하는가
  → 왜 1비트/모듈에 갇혀 있는가
  → D_QR이 공략할 병목이 무엇인가

        ↓

[2단계: 철학 수립]
  02_design_philosophy.md
  → 역사적으로 차원 증가는 항상 기술의 진보였다
  → D_QR 4대 설계 철학 (단순/내구/무한용량/장소무제한)
  → 용량 한계를 없애면 무엇이 가능한가

        ↓

[3단계: 기술 수식화]
  03_dqr_specification.md
  → T ∈ ℤ^(H×W×D) — 모듈을 스칼라에서 벡터로
  → 인코딩/렌더링/디코딩 함수 수식 정의
  → 채널별 ECC, 마스킹, Neural D_QR

        ↓

[4단계: 숫자 검증]
  04_capacity_analysis.md
  → QR vs JAB vs D_QR 이론 용량 비교
  → 현실 보정 후 실효 용량 (BER 모델)
  → Phase별 달성 가능 목표

        ↓

[5단계: 기술 지형]
  05_technology_survey.md
  → 기존 고용량 코드 전수 조사 (JAB, HCCB 등)
  → 광학/홀로그래픽/양자/OAM 인코딩 최신 연구
  → 단기~장기 적용 가능성 매트릭스

        ↓

[6단계: 수익화 전략]
  06_commercialization.md
  → QR 창시자는 어떻게 했는가
  → D_QR로 돈을 버는 현실적인 경로
  → 지금 해야 할 것: 특허 전략
```

---

## 문서별 한 줄 요약

| 파일 | 핵심 질문 | 핵심 답 |
|------|---------|--------|
| `01_qr_fundamentals.md` | QR이 어떻게 작동하는가? | 파인더+RS ECC+마스킹으로 1비트/모듈 달성 |
| `02_design_philosophy.md` | D_QR의 설계 원칙은? | 단순/내구/무한용량/장소무제한 4대 철학 |
| `03_dqr_specification.md` | D_QR을 수식으로 어떻게 정의하는가? | T[i,j,:]가 D차원 벡터인 텐서 코드 |
| `04_capacity_analysis.md` | 실제로 몇 배 더 담을 수 있는가? | Phase 1: QR 4~5배, Phase 3: 13배 |
| `05_technology_survey.md` | 관련 기술 지형은? | 메타서피스 36채널, 페타비트/디스크 실증 |
| `06_commercialization.md` | 발명하면 어떻게 돈을 버는가? | 특허+오픈스펙+SDK/API 모델 |
| `07_3d_dimensional.md` | 3D 물리 공간이 추가 차원을 제공하는가? | Z-depth, 다시점, 섀도우 코드 3가지 경로 |
| `08_perception_theory.md` | 2D 센서로 N차원을 담을 수 있는가? | 차원 링크 체인으로 수학적 무한 차원 ✅ 증명 |
| `09_design_consolidation.md` | 방향성을 어떻게 재정의하는가? | 단일색 기하 + 점진적 공개 + soft-decision, 무한 용량 주장 폐기 |
| `10_patent_fto.md` | 위협 특허 3건을 회피할 수 있는가? | 3건 모두 회피 가능 (분리자 결여·QR 한정·공간 분리) + 설계 제약 확정 |
| `11_research_roadmap.md` | 다음 6개월은 무엇을 하는가? | Honeywell 패밀리(high) 발견 → 표 일치율 메트릭 교체 → 10주 내 가출원 (exp11~16) |
| `12_hologram_3d.md` | D_QR이 3차원을 표현할 수 있는가? | ✅ 가능 — 3D가 "되는" 게 아니라 3D로 "재생되는" 것 (단일색 창립 가설 회수, C₃=관측 기하 채널) |
| `13_physical_validation.md` | 합성 강건성이 실물에서도 성립하는가? | exp16~18 합성 포락선을 인쇄 mm·촬영 제약·합격 기준으로 환산한 실행 프로토콜 (exp19, 하드웨어 대기) |

---

## 핵심 수식 빠른 참조

```
D_QR 텐서:    T ∈ ℤ^(H × W × D)
모듈 값:      T[i,j,k] ∈ {0, ..., L_k - 1}
모듈당 비트:  B_module = Σ ⌊log₂ L_k⌋
인코딩:       E : {0,1}^N → ℤ^(H×W×D)
렌더링:       R : ℤ^(H×W×D) → ℝ^(Hm × Wm × 3)
디코딩:       D : ℝ^(Hm × Wm × 3) → {0,1}^N
```

---

## Phase 로드맵 빠른 참조

| Phase | 차원 구성 | bits/module | QR 대비 | 구현 조건 |
|-------|---------|-----------|--------|---------|
| **1** | c₁(색조) + c₂(채도) + c₃(명도) | **7 bits** | **4~5×** | 스마트폰 카메라 |
| **2** | Phase 1 + p₁(방향) + p₂(패턴) | **12 bits** | **8~10×** | 모듈 ≥10px + CNN |
| **3** | Phase 2 + f₁(DCT저주) + f₂(DCT고주) | **15 bits** | **12~14×** | 화면 전용 |
| **4+** | Phase 3 + 편광/OAM/양자 | **36채널+** | **100×+** | 미래 하드웨어 |

---

## 미해결 핵심 문제 (전 문서 통합)

### 즉시 해결 필요 ❌
1. **색상 팔레트 BER 실측**: 8색 HSV 팔레트를 화면에 렌더링 → 카메라 촬영 → 실제 오인식률 측정
2. **왕복 테스트**: encode → render → decode → 원본 비트 일치 확인
3. **포맷 정보 확장 설계**: QR의 15비트 포맷 정보로는 D_QR 메타데이터 부족

### 중기 연구 필요 ⚠️
4. **파인더 패턴 재설계**: 기존 QR 리더 오검출 방지
5. **채널별 ECC 레이트 최적화**: 실측 BER 기반 채널별 최적 레이트
6. **패턴 채널 CNN 디코더**: 학습 데이터 수집 및 모델 훈련

### 장기 탐구 🔬
7. **Compressive Sensing ECC**: Reed-Solomon 대체 여부 타당성
8. **Neural D_QR**: 오토인코더 기반 학습 인코딩
9. **편광/OAM 채널**: 스마트폰 하드웨어 발전 시 통합

---

## 논문 마스터 목록 (전 문서 통합)

### 코드 시스템
| 논문 | 연도 | 링크 |
|------|------|------|
| ISO/IEC 18004 QR 표준 | 2015 | https://www.iso.org/standard/62021.html |
| JAB Code 공식 스펙 | 2020 | https://jabcode.org/ |
| Microsoft HCCB | 2005 | https://www.microsoft.com/en-us/research/project/high-capacity-color-barcodes-hccb/ |
| HCC2D 스펙 | 2015 | https://free-barcode.com/barcode/barcode-types/hcc2d-high-capacity-colored.asp |
| Deep Learning QR SR | 2024 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12192636/ |
| Reed-Solomon for Coders | - | https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders |
| QR Code Tutorial (Thonky) | - | https://www.thonky.com/qr-code-tutorial/ |
| 2D Color Barcodes for Mobile | 2015 | https://www.academia.edu/15020602/2D_color_barcodes_for_mobile_phones |

### 광학 다중화 & 메타서피스
| 논문 | 연도 | 링크 |
|------|------|------|
| **36채널 메타서피스 홀로그래피** (POSTECH) | 2025 | https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202504634 |
| **16채널 암호화 메타 광학** | 2025 | https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/adma.202419322 |
| 구조색 메타 나노프린팅 3채널 (武汉大学) | 2024 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11636407/ |
| **1 페타비트/디스크** (Nature) | 2024 | https://www.nature.com/articles/s41586-023-06980-y |
| Information Multiplexing → Metaholography | 2024 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11501936/ |
| Four-Dimensional Key Design | 2014 | https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-17-20727&id=299634 |

### OAM 다중화
| 논문 | 연도 | 링크 |
|------|------|------|
| **OAML: OAM 격자 홀로그래피 (9×)** | 2024 | https://www.spiedigitallibrary.org/journals/advanced-photonics-nexus/volume-3/issue-01/016005/ |
| **OAM 연산자 다중화 (9×)** | 2025 | https://www.nature.com/articles/s41377-025-02107-2 |

### 양자 & 편광
| 논문 | 연도 | 링크 |
|------|------|------|
| Quantum-Enhanced Barcode Decoding | 2020 | https://link.aps.org/doi/10.1103/PhysRevApplied.14.064026 |
| Quantum Barcodes via Persistent Homology | 2025 | https://arxiv.org/html/2504.10468 |
| PolarTag: Invisible Data via Polarization | 2020 | https://www.cs.columbia.edu/~xia/publication/hotmobile20-polartag/hotmobile20-polartag.pdf |
| Project Silica (Microsoft) | 2024 | https://www.microsoft.com/en-us/research/project/project-silica/ |

### 차원 압축 이론
| 논문 | 연도 | 링크 |
|------|------|------|
| Compressive Sensing on Manifolds | 2019 | https://link.springer.com/article/10.1007/s00466-019-01703-5 |
| Tensor Networks (MPS) | - | https://tensornetwork.org/mps/ |
| HiFiC Neural Codec | 2020 | https://arxiv.org/abs/2006.09965 |
| Learned Image Compression | 2016 | https://arxiv.org/abs/1611.01704 |

---

*이 인덱스는 모든 문서의 변경에 맞춰 업데이트됩니다.*  
*최종 업데이트: 2026년 5월 28일*
