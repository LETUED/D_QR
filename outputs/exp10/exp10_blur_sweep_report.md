# exp10 블러 스윕 — dual-grain 점진적 공개 생존 경계

- 페이로드: coarse 48B / fine 206B
- 격자: r=12 (coarse 230셀 · fine 218셀, 패리티 인터리브)
- ECC: coarse 30% (nsym=76) / fine 20% (nsym=51), hex_size=32
- 블러는 결정적(GaussianBlur) → 성공률은 0/100%

## 생존 경계

| 스트림 | 생존 한계 | 붕괴 시점 |
|--------|----------|----------|
| coarse | ≤ 10.0px | 10.5px |
| fine   | ≤ 5.5px | 6.0px |

**점진적 공개 대역: 6.0~10.0px** — 이 구간에서 fine은 예외 없이 None으로 강등되고 coarse만 열림.

> **주의**: 블러 6.0px에서 fine 사다리가 잘못된 데이터를 '성공'으로 반환(미검출 오복호). 소거를 nsym 전량까지 쓰면 검증 여력이 0이 되기 때문 — 후속 exp에서 fine 신뢰도 임계 게이트 (예: conf < 0.15 → 강등) 추가 검토.

## 상세 결과

| 블러(px) | coarse | coarse 모드 | fine | fine 모드 | fine 신뢰도 |
|----------|--------|-------------|------|-----------|-------------|
| 0.0 | O | hard | O | hard | 0.992 |
| 0.5 | O | hard | O | hard | 0.992 |
| 1.0 | O | hard | O | hard | 0.992 |
| 1.5 | O | hard | O | hard | 0.975 |
| 2.0 | O | hard | O | hard | 0.908 |
| 2.5 | O | hard | O | hard | 0.810 |
| 3.0 | O | hard | O | hard | 0.705 |
| 3.5 | O | hard | O | hard | 0.594 |
| 4.0 | O | hard | O | hard | 0.482 |
| 4.5 | O | hard | O | hard | 0.371 |
| 5.0 | O | hard | O | hard | 0.271 |
| 5.5 | O | hard | O | hard | 0.181 |
| 6.0 | O | hard | X | soft | 0.115 |
| 6.5 | O | hard | X | — | 0.092 |
| 7.0 | O | hard | X | — | 0.085 |
| 7.5 | O | hard | X | — | 0.090 |
| 8.0 | O | hard | X | — | 0.090 |
| 8.5 | O | hard | X | — | 0.085 |
| 9.0 | O | hard | X | — | 0.078 |
| 9.5 | O | hard | X | — | 0.072 |
| 10.0 | O | hard | X | — | 0.069 |
