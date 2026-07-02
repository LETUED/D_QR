"""
exp13 포맷 링 — 18비트 포맷 워드 인코딩/디코딩.

포맷 워드 (MSB-first, 총 18비트):
  version    4b  = 13 (exp13)
  profile    2b  = 0:A(6비트/셀) 1:B(12비트/셀) 2~3:예비
  ecc_level  2b  = ECC_LEVELS 인덱스 (nsym을 여기서 유도 — 외부 의존 제거)
  radius_idx 4b  = GRID_RADII 인덱스 (기하 자기추정 결과와 대조 검증용)
  mask_id    3b  = 마스크 자리 (exp13에서는 항상 0 — 향후 마스킹 도입 대비)
  scramble   1b  = exp15b LFSR 스크램블 여부 (src/core/scramble — 전부-0
                   코드워드 퇴화 회피, 기본 1). 구 exp13 코드는 예비 비트가
                   0이었으므로 scramble=0으로 파싱되어 그대로 호환된다.
  reserve    2b  = 예비 (0)

[보호 방식 선택 근거 — 3중 반복 + 비트별 다수결 (BCH(15,5) 대신)]
QR의 포맷 정보는 BCH(15,5)로 5비트를 15비트로 보호하지만:
  · 우리 워드는 18비트 → BCH(15,5)면 4블록 = 60셀, 3중 반복은 54셀로
    더 적은 예약 셀을 쓰면서 사본당 워드 전체를 담는다.
  · 세 사본이 세 파인더 옆에 공간 분산 → exp11/11b에서 실측된 지배적
    위협(국소 군집 오염)이 한 사본 전체를 덮어도 나머지 둘의 다수결로
    무손실 복원. 동일 비트 위치가 3곳에 흩어지는 반복이 군집 오염
    모델에 BCH보다 직접적으로 부합한다.
  · 다수결은 순수 정수 표결 — exp11 표 일치율 원리와 동일 계열 (FTO 안전,
    구현·검증 비용 최소).
"""

# ECC 레벨 → RS rate (오름차순). nsym = max(2, min(127, int(255·rate))).
ECC_LEVELS = (0.10, 0.20, 0.30, 0.40)

FORMAT_VERSION = 13

PROFILE_IDS = {'A': 0, 'B': 1}
PROFILE_BY_ID = ('A', 'B')

# exp9 프로파일 기본 ECC와 동일: A=30%(level 2), B=20%(level 1)
DEFAULT_ECC_LEVEL = {'A': 2, 'B': 1}

# (필드명, 비트폭) — 합 18. exp15b: 구 reserve 3b의 첫 비트를 scramble로
# 승격 (구 코드는 0 → scramble off로 파싱, 무손실 호환)
FIELD_WIDTHS = (('version', 4), ('profile', 2), ('ecc_level', 2),
                ('radius_idx', 4), ('mask_id', 3), ('scramble', 1),
                ('reserve', 2))
FORMAT_BITS = sum(w for _, w in FIELD_WIDTHS)   # = 18


def nsym_for_level(level: int) -> int:
    if not (0 <= level < len(ECC_LEVELS)):
        raise ValueError(f"ecc_level은 0~{len(ECC_LEVELS) - 1}: {level}")
    return max(2, min(127, int(255 * ECC_LEVELS[level])))


def pack_format(fields: dict) -> list:
    """필드 dict → 18비트 리스트 (MSB-first)."""
    bits = []
    for name, width in FIELD_WIDTHS:
        val = int(fields[name])
        if not (0 <= val < (1 << width)):
            raise ValueError(f"{name}={val}이 {width}비트 범위를 벗어남")
        bits.extend((val >> (width - 1 - b)) & 1 for b in range(width))
    return bits


def parse_format(bits: list) -> dict:
    """18비트 리스트 → 필드 dict."""
    if len(bits) != FORMAT_BITS:
        raise ValueError(f"포맷 비트 수 불일치: {len(bits)} != {FORMAT_BITS}")
    fields, pos = {}, 0
    for name, width in FIELD_WIDTHS:
        val = 0
        for b in range(width):
            val = (val << 1) | (1 if bits[pos + b] else 0)
        fields[name] = val
        pos += width
    return fields


def majority_bits(copies: list) -> list:
    """사본별 비트 리스트 → 비트 위치별 다수결 (사본 3개, 동수 없음)."""
    n = len(copies)
    return [1 if sum(c[i] for c in copies) * 2 > n else 0
            for i in range(len(copies[0]))]
