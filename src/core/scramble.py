"""
LFSR 기반 결정론 스크램블 — exp15b 전부-0 코드워드 퇴화 봉쇄 ② (인코더 측).

목적: QR의 마스크 XOR과 동등한 고정 의사난수 마스크를 메시지에 XOR한 뒤
RS 인코딩한다. 전부-0 근방 페이로드(빈 데이터, 0 패딩 위주 메시지)도
비자명(non-trivial) 코드워드가 되어, 디코더 가드(softdecode.
is_degenerate_message)가 정상 페이로드를 거부할 가능성을 구조적으로 배제한다.

**CRC 아님 (FTO 제약 3 무관)**: 이 변환은 검출(detection) 기능이 전혀 없다 —
잉여 비트를 추가하지 않는 위치별 고정 XOR(전단사 변환)일 뿐이며, 오류를
탐지하거나 무결성을 검사하지 않는다. QR 규격의 데이터 마스킹(ISO/IEC
18004 §7.8)과 같은 범주의 변조 방지 변환이다. 밝기 연속량도 사용하지 않는다.

LFSR: 8비트 Galois LFSR, 원시다항식 x^8+x^6+x^5+x^4+1 (탭 0xB8), 시드 0xA7.
  상태가 0이 될 수 없으므로 **마스크의 모든 바이트는 비0** — 전부-0 메시지를
  XOR하면 마스크 자체(전 바이트 비0, 비상수)가 되어 퇴화 가드와 충돌하지
  않는다. 주기 255바이트 후 반복 (마스크 결정론·위치 고정 — 복호 시 동일
  마스크 XOR로 원복, 스크램블은 자기 역원).
"""

_LFSR_TAPS = 0xB8   # x^8 + x^6 + x^5 + x^4 + 1 (원시다항식, 주기 255)
_LFSR_SEED = 0xA7   # 비0 임의 고정값 — 규약의 일부 (변경 시 호환 단절)

_mask_cache = bytearray()


def scramble_mask(n: int) -> bytes:
    """위치 0부터 n바이트의 고정 의사난수 마스크 (결정론, 전 바이트 비0)."""
    while len(_mask_cache) < n:
        state = _mask_cache[-1] if _mask_cache else _LFSR_SEED
        # Galois LFSR 1바이트 전진: 비트별 시프트+탭 XOR을 8회
        for _ in range(8):
            lsb = state & 1
            state >>= 1
            if lsb:
                state ^= _LFSR_TAPS
        _mask_cache.append(state)
    return bytes(_mask_cache[:n])


def xor_scramble(data: bytes) -> bytes:
    """
    데이터를 위치 고정 마스크와 XOR (자기 역원 — 두 번 적용하면 원복).

    인코더: RS 인코딩 직전의 (패딩 포함) 메시지에 적용.
    디코더: RS 복호 성공 후의 메시지에 동일하게 적용해 원복.
    """
    mask = scramble_mask(len(data))
    return bytes(b ^ m for b, m in zip(data, mask))
