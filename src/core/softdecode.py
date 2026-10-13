"""
RS 소거 사다리 (soft-decision erasure ladder) — exp9·exp10·exp11·exp12 공용.

RS 정정 한계: 2·오류 + 소거 ≤ nsym
  오류만        → 최대 nsym/2 정정
  위치 아는 소거 → 최대 nsym   정정 (2배)

복호 전략: hard 시도 → 실패 시 저신뢰 바이트를
nsym/3 → 2nsym/3 → (nsym-2) 개씩 소거로 표시하며 재시도.
(exp9 decoder_v9에서 도입 — exp10이 스트림별 게이트로 재사용하기 위해 추출)

exp12 미검출 오복호 3층 방어 중 1·2층이 이 파일에 구현됨:

  1층 — 사다리 상한 (nsym → nsym-2):
    소거를 nsym 전량까지 쓰면 미지수 nsym개 = 방정식 nsym개라 연립이
    항상 풀려 버려, 비소거 위치에 오류가 남아 있어도 '성공'을 반환한다
    (exp10 blur 6.0px·exp11 프로파일 B 고오염에서 실측 관측된 미검출 오복호).
    최종 단계를 nsym-2로 제한해 검증 여력(신드롬 잉여) 최소 2심볼을 확보.

  2층 — 재인코딩 대조 (CRC 아님 — FTO 제약 3 준수):
    RS 복호 성공 후 복호 결과를 RSCodec으로 재인코딩해 수신 코드워드와
    바이트 단위 비교. 불일치 위치가 (소거 표시 위치 ∪ RS가 보고한 정정
    위치)를 벗어나면 오복호로 판정하고 해당 단계를 실패 처리. 결정론적.

exp11b 추가 — 결함③ 이중 방어 (budget=True):
    exp12 잔존 오복호 84건의 단일 시그니처 = "최종 소거 단계(소거 49)에서
    BM 오류 탐색이 신드롬 여력을 재소진(보고 정정 수 52~65 > 소거 49)".

  3-a 청크별 errata 예산 검사 (chunk_errata_budget_ok):
    RS 유일 복호 반경은 청크별로 2·오류 + 소거 ≤ nsym. 이를 벗어난
    복호는 '대수적으로 보증 불가' → 실패 처리. 오류 수는 재인코딩
    대조(2층과 동일한 결정론적 비교)에서 소거 외 불일치 위치를 세어
    전역 좌표로 정확히 산출 — reedsolo가 반환하는 청크 상대 errata
    평탄 연결의 모호성을 우회. slack 인자로 nsym-2(엄격)도 선택 가능.

exp15b 추가 — 4층: 퇴화 코드워드 가드 (reject_degenerate):
    exp15 기준선 비교에서 관측된 D_QR 유일의 미검출 오복호 7건은 전부
    "전부-0 코드워드 퇴화" 단일 모드였다 — RS 부호는 선형이라 전부-0은
    항상 유효 코드워드이고, 재인코딩 대조(2층)도 encode(0…0)=0…0이라
    통과한다. 군집 블롯이 심볼 대부분을 덮으면 판독 코드워드가 전부-0
    근방으로 이동하고 사다리가 전부-0 흡인자에 수렴한다 (증거:
    outputs/exp15/exp15_wrong_blot2_trial8.png).

    가드: 복호 결과 메시지가 전부-0 또는 상수(모든 바이트 동일)면 거부.
    정상 페이로드(길이 프리픽스·텍스트)가 상수 메시지를 만드는 경우는
    구조적으로 배제 가능하며, 인코더 스크램블(src/core/scramble — 전부-0
    근방 페이로드도 비자명 코드워드로 변환)이 이를 보증한다. CRC가 아니라
    특정 퇴화 출력에 대한 결정론 거부 규칙이므로 FTO 제약 3 무관.
    기본값 False — exp9~12 산출물 동결 보존, v11b·v13만 True로 활성.

  3-b 최종 단계 순수 소거 복호 (only_erasures):
    이미지 실측에서 관측된 사다리 오복호는 전부 "최종 단계(nsym-2 소거)
    + BM 오류 1개 = 2e+s가 정확히 nsym"의 형태였다(exp11b 진단).
    최종 단계에서 BM 오류 탐색 자체를 끄면(순수 소거 복호) 신드롬 여력
    2심볼이 온전히 검증에 쓰여 이 경로가 기계적으로 차단된다. 반면
    중간 단계의 BM 대량 정정(예: 소거 50 + 오류 13)은 정상 복호이므로
    유지 — slack=2 일괄 거부보다 정확하게 시그니처만 걸러낸다.
"""
from reedsolo import RSCodec, ReedSolomonError

# 1층: 사다리 최종 단계에 남겨둘 검증 여력 (심볼 수)
LADDER_CAP_SLACK = 2


def ladder_steps(nsym: int, cap: bool = True) -> list:
    """
    소거 사다리 단계 목록.

    cap=True (기본, exp12 1층): 최종 단계 = nsym - LADDER_CAP_SLACK.
    cap=False: 구버전(exp9~11) 동작 재현 — 최종 단계 nsym (벤치마크 비교용).
    """
    top = max(0, nsym - LADDER_CAP_SLACK) if cap else nsym
    steps = []
    for k in (0, nsym // 3, (2 * nsym) // 3, top):
        k = min(k, top)
        if k not in steps:
            steps.append(k)
    return steps


def is_degenerate_message(msg: bytes) -> bool:
    """
    4층 가드 판정: 메시지가 전부-0 또는 상수(모든 바이트 동일)인가.

    전부-0 메시지 ⟺ 전부-0 코드워드 (RS 인코딩은 선형·계통적) — exp15에서
    실측된 미검출 오복호의 유일 모드. 상수 메시지도 함께 거부한다 — 정상
    페이로드가 상수가 되는 경우는 구조적으로 없으며(스크램블 시 보증),
    같은 계열의 퇴화 흡인자를 보수적으로 차단한다. 결정론적이고 검출
    잉여(체크섬)를 쓰지 않으므로 CRC 아님 (FTO 제약 3 무관).
    """
    return len(msg) > 0 and len(set(msg)) == 1


def reencode_mismatch_ok(rsc: RSCodec, codeword: bytes, decoded_msg: bytes,
                         erase_pos, errata_pos) -> bool:
    """
    2층 재인코딩 대조: 복호 결과를 재인코딩해 수신 코드워드와 비교.

    수신 코드워드와 재인코딩 코드워드의 불일치 위치 전부가
    (소거 표시 위치 ∪ RS 보고 정정 위치) 안에 들어야 정상 복호.
    벗어나는 불일치가 하나라도 있으면 오복호 의심 → False.

    주의: reedsolo는 255바이트 초과 코드워드를 청크로 나눠 복호하며
    errata_pos를 '청크 상대 위치'로 평탄 연결해 반환한다(전역 오프셋 없음).
    따라서 전역 위치 g는 (g % rsc.nsize)로 환산해 대조한다 —
    단일 청크(≤255B)에서는 g % nsize == g 라서 정확 일치 검사와 동일.
    """
    reencoded = bytes(rsc.encode(decoded_msg))
    if len(reencoded) != len(codeword):
        return False

    erase_set = set(erase_pos or [])
    errata_rel = set(errata_pos or [])
    nsize = rsc.nsize
    for g in range(len(codeword)):
        if codeword[g] == reencoded[g]:
            continue
        if g in erase_set or (g % nsize) in errata_rel:
            continue
        return False  # 보고되지 않은 위치가 바뀜 → 오복호
    return True


def chunk_errata_budget_ok(rsc: RSCodec, codeword: bytes, decoded_msg: bytes,
                           erase_pos, slack: int = 0) -> bool:
    """
    exp11b 결함③-a — 청크별 errata 예산 검사.

    청크 c(전역 [c·nsize, (c+1)·nsize) — reedsolo decode의 분할과 동일)마다
        2·(소거 외 정정 위치 수) + (공급한 소거 수) ≤ nsym - slack
    을 검사. 정정 위치는 재인코딩 코드워드와의 불일치 위치(전역 좌표)로
    산출하므로 결정론적이며, 공급한 소거는 실제 값 변경 여부와 무관하게
    전부 센다 (Forney GMD의 보증 부등식 2e+s ≤ d-1과 같은 형태 — 보수적).

    slack: 검증 여력 (기본 0 → RS 유일 복호 반경 2e+s ≤ nsym).
           exp12 잔존 84건(2e+s = 55~81 > nsym=51)은 slack=0으로 전부 차단.
           slack=LADDER_CAP_SLACK(2)는 경계대(2e+s ∈ {nsym-1, nsym})까지
           일괄 거부하는 엄격 모드 — 정상 대량 BM 정정(예: 소거 50+오류 13,
           2e+s=76=nsym)도 함께 잃으므로 기본값이 아니다.
    """
    reencoded = bytes(rsc.encode(decoded_msg))
    if len(reencoded) != len(codeword):
        return False

    nsize = rsc.nsize
    budget = rsc.nsym - slack
    erase_set = set(erase_pos or [])
    n_chunks = (len(codeword) + nsize - 1) // nsize
    for c in range(n_chunks):
        lo, hi = c * nsize, min((c + 1) * nsize, len(codeword))
        erasures = sum(1 for g in erase_set if lo <= g < hi)
        errors = sum(1 for g in range(lo, hi)
                     if codeword[g] != reencoded[g] and g not in erase_set)
        if 2 * errors + erasures > budget:
            return False
    return True


def erasure_ladder_decode(codeword: bytes, byte_conf: list,
                          nsym: int, data_len: int,
                          cap: bool = True, verify: bool = True,
                          budget: bool = False,
                          budget_slack: int = 0,
                          steps: list = None,
                          reject_degenerate: bool = False) -> dict:
    """
    소거 사다리 복호 (exp12 1·2층 방어 기본 적용).

    byte_conf: 바이트별 신뢰도 (작을수록 애매 → 먼저 소거 후보)
    cap      : 1층 사다리 상한 (False = exp9~11 구동작, 비교 실험용)
    verify   : 2층 재인코딩 대조 (False = 구동작)
    budget   : 결함③ 이중 방어 — 청크별 errata 예산 검사(3-a) +
               최종 단계 순수 소거 복호(3-b). 기본 False로 exp9~12
               산출물 동결 보존, decoder_v11b가 True로 사용.
    budget_slack: 3-a 예산 여유 (0 = RS 유일 복호 반경 2e+s ≤ nsym,
               2 = nsym-2 엄격 — 중간 단계의 정상 대량 BM 정정까지
               거부하므로 비교 실험용)
    steps    : 소거 단계 직접 지정 (None = ladder_steps(nsym, cap) —
               cap 상한은 호출 측 책임)
    reject_degenerate: exp15b 4층 가드 — 복호 메시지가 전부-0/상수면 해당
               단계를 실패 처리 (전부-0 코드워드 퇴화 차단). 기본 False로
               exp9~12 동작 동결, decoder_v11b·decoder_v13이 True로 사용.
    Returns: {'data', 'mode', 'erasures_used', 'errata_count', 'verified'}
    Raises: ReedSolomonError (사다리 소진 또는 전 단계 대조 불일치 시)
    """
    rsc = RSCodec(nsym)
    order = sorted(range(len(codeword)), key=lambda i: byte_conf[i])

    steps_list = list(steps if steps is not None
                      else ladder_steps(nsym, cap=cap))
    top = max(steps_list) if steps_list else 0

    last_err = None
    for k in steps_list:
        erase = sorted(order[:k]) if k else None
        # 3-b: 최종 단계는 BM 오류 탐색 금지 (신드롬 여력 2심볼을 검증에 보존)
        only_e = bool(budget and k == top and k > 0)
        try:
            decoded, _, errata = rsc.decode(codeword, erase_pos=erase,
                                            only_erasures=only_e)
        except ReedSolomonError as e:
            last_err = e
            continue
        # 4층 (exp15b): 퇴화 메시지 거부 — 전부-0 흡인자는 verify(2층)를
        # 항상 통과하므로(선형성) 가장 먼저, RS 성공 직후에 차단한다
        if reject_degenerate and is_degenerate_message(bytes(decoded)):
            last_err = ReedSolomonError(
                f"퇴화 코드워드 거부 (소거 {k}개 단계) — 전부-0/상수 메시지")
            continue
        if verify and not reencode_mismatch_ok(rsc, codeword, bytes(decoded),
                                               erase, errata):
            last_err = ReedSolomonError(
                f"재인코딩 대조 불일치 (소거 {k}개 단계) — 오복호 차단")
            continue
        if budget and not chunk_errata_budget_ok(rsc, codeword,
                                                 bytes(decoded), erase,
                                                 slack=budget_slack):
            last_err = ReedSolomonError(
                f"errata 예산 초과 (소거 {k}개 단계) — 보증 불가 복호 차단")
            continue
        return {
            'data': bytes(decoded)[:data_len],
            'mode': 'hard' if k == 0 else 'soft',
            'erasures_used': k,
            'errata_count': len(errata),
            'verified': bool(verify),
        }
    raise ReedSolomonError(f"soft-decision 복호 실패 (사다리 소진): {last_err}")
