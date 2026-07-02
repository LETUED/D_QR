"""
공간 소거 스윕(spatial erasure sweep) — exp11b 결함① 해소 모듈.

exp11 중단규칙 결함①: 균일 회색 패치가 삼각형의 샘플점 N개를 전부 덮으면
표결이 '만장일치-오류'가 되어 최대 신뢰도를 받는다 → 신뢰도 순위 기반
소거 사다리는 이 바이트를 소거 후보에서 구조적으로 제외한다
(국소 패치에서 margin 대비 최대 48%p 열세의 직접 원인).

해법: 국소 오염은 정의상 공간적으로 뭉친다. 데이터 셀을 격자 중심각
기준의 연속 쐐기(wedge) 그룹 G개로 분할하고(그룹당 바이트 수 ≤
(nsym-2)/2 보장), 신뢰도 사다리가 실패하면 각 그룹·그룹 쌍을 통째로
소거하는 시행을 순차 실행한다. 올바른 그룹 가설이 패치를 정확히 커버하면
나머지 산재 오류는 RS 오류 정정(BM)이 흡수한다.

FTO 위치 (위반 아님 — 설계 문서 명시 사항):
  * 순수 공간 시행(기본 1차 패스)은 어떤 신뢰도 메트릭도 사용하지 않는
    '위치 가설' 시도다 — 신뢰도→소거 매핑을 청구하는 Honeywell
    US 10,235,547 울타리(반사율/contrast/erasure-gap 연속량) 바깥이며,
    Forney GMD(1966)·소거 시행 복호(erasure trial decoding) 계열
    선행기술 위에 서 있어 안전하다.
  * 2차 충전(conf-fill) 패스는 이산 표 개수(다수표-소수표)만 보조로
    사용한다 — FTO 제약 1이 명시적으로 허용하는 양이며, 밝기 연속량은
    어디에도 쓰지 않는다.
  * 모든 시행에 exp12 2층(재인코딩 대조)과 exp11b 청크 errata 예산
    검사를 적용해 오수락(잘못된 가설의 침묵 오정정)을 차단한다.
"""
import math

from reedsolo import RSCodec, ReedSolomonError

from src.core.softdecode import (LADDER_CAP_SLACK, chunk_errata_budget_ok,
                                 is_degenerate_message, reencode_mismatch_ok)
from src.core.hex_grid import axial_to_pixel, build_hex_grid, build_reserved


def cell_byte_span(cell_index: int, bits_per_cell: int,
                   codeword_len: int) -> range:
    """셀 k(데이터 셀 q-major 순번)가 비트를 기여하는 코드워드 바이트 범위."""
    first_bit = cell_index * bits_per_cell
    last_bit = first_bit + bits_per_cell - 1
    lo = first_bit // 8
    hi = min(last_bit // 8, codeword_len - 1)
    if lo >= codeword_len:
        return range(0)
    return range(lo, hi + 1)


def build_spatial_groups(radius: int, bits_per_cell: int,
                         codeword_len: int, nsym: int,
                         data_cells: list = None) -> list:
    """
    데이터 셀을 공간 연속 쐐기 그룹으로 분할 → 그룹별 바이트 인덱스 목록.

    데이터 운반 셀(코드워드 바이트에 비트를 기여하는 셀)을 격자 중심
    기준 각도순으로 정렬한 뒤, 그룹 바이트 수가 (nsym-2)//2를 넘지 않는
    한도에서 연속 구간을 잘라 그룹화한다. 인접 셀이 한 바이트를 공유하는
    경계 바이트는 양쪽 그룹에 모두 포함된다(소거 커버리지 보수적 확보).

    data_cells: 데이터 셀 (q,r) 목록 — 순번 k가 인코더의 비트 배치 순서와
        일치해야 한다(cell_byte_span의 전제). None이면 v9 예약 셀 기준
        (exp11b 기존 동작 동결). exp13처럼 예약 셀이 다른 배치는 자신의
        데이터 셀 목록(예: layout.data_cells_v13)을 넘겨 그룹을 유도한다.

    Returns: [sorted byte-index list, ...] — 합집합이 코드워드 전체를 덮음.
    """
    limit = (nsym - LADDER_CAP_SLACK) // 2
    # 셀 하나의 바이트 스팬(≤ bpc/8 + 1)조차 못 담는 nsym이면 분할 불가
    min_span = bits_per_cell // 8 + 2
    if limit < min_span:
        raise ValueError(f"nsym={nsym}이 너무 작아 공간 그룹 분할 불가 "
                         f"(그룹 한도 {limit} < 셀 스팬 {min_span})")

    if data_cells is None:
        reserved = build_reserved(radius)
        data_cells = [c for c in build_hex_grid(radius) if c not in reserved]

    # 데이터 운반 셀만 (이후 셀은 패딩 0 — 코드워드 바이트와 무관)
    carriers = []
    for k, (q, r) in enumerate(data_cells):
        span = cell_byte_span(k, bits_per_cell, codeword_len)
        if len(span):
            carriers.append((k, q, r, span))

    # 격자 중심각 순 정렬 → 연속 구간 = 공간적으로 연속한 쐐기(부채꼴)
    # hex_size=1 단위 기하만 사용 — 렌더 파라미터(픽셀 크기)와 무관
    def angle_key(item):
        _, q, r, _ = item
        x, y = axial_to_pixel(q, r, 1.0)
        return (math.atan2(y, x), x * x + y * y)

    carriers.sort(key=angle_key)

    groups, cur = [], set()
    for _, _, _, span in carriers:
        nxt = cur | set(span)
        if cur and len(nxt) > limit:
            groups.append(sorted(cur))
            cur = set(span)
        else:
            cur = nxt
    if cur:
        groups.append(sorted(cur))
    return groups


def _trial_sets(groups: list, limit: int) -> list:
    """시행할 소거 가설 목록: 단일 그룹 → 인접(순환) 쌍 → 비인접 쌍."""
    G = len(groups)
    sets, seen = [], set()

    def push(tag, byte_set):
        if not byte_set or len(byte_set) > limit:
            return
        key = frozenset(byte_set)
        if key in seen:
            return
        seen.add(key)
        sets.append((tag, byte_set))

    for i in range(G):
        push(f"single:{i}", set(groups[i]))
    for i in range(G):                       # 패치가 그룹 경계에 걸친 경우
        j = (i + 1) % G
        if j != i:
            push(f"pair:{i}+{j}", set(groups[i]) | set(groups[j]))
    for i in range(G):                       # 떨어진 오염 두 덩어리
        for j in range(i + 2, G):
            if i == 0 and j == G - 1:
                continue                     # 순환 인접 — 위에서 처리됨
            push(f"pair:{i}+{j}", set(groups[i]) | set(groups[j]))
    return sets


# 합의 수락 임계: 동일 데이터로 수렴한 독립 가설 수의 최소값 / 우세 배율.
# 실측 근거 (patches A90·B40~60 진단, 가설 49개 기준):
#   참 코드워드가 도달 가능한 경우 → 동일 데이터 다중도 7~25
#   오정정 후보(verify·budget 통과한 잘못된 코드워드) → 다중도 1, 드물게 2
# 첫 수락(first-accept) 방식은 고오염에서 오정정 후보를 그대로 반환하는
# 침묵 오정정 경로가 되므로(B60에서 10/10 관측), 다중 가설 합의를 강제한다.
#
# 추가 규칙 — 순수 시행 포함 필수(REQUIRE_PURE): 충전(conf-fill) 시행은
# 그룹이 달라도 충전 바이트가 대부분 겹쳐 상호 독립이 아니다 → 잘못된
# 코드워드가 상관된 fill 시행 3개로 합의 3을 위조하는 사례가 실측됨
# (patches B 50회×8레벨에서 4건, 전부 fill-only·합의 정확히 3).
# 정당한 군집 오염 복구는 정의상 순수 위치 가설(fill=0)이 패치를 직접
# 커버하므로, 최다 후보 집합에 순수 시행이 하나도 없으면 기각한다.
MIN_CONSENSUS = 3
DOMINANCE_RATIO = 2
REQUIRE_PURE = True


def spatial_erasure_sweep(codeword: bytes, byte_conf: list, nsym: int,
                          data_len: int, groups: list,
                          conf_fill: bool = True,
                          budget_slack: int = LADDER_CAP_SLACK,
                          reject_degenerate: bool = True) -> dict:
    """
    공간 소거 스윕 복호 — 신뢰도 사다리 실패 후의 2차 방어.

    1차 패스(순수): 각 가설 집합 E를 통째로 소거 — 신뢰도 메트릭 불사용
      (위치 가설 시행, Forney GMD 계열 — 모듈 docstring의 FTO 위치 참조).
    2차 패스(conf_fill): E에 더해 잔여 소거 예산을 저신뢰(이산 표차 순)
      바이트로 충전 — 패치(만장일치-오류)와 산재 표 분열이 공존하는 경우.

    후보 예산(budget_slack=2, nsym-2): 사다리(slack=0)와 달리 스윕 후보는
    검증 여력 2심볼을 강제한다. 실측 근거 — 산재 패치에서 스윕이 만든
    오정정 후보 41개는 전원 2e+s ∈ {nsym-1, nsym} 경계대였고, 군집 오염의
    참 후보는 1개 예외 빼고 전부 2e+s ≤ nsym-2 (가설이 오염을 제대로
    덮으면 여력이 남는 게 정상). 경계대 일괄 기각으로 오정정 후보가
    전멸하고 참 후보는 합의 정족수를 유지한다.

    수락 규칙(합의): 전 가설을 끝까지 시행해 verify·budget을 통과한
    후보들을 복호 데이터별로 집계하고, ⑴ 다중도 ≥ MIN_CONSENSUS,
    ⑵ 2위 후보의 DOMINANCE_RATIO배 이상, ⑶ 순수(fill=0) 시행 포함
    (REQUIRE_PURE — 상관된 fill 시행의 합의 위조 차단)일 때만 반환한다.
    잘못된 코드워드는 특정 소거 패턴에서만 우연히 나타나는 반면(다중도 1~2)
    참 코드워드는 그것을 덮는 여러 가설에서 일관되게 재현되므로,
    합의 규칙이 스윕의 침묵 오정정 경로를 결정론적으로 차단한다.

    reject_degenerate (exp15b 4층 가드): 전부-0/상수 메시지 후보 거부 —
    전부-0 코드워드는 항상 유효(RS 선형성)하고 재인코딩 대조를 통과하며,
    심볼 대부분이 덮인 이미지에서는 *여러 가설이 일관되게* 전부-0으로
    수렴해 합의 규칙마저 통과할 수 있다(exp15 blot 실측 7건). 이 모듈은
    exp11b 이후 전용이므로 기본 True (사다리 쪽은 기본 False로 동결).

    Returns: erasure_ladder_decode와 같은 키
             + {'sweep_trial', 'fill_used', 'consensus', 'runner_up'}
    Raises: ReedSolomonError (전 시행 실패 또는 합의 미달 시)
    """
    rsc = RSCodec(nsym)
    limit = nsym - LADDER_CAP_SLACK
    n = len(codeword)
    trials = _trial_sets(groups, limit)

    def attempt(erase_set, tag, fill_used):
        erase = sorted(b for b in erase_set if b < n)
        try:
            decoded, _, errata = rsc.decode(codeword, erase_pos=erase)
        except ReedSolomonError:
            return None
        decoded = bytes(decoded)
        # 4층 (exp15b): 전부-0 흡인자는 verify·합의를 모두 통과할 수 있다
        if reject_degenerate and is_degenerate_message(decoded):
            return None
        if not reencode_mismatch_ok(rsc, codeword, decoded, erase, errata):
            return None
        if not chunk_errata_budget_ok(rsc, codeword, decoded, erase,
                                      slack=budget_slack):
            return None
        return {
            'data': decoded[:data_len],
            'mode': 'sweep',
            'erasures_used': len(erase),
            'errata_count': len(errata),
            'verified': True,
            'sweep_trial': tag,
            'fill_used': fill_used,
        }

    # ── 후보 전수 수집 (조기 반환 없음 — 합의 집계가 목적) ────────────
    candidates = {}          # data → [후보 dict, ...]
    for tag, E in trials:
        # 1차: 순수 공간 가설 (신뢰도 불사용)
        res = attempt(E, tag, fill_used=0)
        if res:
            candidates.setdefault(res['data'], []).append(res)
        # 2차: 저신뢰 충전 (이산 표 개수만 — FTO 제약 1 허용 범위)
        if conf_fill:
            order = sorted((b for b in range(n) if b not in E),
                           key=lambda b: byte_conf[b])
            room = limit - len(E)
            if room <= 0:
                continue
            for fill in sorted({room // 2, room} - {0}):
                res = attempt(E | set(order[:fill]), tag, fill_used=fill)
                if res:
                    candidates.setdefault(res['data'], []).append(res)

    if not candidates:
        raise ReedSolomonError(
            f"공간 소거 스윕 실패 ({len(trials)}개 가설 전부 기각)")

    ranked = sorted(candidates.items(), key=lambda kv: -len(kv[1]))
    top_data, top_list = ranked[0]
    runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
    if (len(top_list) < MIN_CONSENSUS
            or len(top_list) < DOMINANCE_RATIO * runner_up):
        raise ReedSolomonError(
            f"공간 소거 스윕 합의 미달 (최다 {len(top_list)}, "
            f"2위 {runner_up}, 기준 ≥{MIN_CONSENSUS} 및 ≥{DOMINANCE_RATIO}×2위)")
    pure_hits = [c for c in top_list if c['fill_used'] == 0]
    if REQUIRE_PURE and not pure_hits:
        raise ReedSolomonError(
            "공간 소거 스윕 기각 — 최다 후보가 충전 시행으로만 구성 "
            "(상관된 fill 합의 위조 차단)")

    best = pure_hits[0] if pure_hits else top_list[0]
    best['consensus'] = len(top_list)
    best['runner_up'] = runner_up
    return best
