"""
exp12 침묵 오정정 3층 방어 검증 벤치마크
실행: py -X utf8 benchmarks/exp12_miscorrection.py [--n 10000]  (프로젝트 루트)

① 코드워드 레벨 (N≥10,000, seed 고정):
   exp10 fine 스트림 구성(nsym=51, data 206B → 코드워드 308B, 2청크)에
   무작위 바이트 오류 주입 → 소거 사다리 복호.
   방어 전(cap=False, verify=False) vs 후(1·2층 기본 적용)의
   오정정률(성공 반환했는데 원본 불일치)·false reject를 비교.

② 게이트 캘리브레이션: exp10 표준 코드 블러 스윕 0~10px에서
   fine vote 신뢰도(exp11 표결 메트릭, N=9)를 실측 → 임계값 근거 기록.

③ 이미지 레벨: exp10 blur 6.0px 침묵 오정정 케이스 재현 →
   1층 단독 / 1+2층 / 3층 게이트 각각의 차단 여부 확인.

④ false reject: 방어로 인해 정상 복호가 실패로 바뀌는 손실 측정
   (코드워드 레벨 + 이미지 레벨 블러 스윕).

산출물: outputs/exp12/exp12_miscorrection_report.md
        outputs/exp12/exp12_miscorrection_chart.png
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import ImageFilter
from reedsolo import RSCodec, ReedSolomonError

from src.core.softdecode import (erasure_ladder_decode, ladder_steps,
                                 LADDER_CAP_SLACK)
from src.exp10_progressive.decoder_v10 import DQRDecoderV10
from src.exp10_progressive.encoder_v10 import split_data_cells
from src.exp12_defense.decoder_v12 import (DQRDecoderV12,
                                           FINE_GATE_THRESHOLD,
                                           GATE_SAMPLE_POINTS)
from benchmarks.exp10_blur_sweep import make_code

SEED = 42
# exp10 fine 스트림과 동일 구성 — 코드워드 308B(>255)라 reedsolo 청크 경로까지 검증
SIM_NSYM = 51
SIM_DATA_LEN = 206
# 오류 수 범위: 오류 단독 한계(25개) 아래부터 풀소거 한계(51개) 너머까지
SIM_ERR_RANGE = (10, 81)
BLUR_LEVELS = [x * 0.5 for x in range(0, 21)]
MISCORRECTION_BLUR = 6.0  # exp10에서 침묵 오정정이 관측된 레벨


# ── ① 코드워드 레벨 시뮬레이션 ──────────────────────────────────────────

def simulate_codeword_level(n_trials: int) -> dict:
    """
    무작위 오류 주입 → 사다리 복호, 방어 전/후 결과 분류.

    신뢰도 모델: 오류 바이트 conf ~ U[0,60], 정상 바이트 conf ~ U[20,128]
    — 구간이 겹치게 해 '신뢰도가 오류 위치를 완벽히 가리키지 못하는'
    블러·오염 실측 상황을 모사 (완벽한 conf면 풀소거 단계가 항상 옳아
    오정정이 원천 차단돼 비교가 무의미해짐).
    """
    rng = np.random.default_rng(SEED)
    rsc = RSCodec(SIM_NSYM)
    cw_len = SIM_DATA_LEN + SIM_NSYM * 2  # 2청크 (204+51) + (2+51)

    tally = {
        'legacy': {'correct': 0, 'wrong': 0, 'fail': 0},
        'defended': {'correct': 0, 'wrong': 0, 'fail': 0},
    }
    cross = {}            # (legacy 결과, defended 결과) → 건수 (3×3 전이표)
    false_reject = 0      # legacy 정상 복호였는데 방어 후 실패
    blocked_wrong = 0     # legacy 오정정이었는데 방어 후 실패(차단 성공)
    survived_wrong = []   # 방어 후에도 오정정 (시행 번호 기록)
    survived_detail = []  # 잔존 오정정 상세 (사다리 단계·정정 수)

    t0 = time.time()
    for trial in range(n_trials):
        data = bytes(rng.integers(0, 256, SIM_DATA_LEN, dtype=np.uint8))
        cw = bytearray(rsc.encode(data))
        assert len(cw) == cw_len

        n_err = int(rng.integers(*SIM_ERR_RANGE))
        pos = rng.choice(cw_len, n_err, replace=False)
        for p in pos:
            cw[p] ^= int(rng.integers(1, 256))  # XOR 비영 → 반드시 값 변화
        err_set = set(int(p) for p in pos)

        conf = [int(rng.integers(0, 61)) if i in err_set
                else int(rng.integers(20, 129)) for i in range(cw_len)]
        cw = bytes(cw)

        outcomes = {}
        for name, kw in (('legacy', dict(cap=False, verify=False)),
                         ('defended', dict(cap=True, verify=True))):
            try:
                res = erasure_ladder_decode(cw, conf, SIM_NSYM,
                                            SIM_DATA_LEN, **kw)
                outcomes[name] = 'correct' if res['data'] == data else 'wrong'
            except ReedSolomonError:
                res = None
                outcomes[name] = 'fail'
            tally[name][outcomes[name]] += 1
            # 방어 후 잔존 오정정은 메커니즘 분석을 위해 상세 기록
            if name == 'defended' and outcomes[name] == 'wrong':
                survived_detail.append(dict(
                    trial=trial, n_err=n_err,
                    erasures_used=res['erasures_used'],
                    errata_count=res['errata_count']))

        key = (outcomes['legacy'], outcomes['defended'])
        cross[key] = cross.get(key, 0) + 1
        if outcomes['legacy'] == 'correct' and outcomes['defended'] == 'fail':
            false_reject += 1
        if outcomes['legacy'] == 'wrong' and outcomes['defended'] == 'fail':
            blocked_wrong += 1
        if outcomes['defended'] == 'wrong':
            survived_wrong.append(trial)

        if (trial + 1) % 1000 == 0:
            print(f"  {trial + 1}/{n_trials} "
                  f"({time.time() - t0:.0f}s) "
                  f"legacy wrong={tally['legacy']['wrong']} "
                  f"defended wrong={tally['defended']['wrong']}")

    return {
        'n_trials': n_trials,
        'tally': tally,
        'false_reject': false_reject,
        'blocked_wrong': blocked_wrong,
        'survived_wrong': survived_wrong,
        'survived_detail': survived_detail,
        'cross': cross,
        'elapsed': time.time() - t0,
    }


# ── ②③④ 이미지 레벨 ────────────────────────────────────────────────────

def legacy_fine_decode(dec10, pixels, fine_cells, params, fine):
    """exp10 구동작(방어 전) fine 복호 재현 → 'correct'|'wrong'|'fail'."""
    cw_f, conf_f = dec10._read_stream(
        pixels, fine_cells, 12, params['hex_size'],
        params['origin_x'], params['origin_y'], params['codeword_len_f'])
    try:
        res = erasure_ladder_decode(cw_f, conf_f, params['nsym_f'],
                                    params['fine_len'],
                                    cap=False, verify=False)
        return ('correct' if res['data'] == fine else 'wrong'), res
    except ReedSolomonError:
        return 'fail', None


def image_level_sweep():
    """블러 스윕: 방어 전 결과 + vote 신뢰도 + v12(3층 전체) 결과."""
    coarse, fine, img, params, result = make_code()
    dec10 = DQRDecoderV10()
    v12 = DQRDecoderV12()           # 게이트 = 출하 임계값
    v12_nogate = DQRDecoderV12(gate_threshold=0.0)  # 1+2층만 (게이트 무력화)
    _, fine_cells = split_data_cells(params['radius'])

    rows = []
    for lv in BLUR_LEVELS:
        deg = img.filter(ImageFilter.GaussianBlur(lv)) if lv > 0 else img
        pixels = np.array(deg.convert('RGB'))

        legacy, _ = legacy_fine_decode(dec10, pixels, fine_cells, params, fine)

        s12 = v12.decode_with_stats(deg, **params)
        if s12['fine'] is None:
            v12_out = 'gated' if not s12['rs_attempted'] else 'fail'
        else:
            v12_out = 'correct' if s12['fine'] == fine else 'wrong'

        s12n = v12_nogate.decode_with_stats(deg, **params)
        if s12n['fine'] is None:
            l12_out = 'fail'
        else:
            l12_out = 'correct' if s12n['fine'] == fine else 'wrong'

        rows.append(dict(blur=lv, vote_conf=s12['fine_vote_confidence'],
                         legacy=legacy, layers12=l12_out, v12=v12_out))
        print(f"  blur={lv:4.1f}  vote_conf={s12['fine_vote_confidence']:.4f}  "
              f"legacy={legacy:7s}  1+2층={l12_out:7s}  v12={v12_out}")
    return rows, params, result


# ── 보고서·차트 ─────────────────────────────────────────────────────────

def write_outputs(sim, rows, params, result):
    out_dir = Path("outputs/exp12")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = sim['n_trials']
    tl, td = sim['tally']['legacy'], sim['tally']['defended']

    # 캘리브레이션 근거 수치
    correct_confs = [r['vote_conf'] for r in rows if r['legacy'] == 'correct']
    wrong_confs = [r['vote_conf'] for r in rows if r['legacy'] == 'wrong']
    last_ok_conf = min(correct_confs) if correct_confs else None
    mis_conf = max(wrong_confs) if wrong_confs else None

    blur6 = next(r for r in rows if r['blur'] == MISCORRECTION_BLUR)
    img_false_reject = sum(1 for r in rows
                           if r['legacy'] == 'correct' and r['v12'] != 'correct')

    # ── 차트 ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cats = ['correct', 'wrong\n(오정정)', 'fail']
    x = np.arange(3)
    w = 0.36
    lv_ = [tl['correct'], tl['wrong'], tl['fail']]
    dv_ = [td['correct'], td['wrong'], td['fail']]
    ax1.bar(x - w / 2, [v / n * 100 for v in lv_], w,
            color='gray', label='방어 전 (cap·verify 없음)')
    ax1.bar(x + w / 2, [v / n * 100 for v in dv_], w,
            color='seagreen', label='방어 후 (1+2층)')
    for xi, (a, b) in zip(x, zip(lv_, dv_)):
        ax1.text(xi - w / 2, a / n * 100 + 1, str(a), ha='center', fontsize=9)
        ax1.text(xi + w / 2, b / n * 100 + 1, str(b), ha='center', fontsize=9)
    ax1.set_xticks(x, cats)
    ax1.set_ylabel('비율 (%)')
    ax1.set_title(f'① 코드워드 레벨 (N={n:,}, nsym={SIM_NSYM}, '
                  f'오류 {SIM_ERR_RANGE[0]}~{SIM_ERR_RANGE[1]-1}개)')
    ax1.grid(alpha=0.3, axis='y')
    ax1.legend()

    xs = [r['blur'] for r in rows]
    cs = [r['vote_conf'] for r in rows]
    color_map = {'correct': 'seagreen', 'wrong': 'crimson', 'fail': 'gray'}
    ax2.plot(xs, cs, '-', color='darkorange', zorder=1)
    for outcome in ('correct', 'wrong', 'fail'):
        pts = [(r['blur'], r['vote_conf']) for r in rows
               if r['legacy'] == outcome]
        if pts:
            ax2.scatter(*zip(*pts), color=color_map[outcome], zorder=2,
                        label=f'방어 전 fine {outcome}'
                              + (' (침묵 오정정)' if outcome == 'wrong' else ''))
    ax2.axhline(FINE_GATE_THRESHOLD, color='navy', ls='--',
                label=f'3층 게이트 임계 {FINE_GATE_THRESHOLD}')
    ax2.set_xlabel('GaussianBlur 반경 (px)')
    ax2.set_ylabel(f'fine vote 신뢰도 (N={GATE_SAMPLE_POINTS} 표결)')
    ax2.set_title('② 게이트 캘리브레이션 — vote 신뢰도 vs 블러')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    chart_path = out_dir / 'exp12_miscorrection_chart.png'
    fig.savefig(chart_path, dpi=110)
    print(f"차트 저장: {chart_path}")

    # ── 보고서 ──
    def rate(c, d=n):
        return f"{c}/{d:,} = {c / d:.2e}" if c else f"0/{d:,} (< {1 / d:.0e})"

    steps_before = ladder_steps(SIM_NSYM, cap=False)
    steps_after = ladder_steps(SIM_NSYM, cap=True)
    lines = [
        "# exp12 침묵 오정정 3층 방어 — 검증 보고서",
        "",
        "## 방어 구성",
        "",
        f"- **1층 사다리 상한**: 최종 소거 단계 nsym → nsym-{LADDER_CAP_SLACK} "
        f"(예: nsym={SIM_NSYM} 사다리 {steps_before} → {steps_after}). "
        "풀소거(nsym)는 미지수=방정식이라 연립이 항상 풀려 비소거 위치의 "
        "오류가 그대로 '성공'으로 통과하는 구조적 침묵 오정정 경로 — 이를 제거.",
        "- **2층 재인코딩 대조** (CRC 아님 — FTO 제약 3 준수): 복호 결과를 "
        "RSCodec으로 재인코딩해 수신 코드워드와 비교, 불일치 위치가 "
        "(소거 표시 ∪ RS 보고 정정 위치)를 벗어나면 실패 처리. 결정론적.",
        f"- **3층 신뢰도 게이트**: fine 스트림 평균 바이트 vote 신뢰도 "
        f"(exp11 표결 메트릭, N={GATE_SAMPLE_POINTS}) < "
        f"**{FINE_GATE_THRESHOLD}** 이면 RS 시도 자체를 생략하고 fine=None. "
        "이산 표 개수만 사용 — 밝기 연속량 불사용 (FTO 제약 1 준수).",
        "",
        f"## ① 코드워드 레벨 시뮬레이션 (N={n:,}, seed={SEED})",
        "",
        f"- 구성: exp10 fine 스트림과 동일 (nsym={SIM_NSYM}, data "
        f"{SIM_DATA_LEN}B, 코드워드 {SIM_DATA_LEN + 2 * SIM_NSYM}B → "
        "reedsolo 2청크 경로 포함)",
        f"- 오류 주입: 시행당 무작위 위치 {SIM_ERR_RANGE[0]}~"
        f"{SIM_ERR_RANGE[1] - 1}개 바이트 XOR (비영 보장)",
        "- 신뢰도 모델: 오류 바이트 U[0,60], 정상 바이트 U[20,128] — 구간을 "
        "겹쳐 신뢰도가 오류 위치를 완벽히 못 가리키는 실측 상황(블러·오염) 모사",
        "",
        "| 결과 | 방어 전 | 방어 후 (1+2층) |",
        "|------|---------|------------------|",
        f"| 정상 복호 (correct) | {tl['correct']:,} ({tl['correct']/n:.1%}) "
        f"| {td['correct']:,} ({td['correct']/n:.1%}) |",
        f"| **침묵 오정정 (wrong)** | **{tl['wrong']:,} ({tl['wrong']/n:.2%})** "
        f"| **{td['wrong']:,} ({td['wrong']/n:.2%})** |",
        f"| 실패 반환 (fail) | {tl['fail']:,} ({tl['fail']/n:.1%}) "
        f"| {td['fail']:,} ({td['fail']/n:.1%}) |",
        "",
        f"- **오정정률 (전체 시행 대비)**: 방어 전 {rate(tl['wrong'])} → "
        f"방어 후 {rate(td['wrong'])}",
        f"- 오정정 차단: 방어 전 오정정 {tl['wrong']:,}건 중 "
        f"{sim['blocked_wrong']:,}건이 방어 후 실패(차단)로 전환",
        f"- **false reject (방어 비용)**: 방어 전 정상 복호였다가 방어 후 "
        f"실패로 바뀐 시행 {rate(sim['false_reject'])}",
        f"- **신규 오정정 (방어가 만든 wrong)**: 방어 전 correct였다가 방어 후 "
        f"wrong이 된 시행 {rate(sim['cross'].get(('correct', 'wrong'), 0))}",
        f"- 소요 시간: {sim['elapsed']:.0f}s",
        "",
        "### 전이표 (방어 전 → 방어 후, 시행 수)",
        "",
        "| 방어 전 \\ 방어 후 | correct | wrong | fail |",
        "|--------------------|---------|-------|------|",
    ] + [
        f"| {lo} | " + " | ".join(
            f"{sim['cross'].get((lo, do_), 0):,}"
            for do_ in ('correct', 'wrong', 'fail')) + " |"
        for lo in ('correct', 'wrong', 'fail')
    ] + [
        "",
    ]
    if sim['survived_detail']:
        det = sim['survived_detail']
        rungs = sorted({d['erasures_used'] for d in det})
        n_err_min = min(d['n_err'] for d in det)
        n_err_max = max(d['n_err'] for d in det)
        lines += [
            "### 잔존 오정정 분석 (방어 후에도 wrong)",
            "",
            f"- 발생 단계: 전부 소거 {rungs}개 단계 (사다리 최종 단계)",
            f"- 주입 오류 수 범위: {n_err_min}~{n_err_max}개 "
            f"(절반거리 {SIM_NSYM // 2}개를 한참 넘는 영역)",
            "- 메커니즘: 최종 단계에서도 RS가 남은 신드롬 여력을 오류 "
            "탐색(BM)에 마저 소진하면(보고 정정 수 > 소거 수) 검증 여력이 "
            "다시 0이 되어 다른 유효 코드워드로 수렴할 수 있다 — 보고된 "
            "정정 위치 안의 불일치라 2층 대조도 통과한다 (RS 반경 밖 복호의 "
            "본질적 한계). 후속 후보: errata 예산 검사 "
            "(청크별 2·오류+소거 ≤ nsym-2 강제).",
            f"- 전이표의 fail→wrong "
            f"{sim['cross'].get(('fail', 'wrong'), 0)}건: 구동작은 "
            "풀소거(nsym) 단계에서 예외를 던지고 끝났지만, 새 최종 "
            "단계(nsym-2)는 BM 오류 탐색 여지가 있어 오히려 잘못된 "
            "코드워드로 '성공' 수렴한 경우. 사다리 상한은 '항상 해가 "
            "존재하는' 풀소거 경로를 제거할 뿐 BM 소진 경로까지 막지는 "
            "못한다 — 총량은 그래도 순감소(방어 전 대비)이며, 일소하려면 "
            "위 errata 예산 검사가 필요.",
            "",
            "| 시행 (상위 10) | 주입 오류 수 | 소거 단계 | RS 보고 정정 수 |",
            "|------|-------------|-----------|----------------|",
        ]
        for d in det[:10]:
            lines.append(f"| {d['trial']} | {d['n_err']} | "
                         f"{d['erasures_used']} | {d['errata_count']} |")
        lines.append("")
    lines += [
        "## ② 3층 게이트 캘리브레이션 (블러 스윕 실측)",
        "",
        f"- 마지막 정상 복호 vote 신뢰도 (blur 5.5px): **{last_ok_conf:.4f}**",
        f"- 침묵 오정정 지점 vote 신뢰도 (blur {MISCORRECTION_BLUR}px): "
        f"**{mis_conf:.4f}**" if mis_conf is not None else
        "- (이번 스윕에서 방어 전 오정정 미관측)",
        f"- **임계값 = 중간값 {FINE_GATE_THRESHOLD}** "
        "(decoder_v12.FINE_GATE_THRESHOLD)",
        "",
        "> **정직 고지 — 비단조성**: vote 신뢰도는 블러에 단조 감소가 아니다. "
        "6.5px 이상에서 0.41~0.44 고원이 형성돼 임계를 넘는다 (exp11 블러 "
        "절벽과 동근원 — t=0.5 외곽 샘플이 이웃 번짐을 흡수해 표가 안정적으로 "
        "갈라짐). 따라서 3층 게이트는 단독 방어가 아니며, 고원 구간에서 RS가 "
        "시도되더라도 1·2층이 결정론적으로 차단한다 (아래 표의 1+2층 열 참조). "
        "게이트의 효용은 ⑴ 관측된 오정정 지점(6.0px)의 사전 차단과 "
        "⑵ 저신뢰 구간에서 RS 시도 비용 절약이다.",
        "",
        "## ③ 이미지 레벨 — blur 6.0px 침묵 오정정 케이스",
        "",
        f"| 단계 | blur {MISCORRECTION_BLUR}px 결과 |",
        "|------|------------------|",
        f"| 방어 전 (exp10 구동작) | **{blur6['legacy']}** "
        f"{'← 침묵 오정정 재현' if blur6['legacy'] == 'wrong' else ''} |",
        f"| 1+2층만 (게이트 무력화) | {blur6['layers12']} |",
        f"| 3층 전체 (v12) | {blur6['v12']} "
        f"(vote 신뢰도 {blur6['vote_conf']:.4f} < {FINE_GATE_THRESHOLD} → "
        "RS 미시도) |",
        "",
        "## ④ 블러 스윕 상세 (이미지 레벨 false reject 포함)",
        "",
        f"- 이미지 레벨 false reject: 방어 전 정상이던 블러 레벨 "
        f"{len(correct_confs)}개 중 v12에서 손실 **{img_false_reject}개**",
        "",
        "| 블러(px) | vote 신뢰도 | 방어 전 | 1+2층 | v12 (3층) |",
        "|----------|------------|---------|-------|-----------|",
    ]
    for r in rows:
        lines.append(f"| {r['blur']:.1f} | {r['vote_conf']:.4f} | "
                     f"{r['legacy']} | {r['layers12']} | {r['v12']} |")
    lines += [
        "",
        "(v12 열: gated = 3층 게이트가 RS 시도 자체를 차단, "
        "fail = RS 시도 후 1·2층/사다리에서 실패 반환)",
        "",
        "## FTO 준수 확인",
        "",
        "- 1·2층: 신뢰도 메트릭과 무관한 대수적 방어 (사다리 상한·재인코딩 "
        "대조) — CRC 불사용 (제약 3).",
        "- 3층: 이산 표 개수(다수표-소수표)만 사용, 밝기 마진·반사율 연속량 "
        "불사용 (제약 1). 샘플 이진화는 전역 고정 임계 128 (제약 2).",
        "",
    ]
    report_path = out_dir / 'exp12_miscorrection_report.md'
    report_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"보고서 저장: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000,
                        help='코드워드 레벨 시행 수 (기본 10,000)')
    args = parser.parse_args()

    print(f"=== ① 코드워드 레벨 시뮬레이션 (N={args.n:,}) ===")
    sim = simulate_codeword_level(args.n)

    print("\n=== ②③④ 이미지 레벨 블러 스윕 ===")
    rows, params, result = image_level_sweep()

    write_outputs(sim, rows, params, result)

    blur6 = next(r for r in rows if r['blur'] == MISCORRECTION_BLUR)
    print("\n=== 요약 ===")
    print(f"오정정률: 방어 전 {sim['tally']['legacy']['wrong']}/{args.n} → "
          f"방어 후 {sim['tally']['defended']['wrong']}/{args.n}")
    print(f"false reject (코드워드): {sim['false_reject']}/{args.n}")
    print(f"blur 6.0px: legacy={blur6['legacy']} → v12={blur6['v12']}")


if __name__ == "__main__":
    main()
