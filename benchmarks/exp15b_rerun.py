"""
exp15b — 전부-0 코드워드 퇴화 봉쇄 재측정 (4층 가드 적용 v11b)

exp15에서 발견된 D_QR 유일의 침묵 오정정 7건(전부 전부-0 코드워드 퇴화)을
가드(softdecode.is_degenerate_message, v11b 기본 활성)가 봉쇄하는지,
성공률 손실 없이(7건은 어차피 틀린 반환 → 거부로 전환) 재측정한다.

측정 ① exp15 블롯 채널(blot 1/2/3) 재실행 — exp15와 동일 페이로드·동일
  면적 정규화·동일 시드(SEED+trial) → 동일 열화 이미지. D_QR만 재측정
  (QR·DM 판독 경로는 변경 없음 — exp15 partial 결과를 그대로 인용).
  목표: 오정정 7→0, ok 건수 변동 0.

측정 ② exp11b 이미지 오정정 스위트(5 벤치 × 8 레벨 × 50 = 2,000 이미지)
  재실행 — v11b(가드)만. 목표: 오정정 0 유지 + 성공률 exp11b 측정과 동일
  (전부-0 모드는 blot 시나리오 전용이라 이 스위트의 성공 경로와 무관).

측정 ③ exp15 실측 오정정 케이스(exp15_wrong_blot2_trial8.png) 직접 재현 —
  가드 해제 시 전부-0 재현, 가드 적용 시 거부 (tests/test_exp15b_guard.py
  테스트 ③과 동일 검사의 벤치마크 기록판).

실행 (프로젝트 루트에서):
  py -X utf8 benchmarks/exp15b_rerun.py            # 전체 + 보고서
  py -X utf8 benchmarks/exp15b_rerun.py --finalize # 집계·보고서만

산출물: outputs/exp15b/exp15b_report.md
        outputs/exp15b/exp15b_summary.json
        outputs/exp15b/partial/*.json (중간 저장 — 재실행 시 건너뜀)
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from reedsolo import ReedSolomonError

# exp15와 동일 조건(페이로드·면적 정규화·시드·열화 모델)을 그대로 가져온다
from benchmarks.exp15_baseline import (CHANNELS, PAYLOAD, SEED, TRIALS,
                                       build_dqr, classify)
# exp11b 스위트와 동일 조건 (exp11 ablation의 열화·코드 생성 재사용)
from benchmarks.exp11_ablation import BENCHES, SWEEPS, make_code

from src.exp11_consensus.decoder_v11b import DQRDecoderV11B

OUT_DIR = Path('outputs/exp15b')
PARTIAL_DIR = OUT_DIR / 'partial'
EXP15_PARTIAL = Path('outputs/exp15/partial')
EXP11B_SUMMARY = Path('outputs/exp11b/exp11b_ablation_summary.json')
WRONG_PNG = Path('outputs/exp15/exp15_wrong_blot2_trial8.png')

BLOT_LEVELS = (1, 2, 3)
SUITE_TRIALS = 50          # exp11b ablation과 동일


# ── 측정 ① exp15 블롯 채널 재실행 (D_QR만, 가드 on/off 동시 계수) ────────

def run_blot(level):
    path = PARTIAL_DIR / f"blot_{level}.json"
    if path.exists():
        print(f"  [blot lv={level}] 이미 측정됨 — 건너뜀")
        return
    sym = build_dqr()                       # 가드 on (v11b 기본)
    dec_off = DQRDecoderV11B(reject_degenerate=False)
    params = sym['params']
    cfg = CHANNELS['blot']

    cnt_on = {'ok': 0, 'wrong': 0, 'fail': 0}
    cnt_off = {'ok': 0, 'wrong': 0, 'fail': 0}
    wrong_trials_off, wrong_trials_on = [], []
    t0 = time.time()
    for t in range(TRIALS):
        # exp15 run_condition과 동일: 심볼로지별 rng 재생성(SEED+t) →
        # exp15에서 D_QR이 본 것과 같은 열화 이미지
        rng = np.random.default_rng(SEED + t)
        deg = cfg['fn'](sym['canvas'], level, rng)

        out_on = sym['read'](deg)
        c_on = classify(out_on)
        cnt_on[c_on] += 1
        if c_on == 'wrong':
            wrong_trials_on.append(t)

        try:
            out_off = dec_off.decode_with_stats(deg, **params)['data']
        except Exception:
            out_off = None
        c_off = classify(out_off)
        cnt_off[c_off] += 1
        if c_off == 'wrong':
            wrong_trials_off.append(t)

    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'trials': TRIALS,
        'guarded': cnt_on, 'unguarded': cnt_off,
        'wrong_trials_guarded': wrong_trials_on,
        'wrong_trials_unguarded': wrong_trials_off,
    }, ensure_ascii=False), encoding='utf-8')
    print(f"  [blot lv={level}] 가드 on: ok={cnt_on['ok']} W={cnt_on['wrong']} "
          f"/ off: ok={cnt_off['ok']} W={cnt_off['wrong']} "
          f"(off W 시행 {wrong_trials_off})  ({time.time() - t0:.0f}s)",
          flush=True)


# ── 측정 ② exp11b 2,000 이미지 스위트 재실행 (v11b 가드만) ──────────────

def run_suite_bench(sweep, profile):
    path = PARTIAL_DIR / f"suite_{sweep}_{profile}.json"
    if path.exists():
        print(f"  [{profile}/{sweep}] 이미 측정됨 — 건너뜀")
        return
    cfg = SWEEPS[sweep]
    data, img, params = make_code(profile)
    dec = DQRDecoderV11B()                  # 가드 on (기본)

    rates, wrongs = [], []
    for lv in cfg['levels']:
        cnt = {'ok': 0, 'wrong': 0, 'fail': 0}
        for t in range(SUITE_TRIALS):
            rng = np.random.default_rng(SEED + t)
            deg = cfg['fn'](img, lv, rng)
            try:
                out = dec.decode_with_stats(deg, **params)['data']
                cnt['ok' if out == data else 'wrong'] += 1
            except Exception:
                cnt['fail'] += 1
        rates.append(100.0 * cnt['ok'] / SUITE_TRIALS)
        wrongs.append(cnt['wrong'])
        print(f"  [{profile}/{sweep}] lv={lv}: {rates[-1]:.0f}%"
              + (f" W({wrongs[-1]})" if wrongs[-1] else ""), flush=True)

    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'rates': rates, 'wrongs': wrongs},
                               ensure_ascii=False), encoding='utf-8')


# ── 측정 ③ 실측 오정정 케이스 직접 재현 ─────────────────────────────────

def run_case_repro():
    path = PARTIAL_DIR / "case_blot2_trial8.json"
    if path.exists():
        print("  [case] 이미 측정됨 — 건너뜀")
        return
    if not WRONG_PNG.exists():
        raise FileNotFoundError(f"증거 이미지 없음: {WRONG_PNG}")
    sym = build_dqr()
    img = Image.open(WRONG_PNG)

    try:
        res = DQRDecoderV11B(reject_degenerate=False).decode_with_stats(
            img, **sym['params'])
        before = {'outcome': classify(res['data']),
                  'all_zero': res['data'] == b"\x00" * len(PAYLOAD),
                  'mode': res['mode'], 'erasures': res['erasures_used'],
                  'errata': res['errata_count']}
    except Exception as e:
        before = {'outcome': 'fail', 'exc': type(e).__name__}

    try:
        res = DQRDecoderV11B().decode_with_stats(img, **sym['params'])
        after = {'outcome': classify(res['data'])}
    except ReedSolomonError as e:
        after = {'outcome': 'fail', 'reason': str(e)[:120]}
    except Exception as e:
        after = {'outcome': 'fail', 'exc': type(e).__name__}

    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'before': before, 'after': after},
                               ensure_ascii=False, indent=2),
                    encoding='utf-8')
    print(f"  [case] 가드 전: {before} → 가드 후: {after}", flush=True)


# ── 집계·보고서 ──────────────────────────────────────────────────────────

def finalize():
    # 측정 ① 집계 + exp15 원본 partial 대조
    blot_rows = []
    total_off_w = total_on_w = 0
    ok_delta_total = 0
    for lv in BLOT_LEVELS:
        cur = json.loads((PARTIAL_DIR / f"blot_{lv}.json")
                         .read_text(encoding='utf-8'))
        orig = json.loads((EXP15_PARTIAL / f"blot_{lv}.json")
                          .read_text(encoding='utf-8'))['counts']['dqr']
        on, off = cur['guarded'], cur['unguarded']
        repro_ok = (off['ok'] == orig['ok'] and off['wrong'] == orig['wrong'])
        ok_delta = on['ok'] - orig['ok']
        ok_delta_total += ok_delta
        total_off_w += off['wrong']
        total_on_w += on['wrong']
        blot_rows.append((lv, orig, off, on, repro_ok, ok_delta,
                          cur['wrong_trials_unguarded']))

    # 측정 ② 집계 + exp11b 원본 대조
    suite_rows, suite_w_total = [], 0
    exp11b = json.loads(EXP11B_SUMMARY.read_text(encoding='utf-8'))
    for sweep, profile in BENCHES:
        cur = json.loads((PARTIAL_DIR / f"suite_{sweep}_{profile}.json")
                         .read_text(encoding='utf-8'))
        orig = exp11b['benches'][f"{sweep}/{profile}"]
        suite_w_total += sum(cur['wrongs'])
        suite_rows.append((sweep, profile, SWEEPS[sweep]['levels'],
                           orig['rates']['v11b'], cur['rates'],
                           orig['wrongs']['v11b'], cur['wrongs']))

    case = json.loads((PARTIAL_DIR / "case_blot2_trial8.json")
                      .read_text(encoding='utf-8'))

    n_suite = len(BENCHES) * 8 * SUITE_TRIALS
    goal_blot = total_on_w == 0 and ok_delta_total == 0
    goal_suite = suite_w_total == 0
    goal_case = (case['before'].get('all_zero') is True
                 and case['after']['outcome'] == 'fail')

    lines = [
        "# exp15b — 전부-0 코드워드 퇴화 봉쇄: \"무오정정\" 주장의 "
        "마지막 조각",
        "",
        "exp15 기준선 비교에서 D_QR(v11b)의 유일한 침묵 오정정 7건은 전부 "
        "**전부-0 코드워드 퇴화** 단일 모드였다 — RS 부호의 선형성상 전부-0은 "
        "항상 유효 코드워드라 재인코딩 대조(2층)도 통과하고, v11b 3층 방어가 "
        "무력했다. exp15b는 이를 2층으로 봉쇄한다:",
        "",
        "1. **디코더 가드 (4층째 방어, 필수)** — 복호 결과 메시지가 전부-0 "
        "또는 상수(모든 바이트 동일)면 거부 "
        "(`src/core/softdecode.is_degenerate_message`). v11b·v13의 사다리·"
        "공간 소거 스윕 양 경로에 적용, exp9~12 기본 동작은 옵션 기본값 "
        "False로 동결. 결정론적이며 검출 잉여(체크섬)를 쓰지 않으므로 CRC "
        "아님(FTO 제약 3 무관), 밝기 연속량도 불사용.",
        "2. **인코더 스크램블 (벨트+서스펜더)** — QR 마스크 XOR과 동등한 "
        "LFSR 고정 의사난수 마스크(`src/core/scramble`, 탭 0xB8·시드 0xA7, "
        "주기 255, 전 바이트 비0)를 RS 인코딩 전 메시지에 XOR. 전부-0 근방 "
        "페이로드(빈 데이터·0 패딩)도 비자명 코드워드가 되어 가드가 정상 "
        "페이로드를 거부할 일이 구조적으로 없다. v13은 기본 on(포맷 링 구 "
        "예비 비트 1개에 기록 — 구 코드는 0으로 파싱되어 호환), v9 경로는 "
        "호환성 위해 기본 off 옵션. 검출 기능이 없는 전단사 변환 — CRC 아님.",
        "",
        "## 측정 ③ — exp15 실측 오정정 케이스 직접 재현 "
        "(exp15_wrong_blot2_trial8.png)",
        "",
        "| 구분 | 결과 |",
        "|------|------|",
        f"| 가드 해제 (종전 v11b) | {case['before']} |",
        f"| 가드 적용 (exp15b) | {case['after']} |",
        f"| 판정 | {'**봉쇄 확인**' if goal_case else '**FAIL**'} — "
        "종전 오정정(전부-0, mode=soft, 소거 25, errata 48)이 그대로 "
        "재현되고, 가드가 같은 입력을 명시적으로 거부 |",
        "",
        f"## 측정 ① — exp15 블롯 채널 재실행 (N={TRIALS}/레벨, seed={SEED}, "
        "동일 열화 이미지)",
        "",
        "QR·DataMatrix 판독 경로는 변경이 없으므로 exp15 측정값을 그대로 "
        "인용하고, D_QR만 가드 전/후를 같은 이미지에서 재측정했다. '가드 전 "
        "재현' 열이 exp15 원본과 일치해야 동일 조건임이 입증된다.",
        "",
        "| 레벨 | exp15 원본 (ok/W/fail) | 가드 전 재현 | 가드 후 (exp15b) "
        "| 재현 일치 | ok 변동 | 종전 W 시행 |",
        "|------|------------------------|--------------|------------------"
        "|-----------|---------|-------------|",
    ]
    for lv, orig, off, on, repro_ok, ok_delta, w_trials in blot_rows:
        lines.append(
            f"| blot={lv} | {orig['ok']}/{orig['wrong']}/{orig['fail']} "
            f"| {off['ok']}/{off['wrong']}/{off['fail']} "
            f"| {on['ok']}/**{on['wrong']}**/{on['fail']} "
            f"| {'일치' if repro_ok else '**불일치**'} "
            f"| {ok_delta:+d} | {w_trials} |")
    lines += [
        "",
        f"- 침묵 오정정: 가드 전 {total_off_w}건 → 가드 후 "
        f"**{total_on_w}건** (목표 7→0: "
        f"{'달성' if total_on_w == 0 else '**미달**'})",
        f"- 성공(ok) 건수 변동 합계: {ok_delta_total:+d} (목표 0: "
        f"{'달성' if ok_delta_total == 0 else '**미달**'}) — 종전 오정정은 "
        "어차피 틀린 반환이었으므로 전건 거부(fail)로 전환될 뿐, 정상 복호는 "
        "건드리지 않는다.",
        "",
        f"## 측정 ② — exp11b 이미지 오정정 스위트 재실행 "
        f"({n_suite:,} 이미지, v11b 가드)",
        "",
        "exp11b ablation과 동일 조건(5 벤치 × 8 레벨 × 50시행, seed=42). "
        "괄호 안은 exp11b 원본 v11b 성공률 — 가드가 정상 복호를 잃지 않음을 "
        "확인한다.",
        "",
    ]
    for sweep, profile, levels, orig_r, cur_r, orig_w, cur_w in suite_rows:
        diffs = [c - o for c, o in zip(cur_r, orig_r)]
        max_diff = max(abs(d) for d in diffs)
        lines.append(f"### {profile} / {SWEEPS[sweep]['title']}")
        lines.append("")
        lines.append("| 레벨 | " + " | ".join(str(lv) for lv in levels) + " |")
        lines.append("|------|" + "|".join(["----"] * len(levels)) + "|")
        lines.append("| 가드 v11b | " + " | ".join(
            f"{c:.0f}%" + (f" W({w})" if w else "")
            for c, w in zip(cur_r, cur_w)) + " |")
        lines.append("| exp11b 원본 | " + " | ".join(
            f"{o:.0f}%" + (f" W({w})" if w else "")
            for o, w in zip(orig_r, orig_w)) + " |")
        lines.append("")
        lines.append(f"성공률 최대 편차 {max_diff:.0f}%p, 가드 후 오정정 "
                     f"{sum(cur_w)}건.")
        lines.append("")
    lines += [
        f"- 스위트 전체 침묵 오정정: **{suite_w_total}건** / {n_suite:,} "
        f"이미지 (목표 0 유지: {'달성' if goal_suite else '**미달**'})",
        "",
        "## \"무오정정\" 주장의 최종 상태",
        "",
    ]
    all_pass = goal_blot and goal_suite and goal_case
    if all_pass:
        lines += [
            "exp15 보고서의 단서(\"전부-0 거부 가드 적용 전\")는 **이 측정으로 "
            "해소되었다.** 갱신된 주장:",
            "",
            "> D_QR(v11b+가드)은 측정된 전 조건 — exp15 기준선 비교 1,800 "
            "이미지(블롯 채널은 가드 적용 재측정), exp11b 스위트 2,000 이미지, "
            "exp12 코드워드 시뮬 10,000회 — 에서 **침묵 오정정 0건**이다. "
            "틀린 데이터를 성공으로 반환하지 않는다.",
            "",
            "방어는 4층 + 인코더 1층:",
            "",
            "| 층 | 방어 | 도입 | 차단 대상 |",
            "|----|------|------|----------|",
            "| 1 | 사다리 상한 (nsym-2) | exp12 | 소거 전량 사용 시 검증 "
            "여력 0 |",
            "| 2 | 재인코딩 대조 | exp12 | 보고되지 않은 위치 변경 (CRC "
            "아님) |",
            "| 3 | errata 예산 + 최종 단계 순수 소거 | exp11b | BM 신드롬 "
            "여력 재소진 |",
            "| 4 | **퇴화 코드워드 가드** | **exp15b** | **전부-0/상수 "
            "흡인자 (선형성상 1~3층 통과)** |",
            "| 인코더 | LFSR 스크램블 (v13 기본 on) | exp15b | 정상 "
            "페이로드의 퇴화 코드워드 근방 진입 자체 |",
            "",
            "한계 (정직 보고): '무오정정'은 측정된 열화 모델·조건에 대한 "
            "실측 주장이며 수학적 증명이 아니다. RS 유일 복호 반경 밖의 "
            "수락(공간 스윕 합의)은 잔여 확률적 오수락 가능성이 원리적으로 "
            "남는다 — 다만 4층 방어와 합의 규칙이 알려진 모든 실측 모드를 "
            "차단했고, 새 모드가 관측되면 같은 방법(전수 추적 → 결정론 "
            "가드)으로 봉쇄한다. v9 경로(포맷 링 없음)는 스크램블 여부를 "
            "수신 측과 약속해야 하므로 기본 off — 가드(디코더 측)만으로도 "
            "퇴화 모드는 차단된다.",
        ]
    else:
        lines += [
            "**목표 미달 — 단서 유지 필요.** 위 표에서 미달 항목을 확인할 것.",
        ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / 'exp15b_report.md'
    report_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"보고서 저장: {report_path}")

    summary = {
        'goal_blot_7_to_0': goal_blot,
        'goal_suite_0': goal_suite,
        'goal_case_repro': goal_case,
        'all_pass': all_pass,
        'blot': {str(lv): {'orig': orig, 'unguarded': off, 'guarded': on,
                           'repro_match': repro_ok, 'ok_delta': ok_delta}
                 for lv, orig, off, on, repro_ok, ok_delta, _ in blot_rows},
        'suite_wrong_total': suite_w_total,
        'suite': {f"{s}/{p}": {'levels': lvs, 'orig_rates': orr,
                               'guarded_rates': cr, 'orig_wrongs': ow,
                               'guarded_wrongs': cw}
                  for s, p, lvs, orr, cr, ow, cw in suite_rows},
        'case_blot2_trial8': case,
    }
    (OUT_DIR / 'exp15b_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"요약 저장: {OUT_DIR / 'exp15b_summary.json'}")
    print(f"\n=== 판정: blot 7→{total_on_w} ({'PASS' if goal_blot else 'FAIL'})"
          f" / 스위트 W={suite_w_total} ({'PASS' if goal_suite else 'FAIL'})"
          f" / 케이스 재현 ({'PASS' if goal_case else 'FAIL'}) ===")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description='exp15b 가드 재측정')
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if args.finalize:
        finalize()
        return

    print("=== 측정 ③ 실측 케이스 재현 ===")
    run_case_repro()
    print("=== 측정 ① exp15 블롯 채널 (가드 on/off) ===")
    for lv in BLOT_LEVELS:
        run_blot(lv)
    print("=== 측정 ② exp11b 이미지 스위트 (v11b 가드) ===")
    for sweep, profile in BENCHES:
        print(f"--- {profile}/{sweep} ---")
        run_suite_bench(sweep, profile)
    finalize()


if __name__ == "__main__":
    main()
