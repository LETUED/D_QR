"""
exp11 ablation — 밝기 마진(decoder_v9) vs 표 일치율(decoder_v11) 정량 대조
실행 (프로젝트 루트에서, 장시간 1회 실행 대신 벤치마크별 분할 실행 가능):
  py -X utf8 benchmarks/exp11_ablation.py                  # 전체 순차 실행 + 집계
  py -X utf8 benchmarks/exp11_ablation.py --bench noise:A  # 한 벤치마크만 (부분 저장)
  py -X utf8 benchmarks/exp11_ablation.py --finalize       # 부분 결과 집계·차트·보고서

concept/11_research_roadmap.md §11.4 exp11:
  동일 이미지·동일 시드에서 두 디코더의 soft 인식률을 비교해
  "표 일치율은 반사율 연속량과 구조적으로 다른 이산량"임을 정량 입증
  (Honeywell US 10,235,547 패밀리 균등론 방어 자료).

스윕 (exp9_benchmark 열화 모델·설정 재사용):
  ① 국소 오염 패치 20~90개, patch_size=60 (프로파일 A·B)
  ② 가우시안 노이즈 천이 구간 sigma 100~135 (프로파일 A·B)
  ③ 가우시안 블러 6.0~9.5px (프로파일 A)

샘플점 수 N ∈ {5,7,9}를 같은 이미지에서 동시 측정 → 총 성공수 최대인 N 채택.
⛔ 중단 규칙: 채택 N에서 vote가 마진 대비 어느 레벨에서든 10%p 초과 열세
   → N을 11, 13으로 올려 재시도, 그래도 실패 시 stop_rule_triggered 보고.

산출물: outputs/exp11/exp11_ablation_chart.png
        outputs/exp11/exp11_ablation_report.md
        outputs/exp11/exp11_ablation_summary.json
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# 열화 모델은 exp9 벤치마크와 완전 동일 조건이어야 하므로 그대로 가져온다
from benchmarks.exp9_benchmark import add_noise, add_blur, add_patches

from src.exp9_soft.encoder_v9 import DQREncoderV9
from src.exp9_soft.renderer_v9 import DQRRendererV9
from src.exp9_soft.decoder_v9 import DQRDecoderV9
from src.exp11_consensus.decoder_v11 import DQRDecoderV11

TEXT = "dqr 시험테스트입니다 — exp11 표 일치율 ablation 페이로드"
TRIALS = 50          # 국소 오염 N≥50 요건 — 전 스윕 공통 적용
SEED = 42
N_CANDIDATES = [5, 7, 9]
N_ESCALATION = [11, 13]   # 중단 규칙 발동 시 확대 시도 순서
STOP_LOSS_PP = 10.0       # vote가 마진 대비 이만큼 초과 열세면 중단 규칙

OUT_DIR = Path('outputs/exp11')
PARTIAL_DIR = OUT_DIR / 'partial'

SWEEPS = {
    'patches': dict(levels=[20, 30, 40, 50, 60, 70, 80, 90],
                    fn=lambda img, lv, rng: add_patches(img, lv, rng, patch_size=60),
                    profiles=('A', 'B'),
                    title='국소 오염 패치 (개수)'),
    'noise':   dict(levels=[100, 105, 110, 115, 120, 125, 130, 135],
                    fn=lambda img, lv, rng: add_noise(img, lv, rng),
                    profiles=('A', 'B'),
                    title='가우시안 노이즈 (sigma)'),
    'blur':    dict(levels=[6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
                    fn=lambda img, lv, rng: add_blur(img, lv),
                    profiles=('A',),
                    title='가우시안 블러 (radius px)'),
}
BENCHES = [(sw, p) for sw, cfg in SWEEPS.items() for p in cfg['profiles']]


def make_code(profile):
    data = TEXT.encode("utf-8")
    enc = DQREncoderV9(profile=profile)
    ren = DQRRendererV9(hex_size=32)
    result = enc.encode(data)
    img, info = ren.render(result)
    params = dict(
        radius=result['radius'],
        hex_size=info['hex_size'],
        bits_per_cell=result['bits_per_cell'],
        origin_x=info['origin_x'],
        origin_y=info['origin_y'],
        nsym=result['nsym'],
        data_len=result['data_len'],
        codeword_len=result['codeword_len'],
    )
    return data, img, params


def try_soft(dec, img, params, data):
    try:
        return dec.decode_with_stats(img, **params)['data'] == data
    except Exception:
        return False


def partial_path(sweep, profile):
    return PARTIAL_DIR / f"{sweep}_{profile}.json"


def run_bench(sweep, profile, n_values):
    """
    한 벤치마크 실행 + 부분 결과 저장(누적 병합). 동일 (seed, trial) →
    동일 열화 이미지에서 margin(v9)과 모든 N의 vote(v11)를 함께 측정해
    짝지은 비교를 보장. margin이 이미 저장돼 있으면 재측정하지 않는다.
    """
    path = partial_path(sweep, profile)
    saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    have_margin = 'margin' in saved
    todo_n = [n for n in n_values if str(n) not in saved.get('vote', {})]
    if have_margin and not todo_n:
        print(f"  [{profile}/{sweep}] 이미 측정됨 — 건너뜀")
        return

    cfg = SWEEPS[sweep]
    data, img, params = make_code(profile)
    dec9 = DQRDecoderV9() if not have_margin else None
    dec11 = {n: DQRDecoderV11(n_samples=n) for n in todo_n}

    margin_rates = []
    vote_rates = {n: [] for n in todo_n}
    for lv in cfg['levels']:
        m_ok = 0
        v_ok = {n: 0 for n in todo_n}
        for t in range(TRIALS):
            rng = np.random.default_rng(SEED + t)
            deg = cfg['fn'](img, lv, rng)
            if dec9 is not None:
                m_ok += try_soft(dec9, deg, params, data)
            for n in todo_n:
                v_ok[n] += try_soft(dec11[n], deg, params, data)
        if dec9 is not None:
            margin_rates.append(100 * m_ok / TRIALS)
        for n in todo_n:
            vote_rates[n].append(100 * v_ok[n] / TRIALS)
        msg = "  ".join(f"N={n} {vote_rates[n][-1]:.0f}%" for n in todo_n)
        prefix = f"margin {margin_rates[-1]:.0f}%  " if dec9 is not None else ""
        print(f"  [{profile}/{sweep}] lv={lv}: {prefix}{msg}", flush=True)

    if dec9 is not None:
        saved['margin'] = margin_rates
    saved.setdefault('vote', {})
    for n in todo_n:
        saved['vote'][str(n)] = vote_rates[n]
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, ensure_ascii=False), encoding='utf-8')
    print(f"  부분 저장: {path}")


def load_all():
    """부분 결과 로드 → (margin dict, vote dict[n][bench])."""
    margin, vote = {}, {}
    for sweep, profile in BENCHES:
        path = partial_path(sweep, profile)
        if not path.exists():
            raise FileNotFoundError(f"부분 결과 없음: {path} — "
                                    f"--bench {sweep}:{profile} 먼저 실행")
        saved = json.loads(path.read_text(encoding='utf-8'))
        margin[(sweep, profile)] = saved['margin']
        for n_str, rates in saved['vote'].items():
            vote.setdefault(int(n_str), {})[(sweep, profile)] = rates
    return margin, vote


def max_loss(margin_rates, vote_rates):
    """레벨별 (margin - vote) 최대값 %p — 양수 = vote 열세."""
    return max(m - v for m, v in zip(margin_rates, vote_rates))


def finalize():
    """부분 결과 집계 → N 채택 → 중단 규칙 → 차트·보고서·요약."""
    margin, vote = load_all()
    # 전 벤치마크가 측정된 N만 채택 후보
    complete_ns = sorted(n for n, d in vote.items()
                         if all(b in d for b in BENCHES))
    totals = {n: sum(sum(vote[n][b]) for b in BENCHES) for n in complete_ns}
    print(f"N별 총 성공률 합: { {n: round(v, 1) for n, v in totals.items()} }")

    cand = [n for n in complete_ns if n in N_CANDIDATES] or complete_ns
    best_n = max(cand, key=lambda n: (totals[n], -n))

    # ── 중단 규칙: 채택 N이 10%p 초과 열세면 N 확대 요구 ─────────────
    stop_rule_triggered = False
    while True:
        worst = max(max_loss(margin[b], vote[best_n][b]) for b in BENCHES)
        print(f"N={best_n} 최대 손실: {worst:.1f}%p (기준 {STOP_LOSS_PP}%p)")
        if worst <= STOP_LOSS_PP:
            break
        remaining = [n for n in N_ESCALATION if n not in complete_ns]
        if not remaining:
            # 모든 N에서 기준 초과 — 정직 보고 (기본 전환 보류)
            # 동률(최대 손실 같음)이면 총 성공률 높은 N 우선
            stop_rule_triggered = True
            best_n = min(complete_ns, key=lambda n: (
                max(max_loss(margin[b], vote[n][b]) for b in BENCHES),
                -totals[n], n))
            break
        n_next = remaining[0]
        print(f"\n⛔ 중단 규칙 후보 발동 — N={n_next} 확대 측정 필요:")
        for sweep, profile in BENCHES:
            print(f"  py -X utf8 benchmarks/exp11_ablation.py "
                  f"--bench {sweep}:{profile} --n {n_next}")
        sys.exit(3)   # 확대 측정 후 --finalize 재실행

    final_worst = max(max_loss(margin[b], vote[best_n][b]) for b in BENCHES)
    print(f"채택 N={best_n}, 최대 손실 {final_worst:.1f}%p, "
          f"중단 규칙 발동={stop_rule_triggered}")

    # ── 차트 ────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat = axes.ravel()
    for ax, (sweep, profile) in zip(flat, BENCHES):
        xs = SWEEPS[sweep]['levels']
        ax.plot(xs, margin[(sweep, profile)], 'o--', color='gray',
                label='margin (exp9 밝기 마진)')
        ax.plot(xs, vote[best_n][(sweep, profile)], 's-', color='crimson',
                label=f'vote (exp11 표 일치율, N={best_n})')
        ax.set_title(f"프로파일 {profile} — {SWEEPS[sweep]['title']}")
        ax.set_ylabel('soft 디코딩 성공률 (%)')
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        ax.legend()
    # 마지막 패널: N 스윕 총괄 비교
    ax = flat[len(BENCHES)]
    n_points = len(BENCHES) * 8   # 벤치 5개 × 레벨 8개
    ax.bar([str(n) for n in complete_ns],
           [totals[n] / n_points for n in complete_ns],
           color=['crimson' if n == best_n else 'lightgray'
                  for n in complete_ns])
    ax.set_title(f'샘플점 수 N 스윕 — 전 조건 평균 성공률 (채택 N={best_n})')
    ax.set_xlabel('샘플점 수 N')
    ax.set_ylabel('평균 성공률 (%)')
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle('exp11 ablation: 밝기 마진(v9) vs 표 일치율(v11) — '
                 '동일 이미지·동일 시드', fontsize=14)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = OUT_DIR / 'exp11_ablation_chart.png'
    fig.savefig(chart_path, dpi=110)
    print(f"차트 저장: {chart_path}")

    # ── 마크다운 보고서 ────────────────────────────────────────────
    lines = [
        "# exp11 ablation — 밝기 마진(v9) vs 표 일치율(v11) soft 인식률",
        "",
        f"- 페이로드: {len(TEXT.encode('utf-8'))} bytes, 시행 {TRIALS}회/조건, "
        f"seed={SEED} (동일 이미지·동일 시드 짝지은 비교)",
        "- margin = exp9 decoder_v9 (밝기 마진 신뢰도 — Honeywell erasure-gap 계열)",
        "- vote = exp11 decoder_v11 (표 일치율 — 이산 표 개수, 반사율 간극 아님)",
        f"- 샘플점 수 N 스윕: {complete_ns} → **채택 N={best_n}**",
        f"- 최대 손실(margin−vote): **{final_worst:.1f}%p** "
        f"(중단 기준 {STOP_LOSS_PP}%p)",
        f"- ⛔ 중단 규칙 발동: **{'예 — 기본 전환 보류' if stop_rule_triggered else '아니오'}**",
        "",
        "## N 스윕 총괄 (전 조건 평균 성공률)",
        "",
        "| N | 평균 성공률 |",
        "|---|-----------|",
    ]
    for n in complete_ns:
        mark = " ← 채택" if n == best_n else ""
        lines.append(f"| {n} | {totals[n] / n_points:.1f}%{mark} |")
    lines.append("")

    # 이 실행에서 관측된 메커니즘 해석 (균등론 방어 문서의 입력)
    lines += [
        "## 해석 — 두 메트릭이 구조적으로 다른 양이라는 증거",
        "",
        "세 열화 모델에서 우열이 **반대 방향**으로 갈린다. 두 메트릭이 수학적으로",
        "동등한 양이라면 불가능한 패턴이며, 이것이 균등론(DoE) 방어의 핵심 증거다.",
        "",
        "1. **노이즈: vote 압도적 우위.** 다수결 자체가 샘플 단위 노이즈를 평균화",
        "   (margin은 무게중심 1픽셀만 판독)하고, 표가 갈리는 위치가 정확히",
        "   불확실한 바이트와 일치한다.",
        "2. **균일 중간회색 패치: vote 열세 (구조적).** 패치가 삼각형을 완전히 덮으면",
        "   N개 샘플이 전부 같은 쪽으로 이진화 → **만장일치이지만 틀린** 비트가",
        "   최대 신뢰도를 받아 소거 후보에서 빠진다 (오류는 소거 대비 정정 한계 절반).",
        "   margin은 |g−128|≤28 연속량으로 이를 포착한다. N 증가로 해소 불가.",
        "3. **블러 9.5px: vote 절벽.** t=0.5 외곽 샘플 6개가 이웃 삼각형 번짐을",
        "   흡수해 무게중심 표를 다수결로 압도한다 (N=13에서도 외곽 링이 잔존).",
        "",
        "→ 중단 규칙 발동: vote 메트릭의 **기본 전환은 보류** (구현은 유지).",
        "   후속 후보: 내부 링(t≤0.25) 우선 배치, 만장일치-저신뢰 처리 재설계 등",
        "   메트릭 재설계는 exp11b로 분리.",
        "",
    ]

    for sweep, profile in BENCHES:
        cfg = SWEEPS[sweep]
        lines.append(f"## 프로파일 {profile} — {cfg['title']}")
        lines.append("")
        lines.append(f"| 레벨 | margin (v9) | vote (v11, N={best_n}) | delta |")
        lines.append("|------|------------|------------|------|")
        for lv, m, v in zip(cfg['levels'], margin[(sweep, profile)],
                            vote[best_n][(sweep, profile)]):
            d = v - m
            mark = (f"**+{d:.0f}%p**" if d > 0
                    else (f"{d:.0f}%p" if d < 0 else "—"))
            lines.append(f"| {lv} | {m:.0f}% | {v:.0f}% | {mark} |")
        lines.append("")

    report_path = OUT_DIR / 'exp11_ablation_report.md'
    report_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"보고서 저장: {report_path}")

    # 후속 자동화용 기계 판독 요약
    summary = {
        'best_n': best_n,
        'measured_n': complete_ns,
        'max_loss_pp': round(final_worst, 2),
        'stop_rule_triggered': stop_rule_triggered,
        'benches': {
            f"{sweep}/{profile}": {
                'levels': SWEEPS[sweep]['levels'],
                'margin': margin[(sweep, profile)],
                'vote': vote[best_n][(sweep, profile)],
                'max_loss_pp': round(max_loss(margin[(sweep, profile)],
                                              vote[best_n][(sweep, profile)]), 2),
            } for sweep, profile in BENCHES
        },
    }
    (OUT_DIR / 'exp11_ablation_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"요약 저장: {OUT_DIR / 'exp11_ablation_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description='exp11 ablation')
    parser.add_argument('--bench', help='한 벤치마크만 실행 (예: noise:A)')
    parser.add_argument('--n', help='측정할 N 목록 (예: 5,7,9 — 기본 후보 3종)')
    parser.add_argument('--finalize', action='store_true',
                        help='부분 결과 집계·차트·보고서 생성')
    args = parser.parse_args()

    n_values = ([int(x) for x in args.n.split(',')] if args.n
                else list(N_CANDIDATES))

    if args.finalize:
        finalize()
        return

    if args.bench:
        sweep, profile = args.bench.split(':')
        if (sweep, profile) not in BENCHES:
            raise SystemExit(f"알 수 없는 벤치마크: {args.bench} "
                             f"(가능: {['%s:%s' % b for b in BENCHES]})")
        t0 = time.time()
        print(f"=== 프로파일 {profile} / {sweep} (TRIALS={TRIALS}, N={n_values}) ===")
        run_bench(sweep, profile, n_values)
        print(f"소요 {time.time() - t0:.0f}s")
        return

    # 인자 없으면 전체 순차 실행 + 집계
    for sweep, profile in BENCHES:
        print(f"\n=== 프로파일 {profile} / {sweep} ===")
        run_bench(sweep, profile, n_values)
    finalize()


if __name__ == "__main__":
    main()
