"""
exp10 블러 스윕 벤치마크 — coarse/fine 스트림별 생존 경계 측정
실행: py -X utf8 benchmarks/exp10_blur_sweep.py  (프로젝트 루트에서)

GaussianBlur 0~10px(0.5px 간격, 블러는 결정적)에서 각 스트림의
복호 성공 여부와 fine 신뢰도를 기록 → 점진적 공개의 작동 구간
(coarse만 열리는 중간 대역)을 정량화.

산출물: outputs/exp10/exp10_blur_sweep_chart.png
        outputs/exp10/exp10_blur_sweep_report.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import ImageFilter

from src.exp10_progressive.encoder_v10 import DQREncoderV10
from src.exp10_progressive.renderer_v10 import DQRRendererV10
from src.exp10_progressive.decoder_v10 import DQRDecoderV10

COARSE_TEXT = "https://dqr.kr/v10 — exp10 코어 페이로드"
FINE_TEXT = ("exp10 확장 페이로드: 신뢰도 게이트 점진적 공개 실증용 텍스트. "
             "고해상 촬영에서만 열리는 부가 정보(상세 설명·서명·메타데이터)를 "
             "fine 스트림에 싣는다.")

BLUR_LEVELS = [x * 0.5 for x in range(0, 21)]  # 0.0 ~ 10.0px


def make_code():
    coarse = COARSE_TEXT.encode("utf-8")
    fine = FINE_TEXT.encode("utf-8")
    enc = DQREncoderV10()
    ren = DQRRendererV10(hex_size=32)
    result = enc.encode(coarse, fine)
    img, info = ren.render(result)
    params = dict(
        radius=result['radius'],
        hex_size=info['hex_size'],
        origin_x=info['origin_x'],
        origin_y=info['origin_y'],
        nsym_c=result['nsym_c'],
        nsym_f=result['nsym_f'],
        coarse_len=result['coarse_len'],
        fine_len=result['fine_len'],
        codeword_len_c=result['codeword_len_c'],
        codeword_len_f=result['codeword_len_f'],
    )
    return coarse, fine, img, params, result


def run_sweep():
    coarse, fine, img, params, result = make_code()
    dec = DQRDecoderV10()
    rows = []
    for lv in BLUR_LEVELS:
        deg = img.filter(ImageFilter.GaussianBlur(lv)) if lv > 0 else img
        try:
            stats = dec.decode_with_stats(deg, **params)
            c_ok = stats['coarse'] == coarse
            f_ok = stats['fine'] == fine
            conf = stats['fine_confidence']
            c_mode = stats['coarse_mode']
            f_mode = stats['fine_mode'] or '—'
        except Exception:
            # coarse 사다리 소진 = 코드 전체 판독 불가
            c_ok = f_ok = False
            conf = 0.0
            c_mode = f_mode = '—'
        rows.append((lv, c_ok, f_ok, conf, c_mode, f_mode))
        print(f"  blur={lv:4.1f}px  coarse={'O' if c_ok else 'X'}({c_mode})  "
              f"fine={'O' if f_ok else 'X'}({f_mode})  conf={conf:.3f}")
    return rows, result


def find_boundary(rows, idx):
    """성공이 유지되는 마지막 블러 레벨 (첫 실패 직전)."""
    last_ok = None
    for row in rows:
        if row[idx]:
            last_ok = row[0]
        else:
            break
    return last_ok


def probe_coarse_collapse(rows):
    """coarse가 스윕 범위(10px) 내 미붕괴 시 실제 붕괴점을 추가 탐침."""
    if not rows[-1][1]:
        return None  # 스윕 안에서 이미 붕괴
    coarse, fine, img, params, _ = make_code()
    dec = DQRDecoderV10()
    lv = BLUR_LEVELS[-1]
    while lv < 20.0:
        lv += 0.5
        deg = img.filter(ImageFilter.GaussianBlur(lv))
        try:
            ok = dec.decode_with_stats(deg, **params)['coarse'] == coarse
        except Exception:
            ok = False
        print(f"  [탐침] blur={lv:4.1f}px  coarse={'O' if ok else 'X'}")
        if not ok:
            return lv
    return None


def main():
    Path("outputs/exp10").mkdir(parents=True, exist_ok=True)
    print("=== exp10 블러 스윕 (0~10px, 0.5px 간격) ===")
    rows, result = run_sweep()

    b_coarse = find_boundary(rows, 1)
    b_fine = find_boundary(rows, 2)
    coarse_collapse = probe_coarse_collapse(rows)
    if coarse_collapse is not None and b_coarse == BLUR_LEVELS[-1]:
        b_coarse = coarse_collapse - 0.5
    print(f"\n생존 경계: coarse ≤ {b_coarse}px / fine ≤ {b_fine}px")
    print(f"점진적 공개 대역(coarse만): {b_fine + 0.5}~{b_coarse}px")

    # ── 차트 ────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    xs = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(xs, [100 if r[1] else 0 for r in rows], 's-',
             color='navy', label='coarse (6bit/셀, ECC 30%)')
    ax1.plot(xs, [100 if r[2] else 0 for r in rows], 'o--',
             color='crimson', label='fine (12bit/셀, ECC 20%)')
    if b_fine is not None and b_coarse is not None:
        ax1.axvspan(b_fine + 0.25, b_coarse + 0.25, alpha=0.12, color='orange',
                    label='점진적 공개 대역 (coarse만 판독)')
    ax1.set_ylabel('디코딩 성공률 (%)')
    ax1.set_ylim(-5, 105)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='center left')
    ax1.set_title('exp10: dual-grain 블러 스윕 — 신뢰도 게이트 점진적 공개')

    ax2.plot(xs, [r[3] for r in rows], 'd-', color='darkorange')
    ax2.set_xlabel('GaussianBlur 반경 (px)')
    ax2.set_ylabel('fine 평균 신뢰도 (0~1)')
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    chart_path = 'outputs/exp10/exp10_blur_sweep_chart.png'
    fig.savefig(chart_path, dpi=110)
    print(f"차트 저장: {chart_path}")

    # ── 마크다운 보고서 ────────────────────────────────────────────
    cap = result['capacity']
    lines = [
        "# exp10 블러 스윕 — dual-grain 점진적 공개 생존 경계",
        "",
        f"- 페이로드: coarse {result['coarse_len']}B / fine {result['fine_len']}B",
        f"- 격자: r={result['radius']} "
        f"(coarse {cap['coarse_cells']}셀 · fine {cap['fine_cells']}셀, "
        f"패리티 인터리브)",
        f"- ECC: coarse 30% (nsym={result['nsym_c']}) / "
        f"fine 20% (nsym={result['nsym_f']}), hex_size=32",
        "- 블러는 결정적(GaussianBlur) → 성공률은 0/100%",
        "",
        "## 생존 경계",
        "",
        f"| 스트림 | 생존 한계 | 붕괴 시점 |",
        f"|--------|----------|----------|",
        f"| coarse | ≤ {b_coarse}px | "
        f"{f'{b_coarse + 0.5}px' if coarse_collapse or b_coarse < BLUR_LEVELS[-1] else '> 20px (미붕괴)'} |",
        f"| fine   | ≤ {b_fine}px | {b_fine + 0.5}px |",
        "",
        f"**점진적 공개 대역: {b_fine + 0.5}~{b_coarse}px** — "
        "이 구간에서 fine은 예외 없이 None으로 강등되고 coarse만 열림.",
        "",
    ]
    # 침묵 오정정(잘못된 데이터인데 사다리가 '성공' 반환) 관측 시 경고 기록
    # — 소거 nsym 전량 단계는 RS 검증 여력이 0이라 발생 가능
    silent = [f"{lv:.1f}" for lv, c_ok, f_ok, _conf, _cm, fm in rows
              if (not f_ok) and fm != '—']
    if silent:
        lines += [
            f"> **주의**: 블러 {', '.join(silent)}px에서 fine 사다리가 잘못된 "
            "데이터를 '성공'으로 반환(침묵 오정정). 소거를 nsym 전량까지 쓰면 "
            "검증 여력이 0이 되기 때문 — 후속 exp에서 fine 신뢰도 임계 게이트 "
            "(예: conf < 0.15 → 강등) 추가 검토.",
            "",
        ]
    lines += [
        "## 상세 결과",
        "",
        "| 블러(px) | coarse | coarse 모드 | fine | fine 모드 | fine 신뢰도 |",
        "|----------|--------|-------------|------|-----------|-------------|",
    ]
    for lv, c_ok, f_ok, conf, c_mode, f_mode in rows:
        lines.append(f"| {lv:.1f} | {'O' if c_ok else 'X'} | {c_mode} | "
                     f"{'O' if f_ok else 'X'} | {f_mode} | {conf:.3f} |")
    lines.append("")
    report_path = 'outputs/exp10/exp10_blur_sweep_report.md'
    Path(report_path).write_text("\n".join(lines), encoding='utf-8')
    print(f"보고서 저장: {report_path}")


if __name__ == "__main__":
    main()
