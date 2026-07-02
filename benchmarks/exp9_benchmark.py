"""
exp9 인식률 벤치마크 — hard vs soft-decision 정량 비교
실행: py -X utf8 exp9_benchmark.py

열화 모델:
  1) 가우시안 노이즈 (sigma 0~140) — 인쇄 입자·센서 노이즈 모사
  2) 가우시안 블러 (radius 0~8px) — 초점 흐림·저해상 촬영 모사
  3) 국소 오염 패치 (0~14개) — 얼룩·스크래치 모사

각 조건 × N회 시행 → 성공률(%) 비교.
산출물: outputs/exp9/exp9_benchmark_chart.png, outputs/exp9/exp9_benchmark_report.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageFilter

from src.exp9_soft.encoder_v9 import DQREncoderV9
from src.exp9_soft.renderer_v9 import DQRRendererV9
from src.exp9_soft.decoder_v9 import DQRDecoderV9

TEXT = "dqr 시험테스트입니다 — exp9 soft-decision 벤치마크용 페이로드"
TRIALS = 20
SEED = 42


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


def add_noise(img, sigma, rng):
    arr = np.array(img, dtype=np.float64)
    arr += rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_blur(img, radius):
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius))


def add_patches(img, n_patches, rng, patch_size=40):
    """국소 오염: 중간 회색(임계값 근처) 사각 패치 — 얼룩 모사."""
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    for _ in range(n_patches):
        x = rng.integers(0, max(1, w - patch_size))
        y = rng.integers(0, max(1, h - patch_size))
        gray = rng.integers(100, 156)  # 임계값 128 근처 = 가장 악성
        arr[y:y+patch_size, x:x+patch_size] = gray
    return Image.fromarray(arr)


def try_decode(dec, img, params, data, soft):
    try:
        if soft:
            out = dec.decode_with_stats(img, **params)['data']
        else:
            out = dec.decode_hard_only(img, **params)
        return out == data
    except Exception:
        return False


def run_sweep(profile, degrade_name, levels, degrade_fn):
    """각 레벨에서 hard/soft 성공률(%) 측정."""
    data, img, params = make_code(profile)
    dec = DQRDecoderV9()
    rows = []
    for lv in levels:
        hard_ok = soft_ok = 0
        for t in range(TRIALS):
            rng = np.random.default_rng(SEED + t)
            deg = degrade_fn(img, lv, rng)
            hard_ok += try_decode(dec, deg, params, data, soft=False)
            soft_ok += try_decode(dec, deg, params, data, soft=True)
        rows.append((lv, 100 * hard_ok / TRIALS, 100 * soft_ok / TRIALS))
        print(f"  [{profile}/{degrade_name}] lv={lv}: "
              f"hard {rows[-1][1]:.0f}%  soft {rows[-1][2]:.0f}%")
    return rows


def main():
    # RS 복호는 절벽(cliff) 특성 → 천이 구간을 세밀하게 스캔해야 차이가 보임
    sweeps = {
        'noise':   ([100, 105, 110, 115, 120, 125, 130, 135],
                    lambda img, lv, rng: add_noise(img, lv, rng)),
        'blur':    ([6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
                    lambda img, lv, rng: add_blur(img, lv)),
        'patches': ([20, 30, 40, 50, 60, 70, 80, 90],
                    lambda img, lv, rng: add_patches(img, lv, rng, patch_size=60)),
    }

    results = {}
    for profile in ('A', 'B'):
        for name, (levels, fn) in sweeps.items():
            print(f"\n=== 프로파일 {profile} / {name} ===")
            results[(profile, name)] = run_sweep(profile, name, levels, fn)

    # ── 차트 ────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    titles = {'noise': '가우시안 노이즈 (sigma)',
              'blur': '가우시안 블러 (radius px)',
              'patches': '국소 오염 패치 (개수)'}
    for i, profile in enumerate(('A', 'B')):
        for j, name in enumerate(('noise', 'blur', 'patches')):
            ax = axes[i][j]
            rows = results[(profile, name)]
            xs = [r[0] for r in rows]
            ax.plot(xs, [r[1] for r in rows], 'o--', color='gray', label='hard (기존 v6 방식)')
            ax.plot(xs, [r[2] for r in rows], 's-', color='crimson', label='soft (exp9 소거)')
            ax.set_title(f"프로파일 {profile} — {titles[name]}")
            ax.set_ylabel('디코딩 성공률 (%)')
            ax.set_ylim(-5, 105)
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle('exp9: hard vs soft-decision 인식률 벤치마크', fontsize=14)
    fig.tight_layout()
    Path('outputs/exp9').mkdir(parents=True, exist_ok=True)
    chart_path = 'outputs/exp9/exp9_benchmark_chart.png'
    fig.savefig(chart_path, dpi=110)
    print(f"\n차트 저장: {chart_path}")

    # ── 마크다운 보고서 ────────────────────────────────────────────
    lines = [
        "# exp9 벤치마크 — hard vs soft-decision 인식률",
        "",
        f"- 페이로드: {len(TEXT.encode('utf-8'))} bytes, 시행 {TRIALS}회/조건, seed={SEED}",
        "- hard = 기존 v6 방식(임계값 이진화만), soft = exp9 소거 사다리",
        "",
    ]
    for profile in ('A', 'B'):
        lines.append(f"## 프로파일 {profile}")
        lines.append("")
        for name in ('noise', 'blur', 'patches'):
            rows = results[(profile, name)]
            lines.append(f"### {titles[name]}")
            lines.append("")
            lines.append("| 레벨 | hard 성공률 | soft 성공률 | 개선 |")
            lines.append("|------|-----------|-----------|------|")
            for lv, h, s in rows:
                diff = s - h
                mark = f"**+{diff:.0f}%p**" if diff > 0 else (f"{diff:.0f}%p" if diff < 0 else "—")
                lines.append(f"| {lv} | {h:.0f}% | {s:.0f}% | {mark} |")
            lines.append("")
    report_path = 'outputs/exp9/exp9_benchmark_report.md'
    Path(report_path).write_text("\n".join(lines), encoding='utf-8')
    print(f"보고서 저장: {report_path}")


if __name__ == "__main__":
    main()
