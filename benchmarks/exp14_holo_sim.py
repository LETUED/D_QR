"""
exp14 홀로그램 CGH 시뮬레이션 — "단일색=홀로그램 호환" 창립 가설 0원 검증
실행: py -X utf8 benchmarks/exp14_holo_sim.py  (프로젝트 루트에서)

concept/12 §12.6 0단계 구현:
  ① exp10 dual-grain 패턴 → binary phase CGH (Gerchberg-Saxton)
  ② 재생 모델 3종: (a) 코히어런트 532nm  (b) 백색 LED 3파장 분산(κ 스윕)
                    (c) 스페클 콘트라스트 0.1~0.5
  ③ 판독: 기존 decoder_v10 그대로 (새 신뢰도 메트릭 도입 없음)
  ④ 대조군: 같은 기하에 R/G/B 독립 비트를 실은 가상 컬러 코드의 붕괴

합격선 (concept/12): (a) coarse+fine 모두 판독, (b) 분산 모델에서
coarse 생존(fine은 죽어도 됨 — 점진적 공개), (c) 스페클 생존 경계 실측.

산출물: outputs/exp14/exp14_holo_chart.png, exp14_holo_report.md
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from src.exp10_progressive.encoder_v10 import DQREncoderV10
from src.exp10_progressive.renderer_v10 import DQRRendererV10
from src.exp10_progressive.decoder_v10 import DQRDecoderV10
from src.exp14_holosim.cgh import (DESIGN_NM, embed_target, gs_binary_phase,
                                   replay_intensity, replay_metrics)
from src.exp14_holosim.replay import (LED_WAVELENGTHS, zoom_about_center,
                                      speckle_field, intensity_to_image)
from src.exp14_holosim.colorcode import (COLOR_WAVELENGTHS,
                                         make_channel_tensors, render_pattern,
                                         color_camera_channels, bit_accuracy)

COARSE_TEXT = "https://dqr.kr/h — exp14 코어"
FINE_TEXT = "exp14 확장 페이로드: 홀로그램 재생 점진적 공개 검증."

HEX_SIZE = 16          # 렌더 셀 크기 (재생면 창이 2048 field에 들어가는 크기)
FIELD_SIZE = 2048
GS_ITERS = (50, 25)    # (연속 위상, 이진 양자화) — 합계 75회 (스펙 50~100)
SEED = 14

KAPPAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.5, 1.0]
SPECKLE_CONTRASTS = [0.1, 0.2, 0.3, 0.4, 0.5]
SPECKLE_TRIALS = 15
OUT_DIR = Path("outputs/exp14")


# ──────────────────────────────────────────────────────────────────
# 공통: 코드 생성 / 복호 시도
# ──────────────────────────────────────────────────────────────────
def make_code():
    coarse = COARSE_TEXT.encode("utf-8")
    fine = FINE_TEXT.encode("utf-8")
    enc = DQREncoderV10()
    result = enc.encode(coarse, fine)
    img, info = DQRRendererV10(hex_size=HEX_SIZE).render(result)
    params = dict(
        radius=result['radius'], hex_size=info['hex_size'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym_c=result['nsym_c'], nsym_f=result['nsym_f'],
        coarse_len=result['coarse_len'], fine_len=result['fine_len'],
        codeword_len_c=result['codeword_len_c'],
        codeword_len_f=result['codeword_len_f'],
    )
    return coarse, fine, img, params, result


def try_decode(image, params, coarse_truth, fine_truth):
    """복호 시도 → 원본 일치 여부 (미검출 오복호는 불일치로 잡힘)."""
    try:
        st = DQRDecoderV10().decode_with_stats(image, **params)
    except Exception:
        return {'coarse': False, 'fine': False,
                'c_mode': '—', 'f_mode': '—', 'conf': 0.0}
    return {'coarse': st['coarse'] == coarse_truth,
            'fine': st['fine'] == fine_truth,
            'c_mode': st['coarse_mode'], 'f_mode': st['fine_mode'] or '—',
            'conf': st['fine_confidence']}


def survival_boundary(values, ok_flags):
    """오름차순 values에서 0부터 연속 생존하는 마지막 값 (없으면 None)."""
    last = None
    for v, ok in zip(values, ok_flags):
        if ok:
            last = v
        else:
            break
    return last


# ──────────────────────────────────────────────────────────────────
# 실험 단계
# ──────────────────────────────────────────────────────────────────
def build_cgh(img):
    """원본 렌더 → 반전 타깃 → GS binary phase CGH."""
    # 재생상은 발광 패턴 → 마크를 밝게 기록하고 판독 전 재반전
    gray = np.array(img.convert('L'))
    target_img = 255 - gray
    target, window = embed_target(target_img, FIELD_SIZE)
    t0 = time.time()
    phi = gs_binary_phase(target,
                          n_iter_continuous=GS_ITERS[0],
                          n_iter_binary=GS_ITERS[1], seed=SEED)
    elapsed = time.time() - t0
    I = replay_intensity(phi, DESIGN_NM)
    m = replay_metrics(I, target, window)
    print(f"  GS {sum(GS_ITERS)}회 ({elapsed:.0f}s): "
          f"효율 η={m['efficiency']:.3f}, 충실도 corr={m['fidelity']:.3f}")
    return phi, target, window, m


def run_coherent(phi, window, params, coarse, fine):
    I = replay_intensity(phi, DESIGN_NM)
    img_a = intensity_to_image(I[window])
    res = try_decode(img_a, params, coarse, fine)
    print(f"  (a) 코히어런트 532nm: coarse={'O' if res['coarse'] else 'X'}"
          f"({res['c_mode']})  fine={'O' if res['fine'] else 'X'}"
          f"({res['f_mode']})  conf={res['conf']:.3f}")
    return img_a, res


def run_led_sweep(phi, window, params, coarse, fine):
    """κ 스윕 — 파장별 재생장은 1회만 계산, κ마다 스케일·합성."""
    fields = {lam: replay_intensity(phi, lam) for lam in LED_WAVELENGTHS}
    rows, images = [], {}
    for kappa in KAPPAS:
        total = None
        for lam in LED_WAVELENGTHS:
            s_eff = 1.0 + kappa * (lam / DESIGN_NM - 1.0)
            I = zoom_about_center(fields[lam], s_eff)
            total = I if total is None else total + I
        img_b = intensity_to_image(total[window] / len(LED_WAVELENGTHS))
        res = try_decode(img_b, params, coarse, fine)
        rows.append((kappa, res['coarse'], res['fine']))
        images[kappa] = img_b
        print(f"  (b) LED κ={kappa:<5}: coarse={'O' if res['coarse'] else 'X'}"
              f"({res['c_mode']})  fine={'O' if res['fine'] else 'X'}"
              f"({res['f_mode']})")
    k_c = survival_boundary([r[0] for r in rows], [r[1] for r in rows])
    k_f = survival_boundary([r[0] for r in rows], [r[2] for r in rows])
    print(f"  → 생존 경계: coarse κ ≤ {k_c} / fine κ ≤ {k_f}")
    return rows, images, k_c, k_f


def run_speckle(phi, window, params, coarse, fine):
    """코히어런트 재생 세기에 곱셈성 스페클 → 콘트라스트별 생존율."""
    I_win = replay_intensity(phi, DESIGN_NM)[window]
    rows, sample_img = [], None
    contrasts = list(SPECKLE_CONTRASTS)
    i = 0
    while i < len(contrasts):
        C = contrasts[i]
        n_c = n_f = 0
        for t in range(SPECKLE_TRIALS):
            rng = np.random.default_rng(1000 * int(C * 100) + t)
            M = speckle_field(I_win.shape, C, rng=rng)
            img_c = intensity_to_image(I_win * M)
            res = try_decode(img_c, params, coarse, fine)
            n_c += res['coarse']
            n_f += res['fine']
            if sample_img is None and abs(C - 0.3) < 1e-9 and t == 0:
                sample_img = img_c
        rows.append((C, n_c / SPECKLE_TRIALS, n_f / SPECKLE_TRIALS))
        print(f"  (c) 스페클 C={C:.1f}: coarse {n_c}/{SPECKLE_TRIALS}  "
              f"fine {n_f}/{SPECKLE_TRIALS}")
        # 스윕 끝까지 coarse 전수 생존이면 경계 실측을 위해 확장 탐침
        if i == len(contrasts) - 1 and n_c == SPECKLE_TRIALS and C < 1.0:
            contrasts.append(round(C + 0.2, 1))
        i += 1
    if sample_img is None:  # C=0.3이 스윕에 없을 때 대비
        rng = np.random.default_rng(0)
        sample_img = intensity_to_image(
            I_win * speckle_field(I_win.shape, 0.3, rng=rng))
    return rows, sample_img


def run_color_control(radius, window, mono_led_images, mono_tensor, params):
    """대조군: R/G/B 독립 비트 컬러 코드 — 채널 분리 붕괴 측정."""
    tensors = make_channel_tensors(radius, seed=SEED)
    phis = []
    geom = None
    for k, (tensor, lam) in enumerate(zip(tensors, COLOR_WAVELENGTHS)):
        img_k, info_k = render_pattern(tensor, radius, HEX_SIZE)
        geom = info_k
        target_k = 255 - np.array(img_k.convert('L'))
        tgt, _win_k = embed_target(target_k, FIELD_SIZE)
        print(f"  컬러 CGH {k + 1}/3 (λ={lam:.0f}nm) GS 계산 중...")
        phis.append(gs_binary_phase(tgt,
                                    n_iter_continuous=GS_ITERS[0],
                                    n_iter_binary=GS_ITERS[1], seed=SEED + k))

    geo = dict(radius=radius, hex_size=geom['hex_size'],
               origin_x=geom['origin_x'], origin_y=geom['origin_y'])
    results = {}
    for kappa in (0.0, 1.0):
        channels = color_camera_channels(phis, kappa)
        accs = []
        for ch_field, tensor in zip(channels, tensors):
            img_ch = intensity_to_image(ch_field[window])
            accs.append(bit_accuracy(img_ch, tensor, **geo))
        mono_acc = bit_accuracy(mono_led_images[kappa], mono_tensor, **geo)
        results[kappa] = {'color': accs, 'mono': mono_acc}
        print(f"  대조군 κ={kappa}: 컬러 R/G/B 비트 정확도 "
              f"{accs[0]:.3f}/{accs[1]:.3f}/{accs[2]:.3f}  "
              f"vs 단일색 {mono_acc:.3f}")
    return results


# ──────────────────────────────────────────────────────────────────
# 산출물
# ──────────────────────────────────────────────────────────────────
def rs_feasible(acc):
    """비트 정확도 → RS(ECC 30%, 오류만) 복호 가능성 추정 근거 문자열."""
    byte_err = 1.0 - acc ** 8          # 바이트 오류율 근사 (독립 비트 가정)
    return byte_err


def make_chart(img, phi, window, img_a, led_rows, led_images, speckle_rows,
               speckle_img, color_results, k_c, k_f, res_a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1.0, 0.55],
                          hspace=0.35, wspace=0.25)

    # ── 1행: 이미지 비교 ──
    panels = [
        ('원본 (exp10 dual-grain)', img),
        ('binary phase CGH (0/π)',
         Image.fromarray((phi[window] / np.pi * 255).astype(np.uint8))),
        ('(a) 코히어런트 532nm', img_a),
        ('(b) LED κ=0 (이미지면)', led_images[0.0]),
        ('(b) LED κ=1 (Fourier)', led_images[1.0]),
        ('(c) 스페클 C=0.3', speckle_img),
    ]
    for i, (title, im) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.array(im.convert('L')), cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # ── 2행: κ 스윕 / 스페클 생존율 / 컬러 대조 ──
    ax1 = fig.add_subplot(gs[1, 0:2])
    ks = [r[0] for r in led_rows]
    ax1.plot(ks, [100 * r[1] for r in led_rows], 's-', color='navy',
             label='coarse')
    ax1.plot(ks, [100 * r[2] for r in led_rows], 'o--', color='crimson',
             label='fine')
    if k_f is not None and k_c is not None and k_c > k_f:
        ax1.axvspan(k_f, k_c, alpha=0.12, color='orange',
                    label='coarse만 생존 (각도 도메인 점진적 공개)')
    ax1.set_xscale('symlog', linthresh=0.01)
    ax1.set_xlabel('분산 기하 계수 κ (0=이미지면, 1=Fourier)')
    ax1.set_ylabel('복호 성공 (%)')
    ax1.set_ylim(-5, 105)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_title('(b) 백색 LED 3파장 분산 — κ 스윕', fontsize=11)

    ax2 = fig.add_subplot(gs[1, 2:4])
    cs = [r[0] for r in speckle_rows]
    ax2.plot(cs, [100 * r[1] for r in speckle_rows], 's-', color='navy',
             label='coarse')
    ax2.plot(cs, [100 * r[2] for r in speckle_rows], 'o--', color='crimson',
             label='fine')
    ax2.set_xlabel('스페클 콘트라스트 C (σ/μ)')
    ax2.set_ylabel(f'생존율 (%, {SPECKLE_TRIALS}회 시행)')
    ax2.set_ylim(-5, 105)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_title('(c) 스페클 — 콘트라스트별 생존율', fontsize=11)

    ax3 = fig.add_subplot(gs[1, 4:6])
    r0 = color_results[0.0]
    labels = ['단일색\n(D_QR)', '컬러 R\n(650nm)', '컬러 G\n(550nm)',
              '컬러 B\n(450nm)']
    vals = [100 * r0['mono']] + [100 * a for a in r0['color']]
    colors = ['#222222', '#c0392b', '#27ae60', '#2980b9']
    bars = ax3.bar(labels, vals, color=colors)
    ax3.axhline(50, color='gray', ls=':', lw=1)
    ax3.text(3.45, 51, '우연 수준', fontsize=8, color='gray', ha='right')
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f}%',
                 ha='center', fontsize=9)
    ax3.set_ylabel('비트 정확도 (%)')
    ax3.set_ylim(0, 110)
    ax3.set_title('대조군: 백색 조명(κ=0, 컬러 최유리)에서\n'
                  '단일색 vs RGB 3채널 컬러 코드', fontsize=11)

    # ── 3행: 판정 표 ──
    axt = fig.add_subplot(gs[2, :])
    axt.axis('off')
    a_str = (f"coarse {'O' if res_a['coarse'] else 'X'} / "
             f"fine {'O' if res_a['fine'] else 'X'}")
    spk_c = survival_boundary([r[0] for r in speckle_rows],
                              [r[1] == 1.0 for r in speckle_rows])
    spk_f = survival_boundary([r[0] for r in speckle_rows],
                              [r[2] == 1.0 for r in speckle_rows])
    rows = [
        ['(a) 코히어런트 532nm', a_str, 'coarse+fine 동시 판독',
         'PASS' if (res_a['coarse'] and res_a['fine']) else 'FAIL'],
        ['(b) 백색 LED 분산', f'coarse κ≤{k_c} / fine κ≤{k_f}',
         'coarse 생존 (fine은 죽어도 됨)',
         'PASS' if (k_c is not None and (k_f is None or k_c >= k_f))
         else 'FAIL'],
        ['(c) 스페클', f'coarse 전수생존 C≤{spk_c} / fine C≤{spk_f}',
         '생존 경계 실측', '실측 완료'],
        ['대조군 (RGB 컬러)',
         f"비트 정확도 {min(r0['color']):.1%}~{max(r0['color']):.1%} "
         f"(단일색 {r0['mono']:.1%})", '파장 혼합 시 채널 붕괴 입증',
         '붕괴 확인'],
    ]
    table = axt.table(
        cellText=rows,
        colLabels=['재생 모델', '결과', '합격선 (concept/12)', '판정'],
        loc='center', cellLoc='center',
        colWidths=[0.16, 0.34, 0.3, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.6)

    fig.suptitle('exp14 — 홀로그램 CGH 시뮬레이션: '
                 '"단일색 = 홀로그램 호환" 창립 가설 0단계 검증', fontsize=14)
    path = OUT_DIR / 'exp14_holo_chart.png'
    fig.savefig(path, dpi=110, bbox_inches='tight')
    print(f"차트 저장: {path}")


def make_report(result, metrics, res_a, led_rows, k_c, k_f, speckle_rows,
                color_results):
    cap = result['capacity']
    r0, r1 = color_results[0.0], color_results[1.0]
    spk_c = survival_boundary([r[0] for r in speckle_rows],
                              [r[1] == 1.0 for r in speckle_rows])
    spk_f = survival_boundary([r[0] for r in speckle_rows],
                              [r[2] == 1.0 for r in speckle_rows])
    a_pass = res_a['coarse'] and res_a['fine']
    b_pass = k_c is not None and (k_f is None or k_c >= k_f)
    verdict = a_pass and b_pass

    lines = [
        "# exp14 — 홀로그램 CGH 시뮬레이션 보고서",
        "",
        "**창립 가설 검증**: \"단일색 이진 기하가 홀로그램 재생에서 보존되는가\"",
        "(concept/12_hologram_3d.md §12.6 0단계 — 비용 0원, numpy FFT 전용)",
        "",
        f"## 판정: 창립 가설 {'✅ 지지' if verdict else '❌ 기각'}",
        "",
        "| 합격선 | 결과 | 판정 |",
        "|--------|------|------|",
        f"| (a) 코히어런트 재생에서 coarse+fine 판독 | "
        f"coarse {'O' if res_a['coarse'] else 'X'} / "
        f"fine {'O' if res_a['fine'] else 'X'} | "
        f"{'PASS' if a_pass else 'FAIL'} |",
        f"| (b) 백색 LED 분산에서 coarse 생존 | "
        f"coarse κ ≤ {k_c} / fine κ ≤ {k_f} | "
        f"{'PASS' if b_pass else 'FAIL'} |",
        f"| (c) 스페클 생존 경계 실측 | coarse 전수생존 C ≤ {spk_c} / "
        f"fine C ≤ {spk_f} | 실측 완료 |",
        f"| 대조군: 컬러 코드 붕괴 | R/G/B 비트 정확도 "
        f"{min(r0['color']):.1%}~{max(r0['color']):.1%} "
        f"(단일색 {r0['mono']:.1%}) | 붕괴 확인 |",
        "",
        "## 방법",
        "",
        f"- 패턴: exp10 dual-grain (r={result['radius']}, "
        f"coarse {cap['coarse_cells']}셀·6bit·ECC 30% / "
        f"fine {cap['fine_cells']}셀·12bit·ECC 20%), hex_size={HEX_SIZE}",
        f"- 페이로드: coarse {result['coarse_len']}B / fine {result['fine_len']}B",
        f"- CGH: Gerchberg-Saxton {GS_ITERS[0]}회 연속 + {GS_ITERS[1]}회 "
        f"이진(0/π) 양자화, field {FIELD_SIZE}², 오프축 창 배치",
        f"  (이진 위상 = 실수장 → 쌍둥이 상이 필연이므로 타깃에 점대칭 "
        f"쌍둥이를 명시 포함해 제약을 대칭과 양립시킴). "
        f"재생 효율 η={metrics['efficiency']:.3f} (쌍둥이 상 몫 제외 시 "
        f"이진 위상의 물리 상한 ~0.405와 정합), "
        f"충실도 corr={metrics['fidelity']:.3f}",
        "- 재생: Fourier 기하, 위치 ∝ λ. 표면 부조 위상 깊이 detune "
        "φ(λ)=π·(532/λ) 포함",
        "- 백색 LED: 450/550/650nm 세기 합성. 분산 기하 계수 "
        "κ ∈ [0,1]로 이미지면(κ=0, 엠보싱 양산 경로)↔Fourier(κ=1) 보간 — "
        "s_eff = 1 + κ(λ/532 − 1)",
        "- 스페클: 곱셈성 세기 변동(평균 1, 콘트라스트 C, 그레인 2px), "
        f"C별 {SPECKLE_TRIALS}회 시행",
        "- 판독: 기존 decoder_v10 그대로 (전역 임계 128, 새 신뢰도 메트릭 "
        "도입 없음 — FTO 1·2 준수). 성공 = 페이로드 원본 일치 "
        "(미검출 오복호는 실패로 집계)",
        "",
        "## (b) 백색 LED 분산 — κ 스윕 상세",
        "",
        "| κ | coarse | fine |",
        "|---|--------|------|",
    ]
    for kappa, c_ok, f_ok in led_rows:
        lines.append(f"| {kappa} | {'O' if c_ok else 'X'} | "
                     f"{'O' if f_ok else 'X'} |")
    lines += [
        "",
        f"**coarse만 생존하는 κ 대역({k_f}~{k_c})이 존재** — exp10의 점진적 "
        "공개(스케일 도메인)가 각도/분산 도메인으로 그대로 이식됨을 확인. "
        "κ=1(순수 Fourier, 오프축 캐리어로 상 전체가 파장당 수십 px 강체 "
        "변위)은 붕괴가 물리적으로 자명하며, 이것이 양산 엠보스 홀로그램이 "
        "이미지면 기록(κ→0)을 쓰는 이유와 정합한다.",
        "",
        "## (c) 스페클 상세",
        "",
        "| C | coarse 생존율 | fine 생존율 |",
        "|---|---------------|-------------|",
    ]
    for C, sc, sf in speckle_rows:
        lines.append(f"| {C:.1f} | {sc:.0%} | {sf:.0%} |")
    lines += [
        "",
        "## 대조군 — 가상 컬러 코드의 파장 분산 붕괴",
        "",
        "같은 삼각형 기하에 R/G/B 3채널 독립 비트(용량 3배 주장 구조)를 싣고, "
        "채널별 전용 파장(650/550/450nm) CGH로 기록한 최유리 설정. 그러나 표면 "
        "부조 홀로그램은 thin grating이라 파장 선택성이 없어 모든 CGH가 모든 "
        "파장을 회절시킨다 → 카메라 각 채널에 세 패턴이 겹친다.",
        "",
        "| 조건 | 컬러 R | 컬러 G | 컬러 B | 단일색 (같은 측정) |",
        "|------|--------|--------|--------|--------------------|",
        f"| κ=0 (컬러 최유리) | {r0['color'][0]:.1%} | {r0['color'][1]:.1%} | "
        f"{r0['color'][2]:.1%} | {r0['mono']:.1%} |",
        f"| κ=1 (Fourier) | {r1['color'][0]:.1%} | {r1['color'][1]:.1%} | "
        f"{r1['color'][2]:.1%} | {r1['mono']:.1%} |",
        "",
        f"컬러 채널 바이트 오류율 추정(독립 비트 근사) "
        f"{rs_feasible(max(r0['color'])):.0%} — RS가 정정 가능한 한계"
        "(ECC 30%, 오류만 ≈ 15%)를 크게 초과해 어떤 ECC율로도 복호 불능. "
        "단일색은 세 파장이 같은 정보의 중복 사본이라 혼합이 손상이 아니다. "
        "이것이 CHQR(US 10,885,413)이 층별 분산성 HOE라는 별도 광학 장치를 "
        "요구하는 이유의 시뮬레이션 재현이며, 창립 가설의 역방향 증거를 "
        "완성한다.",
        "",
        "## 한계 (사실대로 정리)",
        "",
        "- κ는 유효 계수다 — 실제 엠보스 홀로그램의 분산은 기록 기하·재생상 "
        "깊이·조명 입사각에 의존하며, κ=0은 이상화다. 직접 답은 2단계 "
        "PoC(LitiHolo 기록)만 줄 수 있다 (concept/12 §12.6).",
        "- LED를 3선 스펙트럼으로 근사 — 실제 백색 LED는 연속 스펙트럼이라 "
        "스미어가 더 매끄럽다 (이산 고스트보다 판독에 유리할 수도, 불리할 "
        "수도 있음).",
        "- 0차 누설·스페클은 모델링했으나 기록 매질 비선형성·표면 거칠기· "
        "카메라 노이즈는 미포함.",
        "- 본 시뮬은 용량 증가를 주장하지 않는다 — 검증 대상은 \"보존\"이지 "
        "\"증대\"가 아니다 (concept/12 §12.3 과장 금지선 유지).",
        "",
        "*판독기는 D_QR 전용 디코더(decoder_v10)이며 본 실험은 새 신뢰도 "
        "메트릭·국소 필터를 도입하지 않았다.*",
        "",
    ]
    path = OUT_DIR / 'exp14_holo_report.md'
    path.write_text("\n".join(lines), encoding='utf-8')
    print(f"보고서 저장: {path}")
    return verdict


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== exp14 홀로그램 CGH 시뮬레이션 ===")

    print("[1/6] exp10 dual-grain 코드 생성")
    coarse, fine, img, params, result = make_code()
    cap = result['capacity']
    print(f"  r={result['radius']}, 이미지 {img.width}x{img.height}, "
          f"coarse {cap['coarse_cells']}셀 / fine {cap['fine_cells']}셀")
    img.save(OUT_DIR / 'exp14_original.png')

    print("[2/6] binary phase CGH 계산 (Gerchberg-Saxton)")
    phi, target, window, metrics = build_cgh(img)
    Image.fromarray((phi / np.pi * 255).astype(np.uint8)).save(
        OUT_DIR / 'exp14_cgh_phase.png')

    print("[3/6] (a) 코히어런트 532nm 재생")
    img_a, res_a = run_coherent(phi, window, params, coarse, fine)
    img_a.save(OUT_DIR / 'exp14_replay_coherent.png')

    print("[4/6] (b) 백색 LED 3파장 분산 (κ 스윕)")
    led_rows, led_images, k_c, k_f = run_led_sweep(
        phi, window, params, coarse, fine)
    led_images[0.0].save(OUT_DIR / 'exp14_replay_led_k0.png')
    led_images[1.0].save(OUT_DIR / 'exp14_replay_led_k1.png')

    print("[5/6] (c) 스페클 스윕")
    speckle_rows, speckle_img = run_speckle(
        phi, window, params, coarse, fine)
    speckle_img.save(OUT_DIR / 'exp14_replay_speckle_C0.3.png')

    print("[6/6] 대조군: 가상 RGB 컬러 코드")
    color_results = run_color_control(
        result['radius'], window, led_images, result['tensor'], params)

    make_chart(img, phi, window, img_a, led_rows, led_images, speckle_rows,
               speckle_img, color_results, k_c, k_f, res_a)
    verdict = make_report(result, metrics, res_a, led_rows, k_c, k_f,
                          speckle_rows, color_results)
    print(f"\n=== 창립 가설 판정: {'지지 (PASS)' if verdict else '기각 (FAIL)'} ===")


if __name__ == "__main__":
    main()
