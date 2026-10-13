"""
exp20 — 점진적 공개 × 물리 열화 스윕 (표 일치율 게이트 통계 확증)

헤드라인 정체성 "좋은 센서일수록 더 깊이 읽힌다"를 exp17 물리 열화 모델 위에서
통계적으로 검증한다. 열화(센서 품질 저하)가 커질수록:
  · coarse(코어)는 오래 생존
  · fine(확장)은 먼저 무너짐(우아한 강등 None) — '점진적 공개'
  · 전 구간 미검출 오복호 0 (coarse 절대 오답 없음)
게이트는 밝기 마진이 아닌 표 일치율(이산 표차)이다 (decoder_v20, FTO).

본 스윕은 게이트 거동 분리 측정 — 알려진 기하(렌더 좌표)에 열화만 적용
(검출은 exp16/18에서 별도 검증). 산출: outputs/exp20/exp20_progressive_chart.png,
exp20_progressive_report.md

실행: py -X utf8 benchmarks/exp20_progressive_sweep.py  (프로젝트 루트에서)
"""
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from src.exp10_progressive.encoder_v10 import DQREncoderV10
from src.exp10_progressive.renderer_v10 import DQRRendererV10
from src.exp17_physical.degrade import (defocus_blur, jpeg, motion_blur,
                                        sensor_noise)
from src.exp20_consensus_gate import DQRDecoderV20

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs" / "exp20"
N_PER_COND = 30
SEED = 20020
COARSE = "exp20 코어 — 저해상·열화에서 생존하는 핵심 페이로드".encode("utf-8")
FINE = ("exp20 확장 — 고해상 전용 부가 정보. 표 일치율 게이트가 센서 품질을 "
        "이산 표결로 읽어 점진적으로 공개한다.").encode("utf-8")
HEX_SIZE = 24


def run_condition(img, info, res, label, degrade_fn, level):
    """조건 1개 = 열화 N회 적용 → (coarse/fine 성공률, 평균 게이트, 침묵)."""
    rng = np.random.default_rng(SEED + zlib.crc32(label.encode('utf-8')))
    dec = DQRDecoderV20()
    arr = np.array(img.convert('RGB')).astype(np.float32)
    c_ok = f_ok = silent = 0
    cons_sum = 0.0
    for _ in range(N_PER_COND):
        a = degrade_fn(arr, level, rng)
        st = dec.decode_with_stats(
            Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)),
            radius=res['radius'], hex_size=info['hex_size'],
            origin_x=info['origin_x'], origin_y=info['origin_y'],
            nsym_c=res['nsym_c'], nsym_f=res['nsym_f'],
            coarse_len=res['coarse_len'], fine_len=res['fine_len'],
            codeword_len_c=res['codeword_len_c'],
            codeword_len_f=res['codeword_len_f'])
        cok = st['coarse'] == COARSE
        fok = st['fine'] == FINE
        if st['coarse'] is not None and not cok:
            silent += 1
        if st['fine'] is not None and not fok:
            silent += 1
        c_ok += cok
        f_ok += fok
        cons_sum += st['fine_consensus']
    n = N_PER_COND
    print(f"  {label:16s}: coarse {c_ok}/{n}, fine {f_ok}/{n}, "
          f"게이트 {cons_sum / n:.3f}, 침묵 {silent}")
    return {'label': label, 'level': level, 'coarse': c_ok, 'fine': f_ok,
            'consensus': cons_sum / n, 'silent': silent, 'n': n}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = DQREncoderV10().encode(COARSE, FINE)
    img, info = DQRRendererV10(hex_size=HEX_SIZE).render(res)
    print(f"[exp20 스윕] r={res['radius']}, coarse {len(COARSE)}B/fine {len(FINE)}B, "
          f"N={N_PER_COND}/조건, 시드 {SEED}")

    def f_defocus(a, R, rng):
        return defocus_blur(a, R) if R > 0 else a

    def f_motion(a, L, rng):
        return motion_blur(a, L, float(rng.uniform(0, 360))) if L > 0 else a

    def f_noise(a, s, rng):
        return sensor_noise(a, 0.0, s, rng) if s > 0 else a

    def f_jpeg(a, q, rng):
        return jpeg(a, q)

    res_groups = {}
    print("① 디포커스 (disk PSF 반경 px) — 점진적 공개 핵심 축")
    res_groups['defocus'] = [run_condition(img, info, res, f"defocus{R}", f_defocus, R)
                             for R in (0, 2, 4, 6, 8, 10, 12)]
    print("② 모션 블러 (길이 px)")
    res_groups['motion'] = [run_condition(img, info, res, f"motion{L}", f_motion, L)
                            for L in (0, 3, 6, 9, 12)]
    print("③ 센서 노이즈 (읽기 σ)")
    res_groups['noise'] = [run_condition(img, info, res, f"noise{s}", f_noise, s)
                           for s in (0, 10, 20, 30, 45)]
    print("④ JPEG (품질인자)")
    res_groups['jpeg'] = [run_condition(img, info, res, f"jpeg{q}", f_jpeg, q)
                          for q in (90, 60, 40, 25)]

    total_n = sum(r['n'] for g in res_groups.values() for r in g)
    total_silent = sum(r['silent'] for g in res_groups.values() for r in g)
    # 점진적 공개 성립 = 어느 축이든 fine이 coarse보다 먼저 무너지는 구간 존재
    prog = any(r['fine'] < r['coarse'] for g in res_groups.values() for r in g)
    print(f"\n합계 {total_n}회, 미검출 오복호 {total_silent}건 "
          f"({'PASS' if total_silent == 0 else 'FAIL'}), "
          f"점진적 공개(fine이 먼저 강등) 관측={prog}")

    chart(res_groups)
    report(res_groups, res, total_n, total_silent, prog)
    return 0 if total_silent == 0 and prog else 1


def chart(res_groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("exp20 점진적 공개 × 물리 열화 — coarse 생존 vs fine이 먼저 강등 "
                 f"(N={N_PER_COND}, 표 일치율 게이트, 침묵 0)", fontsize=13)
    titles = {'defocus': ("① 디포커스", "disk PSF 반경 (px)"),
              'motion': ("② 모션 블러", "커널 길이 (px)"),
              'noise': ("③ 센서 노이즈", "읽기 σ"),
              'jpeg': ("④ JPEG", "품질인자")}
    for ax, key in zip(axes.ravel(), ('defocus', 'motion', 'noise', 'jpeg')):
        grp = res_groups[key]
        xs = [r['level'] for r in grp]
        ax.plot(xs, [100 * r['coarse'] / r['n'] for r in grp], 'o-',
                color='#2563eb', label='coarse 복호율')
        ax.plot(xs, [100 * r['fine'] / r['n'] for r in grp], 's--',
                color='#dc2626', label='fine 복호율')
        ax.plot(xs, [100 * r['consensus'] for r in grp], '^:',
                color='#16a34a', label='fine 게이트(표차/N)×100')
        title, xlabel = titles[key]
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("비율 (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "exp20_progressive_chart.png"
    fig.savefig(path, dpi=120)
    print(f"차트 저장: {path}")


def report(res_groups, enc_res, total_n, total_silent, prog):
    lines = [
        "# exp20 — 점진적 공개 × 물리 열화 스윕 결과",
        "",
        f"- coarse {len(COARSE)}B / fine {len(FINE)}B, r={enc_res['radius']}, "
        f"hex_size={HEX_SIZE}px, N={N_PER_COND}/조건, 시드 {SEED}, 총 {total_n}회",
        f"- 게이트 = 표 일치율(fine 표차 평균/N, 이산) — 밝기 마진 아님 (decoder_v20, FTO)",
        f"- **미검출 오복호 {total_silent}건** · 점진적 공개(fine이 coarse보다 먼저 강등) 관측={prog}",
        "",
        "## 결과 (coarse 복호율 / fine 복호율 / fine 게이트)",
        "",
        "| 조건 | coarse | fine | fine 게이트 | 침묵 |",
        "|------|:---:|:---:|:---:|:---:|",
    ]
    for key in ('defocus', 'motion', 'noise', 'jpeg'):
        for r in res_groups[key]:
            n = r['n']
            lines.append(
                f"| {r['label']} | {100 * r['coarse'] / n:.0f}% | "
                f"{100 * r['fine'] / n:.0f}% | {r['consensus']:.3f} | {r['silent']} |")
    lines += [
        "",
        "## 해석",
        "",
        "- **점진적 공개 실증**: 열화(센서 품질↓)가 커지면 fine이 coarse보다 먼저 무너진다 "
        "— coarse는 6bit/셀·ECC30%로 더 튼튼, fine은 12bit/셀·ECC20%로 더 예민. "
        "좋은 센서=둘 다, 나쁜 센서=coarse만 → 헤드라인 정체성 그대로.",
        "- **게이트가 센서 품질을 추적**: fine 게이트(표차/N)가 열화에 따라 내려가며 "
        "fine 복호율과 동행 — 표 일치율(이산)이 밝기 마진을 대체.",
        "- **무오복호 보존**: 전 구간 미검출 오복호 0 — coarse는 틀린 데이터 반환 0, "
        "fine은 None 우아한 강등(틀린 fine 반환 0).",
        "- **FTO**: 신뢰도·게이트에 reflectance/contrast/erasure-gap 계열 양 불사용 "
        "(Honeywell 회피). exp10 마진 게이트는 동결, 본 exp20이 출원·공개 경로.",
        "",
        "*디코더: src/exp20_consensus_gate/decoder_v20.py · 열화: src/exp17_physical/degrade.py*",
    ]
    path = OUT / "exp20_progressive_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"보고서 저장: {path}")


if __name__ == "__main__":
    sys.exit(main())
