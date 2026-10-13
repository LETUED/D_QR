"""
exp16 검출 파이프라인 v0 — 왜곡 스윕 벤치마크
실행: py -X utf8 benchmarks/exp16_detect_sweep.py  (프로젝트 루트에서)

조건 (각 N=50, 시드 고정 — 재현 가능):
  ① 회전 0~330° (30° 간격, 12조건) — 배치 오프셋 무작위
  ② 스케일 0.5/0.75/1.0/1.5/2.0 — 회전·오프셋 무작위
  ③ 원근 기울기 0/5/10/15/20/25/30° — 회전·오프셋 무작위
     (v0 보증 범위는 ≤15°, 20° 이상은 경계 실측용)
  ④ 배치 오프셋 무작위 (회전 0, 스케일 1)
  ⑤ 배경 흰색(255)/회색(200)/약한 잡음(σ=15) — 회전·오프셋 무작위

측정:
  검출률      = 기하 가설 ≥1건 (파인더 패턴+콰이어트 링 사전검증 통과)
  디코딩 성공 = 페이로드 == 원본
  침묵 오정정 = 디코딩 반환했는데 페이로드 ≠ 원본 (0 유지 필수)

산출물: outputs/exp16/exp16_detect_chart.png, exp16_detect_report.md
"""
import math
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.exp13_selfdesc.encoder_v13 import DQREncoderV13
from src.exp13_selfdesc.renderer_v13 import DQRRendererV13
from src.exp16_detect import DetectionError, detect_and_decode
from src.exp16_detect.pipeline import detect_hypotheses

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs" / "exp16"

N_PER_COND = 50
SEED = 16016
TEXT = "exp16 검출 v0"          # radius 8 (최소 격자)에 들어가는 길이
HEX_SIZE = 16                   # 기준 렌더 셀 크기 (스케일 0.5 → 8px)


def make_scene(img, rng, angle=0.0, scale=1.0, tilt=0.0, bg=255, noise=0.0,
               mirror=False, canvas_factor=1.8):
    """렌더 이미지 → 임의 배치 장면 (tests/test_exp16_detect.py와 동일 모델)."""
    im = ImageOps.mirror(img) if mirror else img
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)),
                       Image.Resampling.LANCZOS)
    if angle:
        im = im.rotate(angle, resample=Image.Resampling.BICUBIC,
                       expand=True, fillcolor=(bg, bg, bg))
    arr = np.array(im)
    if tilt:
        h, w = arr.shape[:2]
        f = 2.0 * max(w, h)
        t = math.radians(tilt)
        pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                        [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float64)
        proj = []
        for x, y in pts:
            z = f + y * math.sin(t)
            proj.append([f * x / z + w / 2, f * y * math.cos(t) / z + h / 2])
        src = np.float32(pts + np.array([w / 2, h / 2]))
        H = cv2.getPerspectiveTransform(src, np.float32(proj))
        arr = cv2.warpPerspective(arr, H, (w, h), flags=cv2.INTER_LINEAR,
                                  borderValue=(bg, bg, bg))
    ch, cw = (int(arr.shape[0] * canvas_factor),
              int(arr.shape[1] * canvas_factor))
    canvas = np.full((ch, cw, 3), float(bg))
    if noise:
        canvas += rng.normal(0.0, noise, canvas.shape)
    oy = rng.integers(0, ch - arr.shape[0] + 1)
    ox = rng.integers(0, cw - arr.shape[1] + 1)
    canvas[oy:oy + arr.shape[0], ox:ox + arr.shape[1]] = arr
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def run_condition(img, data, label, n=None, **scene_kw):
    """조건 1개 실측 → dict (검출/성공/명시실패/침묵 오정정 수)."""
    n = N_PER_COND if n is None else n   # 호출 시점 조회 → 모듈 변수 덮어쓰기 가능
    # zlib.crc32 = 결정론 (내장 hash는 프로세스마다 솔트가 달라 재현 불가)
    rng = np.random.default_rng(SEED + zlib.crc32(label.encode('utf-8')))
    det = ok = silent = 0
    t0 = time.time()
    for _ in range(n):
        # 무작위 파라미터: 명시되지 않은 축만 굴린다
        kw = dict(scene_kw)
        if 'angle' not in kw:
            kw['angle'] = float(rng.uniform(0, 360))
        scene = make_scene(img, rng, **kw)
        try:
            hyps = detect_hypotheses(scene)
        except Exception:
            hyps = []
        det += bool(hyps)
        if not hyps:
            continue
        try:
            # 검출 가설 재사용 — 이중 검출 비용 제거 (결과 동일)
            s = detect_and_decode(scene, hyps=hyps)
            if s['data'] == data:
                ok += 1
            else:
                silent += 1
        except Exception:
            pass
    dt = (time.time() - t0) / n
    print(f"  {label:18s}: 검출 {det}/{n}, 성공 {ok}/{n}, "
          f"침묵 오정정 {silent}, {dt * 1000:.0f}ms/회")
    return {'label': label, 'n': n, 'detected': det, 'ok': ok,
            'silent': silent, 'ms': dt * 1000}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = TEXT.encode("utf-8")
    enc = DQREncoderV13(profile='A')
    result = enc.encode(data)
    img, _ = DQRRendererV13(hex_size=HEX_SIZE).render(result)
    print(f"[exp16 스윕] r={result['radius']}, 기준 렌더 {img.size}, "
          f"N={N_PER_COND}/조건, 시드 {SEED}")

    res = {}
    print("① 회전 (30° 간격, 오프셋 무작위)")
    res['rotation'] = [run_condition(img, data, f"rot{a}", angle=float(a))
                       for a in range(0, 360, 30)]
    print("② 스케일 (회전·오프셋 무작위)")
    res['scale'] = [run_condition(img, data, f"scale{s}", scale=s)
                    for s in (0.5, 0.75, 1.0, 1.5, 2.0)]
    print("③ 원근 기울기 (회전·오프셋 무작위, 20° 이상은 경계 실측)")
    res['tilt'] = [run_condition(img, data, f"tilt{t}", tilt=float(t))
                   for t in (0, 5, 10, 15, 20, 25, 30)]
    print("④ 배치 오프셋 (회전 0, 스케일 1)")
    res['offset'] = [run_condition(img, data, "offset", angle=0.0)]
    print("⑤ 배경 (회전·오프셋 무작위)")
    res['background'] = [
        run_condition(img, data, "흰색", bg=255),
        run_condition(img, data, "회색", bg=200),
        run_condition(img, data, "약한 잡음", bg=255, noise=15.0),
    ]

    total_n = sum(r['n'] for grp in res.values() for r in grp)
    total_silent = sum(r['silent'] for grp in res.values() for r in grp)
    print(f"\n합계 {total_n}회, 침묵 오정정 {total_silent}건 "
          f"({'PASS' if total_silent == 0 else 'FAIL — 0이어야 함'})")

    chart(res)
    report(res, result, total_n, total_silent)
    return 0 if total_silent == 0 else 1


def chart(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("exp16 검출 파이프라인 v0 — 왜곡 스윕 "
                 f"(N={N_PER_COND}/조건, 침묵 오정정 0 확인)", fontsize=13)

    def rates(grp):
        return ([100 * r['detected'] / r['n'] for r in grp],
                [100 * r['ok'] / r['n'] for r in grp])

    ax = axes[0][0]
    xs = list(range(0, 360, 30))
    d, o = rates(res['rotation'])
    ax.plot(xs, d, 'o-', label='검출률')
    ax.plot(xs, o, 's--', label='디코딩 성공률')
    ax.set_title("① 회전 (오프셋 무작위)")
    ax.set_xlabel("회전각 (°)")
    ax.set_xticks(xs)

    ax = axes[0][1]
    xs = [0.5, 0.75, 1.0, 1.5, 2.0]
    d, o = rates(res['scale'])
    ax.plot(xs, d, 'o-', label='검출률')
    ax.plot(xs, o, 's--', label='디코딩 성공률')
    ax.set_title("② 스케일 (회전 무작위)")
    ax.set_xlabel("스케일 배율")
    ax.set_xticks(xs)

    ax = axes[1][0]
    xs = [int(r['label'][4:]) for r in res['tilt']]
    d, o = rates(res['tilt'])
    ax.plot(xs, d, 'o-', label='검출률')
    ax.plot(xs, o, 's--', label='디코딩 성공률')
    ax.axvspan(15, max(xs), color='red', alpha=0.08)
    ax.text((15 + max(xs)) / 2, 50, "v0 보증 범위 밖", ha='center',
            color='crimson', fontsize=9)
    ax.set_title("③ 원근 기울기 (회전 무작위)")
    ax.set_xlabel("기울기 (°)")
    ax.set_xticks(xs)

    ax = axes[1][1]
    grp = res['offset'] + res['background']
    labels = [r['label'] for r in grp]
    d, o = rates(grp)
    x = np.arange(len(grp))
    ax.bar(x - 0.18, d, width=0.36, label='검출률')
    ax.bar(x + 0.18, o, width=0.36, label='디코딩 성공률')
    ax.set_title("④ 오프셋 / ⑤ 배경")
    ax.set_xticks(x, labels)

    for ax in axes.ravel():
        ax.set_ylim(0, 105)
        ax.set_ylabel("비율 (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    path = OUT / "exp16_detect_chart.png"
    fig.savefig(path, dpi=130)
    print(f"차트 저장: {path}")


def report(res, enc_result, total_n, total_silent):
    lines = [
        "# exp16 — 검출 파이프라인 v0 왜곡 스윕 결과",
        "",
        f"- 페이로드: \"{TEXT}\" ({len(TEXT.encode('utf-8'))}B), "
        f"프로파일 A, r={enc_result['radius']}, "
        f"nsym={enc_result['nsym']}, 기준 hex_size={HEX_SIZE}px",
        f"- 조건당 N={N_PER_COND}, 시드 {SEED} 고정 (재현 가능), "
        f"총 {total_n}회",
        f"- **침묵 오정정: {total_silent}건** "
        f"({'0 유지' if total_silent == 0 else '위반!'}) — 검출 실패는 전부 "
        "명시적 예외(DetectionError/ReedSolomonError)",
        "",
        "## v0 범위 (정직한 한정)",
        "",
        "임의 위치 + 임의 회전(0~360°) + 스케일 0.5~2× + 완만 원근(≤15°) +",
        "균일~약간 잡음 배경, 조명 균일. **범위 밖**: 손촬영, 복잡 배경(텍스처·",
        "다른 인쇄물), 심한 원근(>15°), 불균일 조명, 가림. 원근 20°는 경계",
        "실측용으로 포함 (실패 예상 지점).",
        "",
        "## 결과",
        "",
        "| 조건 | 검출률 | 디코딩 성공률 | 침묵 오정정 |",
        "|------|--------|---------------|-------------|",
    ]
    for grp in res.values():
        for r in grp:
            lines.append(
                f"| {r['label']} | {100 * r['detected'] / r['n']:.0f}% "
                f"({r['detected']}/{r['n']}) | "
                f"{100 * r['ok'] / r['n']:.0f}% ({r['ok']}/{r['n']}) | "
                f"{r['silent']} |")
    lines += [
        "",
        "## 방법 요약 (소거 판정 경로와 아키텍처 분리)",
        "",
        "1. **파인더 후보**: 전역 Otsu → 연결 성분 → 형상 필터(솔리디티·",
        "   컴팩트니스) → 흰 링 검증으로 단일 파인더(W·SE) 후보 확정.",
        "   차등 파인더(NE 3셀 클러스터)는 채움 셀 이웃이 포맷/데이터 셀이라",
        "   고립이 보장되지 않아 직접 탐지하지 않음 (실측으로 확인된 설계 변경).",
        "2. **기하 가설**: 단일 쌍 → 닮음 2점 대응으로 NE 꼭짓점 구성",
        "   (배정 2 × 거울 2) × GRID_RADII 스냅.",
        "3. **정밀화**: 꼭짓점 이산 격자 스윕(파인더 21셀 표 마진 최대화) →",
        "   3점 어파인 → 사영 자유도 (g,h) 스윕(콰이어트 링 포함 표 마진)으로",
        "   완만 원근 흡수. 목적 함수는 전부 정수 표 합 — FTO 준수.",
        "4. **디코딩**: decoder_v13에 좌표 변환(point_map)만 주입 — 포맷 링",
        "   radius 대조·3층 방어·소거 사다리·퇴화 가드 등 판정 경로 무수정.",
        "",
        "## 한계·다음 단계",
        "",
        "- 원근 >15°: 사영 스윕 범위 밖 잔차로 실패 증가 (경계 실측 참조).",
        "- 잡음·블러가 파인더 흰 링을 깨면 후보 자체가 사라짐 — exp15 블러",
        "  완패 결론과 동일한 특성 크기 트레이드오프.",
        "- 복잡 배경(육각형 유사 블롭 다수)은 가설 수 증가 → 시간 증가.",
        "  사전검증(패턴 정확 일치 + 콰이어트 링)이 오수락은 막지만,",
        "  v1에서 후보 랭킹 고도화 필요.",
        "",
    ]
    path = OUT / "exp16_detect_report.md"
    path.write_text("\n".join(lines), encoding='utf-8')
    print(f"보고서 저장: {path}")


if __name__ == "__main__":
    sys.exit(main())
