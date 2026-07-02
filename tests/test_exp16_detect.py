"""
exp16 테스트 — 검출 파이프라인 v0 (임의 위치·회전·완만 원근 → 자기기술 디코딩)
실행: py -X utf8 tests/test_exp16_detect.py  (프로젝트 루트에서)

① 파인더 후보 탐지: 합성 장면에서 단일 파인더(W·SE) 2개를 흰 링 검증으로
   정확 위치(±2px) 검출
② 어파인 정확도: 기지(旣知) 변환(회전·스케일·평행이동) 장면에서 추정
   사상이 세 파인더 중심을 ±1.5px로 사상
③ 검출 왕복: 프로파일 A/B × {회전·스케일·오프셋·배경·원근} 표본 케이스
   에서 검출 디코딩 == 원본 (메타데이터 없는 임의 배치 PNG)
④ 거울상: 거울 입력 검출 디코딩 + detection.mirrored 플래그
⑤ 명시적 실패: 코드 없는 이미지(빈/잡음/블롭), 파인더 가림 → 예외
   (침묵 오정정 경로 없음)
⑥ 침묵 오정정 0: ③·④ 전 케이스에서 성공 시 페이로드 일치 단언 (집계 재확인)
⑦ 아키텍처 분리 (FTO): src/exp16_detect/*가 소거·신뢰도 경로
   (softdecode·spatial·consensus 등)를 임포트하지 않음을 소스 검사로 단언
⑧ 회귀: test_exp13_selfdesc.py 전체 PASS (decoder_v13 훅 추가가 기존
   규약 경로에 영향 없음 — exp13이 내부에서 exp9/exp11 회귀까지 수행)
"""
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.core.hex_grid import axial_to_pixel, finder_centers
from src.exp13_selfdesc.encoder_v13 import DQREncoderV13
from src.exp13_selfdesc.renderer_v13 import DQRRendererV13
from src.exp16_detect import detect_and_decode
from src.exp16_detect.finder_detect import (classify_singles,
                                            find_blob_candidates,
                                            otsu_threshold, to_gray)
from src.exp16_detect.pipeline import detect_hypotheses

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs" / "exp16"
TEXT = "dqr exp16 검출 파이프라인 v0 — 임의 배치 검증"


def make_code(profile, hex_size=16):
    data = TEXT.encode("utf-8")
    enc = DQREncoderV13(profile=profile)
    result = enc.encode(data)
    img, info = DQRRendererV13(hex_size=hex_size).render(result)
    return data, result, img, info


def make_scene(img, angle=0.0, scale=1.0, tilt=0.0, offset=(0.3, 0.4),
               bg=255, noise=0.0, mirror=False, canvas_factor=1.8, seed=0):
    """렌더 이미지 → 임의 배치 장면 합성 (벤치마크와 동일 모델)."""
    rng = np.random.default_rng(seed)
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
        # 평면을 수평축 둘레로 t 회전한 핀홀 투영 (초점 f)
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
    canvas = np.full((ch, cw, 3), bg, dtype=np.float64)
    if noise:
        canvas += rng.normal(0, noise, canvas.shape)
    oy = int((ch - arr.shape[0]) * offset[1])
    ox = int((cw - arr.shape[1]) * offset[0])
    canvas[oy:oy + arr.shape[0], ox:ox + arr.shape[1]] = arr
    return (Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)),
            (ox, oy))


def _strip_and_reload(img, path):
    """PNG 저장(메타데이터 미기록) 후 재로드 — 메타 부재 단언."""
    img.save(path)
    loaded = Image.open(path)
    dqr_keys = [k for k in loaded.info if str(k).startswith("dqr_")]
    assert not dqr_keys, f"메타데이터 잔존: {dqr_keys}"
    return loaded


def test_finder_candidates():
    """① 단일 파인더(W·SE) 2개 정확 위치 검출 (회전 장면, ±2px)."""
    _, result, img, info = make_code('A')
    angle = 30.0
    scene, (ox, oy) = make_scene(img, angle=angle)

    # 진리값: 렌더 좌표 → rotate(expand) 좌표 → 캔버스 좌표
    rad = math.radians(angle)   # PIL rotate는 반시계 (이미지 y축 아래 기준)
    w0, h0 = img.width, img.height
    cx0, cy0 = w0 / 2, h0 / 2
    rw = abs(w0 * math.cos(rad)) + abs(h0 * math.sin(rad))
    rh = abs(w0 * math.sin(rad)) + abs(h0 * math.cos(rad))
    truth = []
    for cq, cr in finder_centers(result['radius'])[1:]:   # W, SE 단일만
        x, y = axial_to_pixel(cq, cr, info['hex_size'],
                              info['origin_x'], info['origin_y'])
        dx, dy = x - cx0, y - cy0
        tx = dx * math.cos(rad) + dy * math.sin(rad) + rw / 2 + ox
        ty = -dx * math.sin(rad) + dy * math.cos(rad) + rh / 2 + oy
        truth.append((tx, ty))

    gray = to_gray(scene)
    thr = otsu_threshold(gray)
    cands, _ = find_blob_candidates(gray, thr)
    singles = classify_singles(gray, thr, cands)
    ok = True
    for tx, ty in truth:
        d = min(math.hypot(s['cx'] - tx, s['cy'] - ty) for s in singles)
        ok &= d <= 2.0
    print(f"[테스트 ①] 단일 파인더 검출: {'PASS' if ok else 'FAIL'} "
          f"(후보 {len(cands)}개 중 단일 {len(singles)}개, 위치 오차 ≤2px)")
    return ok


def test_affine_accuracy():
    """② 기지 변환(회전 50°+스케일 1.25+오프셋)에서 사상 오차 ≤1.5px."""
    _, result, img, info = make_code('A')
    angle, scale = 50.0, 1.25
    scene, (ox, oy) = make_scene(img, angle=angle, scale=scale,
                                 offset=(0.7, 0.2))
    hyps = detect_hypotheses(scene)
    if not hyps:
        print("[테스트 ②] 어파인 정확도: FAIL (가설 없음)")
        return False
    hyp = hyps[0]

    # 진리값 합성 변환으로 세 파인더 중심 위치 계산
    rad = math.radians(angle)
    w0, h0 = img.width * scale, img.height * scale
    rw = abs(w0 * math.cos(rad)) + abs(h0 * math.sin(rad))
    rh = abs(w0 * math.sin(rad)) + abs(h0 * math.cos(rad))
    ok = True
    errs = []
    for cq, cr in finder_centers(result['radius']):
        x, y = axial_to_pixel(cq, cr, info['hex_size'],
                              info['origin_x'], info['origin_y'])
        dx, dy = x * scale - w0 / 2, y * scale - h0 / 2
        tx = dx * math.cos(rad) + dy * math.sin(rad) + rw / 2 + ox
        ty = -dx * math.sin(rad) + dy * math.cos(rad) + rh / 2 + oy
        # 추정 사상: 표준 공간(hex_size=1) 파인더 중심 → 이미지
        ex, ey = hyp['map_fn'](*axial_to_pixel(cq, cr, 1.0))
        errs.append(math.hypot(float(ex) - tx, float(ey) - ty))
    ok = max(errs) <= 1.5 and hyp['radius'] == result['radius']
    print(f"[테스트 ②] 어파인 정확도: {'PASS' if ok else 'FAIL'} "
          f"(최대 오차 {max(errs):.2f}px, r={hyp['radius']}, "
          f"hs={hyp['hex_size_img']:.2f})")
    return ok


# (이름, 장면 파라미터) — 회전/스케일/원근/배경/오프셋 표본
ROUNDTRIP_CASES = [
    ('기본', dict()),
    ('회전45', dict(angle=45)),
    ('회전137+축소0.5', dict(angle=137, scale=0.5)),
    ('회전200+확대2.0', dict(angle=200, scale=2.0, offset=(0.8, 0.1))),
    ('회전77+원근10', dict(angle=77, tilt=10)),
    ('회전290+원근15', dict(angle=290, tilt=15, scale=0.75)),
    ('회색배경+잡음', dict(angle=10, bg=200, noise=10)),
    ('복합', dict(angle=350, scale=1.5, tilt=5, bg=230, noise=15,
                  offset=(0.1, 0.9))),
]


def test_detect_roundtrip():
    """③+⑥ 임의 배치 왕복 — 성공 시 페이로드 일치(침묵 오정정 0) 단언."""
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for profile in ('A', 'B'):
        data, _, img, _ = make_code(profile)
        for name, kw in ROUNDTRIP_CASES:
            scene, _ = make_scene(img, **kw)
            path = OUT / f"exp16_scene_{profile}_{name}.png"
            loaded = _strip_and_reload(scene, path)
            try:
                s = detect_and_decode(loaded)
                good = s['data'] == data
                silent = not good     # 디코딩됐는데 불일치 = 침묵 오정정
            except Exception as e:
                good, silent = False, False
                s = {'mode': f"EXC:{type(e).__name__}"}
            assert not silent, f"침묵 오정정 발생: {profile} {name}"
            ok &= good
            print(f"[테스트 ③] 왕복 {profile} {name:14s}: "
                  f"{'PASS' if good else 'FAIL'} (mode={s.get('mode')})")
    return ok


def test_mirror():
    """④ 거울상 입력 — 검출 디코딩 + mirrored 플래그."""
    ok = True
    for profile in ('A', 'B'):
        data, _, img, _ = make_code(profile)
        scene, _ = make_scene(img, angle=33, mirror=True)
        try:
            s = detect_and_decode(scene)
            good = s['data'] == data and s['detection']['mirrored']
        except Exception:
            good = False
        ok &= good
        print(f"[테스트 ④] 거울상 {profile}: {'PASS' if good else 'FAIL'}")
    return ok


def test_explicit_failure():
    """⑤ 코드 없음·파인더 가림 → 명시적 예외 (침묵 경로 없음)."""
    rng = np.random.default_rng(42)
    blank = Image.fromarray(np.full((400, 500, 3), 255, dtype=np.uint8))
    noisy = Image.fromarray(
        rng.integers(0, 256, (400, 500, 3), dtype=np.uint8).astype(np.uint8))
    blobs = np.full((400, 500, 3), 255, dtype=np.uint8)
    for _ in range(8):     # 무작위 검정 원 — 파인더 비슷한 미끼
        x, y, r = rng.integers(50, 450), rng.integers(50, 350), rng.integers(8, 25)
        cv2.circle(blobs, (int(x), int(y)), int(r), (0, 0, 0), -1)
    blob_img = Image.fromarray(blobs)

    data, _, img, _ = make_code('A')
    scene, _ = make_scene(img, angle=70)
    # 장면 좌반의 어두운 픽셀 전부 흰색 → 파인더 1개 이상 소실 (가설 불가)
    occluded = np.array(scene)
    left = occluded[:, :occluded.shape[1] // 2]
    left[left.sum(axis=2) < 384] = 255
    occ_img = Image.fromarray(occluded)

    ok = True
    for name, im in (('빈 이미지', blank), ('순수 잡음', noisy),
                     ('블롭 미끼', blob_img), ('파인더 가림', occ_img)):
        try:
            s = detect_and_decode(im)
            verdict = 'wrong' if s['data'] != data else 'ok'
        except Exception as e:
            verdict = f"명시적 실패({type(e).__name__})"
        passed = verdict.startswith('명시적')
        ok &= passed
        print(f"[테스트 ⑤] {name:8s}: {'PASS' if passed else 'FAIL'} ({verdict})")
    return ok


def test_architecture_separation():
    """⑦ FTO 분리 — 검출 모듈이 소거·신뢰도 경로를 임포트하지 않음."""
    forbidden = re.compile(
        r"^\s*(from|import)\s+\S*(softdecode|spatial|consensus|scramble|"
        r"bitutils|reedsolo)", re.MULTILINE)
    ok = True
    for f in sorted((ROOT / "src" / "exp16_detect").glob("*.py")):
        hits = forbidden.findall(f.read_text(encoding='utf-8'))
        clean = not hits
        ok &= clean
        if not clean:
            print(f"[테스트 ⑦] {f.name}: FAIL (금지 임포트 {hits})")
    # FTO 주석 명시 확인
    tagged = all("FTO" in (ROOT / "src" / "exp16_detect" / n).read_text(
        encoding='utf-8') for n in ("finder_detect.py", "affine.py",
                                    "pipeline.py"))
    ok &= tagged
    print(f"[테스트 ⑦] 아키텍처 분리(FTO): {'PASS' if ok else 'FAIL'} "
          f"(금지 임포트 없음, FTO 주석 명시={tagged})")
    return ok


def test_regression():
    """⑧ exp13 전체 회귀 (내부에서 exp9/exp11·CLI 통합까지 수행)."""
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(Path("tests") /
                                           "test_exp13_selfdesc.py")],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    passed = proc.returncode == 0
    print(f"[테스트 ⑧] 회귀 test_exp13_selfdesc.py: "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        print((proc.stdout or "")[-3000:])
        print((proc.stderr or "")[-2000:])
    return passed


if __name__ == "__main__":
    results = [
        test_finder_candidates(),
        test_affine_accuracy(),
        test_detect_roundtrip(),
        test_mirror(),
        test_explicit_failure(),
        test_architecture_separation(),
        test_regression(),
    ]
    print()
    if all(results):
        print("=== exp16 전체 테스트 PASS ===")
    else:
        print("=== exp16 테스트 FAIL ===")
        sys.exit(1)
