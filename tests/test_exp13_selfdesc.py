"""
exp13 테스트 — 자기기술 코드 (포맷 링 + 파인더 비대칭화 + 기하 자기추정)
실행: py -X utf8 tests/test_exp13_selfdesc.py  (프로젝트 루트에서)

① 파인더 비대칭화 증명: (a) 기존 동일 형상 배치는 회전 3 × 거울상에서
   구분 불가(D3 — hex_grid 주석 오류의 실증), (b) exp13 차등 파인더는
   6가지 방향 가설 중 항등에서만 패턴 일치 (axial 전수 검증)
② 포맷 링 불변식: 셀 18×3 사본 격자 내·서로소·파인더와 비중첩 (전 GRID_RADII),
   pack/parse 왕복, 1사본 전파괴 다수결 복원
③ 기하 자기추정: 렌더 이미지 크기 → (radius, hex_size, origin) 역산 정확
④ 방향 판별 유일성 (이미지): 회전 0/120/240 × 거울상 6변형 각각에서
   파인더 패턴 가설이 정확히 1개 생존
⑤ 자기기술 왕복: 메타데이터 전무한 PNG 파일에서 프로파일 A/B ×
   회전 0°/120°/240° = 6케이스 왕복 100%
⑥ 거울상 왕복 (보너스): 거울상·거울상+회전 입력도 복호
⑦ CLI 통합: dqr_encode --exp13 → 메타데이터 없는 PNG → dqr_decode 자동 인식
⑧ 회귀: exp9/exp11 왕복 테스트 PASS (hex_grid 주석 정정이 동작 불변임을 확인)
⑨ 군집 블롯 오염: 메타데이터 없는 PNG + 360px 균일 회색 블롯 —
   사다리 실패(스윕 비활성 시 명시적 실패, 미검출 오복호 0)를 v13 공간 소거
   스윕(exp13 셀 배치 그룹 유도)이 다중 가설 합의로 정확 복구
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageOps

from src.core.hex_grid import (HEX_DIRECTIONS, build_hex_grid,
                               finder_cell_set, finder_centers)
from src.exp9_soft.encoder_v9 import GRID_RADII
from src.exp13_selfdesc.encoder_v13 import DQREncoderV13
from src.exp13_selfdesc.renderer_v13 import DQRRendererV13
from src.exp13_selfdesc.decoder_v13 import DQRDecoderV13
from src.exp13_selfdesc.format_ring import (FORMAT_BITS, majority_bits,
                                            pack_format, parse_format)
from src.exp13_selfdesc.layout import (axial_transform, build_reserved_v13,
                                       estimate_geometry,
                                       expected_finder_pattern,
                                       format_cells_per_finder)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs" / "exp13"
TEXT = "dqr 시험테스트입니다 — exp13 자기기술(포맷 링+차등 파인더) 검증"

ALL_HYPOTHESES = [(k, m) for m in (False, True) for k in (0, 1, 2)]

# 테스트 ⑨의 결정론적 군집 오염 (사전 탐색으로 고정한 파라미터):
# (size, fy, fx) — 이미지 좌상단 기준 비율 위치에 중간 회색(140) 정사각.
# exp11b CLUSTER_BLOT와 같은 모델이되 exp13 셀 배치(포맷 링 예약)에 맞춰
# 사다리 실패·스윕 복구가 재현되는 위치를 따로 탐색해 고정했다.
CLUSTER_BLOT_V13 = (360, 0.60, 0.25)


def make_code(profile):
    data = TEXT.encode("utf-8")
    enc = DQREncoderV13(profile=profile)
    result = enc.encode(data)
    img, info = DQRRendererV13(hex_size=32).render(result)
    return data, result, img, info


def rotated(img, deg):
    if deg == 0:
        return img
    return img.rotate(deg, resample=Image.Resampling.BICUBIC,
                      fillcolor=(255, 255, 255))


def test_finder_asymmetry_axial():
    """① (a) 기존 배치의 D3 모호성 실증 + (b) 차등 파인더의 유일 판별."""
    ok = True
    for radius in GRID_RADII[:3]:
        # (a) 동일 형상 파인더(중심만 채움) → 회전 3가지 전부 + 반사
        #     (q,r)→(-r,-q)에서 패턴 불변 = 방향 구분 불가 (주석 오류의 실증)
        uniform = {}
        for cq, cr in finder_centers(radius):
            uniform[(cq, cr)] = 1
            for dq, dr in HEX_DIRECTIONS:
                uniform[(cq + dq, cr + dr)] = 0
        rot_ambiguous = all(
            all(uniform.get(axial_transform(q, r, k, False)) == b
                for (q, r), b in uniform.items())
            for k in (1, 2))
        mirror_ambiguous = all(uniform.get((-r, -q)) == b
                               for (q, r), b in uniform.items())

        # (b) exp13 차등 패턴 → 6가설 중 항등 (0, False) 에서만 일치
        pattern = {(q, r): b for q, r, b in expected_finder_pattern(radius)}
        matches = []
        for k, m in ALL_HYPOTHESES:
            hit = all(pattern.get(axial_transform(q, r, k, m)) == b
                      for (q, r), b in pattern.items())
            if hit:
                matches.append((k, m))
        unique = (matches == [(0, False)])

        case_ok = rot_ambiguous and mirror_ambiguous and unique
        ok &= case_ok
        print(f"[테스트 ①] r={radius:2d}: {'PASS' if case_ok else 'FAIL'} "
              f"(구배치 회전모호={rot_ambiguous}, 거울모호={mirror_ambiguous}, "
              f"차등 파인더 유일판별={unique}, 일치 가설={matches})")
    return ok


def test_format_ring_invariants():
    """② 포맷 셀 배치 불변식 + 워드 pack/parse + 사본 다수결."""
    ok = True
    for radius in GRID_RADII:
        copies = format_cells_per_finder(radius)
        fcells = set()
        for c in finder_centers(radius):
            fcells |= finder_cell_set(*c)
        sets = [set(c) for c in copies]
        sizes_ok = all(len(c) == FORMAT_BITS for c in copies)
        disjoint = (len(sets[0] | sets[1] | sets[2])
                    == 3 * FORMAT_BITS) and not (sets[0] | sets[1] | sets[2]) & fcells
        in_grid_ok = (sets[0] | sets[1] | sets[2]) <= set(build_hex_grid(radius))
        reserved = build_reserved_v13(radius)
        reserved_ok = len(reserved) == 21 + 3 * FORMAT_BITS
        ok &= sizes_ok and disjoint and in_grid_ok and reserved_ok
        if not (sizes_ok and disjoint and in_grid_ok and reserved_ok):
            print(f"[테스트 ②] r={radius}: FAIL (크기={sizes_ok}, "
                  f"서로소={disjoint}, 격자내={in_grid_ok}, 예약={reserved_ok})")
    print(f"[테스트 ②] 포맷 셀 배치 (전 GRID_RADII): {'PASS' if ok else 'FAIL'}")

    # pack/parse 왕복 (필드 경계값 표본 — exp15b: scramble 1b + reserve 2b)
    rt_ok = True
    for fields in ({'version': 13, 'profile': 0, 'ecc_level': 2,
                    'radius_idx': 1, 'mask_id': 0, 'scramble': 1,
                    'reserve': 0},
                   {'version': 15, 'profile': 3, 'ecc_level': 3,
                    'radius_idx': 15, 'mask_id': 7, 'scramble': 1,
                    'reserve': 3},
                   {'version': 0, 'profile': 1, 'ecc_level': 0,
                    'radius_idx': 0, 'mask_id': 5, 'scramble': 0,
                    'reserve': 3}):
        bits = pack_format(fields)
        rt_ok &= (len(bits) == FORMAT_BITS and parse_format(bits) == fields)
    print(f"[테스트 ②] 포맷 워드 pack/parse 왕복: {'PASS' if rt_ok else 'FAIL'}")

    # 1사본 전파괴(반전) → 나머지 2사본 다수결로 무손실 복원
    word = pack_format({'version': 13, 'profile': 1, 'ecc_level': 1,
                        'radius_idx': 2, 'mask_id': 0, 'scramble': 1,
                        'reserve': 0})
    corrupted = [[1 - b for b in word], list(word), list(word)]
    maj_ok = majority_bits(corrupted) == list(word)
    print(f"[테스트 ②] 1사본 전파괴 다수결 복원: {'PASS' if maj_ok else 'FAIL'}")
    return ok and rt_ok and maj_ok


def test_geometry_estimation():
    """③ 이미지 크기 → (radius, hex_size, origin) 역산 정확성."""
    ok = True
    for profile in ('A', 'B'):
        _, result, img, info = make_code(profile)
        cands = estimate_geometry(img.width, img.height)
        unique = len(cands) == 1
        g = cands[0] if cands else {}
        accurate = (unique and g['radius'] == result['radius']
                    and abs(g['hex_size'] - info['hex_size']) < 0.05
                    and abs(g['origin_x'] - info['origin_x']) < 1.0
                    and abs(g['origin_y'] - info['origin_y']) < 1.0)
        ok &= accurate
        print(f"[테스트 ③] 기하 자기추정 {profile}: "
              f"{'PASS' if accurate else 'FAIL'} "
              f"(후보 {len(cands)}개, r={g.get('radius')}, "
              f"hs={g.get('hex_size', 0):.4f}, "
              f"Δox={abs(g.get('origin_x', 0) - info['origin_x']):.3f}px)")
    return ok


def test_orientation_unique_on_images():
    """④ 6가지 변형 이미지 각각에서 방향 가설이 정확히 1개 생존."""
    data, result, img, info = make_code('A')
    dec = DQRDecoderV13()
    variants = [
        ('rot0', img, (0, False)),
        ('rot120', rotated(img, 120), None),
        ('rot240', rotated(img, 240), None),
        ('mirror', ImageOps.mirror(img), None),
        ('mirror+120', rotated(ImageOps.mirror(img), 120), None),
        ('mirror+240', rotated(ImageOps.mirror(img), 240), None),
    ]
    ok = True
    seen = []
    for name, im, expect in variants:
        gray, thr = dec._gray_and_thr(im)
        geo = estimate_geometry(im.width, im.height)[0]
        matches = dec.detect_orientation(gray, thr, geo['radius'],
                                         geo['hex_size'], geo['origin_x'],
                                         geo['origin_y'])
        unique = len(matches) == 1
        exact = (expect is None) or (matches == [expect])
        ok &= unique and exact
        seen.append(matches[0] if unique else None)
        print(f"[테스트 ④] 방향 유일성 {name:11s}: "
              f"{'PASS' if unique and exact else 'FAIL'} (가설={matches})")
    # 6변형이 서로 다른 6가설에 대응 (전 가설 공간 커버)
    cover = len(set(seen)) == 6 and None not in seen
    ok &= cover
    print(f"[테스트 ④] 6가설 전수 커버: {'PASS' if cover else 'FAIL'} ({seen})")
    return ok


def _strip_and_reload(img, path):
    """PNG로 저장(메타데이터 미기록) 후 재로드 — 메타 부재를 단언."""
    img.save(path)
    loaded = Image.open(path)
    dqr_keys = [k for k in loaded.info if str(k).startswith("dqr_")]
    assert not dqr_keys, f"메타데이터 잔존: {dqr_keys}"
    return loaded


def test_selfdescribing_roundtrip():
    """⑤ 메타데이터 없는 PNG: 프로파일 A/B × 회전 0/120/240 왕복 100%."""
    OUT.mkdir(parents=True, exist_ok=True)
    dec = DQRDecoderV13()
    n_ok, n_total = 0, 0
    ok = True
    for profile in ('A', 'B'):
        data, result, img, _ = make_code(profile)
        for deg in (0, 120, 240):
            n_total += 1
            path = OUT / f"exp13_selfdesc_{profile}_rot{deg}.png"
            loaded = _strip_and_reload(rotated(img, deg), path)
            try:
                s = dec.decode_with_stats(loaded)
                good = (s['data'] == data and s['profile'] == profile
                        and s['radius'] == result['radius']
                        and s['nsym'] == result['nsym'])
            except Exception as e:
                good, s = False, {'mode': f"EXC:{type(e).__name__}"}
            n_ok += good
            ok &= good
            print(f"[테스트 ⑤] 왕복 {profile} rot{deg:3d}: "
                  f"{'PASS' if good else 'FAIL'} (mode={s.get('mode')}, "
                  f"orient={s.get('orientation')})")
    print(f"[테스트 ⑤] 자기기술 왕복 합계: {n_ok}/{n_total} "
          f"({'PASS' if n_ok == n_total else 'FAIL'})")
    return ok


def test_mirror_roundtrip():
    """⑥ (보너스) 거울상 입력 — 인쇄 뒷면 투과·전면카메라 반전 시나리오."""
    OUT.mkdir(parents=True, exist_ok=True)
    dec = DQRDecoderV13()
    ok = True
    for profile in ('A', 'B'):
        data, _, img, _ = make_code(profile)
        for name, im in (('mirror', ImageOps.mirror(img)),
                         ('mirror_rot120', rotated(ImageOps.mirror(img), 120))):
            path = OUT / f"exp13_selfdesc_{profile}_{name}.png"
            loaded = _strip_and_reload(im, path)
            try:
                s = dec.decode_with_stats(loaded)
                good = s['data'] == data and s['orientation']['mirrored']
            except Exception:
                good = False
            ok &= good
            print(f"[테스트 ⑥] 거울상 {profile} {name:14s}: "
                  f"{'PASS' if good else 'FAIL'}")
    return ok


def test_cli_integration():
    """⑦ dqr_encode --exp13 → 메타 없는 PNG → dqr_decode 자동 인식."""
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "exp13_cli_test.png"
    enc_proc = subprocess.run(
        [sys.executable, "-X", "utf8", "dqr_encode.py", TEXT,
         "-o", str(png), "--exp13", "--profile", "B"],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    enc_ok = enc_proc.returncode == 0 and png.exists()

    meta_ok = False
    if enc_ok:
        info = Image.open(png).info
        meta_ok = not any(str(k).startswith("dqr_") for k in info)

    dec_proc = subprocess.run(
        [sys.executable, "-X", "utf8", "dqr_decode.py", str(png)],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    dec_ok = dec_proc.returncode == 0 and TEXT in (dec_proc.stdout or "")

    ok = enc_ok and meta_ok and dec_ok
    print(f"[테스트 ⑦] CLI 통합: {'PASS' if ok else 'FAIL'} "
          f"(인코딩={enc_ok}, 메타 부재={meta_ok}, 디코딩={dec_ok})")
    if not ok:
        print((enc_proc.stdout or "")[-1000:], (enc_proc.stderr or "")[-1000:])
        print((dec_proc.stdout or "")[-1000:], (dec_proc.stderr or "")[-1000:])
    return ok


def test_regression():
    """⑧ 기존 exp9/exp11 왕복 회귀 (hex_grid 주석 정정 = 동작 불변)."""
    ok = True
    for name in ("test_exp9_roundtrip.py", "test_exp11_roundtrip.py"):
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", str(Path("tests") / name)],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        passed = proc.returncode == 0
        ok &= passed
        print(f"[테스트 ⑧] 회귀 {name:26s}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print((proc.stdout or "")[-2000:])
            print((proc.stderr or "")[-2000:])
    return ok


def test_cluster_sweep_recovery():
    """⑨ 군집 블롯: v13 스윕 합의 복구 + 스윕 비활성 시 미검출 오복호 0."""
    OUT.mkdir(parents=True, exist_ok=True)
    data, result, img, _ = make_code('A')
    size, fy, fx = CLUSTER_BLOT_V13
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    y, x = int(h * fy), int(w * fx)
    arr[y:y + size, x:x + size] = 140   # 임계(128) 위 균일 회색
    path = OUT / "exp13_cluster_blot_A.png"
    loaded = _strip_and_reload(Image.fromarray(arr), path)

    # 스윕 활성(기본): 메타데이터 없이 정확 복구 + 합의 수락
    try:
        s = DQRDecoderV13().decode_with_stats(loaded)
        recovered = s['data'] == data
        sweep_used = s['mode'] == 'sweep'
        consensus = s.get('consensus', 0)
    except Exception:
        recovered, sweep_used, consensus = False, False, 0

    # 스윕 비활성: 사다리만으로는 명시적 실패여야 한다
    # ('wrong' = 미검출 오복호 → FAIL, 'ok' = 블롯이 스윕을 강제하지 못함 → FAIL)
    try:
        s0 = DQRDecoderV13(use_sweep=False).decode_with_stats(loaded)
        no_sweep = 'ok' if s0['data'] == data else 'wrong'
    except Exception:
        no_sweep = 'fail'

    ok = (recovered and sweep_used and consensus >= 3
          and no_sweep == 'fail')
    print(f"[테스트 ⑨] 군집 블롯 A {size}px: {'PASS' if ok else 'FAIL'} "
          f"(스윕 복구={recovered}, mode sweep={sweep_used}, "
          f"합의 {consensus}건, 스윕 비활성={no_sweep})")
    return ok


if __name__ == "__main__":
    results = [
        test_finder_asymmetry_axial(),
        test_format_ring_invariants(),
        test_geometry_estimation(),
        test_orientation_unique_on_images(),
        test_selfdescribing_roundtrip(),
        test_mirror_roundtrip(),
        test_cli_integration(),
        test_regression(),
        test_cluster_sweep_recovery(),
    ]
    print()
    if all(results):
        print("=== exp13 전체 테스트 PASS ===")
    else:
        print("=== exp13 테스트 FAIL ===")
        sys.exit(1)
