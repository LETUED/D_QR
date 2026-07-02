"""
exp15 스모크 테스트 — 기준선 생성·판독 왕복
실행: py -X utf8 tests/test_exp15_smoke.py  (프로젝트 루트에서)

① 가용성: QR(qrcode+cv2/pyzbar)·DataMatrix(ppf+pylibdmtx) 중 최소 1개 +
   D_QR은 항상 가용해야 함
② 깨끗한 캔버스 왕복: 가용한 모든 심볼로지에서 read(canvas) == 페이로드
   (DataMatrix는 upper-shift 직접 인코딩의 정당성 검증을 겸한다)
③ 캔버스 규격: 800×800 RGB 백색 배경
④ 열화 모델 동작: 각 채널 함수가 동일 크기 이미지를 반환
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from benchmarks.exp15_baseline import (AVAILABILITY, CANVAS, CHANNELS,
                                       PAYLOAD, SEED, SYM_LABELS,
                                       build_symbols)


def test_availability(syms):
    ok = 'dqr' in syms and len(syms) >= 2
    print(f"[테스트 ①] 가용성: {'PASS' if ok else 'FAIL'} "
          f"(심볼로지 {list(syms)})")
    for name, status in AVAILABILITY.items():
        print(f"    {name}: {status}")
    return ok


def test_clean_roundtrip(syms):
    ok = True
    for k, sym in syms.items():
        out = sym['read'](sym['canvas'])
        case_ok = out == PAYLOAD
        ok &= case_ok
        print(f"[테스트 ②] 왕복 {SYM_LABELS[k]:18s}: "
              f"{'PASS' if case_ok else 'FAIL'} "
              f"({sym['meta']['symbology']})")
    return ok


def test_canvas_spec(syms):
    ok = True
    for k, sym in syms.items():
        c = sym['canvas']
        corners_white = all(
            c.getpixel(p) == (255, 255, 255)
            for p in [(0, 0), (CANVAS - 1, 0), (0, CANVAS - 1),
                      (CANVAS - 1, CANVAS - 1)])
        case_ok = (c.size == (CANVAS, CANVAS) and c.mode == 'RGB'
                   and corners_white)
        ok &= case_ok
        print(f"[테스트 ③] 캔버스 {SYM_LABELS[k]:18s}: "
              f"{'PASS' if case_ok else 'FAIL'} (size={c.size})")
    return ok


def test_channels(syms):
    img = next(iter(syms.values()))['canvas']
    ok = True
    for ch, cfg in CHANNELS.items():
        rng = np.random.default_rng(SEED)
        deg = cfg['fn'](img, cfg['levels'][0], rng)
        case_ok = deg.size == img.size
        ok &= case_ok
        print(f"[테스트 ④] 채널 {ch:8s}: {'PASS' if case_ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    syms = build_symbols()
    results = [
        test_availability(syms),
        test_clean_roundtrip(syms),
        test_canvas_spec(syms),
        test_channels(syms),
    ]
    print()
    if all(results):
        print("=== exp15 스모크 테스트 PASS ===")
    else:
        print("=== exp15 스모크 테스트 FAIL ===")
        sys.exit(1)
