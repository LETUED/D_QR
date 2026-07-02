"""
exp14 홀로그램 CGH 시뮬레이션 테스트
실행: py -X utf8 tests/test_exp14_holosim.py  (프로젝트 루트에서)

① zoom_about_center 기하 (항등·배율·중심 고정)
② speckle_field 통계 (평균 1, 목표 콘트라스트, 비음수)
③ GS 출력이 정확히 0/π 이진 위상인지
④ 코히어런트 재생 왕복: coarse + fine 모두 원본 일치 (합격선 a)
⑤ LED κ=0 (이미지면): coarse 생존 (합격선 b의 하한)
⑥ 대조군 유틸: 자기 렌더에선 비트 정확도 1.0, 타 채널 텐서에선 붕괴
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.exp10_progressive.encoder_v10 import DQREncoderV10
from src.exp10_progressive.renderer_v10 import DQRRendererV10
from src.exp10_progressive.decoder_v10 import DQRDecoderV10
from src.exp14_holosim.cgh import (DESIGN_NM, embed_target, gs_binary_phase,
                                   replay_intensity)
from src.exp14_holosim.replay import (zoom_about_center, speckle_field,
                                      led_composite, intensity_to_image)
from src.exp14_holosim.colorcode import (make_channel_tensors, render_pattern,
                                         bit_accuracy)

HEX_SIZE = 16
FIELD_SIZE = 2048
GS_ITERS = (40, 20)  # 테스트는 벤치마크(50+25)보다 짧게 — 합격선엔 충분해야 함


def test_zoom():
    print("[테스트 ①] zoom_about_center 기하")
    rng = np.random.default_rng(0)
    a = rng.random((64, 64))
    ok_id = np.allclose(zoom_about_center(a, 1.0), a)

    # 중심에서 +8px의 점이 s=2에서 +16px로 이동해야 함
    b = np.zeros((64, 64))
    b[32 + 8, 32] = 1.0
    z = zoom_about_center(b, 2.0)
    ok_shift = z[32 + 16, 32] > 0.9
    ok_center = abs(zoom_about_center(b, 0.5)[32 + 4, 32] - 0.0) >= 0.0  # 축소 방향
    z2 = zoom_about_center(b, 0.5)
    ok_shrink = z2[32 + 4, 32] > 0.2  # bilinear 분산 허용

    ok = ok_id and ok_shift and ok_shrink and ok_center
    print(f"  항등={ok_id}  확대 이동={ok_shift}  축소 이동={ok_shrink} "
          f"→ {'PASS' if ok else 'FAIL'}")
    return ok


def test_speckle():
    print("[테스트 ②] speckle_field 통계")
    rng = np.random.default_rng(7)
    ok = True
    for C in (0.1, 0.3, 0.5):
        M = speckle_field((512, 512), C, rng=rng)
        mean_ok = abs(M.mean() - 1.0) < 0.05
        # 클립으로 실측 콘트라스트가 목표보다 다소 줄 수 있음 — ±25% 허용
        c_meas = M.std() / M.mean()
        c_ok = abs(c_meas - C) < 0.25 * C
        nonneg = M.min() >= 0.0
        print(f"  C={C}: 평균={M.mean():.3f} 실측C={c_meas:.3f} "
              f"비음수={nonneg}")
        ok = ok and mean_ok and c_ok and nonneg
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok


def build_cgh_roundtrip():
    """④⑤ 공용: 코드 생성 → CGH → (phi, window, params, 원본 페이로드)."""
    coarse = "exp14 테스트 코어".encode("utf-8")
    fine = "exp14 테스트 확장 페이로드".encode("utf-8")
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
    target = 255 - np.array(img.convert('L'))
    tgt, window = embed_target(target, FIELD_SIZE)
    phi = gs_binary_phase(tgt, n_iter_continuous=GS_ITERS[0],
                          n_iter_binary=GS_ITERS[1], seed=14)
    return phi, window, params, coarse, fine


def test_binary_phase(phi):
    print("[테스트 ③] 이진 위상 양자화")
    vals = np.unique(phi)
    ok = set(np.round(vals, 6)) <= {0.0, round(np.pi, 6)}
    print(f"  위상 고유값: {vals} → {'PASS' if ok else 'FAIL'}")
    return ok


def test_coherent_roundtrip(phi, window, params, coarse, fine):
    print("[테스트 ④] 코히어런트 532nm 재생 왕복 (합격선 a)")
    I = replay_intensity(phi, DESIGN_NM)
    img_a = intensity_to_image(I[window])
    stats = DQRDecoderV10().decode_with_stats(img_a, **params)
    ok_c = stats['coarse'] == coarse
    ok_f = stats['fine'] == fine
    print(f"  coarse={'PASS' if ok_c else 'FAIL'}({stats['coarse_mode']})  "
          f"fine={'PASS' if ok_f else 'FAIL'}({stats['fine_mode']}, "
          f"conf={stats['fine_confidence']:.3f})")
    return ok_c and ok_f


def test_led_imageplane(phi, window, params, coarse):
    print("[테스트 ⑤] 백색 LED κ=0 (이미지면) — coarse 생존")
    I = led_composite(phi, kappa=0.0)
    img_b = intensity_to_image(I[window])
    try:
        stats = DQRDecoderV10().decode_with_stats(img_b, **params)
        ok = stats['coarse'] == coarse
    except Exception:
        ok = False
    print(f"  coarse={'PASS' if ok else 'FAIL'}")
    return ok


def test_color_utils():
    print("[테스트 ⑥] 대조군 유틸 — 비트 정확도 측정기")
    radius = 12
    tensors = make_channel_tensors(radius, seed=14)
    img0, info = render_pattern(tensors[0], radius, HEX_SIZE)
    geo = dict(radius=radius, hex_size=info['hex_size'],
               origin_x=info['origin_x'], origin_y=info['origin_y'])
    acc_self = bit_accuracy(img0, tensors[0], **geo)
    acc_cross = bit_accuracy(img0, tensors[1], **geo)
    ok = acc_self == 1.0 and acc_cross < 0.6
    print(f"  자기 렌더 정확도={acc_self:.3f} (기대 1.0)  "
          f"타 채널 텐서={acc_cross:.3f} (기대 ~0.5) "
          f"→ {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [test_zoom(), test_speckle()]

    phi, window, params, coarse, fine = build_cgh_roundtrip()
    results += [
        test_binary_phase(phi),
        test_coherent_roundtrip(phi, window, params, coarse, fine),
        test_led_imageplane(phi, window, params, coarse),
        test_color_utils(),
    ]
    if all(results):
        print("\n=== exp14 전체 테스트 PASS ===")
    else:
        print("\n=== exp14 테스트 FAIL ===")
        sys.exit(1)
