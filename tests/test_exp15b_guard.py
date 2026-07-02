"""
exp15b 테스트 — 전부-0 코드워드 퇴화 봉쇄 (4층 가드 + 인코더 스크램블)
실행: py -X utf8 tests/test_exp15b_guard.py  (프로젝트 루트에서)

① 가드 판정 단위: is_degenerate_message — 전부-0/상수 거부, 정상 통과
② 코드워드 레벨: 전부-0 코드워드를 사다리가 reject_degenerate=True에서
   거부, False(exp9~12 동결 기본값)에서는 종전대로 반환
③ exp15 실측 오정정 재현: outputs/exp15/exp15_wrong_blot2_trial8.png —
   가드 해제 시 전부-0 페이로드 오정정 재현(mode=soft, 소거 25, errata 48),
   가드 적용(v11b 기본) 시 명시적 거부
④ 스크램블 단위: LFSR 마스크 결정론·자기 역원·전 바이트 비0·전 주기 비상수
   (CRC 아님 — 검출 기능 없는 전단사 변환)
⑤ v13 왕복: scramble on(기본)/off × 일반·0 위주·빈 페이로드 + 포맷 링
   scramble 비트 기록 검증. 빈 페이로드 + scramble off = 전부-0 메시지는
   가드의 명시적 거부(침묵 오정정 아님)가 올바른 동작임을 문서화
⑥ v9 경로 스크램블 왕복: encoder_v9(scramble=True) + v11b(descramble=True),
   기본(off) 경로 동작 불변 확인
⑦ 가드 무해성: 정상 페이로드의 깨끗한/열화 이미지 복호에 가드가 영향 없음
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

from src.core.scramble import scramble_mask, xor_scramble
from src.core.softdecode import erasure_ladder_decode, is_degenerate_message
from src.exp9_soft.encoder_v9 import DQREncoderV9
from src.exp9_soft.renderer_v9 import DQRRendererV9
from src.exp11_consensus.decoder_v11b import DQRDecoderV11B
from src.exp13_selfdesc.encoder_v13 import DQREncoderV13
from src.exp13_selfdesc.renderer_v13 import DQRRendererV13
from src.exp13_selfdesc.decoder_v13 import DQRDecoderV13

ROOT = Path(__file__).parent.parent
WRONG_PNG = ROOT / "outputs" / "exp15" / "exp15_wrong_blot2_trial8.png"


def test_degenerate_predicate():
    """① is_degenerate_message 단위 판정."""
    cases = [
        (b"\x00" * 59, True, "전부-0"),
        (b"\x41" * 10, True, "상수(0x41)"),
        (b"\x00", True, "1바이트 0"),
        (b"", False, "빈 메시지 (가드 대상 아님 — len 0)"),
        (b"\x00\x01", False, "비상수 2바이트"),
        ("정상 텍스트".encode(), False, "정상 텍스트"),
    ]
    ok = True
    for msg, expect, desc in cases:
        got = is_degenerate_message(msg)
        case_ok = got == expect
        ok &= case_ok
        print(f"[테스트 ①] {desc:28s}: {'PASS' if case_ok else 'FAIL'} "
              f"(판정={got}, 기대={expect})")
    return ok


def test_ladder_guard_codeword_level():
    """② 전부-0 코드워드: 가드 on → 거부 / off(동결 기본) → 종전 반환."""
    nsym, data_len = 76, 59
    cw_len = data_len + nsym
    zero_cw = bytes(cw_len)
    conf = [1] * cw_len   # 균일 저신뢰 — 어느 단계든 RS는 전부-0을 '성공'시킴

    # off (기본값) — exp9~12 동작 동결: 전부-0을 그대로 반환해야 한다
    legacy = erasure_ladder_decode(zero_cw, conf, nsym, data_len)
    legacy_ok = legacy['data'] == bytes(data_len)

    # on — ReedSolomonError로 명시적 거부
    try:
        erasure_ladder_decode(zero_cw, conf, nsym, data_len,
                              reject_degenerate=True)
        guard_ok = False
    except ReedSolomonError:
        guard_ok = True

    # 정상 코드워드는 가드 on에서도 통과해야 한다 (무해성)
    data = bytes(range(1, 60))
    cw = bytes(RSCodec(nsym).encode(data))
    res = erasure_ladder_decode(cw, [9] * len(cw), nsym, data_len,
                                reject_degenerate=True)
    clean_ok = res['data'] == data

    ok = legacy_ok and guard_ok and clean_ok
    print(f"[테스트 ②] 코드워드 레벨 가드: {'PASS' if ok else 'FAIL'} "
          f"(off 동결={legacy_ok}, on 거부={guard_ok}, 정상 통과={clean_ok})")
    return ok


def test_exp15_wrong_case_rejected():
    """③ exp15 blot=2 시행 8 실측 오정정 — 가드 전 재현 / 가드 후 거부."""
    if not WRONG_PNG.exists():
        print(f"[테스트 ③] SKIP — 증거 이미지 없음: {WRONG_PNG}")
        return True
    from benchmarks.exp15_baseline import build_dqr, PAYLOAD
    sym = build_dqr()
    img = Image.open(WRONG_PNG)

    # 가드 해제 — 종전 침묵 오정정(전부-0, mode=soft) 그대로 재현돼야
    # '같은 사례를 막았다'는 주장이 성립한다
    dec_off = DQRDecoderV11B(reject_degenerate=False)
    try:
        res = dec_off.decode_with_stats(img, **sym['params'])
        reproduced = (res['data'] == b"\x00" * len(PAYLOAD)
                      and res['mode'] == 'soft')
        detail = (f"mode={res['mode']}, 소거 {res['erasures_used']}, "
                  f"errata {res['errata_count']}")
    except Exception as e:
        reproduced, detail = False, f"EXC {type(e).__name__}"

    # 가드 적용 (v11b 기본) — 무언가 반환하면 안 된다 (명시적 거부)
    dec_on = DQRDecoderV11B()
    try:
        out = dec_on.decode_with_stats(img, **sym['params'])
        rejected, wrong_returned = False, out['data'] != PAYLOAD
    except ReedSolomonError:
        rejected, wrong_returned = True, False
    except Exception:
        rejected, wrong_returned = True, False   # 다른 명시 실패도 거부로 인정

    ok = reproduced and rejected and not wrong_returned
    print(f"[테스트 ③] blot2 시행8 재현→봉쇄: {'PASS' if ok else 'FAIL'} "
          f"(가드 전 전부-0 재현={reproduced} [{detail}], "
          f"가드 후 거부={rejected})")
    return ok


def test_scramble_unit():
    """④ LFSR 스크램블: 결정론·자기 역원·전 바이트 비0·전 주기 비상수."""
    m1, m2 = scramble_mask(1000), scramble_mask(1000)
    deterministic = m1 == m2
    nonzero = all(b != 0 for b in m1)
    full_period = len(set(m1[:255])) == 255   # 원시다항식 → 주기 255 전수
    data = bytes(range(256)) * 3
    involution = xor_scramble(xor_scramble(data)) == data
    # 전부-0 메시지 → 마스크 자체 = 비퇴화 (가드와의 합 보증)
    zero_safe = not is_degenerate_message(xor_scramble(bytes(100)))
    ok = deterministic and nonzero and full_period and involution and zero_safe
    print(f"[테스트 ④] 스크램블 단위: {'PASS' if ok else 'FAIL'} "
          f"(결정론={deterministic}, 비0={nonzero}, 주기255={full_period}, "
          f"역원={involution}, 전부-0 안전={zero_safe})")
    return ok


def test_v13_scramble_roundtrip():
    """⑤ v13 왕복 (scramble on/off × 페이로드) + 포맷 비트 + 퇴화 엣지."""
    ren = DQRRendererV13(hex_size=32)
    dec = DQRDecoderV13()
    ok = True
    cases = [
        (True, "exp15b 스크램블 왕복 테스트".encode(), 'ok'),
        (True, b"\x00" * 40, 'ok'),      # 0 위주 페이로드 — 스크램블이 보호
        (True, b"", 'ok'),               # 빈 페이로드 — 스크램블 시 정상
        (False, "스크램블 off 호환".encode(), 'ok'),
        (False, b"\x00" * 40, 'ok'),     # 길이 프리픽스 0x28이 비상수 보장
        (False, b"", 'fail'),            # 전부-0 메시지 → 가드의 명시적 거부
    ]
    for scr, data, expect in cases:
        enc = DQREncoderV13(profile='A', scramble=scr)
        result = enc.encode(data)
        img, _ = ren.render(result)
        bit_ok = result['format_fields']['scramble'] == (1 if scr else 0)
        try:
            s = dec.decode_with_stats(img)
            got = 'ok' if (s['data'] == data
                           and s['format']['scramble'] == (1 if scr else 0)) \
                else 'wrong'
        except Exception:
            got = 'fail'
        case_ok = bit_ok and got == expect
        ok &= case_ok
        print(f"[테스트 ⑤] v13 scr={str(scr):5s} len={len(data):2d}: "
              f"{'PASS' if case_ok else 'FAIL'} "
              f"(결과={got}, 기대={expect}, 포맷비트={bit_ok})")
    return ok


def test_v9_scramble_roundtrip():
    """⑥ v9 인코더 scramble 옵션 + v11b descramble 왕복 (기본 off 불변)."""
    data = "v9 경로 스크램블 호환 테스트".encode()
    ren = DQRRendererV9(hex_size=32)

    def params_of(result, info):
        return dict(radius=result['radius'], hex_size=info['hex_size'],
                    bits_per_cell=result['bits_per_cell'],
                    origin_x=info['origin_x'], origin_y=info['origin_y'],
                    nsym=result['nsym'], data_len=result['data_len'],
                    codeword_len=result['codeword_len'])

    # scramble on + descramble on
    r1 = DQREncoderV9(profile='A', scramble=True).encode(data)
    img1, info1 = ren.render(r1)
    out1 = DQRDecoderV11B(descramble=True).decode_with_stats(
        img1, **params_of(r1, info1))['data']
    on_ok = out1 == data and r1['scrambled']

    # 기본 (둘 다 off) — 기존 동작 불변
    r0 = DQREncoderV9(profile='A').encode(data)
    img0, info0 = ren.render(r0)
    out0 = DQRDecoderV11B().decode_with_stats(
        img0, **params_of(r0, info0))['data']
    off_ok = out0 == data and not r0['scrambled']

    # 불일치 조합(인코더 on + 디코더 off)은 원문이 나오면 안 된다 —
    # 스크램블이 실제로 코드워드를 바꿨다는 증거
    out_mismatch = DQRDecoderV11B().decode_with_stats(
        img1, **params_of(r1, info1))['data']
    changed_ok = out_mismatch != data and xor_scramble(out_mismatch) == data

    ok = on_ok and off_ok and changed_ok
    print(f"[테스트 ⑥] v9 스크램블 왕복: {'PASS' if ok else 'FAIL'} "
          f"(on={on_ok}, 기본 off 불변={off_ok}, 변환 실효={changed_ok})")
    return ok


def test_guard_harmless_on_degraded():
    """⑦ 가드 무해성: 정상 페이로드의 노이즈 열화 이미지 복호 결과 동일."""
    from benchmarks.exp9_benchmark import add_noise
    data = "가드 무해성 — 정상 복호 보존".encode()
    enc = DQREncoderV9(profile='A')
    result = enc.encode(data)
    img, info = DQRRendererV9(hex_size=32).render(result)
    params = dict(radius=result['radius'], hex_size=info['hex_size'],
                  bits_per_cell=result['bits_per_cell'],
                  origin_x=info['origin_x'], origin_y=info['origin_y'],
                  nsym=result['nsym'], data_len=result['data_len'],
                  codeword_len=result['codeword_len'])
    dec_on = DQRDecoderV11B()                          # 가드 기본 on
    dec_off = DQRDecoderV11B(reject_degenerate=False)  # 종전 동작

    ok = True
    for t in range(10):
        rng = np.random.default_rng(42 + t)
        deg = add_noise(img, 100, rng)
        def run(d):
            try:
                return d.decode_with_stats(deg, **params)['data']
            except Exception:
                return None
        same = run(dec_on) == run(dec_off) == data
        ok &= same
    print(f"[테스트 ⑦] 가드 무해성 (노이즈 σ100 ×10): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [
        test_degenerate_predicate(),
        test_ladder_guard_codeword_level(),
        test_exp15_wrong_case_rejected(),
        test_scramble_unit(),
        test_v13_scramble_roundtrip(),
        test_v9_scramble_roundtrip(),
        test_guard_harmless_on_degraded(),
    ]
    print()
    if all(results):
        print("=== exp15b 전체 테스트 PASS ===")
    else:
        print("=== exp15b 테스트 FAIL ===")
        sys.exit(1)
