"""
exp12 침묵 오정정 3층 방어 단위 테스트 + 회귀 확인
실행: py -X utf8 tests/test_exp12_defense.py  (프로젝트 루트에서)

① 1층: 사다리 단계 목록이 nsym-2로 상한 적용되는지
② 2층: 재인코딩 대조가 보고되지 않은 위치의 불일치를 잡는지
③ 1층 통합: 풀소거 침묵 오정정 시나리오(결정론적 구성)가
   구동작에서는 wrong-success, 방어 후에는 차단되는지
④ 3층: 깨끗한 이미지는 게이트 통과+fine 복호, 강블러는 RS 미시도 차단
⑤ exp10 blur 6.0px 실측 침묵 오정정 케이스가 v12에서 차단되는지
⑥ 회귀: exp9/exp10/exp11 왕복 테스트가 여전히 PASS인지 (서브프로세스)
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reedsolo import RSCodec, ReedSolomonError

from src.core.softdecode import (erasure_ladder_decode, ladder_steps,
                                 reencode_mismatch_ok, LADDER_CAP_SLACK)

ROOT = Path(__file__).parent.parent


def test_layer1_ladder_cap():
    """① 사다리 최종 단계 = nsym-2 (cap=False면 구동작 nsym)."""
    ok = True
    ok &= ladder_steps(51) == [0, 17, 34, 49]
    ok &= ladder_steps(51, cap=False) == [0, 17, 34, 51]
    ok &= ladder_steps(76) == [0, 25, 50, 74]
    # 모든 nsym에서 상한 보장 + 단계 중복 없음
    for nsym in range(2, 128):
        steps = ladder_steps(nsym)
        ok &= max(steps) <= nsym - LADDER_CAP_SLACK or max(steps) == 0
        ok &= len(steps) == len(set(steps))
        ok &= steps == sorted(steps)
    print(f"[테스트 ①] 1층 사다리 상한 (nsym → nsym-{LADDER_CAP_SLACK})"
          f" : {'PASS' if ok else 'FAIL'}")
    return ok


def test_layer2_reencode_check():
    """② 재인코딩 대조: 불일치가 보고 위치 안이면 통과, 밖이면 차단."""
    nsym = 10
    rsc = RSCodec(nsym)
    data = b"exp12 layer2 check"
    codeword = bytearray(rsc.encode(data))
    codeword[3] ^= 0x5A  # 위치 3 오염

    # RS가 위치 3을 정정했다고 보고한 경우 → 정상
    ok_reported = reencode_mismatch_ok(rsc, bytes(codeword), data,
                                       erase_pos=None, errata_pos=[3])
    # 아무 정정도 보고하지 않았는데 코드워드가 다름 → 오정정 의심
    ok_unreported = reencode_mismatch_ok(rsc, bytes(codeword), data,
                                         erase_pos=None, errata_pos=[])
    # 소거로 표시된 위치의 불일치는 허용
    ok_erased = reencode_mismatch_ok(rsc, bytes(codeword), data,
                                     erase_pos=[3], errata_pos=[])
    ok = ok_reported and (not ok_unreported) and ok_erased
    print(f"[테스트 ②] 2층 재인코딩 대조"
          f" : {'PASS' if ok else 'FAIL'} "
          f"(보고됨={ok_reported}, 미보고 차단={not ok_unreported}, "
          f"소거 허용={ok_erased})")
    return ok


def test_layer1_blocks_full_erasure_miscorrection():
    """
    ③ 풀소거 침묵 오정정의 결정론적 재현 + 차단.

    nsym=8, 오류 5개를 '신뢰도가 높은' 위치에 주입하고 신뢰도 최하위
    8개는 전부 정상 바이트로 구성 → hard·중간 단계는 2·5+k > 8로 실패,
    구동작의 풀소거(k=8) 단계는 미지수 8 = 방정식 8이라 항상 풀리면서
    비소거 위치의 오류 5개가 그대로 남은 다른 코드워드를 '성공' 반환.
    """
    nsym = 8
    rsc = RSCodec(nsym)
    data = b"exp12 layer1"          # 12B → 코드워드 20B
    codeword = bytearray(rsc.encode(data))
    n = len(codeword)

    err_pos = [0, 3, 6, 9, 12]      # 오류 5개 (> nsym/2 = 4 → hard 불가)
    for p in err_pos:
        codeword[p] ^= 0xA5
    # 신뢰도 조작: 오류 위치는 높게, 정상 위치 중 8개는 낮게
    conf = [100 if i in err_pos else 10 for i in range(n)]

    try:
        res = erasure_ladder_decode(bytes(codeword), conf, nsym, len(data),
                                    cap=False, verify=False)
        legacy_wrong = res['data'] != data and res['erasures_used'] == nsym
    except ReedSolomonError:
        legacy_wrong = False  # 구동작이 오정정을 안 냈다면 시나리오 무효

    try:
        erasure_ladder_decode(bytes(codeword), conf, nsym, len(data))
        defended_blocked = False
    except ReedSolomonError:
        defended_blocked = True

    ok = legacy_wrong and defended_blocked
    print(f"[테스트 ③] 풀소거 오정정 재현·차단"
          f" : {'PASS' if ok else 'FAIL'} "
          f"(구동작 오정정={legacy_wrong}, 방어 후 차단={defended_blocked})")
    return ok


def _build_exp10_code():
    from src.exp10_progressive.encoder_v10 import DQREncoderV10
    from src.exp10_progressive.renderer_v10 import DQRRendererV10
    coarse = "https://dqr.kr/v10 — exp10 코어 페이로드".encode("utf-8")
    fine = ("exp10 확장 페이로드: 신뢰도 게이트 점진적 공개 실증용 텍스트. "
            "고해상 촬영에서만 열리는 부가 정보(상세 설명·서명·메타데이터)를 "
            "fine 스트림에 싣는다.").encode("utf-8")
    result = DQREncoderV10().encode(coarse, fine)
    img, info = DQRRendererV10(hex_size=32).render(result)
    params = dict(
        radius=result['radius'], hex_size=info['hex_size'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym_c=result['nsym_c'], nsym_f=result['nsym_f'],
        coarse_len=result['coarse_len'], fine_len=result['fine_len'],
        codeword_len_c=result['codeword_len_c'],
        codeword_len_f=result['codeword_len_f'])
    return coarse, fine, img, params


def test_layer3_gate():
    """④ 3층 게이트: 깨끗한 이미지 통과, 강블러(8px)는 RS 미시도 차단."""
    from PIL import ImageFilter
    from src.exp12_defense.decoder_v12 import DQRDecoderV12

    coarse, fine, img, params = _build_exp10_code()
    dec = DQRDecoderV12()

    s_clean = dec.decode_with_stats(img, **params)
    clean_ok = (s_clean['gate_passed'] and s_clean['rs_attempted']
                and s_clean['fine'] == fine and s_clean['coarse'] == coarse
                and s_clean['fine_verified'] is True)

    # blur 8.0px: vote 신뢰도 고원(≈0.44)이 임계(0.405)를 넘어 게이트는
    # 통과하지만, 1·2층 사다리에서 실패해 fine=None으로 강등돼야 한다.
    blurred = img.filter(ImageFilter.GaussianBlur(8.0))
    s_blur = dec.decode_with_stats(blurred, **params)
    blur_ok = (s_blur['fine'] is None and s_blur['coarse'] == coarse)

    # 게이트 자체의 RS 미시도 차단은 임계 직하 케이스(blur 6.0px)에서 확인
    # → 테스트 ⑤. 여기서는 임계를 올린 디코더로 차단 경로를 직접 검증.
    dec_strict = DQRDecoderV12(gate_threshold=0.99)
    s_strict = dec_strict.decode_with_stats(blurred, **params)
    strict_ok = (not s_strict['gate_passed']
                 and not s_strict['rs_attempted']
                 and s_strict['fine'] is None)

    ok = clean_ok and blur_ok and strict_ok
    print(f"[테스트 ④] 3층 게이트"
          f" : {'PASS' if ok else 'FAIL'} "
          f"(클린 통과·복호={clean_ok}, 블러 강등={blur_ok}, "
          f"게이트 차단 경로={strict_ok})")
    return ok, (coarse, fine, img, params)


def test_blur6_miscorrection_blocked(code):
    """⑤ exp10 blur 6.0px 실측 침묵 오정정 케이스 — 3층 모두에서 차단."""
    import numpy as np
    from PIL import ImageFilter
    from src.exp10_progressive.decoder_v10 import DQRDecoderV10
    from src.exp10_progressive.encoder_v10 import split_data_cells
    from src.exp12_defense.decoder_v12 import DQRDecoderV12

    coarse, fine, img, params = code
    deg = img.filter(ImageFilter.GaussianBlur(6.0))

    # 방어 전(구동작) 재현: fine이 잘못된 데이터를 '성공'으로 반환해야
    # 이 케이스가 여전히 유효한 회귀 표본임이 입증된다
    dec10 = DQRDecoderV10()
    _, fine_cells = split_data_cells(params['radius'])
    pixels = np.array(deg.convert('RGB'))
    cw_f, conf_f = dec10._read_stream(
        pixels, fine_cells, 12, params['hex_size'],
        params['origin_x'], params['origin_y'], params['codeword_len_f'])
    try:
        res = erasure_ladder_decode(cw_f, conf_f, params['nsym_f'],
                                    params['fine_len'],
                                    cap=False, verify=False)
        legacy_silent = res['data'] != fine  # 성공 반환 + 원본 불일치
    except ReedSolomonError:
        legacy_silent = False

    # 방어 후: v12는 fine=None (게이트 차단), 게이트 무력화해도 1·2층이 차단
    s12 = DQRDecoderV12().decode_with_stats(deg, **params)
    gated = s12['fine'] is None and not s12['rs_attempted']
    s12n = DQRDecoderV12(gate_threshold=0.0).decode_with_stats(deg, **params)
    layers12_block = s12n['fine'] is None and s12n['rs_attempted']

    ok = legacy_silent and gated and layers12_block
    print(f"[테스트 ⑤] blur 6.0px 침묵 오정정 차단"
          f" : {'PASS' if ok else 'FAIL'} "
          f"(구동작 오정정 재현={legacy_silent}, 게이트 차단={gated}, "
          f"1+2층 차단={layers12_block})")
    return ok


def test_regression_exp9_10_11():
    """⑥ 기존 왕복 테스트 회귀 확인 (1·2층이 기본 적용된 상태)."""
    ok = True
    for name in ("test_exp9_roundtrip.py", "test_exp10_roundtrip.py",
                 "test_exp11_roundtrip.py"):
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", str(Path("tests") / name)],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        passed = proc.returncode == 0
        ok &= passed
        print(f"[테스트 ⑥] 회귀 {name:28s}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
    return ok


if __name__ == "__main__":
    results = [
        test_layer1_ladder_cap(),
        test_layer2_reencode_check(),
        test_layer1_blocks_full_erasure_miscorrection(),
    ]
    gate_ok, code = test_layer3_gate()
    results.append(gate_ok)
    results.append(test_blur6_miscorrection_blocked(code))
    results.append(test_regression_exp9_10_11())

    print()
    if all(results):
        print("=== exp12 전체 방어 테스트 PASS ===")
    else:
        print("=== exp12 방어 테스트 FAIL ===")
        sys.exit(1)
