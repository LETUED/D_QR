"""
exp11b 테스트 — exp11 중단규칙 3대 결함 해소 검증
실행: py -X utf8 tests/test_exp11b.py  (프로젝트 루트에서)

① 왕복: 깨끗한 이미지 프로파일 A/B 인코딩 → v11b 복호 → 원본 일치
② FTO 불변식: 두 링 패스 모두 바이트 신뢰도가 이산 표 개수인지
③ 공간 그룹 불변식: 그룹당 바이트 ≤ (nsym-2)//2, 합집합 = 코드워드 전체
④ 결함② 블러 절벽: blur 9.5px에서 v11 실패 / v11b 내부 링 패스로 복구
⑤ 결함① 클러스터 오염: v11이 실패(A에선 침묵 오정정)하는 군집 오염을
   v11b 공간 소거 스윕이 합의 수락으로 정확 복구
⑥ 결함③-a 단위: 청크 errata 예산 검사의 경계 동작
⑦ 결함③ 통합: exp12 코드워드 시뮬 축소판(N=1000, seed=42 동일 모델)에서
   1+2층 잔존 오정정 > 0 이지만 v11b 방어는 0
⑧ 회귀: 기존 exp9/exp10/exp11 왕복 테스트가 여전히 PASS (서브프로세스)
"""
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageFilter
from reedsolo import RSCodec, ReedSolomonError

from src.core.consensus import INNER_RINGS
from src.core.hex_grid import build_hex_grid, build_reserved
from src.core.softdecode import (LADDER_CAP_SLACK, chunk_errata_budget_ok,
                                 erasure_ladder_decode)
from src.exp9_soft.encoder_v9 import DQREncoderV9
from src.exp9_soft.renderer_v9 import DQRRendererV9
from src.exp11_consensus.decoder_v11 import DQRDecoderV11
from src.exp11_consensus.decoder_v11b import DQRDecoderV11B
from src.exp11_consensus.spatial import build_spatial_groups

ROOT = Path(__file__).parent.parent
TEXT = "dqr 시험테스트입니다 — exp11b 3대 결함 해소 검증"

# 테스트 ⑤의 결정론적 군집 오염 (사전 탐색으로 고정한 파라미터):
# (size, fy, fx) — 코드 이미지 좌상단 기준 비율 위치에 중간 회색(140) 정사각
CLUSTER_BLOT = {'A': (360, 0.45, 0.25), 'B': (240, 0.30, 0.15)}

# build_spatial_groups 동작 동결 골든 해시 (data_cells 인자 일반화 이전 산출):
# 기본 인자(v9 예약 셀) 결과가 리팩토링 후에도 바이트 단위 동일해야 한다
GROUP_FREEZE_SHA = {
    'A': '0c34e1ccc4fa87e59e032480cadc6571d6db8676c46ef2a6c7d1d14bf57eda52',
    'B': 'bec114d060c4b43c6baf6a643d7e964f13c91f2b38db81c42b1f6e1a36cd0d6c',
}


def make_code(profile):
    data = TEXT.encode("utf-8")
    enc = DQREncoderV9(profile=profile)   # 인코딩 측은 exp9 그대로
    ren = DQRRendererV9(hex_size=32)
    result = enc.encode(data)
    img, info = ren.render(result)
    params = dict(
        radius=result['radius'], hex_size=info['hex_size'],
        bits_per_cell=result['bits_per_cell'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym=result['nsym'], data_len=result['data_len'],
        codeword_len=result['codeword_len'])
    return data, img, params


def outcome(dec, img, params, data):
    try:
        s = dec.decode_with_stats(img, **params)
        return ('ok' if s['data'] == data else 'wrong'), s
    except Exception:
        return 'fail', None


def test_clean_roundtrip():
    ok = True
    for profile in ('A', 'B'):
        data, img, params = make_code(profile)
        r, s = outcome(DQRDecoderV11B(), img, params, data)
        ok &= (r == 'ok' and s['mode'] == 'hard')
        print(f"[테스트 ①] 왕복 프로파일 {profile}: "
              f"{'PASS' if r == 'ok' else 'FAIL'} (mode={s and s['mode']})")
    return ok


def test_fto_discrete_conf():
    """두 링 패스 모두 신뢰도가 이산 표 개수(1~N 홀수)인지."""
    ok = True
    for profile in ('A', 'B'):
        data, img, params = make_code(profile)
        dec = DQRDecoderV11B()
        for rings in (dec.rings, dec.fallback_rings):
            dec._active_rings = rings
            _, byte_conf = dec._read_codeword(
                img, params['radius'], params['hex_size'],
                params['bits_per_cell'], params['origin_x'],
                params['origin_y'], params['codeword_len'])
            ok &= all(isinstance(c, int) and 1 <= c <= dec.n_samples
                      and c % 2 == 1 for c in byte_conf)
    print(f"[테스트 ②] FTO 이산 신뢰도 (양 패스): {'PASS' if ok else 'FAIL'}")
    return ok


def test_spatial_group_invariants():
    ok = True
    for profile in ('A', 'B'):
        _, _, params = make_code(profile)
        nsym, cw_len = params['nsym'], params['codeword_len']
        groups = build_spatial_groups(params['radius'],
                                      params['bits_per_cell'], cw_len, nsym)
        limit = (nsym - LADDER_CAP_SLACK) // 2
        sizes_ok = all(len(g) <= limit for g in groups)
        coverage = set().union(*map(set, groups)) == set(range(cw_len))
        sorted_ok = all(g == sorted(g) for g in groups)

        # data_cells 인자 일반화의 동작 동결: 기본 인자 = 골든 해시,
        # v9 셀 목록 명시 전달 = 기본 인자와 동일 결과
        frozen = (hashlib.sha256(repr(groups).encode()).hexdigest()
                  == GROUP_FREEZE_SHA[profile])
        reserved = build_reserved(params['radius'])
        v9_cells = [c for c in build_hex_grid(params['radius'])
                    if c not in reserved]
        explicit_same = build_spatial_groups(
            params['radius'], params['bits_per_cell'], cw_len, nsym,
            data_cells=v9_cells) == groups

        case_ok = (sizes_ok and coverage and sorted_ok
                   and frozen and explicit_same)
        ok &= case_ok
        print(f"[테스트 ③] 공간 그룹 {profile}: "
              f"{'PASS' if case_ok else 'FAIL'} "
              f"(그룹 {len(groups)}개, 한도 {limit}, "
              f"크기 {[len(g) for g in groups]}, "
              f"동결={frozen}, 명시인자 동일={explicit_same})")
    return ok


def test_blur_cliff_fixed():
    """결함②: blur 9.5px — v11 실패, v11b는 내부 링 패스로 복구."""
    data, img, params = make_code('A')
    deg = img.filter(ImageFilter.GaussianBlur(9.5))
    r11, _ = outcome(DQRDecoderV11(), deg, params, data)
    r11b, s = outcome(DQRDecoderV11B(), deg, params, data)
    inner_used = s is not None and tuple(s.get('rings_used', ())) == INNER_RINGS
    ok = (r11 != 'ok') and (r11b == 'ok') and inner_used
    print(f"[테스트 ④] 블러 9.5px 절벽 해소: {'PASS' if ok else 'FAIL'} "
          f"(v11={r11}, v11b={r11b}, 내부 링 사용={inner_used})")
    return ok


def test_cluster_sweep_recovery():
    """결함①: 군집 오염 — v11 실패/오정정, v11b 공간 스윕 합의 복구."""
    ok = True
    for profile in ('A', 'B'):
        data, img, params = make_code(profile)
        size, fy, fx = CLUSTER_BLOT[profile]
        arr = np.array(img).copy()
        h, w = arr.shape[:2]
        y, x = int(h * fy), int(w * fx)
        arr[y:y + size, x:x + size] = 140   # 임계(128) 위 균일 회색
        deg = Image.fromarray(arr)

        r11, _ = outcome(DQRDecoderV11(), deg, params, data)
        r11b, s = outcome(DQRDecoderV11B(), deg, params, data)
        sweep_used = s is not None and s.get('mode') == 'sweep'
        consensus = s.get('consensus', 0) if s else 0
        case_ok = (r11 != 'ok' and r11b == 'ok' and sweep_used
                   and consensus >= 3)
        ok &= case_ok
        print(f"[테스트 ⑤] 군집 오염 {profile}: "
              f"{'PASS' if case_ok else 'FAIL'} "
              f"(v11={r11}, v11b={r11b}, sweep={sweep_used}, "
              f"합의 {consensus}건)")
    return ok


def test_budget_check_unit():
    """결함③-a 단위: 2·오류+소거 ≤ nsym-slack 경계."""
    nsym = 10
    rsc = RSCodec(nsym)
    data = b"exp11b budget"
    cw = bytes(rsc.encode(data))

    # 소거 8 + 오류 1(소거 외 위치 불일치) → 2+8=10 = nsym → slack0 통과
    bad = bytearray(cw)
    bad[0] ^= 0x11                      # 위치 0 = '오류' (소거 목록 밖)
    erase = list(range(1, 9))           # 8개 소거
    s0 = chunk_errata_budget_ok(rsc, bytes(bad), data, erase, slack=0)
    s2 = chunk_errata_budget_ok(rsc, bytes(bad), data, erase, slack=2)

    # 소거 9 + 오류 1 → 11 > nsym → slack0도 거부
    erase9 = list(range(1, 10))
    s0_over = chunk_errata_budget_ok(rsc, bytes(bad), data, erase9, slack=0)

    ok = s0 and (not s2) and (not s0_over)
    print(f"[테스트 ⑥] errata 예산 경계: {'PASS' if ok else 'FAIL'} "
          f"(slack0 경계 통과={s0}, slack2 거부={not s2}, "
          f"초과 거부={not s0_over})")
    return ok


def test_codeword_sim_defense():
    """결함③ 통합: exp12 시뮬 축소판 — 1+2층 잔존 > 0, v11b 방어 = 0."""
    from benchmarks.exp12_miscorrection import (SIM_NSYM, SIM_DATA_LEN,
                                                SIM_ERR_RANGE)
    N = 1000
    rng = np.random.default_rng(42)
    rsc = RSCodec(SIM_NSYM)
    cw_len = SIM_DATA_LEN + SIM_NSYM * 2
    wrong = {'defended': 0, 'v11b': 0}
    correct_v11b = 0
    for _ in range(N):
        data = bytes(rng.integers(0, 256, SIM_DATA_LEN, dtype=np.uint8))
        cw = bytearray(rsc.encode(data))
        n_err = int(rng.integers(*SIM_ERR_RANGE))
        pos = rng.choice(cw_len, n_err, replace=False)
        for p in pos:
            cw[p] ^= int(rng.integers(1, 256))
        err_set = set(int(p) for p in pos)
        conf = [int(rng.integers(0, 61)) if i in err_set
                else int(rng.integers(20, 129)) for i in range(cw_len)]
        cw = bytes(cw)
        for name, kw in (('defended', {}),
                         ('v11b', dict(budget=True))):
            try:
                res = erasure_ladder_decode(cw, conf, SIM_NSYM,
                                            SIM_DATA_LEN, **kw)
                if res['data'] != data:
                    wrong[name] += 1
                elif name == 'v11b':
                    correct_v11b += 1
            except ReedSolomonError:
                pass
    ok = wrong['defended'] > 0 and wrong['v11b'] == 0 and correct_v11b > 0
    print(f"[테스트 ⑦] 코드워드 시뮬 축소판 (N={N}): "
          f"{'PASS' if ok else 'FAIL'} "
          f"(1+2층 오정정={wrong['defended']}, v11b 오정정={wrong['v11b']}, "
          f"v11b 정상복호={correct_v11b})")
    return ok


def test_regression():
    """⑧ 기존 왕복 테스트 회귀 (v11b 추가가 구버전 동작을 바꾸지 않음)."""
    ok = True
    for name in ("test_exp9_roundtrip.py", "test_exp10_roundtrip.py",
                 "test_exp11_roundtrip.py"):
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", str(Path("tests") / name)],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        passed = proc.returncode == 0
        ok &= passed
        print(f"[테스트 ⑧] 회귀 {name:28s}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
    return ok


if __name__ == "__main__":
    results = [
        test_clean_roundtrip(),
        test_fto_discrete_conf(),
        test_spatial_group_invariants(),
        test_blur_cliff_fixed(),
        test_cluster_sweep_recovery(),
        test_budget_check_unit(),
        test_codeword_sim_defense(),
        test_regression(),
    ]
    print()
    if all(results):
        print("=== exp11b 전체 테스트 PASS ===")
    else:
        print("=== exp11b 테스트 FAIL ===")
        sys.exit(1)
