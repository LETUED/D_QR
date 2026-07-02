"""
exp11 왕복 테스트 — vote consensus 디코더 검증
  1) 깨끗한 이미지: 프로파일 A/B 인코딩 → 렌더 → vote 디코딩 → 원본 일치
  2) 오염 이미지: 임계 근처 회색 패치 살포 → 소거 사다리 soft 복구 확인
  3) FTO 불변식: 바이트 신뢰도가 전부 이산 정수(표 개수)인지 확인

실행: py -X utf8 tests/test_exp11_roundtrip.py  (프로젝트 루트에서)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from src.exp9_soft.encoder_v9 import DQREncoderV9, PROFILES
from src.exp9_soft.renderer_v9 import DQRRendererV9
from src.exp11_consensus.decoder_v11 import DQRDecoderV11

TEXT = "dqr 시험테스트입니다 — exp11 표 일치율(vote consensus) 검증"
PATCH_SEED = 7
# 임계 근처 회색 패치 수 — hard 단독으로는 부족해 소거 사다리(soft)가
# 실제로 발동하는 수준 (프로파일별 격자 크기가 달라 따로 설정)
PATCH_COUNT = {'A': 100, 'B': 50}
PATCH_SIZE = 60


def make_code(profile):
    data = TEXT.encode("utf-8")
    enc = DQREncoderV9(profile=profile)   # 인코딩 측은 exp9 그대로 (변경 없음)
    ren = DQRRendererV9(hex_size=32)
    result = enc.encode(data)
    img, info = ren.render(result)
    params = dict(
        radius=result['radius'],
        hex_size=info['hex_size'],
        bits_per_cell=result['bits_per_cell'],
        origin_x=info['origin_x'],
        origin_y=info['origin_y'],
        nsym=result['nsym'],
        data_len=result['data_len'],
        codeword_len=result['codeword_len'],
    )
    return data, img, params


def add_patches(img, n_patches, rng, patch_size=PATCH_SIZE):
    """exp9_benchmark과 동일한 오염 모델 — 임계값 128 근처 회색 패치."""
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    for _ in range(n_patches):
        x = rng.integers(0, max(1, w - patch_size))
        y = rng.integers(0, max(1, h - patch_size))
        gray = rng.integers(100, 156)
        arr[y:y+patch_size, x:x+patch_size] = gray
    return Image.fromarray(arr)


def clean_roundtrip(profile):
    data, img, params = make_code(profile)
    dec = DQRDecoderV11()

    Path("outputs/exp11").mkdir(parents=True, exist_ok=True)
    out_path = f"outputs/exp11/exp11_roundtrip_profile{profile}.png"
    img.save(out_path)

    stats = dec.decode_with_stats(img, **params)
    ok = stats['data'] == data

    # FTO 불변식: 신뢰도는 이산 표 개수 (홀수 N이면 표차도 홀수, 1~N 범위)
    _, byte_conf = dec._read_codeword(
        img, params['radius'], params['hex_size'], params['bits_per_cell'],
        params['origin_x'], params['origin_y'], params['codeword_len'])
    conf_discrete = all(
        isinstance(c, int) and 1 <= c <= dec.n_samples and c % 2 == 1
        for c in byte_conf)

    print(f"[프로파일 {profile}] {PROFILES[profile]['desc']}")
    print(f"  샘플점/삼각형 : {dec.n_samples}")
    print(f"  복호 모드     : {stats['mode']} (소거 {stats['erasures_used']}개)")
    print(f"  신뢰도 이산성 : {'PASS' if conf_discrete else 'FAIL'}")
    print(f"  이미지        : {out_path} ({img.width}x{img.height})")
    print(f"  왕복 결과     : {'PASS' if ok else 'FAIL'}")
    print()
    return ok and conf_discrete


def contaminated_recovery(profile):
    data, img, params = make_code(profile)
    dec = DQRDecoderV11()

    rng = np.random.default_rng(PATCH_SEED)
    n_patches = PATCH_COUNT[profile]
    dirty = add_patches(img, n_patches, rng)

    out_path = f"outputs/exp11/exp11_contaminated_profile{profile}.png"
    dirty.save(out_path)

    try:
        stats = dec.decode_with_stats(dirty, **params)
        ok = stats['data'] == data
        mode = stats['mode']
        erasures = stats['erasures_used']
    except Exception as e:
        ok, mode, erasures = False, f"실패({e})", "-"

    # 소거 사다리가 실제로 발동했는지 확인 (hard로 풀리면 오염이 부족한 것)
    soft_used = (mode == 'soft')

    print(f"[프로파일 {profile}] 오염 복구 (패치 {n_patches}개, seed={PATCH_SEED})")
    print(f"  복호 모드     : {mode} (소거 {erasures}개)")
    print(f"  soft 발동     : {'PASS' if soft_used else 'FAIL'}")
    print(f"  이미지        : {out_path}")
    print(f"  복구 결과     : {'PASS' if ok else 'FAIL'}")
    print()
    return ok and soft_used


if __name__ == "__main__":
    results = [clean_roundtrip(p) for p in ("A", "B")]
    results += [contaminated_recovery(p) for p in ("A", "B")]
    if all(results):
        print("=== exp11 전체 왕복 테스트 PASS ===")
    else:
        print("=== exp11 왕복 테스트 FAIL ===")
        sys.exit(1)
