"""
exp9 왕복 테스트 — 프로파일 A/B 인코딩 → 렌더 → soft 디코딩 → 원본 일치 확인
실행: py -X utf8 test_exp9_roundtrip.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exp9_soft.encoder_v9 import DQREncoderV9, PROFILES
from src.exp9_soft.renderer_v9 import DQRRendererV9
from src.exp9_soft.decoder_v9 import DQRDecoderV9

TEXT = "dqr 시험테스트입니다 — exp9 가변 프로파일 + soft-decision 검증"


def roundtrip(profile: str):
    data = TEXT.encode("utf-8")
    enc = DQREncoderV9(profile=profile)
    ren = DQRRendererV9(hex_size=32)
    dec = DQRDecoderV9()

    result = enc.encode(data)
    img, info = ren.render(result)

    Path("outputs/exp9").mkdir(parents=True, exist_ok=True)
    out_path = f"outputs/exp9/exp9_roundtrip_profile{profile}.png"
    img.save(out_path)

    stats = dec.decode_with_stats(
        img,
        radius=result['radius'],
        hex_size=info['hex_size'],
        bits_per_cell=result['bits_per_cell'],
        origin_x=info['origin_x'],
        origin_y=info['origin_y'],
        nsym=result['nsym'],
        data_len=result['data_len'],
        codeword_len=result['codeword_len'],
    )

    ok = stats['data'] == data
    cap = result['capacity']
    print(f"[프로파일 {profile}] {PROFILES[profile]['desc']}")
    print(f"  비트/셀      : {result['bits_per_cell']}")
    print(f"  ECC          : {cap['ecc_rate']*100:.0f}% (nsym={result['nsym']})")
    print(f"  격자         : r={result['radius']} ({cap['data_cells']} 데이터 셀)")
    print(f"  최대 용량    : {cap['max_data_bytes']} bytes")
    print(f"  복호 모드    : {stats['mode']} (소거 {stats['erasures_used']}개)")
    print(f"  이미지       : {out_path} ({img.width}x{img.height})")
    print(f"  왕복 결과    : {'PASS' if ok else 'FAIL'}")
    print()
    return ok


if __name__ == "__main__":
    results = [roundtrip(p) for p in ("A", "B")]
    if all(results):
        print("=== exp9 전체 왕복 테스트 PASS ===")
    else:
        print("=== exp9 왕복 테스트 FAIL ===")
        sys.exit(1)
