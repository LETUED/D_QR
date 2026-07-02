"""
exp10 왕복 테스트 — dual-grain 점진적 공개 검증
실행: py -X utf8 tests/test_exp10_roundtrip.py  (프로젝트 루트에서)

① 깨끗한 이미지 → coarse + fine 모두 원본 일치
② GaussianBlur 7.5px → coarse PASS + fine None (우아한 강등 실증)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import ImageFilter

from src.exp10_progressive.encoder_v10 import DQREncoderV10, GRAINS
from src.exp10_progressive.renderer_v10 import DQRRendererV10
from src.exp10_progressive.decoder_v10 import DQRDecoderV10

COARSE_TEXT = "https://dqr.kr/v10 — exp10 코어 페이로드"
FINE_TEXT = ("exp10 확장 페이로드: 신뢰도 게이트 점진적 공개 실증용 텍스트. "
             "고해상 촬영에서만 열리는 부가 정보(상세 설명·서명·메타데이터)를 "
             "fine 스트림에 싣는다.")

BLUR_PX = 7.5


def build_code():
    coarse = COARSE_TEXT.encode("utf-8")
    fine = FINE_TEXT.encode("utf-8")
    enc = DQREncoderV10()
    ren = DQRRendererV10(hex_size=32)
    result = enc.encode(coarse, fine)
    img, info = ren.render(result)
    params = dict(
        radius=result['radius'],
        hex_size=info['hex_size'],
        origin_x=info['origin_x'],
        origin_y=info['origin_y'],
        nsym_c=result['nsym_c'],
        nsym_f=result['nsym_f'],
        coarse_len=result['coarse_len'],
        fine_len=result['fine_len'],
        codeword_len_c=result['codeword_len_c'],
        codeword_len_f=result['codeword_len_f'],
    )
    return coarse, fine, img, params, result


def test_clean(coarse, fine, img, params, result):
    """① 깨끗한 이미지: 두 스트림 모두 원본 일치해야 함."""
    out_path = "outputs/exp10/exp10_roundtrip_clean.png"
    img.save(out_path)

    stats = DQRDecoderV10().decode_with_stats(img, **params)
    ok_c = stats['coarse'] == coarse
    ok_f = stats['fine'] == fine

    cap = result['capacity']
    print("[테스트 ①] 깨끗한 이미지 — coarse + fine 동시 복호")
    print(f"  격자           : r={result['radius']} "
          f"(coarse {cap['coarse_cells']}셀 / fine {cap['fine_cells']}셀)")
    print(f"  용량           : coarse {cap['max_coarse_bytes']}B (ECC 30%) / "
          f"fine {cap['max_fine_bytes']}B (ECC 20%)")
    print(f"  coarse         : {'PASS' if ok_c else 'FAIL'} "
          f"(mode={stats['coarse_mode']})")
    print(f"  fine           : {'PASS' if ok_f else 'FAIL'} "
          f"(mode={stats['fine_mode']}, conf={stats['fine_confidence']:.3f})")
    print(f"  이미지         : {out_path} ({img.width}x{img.height})")
    print()
    return ok_c and ok_f


def test_blur_degradation(coarse, fine, img, params):
    """② 블러 7.5px: coarse 생존 + fine은 None으로 우아하게 강등돼야 함."""
    blurred = img.filter(ImageFilter.GaussianBlur(BLUR_PX))
    out_path = f"outputs/exp10/exp10_roundtrip_blur{BLUR_PX}.png"
    blurred.save(out_path)

    stats = DQRDecoderV10().decode_with_stats(blurred, **params)
    ok_c = stats['coarse'] == coarse
    ok_f_none = stats['fine'] is None  # 예외 없이 강등돼야 점진적 공개 성립

    print(f"[테스트 ②] GaussianBlur {BLUR_PX}px — 점진적 공개(우아한 강등)")
    print(f"  coarse         : {'PASS (원본 일치)' if ok_c else 'FAIL'} "
          f"(mode={stats['coarse_mode']})")
    print(f"  fine           : {'PASS (None 강등)' if ok_f_none else 'FAIL (강등 안 됨)'} "
          f"(conf={stats['fine_confidence']:.3f})")
    print(f"  이미지         : {out_path}")
    print()
    return ok_c and ok_f_none


if __name__ == "__main__":
    Path("outputs/exp10").mkdir(parents=True, exist_ok=True)
    coarse, fine, img, params, result = build_code()

    results = [
        test_clean(coarse, fine, img, params, result),
        test_blur_degradation(coarse, fine, img, params),
    ]
    if all(results):
        print("=== exp10 전체 왕복 테스트 PASS ===")
    else:
        print("=== exp10 왕복 테스트 FAIL ===")
        sys.exit(1)
