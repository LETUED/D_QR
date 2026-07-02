"""
D_QR Phase 7(프랙탈) + Phase 8(DLC) + MultiDimData 왕복 테스트
실행: py -X utf8 test_roundtrip_v7v8.py
"""
import sys
import os
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exp7_fractal.encoder_v7 import DQREncoderV7
from src.exp7_fractal.renderer_v7 import DQRRendererV7
from src.exp7_fractal.decoder_v7 import DQRDecoderV7

from src.exp8_dlc.encoder_v8 import DQREncoderV8
from src.exp8_dlc.renderer_v8 import DQRRendererV8
from src.exp8_dlc.decoder_v8 import DQRDecoderV8

from src.core.multidim import MultiDimData


def hr(char="-", n=60):
    print(char * n)


# ── exp7: 프랙탈 중첩 헥사곤 ─────────────────────────────────────────

def test_v7(label: str, data: bytes):
    hr("=")
    print(f"[exp7/프랙탈] 테스트: {label}")
    hr()

    enc = DQREncoderV7(outer_radius=8, ecc_rate=0.20)
    ren = DQRRendererV7()
    dec = DQRDecoderV7()

    result = enc.encode(data)
    cap = result['capacity']

    print(f"원본 데이터        : {len(data)} bytes")
    print(f"외부 격자 반지름   : {cap['outer_radius']}  ({cap['outer_cells']} 데이터 셀)")
    print(f"내부 셀/외부 셀    : {cap['inner_cells_per_outer']}  → 총 {cap['total_inner_cells']} 내부 셀")
    print(f"bits/외부셀        : {cap['bits_per_outer']}  (7 × 6)")
    print(f"프랙탈 레벨        : {cap['fractal_levels']}")
    print(f"최대 용량          : {cap['max_data_bytes']} bytes")

    img, info = ren.render(result)
    os.makedirs("outputs/exp7", exist_ok=True)
    os.makedirs("outputs/exp8", exist_ok=True)
    safe = label.replace(" ", "_").replace("/", "-")[:40]
    path = f"outputs/exp7/exp7_dqr_{safe}.png"
    img.save(path)
    print(f"이미지 저장        : {path}  ({img.width}x{img.height}px)")

    recovered = dec.decode(
        img,
        outer_radius=result['outer_radius'],
        outer_hex_size=info['outer_hex_size'],
        inner_hex_size=info['inner_hex_size'],
        origin_x=info['origin_x'],
        origin_y=info['origin_y'],
        nsym=result['nsym'],
        data_len=result['data_len'],
        codeword_len=result['codeword_len'],
    )

    ok = recovered == data
    print(f"왕복 검증          : {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  원본  : {data[:40]}")
        print(f"  복원  : {recovered[:40]}")
    print()
    return ok


# ── exp8: DLC 듀얼 스트림 ────────────────────────────────────────────

def test_v8(label: str, data_a: bytes, data_b: bytes):
    hr("=")
    print(f"[exp8/DLC] 테스트: {label}")
    hr()

    enc = DQREncoderV8(ecc_rate=0.20)
    ren = DQRRendererV8(hex_size=32)
    dec = DQRDecoderV8()

    result = enc.encode(data_a, data_b)
    cap = result['capacity']

    print(f"Stream A           : {len(data_a)} bytes  → {cap['stream_a_cells']} 셀")
    print(f"Stream B           : {len(data_b)} bytes  → {cap['stream_b_cells']} 셀")
    print(f"격자 반지름        : {cap['radius']}")
    print(f"스트림/이미지      : {cap['independent_streams']} 개 독립 채널")
    print(f"스트림당 최대 용량 : {cap['max_bytes_per_stream']} bytes")

    img, info = ren.render(result)
    os.makedirs("outputs/exp7", exist_ok=True)
    os.makedirs("outputs/exp8", exist_ok=True)
    safe = label.replace(" ", "_").replace("/", "-")[:40]
    path = f"outputs/exp8/exp8_dqr_{safe}.png"
    img.save(path)
    print(f"이미지 저장        : {path}  ({img.width}x{img.height}px)")

    rec_a, rec_b = dec.decode_both(img, result, info)

    ok_a = rec_a == data_a
    ok_b = rec_b == data_b
    print(f"Stream A 검증      : {'PASS' if ok_a else 'FAIL'}")
    print(f"Stream B 검증      : {'PASS' if ok_b else 'FAIL'}")
    if not ok_a:
        print(f"  A 원본: {data_a[:30]}  복원: {rec_a[:30]}")
    if not ok_b:
        print(f"  B 원본: {data_b[:30]}  복원: {rec_b[:30]}")
    print()
    return ok_a and ok_b


# ── MultiDimData 테스트 ──────────────────────────────────────────────

def test_multidim():
    hr("=")
    print("[MultiDimData] 다차원 정보 구조체 테스트")
    hr()

    # 3차원 정보: 텍스트 + 메타데이터 + 해시 증명
    text    = "D_QR로 인코딩된 다차원 정보입니다.".encode("utf-8")
    meta    = b"author:donghwan,date:2026-05-29,version:exp8"
    proof   = hashlib.sha256(text).digest()  # 32 bytes

    mdd = MultiDimData(text=text, meta=meta, proof=proof)
    print(mdd.summary())
    print(f"직렬화 크기        : {len(mdd.to_bytes())} bytes")

    # 역직렬화 검증
    mdd2 = MultiDimData.from_bytes(mdd.to_bytes())
    assert mdd2['text'] == text
    assert mdd2['meta'] == meta
    assert mdd2['proof'] == proof
    print("직렬화/역직렬화    : PASS")

    # exp8 듀얼 스트림으로 인코딩
    a_bytes, b_bytes = mdd.split_two()
    print(f"Stream A ({mdd.dim_names()[:len(mdd.dim_names())//2]}): {len(a_bytes)} bytes")
    print(f"Stream B ({mdd.dim_names()[len(mdd.dim_names())//2:]}): {len(b_bytes)} bytes")

    enc = DQREncoderV8(ecc_rate=0.20)
    ren = DQRRendererV8(hex_size=32)
    dec = DQRDecoderV8()

    result = enc.encode(a_bytes, b_bytes)
    img, info = ren.render(result)
    os.makedirs("outputs/exp8", exist_ok=True)
    img.save("outputs/exp8/exp8_dqr_multidim.png")
    print(f"이미지 저장        : outputs/exp8/exp8_dqr_multidim.png  ({img.width}x{img.height}px)")

    rec_a, rec_b = dec.decode_both(img, result, info)
    mdd_recovered = MultiDimData.from_two(rec_a, rec_b)

    ok = (mdd_recovered['text'] == text and
          mdd_recovered['meta'] == meta and
          mdd_recovered['proof'] == proof)
    print(f"MultiDimData 왕복  : {'PASS' if ok else 'FAIL'}")
    print()
    return ok


# ── 전체 벤치마크 ────────────────────────────────────────────────────

def benchmark():
    hr("=")
    print("exp4~exp8 용량 비교 (ECC 20%, r=8 기준)")
    hr()

    from src.exp4_geometric.encoder_v4 import DQREncoderV4
    from src.exp5_zdepth.encoder_v5 import DQREncoderV5
    from src.exp6_subtri.encoder_v6 import DQREncoderV6

    enc4 = DQREncoderV4(ecc_rate=0.20)
    enc5 = DQREncoderV5(ecc_rate=0.20)
    enc6 = DQREncoderV6(ecc_rate=0.20)
    enc7 = DQREncoderV7(outer_radius=8, ecc_rate=0.20)
    enc8 = DQREncoderV8(ecc_rate=0.20)

    print(f"{'실험':>6}  {'방식':>22}  {'r=8 용량':>12}  {'특징'}")
    hr("-", 75)
    print(f"  exp4   {'6비트/셀 삼각형':>22}  {str(enc4.max_data_bytes(8))+' B':>12}  단색 기하")
    print(f"  exp5   {'8비트/셀 (삼각+Z)':>22}  {str(enc5.max_data_bytes(8))+' B':>12}  3D 부조 채널")
    print(f"  exp6   {'12비트/셀 서브삼각':>22}  {str(enc6.max_data_bytes(8))+' B':>12}  수학적 밀도 증가")
    print(f"  exp7   {'42비트/외부셀 프랙탈':>22}  {str(enc7.max_data_bytes())+' B':>12}  2레벨 자기유사")
    p8 = enc8.max_data_bytes_per_stream(8) * 2
    print(f"  exp8   {'12비트/셀 × 2스트림':>22}  {str(p8)+' B':>12}  차원 링크 체인")
    hr("-", 75)
    print()
    print("  exp7: exp6 대비 3.5× 더 많은 용량 (동일 면적, 단 더 큰 셀)")
    print("  exp8: 동일 용량이지만 2개의 독립 데이터셋 → 차원 분리")


if __name__ == "__main__":
    all_pass = True

    # exp7 테스트
    all_pass &= test_v7("짧은_문장",
                        b"Hello, Fractal D_QR! 2-level nested hex.")
    all_pass &= test_v7("바이너리",
                        bytes(range(256)))
    all_pass &= test_v7("대용량",
                        ("프랙탈 중첩 헥사곤: 동일 스케일에서 2레벨 자기유사 구조. " * 8).encode("utf-8"))

    # exp8 테스트
    all_pass &= test_v8(
        "공개_비공개",
        b"PUBLIC: This is the main payload. Anyone can read this.",
        b"PRIVATE: Hidden dimension. Access requires DLC decoder.",
    )
    all_pass &= test_v8(
        "바이너리_듀얼",
        bytes(range(128)),
        bytes(range(128, 256)),
    )

    # MultiDimData 테스트
    all_pass &= test_multidim()

    hr("=")
    print(f"전체 결과: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    benchmark()
