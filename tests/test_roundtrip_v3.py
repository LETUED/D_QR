"""
D_QR Phase 3 왕복 테스트 — 육각형 격자 (exp3)
실행: py -X utf8 test_roundtrip_v3.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exp3_hexgrid.encoder_v3 import DQREncoderV3, GRID_RADII
from src.exp3_hexgrid.renderer_v3 import DQRRendererV3
from src.exp3_hexgrid.decoder_v3 import DQRDecoderV3


def hr(char="-", n=60):
    print(char * n)


def test(label: str, data: bytes, ecc_rate: float = 0.30, hex_size: int = 18):
    hr("=")
    print(f"[exp3] 테스트: {label}")
    hr()

    enc = DQREncoderV3(ecc_rate=ecc_rate)
    ren = DQRRendererV3(hex_size=hex_size)
    dec = DQRDecoderV3()

    result = enc.encode(data)
    cap = result['capacity']
    nsym_c1, nsym_c2, nsym_c3 = result['nsym']

    print(f"원본 데이터      : {len(data)} bytes")
    print(f"격자 반지름      : {result['radius']}  ({cap['total_cells']} 셀)")
    print(f"데이터 셀        : {cap['data_cells']} 개  (예약 {cap['reserved_cells']} 셀)")
    print(f"ECC (c1/c2/c3)  : nsym={nsym_c1}/{nsym_c2}/{nsym_c3}")
    print(f"최대 용량        : {cap['max_data_bytes']} bytes")

    img, render_info = ren.render(result)
    os.makedirs("outputs/exp3", exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "-")[:40]
    img_path = f"outputs/exp3/exp3_dqr_{safe_label}.png"
    img.save(img_path)
    print(f"이미지 저장      : {img_path}  ({img.width}x{img.height}px)")

    recovered = dec.decode(
        img,
        radius=result['radius'],
        hex_size=render_info['hex_size'],
        origin_x=render_info['origin_x'],
        origin_y=render_info['origin_y'],
        nsym=result['nsym'],
        data_len=result['data_len'],
        codeword_lens=result['codeword_lens'],
        channel_data_bytes=result['channel_data_bytes'],
    )

    ok = recovered == data
    print(f"왕복 검증        : {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  원본   : {data[:40]}")
        print(f"  복원   : {recovered[:40]}")
    print()
    return ok


def benchmark():
    hr("=")
    print("QR / exp1 / exp2 / exp3 용량 비교 (ECC 30%)")
    hr()

    from src.exp1_color.encoder import DQREncoder
    from src.exp2_channel_ecc.encoder_v2 import DQREncoderV2

    enc1 = DQREncoder(ecc_rate=0.30)
    enc2 = DQREncoderV2(ecc_rate=0.30)
    enc3 = DQREncoderV3(ecc_rate=0.30)
    qr = {21: 7, 37: 14, 57: 24, 97: 62, 177: 287}

    # 비슷한 크기의 격자끼리 비교
    print(f"{'격자':>12}  {'QR(ECC-H)':>10}  {'exp1(□)':>10}  {'exp2(□)':>10}  {'exp3(⬡)':>10}")
    hr("-", 65)

    sq_sizes = [21, 37, 57, 97, 177]
    hex_radii = [8, 12, 18, 25, 35]

    for sq, hr_ in zip(sq_sizes, hex_radii):
        qr_b = qr.get(sq, "?")
        p1 = enc1.max_data_bytes(sq)
        p2 = enc2.max_data_bytes(sq)
        p3 = enc3.max_data_bytes(hr_)
        print(f"  sq{sq:>3}/r{hr_:<2}  {str(qr_b)+' B':>10}  {str(p1)+' B':>10}  "
              f"{str(p2)+' B':>10}  {str(p3)+' B':>10}")

    hr("-", 65)
    print()
    print("  exp3 셀 수 (3R²+3R+1 - 21):")
    for r in hex_radii:
        n_cells = 3*r*r + 3*r + 1 - 21
        print(f"    radius={r:>2}: {n_cells:>5} 데이터 셀  →  {enc3.max_data_bytes(r):>6} bytes")


if __name__ == "__main__":
    all_pass = True

    all_pass &= test("영문_짧은_문장",
                     b"Hello, Hex D_QR! Phase 3 geometric code.")
    all_pass &= test("한국어_텍스트",
                     "D_QR Phase 3 — 육각형 격자 기하 코드.".encode("utf-8"))
    all_pass &= test("바이너리_데이터",
                     bytes(range(256)))
    all_pass &= test("대용량_텍스트",
                     ("D_QR Phase 3: 육각형 메시는 3축 좌표계로 위치 정보가 추가 차원이 된다. " * 10).encode("utf-8"))

    hr("=")
    print(f"전체 결과: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    benchmark()
