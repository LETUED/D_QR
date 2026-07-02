"""
D_QR Phase 6 왕복 테스트 — 12-서브삼각형 이진 (exp6)
실행: py -X utf8 test_roundtrip_v6.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exp6_subtri.encoder_v6 import DQREncoderV6
from src.exp6_subtri.renderer_v6 import DQRRendererV6
from src.exp6_subtri.decoder_v6 import DQRDecoderV6


def hr(char="-", n=60):
    print(char * n)


def test(label: str, data: bytes, ecc_rate: float = 0.20, hex_size: int = 32):
    hr("=")
    print(f"[exp6] 테스트: {label}")
    hr()

    enc = DQREncoderV6(ecc_rate=ecc_rate)
    ren = DQRRendererV6(hex_size=hex_size)
    dec = DQRDecoderV6()

    result = enc.encode(data)
    cap = result['capacity']

    print(f"원본 데이터      : {len(data)} bytes")
    print(f"격자 반지름      : {result['radius']}  ({cap['total_cells']} 셀)")
    print(f"데이터 셀        : {cap['data_cells']} 개")
    print(f"bits/cell        : {cap['bits_per_cell']}  ({cap['sub_triangles']} 서브삼각형)")
    print(f"ECC rate         : {ecc_rate*100:.0f}%  (nsym={result['nsym']})")
    print(f"최대 용량        : {cap['max_data_bytes']} bytes")

    img, render_info = ren.render(result)
    os.makedirs("outputs/exp6", exist_ok=True)
    safe = label.replace(" ", "_").replace("/", "-")[:40]
    img_path = f"outputs/exp6/exp6_dqr_{safe}.png"
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
        codeword_len=result['codeword_len'],
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
    print("exp4(6비트) vs exp6(12비트) 용량 비교 (ECC 20%)")
    hr()

    from src.exp4_geometric.encoder_v4 import DQREncoderV4
    from src.exp5_zdepth.encoder_v5 import DQREncoderV5

    enc4 = DQREncoderV4(ecc_rate=0.20)
    enc5 = DQREncoderV5(ecc_rate=0.20)
    enc6 = DQREncoderV6(ecc_rate=0.20)
    qr   = {8: 7, 12: 14, 18: 24, 25: 62, 35: 287}

    print(f"{'반지름':>6}  {'QR':>8}  {'exp4 6bit':>12}  {'exp5 8bit':>12}  {'exp6 12bit':>12}")
    hr("-", 58)
    for r in [8, 12, 18, 25, 35]:
        q = qr.get(r, "?")
        p4 = enc4.max_data_bytes(r)
        p5 = enc5.max_data_bytes(r)
        p6 = enc6.max_data_bytes(r)
        print(f"  r={r:<2}  {str(q)+' B':>8}  {str(p4)+' B':>12}  "
              f"{str(p5)+' B':>12}  {str(p6)+' B':>12}")
    hr("-", 58)
    print()
    print("  exp6 = exp4의 2배 (동일 면적, 동일 ECC rate)")
    print("  차원 링크 체인: exp6은 2D 수학적 밀도 증가 방향의 첫 단계")


if __name__ == "__main__":
    all_pass = True

    all_pass &= test("영문_짧은_문장",
                     b"Hello, 12-bit sub-triangle D_QR! Phase 6.")
    all_pass &= test("한국어_텍스트",
                     "D_QR Phase 6 — 12 서브삼각형, 12비트/셀.".encode("utf-8"))
    all_pass &= test("바이너리_데이터",
                     bytes(range(256)))
    all_pass &= test("대용량_텍스트",
                     ("수학적 다차원 밀도 극대화: 12비트/셀로 exp4 대비 2배 용량. " * 12).encode("utf-8"))

    hr("=")
    print(f"전체 결과: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    benchmark()
