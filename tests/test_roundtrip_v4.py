"""
D_QR Phase 4 왕복 테스트 — 기하 서브삼각형 단색 (exp4)
실행: py -X utf8 test_roundtrip_v4.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exp4_geometric.encoder_v4 import DQREncoderV4, GRID_RADII
from src.exp4_geometric.renderer_v4 import DQRRendererV4
from src.exp4_geometric.decoder_v4 import DQRDecoderV4


def hr(char="-", n=60):
    print(char * n)


def test(label: str, data: bytes,
         ecc_rate: float = 0.20,
         hex_size: int = 28,
         fg=(0, 0, 0), bg=(255, 255, 255)):
    hr("=")
    print(f"[exp4] 테스트: {label}")
    hr()

    enc = DQREncoderV4(ecc_rate=ecc_rate)
    ren = DQRRendererV4(hex_size=hex_size, fg_color=fg, bg_color=bg)
    # fg/bg 밝기 중간값으로 threshold 자동 설정
    fg_b = (fg[0] + fg[1] + fg[2]) // 3
    bg_b = (bg[0] + bg[1] + bg[2]) // 3
    threshold = (fg_b + bg_b) // 2 or 1  # 0 방지
    invert = fg_b > bg_b  # fg가 bg보다 밝으면 반전
    dec = DQRDecoderV4(threshold=threshold, invert=invert)

    result = enc.encode(data)
    cap = result['capacity']

    print(f"원본 데이터      : {len(data)} bytes")
    print(f"격자 반지름      : {result['radius']}  ({cap['total_cells']} 셀)")
    print(f"데이터 셀        : {cap['data_cells']} 개")
    print(f"bits/cell        : {cap['bits_per_cell']} (기하 삼각형)")
    print(f"ECC rate         : {ecc_rate*100:.0f}%  (nsym={result['nsym']})")
    print(f"최대 용량        : {cap['max_data_bytes']} bytes")
    print(f"색상 의존        : {cap['color_dependent']}")

    img, render_info = ren.render(result)
    os.makedirs("outputs/exp4", exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "-")[:40]
    img_path = f"outputs/exp4/exp4_dqr_{safe_label}.png"
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
    print("exp3(색상,30%) vs exp4(기하,20%) 비교")
    hr()

    from src.exp3_hexgrid.encoder_v3 import DQREncoderV3

    enc3 = DQREncoderV3(ecc_rate=0.30)
    enc4_20 = DQREncoderV4(ecc_rate=0.20)
    enc4_30 = DQREncoderV4(ecc_rate=0.30)

    qr = {8: 7, 12: 14, 18: 24, 25: 62, 35: 287}

    print(f"{'반지름':>6}  {'QR':>8}  {'exp3 색상':>10}  {'exp4 기하20%':>12}  {'exp4 기하30%':>12}")
    hr("-", 60)
    for r in [8, 12, 18, 25, 35]:
        q = qr.get(r, "?")
        p3 = enc3.max_data_bytes(r)
        p4_20 = enc4_20.max_data_bytes(r)
        p4_30 = enc4_30.max_data_bytes(r)
        print(f"  r={r:<2}  {str(q)+' B':>8}  {str(p3)+' B':>10}  "
              f"{str(p4_20)+' B':>12}  {str(p4_30)+' B':>12}")

    hr("-", 60)
    print()
    print("  핵심 차이: exp4는 조명/색상 무관 — BER 대폭 감소")
    print("  동일 신뢰도 기준 시: ECC rate 낮출 수 있어 실질 용량 경쟁력 있음")
    print()
    print("  Phase 4 후속 방향:")
    print("    → g₁: 셀 내 삼각형 방향 추가 (12방향 = +3.58비트)")
    print("    → 3축 RS ECC: q/r/s 방향 독립 적용")
    print("    → 카메라 디코더: 파인더 3점으로 원근 보정")


if __name__ == "__main__":
    all_pass = True

    # 기본 흑백 테스트
    all_pass &= test("영문_짧은_문장",
                     b"Hello, Geometric D_QR! No color channels.")
    all_pass &= test("한국어_텍스트",
                     "D_QR Phase 4 — 순수 기하 인코딩, 색상 독립.".encode("utf-8"))
    all_pass &= test("바이너리_데이터",
                     bytes(range(256)))
    all_pass &= test("대용량_텍스트",
                     ("기하학적 삼각형 서브패턴으로 인코딩된 D_QR. 홀로그램에도 동일 적용. " * 8).encode("utf-8"))

    # 홀로그램 색상 시뮬레이션 (fg=녹색, bg=검정)
    all_pass &= test("홀로그램_시뮬레이션",
                     b"Hologram D_QR: green on black. Same encoding.",
                     fg=(0, 220, 80), bg=(0, 0, 0))

    hr("=")
    print(f"전체 결과: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    benchmark()
