"""
D_QR Phase 5 왕복 테스트 — 3D 부조 인코딩 (exp5)
실행: py -X utf8 test_roundtrip_v5.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exp5_zdepth.encoder_v5 import DQREncoderV5, GRID_RADII
from src.exp5_zdepth.renderer_v5 import DQRRendererV5
from src.exp5_zdepth.decoder_v5 import DQRDecoderV5


def hr(char="-", n=60):
    print(char * n)


def test(label: str, data: bytes, ecc_rate: float = 0.20, hex_size: int = 28):
    hr("=")
    print(f"[exp5] 테스트: {label}")
    hr()

    enc = DQREncoderV5(ecc_rate=ecc_rate)
    ren = DQRRendererV5(hex_size=hex_size)
    dec = DQRDecoderV5()

    result = enc.encode(data)
    cap = result['capacity']

    print(f"원본 데이터      : {len(data)} bytes")
    print(f"격자 반지름      : {result['radius']}  ({cap['total_cells']} 셀)")
    print(f"데이터 셀        : {cap['data_cells']} 개")
    print(f"bits/cell        : {cap['bits_per_cell']}  "
          f"(삼각형 {cap['bits_triangle']}비트 + z-depth {cap['bits_height']}비트)")
    print(f"ECC rate         : {ecc_rate*100:.0f}%  (nsym={result['nsym']})")
    print(f"최대 용량        : {cap['max_data_bytes']} bytes")

    img_2d, img_3d, render_info = ren.render(result)
    os.makedirs("outputs/exp5", exist_ok=True)
    safe = label.replace(" ", "_").replace("/", "-")[:40]

    path_2d = f"outputs/exp5/exp5_dqr_{safe}_flat.png"
    img_2d.save(path_2d)
    print(f"2D 저장 (top-down): {path_2d}  ({img_2d.width}x{img_2d.height}px)")

    if img_3d is not None:
        path_3d = f"outputs/exp5/exp5_dqr_{safe}_3d.png"
        img_3d.save(path_3d)
        print(f"3D 저장 (matplotlib): {path_3d}")
    else:
        print("3D 렌더링: matplotlib 없음, 생략")

    recovered = dec.decode(
        img_2d,
        heights=result['heights'],   # 프로토타입: 시뮬레이션 높이
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
    print("exp4(6비트,20%) vs exp5(8비트,20%) 용량 비교")
    hr()

    from src.exp4_geometric.encoder_v4 import DQREncoderV4

    enc4 = DQREncoderV4(ecc_rate=0.20)
    enc5 = DQREncoderV5(ecc_rate=0.20)

    print(f"{'반지름':>6}  {'exp4 6bit':>12}  {'exp5 8bit':>12}  {'증가율':>8}")
    hr("-", 45)
    for r in [8, 12, 18, 25, 35]:
        p4 = enc4.max_data_bytes(r)
        p5 = enc5.max_data_bytes(r)
        ratio = f"+{(p5/p4-1)*100:.0f}%"
        print(f"  r={r:<2}  {str(p4)+' B':>12}  {str(p5)+' B':>12}  {ratio:>8}")

    hr("-", 45)
    print()
    print("  Z-depth 2비트 추가로 용량 약 30~65% 증가")
    print("  z-height는 3D프린터/CNC로 제조, 구조광/LiDAR로 스캔 디코딩")
    print()
    print("  다음 연구 방향:")
    print("    → 3방향 투영 코드: 1 오브젝트 × 3 카메라 = 3× 용량")
    print("    → 시간 차원: 30fps 동적 코드 = 30× 레이어")


if __name__ == "__main__":
    all_pass = True

    all_pass &= test("영문_짧은_문장",
                     b"Hello, 3D D_QR! Relief encoding test.")
    all_pass &= test("한국어_텍스트",
                     "D_QR Phase 5 — Z축 부조 인코딩, 8비트/셀.".encode("utf-8"))
    all_pass &= test("바이너리_데이터",
                     bytes(range(256)))
    all_pass &= test("대용량_텍스트",
                     ("3D 공간의 Z축이 추가 데이터 차원이 된다. 부조 높이 = 2비트/셀. " * 8).encode("utf-8"))

    hr("=")
    print(f"전체 결과: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    benchmark()
