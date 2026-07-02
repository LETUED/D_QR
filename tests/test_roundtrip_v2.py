"""
D_QR Phase 2 왕복 테스트 — 채널 독립 RS ECC (exp2)
실행: py -X utf8 test_roundtrip_v2.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exp2_channel_ecc.encoder_v2 import DQREncoderV2
from src.exp2_channel_ecc.decoder_v2 import DQRDecoderV2
from src.exp1_color.renderer import DQRRenderer


def hr(char="-", n=60):
    print(char * n)


def test(label: str, data: bytes, ecc_rate: float = 0.30, module_px: int = 20):
    hr("=")
    print(f"[exp2] 테스트: {label}")
    hr()

    enc = DQREncoderV2(ecc_rate=ecc_rate)
    ren = DQRRenderer(module_px=module_px)
    dec = DQRDecoderV2()

    result = enc.encode(data)
    cap = result['capacity']
    nsym_c1, nsym_c2, nsym_c3 = result['nsym']

    print(f"원본 데이터      : {len(data)} bytes")
    print(f"그리드 크기      : {result['grid_size']} x {result['grid_size']} 모듈")
    print(f"데이터 모듈      : {cap['data_modules']} 개")
    print(f"ECC (c1/c2/c3)  : nsym={nsym_c1}/{nsym_c2}/{nsym_c3}")
    print(f"최대 용량        : {cap['max_data_bytes']} bytes")
    print(f"bits/module      : {cap['bits_per_module']} (채널 독립 ECC)")

    img = ren.render(result)
    os.makedirs("outputs/exp2", exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "-")[:40]
    img_path = f"outputs/exp2/exp2_dqr_{safe_label}.png"
    img.save(img_path)
    print(f"이미지 저장      : {img_path}  ({img.width}x{img.height}px)")

    recovered = dec.decode(
        img,
        grid_size=result['grid_size'],
        module_px=module_px,
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
    print("Phase 1 vs Phase 2 용량 비교 (ECC 30%)")
    hr()

    from src.exp1_color.encoder import DQREncoder, _build_reserved_mask, BITS_PER_MODULE
    import numpy as np
    from src.core.constants import GRID_SIZES

    enc1 = DQREncoder(ecc_rate=0.30)
    enc2 = DQREncoderV2(ecc_rate=0.30)

    # Phase 2 ECC 분배 정보
    nsym1 = enc2.nsym_c1
    nsym2 = enc2.nsym_c2
    nsym3 = enc2.nsym_c3
    print(f"  Phase 2 ECC 분배: c1={nsym1}, c2={nsym2}, c3={nsym3} (합={nsym1+nsym2+nsym3})")
    print(f"  Phase 1 ECC:      nsym={enc1.nsym} (단일 블록)")
    hr("-", 60)
    print(f"{'크기':>6}  {'Phase1':>10}  {'Phase2':>10}  {'배율':>6}  {'vs QR-H':>8}")
    hr("-", 60)

    qr_capacity = {21: 7, 37: 14, 57: 24, 97: 62, 177: 287}

    for S in GRID_SIZES:
        p1 = enc1.max_data_bytes(S)
        p2 = enc2.max_data_bytes(S)
        ratio = f"{p2/p1:.2f}x"
        qr = qr_capacity.get(S, 1)
        vs_qr = f"{p2/qr:.1f}x"
        print(f"{S:>3}x{S:<3}  {p1:>8} B  {p2:>8} B  {ratio:>6}  {vs_qr:>8}")


if __name__ == "__main__":
    all_pass = True

    all_pass &= test(
        "영문_짧은_문장",
        b"Hello, D_QR Phase 2! Channel-independent ECC.",
    )
    all_pass &= test(
        "한국어_텍스트",
        "D_QR Phase 2 — 채널 독립 RS ECC 프로토타입.".encode("utf-8"),
    )
    all_pass &= test(
        "바이너리_데이터",
        bytes(range(256)),
    )
    all_pass &= test(
        "대용량_텍스트",
        ("D_QR Phase 2: 각 색상 채널이 독립적으로 오류를 정정한다. " * 10).encode("utf-8"),
    )

    hr("=")
    print(f"전체 결과: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    benchmark()
