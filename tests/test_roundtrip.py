"""
D_QR Phase 1 왕복 테스트 (encode → render → decode → 검증)
실행: py test_roundtrip.py
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exp1_color.encoder import DQREncoder
from src.exp1_color.renderer import DQRRenderer
from src.exp1_color.decoder import DQRDecoder


def hr(char="-", n=60):
    print(char * n)


def test(label: str, data: bytes, ecc_rate: float = 0.30, module_px: int = 20):
    hr("=")
    print(f"테스트: {label}")
    hr()

    enc = DQREncoder(ecc_rate=ecc_rate)
    ren = DQRRenderer(module_px=module_px)
    dec = DQRDecoder()

    # ── 인코딩 ──────────────────────────────────────────────────────
    result = enc.encode(data)
    cap = result['capacity']

    print(f"원본 데이터  : {len(data)} bytes")
    print(f"그리드 크기  : {result['grid_size']} × {result['grid_size']} 모듈")
    print(f"데이터 모듈  : {cap['data_modules']} 개")
    print(f"ECC 강도     : {ecc_rate*100:.0f}% ({result['nsym']} bytes)")
    print(f"최대 용량    : {cap['max_data_bytes']} bytes")
    print(f"bits/module  : {cap['bits_per_module']} (QR의 7배 이론)")

    # ── 렌더링 ──────────────────────────────────────────────────────
    img = ren.render(result)
    os.makedirs("outputs/exp1", exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "-")[:40]
    img_path = f"outputs/exp1/exp1_dqr_{safe_label}.png"
    img.save(img_path)
    print(f"이미지 저장  : {img_path}  ({img.width}×{img.height}px)")

    # ── 디코딩 ──────────────────────────────────────────────────────
    recovered = dec.decode(
        img,
        grid_size=result['grid_size'],
        module_px=module_px,
        nsym=result['nsym'],
        data_len=result['data_len'],
        codeword_len=result['codeword_len'],
    )

    # ── 검증 ────────────────────────────────────────────────────────
    ok = recovered == data
    print(f"왕복 검증    : {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  원본    : {data[:40]}")
        print(f"  복원    : {recovered[:40]}")
    print()
    return ok


def benchmark():
    """QR vs D_QR 이론 용량 비교."""
    hr("=")
    print("QR vs D_QR Phase 1 용량 비교 (ECC 30% 기준)")
    hr()

    # QR 공식 수치 (ECC-H 기준)
    qr_capacity = {21: 7, 37: 14, 57: 24, 97: 62, 177: 287}

    enc = DQREncoder(ecc_rate=0.30)
    from src.exp1_color.encoder import _build_reserved_mask, BITS_PER_MODULE
    import math, numpy as np
    from src.core.constants import GRID_SIZES

    print(f"{'크기':>6}  {'QR(ECC-H)':>12}  {'D_QR P1':>12}  {'배율':>6}")
    hr("-", 45)
    for S in GRID_SIZES:
        reserved = _build_reserved_mask(S)
        n_mod = int(np.sum(~reserved))
        total_bytes = (n_mod * BITS_PER_MODULE) // 8
        dqr_bytes = math.floor(total_bytes * 0.70)
        qr_bytes = qr_capacity.get(S, "?")
        ratio = f"{dqr_bytes / qr_bytes:.1f}×" if isinstance(qr_bytes, int) else "?"
        print(f"{S:>3}×{S:<3}  {str(qr_bytes)+' bytes':>12}  {str(dqr_bytes)+' bytes':>12}  {ratio:>6}")


if __name__ == "__main__":
    all_pass = True

    # ── 테스트 케이스 ────────────────────────────────────────────────
    all_pass &= test(
        "영문 짧은 문장",
        b"Hello, D_QR! This is Phase 1 prototype.",
    )

    all_pass &= test(
        "한국어 텍스트",
        "D_QR 프로토타입 — 2026년 5월 28일. 동환의 연구.".encode("utf-8"),
    )

    all_pass &= test(
        "바이너리 데이터",
        bytes(range(256)),
    )

    all_pass &= test(
        "대용량 텍스트",
        ("D_QR은 기존 QR 코드의 용량 한계를 다차원 텐서 인코딩으로 극복하는 새로운 코드 시스템이다. " * 8).encode("utf-8"),
    )

    hr("=")
    print(f"전체 결과: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    benchmark()
