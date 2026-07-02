"""
dqr_encode.py — D_QR 인코더 CLI (exp9 가변 프로파일 / exp13 자기기술)
사용: py -X utf8 dqr_encode.py "인코딩할 텍스트" [-o 출력경로.png] [--profile A|B] [--exp13]

  프로파일 A: 6비트/셀,  ECC 30% — 인쇄·실외 (인식률 우선, 기본)
  프로파일 B: 12비트/셀, ECC 20% — 화면·근접 (밀도 우선)

기본(exp9): 디코딩 파라미터를 PNG tEXt 메타데이터에 내장.
--exp13   : 포맷 링 + 차등 파인더로 파라미터를 이미지 자체에 내장 —
            PNG 메타데이터를 일절 쓰지 않는 자기기술(self-describing) 코드.
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import PngImagePlugin
from src.exp9_soft.encoder_v9 import DQREncoderV9, PROFILES
from src.exp9_soft.renderer_v9 import DQRRendererV9


def encode_exp13(text: str, output_path: str, profile: str = "A",
                 ecc_level: int = None, hex_size: int = 32):
    """exp13 자기기술 인코딩 — PNG 메타데이터 없이 이미지만으로 디코딩."""
    from src.exp13_selfdesc.encoder_v13 import DQREncoderV13
    from src.exp13_selfdesc.renderer_v13 import DQRRendererV13
    from src.exp13_selfdesc.format_ring import ECC_LEVELS

    data = text.encode("utf-8")
    enc = DQREncoderV13(profile=profile, ecc_level=ecc_level)
    result = enc.encode(data)
    img, _info = DQRRendererV13(hex_size=hex_size).render(result)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)   # 메타데이터 미기록 — 의도된 동작

    cap = result['capacity']
    print(f"[인코딩 완료 — exp13 자기기술]")
    print(f"  입력 텍스트  : {text}")
    print(f"  바이트 크기  : {len(data)} bytes")
    print(f"  프로파일     : {result['profile']} ({result['bits_per_cell']}비트/셀)")
    print(f"  격자 반지름  : r={result['radius']}  ({cap['total_cells']} 셀)")
    print(f"  ECC 레벨     : {result['ecc_level']} "
          f"({ECC_LEVELS[result['ecc_level']]*100:.0f}%, nsym={result['nsym']})")
    print(f"  이미지 크기  : {img.width} × {img.height} px")
    print(f"  저장 경로    : {output_path}")
    print(f"  (PNG 메타데이터 없음 — 포맷 링·차등 파인더가 파라미터를 내장)")


def encode(text: str, output_path: str, profile: str = "A",
           ecc_rate: float = None, hex_size: int = 32):
    data = text.encode("utf-8")

    enc = DQREncoderV9(profile=profile, ecc_rate=ecc_rate)
    ren = DQRRendererV9(hex_size=hex_size)

    result = enc.encode(data)
    img, render_info = ren.render(result)

    # ── 디코딩에 필요한 모든 파라미터를 PNG 메타데이터에 내장 ──────
    meta = PngImagePlugin.PngInfo()
    meta.add_text("dqr_version",       "exp9")
    meta.add_text("dqr_profile",       result['profile'])
    meta.add_text("dqr_bits_per_cell", str(result['bits_per_cell']))
    meta.add_text("dqr_radius",        str(result['radius']))
    meta.add_text("dqr_hex_size",      str(render_info['hex_size']))
    meta.add_text("dqr_origin_x",      str(render_info['origin_x']))
    meta.add_text("dqr_origin_y",      str(render_info['origin_y']))
    meta.add_text("dqr_nsym",          str(result['nsym']))
    meta.add_text("dqr_data_len",      str(result['data_len']))
    meta.add_text("dqr_codeword_len",  str(result['codeword_len']))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path, pnginfo=meta)

    cap = result['capacity']
    print(f"[인코딩 완료]")
    print(f"  입력 텍스트  : {text}")
    print(f"  바이트 크기  : {len(data)} bytes")
    print(f"  프로파일     : {result['profile']} ({PROFILES[result['profile']]['desc']})")
    print(f"  비트/셀      : {result['bits_per_cell']}")
    print(f"  격자 반지름  : r={result['radius']}  ({cap['total_cells']} 셀)")
    print(f"  ECC          : {cap['ecc_rate']*100:.0f}%  (nsym={result['nsym']})")
    print(f"  이미지 크기  : {img.width} × {img.height} px")
    print(f"  저장 경로    : {output_path}")
    print(f"  (PNG 메타데이터에 디코딩 파라미터 내장 완료)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D_QR 인코더 (exp9)")
    parser.add_argument("text", help="인코딩할 텍스트")
    parser.add_argument("-o", "--output", default="outputs/dqr_encoded.png",
                        help="출력 PNG 경로 (기본: outputs/dqr_encoded.png)")
    parser.add_argument("--profile", choices=list(PROFILES), default="A",
                        help="A=인쇄·실외(인식률), B=화면·근접(밀도) (기본: A)")
    parser.add_argument("--ecc", type=float, default=None,
                        help="ECC rate 수동 지정 (기본: 프로파일 기본값, exp9 전용)")
    parser.add_argument("--hex-size", type=int, default=32, help="헥사곤 크기 px (기본: 32)")
    parser.add_argument("--exp13", action="store_true",
                        help="자기기술 코드 — PNG 메타데이터 없이 이미지만으로 디코딩")
    parser.add_argument("--ecc-level", type=int, default=None, choices=(0, 1, 2, 3),
                        help="exp13 ECC 레벨 0~3 = 10/20/30/40%% (기본: 프로파일 기본)")
    args = parser.parse_args()

    if args.exp13:
        if args.ecc is not None:
            parser.error("--exp13에서는 --ecc 대신 --ecc-level(0~3)을 사용하세요 "
                         "(포맷 링 2비트로 인코딩되는 이산 레벨)")
        encode_exp13(args.text, args.output, args.profile,
                     args.ecc_level, args.hex_size)
    else:
        if args.ecc_level is not None:
            parser.error("--ecc-level은 --exp13 전용입니다")
        encode(args.text, args.output, args.profile, args.ecc, args.hex_size)
