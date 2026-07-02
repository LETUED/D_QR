"""
dqr_decode.py — D_QR 디코더 CLI (이미지 파일만으로 디코딩)
사용: py -X utf8 dqr_decode.py 이미지경로.png

PNG 메타데이터가 있으면 메타데이터 경로(구버전 호환), 없으면 exp13
자기기술 경로 — 기하 자기추정 + 차등 파인더 방향 판별 + 포맷 링.
  메타 exp9 → soft-decision 소거 복호 (decoder_v9)
  메타 exp6 → hard decision (decoder_v6, 구버전 호환)
  메타 없음 → exp13 자기기술 복호 (decoder_v13)
  메타 없음 + 자기추정 실패(규약 외 이미지: 임의 배치·회전·완만 원근)
            → exp16 검출 폴백 (src/exp16_detect — 기존 경로 보존)
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image


def decode_exp13(img, threshold: int = 128):
    """메타데이터 없는 이미지 → exp13 자기기술 복호."""
    from src.exp13_selfdesc.decoder_v13 import DQRDecoderV13

    dec = DQRDecoderV13(threshold=threshold)
    stats = dec.decode_with_stats(img)
    orient = stats['orientation']
    print(f"[자기추정 완료 — exp13]")
    print(f"  격자 반지름  : r={stats['radius']}")
    print(f"  hex_size     : {stats['hex_size']:.3f}px")
    print(f"  프로파일     : {stats['profile']} ({stats['bits_per_cell']}비트/셀)")
    print(f"  ECC 레벨     : {stats['ecc_level']} (nsym={stats['nsym']})")
    print(f"  방향         : 회전 {orient['rot_k'] * 120}°"
          f"{', 거울상' if orient['mirrored'] else ''}")
    print(f"[복호 통계] 모드={stats['mode']}, 소거={stats['erasures_used']}, "
          f"정정={stats['errata_count']}바이트")
    text = stats['data'].decode("utf-8")
    print(f"[디코딩 결과]")
    print(f"  {text}")
    return text


def decode_exp16(img, threshold: int = 128):
    """exp13 자기추정 실패(규약 외 이미지) → exp16 검출 파이프라인 폴백."""
    from src.exp16_detect import detect_and_decode

    stats = detect_and_decode(img, threshold=threshold)
    det = stats['detection']
    orient = stats['orientation']
    print(f"[검출 완료 — exp16]")
    print(f"  격자 반지름  : r={stats['radius']}")
    print(f"  hex_size     : {det['hex_size_img']:.3f}px (어파인 스케일 추정)")
    print(f"  거울상       : {'예' if det['mirrored'] else '아니오'}")
    print(f"  가설         : {det['n_hypotheses']}건 중 표 마진 "
          f"{det['score']}점 채택")
    print(f"  프로파일     : {stats['profile']} "
          f"({stats['bits_per_cell']}비트/셀)")
    print(f"  ECC 레벨     : {stats['ecc_level']} (nsym={stats['nsym']})")
    print(f"[복호 통계] 모드={stats['mode']}, 소거={stats['erasures_used']}, "
          f"정정={stats['errata_count']}바이트")
    text = stats['data'].decode("utf-8")
    print(f"[디코딩 결과]")
    print(f"  {text}")
    return text


def decode(image_path: str, threshold: int = 128):
    img = Image.open(image_path)
    meta = img.info  # PNG tEXt 메타데이터

    # ── 메타데이터 없으면 exp13 자기기술 경로 ───────────────────────
    required = ["dqr_version", "dqr_radius", "dqr_hex_size",
                "dqr_origin_x", "dqr_origin_y",
                "dqr_nsym", "dqr_data_len", "dqr_codeword_len"]

    missing = [k for k in required if k not in meta]
    if missing:
        print(f"[안내] PNG 메타데이터 없음 → exp13 자기기술 경로로 디코딩")
        try:
            return decode_exp13(img, threshold)
        except ValueError as e:
            # 자기추정 실패(렌더링 규약 외: 임의 배치·회전·원근)만 폴백.
            # ReedSolomonError 등 복호 실패는 그대로 전파 — 기존 경로 보존
            print(f"[안내] exp13 자기추정 실패({e}) → exp16 검출 경로 폴백")
            return decode_exp16(img, threshold)

    version      = meta["dqr_version"]
    radius       = int(meta["dqr_radius"])
    hex_size     = float(meta["dqr_hex_size"])
    origin_x     = float(meta["dqr_origin_x"])
    origin_y     = float(meta["dqr_origin_y"])
    nsym         = int(meta["dqr_nsym"])
    data_len     = int(meta["dqr_data_len"])
    codeword_len = int(meta["dqr_codeword_len"])

    print(f"[메타데이터 읽기 완료]")
    print(f"  버전         : {version}")
    print(f"  격자 반지름  : r={radius}")
    print(f"  hex_size     : {hex_size}px")
    print(f"  nsym         : {nsym}")
    print(f"  data_len     : {data_len} bytes")
    print(f"  이미지 크기  : {img.width} × {img.height} px")

    # ── 버전별 디코딩 ───────────────────────────────────────────────
    if version == "exp9":
        from src.exp9_soft.decoder_v9 import DQRDecoderV9
        profile = meta.get("dqr_profile", "A")
        bits_per_cell = int(meta.get("dqr_bits_per_cell", "6"))
        print(f"  프로파일     : {profile} ({bits_per_cell}비트/셀)")
        print()

        dec = DQRDecoderV9(threshold=threshold)
        stats = dec.decode_with_stats(
            img,
            radius=radius,
            hex_size=hex_size,
            bits_per_cell=bits_per_cell,
            origin_x=origin_x,
            origin_y=origin_y,
            nsym=nsym,
            data_len=data_len,
            codeword_len=codeword_len,
        )
        raw = stats['data']
        print(f"[복호 통계] 모드={stats['mode']}, 소거={stats['erasures_used']}, "
              f"정정={stats['errata_count']}바이트")
    else:
        from src.exp6_subtri.decoder_v6 import DQRDecoderV6
        print()
        dec = DQRDecoderV6(threshold=threshold)
        raw = dec.decode(
            img,
            radius=radius,
            hex_size=hex_size,
            origin_x=origin_x,
            origin_y=origin_y,
            nsym=nsym,
            data_len=data_len,
            codeword_len=codeword_len,
        )

    text = raw.decode("utf-8")
    print(f"[디코딩 결과]")
    print(f"  {text}")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D_QR 디코더 (이미지만으로 디코딩)")
    parser.add_argument("image", help="D_QR PNG 이미지 경로")
    parser.add_argument("--threshold", type=int, default=128,
                        help="밝기 임계값 (기본: 128)")
    args = parser.parse_args()

    decode(args.image, args.threshold)
