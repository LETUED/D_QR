"""
D_QR 연구 진화 갤러리 빌더 — exp1 ~ exp20
(exp12 3층 방어는 디코더 내부 계층이라 별도 카드 없음 — exp11b에 통합)
원본 데이터 → 인코딩된 D_QR 이미지 → 디코딩 결과를 exp별 카드로 비교.

실행: py -X utf8 scripts/build_gallery.py   (프로젝트 루트에서)
산출: outputs/gallery/index.html + expN_*.png
"""
import sys
import io
import html
import base64
import datetime
import shutil
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

GALLERY_DIR = Path("outputs/gallery")

# 표준 페이로드 — 모든 exp에 동일하게 투입해 진화 비교의 기준선으로 삼는다
STANDARD_TEXT = ("D_QR 진화 데모 — 한국어 페이로드 시험 | "
                 "단일색 기하 코드의 연구 진화 갤러리 (2026-06-12)")
STANDARD_BYTES = STANDARD_TEXT.encode("utf-8")

# exp10 fine 스트림용 확장 텍스트 (고해상 전용 부가 정보 데모)
FINE_TEXT = ("exp10 fine 확장 페이로드: 신뢰도 게이트(FINE_GATE_THRESHOLD=0.405)를 "
             "통과해야 열리는 부가 정보. coarse는 블러 10px 생존, fine은 6px에서 붕괴.")
FINE_BYTES = FINE_TEXT.encode("utf-8")


def make_demo_png() -> bytes:
    """16×16 컬러 데모 이미지 — 바이너리 페이로드 왕복 검증용 (약 90 bytes)."""
    img = Image.new("RGB", (16, 16))
    px = img.load()
    for y in range(16):
        for x in range(16):
            # 대각선 띠가 있는 그라데이션: 픽셀 단위 왕복 오류가 눈에 띄도록
            if (x + y) % 8 < 2:
                px[x, y] = (255, 255, 255)
            else:
                px[x, y] = (x * 16 % 256, y * 16 % 256, (x * y) % 256)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


DEMO_PNG = make_demo_png()


def png_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def text_demo(label, original_bytes, code_path, recovered, extra=""):
    """텍스트 페이로드 데모 결과 레코드."""
    ok = recovered == original_bytes
    return {
        "kind": "text", "label": label, "ok": ok,
        "original_text": original_bytes.decode("utf-8", errors="replace"),
        "original_size": len(original_bytes),
        "code_file": Path(code_path).name,
        "decoded_text": (recovered or b"").decode("utf-8", errors="replace"),
        "extra": extra, "error": None,
    }


def binary_demo(label, original_bytes, code_path, recovered, extra=""):
    """바이너리(PNG) 페이로드 데모 결과 레코드 — 디코딩 PNG도 파일로 저장."""
    ok = recovered == original_bytes
    rec = {
        "kind": "binary", "label": label, "ok": ok,
        "original_uri": png_data_uri(original_bytes),
        "original_size": len(original_bytes),
        "code_file": Path(code_path).name,
        "decoded_uri": None, "extra": extra, "error": None,
    }
    if ok and recovered:
        # 디코딩된 바이트 자체가 유효한 PNG임을 파일로 증명
        exp_tag = Path(code_path).name.split("_")[0]
        dec_path = GALLERY_DIR / f"{exp_tag}_decoded_binary.png"
        dec_path.write_bytes(recovered)
        rec["decoded_uri"] = png_data_uri(recovered)
        rec["decoded_file"] = dec_path.name
    return rec


def fail_demo(label, kind, error):
    return {"kind": kind, "label": label, "ok": False, "error": error}


def compare_demo(label, ok, cols, table_html, extra=""):
    """exp15 기준선 비교 데모 — 동일 면적 심볼 3장 나란히 + 비교표."""
    return {"kind": "compare", "label": label, "ok": ok, "cols": cols,
            "table_html": table_html, "extra": extra, "error": None}


def holo_demo(label, ok, original_file, cgh_file, replay_file,
              decoded_text, extra=""):
    """exp14 홀로그램 데모 — 원본 패턴 → CGH 위상 → 재생상+디코딩 3열."""
    return {
        "kind": "holo", "label": label, "ok": ok,
        "original_file": original_file, "cgh_file": cgh_file,
        "replay_file": replay_file, "decoded_text": decoded_text,
        "extra": extra, "error": None,
        "headers": ("원본 D_QR 패턴", "binary phase CGH (0/π)",
                    "홀로그램 재생상 → 디코딩"),
    }


# ── exp별 러너 ──────────────────────────────────────────────────────────────

def run_exp1():
    from src.exp1_color.encoder import DQREncoder
    from src.exp1_color.renderer import DQRRenderer
    from src.exp1_color.decoder import DQRDecoder

    enc, ren, dec = DQREncoder(ecc_rate=0.30), DQRRenderer(module_px=20), DQRDecoder()
    result = enc.encode(STANDARD_BYTES)
    img = ren.render(result)
    path = GALLERY_DIR / "exp1_code.png"
    img.save(path)
    recovered = dec.decode(
        img, grid_size=result['grid_size'], module_px=20,
        nsym=result['nsym'], data_len=result['data_len'],
        codeword_len=result['codeword_len'])
    cap = result['capacity']
    metrics = [f"bits/module: {cap['bits_per_module']}", "ECC 30% (단일 RS 블록)",
               f"그리드 {result['grid_size']}×{result['grid_size']}, 최대 {cap['max_data_bytes']}B",
               "벤치마크: QR(ECC-H) 대비 이론 용량 약 7배 — 단 색상 채널은 조명 의존"]
    return metrics, [text_demo("텍스트", STANDARD_BYTES, path, recovered)]


def run_exp2():
    from src.exp2_channel_ecc.encoder_v2 import DQREncoderV2
    from src.exp2_channel_ecc.decoder_v2 import DQRDecoderV2
    from src.exp1_color.renderer import DQRRenderer

    enc, ren, dec = DQREncoderV2(ecc_rate=0.30), DQRRenderer(module_px=20), DQRDecoderV2()
    result = enc.encode(STANDARD_BYTES)
    img = ren.render(result)
    path = GALLERY_DIR / "exp2_code.png"
    img.save(path)
    recovered = dec.decode(
        img, grid_size=result['grid_size'], module_px=20,
        nsym=result['nsym'], data_len=result['data_len'],
        codeword_lens=result['codeword_lens'],
        channel_data_bytes=result['channel_data_bytes'])
    cap = result['capacity']
    n1, n2, n3 = result['nsym']
    metrics = [f"bits/module: {cap['bits_per_module']}",
               f"ECC 채널 독립 RS: nsym={n1}/{n2}/{n3}",
               f"최대 {cap['max_data_bytes']}B",
               "벤치마크: 채널별 오류 격리 — 한 채널 오염이 다른 채널을 침범하지 않음"]
    return metrics, [text_demo("텍스트", STANDARD_BYTES, path, recovered)]


def run_exp3():
    from src.exp3_hexgrid.encoder_v3 import DQREncoderV3
    from src.exp3_hexgrid.renderer_v3 import DQRRendererV3
    from src.exp3_hexgrid.decoder_v3 import DQRDecoderV3

    enc, ren, dec = DQREncoderV3(ecc_rate=0.30), DQRRendererV3(hex_size=18), DQRDecoderV3()
    result = enc.encode(STANDARD_BYTES)
    img, info = ren.render(result)
    path = GALLERY_DIR / "exp3_code.png"
    img.save(path)
    recovered = dec.decode(
        img, radius=result['radius'], hex_size=info['hex_size'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym=result['nsym'], data_len=result['data_len'],
        codeword_lens=result['codeword_lens'],
        channel_data_bytes=result['channel_data_bytes'])
    cap = result['capacity']
    metrics = [f"육각 격자 radius={result['radius']} ({cap['total_cells']}셀, 3R²+3R+1)",
               "ECC 30% (채널 독립 유지)",
               f"최대 {cap['max_data_bytes']}B",
               "벤치마크: 동일 면적에서 사각 격자 대비 패킹 밀도 우위 + 3축(q/r/s) 좌표계"]
    return metrics, [text_demo("텍스트", STANDARD_BYTES, path, recovered)]


def run_exp4():
    from src.exp4_geometric.encoder_v4 import DQREncoderV4
    from src.exp4_geometric.renderer_v4 import DQRRendererV4
    from src.exp4_geometric.decoder_v4 import DQRDecoderV4

    enc = DQREncoderV4(ecc_rate=0.20)
    ren = DQRRendererV4(hex_size=28, fg_color=(0, 0, 0), bg_color=(255, 255, 255))
    dec = DQRDecoderV4(threshold=127, invert=False)
    result = enc.encode(STANDARD_BYTES)
    img, info = ren.render(result)
    path = GALLERY_DIR / "exp4_code.png"
    img.save(path)
    recovered = dec.decode(
        img, radius=result['radius'], hex_size=info['hex_size'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym=result['nsym'], data_len=result['data_len'],
        codeword_len=result['codeword_len'])
    cap = result['capacity']
    metrics = [f"bits/cell: {cap['bits_per_cell']} (primary 삼각형 6개, 이진 채움)",
               f"ECC 20% (nsym={result['nsym']})",
               f"최대 {cap['max_data_bytes']}B, 색상 의존: {cap['color_dependent']}",
               "벤치마크: 조명·색상 무관 — BER 대폭 감소, 홀로그램(녹색/검정)에도 동일 적용"]
    return metrics, [text_demo("텍스트", STANDARD_BYTES, path, recovered)]


def run_exp5():
    from src.exp5_zdepth.encoder_v5 import DQREncoderV5
    from src.exp5_zdepth.renderer_v5 import DQRRendererV5
    from src.exp5_zdepth.decoder_v5 import DQRDecoderV5

    enc, ren, dec = DQREncoderV5(ecc_rate=0.20), DQRRendererV5(hex_size=28), DQRDecoderV5()
    result = enc.encode(STANDARD_BYTES)
    img_2d, _img_3d, info = ren.render(result)  # 갤러리에는 flat 2D만
    path = GALLERY_DIR / "exp5_code.png"
    img_2d.save(path)
    recovered = dec.decode(
        img_2d, heights=result['heights'],  # 프로토타입: 시뮬레이션 높이
        radius=result['radius'], hex_size=info['hex_size'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym=result['nsym'], data_len=result['data_len'],
        codeword_len=result['codeword_len'])
    cap = result['capacity']
    metrics = [f"bits/cell: {cap['bits_per_cell']} (삼각형 {cap['bits_triangle']} + z-depth {cap['bits_height']})",
               f"ECC 20% (nsym={result['nsym']})",
               f"최대 {cap['max_data_bytes']}B",
               "벤치마크: z-depth 2비트로 용량 +30~65% — 단 제조(3D프린트)·스캔(LiDAR) 비용으로 강등"]
    return metrics, [text_demo("텍스트(flat 2D, 높이는 시뮬레이션)", STANDARD_BYTES, path, recovered)]


def _roundtrip_v6(data: bytes, path: Path):
    from src.exp6_subtri.encoder_v6 import DQREncoderV6
    from src.exp6_subtri.renderer_v6 import DQRRendererV6
    from src.exp6_subtri.decoder_v6 import DQRDecoderV6

    enc, ren, dec = DQREncoderV6(ecc_rate=0.20), DQRRendererV6(hex_size=32), DQRDecoderV6()
    result = enc.encode(data)
    img, info = ren.render(result)
    img.save(path)
    recovered = dec.decode(
        img, radius=result['radius'], hex_size=info['hex_size'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym=result['nsym'], data_len=result['data_len'],
        codeword_len=result['codeword_len'])
    return result, recovered


def run_exp6():
    result, rec_t = _roundtrip_v6(STANDARD_BYTES, GALLERY_DIR / "exp6_code.png")
    _, rec_b = _roundtrip_v6(DEMO_PNG, GALLERY_DIR / "exp6_code_binary.png")
    cap = result['capacity']
    metrics = [f"bits/cell: {cap['bits_per_cell']} ({cap['sub_triangles']} 서브삼각형)",
               f"ECC 20% (nsym={result['nsym']})",
               f"최대 {cap['max_data_bytes']}B",
               "벤치마크: exp4 대비 정확히 2배 용량 (동일 면적·동일 ECC)"]
    return metrics, [
        text_demo("텍스트", STANDARD_BYTES, GALLERY_DIR / "exp6_code.png", rec_t),
        binary_demo("바이너리 PNG (16×16)", DEMO_PNG, GALLERY_DIR / "exp6_code_binary.png", rec_b),
    ]


def _roundtrip_v7(data: bytes, path: Path):
    from src.exp7_fractal.encoder_v7 import DQREncoderV7
    from src.exp7_fractal.renderer_v7 import DQRRendererV7
    from src.exp7_fractal.decoder_v7 import DQRDecoderV7

    enc, ren, dec = DQREncoderV7(outer_radius=8, ecc_rate=0.20), DQRRendererV7(), DQRDecoderV7()
    result = enc.encode(data)
    img, info = ren.render(result)
    img.save(path)
    recovered = dec.decode(
        img, outer_radius=result['outer_radius'],
        outer_hex_size=info['outer_hex_size'], inner_hex_size=info['inner_hex_size'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym=result['nsym'], data_len=result['data_len'],
        codeword_len=result['codeword_len'])
    return result, recovered


def run_exp7():
    result, rec_t = _roundtrip_v7(STANDARD_BYTES, GALLERY_DIR / "exp7_code.png")
    _, rec_b = _roundtrip_v7(DEMO_PNG, GALLERY_DIR / "exp7_code_binary.png")
    cap = result['capacity']
    metrics = [f"bits/외부셀: {cap['bits_per_outer']} (프랙탈 {cap['fractal_levels']}레벨, 내부 {cap['inner_cells_per_outer']}셀)",
               f"ECC 20% (nsym={result['nsym']})",
               f"최대 {cap['max_data_bytes']}B (r={cap['outer_radius']})",
               "벤치마크: exp6 대비 3.5배 용량 — 단 내부 셀이 작아져 해상도 요구 급증"]
    return metrics, [
        text_demo("텍스트", STANDARD_BYTES, GALLERY_DIR / "exp7_code.png", rec_t),
        binary_demo("바이너리 PNG (16×16)", DEMO_PNG, GALLERY_DIR / "exp7_code_binary.png", rec_b),
    ]


def run_exp8():
    from src.exp8_dlc.encoder_v8 import DQREncoderV8
    from src.exp8_dlc.renderer_v8 import DQRRendererV8
    from src.exp8_dlc.decoder_v8 import DQRDecoderV8

    enc, ren, dec = DQREncoderV8(ecc_rate=0.20), DQRRendererV8(hex_size=32), DQRDecoderV8()
    # 듀얼 스트림: A=공개 텍스트, B=비공개 바이너리(PNG) — 한 이미지에 두 차원
    result = enc.encode(STANDARD_BYTES, DEMO_PNG)
    img, info = ren.render(result)
    path = GALLERY_DIR / "exp8_code.png"
    img.save(path)
    rec_a, rec_b = dec.decode_both(img, result, info)
    cap = result['capacity']
    metrics = [f"스트림 {cap['independent_streams']}개 독립 채널 (A {cap['stream_a_cells']}셀 / B {cap['stream_b_cells']}셀)",
               "ECC 20% (스트림별 독립 RS)",
               f"스트림당 최대 {cap['max_bytes_per_stream']}B",
               "벤치마크: 동일 이미지에서 공개/비공개 데이터셋 분리 — 차원 링크 체인의 시작"]
    return metrics, [
        text_demo("Stream A — 공개 텍스트", STANDARD_BYTES, path, rec_a),
        binary_demo("Stream B — 비공개 PNG (같은 이미지)", DEMO_PNG, path, rec_b),
    ]


def _roundtrip_v9(data: bytes, path: Path, decoder=None):
    """exp9 인코더 왕복 — decoder를 바꿔치우면 exp11 데모가 된다."""
    from src.exp9_soft.encoder_v9 import DQREncoderV9
    from src.exp9_soft.renderer_v9 import DQRRendererV9
    from src.exp9_soft.decoder_v9 import DQRDecoderV9

    enc, ren = DQREncoderV9(profile="A"), DQRRendererV9(hex_size=32)
    dec = decoder or DQRDecoderV9()
    result = enc.encode(data)
    img, info = ren.render(result)
    img.save(path)
    stats = dec.decode_with_stats(
        img, radius=result['radius'], hex_size=info['hex_size'],
        bits_per_cell=result['bits_per_cell'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym=result['nsym'], data_len=result['data_len'],
        codeword_len=result['codeword_len'])
    return result, stats


def run_exp9():
    result, stats_t = _roundtrip_v9(STANDARD_BYTES, GALLERY_DIR / "exp9_code.png")
    _, stats_b = _roundtrip_v9(DEMO_PNG, GALLERY_DIR / "exp9_code_binary.png")
    cap = result['capacity']
    metrics = [f"bits/cell: {result['bits_per_cell']} (프로파일 A — 인쇄·실외)",
               f"ECC {cap['ecc_rate']*100:.0f}% (nsym={result['nsym']})",
               f"최대 {cap['max_data_bytes']}B (r={result['radius']})",
               "벤치마크: 밝기 마진 신뢰도 + 소거 사다리 — 국소 오염 복구 hard 25% → soft 100%"]
    return metrics, [
        text_demo("텍스트", STANDARD_BYTES, GALLERY_DIR / "exp9_code.png",
                  stats_t['data'], extra=f"복호 모드: {stats_t['mode']} (소거 {stats_t['erasures_used']}개)"),
        binary_demo("바이너리 PNG (16×16)", DEMO_PNG, GALLERY_DIR / "exp9_code_binary.png",
                    stats_b['data'], extra=f"복호 모드: {stats_b['mode']}"),
    ]


def run_exp10():
    from src.exp10_progressive.encoder_v10 import DQREncoderV10
    from src.exp10_progressive.renderer_v10 import DQRRendererV10
    from src.exp10_progressive.decoder_v10 import DQRDecoderV10

    enc, ren, dec = DQREncoderV10(), DQRRendererV10(hex_size=32), DQRDecoderV10()
    # coarse=코어 텍스트(저해상 생존), fine=바이너리 PNG(고해상 전용) — 한 이미지
    result = enc.encode(STANDARD_BYTES, DEMO_PNG)
    img, info = ren.render(result)
    path = GALLERY_DIR / "exp10_code.png"
    img.save(path)
    stats = dec.decode_with_stats(
        img, radius=result['radius'], hex_size=info['hex_size'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym_c=result['nsym_c'], nsym_f=result['nsym_f'],
        coarse_len=result['coarse_len'], fine_len=result['fine_len'],
        codeword_len_c=result['codeword_len_c'],
        codeword_len_f=result['codeword_len_f'])
    cap = result['capacity']
    metrics = [f"dual-grain: coarse 6bit/셀 ECC 30% ({cap['coarse_cells']}셀) + fine 12bit/셀 ECC 20% ({cap['fine_cells']}셀)",
               f"용량: coarse {cap['max_coarse_bytes']}B / fine {cap['max_fine_bytes']}B",
               "신뢰도 게이트 FINE_GATE_THRESHOLD=0.405 (vote 메트릭, 단독 방어 아님 — 1·2층이 보강)",
               "벤치마크: 블러 6.0px에서 fine 붕괴(게이트 차단), coarse는 10px까지 생존"]
    return metrics, [
        text_demo("coarse — 코어 텍스트 (저해상 생존)", STANDARD_BYTES, path,
                  stats['coarse'], extra=f"mode={stats['coarse_mode']}"),
        binary_demo("fine — 확장 PNG (고해상 전용, 같은 이미지)", DEMO_PNG, path,
                    stats['fine'],
                    extra=f"mode={stats['fine_mode']}, vote 신뢰도={stats['fine_confidence']:.3f} (게이트 0.405 통과)"),
    ]


def run_exp11():
    from src.exp11_consensus.decoder_v11 import DQRDecoderV11

    dec = DQRDecoderV11()
    result, stats_t = _roundtrip_v9(STANDARD_BYTES, GALLERY_DIR / "exp11_code.png", decoder=dec)
    _, stats_b = _roundtrip_v9(DEMO_PNG, GALLERY_DIR / "exp11_code_binary.png", decoder=DQRDecoderV11())
    cap = result['capacity']
    metrics = [f"인코딩은 exp9 그대로 — 디코더만 교체 (샘플점 N={dec.n_samples}/삼각형)",
               f"ECC {cap['ecc_rate']*100:.0f}% (nsym={result['nsym']})",
               "신뢰도 = 이산 표차(vote consensus), 전역 임계 128 — 연속 밝기 마진 미사용",
               "벤치마크: Honeywell US 10,235,547 패밀리(반사율/대비 연속 신뢰도) FTO 회피 달성"]
    return metrics, [
        text_demo("텍스트 (exp9 인코더 + v11 디코더)", STANDARD_BYTES, GALLERY_DIR / "exp11_code.png",
                  stats_t['data'], extra=f"복호 모드: {stats_t['mode']} (소거 {stats_t['erasures_used']}개)"),
        binary_demo("바이너리 PNG (16×16)", DEMO_PNG, GALLERY_DIR / "exp11_code_binary.png",
                    stats_b['data'], extra=f"복호 모드: {stats_b['mode']}"),
    ]


def run_exp11b():
    from src.exp11_consensus.decoder_v11b import DQRDecoderV11B

    dec = DQRDecoderV11B()
    result, stats_t = _roundtrip_v9(STANDARD_BYTES,
                                    GALLERY_DIR / "exp11b_code.png",
                                    decoder=dec)
    _, stats_b = _roundtrip_v9(DEMO_PNG, GALLERY_DIR / "exp11b_code_binary.png",
                               decoder=DQRDecoderV11B())
    cap = result['capacity']
    # 핵심 수치 출처: outputs/exp11b/exp11b_ablation_summary.json (3자 비교)
    metrics = [
        "3층 해소: ①공간 소거 스윕+합의 수락(만장일치-오류) ②내부 링 폴백 t≤0.3(블러 절벽) "
        "③청크 errata 예산+최종 단계 순수 소거(잔존 오정정 84건 시그니처)",
        f"ECC {cap['ecc_rate']*100:.0f}% (nsym={result['nsym']}) — 인코딩은 exp9 그대로, 디코더만 v11b",
        "3자 비교(밝기 마진/표결 v11/v11b): 노이즈 σ135 → 0%/100%/100% · 블러 9.5px → 100%/0%/100%",
        "오정정: 이미지 실측 합계 176/82/0건, 코드워드 시뮬 1만 회 237→84(exp12 방어)→0건 — false reject 0",
        "잔존 한계(정직 고지): 균일 패치 일치율은 margin 대비 최대 A 28%p / B 44%p 열세 — FTO 1 준수의 구조적 비용",
    ]
    return metrics, [
        text_demo("텍스트 (exp9 인코더 + v11b 디코더)", STANDARD_BYTES,
                  GALLERY_DIR / "exp11b_code.png", stats_t['data'],
                  extra=f"복호 모드: {stats_t['mode']} (소거 {stats_t['erasures_used']}개, "
                        f"링 패스 {stats_t['rings_used']})"),
        binary_demo("바이너리 PNG (16×16)", DEMO_PNG,
                    GALLERY_DIR / "exp11b_code_binary.png", stats_b['data'],
                    extra=f"복호 모드: {stats_b['mode']} (링 패스 {stats_b['rings_used']})"),
    ]


def run_exp13():
    from src.exp13_selfdesc.encoder_v13 import DQREncoderV13
    from src.exp13_selfdesc.renderer_v13 import DQRRendererV13
    from src.exp13_selfdesc.decoder_v13 import DQRDecoderV13

    dec = DQRDecoderV13()

    def selfdesc_extra(st):
        o = st['orientation']
        return (f"자기추정(메타데이터 0): r={st['radius']}, "
                f"hex={st['hex_size']:.1f}px, 프로파일 {st['profile']}, "
                f"ECC lv{st['ecc_level']} (nsym={st['nsym']}), "
                f"회전 {o['rot_k'] * 120}°{', 거울상' if o['mirrored'] else ''} "
                f"— 모드 {st['mode']}, 소거 {st['erasures_used']}개")

    # 텍스트: 프로파일 A, 정방향 — 디스크의 PNG만으로 복호 (파라미터 일절 미전달)
    result = DQREncoderV13(profile='A').encode(STANDARD_BYTES)
    img, _ = DQRRendererV13(hex_size=32).render(result)
    path_t = GALLERY_DIR / "exp13_code.png"
    img.save(path_t)
    st_t = dec.decode_with_stats(Image.open(path_t))

    # 바이너리: 프로파일 B + 회전 120° 입력 — 방향까지 자기판별
    result_b = DQREncoderV13(profile='B').encode(DEMO_PNG)
    img_b, _ = DQRRendererV13(hex_size=32).render(result_b)
    img_b = img_b.rotate(120, resample=Image.Resampling.BICUBIC,
                         fillcolor=(255, 255, 255))
    path_b = GALLERY_DIR / "exp13_code_binary_rot120.png"
    img_b.save(path_b)
    st_b = dec.decode_with_stats(Image.open(path_b))

    cap = result['capacity']
    metrics = [
        "예약 셀 = 파인더 21 + 포맷 링 54 (18비트×3사본 비트별 다수결) — version·profile·ECC·radius 내장",
        "차등 파인더(NE 이웃 2셀 채움)로 D3 대칭 깨기 — 회전 3×거울상 6가지 가설 중 유일 판별",
        "길이 프리픽스 4B + 용량 0패딩 → codeword_len이 radius·nsym에서 유일 결정 (외부 메타데이터 불요)",
        f"최대 {cap['max_data_bytes']}B (r={result['radius']}, 프로파일 A, ECC {cap['ecc_rate']*100:.0f}%)",
        "벤치마크: 프로파일 A/B × 회전·거울상 6변형 왕복 100% — dqr_decode CLI가 메타 없는 PNG 자동 인식",
    ]
    return metrics, [
        text_demo("텍스트 — 이미지 파일만으로 디코딩 (메타데이터 없음)",
                  STANDARD_BYTES, path_t, st_t['data'],
                  extra=selfdesc_extra(st_t)),
        binary_demo("바이너리 PNG — 회전 120° 입력 (프로파일 B, 메타데이터 없음)",
                    DEMO_PNG, path_b, st_b['data'],
                    extra=selfdesc_extra(st_b)),
    ]


def _exp16_scene(img, angle=0.0, scale=1.0, tilt=0.0, offset=(0.3, 0.4),
                 bg=255, canvas_factor=1.7):
    """렌더 이미지 → 임의 배치 장면 (회전·스케일·완만 원근·오프셋).
    벤치마크 make_scene과 동일 모델의 시드 없는 결정론 버전."""
    import math

    import cv2
    import numpy as np
    from PIL import ImageOps

    im = img
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)),
                       Image.Resampling.LANCZOS)
    if angle:
        im = im.rotate(angle, resample=Image.Resampling.BICUBIC,
                       expand=True, fillcolor=(bg, bg, bg))
    arr = np.array(im)
    if tilt:
        h, w = arr.shape[:2]
        f = 2.0 * max(w, h)
        t = math.radians(tilt)
        pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                        [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float64)
        proj = []
        for x, y in pts:
            z = f + y * math.sin(t)
            proj.append([f * x / z + w / 2, f * y * math.cos(t) / z + h / 2])
        src = np.float32(pts + np.array([w / 2, h / 2]))
        H = cv2.getPerspectiveTransform(src, np.float32(proj))
        arr = cv2.warpPerspective(arr, H, (w, h), flags=cv2.INTER_LINEAR,
                                  borderValue=(bg, bg, bg))
    ch, cw = (int(arr.shape[0] * canvas_factor),
              int(arr.shape[1] * canvas_factor))
    canvas = np.full((ch, cw, 3), bg, dtype=np.uint8)
    oy = int((ch - arr.shape[0]) * offset[1])
    ox = int((cw - arr.shape[1]) * offset[0])
    canvas[oy:oy + arr.shape[0], ox:ox + arr.shape[1]] = arr
    return Image.fromarray(canvas)


def run_exp16():
    from src.exp13_selfdesc.encoder_v13 import DQREncoderV13
    from src.exp13_selfdesc.renderer_v13 import DQRRendererV13
    from src.exp16_detect import detect_and_decode

    def detect_extra(st):
        d = st['detection']
        return (f"검출(메타데이터 0): 파인더 자동 탐지 → 어파인 추정, "
                f"r={d['radius']}, hex≈{d['hex_size_img']:.1f}px, "
                f"가설 {d['n_hypotheses']}건 중 표마진 최상위 채택"
                f"{', 거울상' if d['mirrored'] else ''} — 복호 모드 {st['mode']}")

    # 텍스트: 프로파일 A, 회전 137° + 축소 0.7 + 오프셋 — 임의 배치 PNG
    result = DQREncoderV13(profile='A').encode(STANDARD_BYTES)
    img, _ = DQRRendererV13(hex_size=18).render(result)
    scene_t = _exp16_scene(img, angle=137, scale=0.7, offset=(0.7, 0.2))
    path_t = GALLERY_DIR / "exp16_scene.png"
    scene_t.save(path_t)
    st_t = detect_and_decode(Image.open(path_t))

    # 바이너리: 프로파일 B + 원근 12° + 회전 290° — 완만 기울기까지 흡수
    result_b = DQREncoderV13(profile='B').encode(DEMO_PNG)
    img_b, _ = DQRRendererV13(hex_size=18).render(result_b)
    scene_b = _exp16_scene(img_b, angle=290, tilt=12, scale=0.85,
                           offset=(0.25, 0.7))
    path_b = GALLERY_DIR / "exp16_scene_binary.png"
    scene_b.save(path_b)
    st_b = detect_and_decode(Image.open(path_b))

    # 스윕 수치 출처: benchmarks/exp16_detect_sweep.py 실측
    metrics = [
        "검출 = 전역 Otsu → 연결 성분 → 형상 필터 → 흰 링 검증으로 단일 파인더 후보 → "
        "닮음 2점 대응으로 NE 꼭짓점 가설(배정 2×거울 2)×반경 스냅",
        "정밀화 = 꼭짓점 이산 격자 스윕 + 3점 어파인 + 사영 (g,h) 스윕 — 목적함수는 "
        "전부 정수 표 합(연속 밝기량 없음, FTO 준수)",
        "디코딩 = decoder_v13에 좌표 사상만 주입 — 포맷 링 대조·3층 방어·소거 사다리·"
        "퇴화 가드 등 판정 경로 무수정 (아키텍처 분리)",
        "스윕 1,400회(조건 28×N=50): 회전 0~360°·스케일 0.5~2×·원근 ≤20° 100% 검출·복호, "
        "원근 25°/30°는 우아한 저하(검출 92%/84%) — 전 구간 침묵 오정정 0건",
        "v1(exp18): 쌍 순위를 격자 반경 스냅(이산 기하)으로 보강 — 모션 블러·글레어 강건성 향상, "
        "기하 스윕 회귀 없음. 잔여 v1 과제: 손촬영·복잡 배경·심한 원근>20°",
    ]
    return metrics, [
        text_demo("텍스트 — 회전 137°+축소 장면에서 위치정보 0으로 검출·디코딩",
                  STANDARD_BYTES, path_t, st_t['data'],
                  extra=detect_extra(st_t)),
        binary_demo("바이너리 PNG — 회전 290°+원근 12° 장면 검출·디코딩 (프로파일 B)",
                    DEMO_PNG, path_b, st_b['data'],
                    extra=detect_extra(st_b)),
    ]


def run_exp17():
    import numpy as np

    from src.exp13_selfdesc.encoder_v13 import DQREncoderV13
    from src.exp13_selfdesc.renderer_v13 import DQRRendererV13
    from src.exp16_detect import detect_and_decode
    from src.exp17_physical import PRESETS, camera_pipeline

    def degrade_extra(st, preset):
        d = st['detection']
        return (f"물리 열화 '{preset}' 통과 후 검출(메타·위치정보 0): "
                f"r={d['radius']}, hex≈{d['hex_size_img']:.1f}px, "
                f"가설 {d['n_hypotheses']}건 — 복호 모드 {st['mode']}")

    def make_degraded(profile, payload, preset, scene_kw, seed):
        result = DQREncoderV13(profile=profile).encode(payload)
        img, _ = DQRRendererV13(hex_size=22).render(result)
        scene = _exp16_scene(img, **scene_kw)
        rng = np.random.default_rng(seed)
        arr = camera_pipeline(np.array(scene.convert('RGB')), rng,
                              **PRESETS[preset])
        return Image.fromarray(arr)

    # 텍스트: moderate 프리셋(평범한 실내 핸드헬드: 디포커스3·모션5·비네팅·글레어·노이즈·JPEG70)
    scene_t = make_degraded('A', STANDARD_BYTES, 'moderate',
                            dict(angle=18, tilt=6, offset=(0.5, 0.4)), seed=171)
    path_t = GALLERY_DIR / "exp17_scene.png"
    scene_t.save(path_t)
    st_t = detect_and_decode(Image.open(path_t))

    # 바이너리: mild 프리셋(양호한 촬영: 약한 디포커스·비네팅·노이즈·JPEG90)
    scene_b = make_degraded('B', DEMO_PNG, 'mild',
                            dict(angle=42, scale=0.95, offset=(0.45, 0.5)),
                            seed=172)
    path_b = GALLERY_DIR / "exp17_scene_binary.png"
    scene_b.save(path_b)
    st_b = detect_and_decode(Image.open(path_b))

    # 스윕 수치 출처: benchmarks/exp17_physical_sweep.py 실측
    metrics = [
        "열화 모델(물리 수식·출처는 degrade.py 주석): 디포커스 disk PSF(Mannan&Langer) · "
        "손떨림 선형 모션 · 비네팅 cos⁴(Aggarwal) · 정반사 글레어 · "
        "Poisson-Gaussian 센서 노이즈(Foi) · Brown-Conrady 렌즈 왜곡 · JPEG 양자화 · 인쇄 도트게인",
        "물리 발생 순서로 직렬 적용: 인쇄 → 광학 → 센서 → 저장. 회전 0~360°+원근 배치 위에 입힘",
        "검출 스윕(조건 32×N=50, exp18 v1 적용): 디포커스 R6·노이즈 전ISO·JPEG QF30·비네팅·"
        "렌즈왜곡·모션 L5 100%, 모션 L9 96%·글레어 100% — severe 다중중첩만 v0 범위 밖, "
        "전 구간 침묵 오정정 0건",
        "exp18 검출 v1: 모션 진단(가짜 파인더 폭증→쌍 상한 포화)을 격자 반경 스냅 순위로 해소 "
        "— 모션 L9 검출 57→96%, 글레어 90→100% (이산 기하 판별, 밝기량 아님)",
        "FTO: 열화 모듈은 입력 생성 전용 — 소거·신뢰도 판정 경로(softdecode·spatial·consensus) "
        "미임포트, 디코더는 열화 존재를 모름",
        "다음 단계: 합성을 넘어 실물 인쇄→카메라 촬영(exp19 후보) — 레이저 각인 알루미늄·종이 실촬영",
    ]
    return metrics, [
        text_demo("텍스트 — moderate 물리 열화(디포커스·모션·글레어·노이즈·JPEG70) 통과 검출",
                  STANDARD_BYTES, path_t, st_t['data'],
                  extra=degrade_extra(st_t, 'moderate')),
        binary_demo("바이너리 PNG — mild 물리 열화(양호한 촬영) 통과 검출 (프로파일 B)",
                    DEMO_PNG, path_b, st_b['data'],
                    extra=degrade_extra(st_b, 'mild')),
    ]


def run_exp20():
    import numpy as np

    from src.exp10_progressive.encoder_v10 import DQREncoderV10
    from src.exp10_progressive.renderer_v10 import DQRRendererV10
    from src.exp17_physical.degrade import defocus_blur
    from src.exp20_consensus_gate import DQRDecoderV20

    enc, ren, dec = DQREncoderV10(), DQRRendererV10(hex_size=24), DQRDecoderV20()
    result = enc.encode(STANDARD_BYTES, DEMO_PNG)
    img, info = ren.render(result)
    path = GALLERY_DIR / "exp20_code.png"
    img.save(path)

    def decode(image):
        return dec.decode_with_stats(
            image, radius=result['radius'], hex_size=info['hex_size'],
            origin_x=info['origin_x'], origin_y=info['origin_y'],
            nsym_c=result['nsym_c'], nsym_f=result['nsym_f'],
            coarse_len=result['coarse_len'], fine_len=result['fine_len'],
            codeword_len_c=result['codeword_len_c'],
            codeword_len_f=result['codeword_len_f'])

    clean = decode(img)
    # 좋은 센서(클린) → 둘 다, 나쁜 센서(디포커스 R8) → coarse만 (점진적 공개)
    arr = np.array(img.convert('RGB')).astype(np.float32)
    degraded_img = Image.fromarray(
        np.clip(defocus_blur(arr, 8.0), 0, 255).astype(np.uint8))
    degraded_img.save(GALLERY_DIR / "exp20_degraded.png")
    deg = decode(degraded_img)

    cap = result['capacity']
    metrics = [
        f"dual-grain: coarse 6bit/셀 ECC30% ({cap['coarse_cells']}셀) + fine 12bit/셀 ECC20% ({cap['fine_cells']}셀)",
        "FTO 이식(exp20): 점진적 공개 게이트를 밝기 마진→표 일치율 이산(표차 평균/N)으로 교체 "
        "— exp10 마진 경로(Honeywell 회피)는 동결, 출원·공개는 exp20",
        f"좋은 센서(클린): coarse+fine 둘 다 복호 (fine_consensus={clean['fine_consensus']:.3f}) · "
        f"나쁜 센서(디포커스 R8): coarse만 생존, fine 게이트 차단(None, consensus={deg['fine_consensus']:.3f})",
        "= 헤드라인 정체성 '좋은 센서일수록 더 깊이 읽힌다'를 밝기 마진 없이 구현 · "
        "전 단계 침묵 오정정 0(coarse 절대 오답 없음, fine은 우아한 강등)",
    ]
    return metrics, [
        text_demo("coarse — 코어 텍스트 (저해상·열화 생존)", STANDARD_BYTES, path,
                  clean['coarse'], extra=f"클린 mode={clean['coarse_mode']} · "
                  f"디포커스 R8에서도 coarse 복호 OK={deg['coarse'] == STANDARD_BYTES}"),
        binary_demo("fine — 확장 PNG (고해상 전용, 표결 게이트)", DEMO_PNG, path,
                    clean['fine'],
                    extra=f"클린 mode={clean['fine_mode']}, consensus={clean['fine_consensus']:.3f} · "
                          f"디포커스 R8: fine={'None(게이트 차단)' if deg['fine'] is None else '복호'}"),
    ]


def run_exp14():
    import numpy as np
    from src.exp10_progressive.encoder_v10 import DQREncoderV10
    from src.exp10_progressive.renderer_v10 import DQRRendererV10
    from src.exp10_progressive.decoder_v10 import DQRDecoderV10

    # 벤치마크(benchmarks/exp14_holo_sim.py)와 동일 페이로드·기하
    coarse = "https://dqr.kr/h — exp14 코어".encode("utf-8")
    fine = "exp14 확장 페이로드: 홀로그램 재생 점진적 공개 검증.".encode("utf-8")

    result = DQREncoderV10().encode(coarse, fine)
    img, info = DQRRendererV10(hex_size=16).render(result)
    params = dict(
        radius=result['radius'], hex_size=info['hex_size'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym_c=result['nsym_c'], nsym_f=result['nsym_f'],
        coarse_len=result['coarse_len'], fine_len=result['fine_len'],
        codeword_len_c=result['codeword_len_c'],
        codeword_len_f=result['codeword_len_f'])

    names = {"original": "exp14_original.png", "cgh": "exp14_cgh_phase.png",
             "coherent": "exp14_replay_coherent.png",
             "led0": "exp14_replay_led_k0.png"}
    bench_dir = Path("outputs/exp14")

    # 벤치마크 산출물이 현재 코드의 렌더와 픽셀 단위 일치할 때만 재사용
    reuse = all((bench_dir / n).exists() for n in names.values())
    if reuse:
        with Image.open(bench_dir / names["original"]) as saved:
            reuse = (saved.size == img.size and np.array_equal(
                np.array(saved.convert("RGB")), np.array(img.convert("RGB"))))
    if reuse:
        for n in names.values():
            shutil.copyfile(bench_dir / n, GALLERY_DIR / n)
    else:
        # 산출물 없음/불일치 → CGH 재계산 (GS 50+25회, 수십 초 소요)
        from src.exp14_holosim.cgh import (DESIGN_NM, embed_target,
                                           gs_binary_phase, replay_intensity)
        from src.exp14_holosim.replay import (LED_WAVELENGTHS,
                                              intensity_to_image)
        target, window = embed_target(255 - np.array(img.convert('L')), 2048)
        phi = gs_binary_phase(target, n_iter_continuous=50,
                              n_iter_binary=25, seed=14)
        img.save(GALLERY_DIR / names["original"])
        Image.fromarray((phi / np.pi * 255).astype(np.uint8)).save(
            GALLERY_DIR / names["cgh"])
        intensity_to_image(replay_intensity(phi, DESIGN_NM)[window]).save(
            GALLERY_DIR / names["coherent"])
        total = None
        for lam in LED_WAVELENGTHS:     # κ=0 → s_eff=1, zoom 불필요
            I = replay_intensity(phi, lam)
            total = I if total is None else total + I
        intensity_to_image(total[window] / len(LED_WAVELENGTHS)).save(
            GALLERY_DIR / names["led0"])

    dec = DQRDecoderV10()

    def attempt(name):
        with Image.open(GALLERY_DIR / name) as im:
            try:
                st = dec.decode_with_stats(im.convert('RGB'), **params)
            except Exception as e:
                return False, False, f"복호 실패: {e}", ""
        c_ok, f_ok = st['coarse'] == coarse, st['fine'] == fine
        extra = (f"coarse {'O' if c_ok else 'X'} ({st['coarse_mode']}) / "
                 f"fine {'O' if f_ok else 'X'} ({st['fine_mode'] or '—'}), "
                 f"vote 신뢰도={st['fine_confidence']:.3f}")
        decoded = (st['coarse'] or b"").decode("utf-8", errors="replace")
        if f_ok:
            decoded += "\n" + (st['fine'] or b"").decode("utf-8", errors="replace")
        return c_ok, f_ok, extra, decoded

    c_a, f_a, ex_a, dec_a = attempt(names["coherent"])
    c_b, f_b, ex_b, dec_b = attempt(names["led0"])

    cap = result['capacity']
    # 스윕 수치 출처: outputs/exp14/exp14_holo_report.md (벤치마크 실측)
    metrics = [
        f"패턴: exp10 dual-grain (r={result['radius']}, coarse {cap['coarse_cells']}셀 / "
        f"fine {cap['fine_cells']}셀), CGH = Gerchberg-Saxton 50+25회, field 2048², 오프축 창",
        "재생 효율 η=0.421 (이진 위상 물리 상한 ~0.405와 정합, 쌍둥이 상 몫 제외), 충실도 corr=0.971",
        "(b) 백색 LED 3파장 분산: coarse κ≤0.05 / fine κ≤0.02 — coarse만 생존하는 대역 존재 = "
        "점진적 공개가 분산 도메인으로 이식",
        "(c) 스페클: coarse 전수생존 C≤0.5 / fine C≤0.4 — 생존 경계 실측",
        "대조군: RGB 3채널 컬러 코드는 백색 조명에서 비트 정확도 68~74%로 붕괴 (단일색 100%) — "
        "창립 가설 '단일색=홀로그램 호환' 지지 (판독은 decoder_v10 그대로, 새 신뢰도 메트릭 없음)",
    ]
    return metrics, [
        holo_demo("(a) 코히어런트 532nm 재생 — 합격선: coarse+fine 동시 판독",
                  c_a and f_a, names["original"], names["cgh"],
                  names["coherent"], dec_a, extra=ex_a),
        holo_demo("(b) 백색 LED 3파장 분산 κ=0 (이미지면, 엠보싱 양산 경로) — 합격선: coarse 생존",
                  c_b, names["original"], names["cgh"],
                  names["led0"], dec_b, extra=ex_b),
    ]


def run_exp15():
    import json
    from benchmarks.exp15_baseline import SYM_LABELS, build_symbols

    # 측정 수치 출처: benchmarks/exp15_baseline.py 실측 결과 (조건 18 × N=100)
    summary_path = Path("outputs/exp15/exp15_baseline_summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} 없음 — py -X utf8 benchmarks/exp15_baseline.py 먼저 실행")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    syms = build_symbols()
    order = [k for k in ("qr", "dm", "dqr") if k in syms]
    from benchmarks.exp15_baseline import PAYLOAD as EXP15_PAYLOAD

    cols, all_ok = [], True
    for k in order:
        sym = syms[k]
        path = GALLERY_DIR / f"exp15_{k}.png"
        sym["canvas"].save(path)
        ok = sym["read"](sym["canvas"]) == EXP15_PAYLOAD   # 깨끗한 캔버스 왕복
        all_ok &= ok
        m = sym["meta"]
        cols.append({
            "header": SYM_LABELS[k], "file": path.name, "ok": ok,
            "sub": (f"{m['symbology']} · 특성 {m.get('feature_px', '?')}px · "
                    f"ECC {m.get('ecc_share', '?')}% · {m['reader']}"),
        })
    for k in ("qr", "dm", "dqr"):     # 기준선 라이브러리 실패도 정직하게 표기
        if k not in syms:
            all_ok = False
            cols.append({"header": SYM_LABELS[k], "file": None, "ok": False,
                         "sub": "비교 불가 — 생성/판독 라이브러리 실패"})

    # 비교표: 총괄(성공/오정정/거부) + 대표 조건별 성공률
    tot = summary["totals"]
    conds = summary["conditions"]
    n_cond = summary["trials_per_condition"]
    n_total = n_cond * len(conds)
    picks = [("noise/140", "노이즈 σ140"), ("scatter/20", "산포 20패치"),
             ("blot/2", "블롯 2개"), ("blur/6", "블러 6px")]

    def pct(c):
        s = f"{100 * c['ok'] / n_cond:.0f}%"
        return s + (f" <b>W{c['wrong']}</b>" if c["wrong"] else "")

    rows = []
    for k in order:
        t = tot[k]
        cells = [f"<td>{esc(SYM_LABELS[k])}</td>",
                 f"<td>{t['ok']:,} / {n_total:,}</td>",
                 f"<td><b>{t['wrong']}</b></td>", f"<td>{t['fail']:,}</td>"]
        cells += [f"<td>{pct(conds[c]['counts'][k])}</td>" for c, _ in picks]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    table_html = (
        '<table class="timeline" style="margin:14px 0 0">'
        "<tr><th>심볼로지</th><th>성공 (전 조건)</th><th>침묵 오정정</th>"
        "<th>거부</th>"
        + "".join(f"<th>{esc(lbl)}</th>" for _, lbl in picks)
        + "</tr>" + "".join(rows) + "</table>")

    w_dqr = tot.get("dqr", {}).get("wrong", 0)
    ecc_line = " / ".join(
        f"{SYM_LABELS[k]} {summary['meta'][k].get('ecc_share', '?')}%"
        for k in order)
    w_line = (f"침묵 오정정(핵심 지표): "
              + " / ".join(f"{SYM_LABELS[k]} {tot[k]['wrong']}건" for k in order))
    if w_dqr:
        w_line += (f" — D_QR {w_dqr}건은 전부-0 코드워드 퇴화 단일 모드"
                   "(군집 블롯 한정, RS 선형성상 전부-0은 항상 유효 코드워드)")

    # exp15b 후속 측정: 4층 퇴화 가드 적용 후 결과 (benchmarks/exp15b_rerun.py)
    b_path = Path("outputs/exp15b/exp15b_summary.json")
    if b_path.exists():
        b = json.loads(b_path.read_text(encoding="utf-8"))
        w_before = sum(v["unguarded"]["wrong"] for v in b["blot"].values())
        w_after = sum(v["guarded"]["wrong"] for v in b["blot"].values())
        ok_delta = sum(v["ok_delta"] for v in b["blot"].values())
        guard_line = (
            f"exp15b 4층 가드(softdecode.is_degenerate_message — 전부-0/상수 메시지 결정론 거부, "
            f"CRC 아님)로 봉쇄: 블롯 채널 동일 이미지 재측정 오정정 {w_before}건 → {w_after}건, "
            f"정상 복호 변동 {ok_delta:+d}건 · exp11b 스위트 2,000 이미지 오정정 "
            f"{b['suite_wrong_total']}건 유지 · 실측 케이스(blot2 trial8) 직접 재현·거부 확인 — "
            f"v11b(기본 on)·v13·공간 소거 스윕 양 경로 활성, exp9~12 기본값 off 동결 "
            f"+ 인코더 LFSR 스크램블(v13 기본 on)")
    else:
        guard_line = ("exp15b 가드 측정 산출물 없음 — "
                      "py -X utf8 benchmarks/exp15b_rerun.py 실행 필요")
    metrics = [
        f"공정 비교 규약: 동일 페이로드 {summary['payload_bytes']}B · 동일 점유 면적(640² 상당 → 800² 캔버스) · "
        f"동일 열화(시드 {summary['seed']}, exp9/exp11b 모델) — 조건 {len(conds)}개 × N={n_cond}",
        f"ECC 오버헤드 비동일(명시): {ecc_line} — 짧은 페이로드에서 D_QR 실효 ECC가 가장 높은 축",
        "검출 비대칭(명시): D_QR만 기하 파라미터 제공 — D_QR 성공률은 검출 비용 제외 상한. "
        "단 침묵 오정정 지표는 '반환했으면 책임진다' 기준이라 비대칭 무관",
        w_line,
        guard_line,
        "지는 구간(정직 고지): 블러 6px에서 DataMatrix 100% vs D_QR 0% — 동일 면적 정규화에서 "
        "고밀도 기하의 비트당 특성이 가장 작은 데서 오는 구조적 비용 (네이티브 해상도에선 9.5px 생존)",
    ]
    return metrics, [
        compare_demo(
            "동일 면적 정규화 캔버스 — 깨끗한 상태 왕복 검증 + 열화 비교표 "
            "(수치: outputs/exp15/exp15_baseline_report.md 실측)",
            all_ok, cols, table_html,
            extra="PASS 기준: 깨끗한 캔버스에서 각 판독기가 동일 59B 페이로드를 정확 반환. "
                  "표의 성공률·오정정(W)은 조건당 N=100 실측."),
    ]


# ── exp 메타데이터 (진화 노트) ───────────────────────────────────────────────

EXPS = [
    ("exp1",  "색상 3채널 인코딩", "RGB 3채널 × 2비트 = 6비트/모듈. QR 대비 7배 이론 용량의 출발점.", run_exp1),
    ("exp2",  "채널 독립 ECC", "exp1의 단일 RS 블록을 채널별 독립 RS로 분리 — 채널 국소 오염 격리.", run_exp2),
    ("exp3",  "육각 격자", "사각 격자를 axial 육각 격자(3R²+3R+1셀)로 전환 — 패킹 밀도·3축 좌표 확보.", run_exp3),
    ("exp4",  "색상 제거 — 순수 기하", "색상 채널 폐기, primary 삼각형 6개 이진 채움. 조명·색 의존성 제거 = 정체성 확립.", run_exp4),
    ("exp5",  "Z-depth (강등)", "부조 높이 2비트 추가 시도 — 용량 +30~65%였으나 제조·스캔 비용으로 강등.", run_exp5),
    ("exp6",  "12 서브삼각형", "primary 삼각형을 12 서브삼각형으로 분할 — exp4 대비 정확히 2배 밀도.", run_exp6),
    ("exp7",  "프랙탈 중첩", "셀 내부에 2레벨 자기유사 헥사곤 — 42비트/외부셀, 해상도 요구와의 트레이드오프.", run_exp7),
    ("exp8",  "듀얼 스트림 (DLC)", "한 이미지에 독립 스트림 2개 — 공개/비공개 차원 분리, 차원 링크 체인 시작.", run_exp8),
    ("exp9",  "Soft-decision 소거", "밝기 마진 신뢰도 + 소거 사다리 — 국소 오염 복구율 hard 25% → soft 100%.", run_exp9),
    ("exp10", "Dual-grain 점진적 공개", "coarse(6bit·ECC30%)/fine(12bit·ECC20%) 공간 분리 + 신뢰도 게이트 — 좋은 센서일수록 더 깊이 읽힌다.", run_exp10),
    ("exp11", "표 일치율 메트릭 (특허 회피)", "연속 밝기 마진을 이산 표차(vote consensus, N=9)로 교체 — Honeywell FTO 제약 준수.", run_exp11),
    ("exp11b", "표결 3대 결함 해소", "exp11 중단규칙 3대 결함(만장일치-오류·블러 절벽·errata 재소진)을 공간 소거 스윕·내부 링 폴백·예산 검사로 해소 — 오정정 0 달성 (exp12 방어 계층 통합).", run_exp11b),
    ("exp13", "자기기술 코드", "포맷 링 + 차등 파인더 + 기하 자기추정 — 메타데이터 전무한 PNG만으로 디코딩 (회전·거울상 불변).", run_exp13),
    ("exp14", "홀로그램 CGH 시뮬", "창립 가설 '단일색=홀로그램 호환' 0단계 검증 — binary phase CGH 재생에서 coarse+fine 보존, LED 분산에선 coarse만 생존(점진적 공개 이식), RGB 컬러 코드는 붕괴.", run_exp14),
    ("exp15", "외부 기준선 비교 + 퇴화 가드", "QR(ECC-H)·DataMatrix(ECC200) 상용 구현과 동일 페이로드·면적·열화로 비교 — 노이즈·국소 오염 우위, 블러 열세 정직 계수. 침묵 오정정 7건(전부-0 퇴화)은 exp15b 4층 가드로 0건 봉쇄.", run_exp15),
    ("exp16", "검출 파이프라인 v0", "마지막 critical 갭 해소 — 메타데이터·위치정보 0인 임의 배치 PNG에서 파인더 자동 탐지 → 어파인+사영 추정 → 자기기술 디코딩. 회전 0~360°·스케일 0.5~2×·원근 ≤20° 100%, 전 구간 침묵 오정정 0. 검출 모듈은 소거 판정 경로와 아키텍처 분리(FTO).", run_exp16),
    ("exp17", "물리 카메라 열화 모델", "exp16의 이상적 합성을 넘어 인쇄→촬영 간극을 물리 모델로 근사 — 디포커스·모션·비네팅·글레어·Poisson-Gaussian 노이즈·렌즈왜곡·JPEG·도트게인을 발생 순서로 직렬 적용. 약~중간 열화 강건, 전 구간 침묵 오정정 0. 열화 모듈은 입력 생성 전용(FTO 분리).", run_exp17),
    ("exp20", "점진적 공개 게이트 FTO 이식", "exp10 dual-grain 게이트가 밝기 마진(Honeywell 침해 메트릭) 의존이던 것을 발견 → 표 일치율 이산 게이트로 이식. 헤드라인 정체성 '좋은 센서일수록 더 깊이 읽힌다'를 exp17 물리 열화 위에서 실증: 클린은 coarse+fine, 디포커스 R8은 coarse만(fine 게이트 차단). 침묵 오정정 0.", run_exp20),
]


# ── HTML 생성 ────────────────────────────────────────────────────────────────

CSS = """
:root { --bg:#0f1117; --card:#1a1d27; --line:#2a2e3d; --fg:#d8dce8; --dim:#8b91a5;
        --accent:#5eead4; --pass:#22c55e; --fail:#ef4444; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); margin:0; padding:32px 24px;
       font-family:'Malgun Gothic','Segoe UI',sans-serif; line-height:1.55; }
.wrap { max-width:1180px; margin:0 auto; }
h1 { color:var(--accent); font-size:26px; margin:0 0 4px; }
.sub { color:var(--dim); margin-bottom:28px; font-size:14px; }
table.timeline { width:100%; border-collapse:collapse; margin-bottom:36px; font-size:13px; }
table.timeline th, table.timeline td { border:1px solid var(--line); padding:7px 10px; text-align:left; }
table.timeline th { background:#222636; color:var(--accent); }
table.timeline tr:nth-child(even) td { background:#171a24; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px;
        padding:22px 24px; margin-bottom:28px; }
.card h2 { margin:0 0 2px; font-size:19px; }
.card h2 .tag { color:var(--accent); margin-right:8px; }
.note { color:var(--dim); font-size:13.5px; margin:2px 0 12px; }
ul.metrics { margin:0 0 16px; padding-left:18px; font-size:13px; color:var(--dim); }
ul.metrics li { margin:2px 0; }
.demo { display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; align-items:start;
        border-top:1px dashed var(--line); padding-top:14px; margin-top:14px; }
.demo .col h3 { font-size:12px; color:var(--dim); text-transform:uppercase;
                letter-spacing:.06em; margin:0 0 8px; }
.demo .col h3 .arrow { color:var(--accent); }
pre.data { background:#11131c; border:1px solid var(--line); border-radius:8px;
           padding:10px 12px; font-size:12.5px; white-space:pre-wrap; word-break:break-all;
           margin:0; max-height:160px; overflow:auto; }
.codeimg { display:block; max-width:100%; max-height:260px; border-radius:8px;
           border:1px solid var(--line); background:#fff; }
.pixel { image-rendering:pixelated; width:96px; height:96px; border:1px solid var(--line);
         border-radius:6px; background:#fff; }
.badge { display:inline-block; padding:2px 12px; border-radius:999px; font-size:12px;
         font-weight:bold; margin-bottom:8px; }
.badge.pass { background:rgba(34,197,94,.15); color:var(--pass); border:1px solid var(--pass); }
.badge.fail { background:rgba(239,68,68,.15); color:var(--fail); border:1px solid var(--fail); }
.extra { color:var(--dim); font-size:12px; margin-top:6px; }
.size { color:var(--dim); font-size:11.5px; margin-top:4px; }
.demolabel { font-size:13px; font-weight:bold; color:var(--accent); margin-top:18px; }
.errbox { background:rgba(239,68,68,.08); border:1px solid var(--fail); border-radius:8px;
          padding:12px 14px; font-size:13px; color:#fca5a5; }
a { color:var(--accent); }
footer { color:var(--dim); font-size:12px; margin-top:24px; }
"""


def esc(s):
    return html.escape(str(s))


def render_demo_html(d):
    if d.get("error"):
        return (f'<div class="demolabel">{esc(d["label"])}</div>'
                f'<div class="errbox">실패 사유 (정직 고지): {esc(d["error"])}</div>')

    badge = ('<span class="badge pass">PASS</span>' if d["ok"]
             else '<span class="badge fail">FAIL</span>')
    extra = f'<div class="extra">{esc(d["extra"])}</div>' if d.get("extra") else ""

    if d["kind"] == "holo":
        h1, h2, h3 = d["headers"]

        def img_col(fname, alt):
            return (f'<a href="{fname}" target="_blank" title="클릭 시 원본 크기">'
                    f'<img class="codeimg" src="{fname}" alt="{esc(alt)}"></a>'
                    f'<div class="size">{fname}</div>')

        col3 = (f'{badge}{img_col(d["replay_file"], h3)}'
                f'<pre class="data">{esc(d["decoded_text"])}</pre>{extra}')
        return (f'<div class="demolabel">{esc(d["label"])}</div>'
                f'<div class="demo">'
                f'<div class="col"><h3>{esc(h1)} <span class="arrow">→</span></h3>'
                f'{img_col(d["original_file"], h1)}</div>'
                f'<div class="col"><h3>{esc(h2)} <span class="arrow">→</span></h3>'
                f'{img_col(d["cgh_file"], h2)}</div>'
                f'<div class="col"><h3>{esc(h3)}</h3>{col3}</div>'
                f'</div>')

    if d["kind"] == "compare":
        cols_html = []
        for c in d["cols"]:
            cb = ('<span class="badge pass">PASS</span>' if c["ok"]
                  else '<span class="badge fail">FAIL</span>')
            img = (f'<a href="{c["file"]}" target="_blank" title="클릭 시 원본 크기">'
                   f'<img class="codeimg" src="{c["file"]}" alt="{esc(c["header"])}"></a>'
                   if c["file"] else
                   '<div class="errbox">이미지 없음</div>')
            cols_html.append(
                f'<div class="col"><h3>{esc(c["header"])}</h3>{cb}{img}'
                f'<div class="size">{esc(c["sub"])}</div></div>')
        return (f'<div class="demolabel">{esc(d["label"])}</div>'
                f'<div class="demo">{"".join(cols_html)}</div>'
                f'{d["table_html"]}{extra}')

    if d["kind"] == "text":
        col1 = (f'<pre class="data">{esc(d["original_text"])}</pre>'
                f'<div class="size">UTF-8 {d["original_size"]} bytes</div>')
        col3 = f'{badge}<pre class="data">{esc(d["decoded_text"])}</pre>{extra}'
    else:
        col1 = (f'<img class="pixel" src="{d["original_uri"]}" alt="원본 PNG">'
                f'<div class="size">PNG {d["original_size"]} bytes (16×16)</div>')
        if d.get("decoded_uri"):
            col3 = (f'{badge}<img class="pixel" src="{d["decoded_uri"]}" alt="디코딩 PNG">'
                    f'<div class="size">디코딩 바이트 = 원본 바이트 (<a href="{d["decoded_file"]}">파일</a>)</div>{extra}')
        else:
            col3 = f'{badge}<div class="size">디코딩 바이트가 원본과 불일치</div>{extra}'

    col2 = (f'<a href="{d["code_file"]}" target="_blank" title="클릭 시 원본 크기">'
            f'<img class="codeimg" src="{d["code_file"]}" alt="D_QR 코드"></a>'
            f'<div class="size">{d["code_file"]}</div>')

    return (f'<div class="demolabel">{esc(d["label"])}</div>'
            f'<div class="demo">'
            f'<div class="col"><h3>원본 데이터 <span class="arrow">→</span></h3>{col1}</div>'
            f'<div class="col"><h3>D_QR 이미지 <span class="arrow">→</span></h3>{col2}</div>'
            f'<div class="col"><h3>디코딩 결과</h3>{col3}</div>'
            f'</div>')


def build_html(cards):
    rows = []
    for c in cards:
        status = ("오류" if c["error"] else
                  " / ".join(("PASS" if d["ok"] else "FAIL") for d in c["demos"]))
        rows.append(f'<tr><td><a href="#{c["exp"]}">{c["exp"]}</a></td>'
                    f'<td>{esc(c["title"])}</td><td>{esc(c["note"])}</td>'
                    f'<td>{esc(status)}</td></tr>')

    sections = []
    for c in cards:
        body = ""
        if c["error"]:
            body = f'<div class="errbox">이 실험은 갤러리 빌드 중 실패했습니다 (정직 고지):<br>{esc(c["error"])}</div>'
        else:
            body = "".join(f"<li>{esc(m)}</li>" for m in c["metrics"])
            body = f'<ul class="metrics">{body}</ul>'
            body += "".join(render_demo_html(d) for d in c["demos"])
        sections.append(
            f'<div class="card" id="{c["exp"]}">'
            f'<h2><span class="tag">{c["exp"]}</span>{esc(c["title"])}</h2>'
            f'<div class="note">{esc(c["note"])}</div>{body}</div>')

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>D_QR 연구 진화 갤러리</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>D_QR 연구 진화 갤러리</h1>
<div class="sub">원본 데이터 → 인코딩된 D_QR 이미지 → 디코딩 결과 — exp1부터 exp20까지의 진화 (빌드: {datetime.date.today().isoformat()}, py -X utf8 scripts/build_gallery.py)</div>
<table class="timeline">
<tr><th>실험</th><th>이름</th><th>무엇이 바뀌었나</th><th>왕복 검증</th></tr>
{''.join(rows)}
</table>
{''.join(sections)}
<footer>정체성: 오염에 강하고, 좋은 센서일수록 더 깊이 읽히는 단일색 기하 코드 ·
exp5의 z-depth는 강등된 분기, exp11/exp11b는 Honeywell FTO 회피를 위한 디코더 교체(인코딩 포맷은 exp9와 동일) ·
exp12(3층 방어)는 디코더 내부 계층이라 별도 카드 없이 exp11b에 통합 ·
exp14의 스윕 수치는 benchmarks/exp14_holo_sim.py 실측(outputs/exp14/exp14_holo_report.md) ·
exp15의 비교표 수치는 benchmarks/exp15_baseline.py 실측(outputs/exp15/exp15_baseline_report.md) — 비교 대상 심볼로지는 상용 라이브러리(qrcode·cv2·ppf.datamatrix·pylibdmtx) 그대로 ·
exp15b 퇴화 가드 수치는 benchmarks/exp15b_rerun.py 실측(outputs/exp15b/exp15b_report.md) ·
exp16 검출 스윕 수치는 benchmarks/exp16_detect_sweep.py 실측(outputs/exp16/exp16_detect_report.md) — 검출 모듈은 위치 추정 전용, 소거·신뢰도 판정과 무관(FTO 아키텍처 분리) ·
exp17 물리 열화 스윕 수치는 benchmarks/exp17_physical_sweep.py 실측(outputs/exp17/exp17_physical_report.md) — 열화 모듈(src/exp17_physical)은 입력 생성 전용, 인쇄→촬영 간극의 합성 근사이며 실물 촬영은 후속 과제.</footer>
</div>
</body>
</html>"""


def main():
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    cards = []
    total_pass = total_fail = 0

    for exp, title, note, runner in EXPS:
        card = {"exp": exp, "title": title, "note": note, "metrics": [], "demos": [], "error": None}
        try:
            metrics, demos = runner()
            card["metrics"], card["demos"] = metrics, demos
            for d in demos:
                status = "PASS" if d["ok"] else "FAIL"
                if d["ok"]:
                    total_pass += 1
                else:
                    total_fail += 1
                print(f"[{exp}] {d['label']}: {status}")
        except Exception:
            # 한 exp의 실패가 갤러리 전체를 중단시키지 않는다 — 카드에 정직하게 기록
            card["error"] = traceback.format_exc(limit=3)
            total_fail += 1
            print(f"[{exp}] 빌드 실패:\n{card['error']}")
        cards.append(card)

    out = GALLERY_DIR / "index.html"
    out.write_text(build_html(cards), encoding="utf-8")
    print()
    print(f"갤러리 생성: {out.resolve()}")
    print(f"왕복 검증: PASS {total_pass} / FAIL {total_fail}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
