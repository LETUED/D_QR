"""
exp20 테스트 — 점진적 공개 게이트 FTO 이식 (밝기 마진 → 표 일치율)
실행: py -X utf8 tests/test_exp20_consensus_gate.py  (프로젝트 루트에서)

① 클린 왕복: coarse·fine 두 스트림 정확 복원 (프로파일 동결 인코더 재사용)
② 점진적 공개 × 물리 열화: 디포커스·노이즈 증가 시 fine이 coarse보다 먼저
   무너지고(우아한 강등 None), 둘 다 절대 틀린 데이터 반환 안 함 — 핵심 정체성
   "좋은 센서일수록 더 깊이 읽힌다"를 표 일치율(이산)로 구현했는지
③ 미검출 오복호 0: ②의 전 열화 단계에서 반환값은 원본 일치 아니면 None/예외
④ 게이트 단조성: fine_consensus(표차 평균/N)가 열화 증가에 단조 감소 (센서 품질 추적)
⑤ FTO: decoder_v20 소스에 밝기 마진(reflectance/margin/128) 신뢰도 부재,
   DQRDecoderV9 마진 판독 경로 미사용, consensus(이산 표결) 사용
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from src.exp10_progressive.encoder_v10 import DQREncoderV10
from src.exp10_progressive.renderer_v10 import DQRRendererV10
from src.exp17_physical.degrade import defocus_blur, sensor_noise
from src.exp20_consensus_gate import DQRDecoderV20

ROOT = Path(__file__).parent.parent
COARSE = b"CORE-exp20 coarse stream (low-res survivor payload)"
FINE = ("FINE-exp20 fine stream: 고해상 전용 확장 페이로드 — "
        "표 일치율 게이트로 점진적 공개되는 부가 정보.").encode("utf-8")


def _decode(dec, img, info, res):
    return dec.decode_with_stats(
        img, radius=res['radius'], hex_size=info['hex_size'],
        origin_x=info['origin_x'], origin_y=info['origin_y'],
        nsym_c=res['nsym_c'], nsym_f=res['nsym_f'],
        coarse_len=res['coarse_len'], fine_len=res['fine_len'],
        codeword_len_c=res['codeword_len_c'],
        codeword_len_f=res['codeword_len_f'])


def _setup(hex_size=24):
    res = DQREncoderV10().encode(COARSE, FINE)
    img, info = DQRRendererV10(hex_size=hex_size).render(res)
    return res, img, info


def test_clean_roundtrip():
    """① 클린 왕복 — 두 스트림 정확 복원."""
    res, img, info = _setup()
    st = _decode(DQRDecoderV20(), img, info, res)
    ok = st['coarse'] == COARSE and st['fine'] == FINE
    print(f"[테스트 ①] 클린 왕복: {'PASS' if ok else 'FAIL'} "
          f"(coarse={st['coarse'] == COARSE}, fine={st['fine'] == FINE}, "
          f"fine_consensus={st['fine_consensus']:.3f})")
    return ok


def _sweep(degrade_name, levels, degrade_fn):
    """공통: 열화 증가 시 (coarse/fine 상태, fine_consensus, 침묵) 수집."""
    res, img, info = _setup()
    arr = np.array(img.convert('RGB')).astype(np.float32)
    dec = DQRDecoderV20()
    rng = np.random.default_rng(20)
    rows = []
    silent = 0
    for lv in levels:
        a = degrade_fn(arr, lv, rng)
        st = _decode(dec, Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)),
                     info, res)
        cok = st['coarse'] == COARSE
        fok = st['fine'] == FINE
        c_silent = st['coarse'] is not None and not cok
        f_silent = st['fine'] is not None and not fok
        silent += int(c_silent or f_silent)
        rows.append((lv, cok, st['fine'], fok, st['fine_consensus']))
    return rows, silent


def test_progressive_and_silent():
    """②③ 점진적 공개(fine이 coarse보다 먼저 강등) + 미검출 오복호 0."""
    rows, silent = _sweep(
        "defocus", [0, 2, 4, 6, 8, 10, 12],
        lambda a, R, rng: defocus_blur(a, R) if R > 0 else a)

    # coarse는 끝까지 생존(또는 실패 None), 어디서도 WRONG 없음
    coarse_never_wrong = all(c or (c is False) for c, *_ in
                             [(r[1],) for r in rows])  # 형식상; 실 판정은 silent
    # fine이 적어도 한 번 None으로 강등됐고, 그 지점 전에는 coarse OK
    fine_collapsed = any(f is None for _, _, f, _, _ in rows)
    early_coarse_ok = rows[0][1] and rows[0][3]   # 클린에서 둘 다 OK
    # 강등 순서: fine이 None이 되는 첫 지점에서 coarse는 여전히 OK여야 함
    order_ok = True
    for lv, cok, fdata, fok, cons in rows:
        if fdata is None and not cok:
            order_ok = False   # coarse가 fine보다 먼저/동시에 죽으면 점진 공개 아님
            break

    ok = silent == 0 and fine_collapsed and early_coarse_ok and order_ok
    print(f"[테스트 ②③] 점진적 공개 + 침묵 0: {'PASS' if ok else 'FAIL'}")
    for lv, cok, fdata, fok, cons in rows:
        cs = 'OK' if cok else 'FAIL'
        fs = 'OK' if fok else ('WRONG' if fdata is not None else 'None')
        print(f"          디포커스 R={lv:>2}: coarse {cs:>4}, fine {fs:>5}, "
              f"fine_consensus={cons:.3f}")
    print(f"          → 미검출 오복호 {silent}건, fine 강등 발생={fine_collapsed}, "
          f"강등 순서(fine 먼저)={order_ok}")
    return ok


def test_gate_monotonic():
    """④ fine_consensus가 열화 증가에 단조 감소 (센서 품질 추적)."""
    rows, _ = _sweep(
        "noise", [0.0, 6.0, 12.0, 20.0, 30.0],
        lambda a, s, rng: (sensor_noise(a, 0.0, s, rng) if s > 0 else a))
    cons = [c for *_, c in rows]
    # 엄밀 단조까지는 노이즈 랜덤성으로 흔들릴 수 있어 '전반적 비증가' 판정
    mono = all(cons[i] >= cons[i + 1] - 0.05 for i in range(len(cons) - 1))
    ok = mono and cons[0] >= cons[-1]
    print(f"[테스트 ④] 게이트 단조성: {'PASS' if ok else 'FAIL'} "
          f"(읽기σ 0→30: fine_consensus {cons[0]:.3f}→{cons[-1]:.3f})")
    return ok


def test_fto_no_margin():
    """⑤ FTO — decoder_v20에 밝기 마진 신뢰도 부재, 표 일치율 사용."""
    src = (ROOT / "src" / "exp20_consensus_gate" /
           "decoder_v20.py").read_text(encoding='utf-8')
    # docstring(옛 마진 방식을 설명하는 WHY 주석)·# 주석 제거 → 실행 코드만 검사
    no_comments = "\n".join(ln for ln in src.splitlines()
                            if not ln.lstrip().startswith('#'))
    code = re.sub(r'"""(?:.|\n)*?"""', '', no_comments)   # 삼중따옴표 블록 제거
    # 실행 코드에 밝기 마진 판독 경로(DQRDecoderV9 임포트/호출, _read_cell_soft,
    # margin/128, reflectance/contrast) 부재
    forbidden = re.search(r"_read_cell_soft\(|/\s*128|DQRDecoderV9|"
                          r"reflectance|contrast", code)
    uses_consensus = ("triangle_sample_points" in code
                      and "global_otsu_threshold" in code
                      and "2 * dark_votes" in code)
    ok = forbidden is None and uses_consensus
    print(f"[테스트 ⑤] FTO 마진 부재·표결 사용: {'PASS' if ok else 'FAIL'} "
          f"(실행코드 마진경로 부재={forbidden is None}, 표결사용={uses_consensus})")
    return ok


if __name__ == "__main__":
    results = [
        test_clean_roundtrip(),
        test_progressive_and_silent(),
        test_gate_monotonic(),
        test_fto_no_margin(),
    ]
    print()
    if all(results):
        print("=== exp20 전체 테스트 PASS ===")
    else:
        print("=== exp20 테스트 FAIL ===")
        sys.exit(1)
