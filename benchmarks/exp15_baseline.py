"""
exp15 — 외부 기준선 비교: QR(ECC-H)·DataMatrix(ECC200) vs D_QR(프로파일 A + v11b)

D_QR의 "오염 내성 + 무오정정" 주장을 상용 심볼로지 구현과 동일 조건에서 검증한다.
핵심 차별 지표는 성공률이 아니라 **침묵 오정정률** — 디코더가 무언가를
반환했는데 원본과 다른 경우(cv2/pylibdmtx가 오염에서 틀린 페이로드를 내는가).

공정 비교 규약:
  - 동일 페이로드 (한국어+영문 UTF-8, 약 60바이트)
  - 동일 점유 면적: 각 심볼(자기 규격의 quiet zone 포함)을 동일 픽셀 면적
    (640×640 상당)으로 LANCZOS 정규화 리사이즈 → 800×800 백색 캔버스 중앙 배치
  - ECC 수준: QR은 ECC-H, DataMatrix는 ECC200 고정 수준, D_QR은 프로파일 A
    (nsym=255×30%) — ECC 오버헤드 차이는 보고서에 명시
  - D_QR 판독은 기하 파라미터를 벤치마크가 제공(자체 검출 없음) — 비대칭은
    보고서 공정성 단락에 명시

열화 채널 (exp9/exp11b와 동일 모델 add_noise/add_blur/add_patches, seed=42):
  ① scatter — 산포 오염 패치 20~90개 (patch_size=60, 임계 근처 회색)
  ① blot    — 군집 블롯 1~3개 (160~360px 균일 회색 — exp11b 테스트 ⑤ 모델)
  ② noise   — 가우시안 노이즈 σ60~140
  ③ blur    — 가우시안 블러 1~10px

측정 (조건당 N=100): 성공률 / 침묵 오정정률 / 거부율 + Wilson 95% CI.

실행 (프로젝트 루트에서):
  py -X utf8 benchmarks/exp15_baseline.py                     # 전체 + 집계
  py -X utf8 benchmarks/exp15_baseline.py --channel noise     # 한 채널만
  py -X utf8 benchmarks/exp15_baseline.py --finalize          # 집계·차트·보고서
  py -X utf8 benchmarks/exp15_baseline.py --trials 10         # 빠른 점검

산출물: outputs/exp15/exp15_baseline_chart.png
        outputs/exp15/exp15_baseline_report.md
        outputs/exp15/exp15_baseline_summary.json
        outputs/exp15/partial/*.json (조건별 중간 저장 — 재실행 시 건너뜀)
"""
import sys
import json
import math
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

# 열화 모델은 exp9/exp11b 벤치마크와 동일 조건이어야 하므로 그대로 가져온다
from benchmarks.exp9_benchmark import add_noise, add_blur, add_patches

from src.exp9_soft.encoder_v9 import DQREncoderV9
from src.exp9_soft.renderer_v9 import DQRRendererV9
from src.exp11_consensus.decoder_v11b import DQRDecoderV11B

TEXT = "D_QR exp15 기준선 비교 — 오염 내성 페이로드!"
PAYLOAD = TEXT.encode('utf-8')          # 59 bytes

TRIALS = 100
SEED = 42                               # exp11b와 동일 시드 체계 (SEED + trial)
CANVAS = 800
SYMBOL_AREA = 640 * 640                 # 동일 인쇄 면적 (quiet zone 포함)
DM_TIMEOUT_MS = 1500                    # pylibdmtx 시행당 상한 — 보고서에 명시

OUT_DIR = Path('outputs/exp15')
PARTIAL_DIR = OUT_DIR / 'partial'

# ── 기준선 라이브러리 가용성 (실패해도 전체 중단하지 않는다) ─────────────

AVAILABILITY = {}   # name -> 'ok' | 사유 문자열


def _try_import(name, fn):
    try:
        mod = fn()
        AVAILABILITY[name] = 'ok'
        return mod
    except Exception as e:
        AVAILABILITY[name] = f"{type(e).__name__}: {e}"
        return None

_qrcode = _try_import('qrcode', lambda: __import__('qrcode'))
_cv2 = _try_import('opencv (cv2)', lambda: __import__('cv2'))
_pyzbar = None
if _cv2 is None:
    _pyzbar = _try_import('pyzbar',
                          lambda: __import__('pyzbar.pyzbar',
                                             fromlist=['decode']))


def _import_ppf():
    from ppf.datamatrix import DataMatrix
    return DataMatrix

_DataMatrix = _try_import('ppf.datamatrix', _import_ppf)


def _import_dmtx():
    from pylibdmtx.pylibdmtx import decode as dm_decode
    return dm_decode

_dm_decode = _try_import('pylibdmtx', _import_dmtx)


# ── 심볼 생성 (동일 면적 정규화) ─────────────────────────────────────────

def _normalize_to_canvas(sym_img):
    """심볼 이미지를 SYMBOL_AREA 면적으로 리사이즈 후 캔버스 중앙 배치."""
    w0, h0 = sym_img.size
    s = math.sqrt(SYMBOL_AREA / (w0 * h0))
    nw, nh = round(w0 * s), round(h0 * s)
    resized = sym_img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new('RGB', (CANVAS, CANVAS), (255, 255, 255))
    dx, dy = (CANVAS - nw) // 2, (CANVAS - nh) // 2
    canvas.paste(resized, (dx, dy))
    return canvas, (nw / w0, nh / h0, dx, dy)


def build_qr():
    """QR ECC-H (qrcode 생성, cv2 또는 pyzbar 판독)."""
    if _qrcode is None or (_cv2 is None and _pyzbar is None):
        return None
    import qrcode
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H,
                       box_size=10, border=4)   # border 4모듈 = QR 규격 quiet
    qr.add_data(PAYLOAD, optimize=0)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white').convert('RGB')
    canvas, (sx, _, _, _) = _normalize_to_canvas(img)

    meta = {'symbology': f'QR v{qr.version} ECC-H',
            'modules': qr.modules_count, 'native_px': img.size[0],
            'feature_px': round(10 * sx, 1)}   # 정규화 후 모듈 한 변
    try:    # 코드워드 구성(데이터/전체) — 실패해도 벤치마크는 계속
        from qrcode.base import rs_blocks
        blocks = rs_blocks(qr.version, qrcode.constants.ERROR_CORRECT_H)
        total = sum(b.total_count for b in blocks)
        data = sum(b.data_count for b in blocks)
        meta.update(total_codewords=total, data_codewords=data,
                    ecc_share=round(100 * (total - data) / total, 1))
    except Exception:
        pass

    if _cv2 is not None:
        det = _cv2.QRCodeDetector()

        def read(image):
            try:
                txt, pts, _ = det.detectAndDecode(
                    np.array(image.convert('L')))
                if pts is None or txt == '':
                    return None
                return txt.encode('utf-8')
            except Exception:
                return None
        meta['reader'] = 'cv2.QRCodeDetector'
    else:
        def read(image):
            try:
                res = _pyzbar.decode(image)
                return res[0].data if res else None
            except Exception:
                return None
        meta['reader'] = 'pyzbar'
    return {'read': read, 'canvas': canvas, 'meta': meta}


def _dm_codewords(data: bytes) -> bytes:
    """
    DataMatrix ECC200 ascii 코드워드 인코딩 (upper shift 포함).

    ppf.datamatrix의 ascii 코덱은 확장 ASCII(>127)를 지원하지 않아
    한국어 UTF-8 바이트를 못 싣는다 → ISO/IEC 16022 표준 upper shift
    (235 + (b-128)+1)로 직접 인코딩해 주입한다. 정당성은 pylibdmtx
    왕복(스모크 테스트)으로 검증된다.
    """
    out = []
    for b in data:
        if b < 128:
            out.append(b + 1)
        else:
            out.append(235)
            out.append((b - 128) + 1)
    return bytes(out)


class _RawCodewords:
    """ppf.datamatrix가 호출하는 message.encode(codec)에 코드워드를 그대로 공급."""

    def __init__(self, cw):
        self._cw = cw

    def encode(self, _codec):
        return self._cw


def _dm_symbol_params(n_codewords):
    """ppf.datamatrix 정사각 심볼 선택 로직 복제 → (변 모듈수, 총 cw, 검사 cw)."""
    k = [5, 7, 10, 12, 14, 18, 20, 24, 28, 36, 42, 48, 56, 68,
         84, 112, 144, 192, 224, 272, 336, 408, 496, 620]
    w = h = 6
    inc, j = 2, -1
    while True:
        j += 1
        if j == len(k):
            raise ValueError('too long')
        if w > 11 * inc:
            inc = 4 + inc & 12
        h += inc
        w = h
        bc = (w * h) >> 3
        if bc - k[j] >= n_codewords:
            return w, bc, k[j]


def build_dm():
    """DataMatrix ECC200 (ppf.datamatrix 생성, pylibdmtx 판독)."""
    if _DataMatrix is None or _dm_decode is None:
        return None
    cw = _dm_codewords(PAYLOAD)
    dm = _DataMatrix('x')               # 자리표시 — 코드워드 직접 주입
    dm.message = _RawCodewords(cw)
    dm.codecs = ['datamatrix.ascii']
    m = np.array(dm.matrix, dtype=np.uint8)

    box, quiet = 10, 1                  # quiet 1모듈 = ECC200 규격 최소
    big = np.kron(1 - m, np.ones((box, box), dtype=np.uint8)) * 255
    h, w = big.shape
    arr = np.full((h + 2 * quiet * box, w + 2 * quiet * box), 255,
                  dtype=np.uint8)
    arr[quiet * box:quiet * box + h, quiet * box:quiet * box + w] = big
    img = Image.fromarray(arr).convert('RGB')
    canvas, (sx, _, _, _) = _normalize_to_canvas(img)

    side, total_cw, ecc_cw = _dm_symbol_params(len(cw))
    meta = {'symbology': f'DataMatrix {m.shape[1]}x{m.shape[0]} ECC200',
            'modules': m.shape[0], 'native_px': img.size[0],
            'feature_px': round(box * sx, 1),   # 정규화 후 모듈 한 변
            'total_codewords': total_cw, 'ecc_codewords': ecc_cw,
            'ecc_share': round(100 * ecc_cw / total_cw, 1),
            'reader': f'pylibdmtx (timeout {DM_TIMEOUT_MS}ms)'}

    def read(image):
        try:
            res = _dm_decode(image, timeout=DM_TIMEOUT_MS, max_count=1)
            return res[0].data if res else None
        except Exception:
            return None
    return {'read': read, 'canvas': canvas, 'meta': meta}


def build_dqr():
    """D_QR 프로파일 A (exp9 인코더/렌더러 + exp11b v11b 3층 디코더)."""
    enc = DQREncoderV9(profile='A')
    ren = DQRRendererV9(hex_size=32)
    result = enc.encode(PAYLOAD)
    img, info = ren.render(result)
    canvas, (sx, sy, dx, dy) = _normalize_to_canvas(img)

    # 면적 정규화 리사이즈에 맞춰 기하 파라미터를 비례 변환
    params = dict(
        radius=result['radius'],
        hex_size=info['hex_size'] * sx,
        bits_per_cell=result['bits_per_cell'],
        origin_x=info['origin_x'] * sx + dx,
        origin_y=info['origin_y'] * sy + dy,
        nsym=result['nsym'],
        data_len=result['data_len'],
        codeword_len=result['codeword_len'],
    )
    dec = DQRDecoderV11B()

    meta = {'symbology': f"D_QR 프로파일 A (radius {result['radius']}, "
                         f"6bit/셀) + v11b",
            'native_px': img.size[0],
            'feature_px': round(info['hex_size'] * sx, 1),  # 정규화 후 hex 변
            'total_codewords': result['codeword_len'],
            'ecc_codewords': result['nsym'],
            'ecc_share': round(100 * result['nsym']
                               / result['codeword_len'], 1),
            'reader': 'DQRDecoderV11B (기하 파라미터 제공)'}

    def read(image):
        try:
            return dec.decode_with_stats(image, **params)['data']
        except Exception:
            return None
    # params 노출 — exp15b 가드 단위 테스트가 동일 기하로 재현하는 데 사용
    return {'read': read, 'canvas': canvas, 'meta': meta, 'params': params}


SYM_BUILDERS = [('qr', build_qr), ('dm', build_dm), ('dqr', build_dqr)]
SYM_LABELS = {'qr': 'QR ECC-H', 'dm': 'DataMatrix ECC200',
              'dqr': 'D_QR A+v11b'}
SYM_COLORS = {'qr': 'steelblue', 'dm': 'darkorange', 'dqr': 'crimson'}


def build_symbols():
    """가용한 심볼로지만 생성 — 실패한 기준선은 보고서에 '비교 불가' 표기."""
    syms = {}
    for key, fn in SYM_BUILDERS:
        try:
            s = fn()
        except Exception as e:
            AVAILABILITY[f'{key} (생성)'] = f"{type(e).__name__}: {e}"
            s = None
        if s is not None:
            syms[key] = s
    return syms


# ── 열화 채널 ────────────────────────────────────────────────────────────

def add_blots(img, n_blots, rng):
    """군집 블롯: 160~360px 균일 회색 정사각 — exp11b 테스트 ⑤ 모델의 무작위판."""
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    for _ in range(n_blots):
        size = int(rng.integers(160, 361))
        x = int(rng.integers(0, max(1, w - size)))
        y = int(rng.integers(0, max(1, h - size)))
        gray = int(rng.integers(100, 156))   # 임계 128 근처 = 가장 악성
        arr[y:y + size, x:x + size] = gray
    return Image.fromarray(arr)


CHANNELS = {
    'scatter': dict(levels=[20, 40, 60, 90],
                    fn=lambda img, lv, rng: add_patches(img, lv, rng,
                                                        patch_size=60),
                    title='국소 오염 — 산포 패치', xlabel='패치 개수'),
    'blot':    dict(levels=[1, 2, 3],
                    fn=lambda img, lv, rng: add_blots(img, lv, rng),
                    title='국소 오염 — 군집 블롯', xlabel='블롯 개수'),
    'noise':   dict(levels=[60, 80, 100, 120, 140],
                    fn=lambda img, lv, rng: add_noise(img, lv, rng),
                    title='가우시안 노이즈', xlabel='sigma'),
    'blur':    dict(levels=[1, 2, 4, 6, 8, 10],
                    fn=lambda img, lv, rng: add_blur(img, lv),
                    title='가우시안 블러', xlabel='radius (px)'),
}


def classify(out):
    """'ok' | 'wrong'(침묵 오정정) | 'fail'(거부)."""
    if out is None or len(out) == 0:
        return 'fail'
    return 'ok' if out == PAYLOAD else 'wrong'


def wilson(k, n, z=1.96):
    """Wilson score 95% CI (%, 양끝)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, centre - hw), 100 * min(1.0, centre + hw)


def partial_path(channel, level):
    return PARTIAL_DIR / f"{channel}_{level}.json"


def run_condition(channel, level, syms, trials=TRIALS):
    """한 (채널, 레벨) 조건: 모든 심볼로지를 동일 열화(동일 시드)로 측정."""
    path = partial_path(channel, level)
    if path.exists():
        print(f"  [{channel} lv={level}] 이미 측정됨 — 건너뜀")
        return
    cfg = CHANNELS[channel]
    cnt = {k: {'ok': 0, 'wrong': 0, 'fail': 0} for k in syms}
    t0 = time.time()
    for t in range(trials):
        for key, sym in syms.items():
            # 심볼로지별로 rng를 재생성 → 동일 (시드, 시행)에 동일 열화 파라미터
            rng = np.random.default_rng(SEED + t)
            deg = cfg['fn'](sym['canvas'], level, rng)
            cnt[key][classify(sym['read'](deg))] += 1
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'trials': trials, 'counts': cnt},
                               ensure_ascii=False), encoding='utf-8')
    msg = "  ".join(
        f"{k} {100 * cnt[k]['ok'] / trials:.0f}%"
        + (f"(W{cnt[k]['wrong']})" if cnt[k]['wrong'] else "")
        for k in cnt)
    print(f"  [{channel} lv={level}] {msg}  ({time.time() - t0:.0f}s)",
          flush=True)


# ── 집계·차트·보고서 ─────────────────────────────────────────────────────

def load_all(sym_keys):
    res = {}
    for ch, cfg in CHANNELS.items():
        for lv in cfg['levels']:
            path = partial_path(ch, lv)
            if not path.exists():
                raise FileNotFoundError(f"부분 결과 없음: {path} — "
                                        f"--channel {ch} 먼저 실행")
            saved = json.loads(path.read_text(encoding='utf-8'))
            for k in sym_keys:
                if k not in saved['counts']:
                    raise KeyError(f"{path}에 {k} 측정 없음")
            res[(ch, lv)] = saved
    return res


def make_chart(res, sym_keys, syms_meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    flat = axes.ravel()

    # 패널 0~3: 채널별 성공률 곡선 (Wilson CI 오차막대)
    for ax, (ch, cfg) in zip(flat, CHANNELS.items()):
        xs = cfg['levels']
        for k in sym_keys:
            rates, lo_err, hi_err = [], [], []
            for lv in xs:
                saved = res[(ch, lv)]
                n = saved['trials']
                ok = saved['counts'][k]['ok']
                r = 100 * ok / n
                lo, hi = wilson(ok, n)
                rates.append(r)
                lo_err.append(max(0.0, r - lo))   # 부동소수 음수 방지
                hi_err.append(max(0.0, hi - r))
            ax.errorbar(xs, rates, yerr=[lo_err, hi_err], marker='o',
                        capsize=3, color=SYM_COLORS[k], label=SYM_LABELS[k])
        ax.set_title(cfg['title'])
        ax.set_xlabel(cfg['xlabel'])
        ax.set_ylabel('성공률 (%) — Wilson 95% CI')
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    n_total = sum(res[(ch, lv)]['trials']
                  for ch, cfg in CHANNELS.items() for lv in cfg['levels'])

    # 패널 4: 침묵 오정정 총계 (핵심 차별 지표)
    ax = flat[4]
    tot_w = {k: sum(res[(ch, lv)]['counts'][k]['wrong']
                    for ch, cfg in CHANNELS.items() for lv in cfg['levels'])
             for k in sym_keys}
    bars = ax.bar([SYM_LABELS[k] for k in sym_keys],
                  [tot_w[k] for k in sym_keys],
                  color=[SYM_COLORS[k] for k in sym_keys])
    for b, k in zip(bars, sym_keys):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                str(tot_w[k]), ha='center', fontsize=11, fontweight='bold')
    ax.set_title(f'침묵 오정정 총계 — 틀린 페이로드를 성공으로 반환 '
                 f'(전 조건 합산, {n_total:,} 이미지/심볼로지)')
    ax.set_ylabel('건수')
    ax.grid(alpha=0.3, axis='y')

    # 패널 5: 채널별 침묵 오정정 분해
    ax = flat[5]
    chs = list(CHANNELS)
    width = 0.8 / len(sym_keys)
    for i, k in enumerate(sym_keys):
        ws = [sum(res[(ch, lv)]['counts'][k]['wrong']
                  for lv in CHANNELS[ch]['levels']) for ch in chs]
        pos = [j + (i - (len(sym_keys) - 1) / 2) * width
               for j in range(len(chs))]
        bars = ax.bar(pos, ws, width=width, color=SYM_COLORS[k],
                      label=SYM_LABELS[k])
        for b, w_ in zip(bars, ws):
            if w_:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                        str(w_), ha='center', fontsize=8)
    ax.set_xticks(range(len(chs)))
    ax.set_xticklabels([CHANNELS[c]['title'] for c in chs], fontsize=8)
    ax.set_title('채널별 침묵 오정정 건수')
    ax.set_ylabel('건수')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=8)

    fig.suptitle('exp15: QR(ECC-H)·DataMatrix(ECC200) vs D_QR(A+v11b) — '
                 f'동일 페이로드({len(PAYLOAD)}B)·동일 면적·동일 열화·'
                 f'N={TRIALS}/조건', fontsize=14)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = OUT_DIR / 'exp15_baseline_chart.png'
    fig.savefig(chart_path, dpi=110)
    print(f"차트 저장: {chart_path}")


def fmt_cell(c, n):
    """성공률 [CI] (W/F 건수) 셀 포맷."""
    r = 100 * c['ok'] / n
    lo, hi = wilson(c['ok'], n)
    s = f"{r:.0f}% [{lo:.1f}, {hi:.1f}]"
    if c['wrong']:
        s += f" **W{c['wrong']}**"
    return s


def make_report(res, sym_keys, syms):
    n_conds = sum(len(cfg['levels']) for cfg in CHANNELS.values())
    n_total = n_conds * TRIALS
    tot = {k: {'ok': 0, 'wrong': 0, 'fail': 0} for k in sym_keys}
    for saved in res.values():
        for k in sym_keys:
            for o in ('ok', 'wrong', 'fail'):
                tot[k][o] += saved['counts'][k][o]

    missing = [k for k, _ in SYM_BUILDERS if k not in sym_keys]

    lines = [
        "# exp15 — 외부 기준선 비교: QR·DataMatrix vs D_QR "
        "(오염 내성 + 무오정정)",
        "",
        f"- 페이로드: `{TEXT}` ({len(PAYLOAD)} bytes UTF-8, 한국어+영문)",
        f"- 조건 {n_conds}개 × N={TRIALS} (조건당), seed={SEED} — "
        "exp9/exp11b와 동일 열화 모델·동일 시드 체계",
        "- 동일 점유 면적: 각 심볼(자기 규격 quiet zone 포함)을 "
        f"{int(math.sqrt(SYMBOL_AREA))}×{int(math.sqrt(SYMBOL_AREA))}px "
        f"상당 면적으로 LANCZOS 정규화 → {CANVAS}×{CANVAS} 백색 캔버스 "
        "중앙 배치",
        "- **침묵 오정정(W) = 디코더가 무언가를 반환했는데 원본과 다름** — "
        "핵심 차별 지표",
        "",
        "## 기준선 가용성",
        "",
        "| 라이브러리 | 상태 |",
        "|-----------|------|",
    ]
    for name, status in AVAILABILITY.items():
        lines.append(f"| {name} | {'사용 가능' if status == 'ok' else status} |")
    for k in missing:
        lines.append(f"| **{SYM_LABELS[k]}** | **비교 불가 — 생성/판독 "
                     "라이브러리 실패 (위 사유 참조)** |")
    lines += [
        "",
        "## 심볼 구성과 ECC 오버헤드 (동일하지 않음 — 명시)",
        "",
        "| 심볼로지 | 구성 | 총 코드워드 | ECC 코드워드 | ECC 비중 | 판독기 |",
        "|---------|------|------------|-------------|---------|--------|",
    ]
    for k in sym_keys:
        m = syms[k]['meta']
        total = m.get('total_codewords', '?')
        if k == 'qr':
            ecc = (m['total_codewords'] - m['data_codewords']
                   if 'data_codewords' in m else '?')
        elif k == 'dqr':
            ecc = m['ecc_codewords']
        else:
            ecc = m.get('ecc_codewords', '?')
        share = m.get('ecc_share', '?')
        lines.append(f"| {SYM_LABELS[k]} | {m['symbology']} | {total} | "
                     f"{ecc} | {share}% | {m['reader']} |")
    lines += [
        "",
        "ECC 수준이 같지 않다는 점에 주의: QR은 최고 수준(H, 코드워드 "
        "약 30% 복원), DataMatrix ECC200은 심볼 크기에 따라 고정된 단일 "
        "수준(이 크기에서 ECC 비중 위 표 참조 — 선택 불가), D_QR 프로파일 "
        "A는 nsym=76 (이 페이로드 길이에서는 코드워드의 절반 이상이 "
        "패리티). 짧은 페이로드에서 D_QR의 실효 ECC 비중이 가장 높은 축에 "
        "속하므로, 성공률 우위 해석 시 이 차이를 감안해야 한다. 반대로 "
        "침묵 오정정은 ECC 양이 아니라 **수락 규칙**의 문제다 — ECC가 "
        "많아도 잘못된 정정을 수락하면 오정정이 발생한다.",
        "",
        "## 공정성 논의",
        "",
        "이 벤치마크에서 D_QR 판독은 기하 파라미터(격자 반경·셀 크기·"
        "원점)를 벤치마크로부터 제공받는다. QR·DataMatrix 디코더는 파인더 "
        "패턴 기반 자체 검출을 수행하므로, 이들의 실패에는 '검출 실패'가 "
        "포함되고 D_QR에는 포함되지 않는다 — **이 비대칭은 D_QR에 유리**하며 "
        "D_QR 성공률은 검출 비용을 제외한 상한으로 읽어야 한다 (D_QR의 "
        "메타데이터 없는 자기기술 검출은 exp13에서 별도 검증 중이며 아직 "
        "공간 소거 스윕과 통합되지 않았다). 반면 (1) 열화는 세 심볼로지에 "
        "동일 시드·동일 파라미터로 가해지고, (2) 점유 면적이 동일하며, "
        "(3) 침묵 오정정 지표는 검출 단계와 무관하게 '반환했으면 책임진다' "
        "기준이므로 검출 비대칭의 영향을 받지 않는다 — 오히려 검출 실패가 "
        "많을수록 반환 자체가 줄어 오정정 기회도 준다. pylibdmtx에는 "
        f"시행당 {DM_TIMEOUT_MS}ms 탐색 상한을 두었다(무한 탐색 방지 — "
        "거부율에 보수적으로 작용할 수 있음).",
        "",
        "면적 정규화의 부수 효과로 정규화 후 최소 특성 크기가 심볼로지마다 "
        "다르다: " + ", ".join(
            f"{SYM_LABELS[k]} {syms[k]['meta'].get('feature_px', '?')}px"
            + ('(모듈)' if k != 'dqr' else '(hex 변, 비트 단위인 삼각형은 '
               '이보다 작음)')
            for k in sym_keys)
        + ". D_QR은 셀당 6비트를 삼각형에 싣는 고밀도 기하라 동일 면적에서 "
        "비트당 특성이 가장 작다 — 블러 채널의 D_QR 열세는 상당 부분 이 "
        "기하적 트레이드오프(밀도 ↔ 블러 내성)의 직접 비용이며, exp11b "
        "네이티브 해상도(hex 32px)에서는 9.5px 블러까지 견뎠다.",
        "",
        "## 결과 총괄",
        "",
        f"| 심볼로지 | 성공 | **침묵 오정정** | 거부 | 오정정률 "
        f"(전 {n_total:,}회 중) |",
        "|---------|------|---------------|------|----------------|",
    ]
    for k in sym_keys:
        t = tot[k]
        lo, hi = wilson(t['wrong'], n_total)
        lines.append(f"| {SYM_LABELS[k]} | {t['ok']:,} | **{t['wrong']:,}** "
                     f"| {t['fail']:,} | {100 * t['wrong'] / n_total:.2f}% "
                     f"[{lo:.2f}, {hi:.2f}] |")
    lines.append("")

    for ch, cfg in CHANNELS.items():
        lines.append(f"## {cfg['title']} ({cfg['xlabel']})")
        lines.append("")
        header = "| 레벨 | " + " | ".join(
            f"{SYM_LABELS[k]} 성공률 [95% CI]" for k in sym_keys) + " |"
        lines.append(header)
        lines.append("|------|" + "|".join(["-----"] * len(sym_keys)) + "|")
        for lv in cfg['levels']:
            saved = res[(ch, lv)]
            n = saved['trials']
            cells = [fmt_cell(saved['counts'][k], n) for k in sym_keys]
            lines.append(f"| {lv} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("W = 침묵 오정정 건수 (성공으로 위장한 틀린 반환), "
                     "표기 없는 잔여분은 거부.")
        lines.append("")

    report_path = OUT_DIR / 'exp15_baseline_report.md'
    return lines, report_path, tot, n_total


def finalize():
    syms = build_symbols()
    sym_keys = list(syms)
    res = load_all(sym_keys)
    make_chart(res, sym_keys, {k: syms[k]['meta'] for k in sym_keys})
    lines, report_path, tot, n_total = make_report(res, sym_keys, syms)

    # 정직한 해석 단락 — 실측에서 D_QR이 지는 구간을 자동 추출해 명시
    losing = []
    for ch, cfg in CHANNELS.items():
        for lv in cfg['levels']:
            saved = res[(ch, lv)]
            n = saved['trials']
            if 'dqr' not in saved['counts']:
                continue
            dqr_r = saved['counts']['dqr']['ok'] / n
            for k in sym_keys:
                if k == 'dqr':
                    continue
                base_r = saved['counts'][k]['ok'] / n
                if base_r - dqr_r >= 0.10:   # 10%p 이상 열세만 명시
                    losing.append(
                        f"- {cfg['title']} {cfg['xlabel']}={lv}: "
                        f"{SYM_LABELS[k]} {100 * base_r:.0f}% vs D_QR "
                        f"{100 * dqr_r:.0f}% ({100 * (base_r - dqr_r):.0f}%p "
                        f"열세)")
    lines += [
        "## D_QR이 지는 구간 (정직 보고)",
        "",
    ]
    lines += (losing if losing
              else ["- 측정 조건에서 10%p 이상 열세 구간 없음 (단, ECC "
                    "오버헤드·검출 비대칭 차이를 감안할 것 — 위 공정성 논의)"])
    # D_QR 침묵 오정정이 관측되면 조건별 분해를 명시 (정직 보고)
    dqr_wrong_conds = [
        (ch, lv, res[(ch, lv)]['counts']['dqr']['wrong'])
        for ch, cfg in CHANNELS.items() for lv in cfg['levels']
        if 'dqr' in res[(ch, lv)]['counts']
        and res[(ch, lv)]['counts']['dqr']['wrong'] > 0]
    if dqr_wrong_conds:
        lines += [
            "",
            "## D_QR 침묵 오정정 분석 — 전부-0 코드워드 퇴화 (정직 보고)",
            "",
            "측정에서 D_QR(v11b)에 침묵 오정정이 발생했다: "
            + ", ".join(f"{CHANNELS[ch]['title']} {CHANNELS[ch]['xlabel']}"
                        f"={lv}에서 {w}건" for ch, lv, w in dqr_wrong_conds)
            + ". 전수 재현 검사 결과 **반환값이 모두 전부-0 페이로드(59바이트 "
            "0x00)로 동일한 단일 퇴화 모드**였다 (blot=2 시행 8; blot=3 시행 "
            "3·6·8·13·54·57 — 한 사례 정밀 추적: mode=soft, 소거 25, errata "
            "48).",
            "",
            "원인: RS 부호는 선형이므로 **전부-0 코드워드는 항상 유효**하고, "
            "재인코딩 대조(2층)도 encode(0…0)=0…0이므로 통과한다. 군집 "
            "블롯 2~3개(개당 160~360px, 밝은 회색)가 심볼 대부분을 덮으면 "
            "덮인 셀이 전부 비트 0으로 읽혀 판독 코드워드가 전부-0 근방으로 "
            "이동하고, 소거 사다리가 잔여 진짜 데이터를 '오류'로 정정해 "
            "전부-0 흡인자에 수렴한다. exp11b의 3층 방어(소거 사다리 상한·"
            "재인코딩 대조·errata 예산)는 이 퇴화를 막지 못한다 — exp11b "
            "벤치마크(2,000 이미지)에서 0건이었던 이유는 그 열화 모델에 "
            "'심볼 대부분을 덮는 밝은 군집 오염'이 없었기 때문이다. 또한 "
            "이 7건에는 검출 비대칭도 기여한다 — QR·DataMatrix는 같은 "
            "이미지에서 파인더가 파괴되어 시도 자체를 거부했지만, D_QR은 "
            "기하 파라미터를 제공받아 거의 빈 심볼을 끝까지 복호했다. "
            "그럼에도 '반환했으면 책임진다' 기준에서 변명이 되지 않으므로 "
            "오정정으로 정직하게 계수한다.",
            "",
            "후속 과제(디코더 측, exp15 범위 밖): 전부-0(또는 상수) 코드워드 "
            "거부 규칙. 이는 특정 퇴화 출력에 대한 가드이지 페이로드 "
            "체크섬(CRC, FTO 제약 3)이 아니며, 밝기 연속량도 쓰지 않으므로 "
            "FTO 안전하다. 인코더 측 대안으로는 QR의 마스킹에 해당하는 "
            "전부-0 회피 스크램블이 있다.",
        ]
    lines += [
        "",
        "## 결론",
        "",
        f"- 전 {n_total:,}회/심볼로지 측정에서 침묵 오정정: D_QR(v11b) "
        f"**{tot.get('dqr', {}).get('wrong', '?')}건** vs "
        + ", ".join(f"{SYM_LABELS[k]} {tot[k]['wrong']}건"
                    for k in sym_keys if k != 'dqr') + ".",
        "- D_QR의 포지셔닝 주장은 '성공률 우위'가 아니라 **무오정정**(틀린 "
        "데이터를 절대 성공으로 반환하지 않음)이다 — 성공률에서 지는 "
        "구간이 있어도 이 주장과 모순되지 않는다."
        + (" 단, 위 전부-0 퇴화 모드가 해소되기 전까지 무오정정 주장에는 "
           "'전부-0 거부 가드 적용 전' 단서가 붙어야 한다."
           if dqr_wrong_conds else ""),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"보고서 저장: {report_path}")

    summary = {
        'payload_bytes': len(PAYLOAD),
        'trials_per_condition': TRIALS,
        'seed': SEED,
        'availability': AVAILABILITY,
        'meta': {k: syms[k]['meta'] for k in sym_keys},
        'totals': tot,
        'conditions': {
            f"{ch}/{lv}": res[(ch, lv)]
            for ch, cfg in CHANNELS.items() for lv in cfg['levels']
        },
    }
    (OUT_DIR / 'exp15_baseline_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"요약 저장: {OUT_DIR / 'exp15_baseline_summary.json'}")
    print("\n=== 총괄 ===")
    for k in sym_keys:
        print(f"  {SYM_LABELS[k]:18s} ok={tot[k]['ok']:5d} "
              f"wrong={tot[k]['wrong']:4d} fail={tot[k]['fail']:5d}")


def main():
    global TRIALS
    parser = argparse.ArgumentParser(description='exp15 외부 기준선 비교')
    parser.add_argument('--channel', help='한 채널만 (scatter|blot|noise|blur)')
    parser.add_argument('--trials', type=int, default=TRIALS)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()

    TRIALS = args.trials

    if args.finalize:
        finalize()
        return

    print("기준선 가용성:")
    syms = build_symbols()
    for name, status in AVAILABILITY.items():
        print(f"  {name}: {status}")
    print(f"측정 심볼로지: {list(syms)}  (페이로드 {len(PAYLOAD)}B, "
          f"N={TRIALS}/조건)")

    channels = [args.channel] if args.channel else list(CHANNELS)
    for ch in channels:
        if ch not in CHANNELS:
            raise SystemExit(f"알 수 없는 채널: {ch}")
        print(f"\n=== 채널 {ch} ===")
        for lv in CHANNELS[ch]['levels']:
            run_condition(ch, lv, syms, trials=TRIALS)

    if not args.channel:
        finalize()


if __name__ == "__main__":
    main()
