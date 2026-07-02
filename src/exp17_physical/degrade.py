"""
exp17 — 물리 기반 카메라 열화 모델 (인쇄→촬영 간극 합성 근사)

exp16 검출 스윕은 1,400회 전부 '프로그램 변환'(회전·스케일·원근)만 거친
이상적 장면이었다. 실제 인쇄→모바일 촬영에서 판독을 깨는 것은 광학·센서·
저장 경로의 열화다. 본 모듈은 그 열화를 물리 모델로 합성해 검출 파이프라인을
실물에 한 걸음 더 가깝게 시험한다 (진짜 카메라 검증은 exp18 실물 캠페인 과제).

FTO: 본 모듈은 '입력 이미지 생성기'다. 테스트 장면을 더 어렵게 만들 뿐,
소거·신뢰도 판정 경로(softdecode·spatial·consensus)와 무관하며 임포트하지
않는다. 디코더는 이 모듈의 존재를 모른다.

각 열화의 수식·파라미터 근거 (조사: concept 로드맵 exp17 항목):
  · 디포커스 disk PSF / Airy: Mannan & Langer CRV2016
    https://cim.mcgill.ca/~langer/MY_PAPERS/MannanLanger-CRV16-DFDModel.pdf
  · 선형 모션 커널: Askari Javaran et al., CMMM 2021
    https://www.hindawi.com/journals/cmmm/2021/6048137/
  · 비네팅 cos⁴ 법칙: Aggarwal et al. (UIUC)
    https://vision.ai.illinois.edu/.../on Cosine-fourth and Vignetting Effects in Real Lenses.pdf
  · Poisson-Gaussian 센서 노이즈(클리핑 포함): Foi et al., IEEE TIP 2008
    https://www.researchgate.net/publication/23249686
  · Brown-Conrady 방사 왜곡: OpenCV 보정 모델
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
  · 도트 게인(Murray-Davies): https://en.wikipedia.org/wiki/Dot_gain
  · 합성 광학 채널로 QR 강건성 시험한 선행: SBrT Channel Distortion to QR
    https://biblioteca.sbrt.org.br/articlefile/1751.pdf
"""
import math

import cv2
import numpy as np


def _as_f32(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def defocus_blur(arr, radius_px: float) -> np.ndarray:
    """기하광학 디포커스 = 착란원(CoC) 반경 radius_px의 균일 디스크 PSF.

    큰 디포커스에서 PSF는 pillbox(원판)에 수렴한다 (Mannan&Langer).
    radius<1이면 회절 우세 영역이라 σ=radius/2 가우시안으로 대체.
    """
    if radius_px <= 0:
        return arr
    if radius_px < 1.0:
        return cv2.GaussianBlur(arr, (0, 0), sigmaX=radius_px / 2.0)
    r = int(math.ceil(radius_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy <= radius_px * radius_px).astype(np.float32)
    disk /= disk.sum()
    return cv2.filter2D(_as_f32(arr), -1, disk, borderType=cv2.BORDER_REPLICATE)


def motion_blur(arr, length_px: float, angle_deg: float) -> np.ndarray:
    """손떨림 선형 모션 = 길이 length_px·각도 angle_deg 직선 커널 컨볼루션.

    L(px)=v·t/p (상면 속도×노출/픽셀피치). 실내 핸드헬드 t≈1/30~1/15s에서
    현실 길이 3~15px (Askari Javaran 2021).
    """
    L = int(round(length_px))
    if L < 2:
        return arr
    kernel = np.zeros((L, L), dtype=np.float32)
    cv2.line(kernel, (0, L // 2), (L - 1, L // 2), 1.0, 1)
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (L, L))
    s = kernel.sum()
    if s <= 0:
        return arr
    kernel /= s
    return cv2.filter2D(_as_f32(arr), -1, kernel, borderType=cv2.BORDER_REPLICATE)


def vignette(arr, alpha: float, cx_frac: float = 0.5,
             cy_frac: float = 0.5) -> np.ndarray:
    """주변부 광량 저하 마스크 곱. I(r)=1-alpha·r² (정규화 반경 r∈[0,1]).

    자연 비네팅 cos⁴ 법칙의 2차 근사 (28mm 렌즈 모서리 ≈ 중심의 39%).
    alpha∈[0.3,0.7]가 약~강. 광축(cx,cy)은 코드 중심에서 벗어날 수 있음.
    """
    if alpha <= 0:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = cx_frac * w, cy_frac * h
    rmax = math.hypot(max(cx, w - cx), max(cy, h - cy))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rmax
    mask = np.clip(1.0 - alpha * r * r, 0.05, 1.0)
    out = _as_f32(arr) * (mask[..., None] if arr.ndim == 3 else mask)
    return out


def glare(arr, intensity: float, cx_frac: float, cy_frac: float,
          sigma_frac: float = 0.18) -> np.ndarray:
    """정반사 하이라이트 = 클리핑되는 가우시안 핫스팟 가산.

    금속(레이저 각인)·코팅지의 좁고 강한 스펙큘러. intensity는 0~255 가산 진폭.
    """
    if intensity <= 0:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = cx_frac * w, cy_frac * h
    sigma = sigma_frac * min(h, w)
    hot = intensity * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) /
                             (2.0 * sigma * sigma))
    out = _as_f32(arr) + (hot[..., None] if arr.ndim == 3 else hot)
    return out


def sensor_noise(arr, shot_scale: float, read_sigma: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Poisson(샷)-Gaussian(읽기) 혼합 노이즈 + 클리핑.

    y = Poisson(x·scale)/scale + N(0, read_sigma²), 0~255 클리핑 (Foi 2008).
    shot_scale↓ → 광자 수↓ → 샷 노이즈↑ (저조도/고ISO). read_sigma는 ISO에 비례.
    """
    f = np.clip(_as_f32(arr), 0, 255)
    if shot_scale > 0:
        # 정규화 신호에 비례하는 광자 분산 — scale이 작을수록 SNR 악화
        f = rng.poisson(np.clip(f, 0, None) * shot_scale).astype(np.float32) / shot_scale
    if read_sigma > 0:
        f = f + rng.normal(0.0, read_sigma, f.shape).astype(np.float32)
    return np.clip(f, 0, 255)


def lens_distortion(arr, k1: float, k2: float = 0.0) -> np.ndarray:
    """Brown-Conrady 방사 왜곡 주입 (k1>0 배럴, k1<0 핀쿠션).

    모바일 광각 전형 |k1|≈0.01~0.3. cv2 보정맵을 역으로 써서 왜곡을 '추가'.
    """
    if k1 == 0.0 and k2 == 0.0:
        return arr
    h, w = arr.shape[:2]
    fx = fy = max(h, w)
    K = np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1.0]])
    dist = np.array([k1, k2, 0.0, 0.0, 0.0])
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h),
                                             cv2.CV_32FC1)
    return cv2.remap(_as_f32(arr), map1, map2, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def jpeg(arr, quality: int) -> np.ndarray:
    """JPEG 손실(DCT 양자화 블로킹). 모바일 저장 QF 70~95, 위협 QF<60."""
    u8 = np.clip(_as_f32(arr), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR) if u8.ndim == 3 else u8
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return arr
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR if u8.ndim == 3
                       else cv2.IMREAD_GRAYSCALE)
    if dec.ndim == 3:
        dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return dec.astype(np.float32)


def dot_gain(arr, erode_px: int, optical_sigma: float) -> np.ndarray:
    """인쇄 도트 게인 = 잉크 번짐(어두운 모듈 팽창) + 기재 광산란.

    Murray-Davies 도트게인의 모폴로지 근사: 어두운 영역 dilation(=밝은
    배경에서 erode) + 가우시안 산란. 코팅지 150lpi 40% 톤에서 TVI≈23%.
    """
    out = _as_f32(arr)
    if erode_px > 0:
        # 밝은 배경(255)에서 erode → 어두운 잉크가 번져 굵어짐
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * erode_px + 1, 2 * erode_px + 1))
        out = cv2.erode(out, k)
    if optical_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=optical_sigma)
    return out


# ── 합성 파이프라인 (물리적 발생 순서) ──────────────────────────────────────
# 인쇄(도트게인) → 광학(디포커스·모션·비네팅·글레어·왜곡) → 센서(노이즈)
#                → 저장(JPEG). 각 단계는 앞 단계 출력을 받는다.

# 현실 근접도 순 프리셋 (조사 결론: 노이즈>조명>디포커스>모션>JPEG).
# 'mild'=양호한 촬영, 'moderate'=평범한 실내 핸드헬드, 'severe'=악조건.
PRESETS = {
    'clean': dict(),
    'mild': dict(defocus=1.5, vignette=0.25, shot_scale=0.30, read_sigma=3.0,
                 jpeg_q=90),
    'moderate': dict(dot_erode=1, dot_sigma=0.6, defocus=3.0, motion_len=5.0,
                     vignette=0.45, glare=70.0, shot_scale=0.12,
                     read_sigma=6.0, k1=0.08, jpeg_q=70),
    'severe': dict(dot_erode=1, dot_sigma=0.8, defocus=5.0, motion_len=11.0,
                   vignette=0.6, glare=110.0, shot_scale=0.06, read_sigma=10.0,
                   k1=0.18, jpeg_q=45),
}


def camera_pipeline(arr, rng: np.random.Generator, *, dot_erode=0, dot_sigma=0.0,
                    defocus=0.0, motion_len=0.0, motion_angle=None,
                    vignette=0.0, glare=0.0, shot_scale=0.0, read_sigma=0.0,
                    k1=0.0, k2=0.0, jpeg_q=0) -> np.ndarray:
    """물리 발생 순서로 열화를 직렬 적용. 반환은 uint8 RGB.

    무작위 축(모션 각도·광축·글레어 위치)은 rng로 굴려 조건당 다양성 확보.
    """
    out = _as_f32(arr)
    if dot_erode or dot_sigma:
        out = dot_gain(out, dot_erode, dot_sigma)
    if defocus:
        out = defocus_blur(out, defocus)
    if motion_len:
        ang = float(rng.uniform(0, 360)) if motion_angle is None else motion_angle
        out = motion_blur(out, motion_len, ang)
    if vignette:
        # 광축을 코드 중심에서 약간 비껴 놓아 기울어진 음영 모사
        out = globals()['vignette'](out, vignette,
                                    cx_frac=float(rng.uniform(0.35, 0.65)),
                                    cy_frac=float(rng.uniform(0.35, 0.65)))
    if glare:
        out = globals()['glare'](out, glare, cx_frac=float(rng.uniform(0.2, 0.8)),
                                 cy_frac=float(rng.uniform(0.2, 0.8)))
    if k1 or k2:
        out = lens_distortion(out, k1, k2)
    if shot_scale or read_sigma:
        out = sensor_noise(out, shot_scale, read_sigma, rng)
    if jpeg_q:
        out = jpeg(out, jpeg_q)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_preset(arr, preset_name: str, rng: np.random.Generator) -> np.ndarray:
    """이름 프리셋으로 camera_pipeline 호출."""
    return camera_pipeline(arr, rng, **PRESETS[preset_name])
