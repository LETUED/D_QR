"""
exp14 — 재생 모델 3종 (코히어런트 / 백색 LED 분산 / 스페클)

백색 LED 분산 모델:
  Fourier 홀로그램의 재생상 위치는 x = λ·f·ν (ν = 격자 주파수) → 위치 ∝ λ.
  3파장(450/550/650nm)의 재생상을 DC(광축) 중심 등방 스케일 s로 변형해
  세기 합성. 순수 Fourier 기하(s = λ/λd)는 오프축 캐리어 때문에 상 전체가
  수십 px 강체 변위하는 최악 조건이고, 이미지면 홀로그램(엠보싱 양산 경로,
  재생상이 홀로그램면에 위치)은 분산 변위가 0에 수렴한다. 두 극단을
  κ ∈ [0,1] 하나로 보간한다:
      s_eff(λ, κ) = 1 + κ·(λ/λd − 1)
      κ = 1 : Fourier(원거리) 기하 — 분산 최악
      κ = 0 : 이상 이미지면 기하 — 분산 0 (위상 detune 효과만 잔존)
  κ의 물리적 해석: 재생상 깊이 / Fourier 초점거리 비율의 유효값.

스페클 모델:
  재생장의 무작위 위상 간섭 → 곱셈성 세기 변동. 평균 1, 콘트라스트
  (σ/μ) = C인 상관 그레인 필드를 재생 세기에 곱한다.
"""
import numpy as np
from PIL import Image

from src.exp14_holosim.cgh import DESIGN_NM, replay_intensity

LED_WAVELENGTHS = (450.0, 550.0, 650.0)  # 백색 LED 근사 3파장 (B/G/R)


def zoom_about_center(arr: np.ndarray, s: float) -> np.ndarray:
    """배열 중심(N//2 = FFT DC)을 고정점으로 등방 배율 s 적용 (bilinear)."""
    if abs(s - 1.0) < 1e-12:
        return arr.copy()
    n_rows, n_cols = arr.shape
    cy, cx = n_rows // 2, n_cols // 2
    # 출력 좌표 → 입력 좌표 역변환 (배율 s = 내용이 s배 확대)
    ys = (np.arange(n_rows) - cy) / s + cy
    xs = (np.arange(n_cols) - cx) / s + cx

    y0 = np.clip(np.floor(ys).astype(int), 0, n_rows - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, n_cols - 2)
    wy = np.clip(ys - y0, 0.0, 1.0)[:, None]
    wx = np.clip(xs - x0, 0.0, 1.0)[None, :]

    out = (arr[np.ix_(y0, x0)] * (1 - wy) * (1 - wx)
           + arr[np.ix_(y0, x0 + 1)] * (1 - wy) * wx
           + arr[np.ix_(y0 + 1, x0)] * wy * (1 - wx)
           + arr[np.ix_(y0 + 1, x0 + 1)] * wy * wx)

    # 입력 범위 밖으로 나간 좌표는 0 (시야 밖)
    out[(ys < 0) | (ys > n_rows - 1), :] = 0.0
    out[:, (xs < 0) | (xs > n_cols - 1)] = 0.0
    return out


def led_composite(phi: np.ndarray, kappa: float,
                  wavelengths: tuple = LED_WAVELENGTHS,
                  design_nm: float = DESIGN_NM) -> np.ndarray:
    """백색 LED 3파장 합성 재생 세기 (κ: 분산 기하 계수)."""
    total = None
    for lam in wavelengths:
        I = replay_intensity(phi, lam, design_nm)
        s_eff = 1.0 + kappa * (lam / design_nm - 1.0)
        I = zoom_about_center(I, s_eff)
        total = I if total is None else total + I
    return total / len(wavelengths)


def speckle_field(shape: tuple, contrast: float, grain_px: float = 2.0,
                  rng: np.random.Generator = None) -> np.ndarray:
    """
    곱셈성 스페클 필드 — 평균 1, 콘트라스트(σ/μ) ≈ contrast, 상관 길이 grain_px.

    완전 발달 스페클(상관 복소 가우시안의 세기, C=1)을 평활화한 뒤
    목표 콘트라스트로 선형 혼합. 음수는 0으로 클립(세기는 비음수).
    """
    if rng is None:
        rng = np.random.default_rng()
    g = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    # 주파수 가우시안 필터로 스페클 그레인 크기 부여
    H = np.exp(-2.0 * (np.pi * grain_px) ** 2 * (fy ** 2 + fx ** 2))
    s = np.fft.ifft2(np.fft.fft2(g) * H)
    I = s.real ** 2 + s.imag ** 2
    I /= I.mean()
    c0 = I.std()  # 평활화 후 실측 콘트라스트 (목표치 보정용)
    M = 1.0 + (contrast / c0) * (I - 1.0)
    return np.clip(M, 0.0, None)


def intensity_to_image(I_win: np.ndarray, exposure: float = 0.5,
                       percentile: float = 99.0) -> Image.Image:
    """
    창 내 재생 세기 → 판독용 이미지.

    노출 스케일 = exposure × P{percentile}(I): 카메라 AGC에 해당하는
    전역 단일 스케일(국소 처리 없음 — FTO 2 준수). 재생상은 발광
    패턴이므로 반전해 인쇄 규약(어두운 마크/밝은 배경)으로 되돌린다.
    """
    scale = float(np.percentile(I_win, percentile)) * exposure
    if scale <= 0.0:
        scale = float(I_win.max()) or 1.0
    val = np.clip(I_win / scale, 0.0, 1.0)
    gray = (255.0 - np.round(val * 255.0)).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), 'RGB')
