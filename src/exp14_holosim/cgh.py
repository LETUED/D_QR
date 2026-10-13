"""
exp14 — binary phase CGH 생성 (Gerchberg-Saxton, numpy FFT 전용)

D_QR 패턴(진폭 타깃)을 Fourier 면에 재생하는 0/π 이진 위상 홀로그램을
계산한다. concept/12 §12.6 0단계 — 추가 의존성 금지, 비용 0원.

물리 규약:
  홀로그램면 u(x,y) = exp(i·φ), φ ∈ {0, π}
    → 표면 부조(엠보싱) 모델: 위상 깊이는 λ_design에서 정확히 π,
      다른 파장에선 φ(λ) = π·(λ_design/λ) 로 detune (replay 측에서 적용)
  재생면 = fftshift(FFT(u)) — Fourier(원거리) 기하, 재생상 위치 ∝ λ

오프셋 배치인 이유: 이진 위상 홀로그램은 실수장이므로 재생상이
점대칭(쌍둥이 상)을 강제당한다 → 타깃 창을 DC 축에서 수직으로
오프셋해 쌍둥이 상·0차 스파이크와 겹치지 않게 분리.
"""
import numpy as np

DESIGN_NM = 532.0  # 설계 파장 (녹색 레이저 포인터/DPSS 표준)


def embed_target(img_gray: np.ndarray, field_size: int = 2048,
                 margin_from_dc: int = 24) -> tuple:
    """
    그레이스케일 타깃(밝은 값 = 높은 진폭)을 재생면 배열에 오프셋 배치.

    img_gray: (H, W) uint8/float — 호출 측에서 "마크=밝음"으로 반전해 전달
              (홀로그램 재생상은 발광 패턴이므로 기하 마크를 밝게 기록)
    Returns: (target_amp [field_size² float64], window (slice, slice))
    """
    H, W = img_gray.shape
    cy = cx = field_size // 2
    r0 = cy - H - margin_from_dc          # 창 하단이 DC 위 margin px에 오도록
    c0 = cx - W // 2
    if r0 < 0 or c0 < 0 or c0 + W > field_size:
        raise ValueError(
            f"타깃 {W}x{H}가 field {field_size}의 오프셋 반평면에 안 들어감")

    target = np.zeros((field_size, field_size), dtype=np.float64)
    target[r0:r0 + H, c0:c0 + W] = np.asarray(img_gray, dtype=np.float64) / 255.0
    window = (slice(r0, r0 + H), slice(c0, c0 + W))
    return target, window


def conjugate_twin(arr: np.ndarray) -> np.ndarray:
    """fftshift된 배열의 DC(N//2) 기준 점대칭 변환 — 켤레대칭 쌍둥이 위치."""
    return np.roll(np.roll(arr[::-1, ::-1], 1, axis=0), 1, axis=1)


def gs_binary_phase(target_amp: np.ndarray,
                    n_iter_continuous: int = 50, n_iter_binary: int = 25,
                    seed: int = 14) -> np.ndarray:
    """
    Gerchberg-Saxton — 연속 위상 단계로 수렴시킨 뒤 0/π 이진 양자화 단계.

    이진 위상 홀로그램 u ∈ {+1, −1}은 실수장 → 재생면이 켤레대칭을
    강제당한다. 비대칭 타깃을 그대로 제약하면 GS가 모순된 제약과
    싸우다 발산하므로, 쌍둥이 상(점대칭 사본)을 타깃에 명시적으로
    포함시켜 제약을 대칭과 양립시킨다 (binary CGH 표준 기법).
    진폭 제약은 전면(full-plane) 적용 — 위상/부호만 쓰므로 스케일 무관.

    Returns: phi (field² float64), 값 ∈ {0, π}
    """
    A = np.maximum(target_amp, conjugate_twin(target_amp))
    rng = np.random.default_rng(seed)
    # 초기 재생면: 타깃 진폭 + 무작위 위상 (GS 표준 초기화)
    F = A * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, A.shape))

    u = None
    for it in range(n_iter_continuous + n_iter_binary):
        h = np.fft.ifft2(np.fft.ifftshift(F))
        if it < n_iter_continuous:
            u = np.exp(1j * np.angle(h))                  # 위상만(연속) 단계
        else:
            u = np.where(h.real >= 0.0, 1.0 + 0j, -1.0 + 0j)  # 0/π 양자화 단계
        F2 = np.fft.fftshift(np.fft.fft2(u))
        F = A * np.exp(1j * np.angle(F2))                 # 전면 진폭 제약

    return np.where(u.real >= 0.0, 0.0, np.pi)


def replay_intensity(phi: np.ndarray, wavelength_nm: float,
                     design_nm: float = DESIGN_NM) -> np.ndarray:
    """
    단일 파장 코히어런트 재생 세기 |FFT(exp(i·φ·λd/λ))|².

    표면 부조의 광경로차는 고정이므로 위상 깊이는 파장에 반비례:
    설계 파장에서 π → λ에서 π·(λd/λ). detune은 회절 효율 저하와
    0차 누설로 나타난다 (0차는 DC에 위치 — 신호 창 밖).
    """
    detune = design_nm / wavelength_nm
    u = np.exp(1j * phi * detune)
    F = np.fft.fftshift(np.fft.fft2(u))
    return F.real ** 2 + F.imag ** 2


def replay_metrics(intensity: np.ndarray, target_amp: np.ndarray,
                   window: tuple) -> dict:
    """재생 품질: 창 내 에너지 효율 η, 진폭-타깃 피어슨 상관."""
    I_win = intensity[window]
    eta = float(I_win.sum() / intensity.sum())
    a = np.sqrt(I_win).ravel()
    t = target_amp[window].ravel()
    corr = float(np.corrcoef(a, t)[0, 1])
    return {'efficiency': eta, 'fidelity': corr}
