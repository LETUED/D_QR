"""
exp14 대조군 — 가상 컬러 코드의 파장 분산 붕괴 모델

가상 컬러 코드: 같은 육각/삼각 기하에서 각 삼각형이 R/G/B 3채널 독립
비트를 가진다(용량 3배를 약속하는 구조). 각 채널 패턴을 자기 파장
(650/550/450nm) 전용 binary phase CGH로 기록했다고 가정 — 컬러 코드에
가장 유리한 설정이다.

붕괴 메커니즘: 표면 부조 홀로그램은 thin grating이라 파장 선택성이
없다. 어떤 CGH든 모든 파장을 회절시키므로, 카메라 R채널에는 R패턴뿐
아니라 G/B 패턴의 적색(650nm) 회절상도 같은 자리에 겹친다 → 채널 분리
불가. 이것이 concept/12가 인용한 CHQR(US 10,885,413)이 층별 분산성
HOE라는 별도 광학 장치를 요구하는 이유다. 단일색 D_QR은 세 파장이
같은 정보의 중복 사본이라 혼합이 손상이 아니다.
"""
import numpy as np

from src.core.hex_grid import build_reserved, finder_centers
from src.exp9_soft.decoder_v9 import DQRDecoderV9
from src.exp10_progressive.encoder_v10 import GRAINS, split_data_cells
from src.exp10_progressive.renderer_v10 import DQRRendererV10
from src.exp14_holosim.cgh import replay_intensity
from src.exp14_holosim.replay import zoom_about_center

# 채널 순서 R/G/B와 기록 파장
COLOR_WAVELENGTHS = (650.0, 550.0, 450.0)


def make_channel_tensors(radius: int, seed: int = 14) -> list:
    """같은 셀 집합 위 채널별 독립 비트 텐서 3개 (용량 3배 모델)."""
    coarse_cells, fine_cells = split_data_cells(radius)
    rng = np.random.default_rng(seed)
    tensors = []
    for _ in range(3):
        t = {}
        for q, r in coarse_cells:
            t[(q, r)] = int(rng.integers(0, 1 << GRAINS['coarse']['bits_per_cell']))
        for q, r in fine_cells:
            t[(q, r)] = int(rng.integers(0, 1 << GRAINS['fine']['bits_per_cell']))
        tensors.append(t)
    return tensors


def render_pattern(tensor: dict, radius: int, hex_size: int) -> tuple:
    """단일 채널 패턴을 D_QR 기하 그대로 렌더 → (PIL 이미지, info)."""
    encoded = {
        'radius': radius,
        'tensor': tensor,
        'finder_centers': finder_centers(radius),
        'reserved': build_reserved(radius),
    }
    return DQRRendererV10(hex_size=hex_size).render(encoded)


def color_camera_channels(phis: list, kappa: float,
                          wavelengths: tuple = COLOR_WAVELENGTHS) -> list:
    """
    백색 3파장 조명 하 카메라 채널별 세기.

    카메라 채널 m(λ_m 이상 협대역 필터 가정 — 컬러 코드에 유리)은
    세 CGH 모두의 λ_m 회절상을 합산해 본다:
      I_m = Σ_k zoom(|FFT(exp(i·φ_k·λ_k/λ_m))|², s_eff(λ_m/λ_k, κ))
    """
    channels = []
    for lam_m in wavelengths:
        total = None
        for phi_k, lam_k in zip(phis, wavelengths):
            I = replay_intensity(phi_k, lam_m, design_nm=lam_k)
            s_eff = 1.0 + kappa * (lam_m / lam_k - 1.0)
            I = zoom_about_center(I, s_eff)
            total = I if total is None else total + I
        channels.append(total)
    return channels


def bit_accuracy(image, tensor: dict, radius: int, hex_size: float,
                 origin_x: float, origin_y: float,
                 threshold: int = 128) -> float:
    """
    이미지의 삼각형 비트를 텐서 진리값과 대조한 비트 정확도 (0~1).

    셀 판독은 exp9 soft reader 재사용 — 전역 임계(FTO 허용) 그대로.
    복호가 아닌 비트 수준 측정: 컬러 대조군은 RS 스트림 구조가 없으므로
    채널 붕괴를 공정하게 비교하려면 비트 정확도가 공통 단위다.
    """
    reader = DQRDecoderV9(threshold=threshold)
    pixels = np.array(image.convert('RGB'))
    coarse_cells, fine_cells = split_data_cells(radius)

    correct, total = 0, 0
    for cells, bpc in ((coarse_cells, GRAINS['coarse']['bits_per_cell']),
                       (fine_cells, GRAINS['fine']['bits_per_cell'])):
        for q, r in cells:
            bits, _ = reader._read_cell_soft(
                pixels, q, r, bpc, hex_size, origin_x, origin_y)
            truth = tensor.get((q, r), 0)
            for i, b in enumerate(bits):
                correct += int(b == ((truth >> (bpc - 1 - i)) & 1))
                total += 1
    return correct / total if total else 0.0
