"""
D_QR Phase 13 인코더 — 자기기술 코드 (exp13)

exp9 인코더와의 차이 (모두 "메타데이터 외부 의존 제거"가 목적):
  · 페이로드 선두에 4바이트 big-endian 길이 프리픽스 → data_len 내장
  · 메시지를 격자 용량까지 0 패딩 → codeword_len이 radius에서 결정
    (디코더가 코드워드 길이를 따로 알 필요 없음 — QR 패드 코드워드와 동일 발상)
  · nsym은 ECC 레벨(포맷 링 2비트)에서 유도
  · radius는 GRID_RADII 한정 (포맷 링 반지름 인덱스 4비트)
  · 예약 셀 = 파인더 21 + 포맷 링 54 (build_reserved_v13)

셀 기하·비트 배치 규약은 exp9 그대로 (프로파일 A: primary 6삼각형,
B: 서브 12삼각형, q-major 셀 순서, MSB-first 비트 패킹).
"""
import math

from reedsolo import RSCodec

from src.core.bitutils import _bytes_to_bits
from src.core.hex_grid import build_hex_grid, finder_centers
from src.core.scramble import xor_scramble
from src.exp9_soft.encoder_v9 import GRID_RADII, PROFILES
from src.exp13_selfdesc.format_ring import (DEFAULT_ECC_LEVEL, ECC_LEVELS,
                                            FORMAT_VERSION, PROFILE_IDS,
                                            nsym_for_level, pack_format)
from src.exp13_selfdesc.layout import (EXP_TAG, build_reserved_v13,
                                       codeword_length, data_cells_v13,
                                       format_cells_per_finder,
                                       message_capacity)

LEN_PREFIX = 4   # 길이 프리픽스 바이트 수


class DQREncoderV13:
    """
    D_QR Phase 13 인코더 — 자기기술.

    사용 예:
        enc = DQREncoderV13(profile='A')          # ECC 레벨은 프로파일 기본
        result = enc.encode("데이터".encode())
        img, info = DQRRendererV13().render(result)
    """

    def __init__(self, profile: str = 'A', ecc_level: int = None,
                 scramble: bool = True):
        # scramble (exp15b, 기본 on): RS 인코딩 직전의 패딩 포함 메시지에
        # LFSR 고정 마스크 XOR (src/core/scramble) — 전부-0 근방 페이로드
        # (빈 데이터, 0 위주 바이트열)도 비자명 코드워드가 되어 디코더의
        # 퇴화 코드워드 가드(4층)가 정상 페이로드를 거부할 일이 없어진다.
        # CRC 아님(검출 기능 없음, 전단사 변환만 — FTO 제약 3 무관).
        # 여부는 포맷 링 scramble 비트(구 예비 비트)에 기록 → 자기기술 유지.
        if profile not in PROFILE_IDS:
            raise ValueError(f"profile은 {list(PROFILE_IDS)} 중 하나: {profile}")
        self.scramble = scramble
        self.profile = profile
        self.bits_per_cell = PROFILES[profile]['bits_per_cell']
        self.ecc_level = (DEFAULT_ECC_LEVEL[profile]
                          if ecc_level is None else int(ecc_level))
        self.nsym = nsym_for_level(self.ecc_level)

    def max_data_bytes(self, radius: int) -> int:
        """길이 프리픽스를 제외한 순수 데이터 용량."""
        return message_capacity(radius, self.bits_per_cell, self.nsym) - LEN_PREFIX

    def _choose_radius(self, msg_len: int) -> int:
        for radius in GRID_RADII:
            if message_capacity(radius, self.bits_per_cell, self.nsym) >= msg_len:
                return radius
        raise ValueError(
            f"데이터가 너무 큽니다: {msg_len - LEN_PREFIX} bytes "
            f"(exp13은 포맷 링 반지름 인덱스 4비트 → GRID_RADII 한정)")

    def encode(self, data: bytes) -> dict:
        msg = len(data).to_bytes(LEN_PREFIX, 'big') + data
        radius = self._choose_radius(len(msg))
        cap = message_capacity(radius, self.bits_per_cell, self.nsym)

        # 용량까지 0 패딩 → codeword_len이 radius·nsym·bpc에서 유일하게 결정
        msg = msg + b'\x00' * (cap - len(msg))
        if self.scramble:
            # 패딩 후 전체 메시지에 XOR — 0 패딩 구간도 비0 마스크 값이 되어
            # 전부-0 코드워드 퇴화 근방에서 멀어진다 (exp15b ② 봉쇄)
            msg = xor_scramble(msg)
        codeword = bytes(RSCodec(self.nsym).encode(msg))
        cw_len = codeword_length(cap, self.nsym)
        assert len(codeword) == cw_len, "codeword_length 규약 불일치"

        bpc = self.bits_per_cell
        cells = data_cells_v13(radius)
        bits = _bytes_to_bits(codeword)
        n_cells_needed = math.ceil(len(bits) / bpc)
        bits += [0] * (n_cells_needed * bpc - len(bits))

        tensor = {}
        for k, (q, r) in enumerate(cells):
            if k < n_cells_needed:
                start = k * bpc
                val = 0
                for b in range(bpc):
                    val |= bits[start + b] << (bpc - 1 - b)
                tensor[(q, r)] = val
            else:
                tensor[(q, r)] = 0

        # 포맷 링: 18비트 워드를 파인더당 1사본씩 3중 반복
        fmt_fields = {
            'version': FORMAT_VERSION,
            'profile': PROFILE_IDS[self.profile],
            'ecc_level': self.ecc_level,
            'radius_idx': GRID_RADII.index(radius),
            'mask_id': 0,
            'scramble': 1 if self.scramble else 0,
            'reserve': 0,
        }
        fmt_bits = pack_format(fmt_fields)
        format_cell_bits = {}
        for copy_cells in format_cells_per_finder(radius):
            for i, cell in enumerate(copy_cells):
                format_cell_bits[cell] = fmt_bits[i]

        return {
            'tensor': tensor,
            'reserved': build_reserved_v13(radius),
            'finder_centers': finder_centers(radius),
            'format_cell_bits': format_cell_bits,
            'format_fields': fmt_fields,
            'radius': radius,
            'profile': self.profile,
            'bits_per_cell': bpc,
            'ecc_level': self.ecc_level,
            'nsym': self.nsym,
            'codeword_len': cw_len,
            'msg_len': cap,
            'data_len': len(data),
            'exp_tag': EXP_TAG,
            'capacity': {
                'radius': radius,
                'total_cells': len(build_hex_grid(radius)),
                'data_cells': len(cells),
                'bits_per_cell': bpc,
                'max_data_bytes': cap - LEN_PREFIX,
                'ecc_rate': ECC_LEVELS[self.ecc_level],
                'nsym': self.nsym,
                'profile': self.profile,
                'encoding': f"exp13_selfdesc_profile_{self.profile}_{bpc}bit",
                'color_dependent': False,
            },
        }
