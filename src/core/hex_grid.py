"""
육각형 격자 좌표계 유틸리티 (pointy-top 방향)

좌표계: Axial (q, r), 제약 조건 q+r+s=0 (s=-q-r)
격자 크기: radius=R → 셀 수 = 3R²+3R+1
"""
import math

# Axial 좌표 기준 6방향 이웃 오프셋 (pointy-top)
# E, SE, SW, W, NW, NE
HEX_DIRECTIONS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


def in_grid(q: int, r: int, radius: int) -> bool:
    return max(abs(q), abs(r), abs(q + r)) <= radius


def build_hex_grid(radius: int) -> list:
    """반지름 radius의 모든 셀 목록, q-major 정렬."""
    cells = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            cells.append((q, r))
    return cells


def hex_neighbors(q: int, r: int) -> list:
    return [(q + dq, r + dr) for dq, dr in HEX_DIRECTIONS]


def finder_cell_set(cq: int, cr: int) -> frozenset:
    """파인더 마커 7셀: 중심 + 6이웃."""
    cells = [(cq, cr)] + hex_neighbors(cq, cr)
    return frozenset(cells)


def finder_centers(radius: int) -> list:
    """
    3개 파인더 중심 위치 (NE·W·SE 방향).
    radius-1 위치에 배치해 이웃(ring-1)이 격자 안에 들어오도록 함.

    주의(exp13에서 정정): 이 배치는 비대칭이 아니다 — 세 중심은
    120° 회전 ρ(q,r)=(-q-r,q)로 순환하고 반사 (q,r)→(-r,-q)에 집합
    불변인 D3 대칭이라, 동일 형상 파인더로는 회전 3 × 거울상 2 =
    6가지 방향 가설이 구분 불가능하다. 방향 모호성 제거(형상 차등
    파인더)는 src/exp13_selfdesc/layout.py 참조.
    """
    r = radius - 1
    return [
        (r, -r),   # NE
        (-r, 0),   # W
        (0, r),    # SE
    ]


def build_reserved(radius: int) -> frozenset:
    """예약 셀(파인더 패턴 전체) frozenset."""
    reserved = set()
    for cq, cr in finder_centers(radius):
        reserved |= finder_cell_set(cq, cr)
    return frozenset(reserved)


def axial_to_pixel(q: int, r: int, hex_size: float,
                   ox: float = 0.0, oy: float = 0.0) -> tuple:
    """Axial → 픽셀 중심 좌표 (pointy-top)."""
    x = hex_size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = hex_size * 1.5 * r
    return (x + ox, y + oy)


def hex_vertices(q: int, r: int, hex_size: float,
                 ox: float = 0.0, oy: float = 0.0) -> list:
    """
    셀 (q,r)의 6개 꼭짓점 픽셀 좌표 (pointy-top).
    각도: 60°*i - 30° (i=0..5)
    """
    cx, cy = axial_to_pixel(q, r, hex_size, ox, oy)
    verts = []
    for i in range(6):
        rad = math.radians(60 * i - 30)
        verts.append((cx + hex_size * math.cos(rad),
                      cy + hex_size * math.sin(rad)))
    return verts


def hex_primary_triangles(q: int, r: int, hex_size: float,
                          ox: float = 0.0, oy: float = 0.0) -> list:
    """
    셀 (q,r)의 6개 primary 삼각형 반환.
    각 삼각형: (중심, 꼭짓점i, 꼭짓점i+1)

    반환: [(verts_3, centroid), ...] 6개 튜플
    """
    verts = hex_vertices(q, r, hex_size, ox, oy)
    cx, cy = axial_to_pixel(q, r, hex_size, ox, oy)
    tris = []
    for i in range(6):
        v_a = verts[i]
        v_b = verts[(i + 1) % 6]
        tri = [(cx, cy), v_a, v_b]
        centroid = ((cx + v_a[0] + v_b[0]) / 3, (cy + v_a[1] + v_b[1]) / 3)
        tris.append((tri, centroid))
    return tris


def hex_sub_triangles(q: int, r: int, hex_size: float,
                      ox: float = 0.0, oy: float = 0.0) -> list:
    """
    셀 (q,r)의 12개 서브삼각형 반환.
    각 primary 삼각형(6개)을 외접 변의 중점으로 절반 분할 → 12개.

    반환: [(verts_3, centroid), ...] 12개 튜플
      verts_3   : 삼각형 꼭짓점 3개 [(x,y), (x,y), (x,y)]
      centroid  : 무게중심 (x, y)
    """
    verts = hex_vertices(q, r, hex_size, ox, oy)
    cx, cy = axial_to_pixel(q, r, hex_size, ox, oy)

    sub_tris = []
    for i in range(6):
        v_a = verts[i]
        v_b = verts[(i + 1) % 6]
        m = ((v_a[0] + v_b[0]) / 2, (v_a[1] + v_b[1]) / 2)

        # 서브A: 중심, v_a, 중점 m
        tri_a = [(cx, cy), v_a, m]
        cen_a = ((cx + v_a[0] + m[0]) / 3, (cy + v_a[1] + m[1]) / 3)
        sub_tris.append((tri_a, cen_a))

        # 서브B: 중심, m, v_b
        tri_b = [(cx, cy), m, v_b]
        cen_b = ((cx + m[0] + v_b[0]) / 3, (cy + m[1] + v_b[1]) / 3)
        sub_tris.append((tri_b, cen_b))

    return sub_tris  # 12개


def grid_bounding_box(radius: int, hex_size: float) -> tuple:
    """셀 중심 기준 바운딩 박스 (min_x, min_y, max_x, max_y)."""
    xs, ys = [], []
    for q, r in build_hex_grid(radius):
        cx, cy = axial_to_pixel(q, r, hex_size)
        xs.append(cx)
        ys.append(cy)
    return (min(xs) - hex_size, min(ys) - hex_size,
            max(xs) + hex_size, max(ys) + hex_size)
