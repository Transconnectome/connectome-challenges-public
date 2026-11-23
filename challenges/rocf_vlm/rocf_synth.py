"""ROCF(Rey-Osterrieth Complex Figure) 18단위 클린룸 기하·5 Hz 운동학·멀티모달 과정 렌더러.

현상 1 (정적 점수 천장효과 및 임상 교란):
  - 인지 정상(CU) 참가자는 모사(Copy, ~34.7/36)와 지연회상(Recall, ~33.5/36) 모두 천장 근처에 몰린다.
  - 경도인지장애(MCI)는 아밀로이드 양성률이 높고(~66% vs CU ~25%) 정적 점수가 낮지만,
    정적 손상도·정적 PNG 픽셀은 (연령, 학력, 진단)이 주어졌을 때 아밀로이드 상태와 조건부 독립이다.

현상 2 (과정 기반 시공간 아밀로이드 신호):
  - 아밀로이드 관련 잠재 과정 요인(theta_proc)은 정적 완성도가 아니라
    (1) 5 Hz 펜 들림(pen_down=0) 구조 경계 계획 멈춤과
    (2) 모사 대비 지연회상에서의 위계 역전(주변 세부 단위 선행·골격 지연)·공간 진행 괴리·단위 내부 재방문
    에만 반영된다. 펜이 닿아 있는 동안의 그리기 속력(v_draw)은 아밀로이드와 무관하다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.special import expit

# 기준 도형 좌표계 (fig_mm): Unit 2(큰 직사각형) 좌상단 (0, 0), 우하단 (W, H)
FIG_W_MM = 96.0
FIG_H_MM = 68.0
FIG_CX_MM = 48.0
FIG_CY_MM = 34.0
FIG_SLOPE = FIG_H_MM / FIG_W_MM  # k = 68 / 96

# 디지털 태블릿 캔버스 (canvas_mm) 및 정적 이미지 해상도
CANVAS_W_MM = 210.0
CANVAS_H_MM = 150.0
CANVAS_CENTER_MM = (105.0, 75.0)

IMAGE_W_PX = 672
IMAGE_H_PX = 480
SUPERSAMPLE = 2

SAMPLING_HZ = 5.0
DT_SEC = 1.0 / SAMPLING_HZ  # 0.20s


def _arr(pts: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray(pts, dtype=float)


def _build_canonical_units() -> list[dict[str, Any]]:
    """Osterrieth (1944) 표준 18개 채점 단위의 기준 폴리라인(fig_mm)."""
    w, h, cx, cy, k = FIG_W_MM, FIG_H_MM, FIG_CX_MM, FIG_CY_MM, FIG_SLOPE

    # Unit 6: 좌측 내부 직사각형 + 내부 대각선 (우측 꼭짓점이 큰 대각선 y=k*x, y=h-k*x 위에 놓임)
    u6_y0, u6_y1 = 23.0, 45.0
    u6_x0, u6_x1 = 10.0, u6_y0 / k  # 32.47 mm

    # Unit 7: Unit 6 위 수평 선분
    u7_y = 19.0
    u7_x0, u7_x1 = 10.0, u7_y / k

    # Unit 8: 좌상단 대각선 위쪽 4개 평행 수평선 (y in [5, 9, 13, 17])
    u8_ys = [5.0, 9.0, 13.0, 17.0]
    u8_lines = [_arr([(y / k, y), (cx, y)]) for y in u8_ys]

    # Unit 10: 우상단 대각선 위쪽 수직선
    u10_x = 68.0
    u10_y1 = k * (w - u10_x)

    # Unit 11: 우상단 원(28각형) + 점 3개
    c11_x, c11_y, c11_r = 76.0, 24.2, 5.8
    angles = np.linspace(0.0, 2.0 * np.pi, 29)
    circle_28 = np.column_stack([c11_x + c11_r * np.cos(angles), c11_y + c11_r * np.sin(angles)])
    dots_11 = [
        _arr([(73.9, 22.5), (74.5, 22.9)]),
        _arr([(77.5, 22.5), (78.1, 22.9)]),
        _arr([(75.7, 26.2), (76.3, 26.6)]),
    ]

    # Unit 12: 우하단 대각선 위 5개 수직 눈금선
    diag_len = float(np.hypot(cx, cy))
    nx, ny = -cy / diag_len, cx / diag_len
    tick_half = 4.0
    ticks_12 = []
    for s in (0.22, 0.36, 0.50, 0.64, 0.78):
        px, py = cx + cx * s, cy + cy * s
        ticks_12.append(_arr([(px - tick_half * nx, py - tick_half * ny),
                              (px + tick_half * nx, py + tick_half * ny)]))

    # Unit 13~16: 우측 이등변삼각형 및 내부·정점 요소
    apex_x, apex_y = 128.0, cy
    u15_x = 108.0
    u15_half_h = cy * (apex_x - u15_x) / (apex_x - w)

    units = [
        {
            "unit_id": 1,
            "name": "left_exterior_cross",
            "tier": 1,
            "quadrant": 0,
            "polylines": [
                _arr([(-10.0, 4.0), (-10.0, 51.0)]),
                _arr([(-18.0, 17.0), (0.0, 17.0)]),
                _arr([(-10.0, 51.0), (0.0, 51.0)]),
            ],
        },
        {
            "unit_id": 2,
            "name": "large_rectangle",
            "tier": 0,
            "quadrant": 0,
            "polylines": [
                _arr([(0.0, 0.0), (w, 0.0)]),
                _arr([(w, 0.0), (w, h)]),
                _arr([(w, h), (0.0, h)]),
                _arr([(0.0, h), (0.0, 0.0)]),
            ],
        },
        {
            "unit_id": 3,
            "name": "diagonal_cross",
            "tier": 0,
            "quadrant": 1,
            "polylines": [
                _arr([(0.0, 0.0), (w, h)]),
                _arr([(w, 0.0), (0.0, h)]),
            ],
        },
        {
            "unit_id": 4,
            "name": "horizontal_midline",
            "tier": 0,
            "quadrant": 0,
            "polylines": [
                _arr([(0.0, cy), (cx, cy)]),
                _arr([(cx, cy), (w, cy)]),
            ],
        },
        {
            "unit_id": 5,
            "name": "vertical_midline",
            "tier": 0,
            "quadrant": 1,
            "polylines": [
                _arr([(cx, 0.0), (cx, cy)]),
                _arr([(cx, cy), (cx, h)]),
            ],
        },
        {
            "unit_id": 6,
            "name": "left_inner_box",
            "tier": 2,
            "quadrant": 0,
            "polylines": [
                _arr([(u6_x0, u6_y0), (u6_x1, u6_y0)]),
                _arr([(u6_x1, u6_y0), (u6_x1, u6_y1)]),
                _arr([(u6_x1, u6_y1), (u6_x0, u6_y1)]),
                _arr([(u6_x0, u6_y1), (u6_x0, u6_y0)]),
                _arr([(u6_x0, u6_y0), (u6_x1, u6_y1)]),
                _arr([(u6_x1, u6_y0), (u6_x0, u6_y1)]),
            ],
        },
        {
            "unit_id": 7,
            "name": "segment_above_left_box",
            "tier": 2,
            "quadrant": 0,
            "polylines": [_arr([(u7_x0, u7_y), (u7_x1, u7_y)])],
        },
        {
            "unit_id": 8,
            "name": "four_parallel_upper_left",
            "tier": 2,
            "quadrant": 0,
            "polylines": u8_lines,
        },
        {
            "unit_id": 9,
            "name": "upper_right_triangle",
            "tier": 2,
            "quadrant": 1,
            "polylines": [
                _arr([(cx, 0.0), (cx, -13.0)]),
                _arr([(cx, -13.0), (w, 0.0)]),
            ],
        },
        {
            "unit_id": 10,
            "name": "upper_right_vertical_line",
            "tier": 2,
            "quadrant": 1,
            "polylines": [_arr([(u10_x, 0.0), (u10_x, u10_y1)])],
        },
        {
            "unit_id": 11,
            "name": "circle_with_three_dots",
            "tier": 2,
            "quadrant": 1,
            "polylines": [circle_28, *dots_11],
        },
        {
            "unit_id": 12,
            "name": "five_ticks_lower_right",
            "tier": 2,
            "quadrant": 2,
            "polylines": ticks_12,
        },
        {
            "unit_id": 13,
            "name": "right_triangle_sides",
            "tier": 1,
            "quadrant": 2,
            "polylines": [
                _arr([(w, 0.0), (apex_x, apex_y)]),
                _arr([(apex_x, apex_y), (w, h)]),
            ],
        },
        {
            "unit_id": 14,
            "name": "right_apex_diamond",
            "tier": 2,
            "quadrant": 2,
            "polylines": [
                _arr([(apex_x, apex_y), (134.0, 41.0), (apex_x, 48.0)]),
                _arr([(apex_x, 48.0), (122.0, 41.0), (apex_x, apex_y)]),
            ],
        },
        {
            "unit_id": 15,
            "name": "vertical_in_right_triangle",
            "tier": 2,
            "quadrant": 2,
            "polylines": [_arr([(u15_x, cy - u15_half_h), (u15_x, cy + u15_half_h)])],
        },
        {
            "unit_id": 16,
            "name": "horizontal_in_right_triangle",
            "tier": 1,
            "quadrant": 2,
            "polylines": [_arr([(w, cy), (apex_x, cy)])],
        },
        {
            "unit_id": 17,
            "name": "bottom_cross",
            "tier": 1,
            "quadrant": 3,
            "polylines": [
                _arr([(36.0, h), (36.0, 86.5)]),
                _arr([(22.0, 77.5), (46.0, 77.5)]),
            ],
        },
        {
            "unit_id": 18,
            "name": "lower_left_square_diag",
            "tier": 1,
            "quadrant": 3,
            "polylines": [
                _arr([(0.0, h), (0.0, 86.0)]),
                _arr([(0.0, 86.0), (18.0, 86.0)]),
                _arr([(18.0, 86.0), (18.0, h)]),
                _arr([(0.0, h), (18.0, 86.0)]),
            ],
        },
    ]
    return units


CANONICAL_UNITS: list[dict[str, Any]] = _build_canonical_units()
CANONICAL_UNIT_MAP: dict[int, dict[str, Any]] = {u["unit_id"]: u for u in CANONICAL_UNITS}

TIER_0_IDS = [2, 3, 4, 5]
TIER_1_IDS = [1, 13, 16, 17, 18]
TIER_2_IDS = [6, 7, 8, 9, 10, 11, 12, 14, 15]


def default_stats() -> dict[str, Any]:
    """합성 코호트 기본 모수 사전 (--stats 생략 시 사용)."""
    return {
        "source": {"method": "default", "note": "실제 데이터에서 추출하지 않은 예시 값"},
        "age_mean": 71.5,
        "age_sd": 6.0,
        "edu_mean": 12.5,
        "edu_sd": 3.5,
        "male_prob": 0.48,
        "apoe4_probs": [0.74, 0.23, 0.03],
        "mci_logit": {"intercept": -0.75, "z_age": 0.45, "z_edu": -0.25, "apoe4": 0.35},
        "amyloid_logit": {
            "intercept": -1.58,
            "mci": 1.72,
            "age_per_10y_from_70": 0.52,
            "apoe4": 1.35,
            "z_edu": -0.10,
        },
    }


def score_osterrieth_unit(present: bool, displaced: bool, distorted: bool) -> float:
    """단일 Osterrieth 단위 채점 (0.0, 0.5, 1.0, 2.0)."""
    if not present:
        return 0.0
    if not displaced and not distorted:
        return 2.0
    if displaced and distorted:
        return 0.5
    return 1.0


def compute_osterrieth_score(unit_states: dict[int, dict[str, bool]]) -> float:
    """18개 단위의 상태 사전에서 36점 만점 Osterrieth 총점을 계산한다."""
    total = 0.0
    for uid in range(1, 19):
        st = unit_states[uid]
        total += score_osterrieth_unit(st["present"], st["displaced"], st["distorted"])
    return float(total)


def _sample_unit_states_cu(
    condition: str,
    z_age: float,
    z_edu: float,
    rng: np.random.Generator,
) -> dict[int, dict[str, bool]]:
    """CU 정적 단위 상태 샘플링: 아밀로이드와 조건부 완전 독립."""
    burden = float(np.clip(0.22 * z_age - 0.14 * z_edu + rng.normal(0.0, 0.35), -0.9, 1.2))
    states: dict[int, dict[str, bool]] = {}

    if condition == "copy":
        lo, hi = 33.0, 36.0
        for u in CANONICAL_UNITS:
            uid, tier = u["unit_id"], u["tier"]
            if tier == 0:
                present = True
                displaced = False
                distorted = bool(rng.uniform() < 0.008)
            elif tier == 1:
                present = True
                displaced = bool(rng.uniform() < 0.012)
                distorted = bool(rng.uniform() < (0.032 + 0.012 * max(0.0, burden)))
            else:
                present = bool(rng.uniform() > (0.008 + 0.006 * max(0.0, burden)))
                displaced = bool(present and rng.uniform() < (0.028 + 0.012 * max(0.0, burden)))
                distorted = bool(present and rng.uniform() < (0.062 + 0.020 * max(0.0, burden)))
            states[uid] = {"present": present, "displaced": displaced, "distorted": distorted}
    else:
        lo, hi = 31.0, 35.5
        for u in CANONICAL_UNITS:
            uid, tier = u["unit_id"], u["tier"]
            if tier == 0:
                present = True
                displaced = False
                distorted = bool(rng.uniform() < 0.015)
            elif tier == 1:
                present = bool(rng.uniform() > 0.010)
                displaced = bool(present and rng.uniform() < 0.028)
                distorted = bool(present and rng.uniform() < (0.065 + 0.020 * max(0.0, burden)))
            else:
                present = bool(rng.uniform() > (0.032 + 0.015 * max(0.0, burden)))
                displaced = bool(present and rng.uniform() < (0.045 + 0.018 * max(0.0, burden)))
                distorted = bool(present and rng.uniform() < (0.095 + 0.028 * max(0.0, burden)))
            states[uid] = {"present": present, "displaced": displaced, "distorted": distorted}

    # 목표 구간 정합 보정 (여전히 rng와 L0만 사용)
    detail_order = list(rng.permutation(TIER_2_IDS + TIER_1_IDS))
    for uid in detail_order:
        sc = compute_osterrieth_score(states)
        if sc >= lo:
            break
        states[uid] = {"present": True, "displaced": False, "distorted": False}

    for uid in detail_order:
        sc = compute_osterrieth_score(states)
        if sc <= hi:
            break
        if states[uid]["present"] and not states[uid]["distorted"]:
            states[uid]["distorted"] = True

    return states


def _sample_unit_states_mci(
    condition: str,
    s_mci: float,
    rng: np.random.Generator,
) -> dict[int, dict[str, bool]]:
    """MCI 정적 단위 상태 샘플링: s_mci는 (z_age, z_edu, rng)로만 결정되며 아밀로이드와 완전 독립."""
    states: dict[int, dict[str, bool]] = {}
    if condition == "copy":
        lo, hi = 16.0, 30.0
        for u in CANONICAL_UNITS:
            uid, tier = u["unit_id"], u["tier"]
            if uid == 2:
                present = True
            elif tier == 0:
                present = bool(rng.uniform() < (0.98 - 0.10 * s_mci))
            elif tier == 1:
                present = bool(rng.uniform() < (0.94 - 0.22 * s_mci))
            else:
                present = bool(rng.uniform() < (0.86 - 0.36 * s_mci))
            displaced = bool(present and rng.uniform() < (0.10 + 0.20 * s_mci))
            distorted = bool(present and rng.uniform() < (0.14 + 0.24 * s_mci))
            states[uid] = {"present": present, "displaced": displaced, "distorted": distorted}
    else:
        lo, hi = 4.0, 21.0
        for u in CANONICAL_UNITS:
            uid, tier = u["unit_id"], u["tier"]
            if uid == 2:
                present = True
            elif tier == 0:
                present = bool(rng.uniform() < (0.90 - 0.28 * s_mci))
            elif tier == 1:
                present = bool(rng.uniform() < (0.72 - 0.38 * s_mci))
            else:
                present = bool(rng.uniform() < (0.50 - 0.32 * s_mci))
            displaced = bool(present and rng.uniform() < (0.18 + 0.26 * s_mci))
            distorted = bool(present and rng.uniform() < (0.22 + 0.28 * s_mci))
            states[uid] = {"present": present, "displaced": displaced, "distorted": distorted}

    # 최소 4개 단위 출현 및 목표 점수 범위 유지
    order_up = list(rng.permutation(TIER_0_IDS + TIER_1_IDS + TIER_2_IDS))
    while sum(1 for v in states.values() if v["present"]) < 4:
        for uid in order_up:
            if not states[uid]["present"]:
                states[uid] = {"present": True, "displaced": True, "distorted": True}
                break

    for uid in order_up:
        sc = compute_osterrieth_score(states)
        if sc >= lo:
            break
        if not states[uid]["present"]:
            states[uid] = {"present": True, "displaced": False, "distorted": True}
        elif states[uid]["displaced"] and states[uid]["distorted"]:
            states[uid]["displaced"] = False
        elif states[uid]["distorted"] or states[uid]["displaced"]:
            states[uid]["displaced"] = False
            states[uid]["distorted"] = False

    order_down = list(rng.permutation(TIER_2_IDS + TIER_1_IDS))
    for uid in order_down:
        sc = compute_osterrieth_score(states)
        if sc <= hi:
            break
        if states[uid]["present"]:
            if not states[uid]["distorted"]:
                states[uid]["distorted"] = True
            elif not states[uid]["displaced"]:
                states[uid]["displaced"] = True
            elif sum(1 for v in states.values() if v["present"]) > 4:
                states[uid]["present"] = False

    return states


def _realize_unit_polylines_canvas(
    unit: dict[str, Any],
    state: dict[str, bool],
    placement: dict[str, float],
    v_draw: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """단위의 정적 상태·캔버스 배치·펜 속력만으로 5 Hz 점열을 생성한다 (theta_proc와 완전 무관)."""
    if not state["present"]:
        return []

    uid = unit["unit_id"]
    tier = unit["tier"]
    quadrant = unit["quadrant"]
    raw_polys = [p.copy() for p in unit["polylines"]]

    # Unit 11 원(28각형): 자연스러운 매끄러운 반경 흔들림 추가
    if uid == 11:
        ring = raw_polys[0]
        center = ring[:-1].mean(axis=0)
        angles = np.linspace(0.0, 2.0 * np.pi, len(ring))
        ph1, ph2 = rng.uniform(0.0, 2.0 * np.pi, size=2)
        radial_mod = 1.0 + 0.045 * np.sin(2.0 * angles + ph1) + 0.030 * np.cos(3.0 * angles + ph2)
        radial_mod[-1] = radial_mod[0]
        raw_polys[0] = center + (ring - center) * radial_mod[:, None]

    all_pts = np.vstack(raw_polys)
    u_center = all_pts.mean(axis=0)

    # 왜곡(distorted): 비등방 스케일 + 국소 회전/기울임
    if state["distorted"]:
        sx = float(rng.choice([rng.uniform(0.68, 0.80), rng.uniform(1.20, 1.34)]))
        sy = float(rng.choice([rng.uniform(0.70, 0.82), rng.uniform(1.18, 1.32)]))
        rot = np.deg2rad(rng.uniform(-14.0, 14.0))
        ca, sa = float(np.cos(rot)), float(np.sin(rot))
        rot_m = np.array([[ca, -sa], [sa, ca]])
        for i, p in enumerate(raw_polys):
            rel = (p - u_center) * np.array([sx, sy])
            raw_polys[i] = u_center + rel @ rot_m.T

    # 위치 이동(displaced): 도형 내부에서 눈에 띄는 평행 이동
    if state["displaced"]:
        shift_mag = rng.uniform(8.5, 14.0)
        shift_ang = rng.uniform(0.0, 2.0 * np.pi)
        disp = np.array([shift_mag * np.cos(shift_ang), shift_mag * np.sin(shift_ang)])
        raw_polys = [p + disp for p in raw_polys]

    # 전역 캔버스 변환 (fig_mm -> canvas_mm)
    scale = placement["scale"]
    rad = np.deg2rad(placement["tilt_deg"])
    ct, st = float(np.cos(rad)), float(np.sin(rad))
    rot_global = np.array([[ct, -st], [st, ct]])
    origin_fig = np.array([FIG_CX_MM, FIG_CY_MM])
    origin_canvas = np.array([CANVAS_CENTER_MM[0] + placement["dx0"], CANVAS_CENTER_MM[1] + placement["dy0"]])

    substrokes: list[dict[str, Any]] = []
    for sub_idx, poly in enumerate(raw_polys):
        # 미세한 손 떨림 (끝점 정렬 오차)
        tremor = rng.normal(0.0, 0.24, size=poly.shape)
        poly_j = poly + tremor
        pts_c = origin_canvas + ((poly_j - origin_fig) * scale) @ rot_global.T
        pts_c[:, 0] = np.clip(pts_c[:, 0], 3.0, CANVAS_W_MM - 3.0)
        pts_c[:, 1] = np.clip(pts_c[:, 1], 3.0, CANVAS_H_MM - 3.0)

        seg_vecs = np.diff(pts_c, axis=0)
        seg_lens = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])
        total_len = float(seg_lens.sum())

        # 곡률·길이에 따른 자연 감속 (오직 기하학적 특성만 반영, 아밀로이드와 완전 무관)
        if uid == 11 and sub_idx == 0:
            eff_speed = 0.42 * v_draw
        elif total_len < 12.0:
            eff_speed = 0.55 * v_draw
        else:
            eff_speed = v_draw

        step_mm = max(1.6, eff_speed * DT_SEC)
        if len(pts_c) <= 2:
            n_samples = max(2, int(round(total_len / step_mm)) + 1)
            s_grid = np.linspace(0.0, 1.0, n_samples)
            sampled = pts_c[0] + s_grid[:, None] * (pts_c[-1] - pts_c[0])
        else:
            cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
            if total_len < 1e-6:
                sampled = pts_c[:2]
            else:
                n_samples = max(len(pts_c), int(round(total_len / max(1.8, 0.65 * step_mm))) + 1)
                d_grid = np.linspace(0.0, total_len, n_samples)
                xs = np.interp(d_grid, cum, pts_c[:, 0])
                ys = np.interp(d_grid, cum, pts_c[:, 1])
                sampled = np.column_stack([xs, ys])

        substrokes.append({
            "unit_id": uid,
            "tier": tier,
            "quadrant": quadrant,
            "sub_idx": sub_idx,
            "points": sampled,
            "length_mm": total_len,
            "centroid_mm": sampled.mean(axis=0),
        })
    return substrokes


def _order_and_time_substrokes(
    unit_substrokes: dict[int, list[dict[str, Any]]],
    condition: str,
    theta_proc: float,
    rng: np.random.Generator,
    shift: bool = False,
) -> pd.DataFrame:
    """잠재 과정 요인 theta_proc에 따라 획 순서(위계 역전·재방문)와 5 Hz 경계 멈춤을 부여한다.

    주의: 개별 pen_down==1 획의 좌표 집합은 이미 정적으로 고정되어 있으므로,
    정적 PNG 픽셀과 획 셔플(shuffle_trial_strokes) 분포는 theta_proc에 대해 정확히 불변이다.
    """
    present_uids = [uid for uid, subs in unit_substrokes.items() if subs]
    u_pos = max(0.0, theta_proc + 0.55)

    # 위계 역전 및 단편화(재방문) 강도: 지연회상에서 훨씬 강하게 발현
    if condition == "copy":
        p_invert = float(expit(1.15 * (theta_proc - 1.35)))
        p_split = float(0.06 + 0.34 * expit(1.25 * (theta_proc - 0.85)))
        quad_perm = [0, 1, 2, 3] if rng.uniform() > 0.22 * p_invert else list(rng.permutation([0, 1, 2, 3]))
    else:
        p_invert = float(expit(1.75 * (theta_proc - 0.28)))
        p_split = float(0.07 + 0.68 * expit(1.55 * (theta_proc - 0.22)))
        if rng.uniform() < p_invert:
            # 모사(좌상->우상->우하->좌하)와 정반대·비정형 공간 진행 유도
            quad_perm = [2, 1, 3, 0] if rng.uniform() < 0.65 else [3, 2, 1, 0]
        else:
            quad_perm = [0, 1, 2, 3]

    quad_rank = {q: idx for idx, q in enumerate(quad_perm)}

    # 단위별 묶음을 (초기 묶음, 재방문 후기 묶음)으로 나눈다 (획 자체는 쪼개지 않고 기존 다중 보조획만 분리)
    chunks: list[dict[str, Any]] = []
    for uid in present_uids:
        subs = list(unit_substrokes[uid])
        tier = subs[0]["tier"]
        quad = subs[0]["quadrant"]

        if len(subs) >= 2 and rng.uniform() < p_split:
            cut = int(rng.integers(1, len(subs)))
            part_a, part_b = subs[:cut], subs[cut:]
            chunks.append({"uid": uid, "tier": tier, "quad": quad, "revisit": 0, "subs": part_a})
            chunks.append({"uid": uid, "tier": tier, "quad": quad, "revisit": 1, "subs": part_b})
        else:
            chunks.append({"uid": uid, "tier": tier, "quad": quad, "revisit": 0, "subs": subs})

    # 각 묶음의 우선순위 점수 계산:
    # - 낮은 theta_proc: Tier 0 -> Tier 1 -> Tier 2 (골격 우선) + 일관된 사분면 순서
    # - 높은 theta_proc (특히 Recall): 주변 Tier 2 세부 묶음이 Tier 0 골격 중앙선/대각선보다 먼저 시작됨
    invert_active = rng.uniform() < p_invert
    keyed_chunks: list[tuple[float, int, dict[str, Any]]] = []
    for idx, ch in enumerate(chunks):
        tier = ch["tier"]
        q_score = 0.22 * quad_rank[ch["quad"]]
        noise = float(rng.normal(0.0, 0.24))

        if not invert_active:
            base = float(tier) * 1.65 + q_score + 1.45 * ch["revisit"] + noise
        else:
            if condition == "recall":
                # 주변 세부 단위(Tier 2) 일부를 최선두에 배치하고 골격(Tier 0) 완성을 뒤로 미룸
                if tier == 2 and ch["revisit"] == 0 and rng.uniform() < (0.55 + 0.30 * p_invert):
                    tier_rank = 0.15
                elif tier == 1 and ch["revisit"] == 0 and rng.uniform() < 0.45:
                    tier_rank = 0.95
                elif tier == 0:
                    tier_rank = 1.75 + (0.65 if ch["uid"] in (3, 4, 5) else 0.20)
                else:
                    tier_rank = 2.45
                base = tier_rank + 0.55 * q_score + 1.80 * ch["revisit"] + noise
            else:
                tier_rank = 0.65 if (tier == 2 and rng.uniform() < 0.35) else float(tier) * 1.25
                base = tier_rank + 0.35 * q_score + 1.35 * ch["revisit"] + noise

        keyed_chunks.append((base, idx, ch))

    keyed_chunks.sort(key=lambda item: (item[0], item[1]))
    ordered_strokes: list[dict[str, Any]] = []
    for _, _, ch in keyed_chunks:
        ordered_strokes.extend(ch["subs"])

    # 5 Hz 타임라인 조립: 단위 내부 보조획 이동은 ~0.25s로 일정, 구조 경계만 u_pos에 지수 비례
    rows_t: list[float] = []
    rows_x: list[float] = []
    rows_y: list[float] = []
    rows_p: list[int] = []

    t_cur = 0.0
    u_cap = min(u_pos, 2.75)

    for s_idx, st in enumerate(ordered_strokes):
        pts = st["points"]
        if s_idx > 0:
            prev = ordered_strokes[s_idx - 1]
            p_prev = prev["points"][-1]
            p_next = pts[0]
            jump_mm = float(np.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1]))
            same_unit = prev["unit_id"] == st["unit_id"]
            cross_tier = prev["tier"] != st["tier"]
            scale_transition = (prev["tier"] == 0 and st["tier"] == 2) or (prev["tier"] == 2 and st["tier"] == 0)
            long_jump = jump_mm >= 35.0

            if same_unit and not long_jump:
                # 같은 단위 내부 보조획 이동: ~0.25s (1~2 샘플, theta_proc와 무관!)
                n_hover = 1 if rng.uniform() < 0.75 else 2
            elif cross_tier or long_jump or scale_transition:
                # 구조 경계 계획 멈춤: max(0, theta_proc + 0.55)에 따라 지수 증가
                mult = 1.25 if scale_transition else 1.0
                if long_jump:
                    mult *= 1.18
                jitter = float(np.clip(rng.lognormal(0.0, 0.22), 0.65, 1.55))
                if condition == "copy":
                    pause_s = 0.32 + 0.20 * mult * (np.exp(0.80 * u_cap) - 0.35) * jitter
                else:
                    pause_s = 0.38 + 0.34 * mult * (np.exp(1.02 * u_cap) - 0.30) * jitter
                n_hover = max(1, int(round(pause_s / DT_SEC)))
            else:
                # 같은 계층 인접 단위 이동: 짧은 기본 멈춤 + 약한 경향
                pause_s = 0.30 + (0.06 if condition == "copy" else 0.10) * u_cap * float(rng.uniform(0.7, 1.3))
                n_hover = max(1, int(round(pause_s / DT_SEC)))

            alphas = np.linspace(0.0, 1.0, n_hover + 2)[1:-1]
            hover_xy = p_prev[None, :] + alphas[:, None] * (p_next - p_prev)[None, :]
            if n_hover >= 3:
                # 긴 계획 멈춤 시 목표 지점 부근에서 미세하게 머무름
                hover_xy += rng.normal(0.0, 0.45, size=hover_xy.shape)

            keep_mask = np.ones(n_hover, dtype=bool)
            if shift and n_hover >= 2:
                drop_draw = rng.uniform(size=n_hover) < 0.06
                if drop_draw.all():
                    drop_draw[0] = False
                keep_mask = ~drop_draw

            for h_idx in range(n_hover):
                t_cur = round(t_cur + DT_SEC, 4)
                if keep_mask[h_idx]:
                    rows_t.append(t_cur)
                    rows_x.append(float(hover_xy[h_idx, 0]))
                    rows_y.append(float(hover_xy[h_idx, 1]))
                    rows_p.append(0)

        for pt_idx, pt in enumerate(pts):
            if s_idx > 0 or pt_idx > 0:
                t_cur = round(t_cur + DT_SEC, 4)
            rows_t.append(t_cur)
            rows_x.append(float(pt[0]))
            rows_y.append(float(pt[1]))
            rows_p.append(1)

    xs = np.asarray(rows_x, dtype=float)
    ys = np.asarray(rows_y, dtype=float)
    if shift:
        xs = np.clip(xs + rng.normal(0.0, 0.35, size=xs.shape), 0.0, CANVAS_W_MM)
        ys = np.clip(ys + rng.normal(0.0, 0.35, size=ys.shape), 0.0, CANVAS_H_MM)

    return pd.DataFrame({
        "t_sec": np.round(rows_t, 2),
        "x_mm": np.round(xs, 2),
        "y_mm": np.round(ys, 2),
        "pen_down": np.asarray(rows_p, dtype=int),
    })


def extract_pen_down_strokes(df: pd.DataFrame) -> list[pd.DataFrame]:
    """연속된 pen_down == 1 구간들을 순서대로 추출한다."""
    if df.empty:
        return []
    pen = df["pen_down"].to_numpy(dtype=int)
    strokes: list[pd.DataFrame] = []
    n = len(df)
    i = 0
    while i < n:
        if pen[i] == 1:
            j = i + 1
            while j < n and pen[j] == 1:
                j += 1
            strokes.append(df.iloc[i:j].copy())
            i = j
        else:
            i += 1
    return strokes


def shuffle_trial_strokes(
    df: pd.DataFrame,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """연속된 pen_down==1 획 순서를 무작위로 섞고 획 사이에 일정한 1샘플(0.20s) 이동을 넣는다.

    정적 PNG 픽셀은 100% 동일하게 유지하면서 시간 순서와 멈춤 길이 신호만 완전히 제거한다.
    """
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    strokes = extract_pen_down_strokes(df)
    if len(strokes) <= 1:
        return df.copy().reset_index(drop=True)

    perm = gen.permutation(len(strokes))
    rows_t: list[float] = []
    rows_x: list[float] = []
    rows_y: list[float] = []
    rows_p: list[int] = []

    t_cur = 0.0
    for out_idx, st_idx in enumerate(perm):
        st = strokes[int(st_idx)]
        xs = st["x_mm"].to_numpy(dtype=float)
        ys = st["y_mm"].to_numpy(dtype=float)

        if out_idx > 0:
            prev_st = strokes[int(perm[out_idx - 1])]
            mx = 0.5 * (float(prev_st["x_mm"].iloc[-1]) + float(xs[0]))
            my = 0.5 * (float(prev_st["y_mm"].iloc[-1]) + float(ys[0]))
            t_cur = round(t_cur + DT_SEC, 4)
            rows_t.append(t_cur)
            rows_x.append(round(mx, 2))
            rows_y.append(round(my, 2))
            rows_p.append(0)

        for k in range(len(xs)):
            if out_idx > 0 or k > 0:
                t_cur = round(t_cur + DT_SEC, 4)
            rows_t.append(t_cur)
            rows_x.append(float(xs[k]))
            rows_y.append(float(ys[k]))
            rows_p.append(1)

    return pd.DataFrame({
        "t_sec": np.round(rows_t, 2),
        "x_mm": rows_x,
        "y_mm": rows_y,
        "pen_down": np.asarray(rows_p, dtype=int),
    })


def render_static_png(
    df: pd.DataFrame,
    out_path: Path | str | None = None,
    bg_gray: int = 255,
    stroke_gray: int = 30,
) -> Image.Image:
    """2배 슈퍼샘플링(1344x960 -> 672x480) 단일 밝기(30) 8비트 그레이스케일 PNG 렌더링.

    동일한 잉크 밝기(30)로 덮어쓰므로 그리기 순서나 멈춤 시간이 정적 픽셀에 전혀 새지 않는다.
    """
    hi_w, hi_h = IMAGE_W_PX * SUPERSAMPLE, IMAGE_H_PX * SUPERSAMPLE
    sx = hi_w / CANVAS_W_MM
    sy = hi_h / CANVAS_H_MM

    canvas = Image.new("L", (hi_w, hi_h), color=int(bg_gray))
    draw = ImageDraw.Draw(canvas)

    if not df.empty:
        pen = df["pen_down"].to_numpy(dtype=int)
        xs = df["x_mm"].to_numpy(dtype=float) * sx
        ys = df["y_mm"].to_numpy(dtype=float) * sy
        n = len(pen)
        i = 0
        while i < n:
            if pen[i] == 1:
                j = i + 1
                while j < n and pen[j] == 1:
                    j += 1
                if j - i == 1:
                    x0, y0 = float(xs[i]), float(ys[i])
                    draw.ellipse((x0 - 2.0, y0 - 2.0, x0 + 2.0, y0 + 2.0), fill=int(stroke_gray))
                else:
                    pts = list(zip(xs[i:j].tolist(), ys[i:j].tolist()))
                    draw.line(pts, fill=int(stroke_gray), width=4, joint="curve")
                i = j
            else:
                i += 1

    img = canvas.resize((IMAGE_W_PX, IMAGE_H_PX), resample=Image.Resampling.BOX)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="PNG", compress_level=1)
    return img


def _time_color_rgb(frac: float) -> tuple[int, int, int]:
    """정규화 진행 시각(0=초기 파랑 -> 0.5=청록/황금 -> 1=후기 빨강) RGB 색상."""
    f = float(np.clip(frac, 0.0, 1.0))
    stops = [
        (0.00, (32, 82, 204)),
        (0.35, (28, 162, 168)),
        (0.68, (230, 159, 24)),
        (1.00, (208, 42, 42)),
    ]
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        if f <= t1:
            w = 0.0 if t1 == t0 else (f - t0) / (t1 - t0)
            return (
                int(round(c0[0] + w * (c1[0] - c0[0]))),
                int(round(c0[1] + w * (c1[1] - c0[1]))),
                int(round(c0[2] + w * (c1[2] - c0[2]))),
            )
    return stops[-1][1]


def _draw_trajectory_box(
    draw: ImageDraw.ImageDraw,
    df: pd.DataFrame,
    box: tuple[int, int, int, int],
    title: str,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(252, 253, 255), outline=(205, 212, 224), width=2)
    draw.text((x0 + 12, y0 + 8), title, fill=(25, 32, 48), font=font)

    pad_x, pad_top, pad_bot = 14, 30, 12
    pw = max(10, (x1 - x0) - 2 * pad_x)
    ph = max(10, (y1 - y0) - pad_top - pad_bot)

    def to_px(xm: float, ym: float) -> tuple[float, float]:
        return (
            x0 + pad_x + (xm / CANVAS_W_MM) * pw,
            y0 + pad_top + (ym / CANVAS_H_MM) * ph,
        )

    strokes = extract_pen_down_strokes(df)
    n_st = max(1, len(strokes))

    # 긴 이동 및 구조 경계 계획 멈춤 표시
    for idx in range(1, len(strokes)):
        prev_st = strokes[idx - 1]
        cur_st = strokes[idx]
        t_end = float(prev_st["t_sec"].iloc[-1])
        t_start = float(cur_st["t_sec"].iloc[0])
        pause_s = max(0.0, t_start - t_end)
        p_a = (float(prev_st["x_mm"].iloc[-1]), float(prev_st["y_mm"].iloc[-1]))
        p_b = (float(cur_st["x_mm"].iloc[0]), float(cur_st["y_mm"].iloc[0]))
        jump_mm = float(np.hypot(p_b[0] - p_a[0], p_b[1] - p_a[1]))

        xa, ya = to_px(*p_a)
        xb, yb = to_px(*p_b)
        if jump_mm >= 35.0:
            draw.line([(xa, ya), (xb, yb)], fill=(212, 216, 226), width=1)
        if pause_s >= 0.85:
            mx, my = 0.5 * (xa + xb), 0.5 * (ya + yb)
            rad = float(np.clip(3.0 + 2.8 * pause_s, 4.0, 14.0))
            draw.ellipse((mx - rad, my - rad, mx + rad, my + rad), outline=(186, 68, 178), width=2)

    # 시간 색상 부호화 획 그리기
    for idx, st in enumerate(strokes):
        col = _time_color_rgb(idx / max(1, n_st - 1))
        pts = [to_px(float(x), float(y)) for x, y in zip(st["x_mm"], st["y_mm"])]
        if len(pts) == 1:
            px, py = pts[0]
            draw.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=col)
        else:
            draw.line(pts, fill=col, width=3)


def render_process_panel(
    copy_df: pd.DataFrame,
    recall_df: pd.DataFrame,
    out_path_or_pil: Path | str | None = None,
) -> Image.Image:
    """모사·지연회상 시간 색상 궤적 + 멈춤 원 + 지연회상 4단계 누적 스냅샷 RGB 패널(960x540)."""
    width, height = 960, 540
    panel = Image.new("RGB", (width, height), color=(244, 246, 250))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    # 상단: 모사 vs 지연회상 시간 부호화 궤적
    _draw_trajectory_box(
        draw,
        copy_df,
        (12, 10, 474, 326),
        "COPY | Stroke Order (Blue=Early -> Red=Late) & Pauses (Rings)",
        font,
    )
    _draw_trajectory_box(
        draw,
        recall_df,
        (486, 10, 948, 326),
        "DELAYED RECALL | Stroke Order & Planning Pauses (Rings >=0.85s)",
        font,
    )

    # 중앙 범례 막대
    bar_x0, bar_y0, bar_w, bar_h = 315, 333, 330, 10
    for i in range(bar_w):
        c = _time_color_rgb(i / max(1, bar_w - 1))
        draw.line([(bar_x0 + i, bar_y0), (bar_x0 + i, bar_y0 + bar_h)], fill=c)
    draw.text((232, 331), "Early (0%)", fill=(32, 82, 204), font=font)
    draw.text((654, 331), "Late (100%) | Purple Ring = Long Pause", fill=(120, 42, 92), font=font)

    # 하단: 지연회상 25% / 50% / 75% / 100% 누적 진행 스냅샷
    r_strokes = extract_pen_down_strokes(recall_df)
    n_r = len(r_strokes)
    fracs = [0.25, 0.50, 0.75, 1.00]
    card_w = 226
    gap = 10
    start_x = 12
    top_y, bot_y = 350, 530

    for q_idx, frac in enumerate(fracs):
        cx0 = start_x + q_idx * (card_w + gap)
        cx1 = cx0 + card_w
        draw.rectangle((cx0, top_y, cx1, bot_y), fill=(255, 255, 255), outline=(210, 216, 226), width=1)
        draw.text((cx0 + 8, top_y + 6), f"Recall Progress {int(frac * 100)}%", fill=(35, 42, 58), font=font)

        prev_cnt = 0 if q_idx == 0 else int(np.ceil(fracs[q_idx - 1] * n_r))
        cur_cnt = int(np.ceil(frac * n_r))
        pw, ph = card_w - 16, (bot_y - top_y) - 30

        for s_i in range(min(cur_cnt, n_r)):
            st = r_strokes[s_i]
            pts = [
                (
                    cx0 + 8 + (float(x) / CANVAS_W_MM) * pw,
                    top_y + 24 + (float(y) / CANVAS_H_MM) * ph,
                )
                for x, y in zip(st["x_mm"], st["y_mm"])
            ]
            col = (206, 44, 54) if s_i >= prev_cnt else (58, 68, 84)
            w_line = 2 if s_i >= prev_cnt else 2
            if len(pts) == 1:
                px, py = pts[0]
                draw.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=col)
            else:
                draw.line(pts, fill=col, width=w_line)

    if out_path_or_pil is not None and isinstance(out_path_or_pil, (str, Path)):
        Path(out_path_or_pil).parent.mkdir(parents=True, exist_ok=True)
        panel.save(out_path_or_pil, format="PNG", compress_level=1)
    return panel


def simulate_subject_trials(
    subject_row: dict[str, Any],
    beta_ab: float,
    static_seed: int,
    process_seed: int,
    shift: bool = False,
) -> dict[str, Any]:
    """단일 참가자의 정적 기하(static_seed)·시공간 과정(process_seed)·학습 전용 뇌지표를 생성한다.

    엄수 불변량:
      - 정적 36점 채점과 정적 PNG 픽셀, 펜 속력(v_draw)은 diagnosis가 주어졌을 때
        amyloid_pos 및 amyloid_centiloid와 정확히 독립이다 (진단군 내 정적 AUROC = 0.50).
      - --null-effect(beta_ab == 0.0)에서는 theta_proc 역시 diagnosis 조건부로 amyloid와
        정확히 독립이며, 학습 전용 뇌지표(hippocampal_md 등)는 계속 amyloid·진단을 반영한다.
    """
    rng_static = np.random.default_rng(static_seed)
    rng_proc = np.random.default_rng(process_seed)

    age = float(subject_row["age"])
    is_mci = 1 if subject_row["diagnosis"] == "MCI" else 0
    amyloid_pos = int(subject_row["amyloid_pos"])
    centiloid = float(subject_row["amyloid_centiloid"])

    z_age = (age - 71.5) / 6.0
    z_cl = (centiloid - 30.0) / 35.0

    # 비특이 혈관성 운동 감속 요인 (아밀로이드와 완전 독립, wmh_vol_log 및 v_draw와만 연결)
    z_vasc = float(rng_static.normal(0.0, 1.0))

    # 펜이 닿아 있는 동안의 속력(mm/s): diagnosis와 비특이 혈관 잡음에만 의존 (아밀로이드 독립!)
    v_draw = float(np.clip(rng_static.normal(38.0 - 5.5 * is_mci - 2.2 * z_vasc, 3.8), 20.0, 55.0))

    # 전역 배치를 모사·지연회상 독립 샘플링
    placements = {}
    for cond in ("copy", "recall"):
        placements[cond] = {
            "scale": float(rng_static.uniform(0.92, 1.04)),
            "tilt_deg": float(rng_static.uniform(-2.0, 2.0)),
            "dx0": float(rng_static.uniform(-4.5, 4.5)),
            "dy0": float(rng_static.uniform(-4.5, 4.5)),
        }

    # 현상 1: 정적 기하·누락·왜곡은 diagnosis와 rng_static에만 의존 (age/amyloid 누출 0)
    if is_mci:
        s_mci = float(np.clip(0.50 + rng_static.normal(0.0, 0.18), 0.12, 0.92))
        states_copy = _sample_unit_states_mci("copy", s_mci, rng_static)
        states_recall = _sample_unit_states_mci("recall", s_mci, rng_static)
    else:
        s_mci = 0.0
        states_copy = _sample_unit_states_cu("copy", 0.0, 0.0, rng_static)
        states_recall = _sample_unit_states_cu("recall", 0.0, 0.0, rng_static)

    copy_score = compute_osterrieth_score(states_copy)
    recall_score = compute_osterrieth_score(states_recall)

    subs_copy = {
        u["unit_id"]: _realize_unit_polylines_canvas(u, states_copy[u["unit_id"]], placements["copy"], v_draw, rng_static)
        for u in CANONICAL_UNITS
    }
    subs_recall = {
        u["unit_id"]: _realize_unit_polylines_canvas(u, states_recall[u["unit_id"]], placements["recall"], v_draw, rng_static)
        for u in CANONICAL_UNITS
    }

    # 현상 2: 아밀로이드 조건부 과정 잠재 요인 (연속 Centiloid + 이진 양성 반영, 진단군 내 age 교란 제거)
    ab_process_component = (
        float(beta_ab) * (0.72 * amyloid_pos + 0.36 * float(np.clip(z_cl, -0.8, 2.2)))
        if beta_ab > 0.0
        else 0.0
    )
    proc_noise = float(rng_proc.normal(0.0, 0.64))
    theta_proc = float(ab_process_component + 0.55 * is_mci + proc_noise)

    # 학습 전용(train_privileged.csv) 합성 뇌영상 지표 (평가 test에서는 완전히 숨겨짐)
    # 미세구조 확산 지표(hippocampal_md, precuneus_md)는 조기 아밀로이드·과정 부하에 민감하고,
    # 거시 위축 지표(hippocampal_vol, entorhinal_thickness)는 MCI 위축·연령에 주로 반응한다.
    rng_brain = np.random.default_rng(static_seed ^ 0x5A3C9E1B)
    shared_micro = ab_process_component + 0.35 * proc_noise if beta_ab > 0.0 else 0.0
    hippocampal_md = float(
        np.clip(
            0.860
            + 0.055 * is_mci
            + 0.020 * z_age
            + 0.036 * amyloid_pos
            + 0.024 * float(np.clip(z_cl, -0.8, 2.2))
            + 0.028 * float(np.clip(shared_micro, -1.5, 2.5))
            + rng_brain.normal(0.0, 0.024),
            0.720,
            1.350,
        )
    )
    precuneus_md = float(
        np.clip(
            0.800
            + 0.036 * is_mci
            + 0.016 * z_age
            + 0.040 * amyloid_pos
            + 0.026 * float(np.clip(z_cl, -0.8, 2.2))
            + 0.032 * float(np.clip(shared_micro, -1.5, 2.5))
            + rng_brain.normal(0.0, 0.022),
            0.680,
            1.280,
        )
    )
    parietal_thickness = float(
        np.clip(
            2.340
            - 0.145 * is_mci
            - 0.045 * z_age
            - 0.048 * amyloid_pos
            - 0.040 * s_mci
            - 0.022 * float(np.clip(shared_micro, -1.5, 2.5))
            + rng_brain.normal(0.0, 0.062),
            1.650,
            2.800,
        )
    )
    hippocampal_vol = float(
        np.clip(
            3.840
            - 0.540 * is_mci
            - 0.135 * z_age
            - 0.065 * amyloid_pos
            - 0.220 * s_mci
            + rng_brain.normal(0.0, 0.155),
            2.100,
            4.650,
        )
    )
    entorhinal_thickness = float(
        np.clip(
            3.360
            - 0.360 * is_mci
            - 0.085 * z_age
            - 0.075 * amyloid_pos
            - 0.140 * s_mci
            + rng_brain.normal(0.0, 0.115),
            2.050,
            4.050,
        )
    )
    wmh_vol_log = float(
        np.clip(
            0.460
            + 0.340 * z_age
            + 0.160 * is_mci
            + 0.300 * z_vasc
            + rng_brain.normal(0.0, 0.200),
            -0.600,
            2.200,
        )
    )

    copy_df = _order_and_time_substrokes(subs_copy, "copy", theta_proc, rng_proc, shift=shift)
    recall_df = _order_and_time_substrokes(subs_recall, "recall", theta_proc, rng_proc, shift=shift)

    return {
        "copy_score_36": copy_score,
        "recall_score_36": recall_score,
        "hippocampal_md": round(hippocampal_md, 4),
        "hippocampal_vol": round(hippocampal_vol, 3),
        "entorhinal_thickness": round(entorhinal_thickness, 3),
        "precuneus_md": round(precuneus_md, 4),
        "parietal_thickness": round(parietal_thickness, 3),
        "wmh_vol_log": round(wmh_vol_log, 3),
        "v_draw_mm_s": round(v_draw, 3),
        "s_mci": round(s_mci, 4),
        "theta_proc": round(theta_proc, 4),
        "copy_df": copy_df,
        "recall_df": recall_df,
    }


def compute_within_dx_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    diagnoses: np.ndarray,
) -> dict[str, float]:
    """전체 CU+MCI AUROC와 진단군 내 층화 AUROC(CU 단독·MCI 단독·표본수 가중평균)를 계산한다."""
    from sklearn.metrics import roc_auc_score

    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_prob, dtype=float)
    dx_arr = np.asarray(diagnoses, dtype=str)

    def _auc(yt: np.ndarray, yp: np.ndarray) -> float:
        if yt.size < 2 or np.unique(yt).size < 2:
            return float("nan")
        return float(roc_auc_score(yt, yp))

    pooled = _auc(y_arr, p_arr)
    cu_mask = dx_arr == "CU"
    mci_mask = dx_arr == "MCI"
    auc_cu = _auc(y_arr[cu_mask], p_arr[cu_mask])
    auc_mci = _auc(y_arr[mci_mask], p_arr[mci_mask])

    parts: list[tuple[int, float]] = []
    if not np.isnan(auc_cu):
        parts.append((int(cu_mask.sum()), auc_cu))
    if not np.isnan(auc_mci):
        parts.append((int(mci_mask.sum()), auc_mci))

    if parts:
        total_w = sum(w for w, _ in parts)
        within_dx = float(sum(w * v for w, v in parts) / total_w)
    else:
        within_dx = float("nan")

    return {
        "auroc_pooled": round(pooled, 4) if not np.isnan(pooled) else float("nan"),
        "auroc_cu": round(auc_cu, 4) if not np.isnan(auc_cu) else float("nan"),
        "auroc_mci": round(auc_mci, 4) if not np.isnan(auc_mci) else float("nan"),
        "within_dx_auroc": round(within_dx, 4) if not np.isnan(within_dx) else float("nan"),
    }


def simulate_cohort_metadata(
    n_subjects: int = 400,
    seed: int = 42,
    stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """N명의 인구통계·유전형·임상진단·아밀로이드 라벨 및 연속 Centiloid 생성."""
    st = default_stats() if stats is None else stats
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(st["age_mean"], st["age_sd"], size=n_subjects), 58.0, 88.0)
    z_age = (age - st["age_mean"]) / st["age_sd"]
    sex = rng.choice(["F", "M"], size=n_subjects, p=[1.0 - st["male_prob"], st["male_prob"]])
    edu_years = np.clip(np.round(rng.normal(st["edu_mean"] - 0.8 * z_age, st["edu_sd"], size=n_subjects)), 4.0, 20.0)
    z_edu = (edu_years - st["edu_mean"]) / st["edu_sd"]
    apoe4_count = rng.choice([0, 1, 2], size=n_subjects, p=st["apoe4_probs"])

    mc = st["mci_logit"]
    logit_mci = mc["intercept"] + mc["z_age"] * z_age + mc["z_edu"] * z_edu + mc["apoe4"] * apoe4_count
    is_mci = (rng.uniform(size=n_subjects) < expit(logit_mci)).astype(int)
    diagnosis = np.where(is_mci == 1, "MCI", "CU")

    ab = st["amyloid_logit"]
    logit_ab = (
        ab["intercept"]
        + ab["mci"] * is_mci
        + ab["age_per_10y_from_70"] * ((age - 70.0) / 10.0)
        + ab["apoe4"] * apoe4_count
        + ab["z_edu"] * z_edu
    )
    amyloid_pos = (rng.uniform(size=n_subjects) < expit(logit_ab)).astype(int)

    # 연속형 아밀로이드 PET 부하 (Centiloid, 양성 기준 ~25 CL): train 특권 회귀·정렬 타깃
    cl_neg = np.clip(rng.normal(7.5 + 2.5 * is_mci + 1.8 * z_age, 8.0, size=n_subjects), -5.0, 24.5)
    cl_pos = np.clip(
        rng.normal(56.0 + 16.0 * is_mci + 6.0 * apoe4_count + 3.5 * z_age, 17.0, size=n_subjects),
        25.5,
        130.0,
    )
    amyloid_centiloid = np.where(amyloid_pos == 1, cl_pos, cl_neg)

    return pd.DataFrame({
        "subject_id": [f"sub-{i:04d}" for i in range(1, n_subjects + 1)],
        "age": np.round(age, 1),
        "sex": sex,
        "edu_years": edu_years.astype(float),
        "apoe4_count": apoe4_count.astype(int),
        "diagnosis": diagnosis,
        "amyloid_pos": amyloid_pos.astype(int),
        "amyloid_centiloid": np.round(amyloid_centiloid, 1),
    })
