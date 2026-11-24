"""ROCF 5 Hz 펜 궤적 시각화 패널 및 진단군 내 층화 평가 지표 유틸리티.

공개 챌린지 노트북과 분석 코드에서 다음 두 기능을 제공합니다.
- `render_process_panel(copy_df, recall_df)`: 시간 순서 색상·긴 멈춤 표시·지연 회상 진행 스냅샷 RGB 패널 생성
- `compute_within_dx_metrics(y_true, y_prob, diagnoses)`: 전체 및 진단군(`CU`, `MCI`) 내부 층화 AUROC 계산
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CANVAS_W_MM = 210.0
CANVAS_H_MM = 150.0


def extract_pen_down_strokes(df: pd.DataFrame) -> list[pd.DataFrame]:
    """연속된 `pen_down == 1` 구간들을 시간 순서대로 추출한다."""
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
            strokes.append(df.iloc[i:j].reset_index(drop=True))
            i = j
        else:
            i += 1
    return strokes


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
    """모사·지연회상 시간 색상 궤적 + 긴 멈춤 표시 + 지연회상 4단계 누적 스냅샷 RGB 패널(960x540)."""
    width, height = 960, 540
    panel = Image.new("RGB", (width, height), color=(244, 246, 250))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

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

    bar_x0, bar_y0, bar_w, bar_h = 315, 333, 330, 10
    for i in range(bar_w):
        c = _time_color_rgb(i / max(1, bar_w - 1))
        draw.line([(bar_x0 + i, bar_y0), (bar_x0 + i, bar_y0 + bar_h)], fill=c)
    draw.text((232, 331), "Early (0%)", fill=(32, 82, 204), font=font)
    draw.text((654, 331), "Late (100%) | Purple Ring = Long Pause", fill=(120, 42, 92), font=font)

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
            if len(pts) == 1:
                px, py = pts[0]
                draw.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=col)
            else:
                draw.line(pts, fill=col, width=2)

    if out_path_or_pil is not None and isinstance(out_path_or_pil, (str, Path)):
        Path(out_path_or_pil).parent.mkdir(parents=True, exist_ok=True)
        panel.save(out_path_or_pil, format="PNG", compress_level=1)
    return panel


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
