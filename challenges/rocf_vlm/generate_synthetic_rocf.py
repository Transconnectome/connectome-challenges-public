"""합성 ROCF 특권 정보 활용 학습(Learning Using Privileged Information) 코호트 생성기: 학습 전용 뇌·임상 특권 정보 + 평가 시 ROCF 그림·과정 전용 입력.

    python generate_synthetic_rocf.py                           # 기본값(N=400, seed=42) 생성
    python generate_synthetic_rocf.py --seed 7 --out data_seed7
    python generate_synthetic_rocf.py --null-effect --out data_null

출력 (data/):
    participants.csv      참가자 분할 표 (subject_id,split 만 포함 — test 시점 ROCF 전용 보장)
    train_privileged.csv  학습(train) 전용 특권 정보 (아밀로이드 라벨·Centiloid·해마 MD 등 뇌지표·임상·36점 채점)
    images/*.png          모사·지연회상 672x480 8비트 정적 그레이스케일 이미지
    strokes/*.csv         5 Hz 디지털 태블릿 궤적 (t_sec,x_mm,y_mm,pen_down)
    ground_truth.json     정답(참가자 파이프라인 열람 금지): test 라벨·진단·참고 채점·기준 지표
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rocf_synth import (
    compute_within_dx_metrics,
    default_stats,
    render_static_png,
    simulate_cohort_metadata,
    simulate_subject_trials,
)

TRAIN_PRIVILEGED_COLUMNS = [
    "subject_id",
    "amyloid_pos",
    "amyloid_centiloid",
    "diagnosis",
    "age",
    "sex",
    "edu_years",
    "apoe4_count",
    "copy_score_36",
    "recall_score_36",
    "hippocampal_md",
    "hippocampal_vol",
    "entorhinal_thickness",
    "precuneus_md",
    "parietal_thickness",
    "wmh_vol_log",
]


def make_stratified_split(df: pd.DataFrame, seed: int, test_fraction: float = 0.30) -> np.ndarray:
    """진단(diagnosis)과 아밀로이드 상태(amyloid_pos)를 동시에 층화한 70/30 train/test 분할."""
    strata = df["diagnosis"].astype(str) + "_" + df["amyloid_pos"].astype(str)
    counts = strata.value_counts()
    if (counts < 2).any():
        strata = df["amyloid_pos"].astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed + 1009)
    _, test_idx = next(splitter.split(np.zeros(len(df)), strata))
    split = np.full(len(df), "train", dtype=object)
    split[test_idx] = "test"
    return split


def _write_stroke_csv_fast(df: pd.DataFrame, out_path: Path) -> None:
    """pandas to_csv 오버헤드 없이 정확한 4열 태블릿 CSV를 빠르게 기록한다."""
    t_arr = df["t_sec"].to_numpy(dtype=float)
    x_arr = df["x_mm"].to_numpy(dtype=float)
    y_arr = df["y_mm"].to_numpy(dtype=float)
    p_arr = df["pen_down"].to_numpy(dtype=int)
    lines = ["t_sec,x_mm,y_mm,pen_down"]
    lines.extend(f"{t:.2f},{x:.2f},{y:.2f},{p:d}" for t, x, y, p in zip(t_arr, x_arr, y_arr, p_arr))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_reference_metrics(full_df: pd.DataFrame, theta_arr: np.ndarray) -> dict[str, float]:
    """ground_truth.json 기록용 기준 모형(정적 36점 vs 잠재 theta_proc 오라클) test 지표."""
    is_train = (full_df["split"] == "train").to_numpy()
    is_test = ~is_train
    y = full_df["amyloid_pos"].to_numpy(dtype=int)
    dx_te = full_df.loc[is_test, "diagnosis"].to_numpy(dtype=str)
    y_te = y[is_test]

    x_static = np.column_stack([
        full_df["copy_score_36"].to_numpy(dtype=float),
        full_df["recall_score_36"].to_numpy(dtype=float),
    ])
    x_oracle = np.column_stack([theta_arr, x_static])

    def fit_pred(mat: np.ndarray) -> np.ndarray:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
        clf.fit(mat[is_train], y[is_train])
        return clf.predict_proba(mat[is_test])[:, 1]

    m_stat = compute_within_dx_metrics(y_te, fit_pred(x_static), dx_te)
    m_ora = compute_within_dx_metrics(y_te, fit_pred(x_oracle), dx_te)

    return {
        "test_auroc_pooled_static_36pt": m_stat["auroc_pooled"],
        "test_auroc_cu_static_36pt": m_stat["auroc_cu"],
        "test_auroc_mci_static_36pt": m_stat["auroc_mci"],
        "test_within_dx_auroc_static_36pt": m_stat["within_dx_auroc"],
        "test_auroc_pooled_oracle_process": m_ora["auroc_pooled"],
        "test_auroc_cu_oracle_process": m_ora["auroc_cu"],
        "test_auroc_mci_oracle_process": m_ora["auroc_mci"],
        "test_within_dx_auroc_oracle_process": m_ora["within_dx_auroc"],
    }


def generate_dataset(
    out_dir: Path,
    seed: int = 42,
    n_subjects: int = 400,
    null_effect: bool = False,
    amyloid_effect: float = 0.80,
    distribution_shift: bool = False,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """전체 합성 ROCF 특권 정보 활용 학습(Learning Using Privileged Information) 데이터셋을 생성해 out_dir에 저장하고 ground_truth 사전을 반환한다."""
    st = default_stats() if stats is None else stats
    beta_ab = 0.0 if null_effect else float(amyloid_effect)

    meta = simulate_cohort_metadata(n_subjects=n_subjects, seed=seed, stats=st)
    meta.insert(1, "split", make_stratified_split(meta, seed=seed))

    ss_root = np.random.SeedSequence(seed)
    subj_seqs = ss_root.spawn(n_subjects)

    images_dir = out_dir / "images"
    strokes_dir = out_dir / "strokes"
    images_dir.mkdir(parents=True, exist_ok=True)
    strokes_dir.mkdir(parents=True, exist_ok=True)

    bg_gray = 246 if distribution_shift else 255
    sim_cols: dict[str, np.ndarray] = {
        k: np.zeros(n_subjects, dtype=float)
        for k in (
            "copy_score_36",
            "recall_score_36",
            "hippocampal_md",
            "hippocampal_vol",
            "entorhinal_thickness",
            "precuneus_md",
            "parietal_thickness",
            "wmh_vol_log",
            "theta_proc",
        )
    }

    records = meta.to_dict("records")
    seeds_pair = []
    for idx in range(n_subjects):
        s_static, s_proc = subj_seqs[idx].spawn(2)
        seeds_pair.append((int(s_static.generate_state(1)[0]), int(s_proc.generate_state(1)[0])))

    for idx in range(n_subjects):
        row = records[idx]
        static_seed, proc_seed = seeds_pair[idx]
        sim = simulate_subject_trials(
            subject_row=row,
            beta_ab=beta_ab,
            static_seed=static_seed,
            process_seed=proc_seed,
            shift=distribution_shift,
        )
        sid = row["subject_id"]
        _write_stroke_csv_fast(sim["copy_df"], strokes_dir / f"{sid}_copy.csv")
        _write_stroke_csv_fast(sim["recall_df"], strokes_dir / f"{sid}_recall.csv")
        render_static_png(sim["copy_df"], images_dir / f"{sid}_copy.png", bg_gray=bg_gray)
        render_static_png(sim["recall_df"], images_dir / f"{sid}_recall.png", bg_gray=bg_gray)
        for col in sim_cols:
            sim_cols[col][idx] = sim[col]

    for col, vals in sim_cols.items():
        if col != "theta_proc":
            meta[col] = vals
    theta_vals = sim_cols["theta_proc"]

    # 요약 지표 계산(결측 주입 전 참값 기준)
    cu_mask = meta["diagnosis"] == "CU"
    mci_mask = meta["diagnosis"] == "MCI"
    cohort_summary = {
        "n_total": int(len(meta)),
        "n_train": int((meta["split"] == "train").sum()),
        "n_test": int((meta["split"] == "test").sum()),
        "cu_count": int(cu_mask.sum()),
        "mci_count": int(mci_mask.sum()),
        "overall_amyloid_pos_rate": round(float(meta["amyloid_pos"].mean()), 4),
        "cu_amyloid_pos_rate": round(float(meta.loc[cu_mask, "amyloid_pos"].mean()), 4),
        "mci_amyloid_pos_rate": round(float(meta.loc[mci_mask, "amyloid_pos"].mean()), 4),
        "cu_copy_mean": round(float(meta.loc[cu_mask, "copy_score_36"].mean()), 2),
        "cu_copy_sd": round(float(meta.loc[cu_mask, "copy_score_36"].std(ddof=1)), 2),
        "cu_recall_mean": round(float(meta.loc[cu_mask, "recall_score_36"].mean()), 2),
        "cu_recall_sd": round(float(meta.loc[cu_mask, "recall_score_36"].std(ddof=1)), 2),
        "mci_copy_mean": round(float(meta.loc[mci_mask, "copy_score_36"].mean()), 2),
        "mci_copy_sd": round(float(meta.loc[mci_mask, "copy_score_36"].std(ddof=1)), 2),
        "mci_recall_mean": round(float(meta.loc[mci_mask, "recall_score_36"].mean()), 2),
        "mci_recall_sd": round(float(meta.loc[mci_mask, "recall_score_36"].std(ddof=1)), 2),
        "cu_hippocampal_md_ab_neg": round(float(meta.loc[cu_mask & (meta["amyloid_pos"] == 0), "hippocampal_md"].mean()), 4),
        "cu_hippocampal_md_ab_pos": round(float(meta.loc[cu_mask & (meta["amyloid_pos"] == 1), "hippocampal_md"].mean()), 4),
    }
    ref_metrics = _compute_reference_metrics(meta, theta_vals)

    # 1) participants.csv: subject_id,split 만 기록 (평가 시점 ROCF 그림·과정 전용 입력 강제)
    meta[["subject_id", "split"]].to_csv(out_dir / "participants.csv", index=False)

    # 2) train_privileged.csv: train 참가자의 모든 뇌지표·임상·채점·정답 제공 (test 행은 0개!)
    train_priv = meta.loc[meta["split"] == "train", TRAIN_PRIVILEGED_COLUMNS].copy().reset_index(drop=True)
    if distribution_shift:
        rng_miss = np.random.default_rng(seed + 7717)
        n_miss = max(1, int(round(0.05 * len(train_priv))))
        for col in ("hippocampal_md", "precuneus_md", "edu_years", "copy_score_36"):
            miss_idx = rng_miss.choice(len(train_priv), size=n_miss, replace=False)
            train_priv.loc[miss_idx, col] = np.nan
    train_priv.to_csv(out_dir / "train_privileged.csv", index=False, na_rep="")

    # 3) ground_truth.json: 블라인드 test 정답·진단·참고 36점 (채점 시 디렉터리 밖으로 격리)
    test_rows = meta[meta["split"] == "test"]
    test_labels = {str(r.subject_id): int(r.amyloid_pos) for r in test_rows.itertuples(index=False)}
    test_diagnoses = {str(r.subject_id): str(r.diagnosis) for r in test_rows.itertuples(index=False)}
    test_copy_36 = {str(r.subject_id): float(r.copy_score_36) for r in test_rows.itertuples(index=False)}
    test_recall_36 = {str(r.subject_id): float(r.recall_score_36) for r in test_rows.itertuples(index=False)}

    gt_payload = {
        "note": (
            "연습용 합성 ROCF 특권 정보 활용 학습(Learning Using Privileged Information) 코호트 정답 파일입니다. 제출 파이프라인은 이 파일을 절대 읽으면 안 됩니다. "
            "학습(train)에는 train_privileged.csv의 모든 뇌·임상 특권 정보를 활용할 수 있으나, "
            "평가(test)에서는 오직 images/*.png(그림)와 strokes/*.csv(5 Hz 그리는 과정)만 입력으로 주어집니다. "
            "정적 그림은 diagnosis 조건부로 아밀로이드와 정확히 독립(진단군 내 AUROC ≈ 0.50)입니다."
        ),
        "parameters": {
            "seed": int(seed),
            "n_subjects": int(n_subjects),
            "null_effect": bool(null_effect),
            "amyloid_effect": float(amyloid_effect),
            "effective_beta_ab": float(beta_ab),
            "distribution_shift": bool(distribution_shift),
            "sampling_hz": 5.0,
            "canvas_mm": [210.0, 150.0],
            "image_px": [672, 480],
        },
        "stats_source": st.get("source", {"method": "default", "note": "실제 데이터에서 추출하지 않은 예시 값"}),
        "cohort_summary": cohort_summary,
        "test_labels": test_labels,
        "test_diagnoses": test_diagnoses,
        "test_copy_scores_36": test_copy_36,
        "test_recall_scores_36": test_recall_36,
        "reference_metrics": ref_metrics,
    }
    (out_dir / "ground_truth.json").write_text(
        json.dumps(gt_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return gt_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42, help="난수 시드 (기본값: 42)")
    parser.add_argument("--n-subjects", type=int, default=400, help="생성할 참가자 수 (기본값: 400)")
    parser.add_argument("--null-effect", action="store_true", help="진단군 내 아밀로이드 과정 신호를 0으로 설정 (거짓 양성 검정용)")
    parser.add_argument("--amyloid-effect", type=float, default=0.80, help="잠재 과정 요인의 아밀로이드 효과 크기 (기본값: 0.80)")
    parser.add_argument("--distribution-shift", action="store_true", help="센서 잡음·호버 누락·배경 밝기 변화·학습 특권표 5%% 결측 주입")
    parser.add_argument(
        "--stats",
        type=Path,
        help="선택: 그룹 요약 통계 JSON. 연습 과제에는 필요 없고 공개본에는 통계 파일이 없습니다. 생략하면 기본값으로 생성",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data", help="출력 디렉터리")
    args = parser.parse_args()

    stats = json.loads(args.stats.read_text(encoding="utf-8")) if args.stats else default_stats()
    gt = generate_dataset(
        out_dir=args.out,
        seed=args.seed,
        n_subjects=args.n_subjects,
        null_effect=args.null_effect,
        amyloid_effect=args.amyloid_effect,
        distribution_shift=args.distribution_shift,
        stats=stats,
    )
    cs = gt["cohort_summary"]
    print(f"저장 위치: {args.out.resolve()}")
    print(
        f"총 {cs['n_total']}명 (train {cs['n_train']} / test {cs['n_test']}) | "
        f"CU {cs['cu_count']}명 (Aβ+ {cs['cu_amyloid_pos_rate']*100:.1f}%, Copy {cs['cu_copy_mean']:.1f}, Recall {cs['cu_recall_mean']:.1f}) | "
        f"MCI {cs['mci_count']}명 (Aβ+ {cs['mci_amyloid_pos_rate']*100:.1f}%, Copy {cs['mci_copy_mean']:.1f}, Recall {cs['mci_recall_mean']:.1f})"
    )


if __name__ == "__main__":
    main()
