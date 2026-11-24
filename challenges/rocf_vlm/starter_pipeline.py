"""ROCF 아밀로이드 예측 스타터 템플릿 (실행 규격 + 정적 그림 기준선).

    python starter_pipeline.py --data-dir data --out-dir submissions/starter --device cpu

안내:
    - 이 스타터는 제출 규격(--data-dir, --out-dir, --device, --seed, predictions.csv, summary.json)과
      완성된 정적 흑백 그림(images/*.png)만 사용하는 간단한 기준선을 제공합니다.
    - 정적 그림만 사용하는 기준선으로, 자체 검증된 신호를 주장하지 않도록 claim_signal=false를 출력합니다.
    - 최종 제출 파이프라인(run_pipeline.py)에서는 그리는 과정(strokes/*.csv)과 학습 전용 뇌·임상 정보
      (train_privileged.csv)를 함께 활용하고, 신호가 진정한지 스스로 검증한 결과를 반영하세요.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_static_image_features(copy_png: Path, recall_png: Path) -> np.ndarray:
    """정적 흑백 PNG에서만 잉크 밀도·사분면 분포·모사 대비 회상 유지율을 추출한다.

    배경 밝기 차이에 견디도록 95백분위 배경 추정 적응 임계값을 사용한다.
    """
    feats: list[float] = []
    ink_means: list[float] = []
    quad_vecs: list[np.ndarray] = []

    for png_path in (copy_png, recall_png):
        arr = np.asarray(Image.open(png_path).convert("L"), dtype=float)
        bg = float(np.percentile(arr, 95))
        ink = (arr < (bg - 45.0)).astype(float)
        mean_ink = float(ink.mean())
        ink_means.append(mean_ink)

        h2, w2 = ink.shape[0] // 2, ink.shape[1] // 2
        qv = np.array([
            ink[:h2, :w2].mean(),
            ink[:h2, w2:].mean(),
            ink[h2:, :w2].mean(),
            ink[h2:, w2:].mean(),
        ], dtype=float)
        quad_vecs.append(qv)

        ys, xs = np.nonzero(ink > 0.5)
        if xs.size > 0:
            bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / float(ink.size)
        else:
            bbox_area = 0.0

        feats.extend([mean_ink, float(qv.std()), bbox_area, *qv.tolist()])

    retention = float(ink_means[1] / (ink_means[0] + 1e-6))
    q_corr = 0.0
    if np.std(quad_vecs[0]) > 1e-6 and np.std(quad_vecs[1]) > 1e-6:
        q_corr = float(np.corrcoef(quad_vecs[0], quad_vecs[1])[0, 1])
    feats.extend([retention, q_corr])
    return np.asarray(feats, dtype=float)


def run_pipeline(
    data_dir: Path,
    out_dir: Path,
    requested_device: str = "cpu",
    seed: int = 42,
) -> dict:
    """정적 그림 기준선 학습과 블라인드 test 예측을 수행한다."""
    participants = pd.read_csv(data_dir / "participants.csv", dtype={"subject_id": str})
    train_priv = pd.read_csv(data_dir / "train_privileged.csv", dtype={"subject_id": str})

    train_ids = participants.loc[participants["split"] == "train", "subject_id"].astype(str).tolist()
    test_ids = participants.loc[participants["split"] == "test", "subject_id"].astype(str).tolist()

    # train_privileged.csv를 train 순서에 정렬 (test 참가자는 train_privileged.csv에 없음)
    priv_by_id = train_priv.set_index("subject_id").loc[train_ids].reset_index()
    y_train = priv_by_id["amyloid_pos"].to_numpy(dtype=int)

    images_dir = data_dir / "images"
    x_train = np.vstack([
        extract_static_image_features(images_dir / f"{sid}_copy.png", images_dir / f"{sid}_recall.png")
        for sid in train_ids
    ])
    x_test = np.vstack([
        extract_static_image_features(images_dir / f"{sid}_copy.png", images_dir / f"{sid}_recall.png")
        for sid in test_ids
    ])

    final_clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=0.5, max_iter=500, random_state=seed)),
    ])
    final_clf.fit(x_train, y_train)
    test_prob = np.clip(final_clf.predict_proba(x_test)[:, 1], 1e-6, 1.0 - 1e-6)

    device_used = "cpu"
    if requested_device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                device_used = "cuda"
        except Exception:
            device_used = "cpu"

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "subject_id": test_ids,
        "y_prob": np.round(test_prob, 6),
    }).to_csv(out_dir / "predictions.csv", index=False)

    summary = {
        "claim_signal": False,
        "method_summary": (
            "Static-image-only starter baseline using grayscale ink density, "
            "bounding-box area, and quadrant summary features from Copy and Recall PNGs "
            "with LogisticRegression."
        ),
        "device_used": device_used,
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data", help="입력 데이터 디렉터리")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "submissions" / "starter",
        help="출력 디렉터리 (predictions.csv, summary.json)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="연산 장치 (cpu 또는 cuda)")
    parser.add_argument("--seed", type=int, default=42, help="모델 난수 시드")
    args = parser.parse_args()

    summary = run_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        requested_device=args.device,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
