"""ROCF 특권 정보 활용 학습(Learning Using Privileged Information) 아밀로이드 예측 파이프라인 실행 규격 점검기.

    python smoke_test.py                                   # 현재 폴더의 run_pipeline.py (없으면 starter_pipeline.py)
    python smoke_test.py --pipeline run_pipeline.py --device cpu
    python smoke_test.py --pipeline run_pipeline.py --device cuda

점검 항목:
    (a) 파이프라인 폴더에 사전 학습 가중치·직렬화 파일(.pt .pth .safetensors .ckpt .onnx .bin .pkl .joblib)이 없는지
    (b) 제공된 합성 데이터(data/ 또는 data.zip)에서 중립 폴더명(eval_run_01)의 축소 코호트(N=100, train 70 / test 30,
        test는 ROCF 그림·과정 전용)를 구성해 오프라인(OPENROUTER_API_KEY="")으로 실행하고,
        predictions.csv(subject_id, y_prob) / summary.json(claim_signal, method_summary) 규격과
        N=400 환산 실행 시간(상한 600초, 노트북 CPU 120초 이내 권장)을 확인
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

BANNED_SUFFIXES = {".pt", ".pth", ".safetensors", ".ckpt", ".onnx", ".bin", ".pkl", ".joblib"}
IGNORED_DIR_NAMES = {"outputs", "submissions", ".git", "__pycache__", ".venv", ".ipynb_checkpoints"}
FULL_N = 400
RUNTIME_LIMIT_SEC = 600.0
CPU_RECOMMENDED_SEC = 120.0
SMOKE_N_TRAIN = 70
SMOKE_N_TEST = 30
SMOKE_N = SMOKE_N_TRAIN + SMOKE_N_TEST


class SmokeFailure(RuntimeError):
    """하드 실패(종료 코드 1)."""


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def resolve_pipeline(arg: str | None) -> Path:
    if arg:
        path = Path(arg).expanduser().resolve()
    elif (Path.cwd() / "run_pipeline.py").exists():
        path = (Path.cwd() / "run_pipeline.py").resolve()
    else:
        path = HERE / "starter_pipeline.py"
    if not path.is_file():
        raise SmokeFailure(f"파이프라인 파일을 찾을 수 없습니다: {path.name}")
    return path


def _is_ignored_dir(name: str) -> bool:
    return name in IGNORED_DIR_NAMES or name.startswith("data")


def find_weight_files(root: Path) -> list[Path]:
    """파이프라인 폴더 아래 금지된 가중치·직렬화 파일 목록 (데이터·출력 폴더 제외)."""
    hits: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _is_ignored_dir(d)]
        for fn in filenames:
            if Path(fn).suffix.lower() in BANNED_SUFFIXES:
                hits.append(Path(dirpath, fn).relative_to(root))
    return sorted(hits)


def _locate_source_data(data_arg: str | None, scratch_root: Path) -> Path:
    """제공된 데이터 폴더(data/) 또는 압축 파일(data.zip)을 찾아 입력 디렉터리를 반환한다."""
    candidates: list[Path] = []
    if data_arg:
        candidates.append(Path(data_arg).expanduser().resolve())
    else:
        candidates.extend([
            Path.cwd() / "data",
            HERE / "data",
            Path.cwd() / "data.zip",
            HERE / "data.zip",
            HERE.parents[1] / "public" / "challenges" / "rocf_vlm" / "data.zip",
        ])

    for cand in candidates:
        if cand.is_dir() and (cand / "participants.csv").is_file():
            return cand
        if cand.is_file() and cand.suffix.lower() == ".zip":
            unpack_dir = scratch_root / "unpacked_source"
            with zipfile.ZipFile(cand, "r") as zf:
                zf.extractall(unpack_dir)
            inner = unpack_dir / "data"
            if (inner / "participants.csv").is_file():
                return inner
            if (unpack_dir / "participants.csv").is_file():
                return unpack_dir
            raise SmokeFailure(f"압축 파일({cand.name}) 안에 participants.csv가 없습니다.")

    raise SmokeFailure(
        "합성 데이터를 찾을 수 없습니다. `python -m zipfile -e data.zip .`로 압축을 풀거나 "
        "`--data-dir <데이터 폴더 또는 data.zip>` 옵션을 지정하세요."
    )


def prepare_smoke_cohort(src_dir: Path, out_dir: Path, seed: int) -> int:
    """제공된 데이터에서 중립 폴더(out_dir)로 N=100(train 70 / test 30) 부분 코호트를 구성한다."""
    import numpy as np
    import pandas as pd

    for name in ("participants.csv", "train_privileged.csv", "images", "strokes"):
        if not (src_dir / name).exists():
            raise SmokeFailure(f"원본 데이터 폴더에 {name}이(가) 없습니다.")

    part = pd.read_csv(src_dir / "participants.csv", dtype={"subject_id": str})
    priv = pd.read_csv(src_dir / "train_privileged.csv", dtype={"subject_id": str})
    if list(part.columns) != ["subject_id", "split"]:
        raise SmokeFailure(f"participants.csv 규격 오류: {list(part.columns)}")

    train_pool = part.loc[part["split"] == "train", "subject_id"].astype(str).tolist()
    test_pool = part.loc[part["split"] == "test", "subject_id"].astype(str).tolist()
    if len(train_pool) < 10 or len(test_pool) < 4:
        raise SmokeFailure(f"표본 수가 너무 적습니다 (train={len(train_pool)}, test={len(test_pool)}).")

    rng = np.random.default_rng(seed)
    n_train = min(SMOKE_N_TRAIN, len(train_pool))
    n_test = min(SMOKE_N_TEST, len(test_pool))

    # train은 진단군×아밀로이드 층을 유지하며 추출
    priv_idx = priv.set_index("subject_id").loc[train_pool].reset_index()
    strata = priv_idx["diagnosis"].astype(str) + "_" + priv_idx["amyloid_pos"].astype(str)
    selected_train: list[str] = []
    if n_train < len(train_pool):
        for _, grp in priv_idx.groupby(strata):
            sids = grp["subject_id"].astype(str).tolist()
            k = max(1, int(round(n_train * len(sids) / len(train_pool))))
            k = min(k, len(sids))
            chosen = rng.choice(sids, size=k, replace=False).tolist()
            selected_train.extend(chosen)
        selected_train = sorted(set(selected_train))
        if len(selected_train) > n_train:
            selected_train = sorted(rng.choice(selected_train, size=n_train, replace=False).tolist())
        elif len(selected_train) < n_train:
            rem = [s for s in train_pool if s not in set(selected_train)]
            extra = rng.choice(rem, size=n_train - len(selected_train), replace=False).tolist()
            selected_train = sorted(selected_train + extra)
    else:
        selected_train = sorted(train_pool)

    selected_test = sorted(rng.choice(test_pool, size=n_test, replace=False).tolist())
    selected_set = set(selected_train) | set(selected_test)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "strokes").mkdir(parents=True, exist_ok=True)

    sub_part = part.loc[part["subject_id"].isin(selected_set)].reset_index(drop=True)
    sub_priv = priv.loc[priv["subject_id"].isin(set(selected_train))].reset_index(drop=True)
    sub_part.to_csv(out_dir / "participants.csv", index=False)
    sub_priv.to_csv(out_dir / "train_privileged.csv", index=False)

    for sid in sub_part["subject_id"].astype(str):
        for trial in ("copy", "recall"):
            shutil.copyfile(src_dir / "images" / f"{sid}_{trial}.png", out_dir / "images" / f"{sid}_{trial}.png")
            shutil.copyfile(src_dir / "strokes" / f"{sid}_{trial}.csv", out_dir / "strokes" / f"{sid}_{trial}.csv")

    return int(len(sub_part))


def run_pipeline(pipeline: Path, data_dir: Path, out_dir: Path, device: str, seed: int, timeout: float) -> float:
    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = ""  # 외부 API 없이 오프라인으로 동작해야 함
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(HERE), env.get("PYTHONPATH", "")) if p)
    cmd = [
        sys.executable,
        str(pipeline),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    t0 = time.perf_counter()
    try:
        res = subprocess.run(cmd, cwd=str(pipeline.parent), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SmokeFailure(f"파이프라인이 {timeout:.0f}초 안에 끝나지 않았습니다.") from exc
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        tail = (res.stderr or res.stdout)[-3000:]
        raise SmokeFailure(f"파이프라인 종료 코드 {res.returncode}:\n{tail}")
    return elapsed


def check_outputs(data_dir: Path, out_dir: Path) -> dict:
    import numpy as np
    import pandas as pd

    part = pd.read_csv(data_dir / "participants.csv", dtype={"subject_id": str})
    if list(part.columns) != ["subject_id", "split"]:
        raise SmokeFailure(f"participants.csv 규격 오류: {list(part.columns)}")
    test_ids = set(part.loc[part["split"] == "test", "subject_id"].astype(str))

    pred_path = out_dir / "predictions.csv"
    if not pred_path.exists():
        raise SmokeFailure("predictions.csv가 없습니다.")
    pred = pd.read_csv(pred_path, dtype={"subject_id": str})
    if list(pred.columns) != ["subject_id", "y_prob"]:
        raise SmokeFailure(f"predictions.csv 열은 subject_id,y_prob 이어야 합니다 (현재: {list(pred.columns)}).")
    if pred["subject_id"].duplicated().any():
        raise SmokeFailure("predictions.csv에 중복 subject_id가 있습니다.")
    got = set(pred["subject_id"].astype(str))
    if got != test_ids:
        raise SmokeFailure(f"test 참가자 불일치: 누락 {len(test_ids - got)}명, 불필요 {len(got - test_ids)}명.")
    y_prob = pd.to_numeric(pred["y_prob"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(y_prob)):
        raise SmokeFailure("y_prob에 NaN 또는 숫자가 아닌 값이 있습니다.")
    if (y_prob < 0).any() or (y_prob > 1).any():
        raise SmokeFailure("y_prob는 0 이상 1 이하여야 합니다.")

    sum_path = out_dir / "summary.json"
    if not sum_path.exists():
        raise SmokeFailure("summary.json이 없습니다.")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise SmokeFailure("summary.json은 JSON 객체(dict)여야 합니다.")
    if "claim_signal" not in summary or not isinstance(summary["claim_signal"], bool):
        raise SmokeFailure("summary.json의 claim_signal은 true/false(bool)여야 합니다.")
    method_summary = summary.get("method_summary")
    if not isinstance(method_summary, str) or not method_summary.strip():
        raise SmokeFailure("summary.json의 method_summary는 비어 있지 않은 문자열(str)이어야 합니다.")
    _ok(
        f"predictions.csv {len(pred)}행 (test 전원, NaN 없음, 0–1 범위) / "
        "summary.json 필수 키(claim_signal, method_summary) 확인"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pipeline", help="점검할 파이프라인 파일 (기본: 현재 폴더 run_pipeline.py, 없으면 starter_pipeline.py)")
    parser.add_argument("--data-dir", help="입력 합성 데이터 폴더 또는 data.zip 경로 (기본: data/ 또는 data.zip 자동 탐색)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="파이프라인에 전달할 연산 장치")
    parser.add_argument("--seed", type=int, default=123, help="점검용 축소 코호트 추출 및 실행 시드")
    parser.add_argument("--keep", action="store_true", help="임시 폴더를 지우지 않고 남김 (디버깅용)")
    args = parser.parse_args()

    tmp_root = Path(tempfile.mkdtemp(prefix="rocf_check_"))
    try:
        pipeline = resolve_pipeline(args.pipeline)
        print(f"점검 대상: {pipeline.name} (device={args.device})")

        # (a) 사전 학습 가중치 반입 금지
        hits = find_weight_files(pipeline.parent)
        if hits:
            raise SmokeFailure("금지된 가중치·직렬화 파일이 있습니다: " + ", ".join(str(h) for h in hits[:10]))
        _ok("가중치·직렬화 파일 없음")

        if args.device == "cuda":
            _warn(
                "GPU 메모리는 별도 프로세스라 여기서 직접 잴 수 없습니다. "
                "다른 터미널에서 `nvidia-smi --query-gpu=memory.used --format=csv -l 1`로 "
                "최대 사용량이 16 GB 이하인지 확인하세요."
            )

        # (b) 중립 폴더명(eval_run_01) 축소 코호트 실행 + 시간 환산
        src_data = _locate_source_data(args.data_dir, tmp_root)
        data_1 = tmp_root / "eval_run_01" / "input"
        out_1 = tmp_root / "eval_run_01" / "output"
        n_used = prepare_smoke_cohort(src_data, data_1, args.seed)
        _ok(f"점검용 축소 코호트 N={n_used} 구성 (test ROCF 전용)")
        elapsed = run_pipeline(pipeline, data_1, out_1, args.device, args.seed, timeout=RUNTIME_LIMIT_SEC)
        summary = check_outputs(data_1, out_1)
        projected = elapsed * FULL_N / max(1, n_used)
        print(
            f"실행 시간 {elapsed:.1f}초 → N={FULL_N} 환산 약 {projected:.1f}초 (상한 {RUNTIME_LIMIT_SEC:.0f}초) | "
            f"claim_signal={summary['claim_signal']}"
        )
        if projected >= RUNTIME_LIMIT_SEC:
            raise SmokeFailure(f"N={FULL_N} 환산 실행 시간이 {RUNTIME_LIMIT_SEC:.0f}초를 넘습니다.")
        if args.device == "cpu" and projected > CPU_RECOMMENDED_SEC:
            _warn(
                f"CPU 환산 시간({projected:.0f}초)이 면접 권장 기준({CPU_RECOMMENDED_SEC:.0f}초)을 넘습니다. "
                "면접장 즉석 재실행을 위해 경량화를 권장합니다."
            )
        _ok("실행 시간 상한 통과")
    except SmokeFailure as exc:
        print(f"[FAIL] {exc}")
        return 1
    finally:
        if args.keep:
            print(f"임시 폴더 유지: {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    print("[PASS] smoke_test 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
