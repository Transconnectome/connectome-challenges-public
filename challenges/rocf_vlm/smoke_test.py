"""ROCF 특권 정보 활용 학습(Learning Using Privileged Information) 아밀로이드 예측 파이프라인 단일 실행 규격 점검기.

    python smoke_test.py                                   # 현재 폴더의 run_pipeline.py (없으면 starter_pipeline.py)
    python smoke_test.py --pipeline run_pipeline.py --device cpu
    python smoke_test.py --pipeline run_pipeline.py --device cuda

점검 항목:
    (a) 파이프라인 폴더에 사전 학습 가중치·직렬화 파일(.pt .pth .safetensors .ckpt .onnx .bin .pkl .joblib)이 없는지
    (b) 중립 폴더명(eval_run_01)의 합성 코호트(N=100, test는 ROCF 그림·과정 전용, ground_truth.json 삭제)에서
        오프라인(OPENROUTER_API_KEY="")으로 실행해 predictions.csv / summary.json 규격과
        N=400 환산 실행 시간(상한 600초, 노트북 CPU 120초 이내 권장)을 확인
    (c) 중립 폴더명(eval_run_02)의 영효과(--null-effect, N=120) 데이터에서 규격을 확인하고 거짓 양성 주장 시 경고
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generate_synthetic_rocf.py"

BANNED_SUFFIXES = {".pt", ".pth", ".safetensors", ".ckpt", ".onnx", ".bin", ".pkl", ".joblib"}
IGNORED_DIR_NAMES = {"outputs", "submissions", ".git", "__pycache__", ".venv", ".ipynb_checkpoints"}
REQUIRED_KEYS = (
    "cv_auroc_pooled",
    "cv_auroc_cu",
    "cv_auroc_mci",
    "cv_within_dx_auroc",
    "cv_within_dx_auroc_ci95",
    "cv_shuffle_within_dx_auroc",
    "claim_true_amyloid_signal",
)
FULL_N = 400
RUNTIME_LIMIT_SEC = 600.0
CPU_RECOMMENDED_SEC = 120.0
SMOKE_N = 100
NULL_N = 120


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


def generate_cohort(out_dir: Path, seed: int, n_subjects: int, null_effect: bool) -> None:
    cmd = [sys.executable, str(GENERATOR), "--seed", str(seed), "--n-subjects", str(n_subjects), "--out", str(out_dir)]
    if null_effect:
        cmd.append("--null-effect")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise SmokeFailure(f"합성 데이터 생성 실패:\n{res.stderr[-2000:]}")
    gt = out_dir / "ground_truth.json"
    if gt.exists():
        gt.unlink()  # 파이프라인이 정답 파일 없이도 돌아가는지 확인
    for name in ("participants.csv", "train_privileged.csv", "images", "strokes"):
        if not (out_dir / name).exists():
            raise SmokeFailure(f"생성된 데이터에 {name}이(가) 없습니다.")


def run_pipeline(pipeline: Path, data_dir: Path, out_dir: Path, device: str, timeout: float) -> float:
    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = ""  # 외부 API 없이 오프라인으로 동작해야 함
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(HERE), env.get("PYTHONPATH", "")) if p)
    cmd = [sys.executable, str(pipeline), "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--device", device]
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
    missing = [k for k in REQUIRED_KEYS if k not in summary]
    if missing:
        raise SmokeFailure(f"summary.json 필수 키 누락: {missing}")
    if not isinstance(summary["claim_true_amyloid_signal"], bool):
        raise SmokeFailure("claim_true_amyloid_signal은 true/false(bool)여야 합니다.")
    ci = summary["cv_within_dx_auroc_ci95"]
    if not (isinstance(ci, list) and len(ci) == 2 and all(isinstance(v, (int, float)) for v in ci)):
        raise SmokeFailure("cv_within_dx_auroc_ci95는 [하한, 상한] 숫자 2개 목록이어야 합니다.")
    for key in REQUIRED_KEYS[:6]:
        if key == "cv_within_dx_auroc_ci95":
            continue
        val = summary[key]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise SmokeFailure(f"{key}는 숫자여야 합니다.")
        if math.isnan(float(val)) and key not in ("cv_auroc_cu", "cv_auroc_mci"):
            raise SmokeFailure(f"{key}가 NaN입니다.")
    _ok(f"predictions.csv {len(pred)}행 (test 전원, NaN 없음, 0–1 범위) / summary.json 필수 키 7개")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pipeline", help="점검할 파이프라인 파일 (기본: 현재 폴더 run_pipeline.py, 없으면 starter_pipeline.py)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="파이프라인에 전달할 연산 장치")
    parser.add_argument("--seed", type=int, default=123, help="점검용 합성 데이터 시드")
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

        # (b) 중립 폴더명(eval_run_01) 실행 + 시간 환산
        data_1 = tmp_root / "eval_run_01" / "input"
        out_1 = tmp_root / "eval_run_01" / "output"
        generate_cohort(data_1, args.seed, SMOKE_N, null_effect=False)
        _ok(f"합성 코호트 N={SMOKE_N} 생성 (test ROCF 전용, ground_truth.json 삭제)")
        elapsed = run_pipeline(pipeline, data_1, out_1, args.device, timeout=RUNTIME_LIMIT_SEC)
        summary = check_outputs(data_1, out_1)
        projected = elapsed * FULL_N / SMOKE_N
        print(
            f"실행 시간 {elapsed:.1f}초 → N={FULL_N} 환산 약 {projected:.1f}초 (상한 {RUNTIME_LIMIT_SEC:.0f}초) | "
            f"전체 AUROC {summary['cv_auroc_pooled']}, 진단군 내 AUROC {summary['cv_within_dx_auroc']}, "
            f"claim {summary['claim_true_amyloid_signal']}"
        )
        if projected >= RUNTIME_LIMIT_SEC:
            raise SmokeFailure(f"N={FULL_N} 환산 실행 시간이 {RUNTIME_LIMIT_SEC:.0f}초를 넘습니다.")
        if args.device == "cpu" and projected > CPU_RECOMMENDED_SEC:
            _warn(
                f"CPU 환산 시간({projected:.0f}초)이 면접 권장 기준({CPU_RECOMMENDED_SEC:.0f}초)을 넘습니다. "
                "면접장 즉석 재실행을 위해 경량화를 권장합니다."
            )
        _ok("실행 시간 상한 통과")

        # (c) 중립 폴더명(eval_run_02) 영효과 검증: 경로명 힌트 없음
        data_2 = tmp_root / "eval_run_02" / "input"
        out_2 = tmp_root / "eval_run_02" / "output"
        generate_cohort(data_2, args.seed + 1, NULL_N, null_effect=True)
        run_pipeline(pipeline, data_2, out_2, args.device, timeout=RUNTIME_LIMIT_SEC)
        null_summary = check_outputs(data_2, out_2)
        if null_summary["claim_true_amyloid_signal"]:
            _warn(
                f"영효과(N={NULL_N}) 데이터에서 claim_true_amyloid_signal=true 입니다 "
                f"(진단군 내 AUROC {null_summary['cv_within_dx_auroc']}, CI {null_summary['cv_within_dx_auroc_ci95']}). "
                "판정 규칙이 거짓 양성에 취약할 수 있습니다."
            )
        else:
            _ok(f"영효과 데이터에서 신호를 주장하지 않음 (진단군 내 AUROC {null_summary['cv_within_dx_auroc']})")
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
