"""합성 fMRI 데이터 생성기 (BrainIAK fmrisim + 머리 움직임 아티팩트).

실제 데이터를 공유할 수 없을 때 쓰는 연습용 데이터입니다. 영역 사이의 신호 구조를
정답으로 알고 있으므로, 분석 결과를 정답과 비교할 수 있습니다.

    python generate_synthetic_fmri.py                          # 기본값: A·B 연결, C 독립
    python generate_synthetic_fmri.py --seed 7                 # 다른 참가자용 변형
    python generate_synthetic_fmri.py --stats fmri_stats.json  # 실제 데이터의 그룹 통계로 생성
    python generate_synthetic_fmri.py --voxel-mm 3 --n-volumes 400   # 더 큰 데이터

fmri_stats.json은 extract_fmri_stats.py로 만든 그룹 요약 통계입니다.

출력 (data/):
    bold.nii.gz        4D fMRI 볼륨 (x, y, z, 시간)
    rois.nii.gz        영역 라벨 (0=배경, 1, 2, 3, ...)
    confounds.tsv      시점별 머리 움직임 크기 (framewise_displacement, mm)
    ground_truth.json  정답: 잡음·움직임을 넣기 전 영역 간 상관과 생성 파라미터
"""
import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from brainiak.utils import fmrisim as sim
from scipy import ndimage
from scipy.signal import lfilter

BRAIN_MM = np.array([192, 228, 192])   # 볼륨 범위. 표준 공간(MNI)에 정렬되지 않음
VOXEL_MM = 6.0
TR = 2.0
N_TR = 200                      # 400초 스캔
RES = 10.0                      # 신경 신호 시간 해상도 (Hz)
ROI_RADIUS_MM = 12.0            # 영역(구) 반지름
ROI_MIN_GAP_MM = 60.0           # 영역 중심 사이 최소 거리
CNR = 1.5                       # 복셀별 신호 SD / 잡음 SD (기본 모드)
CNR_STATS = 3.0                 # 통계 모드: 실제 atlas 영역처럼 영역 평균 잡음이 작도록
COUPLING = 0.8                  # 기본값에서 A·B 고유 성분 비율 (작을수록 A-B 상관이 큼)
SPIKES_PER_200 = 10             # 기본값: 200 시점당 큰 움직임 수
MOTION_GAIN = 4.0               # 기본값: 움직임 1 mm당 전체 뇌 신호 변화 (%)
SPIKE_FD = 0.5                  # 큰 움직임 기준 (mm)
NOISE = {"snr": 50, "sfnr": 70, "fwhm": 4, "auto_reg_rho": [0.5],
         "physiological_sigma": 1, "drift_sigma": 1, "task_sigma": 0,
         "auto_reg_sigma": 1, "matched": 1}


def slow_signal(rng, n, cutoff_hz=0.05):
    """저주파 자발 신경 변동 (1차 저역통과 잡음), 표준화."""
    a = np.exp(-2 * np.pi * cutoff_hz / RES)
    x = lfilter([1 - a], [1, -a], rng.standard_normal(n))
    return x / x.std()


def ball(radius):
    r = int(np.floor(radius))
    g = np.indices((2 * r + 1,) * 3) - r
    return np.sqrt((g ** 2).sum(0)) <= radius


def sphere(dims, center, radius):
    g = np.indices(dims)
    return np.sqrt(sum((g[i] - center[i]) ** 2 for i in range(3))) <= radius


def sphere_scores(mask, template, radius):
    """각 위치에 구를 놓았을 때 회색질 템플릿 평균. 구가 뇌 밖으로 나가면 -1."""
    b = ball(radius)
    mean_t = ndimage.correlate(template, b.astype(float), mode="constant") / b.sum()
    inside = ndimage.binary_erosion(mask > 0, structure=b)
    return np.where(inside, mean_t, -1.0)


def choose_default_rois(score, voxel, radius):
    """A와 B는 x축 대칭 위치, C는 가운데 선 근처에서 A·B와 떨어진 위치."""
    dims = np.array(score.shape)
    r, mid = int(radius), dims[0] - 1
    pair = np.minimum(score, score[::-1])
    allowed = np.zeros(dims, bool)
    zc, dz = dims[2] // 2, round(24 / voxel)
    allowed[r:mid // 2 - round(12 / voxel), r:dims[1] - r, zc - dz:zc + dz + 1] = True
    a = np.array(np.unravel_index(np.argmax(np.where(allowed, pair, -2)), dims))
    b = np.array([mid - a[0], a[1], a[2]])

    k = max(1, round(6 / voxel))
    allowed = np.zeros(dims, bool)
    allowed[mid // 2 - k:mid // 2 + k + 1, r:dims[1] - r, r:dims[2] - r] = True
    g = np.indices(dims)
    far = np.ones(dims, bool)
    for c in (a, b):
        far &= np.sqrt(sum((g[i] - c[i]) ** 2 for i in range(3))) >= ROI_MIN_GAP_MM / voxel
    c = np.array(np.unravel_index(np.argmax(np.where(allowed & far, score, -2)), dims))
    return [a, b, c]


def choose_rois(score, voxel, n):
    """템플릿 점수가 높은 곳부터 서로 떨어진 n개 위치를 고른다."""
    dims = score.shape
    g = np.indices(dims)
    gap = ROI_MIN_GAP_MM
    while gap > 0:
        s, centers = score.copy(), []
        for _ in range(n):
            if s.max() < 0:
                break
            c = np.array(np.unravel_index(np.argmax(s), dims))
            centers.append(c)
            s[np.sqrt(sum((g[i] - c[i]) ** 2 for i in range(3))) < gap / voxel] = -1
        if len(centers) == n:
            return centers
        gap -= 10
    raise SystemExit(f"영역 {n}개를 서로 떨어뜨려 놓을 수 없습니다.")


def nearest_psd(corr):
    w, v = np.linalg.eigh(corr)
    c = (v * np.clip(w, 1e-4, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats", type=Path, help="extract_fmri_stats.py가 만든 JSON (연구원용. 생략하면 기본값으로 생성)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voxel-mm", type=float, default=VOXEL_MM,
                        help="복셀 크기. 3이면 64×76×64 (데이터 약 8배)")
    parser.add_argument("--n-volumes", type=int, help="시점 수 (기본: 200 또는 통계의 값)")
    parser.add_argument("--cnr", type=float,
                        help=f"복셀별 신호 SD / 잡음 SD (기본 {CNR}, --stats 사용 시 {CNR_STATS})")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stats = json.loads(args.stats.read_text()) if args.stats else None
    cnr = args.cnr or (CNR_STATS if stats else CNR)
    voxel = args.voxel_mm
    dims = np.round(BRAIN_MM / voxel).astype(int)
    radius = ROI_RADIUS_MM / voxel
    tr = float(stats["tr_s"]) if stats else TR
    n_tr = args.n_volumes or (int(stats["n_volumes"]) if stats else N_TR)
    noise_dict = dict(NOISE, voxel_size=[voxel] * 3)
    if stats:
        n = stats["noise"]
        noise_dict.update(sfnr=n["sfnr"], fwhm=n["fwhm_mm"], auto_reg_rho=n["auto_reg_rho"],
                          ma_rho=n["ma_rho"], snr=n["snr"] or NOISE["snr"])

    start = time.time()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # fmrisim은 numpy 전역 난수를 사용

    mask, template = sim.mask_brain(np.zeros(dims), mask_self=False)
    score = sphere_scores(mask, template, radius)
    if stats:
        names = stats["rois"]["names"]
        target = nearest_psd(np.array(stats["rois"]["correlation"]))
        centers = choose_rois(score, voxel, len(names))
    else:
        names, target = ["A", "B", "C"], None
        centers = choose_default_rois(score, voxel, radius)
    rois = [sphere(dims, c, radius) for c in centers]

    # 1) 정답 신경 신호 -> 혈류 반응(HRF) -> TR 간격 BOLD
    n_hi = int(round(n_tr * tr * RES))
    if stats:   # 실제 데이터의 그룹 평균 상관을 갖도록 독립 신호를 섞음
        indep = np.column_stack([slow_signal(rng, n_hi) for _ in names])
        neural = indep @ np.linalg.cholesky(target).T
    else:
        shared = slow_signal(rng, n_hi)
        neural = np.column_stack([shared + COUPLING * slow_signal(rng, n_hi),
                                  shared + COUPLING * slow_signal(rng, n_hi),
                                  slow_signal(rng, n_hi)])
    bold = sim.convolve_hrf(neural, tr_duration=tr, temporal_resolution=RES)[:n_tr]
    truth = np.corrcoef(bold.T)

    # 2) 스캐너·생리 잡음 (matched=1: 요청한 잡음 특성에 맞춰 반복 조정)
    noise = sim.generate_noise(dimensions=dims, stimfunction_tr=np.zeros((n_tr, 1)),
                               tr_duration=tr, template=template, mask=mask,
                               noise_dict=noise_dict)

    # 3) 영역별 신호를 잡음 크기에 맞춰 삽입
    data = noise
    for k, roi in enumerate(rois):
        sig = np.tile(bold[:, [k]], (1, int(roi.sum())))
        sig = sim.compute_signal_change(sig, noise[roi].T, noise_dict, [cnr],
                                        "CNR_Signal-SD/Noise-SD")
        data[roi] += sig.T

    # 4) 머리 움직임: 큰 움직임 시점에 뇌 전체 신호가 함께 흔들림 -> 가짜 상관
    if stats:
        m = stats["motion"]
        fd = np.abs(rng.normal(m["fd_baseline_mm"]["mean"], m["fd_baseline_mm"]["sd"], n_tr))
        n_spikes = int(round(m["spike_fraction"] * n_tr))
        gain = m["gain_pct_per_mm"]
    else:
        fd = np.abs(rng.normal(0.12, 0.05, n_tr))
        n_spikes = int(round(SPIKES_PER_200 * n_tr / 200))
        gain = MOTION_GAIN
    spikes = np.sort(rng.choice(np.arange(5, n_tr - 5), n_spikes, replace=False))
    if stats and n_spikes:   # 기준값 위로 치우친 분포(lognormal)로 평균·SD를 맞춤
        excess = max(m["spike_fd_mm"]["mean"] - SPIKE_FD, 0.05)
        sigma2 = np.log(1 + (m["spike_fd_mm"]["sd"] or 0.0) ** 2 / excess ** 2)
        fd[spikes] = SPIKE_FD + rng.lognormal(np.log(excess) - sigma2 / 2, np.sqrt(sigma2), n_spikes)
    else:
        fd[spikes] += rng.uniform(0.6, 1.5, n_spikes)
    artifact = np.zeros(n_tr)
    for s in spikes:  # 움직임 직후 한 장까지 영향이 남음
        size = (fd[s] * gain / 100) * rng.choice([-1, 1])
        artifact[s] += size
        artifact[s + 1] += 0.5 * size
    brain = mask > 0
    data[brain] += data[brain].mean(axis=1, keepdims=True) * artifact

    # 5) 저장
    affine = np.diag([voxel] * 3 + [1.0])
    header = nib.Nifti1Header()
    header.set_xyzt_units("mm", "sec")
    header["pixdim"][4] = tr
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine, header),
             args.out / "bold.nii.gz")
    labels = np.zeros(dims, dtype=np.int16)
    for k, roi in enumerate(rois, start=1):
        labels[roi] = k
    nib.save(nib.Nifti1Image(labels, affine), args.out / "rois.nii.gz")
    pd.DataFrame({"framewise_displacement": fd.round(4)}).to_csv(
        args.out / "confounds.tsv", sep="\t", index=False)

    pairs = lambda c: {f"{names[i]}-{names[j]}": round(float(c[i, j]), 3)
                       for i in range(len(names)) for j in range(i + 1, len(names))}
    result = {
        "note": "잡음과 움직임을 넣기 전, 혈류 반응까지 반영한 영역 간 상관계수입니다.",
        "true_correlation": pairs(truth),
        "structure": ("영역 간 상관은 실제 데이터의 그룹 평균 상관을 목표로 만들었습니다."
                      if stats else "A와 B는 공통 신경 신호를 공유하고, C는 독립입니다."),
        "roi_labels": {n: k for k, n in enumerate(names, start=1)},
        "motion_spike_volumes": spikes.tolist(),
        "roi_centers_voxel": {n: c.tolist() for n, c in zip(names, centers)},
        "parameters": {"seed": args.seed, "tr_s": tr, "n_volumes": n_tr, "voxel_mm": voxel,
                       "dims": dims.tolist(), "cnr": cnr, "motion_gain_pct_per_mm": gain,
                       "noise": noise_dict},
        "stats_source": stats["source"] if stats else {"method": "default"},
    }
    if stats:
        result["target_correlation"] = pairs(target)
    (args.out / "ground_truth.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    size_gb = data.size * 4 / 1e9
    print(f"저장 위치: {args.out.resolve()} ({time.time() - start:.0f}초, "
          f"{'x'.join(map(str, dims))}×{n_tr}, float32 {size_gb:.2f} GB)")
    print("정답 상관계수:", pairs(truth))


if __name__ == "__main__":
    main()
