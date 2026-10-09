"""합성 EEG 생성기: 배경 뇌파 + 눈 깜빡임 + 좌우 눈 움직임(함정).

    python generate_synthetic_eeg.py                         # 기본값(예시 통계)으로 생성
    python generate_synthetic_eeg.py --seed 7 --out data_seed7


출력 (data/):
    eeg_raw.fif        19채널 EEG, 평균 기준 (MNE로 읽으면 V 단위)
    ground_truth.json  정답: 눈 깜빡임 시점, 좌우 눈 움직임 구간, 생성에 쓴 통계
"""
import argparse
import json
from pathlib import Path

import mne
import numpy as np

from eeg_synth import FRONTAL, default_stats, montage, psd_model

DURATION = 180.0                 # 초
SACCADE_RATE = 4.0               # 분당 좌우 눈 움직임 목표 횟수 (깜빡임과 겹치면 빼므로 실제는 더 적음)
SACCADE_UV = (60.0, 15.0)        # F7/F8 진폭 평균, SD (µV)
SACCADE_TOPO = {"F7": 1.0, "F8": -1.0, "Fp1": 0.35, "Fp2": -0.35, "T7": 0.3,
                "T8": -0.3, "F3": 0.15, "F4": -0.15}   # 왼쪽을 볼 때 부호


def background(stats, n, sfreq, rng):
    """채널별 목표 스펙트럼과 채널 간 상관을 갖는 배경 뇌파 (µV)."""
    n_ch = len(stats["channels"])
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    # 평균 기준 데이터의 상관 행렬은 특이 행렬에 가까우므로 고유값을 살짝 올려 분해
    w, v = np.linalg.eigh(np.array(stats["correlation"]))
    corr = (v * np.clip(w, 1e-4, None)) @ v.T
    chol = np.linalg.cholesky(corr)
    white = chol @ (rng.standard_normal((n_ch, freqs.size))
                    + 1j * rng.standard_normal((n_ch, freqs.size)))
    ap, al = stats["aperiodic"], stats["alpha"]
    spectra = np.array([psd_model(freqs, ap["offset"][i], ap["exponent"][i],
                                  al["power"][i], al["freq"], al["bandwidth"])
                        for i in range(n_ch)])
    coef = white * np.sqrt(spectra * sfreq * n / 4)
    coef[:, 0] = 0
    return np.fft.irfft(coef, n=n, axis=1)


def event_times(rate_per_min, duration, min_gap, rng, margin=1.0, avoid=()):
    """최소 간격을 지키는 무작위 사건 시점 (초)."""
    times, t = [], margin
    while True:
        t += min_gap + rng.exponential(max(60.0 / rate_per_min - min_gap, 0.1))
        if t > duration - margin:
            return np.array(times)
        if all(abs(t - a) > 1.0 for a in avoid):
            times.append(t)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats", type=Path, help="선택: 그룹 요약 통계 JSON. 연습 과제에는 필요 없고 공개본에는 통계 파일이 없습니다. 생략하면 기본값으로 생성")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()

    stats = json.loads(args.stats.read_text()) if args.stats else default_stats()
    rng = np.random.default_rng(args.seed)
    chs, sfreq = stats["channels"], float(stats["sfreq"])
    n = int(args.duration * sfreq)
    data = background(stats, n, sfreq, rng)

    # 눈 깜빡임: 앞이마(Fp1·Fp2)에서 가장 크고 뒤로 갈수록 작아지는 파형
    blink = stats["blink"]
    template = np.array(blink["template"])
    topo = np.array(blink["topography"])
    onset = int(round(-blink["window_s"][0] * sfreq))    # 파형 시작부터 정점까지
    blink_t = event_times(blink["rate_per_min"], args.duration, 1.0, rng)
    amps = np.clip(rng.normal(blink["amplitude_uv"]["mean"], blink["amplitude_uv"]["sd"],
                              blink_t.size), 0.3 * blink["amplitude_uv"]["mean"], None)
    for t, a in zip(blink_t, amps):
        start = int(round(t * sfreq)) - onset
        data[:, start:start + template.size] += np.outer(topo * a, template)

    # 좌우 눈 움직임: F7·F8에 부호가 반대인 계단 모양 (함정)
    sac_t = event_times(SACCADE_RATE, args.duration - 2, 2.0, rng, avoid=blink_t)
    sac_topo = np.array([SACCADE_TOPO.get(c, 0.0) for c in chs])
    ramp = int(0.03 * sfreq)
    saccades = []
    for t in sac_t:
        direction = rng.choice([-1, 1])            # +1: 왼쪽, -1: 오른쪽
        amp = max(rng.normal(*SACCADE_UV), 20.0) * direction
        hold = rng.uniform(0.5, 1.5)
        s, e = int(round(t * sfreq)), int(round((t + hold) * sfreq))
        shape = np.zeros(n)
        shape[s:e] = 1.0
        shape = np.convolve(shape, np.ones(ramp) / ramp, "same")
        data += np.outer(sac_topo * amp, shape)
        saccades.append({"onset_s": round(t, 3), "duration_s": round(hold, 3),
                         "direction": "left" if direction > 0 else "right",
                         "f7_amplitude_uv": round(float(amp), 1)})

    # 모든 채널의 평균을 기준 전극으로 씀
    data -= data.mean(axis=0, keepdims=True)

    # 저장
    args.out.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(chs, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)   # MNE는 V 단위
    raw.set_montage(montage())
    raw.save(args.out / "eeg_raw.fif", overwrite=True, verbose=False)

    frontal = np.array([chs.index(c) for c in FRONTAL])
    (args.out / "ground_truth.json").write_text(json.dumps({
        "note": "눈 깜빡임 시점은 Fp1·Fp2 파형의 정점 시각입니다.",
        "blinks": [{"peak_s": round(float(t), 3), "fp_amplitude_uv": round(float(a * topo[frontal].mean()), 1)}
                   for t, a in zip(blink_t, amps)],
        "saccades": saccades,
        "parameters": {"seed": args.seed, "duration_s": args.duration, "sfreq": sfreq,
                       "saccade_rate_per_min": SACCADE_RATE, "saccade_uv": SACCADE_UV},
        "stats_source": stats["source"],
    }, ensure_ascii=False, indent=2))
    print(f"저장 위치: {args.out.resolve()}")
    print(f"눈 깜빡임 {blink_t.size}회, 좌우 눈 움직임 {len(saccades)}회, "
          f"요약 통계: {'기본값' if stats['source']['method'] == 'default' else stats['source']['method']}")


if __name__ == "__main__":
    main()
