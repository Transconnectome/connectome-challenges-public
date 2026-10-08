"""EEG 합성에 쓰는 공통 정의: 채널, 기본 통계, 스펙트럼 모형."""
import numpy as np

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
            "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
FRONTAL = ["Fp1", "Fp2"]              # 눈 깜빡임 기준 채널
POSTERIOR = ["O1", "O2", "Pz"]        # 알파파 주파수 추정 채널
OLD_NAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
SFREQ = 250.0
BLINK_WINDOW = (-0.2, 0.6)            # 깜빡임 파형 구간 (초, 정점 기준)


def standard_name(name):
    """'FP1', 'Fp1.', 'EEG Fp1-REF', 'T3' 같은 이름을 10-20 표준 이름으로 맞춘다."""
    n = name.upper().replace("EEG", "").replace("-REF", "").replace("-LE", "")
    n = n.strip(" .-_")
    for std in CHANNELS + list(OLD_NAMES):
        if n == std.upper():
            return OLD_NAMES.get(std, std)
    return None


def psd_model(freqs, offset, exponent, alpha_power, alpha_freq, alpha_bw):
    """log10 PSD = offset - exponent*log10(f) + 알파 가우시안 (µV²/Hz). 1 Hz 아래는 평평하게."""
    f = np.maximum(freqs, 1.0)
    log_p = offset - exponent * np.log10(f)
    log_p = log_p + alpha_power * np.exp(-(freqs - alpha_freq) ** 2 / (2 * alpha_bw ** 2))
    return 10 ** log_p


def default_blink_template():
    t = np.arange(*[int(round(x * SFREQ)) for x in BLINK_WINDOW]) / SFREQ
    rise, fall = 0.12, 0.25
    w = np.where(t < 0, np.cos(np.clip(-t / rise, 0, 1) * np.pi / 2) ** 2,
                 np.cos(np.clip(t / fall, 0, 1) * np.pi / 2) ** 2)
    return w


def montage():
    """10-20 표준 전극 위치. MNE 1.13부터 standard_1020 이름이 colin27_1020으로 바뀜."""
    import mne
    names = mne.channels.get_builtin_montages()
    return mne.channels.make_standard_montage(
        "colin27_1020" if "colin27_1020" in names else "standard_1020")


def avg_ref_topography(topo):
    """평균 기준으로 바꾼 뒤 Fp1·Fp2 평균이 1이 되도록 맞춘 깜빡임 두피 분포."""
    t = np.asarray(topo, float)
    t = t - t.mean()
    return t / t[[CHANNELS.index(c) for c in FRONTAL]].mean()


def default_stats():
    """실제 데이터 없이 쓰는 기본값 (B 방식). 값은 문헌상 전형적인 범위에서 고른 예시다."""
    pos = montage().get_positions()["ch_pos"]
    xyz = np.array([pos[c] for c in CHANNELS])
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    correlation = np.exp(-dist / 0.06)          # 가까운 채널일수록 신호가 비슷함

    alpha = {"O1": 0.8, "O2": 0.8, "Pz": 0.6, "P3": 0.6, "P4": 0.6, "P7": 0.5,
             "P8": 0.5, "C3": 0.35, "Cz": 0.35, "C4": 0.35, "T7": 0.25, "T8": 0.25}
    topo = {"Fp1": 1.0, "Fp2": 1.0, "F7": 0.45, "F3": 0.55, "Fz": 0.6, "F4": 0.55,
            "F8": 0.45, "T7": 0.15, "C3": 0.2, "Cz": 0.22, "C4": 0.2, "T8": 0.15,
            "P7": 0.06, "P3": 0.08, "Pz": 0.09, "P4": 0.08, "P8": 0.06, "O1": 0.04, "O2": 0.04}
    return {
        "source": {"method": "default", "note": "실제 데이터에서 추출하지 않은 예시 값"},
        "channels": CHANNELS,
        "sfreq": SFREQ,
        "aperiodic": {"offset": [1.5] * len(CHANNELS), "exponent": [1.2] * len(CHANNELS)},
        "alpha": {"freq": 10.0, "bandwidth": 1.0,
                  "power": [alpha.get(c, 0.2) for c in CHANNELS]},
        "correlation": correlation.round(4).tolist(),
        "blink": {"rate_per_min": 15.0, "amplitude_uv": {"mean": 150.0, "sd": 40.0},
                  "window_s": list(BLINK_WINDOW),
                  "template": default_blink_template().round(4).tolist(),
                  "topography": avg_ref_topography([topo[c] for c in CHANNELS]).round(3).tolist()},
    }
