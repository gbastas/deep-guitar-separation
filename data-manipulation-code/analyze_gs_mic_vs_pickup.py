#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, glob, re, math
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
EPS = 1e-12



def compute_active_mask(y, frame_length=2048, hop=1024, thresh_db=-50.0):
    """Boolean mask of frames where short-time RMS > threshold (dBFS)."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop, center=True, pad_mode="reflect")[0]
    rms_db = librosa.amplitude_to_db(rms + EPS, ref=1.0)
    return rms_db > thresh_db  # shape (T,)

def level_match_to_reference(y_target, y_ref, mask_frames, n_fft=4096, hop=1024):
    """
    Scale y_target so that its RMS (over active frames) matches y_ref.
    We compute RMS in time domain over samples that fall in active frames.
    """
    # map frame mask to sample mask
    frame_centers = librosa.frames_to_samples(np.arange(len(mask_frames)), hop_length=hop)
    # create a per-sample mask by marking samples within each active frame window
    sample_mask = np.zeros_like(y_ref, dtype=bool)
    half = n_fft // 2
    for c, active in zip(frame_centers, mask_frames):
        if not active: 
            continue
        i0 = max(0, c - half)
        i1 = min(len(sample_mask), c + half)
        sample_mask[i0:i1] = True
    # fall back if mask is empty
    if not np.any(sample_mask):
        sample_mask[:] = True

    def rms(sig):
        x = sig[sample_mask]
        return float(np.sqrt(np.mean(np.square(x) + EPS)))

    rms_ref = rms(y_ref)
    rms_tgt = rms(y_target)
    gain = 1.0 if rms_tgt == 0 else (rms_ref / rms_tgt)
    return y_target * gain, gain

def stft_logmag(y, n_fft=4096, hop=1024):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    M = np.log10(np.abs(S) + EPS) * 20.0  # dB
    freqs = librosa.fft_frequencies(sr=librosa.get_samplerate(None) or 44100, n_fft=n_fft)
    return freqs, M  # shape: (F, T)


def find_pairs(mic_dir, mix_dir):
    """Match *_mic.wav to *_mix.wav by common stem."""
    mic_files = sorted(glob.glob(os.path.join(mic_dir, "*_mic.wav")))
    pairs = []
    for mic_path in mic_files:
        stem = os.path.basename(mic_path).replace("_mic.wav", "")
        mix_path = os.path.join(mix_dir, stem + "_mix.wav")
        if os.path.exists(mix_path):
            pairs.append((stem, mic_path, mix_path))
    return pairs

def label_subset(stem):
    # crude but effective
    if "_solo" in stem:
        return "solo"
    if "_comp" in stem:
        return "comp"
    return "unknown"

def estimate_lag_samples(x, y, sr, max_ms=50):
    """Estimate best circular-free lag (in samples) within ±max_ms via cross-correlation."""
    max_lag = int(sr * max_ms / 1000.0)
    # downsample a bit for speed & robustness
    dec = max(1, sr // 8000)
    x_d, y_d = x[::dec], y[::dec]
    max_lag_d = max(1, max_lag // dec)
    # zero-mean to focus on shape
    x_d = x_d - np.mean(x_d); y_d = y_d - np.mean(y_d)
    # correlate
    corr = np.correlate(y_d, x_d, mode="full")  # align x to y
    lags = np.arange(-len(x_d)+1, len(y_d))
    # restrict
    mask = (lags >= -max_lag_d) & (lags <= max_lag_d)
    lag_d = lags[mask][np.argmax(corr[mask])]
    return int(lag_d * dec)  # samples (x shifted by this to match y)

def align_on_lag(x, y, lag):
    """Return time-aligned (crop to common length). Shift x by lag (samples) to match y."""
    if lag > 0:
        x = x[lag:]
    elif lag < 0:
        y = y[-lag:]
    L = min(len(x), len(y))
    return x[:L], y[:L]

def mean_log_power_spectrum(x, sr, n_fft=4096, hop=1024, fmin=40, fmax=10000):
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    psd = np.mean(np.abs(S)**2, axis=1)  # average across time
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= fmin) & (freqs <= fmax)
    L = 10.0 * np.log10(psd + EPS)
    return freqs[band], L[band]

def band_mean(freqs, vals_db, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    if not np.any(m): return np.nan
    return float(np.mean(vals_db[m]))

def fit_tilt_db_per_decade(freqs, delta_db):
    """Fit Δ(f) ~ a*log10(f) + b over valid band, return slope a (dB/decade)."""
    x = np.log10(np.clip(freqs, 1, None))
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, delta_db, rcond=None)[0]
    return float(a)

def process_pair(mic_path, mix_path, sr_target=44100):
    # load mono
    y_mic, sr_mic = librosa.load(mic_path, sr=sr_target, mono=True)
    y_mix, sr_mix = librosa.load(mix_path, sr=sr_target, mono=True)

    # align time by coarse xcorr
    lag = estimate_lag_samples(y_mix, y_mic, sr_target, max_ms=50)  # shift mix to mic
    y_mix_a, y_mic_a = align_on_lag(y_mix, y_mic, lag)

    # same length?
    L = min(len(y_mix_a), len(y_mic_a))
    y_mix_a = y_mix_a[:L]; y_mic_a = y_mic_a[:L]

    # spectra
    f, L_mic = mean_log_power_spectrum(y_mic_a, sr_target)
    _, L_mix = mean_log_power_spectrum(y_mix_a, sr_target)

    # ensure same length (n_fft identical so should match)
    m = np.isfinite(L_mic) & np.isfinite(L_mix)
    f, L_mic, L_mix = f[m], L_mic[m], L_mix[m]

    # Δ(f) pre-normalization
    delta_raw = L_mic - L_mix

    # remove overall gain: subtract median Δ over a mid band (300–3000 Hz)
    mid = (f >= 300) & (f <= 3000)
    offset = np.median(delta_raw[mid]) if np.any(mid) else np.median(delta_raw)
    delta = delta_raw - offset

    # metrics
    low = band_mean(f, delta, 80, 200)
    lowmid = band_mean(f, delta, 200, 800)
    midb = band_mean(f, delta, 800, 3000)
    high = band_mean(f, delta, 3000, 8000)
    slope = fit_tilt_db_per_decade(f[(f>=80)&(f<=8000)], delta[(f>=80)&(f<=8000)])
    rms_delta = float(np.sqrt(np.mean(delta**2)))
    lag_ms = 1000.0 * lag / sr_target

    return {
        "low_80_200_db": round(low, 3),
        "lowmid_200_800_db": round(lowmid, 3),
        "mid_0p8_3k_db": round(midb,3),
        "high_3_8k_db": round(high,3),
        "tilt_db_per_decade": round(slope,3),
        "rms_delta_db": round(rms_delta,3),
        "lag_ms": round(lag_ms,3),
        "f": f, 
        "delta_db": delta,
    }

    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic_dir", required=True, help="e.g. datasets/GuitarSet/data/mic")
    ap.add_argument("--mix_dir", required=True, help="e.g. datasets/GuitarSet/data/mix")
    ap.add_argument("--out_dir", default="gs_spectral_report", help="output folder")
    ap.add_argument("--sr", type=int, default=44100)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("ready to gather pairs of mixture.")
    pairs = find_pairs(args.mic_dir, args.mix_dir)
    if not pairs:
        print("No pairs found. Check paths and filenames.")
        return

    rows = []
    deltas_all = []
    # for stem, mic_path, mix_path in pairs:
    count=0
    for stem, mic_path, mix_path in tqdm(pairs, total=len(pairs), desc="Processing pairs"):        
        subset = label_subset(stem)
        print('subset', subset)
        print('stem', stem)
        print('mic_path', mic_path)
        res = process_pair(mic_path, mix_path, sr_target=args.sr)
        rows.append({
            "track": stem,
            "subset": subset,
            **{k:v for k,v in res.items() if k not in ("f","delta_db")}
        })
        deltas_all.append((stem, res["f"], res["delta_db"], subset))
        count+=1
        # if count == 2:
        #     break

    # df = pd.DataFrame(rows).sort_values(["subset","track"])
    # df.to_csv(os.path.join(args.out_dir, "mic_vs_pickup_spectral_summary.csv"), index=False)
    # print(f"Saved per-track metrics -> {os.path.join(args.out_dir, 'mic_vs_pickup_spectral_summary.csv')}")


    # ---- Per-track CSV
    df = pd.DataFrame(rows).sort_values(["track"])
    pertrack_csv = os.path.join(args.out_dir, "mic_vs_pickup_spectral_summary.csv")
    df.to_csv(pertrack_csv, index=False)
    print(f"Saved per-track metrics -> {pertrack_csv}")

    # ---- Global mean/std CSV (across tracks)
    numeric_cols = [
        "low_80_200_db", "lowmid_200_800_db", "mid_0p8_3k_db", "high_3_8k_db",
        "tilt_db_per_decade", "rms_delta_db", "lag_ms"#, "gain_applied_db"
    ]
    stats = df[numeric_cols].agg(['mean', 'std']).T.reset_index()
    stats.columns = ["metric", "mean", "std"]
    stats["mean"] = stats["mean"].round(3)
    stats["std"] = stats["std"].round(3)
    stats_csv = os.path.join(args.out_dir, "mic_vs_pickup_spectral_summary_stats.csv")
    stats.to_csv(stats_csv, index=False)
    print(f"Saved global stats -> {stats_csv}")

    def aggregate_plot(tag, items):
        # Interpolate to common log-f grid for averaging
        fmin, fmax = 80, 8000
        f_grid = np.geomspace(fmin, fmax, 256)
        M = []
        for _, f, d, _ in items:
            d_i = np.interp(f_grid, f, d)
            M.append(d_i)
        M = np.vstack(M)
        mu = np.nanmean(M, axis=0)
        se = np.nanstd(M, axis=0, ddof=1) / math.sqrt(M.shape[0])
        ci = 1.96 * se

        # bigger fonts only (no other changes)
        with mpl.rc_context({
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14
        }):
            plt.figure(figsize=(8,4))
            plt.semilogx(f_grid, mu, lw=2, label=f"Δ(f): mic − pickup ({tag})")
            plt.fill_between(f_grid, mu-ci, mu+ci, alpha=0.2)
            for (lo,hi) in [(80,200),(200,800),(800,3000),(3000,8000)]:
                plt.axvspan(lo, hi, color='k', alpha=0.03)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Δ(f) [dB]")
            plt.title("Spectral coloration (mic − pickup)")
            plt.grid(True, which="both", ls=":", lw=0.5)
            plt.ylim(mu.min()-6, mu.max()+6)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"delta_avg_{tag}.png"), dpi=150)
            plt.close()


    aggregate_plot("all", deltas_all)
    # aggregate_plot("solo", [x for x in deltas_all if x[3]=="solo"])
    # aggregate_plot("comp", [x for x in deltas_all if x[3]=="comp"])
    print(f"Saved aggregate plots in {args.out_dir}")

if __name__ == "__main__":
    main()
