#!/usr/bin/env python3
"""
Compute & plot relative spectral coloration (mic vs pickup).

Δ(f) = mean_k[ 20*log10 |STFT{mic}(f,k)| ] - mean_k[ 20*log10 |STFT{pickup}(f,k)| ]

Usage:
  python coloration_mic_vs_pickup.py \
      --pickup path/to/pickup.wav \
      --mic path/to/mic.wav \
      --sr 44100 \
      --n_fft 4096 --hop 1024 \
      --out_fig coloration_mic_vs_pickup.png \
      --out_csv coloration_mic_vs_pickup.csv
"""

import argparse
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
import csv
from typing import Tuple, Dict

def stft_logmag_db(x: np.ndarray, sr: int, n_fft: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return frequency axis (Hz) and time-averaged log-magnitude in dB."""
    eps = 1e-12
    S = lr.stft(x, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    mag = np.abs(S) + eps
    L = 20.0 * np.log10(mag)
    L_mean = L.mean(axis=1)  # average over frames
    freqs = np.linspace(0, sr/2, L_mean.shape[0])
    return freqs, L_mean

def band_means(freqs: np.ndarray, delta_db: np.ndarray) -> Dict[str, float]:
    """Band-averaged Δ(f) over fixed musical bands."""
    bands = {
        "low (80–200 Hz)": (80, 200),
        "low-mid (200–800 Hz)": (200, 800),
        "mid (0.8–3 kHz)": (800, 3000),
        "high (3–8 kHz)": (3000, 8000),
    }
    out = {}
    for name, (f1, f2) in bands.items():
        m = (freqs >= f1) & (freqs <= f2)
        out[name] = float(delta_db[m].mean()) if np.any(m) else np.nan
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickup", required=True, help="Path to pickup WAV")
    ap.add_argument("--mic", required=True, help="Path to mic/mix WAV")
    ap.add_argument("--sr", type=int, default=44100, help="Analysis sample rate")
    ap.add_argument("--n_fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=1024)
    ap.add_argument("--out_fig", default="coloration_mic_vs_pickup.png")
    ap.add_argument("--out_csv", default=None, help="Optional CSV to dump Δ(f)")
    ap.add_argument("--trim_db", type=float, default=None,
                    help="Optional librosa.effects.trim top_db (e.g., 60) to remove leading/trailing silence")
    args = ap.parse_args()

    # Load (mono), resample
    x, sr_x = lr.load(args.pickup, sr=args.sr, mono=True)
    y, sr_y = lr.load(args.mic, sr=args.sr, mono=True)

    # Optional silence trim (independently per signal)
    if args.trim_db is not None:
        x, _ = lr.effects.trim(x, top_db=args.trim_db)
        y, _ = lr.effects.trim(y, top_db=args.trim_db)

    # Match durations (pad/truncate to min length)
    N = min(len(x), len(y))
    x = x[:N]
    y = y[:N]

    # Spectral log-mag (dB), averaged over time
    freqs, Lx = stft_logmag_db(x, args.sr, args.n_fft, args.hop)
    _,    Ly = stft_logmag_db(y, args.sr, args.n_fft, args.hop)

    # Relative coloration Δ(f) in dB (mic − pickup)
    delta_db = Ly - Lx

    # Print band means
    means = band_means(freqs, delta_db)
    print("\nRelative spectral coloration (mic − pickup), band-averaged [dB]:")
    for k, v in means.items():
        print(f"  {k:>18}: {v:+.2f} dB")

    # Save CSV if requested
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["freq_hz", "delta_db_mic_minus_pickup", "mic_logmag_db", "pickup_logmag_db"])
            for f_hz, d, ly, lx in zip(freqs, delta_db, Ly, Lx):
                w.writerow([f_hz, d, ly, lx])
        print(f"Saved CSV: {args.out_csv}")

    # Plot
    plt.figure(figsize=(9, 4.8))
    plt.plot(freqs, delta_db, linewidth=1.3)
    plt.xlim(20, min(args.sr/2, 20000))
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Δ log-magnitude (dB)  [mic − pickup]")
    plt.title("Relative Spectral Coloration")
    plt.grid(True, which="both", alpha=0.2)

    # Shade bands & annotate means
    bands = [(80,200), (200,800), (800,3000), (3000,8000)]
    labels = list(means.keys())
    for (f1,f2), lab in zip(bands, labels):
        plt.axvspan(f1, f2, color="0.85", alpha=0.4)
        m = (freqs >= f1) & (freqs <= f2)
        if np.any(m):
            plt.hlines(means[lab], f1, f2, linestyles="--")

    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=200)
    print(f"Saved figure: {args.out_fig}")

if __name__ == "__main__":
    main()
