import argparse, numpy as np, pandas as pd

def boots_macro(vals, n_boot=2000, seed=1337):
    """Bootstrap mean & 95% CI over a 1D array (ignores NaNs)."""
    v = np.array(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    N = v.size
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        means.append(np.nanmean(v[idx]))
    means = np.array(means, dtype=float)
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def fmt_pm(mean, lo, hi, dec=3):
    """Format as LaTeX mean ± half-width CI, e.g., 0.896 ± 0.012."""
    if np.isnan(mean):
        return r"--"
    hw = (hi - lo) / 2.0
    return rf"\({mean:.{dec}f} \pm {hw:.{dec}f}\)"

def get_subset(df, name):
    if name == "all" or "subset" not in df.columns:
        return df
    return df[df["subset"] == name]

def extract_block(df, subset, n_boot, seed):
    d = get_subset(df, subset)
    # Pitch: macro P/R/F1 from frame_F0_*
    PP_m, PP_lo, PP_hi = boots_macro(d["frame_F0_p"].values,  n_boot, seed)
    PR_m, PR_lo, PR_hi = boots_macro(d["frame_F0_r"].values,  n_boot, seed)
    PF_m, PF_lo, PF_hi = boots_macro(d["frame_F0_f1"].values, n_boot, seed)
    # Tab: macro P/R/F1 from frame_*
    TP_m, TP_lo, TP_hi = boots_macro(d["frame_p"].values,     n_boot, seed)
    TR_m, TR_lo, TR_hi = boots_macro(d["frame_r"].values,     n_boot, seed)
    TF_m, TF_lo, TF_hi = boots_macro(d["frame_f1"].values,    n_boot, seed)
    # TDR (frame)
    TDR_m, TDR_lo, TDR_hi = boots_macro(d["frame_tdr"].values, n_boot, seed)
    return dict(
        PP=(PP_m, PP_lo, PP_hi), PR=(PR_m, PR_lo, PR_hi), PF=(PF_m, PF_lo, PF_hi),
        TP=(TP_m, TP_lo, TP_hi), TR=(TR_m, TR_lo, TR_hi), TF=(TF_m, TF_lo, TF_hi),
        TDR=(TDR_m, TDR_lo, TDR_hi)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfile", required=True, help="Path to perfile_metrics.csv")
    ap.add_argument("--input_mix", required=True, help='e.g. "pickup" or "mic"')
    ap.add_argument("--separated", required=True, help='e.g. "Ground-Truth" or "Wave-U-Net"')
    ap.add_argument("--trainset",  required=True, help='e.g. "\\ding{55}" or "\\checkmark"')
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed",   type=int, default=1337)
    ap.add_argument("--rowcolor", default="", help='e.g. "\\rowcolor{gray!20} " (include trailing space if used)')
    args = ap.parse_args()

    df = pd.read_csv(args.perfile)

    # ALL (global): PP, PR, PF, TP, TR, TF, TDR
    all_blk = extract_block(df, "all", args.n_boot, args.seed)

    # SOLO/COMP: PF_solo/comp, TF_solo/comp (F1 only)
    solo = get_subset(df, "solo")
    comp = get_subset(df, "comp")
    PFs_m, PFs_lo, PFs_hi = boots_macro(solo["frame_F0_f1"].values, args.n_boot, args.seed) if len(solo) else (float("nan"),)*3
    TFs_m, TFs_lo, TFs_hi = boots_macro(solo["frame_f1"].values,    args.n_boot, args.seed) if len(solo) else (float("nan"),)*3
    PFc_m, PFc_lo, PFc_hi = boots_macro(comp["frame_F0_f1"].values, args.n_boot, args.seed) if len(comp) else (float("nan"),)*3
    TFc_m, TFc_lo, TFc_hi = boots_macro(comp["frame_f1"].values,    args.n_boot, args.seed) if len(comp) else (float("nan"),)*3

    # Format cells
    PP = fmt_pm(*all_blk["PP"]); PR = fmt_pm(*all_blk["PR"]); PF = fmt_pm(*all_blk["PF"])
    TP = fmt_pm(*all_blk["TP"]); TR = fmt_pm(*all_blk["TR"]); TF = fmt_pm(*all_blk["TF"])
    TDR = fmt_pm(*all_blk["TDR"])
    PF_solo = fmt_pm(PFs_m, PFs_lo, PFs_hi); TF_solo = fmt_pm(TFs_m, TFs_lo, TFs_hi)
    PF_comp = fmt_pm(PFc_m, PFc_lo, PFc_hi); TF_comp = fmt_pm(TFc_m, TFc_lo, TFc_hi)

    # Build LaTeX row
    row = (
        f'{args.rowcolor}{args.input_mix} & {args.separated} & {args.trainset} & '
        f'{PP} & {PR} & {PF} & {TP} & {TR} & {TF} & {TDR} & '
        f'{PF_solo} & {TF_solo} & {PF_comp} & {TF_comp} \\\\'
    )
    print(row)

if __name__ == "__main__":
    main()
