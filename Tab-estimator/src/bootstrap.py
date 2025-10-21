# pickup Ground-Truth GS-Aux-Pckp 
# python src/bootstrap.py 202407051618_senv_mix_7target 192 --bootstrap 10000
# PP & PR & PF & TP & TR & TF & TDR & PFSolo & TFSolo & PFComp & TFComp 
# 0.883 ± 0.012 & 0.912 ± 0.013 & 0.897 ± 0.011 & 0.882 ± 0.012 & 0.909 ± 0.013 & 0.896 ± 0.011 & 1.001 ± 0.001 & 0.914 ± 0.011 & 0.912 ± 0.011 & 0.891 ± 0.015 & 0.890 ± 0.015 
# 0.857 ± 0.014 & 0.905 ± 0.013 & 0.878 ± 0.013 & 0.856 ± 0.014 & 0.903 ± 0.013 & 0.877 ± 0.013 & 1.001 ± 0.001 & 0.893 ± 0.013 & 0.891 ± 0.013 & 0.864 ± 0.022 & 0.863 ± 0.022 

# pickup ✗ ✗ 
# python src/bootstrap.py 202407042127_senv_mix_standard 192 --bootstrap 10000
# PP & PR & PF & TP & TR & TF & TDR & PFSolo & TFSolo & PFComp & TFComp 
# 0.876 ± 0.013 & 0.868 ± 0.015 & 0.872 ± 0.012 & 0.864 ± 0.013 & 0.857 ± 0.017 & 0.861 ± 0.013 & 0.990 ± 0.005 & 0.908 ± 0.011 & 0.897 ± 0.016 & 0.859 ± 0.016 & 0.848 ± 0.018 
# 0.862 ± 0.014 & 0.874 ± 0.015 & 0.865 ± 0.012 & 0.851 ± 0.015 & 0.864 ± 0.015 & 0.854 ± 0.014 & 0.990 ± 0.006 & 0.890 ± 0.014 & 0.880 ± 0.016 & 0.839 ± 0.021 & 0.828 ± 0.022 

# python src/bootstrap.py 202508201829_senv_mix-preds_7-pred 192 --bootstrap 10000

# pickup multi-W-U-Net Pckp 
# python src/bootstrap.py 202407042308_senv_mix_7pred 192 --bootstrap 10000
# PP & PR & PF & TP & TR & TF & TDR & PFSolo & TFSolo & PFComp & TFComp 
# 0.866 ± 0.013 & 0.891 ± 0.015 & 0.878 ± 0.012 & 0.858 ± 0.014 & 0.882 ± 0.017 & 0.870 ± 0.014 & 0.993 ± 0.006 & 0.909 ± 0.011 & 0.895 ± 0.018 & 0.867 ± 0.017 & 0.861 ± 0.018 
# 0.847 ± 0.015 & 0.882 ± 0.016 & 0.865 ± 0.013 & 0.839 ± 0.016 & 0.873 ± 0.018 & 0.856 ± 0.015 & 0.993 ± 0.006 & 0.899 ± 0.012 & 0.886 ± 0.019 & 0.851 ± 0.019 & 0.844 ± 0.020 

# pickup multi-W-U-Net GS-Aux-Pckp 
# # python src/bootstrap.py 202508142123_senv_mix-preds-by-gscustmix_7-pred 192 --bootstrap 10000
# 0.870 ± 0.013 & 0.868 ± 0.017 & 0.869 ± 0.013 & 0.858 ± 0.014 & 0.855 ± 0.018 & 0.856 ± 0.014 & 0.988 ± 0.006 & 0.907 ± 0.012 & 0.893 ± 0.020 & 0.855 ± 0.018 & 0.843 ± 0.019 
# 0.858 ± 0.014 & 0.875 ± 0.014 & 0.864 ± 0.013 & 0.846 ± 0.016 & 0.863 ± 0.016 & 0.851 ± 0.016 & 0.987 ± 0.009 & 0.890 ± 0.015 & 0.879 ± 0.020 & 0.837 ± 0.021 & 0.824 ± 0.023 

# pickup multi-W-U-Net GS-Aux-Pckp+ADGP-Pckp
# python src/bootstrap.py 202508191636_senv_mix-preds-by-gscustmix-pdgp_7-pred 192 --bootstrap 10000
# 0.869 ± 0.013 & 0.882 ± 0.016 & 0.875 ± 0.013 & 0.860 ± 0.014 & 0.871 ± 0.017 & 0.865 ± 0.013 & 0.991 ± 0.005 & 0.910 ± 0.010 & 0.895 ± 0.015 & 0.863 ± 0.017 & 0.855 ± 0.018 
# 0.854 ± 0.014 & 0.884 ± 0.013 & 0.866 ± 0.013 & 0.845 ± 0.015 & 0.872 ± 0.014 & 0.856 ± 0.014 & 0.990 ± 0.006 & 0.894 ± 0.013 & 0.882 ± 0.016 & 0.839 ± 0.021 & 0.830 ± 0.022 

# ------------------------------------------------------------------

# mic Ground-Truth ✗
# python src/bootstrap.py 202407042101_senv_mic_7target  192 --bootstrap 10000
# PP & PR & PF & TP & TR & TF & TDR & PFSolo & TFSolo & PFComp & TFComp 
# 0.877 ± 0.013 & 0.910 ± 0.013 & 0.893 ± 0.012 & 0.876 ± 0.012 & 0.908 ± 0.013 & 0.892 ± 0.012 & 1.001 ± 0.001 & 0.913 ± 0.011 & 0.909 ± 0.012 & 0.886 ± 0.016 & 0.885 ± 0.016 
# 0.857 ± 0.014 & 0.905 ± 0.013 & 0.878 ± 0.013 & 0.856 ± 0.014 & 0.902 ± 0.013 & 0.876 ± 0.014 & 1.000 ± 0.000 & 0.891 ± 0.015 & 0.888 ± 0.015 & 0.865 ± 0.022 & 0.865 ± 0.022 

# mic ✗ ✗
# python src/bootstrap.py 202407041716_senv_mic_standard 192 --bootstrap 10000
# PP & PR & PF & TP & TR & TF & TDR & PFSolo & TFSolo & PFComp & TFComp 
# 0.855 ± 0.013 & 0.872 ± 0.018 & 0.863 ± 0.013 & 0.828 ± 0.015 & 0.848 ± 0.022 & 0.838 ± 0.017 & 0.976 ± 0.009 & 0.894 ± 0.012 & 0.866 ± 0.017 & 0.852 ± 0.018 & 0.828 ± 0.023 
# 0.843 ± 0.014 & 0.872 ± 0.016 & 0.854 ± 0.013 & 0.816 ± 0.017 & 0.850 ± 0.018 & 0.829 ± 0.016 & 0.973 ± 0.010 & 0.879 ± 0.016 & 0.856 ± 0.018 & 0.828 ± 0.022 & 0.803 ± 0.028

# mic multi-W-U-Net Mic 
# python src/bootstrap.py 202407032134_senv_mic_7pred 192 --bootstrap 10000 
# PP & PR & PF & TP & TR & TF & TDR & PFSolo & TFSolo & PFComp & TFComp 
# 0.855 ± 0.013 & 0.844 ± 0.017 & 0.850 ± 0.013 & 0.775 ± 0.028 & 0.769 ± 0.034 & 0.772 ± 0.031 & 0.914 ± 0.028 & 0.893 ± 0.012 & 0.821 ± 0.027 & 0.835 ± 0.018 & 0.754 ± 0.040 
# 0.845 ± 0.014 & 0.855 ± 0.015 & 0.846 ± 0.013 & 0.763 ± 0.025 & 0.781 ± 0.026 & 0.769 ± 0.024 & 0.911 ± 0.024 & 0.880 ± 0.014 & 0.817 ± 0.025 & 0.813 ± 0.021 & 0.722 ± 0.040 

# mic multi-W-U-Net Mic+ADGP-Mic
# python src/bootstrap.py 202502062000_senv_mic_fakemic-nopret 192 --bootstrap 10000
# 0.855 ± 0.014 & 0.842 ± 0.018 & 0.848 ± 0.014 & 0.783 ± 0.024 & 0.777 ± 0.031 & 0.780 ± 0.027 & 0.926 ± 0.024 & 0.895 ± 0.011 & 0.839 ± 0.024 & 0.832 ± 0.019 & 0.758 ± 0.036 
# 0.845 ± 0.014 & 0.854 ± 0.015 & 0.846 ± 0.013 & 0.774 ± 0.023 & 0.793 ± 0.024 & 0.780 ± 0.023 & 0.927 ± 0.022 & 0.880 ± 0.014 & 0.831 ± 0.023 & 0.812 ± 0.021 & 0.730 ± 0.039 


# mic multi-W-U-Net GS-Aux-Mic
# python src/bootstrap.py 202407051631_senv_pseudomic_7pred 192 --bootstrap 10000
# PP & PR & PF & TP & TR & TF & TDR & PFSolo & TFSolo & PFComp & TFComp 
# 0.861 ± 0.013 & 0.849 ± 0.019 & 0.855 ± 0.013 & 0.812 ± 0.019 & 0.803 ± 0.028 & 0.807 ± 0.023 & 0.949 ± 0.018 & 0.898 ± 0.012 & 0.847 ± 0.022 & 0.839 ± 0.018 & 0.793 ± 0.030 
# 0.849 ± 0.014 & 0.851 ± 0.015 & 0.846 ± 0.013 & 0.795 ± 0.020 & 0.805 ± 0.021 & 0.796 ± 0.020 & 0.943 ± 0.018 & 0.879 ± 0.015 & 0.837 ± 0.020 & 0.812 ± 0.021 & 0.757 ± 0.035
 
# mic multi-W-U-Net GS-Aux-Mic+ADGP-Mic
# python src/bootstrap.py 202502062033_senv_mic_preds_by_guit_pseudoboth_sep_all_solos_fake_7-pred 192 --bootstrap 10000
# PP & PR & PF & TP & TR & TF & TDR & PFSolo & TFSolo & PFComp & TFComp 
# 0.854 ± 0.013 & 0.859 ± 0.019 & 0.857 ± 0.013 & 0.798 ± 0.021 & 0.806 ± 0.030 & 0.802 ± 0.024 & 0.941 ± 0.020 & 0.896 ± 0.011 & 0.843 ± 0.026 & 0.843 ± 0.019 & 0.787 ± 0.033 
# 0.846 ± 0.014 & 0.862 ± 0.015 & 0.850 ± 0.013 & 0.789 ± 0.021 & 0.810 ± 0.024 & 0.796 ± 0.022 & 0.937 ± 0.021 & 0.881 ± 0.014 & 0.838 ± 0.024 & 0.819 ± 0.021 & 0.754 ± 0.037

import glob
import numpy as np
import os
import torch
import pandas as pd
import tqdm
import yaml
from network import TabEstimator
import argparse
from sklearn.metrics import precision_recall_fscore_support



import csv
from typing import Dict, List


from sklearn.metrics import precision_recall_fscore_support

def _bin_prf(y_pred_bin: np.ndarray, y_true_bin: np.ndarray):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0)
    return float(p), float(r), float(f1)

# ---------- utils (micro only) ----------

def prf_from_counts(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return p, r, f

def confusion_counts(pred: np.ndarray, gt: np.ndarray):
    """Binary arrays -> (tp, fp, fn, tn) as ints."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int(np.sum(pred_b & gt_b))
    fp = int(np.sum(pred_b & ~gt_b))
    fn = int(np.sum(~pred_b & gt_b))
    tn = pred_b.size - tp - fp - fn
    return tp, fp, fn, tn

def tab2pitch(tab):
    rel_string_pitches = [0, 5, 10, 15, 19, 24]
    argmax_index = np.argmax(tab, axis=2)
    pitch = np.zeros((len(tab), 44))
    for time in range(len(tab)):
        for string in range(6):
            if argmax_index[time, string] < 20:
                pitch[time, string and 0 or 0]  # no-op to keep structure; replaced below
    pitch = np.zeros((len(tab), 44))
    for t in range(len(tab)):
        for s in range(6):
            a = argmax_index[t, s]
            if a < 20:
                pitch[t, a + rel_string_pitches[s]] = 1
    return pitch

# micro-only bootstrap over per-file counts
def bootstrap_micro(perfile: pd.DataFrame, n_boot: int = 10000, seed: int = 1337,
                    stratified: bool = True) -> pd.DataFrame:
    """
    Bootstrap over files. For each resample, pool counts across sampled files and compute:
      - frame, note, frameF0, noteF0: micro P/R/F1 from TP/FP/FN
      - frame_tdr, note_tdr: micro TDR from pooled (num/den)
    Returns mean/std/CI; supports subsets 'all' (always) and 'solo'/'comp' if present.
    """
    rng = np.random.default_rng(seed)
    subsets = ['all']
    if 'subset' in perfile.columns:
        subs_found = [s for s in ['solo', 'comp'] if s in perfile['subset'].unique()]
        subsets += subs_found

    rows = []

    def summarize(a):
        a = np.asarray(a, dtype=float)
        return float(np.mean(a)), float(np.std(a, ddof=1)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    for subset in subsets:
        if subset == 'all':
            df = perfile.copy()
            has_groups = 'subset' in df.columns and set(df['subset']) & {'solo', 'comp'}
            n_solo = int((df['subset'] == 'solo').sum()) if has_groups else 0
            n_comp = int((df['subset'] == 'comp').sum()) if has_groups else 0
        else:
            df = perfile[perfile['subset'] == subset].copy()
            has_groups = False
            n_solo = n_comp = 0

        N = len(df)
        if N == 0:
            continue

        def resample_df():
            if stratified and subset == 'all' and has_groups and (n_solo + n_comp) > 0:
                s = df[df['subset'] == 'solo']; c = df[df['subset'] == 'comp']
                idx_s = rng.integers(0, len(s), size=n_solo) if n_solo > 0 else np.array([], int)
                idx_c = rng.integers(0, len(c), size=n_comp) if n_comp > 0 else np.array([], int)
                return pd.concat([
                    s.iloc[idx_s] if n_solo > 0 else s.iloc[0:0],
                    c.iloc[idx_c] if n_comp > 0 else c.iloc[0:0]
                ], ignore_index=True)
            else:
                idx = rng.integers(0, N, size=N)
                return df.iloc[idx]

        # ---- micro P/R/F1 from pooled counts ----
        def one_metric(tp_col, fp_col, fn_col, prefix):
            p_vals, r_vals, f_vals = [], [], []
            for _ in range(n_boot):
                samp = resample_df()
                tp = float(samp[tp_col].sum()); fp = float(samp[fp_col].sum()); fn = float(samp[fn_col].sum())
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
                p_vals.append(p); r_vals.append(r); f_vals.append(f)
            for name, arr in [('p', p_vals), ('r', r_vals), ('f1', f_vals)]:
                mean, std, lo, hi = summarize(arr)
                rows.append(dict(subset=subset, metric=f'{prefix}_{name}',
                                 mean=mean, std=std, ci95_lo=lo, ci95_hi=hi))

        # def one_macro_metric(cols, prefix):
        #     for name in cols:
        #         vals = []
        #         for _ in range(n_boot):
        #             samp = resample_df()
        #             vals.append(float(samp[name].mean()))
        #         mean, std, lo, hi = summarize(vals)
        #         rows.append(dict(subset=subset, metric=f'{prefix}_{name[-1]}',
        #                         mean=mean, std=std, ci95_lo=lo, ci95_hi=hi))

        one_metric('frame_tp',  'frame_fp',  'frame_fn',  'frame')
        one_metric('frameF0_tp','frameF0_fp','frameF0_fn','frameF0')
        # one_macro_metric(['note_p','note_r','note_f1'], 'note')
        # one_macro_metric(['noteF0_p','noteF0_r','noteF0_f1'], 'noteF0')

        def one_metric_macro(col_name):
            vals = []
            for _ in range(n_boot):
                samp = resample_df()
                vals.append(float(samp[col_name].mean()))  # macro mean in this resample
            mean, std, lo, hi = summarize(vals)
            rows.append(dict(subset=subset, metric=col_name, mean=round(mean,3), std=round(std,3),
                            ci95_lo=round(lo,3), ci95_hi=round(hi,3)))


        for m in ['note_p','note_r','note_f1','noteF0_p','noteF0_r','noteF0_f1','note_tdr']:
            if m in df.columns:
                one_metric_macro(m)

        # ---- micro TDR from pooled numerators/denominators ----
        def one_tdr(num_col, den_col, name):
            if num_col in df.columns and den_col in df.columns:
                vals = []
                for _ in range(n_boot):
                    samp = resample_df()
                    num = float(samp[num_col].sum())
                    den = float(samp[den_col].sum())
                    tdr = (num / den) if den > 0 else 0.0
                    vals.append(tdr)
                mean, std, lo, hi = summarize(vals)
                rows.append(dict(subset=subset, metric=name,
                                 mean=mean, std=std, ci95_lo=lo, ci95_hi=hi))

        one_tdr('frame_tdr_num', 'frame_tdr_den', 'frame_tdr')
        one_tdr('note_tdr_num',  'note_tdr_den',  'note_tdr')

    return pd.DataFrame(rows).round(3)

# ---------- core eval (micro only) ----------

def calc_score(args, test_num, trained_model, use_model_epoch, config_path,
               plot_results=False, input_as_random_noize=False,
               make_notelvl_from_framelvl=False, verbose=True):
    with open(config_path) as f:
        obj = yaml.safe_load(f)
        input_feature_type = obj["input_feature_type"]
        mode = obj["mode"]
        encoder_type = obj["encoder_type"]
        use_custom_decimation_func = obj["use_custom_decimation_func"]
        use_conv_stack = obj["use_conv_stack"]
        down_sampling_rate = obj["down_sampling_rate"]
        hop_length = obj["hop_length"]
        cqt_n_bins = obj["cqt_n_bins"]
        partition_mode = obj["partition_mode"]
        feat_mode = obj["feat_mode"]
        npz_path = obj["npz_path"]
        encoder_heads = obj["encoder_heads"]
        encoder_layers = obj["encoder_layers"]

    if input_feature_type == "cqt":
        n_bins = cqt_n_bins
    elif input_feature_type == "melspec":
        n_bins = 128
    n_channels = 7 if (feat_mode in ['7-pred','7-target']) else 1

    model_path = f"model/{trained_model}/testNo0{test_num}/epoch{use_model_epoch}.model"
    model = TabEstimator(mode, encoder_type, use_custom_decimation_func, use_conv_stack,
                         n_bins, hop_length, down_sampling_rate,
                         n_channels=n_channels, encoder_heads=encoder_heads, encoder_layers=encoder_layers)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    if verbose:
        print(f"{test_num=}, {mode=}")

    if partition_mode == 'senvaityte':
        test_data_path = os.path.join("data", npz_path, feat_mode, "split", "*.npz")
        test_data_list = np.array(glob.glob(test_data_path, recursive=True))
        SenveityteNameList = os.path.join('../datasets', 'NMFtestSet.csv')
        with open(SenveityteNameList, newline='') as csvfile:
            testreader = csv.reader(csvfile, delimiter=',')
            testfiles = ['_'.join(row[4].split('_',2)[:2]) for row in testreader]
        test_data_list = [p for p in test_data_list if '_'.join(os.path.split(p)[1].split('_')[:2]) in testfiles]
    else:
        test_data_path = os.path.join("data", "npz", "original", "split", f"0{test_num}_*.npz")
        test_data_list = np.array(glob.glob(test_data_path, recursive=True))

    # per-file counts for micro bootstrapping
    per_file_rows: List[Dict] = []

    # global (micro) counters (overall + subsets)
    # totals = dict(
    #     frame_tp=0, frame_fp=0, frame_fn=0,
    #     note_tp=0,  note_fp=0,  note_fn=0,
    #     frameF0_tp=0, frameF0_fp=0, frameF0_fn=0,
    #     noteF0_tp=0,  noteF0_fp=0,  noteF0_fn=0
    # )
    totals = dict(
        frame_tp=0, frame_fp=0, frame_fn=0,
        frameF0_tp=0, frameF0_fp=0, frameF0_fn=0
    )

    totals_solo = totals.copy()
    totals_comp = totals.copy()

    for npz_filename in tqdm.tqdm(test_data_list):
        npz_file = np.load(npz_filename)
        if input_feature_type == "cqt":
            input_features = torch.from_numpy(npz_file["cqt"])
        elif input_feature_type == "melspec":
            input_features = torch.from_numpy(npz_file["mel_spec"])

        note_gt = torch.from_numpy(npz_file["tab"])
        frame_gt = torch.from_numpy(npz_file["frame_tab"])
        note_F0_gt = npz_file["F0"]
        frame_F0_gt = npz_file["frame_F0"]

        bpm = torch.from_numpy(npz_file["tempo"])
        bpm = torch.unsqueeze(bpm, 0)

        frame_len = torch.zeros(1); frame_len[0] = input_features.shape[1]
        note_len = torch.zeros(1);  note_len[0]  = note_gt.shape[0]

        if input_as_random_noize:
            input_features = torch.rand(input_features.shape)
        input_features = torch.unsqueeze(input_features, 0)

        with torch.no_grad():
            frame_pred, note_pred, _ = model(input_features.float(), frame_len, note_len, bpm)

        # to one-hot picks (argmax per string)
        frame_pred = torch.squeeze(frame_pred, 0)
        ai = np.argmax(frame_pred.detach().numpy(), axis=2)
        f = np.zeros((len(frame_pred), 6, 21)); 
        for t in range(len(frame_pred)):
            for s in range(6):
                f[t, s, ai[t, s]] = 1
        frame_pred = f
        frame_F0_from_tab_pred = tab2pitch(frame_pred)

        note_pred = torch.squeeze(note_pred, 0)
        ai = np.argmax(note_pred.detach().numpy(), axis=2)
        n = np.zeros((len(note_pred), 6, 21))
        for t in range(len(note_pred)):
            for s in range(6):
                n[t, s, ai[t, s]] = 1
        note_pred = n
        note_F0_from_tab_pred = tab2pitch(note_pred)

        # numpy ground truth
        frame_gt = frame_gt.detach().numpy()
        note_gt = note_gt.detach().numpy()

        # remove 'not played' class then flatten
        frame_pred_b = frame_pred[:, :, :-1].flatten()
        frame_gt_b   = frame_gt[:,   :, :-1].flatten()
        note_pred_b  = note_pred[:,  :, :-1].flatten()
        note_gt_b    = note_gt[:,    :, :-1].flatten()

        note_p, note_r, note_f1 = _bin_prf(note_pred_b, note_gt_b)


        frameF0_pred_b = frame_F0_from_tab_pred.flatten()
        frameF0_gt_b   = frame_F0_gt.flatten()
        noteF0_pred_b  = note_F0_from_tab_pred.flatten()
        noteF0_gt_b    = note_F0_gt.flatten()



        # --- NEW --- # NOTE
        frame_TP_tab = float(np.multiply(frame_gt[:, :, :-1], frame_pred[:, :, :-1]).sum())
        frame_TP_F0  = float(np.multiply(frame_F0_gt, tab2pitch(frame_pred)).sum())



        # note TDR (per file)
        note_TP_tab = float(np.multiply(note_gt[:, :, :-1], note_pred[:, :, :-1]).sum())
        note_TP_F0  = float(np.multiply(note_F0_gt, note_F0_from_tab_pred).sum())
        note_tdr = (note_TP_tab / note_TP_F0) if note_TP_F0 > 0 else 0.0

        # counts (file)
        fr_tp, fr_fp, fr_fn, _   = confusion_counts(frame_pred_b, frame_gt_b)
        note_p, note_r, note_f1 = _bin_prf(note_pred_b, note_gt_b)
        frF0_tp, frF0_fp, frF0_fn, _ = confusion_counts(frameF0_pred_b, frameF0_gt_b)
        noteF0_p, noteF0_r, noteF0_f1 = _bin_prf(noteF0_pred_b, noteF0_gt_b)

        subset = 'solo' if "_solo_" in npz_filename else ('comp' if "_comp_" in npz_filename else 'all')

        # per_file_rows.append(dict(
        #     file=os.path.basename(npz_filename),
        #     subset=subset,
        #     frame_tp=fr_tp, frame_fp=fr_fp, frame_fn=fr_fn,
        #     note_tp=no_tp,  note_fp=no_fp,  note_fn=no_fn,
        #     frameF0_tp=frF0_tp, frameF0_fp=frF0_fp, frameF0_fn=frF0_fn,
        #     noteF0_tp=noF0_tp,  noteF0_fp=noF0_fp,  noteF0_fn=noF0_fn,
        #     frame_tdr_num=frame_TP_tab, frame_tdr_den=frame_TP_F0, # NOTE
        #     note_tdr=note_tdr,  # note_tdr_den=note_TP_F0,
        # ))        

        per_file_rows.append(dict(
            file=os.path.basename(npz_filename),
            subset=subset,
            # ---- frame (micro counts stay as-is) ----
            frame_tp=fr_tp, frame_fp=fr_fp, frame_fn=fr_fn,
            frameF0_tp=frF0_tp, frameF0_fp=frF0_fp, frameF0_fn=frF0_fn,
            frame_tdr_num=frame_TP_tab, frame_tdr_den=frame_TP_F0,
            # ---- NOTE (macro per-file scores) ----
            note_p=note_p, note_r=note_r, note_f1=note_f1,
            noteF0_p=noteF0_p, noteF0_r=noteF0_r, noteF0_f1=noteF0_f1,
            note_tdr=note_tdr,
        ))


        # accumulate totals (overall)
        for k, v in dict(frame_tp=fr_tp, frame_fp=fr_fp, frame_fn=fr_fn,
                        #  note_tp=no_tp,  note_fp=no_fp,  note_fn=no_fn,
                         frameF0_tp=frF0_tp, frameF0_fp=frF0_fp, frameF0_fn=frF0_fn,
                        #  noteF0_tp=noF0_tp,  noteF0_fp=noF0_fp,  noteF0_fn=noF0_fn
                         ).items():
            totals[k] += v
            if subset == 'solo':
                totals_solo[k] += v
            elif subset == 'comp':
                totals_comp[k] += v

    # overall micro metrics
    frame_p, frame_r, frame_f1 = prf_from_counts(totals['frame_tp'], totals['frame_fp'], totals['frame_fn'])
    # note_p,  note_r,  note_f1  = prf_from_counts(totals['note_tp'],  totals['note_fp'],  totals['note_fn'])
    frameF0_p, frameF0_r, frameF0_f1 = prf_from_counts(totals['frameF0_tp'], totals['frameF0_fp'], totals['frameF0_fn'])
    # noteF0_p,  noteF0_r,  noteF0_f1  = prf_from_counts(totals['noteF0_tp'],  totals['noteF0_fp'],  totals['noteF0_fn'])

    # ---- note (MACRO over files) ----
    per_file_df = pd.DataFrame(per_file_rows)
    note_p  = float(per_file_df['note_p'].mean())   if len(per_file_df) else 0.0
    note_r  = float(per_file_df['note_r'].mean())   if len(per_file_df) else 0.0
    note_f1 = float(per_file_df['note_f1'].mean())  if len(per_file_df) else 0.0
    noteF0_p  = float(per_file_df['noteF0_p'].mean())   if len(per_file_df) else 0.0
    noteF0_r  = float(per_file_df['noteF0_r'].mean())   if len(per_file_df) else 0.0
    noteF0_f1 = float(per_file_df['noteF0_f1'].mean())  if len(per_file_df) else 0.0
    # For subsets (solo/comp) F1s as macro:
    if 'subset' in per_file_df.columns:
        s_note_f1    = float(per_file_df[per_file_df['subset']=='solo']['note_f1'].mean())    if (per_file_df['subset']=='solo').any() else 0.0
        c_note_f1    = float(per_file_df[per_file_df['subset']=='comp']['note_f1'].mean())    if (per_file_df['subset']=='comp').any() else 0.0
        s_noteF0_f1  = float(per_file_df[per_file_df['subset']=='solo']['noteF0_f1'].mean())  if (per_file_df['subset']=='solo').any() else 0.0
        c_noteF0_f1  = float(per_file_df[per_file_df['subset']=='comp']['noteF0_f1'].mean())  if (per_file_df['subset']=='comp').any() else 0.0
    else:
        s_note_f1 = c_note_f1 = s_noteF0_f1 = c_noteF0_f1 = 0.0
    # If you want macro TDR in the summary row too (optional):
    note_tdr_macro = float(per_file_df['note_tdr'].mean()) if len(per_file_df) else 0.0



    # # subset micro metrics (solo/comp) if present
    # def safe_prf(t):
    #     return prf_from_counts(t['frame_tp'], t['frame_fp'], t['frame_fn']), \
    #            prf_from_counts(t['note_tp'],  t['note_fp'],  t['note_fn']), \
    #            prf_from_counts(t['frameF0_tp'], t['frameF0_fp'], t['frameF0_fn']), \
    #            prf_from_counts(t['noteF0_tp'],  t['noteF0_fp'],  t['noteF0_fn'])

    def safe_prf(t):
        return (
            prf_from_counts(t['frame_tp'], t['frame_fp'], t['frame_fn']),
            prf_from_counts(t['frameF0_tp'], t['frameF0_fp'], t['frameF0_fn'])
        )

    solo_present = any(r['subset']=='solo' for r in per_file_rows)
    comp_present = any(r['subset']=='comp' for r in per_file_rows)

    if solo_present:
        (s_frame_p, s_frame_r, s_frame_f1), (s_frameF0_p, s_frameF0_r, s_frameF0_f1) = safe_prf(totals_solo)
    else:
        s_frame_p = s_frame_r = s_frame_f1 = 0.0
        s_frameF0_p = s_frameF0_r = s_frameF0_f1 = 0.0

    if comp_present:
        (c_frame_p, c_frame_r, c_frame_f1), (c_frameF0_p, c_frameF0_r, c_frameF0_f1) = safe_prf(totals_comp)
    else:
        c_frame_p = c_frame_r = c_frame_f1 = 0.0
        c_frameF0_p = c_frameF0_r = c_frameF0_f1 = 0.0


    # if solo_present:
    #     (s_frame_p, s_frame_r, s_frame_f1), (s_note_p, s_note_r, s_note_f1), \
    #     (s_frameF0_p, s_frameF0_r, s_frameF0_f1), (s_noteF0_p, s_noteF0_r, s_noteF0_f1) = safe_prf(totals_solo)
    # else:
    #     s_frame_p=s_frame_r=s_frame_f1=s_note_p=s_note_r=s_note_f1= \
    #     s_frameF0_p=s_frameF0_r=s_frameF0_f1=s_noteF0_p=s_noteF0_r=s_noteF0_f1 = 0.0

    # if comp_present:
    #     (c_frame_p, c_frame_r, c_frame_f1), (c_note_p, c_note_r, c_note_f1), \
    #     (c_frameF0_p, c_frameF0_r, c_frameF0_f1), (c_noteF0_p, c_noteF0_r, c_noteF0_f1) = safe_prf(totals_comp)
    # else:
    #     c_frame_p=c_frame_r=c_frame_f1=c_note_p=c_note_r=c_note_f1= \
    #     c_frameF0_p=c_frameF0_r=c_frameF0_f1=c_noteF0_p=c_noteF0_r=c_noteF0_f1 = 0.0

    # pack results (MICRO ONLY)
    result = pd.DataFrame([[
        round(frame_p, 3),   round(frame_r, 3),   round(frame_f1, 3),
        round(note_p, 3),    round(note_r, 3),    round(note_f1, 3),
        round(frameF0_p, 3), round(frameF0_r, 3), round(frameF0_f1, 3),
        round(noteF0_p, 3),  round(noteF0_r, 3),  round(noteF0_f1, 3),
        round(s_frame_f1, 3), round(c_frame_f1, 3),
        round(s_note_f1, 3),  round(c_note_f1, 3),
        round(s_frameF0_f1, 3), round(c_frameF0_f1, 3),
        round(s_noteF0_f1, 3),  round(c_noteF0_f1, 3)
    ]],
    columns=[
        "frame_p","frame_r","frame_f1",
        "note_p","note_r","note_f1",
        "frameF0_p","frameF0_r","frameF0_f1",
        "noteF0_p","noteF0_r","noteF0_f1",
        "frame_f1_solo","frame_f1_comp",
        "note_f1_solo","note_f1_comp",
        "frameF0_f1_solo","frameF0_f1_comp",
        "noteF0_f1_solo","noteF0_f1_comp",
    ],
    index=[f"No0{test_num}"])

    per_file_df = pd.DataFrame(per_file_rows)
    return result, per_file_df


def main(args):
    trained_model = args.model
    use_model_epoch = args.epoch
    verbose = args.verbose

    result_path = os.path.join("result")
    os.makedirs(result_path, exist_ok=True)

    result = pd.DataFrame()
    config_path = os.path.join("model", f"{trained_model}", "config.yaml")

    with open(config_path) as f:
        obj = yaml.safe_load(f)
        mode = obj["mode"]
        partition_mode = obj["partition_mode"]

    all_perfile = []
    csv_path = os.path.join(result_path, f"{mode}", f"{trained_model}_epoch{use_model_epoch}", "metrics.csv")

    if partition_mode == 'senvaityte':
        df, perfile = calc_score(args, 0, trained_model, use_model_epoch, config_path, verbose=verbose)
        result = pd.concat([result, df])
        all_perfile.append(perfile)
    else:
        for test_num in range(6):
            print(f"Player No. {test_num}")
            df, perfile = calc_score(args, test_num, trained_model, use_model_epoch, config_path, verbose=verbose)
            result = pd.concat([result, df])
            all_perfile.append(perfile)

    # save micro-only metrics
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    result.to_csv(csv_path, float_format="%.3f")

    # save per-file counts for possible micro bootstrapping
    if len(all_perfile):
        perfile_df = pd.concat(all_perfile, ignore_index=True)
        perfile_csv = os.path.join(os.path.dirname(csv_path), "perfile_metrics.csv")
        perfile_df.to_csv(perfile_csv, index=False)

        if args.bootstrap > 0:
            boots = bootstrap_micro(
                perfile_df,
                n_boot=args.bootstrap,
                seed=args.seed,
                stratified=(not args.no_stratify)
            )
            boots_csv = os.path.join(os.path.dirname(csv_path), "bootstrap_summary.csv")
            boots.to_csv(boots_csv, index=False)
            if verbose:
                print(f"Saved bootstrap summary -> {boots_csv}")
                print(f"(n_boot={args.bootstrap}, stratified={not args.no_stratify}, seed={args.seed})")



            print_bootstrap_summary(boots)


    if verbose:
        print(f"Saved metrics -> {csv_path}")

def print_bootstrap_summary(boot_df: pd.DataFrame):
    """
    Prints two LaTeX rows (header + FRAME row + NOTE row) in this order:
    PP PR PF  TP TR TF  TDR  PF_solo TF_solo PF_comp TF_comp

    Micro metrics only. Values are mean ± 95% CI half-width (3 decimals).
    Expects boot_df columns: subset, metric, mean, ci95_lo, ci95_hi.
    FRAME uses:  frameF0_p/r/f1  and  frame_p/r/f1, plus frame_tdr (if present).
    NOTE  uses:  noteF0_p/r/f1   and   note_p/r/f1,  plus note_tdr  (if present).
    """

    def get_mean_hw(subset, metric):
        m = boot_df[(boot_df['subset'] == subset) & (boot_df['metric'] == metric)]
        if m.empty:
            return None, None
        mean = float(m.iloc[0]['mean'])
        hw = float(m.iloc[0]['ci95_hi'] - m.iloc[0]['ci95_lo']) / 2.0
        return round(mean, 3), round(hw, 3)

    def fmt(val_hw):
        m, hw = val_hw
        return r"{:.3f} ± {:.3f}".format(m, hw) if (m is not None and hw is not None) else r"—"

    # ---------- FRAME (overall) ----------
    f_PP, f_PP_hw = get_mean_hw('all',  'frameF0_p')
    f_PR, f_PR_hw = get_mean_hw('all',  'frameF0_r')
    f_PF, f_PF_hw = get_mean_hw('all',  'frameF0_f1')
    f_TP, f_TP_hw = get_mean_hw('all',  'frame_p')
    f_TR, f_TR_hw = get_mean_hw('all',  'frame_r')
    f_TF, f_TF_hw = get_mean_hw('all',  'frame_f1')
    f_TDR,f_TDR_hw= get_mean_hw('all',  'frame_tdr')

    # FRAME solo/comp (F1 only)
    f_PFs, f_PFs_hw = get_mean_hw('solo', 'frameF0_f1')
    f_TFs, f_TFs_hw = get_mean_hw('solo', 'frame_f1')
    f_PFc, f_PFc_hw = get_mean_hw('comp', 'frameF0_f1')
    f_TFc, f_TFc_hw = get_mean_hw('comp', 'frame_f1')

    # ---------- NOTE (overall) ----------
    n_PP, n_PP_hw = get_mean_hw('all',  'noteF0_p')
    n_PR, n_PR_hw = get_mean_hw('all',  'noteF0_r')
    n_PF, n_PF_hw = get_mean_hw('all',  'noteF0_f1')
    n_TP, n_TP_hw = get_mean_hw('all',  'note_p')
    n_TR, n_TR_hw = get_mean_hw('all',  'note_r')
    n_TF, n_TF_hw = get_mean_hw('all',  'note_f1')
    n_TDR,n_TDR_hw= get_mean_hw('all',  'note_tdr')

    # NOTE solo/comp (F1 only)
    n_PFs, n_PFs_hw = get_mean_hw('solo', 'noteF0_f1')
    n_TFs, n_TFs_hw = get_mean_hw('solo', 'note_f1')
    n_PFc, n_PFc_hw = get_mean_hw('comp', 'noteF0_f1')
    n_TFc, n_TFc_hw = get_mean_hw('comp', 'note_f1')

    # ---------- Header ----------
    print(r"PP & PR & PF & "
          r"TP & TR & TF & TDR & "
          r"PFSolo & TFSolo & "
          r"PFComp & TFComp ")

    # ---------- FRAME row ----------
    print("{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} ".format(
        fmt((f_PP, f_PP_hw)), fmt((f_PR, f_PR_hw)), fmt((f_PF, f_PF_hw)),
        fmt((f_TP, f_TP_hw)), fmt((f_TR, f_TR_hw)), fmt((f_TF, f_TF_hw)),
        fmt((f_TDR, f_TDR_hw)),
        fmt((f_PFs, f_PFs_hw)), fmt((f_TFs, f_TFs_hw)),
        fmt((f_PFc, f_PFc_hw)), fmt((f_TFc, f_TFc_hw)),
    ))

    # ---------- NOTE row ----------
    print("{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} ".format(
        fmt((n_PP, n_PP_hw)), fmt((n_PR, n_PR_hw)), fmt((n_PF, n_PF_hw)),
        fmt((n_TP, n_TP_hw)), fmt((n_TR, n_TR_hw)), fmt((n_TF, n_TF_hw)),
        fmt((n_TDR, n_TDR_hw)),
        fmt((n_PFs, n_PFs_hw)), fmt((n_TFs, n_TFs_hw)),
        fmt((n_PFc, n_PFc_hw)), fmt((n_TFc, n_TFc_hw)),
    ))



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='micro-only evaluation')
    parser.add_argument("model", type=str)
    parser.add_argument("epoch", type=int)
    parser.add_argument("-v","--verbose", action="store_true", default=False)
    parser.add_argument("--bootstrap", type=int, default=10000, help="Number of bootstrap resamples (0 = disable).")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for bootstrap.")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified bootstrap (solo/comp preserved).")
    args = parser.parse_args()
    main(args)
