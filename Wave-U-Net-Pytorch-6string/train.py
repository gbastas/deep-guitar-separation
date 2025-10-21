# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/datasep-mic/ --cuda --hdf_dir hdfs/hdf_guit-mic --checkpoint_dir mycheckpoints/waveunet_guit_mic-mdgp/ --channels 1 --patience -1 --version pseudo
# 1.19 \pm 0.38 & 9.59 \pm 1.34 & 0.34 \pm 0.41 & 9.52 \pm 1.28 & 1.27 \pm 0.52 & 1.10 \pm 0.56 & 5.01 \pm 0.42 & 14.03 \pm 1.36 \\

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/datasep-mic/ --cuda --hdf_dir hdfs/hdf_guit-mic --checkpoint_dir mycheckpoints/waveunet_guit_mic/ --channels 1 --patience -1 --version pseudo
# 0.90 \pm 0.43 & 8.40 \pm 1.36 & 0.29 \pm 0.38 & 9.59 \pm 1.30 & 0.92 \pm 0.64 & 0.89 \pm 0.58 & 5.05 \pm 0.42 & 14.13 \pm 1.31 \\

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/datasep-mic/ --cuda --hdf_dir hdfs/hdf_guit-mic/ --checkpoint_dir mycheckpoints/waveunet_guit_gscustmic/ --channels 1 --patience -1 --version pseudo
# -2.33 \pm 0.33 & 0.45 \pm 1.18 & -1.51 \pm 0.33 & 8.14 \pm 0.99 & -2.66 \pm 0.42 & -2.00 \pm 0.51 & 5.19 \pm 0.44 & 11.09 \pm 1.30 \\

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/datasep-mic/ --cuda --hdf_dir hdfs/hdf_guit-mic/ --checkpoint_dir mycheckpoints/waveunet_guit_gscustmic_mdgp/ --channels 1 --patience -1 --version pseudo
# -2.25 \pm 0.30 & 1.97 \pm 0.98 & -1.28 \pm 0.40 & 8.42 \pm 0.99 & -2.47 \pm 0.40 & -2.02 \pm 0.43 & 5.30 \pm 0.42 & 11.54 \pm 1.24 \\
    
#  CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/datasep-hex/ --cuda --hdf_dir hdfs/hdf_guit-hex/ --checkpoint_dir mycheckpoints/waveunet_guit_hex/ --channels 1 --patience -1 --version pseudo  
# 7.40 \pm 0.74 & 16.61 \pm 1.29 & 8.09 \pm 0.69 & 12.17 \pm 1.53 & 8.69 \pm 1.16 & 6.12 \pm 0.72 & 6.99 \pm 0.58 & 17.35 \pm 1.58 \\
    
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/datasep-mix/ --cuda --hdf_dir hdfs/hdf_guit-mix --checkpoint_dir mycheckpoints/waveunet_guit_mix/ --channels 1 --patience -1 --version pseudo
# 5.75 \pm 0.71 & 15.99 \pm 1.28 & 6.70 \pm 0.61 & 11.87 \pm 1.49 & 6.56 \pm 1.01 & 4.94 \pm 0.91 & 6.79 \pm 0.54 & 16.95 \pm 1.57 \\
    
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/datasep-mix/ --cuda --hdf_dir hdfs/hdf_guit-mix --checkpoint_dir mycheckpoints/waveunet_guit_mix_pdgp/ --channels 1 --patience -1 --version pseudo
# 5.89 \pm 0.68 & 16.66 \pm 1.19 & 6.72 \pm 0.59 & 11.77 \pm 1.48 & 6.66 \pm 0.97 & 5.11 \pm 0.90 & 6.72 \pm 0.52 & 16.82 \pm 1.56 \\

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/datasep-mix/ --cuda --hdf_dir hdfs/hdf_guit-mix/ --checkpoint_dir checkpoints/waveunet_guit_gscustmix-pdgp/ --channels 1 --patience -1 --version pseudo
#     0.74 \pm 0.41 & 3.13 \pm 1.17 & 2.19 \pm 0.42 & 6.05 \pm 0.67 & 0.66 \pm 0.66 & 0.81 \pm 0.43 & 3.98 \pm 0.38 & 8.11 \pm 0.79 \\
    
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/datasep-mix/ --cuda --hdf_dir hdfs/hdf_guit-mix/ --checkpoint_dir checkpoints/waveunet_guit_gscustmix/ --channels 1 --patience -1 --version pseudo    
# 1.79 \pm 0.47 & 5.14 \pm 1.15 & 1.52 \pm 0.40 & 8.45 \pm 1.04 & 2.55 \pm 0.66 & 1.03 \pm 0.57 & 5.45 \pm 0.40 & 11.45 \pm 1.43 \\
    
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/datasep-mix/ --cuda --hdf_dir hdfs/hdf_guit-mix/ --checkpoint_dir checkpoints/waveunet_guit_gscustmix-pdgp/ --channels 1 --patience -1 --version pseudo
# 2.46 \pm 0.45 & 8.63 \pm 1.27 & 2.47 \pm 0.44 & 8.62 \pm 1.00 & 2.77 \pm 0.61 & 2.15 \pm 0.63 & 5.40 \pm 0.36 & 11.84 \pm 1.15 \\

import argparse
import os
import time
from functools import partial

import torch
# torch.backends.cudnn.benchmark = True #NEW

import pickle
import numpy as np

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

import model.utils as model_utils
import utils
from data.dataset import SeparationDataset
from data.musdb import get_musdb_folds, get_musdbhq
from data.utils import crop_targets, random_amplify
from test import evaluate, validate
from model.waveunet import Waveunet
import csv
import glob
# torch.backends.cudnn.benchmark = True

import csv
import numpy as np

def _fmt_pm(mean, lo, hi, dec=2):
    if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
        return "--"
    return f"{mean:.{dec}f} \\pm {((hi - lo) / 2):.{dec}f}"


def _collect_track_values(perfs, instruments, metric_name):
    """
    perfs: list of per-song dicts like perfs[...] you already build
    instruments: list like ["E","A","D","G","B","e"]
    metric_name: "SDR" | "SIR" | "SAR" | "SI-SDR"
    returns: dict inst->list_of_track_values, plus key 'overall' (avg across inst per track)
    """
    vals = {inst: [] for inst in instruments}
    overall = []
    for song in perfs:
        per_song = []
        for inst in instruments:
            v = song[inst][metric_name]
            if isinstance(v, (list, tuple, np.ndarray)):
                per_track = float(np.nanmean(np.asarray(v)))
            else:
                per_track = float(v)
            vals[inst].append(per_track)
            per_song.append(per_track)
        if len(per_song) > 0:
            overall.append(float(np.mean(per_song)))
    vals["overall"] = overall
    return vals

def _bootstrap_mean_ci(values, B=2000, alpha=0.05, seed=1337):
    """
    Classic nonparametric bootstrap for the mean.
    Returns (mean, ci_lo, ci_hi).
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1 - alpha/2)])
    return (mean, float(lo), float(hi))


def main(args):

    # MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)

    features=None

    print('cuda', args.cuda)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    writer = SummaryWriter(args.log_dir)

    ### DATASET
    musdb = get_musdb_folds(args.dataset_dir, version=args.version, guitID=args.split)
    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop_targets, shapes=model.shapes)
    # Data augmentation function for training
    augment_func = partial(random_amplify, shapes=model.shapes, min=0.7, max=1.0)
    train_data = SeparationDataset(musdb, "train", args.instruments, args.sr, args.channels, model.shapes, True, args.hdf_dir, audio_transform=augment_func, features=features) # NOTE: augmentation
    val_data = SeparationDataset(musdb, "val", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func, features=features)

        
    print('No comp/solo distinct val scores will be considered')
    test_data = SeparationDataset(musdb, "test", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func, features=features)

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)


    ##### TRAINING ####

    # Set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    # Set up optimiser
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf,
             "best_comp_loss" : np.Inf,
             "best_solo_loss" : np.Inf,
             "best_mic_loss" : np.Inf,
             "best_mix_loss" : np.Inf,
             "best_hex_cln_loss" : np.Inf}
             

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)
        state["best_loss"]=np.Inf 
        state["best_comp_loss"]=np.Inf 
        state["best_solo_loss"]=np.Inf 
        state["best_mic_loss"]=np.Inf 
        state["best_mix_loss"]=np.Inf 
        state["best_hex_cln_loss"]=np.Inf 

    print('TRAINING START')
    while state["worse_epochs"] < args.patience:
        print("Training one epoch from iteration " + str(state["step"]))
        avg_time = 0.
        model.train()
        with tqdm(total=len(train_data) // args.batch_size) as pbar:
            np.random.seed()
            for example_num, (x, targets) in enumerate(dataloader):
                if args.cuda:
                    x = x.cuda()
                    for k in list(targets.keys()):
                        targets[k] = targets[k].cuda()

                t = time.time()

                # Set LR for this iteration
                utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

                # Compute loss for each instrument/model
                optimizer.zero_grad()
                outputs, avg_loss = model_utils.compute_loss(model, x, targets, criterion, compute_grad=True)

                optimizer.step()

                state["step"] += 1

                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                writer.add_scalar("train_loss", avg_loss, state["step"])


                pbar.update(1)

        # VALIDATE
        val_loss = validate(args, model, criterion, val_data)


            
        print("VALIDATION FINISHED: LOSS: " + str(val_loss))
        writer.add_scalar("val_loss", val_loss, state["step"])


        # EARLY STOPPING CHECK
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            checkpoint_best_path = os.path.join(args.checkpoint_dir, "best_checkpoint_" + str(state["step"]))
            try:
                os.remove(checkpoint_best_path_prev)
            except Exception as e:
                print('Caught exception:', e)

            print("Saving model...")
            
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_best_path
            model_utils.save_model(model, optimizer, state, checkpoint_best_path)


        # CHECKPOINT
        print("Saving model...")
        model_utils.save_model(model, optimizer, state, checkpoint_path)
        try:
            os.remove(checkpoint_path_prev)
        except Exception as e:
            print('Caught exception:', e)        
        checkpoint_path_prev = checkpoint_path
        try:
            checkpoint_best_path_prev = checkpoint_best_path
        except Exception as e:
            print("MyException", e)


        state["epochs"] += 1

    #### TESTING ####
    print("TESTING")

    # Load best model based on validation loss
    if args.patience > 0:
        state = model_utils.load_model(model, None, state["best_checkpoint"], args.cuda) 
    elif args.load_model is not None:
        state = model_utils.load_model(model, None, args.load_model, args.cuda)
    else:
        #  NEW: for easier testing
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, "best_checkpoint_*")))
        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No best_checkpoint_# file found in {args.checkpoint_dir}")
        args.load_model = checkpoints[-1]  # or use [0] for lowest step
        state = model_utils.load_model(model, None, args.load_model, args.cuda)
        
    if args.patience>=0:
        test_loss = validate(args, model, criterion, test_data)
        print("TEST FINISHED: LOSS: " + str(test_loss))
        writer.add_scalar("test_loss", test_loss, state["step"])

    test_metrics = {}

    # Mir_eval metrics
    test_metrics['total'], test_metrics['comp'], test_metrics['solo'] = evaluate(args, musdb["test"], model, args.instruments) 

    latex_cache = {"total": {}, "comp": {}, "solo": {}}

    # Dump all metrics results into pickle file for later analysis if needed
    with open(os.path.join(args.checkpoint_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics['total'], f)

    for key,metrics in reversed(test_metrics.items()):
        print()
        print('!!Evaluating on '+ key + ' test set!!')


        # -------- Bootstrap CIs (across tracks) --------
        BOOTSTRAP = True
        B          = 10000      # e.g., 2k resamples
        ALPHA      = 0.05
        SEED_BASE  = 1337

        if BOOTSTRAP:
            metrics_to_do = ["SDR", "SIR", "SAR", "SI-SDR"]
            # collect per-track arrays, then bootstrap per metric/instrument + overall
            collected = {
                m: _collect_track_values(metrics, args.instruments, m) for m in metrics_to_do
            }


            # Cache overall means/CIs for LaTeX row
            for m in ["SDR", "SIR", "SAR", "SI-SDR"]:
                mean, lo, hi = _bootstrap_mean_ci(
                    collected[m]["overall"],
                    B=B, alpha=ALPHA,
                    seed=SEED_BASE + hash((key, m, "overall")) % 10_000_000
                )
                latex_cache[key][m] = (mean, lo, hi)

            # write a new CSV with CIs (one row per instrument, plus 'overall')
            if args.patience > 0:
                ci_csv_path = os.path.join(
                    args.checkpoint_dir, f"{key}_bootstrap_CI_{state['best_checkpoint'].split('_')[-1]}.csv"
                )
            else:
                ci_csv_path = os.path.join(
                    args.checkpoint_dir, f"{key}_bootstrap_CI_{args.load_model.split('_')[-1]}.csv"
                )

            with open(ci_csv_path, "w", newline="") as fci:
                w = csv.writer(fci)
                # header: Instrument, then mean/CI for each metric
                header = ["Instrument"]
                for m in metrics_to_do:
                    header += [f"{m}_mean", f"{m}_ci95_lo", f"{m}_ci95_hi"]
                w.writerow(header)



                # Pretty print (paper-ready) for TOTAL split: Overall SDR with 95% CI
                if key.lower() == "total":
                    sdr_overall_vals = collected["SDR"]["overall"]
                    m, lo, hi = _bootstrap_mean_ci(
                        sdr_overall_vals, B=B, alpha=ALPHA, seed=SEED_BASE
                    )
                    half = (hi - lo) / 2.0
                    print()
                    print(f"[total] Overall SDR (95% CI): {m:.2f} dB [{lo:.2f}, {hi:.2f}]")
                    print(f"[total] i.e., {m:.2f} ± {half:.2f} dB")
                    # (Optional LaTeX-friendly line)
                    # print(f"SDR = {m:.2f}\\,dB\\;[{lo:.2f},\\,{hi:.2f}]\\;\\text{{(95\\% CI)}}")
                    print("\nLaTeX row (mean \\pm half 95\\% CI):")
                    latex_line = " & ".join([
                        _fmt_pm(*latex_cache.get("total", {}).get("SDR", (np.nan, np.nan, np.nan))),
                        _fmt_pm(*latex_cache.get("total", {}).get("SIR", (np.nan, np.nan, np.nan))),
                        _fmt_pm(*latex_cache.get("total", {}).get("SAR", (np.nan, np.nan, np.nan))),
                        _fmt_pm(*latex_cache.get("total", {}).get("SI-SDR", (np.nan, np.nan, np.nan))),
                        _fmt_pm(*latex_cache.get("solo",  {}).get("SDR", (np.nan, np.nan, np.nan))),
                        _fmt_pm(*latex_cache.get("comp",  {}).get("SDR", (np.nan, np.nan, np.nan))),
                        _fmt_pm(*latex_cache.get("solo",  {}).get("SI-SDR", (np.nan, np.nan, np.nan))),
                        _fmt_pm(*latex_cache.get("comp",  {}).get("SI-SDR", (np.nan, np.nan, np.nan))),
                    ]) + r" \\"
                    print(latex_line)
                    print()



                # instruments + overall
                for inst in list(args.instruments) + ["overall"]:
                    row = [inst]
                    for j, m in enumerate(metrics_to_do):
                        mean, lo, hi = _bootstrap_mean_ci(
                            collected[m][inst],
                            B=B,
                            alpha=ALPHA,
                            seed=SEED_BASE + hash((key, inst, m)) % 10_000_000
                        )
                        row += [round(mean, 3), round(lo, 3), round(hi, 3)]
                    w.writerow(row)

            print(f"[bootstrap] wrote 95% CIs -> {ci_csv_path}")




        # Write most important metrics into Tensorboard log
        avg_SDRs = {inst : np.mean([np.nanmean(song[inst]["SDR"]) for song in metrics]) for inst in args.instruments}
        avg_SIRs = {inst : np.mean([np.nanmean(song[inst]["SIR"]) for song in metrics]) for inst in args.instruments}
        avg_SARs = {inst : np.mean([np.nanmean(song[inst]["SAR"]) for song in metrics]) for inst in args.instruments}
        avg_SISDRs = {inst : np.mean([np.nanmean(song[inst]["SI-SDR"]) for song in metrics]) for inst in args.instruments}
        if args.patience > 0: # = right after train is complete
            resfile = open(args.checkpoint_dir+'/'+key+'_results_'+ state["best_checkpoint"].split('_')[-1] +'.csv', 'w')
        else:
            resfile = open(args.checkpoint_dir+'/'+key+'_results_'+ args.load_model.split('_')[-1] +'.csv', 'w')
        csvwriter = csv.writer(resfile)
        csvwriter.writerow([" ","SDR", "SIR", "SAR", "SI-SDR"])

        for inst in args.instruments:
            csvwriter.writerow([inst, round(avg_SDRs[inst],3), round(avg_SIRs[inst],3), round(avg_SARs[inst],3), round(avg_SISDRs[inst],3)])
        #     print(inst, round(avg_SDRs[inst],3), round(avg_SIRs[inst],3), round(avg_SARs[inst],3), round(avg_SISDRs[inst],3))



        overall_SDR = np.mean([v for v in avg_SDRs.values()])
        overall_SIR = np.mean([v for v in avg_SIRs.values()])
        overall_SAR = np.mean([v for v in avg_SARs.values()])
        overall_SISDR = np.mean([v for v in avg_SISDRs.values()])

        csvwriter.writerow([" "])
        csvwriter.writerow([round(overall_SDR,3), round(overall_SIR,3), round(overall_SAR,3), round(overall_SISDR,3)])

        print("SDR: " + str(overall_SDR))
        print("SIR: " + str(overall_SIR))
        print("SAR: " + str(overall_SAR))
        print("SI-SDR: " + str(overall_SISDR))

    writer.close()
    resfile.close()

if __name__ == '__main__':
    ## TRAIN PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["E", "A", "D", "G", "B", "e"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")") # __gbastas__
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--log_dir', type=str, default='logs/waveunet',
                        help='Folder to write logs into')
    parser.add_argument('--dataset_dir', type=str, default="/mnt/windaten/Datasets/MUSDB18HQ",
                        help='Dataset path')
    parser.add_argument('--hdf_dir', type=str, default="hdf",
                        help='Dataset path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet',
                        help='Folder to write checkpoints into')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate in LR cycle (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=5e-5,
                        help='Minimum learning rate in LR cycle (default: 5e-5)')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of LR cycles per epoch')
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=1,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=2.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--example_freq', type=int, default=200,
                        help="Write an audio summary into Tensorboard logs every X training iterations")
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--version', type=str, default='HQ', help='"cross-val" alternatively')
    args = parser.parse_args()

    args.log_dir = 'logs/'+args.checkpoint_dir.split('/')[-1]

    if args.split>=0:
        args.log_dir = args.log_dir + '_' + str(args.split)
        args.checkpoint_dir = args.checkpoint_dir + '_' + str(args.split)
        args.hdf_dir = args.hdf_dir + '_' + str(args.split)

    main(args)
