import os as _os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"]:
    _os.environ[_v] = "1"           # force (not setdefault)
_os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tkinter import E
# import museval
from tqdm import tqdm

import numpy as np
import torch
# torch.backends.cudnn.benchmark = True #NEW

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr_fn

import data.utils
import model.utils as model_utils
import utils
import matplotlib.pyplot as plt
import os
import random
# ---- put this at the very top of test.py (before importing numpy/torch/museval) ----

import museval

# --- bss_eval_worker.py-style helper (top-level in your file)
import numpy as np
import museval
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile, uuid, numpy as np


def _limit_threads_in_child():
    import os
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"]:
        os.environ[var] = "1"
    try:
        # best-effort hard cap if available
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass

def enable_strict_determinism(device="gpu", seed=1337):
    # seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # cuDNN/cublas determinism (GPU)
    torch.backends.cudnn.benchmark = False
    if device == "gpu":
        torch.backends.cudnn.deterministic = True
        # Disable TF32 (Ampere+ can otherwise diverge from CPU FP32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Enforce deterministic algos 
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
        torch.use_deterministic_algorithms(True)
    else:
        # CPU branch: nothing special beyond seeds; CPU ops are deterministic.
        torch.use_deterministic_algorithms(True)



def bss_eval_worker(t_path, p_path):
    """Load memmapped arrays and run museval on them."""
    t = np.load(t_path, mmap_mode='r')   # shape (K, T, C)
    p = np.load(p_path, mmap_mode='r')   # shape (K, T, C)
    SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(t, p)
    # cleanup temp files
    try:
        os.remove(t_path); os.remove(p_path)
    except Exception:
        pass
    return SDR, ISR, SIR, SAR



def compute_model_output(model, inputs):
    '''
    Computes outputs of model with given inputs. Does NOT allow propagating gradients! See compute_loss for training.
    Procedure depends on whether we have one model for each source or not
    :param model: Model to train with
    :param compute_grad: Whether to compute gradients
    :return: Model outputs, Average loss over batch
    '''
    all_outputs = {}

    if model.separate:
        for inst in model.instruments:
            output = model(inputs, inst)
            all_outputs[inst] = output[inst].detach().clone()
    else:
        all_outputs = model(inputs)

    return all_outputs

def validate(args, model, criterion, test_data):
    '''
    Iterate with a given model over a given test dataset and compute the desired loss
    :param args: Options dictionary
    :param model: Pytorch model
    :param criterion: Loss function to use (similar to Pytorch criterions)
    :param test_data: Test dataset (Pytorch dataset)
    :return:
    '''
    # PREPARE DATA
    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    # VALIDATE
    model.eval()
    total_loss = 0.
    with tqdm(total=len(test_data) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, targets) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                for k in list(targets.keys()):
                    targets[k] = targets[k].cuda()

            _, avg_loss = model_utils.compute_loss(model, x, targets, criterion)

            total_loss += (1. / float(example_num + 1)) * (avg_loss - total_loss)

            pbar.set_description("Current loss: {:.4f}".format(total_loss))
            pbar.update(1)

    return total_loss


def predict(audio, model, batch_windows=16, use_amp=False):
    """
    Batched window inference for Wave-U-Net-style models.
    audio: np.ndarray, shape (C, T)
    returns: dict {inst: np.ndarray (C, T)}
    """
    device = next(model.parameters()).device
    # device = "cpu"
    model.eval()

    # ensure numpy (C,T)
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    expected_outputs = audio.shape[1]
    out_shift  = model.shapes["output_frames"]
    in_frames  = model.shapes["input_frames"]
    out_frames = model.shapes["output_frames"]

    # pad to multiple of hop
    r = expected_outputs % out_shift
    if r:
        audio = np.pad(audio, [(0,0), (0, out_shift - r)], mode="constant")

    target_outputs = audio.shape[1]

    # context pad front/back
    pad_front = model.shapes["output_start_frame"]
    pad_back  = in_frames - model.shapes["output_end_frame"]
    audio = np.pad(audio, [(0,0), (pad_front, pad_back)], mode="constant")

    starts = np.arange(0, target_outputs, out_shift)

    # preallocate outputs on GPU; fill then bring back
    outs = {
        k: torch.zeros((audio.shape[0], target_outputs), dtype=torch.float32, device=device)
        for k in model.instruments
    }

    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else torch.cpu.amp.autocast
    with torch.inference_mode(), amp_ctx():
        for i in range(0, len(starts), batch_windows):
            chunk_starts = starts[i:i+batch_windows]
            # (B, C, T)
            batch = np.stack([audio[:, s:s+in_frames] for s in chunk_starts], axis=0)
            batch = torch.from_numpy(batch).to(device, non_blocking=True)

            # forward
            if getattr(model, "separate", False):
                out_dict = {}
                for inst in model.instruments:
                    # expect dict with key 'inst' -> (B,C,out_frames)
                    tmp = model(batch, inst)
                    out_dict[inst] = tmp[inst]
            else:
                # expect dict of tensors (B,C,out_frames)
                out_dict = model(batch)

            # write back
            for b, s in enumerate(chunk_starts):
                e = s + out_frames
                for k, o in out_dict.items():
                    outs[k][:, s:e] = o[b]

    # crop and move to numpy
    outs = {k: outs[k][:, :expected_outputs].detach().cpu().numpy() for k in outs}
    return outs


def predict_song(args, audio_path, model, batch_windows=16, use_amp=False):
    """
    Loads a mixture, adapts channels/SR, runs batched predict, adapts back to the original SR/channels.
    Returns dict {inst: np.ndarray (C, T)} at the original file SR and channel count.
    """
    model.eval()

    # Load mixture (C, T) at native SR
    mix_audio, mix_sr = data.utils.load(audio_path, sr=None, mono=False)
    mix_channels, mix_len = mix_audio.shape[0], mix_audio.shape[1]

    # Adapt channels for model
    if args.channels == 1:
        mix_for_model = np.mean(mix_audio, axis=0, keepdims=True)  # (1, T)
    else:
        mix_for_model = mix_audio if mix_channels == args.channels else np.tile(mix_audio, (args.channels, 1))

    # Resample to model SR if needed
    if mix_sr != args.sr:
        mix_for_model = data.utils.resample(mix_for_model, mix_sr, args.sr)

    # Predict at model SR/channels
    pred = predict(mix_for_model, model, batch_windows=batch_windows, use_amp=use_amp)  # dict of (C, Tm)

    # Resample back to original SR if needed, and trim/pad to original length
    out = {}
    for k, y in pred.items():
        y2 = data.utils.resample(y, args.sr, mix_sr) if mix_sr != args.sr else y
        # trim/pad to original length
        diff = y2.shape[1] - mix_len
        if diff > 0:
            y2 = y2[:, :-diff]
        elif diff < 0:
            y2 = np.pad(y2, [(0,0), (0, -diff)], mode="constant")

        # Adapt channel count back to mix
        if mix_channels > args.channels:
            # model mono -> duplicate
            y2 = np.tile(y2, (mix_channels, 1))
        elif mix_channels < args.channels:
            # model stereo -> average
            y2 = np.mean(y2, axis=0, keepdims=True)

        out[k] = np.asfortranarray(y2)

    return out


# def evaluate(args, dataset, model, instruments, batch_windows=16, use_amp=False, use_museval=True):
#     """
#     Evaluates a model on a dataset.
#     Returns (perfs, perfs_comp, perfs_solo) where each is a list of per-track dicts:
#       song[name] = {"SDR": ndarray(frames), "ISR": ..., "SIR": ..., "SAR": ..., "SI-SDR": float}
#     """
#     device = next(model.parameters()).device
#     perfs, perfs_comp, perfs_solo = [], [], []

def _mix_channels_st(x):
    # (S, T, C) or (S, T) -> (S, T) by averaging channels for metric
    return x.mean(axis=2) if x.ndim == 3 else x

#     model.eval()
#     with torch.inference_mode():
#         for count, example in enumerate(dataset):
#             track_name = example["mix"].split('/')[-2]
#             print("Evaluating " + example["mix"])

#             # References (S, T, C) at native SR
#             target_sources = np.stack([
#                 data.utils.load(example[instrument], sr=None, mono=False)[0].T for instrument in instruments
#             ])  # (S, T, C)

#             # Predictions (dict of (C, T)) at native SR, then (S, T, C)
#             pred_dict = predict_song(args, example["mix"], model,
#                                      batch_windows=batch_windows, use_amp=use_amp)
#             pred_sources = np.stack([pred_dict[k].T for k in instruments])  # (S, T, C)

#             # Align lengths just in case
#             n = min(pred_sources.shape[1], target_sources.shape[1])
#             pred_sources   = pred_sources[:, :n]
#             target_sources = target_sources[:, :n]

#             # ---- SI-SDR (vectorized, on GPU) ----
#             if count==0:
#                 t = torch.from_numpy(_mix_channels_st(target_sources)).to(device=device, dtype=torch.float32)  # (S, T)
#                 p = torch.from_numpy(_mix_channels_st(pred_sources)).to(device=device, dtype=torch.float32)   # (S, T)
#                 # si_vec = si_sdr_fn(p, t, reduction='none')  # (S,)
#                 si_vec = si_sdr_fn(p, t)  # (S,)
#                 si_vals = si_vec.detach().cpu().tolist()

#             # ---- BSS Eval v4 (optional; CPU) ----
#             if use_museval:
#                 SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(target_sources, pred_sources)
#             else:
#                 # placeholders matching shapes (frames,) per source if skipping museval
#                 frames = max(1, si_vec.numel())  # dummy
#                 SDR = [np.array([np.nan])] * len(instruments)
#                 ISR = [np.array([np.nan])] * len(instruments)
#                 SIR = [np.array([np.nan])] * len(instruments)
#                 SAR = [np.array([np.nan])] * len(instruments)

#             # pack per-song dict
#             song = {}
#             for idx, name in enumerate(instruments):
#                 song[name] = {
#                     "SDR": SDR[idx], "ISR": ISR[idx], "SIR": SIR[idx], "SAR": SAR[idx],
#                     "SI-SDR": si_vals[idx]
#                 }
#             perfs.append(song)

#             # strata
#             mix_path = example["mix"]
#             if '_comp' in mix_path:
#                 perfs_comp.append(song)
#             if '_solo' in mix_path:
#                 perfs_solo.append(song)

#     return perfs, perfs_comp, perfs_solo

# --- evaluate (uses your original predict and predict_song)
def evaluate(args, dataset, model, instruments):
    """
    Parallel museval; SI-SDR on CPU (same shape usage as your old code).
    """
    # knobs (no args usage)
    BSS_DISABLE  = False   # set True to skip museval
    BSS_WORKERS  = 32      # e.g., 32 on your 48-core box
    BSS_SECONDS  = 0       # 0/<=0 = full track (no cropping)
    # enable_strict_determinism(device="gpu")
    perfs, perfs_comp, perfs_solo = [], [], []

    # process pool for museval
    n_workers = min(32, (os.cpu_count() or 1))  # or pick your own number
    pool = None if BSS_DISABLE else ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_limit_threads_in_child
    )
    print(f"[bss_eval] using {n_workers} workers; OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")    
    
    # pool   = None if BSS_DISABLE else ProcessPoolExecutor(max_workers=BSS_WORKERS)
    tmpdir = tempfile.mkdtemp(prefix="bss_eval_")
    pending = []  # list of (future, song_dict)

    model.eval()
    with torch.no_grad():
        for count, example in enumerate(dataset):
            track_name = example["mix"].split('/')[-2]
            print("Evaluating " + example["mix"])

            # Load refs (K,T,C)
            target_sources = np.stack([
                data.utils.load(example[instrument], sr=None, mono=False)[0].T
                for instrument in instruments
            ])

            # Predict using mixture -> dict {inst: (C,T)} -> stack (K,T,C)
            pred_dict = predict_song(args, example["mix"], model)
            pred_sources = np.stack([pred_dict[key].T for key in instruments])

            # Align lengths (like before)
            n = min(pred_sources.shape[1], target_sources.shape[1])
            pred_sources   = pred_sources[:, :n, :]
            target_sources = target_sources[:, :n, :]

            # # # ---- SI-SDR (CPU; same shapes as your old code)
            # if count==0:
            si_sdr_metric = ScaleInvariantSignalDistortionRatio()
            si_sdr_values = []
            for i in range(len(instruments)):
                # old path: pass tensors with shape (T, C)
                est = torch.tensor(pred_sources[i])      # (T, C)
                ref = torch.tensor(target_sources[i])    # (T, C)
                si_sdr_values.append(si_sdr_metric(est, ref).item())
            # Convert target_sources to torch tensors and move them to GPU
            # target_sources_tensor = torch.tensor(target_sources).float()
            # pred_sources_tensor = torch.tensor(pred_sources).float()            
            # if args.cuda:
            #     target_sources_tensor = target_sources_tensor.cuda()
            #     pred_sources_tensor = pred_sources_tensor.cuda()            
                
            # # Evaluate SI-SDR
            # si_sdr_metric = ScaleInvariantSignalDistortionRatio()

            # si_sdr_values = []
            # for i in range(len(instruments)):
            #     si_sdr_value = si_sdr_metric(
            #         torch.tensor(pred_sources[i]),
            #         torch.tensor(target_sources[i])
            #     ).item()
                # si_sdr_values.append(si_sdr_value)



            # init song dict (will fill SDR/ISR/SIR/SAR later)
            song = {name: {"SDR": None, "ISR": None, "SIR": None, "SAR": None,
                           "SI-SDR": si_sdr_values[idx]}
                    for idx, name in enumerate(instruments)}
            perfs.append(song)
            if "_comp" in example["mix"]:
                perfs_comp.append(song)
            if "_solo" in example["mix"]:
                perfs_solo.append(song)

            # ---- enqueue museval job
            if not BSS_DISABLE:
                ts = target_sources
                ps = pred_sources
                if BSS_SECONDS and BSS_SECONDS > 0:
                    # optional center crop (disabled when BSS_SECONDS <= 0)
                    max_samps = int(args.sr * BSS_SECONDS)
                    mid  = ts.shape[1] // 2
                    half = max_samps // 2
                    sl = slice(max(0, mid - half), min(ts.shape[1], mid + half))
                    ts = ts[:, sl, :]
                    ps = ps[:, sl, :]

                # write arrays to temp files (avoids big IPC pickles)
                t_path = os.path.join(tmpdir, f"t_{uuid.uuid4().hex}.npy")
                p_path = os.path.join(tmpdir, f"p_{uuid.uuid4().hex}.npy")
                # keep dtype as-is to avoid numeric drift
                np.save(t_path, ts)
                np.save(p_path, ps)

                fut = pool.submit(bss_eval_worker, t_path, p_path)
                pending.append((fut, song))

    # ---- collect museval results
    if pending:
        for fut, song in pending:
            SDR, ISR, SIR, SAR = fut.result()
            for i, name in enumerate(instruments):
                song[name]["SDR"] = SDR[i]
                song[name]["ISR"] = ISR[i]
                song[name]["SIR"] = SIR[i]
                song[name]["SAR"] = SAR[i]
        pool.shutdown(wait=True)

    return perfs, perfs_comp, perfs_solo