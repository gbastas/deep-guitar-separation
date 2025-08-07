from tkinter import E
import museval
from tqdm import tqdm

import numpy as np
import torch
from torchmetrics import ScaleInvariantSignalDistortionRatio


import data.utils
import model.utils as model_utils
import utils
import matplotlib.pyplot as plt
import os


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

def predict(audio, model):
    '''
    Predict sources for a given audio input signal, with a given model. Audio is split into chunks to make predictions on each chunk before they are concatenated.
    :param audio: Audio input tensor, either Pytorch tensor or numpy array
    :param model: Pytorch model
    :return: Source predictions, dictionary with source names as keys
    '''
    if isinstance(audio, torch.Tensor):
        is_cuda = audio.is_cuda()
        audio = audio.detach().cpu().numpy()
        return_mode = "pytorch"
    else:
        return_mode = "numpy"

    expected_outputs = audio.shape[1]

    # Pad input if it is not divisible in length by the frame shift number
    output_shift = model.shapes["output_frames"]
    pad_back = audio.shape[1] % output_shift
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0,0), (0, pad_back)], mode="constant", constant_values=0.0)

    target_outputs = audio.shape[1]
    outputs = {key: np.zeros(audio.shape, np.float32) for key in model.instruments}

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"]
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    audio = np.pad(audio, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)

    # Iterate over mixture magnitudes, fetch network prediction
    with torch.no_grad():
        for target_start_pos in range(0, target_outputs, model.shapes["output_frames"]):
            # Prepare mixture excerpt by selecting time interval
            curr_input = audio[:, target_start_pos:target_start_pos + model.shapes["input_frames"]] # Since audio was front-padded input of [targetpos:targetpos+inputframes] actually predicts [targetpos:targetpos+outputframes] target range

            # Convert to Pytorch tensor for model prediction
            curr_input = torch.from_numpy(curr_input).unsqueeze(0)

            # Predict
            for key, curr_targets in compute_model_output(model, curr_input).items():
                outputs[key][:,target_start_pos:target_start_pos+model.shapes["output_frames"]] = curr_targets.squeeze(0).cpu().numpy()

    # Crop to expected length (since we padded to handle the frame shift)
    outputs = {key : outputs[key][:,:expected_outputs] for key in outputs.keys()}

    if return_mode == "pytorch":
        outputs = torch.from_numpy(outputs)
        if is_cuda:
            outputs = outputs.cuda()
    return outputs

def predict_song(args, audio_path, model):
    '''
    Predicts sources for an audio file for which the file path is given, using a given model.
    Takes care of resampling the input audio to the models sampling rate and resampling predictions back to input sampling rate.
    :param args: Options dictionary
    :param audio_path: Path to mixture audio file
    :param model: Pytorch model
    :return: Source estimates given as dictionary with keys as source names
    '''
    model.eval()

    # Load mixture in original sampling rate
    mix_audio, mix_sr = data.utils.load(audio_path, sr=None, mono=False)
    mix_channels = mix_audio.shape[0]
    mix_len = mix_audio.shape[1]

    # Adapt mixture channels to required input channels
    if args.channels == 1:
        mix_audio = np.mean(mix_audio, axis=0, keepdims=True)
    else:
        if mix_channels == 1: # Duplicate channels if input is mono but model is stereo
            mix_audio = np.tile(mix_audio, [args.channels, 1])
        else:
            assert(mix_channels == args.channels)

    # resample to model sampling rate
    mix_audio = data.utils.resample(mix_audio, mix_sr, args.sr)

    sources = predict(mix_audio, model)

    # Resample back to mixture sampling rate in case we had model on different sampling rate
    sources = {key : data.utils.resample(sources[key], args.sr, mix_sr) for key in sources.keys()}

    # In case we had to pad the mixture at the end, or we have a few samples too many due to inconsistent down- and upsamṕling, remove those samples from source prediction now
    for key in sources.keys():
        diff = sources[key].shape[1] - mix_len
        if diff > 0:
            print("WARNING: Cropping " + str(diff) + " samples")
            sources[key] = sources[key][:, :-diff]
        elif diff < 0:
            print("WARNING: Padding output by " + str(diff) + " samples")
            sources[key] = np.pad(sources[key], [(0,0), (0, -diff)], "constant", 0.0)

        # Adapt channels
        if mix_channels > args.channels:
            assert(args.channels == 1)
            # Duplicate mono predictions
            sources[key] = np.tile(sources[key], [mix_channels, 1])
        elif mix_channels < args.channels:
            assert(mix_channels == 1)
            # Reduce model output to mono
            sources[key] = np.mean(sources[key], axis=0, keepdims=True)

        sources[key] = np.asfortranarray(sources[key]) # So librosa does not complain if we want to save it

    return sources

def plot_waveforms(ref, est, sr, out_dir, track_name, instr_name, sdr, si_sdr, hop_size_s = 1.5, win_size_s = 2):

    os.makedirs(out_dir, exist_ok=True)

    ref = np.asarray(ref).squeeze()   # (984505,1) → (984505,)
    est = np.asarray(est).squeeze()

    avg_sdr    = float(np.nanmean(sdr))


    n = min(len(ref), len(est))
    ref = ref[:n]
    est = est[:n]
    time = np.linspace(0, n / sr, num=n)
    max_amp = max(np.max(np.abs(ref)), np.max(np.abs(est)))

    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    # Panel 1: overlay
    axs[0].plot(time, ref,   alpha=0.6, label="Ref.")
    axs[0].plot(time, est,   alpha=0.6, label="Est.")
    axs[0].fill_between(time, ref, est,
                        where=(ref > est), facecolor='blue', alpha=0.2)
    axs[0].fill_between(time, ref, est,
                        where=(ref < est), facecolor='red',  alpha=0.2)
    axs[0].set_ylim(-max_amp, max_amp)
    axs[0].set_xlabel("Time (s)", fontsize=14)
    axs[0].set_ylabel("Amplitude", fontsize=14)
    axs[0].legend(loc="upper right", fontsize=13, framealpha=0.3)

    # Panel 2: frame SDR + SI‑SDR
    fr = np.asarray(sdr).flatten()
    frame_times = hop_size_s * np.arange(fr.shape[0]) + (win_size_s/2)
    axs[1].plot(frame_times, fr, marker='o', linestyle='-',
                color='green', markersize=5, alpha=0.8, label=r"$\mathrm{SDR}^r$")
                # label="Windowed SDR")
    axs[1].axhline(y=si_sdr, color='red', linestyle='--',
                   linewidth=1.5, label="SI‑SDR")
    axs[1].axhline(y=avg_sdr, color='green', linestyle='--',
                   linewidth=1.5, label="SDR")    
    axs[1].set_ylim(min(-0.5, np.nanmin(fr)) - 0.9,
                    max(si_sdr, np.nanmax(fr)) + 0.9)
    axs[1].set_xlabel("Time (s)", fontsize=14)
    axs[1].set_ylabel("(SI‑)SDR (dB)", fontsize=14)
    axs[1].legend(loc="upper right", fontsize=13, framealpha=0.9)


    # Title and subtitle with metrics
    fig.subplots_adjust(top=0.88)  # make room for subtitle
    axs[0].set_title(f"SDR: {avg_sdr:.2f} dB    |    SI-SDR: {si_sdr:.2f} dB",
                     fontsize=12)


    plt.tight_layout()
    filename = f"{track_name}__{instr_name}.png"
    outpath = os.path.join(out_dir, filename)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  ▶ saved plot: {outpath}")



def evaluate(args, dataset, model, instruments):
    '''
    Evaluates a given model on a given dataset
    :param args: Options dict
    :param dataset: Dataset object
    :param model: Pytorch model
    :param instruments: List of source names
    :return: Performance metric dictionary, list with each element describing one dataset sample's results
    '''
    perfs = list()
    perfs_comp = list()
    perfs_solo = list()
    model.eval()
    with torch.no_grad():
        # for example in dataset:
        for example_num, example in enumerate(dataset):
            # if example_num > 1: #__gbastas_
            #     break                
            track_name = example["mix"].split('/')[-2]            
            print("Evaluating " + example["mix"])

            # Load source references in their original sr and channel number
            target_sources = np.stack([data.utils.load(example[instrument], sr=None, mono=False)[0].T for instrument in instruments])




            # Predict using mixture
            pred_sources = predict_song(args, example["mix"], model)
            pred_sources = np.stack([pred_sources[key].T for key in instruments])



            ### ___________ ###
            n = min(pred_sources.shape[1], target_sources.shape[1])
            pred_sources = pred_sources[:,:n]
            target_sources = target_sources[:,:n]
            ### ___________ ###

            # Evaluate
            SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(target_sources, pred_sources)


            # # Convert target_sources to torch tensors and move them to GPU
            # target_sources_tensor = torch.tensor(target_sources).float()
            # pred_sources_tensor = torch.tensor(pred_sources).float()            
            # if args.cuda:
            #     target_sources_tensor = target_sources_tensor.cuda()
            #     pred_sources_tensor = pred_sources_tensor.cuda()            
                
            # Evaluate SI-SDR
            si_sdr_metric = ScaleInvariantSignalDistortionRatio()

            si_sdr_values = []
            for i in range(len(instruments)):
                si_sdr_value = si_sdr_metric(
                    torch.tensor(pred_sources[i]),
                    torch.tensor(target_sources[i])
                ).item()
                si_sdr_values.append(si_sdr_value)


            song = {}
            for idx, name in enumerate(instruments):
                song[name] = {"SDR" : SDR[idx], "ISR" : ISR[idx], "SIR" : SIR[idx], "SAR" : SAR[idx], "SI-SDR": si_sdr_values[idx]}
                # plot with metrics
                plot_waveforms(
                    ref=target_sources[idx],
                    est=pred_sources[idx],
                    sr=args.sr,
                    out_dir="../img_insights",
                    track_name=track_name,
                    instr_name=name,
                    sdr=SDR[idx],
                    si_sdr=si_sdr_values[idx]
                )            

            # song = {}
            # for idx, name in enumerate(instruments):
            #     song[name] = {"SDR" : SDR[idx], "ISR" : ISR[idx], "SIR" : SIR[idx], "SAR" : SAR[idx], "SI-SDR": si_sdr_values[idx]}
            perfs.append(song)
            # print('song', song)

            if '_comp' in example['mix']:
                # print('found comp')
                perfs_comp.append(song)

            if '_solo' in example['mix']:
                # print('found solo')
                perfs_solo.append(song)

    return perfs, perfs_comp, perfs_solo


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