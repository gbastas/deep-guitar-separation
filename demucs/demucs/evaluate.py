# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

from concurrent import futures
import logging

from dora.log import LogProgress
import numpy as np
import musdb
import museval
import torch as th

from .apply import apply_model
from .audio import convert_audio, save_audio
from . import distrib
from .utils import DummyPoolExecutor


logger = logging.getLogger(__name__)

import librosa
import os

import torchmetrics
import torch

# class Track:
#     def __init__(self, name, path, sample_rate):
#         self.name = name
#         self.path = path
#         self.sample_rate = sample_rate
#         self.audio, self.targets = self._load_audio()
#     # def _load_audio(self):
#     #     mix_path = os.path.join(self.path, 'mixture.wav')
#     #     audio, _ = librosa.load(mix_path, sr=self.sample_rate, mono=False)
#     #     return audio

#     def _load_audio(self):
#         audio_files = {}
#         targets = {}
        
#         for stem_name in os.listdir(self.path):
#             if stem_name.endswith('.wav'):
#                 stem_path = os.path.join(self.path, stem_name)
#                 audio, _ = librosa.load(stem_path, sr=self.sample_rate, mono=False)
#                 stem_name_without_ext = os.path.splitext(stem_name)[0]
#                 audio_files[stem_name_without_ext] = audio
                
#                 if stem_name_without_ext != 'mixture':
#                     targets[stem_name_without_ext] = audio

#         return audio_files.get('mixture', None), targets
    # def _load_audio(self):
    #     audio_files = {}
    #     for stem_name in os.listdir(self.path):
    #         if stem_name.endswith('.wav'):
    #             stem_path = os.path.join(self.path, stem_name)
    #             audio, _ = librosa.load(stem_path, sr=self.sample_rate, mono=False)
    #             stem_name_without_ext = os.path.splitext(stem_name)[0]
    #             audio_files[stem_name_without_ext] = audio
    #     return audio_files


class Stem:
    def __init__(self, path, sample_rate):
        self.path = path
        self.sample_rate = sample_rate
        self.audio = self._load_audio()

    def _load_audio(self):
        audio, _ = librosa.load(self.path, sr=self.sample_rate, mono=False)
        return audio

class Track:
    def __init__(self, name, path, sample_rate):
        self.name = name
        self.path = path
        self.sample_rate = sample_rate
        self.audio, self.targets = self._load_audio()
        
    def _load_audio(self):
        audio_files = {}
        targets = {}

        for stem_name in os.listdir(self.path):
            if stem_name.endswith('.wav'):
                stem_path = os.path.join(self.path, stem_name)
                stem_name_without_ext = os.path.splitext(stem_name)[0]
                
                # Creating a Stem object which has an 'audio' attribute.
                stem_obj = Stem(path=stem_path, sample_rate=self.sample_rate)

                if stem_name_without_ext == 'mixture':
                    mixture = stem_obj
                else:
                    targets[stem_name_without_ext] = stem_obj
        
        return mixture.audio, targets

class MyDataset:
    def __init__(self, root_dir, subset='test', sample_rate=44100):
        self.root_dir = root_dir
        self.subset = subset
        self.sample_rate = sample_rate
        self.tracks = self._load_tracks()

    def _load_tracks(self):
        subset_path = os.path.join(self.root_dir, self.subset)
        tracks = []
        
        for track_name in os.listdir(subset_path):
            track_path = os.path.join(subset_path, track_name)
            
            if os.path.isdir(track_path):
                track = Track(name=track_name, path=track_path, sample_rate=self.sample_rate)
                tracks.append(track)
                
        return tracks

    def __len__(self):
        return len(self.tracks)


def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = th.sum(th.square(references), dim=(2, 3))
    den = th.sum(th.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * th.log10(num / den)
    return scores


def eval_track(references, estimates, win, hop, compute_sdr=True):
    si_sdr_metric = torchmetrics.audio.ScaleInvariantSignalDistortionRatio()

    references = references.transpose(1, 2).double() # (6, 976691, 1) 
    estimates = estimates.transpose(1, 2).double()
    new_scores = new_sdr(references.cpu()[None], estimates.cpu()[None])[0]

    if not compute_sdr:

        return None, new_scores
    else:
        # Compute SI-SDR for each source in batch

        references = references.numpy()
        estimates = estimates.numpy()
        
        print('win', win) #44100
        print('hop',hop) #44100

        SDR, ISR, SIR, SAR, _  = museval.metrics.bss_eval(references, estimates) #, window=win, hop=hop)th
        sdr_scores = (SDR, ISR, SIR, SAR)

        # sdr_scores = museval.metrics.bss_eval(
        #     references, estimates,
        #     compute_permutation=False,
        #     window=win,
        #     hop=hop,
        #     framewise_filters=False,
        #     bsseval_sources_version=False)[:-1]


        print("references.shape:", references.shape) # printing result: (6, 976691, 1) 

        references = references.transpose(0, 2, 1)  # Shape: (6, 1, 976691)
        estimates = estimates.transpose(0, 2, 1)   # Shape: (6, 1, 976691)

        # Compute SI-SDR for each source individually
        si_sdr_values = []
        for i in range(references.shape[0]):  
            si_sdr_value = si_sdr_metric(
                torch.tensor(estimates[i]).squeeze(-1),  # Shape: (976691,)
                torch.tensor(references[i]).squeeze(-1)  # Shape: (976691,)
            ).item()
            si_sdr_values.append(si_sdr_value)

        # references = references.transpose(0, 2, 1)  # Shape: (6, 1, 976691)
        # estimates = estimates.transpose(0, 2, 1)   # Shape: (6, 1, 976691)
        # # Remove last dimension for `museval`
        # references = references.squeeze(-1)  # Shape: (6, 976691)
        # estimates = estimates.squeeze(-1)    # Shape: (6, 976691)
        # # Compute SDR scores using `museval`
        # SDR, ISR, SIR, SAR, _  = museval.metrics.bss_eval(references, estimates)
        # sdr_scores = (SDR, ISR, SIR, SAR)


        scores = (*sdr_scores, si_sdr_values)  # Append SI-SDR as a list of values

    return scores, new_scores


def evaluate(solver, compute_sdr=False):
    """
    Evaluate model using museval.
    compute_sdr=False means using only the MDX definition of the SDR, which
    is much faster to evaluate.
    """

    args = solver.args


    output_dir = solver.folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = solver.folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    # we load tracks from the original musdb set
    if args.test.nonhq is None:
        # test_set = musdb.DB(args.dset.musdb, subsets=["test"], is_wav=True)
        test_set = MyDataset(args.dset.musdb, subset='test', sample_rate=44100) # __gb__
    else:
        test_set = musdb.DB(args.test.nonhq, subsets=["test"], is_wav=False)
    src_rate = args.dset.musdb_samplerate


    # NO multithredding!
    eval_device = 'cpu'  # or 'cuda', depending on your preference and hardware
    # eval_device = 'cuda'  # or 'cuda', depending on your preference and hardware

    model = solver.model
    win = int(1. * model.samplerate)
    hop = int(1. * model.samplerate)

    print('test_set', len(test_set))

    indexes = range(len(test_set))
    indexes = LogProgress(logger, indexes, updates=args.misc.num_prints, name='Eval')

    tracks = {}  # assuming this is where results are stored

    for index in indexes:
        track = test_set.tracks[index]
        mix = th.from_numpy(track.audio).t().float()
        if mix.dim() == 1:
            mix = mix[None]
        mix = mix.to(solver.device)
        ref = mix.mean(dim=0)  # mono mixture
        mix = (mix - ref.mean()) / ref.std()
        mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)
        
        estimates = apply_model(model, mix[None],
                                shifts=args.test.shifts, split=args.test.split,
                                overlap=args.test.overlap)[0]
        estimates = estimates * ref.std() + ref.mean()
        estimates = estimates.to(eval_device)

        references = th.stack(
            [th.from_numpy(track.targets[name].audio).t() for name in model.sources])
        if references.dim() == 2:
            references = references[:, None]
        references = references.to(eval_device)
        references = convert_audio(references, src_rate,
                                model.samplerate, model.audio_channels)
        
        if args.test.save:
            folder = solver.folder / "wav" / track.name
            folder.mkdir(exist_ok=True, parents=True)
            for name, estimate in zip(model.sources, estimates):
                save_audio(estimate.cpu(), folder / (name + ".mp3"), model.samplerate)

        # Here we just call `eval_track` directly and store the result
        scores_nsdrs = eval_track(references, estimates, win=win, hop=hop, compute_sdr=compute_sdr)
        
        scores, nsdrs = scores_nsdrs

        tracks[track.name] = {}
        for idx, target in enumerate(model.sources):
            tracks[track.name][target] = {'nsdr': [float(nsdrs[idx])]}
            # print('ZZZZZZZZZZZ', scores)
            if scores is not None:
                (sdr, isr, sir, sar, si_sdr) = scores
                # print('SDR:',  sdr[idx].tolist())
                values = {
                    "SDR": sdr[idx].tolist(),
                    "SIR": sir[idx].tolist(),
                    "ISR": isr[idx].tolist(),
                    "SAR": sar[idx].tolist(),
                    "SI-SDR": si_sdr[idx]#.tolist()  
                }
                tracks[track.name][target].update(values)
                # print('values', values)

        all_tracks = {}
        for src in range(distrib.world_size):
            all_tracks.update(distrib.share(tracks, src))

        # result = {}
        # metric_names = next(iter(all_tracks.values()))[model.sources[0]]
        # for metric_name in metric_names:
        #     avg = 0
        #     avg_of_medians = 0
        #     for source in model.sources:
        #         medians = [
        #             np.nanmedian(all_tracks[track][source][metric_name])
        #             for track in all_tracks.keys()]
        #         mean = np.mean(medians)
        #         median = np.median(medians)
        #         result[metric_name.lower() + "_" + source] = mean
        #         result[metric_name.lower() + "_med" + "_" + source] = median
        #         avg += mean / len(model.sources)
        #         avg_of_medians += median / len(model.sources)
        #     result[metric_name.lower()] = avg
        #     print(metric_name.lower(),':', avg)
        #     result[metric_name.lower() + "_med"] = avg_of_medians
        # return result

        # _gbastas_: mean method
        result = {}
        metric_names = next(iter(all_tracks.values()))[model.sources[0]]
        for metric_name in metric_names:
            avg = 0
            for source in model.sources:
                means = [
                    np.nanmean(all_tracks[track][source][metric_name])
                    for track in all_tracks.keys()
                ]
                overall_mean = np.mean(means)
                result[metric_name.lower() + "_" + source] = overall_mean
                avg += overall_mean / len(model.sources)
            result[metric_name.lower()] = avg
            print(metric_name.lower(), ':', avg)
        return result