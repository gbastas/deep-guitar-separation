import numpy as np
import yaml
import pretty_midi
import os
import glob
import tqdm
import sys
from scipy.io import wavfile
import librosa
import math
from multiprocessing import Pool
from itertools import repeat
import argparse
import shutil

def pitch_to_nfrets(pitch, string_name):
    string_midi_pitches = [40, 45, 50, 55, 59, 64]
    base_pitch_key = {"E string": 40,
                      "A string": 45,
                      "D string": 50,
                      "G string": 55,
                      "B string": 59,
                      "e string": 64}

    string_n_key = {"E string": 0,
                    "A string": 1,
                    "D string": 2,
                    "G string": 3,
                    "B string": 4,
                    "e string": 5}
    base_pitch = base_pitch_key[string_name]
    string_n = string_n_key[string_name]
    return pitch - base_pitch, string_n


def process_cqt(data, sr_original, **kwargs):

    down_sampling_rate = kwargs["down_sampling_rate"]
    bins_per_octave = kwargs["bins_per_octave"]
    n_bins = kwargs["n_bins"]
    hop_length = kwargs["hop_length"]

    data = data.astype(float)
    data = librosa.util.normalize(data)
    data = librosa.resample(data, sr_original, down_sampling_rate)
    data = np.abs(librosa.cqt(data,
                              hop_length=hop_length,
                              sr=down_sampling_rate,
                              n_bins=n_bins,
                              bins_per_octave=bins_per_octave))
    return data


def process_mel_spec(data, sr_original, **kwargs):
    down_sampling_rate = kwargs["down_sampling_rate"]
    hop_length = kwargs["hop_length"]
    data = data.astype(float)
    data = librosa.util.normalize(data)
    data = librosa.resample(data, sr_original, down_sampling_rate)
    data = np.abs(librosa.feature.melspectrogram(data,
                                                 sr=down_sampling_rate,
                                                 n_fft=2048,
                                                 hop_length=hop_length))
    return data


def main(midi_filename_list, kwargs):
    note_resolution = kwargs["note_resolution"]
    down_sampling_rate = kwargs["down_sampling_rate"]
    hop_length = kwargs["hop_length"]
    midi_filename_list = midi_filename_list.split("\n")

    if "auto_quantized" in midi_filename_list[0]:
        npz_dir = os.path.join("data", "npz", f"auto_quantized_{note_resolution}")
        audio_dir = os.path.join("data", "wav", "auto_quantized_16")
    elif "original" in midi_filename_list[0]:
        # npz_dir = os.path.join("data", "npz", "original")
        npz_dir = os.path.join("data", "npz_"+args.audio_dir, args.mode)
        # if not os.path.exists(npz_dir):
        os.makedirs(npz_dir, exist_ok=True)
        # audio_dir = os.path.join("GuitarSet", "audio_mono-mic")
        # audio_dir = os.path.join("GuitarSet", "audio","datasep-mic") # __gbastas__
        # audio_dir = os.path.join("GuitarSet", "audio", args.audio_dir) # __gbastas__
        audio_dir = os.path.join("../datasets", args.audio_dir) # __gbastas__
        print('audio_dir', audio_dir)
    else:
        print("error")

    norm_len = kwargs["down_sampling_rate"] / kwargs["hop_length"]

    for midi_filename in midi_filename_list:
        # TODO: more like this below
        if args.mode=='standard':
            suffices=['mixture']
        elif args.mode=='7-target':
            suffices = ['mixture', 'E', 'A', 'G', 'D', 'B', 'e']
        elif args.mode=='7-pred':
            suffices = ['mixture', 'pred_E', 'pred_A', 'pred_G', 'pred_D', 'pred_B', 'pred_e']
        else:
            print("MyError: Wrong --mode argument!")
            exit(0)
            
        all_cqt_feats = []
        all_log_cqt_feats = []
        all_mel_spec_feats = []
        
        # ######################################################################################
        for suffix in suffices:
            audio_filename = os.path.join(
                audio_dir, os.path.split(midi_filename)[1][:-4] + '/'+ suffix + ".wav")
        ##########################################################################################
        
            # audio_filename = os.path.join(
            #     audio_dir, os.path.split(midi_filename)[1][:-4] + "_mic.wav")
            sr_original, audio_file = wavfile.read(audio_filename)
            cqt = process_cqt(audio_file, sr_original, **kwargs)
            log_cqt = librosa.amplitude_to_db(np.abs(cqt))
            mel_spec = process_mel_spec(audio_file, sr_original, **kwargs)
            cqt = cqt.T
            log_cqt = log_cqt.T
            mel_spec = mel_spec.T

            midi_file = pretty_midi.PrettyMIDI(midi_filename)

            tempo = float(midi_filename.split('-')[1])
            note_dur = 60 / tempo / note_resolution * 4

            len_in_notes = int(math.ceil(round(midi_file.get_end_time(
            ) / note_dur) / (note_resolution * 4)) * (note_resolution * 4))

            feature_len = int(len_in_notes * note_dur *
                            (down_sampling_rate / hop_length))
            cqt = cqt[:feature_len]
            mel_spec = mel_spec[:feature_len]

            print('main', audio_filename, suffix, cqt.shape[0], mel_spec.shape[0])
            assert cqt.shape[0] == mel_spec.shape[0]

            all_cqt_feats.append(cqt)
            all_log_cqt_feats.append(log_cqt)
            all_mel_spec_feats.append(mel_spec)

        npz_path = os.path.join(npz_dir, os.path.split(midi_filename)[1][:-4])


        # Convert the lists of arrays into 3D arrays: (n_string, n_frames, n_freqbands)
        cqt = np.stack(all_cqt_feats, axis=0)
        log_cqt = np.stack(all_log_cqt_feats, axis=0)
        mel_spec = np.stack(all_mel_spec_feats, axis=0)



        # process note-wise tab
        tab = np.zeros((len_in_notes, 6, 21))
        tab[:, :, 20] = 1
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                frets, string_n = pitch_to_nfrets(note.pitch, string_name)
                tab[int(round(note.start / note_dur)) : int(round(note.end / note_dur)), string_n, 20] = 0
                tab[int(round(note.start / note_dur)) : int(round(note.end / note_dur)), string_n, frets] = 1

        # process note-wise tab onset
        tab_onset = np.zeros((len_in_notes, 6, 21))
        tab_onset[:, :, 20] = 1
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                frets, string_n = pitch_to_nfrets(note.pitch, string_name)
                if int(round(note.start / note_dur)) >= len_in_notes:
                    break
                tab_onset[int(round(note.start / note_dur)), string_n, 20] = 0
                tab_onset[int(round(note.start / note_dur)),
                        string_n, frets] = 1

        # process frame-wise tab
        frame_tab = np.zeros((feature_len, 6, 21))
        frame_tab[:, :, 20] = 1
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                frets, string_n = pitch_to_nfrets(note.pitch, string_name)
                frame_tab[int(round(note.start * norm_len)) : int(round(note.end * norm_len)), string_n, 20] = 0
                frame_tab[int(round(note.start * norm_len)) : int(round(note.end * norm_len)), string_n, frets] = 1

        # process frame-wise tab onset
        frame_tab_onset = np.zeros((feature_len, 6, 21))
        frame_tab_onset[:, :, 20] = 1
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                frets, string_n = pitch_to_nfrets(note.pitch, string_name)
                frame_tab_onset[int(
                    round(note.start * norm_len)), string_n, 20] = 0
                frame_tab_onset[int(round(note.start * norm_len)),
                                string_n, frets] = 1

        # process note-wise F0
        F0 = np.zeros((len_in_notes, 44))
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                pitch = note.pitch - 40
                F0[int(round(note.start / note_dur)) : int(round(note.end / note_dur)), pitch] = 1

        # process note-wise F0 onset
        F0_onset = np.zeros((len_in_notes, 44))
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                if int(round(note.start / note_dur)) >= len_in_notes:
                    break
                pitch = note.pitch - 40
                F0_onset[int(round(note.start / note_dur)), pitch] = 1

        # process frame-wise F0
        frame_F0 = np.zeros((feature_len, 44))
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                pitch = note.pitch - 40
                frame_F0[int(round(note.start * norm_len)) : int(round(note.end * norm_len)), pitch] = 1

        # process frame-wise F0 onset
        frame_F0_onset = np.zeros((feature_len, 44))
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                pitch = note.pitch - 40
                frame_F0_onset[int(round(note.start * norm_len)), pitch] = 1

        """
        # 0~19 : number of frets
        # 20 : not played

        """
        np.savez_compressed(npz_path,
                            cqt=cqt,
                            log_cqt=log_cqt,
                            mel_spec=mel_spec,
                            tab=tab,
                            tab_onset=tab_onset,
                            frame_tab=frame_tab,
                            frame_tab_onset=frame_tab_onset,
                            F0=F0,
                            F0_onset=F0_onset,
                            frame_F0=frame_F0,
                            frame_F0_onset=frame_F0_onset,
                            tempo=tempo,
                            len_in_notes=len_in_notes)
        split_save(npz_path, cqt, mel_spec, tab, tab_onset, frame_tab, frame_tab_onset, F0, F0_onset,
                frame_F0, frame_F0_onset,  tempo, note_resolution)


def split_save(npz_path, cqt, mel_spec, tab, tab_onset, frame_tab, frame_tab_onset, F0, F0_onset, frame_F0, frame_F0_onset, tempo, note_resolution):
    # feature_len = cqt.shape[0]
    feature_len = cqt.shape[1]

    note_len = tab.shape[0]
    note_len_4bars = int(round(float(note_len) / float(note_resolution * 4)))
    feature_len_4bars = int(round(feature_len / note_len_4bars))
    pad_len = 0
    split_npz_path = os.path.join(os.path.split(npz_path)[0], "split")

    if not(os.path.exists(split_npz_path)):
        os.makedirs(split_npz_path)

    for n_4bars in range(note_len_4bars):
        split_npz_filename = os.path.join(
            split_npz_path, (os.path.split(npz_path)[1] + f"_0{n_4bars}"))

        split_tab = tab[(note_resolution * 4) *
                        n_4bars: (note_resolution * 4) * (n_4bars+1)]
        split_tab_onset = tab_onset[(note_resolution * 4) *
                                    n_4bars: (note_resolution * 4) * (n_4bars+1)]
        split_F0 = F0[(note_resolution * 4) *
                      n_4bars: (note_resolution * 4) * (n_4bars+1)]
        split_F0_onset = F0_onset[(note_resolution * 4)
                                  * n_4bars: (note_resolution * 4) * (n_4bars+1)]

        cqt_zero_pad = np.zeros((cqt.shape[0], pad_len, cqt.shape[2]))
        mel_spec_zero_pad = np.zeros((mel_spec.shape[0], pad_len, mel_spec.shape[2]))        
        # cqt_zero_pad = np.zeros((pad_len, cqt.shape[1]))
        # mel_spec_zero_pad = np.zeros((pad_len, mel_spec.shape[1]))
        
        frame_F0_zero_pad = np.zeros((pad_len, frame_F0.shape[1]))
        frame_tab_zero_pad = np.zeros((pad_len, 6, 21))

        if n_4bars == 0:
            split_cqt = np.append(
                cqt_zero_pad, cqt[:, feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len,:], axis=1)
            split_mel_spec = np.append(
                mel_spec_zero_pad, mel_spec[:, feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len,:], axis=1)
            split_frame_tab = np.append(
                frame_tab_zero_pad, frame_tab[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len], axis=0)
            split_frame_tab_onset = np.append(
                frame_tab_zero_pad, frame_tab_onset[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len], axis=0)
            split_frame_F0 = np.append(
                frame_F0_zero_pad, frame_F0[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len], axis=0)
            split_frame_F0_onset = np.append(
                frame_F0_zero_pad, frame_F0_onset[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len], axis=0)
        elif n_4bars == note_len_4bars-1:
            split_cqt = np.append(
                cqt[:, feature_len_4bars * n_4bars - pad_len: feature_len_4bars * (n_4bars+1),:], cqt_zero_pad, axis=1)
            split_mel_spec = np.append(
                mel_spec[:, feature_len_4bars * n_4bars - pad_len: feature_len_4bars * (n_4bars+1),:], mel_spec_zero_pad, axis=1)
            split_frame_tab = np.append(
                frame_tab[feature_len_4bars * n_4bars - pad_len: feature_len_4bars *
                          (n_4bars+1)], frame_tab_zero_pad, axis=0)
            split_frame_tab_onset = np.append(
                frame_tab_onset[feature_len_4bars * n_4bars - pad_len: feature_len_4bars *
                                (n_4bars+1)], frame_tab_zero_pad, axis=0)
            split_frame_F0 = np.append(
                frame_F0[feature_len_4bars * n_4bars - pad_len: feature_len_4bars *
                         (n_4bars+1)], frame_F0_zero_pad, axis=0)
            split_frame_F0_onset = np.append(
                frame_F0_onset[feature_len_4bars * n_4bars - pad_len: feature_len_4bars *
                               (n_4bars+1)], frame_F0_zero_pad, axis=0)
        else:
            split_cqt = cqt[:, feature_len_4bars * n_4bars -
                            pad_len: feature_len_4bars * (n_4bars+1) + pad_len,:]
            split_mel_spec = mel_spec[:, feature_len_4bars * n_4bars -
                                      pad_len: feature_len_4bars * (n_4bars+1) + pad_len,:]
            split_frame_tab = frame_tab[feature_len_4bars * n_4bars -
                                        pad_len: feature_len_4bars * (n_4bars+1) + pad_len]
            split_frame_tab_onset = frame_tab_onset[feature_len_4bars * n_4bars -
                                                    pad_len: feature_len_4bars * (n_4bars+1) + pad_len]
            split_frame_F0 = frame_F0[feature_len_4bars * n_4bars -
                                      pad_len: feature_len_4bars * (n_4bars+1) + pad_len]
            split_frame_F0_onset = frame_F0_onset[feature_len_4bars * n_4bars -
                                                  pad_len: feature_len_4bars * (n_4bars+1) + pad_len]


        assert split_cqt.shape[1] == split_mel_spec.shape[1] == split_frame_F0.shape[0] == split_frame_F0_onset.shape[0] == split_frame_tab_onset.shape[0] == split_frame_tab_onset.shape[0]

        np.savez_compressed(split_npz_filename,
                            cqt=split_cqt,
                            log_cqt=librosa.amplitude_to_db(np.abs(split_cqt)),
                            mel_spec=split_mel_spec,
                            tab=split_tab,
                            tab_onset=split_tab_onset,
                            frame_tab=split_frame_tab,
                            frame_tab_onset=split_frame_tab_onset,
                            F0=split_F0,
                            F0_onset=split_F0_onset,
                            frame_F0=split_frame_F0,
                            frame_F0_onset=split_frame_F0_onset,
                            tempo=tempo,
                            len_in_notes=(note_resolution * 4))
    return


def preprocess_input_dir(input_dir):
    """
    Moves all subdirectories from 'train' and 'test' into input_dir
    and removes the empty 'train' and 'test' subdirectories.
    """
    subdirs = ["train", "test"]
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                if os.path.isdir(item_path):  # Check if it's a directory
                    shutil.move(item_path, input_dir)  # Move the subdirectory
            # Remove the now-empty directory
            os.rmdir(subdir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    with open("src/config.yaml") as f:
        obj = yaml.safe_load(f)
        note_resolution = obj["note_resolution"]
        down_sampling_rate = obj["down_sampling_rate"]
        bins_per_octave = obj["bins_per_octave"]
        n_bins = obj["cqt_n_bins"]
        hop_length = obj["hop_length"]
        n_cores = obj["n_cores"]


    # parser.add_argument("-d", "--npz_featdir", type=str, required=True,
    #                     help="(e.g. npz_datasep-mic-preds)'")

    parser.add_argument("-m", "--mode", type=str, required=True,
                        help="'standard' or '7-target' or '7-pred'")

    parser.add_argument("-a", "--audio_dir", type=str, required=True,
                        help="e.g. datasep-mic-preds")
    


    args = parser.parse_args()

    kwargs = {
        "note_resolution": note_resolution,
        "down_sampling_rate": down_sampling_rate,
        "bins_per_octave": bins_per_octave,
        "n_bins": n_bins,
        "hop_length": hop_length
    }


    preprocess_input_dir(os.path.join("../datasets", args.audio_dir))
    
    midi_dir = os.path.join("data", "midi")
    # auto quantized
    #npz_dir = os.path.join("data", "npz", f"auto_quantized_{note_resolution}")
    #midi_file_path = os.path.join(midi_dir, f"auto_quantized_{note_resolution}", "*")

    # original
    # npz_dir = os.path.join("data", 'npz_'+args.audio_dir, args.mode)
    midi_file_path = os.path.join(midi_dir, "original", "*")
    print('midi_file_path', midi_file_path)


    midi_filename_list = glob.glob(midi_file_path)
    midi_filename_list.sort()


    # paralell process
    p = Pool(n_cores)
    p.starmap(main, zip(midi_filename_list, repeat(kwargs)))
    p.close()  # or p.terminate()
    p.join()

    # check for missing file
    for midi_filename in midi_filename_list:
        name = os.path.split(midi_filename)[1][:-4]
        print(name)
        if not os.path.exists("data/npz/original/" + name + ".npz"):
            print(f"{name} does not exist!")
