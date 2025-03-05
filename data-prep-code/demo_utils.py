import sys
from pathlib import Path
import glob
import librosa
# import crepe
import matplotlib.pyplot as plt 
import torch
import numpy as np

sys.path.append('./onset')
from model import TCN
from torch.autograd import Variable
import librosa

sys.path.append('./src')
from track_class import Tablature, TabElement, Annotations #, TrackInstance
from constants_parser import Constants
from helper import printProgressBar
from madmom.features.onsets import peak_picking, CNNOnsetProcessor, OnsetPeakPickingProcessor, RNNOnsetProcessor
from madmom.audio.filters import hz2midi
from inharmonic_Analysis import NoteInstance #, iterative_compute_of_partials_and_betas, compute_beta_with_regression
from Inharmonic_Detector import InharmonicDetector
import utils
import os
import jams

import warnings
warnings.filterwarnings("ignore") 

def train(constants, StrBetaObj):

    for string in range(len(constants.tuning)):
        printProgressBar(string, 5, decimals=0, length=50)
        train_single_string(constants, StrBetaObj, string)		

    StrBetaObj.list_to_medians() # betas_list_array to betas_array
    StrBetaObj.set_limits(constants)
    print('Beta estimations:')
    print([round(beta_list[0]*1000, 6) for beta_list in StrBetaObj.betas_array])
    return StrBetaObj


def train_single_string(constants, StrBetaObj, string):
    open_midi = constants.tuning[string]
    # for train_fret in constants.train_frets: # NOTE: don't delte this line. it means sth: the correct way!
    train_fret = 0 # or constants.train_frets[0]
    midi_train = open_midi + train_fret
    print(constants.training_path + str(string) + "*.wav")
    path_to_train_data = os.path.join(constants.training_path, str(string)+"*.wav")
    print(path_to_train_data)
    note_filepath = glob.glob(path_to_train_data)[0]
    print('Training on ', note_filepath)
    note_audio, sr = librosa.load(note_filepath, sr=constants.sampling_rate)

    # Onset Detection && Audio Corpping
    onsets = get_onsets(note_filepath, constants,  0.95)
    try: # Sometimes onsets are tricky to be found correctly in very small recs
        onset_time = onsets[0]
        print(note_filepath, 'otime:', onset_time, '(s)')
    
        PLUS=0.2
    
        # TODO: check onset time PLUS
        start = int(round((onset_time+PLUS)*(constants.sampling_rate)))
        end = int(round((onset_time+PLUS+constants.crop_win) * (constants.sampling_rate)))
    

        if len(note_audio) - start > 44100: # > 1 sec
            note_audio = note_audio[start:end]
        else:
            onset_time=0

    except IndexError:
        print("[MyWarning] No onset found for training sample!")
        onset_time=0

    note_instance = init_note_instance(note_audio, midi_train, string, onset_time, constants)         
    StrBetaObj.add_to_list(note_instance)

    return StrBetaObj


def init_note_instance(instance_audio, midi_note, string, onset_time,  constants : Constants):
    fundamental = librosa.midi_to_hz(midi_note)
    note_instance = NoteInstance(-1, onset_time, instance_audio, constants.sampling_rate, constants, Training = True) # recompute fundamental
    note_instance.string = string
    note_instance.fret = midi_note - constants.tuning[note_instance.string]
    return note_instance  


def predictTabSingleNote(tab_element : TabElement, constants : Constants, StrBetaObj, annos_f0=None): # adapted from main code by removing 'annotations' arguments
    note_instance = NoteInstance(tab_element.id, tab_element.onset, tab_element.note_audio, constants.sampling_rate, constants, annos_f0=annos_f0) 
    InharmonicDetectorObj = InharmonicDetector(note_instance, StrBetaObj)

    if constants.detector == 'barbancho':
        InharmonicDetectorObj.DetectStringBarbancho(constants.betafunc, constants)
    elif constants.detector == 'custom':
        InharmonicDetectorObj.DetectString( constants.betafunc, constants)
    else:
        print('[MyError] This detector name is not valid:', constants.detector)
        exit(1)

    tab_element.string = note_instance.string
    tab_element.fundamental = note_instance.fundamental

    if tab_element.string != 6: # 6 marks inconclusive
        tab_element.fret = utils.hz_to_midi(note_instance.fundamental) - constants.tuning[note_instance.string]
    else: # inconclusive
        tab_element.fret = -1

    return tab_element, note_instance


def predictTab(tablature : Tablature, constants : Constants, StrBetaObj, annos_pitches=None): # adapted from main code by removing 'annotations' arguments
    def close_event(): # https://stackoverflow.com/questions/30364770/how-to-set-timeout-to-pyplot-show-in-matplotlib
        plt.close() #timer calls this function after 3 seconds and closes the window 

    """Inharmonic prediction of tablature """
    Strings, Frets, Freqs = [], [], []
    for i, tab_element in enumerate(tablature.tabList):
  
        try:
            tab_element, note_instance = predictTabSingleNote(tab_element, constants, StrBetaObj, annos_f0=annos_pitches[i])
        except Exception as e:
            print("[MyWarning] Couldn't find any comb for frequency ", tab_element.fundamental)
            continue 
        
        Strings.append(tab_element.string)
        Frets.append(tab_element.fret)
        Freqs.append(note_instance.fundamental)
        # self.onset = onset
        # self.string = string
        # self.fundamental = fundamental
        # tablature.tabList[i].string = tab_element.string



        # Plot DFT and partial deviation
        # print('BBBBBBBBBB', constants.plot, type(constants.plot))
        if constants.plot==True and constants.detector == 'custom':
            # print('BBBBBABBBB', constants.plot)
            fig = plt.figure(figsize=(15, 10))
            timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
            timer.add_callback(close_event)
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            #  TODO: fix lim
            peak_freqs = [partial.partial_Hz for partial in note_instance.partials]
            peaks_idx = [partial.peak_idx for partial in note_instance.partials]
            
            # TODO: deal with annos_instance 
            # note_instance.plot_partial_deviations(lim=30, res=note_instance.abc, ax=ax1, note_instance=note_instance, annos_string=annos_instance.string, tab_element=tab_element) #, peaks_idx=Peaks_Idx)
            note_instance.plot_partial_deviations(lim=30, res=note_instance.abc, ax=ax1, note_instance=note_instance, annos_string=None, tab_instance=tab_element) #, peaks_idx=Peaks_Idx)
            note_instance.plot_DFT(peak_freqs, peaks_idx, lim=30, ax=ax2)   

            plt.show()



    return Strings, Frets, Freqs


def read_tablature_from_GuitarSet(jam_name, constants):
    """function to read a jam file and return the annotations needed"""
    string = 0
    with open(jam_name) as fp:
        try:
            jam = jams.load(fp)
        except:
            print('failed again!!!!!!!!!')
            
    tups = []
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    for string_tran in annos:
        for note in string_tran:
            # print('note', note)
            # aaa
            onset = note[0]
            duration = note[1]
            midi_note = note[2]
            fundamental = librosa.midi_to_hz(midi_note)
            # fret = int(round(midi_note - constants.tuning[string]))
            tups.append((onset, duration, fundamental, string))
        string += 1
    tups.sort(key=lambda x: x[0]) # sort by onset time
    onsets, durations, fundamentals, strings = zip(*tups)
    return Annotations(onsets, durations, fundamentals, strings, constants)


def get_onsets(wav_filepath, constants, th=0.5):
    hop=512
    w_size=1764
    nhid=150
    levels=4
    ksize=3
    dropout=0.25

    audio_data_mix, sr = librosa.load(wav_filepath, sr=None)
    print('Feature Exctraction')
    audio_feats = librosa.feature.melspectrogram(y=audio_data_mix, sr=sr, n_mels=40, n_fft=w_size, hop_length=hop).T
    audio_feats = torch.Tensor(audio_feats.astype(np.float64))
    n_audio_channels = [nhid] * levels # e.g. [150] * 4

    print('Onset 	Model Running...')
    model = TCN(40, 2, n_audio_channels, ksize, dropout=dropout, dilations=True)
    # print('audio_feats', audio_feats.size())
    if len(constants.tuning)==6: # i.e. if guitar
        model_name = "./models/TCN_Audio_0.pt"
    # else:
    #     # model_name = "./models/TCN_Audio_0.pt"
    #     model_name = "./models/TCN_ViolStrings_Audio_0.pt"

    model = torch.load(model_name, map_location=torch.device('cpu'))
    with torch.no_grad():
        x = Variable(audio_feats, requires_grad=True) # _greg_
        output = model(x.unsqueeze(0))	

    print('Onset Model Results ready!!')
    output = output.squeeze(0).cpu().detach()
    if len(constants.tuning)==6: # i.e. if guitar
        oframes = peak_picking(activations=output[:,0].numpy(), threshold=th, pre_max=2, post_max=2) # madmom method
    # else:
    #     oframes = peak_picking(activations=output[:,0].numpy(), threshold=0.98, pre_max=8, post_max=8) # madmom method
    predicted_times = librosa.core.frames_to_time(oframes, sr=sr, hop_length=hop) ## ?? why not w_size

    # proc = OnsetPeakPickingProcessor(threshold=0.5) 
    # act = CNNOnsetProcessor()(wav_filepath)
    # # act = RNNOnsetProcessor()(wav_filepath)
    # predicted_times = proc(act)

    # else:
    # if len(constants.tuning)!=6: # i.e. if cello
    #     proc = OnsetPeakPickingProcessor(threshold=0.4) 
    #     act = CNNOnsetProcessor()(wav_filepath)
    #     act2 = RNNOnsetProcessor()(wav_filepath)
    #     # predicted_times = proc(act)
    #     predicted_times_madom = proc(act)
    #     predicted_times_madom2 = proc(act2)
    #     merged_array = np.concatenate((predicted_times_madom, predicted_times_madom2))
    #     predicted_times = np.sort(merged_array)
        # merged_array = np.concatenate((predicted_times, predicted_times_madom))



    return predicted_times

