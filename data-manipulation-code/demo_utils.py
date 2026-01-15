import sys
import numpy as np

import librosa

sys.path.append('./src')
from track_class import  Tablature #, Annotations #, TrackInstance

import jams

import warnings
warnings.filterwarnings("ignore") 

# TODO: define constants
constants = {'tuning': [40, 45, 50, 55, 59, 64],
             'no_of_frets': 23,
             'sampling_rate': 44100,
             'crop_win': 3
}

def read_tablature_from_GuitarSet(jam_name, constants, audio=[]):
    """function to read a jam file and return the annotations needed"""
    string = 0
    try:
        with open(jam_name) as fp:
            jam = jams.load(fp)
    except Exception as e:
        print(f"Error loading {jam_name}: {e}")

            
    tups = []
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    for string_tran in annos:
        for note in string_tran:
            # print('note', note)
            onset = note[0]
            duration = note[1]
            midi_note = note[2]
            fundamental = librosa.midi_to_hz(midi_note)
            tups.append((onset, duration, fundamental, string))
        string += 1
    tups.sort(key=lambda x: x[0]) # sort by onset time
    onsets, durations, fundamentals, strings = zip(*tups)
    return Tablature(onsets, durations, audio, constants=constants, strings=strings, fundamentals=fundamentals)
    

class Tablature():
    def __init__(self, onsets, durations, audio, constants : Constants, strings = None, fundamentals = None):
        self.audio = audio
        self.sampling_rate = constants.sampling_rate
        self.constants = constants
        self.tabList = []
        note_audio=[]
        for i, onset in enumerate(onsets):
            if audio != [] and i+1 < len(onsets):
                start = int(round((onset)*(constants.sampling_rate)))
                endtime =  constants.crop_win if onsets[i+1] - onsets[i] > constants.crop_win else min(onsets[i+1] - onsets[i], constants.crop_win)
                end = int(round((onset+endtime)*(constants.sampling_rate)))
                note_audio = audio[start:end]
            if strings:
                    self.tabList.append(TabElement(i, onset, onsets[i]+durations[i], strings[i], note_audio, constants, fundamentals[i])) 
            else:
                self.tabList.append(TabElement(i, onset, onsets[i]+durations[i], 6, note_audio, constants)) # 6 is for initialization 


    def __getitem__(self, item):
        '''added so crossover functions from deap can be incorporated easily'''
        return self.tabList[item]

    def __len__(self):
        return len(self.tabList)


class TabElement():
    def __init__(self, id, onset, offset, string, note_audio, constants : Constants, fundamental=None):
        self.id = id
        self.onset = onset
        self.offset = offset
        self.string = string
        self.fundamental = fundamental
        self.fret = -1
        if self.string in list(range(0,6)) and fundamental:
            self.fret = utils.hz_to_midi(fundamental) - constants.tuning[self.string]
        self.note_audio = note_audio