import sys
import numpy as np

import librosa

sys.path.append('./src')
from track_class import  Annotations #, TrackInstance

import jams

import warnings
warnings.filterwarnings("ignore") 


def read_tablature_from_GuitarSet(jam_name, constants):
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

