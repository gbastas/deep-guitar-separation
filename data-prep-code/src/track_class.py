from constants_parser import Constants
import utils

class Tablature():
    def __init__(self, onsets, durations, audio, constants : Constants, strings = None, fundamentals = None, annotation = False):
        self.audio = audio
        self.sampling_rate = constants.sampling_rate
        self.constants = constants
        self.tabList = []
        note_audio=[]
        for i, onset in enumerate(onsets):
            if annotation == False and i+1 < len(onsets):
                start = int(round((onset)*(constants.sampling_rate)))
                endtime = onsets[i]+durations[i]
                offset = endtime
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

class Annotations():
    def __init__(self, onsets, durations, fundamentals, strings, constants : Constants):
        self.tablature = Tablature(onsets, durations, audio=[], constants=constants, strings=strings, fundamentals=fundamentals, annotation = True)







