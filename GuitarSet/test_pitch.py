# Check this repo for more info: https://gitlab.com/stefanoskoutoupis/fresh-guit-proc.git
# go to track_class.py 


import jams
import librosa
import numpy as np
from madmom.audio.filters import hz2midi
from madmom.evaluation.onsets import onset_evaluation
from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
import crepe
from initialize_workspace import workspace_class, initialize_workspace

from matplotlib import lines as mlines, pyplot as plt
import librosa.display

from scipy.io import wavfile
from tensorflow.keras.utils import to_categorical

import sys
np.set_printoptions(threshold=sys.maxsize)

from Metrics import pitch_precision, pitch_recall, pitch_f_measure


def convert_name(name, dataset = None, mode = 'to_wav'):
    '''modes --> {to_wav, to_jams}
    dataset --> {hex_cln, mic, mix}'''
    if dataset == None:
        if '.wav' in name:
            return name[:-4] + '.jams'
        elif '.jams' in name:
            return name[:-5] + '.wav'
        else:
            raise NameError('Not proper track extension neither wav or jams')
    elif mode == 'to_wav':
        folder_len = name.rfind('/')
        temp_name = name[folder_len+1:-5]
        name = (workspace.workspace_folder+'/' +
                            dataset + '/' + temp_name + 
                                '_' + dataset + '.wav')

    elif mode == 'to_jams':
        folder_len = name.rfind('/')
        temp_name = name[folder_len+1:len(dataset)-4]
        name = workspace.annotations_folder + temp_name + '.jams'
    else:
        raise NameError('Not Proper Mode choose to_wav or to_jams')
    if mode == 'to_jams' and dataset not in name:
        return name
    elif mode == 'to_wav' and dataset in name:
        return name
    else:
        raise NameError('Validity check failed name to be returned is {}'.format(name))
    return name


class TrackInstance():
    def __init__(self, jam_name, dataset): #, dataset
        self.jam_name = jam_name
        self.track_name = convert_name(jam_name, dataset, 'to_wav')
        self.true_tablature = self.read_tablature_from_jams()
        print(self.track_name)
        audio, sr = librosa.load(self.track_name, sr=44100, mono=False)
        self.audio = audio
        self.sr = sr
        self.predicted_tablature = None
        self.rnn_tablature = None
        self.predicted_strings = None
        self.highest_fret = 19
        self.num_classes = self.highest_fret + 2

    def predict_tablature(self, mode = 'FromCNN'):
        '''mode --> {From_Annos, FromCNN} first reads 
        from annotations onset-pitch second estimates'''
        strings = []
        if mode == 'FromCNN':
            onsets, midi_notes = self.predict_notes_onsets()
            self.temp_tablature = Tablature(onsets, midi_notes, [])

            
    def read_tablature_from_jams(self):
        str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
        s = 0
        jam = jams.load(self.jam_name)
        tablature = []
        onsets = []
        midi_notes = []
        strings = []
        
        annos = jam.search(namespace='note_midi')
        if len(annos) == 0:
            annos = jam.search(namespace='pitch_midi')
        for string_tran in annos:
            for note in string_tran:
                start_time = note[0]
                end_time = start_time + 0.06
                midi_note = note[2]
                fret = int(round(midi_note - str_midi_dict[s]))
                string = s
                tablature.append([round(midi_note),string,fret,start_time,end_time])
            s += 1
        tablature.sort(key=lambda x: x[3])

        for instance in tablature:
            onsets.append((instance[3], 1))
            midi_notes.append((instance[0], 1))
            strings.append((instance[1], 1))

        return Tablature(onsets, midi_notes, strings)
    

    def predict_pyin(self): # __gbastas__
        f0 = [None,None,None,None,None,None]
        # string_range

        for string_num in range(6):
            string_range=[{'min':'E2', 'max':'B3'}, {'min':'A2', 'max':'E4'}, {'min':'D3', 'max':'A4'}, {'min':'G3', 'max':'D5'}, {'min':'B3', 'max':'G5'}, {'min':'E4', 'max':'B5'}]
            f0[string_num], voiced_flag, voiced_prob = librosa.pyin(self.audio[string_num,:], fmin=librosa.note_to_hz('E2'),fmax=librosa.note_to_hz('E6'),sr=44100) # hop_length=512 TODO: change range according to string
            # f0[string_num], _, _ = librosa.pyin(self.audio[string_num,:], fmin=librosa.note_to_hz(string_range[string_num]['min']),fmax=librosa.note_to_hz(string_range[string_num]['max']), sr=44100) # hop_length=512 TODO: change range according to string
            times = librosa.times_like(f0[string_num], sr=44100)  #hop_length=512
            print('pyin f0 (raw)',len(f0[string_num]))
            print('pyin times',len(times))

            D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio[string_num,:])), ref=np.max) #hop_length=512
            fig, ax = plt.subplots()
            img = librosa.display.specshow(D, x_axis='time', y_axis='log',  sr=44100, ax=ax)
            ax.set(title='pYIN fundamental frequency estimation')
            fig.colorbar(img, ax=ax, format="%+2.f dB")
            ax.plot(times, f0[string_num], label='f0', color='cyan', linewidth=3)
            ax.legend(loc='upper right')
            plt.savefig('visual'+str(string_num)+'.png')   
        return f0, times


    def categorical(self, label): # __gbastas__ copied from tab-cnn repo
        return to_categorical(label, self.num_classes)

    def correct_numbering(self, n): # __gbastas__ copied from tab-cnn repo
        n += 1
        if n < 0 or n > self.highest_fret:
            n = 0
        return n

    def clean_label(self, label): # __gbastas__ copied from tab-cnn repo
        label = [self.correct_numbering(n) for n in label]
        return self.categorical(label)
    
    def clean_labels(self, labels): # __gbastas__ copied from tab-cnn repo
        return np.array([self.clean_label(label) for label in labels])

    def isNaN(self, num):
        return num != num

    def load_rep_and_labels_from_raw_file(self, filename, f0_pred): # __gbastas__ copied from tab-cnn repo and changed it a bit
        string_midi_pitches = [40,45,50,55,59,64]
        file_audio = self.track_name
        file_anno = self.jam_name
        jam = jams.load(file_anno)
        self.sr_original, data = wavfile.read(file_audio)
        self.sr_curr = self.sr_original
        
        frame_indices = range(len(f0_pred[0]))
        times = librosa.frames_to_time(frame_indices, sr = self.sr_curr, hop_length=512)

        # loop over all strings and sample annotations
        labels = []
        preds = []
        for string_num in range(6):
            # Predictions
            tmp = np.array([-1] * len(f0_pred[0]))
            for i in frame_indices:
                if not self.isNaN(f0_pred[string_num][i]):
                    # replace midi pitch values with fret numbers
                    tmp[i] = int(round(hz2midi(f0_pred[string_num][i])) - string_midi_pitches[string_num])
                    if tmp[i]<0:
                        tmp[i]=-1
            preds.append([tmp])

            # Annotations
            anno = jam.annotations["note_midi"][string_num]
            string_label_samples = anno.to_samples(times)
            # replace midi pitch values with fret numbers
            for i in frame_indices:
                if string_label_samples[i] == []:
                    string_label_samples[i] = -1
                else:
                    string_label_samples[i] = int(round(string_label_samples[i][0]) - string_midi_pitches[string_num])
            labels.append([string_label_samples])
            
        labels = np.array(labels)
        labels = np.squeeze(labels)
        labels = np.swapaxes(labels,0,1)
        
        # clean labels
        labels = self.clean_labels(labels)
        print('labels',labels.shape)

        preds = np.array(preds)
        preds = np.squeeze(preds)
        preds = np.swapaxes(preds,0,1)

        # print(preds)

        preds = self.clean_labels(preds) # NOTE: maybe not good
        
        print('preds', preds.shape)

        return labels, preds


class Predictions():
        def __init__(self, tup):
            prediction, confidence = tup
            self.prediction = prediction
            self.confidence = confidence
            return None

class Onset(Predictions):
    def some_func():
        pass

class MidiNote(Predictions):
    def some_func():
        pass

class String(Predictions):
    def some_func():
        pass


    
class Tablature():

    def __init__(self, onsets = [], midi_notes = [], strings = []):
        self.tab_len = len(onsets)
        self.onsets = [Onset(x) for x in onsets]
        self.midi_notes = [MidiNote(x) for x in midi_notes]
        self.strings = [String(x) for x in strings]
    

def tab2pitch(tab):
    pitch_vector = np.zeros(44)
    string_pitches = [40, 45, 50, 55, 59, 64]
    for string_num in range(len(tab)):
        fret_vector = tab[string_num]
        fret_class = np.argmax(fret_vector, -1)
        # 0 means that the string is closed 
        if fret_class > 0:
            pitch_num = fret_class + string_pitches[string_num] - 41
            pitch_vector[pitch_num] = 1
    return pitch_vector


if __name__ == '__main__':


    dataset = 'hex_cln'
    workspace = initialize_workspace('./data/')
    annotations_folder = 'data/annos'
    jam_name = annotations_folder+'/05_BN1-147-Gb_solo.jams'

    x = TrackInstance(jam_name, dataset)

    f0_pred, times = x.predict_pyin()
    tab_gt, tab_pred = x.load_rep_and_labels_from_raw_file(jam_name, f0_pred)
 
    print(pitch_precision(tab_pred, tab_gt))
    


    # x.predict_tablature()
    # x.get_accuracy_of_prediction('FromCNN')


