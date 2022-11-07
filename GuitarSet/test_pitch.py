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
    
    def predict_onsets(self):
        proc_0 = CNNOnsetProcessor()
        proc_1 = OnsetPeakPickingProcessor(threshold = 0.95,fps=100)
        predicts = proc_1(proc_0(self.track_name))

        #=====manually adding true onsets
        #predicts = [onset.prediction for onset in self.true_tablature.onsets]
        #====

        return list(zip(predicts, [1]*len(predicts))) # here correct it when i can get confidence


    def predict_pyin(self): # __gbastas__
        f0, voiced_flag, voiced_probs = librosa.pyin(self.audio, fmin=librosa.note_to_hz('E2'),fmax=librosa.note_to_hz('E6')) #hop_length=512
        times = librosa.times_like(f0)  #hop_length=512
        print('pyin f0 (raw)',len(f0))
        print('pyin times',len(times))

        D = librosa.amplitude_to_db(np.abs(librosa.stft(x.audio)), ref=np.max) #hop_length=512
        fig, ax = plt.subplots()
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
        ax.set(title='pYIN fundamental frequency estimation')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
        ax.legend(loc='upper right')
        plt.savefig('visual.png')   
        return f0, times

    def load_rep_and_labels_from_raw_file(self, filename, f0): # __gbastas__ copied from tab-cnn repo
        string_midi_pitches = [40,45,50,55,59,64]
        file_audio = self.track_name
        file_anno = self.jam_name
        jam = jams.load(file_anno)
        self.sr_original, data = wavfile.read(file_audio)
        self.sr_curr = self.sr_original
        
        frame_indices = range(len(f0))
        times = librosa.frames_to_time(frame_indices, sr = self.sr_curr, hop_length=512)
        # times = librosa.frames_to_time(frame_indices, sr = self.sr_curr, hop_length=512)
        # print(jam.annotations["note_midi"][4]['data'].value)
        # print(anno.to_samples(times))

        # aaa
        # loop over all strings and sample annotations
        labels = []
        f0_gt = np.array([-1]*len(f0)) #np.array(len(f0))
        for string_num in range(6):
            # anno = annotations
            anno = jam.annotations["note_midi"][string_num]
            string_label_samples = anno.to_samples(times)
            print(string_label_samples)
            # aaa
            # # replace midi pitch values with fret numbers
            for i in frame_indices:
                if string_label_samples[i] == []:
                    string_label_samples[i] = -1
                    # midi_gt.append(-1)
                else:
                    string_label_samples[i] = string_label_samples[i][0]
                    # print(string_label_samples[i])
                    # string_label_samples[i] = int(round(string_label_samples[i][0]) - string_midi_pitches[string_num])
                    # print(round(string_label_samples[i]))
                    f0_gt[i] = string_label_samples[i]

            labels.append([string_label_samples])

        # midi_frames = librosa.samples_to_frames(string_label_samples)

            # print(string_label_samples) 
        # print(string_label_samples[:10])
        # midi_gt_f = librosa.samples_to_frames(midi_gt)
        # print(len(midi_gt_f))
        # print(string_labe_samples.flatten)

        # print(np.count_nonzero(string_label_samples != []))
        # aaa

        labels = np.array(labels)
        # # remove the extra dimension 
        labels = np.squeeze(labels)
        labels = np.swapaxes(labels,0,1)
        
        # print(f0_gt)
        # print(labels.shape)

        # aaa
        # # clean labels
        # labels = self.clean_labels(labels) # NOTE: to avoid wrong dataset annotations
        
        # # store and return
        # labels

        return f0_gt



    def predict_notes_at_time(self,onsets):
        midi_notes = []

        local_onsets = list(zip(*onsets))[0]
        time, frequency, confidence, activation = crepe.predict(self.audio, self.sr, viterbi=True)
        time = list(time)
        frequency = list(frequency)
        confidence = list(confidence)
        for x in local_onsets:
            ind = time.index(round(x,2))
            f_ind = ind # check to get better predictions
            max_so_far = 0
            if ind != 0 and ind<len(local_onsets)-3:
                for p in range(ind-1,ind+2): # maybe can fix fundumental here, and not just round it
                    if confidence[p]>max_so_far:
                        f_ind = p
                        max_so_far = confidence[p]
            midi_notes.append((round(hz2midi(frequency[f_ind])), confidence[f_ind]))
        return midi_notes

    def predict_notes_onsets(self):
        onsets = self.predict_onsets()
        midi_notes = self.predict_notes_at_time(onsets)
        # print(onsets, midi_notes)
        # print(midi_notes)
        return onsets, midi_notes

    
    def get_accuracy_of_prediction(self, mode = 'FromAnnos'):
        tab = self.temp_tablature # __now__

        pre_on = [onset.prediction for onset in tab.onsets]
        # pre_on = [onset.prediction for (onset, t_midi, p_midi) in zip(tab.onsets,  self.true_tablature.midi_notes, tab.midi_notes) if p_midi.prediction==t_midi.prediction ] # __gbastas__
        tru_on = [onset.prediction for onset in self.true_tablature.onsets]
        tp, fp, tn, fn, errors = onset_evaluation(pre_on, tru_on, window = 0.025)

        recall = len(tp)/(len(tp)+len(fn))
        precision = len(tp)/(len(tp)+len(fp))
        f1_measure = 2*recall*precision/(recall+precision)
        print()
        print('Onsets Recall is {} and Precision is {} and F1_measure is {}'.format(recall, precision, f1_measure))
        print()


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
    

if __name__ == '__main__':

    dataset = 'mix'
    workspace = initialize_workspace('./data/')
    annotations_folder = 'data/annos'
    jam_name = annotations_folder+'/05_BN1-147-Gb_solo.jams'

    x = TrackInstance(jam_name, dataset)

    f0_pred, times = x.predict_pyin()
    f0_gt = x.load_rep_and_labels_from_raw_file(jam_name, f0_pred)
 
    # print([midi_note.prediction for midi_note in x.true_tablature.midi_notes])

    # aaa

    print(type(f0_gt))
    print(f0_gt.shape)

    print(type(f0_pred))
    print(f0_pred.shape)

    print(f0_gt[100:180])
    print(f0_pred[100:180])

    print(pitch_precision(f0_pred, f0_gt))
    
    # x.predict_tablature()
    # x.get_accuracy_of_prediction('FromCNN')


