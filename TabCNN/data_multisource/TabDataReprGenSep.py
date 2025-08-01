import os
import numpy as np
import jams
from scipy.io import wavfile
import sys
import librosa
from keras.utils import to_categorical
import argparse
import shutil

class TabDataReprGen:
    
    # def __init__(self, input_dir, mode="c", ):
    def __init__(self, input_path, mode="c", ):
        # file path to the GuitarSet dataset
        path = "GuitarSet/"
        # self.path_audio = path + "audio/audio_mic/"
        # self.path_audio = path + "audio/datasep-mic/"
        # self.path_audio = path + "audio/datasep-mic-preds/"
        # self.path_audio = path + "audio/datasep-mix-preds/"
        # self.path_audio = path + "audio/datasep-mic-preds_pseudoboth_wn/"
        # self.path_audio = path + "audio/datasep-mic_preds_by_guit_pseudoboth_sep_all_solos_fake/"
        # self.path_audio = path + "audio/datasep-mic_preds_guit_mic_fakemic/"
        # self.path_audio = path + "audio/datasep-mic_preds_guit_mic_fakemic-nopret/"
        # self.path_audio = path + "audio/datasep-customic-pretonmic-preds/"
        # self.path_audio = os.path.join(path,'audio', input_dir)
        # self.path_audio = os.path.join('../../datasets', input_dir)
        self.path_audio = input_path
        
        
        # self.path_audio = path + "audio/audio_mix/"
        self.path_anno = path + "annotation/"
        
        # labeling parameters
        self.string_midi_pitches = [40,45,50,55,59,64]
        self.highest_fret = 19
        self.num_classes = self.highest_fret + 2 # for open/closed
        
        # prepresentation and its labels storage
        self.output = {}
        
        #
        self.preproc_mode = mode
        self.downsample = True
        self.normalize = True
        self.sr_downs = 22050
        
        # CQT parameters
        self.cqt_n_bins = 192
        self.cqt_bins_per_octave = 24
        
        # STFT parameters
        self.n_fft = 2048
        self.hop_length = 512
        
        # save file path
        # self.save_path = "spec_repr/" + self.preproc_mode + "/"
        # self.save_path = "spec_repr_datasepmix_target7/" + self.preproc_mode + "/"
        # self.save_path = "spec_repr_target_nomix/" + self.preproc_mode + "/"
        # self.save_path = "spec_repr_datasepmic_preds/" + self.preproc_mode + "/"
        # self.save_path = "spec_repr_datasepmix_preds/" + self.preproc_mode + "/"
        # self.save_path = "spec_repr_datasepmic_preds_pseudoboth_wn_gdb/" + self.preproc_mode + "/"
        # self.save_path = "spec_repr_datasepmic_preds_by_guit_pseudoboth_sep_all_solos_fake/" + self.preproc_mode + "/"
        # self.save_path = "spec_repr_datasepmic_preds_guit_mic_fakemic/" + self.preproc_mode + "/"
        # self.save_path = "spec_repr_datasepmic_preds_guit_mic_fakemic-nopret/" + self.preproc_mode + "/"
        # self.save_path = "spec_repr_datasep-customic-pretonmic-preds/" + self.preproc_mode + "/"
        # self.save_path = os.path.join("spec_repr_"+ input_dir, self.preproc_mode)
        self.save_path = os.path.join("spec_repr_"+ os.path.basename(input_path), self.preproc_mode)
        
        # self.save_path = "spec_repr_datasepmic_preds_pseudoboth_w/" + self.preproc_mode + "/"

    def load_rep_and_labels_from_raw_file(self, filename, suffix=None):
        # file_audio = self.path_audio + filename + "_mic.wav"
        # file_audio = self.path_audio + filename + "_mix.wav"
        file_anno = self.path_anno + filename + ".jams"
        jam = jams.load(file_anno)

    

        x = []
        # for suffix in ['E', 'A', 'G', 'D', 'B', 'e']:
        # for suffix in ['mixture', 'E', 'A', 'G', 'D', 'B', 'e']:
        for suffix in ['mixture', 'pred_E', 'pred_A', 'pred_G', 'pred_D', 'pred_B', 'pred_e']:
        # for suffix in ['mixture', 'pred_G', 'pred_D', 'pred_B']:
            file_audio = os.path.join(self.path_audio, filename, suffix + ".wav")
            self.sr_original, data = wavfile.read(file_audio)
            self.sr_curr = self.sr_original
            feat = np.swapaxes(self.preprocess_audio(data),0,1) # T, 912
            x.append(feat) 

        x = np.array(x)
        # print('x', x.shape)
        self.output["repr"] = np.transpose( x,  axes=(1, 2, 0))  # T, 192, 7
        # print('output', self.output["repr"].shape)

        # construct labels
        frame_indices = list(range(len(self.output["repr"])))
        times = librosa.frames_to_time(frame_indices, sr = self.sr_curr, hop_length=self.hop_length)
        
        # loop over all strings and sample annotations
        labels = []
        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            string_label_samples = anno.to_samples(times)
            # replace midi pitch values with fret numbers
            for i in frame_indices:
                if string_label_samples[i] == []:
                    string_label_samples[i] = -1
                else:
                    string_label_samples[i] = int(round(string_label_samples[i][0]) - self.string_midi_pitches[string_num])
            labels.append([string_label_samples])
            
        labels = np.array(labels)
        # remove the extra dimension 
        labels = np.squeeze(labels)
        labels = np.swapaxes(labels,0,1)
        
        # clean labels
        labels = self.clean_labels(labels)
        
        # store and return
        self.output["labels"] = labels
        return len(labels)
    
    def correct_numbering(self, n):
        n += 1
        if n < 0 or n > self.highest_fret:
            n = 0
        return n
    
    def categorical(self, label):
        return to_categorical(label, self.num_classes)
    
    def clean_label(self, label):
        label = [self.correct_numbering(n) for n in label]
        return self.categorical(label)
    
    def clean_labels(self, labels):
        return np.array([self.clean_label(label) for label in labels])

    def preprocess_audio(self, data):
        data = data.astype(float)
        if self.normalize:
            data = librosa.util.normalize(data)
        if self.downsample:
            data = librosa.resample(data, self.sr_original, self.sr_downs)
            self.sr_curr = self.sr_downs
        if self.preproc_mode == "c":
            data = np.abs(librosa.cqt(data,
                hop_length=self.hop_length, 
                sr=self.sr_curr, 
                n_bins=self.cqt_n_bins, 
                bins_per_octave=self.cqt_bins_per_octave))
        elif self.preproc_mode == "m":
            data = librosa.feature.melspectrogram(y=data, sr=self.sr_curr, n_fft=self.n_fft, hop_length=self.hop_length)
        elif self.preproc_mode == "cm":
            cqt = np.abs(librosa.cqt(data, 
                hop_length=self.hop_length, 
                sr=self.sr_curr, 
                n_bins=self.cqt_n_bins, 
                bins_per_octave=self.cqt_bins_per_octave))
            mel = librosa.feature.melspectrogram(y=data, sr=self.sr_curr, n_fft=self.n_fft, hop_length=self.hop_length)
            data = np.concatenate((cqt,mel),axis = 0)
        elif self.preproc_mode == "s":
            data = np.abs(librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length))
        else:
            print("invalid representation mode.")

        return data

    def save_data(self, filename):
        np.savez(filename, **self.output)
 

    def get_nth_filename(self, n):
        # returns the filename with no extension
        filenames = np.sort(np.array(os.listdir(self.path_anno)))
        filenames = list(filter(lambda x: x[-5:] == ".jams", filenames))
        return filenames[n][:-5]
    
    def load_and_save_repr_nth_file(self, n):
        # filename has no extenstion
        filename = self.get_nth_filename(n)
        num_frames = self.load_rep_and_labels_from_raw_file(filename)
        print("done: " + filename + ", " + str(num_frames) + " frames")
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_data(os.path.join(save_path, filename + ".npz"))
        shutil.copy('id.csv', os.path.join("spec_repr_"+ os.path.basename(self.path_audio)))
        
def main(args):
    # print(args)
    n = args[0]
    m = args[1]
    i = args[2]
    # gen = TabDataReprGen(input_dir=i, mode=m)
    gen = TabDataReprGen(input_path=i, mode=m)
    gen.load_and_save_repr_nth_file(n)
    
if __name__ == "__main__":


    main(args)



                
