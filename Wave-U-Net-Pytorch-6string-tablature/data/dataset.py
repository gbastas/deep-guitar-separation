import os

import h5py
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import jams
import librosa
from tensorflow.keras.utils import to_categorical


from data.utils import load
np.set_printoptions(threshold=1000)



class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, instruments, sr, channels, shapes, random_hops, hdf_dir, audio_transform=None, in_memory=False, fakeframes_n=None, resample=None):
        '''
        Initialises a source separation dataset
        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(SeparationDataset, self).__init__()


        self.fakeframes_n = fakeframes_n
        self.string_midi_pitches = [40,45,50,55,59,64]

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")

        self.resample = resample
        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments
        self.hop_length=512
        self.highest_fret = 19
        self.num_classes = self.highest_fret + 2 # for open/closed


        print('[data/dataset.py] PREPARE HDF FILE')

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_dir):

            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            with h5py.File(self.hdf_dir, "w") as f:
                f.attrs["sr"] = sr
                f.attrs["channels"] = channels
                f.attrs["instruments"] = instruments

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])):
                    # Load mix
                    mix_audio, _ = load(example["mix"], sr=self.sr, mono=(self.channels == 1))

                    source_audios = []
                    for source in instruments:
                        # In this case, read in audio and convert to target sampling rate
                        source_audio, _ = load(example[source], sr=self.sr, mono=(self.channels == 1))
                        source_audios.append(source_audio)
                    source_audios = np.concatenate(source_audios, axis=0)
                    
                    ### sometimes one extra sample ###
                    n = min(source_audios.shape[1], mix_audio.shape[1])
                    source_audios = source_audios[:,:n]
                    mix_audio = mix_audio[:,:n]
                    assert(source_audios.shape[1] == mix_audio.shape[1])

                    # Find jam file
                    mix_path = example["mix"]
                    track_name = mix_path.split('/')[-2]
                    file_anno = "data-tab/GuitarSet/annotation/" + track_name +".jams"
                    jam = jams.load(file_anno)              
                    num_frets = 21  # Number of frets, including open and close string as a 'fret'

                    #### Compute CQT times ##### 
                    x = mix_audio[0]
                    x_cqt = np.swapaxes(self.preprocess_audio(x),0,1) # (1379, 192)
                    sample_indices = range(len(x_cqt))
                    cqt_times = librosa.frames_to_time(sample_indices, sr=22050, hop_length=512) # frame-indices to timestamps (Wiggins et al. resolution)
                    ############################


                    ######### Sample annotation labels to match the CQT frame rate ############
                    labels = []
                    for string_num, string_name in enumerate(self.instruments):
                        anno = jam.annotations["note_midi"][string_num]
                        string_label_frames = anno.to_samples(cqt_times) # jams function to sample the annotation at specified times.
                        # replace midi pitch values with fret numbers
                        for i in sample_indices:
                            if string_label_frames[i] == []:
                                string_label_frames[i] = -1
                            else:
                                string_label_frames[i] = int(round(string_label_frames[i][0]) - self.string_midi_pitches[string_num])
                        labels.append([string_label_frames])

                    labels = np.array(labels)
                    labels = np.squeeze(labels)
                    labels = np.swapaxes(labels,0,1)                                
                    labels = self.clean_labels(labels)
                    labels = np.transpose(labels, (1, 2, 0)) # N x 6 x 21 --> 6 x 21 x N
                    ###########################################################################


                    ######### Upsample labels to match audio ##########
                    labels_samples_dict = {}
                    for string_num, string_name in enumerate(self.instruments):
                        ann = jam.annotations["note_midi"][string_num]

                        # Initialize the label array for this string with 0 except from the first line (see, choose inactive/close string as default state)
                        string_sample_labels = np.full((num_frets, mix_audio.shape[1]), 0) # (21, audio_samples_n)
                        string_sample_labels[0,:]=1
                        for obs in ann:
                            start_sample_index = round(obs.time * self.sr)
                            end_sample_index = round((obs.time + obs.duration) * self.sr)  # Add duration to the start time          
                            fret_number = round(obs.value) - self.string_midi_pitches[string_num]
                            if 0 <= fret_number < num_frets:
                                string_sample_labels[fret_number, start_sample_index:end_sample_index] = 1                        
                                string_sample_labels[0, start_sample_index:end_sample_index] = 0  # Marking the fret as played at this time
                            else:
                                print(f"Warning: Fret number {fret_number} out of range for string {string_name}")

                        labels_samples_dict[string_name] = string_sample_labels
                    ###################################################   

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=mix_audio.shape, dtype=mix_audio.dtype, data=mix_audio)
                    grp.create_dataset("cqt_inputs", shape=x_cqt.shape, dtype=x_cqt.dtype, data=x_cqt)
                    grp.create_dataset("targets", shape=source_audios.shape, dtype=source_audios.dtype, data=source_audios)
                    grp.attrs["length"] = mix_audio.shape[1]
                    grp.attrs["target_length"] = source_audios.shape[1]                   
                    grp.attrs["mix_path"] = example["mix"]  

                    # Store labels into the HDF5 file
                    labels_grp = grp.create_group("labels")
                    for idx, (string_name, string_sample_labels) in enumerate(labels_samples_dict.items()):
                        string_grp = labels_grp.create_group(string_name)
                        string_grp.create_dataset("label_array", data=string_sample_labels.astype(np.uint8), dtype='uint8')
                        string_grp.create_dataset("label_short_array", data=labels[idx,:,:]) # labels matching the CQT frame rate of Wiggins et al. (96.6 frames for 22.22 sec input).

        else:
            print('[dataset.py] hdf dir already exists. Deal with it!')


    
        # Check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_dir, "r") as f:

            if f.attrs["sr"] != sr or \
                    f.attrs["channels"] != channels or \
                    list(f.attrs["instruments"]) != instruments:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate and channel or instruments are not as expected. Did you load an out-dated HDF file?")

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(song_idx)].attrs["target_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]


    ############## FROM TABCNN REPO (Wiggins et al.) ###############
    def preprocess_audio(self, data):
        data = data.astype(float)
        data = librosa.util.normalize(data)
        data = librosa.resample(data, 44100, 22050)
        data = np.abs(librosa.cqt(data,
            hop_length=512, 
            sr=22050, 
            n_bins=192, 
            bins_per_octave=24))
        return data

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
    ##############################################################
    
    
    def __getitem__(self, index):
        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver=driver)

        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx - 1]

        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        target_length = self.hdf_dataset[str(audio_idx)].attrs["target_length"]

        # Determine position where to start targets
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(target_length - self.shapes["output_frames"] + 1, 1))
        else:
            # Map item index to sample position within song
            start_target_pos = index * self.shapes["output_frames"]

        # READ INPUTS
        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.hdf_dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        targets = self.hdf_dataset[str(audio_idx)]["targets"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            targets = np.pad(targets, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        targets = {inst : np.expand_dims(targets[idx], axis=0) for idx, inst in enumerate(self.instruments)} # __gbastas__


        ############## Targets' Cropping with data.utils.crop_targets() (and augmentation!) ###################
        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, targets = self.audio_transform(audio, targets)       
        #######################################################################################################

        fakeframes_n = self.fakeframes_n



        ### CQT ###
        x_cqt = self.hdf_dataset[str(audio_idx)]["cqt_inputs"]
        x_cqt = np.swapaxes(x_cqt,0,1) # (192, Τ)

        start_target_frame = round( start_target_pos / (2*self.hop_length) )
        start_frame = start_target_frame - 4 
        # start_frame = start_target_frame - 5 
        if start_frame < 0:
            # Pad manually since audio signal was too short
            pad_front_cqt = abs(start_frame)
            start_frame = 0
        else:
            pad_front_cqt = 0


        # Check back padding
        end_frame = start_target_frame - 4 + fakeframes_n + 8
        # end_frame = start_target_frame - 5 + fakeframes_n + 10
        if end_frame > x_cqt.shape[1]:
            pad_back_cqt = end_frame - x_cqt.shape[1]
            end_frame = x_cqt.shape[1]
        else:
            pad_back_cqt = 0

        x_cqt_section = x_cqt[:, start_frame:end_frame]

        if pad_front_cqt > 0 or pad_back_cqt > 0:
            x_cqt_section = np.pad(x_cqt_section, [(0, 0), (pad_front_cqt, pad_back_cqt)], mode="constant", constant_values=0.0)

  
        x_cqt_section = x_cqt_section.astype(np.float32)
        original_frames = x_cqt_section.shape[1] 


        resampled_labels_dict = {}
        labels_grp = self.hdf_dataset[str(audio_idx)]["labels"]
        for string_name in labels_grp.keys():
            string_grp = labels_grp[string_name]

            ######## NEW #######
            label_short_array = string_grp['label_short_array'][:]
            label_short_array_section = label_short_array[:, start_frame:end_frame]
            if pad_front_cqt > 0 or pad_back_cqt > 0:
                label_short_array_section = np.pad(label_short_array_section, [(0, 0), (pad_front_cqt, pad_back_cqt)], mode="constant", constant_values=0.0)
            label_short_array_section = label_short_array_section[:, 4:-4] # from 96 (input) to 87! NEW! 
            # label_short_array_section = label_short_array_section[:, 5:-5] # from 93 (input) to 83! NEW! 

            ######## NEW #######

            # NOTE: provisional
            resampled_labels_dict[string_name] = label_short_array_section # shape: (21, T), see one-hot style


        return audio, targets, resampled_labels_dict, x_cqt_section        # return audio, targets

    def __len__(self):
        return self.length
    









