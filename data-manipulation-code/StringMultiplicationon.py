import demo_utils

import argparse
import os
import sys
import crepe 

sys.path.append('./src')
from track_class import Tablature
from helper import printProgressBar
from constants_parser import Constants
import soundfile as sf
import utils
import matplotlib.pyplot as plt
import numpy as np
import librosa
import random

import numpy as np
import random
import os
import soundfile as sf

def augment_data(isolated_channels, num_samples_to_generate=100):
    """
    Function to balance the distribution of strings by rendering notes for underrepresented strings.
    This renders samples in such a way that strings E, A, and e (underrepresented) are processed last, 
    while strings D, G, and B are processed first.
    
    Args:
    isolated_channels (list): A list of isolated channels (for each string) containing tuples of (note_count, audio_sample).
    num_samples_to_generate (int): Total number of new samples to generate.
    """
    L = len(isolated_channels[0]) # 360
    print('L', L)
    augmented_data=[]
    # print(isolated_channels[0])
    notes_per_string_count_songwise_aug = [0]*6    
    for i in range(1,35):
        E_prominent_song = [None, None, None, None, None, None]
        E_notes, E_sample = isolated_channels[0][-i]
        print('E_notes', E_notes)
        notes_per_string_count_songwise_aug[0] += E_notes
        E_prominent_song[0] = E_sample
        for string in (1,2,3,4,5):
            idx = random.choice(range(0,L))
            # print('idx', idx)
            # print('is',isolated_channels[string][idx])
            n_notes, E_prominent_song[string] = isolated_channels[string][idx]
            notes_per_string_count_songwise_aug[string] += n_notes
            
        A_prominent_song = [None, None, None, None, None, None]
        A_notes, A_sample = isolated_channels[1][-i]
        notes_per_string_count_songwise_aug[1] += A_notes
        A_prominent_song[1] = A_sample
        for string in (0,2,3,4,5):
            idx = random.choice(range(0,L))
            n_notes, A_prominent_song[string] = isolated_channels[string][idx]
            notes_per_string_count_songwise_aug[string] += n_notes
        
        e_prominent_song = [None, None, None, None, None, None]
        e_notes, e_sample = isolated_channels[5][-i]
        notes_per_string_count_songwise_aug[5] += e_notes
        e_prominent_song[5] = e_sample
        for string in (0,1,2,3,4):
            idx = random.choice(range(0,L))
            n_notes, e_prominent_song[string] = isolated_channels[string][idx]
            notes_per_string_count_songwise_aug[string] += n_notes

        augmented_data.append(E_prominent_song)
        augmented_data.append(A_prominent_song)
        augmented_data.append(e_prominent_song)  
    
    print('notes_per_string_count_songwise_aug', notes_per_string_count_songwise_aug)
    plot_note_hist(notes_per_string_count_songwise_aug)    
    return augmented_data

def GuitarSetProcessing(constants : Constants, args):
    """ function that runs tests on the jams files mentioned in the given file 
    and plots the confusion matrixes for both the genetic and inharmonic results."""

    constants.crop_win=3
    if args.all_solos:
        listofjamfiles = 'allsolos.txt'
    elif args.all_tracks:
        listofjamfiles = 'alltracks.txt'                    
    elif args.all_comps:
        listofjamfiles = 'allcomps.txt'
    else:
        listofjamfiles = 'names.txt'

    with open(os.path.join(args.dataset_names_path, listofjamfiles)) as n:
        lines = n.readlines()

    Matrix = np.zeros((6,25))
    count_omitted_note_events = 0
    count_total_note_events = 0
    Strings_gt_total_count = [0]*6
    isolated_channels = [[] for _ in range(6)]
    print('Ongoing Pitch-Fret-String Estimation...') # __new__
    for count, name in enumerate(lines): # iterate over filenames
        # if count <
        name = name.replace('\n', '')         # e.g. '02_SS2-88-F_solo.jams'
        # if count>50:
        #     break
        annosfilepath = os.path.join(constants.annos_path, name)
        printProgressBar(count,len(lines),decimals=0, length=50)

        audiofilepath = os.path.join(constants.track_path,args.dataset,name[:-5] + '_' + args.dataset +'.wav') # TODO: set dataset to either mix or mic in constants.ini
        # print('audiofilepath', audiofilepath)
        annotations = demo_utils.read_tablature_from_GuitarSet(annosfilepath, constants)   
        annos_tab_list = annotations.tabList
        audio, _ = librosa.load(audiofilepath, sr=constants.sampling_rate, mono=False) 

        # print('audiofilepath', audiofilepath)
        # print('audioshape', audio.shape)
        test_strings = [tab_element.string for tab_element in annos_tab_list]
        # if args.action == 'pseudo_sep':
        # dest_path = './aug_data_'+args.dataset+'_wn/'+name[:-5]+'_hex_'+args.dataset+'/'
        # Assuming 'audio' is already defined and you want to match its length
        len_audio = len(audio)  # Length of your audio signal
        # print('audio', audio.shape)
        # hex_audio = np.random.normal(0, 0.00005, (6, len_audio))  # mean=0, std=0.00005          WHITE NOISE!!!
        # isolated_channels = [[] for _ in range(6)]
        notes_per_string_count_songwise = [0]*6
        for string in test_strings:
            count_total_note_events+=1
            Strings_gt_total_count[string] += 1
            notes_per_string_count_songwise[string] += 1        
            
        # --- store channels for augmentation ---
        datasep_mic_test_path_asref = "../datasets/GuitarSet/datasep-mic/test/"
        ref_dir = os.path.join(datasep_mic_test_path_asref, name[:-5])
        if not os.path.exists(ref_dir): # only if is not a test song gather sources
            for string in range(6):
                isolated_channels[string].append((notes_per_string_count_songwise[string], np.array(audio[string])))
        else:
            if not args.plot:
                print("Found a test song. So it's not included in overshuffl list.")

    # print(isolated_channels)
    # aaaaaaaaa
    # Sort the isolated_channels based on the first element of each tuple (note count)
    for string in range(6):
        isolated_channels[string] = sorted(isolated_channels[string], key=lambda x: x[0])            

    

    # total_notes = sum(Strings_gt_total_count)
    # inverse_distribution = [total_notes / count if count > 0 else 0 for count in Strings_gt_total_count]
    # max_inverse = max(inverse_distribution)
    # normalized_inverse = [inv / max_inverse for inv in inverse_distribution]

    # # Output the inverse distribution
    # print("Inverse Distribution (normalized):", normalized_inverse)

    # augmented_channels = augment_data(isolated_channels, normalized_inverse, num_samples_to_generate=100)
    augmented_channels = augment_data(isolated_channels, num_samples_to_generate=100)
    for i, channelled_song_as_list in enumerate(augmented_channels):
        # for aud in channelled_song_as_list:
        #     print('aud', aud)
        lengths = np.array([len(aud) for aud in channelled_song_as_list])  
        max_length = np.max(lengths)
        hex_audio = np.random.normal(0, 0.00005, (6, max_length))  # mean=0, std=0.00005 
        for string, aud in enumerate(channelled_song_as_list):
            s = random.randint(0, max_length - len(aud))
            L = len(aud)
            hex_audio[string, s:L+s] = np.array(aud)

        dest_path = './aug_data_'+args.dataset+'/shuffl_'+str(i)+args.dataset+'/'
        
        os.makedirs(dest_path, exist_ok=True)

        sf.write(dest_path+'E.wav', hex_audio[0,:], constants.sampling_rate)
        sf.write(dest_path+'A.wav', hex_audio[1,:], constants.sampling_rate)
        sf.write(dest_path+'D.wav', hex_audio[2,:], constants.sampling_rate)
        sf.write(dest_path+'G.wav', hex_audio[3,:], constants.sampling_rate)
        sf.write(dest_path+'B.wav', hex_audio[4,:], constants.sampling_rate)
        sf.write(dest_path+'e.wav', hex_audio[5,:], constants.sampling_rate)					

        hex_audio = np.sum(hex_audio, axis=0)
        sf.write(dest_path+'mixture.wav', hex_audio, constants.sampling_rate)


    print('Ommited ' + str(count_omitted_note_events) + ' note events out of ' + str(count_total_note_events) +'.')
    if args.plot:
        plot_note_hist(Strings_gt_total_count)

def plot_note_hist(Strings_gt_total_count):
    plt.figure(figsize=(30,10))
    plt.rc('font', size=38)
    plt.rc('axes', titlesize=50)

    plt.yticks(rotation=30)

    plt.xlabel('Guitar Strings', fontsize = 44, fontweight='bold')
    plt.ylabel('No. of note instances', fontsize = 44, fontweight='bold')

    plt.bar(['E','A','D','G','B','e'], Strings_gt_total_count, color='maroon',  width=0.7)
    plt.bar(['E','A','D','G','B','e'], Strings_gt_total_count, width=0.7)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    plt.savefig('bar_total.png', bbox_inches='tight')        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-pred_onset', action='store_true', help='')
    parser.add_argument('-create_no', action='store_true', help='')
    parser.add_argument('--action', type=str, help='gather_notes, pseudo_sep, pseudocomp_sep')
    parser.add_argument('--all_solos', action='store_true', help='if True: allsolos.txt, else: names.txt')
    parser.add_argument('--all_comps', action='store_true', help='if True: allsolos.txt, else: names.txt')
    parser.add_argument('--all_tracks', action='store_true', help='if True: allsolos.txt, else: names.txt')
    parser.add_argument('--dataset', type=str, default='hex_cln', help='or "mix" or "mic"')
    parser.add_argument('--plot', action='store_true', help='')
    parser.add_argument('--crop_win', type=int, default=3, help='')
    parser.add_argument('--dataset_names_path', type=str, default='../datasets/GuitarSet/')
    
    

    args = parser.parse_args()

    config_path = 'constants.ini'
    # workspace_folder = '../datasets/GuitarSet/'

    # constants = Constants(config_path, workspace_folder)    
    constants = Constants(config_path, args.dataset_names_path)
    
    # dataset_names_path = workspace_folder

    GuitarSetProcessing(constants, args)
