import demo_utils

import argparse
import os
import sys
sys.path.append('./src')
from helper import printProgressBar
from constants_parser import Constants
import soundfile as sf
import utils
import matplotlib.pyplot as plt
import numpy as np
import librosa
import random



def GuitarSetProcessing(constants : Constants):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    listoftracksfile = 'allsolos.txt'
    dataset_names_path = '../datasets/GuitarSet/'

    if args.pickup:
        dataset = 'mix'
    else:
        dataset = 'mic'

    with open(os.path.join(dataset_names_path, listoftracksfile)) as n:
        lines = n.readlines()

    Matrix = np.zeros((6,25))
    count_omitted_note_events = 0
    count_total_note_events = 0
    Strings_gt_total_count = [0]*6
    duration_inner = 0.0

    print('Ongoing Pitch-Fret-String Estimation...') # __new__
    for count, name in enumerate(lines): # iterate over filenames       
        name = name.replace('\n', '')         # e.g. '02_SS2-88-F_solo.jams'
        annosfilepath = os.path.join(constants.annos_path, name)

        printProgressBar(count,len(lines),decimals=0, length=50)

        audiofilepath = os.path.join(constants.track_path, dataset,name[:-5] + '_' + dataset +'.wav') # TODO: set dataset to either mix or mic in constants.ini
        annotations = demo_utils.read_tablature_from_GuitarSet(annosfilepath, constants)   
        annos_tab_list = annotations.tabList
# 
        annos_pitches = [instance.fundamental for instance in annos_tab_list]

        audio, _ = librosa.load(audiofilepath, sr=constants.sampling_rate) 
        len_audio = len(audio)  # Length of your audio signal

        test_onsets = [tab_element.onset for tab_element in annos_tab_list]
        test_offsets = [tab_element.offset for tab_element in annos_tab_list]


        test_strings = [tab_element.string for tab_element in annos_tab_list]
        test_frets = [tab_element.fret for tab_element in annos_tab_list]

        dest_path_auxsolo = './guit-aux-'+constants.dataset+'/'+name[:-9]+'_aux-solo_'+constants.dataset+'/'
        dest_path_auxcomp = './guit-aux-'+constants.dataset+'/'+name[:-9]+'_aux-comp_' + constants.dataset + '/'

        # Initialize
        hex_audio_solo = np.random.normal(0, 0.00005, (6, len_audio)) #  x′ 
        hex_audio_comp = [[0]]*6                                      #  x″ 
        
        # Procedure
        for i, (fret, string, onset, offset) in enumerate(zip(test_frets, test_strings, test_onsets, test_offsets)):
            start = int(round((onset)*(constants.sampling_rate)))
            end = int(round((offset)*(constants.sampling_rate)))

            if i<len(test_onsets)-1:
                endtime = min(offset, test_onsets[i+1])
            else:
                endtime = offset
            end = int(round((endtime)*(constants.sampling_rate)))

            count_total_note_events+=1
            is_chord = False
            
            # check neighbor onsets (at various event-distances) don't "co-occur" as in a chord
            for j in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6):
                idx = i + j
                if 0 <= idx < len(test_onsets):
                    if string != test_strings[idx]: # only consider if it's on a different string
                        dt = abs(test_onsets[idx] - test_onsets[i])
                        if dt < 0.06:
                            is_chord = True
                            break  # stop as soon as we find any chord‐like neighbor

            if is_chord:
                count_omitted_note_events += 1
                Strings_gt_total_count[string] += 1
                continue            

            ########################### note concat ###############################
            hex_audio_solo[string, start:end] = audio[start:end]                     # copy segment x[oᵢ:oᵢ₊₁] → xₛᵢ′
            hex_audio_comp[string] = hex_audio_comp[string] + list(audio[start:end]) # append same segment → xₛᵢ″
            ########################################################################

        hex_audio_comp = set_fixed_comp_len(hex_audio_comp)

        store_hex_audio(hex_audio_solo, dest_path_auxsolo)
        store_hex_audio(hex_audio_comp, dest_path_auxcomp)

        
    print('E and s5', count_active0, count_active1, count_active5)  
    print('Ommited ' + str(count_omitted_note_events) + ' note events out of ' + str(count_total_note_events) +'.')


def set_fixed_comp_len(hex_audio_comp):
    lengths = np.array([len(aud) for aud in hex_audio_comp])
    second_longer = np.argsort(lengths)[-2]
    hex_audio = np.random.normal(0, 0.00005, (6, lengths[second_longer]))  # mean=0, std=0.00005 
    # add silent start and end to all channels shorter than the 2nd longest and cut the actual longest to match the 2nd
    for string, aud in enumerate(hex_audio_comp):
        if len(aud)<lengths[second_longer]:
            s = random.randint(0, lengths[second_longer] - len(aud)-1)
        else:
            s = 0
        L = min(len(aud), lengths[second_longer])
        hex_audio[string, s:L+s] = np.array(aud[:L])
    return hex_audio
    
def store_hex_audio(hex_audio, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    sf.write(dest_path+'E.wav', hex_audio[0,:], constants.sampling_rate)
    sf.write(dest_path+'A.wav', hex_audio[1,:], constants.sampling_rate)
    sf.write(dest_path+'D.wav', hex_audio[2,:], constants.sampling_rate)
    sf.write(dest_path+'G.wav', hex_audio[3,:], constants.sampling_rate)
    sf.write(dest_path+'B.wav', hex_audio[4,:], constants.sampling_rate)
    sf.write(dest_path+'e.wav', hex_audio[5,:], constants.sampling_rate)					
    hex_audio = np.sum(hex_audio, axis=0)
    sf.write(dest_path+'mixture.wav', hex_audio, constants.sampling_rate)
    duration_inner += len(hex_audio) / constants.sampling_rate
    
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
    parser.add_argument('--pickup', action='store_true', help='pickup(="mix") else "mic"')
    parser.add_argument('--plot', action='store_true', help='')  

    args = parser.parse_args()

    config_path = 'constants.ini'
    workspace_folder = '../datasets/GuitarSet/'

    constants = Constants(config_path, workspace_folder)    
    
    constants.dataset_names_path = workspace_folder

    GuitarSetProcessing(constants)



