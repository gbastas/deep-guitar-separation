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



def GuitarSetProcessing(constants : Constants):
    """ function that runs tests on the jams files mentioned in the given file 
    and plots the confusion matrixes for both the genetic and inharmonic results."""


    constants.crop_win=3
    if args.all_solos:
        constants.listoftracksfile = 'allsolos.txt'
    elif args.all_tracks:
        constants.listoftracksfile = 'alltracks.txt'                    
    else:
        constants.listoftracksfile = 'names.txt'


    if args.pickup:
        constants.dataset = 'mix'
    else:
        constants.dataset = 'mic'


    with open(os.path.join(constants.dataset_names_path, constants.listoftracksfile)) as n:
        lines = n.readlines()

    Matrix = np.zeros((6,25))
    count_omitted_note_events = 0
    count_total_note_events = 0
    Strings_gt_total_count = [0]*6
    print('Ongoing Pitch-Fret-String Estimation...') # __new__
    for count, name in enumerate(lines): # iterate over filenames
        # if count <
        name = name.replace('\n', '')         # e.g. '02_SS2-88-F_solo.jams'
        # print('testfile', name)
        annosfilepath = os.path.join(constants.annos_path, name)

        printProgressBar(count,len(lines),decimals=0, length=50)

        audiofilepath = os.path.join(constants.track_path,constants.dataset,name[:-5] + '_' + constants.dataset +'.wav') # TODO: set dataset to either mix or mic in constants.ini
        annotations = demo_utils.read_tablature_from_GuitarSet(annosfilepath, constants)   
        annos_tab_list = annotations.tabList
# 
        annos_pitches = [instance.fundamental for instance in annos_tab_list]

        audio, _ = librosa.load(audiofilepath, sr=constants.sampling_rate) 

        test_onsets = [tab_element.onset for tab_element in annos_tab_list]
        test_offsets = [tab_element.offset for tab_element in annos_tab_list]


        test_strings = [tab_element.string for tab_element in annos_tab_list]
        test_frets = [tab_element.fret for tab_element in annos_tab_list]


        # to get the note audio instances:
        # tablature = Tablature(test_onsets, test_offsets, audio, constants)


        # if args.action == 'pseudo_sep':
        if args.all_solos:
            dest_path = './pseudo_sep_all_solos_'+constants.dataset+'_wn/'+name[:-5]+'_hex_'+constants.dataset+'/'
        elif args.all_tracks:
            dest_path = './pseudo_sep_all_tracks_'+constants.dataset+'_wn/'+name[:-5]+'_hex_'+constants.dataset+'/'
        else:
            dest_path = './pseudo_sep_few_solos_'+constants.dataset+'_wn/'+name[:-5]+'_hex_'+constants.dataset+'/'
        # Assuming 'audio' is already defined and you want to match its length
        len_audio = len(audio)  # Length of your audio signal
        hex_audio = np.random.normal(0, 0.00005, (6, len_audio))  # mean=0, std=0.00005          WHITE NOISE!!!

        for i, (fret, string, onset, offset) in enumerate(zip(test_frets, test_strings, test_onsets, test_offsets)):
            start = int(round((onset)*(constants.sampling_rate)))
            # end = int(round((offset)*(constants.sampling_rate)))
            if i<len(test_onsets)-1:
                endtime = min(offset, test_onsets[i+1])
            else:
                endtime = offset
            end = int(round((endtime)*(constants.sampling_rate)))

            count_total_note_events+=1
            # avoid chords
            is_chord = False
            if args.all_solos:
                # check neighbors at distances -2, -1, +1, +2
                for j in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6):
                    idx = i + j
                    if 0 <= idx < len(test_onsets):
                        # only consider if it's on a different string
                        if string != test_strings[idx]:
                            dt = abs(test_onsets[idx] - test_onsets[i])
                            # interval overlap?
                            overlap = (test_onsets[idx] < test_offsets[i] and
                                    test_onsets[i]   < test_offsets[idx])
                            # if dt < 0.06 or overlap:
                            if dt < 0.06:# or overlap:
                                is_chord = True
                                # you can log how far away the neighbor was:
                                # print(f'"chord" via neighbor {j}: dt={dt:.3f}, overlap={overlap}')
                                break  # stop as soon as we find any chord‐like neighbor                           

            if is_chord:
                count_omitted_note_events += 1
                Strings_gt_total_count[string] += 1
                continue

            # print('len_audio', len(audio[start:end]))
            # print('len_hex_audio', len(hex_audio[string, start:end]))
            hex_audio[string, start:end] = audio[start:end]
            # print('len_hex_audio-next', len(hex_audio[string, start:end]))
            # print()


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
    parser.add_argument('--pickup', action='store_true', help='pickup(="mix") else "mic"')
    parser.add_argument('--plot', action='store_true', help='')
    

    args = parser.parse_args()

    config_path = 'constants.ini'
    workspace_folder = '../datasets/GuitarSet/'

    constants = Constants(config_path, workspace_folder)    
    
    constants.dataset_names_path = workspace_folder

    GuitarSetProcessing(constants)
