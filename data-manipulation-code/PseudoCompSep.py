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

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    constants.crop_win=3
    if args.all_solos:
        constants.listoftracksfile = 'allsolos.txt'
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
    count_active0 = 0
    count_active5 = 0
    count_active1 = 0 
    duration_inner = 0.0
    duration_inter = 0.0
    past_channels = [[] for _ in range(6)]
    print('Ongoing Pitch-Fret-String Estimation...') # __new__
    for count, name in enumerate(lines): # iterate over filenames
        
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

        if args.all_solos:
            dest_path = './pseudocomp_sep_all_solos_'+constants.dataset+'_wn/'+name[:-9]+'comp_shuffl_hex_' + constants.dataset + '/'
        else:
            dest_path = './pseudocomp_sep_few_solos_'+constants.dataset+'_wn/'+name[:-9]+'comp_shuffl_hex_' + constants.dataset + '/'

        hex_audio_comp = [[0]]*6
        
        for i, (fret, string, onset, offset) in enumerate(zip(test_frets, test_strings, test_onsets, test_offsets)):
            start = int(round((onset)*(constants.sampling_rate)))
            end = int(round((offset)*(constants.sampling_rate)))

            # avoid chords
            is_chord = False
            if args.all_solos:
                # check neighbor onsets at different distances 
                for j in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6):
                    idx = i + j
                    if 0 <= idx < len(test_onsets):
                        # only consider if it's on a different string
                        if string != test_strings[idx]:
                            # timing difference
                            dt = abs(test_onsets[idx] - test_onsets[i])
                            # interval overlap?
                            overlap = (test_onsets[idx] < test_offsets[i] and
                                    test_onsets[i]   < test_offsets[idx])
                            if dt < 0.06 or overlap:
                                is_chord = True
                                # print(f'"chord" via neighbor {j}: dt={dt:.3f}, overlap={overlap}')
                                break  # stop as soon as we find any chord‐like neighbor
                            # if overlap:
                            #         overlap_start = max(onset, test_onsets[idx]-0.25)
                            #         t_overlap = overlap_start - onset
                            #         end = start + int(round(t_overlap * constants.sampling_rate))

            if is_chord:
                count_omitted_note_events += 1
                Strings_gt_total_count[string] += 1
                continue            

            ########################### note concat ###############################
            hex_audio_comp[string] = hex_audio_comp[string] + list(audio[start:end])
            ########################################################################

        lengths = np.array([len(aud) for aud in hex_audio_comp])
        # # Counting durations per string
        # for string in (0, 1, 5):
        #     length_in_sec = len(hex_audio_comp[string])/constants.sampling_rate
        #     if string==0: # 17
        #         if length_in_sec > 2:
        #             print('string0', length_in_sec, 's')
        #             count_active0+=1 
        #         else:
        #             print('s0 no active')
                    
        #     if string==1: # 54
        #         if length_in_sec > 2:
        #             print('string1', length_in_sec, 's')
        #             count_active1+=1 
        #         else:
        #             print('s1 no active')

        #     elif string==5: # 106
        #         if length_in_sec > 2:
        #             print('string5', length_in_sec, 's')
        #             count_active5+=1 
        #         else:
        #             print('s5 no active')
        # print()        
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

        # --- store channels for augmentation ---
        datasep_mic_test_path_asref = "../datasets/GuitarSet/datasep-mic/test/"
        ref_dir = os.path.join(datasep_mic_test_path_asref, name[:-5])
        if not os.path.exists(ref_dir): # only if is not a test song gather sources
            for string in range(6):
                length_in_sec = len(hex_audio_comp[string])/constants.sampling_rate
                if length_in_sec > 1:
                    past_channels[string].append(np.array(hex_audio_comp[string]))
                    if string == 0:
                        count_active0+=1 
                        # print('string0', length_in_sec, 's')            
        else:
            print("Found a test song. So it's not included in overshuffl list.")

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
        
    print('E and s5', count_active0, count_active1, count_active5)  
    print('Ommited ' + str(count_omitted_note_events) + ' note events out of ' + str(count_total_note_events) +'.')
    print('Number of  saved examples:', len(past_channels[0]))
    

    # for i in range(600):
    for i in range(1500):
        
        # print(past_channels[0])
        # AAAAAAAAA
        # pick random past channels excluding current
        rand_channels = [random.choice(past_channels[i]) for i in range(6)]
        lengths = [len(aud) for aud in rand_channels]
        second_longer = np.argsort(lengths)[-2]
        
        hex_audio = np.random.normal(0, 0.00005, (6, lengths[second_longer]))  # mean=0, std=0.00005     
    
    
        # add silent start and end to all channels shorter than the 2nd longest and cut the actual longest to match the 2nd
        for string, aud in enumerate(rand_channels):
            if len(aud)<lengths[second_longer]:
                s = random.randint(0, lengths[second_longer] - len(aud)-1)
            else:
                s = 0
            L = min(len(aud), lengths[second_longer])
            hex_audio[string, s:L+s] = np.array(aud[:L])
        # print('BBBBBBBB', dest_path)

        if args.all_solos:
            dest_path = './pseudocomp_sep_all_solos_'+constants.dataset+'_wn/'+'comp_overshuffl_hex_' + str(i) + '_' + constants.dataset + '/'
        else:
            dest_path = './pseudocomp_sep_few_solos_'+constants.dataset+'_wn/'+'comp_overshuffl_hex_' + str(i) + '_' + constants.dataset + '/'
        
        os.makedirs(dest_path, exist_ok=True)

        scaling_factor = np.random.uniform(0.75, 1.0)
        sf.write(dest_path+'E.wav', hex_audio[0,:]*scaling_factor, constants.sampling_rate)
        scaling_factor = np.random.uniform(0.75, 1.0)
        sf.write(dest_path+'A.wav', hex_audio[1,:]*scaling_factor, constants.sampling_rate)
        scaling_factor = np.random.uniform(0.75, 1.0)
        sf.write(dest_path+'D.wav', hex_audio[2,:]*scaling_factor, constants.sampling_rate)
        scaling_factor = np.random.uniform(0.75, 1.0)
        sf.write(dest_path+'G.wav', hex_audio[3,:]*scaling_factor, constants.sampling_rate)
        scaling_factor = np.random.uniform(0.75, 1.0)
        sf.write(dest_path+'B.wav', hex_audio[4,:]*scaling_factor, constants.sampling_rate)
        scaling_factor = np.random.uniform(0.75, 1.0)
        sf.write(dest_path+'e.wav', hex_audio[5,:]*scaling_factor, constants.sampling_rate)					

        hex_audio = np.sum(hex_audio, axis=0)
        sf.write(dest_path+'mixture.wav', hex_audio, constants.sampling_rate)
        duration_inter += len(hex_audio) / constants.sampling_rate
        
    print(f"Total inner-song mixture time: {duration_inner:.2f} seconds ({duration_inner / 60:.2f} minutes)")
    print(f"Total inter-song mixture time (from shuffled past channels): {duration_inter:.2f} seconds ({duration_inter / 60:.2f} minutes)")
    
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



