import demo_utils

import argparse
import os
import sys
# import crepe 

sys.path.append('./src')
from track_class import Tablature
from helper import printProgressBar
from constants_parser import Constants
import soundfile as sf
# import utils
import matplotlib.pyplot as plt
import numpy as np
import librosa
# import random

def print_tablature(tab_list, max_chars_per_line=180):
    # Initialize the tablature lines
    tab_lines = ['-' * (len(tab_list) * 3) for _ in range(6)]

    # Fill in the tablature lines with the fret numbers
    for i, tab_element in enumerate(tab_list):
        fret_str = str(tab_element.fret) if tab_element.fret >= 10 else f' {tab_element.fret}'
        if tab_element.string < 6:
            tab_lines[tab_element.string] = tab_lines[tab_element.string][:i*3] + fret_str + '-' + tab_lines[tab_element.string][i*3+3:]

    # Print the tablature
    for line in reversed(tab_lines):
        for i in range(0, len(line), max_chars_per_line):
            print(line[i:i+max_chars_per_line])
            break



def GuitarSetProcessing(constants : Constants):
    """ function that runs tests on the jams files mentioned in the given file 
    and plots the confusion matrixes for both the genetic and inharmonic results."""

    if args.action == 'gather_notes':

        notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

        standard_tuning = ['E', 'A', 'D', 'G', 'B', 'E']
        Note = [ ['None']*25]*6
        Note = np.array(Note)

        for i, note in enumerate(standard_tuning):
            # Find the index of the open note for each string
            start_idx = notes.index(note)
            # Fill in the matrix row by row, with cyclic index into notes
            for fret in range(25):
                Note[i,fret] = notes[(start_idx + fret) % 12]


    if args.action in ['gather_notes', 'pseudo_sep', 'pseudocomp_sep', 'plot']:
        constants.crop_win=3
        if args.all_solos:
            constants.listoftracksfile = 'allsolos.txt'
        elif args.all_comps:
            constants.listoftracksfile = 'allcomps.txt'
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
    print('Gathering Notes...') # __new__
    for count, name in enumerate(lines): # iterate over filenames
        
        name = name.replace('\n', '')         # e.g. '02_SS2-88-F_solo.jams'
        # print('testfile', name)
        jam_annosfilepath = os.path.join(constants.annos_path, name)
        printProgressBar(count,len(lines),decimals=0, length=50)

        audiofilepath = os.path.join(constants.track_path, constants.dataset, name[:-5] + '_' + constants.dataset +'.wav') 

        annotations = demo_utils.read_tablature_from_GuitarSet(jam_annosfilepath, constants)   
        annos_tab_list = annotations.tabList

        annos_pitches = [instance.fundamental for instance in annos_tab_list]

        audio, _ = librosa.load(audiofilepath, sr=constants.sampling_rate) 
        test_onsets = [tab_element.onset for tab_element in annos_tab_list]
        test_offsets = [tab_element.offset for tab_element in annos_tab_list]


        test_strings = [tab_element.string for tab_element in annos_tab_list]
        test_frets = [tab_element.fret for tab_element in annos_tab_list]

        # Exclude test songs!
        datasep_mic_test_path_asref = "../datasets/GuitarSet/datasep-mic/test/" 
        ref_dir = os.path.join(datasep_mic_test_path_asref, name[:-5])
        if os.path.exists(ref_dir): # only if is not a test song gather sources
            continue



        if args.action == "gather_notes":
            # NOTE: To create note_instances dataset uncomment and in constants.ini set crop_win=3 and listoftracksfile = allsolos.txt
            note_instances_dir = './note_instances_' +constants.dataset+'/data/train/guitar'
            onset_instances_dir = './note_instances_'+constants.dataset+'/data/onsets/guitar'
            # os.makedirs(note_instances_dir, exist_ok=True)            
            # for tabelement in tablature.tabList:
            for i, (fret, string, onset, offset) in enumerate(zip(test_frets, test_strings, test_onsets, test_offsets)):
                start = int(round((onset)*(constants.sampling_rate)))
                end = int(round((offset)*(constants.sampling_rate)))
                count_total_note_events+=1                
                
                # avoid chords
                is_chord = False
                # check neighbors at distances -2, -1, +1, +2
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
                                
                if is_chord:
                    count_omitted_note_events += 1
                    Strings_gt_total_count[string] += 1
                    continue
                
                note_audio = audio[start:end]                                                                           
                duration_in_seconds = len(note_audio) / constants.sampling_rate
                # print(f"Duration of note_audio: {duration_in_seconds:.2f} seconds")

                # # crepe_pitch
                # sr = constants.sampling_rate
                # (_, freqs, confs, _)  = crepe.predict(note_audio, sr=sr, viterbi=True)

                # id = np.argmax(confs)
                # confidence = confs[id]
                # frequency = freqs[id]
                # if confidence<0.8:
                #     print('LOW PITCH ESTIMATION CONFIDENCE!!')
                #     continue
                # if len(note_audio)/constants.sampling_rate<0.08: # not less than 0.5 seconds
                #     continue            

                # try:
                #     note = utils.hz_to_midi(frequency)
                # except TypeError as e:
                #     print('[MyWaring] freq:', frequency,  e)
                #     continue
                
                # note = librosa.midi_to_note(note)

                # # Compare estimated note with annotated note (this is to gather only clean instances)
                # # fret = tabelement.fret
                # # string = tabelement.string
                # if note[:-1].replace('♯','#')!= Note[string, fret]:
                #     print('Notes', note[:-1].replace('♯','#'), Note[string, fret])
                #     print('String-fret', string, fret)  
                #     continue

                Matrix[string,fret]+=1
                if Matrix[string,fret]>100:
                    continue

                # Store the note audio in the corresponding directory
                # note_audio = tabelement.note_audio
                note_instances_stringdir = os.path.join(note_instances_dir+str(int(Matrix[string,fret])), 'string'+str(string+1))
                os.makedirs(note_instances_stringdir, exist_ok=True)
                dest_path = os.path.join(note_instances_stringdir, str(fret))+'.wav'
                sf.write(dest_path, note_audio, constants.sampling_rate)

                # Handle txt file creation and replacing .wav with .txt
                txt_filename = os.path.splitext(os.path.basename(dest_path))[0] + '.txt'
                txt_file_stringdir = os.path.join(onset_instances_dir+str(int(Matrix[string,fret])), 'string'+str(string+1))
                os.makedirs(txt_file_stringdir, exist_ok=True)
                dest_path = os.path.join(txt_file_stringdir, str(fret))+'.txt'
                with open(dest_path, 'w') as txt_file:
                    txt_file.write("0")

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
