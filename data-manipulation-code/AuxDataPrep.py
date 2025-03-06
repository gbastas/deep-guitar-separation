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


    if args.action in ['gather_notes', 'pseudo_sep', 'pseudocomp_sep']:
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

    for count, name in enumerate(lines): # iterate over filenames
        
        name = name.replace('\n', '')         # e.g. '02_SS2-88-F_solo.jams'
        print('testfile', name)
        annosfilepath = os.path.join(constants.annos_path, name)

        # if constants.verbose:
        #     print()
        #     print('Audio-based detection running...')
        #     print(name, count,'/',len(lines))
        # else:
        printProgressBar(count,len(lines),decimals=0, length=50)

        tablature=None
        audiofilepath = constants.track_path + name[:-5] + '_' + constants.dataset +'.wav' # TODO: set dataset to either mix or mic in constants.ini
        annotations = demo_utils.read_tablature_from_GuitarSet(annosfilepath, constants)   
        annos_tab_list = annotations.tablature.tabList

        annos_pitches = [instance.fundamental for instance in annos_tab_list]

        audio, _ = librosa.load(audiofilepath, sr=constants.sampling_rate) 

        # if args.pred_onset:
        #     test_onsets = demo_utils.get_onsets(audiofilepath, constants)
        # else:
        test_onsets = [tab_element.onset for tab_element in annos_tab_list]
        test_offsets = [tab_element.offset for tab_element in annos_tab_list]


        test_strings = [tab_element.string for tab_element in annos_tab_list]
        test_frets = [tab_element.fret for tab_element in annos_tab_list]

        print('Ongoing Pitch-Fret-String Estimation...') # __new__

        # to get the note audio instances:
        tablature = Tablature(test_onsets, test_offsets, audio, constants)

        if args.action == 'pseudo_sep':
            if args.all_solos:
                dest_path = './pseudo_sep_all_solos_'+constants.dataset+'_wn/'+name[:-5]+'_hex_'+constants.dataset+'/'
            else:
                dest_path = './pseudo_sep_few_solos'+constants.dataset+'_wn/'+name[:-5]+'_hex_'+constants.dataset+'/'
            
            # Assuming 'audio' is already defined and you want to match its length
            len_audio = len(audio)  # Length of your audio signal
            hex_audio = np.random.normal(0, 0.00005, (6, len_audio))  # mean=0, std=0.00005          WHITE NOISE!!!
            prev_string = -1
            prev_start = 0
            start_diff=0
            for i, (fret, string, onset, offset) in enumerate(zip(test_frets, test_strings, test_onsets, test_offsets)):
                start = int(round((onset)*(constants.sampling_rate)))
                end = int(round((offset)*(constants.sampling_rate)))

                count_total_note_events+=1
                # avoid chords
                if i>0 and i<len(test_onsets)-1 and (test_onsets[i+1]-test_onsets[i]<0.06 or test_onsets[i]-test_onsets[i-1]<0.06): # (test_onsets[i+1] < offset or prev_offset > onset): 
                    # print('"chord" occurrence!')#, test_onsets[i+1]-test_onsets[i], test_onsets[i]-test_onsets[i-1])
                    count_omitted_note_events+=1
                    Strings_gt_total_count[string]+=1
                    prev_offset = offset
                    start_diff += start - prev_start
                    prev_start = start
                    continue
                
                # render the same-string melody in a continuum
                if string == prev_string:
                    start = prev_start
                hex_audio[string, start:end] = audio[start:end]

                prev_string=string
                prev_start=start
                prev_end=end
                prev_offset = offset

            if args.all_tracks or args.all_comps:
                continue

            os.makedirs(dest_path, exist_ok=True)

            sf.write(dest_path+'E.wav', hex_audio[0,:], constants.sampling_rate)
            sf.write(dest_path+'A.wav', hex_audio[1,:], constants.sampling_rate)
            sf.write(dest_path+'D.wav', hex_audio[2,:], constants.sampling_rate)
            sf.write(dest_path+'G.wav', hex_audio[3,:], constants.sampling_rate)
            sf.write(dest_path+'B.wav', hex_audio[4,:], constants.sampling_rate)
            sf.write(dest_path+'e.wav', hex_audio[5,:], constants.sampling_rate)					

            hex_audio = np.sum(hex_audio, axis=0)
            sf.write(dest_path+'mixture.wav', hex_audio, constants.sampling_rate)

        if args.action == 'pseudocomp_sep':
            dest_path = './pseudocomp_sep_all_notes/'+name[:-9]+'comp_shuffl_hex_mic/'

            # hex_audio=np.zeros((6, len(audio)))
            hex_audio_comp = [[0]]*6
            prev_string = -1
            prev_start = 0
            prev_offset = 0
            for i, (fret, string, onset, offset) in enumerate(zip(test_frets, test_strings, test_onsets, test_offsets)):
                start = int(round((onset)*(constants.sampling_rate)))
                end = int(round((offset)*(constants.sampling_rate)))

                # render the same-string melody in a continuum
                if string == prev_string:
                    start = prev_start

                # avoid chords
                if i>0 and i<len(test_onsets)-1 and (test_onsets[i+1]-test_onsets[i]<0.06 or test_onsets[i]-test_onsets[i-1]<0.06): # (test_onsets[i+1] < offset or prev_offset > onset): 
                    
                # if i<len(test_onsets)-1 and test_onsets[i+1]-0.06<test_onsets[i]: #and (test_onsets[i+1]<offset or prev_offset>onset) : 
                    prev_offset = offset
                    continue

                hex_audio_comp[string] = hex_audio_comp[string] + list(audio[start:end])

                prev_string=string
                prev_start=start
                prev_offset = offset

            lengths = np.array([len(aud) for aud in hex_audio_comp])
            # print('lengths', lengths)
            second_longer = np.argsort(lengths)[-2]

            # hex_audio=np.zeros((6, lengths[second_longer]))
            len_audio = len(audio)  # Length of your audio signal
            hex_audio = np.random.normal(0, 0.00005, (6, lengths[second_longer]))  # mean=0, std=0.00005 


            for string, aud in enumerate(hex_audio_comp):
                if len(aud)<lengths[second_longer]:
                    s = random.randint(0, lengths[second_longer] - len(aud)-1)
                else:
                    s = 0
                L = min(len(aud), lengths[second_longer])
                hex_audio[string, s:L+s] = np.array(aud[:L])

            os.makedirs(dest_path, exist_ok=True)

            sf.write(dest_path+'E.wav', hex_audio[0,:], constants.sampling_rate)
            sf.write(dest_path+'A.wav', hex_audio[1,:], constants.sampling_rate)
            sf.write(dest_path+'D.wav', hex_audio[2,:], constants.sampling_rate)
            sf.write(dest_path+'G.wav', hex_audio[3,:], constants.sampling_rate)
            sf.write(dest_path+'B.wav', hex_audio[4,:], constants.sampling_rate)
            sf.write(dest_path+'e.wav', hex_audio[5,:], constants.sampling_rate)					

            hex_audio = np.sum(hex_audio, axis=0)
            sf.write(dest_path+'mixture.wav', hex_audio, constants.sampling_rate)

        if args.action == "gather_notes":
            # NOTE: To create note_instances dataset uncomment and in constants.ini set crop_win=3 and listoftracksfile = allsolos.txt
            note_instances_dir = './note_instances/'+constants.dataset+'guitar'
            for tabelement, annoselement in zip(tablature.tabList, annotations.tablature.tabList):
                note_audio = tabelement.note_audio

                # crepe_pitch
                sr = constants.sampling_rate
                (_, freqs, confs, _)  = crepe.predict(note_audio, sr=sr, viterbi=True)

                id = np.argmax(confs)
                confidence = confs[id]
                frequency = freqs[id]
                if confidence<0.8:
                    print('LOW PITCH ESTIMATION CONFIDENCE!!')
                    continue
                if len(note_audio)/constants.sampling_rate<0.08: # not less than 0.5 seconds
                    continue            

                try:
                    note = utils.hz_to_midi(frequency)
                except TypeError as e:
                    print('[MyWaring] freq:', frequency,  e)
                    continue
                
                note = librosa.midi_to_note(note)

                # Compare estimated note with annotated note (this is to gather only clean instances)
                fret = annoselement.fret
                string = annoselement.string
                if note[:-1].replace('♯','#')!= Note[string, fret]:
                    print('Notes', note[:-1].replace('♯','#'), Note[string, fret])
                    print('String-fret', string, fret)
                    continue

                Matrix[string,fret]+=1
                if Matrix[string,fret]>100:
                    continue

                note_audio = tabelement.note_audio
                
                note_instances_stringdir = os.path.join(note_instances_dir+str(int(Matrix[string,fret])), 'string'+str(string+1))
                os.makedirs(note_instances_stringdir, exist_ok=True)

                dest_path = os.path.join(note_instances_stringdir, str(fret))+'.wav'
                sf.write(dest_path, note_audio, constants.sampling_rate)

    print('Ommited ' + str(count_omitted_note_events) + ' note events out of ' + str(count_total_note_events) +'.')
    if args.all_tracks:
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

    args = parser.parse_args()

    config_path = 'constants.ini'
    workspace_folder = '../datasets/GuitarSet'

    constants = Constants(config_path, workspace_folder)    
    
    constants.dataset_names_path = workspace_folder

    GuitarSetProcessing(constants)
