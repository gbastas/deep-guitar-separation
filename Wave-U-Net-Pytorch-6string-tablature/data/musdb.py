import musdb
import os
import numpy as np
import glob
import random
from data.utils import load, write_wav


def get_musdbhq(database_path):
    '''
    Retrieve audio file paths for MUSDB HQ dataset
    :param database_path: MUSDB HQ root directory
    :return: dictionary with train and test keys, each containing list of samples, each sample containing all audio paths
    '''
    subsets = list()

    for subset in ["train", "test", "val"]:
        print("Loading " + subset + " set...")
        tracks = glob.glob(os.path.join(database_path, subset, "*"))
        samples = list()

        # Go through tracks
        for track_folder in sorted(tracks):
            # Skip track if mixture is already written, assuming this track is done already
            example = dict()
            for stem in ["mix", "E", "A", "D", "G", "B", "e"]: # __gbastas__
                filename = stem if stem != "mix" else "mixture"
                audio_path = os.path.join(track_folder, filename + ".wav")
                example[stem] = audio_path

            samples.append(example)
        subsets.append(samples)
    return subsets


def get_musdbh_crossval(database_path):
    subsets = list()

    for subset in ["guitarist_00", "guitarist_01", "guitarist_02", "guitarist_03", "guitarist_04", "guitarist_05"]:
        print("Loading " + database_path + subset + " set...")
        tracks = glob.glob(os.path.join(database_path, subset, "*"))
        samples = list()

        # Go through tracks
        for track_folder in sorted(tracks):
            # Skip track if mixture is already written, assuming this track is done already
            example = dict()
            for stem in ["mix", "E", "A", "D", "G", "B", "e"]: # __gbastas__
                filename = stem if stem != "mix" else "mixture"
                audio_path = os.path.join(track_folder, filename + ".wav")
                example[stem] = audio_path

            samples.append(example)
        subsets.append(samples)
    return subsets


def get_musdb_folds(root_path, version="HQ", guitID=None):
    val_list_mic, val_list_mix, val_list_hex_cln = [], [], []
    
    if version == "HQ" or version == "HQ-comp" or version == "HQ-solo" or version == 'mic' or version == 'pseudo'  or version == 'fake':
        dataset = get_musdbhq(root_path)
    elif version == "cross-val":
        print("**********CROSS_VALIDATION***********")
        dataset = get_musdbh_crossval(root_path)
    
    if 'AthenuitarSep' in root_path or 'Fake' in root_path or version == 'fake':

        train_list = dataset[0]
        test_list = dataset[1]
        val_list = dataset[2]

        return {"train" : train_list, "val" : val_list, "val_comp": None, "val_solo": None, "test" : test_list,
                'val_mic': None, "val_mix": None, 'val_hex_cln': None}

    elif 'GuitarSet' in root_path:
        if version == "cross-val":
            rnge = [i for i in range(0,6) if i not in [guitID, (guitID+1)%6] ]
            test_list = dataset[guitID]
            val_list = dataset[(guitID+1)%6] 
            train_list = [item for i in rnge for item in dataset[i]] 

        elif version == "HQ":
            train_val_list = dataset[0]
            test_list = dataset[1]
            # print(train_val_list, 'train_val_list')
            
            train_val_list_comp = np.array([track for track in train_val_list if '_comp' in track['mix']])
            train_val_list_solo = np.array([track for track in train_val_list if '_solo' in track['mix']])
            train_val_list_other = np.array([track for track in train_val_list if not '_solo' in track['mix'] and not '_comp' in track['mix']])

            np.random.seed(1337) # Ensure that partitioning is always the same on each run-
            train_list_comp = np.random.choice(train_val_list_comp, 116, replace=False)
            train_list_solo = np.random.choice(train_val_list_solo, 116, replace=False)
            train_list = np.append(train_list_solo, train_list_comp)
            val_list = [elem for elem in train_val_list if elem not in train_list and elem not in train_val_list_other]
            train_list = np.append(train_list, train_val_list_other) # this is to include possible augmentation 'fake' files

        elif version == "HQ-comp": # TODO integrate with if conditions to HQ block above
            train_val_list = dataset[0]
            test_list = dataset[1]
            
            train_val_list_comp = np.array([track for track in train_val_list if '_comp' in track['mix']])
            train_val_list_solo = np.array([track for track in train_val_list if '_solo' in track['mix']])

            train_val_list_other = np.array([track for track in train_val_list if not '_solo' in track['mix'] and not '_comp' in track['mix']])
            print('train_val_list_other', len(train_val_list_other))

            np.random.seed(1337) # Ensure that partitioning is always the same on each run-
            train_list = np.random.choice(train_val_list_comp, 116, replace=False)
            list_solo = np.random.choice(train_val_list_solo, 116, replace=False)

            val_list = [elem for elem in train_val_list if elem not in train_list and elem not in train_val_list_other and elem not in list_solo]

        elif version == "HQ-solo": 
            train_val_list = dataset[0]
            test_list = dataset[1]
            
            train_val_list_solo = np.array([track for track in train_val_list if '_solo' in track['mix']])
            train_val_list_comp = np.array([track for track in train_val_list if '_comp' in track['mix']])

            train_val_list_other = np.array([track for track in train_val_list if not '_solo' in track['mix'] and not '_comp' in track['mix']])

            np.random.seed(1337) # Ensure that partitioning is always the same on each run-
            train_list = np.random.choice(train_val_list_solo, 116, replace=False)
            list_comp = np.random.choice(train_val_list_comp, 116, replace=False)

            val_list = [elem for elem in train_val_list if elem not in train_list and elem not in train_val_list_other and elem not in list_comp]




        elif version in ["mic", 'pseudo']: # mic means mic-mix-hex
            train_val_list = dataset[0]
            
            test_list = dataset[1]
            
            val_list_prefices = ['00_BN2-131-B_solo', '00_BN3-154-E_solo', '00_Funk1-97-C_solo',
                                '00_Funk2-108-Eb_comp', '00_Funk2-108-Eb_solo', '00_Funk2-119-G_solo',
                                '00_Funk3-112-C#_comp', '00_Funk3-98-A_comp', '00_Jazz3-137-Eb_solo', 
                                '00_Rock2-142-D_solo', '00_Rock3-148-C_comp', '00_Rock3-148-C_solo',  
                                '00_SS1-100-C#_comp', '00_SS2-107-Ab_comp', '00_SS2-88-F_comp',       
                                '01_BN1-147-Gb_solo', '01_BN2-131-B_comp', '01_Funk1-114-Ab_solo',    
                                '01_Jazz1-130-D_solo', '01_SS1-68-E_solo', '02_BN3-119-G_comp',       
                                '02_Funk3-98-A_comp', '02_Jazz1-130-D_solo', '02_Jazz2-110-Bb_comp',  
                                '02_Rock2-142-D_comp', '02_Rock2-142-D_solo', '02_Rock3-117-Bb_comp', 
                                '02_SS2-88-F_comp', '02_SS2-88-F_solo', '03_Funk2-119-G_solo',        
                                '03_Funk3-112-C#_comp', '03_Funk3-112-C#_solo', '03_Funk3-98-A_comp', 
                                '03_Jazz1-200-B_solo', '03_Jazz2-110-Bb_comp', '03_Rock1-90-C#_comp', 
                                '03_Rock2-142-D_solo', '03_Rock3-148-C_comp', '03_SS1-100-C#_comp',   
                                '03_SS1-100-C#_solo', '03_SS1-68-E_comp', '03_SS2-107-Ab_comp',       
                                '03_SS2-88-F_solo', '04_BN2-166-Ab_solo', '04_Funk1-114-Ab_solo',     
                                '04_Funk2-108-Eb_comp', '04_Funk2-119-G_comp', '04_Jazz1-130-D_solo', 
                                '04_Jazz1-200-B_solo', '04_Jazz2-110-Bb_solo', '04_Jazz2-187-F#_comp',
                                '04_Jazz3-137-Eb_comp', '04_Rock2-142-D_comp', '04_SS2-107-Ab_solo',  
                                '04_SS3-98-C_solo', '05_Funk2-119-G_comp', '05_Funk3-98-A_solo',      
                                '05_Jazz2-110-Bb_comp', '05_Rock1-90-C#_solo', '05_Rock3-117-Bb_comp',
                                '05_SS2-107-Ab_solo', '05_SS2-88-F_comp', '05_SS2-88-F_solo',         
                                '05_SS3-84-Bb_comp'] # sampled with np.random.seed(1337)

            train_list=[]
            tmp=[]
            for track in train_val_list:
                # print('track', track)
                is_it = '_'.join(track['mix'].split('/')[-2].split('_')[:3])
                # print(is_it)
                if is_it in val_list_prefices: # check prefix (e.g. 00_BN1-129-Eb_comp) to separate between val and train
                    print(track['mix'])
                    if '_mic' in track['mix']:
                        tmp.append(is_it)
                        val_list_mic.append(track)
                    if '_mix' in track['mix']:
                        # print(track['mix'])
                        val_list_mix.append(track)
                    if '_hex_cln' in track['mix']:
                        val_list_hex_cln.append(track)
                else:
                    train_list.append(track)
            
            print('val_list_mic', len(val_list_mic))
            print('val_list_mix', len(val_list_mix))
            print('val_list_hex_cln', len(val_list_hex_cln))

            print('val mix data:', len(val_list_mix))
            print('val hex data:', len(val_list_hex_cln))

        if val_list_mic: # if not empty
            val_list = val_list_mic

        val_list_comp = np.array([track for track in val_list if 'comp' in track['mix']])
        val_list_solo = np.array([track for track in val_list if 'solo' in track['mix']])
        
        val_list_names = np.array([ track['mix'].split('/')[-2] for track in val_list ])

        # print('val_list_names:', val_list_names)
        print('train data:', len(train_list))
        print('val data:', len(val_list))

    else:
        print('[MyError] Check again the args.dataset_dir you have given!')
        exit()

    # return {"train" : train_list, "val" : val_list, "test" : test_list}
    # return {"train" : train_list, "val" : val_list, "val_comp": val_list_comp, "val_solo":val_list_solo, "test" : test_list}
    return {"train" : train_list, "val" : val_list, "val_comp": val_list_comp, "val_solo": val_list_solo, "test" : test_list,
            'val_mic': val_list_mic, "val_mix": val_list_mix, 'val_hex_cln': val_list_hex_cln}

