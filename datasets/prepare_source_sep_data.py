import librosa
import soundfile as sf
import os
import shutil
import csv
import numpy as np
from tqdm import tqdm  # Importing tqdm for progress bar


path = 'datasep/'

try:
    os.makedirs(path)
except Exception as e:
    print('[Caught Error] GootToKnow:', e)
    shutil.rmtree(path)
    os.makedirs(path)

with open('NMFtestSet.csv', newline='') as csvfile:
    testreader = csv.reader(csvfile, delimiter=',')
    testfiles = ['_'.join(row[4].split('_',2)[:2]) for row in testreader] # e.g. [00_Funk1-114-Ab, ...]

# print(testfiles)
print("Train-Test splitting and storring to ./datasets/datasep")

# for gt_filename in os.listdir('GuitarSet/data/hex_cln'):
for gt_filename in tqdm(os.listdir('GuitarSet/data/hex_cln'), desc="Processing files", unit="file"):
    
    guitarist = gt_filename.split('_')[0]
    gt_filepath = 'GuitarSet/data/hex_cln/'+gt_filename
    audio_gt, _ = librosa.load(gt_filepath, mono=False, sr=44100) # 6-channel audiofile

    # Create and Normalize input mixture
    audio_in = librosa.to_mono(audio_gt)   
    audio_in = librosa.util.normalize(audio_in)	

    # sevaitayte splitting
    if '_'.join(gt_filename.split('_',2)[:2]) in testfiles:
        dir_to_store = path+'/test/' + '_'.join(gt_filename.split('_')[:-2])
    else:
        dir_to_store = path+'/train/' + '_'.join(gt_filename.split('_')[:-2])

    os.makedirs(dir_to_store)
    sf.write(dir_to_store+'/E.wav', audio_gt[0,:], 44100, 'PCM_16')
    sf.write(dir_to_store+'/A.wav', audio_gt[1,:], 44100, 'PCM_16')
    sf.write(dir_to_store+'/D.wav', audio_gt[2,:], 44100, 'PCM_16')
    sf.write(dir_to_store+'/G.wav', audio_gt[3,:], 44100, 'PCM_16')
    sf.write(dir_to_store+'/B.wav', audio_gt[4,:], 44100, 'PCM_16')
    sf.write(dir_to_store+'/e.wav', audio_gt[5,:], 44100, 'PCM_16')
    sf.write(dir_to_store+'/mixture.wav', audio_in, 44100, 'PCM_16')

