import librosa
import soundfile as sf
import os
import shutil
import csv
import numpy as np

path = 'datasep-mix/'

try:
    os.makedirs(path)
except Exception as e:
    print('[Caught Error] GootToKnow:', e)
    shutil.rmtree(path)
    os.makedirs(path)

with open('NMFtestSet.csv', newline='') as csvfile:
    testreader = csv.reader(csvfile, delimiter=',')
    testfiles = ['_'.join(row[4].split('_',2)[:2]) for row in testreader] # e.g. [00_Funk1-114-Ab, ...]

print(testfiles)

in_filepath = 'data/mix/'
for gt_filename in os.listdir('data/audio_hex-pickup_debleeded'):
    guitarist = gt_filename.split('_')[0]
    gt_filepath = 'data/audio_hex-pickup_debleeded/'+gt_filename
    audio_gt, _ = librosa.load(gt_filepath, mono=False, sr=44100) # 6-channel audiofile

    # Load and Normalize input from monophonic pickup (see audio_mono-pickup_mix/)
    in_filepath = 'data/mix/'+gt_filename[:-11]+'mix.wav'
    audio_in , _ = librosa.load(in_filepath, mono=True, sr=44100) 
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


    # guitarist-wise splitting
    # dir_to_store = path+'/guitarist_'+guitarist+'/' + '_'.join(gt_filename.split('_')[:-2])
    # os.makedirs(dir_to_store)
    # sf.write(dir_to_store+'/E.wav', audio_gt[0,:], 44100, 'PCM_16')
    # sf.write(dir_to_store+'/A.wav', audio_gt[1,:], 44100, 'PCM_16')
    # sf.write(dir_to_store+'/D.wav', audio_gt[2,:], 44100, 'PCM_16')
    # sf.write(dir_to_store+'/G.wav', audio_gt[3,:], 44100, 'PCM_16')
    # sf.write(dir_to_store+'/B.wav', audio_gt[4,:], 44100, 'PCM_16')
    # sf.write(dir_to_store+'/e.wav', audio_gt[5,:], 44100, 'PCM_16')
    # sf.write(dir_to_store+'/mixture.wav', audio_in, 44100, 'PCM_16')
