# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:20:47 2021

@author: user
"""

import numpy as np
from this import d
import guitarpro as gp
# import data_utils
import numpy as np
import sys
import os
from pathlib import Path
cur_path = Path('src/')
sys.path.append(str(cur_path))

from helper import printProgressBar
import librosa
import argparse
import numpy as np
import random
import soundfile as sf
import warnings
warnings.simplefilter("ignore")

n_empty = 0

def get_audio_sample(args, dataset_no, string, fret):	
	path_to_track = os.path.join(args.note_instances, 'data', 'train',
						 'guitar' + str(dataset_no),
						 'string' + str(string + 1),  
						 str(fret) + '.wav')
	# print('path_to_track', path_to_track)

	try:    
		rec_audio, _ = librosa.load(path_to_track, sr=args.sampling_rate)
		# print("Found note evenet for: strig " + str(string) + ", fret " + str(fret) + ", dataset_no " + str(dataset_no))

	except FileNotFoundError as e:
		# print("Did not found note evenet for: strig " + str(string) + ", fret " + str(fret) + ", dataset_no " + str(dataset_no))
		return None

	return rec_audio


def parse_isolated_note_recordings(args):
	recs = []

	recs = np.empty((6, args.max_fret, args.n_samples), dtype=object)

	for string in range(6):
		for fret in range(args.max_fret):
			for dataset_no in range(args.n_samples):
				# Replace 'None' with your actual array/signal generation logic
				audio_sample = get_audio_sample(args, dataset_no+1, string, fret)
				# Check if the result is None, and if so, assign None to the array
				if audio_sample is None:
					recs[string, fret, dataset_no] = None
				else:
					# If it's not None, assign the result to the array
					recs[string, fret, dataset_no] =  audio_sample
	if np.all(recs == None):
		print("The 'recs' array is full of None values. Didn't found any note samples to parse! Check path again.")
		exit(0)
	print(recs.shape)

	return recs

def render_beat_to_audio(args, beat, dur_samples, recs):
	artif_multichannel_event= np.zeros([6,dur_samples])
	global n_empty
	
	for n in beat.notes: # up to 6 different 'n's
		n_empty=0
		string = 6 - n.string
		fret = n.value	

		if fret > args.max_fret:
			pad = dur_samples
			artif_instance = np.zeros(pad)
			artif_multichannel_event[string,:] = artif_instance
			continue

		c=0
		for element in recs[string,fret,:]:
			if element is None:
				# if c!=0:
				# 	print('No. samples ', c)
				break
			c+=1

		rand = random.choice(range(c))
		rec = recs[string][fret][rand]
  
		scaling_factor = np.random.uniform(0.5, 1.0)
		rec = rec * scaling_factor
  
		lim = min(len(rec), dur_samples)
		pad = max(0, dur_samples - len(rec))

		artif_instance = np.append( rec[:lim] , np.zeros(pad) ) # midi duration might be longer that note instance recording
		artif_multichannel_event[string,:] = artif_instance

	if (artif_multichannel_event == np.zeros([6,dur_samples])).all():
		n_empty+=1
		if n_empty > random.choice([0,0,1,1,2,3]): # (with slight randomness in length) chop out long empty sections
			artif_multichannel_event = np.empty([6,0])

	return artif_multichannel_event

if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('--note_instances', type=str)
	parser.add_argument('--input_dir', type=str, default='gp_token_examples', help='gp_token_examples or DadaGP-v1.1')
	parser.add_argument('--out_dir', type=str, default='adgp-mic', help= 'adgp-mic or adgp-pckp')
	parser.add_argument('--max_fret', type=int, default=18)
	parser.add_argument('--n_samples', type=int, default=100)
	parser.add_argument('--sampling_rate', type=int, default=44100)
	args = parser.parse_args()

	np.random.seed(0) 
	data_percentage=0.03


	# prepare
	main_folder = args.input_dir
	if not os.path.isdir(main_folder):
		print("Input dir" + args.input_dir + "does not exist.")
		exit(1)
	
	print('Loading Note Instances...')
	recs = parse_isolated_note_recordings(args)
	# print('recs', recs.shape)

	level_A_folders = os.listdir(main_folder)

	max_pitch = -1
	min_pitch = 1000

	for level_A_folder in level_A_folders:	# 1/,2/ .., A/, B/ ...
		level_A_path = os.path.join(main_folder, level_A_folder)
		if os.path.isdir( level_A_path ):
			pieces_events = []
			level_B_folders = os.listdir( level_A_path )
			for level_B_folder in level_B_folders:	# artist/
				level_B_path = os.path.join( level_A_path, level_B_folder )
				# for file in os.listdir( level_B_path ):	# *.gp3 and *.gp3.tokens.txt
				files = [f for f in os.listdir(level_B_path) if f.lower().endswith(('.gp3', '.gp4', '.gp5', '.gpx'))]
				for f_idx, file in enumerate(files, start=1):
					print(f"Processing: {level_A_folder}/{level_B_folder}/{file}")
					artif_multichannel_track = np.empty([6,0]) #[None]*6
					if file[-4:-1] == '.gp':
						
						if args.input_dir=="DadaGP-v1.1" and np.random.rand() > data_percentage:
							continue
						try:	
							song = gp.parse(os.path.join(level_B_path,file))
						except:
							continue
							
						tempo_bps = song.tempo / 60

						beat_ticks=960
						for t_id, track in enumerate(song.tracks):
							measures = track.measures
							strings = track.strings
							# check if proper guitar tunning
							proper_guitar = True
							proper_tunning = [64, 59, 55, 50, 45, 40] # make static
							for i, s in enumerate(strings):
								if i >= len(proper_tunning) or s.value != proper_tunning[i]:
									proper_guitar = False
									aborted = True
									break
							if not proper_guitar:
								continue
							if t_id!=0:
								continue
							for m, measure in enumerate(measures):
								printProgressBar(m+1,len(measures),decimals=0, length=50)
								voices = measure.voices
								for voice in voices:
									beats = voice.beats
									for b, beat in enumerate(beats): 
										duration = beat.duration.time
										dur_sec = duration / (tempo_bps * beat_ticks)
										dur_samples = int(args.sampling_rate * dur_sec)
										try:
											artif_multichannel_event = render_beat_to_audio(args, beat, dur_samples, recs)
										except Exception as e:# TypeError IndexError: # if no not-instace (see, sample) exists then ingore!
											# print('[no sample found at all]', e)
											continue

										artif_multichannel_track = np.append(artif_multichannel_track, artif_multichannel_event, axis=1)

							try:
								os.makedirs(args.out_file+'/'+file[:-4]+'_track'+str(t_id)+'/')
							except:
								continue

							try: 
								artif_mono_track = librosa.to_mono(artif_multichannel_track)
								artif_mono_track = librosa.util.normalize(artif_mono_track)	
							except:
								continue
							
							sf.write(args.out_file+'/'+file[:-4]+'_track'+str(t_id)+'/'+'E.wav', artif_multichannel_track[0,:], args.sampling_rate, 'PCM_16')
							sf.write(args.out_file+'/'+file[:-4]+'_track'+str(t_id)+'/'+'A.wav', artif_multichannel_track[1,:], args.sampling_rate, 'PCM_16')
							sf.write(args.out_file+'/'+file[:-4]+'_track'+str(t_id)+'/'+'D.wav', artif_multichannel_track[2,:], args.sampling_rate, 'PCM_16')
							sf.write(args.out_file+'/'+file[:-4]+'_track'+str(t_id)+'/'+'G.wav', artif_multichannel_track[3,:], args.sampling_rate, 'PCM_16')
							sf.write(args.out_file+'/'+file[:-4]+'_track'+str(t_id)+'/'+'B.wav', artif_multichannel_track[4,:], args.sampling_rate, 'PCM_16')
							sf.write(args.out_file+'/'+file[:-4]+'_track'+str(t_id)+'/'+'e.wav', artif_multichannel_track[5,:], args.sampling_rate, 'PCM_16')					
							sf.write(args.out_file+'/'+file[:-4]+'_track'+str(t_id)+'/'+'mixture.wav', artif_mono_track.T, args.sampling_rate, 'PCM_16')

							artif_multichannel_track = np.empty([6,0]) # initialization, see [None]*6

