from tkinter import E
import museval
from tqdm import tqdm

import numpy as np
import torch
from torchmetrics import ScaleInvariantSignalDistortionRatio

import data.utils
import model.utils as model_utils
import utils
import jams
from model import Metrics
from torch.nn import functional as F
import copy
import librosa
from tensorflow.keras.utils import to_categorical


def compute_model_output(model, inputs, x_cqt=None, task='separation'):
    '''
    Computes outputs of model with given inputs. Does NOT allow propagating gradients! See compute_loss for training.
    Procedure depends on whether we have one model for each source or not
    :param model: Model to train with
    :param compute_grad: Whether to compute gradients
    :return: Model outputs, Average loss over batch
    '''
    all_sep_outputs = {}
    all_tab_outputs = {}
    aggr_tab_outputs = {}
    if model.separate:

        inputs = inputs.cuda() 
        if x_cqt is not None:
            x_cqt = x_cqt.cuda()

        output, aggr_tab_out = model(inputs, x_cqt, inst=None) # aggr_tab_out: N x 21 x 6 x T     

        if x_cqt is not None:
            x_cqt = x_cqt.cuda()

        inputs = inputs.cuda() 

        for i, inst in enumerate(model.instruments):

            if task in ['separation', 'multitask']:
                all_sep_outputs[inst] = output[inst]['output'].detach().cpu().clone()

            if task in ['tablature', 'multitask']:
                try: # if TabCNN in args.tab_version
                    all_tab_outputs[inst] = aggr_tab_out[:, :, i, :].detach().clone()
                except TypeError as e: # if not TabCNN
                    # print('[gb] warning: not aggregate test (probably OK if opted)', e)
                    all_tab_outputs[inst] = output[inst]['tab_pred'].detach().clone()
                    all_tab_outputs[inst] = F.softmax(all_tab_outputs[inst], dim=1) # NOTE: this doesn't really change anything

    return all_sep_outputs, all_tab_outputs

def predict(audio, model, args, x_cqt=None):
    '''
    Predict sources for a given audio input signal, with a given model. Audio is split into chunks to make predictions on each chunk before they are concatenated.
    :param audio: Audio input tensor, either Pytorch tensor or numpy array
    :param model: Pytorch model
    :return: Source predictions, dictionary with source names as keys
    '''
    if isinstance(audio, torch.Tensor):
        is_cuda = audio.is_cuda()
        audio = audio.detach().cpu().numpy()
        return_mode = "pytorch"
    else:
        return_mode = "numpy"

    expected_outputs = audio.shape[1]

    # Pad input if it is not divisible in length by the frame shift number
    output_shift = model.shapes["output_frames"]
    pad_back = audio.shape[1] % output_shift
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0,0), (0, pad_back)], mode="constant", constant_values=0.0)

    # #### NOTE: check again (cqt padding in general!) ####
    if pad_back > 0:
        pad_frames_back = int (pad_back / 1024)
        print('pad_frames_back', pad_frames_back)
        mod = x_cqt.shape[1] % args.fakeframes_n
        pad_frames_back = args.fakeframes_n - mod
        print('pad_frames_back', pad_frames_back)
        print('x_cqt', x_cqt.shape)
        
        x_cqt_frontpad = np.pad(x_cqt, [(0,0), (0, pad_frames_back)], mode="constant", constant_values=0.0)


    target_outputs = audio.shape[1]
    tabgt_outputs = x_cqt_frontpad.shape[1]
    fakeframes_n = args.fakeframes_n

    
    sep_outputs = {key: np.zeros(audio.shape, np.float32) for key in model.instruments}
    # tab_outputs = {key: np.zeros((21, audio.shape[1]), np.float32) for key in model.instruments}
    tab_outputs = {key: np.zeros((21, x_cqt_frontpad.shape[1]), np.float32) for key in model.instruments}
    resampled_tab_outputs = {key: np.zeros((21,0)) for key in model.instruments}

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"]
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    audio = np.pad(audio, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)
    
    x_cqt_pad = np.pad(x_cqt_frontpad, [(0,0), (4, 4)], mode="constant", constant_values=0.0) # NEW!

    findices = [0] 

    # Iterate over mixture magnitudes, fetch network prediction
    with torch.no_grad():
 
        for tab_start_pos, target_start_pos in zip(range(0, tabgt_outputs, fakeframes_n), range(0, target_outputs, model.shapes["output_frames"]) ):

            # Prepare mixture excerpt by selecting time interval
            curr_input = audio[:, target_start_pos:target_start_pos + model.shapes["input_frames"]] # Since audio was front-padded input of [targetpos:targetpos+inputframes] actually predicts [targetpos:targetpos+outputframes] target range
            original_length = curr_input.shape[1] 

            if args.task in ['tablature', 'multitask']:# and 'TabCNN' in args.tab_version:

                x_cqt_section = x_cqt_pad[:, tab_start_pos:tab_start_pos+fakeframes_n+8]
                x_cqt_section = x_cqt_section.astype(np.float32)

                original_frames = x_cqt_pad.shape[1]
                target_frames = fakeframes_n+8 # 346 --> 354

                x_cqt_section = torch.from_numpy(x_cqt_section).unsqueeze(0)               
            else:
                x_cqt_section=None 

            # Convert to Pytorch tensor for model prediction
            curr_input = torch.from_numpy(curr_input).unsqueeze(0)


            # Predict source
            if args.task in ['separation', 'multitask']:
                # ******************************************************************************* #                
                for key, curr_targets in compute_model_output(model, curr_input, task=args.task)[0].items(): # [0] because: return all_sep_outputs, all_tab_outputs
                # ******************************************************************************* #                
                    sep_outputs[key][:,target_start_pos:target_start_pos+model.shapes["output_frames"]] = curr_targets.squeeze(0).cpu().numpy() 

            if args.task in ['tablature', 'multitask']: # and args.tab_version!='TabCNN':                        
            
                indices = target_start_pos + np.linspace(0, model.shapes["output_frames"] - 1, fakeframes_n, endpoint=True).astype(int)
                findices += list(indices[1:])
                  
                # ******************************************************************************* #                
                for key, curr_targets in compute_model_output(model, curr_input, x_cqt_section, task=args.task)[1].items():  

                    tab_outputs[key][:,tab_start_pos:tab_start_pos+fakeframes_n] = curr_targets.squeeze(0).cpu().numpy()
                    resampled_tab_outputs[key] = tab_outputs[key] # NOTE: provisional


    # Crop to expected length (since we padded to handle the frame shift)
    sep_outputs = {key : sep_outputs[key][:,:expected_outputs] for key in sep_outputs.keys()}

    if return_mode == "pytorch":
        sep_outputs = torch.from_numpy(sep_outputs)
        if is_cuda:
            sep_outputs = sep_outputs.cuda()

    if return_mode == "pytorch": 
        resampled_tab_outputs = torch.from_numpy(resampled_tab_outputs)
        if is_cuda:
            resampled_tab_outputs = resampled_tab_outputs.cuda()

    return sep_outputs, resampled_tab_outputs, findices


############## FROM TABCNN REPO ###############
def preprocess_audio(data):
    data = data.astype(float)
    data = librosa.util.normalize(data)
    data = librosa.resample(data, 44100, 22050)
    data = np.abs(librosa.cqt(data,
        hop_length=512, 
        sr=22050, 
        n_bins=192, 
        bins_per_octave=24))
    return data

def correct_numbering(n):
    highest_fret=19
    n += 1
    if n < 0 or n > highest_fret:
        n = 0
    return n

def categorical(label):
    num_classes=21
    return to_categorical(label, num_classes)

def clean_label( label):
    label = [correct_numbering(n) for n in label]
    return categorical(label)

def clean_labels(labels):
    return np.array([clean_label(label) for label in labels])
#############################################

def load_tabcnn_jams(args, mix_path, cqt_times, frame_indices):
    track_name = mix_path.split('/')[-2]
    file_anno = "data-tab/GuitarSet/annotation/" + track_name +".jams"

    string_midi_pitches = [40,45,50,55,59,64]

    jam = jams.load(file_anno)              

    # Define the number of frets
    num_frets = 21  # Including open string as a 'fret'

    labels = []
    for string_num, string_name in enumerate(args.instruments):
        anno = jam.annotations["note_midi"][string_num]
        string_label_samples = anno.to_samples(cqt_times)
        # replace midi pitch values with fret numbers
        for i in frame_indices:
            if string_label_samples[i] == []:
                string_label_samples[i] = -1
            else:
                string_label_samples[i] = int(round(string_label_samples[i][0]) - string_midi_pitches[string_num])
        labels.append([string_label_samples])

    labels = np.array(labels)
    labels = np.squeeze(labels)
    labels = np.swapaxes(labels,0,1)                                
    # clean labels
    labels = clean_labels(labels)
    labels = np.transpose(labels, (1, 2, 0)) # N x 6 x 21 --> 6 x 21 x N

    labels_frames_dict = {}
    for string_num, string_name in enumerate(args.instruments):
        labels_frames_dict[string_name] = labels[string_num, :, :]#.long()

    return labels_frames_dict

def load_jams(args, mix_path, mix_audio):
    track_name = mix_path.split('/')[-2]
    file_anno = "data-tab/GuitarSet/annotation/" + track_name +".jams"

    string_midi_pitches = [40,45,50,55,59,64]

    jam = jams.load(file_anno)              

    # Define the number of frets
    num_frets = 21  # Including open string as a 'fret'

    # Initialize a dictionary to hold labels for each string name
    labels_samples_dict = {}

    for string_num, string_name in enumerate(args.instruments):
        ann = jam.annotations["note_midi"][string_num]

        # Initialize the label array for this string with -1
        string_sample_labels = np.full((num_frets, mix_audio.shape[1]), 0)
        string_sample_labels[0,:]=1

        # Process each observation
        for obs in ann:
            # Calculate the sample index corresponding to the observation time
            start_sample_index = round(obs.time * args.sr)
            end_sample_index = round((obs.time + obs.duration) * args.sr)  # Add duration to the start time

            fret_number = round(obs.value) - string_midi_pitches[string_num]           

            if 0 <= fret_number < num_frets:
                string_sample_labels[fret_number, start_sample_index:end_sample_index] = 1                        
                string_sample_labels[0, start_sample_index:end_sample_index] = 0  # Marking the fret as played at this time
            else:
                print(f"Warning: Fret number {fret_number} out of range for string {string_name}")

        # Assign the label array to the corresponding string name in the dictionary
        labels_samples_dict[string_name] = string_sample_labels

    return labels_samples_dict

def predict_song(args, audio_path, model):
    '''
    Predicts sources for an audio file for which the file path is given, using a given model.
    Takes care of resampling the input audio to the models sampling rate and resampling predictions back to input sampling rate.
    :param args: Options dictionary
    :param audio_path: Path to mixture audio file
    :param model: Pytorch model
    :return: Source estimates given as dictionary with keys as source names
    '''
    model.eval()

    resampled_labels_dict=None

    # Load mixture in original sampling rate
    mix_audio, mix_sr = data.utils.load(audio_path, sr=None, mono=False)
    mix_channels = mix_audio.shape[0]
    mix_len = mix_audio.shape[1]

    # Adapt mixture channels to required input channels
    if args.channels == 1:
        mix_audio = np.mean(mix_audio, axis=0, keepdims=True)
    else:
        if mix_channels == 1: # Duplicate channels if input is mono but model is stereo
            mix_audio = np.tile(mix_audio, [args.channels, 1])
        else:
            assert(mix_channels == args.channels)

    # resample to model sampling rate
    mix_audio = data.utils.resample(mix_audio, mix_sr, args.sr)


    # if "TabCNN" in args.tab_version:
    vanilla_audio = copy.deepcopy(mix_audio[0])
    x_cqt = np.swapaxes(preprocess_audio(vanilla_audio),0,1) # (N, 192)
    frame_indices = range(len(x_cqt))
    cqt_times = librosa.frames_to_time(frame_indices, sr=22050, hop_length=512)
    x_cqt = np.swapaxes(x_cqt, 0, 1)

    # ************************************************************************************* #
    sources, tab_output, findices = predict(mix_audio, model, args, x_cqt) # tab_output (21, 4188)
    # ************************************************************************************* #

    if args.task in ['tablature', 'multitask']:                        

        # Get labels and resample them __gbastas__
        np.set_printoptions(threshold=np.inf)
        resampled_labels_dict = {}   
        mix_audio, mix_sr = data.utils.load(audio_path, sr=None, mono=False)
        
        
        ###########NOTE#####################
        # labels_samples_dict = load_jams(args, audio_path, mix_audio)  21
        # # print('findices', findices)
        
        # for key in labels_samples_dict.keys():
        #     # print('resampled_labels_dict', labels_samples_dict[key][0,786063]) 
        #     # print('labels_samples_dict', np.where(labels_samples_dict[key][0,:]==0)[0] )
        #     new_indices = [x for x in findices if x < labels_samples_dict[key].shape[1]] # NOTE: to avoid 'IndexError: index 984693 is out of bounds for axis 1 with size 984505'. Is it good though???
        #     resampled_labels_dict[key] = labels_samples_dict[key][:,new_indices]
        #     depadded_lngth = resampled_labels_dict[key].shape[1]
        #     tab_output[key] = tab_output[key][:,:depadded_lngth]
        #     # print('resampled_labels_dict', np.where(resampled_labels_dict[key][0,:]==0)[0].shape )
        #     # print('resampled_labels_dict', resampled_labels_dict[key][0,:]) 
        ###########NOTE#####################

        # NOTE: provisionl testing
        resampled_labels_dict = load_tabcnn_jams(args, audio_path, cqt_times, frame_indices)#.long()  21

    if args.task in ['separation', 'multitask']:

        # Resample back to mixture sampling rate in case we had model on different sampling rate
        sources = {key : data.utils.resample(sources[key], args.sr, mix_sr) for key in sources.keys()}

        # In case we had to pad the mixture at the end, or we have a few samples too many due to inconsistent down- and upsamṕling, remove those samples from source prediction now
        for key in sources.keys():
            diff = sources[key].shape[1] - mix_len
            if diff > 0:
                print("WARNING: Cropping " + str(diff) + " samples")
                sources[key] = sources[key][:, :-diff]
            elif diff < 0:
                print("WARNING: Padding output by " + str(diff) + " samples")
                sources[key] = np.pad(sources[key], [(0,0), (0, -diff)], "constant", 0.0)

            # Adapt channels
            if mix_channels > args.channels:
                assert(args.channels == 1)
                # Duplicate mono predictions
                sources[key] = np.tile(sources[key], [mix_channels, 1])
            elif mix_channels < args.channels:
                assert(mix_channels == 1)
                # Reduce model output to mono
                sources[key] = np.mean(sources[key], axis=0, keepdims=True)

            sources[key] = np.asfortranarray(sources[key]) # So librosa does not complain if we want to save it

    return sources, tab_output, resampled_labels_dict

def evaluate(args, dataset, model, instruments):
    '''
    Evaluates a given model on a given dataset
    :param args: Options dict
    :param dataset: Dataset object
    :param model: Pytorch model
    :param instruments: List of source names
    :return: Performance metric dictionary, list with each element describing one dataset sample's results
    '''
    perfs = list()
    perfs_comp = list()
    perfs_solo = list()
    model.eval()
    PP = list()
    PR = list()
    PF = list()
    TP = list()
    TR = list()
    TF = list()
    TDR = list()
    with torch.no_grad():
        Y_preds = []
        Y_gts = []        
        # for example in dataset:
        for example_num, example in enumerate(dataset):
            
            print("Evaluating " + example["mix"])

            # Load source references in their original sr and channel number
            target_sources = np.stack([data.utils.load(example[instrument], sr=None, mono=False)[0].T for instrument in instruments])

            ############ Predict using mixture #################
            pred_sources, pred_tab, tab_labels = predict_song(args, example["mix"], model)
            # ************************************************ #

            pred_sources = np.stack([pred_sources[key].T for key in instruments])

            # Evaluate
            if args.task in ['separation', 'multitask']:
                try:
                    SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(target_sources, pred_sources)
                except ValueError as e: # arises with mic-augmentation data
                    print(e)
                    continue


                si_sdr_metric = ScaleInvariantSignalDistortionRatio()

                si_sdr_values = []
                for i in range(len(instruments)):
                    si_sdr_value = si_sdr_metric(
                        torch.tensor(pred_sources[i]),
                        torch.tensor(target_sources[i])
                    ).item()
                    si_sdr_values.append(si_sdr_value)

                song = {}
                for idx, name in enumerate(instruments):
                    song[name] = {"SDR" : SDR[idx], "ISR" : ISR[idx], "SIR" : SIR[idx], "SAR" : SAR[idx], "SI-SDR": si_sdr_values[idx]}
                perfs.append(song)

                if '_comp' in example['mix']:
                    perfs_comp.append(song)

                if '_solo' in example['mix']:
                    perfs_solo.append(song)
                    

            if args.task in ['tablature', 'multitask']:                        
                y_preds, y_gts = [], []
                for key in pred_tab.keys():
                    y_preds.append(pred_tab[key])
                    y_gts.append(tab_labels[key])

                y_preds = np.array(y_preds)    # (6, 21, 78760)
                y_preds = np.transpose(y_preds, (2, 0, 1)) # (78760, 6, 21)

                y_gts = np.array(y_gts)  # (6, 21, 78760)
                y_gts = np.transpose(y_gts, (2, 0, 1)) # (78760, 6, 21)

                extra = y_preds.shape[0] - y_gts.shape[0]
                y_preds = y_preds[:-extra]
                Y_preds.append(y_preds)
                Y_gts.append(y_gts)

                print_tablature(y_preds[:])
                print_tablature(y_gts[:])      

                pp = Metrics.pitch_precision(y_preds, y_gts)
                pr = Metrics.pitch_recall(y_preds, y_gts)
                pf = Metrics.pitch_f_measure(y_preds, y_gts)
                tp = Metrics.tab_precision(y_preds, y_gts)
                tr = Metrics.tab_recall(y_preds, y_gts)
                tf = Metrics.tab_f_measure(y_preds, y_gts)
                tdr = Metrics.tab_disamb(y_preds, y_gts)

                PP.append(pp)
                PR.append(pr)
                PF.append(pf)
                TP.append(tp)
                TR.append(tr)
                TF.append(tf)
                TDR.append(tdr)            
                print('tf', round(tf,2))


        if args.task in ['tablature', 'multitask']:                        

            Y_preds = np.concatenate(Y_preds)
            Y_gts = np.concatenate(Y_gts)

            print('y_preds', y_preds.shape)
            print('y_gts', y_gts.shape) 

            print('Y_preds', Y_preds.shape)
            print('Y_gts', Y_gts.shape)                

            PP = Metrics.pitch_precision(Y_preds, Y_gts)
            PR = Metrics.pitch_recall(Y_preds, Y_gts)
            PF = Metrics.pitch_f_measure(Y_preds, Y_gts)
            TP = Metrics.tab_precision(Y_preds, Y_gts)
            TR = Metrics.tab_recall(Y_preds, Y_gts)
            TF = Metrics.tab_f_measure(Y_preds, Y_gts)
            TDR = Metrics.tab_disamb(Y_preds, Y_gts)


            print('PP', round(np.mean(PP), 3))
            print('PR', round(np.mean(PR), 3))
            print('PF', round(np.mean(PF), 3))
            print('TP', round(np.mean(TP), 3))
            print('TR', round(np.mean(TR), 3))
            print('TF', round(np.mean(TF), 3))
            print('TDR', round(np.mean(TDR), 3))

            return perfs, perfs_comp, perfs_solo   

        if args.task in ['separation', 'multitask']:
            return perfs, perfs_comp, perfs_solo   



def validate(args, model, criterion, tab_criterion, test_data):
    '''
    Iterate with a given model over a given test dataset and compute the desired loss
    :param args: Options dictionary
    :param model: Pytorch model
    :param criterion: Loss function to use (similar to Pytorch criterions)
    :param test_data: Test dataset (Pytorch dataset)
    :return:
    '''
    # PREPARE DATA
    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    # VALIDATE
    model.eval()
    total_loss = 0.
    total_tab_loss = 0.
    with tqdm(total=len(test_data) // args.batch_size) as pbar, torch.no_grad():
        # for example_num, (x, targets) in enumerate(dataloader):
        for example_num, (x, targets, tab_labels, cqt) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                if args.task in ['separation', 'multitask']:
                    for k in list(targets.keys()):
                        targets[k] = targets[k].cuda()
                if args.task in ['tablature', 'multitask']:                        
                    cqt = cqt.cuda()
                    for k in list(tab_labels.keys()):
                        tab_labels[k] = tab_labels[k].long().cuda()

            _, avg_loss, avg_tab_loss, avg_tab_acc = model_utils.compute_loss(model, x, targets, tab_labels, cqt, criterion, tab_criterion, task=args.task, tab_version=args.tab_version)
            total_loss += (1. / float(example_num + 1)) * (avg_loss - total_loss)
            total_tab_loss += (1. / float(example_num + 1)) * (avg_tab_loss - total_tab_loss)


            description = "Current loss: {:.4f}, Current tab loss: {:.4f}, Current tab Acc: {:.4f}".format(
                total_loss, total_tab_loss, avg_tab_acc
            )
            pbar.set_description(description)
            pbar.update(1)

    return total_loss, total_tab_loss, avg_tab_acc

def print_tablature(tab_array_N_6_21, max_chars_per_line=180):
    # Initialize the tablature lines
    tab_array_N_6 = np.array(list(map(tab2bin,tab_array_N_6_21)))
    maxnum =100
    tab = tab_array_N_6
    tab_lines = ['-' * (len(tab[:maxnum,:]) * 3) for _ in range(6)]

    # Fill in the tablature lines with the fret numbers
    for i in range(len(tab[:maxnum,:])):
        for j in range(6):

            if tab[i, j] != -1:
                tab_lines[j]= tab_lines[j][:i*3] +'-' + str(int(tab[i, j])) + '-' + tab_lines[j][i*3+3:]
    for j in range(6):
        print(tab_lines[j])
        
    print()

        
def tab2bin(tab):
    tab_arr = np.zeros(6)
    for string_num in range(len(tab)):
        fret_vector = tab[string_num]
        fret_class = np.argmax(fret_vector, -1)
        # 0 means that the string is closed 
        fret_num = fret_class - 1
        tab_arr[string_num] = fret_num 
    return tab_arr        