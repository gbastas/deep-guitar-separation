import argparse
import os
import time
from functools import partial

import torch
import pickle
import numpy as np

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

import model.utils as model_utils
import utils
from data.dataset import SeparationDataset
from data.musdb import get_musdb_folds, get_musdbhq
from data.utils import crop_targets, random_amplify
# from test import evaluate, validate
import test
from model.waveunet import Waveunet
import csv
from model import Metrics

import warnings
warnings.filterwarnings("ignore")

def main(args):

    # MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]

    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate, tab_version=args.tab_version)


    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    writer = SummaryWriter(args.log_dir)
    
    ### DATASET
    musdb = get_musdb_folds(args.dataset_dir, version=args.version, guitID=args.split)
    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop_targets, shapes=model.shapes)
    # Data augmentation function for training
    augment_func = partial(random_amplify, shapes=model.shapes, min=0.7, max=1.0)
    train_data = SeparationDataset(musdb, "train", args.instruments, args.sr, args.channels, model.shapes, True, args.hdf_dir, audio_transform=augment_func, fakeframes_n=args.fakeframes_n, resample=args.resample) # NOTE: augmentation
    val_data = SeparationDataset(musdb, "val", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func, fakeframes_n=args.fakeframes_n, resample=args.resample)


    test_data = SeparationDataset(musdb, "test", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func,fakeframes_n=args.fakeframes_n, resample=args.resample)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
        
        
    print('train_data', len(train_data))        
        
    ##### TRAINING ####

    # Set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    tab_criterion = nn.CrossEntropyLoss()
    checkpoint_other_best_path = ''

 
    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_sep_loss" : np.Inf,
             "best_tab_loss" : np.Inf,
             "best_acc" : 0,
             "best_comp_loss" : np.Inf,
             "best_solo_loss" : np.Inf,
             "best_mic_loss" : np.Inf,
             "best_mix_loss" : np.Inf,
             "best_hex_cln_loss" : np.Inf}             

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None and args.patience>=0:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        try: # training starts from non-tab WaveUnet
            optimizer = Adam(params=model.parameters(), lr=args.lr)            
            state = model_utils.load_model(model, optimizer, args.load_model, args.cuda, strict=False)
            state["best_sep_loss"]=np.Inf # __gbastas__ This is to train anew
            state["best_tab_loss"]=np.Inf # __gbastas__ This is to train anew
            state["best_acc"]=0 # __gbastas__ This is to train anew
            state["best_solo_loss"]=np.Inf # __gbastas__ This is to train anew
            state["best_mic_loss"]=np.Inf # __gbastas__ This is to train anew
            state["best_mix_loss"]=np.Inf # __gbastas__ This is to train anew
            state["best_hex_cln_loss"]=np.Inf # __gbastas__ This is to train anew  
            if args.task in ["tablature", "multitask"]:          
                if args.freeze:
                    for param in model.parameters(): # __gbastas__ freeze!!!
                        param.requires_grad = False
                model.add_tab_branches() # __gbastas__ 

        except Exception as e: # training starts from tab WaveUnet
            model.add_tab_branches() # __gbastas__
            if args.freeze:
                for param in model.parameters(): # __gbastas__
                    param.requires_grad = False
            state = model_utils.load_model(model, None, args.load_model, args.cuda, strict=False) # NOTE fix this None
            state["best_sep_loss"]=np.Inf # __gbastas__ This is to train anew
            state["best_tab_loss"]=np.Inf # __gbastas__ This is to train anew
            state["best_acc"]=0 # __gbastas__ This is to train anew            
            state["best_solo_loss"]=np.Inf # __gbastas__ This is to train anew
            state["best_mic_loss"]=np.Inf # __gbastas__ This is to train anew
            state["best_mix_loss"]=np.Inf # __gbastas__ This is to train anew
            state["best_hex_cln_loss"]=np.Inf # __gbastas__ This is to train anew  

    print('cuda', args.cuda)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Either prefreezed or not, unfreeze the newly added tab branch parameters so they can be trained
    if args.tab_version == '2up2down-TabCNN':
        for name, param in model.named_parameters():
            # if 'waveunet_head' in name or 'wfc1' in name or 'wfc2' in name:  # Assuming tab branch parameters contain 'tab_branch' in their names
            if 'wfc1' in name or 'wfc2' in name:  # Assuming tab branch parameters contain 'tab_branch' in their names
                param.requires_grad = True        
    elif args.tab_version == '2up2down':
        for name, param in model.named_parameters():
            if 'tab_' in name:  # Assuming tab branch parameters contain 'tab_branch' in their names
                param.requires_grad = True




    # Set up optimizer only for the unfrozen parameters (i.e., the tab branch parameters)
    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    print('TRAINING START')
    while state["worse_epochs"] <= args.patience:
        print("Training one epoch from iteration " + str(state["step"]))
        avg_time = 0.
        model.train()
        with tqdm(total=len(train_data) // args.batch_size) as pbar:
            np.random.seed()
            # for example_num, (x, targets) in enumerate(dataloader):
            # for example_num, (x, targets, tab_labels) in enumerate(dataloader):
            for example_num, (x, targets, tab_labels, cqt) in enumerate(dataloader):
                if args.cuda:
                    x = x.cuda()
                    if args.task in ['separation', 'multitask']:
                        for k in list(targets.keys()):
                            targets[k] = targets[k].cuda()
                    if args.task in ['tablature', 'multitask']:                        
                        cqt = cqt.cuda()
                        for k in list(tab_labels.keys()):
                            tab_labels[k] = tab_labels[k].cuda()

                t = time.time()


                # Set LR for this iteration
                utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

                # Compute loss for each instrument/model
                optimizer.zero_grad()
                _, avg_sep_loss, avg_tab_loss, avg_tab_acc = model_utils.compute_loss(model, x, targets, tab_labels, cqt, criterion, tab_criterion, compute_grad=True, task=args.task, tab_version=args.tab_version)

                optimizer.step()
                state["step"] += 1
                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)
                writer.add_scalar("train_sep_loss", avg_sep_loss, state["step"])
                writer.add_scalar("train_tab_loss", avg_tab_loss, state["step"])
                description = "sep loss: {:.4f}, tab loss: {:.4f}, ".format(
                    avg_sep_loss, avg_tab_loss
                )
                pbar.set_description(description)           
                pbar.update(1)

        # VALIDATE
        val_sep_loss, val_tab_loss, val_tab_acc = test.validate(args, model, criterion, tab_criterion,  val_data)


        print("VALIDATION SEP LOSS: " + str(val_sep_loss))
        print("VALIDATION TAB LOSS: " + str(val_tab_loss)) # __gbastas__
        writer.add_scalar("val_sep_loss", val_sep_loss, state["step"])
        writer.add_scalar("val_tab_loss", val_tab_loss, state["step"])

        # EARLY STOPPING CHECK
        if args.task == 'separation': # NOTE: not including multitask here means that if args.task==multitask the early stopping will depend on the tab results!
            best_loss_for_stopping = "best_sep_loss"
            best_loss_other =  "best_tab_loss"
            best_checkpoint_for_stopping = "sep_best_checkpoint"
            best_checkpoint_other = "tab_best_checkpoint"
            val_loss_for_stopping = val_sep_loss
            val_loss_other = val_tab_loss
        elif args.task in ['tablature', 'multitask']: # NOTE
            best_loss_for_stopping = "best_tab_loss"
            best_loss_other = "best_sep_loss"
            best_checkpoint_for_stopping = "tab_best_checkpoint"
            best_checkpoint_other = "sep_best_checkpoint"
            val_loss_for_stopping = val_tab_loss
            val_loss_other = val_sep_loss
        
        if val_loss_for_stopping >= state[best_loss_for_stopping]: 
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON TAB VALIDATION SET!")
            checkpoint_best_path = os.path.join(args.checkpoint_dir, best_checkpoint_for_stopping + "_" + str(state["step"]))
            try:
                os.remove(checkpoint_best_path_prev)
            except Exception as e:
                print('Caught exception:', e)

            print("Saving best tab model...")
            state["worse_epochs"] = 0
            state[best_loss_for_stopping] = val_loss_for_stopping #__gbastas__
            state[best_checkpoint_for_stopping] = checkpoint_best_path
            model_utils.save_model(model, optimizer, state, checkpoint_best_path)

        # save extra __gbastas__
        if val_loss_other < state[best_loss_other]:
            print("MODEL IMPROVED ON sep VALIDATION SET!")
            checkpoint_other_best_path = os.path.join(args.checkpoint_dir, best_checkpoint_other + "_" + str(state["step"]))
            try:
                os.remove(checkpoint_other_best_path_prev)
            except Exception as e:
                print('Caught exception sep:', e)
            print("Saving best sep model...")           
            state[best_loss_other] = val_loss_other
            model_utils.save_model(model, optimizer, state, checkpoint_other_best_path)

        # CHECKPOINT
        print("Saving model...")
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        model_utils.save_model(model, optimizer, state, checkpoint_path)
        try:
            os.remove(checkpoint_path_prev)
        except Exception as e:
            print('Caught exception:', e)        
        checkpoint_path_prev = checkpoint_path
        try:
            checkpoint_best_path_prev = checkpoint_best_path
            checkpoint_other_best_path_prev = checkpoint_other_best_path 
        except Exception as e:
            print("MyException", e)


        state["epochs"] += 1
        if args.patience==0:
            break
    # ************************************************************************************** #

    print("TESTING") 

    # Load best model based on validation loss
    if args.patience >= 0: # i.e., the last phase of the training phase, right after early stopping
        try:
            state = model_utils.load_model(model, None, state[best_checkpoint_for_stopping], args.cuda, strict=False) 
        except Exception as e:
            model.add_tab_branches() 
            for param in model.parameters(): # __gbastas__
                param.requires_grad = False
            for name, param in model.named_parameters():
                if 'tab_' in name:  # Assuming tab branch parameters contain 'tab_branch' in their names
                    param.requires_grad = True
            optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            state = model_utils.load_model(model, optimizer, state[best_checkpoint_for_stopping], args.cuda, strict=False)        
        
        test_loss, test_tab_loss, test_tab_acc = test.validate(args, model, criterion, tab_criterion,  test_data)

        print("TEST FINISHED: LOSS: " + str(test_loss))
        print("TEST FINISHED: TAB LOSS: " + str(test_tab_loss))
        print("TEST FINISHED: TAB ACC: " + str(test_tab_acc))
        writer.add_scalar("test_loss", test_loss, state["step"])
        writer.add_scalar("test_tab_loss", test_tab_loss, state["step"])

    else: # just testing without training (e.g. --patience -1)
        try: # loading not-Tab WaveUnet
            state = model_utils.load_model(model, None, args.load_model, args.cuda, strict=False)
            if args.task in ['tablature', 'multitask']:
                model.add_tab_branches() 
            model.to(device)

        except Exception as e: # loading Tab WaveUnet 
            if args.task in ['tablature', 'multitask']:
                model.add_tab_branches()
            else:
                print("[gb] Something's wrong!") 

            state = model_utils.load_model(model, None, args.load_model, args.cuda, strict=False)        
            device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
            model.to(device)

    test_metrics = {}

    # Mir_eval metrics
    test_metrics['total'], test_metrics['comp'], test_metrics['solo'] = test.evaluate(args, musdb["test"], model, args.instruments) 

    # Dump all metrics results into pickle file for later analysis if needed
    with open(os.path.join(args.checkpoint_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics['total'], f)

    for key, test_metrics in reversed(test_metrics.items()):

        print()
        print('!!Evaluating on '+ key + ' test set!!')
        print()

        print('test_metrics', len(test_metrics))

        # Write most important metrics into Tensorboard log
        avg_SDRs = {inst : np.mean([np.nanmean(song[inst]["SDR"]) for song in test_metrics]) for inst in args.instruments}
        avg_SIRs = {inst : np.mean([np.nanmean(song[inst]["SIR"]) for song in test_metrics]) for inst in args.instruments}
        avg_SARs = {inst : np.mean([np.nanmean(song[inst]["SAR"]) for song in test_metrics]) for inst in args.instruments}
        avg_SISDRs = {inst : np.mean([np.nanmean(song[inst]["SI-SDR"]) for song in test_metrics]) for inst in args.instruments}
        if args.patience > 0: # = right after train is complete
            resfile = open(args.checkpoint_dir+'/'+key+'_results_'+ state["best_checkpoint"].split('_')[-1] +'.csv', 'w')
        else:
            resfile = open(args.checkpoint_dir+'/'+key+'_results_'+ args.load_model.split('_')[-1] +'.csv', 'w')
        csvwriter = csv.writer(resfile)
        csvwriter.writerow([" ","SDR", "SIR", "SAR", "SI-SDR"])
        print(" ","SDR", "SIR", "SAR", "SI-SDR")
        
        for inst in args.instruments:
            csvwriter.writerow([inst, round(avg_SDRs[inst],3), round(avg_SIRs[inst],3), round(avg_SARs[inst],3), round(avg_SISDRs[inst],3)])
            print(inst, round(avg_SDRs[inst],3), round(avg_SIRs[inst],3), round(avg_SARs[inst],3), round(avg_SISDRs[inst],3))

        print()

        overall_SDR = np.mean([v for v in avg_SDRs.values()])
        overall_SIR = np.mean([v for v in avg_SIRs.values()])
        overall_SAR = np.mean([v for v in avg_SARs.values()])
        overall_SISDR = np.mean([v for v in avg_SISDRs.values()])

        csvwriter.writerow([" "])
        csvwriter.writerow([round(overall_SDR,3), round(overall_SIR,3), round(overall_SAR,3), round(overall_SISDR,3)])

        print("SDR: " + str(overall_SDR))
        print("SIR: " + str(overall_SIR))
        print("SAR: " + str(overall_SAR))
        print("SI-SDR: " + str(overall_SISDR))

    writer.close()
    resfile.close()

if __name__ == '__main__':
    ## TRAIN PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["E", "A", "D", "G", "B", "e"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")") # __gbastas__
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--log_dir', type=str, default='logs/waveunet',
                        help='Folder to write logs into')
    parser.add_argument('--dataset_dir', type=str, default="/mnt/windaten/Datasets/MUSDB18HQ",
                        help='Dataset path')
    parser.add_argument('--hdf_dir', type=str, default="hdf",
                        help='Dataset path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet',
                        help='Folder to write checkpoints into')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate in LR cycle (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=5e-5,
                        help='Minimum learning rate in LR cycle (default: 5e-5)')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of LR cycles per epoch')
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=1,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=2.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--example_freq', type=int, default=200,
                        help="Write an audio summary into Tensorboard logs every X training iterations")
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")
    parser.add_argument('--freeze', action='store_true',
                        help="freeze pretrained weights")
    parser.add_argument('--fakeframes_n', type=int, default=349,
                        help="Length of latent features for tablature transcription.")
        
    # __gbastas__
    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--version', type=str, default='HQ', help='"mic", "fake", HQ-comp", "HQ-solo", "cross-val" alternatively')
    parser.add_argument('--tab_version', type=str, default='None', help='4up3down, 4up3down-DSshorts, 4up3down-ds5n4')
    parser.add_argument('--task', type=str, default='separation', help='separation | tablature | multitask')
    parser.add_argument('--resample', action='store_true', help="Resample CQT to match Wave-U-net feature map.")
    args = parser.parse_args()

    args.log_dir = 'logs/'+args.checkpoint_dir.split('/')[-1]

    if args.split>=0:
        args.log_dir = args.log_dir + '_' + str(args.split)
        args.checkpoint_dir = args.checkpoint_dir + '_' + str(args.split)
        args.hdf_dir = args.hdf_dir + '_' + str(args.split)

    main(args)
