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
from test import evaluate, validate
from model.waveunet import Waveunet
import csv

# import openunmix


def main(args):

    # MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)

    # if args.model == 'open-unmix':   
    #     model = openunmix.Unmix('umxhq') 
    #     features='stft'
    
    features=None

    print('cuda', args.cuda)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    # print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    writer = SummaryWriter(args.log_dir)

    ### DATASET
    musdb = get_musdb_folds(args.dataset_dir, version=args.version, guitID=args.split)
    # musdb = get_musdbhq(args.dataset_dir)#, guitID=args.split)
    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop_targets, shapes=model.shapes)
    # Data augmentation function for training
    augment_func = partial(random_amplify, shapes=model.shapes, min=0.7, max=1.0)
    train_data = SeparationDataset(musdb, "train", args.instruments, args.sr, args.channels, model.shapes, True, args.hdf_dir, audio_transform=augment_func, features=features) # NOTE: augmentation
    val_data = SeparationDataset(musdb, "val", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func, features=features)
    # try:
    if args.version=='pseudo':
        try:
            val_comp_data = SeparationDataset(musdb, "val_comp", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func, features=features)
            val_solo_data = SeparationDataset(musdb, "val_solo", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func, features=features)
        except Exception as e:
            print("warning couldn't load val_comp or val_solo")
    else:
        val_comp_loss=np.Inf
        val_solo_loss=np.Inf

    if args.version=='mic':
        val_mic_data = SeparationDataset(musdb, "val_mic", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func, features=features)
        val_mix_data = SeparationDataset(musdb, "val_mix", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func, features=features)
        val_hex_cln_data = SeparationDataset(musdb, "val_hex_cln", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func, features=features)
    else:
        val_mix_loss=np.Inf
        val_hex_cln_loss=np.Inf

        
    print('No comp/solo distinct val scores will be considered')
    test_data = SeparationDataset(musdb, "test", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func, features=features)

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)


    ##### TRAINING ####

    # Set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    # Set up optimiser
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf,
             "best_comp_loss" : np.Inf,
             "best_solo_loss" : np.Inf,
             "best_mic_loss" : np.Inf,
             "best_mix_loss" : np.Inf,
             "best_hex_cln_loss" : np.Inf}
             

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)
        state["best_loss"]=np.Inf # __gbastas__ This is to train anew
        state["best_comp_loss"]=np.Inf # __gbastas__ This is to train anew
        state["best_solo_loss"]=np.Inf # __gbastas__ This is to train anew
        state["best_mic_loss"]=np.Inf # __gbastas__ This is to train anew
        state["best_mix_loss"]=np.Inf # __gbastas__ This is to train anew
        state["best_hex_cln_loss"]=np.Inf # __gbastas__ This is to train anew

    print('TRAINING START')
    while state["worse_epochs"] < args.patience:
        print("Training one epoch from iteration " + str(state["step"]))
        avg_time = 0.
        model.train()
        with tqdm(total=len(train_data) // args.batch_size) as pbar:
            np.random.seed()
            for example_num, (x, targets) in enumerate(dataloader):
                if args.cuda:
                    x = x.cuda()
                    for k in list(targets.keys()):
                        targets[k] = targets[k].cuda()

                t = time.time()

                # Set LR for this iteration
                utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

                # Compute loss for each instrument/model
                optimizer.zero_grad()
                outputs, avg_loss = model_utils.compute_loss(model, x, targets, criterion, compute_grad=True)

                optimizer.step()

                state["step"] += 1

                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                writer.add_scalar("train_loss", avg_loss, state["step"])


                pbar.update(1)

        # VALIDATE
        val_loss = validate(args, model, criterion, val_data)
        try:
            val_comp_loss = validate(args, model, criterion, val_comp_data)
            val_solo_loss = validate(args, model, criterion, val_solo_data)
        except Exception as e:
            print("warning couldn't load val_comp or val_solo")

        if args.version=='mic':
            val_mic_loss = validate(args, model, criterion, val_mic_data)
            val_mix_loss = validate(args, model, criterion, val_mix_data)
            val_hex_cln_loss = validate(args, model, criterion, val_hex_cln_data)
            
        print("VALIDATION FINISHED: LOSS: " + str(val_loss))
        writer.add_scalar("val_loss", val_loss, state["step"])
        writer.add_scalar("val_comp_loss", val_comp_loss, state["step"])
        writer.add_scalar("val_solo_loss", val_solo_loss, state["step"])
        if args.version=='mic':
            writer.add_scalar("val_mix_loss", val_mix_loss, state["step"])
            writer.add_scalar("val_hex_cln_loss", val_hex_cln_loss, state["step"])

        # EARLY STOPPING CHECK
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            checkpoint_best_path = os.path.join(args.checkpoint_dir, "best_checkpoint_" + str(state["step"]))
            try:
                os.remove(checkpoint_best_path_prev)
            except Exception as e:
                print('Caught exception:', e)

            print("Saving model...")
            
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_best_path
            model_utils.save_model(model, optimizer, state, checkpoint_best_path)

        # extra __gbastas__
        if val_comp_loss < state["best_comp_loss"]:
            print("MODEL IMPROVED ON VALIDATION SET!")
            checkpoint_comp_best_path = os.path.join(args.checkpoint_dir, "best_comp_checkpoint_" + str(state["step"]))
            try:
                os.remove(checkpoint_comp_best_path_prev)
            except Exception as e:
                print('Caught exception comp:', e)
            print("Saving best comp model...")      
            state["best_comp_loss"] = val_comp_loss
            model_utils.save_model(model, optimizer, state, checkpoint_comp_best_path)
    
        # extra __gbastas__
        if val_solo_loss < state["best_solo_loss"]:
            print("MODEL IMPROVED ON SOLO VALIDATION SET!")
            checkpoint_solo_best_path = os.path.join(args.checkpoint_dir, "best_solo_checkpoint_" + str(state["step"]))
            try:
                os.remove(checkpoint_solo_best_path_prev)
            except Exception as e:
                print('Caught exception solo:', e)
            print("Saving best solo model...")           
            state["best_solo_loss"] = val_solo_loss
            model_utils.save_model(model, optimizer, state, checkpoint_solo_best_path)

        # extra __gbastas__
        if val_hex_cln_loss < state["best_hex_cln_loss"] and args.version=='mic':
            print("MODEL IMPROVED ON hex_cln VALIDATION SET!")
            checkpoint_hex_cln_best_path = os.path.join(args.checkpoint_dir, "best_hex_cln_checkpoint_" + str(state["step"]))
            try:
                os.remove(checkpoint_hex_cln_best_path_prev)
            except Exception as e:
                print('Caught exception solo:', e)
            print("Saving best solo model...")           
            state["best_hex_cln_loss"] = val_hex_cln_loss
            model_utils.save_model(model, optimizer, state, checkpoint_hex_cln_best_path)

        # # extra __gbastas__
        if val_mic_loss < state["best_mic_loss"] and args.version=='mic':
            print("MODEL IMPROVED ON mic VALIDATION SET!")
            checkpoint_mic_best_path = os.path.join(args.checkpoint_dir, "best_mic_checkpoint_" + str(state["step"]))
            try:
                os.remove(checkpoint_mic_best_path_prev)
            except Exception as e:
                print('Caught exception solo:', e)
            print("Saving best solo model...")           
            state["best_mic_loss"] = val_mic_loss
            model_utils.save_model(model, optimizer, state, checkpoint_mic_best_path)

        # extra __gbastas__
        if val_mix_loss < state["best_mix_loss"] and args.version=='mic':
            print("MODEL IMPROVED ON mix VALIDATION SET!")
            checkpoint_mix_best_path = os.path.join(args.checkpoint_dir, "best_mix_checkpoint_" + str(state["step"]))
            try:
                os.remove(checkpoint_mix_best_path_prev)
            except Exception as e:
                print('Caught exception solo:', e)
            print("Saving best solo model...")           
            state["best_mix_loss"] = val_mix_loss
            model_utils.save_model(model, optimizer, state, checkpoint_mix_best_path)


        # CHECKPOINT
        print("Saving model...")
        model_utils.save_model(model, optimizer, state, checkpoint_path)
        try:
            os.remove(checkpoint_path_prev)
        except Exception as e:
            print('Caught exception:', e)        
        checkpoint_path_prev = checkpoint_path
        try:
            checkpoint_best_path_prev = checkpoint_best_path
        except Exception as e:
            print("MyException", e)

        try:               
            checkpoint_comp_best_path_prev = checkpoint_comp_best_path
            checkpoint_solo_best_path_prev = checkpoint_solo_best_path
        except Exception as e:
            print("MyException", e)
            
        if args.version == 'mic':
            checkpoint_hex_cln_best_path_prev = checkpoint_hex_cln_best_path
            checkpoint_mix_best_path_prev = checkpoint_mix_best_path
            checkpoint_mic_best_path_prev = checkpoint_mic_best_path

        state["epochs"] += 1

    #### TESTING ####
    print("TESTING")

    # Load best model based on validation loss
    if args.patience > 0:
        state = model_utils.load_model(model, None, state["best_checkpoint"], args.cuda) 
    else:
        state = model_utils.load_model(model, None, args.load_model, args.cuda)

    if args.patience>=0:
        test_loss = validate(args, model, criterion, test_data)
        print("TEST FINISHED: LOSS: " + str(test_loss))
        writer.add_scalar("test_loss", test_loss, state["step"])

    test_metrics = {}

    # Mir_eval metrics
    test_metrics['total'], test_metrics['comp'], test_metrics['solo'] = evaluate(args, musdb["test"], model, args.instruments) 

    # Dump all metrics results into pickle file for later analysis if needed
    with open(os.path.join(args.checkpoint_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics['total'], f)

    for key,test_metrics in reversed(test_metrics.items()):
        print()
        print('!!Evaluating on '+ key + ' test set!!')
        print()


        # print('test_metrics', len(test_metrics), len(test_metrics[0]))
        # aaa

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
        # print(" ","SDR", "SIR", "SAR", "SI-SDR")
        
        # for inst in args.instruments:
        #     csvwriter.writerow([inst, round(avg_SDRs[inst],3), round(avg_SIRs[inst],3), round(avg_SARs[inst],3), round(avg_SISDRs[inst],3)])
        #     print(inst, round(avg_SDRs[inst],3), round(avg_SIRs[inst],3), round(avg_SARs[inst],3), round(avg_SISDRs[inst],3))

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

    # __gbastas__
    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--version', type=str, default='HQ', help='"cross-val" alternatively')
    args = parser.parse_args()

    args.log_dir = 'logs/'+args.checkpoint_dir.split('/')[-1]

    if args.split>=0:
        args.log_dir = args.log_dir + '_' + str(args.split)
        args.checkpoint_dir = args.checkpoint_dir + '_' + str(args.split)
        args.hdf_dir = args.hdf_dir + '_' + str(args.split)

    main(args)
