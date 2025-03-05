# Guitar String Separation

This repo is based on https://github.com/f90/Wave-U-Net-Pytorch.


Download from GuitarSet dataset (https://zenodo.org/record/3371780):

    annotations.zip and extract all .jams files to data/annos.
    audio_mono-mic.zip,  audio_mono-pickup_mix.zip and extract all .wav files to data/audio.

On the configuration file constants.ini, specify the folders where annotations and input audio is stored.


## Data Preparation

```
cd datasets/GuitarSet/
python prepare_source_sep_data.py
python prepare_source_sep_data-mic.py
prepare_source_sep_datamono-pickup.py
```
(and simply mv files by-patterns commands)

## For Guitarset Target Reconstructions for Solo-Mic Recordings
Check here:
https://gitlab.com/ilsp-spmd-all/music/tab-demo/-/tree/dev?ref_type=heads#creating-solo-target-reconstructions-from-gutarset
(I normally use tab-demo repo locally)

The directory containing the reconstructions needs to be uploaded and moved to asimov ```datasets/{dir_name}```.
The subdirectories corresponding to each guitar preformance need to be split to train/test datasets:
```
cd datasets/
python pseudo_train_test_split.py -d {dir_name} # this will create ./datasets/{dir_name}/test/
cd {dir_name}
mkdir train
mv * train # ignore warning
mv train/test .
```

## To Create Fake Data
First we need to gather note instances:
https://gitlab.com/ilsp-spmd-all/music/tab-demo/-/tree/dev?ref_type=heads#retrieving-note-instances-from-gutarset

Now Check this README:
https://gitlab.com/ilsp-spmd-all/phds/phd-grigoris/string_separation#fake-full-track-audio-data-from-midi

## Experiment

Clean training:
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf --checkpoint_dir checkpoints/waveunet --log_dir logs/waveunet --channels 1 --patience 200```
```

Quantitative Evaluation:
```
python train.py --dataset_dir ../GuitarSet/{datasep}/ --cuda --hdf_dir hdfs/hdf --checkpoint_dir checkpoints/{waveunet} --log_dir logs/{waveunet} --channels 1 --patience -1 --split # --load_model checkpoints/{checkpoint_dir}/best_checkpoint_#
```

Qualitative Testing:
```
python predict.py --load_model checkpoints/{checkpoint_dir}/best_checkpoint_# --input path/to/wav --cuda
```

## Results Visualisation

```
cd Wave-U-Net-Pytorch-6string
tensorboard --logdir logs/
```


## Experiments

**Wave-U-Net**

python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit --checkpoint_dir checkpoints/waveunet_guit --channels 1 --patience 200 

python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit-comp --checkpoint_dir checkpoints/waveunet_guit-comp --channels 1 --patience 200 --version HQ-comp

python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit-solo --checkpoint_dir checkpoints/waveunet_guit-solo --channels 1 --patience 200 --version HQ-solo

python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit --checkpoint_dir checkpoints/waveunet_guit_monosep --channels 1 --patience 200 --separate 0

**Wave-U-Net-Tab**

python train.py

--dataset_dir ../datasets/GuitarSet/datasep-mix/
--hdfs/{hdf_guit-pret-mix, hdf_guit-pret-mic, hdf_guit-pret-mic-fakemic, hdf_guit-pret-mic-pseudoboth, hdf_guit-pret-mic-pseudoboth-fakemic}
--checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/{waveunet_guit, waveunet_guit_monopickup, waveunet_guit_mic, waveunet_guit_mic_fakemic, waveunet_guit_pseudoboth_wn, waveunet_guit_pseudoboth_sep_all_solos_fake}
--load_model ../Wave-U-Net-Pytorch-6string/checkpoints/{waveunet_guit, waveunet_guit_monopickup, waveunet_guit_mic, waveunet_guit_mic_fakemic, waveunet_guit_pseudoboth_wn, waveunet_guit_pseudoboth_sep_all_solos_fake}/best_checkpoint_{}
--cuda --patience 20 --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down {--freeze}


CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mic/ --hdf_dir hdfs/hdf_guit-pret-mic --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic-pret5up5down-freeze --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic/best_checkpoint_555000 --cuda --patience 20 --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down --freeze

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mic/ --hdf_dir hdfs/hdf_guit-pret-mic --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic-pret5up5down --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic/best_checkpoint_555000 --cuda --patience 20  --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down 

CUDA_VISIBLE_DEVICES=2 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mix/ --hdf_dir hdfs/hdf_guit-pret-mix --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mix-pret5up5down-freeze --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_monopickup/best_checkpoint_540200 --cuda --patience 20  --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down --freeze

CUDA_VISIBLE_DEVICES=3 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mix/ --hdf_dir hdfs/hdf_guit-pret-mix --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mix-pret5up5down --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_monopickup/best_checkpoint_540200 --cuda --patience 20  --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down



