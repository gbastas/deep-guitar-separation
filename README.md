# Separate and Transcribe

This repository contains the code and experiments from the paper: *Separate and Transcribe: Deep Guitar Separation and its Application for Tablature Enhancement.* It provides implementations and dataset manipulation and preparation code used for training, evaluation, and analysis.


## Data Preparation


**GuitarSet**

Download and put in ordet the GuitarSet dataset (https://zenodo.org/record/3371780).

Use the unix commands below:

```
mkdir -p ./datasets/GuitarSet/data/{audio,mic,mix,hex_cln}

wget -P ./datasets/GuitarSet/ https://zenodo.org/record/3371780/files/{audio_mono-mic.zip,audio_mono-pickup_mix.zip,audio_hex-pickup_debleeded.zip}

unzip -j ./datasets/GuitarSet/audio_mono-mic.zip '*.wav' -d ./datasets/GuitarSet/data/mic && \
unzip -j ./datasets/GuitarSet/audio_mono-pickup_mix.zip '*.wav' -d ./datasets/GuitarSet/data/mix && \
unzip -j ./datasets/GuitarSet/audio_hex-pickup_debleeded.zip '*.wav' -d ./datasets/GuitarSet/data/hex_cln

```

To get the dataset ready for training, we follow the *Senvaitytite train-test split* presented in [this repository](https://github.com/daliasen/GuitarStringSeparation-MF-NMF-NMFD):

```
cd datasets/GuitarSet/
python prepare_source_sep_data.py # this create dir datasep/
python prepare_source_sep_data-mic.py # this create dir datasep-mic/
prepare_source_sep_datamono-pickup.py # this create dir datasep-mix/
cd -
```

After running these scripts, the following directories will be created:

- **`datasep/`** – for general source separation data
- **`datasep-mic/`** – for microphone-based separation data
- **`datasep-mix/`** – for pickup-mix separation data



**GSCustomMic: Preparing the Dataset**

Run the following commands to process and extract data:
```
cd data-manipulation-code/
python AuxDataPrep.py --action pseudo_sep --all_solos
python AuxDataPrep.py --action pseudocomp_sep --all_solos
cd -
```

Create the GSCustomMic Directory:
```
cd datasets
mkdir -p GSCustomMic
```

Copy Processed Data into GSCustomMic:
```
cd datasets/GuitarSet
cp -r ./pseudo_sep_all_solos_mic_wn/* ../GSCustomMic
cp -r ./pseudocomp_sep_all_notes/* ../GSCustomMic
```

Perform Train-Test Split:
```
cd ..
python pseudo_train_test_split.py -d GSCustomMic # this will create ./datasets/{dir_name}/test/
cd GSCustomMic
mkdir train
mv * train # ignore warning
mv train/test .
```

**MDGP: Preparing the Dataset**

For the creation of the MDGP dataset we first need to gather note instances from the GuitarSet mic solos:
```
python AuxDataPrep.py --action gather_notes 
```

This command will create dir ```note_instances```.

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

The implementation code for Wave-U-Net is based on [this repository](https://github.com/f90/Wave-U-Net-Pytorch).


```
python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit --checkpoint_dir checkpoints/waveunet_guit --channels 1 --patience 200 

python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit-comp --checkpoint_dir checkpoints/waveunet_guit-comp --channels 1 --patience 200 --version HQ-comp

python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit-solo --checkpoint_dir checkpoints/waveunet_guit-solo --channels 1 --patience 200 --version HQ-solo

python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit --checkpoint_dir checkpoints/waveunet_guit_monosep --channels 1 --patience 200 --separate 0
```

**Wave-U-Net-Tab**

python train.py

--dataset_dir ../datasets/GuitarSet/datasep-mix/
--hdfs/{hdf_guit-pret-mix, hdf_guit-pret-mic, hdf_guit-pret-mic-fakemic, hdf_guit-pret-mic-pseudoboth, hdf_guit-pret-mic-pseudoboth-fakemic}
--checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/{waveunet_guit, waveunet_guit_monopickup, waveunet_guit_mic, waveunet_guit_mic_fakemic, waveunet_guit_pseudoboth_wn, waveunet_guit_pseudoboth_sep_all_solos_fake}
--load_model ../Wave-U-Net-Pytorch-6string/checkpoints/{waveunet_guit, waveunet_guit_monopickup, waveunet_guit_mic, waveunet_guit_mic_fakemic, waveunet_guit_pseudoboth_wn, waveunet_guit_pseudoboth_sep_all_solos_fake}/best_checkpoint_{}
--cuda --patience 20 --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down {--freeze}

```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mic/ --hdf_dir hdfs/hdf_guit-pret-mic --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic-pret5up5down-freeze --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic/best_checkpoint_555000 --cuda --patience 20 --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down --freeze

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mic/ --hdf_dir hdfs/hdf_guit-pret-mic --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic-pret5up5down --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic/best_checkpoint_555000 --cuda --patience 20  --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down 

CUDA_VISIBLE_DEVICES=2 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mix/ --hdf_dir hdfs/hdf_guit-pret-mix --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mix-pret5up5down-freeze --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_monopickup/best_checkpoint_540200 --cuda --patience 20  --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down --freeze

CUDA_VISIBLE_DEVICES=3 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mix/ --hdf_dir hdfs/hdf_guit-pret-mix --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mix-pret5up5down --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_monopickup/best_checkpoint_540200 --cuda --patience 20  --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down

```

