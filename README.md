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
python prepare_source_sep_datamono-pickup.py # this create dir datasep-mix/
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
```

Copy Processed Data into GSCustomMic:
```
mkdir -p GSCustomMic
cp -r ./pseudo_sep_all_solos_mic_wn/* GSCustomMic
cp -r ./pseudocomp_sep_all_notes/* GSCustomMic
```

Perform Train-Test Split: # TODO
```
python pseudo_train_test_split.py -d GSCustomMic # 
```
This will create ```GSCustomMic/{test, train}/```.

Last but not least:
```
mv GSCustomMic/ ../datasets
rm -r ./pseudo_sep_all_solos_mic_wn/
rm -r ./pseudocomp_sep_all_notes/
```

**MDGP: Preparing the Dataset**

For the creation of the MDGP dataset we first need to gather note instances from the GuitarSet mic solos:
```
python AuxDataPrep.py --action gather_notes 
```

This command will create dir ```note_instances/data/```. 

The run the following command:
```
cp -r exps/Guitarset/note_instances/data/train exps/Guitarset/note_instances/data/onsets && find exps/Guitarset/note_instances/data/onsets -type f -name '*.wav' -exec sh -c 'f="{}"; d=$(dirname "$f"); b=$(basename "$f" .wav); rm "$f" && echo 0 > "${d}/${b}.txt"' \;
```
This:
- Copies train → onsets without moving to another directory.
- Finds all .wav files in exps/Guitarset/note_instances/data/onsets/.
- Replaces each .wav file with a .txt file containing 0.


Now Check this README:
https://gitlab.com/ilsp-spmd-all/phds/phd-grigoris/string_separation#fake-full-track-audio-data-from-midi

## Basic commands for training and evaluation of Wave-U-Net for guitar string separation:

(The implementation code for Wave-U-Net is based on [this repository](https://github.com/f90/Wave-U-Net-Pytorch).)


Firstly:
```
cd Wave-U-Net-Pytorvh-6string
```

Clean training:
```
python train.py \
  --dataset_dir ../datasets/GuitarSet/<your_dataset_dir> \
  --hdf_dir hdfs/<choose_dirname> \
  --checkpoint_dir checkpoints/<choose_dirname> \
  --patience 200 \
  --channels 1 \
  --cuda
```

For Quantitative Evaluation:
```
python train.py \
  --dataset_dir datasets/GuitarSet/<your_dataset_dir> \
  --hdf_dir hdfs/<your_hdf_dir> \
  --checkpoint_dir checkpoints/<your_checkpoint_dir> \
  --patience -1 \
  --load_model checkpoints/<your_checkpoint_dir>/best_checkpoint_<N>
```

Qualitative Testing:
```
python predict.py --load_model checkpoints/{checkpoint_dir}/best_checkpoint_<N> --input path/to/wav --cuda
```

Results Visualisation:

```
cd Wave-U-Net-Pytorch-6string
tensorboard --logdir logs/
```

## Basic commands for training and evaluation of Wave-U-Net-Tab for guitar tablature transcription:

Firtly:
```
cd Wave-U-Net-Pytorvh-6string-tablature
```

Clean training:
```
python train.py
--dataset_dir ../datasets/GuitarSet/datasep-mix/
--hdfs/{hdf_guit-pret-mix, hdf_guit-pret-mic, hdf_guit-pret-mic-fakemic, hdf_guit-pret-mic-pseudoboth, hdf_guit-pret-mic-pseudoboth-fakemic}
--checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/{waveunet_guit, waveunet_guit_monopickup, waveunet_guit_mic, waveunet_guit_mic_fakemic, waveunet_guit_pseudoboth_wn, waveunet_guit_pseudoboth_sep_all_solos_fake}
--load_model ../Wave-U-Net-Pytorch-6string/checkpoints/{waveunet_guit, waveunet_guit_monopickup, waveunet_guit_mic, waveunet_guit_mic_fakemic, waveunet_guit_pseudoboth_wn, waveunet_guit_pseudoboth_sep_all_solos_fake}/best_checkpoint_{}
--cuda --patience 20 --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down {--freeze}
```



## Experiments

**Wave-U-Net**

To train the core separation model (multi-channel Wave-U-Net), run:

```
python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit --checkpoint_dir checkpoints/waveunet_guit --channels 1 --patience 200 
```

To train the one-channel Wave-U-Net, run:

```
python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit --checkpoint_dir checkpoints/waveunet_guit_monosep --channels 1 --patience 200 --separate 0
```

To train multi-channel Wave-U-Net solely on Comp or soleley on Solo, run the following commands accrodingly:
```
python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit-comp --checkpoint_dir checkpoints/waveunet_guit-comp --channels 1 --patience 200 --version HQ-comp

python train.py --dataset_dir ../datasets/GuitarSet/datasep/ --cuda --hdf_dir hdfs/hdf_guit-solo --checkpoint_dir checkpoints/waveunet_guit-solo --channels 1 --patience 200 --version HQ-solo

```

**Wave-U-Net-Tab**

```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mic/ --hdf_dir hdfs/hdf_guit-pret-mic --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic-pret5up5down-freeze --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic/best_checkpoint_555000 --cuda --patience 20 --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down --freeze

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mic/ --hdf_dir hdfs/hdf_guit-pret-mic --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic-pret5up5down --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic/best_checkpoint_555000 --cuda --patience 20  --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down 

CUDA_VISIBLE_DEVICES=2 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mix/ --hdf_dir hdfs/hdf_guit-pret-mix --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mix-pret5up5down-freeze --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_monopickup/best_checkpoint_540200 --cuda --patience 20  --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down --freeze

CUDA_VISIBLE_DEVICES=3 python train.py --dataset_dir ../datasets/GuitarSet/datasep-mix/ --hdf_dir hdfs/hdf_guit-pret-mix --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mix-pret5up5down --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_monopickup/best_checkpoint_540200 --cuda --patience 20  --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down

```

