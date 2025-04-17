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
cd datasets/
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
mv GSCustomMic/ ../datasets/datasep-gscustmic/
rm -r ./pseudo_sep_all_solos_mic_wn/
rm -r ./pseudocomp_sep_all_notes/
```

Hence, our dataset for separation is ready:
- **`datasep-gscustmic/`** - to train and test our data on our custom auxiliary **GSCustomMic** dataset

**MDGP: Preparing the Dataset**

For the creation of the MDGP dataset we first need to gather note instances from the GuitarSet mic solos:
```
cd data-manipulation-code
python AuxDataPrep.py --action gather_notes 
```

The command will create dir ```note_instances/data/```.
Next, we need to create audio representationions of the symbolic DadaGP (MIDI) data by rendering the gathered note events accordingly:

```
python midi2audio_recs.py --note_instances note_instances/ --input_dir gp_token_examples --guitar micguitar --n_samples 30
```

The command above will create a set of audio tracks and store them in ```mdgp```. We then need to move this dir to the right place:
```
mv mdgp ../datasets/
cd -
```

Now, we can further insert these tracks into distinct training sets to create the baseses for new separation and transcription experiments:
```
cp -r datasets/datasep-mic datasets/datasep-mic-mdgp
cp -r datasets/datasep-mic datasets/datasep-gscustmic-mdgp
cp -r dadatsets/mdgp/* datasets/datasep-mic-mdgp/train/
cp -r dadatsets/mdgp/* datasets/datasep-gscustmic-mdgp/train/
```

And so we have created:
- **`datasep-mic-mdgp/`** - this is **Mic+MDGP** dataset
- **`datasep-gscustmic-mdgp/`** - this is **GSCustomMic+MDGP** dataset


**Plot GuitarSet Note-String Histogram**

```
cd data-manipulation-code
python AuxDataPrep.py --action plot --all_tracks
```


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

Firstly:
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

**Wave-U-Net ablation experiments**

To train the core separation model (multi-channel Wave-U-Net), run:

```
python train.py --dataset_dir ../datasets/datasep/ --cuda --hdf_dir hdfs/hdf_guit --checkpoint_dir checkpoints/waveunet_guit_hex --channels 1 --patience 200 
```

To train the one-channel Wave-U-Net, run:
```
python train.py --dataset_dir ../datasets/datasep/ --cuda --hdf_dir hdfs/hdf_guit --checkpoint_dir checkpoints/waveunet_guit_monosep --channels 1 --patience 200 --separate 0
```

To train multi-channel Wave-U-Net solely on Comp or soleley on Solo, run the following commands accrodingly:
```
python train.py --dataset_dir ../datasets/datasep/ --cuda --hdf_dir hdfs/hdf_guit-comp --checkpoint_dir checkpoints/waveunet_guit_hex-comp --channels 1 --patience 200 --version HQ-comp
python train.py --dataset_dir ../datasets/datasep/ --cuda --hdf_dir hdfs/hdf_guit-solo --checkpoint_dir checkpoints/waveunet_guit_hex-solo --channels 1 --patience 200 --version HQ-solo
```

To train on Demucs and Demucs HT move to dir ```demucs``` and and follow the instructions in the corresponding README.md file.
[TODO]


**Wave-U-Net dataset-wise experiments**


Runs with no pretraining:
```
python train.py --dataset_dir ../datasets/datasep-mix/ --cuda --hdf_dir hdfs/hdf_guit-mix --checkpoint_dir checkpoints/waveunet_guit_mix --channels 1 --patience 200 
python train.py --dataset_dir ../datasets/datasep-mic/ --cuda --hdf_dir hdfs/hdf_guit-mic --checkpoint_dir checkpoints/waveunet_guit_mic --channels 1 --patience 200 
python train.py --dataset_dir ../datasets/datasep-gscustmic/ --cuda --hdf_dir hdfs/hdf_guit-gscustmic --checkpoint_dir checkpoints/waveunet_guit_gscustmic --channels 1 --patience 200 
python train.py --dataset_dir ../datasets/datasep-mic-mdgp/ --cuda --hdf_dir hdfs/hdf_guit-mic-mdgp --checkpoint_dir checkpoints/waveunet_guit_mic_mdgp --channels 1 --patience 200 
python train.py --dataset_dir ../datasets/datasep-gscustmic-mdgp/ --cuda --hdf_dir hdfs/hdf_guit-gscustmic-mdgp --checkpoint_dir checkpoints/waveunet_guit_gscustmic_mdgp --channels 1 --patience 200 
```

Runs with pre-training:
```
python train.py --dataset_dir ../datasets/datasep-mic/ --cuda --hdf_dir hdfs/hdf_guit-mic --checkpoint_dir checkpoints/waveunet_guit_mic-pretOnHex --load_model checkpoints/waveunet_guit_hex/<best_checkpoint> --channels 1 --patience 200 
python train.py --dataset_dir ../datasets/datasep-mic-mdgp/ --cuda --hdf_dir hdfs/hdf_guit-mic-mdgp --checkpoint_dir checkpoints/waveunet_guit_mic-mdgp-pretOnHex --load_model checkpoints/waveunet_guit_hex/<best_checkpoint> --channels 1 --patience 200 
python train.py --dataset_dir ../datasets/datasep-gscustmic/ --cuda --hdf_dir hdfs/hdf_guit-gscustmic --checkpoint_dir checkpoints/waveunet_guit_gscustmic-pretOnMic --load_model checkpoints/waveunet_guit_mic/<best_checkpoint> --channels 1 --patience 200 
python train.py --dataset_dir ../datasets/datasep-gscustmic/ --cuda --hdf_dir hdfs/hdf_guit-gscustmic --checkpoint_dir checkpoints/waveunet_guit_gscustmic-pretOnHex --load_model checkpoints/waveunet_guit_hex/<best_checkpoint> --channels 1 --patience 200 
python train.py --dataset_dir ../datasets/datasep-gscustmic-mdgp/ --cuda --hdf_dir hdfs/hdf_guit-gscustmic-mdgp --checkpoint_dir checkpoints/waveunet_guit_gscustmic-mdgp-pretOnMic --load_model checkpoints/waveunet_guit_mic/<best_checkpoint> --channels 1 --patience 200
python train.py --dataset_dir ../datasets/datasep-gscustmic-mdgp/ --cuda --hdf_dir hdfs/hdf_guit-gscustmic-mdgp --checkpoint_dir checkpoints/waveunet_guit_gscustmic-mdgp-pretOnHex --load_model checkpoints/waveunet_guit_hex/<best_checkpoint> --channels 1 --patience 200
```



**Wave-U-Net-Tab**

Train with pretrained Wave-U-Net on Pckp:
```
python train.py --dataset_dir ../datasets/GuitarSet/datasep-mix/ --hdf_dir hdfs/hdf_guit-pret-mix --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mix-pret5up5down-freeze --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_monopickup/best_checkpoint_<N> --cuda --patience 20  --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down --freeze
```

Train with pretrained Wave-U-Net on Mic:

```
python train.py --dataset_dir ../datasets/GuitarSet/datasep-mic/ --hdf_dir hdfs/hdf_guit-pret-mic --checkpoint_dir ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic-pret5up5down-freeze --load_model ../Wave-U-Net-Pytorch-6string/checkpoints/waveunet_guit_mic/best_checkpoint_<N> --cuda --patience 20 --batch_size 1 --fakeframes_n 87 --task tablature --tab_version 2up2down --freeze
```

**TabCNN**

for file in ../datasets/GuitarSet/datasep-preds/train/*/mixture.wav; do python predict.py --load_model checkpoints/waveunet_guit/best_checkpoint_511525 --input "$file"; done
for file in ../datasets/GuitarSet/datasep-preds/test/*/mixture.wav; do python predict.py --load_model checkpoints/waveunet_guit/best_checkpoint_511525 --input "$file"; done

for file in ../datasets/GuitarSet/datasep-mic-preds/train/*/mixture.wav; do CUDA_VISIBLE_DEVICES=0 python predict.py --cuda --load_model checkpoints/waveunet_guit_mic/best_checkpoint_555000 --input "$file"; done

for file in ../datasets/GuitarSet/datasep-mic-preds/test/*/mixture.wav; do CUDA_VISIBLE_DEVICES=1 python predict.py --cuda --load_model checkpoints/waveunet_guit_mic/best_checkpoint_555000 --input "$file"; done


for file in ../datasets/GuitarSet/datasep-mix-preds/train/*/mixture.wav; do CUDA_VISIBLE_DEVICES=2 python predict.py --cuda --load_model checkpoints/waveunet_guit_monopickup/best_checkpoint_540200 --input "$file"; done

for file in ../datasets/GuitarSet/datasep-mix-preds/train/*/mixture.wav; do python predict.py --load_model checkpoints/waveunet_guit_monopickup/best_checkpoint_540200 --input "$file"; done

