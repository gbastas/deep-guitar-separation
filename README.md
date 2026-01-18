# Separate and Transcribe

This repository contains the code and experiments from the paper: *Separate and Transcribe: Deep Guitar Separation and its Application for Tablature Enhancement.* It provides implementations, dataset manipulation and preparation code require for training, evaluation, and further analysis of the results.


## Data Preparation


**GuitarSet**

Download and put in order the GuitarSet dataset (https://zenodo.org/record/3371780) by using the unix commands below:

```
mkdir -p ./datasets/GuitarSet/data/{audio,mic,mix,hex_cln}
wget -P ./datasets/GuitarSet/ https://zenodo.org/record/3371780/files/{audio_mono-mic.zip,audio_mono-pickup_mix.zip,audio_hex-pickup_debleeded.zip}
unzip -j ./datasets/GuitarSet/audio_mono-mic.zip '*.wav' -d ./datasets/GuitarSet/data/mic && \
unzip -j ./datasets/GuitarSet/audio_mono-pickup_mix.zip '*.wav' -d ./datasets/GuitarSet/data/mix && \
unzip -j ./datasets/GuitarSet/audio_hex-pickup_debleeded.zip '*.wav' -d ./datasets/GuitarSet/data/hex_cln

```

We follow the *Senvaitytite train-test split* presented in [this repository](https://github.com/daliasen/GuitarStringSeparation-MF-NMF-NMFD). The filenames use for tetsing are stored in ```datasets/NMFtestSet.csv```. To get the dataset ready for training Wave-U-Net as string separator, first run:

```
cd datasets/
python prepare_source_sep_data.py # this creates dir datasep/
python prepare_source_sep_data-mic.py # this creates dir datasep-mic/
python prepare_source_sep_datamono-pickup.py # this creates dir datasep-mix/
cd -
```

After running these scripts, the following directories will be created:

- **`datasep/`** – for general source separation data
- **`datasep-mic/`** – for microphone-based separation data
- **`datasep-mix/`** – for pickup-mix separation data



### Algorithm 1 — Custom Audio Data for GS-Aux

Below we sketch out the algorithm to create GS-Aux, either for mic or pickup data:
```
Input:
  - Audio mixture x ∈ ℝᵀ from GuitarSet solos
  - Onset & string pairs {(oᵢ, sᵢ)}₍ᵢ₌₁₎ᴺ
Output:
  - String-wise diarized x′ ∈ ℝ⁶ˣᵀ
  - Overlaid melodies x″ ∈ ℝ⁶ˣᵀ″

Procedure:
  1. Initialize x′ ← small Gaussian noise, x″ ← ∅
  2. For each note i:
       if ∃j: |oᵢ - oⱼ| < 60 ms and sᵢ ≠ sⱼ → skip (chord)
       else:
         copy segment x[oᵢ:oᵢ₊₁] → x′₍ₛᵢ₎
         append same segment → x″₍ₛᵢ₎
  3. Set T″ ← 2nd-longest {‖x″ₛ‖}
  4. Crop/pad each x″ₛ to length T″
```


Run the following commands to implement Algorithm 1 above and extract **x′** and **x′′** for the dataset **GS-Aux-Mic**:
```
cd data-manipulation-code/
python create_gs-aux-solo.py --all_solos
python create_gs-aux-comp.py --all_solos
```

Copy Processed Data into ```GSCustomMic```:
```
mkdir -p GSCustomMic
cp -r ./pseudo_sep_all_solos_mic_wn/* GSCustomMic
cp -r ./pseudocomp_sep_all_solos_mic_wn/* GSCustomMic
```

Perform Train-Test Split: 
```
python pseudo_train_test_split.py -d GSCustomMic 
```
This will create ```GSCustomMic/{test, train}/```.

Last but not least:
```
mv GSCustomMic/ ../datasets/datasep-gscustmic/
rm -r ./pseudo_sep_all_solos_mic_wn/
rm -r ./pseudocomp_sep_all_solos_mic_wn/
```

Act similarly to create **GS-Aux-Pckp** dataset by running:
```
python create_gs-aux-solo.py --all_solos --pickup
python create_gs-aux-solo.py --all_solos --pickup
mkdir -p GSCustomPckp
cp -r ./pseudo_sep_all_solos_mix_wn/* GSCustomPckp
cp -r ./pseudocomp_sep_all_solos_mix_wn/* GSCustomPckp
python pseudo_train_test_split.py -d GSCustomPckp
mv GSCustomPckp/ ../datasets/datasep-gscustpckp/
rm -r ./pseudo_sep_all_solos_mix_wn/
rm -r ./pseudocomp_sep_all_solos_mix_wn/
```

Hence, our auxiliary datasets for separation are ready:
- **`datasep-gscustmic/`** - to train and test our data on our custom auxiliary **GS-Aux-Mic** dataset
- **`datasep-gscustpckp/`** - to train and test our data on our custom auxiliary **GS-Aux-Pckp** dataset

### Algorithm 2 — Custom Audio Data for ADGP auxiliary Dataset

```
Input:
  - GuitarSet solo audio x ∈ ℝᵀ
  - GuitarSet onset & string annotations {(oᵢ, sᵢ)}ᵢ₌₁ᴺ
  - DadaGP tablature tracks with note times {τₖ}ₖ₌₁ᴹ

Output:
  - Synthesized ADGP dataset (string-wise rendered audio)

Procedure:

Phase I — Note Event Collection
  1. Initialize event bank 𝔈 ← ∅
  2. For each annotated note (oᵢ, sᵢ):
       if ∃j ≠ i such that |oᵢ − oⱼ| < 60 ms and sᵢ ≠ sⱼ:
         skip (chord note)
       else:
         extract event e ← x[oᵢ : oᵢ₊₁]
         store (e, sᵢ) in 𝔈

Phase II — Tablature-Informed Rendering
  3. Randomly sample 5% of tablature tracks {τₖ}
  4. For each selected tablature track τ:
       initialize 6-channel signal y with small Gaussian noise
       for each tablature note (tⱼ, sⱼ) in τ:
         if ∃(e, sⱼ) ∈ 𝔈:
           rescale event e′ ← U[0.5, 1.0] · e
           write e′ into channel sⱼ at time tⱼ
```

Run the following commands to implement Algorithm 2 and extract **ADGP**.

For the creation of the ADGP-Mic dataset in particular we first need to gather note instances from the GuitarSet mic solos:
```
cd data-manipulation-code
python gather_notes.py --all_solos
```

The command will create dir ```note_instances_mic/data/```.
Next, we need to sonify the symbolic DadaGP (MIDI) data by rendering the gathered note events accordingly. This normally takes more than 10 min:

```
python midi2audio_recs.py --note_instances note_instances_mic/ --input_dir gp_token_examples --n_samples 30
```

The command above will create a set of audio tracks and store them in ```mdgp```. We then need to move this dir to the right place:
```
mv mdgp ../datasets/
cd -
```

Now, we can further insert these tracks into distinct training sets to create the bases for new separation and transcription experiments:
```
cp -r datasets/datasep-mic datasets/datasep-mic-mdgp
cp -r datasets/datasep-gscustmic datasets/datasep-gscustmic-mdgp
cp -r datasets/mdgp/* datasets/datasep-mic-mdgp/train/
cp -r datasets/mdgp/* datasets/datasep-gscustmic-mdgp/train/
```


And so we have created:
- **`mdgp/`** - this is **ADGP-Mic** dataset
- **`datasep-mic-mdgp/`** - this is **Mic & ADGP-Mic** dataset
- **`datasep-gscustmic-mdgp/`** - this is **GS-Aux-Mic & ADGP-mic** dataset


Act similarly to create GS-Aux-Pckp by running:
```
cd data-manipulation-code
python gather_notes.py --all_solos --pickup
python midi2audio_recs.py --note_instances note_instances_mic/ --input_dir gp_token_examples --n_samples 30 --out_dir pdgp
mv pdgp ../datasets/
cd -
etc. 
```

**Plot GuitarSet Note-String Histogram**

```
cd data-manipulation-code
python gather_notes.py --plot --all_tracks
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
python train.py --dataset_dir ../datasets/datasep-gscustmic/ --cuda --hdf_dir hdfs/hdf_guit-gscustmic --checkpoint_dir checkpoints/waveunet_guit_gscustmic --channels 1 --patience 200 --version pseudo
python train.py --dataset_dir ../datasets/datasep-mic-mdgp/ --cuda --hdf_dir hdfs/hdf_guit-mic-mdgp --checkpoint_dir checkpoints/waveunet_guit_mic_mdgp --channels 1 --patience 200 # TODO check need of version pseudo
python train.py --dataset_dir ../datasets/datasep-gscustmic-mdgp/ --cuda --hdf_dir hdfs/hdf_guit-gscustmic-mdgp --checkpoint_dir checkpoints/waveunet_guit_gscustmic_mdgp --channels 1 --patience 200  # TODO check need of version pseudo
```

Runs with pre-training:
```
python train.py --dataset_dir ../datasets/datasep-mic/ --cuda --hdf_dir hdfs/hdf_guit-mic --checkpoint_dir checkpoints/waveunet_guit_mic-pretOnHex --load_model checkpoints/waveunet_guit_hex/<best_checkpoint> --channels 1 --patience 200 
python train.py --dataset_dir ../datasets/datasep-mic-mdgp/ --cuda --hdf_dir hdfs/hdf_guit-mic-mdgp --checkpoint_dir checkpoints/waveunet_guit_mic-mdgp-pretOnHex --load_model checkpoints/waveunet_guit_hex/<best_checkpoint> --channels 1 --patience 200 
python train.py --dataset_dir ../datasets/datasep-gscustmic/ --cuda --hdf_dir hdfs/hdf_guit-gscustmic --checkpoint_dir checkpoints/waveunet_guit_gscustmic-pretOnMic --load_model checkpoints/waveunet_guit_mic/<best_checkpoint> --channels 1 --patience 200 --version pseudo
python train.py --dataset_dir ../datasets/datasep-gscustmic/ --cuda --hdf_dir hdfs/hdf_guit-gscustmic --checkpoint_dir checkpoints/waveunet_guit_gscustmic-pretOnHex --load_model checkpoints/waveunet_guit_hex/<best_checkpoint> --channels 1 --patience 200 --version pseudo
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
First, create the separated sources to be used by TabCNN:
```
cd Wave-U-Net-Pytorch-6string
cp -r ../datasets/datasep-{mix, mic} ../datasets/datasep-{mix, mic}-preds-{codename-of-waveunet}
export CUDA_VISIBLE_DEVICES=#

python multi-predict.py --load_model checkpoints/waveunet_guit_{codename-of-waveunet}/best_checkpoint_ --cuda --input_dir ../datasets/datasep-{mix, mic}-preds-{codename-of-waveunet}
```

Then, extract CQTs for all mixtures, reference and estimated sources:
```
cd ../TabCNN/data_multisource
cp -r ../../datasets/GuitarSet/data/annos/ ../data_multisource/GuitarSet/annotation
python Parallel_TabDataReprGenSep.py --input_path ../../datasets/datasep-{mix, mic}-preds-{codename-of-waveunet} 
```

Now, let's train the model
```
cd ../model-tensor-sep
python TabCNN.py --partition senvaityte --n_stfts 7 --data_path ../data_multisource/spec_repr_{codename-f-waveunet} 
```


