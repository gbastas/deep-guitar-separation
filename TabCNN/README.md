**TabCNN**

conda create -n tab-cnn python==3.9
conda activate tab-cnn
pip install -r requirements.txt


TODO: fix directories/paths

First, create the separated sources to be used by TabCNN:
```
cd Wave-U-Net-Pytorch-6string
cp -r ../datasets/datasep-{mix, mic} ../datasets/datasep-{mix, mic}-preds-{codename-of-waveunet}
python multi-predict.py --load_model checkpoints/waveunet_guit_{codename-of-waveunet}/best_checkpoint_ --cuda --input_dir ../datasets/datasep-{mix, mic}-preds-{codename-of-waveunet}
```

Then, extract CQTs for all mixtures, reference and estimated sources:
```
cd ../TabCNN/data_multisource
cp -r ../../datasets/GuitarSet/data/annos/ ../data_multisource/GuitarSet/annotation
python Parallel_TabDataReprGenSep.py --input_path ../../datasets/datasep-{mix, mic}-preds-{codename-of-waveunet} 
cp ../data_multisource/id.csv ../data_multisource/spec_repr_gscustmic-mid-pretOnMic/id.csv

```

Now, let's train the model
```
cd ../model-tensor-sep
python TabCNN.py --partition senvaityte --n_stfts 7 --data_path ../data_multisource/spec_repr_{codename-f-waveunet} 
```

e.g.
cd Wave-U-Net-Pytorch-6string
python multi-predict.py --load_model checkpoints/waveunet_guit_gscustmix-pretOnHex/best_checkpoint_549211 --input_dir ../datasets/datasep-mix-preds-gscustmix-pretOnHex --cuda
cd ../TabCNN/data_multisource
python Parallel_TabDataReprGenSep.py --input_path ../../datasets/datasep-mic-preds-gscustmic-overcomp-alt-pretOnMic
python TabCNN.py --partition senvaityte --data_path ../data_multisource/spec_repr_datasep-mic-preds-gscustmic-overcomp-alt-pretOnMic --n_stfts 7



<!-- 
Baseline TabCNN


mkdir data-orig/GuitarSet
ln -s ~/separate-and-transcribe/datasets/GuitarSet/data/annos/ ~/separate-and-transcribe/TabCNN/data-orig/GuitarSet/annotation
mkdir -p ~/separate-and-transcribe/TabCNN/data-orig//GuitarSet/audio/audio-mic
mkdir -p ~/separate-and-transcribe/TabCNN/data-orig//GuitarSet/audio/audio_hex-pickup_debleeded
ln -s ~/separate-and-transcribe/datasets/GuitarSet/data/mic ~/separate-and-transcribe/TabCNN/data-orig/GuitarSet/audio/audio_mic
ln -s ~/separate-and-transcribe/datasets/GuitarSet/data/mix ~/separate-and-transcribe/TabCNN/data-orig/GuitarSet/audio_mix
ln -s ~/separate-and-transcribe/datasets/GuitarSet/data/audio_hex-pickup_debleeded ~/separate-and-transcribe/TabCNN/data-orig/GuitarSet/audio/audio_hex-pickup_debleeded

mkdir data_multisource/GuitarSet/
ln -s ~/separate-and-transcribe/datasets/GuitarSet/data/annos/ ~/separate-and-transcribe/TabCNN/data_multisource/GuitarSet/annotation
cp -r ../datasets/GuitarSet/data/annos/ ../data_multisource/GuitarSet/annotation


![alt text](image.png)


cd data
python Parallel_TabDataReprGenSep.py --input_dir datasep-customic-pretonmic-preds

cd model-tensor-sep
CUDA_VISIBLE_DEVICES=0 python TabCNN.py --partition senvaityte
CUDA_VISIBLE_DEVICES=0 python TabCNN.py --partition senvaityte --data_path ../data/spec_repr_datasepmix_preds/ --epochs 10
CUDA_VISBLE_DEVICES=3 python TabCNN.py --partition_mode senvaityte --n_stfts 6 --data_path ../data/spec_repr_datasepmic_preds/ --epochs 10
CUDA_VISIBLE_DEVICES=2 python TabCNN.py --partition senvaityte --data_path ../data/spec_repr_datasep-customic-pretonmic-preds --n_stfts 7


python Parallel_TabDataReprGenSep.py --input_dir datasep-customicWithFake-pretonmic-preds
CUDA_VISIBLE_DEVICES=3 python TabCNN.py --partition senvaityte --data_path ../data/spec_repr_datasep-customicWithF --n_stfts 7
<!-- >>> datapipe['y_pred'].shape
(1824, 6, 21)
>>> data['y_pred'].shape
(74194, 6, 21) -->
cross-test
```
CUDA_VISIBLE_DEVICES=2 python TabCNN.py --test --saved_exp "c 2024-04-19 013637_datasepmix_preds" --data_path "../data/spec_repr_datasepmic_preds_pseudoboth_wn" --n_stfts 7 --partition senvaityte
```

# tab-cnn-torch gbastas notes

```
conda activate separation
cd model-torch
CUDA_VISIBLE_DEVICES=# python TabCNN.py
```
or 
```
conda activate separation
cd model
CUDA_VISIBLE_DEVICES=# python TabCNN.py {--partition_mode senvaityte} {{--test}}
```
mic results Senvaityte spit
pp [0.8927646938157522]
pr [0.7888195541285828]
pf [0.8375795010905625]
tp [0.8455901520637219]
tr [0.7534194724979785]
tf [0.7968483447896243]
tdr [0.947159041930296]

mix results Senvaityte spit
pp [0.8927646938157522]
pr [0.7888195541285828]
pf [0.8375795010905625]
tp [0.8455901520637219]
tr [0.7534194724979785]
tf [0.7968483447896243]
tdr [0.947159041930296]

# TODO

- save all epochs
- retrain from specific epoch
- run for more epochs
- check results for solo/comp separately -->



