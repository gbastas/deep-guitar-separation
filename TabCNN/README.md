Baseline TabCNN


mkdir data-orig/GuitarSet
ln -s ~/separate-and-transcribe/datasets/GuitarSet/data/annos/ ~/separate-and-transcribe/TabCNN/data-orig/GuitarSet/annotation


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
- check results for solo/comp separately



