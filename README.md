# Guitar String Separation

This repo is based on https://github.com/f90/Wave-U-Net-Pytorch.

## Datapreparation

```
cd exps/GuitarSet/
python prepare_source_sep_data.py
```
(and simply mv files by-patterns commands)

## Experiment

Clean training:
```
python train.py --dataset_dir datasets/GuitarSep/senvaityte_norm/ --cuda --hdf_dir hdfs/hdf --checkpoint_dir checkpoints/waveunet --log_dir logs/waveunet --channels 1 --patience 80
```

Testing:
```
python train.py --dataset_dir datasets/GuitarSep/senvaityte_norm/ --cuda --hdf_dir hdfs/hdf --checkpoint_dir checkpoints/waveunet --log_dir logs/waveunet --channels 1 --patience -1
```

## Results Visualisation

```
cd Wave-U-Net-Pytorch-6string
tensorboard --log_dir logs/
```