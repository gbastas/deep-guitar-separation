This code is based on repo https://github.com/facebookresearch/demucs. 

We have created `conf/variant/guitarsep_demucs.yaml` and `conf/variant.guitarsep_hdemucs.yaml` as variants of the original demucs configuration `conf/variant/congi_original.yaml` (that was made to be trained on MUSEDB). The main code changes were made in `./demucs/wav.py` to load training, validation and test data correctly. Also in `./demucs/evaluate.py` so that the averaged (not median) SDR values are considered as for Wave-U-Net.  

To train the standard Demucs model on GuitarSet:
```
CUDA_VISIBLE_DEVICES=0 dora run -d variant=guitarsep_demucs
python3 -m tools.export e96d2e8c
python3 -m tools.test_pretrained --repo ./release_models -n e96d2e8c test.shifts=0
demucs --repo ./release_models -n e96d2e8c 00_BN1-129-Eb_comp/mixture.wav
```
To train the standard HT Demucs model on GuitarSet:
```
CUDA_VISIBLE_DEVICES=0 dora run -d -f variant=guitarsep_hdemucs
python3 -m tools.export 0f9500b1
python3 -m tools.test_pretrained --repo ./release_models -n 0f9500b1 test.shifts=0
```