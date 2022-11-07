This directory tree is expected in order to store your data and results:

```
.
├── constants.ini
├── data
│   ├── annos
│   ├── audio
│   └── train
└── results
```

In order to obtain data splits, either by guitarist or following the Senvaityte split, run:
```
python prepare_source_sep_data.py
```


To text pitch and onset accuracy levels with crepe and madomom accordingsly, run:
```
python tets_pitch.py
```