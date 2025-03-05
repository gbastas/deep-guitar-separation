# tab-demo


## Retrieving Note Instances from GutarSet

```
cd backend
python AuxDataPrep.py --action gather_notes
```
A dir note_instances will be created.


## Creating Solo Target reconstructions from GutarSet

```
cd backend
python AuxDataPrep.py --action pseudo_sep {--all_solos}
python AuxDataPrep.py --action pseudocomp_sep {--all_solos}
```


