
# Training

Training data are stored in a dir of this type"
```
./data/train*
```

An instance StringBetas() class (see string_beta.py) is created and the beta values values for each string are learned by running demo_utils.train(). Each note is handled autonomously. 

For each one we call StringBetasObj.input_instance(note_audio, ....) loading the whole train open-string audio recording.

There, a NoteInstance() object is created (see inharmonicAnalysis.py)

```
demo_utils.train()  # [demo_utils.py]
        ---
        |i| # loop over strings
        --> get_onsets (Onset Detection & Audio Crop) 
        --> init_note_instance(string{i}_note_audio, ...) 
            --> inharmonic_Analysis.NoteInstance() # init() [inharmonic_Analysis.py]
                --> self.fundamental = self.bancho_pitch_estimation()
                --> self.iterative_compute_of_partials_and_betas()
                    ---- *******************************************************
                    |lim| # loop over increasing set of partials [6, 31, step=2]
                    ---> self.partials = self.find_partials(lim, ...) # compute partials for note instance. window arround k*f0 that the partials are tracked searching for highest peak.
                        --- 
                        |k| # loop over for peak picking around estimated partial posittions 
                        --> center_freq = self.__ployfit_centering_func(k,f0,a,b,c) # Barbancho et al. Section IV.C.§2
                        --> filtered = self.__zero_out()
                        --> peaks_idx = [np.argmax(np.abs(filtered))]
                        --> PartialList.append(Partial(freqHz=center_freq, ..., peak_idx=peaks_idx[0])) [inharmonic_Analysis.py]
                    ---> self.deviations = self.__compute_deviations(self.fundamental, self.partials) #  = [abs(partial.partial_Hz - partial.order*f0) for partial in partials]
                    ---> self.beta, self.abc = self.__compute_beta_with_regression(self.deviations, orders) 
                        --> [a,b,c] = compute_least(orders, deviations) # lsq

```
