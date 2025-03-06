import math
import matplotlib.pyplot as plt

def hz_to_midi(fundamental):
    return round(12*math.log(fundamental/440,2)+69)

def midi_to_hz(midi):
    return 440*2**((midi-69)/12)

def determine_combinations(fundamental, constants): # changed by __gbastas__
    res = []

    midi_note = hz_to_midi(fundamental)
    fretboard = [range(x, x + constants.no_of_frets) for x in constants.tuning]
    for index, x in enumerate(fretboard):
        if midi_note in list(x):
            res.append((index, midi_note-constants.tuning[index])) # index is string, second is fret

    if res == []:
        print('No combinations were found.')
        # exit()

    return res

def close_event(self): # just for plotting
    plt.close() #timer calls this function after 3 seconds and closes the window     