import numpy as np
import keras
    
def tab2pitch(tab):
    pitch_vector = np.zeros(44)
    string_pitches = [40, 45, 50, 55, 59, 64]
    for string_num in range(len(tab)):
        # print('tab', tab[:, :].shape)
        # print('string_num', string_num)
        fret_vector = tab[string_num]
        fret_class = np.argmax(fret_vector, -1)
        # print('fret_class', fret_class)
        # 0 means that the string is closed 
        if fret_class > 0:
            pitch_num = fret_class + string_pitches[string_num] - 41
            # print('fret_class', fret_class, 'pitch_num', pitch_num) # **************************
            # print('tab', np.round(tab[string_num, :],1))
            pitch_vector[pitch_num] = 1
    # if np.where(pitch_vector == 1)[0].any():
    #     print('pitch_vector', np.where(pitch_vector == 1))  
    # print('pitch_vector', pitch_vector)
    
    return pitch_vector

def tab2bin(tab):
    tab_arr = np.zeros((6,20))
    for string_num in range(len(tab)):
        fret_vector = tab[string_num]
        fret_class = np.argmax(fret_vector, -1)
        # 0 means that the string is closed 
        if fret_class > 0:
            fret_num = fret_class - 1
            tab_arr[string_num][fret_num] = 1
    return tab_arr


def pitch_precision(pred, gt):
    # print('pred', pred.shape)
    # pred: (3798, 6, 21)
    pitch_pred = np.array(list(map(tab2pitch,pred)))
    # aaa
    pitch_gt = np.array(list(map(tab2pitch,gt)))
    # print('pitch_pred', pitch_pred.shape)
    pitch_gt = np.array(list(map(tab2pitch,gt)))
    # print('pitch_gt', pitch_gt.shape)
    numerator = np.sum(np.multiply(pitch_pred, pitch_gt).flatten())
    denominator = np.sum(pitch_pred.flatten())
    # print('pitch_pred_flatten', pitch_pred.flatten().shape)
    # print('pitch_pred_flatten == 1', np.where(pitch_pred.flatten() == 1))
    # print('pitch_pred == 1', np.where(pitch_pred == 1))
    # print('denominator',denominator)
    return (1.0 * numerator) / denominator

def pitch_recall(pred, gt):
    pitch_pred = np.array(list(map(tab2pitch,pred)))
    pitch_gt = np.array(list(map(tab2pitch,gt)))
    numerator = np.sum(np.multiply(pitch_pred, pitch_gt).flatten())
    denominator = np.sum(pitch_gt.flatten())
    return (1.0 * numerator) / denominator

def pitch_f_measure(pred, gt):
    p = pitch_precision(pred, gt)
    r = pitch_recall(pred, gt)
    f = (2 * p * r) / (p + r)
    return f

def tab_precision(pred, gt):
    # get rid of "closed" class, as we only want to count positives
    tab_pred = np.array(list(map(tab2bin,pred)))
    tab_gt = np.array(list(map(tab2bin,gt)))
    numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
    denominator = np.sum(tab_pred.flatten())
    return (1.0 * numerator) / denominator

def tab_recall(pred, gt):
    # get rid of "closed" class, as we only want to count positives
    tab_pred = np.array(list(map(tab2bin,pred)))
    tab_gt = np.array(list(map(tab2bin,gt)))
    numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
    denominator = np.sum(tab_gt.flatten())
    return (1.0 * numerator) / denominator

def tab_f_measure(pred, gt):
    p = tab_precision(pred, gt)
    r = tab_recall(pred, gt)
    f = (2 * p * r) / (p + r)
    return f

def tab_disamb(pred, gt):
    tp = tab_precision(pred, gt)
    pp = pitch_precision(pred, gt)
    return tp / pp