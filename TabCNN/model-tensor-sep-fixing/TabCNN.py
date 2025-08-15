''' A CNN to classify 6 fret-string positions
    at the frame level during guitar performance
'''

from __future__ import print_function
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda
from keras import backend as K
from DataGenerator import DataGenerator
import pandas as pd
import numpy as np
import datetime
from Metrics import *
import tensorflow as tf
import argparse
import csv



import random
import os

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Disable GPU non-determinism
    tf.config.experimental.enable_op_determinism()



class TabCNN:
    
    def __init__(self, 
                 batch_size=128, 
                 epochs=8,
                #  epochs=1, # TODO: comment-out
                 con_win_size = 9,
                 spec_repr="c",
                 data_path="",
                 id_file="id.csv",
                 save_path="./saved/",
                 test=False,
                 partition_mode='cross-val',
                 n_stfts=1):   

        self._set_gpu_config()

        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path
        self.partition_mode = partition_mode
        self.n_stfts = n_stfts

        self.load_IDs()

        if test:
            self.save_folder = os.path.join(self.save_path, args.saved_exp)
        else:
            self.save_folder = os.path.join(self.save_path, self.spec_repr + " " + datetime.datetime.now().strftime("%Y-%m-%d %H%M%S"))

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.log_file = os.path.join(self.save_folder, "log.txt")
        
        self.metrics = {}
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []
        self.metrics["data"] = ["g0","g1","g2","g3","g4","g5","mean","std dev"]

        # Add solo metrics
        self.metrics["pp_solo"] = []
        self.metrics["pr_solo"] = []
        self.metrics["pf_solo"] = []
        self.metrics["tp_solo"] = []
        self.metrics["tr_solo"] = []
        self.metrics["tf_solo"] = []
        self.metrics["tdr_solo"] = []

        # Add comp metrics
        self.metrics["pp_comp"] = []
        self.metrics["pr_comp"] = []
        self.metrics["pf_comp"] = []
        self.metrics["tp_comp"] = []
        self.metrics["tr_comp"] = []
        self.metrics["tf_comp"] = []
        self.metrics["tdr_comp"] = []        
        
        if self.spec_repr == "c":
            self.input_shape = (192, self.con_win_size, self.n_stfts)
        elif self.spec_repr == "m":
            self.input_shape = (128, self.con_win_size, 1)
        elif self.spec_repr == "cm":
            self.input_shape = (320, self.con_win_size, 1)
        elif self.spec_repr == "s":
            self.input_shape = (1025, self.con_win_size, 1)
            
        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6

    def _set_gpu_config(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("GPU memory growth set to True")
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU found. Using CPU instead.")

    def load_IDs(self):
        csv_file = os.path.join(self.data_path, self.id_file)
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])
        
    def partition_data(self, data_split):
        self.data_split = data_split
        self.partition = {}
        self.partition["training"] = []
        self.partition["validation"] = []
        self.partition["validation-solos"] = []
        self.partition["validation-comps"] = []

        if self.partition_mode == 'senvaityte':
            with open('../data_multisource/GuitarSet/NMFtestSet.csv', newline='') as csvfile:
                testreader = csv.reader(csvfile, delimiter=',')
                testfiles = ['_'.join(row[4].split('_',2)[:2]) for row in testreader] # e.g. 00_Funk1-114-Ab_comp_hex_cln.wav --> 00_Funk1-114-Ab

            for ID in self.list_IDs:
                # print('ID:', ID)
                recording_name = '_'.join(ID.split("_")[:2]) # e.g. 04_Jazz3-150-C_solo_0 --> 04_Jazz3-150-C
                performance_style = ID.split("_")[-2]
                if recording_name in testfiles:
                    self.partition["validation"].append(ID)
                    if performance_style=='solo':
                        self.partition["validation-solos"].append(ID)
                    if performance_style=='comp':
                        self.partition["validation-comps"].append(ID)

                else:            
                    self.partition["training"].append(ID)

        if self.partition_mode == 'cross-val':
            for ID in self.list_IDs:
                guitarist = int(ID.split("_")[0]) # e.g. 04_Jazz3-150-C_solo_0 --> 4
                if guitarist == data_split:
                    self.partition["validation"].append(ID)
                else:
                    self.partition["training"].append(ID)

        # print('self.n_stfts', self.n_stfts)
        if not args.test:
            self.training_generator = DataGenerator(self.partition['training'], 
                                                    data_path=self.data_path, 
                                                    batch_size=self.batch_size, 
                                                    shuffle=True,
                                                    spec_repr=self.spec_repr, 
                                                    con_win_size=self.con_win_size,
                                                    n_stfts=self.n_stfts)
        

        print(len(self.partition['validation']))
        print(len(self.partition['validation-solos']))

        self.validation_generator = DataGenerator(self.partition['validation'], 
                                                data_path=self.data_path, 
                                                batch_size=len(self.partition['validation']), 
                                                shuffle=False,
                                                spec_repr=self.spec_repr, 
                                                con_win_size=self.con_win_size,
                                                n_stfts=self.n_stfts)

        self.validation_generator_solo = DataGenerator(self.partition['validation-solos'], 
                                                data_path=self.data_path, 
                                                batch_size=len(self.partition['validation-solos']), 
                                                shuffle=False,
                                                spec_repr=self.spec_repr, 
                                                con_win_size=self.con_win_size,
                                                n_stfts=self.n_stfts)


        self.validation_generator_comp = DataGenerator(self.partition['validation-comps'], 
                                                data_path=self.data_path, 
                                                batch_size=len(self.partition['validation-comps']), 
                                                shuffle=False,
                                                spec_repr=self.spec_repr, 
                                                con_win_size=self.con_win_size,
                                                n_stfts=self.n_stfts)


        self.split_folder = os.path.join(self.save_folder, str(self.data_split))
        if not os.path.exists(self.split_folder):
            os.makedirs(self.split_folder)
                
    def log_model(self):
        with open(self.log_file,'w') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nspec_repr: " + str(self.spec_repr))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
       
    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:,i,:]), axis=1))
        return K.concatenate(string_sm, axis=1)
    
    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(target[:,i,:], output[:,i,:])
        return loss
    
    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
           
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))   
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes * self.num_strings)) # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        model.compile(loss=self.catcross_by_string,
                      optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
                    #   optimizer=keras.optimizers.Adadelta(), # old
                      metrics=[self.avg_acc])
        
        self.model = model

    def train(self):
        self.model.fit_generator(generator=self.training_generator,
                    validation_data=None,
                    epochs=self.epochs,
                    verbose=1,
                    use_multiprocessing=True,
                    # use_multiprocessing=False, #  to avoid error
                    workers=32)
                    # workers=14)
        # )
    def save_weights(self):
        self.model.save_weights(os.path.join(self.split_folder, "weights.h5"))


    # def load_weights(self):
    #     self.model.load_weights(os.path.join(tabcnn.split_folder, "weights.h5"))
    def load_weights(self):
        weights_path = os.path.join(self.save_folder, "0", "weights.h5")  # Ensure the correct path
        self.model.load_weights(weights_path)
        print(f"Loaded model weights from {weights_path}")

    def test(self):
        # Get the test data
        self.X_test, self.y_gt = self.validation_generator[0]
        print("X_test and y_gt loaded")

        self.X_test_solo, self.y_gt_solo = self.validation_generator_solo[0]
        print("X_test_solo and y_gt_solo loaded")

        self.X_test_comp, self.y_gt_comp = self.validation_generator_comp[0]
        print("X_test_comp and y_gt_comp loaded")
        
        # Predict for all categories
        self.y_pred = self.model.predict(self.X_test)
        print(f"Prediction for X_test: {self.y_pred.shape}")

        self.y_pred_solo = self.model.predict(self.X_test_solo)
        print(f"Prediction for X_test_solo: {self.y_pred_solo.shape}")

        self.y_pred_comp = self.model.predict(self.X_test_comp)
        print(f"Prediction for X_test_comp: {self.y_pred_comp.shape}")
        

    def save_predictions(self):
        # predictions_path = os.path.join(self.save_folder, "predictions.npz")
        
        # if os.path.exists(predictions_path):
        #     print(f"Predictions file already exists: {predictions_path}. Skipping save.")
        #     return predictions_path        
        
        np.savez(
            os.path.join(self.save_folder, "predictions.npz"),
            y_pred=self.y_pred,
            y_gt=self.y_gt,
            y_pred_solo=self.y_pred_solo,
            y_gt_solo=self.y_gt_solo,
            y_pred_comp=self.y_pred_comp,
            y_gt_comp=self.y_gt_comp
        )

        return os.path.join(self.save_folder, "predictions.npz")

    def load_predictions(self):
        data = np.load(os.path.join(self.save_folder, "predictions.npz"))
        self.y_pred = data['y_pred']
        self.y_gt = data['y_gt']
        self.y_pred_solo = data['y_pred_solo']
        self.y_gt_solo = data['y_gt_solo']
        self.y_pred_comp = data['y_pred_comp']
        self.y_gt_comp = data['y_gt_comp'] 
        # return y_pred, y_gt


    def evaluate(self):
        print('avg_acc', self.avg_acc(self.y_pred, self.y_gt))
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))

        # Evaluate for solos
        print('avg_acc_solo', self.avg_acc(self.y_pred_solo, self.y_gt_solo))
        self.metrics["pp_solo"].append(pitch_precision(self.y_pred_solo, self.y_gt_solo))
        self.metrics["pr_solo"].append(pitch_recall(self.y_pred_solo, self.y_gt_solo))
        self.metrics["pf_solo"].append(pitch_f_measure(self.y_pred_solo, self.y_gt_solo))
        self.metrics["tp_solo"].append(tab_precision(self.y_pred_solo, self.y_gt_solo))
        self.metrics["tr_solo"].append(tab_recall(self.y_pred_solo, self.y_gt_solo))
        self.metrics["tf_solo"].append(tab_f_measure(self.y_pred_solo, self.y_gt_solo))
        self.metrics["tdr_solo"].append(tab_disamb(self.y_pred_solo, self.y_gt_solo))

        # Evaluate for comps
        print('avg_acc_comp', self.avg_acc(self.y_pred_comp, self.y_gt_comp))
        self.metrics["pp_comp"].append(pitch_precision(self.y_pred_comp, self.y_gt_comp))
        self.metrics["pr_comp"].append(pitch_recall(self.y_pred_comp, self.y_gt_comp))
        self.metrics["pf_comp"].append(pitch_f_measure(self.y_pred_comp, self.y_gt_comp))
        self.metrics["tp_comp"].append(tab_precision(self.y_pred_comp, self.y_gt_comp))
        self.metrics["tr_comp"].append(tab_recall(self.y_pred_comp, self.y_gt_comp))
        self.metrics["tf_comp"].append(tab_f_measure(self.y_pred_comp, self.y_gt_comp))
        self.metrics["tdr_comp"].append(tab_disamb(self.y_pred_comp, self.y_gt_comp))

    def save_results_csv(self):
        # results_path = os.path.join(self.save_folder, "results.csv")
        
        # if os.path.exists(results_path):
        #     print(f"Results file already exists: {results_path}. Skipping save.")
        #     return        
        output = {}
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]

        # Metrics for solos
        for key in self.metrics.keys():
            if 'solo' in key:
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]
        
        # Metrics for comps
        for key in self.metrics.keys():
            if 'comp' in key:
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]

        # output["data"] =  self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(os.path.join(self.save_folder, "results.csv"))
        
##################################
########### EXPERIMENT ###########
##################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on guitar data.')

    parser.add_argument('--test', action="store_true")
    parser.add_argument('--partition_mode', type=str, default="cross-val", help='cross-val or senvaityte.')
    parser.add_argument('--data_path', type=str, default="../data/spec_repr_datasepmix_target7/", help='spec_repr_mix OR spec_repr_mic')
    parser.add_argument('--load2retrain', action="store_true", help="Load existing weights for retraining instead of starting from scratch.")
    parser.add_argument('--n_stfts', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--saved_exp', type=str, help='Experiment ID for the saved model (e.g., c 2024-11-12 160033)')
    args = parser.parse_args()

    if args.test and not args.saved_exp:
        raise ValueError("The --test flag requires the --saved_exp argument to specify the experiment directory.")
    
    tabcnn = TabCNN(epochs=args.epochs, data_path=args.data_path, test=args.test, partition_mode=args.partition_mode, n_stfts=args.n_stfts)

    print("logging model...")
    tabcnn.build_model()
    tabcnn.log_model()

    for fold in range(6): 
    # for fold in range(1): # TODO: comment-out
        print("\nfold " + str(fold))
        tabcnn.partition_data(fold)
        print("building model...")
        tabcnn.build_model() 
        if args.load2retrain:
            tabcnn.load_weights()
        if not args.test:
            print("training...")
            tabcnn.train()
            tabcnn.save_weights()
        
        print("testing...")

        tabcnn.load_weights()
        print("Weights loaded!")

        tabcnn.test()
        print("Test Finished!")
        path = tabcnn.save_predictions()
        # else:
        #     path=os.path.join(tabcnn.save_folder, "predictions.npz")
        print("Predictions Saved!", path)
        tabcnn.load_predictions()    
        print("Evaluation...")
        tabcnn.evaluate()
        if args.partition_mode == 'senvaityte':
            break
    print("saving results...")
    tabcnn.save_results_csv()
