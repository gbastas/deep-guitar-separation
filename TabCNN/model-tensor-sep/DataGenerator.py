import numpy as np
import keras

from keras.utils.data_utils import Sequence
import os 
from collections import OrderedDict
# class DataGenerator(keras.utils.all_utils.Sequence):
    
class DataGenerator(Sequence):
    def __init__(self, list_IDs, data_path="../data/spec_repr_target/", batch_size=128,
                    shuffle=True, label_dim=(6,21), spec_repr="c", con_win_size=9, n_stfts=1,
                    max_cached_files=64, use_memmap=True):        
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_dim = label_dim
        self.spec_repr = spec_repr
        self.con_win_size = con_win_size
        self.halfwin = con_win_size // 2
        self.n_stfts = n_stfts
        
        # NEW
        self.max_cached_files = max_cached_files
        self.use_memmap = use_memmap
        self._cache = OrderedDict()  # filename -> (padded_repr, labels)

        if self.spec_repr == "c":
            self.X_dim = (self.batch_size, 192, self.con_win_size, self.n_stfts)
        elif self.spec_repr == "m":
            self.X_dim = (self.batch_size, 128, self.con_win_size, 1)
        elif self.spec_repr == "cm":
            self.X_dim = (self.batch_size, 320, self.con_win_size, 1)
        elif self.spec_repr == "s":
            self.X_dim = (self.batch_size, 1025, self.con_win_size, 1)
            
        self.y_dim = (self.batch_size, self.label_dim[0], self.label_dim[1])
        
        self.on_epoch_end()
        
    def __len__(self):
        # number of batches per epoch
        return int(np.floor(float(len(self.list_IDs)) / self.batch_size))
    
    def __getitem__(self, index):
        # generate indices of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y

    def get_ids_for_batch(self, index):
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return [self.list_IDs[k] for k in idxs]

    def _load_file_cached(self, full_path):
        # LRU hit
        if full_path in self._cache:
            self._cache.move_to_end(full_path)
            return self._cache[full_path]

        loaded = np.load(full_path, mmap_mode='r' if self.use_memmap else None, allow_pickle=False)
        repr_array = loaded["repr"]
        if repr_array.ndim < 3:
            repr_array = np.expand_dims(repr_array, axis=-1)
        # pad once per file
        padded = np.pad(repr_array, [(self.halfwin, self.halfwin), (0,0), (0,0)], mode='constant')
        labels = loaded["labels"]

        # LRU insert
        if len(self._cache) >= self.max_cached_files:
            self._cache.popitem(last=False)
        self._cache[full_path] = (padded, labels)
        return padded, labels


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty(self.X_dim, dtype=np.float32)
        y = np.empty(self.y_dim, dtype=np.float32)

        data_dir = os.path.join(self.data_path, self.spec_repr)

        for i, ID in enumerate(list_IDs_temp):
            filename = "_".join(ID.split("_")[:-1]) + ".npz"
            frame_idx = int(ID.split("_")[-1])
            full_path = os.path.join(data_dir, filename)

            padded_repr, labels = self._load_file_cached(full_path)

            # window [frame_idx : frame_idx + con_win_size] on the already padded array
            sample_x = padded_repr[frame_idx:frame_idx + self.con_win_size]  # (win, F, C)
            # moveaxis is cheap and keeps view semantics when possible
            X[i] = np.moveaxis(sample_x, 0, 1)
            y[i] = labels[frame_idx].astype(np.float32, copy=False)

        return X, y



    # def __data_generation(self, list_IDs_temp):
    #     #Generates data containing batch_size samples
    #     # X : (n_samples, *dim, n_channels)
        
    #     # Initialization
    #     X = np.empty(self.X_dim)
    #     x = [[] for _ in range(len(list_IDs_temp))]  # List of lists for each ID
    #     y = np.empty(self.y_dim)

    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
            
    #         # determine filename
    #         data_dir = os.path.join(self.data_path , self.spec_repr)
    #         filename = "_".join(ID.split("_")[:-1]) + ".npz"
    #         frame_idx = int(ID.split("_")[-1])
            
    #         # load a context window centered around the frame index


    #         loaded = np.load(os.path.join(data_dir, filename[:-4] + ".npz"))
    #         # print('loaded["repr"].shape', loaded["repr"].shape)
    #         repr_array = loaded["repr"]
    #         if len(loaded["repr"].shape)<3:
    #             repr_array = np.expand_dims(loaded["repr"], axis=-1)
    #         full_x = np.pad(repr_array, [(self.halfwin,self.halfwin), (0,0), (0,0)], mode='constant')
    #         sample_x = full_x[frame_idx : frame_idx + self.con_win_size]
    #         # print('sample_x', sample_x.shape )
    #         # print('X[i,]', X[i,].shape )
    #         X[i,] = np.swapaxes(sample_x, 0, 1) # (128, 192,9,7)
    #         # print('X', X.shape )

    #         # Store label
    #         y[i,] = loaded["labels"][frame_idx]

    #     return X, y
        


            # X[i,] = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    