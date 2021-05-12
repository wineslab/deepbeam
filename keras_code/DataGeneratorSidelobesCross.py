import numpy as np
import math
import keras
import random

from DataGeneratorSidelobes_w_num_tx_beams import DataGeneratorSidelobes
from keras.utils.io_utils import HDF5Matrix



class DataGeneratorSidelobesCross(keras.utils.Sequence):
    'Generates data for cross training and testing in Keras.'

    def __init__(
        self,
        indexes,
        batch_size,
        data_path,  # data_path can be a list or tuple, and indexes must be divisible by len(data_path)
        num_tx_beams,
        num_blocks_per_frame,
        num_samples_per_block,
        how_many_blocks_per_frame,
        shuffle=False,
        is_2d=False
    ):
        'Initialization'
        self.indexes = indexes
        self.batch_size = batch_size
        self.data_path = data_path
        self.shuffle = shuffle
        self.cache = {}

        self.num_blocks_per_frame = num_blocks_per_frame
        self.num_samples_per_block = num_samples_per_block
        self.how_many_blocks_per_frame = how_many_blocks_per_frame
        self.num_tx_beam = num_tx_beams
        self.num_rx_beam = 1
        self.is_2d = is_2d
        self.cache_rate = 10

        self.num_datasets = len(data_path)
        self.child_dg = []
        self.dg_to_use = -1
        indexes_list = list(xrange(len(indexes)))
        random.shuffle(indexes_list)
        num_elem_per_dg = len(indexes_list) / self.num_datasets
        print("Number of elements from each DG %d" % num_elem_per_dg)
        for i in range(self.num_datasets):
            start_idx = i * num_elem_per_dg
            end_idx = (i + 1) * num_elem_per_dg
            this_dg_indexes = [self.indexes[x] for x in sorted(indexes_list[start_idx:end_idx])]
            # print(this_dg_indexes)
            self.child_dg.append(
                DataGeneratorSidelobes(
                    this_dg_indexes,
                    self.batch_size,
                    data_path[i],
                    self.num_tx_beam,
                    self.num_blocks_per_frame,
                    self.num_samples_per_block,
                    self.how_many_blocks_per_frame,
                    self.shuffle,
                    self.is_2d
                )
            )


    def __len__(self):
        'Denotes the number of batches per epoch.'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def build_cache(self):
        '''Add indexes to cache.'''
        size = self.__len__()
        for i in range(size):
            print("Adding to cache " + str(i) + "...")
            self.cache[i] = self.__getitem__(i)

    def read_from_cache(self, index):
        return None if index not in self.cache else self.cache[index]

    def __getitem__(self, index):
        'Generate one batch of data.'

        if index in self.cache:
            return self.cache[index]

        # round robin
        index_dg = int(math.floor(index / self.num_datasets))
        self.dg_to_use = (self.dg_to_use + 1) % self.num_datasets
        # print("Actual index %d dg %d" % (index_dg, self.dg_to_use))
        return self.child_dg[self.dg_to_use].__getitem__(index_dg)
