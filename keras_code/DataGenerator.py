import numpy as np
import keras

from keras.utils.io_utils import HDF5Matrix


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras.'

    def __init__(
        self,
        indexes,
        batch_size,
        data_path,
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
        self.iq = HDF5Matrix(self.data_path, "iq")
        self.tx_beam = HDF5Matrix(self.data_path, "tx_beam")
        self.rx_beam = HDF5Matrix(self.data_path, "rx_beam")
        self.gain = HDF5Matrix(self.data_path, "gain")
        self.num_blocks_per_frame = num_blocks_per_frame
        self.num_samples_per_block = num_samples_per_block
        self.how_many_blocks_per_frame = how_many_blocks_per_frame
        self.num_tx_beam = num_tx_beams
        self.num_rx_beam = 1
        self.is_2d = is_2d
        self.cache_rate = 10

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

        indexes = self.indexes[index * self.batch_size:
                               (index + 1) * self.batch_size]

        # print(index, indexes)
        # print("indexes %s" % indexes)
        X = np.empty((self.batch_size, self.how_many_blocks_per_frame,
                      self.num_samples_per_block, 2))
        y = np.zeros((self.batch_size,
                      self.num_tx_beam * self.num_rx_beam), dtype=int)

        for i, idx in enumerate(indexes):

            first_iq = idx * self.num_blocks_per_frame * self.num_samples_per_block
            last_iq = first_iq + self.how_many_blocks_per_frame * self.num_samples_per_block

            X[i, ] = np.reshape(
                self.iq[first_iq:last_iq],
                (self.how_many_blocks_per_frame, self.num_samples_per_block, 2)
            )

            tx_beam_samples = self.tx_beam[first_iq:last_iq]

            # check that all the IQ samples we read are for the same TX beam
            if(len(set(tx_beam_samples)) != 1):
                raise ValueError('selected tx samples for different beams')

            y[i, int(tx_beam_samples[0])] = 1
        if not self.is_2d:
            # this is done just to use 2D models and add a new dimension
            X = np.squeeze(X)

        return X, y

    def __fetch_index(self, index):
        self.x_out = self.X[index]
        self.y_out = self.Y[index]

    # def on_epoch_end(self):
    #    'Updates indexes after each epoch'
    #    self.indexes = self.indexes
