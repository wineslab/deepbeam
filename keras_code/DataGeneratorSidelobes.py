import numpy as np
import keras

from keras.utils.io_utils import HDF5Matrix


class DataGeneratorSidelobes(keras.utils.Sequence):
    'Generates data for Keras.'

    def __init__(
        self,
        indexes,
        batch_size,
        data_path,
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
        self.angle = HDF5Matrix(self.data_path, "angle")
        self.num_blocks_per_frame = num_blocks_per_frame
        self.num_samples_per_block = num_samples_per_block
        self.how_many_blocks_per_frame = how_many_blocks_per_frame
        self.num_tx_beam = 3
        self.num_rx_beam = 1
        self.num_angles = 3
        self.is_2d = is_2d
        self.cache_rate = 10
    #     self.build_cache()
    #     self.on_epoch_end()
    #
    # def on_epoch_end(self):
    #     self.indexes = np.arange(len(self.ids))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

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
                      self.num_angles * self.num_rx_beam), dtype=int)

        for i, idx in enumerate(indexes):
            first_iq = idx * self.num_blocks_per_frame * self.num_samples_per_block
            last_iq = first_iq + self.how_many_blocks_per_frame * self.num_samples_per_block

            X[i, ] = np.reshape(
                self.iq[first_iq:last_iq],
                (self.how_many_blocks_per_frame, self.num_samples_per_block, 2)
            )

            angle_samples = self.angle[first_iq:last_iq]

            # check that all the IQ samples we read are for the same TX beam
            if(len(set(angle_samples)) != 1):
                raise ValueError('selected samples for different angles')

            if(angle_samples[0] == -45):
                y[i, 0] = 1
            elif(angle_samples[0] == 0):
                y[i, 1] = 1
            elif(angle_samples[0] == 45):
                y[i, 2] = 1
            else:
                raise ValueError('unknown angle for class')

        # this is done just to use 2D models and add a new dimension
        # if self.is_2d:
        #     X = np.expand_dims(X, 1)

        return X, y

    def __fetch_index(self, index):
        self.x_out = self.X[index]
        self.y_out = self.Y[index]

    # def on_epoch_end(self):
    #    'Updates indexes after each epoch'
    #    self.indexes = self.indexes
