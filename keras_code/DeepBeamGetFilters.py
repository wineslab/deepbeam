import argparse
import h5py
import json
import keras
from keras.models import load_model
import numpy as np
import pickle as pkl
import time
import os
import collections


from keras.models import Model

from keras.utils.io_utils import HDF5Matrix
from keras.models import model_from_json
from DataGenerator import DataGenerator

from Utils import *

class DeepBeamGetFilters(object):

    def __init__(self):
        '''Initialize class variables.'''
        self.args = self.parse_arguments()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.id_gpu)

        self.layer_num = self.args.layer_num
        self.num_classes = self.args.num_classes
        self.is_2d = False
        self.load_from_json(self.args.model_dir_path)
        self.load_testing_data(self.args.model_dir_path)
        self.show_filters(self.layer_num)

    def load_from_json(self, folder):
        # load json and create model
        json_file = open(folder + "/model_arch.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(folder + "/DeepBeam_model.hdf5")
        print("Loaded model from disk")

    def show_filters(self, layer_num):

        model = self.model
        model.summary()

        # kernels_dict = collections.defaultdict(list)
        # dims = model.layers[layer_num].get_weights()[0].shape[2]
        #
        # for dim in range(dims):
        #     kernels = conv2_get_kernels(model, dim, layer_num)
        #     for idx, kernel_dim in enumerate(kernels):
        #         kernels_dict[idx] += kernel_dim.tolist(),


        # new_x, new_y, idx = [], [], 0
        # for b in range(self.num_of_batches):
        #     X, y = self.test_generator.__getitem__(b)
        #     for i, v in enumerate(y):
        #         cl = np.argmax(v)
        #         if (cl == 1 or cl == 12):
        #             new_x += X[i],
        #             new_y += cl,
        #             idx += 1
        #
        # new_x, new_y = np.array(new_x), np.array(new_y)
        # a = {'X': new_x, 'y': new_y}
        #
        # pkl.dump(a, open("/home/frestuc/projects/beam_learning_mmwave/classes_1_12.pkl", "wb"))

        # print("Loading...")
        # dict = pkl.load(open("/home/frestuc/projects/beam_learning_mmwave/classes_1_12.pkl", "rb"))
        # print("Loaded")

        # X, y = dict['X'], dict['y']
        #
        # print(X.shape, y.shape)

        conv_0 = Model(inputs=model.input,
                       outputs=model.layers[1].output)

        flatten = Model(inputs=model.input,
                       outputs=model.layers[14].output)

        classes = {
            x: collections.defaultdict(int)
            for x in range(self.num_classes)
        }
        counts = [
            0
            for i in range(self.num_classes)
        ]

        my_model = conv_0

        for b in range(self.num_of_batches):
            print("Analyzing batch #:  " + str(b) + "...")
            X, y = self.test_generator.__getitem__(b)
            for m, x in enumerate(X):
                print("Index in batch: " + str(m))
                inp = np.expand_dims(x, axis=0)
                out = np.squeeze(my_model.predict(inp))
                out = np.transpose(out, (2, 0, 1))
                for i in range(out.shape[0]): # filter num
                    avg_activation = 0
                    for j in range(out.shape[1]):
                        for k in range(out.shape[2]):
                            avg_activation += out[i, j, k]
                    avg_activation /= (out.shape[1] * out.shape[2])
                    class_id = np.argmax(y[m])
                    classes[class_id][i] += avg_activation
                    counts[class_id] += 1

            print(classes)
            print(counts)

        # for m, x in enumerate(X):
        #     print("IDX: " + str(m))
        #     inp = np.expand_dims(x, axis=0)
        #     out = np.squeeze(my_model.predict(inp))
        #     out = np.transpose(out, (2, 0, 1))
        #     print(out.shape) # 64, 5, 2048
        #     if y[m] == 1:
        #         for i in range(out.shape[0]):
        #             avg_activation = 0
        #             for j in range(out.shape[1]):
        #                 for k in range(out.shape[2]):
        #                     avg_activation += out[i, j, k]
        #             avg_activation /= (out.shape[1] * out.shape[2])
        #             if y[m] == 1:
        #                 first_class[i] += avg_activation
        #                 num_11 += 1
        #
        # print(first_class)
        # print(second_class)



    def load_testing_data(self, model_dir_path):
        '''Load data from path into framework.'''

        print('--------- Loading from File indexes.pkl ---------')

        if os.path.exists(model_dir_path + "/indexes_DeepBeam.pkl"):
            # Getting back the objects:
            with open(model_dir_path + "/indexes_DeepBeam.pkl", 'rb') as f:  # Python 3: open(..., 'rb') note that indexes
                data_loaded = pkl.load(f)

            # To Do: this should be transformed into a dictionary and you pull 'test_indexes only'. Kinda hardcoded to be fixed later

            self.test_indexes = data_loaded[-1]

            print('*********************  Generating testing data *********************')
            self.test_generator = DataGenerator(indexes=self.test_indexes,
                                                batch_size=self.args.batch_size,
                                                data_path=self.args.data_path,
                                                num_tx_beams=self.args.num_classes,
                                                num_blocks_per_frame=self.args.num_blocks_per_frame,
                                                num_samples_per_block=self.args.num_samples_per_block,
                                                how_many_blocks_per_frame=self.args.how_many_blocks_per_frame,
                                                is_2d = self.is_2d)

            self.num_of_batches = len(self.test_indexes) / self.args.batch_size
            print("Number of test batches: " + str(self.num_of_batches))
        else:
            print('I have no data to load, please give me data (e.g., indexes.pkl)')



    def parse_arguments(self):
        '''Parse input user arguments.'''

        parser = argparse.ArgumentParser(description='Testing-only pipeline',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--model_dir_path', type=str,
                            help='Path of the model.')

        parser.add_argument('--layer_num', type=int,
                            help='Layer index.')

        parser.add_argument('--id_gpu', type=int, default=2,
                            help='GPU to use.')


        parser.add_argument('--num_classes', type=int, default=24,
                            help='Number of classes in the dataset.')

        parser.add_argument('--num_samples_per_block', type=int, default=2048,
                            help='Number of blocks per frame.')

        parser.add_argument('--num_blocks_per_frame', type=int, default=15,
                            help='Total number of blocks per frame.')

        parser.add_argument('--how_many_blocks_per_frame', type=int, default=1,
                            help='Number of blocks per frame I take.')

        parser.add_argument('--data_path', type=str,
                            default='/mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5',
                            help='Path to data.')

        parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for model optimization.')

        return parser.parse_args()

if __name__ == '__main__':
    DeepBeamGetFilters()
