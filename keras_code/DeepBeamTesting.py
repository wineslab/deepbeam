import argparse
# import h5py
# import json
# import keras
# from keras.models import load_model
import numpy as np
import pickle as pkl
# import time
import os

# from keras.models import Model
from keras.optimizers import Adam

# from keras.utils.io_utils import HDF5Matrix
from DataGenerator import DataGenerator
from DataGeneratorAoa import DataGeneratorAoa
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json

from Utils import plot_confusion_matrix

class DeepBeamTesting(object):

    def __init__(self):
        '''Initialize class variables.'''
        self.args = self.parse_arguments()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.id_gpu)

        self.is_2d = self.args.is_2d
        self.num_classes = self.args.num_classes

        self.load_from_json(self.args.model_dir_path)
        self.load_testing_data(self.args.model_dir_path)
        self.test_model()

    def load_from_json(self, folder):
        # load json and create model
        json_file = open(folder + "/model_arch.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(folder + "/DeepBeam_model.hdf5")
        print("Loaded model from disk")

    def load_testing_data(self, model_dir_path):
        '''Load data from path into framework.'''

        print('--------- Loading from File indexes.pkl ---------')

        if os.path.exists(self.args.indexes_path + "/indexes_DeepBeam.pkl"):
            # Getting back the objects:
            # Python 3: open(..., 'rb') note that indexes
            with open(
                    self.args.indexes_path + "/indexes_DeepBeam.pkl", 'rb')\
                    as f:
                data_loaded = pkl.load(f)

            self.test_indexes = data_loaded[-1]

            print('*******  Generating testing data *******')

            if (self.args.num_classes > 3):
                self.test_generator = DataGenerator(
                    indexes=self.test_indexes,
                    batch_size=self.args.batch_size,
                    data_path=self.args.data_path,
                    num_tx_beams=self.args.num_classes,
                    num_blocks_per_frame=self.args.num_blocks_per_frame,
                    num_samples_per_block=self.args.num_samples_per_block,
                    how_many_blocks_per_frame=self.args.how_many_blocks_per_frame,
                    is_2d=self.is_2d)
            else:  # if only 3 classes are used, it is AoA detection and use DataGeneratorAoa
                self.test_generator = DataGeneratorAoa(
                    indexes=self.test_indexes,
                    batch_size=self.args.batch_size,
                    data_path=self.args.data_path,
                    num_blocks_per_frame=self.args.num_blocks_per_frame,
                    num_samples_per_block=self.args.num_samples_per_block,
                    how_many_blocks_per_frame=self.args.how_many_blocks_per_frame,
                    is_2d=self.is_2d)

            self.num_of_batches = len(self.test_indexes) / self.args.batch_size
            print("Number of test batches: " + str(self.num_of_batches))
        else:
            print('I have no data to load, please give me data (e.g., indexes.pkl)')

    def get_predicted_label(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        predicted_label = unique[np.argmax(counts)]
        return predicted_label

    def test_model(self):
        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

        if self.args.score_only:
            score = self.model.evaluate_generator(
                self.test_generator,
                verbose=1,
                use_multiprocessing=False
            )
            print("score is: " + str(score))
            return

        score_predict = self.model.predict_generator(
            self.test_generator,
            verbose=1,
            use_multiprocessing=False
        )

        label_predict = np.argmax(score_predict, 1)
        label_true = np.zeros(label_predict.shape)

        idx = 0
        print("label predict shape: " + str(label_predict.shape))
        for i in range(self.num_of_batches):
            x, batch_y = self.test_generator.__getitem__(i)
            for y in batch_y:
                label_true[idx] = np.argmax(y)
                idx += 1

        con_matrix = confusion_matrix(label_true, label_predict)
        con_matrix_perc = con_matrix / con_matrix.astype(np.float).sum(axis=1)
        example_accuracy = np.mean(np.diag(con_matrix_perc))

        my_dict = {'example_accuracy': example_accuracy,
                   'confusion_matrix': con_matrix_perc}

        print('Example Accuracy: ', example_accuracy)

        # Saving the objects:

        # Python 3: open(..., 'wb')
        with open(self.args.model_dir_path + "/" + self.args.file_save_accuracy, 'wb') as f:
            pkl.dump(my_dict, f)

        num_classes = con_matrix_perc[0].shape[0]

        if self.args.plot_confusion:
            plot_confusion_matrix(
                con_matrix,
                [i for i in range(num_classes)],
                self.args.model_dir_path + "/conf_matrix.png"
            )
        print("Done!")

    def parse_arguments(self):
        '''Parse input user arguments.'''

        parser = argparse.ArgumentParser(description='Testing-only pipeline',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--id_gpu', type=int, default=2,
                            help='GPU to use.')

        parser.add_argument('--is_2d', type=int, default=1,
                            help='Specify if model is 2D or not.')

        parser.add_argument('--model_dir_path', type=str,
                            help='Path of the model.')

        parser.add_argument('--indexes_path', type=str,
                            help='Folder where to get the indexes.')

        parser.add_argument('--file_save_accuracy', type=str,
                            help='Path to pickle file for accuracy.')

        parser.add_argument('--num_classes', type=int, default=24,
                            help='Number of classes in the dataset.')

        parser.add_argument('--num_samples_per_block', type=int, default=2048,
                            help='Number of blocks per frame.')

        parser.add_argument('--num_blocks_per_frame', type=int, default=15,
                            help='Total number of blocks per frame.')

        parser.add_argument('--how_many_blocks_per_frame', type=int, default=1,
                            help='Number of blocks per frame I take.')

        parser.add_argument('--plot_confusion', type=int,
                            default=0,
                            help='Plot confusion matrix')

        parser.add_argument('--score_only', type=int,
                            default=1,
                            help='Compute only score.')

        parser.add_argument('--data_path', type=str,
                            default='./',
                            help='Path to data.')

        parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for model optimization.')

        return parser.parse_args()


if __name__ == '__main__':
    DeepBeamTesting()
