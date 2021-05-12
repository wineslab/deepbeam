from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
import numpy as np
import itertools


def build_model_1d(
    num_of_conv_layers,
    num_of_kernels,
    kernel_size,
    num_of_dense_layers,
    size_of_dense_layers,
    num_samples_per_block,
    num_beams
):
    '''Build model architecture.'''
    print('*************** Building Baseline Model ***************')
    inputs = Input(shape=(num_samples_per_block, 2), name='Input')

    x = Conv1D(num_of_kernels, kernel_size=kernel_size, padding='same', name='Conv_0')(inputs)
    x = MaxPooling1D(pool_size=2, data_format='channels_last', name='MaxPool_0')(x)

    for i in range(1, num_of_conv_layers):
        x = Conv1D(num_of_kernels, kernel_size=kernel_size, padding='same', name='Conv_'+str(i))(x)
        x = MaxPooling1D(pool_size=2, data_format='channels_last', name='MaxPool'+str(i))(x)

    x = Flatten(name='Flatten')(x)

    for i in range(num_of_dense_layers):
        x = Dense(size_of_dense_layers, activation='linear', name='Dense_'+str(i))(x)

    x = Dense(num_beams, activation='softmax', name='Softmax')(x)

    return Model(inputs=inputs, outputs=x)


def build_model(
    num_of_conv_layers,
    num_of_kernels,
    kernel_size,
    num_of_dense_layers,
    size_of_dense_layers,
    how_many_blocks_per_frame,
    num_samples_per_block,
    num_beams
):
    '''Build model architecture.'''
    print('*************** Building Baseline Model ***************')
    inputs = Input(shape=(how_many_blocks_per_frame, num_samples_per_block, 2), name='Input')

    x = Conv2D(num_of_kernels, kernel_size=(1, kernel_size), padding='same', name='Conv_0')(inputs)
    x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_0')(x)

    for i in range(1, num_of_conv_layers):
        x = Conv2D(num_of_kernels, kernel_size=(1, kernel_size), padding='same', name='Conv_'+str(i))(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool'+str(i))(x)


    x = Flatten(name='Flatten')(x)

    for i in range(num_of_dense_layers):
        x = Dense(size_of_dense_layers, activation='linear', name='Dense_'+str(i))(x)

    x = Dense(num_beams, activation='softmax', name='Softmax')(x)

    return Model(inputs=inputs, outputs=x)



def plot_confusion_matrix(cm,
                          classes,
                          save_path,
                          normalize=True,
                          show_image=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] < 0.01:
            cm[i, j] = 0
        plt.text(j, i, format(cm[i, j], fmt if cm[i, j] else '.0f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if show_image:
        plt.show()
    if not show_image:
        plt.savefig(save_path)


def confusion_matrix_to_formatted_output(cm,
                          save_path,
                          normalize=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    with open(save_path, "w") as file:
        file.write("\\begin{tikzpicture}\n")
        file.write("\\begin{axis}[enlargelimits=false,colorbar,colormap/Purples]\n")
        file.write("\\addplot [matrix plot,point meta=explicit]\n")
        file.write(" coordinates {\n")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                file.write(
                    "(" + str(i) + "," + str(j) + ") [" + str(cm[i, j]) + "] ")
            file.write("\n\n")
        file.write("};\n")
        file.write("\\end{axis}\n")
        file.write("\\end{tikzpicture}")
