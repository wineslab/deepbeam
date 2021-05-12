from DataGenerator import DataGenerator
import numpy as np


def main():
    gains = [40, 50, 60]
    tx_beams = np.arange(0, 24)
    num_frames_for_gain_tx_beam_pair = 10000

    # Order is gain *

    indexes = np.arange(
        0,
        num_frames_for_gain_tx_beam_pair * len(tx_beams) * len(gains)
    )
    batch_size = 32
    data_path = '/media/michele/rx-12.h5'
    num_blocks_per_frame = 15
    how_many_blocks_per_frame = 15
    num_samples_per_block = 2048

    dg = DataGenerator(
        indexes,
        batch_size,
        data_path,
        num_blocks_per_frame,
        num_samples_per_block,
        how_many_blocks_per_frame,
        shuffle=True,
        is_2d=False
    )

    batch_gain_tx_beam = num_frames_for_gain_tx_beam_pair / batch_size


    # for [i_g, val_g] in enumerate(gains):
    #     print("Gain: " + str(val_g))
    #     for [i_t, val_t] in enumerate(tx_beams):
    #         print("Beam idx: " + str(val_t))
    #         batch_index = (i_g * len(tx_beams) * batch_gain_tx_beam) + i_t * batch_gain_tx_beam
    #         print("Batch idx: " + str(batch_index))
    #         [batch, batch_y] = dg.__getitem__(batch_index)
    #         print("tx_beam %d y % s" % (val_t, batch_y[0]))
    #         # print(batch_y[0])


    for i in range(dg.__len__()):
        print("Batch idx: " + str(i))
        [batch, batch_y] = dg.__getitem__(i)
        print("tx_beam %s %s y %s %s" % (batch[0][0], batch[-1][0], batch_y[0], batch_y[-1]))




if __name__ == '__main__':
    main()
