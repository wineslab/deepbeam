from DataGeneratorAoa import DataGeneratorAoa
import numpy as np


def main():
    tx_beams = [4, 12, 20]
    gains = [40, 50, 60]
    angles = [-45, 0, 45]
    num_frames_for_gain_tx_beam_pair = 10000

    indexes = np.arange(
        0,
        num_frames_for_gain_tx_beam_pair * len(tx_beams) * len(gains) * len(angles)
    )
    batch_size = 32
    data_path = './'
    num_blocks_per_frame = 15
    how_many_blocks_per_frame = 15
    num_samples_per_block = 2048

    dg = DataGeneratorAoa(
        indexes,
        batch_size,
        data_path,
        num_blocks_per_frame,
        num_samples_per_block,
        how_many_blocks_per_frame,
        shuffle=True,
        is_2d=False
    )

    batch_gain_num = num_frames_for_gain_tx_beam_pair / batch_size

    # for [i_g, val_g] in enumerate(gains):
    #     for [i_t, val_t] in enumerate(tx_beams):
    #         for [i_a, val_a] in enumerate(angles):
    #             batch_index = (i_g * len(tx_beams) * len(angles) * batch_gain_num) +\
    #                 (i_t * len(angles) * batch_gain_num) +\
    #                 i_a * batch_gain_num
    #             [batch, batch_y] = dg.__getitem__(batch_index)
    #             # print(batch[0])
    #             print("angle %d y % s" % (val_a, batch_y[-1]))
    #             # print(batch_y[0])

    for i in range(dg.__len__()):
        print("Batch idx: " + str(i))
        [batch, batch_y] = dg.__getitem__(i)
        print("X %s %s y %s %s" % (batch[0][0], batch[-1][0], batch_y[0], batch_y[-1]))



if __name__ == '__main__':
    main()
