import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# import tikzplotlib


def main():
    counts = [6400, 6400, 12800, 6400, 6400, 6400, 6400, 12800, 6400, 6400, 6400,
              6400, 12800, 6400, 6400, 6400, 6400, 12800, 6400, 6400, 6400, 6400, 6400, 6400]

    # read full 2D file
    beams = pd.read_csv('../data_preproc/beam_measurements_ni/rx_2D_05_20_measurement.csv',
                        skiprows=15, sep='; ', header=None)

    cut_0 = beams.loc[(beams[2] == -0.005)]
    azimuth_angles = cut_0[1]
    azimuth_angles_norm = azimuth_angles - min(azimuth_angles)
    azimuth_angles_norm /= max(azimuth_angles_norm) 

    # Python 3: open(..., 'rb') note that indexes
    with open("../results/filters_viz/classes.pkl", 'rb') as f:
        classes = pkl.load(f)

    beams = np.arange(0, 24)

    max_indeces = np.empty(24)

    for beam_id in beams:
        values = np.array(classes[beam_id].items())[:, 1]
        print(values)
        norm_values = values / counts[beam_id]
        norm_values /= max(norm_values)

        max_indeces[beam_id] = np.argmax(norm_values)

        beam_pattern = cut_0[beam_id + 3]
        beam_pattern_norm = beam_pattern - min(beam_pattern)
        beam_pattern_norm /= max(beam_pattern_norm)

        print(norm_values)

        fig, ax = plt.subplots()
        ax.plot(azimuth_angles_norm * 64, beam_pattern_norm)
        ax.stem(norm_values, use_line_collection=True)
        ax.grid()
        plt.show()

    plt.stem(max_indeces, use_line_collection=True)
    plt.show()

if __name__ == '__main__':
    main()
