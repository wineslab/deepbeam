import pickle
from Utils import plot_confusion_matrix, confusion_matrix_to_formatted_output

num_classes=3
import matplotlib.pyplot as plt
import numpy as np
import itertools

root = "/home/frestuc/projects/beam_results/saved_models/aoa/mixed/jj_0_jj_1_tm_0_tm_1/DeepBeam_cl_7_nk_64_ks_7_dl_2_sd_128_bf_1_srn_all_2dbeam_0_2dmodel_1_ne_10_bs_100/"
file = "testing_results.pkl"
dict = pickle.load(open(root+file, "rb"))

print("Accuracy is: " + str(dict["example_accuracy"]))
cm = dict["confusion_matrix"]

plot_confusion_matrix(
    cm,
    [i for i in range(num_classes)],
    root
)

confusion_matrix_to_formatted_output(
    cm,
    root + "formatted_cm.dat",
    normalize=True
)
