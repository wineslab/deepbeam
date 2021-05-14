import pickle
from Utils import plot_confusion_matrix, confusion_matrix_to_formatted_output

import matplotlib.pyplot as plt
import numpy as np
import itertools

num_classes = 3

root = "path_to_output_file"
file = "testing_results.pkl"
dict = pickle.load(open(root + file, "rb"))

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
