#!/bin/bash

# Customize this for your CUDA install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.0/lib64/

# The GPU id depends on the number of GPUs in your system 
id_gpu=0
gpu=2

# Path with the trained model
root=./
# Model name
model=DeepBeam_cl_7_nk_64_ks_7_dl_2_sd_128_bf_1_srn_all_2dbeam_1_2dmodel_1_ne_20_bs_100
indexes_root=$root
indexes_model=$model

# Path for the HDF5 file (including file name)
data_path=./filename.h5

# Parameters that need to match the training information
how_many_blocks_per_frame=5
num_blocks_per_frame=15
num_samples_per_block=2048
batch_size=100
is_2d=1

# Output parameters
plot_confusion=0
score_only=1

# Number of classes (3 for AoA, X for X-beams codebook)
num_classes=12

# Output file
file_save_accuracy=testing_results.pkl
# file_save_accuracy=testing_results.pkl



python ./DeepBeamTesting.py \
        --data_path $data_path \
        --model_dir_path $root$model \
        --batch_size $batch_size \
        --file_save_accuracy $file_save_accuracy \
        --indexes_path $indexes_root$indexes_model \
        --num_classes $num_classes \
        --num_blocks_per_frame $num_blocks_per_frame \
        --how_many_blocks_per_frame $how_many_blocks_per_frame \
        --num_samples_per_block $num_samples_per_block \
        --id_gpu $gpu \
        --is_2d $is_2d \
        --score_only $score_only \
        --plot_confusion $plot_confusion
