#!/bin/bash

# Customize this for your CUDA install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.0/lib64/

# The GPU id depends on the number of GPUs in your system 
id_gpu=0
test_only=0

# Dataset parameters - please refer to the README.md file for a detailed explanation

# Number of frames collected for each (gain, beam) pair
num_frames_for_gain_tx_beam_pair=10000

# Number of blocks collected for each frame
num_blocks_per_frame=15

# Number of blocks actually used for training and testing
# Needs to be less than or equal the num_blocks_per_frame
how_many_blocks_per_frame=1

# Number of samples for each block
num_samples_per_block=2048

# Number of gain values
num_gains=3

# Number of beams in the dataset
num_beams=3

# TODO
is_2d_beam=0
is_2d_model=1

# Select one among low, mid, high, all to filter by SNR (gain) values
snr="all"


# Training parameters
epochs=10
batch_size=100
train_perc=0.60
valid_perc=0.00
test_perc=0.40
save_best_only=1
stop_param="acc"

# Model parameters (Fig. 4 of the DeepBeam paper https://arxiv.org/pdf/2012.14350.pdf)
kernel_size=7
num_of_kernels=64
num_of_conv_layers=7
num_of_dense_layers=2
size_of_dense_layers=128
patience=100

# DeepBeam network
# This is where you can store pkl files with the models
root=./
# These are the paths to the HDF5 files
d1=/filename1.h5
d2=/filename2.h5
d3=/filename3.h5
d4=/filename4.h5

save_path=$root
save_path+="DeepBeam_cl_$num_of_conv_layers"
save_path+="_nk_$num_of_kernels"
save_path+="_ks_$kernel_size"
save_path+="_dl_$num_of_dense_layers"
save_path+="_sd_$size_of_dense_layers"
save_path+="_bf_$how_many_blocks_per_frame"
save_path+="_srn_$snr"
save_path+="_2dbeam_$is_2d_beam"
save_path+="_2dmodel_$is_2d_model"
save_path+="_ne_$epochs"
save_path+="_bs_$batch_size"

python ./DeepBeamMixed.py \
    --batch_size $batch_size \
    --train_cnn \
    --test_only $test_only \
    --epochs $epochs \
    --save_best_only $save_best_only \
    --stop_param $stop_param \
    --snr $snr \
    --num_blocks_per_frame $num_blocks_per_frame \
    --how_many_blocks_per_frame $how_many_blocks_per_frame \
    --num_samples_per_block $num_samples_per_block \
    --num_frames_for_gain_tx_beam_pair $num_frames_for_gain_tx_beam_pair \
    --num_gains $num_gains \
    --num_beams $num_beams \
    --train_perc $train_perc \
    --valid_perc $valid_perc \
    --test_perc $test_perc \
    --datasets $d1 $d1 $d3 $d4 \
    --kernel_size $kernel_size \
    --num_of_kernels $num_of_kernels \
    --num_of_conv_layers $num_of_conv_layers \
    --num_of_dense_layers $num_of_dense_layers \
    --size_of_dense_layers $size_of_dense_layers \
    --id_gpu $id_gpu \
    --is_2d_model $is_2d_model \
    --patience $patience \
	--save_path $save_path
