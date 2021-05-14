#!/bin/bash

# path
root=./

# model folder
model=DeepBeam_cl_7_nk_64_ks_7_dl_2_sd_128_bf_5_srn_all_swap_1_ne_20_bs_100

# HDF5 path
data_path=/filename.h5

layer_num=1
id_gpu=0

num_blocks_per_frame=15
num_samples_per_block=2048
how_many_blocks_per_frame=5
batch_size=100
num_classes=24

python2 ./DeepBeamGetFilters.py \
        --model_dir_path $root$model \
        --layer_num $layer_num \
        --id_gpu $id_gpu \
        --data_path $data_path \
        --batch_size $batch_size \
        --num_classes $num_classes \
        --num_blocks_per_frame $num_blocks_per_frame \
        --how_many_blocks_per_frame $how_many_blocks_per_frame \
        --num_samples_per_block $num_samples_per_block \
