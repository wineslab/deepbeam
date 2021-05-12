#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/

id_gpu=2
test_only=0

# Dataset parameters

num_blocks_per_frame=15
num_frames_for_gain_tx_beam_pair=10000
num_samples_per_block=2048
num_gains=3
num_beams=3
num_angles=3
snr="all"

# Training parameters

epochs=10
batch_size=100
train_perc=0.60
valid_perc=0.00
test_perc=0.40
save_best_only=1
stop_param="acc"

# Model parameters

how_many_blocks_per_frame=1
kernel_size=7
num_of_kernels=64
num_of_conv_layers=7
num_of_dense_layers=2
size_of_dense_layers=128
is_1d=0
patience=100

root=/home/frestuc/projects/beam_results/saved_models/obstacle/aoa/
data_path=/media/michele/obstacle-rx-12-aoa-tx-tm-0-rx-tm-1.h5

save_path=$root
save_path+="DeepBeam_cl_$num_of_conv_layers"
save_path+="_nk_$num_of_kernels"
save_path+="_ks_$kernel_size"
save_path+="_dl_$num_of_dense_layers"
save_path+="_sd_$size_of_dense_layers"
save_path+="_is1d_$is_1d"
save_path+="_bf_$how_many_blocks_per_frame"
save_path+="_srn_$snr"
save_path+="_ne_$epochs"
save_path+="_bs_$batch_size"

python2 ./DeepBeamSideLobe.py \
    --data_path $data_path \
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
    --is_1d $is_1d \
    --num_gains $num_gains \
    --num_beams $num_beams \
    --num_angles $num_angles\
    --train_perc $train_perc \
    --valid_perc $valid_perc \
    --test_perc $test_perc \
    --kernel_size $kernel_size \
    --num_of_kernels $num_of_kernels \
    --num_of_conv_layers $num_of_conv_layers \
    --num_of_dense_layers $num_of_dense_layers \
    --size_of_dense_layers $size_of_dense_layers \
    --id_gpu $id_gpu \
    --patience $patience \
	--save_path $save_path

# 720000 / 32 = 22500 batches
