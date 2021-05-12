#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/

id_gpu=0
gpu=2
#

root=/home/frestuc/projects/beam_results/saved_models/aoa/mixed/jj_0_jj_1_tm_0_tm_1/
model=DeepBeam_cl_7_nk_64_ks_7_dl_2_sd_128_bf_1_srn_all_2dbeam_0_2dmodel_1_ne_10_bs_100

indexes_root=$root
indexes_model=$model

d1=/media/michele/rx-12-aoa-tx-tm-0-rx-jj-0.h5
d2=/media/michele/rx-12-aoa-tx-tm-0-rx-jj-1.h5
d3=/media/michele/rx-12-aoa-tx-tm-0-rx-tm-1.h5
d4=/media/michele/rx-12-aoa-tx-tm-1-rx-tm-0.h5

how_many_blocks_per_frame=1
num_classes=3

# file_save_accuracy=testing_results_one_other.pkl
file_save_accuracy=testing_results.pkl

num_blocks_per_frame=15
num_samples_per_block=2048
batch_size=100
is_2d=1
plot_confusion=1
score_only=0

# diagonal-rx-12-2d-codebook-tx-tm-0-rx-tm-1.h5
# diagonal-rx-12-2d-codebook-tx-tm-0-rx-tm-1-v2.h5
# diagonal-rx-12-aoa-tx-tm-0-rx-tm-1.h5
# diagonal-rx-12-tx-tm-0-rx-tm-1.h5
# obstacle-rx-12-2d-codebook-tx-tm-0-rx-tm-1.h5
# obstacle-rx-12-aoa-tx-tm-0-rx-tm-1.h5
# obstacle-rx-12-tx-tm-0-rx-tm-1.h5
# rx-12-2d-codebook-tx-jj-0-rx-tm-1.h5
# rx-12-2d-codebook-tx-jj-1-rx-tm-1.h5
# rx-12-2d-codebook-tx-tm-1-rx-tm-0.h5
# rx-12-2d-codebook-tx-tm-0-rx-tm-1.h5
# rx-12-aoa-tx-tm-0-rx-jj-0.h5
# rx-12-aoa-tx-tm-0-rx-jj-1.h5
# rx-12-aoa-tx-tm-0-rx-tm-1.h5
# rx-12-aoa-tx-tm-1-rx-tm-0.h5
# rx-12-tx-tm-0-rx-tm-1.h5
# rx-12-tx-tm-1-rx-tm-0.h5
# rx-12-tx-jj-0-rx-tm-1.h5
# rx-12-tx-jj-1-rx-tm-1.h5


python2 ./DeepBeamTestingMixed.py \
        --model_dir_path $root$model \
        --batch_size $batch_size \
        --file_save_accuracy $file_save_accuracy \
        --datasets $d1 $d1 $d3 $d4 \
        --indexes_path $indexes_root$indexes_model \
        --num_classes $num_classes \
        --num_blocks_per_frame $num_blocks_per_frame \
        --how_many_blocks_per_frame $how_many_blocks_per_frame \
        --num_samples_per_block $num_samples_per_block \
        --id_gpu $gpu \
        --is_2d $is_2d \
        --score_only $score_only \
        --plot_confusion $plot_confusion
