#!/bin/bash

# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

# train options
stage=0
stop_stage=100

# horovod options

horovod_cmd="horovodrun -np 1 -H localhost:1"
horovod_prefix="horovod_"


# seletion options 
pretrain=false    # pretrain options, we provide Masked Predictive Coding (MPC) pretraining, default false 
rnnlm=true  # rnn language model training is provided ,set to false,if use ngram language model
offline=false	# storage features offline options, set to true if needed, default false
use_wfst=false  # decode options
# data options
dataset_dir=data/hkust_resource

if [ ! -d "$dataset_dir" ]; then
  echo "\no such directory $dataset_dir"
  echo "you shuld find the HKUST Resource..... "
fi

# source some path
. ./tools/env.sh

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

# prepare hkust data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    
    echo "Preparing data"
    bash examples/asr/hkust/local/hkust_data_prep.sh $dataset_dir examples/asr/hkust/data
fi

# calculate cmvn

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "extract cmvn"
    cat examples/asr/hkust/data/train.csv > examples/asr/hkust/data/all.csv
    tail -n +2 examples/asr/hkust/data/dev.csv >> examples/asr/hkust/data/all.csv
    CUDA_VISIBLE_DEVICES='' python athena/cmvn_main.py \
        examples/asr/hkust/configs/cmvn.json examples/asr/hkust/data/all.csv || exit 1
fi

# storage features offline

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && $offline; then
    echo "storage features offline"
    python athena/tools/storage_features_offline.py examples/asr/hkust/configs/storage_features_offline.json
fi

## pretrain stage mpc 
if $pretrain;then
   if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Pretraining with mpc"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/hkust/configs/mpc.json || exit 1
   fi
fi

# Multi-task training stage 

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Multi-task training"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/hkust/configs/mtl_transformer_sp.json || exit 1
fi

# prepare language model 
if $rnnlm;then
   # training rnnlm
   if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        echo "training rnnlm"
		bash examples/asr/hkust/local/aishell_train_rnnlm.sh
   fi
else
   # training ngram lm
   if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        echo "training ngram lm"
		bash examples/asr/hkust/local/hkust_train_lm.sh
   fi
fi

# decode

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Decoding"
    python athena/inference.py \
        examples/asr/hkust/configs/mtl_transformer_sp.json || exit 1
fi

# score-computing stage

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
   echo "computing score with sclite ..."
   bash examples/asr/hkust/local/run_score.sh inference.log score_hkust examples/asr/hkust/data/vocab
fi

echo "$0: Finished HKUST training examples"
