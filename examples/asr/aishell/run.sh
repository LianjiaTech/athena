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
use_wfst=false  # decode options
offline=false	# storage features offline options, set to true if needed, default false 

# source some path
. ./tools/env.sh

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

# data options

dataset_dir=examples/asr/aishell/data/data_aishell

if [ ! -d "$dataset_dir" ]; then
  echo "\no such directory $dataset_dir"
  echo "downloading data from www.openslr.org..... "
  bash examples/asr/aishell/local/aishell_download_and_untar.sh examples/asr/aishell/data \
                                                           www.openslr.org/resources/33 \
														   data_aishell
fi

# prepare aishell data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    
    echo "Preparing data and Creating csv"
    bash examples/asr/aishell/local/aishell_data_prep.sh $dataset_dir examples/asr/aishell/data
fi

# calculate cmvn

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "extract cmvn"
    cat examples/asr/aishell/data/train.csv > examples/asr/aishell/data/all.csv
    tail -n +2 examples/asr/aishell/data/dev.csv >> examples/asr/aishell/data/all.csv
    tail -n +2 examples/asr/aishell/data/test.csv >> examples/asr/aishell/data/all.csv
    CUDA_VISIBLE_DEVICES='' python athena/cmvn_main.py \
        examples/asr/aishell/configs/cmvn.json examples/asr/aishell/data/all.csv || exit 1
fi

# storage features offline

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && $offline; then
    echo "storage features offline"
    python athena/tools/storage_features_offline.py examples/asr/aishell/configs/storage_features_offline.json
fi

## pretrain stage 
if $pretrain;then
   if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Pretraining with mpc"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/aishell/configs/mpc.json || exit 1
   fi
fi

# Multi-task training stage 

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Multi-task training"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/aishell/configs/mtl_transformer_sp.json || exit 1
fi

# prepare language model 
if $rnnlm;then
   # training rnnlm
   if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        echo "training rnnlm"
		bash examples/asr/aishell/local/aishell_train_rnnlm.sh
   fi
else
   # training ngram lm
   if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        echo "training ngram lm"
		bash examples/asr/aishell/local/aishell_train_lm.sh
   fi
fi

# decode

if $use_wfst;then
   # wfst decoding
   if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Running decode with WFST..."
      echo "For now we will simply download LG.fst and words.txt from athena-decoder project"
      echo "Feel free to checkout graph creation manual at https://github.com/athena-team/athena-decoder#build-graph"
      git clone https://github.com/athena-team/athena-decoder
      cp athena-decoder/examples/aishell/graph/LG.fst examples/asr/aishell/data/
      cp athena-decoder/examples/aishell/graph/words.txt examples/asr/aishell/data/
      python athena/inference.py \
          examples/asr/aishell/configs/mtl_transformer_sp_wfst.json || exit 1
   fi
else
   # beam search decoding, rnnlm default
   if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Running decode with beam search..."
      python athena/inference.py \
        examples/asr/aishell/configs/mtl_transformer_sp.json || exit 1
   fi
fi

# score-computing stage

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
   echo "computing score with sclite ..."
   bash examples/asr/aishell/local/run_score.sh inference.log score_aishell examples/asr/aishell/data/vocab
fi


echo "$0: Finished AISHELL training examples"
