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

sox=`which sox`
if [ ! -x $sox ]; then
   echo "sox is not installed !! "
   echo "Ubuntu，you can run: sudo apt-get update && sudo apt-get install sox"
   echo "Centos，you can run: sudo yum install sox"
  exit 1
fi

# Tool to convert sph audio format to wav format.
  
  bash tools/install_sph2pipe.sh &>install_sph2pipe.log.txt
  
# Tool to train ngram LM, reference to https://github.com/kpu/kenlm



  bash tools/install_kenlm.sh &>install_sph2pipe.log.txt
# Tool to compute score with sclite
  sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
  bash tools/install_sclite.sh &>install_sph2pipe.log.txt

# Tool to deal text

  bash tools/install_spm.sh &>install_sph2pipe.log.txt

