# Athena

*Athena* is an open-source implementation of end-to-end speech processing engine. Our vision is to empower both industrial application and academic research on end-to-end models for speech processing. To make speech processing available to everyone, we're also releasing example implementation and recipe on some opensource dataset for various tasks (Automatic Speech Recognition, Speech Synthesis, Voice Conversion, Speaker Recognition, etc).

All of our models are implemented in Tensorflow>=2.0.1. For ease of use, we provide Kaldi-free pythonic feature extractor with [Athena_transform](https://github.com/athena-team/athena-transform).

## 1) Table of Contents

- [Athena](#athena)
  - [1) Table of Contents](#1-table-of-contents)
  - [2) Key Features](#2-key-features)
  - [3) Installation](#3-installation)
    - [3.1) Clone *athena* package](#31-clone-athena-package)
    - [3.2) Check system level installations](#32-check-system-level-installations)
    - [3.3) Creating a virtual environment [Optional]](#33-creating-a-virtual-environment-optional)
    - [3.4) Install *tensorflow* backend](#34-install-tensorflow-backend)
    - [3.5) Install *horovod* for multiple-device training [Optional]](#35-install-horovod-for-multiple-device-training-optional)
    - [3.6) Install *sph2pipe*, *spm*, *kenlm*, *sclite* for ASR Tasks [Optional]](#36-install-sph2pipe-spm-kenlm-sclite-for-asr-tasks-optional)
    - [3.7) Install *pydecoder* for WFST decoding [Optional]](#37-install-pydecoder-for-wfst-decoding-optional)
    - [3.8) Install *athena* package](#38-install-athena-package)
    - [3.9) Test your installation](#39-test-your-installation)
    - [Notes](#notes)
 - [4) Training](#4-training)
    - [4.1) Prepare the data](#41-prepare-the-data)
    - [4.2) Setting the Configuration File](#42-setting-the-configuration-file)
    - [4.3) Data normalization](#43-data-normalization)
    - [4.4) Storage Features Offline](#44-storage-features-offline)
    - [4.5) Train a Model](#45-train-a-model)
    - [4.6) Evaluate a model](#46-evaluate-a-model)
    - [4.7) Scoring](#47-scoring)
  - [5) Decoding with WFST](#5-decoding-with-wfst)
    - [5.1) WFST graph creation](#51-wfst-graph-creation)
    - [5.2) WFST decoding](#52-wfst-decoding)
  - [6) Deployment](#6-deployment)
  - [7) Self-supervised speech representation learning](#7-self-supervised-speech-representation-learning)
    - [7.1) MPC](#71-mpc)
    - [7.2) Speech SimCLR](#72-speech-simclr)
  - [8) Results](#8-results)
    - [8.1) ASR](#81-asr)
  - [9) Directory Structure](#9-directory-structure)

## 2) Key Features

- Hybrid Attention/CTC based end-to-end ASR
- Speech-Transformer
- Unsupervised pre-training
- Multi-GPU training on one machine or across multiple machines with Horovod
- WFST creation and WFST-based decoding
- Deployment with Tensorflow C++

## 3) Installation
We provide the installation steps of tensorflow 2.3.1. The corresponding linux system environment is : cuda:10.1, ubuntu18.04. 
If your server installed docker, you can pull docker image : docker pull nvidia/cuda:10.1-devel-ubuntu18.04, and installing the python requirements: apt update && apt install python3 && apt install python3-venv && apt install python3-pip.
We also provide a script include all installation steps:

```bash
# clone athena package,and run one step installation
git clone https://github.com/athena-team/athena.git
cd athena
bash one_installation.sh
```
If you want to use one_installation.sh, you can ignore the following steps!!!

### 3.1) Clone *athena* package

```bash
# In this step,you must install git( sudo apt-get update && sudo apt-get install git)
git clone https://github.com/athena-team/athena.git
```

### 3.2) Check system level installations
To check the base prerequisites for Athena

```bash
cd athena
bash check_source.sh
```

### 3.3) Creating a virtual environment [Optional]

This project has only been tested on Python 3. We highly recommend creating a virtual environment and installing the python requirements there. 

```bash
# Setting up virtual environment
apt-get install python3-venv
python3 -m venv venv_athena
source venv_athena/bin/activate
```

### 3.4) Install *tensorflow* backend

For more information, you can checkout the [tensorflow website](https://github.com/tensorflow/tensorflow). 


```bash
# we highly recommend firstly update pip, if you find tensorflow download very slow, you can add "-i https://pypi.tuna.tsinghua.edu.cn/simple", eg: pip install tensorflow==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip
pip install tensorflow==2.3.1
```

### 3.5) Install *horovod* for multiple-device training [Optional]

For multiple GPU/CPU training
You have to install the *horovod*, you can find out more information from the [horovod website](https://github.com/horovod/horovod#install). 
We provide a installation steps as reference,you can run the script in `tools/`.

```bash
cd athena
bash tools/install_horovod.sh
```

### 3.6) Install *sph2pipe*, *spm*, *kenlm*, *sclite* for ASR Tasks [Optional]

These packages are usually required for ASR tasks, we assume they have been installed when running the recipe for ASR tasks. You can find installation scripts of them in `tools/`, and a general installation script as reference:

```bash
cd athena
bash tools/install_tools_for_asr.sh
```

### 3.7) Install *pydecoder* for WFST decoding [Optional]

For WFST decoding
You have to install *pydecoder*, installation guide for *pydecoder* can be found [athena-decoder website](https://github.com/athena-team/athena-decoder#installation)

### 3.8) Install *athena* package

```bash
cd athena
pip install -r requirements.txt
python setup.py bdist_wheel sdist
python -m pip install --ignore-installed dist/athena-0.1.0*.whl
```

- Once athena is successfully installed, you should do `source tools/env.sh` firstly before doing other things.

### 3.9) Test your installation

- On a single cpu/gpu

```bash
source tools/env.sh
python examples/translate/spa-eng-example/prepare_data.py examples/translate/spa-eng-example/data/train.csv
python athena/main.py examples/translate/spa-eng-example/transformer.json
```

- On multiple cpu/gpu in one machine (you should make sure your hovorod is successfully installed)

```bash
source tools/env.sh
python examples/translate/spa-eng-example/prepare_data.py examples/translate/spa-eng-example/data/train.csv
horovodrun -np 4 -H localhost:4 python athena/horovod_main.py examples/translate/spa-eng-example/transformer.json
```

### Notes

- If you see errors such as `ERROR: Cannot uninstall 'wrapt'` while installing TensorFlow, try updating it using command `conda update wrapt`. Same for similar dependencies such as `entrypoints`, `llvmlite` and so on.
- You may want to make sure you have `g++` version 7 or above to make sure you can successfully install TensorFlow.

## 4) Training
We will use ASR task TIMIT as an example to walk you through the whole training process. The recipe for this tutorial can be found at ```examples/asr/timit/run_101.sh```.

### 4.1) Prepare the data
The data for TIMIT can be found [here](https://www.kaggle.com/mfekadu/darpa-timit-acousticphonetic-continuous-speech) or [here](https://ndownloader.figshare.com/files/10256148). First, we need to download the data and place it at ```examples/asr/timit/data/TIMIT```.
Then we will run the following scripts, which will do some data precessing and generate data csv for train, dev and test set of TIMIT.
```
mkdir -p examples/asr/timit/data
python examples/asr/timit/local/prepare_data.py examples/asr/timit/data/TIMIT examples/asr/timit/data
```
Below is an example csv we generated, it contains the absolute path of input audio, its length, its transcript and its speaker
```csv
wav_filename	wav_length_ms	transcript	speaker
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SI1456.WAV	3065	sil dh iy z eh er er vcl g ae sh vcl b ah vcl b ax sh epi m ey cl k hh ay l ix f ah ng cl sh epi en el th er m el vcl b eh r ix er z sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SX286.WAV	3283	sil ih n eh v r ih m ey vcl jh er cl k l ow v er l iy f cl t r ae f ix cl k s ah m cl t ay m z vcl g eh cl s vcl b ae cl t ah cl p sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SX196.WAV	1740	sil hh aw vcl d uw ao r sh cl ch er zh epi m ey cl p er l vcl d z sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SX106.WAV	2214	sil eh hh y uw vcl jh cl t ae cl p ix sh cl t r ix hh ah ng ix n er hh ah l w ey sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SX16.WAV	1926	sil ey r ow l el v w ay er l ey n ih er dh ax w ao l sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SI2086.WAV	2745	sil ae vcl b s el uw sh en f ao r hh ix z l ay hh sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SX376.WAV	2464	sil w ih m ix n m ey n eh v er vcl b ix cl k ah ng cl k ax m cl p l iy cl l iy cl k w el cl t ax m eh n sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SI826.WAV	3596	sil k ao sh en cl k en cl t ih n y uw s ix vcl m ih n ax sh cl t r ey sh en ix z epi n aa vcl r eh cl k m eh n d ix f ax l ae cl t ey dx ng cl k aw z sil	MCLM0
```

### 4.2) Setting the Configuration File

All of our training/ inference configurations are written in config.json. Below is an example configuration file with comments to help you understand.

```jsonc
{
  "batch_size":16,
  "num_epochs":20,
  "sorta_epoch":1,  # keep batches sorted for sorta_epoch, this helps with the convergence of models
  "ckpt":"examples/asr/timit/ckpts/mtl_transformer_ctc_sp/",
  "summary_dir":"examples/asr/timit/ckpts/mtl_transformer_ctc_sp/event",

  "solver_gpu":[0],
  "solver_config":{
    "clip_norm":100,  # clip gradients into a norm of 100
    "log_interval":10,  # print logs for log_interval steps
    "enable_tf_function":true  # enable tf_function to make training faster
  },

  "model":"mtl_transformer_ctc",  # the type of model this training uses, it's a multi-task transformer based model
  "num_classes": null,
  "pretrained_model": null,
  "model_config":{
    "model":"speech_transformer",
    "model_config":{
      "return_encoder_output":true,  # whether to return encoder only or encoder + decoder
      "num_filters":256,  # dimension of cnn filter
      "d_model":256,  # dimension of transformer
      "num_heads":8,  # heads of transformer
      "num_encoder_layers":9,
      "num_decoder_layers":3,
      "dff":1024,  # dimension of feed forward layer
      "rate":0.2,  # dropout rate for transformer
      "label_smoothing_rate":0.0,  # label smoothing rate for output logits
      "schedual_sampling_rate":1.0  # scheduled sampling rate for decoder
    },
    "mtl_weight":0.5
  },

  "inference_config":{
    "decoder_type":"beam_search_decoder",  # use beam search instead of argmax
    "beam_size":10,
    "ctc_weight":0.0,  # weight for ctc joint decoding
    "model_avg_num":10  # averaging checkpoints gives better results than using single checkpoint with best loss/ metrics
  },

  "optimizer":"warmup_adam",
  "optimizer_config":{  # configs for warmup optimizer
    "d_model":256,
    "warmup_steps":4000,
    "k":1
  },


  "dataset_builder": "speech_recognition_dataset",
  "num_data_threads": 1,
  "trainset_config":{
    "data_csv": "examples/asr/timit/data/train.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},  # config for feature extraction
    "cmvn_file":"examples/asr/timit/data/cmvn",  # mean and variance of FBank
    "text_config": {"type":"eng_vocab", "model":"examples/asr/timit/data/vocab"},  # vocab list
    "speed_permutation": [0.9, 1.0, 1.1],  # use speed perturbation to increase data diversitty
    "input_length_range":[10, 8000]  # range of audio input length
  },
  "devset_config":{
    "data_csv": "examples/asr/timit/data/dev.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/timit/data/cmvn",
    "text_config": {"type":"eng_vocab", "model":"examples/asr/timit/data/vocab"},
    "input_length_range":[10, 8000]
  },
  "testset_config":{
    "data_csv": "examples/asr/timit/data/test.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/timit/data/cmvn",
    "text_config": {"type":"eng_vocab", "model":"examples/asr/timit/data/vocab"}
  }
}
```

To get state-of-the-art models, we usually need to train for more epochs and use ctc joint decoding with language model. These are omitted for to make this tutorial easier to understand.

### 4.3) Data normalization
Data normalization is important for the convergence of neural network models. With the generated csv file, we will compute the cmvn file like this
```
python athena/cmvn_main.py examples/asr/$dataset_name/configs/mpc.json examples/asr/$dataset_name/data/all.csv
```
The generated cmvn files will be found at ```examples/asr/timit/data/cmvn```.

### 4.4) Storage Features Offline

This step is optional. `athena/tools/storage_features_offline.py` will be a good choice to store the features of training data offline in advance if you want to save the time of data processing. In subsequent training, kaldiio can be used to read them directly. The specific operation is:
```
python athena/tools/storage_features_offline.py examples/asr/aishell/configs/storage_features_offline.json
```
Below is an example json configuration file to help you understand.

```jsonc
{
  "dataset_builder": "speech_recognition_dataset_kaldiio",
  "num_data_threads": 1,
  "trainset_config":{
    "data_scps_dir": "examples/asr/aishell/data/train",
    "data_csv": "examples/asr/aishell/data/train.csv",
    "audio_config": {"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/aishell/data/cmvn",
    "text_config": {"type":"vocab", "model":"examples/asr/aishell/data/vocab"},
    "input_length_range":[10, 8000],
    "speed_permutation": [0.9, 1.0, 1.1],
    "spectral_augmentation":{"warp_for_time": false, "num_t_mask": 2, "num_f_mask": 2, "max_t": 50, "max_f": 10, "max_w": 80},
    "apply_cmvn": true,
    "global_cmvn": true,
    "offline": true
  },  
  "devset_config":{
    "data_scps_dir": "examples/asr/aishell/data/dev",
    "data_csv": "examples/asr/aishell/data/dev.csv",
    "audio_config": {"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/aishell/data/cmvn",
    "text_config": {"type":"vocab", "model":"examples/asr/aishell/data/vocab"},
    "input_length_range":[10, 8000],
    "apply_cmvn": true,
    "global_cmvn": true,
    "offline": true
  },  
  "testset_config":{
    "data_scps_dir": "examples/asr/aishell/data/test",
    "data_csv": "examples/asr/aishell/data/test.csv",
    "audio_config": {"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/aishell/data/cmvn",
    "text_config": {"type":"vocab", "model":"examples/asr/aishell/data/vocab"},
    "apply_cmvn": true,
    "global_cmvn": true,
    "offline": true
  }
}
```

It should be noted that ```"offline": true```. ```"apply_cmvn"``` indicates whether CMVN processing is required, and it is set to true by default. ```"global_cmvn"``` indicates whether CMVN processing is global, and it is set to true by default.

### 4.5) Train a Model

With all the above preparation done, training becomes straight-forward. `athena/main.py` is the entry point of the training module. Just run:
```
$ python athena/main.py examples/asr/timit/configs/mtl_transformer_sp_101.json
```

Please install Horovod and MPI at first, if you want to train model using multi-gpu. See the [Horovod page](https://github.com/horovod/horovod) for more instructions.

To run on a machine with 4 GPUs with Athena:
```
$ horovodrun -np 4 -H localhost:4 python athena/horovod_main.py examples/asr/timit/configs/mtl_transformer_sp_101.json
```

To run on 4 machines with 4 GPUs each with Athena:
```
$ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python athena/horovod_main.py examples/asr/timit/configs/mtl_transformer_sp_101.json
```

### 4.6) Evaluate a model
All of our inference related scripts are merged into inference.py. `athena/inference.py` is the entry point of inference. Just run:
```
python athena/inference.py examples/asr/timit/configs/mtl_transformer_sp_101.json
```
A file named `inference.log` will be generated, which contains the log of decoding. `inference.log` is very important to get correct scoring results, and it will be overwrited if you run `athena/inference.py` multiple times.

### 4.7) Scoring
For scoring, you will need to install [sclite](https://github.com/usnistgov/SCTK) first. The results of scoring can be found in `score/score_map/inference.log.result.map.sys`. The last few lines will look like this
```
|================================================================|
| Sum/Avg|  192   7215 | 84.4   11.4    4.3    3.2   18.8   99.5 |
|================================================================|
|  Mean  |  1.0   37.6 | 84.7   11.4    3.9    3.3   18.6   99.5 |
|  S.D.  |  0.0   11.7 |  7.7    6.3    4.2    3.6    9.0    7.2 |
| Median |  1.0   36.0 | 85.0   10.8    2.9    2.8   17.5  100.0 |
|----------------------------------------------------------------|
```
The line with ```Sum/Avg``` is usually what you should be looking for if you just want an overall PER result. In this case, 11.4 is the substitution error, 4.3 is the deletion error, 3.2 is the insertion error and 18.8 is the total PER.

## 7) Self-supervised speech representation learning

### 7.1) MPC
Masked Predictive Coding (MPC) uses masked reconstruction objective to perform predictive coding on transformer based models. It achieved significant improvements on various speech recognition datasets. For more information, please refer to following paper(s).

[Improving Transformer-based Speech Recognition Using Unsupervised Pre-training](https://arxiv.org/abs/1910.09932.pdf)

[A Further Study of Unsupervised Pre-training for Transformer Based Speech Recognition](https://arxiv.org/pdf/2005.09862.pdf)

MPC models can be trained by running ```python athena/main.py examples/asr/*/configs/mpc.json```. To use pretrained MPC model in ASR training, simply set the "pretrained_model" section in ASR json config to the checkpoint dir of MPC model and proceed training.

### 7.2) Speech SimCLR
Speech SimCLR is a new self-supervised objective for speech representation learning. During training, Speech SimCLR applies augmentation on raw speech and its spectrogram. Its objective is the combination of contrastive loss that maximizes agreement between differently augmented samples in the latent space and reconstruction loss of input representation. For more information, please refer to following paper(s).

[Speech SimCLR: Combining Contrastive and Reconstruction Objective for Self-supervised Speech Representation Learning](https://arxiv.org/abs/2010.13991.pdf)

For now, pre-training with Speech SimCLR is only supported for Librispeech. You can run it with ```python athena/main.py examples/asr/librispeech/configs/speech_simclr.json```. For feature extraction, simply run ```python athena/inference.py examples/asr/librispeech/configs/speech_simclr.json```. The pre-trained Speech SimCLR models can be found [here](https://drive.google.com/file/d/1YYFmtB1RHRuw8s7lPWLxjihye9ssI5ax/view?usp=sharing).

## 8) Results

### 8.1) ASR

Language  | Model Name | Training Data | Hours of Speech | Error Rate
:-----------: | :------------: | :----------: |  -------: | -------:
English  | Transformer | [LibriSpeech Dataset](http://www.openslr.org/12/) | 960 h | 3.1% (WER)
Mandarin | Transformer | HKUST Dataset | 151 h | 22.75% (CER)
Mandarin | Transformer | [AISHELL Dataset](http://www.openslr.org/33/) | 178 h | 6.6% (CER)

To compare with other published results, see [wer_are_we.md](docs/tutorials/wer_are_we.md).

## 9) Directory Structure

Below is the basic directory structure for Athena

```bash
|-- Athena
|   |-- data  # - root directory for input-related operations
|   |   |-- datasets  # custom datasets for ASR, TTS and pre-training
|   |-- layers  # some layers
|   |-- models  # some models
|   |-- tools # contains various tools, e.g. decoding tools
|   |-- transform # custom featureizer based on C++
|   |   |-- feats
|   |   |   |-- ops # c++ code on tensorflow ops
|   |-- utils # utils, e.g. checkpoit, learning_rate, metric, etc
|-- deploy  # deployment with Tensorflow C++
|   |-- include
|   |-- src
|-- docker
|-- docs  # docs
|-- examples  # example scripts for ASR, TTS, etc
|   |-- asr  # each subdirectory contains a data preparation scripts and a run script for the task
|   |   |-- aishell
|   |   |-- hkust
|   |   |-- librispeech
|-- tools  # need to source env.sh before training
```
