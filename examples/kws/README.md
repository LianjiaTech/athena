# 1. prepare data
prepare the video and audio data and extract feature for training procedure
### 1.1 create data dir to contain misp audio and video feature data
```bash
mkdir -p examples/kws/misp/data
```
### 1.2 prepare audio feature dir for audio model; we assume that all the audio have been in the feature format.
```bash
# the dir contains feats.scp and labels.scp
mkdir -p examples/kws/misp/data/train
mkdir -p examples/kws/misp/data/dev
```
### 1.3 prepare audio+video feature dir for audio-video model
```bash
# the dir contains feats.scp , labels.scp and video.scp
mkdir -p examples/kws/misp/data/train_av
mkdir -p examples/kws/misp/data/dev_av

```


# 2. train model
4 kinds of models are offered now:
* conformer/transformer using only audio or video data
* fine tune model for conformer/transformer using focal-loss and label_smoothing
* audio-visual transformer model using both audio and video data by 2 kinds of fusion operation
* Majority Vote by all models

### 2.1 train audio transformer/conformer
1. run the following commands to start training audio transformer/conformer
```bash
python athena/main.py examples/kws/misp/configs/kws_audio_conformer.json
python athena/main.py examples/kws/misp/configs/kws_audio_transformer.json
```
2. if you have multiple GPUs , you can train models parallel using the following commands
```bash
python athena/horovod_main.py examples/kws/misp/configs/kws_audio_conformer.json
python athena/horovod_main.py examples/kws/misp/configs/kws_audio_transformer.json
```
3. the model will be stored in ``examples/kws/misp/ckpts/kws_audio_conformer`` and ``examples/kws/misp/ckpts/kws_audio_transformer``

### 2.2 fine-tune audio transformer using focal-loss
1. focal-loss wii be used to fine tune model to get improvements
```bash
python athena/main.py examples/kws/misp/configs/kws_audio_transformer_finuetune_ft.json
```
### 2.3 train audio-video transformer
1. train model using multi-moda data and the model will be stored in ``examples/kws/misp/ckpts/kws_av_transformer``
```bash
python athena/main.py examples/kws/misp/configs/kws_av_transformer.json
```


# 3. test model

### 3.1 test audio transformer/conformer
1. test the trained model and the FRR and FAR will be shown
```bash
python examples/kws/test_main.py examples/kws/misp/configs/kws_audio_conformer.json
python examples/kws/test_main.py examples/kws/misp/configs/kws_audio_transformer.json
```

### 3.2 test audio-video transformer
1. test the trained model
```bash
python examples/kws/test_main_av.py examples/kws/misp/configs/kws_av_transformer.json
```

# 4. model vote
1. As you have got audio transformer and audio-video transformer, you can use mode vote to get better results

# 5. About MISP Challenge 2021
### 5.1 MISP Challenge 2021 webset:https://mispchallenge.github.io/index.html
### 5.2 Our final score is 0.091 and ranked 3rd among all the 17 teams
### 5.3 the Paper ["AUDIO-VISUAL WAKE WORD SPOTTING SYSTEM FOR MISP CHALLENGE 2021"](https://arxiv.org/abs/2204.08686) have been accepted by ICASSP 2022

# Citation
``` bibtex
@INPROCEEDINGS{9747216,
  author={Cheng, Ming and Wang, Haoxu and Wang, Yechen and Li, Ming},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={The DKU Audio-Visual Wake Word Spotting System for the 2021 MISP Challenge}, 
  year={2022},
  volume={},
  number={},
  pages={9256-9260},
  doi={10.1109/ICASSP43922.2022.9747216}}
```