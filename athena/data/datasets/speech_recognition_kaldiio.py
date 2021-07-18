# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Shuaijiang Zhao; Ne Luo; Yanguang Xu
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
# pylint: disable=no-member, invalid-name
""" audio dataset """

import os
import sys
import numpy as np
from ...utils.logger import LOG as logging
import tensorflow as tf
import kaldiio
from .speech_recognition import SpeechRecognitionDatasetBuilder
from .preprocess import SpecAugment

class SpeechRecognitionDatasetKaldiIOBuilder(SpeechRecognitionDatasetBuilder):
    """SpeechRecognitionDatasetKaldiIOBuilder
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "text_config": {"type":"vocab", "model":"athena/utils/vocabs/ch-en.vocab"},
        "cmvn_file": None,
        "remove_unk": True,
        "sort_and_filter": True,
        "input_length_range": [20, 50000],
        "output_length_range": [1, 10000],
        "speed_permutation": [1.0],
        "spectral_augmentation": None,
        "data_csv": None,
        "data_scps_dir": None,
        "words": None,
        "apply_cmvn": True,
        "global_cmvn": True,
        "offline": False
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        if self.hparams.data_scps_dir == None:
            pass
            logging.warning("created a fake dataset")
        if self.hparams.data_scps_dir is not None and not self.hparams.offline:
            self.preprocess_data(self.hparams.data_scps_dir, apply_sort_filter=self.hparams.sort_and_filter)
        if self.hparams.spectral_augmentation is not None:
            self.spectral_augmentation = SpecAugment(self.hparams.spectral_augmentation)

    def storage_features_offline(self, file_path, file_dir):
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        if not os.path.exists(file_path):
            logging.error("{} does not exist".format(file_path))
            sys.exit()
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
        headers = lines[0]
        lines = lines[1:]
        lines = [line.split("\t") for line in lines]
        self.entries = [tuple(line) for line in lines]

        # handling speakers
        if self.hparams.global_cmvn or "speaker" not in headers.split("\t"):
            entries = self.entries
            self.entries = []
            for wav_filename, wav_len, transcripts, _ in entries:
                self.entries.append(
                    tuple([wav_filename, wav_len, transcripts, "global"])
                )
            
        # handling speed
        entries = self.entries
        self.entries = []
        if len(self.hparams.speed_permutation) > 1:
            logging.info("perform speed permutation")
        for speed in self.hparams.speed_permutation:
            for wav_filename, wav_len, transcripts, speaker in entries:
                self.entries.append(
                    tuple([wav_filename,
                    float(wav_len) / float(speed), transcripts, speed, speaker
                ]))
            
        self.entries.sort(key=lambda item: float(item[1]))
        # handling special case for text_featurizer
        if self.text_featurizer.model_type == "text":
            _, _, all_transcripts, _, _ = zip(*self.entries)
            self.text_featurizer.load_model(all_transcripts)
            
        # apply some filter
        unk = self.text_featurizer.unk_index
        if self.hparams.remove_unk and unk != -1:
            self.entries = list(filter(lambda x: unk not in
                                self.text_featurizer.encode(x[2]), self.entries))
        self.entries = list(filter(lambda x: int(x[1]) in
                            range(self.hparams.input_length_range[0],
                            self.hparams.input_length_range[1]), self.entries))
        self.entries = list(filter(lambda x: len(x[2]) in
                            range(self.hparams.output_length_range[0],
                            self.hparams.output_length_range[1]), self.entries))

        feat_dict = {}
        label_dict = {}
        for i in range(len(self.entries)):
            audio_data, _, transcripts, speed, speaker = self.entries[i]
            feat = self.audio_featurizer(audio_data, speed=speed)
            if self.hparams.apply_cmvn:
                if os.path.exists(self.hparams.cmvn_file): 
                    feat = self.feature_normalizer(feat, speaker)
                else:
                    logging.warning("{} does not exist".format(self.hparams.cmvn_file))
            if self.hparams.spectral_augmentation is not None:
                feat = self.spectral_augmentation(feat)
            feat = np.reshape(feat, (-1, self.hparams.audio_config["filterbank_channel_count"]))
            label = self.text_featurizer.encode(transcripts)
            if str(speed) == "1.0":
                audio_data = os.path.basename(audio_data).replace(".wav","")
            else:
                audio_data = os.path.basename(audio_data).replace(".wav","_"+str(speed))
            feat_dict[audio_data] = feat
            label_dict[audio_data] = np.array(label, dtype='int32')
        kaldiio.save_ark(os.path.join(file_dir,"feats.ark"), feat_dict, scp=os.path.join(file_dir,"feats.scp"))
        kaldiio.save_ark(os.path.join(file_dir,"labels.ark"), label_dict, scp=os.path.join(file_dir,"labels.scp"))
        return self

    def preprocess_data(self, file_dir, apply_sort_filter=True):
        """ Generate a list of tuples (feat_key, speaker). """
        logging.info("Loading kaldi-format feats.scp, labels.scp " + \
            "and utt2spk (optional) from {}".format(file_dir))
        self.kaldi_io_feats = kaldiio.load_scp(os.path.join(file_dir, "feats.scp"))
        self.kaldi_io_labels = kaldiio.load_scp(os.path.join(file_dir, "labels.scp"))

        # data checking
        if self.kaldi_io_feats.keys() != self.kaldi_io_labels.keys():
            logging.info("Error: feats.scp and labels.scp does not contain same keys, " + \
                "please check your data.")
            sys.exit()

        # initialize all speakers with 'global'
        # unless 'utterance_key speaker' is specified in "utt2spk"
        self.speakers = dict.fromkeys(self.kaldi_io_feats.keys(), 'global')
        if os.path.exists(os.path.join(file_dir, "utt2spk")):
            with open(os.path.join(file_dir, "utt2spk"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    key, spk = line.strip().split(" ", 1)
                    self.speakers[key] = spk

        self.entries = []
        for key in self.kaldi_io_feats.keys():
            self.entries.append(tuple([key, self.speakers[key]]))

        if apply_sort_filter:
            logging.info("Sorting and filtering data, this is very slow, please be patient ...")
            self.entries.sort(key=lambda item: self.kaldi_io_feats[item[0]].shape[0])
            unk = self.text_featurizer.unk_index
            if self.hparams.remove_unk and unk != -1:
                self.entries = list(filter(lambda x: unk not in
                                    self.kaldi_io_labels[x[0]], self.entries))
            # filter length of frames
            self.entries = list(filter(lambda x: self.kaldi_io_feats[x[0]].shape[0] in
                                range(self.hparams.input_length_range[0],
                                self.hparams.input_length_range[1]), self.entries))
            self.entries = list(filter(lambda x: self.kaldi_io_labels[x[0]].shape[0] in
                                range(self.hparams.output_length_range[0],
                                self.hparams.output_length_range[1]), self.entries))
        return self

    def __getitem__(self, index):
        key, speaker = self.entries[index]
        feat = self.kaldi_io_feats[key]
        feat = feat.reshape(feat.shape[0], feat.shape[1], 1)
        feat = tf.convert_to_tensor(feat)
        if self.hparams.apply_cmvn:
            feat = self.feature_normalizer(feat, speaker)
        if self.hparams.spectral_augmentation is not None:
            feat = self.spectral_augmentation(feat)
        label = list(self.kaldi_io_labels[key])

        feat_length = feat.shape[0]
        label_length = len(label)
        return {
            "input": feat,
            "input_length": feat_length,
            "output_length": label_length,
            "output": label,
        }

    def compute_cmvn_if_necessary(self, is_necessary=True):
        """ compute cmvn file
        """
        if not is_necessary:
            return self
        if os.path.exists(self.hparams.cmvn_file):
            return self
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        with tf.device("/cpu:0"):
            self.feature_normalizer.compute_cmvn_kaldiio(
                self.entries, self.speakers, self.kaldi_io_feats, feature_dim
            )
        self.feature_normalizer.save_cmvn(["speaker", "mean", "var"])
        return self
