# coding=utf-8
# Copyright (C) ATHENA AUTHORS; Yang Han; Yanguang Xu Jianwei Sun;
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
# pylint: disable=invalid-name, no-member, redefined-outer-name
import sys
import os
import tensorflow as tf
from absl import logging
from athena import *
from athena.main import (
    parse_jsonfile,
    SUPPORTED_DATASET_BUILDER,
    build_model_from_jsonfile
)
import json
import tqdm
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SmoothWakeup:
    def __init__(self, scores):
        self.scores = tf.squeeze(scores)
        self.smooth_window = 5
        self.label_num = 8 # wake up numbers 
        self.wakeup = [3, 4, 5, 6, 3, 4, 5, 6]
        self.max_window = 300 
        self.nframe = tf.shape(self.scores)[0]
        self.smooth_scores = []
        self.threshold = 0.3
        self.thresholds = [0.1, -1, 0.1, 0.1,-1, -1, -1, -1]
        self.locations = []
        self.wakeup_frame = -1
        self.best_score = -1.0

    def cal_confidence(self, end):
        if end < self.label_num - 1:
            return 0.0
        start = max(0, end - self.max_window)
        window_size = end - start + 1

        cfd = [[0.0 for _ in range(self.label_num+1)] for _ in range(window_size+1)]
        self.locations = [[-1 for _ in range(self.label_num+1)] for _ in range(window_size+1)]

        for i in range(window_size+1):
            cfd[i][0] = 1.0
        for frame in range(1, window_size+1):
            for label in range(1, self.label_num+1):
                if label > frame: continue
                i = frame - 1
                m = label - 1
                if cfd[frame-1][label] > cfd[frame-1][label-1] * self.smooth_scores[start+i][self.wakeup[m]]:
                    cfd[frame][label] = cfd[frame-1][label]
                    self.locations[frame][label] = self.locations[frame-1][label]
                else:
                    cfd[frame][label] = cfd[frame-1][label-1] * self.smooth_scores[start+i][self.wakeup[m]]
                    self.locations[frame][label] = frame

        cfd_score = pow(cfd[window_size][self.label_num], 1.0/self.label_num)
        return cfd_score

    def show_smooth_scores(self):
        for scores in self.smooth_scores:
            logging.info(list(scores))


    def smooth(self):
        for i in range(self.nframe):
            start = max(0, i-self.smooth_window)
            smooth_score = tf.reduce_mean(self.scores[start:i+1,:], axis=0)
            self.smooth_scores.append(smooth_score.numpy())

    def backtrace(self):
        if self.wakeup_frame == -1:
            logging.info("do not wakeup, cannot backtrace")
            return

        end = self.wakeup_frame
        start = max(0, end - self.max_window)
        window_size = end - start + 1

        frame = window_size
        label = self.label_num
        while frame > 0:
            location = self.locations[frame][label]
            if location != -1:
                logging.info("label:{} at:{} frame score:{}".format(self.wakeup[label-1], start + location -1 , self.smooth_scores[start+location-1][self.wakeup[label-1]]))
            frame = location
            label -= 1

    def predict(self):
        for current_frame in range(self.nframe):
            cfd_score = self.cal_confidence(current_frame)
            #logging.info("the {} frame confidence is {}".format(current_frame, cfd_score))
            if cfd_score > self.best_score:
                self.best_score = cfd_score

            if cfd_score >= self.threshold:
                self.wakeup_frame = current_frame
                return (True, cfd_score)
        return (False, cfd_score)

def test(jsonfile, ckpt="all", accuracy=True, false_alarm=True):
    p, checkpointer, dataset_builder = build_model_from_jsonfile(jsonfile)
    avg_file = os.path.join(checkpointer.checkpoint_directory, 'n_best_dev')
    if not os.path.exists(avg_file):
        logging.info("n_best file do not exists")
        return
    ckpt_metrics_dict = {}
    with open(avg_file) as f:
        for line in f:
            key, val = line.split('\t')
            ckpt_metrics_dict[key] = float(val.strip())
    ckpt_list = [k for k in ckpt_metrics_dict.keys()]
    if ckpt == "all":
        logging.info('checkpoint list: %s' % ckpt_list)
    else:
        if ckpt not in ckpt_list:
            logging.info("ckpt:{} do not exists".format(ckpt))
            return
        logging.info('checkpoint list: %s' % ckpt)
        ckpt_list = [ckpt]

    for key in ckpt_list:
        ckpt_path = os.path.join(checkpointer.checkpoint_directory, key)
        checkpointer.restore(ckpt_path)

        if accuracy:
            testset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.testset_config)
            accuracy = test_wakeup_accuracy(model, testset_builder)
            logging.info("ckpt:{} accuracy:{}".format(key, accuracy))

        if false_alarm:
            faset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.faset_config)
            fa =  test_false_alarm(model,faset_builder)
            logging.info("ckpt:{} false_alarm:{}".format(key, fa))
        
def test_wakeup_accuracy(model, builder, debug=False):
    testset = builder.as_dataset(1)
    total, count = 0, 0
    for samples in tqdm.tqdm(testset, total=len(builder)):
        scores = model.tflite_model(samples['input'])
        sw = SmoothWakeup(scores)
        sw.smooth()
        sign, cfd = sw.predict()
        if sign:
            if debug: logging.info("confidence scores: {}".format(cfd))
            count += 1
        else:
            if debug: logging.info("Do not Wakeup, best confidence score is {}".format(sw.best_score))
        total += 1
    return count/total

def test_false_alarm(model, builder, debug=False):
    faset = builder.as_dataset(1)
    count = 0
    for samples in tqdm.tqdm(faset, total=len(builder)):
        scores = model.tflite_model(samples['input'])
        sw = SmoothWakeup(scores)
        sw.smooth()
        sign, cfd = sw.predict()
        if sign:
            if debug: logging.info("False Alarm, confidence score is {}".format(cfd))
            count += 1
    return count

def test_frame_accuracy(model, testset, debug=False):
    global_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for batch, samples in enumerate(testset):
        predictions = model.tflite_model(samples['input'])
        local_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        local_metric.update_state(samples['output'], predictions)
        if debug: logging.info("batch:{} local frame level accuracy:{}".format(batch, local_metric.result().numpy()))
        global_metric.update_state(samples['output'], predictions)
    return global_metric.result().numpy()


def test_e2e(jsonfile, ckpt="all",):
    p, checkpointer, dataset_builder = build_model_from_jsonfile(jsonfile)
    avg_file = os.path.join(checkpointer.checkpoint_directory, 'n_best_dev')
    if not os.path.exists(avg_file):
        logging.info("n_best file do not exists")
        return
    ckpt_metrics_dict = {}
    with open(avg_file) as f:
        for line in f:
            key, val,_ = line.split('\t') #you can select the model using the "loss" or "dev acc" as you want
            ckpt_metrics_dict[key] = float(val.strip())
    ckpt_metrics_dict = sorted(ckpt_metrics_dict.items(), key=lambda x: x[1], reverse=True)
    ckpt_list = [k[0] for k in ckpt_metrics_dict]
    for key in ckpt_list:
        # ckpt_path = os.path.join(checkpointer.checkpoint_directory, 'ckpt-1')
        ckpt_path = os.path.join(checkpointer.checkpoint_directory, key)
        checkpointer.restore(ckpt_path)
        #checkpointer.compute_nbest_avg(10)
        testset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.testset_config)
        TP, FP, FN, TN = 0, 0, 0, 0
        for samples in testset_builder.as_dataset(1):
            samples = model.prepare_samples(samples)
            frame = math.ceil(samples["input"].shape[1]/4)
            if frame > samples["video"].shape[1]:
                frame = samples["video"].shape[1]
                audio_frame = frame * 4
                audio_frames = tf.ones(samples["input_length"].shape,dtype=tf.dtypes.int32)*audio_frame
                samples["input"] = samples["input"][:, :audio_frame, :]
                samples["video"] = samples["video"][:, :frame, :]
                samples["input_length"] = tf.where(samples["input_length"]>audio_frames,audio_frames,samples["input_length"])
            else:
                samples["video"] = samples["video"][:, :frame, :]

            prediction = model(samples, training=False) - 0.06
            prediction = tf.math.sigmoid(prediction, name="sigmoid")
            # print("prediction: "+str(prediction))
            prediction = tf.round(prediction)

            if prediction == [1] and samples['output'] == [1]:
                TP += 1
            elif prediction == [1] and samples['output'] == [0]:
                FP += 1
            elif prediction == [0] and samples['output'] == [1]:
                FN += 1
            elif prediction == [0] and samples['output'] == [0]:
                TN += 1
            '''
            if num%10==0:
                logging.info(f'model:{key} TP:{TP} FP:{FR} FN:{FN} TN:{TN}')
            '''
        FAR = FP/(FP + TN)
        FRR = FN/(FN + TP)
        score = FAR + FRR
        logging.info(f'model:{key} FAR:{FAR} FRR:{FRR} score:{score}')


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)
    jsonfile = sys.argv[1]
    p = parse_jsonfile(jsonfile)
    #test(jsonfile, ckpt=p.test_ckpt, accuracy=True, false_alarm=False)
    #import pdb;pdb.set_trace()
    #test_e2e(jsonfile, ckpt=p.test_ckpt)
    test_e2e(jsonfile, ckpt="all")




