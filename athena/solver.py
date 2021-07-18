# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Jianwei Sun; Ruixiong Zhang; Dongwei Jiang; Chunxin Xiao
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
# pylint: disable=arguments-differ
# pylint: disable=no-member
""" high-level abstraction of different stages in speech processing """

import warnings
import time
import os
import tensorflow as tf
from .utils.logger import LOG as logging
try:
    import horovod.tensorflow as hvd
except ImportError:
    print("There is some problem with your horovod installation. But it wouldn't affect single-gpu training")
import pyworld
import numpy as np
import librosa
from .utils.hparam import register_and_parse_hparams
from .utils.metric_check import MetricChecker
from .utils.misc import validate_seqs
from .metrics import CharactorAccuracy


class BaseSolver(tf.keras.Model):
    """Base Solver.
    """
    default_config = {
        "clip_norm": 100.0,
        "log_interval": 10,
        "ckpt_interval": 10000,
        "enable_tf_function": True
    }
    def __init__(self, model, optimizer, sample_signature, eval_sample_signature=None,
                 config=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.metric_checker = MetricChecker(self.optimizer)
        self.sample_signature = sample_signature
        self.eval_sample_signature = eval_sample_signature
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

    @staticmethod
    def initialize_devices(solver_gpus=None):
        """ initialize hvd devices, should be called firstly """
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # means we're running in GPU mode
        if len(gpus) != 0:
            # If the list of solver gpus is empty, the first gpu will be used.
            if len(solver_gpus) == 0:
                solver_gpus.append(0)
            assert len(gpus) >= len(solver_gpus)
            for idx in solver_gpus:
                tf.config.experimental.set_visible_devices(gpus[idx], "GPU")

    @staticmethod
    def clip_by_norm(grads, norm):
        """ clip norm using tf.clip_by_norm """
        if norm <= 0:
            return grads
        grads = [
            None if gradient is None else tf.clip_by_norm(gradient, norm)
            for gradient in grads
        ]
        return grads

    def train_step(self, samples):
        """ train the model 1 step """
        with tf.GradientTape() as tape:
            # outputs of a forward run of model, potentially contains more than one item
            outputs = self.model(samples, training=True)
            loss, metrics = self.model.get_loss(outputs, samples, training=True)
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics

    def train(self, dataset, checkpointer, pbar, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        step=self.optimizer.iterations.numpy()
        pbar.update(step)
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            train_step = tf.function(train_step, input_signature=self.sample_signature)
        for batch, samples in enumerate(dataset.take(total_batches)):
            # train 1 step
            samples = self.model.prepare_samples(samples)
            loss, metrics = train_step(samples)
            if batch % self.hparams.log_interval == 0:
                pbar.set_description('Training ')  # it=iterations
                logging.info(self.metric_checker(loss, metrics))
                self.model.reset_metrics()
                pbar.update(self.hparams.log_interval)
            if batch % self.hparams.ckpt_interval == 0:
                checkpointer(loss, metrics)

    def evaluate_step(self, samples):
        """ evaluate the model 1 step """
        # outputs of a forward run of model, potentially contains more than one item
        outputs = self.model(samples, training=False)
        loss, metrics = self.model.get_loss(outputs, samples, training=False)
        return loss, metrics

    def evaluate(self, dataset, epoch):
        """ evaluate the model """
        loss_metric = tf.keras.metrics.Mean(name="AverageLoss")
        loss, metrics = None, None
        evaluate_step = self.evaluate_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            evaluate_step = tf.function(evaluate_step, input_signature=self.eval_sample_signature)
        self.model.reset_metrics()  # init metric.result() with 0
        for batch, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            loss, metrics = evaluate_step(samples)
            if batch % self.hparams.log_interval == 0:
                logging.info(self.metric_checker(loss, metrics, -2))
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
            loss_metric.update_state(total_loss)
        logging.info(self.metric_checker(loss_metric.result(), metrics, evaluate_epoch=epoch))
        self.model.reset_metrics()
        return loss_metric.result(), metrics

class HorovodSolver(BaseSolver):
    """ A multi-processer solver based on Horovod """

    @staticmethod
    def initialize_devices(solver_gpus=None):
        """initialize hvd devices, should be called firstly

        For examples, if you have two machines and each of them contains 4 gpus:
        1. run with command horovodrun -np 6 -H ip1:2,ip2:4 and set solver_gpus to be [0,3,0,1,2,3],
           then the first gpu and the last gpu on machine1 and all gpus on machine2 will be used.
        2. run with command horovodrun -np 6 -H ip1:2,ip2:4 and set solver_gpus to be [],
           then the first 2 gpus on machine1 and all gpus on machine2 will be used.

        Args:
            solver_gpus ([list]): a list to specify gpus being used.

        Raises:
            ValueError: If the list of solver gpus is not empty,
                        its size should not be smaller than that of horovod configuration.
        """
        hvd.init()
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if len(gpus) != 0:
            if len(solver_gpus) > 0:
                if len(solver_gpus) < hvd.size():
                    raise ValueError("If the list of solver gpus is not empty, its size should " +
                                     "not be smaller than that of horovod configuration")
                tf.config.experimental.set_visible_devices(gpus[solver_gpus[hvd.rank()]], "GPU")
            # If the list of solver gpus is empty, the first hvd.size() gpus will be used.
            else:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    def train_step(self, samples):
        """ train the model 1 step """
        with tf.GradientTape() as tape:
            # outputs of a forward run of model, potentially contains more than one item
            outputs = self.model(samples, training=True)
            loss, metrics = self.model.get_loss(outputs, samples, training=True)
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics

    def train(self, dataset, checkpointer, pbar, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        step=self.optimizer.iterations.numpy()
        pbar.update(step)
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            train_step = tf.function(train_step, input_signature=self.sample_signature)
        for batch, samples in enumerate(dataset.take(total_batches)):
            # train 1 step
            samples = self.model.prepare_samples(samples)
            loss, metrics = train_step(samples)
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            #
            # Note: broadcast should be done after the first gradient step to ensure optimizer
            # initialization.
            if batch == 0:
                hvd.broadcast_variables(self.model.trainable_variables, root_rank=0)
                hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)
            if hvd.rank() == 0:
                if batch % self.hparams.log_interval == 0:
                    pbar.set_description('Processing ')  # it=iterations
                    logging.info(self.metric_checker(loss, metrics))
                    self.model.reset_metrics()
                    pbar.update(self.hparams.log_interval)
                if batch % self.hparams.ckpt_interval == 0:
                    checkpointer(loss, metrics)

    def evaluate(self, dataset, epoch=0):
        """ evaluate the model """
        loss_metric = tf.keras.metrics.Mean(name="AverageLoss")
        loss, metrics = None, None
        evaluate_step = self.evaluate_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            evaluate_step = tf.function(evaluate_step, input_signature=self.eval_sample_signature)
        self.model.reset_metrics()
        for batch, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            loss, metrics = evaluate_step(samples)
            if batch % self.hparams.log_interval == 0 and hvd.rank() == 0:
                logging.info(self.metric_checker(loss, metrics, -2))
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
            loss_metric.update_state(total_loss)
        if hvd.rank() == 0:
            logging.info(self.metric_checker(loss_metric.result(), metrics, evaluate_epoch=epoch))
            self.model.reset_metrics()
        return loss_metric.result(), metrics


class DecoderSolver(BaseSolver):
    """ DecoderSolver
    """
    default_config = {
        "inference_type": "asr",
        "decoder_type": "wfst_decoder",
        "model_avg_num": 1,
        "beam_size": 4,
        "ctc_weight": 0.0,
        "lm_weight": 0.1,
        "lm_type": "",
        "lm_path": None,
        "acoustic_scale": 10.0,
        "max_active": 80,
        "min_active": 0,
        "wfst_beam": 30.0,
        "max_seq_len": 100,
        "wfst_graph": None
    }

    # pylint: disable=super-init-not-called
    def __init__(self, model, data_descriptions=None, config=None):
        super().__init__(model, None, None)
        self.model = model
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.lm_model = None
        if self.hparams.lm_type == "rnn":
            from athena.main import build_model_from_jsonfile
            _, self.lm_model, _, lm_checkpointer = build_model_from_jsonfile(self.hparams.lm_path)
            lm_checkpointer.restore_from_best()

    def inference(self, dataset, rank_size=1):
        """ decode the model """
        if dataset is None:
            return
        metric = CharactorAccuracy(rank_size=rank_size)
        st = time.time()
        for _, samples in enumerate(dataset):
            begin = time.time()
            samples = self.model.prepare_samples(samples)
            predictions = self.model.decode(samples, self.hparams, self.lm_model)
            predictions = tf.cast(predictions, tf.int64)
            validated_preds, _ = validate_seqs(predictions, self.model.eos)
            validated_preds = tf.cast(validated_preds, tf.int64)
            num_errs, _ = metric.update_state(validated_preds, samples)
            reports = (
                "predictions: %s\tlabels: %s\terrs: %d\tavg_acc: %.4f\tsec/iter: %.4f"
                % (
                    predictions,
                    samples["output"].numpy(),
                    num_errs,
                    metric.result(),
                    time.time() - begin,
                )
            )
            logging.info(reports)
        ed = time.time()
        logging.info("decoding finished, cost %.4f s"%(ed-st))

