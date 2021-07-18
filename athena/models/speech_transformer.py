# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Xiaoning Lei
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
# Only support eager mode
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes

""" speech transformer implementation"""

from ..utils.logger import LOG as logging
import tensorflow as tf
from .base import BaseModel
from ..loss import Seq2SeqSparseCategoricalCrossentropy
from ..metrics import Seq2SeqSparseCategoricalAccuracy
from ..utils.misc import generate_square_subsequent_mask, insert_sos_in_labels, create_multihead_mask, mask_finished_preds, mask_finished_scores
from ..layers.commons import PositionalEncoding
from ..layers.transformer import Transformer
from ..utils.hparam import register_and_parse_hparams


class SpeechTransformer(BaseModel):
    """ Standard implementation of a SpeechTransformer. Model mainly consists of three parts:
    the x_net for input preparation, the y_net for output preparation and the transformer itself
    """
    default_config = {
        "return_encoder_output": False,
        "num_filters": 512,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 12,
        "num_decoder_layers": 6,
        "dff": 1280,
        "rate": 0.1,
        "schedual_sampling_rate": 0.9,
        "label_smoothing_rate": 0.0,
        "unidirectional": False,
        "look_ahead": 0,
        "conv_module_kernel_size": 0
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.num_class = data_descriptions.num_class + 1
        self.sos = self.num_class - 1
        self.eos = self.num_class - 1
        ls_rate = self.hparams.label_smoothing_rate
        self.loss_function = Seq2SeqSparseCategoricalCrossentropy(
            num_classes=self.num_class, eos=self.eos, label_smoothing=ls_rate
        )
        self.metric = Seq2SeqSparseCategoricalAccuracy(eos=self.eos, name="Accuracy")

        # for the x_net
        num_filters = self.hparams.num_filters
        d_model = self.hparams.d_model
        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"], dtype=tf.float32)
        inner = layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(input_features)
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu6(inner)
        inner = layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(inner)
        inner = layers.BatchNormalization()(inner)

        inner = tf.nn.relu6(inner)
        _, _, dim, channels = inner.get_shape().as_list()
        output_dim = dim * channels
        inner = layers.Reshape((-1, output_dim))(inner)

        inner = layers.Dense(d_model, activation=tf.nn.relu6)(inner)
        inner = PositionalEncoding(d_model, scale=False)(inner)
        inner = layers.Dropout(self.hparams.rate)(inner)  # self.hparams.rate
        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")
        print(self.x_net.summary())

        # y_net for target
        input_labels = layers.Input(shape=data_descriptions.sample_shape["output"], dtype=tf.int32)
        inner = layers.Embedding(self.num_class, d_model)(input_labels)
        inner = PositionalEncoding(d_model, scale=True)(inner)
        inner = layers.Dropout(self.hparams.rate)(inner)
        self.y_net = tf.keras.Model(inputs=input_labels, outputs=inner, name="y_net")
        print(self.y_net.summary())

        # transformer layer
        self.transformer = Transformer(
            self.hparams.d_model,
            self.hparams.num_heads,
            self.hparams.num_encoder_layers,
            self.hparams.num_decoder_layers,
            self.hparams.dff,
            self.hparams.rate,
            unidirectional=self.hparams.unidirectional,
            look_ahead=self.hparams.look_ahead,
            conv_module_kernel_size=self.hparams.conv_module_kernel_size
        )

        # last layer for output
        self.final_layer = layers.Dense(self.num_class, input_shape=(d_model,))

        # some temp function
        self.random_num = tf.random_uniform_initializer(0, 1)

    def call(self, samples, training: bool = None):
        x0 = samples["input"]
        y0 = insert_sos_in_labels(samples["output"], self.sos)
        x = self.x_net(x0, training=training)
        y = self.y_net(y0, training=training)
        input_length = self.compute_logit_length(samples)
        input_mask, output_mask = create_multihead_mask(x, input_length, y0)
        y, encoder_output = self.transformer(
            x,
            y,
            input_mask,
            output_mask,
            input_mask,
            training=training,
            return_encoder_output=True,
        )
        y = self.final_layer(y)
        if self.hparams.return_encoder_output:
            return y, encoder_output
        return y

    def compute_logit_length(self, samples):
        """ used for get logit length """
        input_length = tf.cast(samples["input_length"], tf.float32)
        logit_length = tf.math.ceil(input_length / 2)
        logit_length = tf.math.ceil(logit_length / 2)
        logit_length = tf.cast(logit_length, tf.int32)
        return logit_length

    def beam_search(self, samples, hparams, lm_model=None):
        """ batch beam search for transformer model

        Args:
            samples: the data source to be decoded
            beam_size: beam size
            lm_model: rnnlm that used for beam search

        """
        x0 = samples["input"]
        batch_size = tf.shape(x0)[0]
        beam_size = hparams.beam_size
        x = self.x_net(x0, training=False)
        input_length = self.compute_logit_length(samples)
        input_mask, _ = create_multihead_mask(x, input_length, None)
        encoder_output = self.transformer.encoder(x, input_mask, training=False)
        maxlen = tf.shape(encoder_output)[1]
        encoder_dim = tf.shape(encoder_output)[2]
        running_size = batch_size * beam_size

        # repeat for beam_size
        input_mask = tf.tile(input_mask, [beam_size, 1, 1, 1])
        encoder_output = tf.tile(encoder_output, [beam_size, 1, 1])

        hyps = tf.ones([running_size, 1], dtype=tf.int32) * self.sos
        scores = tf.constant([[0.0]+[-float('inf')] * (beam_size-1)], shape=(beam_size, 1), dtype=tf.float32)
        scores = tf.tile(scores, [batch_size, 1])

        end_flag = tf.zeros_like(scores, dtype=tf.int32)

        for step in tf.range(1, maxlen+1):
            if tf.reduce_sum(end_flag) == running_size:
                break
            output_mask = generate_square_subsequent_mask(step)

            y0 = self.y_net(hyps, training=False)

            decoder_outputs = self.transformer.decoder(
                y0,
                encoder_output,
                tgt_mask=output_mask,
                memory_mask=input_mask,
                training=False,
            )
            logits = self.final_layer(decoder_outputs)
            logit = logits[:, -1, :]
            logprob = tf.math.log(tf.nn.softmax(logit))

            # add language model score
            if lm_model is not None:
                lm_logits = lm_model.rnnlm(hyps, training=False)
                lm_logit = lm_logits[:, -1, :]
                lm_logprob = tf.math.log(tf.nn.softmax(lm_logit))
                lm_weight = hparams.lm_weight
                logprob = ((1-lm_weight) * logprob) + (lm_weight * lm_logprob)

            top_k_logp, top_k_index = tf.math.top_k(logprob, k=beam_size)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)

            scores = scores + top_k_logp
            scores = tf.reshape(scores, [batch_size, beam_size*beam_size])

            scores, best_k_index = tf.math.top_k(scores, k=beam_size)
            scores = tf.reshape(scores, [batch_size*beam_size, 1])

            row_index = best_k_index // beam_size
            col_index = best_k_index % beam_size

            batch_index = tf.range(batch_size)
            batch_index = tf.reshape(batch_index, [batch_size, 1])
            batch_index = tf.tile(batch_index, [1, beam_size])
            batch_index = tf.reshape(batch_index, [batch_size, beam_size, 1])

            row_index = tf.expand_dims(row_index, axis=2)
            col_index = tf.expand_dims(col_index, axis=2)
            indices = tf.concat([batch_index, row_index, col_index], axis=2)


            top_k_index = tf.reshape(top_k_index, [batch_size, beam_size, beam_size])
            best_k_pred = tf.gather_nd(top_k_index, indices)
            best_k_pred = tf.reshape(best_k_pred, [batch_size*beam_size, 1])

            last_best_k_index = batch_index * beam_size + row_index
            last_best_k_index = tf.reshape(last_best_k_index, [running_size, 1])
            last_best_k_hyps = tf.gather(hyps, last_best_k_index,axis=0)
            last_best_k_hyps = tf.reshape(last_best_k_hyps, [running_size, -1])
            hyps = tf.concat([last_best_k_hyps, best_k_pred], axis=1)

            end_flag = tf.equal(hyps[:,-1], self.eos)
            end_flag = tf.cast(end_flag, dtype=tf.int32)
            end_flag = tf.expand_dims(end_flag, axis=1)

        batch_index = tf.range(batch_size)
        batch_index = tf.reshape(batch_index, [batch_size, 1])

        scores = tf.reshape(scores, [batch_size, beam_size])
        best_index = tf.argmax(scores, axis=1)
        best_index = tf.reshape(best_index, [batch_size, 1])
        best_index = tf.cast(best_index, dtype=tf.int32)
        best_index = tf.concat([batch_index, best_index], axis=1)

        hyps = tf.reshape(hyps, [batch_size, beam_size, -1])
        best_hyps = tf.gather_nd(hyps, best_index)
        return best_hyps
            

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        if model_type == "":
            return
        if model_type == "mpc":
            logging.info("loading from pretrained mpc model")
            self.x_net = pretrained_model.x_net
            self.transformer.encoder = pretrained_model.encoder
        elif model_type == "SpeechTransformer":
            logging.info("loading from pretrained SpeechTransformer model")
            self.x_net = pretrained_model.x_net
            self.y_net = pretrained_model.y_net
            self.transformer = pretrained_model.transformer
            self.final_layer = pretrained_model.final_layer
        else:
            raise ValueError("NOT SUPPORTED")



