# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Wubo Li
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
# Only support eager mode and TF>=2.0.0
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes
""" a implementation of multi-task model with attention and ctc loss """

import io
import tensorflow as tf
from tensorflow.keras.layers import Dense
from .base import BaseModel
from ..loss import CTCLoss
from ..metrics import CTCAccuracy
from .speech_transformer import SpeechTransformer
from .speech_conformer import SpeechConformer
from ..utils.hparam import register_and_parse_hparams

class MtlTransformerCtc(BaseModel):
    """ In speech recognition, adding CTC loss to Attention-based seq-to-seq model is known to
    help convergence. It usually gives better results than using attention alone.
    """

    SUPPORTED_MODEL = {
        "speech_transformer": SpeechTransformer,
        "speech_conformer": SpeechConformer,
    }

    default_config = {
        "model": "speech_transformer",
        "model_config":  {"return_encoder_output": True},
        "mtl_weight": 0.5
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.num_class = data_descriptions.num_class + 1
        self.sos = self.num_class - 1
        self.eos = self.num_class - 1

        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.loss_function = CTCLoss(blank_index=-1)
        self.metric = CTCAccuracy()
        self.model = self.SUPPORTED_MODEL[self.hparams.model](
            data_descriptions, self.hparams.model_config
        )
        self.decoder = Dense(self.num_class)


    def call(self, samples, training=None):
        """ call function in keras layers """
        attention_logits, encoder_output = self.model(samples, training=training)
        ctc_logits = self.decoder(encoder_output, training=training)
        return attention_logits, ctc_logits

    def get_loss(self, outputs, samples, training=None):
        """ get loss used for training """
        attention_logits, ctc_logits = outputs
        logit_length = self.compute_logit_length(samples)
        extra_loss = self.loss_function(ctc_logits, samples, logit_length)
        self.metric(ctc_logits, samples, logit_length)

        main_loss, metrics = self.model.get_loss(attention_logits, samples, training=training)
        mtl_weight = self.hparams.mtl_weight
        loss = mtl_weight * main_loss + (1.0 - mtl_weight) * extra_loss
        metrics[self.metric.name] = self.metric.result()
        return loss, metrics

    def compute_logit_length(self, samples):
        """ compute the logit length """
        return self.model.compute_logit_length(samples)

    def reset_metrics(self):
        """ reset the metrics """
        self.metric.reset_states()
        self.model.reset_metrics()

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        """ A more general-purpose interface for pretrained model restoration

	    Args:
	        pretrained_model: checkpoint path of mpc model
	        model_type: the type of pretrained model to restore
	    """
        self.model.restore_from_pretrained_model(pretrained_model, model_type)

    def decode(self, samples, hparams, lm_model=None):
        """
        Initialization of the model for decoding,
        decoder is called here to create predictions

        Args:
            samples: the data source to be decoded
            hparams: decoding configs are included here
            lm_model: lm model
        Returns::

            predictions: the corresponding decoding results
        """
        predictions = self.model.beam_search(samples, hparams, lm_model)
        return predictions

