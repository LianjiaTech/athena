# Copyright (C) ATHENA AUTHORS; Yanguang Xu
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
# pylint: disable=invalid-name
"""storage features offline"""
import sys
import json
from absl import logging
from athena.main import parse_config, SUPPORTED_DATASET_BUILDER
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    jsonfile = sys.argv[1]

    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)

    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.trainset_config)
    dataset_builder.storage_features_offline(p.trainset_config["data_csv"],p.trainset_config["data_scps_dir"])
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.devset_config)
    dataset_builder.storage_features_offline(p.devset_config["data_csv"],p.devset_config["data_scps_dir"])
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.testset_config)
    dataset_builder.storage_features_offline(p.testset_config["data_csv"],p.testset_config["data_scps_dir"])
