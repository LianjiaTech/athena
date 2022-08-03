from absl import logging
import numpy as np
import os
from athena_wakeup import SpeechWakeupFramewiseDatasetKaldiIOBuilder
#np.set_printoptions(threshold=np.inf, suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test():
    dataset_builder = SpeechWakeupFramewiseDatasetKaldiIOBuilder(
        {
            "data_dir":"examples/kws/xhxh/data/test",
            "left_context":10,
            "right_context":10,
            "feat_dim":63
        }
    )
    dataset = dataset_builder.as_dataset(batch_size=32)
    for batch, item in enumerate(dataset):
        logging.info(item["input"].shape)
        logging.info(item["output"])
        logging.info(item["input_length"])
        logging.info(item["output_length"])
        if batch == 2: break

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    test()
