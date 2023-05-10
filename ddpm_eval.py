
import os
import random
import sys

import numpy as np
import torch
import yaml

from evaluation.eval_cdm import run_inference as run_inference_only_cdm
from evaluation.evaluate_lidc_sampling_speed import eval_lidc_sampling_speed
from evaluation.evaluate_lidc_uncertainty import eval_lidc_uncertainty


def set_seeds(seed: int):
    """Function that sets all relevant seeds
    :param seed: Seed to use
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv):
    set_seeds(0)

    params_file = "params_eval.yml"
    if len(argv) == 2 and "params_" in argv[1]:
        params_file = argv[1]
        print(f"Overriding params file with {params_file}...")

    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    params["params_file"] = params_file
    if 'cityscapes' in params['dataset_file']:
        run_inference_only_cdm(params)
    else:
        raise ValueError(f"Only cityscapes is supported in this branch dataset instead got {params['dataset_file']}")


if __name__ == "__main__":
    main(sys.argv)

