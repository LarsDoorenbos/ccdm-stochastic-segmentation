
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

    """Function that sets all relevant seeds (by Claudio)
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

    if 'lidc_sampling_speed' in params['dataset_file']:
        params['dataset_file'] = "datasets.lidc"
        eval_lidc_sampling_speed(params)
        return
    if 'lidc' in params['dataset_file']:
        eval_lidc_uncertainty(params)
        return
    
    if params['run_cdm_eval_v2']:
        run_inference_only_cdm(params)
        return


if __name__ == "__main__":
    main(sys.argv)

