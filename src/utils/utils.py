import os
import yaml

import torch


def use_best_hyperparams(args, dataset_name):
    best_params_file_path = "best_hyperparams.yml"
    with open(best_params_file_path, "r") as file:
        hyperparams = yaml.safe_load(file)

    for name, value in hyperparams[dataset_name].items():
        if hasattr(args, name):
            setattr(args, name, value)
        else:
            raise ValueError(f"Trying to set non existing parameter: {name}")

    return args


def get_available_accelerator():
    if torch.cuda.is_available():
        return "gpu"
    # Keep the following commented out as some of the operations
    # we use are currently not supported by mps
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     return "mps"
    else:
        return "cpu"
