from run_experiments import run_experiments_for_models
from models_configurations import models_configurations
from seeds import seeds
from epfl_utils import PROJECT_NAME, datamodule
import sys
from os.path import abspath, relpath, dirname, join
from copy import deepcopy

experiment_utils_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "experiments_utils")
)
sys.path.append(experiment_utils_module_path)


store_preds_path = abspath(
    join(dirname(relpath(__file__)), "..", "results", "rgb_infrared_combined")
)


def get_x_method(batch):
    return batch["combined_img"]


project_name = PROJECT_NAME
experiment_type = "rgb_infrared_combined"


def change_in_channels_to_4(inputs):
    backbone_fun, kwargs, experiment_name = inputs
    kwargs["in_channels"] = 4
    return backbone_fun, kwargs, experiment_name


updated_models_configurations = deepcopy(models_configurations)
updated_models_configurations = list(
    map(change_in_channels_to_4, updated_models_configurations)
)


def run_rgb_infrared_combined_experiments():
    run_experiments_for_models(
        updated_models_configurations,
        datamodule,
        seeds,
        get_x_method,
        store_preds_path,
        project_name,
        experiment_type,
    )


if __name__ == "__main__":
    run_rgb_infrared_combined_experiments()
