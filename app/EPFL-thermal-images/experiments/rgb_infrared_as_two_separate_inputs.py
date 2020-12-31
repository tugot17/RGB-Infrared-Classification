from epfl_utils import PROJECT_NAME, datamodule
import sys
from os.path import abspath, relpath, dirname, join
from copy import deepcopy

experiment_utils_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "experiments_utils")
)
sys.path.append(experiment_utils_module_path)

from seeds import seeds
from models_configurations import models_configurations
from run_experiments import run_experiments_for_models_with_two_separate_backbones


store_preds_path = abspath(
    join(
        dirname(relpath(__file__)),
        "..",
        "results",
        "rgb_infrared_as_two_separate_inputs",
    )
)


def get_x_method(batch):
    return (batch["rgb_img"], batch["infrared_img"])


project_name = PROJECT_NAME
experiment_type = "rgb_infrared_as_two_separate_inputs"


def remove_activation_from_last_layer(inputs):
    backbone_fun, kwargs, experiment_name = inputs
    del kwargs["activation"]
    return backbone_fun, kwargs, experiment_name


updated_models_configurations = deepcopy(models_configurations)
updated_models_configurations = list(
    map(remove_activation_from_last_layer, updated_models_configurations)
)


def run_rgb_infrared_as_two_separate_inputs_experiments():
    run_experiments_for_models_with_two_separate_backbones(
        updated_models_configurations,
        datamodule,
        seeds,
        get_x_method,
        store_preds_path,
        project_name,
        experiment_type,
    )


if __name__ == "__main__":
    run_rgb_infrared_as_two_separate_inputs_experiments()
