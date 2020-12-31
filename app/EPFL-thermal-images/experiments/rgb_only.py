from run_experiments import run_experiments_for_models
from models_configurations import models_configurations
from seeds import seeds
from epfl_utils import PROJECT_NAME, datamodule
import sys
from os.path import abspath, relpath, dirname, join

experiment_utils_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "experiments_utils")
)
sys.path.append(experiment_utils_module_path)


store_preds_path = abspath(
    join(dirname(relpath(__file__)), "..", "results", "rgb_only")
)


def get_x_method(batch):
    return batch["rgb_img"]


project_name = PROJECT_NAME
experiment_type = "rgb_only"


def run_rgb_only_experiments():
    run_experiments_for_models(
        models_configurations,
        datamodule,
        seeds,
        get_x_method,
        store_preds_path,
        project_name,
        experiment_type,
    )


if __name__ == "__main__":
    run_rgb_only_experiments()
