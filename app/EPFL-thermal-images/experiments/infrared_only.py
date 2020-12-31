from epfl_utils import PROJECT_NAME, datamodule
import sys
from os.path import abspath, relpath, dirname, join

experiment_utils_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "experiments_utils")
)
sys.path.append(experiment_utils_module_path)

from seeds import seeds
from models_configurations import models_configurations
from run_experiments import run_experiments_for_models


store_preds_path = abspath(
    join(dirname(relpath(__file__)), "..", "results", "infrared_only")
)


def get_x_method(batch):
    return batch["infrared_img"]


project_name = PROJECT_NAME
experiment_type = "infrared_only"


def run_infrared_only_experiments():
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
    run_infrared_only_experiments()
