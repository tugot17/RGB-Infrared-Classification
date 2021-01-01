import sys
from os.path import abspath, relpath, dirname, join

experiment_utils_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "experiments_utils")
)
sys.path.append(experiment_utils_module_path)

experiment_utils_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "image_classification")
)
sys.path.append(experiment_utils_module_path)

from models.resnet_model import ResnetImageClassificationLightningModule
from models.densenet_model import DensenetImageClassificationLightningModule
from models.efficientnet_model import EfficientnetImageClassificationLightningModule

from run_experiments import run_experiments_for_models
from models_configurations import (
    resnet_configurations,
    densnet_configurations,
    efficientnet_configurations,
)
from seeds import seeds
from epfl_utils import PROJECT_NAME, datamodule

store_preds_path = abspath(
    join(dirname(relpath(__file__)), "..", "results", "rgb_infrared_combined")
)

project_name = PROJECT_NAME
experiment_type = "rgb_only"

num_classes = 9
in_channels = 4


def get_x_method(batch):
    return batch["combined_img"]


def run_rgb_infrared_combined_experiments():
    reset_init_fun = ResnetImageClassificationLightningModule

    run_experiments_for_models(
        reset_init_fun,
        resnet_configurations,
        datamodule,
        seeds,
        get_x_method,
        num_classes,
        in_channels,
        store_preds_path,
        project_name,
        experiment_type,
    )

    densnet_init_fun = DensenetImageClassificationLightningModule

    run_experiments_for_models(
        densnet_init_fun,
        densnet_configurations,
        datamodule,
        seeds,
        get_x_method,
        num_classes,
        in_channels,
        store_preds_path,
        project_name,
        experiment_type,
    )

    efficientnet_init_fun = EfficientnetImageClassificationLightningModule

    run_experiments_for_models(
        efficientnet_init_fun,
        efficientnet_configurations,
        datamodule,
        seeds,
        get_x_method,
        num_classes,
        in_channels,
        store_preds_path,
        project_name,
        experiment_type,
    )


if __name__ == "__main__":
    run_rgb_infrared_combined_experiments()
