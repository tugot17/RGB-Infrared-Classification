from copy import deepcopy
from epfl_utils import PROJECT_NAME, datamodule
from seeds import seeds
from run_experiments import run_experiments_for_models_with_two_separate_backbones
from models_with_two_separate_backbones.resnet_with_two_separate_backbones import (
    ResnetLightningModuleWithTwoBackbones, )
from models_with_two_separate_backbones.efficientnet_with_two_separate_backbones import (
    EfficientNetLightningModuleWithTwoBackbones, )
from models_with_two_separate_backbones.densnet_with_two_separate_backbones import (
    DenseLightningModuleWithTwoBackbones, )
from models_configurations import (
    densnet_configurations,
    efficientnet_configurations,
    resnet_configurations,
)
import sys
from os.path import abspath, relpath, dirname, join

experiment_utils_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "experiments_utils")
)
sys.path.append(experiment_utils_module_path)


image_classification_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "image_classification")
)
sys.path.append(image_classification_module_path)


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


num_classes = 9


def run_rgb_infrared_as_two_separate_inputs_experiments():
    densenet_init_fun = DenseLightningModuleWithTwoBackbones

    run_experiments_for_models_with_two_separate_backbones(
        densenet_init_fun,
        densnet_configurations,
        datamodule,
        seeds,
        get_x_method,
        num_classes,
        store_preds_path,
        project_name,
        experiment_type,
    )

    efficientnet_init_fun = EfficientNetLightningModuleWithTwoBackbones

    run_experiments_for_models_with_two_separate_backbones(
        efficientnet_init_fun,
        efficientnet_configurations,
        datamodule,
        seeds,
        get_x_method,
        num_classes,
        store_preds_path,
        project_name,
        experiment_type,
    )

    resnet_init_fun = ResnetLightningModuleWithTwoBackbones

    run_experiments_for_models_with_two_separate_backbones(
        resnet_init_fun,
        resnet_configurations,
        datamodule,
        seeds,
        get_x_method,
        num_classes,
        store_preds_path,
        project_name,
        experiment_type,
    )


if __name__ == "__main__":
    run_rgb_infrared_as_two_separate_inputs_experiments()
