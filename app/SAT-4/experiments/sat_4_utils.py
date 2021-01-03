import sys
from os.path import abspath, relpath, dirname, join

image_segmentation_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "image_classification")
)
sys.path.append(image_segmentation_module_path)

experiment_utils_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "experiments_utils")
)
sys.path.append(experiment_utils_module_path)

from augmentations import (
    train_common_transform,
    test_common_transform,
    rgb_transform,
    infrared_transform,
)
from dataset import ImageClassificationDataset
from datamodule import ImageClassificationDatamodule


PROJECT_NAME = "EPFL_RGB_NIR"

train_df_json_path = abspath(join(dirname(relpath(__file__)), "..", "train.json"))
val_df_json_path = abspath(join(dirname(relpath(__file__)), "..", "val.json"))

train_common_transform = train_common_transform
test_common_transform = test_common_transform

rgb_transform = rgb_transform
infrared_transform = infrared_transform

label_map = {"barren land": 0, "trees": 1, "grassland": 2, "none": 3}


train_set = ImageClassificationDataset(
    train_df_json_path,
    label_map,
    train_common_transform,
    rgb_transform,
    infrared_transform,
)

val_set = ImageClassificationDataset(
    val_df_json_path,
    label_map,
    test_common_transform,
    rgb_transform,
    infrared_transform,
)

BATCH_SIZE = 8

datamodule = ImageClassificationDatamodule(BATCH_SIZE, train_set, val_set)
