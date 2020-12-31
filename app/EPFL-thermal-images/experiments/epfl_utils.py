import sys
from os.path import abspath, relpath, dirname, join

image_segmentation_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "..", "image_classification")
)
sys.path.append(image_segmentation_module_path)

from datamodule import ImageClassificationDatamodule
from dataset import ImageClassificationDataset

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


PROJECT_NAME = "EPFL_RGB_NIR"

train_df_json_path = abspath(join(dirname(relpath(__file__)), "..", "train.json"))
test_df_json_path = abspath(join(dirname(relpath(__file__)), "..", "test.json"))

train_common_transform = train_common_transform
test_common_transform = test_common_transform

rgb_transform = rgb_transform
infrared_transform = infrared_transform

label_map = {
    "forest": 0,
    "indoor": 1,
    "street": 2,
    "field": 3,
    "country": 4,
    "urban": 5,
    "mountain": 6,
    "oldbuilding": 7,
    "water": 8,
}


train_set = ImageClassificationDataset(
    train_df_json_path,
    label_map,
    train_common_transform,
    rgb_transform,
    infrared_transform,
)

val_set = ImageClassificationDataset(
    test_df_json_path,
    label_map,
    test_common_transform,
    rgb_transform,
    infrared_transform,
)

BATCH_SIZE = 8

datamodule = ImageClassificationDatamodule(BATCH_SIZE, train_set, val_set)
