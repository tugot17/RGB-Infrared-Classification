import sys
from os.path import abspath, relpath, dirname, join

upper_dir = abspath(join(dirname(relpath(__file__)), ".."))

sys.path.append(upper_dir)

from torch import nn
from base_model import ImageClassificationLightningModule
from typing import Callable


class ResnetImageClassificationLightningModule(ImageClassificationLightningModule):
    lr = 1e-3

    def __init__(
        self,
        backbone_fun,
        kwargs,
        get_x_method: Callable,
        num_classes: int,
        in_channels=3,
    ):

        backbone = self._initialize_backbone(
            backbone_fun, kwargs, num_classes, in_channels, True
        )

        super().__init__(backbone, get_x_method, num_classes)

    def _initialize_backbone(
        self,
        backbone_fun,
        kwargs,
        num_classes,
        in_channels=3,
        first_layer_pretrained=False,
    ):
        backbone = backbone_fun(**kwargs)

        if in_channels != 3:
            weight = backbone.conv1.weight.data.clone()

            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            if first_layer_pretrained:
                backbone.conv1.weight.data[:, :3] = weight
                backbone.conv1.weight.data[:, 3] = weight.mean(dim=1)

        num_ftrs = backbone.fc.in_features

        backbone.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

        return backbone
