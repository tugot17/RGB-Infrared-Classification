from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from torch import nn
from base_model import ImageClassificationLightningModule
from typing import Callable
import sys
from os.path import join, relpath, dirname


class EfficientnetImageClassificationLightningModule(
    ImageClassificationLightningModule
):
    lr = 1e-3

    def __init__(
        self,
        backbone_fun,
        kwargs,
        get_x_method: Callable,
        num_classes,
        in_channels=3,
    ):

        kwargs["num_classes"] = num_classes
        kwargs["in_channels"] = in_channels
        backbone = self._initialize_backbone(backbone_fun, kwargs)

        super().__init__(backbone, get_x_method, num_classes)

    def _initialize_backbone(self, backbone_fun, kwargs, first_layer_pretrained=False):

        if kwargs["in_channels"] != 3 and first_layer_pretrained:
            backbone = backbone_fun(**kwargs)

            weight = backbone._conv_stem.weight.data.clone()
            bias = None
            in_channels = 4

            backbone._conv_stem = Conv2dStaticSamePadding(
                in_channels, 32, (3, 3), image_size=224
            )

            backbone._conv_stem.weight.data[:, :3] = weight
            backbone._conv_stem.weight.data[:, 3] = weight.mean(dim=1)
            backbone._conv_stem.bias = bias
        else:
            backbone = backbone_fun(**kwargs)

        return backbone
