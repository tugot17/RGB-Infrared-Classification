import sys
from os.path import abspath, relpath, dirname, join

upper_dir = abspath(join(dirname(relpath(__file__)), ".."))

sys.path.append(upper_dir)

from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from base_model import ImageClassificationLightningModule
from typing import Callable


class EfficientnetImageClassificationLightningModule(
    ImageClassificationLightningModule
):
    lr = 1e-3

    def __init__(
        self, backbone_fun, kwargs, get_x_method: Callable, num_classes, in_channels=3
    ):

        kwargs["num_classes"] = num_classes
        backbone = self._initialize_backbone(backbone_fun, kwargs, True, in_channels)

        super().__init__(backbone, get_x_method, num_classes)

    def _initialize_backbone(
        self, backbone_fun, kwargs, first_layer_pretrained=False, in_channels=3
    ):

        try:
            backbone = backbone_fun(**kwargs)
        except Exception as e:
            kwargs["override_params"] = {"num_classes": kwargs["num_classes"]}
            del kwargs["num_classes"]
            backbone = backbone_fun(**kwargs)

        if in_channels != 3:
            weight = backbone._conv_stem.weight.data.clone()
            bias = None
            in_channels = in_channels

            backbone._conv_stem = Conv2dStaticSamePadding(
                in_channels, 32, (3, 3), image_size=224
            )

            if first_layer_pretrained:
                backbone._conv_stem.weight.data[:, :3] = weight
                backbone._conv_stem.weight.data[:, 3] = weight.mean(dim=1)
                backbone._conv_stem.bias = bias

        return backbone
