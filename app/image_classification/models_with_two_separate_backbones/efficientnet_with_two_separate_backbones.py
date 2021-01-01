from torch import nn, cat
from base_model import ImageClassificationLightningModule
from typing import Callable
import sys
from os.path import join, relpath, dirname

upper_dir = join(dirname(relpath(__file__)), "..")
sys.path.append(upper_dir)


class EfficientNetLightningModuleWithTwoBackbones(ImageClassificationLightningModule):
    def __init__(
        self,
        backbone_rgb: nn.Module,
        backbone_infrared: nn.Module,
        get_x_method: Callable,
        num_classes: int,
    ):
        super().__init__(backbone_rgb, get_x_method, num_classes)

        num_output_nodes = backbone_rgb._fc.out_features
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * num_output_nodes, num_classes),
            self.activation,
        )

        self.backbone_rgb = backbone_rgb
        self.backbone_infrared = backbone_infrared

    def forward(self, x):
        rgb_batch, infrared_batch = x

        rgb_latent_space = self.backbone_rgb(rgb_batch)
        nir_latent_space = self.backbone_infrared(infrared_batch)

        combined_latent_representation = cat(
            (rgb_latent_space, nir_latent_space), dim=1
        )

        return self.classifier(combined_latent_representation)
