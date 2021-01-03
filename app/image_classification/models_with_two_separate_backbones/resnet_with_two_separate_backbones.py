import sys
from os.path import join, relpath, dirname

upper_dir = join(dirname(relpath(__file__)), "..")
sys.path.append(upper_dir)

from torch import nn, cat
from base_model import ImageClassificationLightningModule
from typing import Callable


class ResnetLightningModuleWithTwoBackbones(ImageClassificationLightningModule):
    def __init__(
        self,
        backbone_rgb: nn.Module,
        backbone_infrared: nn.Module,
        get_x_method: Callable,
        num_classes: int,
    ):
        super().__init__(backbone_rgb, get_x_method, num_classes)

        num_output_nodes = backbone_rgb.fc.in_features

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * num_output_nodes, num_classes),
            self.activation,
        )

        self.backbone_rgb = nn.Sequential(*list(backbone_rgb.children())[:-1])
        self.backbone_infrared = nn.Sequential(*list(backbone_infrared.children())[:-1])

    def forward(self, x):
        rgb_batch, infrared_batch = x
        rgb_latent_space = self.backbone_rgb(rgb_batch)
        nir_latent_space = self.backbone_infrared(infrared_batch)

        combined_latent_representation = cat(
            (
                rgb_latent_space.view(rgb_latent_space.size(0), -1),
                nir_latent_space.view(nir_latent_space.size(0), -1),
            ),
            dim=1,
        )

        return self.classifier(combined_latent_representation)
