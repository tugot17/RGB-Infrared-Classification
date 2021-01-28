from typing import Callable
from base_model import ImageClassificationLightningModule
from torch import nn, cat
import torch.nn.functional as F
import sys
from os.path import join, relpath, dirname

upper_dir = join(dirname(relpath(__file__)), "..")
sys.path.append(upper_dir)


class DenseLightningModuleWithTwoBackbones(ImageClassificationLightningModule):
    def __init__(
        self,
        backbone_rgb: nn.Module,
        backbone_infrared: nn.Module,
        get_x_method: Callable,
        num_classes: int,
    ):
        super().__init__(backbone_rgb, get_x_method, num_classes)

        num_output_nodes = backbone_rgb.classifier.in_features
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * num_output_nodes, num_classes),
            self.activation,
        )

        self.backbone_rgb = backbone_rgb
        self.backbone_infrared = backbone_infrared

    def forward(self, x):
        rgb_batch, infrared_batch = x

        rgb_latent_space = self.backbone_rgb.features(rgb_batch)
        nir_latent_space = self.backbone_infrared.features(infrared_batch)

        rgb_latent_space = F.adaptive_avg_pool2d(rgb_latent_space, (1, 1))
        rgb_latent_space = rgb_latent_space.view(rgb_latent_space.size(0), -1)

        nir_latent_space = F.adaptive_avg_pool2d(nir_latent_space, (1, 1))

        nir_latent_space = nir_latent_space.view(nir_latent_space.size(0), -1)

        combined_latent_representation = cat(
            (
                rgb_latent_space.view(rgb_latent_space.size(0), -1),
                nir_latent_space.view(nir_latent_space.size(0), -1),
            ),
            dim=1,
        )

        return self.classifier(combined_latent_representation)
