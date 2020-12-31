from typing import Callable

from torch.optim import Adam, lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy, Precision, Recall
from torch import nn


class ImageClassificationLightningModule(pl.LightningModule):
    lr = 1e-3

    def __init__(self, backbone, get_x_method: Callable, num_classes: int):
        super().__init__()

        self.criterion = nn.NLLLoss()

        self.metrics = {
            "accuracy": Accuracy(),
            "recall": Recall(num_classes=num_classes),
            "precision": Precision(num_classes=num_classes),
        }

        self.backbone = backbone

        self.num_classes = num_classes

        self.get_x_method = get_x_method
        
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.activation(self.backbone(x))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x = self.get_x_method(batch)
        y = batch["label"]

        logits = self.forward(x)

        loss = self.criterion(logits, y)

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_x_method(batch)
        y = batch["label"]

        logits = self.forward(x)

        loss = self.criterion(logits, y)

        metrics_dict = {
            f"val/{name}": metric.to(self.device)(logits, y)
            for name, metric in self.metrics.items()
        }

        self.log_dict({**{"val/loss": loss}, **metrics_dict})

        return loss
