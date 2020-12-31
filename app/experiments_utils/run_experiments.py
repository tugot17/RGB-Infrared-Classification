from model_with_two_separate_backbones import (
    ImageSegmentationLightningModuleWithTwoBackbones,
)
from model import ImageSegmentationLightningModule
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from predict import predict

import sys
from os.path import abspath, relpath, dirname, join

image_segmentation_module_path = abspath(
    join(dirname(relpath(__file__)), "..", "image_segmentation")
)
sys.path.append(image_segmentation_module_path)


MAX_EPOCHS = 1


def run_experiment(lightning_model, datamodule, seed, get_x_method, logger):
    seed_everything(seed)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        gpus=1,
        deterministic=True,
        accumulate_grad_batches=4,
        checkpoint_callback=False,
        callbacks=[EarlyStopping(monitor="val/loss")],
        logger=logger,
    )

    trainer.fit(lightning_model, datamodule)
    lightning_model.eval()

    preds = predict(lightning_model, datamodule.val_dataloader(), "cuda", get_x_method)

    lightning_model = lightning_model.cpu()

    wandb.finish()

    return preds


def run_experiments_for_models(
    models_configurations,
    dm,
    seeds,
    get_x_method,
    store_preds_path,
    project_name,
    experiment_type,
):
    for backbone_fun, kwargs, model_name in models_configurations:

        experiment_type_plus_model_name = f"{model_name}_{experiment_type}"

        predictions_for_seeds = []

        for seed in seeds:
            logger = WandbLogger(
                name=experiment_type_plus_model_name, project=project_name
            )

            backbone = backbone_fun(**kwargs)
            model = ImageSegmentationLightningModule(backbone, get_x_method)

            preds = run_experiment(model, dm, seed, get_x_method, logger)
            predictions_for_seeds.append(preds)

        save_preds_path = join(store_preds_path, f"{model_name}.pt")

        torch.save(
            torch.stack(predictions_for_seeds).mean(dim=0).type(torch.float16),
            save_preds_path,
        )


def run_experiments_for_models_with_two_separate_backbones(
    models_configurations,
    dm,
    seeds,
    get_x_method,
    store_preds_path,
    project_name,
    experiment_type,
):
    for backbone_fun, kwargs, model_name in models_configurations:

        experiment_type_plus_model_name = f"{model_name}_{experiment_type}"

        predictions_for_seeds = []

        for seed in seeds:
            logger = WandbLogger(
                name=experiment_type_plus_model_name, project=project_name
            )

            backbone_rgb = backbone_fun(**kwargs)
            backbone_infrared = backbone_fun(**kwargs)

            model = ImageSegmentationLightningModuleWithTwoBackbones(
                backbone_rgb, backbone_infrared, get_x_method
            )

            preds = run_experiment(model, dm, seed, get_x_method, logger)
            predictions_for_seeds.append(preds)

        save_preds_path = join(store_preds_path, f"{model_name}.pt")

        torch.save(
            torch.stack(predictions_for_seeds).mean(dim=0).type(torch.float16),
            save_preds_path,
        )
