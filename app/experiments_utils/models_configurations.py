import torchvision.models as models
from efficientnet_pytorch import EfficientNet

# from .classes import classes

resnet_backbone_models = [
    (models.resnet50, {"pretrained": True}, "resnet_50_pretrained"),
    (models.resnet50, {"pretrained": False}, "resnet_50_from_scratch"),
    (models.resnet18, {"pretrained": True}, "resnet_18_pretrained"),
    (models.resnet18, {"pretrained": False}, "resnet_18_from_scratch"),
]

densnet_backbone_models = [
    (models.densenet121, {"pretrained": True}, "densenet_121_pretrained"),
    (models.densenet121, {"pretrained": False}, "densenet_121_from_scratch"),
    (models.densenet169, {"pretrained": True}, "densenet_169_pretrained"),
    (models.densenet169, {"pretrained": False}, "densenet_169_from_scratch"),
]

efficientnet_backbone_models = [
    (
        EfficientNet.from_pretrained,
        {"model_name": "efficientnet-b0"},
        "efficientnet_b0_pretrained",
    ),
    (
        EfficientNet.from_name,
        {"model_name": "efficientnet-b0"},
        "efficientnet_b0_from_scratch",
    ),
    (
        EfficientNet.from_pretrained,
        {"model_name": "efficientnet-b2"},
        "efficientnet_b2_pretrained",
    ),
    (
        EfficientNet.from_name,
        {
            "model_name": "efficientnet-b2",
        },
        "efficientnet_b2_from_scratch",
    ),
]
