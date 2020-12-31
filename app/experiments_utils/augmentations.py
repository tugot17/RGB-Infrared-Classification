import albumentations as A
from albumentations.pytorch import ToTensorV2

train_common_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(),
        A.Rotate(limit=5, p=0.9),
    ],
    additional_targets={"infrared_img": "image"},
)

test_common_transform = A.Compose(
    [
        A.Resize(224, 224),
    ],
    additional_targets={"infrared_img": "image"},
)

rgb_transform = A.Compose(
    [
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
)

infrared_transform = A.Compose(
    [
        A.Normalize(
            mean=[0.5],
            std=[0.25],
        ),
        ToTensorV2(),
    ]
)
