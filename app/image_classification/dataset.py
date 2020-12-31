from typing import List, Dict
import pandas as pd
import cv2
from torch.utils.data import Dataset
import numpy as np
from albumentations.core.composition import Compose
import torch


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        df_json_path: str,
        label_map: Dict,
        common_transform: Compose,
        rgb_transform: Compose,
        infrared_transform: Compose,
    ):
        df = pd.read_json(df_json_path)

        self.rgb_img_fps = df.rgb.tolist()
        self.infrared_img_fps = df.infrared.tolist()
        self.labels = df.label.tolist()

        self.common_transform = common_transform
        self.rgb_transform = rgb_transform
        self.infrared_transform = infrared_transform

        self.label_map = label_map

    def __getitem__(self, idx):
        original_rgb_img = cv2.imread(self.rgb_img_fps[idx], -1)
        original_rgb_img = cv2.cvtColor(original_rgb_img, cv2.COLOR_BGR2RGB)

        original_infrared_img = cv2.imread(self.infrared_img_fps[idx], -1)
        original_infrared_img = np.expand_dims(original_infrared_img, axis=-1)

        transformed = self.common_transform(
            image=original_rgb_img, infrared_img=original_infrared_img
        )
        rgb_img = transformed["image"]
        infrared_img = transformed["infrared_img"]

        transformed = self.rgb_transform(image=rgb_img)
        rgb_img = transformed["image"]

        transformed = self.infrared_transform(image=infrared_img)
        infrared_img = transformed["image"]

        combined_img = torch.cat((rgb_img, infrared_img), 0)
        infrared_img = infrared_img.repeat(3, 1, 1)

        original_label = self.labels[idx]
        label = self.label_map[original_label]

        item_dict = {
            "original_rgb_img": original_rgb_img,
            "original_infrared_img": original_infrared_img,
            "rgb_img": rgb_img,
            "infrared_img": infrared_img,
            "combined_img": combined_img,
            "original_label": original_label,
            "label": label,
        }

        return item_dict

    def __len__(self):
        return len(self.rgb_img_fps)
