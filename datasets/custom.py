import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class CustomSegmentation(data.Dataset):
    """Custom Segmentation Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    CustomSegmentationClass = namedtuple('CustomSegmentationClass', ['name', 'id', 'color'])
    classes = [
        CustomSegmentationClass('background', 0, (0, 0, 0)),
        CustomSegmentationClass('text', 1, (255, 0, 255)),
        CustomSegmentationClass('illustration', 2, (255, 255, 0))
    ]

    train_id_to_color = [c.color for c in classes]
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.id for c in classes])

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.img_dir = os.path.join(self.root, split, 'images')

        self.mask_dir = os.path.join(self.root, split, 'masks')
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.img_dir) or not os.path.isdir(self.mask_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" are inside the "root" directory')

        for file_name in os.listdir(self.img_dir):
            self.images.append(os.path.join(self.img_dir, file_name))
            base_name, _ = os.path.splitext(file_name)
            self.targets.append(os.path.join(self.mask_dir, base_name + '.png'))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        if self.transform:
            image, target = self.transform(image, target)
        
        print(f"Image: {self.images[index]}")
        print(f"Mask:  {self.targets[index]}")
        print(f"Non zeros: {torch.count_nonzero(target)}")
        print(f"bincount: {torch.bincount(torch.flatten(target))}")

        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)