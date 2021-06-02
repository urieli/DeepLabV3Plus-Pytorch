#!/usr/bin/env python
from setuptools import setup, find_packages
import sys

setup(
  name="DeepLabV3Plus-Pytorch",
  version="0.0.1",
  license="MIT",
  url="https://github.com/VainF/DeepLabV3Plus-Pytorch",
  description="DeepLabv3, DeepLabv3+ and pretrained weights on VOC & Cityscapes.",
  packages=find_packages(),
  install_requires=[
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    "numpy>=1.18.1,<1.19.0",
    "Pillow>=8.2.0",
    "tqdm>=4.54.1",
    "scikit-learn>=0.23.2",
    "matplotlib>=3.3.3",
    "visdom>=0.1.8.9",
    "tensorboard>=2.0.0",
    "tensorboardX>=2.2"
  ],
  zip_safe=False,
)