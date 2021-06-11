from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, CustomSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from tensorboardX import SummaryWriter

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
  parser = argparse.ArgumentParser()

  # Datset Options
  parser.add_argument("--data_root", type=str, default='./datasets/data',
                      help="path to Dataset")
  parser.add_argument("--dataset", type=str, default='custom',
                      choices=['voc', 'cityscapes', 'custom'], help='Name of dataset')
  parser.add_argument("--num_classes", type=int, default=None,
                      help="num classes (default: None)")

  # Deeplab Options
  parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                      choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                               'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                               'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
  parser.add_argument("--separable_conv", action='store_true', default=False,
                      help="apply separable conv to decoder and aspp")
  parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

  parser.add_argument("--ckpt", default=None, type=str,
                      help="restore from checkpoint")
  parser.add_argument("--continue_training", action='store_true', default=False)

  parser.add_argument("--gpu_id", type=str, default='0',
                      help="GPU ID")
  parser.add_argument("--random_seed", type=int, default=1,
                      help="random seed (default: 1)")
  parser.add_argument("--total_itrs", type=int, default=30e3,
                      help="epoch number (default: 30k)")
  parser.add_argument("--lr", type=float, default=0.01,
                      help="learning rate (default: 0.01)")
  parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                      help="learning rate scheduler policy")
  parser.add_argument("--weight_decay", type=float, default=1e-4,
                      help='weight decay (default: 1e-4)')
  parser.add_argument("--step_size", type=int, default=10000)
  
  # Save options
  parser.add_argument("--save_dir", type=str, default="checkpoints",
                      help="path to save models")
  return parser


def get_dataset(opts):
  """ Dataset And Augmentation
  """
  if opts.dataset == 'voc':
    train_transform = et.ExtCompose([
      #et.ExtResize(size=opts.crop_size),
      et.ExtRandomScale((0.5, 2.0)),
      et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
      et.ExtRandomHorizontalFlip(),
      et.ExtToTensor(),
      et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
      val_transform = et.ExtCompose([
        et.ExtResize(opts.crop_size),
        et.ExtCenterCrop(opts.crop_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
      ])
    else:
      val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
      ])
    train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                image_set='train', download=opts.download, transform=train_transform)
    val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                              image_set='val', download=False, transform=val_transform)

  elif opts.dataset == 'cityscapes':
    train_transform = et.ExtCompose([
      #et.ExtResize( 512 ),
      et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
      et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
      et.ExtRandomHorizontalFlip(),
      et.ExtToTensor(),
      et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
      #et.ExtResize( 512 ),
      et.ExtToTensor(),
      et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    ])

    train_dst = Cityscapes(root=opts.data_root,
                           split='train', transform=train_transform)
    val_dst = Cityscapes(root=opts.data_root,
                         split='val', transform=val_transform)

  elif opts.dataset == 'custom':
    train_transform = et.ExtCompose([
      et.ExtToTensor(),
      et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
      et.ExtToTensor(),
      et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    ])

    train_dst = CustomSegmentation(root=opts.data_root,
                                   split='train', transform=train_transform)
    val_dst = CustomSegmentation(root=opts.data_root,
                                 split='val', transform=val_transform)
  return train_dst, val_dst


def main():
  opts = get_argparser().parse_args()
  if opts.dataset.lower() == 'voc':
    opts.num_classes = 21
  elif opts.dataset.lower() == 'cityscapes':
    opts.num_classes = 19
  elif opts.dataset.lower() == 'custom':
    opts.num_classes = 3

  os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("Device: %s" % device)

  # Setup random seed
  torch.manual_seed(opts.random_seed)
  np.random.seed(opts.random_seed)
  random.seed(opts.random_seed)

  # Setup dataloader
  if opts.dataset=='voc' and not opts.crop_val:
    opts.val_batch_size = 1

  train_dst, val_dst = get_dataset(opts)
  train_loader = data.DataLoader(
    train_dst, batch_size=1, shuffle=True, num_workers=1)


  # Set up model
  model_map = {
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
  }

  model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
  if opts.separable_conv and 'plus' in opts.model:
    network.convert_to_separable_conv(model.classifier)
  utils.set_bn_momentum(model.backbone, momentum=0.01)


  # Set up optimizer
  optimizer = torch.optim.SGD(params=[
    {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
    {'params': model.classifier.parameters(), 'lr': opts.lr},
  ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
  #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
  #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
  if opts.lr_policy=='poly':
    scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
  elif opts.lr_policy=='step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

  def save_ckpt(path):
    """ save current model
    """
    torch.save({
      "cur_itrs": cur_itrs,
      "model_state": model.module.state_dict(),
      "optimizer_state": optimizer.state_dict(),
      "scheduler_state": scheduler.state_dict(),
      "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)

  utils.mkdir('checkpoints')
  # Restore
  best_score = 0.0
  cur_itrs = 0
  cur_epochs = 0
  if opts.ckpt is not None and os.path.isfile(opts.ckpt):
    # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    if opts.continue_training:
      optimizer.load_state_dict(checkpoint["optimizer_state"])
      scheduler.load_state_dict(checkpoint["scheduler_state"])
      cur_itrs = checkpoint["cur_itrs"]
      best_score = checkpoint['best_score']
      print("Training state restored from %s" % opts.ckpt)
    print("Model restored from %s" % opts.ckpt)
    del checkpoint  # free memory
  else:
    print("[!] Retrain")
    model = nn.DataParallel(model)
    model.to(device)

  #==========  Loop   ==========#

  # Switch the model to eval mode
  model.eval()

  for (images, labels) in train_loader:
    images = images.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.long)
    
    outputs = model(images)
    output = outputs.detach().cpu().numpy()[0]
    preds = outputs.detach().max(dim=1)[1].cpu().numpy()

    image = images[0].detach().cpu().numpy()
    pred = preds[0]

    print(f"input size: {images.size()}")
    #print(f"image values: {image.flatten()[100000:100300]}")
    #print(f"output values: {output.flatten()[100000:100300]}")
    #print(f"pred values: {pred.flatten()[100000:100300]}")
  
    break

  # An example input you would normally provide to your model's forward() method.
  example = images

  
  # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
  traced_script_module = torch.jit.trace(model, example)
  
  # Save the TorchScript model
  traced_script_module.save(os.path.join(opts.save_dir, "torchscript_model.pt"))

if __name__ == '__main__':
  main()
