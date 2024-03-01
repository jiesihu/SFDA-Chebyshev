from datetime import datetime
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer

# Custom includes
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from networks.GAN import BoundaryDiscriminator, UncertaintyDiscriminator
import pathlib
import h5py
import numpy as np

from Dataloader import General2DSegmentation

here = osp.dirname(osp.abspath('__file__'))

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--resume', default=None, help='checkpoint path')

parser.add_argument(
    '--backbone', type=str, default='mobilenet', help='set backbone'
)
parser.add_argument(
    '--datasetS', type=str, default='Domain_source', help='Folder contains images and masks'
)
parser.add_argument(
    '--batch-size', type=int, default=16, help='batch size for training the model'
)
parser.add_argument(
    '--group-num', type=int, default=1, help='group number for group normalization'
)
parser.add_argument(
    '--max-epoch', type=int, default=200, help='max epoch'
)
parser.add_argument(
    '--stop-epoch', type=int, default=200, help='stop epoch'
)
parser.add_argument(
    '--warmup-epoch', type=int, default=-1, help='warmup epoch begin train GAN'
)

parser.add_argument(
    '--interval-validate', type=int, default=10, help='interval epoch number to valide the model'
)
parser.add_argument(
    '--lr-gen', type=float, default=1e-3, help='learning rate',
)
parser.add_argument(
    '--lr-dis', type=float, default=2.5e-5, help='learning rate',
)
parser.add_argument(
    '--lr-decrease-rate', type=float, default=0.1, help='ratio multiplied to initial lr',
)
parser.add_argument(
    '--weight-decay', type=float, default=0.0005, help='weight decay',
)
parser.add_argument(
    '--momentum', type=float, default=0.99, help='momentum',
)
parser.add_argument(
    '--data-dir',
    default='/your_data_path',
    help='data root path'
)
parser.add_argument(
    '--out-stride',
    type=int,
    default=16,
    help='out-stride of deeplabv3+',
)
parser.add_argument(
    '--sync-bn',
    type=bool,
    default=True,
    help='sync-bn in deeplabv3+',
)
parser.add_argument(
    '--freeze-bn',
    type=bool,
    default=False,
    help='freeze batch normalization of deeplabv3+',
)

args = parser.parse_args()

args.model = 'FCN8s'

now = datetime.now()
args.out = osp.join(here, 'logs', args.datasetS, now.strftime('%Y%m%d_%H%M%S.%f'))

os.makedirs(args.out)
with open(osp.join(args.out, 'config.yaml'), 'w') as f:
    yaml.safe_dump(args.__dict__, f, default_flow_style=False)


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cuda = torch.cuda.is_available()
seed_ = 1337

torch.manual_seed(seed_)
if cuda:
    torch.cuda.manual_seed(seed_)

from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
    RandGibbsNoise,
    RandAdjustContrast,
    Resize)
from monai.transforms.intensity.array import RandShiftIntensity,RandGaussianNoise
from monai.transforms.spatial.array import RandFlip, Rand2DElastic,RandAxisFlip,RandAffine


window_size = 448
train_imtrans = Compose([
                #ScaleIntensity(),
                RandSpatialCrop((window_size, window_size), random_size=False),
                RandRotate90(prob=0.2, spatial_axes=(0, 1)),
                RandAxisFlip(prob = 0.2),
                RandAffine(scale_range = 0.5 ,rotate_range = np.pi/6,prob = 0.2,padding_mode = 'zeros',mode = "bilinear"),
                RandAffine(scale_range = 0.5 ,rotate_range = 0,prob = 0.4,padding_mode = 'zeros',mode = "bilinear"),
                Rand2DElastic(prob=0.2,spacing=(30, 30),magnitude_range=(1, 2),padding_mode="zeros",), 
                RandAdjustContrast(prob=0.2,gamma = (0.7,2)),
                RandGibbsNoise(prob=0.2, alpha=(0.0,0.5)),
                RandGaussianNoise(std = 0.01,prob = 0.2),
                RandShiftIntensity(offsets = 0.2, prob = 0.5),
                Resize([512,512],mode = 'area'),
                EnsureType(),])
train_segtrans = Compose([
                #ScaleIntensity(),
                RandSpatialCrop((window_size, window_size), random_size=False),
                RandRotate90(prob=0.2, spatial_axes=(0, 1)),
                RandAxisFlip(prob = 0.2),
                RandAffine(scale_range = 0.5, rotate_range = np.pi/6,prob = 0.2,padding_mode = 'zeros',mode = "nearest"),
                RandAffine(scale_range = 0.5 ,rotate_range = 0,prob = 0.4,padding_mode = 'zeros',mode = "bilinear"),
                Rand2DElastic(prob=0.2,spacing=(30, 30),magnitude_range=(1, 2),padding_mode="zeros",mode = "nearest"),
                Resize([512,512],mode = 'nearest'),
                EnsureType(),])
val_imtrans = Compose([Resize([512,512],mode = 'area'),EnsureType()])
val_segtrans = Compose([Resize([512,512],mode = 'nearest'),EnsureType()])

domain = General2DSegmentation(base_dir=args.data_dir, dataset=args.datasetS, train_imtrans = train_imtrans,train_segtrans = train_segtrans, split='train')
domain_val = General2DSegmentation(base_dir=args.data_dir, dataset=args.datasetS, train_imtrans = val_imtrans, train_segtrans = val_segtrans, split='test')

domain_loaderS = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)


print(f'-------------The back bone is {args.backbone}-------------')
model_gen = DeepLab(num_classes=1, backbone=args.backbone, output_stride=args.out_stride,
                    sync_bn=args.sync_bn, freeze_bn=args.freeze_bn, BraTs = False).cuda()

model_dis = BoundaryDiscriminator().cuda() # ???
model_dis2 = UncertaintyDiscriminator().cuda()

start_epoch = 0
start_iteration = 0

# 3. optimizer

optim_gen = torch.optim.Adam(
    model_gen.parameters(),
    lr=args.lr_gen,
    betas=(0.9, 0.99)
)
optim_dis = torch.optim.SGD(
    model_dis.parameters(),
    lr=args.lr_dis,
    momentum=args.momentum,
    weight_decay=args.weight_decay
)
optim_dis2 = torch.optim.SGD(
    model_dis2.parameters(),
    lr=args.lr_dis,
    momentum=args.momentum,
    weight_decay=args.weight_decay
)

trainer = Trainer.Trainer_2DImages(
    cuda=cuda,
    model_gen=model_gen,
    model_dis=model_dis,
    model_uncertainty_dis=model_dis2,
    optimizer_gen=optim_gen,
    optimizer_dis=optim_dis,
    optimizer_uncertainty_dis=optim_dis2,
    lr_gen=args.lr_gen,
    lr_dis=args.lr_dis,
    lr_decrease_rate=args.lr_decrease_rate,
    val_loader=domain_loader_val,
    domain_loaderS=domain_loaderS,
    domain_loaderT=domain_loader_val,
    out=args.out,
    max_epoch=args.max_epoch,
    stop_epoch=args.stop_epoch,
    interval_validate=args.interval_validate,
    batch_size=args.batch_size,
    warmup_epoch=args.warmup_epoch,
)
trainer.epoch = start_epoch
trainer.iteration = start_iteration
trainer.train()