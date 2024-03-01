import argparse
import os
import os.path as osp
import json 
import sys

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from utils import losses
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
import networks.deeplabv3_eval as netd_eval
import cv2
import torch.backends.cudnn as cudnn
import random
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import time

import monai
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,)

from Dataloader import General2DSegmentation,General2DSegmentation_ST




parser = argparse.ArgumentParser()
parser.add_argument('--model-file', type=str, default='./logs/Domain_ISBI/20240131_164228.060498/checkpoint_197.pth.tar')
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--target', type=str, default='Domain_target',help = 'The folder name of target domain')
parser.add_argument('--opt', type=str, default='torch.optim.Adam(param, lr=0.00001, betas=(0.9, 0.99))')
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--data-dir', default='/your_data_path', help='data root path')
parser.add_argument('--out-stride',type=int,default=16)
parser.add_argument('--seed',type=int,default=4)
parser.add_argument('--sync-bn',type=bool,default=True)
parser.add_argument('--freeze-bn',type=bool,default=False)
parser.add_argument('--source_eval',type=bool,default=False)
parser.add_argument('--train_eval',type=bool,default=False)
parser.add_argument('--eval_mode',type=bool,default=False)
parser.add_argument('--ent',type=bool,default=True)
parser.add_argument('--gamma0', type=float, default=1000)
parser.add_argument('--dropout_times',type=int,default=10)
parser.add_argument('--P_thres', type=float, default=0.05)
parser.add_argument('--P_thres_start', type=float, default=0.001)
parser.add_argument('--m', type=float, default=0.999)
parser.add_argument('--lambda0', type=float, default=0.1)
parser.add_argument('--name', type=str, default='',help = 'Recode the script name.')
parser.add_argument('--train_shuffle',type=bool,default=False)
parser.add_argument('--div_loss', type=float, default=0.1, help = 'Coef. for diversity loss')
parser.add_argument('--thres', type=float, default=0.5,help = 'Threshold for probability map')


args = parser.parse_args()
args.name = os.path.basename(sys.argv[0])
print(args)



bceloss = torch.nn.BCELoss(reduction='none')
seed = args.seed
savefig = False
get_hd = True
model_save = True
if True:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all gpus


os.chdir(args.save_path)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
model_file = args.model_file


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


train_imtrans = Compose([
                #ScaleIntensity(),
                RandAdjustContrast(prob=0.3,gamma = (0.7,2)),
                RandGibbsNoise(prob=0.3, alpha=(0.0,0.5)),
                RandGaussianNoise(std = 0.01,prob = 0.3),
                RandShiftIntensity(offsets = 0.2, prob = 0.3),
                Resize([512,512],mode = 'area'),
                EnsureType(),])
train_segtrans = Compose([
                #ScaleIntensity(),
                Resize([512,512],mode = 'nearest'),
                EnsureType(),])
val_imtrans = Compose([Resize([512,512],mode = 'area'),EnsureType()])
val_segtrans = Compose([Resize([512,512],mode = 'nearest'),EnsureType()])


domain = General2DSegmentation_ST(base_dir=args.data_dir, dataset=args.target, train_imtrans = train_imtrans, train_imtrans_teacher = val_imtrans,
                             train_segtrans = train_segtrans, split='train')
domain_val = General2DSegmentation(base_dir=args.data_dir, dataset=args.target, train_imtrans = val_imtrans, train_segtrans = val_segtrans, split='test')

domain_loaderS = DataLoader(domain, batch_size=8, shuffle=args.train_shuffle, num_workers=2, pin_memory=True)
domain_loader_val = DataLoader(domain_val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# teacher loader
db_train_t = General2DSegmentation(base_dir=args.data_dir, dataset=args.target, train_imtrans = val_imtrans, train_segtrans = val_segtrans, split='train')
train_loader_t = DataLoader(db_train_t, batch_size=16, shuffle=False, num_workers=2,pin_memory=True)

# 2. model
model = netd.DeepLab(num_classes=1, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn, BraTs = False).cuda()
model_eval = netd_eval.DeepLab(num_classes=1, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn, BraTs = True).cuda()

print('==> Loading %s model file: %s' %
      (model.__class__.__name__, model_file))
checkpoint = torch.load(model_file)
model.load_state_dict(checkpoint['model_state_dict'])
model.train()
var_list = model.named_parameters()
param = model.parameters()
print(args.opt)
optim_gen = eval(args.opt)

# load a dict
dict_ = model.state_dict(keep_vars = True)

best_avg = 0.0
best_std = 0.0
best_hd_std = 0.0
iter_num = 0


# save hyper parameters
writer = SummaryWriter(comment = '-GPU'+str(args.gpu))
converted_dict = vars(args)
with open(os.path.join('./'+writer.log_dir,"argparse.json"), "w") as outfile:
    json.dump(converted_dict, outfile)

Record_every_step = {'Epoch':[], 'val_dice':[],'avg_dice':[],'hd':[],'avg_hd':[]}


# 2. model
model_teacher = netd.DeepLab(num_classes=1, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn, BraTs = False).cuda()
print('==> Loading %s model file: %s' %
      (model_teacher.__class__.__name__, model_file))
checkpoint = torch.load(model_file)
model_teacher.load_state_dict(checkpoint['model_state_dict'])
model_teacher.train()
model_teacher.requires_grad_(False)
print("Teacher model load sucessfully")



for epoch_num in tqdm.tqdm(range(args.epoch), ncols=70):
    
    model.train()
    for (batch_idx,sample) in tqdm.tqdm(
                    enumerate(domain_loaderS), total=len(domain_loaderS),
                    desc='Train iteration', ncols=80,
                    leave=False): #zip(enumerate(domain_loaderS),train_loader_t):
        data, target, img_name = sample['image'], sample['map'], sample['img_name']
        data_t, target_t, img_name_t = sample['image_teacher'], target, img_name
        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        # Get student output
        # turn on the batch-norm test
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or ('BatchNorm' in m.__class__.__name__):
                m.eval()
        
        # get the uncertainty map
        preds = torch.zeros([args.dropout_times, data.shape[0], 1, data.shape[2], data.shape[3]]).cuda()
        features = torch.zeros([args.dropout_times, data.shape[0], 305, 128, 128]).cuda()
        for i in range(args.dropout_times):
            with torch.no_grad():
                preds[i,...], _, features[i,...] = model(data[:,:])
                
        preds1 = torch.sigmoid(preds)
        preds1_avg = torch.mean(preds1,dim=0).clone()
        preds = torch.sigmoid(preds)
        std_map = torch.std(preds,dim=0).clone()
        pseudo_label_s = get_PseduoLabel(preds1,thres = args.thres).detach()
        mask_s,cheb_map = compute_mask_cheb(std_map,preds1_avg,thres = args.thres,P_thres =  min(iter_num/len(domain_loaderS)*0.05+args.P_thres_start,args.P_thres))

        cheb_map_f = 1-cheb_map
        proto_pseudo_s = get_proto_pseudo_cheb(torch.cat([pseudo_label_s,pseudo_label_s],dim=1), 
                                               features, torch.cat([mask_s,mask_s],dim=1), torch.cat([cheb_map_f,cheb_map_f],dim=1),data)
        
        mask_s*=(proto_pseudo_s[:,0:1,:]==pseudo_label_s)
                
        # Get teacher output
        # update teacher with momentum
        student_dict = model.state_dict()
        teacher_dict = model_teacher.state_dict()
        update_dic = {k: v*(1.0-args.m) + teacher_dict[k]*args.m for k, v in student_dict.items() if k in teacher_dict}
        model_teacher.load_state_dict(update_dic)
            
        # turn on the batch-norm test
        model_teacher.train()    
        for m in model_teacher.modules():
            if isinstance(m, nn.BatchNorm2d) or ('BatchNorm' in m.__class__.__name__):
                m.eval()
                
        preds = torch.zeros([args.dropout_times, data.shape[0], 1, data.shape[2], data.shape[3]]).cuda()
        features = torch.zeros([args.dropout_times, data.shape[0], 305, 128, 128]).cuda()
        
        if torch.cuda.is_available():
            data_t, target_t = data_t.cuda(), target_t.cuda()
        data_t, target_t = Variable(data_t), Variable(target_t)
        for i in range(args.dropout_times):
            with torch.no_grad():
                preds[i,...], _, features[i,...] = model_teacher(data_t)
        preds1 = torch.sigmoid(preds)
        preds = torch.sigmoid(preds)
        uncertain_map = torch.std(preds,dim=0).detach()
        pseudo_label = get_PseduoLabel(preds1,thres = args.thres).detach()
        prob_temp=torch.mean(preds1,dim=0).detach()
        mask,cheb_map = compute_mask_cheb(uncertain_map,prob_temp,thres = args.thres,P_thres =  min(iter_num/len(domain_loaderS)*0.05+0.0005,args.P_thres))
        
        cheb_map_f = 1-cheb_map
        proto_pseudo = get_proto_pseudo_cheb(torch.cat([pseudo_label,pseudo_label],dim=1), 
                                             features, torch.cat([mask,mask],dim=1), torch.cat([cheb_map_f,cheb_map_f],dim=1),data)
        
        mask*=(proto_pseudo[:,0:1,:]==pseudo_label)

        model.train()
        
        prediction, _, feature = model(data[:,:,:])
        prediction = torch.sigmoid(prediction)
        
        optim_gen.zero_grad()
        loss_seg = 0
 
        # student uncertainty map
        student_weight = torch.exp(-args.gamma0*std_map.mean(dim = (2,3)))
        # teacher uncertainty map
        teacher_weight = torch.exp(-args.gamma0*uncertain_map.mean(dim = (2,3)))
        denominator_temp = student_weight+teacher_weight
        student_weight /= denominator_temp
        teacher_weight /= denominator_temp
        expanded_student_weight = student_weight.view(student_weight.shape[0],student_weight.shape[1] , 1, 1).expand(std_map.shape)
        expanded_teacher_weight = teacher_weight.view(teacher_weight.shape[0],teacher_weight.shape[1] , 1, 1).expand(std_map.shape)

        # teacher CE loss
        loss_seg_pixel = bceloss(prediction, pseudo_label)
        loss_seg = torch.sum(mask * loss_seg_pixel*expanded_teacher_weight) / torch.sum(mask)

        # student CE loss
        loss_seg_pixel_student = bceloss(prediction, pseudo_label_s)
        student_loss = torch.sum(mask_s * loss_seg_pixel_student*expanded_student_weight) / torch.sum(mask_s)
        loss_seg += student_loss
        
        div_loss = prediction.mean([0,2,3])*prediction.mean([0,2,3]).log()+(1-prediction.mean([0,2,3]))*(1-prediction.mean([0,2,3])).log()
        div_loss=div_loss.mean()
        loss_seg+=div_loss*args.div_loss
        
        loss_seg.backward()

        optim_gen.step()
        iter_num = iter_num + 1
        
    # Evaluation every epoch
    model.eval()
    val_dice, val_hd,results = eval_model(model, domain_loader_val, threshold_ = args.thres)

    # record and print
    Record_every_step['Epoch'].append(epoch_num); Record_every_step['val_dice'].append(val_dice); 
    Record_every_step['avg_dice'].append(val_dice); Record_every_step['hd'].append(val_hd); 
    Record_every_step['avg_hd'].append(val_hd); 

    if val_dice>best_avg:
        best_avg = val_dice
        best_avg_hd = val_hd
        best_std = np.std(results['dice'])
        best_hd_std = np.std(results['hd'])
        if model_save:
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_avg': val_dice,
                'best_avg_hd': val_hd,
            }, os.path.join('./'+writer.log_dir,"best_model.pth.tar"))

    print('Std dice:',round(np.std(results['dice']),4),'Std val_hd:',round(np.std(results['hd']),4),)
    print("Dice: %.4f HD: %.4f" % (val_dice, val_hd))
    print("best Dice: %.4f best HD: %.4f " %(best_avg, best_avg_hd))

    # Record some results
    with open(os.path.join('./'+writer.log_dir,"Record_every_step.json"), "w") as outfile:
        json.dump(Record_every_step, outfile)
    with open(os.path.join('./'+writer.log_dir,"Best_metric.json"), "w") as outfile:
        json.dump({'best_avg':best_avg,'best_avg_hd':best_avg_hd,'best_std':best_std,'best_hd_std':best_hd_std}, outfile)

