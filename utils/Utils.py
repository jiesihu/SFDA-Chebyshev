
# from scipy.misc import imsave
import os.path as osp
import numpy as np
import os
import cv2
from skimage import morphology
import scipy
from PIL import Image
from matplotlib.pyplot import imsave
# from keras.preprocessing import image
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from skimage import measure, draw


import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib.pyplot import imsave
from datetime import datetime
import pytz
import cv2
import torch.backends.cudnn as cudnn
import random
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# from scipy.misc import imsave
from utils.metrics import *
import cv2

savefig = False
get_hd = True
model_save = True
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    
    
class RandomRotate_tensor(object):
    def __init__(self, prob):
        self.prob = prob
        self.k = 1
        self.rand = 0

    def __call__(self, sample):
        self.k = random.randint(1, 4)
        self.rand = random.random()
        if self.rand > self.prob:
            return torch.rot90(sample,k = self.k,dims = [2,3])
        else:
            return sample
        
    def same_transform(self,sample):
        if self.rand > self.prob:
            return torch.rot90(sample,k = self.k,dims = [2,3])
        else:
            return sample

def change_model(model):
    '''
    Change the setting of batch-norm in the model
    '''
    parameters = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or ('BatchNorm' in m.__class__.__name__):
            try:
                m.requires_grad_(True)
            except:
                list(model.children())[1].aspp1.bn.training = True
                for mm in m.parameters():
                    mm.requires_grad==True
            m.track_running_stats = True
#             m.running_mean = None
#             m.running_var = None
            parameters += list(m.parameters())
    return parameters

def get_attribute_of_model(model,list_name,back_index = 0):
    '''
    Use the name from .state_dict() to track the module in model
    '''
    if len(list_name)>back_index:
        module = getattr(model,list_name[0])
        list_name_next = list_name[1:]
        return get_attribute_of_model(module,list_name_next,back_index)
    else:
        return model   

def eval_model(model_eval, test_loader,threshold_ = 0.5):
    val_dice = 0.0;datanum_cnt = 0.0
    WT_hd = 0.0;datanum_cnt = 0.0;datanum_cnt_cup = 0.0
    results = {'dice':[],'hd':[]}

    with torch.no_grad():
        for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), ncols=70): #enumerate(test_loader):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            prediction, boundary, _ = model_eval(data)
            prediction = torch.sigmoid(prediction)

            target_numpy = target.data.cpu()
            prediction = prediction.data.cpu()
            
            prediction[prediction>threshold_] = 1;prediction[prediction <= threshold_] = 0


            WT_dice = dice_coefficient_numpy(prediction[:,0, ...], target_numpy[:, 0, ...])

            for i in range(prediction.shape[0]):
                hd_tmp = hd_numpy(prediction[i, 0, ...], target_numpy[i, 0, ...], get_hd)
                if np.isnan(hd_tmp):
                    datanum_cnt_cup -= 1.0
                else:
                    WT_hd += hd_tmp
                    results['hd'].append(hd_tmp)

            val_dice += np.sum(WT_dice)
            
            results['dice']+=list(WT_dice)
            datanum_cnt += float(prediction.shape[0])
            datanum_cnt_cup += float(prediction.shape[0])

    val_dice /= datanum_cnt
    WT_hd /= datanum_cnt_cup
    return val_dice, WT_hd,results



def get_PseduoLabel(preds1,thres = 0.5):
    prediction = torch.mean(preds1,dim=0)
    pseudo_label = prediction.clone()
    pseudo_label[pseudo_label > thres] = 1.0; pseudo_label[pseudo_label <= thres] = 0.0
    return pseudo_label
def compute_mask(uncertain_map,prob_temp,thres = 0.5,P_thres = 0.025):
    # use chebyshev method to find the mask
    cheb_map = torch.square(uncertain_map)/(torch.square(uncertain_map)+torch.square(prob_temp-thres))
    mask_ = cheb_map<P_thres
    mask_ = mask_.int()
    return mask_

def _momentum_update_key_encoder(m,src_model,momentum_model):
    """
    Momentum update of the key encoder
    """
    # encoder_q -> encoder_k
    for param_q, param_k in zip(
        src_model.parameters(), momentum_model.parameters()
    ):
        param_k.data = param_k.data * m + param_q.data * (1.0 - m)

def get_PseduoLabel(preds1,thres = 0.5):
    prediction = torch.mean(preds1,dim=0)
    pseudo_label = prediction.clone()
    pseudo_label[pseudo_label > thres] = 1.0; pseudo_label[pseudo_label <= thres] = 0.0
    return pseudo_label
def compute_mask(uncertain_map,prob_temp,thres = 0.5,P_thres = 0.025):
    # use chebyshev method to find the mask
    cheb_map = torch.square(uncertain_map)/(torch.square(uncertain_map)+torch.square(prob_temp-thres))
    mask_ = cheb_map<P_thres
    mask_ = mask_.int()
    return mask_
def compute_mask_cheb(uncertain_map,prob_temp,thres = 0.5,P_thres = 0.025):
    # use chebyshev method to find the mask
    cheb_map = torch.square(uncertain_map)/(torch.square(uncertain_map)+torch.square(prob_temp-thres))
    mask_ = cheb_map<P_thres
    mask_ = mask_.int()
    return mask_, cheb_map
# pseudo_label-->pseudo_label, features-->features, std_map-->mask, prediction-->prob_temp
def get_proto_pseudo_cheb(pseudo_label, features, mask, prediction,data):
    feature = torch.mean(features,dim=0)
    thres_uncertainty = 0.05
    target_0_obj = F.interpolate(pseudo_label[:,0:1,...], size=feature.size()[2:], mode='nearest')
    target_1_obj = F.interpolate(pseudo_label[:, 1:, ...], size=feature.size()[2:], mode='nearest')
    prediction_small = F.interpolate(prediction, size=feature.size()[2:], mode='bilinear', align_corners=True)
    mask_small = F.interpolate(mask.float(), size=feature.size()[2:], mode='bilinear', align_corners=True)
    target_0_bck = 1.0 - target_0_obj;target_1_bck = 1.0 - target_1_obj

    mask_0_obj = mask_small[:,0:1,:]
    mask_0_bck = mask_small[:,0:1,:]
    mask_1_obj = mask_small[:,1:,:]
    mask_1_bck = mask_small[:,1:,:]

    feature_0_obj = feature * target_0_obj*mask_0_obj;feature_1_obj = feature * target_1_obj*mask_1_obj
    feature_0_bck = feature * target_0_bck*mask_0_bck;feature_1_bck = feature * target_1_bck*mask_1_bck

    centroid_0_obj = torch.sum(feature_0_obj*prediction_small[:,0:1,...], dim=[2,3], keepdim=True)
    centroid_1_obj = torch.sum(feature_1_obj*prediction_small[:,1:,...], dim=[2,3], keepdim=True)
    centroid_0_bck = torch.sum(feature_0_bck*prediction_small[:,0:1,...], dim=[2,3], keepdim=True)
    centroid_1_bck = torch.sum(feature_1_bck*prediction_small[:,1:,...], dim=[2,3], keepdim=True)
    target_0_obj_cnt = torch.sum(mask_0_obj*target_0_obj*prediction_small[:,0:1,...], dim=[2,3], keepdim=True)
    target_1_obj_cnt = torch.sum(mask_1_obj*target_1_obj*prediction_small[:,1:,...], dim=[2,3], keepdim=True)
    target_0_bck_cnt = torch.sum(mask_0_bck*target_0_bck*prediction_small[:,0:1,...], dim=[2,3], keepdim=True)
    target_1_bck_cnt = torch.sum(mask_1_bck*target_1_bck*prediction_small[:,1:,...], dim=[2,3], keepdim=True)

    centroid_0_obj /= target_0_obj_cnt; centroid_1_obj /= target_1_obj_cnt
    centroid_0_bck /= target_0_bck_cnt; centroid_1_bck /= target_1_bck_cnt

    distance_0_obj = torch.sum(torch.pow(feature - centroid_0_obj, 2), dim=1, keepdim=True)
    distance_0_bck = torch.sum(torch.pow(feature - centroid_0_bck, 2), dim=1, keepdim=True)
    distance_1_obj = torch.sum(torch.pow(feature - centroid_1_obj, 2), dim=1, keepdim=True)
    distance_1_bck = torch.sum(torch.pow(feature - centroid_1_bck, 2), dim=1, keepdim=True)

    proto_pseudo_0 = torch.zeros([data.shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()
    proto_pseudo_1 = torch.zeros([data.shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()

    proto_pseudo_0[distance_0_obj < distance_0_bck] = 1.0
    proto_pseudo_1[distance_1_obj < distance_1_bck] = 1.0
    proto_pseudo = torch.cat((proto_pseudo_0, proto_pseudo_1), dim=1)
    proto_pseudo = F.interpolate(proto_pseudo, size=data.size()[2:], mode='nearest')
    return proto_pseudo
