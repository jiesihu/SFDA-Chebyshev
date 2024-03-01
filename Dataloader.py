from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import h5py
import random

class General2DSegmentation(Dataset):
    """
    2D segmentation dataset
    """

    def __init__(self,
                 base_dir='',
                 dataset='Brats20',
                 split='train',
                 testid=None,
                 train_imtrans = None,
                 train_segtrans = None,
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []

        self._image_dir = os.path.join(self._base_dir, dataset, split,'image')
        print(self._image_dir)
        imagelist = sorted(glob(self._image_dir + "/*.png"))
        for image_path in imagelist:
            self.image_list.append({'image': image_path})
        self.transform = train_imtrans
        self.transform_seg = train_segtrans
        
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):            
        image_np = Image.open(self.image_list[index]['image']).convert('RGB')
        mask = Image.open(self.image_list[index]['image'].replace('image','mask'))
        if mask.mode == 'RGB':
            mask = mask.convert('L')
            
        _img_name = self.image_list[index]['image'].split('/')[-1]
        
        image_np = np.array(image_np)
        mask = np.array(mask)
        image_np = np.moveaxis(image_np, -1, 0).astype('float')
        image_np /= 255
        mask = mask[None,:,:]
        
        if self.transform is not None:
            _seed = random.randint(1,100000)
            self.transform.set_random_state(seed=_seed)
            self.transform_seg.set_random_state(seed=_seed)
            image_np = self.transform(image_np)
            mask =  self.transform_seg(mask)
        anco_sample = {'image': image_np.float(), 'map': (mask>0).float(), 'img_name': _img_name}

        return anco_sample


    def __str__(self):
        return '2D images(split=' + str(self.split) + ')'



class General2DSegmentation_ST(Dataset):
    """
    2D segmentation dataset
    Output the data for both student and teacher model
    """

    def __init__(self,
                 base_dir='',
                 dataset='Brats20',
                 split='train',
                 testid=None,
                 train_imtrans = None,
                 train_segtrans = None,
                 train_imtrans_teacher = None,
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []

        self._image_dir = os.path.join(self._base_dir, dataset, split,'image')
        print(self._image_dir)
        imagelist = sorted(glob(self._image_dir + "/*.png"))
        for image_path in imagelist:
            self.image_list.append({'image': image_path})
        self.transform = train_imtrans
        self.transform_seg = train_segtrans
        self.transform_teacher = train_imtrans_teacher
        
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):            
        image_np = Image.open(self.image_list[index]['image']).convert('RGB')
        mask = Image.open(self.image_list[index]['image'].replace('image','mask'))
        if mask.mode == 'RGB':
            mask = mask.convert('L')
            
        _img_name = self.image_list[index]['image'].split('/')[-1]
        
        image_np = np.array(image_np)
        mask = np.array(mask)
        image_np = np.moveaxis(image_np, -1, 0).astype('float')
        image_np /= 255
        mask = mask[None,:,:]
        
        if self.transform is not None:
            _seed = random.randint(1,100000)
            self.transform.set_random_state(seed=_seed)
            self.transform_seg.set_random_state(seed=_seed)
            self.transform_teacher.set_random_state(seed=_seed)
            
            image_np_teacher = self.transform_teacher(image_np)
            image_np = self.transform(image_np)
            mask =  self.transform_seg(mask)
        anco_sample = {'image': image_np.float(),'image_teacher': image_np_teacher.float(), 'map': (mask>0).float(), 'img_name': _img_name}

        return anco_sample


    def __str__(self):
        return '2D images(split=' + str(self.split) + ')'







