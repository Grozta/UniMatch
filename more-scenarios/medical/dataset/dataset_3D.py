from dataset.transform import random_rot_flip, random_rotate, obtain_cutmix_box_3d
from copy import deepcopy
import math
import numpy as np
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset
from dataset.dataset_transform import ColorJitter, NoiseJitter ,RandomCropD
from util.tools import *

class Dataset_3D(Dataset):
    def __init__(self, mode, config, nsample=None,Validation_all = False):
        
        self.mode = mode
        self.size =config["dataset_output_size"]
        self.window_level=config["data_window_level"]
        self.Validation_all = Validation_all
        self.patch_size = config["patch_size"]

        dataset_info = load_json(config["dataset_info_path"])
        if mode == 'train_l':
            self.ids = [data["precessed_image_npz"] for identif, data in dataset_info["train_case_list"].items()]
            if nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        elif mode == 'train_u':
            self.ids = [data["precessed_image_npz"] for identif, data in dataset_info["unlabeled"].items()]
        else:
            
            tarin_set = [data["precessed_image_npz"] for identif, data in dataset_info["train_case_list"].items()]
            val_set = [data["precessed_image_npz"] for identif, data in dataset_info["val_case_list"].items()]
            if Validation_all:
                self.ids = tarin_set+val_set
            else:
                self.ids = val_set

        self.color_jitter = ColorJitter()
        self.noise_and_blur_jitter = NoiseJitter()
        self.random_crop = RandomCropD(self.size)

    def __getitem__(self, item):
        id_name = self.ids[item]
        sample = np.load(id_name)['data']
        img = sample[0]
        if len(sample)>1:
            mask = sample[1]
        else:
            mask= None
        # 设定窗宽窗位
        img = np.clip(img,  self.window_level[0],  self.window_level[1])
        # 归一化
        img = (img - np.mean(img)) / (np.std(img) + 1e-5)

        if self.mode == 'val':
            x, y, z = img.shape
            img = zoom(img, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(mask).long()

        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)

        # x, y, z = img.shape
        # img = zoom(img, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)
        # if  mask is not None:
        #     mask = zoom(mask, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)

        if self.mode == 'train_l':
            patch_list=[]
            x, y, z = img.shape
            img_0 = zoom(img, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)
            mask_0 = zoom(mask, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)
            patch_list.append((torch.from_numpy(img_0).unsqueeze(0).float(), torch.from_numpy(np.array(mask_0)).long()))    

            for i in range(1,self.patch_size):
                # 随机裁剪出patch
                patch_item = self.random_crop({'image':img,'label':mask})
                # 加入到patch_list中
                patch_list.append((torch.from_numpy(patch_item['image']).unsqueeze(0).float(), torch.from_numpy(np.array(patch_item['label'])).long()))
            return patch_list
        
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        #img = torch.from_numpy(np.array(img)).unsqueeze(0).float()

        img_s1, img_s2 = self.color_jitter(img_s1),self.color_jitter(img_s2)
        img_s1, img_s2 = self.noise_and_blur_jitter(img_s1),self.noise_and_blur_jitter(img_s2)
        #img_s1, img_s2 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float(), torch.from_numpy(np.array(img_s2)).unsqueeze(0).float()
        
        patch_list=[]
        x, y, z = img.shape
        img_0 = zoom(img, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)
        img_s1_0 = zoom(img_s1, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)
        img_s2_0 = zoom(img_s2, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)
        img_0 = torch.from_numpy(np.array(img_0)).unsqueeze(0).float()
        img_s1_0, img_s2_0 = torch.from_numpy(np.array(img_s1_0)).unsqueeze(0).float(), torch.from_numpy(np.array(img_s2_0)).unsqueeze(0).float()
        cutmix_box1, cutmix_box2 = obtain_cutmix_box_3d(self.size, p=0.5), obtain_cutmix_box_3d(self.size, p=0.5)
        cutmix_box1, cutmix_box2 = cutmix_box1.long(), cutmix_box2.long()
        patch_list.append((img_0,img_s1_0,img_s2_0,cutmix_box1,cutmix_box2))    

        for i in range(1,self.patch_size):
            # 随机裁剪出patch
            patch_item_1 = self.random_crop({'image':img,'label':img_s1})
            patch_item_2 = self.random_crop({'image':img,'label':img_s2})
            img_0 = torch.from_numpy(np.array(patch_item_1['image'])).unsqueeze(0).float()
            img_s1_0, img_s2_0 = torch.from_numpy(np.array(patch_item_1['label'])).unsqueeze(0).float(), torch.from_numpy(np.array(patch_item_2['label'])).unsqueeze(0).float()
            cutmix_box1, cutmix_box2 = obtain_cutmix_box_3d(self.size, p=0.5), obtain_cutmix_box_3d(self.size, p=0.5)
            cutmix_box1, cutmix_box2 = cutmix_box1.long(), cutmix_box2.long()
            # 加入到patch_list中
            patch_list.append((img_0,img_s1_0,img_s2_0,cutmix_box1,cutmix_box2))  
        return patch_list

    def __len__(self):
        return len(self.ids)
