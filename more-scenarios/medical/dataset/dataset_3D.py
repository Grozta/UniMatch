from dataset.transform import random_rot_flip, random_rotate, obtain_cutmix_box_3d
from copy import deepcopy
import math
import numpy as np
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset
from dataset.dataset_transform import ColorJitter, NoiseJitter
from util.tools import *

class Dataset_3D(Dataset):
    def __init__(self, mode, config, nsample=None):
        
        self.mode = mode
        self.size =config["dataset_output_size"]
        self.window_level=config["data_window_level"]

        dataset_info = load_json(config["dataset_info_path"])
        if mode == 'train_l':
            self.ids = [data["precessed_image_npz"] for identif, data in dataset_info["train_case_list"].items()]
            if nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        elif mode == 'train_u':
            self.ids = [data["precessed_image_npz"] for identif, data in dataset_info["unlabeled"].items()]
        else:
            self.ids = [data["precessed_image_npz"] for identif, data in dataset_info["val_case_list"].items()]

        self.color_jitter = ColorJitter()
        self.noise_and_blur_jitter = NoiseJitter()

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

        x, y, z = img.shape
        img = zoom(img, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)
        if  mask is not None:
            mask = zoom(mask, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)

        if self.mode == 'train_l':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()
        
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float()

        img_s1, img_s2 = self.color_jitter(img_s1),self.color_jitter(img_s2)
        img_s1, img_s2 = self.noise_and_blur_jitter(img_s1),self.noise_and_blur_jitter(img_s2)
        img_s1, img_s2 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float(), torch.from_numpy(np.array(img_s2)).unsqueeze(0).float()

        cutmix_box1, cutmix_box2 = obtain_cutmix_box_3d(self.size, p=0.5), obtain_cutmix_box_3d(self.size, p=0.5)
        
        return img, img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
