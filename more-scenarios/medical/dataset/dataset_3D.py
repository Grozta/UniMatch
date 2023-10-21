from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box
from copy import deepcopy
import math
import numpy as np
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset
from torchvision import transforms
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


    def __getitem__(self, item):
        id = self.ids[item]
        sample = load_pickle(id)
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
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)

        x, y, z = img.shape
        img = zoom(img, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)
        mask = zoom(mask, (self.size[0] / x, self.size[1] / y,self.size[2] / z), order=0)

        if self.mode == 'train_l':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()

        img = Image.fromarray((img * 500).astype(np.uint8))
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 500.0

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        return img, img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
