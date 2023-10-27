import argparse
import os
import pprint
import shutil
from tqdm import tqdm
import torch
import numpy as np
from torch import nn
from torch.cuda import amp
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset_3D import Dataset_3D
from model.unet_3D import UNet_3D
from util.utils import AverageMeter, count_params, Logger
from monai.losses.dice import DiceLoss
from util.tools import *


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str,default="configs/flare22_val.yaml")
parser.add_argument('--save_path', type=str,default="exp/supervised_unet/home_01/val")

def main():
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    output_dir = os.path.abspath(args.save_path)
    maybe_mkdir_p(output_dir)
    if cfg["Validation_all"]:
        cfg["output_json"]= "val_all.json"
        cfg["log_dir"]= "val_log_all"
    else:
        cfg["output_json"]= "val_only.json"
        cfg["log_dir"]= "val_log_only"

    log_dir = join(output_dir,cfg["log_dir"])
    maybe_mkdir_p(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    valset = Dataset_3D('val',cfg,Validation_all=cfg["Validation_all"])

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=cfg['num_workers'],
                           drop_last=False)

    cudnn.enabled = True
    cudnn.benchmark = True
    model = UNet_3D(in_chns=1, class_num=cfg['nclass'],feature_chns = cfg['feature_chns'],dropout=cfg['dropout'])

    if os.path.exists(cfg["pretrained_model_path"]):
        checkpoint = torch.load(cfg["pretrained_model_path"])
        model.load_state_dict(checkpoint['model'])

    else:
        print("model can not found! return")
        return
    model.cuda()
    
    model.eval()
    dice_val = []
    with torch.no_grad():
        val_loop = tqdm(enumerate(valloader), total =len(valloader))
        for i, (img, mask) in val_loop:
            dice_class = [0] * (cfg['nclass'] - 1)
            img, mask = img.cuda(), mask.cuda()

            d, h, w = mask.shape[-3:]
            
            pred = model(img)
            
            pred = F.interpolate(pred, (d, h, w), mode='trilinear', align_corners=False)
            pred = pred.argmax(dim=1)

            for cls in range(1, cfg['nclass']):
                inter = ((pred == cls) * (mask == cls)).sum().item()
                union = (pred == cls).sum().item() + (mask == cls).sum().item()
                dice_class[cls-1] = 2.0 * inter / union
                writer.add_scalars('val/Dice',{cfg['class_name_list'][cls-1]: dice_class[cls-1]}, i)

            dice_val.append(dice_class)
            val_loop.set_postfix(avg_dice = sum(dice_class)/len(dice_class))
    
    mean_dice_in_vallist = np.array(dice_val).sum(axis=0)/(len(valloader))
    mean_dice_list = mean_dice_in_vallist.tolist()
    mean_dice = sum(mean_dice_list)/len(mean_dice_list)

    for (cls_idx, dice) in enumerate(mean_dice_list):
        print('***** Evaluation ***** >>>> Class [{:} {:}] MeanDice: '
                    '{:.4f} '.format(cls_idx+1, cfg['class_name_list'][cls_idx], dice))

    res = {
        'val_items_number': len(valloader),
        "val_res":dice_val,
        "val_mean_dice":mean_dice_list,
        "mean_dice":mean_dice
    }

    save_json(res,join(output_dir,cfg['output_json']))

if __name__ == '__main__':
    main()
