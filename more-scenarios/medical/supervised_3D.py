import argparse
import logging
import os
import pprint
import shutil

import torch
import numpy as np
from torch import nn
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
parser.add_argument('--config', type=str,default="configs/flare22.yaml")
parser.add_argument('--restart_train', required=False, default=True, action="store_true",)

def main():
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cfg["restart_train"]= args.restart_train
    output_dir = os.path.abspath(join(cfg["output_dir_root"],cfg["project_name"],cfg["train_name"]))
    if args.restart_train and isdir(output_dir):
        shutil.rmtree(output_dir)
    maybe_mkdir_p(output_dir)

    log_dir = join(output_dir,"log")
    maybe_mkdir_p(log_dir)
    logger = Logger(join(log_dir,"log.txt")).logger
    logger.info('{}\n'.format(pprint.pformat(cfg)))
    
    writer = SummaryWriter(log_dir=log_dir)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = UNet_3D(in_chns=1, class_num=cfg['nclass'],feature_chns = cfg['feature_chns'],dropout=cfg['dropout'])
  
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)

    model.cuda()

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(to_onehot_y=True,softmax=True)

    trainset = Dataset_3D('train_l',cfg)
    valset = Dataset_3D('val',cfg)

    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=cfg['num_workers'], drop_last=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=cfg['num_workers'],
                           drop_last=False)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(output_dir, 'latest.pth')):
        checkpoint = torch.load(os.path.join(output_dir, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = (criterion_ce(pred, mask) + criterion_dice(pred, mask.unsqueeze(1))) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            writer.add_scalar('train/loss_all', loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        model.eval()
        dice_val = []
        with torch.no_grad():
            for img, mask in valloader:
                dice_class = [0] * (cfg['nclass'] - 1)
                img, mask = img.cuda(), mask.cuda()

                d, h, w = mask.shape[-3:]
                
                pred = model(img)
                
                pred = F.interpolate(pred, (d, h, w), mode='trilinear', align_corners=False)
                pred = pred.argmax(dim=1)

                for cls in range(1, cfg['nclass']):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls-1] += 2.0 * inter / union
                dice_val.append(dice_class)
         
        mean_dice_in_vallist = np.array(dice_val).sum(axis=0)/(cfg['nclass'] - 1)
        mean_dice_list = mean_dice_in_vallist.tolist()
        mean_dice = sum(mean_dice_list)/len(mean_dice_list)

        for (cls_idx, dice) in enumerate(mean_dice_list):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] MeanDice: '
                        '{:.4f} '.format(cls_idx+1, cfg['class_name_list'][cls_idx], dice))
        
        writer.add_scalar('eval/MeanDice', mean_dice, epoch)
        for i, dice in enumerate(dice_class):
            writer.add_scalar('eval/%s_dice' % (cfg['class_name_list'][i]), dice, epoch)

        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(checkpoint, os.path.join(output_dir, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))


if __name__ == '__main__':
    main()
