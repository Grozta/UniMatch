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
from util.utils import AverageMeter, count_params, Logger, DiceLoss
from util.tools import *
import time


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str,default="configs/flare22_unimatch.yaml")
parser.add_argument('--save_path', type=str,default="exp/unimatch_unet/home_01_144")
parser.add_argument('--restart_train', required=False, default=False, action="store_true",)

def main():
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cfg["restart_train"]= args.restart_train
    output_dir = os.path.abspath(args.save_path)
    if args.restart_train and isdir(output_dir):
        shutil.rmtree(output_dir)
    maybe_mkdir_p(output_dir)
    shutil.copy(args.config,output_dir)

    log_dir = join(output_dir,"log")
    maybe_mkdir_p(log_dir)
    logger = Logger(join(log_dir,"log.txt")).logger
    logger.info(f"train start! output_dir:{output_dir}")
    logger.info('{}\n'.format(pprint.pformat(cfg)))
    
    writer = SummaryWriter(log_dir=log_dir)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = UNet_3D(in_chns=1, class_num=cfg['nclass'],feature_chns = cfg['feature_chns'],dropout=cfg['dropout'])
  
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg['nclass'])
    
    trainset_u = Dataset_3D('train_u',cfg )
    trainset_l = Dataset_3D('train_l',cfg, nsample=len(trainset_u.ids))
    valset = Dataset_3D('val',cfg)

    total_iters = cfg['per_epoch_resample_count']  * cfg['epochs'] 
    previous_best = 0.0
    epoch = -1

    model.cuda()
    
    if os.path.exists(os.path.join(output_dir, 'latest.pth')):
        checkpoint = torch.load(os.path.join(output_dir, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    elif os.path.exists(cfg["pretrained_model_path"]):
        checkpoint = torch.load(cfg["pretrained_model_path"])
        model.load_state_dict(checkpoint['model'])
        logger.info('************ Load from pretrained model')
    else:
        logger.info('************ start oringinal model')

    scaler = amp.GradScaler()
    
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainset_l.reset_sample_pool(epoch,cfg["per_epoch_resample_count"])
        trainset_u.reset_sample_pool(epoch,cfg["per_epoch_resample_count"])

        trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=cfg['num_workers'], drop_last=True,shuffle=False)
        trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=cfg['num_workers'], drop_last=True,shuffle=False)
        trainloader_u_mix = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=cfg['num_workers'], drop_last=True,shuffle=True)
        valloader = DataLoader(valset, batch_size=1, 
                           pin_memory=True, num_workers=cfg['num_workers'], drop_last=False,shuffle=False)
        
        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)
        
        train_loop = tqdm(enumerate(loader), total =len(trainloader_u),leave= True)
        train_loop.set_description(f'Train[{epoch}/{cfg["epochs"]}]')

        with amp.autocast(enabled=cfg["is_amp_train"]):
            for i, ((img_x, mask_x),
                    (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
                    (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in train_loop:
                
                iter_start_time = time.time()
                with torch.no_grad():
                    model.eval()
                    img_u_w_mix = img_u_w_mix.cuda()
                    with amp.autocast(enabled=cfg["is_amp_train"]):
                        pred_u_w_mix = model(img_u_w_mix).detach().cpu().float()
                    conf_u_w_mix = (pred_u_w_mix.softmax(dim=1).max(dim=1)[0]).float()
                    mask_u_w_mix = (pred_u_w_mix.argmax(dim=1))
                    
                    if cfg["is_dynamic_empty_cache"]:
                        del img_u_w_mix
                        torch.cuda.empty_cache()

                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                    img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
                img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                    img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

                model.train()

                optimizer.zero_grad()

                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

                img_x, img_u_w = img_x.cuda(),img_u_w.cuda()
                img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
                with amp.autocast(enabled=cfg["is_amp_train"]):
                    preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])
                pred_u_w_fp = preds_fp[num_lb:]
                with amp.autocast(enabled=cfg["is_amp_train"]):
                    pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

                pred_u_w = pred_u_w.detach().cpu().float()
                if cfg["is_dynamic_empty_cache"]:
                    torch.cuda.empty_cache()
                conf_u_w = (pred_u_w.softmax(dim=1).max(dim=1)[0]).float()
                mask_u_w = pred_u_w.argmax(dim=1)

                mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
                mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
                conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

                mask_x = mask_x.cuda()
                mask_u_w_cutmixed1 = mask_u_w_cutmixed1.cuda().float()
                mask_u_w_cutmixed2 = mask_u_w_cutmixed2.cuda().float()
                mask_u_w = mask_u_w.cuda()

                ce_loss = criterion_ce(pred_x.float(), mask_x) + 1e-8
                dice_loss = criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1))

                loss_x = (ce_loss+dice_loss)/ 2.0

                loss_u_s1 = criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1),
                                        ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']))
                
                loss_u_s2 = criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1),
                                        ignore=(conf_u_w_cutmixed2 < cfg['conf_thresh']))
                
                loss_u_w_fp = criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1),
                                            ignore=(conf_u_w < cfg['conf_thresh']))
                
                loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss.update(loss.item())
                total_loss_x.update(loss_x.item())
                total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
                total_loss_w_fp.update(loss_u_w_fp.item())
                
                mask_ratio = (conf_u_w >= cfg['conf_thresh']).sum() / conf_u_w.numel()
                total_mask_ratio.update(mask_ratio.item())
                
                iters = epoch * len(trainloader_u) + i
                lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
                if (i % (len(trainloader_u) // 8) == 0):
                    logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                                '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, 
                                                total_loss_w_fp.avg, total_mask_ratio.avg))

                train_loop.set_postfix(loss = total_loss.avg) 
                iter_end_time = time.time()
                logger.info(f'[epoch:{epoch}|iter:{i}] use time: {iter_end_time-iter_start_time}')

            if cfg["is_dynamic_empty_cache"]:
                torch.cuda.empty_cache()

            model.eval()
            dice_val = []
            with torch.no_grad():
                val_loop = tqdm(enumerate(valloader), total =len(valloader),leave= False)
                val_loop.set_description(f'Val [{epoch}/{cfg["epochs"]}]')
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
                    dice_val.append(dice_class)
                    train_loop.set_postfix(avg_dice = sum(dice_class)/len(dice_class))
            
            if cfg["is_dynamic_empty_cache"]:
                del img, mask, pred
                torch.cuda.empty_cache()

            mean_dice_in_vallist = np.array(dice_val).sum(axis=0)/(cfg['nclass'] - 1)
            mean_dice_list = mean_dice_in_vallist.tolist()
            mean_dice = sum(mean_dice_list)/len(mean_dice_list)
            
            for (cls_idx, dice) in enumerate(mean_dice_list):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] MeanDice: '
                            '{:.4f} '.format(cls_idx+1, cfg['class_name_list'][cls_idx], dice))
            
            writer.add_scalar('eval/MeanDice', mean_dice, epoch)
            for i, dice in enumerate(mean_dice_list):
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
