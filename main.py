

import time
import datetime
import argparse
import os.path as osp
import numpy as np
import torch

from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from model import *
from data import build_dataloader
from test import test, test_prcc
from train import train
from tools.utils import save_checkpoint, set_seed

VID_DATASET = ['ccvid']

def parse_option():
    parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    return config


def main(config):


    # Build dataloader
    if config.DATA.DATASET == 'prcc':
        trainloader, synloader, queryloader_same, queryloader_diff, galleryloader, dataset = build_dataloader(config)
    else:
        trainloader, synloader, queryloader, galleryloader, dataset = build_dataloader(config)

    # Build model
    clip_model = ClipModel(clip_backbone_name=config.MODEL.CLIP_NAME)
    model = Model(config=config, num_classes=dataset.num_train_pids)
    clip_model = clip_model.cuda()
    clip_model.eval()

    # # Optimizers 
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.OPTIMIZER.LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.TRAIN.LR_SCHEDULER.STEPSIZE, gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    model = model.cuda()

    if config.EVAL_MODE:
        print("Evaluate only")
        with torch.no_grad():
            if config.DATA.DATASET == 'prcc':
                test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                test(config, model, queryloader, galleryloader, dataset)
        return

    # ----------
    #  Training
    # ----------
    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):

        if config.TEST.EVAL_STEP > 0 and (epoch) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            print("==> Test")
            torch.cuda.empty_cache()
            if config.DATA.DATASET == 'prcc':
                rank1 = test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                rank1 = test(config, model, queryloader, galleryloader, dataset)
            torch.cuda.empty_cache()
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            model_state_dict = model.state_dict()
            save_checkpoint({
                'model_state_dict': model_state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

        start_train_time = time.time()
        train(epoch, model, clip_model, optimizer, trainloader)
        train_time += round(time.time() - start_train_time)
        scheduler.step()

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

if __name__ == '__main__':
    config = parse_option()
    set_seed(config.SEED)
    main(config)