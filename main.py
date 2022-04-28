import argparse
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import init_args, params
import data
import models
from models import *
from utils import utils, torch_utils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validation(args, pr, net, criterion, data_loader, device='cuda'):
    # import pdb; pdb.set_trace()
    net.eval()
    pred_all = torch.tensor([]).to(device)
    target_all = torch.tensor([]).to(device)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Validation"):
            pred, target = predict(args, pr, net, batch, device)
            pred_all = torch.cat([pred_all, pred], dim=0)
            target_all = torch.cat([target_all, target], dim=0)

    res = criterion.evaluate(pred_all, target_all)
    torch.cuda.empty_cache()
    net.train()
    return res


def predict(args, pr, net, batch, device):
    if args.input == 'image': 
        if pr.objective in ['depth_order']:
            input_1, input_2 = batch['img_1'].to(device), batch['img_2'].to(device)
            pred = net(input_1, input_2)   
            target = batch[pr.objective].to(device)  
        elif pr.objective in ['obstacle']:
            input_1 =  batch['img'].to(device)
            pred = net(input_1)
            target = batch[pr.objective].to(device)
    elif args.input == 'audio':
        if pr.objective in ['depth_order']:
            input_1, input_2 = batch['audio_1'].to(device), batch['audio_2'].to(device)
            pred = net(input_1, input_2)
            target = batch[pr.objective].to(device)
        elif pr.objective in ['obstacle']:
            input_1 =  batch['audio'].to(device)
            pred = net(input_1)
            target = batch[pr.objective].to(device)
    elif args.input == 'both':
        img_1, img_2 = batch['img_1'].to(device), batch['img_2'].to(device)
        audio_1, audio_2 = batch['audio_1'].to(device), batch['audio_2'].to(device)
        out = net(audio_1, audio_2, img_1, img_2)
        pred = out['pred']
        target = out['target']
    return pred, target


def train(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- get parameters for audio ----- #
    fn = getattr(params, args.setting)
    pr = fn()
    # ----- make dirs for checkpoints ----- #
    sys.stdout = utils.LoggerOutput(os.path.join('checkpoints', args.exp, 'log.txt'))
    os.makedirs('./checkpoints/' + args.exp, exist_ok=True)

    writer = SummaryWriter(os.path.join('./checkpoints', args.exp, 'visualization'))
    # ------------------------------------- #
    
    tqdm.write('{}'.format(args)) 
    tqdm.write('{}'.format(pr))
    # ------------------------------------ #

    
    # ----- Dataset and Dataloader ----- #
    train_dataset, train_loader = torch_utils.get_dataloader(args, pr, split='train', shuffle=True, drop_last=False)
    val_dataset, val_loader = torch_utils.get_dataloader(args, pr, split='val', shuffle=False, drop_last=False)
    # --------------------------------- #

    # ----- Network ----- #
    net = models.__dict__[pr.net](args, pr, device=device).to(device)
    criterion = models.__dict__[pr.loss](args, pr, device)
    optimizer = torch_utils.make_optimizer(net, args)
    # --------------------- #

    # -------- Loading checkpoints weights ------------- #
    if args.resume:
        resume = './checkpoints/' + args.resume
        net, args.start_epoch = torch_utils.load_model(resume, net, device=device, strict=True)
        if args.resume_optim:
            tqdm.write('loading optimizer...')
            optim_state = torch.load(resume)['optimizer']
            optimizer.load_state_dict(optim_state)
            tqdm.write('loaded optimizer!')
        else:
            args.start_epoch = 0

    # ------------------- 
    net = nn.DataParallel(net, device_ids=gpu_ids)
    #  --------- Random or resume validation ------------ #
    res = validation(args, pr, net, criterion, val_loader, device)
    writer.add_scalars('SFS' + '/validation', res, args.start_epoch)
    tqdm.write("Beginning, Validation results: {}".format(res))
    tqdm.write('\n')

    # ----------------- Training ---------------- #
    # import pdb; pdb.set_trace()
    VALID_STEP = args.valid_step
    for epoch in range(args.start_epoch, args.epochs):
        running_loss = 0.0
        torch_utils.adjust_learning_rate(optimizer, epoch, args, pr)
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            pred, target = predict(args, pr, net, batch, device)
            loss = criterion(pred, target)        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                tqdm.write("Epoch: {}/{}, step: {}/{}, loss: {}".format(epoch+1, args.epochs, step+1, len(train_loader), loss))
                running_loss += loss.item()

            current_step = epoch * len(train_loader) + step + 1
            BOARD_STEP = 3
            if (step+1) % BOARD_STEP == 0:
                writer.add_scalar('SFS' + '/training loss', running_loss / BOARD_STEP, current_step)
                running_loss = 0.0
            
            if (current_step + 1) % VALID_STEP == 0 and args.valid_by_step:
                res = validation(args, pr, net, criterion, val_loader, device)
                writer.add_scalars('SFS' + '/validation-on-Step', res, current_step)
                tqdm.write("Step: {}/{}, Validation results: {}".format(current_step , args.epochs * len(train_loader), res))
        
        
        # ----------- Validtion -------------- #
        if (epoch + 1) % VALID_STEP == 0 and not args.valid_by_step:
            res = validation(args, pr, net, criterion, val_loader, device)
            writer.add_scalars('SFS' + '/validation', res, epoch + 1)
            tqdm.write("Epoch: {}/{}, Validation results: {}".format(epoch + 1, args.epochs, res))

        # ---------- Save model ----------- #
        SAVE_STEP = args.save_step
        if (epoch + 1) % SAVE_STEP == 0:
            path = os.path.join('./checkpoints', args.exp, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar')
            torch.save({'epoch': epoch + 1,
                        'step': current_step,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        },
                        path)
        # --------------------------------- #
    torch.cuda.empty_cache()
    tqdm.write('Training Complete!')
    writer.close()


def test(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- get parameters for audio ----- #
    fn = getattr(params, args.setting)
    pr = fn()
    if args.list_test:
        pr.list_test = args.list_test
    # ----- make dirs for results ----- #
    sys.stdout = utils.LoggerOutput(os.path.join('results', args.exp, 'log.txt'))
    os.makedirs('./results/' + args.exp, exist_ok=True)
    # ------------------------------------- #
    tqdm.write('{}'.format(args)) 
    tqdm.write('{}'.format(pr))
    # ------------------------------------ #
    # ----- Dataset and Dataloader ----- #
    test_dataset, test_loader = torch_utils.get_dataloader(args, pr, split='test', shuffle=False, drop_last=False)
    # --------------------------------- #
    # ----- Network ----- #
    net = models.__dict__[pr.net](args, pr, device=device).to(device)
    criterion = models.__dict__[pr.loss](args, pr, device)
    # -------- Loading checkpoints weights ------------- #
    if args.resume:
        resume = './checkpoints/' + args.resume
        net, _ = torch_utils.load_model(resume, net, device=device, strict=True)

    # ------------------- #
    net = nn.DataParallel(net, device_ids=gpu_ids)
    #  --------- Testing ------------ #
    res = validation(args, pr, net, criterion, test_loader, device)
    tqdm.write("Testing results: {}".format(res))


if __name__ == '__main__':
    args = init_args()
    if args.test_mode:
        test(args, DEVICE)
    else:
        train(args, DEVICE)