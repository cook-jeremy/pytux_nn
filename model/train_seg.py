import torch
import torch.utils.tensorboard as tb
import numpy as np
from dataloader import load_seg_data
from torch.utils.data import Dataset, DataLoader
from puck_seg import PuckSeg, save_model
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision
import os
from os import path
import shutil

def train(args):
    model = PuckSeg()
    train_logger = None
    if args.log_dir is not None:
        train_path = path.join(args.log_dir, 'train')
        if os.path.exists(train_path):
            shutil.rmtree(train_path)
        train_logger = tb.SummaryWriter(train_path)

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'puck_seg.th')))

    #loss = torch.nn.L1Loss()
    loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    
    #train_data = load_data(num_workers=args.num_workers)
    BATCH_SIZE = 32
    train_data = load_seg_data(num_workers=args.num_workers, batch_size=BATCH_SIZE)

    print('STARTING TRAINING')

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        losses = []
        for img, label in train_data:
            
            img, label = img.to(device), label.to(device)

            pred = model(img)
            
            loss_val = loss(pred, label)
            loss_val[label == 1] *= 100
            loss_val = loss_val.mean()

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                if global_step % 50 == 0:
                    log(train_logger, img, label, pred, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            
            losses.append(loss_val.detach().cpu().numpy())
        
        avg_loss = np.mean(losses)
        
        print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
        save_model(model)

    save_model(model)

def log(logger, img, label, pred, global_step):
    fig, ax = plt.subplots(1, 1)
    im = TF.to_pil_image(img[0].cpu())
    ax.imshow(im)
    logger.add_figure('image', fig, global_step)
    del ax, fig

    # fig, ax = plt.subplots(1, 1)
    # pred_im = TF.to_pil_image(pred.cpu().detach().numpy()[0][0])
    # ax.imshow(pred_im)
    # logger.add_figure('pred', fig, global_step)
    # del ax, fig

    label_im = label.cpu().detach()[0][0]
    label_grid = (torchvision.utils.make_grid(label_im) * 255).byte()
    logger.add_image('label', label_grid, global_step=global_step)

    pred_im = pred.cpu().detach()[0][0]
    pred_grid = (torchvision.utils.make_grid(pred_im) * 255).byte()
    logger.add_image('pred', pred_grid, global_step=global_step)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    #parser.add_argument('-t', '--transform', default='')

    args = parser.parse_args()
    train(args)
