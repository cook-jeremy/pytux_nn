import torch
import torch.utils.tensorboard as tb
import numpy as np
from .dataloader import load_data
from torch.utils.data import Dataset, DataLoader
from .model import PuckDetector, save_model

def train(args):
    from os import path
    model = PuckDetector()
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'puck.th')))

    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    #train_data = load_data(num_workers=args.num_workers)
    train_data = load_data(num_workers=4, batch_size=64)

    print('STARTING TRAINING')

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        losses = []
        for img, label in train_data:

            img, label = img.to(device), label.to(device)

            pred = model(img)
            loss_val = loss(pred, label)

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
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    import torchvision
    fig, ax = plt.subplots(1, 1)
    im = TF.to_pil_image(img[0].cpu())
    resize = torchvision.transforms.Resize([300, 400])
    im = resize(im)
    ax.imshow(im)
    print(tuple(label[0].cpu().detach().numpy()))
    print(tuple(pred[0].cpu().detach().numpy()))
    ax.add_artist(plt.Circle(tuple(label[0].cpu().detach().numpy()), 10, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(tuple(pred[0].cpu().detach().numpy()), 10, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


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
