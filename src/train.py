import argparse
import os
import sys
import numpy as np
import warnings
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model import MelanomaNet
import datasets
import configure
import utils

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--arch', metavar='ARCH', default='efficientnet-b0',
                        help='model architecture (default: efficientnet-b0)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_workers", default=36, type=int,
                        help="How many sub-processes to use for data.")
    parser.add_argument("--per_gpu_batch_size", default=6, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--image_width", default=256, type=int)
    parser.add_argument("--image_height", default=256, type=int)
    parser.add_argument("--log",
                        action="store_true",
                        help='write training history')
    parser.add_argument("--resume",
                        action="store_true",
                        help='training model from check point')
    parser.add_argument("--epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()


def train(dataloader, model, criterion, optimizer, args):
    model.train()

    train_loss = 0.0
    for i, (images, target) in enumerate(dataloader):
        images = images.to(args.device)
        target = target.to(args.device)
        output = model(images)

        loss = criterion(output.view(-1), target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(dataloader)

    return train_loss


def valid(dataloader, model, criterion, args):
    model.eval()

    with torch.no_grad():
        valid_loss = 0.0
        y_true, y_score = [], []
        for i, (images, target) in enumerate(dataloader):
            bs, c, h, w = images.size()
            images = images.to(args.device)
            target = target.to(args.device)

            # dihedral TTA
            images = torch.stack([images, images.flip(-1),
                                  images.flip(-2), images.flip(-1, -2),
                                  images.transpose(-1, -2), images.transpose(-1, -2).flip(-1),
                                  images.transpose(-1, -2).flip(-2), images.transpose(-1, -2).flip(-1, -2)], 1)
            images = images.view(-1, c, h, w)

            output = model(images).view(bs, 8, -1).mean(1).view(-1)
            loss = criterion(output, target.float())
            valid_loss += loss.item() / len(dataloader)

            y_true.append(target.detach().cpu().numpy())
            y_score.append(torch.sigmoid(output.detach()).cpu().numpy())

        y_score = np.concatenate(y_score)
        y_true = np.concatenate(y_true)

        return valid_loss, y_true, y_score


def main():
    args = parse_args()

    # set random seed
    utils.seed_torch(args.seed)

    # Setup CUDA, GPU
    if not torch.cuda.is_available():
        print("cuda is not available")
        exit(0)
    else:
        args.device = torch.device("cuda")
        args.n_gpus = torch.cuda.device_count()
        print(f"available cuda: {args.n_gpus}")

    # Setup model
    model = MelanomaNet(arch=args.arch)
    model_path = f'{configure.MODEL_PATH}/{args.arch}_fold_{args.fold}.pth'

    if args.resume:
        assert os.path.exists(model_path), "checkpoint does not exist"
        state_dict = torch.load(model_path)
        valid_score = state_dict['valid_score']
        threshold = state_dict['threshold']
        print(f"load model from checkpoint, threshold: {threshold}, valid score: {state_dict['valid_score']:0.3f}")
        model.load_state_dict(state_dict['state_dict'])
        best_score = valid_score
        args.learning_rate = 3e-05
    else:
        best_score = 0.0

    if args.n_gpus > 1:
        model = torch.nn.DataParallel(module=model)
    model.to(args.device)

    # Setup data
    total_batch_size = args.per_gpu_batch_size * args.n_gpus
    train_loader, valid_loader = datasets.get_dataloader(
        fold=args.fold,
        batch_size=total_batch_size,
        num_workers=args.num_workers)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    """ Train the model """
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_prefix = f'{current_time}_{args.arch}_fold_{args.fold}_{args.patch_size}_{args.num_patches}'
    log_dir = os.path.join(configure.TRAINING_LOG_PATH,
                           log_prefix)

    tb_writer = None
    if args.log:
        tb_writer = SummaryWriter(log_dir=log_dir)

    print(f'training started: {current_time}')
    for epoch in range(args.epochs):
        train_loss = train(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            args=args)

        valid_loss, y_true, y_score = valid(
            dataloader=valid_loader,
            model=model,
            criterion=criterion,
            args=args)

        valid_score = roc_auc_score(y_true=y_true, y_score=y_score)

        learning_rate = scheduler.get_lr()[0]
        if args.log:
            tb_writer.add_scalar("learning_rate", learning_rate, epoch)
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/valid", valid_loss, epoch)
            tb_writer.add_scalar("Score/valid", valid_score, epoch)

            # Log the confusion matrix as an image summary.
            figure = utils.plot_roc_curve(y_true=y_true, y_score=y_score)
            figure = utils.plot_to_image(figure)
            tb_writer.add_image("ROC curve", figure, epoch)

        if valid_score > best_score:
            best_score = valid_score
            state = {'state_dict': model.module.state_dict(),
                     'train_loss': train_loss,
                     'valid_loss': valid_loss,
                     'valid_score': valid_score}
            torch.save(state, model_path)

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        print(f"epoch:{epoch:02d}, "
              f"train:{train_loss:0.3f}, valid:{valid_loss:0.3f}, "
              f"score:{valid_score:0.3f}, best:{best_score:0.3f}, date:{current_time}")

        scheduler.step()

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    print(f'training finished: {current_time}')

    if args.log:
        tb_writer.close()


if __name__ == "__main__":
    main()
