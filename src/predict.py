import argparse
import os
import sys
import numpy as np
import warnings
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
import subprocess

from model import MelanomaNet
import datasets
import configure

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--arch', metavar='ARCH', default='efficientnet-b0',
                        help='model architecture (default: efficientnet-b0)')
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_workers", default=24, type=int,
                        help="How many sub-processes to use for data.")
    parser.add_argument("--per_gpu_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--submit",
                        action="store_true",
                        help='if true, prediction will be submitted')

    return parser.parse_args()


def predict(dataloader, model, args):
    with torch.no_grad():
        y_score = []
        for i, images in enumerate(dataloader):
            bs, c, h, w = images.size()
            images = images.to(args.device)

            # dihedral TTA
            images = torch.stack([images, images.flip(-1),
                                  images.flip(-2), images.flip(-1, -2),
                                  images.transpose(-1, -2), images.transpose(-1, -2).flip(-1),
                                  images.transpose(-1, -2).flip(-2), images.transpose(-1, -2).flip(-1, -2)], 1)
            images = images.view(-1, c, h, w)
            output = model(images).view(bs, 8, -1).mean(1).view(-1)

            y_score.append(torch.sigmoid(output.detach()).cpu().numpy())

        y_score = np.concatenate(y_score)

        return y_score


def main():
    args = parse_args()

    # Setup CUDA, GPU
    if not torch.cuda.is_available():
        print("cuda is not available")
        exit(0)
    else:
        args.device = torch.device("cuda")

    # Setup model
    model = MelanomaNet(arch=args.arch, pretrained=False)
    state_dict = torch.load(f'{configure.MODEL_PATH}/{args.arch}_fold_{args.fold}.pth')
    valid_score = state_dict['valid_score']
    model.load_state_dict(state_dict['state_dict'])
    model.to(args.device)
    model.eval()

    # Setup data
    df = pd.read_csv(configure.TEST_DF)
    test_dataset = datasets.MelanomaDataset(df=df,
                                            image_dir=configure.TEST_IMAGE_PATH,
                                            train=False,
                                            transform=None)

    valid_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=args.per_gpu_batch_size,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  shuffle=False)

    df['target'] = predict(valid_dataloader, model, args=args)
    df['target'] = df['target'].astype(float)
    df = df[['image_name', 'target']]

    current_time = datetime.now().strftime('%b%d_%H_%M_%S')
    filename = f'{configure.SUBMISSION_PATH}/{args.arch}_fold_{args.fold}_valid_{valid_score:0.3f}_{current_time}.csv'
    df.to_csv(filename, index=False)

    if args.submit:
        # submit the prediction to kaggle
        subprocess.run(["kaggle",
                        "competitions",
                        "submit",
                        "-c", "siim-isic-melanoma-classification",
                        "-f", filename,
                        "-m", current_time])


if __name__ == "__main__":
    main()
