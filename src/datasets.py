import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import RandomRotate90, Transpose, ShiftScaleRotate, Flip, Compose
from albumentations import RandomResizedCrop

import configure


class MelanomaDataset(Dataset):
    def __init__(self, df, image_dir, image_width, image_height, train, transform):
        self.df = df
        self.image_dir = image_dir
        self.image_width = image_width
        self.image_height = image_height
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df['image_name'].values[idx]
        image = cv2.imread(f"{self.image_dir}/{image_name}.jpg")
        image = cv2.resize(image, (self.image_width, self.image_height))

        if self.transform:
            image = self.transform(image=image)['image']

        image = torch.from_numpy(image / 255.0).float()
        image = image.permute(2, 0, 1)

        if self.train:
            target = self.df['target'].values[idx]
            return image, target
        else:
            return image


def get_transforms():
    return Compose([
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=90, p=0.5),
    ])


def get_dataloader(image_dir, image_width, image_height, fold, batch_size, num_workers):
    df_train = pd.read_csv(f'{configure.SPLIT_FOLDER}/fold_{fold}_train.csv')
    df_valid = pd.read_csv(f'{configure.SPLIT_FOLDER}/fold_{fold}_valid.csv')

    train_dataset = MelanomaDataset(df=df_train,
                                    image_dir=image_dir,
                                    image_width=image_width,
                                    image_height=image_height,
                                    train=True,
                                    transform=get_transforms())

    valid_dataset = MelanomaDataset(df=df_valid,
                                    image_dir=image_dir,
                                    image_width=image_width,
                                    image_height=image_height,
                                    train=True,
                                    transform=None)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=True)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=4,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  shuffle=False)

    return train_dataloader, valid_dataloader
