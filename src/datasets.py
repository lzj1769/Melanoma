import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)

import configure


class MelanomaDataset(Dataset):
    def __init__(self, df, image_dir, train, transform):
        self.df = df
        self.image_dir = image_dir
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df['image_name'].values[idx]
        image = cv2.imread(f"{self.image_dir}/{image_name}.jpg", cv2.IMREAD_COLOR)

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
    Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=90, p=0.5),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=0.8)


def get_dataloader(image_dir, fold, batch_size, num_workers):
    df_fold = pd.read_csv(configure.FOLDER_DF)

    df_train = df_fold[df_fold['fold'] != fold]
    df_valid = df_fold[df_fold['fold'] == fold]

    print(f"training images:{len(df_train)}")
    print(f"validation images:{len(df_valid)}")

    train_dataset = MelanomaDataset(df=df_train,
                                    image_dir=image_dir,
                                    train=True,
                                    transform=get_transforms())

    valid_dataset = MelanomaDataset(df=df_valid,
                                    image_dir=image_dir,
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
