# input data configuration
DATA_PATH = "/home/rwth0455/kaggle/siim-isic-melanoma-classification"
TRAIN_IMAGE_PATH = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/jpeg/train"
TEST_IMAGE_PATH = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/jpeg/test"
TRAIN_DF = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/train.csv"
TEST_DF = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/test.csv"

# 5-fold split data
SPLIT_FOLDER = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/split"

# model weights configuration
MODEL_PATH = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/model"

# training log
TRAINING_LOG_PATH = "/home/rs619065/kaggle/Melanoma/log"

config = {'efficientnet-b0': {'batch_size': 128,
                              'image_width': 224,
                              'image_height': 224}}
