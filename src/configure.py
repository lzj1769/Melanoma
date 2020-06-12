# input data configuration
DATA_PATH = "/home/rwth0455/kaggle/siim-isic-melanoma-classification"
TRAIN_IMAGE_PATH = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/512x512-dataset-melanoma"
TEST_IMAGE_PATH = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/512x512-test"
TRAIN_DF = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/train.csv"
TEST_DF = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/test.csv"

# 5-fold split data
FOLDER_DF = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/folds_08062020.csv"

# model weights configuration
MODEL_PATH = "/home/rwth0455/kaggle/siim-isic-melanoma-classification/model"

# training log
TRAINING_LOG_PATH = "/home/rs619065/kaggle/Melanoma/log"

# submission directory
SUBMISSION_PATH = "/home/rs619065/kaggle/Melanoma/submission"

config = {'efficientnet-b0': {'batch_size': 128},
          'efficientnet-b1': {'batch_size': 16}}

