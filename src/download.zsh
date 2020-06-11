#!/usr/local_rwth/bin/zsh

#SBATCH -J download
#SBATCH -o ./cluster_out/download.txt
#SBATCH -e ./cluster_err/download.txt

#SBATCH -t 5:00:00 --mem=10G
#SBATCH -A rwth0429

source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

kaggle competitions download siim-isic-melanoma-classification -p /home/rwth0455/kaggle
