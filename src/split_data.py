import os
import pandas as pd
from sklearn.model_selection import GroupKFold
import configure

df = pd.read_csv(configure.TRAIN_DF)

print(f"total image: {len(df)}")

groups_by_patient_id_list = df['patient_id'].copy().tolist()

y_labels = df["target"].values

gkf = GroupKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(gkf.split(df, y_labels, groups=groups_by_patient_id_list)):
    df_train, df_valid = df.iloc[train_index], df.iloc[valid_index]
    df_train.to_csv(os.path.join(configure.SPLIT_FOLDER, "fold_{}_train.csv".format(i)))
    df_valid.to_csv(os.path.join(configure.SPLIT_FOLDER, "fold_{}_valid.csv".format(i)))
