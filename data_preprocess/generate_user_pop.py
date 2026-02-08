import pandas as pd
import numpy as np
import os
import argparse


def generate_user_pop(dataset_name, dataset_dir):
    data_path = os.path.join(dataset_dir, dataset_name)

    train_df = pd.read_pickle(os.path.join(data_path, 'train_data.df'))

    # 获取最大 user id
    max_user_id = train_df['user_id'].max()
    user_pop = np.zeros(max_user_id + 1, dtype=np.int32)

    for _, row in train_df.iterrows():
        uid = int(row['user_id'])
        length = row['len_seq']
        user_pop[uid] = length

    save_path = os.path.join(data_path, 'users_pop.npy')
    np.save(save_path, user_pop)
    print(f"Saved users_pop.npy to {save_path}")
    print(f"Stats: Max={user_pop.max()}, Mean={user_pop.mean():.2f}")

if __name__ == "__main__":
    # Dataset: Beauty/Sports_and_Outdoors/Toys_and_Games/Electronics
    generate_user_pop("Electronics", "../dataset")