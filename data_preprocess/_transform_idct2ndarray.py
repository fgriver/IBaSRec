#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
_transform_idct2ndarray.py

将 dict 格式的 embedding（{item_id: embedding_list}）转换为
numpy.ndarray 格式，shape = (item_num, emb_dim)。

dataset:Beauty, Sports_and_Outdoors, Toys_and_Games, ml-1m, Electronics

用法示例：
    python _transform_idct2ndarray.py \
        --input_path ../dataset/Electronics/qwen3_emb.pickle \
        --output_path ../dataset/Electronics/qwen3_emb_matrix.pickle
"""

import os
import argparse
import pickle
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Transform dict embedding → ndarray embedding")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Input pickle file (dict format).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output pickle file (ndarray format).")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[Info] Loading dict embedding from: {args.input_path}")
    with open(args.input_path, "rb") as f:
        emb_dict = pickle.load(f)

    if not isinstance(emb_dict, dict):
        raise ValueError("[Error] Input embedding must be dict {item_id: embedding_vector}")

    print(f"[Info] Total items in dict: {len(emb_dict)}")

    # ---- 1. 排序 key ----
    item_ids = sorted(emb_dict.keys())
    print(f"[Info] item_id range: {item_ids[0]} → {item_ids[-1]}")

    # ---- 2. 检查连续性 ----
    expected_last = len(item_ids) - 1
    if item_ids[-1] != expected_last:
        print("[Warning] item_id 不连续！")
        print(f"  最大 item_id = {item_ids[-1]}, 但总数 = {len(item_ids)}")
        print("  缺失的 embedding 会被默认填 0")

    # ---- 3. 构造 ndarray ----
    emb_dim = len(emb_dict[item_ids[0]])
    item_num = item_ids[-1] + 1  # 即使缺失也要容纳到最大 id

    emb_matrix = np.zeros((item_num, emb_dim), dtype=np.float32)

    for item_id in item_ids:
        emb_matrix[item_id] = np.asarray(emb_dict[item_id], dtype=np.float32)

    print(f"[Info] Output ndarray shape: {emb_matrix.shape}")

    # ---- 4. 保存 ndarray ----
    with open(args.output_path, "wb") as f:
        pickle.dump(emb_matrix, f)

    print(f"[Info] Saved ndarray embedding to: {args.output_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
