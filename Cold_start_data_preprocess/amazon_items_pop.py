import os
import argparse
import numpy as np
import pandas as pd


def build_items_pop(interactions_path: str, dataset: str, out_dir: str, item_col: str = "item_id"):
    interactions_path = os.path.join(interactions_path, f"{dataset}/interactions.csv")
    out_dir = os.path.join(out_dir, f"{dataset}")

    df = pd.read_csv(interactions_path)

    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' not found in {interactions_path}. "
                         f"Found columns: {list(df.columns)}")

    # item_id 计数（频次）
    counts = df[item_col].value_counts(sort=False)  # index=item_id, value=count

    # 推断 item_num（假设 item_id 是 [0..max_id] 的整数）
    max_id = int(counts.index.max())
    item_num = max_id + 1

    items_pop = np.zeros(item_num, dtype=np.float32)
    items_pop[counts.index.astype(int).to_numpy()] = counts.to_numpy(dtype=np.float32)

    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, "items_pop.npy")
    np.save(npy_path, items_pop)

    # 可选：保存 csv 方便查看
    pop_df = pd.DataFrame({"item_id": np.arange(item_num), "pop": items_pop.astype(np.int64)})
    csv_path = os.path.join(out_dir, "items_pop.csv")
    pop_df.to_csv(csv_path, index=False)

    # 打印一些统计信息
    print(f"[OK] Saved: {npy_path}")
    print(f"[OK] Saved: {csv_path}")
    print(f"[Stats] item_num={item_num}, total_interactions={len(df)}")
    print(f"[Stats] min_pop={items_pop.min()}, max_pop={items_pop.max()}, mean_pop={items_pop.mean():.4f}")
    print(f"[Stats] zero_pop_items={(items_pop == 0).sum()} (should be 0 if ids are compact)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_path", type=str,
                        default='../cold_start_dataset',
                        help="Path to interactions.csv")
    parser.add_argument('--dataset', type=str,
                        default='Toys_and_Games',
                        help="Dataset name: Beauty/Sports_and_Outdoors/Toys_and_Games")
    parser.add_argument("--out_dir", type=str,
                        default='../cold_start_dataset',
                        help="Directory to save items_pop.npy")
    parser.add_argument("--item_col", type=str, default="item_id",
                        help="Item id column name (default: item_id)")
    args = parser.parse_args()

    build_items_pop(args.interaction_path, args.dataset, args.out_dir, args.item_col)
