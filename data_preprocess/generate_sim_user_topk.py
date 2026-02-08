"""
Generate top-k similar users based on LLM user embedding matrix (cosine similarity).

Input:
  ../dataset/{dataset_name}/{llm_model}_user_emb_matrix.pickle
Output:
  ../dataset/{dataset_name}/sim_user_{topk}.pkl

The output is a dict:
  sim_user[u] = [u1, u2, ..., u_topk]   (exclude self)

Notes:
- Many SR pipelines reserve row 0 as padding user (all zeros). We auto-detect and exclude it.
- Recommend installing faiss for speed:
    pip install faiss-cpu
"""

import os
import sys
import argparse
import pickle
import numpy as np


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def to_numpy_matrix(x) -> np.ndarray:
    # support np.ndarray / list / torch tensor (if user accidentally saved tensor)
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={x.shape}")
    return x.astype(np.float32, copy=False)


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + eps)


def detect_user_id_start(U: np.ndarray, eps: float = 1e-8) -> int:
    """
    If row0 is (almost) all zeros => treat as padding and start user_id from 1.
    Else start from 0.
    """
    row0_norm = float(np.linalg.norm(U[0]))
    if row0_norm < eps:
        return 1
    return 0


def build_sim_users_faiss(X: np.ndarray, valid_user_ids: np.ndarray, topk: int) -> dict:
    """
    X: (N, D) normalized float32 vectors for valid users only
    valid_user_ids: (N,) mapping from X row index -> actual user_id
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise ImportError("FAISS not available") from e

    D = X.shape[1]
    index = faiss.IndexFlatIP(D)  # cosine == inner product after normalization
    index.add(X)

    # query all
    k_search = min(topk + 1, X.shape[0])  # +1 to drop self
    _, I = index.search(X, k_search)      # I: (N, k_search), indices in [0..N-1]

    sim_user = {}
    for i in range(X.shape[0]):
        u = int(valid_user_ids[i])
        nbrs = []
        for j in I[i]:
            v = int(valid_user_ids[int(j)])
            if v == u:
                continue
            nbrs.append(v)
            if len(nbrs) >= topk:
                break
        sim_user[u] = nbrs
    return sim_user


def build_sim_users_numpy_block(X: np.ndarray, valid_user_ids: np.ndarray, topk: int, block_size: int) -> dict:
    """
    Blocked brute-force cosine top-k using matrix multiply, avoids full NxN storage.
    Complexity still O(N^2), but memory-safe.

    X must be L2-normalized float32. Cosine = X @ X.T
    """
    N, D = X.shape
    sim_user = {}

    XT = X.T  # (D, N)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Q = X[start:end]              # (B, D)
        sims = Q @ XT                 # (B, N)

        # exclude self: sims[r, start+r] = -inf
        rows = np.arange(end - start)
        cols = start + rows
        sims[rows, cols] = -np.inf

        # take topk indices (unsorted) then sort by similarity
        k_eff = min(topk, N - 1)
        idx_part = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]  # (B, k_eff)

        # sort within topk
        part_scores = np.take_along_axis(sims, idx_part, axis=1)             # (B, k_eff)
        order = np.argsort(-part_scores, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)             # (B, k_eff)

        for r in range(end - start):
            u = int(valid_user_ids[start + r])
            nbr_ids = valid_user_ids[idx_sorted[r]].astype(np.int64).tolist()
            # already excludes self via -inf, but keep safe:
            nbr_ids = [v for v in nbr_ids if v != u][:topk]
            sim_user[u] = nbr_ids

    return sim_user


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Electronics", help="Beauty, Sports_and_Outdoors, Toys_and_Games, Electronics")
    parser.add_argument("--dataset_dir", type=str, default="../dataset",
                        help="Dataset root dir. If running inside data_preprocess, use ../dataset")
    parser.add_argument("--llm_model", type=str, default="qwen3", help="prefix for user embedding file name")
    parser.add_argument("--user_emb_matrix_file", type=str, default="",
                        help="Optional override. If empty, use {llm_model}_user_emb_matrix.pickle")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "faiss", "numpy"],
                        help="auto: try faiss then fallback to numpy")
    parser.add_argument("--block_size", type=int, default=256, help="Only for numpy method")
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_dir, args.dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"dataset folder not found: {dataset_path}")

    emb_file = args.user_emb_matrix_file.strip()
    if emb_file == "":
        emb_file = f"{args.llm_model}_user_emb_matrix.pickle"
    emb_path = os.path.join(dataset_path, emb_file)

    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"user embedding matrix not found: {emb_path}")

    print(f"[INFO] Loading user embedding matrix from: {emb_path}")
    U = to_numpy_matrix(load_pickle(emb_path))
    print(f"[INFO] U shape = {U.shape}, dtype = {U.dtype}")

    # detect padding row
    start_uid = detect_user_id_start(U)
    if start_uid == 1:
        print("[INFO] Detected padding row at user_id=0 (row0 ~ zeros). Will exclude it.")
    else:
        print("[INFO] No obvious padding row. Will include user_id=0 as a valid user.")

    valid_user_ids = np.arange(start_uid, U.shape[0], dtype=np.int64)
    X = U[valid_user_ids]  # (Nvalid, D)
    if X.shape[0] <= 1:
        raise ValueError(f"Not enough users to retrieve neighbors. valid_users={X.shape[0]}")

    # normalize for cosine
    X = l2_normalize(X)

    # compute sim users
    sim_user = None
    used_method = None

    if args.method in ("auto", "faiss"):
        try:
            print("[INFO] Trying FAISS method...")
            sim_user = build_sim_users_faiss(X, valid_user_ids, args.topk)
            used_method = "faiss"
        except Exception as e:
            if args.method == "faiss":
                raise
            print(f"[WARN] FAISS not available or failed ({e}). Falling back to numpy block method...")

    if sim_user is None:
        print("[INFO] Using numpy block method (may be slow for large user counts)...")
        sim_user = build_sim_users_numpy_block(X, valid_user_ids, args.topk, args.block_size)
        used_method = "numpy"

    # out_path = os.path.join(dataset_path, f"sim_user_{args.topk}.pkl")
    # save_pickle(sim_user, out_path)
    # print(f"[INFO] Done. method={used_method}, saved to: {out_path}")
    #
    # # quick sanity check
    # some_u = int(valid_user_ids[min(1, len(valid_user_ids) - 1)])
    # print(f"[INFO] Example user {some_u} top{args.topk} neighbors (first 10): {sim_user[some_u][:10]}")
    # print("[INFO] Sanity: all lists should have length topk (unless user count is too small).")
    # -------- dict -> ndarray --------
    user_num = U.shape[0]
    K = args.topk
    sim_mat = np.zeros((user_num, K), dtype=np.int64)

    # 1) 对所有 user 先默认填自己（避免空行）
    for u in range(user_num):
        sim_mat[u, :] = u

    # 2) 只填有效用户的检索结果（valid_user_ids 从 start_uid 开始）
    for u in valid_user_ids:
        u = int(u)
        nbrs = sim_user.get(u, None)
        if nbrs is None:
            continue

        # 保险：确保长度是 K
        nbrs = list(nbrs)[:K]
        if len(nbrs) < K:
            nbrs = nbrs + [nbrs[-1] if len(nbrs) > 0 else u] * (K - len(nbrs))

        sim_mat[u, :] = np.asarray(nbrs, dtype=np.int64)

    # 可选：如果 start_uid==1（0 是 padding user），明确让 0 行全 0
    if start_uid == 1:
        sim_mat[0, :] = 0

    # -------- save ndarray --------
    out_path = os.path.join(dataset_path, f"sim_user_{K}.pkl")
    save_pickle(sim_mat, out_path)
    print(f"[INFO] Done. method={used_method}, saved ndarray to: {out_path}")
    print(f"[INFO] sim_mat shape={sim_mat.shape}, dtype={sim_mat.dtype}")

    # quick sanity check
    some_u = int(valid_user_ids[min(1, len(valid_user_ids) - 1)])
    print(f"[INFO] Example user {some_u} top{K} neighbors (first 10): {sim_mat[some_u, :10].tolist()}")


if __name__ == "__main__":
    main()
