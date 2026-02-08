#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_user_embedding_LOO_amazon.py

Generate *user* semantic embeddings (u_llm) for LLM-ESR style retrieval / distillation,
based on users' historical interaction sequences + item titles (from item_text.csv).

This script is intentionally written to match your current Amazon LOO preprocessing outputs:
  <dataset_dir>/<dataset_name>/
    - interactions.csv            (columns: user_id, item_id, unixReviewTime)
    - item_text.csv               (columns: item_id, title, description, category)
    - (optional) train_data.df / val_data.df / test_data.df

We build each user's "history text" from their *training history*:
  history = full_sequence_excluding_last2 = seq[:-2]
which matches the common LOO split used in many codebases (train/valid/test = -2/-1 holdout).

Outputs:
  1) {llm_model}_user_emb.pickle  : dict {user_id(int): embedding(list[float])}
  2) {llm_model}_user_emb_matrix.pickle (optional) : np.ndarray (user_num, emb_dim) with row=user_id
     (created with --also_save_matrix)

Requires:
  pip install dashscope pandas tqdm numpy

Example:
  export DASHSCOPE_API_KEY=sk-f8d81c60072d45abbc150849b431044f

  nohup python generate_user_embedding_LOO_amazon.py \
      --dataset_name Electronics \
      --dataset_dir ../dataset \
      --llm_model qwen3 \
      --embedding_model text-embedding-v4 \
      --max_workers 8 \
      --also_save_matrix \
      > Electronics_generate_user_emb.log 2>&1 &
"""

import os
import time
import json
import pickle
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import dashscope
from dashscope import TextEmbedding


# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--llm_model", type=str, default="qwen3",
                   help="Logical name only used for output filename prefix.")
    p.add_argument("--embedding_model", type=str, default="text-embedding-v4",
                   help="DashScope embedding model id.")
    p.add_argument("--max_workers", type=int, default=5)
    p.add_argument("--output_name", type=str, default=None,
                   help="Output pickle filename for dict. Default: {llm_model}_user_emb.pickle")
    p.add_argument("--also_save_matrix", action="store_true",
                   help="Also save {llm_model}_user_emb_matrix.pickle as ndarray (row=user_id).")

    # Prompt controls
    p.add_argument("--template", type=str,
                   default="The user has interacted with the following items:\n<HISTORY>\nPlease conclude the user's preference.",
                   help="Prompt template. Must include <HISTORY> placeholder.")
    p.add_argument("--max_hist_chars", type=int, default=8000,
                   help="Max length of <HISTORY> string after concatenation.")
    p.add_argument("--max_prompt_chars", type=int, default=4096,
                   help="Max length of final prompt text sent to embedding API.")

    # Rate limit
    p.add_argument("--rate_limit_qps", type=float, default=2.0,
                   help="Global QPS cap across threads (important to avoid 429).")

    return p.parse_args()


# ----------------------------
# Global rate limiter (same style as your item script)
# ----------------------------
_rate_lock = threading.Lock()
_next_available_time = 0.0


def rate_limited_call(rate_limit_qps: float, **kwargs):
    global _next_available_time
    while True:
        with _rate_lock:
            now = time.time()
            wait = _next_available_time - now
            if wait <= 0:
                rsp = TextEmbedding.call(**kwargs)
                _next_available_time = max(now, time.time()) + 1.0 / max(rate_limit_qps, 1e-6)
                return rsp
        time.sleep(wait if wait > 0 else 0.01)


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def build_user_prompt(
    item_ids: list[int],
    itemid2title: dict[int, str],
    template: str,
    max_hist_chars: int,
) -> str | None:
    titles = []
    for it in item_ids:
        t = itemid2title.get(int(it), "")
        t = safe_str(t)
        if t != "":
            titles.append(t)

    if len(titles) == 0:
        return None

    hist_str = ", ".join(titles)
    if len(hist_str) > max_hist_chars:
        # keep the most recent part
        hist_str = hist_str[-max_hist_chars:]

    return template.replace("<HISTORY>", hist_str)


def fetch_embedding(user_id: int, prompt: str, embed_model: str, rate_limit_qps: float, retry: int = 5):
    # Note: DashScope embedding API can accept long text, but we still clip by caller.
    for attempt in range(retry):
        try:
            rsp = rate_limited_call(
                rate_limit_qps,
                model=embed_model,
                input=prompt,
                text_type="document",
            )

            status_code = getattr(rsp, "status_code", 200)
            code = getattr(rsp, "code", "")
            message = getattr(rsp, "message", "")

            if status_code == 200:
                out = getattr(rsp, "output", None)
                if (
                    out is None
                    or "embeddings" not in out
                    or not isinstance(out["embeddings"], list)
                    or len(out["embeddings"]) == 0
                    or "embedding" not in out["embeddings"][0]
                ):
                    raise RuntimeError(f"Invalid embedding response structure: {repr(rsp)}")
                emb = out["embeddings"][0]["embedding"]
                return user_id, emb

            if status_code == 429:
                backoff = 2.0 * (attempt + 1)
                print(f"[Throttle] user {user_id}, attempt {attempt+1}/{retry}, code={code}, msg={message}, backoff={backoff:.1f}s")
                time.sleep(backoff)
                continue

            print(f"[Error] user {user_id}, status_code={status_code}, code={code}, msg={message}")
            return user_id, None

        except Exception as e:
            print(f"[Error] user {user_id}, attempt {attempt+1}/{retry}: {e}")
            time.sleep(1.0)

    return user_id, None


def main():
    args = parse_args()

    if "DASHSCOPE_API_KEY" not in os.environ:
        raise ValueError("Please set environment variable DASHSCOPE_API_KEY first, e.g., export DASHSCOPE_API_KEY=xxxx")

    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

    dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
    inter_path = os.path.join(dataset_dir, "interactions.csv")
    item_text_path = os.path.join(dataset_dir, "item_text.csv")

    out_name = args.output_name or f"{args.llm_model}_user_emb.pickle"
    out_path = os.path.join(dataset_dir, out_name)

    print(f"[Info] Loading interactions: {inter_path}")
    inter_df = pd.read_csv(inter_path)
    need_cols = {"user_id", "item_id", "unixReviewTime"}
    if not need_cols.issubset(set(inter_df.columns)):
        raise ValueError(f"interactions.csv must include {need_cols}, got={list(inter_df.columns)}")

    inter_df = inter_df.sort_values(["user_id", "unixReviewTime"]).reset_index(drop=True)

    print(f"[Info] Loading item_text: {item_text_path}")
    item_df = pd.read_csv(item_text_path)
    if "item_id" not in item_df.columns or "title" not in item_df.columns:
        raise ValueError(f"item_text.csv must include columns ['item_id','title'], got={list(item_df.columns)}")

    item_df["item_id"] = item_df["item_id"].astype(int)
    itemid2title = dict(zip(item_df["item_id"].tolist(), item_df["title"].tolist()))

    # Build user sequences
    user2seq: dict[int, list[int]] = {}
    for r in inter_df.itertuples(index=False):
        user2seq.setdefault(int(r.user_id), []).append(int(r.item_id))

    # Build prompts
    user_prompts = {}
    skipped = 0
    for u, seq in user2seq.items():
        if len(seq) < 3:
            skipped += 1
            continue
        hist = seq[:-2]  # IMPORTANT: exclude (valid,test)
        prompt = build_user_prompt(hist, itemid2title, args.template, args.max_hist_chars)
        if prompt is None:
            skipped += 1
            continue
        if len(prompt) > args.max_prompt_chars:
            prompt = prompt[: args.max_prompt_chars - 1]
        user_prompts[u] = prompt

    print(f"[Info] Built prompts for {len(user_prompts)} users (skipped {skipped}).")

    # Resume cache if exists
    user_emb = {}
    if os.path.exists(out_path):
        try:
            user_emb = pickle.load(open(out_path, "rb"))
            if not isinstance(user_emb, dict):
                print("[Warn] Existing output is not a dict; ignoring it.")
                user_emb = {}
            else:
                print(f"[Info] Resuming from cache: {out_path} (already have {len(user_emb)} users)")
        except Exception as e:
            print(f"[Warn] Failed to load cache {out_path}: {e}")

    # Multithread embedding
    todo = [(u, user_prompts[u]) for u in user_prompts.keys() if u not in user_emb]
    print(f"[Info] Need to embed {len(todo)} users (total {len(user_prompts)}).")

    if len(todo) > 0:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [
                ex.submit(fetch_embedding, u, prompt, args.embedding_model, args.rate_limit_qps)
                for u, prompt in todo
            ]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                u, emb = fut.result()
                if emb is not None:
                    user_emb[int(u)] = emb

                # periodic save (every ~200 successes)
                if len(user_emb) % 200 == 0:
                    with open(out_path, "wb") as f:
                        pickle.dump(user_emb, f)

    # Final save
    with open(out_path, "wb") as f:
        pickle.dump(user_emb, f)

    print(f"[OK] Saved dict user embeddings: {out_path}  (count={len(user_emb)})")

    if args.also_save_matrix:
        # infer user_num from max user_id
        user_ids = sorted(user_emb.keys())
        if len(user_ids) == 0:
            raise RuntimeError("No user embeddings generated. Cannot save matrix.")
        emb_dim = len(user_emb[user_ids[0]])
        user_num = max(user_ids) + 1
        mat = np.zeros((user_num, emb_dim), dtype=np.float32)
        for u in user_ids:
            mat[int(u)] = np.asarray(user_emb[u], dtype=np.float32)
        mat_path = os.path.join(dataset_dir, f"{args.llm_model}_user_emb_matrix.pickle")
        with open(mat_path, "wb") as f:
            pickle.dump(mat, f)
        print(f"[OK] Saved ndarray user embedding matrix: {mat_path}  shape={mat.shape}")


if __name__ == "__main__":
    main()
