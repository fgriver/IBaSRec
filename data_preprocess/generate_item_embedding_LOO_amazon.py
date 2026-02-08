#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_item_embedding_LOO_amazon.py

多线程调用 Qwen text-embedding-v4，为 item 生成 semantic embedding，
并保存为 {llm_model}_emb.pickle 或自定义文件名。
['Sports_and_Outdoors', 'Beauty', 'Toys_and_Games', 'Yelp', 'Electronics']
使用方式示例：

    export DASHSCOPE_API_KEY=sk-f8d81c60072d45abbc150849b431044f

    nohup python generate_item_embedding_LOO_amazon.py \
        --dataset_name Electronics \
        --dataset_dir ../dataset \
        --llm_model qwen3 \
        --embedding_model text-embedding-v4 \
        --max_workers 10 \
        --output_name qwen3_emb.pickle \
        > Electronics_generate_log.txt 2>&1 &
"""

import os
import time
import pickle
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
from tqdm import tqdm

import dashscope
from dashscope import TextEmbedding


# ============================================================
# 0. 解析命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate item semantic embeddings with Qwen.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name, e.g., Beauty / Sports / Toys."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Root directory where <dataset_name>/item_text.csv is located."
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="qwen3",
        help="Logical LLM model name (only used to name output file)."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-v4",
        help="Qwen embedding model name (dashscope model id)."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="ThreadPoolExecutor worker count. 建议先用 3~5，避免频率过高。"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output pickle filename. Default: {llm_model}_emb.pickle"
    )

    return parser.parse_args()


args = parse_args()

# ============================================================
# 1. DashScope API Key 设置
# ============================================================
if "DASHSCOPE_API_KEY" not in os.environ:
    raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY，例如：export DASHSCOPE_API_KEY=xxxx")

dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

EMBED_MODEL = args.embedding_model

# -------- 全局限流参数（关键） --------
# 根据你账号的 QPS 限额调（比如 1 ~ 3 都比较稳）
RATE_LIMIT_QPS = 2.0

_rate_lock = threading.Lock()
_next_available_time = 0.0


def rate_limited_call(**kwargs):
    """
    所有 TextEmbedding.call 必须通过这里，保证全局 QPS 不超限。
    """
    global _next_available_time

    while True:
        with _rate_lock:
            now = time.time()
            wait = _next_available_time - now
            if wait > 0:
                # 还没到下一次请求的时间
                pass
            else:
                # 可以发起请求
                rsp = TextEmbedding.call(**kwargs)
                # 更新下一次可用时间
                _next_available_time = max(now, time.time()) + 1.0 / RATE_LIMIT_QPS
                return rsp

        # 没轮到自己，短暂 sleep 再试
        time.sleep(wait if wait > 0 else 0.01)


# ============================================================
# 2. 读取 item_text.csv
# ============================================================
csv_path = os.path.join(args.dataset_dir, args.dataset_name, "item_text.csv")
output_name = args.output_name or f"{args.llm_model}_emb.pickle"
output_path = os.path.join(args.dataset_dir, args.dataset_name, output_name)

print(f"[Info] Loading item_text from: {csv_path}")
df = pd.read_csv(csv_path)

required_cols = ["item_id", "title", "description", "category"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"item_text.csv 缺少必要字段: {missing_cols}, 实际字段: {df.columns.tolist()}")

print(f"[Info] Total rows in item_text.csv: {len(df)}")

df["item_id"] = df["item_id"].astype(int)


# ============================================================
# 3. Prompt 构造（带安全兜底）
# ============================================================
def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def build_item_prompt(row_dict):
    """
    根据结构化字段构造 prompt：
      - title / description 允许单独存在
      - 若 description 为空，则 fallback 到 title
      - 若两者都空 or category 空，则返回 None（上游跳过该样本）
    """
    title = safe_str(row_dict.get("title"))
    desc = safe_str(row_dict.get("description"))
    cat = safe_str(row_dict.get("category"))

    main_desc = desc if desc != "" else title

    if main_desc == "" or cat == "":
        return None

    if len(main_desc) > 4000:
        main_desc = main_desc[:4000]

    prompt = f"""Item Summary:
Title: {title if title != "" else "(No explicit title)"}

Category: {cat}

Core Purpose:
Describe what type of user would buy this item and what need it satisfies.

Short Description:
{main_desc}
"""
    return prompt


# ============================================================
# 4. 单条 API 调用（带限流 + 429 退避 + 重试）
# ============================================================
def fetch_embedding(row_dict, retry=5):
    """
    row_dict 必须包含:
      - item_id
      - prompt
    返回: (item_id, embedding 或 None)
    """
    item_id = row_dict["item_id"]
    text = row_dict["prompt"]

    for attempt in range(retry):
        try:
            rsp = rate_limited_call(
                model=EMBED_MODEL,
                input=text,
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
                return item_id, emb

            elif status_code == 429:
                # 频率限制：指数退避
                backoff = 2.0 * (attempt + 1)
                print(f"[Throttle] item {item_id}, attempt {attempt+1}/{retry}, "
                      f"code={code}, msg={message}, backoff={backoff:.1f}s")
                time.sleep(backoff)
                continue

            else:
                # 其它错误：打印出来，直接放弃
                print(f"[Error] item {item_id}, status_code={status_code}, code={code}, msg={message}")
                return item_id, None

        except Exception as e:
            print(f"[Error] item {item_id}, attempt {attempt+1}/{retry}: {e}")
            time.sleep(1.0)

    # 多次重试失败
    return item_id, None


# ============================================================
# 5. 多线程主流程
# ============================================================
def generate_embeddings_multithread(df, max_workers=5):
    row_dicts = []
    skipped_prompts = 0

    for row in df.itertuples():
        d = row._asdict()
        prompt = build_item_prompt(d)
        if prompt is None:
            skipped_prompts += 1
            continue
        row_dicts.append({"item_id": d["item_id"], "prompt": prompt})

    print(f"[Info] Rows with valid prompts: {len(row_dicts)} (skipped {skipped_prompts} rows with empty/invalid text)")

    embeddings = {}
    failed_items = 0

    print(f"[Info] Start generating embeddings with {max_workers} threads using model '{EMBED_MODEL}' ...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_embedding, r) for r in row_dicts]

        for future in tqdm(as_completed(futures), total=len(futures)):
            item_id, emb = future.result()
            if emb is not None:
                embeddings[item_id] = emb
            else:
                failed_items += 1

    print(f"[Info] Finished. Successful embeddings: {len(embeddings)}, failed after retry: {failed_items}")
    return embeddings


# ============================================================
# 6. 运行 & 保存
# ============================================================
item_sem_emb = generate_embeddings_multithread(df, max_workers=args.max_workers)

with open(output_path, "wb") as f:
    pickle.dump(item_sem_emb, f)

print(f"[Info] Saved {len(item_sem_emb)} embeddings to: {output_path}")
