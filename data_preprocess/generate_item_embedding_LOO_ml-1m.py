#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_item_embedding_ml1m.py

适配 ML-1M 数据集：
多线程调用 Qwen text-embedding-v4，为 Movie 生成 semantic embedding。
由于 ML-1M 缺少 description，逻辑上会自动 Fallback 使用 Title + Category 构建 Prompt。

export DASHSCOPE_API_KEY=sk-f8d81c60072d45abbc150849b431044f

nohup python generate_item_embedding_LOO_ml-1m.py \
    --dataset_name ml-1m \
    --dataset_dir ../dataset \
    --llm_model qwen3 \
    --max_workers 5 \
    --output_name qwen3_emb_change.pickle \
    > ml-1m_change_generate_log.txt 2>&1 &
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
    parser = argparse.ArgumentParser(description="Generate ML-1M item embeddings with Qwen.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ml-1m",
        help="Dataset name, default is 'ml-1m'."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
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
        help="ThreadPoolExecutor worker count. 建议 3~5。"
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
    # 你也可以在这里直接写死 key，但不推荐
    # dashscope.api_key = "sk-..."
    raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY，例如：export DASHSCOPE_API_KEY=sk-xxxx")
else:
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

EMBED_MODEL = args.embedding_model

# -------- 全局限流参数 --------
RATE_LIMIT_QPS = 2.0  # 根据你的账号限额调整
_rate_lock = threading.Lock()
_next_available_time = 0.0


def rate_limited_call(**kwargs):
    """全局 QPS 限流控制"""
    global _next_available_time
    while True:
        with _rate_lock:
            now = time.time()
            wait = _next_available_time - now
            if wait <= 0:
                # 预支时间窗口
                _next_available_time = max(now, time.time()) + 1.0 / RATE_LIMIT_QPS
                # 发起请求
                try:
                    return TextEmbedding.call(**kwargs)
                except Exception as e:
                    # 如果发生网络层面的异常，不要卡死锁，抛出让外层重试
                    raise e

        # 如果需要等待，释放锁并在锁外 sleep
        time.sleep(wait if wait > 0 else 0.01)


# ============================================================
# 2. 读取 item_text.csv
# ============================================================
csv_path = os.path.join(args.dataset_dir, args.dataset_name, "item_text.csv")
output_name = args.output_name or f"{args.llm_model}_emb.pickle"
output_path = os.path.join(args.dataset_dir, args.dataset_name, output_name)

print(f"[Info] Loading item_text from: {csv_path}")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"找不到文件: {csv_path}。请先运行上一频的数据处理脚本。")

df = pd.read_csv(csv_path)

# ML-1M 的 item_text 应该包含 item_id, title, description(可能全空), category
required_cols = ["item_id", "title"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"item_text.csv 缺少必要字段: {missing_cols}")

print(f"[Info] Total rows in item_text.csv: {len(df)}")
df["item_id"] = df["item_id"].astype(int)


# ============================================================
# 3. Prompt 构造 (针对 Movie 数据适配)
# ============================================================
def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def build_item_prompt(row_dict):
    """
    改进版 Prompt：
    1. 将 'Animation|Comedy' 转换为自然语言 'Animation, Comedy'.
    2. 增加 Context 引导，试图激活 Embedding 模型内部关于著名电影的知识。
    3. 增加 'Audience' 和 'Theme' 的语义暗示。
    """
    title = safe_str(row_dict.get("title"))
    cat = safe_str(row_dict.get("category"))

    # 1. 清洗流派格式
    # "Animation|Children's|Comedy" -> "Animation, Children's, and Comedy"
    if "|" in cat:
        cats = cat.split("|")
        if len(cats) > 1:
            cat_str = ", ".join(cats[:-1]) + " and " + cats[-1]
        else:
            cat_str = cats[0]
    else:
        cat_str = cat

    # 2. 构造 Rich Prompt
    # 相比之前的简单拼接，这里用自然语言把 Title 和 Genre 融合，
    # 并显式要求模型关注“剧情”和“风格”。
    prompt = (
        f"Identify the movie titled '{title}'. "
        f"It is a {cat_str} film released around the year {title[-5:-1] if title[-1] == ')' else 'unknown'}. "
        f"Represent the semantic content of this movie, focusing on its plot themes, "
        f"visual style, and the typical audience appeal associated with {cat_str} movies."
    )

    return prompt

# def build_item_prompt(row_dict):
#     """
#     ML-1M 专用 Prompt 构造：
#     由于大部分 description 为空，主要依赖 Title 和 Genres (Category)。
#     """
#     title = safe_str(row_dict.get("title"))
#     desc = safe_str(row_dict.get("description"))
#     cat = safe_str(row_dict.get("category"))
#
#     # 如果 description 是空的，使用 title 作为主要描述
#     main_desc = desc if desc != "" else title
#
#     # 如果连 title 和 category 都没有，跳过
#     if main_desc == "" and cat == "":
#         return None
#
#     # 截断过长文本 (Embedding 模型通常有 token 限制)
#     if len(main_desc) > 2000:
#         main_desc = main_desc[:2000]
#
#     # 构造针对电影内容的 Prompt
#     prompt = f"""Movie Item Summary:
# Title: {title if title != "" else "(Unknown Title)"}
#
# Genres/Category: {cat}
#
# Context:
# This is a movie item from the MovieLens dataset.
#
# Description/Plot:
# {main_desc}
# """
#     return prompt


# ============================================================
# 4. 单条 API 调用
# ============================================================
def fetch_embedding(row_dict, retry=5):
    item_id = row_dict["item_id"]
    text = row_dict["prompt"]

    for attempt in range(retry):
        try:
            rsp = rate_limited_call(
                model=EMBED_MODEL,
                input=text,
                text_type="document"
            )

            status_code = getattr(rsp, "status_code", 200)

            if status_code == 200:
                out = getattr(rsp, "output", None)
                if (out is None or "embeddings" not in out or not out["embeddings"]):
                    raise RuntimeError(f"Empty embedding result: {rsp}")

                emb = out["embeddings"][0]["embedding"]
                return item_id, emb

            elif status_code == 429:
                # 触发限流，退避等待
                backoff = 2.0 * (attempt + 1)
                print(f"[Throttle] item {item_id}, wait {backoff}s...")
                time.sleep(backoff)
                continue

            else:
                # 其他错误
                msg = getattr(rsp, "message", "Unknown error")
                print(f"[Error] item {item_id} code={status_code}: {msg}")
                return item_id, None

        except Exception as e:
            print(f"[Exception] item {item_id}, retry {attempt + 1}/{retry}: {e}")
            time.sleep(1.0)

    return item_id, None


# ============================================================
# 5. 多线程执行
# ============================================================
def generate_embeddings_multithread(df, max_workers=5):
    row_dicts = []

    for row in df.itertuples():
        d = row._asdict()
        prompt = build_item_prompt(d)
        if prompt:
            row_dicts.append({"item_id": d["item_id"], "prompt": prompt})

    print(f"[Info] Ready to embed {len(row_dicts)} items.")

    embeddings = {}
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_embedding, r) for r in row_dicts]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
            item_id, emb = future.result()
            if emb is not None:
                embeddings[item_id] = emb
            else:
                failed_count += 1

    print(f"[Info] Done. Success: {len(embeddings)}, Failed: {failed_count}")
    return embeddings


# ============================================================
# 6. 保存结果
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    item_sem_emb = generate_embeddings_multithread(df, max_workers=args.max_workers)

    with open(output_path, "wb") as f:
        pickle.dump(item_sem_emb, f)

    print(f"[Success] Embeddings saved to: {output_path}")