# generate_embedding_enriched_ml1m.py
import os
import time
import pickle
import argparse
import threading
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import dashscope
    from dashscope import TextEmbedding
except ImportError:
    print("请先安装 dashscope")
    exit()


# ==========================================
# 配置与参数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    # 注意：默认读取上一步生成的 enriched 文件
    parser.add_argument("--input_csv", type=str, default="../dataset/ml-1m/item_text_enriched.csv")
    parser.add_argument("--output_dir", type=str, default="../dataset/ml-1m")
    # 输出文件名建议包含 'enriched' 以示区分
    parser.add_argument("--output_name", type=str, default="qwen3_emb_enriched_matrix.pickle")
    parser.add_argument("--llm_model", type=str, default="text-embedding-v4")
    parser.add_argument("--max_workers", type=int, default=5)
    return parser.parse_args()


# 全局限流锁
_rate_lock = threading.Lock()
_next_available_time = 0.0
RATE_LIMIT_QPS = 5.0  # text-embedding-v4 的 QPS 限制通常较高，可设为 5~10


def rate_limited_call(**kwargs):
    global _next_available_time
    while True:
        with _rate_lock:
            now = time.time()
            wait = _next_available_time - now
            if wait <= 0:
                _next_available_time = max(now, time.time()) + 1.0 / RATE_LIMIT_QPS
                try:
                    return TextEmbedding.call(**kwargs)
                except Exception as e:
                    raise e
        time.sleep(wait if wait > 0 else 0.01)


# ==========================================
# Prompt 构造 (关键修改点)
# ==========================================
def build_rich_prompt(row):
    """
    构造富含语义的 Prompt
    """
    title = str(row['title'])
    cat = str(row['category']).replace('|', ', ')
    desc = str(row['description'])  # 这是 Qwen 生成的详细剧情

    # 语义增强型 Prompt
    prompt = f"""Movie Entity:
Title: {title}
Genres: {cat}

Plot Summary:
{desc}

Task:
Generate a semantic representation that captures the narrative style, thematic elements, and user appeal of this movie based on the plot provided above.
"""
    return prompt


# ==========================================
# Embedding 获取逻辑
# ==========================================
def fetch_embedding(row_dict, model_name):
    item_id = row_dict['item_id']
    text = row_dict['prompt']

    for attempt in range(5):
        try:
            rsp = rate_limited_call(
                model=model_name,
                input=text,
                text_type="document"
            )

            if hasattr(rsp, 'status_code') and rsp.status_code == 200:
                embeddings = rsp.output['embeddings']
                if embeddings and len(embeddings) > 0:
                    return item_id, embeddings[0]['embedding']

            # 处理限流
            if hasattr(rsp, 'status_code') and rsp.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue

            print(f"[Error] Item {item_id}: {getattr(rsp, 'message', 'Unknown error')}")
            return item_id, None

        except Exception as e:
            # print(f"[Exception] Item {item_id}: {e}")
            time.sleep(1)

    return item_id, None


# ==========================================
# 主流程
# ==========================================
def main():
    """
    export DASHSCOPE_API_KEY=sk-f8d81c60072d45abbc150849b431044f
    python generate_embedding_enriched_ml1m.py
    """
    args = parse_args()

    # 设置 API Key
    if not dashscope.api_key:
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    # 1. 读取数据
    if not os.path.exists(args.input_csv):
        print(f"错误：找不到输入文件 {args.input_csv}。请先运行 generate_plot_ml1m.py")
        return

    df = pd.read_csv(args.input_csv)
    print(f"读取数据成功，共 {len(df)} 条。")

    # 2. 准备 Prompt
    tasks = []
    for row in df.itertuples():
        d = row._asdict()
        prompt = build_rich_prompt(d)
        tasks.append({"item_id": d['item_id'], "prompt": prompt})

    # 3. 多线程生成
    embeddings_map = {}
    failed_count = 0

    print(f"开始生成 Embedding (Model: {args.llm_model})...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(fetch_embedding, t, args.llm_model) for t in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
            item_id, emb = future.result()
            if emb is not None:
                embeddings_map[item_id] = emb
            else:
                failed_count += 1

    print(f"生成完成。成功: {len(embeddings_map)}, 失败: {failed_count}")

    # 4. 整理并保存为 Pickle (List of lists 格式，按 item_id 排序)
    # 确保 item_id 是连续的 0 ~ N-1
    max_id = df['item_id'].max()
    final_matrix = []

    # 维度推断
    sample_dim = len(next(iter(embeddings_map.values())))

    # 按 ID 顺序填充，缺失的用全 0 补全 (极少情况)
    for i in range(max_id + 1):
        if i in embeddings_map:
            final_matrix.append(embeddings_map[i])
        else:
            print(f"[Warning] Missing embedding for item_id {i}, filling with zeros.")
            final_matrix.append([0.0] * sample_dim)

    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, "wb") as f:
        pickle.dump(final_matrix, f)

    print(f"Embedding 矩阵已保存至: {output_path}")
    print(f"矩阵形状: ({len(final_matrix)}, {sample_dim})")

    # 同时保存一个 data_statis.df (BetaFuse 代码需要读取这个来获取 item_num)
    statis_path = os.path.join(args.output_dir, "data_statis.df")
    pd.DataFrame([{"item_num": len(final_matrix), "user_num": 6040}]).to_pickle(statis_path)
    print(f"更新了统计文件: {statis_path}")


if __name__ == "__main__":
    main()