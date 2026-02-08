# generate_plot_ml1m.py
import os
import time
import threading
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# DashScope SDK
"""
export DASHSCOPE_API_KEY=sk-f8d81c60072d45abbc150849b431044f
python generate_plot_ml1m.py
"""

try:
    import dashscope
    from dashscope import Generation
except ImportError:
    print("请先安装 dashscope: pip install dashscope")
    exit()

# ==========================================
# 配置区域
# ==========================================
# 建议将 API Key 放在环境变量中，或者在这里临时填入
# dashscope.api_key = "sk-..."
if not dashscope.api_key:
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

if not dashscope.api_key:
    raise ValueError("未找到 API Key，请设置环境变量 DASHSCOPE_API_KEY 或在代码中指定。")

MODEL_NAME = "qwen-turbo"  # 便宜且速度快，足够处理电影知识
MAX_WORKERS = 10  # 并发线程数，根据您的 API 限流额度调整


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../dataset/ml-1m/item_text.csv",
                        help="原始 item_text.csv 路径")
    parser.add_argument("--output_file", type=str, default="../dataset/ml-1m/item_text_enriched.csv",
                        help="增强后的输出路径")
    return parser.parse_args()


# ==========================================
# 核心生成逻辑
# ==========================================
def generate_one_plot(item_data):
    """
    调用 LLM 为单个物品生成简介
    """
    item_id = item_data['item_id']
    title = str(item_data['title'])
    category = str(item_data['category']).replace('|', ', ')

    # 构造 Prompt：要求事实性描述，不包含废话
    prompt = f"""You are a movie expert. 
Task: Provide a concise but information-rich plot summary (about 50-80 words) for the movie.

Movie Entity:
Title: "{title}"
Genres: {category}

Requirements:
1. Focus on the main storyline, central conflict, and key themes.
2. Do not include release dates, actor names, or 'this movie is about...' phrases unless necessary.
3. Output ONLY the summary text in English.
"""

    # 重试机制
    for attempt in range(3):
        try:
            # 调用 Qwen-Turbo
            resp = Generation.call(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                result_format='message'
            )

            if resp.status_code == 200:
                content = resp.output.choices[0].message.content.strip()
                return item_id, content

            elif resp.status_code == 429:  # 限流
                time.sleep(2 * (attempt + 1))
                continue
            else:
                # 其他错误
                print(f"[Error] Item {item_id}: {resp.message}")
                return item_id, ""

        except Exception as e:
            print(f"[Exception] Item {item_id}: {e}")
            time.sleep(1)

    return item_id, ""


# ==========================================
# 主流程
# ==========================================
def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        print(f"错误：找不到输入文件 {args.input_file}")
        return

    print(f"读取数据: {args.input_file}")
    df = pd.read_csv(args.input_file)
    print(f"总共有 {len(df)} 部电影需要生成简介。")

    # 结果容器
    results = {}
    lock = threading.Lock()

    # 使用线程池并发请求
    print(f"开始生成 (Model: {MODEL_NAME}, Workers: {MAX_WORKERS})...")

    row_list = [row._asdict() for row in df.itertuples()]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        futures = [executor.submit(generate_one_plot, row) for row in row_list]

        # 进度条监控
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            item_id, plot = future.result()
            with lock:
                results[item_id] = plot

    # 将结果合并回 DataFrame
    print("生成完毕，正在合并数据...")
    df['description'] = df['item_id'].map(results)

    # 简单的后处理：如果没有生成成功，用 Title 填充，防止空值报错
    df['description'] = df['description'].fillna(df['title'])
    empty_mask = df['description'].str.strip() == ""
    df.loc[empty_mask, 'description'] = df.loc[empty_mask, 'title']

    # 保存
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"增强后的数据已保存至: {args.output_file}")

    # 打印样例
    print("\n=== 样例数据 ===")
    print(df[['item_id', 'title', 'description']].head(3))


if __name__ == "__main__":
    main()