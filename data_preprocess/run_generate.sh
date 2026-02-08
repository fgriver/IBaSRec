#!/bin/bash

# ----------------------------------------
# 设置 API Key
# ----------------------------------------
export DASHSCOPE_API_KEY="sk-f8d81c60072d45abbc150849b431044f"

# ----------------------------------------
# 数据集列表（你可以自行扩展）
# ----------------------------------------
DATASETS=("Sports_and_Outdoors" "Toys_and_Games")

# ----------------------------------------
# 其他参数
# ----------------------------------------
DATASET_DIR="../dataset"
LLM_MODEL="qwen3"
EMBED_MODEL="text-embedding-v4"
MAX_WORKERS=10

# ----------------------------------------
# 遍历数据集并运行 embedding 生成脚本
# ----------------------------------------
for NAME in "${DATASETS[@]}"; do

    LOG_FILE="${NAME}_generate_log.txt"
    OUTPUT_NAME="${LLM_MODEL}_emb.pickle"

    echo "======================================================="
    echo "Running dataset: ${NAME}"
    echo "Log file: ${LOG_FILE}"
    echo "Output pickle: ${OUTPUT_NAME}"
    echo "======================================================="

    nohup python generate_item_embedding_LOO_amazon.py \
        --dataset_name "${NAME}" \
        --dataset_dir "${DATASET_DIR}" \
        --llm_model "${LLM_MODEL}" \
        --embedding_model "${EMBED_MODEL}" \
        --max_workers ${MAX_WORKERS} \
        --output_name "${OUTPUT_NAME}" \
        > "${LOG_FILE}" 2>&1 &

    echo "Task for ${NAME} started in background."
    echo
done

echo "All dataset embedding tasks have been launched."
