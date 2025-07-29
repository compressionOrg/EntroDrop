export CUDA_VISIBLE_DEVICES=3

#conda activate grasp

set -x

MODEL_NAME="meta-llama/Llama-3.1-8B"
NUM_PRUNE_LAYERS=12
# 循环执行不同的 num_prune 值
for alpha in 0.7; do # 
    echo "Running with alpha=$alpha"
    python run_shortgpt_l1_alpha.py  --alpha  $alpha --model_name $MODEL_NAME --save_model --num_prune_layers $NUM_PRUNE_LAYERS
    echo "Completed alpha=$alpha"
    echo "=" * 50
done

set +x