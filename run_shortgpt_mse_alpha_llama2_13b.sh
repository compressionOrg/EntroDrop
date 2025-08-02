export CUDA_VISIBLE_DEVICES=0,1

#conda activate grasp

set -x
MODEL_NAME="meta-llama/Llama-2-13b-hf"
NUM_PRUNE_LAYERS=7
# 循环执行不同的 num_prune 值
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do # 
    echo "Running with alpha=$alpha"
    python run_shortgpt_mse_alpha_multigpu.py  --alpha  $alpha --model_name $MODEL_NAME --num_prune_layers $NUM_PRUNE_LAYERS
    echo "Completed alpha=$alpha"
    echo "=" * 50
done

set +x