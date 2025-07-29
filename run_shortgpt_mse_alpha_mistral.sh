export CUDA_VISIBLE_DEVICES=2

#conda activate grasp

set -x
MODEL_NAME="mistralai/Mistral-7B-v0.3"
# 循环执行不同的 num_prune 值
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do # 
    echo "Running with alpha=$alpha"
    python run_shortgpt_mse_alpha.py  --alpha  $alpha --model_name $MODEL_NAME
    echo "Completed alpha=$alpha"
    echo "=" * 50
done

set +x