export CUDA_VISIBLE_DEVICES=2

#conda activate grasp

set -x

MODEL_NAME="baichuan-inc/Baichuan2-7B-Base"

# 循环执行不同的 num_prune 值
for alpha in 0.2 ; do # 
    echo "Running with alpha=$alpha"
    python run_shortgpt_l1_alpha.py  --alpha  $alpha --model_name $MODEL_NAME
    echo "Completed alpha=$alpha"
    echo "=" * 50
done

set +x