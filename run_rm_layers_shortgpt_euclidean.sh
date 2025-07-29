export CUDA_VISIBLE_DEVICES=0

#conda activate grasp
# MSE*(1-COS)
set -x
layers_order=""
log_file="llama3.1_8b_shortgpt_euclidean.log"
# 循环执行不同的 num_prune 值
for num_prune in 2 4 6 8 10 12; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x