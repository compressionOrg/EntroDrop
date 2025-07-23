export CUDA_VISIBLE_DEVICES=0

#conda activate grasp
# MSE*(1-COS)
set -x
layers_order="25,24,26,23,27,22,28,20,21,19,29,18,17,16,10,11,13,15,14,9,12,8,30,7,3,6,4,2,5,1,0,31"
log_file="llama3.1_8b_shortgpt_mse.log"
# 循环执行不同的 num_prune 值
for num_prune in 4 6 8 10 12 14 16; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x