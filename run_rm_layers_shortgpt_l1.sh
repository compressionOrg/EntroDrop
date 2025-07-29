export CUDA_VISIBLE_DEVICES=0

#conda activate grasp
# MSE*(1-COS)
set -x
layers_order="24,25,23,26,22,20,27,19,21,28,18,29,17,10,11,16,13,9,14,15,12,8,3,7,6,2,4,5,30,1,0,31"
log_file="llama3.1_8b_shortgpt_l1.log"
# 循环执行不同的 num_prune 值
for num_prune in 6 8 10 12 14 16; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x