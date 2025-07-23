export CUDA_VISIBLE_DEVICES=2

#conda activate grasp
# MSE*(1-COS)
set -x
layers_order="24,25,26,27,28,29,30,31,19,18,20,8,7,6,9,10,17,21,11,5,16,4,12,13,14,15,3,2,22,1,0,23"
log_file="llama3.1_8b_uidl_mse.log"
# 循环执行不同的 num_prune 值
for num_prune in 10 12 14 16; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x