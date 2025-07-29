export CUDA_VISIBLE_DEVICES=0

#conda activate grasp
# MSE*(1-COS)
set -x
layers_order="29,30,27,26,20,16,13,25,15,19,18,17,23,14,24,21,12,5,4,8,2,3,0,1,7,10,6,11,9,22,28,31"
log_file="llama3.1_8b_resnet_mse2.log"
# 循环执行不同的 num_prune 值
for num_prune in 4 6 8 10 12 14 16; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x