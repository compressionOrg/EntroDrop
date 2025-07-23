export CUDA_VISIBLE_DEVICES=0

#conda activate grasp

set -x
layers_order="29,30,28,27,23,17,13,25,14,21,20,19,22,16,24,26,15,7,6,12,4,5,2,3,8,11,1,9,0,10,18,31"
log_file="llama3.1_8b_resnet_l1.log"
# 循环执行不同的 num_prune 值
for num_prune in 6 10 14; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x