export CUDA_VISIBLE_DEVICES=0

#conda activate grasp

set -x
layers_order="29,30,28,27,24,16,14,25,13,20,21,19,23,17,22,26,15,9,7,12,4,6,1,3,5,11,0,8,2,10,18,31"
log_file="llama3.1_8b_resnet.log"
# 循环执行不同的 num_prune 值
for num_prune in 10 14; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --layers_order  ${layers_order}  --num_prune $num_prune --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x