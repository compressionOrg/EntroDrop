export CUDA_VISIBLE_DEVICES=2

#conda activate grasp

set -x
layers_order="24,25,26,27,28,29,30,31,21,20,19,22,18,17,16,15,8,14,10,7,9,11,13,6,12,5,4,3,2,23,1,0"
log_file="llama3.1_8b_uidl.log"
# 循环执行不同的 num_prune 值
for num_prune in 4 6 8 10 12 14 16; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x