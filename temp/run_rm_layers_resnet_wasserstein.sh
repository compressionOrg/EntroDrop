export CUDA_VISIBLE_DEVICES=0

#conda activate grasp

set -x
layers_order="29,30,13,15,7,0,1,3,8,5,4,2,9,6,10,18,16,11,12,17,19,14,20,21,23,25,22,26,24,27,28,31"
log_file="llama3.1_8b_wasserstein.log"
# 循环执行不同的 num_prune 值
for num_prune in 4 6 8 10 12 14 16; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x