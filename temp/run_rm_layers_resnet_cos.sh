export CUDA_VISIBLE_DEVICES=0

#conda activate grasp

set -x
layers_order="29,30,28,27,23,18,15,26,17,21,20,19,25,16,24,22,14,11,8,12,4,5,1,3,6,9,0,7,2,10,13,31"
log_file="llama3.1_8b_cosine.log"
# 循环执行不同的 num_prune 值
for num_prune in 6 8 10 12 14 16; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x