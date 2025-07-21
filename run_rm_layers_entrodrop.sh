export CUDA_VISIBLE_DEVICES=2

#conda activate grasp

set -x

layers_order="29,30,28,26,18,16,17,24,6,20,22,4,19,15,25,27,23,7,8,0,3,13,10,5,14,21,1,12,2,9,11,31"

# 循环执行不同的 num_prune 值
for num_prune in 4 8 12 16; do
    echo "Running with num_prune=$num_prune" 
    python run_rm_layers.py --layers_to_remove  ${layers_order}  --num_prune $num_prune
    echo "Completed num_prune=$num_prune" 
    echo "=" * 50
done

set +x