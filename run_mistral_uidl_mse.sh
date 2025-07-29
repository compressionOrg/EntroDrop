export CUDA_VISIBLE_DEVICES=0

#conda activate grasp

set -x
layers_order="24,25,26,27,28,29,30,31,20,19,21,18,17,9,22,16,8,10,11,15,7,13,12,14,6,5,4,3,2,1,0,23"
log_file="mistral_7b_uidl_mse.log"
# 循环执行不同的 num_prune 值
for num_prune in 6 8 10 12 14 16; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers_mistral.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x