export CUDA_VISIBLE_DEVICES=0

#conda activate grasp

set -x
layers_order="27,26,24,25,23,22,28,21,29,20,30,19,13,17,18,12,15,14,16,11,10,9,8,7,6,3,4,5,2,1,0,31"
log_file="mistral_7b_shortgpt_mse.log"
# 循环执行不同的 num_prune 值
for num_prune in 4 6 8 10 12 14 16; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers_mistral.py --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x