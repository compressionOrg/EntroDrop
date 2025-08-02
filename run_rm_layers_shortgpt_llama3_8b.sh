export CUDA_VISIBLE_DEVICES=0

#conda activate grasp
# MSE*(1-COS)
set -x

MODEL_NAME="meta-llama/Llama-3.1-8B"
layers_order="25,26,24,27,23,28,22,29,20,21,19,18,30,17,16,11,10,13,15,14,9,12,8,7,3,6,4,2,5,1,31,0"
log_file="run_rm_layers_llama3_8b_shortgpt.log"
# 循环执行不同的 num_prune 值
for num_prune in 2; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --model_name $MODEL_NAME --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file --eval_ppl "wikitext2,ptb" --tasks ""
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x