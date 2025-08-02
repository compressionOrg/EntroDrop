export CUDA_VISIBLE_DEVICES=0

#conda activate grasp
# MSE*(1-COS)
set -x

MODEL_NAME="meta-llama/Llama-2-7b-hf"
layers_order="27,26,28,24,29,25,23,22,21,30,19,20,18,17,14,16,15,12,13,11,10,9,8,7,6,3,5,2,4,1,31,0"
log_file="run_rm_layers_llama2_7b_shortgpt.log"
# 循环执行不同的 num_prune 值
for num_prune in 4 6 8 10 12; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --model_name $MODEL_NAME --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file --eval_ppl "wikitext2,ptb" --tasks ""
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x