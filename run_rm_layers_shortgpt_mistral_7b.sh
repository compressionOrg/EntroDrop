export CUDA_VISIBLE_DEVICES=0

#conda activate grasp
# MSE*(1-COS)
set -x

MODEL_NAME="mistralai/Mistral-7B-v0.3"
layers_order="27,26,25,24,23,22,28,29,21,30,20,19,13,17,18,12,15,16,14,11,10,9,8,7,6,3,4,5,2,1,31,0"
log_file="run_rm_layers_mistral_7b_shortgpt.log"
# 循环执行不同的 num_prune 值
for num_prune in 2 4 6 8 10 12; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --model_name $MODEL_NAME --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file --eval_ppl "wikitext2,ptb" --tasks ""
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x