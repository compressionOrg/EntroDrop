export CUDA_VISIBLE_DEVICES=0

#conda activate grasp
# MSE*(1-COS)
set -x

MODEL_NAME="meta-llama/Llama-3.1-8B"
layers_order="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
log_file="run_rm_layers_llama3_8b_shortgpt_l1.log"
# 循环执行不同的 num_prune 值
for num_prune in 8 10 12; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers.py --model_name $MODEL_NAME --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file --eval_ppl "wikitext2" --task "mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq"
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x