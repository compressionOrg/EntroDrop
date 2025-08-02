export CUDA_VISIBLE_DEVICES=0

#conda activate grasp

set -x
layers_order="24,25,23,22,20,26,19,21,27,18,28,17,10,11,16,13,14,15,9,12,8,29,7,3,6,2,4,5,1,30,0,31"
model_name="meta-llama/Llama-3.1-8B"
log_file="run_rm_layers_infer_llama3_l1.log"
# 循环执行不同的 num_prune 值
for num_prune in 6; do
    echo "Running with num_prune=$num_prune"
    python run_rm_layers_infer.py --model_name ${model_name} --layers_order  ${layers_order}  --num_prune $num_prune  --log_file $log_file --run_inference_eval --eval_original_model
    echo "Completed num_prune=$num_prune"
    echo "=" * 50
done

set +x