export CUDA_VISIBLE_DEVICES=0

#conda activate grasp

set -x

# 定义alpha值和对应的layers_order数组
declare -A alpha_orders
alpha_orders["0.1"]="27,26,25,24,23,22,28,21,29,30,20,19,13,17,18,12,15,14,16,11,10,9,8,7,6,3,4,5,2,1,31,0"
alpha_orders["0.2"]="27,26,24,25,23,22,28,21,29,30,20,13,19,12,17,18,14,15,16,11,10,9,8,7,6,3,4,5,2,1,0,31"
alpha_orders["0.3"]="27,26,24,25,23,22,28,21,29,20,30,13,12,19,17,14,15,18,16,11,10,9,8,7,6,3,4,5,2,1,0,31"
alpha_orders["0.4"]="24,27,25,26,23,22,21,28,20,29,30,13,12,14,15,17,19,16,18,11,10,9,8,7,3,6,4,5,2,1,0,31"
alpha_orders["0.5"]="24,22,25,23,27,26,21,28,20,13,12,14,29,15,17,16,11,19,18,30,10,9,8,7,3,6,4,5,2,1,0,31"
alpha_orders["0.6"]="22,24,23,25,26,27,21,20,13,12,28,14,15,11,16,17,18,10,9,19,29,8,30,3,7,6,4,5,2,1,0,31"
alpha_orders["0.7"]="22,23,24,25,13,26,21,12,27,14,11,15,20,16,10,9,17,18,8,19,28,3,7,4,6,5,29,2,30,1,0,31"
alpha_orders["0.8"]="13,12,11,14,9,10,22,15,23,24,8,16,3,21,25,4,17,6,7,20,26,5,27,18,2,19,28,1,29,30,0,31"
alpha_orders["0.9"]="3,12,4,9,13,11,10,8,2,5,6,7,14,15,1,16,17,22,18,21,23,20,24,19,25,26,27,0,28,29,30,31"

MODEL_NAME="mistralai/Mistral-7B-v0.3"
MODEL_SHORT_NAME="${MODEL_NAME##*/}"


# 循环执行不同的alpha值和对应的layers_order
for alpha in  "0.3" "0.4" "0.5" "0.6" "0.7"; do  
    layers_order="${alpha_orders[$alpha]}"
    log_file="run_rm_layes_l1_${MODEL_SHORT_NAME}_alpha${alpha}.log"
    
    echo "Running with alpha=$alpha"
    echo "Layers order: $layers_order"
    
    # 循环执行不同的 num_prune 值
    for num_prune in 7; do
        echo "  Running with num_prune=$num_prune"
        python run_rm_layers.py --model_name $MODEL_NAME --layers_order "${layers_order}" --num_prune $num_prune --log_file $log_file  --eval_ppl "wikitext2,ptb" # --tasks ""
        echo "  Completed num_prune=$num_prune"
    done
    
    echo "Completed alpha=$alpha"
    echo $(printf '=%.0s' {1..50})
done

set +x