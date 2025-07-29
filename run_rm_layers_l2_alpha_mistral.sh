export CUDA_VISIBLE_DEVICES=2

#conda activate grasp

set -x

# 定义alpha值和对应的layers_order数组
declare -A alpha_orders
alpha_orders["0.1"]="27,26,25,24,23,22,28,21,29,30,20,19,13,17,18,12,15,16,14,11,10,9,8,7,6,3,4,5,2,1,31,0"
alpha_orders["0.2"]="27,26,25,24,23,22,28,21,29,30,20,19,13,17,18,12,15,16,14,11,10,9,8,7,6,3,4,5,2,1,0,31"
alpha_orders["0.3"]="27,26,25,24,23,22,28,21,29,30,20,19,13,17,18,12,15,14,16,11,10,9,8,7,6,3,4,5,2,1,0,31"
alpha_orders["0.4"]="27,26,24,25,23,22,28,21,29,20,30,19,13,17,18,12,15,14,16,11,10,9,8,7,6,3,4,5,2,1,0,31"
alpha_orders["0.5"]="27,26,24,25,23,22,28,21,29,20,30,13,19,17,12,18,15,14,16,11,10,9,8,7,6,3,4,5,2,1,0,31"
alpha_orders["0.6"]="27,24,26,25,23,22,21,28,20,29,30,13,19,17,12,18,15,14,16,11,10,9,8,7,6,3,4,5,2,1,0,31"
alpha_orders["0.7"]="27,24,26,25,23,22,21,28,20,29,30,13,19,17,12,18,15,14,16,11,10,9,8,7,6,3,4,5,2,1,0,31"
alpha_orders["0.8"]="24,25,26,27,23,22,21,28,20,13,29,12,17,19,14,15,16,18,11,10,30,9,8,7,6,3,4,5,2,1,0,31"
alpha_orders["0.9"]="22,24,23,25,26,27,21,20,13,12,14,15,17,16,28,11,18,19,10,9,8,7,29,6,3,4,5,2,30,1,0,31"

MODEL_NAME="mistralai/Mistral-7B-v0.3"
MODEL_SHORT_NAME="${MODEL_NAME##*/}"


# 循环执行不同的alpha值和对应的layers_order
for alpha in  "0.7"; do  
    layers_order="${alpha_orders[$alpha]}"
    log_file="run_rm_layes_l2_${MODEL_SHORT_NAME}_alpha${alpha}.log"
    
    echo "Running with alpha=$alpha"
    echo "Layers order: $layers_order"
    
    # 循环执行不同的 num_prune 值
    for num_prune in 7; do
        echo "  Running with num_prune=$num_prune"
        python run_rm_layers.py --model_name $MODEL_NAME --layers_order "${layers_order}" --num_prune $num_prune --log_file $log_file --eval_ppl "wikitext2,ptb" # --tasks "" 
        echo "  Completed num_prune=$num_prune"
    done
    
    echo "Completed alpha=$alpha"
    echo $(printf '=%.0s' {1..50})
done

set +x