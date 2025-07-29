export CUDA_VISIBLE_DEVICES=1

#conda activate grasp

set -x

# 定义alpha值和对应的layers_order数组
declare -A alpha_orders
alpha_orders["0.1"]="27,24,26,25,23,28,29,21,22,19,20,18,30,17,14,15,16,12,13,11,10,9,8,7,6,3,5,2,4,1,0,31"
alpha_orders["0.2"]="24,27,26,23,25,28,21,22,29,19,20,18,17,14,15,12,16,13,11,30,10,9,8,7,6,3,5,2,4,1,0,31"
alpha_orders["0.3"]="24,23,27,25,26,21,22,28,19,20,18,29,17,14,12,15,11,13,16,10,9,8,7,6,3,30,5,2,4,1,0,31"
alpha_orders["0.4"]="24,23,21,25,27,26,22,19,18,20,28,17,14,12,29,11,15,13,16,10,9,8,7,6,3,2,5,4,30,1,0,31"
alpha_orders["0.5"]="23,21,24,19,22,25,18,27,26,20,17,14,12,11,13,15,28,16,10,9,8,7,6,3,29,2,5,4,1,30,0,31"
alpha_orders["0.6"]="21,19,23,24,18,14,12,11,17,13,22,15,25,20,10,27,16,26,9,8,3,7,6,2,5,4,28,29,1,30,0,31"
alpha_orders["0.7"]="11,12,14,13,19,10,18,15,17,21,9,8,3,7,6,23,2,5,16,4,24,20,22,25,26,27,28,1,29,0,30,31"
alpha_orders["0.8"]="11,12,14,10,13,3,2,9,8,6,7,5,4,15,17,18,19,16,21,23,20,24,22,1,25,26,27,0,28,29,30,31"
alpha_orders["0.9"]="3,2,4,5,6,11,7,8,12,10,9,13,14,15,1,17,16,18,19,0,21,20,23,22,24,25,26,27,28,29,30,31"

# 循环执行不同的alpha值和对应的layers_order
for alpha in  "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"; do  
    layers_order="${alpha_orders[$alpha]}"
    log_file="llama2_7b_shortgpt_l2_alpha_${alpha}.log"
    
    echo "Running with alpha=$alpha"
    echo "Layers order: $layers_order"
    
    # 循环执行不同的 num_prune 值
    for num_prune in 9; do
        echo "  Running with num_prune=$num_prune"
        python run_rm_layers.py --layers_order "${layers_order}" --num_prune $num_prune --log_file $log_file --tasks "" --eval_ppl "wikitext2,ptb"
        echo "  Completed num_prune=$num_prune"
    done
    
    echo "Completed alpha=$alpha"
    echo $(printf '=%.0s' {1..50})
done

set +x