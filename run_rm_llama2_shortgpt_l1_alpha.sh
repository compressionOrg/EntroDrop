export CUDA_VISIBLE_DEVICES=1

#conda activate grasp
# MSE*(1-COS)
set -x

# 定义alpha值和对应的layers_order数组
declare -A alpha_orders
alpha_orders["0.1"]="24,27,26,25,23,28,29,21,22,19,20,18,30,17,14,15,12,16,13,11,10,9,8,7,6,3,2,5,4,1,0,31"
alpha_orders["0.2"]="24,26,27,23,25,21,28,22,29,19,20,18,17,14,12,11,30,15,13,16,10,9,8,7,3,6,2,5,4,1,0,31"
alpha_orders["0.3"]="24,23,26,21,27,25,22,28,19,20,29,18,17,14,12,11,13,15,16,10,9,8,3,2,7,6,30,5,4,1,0,31"
alpha_orders["0.4"]="24,23,21,26,22,19,25,27,18,20,11,14,12,28,17,13,15,10,29,16,3,9,2,8,6,7,5,4,30,1,0,31"
alpha_orders["0.5"]="11,21,24,12,23,19,14,3,2,13,18,22,10,26,17,25,15,9,20,8,27,6,4,5,7,16,28,29,1,30,0,31"
alpha_orders["0.6"]="2,3,11,12,4,10,5,14,6,13,8,9,7,19,15,21,18,17,24,23,22,16,20,26,25,27,1,28,29,0,30,31"
alpha_orders["0.7"]="2,3,4,5,6,11,7,12,8,10,9,13,14,15,1,17,19,18,16,21,23,24,20,22,25,26,27,0,28,29,30,31"
alpha_orders["0.8"]="2,3,4,5,6,7,1,8,11,9,10,12,13,14,15,0,17,16,18,19,21,20,23,24,22,25,26,27,28,29,30,31"
alpha_orders["0.9"]="2,3,1,4,5,6,0,7,8,9,11,10,12,13,14,15,17,16,18,19,21,20,23,22,24,25,26,27,28,29,30,31"

# 循环执行不同的alpha值和对应的layers_order
for alpha in  "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"; do
    layers_order="${alpha_orders[$alpha]}"
    log_file="llama2_7b_shortgpt_l1_alpha_${alpha}.log"
    
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