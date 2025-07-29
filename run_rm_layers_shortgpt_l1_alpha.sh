export CUDA_VISIBLE_DEVICES=0

#conda activate grasp
# MSE*(1-COS)
set -x

# 定义alpha值和对应的layers_order数组
declare -A alpha_orders
alpha_orders["0.1"]="25,24,26,27,23,28,22,20,21,29,19,18,17,30,16,11,10,13,15,14,9,12,8,7,3,6,4,2,5,1,0,31"
alpha_orders["0.2"]="25,24,26,23,27,22,28,20,21,19,29,18,17,30,16,10,11,13,15,14,9,12,8,7,3,6,2,4,5,1,0,31"
alpha_orders["0.3"]="25,24,26,23,27,22,28,20,21,19,29,18,17,16,11,10,13,30,15,14,9,12,8,7,3,6,4,2,5,1,0,31"
alpha_orders["0.4"]="24,25,23,26,22,27,20,21,28,19,18,29,17,10,16,11,13,14,15,9,12,8,30,3,7,6,2,4,5,1,0,31"
alpha_orders["0.5"]="24,25,23,26,22,20,27,19,21,28,18,29,17,10,11,16,13,9,14,15,12,8,3,7,6,2,4,5,30,1,0,31"
alpha_orders["0.6"]="24,23,25,22,20,26,19,21,27,18,28,17,10,11,13,16,9,14,29,8,12,15,3,7,2,6,4,5,1,30,0,31"
alpha_orders["0.7"]="24,23,20,25,22,19,21,26,18,27,10,11,17,13,9,8,16,28,12,3,14,15,2,7,6,4,5,1,29,30,0,31"
alpha_orders["0.8"]="20,19,23,10,22,24,11,3,18,25,21,9,2,8,13,26,12,17,14,4,7,16,6,1,15,5,27,28,29,0,30,31"
alpha_orders["0.9"]="2,3,1,4,10,11,8,9,5,6,7,13,12,19,20,14,18,15,16,17,22,23,21,24,0,25,26,27,28,29,30,31"

# 循环执行不同的alpha值和对应的layers_order
for alpha in  "0.7"; do  # "0.3" "0.4" "0.6"
    layers_order="${alpha_orders[$alpha]}"
    log_file="llama3.1_8b_shortgpt_l1_alpha_${alpha}.log"
    
    echo "Running with alpha=$alpha"
    echo "Layers order: $layers_order"
    
    # 循环执行不同的 num_prune 值
    for num_prune in 4 6 14 16; do
        echo "  Running with num_prune=$num_prune"
        python run_rm_layers.py --layers_order "${layers_order}" --num_prune $num_prune --log_file $log_file
        echo "  Completed num_prune=$num_prune"
    done
    
    echo "Completed alpha=$alpha"
    echo $(printf '=%.0s' {1..50})
done

set +x