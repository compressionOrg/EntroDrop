export CUDA_VISIBLE_DEVICES=0

python evaluate.py \
    --hf \
    --model_name_or_path 'XiaodongChen/Llama-3.1-5.4B' \
    --eval_ppl "wikitext2,ptb" \
    --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,mathqa \
    --batch_size 1 \
    --log_file "evaluate_llama3_5.4B.log"

    