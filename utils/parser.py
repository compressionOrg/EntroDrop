import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/export/share/liangweiyang/AttentionReuse/Llama3-8B-Instruct")
    parser.add_argument("--save_path", type=str, default="./layer_seq.txt")
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--method", type=str, default="EntroDrop")
    parser.add_argument("--label", type=str, default='')

    return parser.parse_args()