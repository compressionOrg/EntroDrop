# EntroDrop

## Introduction
EntroDrop is a project focused on enhancing the efficiency of large language models through an entropy-based pruning strategy. By analyzing the entropy of hidden representations within Transformer-based models, we propose a method that reduces model size while maintaining performance, offering a promising direction for efficient model deployment.

## Project Description
EntroDrop is a project focused on enhancing the efficiency of large language models through an entropy-based pruning strategy. By analyzing the entropy of hidden representations within Transformer-based models, we propose a method that reduces model size while maintaining performance, offering a promising direction for efficient model deployment.

## Installation Instructions
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SalesforceAIResearch/EntroDrop.git
   ```

2. Navigate to the project directory:
   ```bash
   cd EntroDrop
   ```

3. Install the required dependencies using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To use the EntroDrop project, follow these steps:

1. Run the main script with your model path:
   ```bash
   python main.py --model_path YourModelPath
   ```
   This will generate the attention detection sequence and add it to `layer_seq.txt`.

2. Update the `config.json` file with the necessary parameters, for example:
   ```json
   {
     "_name_or_path": "YourModelPath",
     "architectures": ["LlamaForCausalLMDrop"],
     "attention_bias": false,
     "attention_dropout": 0.0,
     "bos_token_id": 128000,
     "drop_layers_order": "29_30_28_27_26_23_25_24_21_22_18_17_19_20_16_15_14_13_10_2_3_12_7_5_8_0_6_9_4_1_11_31",
     "drop_layers": 0,
     "eos_token_id": 128009,
     "head_dim": 128,
     "hidden_act": "silu",
     "hidden_size": 4096,
     "initializer_range": 0.02,
     "intermediate_size": 14336,
     "max_position_embeddings": 8192,
     "mlp_bias": false,
     "model_type": "llama",
     "num_attention_heads": 32,
     "num_hidden_layers": 32,
     "num_key_value_heads": 8,
     "pretraining_tp": 1,
     "rms_norm_eps": 1e-05,
     "rope_scaling": null,
     "rope_theta": 500000.0,
     "tie_word_embeddings": false,
     "torch_dtype": "float16",
     "transformers_version": "4.48.0",
     "use_cache": true,
     "vocab_size": 128256
   }
   ```

3. Evaluate the model:
   ```bash
   bash evaluate.sh
   ```

## Features
- Entropy-based pruning strategy for Transformer-based models.
- Enhanced efficiency while maintaining performance.
- Empirical analysis of entropy trends in hidden representations.

## Evaluation Results
[Provide a summary of the evaluation results and performance metrics.]

## Contributing
[Provide guidelines for contributing to the project.]

## License
[Provide information about the project's license.]

## Contact Information
[Provide contact information for the maintainers or contributors.]


