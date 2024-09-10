# LEP


Official PyTorch implementation of LEP: Leveraging Local Entropy Pruning for Sparsity in Large Language Models
      
## Table of contents

* [Installation](#installation)
* [Usage](#Usage)



## Installation 
--- 
Installation instructions can be found in [INSTALL.md](INSTALL.md).



## Usage

We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`, `lep`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--eval_zero_shot`: Whether to compute accuracy on a zero-shot dataset. 
- `--save`: Specifies the directory where the result will be stored.


### Script example of pruning model

Below is an example command for pruning LLaMA-7B with LEP, to achieve unstructured 50% sparsity.
```
    python main.py \
    --model "Enoch/llama-7b-hf" \
    --prune_method "lep" \
    --sparsity_ratio 0.5 \
    --sparsity_type "unstructured" \
    --save 
```

For structured N:M sparsity, set the argument `--sparsity_type` to "2:4" or "4:8". An illustrative command is provided below:

```
    python main.py \
    --model "Enoch/llama-7b-hf" \
    --prune_method "lep" \
    --sparsity_ratio 0.5 \
    --sparsity_type "2:4" \
    --save 
```

### Zero-shot Evaluation
The following is an example of a command that uses LEP to prune the LLaMA-7B for the 50% condition with zero-shot accuracy.
```
    python main.py \
    --model "Enoch/llama-7b-hf" \
    --prune_method "lep" \
    --sparsity_ratio 0.5 \
    --sparsity_type "unstructured" \
    --save 
    --save_model "pruned_model_llama-7b-hf" \
    --eval_zero_shot
```

### Acknowledgement
The repository is build upon the  [Wanda](https://github.com/locuslab/wanda) and [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repositories.




