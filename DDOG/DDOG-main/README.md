# DDOG: Optimizing Multi-Hop Inference via Dual-Driven Retrieval and Reasoning Path
  


## Prepare Data 
The datasets used in this work, HotPotQA, WikiHop, and CWQ, are publicly available and can be accessed through their respective official repositories: [HotPotQA](https://hotpotqa.github.io/), [WikiHop](https://huggingface.co/datasets/MoE-UNC/wikihop/), [CWQ]( https://github.com/yuancu complex-web-questions-dataset/).  


### 1. KG Generation 

```
python generate_knowledge_triples.py \
    --dataset hotpotqa \
    --input_data_file  data/HotPotQA/dev.json \
    --save_data_file  data/HotPotQA/dev_with_kgs.json 
```

### 2. GNN Retriever
### 3. Dynamic Retriever

### 4. Reasoning Chain Construction

python construct_reasoning_chains.py \
  --dataset hotpotqa \
  --input_data_file data/HotPotQA/dev_with_kgs.json \
  --save_data_file data/HotPotQA/dev_with_reasoning_chains.json \
  --calculate_ranked_prompt_indices \
  --max_chain_length 4  

### 5. Evaluation of DDOG

python evaluation.py \
  --test_file data/hotpotqa/dev_with_reasoning_chains.json \
  --reader llama3 \
  --context_type triples \
  --n_context 5 

 


 

 


 
