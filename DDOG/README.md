# DDOG: Optimizing Multi-Hop Inference via Dual-Driven Retrieval and Reasoning Path 

## Introduction

Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, mitigating issues like hallucinations, limited coverage, and outdated information. While structured data such as knowledge graphs (KGs) are widely used, RAG struggles with complex reasoning over them. GraphRAG improves multi-hop reasoning by leveraging graph structures, but its performance is limited by incomplete KGs, rigid path searches, and semantic gaps. To address these challenges, we propose Dual-Driven Optimization on GraphRAG (DDOG). DDOG combines static graph retrieval and dynamic knowledge expansion to enhance multi-hop reasoning and answer reliability. Specifically, it uses a GNN to retrieve an initial subgraph guided by ERCR rules, generates a preliminary answer via an LLM, and applies a hallucination detector. When needed, a dynamic retrieval mechanism optimizes reasoning paths by filtering candidate triples, producing a final, reliable answer.
  


## Prepare Data 
The datasets used in this work, HotPotQA, WikiHop, and CWQ, are publicly available and can be accessed through their respective official repositories: [HotPotQA](https://hotpotqa.github.io/), [WikiHop](https://huggingface.co/datasets/MoE-UNC/wikihop/), [CWQ]( https://github.com/yuancu/ complex-web-questions-dataset/).  

## Download the Llama 3 Model

Go to Hugging Face:https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct. You will need to share your contact information with Meta to access this model.

### 1. KG Generation 

python generate_knowledge_triples.py \
    --dataset hotpotqa \
    --input_data_file  data/HotPotQA/dev.json \
    --save_data_file  data/HotPotQA/dev_with_kgs.json 


### 2. GNN Retriever

python gnn_retriever.py \
    --dataset hotpotqa \
    --input_data_file data/HotPotQA/dev_with_kgs.json \
    --save_data_file data/HotPotQA/dev_with_gnn_scores.json 
    

### 3. Dynamic Retriever

python dynamic_retriever.py \
    --dataset hotpotqa \
    --input_data_file data/HotPotQA/dev_with_gnn_scores.json \
    --save_data_file data/HotPotQA/dev_with_dynamic_retrieval.json \
    --max_candidates 40

### 4. Reasoning Chain Construction

python construct_reasoning_chains.py \
    --dataset hotpotqa \
    --input_data_file data/HotPotQA/dev_with_dynamic_retrieval.json \
    --save_data_file data/HotPotQA/dev_with_reasoning_chains.json \
    --max_chain_length 4  

### 5. Evaluation of DDOG

python evaluation.py \
    --test_file data/HotPotQA/dev_with_reasoning_chains.json \
    --reader llama3 \
    --context_type triples \
    --n_context 5

 


 

 


 
