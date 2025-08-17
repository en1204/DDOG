import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Set, Tuple, Optional, Union
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

def build_graph_from_triples(triples: List[Tuple[str, str, str]]) -> nx.DiGraph:
    """从三元组构建有向图"""
    G = nx.DiGraph()
    for head, relation, tail in triples:
        G.add_edge(head, tail, relation=relation)
    return G

def filter_by_entity_alignment(
    candidate_triples: List[Tuple[str, str, str]],
    path_triples: List[Tuple[str, str, str]]
) -> List[Tuple[str, str, str]]:
    """
    基于实体对齐原则筛选候选三元组
    Args:
        candidate_triples: 候选三元组列表
        path_triples: 初始路径三元组列表
    Returns:
        List[Tuple[str, str, str]]: 经过实体对齐筛选的三元组列表
    """
    path_entities = set()
    for h, _, t in path_triples:
        path_entities.add(h)
        path_entities.add(t)
        
    aligned_triples = []
    for triple in candidate_triples:
        head, _, tail = triple
        if head in path_entities or tail in path_entities:
            aligned_triples.append(triple)
            
    return aligned_triples

def check_path_connectivity(
    triple: Tuple[str, str, str],
    path_triples: List[Tuple[str, str, str]],
    candidate_triples: List[Tuple[str, str, str]],
    gnn_model: Any,
    reach_threshold: float = 0.7
) -> float:
    """
    使用GNN评估三元组的间接连通性
    Args:
        triple: 待评估的三元组
        path_triples: 初始路径三元组列表
        candidate_triples: 所有候选三元组列表
        gnn_model: GNN模型
        reach_threshold: 可达性阈值
    Returns:
        float: 连通性分数
    """
    head, _, tail = triple
    

    temp_graph = build_graph_from_triples(path_triples + candidate_triples)
    
    connectivity_score = gnn_model.compute_path_score(
        temp_graph,
        head,
        tail,
        reach_threshold
    )
    
    return connectivity_score
def compute_semantic_relevance(
    triple: Tuple[str, str, str],
    query: str,
    path_triples: List[Tuple[str, str, str]],
    llm_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """
    使用Meta-Llama-3-8B-Instruct计算候选三元组与查询的语义相关性
    Args:
        triple: 待评估的三元组
        query: 原始查询
        path_triples: 当前推理路径三元组列表
        llm_model: Meta-Llama-3-8B-Instruct模型
        tokenizer: 对应的分词器
        device: 计算设备
    Returns:
        float: 语义相关性分数 [0,1]
    """
    head, relation, tail = triple
    
    if path_triples:
        path_context = " -> ".join([
            f"({h}, {r}, {t})" 
            for h, r, t in path_triples
        ])
    else:
        path_context = "No existing path"

    prompt = f"""[INST] Given a question and the current reasoning path, evaluate how relevant the new triple is for answering the question.

Question: {query}
Current reasoning path: {path_context}
New triple: ({head}, {relation}, {tail})

Rate the relevance on a scale of 0 to 1, where:
0: Completely irrelevant
1: Highly relevant

Only output a single number between 0 and 1. [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
       
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=5,
                num_return_sequences=1,
                temperature=0.1,  
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        score_text = response.split("[/INST]")[-1].strip()
        try:
            score = float(score_text)
           
            score = max(0.0, min(1.0, score))
        except ValueError:
            logger.warning(f"Failed to parse score from response: {score_text}")
            score = 0.5  
            
    except Exception as e:
        logger.error(f"Error in semantic relevance computation: {str(e)}")
        score = 0.5  
        
    return score

def batch_compute_semantic_relevance(
    triples: List[Tuple[str, str, str]],
    query: str,
    path_triples: List[Tuple[str, str, str]],
    llm_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[float]:
    """
    批量计算多个三元组的语义相关性分数
    Args:
        triples: 待评估的三元组列表
        query: 原始查询
        path_triples: 当前推理路径三元组列表
        llm_model: Meta-Llama-3-8B-Instruct模型
        tokenizer: 对应的分词器
        batch_size: 批处理大小
        device: 计算设备
    Returns:
        List[float]: 语义相关性分数列表
    """
    scores = []
    
    for i in range(0, len(triples), batch_size):
        batch_triples = triples[i:i + batch_size]
        batch_scores = [
            compute_semantic_relevance(
                triple, 
                query, 
                path_triples, 
                llm_model, 
                tokenizer,
                device
            )
            for triple in batch_triples
        ]
        scores.extend(batch_scores)
        
    return scores 

def compute_final_score(
    connectivity_score: float,
    semantic_score: float,
    alpha: float = 0.6,
    beta: float = 0.4
) -> float:
    """
    计算候选三元组的最终综合分数
    Args:
        connectivity_score: 连通性分数
        semantic_score: 语义相关性分数
        alpha: 连通性权重
        beta: 语义相关性权重
    Returns:
        float: 综合分数 [0,1]
    """
    assert abs(alpha + beta - 1.0) < 1e-6, "Weights must sum to 1"
    return alpha * connectivity_score + beta * semantic_score

def validate_triple(triple: Tuple[str, str, str]) -> bool:
    """
    验证三元组的有效性
    Args:
        triple: (head, relation, tail) 三元组
    Returns:
        bool: 是否有效
    """
    head, relation, tail = triple
    
    if not all([head, relation, tail]):
        return False
        
    if not all(isinstance(x, str) for x in [head, relation, tail]):
        return False
        
    if any(len(x.strip()) == 0 for x in [head, relation, tail]):
        return False
        
    return True

def merge_subgraphs(
    subgraphs: List[nx.DiGraph],
    similarity_threshold: float = 0.8
) -> nx.DiGraph:
    """
    合并多个子图
    Args:
        subgraphs: 子图列表
        similarity_threshold: 实体合并的相似度阈值
    Returns:
        nx.DiGraph: 合并后的图
    """
    merged_graph = nx.DiGraph()
    
    for sg in subgraphs:
        for node in sg.nodes():
            if node not in merged_graph:
                merged_graph.add_node(node)
        
        for u, v, data in sg.edges(data=True):
            if not merged_graph.has_edge(u, v):
                merged_graph.add_edge(u, v, **data)
    
    return merged_graph

def find_paths_between_entities(
    G: nx.DiGraph,
    source: str,
    target: str,
    max_length: int = 3
) -> List[List[Tuple[str, str, str]]]:
    """
    查找两个实体之间的所有有效路径
    Args:
        G: 知识图谱
        source: 源实体
        target: 目标实体
        max_length: 最大路径长度
    Returns:
        List[List[Tuple[str, str, str]]]: 路径列表，每个路径是三元组列表
    """
    paths = []
    for path in nx.all_simple_paths(G, source, target, cutoff=max_length):
        triple_path = []
        for i in range(len(path)-1):
            head = path[i]
            tail = path[i+1]
            relation = G[head][tail]['relation']
            triple_path.append((head, relation, tail))
        paths.append(triple_path)
    return paths

def compute_string_similarity(s1: str, s2: str) -> float:
    """
    计算两个字符串的相似度
    Args:
        s1: 第一个字符串
        s2: 第二个字符串
    Returns:
        float: 相似度分数 [0,1]
    """
    return SequenceMatcher(None, s1, s2).ratio()

def get_node_neighbors(
    G: nx.DiGraph,
    node: str,
    k: int = 1
) -> Set[str]:
    """
    获取节点的k跳邻居
    Args:
        G: 知识图谱
        node: 目标节点
        k: 跳数
    Returns:
        Set[str]: 邻居节点集合
    """
    neighbors = set()
    current_nodes = {node}
    
    for _ in range(k):
        next_nodes = set()
        for n in current_nodes:
            next_nodes.update(G.predecessors(n))
            next_nodes.update(G.successors(n))
        neighbors.update(next_nodes)
        current_nodes = next_nodes
        
    neighbors.discard(node)
    return neighbors

def get_relation_distribution(
    triples: List[Tuple[str, str, str]]
) -> Dict[str, int]:
    """
    获取关系分布
    Args:
        triples: 三元组列表
    Returns:
        Dict[str, int]: 关系及其出现次数
    """
    relation_counts = defaultdict(int)
    for _, relation, _ in triples:
        relation_counts[relation] += 1
    return dict(relation_counts) 