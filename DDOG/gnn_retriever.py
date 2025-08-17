import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set, Union
import numpy as np
from collections import defaultdict
from utils.graph_utils import (
    build_graph_from_triples,
    filter_by_entity_alignment,
    check_path_connectivity,
    compute_semantic_relevance,
    compute_final_score,
    find_paths_between_entities,
    batch_compute_semantic_relevance
)
from utils.ercr_utils import ERCRChecker
from dynamic_retriever import DynamicRetriever
logger=logging.getLogger(__name__)

class HallucinationDetector:
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        nli_model_name: str = "microsoft/deberta-v3-base",
        threshold: float = -2.5,
        lambda_coef: float = 0.4,
        confidence_threshold: float = 0.75
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.threshold = threshold
        

        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
        
    
        self.lambda_coef = lambda_coef  
        self.confidence_threshold = confidence_threshold  

    def compute_seq_logprob(self, text: str) -> float:
        """
        计算序列对数概率
        Args:
            text: 输入文本
        Returns:
            float: 序列对数概率
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, :-1, :]  
            labels = inputs["input_ids"][:, 1:]  
            

            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(
                -1, labels.unsqueeze(-1)
            ).squeeze(-1)
            

            seq_logprob = token_log_probs.mean().item()
            
        return seq_logprob

    def compute_nli_consistency(self, query: str, answer: str, triples: List[Tuple[str, str, str]]) -> float:
        """
        Args:
            query: 查询问题
            answer: 生成的答案
            triples: 知识图谱三元组列表
        Returns:
            float: NLI一致性概率值 [0,1]
        """
        premise = f"查询：{query}\n"
        premise += "知识图谱信息：\n"
        for head, relation, tail in triples:
            premise += f"- {head} {relation} {tail}\n"
        

        hypothesis = answer
        
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
    
            entailment_prob = probs[0, 2].item()  
            
        return entailment_prob

    def check_hallucination(self, text: str, query: str = "", triples: List[Tuple[str, str, str]] = None) -> Tuple[bool, float, Dict]:
        """
        检查文本是否存在幻觉，结合Seq-Logprob和NLI语义一致性
        Args:
            text: 输入文本（答案）
            query: 查询问题
            triples: 知识图谱三元组列表
        Returns:
            Tuple[bool, float, Dict]: (是否存在幻觉, 最终置信度分数, 详细置信度信息)
        """
        seq_logprob = self.compute_seq_logprob(text)
        

        seq_confidence = 1.0 / (1.0 + np.exp(-(seq_logprob - self.threshold)))
        
        nli_confidence = 0.0
        if query and triples:
            nli_confidence = self.compute_nli_consistency(query, text, triples)
        
        final_confidence = (self.lambda_coef * seq_confidence + 
                          (1 - self.lambda_coef) * nli_confidence)
        
        has_hallucination = final_confidence < self.confidence_threshold
        
        confidence_info = {
            'seq_logprob': seq_logprob,
            'seq_confidence': seq_confidence,
            'nli_confidence': nli_confidence,
            'final_confidence': final_confidence,
            'lambda_coef': self.lambda_coef,
            'threshold': self.confidence_threshold
        }
        
        return has_hallucination, final_confidence, confidence_info

class GNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.gnn_layers = nn.ModuleList()
        
        self.gnn_layers.append(
            GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        )
        
        for _ in range(num_layers - 2):
            self.gnn_layers.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            )
            
        self.gnn_layers.append(
            GATConv(hidden_dim, output_dim // heads, heads=heads, dropout=dropout)
        )
        
        self.node_classifier = nn.Linear(output_dim, 1)
        

        self.path_attention = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        query_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播"""
        # GNN编码
        for layer in self.gnn_layers[:-1]:
            x = layer(x, edge_index)
            x = torch.relu(x)
        
        x = self.gnn_layers[-1](x, edge_index)
        node_embeddings = x
        
        node_scores = None
        if query_embedding is not None:
            query_expanded = query_embedding.unsqueeze(1).expand(-1, node_embeddings.size(0), -1)
            node_expanded = node_embeddings.unsqueeze(0).expand(query_embedding.size(0), -1, -1)
            
            attention_input = torch.cat([query_expanded, node_expanded], dim=-1)
            node_scores = self.node_classifier(attention_input).squeeze(-1)
        
        return node_embeddings, node_scores

    def compute_path_score(
        self,
        node_embeddings: torch.Tensor,
        source_idx: int,
        target_idx: int,
        threshold: float = 0.7
    ) -> float:
        """计算路径可达性分数"""
        source_emb = node_embeddings[source_idx]
        target_emb = node_embeddings[target_idx]
        
        path_input = torch.cat([source_emb, target_emb], dim=-1)
        attention_score = self.path_attention(path_input)
        path_score = float(attention_score.sigmoid().item())
        
        return 1.0 if path_score >= threshold else 0.0

class GraphRetriever:
    def __init__(
        self,
        model_name="bert-large-portuguese-cased-sts",
        llm_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        hidden_dim=256,
        num_layers=3,
        dropout=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        alpha: float = 0.6,
        beta: float = 0.4,
        reach_threshold: float = 0.7
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name).to(device)
        self.gnn_encoder = GNNEncoder(
            input_dim=768,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        

        self.hallucination_detector = HallucinationDetector(
            model_name=llm_model_name,
            lambda_coef=0.4,  
            confidence_threshold=0.75  
        )
        
        # 初始化ERCR检查器
        self.ercr_checker = ERCRChecker(alpha=0.6, beta=0.4, threshold=0.9)
        
        # 初始化动态检索器
        self.dynamic_retriever = DynamicRetriever(
            model_name=llm_model_name,
            device=device
        )
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
        
        self.alpha = alpha
        self.beta = beta
        self.reach_threshold = reach_threshold
        
    def retrieve(self, graph_data: Dict, query: str, max_nodes: int = 50) -> Dict:
        """检索相关子图并处理幻觉"""

        subgraph = self.retrieve_subgraph(graph_data, query, max_nodes)
        
        nx_subgraph = self._build_nx_subgraph(subgraph, graph_data)
        
        is_ercr_compliant, ercr_analysis = self.ercr_checker.check_ercr_compliance(
            nx_subgraph, query, self
        )
        
        if is_ercr_compliant:
            logger.info("子图满足ERCR规则，进行LLaMA3推理")
            
            answer = self._generate_answer_with_llama3(query, nx_subgraph)
            
            triples = self._extract_triples_from_subgraph(subgraph)
            

            has_hallucination, confidence, confidence_info = self.hallucination_detector.check_hallucination(
                answer, query, triples
            )
            
            should_trigger = self.dynamic_retriever.should_trigger_dynamic_retrieval(
                is_ercr_compliant=True,
                has_hallucination=has_hallucination,
                confidence=confidence,
                confidence_threshold=self.hallucination_detector.confidence_threshold
            )
            
            if should_trigger:
                logger.warning(f"触发动态检索，最终置信度: {confidence:.4f} < {self.hallucination_detector.confidence_threshold}")
                logger.info(f"置信度详情: Seq置信度={confidence_info['seq_confidence']:.4f}, NLI置信度={confidence_info['nli_confidence']:.4f}")
                
                dynamic_result = self.dynamic_retriever.dynamic_retrieval_pipeline(
                    original_query=query,
                    subgraph_info={'subgraph': subgraph},
                    ercr_analysis=ercr_analysis
                )
                
                return {
                    'subgraph': subgraph,
                    'answer': answer,
                    'confidence': confidence,
                    'confidence_info': confidence_info,
                    'ercr_analysis': ercr_analysis,
                    'is_ercr_compliant': True,
                    'has_hallucination': has_hallucination,
                    'dynamic_retrieval_result': dynamic_result
                }
            else:
                return {
                    'subgraph': subgraph,
                    'answer': answer,
                    'confidence': confidence,
                    'confidence_info': confidence_info,
                    'ercr_analysis': ercr_analysis,
                    'is_ercr_compliant': True,
                    'has_hallucination': False
                }
        else:
            logger.warning(f"子图不满足ERCR规则: {ercr_analysis}")
            
            dynamic_result = self.dynamic_retriever.dynamic_retrieval_pipeline(
                original_query=query,
                subgraph_info={'subgraph': subgraph},
                ercr_analysis=ercr_analysis
            )
            
            return {
                'subgraph': subgraph,
                'ercr_analysis': ercr_analysis,
                'is_ercr_compliant': False,
                'dynamic_retrieval_result': dynamic_result
            }
    def retrieve_subgraph(
        self,
        graph_data: Dict,
        query: str,
        max_nodes: int = 50,
        min_score_threshold: float = 0.3
    ) -> Dict:
        """检索相关子图"""
       
        node_features = graph_data['node_features'].to(self.device)
        edge_index = graph_data['edge_index'].to(self.device)
        
        
        query_encoding = self.tokenizer(
            query,
            return_tensors='pt',
            max_length=128,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.text_encoder(**query_encoding).pooler_output
            
        
        node_embeddings, node_scores = self.gnn_encoder(
            node_features,
            edge_index,
            query_embedding
        )
        
       
        selected_mask = node_scores >= min_score_threshold
        selected_indices = torch.where(selected_mask)[0][:max_nodes]
        
       
        subgraph = {
            'selected_indices': selected_indices,
            'node_scores': node_scores[selected_indices],
            'edge_index': self._filter_edges(edge_index, selected_indices)
        }
        
        return subgraph
    
    def check_ercr_compliance(self, subgraph: Dict) -> bool:
        """检查子图是否符合ERCR规则"""
        if not subgraph['selected_indices'].any():
            return False
            

        edge_index = subgraph['edge_index']
        if edge_index.size(1) == 0:
            return False
            

        node_scores = subgraph['node_scores']
        if torch.mean(node_scores) < 0.3:  
            return False
            

        source_nodes = edge_index[0].unique()
        target_nodes = edge_index[1].unique()
        if len(source_nodes) == 0 or len(target_nodes) == 0:
            return False
            
        return True

    def _generate_answer(self, query: str, subgraph: Dict) -> str:
        """使用LLM生成答案"""

        prompt = f"基于以下知识图谱信息回答问题：\n\n问题：{query}\n\n知识图谱信息：\n"
        

        for i, node_id in enumerate(subgraph['selected_indices']):
            node_score = subgraph['node_scores'][i]
            prompt += f"实体{node_id}(相关度:{node_score:.2f})\n"
        

        edge_index = subgraph['edge_index']
        for i in range(edge_index.size(1)):
            head, tail = edge_index[:, i]
            prompt += f"{head} -> {tail}\n"
            
        prompt += "\n请根据上述信息回答问题。"
        
 
        inputs = self.hallucination_detector.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.hallucination_detector.model.generate(
                **inputs,
                max_length=128,
                num_return_sequences=1,
                temperature=0.7
            )
            
        answer = self.hallucination_detector.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return answer

    def _generate_answer_with_llama3(self, query: str, nx_subgraph: nx.Graph) -> str:
        """
        使用Meta-LLaMA-3-8B-Instruct模型生成答案
        Args:
            query: 查询问题
            nx_subgraph: NetworkX格式的子图
        Returns:
            str: LLaMA3生成的答案
        """
        prompt = f"基于以下知识图谱信息回答问题：\n\n问题：{query}\n\n知识图谱信息：\n"
        
        for node_id in nx_subgraph.nodes():
            node_data = nx_subgraph.nodes[node_id]
            node_label = node_data.get('label', f"entity_{node_id}")
            prompt += f"实体: {node_label}\n"
        
        for source, target, edge_data in nx_subgraph.edges(data=True):
            relation = edge_data.get('relation', 'relation')
            source_label = nx_subgraph.nodes[source].get('label', f"entity_{source}")
            target_label = nx_subgraph.nodes[target].get('label', f"entity_{target}")
            prompt += f"{source_label} -{relation}-> {target_label}\n"
            
        prompt += "\n请根据上述信息回答问题。"
        

        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=128,
                num_return_sequences=1,
                temperature=0.7
            )
            
        answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def _build_nx_subgraph(self, subgraph: Dict, graph_data: Dict) -> nx.Graph:
        """
        构建NetworkX子图用于ERCR检查
        Args:
            subgraph: 子图数据
            graph_data: 原始图数据
        Returns:
            nx.Graph: NetworkX图对象
        """
        nx_graph = nx.Graph()
        
        for node_idx in subgraph['selected_indices']:
            node_idx = node_idx.item()
            if hasattr(self, 'id2entity'):
                node_label = self.id2entity.get(node_idx, f"entity_{node_idx}")
            else:
                node_label = f"entity_{node_idx}"
            nx_graph.add_node(node_idx, label=node_label)
        
        edge_index = subgraph['edge_index']
        for i in range(edge_index.size(1)):
            head_idx = edge_index[0, i].item()
            tail_idx = edge_index[1, i].item()
            
            if hasattr(self, 'id2relation'):
                relation = self.id2relation.get(i, f"relation_{i}")
            else:
                relation = f"relation_{i}"
            
            nx_graph.add_edge(head_idx, tail_idx, relation=relation)
        
        return nx_graph
    

    def _extract_triples_from_subgraph(self, subgraph: Dict) -> List[Tuple[str, str, str]]:
        """
        从子图中提取三元组信息
        Args:
            subgraph: 子图数据
        Returns:
            List[Tuple[str, str, str]]: 三元组列表
        """
        triples = []
        edge_index = subgraph['edge_index']
        
        for i in range(edge_index.size(1)):
            head_idx = edge_index[0, i].item()
            tail_idx = edge_index[1, i].item()
            

            if hasattr(self, 'id2entity') and hasattr(self, 'id2relation'):
                head_entity = self.id2entity.get(head_idx, f"entity_{head_idx}")
                tail_entity = self.id2entity.get(tail_idx, f"entity_{tail_idx}")
                relation = self.id2relation.get(i, "relation")
            else:
                head_entity = f"entity_{head_idx}"
                tail_entity = f"entity_{tail_idx}"
                relation = "relation"
            
            triples.append((head_entity, relation, tail_entity))
        
        return triples

    def _dynamic_retrieval(
        self,
        query: str,
        initial_subgraph: Dict,
        external_docs: List[str] = None
    ) -> List[Tuple[str, str, str]]:
        """
        动态检索机制
        Args:
            query: 原始查询
            initial_subgraph: 初始子图
            external_docs: 外部文档列表
        Returns:
            List[Tuple[str, str, str]]: 候选三元组列表
        """
   
        reasoning_path = self._get_reasoning_path(initial_subgraph)
        

        extended_query = self._build_extended_query(query, reasoning_path)
        
    
        candidate_triples = self._extract_triples_from_docs(extended_query, external_docs)
        
        return candidate_triples

    def _build_extended_query(
        self,
        query: str,
        reasoning_path: List[Tuple[str, str, str]]
    ) -> str:
        """
        构建扩展查询
        Args:
            query: 原始查询
            reasoning_path: 推理路径三元组列表
        Returns:
            str: 扩展后的查询
        """
    
        prompt = f"原始问题：{query}\n\n"
        prompt += "已知推理路径：\n"
        

        for head, relation, tail in reasoning_path:
            prompt += f"- {head} -{relation}-> {tail}\n"
            
        prompt += "\n基于以上信息，请生成一个更详细的查询，用于从文档中检索相关知识。"
        
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=256,
                num_return_sequences=1,
                temperature=0.7
            )
            
        extended_query = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return extended_query

    def _extract_triples_from_docs(
        self,
        extended_query: str,
        docs: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        从文档中提取候选三元组
        Args:
            extended_query: 扩展查询
            docs: 文档列表
        Returns:
            List[Tuple[str, str, str]]: 候选三元组列表
        """

        prompt = "请从以下文档中提取与查询相关的知识三元组（主体-关系-客体）：\n\n"
        prompt += f"查询：{extended_query}\n\n"
        prompt += "文档内容：\n"
        
        for i, doc in enumerate(docs, 1):
            prompt += f"文档{i}：{doc}\n"
            
        prompt += "\n请以JSON格式返回提取的三元组，格式如下：\n"
        prompt += '{"triples": [["主体", "关系", "客体"], ...]}'
        
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.3
            )
            
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            result = json.loads(response)
            triples = result.get("triples", [])
            return [tuple(triple) for triple in triples]
        except json.JSONDecodeError:
            logger.error("无法解析LLM返回的JSON响应")
            return []
    
    def _get_reasoning_path(self, subgraph: Dict) -> List[Tuple[str, str, str]]:
        """
        从子图中提取推理路径 
        Args:
            subgraph: 包含边索引和节点信息的子图
        Returns:
            List[Tuple[str, str, str]]: 推理路径三元组列表 [(head, relation, tail), ...]
        """
        reasoning_path = []
        edge_index = subgraph['edge_index']
        selected_indices = subgraph['selected_indices']
        node_scores = subgraph['node_scores']
        
        sorted_indices = torch.argsort(node_scores, descending=True)
        sorted_nodes = selected_indices[sorted_indices]
        
        adj_dict = defaultdict(list)
        for i in range(edge_index.size(1)):
            head_idx = edge_index[0, i].item()
            tail_idx = edge_index[1, i].item()
            if head_idx in selected_indices and tail_idx in selected_indices:
                adj_dict[head_idx].append(tail_idx)
        

        visited = set()
        def dfs(current_node, path):
            if len(path) > 5:  
                return
            visited.add(current_node)
            for next_node in adj_dict[current_node]:
                if next_node not in visited:
                    
                    head = self.id2entity[current_node]
                    tail = self.id2entity[next_node]
                    relation = self.get_edge_relation(current_node, next_node)
                    reasoning_path.append((head, relation, tail))
                    
                    dfs(next_node, path + [next_node])
            visited.remove(current_node)
        
   
        for start_node in sorted_nodes[:3]:  
            if start_node.item() not in visited:
                dfs(start_node.item(), [start_node.item()])
        
        return reasoning_path   

    def _filter_edges(self, edge_index: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
        """过滤边，只保留选中节点之间的边"""
        mask = torch.isin(edge_index[0], selected_indices) & torch.isin(edge_index[1], selected_indices)
        return edge_index[:, mask] 
    
    def filter_and_integrate_candidates(
        self,
        candidate_triples: List[Tuple[str, str, str]],
        query: str,
        initial_path: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        筛选和整合候选三元组
        Args:
            candidate_triples: 候选三元组列表
            query: 原始查询
            initial_path: 初始推理路径
        Returns:
            List[Tuple[str, str, str]]: 筛选后的三元组列表
        """

        aligned_triples = filter_by_entity_alignment(candidate_triples, initial_path)
        
        unaligned_triples = set(candidate_triples) - set(aligned_triples)
        connected_triples = []
        
        for triple in unaligned_triples:
            connectivity_score = check_path_connectivity(
                triple,
                initial_path,
                candidate_triples,
                self.gnn_encoder,
                self.reach_threshold
            )
            if connectivity_score > 0:
                connected_triples.append(triple)
        
        filtered_triples = aligned_triples + connected_triples
        scored_triples = []
        
        semantic_scores = batch_compute_semantic_relevance(
            filtered_triples,
            query,
            initial_path,
            self.llm_model,
            self.llm_tokenizer,
            device=self.device
        )
        
        for triple, semantic_score in zip(filtered_triples, semantic_scores):

            connectivity_score = check_path_connectivity(
                triple,
                initial_path,
                filtered_triples,
                self.gnn_encoder,
                self.reach_threshold
            )
            final_score = compute_final_score(
                connectivity_score,
                semantic_score,
                self.alpha,
                self.beta
            )
            scored_triples.append((triple, final_score))
        
        scored_triples.sort(key=lambda x: x[1], reverse=True)
        return [triple for triple, _ in scored_triples]

    def construct_final_path(
        self,
        query: str,
        query_entity: str,
        target_entity: str,
        filtered_triples: List[Tuple[str, str, str]],
        initial_path: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        构建最终推理路径
        Args:
            query: 查询问题
            query_entity: 查询实体
            target_entity: 目标实体
            filtered_triples: 筛选后的三元组
            initial_path: 初始推理路径
        Returns:
            List[Tuple[str, str, str]]: 最终推理路径
        """
        all_triples = list(set(filtered_triples + initial_path))
        graph = build_graph_from_triples(all_triples)
        paths = find_paths_between_entities(graph, query_entity, target_entity)
        if not paths:
            return initial_path  
            
   
        path_scores = []
        for path in paths:
            
            path_semantic_score = batch_compute_semantic_relevance(
                path,
                query,
                [], 
                self.llm_model,
                self.llm_tokenizer,
                device=self.device
            )
            
            
            avg_semantic_score = sum(path_semantic_score) / len(path_semantic_score)
            

            connectivity_score = 1.0  
            
            final_score = compute_final_score(
                connectivity_score,
                avg_semantic_score,
                self.alpha,
                self.beta
            )
            
            path_scores.append((path, final_score))
        
        best_path = max(path_scores, key=lambda x: x[1])[0]
        return best_path

    def generate_final_answer(
        self,
        query: str,
        reasoning_path: List[Tuple[str, str, str]]
    ) -> str:
        """
        基于最终推理路径生成答案
        Args:
            query: 查询问题
            reasoning_path: 推理路径
        Returns:
            str: 生成的答案
        """
        path_str = " -> ".join([
            f"({h}, {r}, {t})" 
            for h, r, t in reasoning_path
        ])
        
        prompt = f"""[INST] Based on the following reasoning path, please answer the question.

Question: {query}
Reasoning path: {path_str}

Provide a clear and concise answer. [/INST]"""

        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.split("[/INST]")[-1].strip()
            
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            answer = "Failed to generate answer."
            
        return answer