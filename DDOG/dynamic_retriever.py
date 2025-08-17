import torch
import logging
from typing import List, Dict, Tuple, Optional, Set
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import requests
import json
from utils.graph_utils import (
    validate_triple,
    filter_candidate_triples,
    evaluate_triple_quality
)

logger = logging.getLogger(__name__)

class DynamicRetriever:
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_confidence: float = 0.5,
        wikipedia_api_url: str = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    ):
        """
        初始化动态检索器
        Args:
            model_name: 使用的语言模型名称
            device: 计算设备
            max_length: 最大序列长度
            temperature: 采样温度
            top_p: nucleus采样参数
            min_confidence: 最小置信度阈值
            wikipedia_api_url: Wikipedia API URL
        """
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.min_confidence = min_confidence
        self.wikipedia_api_url = wikipedia_api_url
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
            
        self.sep_token = " [SEP] "
        self.triple_sep = " | "
    
    def should_trigger_dynamic_retrieval(
        self,
        is_ercr_compliant: bool,
        has_hallucination: bool,
        confidence: float = None,
        confidence_threshold: float = 0.75
    ) -> bool:
        
        if not is_ercr_compliant:
            logger.info("ERCR规则不满足，触发动态检索")
            return True
        
       
        if has_hallucination:
            logger.info("检测到幻觉，触发动态检索")
            return True
        
        
        if confidence is not None and confidence < confidence_threshold:
            logger.info(f"置信度 {confidence:.4f} 低于阈值 {confidence_threshold}，触发动态检索")
            return True
        
        return False
    
    def form_new_query(
        self,
        original_query: str,
        subgraph_info: Dict,
        ercr_analysis: Dict = None
    ) -> str:
        """
        形成新的查询：原始查询 + 子图信息
        Args:
            original_query: 原始查询
            subgraph_info: 子图信息
            ercr_analysis: ERCR分析结果
        Returns:
            str: 新的扩展查询
        """
        new_query = f"原始查询: {original_query}\n"
        
        if 'subgraph' in subgraph_info:
            subgraph = subgraph_info['subgraph']
            new_query += "已知子图信息:\n"
            
            
            if 'selected_indices' in subgraph:
                new_query += f"节点数量: {len(subgraph['selected_indices'])}\n"
            
            if 'edge_index' in subgraph:
                edge_count = subgraph['edge_index'].size(1) if hasattr(subgraph['edge_index'], 'size') else 0
                new_query += f"边数量: {edge_count}\n"
        
        if ercr_analysis:
            new_query += "ERCR分析结果:\n"
            if 'uncovered_entities' in ercr_analysis:
                uncovered = ercr_analysis['uncovered_entities']
                if uncovered:
                    new_query += f"缺失实体: {', '.join(uncovered)}\n"
            
            if 'uncovered_relations' in ercr_analysis:
                uncovered_rel = ercr_analysis['uncovered_relations']
                if uncovered_rel:
                    new_query += f"缺失关系: {', '.join(uncovered_rel)}\n"
        
        new_query += "\n请基于以上信息，从外部知识库中检索相关的知识三元组来补充缺失的信息。"
        
        return new_query
    
    def search_wikipedia(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Dict]:
        """
        从Wikipedia检索相关信息
        Args:
            query: 查询文本
            max_results: 最大结果数量
        Returns:
            List[Dict]: Wikipedia搜索结果
        """
        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results,
                'srnamespace': 0
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            search_results = []
            
            if 'query' in data and 'search' in data['query']:
                for result in data['query']['search']:
                    search_results.append({
                        'title': result['title'],
                        'snippet': result['snippet'],
                        'pageid': result['pageid']
                    })
            
            logger.info(f"从Wikipedia检索到 {len(search_results)} 个结果")
            return search_results
            
        except Exception as e:
            logger.error(f"Wikipedia检索失败: {str(e)}")
            return []
    
    def extract_triples_from_wikipedia(
        self,
        wikipedia_results: List[Dict],
        query: str
    ) -> List[Tuple[str, str, str]]:
        """
        从Wikipedia搜索结果中提取三元组
        Args:
            wikipedia_results: Wikipedia搜索结果
            query: 原始查询
        Returns:
            List[Tuple[str, str, str]]: 提取的三元组列表
        """
        all_triples = []
        
        for result in wikipedia_results:
            prompt = (
                f"从以下Wikipedia文本中提取与查询相关的知识三元组：\n\n"
                f"查询: {query}\n\n"
                f"文本: {result['title']}\n{result['snippet']}\n\n"
                f"请提取相关的知识三元组，格式为：主体|关系|客体\n"
                f"只返回三元组，每行一个："
            )
            
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=256,
                        temperature=0.3,
                        num_return_sequences=1
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                triples = self.extract_triples_from_text(generated_text)
                valid_triples = [t for t in triples if validate_triple(t)]
                
                all_triples.extend(valid_triples)
                
            except Exception as e:
                logger.warning(f"处理Wikipedia结果失败: {str(e)}")
                continue
        
        return all_triples
    
    def dynamic_retrieval_pipeline(
        self,
        original_query: str,
        subgraph_info: Dict,
        ercr_analysis: Dict = None,
        max_triples: int = 20
    ) -> Dict:
        """
        完整的动态检索流程
        Args:
            original_query: 原始查询
            subgraph_info: 子图信息
            ercr_analysis: ERCR分析结果
            max_triples: 最大三元组数量
        Returns:
            Dict: 动态检索结果
        """
        logger.info("开始动态检索流程")
        
        new_query = self.form_new_query(original_query, subgraph_info, ercr_analysis)
        logger.info(f"新查询: {new_query[:200]}...")
        
        wikipedia_results = self.search_wikipedia(original_query)
        
        candidate_triples = self.extract_triples_from_wikipedia(wikipedia_results, new_query)
        
        scored_triples = []
        for triple in candidate_triples:
            quality_score = evaluate_triple_quality(triple, [])
            if quality_score >= self.min_confidence:
                scored_triples.append((triple, quality_score))
        
        scored_triples.sort(key=lambda x: x[1], reverse=True)
        selected_triples = [triple for triple, _ in scored_triples[:max_triples]]
        
        logger.info(f"动态检索完成，获得 {len(selected_triples)} 个高质量三元组")
        
        return {
            'new_query': new_query,
            'wikipedia_results': wikipedia_results,
            'candidate_triples': candidate_triples,
            'selected_triples': selected_triples,
            'triple_scores': dict(scored_triples[:max_triples])
        } 