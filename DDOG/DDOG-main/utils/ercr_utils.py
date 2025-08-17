from typing import List, Set, Dict, Tuple, Optional
import networkx as nx
from collections import defaultdict
import logging
import torch

logger = logging.getLogger(__name__)

class ERCRChecker:
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, threshold: float = 0.9):
        
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        
    def extract_query_elements(self, question: str, model) -> Tuple[Set[str], Set[str]]:
        """
        从问题中提取关键实体和关系
        :param question: 问题文本
        :param model: 模型实例，用于实体和关系提取
        :return: 实体集合和关系集合
        """
        try:
            entities = set(model.extract_entities(question))
            relations = set(model.extract_relations(question))
            return entities, relations
        except Exception as e:
            logger.warning(f"实体关系提取失败: {e}")
            return set(), set()
    
    def check_entity_completeness(
        self,
        subgraph_entities: Set[str],
        query_entities: Set[str]
    ) -> Tuple[bool, float]:
        """
        检查实体完整性：子图必须包含问题中所有实体
        """
        if not query_entities:
            return True, 1.0
            
        covered_entities = query_entities & subgraph_entities
        uncovered_entities = query_entities - subgraph_entities
        
        entity_coverage = len(covered_entities) / len(query_entities)
        is_complete = len(uncovered_entities) == 0
        
        if not is_complete:
            logger.warning(f"实体不完整: 缺失实体 {uncovered_entities}")
            
        return is_complete, entity_coverage
    
    def check_relation_completeness(
        self,
        subgraph_relations: Set[str],
        query_relations: Set[str]
    ) -> Tuple[bool, float]:
        """
        检查关系完整性：子图必须包含问题中所有隐含关系
        """
        if not query_relations:
            return True, 1.0
            
        covered_relations = query_relations & subgraph_relations
        uncovered_relations = query_relations - subgraph_relations
        
        relation_coverage = len(covered_relations) / len(query_relations)
        is_complete = len(uncovered_relations) == 0
        
        if not is_complete:
            logger.warning(f"关系不完整: 缺失关系 {uncovered_relations}")
            
        return is_complete, relation_coverage
    
    def check_subgraph_coverage(
        self,
        subgraph: nx.Graph,
        query_entities: Set[str],
        query_relations: Set[str]
    ) -> Tuple[bool, float, Dict]:
        """
        检查子图覆盖率，应用严格的ERCR规则
        """
        subgraph_entities = set(subgraph.nodes())
        subgraph_relations = set()
        
        for _, _, data in subgraph.edges(data=True):
            if 'relation' in data:
                subgraph_relations.add(data['relation'])
            elif 'label' in data:
                subgraph_relations.add(data['label'])
        
        entity_complete, entity_coverage = self.check_entity_completeness(
            subgraph_entities, query_entities
        )
        
        relation_complete, relation_coverage = self.check_relation_completeness(
            subgraph_relations, query_relations
        )
        
        coverage_score = self.alpha * entity_coverage + self.beta * relation_coverage
        
        satisfies_ercr = entity_complete and relation_complete and coverage_score >= self.threshold
        
        analysis_result = {
            'satisfies_ercr': satisfies_ercr,
            'coverage_score': coverage_score,
            'entity_complete': entity_complete,
            'relation_complete': relation_complete,
            'entity_coverage': entity_coverage,
            'relation_coverage': relation_coverage,
            'uncovered_entities': query_entities - subgraph_entities,
            'uncovered_relations': query_relations - subgraph_relations,
            'subgraph_size': len(subgraph.nodes()),
            'subgraph_edges': len(subgraph.edges())
        }
        
        return satisfies_ercr, coverage_score, analysis_result
    
    def check_ercr_compliance(
        self,
        subgraph: nx.Graph,
        query: str,
        model=None
    ) -> Tuple[bool, Dict]:
        """
        检查子图是否符合ERCR规则
        """
        if model is None:
            logger.warning("未提供模型，无法进行ERCR检查")
            return False, {}
        
        query_entities, query_relations = self.extract_query_elements(query, model)
        
        if not query_entities and not query_relations:
            logger.warning("无法从查询中提取实体或关系")
            return False, {}
        
        satisfies_ercr, coverage_score, analysis_result = self.check_subgraph_coverage(
            subgraph, query_entities, query_relations
        )
        
        return satisfies_ercr, analysis_result 