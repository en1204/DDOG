import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from gnn_retriever import GraphRetriever
from utils.graph_utils import compute_semantic_relevance, compute_final_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRetrievalDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        doc_path: str,
        tokenizer,
        max_length: int = 512
    ):
        """初始化数据集"""
        self.tokenizer = tokenizer
        self.max_length = max_length
        

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        with open(doc_path, 'r', encoding='utf-8') as f:
            self.external_docs = json.load(f)
            
        self.entities = set()
        self.relations = set()
        for item in self.data:
            self.entities.update(item['entities'])
            self.relations.update(item['relations'])
            
        self.entity2id = {e: i for i, e in enumerate(self.entities)}
        self.relation2id = {r: i for i, r in enumerate(self.relations)}
        
        self._process_path_information()

    def _process_path_information(self):
        """处理路径信息"""
        for item in self.data:

            item['query_entity'] = item.get('query_entity', '')
            item['target_entity'] = item.get('target_entity', '')
            

            item['ground_truth_path'] = item.get('ground_truth_path', [])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        

        question_encoding = self.tokenizer(
            item['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        doc_ids = item.get('relevant_doc_ids', [])
        relevant_docs = [self.external_docs[i] for i in doc_ids]
        

        node_features = torch.zeros(len(self.entities), 768)  
        edge_index = torch.tensor(item['edge_index'])
        
        return {
            'question_encoding': question_encoding,
            'entity_ids': torch.tensor([self.entity2id[e] for e in item['entities']]),
            'relation_ids': torch.tensor([self.relation2id[r] for r in item['relations']]),
            'edge_index': edge_index,
            'node_features': node_features,
            'answer': item['answer'],
            'external_docs': relevant_docs,
            'ground_truth_triples': item.get('ground_truth_triples', []),
            'query_entity': item['query_entity'],
            'target_entity': item['target_entity'],
            'ground_truth_path': item['ground_truth_path']
        }

def compute_path_loss(
    predicted_path: List[Tuple[str, str, str]],
    ground_truth_path: List[Tuple[str, str, str]]
) -> torch.Tensor:
    """
    计算路径损失
    """
    if not predicted_path or not ground_truth_path:
        return torch.tensor(1.0)
        
    pred_set = set(predicted_path)
    gt_set = set(ground_truth_path)
    
    intersection = len(pred_set.intersection(gt_set))
    union = len(pred_set.union(gt_set))
    
    path_loss = 1.0 - (intersection / union if union > 0 else 0.0)
    return torch.tensor(path_loss)

def compute_semantic_loss(
    model: GraphRetriever,
    predicted_path: List[Tuple[str, str, str]],
    query: str
) -> torch.Tensor:
    """
    计算语义损失
    """
    if not predicted_path:
        return torch.tensor(1.0)
        
    semantic_scores = []
    for triple in predicted_path:
        score = compute_semantic_relevance(
            triple,
            query,
            [],  
            model.llm_model,
            model.llm_tokenizer,
            model.device
        )
        semantic_scores.append(score)
    
    avg_score = sum(semantic_scores) / len(semantic_scores)
    return torch.tensor(1.0 - avg_score)

def train_epoch(
    model: GraphRetriever,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_path_acc = 0
    total_exact_match = 0
    total_f1 = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        answer, filtered_triples, final_path = model.retrieve_and_answer(
            query=batch['question_encoding']['input_ids'][0],
            query_entity=batch['query_entity'],
            target_entity=batch['target_entity'],
            candidate_triples=batch['ground_truth_triples'],
            initial_path=[]
        )
        
        path_loss = compute_path_loss(final_path, batch['ground_truth_path'])
        
        semantic_loss = compute_semantic_loss(
            model,
            final_path,
            batch['question_encoding']['input_ids'][0]
        )
        
        answer_loss = compute_answer_metrics(answer, batch['answer'])
        
        loss = 0.4 * path_loss + 0.3 * semantic_loss + 0.3 * answer_loss['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_path_acc += compute_path_accuracy(final_path, batch['ground_truth_path'])
        total_exact_match += answer_loss['exact_match']
        total_f1 += answer_loss['f1']
    

    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'path_acc': total_path_acc / num_batches,
        'exact_match': total_exact_match / num_batches,
        'f1': total_f1 / num_batches
    }

def evaluate(
    model: GraphRetriever,
    dataloader: DataLoader,
    device: str
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    metrics = {
        'exact_match': 0,
        'f1': 0,
        'precision': 0,
        'recall': 0,
        'avg_num_tokens': 0
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            

            answer, filtered_triples, final_path = model.retrieve_and_answer(
                query=batch['question_encoding']['input_ids'][0],
                query_entity=batch['query_entity'],
                target_entity=batch['target_entity'],
                candidate_triples=batch['ground_truth_triples'],
                initial_path=[]
            )
            
            answer_metrics = compute_answer_metrics(answer, batch['answer'])
            path_metrics = compute_path_metrics(final_path, batch['ground_truth_path'])
            
            for key in metrics:
                if key in answer_metrics:
                    metrics[key] += answer_metrics[key]
                if key in path_metrics:
                    metrics[key] += path_metrics[key]
    

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in metrics.items()}

def compute_answer_metrics(
    predicted_answer: str,
    ground_truth_answer: str
) -> Dict[str, float]:
    """计算答案相关的指标"""

    pred_tokens = set(predicted_answer.split())
    gt_tokens = set(ground_truth_answer.split())
    

    intersection = len(pred_tokens.intersection(gt_tokens))
    precision = intersection / len(pred_tokens) if pred_tokens else 0
    recall = intersection / len(gt_tokens) if gt_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'exact_match': float(predicted_answer == ground_truth_answer),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'avg_num_tokens': len(pred_tokens),
        'loss': 1.0 - f1  
    }

def compute_path_metrics(
    predicted_path: List[Tuple[str, str, str]],
    ground_truth_path: List[Tuple[str, str, str]]
) -> Dict[str, float]:
    """计算路径相关的指标"""
    pred_set = set(predicted_path)
    gt_set = set(ground_truth_path)
    
    intersection = len(pred_set.intersection(gt_set))
    precision = intersection / len(pred_set) if pred_set else 0
    recall = intersection / len(gt_set) if gt_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'path_precision': precision,
        'path_recall': recall,
        'path_f1': f1
    }

def compute_path_accuracy(
    predicted_path: List[Tuple[str, str, str]],
    ground_truth_path: List[Tuple[str, str, str]]
) -> float:
    """计算路径准确率"""
    if not predicted_path or not ground_truth_path:
        return 0.0
        
    pred_set = set(predicted_path)
    gt_set = set(ground_truth_path)
    
    intersection = len(pred_set.intersection(gt_set))
    return intersection / len(gt_set) if gt_set else 0.0

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--doc_path', type=str, required=True, help='外部文档路径')
    parser.add_argument('--model_name', type=str, default='bert-base-chinese', help='预训练模型名称')
    parser.add_argument('--llm_model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='LLM模型名称')
    parser.add_argument('--hidden_dim', type=int, default=256, help='GNN隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='GNN层数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--alpha', type=float, default=0.6, help='连通性权重')
    parser.add_argument('--beta', type=float, default=0.4, help='语义相关性权重')
    parser.add_argument('--reach_threshold', type=float, default=0.7, help='可达性阈值')
    return parser

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model = GraphRetriever(
        model_name=args.model_name,
        llm_model_name=args.llm_model_name,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
        alpha=args.alpha,
        beta=args.beta,
        reach_threshold=args.reach_threshold
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = GraphRetrievalDataset(
        args.data_path,
        args.doc_path,
        tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_f1 = 0
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        train_metrics = train_epoch(model, train_dataloader, optimizer, device)
        logger.info(f"Training metrics: {train_metrics}")

        eval_metrics = evaluate(model, train_dataloader, device)
        logger.info(f"Evaluation metrics: {eval_metrics}")
        
        if eval_metrics['f1'] > best_f1:
            best_f1 = eval_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pt')
            logger.info("Saved best model")

if __name__ == "__main__":
    main() 