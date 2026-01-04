#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态参考数据管理器
实现混合方案：原始数据集 + 动态高质量样本池
"""
import numpy as np
import torch
import faiss
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

class DynamicReferenceManager:
    """动态参考数据管理器"""
    
    def __init__(self, 
                 embedding_model,
                 device='cuda',
                 max_total_size=300,
                 min_novelty_threshold=0.3,
                 original_data_path=None):
        """
        初始化动态参考数据管理器
        
        Args:
            embedding_model: SentenceTransformer模型
            device: 计算设备
            max_total_size: 参考集最大容量
            min_novelty_threshold: 最小novelty阈值
            original_data_path: 原始数据集路径
        """
        self.embedding_model = embedding_model
        self.device = device
        self.max_total_size = max_total_size
        self.min_novelty_threshold = min_novelty_threshold
        
        # 参考数据存储
        self.reference_texts = []
        self.reference_embeddings = None
        self.reference_index = None
        
        # 数据来源标记
        self.data_sources = []  # 'original' 或 'dynamic'
        self.original_count = 0
        self.dynamic_count = 0
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 加载原始数据
        if original_data_path:
            self.load_original_data(original_data_path)
        else:
            self.initialize_with_templates()
            
    def initialize_with_templates(self):
        """使用模板数据初始化（fallback方案）"""
        template_texts = [
            "The food was absolutely delicious and the service was excellent.",
            "Terrible experience, food was cold and service was slow.",
            "Average restaurant with decent food but nothing special.",
            "Amazing atmosphere and great value for money.",
            "Overpriced food with mediocre quality.",
            "Best Italian restaurant in town, highly recommended.",
            "The ambiance was nice but the food was disappointing.",
            "Quick service and tasty food, perfect for lunch.",
            "Romantic setting with exceptional cuisine.",
            "Friendly staff and reasonable prices.",
            "Authentic cuisine with fresh ingredients.",
            "Poor quality food and unfriendly service.",
            "Cozy atmosphere with excellent desserts.",
            "Innovative menu with creative presentations.",
            "Traditional dishes prepared perfectly.",
            "Busy restaurant with good portion sizes.",
            "Elegant dining experience worth the price.",
            "Family-friendly place with kid's menu.",
            "Spicy food with authentic flavors.",
            "Clean restaurant with professional service.",
            "Unique dishes not found elsewhere.",
            "Comfortable seating and pleasant music.",
            "Fresh seafood and great wine selection.",
            "Vegetarian options and healthy choices.",
            "Late night dining with good atmosphere."
        ]
        
        self.reference_texts = template_texts
        self.data_sources = ['template'] * len(template_texts)
        self.original_count = len(template_texts)
        
        print(f"✅ 使用{len(template_texts)}条模板数据初始化参考集")
        self._update_embeddings_and_index()
        
    def load_original_data(self, data_path: str, max_samples: int = 100):
        """从原始数据集加载参考数据"""
        try:
            data_path = Path(data_path)
            if not data_path.exists():
                print(f"⚠️ 原始数据文件不存在: {data_path}")
                self.initialize_with_templates()
                return
                
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取实际评论文本作为参考数据
            original_texts = []
            for item in data[:max_samples]:
                # 处理原始Yelp数据格式: input字段包含"Text: ..."
                if 'input' in item and item['input'].strip():
                    input_text = item['input'].strip()
                    # 提取"Text: "后面的实际评论内容
                    if input_text.startswith('Text: '):
                        review_text = input_text[6:].strip()  # 去掉"Text: "前缀
                        if review_text and len(review_text) > 20:  # 确保评论有一定长度
                            original_texts.append(review_text)
                    elif len(input_text) > 20:  # 如果没有"Text: "前缀但内容足够长
                        original_texts.append(input_text)
                # 备选方案：检查其他可能的字段
                elif 'review_text' in item and item['review_text'].strip():
                    original_texts.append(item['review_text'].strip())
                elif 'text' in item and item['text'].strip():
                    original_texts.append(item['text'].strip())
                    
            if not original_texts:
                print("⚠️ 未找到有效的原始评论数据，使用模板初始化")
                self.initialize_with_templates()
                return
                
            self.reference_texts = original_texts
            self.data_sources = ['original'] * len(original_texts)
            self.original_count = len(original_texts)
            
            print(f"✅ 从原始数据集加载{len(original_texts)}条参考评论数据")
            print(f"   示例评论: {original_texts[0][:100]}...")
            self._update_embeddings_and_index()
            
        except Exception as e:
            print(f"⚠️ 加载原始数据失败: {e}")
            import traceback
            traceback.print_exc()
            self.initialize_with_templates()
    
    def _update_embeddings_and_index(self):
        """更新embeddings和FAISS索引"""
        if not self.reference_texts:
            return
            
        try:
            # 计算embeddings
            self.reference_embeddings = self.embedding_model.encode(
                self.reference_texts,
                convert_to_tensor=True,
                device=self.device
            )
            
            # 更新FAISS索引
            self._rebuild_faiss_index()
            
        except Exception as e:
            print(f"⚠️ 更新embeddings和索引失败: {e}")
    
    def _rebuild_faiss_index(self):
        """重建FAISS索引"""
        if self.reference_embeddings is None:
            return
            
        try:
            embedding_dim = self.reference_embeddings.shape[1]
            self.reference_index = faiss.IndexFlatIP(embedding_dim)
            
            # 归一化embeddings
            ref_embeddings_normalized = self.reference_embeddings.cpu().numpy()
            ref_embeddings_normalized = ref_embeddings_normalized / np.linalg.norm(
                ref_embeddings_normalized, axis=1, keepdims=True
            )
            
            self.reference_index.add(ref_embeddings_normalized.astype('float32'))
            print(f"✅ FAISS索引重建完成，参考数据量: {self.reference_index.ntotal}")
            
        except Exception as e:
            print(f"⚠️ FAISS索引重建失败: {e}")
            self.reference_index = None
    
    def compute_novelty_score(self, text: str) -> float:
        """计算单个文本的novelty分数"""
        # 正确检查tensor和None值
        if (self.reference_index is None or 
            self.reference_embeddings is None or 
            (hasattr(self.reference_embeddings, 'numel') and self.reference_embeddings.numel() == 0)):
            return 1.0  # 如果没有参考数据，认为是新颖的
            
        try:
            # 计算新文本的embedding
            new_embedding = self.embedding_model.encode(
                [text], 
                convert_to_tensor=True,
                device=self.device
            )
            
            # 归一化
            new_embedding_normalized = new_embedding.cpu().numpy()
            new_embedding_normalized = new_embedding_normalized / np.linalg.norm(
                new_embedding_normalized, axis=1, keepdims=True
            )
            
            # 搜索最相似的k个参考样本
            k = min(10, self.reference_index.ntotal)
            similarities, indices = self.reference_index.search(
                new_embedding_normalized.astype('float32'), k
            )
            
            # novelty = 1 - max_similarity
            max_similarity = similarities[0].max() if len(similarities[0]) > 0 else 0.0
            novelty_score = 1.0 - max_similarity
            
            return float(novelty_score)
            
        except Exception as e:
            print(f"⚠️ novelty计算失败: {e}")
            return 0.5  # 默认中等novelty
    
    def add_dynamic_sample(self, text: str, force_add: bool = False) -> bool:
        """添加动态样本到参考池"""
        if not text.strip():
            return False
            
        # 计算novelty分数
        novelty_score = self.compute_novelty_score(text)
        
        # 检查是否满足添加条件
        if not force_add and novelty_score < self.min_novelty_threshold:
            return False
            
        # 如果参考池未满，直接添加
        if len(self.reference_texts) < self.max_total_size:
            self.reference_texts.append(text)
            self.data_sources.append('dynamic')
            self.dynamic_count += 1
            self._update_embeddings_and_index()
            return True
            
        # 如果参考池已满，替换最不新颖的动态样本
        return self._replace_least_novel_dynamic_sample(text, novelty_score)
    
    def _replace_least_novel_dynamic_sample(self, new_text: str, new_novelty: float) -> bool:
        """替换最不新颖的动态样本"""
        # 找到所有动态样本的索引
        dynamic_indices = [
            i for i, source in enumerate(self.data_sources) 
            if source == 'dynamic'
        ]
        
        if not dynamic_indices:
            return False  # 没有动态样本可替换
            
        # 计算所有动态样本的novelty（相对于原始数据）
        min_novelty = float('inf')
        replace_idx = -1
        
        for idx in dynamic_indices:
            current_text = self.reference_texts[idx]
            current_novelty = self.compute_novelty_score(current_text)
            
            if current_novelty < min_novelty:
                min_novelty = current_novelty
                replace_idx = idx
        
        # 如果新样本比最不新颖的动态样本更新颖，则替换
        if new_novelty > min_novelty:
            self.reference_texts[replace_idx] = new_text
            self._update_embeddings_and_index()
            return True
            
        return False
    
    def get_reference_data(self) -> Tuple[List[str], torch.Tensor, faiss.Index]:
        """获取当前参考数据"""
        return self.reference_texts, self.reference_embeddings, self.reference_index
    
    def get_statistics(self) -> Dict:
        """获取参考池统计信息"""
        return {
            'total_count': len(self.reference_texts),
            'original_count': self.original_count,
            'dynamic_count': self.dynamic_count,
            'max_capacity': self.max_total_size,
            'usage_ratio': len(self.reference_texts) / self.max_total_size,
            'source_distribution': {
                source: self.data_sources.count(source) 
                for source in set(self.data_sources)
            }
        }
    
    def batch_add_samples(self, texts: List[str], quality_scores: Optional[List[float]] = None) -> int:
        """批量添加样本，返回成功添加的数量"""
        added_count = 0
        
        # 如果提供了质量分数，按分数排序
        if quality_scores:
            paired_data = list(zip(texts, quality_scores))
            paired_data.sort(key=lambda x: x[1], reverse=True)  # 按质量分数降序
            texts = [text for text, _ in paired_data]
        
        for text in texts:
            if self.add_dynamic_sample(text):
                added_count += 1
                
        return added_count