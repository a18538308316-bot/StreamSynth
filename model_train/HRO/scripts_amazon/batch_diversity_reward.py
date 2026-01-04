#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch级别多样性奖励函数
实现基于局部密度的distinctiveness评分
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

class BatchDiversityCalculator:
    """Batch级别多样性计算器"""
    
    def __init__(self, embedding_model_path, device='cuda', k_penalty=2.0):
        """
        初始化Batch多样性计算器
        
        Args:
            embedding_model_path: 嵌入模型路径
            device: 计算设备
            k_penalty: 指数衰减的惩罚强度参数
        """
        self.device = device
        self.k_penalty = k_penalty
        
        # 加载嵌入模型
        print(f"🔧 初始化Batch多样性计算器...")
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.embedding_model = self.embedding_model.to(device)
        print(f"✅ 嵌入模型已加载到 {device}")
        
    def compute_embeddings(self, texts):
        """计算文本嵌入"""
        if not texts:
            return np.array([])
        
        # 确保输入是字符串列表
        texts = [str(text) for text in texts]
        
        with torch.no_grad():
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_tensor=True, 
                device=self.device,
                show_progress_bar=False
            )
            return embeddings.cpu().numpy()
    
    def calculate_local_density(self, target_embedding, batch_embeddings):
        """
        计算目标样本在batch中的局部密度
        
        Args:
            target_embedding: 目标样本的嵌入向量 (1, d)
            batch_embeddings: batch中所有样本的嵌入向量 (m, d)
            
        Returns:
            local_density: 局部密度值
        """
        if len(batch_embeddings) == 0:
            return 0.0
        
        # 计算cosine相似度
        similarities = cosine_similarity(target_embedding, batch_embeddings)[0]  # (m,)
        
        # 计算proximity权重
        # w_j = sim(x_new, x_j) / sum(sim(x_new, x_k))
        similarity_sum = np.sum(similarities)
        if similarity_sum == 0:
            # 避免除零错误
            proximity_weights = np.ones(len(similarities)) / len(similarities)
        else:
            proximity_weights = similarities / similarity_sum
        
        # 计算加权平均相似度（局部密度）
        local_density = np.sum(proximity_weights * similarities)
        
        return local_density
    
    def calculate_distinctiveness_score(self, local_density):
        """
        将局部密度转换为distinctiveness评分
        
        Args:
            local_density: 局部密度值
            
        Returns:
            distinctiveness_score: 独特性评分 (0-1)
        """
        # 简化后的多样性分数：(1 - density) * 3，并裁剪到[0,1]
        # 之所以乘以3，是放大原始分数，使其对最终奖励更有影响力
        diversity_score = (1.0 - local_density) * 3.0
        # 裁剪到0-1区间
        diversity_score = float(np.clip(diversity_score, 0.0, 1.0))
        return diversity_score
    
    def calculate_batch_diversity_rewards(self, completions):
        """
        计算batch中每个样本的多样性奖励
        
        Args:
            completions: 生成的完成文本列表
            
        Returns:
            diversity_rewards: 每个样本的多样性奖励列表
        """
        if not completions or len(completions) == 0:
            return []
        
        # 如果只有一个样本，给予最高多样性奖励
        if len(completions) == 1:
            return [1.0]
        
        try:
            # 早期塌陷检测：全部completion极短或全相同 => 直接给予轻度探索奖励
            lengths = [len(c.strip()) for c in completions]
            unique_texts = len({c.strip() for c in completions})
            if all(l <= 2 for l in lengths) or unique_texts == 1:
                print("⚠️ 多样性检测: 全部生成极短/完全相同，返回探索基线奖励 0.3")
                return [0.3] * len(completions)

            # 计算所有样本的嵌入
            batch_embeddings = self.compute_embeddings(completions)
            if batch_embeddings.size == 0:
                return [0.0] * len(completions)

            diversity_rewards = []
            for i, _ in enumerate(completions):
                target_embedding = batch_embeddings[i:i+1]
                other_indices = list(range(len(completions)))
                other_indices.remove(i)
                other_embeddings = batch_embeddings[other_indices]
                local_density = self.calculate_local_density(target_embedding, other_embeddings)
                distinctiveness_score = self.calculate_distinctiveness_score(local_density)
                diversity_rewards.append(distinctiveness_score)

            # 二次塌陷检测：全部得分为0则使用token集合差异度回退
            if max(diversity_rewards) == 0.0:
                print("⚠️ 二次检测: 嵌入局部密度全部导致0分，使用fallback基于token差异度")
                import re
                token_sets = [set(re.findall(r"\w+", c.lower())) for c in completions]
                fallback_scores = []
                for i, ts in enumerate(token_sets):
                    others = [token_sets[j] for j in range(len(token_sets)) if j != i]
                    if not ts:
                        fallback_scores.append(0.0); continue
                    # 计算与其他集合的平均Jaccard距离
                    dists = []
                    for o in others:
                        if not o:
                            dists.append(0.0)
                        else:
                            inter = len(ts & o)
                            union = len(ts | o)
                            dists.append(1 - inter / union if union else 0.0)
                    avg_dist = sum(dists)/len(dists) if dists else 0.0
                    # 将距离映射到 [0.2,1.0]，保留探索激励
                    mapped = 0.2 + 0.8 * avg_dist
                    fallback_scores.append(float(np.clip(mapped, 0.0, 1.0)))
                diversity_rewards = fallback_scores
            return diversity_rewards
        except Exception as e:
            print(f"⚠️ Batch多样性计算失败: {e}")
            return [0.5] * len(completions)
    
    def get_batch_diversity_stats(self, completions):
        """获取batch多样性统计信息"""
        rewards = self.calculate_batch_diversity_rewards(completions)
        
        if not rewards:
            return {
                'mean_diversity': 0.0,
                'std_diversity': 0.0,
                'min_diversity': 0.0,
                'max_diversity': 0.0,
                'batch_size': 0
            }
        
        return {
            'mean_diversity': np.mean(rewards),
            'std_diversity': np.std(rewards),
            'min_diversity': np.min(rewards),
            'max_diversity': np.max(rewards),
            'batch_size': len(rewards)
        }

# 全局变量
batch_diversity_calculator = None

def initialize_batch_diversity_calculator(embedding_model_path, device='cuda', k_penalty=2.0):
    """初始化全局batch多样性计算器"""
    global batch_diversity_calculator
    batch_diversity_calculator = BatchDiversityCalculator(embedding_model_path, device, k_penalty)
    print("✅ 全局Batch多样性计算器初始化完成")

def reward_batch_diversity(completions, **kwargs):
    """
    Batch级别多样性奖励函数
    
    Args:
        completions: 生成的完成文本列表
        
    Returns:
        rewards: 每个样本的多样性奖励列表
    """
    global batch_diversity_calculator
    
    if batch_diversity_calculator is None:
        print("⚠️ Batch多样性计算器未初始化，返回中性奖励")
        return [0.5] * len(completions)
    
    try:  # 注意这里的缩进，应该与if语句同级
        # 计算batch多样性奖励（现在calculate_batch_diversity_rewards返回的是
        # 已按 (1 - density) * 3 并裁剪到 [0,1] 的分数）
        diversity_rewards = batch_diversity_calculator.calculate_batch_diversity_rewards(completions)

        # 直接使用该分数作为多样性奖励返回（不再做离散的桶映射）
        stats = batch_diversity_calculator.get_batch_diversity_stats(completions)

        print(f"🎨 Batch多样性奖励 - 平均: {stats['mean_diversity']:.4f}, "
              f"标准差: {stats['std_diversity']:.4f}, "
              f"范围: [{stats['min_diversity']:.4f}, {stats['max_diversity']:.4f}]")

        return diversity_rewards
        
    except Exception as e:
        print(f"⚠️ Batch多样性奖励计算失败: {e}")
        return [0.0] * len(completions)

if __name__ == "__main__":
    # 测试代码
    test_texts = [
        "This is a great Chinese restaurant with excellent service.",
        "I love the Italian food here, especially the pasta.",
        "The Mexican cuisine was amazing, very authentic flavors.",
        "Another Chinese restaurant review, but with different details."
    ]
    
    # 初始化计算器
    embedding_model_path = "/public/home/huzhenlin2023/synthetic_data/all-MiniLM-L6-v2"
    calculator = BatchDiversityCalculator(embedding_model_path, device='cpu')
    
    # 计算多样性奖励
    rewards = calculator.calculate_batch_diversity_rewards(test_texts)
    stats = calculator.get_batch_diversity_stats(test_texts)
    
    print("测试结果:")
    for i, (text, reward) in enumerate(zip(test_texts, rewards)):
        print(f"  样本{i+1}: {reward:.4f} - {text[:50]}...")
    
    print(f"\n统计信息: {stats}")
