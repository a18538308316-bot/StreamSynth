#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NovelSum多样性计算模块 - 支持动态参考数据管理
"""
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Optional
from .dynamic_reference_manager import DynamicReferenceManager

# NovelSum配置
NOVELSUM_CONFIG = {
    'density_power': 0.5,
    'distance_power': 1.0,
    'neighbors': 10,
    'embedding_model_path': '/public/home/huzhenlin2023/synthetic_data/all-MiniLM-L6-v2',
    'max_length': 512,
    # 动态参考数据配置
    'max_reference_size': 300,
    'min_novelty_threshold': 0.1,  # 降低阈值以促进动态样本添加
    'original_data_path': '/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/original_yelp_train_100_dataset.json',
}

class NovelSumDiversityCalculator:
    """NovelSum多样性计算器 - 支持动态参考数据管理"""
    
    def __init__(self, embedding_model_path, device='cuda', max_length=256, 
                 use_dynamic_reference=True, original_data_path=None):
        self.device = device
        self.max_length = max_length
        self.use_dynamic_reference = use_dynamic_reference
        
        # 初始化embedding模型
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.embedding_model.to(device)
        
        # 初始化参考数据管理器
        if use_dynamic_reference:
            self.reference_manager = DynamicReferenceManager(
                embedding_model=self.embedding_model,
                device=device,
                max_total_size=NOVELSUM_CONFIG.get('max_reference_size', 300),
                min_novelty_threshold=NOVELSUM_CONFIG.get('min_novelty_threshold', 0.3),
                original_data_path=original_data_path or NOVELSUM_CONFIG.get('original_data_path')
            )
            print("✅ 动态参考数据管理器初始化完成")
        else:
            # 向后兼容：使用静态参考数据
            self.reference_embeddings = None
            self.reference_index = None
            self.setup_static_reference_data()
        
    def get_reference_data(self):
        """获取参考数据"""
        if self.use_dynamic_reference:
            return self.reference_manager.get_reference_data()
        else:
            return None, self.reference_embeddings, self.reference_index
    
    def add_training_samples(self, texts: List[str], quality_scores: Optional[List[float]] = None):
        """添加训练样本到动态参考池"""
        if not self.use_dynamic_reference:
            return 0
        
        return self.reference_manager.batch_add_samples(texts, quality_scores)
    
    def get_reference_statistics(self):
        """获取参考数据统计信息"""
        if self.use_dynamic_reference:
            return self.reference_manager.get_statistics()
        else:
            total_count = self.reference_embeddings.shape[0] if self.reference_embeddings is not None else 0
            return {
                'total_count': total_count,
                'mode': 'static',
                'source': 'template'
            }
        
    def setup_static_reference_data(self):
        """设置静态参考数据（向后兼容）"""
        # 餐厅评论参考语料（可以替换为实际的参考数据集）
        reference_texts = [
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
        
        print("🔄 设置NovelSum静态参考数据...")
        try:
            self.reference_embeddings = self.embedding_model.encode(
                reference_texts, 
                convert_to_tensor=True,
                device=self.device
            )
            self.setup_faiss_index_static()
            print("✅ NovelSum静态参考数据设置完成")
        except Exception as e:
            print(f"⚠️ NovelSum静态参考数据设置失败: {e}")
            self.reference_embeddings = None
    
    def setup_faiss_index_static(self):
        """设置FAISS索引用于快速相似度搜索（静态版本）"""
        if self.reference_embeddings is None:
            return
        
        try:
            embedding_dim = self.reference_embeddings.shape[1]
            self.reference_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (cosine similarity)
            
            # 归一化embeddings以便使用内积计算余弦相似度
            ref_embeddings_normalized = self.reference_embeddings.cpu().numpy()
            ref_embeddings_normalized = ref_embeddings_normalized / np.linalg.norm(
                ref_embeddings_normalized, axis=1, keepdims=True
            )
            
            self.reference_index.add(ref_embeddings_normalized.astype('float32'))
            print(f"✅ FAISS索引创建完成，参考数据量: {self.reference_index.ntotal}")
        except Exception as e:
            print(f"⚠️ FAISS索引创建失败: {e}")
            self.reference_index = None
    
    def compute_cosine_distance_matrix(self, embeddings):
        """计算余弦距离矩阵"""
        try:
            # 确保转换为numpy数组
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.detach().cpu().numpy()
            else:
                embeddings_np = np.array(embeddings)
            
            # 确保是float64类型以提高数值稳定性
            embeddings_np = embeddings_np.astype(np.float64)
            
            # 计算余弦相似度矩阵
            cosine_sim_matrix = cosine_similarity(embeddings_np)
            # 转换为距离矩阵
            distance_matrix = 1 - cosine_sim_matrix
            
            # 确保返回的是numpy数组
            return np.array(distance_matrix, dtype=np.float64)
            
        except Exception as e:
            print(f"⚠️ 余弦距离矩阵计算失败: {e}")
            n_samples = len(embeddings) if hasattr(embeddings, '__len__') else 1
            return np.ones((n_samples, n_samples), dtype=np.float64)
    
    def compute_local_density(self, embeddings, n_neighbors=10, power=0.5):
        """计算局部密度"""
        try:
            distance_matrix = self.compute_cosine_distance_matrix(embeddings)
            n_samples = distance_matrix.shape[0]
            n_neighbors = min(n_neighbors, n_samples - 1)
            
            if n_neighbors <= 0:
                return np.ones(n_samples, dtype=np.float64)
            
            densities = []
            for i in range(n_samples):
                # 获取到其他所有点的距离（排除自己）
                distances = distance_matrix[i].copy()
                distances = np.delete(distances, i)
                
                if len(distances) == 0:
                    densities.append(1.0)
                    continue
                
                # 找到k个最近邻
                if len(distances) >= n_neighbors:
                    nearest_distances = np.partition(distances, n_neighbors-1)[:n_neighbors]
                else:
                    nearest_distances = distances
                
                # 计算局部密度（距离越小，密度越高）
                avg_distance = np.mean(nearest_distances)
                density = 1.0 / (1.0 + avg_distance)  # 转换为密度值
                density = density ** power  # 应用幂次调整
                densities.append(float(density))
            
            return np.array(densities, dtype=np.float64)
            
        except Exception as e:
            print(f"⚠️ 局部密度计算失败: {e}")
            n_samples = len(embeddings) if hasattr(embeddings, '__len__') else 1
            return np.ones(n_samples, dtype=np.float64)
    
    def weighted_average(self, row, power=1.0):
        """计算平均距离 - 修复版本"""
        try:
            # 确保输入是numpy数组
            row = np.array(row, dtype=np.float64)
            
            # 排除自己（对角线元素应该是0）
            non_zero_distances = row[row > 1e-10]
            
            if len(non_zero_distances) > 0:
                result = np.mean(non_zero_distances)
            else:
                result = np.mean(row)
                
            return float(result)
            
        except Exception as e:
            print(f"⚠️ 平均距离计算失败: {e}")
            return float(np.mean(np.array(row))) if hasattr(row, '__len__') else 0.5
    
    def calculate_novelsum_score(self, texts, density_power=0.5, distance_power=1.0, neighbors=10):
        """计算NovelSum多样性分数"""
        try:
            # 基本验证
            if not texts or len(texts) < 2:
                print(f"⚠️ 文本数量不足: {len(texts) if texts else 0}")
                return 0.5  # 单个样本返回中等分数
            
            # 过滤空文本并确保所有元素都是字符串
            valid_texts = []
            for text in texts:
                if text is not None:
                    # 确保转换为字符串
                    str_text = str(text).strip() if not isinstance(text, str) else text.strip()
                    if len(str_text) > 5:
                        valid_texts.append(str_text)
            
            if len(valid_texts) < 2:
                print(f"⚠️ 有效文本数量不足: {len(valid_texts)}")
                return 0.5
            
            print(f"🔍 DEBUG: 计算{len(valid_texts)}个文本的NovelSum分数")
            
            # 生成文本嵌入
            try:
                embeddings = self.embedding_model.encode(
                    valid_texts, 
                    convert_to_tensor=True,
                    device=self.device
                )
                print(f"🔍 DEBUG: Embeddings shape: {embeddings.shape}")
            except Exception as e:
                print(f"⚠️ 嵌入生成失败: {e}")
                print(f"🔍 DEBUG: valid_texts类型检查: {[type(t) for t in valid_texts[:3]]}")
                return 0.5
            
            # 确保embeddings是正确的tensor类型
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, device=self.device, dtype=torch.float32)
            
            # 检查embeddings的形状
            if embeddings.dim() != 2 or embeddings.shape[0] != len(valid_texts):
                print(f"⚠️ Embeddings形状异常: {embeddings.shape}, 期望: ({len(valid_texts)}, embedding_dim)")
                return 0.5
            
            # 计算局部密度
            try:
                densities = self.compute_local_density(embeddings, neighbors, density_power)
                print(f"🔍 DEBUG: Densities shape: {densities.shape}, 样例值: {densities[:3]}")
            except Exception as e:
                print(f"⚠️ 局部密度计算异常: {e}")
                return 0.5
            
            # 计算距离矩阵
            try:
                distance_matrix = self.compute_cosine_distance_matrix(embeddings)
                print(f"🔍 DEBUG: Distance matrix shape: {distance_matrix.shape}")
            except Exception as e:
                print(f"⚠️ 距离矩阵计算异常: {e}")
                return 0.5
            
            # 计算NovelSum分数 - 修复版本
            scores = []
            for i in range(len(valid_texts)):
                try:
                    row = distance_matrix[i]
                    density_weight = float(densities[i])
                    
                    # 计算平均距离（多样性度量）
                    avg_distance = self.weighted_average(row, distance_power)
                    
                    # 结合密度进行调整
                    # 距离越大表示越新颖，密度越高表示该区域越拥挤
                    # 使用密度的倒数作为调整因子，避免数值过大
                    density_factor = 1.0 / (density_weight + 1e-8)
                    novelsum_score = avg_distance * (1.0 + density_factor)
                    
                    scores.append(float(novelsum_score))
                    
                except Exception as e:
                    print(f"⚠️ 第{i}个样本分数计算失败: {e}")
                    scores.append(0.5)
            
            # 返回平均分数
            if not scores:
                print("⚠️ 没有有效分数")
                return 0.5
                
            final_score = float(np.mean(scores))
            print(f"🔍 DEBUG: 最终NovelSum分数: {final_score}")
            
            # 检查分数的合理性
            if final_score < 0 or final_score > 10 or np.isnan(final_score) or np.isinf(final_score):
                print(f"⚠️ 分数异常: {final_score}, 使用默认值")
                return 0.5
                
            return final_score
            
        except Exception as e:
            print(f"⚠️ NovelSum分数计算失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return 0.5
    
    def calculate_internal_diversity(self, texts):
        """计算内部多样性（批次内样本间的平均距离）"""
        try:
            if len(texts) < 2:
                return 0.5
            
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_tensor=True,
                device=self.device
            )
            
            distance_matrix = self.compute_cosine_distance_matrix(embeddings)
            
            # 计算上三角矩阵的平均值（排除对角线）
            n = distance_matrix.shape[0]
            total_pairs = n * (n - 1) // 2
            
            if total_pairs == 0:
                return 0.5
            
            diversity_sum = 0
            for i in range(n):
                for j in range(i + 1, n):
                    diversity_sum += distance_matrix[i, j]
            
            average_diversity = diversity_sum / total_pairs
            return float(average_diversity)
            
        except Exception as e:
            print(f"⚠️ 内部多样性计算失败: {e}")
            return 0.5

def calculate_restaurant_specific_diversity(texts):
    """计算餐厅评论特色多样性"""
    try:
        if len(texts) < 2:
            return 0.5
        
        # 餐厅评论特定的多样性指标
        aspect_keywords = {
            'food_quality': ['delicious', 'tasty', 'bland', 'awful', 'amazing', 'terrible', 'fresh', 'stale'],
            'service': ['friendly', 'rude', 'quick', 'slow', 'attentive', 'negligent', 'professional'],
            'atmosphere': ['cozy', 'loud', 'romantic', 'casual', 'elegant', 'crowded', 'peaceful'],
            'price': ['expensive', 'cheap', 'reasonable', 'overpriced', 'value', 'affordable', 'costly'],
            'location': ['convenient', 'remote', 'accessible', 'parking', 'downtown', 'suburban']
        }
        
        # 分析每个文本覆盖的方面
        text_aspects = []
        for text in texts:
            text_lower = text.lower()
            aspects = set()
            
            for aspect, keywords in aspect_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    aspects.add(aspect)
            
            text_aspects.append(aspects)
        
        # 计算方面覆盖多样性
        all_aspects = set()
        for aspects in text_aspects:
            all_aspects.update(aspects)
        
        if not all_aspects:
            return 0.3  # 没有识别到特定方面
        
        # 计算Jaccard多样性
        diversities = []
        for i in range(len(text_aspects)):
            for j in range(i + 1, len(text_aspects)):
                aspects1, aspects2 = text_aspects[i], text_aspects[j]
                if len(aspects1) == 0 and len(aspects2) == 0:
                    jaccard = 1.0  # 都为空，完全相似
                else:
                    intersection = len(aspects1.intersection(aspects2))
                    union = len(aspects1.union(aspects2))
                    jaccard = intersection / union if union > 0 else 0.0
                
                diversity = 1.0 - jaccard  # 转换为多样性
                diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.5
        
    except Exception as e:
        print(f"⚠️ 餐厅特色多样性计算失败: {e}")
        return 0.5

def extract_text_content_global(completion):
    """全局函数：从completion中提取实际的文本内容（优化版本）"""
    try:
        # 首先分离出生成内容
        generation = separate_prompt_and_generation_global(completion)
        
        # 从生成内容中提取文本
        if "Text:" in generation:
            text_part = generation.split("Text:")[1]
            if "Label:" in text_part:
                text_part = text_part.split("Label:")[0]
            return text_part.strip()
        
        # 尝试从JSON中提取
        if "{" in generation and "}" in generation:
            try:
                import json
                start_idx = generation.find("{")
                end_idx = generation.rfind("}") + 1
                json_str = generation[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                input_text = parsed_data.get("input", "")
                if input_text.startswith("Text: "):
                    return input_text[6:]
                elif input_text:
                    return input_text
            except json.JSONDecodeError:
                pass
        
        return generation.strip()[:200]  # 如果没有格式，返回前200字符
    except:
        return completion.strip()[:200]

def separate_prompt_and_generation_global(completion):
    """全局函数：从GRPO的completion中分离出真正的模型生成内容"""
    try:
        # 方法1: 查找第一个完整的JSON对象
        json_start = completion.find('{')
        if json_start != -1:
            # 从第一个{开始，找到匹配的}
            brace_count = 0
            json_end = -1
            
            for i in range(json_start, len(completion)):
                if completion[i] == '{':
                    brace_count += 1
                elif completion[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            
            if json_end != -1:
                # 找到第一个完整的JSON
                first_json = completion[json_start:json_end + 1]
                
                # 验证这是否是一个有效的JSON
                try:
                    import json
                    json.loads(first_json)
                    return first_json  # 返回第一个有效JSON
                except json.JSONDecodeError:
                    pass  # 如果不是有效JSON，继续其他方法
        
        # 方法2: 查找assistant标记
        assistant_markers = [
            'assistant:',
            'Assistant:',
            '助手:',
        ]
        
        for marker in assistant_markers:
            pos = completion.find(marker)
            if pos != -1:
                remaining = completion[pos + len(marker):].strip()
                if len(remaining) > 30:  # 确保有足够内容
                    return remaining
        
        # 方法3: 查找生成内容的开始标记
        generation_markers = [
            'Here is',
            'here is',
            'Based on',
            'based on',
            '根据',
            '按照',
            '以下是',
        ]
        
        for marker in generation_markers:
            pos = completion.find(marker)
            if pos != -1 and pos > len(completion) * 0.2:  # 在后80%位置
                remaining = completion[pos:].strip()
                if len(remaining) > 50:  # 确保有足够内容
                    return remaining
        
        # 方法4: 如果completion较长，可能前面是prompt重复，取后半部分
        if len(completion) > 1000:  # 对于长文本
            split_point = int(len(completion) * 0.6)
            return completion[split_point:].strip()
        
        # 最后的fallback
        return completion.strip()
        
    except Exception as e:
        print(f"⚠️ 分离prompt和generation时出错: {e}")
        return completion.strip()