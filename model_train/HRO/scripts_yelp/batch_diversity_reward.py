#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-level diversity reward function.
Implements a distinctiveness score based on local density in embedding space.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

class BatchDiversityCalculator:
    """Batch-level diversity calculator."""
    
    def __init__(self, embedding_model_path, device='cuda', k_penalty=2.0):
        """Initialize a batch diversity calculator.

        Args:
            embedding_model_path: Path to the sentence embedding model.
            device: Device used for computation, e.g. "cuda" or "cpu".
            k_penalty: Penalty strength hyperparameter for exponential decay.
        """
        self.device = device
        self.k_penalty = k_penalty
        
        # Load embedding model
        print(f"🔧 Initializing batch diversity calculator...")
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.embedding_model = self.embedding_model.to(device)
        print(f"✅ Embedding model loaded on {device}")
        
    def compute_embeddings(self, texts):
        """Compute sentence embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        # Ensure the input is a list of strings
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
        """Compute the local density of a target sample within a batch.

        Args:
            target_embedding: Embedding of the target sample with shape (1, d).
            batch_embeddings: Embeddings of all batch samples with shape (m, d).

        Returns:
            local_density: Local density value.
        """
        if len(batch_embeddings) == 0:
            return 0.0
        
        # Compute cosine similarities
        similarities = cosine_similarity(target_embedding, batch_embeddings)[0]  # (m,)
        
        # Compute proximity weights
        # w_j = sim(x_new, x_j) / sum(sim(x_new, x_k))
        similarity_sum = np.sum(similarities)
        if similarity_sum == 0:
            # Avoid division-by-zero issues
            proximity_weights = np.ones(len(similarities)) / len(similarities)
        else:
            proximity_weights = similarities / similarity_sum
        
        # Compute weighted average similarity (local density)
        local_density = np.sum(proximity_weights * similarities)
        
        return local_density
    
    def calculate_distinctiveness_score(self, local_density):
        """Convert local density into a distinctiveness score.

        Args:
            local_density: Local density value.

        Returns:
            distinctiveness_score: Distinctiveness score in [0, 1].
        """
        # Simplified diversity score: (1 - density) * 3, then clipped to [0, 1].
        # Multiplying by 3 amplifies the signal so it has more impact on the reward.
        diversity_score = (1.0 - local_density) * 3.0
        # Clip to [0, 1]
        diversity_score = float(np.clip(diversity_score, 0.0, 1.0))
        return diversity_score
    
    def calculate_batch_diversity_rewards(self, completions):
        """Compute diversity rewards for each sample in a batch.

        Args:
            completions: List of generated completion texts.

        Returns:
            diversity_rewards: List of diversity rewards per sample.
        """
        if not completions or len(completions) == 0:
            return []
        
        # If there is only one sample, give it the maximum diversity reward
        if len(completions) == 1:
            return [1.0]
        
        try:
            # Compute embeddings for all samples
            batch_embeddings = self.compute_embeddings(completions)
            
            if batch_embeddings.size == 0:
                return [0.0] * len(completions)
            
            diversity_rewards = []
            
            for i, completion in enumerate(completions):
                # Embedding of the current sample
                target_embedding = batch_embeddings[i:i+1]  # (1, d)
                
                # Embeddings of other samples in the batch (excluding itself)
                other_indices = list(range(len(completions)))
                other_indices.remove(i)
                other_embeddings = batch_embeddings[other_indices]  # (m-1, d)
                
                # Compute local density
                local_density = self.calculate_local_density(target_embedding, other_embeddings)
                
                # Compute distinctiveness score
                distinctiveness_score = self.calculate_distinctiveness_score(local_density)
                
                diversity_rewards.append(distinctiveness_score)
            
            return diversity_rewards
            
        except Exception as e:
            print(f"⚠️ Failed to compute batch diversity: {e}")
            # Return neutral rewards on failure
            return [0.5] * len(completions)
    
    def get_batch_diversity_stats(self, completions):
        """Return summary statistics for batch diversity rewards."""
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

# Global calculator instance
batch_diversity_calculator = None

def initialize_batch_diversity_calculator(embedding_model_path, device='cuda', k_penalty=2.0):
    """Initialize the global batch diversity calculator instance."""
    global batch_diversity_calculator
    batch_diversity_calculator = BatchDiversityCalculator(embedding_model_path, device, k_penalty)
    print("✅ Global batch diversity calculator initialized")

def reward_batch_diversity(completions, **kwargs):
    """Batch-level diversity reward function.

    Args:
        completions: List of generated completion texts.

    Returns:
        rewards: List of diversity rewards per sample.
    """
    global batch_diversity_calculator
    
    if batch_diversity_calculator is None:
        print("⚠️ Batch diversity calculator is not initialized, returning neutral rewards")
        return [0.5] * len(completions)
    
    try:  # Note: indentation here should align with the preceding if
        # Calculate batch diversity rewards (already scaled as
        # (1 - density) * 3 and clipped to [0, 1])
        diversity_rewards = batch_diversity_calculator.calculate_batch_diversity_rewards(completions)

        # Directly use these scores as diversity rewards (no bucket mapping)
        stats = batch_diversity_calculator.get_batch_diversity_stats(completions)

          print(f"🎨 Batch diversity rewards - mean: {stats['mean_diversity']:.4f}, "
              f"std: {stats['std_diversity']:.4f}, "
              f"range: [{stats['min_diversity']:.4f}, {stats['max_diversity']:.4f}]")

        return diversity_rewards
        
    except Exception as e:
        print(f"⚠️ Failed to compute batch diversity rewards: {e}")
        return [0.0] * len(completions)

if __name__ == "__main__":
    # Simple manual test
    test_texts = [
        "This is a great Chinese restaurant with excellent service.",
        "I love the Italian food here, especially the pasta.",
        "The Mexican cuisine was amazing, very authentic flavors.",
        "Another Chinese restaurant review, but with different details."
    ]
    
    # Initialize calculator (replace with your own model path before use)
    embedding_model_path = "/path/to/all-MiniLM-L6-v2"
    calculator = BatchDiversityCalculator(embedding_model_path, device='cpu')
    
    # Compute diversity rewards
    rewards = calculator.calculate_batch_diversity_rewards(test_texts)
    stats = calculator.get_batch_diversity_stats(test_texts)
    
    print("Test results:")
    for i, (text, reward) in enumerate(zip(test_texts, rewards)):
        print(f"  Sample {i+1}: {reward:.4f} - {text[:50]}...")
    
    print(f"\nStatistics: {stats}")
