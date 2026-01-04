#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NovelSum-based diversity calculation module with dynamic reference data management.
"""
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Optional
from .dynamic_reference_manager import DynamicReferenceManager

# NovelSum configuration
NOVELSUM_CONFIG = {
    'density_power': 0.5,
    'distance_power': 1.0,
    'neighbors': 10,
    'embedding_model_path': '/public/home/huzhenlin2023/synthetic_data/all-MiniLM-L6-v2',
    'max_length': 512,
    # Dynamic reference data configuration
    'max_reference_size': 300,
    # Lower threshold to encourage adding more novel samples to the pool
    'min_novelty_threshold': 0.1,
    'original_data_path': '/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/original_yelp_train_100_dataset.json',
}

class NovelSumDiversityCalculator:
    """NovelSum-based diversity calculator with dynamic reference data management."""
    
    def __init__(self, embedding_model_path, device='cuda', max_length=256, 
                 use_dynamic_reference=True, original_data_path=None):
        self.device = device
        self.max_length = max_length
        self.use_dynamic_reference = use_dynamic_reference
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.embedding_model.to(device)
        
        # Initialize reference data manager
        if use_dynamic_reference:
            self.reference_manager = DynamicReferenceManager(
                embedding_model=self.embedding_model,
                device=device,
                max_total_size=NOVELSUM_CONFIG.get('max_reference_size', 300),
                min_novelty_threshold=NOVELSUM_CONFIG.get('min_novelty_threshold', 0.3),
                original_data_path=original_data_path or NOVELSUM_CONFIG.get('original_data_path')
            )
            print("✅ Dynamic reference data manager initialized")
        else:
            # Backward compatibility: use static reference data
            self.reference_embeddings = None
            self.reference_index = None
            self.setup_static_reference_data()
        
    def get_reference_data(self):
        """Get current reference data (texts, embeddings, index)."""
        if self.use_dynamic_reference:
            return self.reference_manager.get_reference_data()
        else:
            return None, self.reference_embeddings, self.reference_index
    
    def add_training_samples(self, texts: List[str], quality_scores: Optional[List[float]] = None):
        """Add training samples to the dynamic reference pool."""
        if not self.use_dynamic_reference:
            return 0
        
        return self.reference_manager.batch_add_samples(texts, quality_scores)
    
    def get_reference_statistics(self):
        """Get statistics for current reference data."""
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
        """Set up static reference data (backward compatible path)."""
        # Example restaurant review reference corpus (can be replaced with a real dataset)
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
        
        print("🔄 Setting up NovelSum static reference data...")
        try:
            self.reference_embeddings = self.embedding_model.encode(
                reference_texts, 
                convert_to_tensor=True,
                device=self.device
            )
            self.setup_faiss_index_static()
            print("✅ NovelSum static reference data initialized")
        except Exception as e:
            print(f"⚠️ Failed to set NovelSum static reference data: {e}")
            self.reference_embeddings = None
    
    def setup_faiss_index_static(self):
        """Set up FAISS index for fast similarity search (static version)."""
        if self.reference_embeddings is None:
            return
        
        try:
            embedding_dim = self.reference_embeddings.shape[1]
            self.reference_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (cosine similarity)
            
            # Normalize embeddings so inner product corresponds to cosine similarity
            ref_embeddings_normalized = self.reference_embeddings.cpu().numpy()
            ref_embeddings_normalized = ref_embeddings_normalized / np.linalg.norm(
                ref_embeddings_normalized, axis=1, keepdims=True
            )
            
            self.reference_index.add(ref_embeddings_normalized.astype('float32'))
            print(f"✅ FAISS index created, number of reference items: {self.reference_index.ntotal}")
        except Exception as e:
            print(f"⚠️ Failed to create FAISS index: {e}")
            self.reference_index = None
    
    def compute_cosine_distance_matrix(self, embeddings):
        """Compute cosine distance matrix from embeddings."""
        try:
            # Ensure we have a numpy array
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.detach().cpu().numpy()
            else:
                embeddings_np = np.array(embeddings)
            
            # Use float64 for better numerical stability
            embeddings_np = embeddings_np.astype(np.float64)
            
            # Compute cosine similarity matrix
            cosine_sim_matrix = cosine_similarity(embeddings_np)
            # Convert to distance matrix
            distance_matrix = 1 - cosine_sim_matrix
            
            # Ensure we return a numpy array
            return np.array(distance_matrix, dtype=np.float64)
            
        except Exception as e:
            print(f"⚠️ Failed to compute cosine distance matrix: {e}")
            n_samples = len(embeddings) if hasattr(embeddings, '__len__') else 1
            return np.ones((n_samples, n_samples), dtype=np.float64)
    
    def compute_local_density(self, embeddings, n_neighbors=10, power=0.5):
        """Compute local density for each embedding using k-nearest neighbors."""
        try:
            distance_matrix = self.compute_cosine_distance_matrix(embeddings)
            n_samples = distance_matrix.shape[0]
            n_neighbors = min(n_neighbors, n_samples - 1)
            
            if n_neighbors <= 0:
                return np.ones(n_samples, dtype=np.float64)
            
            densities = []
            for i in range(n_samples):
                # Get distances to all other samples (exclude self)
                distances = distance_matrix[i].copy()
                distances = np.delete(distances, i)
                
                if len(distances) == 0:
                    densities.append(1.0)
                    continue
                
                # Find k nearest neighbors
                if len(distances) >= n_neighbors:
                    nearest_distances = np.partition(distances, n_neighbors-1)[:n_neighbors]
                else:
                    nearest_distances = distances
                
                # Compute local density (smaller distance -> higher density)
                avg_distance = np.mean(nearest_distances)
                density = 1.0 / (1.0 + avg_distance)
                density = density ** power  # Apply power adjustment
                densities.append(float(density))
            
            return np.array(densities, dtype=np.float64)
            
        except Exception as e:
            print(f"⚠️ Failed to compute local density: {e}")
            n_samples = len(embeddings) if hasattr(embeddings, '__len__') else 1
            return np.ones(n_samples, dtype=np.float64)
    
    def weighted_average(self, row, power=1.0):
        """Compute average distance (patched version)."""
        try:
            # Ensure input is a numpy array
            row = np.array(row, dtype=np.float64)
            
            # Exclude self (diagonal elements should be 0)
            non_zero_distances = row[row > 1e-10]
            
            if len(non_zero_distances) > 0:
                result = np.mean(non_zero_distances)
            else:
                result = np.mean(row)
                
            return float(result)
            
        except Exception as e:
            print(f"⚠️ Failed to compute average distance: {e}")
            return float(np.mean(np.array(row))) if hasattr(row, '__len__') else 0.5
    
    def calculate_novelsum_score(self, texts, density_power=0.5, distance_power=1.0, neighbors=10):
        """Calculate NovelSum diversity score for a list of texts."""
        try:
            # Basic validation
            if not texts or len(texts) < 2:
                print(f"⚠️ Not enough texts for NovelSum scoring: {len(texts) if texts else 0}")
                return 0.5  # Return neutral score for single sample
            
            # Filter empty texts and ensure everything is a string
            valid_texts = []
            for text in texts:
                if text is not None:
                    # Ensure text is converted to string
                    str_text = str(text).strip() if not isinstance(text, str) else text.strip()
                    if len(str_text) > 5:
                        valid_texts.append(str_text)
            
            if len(valid_texts) < 2:
                print(f"⚠️ Not enough valid texts: {len(valid_texts)}")
                return 0.5
            
            print(f"🔍 DEBUG: Calculating NovelSum score for {len(valid_texts)} texts")
            
            # Generate embeddings for valid texts
            try:
                embeddings = self.embedding_model.encode(
                    valid_texts, 
                    convert_to_tensor=True,
                    device=self.device
                )
                print(f"🔍 DEBUG: Embeddings shape: {embeddings.shape}")
            except Exception as e:
                print(f"⚠️ Failed to generate embeddings: {e}")
                print(f"🔍 DEBUG: valid_texts types: {[type(t) for t in valid_texts[:3]]}")
                return 0.5
            
            # Ensure embeddings are a proper torch tensor
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, device=self.device, dtype=torch.float32)
            
            # Validate embeddings shape
            if embeddings.dim() != 2 or embeddings.shape[0] != len(valid_texts):
                print(f"⚠️ Unexpected embeddings shape: {embeddings.shape}, expected: ({len(valid_texts)}, embedding_dim)")
                return 0.5
            
            # Compute local density for each sample
            try:
                densities = self.compute_local_density(embeddings, neighbors, density_power)
                print(f"🔍 DEBUG: Densities shape: {densities.shape}, sample: {densities[:3]}")
            except Exception as e:
                print(f"⚠️ Error while computing local density: {e}")
                return 0.5
            
            # Compute pairwise distance matrix
            try:
                distance_matrix = self.compute_cosine_distance_matrix(embeddings)
                print(f"🔍 DEBUG: Distance matrix shape: {distance_matrix.shape}")
            except Exception as e:
                print(f"⚠️ Error while computing distance matrix: {e}")
                return 0.5
            
            # Compute NovelSum scores (patched version)
            scores = []
            for i in range(len(valid_texts)):
                try:
                    row = distance_matrix[i]
                    density_weight = float(densities[i])
                    
                    # Compute average distance (diversity measure)
                    avg_distance = self.weighted_average(row, distance_power)
                    
                    # Combine distance and density.
                    # Larger distance means more novelty; higher density means a more crowded region.
                    # Use the inverse of density as an adjustment factor.
                    density_factor = 1.0 / (density_weight + 1e-8)
                    novelsum_score = avg_distance * (1.0 + density_factor)
                    
                    scores.append(float(novelsum_score))
                    
                except Exception as e:
                    print(f"⚠️ Failed to compute NovelSum score for sample {i}: {e}")
                    scores.append(0.5)
            
            # Return average score
            if not scores:
                print("⚠️ No valid NovelSum scores computed")
                return 0.5
                
            final_score = float(np.mean(scores))
            print(f"🔍 DEBUG: Final NovelSum score: {final_score}")
            
            # Sanity check on score range
            if final_score < 0 or final_score > 10 or np.isnan(final_score) or np.isinf(final_score):
                print(f"⚠️ Abnormal NovelSum score: {final_score}, using default value 0.5")
                return 0.5
                
            return final_score
            
        except Exception as e:
            print(f"⚠️ Failed to compute NovelSum score: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return 0.5
    
    def calculate_internal_diversity(self, texts):
        """Compute internal diversity (average pairwise distance within a batch)."""
        try:
            if len(texts) < 2:
                return 0.5
            
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_tensor=True,
                device=self.device
            )
            
            distance_matrix = self.compute_cosine_distance_matrix(embeddings)
            
            # Compute mean of upper-triangular entries (excluding diagonal)
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
            print(f"⚠️ Failed to compute internal diversity: {e}")
            return 0.5

def calculate_restaurant_specific_diversity(texts):
    """Calculate restaurant-review-specific diversity based on aspect coverage."""
    try:
        if len(texts) < 2:
            return 0.5
        
        # Aspect-specific keyword sets for restaurant reviews
        aspect_keywords = {
            'food_quality': ['delicious', 'tasty', 'bland', 'awful', 'amazing', 'terrible', 'fresh', 'stale'],
            'service': ['friendly', 'rude', 'quick', 'slow', 'attentive', 'negligent', 'professional'],
            'atmosphere': ['cozy', 'loud', 'romantic', 'casual', 'elegant', 'crowded', 'peaceful'],
            'price': ['expensive', 'cheap', 'reasonable', 'overpriced', 'value', 'affordable', 'costly'],
            'location': ['convenient', 'remote', 'accessible', 'parking', 'downtown', 'suburban']
        }
        
        # Analyze which aspects each text covers
        text_aspects = []
        for text in texts:
            text_lower = text.lower()
            aspects = set()
            
            for aspect, keywords in aspect_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    aspects.add(aspect)
            
            text_aspects.append(aspects)
        
        # Measure diversity of aspect coverage
        all_aspects = set()
        for aspects in text_aspects:
            all_aspects.update(aspects)
        
        if not all_aspects:
            return 0.3  # Fallback when no aspects are recognized
        
        # Compute Jaccard-based diversity over aspect sets
        diversities = []
        for i in range(len(text_aspects)):
            for j in range(i + 1, len(text_aspects)):
                aspects1, aspects2 = text_aspects[i], text_aspects[j]
                if len(aspects1) == 0 and len(aspects2) == 0:
                    jaccard = 1.0  # Both empty: treat as fully similar
                else:
                    intersection = len(aspects1.intersection(aspects2))
                    union = len(aspects1.union(aspects2))
                    jaccard = intersection / union if union > 0 else 0.0
                
                diversity = 1.0 - jaccard  # Convert to diversity
                diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.5
        
    except Exception as e:
            print(f"⚠️ Failed to compute restaurant-specific diversity: {e}")
        return 0.5

def extract_text_content_global(completion):
    """Global helper: extract actual text content from a completion string (optimized version)."""
    try:
        # First, separate generation content from any prompt prefix
        generation = separate_prompt_and_generation_global(completion)
        
        # Extract "Text" field if present
        if "Text:" in generation:
            text_part = generation.split("Text:")[1]
            if "Label:" in text_part:
                text_part = text_part.split("Label:")[0]
            return text_part.strip()
        
        # Try to extract from JSON block if present
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
        
        # Fallback: return the first 200 characters of the cleaned generation
        return generation.strip()[:200]
    except Exception:
        # Ultimate fallback: return the first 200 characters of raw completion
        return completion.strip()[:200]

def separate_prompt_and_generation_global(completion):
    """Global helper: separate true model generation from GRPO completion (strip prompts)."""
    try:
        # Strategy 1: try to find the first complete JSON object
        json_start = completion.find('{')
        if json_start != -1:
            # From the first '{', find the matching '}'
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
                # Found the first complete JSON span
                first_json = completion[json_start:json_end + 1]
                
                # Validate that this is a proper JSON object
                try:
                    import json
                    json.loads(first_json)
                    # First valid JSON object in the completion
                    return first_json
                except json.JSONDecodeError:
                    # Not a valid JSON object, fall through to other strategies
                    pass
        
        # Strategy 2: search for assistant markers
        assistant_markers = [
            'assistant:',
            'Assistant:',
        ]
        
        for marker in assistant_markers:
            pos = completion.find(marker)
            if pos != -1:
                remaining = completion[pos + len(marker):].strip()
                if len(remaining) > 30:  # Ensure there is enough content
                    return remaining
        
            # Strategy 3: search for generation-start markers
        generation_markers = [
            'Here is',
            'here is',
            'Based on',
            'based on',
        ]
        
        for marker in generation_markers:
            pos = completion.find(marker)
            if pos != -1 and pos > len(completion) * 0.2:  # In the later 80% of the text
                remaining = completion[pos:].strip()
                if len(remaining) > 50:  # Ensure there is enough content
                    return remaining
        
        # Strategy 4: for very long completions, drop the early part (likely prompt echo)
        if len(completion) > 1000:
            split_point = int(len(completion) * 0.6)
            return completion[split_point:].strip()
        
        # Final fallback
        return completion.strip()
        
    except Exception as e:
        print(f"⚠️ Error while separating prompt and generation: {e}")
        return completion.strip()