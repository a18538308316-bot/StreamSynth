#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic reference data manager.
Implements a hybrid scheme: original dataset + dynamically collected high-quality samples.
"""
import numpy as np
import torch
import faiss
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

class DynamicReferenceManager:
    """Dynamic reference data manager."""
    
    def __init__(self, 
                 embedding_model,
                 device='cuda',
                 max_total_size=300,
                 min_novelty_threshold=0.3,
                 original_data_path=None):
        """Initialize dynamic reference data manager.

        Args:
            embedding_model: SentenceTransformer model used to compute embeddings.
            device: Computation device, e.g. ``"cuda"`` or ``"cpu"``.
            max_total_size: Maximum capacity of the reference pool.
            min_novelty_threshold: Minimum novelty score required to add a sample.
            original_data_path: Optional path to an original dataset used to seed references.
        """
        self.embedding_model = embedding_model
        self.device = device
        self.max_total_size = max_total_size
        self.min_novelty_threshold = min_novelty_threshold
        
        # Storage for reference texts and embeddings
        self.reference_texts = []
        self.reference_embeddings = None
        self.reference_index = None
        
        # Source markers
        self.data_sources = []  # 'original' or 'dynamic'
        self.original_count = 0
        self.dynamic_count = 0
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load original data if a path is provided, otherwise fall back to templates
        if original_data_path:
            self.load_original_data(original_data_path)
        else:
            self.initialize_with_templates()
            
    def initialize_with_templates(self):
        """Initialize the reference pool with template data (fallback strategy)."""
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
        
        print(f"✅ Initialized reference pool with {len(template_texts)} template texts")
        self._update_embeddings_and_index()
        
    def load_original_data(self, data_path: str, max_samples: int = 100):
        """Load reference texts from an original dataset."""
        try:
            data_path = Path(data_path)
            if not data_path.exists():
                print(f"⚠️ Original data file does not exist: {data_path}")
                self.initialize_with_templates()
                return
                
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract actual review texts to be used as references
            original_texts = []
            for item in data[:max_samples]:
                # Handle the original Yelp format: the "input" field contains "Text: ..."
                if 'input' in item and item['input'].strip():
                    input_text = item['input'].strip()
                    # Extract the content after the "Text: " prefix
                    if input_text.startswith('Text: '):
                        review_text = input_text[6:].strip()  # Remove the "Text: " prefix
                        if review_text and len(review_text) > 20:  # Ensure the review has a reasonable length
                            original_texts.append(review_text)
                    elif len(input_text) > 20:  # If there is no "Text: " prefix but the content is long enough
                        original_texts.append(input_text)
                # Fallback: check other possible fields
                elif 'review_text' in item and item['review_text'].strip():
                    original_texts.append(item['review_text'].strip())
                elif 'text' in item and item['text'].strip():
                    original_texts.append(item['text'].strip())
                    
            if not original_texts:
                print("⚠️ No valid original review texts found, initializing with template data instead")
                self.initialize_with_templates()
                return
                
            self.reference_texts = original_texts
            self.data_sources = ['original'] * len(original_texts)
            self.original_count = len(original_texts)
            
            print(f"✅ Loaded {len(original_texts)} reference reviews from the original dataset")
            print(f"   Example review: {original_texts[0][:100]}...")
            self._update_embeddings_and_index()
            
        except Exception as e:
            print(f"⚠️ Failed to load original data: {e}")
            import traceback
            traceback.print_exc()
            self.initialize_with_templates()
    
    def _update_embeddings_and_index(self):
        """Update embeddings and the FAISS index."""
        if not self.reference_texts:
            return
            
        try:
            # Compute embeddings
            self.reference_embeddings = self.embedding_model.encode(
                self.reference_texts,
                convert_to_tensor=True,
                device=self.device
            )
            
            # Refresh FAISS index
            self._rebuild_faiss_index()
            
        except Exception as e:
            print(f"⚠️ Failed to update embeddings and index: {e}")
    
    def _rebuild_faiss_index(self):
        """Rebuild the FAISS index from current embeddings."""
        if self.reference_embeddings is None:
            return
            
        try:
            embedding_dim = self.reference_embeddings.shape[1]
            self.reference_index = faiss.IndexFlatIP(embedding_dim)
            
            # Normalize embeddings
            ref_embeddings_normalized = self.reference_embeddings.cpu().numpy()
            ref_embeddings_normalized = ref_embeddings_normalized / np.linalg.norm(
                ref_embeddings_normalized, axis=1, keepdims=True
            )
            
            self.reference_index.add(ref_embeddings_normalized.astype('float32'))
            print(f"✅ FAISS index rebuilt, number of reference samples: {self.reference_index.ntotal}")
            
        except Exception as e:
            print(f"⚠️ Failed to rebuild FAISS index: {e}")
            self.reference_index = None
    
    def compute_novelty_score(self, text: str) -> float:
        """Compute the novelty score for a single text."""
        # Carefully check tensor values and None values
        if (self.reference_index is None or 
            self.reference_embeddings is None or 
            (hasattr(self.reference_embeddings, 'numel') and self.reference_embeddings.numel() == 0)):
            return 1.0  # If there is no reference data, treat the text as novel
            
        try:
            # Compute the embedding for the new text
            new_embedding = self.embedding_model.encode(
                [text], 
                convert_to_tensor=True,
                device=self.device
            )
            
            # Normalize the new embedding
            new_embedding_normalized = new_embedding.cpu().numpy()
            new_embedding_normalized = new_embedding_normalized / np.linalg.norm(
                new_embedding_normalized, axis=1, keepdims=True
            )
            
            # Search the k most similar reference samples
            k = min(10, self.reference_index.ntotal)
            similarities, indices = self.reference_index.search(
                new_embedding_normalized.astype('float32'), k
            )
            
            # novelty = 1 - max_similarity
            max_similarity = similarities[0].max() if len(similarities[0]) > 0 else 0.0
            novelty_score = 1.0 - max_similarity
            
            return float(novelty_score)
            
        except Exception as e:
            print(f"⚠️ Failed to compute novelty score: {e}")
            return 0.5  # Default medium novelty when computation fails
    
    def add_dynamic_sample(self, text: str, force_add: bool = False) -> bool:
        """Add a dynamic sample to the reference pool."""
        if not text.strip():
            return False
            
        # Compute novelty score
        novelty_score = self.compute_novelty_score(text)
        
        # Check whether the sample satisfies the novelty threshold
        if not force_add and novelty_score < self.min_novelty_threshold:
            return False
            
        # If the reference pool is not full, add directly
        if len(self.reference_texts) < self.max_total_size:
            self.reference_texts.append(text)
            self.data_sources.append('dynamic')
            self.dynamic_count += 1
            self._update_embeddings_and_index()
            return True
            
        # If the pool is full, try to replace the least novel dynamic sample
        return self._replace_least_novel_dynamic_sample(text, novelty_score)
    
    def _replace_least_novel_dynamic_sample(self, new_text: str, new_novelty: float) -> bool:
        """Replace the least novel dynamic sample with a new one if it is more novel."""
        # Find indices of all dynamic samples
        dynamic_indices = [
            i for i, source in enumerate(self.data_sources) 
            if source == 'dynamic'
        ]
        
        if not dynamic_indices:
            return False  # No dynamic sample available to replace
            
        # Compute novelty for all dynamic samples (relative to the current reference set)
        min_novelty = float('inf')
        replace_idx = -1
        
        for idx in dynamic_indices:
            current_text = self.reference_texts[idx]
            current_novelty = self.compute_novelty_score(current_text)
            
            if current_novelty < min_novelty:
                min_novelty = current_novelty
                replace_idx = idx
        
        # If the new sample is more novel than the least novel dynamic one, replace it
        if new_novelty > min_novelty:
            self.reference_texts[replace_idx] = new_text
            self._update_embeddings_and_index()
            return True
            
        return False
    
    def get_reference_data(self) -> Tuple[List[str], torch.Tensor, faiss.Index]:
        """Return the current reference texts, embeddings and FAISS index."""
        return self.reference_texts, self.reference_embeddings, self.reference_index
    
    def get_statistics(self) -> Dict:
        """Return statistics about the reference pool."""
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
        """Batch add samples and return the number of successfully added ones."""
        added_count = 0
        
        # If quality scores are provided, sort by score
        if quality_scores:
            paired_data = list(zip(texts, quality_scores))
            paired_data.sort(key=lambda x: x[1], reverse=True)  # Sort by quality score in descending order
            texts = [text for text, _ in paired_data]
        
        for text in texts:
            if self.add_dynamic_sample(text):
                added_count += 1
                
        return added_count