from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import json
import os

class GreekWordEmbeddings:
    def __init__(self, model_path: str = 'models/greek-bert', cache_dir: str = 'cache'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings_cache_file = self.cache_dir / 'embeddings_cache.json'
        self.text_embeddings_cache_file = self.cache_dir / 'text_embeddings_cache.json'
        
        # Load existing cache
        self.embedding_cache = self._load_cache(self.embeddings_cache_file)
        self.text_embedding_cache = self._load_cache(self.text_embeddings_cache_file)
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _load_cache(self, cache_file: Path) -> Dict:
        """Φόρτωση του cache από αρχείο."""
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cache from {cache_file}: {e}")
        return {}
        
    def _save_cache(self, cache: Dict, cache_file: Path):
        """Αποθήκευση του cache σε αρχείο."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving cache to {cache_file}: {e}")
            
    def get_word_embedding(self, word: str) -> torch.Tensor:
        """Get embedding for a single word with caching."""
        # Check cache first
        cache_key = word.lower()
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return torch.tensor(self.embedding_cache[cache_key])
            
        self.cache_misses += 1
        with torch.no_grad():
            inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[0][0].tolist()  # Convert to list for JSON serialization
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            if len(self.embedding_cache) % 100 == 0:  # Save every 100 new entries
                self._save_cache(self.embedding_cache, self.embeddings_cache_file)
                
            return torch.tensor(embedding)
            
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for a piece of text with caching."""
        # Create a cache key (using first 100 chars to avoid too long keys)
        cache_key = text[:100].lower()
        if cache_key in self.text_embedding_cache:
            self.cache_hits += 1
            return torch.tensor(self.text_embedding_cache[cache_key])
            
        self.cache_misses += 1
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model(**inputs)
            embedding = torch.mean(outputs.last_hidden_state[0], dim=0).tolist()
            
            # Cache the result
            self.text_embedding_cache[cache_key] = embedding
            if len(self.text_embedding_cache) % 50 == 0:  # Save every 50 new entries
                self._save_cache(self.text_embedding_cache, self.text_embeddings_cache_file)
                
            return torch.tensor(embedding)
            
    def batch_encode_texts(self, texts: List[str], batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """Encode multiple texts in batches with caching."""
        embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = text[:100].lower()
            if cache_key in self.text_embedding_cache:
                self.cache_hits += 1
                embeddings[text] = torch.tensor(self.text_embedding_cache[cache_key])
            else:
                self.cache_misses += 1
                uncached_texts.append(text)
                uncached_indices.append(i)
                
        # Process uncached texts in batches
        if uncached_texts:
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                with torch.no_grad():
                    inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                    
                    for j, text in enumerate(batch):
                        cache_key = text[:100].lower()
                        embedding = batch_embeddings[j].tolist()
                        self.text_embedding_cache[cache_key] = embedding
                        embeddings[text] = torch.tensor(embedding)
                        
            # Save cache after processing batch
            self._save_cache(self.text_embedding_cache, self.text_embeddings_cache_file)
            
        return embeddings
        
    def get_cache_stats(self) -> Dict[str, float]:
        """Return cache statistics."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            hit_rate = 0
        else:
            hit_rate = self.cache_hits / total * 100
            
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'word_cache_size': len(self.embedding_cache),
            'text_cache_size': len(self.text_embedding_cache)
        }
        
    def clear_cache(self):
        """Clear both in-memory and file caches."""
        self.embedding_cache.clear()
        self.text_embedding_cache.clear()
        if self.embeddings_cache_file.exists():
            os.remove(self.embeddings_cache_file)
        if self.text_embeddings_cache_file.exists():
            os.remove(self.text_embeddings_cache_file)
        self.cache_hits = 0
        self.cache_misses = 0
        
    def calculate_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        # Κανονικοποίηση των διανυσμάτων
        emb1_normalized = emb1 / emb1.norm()
        emb2_normalized = emb2 / emb2.norm()
        # Υπολογισμός ομοιότητας συνημιτόνου
        similarity = torch.dot(emb1_normalized, emb2_normalized)
        return float(similarity)
        
    def find_similar_words(self, word: str, vocabulary: List[str], n: int = 3) -> List[Tuple[str, float]]:
        """Find n most similar words from the given vocabulary."""
        target_embedding = self.get_word_embedding(word)
        similarities = []
        
        for vocab_word in vocabulary:
            if vocab_word == word:
                continue
            vocab_embedding = self.get_word_embedding(vocab_word)
            similarity = self.calculate_similarity(target_embedding, vocab_embedding)
            similarities.append((vocab_word, similarity))
        
        # Ταξινόμηση με βάση την ομοιότητα και επιστροφή των top-n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
        
    def semantic_search(self, query: str, documents: Dict[str, str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Εκτέλεση semantic search στα documents με βάση το query.
        
        Args:
            query: Το ερώτημα αναζήτησης
            documents: Dictionary με {document_id: text}
            top_k: Αριθμός κορυφαίων αποτελεσμάτων
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        try:
            # Υπολογισμός embedding για το query
            query_embedding = self.get_text_embedding(query)
            
            # Υπολογισμός embeddings για τα documents σε batches
            doc_embeddings = self.batch_encode_texts(list(documents.values()))
            
            # Υπολογισμός ομοιότητας με κάθε document
            similarities = []
            for doc_id, doc_text in documents.items():
                doc_embedding = doc_embeddings[doc_text]
                similarity = self.calculate_similarity(query_embedding, doc_embedding)
                similarities.append((doc_id, similarity))
            
            # Ταξινόμηση και επιστροφή των top-k αποτελεσμάτων
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return []
            
    def _load_cache(self, cache_file: Path) -> Dict:
        """Φόρτωση του cache από αρχείο."""
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cache from {cache_file}: {e}")
        return {}
        
    def _save_cache(self, cache: Dict, cache_file: Path):
        """Αποθήκευση του cache σε αρχείο."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving cache to {cache_file}: {e}") 