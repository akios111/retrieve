import logging
import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np
from datetime import datetime
from pathlib import Path

# Ρύθμιση logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from ..preprocessing.text_processor import TextPreprocessor

class SearchEngine:
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.index = {}  # Ανεστραμμένο ευρετήριο
        self.articles = {}  # Αποθήκευση άρθρων
        self.document_vectors = {}  # Διανύσματα εγγράφων για VSM
        self.idf = {}  # IDF τιμές για κάθε όρο
        self.pagerank_scores = {}  # PageRank scores
        self.feature_weights = {  # Βάρη για το Learning to Rank
            'vsm_score': 0.3,
            'bm25_score': 0.3,
            'pagerank': 0.2,
            'title_match': 0.1,
            'freshness': 0.1
        }
        
        self._load_resources()
        
    def _load_resources(self):
        """Φόρτωση πόρων από τα αρχεία."""
        try:
            data_dir = Path(__file__).parent.parent.parent / 'data'
            logger.info(f"Loading resources from directory: {data_dir}")
            
            # Φόρτωση άρθρων
            articles_path = data_dir / 'processed_articles.json'
            logger.info(f"Loading articles from {articles_path}")
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles_list = json.load(f)
                # Μετατροπή λίστας σε λεξικό με κλειδί τον τίτλο
                self.articles = {article['title']: article for article in articles_list}
            logger.info(f"Loaded {len(self.articles)} articles")
            logger.info(f"Sample article titles: {list(self.articles.keys())[:5]}")
                
            # Φόρτωση ευρετηρίου
            index_path = data_dir / 'inverted_index.json'
            logger.info(f"Loading index from {index_path}")
            with open(index_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
            logger.info(f"Loaded index with {len(self.index)} terms")
            logger.info(f"Sample index terms: {list(self.index.keys())[:5]}")
                
            # Φόρτωση διανυσμάτων εγγράφων
            vectors_path = data_dir / 'document_vectors.json'
            logger.info(f"Loading document vectors from {vectors_path}")
            with open(vectors_path, 'r', encoding='utf-8') as f:
                self.document_vectors = json.load(f)
            logger.info(f"Loaded vectors for {len(self.document_vectors)} documents")
            logger.info(f"Sample document vector keys: {list(self.document_vectors.keys())[:5]}")
                
            # Φόρτωση IDF τιμών
            idf_path = data_dir / 'idf_values.json'
            logger.info(f"Loading IDF values from {idf_path}")
            with open(idf_path, 'r', encoding='utf-8') as f:
                self.idf = json.load(f)
            logger.info(f"Loaded {len(self.idf)} IDF values")
            logger.info(f"Sample IDF terms: {list(self.idf.keys())[:5]}")
                
            # Φόρτωση PageRank scores
            metadata_path = data_dir / 'index_metadata.json'
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.pagerank_scores = metadata.get('pagerank_scores', {})
            logger.info(f"Loaded {len(self.pagerank_scores)} PageRank scores")
            logger.info(f"Sample PageRank scores: {list(self.pagerank_scores.items())[:5]}")
                
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            raise
            
    def search(self, query: str, method: str = 'vsm', k: int = 10) -> List[Tuple[str, float]]:
        """
        Εκτέλεση αναζήτησης με την επιλεγμένη μέθοδο.
        
        Args:
            query: Το ερώτημα αναζήτησης
            method: Η μέθοδος αναζήτησης ('boolean', 'vsm', ή 'bm25')
            k: Αριθμός αποτελεσμάτων προς επιστροφή
            
        Returns:
            Λίστα με tuples (τίτλος, σκορ) ταξινομημένη κατά φθίνουσα σειρά σκορ
        """
        try:
            if method == 'boolean':
                results = self.boolean_search(query)
            elif method == 'vsm':
                results = self.vsm_search(query)
            elif method == 'bm25':
                results = self.bm25_search(query)
            else:
                raise ValueError(f"Μη έγκυρη μέθοδος αναζήτησης: {method}")
                
            # Εφαρμογή Learning to Rank
            ranked_results = self.apply_learning_to_rank(query, results)
            
            # Επιστροφή των top-k αποτελεσμάτων
            return ranked_results[:k]
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []
            
    def boolean_search(self, query: str) -> List[Tuple[str, float]]:
        """Εκτέλεση Boolean αναζήτησης."""
        try:
            # Επεξεργασία του ερωτήματος
            processed_query = self.text_processor.clean_text(query)
            terms = self.text_processor.tokenize_and_remove_stopwords(processed_query)
            
            # Εύρεση εγγράφων για κάθε όρο
            results = []
            for term in terms:
                if term in self.index:
                    docs = self.index[term]
                    results.extend([(doc, 1.0) for doc in docs])
                    
            return sorted(results, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in boolean search: {str(e)}")
            return []
            
    def vsm_search(self, query: str) -> List[Tuple[str, float]]:
        """Εκτέλεση Vector Space Model αναζήτησης."""
        try:
            logger.info(f"VSM search - Processing query: {query}")
            # Επεξεργασία του ερωτήματος
            processed_query = self.text_processor.clean_text(query)
            query_terms = self.text_processor.tokenize_and_remove_stopwords(processed_query)
            logger.info(f"VSM search - Query terms: {query_terms}")
            
            # Υπολογισμός διανύσματος ερωτήματος
            query_vector = defaultdict(float)
            for term in query_terms:
                if term in self.idf:
                    query_vector[term] += self.idf[term]
                    logger.info(f"VSM search - Found term '{term}' in IDF")
                else:
                    logger.warning(f"VSM search - Term '{term}' not in IDF")
            logger.info(f"VSM search - Final query vector: {dict(query_vector)}")
                    
            # Υπολογισμός ομοιότητας συνημιτόνου
            scores = []
            for doc_id, doc_vector in self.document_vectors.items():
                score = self._cosine_similarity(query_vector, doc_vector)
                if not np.isnan(score) and score > 0:  # Έλεγχος για θετικά σκορ
                    scores.append((doc_id, float(score)))
                    if len(scores) <= 3:  # Log μόνο για τα πρώτα 3 έγγραφα
                        logger.info(f"VSM search - Document '{doc_id}' got score {score}")
                        common_terms = set(query_vector.keys()) & set(doc_vector.keys())
                        logger.info(f"VSM search - Common terms with '{doc_id}': {common_terms}")
            
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            logger.info(f"VSM search - Found {len(sorted_scores)} results")
            if sorted_scores:
                logger.info(f"VSM search - Top 3 results: {sorted_scores[:3]}")
                # Εμφάνιση των τίτλων των top-3 αποτελεσμάτων
                for doc_id, score in sorted_scores[:3]:
                    article = self.articles.get(doc_id, {})
                    logger.info(f"VSM search - Title: {doc_id}")
                    logger.info(f"VSM search - Score: {score}")
                    logger.info(f"VSM search - Categories: {article.get('categories', [])}")
            
            return sorted_scores
            
        except Exception as e:
            logger.error(f"Error in VSM search: {str(e)}")
            return []
            
    def bm25_search(self, query: str) -> List[Tuple[str, float]]:
        """Εκτέλεση BM25 αναζήτησης."""
        try:
            # Παράμετροι BM25
            k1 = 1.2
            b = 0.75
            
            # Επεξεργασία του ερωτήματος
            processed_query = self.text_processor.clean_text(query)
            query_terms = self.text_processor.tokenize_and_remove_stopwords(processed_query)
            
            scores = defaultdict(float)
            for term in query_terms:
                if term in self.index:
                    for doc_id in self.index[term]:
                        # Υπολογισμός BM25 σκορ
                        tf = self.document_vectors[doc_id].get(term, 0)
                        idf = self.idf.get(term, 0)
                        doc_len = sum(self.document_vectors[doc_id].values())
                        avg_doc_len = np.mean([sum(d.values()) for d in self.document_vectors.values()])
                        
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
                        score = idf * numerator / denominator
                        
                        if not np.isnan(score):  # Έλεγχος για NaN
                            scores[doc_id] += float(score)
                            
            return sorted([(doc_id, score) for doc_id, score in scores.items()], 
                        key=lambda x: x[1], reverse=True)
                        
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            return []
            
    def apply_learning_to_rank(self, query: str, initial_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Εφαρμογή Learning to Rank στα αρχικά αποτελέσματα."""
        try:
            final_scores = []
            for doc_id, initial_score in initial_results:
                if doc_id not in self.articles:
                    continue
                    
                # Υπολογισμός επιμέρους σκορ
                pagerank_score = self.pagerank_scores.get(doc_id, 0)
                title_match_score = self._calculate_title_match_score(query, doc_id)
                freshness_score = self._calculate_freshness_score(doc_id)
                
                # Συνδυασμός σκορ με βάρη
                final_score = (
                    initial_score * self.feature_weights['vsm_score'] +
                    pagerank_score * self.feature_weights['pagerank'] +
                    title_match_score * self.feature_weights['title_match'] +
                    freshness_score * self.feature_weights['freshness']
                )
                
                if not np.isnan(final_score):  # Έλεγχος για NaN
                    final_scores.append((doc_id, float(final_score)))
                    
            return sorted(final_scores, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in learning to rank: {str(e)}")
            return initial_results
            
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Υπολογισμός ομοιότητας συνημιτόνου μεταξύ δύο διανυσμάτων."""
        try:
            intersection = set(vec1.keys()) & set(vec2.keys())
            numerator = sum(vec1[x] * vec2[x] for x in intersection)
            
            sum1 = sum(vec1[x]**2 for x in vec1.keys())
            sum2 = sum(vec2[x]**2 for x in vec2.keys())
            denominator = np.sqrt(sum1) * np.sqrt(sum2)
            
            if denominator == 0:
                return 0.0
                
            return float(numerator) / denominator
            
        except Exception as e:
            logger.error(f"Error in cosine similarity: {str(e)}")
            return 0.0
            
    def _calculate_title_match_score(self, query: str, doc_id: str) -> float:
        """Υπολογισμός σκορ ταιριάσματος τίτλου."""
        try:
            processed_query = set(self.text_processor.tokenize_and_remove_stopwords(query))
            processed_title = set(self.text_processor.tokenize_and_remove_stopwords(doc_id))
            
            intersection = processed_query & processed_title
            if not processed_query or not processed_title:
                return 0.0
                
            return float(len(intersection)) / max(len(processed_query), len(processed_title))
            
        except Exception as e:
            logger.error(f"Error in title match score: {str(e)}")
            return 0.0
            
    def _calculate_freshness_score(self, doc_id: str) -> float:
        """Υπολογισμός σκορ φρεσκάδας του εγγράφου."""
        try:
            article = self.articles.get(doc_id, {})
            date_str = article.get('date')
            if not date_str:
                return 0.0
                
            article_date = datetime.strptime(date_str, '%Y-%m-%d')
            days_old = (datetime.now() - article_date).days
            
            # Εκθετική μείωση με βάση την ηλικία
            return float(np.exp(-days_old / 365))  # Ετήσια μείωση
            
        except Exception as e:
            logger.error(f"Error in freshness score: {str(e)}")
            return 0.0
            
    def update_feature_weights(self, click_data: List[Tuple[str, str, float]]):
        """Ενημέρωση βαρών με βάση τα δεδομένα κλικ."""
        try:
            if not click_data:
                return
                
            # Απλή προσαρμογή βαρών με βάση το χρόνο παραμονής
            for query, doc_id, dwell_time in click_data:
                if dwell_time > 30:  # Θετικό σήμα αν ο χρήστης έμεινε πάνω από 30 δευτερόλεπτα
                    self.feature_weights['title_match'] *= 1.1
                    self.feature_weights['freshness'] *= 0.9
                else:  # Αρνητικό σήμα
                    self.feature_weights['title_match'] *= 0.9
                    self.feature_weights['freshness'] *= 1.1
                    
            # Κανονικοποίηση βαρών
            total = sum(self.feature_weights.values())
            for feature in self.feature_weights:
                self.feature_weights[feature] /= total
                
        except Exception as e:
            logger.error(f"Error updating feature weights: {str(e)}") 