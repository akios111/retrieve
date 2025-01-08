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
from ..preprocessing.greek_embeddings import GreekWordEmbeddings

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
        self.embeddings = GreekWordEmbeddings()
        self.semantic_weight = 0.2  # Μείωση από 0.3 σε 0.2
        
        self._load_resources()
        
    def _load_resources(self):
        """Φόρτωση πόρων από τα αρχεία."""
        try:
            data_dir = Path(__file__).parent.parent.parent / 'data'
            logger.info(f"Loading resources from directory: {data_dir}")
            
            # Φόρτωση άρθρων
            articles_path = data_dir / 'processed_articles.json'
            logger.info(f"Loading articles from {articles_path}")
            with open(articles_path, 'r', encoding='utf-8-sig') as f:
                articles_list = json.load(f)
                # Μετατροπή λίστας σε λεξικό με κλειδί τον τίτλο
                self.articles = {article['title']: article for article in articles_list}
                
            # Έλεγχος κατηγοριών
            articles_with_categories = 0
            sample_categories = []
            for title, article in list(self.articles.items())[:5]:
                categories = article.get('categories', [])
                if categories:
                    articles_with_categories += 1
                    sample_categories.extend(categories)
                    logger.info(f"Article '{title}' has categories: {categories}")
                else:
                    logger.warning(f"Article '{title}' has no categories!")
                    
            logger.info(f"Loaded {len(self.articles)} articles")
            logger.info(f"Found {articles_with_categories} articles with categories in first 5 articles")
            if sample_categories:
                logger.info(f"Sample categories found: {list(set(sample_categories))}")
                
            # Φόρτωση ευρετηρίου
            index_path = data_dir / 'inverted_index.json'
            logger.info(f"Loading index from {index_path}")
            with open(index_path, 'r', encoding='utf-8-sig') as f:
                self.index = json.load(f)
            logger.info(f"Loaded index with {len(self.index)} terms")
            logger.info(f"Sample index terms: {list(self.index.keys())[:5]}")
                
            # Φόρτωση διανυσμάτων εγγράφων
            vectors_path = data_dir / 'document_vectors.json'
            logger.info(f"Loading document vectors from {vectors_path}")
            with open(vectors_path, 'r', encoding='utf-8-sig') as f:
                self.document_vectors = json.load(f)
            logger.info(f"Loaded vectors for {len(self.document_vectors)} documents")
            logger.info(f"Sample document vector keys: {list(self.document_vectors.keys())[:5]}")
                
            # Φόρτωση IDF τιμών
            idf_path = data_dir / 'idf_values.json'
            logger.info(f"Loading IDF values from {idf_path}")
            with open(idf_path, 'r', encoding='utf-8-sig') as f:
                self.idf = json.load(f)
            logger.info(f"Loaded {len(self.idf)} IDF values")
            logger.info(f"Sample IDF terms: {list(self.idf.keys())[:5]}")
                
            # Φόρτωση PageRank scores
            metadata_path = data_dir / 'index_metadata.json'
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8-sig') as f:
                metadata = json.load(f)
                self.pagerank_scores = metadata.get('pagerank_scores', {})
            logger.info(f"Loaded {len(self.pagerank_scores)} PageRank scores")
            logger.info(f"Sample PageRank scores: {list(self.pagerank_scores.items())[:5]}")
                
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            raise
            
    def search(self, query: str, method: str = 'vsm', k: int = 10, categories: List[str] = None, date_from: str = None, date_to: str = None) -> List[Tuple[str, float]]:
        """
        Εκτέλεση αναζήτησης με την επιλεγμένη μέθοδο και semantic search.
        """
        try:
            # Αρχική αναζήτηση με την επιλεγμένη μέθοδο
            initial_results = []
            if method == 'boolean':
                initial_results = self.boolean_search(query)
            elif method == 'vsm':
                initial_results = self.vsm_search(query)
            elif method == 'bm25':
                initial_results = self.bm25_search(query)
            else:
                raise ValueError(f"Μη έγκυρη μέθοδος αναζήτησης: {method}")
                
            # Semantic search
            documents = {doc_id: self.articles[doc_id].get('text', '') for doc_id, _ in initial_results}
            semantic_results = self.embeddings.semantic_search(query, documents, top_k=len(documents))
            
            # Συνδυασμός scores
            combined_scores = {}
            max_initial_score = max((score for _, score in initial_results), default=1.0)
            max_semantic_score = max((score for _, score in semantic_results), default=1.0)
            
            # Κανονικοποίηση και συνδυασμός scores
            for doc_id, score in initial_results:
                normalized_score = score / max_initial_score
                combined_scores[doc_id] = (1 - self.semantic_weight) * normalized_score
                
            for doc_id, score in semantic_results:
                normalized_score = score / max_semantic_score
                if doc_id in combined_scores:
                    combined_scores[doc_id] += self.semantic_weight * normalized_score
                else:
                    combined_scores[doc_id] = self.semantic_weight * normalized_score
            
            # Ταξινόμηση τελικών αποτελεσμάτων
            final_results = [(doc_id, score) for doc_id, score in combined_scores.items()]
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            # Φιλτράρισμα με βάση κατηγορίες και ημερομηνίες
            filtered_results = self._apply_filters(final_results, categories, date_from, date_to)
            
            return filtered_results[:k]
            
        except Exception as e:
            self.logger.error(f"Error in search: {str(e)}")
            return []
            
    def _apply_filters(self, results: List[Tuple[str, float]], categories: List[str] = None, date_from: str = None, date_to: str = None) -> List[Tuple[str, float]]:
        """Εφαρμογή φίλτρων στα αποτελέσματα."""
        filtered_results = []
        
        for doc_id, score in results:
            article = self.articles.get(doc_id)
            if not article:
                continue
                
            # Φιλτράρισμα με βάση τις κατηγορίες
            if categories:
                article_categories = article.get('categories', [])
                if not any(cat in article_categories for cat in categories):
                    continue
                    
            # Φιλτράρισμα με βάση την ημερομηνία
            if date_from or date_to:
                article_date = article.get('date', '')
                if article_date:
                    article_date = datetime.strptime(article_date, '%Y-%m-%d')
                    if date_from:
                        from_date = datetime.strptime(date_from, '%Y-%m-%d')
                        if article_date < from_date:
                            continue
                    if date_to:
                        to_date = datetime.strptime(date_to, '%Y-%m-%d')
                        if article_date > to_date:
                            continue
                            
            filtered_results.append((doc_id, score))
            
        return filtered_results
            
    def boolean_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Εελτιωμένη Boolean αναζήτηση με καλύτερο χειρισμό τελεστών και fuzzy matching.
        """
        try:
            # Προεπεξεργασία του ερωτήματος
            query = query.replace('&&', ' AND ').replace('||', ' OR ').replace('!', ' NOT ')
            tokens = query.split()
            
            # Στοίβες για τελεστές και αποτελέσματα
            results_stack = []
            operators_stack = []
            
            i = 0
            while i < len(tokens):
                token = tokens[i]
                
                if token in ('AND', 'OR', 'NOT'):
                    # Χειρισμός προτεραιότητας τελεστών
                    while (operators_stack and 
                          operators_stack[-1] != '(' and 
                          self._get_operator_precedence(operators_stack[-1]) >= self._get_operator_precedence(token)):
                        self._apply_operator(operators_stack, results_stack)
                    operators_stack.append(token)
                elif token == '(':
                    operators_stack.append(token)
                elif token == ')':
                    while operators_stack and operators_stack[-1] != '(':
                        self._apply_operator(operators_stack, results_stack)
                    if operators_stack and operators_stack[-1] == '(':
                        operators_stack.pop()
                else:
                    # Εύρεση εγγράφων για τον τρέχοντα όρο
                    term_docs = self._get_matching_docs(token)
                    results_stack.append(term_docs)
                
                i += 1
            
            # Εφαρμογή εναπομείναντων τελεστών
            while operators_stack:
                self._apply_operator(operators_stack, results_stack)
            
            # Μετατροπή του τελικού συνόλου σε λίστα με σκορ
            if not results_stack:
                return []
                
            final_docs = results_stack[0]
            return [(doc_id, 1.0) for doc_id in final_docs]
            
        except Exception as e:
            self.logger.error(f"Error in boolean search: {str(e)}")
            return []
            
    def _get_operator_precedence(self, op: str) -> int:
        """Επιστρέφει την προτεραιότητα του τελεστή."""
        precedence = {
            'NOT': 3,
            'AND': 2,
            'OR': 1
        }
        return precedence.get(op, 0)
        
    def _apply_operator(self, operators_stack: List[str], results_stack: List[Set[str]]):
        """Εφαρμόζει τον τελεστή στα αποτελέσματα."""
        if not operators_stack:
            return
            
        op = operators_stack.pop()
        if op == 'NOT':
            if results_stack:
                docs = results_stack.pop()
                all_docs = set(self.articles.keys())
                results_stack.append(all_docs - docs)
        else:  # AND ή OR
            if len(results_stack) >= 2:
                docs2 = results_stack.pop()
                docs1 = results_stack.pop()
                if op == 'AND':
                    results_stack.append(docs1 & docs2)
                else:  # OR
                    results_stack.append(docs1 | docs2)
                    
    def _get_matching_docs(self, term: str) -> Set[str]:
        """Βρίσκει τα έγγραφα που ταιριάζουν με τον όρο, συμπεριλαμβανομένου fuzzy matching."""
        matching_docs = set()
        
        # Επεξεργασία του όρου
        processed_term = self.text_processor.clean_text(term)
        term_tokens = self.text_processor.tokenize_and_remove_stopwords(processed_term)
        
        for token in term_tokens:
            # Ακριβές ταίριασμα
            if token in self.index:
                matching_docs.update(self.index[token].keys())
            
            # Fuzzy matching με βελτιωμένο threshold
            similar_terms = self._find_similar_terms(token, threshold=0.8)
            for similar_term in similar_terms:
                if similar_term in self.index:
                    matching_docs.update(self.index[similar_term].keys())
                    
            # Έλεγχος για ταίριασμα στον τίτλο (με μεγαλύτερο βάρος)
            for doc_id in self.articles:
                if token.lower() in self.articles[doc_id]['title'].lower():
                    matching_docs.add(doc_id)
                    
        return matching_docs
        
    def _find_similar_terms(self, term: str, threshold: float = 0.8) -> Set[str]:
        """Εύρεση παρόμοιων όρων με βάση την απόσταση Levenshtein."""
        similar_terms = set()
        
        # Έλεγχος όλων των όρων στο ευρετήριο
        for index_term in self.index:
            # Υπολογισμός ομοιότητας με βάση την απόσταση Levenshtein
            similarity = self._levenshtein_similarity(term, index_term)
            if similarity >= threshold:
                similar_terms.add(index_term)
                
        return similar_terms
        
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Υπολογισμός ομοιότητας με βάση την απόσταση Levenshtein."""
        if len(s1) < len(s2):
            return self._levenshtein_similarity(s2, s1)
            
        if len(s2) == 0:
            return 0.0
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        # Κανονικοποίηση της απόστασης στο διάστημα [0,1]
        max_length = max(len(s1), len(s2))
        distance = previous_row[-1]
        similarity = 1 - (distance / max_length)
        
        return similarity
        
    def _calculate_boolean_score(self, doc_id: str, query: str) -> float:
        """Υπολογισμός σκορ για boolean αναζήτηση με βάση την ποιότητα του ταιριάσματος."""
        try:
            # Επεξεργασία του ερωτήματος και του εγγράφου
            processed_query = self.text_processor.clean_text(query)
            query_terms = set(self.text_processor.tokenize_and_remove_stopwords(processed_query))
            
            article = self.articles[doc_id]
            doc_text = article['title'] + ' ' + article.get('text', '')
            processed_doc = self.text_processor.clean_text(doc_text)
            doc_terms = set(self.text_processor.tokenize_and_remove_stopwords(processed_doc))
            
            # Υπολογισμός διαφόρων μετρικών
            exact_matches = len(query_terms & doc_terms)
            fuzzy_matches = sum(1 for qt in query_terms 
                              for dt in doc_terms 
                              if self._levenshtein_similarity(qt, dt) >= 0.8)
            
            title_boost = 1.5 if any(qt in article['title'].lower() for qt in query_terms) else 1.0
            
            # Συνδυασμός μετρικών σε τελικό σκορ
            score = (exact_matches + 0.5 * fuzzy_matches) * title_boost
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error calculating boolean score: {str(e)}")
            return 0.0
            
    def vsm_search(self, query: str) -> List[Tuple[str, float]]:
        """Εκτέλεση Vector Space Model αναζήτησης με βελτιωμένο TF-IDF και query expansion."""
        try:
            # Επεξεργασία του ερωτήματος
            processed_query = self.text_processor.clean_text(query)
            query_terms = self.text_processor.tokenize_and_remove_stopwords(processed_query)
            
            # Query expansion
            expanded_terms = self._expand_query(query_terms)
            logger.info(f"Αρχικοί όροι: {query_terms}")
            logger.info(f"Εμπλουτισμένοι όροι: {expanded_terms}")
            
            # Υπολογισμός διανύσματος ερωτήματος με TF-IDF
            query_vector = self._compute_query_vector(expanded_terms)
            
            # Υπολογισμός ομοιότητας με όλα τα έγγραφα
            scores = []
            for doc_id, doc_vector in self.document_vectors.items():
                # Κανονικοποίηση διανύσματος εγγράφου
                normalized_doc_vector = self._normalize_vector(doc_vector)
                
                # Υπολογισμός ομοιότητας
                score = self._cosine_similarity(query_vector, normalized_doc_vector)
                
                if score > 0:  # Φιλτράρισμα μη σχετικών αποτελεσμάτων
                    # Πρόσθετο boost για ακριβή ταιριάσματα στον τίτλο
                    title_boost = self._calculate_title_match_score(query, doc_id)
                    final_score = score * (1 + 0.5 * title_boost)
                    scores.append((doc_id, float(final_score)))
            
            return sorted(scores, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in VSM search: {str(e)}")
            return []
            
    def _expand_query(self, terms: List[str]) -> List[str]:
        """Εμπλουτισμός ερωτήματος με σχετικούς όρους."""
        expanded_terms = list(terms)
        
        for term in terms:
            # Προσθήκη παρόμοιων όρων με βάση Levenshtein
            similar_terms = self._find_similar_terms(term, threshold=0.8)
            expanded_terms.extend(list(similar_terms))
            
            # Προσθήκη συχνά συνεμφανιζόμενων όρων
            cooccurring_terms = self._find_cooccurring_terms(term)
            expanded_terms.extend(cooccurring_terms)
        
        # Αφαίρεση διπλότυπων και διατήρηση μοναδικών όρων
        return list(dict.fromkeys(expanded_terms))
        
    def _find_cooccurring_terms(self, term: str, max_terms: int = 3) -> List[str]:
        """Εύρεση όρων που συχνά εμφανίζονται μαζί με τον δοθέντα όρο."""
        cooccurrence = defaultdict(int)
        
        if term in self.index:
            # Για κάθε έγγραφο που περιέχει τον όρο
            for doc_id in self.index[term].keys():
                if doc_id in self.document_vectors:
                    # Βρες τους όρους με τη μεγαλύτερη TF-IDF τιμή
                    doc_terms = sorted(
                        self.document_vectors[doc_id].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    # Πρόσθεσε τους top όρους στο dictionary
                    for t, _ in doc_terms[:10]:
                        if t != term:
                            cooccurrence[t] += 1
        
        # Επέστρεψε τους όρους με τη μεγαλύτερη συχνότητα συνεμφάνισης
        return [t for t, _ in sorted(
            cooccurrence.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_terms]]
        
    def _compute_query_vector(self, terms: List[str]) -> Dict[str, float]:
        """Υπολογισμός διανύσματος ερωτήματος με TF-IDF."""
        # Υπολογισμός term frequency (TF)
        tf = defaultdict(float)
        for term in terms:
            tf[term] += 1
        
        # Κανονικοποίηση TF και πολλαπλασιασμός με IDF
        vector = {}
        max_tf = max(tf.values()) if tf else 1
        
        for term, freq in tf.items():
            if term in self.idf:
                # Κανονικοποιημένο TF * IDF
                vector[term] = (0.5 + 0.5 * freq / max_tf) * self.idf[term]
        
        return self._normalize_vector(vector)
        
    def _normalize_vector(self, vector: Dict[str, float]) -> Dict[str, float]:
        """Κανονικοποίηση διανύσματος σε μοναδιαίο μήκος."""
        try:
            magnitude = np.sqrt(sum(x * x for x in vector.values()))
            if magnitude > 0:
                return {term: weight/magnitude for term, weight in vector.items()}
            return vector
        except Exception as e:
            logger.error(f"Error in vector normalization: {str(e)}")
            return vector
            
    def bm25_search(self, query: str) -> List[Tuple[str, float]]:
        """Εκτέλεση BM25 αναζήτησης με βελτιωμένες παραμέτρους και term proximity."""
        try:
            # Βελτιστοποιημένες παράμετροι BM25
            k1 = 1.2  # Παράμετρος term frequency saturation
            b = 0.75  # Παράμετρος μήκους εγγράφου
            
            # Επεξεργασία του ερωτήματος
            processed_query = self.text_processor.clean_text(query)
            query_terms = self.text_processor.tokenize_and_remove_stopwords(processed_query)
            
            if not query_terms:
                return []
            
            # Υπολογισμός term frequencies και μήκους εγγράφων
            doc_term_freqs = {}
            doc_lengths = {}
            total_docs = len(self.articles)
            
            # Υπολογισμός document frequencies
            doc_freqs = defaultdict(int)
            for doc_id, article in self.articles.items():
                text = article['title'] + ' ' + article.get('text', '')
                processed_text = self.text_processor.clean_text(text)
                tokens = self.text_processor.tokenize_and_remove_stopwords(processed_text)
                
                # Υπολογισμός term frequencies για το έγγραφο
                term_freqs = defaultdict(int)
                for token in tokens:
                    term_freqs[token] += 1
                
                # Ενημέρωση document frequencies
                doc_terms = set(tokens)
                for term in doc_terms:
                    doc_freqs[term] += 1
                
                doc_term_freqs[doc_id] = term_freqs
                doc_lengths[doc_id] = len(tokens)
            
            # Υπολογισμός μέσου μήκους εγγράφων
            avg_doc_len = np.mean(list(doc_lengths.values()))
            
            # Υπολογισμός BM25 IDF
            idfs = {}
            for term in query_terms:
                df = doc_freqs.get(term, 0)
                if df > 0:
                    idfs[term] = np.log((total_docs - df + 0.5) / (df + 0.5))
                else:
                    idfs[term] = 0
            
            # Υπολογισμός BM25 σκορ για κάθε έγγραφο
            scores = defaultdict(float)
            for doc_id, term_freqs in doc_term_freqs.items():
                doc_len = doc_lengths[doc_id]
                doc_score = 0
                
                for term in query_terms:
                    if term in term_freqs and term in idfs:
                        tf = term_freqs[term]
                        idf = idfs[term]
                        
                        # BM25 term frequency component
                        tf_component = ((k1 + 1) * tf) / (k1 * (1 - b + b * doc_len / avg_doc_len) + tf)
                        
                        # Πρόσθετο βάρος για όρους στον τίτλο
                        title_boost = 1.0
                        if term in self.articles[doc_id]['title'].lower():
                            title_boost = 1.5
                        
                        doc_score += idf * tf_component * title_boost
                
                if doc_score > 0:
                    # Υπολογισμός proximity bonus
                    proximity_score = self._calculate_term_proximity(query_terms, doc_id)
                    
                    # Συνδυασμός σκορ
                    final_score = doc_score * (1 + 0.1 * proximity_score)
                    scores[doc_id] = float(final_score)
            
            # Ταξινόμηση και επιστροφή αποτελεσμάτων
            return sorted([(doc_id, score) for doc_id, score in scores.items()],
                        key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            return []
            
    def _calculate_term_proximity(self, query_terms: List[str], doc_id: str) -> float:
        """Υπολογισμός proximity score με βάση την απόσταση μεταξύ των όρων του ερωτήματος."""
        try:
            article = self.articles[doc_id]
            text = article['title'] + ' ' + article.get('text', '')
            processed_text = self.text_processor.clean_text(text)
            doc_terms = self.text_processor.tokenize_and_remove_stopwords(processed_text)
            
            # Εύρεση θέσεων των όρων του ερωτήματος στο κείμενο
            term_positions = defaultdict(list)
            for i, term in enumerate(doc_terms):
                if term in query_terms:
                    term_positions[term].append(i)
            
            if len(term_positions) < 2:
                return 0.0
            
            # Υπολογισμός ελάχιστης απόστασης μεταξύ όρων
            min_distance = float('inf')
            terms_found = list(term_positions.keys())
            
            for i in range(len(terms_found)):
                for j in range(i + 1, len(terms_found)):
                    term1_positions = term_positions[terms_found[i]]
                    term2_positions = term_positions[terms_found[j]]
                    
                    for pos1 in term1_positions:
                        for pos2 in term2_positions:
                            distance = abs(pos1 - pos2)
                            min_distance = min(min_distance, distance)
            
            # Μετατροπή απόστασης σε σκορ (όσο μικρότερη η απόσταση, τόσο μεγαλύτερο το σκορ)
            if min_distance == float('inf'):
                return 0.0
            return 1.0 / (1.0 + min_distance)
            
        except Exception as e:
            logger.error(f"Error calculating term proximity: {str(e)}")
            return 0.0
            
    def _calculate_exact_phrase_matches(self, query: str, doc_id: str) -> float:
        """Υπολογισμός σκορ για ακριβή ταιριάσματα φράσεων."""
        try:
            article = self.articles[doc_id]
            text = article['title'] + ' ' + article.get('text', '')
            
            # Καθαρισμός και προετοιμασία κειμένων
            processed_query = self.text_processor.clean_text(query)
            processed_text = self.text_processor.clean_text(text)
            
            # Εύρεση ακριβών ταιριασμάτων
            query_phrases = self._get_ngrams(processed_query.split(), 2)  # Διγράμματα
            text_phrases = self._get_ngrams(processed_text.split(), 2)
            
            # Υπολογισμός ποσοστού ταιριασμάτων
            matches = len(set(query_phrases) & set(text_phrases))
            if not query_phrases:
                return 0.0
                
            return float(matches) / len(query_phrases)
            
        except Exception as e:
            logger.error(f"Error calculating exact phrase matches: {str(e)}")
            return 0.0
            
    def _get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Δημιουργία n-grams από λίστα tokens."""
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            
    def apply_learning_to_rank(self, query: str, initial_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Εφαρμογή Learning to Rank με πολλαπλά χαρακτηριστικά και δυναμική προσαρμογή."""
        try:
            # Αρχικοποίηση βαρών για τα χαρακτηριστικά
            weights = {
                'initial_score': 0.25,      # Αρχικό σκορ από τη μέθοδο αναζήτησης
                'pagerank': 0.15,           # PageRank σκορ
                'title_match': 0.20,        # Ταίριασμα τίτλου
                'category_match': 0.15,     # Ταίριασμα κατηγορίας
                'length_score': 0.10,       # Σκορ μήκους άρθρου
                'link_score': 0.10,         # Σκορ συνδέσμων
                'freshness': 0.05           # Χρονική εγγύτητα
            }
            
            # Επεξεργασία ερωτήματος για εξαγωγή κατηγοριών
            query_categories = self._extract_categories_from_query(query)
            
            final_scores = []
            for doc_id, initial_score in initial_results:
                if doc_id not in self.articles:
                    continue
                
                # Υπολογισμός επιμέρους σκορ
                feature_scores = {
                    'initial_score': initial_score,
                    'pagerank': self._get_normalized_pagerank(doc_id),
                    'title_match': self._calculate_title_match_score(query, doc_id),
                    'category_match': self._calculate_category_match(doc_id, query_categories),
                    'length_score': self._calculate_length_score(doc_id),
                    'link_score': self._calculate_link_score(doc_id),
                    'freshness': self._calculate_freshness_score(doc_id)
                }
                
                # Συνδυασμός σκορ με βάρη
                final_score = sum(weights[feature] * feature_scores[feature] 
                                for feature in weights)
                
                # Κανονικοποίηση τελικού σκορ
                if not np.isnan(final_score):
                    final_scores.append((doc_id, float(final_score)))
            
            # Ταξινόμηση αποτελεσμάτων
            return sorted(final_scores, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in learning to rank: {str(e)}")
            return initial_results
            
    def _extract_categories_from_query(self, query: str) -> Set[str]:
        """Εξαγωγή πιθανών κατηγοριών από το ερώτημα."""
        categories = set()
        # Λίστα με όλες τις διαθέσιμες κατηγορίες
        all_categories = {'Επιστήμη', 'Ιστορία', 'Τέχνη', 'Φιλοσοφία', 'Τεχνολογία'}
        
        # Καθαρισμός και tokenization του ερωτήματος
        processed_query = self.text_processor.clean_text(query.lower())
        query_terms = set(self.text_processor.tokenize_and_remove_stopwords(processed_query))
        
        # Εύρεση κατηγοριών που σχετίζονται με τους όρους του ερωτήματος
        for category in all_categories:
            category_terms = set(self.text_processor.tokenize_and_remove_stopwords(
                self.text_processor.clean_text(category.lower())
            ))
            if category_terms & query_terms:
                categories.add(category)
        
        return categories
        
    def _get_normalized_pagerank(self, doc_id: str) -> float:
        """Επιστροφή κανονικοποιημένου PageRank σκορ."""
        try:
            score = self.pagerank_scores.get(doc_id, 0)
            if score == 0:
                return 0.0
            
            # Κανονικοποίηση στο διάστημα [0, 1]
            max_score = max(self.pagerank_scores.values())
            return score / max_score if max_score > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error in normalized pagerank: {str(e)}")
            return 0.0
            
    def _calculate_category_match(self, doc_id: str, query_categories: Set[str]) -> float:
        """Υπολογισμός βαθμού ταιριάσματος κατηγοριών."""
        try:
            article = self.articles.get(doc_id, {})
            doc_categories = set(article.get('categories', []))
            
            if not query_categories or not doc_categories:
                return 0.0
            
            # Υπολογισμός Jaccard similarity μεταξύ των συνόλων κατηγοριών
            intersection = len(query_categories & doc_categories)
            union = len(query_categories | doc_categories)
            
            return float(intersection) / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error in category match: {str(e)}")
            return 0.0
            
    def _calculate_length_score(self, doc_id: str) -> float:
        """Υπολογισμός σκορ με βάση το μήκος του άρθρου."""
        try:
            article = self.articles.get(doc_id, {})
            text = article.get('text', '')
            
            # Υπολογισμός μήκους σε λέξεις
            word_count = len(text.split())
            
            # Κανονικοποίηση με λογαριθμική κλίμακα
            # Προτιμούμε άρθρα μεταξύ 500 και 5000 λέξεων
            if word_count == 0:
                return 0.0
            
            log_count = np.log10(word_count)
            optimal_min = np.log10(500)
            optimal_max = np.log10(5000)
            
            if log_count < optimal_min:
                return log_count / optimal_min
            elif log_count > optimal_max:
                return optimal_max / log_count
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error in length score: {str(e)}")
            return 0.0
            
    def _calculate_link_score(self, doc_id: str) -> float:
        """Υπολογισμός σκορ με βάση τους συνδέσμους του άρθρου."""
        try:
            article = self.articles.get(doc_id, {})
            outgoing_links = len(article.get('links', []))
            incoming_links = sum(1 for _, other_article in self.articles.items()
                               if doc_id in other_article.get('links', []))
            
            # Συνδυασμός εισερχόμενων και εξερχόμενων συνδέσμων
            total_links = outgoing_links + incoming_links
            if total_links == 0:
                return 0.0
            
            # Κανονικοποίηση με λογαριθμική κλίμακα
            return min(1.0, np.log10(1 + total_links) / np.log10(100))
            
        except Exception as e:
            logger.error(f"Error in link score: {str(e)}")
            return 0.0
            
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