import json
import math
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import logging
from pathlib import Path
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import mmh3  # MurmurHash3 για αποδοτικό hashing

class InvertedIndex:
    def __init__(self, use_compression: bool = True):
        self.setup_logging()
        self.index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.document_lengths: Dict[str, int] = {}
        self.document_vectors: Dict[str, Dict[str, float]] = {}
        self.idf: Dict[str, float] = {}
        self.total_documents = 0
        self.use_compression = use_compression
        self.term_mapping = {}  # Για συμπίεση των terms
        self.next_term_id = 0
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _compress_term(self, term: str) -> int:
        """Συμπίεση των terms σε integers για εξοικονόμηση μνήμης."""
        if term not in self.term_mapping:
            self.term_mapping[term] = self.next_term_id
            self.next_term_id += 1
        return self.term_mapping[term]
        
    def _decompress_term(self, term_id: int) -> str:
        """Αποσυμπίεση των terms."""
        for term, tid in self.term_mapping.items():
            if tid == term_id:
                return term
        return ""
        
    def _compress_positions(self, positions: List[int]) -> bytes:
        """Συμπίεση των θέσεων με delta encoding."""
        if not positions:
            return b''
        deltas = []
        prev = 0
        for pos in positions:
            deltas.append(pos - prev)
            prev = pos
        return pickle.dumps(deltas)
        
    def _decompress_positions(self, compressed: bytes) -> List[int]:
        """Αποσυμπίεση των θέσεων."""
        if not compressed:
            return []
        deltas = pickle.loads(compressed)
        positions = []
        current = 0
        for delta in deltas:
            current += delta
            positions.append(current)
        return positions

    def load_processed_articles(self, file_path: str = 'data/processed_articles.json') -> List[Dict]:
        """Φόρτωση των επεξεργασμένων άρθρων."""
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                articles = json.load(f)
            self.logger.info(f'Φορτώθηκαν {len(articles)} επεξεργασμένα άρθρα')
            return articles
        except Exception as e:
            self.logger.error(f'Σφάλμα κατά τη φόρτωση των επεξεργασμένων άρθρων: {str(e)}')
            return []

    def _process_article(self, article: Dict) -> Tuple[str, Dict[str, List[int]], int]:
        """Επεξεργασία ενός άρθρου για παράλληλη εκτέλεση."""
        doc_id = article.get('title', '')
        tokens = article.get('tokens', [])  # Χρησιμοποιούμε το κλειδί 'tokens' αντί για 'lemmatized_tokens'
        doc_length = len(tokens)
        
        # Καταγραφή των θέσεων για κάθε token
        term_positions = defaultdict(list)
        for position, token in enumerate(tokens):
            term_positions[token].append(position)
            
        return doc_id, term_positions, doc_length

    def build_index(self, articles: List[Dict]):
        """Δημιουργία του ανεστραμμένου ευρετηρίου με παράλληλη επεξεργασία."""
        self.total_documents = len(articles)
        document_frequencies = defaultdict(int)
        
        # Παράλληλη επεξεργασία των άρθρων
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._process_article, articles))
            
        # Συγχώνευση των αποτελεσμάτων
        for doc_id, term_positions, doc_length in results:
            self.document_lengths[doc_id] = doc_length
            
            for term, positions in term_positions.items():
                if self.use_compression:
                    term_id = self._compress_term(term)
                    compressed_positions = self._compress_positions(positions)
                    self.index[term_id][doc_id] = compressed_positions
                else:
                    self.index[term][doc_id] = positions
                document_frequencies[term] += 1
                
        # Υπολογισμός IDF και vectors
        self._compute_idf_and_vectors(articles, document_frequencies)
        
        self.logger.info(f'Δημιουργήθηκε ευρετήριο με {len(self.index)} όρους')
        
    def _compute_idf_and_vectors(self, articles: List[Dict], document_frequencies: Dict[str, int]):
        """Υπολογισμός IDF και document vectors."""
        # Υπολογισμός IDF
        for term, df in document_frequencies.items():
            self.idf[term] = math.log(self.total_documents / df)
            
        # Υπολογισμός TF-IDF vectors με παράλληλη επεξεργασία
        def process_doc_vector(article):
            doc_id = article.get('title', '')
            tokens = article.get('tokens', [])  # Χρησιμοποιούμε το κλειδί 'tokens' αντί για 'lemmatized_tokens'
            
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
                
            doc_vector = {}
            for term, tf in term_freq.items():
                if term in self.idf:  # Έλεγχος για την περίπτωση που ο όρος δεν υπάρχει στο IDF
                    tf_idf = (1 + math.log(tf)) * self.idf[term]
                    doc_vector[term] = tf_idf
                    
            # Κανονικοποίηση
            magnitude = math.sqrt(sum(score ** 2 for score in doc_vector.values()))
            if magnitude > 0:
                for term in doc_vector:
                    doc_vector[term] /= magnitude
                    
            return doc_id, doc_vector
            
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_doc_vector, articles))
            
        for doc_id, doc_vector in results:
            self.document_vectors[doc_id] = doc_vector

    def save_index(self, output_dir: str = 'data'):
        """Αποθήκευση του ευρετηρίου με συμπίεση."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Μετατροπή του defaultdict σε κανονικό dict
            index_dict = {k: dict(v) for k, v in self.index.items()}
            
            # Αποθήκευση του ευρετηρίου με pickle για καλύτερη συμπίεση
            index_file = output_path / 'inverted_index.pkl'
            with open(index_file, 'wb') as f:
                pickle.dump({
                    'index': index_dict,
                    'term_mapping': self.term_mapping,
                    'next_term_id': self.next_term_id,
                    'use_compression': self.use_compression
                }, f)
                
            # Αποθήκευση των υπόλοιπων δεδομένων
            metadata_file = output_path / 'index_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'document_lengths': self.document_lengths,
                    'document_vectors': self.document_vectors,
                    'idf': self.idf,
                    'total_documents': self.total_documents
                }, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f'Το ευρετήριο αποθηκεύτηκε επιτυχώς στο {output_dir}')
            
        except Exception as e:
            self.logger.error(f'Σφάλμα κατά την αποθήκευση του ευρετηρίου: {str(e)}')
            
    def load_index(self, input_dir: str = 'data'):
        """Φόρτωση του ευρετηρίου."""
        try:
            input_path = Path(input_dir)
            
            # Φόρτωση του ευρετηρίου
            index_file = input_path / 'inverted_index.pkl'
            with open(index_file, 'rb') as f:
                data = pickle.load(f)
                self.index = data['index']
                self.term_mapping = data['term_mapping']
                self.next_term_id = data['next_term_id']
                self.use_compression = data['use_compression']
                
            # Φόρτωση των μεταδεδομένων
            metadata_file = input_path / 'index_metadata.json'
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.document_lengths = metadata['document_lengths']
                self.document_vectors = metadata['document_vectors']
                self.idf = metadata['idf']
                self.total_documents = metadata['total_documents']
                
            self.logger.info(f'Το ευρετήριο φορτώθηκε επιτυχώς από το {input_dir}')
            
        except Exception as e:
            self.logger.error(f'Σφάλμα κατά τη φόρτωση του ευρετηρίου: {str(e)}')

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Αναζήτηση με βάση το query και επιστροφή των top-k αποτελεσμάτων."""
        # Προεπεξεργασία του query
        query_terms = query.lower().split()
        
        # Υπολογισμός TF-IDF για το query
        query_vector = {}
        term_freq = defaultdict(int)
        for term in query_terms:
            term_freq[term] += 1
            
        # Βάρη για διαφορετικούς τύπους ταιριάσματος
        EXACT_MATCH_WEIGHT = 1.0
        SIMILAR_MATCH_WEIGHT = 0.6
        PARTIAL_MATCH_WEIGHT = 0.4
            
        for term, tf in term_freq.items():
            if term in self.idf:  # Ακριβές ταίριασμα
                query_vector[term] = (1 + math.log(tf)) * self.idf[term] * EXACT_MATCH_WEIGHT
            else:
                # Αναζήτηση παρόμοιων όρων
                similar_terms = self._find_similar_terms(term)
                for similar_term, similarity in similar_terms:
                    if similar_term in self.idf:
                        if similarity > 0.8:  # Πολύ παρόμοιος όρος
                            weight = SIMILAR_MATCH_WEIGHT
                        else:  # Μερικώς παρόμοιος όρος
                            weight = PARTIAL_MATCH_WEIGHT
                        query_vector[similar_term] = (1 + math.log(tf)) * self.idf[similar_term] * weight * similarity
                
        # Κανονικοποίηση του query vector
        magnitude = math.sqrt(sum(score ** 2 for score in query_vector.values()))
        if magnitude > 0:
            for term in query_vector:
                query_vector[term] /= magnitude
                
        # Υπολογισμός ομοιότητας με όλα τα έγγραφα
        scores = []
        for doc_id, doc_vector in self.document_vectors.items():
            score = self._compute_similarity(query_vector, doc_vector)
            if score > 0.01:  # Ελάχιστο όριο σχετικότητας
                # Προσθήκη bonus για τίτλους που περιέχουν όρους του query
                title_bonus = self._compute_title_bonus(doc_id, query_terms)
                final_score = score + title_bonus
                scores.append((doc_id, final_score))
            
        # Επιστροφή των top-k αποτελεσμάτων
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def _find_similar_terms(self, term: str) -> List[Tuple[str, float]]:
        """Εύρεση παρόμοιων όρων στο ευρετήριο με τους βαθμούς ομοιότητας."""
        similar_terms = []
        term_lower = term.lower()
        
        for index_term in self.idf.keys():
            index_term_lower = index_term.lower()
            similarity = self._compute_similarity_score(term_lower, index_term_lower)
            
            # Προσθήκη όρων με ικανοποιητική ομοιότητα
            if similarity > 0.5:
                similar_terms.append((index_term, similarity))
                
        # Ταξινόμηση με βάση την ομοιότητα και επιστροφή των top-5
        return sorted(similar_terms, key=lambda x: x[1], reverse=True)[:5]

    def _compute_title_bonus(self, doc_id: str, query_terms: List[str]) -> float:
        """Υπολογισμός bonus βαθμολογίας για τίτλους που περιέχουν όρους του query."""
        title_lower = doc_id.lower()
        bonus = 0.0
        
        for term in query_terms:
            if term in title_lower:
                bonus += 0.2  # Προσθήκη bonus για κάθε όρο του query που βρίσκεται στον τίτλο
                
        return min(bonus, 0.5)  # Μέγιστο bonus 0.5

    def _compute_similarity_score(self, term1: str, term2: str) -> float:
        """Υπολογισμός ομοιότητας μεταξύ δύο όρων."""
        if not term1 or not term2:
            return 0.0
            
        # Υπολογισμός ομοιότητας χαρακτήρων
        chars1 = set(term1)
        chars2 = set(term2)
        char_intersection = len(chars1.intersection(chars2))
        char_union = len(chars1.union(chars2))
        char_similarity = char_intersection / char_union if char_union > 0 else 0.0
        
        # Υπολογισμός ομοιότητας προθεμάτων
        min_len = min(len(term1), len(term2))
        prefix_len = 0
        for i in range(min_len):
            if term1[i] == term2[i]:
                prefix_len += 1
            else:
                break
        prefix_similarity = prefix_len / min_len if min_len > 0 else 0.0
        
        # Συνδυασμός των δύο μετρικών
        return 0.7 * char_similarity + 0.3 * prefix_similarity

    def _compute_similarity(self, query_vector: Dict[str, float], doc_vector: Dict[str, float]) -> float:
        """Υπολογισμός ομοιότητας μεταξύ query και document vectors."""
        similarity = 0.0
        for term, query_weight in query_vector.items():
            if term in doc_vector:
                similarity += query_weight * doc_vector[term]
        return similarity

    def get_term_frequency(self, term: str) -> Dict[str, int]:
        """Επιστρέφει τη συχνότητα εμφάνισης ενός όρου σε κάθε έγγραφο."""
        if self.use_compression:
            term_id = self._compress_term(term)
            if term_id in self.index:
                return {doc_id: len(self._decompress_positions(positions))
                       for doc_id, positions in self.index[term_id].items()}
        else:
            if term in self.index:
                return {doc_id: len(positions)
                       for doc_id, positions in self.index[term].items()}
        return {}

    def get_document_terms(self, doc_id: str) -> Set[str]:
        """Επιστρέφει όλους τους όρους που εμφανίζονται σε ένα έγγραφο."""
        terms = set()
        for term, postings in self.index.items():
            if doc_id in postings:
                if self.use_compression:
                    terms.add(self._decompress_term(term))
                else:
                    terms.add(term)
        return terms

    def get_memory_usage(self) -> Dict[str, int]:
        """Επιστρέφει στατιστικά χρήσης μνήμης."""
        import sys
        
        memory_stats = {
            'index_size': sys.getsizeof(self.index),
            'document_lengths_size': sys.getsizeof(self.document_lengths),
            'document_vectors_size': sys.getsizeof(self.document_vectors),
            'idf_size': sys.getsizeof(self.idf),
            'term_mapping_size': sys.getsizeof(self.term_mapping)
        }
        
        total_size = sum(memory_stats.values())
        memory_stats['total_size'] = total_size
        
        return memory_stats

def main():
    # Δημιουργία του inverted index με συμπίεση
    indexer = InvertedIndex(use_compression=True)
    
    # Φόρτωση των επεξεργασμένων άρθρων
    articles = indexer.load_processed_articles()
    
    if articles:
        # Δημιουργία του ευρετηρίου
        indexer.build_index(articles)
        
        # Εκτύπωση στατιστικών μνήμης
        memory_stats = indexer.get_memory_usage()
        print("\nΣτατιστικά χρήσης μνήμης:")
        for key, value in memory_stats.items():
            print(f"{key}: {value / 1024 / 1024:.2f} MB")
        
        # Αποθήκευση του ευρετηρίου
        indexer.save_index()

if __name__ == "__main__":
    main() 