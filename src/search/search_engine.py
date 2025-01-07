import json
import math
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import logging
from pathlib import Path
import sys
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Προσθήκη του parent directory στο path για να μπορούμε να κάνουμε import τα άλλα modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.text_processor import TextPreprocessor
from src.preprocessing.greek_embeddings import GreekWordEmbeddings
from src.preprocessing.greek_ner import GreekNER
from src.preprocessing.greek_sentiment import GreekSentimentAnalyzer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class SearchEngine:
    def __init__(self):
        self.setup_logging()
        self.preprocessor = TextPreprocessor()
        self.load_word_embeddings()
        self.ner = GreekNER()
        self.sentiment_analyzer = GreekSentimentAnalyzer()
        self.index = {}
        self.ngram_index = {}
        self.document_vectors = {}
        self.idf = {}
        self.articles = {}
        self.pagerank_scores = {}
        
        # Παράμετροι για Learning to Rank
        self.feature_weights = {
            'vsm_score': 0.3,
            'bm25_score': 0.3,
            'pagerank': 0.1,
            'title_match': 0.15,
            'semantic_sim': 0.1,
            'quality': 0.05
        }
        
        self.user_history = defaultdict(lambda: defaultdict(float))
        self.load_resources()
        
        # Υπολογισμός μέσου μήκους εγγράφων για BM25
        total_length = sum(len(article['lemmatized_tokens']) for article in self.articles.values())
        self.avg_doc_length = total_length / len(self.articles) if self.articles else 0
        
        self.calculate_pagerank()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_resources(self, data_dir: str = 'data'):
        """Φόρτωση όλων των απαραίτητων πόρων."""
        try:
            data_path = Path(data_dir)
            
            # Φόρτωση του ευρετηρίου
            with open(data_path / 'inverted_index.json', 'r', encoding='utf-8') as f:
                self.index = json.load(f)
                
            # Φόρτωση των document vectors
            with open(data_path / 'document_vectors.json', 'r', encoding='utf-8') as f:
                self.document_vectors = json.load(f)
                
            # Φόρτωση των IDF τιμών
            with open(data_path / 'idf_values.json', 'r', encoding='utf-8') as f:
                self.idf = json.load(f)
                
            # Φόρτωση των άρθρων
            with open(data_path / 'processed_articles.json', 'r', encoding='utf-8') as f:
                articles = json.load(f)
                self.articles = {article['title']: article for article in articles}
                
            self.logger.info('Όλοι οι πόροι φορτώθηκαν επιτυχώς')
            
        except Exception as e:
            self.logger.error(f'Σφάλμα κατά τη φόρτωση των πόρων: {str(e)}')

    def calculate_pagerank(self, damping_factor: float = 0.85, iterations: int = 20):
        """Υπολογισμός PageRank scores για τα άρθρα."""
        # Δημιουργία γράφου συνδέσεων με βάση τα κοινά tokens
        graph = defaultdict(set)
        for doc1 in self.articles:
            tokens1 = set(self.articles[doc1]['lemmatized_tokens'])
            for doc2 in self.articles:
                if doc1 != doc2:
                    tokens2 = set(self.articles[doc2]['lemmatized_tokens'])
                    if len(tokens1 & tokens2) > 5:  # Τουλάχιστον 5 κοινά tokens
                        graph[doc1].add(doc2)
        
        # Αρχικοποίηση PageRank scores
        N = len(self.articles)
        scores = {doc: 1/N for doc in self.articles}
        
        # Επαναληπτικός υπολογισμός
        for _ in range(iterations):
            new_scores = {}
            for doc in self.articles:
                # Άθροισμα των scores των εισερχόμενων συνδέσεων
                incoming_score = sum(scores[in_doc] / len(graph[in_doc]) 
                                  for in_doc in self.articles 
                                  if doc in graph[in_doc])
                
                # Υπολογισμός νέου score
                new_scores[doc] = (1 - damping_factor) / N + damping_factor * incoming_score
            
            # Ενημέρωση scores
            scores = new_scores
        
        self.pagerank_scores = scores

    def parse_boolean_query(self, query: str) -> List[List[str]]:
        """
        Βελτιωμένη ανάλυση Boolean ερωτήματος.
        Υποστηρίζει παρενθέσεις και προτεραιότητα τελεστών.
        """
        # Προεπεξεργασία του query
        query = query.replace('&&', ' AND ').replace('||', ' OR ').replace('!', ' NOT ')
        
        # Διαχωρισμός σε tokens
        tokens = query.split()
        
        # Μετατροπή σε DNF
        dnf_terms = []
        current_and_terms = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == 'OR':
                if current_and_terms:
                    dnf_terms.append(current_and_terms)
                    current_and_terms = []
            elif token == 'AND':
                pass
            elif token == 'NOT':
                i += 1
                if i < len(tokens):
                    current_and_terms.append('!' + tokens[i])
            else:
                current_and_terms.append(token)
            
            i += 1
        
        if current_and_terms:
            dnf_terms.append(current_and_terms)
            
        return dnf_terms

    def get_documents_for_term(self, term: str) -> Set[str]:
        """Επιστρέφει το σύνολο των εγγράφων που περιέχουν τον όρο."""
        # Καθαρισμός και προεπεξεργασία του όρου
        clean_term = self.preprocessor.clean_text(term)
        if not clean_term:
            return set()
            
        tokens = self.preprocessor.tokenize_and_remove_stopwords(clean_term)
        if not tokens:
            return set()
            
        # Επέκταση με συνώνυμα
        expanded_tokens = self.preprocessor.expand_tokens_with_synonyms(tokens)
        
        # Lemmatization
        lemmatized_tokens = self.preprocessor.lemmatize_tokens(expanded_tokens)
        if not lemmatized_tokens:
            return set()
            
        # Συνδυασμός αποτελεσμάτων από όλα τα tokens
        results = set()
        for token in lemmatized_tokens:
            if token in self.index:
                results.update(self.index[token].keys())
                
        return results

    def boolean_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Βελτιωμένη boolean αναζήτηση με υποστήριξη σύνθετων εκφράσεων.
        """
        try:
            # Προεπεξεργασία του query
            query = query.replace('&&', ' AND ').replace('||', ' OR ').replace('!', ' NOT ')
            query = ' '.join(query.split())  # Κανονικοποίηση κενών
            
            # Διαχωρισμός σε tokens με διατήρηση των τελεστών
            tokens = []
            current_token = ''
            for char in query:
                if char.isspace():
                    if current_token:
                        tokens.append(current_token)
                        current_token = ''
                elif char in '()':
                    if current_token:
                        tokens.append(current_token)
                        current_token = ''
                    tokens.append(char)
                else:
                    current_token += char
            if current_token:
                tokens.append(current_token)
            
            if not tokens:
                return []
            
            # Στοίβα για τα αποτελέσματα και τους τελεστές
            results_stack = []
            operator_stack = []
            
            i = 0
            while i < len(tokens):
                token = tokens[i]
                
                if token in ('AND', 'OR'):
                    while (operator_stack and operator_stack[-1] != '(' and 
                          (operator_stack[-1] == 'AND' or token == 'OR')):
                        self._apply_operator(results_stack, operator_stack.pop())
                    operator_stack.append(token)
                elif token == 'NOT':
                    # Επόμενο token είναι ο όρος προς άρνηση
                    i += 1
                    if i < len(tokens):
                        term_docs = self.get_documents_for_term(tokens[i])
                        # Άρνηση: όλα τα έγγραφα εκτός από αυτά που περιέχουν τον όρο
                        not_docs = set(self.articles.keys()) - term_docs
                        results_stack.append(not_docs)
                elif token == '(':
                    operator_stack.append(token)
                elif token == ')':
                    # Εκτέλεση όλων των πράξεων μέχρι την αντίστοιχη παρένθεση
                    while operator_stack and operator_stack[-1] != '(':
                        self._apply_operator(results_stack, operator_stack.pop())
                    if operator_stack and operator_stack[-1] == '(':
                        operator_stack.pop()  # Αφαίρεση της '('
                else:
                    # Κανονικός όρος αναζήτησης
                    term_docs = self.get_documents_for_term(token)
                    results_stack.append(term_docs)
                    
                    # Εκτέλεση AND αν υπάρχει στην κορυφή του operator_stack
                    while (operator_stack and operator_stack[-1] == 'AND'):
                        self._apply_operator(results_stack, operator_stack.pop())
                
                i += 1
            
            # Εκτέλεση όλων των εναπομεινάντων πράξεων
            while operator_stack:
                self._apply_operator(results_stack, operator_stack.pop())
            
            # Μετατροπή του τελικού συνόλου σε λίστα με scores
            if results_stack:
                final_docs = results_stack[0]
                return [(doc, 1.0) for doc in final_docs]
            
            return []
            
        except Exception as e:
            self.logger.error(f'Σφάλμα κατά την boolean αναζήτηση: {str(e)}')
            return []

    def _apply_operator(self, results_stack: List[Set[str]], operator: str):
        """
        Εφαρμογή boolean τελεστή σε δύο σύνολα αποτελεσμάτων.
        """
        if len(results_stack) < 2:
            return
        
        right = results_stack.pop()
        left = results_stack.pop()
        
        if operator == 'AND':
            results_stack.append(left & right)
        elif operator == 'OR':
            results_stack.append(left | right)

    def partial_match_ratio(self, s1: str, s2: str) -> float:
        """Υπολογισμός του βαθμού μερικής αντιστοίχισης μεταξύ δύο strings."""
        if not s1 or not s2:
            return 0.0
        
        # Μετατροπή σε σύνολα χαρακτήρων
        set1 = set(s1)
        set2 = set(s2)
        
        # Υπολογισμός Jaccard similarity με βελτιωμένη στάθμιση
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        # Συνδυασμός με sequence matching και n-gram similarity
        jaccard = intersection / union
        
        # Έλεγχος για κοινή ακολουθία χαρακτήρων με βελτιωμένο αλγόριθμο
        i = j = lcs = 0
        while i < len(s1) and j < len(s2):
            if s1[i] == s2[j]:
                lcs += 1
                i += 1
                j += 1
            elif len(s1) - i > len(s2) - j:
                i += 1
            else:
                j += 1
            
        sequence_ratio = 2 * lcs / (len(s1) + len(s2))
        
        # N-gram similarity (για n=2 και n=3)
        def get_ngrams(s: str, n: int) -> Set[str]:
            return set(s[i:i+n] for i in range(len(s)-n+1))
            
        bigrams1 = get_ngrams(s1, 2)
        bigrams2 = get_ngrams(s2, 2)
        trigrams1 = get_ngrams(s1, 3)
        trigrams2 = get_ngrams(s2, 3)
        
        bigram_sim = len(bigrams1 & bigrams2) / max(len(bigrams1 | bigrams2), 1)
        trigram_sim = len(trigrams1 & trigrams2) / max(len(trigrams1 | trigrams2), 1)
        
        # Συνδυασμός όλων των μετρικών με βελτιωμένα βάρη
        return 0.3 * jaccard + 0.3 * sequence_ratio + 0.2 * bigram_sim + 0.2 * trigram_sim

    def create_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Δημιουργία n-grams από μια λίστα tokens."""
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
    def index_document(self, doc_id: str, tokens: List[str]):
        """Ευρετηριοποίηση ενός εγγράφου με υποστήριξη n-grams."""
        # Ευρετηριοποίηση μεμονωμένων tokens
        for pos, token in enumerate(tokens):
            if token not in self.index:
                self.index[token] = {}
            if doc_id not in self.index[token]:
                self.index[token][doc_id] = []
            self.index[token][doc_id].append(pos)
            
        # Ευρετηριοποίηση bigrams
        bigrams = self.create_ngrams(tokens, 2)
        for pos, bigram in enumerate(bigrams):
            if bigram not in self.ngram_index:
                self.ngram_index[bigram] = {}
            if doc_id not in self.ngram_index[bigram]:
                self.ngram_index[bigram][doc_id] = []
            self.ngram_index[bigram][doc_id].append(pos)
            
    def process_query(self, query: str) -> Dict[str, float]:
        """Επεξεργασία του ερωτήματος και δημιουργία TF-IDF vector με προηγμένες λειτουργίες."""
        # Καθαρισμός και tokenization του ερωτήματος
        clean_query = self.preprocessor.clean_text(query)
        query_tokens = self.preprocessor.tokenize_and_remove_stopwords(clean_query)
        
        # Διόρθωση ορθογραφικών λαθών
        corrected_tokens = []
        for token in query_tokens:
            if not self.preprocessor.spell_checker.check_word(token):
                suggestions = self.preprocessor.spell_checker.get_suggestions(token)
                if suggestions:
                    corrected_tokens.append(suggestions[0])  # Προσθήκη της καλύτερης πρότασης
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)
        
        # Επέκταση με συνώνυμα και παρόμοιες λέξεις
        expanded_tokens = self.preprocessor.expand_tokens_with_synonyms(corrected_tokens)
        
        # Προσθήκη fuzzy matching για wildcards
        wildcard_tokens = []
        for token in expanded_tokens:
            if '*' in token:
                pattern = token.replace('*', '.*')
                matches = [term for term in self.index.keys() 
                         if re.match(f'^{pattern}$', term)]
                wildcard_tokens.extend(matches)
            else:
                wildcard_tokens.append(token)
        
        # Lemmatization
        query_tokens = self.preprocessor.lemmatize_tokens(wildcard_tokens)
        
        # Δημιουργία bigrams από το ερώτημα
        query_bigrams = self.create_ngrams(query_tokens, 2)
        
        # Υπολογισμός term frequencies για tokens και bigrams
        query_tf = defaultdict(int)
        for token in query_tokens:
            query_tf[token] += 1
        for bigram in query_bigrams:
            query_tf[bigram] += 1
            
        # Δημιουργία TF-IDF vector με βελτιωμένα βάρη
        query_vector = {}
        
        # Βελτιωμένα field weights
        title_weight = 3.0
        text_weight = 1.5
        bigram_weight = 2.0
        exact_match_weight = 2.5
        fuzzy_match_weight = 1.8
        
        # Προσθήκη term scores με fuzzy matching
        for term, tf in query_tf.items():
            if term in self.idf:  # Ακριβές ταίριασμα
                tf_idf = (1 + math.log(tf)) * self.idf[term] * exact_match_weight
                query_vector[term] = tf_idf
            else:  # Fuzzy matching
                similar_terms = []
                for index_term in self.idf.keys():
                    similarity = self.partial_match_ratio(term, index_term)
                    if similarity > 0.8:  # Υψηλή ομοιότητα
                        similar_terms.append((index_term, similarity))
                
                # Προσθήκη των καλύτερων παρόμοιων όρων
                for similar_term, similarity in sorted(similar_terms, 
                                                    key=lambda x: x[1], 
                                                    reverse=True)[:3]:
                    tf_idf = (1 + math.log(tf)) * self.idf[similar_term] * \
                            fuzzy_match_weight * similarity
                    query_vector[similar_term] = tf_idf
        
        # Κανονικοποίηση του vector
        magnitude = math.sqrt(sum(score ** 2 for score in query_vector.values()))
        if magnitude > 0:
            for term in query_vector:
                query_vector[term] /= magnitude
                
        return query_vector
        
    def vector_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Υπολογισμός ομοιότητας μεταξύ δύο vectors με βελτιστοποιημένα weights."""
        score = 0.0
        
        # Πιο ισορροπημένα field weights
        title_weight = 2.8  # Μειωμένο για καλύτερη ισορροπία
        text_weight = 1.2  # Αυξημένο για καλύτερη αξιοποίηση του κειμένου
        bigram_weight = 2.2  # Μειωμένο για λιγότερο θόρυβο
        exact_match_weight = 2.5  # Αυξημένο για καλύτερη ακρίβεια
        
        # Καταγραφή των θέσεων των terms
        term_positions = defaultdict(list)
        matched_terms = set()
        
        for term in vec1:
            if term in vec2:
                matched_terms.add(term)
                if term in self.index:
                    for doc_id in self.index[term]:
                        term_positions[doc_id].extend(self.index[term][doc_id])
                elif term in self.ngram_index:
                    for doc_id in self.ngram_index[term]:
                        term_positions[doc_id].extend(self.ngram_index[term][doc_id])
        
        # Υελτιστοποιημένος υπολογισμός term importance
        term_importance = {}
        for term in matched_terms:
            importance = 1.0
            for doc_id in term_positions:
                positions = term_positions[doc_id]
                # Αυξημένο bonus για εμφάνιση στην αρχή
                if any(pos < 30 for pos in positions):
                    importance *= 1.5
                # Μειωμένο bonus για συχνή εμφάνιση
                if len(positions) > 3:
                    importance *= (1 + math.log(len(positions)) * 0.15)
            term_importance[term] = importance
        
        for term in vec1:
            if term in vec2:
                # Υελτιστοποιημένα field weights
                if ' ' in term:  # Bigram
                    weight = bigram_weight
                elif term.isupper():  # Τίτλος
                    weight = title_weight
                else:  # Κείμενο
                    weight = text_weight
                    
                # Αυστηρότερα κριτήρια για exact match
                similarity = abs(vec1[term] - vec2[term])
                if similarity < 0.05:  # Σχεδόν τέλεια αντιστοίχιση
                    weight *= exact_match_weight
                elif similarity < 0.2:  # Πολύ καλή αντιστοίχιση
                    weight *= (exact_match_weight * 0.8)
                    
                term_score = vec1[term] * vec2[term] * weight * term_importance.get(term, 1.0)
                
                # Εελτιστοποιημένο proximity bonus
                for doc_id in term_positions:
                    positions = sorted(term_positions[doc_id])
                    if len(positions) > 1:
                        proximity_bonus = 0.0
                        for i, pos1 in enumerate(positions):
                            for j, pos2 in enumerate(positions[i+1:], i+1):
                                distance = abs(pos2 - pos1)
                                if distance <= 2:  # Άμεση γειτνίαση
                                    proximity_bonus += 0.8 * math.exp(-0.4 * distance)
                                elif distance <= 5:  # Κοντινή απόσταση
                                    proximity_bonus += 0.5 * math.exp(-0.3 * distance)
                                elif distance <= 8:  # Μεσαία απόσταση
                                    proximity_bonus += 0.3 * math.exp(-0.2 * distance)
                        term_score *= (1 + proximity_bonus)
                
                # Εελτιστοποιημένο context bonus
                if term in self.index:
                    for doc_id in self.index[term]:
                        positions = self.index[term][doc_id]
                        if any(pos < 20 for pos in positions):  # Πιο αυστηρό κριτήριο για αρχή
                            term_score *= 1.4
                        if len(positions) > 4:  # Πιο αυστηρό κριτήριο για συχνότητα
                            term_score *= (1 + math.log(len(positions)) * 0.08)
                
                score += term_score
        
        # Βελτιστοποιημένη κανονικοποίηση
        magnitude1 = math.sqrt(sum(v * v for v in vec1.values()))
        magnitude2 = math.sqrt(sum(v * v for v in vec2.values()))
        if magnitude1 > 0 and magnitude2 > 0:
            score /= (magnitude1 * magnitude2)
            
        return score

    def calculate_bm25_score(self, query_terms: List[str], doc_id: str, k1: float = 1.8, b: float = 0.75) -> float:
        """Βελτιστοποιημένος υπολογισμός του BM25 score."""
        score = 0.0
        doc_length = len(self.articles[doc_id]['lemmatized_tokens'])
        avg_doc_length = sum(len(doc['lemmatized_tokens']) for doc in self.articles.values()) / len(self.articles)
        
        # Βελτιστοποιημένα field weights
        title_weight = 3.0  # Μειωμένο για καλύτερη ισορροπία
        text_weight = 1.2  # Αυξημένο για καλύτερη αξιοποίηση κειμένου
        bigram_weight = 2.2  # Μειωμένο για λιγότερο θόρυβο
        
        # Εύρεση θέσεων των query terms
        term_positions = defaultdict(list)
        doc_tokens = self.articles[doc_id]['lemmatized_tokens']
        doc_title = self.articles[doc_id]['title'].lower()
        
        # Δημιουργία bigrams
        query_bigrams = self.create_ngrams(query_terms)
        doc_bigrams = self.create_ngrams(doc_tokens)
        
        # Καταγραφή θέσεων
        for i, token in enumerate(doc_tokens):
            if token in query_terms:
                term_positions[token].append(i)
                
        for i, bigram in enumerate(doc_bigrams):
            if bigram in query_bigrams:
                term_positions[bigram].append(i)
        
        # Υπολογισμός scores
        all_terms = query_terms + query_bigrams
        term_scores = {}
        
        for term in all_terms:
            if (term in self.index and doc_id in self.index[term]) or \
               (term in self.ngram_index and doc_id in self.ngram_index[term]):
                
                # Υελτιστοποιημένος υπολογισμός term frequency
                if ' ' in term:  # Bigram
                    positions = self.ngram_index[term].get(doc_id, [])
                    title_tf = sum(1 for pos in positions if pos < len(doc_title.split()))
                    text_tf = len(positions) - title_tf
                    weight = bigram_weight
                else:  # Μεμονωμένος όρος
                    positions = self.index[term].get(doc_id, [])
                    title_tf = sum(1 for pos in positions if pos < len(doc_title.split()))
                    text_tf = len(positions) - title_tf
                    weight = title_weight if title_tf > 0 else text_weight
                
                # Βελτιστοποιημένο weighted term frequency
                tf = (title_weight * 2.5 * title_tf + text_weight * text_tf)
                
                # Υελτιστοποιημένος υπολογισμός IDF
                if ' ' in term:
                    df = len(self.ngram_index[term])
                else:
                    df = len(self.index[term])
                idf = math.log((len(self.articles) - df + 0.5) / (df + 0.5) + 1.2)
                
                # Βελτιστοποιημένο normalized term frequency
                tf_normalized = ((k1 + 1.0) * tf) / (k1 * (1 - b + b * doc_length / avg_doc_length) + tf)
                
                # Βελτιστοποιημένο proximity bonus
                proximity_bonus = 0.0
                if term in term_positions:
                    positions = sorted(term_positions[term])
                    for i, pos1 in enumerate(positions):
                        for j, pos2 in enumerate(positions[i+1:], i+1):
                            distance = abs(pos2 - pos1)
                            if distance <= 2:
                                proximity_bonus += 0.8 * math.exp(-0.4 * distance)
                            elif distance <= 5:
                                proximity_bonus += 0.5 * math.exp(-0.3 * distance)
                            elif distance <= 8:
                                proximity_bonus += 0.3 * math.exp(-0.2 * distance)
                
                # Βελτιστοποιημένο exact match bonus
                exact_match_bonus = 1.0
                if title_tf > 0:
                    if term.lower() in doc_title:
                        exact_match_bonus = 2.5  # Μειωμένο για καλύτερη ισορροπία
                        if title_tf > 1:
                            exact_match_bonus += 0.3 * (title_tf - 1)
                    else:
                        exact_match_bonus = 1.8
                elif text_tf > 0:
                    exact_match_bonus = 1.3
                    if text_tf > 4:  # Πιο αυστηρό κριτήριο
                        exact_match_bonus += 0.15 * math.log(text_tf)
                
                # Υπολογισμός τελικού term score
                term_score = weight * idf * tf_normalized * (1 + proximity_bonus) * exact_match_bonus
                
                # Context bonus
                if any(pos < 30 for pos in positions):
                    term_score *= 1.2
                
                term_scores[term] = term_score
                score += term_score
        
        # Βελτιστοποιημένος συνδυασμός με PageRank
        pagerank_weight = 0.25  # Αυξημένο για καλύτερη ποιότητα αποτελεσμάτων
        final_score = (1 - pagerank_weight) * score + pagerank_weight * self.pagerank_scores[doc_id]
        
        # Βελτιστοποιημένο length normalization
        length_factor = 1.0
        if 150 <= doc_length <= 800:  # Πιο στοχευμένο εύρος
            length_factor = 1.15
        elif doc_length > 1500:  # Πιο αυστηρή ποινή για πολύ μεγάλα έγγραφα
            length_factor = 0.85
        final_score *= length_factor
        
        # Query-length normalization
        query_length_factor = 1 / (1 + 0.8 * math.log(len(all_terms)))
        final_score *= query_length_factor
        
        return final_score
        
    def update_feature_weights(self, click_data: List[Tuple[str, str, float]]):
        """Ενημέρωση των βαρών με βάση τα click data."""
        learning_rate = 0.01
        for query, clicked_doc, dwell_time in click_data:
            # Υπολογισμός features για το clicked document
            features = self._calculate_features(query, clicked_doc)
            
            # Υπολογισμός predicted score
            predicted_score = sum(self.feature_weights[f] * v for f, v in features.items())
            
            # Υπολογισμός actual score με βάση το dwell time
            actual_score = min(1.0, dwell_time / 60.0)  # Κανονικοποίηση στο [0,1]
            
            # Gradient descent
            error = actual_score - predicted_score
            for feature, value in features.items():
                self.feature_weights[feature] += learning_rate * error * value
            
            # Κανονικοποίηση των βαρών
            total = sum(self.feature_weights.values())
            for feature in self.feature_weights:
                self.feature_weights[feature] /= total

    def _calculate_features(self, query: str, doc_id: str) -> Dict[str, float]:
        """Υπολογισμός features για ένα document."""
        features = {}
        
        # VSM score
        query_vector = self.process_query(query)
        features['vsm_score'] = self.vector_similarity(query_vector, self.document_vectors[doc_id])
        
        # BM25 score
        query_terms = self.preprocessor.tokenize_and_remove_stopwords(query)
        features['bm25_score'] = self.calculate_bm25_score(query_terms, doc_id)
        
        # PageRank
        features['pagerank'] = self.pagerank_scores[doc_id]
        
        # Title match
        title = self.articles[doc_id]['title'].lower()
        query = query.lower()
        features['title_match'] = 1.0 if query in title else 0.0
        
        # Position bonus
        doc_tokens = self.articles[doc_id]['lemmatized_tokens']
        first_pos = float('inf')
        for term in query_terms:
            if term in self.index and doc_id in self.index[term]:
                pos = min(self.index[term][doc_id])
                first_pos = min(first_pos, pos)
        features['position'] = 1.0 / (1.0 + first_pos) if first_pos < float('inf') else 0.0
        
        # Length normalization
        doc_length = len(doc_tokens)
        avg_length = sum(len(doc['lemmatized_tokens']) for doc in self.articles.values()) / len(self.articles)
        features['length'] = math.exp(-abs(doc_length - avg_length) / avg_length)
        
        return features

    def update_user_history(self, user_id: str, doc_id: str, interaction_score: float):
        """Ενημέρωση του ιστορικού χρήστη."""
        # Exponential decay για παλιότερες αλληλεπιδράσεις
        decay = 0.95
        for doc in self.user_history[user_id]:
            self.user_history[user_id][doc] *= decay
        
        # Προσθήκη νέας αλληλεπίδρασης
        self.user_history[user_id][doc_id] = interaction_score

    def get_personalized_score(self, user_id: str, doc_id: str) -> float:
        """Υπολογισμός εξατομικευμένου score με βάση το ιστορικό."""
        if user_id not in self.user_history:
            return 0.0
            
        # Εύρεση παρόμοιων documents που έχει αλληλεπιδράσει ο χρήστης
        doc_vector = self.document_vectors[doc_id]
        similar_docs_scores = []
        
        for hist_doc, hist_score in self.user_history[user_id].items():
            if hist_doc in self.document_vectors:
                similarity = self.vector_similarity(doc_vector, self.document_vectors[hist_doc])
                similar_docs_scores.append(hist_score * similarity)
        
        if not similar_docs_scores:
            return 0.0
            
        return sum(similar_docs_scores) / len(similar_docs_scores)

    def search(self, query: str, method: str = 'vsm', k: int = 10) -> List[Tuple[str, float]]:
        """
        Βελτιωμένη μέθοδος αναζήτησης που συνδυάζει πολλαπλές τεχνικές.
        """
        try:
            if not query or not method:
                return []
            
            # Προεπεξεργασία του query
            clean_query = self.preprocessor.clean_text(query)
            if not clean_query:
                return []
            
            # Επιλογή μεθόδου αναζήτησης
            if method == 'boolean':
                results = self.boolean_search(query)
            elif method == 'vsm':
                results = self.vsm_search(query)
            elif method == 'bm25':
                results = self.bm25_search(query)
            else:
                self.logger.warning(f'Μη έγκυρη μέθοδος αναζήτησης: {method}')
                return []
            
            # Εφαρμογή Learning to Rank
            ranked_results = self.apply_learning_to_rank(query, results)
            
            # Επιστροφή των top-k αποτελεσμάτων
            return ranked_results[:k]
            
        except Exception as e:
            self.logger.error(f'Σφάλμα κατά την αναζήτηση: {str(e)}')
            return []

    def apply_learning_to_rank(self, query: str, initial_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Εφαρμογή Learning to Rank με βελτιωμένα features.
        """
        if not initial_results:
            return []
        
        ranked_results = []
        query_tokens = set(self.preprocessor.tokenize_and_remove_stopwords(query))
        
        for doc_id, initial_score in initial_results:
            if doc_id not in self.articles:
                continue
            
            # Υπολογισμός features
            features = {
                'initial_score': initial_score,
                'pagerank': self.pagerank_scores.get(doc_id, 0),
                'title_match': self.calculate_title_match_score(query, doc_id),
                'semantic_sim': self.calculate_semantic_similarity(query, doc_id),
                'freshness': self.calculate_freshness_score(doc_id),
                'quality': self.calculate_quality_score(doc_id)
            }
            
            # Συνδυασμός features με βάρη
            final_score = (
                0.3 * features['initial_score'] +
                0.2 * features['pagerank'] +
                0.2 * features['title_match'] +
                0.15 * features['semantic_sim'] +
                0.1 * features['freshness'] +
                0.05 * features['quality']
            )
            
            ranked_results.append((doc_id, final_score))
        
        # Ταξινόμηση με βάση το τελικό score
        return sorted(ranked_results, key=lambda x: x[1], reverse=True)

    def calculate_title_match_score(self, query: str, doc_id: str) -> float:
        """
        Υπολογισμός score για την αντιστοίχιση στον τίτλο με βελτιωμένη μερική αντιστοίχιση.
        """
        title = self.articles[doc_id]['title'].lower()
        query = query.lower()
        
        # Ακριβής αντιστοίχιση
        if query in title:
            return 1.0
        
        # Μερική αντιστοίχιση με n-grams
        query_ngrams = set(self.get_ngrams(query, n=3))
        title_ngrams = set(self.get_ngrams(title, n=3))
        
        if not query_ngrams or not title_ngrams:
            return 0.0
        
        # Jaccard similarity για n-grams
        similarity = len(query_ngrams & title_ngrams) / len(query_ngrams | title_ngrams)
        return similarity

    def calculate_semantic_similarity(self, query: str, doc_id: str) -> float:
        """
        Υπολογισμός semantic similarity μεταξύ query και εγγράφου.
        """
        # Προεπεξεργασία
        query_tokens = self.preprocessor.tokenize_and_remove_stopwords(query)
        doc_tokens = self.articles[doc_id]['lemmatized_tokens']
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        # Χρήση word embeddings για semantic similarity
        try:
            query_vector = np.mean([self.word_embeddings[t] for t in query_tokens if t in self.word_embeddings], axis=0)
            doc_vector = np.mean([self.word_embeddings[t] for t in doc_tokens if t in self.word_embeddings], axis=0)
            
            if query_vector.size and doc_vector.size:
                return float(np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)))
        except:
            pass
        
        # Fallback σε απλούστερη μέθοδο
        common_tokens = set(query_tokens) & set(doc_tokens)
        return len(common_tokens) / max(len(query_tokens), len(doc_tokens))

    def calculate_freshness_score(self, doc_id: str) -> float:
        """
        Υπολογισμός score για την "φρεσκάδα" του εγγράφου.
        """
        # Απλοποιημένη υλοποίηση - θα μπορούσε να βασίζεται σε πραγματικά timestamps
        return 1.0

    def calculate_quality_score(self, doc_id: str) -> float:
        """
        Υπολογισμός score για την ποιότητα του εγγράφου.
        """
        doc = self.articles[doc_id]
        
        # Παράγοντες ποιότητας
        length_score = min(1.0, len(doc['clean_text']) / 1000)  # Μέγεθος κειμένου
        structure_score = 0.8  # Θα μπορούσε να βασίζεται σε πραγματική ανάλυση δομής
        
        return (length_score + structure_score) / 2

    def get_ngrams(self, text: str, n: int) -> List[str]:
        """
        Δημιουργία n-grams από ένα κείμενο.
        """
        return [text[i:i+n] for i in range(len(text)-n+1)]

    def vsm_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Βελτιωμένη Vector Space Model αναζήτηση με semantic enrichment.
        """
        # Προεπεξεργασία query
        clean_query = self.preprocessor.clean_text(query)
        query_tokens = self.preprocessor.tokenize_and_remove_stopwords(clean_query)
        
        if not query_tokens:
            return []
        
        # Δημιουργία query vector με semantic enrichment
        query_vector = defaultdict(float)
        
        # 1. Βασικό TF-IDF
        for token in query_tokens:
            if token in self.idf:
                query_vector[token] += 1
        
        # Μετατροπή σε TF-IDF
        for term in query_vector:
            tf = 1 + math.log(query_vector[term])  # log normalization
            query_vector[term] = tf * self.idf[term]
        
        # 2. Semantic enrichment
        expanded_tokens = self.preprocessor.expand_tokens_with_synonyms(query_tokens)
        for token in expanded_tokens:
            if token in self.idf and token not in query_vector:
                query_vector[token] = 0.5 * self.idf[token]  # Μειωμένο βάρος για expanded terms
        
        # Υπολογισμός ομοιότητας με όλα τα έγγραφα
        scores = []
        for doc_id, doc_vector in self.document_vectors.items():
            # Cosine similarity
            score = self.improved_cosine_similarity(dict(query_vector), doc_vector)
            if score > 0:
                scores.append((doc_id, score))
        
        # Ταξινόμηση αποτελεσμάτων
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def improved_cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Βελτιωμένος υπολογισμός cosine similarity με positional weighting.
        """
        # Εύρεση κοινών όρων
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0
        
        # Υπολογισμός dot product με positional weighting
        dot_product = 0
        for term in common_terms:
            # Βασικό TF-IDF score
            base_score = vec1[term] * vec2[term]
            
            # Positional boost για όρους κοντά στην αρχή του εγγράφου
            if term in self.index and vec2[term] > 0:
                positions = self.index[term].get(list(vec2.keys())[0], [])
                if positions:
                    min_pos = min(positions)
                    pos_boost = 1 + (1 / (1 + min_pos * 0.01))  # Φθίνουσα συνάρτηση
                    base_score *= pos_boost
            
            dot_product += base_score
        
        # Υπολογισμός μεγεθών
        norm1 = math.sqrt(sum(val * val for val in vec1.values()))
        norm2 = math.sqrt(sum(val * val for val in vec2.values()))
        
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    def bm25_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Βελτιωμένη BM25 αναζήτηση με semantic matching και proximity boosting.
        """
        # Προεπεξεργασία query
        clean_query = self.preprocessor.clean_text(query)
        query_tokens = self.preprocessor.tokenize_and_remove_stopwords(clean_query)
        
        if not query_tokens:
            return []
        
        # Παράμετροι BM25
        k1 = 1.5  # Αυξημένο για καλύτερη απόδοση σε μεγάλα κείμενα
        b = 0.75
        k3 = 1.2  # Παράμετρος για query term frequency
        
        # Υπολογισμός BM25 scores με proximity boosting
        scores = defaultdict(float)
        query_term_freq = Counter(query_tokens)
        
        for token in query_tokens:
            if token in self.index:
                # Query term frequency weight
                qtf_weight = ((k3 + 1) * query_term_freq[token]) / (k3 + query_term_freq[token])
                
                # IDF score με smoothing
                n_docs_with_term = len(self.index[token])
                idf = math.log((len(self.articles) - n_docs_with_term + 0.5) / (n_docs_with_term + 0.5))
                
                for doc_id in self.index[token]:
                    # Term frequency στο έγγραφο
                    tf = len(self.index[token][doc_id])
                    doc_length = len(self.articles[doc_id]['lemmatized_tokens'])
                    
                    # BM25 score
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_length / self.avg_doc_length)
                    bm25_score = idf * (numerator / denominator) * qtf_weight
                    
                    # Proximity boost
                    proximity_score = self.calculate_proximity_score(token, doc_id, query_tokens)
                    
                    # Συνδυασμός scores
                    scores[doc_id] += bm25_score * (1 + 0.2 * proximity_score)
        
        # Semantic matching boost
        semantic_scores = self.calculate_semantic_scores(query_tokens)
        for doc_id in scores:
            if doc_id in semantic_scores:
                scores[doc_id] *= (1 + 0.1 * semantic_scores[doc_id])
        
        # Μετατροπή σε list και ταξινόμηση
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def calculate_proximity_score(self, token: str, doc_id: str, query_tokens: List[str]) -> float:
        """
        Υπολογισμός proximity score με βάση τις θέσεις των query terms.
        """
        if token not in self.index or doc_id not in self.index[token]:
            return 0.0
        
        # Συλλογή θέσεων για όλα τα query terms στο έγγραφο
        term_positions = defaultdict(list)
        for t in query_tokens:
            if t in self.index and doc_id in self.index[t]:
                term_positions[t] = self.index[t][doc_id]
        
        if len(term_positions) < 2:
            return 0.0
        
        # Υπολογισμός ελάχιστης απόστασης μεταξύ των terms
        min_distance = float('inf')
        for t1, pos1_list in term_positions.items():
            for t2, pos2_list in term_positions.items():
                if t1 >= t2:
                    continue
                for pos1 in pos1_list:
                    for pos2 in pos2_list:
                        distance = abs(pos1 - pos2)
                        min_distance = min(min_distance, distance)
        
        # Μετατροπή απόστασης σε score
        if min_distance == float('inf'):
            return 0.0
        return 1 / (1 + min_distance)

    def calculate_semantic_scores(self, query_tokens: List[str]) -> Dict[str, float]:
        """
        Υπολογισμός semantic similarity scores για όλα τα έγγραφα.
        """
        scores = defaultdict(float)
        
        # Χρήση word embeddings για semantic matching
        try:
            query_vectors = [self.word_embeddings[t] for t in query_tokens if t in self.word_embeddings]
            if query_vectors:
                query_vector = np.mean(query_vectors, axis=0)
                
                for doc_id, article in self.articles.items():
                    doc_tokens = article['lemmatized_tokens']
                    doc_vectors = [self.word_embeddings[t] for t in doc_tokens if t in self.word_embeddings]
                    
                    if doc_vectors:
                        doc_vector = np.mean(doc_vectors, axis=0)
                        similarity = float(np.dot(query_vector, doc_vector) / 
                                        (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)))
                        scores[doc_id] = similarity
        except:
            pass
        
        return scores

    def load_word_embeddings(self):
        """
        Βελτιωμένη φόρτωση των word embeddings με caching.
        """
        try:
            # Έλεγχος για cached embeddings
            cache_file = Path('data/cached_embeddings.npz')
            if cache_file.exists():
                self.logger.info('Φόρτωση cached word embeddings...')
                with np.load(cache_file) as data:
                    self.word_embeddings = {word: vec for word, vec in zip(data['words'], data['vectors'])}
                return

            # Φόρτωση από το αρχικό μοντέλο
            self.logger.info('Φόρτωση word embeddings από το μοντέλο...')
            embeddings = GreekWordEmbeddings().get_embeddings()
            
            # Αποθήκευση στην cache
            words = list(embeddings.keys())
            vectors = np.array([embeddings[w] for w in words])
            np.savez(cache_file, words=words, vectors=vectors)
            
            self.word_embeddings = embeddings
            self.logger.info('Word embeddings φορτώθηκαν επιτυχώς')
            
        except Exception as e:
            self.logger.error(f'Σφάλμα κατά τη φόρτωση των word embeddings: {str(e)}')
            self.word_embeddings = {}

def main():
    # Δημιουργία της μηχανής αναζήτησης
    search_engine = SearchEngine()
    
    # Παραδείγματα αναζήτησης με διάφορες μεθόδους
    queries = [
        ("φιλοσοφία && επιστήμη", "boolean"),
        ("τεχνητή || νοημοσύνη", "boolean"),
        ("ιστορία && !τέχνη", "boolean"),
        ("φιλοσοφία και επιστήμη", "vsm"),
        ("τεχνητή νοημοσύνη", "vsm"),
        ("ιστορία της τέχνης", "bm25"),
        ("τεχνολογία και καινοτομία", "bm25")
    ]
    
    # Εκτέλεση των αναζητήσεων
    for query, method in queries:
        print(f'\nΑναζήτηση για: "{query}" (μέθοδος: {method})')
        results = search_engine.search(query, method=method)
        
        if results:
            for i, (title, score) in enumerate(results, 1):
                print(f'\n{i}. {title} (score: {score:.4f})')
        else:
            print('Δεν βρέθηκαν αποτελέσματα.')

if __name__ == "__main__":
    main() 