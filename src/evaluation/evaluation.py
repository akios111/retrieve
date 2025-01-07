import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np
from src.search.search_engine import SearchEngine

class SearchEvaluator:
    def __init__(self):
        self.setup_logging()
        self.search_engine = SearchEngine()
        
        # Εκτύπωση των διαθέσιμων άρθρων
        print("\nΔιαθέσιμα άρθρα στη βάση δεδομένων:")
        print("-" * 80)
        for title in sorted(self.search_engine.articles.keys()):
            print(f"- {title}")
        print("-" * 80 + "\n")
            
        self.test_queries = self.load_test_queries()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_test_queries(self) -> List[Dict]:
        """Φόρτωση των test queries με βελτιωμένα relevant docs."""
        queries = [
            # Boolean queries
            {
                "query": "φιλοσοφία && επιστήμη",
                "method": "boolean",
                "relevant_docs": ["Δυτική φιλοσοφία", "Ιστορία της επιστήμης", "Επιστήμη", "Φιλοσοφία", "Φιλοσοφία της Τεχνητής Νοημοσύνης"]
            },
            {
                "query": "ιστορία && (τέχνη || πολιτισμός)",
                "method": "boolean",
                "relevant_docs": ["Ιστορία της τέχνης", "Πστορία της επιστήμης", "Ιστορία του κόσμου", "Τέχνες"]
            },
            {
                "query": "μαθηματικά && (φυσική || χημεία)",
                "method": "boolean",
                "relevant_docs": ["Μαθηματικά", "Φυσική", "Χημεία", "Ελληνικά μαθηματικά", "Αρχαία Αιγυπτιακά μαθηματικά"]
            },
            {
                "query": "αρχιτεκτονική && τέχνη",
                "method": "boolean",
                "relevant_docs": ["Αρχιτεκτονική", "Νεοελληνική αρχιτεκτονική", "Ισλαμική αρχιτεκτονική", "Τέχνες"]
            },
            {
                "query": "μουσική && τέχνη",
                "method": "boolean",
                "relevant_docs": ["Μουσική", "Τέχνες", "Μουσικολογία"]
            },

            # VSM queries
            {
                "query": "τεχνητή νοημοσύνη",
                "method": "vsm",
                "relevant_docs": ["Φιλοσοφία της Τεχνητής Νοημοσύνης", "Επιστήμη δεδομένων", "Πληροφορική"]
            },
            {
                "query": "βιολογία και επιστήμη",
                "method": "vsm",
                "relevant_docs": ["Βιολογία", "Επιστήμη", "Βιοτεχνολογία", "Βιολόγος"]
            },
            {
                "query": "ιατρική",
                "method": "vsm",
                "relevant_docs": ["Ιατρική", "Αρχαία ελληνική ιατρική", "Ιατρική ειδικότητα", "Ιατρική εξέταση"]
            },
            {
                "query": "θέατρο",
                "method": "vsm",
                "relevant_docs": ["Θέατρο", "Θέατρο και αναπηρία", "Θέατρο κωφών", "Θέατρο της επινόησης"]
            },
            {
                "query": "πληροφορική",
                "method": "vsm",
                "relevant_docs": ["Πληροφορική", "Επιστήμη δεδομένων", "Διδακτική της πληροφορικής"]
            },

            # BM25 queries
            {
                "query": "χημεία",
                "method": "bm25",
                "relevant_docs": ["Χημεία", "Χημειοπληροφορική", "Χημική αντίδραση", "Η κεντρική επιστήμη"]
            },
            {
                "query": "λογοτεχνία",
                "method": "bm25",
                "relevant_docs": ["Λογοτεχνία", "Λογοτεχνία Εσπεράντο", "Λογοτεχνία για νέους ενήλικες"]
            },
            {
                "query": "βιοτεχνολογία",
                "method": "bm25",
                "relevant_docs": ["Βιοτεχνολογία", "Βιολογία", "Βιοϊατρική τεχνολογία"]
            },
            {
                "query": "αρχιτεκτονική",
                "method": "bm25",
                "relevant_docs": ["Αρχιτεκτονική", "Νεοελληνική αρχιτεκτονική", "Ισλαμική αρχιτεκτονική"]
            },
            {
                "query": "ιστορία",
                "method": "bm25",
                "relevant_docs": ["Ιστορία", "Ιστορία του κόσμου", "Ιστορία της τέχνης", "Ιστορία της επιστήμης"]
            }
        ]
        return queries
        
    def calculate_precision(self, retrieved: Set[str], relevant: Set[str]) -> float:
        """Υπολογισμός precision με βελτιωμένη μερική αντιστοίχιση."""
        if not retrieved:
            return 0.0
        
        matches = 0
        for doc in retrieved:
            is_relevant, _ = self.is_relevant_match(doc, relevant)
            if is_relevant:
                matches += 1
        
        return matches / len(retrieved)
        
    def calculate_recall(self, retrieved: Set[str], relevant: Set[str]) -> float:
        """Υπολογισμός recall με βελτιωμένη μερική αντιστοίχιση."""
        if not relevant:
            return 0.0
        
        matches = 0
        for rel in relevant:
            # Έλεγχος για μερική αντιστοίχιση με οποιοδήποτε retrieved doc
            for doc in retrieved:
                is_relevant, _ = self.is_relevant_match(rel, {doc})
                if is_relevant:
                    matches += 1
                    break
        
        return matches / len(relevant)
        
    def calculate_f1(self, precision: float, recall: float) -> float:
        """Υπολογισμός F1-score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
        
    def calculate_ndcg(self, retrieved: List[Tuple[str, float]], relevant: Set[str], k: int = None) -> float:
        """Υπολογισμός NDCG με βελτιωμένη graded relevance."""
        if not retrieved:
            return 0.0
            
        if k is None:
            k = len(retrieved)
            
        dcg = 0.0
        idcg = 0.0
        
        # Υπολογισμός DCG με graded relevance
        for i, (doc, score) in enumerate(retrieved[:k], 1):
            # Υπολογισμός relevance score με βελτιωμένη μερική αντιστοίχιση
            _, similarity = self.is_relevant_match(doc, relevant, threshold=0.0)
            
            # Κλιμακωτή relevance με βάση την ομοιότητα
            if similarity >= 0.9:  # Σχεδόν τέλεια αντιστοίχιση
                rel = 2
            elif similarity >= 0.7:  # Πολύ καλή αντιστοίχιση
                rel = 1
            else:  # Χαμηλή ομοιότητα
                rel = 0
                
            dcg += (2 ** rel - 1) / np.log2(i + 1)
            
        # Υπολογισμός IDCG
        ideal_ranking = [2] * len(relevant) + [0] * (k - len(relevant))
        for i, rel in enumerate(ideal_ranking[:k], 1):
            idcg += (2 ** rel - 1) / np.log2(i + 1)
            
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
        
    def calculate_map(self, retrieved: List[Tuple[str, float]], relevant: Set[str]) -> float:
        """Υπολογισμός MAP με βελτιωμένη μερική αντιστοίχιση."""
        if not retrieved or not relevant:
            return 0.0
            
        precision_sum = 0.0
        relevant_found = 0
        
        for i, (doc, _) in enumerate(retrieved, 1):
            # Έλεγχος για μερική αντιστοίχιση με βελτιωμένη μέθοδο
            is_relevant, similarity = self.is_relevant_match(doc, relevant)
            
            if is_relevant:
                relevant_found += 1
                precision_at_k = relevant_found / i
                # Προσθήκη βάρους με βάση την ποιότητα της αντιστοίχισης
                precision_sum += precision_at_k * similarity
                
        return precision_sum / len(relevant)
        
    def calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Υπολογισμός ομοιότητας μεταξύ δύο strings με μεγαλύτερη ανοχή σε μερικές αντιστοιχίσεις.
        """
        if not str1 or not str2:
            return 0.0
            
        # Μετατροπή σε πεζά και αφαίρεση ειδικών χαρακτήρων
        str1 = ''.join(c.lower() for c in str1 if c.isalnum() or c.isspace())
        str2 = ''.join(c.lower() for c in str2 if c.isalnum() or c.isspace())
        
        # 1. Exact match
        if str1 == str2:
            return 1.0
            
        # 2. Περιέχει το ένα το άλλο
        if str1 in str2 or str2 in str1:
            return 0.9
            
        # 3. Token-based similarity με μεγαλύτερη βαρύτητα
        tokens1 = set(str1.split())
        tokens2 = set(str2.split())
        token_sim = len(tokens1 & tokens2) / max(len(tokens1 | tokens2), 1)
        
        if token_sim > 0.5:  # Αν έχουμε καλή αντιστοίχιση tokens
            return 0.7 + (0.3 * token_sim)
            
        # 4. N-gram similarity για τα υπόλοιπα
        def get_ngrams(s: str, n: int) -> set:
            return set(s[i:i+n] for i in range(len(s)-n+1))
            
        # Bigrams και trigrams με προσαρμοσμένα βάρη
        bigrams1 = get_ngrams(str1, 2)
        bigrams2 = get_ngrams(str2, 2)
        bigram_sim = len(bigrams1 & bigrams2) / max(len(bigrams1 | bigrams2), 1)
        
        trigrams1 = get_ngrams(str1, 3)
        trigrams2 = get_ngrams(str2, 3)
        trigram_sim = len(trigrams1 & trigrams2) / max(len(trigrams1 | trigrams2), 1)
        
        # Συνδυασμός με προσαρμοσμένα βάρη
        return max(0.6 * bigram_sim + 0.4 * trigram_sim, token_sim)

    def is_relevant_match(self, doc: str, relevant: Set[str], threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Βελτιωμένη μέθοδος για έλεγχο αν ένα document είναι relevant.
        Χρησιμοποιεί πολλαπλές μετρικές και επιστρέφει τόσο boolean όσο και similarity score.
        """
        max_similarity = 0.0
        
        for rel_doc in relevant:
            # Βασική string similarity με μεγαλύτερη ανοχή
            str_sim = self.calculate_string_similarity(doc, rel_doc)
            
            # Semantic similarity με βάση κοινές λέξεις
            sem_sim = self.calculate_semantic_similarity(doc, rel_doc)
            
            # Συνδυασμός των similarity scores με προσαρμοσμένα βάρη
            combined_sim = 0.8 * str_sim + 0.2 * sem_sim
            max_similarity = max(max_similarity, combined_sim)
        
        return max_similarity >= threshold, max_similarity

    def calculate_semantic_similarity(self, str1: str, str2: str) -> float:
        """
        Υπολογισμός semantic similarity μεταξύ δύο strings.
        Χρησιμοποιεί απλοποιημένη προσέγγιση βασισμένη σε κοινές λέξεις και συνώνυμα.
        """
        # Μετατροπή σε λέξεις
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        # Κοινές λέξεις
        common_words = words1 & words2
        
        # Απλό semantic score βασισμένο στην επικάλυψη λέξεων
        if not words1 or not words2:
            return 0.0
        
        return len(common_words) / max(len(words1), len(words2))

    def evaluate_query(self, query_info: Dict) -> Dict:
        """
        Βελτιωμένη μέθοδος αξιολόγησης ενός query.
        """
        query = query_info["query"]
        method = query_info["method"]
        relevant_docs = set(query_info["relevant_docs"])
        
        # Εκτέλεση αναζήτησης
        try:
            results = self.search_engine.search(query, method=method)
            retrieved_docs = [(doc, score) for doc, score in results]
            retrieved_set = {doc for doc, _ in retrieved_docs}
            
            # Υπολογισμός μετρικών
            precision = self.calculate_precision(retrieved_set, relevant_docs)
            recall = self.calculate_recall(retrieved_set, relevant_docs)
            f1 = self.calculate_f1(precision, recall)
            ndcg = self.calculate_ndcg(retrieved_docs, relevant_docs)
            map_score = self.calculate_map(retrieved_docs, relevant_docs)
            
            self.logger.info(f"Αποτελέσματα που βρέθηκαν: {[doc for doc, _ in retrieved_docs]}")
            
            # Έλεγχος για relevant docs στη βάση
            found_relevant = set()
            for rel_doc in relevant_docs:
                for doc in self.search_engine.articles.keys():
                    if self.is_relevant_match(rel_doc, {doc})[0]:
                        found_relevant.add(rel_doc)
                        break
            
            if found_relevant:
                self.logger.info(f"Relevant docs που βρέθηκαν στη βάση: {list(found_relevant)}")
            else:
                self.logger.warning("Κανένα από τα relevant docs δεν βρέθηκε στη βάση")
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ndcg": ndcg,
                "map": map_score,
                "num_results": len(retrieved_docs),
                "found_relevant": list(found_relevant)
            }
            
        except Exception as e:
            self.logger.error(f"Σφάλμα κατά την αξιολόγηση του query '{query}': {str(e)}")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ndcg": 0.0,
                "map": 0.0,
                "num_results": 0,
                "found_relevant": []
            }

    def _get_zero_metrics(self) -> Dict[str, float]:
        """Επιστροφή μηδενικών μετρικών."""
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ndcg": 0.0,
            "map": 0.0,
            "num_results": 0,
            "num_relevant_found": 0
        }
        
    def print_evaluation_results(self, results: Dict[str, Dict[str, float]]):
        """Εκτύπωση των αποτελεσμάτων αξιολόγησης με βελτιωμένη μορφοποίηση."""
        print("\nΑποτελέσματα Αξιολόγησης:")
        print("-" * 80)
        
        # Ομαδοποίηση ανά μέθοδο
        method_results = defaultdict(list)
        for query, metrics in results.items():
            if query != "overall":
                method = next((q["method"] for q in self.test_queries if q["query"] == query), None)
                if method:
                    method_results[method].append((query, metrics))
        
        # Εκτύπωση αποτελεσμάτων ανά μέθοδο
        for method in ["boolean", "vsm", "bm25"]:
            print(f"\n{method.upper()} Queries:")
            print("-" * 40)
            
            method_metrics = defaultdict(float)
            num_queries = len(method_results[method])
            
            for query, metrics in method_results[method]:
                print(f"\nQuery: {query}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1-score: {metrics['f1']:.4f}")
                print(f"NDCG: {metrics['ndcg']:.4f}")
                print(f"MAP: {metrics['map']:.4f}")
                print(f"Αριθμός αποτελεσμάτων: {metrics['num_results']}")
                
                # Συγκέντρωση μετρικών
                for metric in ['precision', 'recall', 'f1', 'ndcg', 'map']:
                    method_metrics[metric] += metrics[metric]
            
            if num_queries > 0:
                print(f"\nΜέσες τιμές για {method.upper()}:")
                print("-" * 30)
                for metric in ['precision', 'recall', 'f1', 'ndcg', 'map']:
                    avg_value = method_metrics[metric] / num_queries
                    print(f"Μέσο {metric}: {avg_value:.4f}")
        
        print("\nΣυνολικά Αποτελέσματα:")
        print("-" * 80)
        overall = results["overall"]
        print(f"Μέση Precision: {overall['precision']:.4f}")
        print(f"Μέση Recall: {overall['recall']:.4f}")
        print(f"Μέσο F1-score: {overall['f1']:.4f}")
        print(f"Μέσο NDCG: {overall['ndcg']:.4f}")
        print(f"Μέσο MAP: {overall['map']:.4f}")

    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Βελτιωμένη αξιολόγηση όλων των test queries."""
        results = {}
        overall_metrics = defaultdict(float)
        method_metrics = defaultdict(lambda: defaultdict(float))
        method_counts = defaultdict(int)
        
        for query in self.test_queries:
            query_results = self.evaluate_query(query)
            results[query["query"]] = query_results
            
            # Συγκέντρωση συνολικών μετρικών
            for metric, value in query_results.items():
                if metric not in ['num_results', 'found_relevant']:
                    overall_metrics[metric] += value
                    method_metrics[query["method"]][metric] += value
            
            method_counts[query["method"]] += 1
        
        # Υπολογισμός μέσων τιμών
        num_queries = len(self.test_queries)
        for metric in overall_metrics:
            overall_metrics[metric] /= num_queries
        
        # Υπολογισμός μέσων τιμών ανά μέθοδο
        method_averages = {}
        for method in method_metrics:
            method_averages[method] = {}
            for metric in method_metrics[method]:
                method_averages[method][metric] = method_metrics[method][metric] / method_counts[method]
        
        results["overall"] = dict(overall_metrics)
        results["method_averages"] = method_averages
        
        return results
        
def main():
    evaluator = SearchEvaluator()
    results = evaluator.evaluate_all()
    evaluator.print_evaluation_results(results)
    
if __name__ == "__main__":
    main() 