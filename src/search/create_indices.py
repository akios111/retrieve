import json
import math
from collections import defaultdict
from pathlib import Path
import logging
import sys
import os

# Προσθήκη του parent directory στο path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_articles(data_dir: str = 'data') -> list:
    """Φόρτωση των επεξεργασμένων άρθρων."""
    with open(Path(data_dir) / 'processed_articles.json', 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def create_inverted_index(articles: list) -> dict:
    """Δημιουργία ανεστραμμένου ευρετηρίου."""
    index = defaultdict(dict)
    
    for article in articles:
        doc_id = article['title']
        # Χρήση των tokens για την αναζήτηση
        for position, token in enumerate(article['tokens']):
            if token not in index[token]:
                index[token][doc_id] = []
            index[token][doc_id].append(position)
    
    return dict(index)

def calculate_idf(articles: list, index: dict) -> dict:
    """Υπολογισμός IDF τιμών."""
    N = len(articles)
    idf = {}
    
    for term in index:
        df = len(index[term])  # document frequency
        idf[term] = math.log(N / (df + 1))  # +1 για αποφυγή division by zero
        
    return idf

def create_document_vectors(articles: list, index: dict, idf: dict) -> dict:
    """Δημιουργία διανυσμάτων εγγράφων."""
    vectors = {}
    
    for article in articles:
        doc_id = article['title']
        term_freq = defaultdict(int)
        
        # Υπολογισμός term frequencies
        for token in article['tokens']:
            term_freq[token] += 1
        
        # Δημιουργία διανύσματος με tf-idf τιμές
        vector = {}
        for term, freq in term_freq.items():
            if term in idf:  # Μόνο όροι που υπάρχουν στο ευρετήριο
                tf = 1 + math.log(freq)  # log normalization
                vector[term] = tf * idf[term]
        
        vectors[doc_id] = vector
    
    return vectors

def main():
    logger = setup_logging()
    data_dir = 'data'
    
    try:
        # Φόρτωση άρθρων
        logger.info("Φόρτωση επεξεργασμένων άρθρων...")
        articles = load_articles(data_dir)
        
        # Δημιουργία ανεστραμμένου ευρετηρίου
        logger.info("Δημιουργία ανεστραμμένου ευρετηρίου...")
        index = create_inverted_index(articles)
        
        # Υπολογισμός IDF
        logger.info("Υπολογισμός IDF τιμών...")
        idf = calculate_idf(articles, index)
        
        # Δημιουργία διανυσμάτων εγγράφων
        logger.info("Δημιουργία διανυσμάτων εγγράφων...")
        vectors = create_document_vectors(articles, index, idf)
        
        # Αποθήκευση των αρχείων
        logger.info("Αποθήκευση αρχείων...")
        with open(Path(data_dir) / 'inverted_index.json', 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
            
        with open(Path(data_dir) / 'idf_values.json', 'w', encoding='utf-8') as f:
            json.dump(idf, f, ensure_ascii=False, indent=2)
            
        with open(Path(data_dir) / 'document_vectors.json', 'w', encoding='utf-8') as f:
            json.dump(vectors, f, ensure_ascii=False, indent=2)
            
        logger.info("Η δημιουργία των ευρετηρίων ολοκληρώθηκε επιτυχώς!")
        
    except Exception as e:
        logger.error(f"Σφάλμα κατά τη δημιουργία των ευρετηρίων: {str(e)}")

if __name__ == "__main__":
    main() 