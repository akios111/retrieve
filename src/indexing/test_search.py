from inverted_index import InvertedIndex
import spacy

def test_search():
    # Φόρτωση του spaCy model για lemmatization
    nlp = spacy.load('el_core_news_lg')
    
    # Φόρτωση του ευρετηρίου
    indexer = InvertedIndex(use_compression=True)
    indexer.load_index()
    
    # Λίστα queries για δοκιμή
    test_queries = [
        "Τεχνητή Νοημοσύνη",
        "Αρχαία Ελλάδα",
        "Φυσική Επιστήμη",
        "Μαθηματικά θεωρήματα",
        "Ιστορία της Τέχνης"
    ]
    
    # Εκτέλεση αναζητήσεων
    for query in test_queries:
        print(f"\nΑναζήτηση για: {query}")
        print("-" * 50)
        
        # Lemmatization του query
        doc = nlp(query)
        lemmatized_query = " ".join([token.lemma_ for token in doc])
        
        results = indexer.search(lemmatized_query, k=5)
        
        if results:
            print("Top 5 αποτελέσματα:")
            for doc_id, score in results:
                if score > 0:  # Εμφάνιση μόνο των σχετικών αποτελεσμάτων
                    print(f"- {doc_id} (score: {score:.4f})")
                    
                    # Εμφάνιση των όρων του εγγράφου που ταιριάζουν με το query
                    doc_terms = indexer.get_document_terms(doc_id)
                    query_terms = set(lemmatized_query.lower().split())
                    matching_terms = doc_terms.intersection(query_terms)
                    if matching_terms:
                        print(f"  Όροι που ταιριάζουν: {', '.join(matching_terms)}")
                    
                    # Εμφάνιση συχνοτήτων των όρων του query στο έγγραφο
                    for term in query_terms:
                        freq = indexer.get_term_frequency(term)
                        if doc_id in freq:
                            print(f"  Συχνότητα '{term}': {freq[doc_id]} φορές")
        else:
            print("Δεν βρέθηκαν αποτελέσματα.")
            
        print()

if __name__ == "__main__":
    test_search() 