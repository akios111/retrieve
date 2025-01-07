import json
import os
import re
import string
from typing import List, Dict, Set, Optional
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import logging
from pathlib import Path
import sys
from datetime import datetime
from collections import defaultdict

# Προσθήκη του parent directory στο path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.greek_spell_checker import GreekSpellChecker
from src.preprocessing.greek_embeddings import GreekWordEmbeddings
from src.preprocessing.greek_ner import GreekNER
from src.preprocessing.greek_sentiment import GreekSentimentAnalyzer

class TextPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        
        # Load Greek stopwords
        self.stopwords = set(stopwords.words('greek'))
        
        # Initialize components with local model path
        model_path = os.path.join('models', 'greek-bert')
        self.spell_checker = GreekSpellChecker(model_path=model_path)
        self.embeddings = GreekWordEmbeddings(model_path=model_path)
        self.ner = GreekNER(model_path=model_path)
        self.sentiment = GreekSentimentAnalyzer(model_path=model_path)
        
    def tokenize_greek_text(self, text: str) -> List[str]:
        """Εξειδικευμένο tokenization για ελληνικό κείμενο."""
        # Χειρισμός συντομογραφιών
        text = re.sub(r'(κ\.|π\.χ\.|μ\.Χ\.|π\.Χ\.|κ\.λπ\.|κ\.ά\.|δηλ\.|βλ\.)', r'\1 ', text)
        
        # Εύρεση ελληνικών λέξεων
        pattern = r'[Α-Ωα-ωίϊΐόάέύϋΰήώ]+(?:-[Α-Ωα-ωίϊΐόάέύϋΰήώ]+)*'
        tokens = re.findall(pattern, text)
        return tokens
        
    def normalize_greek_text(self, text: str) -> str:
        """Εκτεταμένη κανονικοποίηση ελληνικού κειμένου."""
        # Βασική κανονικοποίηση
        text = text.lower()
        
        # Επιπλέον κανονικοποιήσεις
        replacements = {
            'αι': 'ε',
            'ει': 'ι',
            'οι': 'ι',
            'υι': 'ι',
            'ου': 'υ',
            'αυ': 'αβ',
            'ευ': 'εβ'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
        
    def validate_greek_text(self, text: str) -> bool:
        """Επικύρωση εγκυρότητας ελληνικού κειμένου."""
        if not text:
            return False
            
        # Έλεγχος για ελάχιστο ποσοστό ελληνικών χαρακτήρων
        greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03FF')
        return greek_chars / len(text) > 0.6
        
    def update_stopwords(self, corpus: List[str], frequency_threshold: float = 0.8):
        """Δυναμική ενημέρωση stopwords βάσει συχνότητας εμφάνισης."""
        word_freq = defaultdict(int)
        total_docs = len(corpus)
        
        for doc in corpus:
            words = set(self.tokenize_greek_text(doc))
            for word in words:
                word_freq[word] += 1
        
        # Προσθήκη λέξεων που εμφανίζονται πολύ συχνά
        frequent_words = {word for word, freq in word_freq.items() 
                         if freq/total_docs > frequency_threshold}
        self.stopwords.update(frequent_words)
        
    def clean_text(self, text: str) -> str:
        """Καθαρισμός κειμένου."""
        # Αφαίρεση HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Αφαίρεση URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Αφαίρεση ειδικών χαρακτήρων
        text = re.sub(r'[^\w\s\u0370-\u03FF]', ' ', text)
        
        # Κανονικοποίηση
        text = self.normalize_greek_text(text)
        
        return text
        
    def process_article(self, article: Dict) -> Optional[Dict]:
        """Ασφαλής επεξεργασία με χειρισμό σφαλμάτων."""
        try:
            if not self.validate_greek_text(article['text']):
                self.logger.warning(f"Το άρθρο {article['title']} δεν περιέχει έγκυρο ελληνικό κείμενο")
                return None
                
            # Καθαρισμός κειμένου
            cleaned_text = self.clean_text(article['text'])
            
            # Tokenization
            tokens = self.tokenize_greek_text(cleaned_text)
            
            # Διόρθωση ορθογραφίας
            tokens = self.spell_checker.check_text(tokens)
            
            # Αφαίρεση stopwords
            tokens = [t for t in tokens if t not in self.stopwords]
            
            # Εύρεση συνωνύμων
            expanded_tokens = []
            for token in tokens:
                expanded_tokens.append(token)
                expanded_tokens.extend(self.embeddings.find_similar_words(token))
            
            # Προσθήκη μεταδεδομένων
            processed = {
                'title': article['title'],
                'tokens': expanded_tokens,
                'metadata': {
                    'processing_date': datetime.now().isoformat(),
                    'word_count': len(tokens),
                    'entities': self.ner.extract_entities(article['text']),
                    'sentiment': self.sentiment.analyze_sentiment(article['text']),
                    'language_stats': {
                        'unique_words': len(set(tokens)),
                        'avg_word_length': sum(len(w) for w in tokens)/len(tokens) if tokens else 0
                    }
                }
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Σφάλμα κατά την επεξεργασία του άρθρου {article['title']}: {str(e)}")
            return None
    
    def load_articles(self, data_dir: str = 'data') -> List[Dict]:
        """Φόρτωση όλων των άρθρων από τα JSON αρχεία."""
        articles = []
        data_path = Path(data_dir)
        
        for json_file in data_path.glob('wikipedia_articles_*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    category_articles = json.load(f)
                    articles.extend(category_articles)
                    self.logger.info(f'Φορτώθηκαν {len(category_articles)} άρθρα από {json_file.name}')
            except Exception as e:
                self.logger.error(f'Σφάλμα κατά τη φόρτωση του {json_file}: {str(e)}')
                
        return articles
    
    def process_all_articles(self, articles: List[Dict]) -> List[Dict]:
        """Επεξεργασία όλων των άρθρων."""
        processed_articles = []
        
        for article in articles:
            processed_article = self.process_article(article)
            if processed_article:
                processed_articles.append(processed_article)
                
        self.logger.info(f'Επεξεργάστηκαν επιτυχώς {len(processed_articles)} άρθρα')
        return processed_articles
    
    def save_processed_articles(self, processed_articles: List[Dict], output_file: str = 'data/processed_articles.json'):
        """Αποθήκευση των επεξεργασμένων άρθρων."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_articles, f, ensure_ascii=False, indent=2)
            self.logger.info(f'Αποθηκεύτηκαν {len(processed_articles)} επεξεργασμένα άρθρα στο {output_file}')
        except Exception as e:
            self.logger.error(f'Σφάλμα κατά την αποθήκευση των επεξεργασμένων άρθρων: {str(e)}') 
    
    def tokenize_and_remove_stopwords(self, text: str) -> List[str]:
        """Tokenization και αφαίρεση stopwords."""
        # Tokenization
        tokens = self.tokenize_greek_text(text)
        
        # Αφαίρεση stopwords
        tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens
        
    def expand_tokens_with_synonyms(self, tokens: List[str]) -> List[str]:
        """Επέκταση των tokens με συνώνυμα."""
        expanded_tokens = []
        for token in tokens:
            # Προσθήκη του αρχικού token
            expanded_tokens.append(token)
            # Προσθήκη συνωνύμων από το word embeddings model
            similar_words = self.embeddings.find_similar_words(token)
            expanded_tokens.extend(similar_words)
        return expanded_tokens 