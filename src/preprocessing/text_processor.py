import json
import os
import re
import string
from typing import List, Dict, Set, Optional
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from pathlib import Path
import sys
from datetime import datetime
from collections import defaultdict

# Προσθήκη του parent directory στο path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.greek_synonyms import GreekSynonyms
from preprocessing.greek_spell_checker import GreekSpellChecker
from preprocessing.greek_embeddings import GreekWordEmbeddings
from preprocessing.greek_ner import GreekNER
from preprocessing.greek_sentiment import GreekSentimentAnalyzer

class TextPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        
        # Load Greek stopwords
        self.stopwords = set(stopwords.words('greek'))
        
        # Initialize components with local model path
        model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'greek-bert'))
        self.spell_checker = GreekSpellChecker(model_path=model_path)
        self.embeddings = GreekWordEmbeddings(model_path=model_path)
        self.ner = GreekNER(model_path=model_path)
        self.sentiment = GreekSentimentAnalyzer(model_path=model_path)
        
    def tokenize_greek_text(self, text: str) -> List[str]:
        """Tokenization ελληνικού κειμένου."""
        try:
            # Καθαρισμός χωρίς να αφαιρούμε τόνους
            text = text.strip()
            
            # Αφαίρεση ειδικών χαρακτήρων διατηρώντας ελληνικούς χαρακτήρες και σημεία στίξης
            text = re.sub(r'[^\u0370-\u03FF\u1F00-\u1FFF.,!;()[\]{}\s\d]', ' ', text)
            
            # Διατήρηση μόνο των απαραίτητων σημείων στίξης
            text = re.sub(r'[.,!;](?=[^\s])', ' ', text)  # Προσθήκη κενού μετά τα σημεία στίξης
            
            # Tokenization με το NLTK
            tokens = word_tokenize(text, language='greek')
            
            # Φιλτράρισμα για έγκυρα tokens (τουλάχιστον ένας ελληνικός χαρακτήρας)
            valid_tokens = []
            for token in tokens:
                if any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in token):
                    valid_tokens.append(token)
                elif token.isdigit() or token in string.punctuation:
                    valid_tokens.append(token)
            
            return valid_tokens
            
        except Exception as e:
            self.logger.error(f"Σφάλμα κατά το tokenization: {str(e)}")
            return []
            
    def normalize_greek_text(self, text: str) -> str:
        """Εκτεταμένη κανονικοποίηση ελληνικού κειμένου."""
        try:
            # Μετατροπή σε πεζά
            text = text.lower().strip()
            
            # Κανονικοποίηση τόνων και διαλυτικών
            accent_map = {
                'ά': 'α', 'έ': 'ε', 'ή': 'η', 'ί': 'ι', 'ό': 'ο', 'ύ': 'υ', 'ώ': 'ω',
                'ἀ': 'α', 'ἐ': 'ε', 'ἠ': 'η', 'ἰ': 'ι', 'ὀ': 'ο', 'ὐ': 'υ', 'ὠ': 'ω',
                'ϊ': 'ι', 'ϋ': 'υ', 'ΐ': 'ι', 'ΰ': 'υ'
            }
            
            for accented, unaccented in accent_map.items():
                text = text.replace(accented, unaccented)
            
            # Αφαίρεση πολλαπλών κενών
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            self.logger.error(f"Σφάλμα κατά την κανονικοποίηση: {str(e)}")
            return text
            
    def validate_greek_text(self, text: str) -> bool:
        """Έλεγχος εγκυρότητας ελληνικού κειμένου."""
        if not text or not isinstance(text, str):
            return False
            
        # Μετρητής ελληνικών χαρακτήρων
        greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03FF')
        total_chars = len(text.strip())
        
        # Τουλάχιστον 30% του κειμένου πρέπει να είναι ελληνικοί χαρακτήρες
        return total_chars > 0 and (greek_chars / total_chars) >= 0.3
        
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
        """Καθαρισμός και προετοιμασία κειμένου."""
        try:
            if not self.validate_greek_text(text):
                return ""
            
            # Αρχικός καθαρισμός
            text = text.strip()
            
            # Tokenization
            tokens = self.tokenize_greek_text(text)
            
            # Κανονικοποίηση κάθε token
            normalized_tokens = [self.normalize_greek_text(token) for token in tokens]
            
            # Επανένωση των tokens
            cleaned_text = ' '.join(normalized_tokens)
            
            # Διόρθωση ορθογραφίας αν χρειάζεται
            if len(cleaned_text) > 0:
                cleaned_text = self.spell_checker.correct_text(cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Σφάλμα κατά τον καθαρισμό κειμένου: {str(e)}")
            return ""
        
    def assign_categories(self, text: str) -> List[str]:
        """Ανάθεση κατηγοριών με βάση το περιεχόμενο του άρθρου."""
        categories = []
        text = text.lower()
        
        # Λέξεις-κλειδιά για κάθε κατηγορία
        category_keywords = {
            'Επιστήμη': ['επιστήμη', 'επιστημονικ', 'έρευνα', 'πείραμα', 'εργαστήριο', 'θεωρία', 'μέθοδος', 'ανακάλυψη'],
            'Ιστορία': ['ιστορία', 'ιστορικ', 'εποχή', 'περίοδος', 'αρχαί', 'μεσαίων', 'πόλεμος', 'αυτοκράτορας', 'βασιλιάς'],
            'Τέχνη': ['τέχνη', 'καλλιτέχν', 'έργο', 'μουσείο', 'γλυπτ', 'ζωγραφ', 'αρχιτεκτονικ', 'μουσική'],
            'Φιλοσοφία': ['φιλοσοφία', 'φιλόσοφος', 'σκέψη', 'λογική', 'ηθική', 'γνώση', 'οντολογία', 'επιστημολογία'],
            'Τεχνολογία': ['τεχνολογία', 'υπολογιστ', 'μηχαν', 'συσκευ', 'εφεύρεση', 'καινοτομία', 'ψηφιακ', 'διαδίκτυο']
        }
        
        # Έλεγχος για κάθε κατηγορία
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
                
        return categories

    def process_article(self, article: Dict) -> Optional[Dict]:
        """Ασφαλής επεξεργασία με χειρισμό σφαλμάτων."""
        try:
            self.logger.info(f"Αρχή επεξεργασίας άρθρου: {article['title']}")
            
            if not self.validate_greek_text(article['text']):
                self.logger.warning(f"Το άρθρο {article['title']} δεν περιέχει έγκυρο ελληνικό κείμενο")
                return None
                
            # Καθαρισμός κειμένου - βασική επεξεργασία
            self.logger.info("Καθαρισμός κειμένου...")
            cleaned_text = self.clean_text(article['text'])
            self.logger.debug(f"Καθαρισμένο κείμενο (πρώτοι 100 χαρακτήρες): {cleaned_text[:100]}")
            
            self.logger.info("Tokenization...")
            tokens = self.tokenize_greek_text(cleaned_text)
            self.logger.debug(f"Πρώτα 10 tokens: {tokens[:10]}")
            
            # Επεξεργασία tokens
            if hasattr(self, 'spell_checker'):
                self.logger.info("Διόρθωση ορθογραφίας...")
                tokens = self.spell_checker.check_text(tokens)
            
            if hasattr(self, 'stopwords'):
                self.logger.info("Αφαίρεση stopwords...")
                tokens = [t for t in tokens if t not in self.stopwords]
            
            # Ανάθεση κατηγοριών
            self.logger.info("Ανάθεση κατηγοριών...")
            categories = self.assign_categories(cleaned_text)
            self.logger.info(f"Κατηγορίες που ανατέθηκαν: {categories}")
            
            # Βασικά μεταδεδομένα
            processed = {
                'title': article['title'],
                'text': article['text'],
                'clean_text': cleaned_text,
                'tokens': tokens,
                'categories': categories,
                'metadata': {
                    'processing_date': datetime.now().isoformat(),
                    'word_count': len(tokens),
                    'language_stats': {
                        'unique_words': len(set(tokens)),
                        'avg_word_length': sum(len(w) for w in tokens)/len(tokens) if tokens else 0
                    }
                }
            }
            
            # Προαιρετική επεξεργασία
            if hasattr(self, 'ner'):
                self.logger.info("Εξαγωγή οντοτήτων...")
                processed['metadata']['entities'] = self.ner.extract_entities(cleaned_text)
                
            if hasattr(self, 'sentiment'):
                self.logger.info("Ανάλυση συναισθήματος...")
                processed['metadata']['sentiment'] = self.sentiment.analyze_sentiment(cleaned_text)
            
            self.logger.info(f"Ολοκλήρωση επεξεργασίας άρθρου: {article['title']}")
            return processed
            
        except Exception as e:
            self.logger.error(f"Σφάλμα κατά την επεξεργασία του άρθρου {article['title']}: {str(e)}")
            return None
    
    def load_articles(self, input_file: str = 'data/articles.json') -> List[Dict]:
        """Φόρτωση των άρθρων."""
        try:
            with open(input_file, 'r', encoding='utf-8-sig') as f:
                articles = json.load(f)
            self.logger.info(f'Φορτώθηκαν {len(articles)} άρθρα από το {input_file}')
            return articles
        except Exception as e:
            self.logger.error(f'Σφάλμα κατά τη φόρτωση των άρθρων: {str(e)}')
            return []
    
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
            # Ensure the data directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save with UTF-8 encoding and BOM
            with open(output_file, 'w', encoding='utf-8-sig') as f:
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
        expanded_tokens = set(tokens)
        
        # Χρήση των tokens ως vocabulary
        vocabulary = list(set(tokens))
        
        for token in tokens:
            try:
                # Εύρεση παρόμοιων λέξεων από το vocabulary
                similar_words = self.embeddings.find_similar_words(token, vocabulary=vocabulary, n=3)
                expanded_tokens.update(word for word, score in similar_words if score > 0.7)
            except Exception as e:
                self.logger.warning(f"Error finding similar words for {token}: {str(e)}")
                continue
                
        return list(expanded_tokens)
        
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatization των tokens."""
        # Προς το παρόν, επιστρέφουμε τα tokens ως έχουν
        # TODO: Υλοποίηση πραγματικού lemmatization για ελληνικά
        return tokens

def main():
    # Δημιουργία του preprocessor
    preprocessor = TextPreprocessor()
    
    # Φόρτωση των άρθρων
    articles = preprocessor.load_articles()
    
    # Επεξεργασία των άρθρων
    processed_articles = preprocessor.process_all_articles(articles)
    
    # Αποθήκευση των επεξεργασμένων άρθρων
    preprocessor.save_processed_articles(processed_articles)

if __name__ == "__main__":
    main() 