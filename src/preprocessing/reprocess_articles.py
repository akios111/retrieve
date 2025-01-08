from text_processor import TextPreprocessor
import logging
from typing import List, Dict, Any
from pathlib import Path
import json

# Ρύθμιση logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def assign_categories(text: str) -> List[str]:
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

def main():
    try:
        # Φόρτωση των άρθρων
        data_dir = Path(__file__).parent.parent.parent / 'data'
        articles = []
        
        # Διάβασμα όλων των αρχείων JSON
        for json_file in data_dir.glob('wikipedia_articles_*.json'):
            logger.info(f"Loading articles from {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                file_articles = json.load(f)
                articles.extend(file_articles)
                
        logger.info(f"Loaded {len(articles)} articles")
        
        # Επεξεργασία των άρθρων
        text_processor = TextPreprocessor()
        processed_articles = []
        
        for article in articles:
            title = article.get('title', '')
            text = article.get('text', '')
            date = article.get('date', '')
            
            # Ανάθεση κατηγοριών
            categories = assign_categories(text)
            logger.info(f"Article '{title}' assigned categories: {categories}")
            
            processed_article = {
                'title': title,
                'text': text,
                'date': date,
                'categories': categories
            }
            
            processed_articles.append(processed_article)
            
        # Αποθήκευση των επεξεργασμένων άρθρων
        output_file = data_dir / 'processed_articles.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_articles, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(processed_articles)} processed articles to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing articles: {str(e)}")
        raise

if __name__ == "__main__":
    main() 