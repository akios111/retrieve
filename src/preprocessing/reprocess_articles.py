import json
import logging
from pathlib import Path
from datetime import datetime
from text_processor import TextPreprocessor
from typing import List, Dict

# Ρύθμιση logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reprocess_articles():
    """Επανεπεξεργασία των άρθρων με τον ενημερωμένο TextPreprocessor."""
    try:
        # Αρχικοποίηση του text processor
        processor = TextPreprocessor()
        
        # Φόρτωση των αρχικών άρθρων
        data_dir = Path(__file__).parent.parent.parent / 'data'
        articles = []
        
        # Φόρτωση όλων των αρχείων άρθρων
        for file_path in data_dir.glob('wikipedia_articles_*.json'):
            if file_path.name != 'wikipedia_articles.json':  # Αποφυγή του συγκεντρωτικού αρχείου
                logger.info(f"Loading articles from {file_path}")
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    file_articles = json.load(f)
                    if isinstance(file_articles, list):
                        articles.extend(file_articles)
                    else:
                        articles.append(file_articles)
        
        logger.info(f"Loaded {len(articles)} articles for processing")
        
        # Επεξεργασία των άρθρων
        processed_articles = []
        for article in articles:
            try:
                processed = processor.process_article(article)
                if processed:
                    processed_articles.append(processed)
            except Exception as e:
                logger.error(f"Error processing article {article.get('title', 'Unknown')}: {str(e)}")
        
        logger.info(f"Successfully processed {len(processed_articles)} articles")
        
        # Δημιουργία backup του τρέχοντος processed_articles.json
        processed_path = data_dir / 'processed_articles.json'
        if processed_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = data_dir / f'processed_articles_backup_{timestamp}.json'
            processed_path.rename(backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Αποθήκευση των επεξεργασμένων άρθρων
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(processed_articles, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved processed articles to {processed_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in reprocess_articles: {str(e)}")
        return False

if __name__ == "__main__":
    reprocess_articles() 