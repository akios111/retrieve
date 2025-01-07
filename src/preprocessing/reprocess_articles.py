from text_processor import TextPreprocessor
import logging

def main():
    # Ρύθμιση logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Δημιουργία του preprocessor
        preprocessor = TextPreprocessor()
        
        # Φόρτωση των άρθρων
        logger.info("Φόρτωση άρθρων...")
        articles = preprocessor.load_articles()
        logger.info(f"Φορτώθηκαν {len(articles)} άρθρα.")
        
        # Επεξεργασία των άρθρων
        logger.info("Επεξεργασία άρθρων...")
        processed_articles = preprocessor.process_all_articles(articles)
        logger.info(f"Επεξεργάστηκαν {len(processed_articles)} άρθρα.")
        
        # Αποθήκευση των επεξεργασμένων άρθρων
        logger.info("Αποθήκευση επεξεργασμένων άρθρων...")
        preprocessor.save_processed_articles(processed_articles)
        logger.info("Η επεξεργασία ολοκληρώθηκε επιτυχώς!")
        
    except Exception as e:
        logger.error(f"Σφάλμα κατά την επανεπεξεργασία: {str(e)}")

if __name__ == "__main__":
    main() 