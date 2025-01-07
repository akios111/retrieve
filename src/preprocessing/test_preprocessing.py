from text_processor import TextPreprocessor
import logging

# Ρύθμιση logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Δημιουργία test article
    test_article = {
        'title': 'Τεχνητή Νοημοσύνη',
        'text': '''
        Η Τεχνητή Νοημοσύνη (Τ.Ν.) ή Artificial Intelligence (A.I.) είναι κλάδος της επιστήμης υπολογιστών
        που ασχολείται με τη σχεδίαση και την υλοποίηση υπολογιστικών συστημάτων που μιμούνται στοιχεία
        της ανθρώπινης συμπεριφοράς τα οποία υπονοούν έστω και στοιχειώδη ευφυΐα: μάθηση, προσαρμοστικότητα,
        εξαγωγή συμπερασμάτων, κατανόηση από συμφραζόμενα, επίλυση προβλημάτων κ.λπ.
        
        Ο Τζον Μακάρθι (John McCarthy) όρισε την Τ.Ν. ως "την επιστήμη και μηχανική της κατασκευής ευφυών
        μηχανών". Η Τ.Ν. ερευνά μεθόδους που επιτρέπουν στους υπολογιστές να συμπεριφέρονται με τρόπο
        που θα μπορούσε να χαρακτηριστεί ευφυής εάν προερχόταν από άνθρωπο.
        
        Σήμερα, η Τ.Ν. χρησιμοποιείται σε πολλούς τομείς όπως:
        - Αναγνώριση εικόνας και ομιλίας
        - Επεξεργασία φυσικής γλώσσας
        - Ρομποτική
        - Συστήματα υποστήριξης αποφάσεων
        '''
    }
    
    # Δημιουργία του preprocessor
    preprocessor = TextPreprocessor()
    
    # Επεξεργασία του άρθρου
    processed = preprocessor.process_article(test_article)
    
    if processed:
        print("\nΕπεξεργασμένο άρθρο:")
        print(f"Τίτλος: {processed['title']}")
        print(f"\nΑριθμός tokens: {len(processed['tokens'])}")
        print(f"Μοναδικές λέξεις: {processed['metadata']['language_stats']['unique_words']}")
        print(f"Μέσο μήκος λέξης: {processed['metadata']['language_stats']['avg_word_length']:.2f}")
        
        print("\nΑναγνωρισμένες οντότητες:")
        for entity_type, entities in processed['metadata']['entities'].items():
            print(f"{entity_type}: {', '.join(entities)}")
            
        print("\nΑνάλυση συναισθήματος:")
        sentiment = processed['metadata']['sentiment']
        print(f"Συναίσθημα: {sentiment['sentiment']}")
        print(f"Βαθμός βεβαιότητας: {sentiment['score']:.2f}")
        
        print("\nΔείγμα tokens:")
        print(processed['tokens'][:20])
    else:
        print("Σφάλμα κατά την επεξεργασία του άρθρου")

if __name__ == '__main__':
    main() 