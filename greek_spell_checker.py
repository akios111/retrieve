from typing import List
import logging

class GreekSpellChecker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using simplified spell checker (no corrections)")
        
    def correct_word(self, word: str) -> str:
        """Διόρθωση ορθογραφικών λαθών (απλοποιημένη έκδοση για testing)."""
        return word
        
    def check_text(self, tokens: List[str]) -> List[str]:
        """Έλεγχος και διόρθωση κειμένου (απλοποιημένη έκδοση για testing)."""
        return tokens # Επιστρέφουμε τα tokens χωρίς διόρθωση 