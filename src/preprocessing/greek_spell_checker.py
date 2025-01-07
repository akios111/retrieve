from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from typing import List

class GreekSpellChecker:
    def __init__(self, model_path: str = 'models/greek-bert'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        
    def check_word(self, word: str) -> str:
        """Check spelling of a single word."""
        try:
            # For now, just return the original word
            # TODO: Implement actual spell checking using the model
            return word
        except Exception as e:
            print(f"Error in spell checking word '{word}': {str(e)}")
            return word
        
    def check_text(self, tokens: List[str]) -> List[str]:
        """Spell check a list of tokens."""
        try:
            return [self.check_word(token) for token in tokens]
        except Exception as e:
            print(f"Error in spell checking: {str(e)}")
            return tokens 