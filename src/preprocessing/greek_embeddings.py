from transformers import AutoModel, AutoTokenizer
import torch
from typing import List

class GreekWordEmbeddings:
    def __init__(self, model_path: str = 'models/greek-bert'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        
    def get_word_embedding(self, word: str) -> torch.Tensor:
        """Get embedding for a single word."""
        with torch.no_grad():
            inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            # Use the [CLS] token embedding as word representation
            return outputs.last_hidden_state[0][0]
            
    def find_similar_words(self, word: str, n: int = 3) -> List[str]:
        """Find n most similar words to the given word."""
        # TODO: Implement actual word similarity using embeddings
        # For now, return empty list to avoid errors
        return [] 