from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Dict

class GreekNER:
    def __init__(self, model_path: str = 'models/greek-bert'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text."""
        try:
            # Χωρίζουμε το κείμενο σε μικρότερα τμήματα
            max_length = 512
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_tokens = self.tokenizer.tokenize(word)
                if current_length + len(word_tokens) > max_length:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word_tokens)
                else:
                    current_chunk.append(word)
                    current_length += len(word_tokens)
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Επεξεργασία κάθε τμήματος
            all_entities = []
            with torch.no_grad():
                for chunk in chunks:
                    inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    # TODO: Implement proper NER logic
                    # For now, return empty list
                
            return all_entities
            
        except Exception as e:
            print(f"Error in NER processing: {str(e)}")
            return [] 