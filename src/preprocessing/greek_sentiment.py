from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict

class GreekSentimentAnalyzer:
    def __init__(self, model_path: str = 'models/greek-bert'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)  # positive, negative, neutral
        self.model.eval()
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
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
            
            # Επεξεργασία κάθε τμήματος και συνδυασμός των αποτελεσμάτων
            total_positive = 0.0
            total_neutral = 0.0
            total_negative = 0.0
            chunk_count = len(chunks)
            
            with torch.no_grad():
                for chunk in chunks:
                    inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    scores = torch.softmax(outputs.logits, dim=1)[0]
                    
                    total_positive += float(scores[0])
                    total_neutral += float(scores[1])
                    total_negative += float(scores[2])
            
            # Υπολογισμός μέσου όρου
            if chunk_count > 0:
                return {
                    'positive': total_positive / chunk_count,
                    'neutral': total_neutral / chunk_count,
                    'negative': total_negative / chunk_count
                }
            else:
                return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
                
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0} 