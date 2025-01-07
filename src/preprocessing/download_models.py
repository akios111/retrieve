from transformers import AutoModel, AutoTokenizer
import torch
import os

def download_models():
    print("Κατέβασμα του ελληνικού BERT μοντέλου...")
    
    # Δημιουργία directory για τα μοντέλα αν δεν υπάρχει
    os.makedirs('models', exist_ok=True)
    
    # Κατέβασμα του μοντέλου και του tokenizer
    model_name = "nlpaueb/bert-base-greek-uncased-v1"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Αποθήκευση τοπικά
        model_path = os.path.join('models', 'greek-bert')
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print("Το μοντέλο κατέβηκε και αποθηκεύτηκε επιτυχώς!")
        
    except Exception as e:
        print(f"Σφάλμα κατά το κατέβασμα του μοντέλου: {str(e)}")

if __name__ == "__main__":
    download_models() 