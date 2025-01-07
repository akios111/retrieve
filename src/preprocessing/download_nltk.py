import nltk

def download_nltk_resources():
    """Κατέβασμα όλων των απαραίτητων NLTK πόρων."""
    resources = [
        'punkt',
        'stopwords',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'wordnet'
    ]
    
    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource)
        
if __name__ == "__main__":
    download_nltk_resources() 