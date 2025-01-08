# Οδηγίες Εγκατάστασης

## 1. Δημιουργία Virtual Environment

```bash
# Δημιουργία virtual environment
python -m venv venv

# Ενεργοποίηση virtual environment
# Για Windows:
venv\Scripts\activate
# Για Linux/Mac:
source venv/bin/activate
```

## 2. Εγκατάσταση Python Packages

```bash
# Εγκατάσταση όλων των απαιτούμενων πακέτων
pip install -r requirements.txt
```

## 3. Κατέβασμα NLTK Resources

```bash
# Εκτέλεση του script για κατέβασμα NLTK resources
python src/preprocessing/download_nltk.py
```

## 4. Εγκατάσταση Greek Models

```bash
# Εγκατάσταση Greek spaCy model
python -m spacy download el_core_news_lg

# Κατέβασμα Greek BERT και άλλων μοντέλων
python src/preprocessing/download_models.py
```

## 5. Επαλήθευση Εγκατάστασης

```bash
# Έλεγχος ότι όλα τα απαραίτητα πακέτα εγκαταστάθηκαν
pip list

# Έλεγχος ότι τα NLTK resources είναι διαθέσιμα
python -c "import nltk; print('punkt' in nltk.data.path)"
python -c "import nltk; print('stopwords' in nltk.data.path)"

# Έλεγχος ότι το spaCy model είναι διαθέσιμο
python -c "import spacy; nlp = spacy.load('el_core_news_lg'); print('SpaCy model loaded successfully')"
```

## Πιθανά Προβλήματα και Λύσεις

### Windows
- Αν αντιμετωπίσετε πρόβλημα με την εγκατάσταση του torch:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- Αν χρειαστείτε το Visual C++ Build Tools:
  1. Κατεβάστε το από: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  2. Εγκαταστήστε το "Desktop development with C++"

### Linux
- Απαιτούμενα system packages:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev build-essential
  ```

### Mac
- Αν αντιμετωπίσετε προβλήματα με το torch:
  ```bash
  pip install torch torchvision torchaudio
  ```

## Επιβεβαίωση Λειτουργίας

Για να επιβεβαιώσετε ότι όλα λειτουργούν σωστά:

```bash

# Εκκίνηση του web server
python web/app.py
```

Μετά την εκκίνηση του server, επισκεφθείτε: http://localhost:5000 