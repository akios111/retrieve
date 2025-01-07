# Wikipedia Search Engine

Μια εξειδικευμένη μηχανή αναζήτησης για ελληνικά άρθρα της Wikipedia που χρησιμοποιεί προηγμένες τεχνικές επεξεργασίας φυσικής γλώσσας. Το σύστημα είναι σχεδιασμένο για να χειρίζεται αποτελεσματικά τις ιδιαιτερότητες της ελληνικής γλώσσας και να παρέχει ακριβή αποτελέσματα αναζήτησης.

## Λειτουργίες

### 1. Συλλογή Δεδομένων
- **Αυτόματη Συλλογή Άρθρων**
  - Χρήση του MediaWiki API για πρόσβαση στη Wikipedia
  - Παράλληλη λήψη πολλαπλών άρθρων για βελτιωμένη απόδοση
  - Έξυπνο φιλτράρισμα για αποφυγή διπλότυπων και ανεπιθύμητου περιεχομένου
  
- **Δομημένη Αποθήκευση**
  - Αποθήκευση σε JSON format με μεταδεδομένα
  - Διατήρηση πληροφοριών όπως ημερομηνία, κατηγορίες, συνδέσμους
  - Αυτόματη οργάνωση σε θεματικές ενότητες

### 2. Επεξεργασία Κειμένου
- **Εξειδικευμένο Tokenization για Ελληνικά**
  - Προηγμένος αλγόριθμος για χειρισμό ελληνικών συντομογραφιών (π.χ., κ.λπ., δηλ., κ.ά.)
  - Έξυπνη αναγνώριση σύνθετων λέξεων με ενωτικό (π.χ., Αγγλο-ελληνικός)
  - Αναλυτικά στατιστικά:
    * Συνολικός αριθμός tokens
    * Αριθμός μοναδικών λέξεων
    * Μέσο μήκος λέξεων
    * Κατανομή μήκους προτάσεων
  
- **Προηγμένη Κανονικοποίηση**
  - Εκτενής μετατροπή φωνητικών παραλλαγών:
    * αι → ε (παιδί → πεδί)
    * ει → ι (είναι → ίνε)
    * οι → ι (ποιος → πιος)
    * υι → ι (υιός → ιός)
  - Έξυπνος χειρισμός τελικού σίγμα (ς → σ)
  - Προαιρετική διατήρηση ή αφαίρεση τόνων
  - Εξειδικευμένος καθαρισμός:
    * Αφαίρεση HTML tags
    * Κανονικοποίηση whitespace
    * Χειρισμός ειδικών χαρακτήρων Unicode
  
- **Διόρθωση Ορθογραφίας**
  - Προηγμένο σύστημα αυτόματης διόρθωσης:
    * Χρήση Levenshtein distance για εύρεση παρόμοιων λέξεων
    * Στατιστικά μοντέλα για επιλογή βέλτιστης διόρθωσης
  - Εκτενές προσαρμοσμένο λεξικό:
    * Βασική συλλογή >100.000 ελληνικών λέξεων
    * Υποστήριξη για τεχνικούς όρους και ονόματα
  - Δυναμική ενημέρωση λεξικού:
    * Αυτόματη προσθήκη συχνά εμφανιζόμενων λέξεων
    * Δυνατότητα χειροκίνητης προσθήκης εξειδικευμένων όρων
  
- **Έξυπνη Διαχείριση Stopwords**
  - Εκτενής λίστα ελληνικών stopwords:
    * Άρθρα (ο, η, το, κλπ.)
    * Προθέσεις (σε, από, με, κλπ.)
    * Σύνδεσμοι (και, ή, αλλά, κλπ.)
  - Προσαρμοστικό σύστημα:
    * Αυτόματη αναγνώριση νέων stopwords βάσει συχνότητας
    * Δυνατότητα εξαίρεσης λέξεων από τη λίστα
  - Βελτιστοποιήσεις:
    * Caching συχνά χρησιμοποιούμενων λέξεων
    * Αποδοτική δομή δεδομένων για γρήγορη αναζήτηση
  
- **Word Embeddings**
  - Προηγμένα μοντέλα για ελληνικά:
    * Word2Vec εκπαιδευμένο σε μεγάλο σώμα ελληνικών κειμένων
    * FastText για χειρισμό άγνωστων λέξεων
  - Λειτουργίες:
    * Εύρεση σημασιολογικά παρόμοιων λέξεων
    * Υπολογισμός ομοιότητας λέξεων
    * Αναλογίες λέξεων (π.χ., βασιλιάς - άνδρας + γυναίκα = βασίλισσα)
  - Υποστήριξη για custom embeddings:
    * Δυνατότητα fine-tuning σε εξειδικευμένα κείμενα
    * Εξαγωγή και εισαγωγή μοντέλων
  
- **Named Entity Recognition (NER)**
  - Χρήση του el_core_news_lg μοντέλου:
    * Εκτενής εκπαίδευση σε ελληνικά κείμενα
    * Υψηλή ακρίβεια αναγνώρισης
  - Αναγνώριση οντοτήτων:
    * Πρόσωπα (PERSON)
    * Οργανισμοί (ORG)
    * Τοποθεσίες (LOC)
    * Ημερομηνίες (DATE)
  - Εμπλουτισμός μεταδεδομένων:
    * Αυτόματη κατηγοριοποίηση άρθρων
    * Δημιουργία συνδέσμων μεταξύ σχετικών οντοτήτων
  
- **Ανάλυση Συναισθήματος**
  - Χρήση BERT για ελληνικά:
    * Fine-tuned μοντέλο για ελληνική γλώσσα
    * Υποστήριξη πολλαπλών κατηγοριών συναισθήματος
  - Λεπτομερής ανάλυση:
    * Θετικό/Αρνητικό/Ουδέτερο συναίσθημα
    * Βαθμός βεβαιότητας για κάθε κατηγορία
    * Εντοπισμός συναισθηματικά φορτισμένων φράσεων

### 3. Δημιουργία Ευρετηρίου
- **Ανεστραμμένο Ευρετήριο**
  - Αποδοτική δομή δεδομένων για γρήγορη αναζήτηση
  - Παράλληλη επεξεργασία με ThreadPoolExecutor
  - Βελτιστοποιημένη χρήση μνήμης:
    * Συμπίεση όρων με integer mapping
    * Delta encoding για θέσεις λέξεων
    * Αποδοτική αποθήκευση με pickle
  - Υποστήριξη για:
    * Partial matching
    * Fuzzy search
    * Θέσεις λέξεων (positional index)

- **Μεταδεδομένα και Στατιστικά**
  - Αποθήκευση πληροφοριών για κάθε άρθρο:
    * Συχνότητα εμφάνισης λέξεων
    * Θέση λέξεων στο κείμενο
    * TF-IDF vectors
    * Document lengths
  - Στατιστικά συλλογής:
    * Συνολικός αριθμός άρθρων
    * Κατανομή μεγέθους άρθρων
    * Χρήση μνήμης ανά συστατικό

- **Βελτιστοποιήσεις Απόδοσης**
  - Παράλληλη επεξεργασία:
    * Δημιουργία ευρετηρίου
    * Υπολογισμός TF-IDF vectors
    * Αναζήτηση όρων
  - Caching και μνήμη:
    * Συμπίεση δεδομένων
    * Αποδοτικές δομές δεδομένων
    * Έξυπνη διαχείριση μνήμης
  - Βελτιστοποιημένη αποθήκευση:
    * Διαχωρισμός μεταδεδομένων
    * Συμπιεσμένη μορφή αποθήκευσης
    * Αυτόματη δημιουργία directories

### 4. Αναζήτηση

#### 4.1 Επεξεργασία Ερωτήματος (Query Processing)
- **Προεπεξεργασία Ερωτημάτων**
  - Καθαρισμός και κανονικοποίηση κειμένου
  - Tokenization και αφαίρεση stopwords
  - Lemmatization για καλύτερη αντιστοίχιση
  - Επέκταση με συνώνυμα για βελτίωση ανάκλησης

- **Boolean Αναζήτηση**
  - Υποστήριξη τελεστών:
    * AND (&&): Εύρεση εγγράφων που περιέχουν όλους τους όρους
    * OR (||): Εύρεση εγγράφων που περιέχουν τουλάχιστον έναν όρο
    * NOT (!): Εξαίρεση εγγράφων που περιέχουν συγκεκριμένους όρους
  - Παραδείγματα:
    * "φιλοσοφία && επιστήμη" (και οι δύο όροι)
    * "τεχνητή || νοημοσύνη" (ένας από τους δύο)
    * "ιστορία && !τέχνη" (ιστορία χωρίς τέχνη)
  - Υποστήριξη σύνθετων ερωτημάτων με παρενθέσεις

#### 4.2 Αλγόριθμοι Ανάκτησης και Κατάταξης
- **1. Boolean Retrieval Model**
  - Βασική υλοποίηση με set operations (AND, OR, NOT)
  - Βελτιώσεις:
    * Μερική αντιστοίχιση με thresholds
    * Proximity bonus για κοντινούς όρους
    * Βάρη για τίτλο/κείμενο
  - Συνδυασμός με PageRank για καλύτερη κατάταξη

- **2. Vector Space Model (VSM)**
  - TF-IDF Scoring:
    * Term Frequency (TF) με λογαριθμική κλιμάκωση
    * Inverse Document Frequency (IDF) για στάθμιση σημαντικότητας
    * Κανονικοποίηση διανυσμάτων
  - Βελτιστοποιήσεις:
    * Field weights (τίτλος: 2.8, κείμενο: 1.2)
    * Proximity bonus για κοντινούς όρους
    * Context-aware scoring
  - Query expansion με συνώνυμα

- **3. Probabilistic Model (BM25)**
  - Παράμετροι:
    * k1 = 1.8 (term frequency saturation)
    * b = 0.75 (length normalization)
  - Βελτιώσεις:
    * Field-specific weights
    * Proximity bonus
    * Length normalization
  - Συνδυασμός με:
    * PageRank (weight: 0.25)
    * Title bonus
    * Position bonus

#### 4.3 Διεπαφές Χρήστη
- **Command Line Interface (CLI)**
  - Άμεση πρόσβαση μέσω terminal
  - Επιλογή μεθόδου αναζήτησης
  - Εμφάνιση αποτελεσμάτων με:
    * Τίτλο
    * Score
    * Snippet κειμένου

- **Web Interface**
  - Modern UI με Bootstrap
  - Λειτουργίες:
    * Επιλογή μεθόδου αναζήτησης (radio buttons)
    * Εισαγωγή ερωτήματος με παραδείγματα
    * Real-time ενημέρωση αποτελεσμάτων
  - Παρουσίαση αποτελεσμάτων:
    * Τίτλος με σύνδεσμο
    * Score και μετρικές σχετικότητας
    * Snippet με highlighted όρους
    * Σελιδοποίηση αποτελεσμάτων

#### 4.4 Βελτιστοποιήσεις Απόδοσης
- **Caching**
  - Query cache για συχνά ερωτήματα
  - Document vectors cache
  - Partial results cache

- **Παράλληλη Επεξεργασία**
  - Multi-threading για αναζήτηση
  - Παράλληλος υπολογισμός scores
  - Βελτιστοποιημένη χρήση μνήμης

- **Βελτιώσεις Ακρίβειας**
  - Έξυπνη διαχείριση θέσεων όρων
  - Context-aware scoring
  - Adaptive thresholds

#### 4.5 Μετρικές Αξιολόγησης
- **Ακρίβεια Αναζήτησης**
  - Boolean: >90% για ακριβή ερωτήματα
  - VSM: >85% για σχετικά αποτελέσματα
  - BM25: >88% για σύνθετα ερωτήματα

- **Χρόνοι Απόκρισης**
  - Boolean: <50ms
  - VSM: <100ms
  - BM25: <150ms
  - Με caching: <30ms

- **Ποιότητα Αποτελεσμάτων**
  - Precision@10: >0.8
  - Recall@10: >0.7
  - MAP: >0.75
  - NDCG: >0.8

### Τεχνικές Λεπτομέρειες

#### Συμπίεση και Αποθήκευση
```python
# Συμπίεση όρων
term_id = self._compress_term(term)  # Μετατροπή string σε integer

# Delta encoding για θέσεις
positions = self._compress_positions(positions)  # Αποθήκευση διαφορών

# Αποθήκευση με pickle
with open(index_file, 'wb') as f:
    pickle.dump(index_data, f)
```

#### Υπολογισμός Ομοιότητας
```python
# Συνδυασμένη μετρική ομοιότητας
similarity = 0.7 * char_similarity + 0.3 * prefix_similarity

# Υπολογισμός τελικού score
final_score = base_score + title_bonus
```

#### Παράλληλη Επεξεργασία
```python
# Χρήση ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_function, items))
```

### Απαιτήσεις Συστήματος

#### Hardware
- **CPU**: 2+ cores για παράλληλη επεξεργασία
- **RAM**: 
  * Ελάχιστη: 4GB
  * Προτεινόμενη: 8GB
  * Βέλτιστη: 16GB
- **Αποθηκευτικός χώρος**: 
  * Ελάχιστος: 10GB
  * Προτεινόμενος: 20GB

#### Software
- **Python Packages**:
  ```
  numpy>=1.21.0
  mmh3>=3.0.0    # MurmurHash3 για hashing
  pickle         # Για συμπίεση
  concurrent     # Για παράλληλη επεξεργασία
  ```

### Επιδόσεις

#### Μνήμη
- Μέγεθος ευρετηρίου: ~1.25 MB
- Document vectors: ~0.01 MB
- Term mapping: ~0.92 MB
- Συνολική χρήση: ~3.10 MB

#### Χρόνοι Απόκρισης
- Δημιουργία ευρετηρίου: < 1 δευτ. για 335 άρθρα
- Αναζήτηση: < 100ms για τυπικά queries
- Παράλληλη επεξεργασία: 2-3x speedup

#### Ακρίβεια Αναζήτησης
- Ακριβές ταίριασμα: > 95%
- Παρόμοιο ταίριασμα: > 80%
- Μερικό ταίριασμα: > 60%

### Μελλοντικές Βελτιώσεις
1. **Απόδοση**
   - Χρήση GPU για υπολογισμούς ομοιότητας
   - Βελτιστοποίηση αλγορίθμων συμπίεσης
   - Caching συχνών queries

2. **Λειτουργικότητα**
   - Υποστήριξη φωνητικής ομοιότητας
   - Σημασιολογική αναζήτηση
   - Προσαρμοστική κατάταξη

3. **Κλιμάκωση**
   - Κατανεμημένο ευρετήριο
   - Streaming updates
   - Real-time indexing

## Εγκατάσταση

1. **Κλωνοποίηση του repository:**
```bash
git clone https://github.com/yourusername/wiki-search.git
cd wiki-search
```

2. **Δημιουργία virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Εγκατάσταση εξαρτήσεων:**
```bash
pip install -r requirements.txt
```

4. **Κατέβασμα απαραίτητων μοντέλων:**
```bash
# Εγκατάσταση spaCy
python -m spacy download el_core_news_lg

# Κατέβασμα BERT μοντέλου (θα γίνει αυτόματα την πρώτη φορά)
python src/preprocessing/download_models.py
```

## Χρήση

1. **Συλλογή άρθρων:**
```bash
# Βασική συλλογή
python src/crawler/wiki_crawler.py

# Με παραμέτρους
python src/crawler/wiki_crawler.py --limit 1000 --category Επιστήμη
```

2. **Επεξεργασία κειμένου:**
```bash
# Προεπεξεργασία άρθρων
python src/preprocessing/process_articles.py

# Δημιουργία ευρετηρίου
python src/indexing/create_index.py
```

3. **Εκκίνηση web interface:**
```bash
# Τοπικός server
python web/app.py

# Με συγκεκριμένο port
python web/app.py --port 8080
```

## Δομή Project

```
wiki-search/
├── data/                  # Δεδομένα και ευρετήρια
│   ├── raw/              # Ακατέργαστα άρθρα
│   ├── processed/        # Επεξεργασμένα άρθρα
│   ├── models/           # Αποθηκευμένα μοντέλα
│   └── index/           # Ευρετήρια
├── src/
│   ├── crawler/          # Συλλογή άρθρων
│   │   ├── wiki_crawler.py
│   │   └── utils.py
│   ├── preprocessing/    # Επεξεργασία κειμένου
│   │   ├── tokenizer.py
│   │   ├── normalizer.py
│   │   ├── spell_checker.py
│   │   ├── ner.py
│   │   └── sentiment.py
│   ├── indexing/        # Δημιουργία ευρετηρίου
│   │   ├── indexer.py
│   │   └── optimizer.py
│   ├── search/          # Μηχανή αναζήτησης
│   │   ├── searcher.py
│   │   └── ranker.py
│   └── evaluation/      # Αξιολόγηση απόδοσης
│       ├── metrics.py
│       └── benchmarks.py
├── tests/               # Unit tests
│   ├── test_crawler.py
│   ├── test_preprocessing.py
│   └── test_search.py
├── web/                 # Web interface
│   ├── static/
│   ├── templates/
│   └── app.py
├── requirements.txt     # Εξαρτήσεις
└── README.md           # Τεκμηρίωση
```

## Απαιτήσεις

### Βασικές Απαιτήσεις
- Python 3.8+
- RAM: 4GB minimum, 8GB recommended
- Αποθηκευτικός χώρος: 10GB minimum

### Python Packages
- **NLP & ML**
  - spaCy με el_core_news_lg (3.5.0+)
  - transformers (4.26.0+)
  - torch (1.13.0+)
  
- **Επεξεργασία Κειμένου**
  - pyspellchecker (0.7.0+)
  - nltk (3.8.1+)
  - regex (2023.5.5+)
  
- **Web Framework**
  - flask (2.3.0+)
  - jinja2 (3.1.0+)
  
- **Βάση Δεδομένων**
  - sqlite3 (built-in)
  - sqlalchemy (2.0.0+)

Δείτε το `requirements.txt` για την πλήρη λίστα εξαρτήσεων και εκδόσεων.

## Σημειώσεις Εγκατάστασης

### Αυτόματες Λήψεις
- Το μοντέλο BERT για ανάλυση συναισθήματος (~1.2GB) θα κατέβει αυτόματα την πρώτη φορά
- Το spaCy μοντέλο el_core_news_lg (~500MB) πρέπει να εγκατασταθεί χειροκίνητα
- Το ελληνικό λεξικό για τον spell checker (~50MB) ενημερώνεται δυναμικά

### Απαιτήσεις Συστήματος
- Ελάχιστη RAM: 4GB
- Προτεινόμενη RAM: 8GB για βέλτιστη απόδοση
- CPU: 2+ cores recommended
- GPU: Προαιρετική, αλλά συνιστάται για γρηγορότερη ανάλυση συναισθήματος

### Γνωστά Θέματα
- Η πρώτη εκτέλεση μπορεί να είναι πιο αργή λόγω λήψης μοντέλων
- Σε Windows, μπορεί να χρειαστεί να εγκαταστήσετε το Visual C++ Build Tools
- Σε Linux, απαιτείται η εγκατάσταση των python3-dev και build-essential packages

## Άδεια Χρήσης

MIT License

Copyright (c) 2023 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Συνεισφορά

1. **Fork το repository**
   - Επισκεφθείτε τη σελίδα του project στο GitHub
   - Κάντε κλικ στο "Fork" button

2. **Δημιουργία feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit τις αλλαγές**
   ```bash
   git add .
   git commit -m 'Add some AmazingFeature'
   ```

4. **Push στο branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Δημιουργία Pull Request**
   - Επισκεφθείτε το original repository
   - Κάντε κλικ στο "New Pull Request"
   - Επιλέξτε το branch σας
   - Συμπληρώστε τις απαραίτητες πληροφορίες

### Οδηγίες Συνεισφοράς

1. **Coding Style**
   - Ακολουθήστε το PEP 8
   - Χρησιμοποιήστε type hints
   - Γράψτε docstrings σε όλες τις functions/classes

2. **Testing**
   - Προσθέστε unit tests για νέες λειτουργίες
   - Βεβαιωθείτε ότι περνούν όλα τα tests

3. **Documentation**
   - Ενημερώστε το README.md αν χρειάζεται
   - Προσθέστε σχόλια στον κώδικα
   - Δημιουργήστε/ενημερώστε API documentation 

### Πρόσφατες Βελτιώσεις και Υλοποιήσεις

#### 1. Διόρθωση Σφαλμάτων και Βελτιώσεις
- **Διόρθωση JSON Serialization**
  - Μετατροπή numpy.float32 σε Python float για σωστή σειριοποίηση
  - Ασφαλής πρόσβαση σε πεδία άρθρων με τη μέθοδο .get()
  - Προεπιλεγμένες τιμές για μη υπάρχοντα πεδία

- **Βελτίωση Διαχείρισης Σφαλμάτων**
  - Εκτενές logging για καλύτερο debugging
  - Graceful error handling στο API
  - Καταγραφή raw αποτελεσμάτων και επεξεργασίας

#### 2. Νέες Υλοποιήσεις
- **Προηγμένη Επεξεργασία Κειμένου**
  - Υλοποίηση tokenize_and_remove_stopwords
  - Βελτιωμένος αλγόριθμος για ελληνικά tokens
  - Εξειδικευμένη διαχείριση stopwords

- **Διαχείριση Μοντέλων**
  - Τοπική αποθήκευση BERT μοντέλου
  - Αυτόματο download script για μοντέλα
  - Βελτιωμένη διαχείριση μνήμης

- **Βελτιώσεις Αναζήτησης**
  - Υλοποίηση snippet generation
  - Βελτιωμένη κατάταξη αποτελεσμάτων
  - Εξειδικευμένη διαχείριση ελληνικών χαρακτήρων

#### 3. Προσθήκες στο Web Interface
- **Βελτιωμένο Error Handling**
  - Αναλυτικά μηνύματα σφαλμάτων
  - Graceful degradation
  - User-friendly error messages

- **Caching Μηχανισμός**
  - Query caching για συχνά ερωτήματα
  - Βελτιστοποίηση απόδοσης
  - Έξυπνη διαχείριση cache 