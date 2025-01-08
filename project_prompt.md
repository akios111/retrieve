# Μηχανή Αναζήτησης για Ελληνικά Επιστημονικά Άρθρα Wikipedia

## Περιγραφή Project
Αναπτύσσουμε μια εξειδικευμένη μηχανή αναζήτησης για ελληνικά επιστημονικά άρθρα της Wikipedia. Το σύστημα υποστηρίζει τρεις μεθόδους αναζήτησης:
1. Boolean Search με τελεστές (AND, OR, NOT)
2. Vector Space Model (VSM)
3. BM25 με semantic enrichment

## Ιστορικό & Εξέλιξη
- Αρχική υλοποίηση με απλό Boolean search
- Προσθήκη VSM για καλύτερη κατάταξη αποτελεσμάτων
- Ενσωμάτωση BM25 με semantic enrichment
- Βελτιώσεις στο evaluation framework για καλύτερη αξιολόγηση
- Πρόσφατη προσθήκη semantic similarity στην αξιολόγηση relevant documents

## Προκλήσεις που Αντιμετωπίστηκαν
1. **Ελληνικό Κείμενο**
   - Δυσκολίες στο tokenization λόγω τονισμού
   - Έλλειψη καλών ελληνικών stopwords
   - Περιορισμένα εργαλεία για ελληνική γλώσσα
   - Ανάγκη για custom λύσεις σε lemmatization

2. **Αξιολόγηση Συστήματος**
   - Αρχικά χαμηλά scores στις μετρικές
   - Προβλήματα με exact matching στα relevant documents
   - Ανάγκη για semantic matching στην αξιολόγηση
   - Περιορισμένος αριθμός διαθέσιμων άρθρων

3. **Τεχνικά Ζητήματα**
   - Προβλήματα με το μέγεθος του console buffer
   - Καθυστερήσεις στο indexing μεγάλων άρθρων
   - Memory issues με μεγάλα word embeddings
   - Ανάγκη για βελτιστοποίηση της απόδοσης

## Αρχιτεκτονική Συστήματος

### 1. Crawler (`src/crawler/`)
- Συλλογή άρθρων από συγκεκριμένες κατηγορίες της ελληνικής Wikipedia
- Επικέντρωση σε επιστημονικά θέματα: Φυσική, Χημεία, Βιολογία, Μαθηματικά, κλπ.
- Αποθήκευση σε JSON format με μεταδεδομένα
- Διαχείριση rate limiting για το Wikipedia API
- Φιλτράρισμα άρθρων βάσει ποιότητας

### 2. Preprocessing (`src/preprocessing/`)
- Καθαρισμός κειμένου
- Tokenization με υποστήριξη ελληνικών
- Αφαίρεση stopwords
- Lemmatization
- Named Entity Recognition
- Word Embeddings για semantic similarity
- Διόρθωση ορθογραφικών λαθών
- Διαχείριση συνωνύμων
- Custom λύσεις για την ελληνική γλώσσα

### 3. Indexing (`src/indexing/`)
- Inverted index για γρήγορη αναζήτηση
- Document vectors για VSM
- Υπολογισμός IDF scores
- PageRank για ranking
- N-gram indexing για partial matching
- Βελτιστοποιημένη δομή για γρήγορη ανάκτηση

### 4. Search Engine (`src/search/`)
- Boolean Search με:
  * Υποστήριξη AND, OR, NOT
  * Παρενθέσεις για σύνθετα queries
  * Proximity search
  * Fuzzy matching
  * Βελτιωμένο parsing για σύνθετα queries

- Vector Space Model με:
  * TF-IDF weighting
  * Cosine similarity
  * Query expansion
  * Semantic enrichment
  * Custom weighting για τίτλους

- BM25 με:
  * Παραμετροποιημένο k1 και b
  * Proximity boosting
  * Semantic matching
  * Title field boosting
  * Fine-tuned παράμετροι

### 5. Evaluation Framework (`src/evaluation/`)
- Test queries με ground truth
- Μετρικές αξιολόγησης:
  * Precision
  * Recall
  * F1-score
  * NDCG
  * MAP
- Υποστήριξη μερικής αντιστοίχισης στα relevant documents
- Semantic similarity στην αξιολόγηση
- Detailed logging για ανάλυση αποτελεσμάτων
- Custom metrics για ελληνικό περιεχόμενο

### 6. Web Interface (`web/`)
- Flask web application
- Διεπαφή αναζήτησης
- Προβολή αποτελεσμάτων
- Highlighting των όρων αναζήτησης
- Responsive design
- Φιλτράρισμα αποτελεσμάτων

## Τρέχουσα Κατάσταση

### Υλοποιημένα Features
- Βασική λειτουργικότητα crawler ✓
- Προεπεξεργασία κειμένου ✓
- Inverted index ✓
- Τρεις μέθοδοι αναζήτησης ✓
- Evaluation framework ✓
- Web interface ✓

### Τνωστά Προβλήματα
1. **Αναζήτηση**
   - Μερικά queries επιστρέφουν μηδενικά αποτελέσματα
   - Ανάγκη για καλύτερο χειρισμό ορθογραφικών λαθών
   - Περιορισμένη semantic understanding

2. **Evaluation**
   - Χαμηλά scores σε συγκεκριμένα queries
   - Προβλήματα με το console buffer στην εκτέλεση
   - Ανάγκη για περισσότερα test cases

3. **Performance**
   - Καθυστερήσεις σε μεγάλα queries
   - Memory issues με πολλά embeddings
   - Ανάγκη για caching

### Τελευταία Αποτελέσματα Αξιολόγησης
- Boolean Queries:
  * Precision: 0.2600
  * Recall: 0.7533
  * F1-score: 0.3787
  * NDCG: 0.3783
  * MAP: 0.2674

- VSM Queries:
  * Precision: 0.4600
  * Recall: 0.8833
  * F1-score: 0.6014
  * NDCG: 0.7382
  * MAP: 0.8072

- BM25 Queries:
  * Precision: 0.4000
  * Recall: 0.4667
  * F1-score: 0.3961
  * NDCG: 0.4030
  * MAP: 0.4190

### Προκλήσεις & Επόμενα Βήματα
1. **Βελτίωση Ακρίβειας**
   - Εμπλουτισμός της βάσης άρθρων
   - Βελτίωση του semantic matching
   - Fine-tuning των παραμέτρων BM25
   - Καλύτερος χειρισμός ελληνικών συνωνύμων
   - Βελτίωση του query expansion

2. **Επέκταση Λειτουργικότητας**
   - Προσθήκη spell checking
   - Υποστήριξη φιλτραρίσματος ανά κατηγορία
   - Personalized ranking
   - Βελτιωμένο UI/UX
   - Analytics dashboard

3. **Τεχνικές Βελτιώσεις**
   - Βελτιστοποίηση απόδοσης
   - Caching μηχανισμοί
   - Παραλληλοποίηση υπολογισμών
   - Μείωση memory footprint
   - Βελτίωση logging

## Τεχνικές Λεπτομέρειες

### Dependencies
- Python 3.8+
- NLTK με ελληνικό support
- scikit-learn για ML components
- NumPy για υπολογισμούς
- PyTorch για embeddings
- Transformers για NLP
- Flask για web interface
- Custom modules για ελληνική γλώσσα

### Δομή Δεδομένων
- `data/`: JSON αρχεία με τα άρθρα
- `models/`: Προ-εκπαιδευμένα μοντέλα
- `src/`: Πηγαίος κώδικας
- `web/`: Web interface
- `evaluation/`: Test data και αποτελέσματα
- `logs/`: Detailed logging

### Εγκατάσταση & Εκτέλεση
```bash
# Εγκατάσταση dependencies
pip install -r requirements.txt

# Κατέβασμα μοντέλων
python src/preprocessing/download_models.py

# Εκτέλεση web interface
python web/app.py
```

### Αξιολόγηση
```bash
# Εκτέλεση evaluation
python -m src.evaluation.evaluation
```

## Σημαντικές Σημειώσεις
1. Το project είναι σε λειτουργική κατάσταση με την τρέχουσα έκδοση στο master branch
2. Όλες οι μετρικές αξιολόγησης δείχνουν βελτίωση μετά τις τελευταίες αλλαγές
3. Το σύστημα είναι έτοιμο για περαιτέρω βελτιώσεις και επεκτάσεις
4. Υπάρχει ανάγκη για περισσότερα ελληνικά άρθρα στη βάση
5. Τα semantic features χρειάζονται fine-tuning
6. Το evaluation framework χρειάζεται επέκταση

## Repository
- GitHub: https://github.com/akios111/retrieve.git
- Τελευταίο stable commit: 176e39c
- Branch structure:
  * master: Stable version
  * develop: Development version
  * feature/*: Νέα features
  * fix/*: Bug fixes

## Επόμενες Ενέργειες
1. Επίλυση του προβλήματος με το console buffer στο evaluation
2. Βελτίωση του semantic matching για καλύτερα αποτελέσματα
3. Εμπλουτισμός της βάσης με περισσότερα άρθρα
4. Προσθήκη caching για βελτίωση απόδοσης
5. Επέκταση του evaluation framework 

## Αξιολόγηση Συστήματος (Ερώτημα 5)

### Μεθοδολογία Αξιολόγησης
1. **Test Queries**
   - Δημιουργία συνόλου test queries για κάθε μέθοδο αναζήτησης
   - Boolean queries με τελεστές AND, OR, NOT
   - Απλά queries για VSM και BM25
   - Ορισμός relevant documents για κάθε query

2. **Μετρικές Αξιολόγησης**
   - Precision: Ακρίβεια των αποτελεσμάτων
   - Recall: Ανάκληση σχετικών εγγράφων
   - F1-score: Αρμονικός μέσος Precision-Recall
   - NDCG: Normalized Discounted Cumulative Gain
   - MAP: Mean Average Precision

3. **Βελτιώσεις στο Framework**
   - Προσθήκη semantic similarity για καλύτερη αντιστοίχιση
   - Υποστήριξη μερικής αντιστοίχισης στα relevant documents
   - Detailed logging για ανάλυση αποτελεσμάτων
   - Custom metrics για ελληνικό περιεχόμενο

### Αποτελέσματα Αξιολόγησης

1. **Boolean Search**
   ```
   Precision: 0.2600
   Recall: 0.7533
   F1-score: 0.3787
   NDCG: 0.3783
   MAP: 0.2674
   ```
   - Καλό recall αλλά χαμηλό precision
   - Ανάγκη για βελτίωση του query parsing
   - Προβλήματα με σύνθετα queries

2. **Vector Space Model**
   ```
   Precision: 0.4600
   Recall: 0.8833
   F1-score: 0.6014
   NDCG: 0.7382
   MAP: 0.8072
   ```
   - Καλύτερη απόδοση σε όλες τις μετρικές
   - Εξαιρετικό recall και καλό MAP
   - Ικανοποιητική κατάταξη αποτελεσμάτων

3. **BM25**
   ```
   Precision: 0.4000
   Recall: 0.4667
   F1-score: 0.3961
   NDCG: 0.4030
   MAP: 0.4190
   ```
   - Μέτρια απόδοση συνολικά
   - Χρειάζεται fine-tuning παραμέτρων
   - Δυνατότητα βελτίωσης με semantic enrichment

### Προβλήματα & Προκλήσεις
1. **Τεχνικά Ζητήματα**
   - Console buffer overflow σε μεγάλα αποτελέσματα
   - Καθυστερήσεις στην εκτέλεση του evaluation
   - Memory issues με word embeddings

2. **Ποιότητα Αποτελεσμάτων**
   - Μηδενικά αποτελέσματα σε κάποια queries
   - Χαμηλό precision σε Boolean αναζήτηση
   - Ανάγκη για καλύτερο semantic matching

3. **Δεδομένα**
   - Περιορισμένος αριθμός διαθέσιμων άρθρων
   - Ανάγκη για εμπλουτισμό της βάσης
   - Βελτίωση των test queries

### Προτεινόμενες Βελτιώσεις
1. **Άμεσες Βελτιώσεις**
   - Επίλυση του προβλήματος με το console buffer
   - Fine-tuning των παραμέτρων BM25
   - Εμπλουτισμός της βάσης άρθρων

2. **Μεσοπρόθεσμες Βελτιώσεις**
   - Βελτίωση του semantic matching
   - Προσθήκη περισσότερων test queries
   - Optimization του evaluation framework

3. **Μακροπρόθεσμες Βελτιώσεις**
   - Υλοποίηση caching μηχανισμών
   - Παραλληλοποίηση υπολογισμών
   - Προσθήκη νέων μετρικών αξιολόγησης

### Συμπεράσματα
1. Το VSM παρουσιάζει την καλύτερη συνολική απόδοση
2. Το Boolean search έχει καλό recall αλλά χρειάζεται βελτίωση στο precision
3. Το BM25 έχει περιθώρια βελτίωσης με fine-tuning
4. Υπάρχει ανάγκη για περισσότερα άρθρα και βελτιωμένο semantic matching
5. Τα τεχνικά ζητήματα (console buffer, memory) πρέπει να επιλυθούν άμεσα 