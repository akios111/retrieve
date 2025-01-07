from flask import Flask, render_template, request, jsonify
import sys
import os
import json
from datetime import datetime

# Προσθήκη του parent directory στο path για να μπορούμε να κάνουμε import τα άλλα modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search.search_engine import SearchEngine

app = Flask(__name__)
search_engine = SearchEngine()

# Cache για συχνά ερωτήματα
query_cache = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/suggest', methods=['POST'])
def suggest():
    """Endpoint για autocomplete suggestions."""
    term = request.form.get('term', '').lower()
    if not term or len(term) < 2:
        return jsonify({'suggestions': []})
        
    # Εύρεση παρόμοιων όρων από το ευρετήριο
    suggestions = []
    for index_term in search_engine.index.keys():
        if index_term.lower().startswith(term):
            suggestions.append(index_term)
            if len(suggestions) >= 10:  # Περιορισμός στις top-10 προτάσεις
                break
                
    return jsonify({'suggestions': suggestions})

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    method = request.form.get('method', 'vsm')
    date_from = request.form.get('date_from')
    date_to = request.form.get('date_to')
    categories = json.loads(request.form.get('categories', '[]'))
    
    if not query:
        return jsonify({'error': 'Το ερώτημα δεν μπορεί να είναι κενό'}), 400
        
    if method not in ['boolean', 'vsm', 'bm25']:
        return jsonify({'error': 'Μη έγκυρη μέθοδος αναζήτησης'}), 400
        
    try:
        # Έλεγχος cache
        cache_key = f"{query}_{method}_{date_from}_{date_to}_{','.join(sorted(categories))}"
        if cache_key in query_cache:
            return jsonify(query_cache[cache_key])
        
        # Εφαρμογή φίλτρων
        results = search_engine.search(query, method=method)
        app.logger.info(f"Raw search results: {results}")  # Debug log
        
        filtered_results = []
        
        for title, score, snippet in results:
            try:
                article = search_engine.articles[title]
                app.logger.info(f"Processing article: {title}")  # Debug log
                
                # Φιλτράρισμα με βάση την ημερομηνία
                if date_from or date_to:
                    article_date = datetime.strptime(article['date'], '%Y-%m-%d')
                    if date_from:
                        from_date = datetime.strptime(date_from, '%Y-%m-%d')
                        if article_date < from_date:
                            continue
                    if date_to:
                        to_date = datetime.strptime(date_to, '%Y-%m-%d')
                        if article_date > to_date:
                            continue
                
                # Φιλτράρισμα με βάση τις κατηγορίες
                if categories:
                    article_categories = set(article['categories'])
                    if not any(cat in article_categories for cat in categories):
                        continue
                
                filtered_results.append({
                    'title': title,
                    'score': float(score),  # Μετατροπή σε float για σωστή σειριοποίηση
                    'snippet': snippet,
                    'categories': article.get('categories', []),
                    'date': article.get('date', '')
                })
                
            except Exception as e:
                app.logger.error(f"Error processing article {title}: {str(e)}")
                continue
        
        response = {
            'query': query,
            'method': method,
            'results': filtered_results
        }
        
        app.logger.info(f"Final response: {response}")  # Debug log
        
        # Αποθήκευση στο cache
        query_cache[cache_key] = response
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Endpoint για συλλογή feedback από τις αλληλεπιδράσεις των χρηστών."""
    try:
        title = request.form.get('title')
        query = request.form.get('query')
        dwell_time = float(request.form.get('dwell_time', 0))
        
        # Ενημέρωση των βαρών με βάση το feedback
        click_data = [(query, title, dwell_time)]
        search_engine.update_feature_weights(click_data)
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 