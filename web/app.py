from flask import Flask, render_template, request, jsonify
import sys
import os
import json
from datetime import datetime
import numpy as np

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
    try:
        query = request.form.get('query', '')
        method = request.form.get('method', 'vsm')
        date_from = request.form.get('date_from')
        date_to = request.form.get('date_to')
        categories_raw = request.form.get('categories', '[]')
        
        try:
            categories = json.loads(categories_raw)
        except json.JSONDecodeError as e:
            app.logger.error(f"Error parsing categories JSON: {str(e)}")
            categories = []
        
        if not query:
            return jsonify({'error': 'Το ερώτημα δεν μπορεί να είναι κενό'}), 400
            
        if method not in ['boolean', 'vsm', 'bm25']:
            return jsonify({'error': 'Μη έγκυρη μέθοδος αναζήτησης'}), 400
            
        # Εκτέλεση αναζήτησης με τα φίλτρα
        results = search_engine.search(
            query=query,
            method=method,
            categories=categories if categories else None,
            date_from=date_from if date_from else None,
            date_to=date_to if date_to else None
        )
        
        if not results:
            return jsonify({
                'query': query,
                'method': method,
                'results': [],
                'message': 'Δεν βρέθηκαν αποτελέσματα'
            })
            
        # Εμπλουτισμός αποτελεσμάτων
        enhanced_results = []
        for title, score in results:
            article = search_engine.articles.get(title)
            if not article:
                continue
                
            # Δημιουργία snippet
            text = article.get('text', '')
            snippet = text[:200] + '...' if len(text) > 200 else text
            
            enhanced_results.append({
                'title': title,
                'score': float(score),
                'snippet': snippet,
                'categories': article.get('categories', []),
                'date': article.get('date', '')
            })
            
        response = {
            'query': query,
            'method': method,
            'results': enhanced_results
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error in search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/article', methods=['POST'])
def get_article():
    try:
        title = request.form.get('title')
        if not title:
            return jsonify({'error': 'No title provided'}), 400
            
        article = search_engine.articles.get(title)
        if not article:
            return jsonify({'error': 'Article not found'}), 404
            
        return jsonify({
            'title': title,
            'content': article.get('text', '')
        })
        
    except Exception as e:
        app.logger.error(f"Error fetching article: {str(e)}")
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