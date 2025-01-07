from src.evaluation.evaluation import SearchEvaluator

def main():
    print("Εκκίνηση αξιολόγησης...")
    
    # Δημιουργία του evaluator
    evaluator = SearchEvaluator()
    
    # Εκτέλεση της αξιολόγησης
    print("\nΕκτέλεση αξιολόγησης...")
    results = evaluator.evaluate_all()
    
    # Εκτύπωση αποτελεσμάτων
    print("\nΑποτελέσματα αξιολόγησης:")
    evaluator.print_evaluation_results(results)

if __name__ == "__main__":
    main() 