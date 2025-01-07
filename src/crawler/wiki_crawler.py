import wikipediaapi
import json
from tqdm import tqdm
from typing import List, Dict, Set
import logging
from pathlib import Path
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class WikipediaCrawler:
    def __init__(self, language: str = "el"):
        """
        Αρχικοποίηση του crawler
        Args:
            language: Η γλώσσα των άρθρων (προεπιλογή: ελληνικά)
        """
        self.setup_logging()
        self.setup_session()
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="WikipediaSearchEngine/1.0"
        )
        self.wiki._session = self.session
        self.collected_articles = set()

    def setup_session(self):
        """Ρύθμιση του session με retry mechanism"""
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def setup_logging(self):
        """Ρύθμιση του logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_article_content(self, title: str) -> Dict:
        """
        Ανάκτηση περιεχομένου άρθρου
        Args:
            title: Ο τίτλος του άρθρου
        Returns:
            Dictionary με τα δεδομένα του άρθρου
        """
        try:
            page = self.wiki.page(title)
            if not page.exists():
                self.logger.warning(f"Το άρθρο '{title}' δεν βρέθηκε")
                return None

            # Προσθήκη καθυστέρησης μεταξύ 1-3 δευτερολέπτων
            time.sleep(random.uniform(1, 3))

            return {
                "title": page.title,
                "text": page.text,
                "url": page.fullurl,
                "summary": page.summary,
                "categories": list(page.categories.keys())
            }
        except Exception as e:
            self.logger.error(f"Σφάλμα κατά την ανάκτηση του άρθρου '{title}': {str(e)}")
            time.sleep(5)  # Μεγαλύτερη καθυστέρηση σε περίπτωση σφάλματος
            return None

    def get_subcategories(self, category_name: str, visited: Set[str] = None) -> Set[str]:
        """
        Ανάκτηση όλων των υποκατηγοριών αναδρομικά
        Args:
            category_name: Το όνομα της κατηγορίας
            visited: Σύνολο με τις κατηγορίες που έχουν ήδη επισκεφθεί
        Returns:
            Σύνολο με όλες τις υποκατηγορίες
        """
        if visited is None:
            visited = set()

        try:
            category = self.wiki.page(f"Category:{category_name}")
            if not category.exists() or category_name in visited:
                return set()

            visited.add(category_name)
            subcats = set()
            
            for member in category.categorymembers.values():
                if "Category:" in member.title:
                    subcat_name = member.title.replace("Category:", "")
                    subcats.add(subcat_name)
                    if len(visited) < 50:  # Μειώνουμε το όριο βάθους
                        subcats.update(self.get_subcategories(subcat_name, visited))
                        time.sleep(random.uniform(1, 2))

            return subcats
        except Exception as e:
            self.logger.error(f"Σφάλμα κατά την ανάκτηση υποκατηγοριών για '{category_name}': {str(e)}")
            time.sleep(5)
            return set()

    def crawl_category(self, category_name: str, max_articles: int = 1000, recursive: bool = True) -> List[Dict]:
        """
        Συλλογή άρθρων από συγκεκριμένη κατηγορία
        Args:
            category_name: Το όνομα της κατηγορίας
            max_articles: Μέγιστος αριθμός άρθρων
            recursive: Αν θα γίνει αναδρομική συλλογή από υποκατηγορίες
        Returns:
            Λίστα με τα δεδομένα των άρθρων
        """
        articles = []
        categories_to_crawl = {category_name}
        
        if recursive:
            self.logger.info(f"Συλλογή υποκατηγοριών για την κατηγορία '{category_name}'")
            categories_to_crawl.update(self.get_subcategories(category_name))
            self.logger.info(f"Βρέθηκαν {len(categories_to_crawl)} κατηγορίες συνολικά")

        for category in tqdm(categories_to_crawl, desc="Κατηγορίες"):
            if len(articles) >= max_articles:
                break

            try:
                category_page = self.wiki.page(f"Category:{category}")
                if not category_page.exists():
                    continue

                for member in tqdm(list(category_page.categorymembers.values()), 
                                desc=f"Άρθρα κατηγορίας {category}", 
                                leave=False):
                    if len(articles) >= max_articles:
                        break

                    if member.ns == 0 and member.title not in self.collected_articles:
                        content = self.get_article_content(member.title)
                        if content:
                            articles.append(content)
                            self.collected_articles.add(member.title)
                            # Αποθήκευση κάθε 10 άρθρα
                            if len(articles) % 10 == 0:
                                self.save_to_json(articles, f"wikipedia_articles_{category_name}.json")

            except Exception as e:
                self.logger.error(f"Σφάλμα κατά την επεξεργασία της κατηγορίας '{category}': {str(e)}")
                time.sleep(5)
                continue

        return articles

    def save_to_json(self, articles: List[Dict], output_file: str):
        """
        Αποθήκευση άρθρων σε αρχείο JSON
        Args:
            articles: Λίστα με τα δεδομένα των άρθρων
            output_file: Το όνομα του αρχείου εξόδου
        """
        try:
            output_path = Path("data") / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Αποθηκεύτηκαν {len(articles)} άρθρα στο {output_file}")
        except Exception as e:
            self.logger.error(f"Σφάλμα κατά την αποθήκευση του αρχείου {output_file}: {str(e)}")

def main():
    """Παράδειγμα χρήσης του crawler"""
    crawler = WikipediaCrawler()
    
    # Λίστα με κατηγορίες για συλλογή
    categories = [
        "Επιστήμη",
        "Τεχνολογία",
        "Ιστορία",
        "Φιλοσοφία",
        "Τέχνη",
        "Μαθηματικά",
        "Φυσική",
        "Χημεία",
        "Βιολογία",
        "Πληροφορική",
        "Ιατρική",
        "Αρχιτεκτονική",
        "Μουσική",
        "Λογοτεχνία",
        "Θέατρο"
    ]
    
    for category in categories:
        try:
            print(f"\nΣυλλογή άρθρων από την κατηγορία: {category}")
            articles = crawler.crawl_category(category, max_articles=200, recursive=True)
            crawler.save_to_json(articles, f"wikipedia_articles_{category}.json")
            print(f"Συλλέχθηκαν {len(articles)} άρθρα για την κατηγορία {category}")
            time.sleep(5)  # Καθυστέρηση μεταξύ κατηγοριών
        except Exception as e:
            print(f"Σφάλμα κατά την επεξεργασία της κατηγορίας {category}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 