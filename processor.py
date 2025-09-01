import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from elasticsearch import helpers

# Import your actual Elasticsearch connection and DAL classes
from Elastic_service.connection import ConnES
from Elastic_service.DAL import DAL

# Download vader_lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


class Enriche:
    """
    Enrichment class for processing and updating documents in Elasticsearch.
    It identifies weapons in text and calculates sentiment, then updates
    the documents with this new information.
    """
    def __init__(self, index_name="tweets", weapons_file_path='./data/weapons.txt'):
        """
        Initializes the Enrichement class.

        Args:
            index_name (str): The name of the Elasticsearch index to work with.
            weapons_file_path (str): Path to the file containing the list of weapons.
        """
        self.es = ConnES.get_instance().connect()
        self.dal = DAL(index_name=index_name, create_index=True)
        self.index_name = index_name
        self.sid = SentimentIntensityAnalyzer()
        self.weapons_list = self._load_weapons(weapons_file_path)
        print(f"Enrichement class initialized for index: '{self.index_name}'")

    def _load_weapons(self, file_path):
        """
        Loads the weapons list from a specified file.
        Each weapon should be on a new line in the file.

        Args:
            file_path (str): The path to the weapons list file.

        Returns:
            list: A list of weapons (strings), lowercased and stripped.
        """
        try:
            with open(file_path, 'r') as f:
                # Read the file line by line, strip whitespace, convert to lowercase
                # and filter out empty lines.
                return [line.strip().lower() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: Weapons file not found at: {file_path}. No weapons loaded.")
            return []
        except Exception as e:
            print(f"Error loading weapons file {file_path}: {e}")
            return []

    def _get_sentiment_score(self, text):
        """
        Calculates the VADER compound sentiment score for a given text.

        Args:
            text (str): The input text for sentiment analysis.

        Returns:
            float: The compound sentiment score. Returns 0.0 for non-string input.
        """
        if not isinstance(text, str):
            return 0.0 # Neutral score for invalid input
        return self.sid.polarity_scores(text)['compound']

    def _get_sentiment_label(self, compound_score):
        """
        Maps a VADER compound sentiment score to a descriptive label (positive, negative, neutral).

        Args:
            compound_score (float): The compound sentiment score.

        Returns:
            str: "positive", "negative", or "neutral".
        """
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"

    def add_weapons_to_docs(self, size=7000):
        """
        Searches for documents containing weapons (from the loaded weapons_list),
        identifies the specific weapons using Elasticsearch highlighting,
        and updates the documents in Elasticsearch with a new 'weapons_found' field.

        Args:
            size (int): The maximum number of documents to retrieve in one search request.
                        Adjust based on your dataset size and Elasticsearch configuration.
        """
        print(f"[{self.index_name}] Starting weapons enrichment...")

        if not self.weapons_list:
            print(f"[{self.index_name}] No weapons loaded. Skipping weapons enrichment.")
            return

        # Build query to match any weapon using match_phrase for accuracy
        query = {
            "query": {
                "bool": {
                    "should": [{"match_phrase": {"text": w}} for w in self.weapons_list],
                    "minimum_should_match": 1
                }
            },
            "_source": False,  # We only need highlights, not the full source document
            "highlight": {
                "fields": {
                    "text": {
                        "pre_tags": ["<weapon>"],  # Custom HTML-like tags for easy parsing
                        "post_tags": ["</weapon>"],
                        "fragment_size": 100,      # Size of the text fragment to return
                        "number_of_fragments": 0,  # 0 means return the full field if multiple matches
                        "no_match_size": 0         # Don't return fragments if no match
                    }
                },
                "require_field_match": True, # Only highlight if 'text' field matched
                "boundary_scanner": "word"   # Ensures highlighting respects word boundaries
            }
        }

        print(f"[{self.index_name}] Searching for documents containing weapons...")
        try:
            res = self.es.search(index=self.index_name, body=query, size=size)
        except Exception as e:
            print(f"[{self.index_name}] Error during weapons search: {e}")
            return

        docs_weapons = []
        for hit in res["hits"]["hits"]:
            doc_id = hit["_id"]
            found_weapons_in_doc = set() # Use a set to automatically handle duplicates

            if "highlight" in hit and "text" in hit["highlight"]:
                for fragment in hit["highlight"]["text"]:
                    # Extract highlighted parts using regex
                    matches = re.findall(r'<weapon>(.*?)</weapon>', fragment, re.IGNORECASE)
                    for match in matches:
                        # Normalize the matched weapon to one of your known weapons
                        # This handles potential casing differences or plural forms if your list is comprehensive
                        for known_weapon in self.weapons_list:
                            if known_weapon.lower() == match.lower(): # Case-insensitive comparison
                                found_weapons_in_doc.add(known_weapon)
                                break # Found a match, move to the next 'match' from regex

            if found_weapons_in_doc:
                docs_weapons.append({"_id": doc_id, "weapons": list(found_weapons_in_doc)})

        print(f"[{self.index_name}] Found {len(docs_weapons)} documents with weapons to update.")

        if not docs_weapons:
            print(f"[{self.index_name}] No documents with weapons found, skipping update.")
            return

        print(f"[{self.index_name}] Starting bulk update for 'weapons_found' field...")
        actions = []
        for doc in docs_weapons:
            actions.append({
                "_op_type": "update",
                "_index": self.index_name,
                "_id": doc["_id"],
                "doc": {
                    "weapons_found": doc["weapons"] # Add a new field 'weapons_found' with the list
                }
            })

        try:
            # Use helpers.bulk for efficient updates
            success_count, failed_count = helpers.bulk(
                self.es, actions, chunk_size=500, request_timeout=60, raise_on_error=False
            )
            print(f"[{self.index_name}] Successfully updated {success_count} documents with 'weapons_found'.")
            if failed_count:
                print(f"[{self.index_name}] {len(failed_count)} documents failed to update (details in logs if raise_on_error was True).")
        except Exception as e:
            print(f"[{self.index_name}] An unexpected error occurred during bulk update for weapons: {e}")

        print(f"[{self.index_name}] Weapons enrichment completed.")

    def add_sentiment_to_docs(self, size=7000):
        """
        Fetches documents that have a 'text' field, calculates their sentiment
        (score and label), and updates them in Elasticsearch with 'sentiment_score'
        and 'sentiment_label' fields.

        Args:
            size (int): The maximum number of documents to retrieve in one search request.
        """
        print(f"[{self.index_name}] Starting sentiment enrichment...")

        # Query to retrieve documents that have a 'text' field.
        # In a production scenario, you might want to specifically query for documents
        # that *do not* yet have 'sentiment_label' to avoid re-processing.
        query = {
            "query": {
                "exists": {
                    "field": "text"
                }
            },
            "_source": ["text"] # We need the actual text content for analysis
        }

        print(f"[{self.index_name}] Fetching documents for sentiment analysis...")
        try:
            res = self.es.search(index=self.index_name, body=query, size=size)
        except Exception as e:
            print(f"[{self.index_name}] Error during sentiment data fetch: {e}")
            return

        docs_to_update = []
        for hit in res["hits"]["hits"]:
            doc_id = hit["_id"]
            text = hit["_source"].get("text") # Safely retrieve 'text' field

            if text:
                compound_score = self._get_sentiment_score(text)
                sentiment_label = self._get_sentiment_label(compound_score)
                docs_to_update.append({
                    "_id": doc_id,
                    "sentiment_score": compound_score,
                    "sentiment_label": sentiment_label
                })
            else:
                print(f"[{self.index_name}] Warning: Document {doc_id} has no 'text' field for sentiment analysis.")


        print(f"[{self.index_name}] Prepared {len(docs_to_update)} documents for sentiment update.")

        if not docs_to_update:
            print(f"[{self.index_name}] No documents to analyze for sentiment, skipping update.")
            return

        print(f"[{self.index_name}] Starting bulk update for sentiment fields...")
        actions = []
        for doc_data in docs_to_update:
            doc_id = doc_data.pop("_id") # Remove _id from the dict as it's for the bulk operation, not the doc content
            actions.append({
                "_op_type": "update",
                "_index": self.index_name,
                "_id": doc_id,
                "doc": doc_data # This dict now contains 'sentiment_score' and 'sentiment_label'
            })

        try:
            success_count, failed_count = helpers.bulk(
                self.es, actions, chunk_size=500, request_timeout=60, raise_on_error=False
            )
            print(f"[{self.index_name}] Successfully updated {success_count} documents with sentiment.")
            if failed_count:
                print(f"[{self.index_name}] {len(failed_count)} documents failed to update (details in logs if raise_on_error was True).")
        except Exception as e:
            print(f"[{self.index_name}] An unexpected error occurred during bulk update for sentiment: {e}")

        print(f"[{self.index_name}] Sentiment enrichment completed.")
