import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from elasticsearch import helpers
import logging
from typing import List, Dict, Optional

# Import your actual Elasticsearch connection and DAL classes
from Elastic_service.connection import ConnES
from Elastic_service.DAL import DAL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.index_name = index_name
        self.es = ConnES.get_instance().connect()
        self.dal = DAL(index_name=index_name, create_index=True)
        self.sid = SentimentIntensityAnalyzer()
        self.weapons_list = self._load_weapons(weapons_file_path)
        self._ensure_mapping()
        logger.info(f"Enrichement class initialized for index: '{self.index_name}'")

    def _load_weapons(self, file_path: str) -> List[str]:
        """Loads weapons list from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                weapons = [line.strip().lower() for line in f if line.strip()]
                logger.info(f"Loaded {len(weapons)} weapons from {file_path}")
                return weapons
        except FileNotFoundError:
            logger.warning(f"Weapons file not found at: {file_path}. No weapons loaded.")
            return []
        except Exception as e:
            logger.error(f"Error loading weapons file {file_path}: {e}")
            return []

    def _ensure_mapping(self):
        """Ensure the index has the correct mapping for enriched fields."""
        mapping = {
            "properties": {
                "weapons_found": {"type": "keyword"},
                "sentiment_score": {"type": "float"},
                "sentiment_label": {"type": "keyword"}
            }
        }
        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name)
                logger.info(f"Created index: {self.index_name}")

            # Update mapping (safe even if fields exist)
            self.es.indices.put_mapping(index=self.index_name, body=mapping)
            logger.info(f"Mapping ensured for index '{self.index_name}': weapons_found, sentiment_score, sentiment_label")
        except Exception as e:
            logger.error(f"Failed to set mapping for index {self.index_name}: {e}")

    def _get_sentiment_score(self, text: str) -> float:
        """Calculate VADER compound sentiment score."""
        if not isinstance(text, str) or not text.strip():
            return 0.0
        return self.sid.polarity_scores(text)['compound']

    def _get_sentiment_label(self, score: float) -> str:
        """Convert compound score to label."""
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        return "neutral"

    def _extract_weapons_from_highlight(self, fragments: List[str]) -> List[str]:
        """Extract weapon names from highlighted fragments."""
        found = set()
        for fragment in fragments:
            matches = re.findall(r'<weapon>(.*?)</weapon>', fragment, re.IGNORECASE)
            for match in matches:
                match_clean = match.strip().lower()
                # Find exact match from known list
                matched_weapon = next((w for w in self.weapons_list if w == match_clean), None)
                if matched_weapon:
                    found.add(matched_weapon)
        return list(found)

    def add_weapons_to_docs(self, batch_size: int = 500):
        """Search for documents containing weapons and enrich with 'weapons_found'."""
        logger.info(f"[{self.index_name}] Starting weapons enrichment...")

        if not self.weapons_list:
            logger.warning(f"[{self.index_name}] No weapons loaded. Skipping weapons enrichment.")
            return

        query = {
            "query": {
                "bool": {
                    "should": [{"match_phrase": {"text": weapon}} for weapon in self.weapons_list],
                    "minimum_should_match": 1
                }
            },
            "_source": False,
            "highlight": {
                "fields": {
                    "text": {
                        "pre_tags": ["<weapon>"],
                        "post_tags": ["</weapon>"],
                        "fragment_size": 150,
                        "number_of_fragments": 0,
                        "no_match_size": 0
                    }
                },
                "require_field_match": True,
                "boundary_scanner": "word"
            }
        }

        try:
            docs_to_update = []
            for hit in helpers.scan(self.es, query=query, index=self.index_name, size=batch_size):
                doc_id = hit["_id"]
                if "highlight" in hit and "text" in hit["highlight"]:
                    weapons = self._extract_weapons_from_highlight(hit["highlight"]["text"])
                    if weapons:
                        docs_to_update.append({"_id": doc_id, "weapons": weapons})

            logger.info(f"[{self.index_name}] Found {len(docs_to_update)} documents with weapons.")

            if not docs_to_update:
                logger.info(f"[{self.index_name}] No documents to update with weapons.")
                return

            actions = [
                {
                    "_op_type": "update",
                    "_index": self.index_name,
                    "_id": doc["_id"],
                    "doc": {"weapons_found": doc["weapons"]}
                }
                for doc in docs_to_update
            ]

            success, failed = helpers.bulk(
                self.es, actions, chunk_size=500, request_timeout=60, raise_on_error=False
            )
            logger.info(f"[{self.index_name}] Successfully updated {success} documents with weapons.")
            if failed:
                logger.warning(f"[{self.index_name}] {len(failed)} documents failed to update (weapons).")

            # Refresh index
            self.es.indices.refresh(index=self.index_name)

        except Exception as e:
            logger.error(f"[{self.index_name}] Error during weapons enrichment: {e}")

    def add_sentiment_to_docs(self, batch_size: int = 1000):
        """Fetch all documents with 'text' and enrich with sentiment."""
        logger.info(f"[{self.index_name}] Starting sentiment enrichment...")

        query = {
            "query": {
                "bool": {
                    "must": [{"exists": {"field": "text"}}],
                    "must_not": [{"exists": {"field": "sentiment_label"}}]  # Avoid reprocessing
                }
            },
            "_source": ["text"]
        }

        try:
            docs_to_update = []
            for hit in helpers.scan(self.es, query=query, index=self.index_name, size=batch_size):
                doc_id = hit["_id"]
                text = hit.get("_source", {}).get("text")
                if text:
                    score = self._get_sentiment_score(text)
                    label = self._get_sentiment_label(score)
                    docs_to_update.append({
                        "_id": doc_id,
                        "sentiment_score": score,
                        "sentiment_label": label
                    })
                else:
                    logger.debug(f"[{self.index_name}] Doc {doc_id} has no text field.")

            logger.info(f"[{self.index_name}] Prepared {len(docs_to_update)} documents for sentiment update.")

            if not docs_to_update:
                logger.info(f"[{self.index_name}] No documents need sentiment update.")
                return

            actions = [
                {
                    "_op_type": "update",
                    "_index": self.index_name,
                    "_id": doc["_id"],
                    "doc": {
                        "sentiment_score": doc["sentiment_score"],
                        "sentiment_label": doc["sentiment_label"]
                    }
                }
                for doc in docs_to_update
            ]

            success, failed = helpers.bulk(
                self.es, actions, chunk_size=500, request_timeout=60, raise_on_error=False
            )
            logger.info(f"[{self.index_name}] Sentiment: {success} updated, {len(failed) if failed else 0} failed.")
            self.es.indices.refresh(index=self.index_name)

        except Exception as e:
            logger.error(f"[{self.index_name}] Error during sentiment enrichment: {e}")

    def test_single_doc(self, doc_id: str) -> Dict:
        """Helper: Test a single document – return its content and enrichments."""
        try:
            doc = self.es.get(index=self.index_name, id=doc_id)
            source = doc["_source"]
            logger.info(f"Document {doc_id} content: {source.get('text', '')[:200]}...")
            logger.info(f"Current enrichment: weapons_found={source.get('weapons_found')}, "
                        f"sentiment_label={source.get('sentiment_label')}, "
                        f"sentiment_score={source.get('sentiment_score')}")
            return source
        except Exception as e:
            logger.error(f"Could not retrieve document {doc_id}: {e}")
            return {}

    def verify_enrichment(self) -> Dict[str, int]:
        """Verify how many documents have enriched fields."""
        try:
            res_weapons = self.es.count(
                index=self.index_name,
                body={"query": {"exists": {"field": "weapons_found"}}}
            )["count"]

            res_sentiment = self.es.count(
                index=self.index_name,
                body={"query": {"exists": {"field": "sentiment_label"}}}
            )["count"]

            total = self.es.count(index=self.index_name)["count"]

            logger.info(f"Verification: {res_weapons}/{total} have weapons_found, "
                        f"{res_sentiment}/{total} have sentiment_label")
            return {
                "total": total,
                "weapons_found": res_weapons,
                "sentiment_label": res_sentiment
            }
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {}

    def preview_delete_by_query(self, query: dict, size: int = 10):
        """Preview documents that match the delete query."""
        try:
            res = self.es.search(
                index=self.index_name,
                body={"query": query, "_source": ["text", "Antisemitic", "sentiment_label", "weapons_found"]},
                size=size
            )
            logger.info(f"Preview of up to {size} documents that would be deleted:")
            for hit in res["hits"]["hits"]:
                logger.info(f"  ID: {hit['_id']} | Text: {hit['_source']['text'][:100]}...")
            return res["hits"]["total"]["value"]
        except Exception as e:
            logger.error(f"Error in preview: {e}")
            return 0

    def clean_non_antisemitic(self):
        """
        Deletes non-antisemitic tweets that:
        - Do NOT contain weapons (weapons_found is missing or empty)
        - Have positive or neutral sentiment (sentiment_label is 'positive' or 'neutral')

        These are considered non-harmful and not relevant for antisemitism monitoring.
        """
        logger.info(f"[{self.index_name}] Starting cleanup of non-antisemitic, non-threatening tweets...")

        query = {
            "bool": {
                "must": [
                    {"term": {"Antisemitic": 0}},
                    {"term": {"sentiment_label": "positive"}}  # Includes positive and neutral
                ],
                "must_not": [
                    {"exists": {"field": "weapons_found"}}  # No weapons found
                ]
            }
        }

        # Also include neutral sentiment
        # We'll run two queries or use a should clause
        full_query = {
            "bool": {
                "must": [
                    {"term": {"Antisemitic": 0}},
                    {
                        "bool": {
                            "should": [
                                {"term": {"sentiment_label": "positive"}},
                                {"term": {"sentiment_label": "neutral"}}
                            ],
                            "minimum_should_match": 1
                        }
                    }
                ],
                "must_not": [
                    {"exists": {"field": "weapons_found"}}
                ]
            }
        }

        deleted = self.dal.delete_by_query(full_query)
        logger.info(
            f"[{self.index_name}] Cleanup completed. Removed {deleted} non-antisemitic, non-threatening tweets.")