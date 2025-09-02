# fetcher.py
from Elastic_service.DAL import DAL
from typing import Dict, List

class Fetcher:
    """
    Fetcher class to get data from Elasticsearch using DAL.
    It runs queries and returns clean results.
    """

    def __init__(self, index_name: str = "tweets"):
        self.dal = DAL(index_name=index_name)

    def get_antisemitic_with_weapon(self) -> List[Dict]:
        """
        Get all tweets that are antisemitic AND have at least one weapon.
        """
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"Antisemitic": 1}},
                        {"exists": {"field": "weapons_found"}}
                    ]
                }
            },
            "size": 10000  # Adjust if needed
        }
        result = self.dal.search(query)
        return [hit["_source"] for hit in result["hits"]["hits"]]

    def get_tweets_with_two_or_more_weapons(self) -> List[Dict]:
        """
        Get all tweets that have 2 or more weapons in 'weapons_found' array.
        Uses script to check array length.
        """
        query = {
            "query": {
                "bool": {
                    "must": {
                        "script": {
                            "script": {
                                "source": "doc['weapons_found.keyword'].size() >= 2",
                                "lang": "painless"
                            }
                        }
                    }
                }
            },
            "size": 10000
        }
        result = self.dal.search(query)
        return [hit["_source"] for hit in result["hits"]["hits"]]