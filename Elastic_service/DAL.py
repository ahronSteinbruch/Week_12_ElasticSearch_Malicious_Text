from .crud import Crud
from .index_init import Index_init
from .connection import ConnES
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class DAL:
    def __init__(self,index_name,create_index=False,mapping=None):
        self.index_name = index_name
        self.crud = Crud(index_name)
        if create_index:
            Index_init(index_name,mapping)
        self.es = ConnES.get_instance().connect()

    def get_all(self):
        return self.crud.search_data({"query": {"match_all": {}}})

    def insert_data(self, data):
        return self.crud.insert_data(data)

    def insert_many(self, data):
        return self.crud.insert_data_bulk(data)

    def delete_by_query(self, query: dict):
        """
        Deletes documents from Elasticsearch based on a query.

        Args:
            query (dict): The Elasticsearch query to match documents for deletion.
        """
        try:
            response = self.es.delete_by_query(
                index=self.index_name,
                body={"query": query},
                request_timeout=60
            )
            deleted = response.get('deleted', 0)
            logger.info(f"Deleted {deleted} documents matching the query.")
            # Refresh the index
            self.es.indices.refresh(index=self.index_name)
            return deleted
        except Exception as e:
            logger.error(f"Error deleting documents by query: {e}")
            return 0
    def search(self, query: dict):
        return self.crud.search_data(query)