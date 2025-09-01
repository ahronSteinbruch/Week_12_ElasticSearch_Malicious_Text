from .connection import ConnES
from uuid import uuid4
from elasticsearch.helpers import bulk
from pprint import pprint
class Crud:
    def __init__(self, index_name):
        self.es = ConnES.get_instance().connect()
        self.index_name = index_name


    def insert_data(self, data):
        """
        Insert a single document into Elasticsearch.
        :param data:
        :return:
        """
        try:
            res = self.es.index(index=self.index_name, document=data)
            print("Data inserted successfully.")
            return res
        except Exception as e:
            print(f"Failed to insert data: {e}")

    def insert_data_bulk(self, data):
        """
        Insert a pandas DataFrame into Elasticsearch using bulk API.
        Each row becomes a document.
        """
        data = data.to_dict(orient='records')
        try:
            actions = \
            [
                {
                    "_index": self.index_name,
                    "_id" :str(uuid4()),
                    "_source": doc
                }
                for doc in data
            ]
            res = bulk(self.es,actions)
            pprint(f"Bulk inserted successfully. Inserted {res[0]} documents.")

        except Exception as e:
            print(f"Failed to insert data: {e}")

    def update_data(self, doc_id, data):
        """
        Update a document in Elasticsearch.
        :param doc_id:
        :param data:
        :return:
        """
        try:
            res = self.es.update(index=self.index_name, id=doc_id, body={"doc": data})
            print("Data updated successfully.")
            return res
        except Exception as e:
            print(f"Failed to update data: {e}")

    def delete_data(self, doc_id):
        """
        Delete a document from Elasticsearch.
        :param doc_id:
        :return:
        """
        try:
            res = self.es.delete(index=self.index_name, id=doc_id)
            print("Data deleted successfully.")
            return res
        except Exception as e:
            print(f"Failed to delete data: {e}")

    def search_data(self, query):
        """
        Search for documents in Elasticsearch.
        :param query:
        :return:
        """
        try:
            res = self.es.search(index=self.index_name, body=query)
            return res
        except Exception as e:
            print(f"Failed to search data: {e}")
