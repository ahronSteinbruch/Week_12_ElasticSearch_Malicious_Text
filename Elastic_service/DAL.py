from .crud import Crud
from .index_init import Index_init
from .connection import ConnES
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
