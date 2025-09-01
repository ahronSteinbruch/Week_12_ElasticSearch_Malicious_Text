from .connection import ConnES

class Index_init:
    def __init__(self,index_name,mapping=None):
        self.es = ConnES.get_instance().connect()
        self.index_name = index_name
        self.create_index()
        if mapping is None:
            self.create_mapping()





    def create_index(self):
        if not self.es.indices.exists(index=self.index_name):
            try:
                self.es.indices.create(index=self.index_name)
                print(f"Index {self.index_name} created successfully.")
            except Exception as e:
                print(f"Failed to create index {self.index_name}: {e}")

    def create_mapping(self):
        try:
            self.es.indices.put_mapping(index=self.index_name, body={
                'properties':{
                    'Tweet_id':{
                        'type': 'float'
                    },
                    'Antisemitic':{
                        'type': 'integer'
                    },
                    'CreateDate':{
                        'type': 'text'
                    },
                    'text':{
                        'type': 'text'
                    },
                    'weapons':{
                        'type': 'keyword'
                    },
                    'sentiment':{
                        'type': 'keyword'
                    }
               }
            })
        except Exception as e:
            print(f"Failed to create mapping for index {self.index_name}: {e}")

    def delete_index(self):
        try:
            self.es.indices.delete(index=self.index_name, ignore_unavailable=True)
            print(f"Index {self.index_name} deleted successfully.")
        except Exception as e:
            print(f"Failed to delete index {self.index_name}: {e}")
