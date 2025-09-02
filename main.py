from pprint import pprint
from data_loader import DynamicDataLoader as Loader
from Elastic_service.DAL import DAL
from processor import Enriche


if __name__ == "__main__":
    loader = Loader()
    df_csv = loader.load("data/tweets_injected.csv")
    dal = DAL("tweets",create_index=True)
    dal.insert_many(df_csv)
    enricher = Enriche(index_name="tweets", weapons_file_path='./data/weapons.txt')

    # Run the weapons enrichment process
    enricher.add_weapons_to_docs() # Adjust size as needed for your dataset
    # Run the sentiment enrichment process
    enricher.add_sentiment_to_docs() # Adjust size as needed

    enricher.clean_non_antisemitic()