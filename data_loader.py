import pandas as pd
import json
import requests
import sqlalchemy


class DynamicDataLoader:
    def load(self, source, source_type=None, **kwargs):
        """
        Load data from a specified source and return a pandas DataFrame.

        :param source: Path, URL, or connection string
        :param source_type: 'csv', 'excel', 'json', 'parquet', 'mysql', 'postgres', 'sqlite', 'api'
        :param kwargs: Additional arguments (like SQL query, API params, etc.)
        """
        if not source_type:
            source_type = self._infer_type(source)

        loader_map = {
            'csv': self._load_csv,
            'excel': self._load_excel,
            'json': self._load_json,
            'parquet': self._load_parquet,
            'mysql': self._load_mysql,
            'postgres': self._load_postgres,
            'sqlite': self._load_sqlite,
            'api': self._load_api
        }

        if source_type not in loader_map:
            raise ValueError(f"Unsupported source type: {source_type}")

        return loader_map[source_type](source, **kwargs)

    def _infer_type(self, source):
        if source.endswith('.csv'):
            return 'csv'
        elif source.endswith(('.xls', '.xlsx')):
            return 'excel'
        elif source.endswith('.json'):
            return 'json'
        elif source.endswith('.parquet'):
            return 'parquet'
        elif source.startswith('http'):
            return 'api'
        else:
            raise ValueError("Unable to infer source type. Please specify source_type.")

    def _load_csv(self, path, **kwargs):
        return pd.read_csv(path, **kwargs)

    def _load_excel(self, path, **kwargs):
        return pd.read_excel(path, **kwargs)

    def _load_json(self, path, **kwargs):
        with open(path, 'r') as f:
            data = json.load(f)
        return pd.json_normalize(data)

    def _load_parquet(self, path, **kwargs):
        return pd.read_parquet(path, **kwargs)

    def _load_mysql(self, conn_str, query=None, **kwargs):
        engine = sqlalchemy.create_engine(conn_str)
        if not query:
            raise ValueError("You must provide a SQL query for MySQL")
        return pd.read_sql(query, engine)

    def _load_postgres(self, conn_str, query=None, **kwargs):
        engine = sqlalchemy.create_engine(conn_str)
        if not query:
            raise ValueError("You must provide a SQL query for PostgreSQL")
        return pd.read_sql(query, engine)

    def _load_sqlite(self, db_path, query=None, **kwargs):
        engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
        if not query:
            raise ValueError("You must provide a SQL query for SQLite")
        return pd.read_sql(query, engine)

    def _load_api(self, url, params=None, headers=None, **kwargs):
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.json_normalize(data)
